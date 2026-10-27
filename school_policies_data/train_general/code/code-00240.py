    # Annotate type here to support type checking
    chunk_base_stride: IntLikeType = old_stride[-1]
    tensor_numel: IntLikeType = 1
    view_numel: IntLikeType = 1

    for tensor_d in range(len(old_shape) - 1, -1, -1):
        tensor_numel *= old_shape[tensor_d]

        if tensor_d == 0 or (
            maybe_guard_or_true(old_shape[tensor_d - 1] != 1)
            and maybe_guard_or_true(
                old_stride[tensor_d - 1] != tensor_numel * chunk_base_stride
            )
        ):
            while view_d >= 0 and (
                maybe_guard_or_true(view_numel < tensor_numel)
                or maybe_guard_or_false(new_shape[view_d] == 1)
            ):
                new_stride[view_d] = view_numel * chunk_base_stride
                view_numel *= new_shape[view_d]
                view_d -= 1

            if maybe_guard_or_true(view_numel != tensor_numel):
                return None

            if tensor_d > 0:
                chunk_base_stride = old_stride[tensor_d - 1]
                tensor_numel = 1
                view_numel = 1
    if view_d != -1:
        return None
    return new_stride


def _optimization_hint_or_none(value: Any) -> int | None:
    if type(value) is int:
        return value
    if isinstance(value, torch.SymInt):
        from torch.fx.experimental.symbolic_shapes import optimization_hint

        return optimization_hint(value, fallback=None)
    return None


def _hinted_eq(lhs: Any, rhs: Any) -> bool:
    from torch.fx.experimental.symbolic_shapes import guard_or_false

    if guard_or_false(lhs == rhs):
        return True

    lhs_hint = _optimization_hint_or_none(lhs)
    rhs_hint = _optimization_hint_or_none(rhs)
    if lhs_hint is None or rhs_hint is None or lhs_hint != rhs_hint:
        return False

    # Hints are not shape semantics. They only show that the equality is true
    # for the traced example, so record the symbolic equality required for the
    # view rather than specializing either side to a concrete value.
    torch._check(lhs == rhs)
    return True


def _is_contiguous_with_hinted_checks(a: torch.Tensor) -> bool:
    from torch.fx.experimental.symbolic_shapes import guard_or_false, is_nested_int

    if guard_or_false(a.numel() < 2):
        return True

    numel_hint = _optimization_hint_or_none(a.numel())
    if numel_hint is None:
        return False
    if numel_hint < 2:
        torch._check(a.numel() < 2)
        return True

    expected_stride = 1
    expected_stride_max = 1
    for size, stride in reversed(tuple(zip(a.shape, a.stride()))):
        if guard_or_false(size == 1):
            continue

        size_hint = _optimization_hint_or_none(size)
        if size_hint == 1:
            torch._check(size == 1)
            continue

        if not _hinted_eq(stride, expected_stride) and not _hinted_eq(
            stride, expected_stride_max
        ):
            return False

        expected_stride_max *= size if is_nested_int(size) else torch.sym_max(size, 1)
        expected_stride *= size

    return True


def _view_has_unbacked_input(
    a: torch.Tensor, shape: ShapeType | tuple[ShapeType]
) -> bool:
    from torch.fx.experimental.symbolic_shapes import has_guarding_hint

    shape = utils.extract_shape_from_varargs(shape, validate=False)

    return (
        any(not has_guarding_hint(s) for s in a.size())
        or any(not has_guarding_hint(s) for s in a.stride())
        or any(not has_guarding_hint(s) for s in shape)
    )


def try_duck_specialization_first(a: torch.Tensor, shape) -> bool:
    """
    Collect candidate (x, y) pairs whose runtime hints match (so they
    *could* be equal), then test whether substituting them into the
    view's numel equation makes the equation symbolically zero.
    Only commit the equalities (adds the real Eq guards via bool(x == y)
    which triggers set_replacement) if the substitution actually solves
    the view validity check.

    This avoids polluting the shape env with spurious guards from probing.

    Returns True if at least one equality was committed.

    Conservative: only considers SymInts that appear as a top-level dim of
    the input or target (skipping the -1 marker), and skips already-equal
    pairs, and skips unbacked symbols.
    """
    import operator
    from functools import reduce

    from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols

    def _atomic_syms(dims) -> list[torch.SymInt]:
        """Keep only SymInts whose expr is a single bare *backed* symbol."""
        out: list[torch.SymInt] = []
        for s in dims:
            if not isinstance(s, torch.SymInt):
                continue
            expr = s.node.expr
            free = expr.free_symbols
            if len(free) != 1 or expr not in free:
                continue
            if free_unbacked_symbols(expr):
                continue
            out.append(s)
        return out

    import sympy

    a_syms = _atomic_syms(a.size())
    target_syms = _atomic_syms(shape)
    if not a_syms or not target_syms:
        return False

    # 1) Bucket symbols by hint, deduplicated by sympy expr (one SymInt per
    #    unique expr per hint).  Each non-trivial bucket is a duck-equality
    #    class: every member of the class will be set equal to the bucket's
    #    canonical symbol.  This keeps `subs` and `candidates` perfectly in
    #    sync so the numel check exactly reflects what we'll commit.
    from collections import defaultdict

    buckets: dict[int, dict[sympy.Expr, torch.SymInt]] = defaultdict(dict)
    for s in list(a_syms) + list(target_syms):
        # setdefault keeps the *first* SymInt seen for each (hint, expr).
        # pyrefly: ignore [bad-index]
        buckets[s.node.hint].setdefault(s.node.expr, s)

    candidates: list[tuple[torch.SymInt, torch.SymInt]] = []
    for members in buckets.values():
        canonical, *rest = members.values()
        for m in rest:
            candidates.append((canonical, m))

    if not candidates:
        return False

    # `subs` is derived from `candidates`, so they cannot drift.
    subs: dict[sympy.Expr, sympy.Expr] = {
        m.node.expr: c.node.expr for c, m in candidates
    }

    # 2) Symbolic test: does the substitution make the view's numel match?
    def _expr_of(s):
        return s.node.expr if isinstance(s, torch.SymInt) else s

    try:
        a_numel = reduce(operator.mul, (_expr_of(s) for s in a.size()), 1)
        shape_numel = reduce(operator.mul, (_expr_of(s) for s in shape), 1)
        diff = sympy.simplify((a_numel - shape_numel).subs(subs))
        if diff != 0:
            return False
    except (TypeError, ValueError, AttributeError, sympy.SympifyError):
        return False

    # 3) Commit the equalities for real (adds Eq guards, triggers set_replacement).
    for c, m in candidates:
        torch._check(c == m)
    return True


def _view_unbacked_meta(
    a: torch.Tensor,
    shape: ShapeType | tuple[ShapeType],
    size_oblivious_enabled: bool = True,
    allow_copy: bool = False,
    tried_duck_specialize: bool = False,
) -> torch.Tensor:
    from torch._prims import view_of
    from torch.fx.experimental.symbolic_shapes import guard_or_false, sym_eq

    # Creates a valid shape
    shape = utils.extract_shape_from_varargs(shape, validate=False)

    # Reshape may be given a shape with a -1 length
    # This indicates that the dimension's length should be inferred
    shape = utils.infer_size(shape, a.numel())

    # Special-cases reshaping zero dim tensors
    if a.ndim == 0:
        _a = a
        for length in shape:
            torch._check(length == 1)
            _a = torch._refs.unsqueeze(_a, -1)
        if _a is a:
            return view_of(a)
        else:
            return _a  # type: ignore[return-value]

    # Special-cases reshaping to zero dim tensors
    if len(shape) == 0:
        _a = a
        for length in a.shape:
            torch._check(length == 1)
            _a = torch._refs.squeeze(_a, -1)
        if _a is a:
            return view_of(a)
        else:
            return _a  # type: ignore[return-value]

    shape_numel = reduce(operator.mul, shape, 1)

    torch._check(
        a.numel() == shape_numel,
        lambda: f"Could not reshape a tensor with shape {a.shape} as a tensor with shape {shape}!",
    )

    if len(shape) == len(a.shape) and guard_or_false(sym_eq(shape, a.shape)):
        return view_of(a)

    if is_contiguous_or_false(a) if size_oblivious_enabled else is_contiguous(a):
        strides = make_contiguous_strides_for(shape)
        return a.as_strided(shape, strides)  # type: ignore[return-value]

    new_strides = _compute_stride(
        a.size(), a.stride(), shape, size_oblivious=size_oblivious_enabled
    )

    if new_strides is not None:
        return a.as_strided(shape, new_strides)  # type: ignore[return-value]

    # If we fail to do size oblivious view, and backed_size_oblivious was on,
    # then we redo everything by looking at hints and guarding instead of failing.
    # Also if the expression has unbacked symbols, then we run again with size_oblivious_enabled=False
    # to throw a data dependent error.

    if size_oblivious_enabled and (
        torch.fx.experimental._config.backed_size_oblivious
        or _view_has_unbacked_input(a, shape)
    ):
        # Heuristic: before falling through to the specializing recursion
        # (which often emits Eq(s, 1) via eval_eager), try to discover
        # cross-symbol equalities between input and target shape symbols.
        # If a pair (x, y) is True at the runtime hint, the shape env will
        # record Eq(x, y) and set_replacement unifies them — letting the
        # size-oblivious view path succeed without specialization on x==1.
        # But with potential duck specializations which are more general.
        if (
            torch.fx.experimental._config.unify_view_symbols_bso_meta
            and not tried_duck_specialize
            and try_duck_specialization_first(a, shape)
        ):
            return _view_unbacked_meta(
                a,
                shape,
                size_oblivious_enabled=True,
                allow_copy=allow_copy,
                tried_duck_specialize=True,
            )

        if _is_contiguous_with_hinted_checks(a):
            strides = make_contiguous_strides_for(shape)
            return a.as_strided(shape, strides)  # type: ignore[return-value]

        return _view_unbacked_meta(
            a, shape, size_oblivious_enabled=False, allow_copy=allow_copy
        )

    # When allow_copy=True (i.e., view_copy), define unbacked semantics
    # as "materialize": clone the input to break aliasing, then reshape.
    if allow_copy:
        strides = make_contiguous_strides_for(shape)
        # pyrefly: ignore[bad-return]
        return a.clone(memory_format=torch.contiguous_format).as_strided(shape, strides)

    msg = f"Cannot view a tensor with shape {a.shape} and strides {a.stride()} as a tensor with shape {shape}!"
    raise ValueError(msg)


@register_op_impl(aten._reshape_copy.default)
def _reshape_copy(
    fake_mode: FakeTensorMode, func: OpOverload, a: FakeTensor, *shape: Any
) -> FakeTensor | Exception:
    if a.is_sparse or a.is_mkldnn:
        return NotImplemented

    # pyrefly: ignore[bad-argument-count]
    shape = utils.infer_size(*shape, a.numel())
    if is_contiguous_or_false(a):
        view = _view_meta(fake_mode, func, a, *shape)
        return typing_cast(
            FakeTensor, view.clone(memory_format=torch.contiguous_format)
        )
    else:
        return _view_meta(
            fake_mode,
            func,
            typing_cast(FakeTensor, a.clone(memory_format=torch.contiguous_format)),
            *shape,
        )


@register_op_impl(aten.view.default)
@register_op_impl(aten._unsafe_view.default)
def _view_meta(
    fake_mode: FakeTensorMode,
    func: OpOverload,
    a: FakeTensor,
    *shape: Any,
    allow_copy: bool = False,
) -> FakeTensor:
    if torch.fx.experimental._config.backed_size_oblivious or _view_has_unbacked_input(
        a, shape
    ):
        return typing_cast(
            FakeTensor, _view_unbacked_meta(a, shape, allow_copy=allow_copy)
        )
    else:
        return typing_cast(
            FakeTensor,
            torch._refs._reshape_view_helper(a, *shape, allow_copy=allow_copy),
        )


@register_op_impl(aten.view_copy.default)
def _view_meta_copy(
    fake_mode: FakeTensorMode,
    func: OpOverload,
    a: FakeTensor,
    *shape: IntLikeType,
    out: FakeTensor | None = None,
) -> FakeTensor:
    # view_copy is the non-aliasing counterpart of view. Eager may succeed on
    # cases where a pure view is impossible (e.g. expand -> flatten) by
    # materializing the result. Match eager by allowing copy-if-needed in meta.
    result = _view_meta(fake_mode, func, a, *shape, allow_copy=True)

    if out is not None:
        return result

    return pytree.tree_map(
        lambda x: x.clone(memory_format=torch.contiguous_format),
        result,
    )


@register_op_impl(aten.repeat_interleave.Tensor)
def repeat_interleave_tensor(
    fake_mode: FakeTensorMode,
    func: OpOverload,
    repeats: FakeTensor,
    output_size: IntLikeType | None = None,
) -> FakeTensor:
    if output_size is None:
        if (
            fake_mode.shape_env is None
            or not fake_mode.shape_env.allow_dynamic_output_shape_ops
        ):
            raise DynamicOutputShapeException(func)

        output_size = fake_mode.shape_env.create_unbacked_symint()

        # Avoid importing sympy at a module level
        from torch.fx.experimental.symbolic_shapes import _constrain_range_for_size

        _constrain_range_for_size(output_size)
        # TODO: consider a memo
    return repeats.new_empty(output_size)  # type: ignore[return-value]


@register_op_impl(torch.ops.aten.item.default)
@register_op_impl(torch.ops.aten._local_scalar_dense.default)
def local_scalar_dense(
    fake_mode: FakeTensorMode, func: OpOverload, arg: FakeTensor
) -> int | float | bool | torch.SymInt | torch.SymFloat | torch.SymBool:
    if (r := arg.item_memo) is not None:
        return r
    if fake_mode.shape_env is None or (
        not fake_mode.shape_env.allow_scalar_outputs
        and not fake_mode.allow_scalar_outputs
    ):
        # Without symints/symfloats, cannot handle this
        raise DataDependentOutputException(func)
    if is_float_dtype(arg.dtype):
        r = fake_mode.shape_env.create_unbacked_symfloat()
    elif is_integer_dtype(arg.dtype):
        r = fake_mode.shape_env.create_unbacked_symint()
    elif is_boolean_dtype(arg.dtype):
        r = fake_mode.shape_env.create_unbacked_symbool()
    else:
        raise NotImplementedError(f"local_scalar_dense/item NYI for {arg.dtype}")
    arg.item_memo = r
    return r


@register_op_impl(torch.ops.aten.nonzero_numpy.default)
def nonzero_numpy(
    fake_mode: FakeTensorMode, func: OpOverload, arg: FakeTensor
) -> list[FakeTensor]:
    return torch.ops.aten.nonzero.default(arg).unbind(1)


@register_op_impl(torch.ops.aten.nonzero.default)
def nonzero(fake_mode: FakeTensorMode, func: OpOverload, arg: FakeTensor) -> FakeTensor:
    if (
        fake_mode.shape_env is None
        or not fake_mode.shape_env.allow_dynamic_output_shape_ops
    ):
        # Without symints/symfloats, cannot handle this
        raise DynamicOutputShapeException(func)

    if (nnz := arg.nonzero_memo) is None:
        # Avoid importing sympy at a module level
        from torch.fx.experimental.symbolic_shapes import (
            _constrain_range_for_size,
            has_free_symbols,
        )
        from torch.utils._sympy.numbers import IntInfinity
        from torch.utils._sympy.value_ranges import bound_sympy

        if not has_free_symbols(arg.numel()) and arg.numel() == 0:
            # If numel is zero, then the output size must be zero.
            # In this case, we must not allocate an unbacked SymInt,
            # because if we do, it will immediately get refined to
            # zero, but this will be inconsistent with size oblivious
            # tests (which will continue to claim that the unbacked
            # symint cannot equal zero).  We could also unconditionally
            # allocate an unbacked SymInt and not refine its range,
            # but this seems more precise.
            nnz = 0
        else:
            nnz = fake_mode.shape_env.create_unbacked_symint()

            maxval = sys.maxsize - 1

            if not has_free_symbols(arg.numel()):
                maxval = int(arg.numel())
            else:
                prod_node = math.prod(arg.shape).node  # type: ignore[union-attr]
                prod_range = bound_sympy(
                    prod_node.expr, prod_node.shape_env.var_to_range
                )
                if isinstance(prod_range.upper, IntInfinity):
                    maxval = sys.maxsize - 1
                else:
                    maxval = prod_range.upper

            _constrain_range_for_size(nnz, max=maxval)

        arg.nonzero_memo = nnz  # pyrefly: ignore[bad-assignment]
    return arg.new_empty_strided((nnz, arg.dim()), (1, nnz), dtype=torch.int64)  # type: ignore[return]


@register_op_impl(torch.ops.aten._padded_dense_to_jagged_forward.default)
def _padded_dense_to_jagged_forward(
    fake_mode: FakeTensorMode,
    func: OpOverload,
    padded: FakeTensor,
    offsets: list[FakeTensor],
    total_L: IntLikeType | None = None,
) -> FakeTensor:
    # only one jagged dim is supported for now
    if len(offsets) != 1:
        raise AssertionError(
            f"Only one jagged dim is supported, got {len(offsets)} offsets"
        )

    if total_L is None:
        if (
            fake_mode.shape_env is None
            or not fake_mode.shape_env.allow_dynamic_output_shape_ops
        ):
            # Without symints/symfloats, cannot handle this
            raise DynamicOutputShapeException(func)

        total_L = fake_mode.shape_env.create_unbacked_symint()

        maxval = sys.maxsize - 1

        # Avoid importing sympy at a module level
        from torch.fx.experimental.symbolic_shapes import (
            _constrain_range_for_size,
            has_free_symbols,
        )

        if not has_free_symbols(padded.numel()):
            maxval = int(padded.numel())

        _constrain_range_for_size(total_L, min=0, max=maxval)

    output_shape = (total_L, *padded.shape[2:])
    return padded.new_empty(output_shape)  # type: ignore[return]


def _compute_slice_index(size: IntLikeType, index: IntLikeType) -> IntLikeType | None:
    from torch.fx.experimental.symbolic_shapes import guard_or_false, sym_and

    if guard_or_false(sym_and(index >= 0, index <= size)):
        return index
    elif guard_or_false(sym_and(index < 0, index >= -size)):
        return index + size
    elif guard_or_false(index < -size):
        return 0
    elif guard_or_false(index > size):
        return size
    elif guard_or_false(index >= 0):
        return torch.sym_min(index, size)
    elif guard_or_false(index < 0):
        return torch.sym_max(index + size, 0)

    return None


@register_op_impl(torch.ops.aten.slice.Tensor)
def slice_forward(
    fake_mode: FakeTensorMode,
    func: OpOverload,
    self: FakeTensor,
    dim: int = 0,
    start: int | None = None,
    end: int | None = None,
    step: int = 1,
) -> FakeTensor:
    from torch.fx.experimental.symbolic_shapes import (
        guard_or_false,
        statically_known_true,
    )

    shape_env = fake_mode.shape_env
    ndim = self.dim()
    if ndim == 0:
        raise RuntimeError("slice() cannot be applied to a 0-dim tensor.")
    dim = canonicalize_dim(self.dim(), dim)
    sizes = list(self.size())
    strides = list(self.stride())

    if step <= 0:
        raise RuntimeError("slice step must be positive")

    # start, end
    start_index = 0 if start is None else _compute_slice_index(sizes[dim], start)
    end_index = (
        sizes[dim]
        if statically_known_true(end == sys.maxsize) or end is None
        else _compute_slice_index(sizes[dim], end)
    )

    # size
    new_size: IntLikeType | None = None
    if start_index is not None and end_index is not None:
        if guard_or_false(end_index >= start_index):
            new_size = (end_index - start_index + step - 1) // step  # type: ignore[bad-assignment]
        elif guard_or_false(start_index >= end_index):
            new_size = 0
        else:
            # Both indices are resolved but we can't statically determine their
            # ordering (e.g., when they involve Min/Max). Compute the size via
            # max(end - start, 0) to avoid creating an unbacked symint.
            diff = torch.sym_max(end_index - start_index, 0)
            new_size = (diff + step - 1) // step  # type: ignore[assignment]

    # create unbacked if case unknown
    if new_size is None:
        if shape_env is None:
            raise AssertionError("Must have shape_env to create symint")
        new_size = shape_env.create_unbacked_symint()
        torch._check(new_size >= 0)
        torch._check(new_size <= sizes[dim])

    # stride
    new_stride = strides[dim] * step

    # storage offset
    if start_index is not None:
        storage_offset = self.storage_offset() + start_index * strides[dim]
    else:
        if shape_env is None:
            raise AssertionError("Must have shape_env to create symint")
        storage_offset = shape_env.create_unbacked_symint()
        torch._check(storage_offset >= 0)

    sizes[dim] = new_size  # type: ignore[unsupported-operation]
    strides[dim] = new_stride
    if self.is_quantized:
        raise NotImplementedError(
            "Slice decomposition for quantized tensors aren't implemented"
        )
    else:
        return self.as_strided(sizes, strides, storage_offset)  # type: ignore[return-value]


@register_op_impl(torch.ops.aten.masked_select.default)
def masked_select(
    fake_mode: FakeTensorMode, func: OpOverload, self: FakeTensor, mask: FakeTensor
) -> FakeTensor:
    if (
        fake_mode.shape_env is None
        or not fake_mode.shape_env.allow_dynamic_output_shape_ops
    ):
        # Without symints/symfloats, cannot handle this
        raise DynamicOutputShapeException(func)

    nnz = fake_mode.shape_env.create_unbacked_symint()

    # see nonzero for commentary
    maxval = sys.maxsize - 1

    # Avoid importing sympy at a module level
    from torch.fx.experimental.symbolic_shapes import (
        _constrain_range_for_size,
        has_free_symbols,
    )
    from torch.utils._sympy.numbers import IntInfinity
    from torch.utils._sympy.value_ranges import bound_sympy

    # If num elements is expressed symbolically, calculate
    # the concrete value based on upper bounds. Otherwise,
    # we can set max val directly.
    if not has_free_symbols(self.numel()):
        num_elements = int(self.numel())
    else:
        prod_node = math.prod(self.shape).node  # type: ignore[union-attr]
        prod_range = bound_sympy(prod_node.expr, prod_node.shape_env.var_to_range)
        if isinstance(prod_range.upper, IntInfinity):
            num_elements = sys.maxsize - 1
        else:
            num_elements = prod_range.upper
    if num_elements > 2:
        maxval = num_elements

    _constrain_range_for_size(nnz, max=maxval)

    return self.new_empty((nnz,))  # type: ignore[return]


@register_op_impl(torch.ops.aten._assert_tensor_metadata.default)
def assert_tensor_metadata(
    fake_mode: FakeTensorMode,
    func: OpOverload,
    t: FakeTensor,
    size: list[int] | None = None,
    stride: list[int] | None = None,
    dtype: torch.dtype | None = None,
    *,
    device: torch.device | None = None,
    layout: torch.layout | None = None,
) -> None:
    if size is not None:
        if t.size() != tuple(size):
            raise AssertionError(
                f"Tensor size mismatch! Expected: {size}, Got: {t.size()}"
            )
    if stride is not None:
        if t.stride() != tuple(stride):
            raise AssertionError(
                f"Tensor stride mismatch! Expected: {stride}, Got: {t.stride()}"
            )
    if dtype is not None:
        if t.dtype != dtype:
            raise AssertionError(
                f"Tensor dtype mismatch! Expected: {dtype}, Got: {t.dtype}"
            )
    if layout is not None:
        if t.layout != layout:
            raise AssertionError(
                f"Tensor layout mismatch! Expected: {layout}, Got: {t.layout}"
            )
    if device is not None:
        if t.device != device:
            raise AssertionError(
                f"Tensor device mismatch! Expected: {device}, Got: {t.device}"
            )


# NB: this must be ordered after local_scalar_dense
@register_op_impl(lambda func: torch.Tag.data_dependent_output in func.tags)
def data_dep(
    fake_mode: FakeTensorMode, func: OpOverload, *args: Any, **kwargs: Any
) -> None:
    raise DataDependentOutputException(func)


# Bool Indices get Expanded as Masks
# See: IndexingUtils.h:expandTensors
def check_no_bool_index_tensors(
    func: OpOverload, self: FakeTensor, indices: list[FakeTensor | None]
) -> None:
    for index in indices:
        if index is not None and index.dtype in (torch.bool, torch.uint8):
            raise DynamicOutputShapeException(func)


def run_and_return_new_tensor_of_input_device(
    fake_mode: FakeTensorMode,
    func: OpOverload,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> FakeTensor:
    # TODO: ref
    _, new_kwargs = _normalize_function_or_error(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )
    out_device = new_kwargs["input"].device
    with in_kernel_invocation_manager(fake_mode):
        out = func(*args, **kwargs)
        if not is_noncontiguous_supported(out_device):
            out = out.new_empty(out.shape)

    if out is new_kwargs["input"]:
        return out  # copy_
    return FakeTensor(fake_mode, out, out_device)


_is_builtin_namespaces = ordered_set("aten", "prims", "prim")


def is_builtin(op: OpOverload) -> bool:
    return op.namespace in _is_builtin_namespaces


def has_meta(func: OpOverload) -> bool:
    return torch._C._dispatch_has_computed_kernel_for_dispatch_key(func.name(), "Meta")


def maybe_to_dense_mkldnn(
    fake_mode: FakeTensorMode,
    a: FakeTensor,
    dtype: torch.dtype | None = None,
    masked_grad: bool | None = None,
) -> object:
    # this function invokes in_kernel_invocation_manager and creates python
    # FakeTensor, revisit later for C++ behaviour
    if (
        not isinstance(a, FakeTensor)  # noqa: ISINSTANCE_FAKE_TENSOR
        or not a.is_mkldnn
    ):
        return NotImplemented

    out_dtype = dtype if dtype is not None else a.dtype
    if a.dtype not in _MKLDNN_TO_DENSE_DTYPES:
        raise RuntimeError(
            "mkldnn_to_dense expects float, bfloat16, half, uint8, int8, "
            "float8_e4m3fn tensor input"
        )
    if out_dtype not in _MKLDNN_TO_DENSE_DTYPES:
        raise RuntimeError(
            "mkldnn tensor only can be converted to be a float, bfloat16, Half, "
            "uint8, int8, float8_e4m3fn cpu tensor"
        )
    if a.dtype in _MKLDNN_DTYPE_CHANGE_RESTRICTED_DTYPES and out_dtype != a.dtype:
        raise RuntimeError(
            "For int8, uint8, float8 mkldnn_tensor input, we should not change "
            "the data type."
        )
    shape = tuple(a.shape)
    with in_kernel_invocation_manager(fake_mode):
        out = torch.empty_strided(
            shape,
            make_contiguous_strides_for(shape),
            dtype=out_dtype,
            device="meta",
        )
    return FakeTensor(fake_mode, out, a.fake_device)


class _ToDenseMkldnn(torch.autograd.Function):
    @staticmethod
    def forward(
        self: torch.Tensor,
        dtype: torch.dtype | None,
        masked_grad: bool | None,
    ) -> torch.Tensor:
        return aten._to_dense.default(self, dtype, masked_grad)

    @staticmethod
    def setup_context(
        ctx: Any,
        inputs: tuple[torch.Tensor, torch.dtype | None, bool | None],
        output: torch.Tensor,
    ) -> None:
        ctx.set_materialize_grads(False)
        ctx.input_is_mkldnn = inputs[0].is_mkldnn
        ctx.input_dtype = inputs[0].dtype

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(
        ctx: Any, grad: torch.Tensor | None
    ) -> tuple[torch.Tensor | None, None, None]:
        if grad is None:
            return None, None, None
        if ctx.input_is_mkldnn:
            return aten.to_mkldnn.default(grad, ctx.input_dtype), None, None
        # torch.func transforms may replay setup_context on a strided wrapper;
        # match that expected gradient layout instead of forcing MKLDNN.
        return grad.to(dtype=ctx.input_dtype), None, None


# Fake MKLDNN tensors keep MKLDNN-ness in a side channel because the backing
# meta tensor is strided.  The native composite reads TensorImpl::layout()
# directly, so this Python composite consults the layout override instead.
def to_dense_composite_impl(
    self: torch.Tensor,
    dtype: torch.dtype | None = None,
    *,
    masked_grad: bool | None = None,
) -> torch.Tensor:
    layout = self.layout
    # This composite is installed globally so direct aten.to_dense can see fake
    # MKLDNN tensors. Keep the common strided path minimal.
    if layout == torch.strided:
        return self.to(dtype=dtype) if dtype is not None else self
    if layout == torch._mkldnn:  # type: ignore[attr-defined]
        return _ToDenseMkldnn.apply(self, dtype, masked_grad)
    if layout in _TO_DENSE_SPARSE_LAYOUTS:
        return aten._to_dense.default(self, dtype, masked_grad)
    torch._check(False, lambda: f"to_dense does not support layout {layout}")
    raise AssertionError("unreachable")


def to_dense_functorch_frontmode_impl(
    self: torch.Tensor,
    dtype: torch.dtype | None = None,
    *,
    masked_grad: bool | None = None,
) -> torch.Tensor:
    interpreter = retrieve_current_functorch_interpreter()
    if interpreter.key() != TransformType.Grad or not is_functorch_wrapped_tensor(self):
        with torch.enable_grad(), interpreter.lower():
            return aten.to_dense.default(self, dtype=dtype, masked_grad=masked_grad)

    level = interpreter.level()

    class _ToDenseGrad(torch.autograd.function._SingleLevelFunction):
        @staticmethod
        def forward(
            ctx: Any,
            self: torch.Tensor,
            dtype: torch.dtype | None,
            masked_grad: bool | None,
        ) -> torch.Tensor:
            ctx.set_materialize_grads(False)
            unwrapped_self = _unwrap_for_grad(self, level)
            ctx.save_for_backward(unwrapped_self)
            ctx.input_dtype = unwrapped_self.dtype
            ctx.input_is_mkldnn = unwrapped_self.is_mkldnn
            ctx.masked_grad = masked_grad
            with torch.enable_grad(), interpreter.lower():
                if unwrapped_self.is_mkldnn:
                    out = _ToDenseMkldnn.apply(unwrapped_self, dtype, masked_grad)
                else:
                    out = aten.to_dense.default(
                        unwrapped_self, dtype=dtype, masked_grad=masked_grad
                    )
            return _wrap_for_grad(out, level)

        @staticmethod
        def backward(ctx: Any, *grads: Any) -> Any:
            (grad,) = grads
            if grad is None:
                return None, None, None
            if ctx.input_is_mkldnn:
                # Functorch grad transforms replay setup_context on a strided
                # wrapper, so return the wrapper's expected dense gradient.
                return grad.to(dtype=ctx.input_dtype), None, None
            (input,) = ctx.saved_tensors
            return (
                aten.to_dense_backward.default(grad, input, ctx.masked_grad),
                None,
                None,
            )

    with enable_single_level_autograd_function():
        # pyrefly: ignore [missing-attribute]
        return _ToDenseGrad.apply(self, dtype, masked_grad)


_to_dense_functorch_lib = torch.library.Library(
    "aten", "IMPL", "FuncTorchDynamicLayerFrontMode"
)
_to_dense_functorch_lib.impl("to_dense", to_dense_functorch_frontmode_impl)


def to_dense_python_tls_impl(
    self: torch.Tensor,
    dtype: torch.dtype | None = None,
    *,
    masked_grad: bool | None = None,
) -> torch.Tensor:
    from torch._subclasses.functional_tensor import FunctionalTensor

    if isinstance(self, (FakeTensor, FunctionalTensor)):  # noqa: ISINSTANCE_FAKE_TENSOR
        return to_dense_composite_impl(self, dtype=dtype, masked_grad=masked_grad)

    with torch._C._ExcludeDispatchKeyGuard(_PYTHON_TLS_SNAPSHOT_KEYSET):
        return aten.to_dense.default(self, dtype=dtype, masked_grad=masked_grad)


# Intercept fake/functional tensor calls before the native composite sees their
# strided wrapper metadata, but keep ordinary eager CPU/CUDA to_dense on the C++
# dispatcher path.
_to_dense_python_tls_lib = torch.library.Library("aten", "IMPL", "PythonTLSSnapshot")
_to_dense_python_tls_lib.impl("to_dense", to_dense_python_tls_impl)


# Keep this as a Python decomposition for FakeTensorMode fallback paths without
# replacing the eager dispatcher entry for every ordinary to_dense() call.
aten.to_dense.default.py_impl(torch._C.DispatchKey.CompositeImplicitAutograd)(
    to_dense_composite_impl
)


@register_op_impl(aten.to_mkldnn.default)
def to_mkldnn(
    fake_mode: FakeTensorMode,
    func: OpOverload,
    a: FakeTensor,
    dtype: torch.dtype | None = None,
) -> object:
    if not isinstance(a, FakeTensor):  # noqa: ISINSTANCE_FAKE_TENSOR
        return NotImplemented

    out_dtype = dtype if dtype is not None else a.dtype
    if a.fake_device.type != "cpu":
        raise RuntimeError("dense_to_mkldnn expects CPU tensor input")
    if a.layout != torch.strided:
        raise RuntimeError("dense_to_mkldnn expects strided tensor input")
    if a.dtype not in _MKLDNN_DTYPES:
        raise RuntimeError(
            "dense_to_mkldnn expects float, bfloat16, half, uint8, int8 tensor input"
        )
    if a.dim() > 5:
        raise RuntimeError("Can't convert cpu tensor with the number of dimensions > 5")
    if a.dtype in (torch.uint8, torch.int8) and out_dtype != a.dtype:
        raise RuntimeError(
            "For int8, uint8 cpu_tensor input, we should not change the data type."
        )
    if out_dtype not in _MKLDNN_DTYPES:
        raise RuntimeError(
            "cpu tensor only can be converted to be a float, bfloat16, half, "
            "uint8, int8 mkldnn tensor"
        )
    shape = tuple(a.shape)
    with in_kernel_invocation_manager(fake_mode):
        # Real MKLDNN tensors report opaque strides as (1, 0, ...); mirror that
        # while carrying the layout bit through dispatch_keys.
        out = torch.empty_strided(
            shape,
            (1,) + (0,) * (len(shape) - 1),
            dtype=out_dtype,
            device="meta",
        )
    return FakeTensor(
        fake_mode, out, a.fake_device, dispatch_keys=_MKLDNN_DISPATCH_KEYS
    )


# These are for the `torch._foreach_...` ops like `torch._foreach_add`.
@register_op_impl(
    lambda func: is_builtin(func)
    and func.name().startswith("aten::_foreach_")
    and has_meta(func)
)
def foreach_run_and_map_input_device(
    fake_mode: FakeTensorMode, func: OpOverload, *args: Any, **kwargs: Any
) -> list[FakeTensor] | None:
    tensor_lists = [
        arg
        for arg in itertools.chain(args, kwargs.values())
        if isinstance(arg, (list, tuple))
        and len(arg)
        and isinstance(arg[0], torch.Tensor)
    ]

    try:
        with in_kernel_invocation_manager(fake_mode):
            out_meta = func(*args, **kwargs)
    except NotImplementedError:
        return NotImplemented

    if not out_meta:
        return out_meta

    if not tensor_lists:
        raise AssertionError("tensor_lists must not be empty")
    out_fake = []

    for i, meta_t in enumerate(out_meta):
        device, _ = FakeTensor._find_common_device(func, [tl[i] for tl in tensor_lists])
        out_fake.append(
            fake_mode.fake_tensor_converter.from_meta_and_device(
                fake_mode, meta_t, device
            )
        )

    return out_fake


# Don't default to default device handling,
# Since op can take in non-zero sized cpu
# index tensors with cuda self
@register_op_impl(aten.index.Tensor)
def index_tensor(
    fake_mode: FakeTensorMode, func: OpOverload, *args: Any, **kwargs: Any
) -> FakeTensor:
    from torch._meta_registrations import meta_index_Tensor

    _, new_kwargs = _normalize_function_or_error(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    out_device = new_kwargs["input"].device
    # ensure nonzero call goes to fake tensor
    with fake_mode:
        out = meta_index_Tensor(*args, **kwargs)
        return out.to(out_device)


# Can take mixed meta/non-meta arguments; the meta registration
# will roughly do the right thing even when given real devices
@register_op_impl(aten._embedding_bag.default)
def embedding_bag(
    fake_mode: FakeTensorMode, func: OpOverload, *args: Any, **kwargs: Any
) -> tuple[FakeTensor, FakeTensor, FakeTensor, FakeTensor]:
    from torch._meta_registrations import meta_embedding_bag

    with fake_mode:
        return meta_embedding_bag(*args, **kwargs)


@register_op_impl(aten.embedding.default)
def embedding(
    fake_mode: FakeTensorMode, func: OpOverload, *args: Any, **kwargs: Any
) -> Any:
    from torch._meta_registrations import embedding as meta_embedding

    _, new_kwargs = _normalize_function_or_error(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )
    weight = new_kwargs["weight"]
    indices = new_kwargs["indices"]
    if (
        weight.device != indices.device
        and weight.device.type != "meta"
        and indices.device.type != "meta"
    ):
        return NotImplemented

    with fake_mode:
        return typing_cast(FakeTensor, meta_embedding(*args, **kwargs))


# takes in multiple-devices, don't default to default device handling
@register_op_impl(aten._unsafe_index_put.default)
@register_op_impl(aten.copy.default)
@register_op_impl(aten.copy_.default)
@register_op_impl(aten.slice_scatter.default)
@register_op_impl(aten.diagonal_scatter.default)
def multi_device_op_default(
    fake_mode: FakeTensorMode, func: OpOverload, *args: Any, **kwargs: Any
) -> FakeTensor:
    return run_and_return_new_tensor_of_input_device(fake_mode, func, args, kwargs)


@register_op_impl(aten.shallow_copy_data_.default)
def _(
    fake_mode: FakeTensorMode, func: OpOverload, *args: Any, **kwargs: Any
) -> FakeTensor:
    _, new_kwargs = _normalize_function_or_error(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )
    source_device = new_kwargs["source"].device
    with in_kernel_invocation_manager(fake_mode):
        func(*args, **kwargs)
    new_kwargs["input"].fake_device = source_device
    return new_kwargs["input"]


# same with multi_device_op_default, but return the input
@register_op_impl(aten.copy.out)
@register_op_impl(aten.slice_scatter.out)
def multi_device_op_out(
    fake_mode: FakeTensorMode, func: OpOverload, *args: Any, **kwargs: Any
) -> FakeTensor:
    with in_kernel_invocation_manager(fake_mode):
        func(*args, **kwargs)

    _, new_kwargs = _normalize_function_or_error(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    return new_kwargs["input"]


@register_op_impl(aten.index_put.default)
@register_op_impl(aten.index_put_.default)
def index_put_impl(
    fake_mode: FakeTensorMode, func: OpOverload, *args: Any, **kwargs: Any
) -> FakeTensor:
    _, new_kwargs = _normalize_function_or_error(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )
    values = new_kwargs["values"]
    self_device = new_kwargs["input"].fake_device
    torch._check(
        self_device == values.fake_device or (values.ndim == 0 and values.numel() == 1),
        lambda: f"Mismatching {func} device between self ({self_device}) and values ({values.device})",
    )

    out = run_and_return_new_tensor_of_input_device(fake_mode, func, args, kwargs)
    if func is aten.index_put_.default:
        return new_kwargs["input"]
    else:
        return out


@register_op_impl(aten._nested_tensor_from_tensor_list.default)
@register_op_impl(aten._nested_tensor_from_tensor_list.out)
@register_op_impl(aten._nested_view_from_buffer.default)
@register_op_impl(aten._nested_view_from_buffer_copy.default)
def nested_tensors_unsupported(
    fake_mode: FakeTensorMode, func: OpOverload, *args: Any, **kwargs: Any
) -> None:
    raise UnsupportedOperatorException(func)


@register_op_impl(
    [
        x
        for x in _device_not_kwarg_ops
        if x
        not in (
            # these are already registered elsewhere
            aten.is_pinned.default,
            aten.to.device,
            aten.to.prim_Device,
            aten._nested_tensor_from_tensor_list.default,
            aten._nested_tensor_from_tensor_list.out,
        )
    ]
)
def nyi(fake_mode: FakeTensorMode, func: OpOverload, *args: Any, **kwargs: Any) -> None:
    if func in _device_not_kwarg_ops:
        raise AssertionError(f"NYI: {func}")


@register_op_impl(
    [
        aten.convolution.default,
        aten._convolution.default,
        aten._convolution.deprecated,
        aten.convolution_backward.default,
    ]
)
def conv(
    fake_mode: FakeTensorMode, func: OpOverload, *args: Any, **kwargs: Any
) -> FakeTensor | tuple[FakeTensor | None, FakeTensor | None, FakeTensor | None]:
    _, new_kwargs = _normalize_function_or_error(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    def expect_fake_tensor(name: str, value: object) -> FakeTensor:
        if not isinstance(value, FakeTensor):  # noqa: ISINSTANCE_FAKE_TENSOR
            raise AssertionError(
                "Expected fake convolution tensor arguments to be FakeTensors, "
                f"but {name} was {type(value).__name__}"
            )
        return value

    input_ = expect_fake_tensor("input", new_kwargs["input"])
    weight = expect_fake_tensor("weight", new_kwargs["weight"])
    device = input_.fake_device
    # Internal passes such as Inductor freezing may run fake propagation over
    # folded convs that do not need to match eager's public input checks.
    if (
        func
        in (
            aten.convolution.default,
            aten._convolution.default,
            aten._convolution.deprecated,
        )
        and input_.dtype != weight.dtype
        and not input_.is_mkldnn
        and not fake_mode.allow_non_fake_inputs
    ):
        raise RuntimeError(
            f"Input type ({input_.dtype}) and weight type "
            f"({weight.dtype}) should be the same"
        )
    for name, value in new_kwargs.items():
        if isinstance(value, torch.Tensor):
            fake_value = expect_fake_tensor(name, value)
            if not _same_device_or_unspecified_index(fake_value.fake_device, device):
                raise RuntimeError(
                    "Expected all tensors to be on the same device, but got "
                    f"{name} is on {fake_value.fake_device}, different from "
                    f"other tensors on {device}"
                )
    # need to re-enable mode so the tensors report fake device
    with fake_mode:
        # if the input is unsqueezed in Convolution.cpp we get segfault
        k = weight.ndim

        # Avoid importing sympy at a module level
        from torch.fx.experimental.symbolic_shapes import has_guarding_hint

        all_hinted = all(has_guarding_hint(s) for s in input_.shape) and all(
            has_guarding_hint(s) for s in weight.shape
        )

        if not all_hinted:
            # TODO: We can make this a little more faithful with best effort
            # channels last detection (but only if it's statically obvious!)
            mem_fmt = None
        elif input_.dim() == k - 1:
            # Unbatched input: eager routes directly to the slow dilated
            # backward kernels, which produce contiguous gradients and bypass
            # the channels-last backend-memory-format selection. Mirror that
            # here; _select_conv_backend would also reject the missing batch
            # dim.
            mem_fmt = None
        else:
            # convolution has "bias" but not "bias_sizes"; convolution_backward
            # has "bias_sizes" but not "bias". .get() handles both with one call.
            bias = new_kwargs.get("bias")
            select_kwargs: dict[str, object] = dict(
                stride=new_kwargs["stride"],
                padding=new_kwargs["padding"],
                dilation=new_kwargs["dilation"],
                transposed=new_kwargs["transposed"],
                output_padding=new_kwargs["output_padding"],
                groups=new_kwargs["groups"],
                bias=bias,
            )
            if bias is None:
                select_kwargs["bias_sizes"] = new_kwargs.get("bias_sizes")
            conv_backend = torch._C._select_conv_backend(
                input_, weight, **select_kwargs
            )
            # Expand 1d -> 2d.
            # Note: Avoid expanding before calling _select_conv_backend,
            # as the function handles 2D expansion internally.
            if k == 3 and not input_.is_mkldnn and not input_.is_xpu:
                # Note: Using input.to(memory_format=contiguous) does not work.
                input_ = input_.contiguous().unsqueeze(2)
                weight = weight.unsqueeze(2)
            mem_fmt = torch._C._conv_determine_backend_memory_format(
                input_, weight, conv_backend
            )

    def convert(
        t: torch.Tensor | None, mem_fmt: torch.memory_format | None
    ) -> FakeTensor | None:
        if t is None:
            return t
        if mem_fmt is not None:
            # channels last only support 4d, try to expand dim then convert it back later.
            if t.dim() == 3 and mem_fmt == torch.channels_last:
                t = t.unsqueeze(2).to(memory_format=mem_fmt).squeeze(2)
            else:
                t = t.to(memory_format=mem_fmt)
        return FakeTensor(fake_mode, t, device)

    with in_kernel_invocation_manager(fake_mode):
        out = func(**new_kwargs)

        if func in (
            aten.convolution.default,
            aten._convolution.default,
            aten._convolution.deprecated,
        ):
            return convert(out, mem_fmt)  # type: ignore[return]
        else:
            return (
                convert(out[0], mem_fmt),
                convert(out[1], mem_fmt),
                convert(out[2], None),
            )


@register_op_impl(torch.ops.aten.bincount.default)
def bincount(
    fake_mode: FakeTensorMode,
    func: OpOverload,
    inputs: FakeTensor,
    weights: FakeTensor | None = None,
    minlength: IntLikeType = 0,
) -> FakeTensor:
    if (
        fake_mode.shape_env is None
        or not fake_mode.shape_env.allow_dynamic_output_shape_ops
    ):
        # Without symints/symfloats, cannot handle this
        raise DynamicOutputShapeException(func)

    from torch.fx.experimental.symbolic_shapes import (
        _constrain_range_for_size,
        has_free_symbols,
    )

    if not has_free_symbols(inputs.numel()) and inputs.numel() == 0:
        return inputs.new_empty(minlength, dtype=torch.int64)  # type: ignore[return]

    new_size = fake_mode.shape_env.create_unbacked_symint()

    _constrain_range_for_size(new_size)
    torch._check(new_size >= minlength)
    if weights is None:
        return inputs.new_empty(new_size, dtype=torch.int64)  # type: ignore[return]

    if weights.device.type == "mps":
        dtype = (
            weights.dtype
            if weights.dtype in (torch.float32, torch.int32, torch.float16)
            else torch.int32
        )
    else:
        dtype = weights.dtype if weights.dtype == torch.float32 else torch.float64
    return weights.new_empty(new_size, dtype=dtype)  # type: ignore[return]


@register_op_impl(torch.ops.aten._pack_padded_sequence.default)
def _pack_padded_sequence(
    fake_mode: FakeTensorMode,
    func: OpOverload,
    inputs: FakeTensor,
    lengths: FakeTensor,
    batch_first: bool,
) -> tuple[FakeTensor, FakeTensor]:
    if (
        fake_mode.shape_env is None
        or not fake_mode.shape_env.allow_dynamic_output_shape_ops
    ):
        # Without symints/symfloats, cannot handle this
        raise DynamicOutputShapeException(func)

    new_batch_size = fake_mode.shape_env.create_unbacked_symint()

    from torch.fx.experimental.symbolic_shapes import _constrain_range_for_size

    _constrain_range_for_size(new_batch_size)

    if not batch_first:
        # Inputs should have shape (batch_size, seq_len, *)
        inputs = inputs.transpose(0, 1)  # type: ignore[assignment]

    res_size = inputs.shape[1:]
    packed_data = inputs.new_empty(res_size)
    batch_size = inputs.new_empty((new_batch_size,))
    return (packed_data, batch_size)  # type: ignore[return]


def _fake_alias(fake_mode: FakeTensorMode, x: FakeTensor) -> FakeTensor:
    with no_python_dispatcher(), in_kernel_invocation_manager(fake_mode):
        out = aten.alias.default(x)
    # Real MKLDNN alias is not public API, but compiler-generated alias nodes
    # must preserve fake MKLDNN layout state through tracing.
    return FakeTensor(fake_mode, out, x.device, dispatch_keys=x.dispatch_keys)


@register_op_impl(aten.alias.default)
def fake_alias(
    fake_mode: FakeTensorMode, func: OpOverload, x: FakeTensor
) -> FakeTensor | object:
    if not isinstance(x, FakeTensor):  # noqa: ISINSTANCE_FAKE_TENSOR
        return NotImplemented
    return _fake_alias(fake_mode, x)


# pyrefly: ignore [implicit-any]
FAST_OP_IMPLEMENTATIONS = {}


# Unlike register_op_impl, these don't do the slow iteration for
# run_impl_check, and these run BEFORE decompositions
def register_fast_op_impl(
    func: OpOverload,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    def impl_decorator(op_impl: Callable[_P, _R]) -> Callable[_P, _R]:
        FAST_OP_IMPLEMENTATIONS[func] = op_impl
        return op_impl

    return impl_decorator


# infer_size_impl in ExpandUtils
def infer_size(
    a: Sequence[IntLikeType], b: Sequence[IntLikeType]
) -> tuple[IntLikeType, ...]:
    from torch.fx.experimental.symbolic_shapes import guard_or_false

    dimsA = len(a)
    dimsB = len(b)
    ndim = max(dimsA, dimsB)
    expandedSizes: list[IntLikeType] = [0] * ndim
    for i in range(ndim - 1, -1, -1):
        offset = ndim - 1 - i
        dimA = dimsA - 1 - offset
        dimB = dimsB - 1 - offset
        sizeA = a[dimA] if dimA >= 0 else 1
        sizeB = b[dimB] if dimB >= 0 else 1

        # NB: It is very important to test for broadcasting, before testing
        # sizeA == sizeB.  This is because the broadcasting tests are likely
        # to be statically known (in particular, if sizeA/sizeB is unbacked
        # but size-like, we will unsoundly assume they never equal 1), but
        # the sizeA == sizeB test may not be statically known.  However, once
        # we have established that no broadcasting is happening, the
        # sizeA == sizeB is now expect_true and we can defer it as a runtime
        # assert (this works because Python will return the terminal
        # expression of an or statement as-is, without bool()'ing it; if this
        # were not the case, we'd need to write this using torch.sym_or() or
        # something like that).
        torch._check(
            guard_or_false(sizeA == 1) or guard_or_false(sizeB == 1) or sizeA == sizeB,
            lambda: f"The size of tensor a ({sizeA}) "
            f"must match the size of tensor b ({sizeB}) "
            f"at non-singleton dimension {i})",
        )
        expandedSizes[i] = sizeB if guard_or_false(sizeA == 1) else sizeA
    return tuple(expandedSizes)


def make_fast_binary_impl(
    slow_ref: Callable[..., Any],
    type_promotion_kind: ELEMENTWISE_TYPE_PROMOTION_KIND = ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
) -> Callable[..., FakeTensor]:
    def fast_binary_impl(mode: FakeTensorMode, *args: Any, **kwargs: Any) -> FakeTensor:
        def slow(msg: str) -> FakeTensor:
            count_label(f"slow {msg}")
            with mode:
                return slow_ref(*args, **kwargs)

        count_label("attempt fast")

        # Fast path (based off of TensorIterator fast path).
        # Unfortunately, there is no way to easily deduplicate
        # this with either the TensorIterator C++ implementation
        # (which we don't want to SymIntify, and also the algorithm
        # here is slightly different from TensorIterator to allow
        # for broadcasting), nor the PrimTorch implementation
        # (which does not actually implement a fast path.)

        operands = args

        # compute_shape
        final_shape: ShapeType | None = None
        for op in operands:
            shape: ShapeType = op.shape if isinstance(op, torch.Tensor) else ()
            if final_shape is None:
                final_shape = shape
            # TODO: Minor optimization: track if the shapes
            # were equal so you can skip the equality check
            # below if unnecessary
            # pyrefly: ignore[bad-assignment]
            final_shape = infer_size(final_shape, shape)
        if final_shape is None:
            raise AssertionError("final_shape must not be None")

        from torch.fx.experimental.symbolic_shapes import guard_or_false, sym_eq

        # Do some extra safety checks to see if the output
        # stride is obvious
        for op in operands:
            if (
                isinstance(op, torch.Tensor)
                and len(op.shape) == len(final_shape)
                # take the slow path if result is not determined.
                and guard_or_false(sym_eq(op.shape, final_shape))  # type: ignore[arg-type]
            ):
                break
        else:
            # if we never break in the for loop above we take the slow path.
            return slow("both tensors nontrivially broadcast")

        # compute_types
        cpu = torch.device("cpu")
        common_device: torch.device = cpu
        common_dtype: torch.dtype | None = None
        has_different_input_dtypes = False
        for op in operands:
            if not isinstance(op, torch.Tensor):
                # Use elementwise_dtypes for the tricky case
                has_different_input_dtypes = True
                continue
            if common_device == cpu and op.device.type != "cpu":
                common_device = op.device
            if common_dtype is None:
                if type_promotion_kind != ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT:
                    has_different_input_dtypes = True
                else:
                    common_dtype = op.dtype
            elif common_dtype != op.dtype:
                has_different_input_dtypes = True

        if has_different_input_dtypes:
            # compute promotion
            # TODO: we don't need the compute type
            _, common_dtype = elementwise_dtypes(
                *operands, type_promotion_kind=type_promotion_kind
            )

        # check all tensors on same device
        # cpu scalars are assumed allow
        current_cpu_scalars_on_non_cpu = 0
        max_cpu_scalars_on_non_cpu = 1  # hard coded atm
        for op in operands:
            if not isinstance(op, torch.Tensor):
                continue
            if common_device != cpu and op.dim() == 0 and op.device == cpu:
                if current_cpu_scalars_on_non_cpu >= max_cpu_scalars_on_non_cpu:
                    return slow("error")
                current_cpu_scalars_on_non_cpu += 1
            elif op.device != common_device:
                return slow("error")

        # compute_fast_setup_type
        definitely_contiguous = True
        definitely_channels_last = True

        # TODO: is_non-overlapping_and_dense not bound from Python
        # no inplace, no out, everything defined

        if is_noncontiguous_supported(common_device):
            for op in operands:
                if not isinstance(op, torch.Tensor):
                    continue
                definitely_contiguous = (
                    definitely_contiguous
                    and is_contiguous_for_memory_format_or_false(
                        op, memory_format=torch.contiguous_format
                    )
                )
                definitely_channels_last = (
                    definitely_channels_last
                    and is_contiguous_for_memory_format_or_false(
                        op, memory_format=torch.channels_last
                    )
                )
        if definitely_contiguous:
            # do contiguous
            count_label("fast is_contiguous")
            return FakeTensor(
                mode,
                torch.empty(
                    final_shape,
                    dtype=common_dtype,
                    device="meta",
                    memory_format=torch.contiguous_format,
                ),
                device=common_device,
            )
        if definitely_channels_last:
            count_label("fast channels_last")
            # do channels last
            return FakeTensor(
                mode,
                torch.empty(
                    final_shape,
                    dtype=common_dtype,
                    device="meta",
                    memory_format=torch.channels_last,
                ),
                device=common_device,
            )

        return slow("no contiguity match")

    return fast_binary_impl


# disable the python dispatcher to avoid decomposing detach() further
# (proxy_mode should still decompose detach() though)
def fast_detach(
    fake_mode: FakeTensorMode | None,
    x: FakeTensor | torch.Tensor,
    include_real: bool = False,
) -> torch.Tensor:
    if (
        not isinstance(x, FakeTensor)  # noqa: ISINSTANCE_FAKE_TENSOR
        or fake_mode is None
    ):
        raise AssertionError(
            "type widening added for cpp faketensor but this is not used yet"
        )
    with no_python_dispatcher(), in_kernel_invocation_manager(fake_mode):
        out = torch.ops.aten.detach.default(x)
    dispatch_keys = x.dispatch_keys
    if include_real:
        return FakeTensor(
            fake_mode,
            out,
            x.device,
            real_tensor=x.real_tensor,
            dispatch_keys=dispatch_keys,
        )
    return FakeTensor(fake_mode, out, x.device, dispatch_keys=dispatch_keys)


def fast_alias(fake_mode: FakeTensorMode, x: FakeTensor) -> FakeTensor:
    return _fake_alias(fake_mode, x)


@functools.cache
def get_fast_op_impls() -> dict[OpOverload, Callable[..., Any]]:
    import torch._refs

    register_fast_op_impl(torch.ops.aten.add.Tensor)(
        make_fast_binary_impl(torch._refs.add)
    )
    register_fast_op_impl(torch.ops.aten.sub.Tensor)(
        make_fast_binary_impl(torch._refs.sub)
    )
    register_fast_op_impl(torch.ops.aten.mul.Tensor)(
        make_fast_binary_impl(torch._refs.mul)
    )  # type: ignore[has-type]
    register_fast_op_impl(torch.ops.aten.div.Tensor)(
        make_fast_binary_impl(
            torch._refs.div,
            type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        )
    )
    register_fast_op_impl(torch.ops.aten.detach.default)(fast_detach)
    register_fast_op_impl(torch.ops.aten.alias.default)(fast_alias)
    return FAST_OP_IMPLEMENTATIONS
<|endoftext|>from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import SymInt
from torch.fx.experimental.sym_node import SymNode
from torch.types import py_sym_types, PySymType


if TYPE_CHECKING:
    import sympy

    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    from .fake_tensor import _DispatchCacheKey, _MetadataIntLike


@dataclass(frozen=True, slots=True)
class _DeconstructedSymNode:
    """
    Represents a SymNode without the associated ShapeEnv
    """

    # n.b. keep the same protocol as SymNode
    _expr: sympy.Expr
    pytype: type
    _hint: int | float | bool | None
    constant: int | float | bool | None
    fx_node: torch.fx.Node

    @staticmethod
    def from_node(node: SymNode) -> _DeconstructedSymNode:
        return _DeconstructedSymNode(
            node._expr,
            node.pytype,
            # pyrefly: ignore[bad-argument-type]
            node._hint,
            node.constant,
            # pyrefly: ignore[bad-argument-type]
            node.fx_node,
        )

    def extract(self, shape_env: ShapeEnv) -> SymNode:
        return SymNode(
            self._expr, shape_env, self.pytype, self._hint, self.constant, self.fx_node
        )

    def __str__(self) -> str:
        return str(self._expr)

    def __repr__(self) -> str:
        return f"_DeconstructedSymNode{{{self._expr!r}, {self.pytype!r}, {self._hint!r}, {self.constant!r}, {self.fx_node!r}}}"

    def __eq__(self, other: object) -> bool:
        raise NotImplementedError

    def __hash__(self) -> int:
        raise NotImplementedError

    # _value_eq to match SymNode
    def _value_eq(self, other: object) -> bool:
        if isinstance(other, (SymNode, _DeconstructedSymNode)):
            return (
                self._expr == other._expr
                and self.pytype == other.pytype
                and self._hint == other._hint
                and self.constant == other.constant
                and self.fx_node == other.fx_node
            )
        else:
            return False

    # _value_hash to match SymNode
    def _value_hash(self) -> int:
        return hash((self._expr, self.pytype, self._hint, self.constant, self.fx_node))


@dataclass(frozen=True, slots=True)
class _DeconstructedSymType:
    """
    Represents a SymInt, SymFloat, SymBool without the associated ShapeEnv
    """

    ty: type[PySymType]
    node: _DeconstructedSymNode

    @staticmethod
    def from_sym_type(value: PySymType) -> _DeconstructedSymType:
        # pyrefly: ignore [bad-argument-type]
        return _DeconstructedSymType(type(value), value.node)

    def extract(self, shape_env: ShapeEnv) -> PySymType:
        return self.ty(self.node.extract(shape_env))

    def __str__(self) -> str:
        return f"{self.ty}({self.node})"

    def __repr__(self) -> str:
        return f"_DeconstructedSymType({self.ty}, {self.node!r})"

    def __eq__(self, other: object) -> bool:
        return NotImplemented

    def __hash__(self) -> int:
        return NotImplemented


@dataclass(frozen=True, slots=True)
class _InputBackref:
    value: int


@dataclass(slots=True)
class _PySymInputStub:
    """
    Represents a SymInt in the cached key. Needed because SymInt doesn't
    support __eq__ or __hash__ directly.
    """

    # value can be:
    #   PySymType: This is the 'normal' SymInt value, wrapped so we can use
    #              hash/eq as value hash/eq (normally SymInt does object
    #              hash/eq).
    #   _DeconstructedSymType: This is used when storing the _PySymInputStub in
    #                          the cache to avoid cyclic ShapeEnv references.
    #   _InputBackref: This is a back-reference to a previous _PySymInputStub in
    #                  the key.
    value: PySymType | _DeconstructedSymType | _InputBackref

    def __init__(
        self, value: PySymType | _DeconstructedSymType | _InputBackref
    ) -> None:
        # For inputs (values in the `key`) we need to keep the PySymType intact
        # - this way if we need to reuse it as an output we can properly copy
        # the original value.
        self.value = value

    def strip_shape_env(self) -> None:
        if isinstance(self.value, py_sym_types):
            self.value = _DeconstructedSymType.from_sym_type(self.value)

    def extract(self, shape_env: ShapeEnv) -> PySymType:
        if isinstance(self.value, _DeconstructedSymType):
            return self.value.extract(shape_env)
        else:
            # We should never see an _InputBackref here - anyone extracting a
            # value should be pulling from the original entry (the one this
            # backref points at).
            if isinstance(self.value, _InputBackref):
                raise AssertionError(
                    "Cannot extract value from _InputBackref - use the original entry"
                )
            return self.value

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return f"_PySymInputStub({self.value!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _PySymInputStub):
            return False
        elif isinstance(self.value, _InputBackref) or isinstance(
            other.value, _InputBackref
        ):
            return self.value == other.value  # type: ignore[bad-return]
        else:
            # pyrefly: ignore [bad-argument-type]
            return self.value.node._value_eq(other.value.node)

    def __hash__(self) -> int:
        if isinstance(self.value, _InputBackref):
            return hash(self.value)
        else:
            return self.value.node._value_hash()


@dataclass(slots=True)
class _SymIntOutputStub:
    """
    Represents a SymInt in the cached output.
    """

    # This is either an `int` which represents the index in the key to copy the
    # SymNode from or it's the deconstructed SymNode itself.
    value: int | _DeconstructedSymNode

    def __init__(self, value: SymInt, key_path: int | None) -> None:
        if key_path is None:
            self.value = _DeconstructedSymNode.from_node(value.node)
        else:
            self.value = key_path

    def extract(self, key: _DispatchCacheKey, shape_env: ShapeEnv) -> SymInt:
        if isinstance(self.value, _DeconstructedSymNode):
            return SymInt(self.value.extract(shape_env))
        else:
            src = key.key[self.value]
            if not isinstance(src, _PySymInputStub) or not isinstance(
                src.value, SymInt
            ):
                raise AssertionError(
                    f"Expected _PySymInputStub with SymInt value, got {type(src)} "
                    f"with {type(src.value) if isinstance(src, _PySymInputStub) else 'N/A'}"
                )
            return src.value

    def __repr__(self) -> str:
        return f"_SymIntOutputStub({self.value!r})"

    def __eq__(self, other: object) -> bool:
        raise NotImplementedError

    def __hash__(self) -> int:
        raise NotImplementedError


@dataclass(slots=True)
class _CacheKeyState:
    """
    State used while building our cache key.
    """

    # We track the SymNodes so when we get the output we can see if it exactly
    # matches one of the inputs so we can uncache it properly.
    sym_node_lookup: dict[int, int]  # id(SymNode) -> index

    # This is a list of all seen input sympy.Symbols. We use it when building
    # the cache entry to see if the output value has any symbols that we didn't
    # see on input. See _has_unrepresented_symbols().
    known_symbols: set[sympy.Symbol]

    # There are cases where we're asked to perform an op when we have no
    # ShapeEnv on the FakeTensorMode - but for SymNodes we MUST have a
    # ShapeEnv. So as we scan if we see a SymNode (with a ShapeEnv) we record it
    # here.
    shape_env: ShapeEnv | None

    def __init__(self, shape_env: ShapeEnv | None = None) -> None:
        self.sym_node_lookup = {}
        self.known_symbols = set()
        self.shape_env = shape_env

    def cache_on_shape_env(self) -> bool:
        """
        Returns true if the CacheKey needs to be cached on the ShapeEnv
        rather than the global cache.

        If our inputs contain a SymNode then we can't cache this operation on
        the global cache because the cached output will implicitly depend on
        guard values which might not be true on some other ShapeEnv. So unless
        we're also going to cache the guards we need to cache this operation on
        the ShapeEnv instead of globally.
        """
        return bool(self.sym_node_lookup)

    def convert_sym_int(self, result: list[object], arg: SymInt) -> None:
        node_id = id(arg.node)
        if node_id in self.sym_node_lookup:
            result.append(_InputBackref(self.sym_node_lookup[node_id]))
        else:
            self.sym_node_lookup[node_id] = len(result)
            self.known_symbols.update(arg.node.expr.free_symbols)
            if self.shape_env is None:
                self.shape_env = arg.node.shape_env
            result.append(_PySymInputStub(arg))

    def convert_output(self, arg: _MetadataIntLike) -> _MetadataIntLike:
        if isinstance(arg, SymInt):
            return _SymIntOutputStub(arg, self.sym_node_lookup.get(id(arg.node), None))
        else:
            return arg
<|endoftext|>from __future__ import annotations

from copy import deepcopy
from itertools import combinations
from typing import Any, NamedTuple, TYPE_CHECKING

import torch
from torch.fx.operator_schemas import _normalize_function_or_error
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map


if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch._ops import OpOverload


class Mutation(NamedTuple):
    op_name: str
    arg_name: str


class Aliasing(NamedTuple):
    op_name: str
    arg_name: str
    output_number: str


# Simplified naming for C++ classes
# pyrefly: ignore [missing-attribute]
SchemaArgument = torch._C._SchemaArgument
# pyrefly: ignore [missing-attribute]
SchemaArgType = torch._C._SchemaArgType
SchemaInfo = torch._C._SchemaInfo

# This TorchDispatchMode Subclass is used to verify op schemas
# This TorchDispatchMode Subclass currently:
#  - Records the called ops
#  - Checks for mutations on all inputs
#  - Checks for aliasing on all inputs


# move these 2 functions here to avoid numpy dependency in testing/_internal/common_utils.py


def is_iterable_of_tensors(iterable: Iterable[Any]) -> bool:
    # Tensor itself is iterable so we check this first
    if isinstance(iterable, torch.Tensor):
        return False
    try:
        # pyrefly: ignore[bad-argument-type]
        if len(iterable) == 0:
            return False
        for t in iter(iterable):
            if not isinstance(t, torch.Tensor):
                return False
    except TypeError:
        return False
    return True


def clone_inputs(args: Iterable[Any]) -> list[Any]:
    inputs: list[Any] = []

    for arg in args:
        if isinstance(arg, torch.Tensor):
            inputs.append(arg.detach().clone())
        elif is_iterable_of_tensors(arg):
            inputs.append([t.detach().clone() for t in arg])
        else:
            inputs.append(arg)

    return inputs


class SchemaCheckMode(TorchDispatchMode):
    def __init__(self) -> None:
        # Information recorded for testing purposes. For example:
        #  - incorrect schemas
        #  - overly conservative schemas
        self.ops: list[str] = []
        self.mutated: list[Mutation] = []
        self.aliasing: list[Aliasing] = []

    def reset_cache(self) -> None:
        self.ops.clear()
        self.mutated.clear()
        self.aliasing.clear()

    def display_ops(self) -> None:
        print(*self.ops, sep=",")

    def __torch_dispatch__(
        self,
        func: OpOverload,
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        def bitwise_equal(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
            if lhs.is_quantized:
                # TODO: This is only OK if can't have NaN quantized; idk if
                # this is actually true
                return torch.equal(lhs, rhs)
            else:
                return torch.allclose(lhs, rhs, equal_nan=True)

        def has_mutated(
            before: Any, after: Any, md: tuple[tuple[int, ...], int] | None
        ) -> bool:
            are_tensors = type(before) is torch.Tensor and type(after) is torch.Tensor
            if (
                are_tensors
                and before.layout != torch.sparse_csr
                and after.layout != torch.sparse_csr
            ):
                return md is not None and not (
                    before.size() == after.size()
                    and bitwise_equal(before, after)
                    and md[0] == after.stride()
                    and md[1] == after._typed_storage()._cdata
                )
            return False

        def has_aliased(lhs: Any, rhs: Any) -> bool:
            try:
                # pyrefly: ignore [missing-attribute]
                return torch._C._overlaps(lhs, rhs)
            except Exception as exception:
                if str(exception).startswith("Cannot inspect value of type "):
                    return False
                else:
                    raise exception

        def standardize_name(name: str) -> str:
            return name if name != "self" else "input"

        def unwrap(e: Any) -> Any:
            if isinstance(e, torch.Tensor) and type(e) is not torch.Tensor:
                try:
                    # pyrefly: ignore[missing-attribute]
                    return e.elem
                except AttributeError:
                    return e
            return e

        def parse_metadata(e: Any) -> tuple[tuple[int, ...], int] | None:
            if isinstance(e, torch.Tensor):
                if type(e) is not torch.Tensor:
                    try:
                        # pyrefly: ignore[missing-attribute]
                        current = e.elem
                        return (
                            deepcopy(current.stride()),
                            current._typed_storage()._cdata,
                        )
                    except AttributeError:
                        return None
                # Sparse CSR tensors do not have strides or storage
                elif e.layout != torch.sparse_csr:
                    return (deepcopy(e.stride()), e._typed_storage()._cdata)
            return None

        self.ops.append(func._schema.name)

        # Clone and process arguments and outputs
        pre_arguments = _normalize_function_or_error(
            func, args, kwargs, normalize_to_only_use_kwargs=True
        ).kwargs

        c_p_args = dict(zip(pre_arguments.keys(), clone_inputs(pre_arguments.values())))
        cloned_arguments = {
            name: tree_map(unwrap, c_p_args.get(name)) for name in c_p_args
        }
        cloned_metadata = {
            name: [
                parse_metadata(a) for a in pytree.tree_leaves(pre_arguments.get(name))
            ]
            for name in pre_arguments
        }

        out = func(*args, **kwargs)
        arguments = {
            name: tree_map(unwrap, pre_arguments.get(name)) for name in pre_arguments
        }
        tuple_out = out if isinstance(out, tuple) else (out,)
        tuple_out = tree_map(unwrap, tuple_out)

        schema_info = SchemaInfo(func._schema)
        # pyrefly: ignore [missing-attribute]
        schema_info.add_argument_values(pre_arguments)

        # Process arguments with outputs
        for i in range(len(func._schema.arguments)):
            arg = func._schema.arguments[i]
            name = standardize_name(arg.name)
            if arguments.get(name) is not None:
                before = cloned_arguments.get(name)
                md = cloned_metadata.get(name)
                after = arguments.get(name)
                for j in range(len(tuple_out)):
                    # aten::_unsafe_view is intended to have incorrect aliasing notation (hence unsafe)
                    unsafe_ops = ("aten::_unsafe_view", "aten::unsafe_split")
                    if (
                        has_aliased(tuple_out[j], after)
                        and func._schema.name not in unsafe_ops
                    ):
                        # pyrefly: ignore [missing-attribute]
                        if not schema_info.may_contain_alias(
                            SchemaArgument(SchemaArgType.output, j),
                            SchemaArgument(SchemaArgType.input, i),
                        ):
                            raise RuntimeError(
                                f"Argument {name} is not defined to alias output but was aliasing"
                            )
                        else:
                            self.aliasing.append(
                                Aliasing(func._schema.name, name, f"output_{j}")
                            )
                    if after is tuple_out[j] and isinstance(after, torch.Tensor):
                        # Only mutable ops e.g. (add_, add.out) are allowed to directly return inputs.
                        if not schema_info.is_mutable(
                            SchemaArgument(SchemaArgType.input, i)
                        ) and func not in [
                            torch.ops.aten.lift.default,
                            torch.ops.aten.lift_fresh.default,
                        ]:
                            raise RuntimeError(
                                f"""\
Dispatcher operators below autograd are not allowed to directly return inputs.
However, we found that `outputs[{str(j)}] is {name}"""
                            )
                if md is not None and any(
                    has_mutated(a, b, c)
                    for a, b, c in zip(
                        pytree.tree_leaves(before), pytree.tree_leaves(after), md
                    )
                ):
                    if not schema_info.is_mutable(
                        SchemaArgument(SchemaArgType.input, i)
                    ):
                        raise RuntimeError(
                            f"Argument {name} is not defined as mutable but was mutated"
                        )
                    else:
                        self.mutated.append(Mutation(func._schema.name, name))

        # Aliasing between outputs
        for i, j in combinations(range(len(func._schema.returns)), 2):
            if has_aliased(tuple_out[i], tuple_out[j]):
                # pyrefly: ignore [missing-attribute]
                if not schema_info.may_contain_alias(
                    SchemaArgument(SchemaArgType.output, i),
                    SchemaArgument(SchemaArgType.output, j),
                ):
                    raise RuntimeError(f"Outputs {i} and {j} alias unexpectedly")

        return out
<|endoftext|>from __future__ import annotations

from typing import Any, TYPE_CHECKING
from typing_extensions import Self

import torch
from torch import Tensor
from torch.autograd import Function


if TYPE_CHECKING:
    from torch._ops import OpOverload
    from torch._prims_common import DeviceLikeType
    from torch.autograd.function import FunctionCtx


class ComplexTensor(Tensor):
    """A class that decomposes all ops on complex Tensors into their real and imaginary parts."""

    _re: Tensor
    _im: Tensor

    def __new__(
        cls,
        real: Tensor,
        imag: Tensor,
        /,
    ) -> Self:
        """Initialize a ComplexTensor from its real and imaginary parts."""
        from ._ops.common import REAL_TO_COMPLEX

        # TODO (hameerabbasi): `torch.compile` sometimes fails here without making these
        # contiguous. Why?
        real = real.contiguous()
        imag = imag.contiguous()

        # TODO (hameerabbasi):
        # What should we do with dtype?
        # We could convert to the complex type (float32 -> complex64), but we
        # can't use that model for say `bfloat16` which does not have a
        # corresponding complex dtype.
        # If we want to support this complex rep using any float type (see
        # https://github.com/pytorch/pytorch/issues/95100)
        # We either need to:
        # 1) add the complex types for say `complexbf32`, knowing they can't really be used anywhere
        #    else.
        # 2) We use the real float dtype here, and it is up to the user to know
        #    that dtype=float<size> here really means complex<2xSize> with dtype
        #    matching that of re/im parts alone
        # I'm going with 1 for now, so that I can make gradcheck and some complex
        # ops work properly, but might want to discuss this in the RFP.
        dtype = REAL_TO_COMPLEX.get(real.dtype)
        if dtype is None:
            raise TypeError(
                "Unsupported dtype for constituent tensors. Supported dtypes are: "
                f"{set(REAL_TO_COMPLEX.keys())!r}."
            )
        storage_offset = real.storage_offset()
        strides = real.stride()
        layout = real.layout
        pin_memory = real.is_pinned()

        if real.shape != imag.shape:
            raise AssertionError(f"Expected imag shape {real.shape}, got {imag.shape}")
        if real.device != imag.device:
            raise AssertionError(
                f"Expected imag device {real.device}, got {imag.device}"
            )
        if real.dtype != imag.dtype:
            raise AssertionError(f"Expected imag dtype {real.dtype}, got {imag.dtype}")
        if real.is_pinned() != imag.is_pinned():
            raise AssertionError(
                f"Expected imag pinning {real.is_pinned()}, got {imag.is_pinned()}"
            )

        res = Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            real.shape,
            device=real.device,
            dtype=dtype,
            storage_offset=storage_offset,
            strides=strides,
            pin_memory=pin_memory,
            layout=layout,
            requires_grad=False,
        )
        res._re = real.detach()
        res._im = imag.detach()

        return res

    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__()

    @property
    def re(self) -> Tensor:
        return self._re

    @property
    def im(self) -> Tensor:
        return self._im

    @property
    def real(self) -> Tensor:  # type: ignore[bad-override]
        return self.re

    @real.setter
    def real(self, value: Tensor) -> None:
        self.re[...] = value

    @property
    def imag(self) -> Tensor:  # type: ignore[bad-override]
        return self.im

    @imag.setter
    def imag(self, value: Tensor) -> None:
        self.im[...] = value

    @classmethod
    def __torch_dispatch__(  # type: ignore[bad-override]
        cls,
        func: OpOverload,
        types: tuple[type, ...],
        # pyrefly: ignore [implicit-any]
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        from ._ops.common import lookup_complex

        kwargs = {} if kwargs is None else kwargs

        impl = lookup_complex(func, *args, **kwargs)
        if impl is None:
            return NotImplemented

        return impl(*args, **kwargs)

    @staticmethod
    def from_interleaved(t: Tensor) -> ComplexTensor:
        re = torch.real(t)
        im = torch.imag(t) if t.dtype.is_complex else torch.zeros_like(t)
        return Complex.apply(re, im)

    def as_interleaved(self) -> Tensor:
        return torch.complex(self.re, self.im)

    @staticmethod
    def __tensor_unflatten__(
        inner_tensors: dict[str, Tensor],
        meta: Any,
        outer_size: tuple[int, ...],
        outer_stride: tuple[int, ...],
    ) -> ComplexTensor:
        re, im = inner_tensors["_re"], inner_tensors["_im"]
        return ComplexTensor(re, im)

    def __tensor_flatten__(self) -> tuple[list[str], Any]:
        return ["_re", "_im"], None

    def __repr__(self, *, tensor_contents: object | None = None) -> str:
        return f"ComplexTensor({self._re!r}, {self._im!r})"

    def is_pinned(self, device: DeviceLikeType | None = None) -> bool:
        return self._re.is_pinned(device)


class Complex(Function):
    @staticmethod
    def forward(  # type: ignore[bad-override]
        ctx: FunctionCtx, real: Tensor, imag: Tensor
    ) -> ComplexTensor:
        return ComplexTensor(real, imag)

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: ComplexTensor) -> tuple[Tensor, Tensor]:  # type: ignore[bad-override]
        return grad_output.real, grad_output.imag
<|endoftext|>from __future__ import annotations

import warnings
from typing import Any, TYPE_CHECKING

import torch

from .._core import ComplexTensor
from .common import (
    _get_func_name,
    COMPLEX_TO_REAL,
    complex_to_real_dtype,
    is_complex,
    OpType,
    promote_tensors,
    register_binary_nonlinear,
    register_complex,
    register_error,
    register_force_test,
    register_simple,
    split_complex_arg,
    split_complex_tensor,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


aten = torch.ops.aten


def register_binary_linear(op: OpType) -> Callable[..., Any]:
    def impl_with_alpha(
        lhs: ComplexTensor,
        rhs: ComplexTensor,
        *args: Any,
        alpha: int | float | complex,
        **kwargs: Any,
    ) -> ComplexTensor:
        return op(lhs, aten.mul(rhs, alpha, *args, **kwargs), *args, **kwargs)

    def impl(
        lhs: ComplexTensor, rhs: ComplexTensor, *args: Any, **kwargs: Any
    ) -> ComplexTensor:
        alpha = kwargs.pop("alpha", None)
        if alpha is not None:
            return impl_with_alpha(lhs, rhs, *args, alpha=alpha, **kwargs)
        a_r, a_i = split_complex_arg(lhs)
        b_r, b_i = split_complex_arg(rhs)
        out_dt, (a_r, a_i, b_r, b_i) = promote_tensors(a_r, a_i, b_r, b_i)
        u = op(a_r, b_r, *args, **kwargs)
        v = op(a_i, b_i, *args, **kwargs)
        return ComplexTensor(u.to(out_dt), v.to(out_dt))

    return register_complex(op, impl)


def register_binary_linear_inplace(
    op: OpType, out_of_place_impl: Callable[..., Any]
) -> Callable[..., Any]:
    def impl(
        lhs: ComplexTensor, rhs: ComplexTensor, *args: Any, **kwargs: Any
    ) -> ComplexTensor:
        result = out_of_place_impl(lhs, rhs, *args, **kwargs)
        lhs.copy_(result)
        return lhs

    return register_complex(op, impl)


@register_complex(aten.real)
def real_impl(self: ComplexTensor) -> torch.Tensor:
    return self.re


@register_complex(aten.imag)
def imag_impl(self: ComplexTensor) -> torch.Tensor:
    return self.im


@register_complex(aten.is_pinned)
def is_pinned_impl(self: ComplexTensor, device: torch.device | None = None) -> bool:
    return self.is_pinned(device)


SIMPLE_OPS_LIST = [
    aten.slice,
    aten.flatten,
    aten.view,
    aten.diagonal,
    aten.expand,
    aten.unsqueeze,
    aten.unsqueeze_,
    aten.mean,
    aten.sum,
    aten.clone,
    aten.flip,
    aten.permute,
    aten.repeat,
    aten.index_select,
    aten.split,
    aten.split_with_sizes,
    aten.cumsum,
    aten.detach,
    aten.select,
    aten.squeeze,
    aten.zero_,
    aten.transpose,
    aten.t,
    aten.gather,
    aten.neg,
]

for simple_op in SIMPLE_OPS_LIST:
    globals()[_get_func_name(simple_op)] = register_simple(simple_op)

# TODO (hameerabbasi): Not being tested
SIMPLE_FORCE_TESTED_OPS = [
    aten.col2im,
    aten.alias,
    aten.lift_fresh,
    aten._unsafe_view,
    aten.index,
    aten.avg_pool2d,
    aten.avg_pool3d,
    aten.avg_pool2d_backward,
    aten.avg_pool3d_backward,
    aten.masked_scatter_backward,
    aten.select_backward,
    aten.slice_backward,
    aten.embedding,
]

for simple_op in SIMPLE_FORCE_TESTED_OPS:
    globals()[_get_func_name(simple_op)] = register_force_test(
        simple_op, register_simple(simple_op)
    )

del simple_op

# some binary ops which we can stamp out
mul_impl = register_binary_nonlinear(aten.mul)
mul__impl = register_binary_linear_inplace(aten.mul_, mul_impl)
mm_impl = register_binary_nonlinear(aten.mm)
dot_impl = register_binary_nonlinear(aten.dot)
bmm_impl = register_binary_nonlinear(aten.bmm)

# TODO (hameerabbasi): Not being tested
convolution_impl = register_force_test(
    aten.convolution, register_binary_nonlinear(aten.convolution)
)

slice_scatter_impl = register_force_test(
    aten.slice_scatter, register_binary_linear(aten.slice_scatter)
)
select_scatter_impl = register_force_test(
    aten.select_scatter, register_binary_linear(aten.select_scatter)
)

add_impl = register_binary_linear(aten.add)
add__impl = register_binary_linear_inplace(aten.add_, add_impl)
sub_impl = register_binary_linear(aten.sub)
sub__impl = register_binary_linear_inplace(aten.sub_, sub_impl)
copy_impl = register_binary_linear(aten.copy)
diagonal_scatter_impl = register_binary_linear(aten.diagonal_scatter)
fill_impl = register_binary_linear(aten.fill)
fill__impl = register_binary_linear_inplace(aten.fill_, fill_impl)


@register_complex(aten.rsub)
def rsub_impl(
    lhs: ComplexTensor, rhs: ComplexTensor, alpha: int | float | complex | None = None
) -> ComplexTensor:
    if alpha is None:
        return torch.sub(rhs, lhs)  # type: ignore[bad-return]
    return torch.sub(rhs, lhs, alpha=alpha)  # type: ignore[bad-return]


@register_complex(aten.div)
@register_complex(aten.true_divide)
def div_impl(
    lhs: ComplexTensor, rhs: ComplexTensor, *, rounding_mode: str | None = None
) -> ComplexTensor:
    if rounding_mode is not None:
        raise NotImplementedError(
            "`rounding_mode` other than `None` not implemented for`ComplexTensor`."
        )
    a_r, a_i = split_complex_arg(lhs)
    if not is_complex(rhs):
        return ComplexTensor(a_r / rhs, a_i / rhs)
    b_r, b_i = split_complex_arg(rhs)
    out_dt, (a_r, a_i, b_r, b_i) = promote_tensors(a_r, a_i, b_r, b_i)
    num_r = a_r * b_r + a_i * b_i
    num_i = a_i * b_r - a_r * b_i
    den = b_r * b_r + b_i * b_i
    return ComplexTensor(
        (num_r / den).to(out_dt),
        (num_i / den).to(out_dt),
    )


@register_complex(aten.reciprocal)
def reciprocal_impl(self: ComplexTensor) -> ComplexTensor:
    self_r, self_i = split_complex_tensor(self)
    out_dt, (self_r, self_i) = promote_tensors(self_r, self_i)
    den = self_r * self_r + self_i * self_i
    return ComplexTensor(
        aten.div(self_r, den).to(out_dt),
        aten.div(-self_i, den).to(out_dt),
    )


# reductions
@register_complex(aten.prod)
def prod_impl(self: ComplexTensor, *args: Any, **kwargs: Any) -> ComplexTensor:
    out_dt, (self,) = promote_tensors(self)
    dtype = kwargs.pop("dtype", out_dt)
    kwargs["dtype"] = complex_to_real_dtype(self.dtype)

    prod_r = torch.prod(torch.abs(self), *args, **kwargs)
    sum_phi = torch.sum(torch.angle(self), *args, **kwargs)
    u = prod_r * torch.cos(sum_phi)
    v = prod_r * torch.sin(sum_phi)
    return ComplexTensor(u, v).to(dtype)  # type: ignore[bad-return]


@register_complex(aten.complex)
def complex_impl(re: torch.Tensor, im: torch.Tensor) -> ComplexTensor:
    return ComplexTensor(re, im)


@register_complex(aten.pow)
def pow_impl(self: ComplexTensor, exponent: ComplexTensor) -> ComplexTensor:
    out_dt, (self, exponent) = promote_tensors(self, exponent)
    return torch.exp(exponent * torch.log(self)).to(out_dt)  # type: ignore[bad-return]


@register_complex(aten.cumprod)
def cumprod_impl(self: ComplexTensor, *args: Any, **kwargs: Any) -> ComplexTensor:
    dtype = kwargs.pop("dtype", self.dtype)
    kwargs["dtype"] = complex_to_real_dtype(dtype)

    prod_r = torch.cumprod(torch.abs(self), *args, **kwargs)
    sum_phi = torch.cumsum(torch.angle(self), *args, **kwargs)
    u = prod_r * torch.cos(sum_phi)
    v = prod_r * torch.sin(sum_phi)
    return ComplexTensor(u, v)


# unary funcs,
# most of these are simple or require some kind of identity
@register_complex(aten.abs)
def abs_impl(self: ComplexTensor) -> torch.Tensor:
    x, y = split_complex_tensor(self)
    out_dt, (x, y) = promote_tensors(x, y)
    result = torch.hypot(x, y)
    return result.to(out_dt)


@register_complex(aten.angle)
def angle_impl(self: ComplexTensor) -> torch.Tensor:
    x, y = split_complex_tensor(self)
    return torch.atan2(y, x)


@register_complex(aten.acos)
def acos_impl(self: ComplexTensor) -> ComplexTensor:
    _, y = split_complex_tensor(self)
    acosh_z = torch.acosh(self)
    if not isinstance(acosh_z, ComplexTensor):
        raise AssertionError(f"acosh_z must be a ComplexTensor, got {type(acosh_z)}")
    acosh_z_re, acosh_z_im = split_complex_tensor(acosh_z)
    sign_im = 2 * torch.signbit(y) - 1
    return ComplexTensor(torch.abs(acosh_z_im), sign_im * torch.abs(acosh_z_re))


@register_complex(aten.asin)
def asin_impl(self: ComplexTensor) -> ComplexTensor:
    x, y = split_complex_tensor(self)
    asinh_iz = torch.asinh(ComplexTensor(-y, x))
    if not isinstance(asinh_iz, ComplexTensor):
        raise AssertionError(f"asinh_iz must be a ComplexTensor, got {type(asinh_iz)}")
    asinh_iz_re, asinh_iz_im = split_complex_tensor(asinh_iz)
    return ComplexTensor(asinh_iz_im, -asinh_iz_re)


@register_complex(aten.atan)
def atan_impl(self: ComplexTensor) -> ComplexTensor:
    x, y = split_complex_tensor(self)
    tanh_iz = torch.atanh(ComplexTensor(-y, x))
    if not isinstance(tanh_iz, ComplexTensor):
        raise AssertionError(f"tanh_iz must be a ComplexTensor, got {type(tanh_iz)}")
    tanh_iz_re, tanh_iz_im = split_complex_tensor(tanh_iz)
    return ComplexTensor(tanh_iz_im, -tanh_iz_re)


@register_complex(aten.asinh)
def asinh_impl(self: ComplexTensor) -> ComplexTensor:
    out_dt, (self,) = promote_tensors(self)
    return torch.log(self + torch.sqrt(self * self + 1)).to(out_dt)  # type: ignore[bad-return]


@register_complex(aten.acosh)
def acosh_impl(self: ComplexTensor) -> ComplexTensor:
    out_dt, (self,) = promote_tensors(self)
    return torch.log(self + torch.sqrt(self * self - 1)).to(out_dt)  # type: ignore[bad-return]


@register_complex(aten.atanh)
def atanh_impl(self: ComplexTensor) -> ComplexTensor:
    x, y = split_complex_tensor(self)
    out_dt, (x, y) = promote_tensors(x, y)

    ret = 0.5 * (
        torch.log(ComplexTensor(1 + x, y)) - torch.log(ComplexTensor(1 - x, -y))
    )
    if not isinstance(ret, ComplexTensor):
        raise AssertionError(f"ret must be a ComplexTensor, got {type(ret)}")
    ret_re, ret_im = split_complex_tensor(ret)

    return ComplexTensor(ret_re.to(out_dt), ret_im.to(out_dt))


@register_complex(aten.cos)
def cos_impl(self: ComplexTensor) -> ComplexTensor:
    x, y = split_complex_tensor(self)
    return torch.cosh(ComplexTensor(-y, x))  # type: ignore[bad-return]


@register_complex(aten.cosh)
def cosh_impl(self: ComplexTensor) -> ComplexTensor:
    x, y = split_complex_tensor(self)
    out_dt, (x, y) = promote_tensors(x, y)
    u = torch.cosh(x) * torch.cos(y)
    v = torch.sinh(x) * torch.sin(y)
    return ComplexTensor(u.to(out_dt), v.to(out_dt))


@register_complex(aten.sin)
def sin_impl(self: ComplexTensor) -> ComplexTensor:
    x, y = split_complex_tensor(self)
    sinh_iz = torch.sinh(ComplexTensor(-y, x))
    if not isinstance(sinh_iz, ComplexTensor):
        raise AssertionError(f"sinh_iz must be a ComplexTensor, got {type(sinh_iz)}")
    sinh_iz_re, sinh_iz_im = split_complex_tensor(sinh_iz)
    return ComplexTensor(sinh_iz_im, -sinh_iz_re)


@register_complex(aten.sinh)
def sinh_impl(self: ComplexTensor) -> ComplexTensor:
    x, y = split_complex_tensor(self)
    out_dt, (x, y) = promote_tensors(x, y)
    u = torch.sinh(x) * torch.cos(y)
    v = torch.cosh(x) * torch.sin(y)
    return ComplexTensor(u.to(out_dt), v.to(out_dt))


@register_complex(aten.tan)
def tan_impl(self: ComplexTensor) -> ComplexTensor:
    x, y = split_complex_tensor(self)
    tanh_iz = torch.tanh(ComplexTensor(-y, x))
    if not isinstance(tanh_iz, ComplexTensor):
        raise AssertionError(f"tanh_iz must be a ComplexTensor, got {type(tanh_iz)}")
    tanh_iz_re, tanh_iz_im = split_complex_tensor(tanh_iz)
    return ComplexTensor(tanh_iz_im, -tanh_iz_re)


@register_complex(aten.tanh)
def tanh_impl(self: ComplexTensor) -> ComplexTensor:
    x, y = split_complex_tensor(self)
    out_dt, (x, y) = promote_tensors(x, y)

    _2x = 2 * x
    _2y = 2 * y
    _d = torch.cosh(_2x) + torch.cos(_2y)
    _2xsh = torch.sinh(_2x)

    out_re = _2xsh / _d
    out_im = torch.sin(_2y) / _d

    return ComplexTensor(out_re.to(out_dt), out_im.to(out_dt))


@register_complex(aten.exp)
def exp_impl(self: ComplexTensor) -> ComplexTensor:
    x, y = split_complex_tensor(self)
    out_dt, (x, y) = promote_tensors(x, y)
    ex = torch.exp(x)
    u = ex * torch.cos(y)
    v = ex * torch.sin(y)
    return ComplexTensor(u.to(out_dt), v.to(out_dt))


@register_complex(aten.expm1)
def expm1_impl(self: ComplexTensor) -> ComplexTensor:
    x, y = split_complex_tensor(self)
    out_dt, (x, y) = promote_tensors(x, y)
    # TODO (hameerabbasi): The two lines below may have numerical issues
    ex = torch.exp(x)
    u = ex * torch.cos(y) - 1
    v = ex * torch.sin(y)
    return ComplexTensor(u.to(out_dt), v.to(out_dt))


@register_complex(aten.log)
def log_impl(self: ComplexTensor) -> ComplexTensor:
    out_dt, (self,) = promote_tensors(self)
    re = torch.log(torch.abs(self))
    im = torch.angle(self)
    return ComplexTensor(re, im).to(out_dt)  # type: ignore[bad-return]


@register_complex(aten.log1p)
def log1p_impl(self: ComplexTensor) -> ComplexTensor:
    x, y = split_complex_tensor(self)
    # TODO (hameerabbasi): The line below may have numerical issues
    return torch.log(ComplexTensor(x + 1, y))  # type: ignore[bad-return]


@register_complex(aten.any)
def any_impl(self: ComplexTensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    x, y = split_complex_tensor(self)
    return torch.any(x, *args, **kwargs) | torch.any(y, *args, **kwargs)


@register_complex(aten.all)
def all_impl(self: ComplexTensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    x, y = split_complex_tensor(self)
    return torch.any(x, *args, **kwargs) & torch.any(y, *args, **kwargs)


@register_complex(aten.eq)
def eq_impl(
    self: ComplexTensor, rhs: ComplexTensor, *args: Any, **kwargs: Any
) -> torch.Tensor:
    a_r, a_i = split_complex_arg(self)
    b_r, b_i = split_complex_arg(rhs)
    return torch.eq(a_r, b_r, *args, **kwargs) & torch.eq(a_i, b_i, *args, **kwargs)


@register_complex(aten.ne)
def ne_impl(
    self: ComplexTensor, rhs: ComplexTensor, *args: Any, **kwargs: Any
) -> torch.Tensor:
    a_r, a_i = split_complex_arg(self)
    b_r, b_i = split_complex_arg(rhs)
    return torch.ne(a_r, b_r, *args, **kwargs) | torch.ne(a_i, b_i, *args, **kwargs)


@register_complex(aten.isnan)
def isnan_impl(self: ComplexTensor) -> torch.Tensor:
    re, im = split_complex_tensor(self)
    return torch.isnan(re) | torch.isnan(im)


@register_complex(aten.isinf)
def isinf_impl(self: ComplexTensor) -> torch.Tensor:
    re, im = split_complex_tensor(self)
    return torch.isinf(re) | torch.isinf(im)


@register_complex(aten.isfinite)
def isfinite_impl(self: ComplexTensor) -> torch.Tensor:
    re, im = split_complex_tensor(self)
    return torch.isfinite(re) & torch.isfinite(im)


@register_complex(aten.isclose)
def isclose_impl(
    self: ComplexTensor,
    rhs: ComplexTensor,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = False,
) -> torch.Tensor:
    abs_diff = torch.abs(self - rhs)
    abs_other = torch.abs(rhs)
    basic_condition = abs_diff <= (rtol * abs_other + atol)

    # This is the nontrivial part
    if equal_nan:
        a_r, a_i = split_complex_tensor(self)
        b_r, b_i = split_complex_arg(rhs)

        a_r_nan = torch.isnan(a_r)
        b_r_nan = torch.isnan(b_r)
        a_i_nan = torch.isnan(a_i)
        b_i_nan = torch.isnan(b_i)
        a_nan = a_r_nan | a_i_nan

        # This logical expression makes sure that the isnan of both the real and imaginary parts
        # matches (so 1 + nan*i doesn't equal nan + 1*i)
        equal_nan_condition = ((a_r_nan == b_r_nan) & (a_i_nan == b_i_nan)) & a_nan
        return basic_condition | equal_nan_condition

    return basic_condition


ERROR_OPS_LIST = [
    aten.lt,
    aten.le,
    aten.gt,
    aten.ge,
    aten.amin,
    aten.amax,
    aten.clamp,
    aten.ceil,
    aten.floor,
    aten.minimum,
    aten.maximum,
    aten.trunc,
    aten.sign,
    aten.argmax,
    aten.argmin,
    aten.sort,
    aten.topk,
    aten.round,
    aten.fmod,
]


ERROR_TYPES = {
    aten.minimum: RuntimeError,
    aten.maximum: RuntimeError,
    aten.argmax: RuntimeError,
    aten.argmin: RuntimeError,
    aten.sort: RuntimeError,
    aten.topk: RuntimeError,
}


for err_op in ERROR_OPS_LIST:
    globals()[_get_func_name(err_op)] = register_error(
        err_op, ERROR_TYPES.get(err_op, NotImplementedError)
    )

del err_op


@register_complex(aten.masked_scatter)
def masked_scatter_impl(
    self: ComplexTensor, mask: torch.Tensor, source: ComplexTensor
) -> ComplexTensor:
    self_r, self_i = split_complex_tensor(self)
    source_r, source_i = split_complex_arg(source)
    ret_r = torch.masked_scatter(self_r, mask, source_r)
    ret_i = torch.masked_scatter(self_i, mask, source_i)

    return ComplexTensor(ret_r, ret_i)


@register_complex(aten.where)
def where_impl(mask: torch.Tensor, x: ComplexTensor, y: ComplexTensor) -> ComplexTensor:
    x_r, x_i = split_complex_arg(x)
    y_r, y_i = split_complex_arg(y)

    ret_r = torch.where(mask, x_r, y_r)
    ret_i = torch.where(mask, x_i, y_i)

    return ComplexTensor(ret_r, ret_i)


@register_complex(aten.full_like)
def full_like_impl(
    input: ComplexTensor,
    fill_value: complex,
    *args: Any,
    dtype: torch.dtype | None = None,
    **kwargs: Any,
) -> torch.Tensor | ComplexTensor:
    # Note: Cannot be merged with the cases below due to the `fill_value` argument
    input_r, input_i = split_complex_tensor(input)
    if dtype is not None and dtype not in COMPLEX_TO_REAL:
        return torch.full_like(input_r, fill_value, *args, dtype=dtype, **kwargs)

    if dtype is not None:
        kwargs["dtype"] = COMPLEX_TO_REAL[dtype]

    fv_r, fv_i = split_complex_arg(fill_value)
    ret_r = torch.full_like(input_r, fv_r, *args, **kwargs)
    ret_i = torch.full_like(input_i, fv_i, *args, **kwargs)

    return ComplexTensor(ret_r, ret_i)


def register_like(op: OpType) -> Callable[..., Any]:
    def impl(
        self: ComplexTensor, *args: Any, dtype: torch.dtype | None = None, **kwargs: Any
    ) -> torch.Tensor | ComplexTensor:
        self_re, self_im = split_complex_tensor(self)

        if dtype is not None and dtype not in COMPLEX_TO_REAL:
            return op(self_re, *args, dtype=dtype, **kwargs)

        if dtype is not None:
            kwargs["dtype"] = COMPLEX_TO_REAL[dtype]

        ret_re = op(self_re, *args, **kwargs)
        ret_im = op(self_im, *args, **kwargs)

        return ComplexTensor(ret_re, ret_im)

    func_name = _get_func_name(op)
    impl.__name__ = func_name
    impl.__qualname__ = func_name

    return register_complex(op, impl)


LIKE_OPS_LIST = [
    aten.empty_like,
    aten.zeros_like,
    aten.randn_like,
    aten.new_zeros,
]

for like_op in LIKE_OPS_LIST:
    globals()[_get_func_name(like_op)] = register_like(like_op)

del like_op


@register_complex(aten.cat)
def cat_impl(tensors: Sequence[ComplexTensor], dim: int = 0) -> ComplexTensor:
    tensors_r = []
    tensors_i = []

    for t in tensors:
        t_r, t_i = split_complex_arg(t)
        tensors_r.append(t_r)
        tensors_i.append(t_i)

    ret_r = torch.cat(tensors_r, dim=dim)
    ret_i = torch.cat(tensors_i, dim=dim)

    return ComplexTensor(ret_r, ret_i)


@register_complex(aten.sgn)
def sgn_impl(self: ComplexTensor) -> ComplexTensor:
    self_r, self_i = split_complex_tensor(self)
    out_dt, (self_r, self_i) = promote_tensors(self_r, self_i)
    abs_self = torch.abs(ComplexTensor(self_r, self_i))
    mask = (self_r != 0) | (self_i != 0)
    masked_sgn = ComplexTensor(
        (self_r / abs_self).to(out_dt), (self_i / abs_self).to(out_dt)
    )
    return torch.where(mask, masked_sgn, 0)  # type: ignore[bad-return]


@register_complex(aten.sqrt)
def sqrt_impl(self: ComplexTensor) -> ComplexTensor:
    self_r, self_i = split_complex_tensor(self)
    out_dt, (self_r, self_i) = promote_tensors(self_r, self_i)
    self = ComplexTensor(self_r, self_i)
    self_abs_sqrt = torch.sqrt(torch.abs(self))
    self_half_angle = 0.5 * torch.angle(self)

    ret_r = self_abs_sqrt * torch.cos(self_half_angle)
    ret_i = self_abs_sqrt * torch.sin(self_half_angle)

    return ComplexTensor(ret_r.to(out_dt), ret_i.to(out_dt))


@register_complex(aten.rsqrt)
def rsqrt_impl(self: ComplexTensor) -> ComplexTensor:
    self_r, self_i = split_complex_tensor(self)
    out_dt, (self_r, self_i) = promote_tensors(self_r, self_i)
    self = ComplexTensor(self_r, self_i)
    self_abs_rsqrt = torch.rsqrt(torch.abs(self))
    self_neg_half_angle = -0.5 * torch.angle(self)

    ret_r = self_abs_rsqrt * torch.cos(self_neg_half_angle)
    ret_i = self_abs_rsqrt * torch.sin(self_neg_half_angle)

    return ComplexTensor(ret_r.to(out_dt), ret_i.to(out_dt))


@register_complex(aten.addmm)
def addmm_impl(
    input: ComplexTensor,
    mat1: ComplexTensor,
    mat2: ComplexTensor,
    out_dtype: torch.dtype | None = None,
    beta: int | float | complex = 1,
    alpha: int | float | complex = 1,
) -> ComplexTensor:
    ret = beta * input + alpha * torch.mm(mat1, mat2)
    if not isinstance(ret, ComplexTensor):
        raise AssertionError(f"ret must be a ComplexTensor, got {type(ret)}")
    ret_r, ret_i = split_complex_tensor(ret)
    if out_dtype is not None:
        out_dtype = COMPLEX_TO_REAL[out_dtype]
        ret_r, ret_i = ret_r.to(out_dtype), ret_i.to(out_dtype)
    return ComplexTensor(ret_r, ret_i)


def elemwise_nonzero(self: ComplexTensor) -> torch.Tensor:
    re, im = split_complex_tensor(self)
    return (re != 0) | (im != 0)


def register_nonzero_impl(op: OpType) -> Callable[..., Any]:
    def nonzero_impl(
        self: ComplexTensor, other: ComplexTensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        return op(elemwise_nonzero(self), elemwise_nonzero(other), *args, **kwargs)

    func_name = _get_func_name(op)
    nonzero_impl.__name__ = func_name
    nonzero_impl.__qualname__ = func_name

    return register_complex(op, nonzero_impl)


logical_and_impl = register_nonzero_impl(aten.logical_and)
logical_or_impl = register_nonzero_impl(aten.logical_or)
logical_xor_impl = register_nonzero_impl(aten.logical_xor)


@register_complex(aten.logical_not)
def logical_not_impl(self: ComplexTensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    return torch.logical_not(elemwise_nonzero(self), *args, **kwargs)


@register_complex(aten.view_as_real)
def view_as_real_impl(self: ComplexTensor) -> torch.Tensor:
    return torch.stack([self.re, self.im], dim=-1)


@register_complex(aten.view_as_complex)
def view_as_complex_impl(self: ComplexTensor) -> torch.Tensor:
    if len(self.shape) < 1 or self.shape[-1] != 2:
        raise RuntimeError("Tensor must have a last dimension of size 2")
    return ComplexTensor(self[..., 0], self[..., 1])


@register_complex(aten.linalg_vector_norm)
def linalg_vector_norm_impl(
    self: ComplexTensor, *args: Any, **kwargs: Any
) -> torch.Tensor:
    return torch.linalg.vector_norm(torch.abs(self), *args, **kwargs)


@register_force_test(aten.copy_)
def copy__impl(
    self: ComplexTensor | torch.Tensor,
    src: ComplexTensor | torch.Tensor,
    *args: Any,
    **kwargs: Any,
) -> ComplexTensor | torch.Tensor:
    if not self.dtype.is_complex:
        warnings.warn(
            "Casting complex values to real discards the imaginary part", UserWarning
        )
        src_re, src_im = split_complex_arg(src)
        return self.copy_(src_re)

    self_re, self_im = split_complex_arg(self)
    src_re, src_im = split_complex_arg(src)

    ret_re = self_re.copy_(src_re, *args, **kwargs)
    ret_im = self_im.copy_(src_im, *args, **kwargs)

    return ComplexTensor(ret_re, ret_im)


@register_complex(aten._local_scalar_dense)
def _local_scalar_dense_impl(self: ComplexTensor, *args: Any, **kwargs: Any) -> complex:
    x, y = split_complex_tensor(self)
    u = aten._local_scalar_dense(x, *args, **kwargs)
    v = aten._local_scalar_dense(y, *args, **kwargs)
    return complex(u, v)


@register_complex(aten.allclose)
def allclose_impl(
    input: torch.Tensor,
    other: torch.Tensor,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> bool:
    # pyrefly: ignore [bad-return]
    return torch.all(
        torch.isclose(input, other, rtol=rtol, atol=atol, equal_nan=equal_nan)
    ).item()  # type: ignore[bad-return]


@register_complex(aten.stack)
def stack_impl(self: list[ComplexTensor], *args: Any, **kwargs: Any) -> ComplexTensor:
    re_im_tuples = [split_complex_arg(self_i) for self_i in self]
    u = torch.stack([c[0] for c in re_im_tuples], *args, **kwargs)
    v = torch.stack([c[1] for c in re_im_tuples], *args, **kwargs)
    return ComplexTensor(u, v)


# TODO (hameerabbasi): Not being tested
@register_complex(aten._conj_physical)
@register_complex(aten.conj_physical)
def conj_physical_impl(self: ComplexTensor) -> ComplexTensor:
    re, im = split_complex_tensor(self)
    return ComplexTensor(re, -im)


@register_complex(aten._conj)
def _conj_impl(self: ComplexTensor) -> ComplexTensor:
    # See "NOTE conj/neg (hameerabbasi)"
    return ComplexTensor(aten.alias(self.re), -self.im)


@register_complex(aten._neg_view)
def _neg_view_impl(self: ComplexTensor) -> ComplexTensor:
    # See "NOTE conj/neg (hameerabbasi)"
    return ComplexTensor(-self.re, -self.im)


@register_complex(aten.resolve_conj)
def resolve_conj_impl(self: ComplexTensor) -> ComplexTensor:
    # See "NOTE conj/neg (hameerabbasi)"
    return self


@register_complex(aten.resolve_neg)
def resolve_neg_impl(self: ComplexTensor) -> ComplexTensor:
    # See "NOTE conj/neg (hameerabbasi)"
    return self


@register_complex(aten.index_add)
def index_add_impl(
    self: ComplexTensor,
    dim: int,
    index: torch.Tensor,
    source: ComplexTensor,
    **kwargs: Any,
) -> ComplexTensor:
    alpha = kwargs.pop("alpha", None)
    if alpha is not None:
        source = source * alpha
    self_re, self_im = split_complex_arg(self)
    source_re, source_im = split_complex_arg(source)

    ret_re = self_re.index_add(dim, index, source_re)
    ret_im = self_im.index_add(dim, index, source_im)

    return ComplexTensor(ret_re, ret_im)


# TODO (hameerabbasi): Not being tested
@register_complex(aten.index_add_)
def index_add__impl(
    self: ComplexTensor,
    dim: int,
    index: torch.Tensor,
    source: ComplexTensor,
    **kwargs: Any,
) -> ComplexTensor:
    alpha = kwargs.pop("alpha", None)
    if alpha is not None:
        source = source * alpha

    self_re, self_im = split_complex_arg(self)
    source_re, source_im = split_complex_arg(source)

    ret_re = self_re.index_add_(dim, index, source_re)
    ret_im = self_im.index_add_(dim, index, source_im)

    return ComplexTensor(ret_re, ret_im)


@register_complex(aten.masked_fill)
def masked_fill_impl(
    self: ComplexTensor, mask: torch.Tensor, value: complex
) -> ComplexTensor:
    self_re, self_im = split_complex_arg(self)
    value_re, value_im = split_complex_arg(value)

    ret_re = self_re.masked_fill(mask, value_re)
    ret_im = self_im.masked_fill(mask, value_im)

    return ComplexTensor(ret_re, ret_im)


# TODO (hameerabbasi): Not being tested
@register_complex(aten.masked_fill_)
def masked_fill__impl(
    self: ComplexTensor, mask: torch.Tensor, value: complex
) -> ComplexTensor:
    self_re, self_im = split_complex_arg(self)
    value_re, value_im = split_complex_arg(value)

    ret_re = self_re.masked_fill_(mask, value_re)
    ret_im = self_im.masked_fill_(mask, value_im)

    return ComplexTensor(ret_re, ret_im)


@register_complex(aten.constant_pad_nd)
def constant_pad_nd_impl(
    self: ComplexTensor, pad: Sequence[int], value: complex | None = None
) -> ComplexTensor:
    self_re, self_im = split_complex_tensor(self)
    if value is None:
        ret_re = aten.constant_pad_nd(self_re, pad)
        ret_im = aten.constant_pad_nd(self_im, pad)
    else:
        value_re, value_im = split_complex_arg(value)
        ret_re = aten.constant_pad_nd(self_re, pad, value_re)
        ret_im = aten.constant_pad_nd(self_im, pad, value_im)

    return ComplexTensor(ret_re, ret_im)


@register_complex(aten.var)
def var_impl(self: ComplexTensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    self_re, self_im = split_complex_tensor(self)
    return torch.var(self_re, *args, **kwargs) + torch.var(self_im, *args, **kwargs)


@register_complex(aten.scatter_add)
def scatter_add_impl(
    self: ComplexTensor, dim: int, index: torch.Tensor, src: ComplexTensor
) -> ComplexTensor:
    self_re, self_im = split_complex_arg(self)
    src_re, src_im = split_complex_arg(src)

    ret_re = torch.scatter_add(self_re, dim, index, src_re)
    ret_im = torch.scatter_add(self_im, dim, index, src_im)

    return ComplexTensor(ret_re, ret_im)


@register_complex(aten.scatter_add_)
def scatter_add__impl(
    self: ComplexTensor, dim: int, index: torch.Tensor, src: ComplexTensor
) -> ComplexTensor:
    self_re, self_im = split_complex_arg(self)
    src_re, src_im = split_complex_arg(src)

    out_re = self_re.scatter_add_(dim, index, src_re)
    out_im = self_im.scatter_add_(dim, index, src_im)

    return ComplexTensor(out_re, out_im)


@register_complex(aten.index_put_)
def index_put__impl(
    self: ComplexTensor,
    indices: tuple[torch.Tensor, ...],
    values: ComplexTensor,
    accumulate: bool = False,
) -> ComplexTensor:
    self_re, self_im = split_complex_arg(self)
    values_re, values_im = split_complex_arg(values)

    out_re = self_re.index_put_(indices, values_re, accumulate=accumulate)
    out_im = self_im.index_put_(indices, values_im, accumulate=accumulate)

    return ComplexTensor(out_re, out_im)


@register_complex(aten.tanh_backward)
def tanh_backward(out_grad: ComplexTensor, y: ComplexTensor) -> ComplexTensor:
    # pyrefly: ignore[bad-return]
    return out_grad * (1.0 - y * y).conj_physical()


@register_complex(aten.diagonal_backward)
def diagonal_backward(
    grad_output: torch.Tensor, input_sizes: list[int], offset: int, dim1: int, dim2: int
) -> torch.Tensor:
    grad_input = grad_output.new_zeros(input_sizes)
    return torch.diagonal_scatter(grad_input, grad_output, offset, dim1, dim2)


def _dt_to_real(dt: torch.dtype | Any) -> torch.dtype | Any:
    if not isinstance(dt, torch.dtype):
        return dt

    return COMPLEX_TO_REAL[dt]


@register_complex(aten.to.dtype_layout)
def to_dtype_layout(
    self: ComplexTensor,
    *,
    dtype: torch.dtype | None,
    device: torch.Device | None = None,
    copy: bool = False,
    _op: OpType = aten.to,
    **kwargs: Any,
) -> ComplexTensor | torch.Tensor:
    if device == self.device:
        device = None

    if dtype == self.dtype:
        dtype = None

    if device is None and dtype is None and not copy:
        return self
    re, im = split_complex_arg(self)
    if dtype is not None and dtype not in COMPLEX_TO_REAL:
        return _op(re, device=device, dtype=dtype, **kwargs)

    dtype = COMPLEX_TO_REAL.get(dtype)  # type: ignore[no-matching-overload]
    out_re = _op(re, device=device, dtype=dtype, **kwargs)
    out_im = _op(im, device=device, dtype=dtype, **kwargs)

    return ComplexTensor(out_re, out_im)


@register_complex(aten.to.device)
def to_device(
    self: ComplexTensor,
    device: torch.Device | None,
    dtype: torch.dtype | None,
    non_blocking: bool = False,
    copy: bool = False,
    memory_format: torch.memory_format | None = torch.preserve_format,
) -> ComplexTensor | torch.Tensor:
    return to_dtype_layout(
        self,
        device=device,
        dtype=dtype,
        non_blocking=non_blocking,
        copy=copy,
        memory_format=memory_format,
    )


@register_complex(aten.to.dtype)
def to_dtype(
    self: ComplexTensor,
    dtype: torch.dtype | None,
    non_blocking: bool = False,
    copy: bool = False,
    memory_format: torch.memory_format | None = torch.preserve_format,
) -> ComplexTensor | torch.Tensor:
    return to_dtype_layout(
        self,
        dtype=dtype,
        non_blocking=non_blocking,
        copy=copy,
        memory_format=memory_format,
    )


@register_complex(aten.to.other)
def to_other(
    self: ComplexTensor,
    other: torch.Tensor,
    non_blocking: bool = False,
    copy: bool = False,
    memory_format: torch.memory_format | None = torch.preserve_format,
) -> ComplexTensor | torch.Tensor:
    return to_dtype_layout(
        self,
        dtype=other.dtype,
        device=other.device,
        non_blocking=non_blocking,
        copy=copy,
        memory_format=memory_format,
    )


@register_complex(aten._to_copy.default)
def _to_copy_default(
    self: ComplexTensor,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.Device | None = None,
    pin_memory: bool = False,
    non_blocking: bool = False,
    memory_format: torch.memory_format | None = None,
) -> ComplexTensor | torch.Tensor:
    return to_dtype_layout(
        self,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        non_blocking=non_blocking,
        memory_format=memory_format,
        _op=aten._to_copy,
    )


@register_complex(aten._to_copy.out)
def _to_copy_out(
    self: ComplexTensor,
    *,
    non_blocking: bool = False,
    memory_format: torch.memory_format | None = None,
    out: torch.Tensor | ComplexTensor,
) -> ComplexTensor | torch.Tensor:
    temp = _to_copy_default(
        self,
        dtype=out.dtype,
        layout=out.layout,
        device=out.device,
        pin_memory=out.is_pinned(),
        non_blocking=non_blocking,
    )
    out.copy_(temp)
    return out
<|endoftext|>import torch

from .._core import ComplexTensor
from .common import (
    complex_to_real_dtype,
    register_complex,
    register_force_test,
    split_complex_tensor,
)


prims = torch.ops.prims
aten = torch.ops.aten


# TODO (hameerabbasi): Not being tested
@register_force_test(prims.convert_element_type)
def convert_element_type_impl(x: ComplexTensor, dtype: torch.dtype) -> ComplexTensor:
    dtype = complex_to_real_dtype(dtype)
    u, v = split_complex_tensor(x)
    u_out = prims.convert_element_type(u, dtype)
    v_out = prims.convert_element_type(v, dtype)

    return ComplexTensor(u_out, v_out)


@register_complex(prims.conj_physical)
def conj_physical_impl(self: ComplexTensor) -> ComplexTensor:
    return aten._conj_physical(self)


@register_complex(prims.conj)
def conj_impl(self: ComplexTensor) -> ComplexTensor:
    return aten._conj(self)
<|endoftext|>from collections.abc import Callable
from typing import Any, overload, TypeAlias
from typing_extensions import Never, ParamSpec, TypeIs, TypeVar

import torch
from torch import Tensor
from torch._decomp import get_decompositions
from torch._ops import OpOverload, OpOverloadPacket
from torch._refs import is_complex as _is_complex
from torch.types import Number
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

from .._core import ComplexTensor


_P = ParamSpec("_P")
_R = TypeVar("_R")

OpType: TypeAlias = OpOverloadPacket | OpOverload

# pyrefly: ignore [implicit-any]
TableType: TypeAlias = dict[OpType, Callable]

# Mapping from ops to implementations
COMPLEX_OPS_TABLE: TableType = {}

# Ops that expand complex multiply into four real ops (different accumulation order
# than native complex BLAS).
BINARY_NONLINEAR_OPS: set[OpType] = set()

COMPLEX_TO_REAL = {
    torch.complex128: torch.float64,
    torch.complex64: torch.float32,
    torch.complex32: torch.float16,
}

REAL_TO_COMPLEX = {v: k for k, v in COMPLEX_TO_REAL.items()}

# Used to promote dtypes in `promote_real_cpu_tensors`
PROMOTE_TYPES = {
    torch.float16: torch.float32,
    torch.bfloat16: torch.float32,
    torch.complex32: torch.complex64,
}


def is_complex_tensor(obj: Any, /) -> TypeIs[ComplexTensor]:
    r"""Returns True if the input is a ComplexTensor, else False

    Args:
        a: any input

    Examples:

        >>> # xdoctest: +SKIP
        >>> from torch.complex import ComplexTensor
        >>> data = torch.zeros((3, 2), dtype=torch.complex64)
        >>> ct = ComplexTensor.from_interleaved(data)
        >>> is_complex_tensor(ct)
        True
    """
    return isinstance(obj, ComplexTensor)


@overload
def promote_tensors(
    *tensors: ComplexTensor,
) -> tuple[torch.dtype, tuple[ComplexTensor, ...]]: ...


@overload
def promote_tensors(
    *tensors: Tensor,
) -> tuple[torch.dtype, tuple[Tensor, ...]]: ...


def promote_tensors(
    *tensors: Tensor | ComplexTensor,
) -> tuple[torch.dtype, tuple[Tensor | ComplexTensor, ...]]:
    """
    Promotes all tensors to a common dtype.
    Additionally promotes CPU tensors to at least `float32`.
    """
    tensor = next(t for t in tensors if isinstance(t, Tensor))
    out_dt = tensor.dtype
    for t in tensors:
        if isinstance(t, Tensor):
            out_dt = torch.promote_types(out_dt, t.dtype)

    prom_dt = PROMOTE_TYPES.get(out_dt, out_dt)
    return out_dt, tuple(
        t.to(prom_dt) if isinstance(t, Tensor) else torch.asarray(t, dtype=prom_dt)
        for t in tensors
    )


def register_complex(
    op: OpType,
    func_impl: Callable[..., Any] | None = None,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]] | Callable[..., Any]:
    """Decorator to register an implementation for some ops in some dispatch tables"""

    def inner(func: Callable[_P, _R]) -> Callable[_P, _R]:
        if COMPLEX_OPS_TABLE.get(op, func) is not func:
            raise RuntimeError(f"Attempted to register multiple functions for {op}")
        COMPLEX_OPS_TABLE[op] = func
        return func

    if func_impl is None:
        return inner

    return inner(func_impl)


FORCE_TEST_LIST: list[OpType] = []


def register_force_test(
    op: OpType, func_impl: Callable[..., Any] | None = None
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]] | Callable[..., Any]:
    """Will attempt to test these ops even if they err on "normal" inputs"""
    FORCE_TEST_LIST.append(op)
    return register_complex(op, func_impl)


DECOMPOSITIONS = get_decompositions(list(torch.ops.aten))  # type: ignore[no-matching-overload]


def lookup_complex(
    func: OpOverload, *args: Any, **kwargs: Any
) -> Callable[..., Any] | None:
    """
    Lookup an impl from the table.

    Try the particular overload first, then the overload packet.

    If nothing is found, try the decompositions with both.
    """
    return COMPLEX_OPS_TABLE.get(
        func,
        COMPLEX_OPS_TABLE.get(
            func.overloadpacket,
            DECOMPOSITIONS.get(func, DECOMPOSITIONS.get(func.overloadpacket)),
        ),
    )


def is_complex(x: Any, /) -> bool:
    """Utility to detect if a given object is (known) to be complex."""
    return (isinstance(x, Tensor) and _is_complex(x)) or isinstance(x, complex)


@overload
def split_complex_arg(
    arg: Tensor | ComplexTensor,
) -> tuple[Tensor, Tensor]: ...


@overload
def split_complex_arg(
    arg: complex | Number,
) -> tuple[Number, Number]: ...


def split_complex_arg(
    arg: Tensor | ComplexTensor | complex | Number,
) -> tuple[Tensor, Tensor] | tuple[Number, Number]:
    """
    Split a complex argument into a real/imaginary component.

    If real, use zero for the imaginary part.
    """
    if isinstance(arg, ComplexTensor):
        return split_complex_tensor(arg)
    if isinstance(arg, Tensor):
        if is_complex(arg):
            return arg.real, arg.imag
        return arg, torch.zeros_like(arg)
    # TODO (hameerabbasi): Should there be a `torch.SymComplex`?
    if isinstance(arg, complex):
        return arg.real, arg.imag
    if isinstance(arg, float | torch.SymFloat):
        return arg, 0.0
    if isinstance(arg, int | torch.SymInt):
        return arg, 0
    if isinstance(arg, bool | torch.SymBool):
        return arg, False
    raise TypeError(f"Expected tensor or number got, {type(arg)}")


def split_complex_tensor(complex_tensor: ComplexTensor) -> tuple[Tensor, Tensor]:
    """Split a ComplexTensor into its real and imaginary parts."""
    return torch.resolve_neg(complex_tensor.re), torch.resolve_neg(complex_tensor.im)


def complex_to_real_dtype(dtype: torch.dtype) -> torch.dtype:
    """Convert a complex dtype to the dtype of its real part. Return other dtypes as-is."""
    return COMPLEX_TO_REAL.get(dtype, dtype)


def _get_op_name(op: OpType) -> str:
    """Get the op name from the op."""
    if isinstance(op, OpOverload):
        op = op.overloadpacket
    return str(op).split(".", 1)[1]


def is_binary_nonlinear_op(op: OpType) -> bool:
    if isinstance(op, OpOverload):
        op = op.overloadpacket
    return op in BINARY_NONLINEAR_OPS


def _get_func_name(op: OpType) -> str:
    """Get the name of the implementation function from the op."""
    return f"{_get_op_name(op)}_impl"


def register_error(
    op: OpType, exc_type: type[Exception] = NotImplementedError
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]] | Callable[..., Any]:
    msg = f"`aten.{_get_op_name(op)}` not implemented for `{ComplexTensor.__name__}`."

    def ordered_impl(*args: Any, **kwargs: Any) -> Never:
        raise exc_type(msg)

    func_name = _get_func_name(op)
    ordered_impl.__name__ = func_name
    ordered_impl.__qualname__ = func_name

    return register_force_test(op, ordered_impl)


def register_binary_nonlinear(
    op: OpType,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]] | Callable[..., Any]:
    """Register a "multiplication-style" op, e.g. aten.mul, aten.mm, ..."""
    BINARY_NONLINEAR_OPS.add(op)

    def impl(
        a: ComplexTensor, b: ComplexTensor, *args: Any, **kwargs: Any
    ) -> ComplexTensor:
        out_dt, (a, b) = promote_tensors(a, b)
        a_r, a_i = split_complex_arg(a)
        b_r, b_i = split_complex_arg(b)
        real = op(a_r, b_r, *args, **kwargs) - op(a_i, b_i, *args, **kwargs)
        imag = op(a_r, b_i, *args, **kwargs) + op(a_i, b_r, *args, **kwargs)
        return ComplexTensor(real, imag).to(out_dt)  # type: ignore[bad-return]

    func_name = _get_func_name(op)
    impl.__name__ = func_name
    impl.__qualname__ = func_name

    return register_complex(op, impl)


def register_simple(
    op: OpType,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]] | Callable[..., Any]:
    """Register an op which can be applied independently to the real and complex parts to get the result."""

    def impl(
        self: ComplexTensor, *args: Any, dtype: torch.dtype | None = None, **kwargs: Any
    ) -> ComplexTensor:
        x, y = split_complex_tensor(self)
        if dtype is not None and dtype not in COMPLEX_TO_REAL:
            raise RuntimeError(
                "Non-complex `dtype` specified, please write custom impl."
            )

        if dtype in COMPLEX_TO_REAL:
            if dtype is None:
                raise AssertionError("dtype must not be None when in COMPLEX_TO_REAL")
            kwargs["dtype"] = COMPLEX_TO_REAL[dtype]

        u = op(x, *args, **kwargs)
        v = op(y, *args, **kwargs)

        u_flat, u_spec = tree_flatten(u)
        v_flat, v_spec = tree_flatten(v)
        if u_spec != v_spec:
            raise AssertionError(f"Tree specs must match: {u_spec} != {v_spec}")
        out_flat = [
            ComplexTensor(ui, vi) for ui, vi in zip(u_flat, v_flat, strict=False)
        ]
        return tree_unflatten(out_flat, u_spec)

    func_name = _get_func_name(op)
    impl.__name__ = func_name
    impl.__qualname__ = func_name

    return register_complex(op, impl)


def _as_complex_tensor(arg: Tensor | Any) -> Tensor | ComplexTensor | Any:
    """Convert a Tensor with complex dtypes to a ComplexTensor. Pass along other args as-is."""
    if (
        not isinstance(arg, ComplexTensor)
        and isinstance(arg, Tensor)
        and arg.dtype in COMPLEX_TO_REAL
    ):
        return ComplexTensor.from_interleaved(arg)
    return arg


def _as_interleaved(arg: ComplexTensor | Any) -> Tensor | Any:
    """Convert a ComplexTensor to a Tensor with a complex dtype. Pass other arguments as-is."""
    if isinstance(arg, ComplexTensor):
        return arg.as_interleaved()
    return arg


class WrapComplexMode(TorchDispatchMode):
    def __torch_dispatch__(
        self,
        func: OpOverload,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if kwargs is None:
            kwargs = {}
        args = tree_map(_as_complex_tensor, args)
        kwargs = tree_map(_as_complex_tensor, kwargs)
        return tree_map(_as_complex_tensor, func(*args, **kwargs))


class ComplexTensorMode(TorchDispatchMode):
    _compile: bool

    """ A TorchDispatchMode to replace any Tensor that has a complex dtype with a ComplexTensor for the computation. """

    def __init__(self, _dispatch_key: Any = None, *, _compile: bool = False) -> None:
        """Initialize a ComplexTensorMode.

        Args:
            _dispatch_key: passed on to TorchDispatchMode
            _compile: Compile the op before the computation
        """
        super().__init__(_dispatch_key)
        self._compile = _compile

    def __torch_dispatch__(
        self,
        func: OpOverload,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if kwargs is None:
            kwargs = {}

        # TODO (hameerabbasi): Test perf with `_compile` set to `True`
        if self._compile:
            func = torch.compile(func)  # type: ignore[bad-assignment]

        args = tree_map(_as_complex_tensor, args)
        kwargs = tree_map(_as_complex_tensor, kwargs)

        return tree_map(_as_interleaved, func(*args, **kwargs))
<|endoftext|># mypy: allow-untyped-defs
import torch


# Pack pairs of int4 values into int8, in row major order; first int4
# value goes into lower order bits, and second int4 value into higher
# order bits of resulting int8 value.
def pack_int4_to_int8(weight):
    if weight.dim() != 2:
        raise AssertionError(f"weight must be 2D, got {weight.dim()}D")
    if weight.shape[1] % 2 != 0:
        raise AssertionError(f"weight.shape[1] must be even, got {weight.shape[1]}")
    if weight.dtype != torch.int8:
        raise AssertionError(f"weight.dtype must be int8, got {weight.dtype}")
    return ((weight[:, 1::2] & 0xF) << 4) | (weight[:, 0::2] & 0xF)


# Unpack quadruples of bits in int8 values into int4 values, in row
# major order; lower 4 bits go into first int4 value goes, and upper 4
# bits go into second int4 value.
def unpack_int8_to_int4(weight):
    if weight.dim() != 2:
        raise AssertionError(f"weight must be 2D, got {weight.dim()}D")
    if weight.dtype != torch.int8:
        raise AssertionError(f"weight.dtype must be int8, got {weight.dtype}")
    return torch.stack((weight & 0xF, (weight >> 4) & 0xF), dim=2).view(
        weight.shape[0], 2 * weight.shape[1]
    )


# Transpose the weight matrix, and then reorder its elements according
# to underlying requirements of CUTLASS library, so that it could be
# used for CUTLASS-based mixed datatypes linear operation.
def quantized_weight_reorder_for_mixed_dtypes_linear_cutlass(
    weight, dtypeq, transpose=False
):
    if weight.dim() != 2:
        raise AssertionError(f"weight must be 2D, got {weight.dim()}D")
    if weight.dtype != torch.int8:
        raise AssertionError(f"weight.dtype must be int8, got {weight.dtype}")
    if dtypeq != torch.int8 and dtypeq != torch.quint4x2:
        raise AssertionError(f"dtypeq must be int8 or quint4x2, got {dtypeq}")
    if weight.device.type != "cuda":
        raise AssertionError(f"weight must be on CUDA, got {weight.device.type}")

    device = weight.device

    # subbyte_transpose
    if not transpose:
        if dtypeq == torch.int8:
            outp = weight.T
        elif dtypeq == torch.quint4x2:
            outp = pack_int4_to_int8(unpack_int8_to_int4(weight.view(torch.int8)).T)
    else:
        outp = weight

    ncols, nrows = outp.shape  # type: ignore[possibly-undefined]
    divisor = 32 if dtypeq == torch.quint4x2 else 64
    if nrows % divisor != 0:
        raise AssertionError(f"nrows must be divisible by {divisor}, got {nrows}")
    if ncols % 64 != 0:
        raise AssertionError(f"ncols must be divisible by 64, got {ncols}")

    # permute_B_rows_for_mixed_gemm
    # (permute cols actually, as transpose is applied first here)
    if dtypeq == torch.quint4x2:
        cols_permuted = (
            torch.tensor(
                [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15],
                device=device,
            )
            + (torch.arange(0, nrows // 16, device=device).reshape(-1, 1) * 16).expand(
                nrows // 16, 16
            )
        ).view(-1)
    else:
        cols_permuted = (
            torch.tensor(
                [0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15],
                device=device,
            )
            + (torch.arange(0, nrows // 16, device=device).reshape(-1, 1) * 16).expand(
                nrows // 16, 16
            )
        ).view(-1)
    # pyrefly: ignore [unbound-name]
    outp = outp.index_copy(1, cols_permuted, outp)

    # interleave_column_major_tensor
    magic0 = 4 if dtypeq == torch.quint4x2 else 2
    magic1 = 32 // magic0

    tmp0 = (
        (torch.arange(0, ncols // magic0, device=device) * (nrows // 4 * magic0))
        .view(-1, 1)
        .repeat(1, nrows // 4 * magic0)
        .view(-1)
    )
    tmp1 = (
        (torch.arange(0, nrows // 4 // magic1, device=device) * (magic0 * magic1))
        .view(-1, 1)
        .repeat(1, magic1)
        .view(-1)
        .repeat(ncols)
    )
    tmp2 = (
        (torch.arange(0, magic0, device=device) * magic1)
        .view(-1, 1)
        .repeat(1, nrows // 4)
        .view(-1)
        .repeat(ncols // magic0)
    )
    tmp3 = torch.arange(0, magic1, device=device).repeat(nrows // 4 * ncols // magic1)

    outp_offsets = tmp0 + tmp1 + tmp2 + tmp3

    tmp = outp.view(-1).view(torch.int32)
    outp = torch.zeros_like(tmp)
    outp.scatter_(0, outp_offsets, tmp)
    outp = outp.view(weight.dtype)

    # add_bias_and_interleave_quantized_tensor_inplace
    tmp = outp.view(-1)

    outp = torch.empty_like(tmp)
    if dtypeq == torch.int8:
        tmp = (tmp.to(torch.int) + 128).to(tmp.dtype)
        outp[0::4] = tmp[0::4]
        outp[1::4] = tmp[2::4]
        outp[2::4] = tmp[1::4]
        outp[3::4] = tmp[3::4]
    elif dtypeq == torch.quint4x2:
        tmp0 = ((tmp & 0xF) + 8) & 0xF
        tmp0 = (tmp0[1::2] << 4) | tmp0[0::2]
        tmp1 = (((tmp >> 4) & 0xF) + 8) & 0xF
        tmp1 = (tmp1[1::2] << 4) | tmp1[0::2]
        outp[0::4] = tmp0[0::2]
        outp[1::4] = tmp0[1::2]
        outp[2::4] = tmp1[0::2]
        outp[3::4] = tmp1[1::2]

    if dtypeq == torch.quint4x2:
        nrows *= 2
        ncols //= 2

    return outp.view(nrows, ncols).view(torch.uint8)
<|endoftext|># mypy: allow-untyped-defs
from .fake_quantize import *  # noqa: F403
from .fuse_modules import fuse_modules
from .fuser_method_mappings import *  # noqa: F403
from .observer import *  # noqa: F403
from .qconfig import *  # noqa: F403
from .quant_type import *  # noqa: F403
from .quantization_mappings import *  # noqa: F403
from .quantize import *  # noqa: F403
from .quantize_jit import *  # noqa: F403
from .stubs import *  # noqa: F403


def default_eval_fn(model, calib_data):
    r"""
    Default evaluation function takes a torch.utils.data.Dataset or a list of
    input Tensors and run the model on the dataset
    """
    for data, _target in calib_data:
        model(data)


__all__ = [
    "QuantWrapper",
    "QuantStub",
    "DeQuantStub",
    # Top level API for eager mode quantization
    "quantize",
    "quantize_dynamic",
    "quantize_qat",
    "prepare",
    "convert",
    "prepare_qat",
    # Top level API for graph mode quantization on TorchScript
    "quantize_jit",
    "quantize_dynamic_jit",
    # pyrefly: ignore [bad-dunder-all]
    "_prepare_ondevice_dynamic_jit",
    # pyrefly: ignore [bad-dunder-all]
    "_convert_ondevice_dynamic_jit",
    # pyrefly: ignore [bad-dunder-all]
    "_quantize_ondevice_dynamic_jit",
    # Top level API for graph mode quantization on GraphModule(torch.fx)
    # 'fuse_fx', 'quantize_fx',  # TODO: add quantize_dynamic_fx
    # 'prepare_fx', 'prepare_dynamic_fx', 'convert_fx',
    "QuantType",  # quantization type
    # custom module APIs
    "get_default_static_quant_module_mappings",
    "get_static_quant_module_class",
    "get_default_dynamic_quant_module_mappings",
    "get_default_qat_module_mappings",
    "get_default_qconfig_propagation_list",
    "get_default_compare_output_module_list",
    "get_quantized_operator",
    "get_fuser_method",
    # Sub functions for `prepare` and `swap_module`
    "propagate_qconfig_",
    "add_quant_dequant",
    "swap_module",
    "default_eval_fn",
    # Observers
    "ObserverBase",
    # pyrefly: ignore [bad-dunder-all]
    "WeightObserver",
    "HistogramObserver",
    "observer",
    "default_observer",
    "default_weight_observer",
    "default_placeholder_observer",
    "default_per_channel_weight_observer",
    # FakeQuantize (for qat)
    "default_fake_quant",
    "default_weight_fake_quant",
    "default_fixed_qparams_range_neg1to1_fake_quant",
    "default_fixed_qparams_range_0to1_fake_quant",
    "default_per_channel_weight_fake_quant",
    "default_histogram_fake_quant",
    # QConfig
    "QConfig",
    "default_qconfig",
    "default_dynamic_qconfig",
    "float16_dynamic_qconfig",
    "float_qparams_weight_only_qconfig",
    # QAT utilities
    "default_qat_qconfig",
    "prepare_qat",
    "quantize_qat",
    # module transformations
    "fuse_modules",
]
<|endoftext|>r"""
This file is in the process of migration to `torch/ao/quantization`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate files under `torch/ao/quantization/fx/`, while adding an import statement
here.
"""

from torch.ao.quantization.fx.convert import convert
from torch.ao.quantization.fx.fuse import fuse

# omitting files that's unlikely to be used right now, for example
# the newly added lower_to_fbgemm etc.
from torch.ao.quantization.fx.prepare import prepare
<|endoftext|># mypy: allow-untyped-defs
import functools
from collections import namedtuple

import torch
import torch.utils._pytree as pytree


# NOTE [CustomOp autograd kernel indirection]
# We register `inner` as the autograd kernel for this custom_op.
# `inner` either calls the autograd formula registered by the user,
# or goes into an `autograd_not_implemented` kernel.
#
# The reason why this indirection exists is
# so that we can swap out the autograd kernel (the PyTorch dispatcher
# doesn't actually allow us to do this). By default, we want
# the `autograd_not_implemented` behavior, but then the user may come
# and register something that is actually a backward formula
def autograd_kernel_indirection(custom_op):
    autograd_fallback = autograd_not_implemented(custom_op)

    def inner(*args, **kwargs):
        if custom_op._has_impl("autograd"):
            kernel = custom_op._get_impl("autograd").func
            return kernel(*args, **kwargs)
        # As explained in NOTE ["backward", "save_for_backward", and "autograd"],
        # after the user gives us "backward" and "save_for_backward", we generate
        # the "autograd" impl. If the user only provided one, then we tell
        # the user they've done something wrong.
        if custom_op._has_impl("save_for_backward") or custom_op._has_impl("backward"):
            missing = (
                "save_for_backward" if custom_op._has_impl("backward") else "backward"
            )
            found = "save_for_backward" if missing == "backward" else "backward"
            loc = custom_op._get_impl(found).location
            raise RuntimeError(
                f"We found a '{found}' registration for {custom_op} at "
                f"{loc} but were unable to find a '{missing}' registration. "
                f"To use the CustomOp API to register a backward formula, "
                f"please provide us both a backward function and a "
                f"'save for backward' function via `impl_backward` and "
                f"`impl_save_for_backward` respectively."
            )
        return autograd_fallback(*args, **kwargs)

    return inner


# TODO(#101191): Use the actual C++ autograd not implemented fallback,
# or change the default autograd fallback to the autograd not implemented fallback.
def autograd_not_implemented(custom_op):
    def kernel(*args, **kwargs):
        if torch.is_grad_enabled() and pytree.tree_any(
            lambda x: isinstance(x, torch.Tensor) and x.requires_grad, (args, kwargs)
        ):
            raise RuntimeError("Autograd has not been implemented for operator")
        with torch._C._AutoDispatchBelowAutograd():
            return custom_op(*args, **kwargs)

    return kernel


def mark_non_differentiable(ctx, output, output_differentiability):
    # Output types are restricted to be:
    # - Tensor
    # - Tensor[]
    # - int, bool, Scalar, float
    # See _check_can_register_backward
    if output_differentiability is not None:
        if not isinstance(output, tuple):
            tuple_output = (output,)
        else:
            tuple_output = output  # type: ignore[assignment]
        if len(output_differentiability) != len(tuple_output):
            raise AssertionError(
                f"output_differentiability length {len(output_differentiability)} "
                f"!= output length {len(tuple_output)}"
            )
        non_differentiable_tensors = []
        for idx, (differentiable, out) in enumerate(
            zip(output_differentiability, tuple_output)
        ):
            if isinstance(out, torch.Tensor):
                if not differentiable:
                    non_differentiable_tensors.append(out)
                continue
            if isinstance(out, list):
                if not differentiable:
                    non_differentiable_tensors.extend(out)
                continue
            if differentiable:
                raise RuntimeError(
                    f"With output_differentiability={output_differentiability}. "
                    f"At idx {idx}, we received an object of type {type(out)} that "
                    f"is not a Tensor, so it cannot have been marked as differentiable in "
                    f"output_differentiability."
                )
        if non_differentiable_tensors:
            ctx.mark_non_differentiable(*non_differentiable_tensors)


def construct_autograd_kernel(
    schema,
    output_differentiability,
    custom_op,
    op_overload,
    save_for_backward_fn,
    backward_fn,
):
    def apply(*args):
        flat_args, spec = pytree.tree_flatten(args)
        out_spec = None

        def forward(ctx, *flat_args):
            ctx.set_materialize_grads(True)
            args = pytree.tree_unflatten(list(flat_args), spec)
            with torch._C._AutoDispatchBelowAutograd():
                output = op_overload(*args)

            # We use the info about args to give better error messages in backward
            args_info = namedtuple_args(schema, pytree.tree_map(type, args))

            save_for_backward_fn_inputs = namedtuple_args(schema, args)
            to_save = save_for_backward_fn(save_for_backward_fn_inputs, output)

            save_pytree_for_backward(ctx, (to_save, args_info))
            mark_non_differentiable(ctx, output, output_differentiability)

            nonlocal out_spec
            flat_output, out_spec = pytree.tree_flatten(output)
            return tuple(flat_output)

        def backward(ctx, *flat_grad_output):
            if out_spec is None:
                raise AssertionError("out_spec is unexpectedly None")
            grads = pytree.tree_unflatten(list(flat_grad_output), out_spec)
            saved, args_info = unpack_saved(ctx)
            # There is nothing on the ctx object for now, it is just there so
            # that we can add additional things in the future.
            inner_ctx = object()
            if not isinstance(grads, tuple):
                grads = (grads,)
            grad_inputs_dict = backward_fn(inner_ctx, saved, *grads)

            # Massage the grad_inputs_dict to a form acceptable by
            # autograd.Function.
            validate_grad_inputs_dict(grad_inputs_dict, custom_op, args_info)
            return grad_inputs_dict_to_flat_tuple(grad_inputs_dict, args_info)

        generated_cls = gen_autograd_function(
            custom_op._opname + "_customop", forward, backward
        )

        flat_output = generated_cls.apply(*flat_args)
        if out_spec is None:
            raise AssertionError("out_spec is unexpectedly None")
        return pytree.tree_unflatten(list(flat_output), out_spec)

    return apply


def gen_autograd_function(name, forward, backward):
    generated_cls = type(
        name,
        (torch.autograd.Function,),
        {
            "forward": staticmethod(forward),
            "backward": staticmethod(backward),
        },
    )
    return generated_cls


@functools.lru_cache
def namedtuple_args_cls(schema):
    attribs = [arg.name for arg in schema.arguments.flat_all]
    name = str(schema.name) + "_args"
    # mypy doesn't support dynamic namedtuple name
    tuple_cls = namedtuple(name, attribs)  # type: ignore[misc]
    return tuple_cls


def namedtuple_args(schema, args):
    if not isinstance(args, tuple):
        raise AssertionError(f"expected tuple, got {type(args)}")
    tuple_cls = namedtuple_args_cls(schema)
    return tuple_cls(*args)


def validate_grad_inputs_dict(grad_inputs_dict, forward_op, args_info):
    def error(what):
        backward = forward_op._get_impl("backward")
        raise RuntimeError(
            f"In the backward function defined for {forward_op} at "
            f"{backward.location} using the CustomOp API, {what}"
        )

    if not isinstance(grad_inputs_dict, dict):
        error(
            f"expected the output of the backward function to be a dict but "
            f"got {type(grad_inputs_dict)}"
        )

    expected_keys = {
        arg.name
        for arg in forward_op._schema.arguments.flat_all
        if arg.type.is_tensor_like()
    }
    actual_keys = grad_inputs_dict.keys()
    if expected_keys != actual_keys:
        error(
            f"expected the returned grad_input dict to have keys "
            f"{expected_keys} but got {actual_keys}. The backward "
            f"function must return a gradient (can be None) for each arg "
            f"to the CustomOp that may be a Tensor or Sequence[Tensor]. "
            f"Args declared to be non-Tensor-like types should not appear "
            f"in the grad_input dict"
        )

    for name, grad in grad_inputs_dict.items():
        arg_info = getattr(args_info, name)

        if isinstance(arg_info, list):
            if not isinstance(grad, (tuple, list)):
                error(
                    f"for input '{name}' expected the grad_input dict to "
                    f"hold a list of gradients but got object of type "
                    f"{type(grad)}."
                )
            if len(grad) != len(arg_info):
                error(
                    f"for input '{name}' expected the grad_input dict to "
                    f"hold a list of {len(arg_info)} gradients but got "
                    f"{len(grad)}"
                )
            for idx, (g, info) in enumerate(zip(grad, arg_info)):
                if g is None:
                    continue
                if not isinstance(g, torch.Tensor):
                    error(
                        f"for input '{name}' expected the grad_input dict to "
                        f"hold a list of None or Tensor gradients but got "
                        f"object of {type(g)} at index {idx}"
                    )
                if not issubclass(info, torch.Tensor):
                    error(
                        f"for input '{name}', got a Tensor as the gradient "
                        f"for the {idx}-th value but expected None because "
                        f"the {idx}-th value was not a Tensor (it was "
                        f"type {arg_info}"
                    )
            continue

        if grad is None:
            continue
        if not isinstance(grad, torch.Tensor):
            error(
                f"got object of type {type(grad)} as the gradient for input "
                f"'{name}', "
                f"but expected the gradient to be either None or a Tensor"
            )
        if not issubclass(arg_info, torch.Tensor):
            error(
                f"got a Tensor as the gradient for input '{name}' but "
                f"expected None as the gradient because input '{name}' "
                f"was not a Tensor (it was type {arg_info})."
            )


def grad_inputs_dict_to_flat_tuple(grad_inputs_dict, args_info):
    result = []
    for name, arg_info in args_info._asdict().items():
        if name not in grad_inputs_dict:
            result.append(pytree.tree_map(lambda x: None, arg_info))
            continue
        result.append(grad_inputs_dict[name])
    return tuple(pytree.tree_leaves(result))


# Saves "stuff" (a pytree) onto the ctx object. Use unpack_saved to unpack it.
# autograd.Function prefers that users use ctx.save_for_backward to
# save Tensors (to avoid reference cycles) and for non-Tensors to go onto the
# ctx object.
def save_pytree_for_backward(ctx, stuff):
    flat_stuff, spec = pytree.tree_flatten(stuff)
    num_elts = len(flat_stuff)
    tensor_idxs = [
        idx for idx, thing in enumerate(flat_stuff) if isinstance(thing, torch.Tensor)
    ]
    non_tensor_idxs = [
        idx
        for idx, thing in enumerate(flat_stuff)
        if not isinstance(thing, torch.Tensor)
    ]
    tensors = [thing for thing in flat_stuff if isinstance(thing, torch.Tensor)]
    non_tensors = [thing for thing in flat_stuff if not isinstance(thing, torch.Tensor)]

    ctx.spec = spec
    ctx.num_elts = num_elts
    ctx.save_for_backward(*tensors)
    ctx.tensor_idxs = tensor_idxs
    ctx.saved_non_tensors = non_tensors
    ctx.non_tensor_idxs = non_tensor_idxs


# Inverse operation to save_pytree_for_backward
def unpack_saved(ctx):
    flat_stuff = [None] * ctx.num_elts
    for tensor, idx in zip(ctx.saved_tensors, ctx.tensor_idxs):
        flat_stuff[idx] = tensor
    for non_tensor, idx in zip(ctx.saved_non_tensors, ctx.non_tensor_idxs):
        flat_stuff[idx] = non_tensor
    stuff = pytree.tree_unflatten(flat_stuff, ctx.spec)
    return stuff
<|endoftext|># mypy: allow-untyped-defs
import dataclasses
import functools
import inspect
import sys
import typing
import warnings
import weakref

import torch
import torch._C as _C
import torch._library.infer_schema
import torch.library as library
from torch._library.infer_schema import infer_schema
from torch.library import get_ctx
from torchgen.model import (
    BaseTy,
    BaseType,
    FunctionSchema,
    ListType,
    OperatorName,
    SchemaKind,
)

from .autograd import autograd_kernel_indirection, construct_autograd_kernel


"""
torch._custom_op is deprecated. We shipped a production-ready version of it into torch.library.
Please use those APIs instead.
"""

__all__ = ["custom_op", "CustomOp", "get_ctx"]


SUPPORTED_DEVICE_TYPE_TO_KEY = {
    "cpu": "CPU",
    "cuda": "CUDA",
}

# We will not let users register CustomOps with anything that could look like
# PyTorch internals to avoid confusion.
RESERVED_NS = {
    "prim",
    "prims",
    "aten",
    "at",
    "torch",
    "pytorch",
}


def warn_deprecated():
    warnings.warn(
        "torch._custom_op is deprecated and will be removed in PyTorch 2.6, please "
        "use the equivalent torch.library API instead.",
        DeprecationWarning,
        stacklevel=2,
    )


def custom_op(qualname: str, manual_schema: str | None = None) -> typing.Callable:
    r"""
    This API is deprecated, please use torch.library.custom_op instead
    """
    warn_deprecated()

    def inner(func):
        if not inspect.isfunction(func):
            raise ValueError(
                f"custom_op(...)(func): Expected `func` to be a Python "
                f"function, got: {type(func)}"
            )

        ns, name = parse_qualname(qualname)
        validate_namespace(ns)
        if func.__name__ != name:
            raise ValueError(
                f"custom_op(qualname='{qualname}', ...)(func): expected `func` "
                f"to have name '{name}' but got '{func.__name__}'. "
                f"Please either change the name of `func` or the qualname that "
                f"is passed to `custom_op`"
            )

        schema = (
            infer_schema(func, mutates_args=())
            if manual_schema is None
            else manual_schema
        )
        schema_str = f"{name}{schema}"
        function_schema = FunctionSchema.parse(schema_str)
        validate_schema(function_schema)
        if manual_schema is not None:
            validate_function_matches_schema(function_schema, func)

        lib = library.Library(ns, "FRAGMENT")
        lib.define(schema_str)
        ophandle = find_ophandle_or_throw(ns, function_schema.name)
        result = CustomOp(
            lib, ns, function_schema, name, ophandle, _private_access=True
        )

        result.__name__ = func.__name__  # pyrefly: ignore [bad-assignment]
        result.__module__ = func.__module__
        result.__doc__ = func.__doc__

        library.impl(lib, result._opname, "Autograd")(
            autograd_kernel_indirection(weakref.proxy(result))
        )

        torch._C._dispatch_set_report_error_callback(
            ophandle, functools.partial(report_error_callback, weakref.proxy(result))
        )

        return result

    return inner


# Global dictionary holding references to all CustomOp objects
# Yes, it keeps all CustomOps alive (see NOTE [CustomOp lifetime])
# Used to query the CustomOp associated with a specific C++ dispatcher operator.
# An example usage is FakeTensor: FakeTensor checks if a specific operator
# has an implementation registered via the CustomOp API.
# Indexed by qualname (e.g. aten::foo)
global_registry: dict[str, "CustomOp"] = {}


class CustomOp:
    r"""
    This API is deprecated, please use torch.library.custom_op instead
    """

    def __init__(
        self, lib, cpp_ns, schema, operator_name, ophandle, *, _private_access=False
    ):
        super().__init__()
        warn_deprecated()
        if not _private_access:
            raise RuntimeError(
                "The CustomOp constructor is private and we do not guarantee "
                "BC for it. Please use custom_op(...) to create a CustomOp object"
            )
        name = f"{cpp_ns}::{operator_name}"
        self._schema = schema
        self._cpp_ns = cpp_ns
        self._lib: library.Library = lib
        self._ophandle: _C._DispatchOperatorHandle = ophandle
        # Has the name of the op, e.g. "foo". We cache here for convenience.
        self._opname: str = operator_name
        # this is _opname but with namespace. e.g. "custom::foo"
        self._qualname: str = name
        self.__name__ = None  # mypy requires this
        # NB: Some of these impls are registered as kernels to DispatchKeys.
        # Modifying the _impls dict directly won't do anything in that case.
        self._impls: dict[str, FuncAndLocation | None] = {}
        # See NOTE [CustomOp autograd kernel indirection]
        self._registered_autograd_kernel_indirection = False

        global_registry[self._qualname] = self

    def _register_autograd_kernel_indirection(self):
        if self._registered_autograd_kernel_indirection:
            raise AssertionError("autograd kernel indirection already registered")
        self._lib.impl(
            self._opname, autograd_kernel_indirection(weakref.proxy(self)), "Autograd"
        )
        self._registered_autograd_kernel_indirection = True

    # Records the impl and the source location in self._impls
    # Note that this doesn't cause torch.library to use the impl, that
    # needs to be done in a separate self._lib.impl call.
    def _register_impl(self, kind, func, stacklevel=2):
        if self._has_impl(kind):
            func_and_location = self._impls[kind]
            if func_and_location is None:
                raise AssertionError("func_and_location is unexpectedly None")
            location = func_and_location.location
            raise RuntimeError(
                f"Attempting to register a {kind} impl for operator {self._qualname} "
                f"that already has a {kind} impl registered from Python at "
                f"{location}. This is not supported."
            )
        frame = inspect.getframeinfo(sys._getframe(stacklevel))
        location = f"{frame.filename}:{frame.lineno}"
        self._impls[kind] = FuncAndLocation(func, location)

    def _get_impl(self, kind):
        return self._impls[kind]

    def _has_impl(self, kind):
        return kind in self._impls

    def _destroy(self):
        # NOTE: [CustomOp lifetime]
        # A CustomOp, once created, lives forever. The mechanism is that the
        # global registry holds a reference to it. However, to make testing
        # easier, we want to be able to destroy CustomOp objects.
        # CustomOp._destroy does the job, though it leaves the CustomOp
        # in a garbage state.
        del self._lib

        opnamespace = getattr(torch.ops, self._cpp_ns)
        if hasattr(opnamespace, self._opname):
            delattr(opnamespace, self._opname)

        del global_registry[self._qualname]

    def __repr__(self):
        return f'<CustomOp(op="{self._qualname}")>'

    def __call__(self, *args, **kwargs):
        # Bypass torch.ops.* and directly do OperatorHandle::callBoxed.
        # Using torch.ops.* is a bit of a pain (it can be slow and it has lifetime
        # issues from caching operators that make testing CustomOp difficult).
        result = _C._dispatch_call_boxed(self._ophandle, *args, **kwargs)
        return result

    def impl(
        self,
        device_types: str | typing.Iterable[str],
        _stacklevel=2,
    ) -> typing.Callable:
        r"""
        This API is deprecated, please use torch.library.custom_op instead
        """
        if isinstance(device_types, str):
            device_types = [device_types]
        for device_type in device_types:
            validate_device_type(device_type)

        def inner(f):
            for device_type in set(device_types):
                self._check_doesnt_have_library_impl(device_type)
                self._register_impl(device_type, f, stacklevel=_stacklevel)
                dispatch_key = SUPPORTED_DEVICE_TYPE_TO_KEY[device_type]
                library.impl(self._lib, self._opname, dispatch_key)(f)
            return f

        return inner

    def _check_doesnt_have_library_impl(self, device_type):
        if self._has_impl(device_type):
            return
        key = SUPPORTED_DEVICE_TYPE_TO_KEY[device_type]
        if _C._dispatch_has_computed_kernel_for_dispatch_key(self._qualname, key):
            raise RuntimeError(
                f"impl(..., device_types={device_type}): the operator {self._qualname} "
                f"already has an implementation for this device type via a "
                f"pre-existing torch.library or TORCH_LIBRARY registration."
            )

    def impl_factory(self) -> typing.Callable:
        r"""Register an implementation for a factory function."""

        def inner(f):
            self._register_impl("factory", f)
            library.impl(self._lib, self._opname, "BackendSelect")(f)
            return f

        return inner

    def impl_abstract(self, _stacklevel=2) -> typing.Callable:
        r"""
        This API is deprecated, please use torch.library.custom_op instead
        """

        def inner(f):
            self._check_doesnt_have_library_meta_impl()
            self._register_impl("abstract", f, stacklevel=_stacklevel)
            location = self._get_impl("abstract").location

            qualname = self._qualname

            # Handle DispatchKey.Meta registration
            @functools.wraps(f)
            def f_with_ctx(*args, **kwargs):
                def error_on_ctx():
                    raise RuntimeError(
                        f"Attempted to call get_ctx() for the meta implementation "
                        f"for {qualname}."
                        f"You have presumably called get_ctx() because the operator "
                        f"has a data-dependent output shape; if so, there is no "
                        f"such meta implementation and this error is the correct "
                        f"behavior. Otherwise, please remove the call to get_ctx() "
                        f"in the implementation registered with impl_abstract "
                        f"at {location}"
                    )

                with torch._library.fake_impl.set_ctx_getter(error_on_ctx):
                    return f(*args, **kwargs)

            self._lib.impl(self._opname, f_with_ctx, "Meta")
            return f

        return inner

    def _check_can_register_backward(self):
        def error(detail):
            raise RuntimeError(
                f"Cannot use torch._custom_ops APIs to register backward "
                f"formula for {detail}. Got operator "
                f"{self._qualname} with schema: {schema}"
            )

        schema = self._schema
        if schema.kind() != SchemaKind.functional:
            error("non-functional operator")

        rets = schema.returns
        if not schema.returns:
            error("operator with no returns")

        if len(rets) <= 0:
            raise AssertionError(f"expected at least one return, got {len(rets)}")
        is_non_mutating_view = any(
            r.annotation is not None and not r.annotation.is_write for r in rets
        )
        if is_non_mutating_view:
            error("operator that returns views")

        # We make assumptions about the schema's return types.
        allowed_return_types = {
            BaseType(BaseTy.int): "int",
            BaseType(BaseTy.SymInt): "SymInt",
            BaseType(BaseTy.bool): "bool",
            BaseType(BaseTy.float): "float",
            BaseType(BaseTy.Tensor): "Tensor",
            ListType(BaseType(BaseTy.Tensor), None): "List[Tensor]",
        }
        for ret in schema.returns:
            if ret.type in allowed_return_types:
                continue
            error(
                f"operator with return not in {list(allowed_return_types.values())} (got {ret.type})"
            )

    def _check_doesnt_have_library_autograd_impl(self):
        if self._registered_autograd_kernel_indirection:
            return

        if _C._dispatch_has_kernel_for_dispatch_key(
            self._qualname, "CompositeImplicitAutograd"
        ):
            raise RuntimeError(
                f"impl_backward/impl_save_for_backward: the operator {self._qualname} "
                f"already has an implementation for this device type via a "
                f"pre-existing registration to DispatchKey::CompositeImplicitAutograd."
                f"CompositeImplicitAutograd operators do not need an autograd formula; "
                f"instead, the operator will decompose into its constituents and those "
                f"can have autograd formulas defined on them."
            )

        # We can improve this by adding "all Autograd<BACKEND> keys", but
        # realistically people will just be using this API for CPU/CUDA for now.
        for key in ["Autograd", "AutogradCPU", "AutogradCUDA"]:
            if _C._dispatch_has_kernel_for_dispatch_key(self._qualname, key):
                raise RuntimeError(
                    f"impl_backward/impl_save_for_backward: "
                    f"the operator {self._qualname} already has an Autograd kernel "
                    f"registered to DispatchKey::{key} via a pre-existing "
                    f"torch.library or TORCH_LIBRARY registration. Please either "
                    f"remove those registrations or don't use the torch._custom_ops APIs"
                )

    def _check_doesnt_have_library_meta_impl(self):
        if self._has_impl("abstract"):
            return

        # If the user's operator is CompositeExplicitAutograd,
        # allow them to impl_abstract. This is being pragmatic
        # (existing custom ops may have CompositeExplicitAutograd
        # registration that don't work with Meta kernels, so this
        # gives them an escape hatch).
        if _C._dispatch_has_kernel_for_dispatch_key(
            self._qualname, "CompositeExplicitAutograd"
        ) and not _C._dispatch_has_kernel_for_dispatch_key(self._qualname, "Meta"):
            return

        # Otherwise, if the user's already has a Meta kernel or their
        # op is CompositeImplicitAutograd or some other alias dispatch key,
        # raise.

        # Special case for CompositeImplicitAutograd
        if _C._dispatch_has_kernel_for_dispatch_key(
            self._qualname, "CompositeImplicitAutograd"
        ):
            raise RuntimeError(
                f"impl_abstract(...): the operator {self._qualname} "
                f"already has an implementation for this device type via a "
                f"pre-existing registration to DispatchKey::CompositeImplicitAutograd."
                f"CompositeImplicitAutograd operators do not need an abstract impl; "
                f"instead, the operator will decompose into its constituents and those "
                f"can have abstract impls defined on them."
            )

        if _C._dispatch_has_kernel_for_dispatch_key(self._qualname, "Meta"):
            raise RuntimeError(
                f"impl_abstract(...): the operator {self._qualname} "
                f"already has a DispatchKey::Meta implementation via a "
                f"pre-existing torch.library or TORCH_LIBRARY registration. "
                f"Please either remove that registration or don't call impl_abstract."
            )

    # NOTE ["backward", "save_for_backward", and "autograd"]
    # As a part of the explicit autograd API, a user must provide us
    # a "save_for_backward" function and a "backward" function.
    # When both of these have been provided, then we automatically
    # construct the "autograd" kernel.
    def _register_autograd_kernel(self):
        if not self._has_impl("backward"):
            raise AssertionError("backward impl must be registered first")
        if not self._has_impl("save_for_backward"):
            raise AssertionError("save_for_backward impl must be registered first")
        kernel = construct_autograd_kernel(
            self._schema,
            self._output_differentiability,
            self,
            get_op(self._qualname),
            self._get_impl("save_for_backward").func,
            self._get_impl("backward").func,
        )
        self._register_impl("autograd", kernel)

    def impl_save_for_backward(self, _stacklevel=2):
        r"""Register a function that tells us what to save for backward.

        Please see impl_backward for more details.
        """

        def inner(f):
            self._check_can_register_backward()
            self._check_doesnt_have_library_autograd_impl()
            if not self._registered_autograd_kernel_indirection:
                self._register_autograd_kernel_indirection()
            self._register_impl("save_for_backward", f, stacklevel=_stacklevel)
            if self._has_impl("backward"):
                self._register_autograd_kernel()

        return inner

    def impl_backward(self, output_differentiability=None, _stacklevel=2):
        r"""
        This API is deprecated, please use torch.library.custom_op instead
        """
        if output_differentiability is not None:

            def yell():
                raise RuntimeError(
                    f"impl_backward(output_differentiability): expected "
                    f"output_differentiability to be a list of bools with "
                    f"length equal to the number of outputs of this CustomOp "
                    f"got: {output_differentiability}"
                )

            if not isinstance(output_differentiability, list):
                yell()
            for diff in output_differentiability:
                if not isinstance(diff, bool):
                    yell()
            if len(self._schema.returns) != len(output_differentiability):
                yell()

        def inner(f):
            self._check_can_register_backward()
            self._check_doesnt_have_library_autograd_impl()
            if not self._registered_autograd_kernel_indirection:
                self._register_autograd_kernel_indirection()
            self._register_impl("backward", f, stacklevel=_stacklevel)
            self._output_differentiability = output_differentiability
            if self._has_impl("save_for_backward"):
                self._register_autograd_kernel()

        return inner


@dataclasses.dataclass
class FuncAndLocation:
    func: typing.Callable
    location: str


def find_ophandle_or_throw(cpp_ns: str, operator_name: OperatorName):
    overload_name = (
        "" if operator_name.overload_name is None else operator_name.overload_name
    )
    return _C._dispatch_find_schema_or_throw(
        f"{cpp_ns}::{str(operator_name.name)}", overload_name
    )


def validate_namespace(ns: str) -> None:
    if "." in ns:
        raise ValueError(
            f'custom_op(..., ns="{ns}"): expected ns to not contain any . (and be a '
            f"valid variable name)"
        )
    if ns in RESERVED_NS:
        raise ValueError(
            f"custom_op(..., ns='{ns}'): '{ns}' is a reserved namespace, "
            f"please choose something else. "
        )


def validate_schema(schema: FunctionSchema) -> None:
    if not torch._library.utils.is_functional_schema(schema):
        raise ValueError(
            f"custom_op only supports functional operators "
            f"(ops that do not mutate any inputs, do not return "
            f"views of the inputs, and has at least one return). "
            f"Got the following non-functional schema: {schema}"
        )

    # For simplicity: don't allow self arguments
    if schema.arguments.self_arg is not None:
        raise ValueError(
            f"custom_op does not support arguments named 'self'. Please "
            f"rename your argument. Got: {schema}"
        )


def parse_qualname(qualname: str) -> tuple[str, str]:
    names = qualname.split("::", 1)
    if len(names) != 2:
        raise ValueError(
            f"Expected there to be a namespace in {qualname}, i.e. The "
            f"operator name should look something like ns::foo"
        )
    if "." in names[1]:
        raise ValueError(
            f"The torch.custom_ops APIs do not handle overloads, "
            f"i.e. operator names with '.' in them. "
            f"Please name your operator something like ns::foo. "
            f"Got: {qualname}"
        )
    return names[0], names[1]


def validate_device_type(device_type: str) -> None:
    if device_type not in SUPPORTED_DEVICE_TYPE_TO_KEY:
        raise ValueError(
            f"CustomOp.impl(device_types=[{device_type}, ...]): we only support device_type "
            f"in {SUPPORTED_DEVICE_TYPE_TO_KEY.keys()}."
        )


def supported_param(param: inspect.Parameter) -> bool:
    return param.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


def validate_function_matches_schema(
    schema: FunctionSchema, func: typing.Callable
) -> None:
    sig = inspect.signature(func)

    if not all(supported_param(p) for p in sig.parameters.values()):
        raise ValueError(
            f"custom_op(..., manual_schema)(func): positional-only args, "
            f"varargs, and kwargs are not supported. Please rewrite `func` "
            f"to not have them. Got `func` with signature: {sig}"
        )

    if (
        any(
            p.annotation is not inspect.Parameter.empty for p in sig.parameters.values()
        )
        or sig.return_annotation is not inspect.Signature.empty
    ):
        raise ValueError(
            f"custom_op(..., manual_schema)(func): When passing in a manual "
            f"schema, we expect `func` to have no type annotations to avoid "
            f"ambiguity. Got `func` with signature: {sig}"
        )

    positional = [
        (name, param)
        for name, param in sig.parameters.items()
        if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]
    kwargonly = [
        (name, param)
        for name, param in sig.parameters.items()
        if param.kind == inspect.Parameter.KEYWORD_ONLY
    ]

    def error():
        raise ValueError(
            f"custom_op(..., manual_schema)(func): When passing in a manual "
            f"schema, we expect `func`'s signature to match `manual_schema` "
            f"(aside from type annotations). "
            f"func's signature: {sig}, manual_schema: {schema}"
        )

    def error_default_args():
        raise ValueError(
            f"custom_op(..., manual_schema)(func): "
            f"neither func nor manual_schema should have default "
            f"arguments. Got "
            f"func's signature: {sig}, manual_schema: {schema}"
        )

    def compare(sig_args, schema_args):
        if len(sig_args) != len(schema_args):
            error()
        for (name, param), arg in zip(sig_args, schema_args):
            if name != arg.name:
                error()
            if param.default is not inspect.Parameter.empty or arg.default is not None:
                error_default_args()

    compare(positional, schema.arguments.flat_positional)
    compare(kwargonly, schema.arguments.flat_kwarg_only)


def report_error_callback(custom_op: typing.Any, key: str) -> None:
    if key == "Undefined":
        raise NotImplementedError(
            f"{custom_op}: There were no Tensor inputs to this operator "
            f"(e.g. you passed an empty list of Tensors). If your operator is a "
            f"factory function (that is, it takes no Tensors and constructs "
            f"a new one), then please use CustomOp.impl_factory to register "
            f"an implementation for it"
        )
    if key == "Meta":
        raise NotImplementedError(
            f"{custom_op}: when running with device='Meta' tensors: there is no "
            f"abstract impl registered for this CustomOp. Please register one via "
            f"CustomOp.impl_abstract to get this CustomOp to work with Meta tensors"
        )
    if key in ("CPU", "CUDA"):
        device = key.lower()
        raise NotImplementedError(
            f"{custom_op}: when running with device='{device}' tensors: there is no "
            f"{device} impl registered for this CustomOp. Please register one via "
            f"CustomOp.impl(device_type='{device}')"
        )
    raise NotImplementedError(
        f"{custom_op}: No implementation for dispatch key {key}. It is likely "
        f"that we have not added this functionality yet, please either open an "
        f"issue or if you're feeling adventurous, use the low-level "
        f"torch.library API"
    )


def custom_op_from_existing(op):
    ns = op.namespace
    lib = torch.library.Library(ns, "FRAGMENT")
    name = op.name().split("::")[-1]
    schema_str = str(op._schema)
    # CustomOp expects the schema string without the namespace
    schema_str = schema_str.rsplit("::", maxsplit=1)[-1]
    schema = FunctionSchema.parse(schema_str)
    return CustomOp(lib, ns, schema, name, op, _private_access=True)


def get_op(qualname):
    def error_not_found():
        raise ValueError(
            f"Could not find the operator {qualname}. Please make sure you have "
            f"already registered the operator and (if registered from C++) "
            f"loaded it via torch.ops.load_library."
        )

    ns, name = parse_qualname(qualname)
    if not hasattr(torch.ops, ns):
        error_not_found()
    opnamespace = getattr(torch.ops, ns)
    if not hasattr(opnamespace, name):
        error_not_found()
    packet = getattr(opnamespace, name)
    if not hasattr(packet, "default"):
        error_not_found()
    return packet.default


def _find_custom_op(qualname, also_check_torch_library=False):
    if qualname in global_registry:
        return global_registry[qualname]
    if not also_check_torch_library:
        raise RuntimeError(
            f'Could not find custom op "{qualname}". Did you register it via '
            f"the torch._custom_ops API?"
        )
    overload = get_op(qualname)
    result = custom_op_from_existing(overload)
    return result


def get_abstract_impl(qualname):
    if qualname not in torch._custom_op.impl.global_registry:
        return None
    custom_op = torch._custom_op.impl.global_registry[qualname]
    if custom_op is None:
        return None
    if not custom_op._has_impl("abstract"):
        return None
    return custom_op._get_impl("abstract").func


def _custom_op_with_schema(qualname, schema, needs_fixed_stride_order=True):
    ns, name = qualname.split("::")
    schema_str = f"{name}{schema}"
    function_schema = FunctionSchema.parse(schema_str)
    validate_schema(function_schema)
    tags = [torch._C.Tag.needs_fixed_stride_order] if needs_fixed_stride_order else []
    lib = library.Library(ns, "FRAGMENT")
    lib.define(schema_str, tags=tags)
    ophandle = find_ophandle_or_throw(ns, function_schema.name)
    result = CustomOp(lib, ns, function_schema, name, ophandle, _private_access=True)
    result._register_autograd_kernel_indirection()

    torch._C._dispatch_set_report_error_callback(
        ophandle, functools.partial(report_error_callback, weakref.proxy(result))
    )
    return get_op(qualname)
<|endoftext|>import torch


def pin_memory(data_ptr: int, size: int) -> None:
    cudart = torch.cuda.cudart()
    succ = int(
        cudart.cudaHostRegister(
            data_ptr,
            size,
            1,  # lines up with 'cudaHostRegisterPortable'
        )
    )

    if succ != 0:
        raise RuntimeError(
            f"Registering memory failed with cudaError: {succ}."
            " It's possible that this is an asynchronous error raised from a previous cuda operation."
            " Consider launching with CUDA_LAUNCH_BLOCKING=1 to debug."
        )


def unpin_memory(data_ptr: int) -> None:
    succ = int(torch.cuda.cudart().cudaHostUnregister(data_ptr))
    if succ != 0:
        raise AssertionError(f"Unpinning shared memory failed with error-code: {succ}")
<|endoftext|>from __future__ import annotations

import functools
import sys
import warnings
from typing import Any
from typing_extensions import deprecated

import torch
from torch.cuda._utils import (
    _check_cuda_bindings,
    _cuda_bindings_driver as _drv,
    _cuda_bindings_runtime as _rt,
    _HAS_CUDA_BINDINGS,
)


__all__ = [
    "GreenContext",
]

_STREAMS_PER_GREEN_CONTEXT_POOL = 32

_WORKQUEUE_SCOPE_VALUES = {
    "device_ctx": 0,
    "balanced": 1,
}

_CONTEXT_STACK_DEPRECATION = (
    "`GreenContext.set_context` and `GreenContext.pop_context` are deprecated. "
    "Please create a stream with `GreenContext.Stream()` and use "
    "`torch.cuda.stream(stream)` instead."
)


# note: this can safely be cached in a process/thread because
# the driver version cannot change during the lifetime of a process
@functools.cache
def _get_driver_version() -> int:
    try:
        # pyrefly: ignore [missing-attribute]
        return _check_cuda_bindings(_drv.cuDriverGetVersion())
    except RuntimeError as e:
        warnings.warn(f"Error while querying CUDA driver version: {e}")
        return -1


def _ensure_driver_version(version: int, message: str) -> None:
    drv_version = _get_driver_version()
    if drv_version < 0 or drv_version < version:
        raise RuntimeError(message)


def _ensure_supported() -> None:
    if torch.version.cuda is None or torch.version.hip is not None:
        raise RuntimeError("Green Context is only supported on Nvidia CUDA")
    if sys.platform == "win32":
        raise RuntimeError("Green Context is not supported on Windows")
    if not _HAS_CUDA_BINDINGS:
        raise RuntimeError("GreenContext requires the cuda.bindings package")
    _ensure_driver_version(12080, "Green Context requires user mode driver 12.8+")


def _ensure_workqueue_supported() -> None:
    _ensure_driver_version(
        13010, "CUDA user mode driver too old to use workqueue configuration!"
    )


def _parse_workqueue_scope(workqueue_scope: str | None) -> int | None:
    if workqueue_scope is None:
        return None
    if workqueue_scope not in _WORKQUEUE_SCOPE_VALUES:
        raise ValueError(
            "workqueue_scope must be 'device_ctx' or 'balanced', "
            f"got '{workqueue_scope}'"
        )
    return _WORKQUEUE_SCOPE_VALUES[workqueue_scope]


class GreenContext:
    r"""Wrapper around a CUDA green context.

    .. warning::
       This API is in beta and may change in future releases.

    CUDA work should be placed on streams created from the green context:

    .. code-block:: python

        ctx = GreenContext(...)
        stream = ctx.Stream()
        with torch.cuda.stream(stream):
            # torch operations here are using resources from `ctx`
            pass

    Green-context streams are custom CUDA streams. Synchronization with other
    streams is the user's responsibility and should be handled with CUDA events,
    as with any other custom stream.
    """

    def __init__(
        self,
        *,
        num_sms: int | None = None,
        workqueue_scope: str | None = None,
        workqueue_concurrency_limit: int | None = None,
        device_id: int | None = None,
    ) -> None:
        r"""Create a CUDA green context.

        At least one of ``num_sms`` or ``workqueue_scope`` must be specified.
        Both can be combined to partition SMs and configure workqueues in the
        same green context.

        Arguments:
            num_sms (int, optional): The number of SMs to use in the green
                context. When ``None``, SMs are not partitioned.
            workqueue_scope (str, optional): Workqueue sharing scope. One of
                ``"device_ctx"`` (shared across all contexts, default driver
                behavior) or ``"balanced"`` (non-overlapping workqueues with
                other balanced green contexts). When ``None``, no workqueue
                configuration is applied.
            workqueue_concurrency_limit (int, optional): Maximum number of
                concurrent stream-ordered workloads for the workqueue. Requires
                ``workqueue_scope`` to be set.
            device_id (int, optional): The device index of green context.
                When ``None``, the current device is used.
        """
        self._device_id = None
        self._green_ctx = None
        self._context = None
        _ensure_supported()

        scope_value = _parse_workqueue_scope(workqueue_scope)
        if scope_value is not None:
            _ensure_workqueue_supported()

        if num_sms is None and scope_value is None:
            raise RuntimeError(
                "At least one of num_sms or workqueue_scope must be specified"
            )
        if workqueue_concurrency_limit is not None and scope_value is None:
            raise RuntimeError(
                "workqueue_concurrency_limit requires workqueue_scope to be set"
            )

        if device_id is None:
            device_id = torch.cuda.current_device()

        # pyrefly: ignore [missing-attribute]
        current_ctx = _check_cuda_bindings(_drv.cuCtxGetCurrent())

        if int(current_ctx) == 0:
            warnings.warn(
                "Attempted to create a green context but there was no primary "
                "context! Creating a primary context...",
                stacklevel=2,
            )
            # pyrefly: ignore [missing-attribute]
            _check_cuda_bindings(_rt.cudaFree(0))

        # pyrefly: ignore [missing-attribute]
        device = _check_cuda_bindings(_drv.cuDeviceGet(device_id))
        resources = []

        if num_sms is not None:
            sm_resource = _check_cuda_bindings(
                _drv.cuDeviceGetDevResource(  # pyrefly: ignore [missing-attribute]
                    device,
                    _drv.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM,  # pyrefly: ignore [missing-attribute]
                )
            )
            if num_sms <= 0 or num_sms > sm_resource.sm.smCount:
                raise RuntimeError(
                    "Invalid number of SMs requested for green context: "
                    f"{num_sms} (device has {sm_resource.sm.smCount} SMs)"
                )
            split_result, nb_groups, _remaining = _check_cuda_bindings(
                _drv.cuDevSmResourceSplitByCount(  # pyrefly: ignore [missing-attribute]
                    1, sm_resource, 0, num_sms
                )
            )
            if nb_groups != 1:
                raise RuntimeError("Failed to create single SM resource group")
            resources.append(split_result[0])

        if scope_value is not None:
            wq_resource = _check_cuda_bindings(
                _drv.cuDeviceGetDevResource(  # pyrefly: ignore [missing-attribute]
                    device,
                    _drv.CUdevResourceType.CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG,  # pyrefly: ignore [missing-attribute]
                )
            )
            wq_resource.wqConfig.sharingScope = scope_value
            if workqueue_concurrency_limit is not None:
                wq_resource.wqConfig.wqConcurrencyLimit = workqueue_concurrency_limit
            resources.append(wq_resource)

        desc = _check_cuda_bindings(
            _drv.cuDevResourceGenerateDesc(  # pyrefly: ignore [missing-attribute]
                resources, len(resources)
            )
        )
        green_ctx = _check_cuda_bindings(
            _drv.cuGreenCtxCreate(  # pyrefly: ignore [missing-attribute]
                desc,
                device,
                _drv.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM,  # pyrefly: ignore [missing-attribute]
            )
        )
        try:
            # pyrefly: ignore [missing-attribute]
            context = _check_cuda_bindings(_drv.cuCtxFromGreenCtx(green_ctx))
            if int(context) == 0:
                raise RuntimeError("Green ctx conversion to regular ctx failed!")
        except Exception:
            # pyrefly: ignore [missing-attribute]
            _check_cuda_bindings(_drv.cuGreenCtxDestroy(green_ctx))
            raise

        self._init_from_cuda_objects(device_id, green_ctx, context)

    def __del__(self) -> None:
        green_ctx = getattr(self, "_green_ctx", None)
        if green_ctx is None:
            return

        # attempt to destroy streams related to this green context
        # we ignore errors to avoid leaking exceptions during __del__
        end = min(_STREAMS_PER_GREEN_CONTEXT_POOL, self._curr_stream_idx + 1)
        for idx in reversed(range(end)):
            try:
                # pyrefly: ignore [missing-attribute]
                _check_cuda_bindings(_drv.cuStreamDestroy(self._green_ctx_streams[idx]))
            except RuntimeError as e:
                warnings.warn(
                    f"Error while destroying green context stream at idx {idx} "
                    f"for green context {green_ctx}: {e}"
                )
        self._green_ctx = None
        try:
            # pyrefly: ignore [missing-attribute]
            _check_cuda_bindings(_drv.cuGreenCtxDestroy(green_ctx))
        except RuntimeError as e:
            warnings.warn(f"Error while destroying green context {green_ctx}: {e}")

    def _init_from_cuda_objects(self, device_id: int, green_ctx, context) -> None:
        self._device_id = device_id
        self._green_ctx = green_ctx
        self._context = context
        self._parent_stream: torch.cuda.Stream | None = None
        self._green_ctx_streams: list[Any | None] = [
            None
        ] * _STREAMS_PER_GREEN_CONTEXT_POOL
        self._curr_stream_idx = -1

    @staticmethod
    def create(
        *,
        num_sms: int | None = None,
        workqueue_scope: str | None = None,
        workqueue_concurrency_limit: int | None = None,
        device_id: int | None = None,
    ) -> GreenContext:
        r"""Create a CUDA green context.

        Kept for compatibility, see `GreenContext` constructor.
        """
        return GreenContext(
            num_sms=num_sms,
            workqueue_scope=workqueue_scope,
            workqueue_concurrency_limit=workqueue_concurrency_limit,
            device_id=device_id,
        )

    @staticmethod
    def max_workqueue_concurrency(device_id: int | None = None) -> int:
        r"""Return the maximum workqueue concurrency limit for the device.

        This queries the device for the default number of concurrent
        stream-ordered workloads supported by workqueue configuration
        resources.

        Arguments:
            device_id (int, optional): The device index to query. When
                ``None``, the current device is used.
        """
        _ensure_supported()
        _ensure_workqueue_supported()
        if device_id is None:
            device_id = torch.cuda.current_device()

        # pyrefly: ignore [missing-attribute]
        device = _check_cuda_bindings(_drv.cuDeviceGet(device_id))
        wq_resource = _check_cuda_bindings(
            _drv.cuDeviceGetDevResource(  # pyrefly: ignore [missing-attribute]
                device,
                _drv.CUdevResourceType.CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG,  # pyrefly: ignore [missing-attribute]
            )
        )
        return wq_resource.wqConfig.wqConcurrencyLimit

    def _ensure_alive(self) -> None:
        if self._green_ctx is None or self._context is None:
            raise RuntimeError("GreenContext has been destroyed")

    @deprecated(_CONTEXT_STACK_DEPRECATION, category=FutureWarning)
    def set_context(self) -> None:
        r"""Make the green context the current context.

        Deprecated. Create streams with :meth:`Stream` and use
        :func:`torch.cuda.stream` instead.
        """
        self._ensure_alive()
        if self._parent_stream is not None:
            raise RuntimeError("set_context called twice before pop_context")
        current_stream = torch.cuda.current_stream()
        self._parent_stream = current_stream

        event = torch.cuda.Event()
        event.record(current_stream)

        # pyrefly: ignore [missing-attribute]
        current_ctx = _check_cuda_bindings(_drv.cuCtxGetCurrent())
        if int(current_ctx) == 0:
            # pyrefly: ignore [missing-attribute]
            _check_cuda_bindings(_drv.cuCtxSetCurrent(self._context))
        else:
            # pyrefly: ignore [missing-attribute]
            _check_cuda_bindings(_drv.cuCtxPushCurrent(self._context))

        green_ctx_stream = torch.cuda.default_stream(self._device_id)
        event.wait(green_ctx_stream)
        torch.cuda.set_stream(green_ctx_stream)

    @deprecated(_CONTEXT_STACK_DEPRECATION, category=FutureWarning)
    def pop_context(self) -> None:
        r"""Assuming the green context is the current context, pop it from the
        context stack and restore the previous context.

        Deprecated. Create streams with :meth:`Stream` and use
        :func:`torch.cuda.stream` instead.
        """
        try:
            self._ensure_alive()
            if self._parent_stream is None:
                raise RuntimeError("pop_context called without matching set_context")

            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream())

            # pyrefly: ignore [missing-attribute]
            popped = _check_cuda_bindings(_drv.cuCtxPopCurrent())
            # pyrefly: ignore [bad-argument-type]
            if int(popped) != int(self._context):
                raise RuntimeError("expected popped context to be the current ctx")

            event.wait(self._parent_stream)
            torch.cuda.set_stream(self._parent_stream)
        finally:
            self._parent_stream = None

    def Stream(self) -> torch.cuda.Stream:
        r"""Return a CUDA stream associated with this green context.

        Use the returned stream with :func:`torch.cuda.stream` to run work on
        the green context. Synchronization with other streams is not automatic;
        use CUDA events as with any other custom stream.
        """
        self._ensure_alive()
        curr_idx = self._curr_stream_idx + 1
        idx = curr_idx % _STREAMS_PER_GREEN_CONTEXT_POOL
        if curr_idx < _STREAMS_PER_GREEN_CONTEXT_POOL:
            green_ctx_stream = _check_cuda_bindings(
                _drv.cuGreenCtxStreamCreate(  # pyrefly: ignore [missing-attribute]
                    self._green_ctx,
                    _drv.CUstream_flags.CU_STREAM_NON_BLOCKING,  # pyrefly: ignore [missing-attribute]
                    0,
                )
            )
            self._green_ctx_streams[idx] = green_ctx_stream
        else:
            green_ctx_stream = self._green_ctx_streams[idx]
        self._curr_stream_idx = curr_idx
        # pyrefly: ignore [bad-argument-type]
        return torch.cuda.ExternalStream(int(green_ctx_stream), self._device_id)
<|endoftext|>from collections.abc import Callable

from torch._utils import CallbackRegistry


EventCreationCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "CUDA event creation"
)
EventDeletionCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "CUDA event deletion"
)
EventRecordCallbacks: "CallbackRegistry[int, int]" = CallbackRegistry(
    "CUDA event record"
)
EventWaitCallbacks: "CallbackRegistry[int, int]" = CallbackRegistry("CUDA event wait")
MemoryAllocationCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "CUDA memory allocation"
)
MemoryDeallocationCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "CUDA memory deallocation"
)
StreamCreationCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "CUDA stream creation"
)
DeviceSynchronizationCallbacks: "CallbackRegistry[[]]" = CallbackRegistry(
    "CUDA device synchronization"
)
StreamSynchronizationCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "CUDA stream synchronization"
)
EventSynchronizationCallbacks: "CallbackRegistry[int]" = CallbackRegistry(
    "CUDA event synchronization"
)


def register_callback_for_event_creation(cb: Callable[[int], None]) -> None:
    EventCreationCallbacks.add_callback(cb)


def register_callback_for_event_deletion(cb: Callable[[int], None]) -> None:
    EventDeletionCallbacks.add_callback(cb)


def register_callback_for_event_record(cb: Callable[[int, int], None]) -> None:
    EventRecordCallbacks.add_callback(cb)


def register_callback_for_event_wait(cb: Callable[[int, int], None]) -> None:
    EventWaitCallbacks.add_callback(cb)


def register_callback_for_memory_allocation(cb: Callable[[int], None]) -> None:
    MemoryAllocationCallbacks.add_callback(cb)


def register_callback_for_memory_deallocation(cb: Callable[[int], None]) -> None:
    MemoryDeallocationCallbacks.add_callback(cb)


def register_callback_for_stream_creation(cb: Callable[[int], None]) -> None:
    StreamCreationCallbacks.add_callback(cb)


def register_callback_for_device_synchronization(cb: Callable[[], None]) -> None:
    DeviceSynchronizationCallbacks.add_callback(cb)


def register_callback_for_stream_synchronization(cb: Callable[[int], None]) -> None:
    StreamSynchronizationCallbacks.add_callback(cb)


def register_callback_for_event_synchronization(cb: Callable[[int], None]) -> None:
    EventSynchronizationCallbacks.add_callback(cb)
<|endoftext|># mypy: allow-untyped-defs
from collections.abc import Iterable

import torch
from torch import Tensor

from . import _lazy_call, _lazy_init, current_device, device_count, is_initialized


__all__ = [
    "get_rng_state",
    "get_rng_state_all",
    "set_rng_state",
    "set_rng_state_all",
    "manual_seed",
    "manual_seed_all",
    "seed",
    "seed_all",
    "initial_seed",
]


def get_rng_state(device: int | str | torch.device = "cuda") -> Tensor:
    r"""Return the random number generator state of the specified GPU as a ByteTensor.

    Args:
        device (torch.device or int, optional): The device to return the RNG state of.
            Default: ``'cuda'`` (i.e., ``torch.device('cuda')``, the current CUDA device).

    .. warning::
        This function eagerly initializes CUDA.
    """
    _lazy_init()
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device("cuda", device)
    idx = device.index
    if idx is None:
        idx = current_device()
    default_generator = torch.cuda.default_generators[idx]
    return default_generator.get_state()


def get_rng_state_all() -> list[Tensor]:
    r"""Return a list of ByteTensor representing the random number states of all devices."""
    results = [get_rng_state(i) for i in range(device_count())]
    return results


def set_rng_state(new_state: Tensor, device: int | str | torch.device = "cuda") -> None:
    r"""Set the random number generator state of the specified GPU.

    Args:
        new_state (torch.ByteTensor): The desired state
        device (torch.device or int, optional): The device to set the RNG state.
            Default: ``'cuda'`` (i.e., ``torch.device('cuda')``, the current CUDA device).
    """
    if not is_initialized():
        with torch._C._DisableFuncTorch():
            # Clone the state because the callback will be triggered
            # later when CUDA is lazy initialized.
            new_state = new_state.clone(memory_format=torch.contiguous_format)
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device("cuda", device)

    def cb():
        idx = device.index
        if idx is None:
            idx = current_device()
        default_generator = torch.cuda.default_generators[idx]
        default_generator.set_state(new_state)

    _lazy_call(cb)


def set_rng_state_all(new_states: Iterable[Tensor]) -> None:
    r"""Set the random number generator state of all devices.

    Args:
        new_states (Iterable of torch.ByteTensor): The desired state for each device.
    """
    for i, state in enumerate(new_states):
        set_rng_state(state, i)


def manual_seed(seed: int) -> None:
    r"""Set the seed for generating random numbers for the current GPU.

    It's safe to call this function if CUDA is not available; in that
    case, it is silently ignored.

    Args:
        seed (int): The desired seed.

    .. warning::
        If you are working with a multi-GPU model, this function is insufficient
