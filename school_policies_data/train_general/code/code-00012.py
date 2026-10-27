    dim = _wrap_axis(_unwrap_if_constexpr(dim), rank)
    assert all(input.shape[i] == other.shape[i] for i in builtins.range(rank) if i !=
               dim), f"tensor dims must match except in the concat dimension {dim}, got {input.shape} and {other.shape}"

    # Join introduces a new minor dim; move it before the concat dim and merge.
    c = join(input, other, _semantic=_semantic)
    order = list(builtins.range(rank))
    order.insert(dim, rank)
    c = permute(c, order, _semantic=_semantic)
    new_shape = list(input.shape)
    new_shape[dim] = input.shape[dim] + other.shape[dim]
    return reshape(c, new_shape, _semantic=_semantic)


@builtin
def join(a, b, _semantic=None):
    """
    Join the given tensors in a new, minor dimension.

    For example, given two tensors of shape (4,8), produces a new tensor of
    shape (4,8,2).  Given two scalars, returns a tensor of shape (2).

    The two inputs are broadcasted to be the same shape.

    If you want to join more than two elements, you can use multiple calls to
    this function.  This reflects the constraint in Triton that tensors must
    have power-of-two sizes.

    join is the inverse of split.

    :param a: The first input tensor.
    :type a: Tensor
    :param b: The second input tensor.
    :type b: Tensor
    """
    return _semantic.join(a, b)


def _unsplat(x, _semantic=None, _generator=None):
    """
    Convert a single-element tensor to a scalar.
    """
    if len(x.shape) == 0:
        return x
    numel = 1
    for d in x.shape:
        numel *= d
    assert numel == 1, "can only unsplat single-element tensors"
    return _semantic.unsplat(x)


@_tensor_member_fn
@builtin
def split(a, _semantic=None, _generator=None) -> tuple[tensor, tensor]:
    """
    Split a tensor in two along its last dim, which must have size 2.

    For example, given a tensor of shape (4,8,2), produces two tensors of shape
    (4,8).  Given a tensor of shape (2), returns two scalars.

    If you want to split into more than two pieces, you can use multiple calls
    to this function (probably plus calling reshape).  This reflects the
    constraint in Triton that tensors must have power-of-two sizes.

    split is the inverse of join.

    :param a: The tensor to split.
    :type a: Tensor
    """
    # If len(a.shape) == 1, i.e. a.shape == [2], we should return two scalars.
    # But _semantic.split can only handle returning tensors.  Work around this by
    # expanding the input to shape [1,2] and then reducing the result.
    was_rank_1 = len(a.shape) == 1
    if was_rank_1:
        a = _semantic.expand_dims(a, 0)

    out_lhs, out_rhs = _semantic.split(a)

    if was_rank_1:
        # Currently `reduce` is the best way to convert a tensor of shape [1] to a scalar.
        out_lhs = _unsplat(out_lhs, _semantic=_semantic, _generator=_generator)
        out_rhs = _unsplat(out_rhs, _semantic=_semantic, _generator=_generator)

    return out_lhs, out_rhs


@_tensor_member_fn
@builtin
def view(input, *shape, _semantic=None):
    """
    Returns a tensor with the same elements as `input` but a different shape.
    The order of the elements may not be preserved.

    :param input: The input tensor.
    :type input: Block
    :param shape: The desired shape.

    :code:`shape` can be passed as a tuple or as individual parameters: ::

        # These are equivalent
        view(x, (32, 32))
        view(x, 32, 32)
    """
    warn("view is deprecated, please use reshape with can_reorder being true.")
    shape = _shape_check_impl(_unwrap_iterable(shape))
    return _semantic.reshape(input, shape, can_reorder=True)


@_tensor_member_fn
@builtin
def item(input, _semantic=None, _generator=None):
    """
    Converts a single-element tensor into a scalar.
    """
    return _unsplat(input, _semantic=_semantic, _generator=_generator)


@_tensor_member_fn
@builtin
def reshape(input, *shape, can_reorder=False, _semantic=None, _generator=None):
    """
    Returns a tensor with the same number of elements as input but with the
    provided shape.

    :param input: The input tensor.
    :type input: Block
    :param shape: The new shape.

    :code:`shape` can be passed as a tuple or as individual parameters: ::

        # These are equivalent
        reshape(x, (32, 32))
        reshape(x, 32, 32)
    """
    shape = _shape_check_impl(_unwrap_iterable(shape))
    if len(shape) == 0:
        return _unsplat(input, _semantic=_semantic, _generator=_generator)
    return _semantic.reshape(input, shape, can_reorder)


def _wrap_axis(axis, ndim):
    if not (-ndim <= axis < ndim):
        raise ValueError(f"invalid axis {axis}. Expected {-ndim} <= axis < {ndim}")

    return axis if axis >= 0 else axis + ndim


@_tensor_member_fn
@builtin
def expand_dims(input, axis, _semantic=None):
    """
    Expand the shape of a tensor, by inserting new length-1 dimensions.

    Axis indices are with respect to the resulting tensor, so
    ``result.shape[axis]`` will be 1 for each axis.

    :param input: The input tensor.
    :type input: tl.tensor
    :param axis: The indices to add new axes
    :type axis: int | Sequence[int]

    """
    input = _semantic.to_tensor(input)
    axis = _unwrap_if_constexpr(axis)
    axes = list(axis) if isinstance(axis, (Sequence, tuple)) else [axis]
    new_ndim = len(input.shape) + len(axes)
    axes = [_wrap_axis(_unwrap_if_constexpr(d), new_ndim) for d in axes]

    if len(set(axes)) != len(axes):
        raise ValueError(f"expand_dims received duplicate axes, normalized axes = {axes}")

    ret = input
    for a in sorted(axes):
        ret = _semantic.expand_dims(ret, a)
    return ret


@_tensor_member_fn
@builtin
def cast(input, dtype: dtype, fp_downcast_rounding: Optional[str] = None, bitcast: bool = False, _semantic=None):
    """
    Casts a tensor to the given :code:`dtype`.

    :param dtype: The target data type.
    :type dtype: tl.dtype
    :param fp_downcast_rounding: The rounding mode for downcasting
        floating-point values. This parameter is only used when self is a
        floating-point tensor and dtype is a floating-point type with a
        smaller bitwidth. Supported values are :code:`"rtne"` (round to
        nearest, ties to even) and :code:`"rtz"` (round towards zero).
    :type fp_downcast_rounding: str, optional
    :param bitcast: If true, the tensor is bitcasted to the given
        :code:`dtype`, instead of being numerically casted.
    :type bitcast: bool, optional
    """
    input = _semantic.to_tensor(input)
    dtype = _unwrap_if_constexpr(dtype)
    fp_downcast_rounding = _unwrap_if_constexpr(fp_downcast_rounding)
    bitcast = _unwrap_if_constexpr(bitcast)
    if bitcast:
        return _semantic.bitcast(input, dtype)
    return _semantic.cast(input, dtype, fp_downcast_rounding)


# -----------------------
# Linear Algebra
# -----------------------


@builtin
def dot(input, other, acc=None, input_precision=None, allow_tf32=None, max_num_imprecise_acc=None, out_dtype=None,
        _semantic=None):
    """
    Returns the matrix product of two blocks.

    The two blocks must both be two-dimensional or three-dimensional and have compatible inner dimensions.
    For three-dimensional blocks, `tl.dot` performs the batched matrix product,
    where the first dimension of each block represents the batch dimension.

    .. warning::
      When using TF32 precision, the float32 inputs may be truncated to TF32 format (19-bit floating point)
      without rounding which may bias the result. For best results, you must round to TF32 explicitly, or load
      the data using `TensorDescriptor` with `round_f32_to_tf32=True`.

    :param input: The first tensor to be multiplied.
    :type input: 2D or 3D tensor of scalar-type in {:code:`int8`, :code:`float8_e5m2`, :code:`float16`, :code:`bfloat16`, :code:`float32`}
    :param other: The second tensor to be multiplied.
    :type other: 2D or 3D tensor of scalar-type in {:code:`int8`, :code:`float8_e5m2`, :code:`float16`, :code:`bfloat16`, :code:`float32`}
    :param acc: The accumulator tensor. If not None, the result is added to this tensor.
    :type acc: 2D or 3D tensor of scalar-type in {:code:`float16`, :code:`float32`, :code:`int32`}
    :param input_precision: How to exercise the Tensor Cores for f32 x f32. If
      the device does not have Tensor Cores or the inputs are not of dtype f32,
      this option is ignored. For devices that do have tensor cores, the
      default precision is tf32.
    :type input_precision: string. Available options for nvidia: :code:`"tf32"`, :code:`"tf32x3"`, :code:`"ieee"`. Default: :code:`"tf32"`. Available options for amd: :code:`"ieee"`, (CDNA3 only) :code:`"tf32"`.
    :param allow_tf32: *Deprecated.* If true, input_precision is set to "tf32".
      Only one of :code:`input_precision` and :code:`allow_tf32` can be
      specified (i.e. at least one must be :code:`None`).
    """
    assert input_precision is None or allow_tf32 is None, "Only one of input_precision and allow_tf32 can be specified"
    if input_precision is None:
        supports_tf32 = "tf32" in _semantic.builder.options.allowed_dot_input_precisions
        input_precision = knobs.language.fp32_default or ("tf32" if (supports_tf32 and
                                                                     (allow_tf32 or allow_tf32 is None)) else "ieee")

    input_precision = _unwrap_if_constexpr(input_precision)
    out_dtype = _unwrap_if_constexpr(out_dtype)
    max_num_imprecise_acc = _unwrap_if_constexpr(max_num_imprecise_acc)
    acc = _unwrap_if_constexpr(acc)

    # check shapes make sense:
    a_shape = list(input.shape)
    b_shape = list(other.shape)
    assert len(a_shape) == len(b_shape) >= 2, "input and other must have equal ranks >= 2"
    assert a_shape[:-2] == b_shape[:-2], "input and other must have equal batch shapes"
    assert a_shape[-1] == b_shape[-2], "input and other must have equal reduction dimensions"

    # compute shape of accumulator:
    c_shape = a_shape[:-1] + [b_shape[-1]]
    if acc is not None:
        assert list(acc.shape) == c_shape, "accumulator shape is incompatible"
    rank = len(c_shape)

    if rank >= 4:
        batch_size = 1
        for i in builtins.range(rank - 2):
            batch_size *= c_shape[i]
        input = _semantic.reshape(input, [batch_size] + a_shape[-2:], can_reorder=False)
        other = _semantic.reshape(other, [batch_size] + b_shape[-2:], can_reorder=False)
        if acc is not None:
            acc = _semantic.reshape(acc, [batch_size] + c_shape[-2:], can_reorder=False)

    res = _semantic.dot(input, other, acc, input_precision, max_num_imprecise_acc, out_dtype)

    if rank >= 4:
        res = _semantic.reshape(res, c_shape, can_reorder=False)

    assert list(res.shape) == c_shape, "output shape is unexpected"
    return res


@builtin
def dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, acc=None, fast_math=False, lhs_k_pack=True,
               rhs_k_pack=True, out_dtype=float32, _semantic=None):
    """
    Returns the matrix product of two blocks in microscaling format.

    lhs and rhs use microscaling formats described here:
    https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf

    Software emulation enables targeting hardware architectures without native microscaling
    operation support. Right now for such case, microscaled lhs/rhs are upcasted to
    :code:`bf16` element type beforehand for dot computation, with one exception:
    for AMD CDNA3 specifically, if one of the inputs is of :code:`fp16` element type,
    the other input is also upcasted to :code:`fp16` element type instead.
    This behavior is experimental and may be subject to change in the future.

    :param lhs: The first tensor to be multiplied.
    :type lhs: 2D tensor representing fp4, fp8 or bf16 elements. Fp4 elements are packed into uint8 inputs with the first element in lower bits. Fp8 are stored as uint8 or the corresponding fp8 type.
    :param lhs_scale: Scale factor for lhs tensor. Shape should be [M, K//group_size] when lhs is [M, K], where group_size is 32 if scales type are `e8m0`.
    :type lhs_scale: e8m0 type represented as an uint8 tensor, or None.
    :param lhs_format: format of the lhs tensor. Available formats: {:code:`e2m1`, :code:`e4m3`, :code:`e5m2`, :code:`bf16`, :code:`fp16`}.
    :type lhs_format: str
    :param rhs: The second tensor to be multiplied.
    :type rhs: 2D tensor representing fp4, fp8 or bf16 elements. Fp4 elements are packed into uint8 inputs with the first element in lower bits. Fp8 are stored as uint8 or the corresponding fp8 type.
    :param rhs_scale: Scale factor for rhs tensor. Shape should be [N, K//group_size] where rhs is [K, N].
                      Important: Do NOT transpose rhs_scale
    :type rhs_scale: e8m0 type represented as an uint8 tensor, or None.
    :param rhs_format: format of the rhs tensor. Available formats: {:code:`e2m1`, :code:`e4m3`, :code:`e5m2`, :code:`bf16`, :code:`fp16`}.
    :type rhs_format: str
    :param acc: The accumulator tensor. If not None, the result is added to this tensor.
    :param lhs_k_pack: If false, the lhs tensor is packed into uint8 along M dimension.
    :type lhs_k_pack: bool, optional
    :param rhs_k_pack: If false, the rhs tensor is packed into uint8 along N dimension.
    :type rhs_k_pack: bool, optional
    """
    lhs_format = _unwrap_if_constexpr(lhs_format)
    rhs_format = _unwrap_if_constexpr(rhs_format)
    acc = _unwrap_if_constexpr(acc)
    fast_math = _unwrap_if_constexpr(fast_math)
    out_dtype = _unwrap_if_constexpr(out_dtype)
    lhs_k_pack = _unwrap_if_constexpr(lhs_k_pack)
    rhs_k_pack = _unwrap_if_constexpr(rhs_k_pack)
    assert out_dtype == float32, "Only float32 is supported for out_dtype at the moment"
    return _semantic.dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, acc, fast_math, lhs_k_pack,
                                rhs_k_pack, out_dtype)


# -----------------------
# Non-Atomic Memory Operations
# -----------------------


@builtin
def load(pointer, mask=None, other=None, boundary_check=(), padding_option="", cache_modifier="", eviction_policy="",
         volatile=False, _semantic=None):
    """
    Return a tensor of data whose values are loaded from memory at location defined by `pointer`:

        (1) If `pointer` is a single element pointer, a scalar is be loaded.  In
            this case:

            - `mask` and `other` must also be scalars,
            - `other` is implicitly typecast to `pointer.dtype.element_ty`, and
            - `boundary_check` and `padding_option` must be empty.

        (2) If `pointer` is an N-dimensional tensor of pointers, an
            N-dimensional tensor is loaded.  In this case:

            - `mask` and `other` are implicitly broadcast to `pointer.shape`,
            - `other` is implicitly typecast to `pointer.dtype.element_ty`, and
            - `boundary_check` and `padding_option` must be empty.

        (3) If `pointer` is a block pointer defined by `make_block_ptr`, a
            tensor is loaded.  In this case:

            - `mask` and `other` must be `None`, and
            - `boundary_check` and `padding_option` can be specified to control the behavior of out-of-bound access.

    :param pointer: Pointer to the data to be loaded
    :type pointer: `triton.PointerType`, or block of `dtype=triton.PointerType`
    :param mask: if `mask[idx]` is false, do not load the data at address `pointer[idx]`
        (must be `None` with block pointers)
    :type mask: Block of `triton.int1`, optional
    :param other: if `mask[idx]` is false, return `other[idx]`. If `other` is `None`, the masked-out value is undefined.
    :type other: Block, optional
    :param boundary_check: tuple of integers, indicating the dimensions which should do the boundary check
    :type boundary_check: tuple of ints, optional
    :param padding_option: should be one of {"", "zero", "nan"}, the padding value to use while out of bounds. "" means an undefined value.
    :param cache_modifier: changes cache option in NVIDIA PTX
    :type cache_modifier: str, optional, should be one of {"", ".ca", ".cg", ".cv"}, where ".ca" stands for
        cache at all levels, ".cg" stands for cache at global level (cache in L2 and below, not L1),
        and ".cv" means don’t cache and fetch again. see
        `cache operator <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators>`_ for more details.
    :param eviction_policy: changes eviction policy in NVIDIA PTX
    :type eviction_policy: str, optional
    :param volatile: changes volatile option in NVIDIA PTX
    :type volatile: bool, optional
    """
    if _is_block_ptr(pointer):
        return pointer.load(mask=mask, other=other, boundary_check=boundary_check, padding_option=padding_option,
                            cache_modifier=cache_modifier, eviction_policy=eviction_policy, volatile=volatile,
                            _semantic=_semantic)

    # `mask` and `other` can be constexpr
    mask = _unwrap_if_constexpr(mask)
    other = _unwrap_if_constexpr(other)
    if mask is not None:
        mask = _semantic.to_tensor(mask)
    if other is not None:
        other = _semantic.to_tensor(other)
    padding_option = _unwrap_if_constexpr(padding_option)
    cache_modifier = _unwrap_if_constexpr(cache_modifier)
    eviction_policy = _unwrap_if_constexpr(eviction_policy)
    volatile = _unwrap_if_constexpr(volatile)
    return _semantic.load(pointer, mask, other, boundary_check, padding_option, cache_modifier, eviction_policy,
                          volatile)


@builtin
def load_tensor_descriptor(desc: tensor_descriptor_base, offsets: Sequence[constexpr | tensor],
                           _semantic=None) -> tensor:
    """Load a block of data from a tensor descriptor."""
    return desc.load(offsets, _semantic=_semantic)


@builtin
def store_tensor_descriptor(desc: tensor_descriptor_base, offsets: Sequence[constexpr | tensor], value: tensor,
                            _semantic=None) -> tensor:
    """Store a block of data to a tensor descriptor."""
    return desc.store(offsets, value, _semantic=_semantic)


@_tensor_member_fn
@builtin
def store(pointer, value, mask=None, boundary_check=(), cache_modifier="", eviction_policy="", _semantic=None):
    """
    Store a tensor of data into memory locations defined by `pointer`.

        (1) If `pointer` is a single element pointer, a scalar is stored.  In
            this case:

            - `mask` must also be scalar, and
            - `boundary_check` and `padding_option` must be empty.

        (2) If `pointer` is an N-dimensional tensor of pointers, an
            N-dimensional block is stored.  In this case:

            - `mask` is implicitly broadcast to `pointer.shape`, and
            - `boundary_check` must be empty.

        (3) If `pointer` is a block pointer defined by `make_block_ptr`, a block
            of data is stored.  In this case:

            - `mask` must be None, and
            - `boundary_check` can be specified to control the behavior of out-of-bound access.

    `value` is implicitly broadcast to `pointer.shape` and typecast to `pointer.dtype.element_ty`.

    :param pointer: The memory location where the elements of `value` are stored
    :type pointer: `triton.PointerType`, or block of `dtype=triton.PointerType`
    :param value: The tensor of elements to be stored
    :type value: Block
    :param mask: If `mask[idx]` is false, do not store `value[idx]` at `pointer[idx]`
    :type mask: Block of triton.int1, optional
    :param boundary_check: tuple of integers, indicating the dimensions which should do the boundary check
    :type boundary_check: tuple of ints, optional
    :param cache_modifier: changes cache option in NVIDIA PTX
    :type cache_modifier: str, optional, should be one of {"", ".wb", ".cg", ".cs", ".wt"}, where ".wb" stands for
        cache write-back all coherent levels, ".cg" stands for cache global, ".cs" stands for cache streaming, ".wt"
        stands for cache write-through, see `cache operator <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators>`_ for more details.
    :param eviction_policy: changes eviction policy in NVIDIA PTX
    :type eviction_policy: str, optional, should be one of {"", "evict_first", "evict_last"}
    """
    if _is_block_ptr(pointer):
        return pointer.store(value, mask=mask, boundary_check=boundary_check, cache_modifier=cache_modifier,
                             eviction_policy=eviction_policy, _semantic=_semantic)

    # `value` can be constexpr
    value = _semantic.to_tensor(value)
    mask = _unwrap_if_constexpr(mask)
    if mask is not None:
        mask = _semantic.to_tensor(mask)
    cache_modifier = _unwrap_if_constexpr(cache_modifier)
    eviction_policy = _unwrap_if_constexpr(eviction_policy)
    return _semantic.store(pointer, value, mask, boundary_check, cache_modifier, eviction_policy)


@builtin
def make_block_ptr(base: tensor, shape, strides, offsets, block_shape, order, _semantic=None):
    """
    Returns a pointer to a block in a parent tensor

    :param base: The base pointer to the parent tensor
    :param shape: The shape of the parent tensor
    :param strides: The strides of the parent tensor
    :param offsets: The offsets to the block
    :param block_shape: The shape of the block
    :param order: The order of the original data format
    """
    warn("tl.make_block_ptr is deprecated. Use TensorDescriptor or tl.make_tensor_descriptor instead.")
    return _block_ptr(base, shape, strides, offsets, block_shape, order, _semantic=_semantic)


@must_use_result(
    "Note that tl.advance does not have any side effects. To move the block pointer, you need to assign the result of tl.advance to a variable."
)
@_tensor_member_fn
@builtin
def advance(base, offsets, _semantic=None):
    """
    Advance a block pointer

    :param base: the block pointer to advance
    :param offsets: the offsets to advance, a tuple by dimension
    """
    if _is_block_ptr(base):
        return base.advance(offsets, _semantic=_semantic)
    raise ValueError("`tl.advance` only supports block pointers created by `tl.make_block_ptr`")


@builtin
def make_tensor_descriptor(
    base: tensor,
    shape: List[tensor],
    strides: List[tensor],
    block_shape: List[constexpr],
    padding_option="zero",
    _semantic=None,
) -> tensor_descriptor:
    """Make a tensor descriptor object

    :param base: the base pointer of the tensor, must be 16-byte aligned
    :param shape: A list of non-negative integers representing the tensor shape
    :param strides: A list of tensor strides. Leading dimensions must be multiples
        of 16-byte strides and the last dimension must be contiguous.
    :param block_shape: The shape of block to be loaded/stored from global memory

    Notes
    *****
    On NVIDIA GPUs with TMA support, this will result in a TMA descriptor object
    and loads and stores from the descriptor will be backed by the TMA hardware.

    Currently only 2-5 dimensional tensors are supported.

    Example
    *******
    .. code-block:: python

        @triton.jit
        def inplace_abs(in_out_ptr, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
            desc = tl.make_tensor_descriptor(
                in_out_ptr,
                shape=[M, N],
                strides=[N, 1],
                block_shape=[M_BLOCK, N_BLOCK],
            )

            moffset = tl.program_id(0) * M_BLOCK
            noffset = tl.program_id(1) * N_BLOCK

            value = desc.load([moffset, noffset])
            desc.store([moffset, noffset], tl.abs(value))

        # TMA descriptors require a global memory allocation
        def alloc_fn(size: int, alignment: int, stream: Optional[int]):
            return torch.empty(size, device="cuda", dtype=torch.int8)

        triton.set_allocator(alloc_fn)

        M, N = 256, 256
        x = torch.randn(M, N, device="cuda")
        M_BLOCK, N_BLOCK = 32, 32
        grid = (M / M_BLOCK, N / N_BLOCK)
        inplace_abs[grid](x, M, N, M_BLOCK, N_BLOCK)

    """

    padding_option = _unwrap_if_constexpr(padding_option)
    return _semantic.make_tensor_descriptor(base, shape, strides, block_shape, padding_option)


# -----------------------
# Atomic Memory Operations
# -----------------------


def _add_atomic_docstr(name: str, has_cmp: bool = False) -> Callable[[T], T]:

    def _decorator(func: T) -> T:
        docstr = f"""
    Performs an atomic {name} at the memory location specified by :code:`pointer`.

    Return the data stored at :code:`pointer` before the atomic operation.

    :param pointer: The memory locations to operate on
    :type pointer: Block of dtype=triton.PointerDType"""
        if has_cmp:
            docstr += """
    :param cmp: The values expected to be found in the atomic object
    :type cmp: Block of dtype=pointer.dtype.element_ty"""
        docstr += """
    :param val: The values with which to perform the atomic operation
    :type val: Block of dtype=pointer.dtype.element_ty
    :param sem: Specifies the memory semantics for the operation. Acceptable values are "acquire",
        "release", "acq_rel" (stands for "ACQUIRE_RELEASE"), and "relaxed". If not provided,
        the function defaults to using "acq_rel" semantics.
    :type sem: str, optional
    :param scope: Defines the scope of threads that observe the synchronizing effect of the atomic operation.
        Acceptable values are "gpu" (default), "cta" (cooperative thread array, thread block), or "sys" (stands for "SYSTEM"). The default value is "gpu".
    :type scope: str, optional
    """
        func.__doc__ = docstr
        return func

    return _decorator


@_tensor_member_fn
@builtin
@_add_atomic_docstr("compare-and-swap", has_cmp=True)
def atomic_cas(pointer, cmp, val, sem=None, scope=None, _semantic=None):
    cmp = _semantic.to_tensor(cmp)
    val = _semantic.to_tensor(val)
    sem = _unwrap_if_constexpr(sem)
    scope = _unwrap_if_constexpr(scope)
    return _semantic.atomic_cas(pointer, cmp, val, sem, scope)


@_tensor_member_fn
@builtin
@_add_atomic_docstr("exchange")
def atomic_xchg(pointer, val, mask=None, sem=None, scope=None, _semantic=None):
    val = _semantic.to_tensor(val)
    sem = _unwrap_if_constexpr(sem)
    scope = _unwrap_if_constexpr(scope)
    mask = _unwrap_if_constexpr(mask)
    return _semantic.atomic_xchg(pointer, val, mask, sem, scope)


@_tensor_member_fn
@builtin
@_add_atomic_docstr("add")
def atomic_add(pointer, val, mask=None, sem=None, scope=None, _semantic=None):
    val = _semantic.to_tensor(val)
    sem = _unwrap_if_constexpr(sem)
    scope = _unwrap_if_constexpr(scope)
    mask = _unwrap_if_constexpr(mask)
    return _semantic.atomic_add(pointer, val, mask, sem, scope)


@_tensor_member_fn
@builtin
@_add_atomic_docstr("max")
def atomic_max(pointer, val, mask=None, sem=None, scope=None, _semantic=None):
    val = _semantic.to_tensor(val)
    sem = _unwrap_if_constexpr(sem)
    scope = _unwrap_if_constexpr(scope)
    mask = _unwrap_if_constexpr(mask)
    return _semantic.atomic_max(pointer, val, mask, sem, scope)


@_tensor_member_fn
@builtin
@_add_atomic_docstr("min")
def atomic_min(pointer, val, mask=None, sem=None, scope=None, _semantic=None):
    val = _semantic.to_tensor(val)
    sem = _unwrap_if_constexpr(sem)
    scope = _unwrap_if_constexpr(scope)
    mask = _unwrap_if_constexpr(mask)
    return _semantic.atomic_min(pointer, val, mask, sem, scope)


@_tensor_member_fn
@builtin
@_add_atomic_docstr("logical and")
def atomic_and(pointer, val, mask=None, sem=None, scope=None, _semantic=None):
    val = _semantic.to_tensor(val)
    sem = _unwrap_if_constexpr(sem)
    scope = _unwrap_if_constexpr(scope)
    mask = _unwrap_if_constexpr(mask)
    return _semantic.atomic_and(pointer, val, mask, sem, scope)


@_tensor_member_fn
@builtin
@_add_atomic_docstr("logical or")
def atomic_or(pointer, val, mask=None, sem=None, scope=None, _semantic=None):
    val = _semantic.to_tensor(val)
    sem = _unwrap_if_constexpr(sem)
    scope = _unwrap_if_constexpr(scope)
    mask = _unwrap_if_constexpr(mask)
    return _semantic.atomic_or(pointer, val, mask, sem, scope)


@_tensor_member_fn
@builtin
@_add_atomic_docstr("logical xor")
def atomic_xor(pointer, val, mask=None, sem=None, scope=None, _semantic=None):
    val = _semantic.to_tensor(val)
    sem = _unwrap_if_constexpr(sem)
    scope = _unwrap_if_constexpr(scope)
    mask = _unwrap_if_constexpr(mask)
    return _semantic.atomic_xor(pointer, val, mask, sem, scope)


# -----------------------
# Conditioning
# -----------------------


@builtin
def where(condition, x, y, _semantic=None):
    """
    Returns a tensor of elements from either :code:`x` or :code:`y`, depending on :code:`condition`.

    Note that :code:`x` and :code:`y` are always evaluated regardless of the value of :code:`condition`.

    If you want to avoid unintended memory operations, use the :code:`mask` arguments in `triton.load` and `triton.store` instead.

    The shape of :code:`x` and :code:`y` are both broadcast to the shape of :code:`condition`.
    :code:`x` and :code:`y` must have the same data type.

    :param condition: When True (nonzero), yield x, otherwise yield y.
    :type condition: Block of triton.bool
    :param x: values selected at indices where condition is True.
    :param y: values selected at indices where condition is False.
    """
    condition = _semantic.to_tensor(condition)
    x = _unwrap_if_constexpr(x)
    y = _unwrap_if_constexpr(y)
    return _semantic.where(condition, x, y)


@builtin
def expect_zero(x, mask, _semantic=None):
    """
    Mark values that are expected to have underflowed to zero.

    In regular compilation this preserves :code:`x`. Debug builds assert that
    :code:`x` is zero wherever :code:`mask` is true. Under FPSAN this becomes
    :code:`where(mask, 0, x)` so sanitized execution observes the intended
    floating-point underflow.

    :param x: values to preserve outside FPSAN mode.
    :param mask: positions where :code:`x` is expected to be zero.
    """
    x = _unwrap_if_constexpr(x)
    mask = _semantic.to_tensor(mask)
    instrumentation_mode = getattr(_semantic.builder.options, "instrumentation_mode", "")
    if "fpsan" in instrumentation_mode:
        return _semantic.where(mask, 0, x)
    if _semantic.builder.options.debug:
        x_tensor = _semantic.to_tensor(x)
        zero = _semantic.to_tensor(0)
        cond = _semantic.or_(_semantic.equal(x_tensor, zero), _semantic.not_(mask))
        _semantic.device_assert(cond, "expect_zero expected x == 0 where mask is true", None)
    return x


# -----------------------
# Math
# -----------------------


@builtin
def add(x, y, sanitize_overflow: constexpr = True, _semantic=None):
    x = _unwrap_if_constexpr(x)
    y = _unwrap_if_constexpr(y)
    return _semantic.add(x, y, sanitize_overflow)


@builtin
def sub(x, y, sanitize_overflow: constexpr = True, _semantic=None):
    x = _unwrap_if_constexpr(x)
    y = _unwrap_if_constexpr(y)
    return _semantic.sub(x, y, sanitize_overflow)


@builtin
def mul(x, y, sanitize_overflow: constexpr = True, _semantic=None):
    x = _unwrap_if_constexpr(x)
    y = _unwrap_if_constexpr(y)
    return _semantic.mul(x, y, sanitize_overflow)


@builtin
def minimum(x, y, propagate_nan: constexpr = PropagateNan.NONE, _semantic=None):
    """
    Computes the element-wise minimum of :code:`x` and :code:`y`.

    :param x: the first input tensor
    :type x: Block
    :param y: the second input tensor
    :type y: Block
    :param propagate_nan: whether to propagate NaN values.
    :type propagate_nan: tl.PropagateNan

    .. seealso:: :class:`tl.PropagateNan`
    """
    x = _semantic.to_tensor(x)
    y = _semantic.to_tensor(y)
    x = _promote_bfloat16_to_float32(x, _semantic=_semantic)
    y = _promote_bfloat16_to_float32(y, _semantic=_semantic)
    propagate_nan = _unwrap_if_constexpr(propagate_nan)
    return _semantic.minimum(x, y, propagate_nan)


@builtin
def maximum(x, y, propagate_nan: constexpr = PropagateNan.NONE, _semantic=None):
    """
    Computes the element-wise maximum of :code:`x` and :code:`y`.

    :param x: the first input tensor
    :type x: Block
    :param y: the second input tensor
    :type y: Block
    :param propagate_nan: whether to propagate NaN values.
    :type propagate_nan: tl.PropagateNan

    .. seealso:: :class:`tl.PropagateNan`
    """
    x = _semantic.to_tensor(x)
    y = _semantic.to_tensor(y)
    x = _promote_bfloat16_to_float32(x, _semantic=_semantic)
    y = _promote_bfloat16_to_float32(y, _semantic=_semantic)
    propagate_nan = _unwrap_if_constexpr(propagate_nan)
    return _semantic.maximum(x, y, propagate_nan)


@builtin
def clamp(x, min, max, propagate_nan: constexpr = PropagateNan.NONE, _semantic=None):
    """
    Clamps the input tensor :code:`x` within the range [min, max].
    Behavior when :code:`min` > :code:`max` is undefined.

    :param x: the input tensor
    :type x: Block
    :param min: the lower bound for clamping
    :type min: Block
    :param max: the upper bound for clamping
    :type max: Block
    :param propagate_nan: whether to propagate NaN values. Applies only to the :code:`x` tensor.
        If either :code:`min` or :code:`max` is NaN, the result is undefined.
    :type propagate_nan: tl.PropagateNan

    .. seealso:: :class:`tl.PropagateNan`
    """
    x = _semantic.to_tensor(x)
    min = _semantic.to_tensor(min)
    max = _semantic.to_tensor(max)
    x = _promote_bfloat16_to_float32(x, _semantic=_semantic)
    min = _promote_bfloat16_to_float32(min, _semantic=_semantic)
    max = _promote_bfloat16_to_float32(max, _semantic=_semantic)

    propagate_nan = _unwrap_if_constexpr(propagate_nan)

    return _semantic.clamp(x, min, max, propagate_nan)


# -----------------------
# Reductions
# -----------------------


def _add_reduction_docstr(name: str, return_indices_arg: str = None, tie_break_arg: str = None,
                          dtype_arg: str = None) -> Callable[[T], T]:

    def _decorator(func: T) -> T:
        docstr = """
    Returns the {name} of all elements in the :code:`input` tensor along the provided :code:`axis`

    The reduction operation should be associative and commutative.

    :param input: the input values
    :type input: Tensor
    :param axis: the dimension along which the reduction should be done. If None, reduce all dimensions
    :type axis: int
    :param keep_dims: if true, keep the reduced dimensions with length 1
    :type keep_dims: bool"""
        if return_indices_arg is not None:
            docstr += f"""
    :param {return_indices_arg}: if true, return index corresponding to the {name} value
    :type {return_indices_arg}: bool"""
        if tie_break_arg is not None:
            docstr += f"""
    :param {tie_break_arg}: if true, in case of a tie (i.e., multiple elements have the same {name} value), return the left-most index for values that aren't NaN
    :type {tie_break_arg}: bool"""
        if dtype_arg is not None:
            docstr += f"""
    :param {dtype_arg}: the desired data type of the returned tensor. If specified, the input tensor is casted to :code:`{dtype_arg}` before the operation is performed. This is useful for preventing data overflows. If not specified, integer and bool dtypes are upcasted to :code:`tl.int32` and float dtypes are upcasted to at least :code:`tl.float32`.
    :type {dtype_arg}: tl.dtype"""

        func.__doc__ = docstr.format(name=name)
        return func

    return _decorator


@contextmanager
def _insertion_guard(builder):
    ip = builder.get_insertion_point()
    yield
    builder.restore_insertion_point(ip)


@_tensor_member_fn
@builtin
def reduce(input, axis, combine_fn, keep_dims=False, _semantic=None, _generator=None):
    """Applies the combine_fn to all elements in :code:`input` tensors along the provided :code:`axis`

    :param input: the input tensor, or tuple of tensors
    :type input: Tensor
    :param axis: the dimension along which the reduction should be done. If None, reduce all dimensions
    :type axis: int | None
    :param combine_fn: a function to combine two groups of scalar tensors (must be marked with @triton.jit)
    :type combine_fn: Callable
    :param keep_dims: if true, keep the reduced dimensions with length 1
    :type keep_dims: bool

    """
    if isinstance(input, tensor):
        return reduce((input, ), axis, combine_fn, keep_dims=keep_dims, _semantic=_semantic, _generator=_generator)[0]

    def make_combine_region(reduce_op):
        param_types = [t.type.scalar for t in input] * 2
        region = reduce_op.get_region(0)
        builder = _semantic.builder
        with _insertion_guard(builder):
            to_ir = lambda T: T.to_ir(builder)
            block = builder.create_block_with_parent(region, list(map(to_ir, param_types)))
            args = [tensor(block.arg(i), ty) for i, ty in enumerate(param_types)]
            results = _generator.call_JitFunction(combine_fn, args, kwargs={})
            if isinstance(results, tensor):
                handles = [results.handle]
            else:
                handles = [r.handle for r in results]
            builder.create_reduce_ret(*handles)

    def expand_ndims(t, ndims):
        for _ in builtins.range(ndims):
            t = expand_dims(t, 0, _semantic=_semantic)
        return t

    axis = _unwrap_if_constexpr(axis)
    keep_dims = _unwrap_if_constexpr(keep_dims)
    if axis is not None:
        axis = _wrap_axis(axis, len(input[0].shape))
    ret = _semantic.reduction(input, axis, make_combine_region)
    if keep_dims:
        if axis is not None:
            ret = tuple(expand_dims(t, axis, _semantic=_semantic) for t in ret)
        else:
            ret = tuple(expand_ndims(t, len(input[0].shape)) for t in ret)
    return ret


@builtin
def _promote_bfloat16_to_float32(t, _semantic=None):
    scalar_ty = t.type.scalar

    # hardware doesn't support FMAX, FMIN, CMP for bfloat16
    if scalar_ty is bfloat16:
        return t.to(float32, _semantic=_semantic)
    return t


@builtin
def _reduce_with_indices(input, axis, combine_fn, keep_dims=False, _semantic=None, _generator=None):
    axis = _unwrap_if_constexpr(axis)
    n = input.shape[axis]
    index = arange(0, n, _semantic=_semantic)

    if len(input.shape) > 1:
        # Broadcast index across the non-reduced axes
        axes_to_expand = [constexpr(d) for d in builtins.range(len(input.shape))]
        del axes_to_expand[axis]
        index = expand_dims(index, axes_to_expand, _semantic=_semantic)
        index = broadcast_to(index, input.shape, _semantic=_semantic)

    rvalue, rindices = reduce((input, index), axis, combine_fn, keep_dims=keep_dims, _semantic=_semantic,
                              _generator=_generator)
    return rvalue, rindices


# -----------------------
# Scans
# -----------------------


def _add_scan_docstr(name: str, dtype_arg: str = None) -> Callable[[T], T]:

    def _decorator(func: T) -> T:
        docstr = """
    Returns the {name} of all elements in the :code:`input` tensor along the provided :code:`axis`

    :param input: the input values
    :type input: Tensor
    :param axis: the dimension along which the scan should be done
    :type axis: int
    :param reverse: if true, the scan is performed in the reverse direction
    :type reverse: bool"""

        if dtype_arg is not None:
            docstr += f"""
    :param {dtype_arg}: the desired data type of the returned tensor. If specified, the input tensor is casted to :code:`{dtype_arg}` before the operation is performed. If not specified, small integer types (< 32 bits) are upcasted to prevent overflow. Note that :code:`tl.bfloat16` inputs are automatically promoted to :code:`tl.float32`.
    :type {dtype_arg}: tl.dtype"""

        func.__doc__ = docstr.format(name=name)
        return func

    return _decorator


@_tensor_member_fn
@builtin
def associative_scan(input, axis, combine_fn, reverse=False, _semantic=None, _generator=None):
    """Applies the combine_fn to each elements with a carry in :code:`input` tensors along the provided :code:`axis` and update the carry

    :param input: the input tensor, or tuple of tensors
    :type input: Tensor
    :param axis: the dimension along which the reduction should be done
    :type axis: int
    :param combine_fn: a function to combine two groups of scalar tensors (must be marked with @triton.jit)
    :type combine_fn: Callable
    :param reverse: whether to apply the associative scan in the reverse direction along axis
    :type reverse: bool

    """
    if isinstance(input, tensor):
        return associative_scan((input, ), axis, combine_fn, reverse, _semantic=_semantic, _generator=_generator)[0]

    def make_combine_region(scan_op):
        param_types = [t.type.scalar for t in input] * 2
        region = scan_op.get_region(0)
        builder = _semantic.builder
        with _insertion_guard(builder):
            to_ir = lambda T: T.to_ir(builder)
            block = builder.create_block_with_parent(region, list(map(to_ir, param_types)))
            args = [tensor(block.arg(i), ty) for i, ty in enumerate(param_types)]
            results = _generator.call_JitFunction(combine_fn, args, kwargs={})
            if isinstance(results, tensor):
                handles = [results.handle]
            else:
                handles = [r.handle for r in results]
            builder.create_scan_ret(*handles)

    axis = _unwrap_if_constexpr(axis)
    reverse = _unwrap_if_constexpr(reverse)
    if axis is not None:
        axis = _wrap_axis(axis, len(input[0].shape))
    return _semantic.associative_scan(input, axis, make_combine_region, reverse)


@_tensor_member_fn
@builtin
def histogram(input, num_bins, mask=None, _semantic=None, _generator=None):
    """computes an histogram based on input tensor with num_bins bins, the bins have a width of 1 and start at 0.

    :param input: the input tensor
    :type input: Tensor
    :param num_bins: number of histogram bins
    :type num_bins: int
    :param mask: if `mask[idx]` is false, exclude `input[idx]` from histogram
    :type mask: Block of `triton.int1`, optional

    """
    num_bins = _unwrap_if_constexpr(num_bins)
    mask = _unwrap_if_constexpr(mask)
    if mask is not None:
        mask = _semantic.to_tensor(mask)
    return _semantic.histogram(input, num_bins, mask)


@_tensor_member_fn
@builtin
def gather(src, index, axis, _semantic=None):
    """Gather from a tensor along a given dimension.

    :param src: the source tensor
    :type src: Tensor
    :param index: the index tensor
    :type index: Tensor
    :param axis: the dimension to gather along
    :type axis: int

    """
    src = _unwrap_if_constexpr(src)
    index = _unwrap_if_constexpr(index)
    axis = _unwrap_if_constexpr(axis)
    return _semantic.gather(src, index, axis)


@builtin
def map_elementwise(
    scalar_fn: Callable[..., Tuple[tensor, ...]],
    *args: tensor,
    pack=1,
    _semantic=None,
    _generator=None,
):
    '''
        Map a scalar function over a tensor.

        The input tensors :code:`args` are implicitly broadcasted to the same shape.

        This may be useful in allowing control flow over single elements in a tensor,
        for example a multi-branch function where one branch is more expensive. With
        :code:`tl.where` you are forced to calculate both sides of the branch, but
        with an if we only execute one side.

        .. highlight:: python
        .. code-block:: python

            @triton.jit
            def selu_scalar(x, alpha):
                if x > 0:
                    return a
                else:
                    return alpha * (tl.exp(x) - 1)

            @triton.jit
            def selu(x, alpha):
                return tl.map_elementwise(selu_scalar, x, alpha)

        :param scalar_fn: the function to map over.
        :param pack: the number of elements to be processed by one function call.
        :return: one tensor or a tuple of tensors, depending on the mapped function.
    '''
    # Build the block for the nested region first to discover the return types
    assert pack >= 1, f"pack must be >= 1, got {pack}"
    in_scalar_tys = [t.type.scalar for t in args]
    builder = _semantic.builder
    block = builder.new_block()
    scalar_args = []
    original_loc = builder.get_loc()
    for i, ty in enumerate(in_scalar_tys):
        for j in builtins.range(pack):
            block.add_argument_at(ty.to_ir(builder), original_loc)
            scalar_args.append(tensor(block.arg(i * pack + j), ty))

    with _insertion_guard(builder):
        builder.set_insertion_point_to_start(block)
        scalar_results = _generator.call_JitFunction(scalar_fn, scalar_args, kwargs={})

        is_single = isinstance(scalar_results, tensor)
        if is_single:
            scalar_results = scalar_results,

        handles = [r.handle for r in scalar_results]
        builder.set_loc(original_loc)
        builder.create_map_elementwise_ret(handles)

    fn_result_types = [x.type for x in scalar_results]
    scalar_result_types = fn_result_types
    if pack > 1:
        scalar_result_types = fn_result_types[::pack]
        for offset in builtins.range(1, pack):
            assert scalar_result_types == fn_result_types[offset::pack], "type mismatch in unpacked results"

    def make_elementwise_region(elementwise_op):
        region = elementwise_op.get_region(0)
        region.push_back(block)

    builder.set_loc(original_loc)
    result = _semantic.map_elementwise(args, scalar_result_types, pack, make_elementwise_region)
    return result[0] if is_single else result


# -----------------------
# Compiler Hint Ops
# -----------------------


@builtin
def debug_barrier(_semantic=None):
    '''
    Insert a barrier to synchronize all threads in a block.
    '''
    return _semantic.debug_barrier()


@builtin
def multiple_of(input, values, _semantic=None):
    """
    Let the compiler know that ``values[d]`` is the largest power of two that
    divides the first element of every contiguous group along dimension ``d``
    of :code:`input` (see :func:`max_contiguous` for the definition of contiguous
    group). ``values`` must have one entry per dimension of :code:`input`.

    For a 1D input with contiguity 1, this is equivalent to saying that every
    element of :code:`input` is a multiple of ``values[0]``. For example, if
    :code:`values` is ``[16]`` and :code:`input` is ``[64, 80, 96, 112]``, the
    hint is valid because 16 divides 64.

    This hint enables alignment-dependent optimizations such as vectorized
    memory accesses.
    """
    if isinstance(values, constexpr):
        values = [values]
    for i, d in enumerate(values):
        if not isinstance(d, constexpr):
            raise TypeError(f"values element {i} must have type `constexpr`")
        if not isinstance(d.value, int):
            raise TypeError(f"values element {i} must have type `constexpr[int]`, got `constexpr[{type(d.value)}]")
    values = [x.value for x in values]
    return _semantic.multiple_of(input, values)


@builtin
def max_contiguous(input, values, _semantic=None):
    """
    Let the compiler know that the elements of :code:`input` along dimension
    ``d`` form contiguous groups of length ``values[d]``. ``values`` must have
    one entry per dimension of :code:`input` and each entry must be a power of
    two.

    A 1D array of ``N`` elements with contiguity ``C`` is viewed as ``N/C``
    runs of ``C`` integers each, where the integers in a run are
    sequentially contiguous. For example, if :code:`values` is ``[4]``, the
    array ``[0, 1, 2, 3, 8, 9, 10, 11]`` satisfies the hint because it
    consists of two runs of 4 contiguous values.

    Together with :func:`multiple_of`, this hint enables vectorized loads and
    stores of contiguous, aligned regions.
    """
    if isinstance(values, constexpr):
        values = [values]
    for i, d in enumerate(values):
        if not isinstance(d, constexpr):
            raise TypeError(f"values element {i} must have type `constexpr`")
        if not isinstance(d.value, int):
            raise TypeError(f"values element {i} must have type `constexpr[int]`, got `constexpr[{type(d.value)}]")
    values = [x.value for x in values]
    return _semantic.max_contiguous(input, values)


@builtin
def max_constancy(input, values, _semantic=None):
    """
    Let the compiler know that the elements of :code:`input` along dimension
    ``d`` form constant groups of length ``values[d]``. ``values`` must have
    one entry per dimension of :code:`input` and each entry must be a power of
    two.

    A 1D array of ``N`` elements with constancy ``C`` is viewed as ``N/C``
    runs of ``C`` identical values. For example, if :code:`values` is ``[4]``,
    the array ``[0, 0, 0, 0, 1, 1, 1, 1]`` satisfies the hint.
    """
    if isinstance(values, constexpr):
        values = [values]
    for i, d in enumerate(values):
        if not isinstance(d, constexpr):
            raise TypeError(f"values element {i} must have type `constexpr`")
        if not isinstance(d.value, int):
            raise TypeError(f"values element {i} must have type `constexpr[int]`, got `constexpr[{type(d.value)}]")
    values = [x.value for x in values]
    return _semantic.max_constancy(input, values)


@builtin
def assume(cond, _semantic=None):
    '''
    Allow compiler to assume the :code:`cond` is True.
    '''
    return _semantic.assume(_semantic.to_tensor(cond))


# -----------------------
# Debugging functions
# -----------------------


@builtin
def static_print(*values, sep: str = " ", end: str = "\n", file=None, flush=False, _semantic=None):
    '''
    Print the values at compile time.  The parameters are the same as the builtin :code:`print`.

    NOTE: Calling the Python builtin :code:`print` is not the same as calling this, it instead maps to :code:`device_print`,
    which has special requirements for the arguments.

    .. highlight:: python
    .. code-block:: python

        tl.static_print(f"BLOCK_SIZE={BLOCK_SIZE}")
    '''
    pass


@builtin
def static_assert(cond, msg="", _semantic=None):
    '''
    Assert the condition at compile time.  Does not require that the :code:`TRITON_DEBUG` environment variable
    is set.

    .. highlight:: python
    .. code-block:: python

        tl.static_assert(BLOCK_SIZE == 1024)
    '''
    pass


@builtin
def device_print(prefix, *args, hex=False, _semantic=None):
    '''
    Print the values at runtime from the device.  String formatting does not work for runtime values, so you should
    provide the values you want to print as arguments.  The first value must be a string, all following values must
    be scalars or tensors.

    Calling the Python builtin :code:`print` is the same as calling this function, and the requirements for the arguments will match
    this function (not the normal requirements for :code:`print`).

    .. highlight:: python
    .. code-block:: python

        tl.device_print("pid", pid)
        print("pid", pid)

    On CUDA, printfs are streamed through a buffer of limited size (on one host,
    we measured the default as 6912 KiB, but this may not be consistent across
    GPUs and CUDA versions).  If you notice some printfs are being dropped, you
    can increase the buffer size by calling

    .. highlight:: python
    .. code-block:: python

        triton.runtime.driver.active.utils.set_printf_fifo_size(size_bytes)

    CUDA may raise an error if you try to change this value after running a
    kernel that uses printfs.  The value set here may only affect the current
    device (so if you have multiple GPUs, you'd need to call it multiple times).

    :param prefix: a prefix to print before the values. This is required to be a string literal.
    :param args: the values to print. They can be any tensor or scalar.
    :param hex: print all values as hex instead of decimal
    '''
    import string
    prefix = _unwrap_if_constexpr(prefix)
    assert isinstance(prefix, str), f"{prefix} is not string"
    b_ascii = True
    for ch in prefix:
        if ch not in string.printable:
            b_ascii = False
            break
    assert b_ascii, f"{prefix} is not an ascii string"
    new_args = []
    for arg in args:
        new_args.append(_semantic.to_tensor(arg))
    return _semantic.device_print(prefix, new_args, hex)


@builtin
def device_assert(cond, msg="", mask=None, _semantic=None):
    '''
    Assert the condition at runtime from the device.  Requires that the environment variable :code:`TRITON_DEBUG`
    is set to a value besides :code:`0` in order for this to have any effect.

    Using the Python :code:`assert` statement is the same as calling this function, except that the second argument
    must be provided and must be a string, e.g. :code:`assert pid == 0, "pid != 0"`.  The environment variable must
    be set for this :code:`assert` statement to have any effect.

    .. highlight:: python
    .. code-block:: python

        tl.device_assert(pid == 0)
        assert pid == 0, f"pid != 0"

    :param cond: the condition to assert. This is required to be a boolean tensor.
    :param msg: the message to print if the assertion fails. This is required to be a string literal.
    '''
    msg = _unwrap_if_constexpr(msg)
    mask = _unwrap_if_constexpr(mask)
    if mask is not None:
        mask = _semantic.to_tensor(mask)
    return _semantic.device_assert(_semantic.to_tensor(cond), msg, mask)


@builtin
def inline_asm_elementwise(asm: str, constraints: str, args: Sequence, dtype: Union[dtype, Sequence[dtype]],
                           is_pure: bool, pack: int, _semantic=None):
    '''
        Execute inline assembly over a tensor.  Essentially, this is :code:`map`
        where the function is inline assembly.

        The input tensors :code:`args` are implicitly broadcasted to the same shape.

        :code:`dtype` can be a tuple of types, in which case the output is a
        tuple of tensors.

        Each invocation of the inline asm processes :code:`pack` elements at a
        time.  Exactly which set of inputs a block receives is unspecified.
        Input elements of size less than 4 bytes are packed into 4-byte
        registers.

        This op does not support empty :code:`dtype` -- the inline asm must
        return at least one tensor, even if you don't need it.  You can work
        around this by returning a dummy tensor of arbitrary type; it shouldn't
        cost you anything if you don't use it.

        Example using
        `PTX <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>`_
        assembly:

        .. highlight:: python
        .. code-block:: python

            @triton.jit
            def kernel(A, B, C, D, BLOCK: tl.constexpr):
                a = tl.load(A + tl.arange(0, BLOCK)) # uint8 tensor
                b = tl.load(B + tl.arange(0, BLOCK)) # float32 tensor

                # For each (a,b) in zip(a,b), perform the following:
                # - Let ai be `a` converted to int32.
                # - Let af be `a` converted to float.
                # - Let m be the max of ai and b.
                # - Return ai and mi.
                # Do the above 4 elements at a time.
                (c, d) = tl.inline_asm_elementwise(
                    asm="""
                    {
                        // Unpack `a` into `ai`.
                        .reg .b8 tmp<4>;
                        mov.b32 {tmp0, tmp1, tmp2, tmp3}, $8;
                        cvt.u32.u8 $0, tmp0;
                        cvt.u32.u8 $1, tmp1;
                        cvt.u32.u8 $2, tmp2;
                        cvt.u32.u8 $3, tmp3;
                    }
                    // Convert `ai` to float.
                    cvt.rn.f32.s32 $4, $0;
                    cvt.rn.f32.s32 $5, $1;
                    cvt.rn.f32.s32 $6, $2;
                    cvt.rn.f32.s32 $7, $3;
                    // Take max of `ai` and `b`.
                    max.f32 $4, $4, $9;
                    max.f32 $5, $5, $10;
                    max.f32 $6, $6, $11;
                    max.f32 $7, $7, $12;
                    """,
                    constraints=(
                        # 8 output registers, namely
                        #   $0=ai0, $1=ai1, $2=ai2, $3=ai3,
                        #   $4=m0,  $5=m1,  $6=m2,  $7=m3.
                        "=r,=r,=r,=r,=r,=r,=r,=r,"
                        # 5 input registers, namely
                        #   $8=ai,
                        #   $9=b0, $10=b1, $11=b2, $12=b3.
                        # The four elements from `a` are all packed into one register.
                        "r,r,r,r,r"),
                    args=[a, b],
                    dtype=(tl.int32, tl.float32),
                    is_pure=True,
                    pack=4,
                )
                tl.store(C + tl.arange(0, BLOCK), c)
                tl.store(D + tl.arange(0, BLOCK), d)

        :param asm: assembly to run.  Must match target's assembly format.
        :param constraints: asm constraints in
            `LLVM format <https://llvm.org/docs/LangRef.html#inline-asm-constraint-string>`_
        :param args: the input tensors, whose values are passed to the asm block
        :param dtype: the element type(s) of the returned tensor(s)
        :param is_pure: if true, the compiler assumes the asm block has no side-effects
        :param pack: the number of elements to be processed by one instance of inline assembly
        :return: one tensor or a tuple of tensors of the given dtypes
    '''
    asm = _unwrap_if_constexpr(asm)
    constraints = _unwrap_if_constexpr(constraints)
    pack = _unwrap_if_constexpr(pack)
    is_pure = _unwrap_if_constexpr(is_pure)

    # Wrap `dtype` in a tuple if it's not already.
    try:
        iter(dtype)  # type: ignore
        has_multiple_outputs = True
    except TypeError:
        has_multiple_outputs = False
        dtype = (dtype, )  # type: ignore

    dtype = typing.cast(Sequence[_DtypeClass], dtype)

    res_tys = dtype
    if dispatch_args := [_semantic.to_tensor(arg) for arg in args]:
        bin_op_type_checking = partial(
            _semantic.binary_op_type_checking_impl,
            arithmetic_check=False,
            allow_lhs_ptr=True,
            allow_rhs_ptr=True,
        )
        broadcast_arg = dispatch_args[0]
        # Get the broadcast shape over all the arguments
        for item in dispatch_args:
            _, broadcast_arg = bin_op_type_checking(item, broadcast_arg)
        if broadcast_arg.shape:
            # Change the shape of each argument based on the broadcast shape
            for i, item in enumerate(dispatch_args):
                dispatch_args[i], _ = bin_op_type_checking(item, broadcast_arg)
            res_tys = [broadcast_arg.type.with_element_ty(dt) for dt in dtype]
    handles = [t.handle for t in dispatch_args]
    builder = _semantic.builder
    call = builder.create_inline_asm(asm, constraints, handles, [ty.to_ir(builder) for ty in res_tys], is_pure, pack)

    if not has_multiple_outputs:
        return tensor(call.get_result(0), res_tys[0])
    return tuple(tensor(call.get_result(i), ty) for i, ty in enumerate(res_tys))


# -----------------------
# Iterators
# -----------------------


class static_range(base_value):
    """
    Iterator that counts upward forever.

    .. highlight:: python
    .. code-block:: python

        @triton.jit
        def kernel(...):
            for i in tl.static_range(10):
                ...
    :note: This is a special iterator used to implement similar semantics to Python's :code:`range` in the context of
        :code:`triton.jit` functions. In addition, it also guides the compiler to unroll the loop aggressively.
    :param arg1: the start value.
    :param arg2: the end value.
    :param step: the step value.
    """

    def __init__(self, arg1, arg2=None, step=None):
        assert isinstance(arg1, constexpr), f"{arg1} used as tl.static_range start value is not a constexpr"
        if step is None:
            self.step = constexpr(1)
        else:
            assert isinstance(step, constexpr), f"{step} used as tl.static_range step value is not a constexpr"
            self.step = step
        if arg2 is None:
            self.start = constexpr(0)
            self.end = arg1
        else:
            assert isinstance(arg2, constexpr), f"{arg2} used as tl.static_range end value is not a constexpr"
            self.start = arg1
            self.end = arg2

    def __iter__(self):
        raise RuntimeError("static_range can only be used in @triton.jit'd functions")

    def __next__(self):
        raise RuntimeError("static_range can only be used in @triton.jit'd functions")


class range(base_value):
    """
    Iterator that counts upward forever.

    .. highlight:: python
    .. code-block:: python

        @triton.jit
        def kernel(...):
            for i in tl.range(10, num_stages=3):
                ...
    :note: This is a special iterator used to implement similar semantics to Python's :code:`range` in the context of
        :code:`triton.jit` functions. In addition, it allows user to pass extra attributes to the compiler.
    :param arg1: the start value.
    :param arg2: the end value.
    :param step: the step value.
    :param num_stages: pipeline the loop into this many stages (so there are
        :code:`num_stages` iterations of the loop in flight at once).

        Note this is subtly different than passing :code:`num_stages` as a
        kernel argument.  The kernel argument only pipelines loads that feed
        into :code:`dot` operations, while this attribute tries to pipeline most
        (though not all) loads in this loop.
    :param loop_unroll_factor: Tells the Triton IR level loop unroller how many
        times to unroll a for loop that this range is used with. Less than 2 for
        this value implies no unrolling.
    :param disallow_acc_multi_buffer: If true, prevent the accumulator of the dot
        operation in the loop to be multi-buffered, if applicable.
    :param flatten: automatically flatten the loop nest starting at this loop to
        create a single flattened loop. The compiler will try to pipeline the
        flattened loop which can avoid stage stalling.
    :param warp_specialize: Enable automatic warp specialization on the loop.
        The compiler will attempt to partition memory, MMA, and vector
        operations in the loop into separate async partitions. This will
        increase the total number of warps required by the kernel.
    :param disable_licm: Tells the compiler it shouldn't hoist loop invariant
        code outside the loop. This is often useful to avoid creating long liveranges
        within a loop.

        Note that warp specialization is only supported on Blackwell GPUs and
        only works on simple matmul loops. Support for arbitrary loops will be
        expanded over time.
    """

    def __init__(self, arg1, arg2=None, step=None, num_stages=None, loop_unroll_factor=None,
                 disallow_acc_multi_buffer=False, flatten=False, warp_specialize=False, disable_licm=False):
        if step is None:
            self.step = constexpr(1)
        else:
            self.step = step
        if arg2 is None:
            self.start = constexpr(0)
            self.end = arg1
        else:
            self.start = arg1
            self.end = arg2
        self.num_stages = num_stages
        self.loop_unroll_factor = loop_unroll_factor
        self.disallow_acc_multi_buffer = disallow_acc_multi_buffer
        self.flatten = flatten
        self.warp_specialize = warp_specialize
        self.disable_licm = disable_licm

    def __iter__(self):
        raise RuntimeError("tl.range can only be used in @triton.jit'd functions")

    def __next__(self):
        raise RuntimeError("tl.range can only be used in @triton.jit'd functions")


class condition(base_value):
    """
    While loop condition wrapper.

    .. highlight:: python
    .. code-block:: python

        @triton.jit
        def kernel(...):
            while tl.condition(c, disable_licm)
                ...
    :note: This is a special wrapper used to annotate while loops in the context of
        :code:`triton.jit` functions. It allows user to pass extra attributes to the compiler.
    :param disable_licm: Tells the compiler it shouldn't hoist loop invariant
        code outside the loop. This is often useful to avoid creating long liveranges
        within a loop.
    """

    def __init__(self, arg1, disable_licm=False):
        self.condition = arg1
        self.disable_licm = disable_licm


# -----------------------
# Extern functions
# -----------------------


def dispatch(func, lib_name: str, lib_path: str, args: list, arg_type_symbol_dict: dict, ret_type: dtype, is_pure: bool,
             _semantic):
    '''
        Dispatch a function to a library
        :param func: the function to dispatch
        :param lib_name: the name of the library
        :param lib_path: the path of the library
        :param args: the arguments of the function
        :param arg_type_symbol_dict: the type of the arguments
        :param ret_type: the type of the return value
        :return: the return value of the function
    '''
    if len(arg_type_symbol_dict) == 0:
        raise ValueError("arg_type_symbol_dict is empty")

    num_args = len(list(arg_type_symbol_dict.keys())[0])
    if len(args) != num_args:
        raise ValueError(f"length of input args does not match."
                         f"Expect {len(args)}, got {num_args}")

    arg_types = []
    arg_list = []
    for arg in args:
        if isinstance(arg, tensor):
            arg_types.append(arg.dtype)
            arg_list.append(arg.handle)
        else:
            arg_types.append(type(arg))
            arg_list.append(arg)
    arg_types = tuple(arg_types)

    if arg_types not in arg_type_symbol_dict:
        raise ValueError(f"input arg type does not match."
                         f"Expect one of {arg_type_symbol_dict.keys()}, got {arg_types}")
    else:
        symbol = arg_type_symbol_dict[arg_types][0]
        builder = _semantic.builder
        return tensor(func(lib_name, lib_path, symbol, arg_list, ret_type.to_ir(builder), is_pure), ret_type)


@builtin
def extern_elementwise(lib_name: str, lib_path: str, args: list, arg_type_symbol_dict: dict, is_pure: bool,
                       _semantic=None):
    '''
        Dispatch an elementwise function to a library
        :param lib_name: the name of the library
        :param lib_path: the path of the library
        :param args: the arguments of the function
        :param arg_type_symbol_dict: the type of the arguments
        :param is_pure: whether the function is pure
        :return: the return value of the function
    '''
    dispatch_args = args.copy()
    all_scalar = True
    arg_types = []
    for i in builtins.range(len(dispatch_args)):
        dispatch_args[i] = _semantic.to_tensor(dispatch_args[i])
        arg_types.append(dispatch_args[i].dtype)
        if dispatch_args[i].type.is_block():
            all_scalar = False

    arg_types = tuple(arg_types)
    ret_type = arg_type_symbol_dict[arg_types][1]
    if len(arg_types) > 0:
        arithmetic_check = True
        # If there's a type tuple that is not supported by the library, we will do arithmetic check
        if arg_types in arg_type_symbol_dict:
            arithmetic_check = False
        broadcast_arg = dispatch_args[0]
        # Get the broadcast shape over all the arguments
        for item in dispatch_args:
            _, broadcast_arg = _semantic.binary_op_type_checking_impl(item, broadcast_arg,
                                                                      arithmetic_check=arithmetic_check)
        # Change the shape of each argument based on the broadcast shape
        for i in builtins.range(len(dispatch_args)):
            dispatch_args[i], _ = _semantic.binary_op_type_checking_impl(dispatch_args[i], broadcast_arg,
                                                                         arithmetic_check=arithmetic_check)
        if not all_scalar:
            ret_type = broadcast_arg.type.with_element_ty(ret_type)
    func = _semantic.builder.create_extern_elementwise
    return dispatch(func, lib_name, lib_path, dispatch_args, arg_type_symbol_dict, ret_type, is_pure, _semantic)


def binary_op_type_legalization(lhs, rhs, semantic):
    '''
        Convert both operands to a single common type
        :param lhs: the left operand
        :param rhs: the right operand
        :param builder: the builder
    '''
    return semantic.binary_op_type_checking_impl(lhs, rhs)


def extern(fn):
    """A decorator for external functions."""
    return builtin(fn)


_NOTHING = object()


def is_negative_zero(x):
    return x == 0.0 and math.copysign(1.0, x) < 0


@builtin
def builtin_max(*args, propagate_nan=_NOTHING, _semantic=None):
    args = _unwrap_if_constexpr(args)
    is_constexpr = all(not isinstance(x, base_value) for x in args)
    if is_constexpr:
        assert propagate_nan is _NOTHING, "propagate_nan is not supported on builtin max"
        assert not any(math.isnan(x) for x in args), "constexpr max does not support NaN values"
        assert not any(is_negative_zero(x) for x in args), "constexpr max does not support negative zero"
        return constexpr(builtins.max(_unwrap_if_constexpr(args)))

    if propagate_nan is _NOTHING:
        propagate_nan = PropagateNan.NONE
    else:
        warn("passing propagate_nan to builtin max is deprecated, use tl.minimum instead")

    assert len(args) >= 2, "min requires at least 2 values"
    max_val = args[0]
    for arg in args[1:]:
        max_val = maximum(max_val, arg, propagate_nan=propagate_nan, _semantic=_semantic)
    if max_val.type.is_block():
        warn("builtin max on non-scalar tensor values is deprecated, use tl.maximum instead")
    return max_val


@builtin
def builtin_min(*args, propagate_nan=_NOTHING, _semantic=None):
    args = _unwrap_if_constexpr(args)
    is_constexpr = all(not isinstance(x, base_value) for x in args)
    if is_constexpr:
        assert propagate_nan is _NOTHING, "propagate_nan is not supported on builtin min"
        assert not any(math.isnan(x) for x in args), "constexpr min does not support NaN values"
        assert not any(is_negative_zero(x) for x in args), "constexpr min does not support negative zero"
        return constexpr(builtins.min(_unwrap_if_constexpr(args)))

    if propagate_nan is _NOTHING:
        propagate_nan = PropagateNan.NONE
    else:
        warn("passing propagate_nan to builtin min is deprecated, use tl.minimum instead")

    assert len(args) >= 2, "min requires at least 2 values"
    min_val = args[0]
    for arg in args[1:]:
        min_val = minimum(min_val, arg, propagate_nan=propagate_nan, _semantic=_semantic)
    if min_val.type.is_block():
        warn("builtin min on non-scalar tensor values is deprecated, use tl.minimum instead")
    return min_val
<|endoftext|>from __future__ import annotations

from ..runtime.jit import jit, constexpr_function
from . import core
from . import math

# constexpr utilities


@constexpr_function
def _log2(i):
    log2 = 0
    n = i
    while n > 1:
        n >>= 1
        log2 += 1
    return log2


@constexpr_function
def _is_power_of_two(i):
    return (i & (i - 1)) == 0 and i != 0


_get_int_dtype = constexpr_function(core.get_int_dtype)

# -----------------------
# Standard library
# -----------------------


@core._tensor_member_fn
@jit
def cdiv(x, div):
    """
    Computes the ceiling division of :code:`x` by :code:`div`

    :param x: the input number
    :type x: Block
    :param div: the divisor
    :type div: Block
    """
    return (x + (div - 1)) // div


@core._tensor_member_fn
@jit
@math._add_math_1arg_docstr("sigmoid")
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


@core._tensor_member_fn
@jit
@math._add_math_1arg_docstr("softmax")
def softmax(x, dim=None, keep_dims=False, ieee_rounding=False):
    if dim is None:
        _dim: core.constexpr = 0
    else:
        _dim: core.constexpr = dim
    z = x - max(x, _dim, keep_dims=keep_dims)
    num = math.exp(z)
    den = sum(num, _dim, keep_dims=keep_dims)
    return math.fdiv(num, den, ieee_rounding)


@core._tensor_member_fn
@jit
def ravel(x, can_reorder=False):
    """
    Returns a contiguous flattened view of :code:`x`.

    :param x: the input tensor
    :type x: Block
    """
    return core.reshape(x, [x.numel], can_reorder=can_reorder)


@jit
def swizzle2d(i, j, size_i, size_j, size_g):
    """
    Transforms the indices of a row-major `size_i * size_j` matrix into
    the indices of a column-major matrix for each group of `size_g` rows.

    For example, for :code:`size_i = size_j = 4` and :code:`size_g = 2`, it will
    transform ::

        [[0 , 1 , 2 , 3 ],
         [4 , 5 , 6 , 7 ],
         [8 , 9 , 10, 11],
         [12, 13, 14, 15]]

    into ::

        [[0, 2,  4 , 6 ],
         [1, 3,  5 , 7 ],
         [8, 10, 12, 14],
         [9, 11, 13, 15]]
    """
    # "unrolled index in array"
    ij = i * size_j + j
    # number of elements in `size_g` groups
    # of `size_j` columns
    size_gj = size_g * size_j
    # index of the group in which (i,j) is
    group_id = ij // size_gj
    # row-index of the first element of this group
    off_i = group_id * size_g
    # last group may have fewer rows
    size_g = core.minimum(size_i - off_i, size_g)
    # linear index with respect to the first element in this group
    ij = ij % size_gj
    # new row and column indices
    new_i = off_i + ij % size_g
    new_j = ij // size_g
    return new_i, new_j


@jit
def zeros(shape, dtype):
    """
    Returns a tensor filled with the scalar value 0 for the given :code:`shape` and :code:`dtype`.

    :param shape: Shape of the new array, e.g., (8, 16) or (8, )
    :type shape: tuple of ints
    :param dtype: Data-type of the new array, e.g., :code:`tl.float16`
    :type dtype: DType
    """
    return core.full(shape, 0, dtype)


@jit
def zeros_like(input):
    """
    Returns a tensor of zeros with the same shape and type as a given tensor.

    :param input: input tensor
    :type input: Tensor
    """
    return zeros(input.shape, input.dtype)


# max and argmax


@jit
def _argmax_combine(value1, index1, value2, index2, tie_break_left):
    if tie_break_left:
        tie = value1 == value2 and index1 < index2
    else:
        tie = False
    gt = value1 > value2 or tie
    v_ret = core.where(gt, value1, value2)
    i_ret = core.where(gt, index1, index2)
    return v_ret, i_ret


@jit
def _argmax_combine_tie_break_left(value1, index1, value2, index2):
    return _argmax_combine(value1, index1, value2, index2, True)


@jit
def _argmax_combine_tie_break_fast(value1, index1, value2, index2):
    return _argmax_combine(value1, index1, value2, index2, False)


@jit
def _elementwise_max(a, b):
    return core.maximum(a, b)


@core._tensor_member_fn
@jit
@core._add_reduction_docstr("maximum", return_indices_arg="return_indices",
                            tie_break_arg="return_indices_tie_break_left")
def max(input, axis=None, return_indices=False, return_indices_tie_break_left=True, keep_dims=False):
    input = core._promote_bfloat16_to_float32(input)
    if return_indices:
        if return_indices_tie_break_left:
            return core._reduce_with_indices(input, axis, _argmax_combine_tie_break_left, keep_dims=keep_dims)
        else:
            return core._reduce_with_indices(input, axis, _argmax_combine_tie_break_fast, keep_dims=keep_dims)
    else:
        if core.constexpr(input.dtype.primitive_bitwidth) < core.constexpr(32):
            if core.constexpr(input.dtype.is_floating()):
                input = input.to(core.float32)
            else:
                assert input.dtype.is_int(), "Expecting input to be integer type"
                input = input.to(core.int32)
        return core.reduce(input, axis, _elementwise_max, keep_dims=keep_dims)


@core._tensor_member_fn
@jit
@core._add_reduction_docstr("maximum index", tie_break_arg="tie_break_left")
def argmax(input, axis, tie_break_left=True, keep_dims=False):
    (_, ret) = max(input, axis, return_indices=True, return_indices_tie_break_left=tie_break_left, keep_dims=keep_dims)
    return ret


# min and argmin


@jit
def _argmin_combine(value1, index1, value2, index2, tie_break_left):
    if tie_break_left:
        tie = value1 == value2 and index1 < index2
    else:
        tie = False
    lt = value1 < value2 or tie
    value_ret = core.where(lt, value1, value2)
    index_ret = core.where(lt, index1, index2)
    return value_ret, index_ret


@jit
def _argmin_combine_tie_break_left(value1, index1, value2, index2):
    return _argmin_combine(value1, index1, value2, index2, True)


@jit
def _argmin_combine_tie_break_fast(value1, index1, value2, index2):
    return _argmin_combine(value1, index1, value2, index2, False)


@jit
def _elementwise_min(a, b):
    return core.minimum(a, b)


@core._tensor_member_fn
@jit
@core._add_reduction_docstr("minimum", return_indices_arg="return_indices",
                            tie_break_arg="return_indices_tie_break_left")
def min(input, axis=None, return_indices=False, return_indices_tie_break_left=True, keep_dims=False):
    input = core._promote_bfloat16_to_float32(input)
    if return_indices:
        if return_indices_tie_break_left:
            return core._reduce_with_indices(input, axis, _argmin_combine_tie_break_left, keep_dims=keep_dims)
        else:
            return core._reduce_with_indices(input, axis, _argmin_combine_tie_break_fast, keep_dims=keep_dims)
    else:
        if core.constexpr(input.dtype.primitive_bitwidth) < 32:
            if core.constexpr(input.dtype.is_floating()):
                input = input.to(core.float32)
            else:
                assert input.dtype.is_int(), "Expecting input to be integer type"
                input = input.to(core.int32)
        return core.reduce(input, axis, _elementwise_min, keep_dims=keep_dims)


@core._tensor_member_fn
@jit
@core._add_reduction_docstr("minimum index", tie_break_arg="tie_break_left")
def argmin(input, axis, tie_break_left=True, keep_dims=False):
    _, ret = min(input, axis, return_indices=True, return_indices_tie_break_left=tie_break_left, keep_dims=keep_dims)
    return ret


@jit
def _sum_combine(a, b):
    return a + b


# sum


@constexpr_function
def _pick_sum_dtype(in_dtype, dtype):
    if dtype is not None:
        return dtype

    # For integer bitwidths less than 32, pick int32 with the same sign to
    # avoid overflow.
    out_dtype = None
    if in_dtype.is_int_signed():
        out_dtype = core.int32 if in_dtype.int_bitwidth < 32 else None
    elif in_dtype.is_int_unsigned():
        out_dtype = core.uint32 if in_dtype.int_bitwidth < 32 else None
    return out_dtype


@core._tensor_member_fn
@jit
@core._add_reduction_docstr("sum", dtype_arg="dtype")
def sum(input, axis=None, keep_dims=False, dtype: core.constexpr = None):
    # Pick a default dtype for the reduction if one was not specified.
    out_dtype: core.constexpr = _pick_sum_dtype(input.dtype, dtype)

    if out_dtype is not None:
        input = input.to(out_dtype)
    return core.reduce(input, axis, _sum_combine, keep_dims=keep_dims)


@jit
def _xor_combine(a, b):
    return a ^ b


# xor sum


@core._tensor_member_fn
@jit
@core._add_reduction_docstr("xor sum")
def xor_sum(input, axis=None, keep_dims=False):
    core.static_assert(input.type.scalar.is_int(), "xor_sum only supported for integers")
    return core.reduce(input, axis, _xor_combine, keep_dims=keep_dims)


# or reduction


@jit
def _or_combine(x, y):
    return x | y


@core._tensor_member_fn
@jit
@core._add_reduction_docstr("reduce_or")
def reduce_or(input, axis, keep_dims=False):
    core.static_assert(input.type.scalar.is_int(), "reduce_or only supported for integers")
    return core.reduce(input, axis, _or_combine, keep_dims=keep_dims)


# cumsum


@core._tensor_member_fn
@jit
@core._add_scan_docstr("cumsum", dtype_arg="dtype")
def cumsum(input, axis=0, reverse=False, dtype: core.constexpr = None):
    # todo rename this to a generic function name

    input = core._promote_bfloat16_to_float32(input)
    out_dtype: core.constexpr = _pick_sum_dtype(input.dtype, dtype)

    if out_dtype is not None:
        input = input.to(out_dtype)

    return core.associative_scan(input, axis, _sum_combine, reverse)


# cumprod


@jit
def _prod_combine(a, b):
    return a * b


@core._tensor_member_fn
@jit
@core._add_scan_docstr("cumprod")
def cumprod(input, axis=0, reverse=False):
    # todo rename this to a generic function name
    input = core._promote_bfloat16_to_float32(input)
    return core.associative_scan(input, axis, _prod_combine, reverse)


# sort


@jit
def _indicator(n_dims: core.constexpr, j: core.constexpr):
    ar = core.arange(0, 2)
    ar = core.reshape(ar, [1] * (n_dims - j - 1) + [2] + [1] * j)
    return ar


@jit
def _compare_and_swap(x, flip, i: core.constexpr):
    # compare-and-swap on the ith *innermost* dimension
    n_dims: core.constexpr = _log2(x.numel)

    # flip along middle dimension (the bitwise XORs will be optimised away):
    idtype = _get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)
    ix = x.to(idtype, bitcast=True)
    iy = ix ^ xor_sum(ix, n_dims - 1 - i, True)
    y = iy.to(x.dtype, bitcast=True)

    # determines whether we are in the right (rather than left) position along the axis:
    is_right = _indicator(n_dims, i)

    # conditional swap:
    ret = core.where((x > y) != (flip ^ is_right), y, x)
    return ret


@jit
def _bitonic_merge_hypercube(x, stage: core.constexpr, order: core.constexpr):
    '''
    order_type 0 == ascending
    order_type 1 == descending
    order_type 2 == alternating
    '''
    # flip denotes whether to re-arrange sub-sequences of elements in ascending or
    # descending order.
    # if flip = 00000000... then all elements will be re-arranged ascendingly at this stage
    # if flip = 00110011... then all the elements will be re-arranged alternatingly (with
    # a stride of 2) at this stage
    if order == 2:
        flip = _indicator(_log2(x.numel), stage)
    else:
        flip = order
    # perform `stage` rounds of `compare-and-swap`
    for i in core.static_range(stage):
        x = _compare_and_swap(x, flip, stage - 1 - i)
    return x


@jit
def _bitonic_merge(x, stage: core.constexpr, order: core.constexpr, n_dims: core.constexpr):
    h = core.reshape(x, [2] * _log2(x.numel))
    h = _bitonic_merge_hypercube(h, stage, order)
    x = core.reshape(h, x.shape)
    return x


@jit
def sort_impl(x, k: core.constexpr = None, dim: core.constexpr = None, descending: core.constexpr = core.CONSTEXPR_0):
    """
    Sorts a tensor along a specified dimension.

    :param x: The input tensor to be sorted.
    :type x: Tensor
    :param dim: The dimension along which to sort the tensor. If None, the tensor is sorted along the last dimension. Currently, only sorting along the last dimension is supported.
    :type dim: int, optional
    :param k: the number of top elements to select. If none, assume k = x.shape[dim]
    :type k: int, optional
    :param descending: If set to True, the tensor is sorted in descending order. If set to False, the tensor is sorted in ascending order.
    :type descending: bool, optional
    """
    # handle default dimension or check that it is the most minor dim
    _dim: core.constexpr = len(x.shape) - 1 if dim is None else dim
    core.static_assert(_dim == len(x.shape) - 1, "only minor dimension is currently supported")

    log_n: core.constexpr = _log2(x.shape[_dim])
    log_k: core.constexpr = log_n if k is None else _log2(k)

    n_dims: core.constexpr = _log2(x.numel)

    # reshape to hypercube:
    h = core.reshape(x, [2] * n_dims if n_dims else [1])

    # run first log_k bitonic sort iterations:
    for i in core.static_range(1, log_k + 1):
        h = _bitonic_merge_hypercube(h, i, 2 if i < log_n else descending)

    # select top k elements using bitonic top-k
    # https://www.doc.ic.ac.uk/~hlgr/pdfs/MassivelyParallelTopK.pdf
    for i in core.static_range(log_k + 1, log_n + 1):
        h = max(h, axis=(_log2(h.numel) - 1 - log_k)) if descending else min(h, axis=(_log2(h.numel) - 1 - log_k))
        h = _bitonic_merge_hypercube(h, log_k, 2 if i < log_n else descending)

    # reshape back:
    x = core.reshape(h, x.shape[:-1] + [2**log_k])
    return x


@jit
def sort(x, dim: core.constexpr = None, descending: core.constexpr = core.CONSTEXPR_0):
    return sort_impl(x, dim=dim, descending=descending)


@jit
def topk(x, k: core.constexpr, dim: core.constexpr = None, descending: core.constexpr = True):
    """
    Returns the k largest (or smallest) elements of the input tensor along the specified dimension.

    The elements are returned in sorted order (largest first).

    :param x: The input tensor.
    :type x: Tensor
    :param k: The number of top elements to return. Must be a power of two.
    :type k: int
    :param dim: The dimension along which to find the top k elements.
                If None, uses the last dimension. Currently only the last dimension is supported.
    :type dim: int, optional
    :param descending: If set to True, returns k largest elements. If set to False, returns k smallest elements.
    :type descending: bool, optional
    :return: A tensor containing the k largest elements along the specified dimension.
    :rtype: Tensor

    Example::

        # Get top 4 elements from a 1D tensor
        x = tl.arange(0, 16)
        top4 = tl.topk(x, 4)  # Returns [15, 14, 13, 12]
    """
    return sort_impl(x, k=k, dim=dim, descending=descending)


@jit
def bitonic_merge(x, dim: core.constexpr = None, descending: core.constexpr = core.CONSTEXPR_0):
    # handle default dimension or check that it is the most minor dim
    _dim: core.constexpr = len(x.shape) - 1 if dim is None else dim
    core.static_assert(_dim == len(x.shape) - 1, "only minor dimension is currently supported")
    n_dims: core.constexpr = _log2(x.shape[-1])
    return _bitonic_merge(x, n_dims, descending, n_dims)


@constexpr_function
def _get_flip_dim(dim, shape):
    if dim is None:
        dim = len(shape) - 1
    if dim < 0:  # flip doesn't work if dim < 0 because the xor-swap for loop will start/end at the wrong index
        dim += len(shape)
    return dim


@core._tensor_member_fn
@jit
def flip(x, dim=None):
    """
    Flips a tensor `x` along the dimension `dim`.

    :param x: the first input tensor
    :type x: Block
    :param dim: the dimension to flip along
    :type dim: int
    """
    core.static_assert(-len(x.shape) <= dim and dim < len(x.shape))
    _dim: core.constexpr = _get_flip_dim(dim, x.shape)
    core.static_assert(_is_power_of_two(x.shape[_dim]))
    steps: core.constexpr = _log2(x.shape[_dim])

    # reshape the swap dimension to (2, 2, ..., 2)
    idtype = _get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)
    y = core.reshape(x.to(idtype, bitcast=True), x.shape[:_dim] + [2] * steps + x.shape[_dim + 1:])
    for i in core.static_range(steps):
        y = y ^ xor_sum(y, _dim + i, True)
    x = core.reshape(y, x.shape).to(x.dtype, bitcast=True)
    return x


@jit
def interleave(a, b):
    """
    Interleaves the values of two tensors along their last dimension. The two tensors must have the same shape.
    Equivalent to `tl.join(a, b).reshape(a.shape[:-1] + [2 * a.shape[-1]])`

    :param a: The first input tensor.
    :type a: Tensor
    :param b: The second input tensor.
    :type b: Tensor
    """
    c = core.join(a, b)

    if len(c.shape) == 1:
        # We must have interleaved two scalars.
        return c
    else:
        # This `else` is necessary because Triton's AST parser doesn't
        # understand that if we take the `if` above we definitely don't run this
        # `else`.
        return core.reshape(c, c.shape[:-2] + [2 * c.shape[-2]])


@jit
def squeeze(x, dim: core.constexpr):
    core.static_assert(x.shape[dim] == 1)
    return x.reshape(x.shape[:dim] + x.shape[dim + 1:])


@jit
def unsqueeze(x, dim: core.constexpr):
    return x.reshape(x.shape[:dim] + (1, ) + x.shape[dim:])
<|endoftext|>def clz(arg0):
    ...


def popc(arg0):
    ...


def byte_perm(arg0, arg1, arg2):
    ...


def mulhi(arg0, arg1):
    ...


def mul24(arg0, arg1):
    ...


def brev(arg0):
    ...


def sad(arg0, arg1, arg2):
    ...


def abs(arg0):
    ...


def floor(arg0):
    ...


def rcp64h(arg0):
    ...


def rsqrt(arg0):
    ...


def ceil(arg0):
    ...


def trunc(arg0):
    ...


def exp2(arg0):
    ...


def saturatef(arg0):
    ...


def fma_rn(arg0, arg1, arg2):
    ...


def fma_rz(arg0, arg1, arg2):
    ...


def fma_rd(arg0, arg1, arg2):
    ...


def fma_ru(arg0, arg1, arg2):
    ...


def fast_dividef(arg0, arg1):
    ...


def div_rn(arg0, arg1):
    ...


def div_rz(arg0, arg1):
    ...


def div_rd(arg0, arg1):
    ...


def div_ru(arg0, arg1):
    ...


def rcp_rn(arg0):
    ...


def rcp_rz(arg0):
    ...


def rcp_rd(arg0):
    ...


def rcp_ru(arg0):
    ...


def sqrt_rn(arg0):
    ...


def sqrt_rz(arg0):
    ...


def sqrt_rd(arg0):
    ...


def sqrt_ru(arg0):
    ...


def sqrt(arg0):
    ...


def add_rn(arg0, arg1):
    ...


def add_rz(arg0, arg1):
    ...


def add_rd(arg0, arg1):
    ...


def add_ru(arg0, arg1):
    ...


def mul_rn(arg0, arg1):
    ...


def mul_rz(arg0, arg1):
    ...


def mul_rd(arg0, arg1):
    ...


def mul_ru(arg0, arg1):
    ...


def double2float_rn(arg0):
    ...


def double2float_rz(arg0):
    ...


def double2float_rd(arg0):
    ...


def double2float_ru(arg0):
    ...


def double2int_rn(arg0):
    ...


def double2int_rz(arg0):
    ...


def double2int_rd(arg0):
    ...


def double2int_ru(arg0):
    ...


def double2uint_rn(arg0):
    ...


def double2uint_rz(arg0):
    ...


def double2uint_rd(arg0):
    ...


def double2uint_ru(arg0):
    ...


def int2double_rn(arg0):
    ...


def uint2double_rn(arg0):
    ...


def float2int_rn(arg0):
    ...


def float2int_rz(arg0):
    ...


def float2int_rd(arg0):
    ...


def float2int_ru(arg0):
    ...


def float2uint_rn(arg0):
    ...


def float2uint_rz(arg0):
    ...


def float2uint_rd(arg0):
    ...


def float2uint_ru(arg0):
    ...


def int2float_rn(arg0):
    ...


def int2float_rz(arg0):
    ...


def int2float_rd(arg0):
    ...


def int2float_ru(arg0):
    ...


def uint2float_rn(arg0):
    ...


def uint2float_rz(arg0):
    ...


def uint2float_rd(arg0):
    ...


def uint2float_ru(arg0):
    ...


def hiloint2double(arg0, arg1):
    ...


def double2loint(arg0):
    ...


def double2hiint(arg0):
    ...


def float2ll_rn(arg0):
    ...


def float2ll_rz(arg0):
    ...


def float2ll_rd(arg0):
    ...


def float2ll_ru(arg0):
    ...


def float2ull_rn(arg0):
    ...


def float2ull_rz(arg0):
    ...


def float2ull_rd(arg0):
    ...


def float2ull_ru(arg0):
    ...


def double2ll_rn(arg0):
    ...


def double2ll_rz(arg0):
    ...


def double2ll_rd(arg0):
    ...


def double2ll_ru(arg0):
    ...


def double2ull_rn(arg0):
    ...


def double2ull_rz(arg0):
    ...


def double2ull_rd(arg0):
    ...


def double2ull_ru(arg0):
    ...


def ll2float_rn(arg0):
    ...


def ll2float_rz(arg0):
    ...


def ll2float_rd(arg0):
    ...


def ll2float_ru(arg0):
    ...


def ull2float_rn(arg0):
    ...


def ull2float_rz(arg0):
    ...


def ull2float_rd(arg0):
    ...


def ull2float_ru(arg0):
    ...


def ll2double_rn(arg0):
    ...


def ll2double_rz(arg0):
    ...


def ll2double_rd(arg0):
    ...


def ll2double_ru(arg0):
    ...


def ull2double_rn(arg0):
    ...


def ull2double_rz(arg0):
    ...


def ull2double_rd(arg0):
    ...


def ull2double_ru(arg0):
    ...


def int_as_float(arg0):
    ...


def float_as_int(arg0):
    ...


def uint_as_float(arg0):
    ...


def float_as_uint(arg0):
    ...


def longlong_as_double(arg0):
    ...


def double_as_longlong(arg0):
    ...


def fast_sinf(arg0):
    ...


def fast_cosf(arg0):
    ...


def fast_log2f(arg0):
    ...


def fast_logf(arg0):
    ...


def fast_expf(arg0):
    ...


def fast_tanhf(arg0):
    ...


def fast_tanf(arg0):
    ...


def fast_exp10f(arg0):
    ...


def fast_log10f(arg0):
    ...


def fast_powf(arg0, arg1):
    ...


def hadd(arg0, arg1):
    ...


def rhadd(arg0, arg1):
    ...


def sub_rn(arg0, arg1):
    ...


def sub_rz(arg0, arg1):
    ...


def sub_rd(arg0, arg1):
    ...


def sub_ru(arg0, arg1):
    ...


def rsqrt_rn(arg0):
    ...


def ffs(arg0):
    ...


def rint(arg0):
    ...


def llrint(arg0):
    ...


def nearbyint(arg0):
    ...


def isnan(arg0):
    ...


def signbit(arg0):
    ...


def copysign(arg0, arg1):
    ...


def finitef(arg0):
    ...


def isinf(arg0):
    ...


def nextafter(arg0, arg1):
    ...


def sin(arg0):
    ...


def cos(arg0):
    ...


def sinpi(arg0):
    ...


def cospi(arg0):
    ...


def tan(arg0):
    ...


def log2(arg0):
    ...


def exp(arg0):
    ...


def exp10(arg0):
    ...


def cosh(arg0):
    ...


def sinh(arg0):
    ...


def tanh(arg0):
    ...


def atan2(arg0, arg1):
    ...


def atan(arg0):
    ...


def asin(arg0):
    ...


def acos(arg0):
    ...


def log(arg0):
    ...


def log10(arg0):
    ...


def log1p(arg0):
    ...


def acosh(arg0):
    ...


def asinh(arg0):
    ...


def atanh(arg0):
    ...


def expm1(arg0):
    ...


def hypot(arg0, arg1):
    ...


def rhypot(arg0, arg1):
    ...


def norm3d(arg0, arg1, arg2):
    ...


def rnorm3d(arg0, arg1, arg2):
    ...


def norm4d(arg0, arg1, arg2, arg3):
    ...


def rnorm4d(arg0, arg1, arg2, arg3):
    ...


def cbrt(arg0):
    ...


def rcbrt(arg0):
    ...


def j0(arg0):
    ...


def j1(arg0):
    ...


def y0(arg0):
    ...


def y1(arg0):
    ...


def yn(arg0, arg1):
    ...


def jn(arg0, arg1):
    ...


def cyl_bessel_i0(arg0):
    ...


def cyl_bessel_i1(arg0):
    ...


def erf(arg0):
    ...


def erfinv(arg0):
    ...


def erfc(arg0):
    ...


def erfcx(arg0):
    ...


def erfcinv(arg0):
    ...


def normcdfinv(arg0):
    ...


def normcdf(arg0):
    ...


def lgamma(arg0):
    ...


def ldexp(arg0, arg1):
    ...


def scalbn(arg0, arg1):
    ...


def fmod(arg0, arg1):
    ...


def remainder(arg0, arg1):
    ...


def fma(arg0, arg1, arg2):
    ...


def pow(arg0, arg1):
    ...


def tgamma(arg0):
    ...


def round(arg0):
    ...


def llround(arg0):
    ...


def fdim(arg0, arg1):
    ...


def ilogb(arg0):
    ...


def logb(arg0):
    ...


def isfinited(arg0):
    ...
<|endoftext|>import pkgutil
from importlib.util import module_from_spec
from sys import modules

_backends = []
for module_finder, module_name, is_pkg in pkgutil.iter_modules(
        __path__,
        prefix=__name__ + ".",
):
    # skip .py files (like libdevice.py)
    if not is_pkg:
        continue

    # import backends (like cuda and hip) that are included during setup.py
    spec = module_finder.find_spec(module_name)
    if spec is None or spec.loader is None:
        continue
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    _backends.append(module_name)
    modules[module_name] = module

__all__ = _backends

del _backends
<|endoftext|>from triton.language import core


@core.extern
def globaltimer(_semantic=None):
    return core.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [], dtype=core.int64, is_pure=False, pack=1,
                                       _semantic=_semantic)


@core.extern
def smid(_semantic=None):
    return core.inline_asm_elementwise("mov.u32 $0, %smid;", "=r", [], dtype=core.int32, is_pure=True, pack=1,
                                       _semantic=_semantic)


@core.builtin
def num_threads(_semantic=None):
    return core.constexpr(_semantic.builder.options.num_warps * 32)


@core.builtin
def num_warps(_semantic=None):
    return core.constexpr(_semantic.builder.options.num_warps)


# ----- FP8E4M3B15 ------
# This data-type is a variant of the standard FP8E4M3 format.
# It was designed for fast software conversion to FP16 on
# nvidia GPUs that do not support it natively.
# This is the same format as FP8E4M3Nv, but:
#   - the exponent bias is 15 instead of 7
#   - 0xff and 0x7f are mapped to +-1.750 instead of +-nan
@core.builtin
def convert_fp8e4b15_to_float16(arg, _semantic=None):
    return core.inline_asm_elementwise(
        "{                                      \n"
        ".reg .b32 a<2>, b<2>;                  \n"
        "prmt.b32 a0, 0, $2, 0x5746;            \n"
        "and.b32 b0, a0, 0x7f007f00;            \n"
        "and.b32 b1, a0, 0x00ff00ff;            \n"
        "and.b32 a1, a0, 0x00800080;            \n"
        "shr.b32  b0, b0, 1;                    \n"
        "add.u32 b1, b1, a1;                    \n"
        "lop3.b32 $0, b0, 0x80008000, a0, 0xf8; \n"
        "shl.b32 $1, b1, 7;                     \n"
        "}                                      \n", "=r,=r,r", [arg], dtype=core.float16, is_pure=True, pack=4,
        _semantic=_semantic)


@core.builtin
def convert_float16_to_fp8e4b15(arg, has_minx2, _semantic=None):
    asm = """{
            .reg .pred p<4>;
            .reg .b32 a<2>, b<2>;
            .reg .b16 c<4>;
            .reg .b16 max_val_f16;
            .reg .b32 max_val_f16x2;
            mov.b16 max_val_f16,   0x3F00;
            mov.b32 max_val_f16x2, 0x3F003F00;
            and.b32 a0, $1, 0x7fff7fff;
            and.b32 a1, $2, 0x7fff7fff;"""
    if has_minx2:
        asm += """min.f16x2 a0, a0, max_val_f16x2;
                  min.f16x2 a1, a1, max_val_f16x2;"""
    else:
        asm += """setp.lt.f16x2  p0|p1, a0, max_val_f16x2;
                  setp.lt.f16x2  p2|p3, a1, max_val_f16x2;
                  mov.b32 {c0, c1}, a0;
                  mov.b32 {c2, c3}, a1;
                  selp.b16  c0, c0, max_val_f16, p0;
                  selp.b16  c1, c1, max_val_f16, p1;
                  selp.b16  c2, c2, max_val_f16, p2;
                  selp.b16  c3, c3, max_val_f16, p3;
                  mov.b32 a0, {c0, c1};
                  mov.b32 a1, {c2, c3};"""
    asm += """mad.lo.u32 a0, a0, 2, 0x00800080;
              mad.lo.u32 a1, a1, 2, 0x00800080;
              lop3.b32 b0, $1, 0x80008000, a0, 0xea;
              lop3.b32 b1, $2, 0x80008000, a1, 0xea;
              prmt.b32 $0, b0, b1, 0x7531;
              }"""
    return core.inline_asm_elementwise(asm, "=r,r,r", [arg], dtype=core.float8e4b15, is_pure=True, pack=4,
                                       _semantic=_semantic)


@core.builtin
def convert_custom_float8(arg, dst_ty, fp_downcast_rounding, has_minx2, _semantic=None):
    if arg.type.scalar.is_fp8e4b15():
        upcast_val = convert_fp8e4b15_to_float16(arg, _semantic=_semantic)
        if dst_ty.scalar.is_fp32():
            upcast_val = upcast_val.to(core.float32, _semantic=_semantic)
        return upcast_val

    assert arg.type.scalar.is_fp16() or arg.type.scalar.is_fp32()
    downcast_val = arg
    if arg.type.scalar.is_fp32():
        downcast_val = downcast_val.to(core.float16, fp_downcast_rounding="rtz", _semantic=_semantic)
    downcast_val = convert_float16_to_fp8e4b15(downcast_val, has_minx2=has_minx2, _semantic=_semantic)
    return downcast_val


@core.builtin
def convert_custom_float8_sm80(arg, dst_ty, fp_downcast_rounding=None, _semantic=None):
    return convert_custom_float8(arg, dst_ty, fp_downcast_rounding, has_minx2=True, _semantic=_semantic)


@core.builtin
def convert_custom_float8_sm70(arg, dst_ty, fp_downcast_rounding=None, _semantic=None):
    return convert_custom_float8(arg, dst_ty, fp_downcast_rounding, has_minx2=False, _semantic=_semantic)
<|endoftext|>from triton.language import core


@core.extern
def clz(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("int32"), ): ("__nv_clz", core.dtype("int32")),
            (core.dtype("int64"), ): ("__nv_clzll", core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def popc(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("int32"), ): ("__nv_popc", core.dtype("int32")),
            (core.dtype("int64"), ): ("__nv_popcll", core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def byte_perm(arg0, arg1, arg2, _semantic=None):
    return core.extern_elementwise("", "", [arg0, arg1, arg2], {
        (core.dtype("int32"), core.dtype("int32"), core.dtype("int32")): ("__nv_byte_perm", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def mulhi(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("int32"), core.dtype("int32")): ("__nv_mulhi", core.dtype("int32")),
            (core.dtype("uint32"), core.dtype("uint32")): ("__nv_umulhi", core.dtype("uint32")),
            (core.dtype("int64"), core.dtype("int64")): ("__nv_mul64hi", core.dtype("int64")),
            (core.dtype("uint64"), core.dtype("uint64")): ("__nv_umul64hi", core.dtype("uint64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def mul24(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("int32"), core.dtype("int32")): ("__nv_mul24", core.dtype("int32")),
            (core.dtype("uint32"), core.dtype("uint32")): ("__nv_umul24", core.dtype("uint32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def brev(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("int32"), ): ("__nv_brev", core.dtype("int32")),
            (core.dtype("int64"), ): ("__nv_brevll", core.dtype("int64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def sad(arg0, arg1, arg2, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.dtype("int32"), core.dtype("int32"), core.dtype("uint32")): ("__nv_sad", core.dtype("int32")),
            (core.dtype("uint32"), core.dtype("uint32"), core.dtype("uint32")): ("__nv_usad", core.dtype("uint32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def abs(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("int32"), ): ("__nv_abs", core.dtype("int32")),
            (core.dtype("int64"), ): ("__nv_llabs", core.dtype("int64")),
            (core.dtype("fp32"), ): ("__nv_fabsf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_fabs", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def floor(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_floorf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_floor", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def rcp64h(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__nv_rcp64h", core.dtype("fp64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def rsqrt(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_rsqrtf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_rsqrt", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def ceil(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp64"), ): ("__nv_ceil", core.dtype("fp64")),
            (core.dtype("fp32"), ): ("__nv_ceilf", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def trunc(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp64"), ): ("__nv_trunc", core.dtype("fp64")),
            (core.dtype("fp32"), ): ("__nv_truncf", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def exp2(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_exp2f", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_exp2", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def saturatef(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_saturatef", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fma_rn(arg0, arg1, arg2, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")): ("__nv_fmaf_rn", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64")): ("__nv_fma_rn", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fma_rz(arg0, arg1, arg2, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")): ("__nv_fmaf_rz", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64")): ("__nv_fma_rz", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fma_rd(arg0, arg1, arg2, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")): ("__nv_fmaf_rd", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64")): ("__nv_fma_rd", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fma_ru(arg0, arg1, arg2, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")): ("__nv_fmaf_ru", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64")): ("__nv_fma_ru", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_dividef(arg0, arg1, _semantic=None):
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__nv_fast_fdividef", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def div_rn(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__nv_fdiv_rn", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__nv_ddiv_rn", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def div_rz(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__nv_fdiv_rz", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__nv_ddiv_rz", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def div_rd(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__nv_fdiv_rd", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__nv_ddiv_rd", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def div_ru(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__nv_fdiv_ru", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__nv_ddiv_ru", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def rcp_rn(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_frcp_rn", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_drcp_rn", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def rcp_rz(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_frcp_rz", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_drcp_rz", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def rcp_rd(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_frcp_rd", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_drcp_rd", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def rcp_ru(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_frcp_ru", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_drcp_ru", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def sqrt_rn(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_fsqrt_rn", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_dsqrt_rn", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def sqrt_rz(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_fsqrt_rz", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_dsqrt_rz", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def sqrt_rd(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_fsqrt_rd", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_dsqrt_rd", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def sqrt_ru(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_fsqrt_ru", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_dsqrt_ru", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def sqrt(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_sqrtf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_sqrt", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def add_rn(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp64"), core.dtype("fp64")): ("__nv_dadd_rn", core.dtype("fp64")),
            (core.dtype("fp32"), core.dtype("fp32")): ("__nv_fadd_rn", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def add_rz(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp64"), core.dtype("fp64")): ("__nv_dadd_rz", core.dtype("fp64")),
            (core.dtype("fp32"), core.dtype("fp32")): ("__nv_fadd_rz", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def add_rd(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp64"), core.dtype("fp64")): ("__nv_dadd_rd", core.dtype("fp64")),
            (core.dtype("fp32"), core.dtype("fp32")): ("__nv_fadd_rd", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def add_ru(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp64"), core.dtype("fp64")): ("__nv_dadd_ru", core.dtype("fp64")),
            (core.dtype("fp32"), core.dtype("fp32")): ("__nv_fadd_ru", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def mul_rn(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp64"), core.dtype("fp64")): ("__nv_dmul_rn", core.dtype("fp64")),
            (core.dtype("fp32"), core.dtype("fp32")): ("__nv_fmul_rn", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def mul_rz(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp64"), core.dtype("fp64")): ("__nv_dmul_rz", core.dtype("fp64")),
            (core.dtype("fp32"), core.dtype("fp32")): ("__nv_fmul_rz", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def mul_rd(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp64"), core.dtype("fp64")): ("__nv_dmul_rd", core.dtype("fp64")),
            (core.dtype("fp32"), core.dtype("fp32")): ("__nv_fmul_rd", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def mul_ru(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
            arg1,
        ], {
            (
                core.dtype("fp64"),
                core.dtype("fp64"),
            ): ("__nv_dmul_ru", core.dtype("fp64")),
            (
                core.dtype("fp32"),
                core.dtype("fp32"),
            ): ("__nv_fmul_ru", core.dtype("fp32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def double2float_rn(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__nv_double2float_rn", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2float_rz(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__nv_double2float_rz", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2float_rd(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__nv_double2float_rd", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2float_ru(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__nv_double2float_ru", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2int_rn(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__nv_double2int_rn", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2int_rz(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__nv_double2int_rz", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2int_rd(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__nv_double2int_rd", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2int_ru(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__nv_double2int_ru", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2uint_rn(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__nv_double2uint_rn", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2uint_rz(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__nv_double2uint_rz", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2uint_rd(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__nv_double2uint_rd", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2uint_ru(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__nv_double2uint_ru", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def int2double_rn(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int32"), ): ("__nv_int2double_rn", core.dtype("fp64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def uint2double_rn(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint32"), ): ("__nv_uint2double_rn", core.dtype("fp64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2int_rn(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_float2int_rn", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2int_rz(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_float2int_rz", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2int_rd(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_float2int_rd", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2int_ru(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_float2int_ru", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2uint_rn(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_float2uint_rn", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2uint_rz(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_float2uint_rz", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2uint_rd(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_float2uint_rd", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2uint_ru(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_float2uint_ru", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def int2float_rn(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int32"), ): ("__nv_int2float_rn", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def int2float_rz(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int32"), ): ("__nv_int2float_rz", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def int2float_rd(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int32"), ): ("__nv_int2float_rd", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def int2float_ru(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int32"), ): ("__nv_int2float_ru", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def uint2float_rn(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint32"), ): ("__nv_uint2float_rn", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def uint2float_rz(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint32"), ): ("__nv_uint2float_rz", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def uint2float_rd(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint32"), ): ("__nv_uint2float_rd", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def uint2float_ru(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint32"), ): ("__nv_uint2float_ru", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def hiloint2double(arg0, arg1, _semantic=None):
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("int32"), core.dtype("int32")): ("__nv_hiloint2double", core.dtype("fp64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2loint(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__nv_double2loint", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2hiint(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__nv_double2hiint", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2ll_rn(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_float2ll_rn", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2ll_rz(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_float2ll_rz", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2ll_rd(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_float2ll_rd", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2ll_ru(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_float2ll_ru", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2ull_rn(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_float2ull_rn", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2ull_rz(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_float2ull_rz", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2ull_rd(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_float2ull_rd", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float2ull_ru(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_float2ull_ru", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2ll_rn(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__nv_double2ll_rn", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2ll_rz(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__nv_double2ll_rz", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2ll_rd(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__nv_double2ll_rd", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2ll_ru(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__nv_double2ll_ru", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2ull_rn(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__nv_double2ull_rn", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2ull_rz(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__nv_double2ull_rz", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2ull_rd(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__nv_double2ull_rd", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double2ull_ru(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__nv_double2ull_ru", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ll2float_rn(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int64"), ): ("__nv_ll2float_rn", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ll2float_rz(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int64"), ): ("__nv_ll2float_rz", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ll2float_rd(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int64"), ): ("__nv_ll2float_rd", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ll2float_ru(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int64"), ): ("__nv_ll2float_ru", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ull2float_rn(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint64"), ): ("__nv_ull2float_rn", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ull2float_rz(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint64"), ): ("__nv_ull2float_rz", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ull2float_rd(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint64"), ): ("__nv_ull2float_rd", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ull2float_ru(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint64"), ): ("__nv_ull2float_ru", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ll2double_rn(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int64"), ): ("__nv_ll2double_rn", core.dtype("fp64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ll2double_rz(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int64"), ): ("__nv_ll2double_rz", core.dtype("fp64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ll2double_rd(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int64"), ): ("__nv_ll2double_rd", core.dtype("fp64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ll2double_ru(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int64"), ): ("__nv_ll2double_ru", core.dtype("fp64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ull2double_rn(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint64"), ): ("__nv_ull2double_rn", core.dtype("fp64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ull2double_rz(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint64"), ): ("__nv_ull2double_rz", core.dtype("fp64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ull2double_rd(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint64"), ): ("__nv_ull2double_rd", core.dtype("fp64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ull2double_ru(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint64"), ): ("__nv_ull2double_ru", core.dtype("fp64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def int_as_float(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int32"), ): ("__nv_int_as_float", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float_as_int(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_float_as_int", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def uint_as_float(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("uint32"), ): ("__nv_uint_as_float", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def float_as_uint(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_float_as_uint", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def longlong_as_double(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("int64"), ): ("__nv_longlong_as_double", core.dtype("fp64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def double_as_longlong(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__nv_double_as_longlong", core.dtype("int64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_sinf(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_fast_sinf", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_cosf(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_fast_cosf", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_log2f(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_fast_log2f", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_logf(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_fast_logf", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_expf(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_fast_expf", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_tanf(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_fast_tanf", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_exp10f(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_fast_exp10f", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_log10f(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_fast_log10f", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_powf(arg0, arg1, _semantic=None):
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__nv_fast_powf", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def hadd(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("int32"), core.dtype("int32")): ("__nv_hadd", core.dtype("int32")),
            (core.dtype("uint32"), core.dtype("uint32")): ("__nv_uhadd", core.dtype("uint32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def rhadd(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("int32"), core.dtype("int32")): ("__nv_rhadd", core.dtype("int32")),
            (core.dtype("uint32"), core.dtype("uint32")): ("__nv_urhadd", core.dtype("uint32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def sub_rn(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__nv_fsub_rn", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__nv_dsub_rn", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def sub_rz(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__nv_fsub_rz", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__nv_dsub_rz", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def sub_rd(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__nv_fsub_rd", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__nv_dsub_rd", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def sub_ru(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__nv_fsub_ru", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__nv_dsub_ru", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def rsqrt_rn(arg0, _semantic=None):
    return core.extern_elementwise("", "", [
        arg0,
    ], {
        (core.dtype("fp32"), ): ("__nv_frsqrt_rn", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def ffs(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("int32"), ): ("__nv_ffs", core.dtype("int32")),
            (core.dtype("int64"), ): ("__nv_ffsll", core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def rint(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__nv_rintf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_rint", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def llrint(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__nv_llrintf", core.dtype("int64")),
            (core.dtype("fp64"), ): ("__nv_llrint", core.dtype("int64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def nearbyint(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__nv_nearbyintf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_nearbyint", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def isnan(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__nv_isnanf", core.dtype("int32")),
            (core.dtype("fp64"), ): ("__nv_isnand", core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic).to(core.int1, _semantic=_semantic)


@core.extern
def signbit(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__nv_signbitf", core.dtype("int32")),
            (core.dtype("fp64"), ): ("__nv_signbitd", core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def copysign(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__nv_copysignf", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__nv_copysign", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def finitef(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_finitef", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic).to(core.int1, _semantic=_semantic)


@core.extern
def isinf(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_isinff", core.dtype("int32")),
            (core.dtype("fp64"), ): ("__nv_isinfd", core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic).to(core.int1, _semantic=_semantic)


@core.extern
def nextafter(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__nv_nextafterf", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__nv_nextafter", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def sin(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_sinf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_sin", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def cos(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_cosf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_cos", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def sinpi(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_sinpif", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_sinpi", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def cospi(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_cospif", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_cospi", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def tan(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_tanf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_tan", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def log2(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_log2f", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_log2", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def exp(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_expf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_exp", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def exp10(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_exp10f", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_exp10", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def cosh(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_coshf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_cosh", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def sinh(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_sinhf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_sinh", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def tanh(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_tanhf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_tanh", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def atan2(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__nv_atan2f", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__nv_atan2", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def atan(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_atanf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_atan", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def asin(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_asinf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_asin", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def acos(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_acosf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_acos", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def log(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_logf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_log", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def log10(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_log10f", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_log10", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def log1p(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_log1pf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_log1p", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def acosh(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_acoshf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_acosh", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def asinh(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_asinhf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_asinh", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def atanh(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_atanhf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_atanh", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def expm1(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_expm1f", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_expm1", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def hypot(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__nv_hypotf", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__nv_hypot", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def rhypot(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__nv_rhypotf", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__nv_rhypot", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def norm3d(arg0, arg1, arg2, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")): ("__nv_norm3df", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64")): ("__nv_norm3d", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def rnorm3d(arg0, arg1, arg2, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")): ("__nv_rnorm3df", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64")): ("__nv_rnorm3d", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def norm4d(arg0, arg1, arg2, arg3, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2, arg3], {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")):
            ("__nv_norm4df", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64")):
            ("__nv_norm4d", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def rnorm4d(arg0, arg1, arg2, arg3, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2, arg3], {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")):
            ("__nv_rnorm4df", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64")):
            ("__nv_rnorm4d", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def cbrt(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_cbrtf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_cbrt", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def rcbrt(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_rcbrtf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_rcbrt", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def j0(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_j0f", core.dtype("fp32")),
        (core.dtype("fp64"), ): ("__nv_j0", core.dtype("fp64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def j1(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_j1f", core.dtype("fp32")),
        (core.dtype("fp64"), ): ("__nv_j1", core.dtype("fp64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def y0(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_y0f", core.dtype("fp32")),
        (core.dtype("fp64"), ): ("__nv_y0", core.dtype("fp64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def y1(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__nv_y1f", core.dtype("fp32")),
        (core.dtype("fp64"), ): ("__nv_y1", core.dtype("fp64")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def yn(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("int32"), core.dtype("fp32")): ("__nv_ynf", core.dtype("fp32")),
            (core.dtype("int32"), core.dtype("fp64")): ("__nv_yn", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def jn(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("int32"), core.dtype("fp32")): ("__nv_jnf", core.dtype("fp32")),
            (core.dtype("int32"), core.dtype("fp64")): ("__nv_jn", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def cyl_bessel_i0(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_cyl_bessel_i0f", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_cyl_bessel_i0", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def cyl_bessel_i1(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_cyl_bessel_i1f", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_cyl_bessel_i1", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def erf(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_erff", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_erf", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def erfinv(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_erfinvf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_erfinv", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def erfc(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_erfcf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_erfc", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def erfcx(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_erfcxf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_erfcx", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def erfcinv(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_erfcinvf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_erfcinv", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def normcdfinv(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_normcdfinvf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_normcdfinv", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def normcdf(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_normcdff", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_normcdf", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def lgamma(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_lgammaf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_lgamma", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def ldexp(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("int32")): ("__nv_ldexpf", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("int32")): ("__nv_ldexp", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def scalbn(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("int32")): ("__nv_scalbnf", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("int32")): ("__nv_scalbn", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fmod(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__nv_fmodf", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__nv_fmod", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def remainder(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__nv_remainderf", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__nv_remainder", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fma(arg0, arg1, arg2, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")): ("__nv_fmaf", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64")): ("__nv_fma", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def pow(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("int32")): ("__nv_powif", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("int32")): ("__nv_powi", core.dtype("fp64")),
            (core.dtype("fp32"), core.dtype("fp32")): ("__nv_powf", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__nv_pow", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def tgamma(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_tgammaf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_tgamma", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def round(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_roundf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_round", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def llround(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_llroundf", core.dtype("int64")),
            (core.dtype("fp64"), ): ("__nv_llround", core.dtype("int64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fdim(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__nv_fdimf", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__nv_fdim", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def ilogb(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_ilogbf", core.dtype("int32")),
            (core.dtype("fp64"), ): ("__nv_ilogb", core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def logb(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__nv_logbf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__nv_logb", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def isfinited(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__nv_isfinited", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic).to(core.int1, _semantic=_semantic)
<|endoftext|>from . import libdevice

from .utils import (globaltimer, num_threads, num_warps, smid, convert_custom_float8_sm70, convert_custom_float8_sm80)
from .gdc import (gdc_launch_dependents, gdc_wait)

__all__ = [
    "libdevice",
    "globaltimer",
    "num_threads",
    "num_warps",
    "smid",
    "convert_custom_float8_sm70",
    "convert_custom_float8_sm80",
    "gdc_launch_dependents",
    "gdc_wait",
]
<|endoftext|>"""
Grid Dependency Control (GDC) is a mechanism used when enabling programmatic dependent launch to launch and
synchronize grids. These APIs expose GDC to the programmer.

Programmatic dependent launch is supported on SM90 (Hopper) and beyond.
For PTX reference on grid dependency control see https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-griddepcontrol.
"""

from triton.language import core


@core.extern
def gdc_wait(_semantic=None):
    """
    GDC wait is a blocking instruction that waits for all instructions in a prior kernel to complete before continuing.
    This ensures all memory operations happening before the wait is visible to instructions after it,
    e.g. if the prior kernel writes to address "x" the new values will be visible in this kernel after the wait.

    This instruction is also safe to execute when programmatic dependent launch is disabled.

    See https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-griddepcontrol for more details.
    """
    core.inline_asm_elementwise("griddepcontrol.wait; // dummy $0", "=r", [], dtype=core.int32, is_pure=False, pack=1,
                                _semantic=_semantic)


@core.extern
def gdc_launch_dependents(_semantic=None):
    """
    This operation when launched with programmatic dependent launch signals that
    the next program may launch once all programs in the current kernel
    call this function or complete.

    Repeated calls to this function have no effect past the first call, and the first call should be
    treated by the programmer as a hint to the runtime system to launch the next kernel.

    This instruction is also safe to execute when programmatic dependent launch is disabled.

    See https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-griddepcontrol for more details.
    """
    core.inline_asm_elementwise("griddepcontrol.launch_dependents; // dummy $0", "=r", [], dtype=core.int32,
                                is_pure=False, pack=1, _semantic=_semantic)
<|endoftext|>from triton.language import core


@core.extern
def memrealtime(_semantic=None):
    """
    Returns a 64-bit real time-counter value
    """
    target_arch = _semantic.builder.options.arch
    asm_str = """s_memrealtime $0
                 s_waitcnt vmcnt(0)"""
    if 'gfx11' in target_arch:
        asm_str = """s_sendmsg_rtn_b64 $0, sendmsg(MSG_RTN_GET_REALTIME)
                     s_waitcnt lgkmcnt(0)"""
    elif 'gfx12' in target_arch:
        asm_str = """s_sendmsg_rtn_b64 $0, sendmsg(MSG_RTN_GET_REALTIME)
                     s_wait_kmcnt 0"""
    return core.inline_asm_elementwise(
        asm_str,
        "=r",
        [],
        dtype=core.int64,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def smid(_semantic=None):
    """
    Returns the compute unit / workgroup processor ID for the current wave.

    GCN/CDNA (gfx9xx): reads CU_ID, SH_ID, and SE_ID fields from HW_REG_HW_ID
    (register 4) as a packed value.  On multi-XCC parts (gfx942/gfx950) the
    XCC_ID is NOT included; values are unique only within a single XCC.

    RDNA (gfx10xx/gfx11xx/gfx12xx): reads WGP_ID from HW_REG_HW_ID1
    (register 23).  Values are unique only within a shader array.
    """
    target_arch = _semantic.builder.options.arch
    if 'gfx9' in target_arch:
        # HW_REG_HW_ID (reg 4), bits [15:8]:
        #   [11:8]  CU_ID  (4 bits)
        #   [12]    SH_ID  (1 bit)
        #   [15:13] SE_ID  (2-3 bits depending on chip)
        asm_str = "s_getreg_b32 $0, hwreg(4, 8, 8)"
    elif 'gfx10' in target_arch or 'gfx11' in target_arch or 'gfx12' in target_arch:
        # HW_REG_HW_ID1 (reg 23), bits [13:10]: WGP_ID (4 bits)
        asm_str = "s_getreg_b32 $0, hwreg(23, 10, 4)"
    else:
        raise ValueError(f"smid is not supported on {target_arch}")
    return core.inline_asm_elementwise(
        asm_str,
        "=r",
        [],
        dtype=core.int32,
        is_pure=True,
        pack=1,
        _semantic=_semantic,
    )


@core.builtin
def num_threads(_semantic=None):
    opts = _semantic.builder.options
    return core.constexpr(opts.num_warps * opts.warp_size)


@core.builtin
def num_warps(_semantic=None):
    return core.constexpr(_semantic.builder.options.num_warps)
<|endoftext|>from triton.language import core


@core.extern
def clz(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("int32"), ): ("__ockl_clz_u32", core.dtype("int32")),
            (core.dtype("int64"), ): ("__ockl_clz_u64", core.dtype("int64")),
        }, is_pure=True, _semantic=_semantic).to(core.int32, _semantic=_semantic)


@core.extern
def popc(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("int32"), ): ("__ockl_popcount_u32", core.dtype("int32")),
            (core.dtype("int64"), ): ("__ockl_popcount_u64", core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def abs(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("int32"), ): ("__triton_hip_iabs", core.dtype("int32")),
            (core.dtype("int64"), ): ("__triton_hip_iabs", core.dtype("int64")),
            (core.dtype("fp32"), ): ("__triton_hip_fabs", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__triton_hip_fabs", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def floor(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_floor_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_floor_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def rsqrt(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_rsqrt_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_rsqrt_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def ceil(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_ceil_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_ceil_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def trunc(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_trunc_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_trunc_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def exp2(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_exp2_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_exp2_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def exp(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_exp_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_exp_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_expf(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__triton_hip_fast_expf", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_tanhf(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__triton_hip_fast_tanhf", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def fast_dividef(arg0, arg1, _semantic=None):
    return core.extern_elementwise("", "", [arg0, arg1], {
        (core.dtype("fp32"), core.dtype("fp32")): ("__triton_hip_fast_fdividef", core.dtype("fp32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def sqrt(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_sqrt_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_sqrt_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def rint(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__triton_hip_rint", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__triton_hip_rint", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def llrint(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__triton_hip_llrint", core.dtype("int64")),
            (core.dtype("fp64"), ): ("__triton_hip_llrint", core.dtype("int64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def nearbyint(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__ocml_nearbyint_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_nearbyint_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def isnan(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__ocml_isnan_f32", core.dtype("int32")),
            (core.dtype("fp64"), ): ("__ocml_isnan_f64", core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic).to(core.int1, _semantic=_semantic)


@core.extern
def signbit(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("__ocml_signbit_f32", core.dtype("int32")),
            (core.dtype("fp64"), ): ("__ocml_signbit_f64", core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def copysign(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__ocml_copysign_f32", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__ocml_copysign_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def isinf(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_isinf_f32", core.dtype("int32")),
            (core.dtype("fp64"), ): ("__ocml_isinf_f64", core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic).to(core.int1, _semantic=_semantic)


@core.extern
def nextafter(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__ocml_nextafter_f32", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__ocml_nextafter_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def sin(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_sin_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_sin_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def cos(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_cos_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_cos_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def tan(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_tan_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_tan_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def log2(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_log2_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_log2_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def cosh(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_cosh_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_cosh_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def sinh(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_sinh_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_sinh_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def tanh(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_tanh_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_tanh_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def atan2(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__ocml_atan2_f32", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__ocml_atan2_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def atan(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_atan_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_atan_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def asin(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_asin_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_asin_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def acos(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_acos_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_acos_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def log(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_log_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_log_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def log10(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_log10_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_log10_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def log1p(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_log1p_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_log1p_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def acosh(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_acosh_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_acosh_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def asinh(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_asinh_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_asinh_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def atanh(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_atanh_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_atanh_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def expm1(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_expm1_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_expm1_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def hypot(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__ocml_hypot_f32", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__ocml_hypot_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def j0(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_j0_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_j0_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def j1(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_j1_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_j1_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def y0(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_y0_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_y0_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def y1(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_y1_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_y1_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def cyl_bessel_i0(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_i0_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_i0_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def cyl_bessel_i1(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_i1_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_i1_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def erf(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_erf_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_erf_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def erfinv(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_erfinv_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_erfinv_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def erfc(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_erfc_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_erfc_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def erfcx(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_erfcx_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_erfcx_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def lgamma(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_lgamma_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_lgamma_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def ldexp(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("int32")): ("__ocml_ldexp_f32", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("int32")): ("__ocml_ldexp_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fmod(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__ocml_fmod_f32", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__ocml_fmod_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def fma(arg0, arg1, arg2, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1, arg2], {
            (core.dtype("fp32"), core.dtype("fp32"), core.dtype("fp32")): ("__ocml_fma_f32", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64"), core.dtype("fp64")): ("__ocml_fma_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def pow(arg0, arg1, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("int32")): ("__ocml_pown_f32", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("int32")): ("__ocml_pown_f64", core.dtype("fp64")),
            (core.dtype("fp32"), core.dtype("fp32")): ("__ocml_pow_f32", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__ocml_pow_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def ilogb(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_ilogb_f32", core.dtype("int32")),
            (core.dtype("fp64"), ): ("__ocml_ilogb_f64", core.dtype("int32")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def round(arg0, _semantic=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__ocml_round_f32", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__ocml_round_f64", core.dtype("fp64")),
        }, is_pure=True, _semantic=_semantic)


@core.extern
def finitef(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("__ocml_isfinite_f32", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic).to(core.int1, _semantic=_semantic)


@core.extern
def isfinited(arg0, _semantic=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("__ocml_isfinite_f64", core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic).to(core.int1, _semantic=_semantic)
<|endoftext|>import argparse
import sys
import os
import runpy
import traceback
from .profile import start, finalize, _select_backend
from .flags import flags
from triton._C.libproton import proton as libproton


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="The proton command utility for profiling scripts and pytest tests.", usage="""
    proton [options] script.py [script_args] [script_options]
    proton [options] pytest [pytest_args] [script_options]
    python -m triton.profiler.proton [options] script.py [script_args] [script_options]
""", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-n", "--name", type=str, help="Name of the profiling session")
    parser.add_argument("-b", "--backend", type=str, help="Profiling backend", default=None,
                        choices=libproton.get_available_profilers())
    parser.add_argument("-c", "--context", type=str, help="Profiling context", default="shadow",
                        choices=["shadow", "python"])
    parser.add_argument("-m", "--mode", type=str, help="Profiling mode", default=None)
    parser.add_argument("-d", "--data", type=str, help="Profiling data", default="tree", choices=["tree", "trace"])
    parser.add_argument("-k", "--hook", type=str, help="Profiling hook", default=None, choices=[None, "triton"])
    parser.add_argument('target_args', nargs=argparse.REMAINDER, help='Subcommand and its arguments')
    args = parser.parse_args()
    return args, args.target_args


def is_pytest(script):
    return os.path.basename(script) == 'pytest'


def execute_as_main(script, args):
    script_path = os.path.abspath(script)

    original_argv = sys.argv
    sys.argv = [script] + args
    # Append the script's directory in case the script uses relative imports
    sys.path.append(os.path.dirname(script_path))

    # Execute in the isolated environment
    try:
        runpy.run_path(script, run_name="__main__")
    except Exception as e:
        print("An error occurred while executing the script:")
        traceback.print_exception(e)
        return 1
    except SystemExit as e:
        return e.code
    except KeyboardInterrupt:
        return 1
    finally:
        sys.argv = original_argv
    return 0


def do_setup_and_execute(target_args):
    # Set the command line mode to avoid any `start` calls in the script.
    flags.command_line = True

    script = target_args[0]
    script_args = target_args[1:] if len(target_args) > 1 else []
    if is_pytest(script):
        import pytest
        return pytest.main(script_args)
    else:
        return execute_as_main(script, script_args)


def run_profiling(args, target_args):
    backend = args.backend if args.backend else _select_backend()

    start(args.name, context=args.context, data=args.data, backend=backend, mode=args.mode, hook=args.hook)

    exitcode = do_setup_and_execute(target_args)

    finalize()
    sys.exit(exitcode)


def main():
    args, target_args = parse_arguments()
    run_profiling(args, target_args)


if __name__ == "__main__":
    main()
<|endoftext|>from functools import wraps

from triton._C.libproton import proton as libproton
from .flags import flags

COMPUTE_METADATA_SCOPE_NAME = libproton.metadata_scope_name
COMPUTE_METADATA_SCOPE_PREFIX = libproton.metadata_scope_prefix


class state:
    """
    A context manager and decorator for entering and exiting a state.

    Usage:
        context manager:
        ```python
        with proton.state("test0"):
            foo[1,](x, y)
        ```

        decorator:
        ```python
        @proton.state("test0")
        def foo(x, y):
            ...
        ```

    Args:
        name (str): The name of the state.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self):
        if not flags.profiling_on:
            return self
        libproton.enter_state(self.name)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if not flags.profiling_on:
            return
        libproton.exit_state()

    def __call__(self, func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            if flags.profiling_on:
                libproton.enter_state(self.name)
            ret = func(*args, **kwargs)
            if flags.profiling_on:
                libproton.exit_state()
            return ret

        return wrapper


class metadata_state(state):

    def __init__(self, kernel_name=None) -> None:
        super().__init__(get_metadata_state_name(kernel_name))


def enter_state(name: str) -> None:
    libproton.enter_state(name)


def exit_state() -> None:
    libproton.exit_state()


def get_metadata_state_name(kernel_name=None) -> str:
    if not kernel_name:
        return COMPUTE_METADATA_SCOPE_NAME
    return f"{COMPUTE_METADATA_SCOPE_PREFIX}{kernel_name}"
<|endoftext|>from typing import Optional
from triton._C.libproton import proton as libproton
from .flags import flags


def depth(session: Optional[int] = 0) -> Optional[int]:
    """
    Get the depth of the context.

    Args:
        session (int): The session ID of the profiling session. Defaults to 0.

    Returns:
        depth (int or None): The depth of the context. If profiling is off, returns None.
    """
    if not flags.profiling_on:
        return None
    return libproton.get_context_depth(session)
<|endoftext|>flops_by_device = {
    "CUDA": {
        "80":
        lambda width, **kwargs: 624e12 / (width / 8),
        "89":
        lambda width, **kwargs: (330.3 * 1e12) / (width / 8),  # TODO(Keren): Implement fp16 acc-> 660.6 fp8
        "90":
        lambda width, num_sms, clock_rate, **kwargs: ((num_sms / 114 * clock_rate / (1755 * 1e3) * 1513) * 1e12) /
        (width / 8),
        "100":
        lambda width, num_sms, clock_rate, **kwargs: (num_sms * 16384 * (clock_rate / 1e3) * 1e6) / (width / 8),
    }
}

amd_bps_by_arch = {
    'gfx90a': 3.2 * 1e12,
    'gfx942': 5.3 * 1e12,
    'gfx950': 8.0 * 1e12,
}

# FP8 Matrix Performance(FLOPS/clock/CU)
# For gfx90a we use the performance of INT8 since it doesn't support FP8 matrix operations.
amd_fp8_flops_by_arch = {'gfx90a': 1024, 'gfx942': 4096, 'gfx950': 8192}


def max_flops(device_type, arch, width, num_sms, clock_rate):
    """
    Calculate the maximum FLOPS for a given device type and width.

    Args:
        device_type (str): The type of device (e.g., "CUDA", "HIP").
        arch (str): The architecture of the device (e.g., "80", "90").
        width (int): The width in bits.
        num_sms (int): The number of streaming multiprocessors.
        clock_rate (float): The clock rate in GHz.

    Returns:
        float: The maximum FLOPS for the given device type and width.
    """
    if device_type == "HIP":
        return amd_fp8_flops_by_arch[arch] * num_sms * clock_rate * 1e3 / (width / 8)

    if device_type not in flops_by_device:
        raise ValueError(f"Unsupported device type: {device_type}")

    if arch not in flops_by_device[device_type]:
        raise ValueError(f"Unsupported architecture: {arch}")

    flops_func = flops_by_device[device_type][arch]

    return flops_func(width, num_sms=num_sms, clock_rate=clock_rate)


def max_bps(device_type, arch, bus_width, memory_clock_rate):
    """
    Calculate the maximum bytes per second for a given bus width and memory clock rate.

    Args:
        bus_width (int): The bus width in bits.
        memory_clock_rate (float): The memory clock rate in GHz.

    Returns:
        float: The maximum bytes per second.
    """
    if device_type == "CUDA":
        return 2 * bus_width * memory_clock_rate * 1e3 / 8
    else:
        assert device_type == "HIP"
        return amd_bps_by_arch[arch]
<|endoftext|>from dataclasses import dataclass, field
from triton._C.libtriton import proton as triton_proton
from typing import List
from enum import Enum

metric_types = {"cycle": triton_proton.METRIC_TYPE.CYCLE}

buffer_strategies = {
    "circular": triton_proton.BUFFER_STRATEGY.CIRCULAR,
    "flush": triton_proton.BUFFER_STRATEGY.FLUSH,
}

buffer_types = {
    "shared": triton_proton.BUFFER_TYPE.SHARED,
    "global": triton_proton.BUFFER_TYPE.GLOBAL,
}

sampling_strategies = {
    "none": triton_proton.SAMPLING_STRATEGY.NONE,
    "selective": triton_proton.SAMPLING_STRATEGY.SELECTIVE,
}

granularities = {
    "cta": triton_proton.GRANULARITY.CTA,
    "warp": triton_proton.GRANULARITY.WARP,
    "warp_2": triton_proton.GRANULARITY.WARP_2,
    "warp_4": triton_proton.GRANULARITY.WARP_4,
    "warp_8": triton_proton.GRANULARITY.WARP_8,
    "warp_group": triton_proton.GRANULARITY.WARP_GROUP,
    "warp_group_2": triton_proton.GRANULARITY.WARP_GROUP_2,
    "warp_group_4": triton_proton.GRANULARITY.WARP_GROUP_4,
    "warp_group_8": triton_proton.GRANULARITY.WARP_GROUP_8,
}


class Optimize(Enum):
    TIMESHIFT = "time_shift"
    SCHED_STORES = "sched_stores"
    SCHED_BARRIERS = "sched_barriers"
    CLOCK32 = "clock32"

    def __str__(self):
        return self.value


optimizations = {
    "time_shift": Optimize.TIMESHIFT,
    "sched_stores": Optimize.SCHED_STORES,
    "sched_barriers": Optimize.SCHED_BARRIERS,
    "clock32": Optimize.CLOCK32,
}


@dataclass(frozen=True)
class BaseMode:
    name: str


@dataclass(frozen=True)
class PCSampling(BaseMode):
    name: str = field(default="pcsampling", init=False)
    interval: int = 1000

    def __post_init__(self):
        if self.interval <= 0:
            raise ValueError("Interval must be a positive integer.")

    def __str__(self):
        return f"{self.name}:interval={self.interval}"


@dataclass(frozen=True)
class InstrumentationMode(BaseMode):
    """Common base class for instrumentation modes with shared configuration."""
    metric_type: triton_proton.METRIC_TYPE = triton_proton.METRIC_TYPE.CYCLE
    sampling_strategy: triton_proton.SAMPLING_STRATEGY = triton_proton.SAMPLING_STRATEGY.NONE
    sampling_options: str = ""
    granularity: triton_proton.GRANULARITY = triton_proton.GRANULARITY.WARP
    buffer_strategy: triton_proton.BUFFER_STRATEGY = triton_proton.BUFFER_STRATEGY.CIRCULAR
    buffer_type: triton_proton.BUFFER_TYPE = triton_proton.BUFFER_TYPE.SHARED
    buffer_size: int = 0
    optimizations: List[Optimize] = field(default_factory=list)

    def __post_init__(self):
        # automatically map string inputs to enums using the global lookup dicts
        mappings = [
            ("metric_type", metric_types),
            ("sampling_strategy", sampling_strategies),
            ("granularity", granularities),
            ("buffer_strategy", buffer_strategies),
            ("buffer_type", buffer_types),
        ]
        for field_name, lookup in mappings:
            value = getattr(self, field_name)
            if isinstance(value, str):
                if value not in lookup:
                    raise ValueError(f"Unknown {field_name}: {value}")
                object.__setattr__(self, field_name, lookup[value])

        values_str = getattr(self, "optimizations")
        if isinstance(values_str, str):
            values = [value.strip() for value in values_str.split(",")] if len(values_str) > 0 else []
            for value in values:
                if value not in optimizations:
                    raise ValueError(f"Unknown optimization: {value}")
            object.__setattr__(self, "optimizations", [optimizations[value] for value in values])

    def __str__(self):
        optimizations_str = ",".join([str(opt) for opt in self.optimizations])
        return (f"{self.name}:metric_type={self.metric_type}:sampling_strategy={self.sampling_strategy}"
                f":sampling_options={self.sampling_options}:granularity={self.granularity}"
                f":buffer_strategy={self.buffer_strategy}:buffer_type={self.buffer_type}"
                f":buffer_size={self.buffer_size}:optimizations={optimizations_str}")


@dataclass(frozen=True)
class Default(InstrumentationMode):
    name: str = field(default="default", init=False)


@dataclass(frozen=True)
class MMA(InstrumentationMode):
    name: str = field(default="mma", init=False)
<|endoftext|>import functools
import triton

from triton._C.libproton import proton as libproton  # type: ignore
from triton._C.libtriton import getenv  # type: ignore
from .flags import flags
from .hooks import HookManager, LaunchHook, InstrumentationHook
from .hooks.hook import Hook
from .mode import BaseMode
from typing import Optional, Union

DEFAULT_PROFILE_NAME = "proton"


def _select_backend() -> str:
    backend = triton.runtime.driver.active.get_current_target().backend
    return libproton.select_profiler_from_triton_backend(backend)


def _get_mode_str(backend: str, mode: Optional[Union[str, BaseMode]]) -> str:
    if backend == "instrumentation":
        prefix = triton.runtime.driver.active.get_current_target().backend
        return f"{prefix}:{mode}" if mode else prefix
    return str(mode) if mode else ""


def _check_env(backend: str) -> None:
    if backend in ("rocprofiler", "roctracer"):
        hip_device_envs = ["HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"]
        for env in hip_device_envs:
            if getenv(env, None) is not None:
                raise ValueError(
                    f"Proton does not work when the environment variable {env} is set on AMD GPUs. Please unset it and use `ROCR_VISIBLE_DEVICES` instead"
                )

    use_blackwell_cupti = backend == "cupti" and triton.runtime.driver.active.get_current_target().arch >= 100

    # Ensure default envs are set for Proton knobs if not already set by the user.
    for attr, desc in triton.knobs.proton.knob_descriptors.items():
        key = desc.key
        if getenv(key, None) is None:
            if key == "TRITON_CUPTI_LIB_PATH" and use_blackwell_cupti:
                # For Blackwell+ GPUs, use the cupti library built for Blackwell.
                triton.knobs.setenv(key, triton.knobs.proton.cupti_lib_blackwell_dir)
                continue
            val = getattr(triton.knobs.proton, attr)
            if val is not None:
                if env_val := triton.knobs.toenv(val):
                    triton.knobs.setenv(key, env_val[0])


def start(
    name: Optional[str] = None,
    *,
    context: Optional[str] = "shadow",
    data: Optional[str] = "tree",
    backend: Optional[str] = None,
    mode: Optional[Union[str, BaseMode]] = None,
    hook: Optional[Union[str, Hook]] = None,
) -> Optional[int]:
    """
    Start profiling with the given name and backend.

    Usage:

        ```python
        proton.start("my_profile")
        # do something
        proton.finalize()
        ```

    Args:
        name (str, optional): The name (with path) of the profiling session.
                              If not provided, the default name is "~/proton.<suffix>", where suffix is the default
                              format according to the data type. For example, if data is "tree", the default name is "~/proton.hatchet".
        context (str, optional): The context to use for profiling.
                                 Available options are ["shadow", "python"].
                                 Defaults to "shadow".
        data (str, optional): The data structure to use for profiling.
                              Available options are ["tree", "trace"].
                              Defaults to "tree".
        backend (str, optional): The backend to use for profiling.
                                 Available options are [None, "cupti", "rocprofiler", "roctracer", "instrumentation"].
                                 Defaults to None, which automatically selects the backend matching the current active runtime.
                                 On AMD GPUs, "rocprofiler" is preferred and will fall back to "roctracer" if
                                 rocprofiler-sdk is not available.
        mode (Union[str, BaseMode], optional): The "mode" to use for profiling, which is specific to the backend.
                                               Can be a string or an instance of BaseMode (or any subclass thereof).
                                               Defaults to None.
                                               For "cupti", available options are [None, "pcsampling", "periodic_flushing"].
                                               For "rocprofiler", available options are [None, "periodic_flushing"].
                                               For "roctracer", available options are [None, "periodic_flushing"].
                                               For "instrumentation", available options are [None].
                                               Each mode has a set of control knobs following with the mode name.
                                               For example, "periodic_flushing" mode has a knob:
                                               - format: The output format of the profiling results. Available options are ["hatchet", "hatchet_msgpack", "chrome_trace"]. Default is "hatchet".
                                               The can be set via `mode="periodic_flushing:format=chrome_trace"`.
        hook (Union[str, Hook], optional): The hook to use for profiling.
                                           You may pass either:
                                           - a string hook name, e.g. "triton" (kernel launch metadata), or
                                           - a custom Hook instance.
                                           Defaults to None.
    Returns:
        session (Optional[int]): The session ID of the profiling session, or None if profiling is disabled.
    """
    if flags.command_line or triton.knobs.proton.disable:
        # Ignore the start() call if the script is run from the command line or profiling is disabled.
        return None

    flags.profiling_on = True

    name = DEFAULT_PROFILE_NAME if name is None else name
    backend = _select_backend() if backend is None else backend
    # Convert mode to its string representation for libproton's runtime
    mode_str = _get_mode_str(backend, mode)

    _check_env(backend)

    session = libproton.start(name, context, data, backend, mode_str)

    if isinstance(hook, Hook):
        HookManager.register(hook, session)
    elif hook == "triton":
        HookManager.register(LaunchHook(), session)
    elif hook is not None:
        raise ValueError(f"Unsupported hook: {hook!r}")
    if backend == "instrumentation":
        HookManager.register(InstrumentationHook(mode), session)

    return session


def activate(session: Optional[int] = None) -> None:
    """
    Activate the specified session.
    The profiling session will be active and data will be recorded.

    Args:
        session (int): The session ID of the profiling session. Defaults to None (all sessions)

    Returns:
        None
    """
    if flags.command_line and session != 0:
        raise ValueError("Only one session can be activated when running from the command line.")

    HookManager.activate(session)

    if session is None:
        libproton.activate_all()
    else:
        libproton.activate(session)


