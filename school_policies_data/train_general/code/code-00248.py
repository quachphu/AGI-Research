Issues:
* Where should these ops live? Couldn't `import torch` if putting these ops in existing torch.distributed files
* Proper support for eager requires inplace ops. We should explore having it as an option for the API.
"""

"""
Functional collectives are asynchronous only and we perform implicit stream synchronization
on behalf of the user.

We use AsyncCollectiveTensor to wrap the result tensor of a collective and it lets us witness
first usage of the tensor and insert cross stream sync at the right place.

The above are the easy bits, the hard one is how we match the Work object returned by
c10d and the tensor AsyncCollectiveTensor wraps. We alloc the tensor inside the collective
op implementation (see ``clone()`` call in ``_all_reduce``) and then it's handled by the
dispatcher which might call other implementations that are allowed to change the returned
tensor - even return a tensor with a different shape (see ``torch.vmap``).

This means the caller of our ops receives a Tensor that is not guaranteed to be the same
allocated by our implementations and that makes pairing The AsyncTensor to the original
tensor a lot harder. This pairing is needed so we can lookup the Work object to use.

Originally, we tried WeakKeyDictionary to map from Tensor to Work, but because Tensor's
identity is not stable across dispatch, the op caller would end up with a different Tensor
instance that would not match any in the dictionary.

With Tensor identity out of the question, we decided use the tensor data pointer, which
should be stable across all the Tensor changes done during dispatch.

We have a dictionary of tensor::data_ptr -> Work that we insert right after we call into c10d.

We use this dictionary when AsyncCollectiveTensor is used to invoke Work::wait()

Finally, we setup a finalizer against the tensor wrapper to observe it getting collected so we
can clean up stale entries in the dictionary.

To eliminate the possibility of races we have a global version counter that is used by the finalizer.

As a wise man said once: Don't cross the streams (https://www.youtube.com/watch?v=wyKQe_i9yyo)

"""

"""
Functional collectives can accept any of these types to describe the ranks participating in collectives.

The different types will be desugared to a canonical format
"""
RANK_TYPES = (
    list[int]
    | list[list[int]]
    | dist.ProcessGroup
    | DeviceMesh
    | tuple["dist.tensor.DeviceMesh", int]
    | c10d.GroupName
)


from torch._utils import _chunk_or_narrow_cat


"""
User facing APIs for functional collectives
-------------------------------------------

These apis are called by user code and expected to work both in eager execution and compilation,
but there are significant differences to how the two modes are implemented underneath.

Eager execution is 'optimized' using a tensor subclass that schedules the synchronization (via wait_tensor() op)
just before the tensor is first used.  Compiled tracing currently relies on the compiler to perform this optimization,
and cannot yet correctly trace the AsyncTensor wrapper class.  In the future, these paths may be unified
if sufficient subclass support is added in dynamo.

Example: all_reduce is an entrypoint API, and other collectives follow a similar pattern.

Here's how it works under torch.compile/dynamo:
all_reduce(...)
  |--> _expand_group(...)               - desugars processgroup into canonical/traceable format
  |--> c10d_functional.all_reduce(...)  - dynamo captures this op call, doesn't trace deeper
  |--> _maybe_wrap_tensor(...)          - wait_tensor() op is immediately called, no AsyncTensor subclass needed

And under eager execution:
all_reduce(...)
  |--> _expand_group(...)               - same as above, but less critical for eager
  |--> c10d_functional.all_reduce(...)  - dispatches to real kernel OR records op in trace
  |--> _maybe_wrap_tensor(...)          - AsyncTensor wrapper applied to returned tensor,
                                          which issues wait_tensor() at the time of first use
"""


def wait_tensor(tensor):
    """
    Wait on a tensor returned by the collectives ops.

    Waiting follows device semantics, which means blocking on CPU and synchronizing streams on CUDA.
    """
    return torch.ops._c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]


def broadcast(self: torch.Tensor, src: int, group: RANK_TYPES, tag: str = ""):
    """
    Broadcasts the tensor to all processes in the given process group.

    Args:
        src (int): Source rank
        group (ProcessGroup or List[int]): The process group to work on.
        tag (str, optional): A unique identifier for the collective. Default: empty string
    """
    group = _resolve_group(group, tag)
    tensor = torch.ops._c10d_functional.broadcast(
        self, src, _group_or_group_name(group)
    )
    return _maybe_wrap_tensor(tensor)


def all_reduce(self: torch.Tensor, reduceOp: str, group: RANK_TYPES, tag: str = ""):
    """
    Reduces the tensor data across all machines in such a way that all get
    the final result.

    The input tensor is left unmodified.

    Group can be one of:
        List[int]: ranks participating in the collective.
        List[List[int]]: 2D mesh of ranks taking part of this collective in MPMD.
        ProcessGroup: Will perform a collective using the ranks and tag of the PG.
        DeviceMesh: Do a SPMD collective over all ranks of the mesh
        (DeviceMesh, int): Do a MPMD collective over one dimension of the DeviceMesh

    :: N.B. If you pass a PG or a 1D list to perform a MPMD collective, the compiler won't be able to recover
    that information and perform collective algebraic optimization. Use other forms of input for that.
    """
    group = _resolve_group(group, tag)
    tensor = torch.ops._c10d_functional.all_reduce(
        self, reduceOp.lower(), _group_or_group_name(group)
    )
    return _maybe_wrap_tensor(tensor)


def all_gather_single(
    self: torch.Tensor,
    gather_dim: int,
    group: RANK_TYPES,
    tag: str = "",
) -> torch.Tensor:
    """
    Gather tensor data across from all machines and concatenate over ``gather_dim``.

    Note that it currently only supports gather_dim = 0.

    The input tensor is left unmodified.
    Group can be one of:
        List[int]: ranks participating in the collective.
        List[List[int]]: 2D mesh of ranks taking part of this collective in MPMD.
        ProcessGroup: Will perform a collective using the ranks and tag of the PG.
        DeviceMesh: Do a SPMD collective over all ranks of the mesh
        (DeviceMesh, int): Do a MPMD collective over one dimension of the DeviceMesh

    :: N.B. If you pass a PG or a 1D list to perform a MPMD collective, the compiler won't be able to recover
    that information and perform collective algebraic optimization. Use other forms of input for that.
    """
    group = _resolve_group(group, tag)
    group_size = c10d._get_group_size_by_name(group)
    tensor = torch.ops._c10d_functional.all_gather_into_tensor(
        self, group_size, _group_or_group_name(group)
    )
    res = _maybe_wrap_tensor(tensor)
    if gather_dim != 0:
        # Check if _maybe_view_chunk_cat can use the view optimization.
        # If not, it will use torch.cat which needs the data anyway, so
        # wait early to avoid AsyncCollectiveTensor dispatch overhead.
        if isinstance(res, AsyncCollectiveTensor):
            shape = list(res.shape)
            numel_between = math.prod(shape[1:gather_dim]) if gather_dim > 1 else 1
            can_use_view = shape[0] == group_size and numel_between == 1
            if not can_use_view:
                res = res.wait()
        res = _maybe_view_chunk_cat(res, group_size, gather_dim)
    return res


def all_gather_single_autograd(
    self: torch.Tensor,
    gather_dim: int,
    group: RANK_TYPES,
    tag: str = "",
):
    """
    Gather tensor data across from all machines and concatenate over ``gather_dim``.

    Note that it currently only supports gather_dim = 0.

    This function is the same as all_gather_single but will propagate the
    backwards gradient across workers.

    See all_gather_single for more details on usage.
    """
    # The base all_gather_single now has autograd support, so we can just call it
    return all_gather_single(self, gather_dim, group, tag)


def reduce_scatter_single(
    self: torch.Tensor,
    reduceOp: str,
    scatter_dim: int,
    group: RANK_TYPES,
    tag: str = "",
):
    """
    Reduces the tensor data across all machines in such a way that all get
    the final result, then scatter the results to corresponding ranks.


    The input tensor is left unmodified.
    Group can be one of:
        List[int]: ranks participating in the collective.
        List[List[int]]: 2D mesh of ranks taking part of this collective in MPMD.
        ProcessGroup: Will perform a collective using the ranks and tag of the PG.
        DeviceMesh: Do a SPMD collective over all ranks of the mesh
        (DeviceMesh, int): Do a MPMD collective over one dimension of the DeviceMesh
    :: N.B. If you pass a PG or a 1D list to perform a MPMD collective, the compiler won't be able to recover
    that information and perform collective algebraic optimization. Use other forms of input for that.
    """
    group = _resolve_group(group, tag)
    group_size = c10d._get_group_size_by_name(group)

    if self.size(scatter_dim) % group_size != 0:
        raise AssertionError(
            f"input dimension 0 ({self.size(0)} must be a multiple of group_size {group_size})"
        )
    if scatter_dim != 0:
        self = _chunk_or_narrow_cat(self, group_size, narrow_dim=scatter_dim, cat_dim=0)

    tensor = torch.ops._c10d_functional.reduce_scatter_tensor(
        self,
        reduceOp.lower(),
        group_size,
        _group_or_group_name(group),
    )
    res = _maybe_wrap_tensor(tensor)
    return res


def reduce_scatter_single_autograd(
    self: torch.Tensor,
    reduceOp: str,
    scatter_dim: int,
    group: RANK_TYPES,
    tag: str = "",
):
    """
    Reduces the tensor data across all machines in such a way that all get
    the final result, then scatter the results to corresponding ranks.

    This function is the same as reduce_scatter_single but will propagate the
    backwards gradient across workers.

    Currently only the "sum" reduceOp is supported.

    See reduce_scatter_single for more details on usage.
    """
    # The base reduce_scatter_single now has autograd support, so we can just call it
    return reduce_scatter_single(self, reduceOp, scatter_dim, group, tag)


# The deprecated *_tensor aliases warn only outside compile: the
# typing_extensions @deprecated wrapper lives in a Dynamo-skipped module, and a
# raw warnings.warn is an untraceable builtin, either of which breaks
# fullgraph=True tracing of these still-supported entry points. Guarding on
# is_compiling() keeps the eager-time deprecation while letting Dynamo trace
# straight through to the _single implementation.
def all_gather_tensor(
    self: torch.Tensor,
    gather_dim: int,
    group: RANK_TYPES,
    tag: str = "",
) -> torch.Tensor:
    if not torch.compiler.is_compiling():
        warnings.warn(
            "`torch.distributed._functional_collectives.all_gather_tensor` is deprecated. "
            "Please use `torch.distributed._functional_collectives.all_gather_single` instead.",
            FutureWarning,
            stacklevel=2,
        )
    return all_gather_single(self, gather_dim, group, tag)


def all_gather_tensor_autograd(
    self: torch.Tensor,
    gather_dim: int,
    group: RANK_TYPES,
    tag: str = "",
):
    if not torch.compiler.is_compiling():
        warnings.warn(
            "`torch.distributed._functional_collectives.all_gather_tensor_autograd` is deprecated. "
            "Please use `torch.distributed._functional_collectives.all_gather_single_autograd` instead.",
            FutureWarning,
            stacklevel=2,
        )
    return all_gather_single_autograd(self, gather_dim, group, tag)


def reduce_scatter_tensor(
    self: torch.Tensor,
    reduceOp: str,
    scatter_dim: int,
    group: RANK_TYPES,
    tag: str = "",
):
    if not torch.compiler.is_compiling():
        warnings.warn(
            "`torch.distributed._functional_collectives.reduce_scatter_tensor` is deprecated. "
            "Please use `torch.distributed._functional_collectives.reduce_scatter_single` instead.",
            FutureWarning,
            stacklevel=2,
        )
    return reduce_scatter_single(self, reduceOp, scatter_dim, group, tag)


def reduce_scatter_tensor_autograd(
    self: torch.Tensor,
    reduceOp: str,
    scatter_dim: int,
    group: RANK_TYPES,
    tag: str = "",
):
    if not torch.compiler.is_compiling():
        warnings.warn(
            "`torch.distributed._functional_collectives.reduce_scatter_tensor_autograd` is deprecated. "
            "Please use `torch.distributed._functional_collectives.reduce_scatter_single_autograd` instead.",
            FutureWarning,
            stacklevel=2,
        )
    return reduce_scatter_single_autograd(self, reduceOp, scatter_dim, group, tag)


def all_reduce_coalesced(
    self: list[torch.Tensor], reduceOp: str, group: RANK_TYPES, tag: str = ""
) -> list[torch.Tensor]:
    """
    Reduces a list of tensors across all machines in such a way that all get
    the final result.

    The all tensors in the input list are left unmodified.

    Group can be one of:
        List[int]: ranks participating in the collective.
        List[List[int]]: 2D mesh of ranks taking part of this collective in MPMD.
        ProcessGroup: Will perform a collective using the ranks and tag of the PG.
        DeviceMesh: Do a SPMD collective over all ranks of the mesh
        (DeviceMesh, int): Do a MPMD collective over one dimension of the DeviceMesh

    :: N.B. If you pass a PG or a 1D list to perform a MPMD collective, the compiler won't be able to recover
    that information and perform collective algebraic optimization. Use other forms of input for that.
    """
    group = _resolve_group(group, tag)
    tensor_list = torch.ops._c10d_functional.all_reduce_coalesced(  # type: ignore[attr-defined]
        self,
        reduceOp.lower(),
        _group_or_group_name(group),
    )
    return list(map(_maybe_wrap_tensor, tensor_list))


def all_gather_single_coalesced(
    self: list[torch.Tensor], group: RANK_TYPES, tag: str = ""
) -> list[torch.Tensor]:
    """
    Gather a list of tensors across from all machines.

    Note that it currently only supports gather_dim = 0.

    The input tensor is left unmodified.
    Group can be one of:
        List[int]: ranks participating in the collective.
        List[List[int]]: 2D mesh of ranks taking part of this collective in MPMD.
        ProcessGroup: Will perform a collective using the ranks and tag of the PG.
        DeviceMesh: Do a SPMD collective over all ranks of the mesh
        (DeviceMesh, int): Do a MPMD collective over one dimension of the DeviceMesh

    :: N.B. If you pass a PG or a 1D list to perform a MPMD collective, the compiler won't be able to recover
    that information and perform collective algebraic optimization. Use other forms of input for that.
    """
    group = _resolve_group(group, tag)
    group_size = c10d._get_group_size_by_name(group)
    tensor_list = torch.ops._c10d_functional.all_gather_into_tensor_coalesced(  # type: ignore[attr-defined]
        self,
        group_size,
        _group_or_group_name(group),
    )
    return list(map(_maybe_wrap_tensor, tensor_list))


def reduce_scatter_single_coalesced(
    inputs: list[torch.Tensor],
    reduceOp: str,
    scatter_dim: list[int],
    group: RANK_TYPES,
    tag: str = "",
) -> list[torch.Tensor]:
    """
    Reduces a list of tensors across all machines in such a way that all get
    the final result, then scatter the results to corresponding ranks.

    The input tensors are left unmodified.
    Group can be one of:
        List[int]: ranks participating in the collective.
        List[List[int]]: 2D mesh of ranks taking part of this collective in MPMD.
        ProcessGroup: Will perform a collective using the ranks and tag of the PG.
        DeviceMesh: Do a SPMD collective over all ranks of the mesh
        (DeviceMesh, int): Do a MPMD collective over one dimension of the DeviceMesh

    :: N.B. If you pass a PG or a 1D list to perform a MPMD collective, the compiler won't be able to recover
    that information and perform collective algebraic optimization. Use other forms of input for that.
    """
    group = _resolve_group(group, tag)
    group_size = c10d._get_group_size_by_name(group)

    if len(scatter_dim) != len(inputs):
        raise AssertionError(
            f"Length of scatter_dim ({len(scatter_dim)}) must equal length of inputs ({len(inputs)})"
        )
    for idx, (dim, tensor) in enumerate(zip(scatter_dim, inputs)):
        if tensor.size(dim) % group_size != 0:
            raise AssertionError(
                f"input dimension {dim} ({tensor.size(dim)} must be a multiple of group_size {group_size} for tensor at index {idx}"
            )
        if dim != 0:
            tensor_list = torch.chunk(tensor, group_size, dim=dim)
            inputs[idx] = torch.cat(tensor_list)

    tensor_list = torch.ops._c10d_functional.reduce_scatter_tensor_coalesced(  # type: ignore[attr-defined]
        inputs,
        reduceOp.lower(),
        group_size,
        _group_or_group_name(group),
    )

    return list(map(_maybe_wrap_tensor, tensor_list))


# Guarded warning rather than the @deprecated wrapper so Dynamo can trace
# through to the _single_coalesced implementation; see the non-coalesced
# aliases above for the rationale.
def all_gather_into_tensor_coalesced(
    self: list[torch.Tensor], group: RANK_TYPES, tag: str = ""
) -> list[torch.Tensor]:
    if not torch.compiler.is_compiling():
        warnings.warn(
            "`torch.distributed._functional_collectives.all_gather_into_tensor_coalesced` is deprecated. "
            "Please use `torch.distributed._functional_collectives.all_gather_single_coalesced` instead.",
            FutureWarning,
            stacklevel=2,
        )
    return all_gather_single_coalesced(self, group, tag)


def reduce_scatter_tensor_coalesced(
    inputs: list[torch.Tensor],
    reduceOp: str,
    scatter_dim: list[int],
    group: RANK_TYPES,
    tag: str = "",
) -> list[torch.Tensor]:
    if not torch.compiler.is_compiling():
        warnings.warn(
            "`torch.distributed._functional_collectives.reduce_scatter_tensor_coalesced` is deprecated. "
            "Please use `torch.distributed._functional_collectives.reduce_scatter_single_coalesced` instead.",
            FutureWarning,
            stacklevel=2,
        )
    return reduce_scatter_single_coalesced(inputs, reduceOp, scatter_dim, group, tag)


# This is a bit unsafe: it checks if the first argument in the schema reports as a non-mutable alias.
# Today, this maps 1:1 with "aten ops that are views".
def _is_view_op(tgt):
    if not isinstance(tgt, torch._ops.OpOverload):
        raise AssertionError(f"Expected torch._ops.OpOverload, got {type(tgt)}")
    # Don't apply the view optimization to any `CompositeImplicitAutograd` ops.
    # See issue: https://github.com/pytorch/pytorch/issues/133421
    if torch._C._dispatch_has_kernel_for_dispatch_key(
        tgt.name(), torch.DispatchKey.CompositeImplicitAutograd
    ):
        return False
    schema = tgt._schema
    if len(schema.arguments) > 0:
        first_arg = schema.arguments[0]
        # check if op is a view
        return first_arg.alias_info is not None and not first_arg.alias_info.is_write


def all_to_all_single(
    self: torch.Tensor,
    output_split_sizes: list[int] | None,
    input_split_sizes: list[int] | None,
    group: RANK_TYPES,
    tag: str = "",
) -> torch.Tensor:
    """
    Each process splits input tensor and then scatters the split list
    to all processes in a group. Then concatenate the received tensors from all
    the processes in the group and return single output tensor.

    Group can be one of:
        List[int]: ranks participating in the collective.
        List[List[int]]: 2D mesh of ranks taking part of this collective in MPMD.
        ProcessGroup: Will perform a collective using the ranks and tag of the PG.
        DeviceMesh: Do a SPMD collective over all ranks of the mesh
        (DeviceMesh, int): Do a MPMD collective over one dimension of the DeviceMesh

    :: N.B. If you pass a PG or a 1D list to perform a MPMD collective, the compiler won't be able to recover
    that information and perform collective algebraic optimization. Use other forms of input for that.
    """
    if output_split_sizes is not None:
        if not all(
            isinstance(size, (int, torch.SymInt)) for size in output_split_sizes
        ):
            raise AssertionError(
                f"All output_split_sizes must be int or SymInt, got {output_split_sizes}"
            )
    if input_split_sizes is not None:
        if not all(isinstance(size, (int, torch.SymInt)) for size in input_split_sizes):
            raise AssertionError(
                f"All input_split_sizes must be int or SymInt, got {input_split_sizes}"
            )
    group = _resolve_group(group, tag)
    group_size = c10d._get_group_size_by_name(group)
    if output_split_sizes is None or input_split_sizes is None:
        if not (output_split_sizes is None and input_split_sizes is None):
            raise AssertionError(
                "output_split_sizes and input_split_sizes must either be "
                "specified together or both set to None"
            )
        output_split_sizes = [self.shape[0] // group_size] * group_size
        input_split_sizes = output_split_sizes
    tensor = torch.ops._c10d_functional.all_to_all_single(  # type: ignore[attr-defined]
        self,
        output_split_sizes,
        input_split_sizes,
        _group_or_group_name(group),
    )
    return _maybe_wrap_tensor(tensor)


def all_to_all_single_autograd(
    self: torch.Tensor,
    output_split_sizes: list[int] | None,
    input_split_sizes: list[int] | None,
    group: RANK_TYPES,
    tag: str = "",
) -> torch.Tensor:
    """
    Same as all_to_all_single but supports autograd.
    """
    # The base all_to_all_single now has autograd support, so we can just call it
    return all_to_all_single(self, output_split_sizes, input_split_sizes, group, tag)


# ============================================================================
# Collective Autograd Functions / Custom Ops
# ============================================================================


def wait_tensor_backward(ctx, grad_output: torch.Tensor):
    """
    Backward for wait_tensor: identity (no-op).
    Wait is just a synchronization primitive, so gradient flows through unchanged.

    Args:
        ctx: Context object
        grad_output: Gradient from downstream operations

    Returns:
        Gradient unchanged (identity)
    """
    return grad_output


def wait_tensor_setup_context(ctx, inputs, output):
    """
    Setup context for wait_tensor backward.
    Args:
        ctx: Context object to save state for backward
        inputs: Tuple of (tensor,)
        output: Output from forward pass
    """
    return


torch.library.register_autograd(
    "_c10d_functional::wait_tensor",
    wait_tensor_backward,
    setup_context=wait_tensor_setup_context,
)


def all_reduce_backward(ctx, grad_output: torch.Tensor):
    """
    Backward for all_reduce: all_reduce with same reduce_op.
    Forward aggregates tensors, backward aggregates gradients.

    Args:
        ctx: Context object
        grad_output: Gradient from downstream operations

    Returns:
        Tuple of (grad_input, grad_group_name, grad_reduce_op)
        grad_group_name and grad_reduce_op are None (not differentiable)
    """
    group_name = ctx.group_name
    reduce_op = ctx.reduce_op

    # Only linear reductions have a well-defined all_reduce backward: for both
    # 'sum' and 'avg' the gradient is an all_reduce of grad_output with the same
    # op (grad wrt each rank's input is the same reduction of the per-rank
    # grad_outputs). Nonlinear ops (min/max/product/premul_sum) do not.
    if reduce_op not in ("sum", "avg"):
        raise RuntimeError(
            f"all_reduce backward only supports 'sum' and 'avg' reductions, got '{reduce_op}'"
        )

    # Backward does all_reduce with the same reduce_op
    output = torch.ops._c10d_functional.all_reduce(
        grad_output.contiguous(), reduce_op, group_name
    )
    return wait_tensor(output), None, None


def all_reduce_setup_context(ctx, inputs, output):
    """
    Setup context for all_reduce backward.
    Args:
        ctx: Context object to save state for backward
        inputs: Tuple of (input, reduce_op, group_name)
        output: Output from forward pass
    """
    input, reduce_op, group_name = inputs
    ctx.group_name = group_name
    ctx.reduce_op = reduce_op.lower()


torch.library.register_autograd(
    "_c10d_functional::all_reduce",
    all_reduce_backward,
    setup_context=all_reduce_setup_context,
)


def all_gather_into_tensor_backward(ctx, grad_output: torch.Tensor):
    """
    Backward for all_gather_into_tensor: reduce_scatter with sum.

    Forward gathers tensors from all ranks, backward scatters gradients back
    with sum reduction.

    Args:
        ctx: Context object with group_name and group_size
        grad_output: Gradient from downstream operations

    Returns:
        Tuple of (grad_input, grad_group_size, grad_group_name)
        grad_group_size and grad_group_name are None (not differentiable)
    """
    group_name = ctx.group_name
    group_size = ctx.group_size

    # Backward is reduce_scatter with sum
    output = torch.ops._c10d_functional.reduce_scatter_tensor(
        grad_output.contiguous(),
        "sum",
        group_size,
        group_name,
    )
    return wait_tensor(output), None, None


def all_gather_into_tensor_setup_context(ctx, inputs, output):
    """
    Setup context for all_gather_into_tensor backward.

    Args:
        ctx: Context object to save state for backward
        inputs: Tuple of (input, group_size, group_name)
        output: Output from forward pass
    """
    input, group_size, group_name = inputs
    ctx.group_name = group_name
    ctx.group_size = group_size


torch.library.register_autograd(
    "_c10d_functional::all_gather_into_tensor",
    all_gather_into_tensor_backward,
    setup_context=all_gather_into_tensor_setup_context,
)


def reduce_scatter_tensor_backward(ctx, grad_output: torch.Tensor):
    """
    Backward for reduce_scatter_tensor: all_gather.

    Forward reduces and scatters tensors to ranks, backward gathers gradients
    from all ranks.

    Args:
        ctx: Context object with group_name, group_size, and reduce_op
        grad_output: Gradient from downstream operations

    Returns:
        Tuple of (grad_input, grad_reduce_op, grad_group_size, grad_group_name)
        grad_reduce_op, grad_group_size, grad_group_name are None (not differentiable)
    """
    group_name = ctx.group_name
    group_size = ctx.group_size
    reduce_op = ctx.reduce_op

    # Lazy validation: check reduce_op only when backward is called
    if reduce_op != "sum":
        raise RuntimeError(
            f"reduce_scatter_tensor backward only supports 'sum' reduction, got '{reduce_op}'"
        )

    # Backward is all_gather
    output = torch.ops._c10d_functional.all_gather_into_tensor(
        grad_output.contiguous(),
        group_size,
        group_name,
    )
    return wait_tensor(output), None, None, None


def reduce_scatter_tensor_setup_context(ctx, inputs, output):
    """
    Setup context for reduce_scatter_tensor backward.

    Args:
        ctx: Context object to save state for backward
        inputs: Tuple of (input, reduce_op, group_size, group_name)
        output: Output from forward pass
    """
    input, reduce_op, group_size, group_name = inputs
    ctx.group_name = group_name
    ctx.group_size = group_size
    ctx.reduce_op = reduce_op.lower()


torch.library.register_autograd(
    "_c10d_functional::reduce_scatter_tensor",
    reduce_scatter_tensor_backward,
    setup_context=reduce_scatter_tensor_setup_context,
)


def all_to_all_single_backward(ctx, grad_output: torch.Tensor):
    """
    Backward for all_to_all_single: all_to_all with reversed split sizes.

    Forward does all-to-all with specified split sizes, backward reverses them.

    Args:
        ctx: Context object with group_name, output_split_sizes, and input_split_sizes
        grad_output: Gradient from downstream operations

    Returns:
        Tuple of (grad_input, grad_output_split_sizes, grad_input_split_sizes, grad_group_name)
        All except grad_input are None (not differentiable)
    """
    group_name = ctx.group_name
    output_split_sizes = ctx.output_split_sizes
    input_split_sizes = ctx.input_split_sizes

    # Backward is all_to_all with reversed split sizes
    output = torch.ops._c10d_functional.all_to_all_single(
        grad_output.contiguous(),
        input_split_sizes,  # Reversed
        output_split_sizes,  # Reversed
        group_name,
    )
    return wait_tensor(output), None, None, None


def all_to_all_single_setup_context(ctx, inputs, output):
    """
    Setup context for all_to_all_single backward.

    Args:
        ctx: Context object to save state for backward
        inputs: Tuple of (input, output_split_sizes, input_split_sizes, group_name)
        output: Output from forward pass
    """
    input, output_split_sizes, input_split_sizes, group_name = inputs
    ctx.group_name = group_name
    ctx.output_split_sizes = output_split_sizes
    ctx.input_split_sizes = input_split_sizes


torch.library.register_autograd(
    "_c10d_functional::all_to_all_single",
    all_to_all_single_backward,
    setup_context=all_to_all_single_setup_context,
)

# Backward compatibility: register autograd for _c10d_functional_autograd ops
# These ops redirect to _c10d_functional but need their own autograd registration
torch.library.register_autograd(
    "_c10d_functional_autograd::all_gather_into_tensor",
    all_gather_into_tensor_backward,
    setup_context=all_gather_into_tensor_setup_context,
)

torch.library.register_autograd(
    "_c10d_functional_autograd::reduce_scatter_tensor",
    reduce_scatter_tensor_backward,
    setup_context=reduce_scatter_tensor_setup_context,
)

torch.library.register_autograd(
    "_c10d_functional_autograd::all_to_all_single",
    all_to_all_single_backward,
    setup_context=all_to_all_single_setup_context,
)


def all_reduce_coalesced_backward(ctx, grad_outputs: list[torch.Tensor]):
    """
    Backward for all_reduce_coalesced: all_reduce each gradient.

    Forward aggregates tensors, backward aggregates gradients.

    Args:
        ctx: Context object with group_name and reduce_op
        grad_outputs: Gradients from downstream operations (one per input tensor)

    Returns:
        Tuple of (grad_inputs..., grad_reduce_op, grad_group_name)
        grad_reduce_op and grad_group_name are None (not differentiable)
    """
    group_name = ctx.group_name
    reduce_op = ctx.reduce_op

    if reduce_op != "sum":
        raise RuntimeError(
            f"all_reduce_coalesced backward only supports 'sum' reduction, got '{reduce_op}'"
        )

    # Backward does all_reduce on list of gradients
    grad_inputs = torch.ops._c10d_functional.all_reduce_coalesced(
        [grad_output.contiguous() for grad_output in grad_outputs],
        reduce_op,
        group_name,
    )
    return (list(map(wait_tensor, grad_inputs)), None, None)


def all_reduce_coalesced_setup_context(ctx, inputs, output):
    """
    Setup context for all_reduce_coalesced backward.

    Args:
        ctx: Context object to save state for backward
        inputs: Tuple of (tensor_list, reduce_op, group_name)
        output: Output from forward pass
    """
    tensor_list, reduce_op, group_name = inputs
    ctx.group_name = group_name
    ctx.reduce_op = reduce_op.lower()


torch.library.register_autograd(
    "_c10d_functional::all_reduce_coalesced",
    all_reduce_coalesced_backward,
    setup_context=all_reduce_coalesced_setup_context,
)


def all_gather_into_tensor_coalesced_backward(ctx, grad_outputs: list[torch.Tensor]):
    """
    Backward for all_gather_into_tensor_coalesced: reduce_scatter each gradient.

    Forward gathers tensors from all ranks, backward scatters gradients back
    with sum reduction.

    Args:
        ctx: Context object with group_name and group_size
        grad_outputs: Gradients from downstream operations (one per input tensor)

    Returns:
        Tuple of (grad_inputs..., grad_group_size, grad_group_name)
        grad_group_size and grad_group_name are None (not differentiable)
    """
    group_name = ctx.group_name
    group_size = ctx.group_size

    # Backward does reduce_scatter on list of gradients
    grad_inputs = torch.ops._c10d_functional.reduce_scatter_tensor_coalesced(
        [grad_output.contiguous() for grad_output in grad_outputs],
        "sum",
        group_size,
        group_name,
    )
    return (list(map(wait_tensor, grad_inputs)), None, None)


def all_gather_into_tensor_coalesced_setup_context(ctx, inputs, output):
    """
    Setup context for all_gather_into_tensor_coalesced backward.

    Args:
        ctx: Context object to save state for backward
        inputs: Tuple of (tensor_list, group_size, group_name)
        output: Output from forward pass
    """
    tensor_list, group_size, group_name = inputs
    ctx.group_name = group_name
    ctx.group_size = group_size


torch.library.register_autograd(
    "_c10d_functional::all_gather_into_tensor_coalesced",
    all_gather_into_tensor_coalesced_backward,
    setup_context=all_gather_into_tensor_coalesced_setup_context,
)


def reduce_scatter_tensor_coalesced_backward(ctx, grad_outputs: list[torch.Tensor]):
    """
    Backward for reduce_scatter_tensor_coalesced: all_gather each gradient.

    Forward reduces and scatters tensors to ranks, backward gathers gradients
    from all ranks.

    Args:
        ctx: Context object with group_name, group_size, and reduce_op
        grad_outputs: Gradients from downstream operations (one per input tensor)

    Returns:
        Tuple of (grad_inputs..., grad_reduce_op, grad_group_size, grad_group_name)
        grad_reduce_op, grad_group_size, grad_group_name are None (not differentiable)
    """
    group_name = ctx.group_name
    group_size = ctx.group_size
    reduce_op = ctx.reduce_op

    # Lazy validation: check reduce_op only when backward is called
    if reduce_op != "sum":
        raise RuntimeError(
            f"reduce_scatter_tensor_coalesced backward only supports 'sum' reduction, got '{reduce_op}'"
        )

    # Backward does all_gather on list of gradients
    grad_inputs = torch.ops._c10d_functional.all_gather_into_tensor_coalesced(
        [grad_output.contiguous() for grad_output in grad_outputs],
        group_size,
        group_name,
    )
    return (list(map(wait_tensor, grad_inputs)), None, None, None)


def reduce_scatter_tensor_coalesced_setup_context(ctx, inputs, output):
    """
    Setup context for reduce_scatter_tensor_coalesced backward.

    Args:
        ctx: Context object to save state for backward
        inputs: Tuple of (tensor_list, reduce_op, group_size, group_name)
        output: Output from forward pass
    """
    tensor_list, reduce_op, group_size, group_name = inputs
    ctx.group_name = group_name
    ctx.group_size = group_size
    ctx.reduce_op = reduce_op.lower()


torch.library.register_autograd(
    "_c10d_functional::reduce_scatter_tensor_coalesced",
    reduce_scatter_tensor_coalesced_backward,
    setup_context=reduce_scatter_tensor_coalesced_setup_context,
)


def permute_tensor(
    self: torch.Tensor,
    src_dst: list[int],
    group: RANK_TYPES,
    tag: str = "",
) -> torch.Tensor:
    """
    Permutes the elements of the tensor according to the given source/destination pairs. `src_dst` should
    be defined such that src_dst[m] == n means m sends to n.

    Group can be one of:
        List[int]: ranks participating in the collective.
        List[List[int]]: 2D mesh of ranks taking part of this collective in MPMD.
        ProcessGroup: Will perform a collective using the ranks and tag of the PG.
        DeviceMesh: Do a SPMD collective over all ranks of the mesh
        (DeviceMesh, int): Do a MPMD collective over one
    """
    t, rankset, group_size = _expand_group(group, tag)
    local_pg = c10d._find_or_create_pg_by_ranks_and_tag(t, rankset, group_size)

    output_split_sizes = [0] * group_size
    input_split_sizes = [0] * group_size
    for src, dst in enumerate(src_dst):
        if src == dist.get_rank(local_pg):
            input_split_sizes[dst] = self.numel()
        if dst == dist.get_rank(local_pg):
            output_split_sizes[src] = self.numel()

    return all_to_all_single(self, output_split_sizes, input_split_sizes, group, tag)


class AsyncCollectiveTensor(torch.Tensor):
    r"""
    A Tensor wrapper subclass that is used to trigger a call to wait
    prior to first use of the underlying tensor.
    Use it inside functional collective pytorch wrappers like the following:
    def functional_collective(self, group, tag):
        tag, rankset, group_size = _expand_group(group, tag)
        tensor = torch.ops.c10d_functional.{collective}(self, tag, rankset, group_size)
        return _maybe_wrap_tensor(tensor)
    """

    elem: torch.Tensor
    completed: bool

    __slots__ = ["elem", "completed"]

    @staticmethod
    def __new__(cls, elem: torch.Tensor):
        r = torch.Tensor._make_wrapper_subclass(
            cls,
            elem.size(),
            strides=elem.stride(),
            storage_offset=elem.storage_offset(),
            dtype=elem.dtype,
            layout=elem.layout,
            device=elem.device,
            requires_grad=elem.requires_grad,
        )
        r.elem = elem
        r.completed = False
        return r

    def __tensor_flatten__(self):
        return ["elem"], None

    def tolist(self):
        return self.trigger_wait().tolist()

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        if meta is not None:
            raise AssertionError(
                "meta must be None for AsyncCollectiveTensor unflatten"
            )
        elem = inner_tensors["elem"]
        return AsyncCollectiveTensor(elem)

    def __coerce_same_metadata_as_tangent__(
        self, expected_metadata: Any, expected_type: type | None = None
    ):
        if expected_type is not torch.Tensor:
            return None

        return self.trigger_wait()

    def __repr__(self) -> str:  # type: ignore[override]
        return f"AsyncCollectiveTensor({self.trigger_wait()})"

    def trigger_wait(self):
        if not self.completed:
            out = wait_tensor(self.elem)
            self.completed = True
            return out
        else:
            return self.elem

    def wait(self) -> torch.Tensor:
        return wait_tensor(self.elem)

    def _get_acs_underlying_tensor(self):
        """This method enables  _functional_collectives_impl to test if a tensor is an ACS"""
        return self.elem

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):  # type: ignore[override]
        if func is torch.ops.aten.view.default:
            # Fast handle aten.view as a lot of view related op goes to aten.view
            # eventually, this avoids pytree slowdown

            res = func(args[0].elem, args[1])
            wrapper_res = AsyncCollectiveTensor(res)
            return wrapper_res

        is_view_op = _is_view_op(func)

        def unwrap(e: AsyncCollectiveTensor):
            # wait_tensor is idepotent and will do stream sync only once
            if not is_view_op:
                return e.trigger_wait()
            return e.elem

        def wrap(e: torch.Tensor):
            # wait_tensor is idepotent and will do stream sync only once
            if isinstance(e, AsyncCollectiveTensor):
                raise AssertionError(
                    "Cannot wrap an AsyncCollectiveTensor inside another AsyncCollectiveTensor"
                )
            res = AsyncCollectiveTensor(e)
            return res

        unwrapped_args = tree_map_only(AsyncCollectiveTensor, unwrap, args)
        unwrapped_kwargs = tree_map_only(AsyncCollectiveTensor, unwrap, kwargs)

        # we don't wrap the result as it doesn't need to be waited on.
        out = func(*unwrapped_args, **unwrapped_kwargs)

        # View ops don't require a sync, so we should re-wrap the outputs.
        if is_view_op:
            out = tree_map_only(torch.Tensor, wrap, out)

        return out

    def numpy(self):  # type: ignore[override]
        return self.wait().numpy()


"""
Utils and infrastructure for tracing support
"""


def _expand_group(group: RANK_TYPES, tag: str = "") -> tuple[str, list[int], int]:
    """
    _expand_group desugars the different RANK_TYPES types into a canonical format that is traceable.

    By having this be part of the explicit eager codepath, we avoid having to specialize behavior inside
    torchdynamo and can still interoperate with processgroup objects or other untraceable forms.
    """
    # had to define this hack _inside_ expand_group to avoid
    # graph_break [('torch.* op returned non-Tensor int
    # caused by 'cast_*` functions being treated as 'torch.*' ops (iiuc)
    if TYPE_CHECKING:

        def cast_listlistint(x):
            return cast(list[list[int]], x)

        def cast_listint(x):
            return cast(list[int], x)

    else:
        # fake cast op for use at runtime since dynamo doesn't support real cast
        # also, dynamo didn't like encountering 'typing' objects ()
        # NotImplementedError: argument of type: <class 'typing._GenericAlias'>
        def cast_listlistint(x):
            return x

        def cast_listint(x):
            return x

    rankset: list[int]
    if isinstance(group, list):
        if isinstance(group[0], list):
            nested_list = cast_listlistint(group)
            rankset = []
            group_size = -1
            for rs in nested_list:
                rankset.extend(rs)
                if group_size != -1 and group_size != len(rs):
                    raise ValueError(
                        f"group sizes must be identical found {group_size} and {len(rs)}"
                    )
                group_size = len(rs)
        else:
            rankset = cast_listint(group)
            group_size = len(rankset)
    elif isinstance(group, dist.ProcessGroup):
        rankset = dist.get_process_group_ranks(group)
        group_size = len(rankset)
        tag = tag or c10d._get_group_tag(group)
    elif isinstance(group, DeviceMesh):
        if group.ndim != 1:
            raise AssertionError(
                "Only 1D mesh is supported, pass in (DeviceMesh, int) together if mesh > 1D"
            )
        pg = group.get_group()
        rankset = dist.get_process_group_ranks(pg)
        group_size = len(rankset)
        tag = tag or c10d._get_group_tag(pg)
    elif isinstance(group, tuple):
        if (
            len(group) == 2
            and isinstance(group[0], DeviceMesh)
            and isinstance(group[1], int)
        ):
            dmesh = group[0]
            dim = group[1]
            pg = dmesh.get_group(dim)
            rankset = dist.get_process_group_ranks(pg)
            group_size = len(rankset)
            tag = tag or c10d._get_group_tag(pg)
        else:
            raise ValueError("Invalid tuple for group must be (DeviceMesh, int)")
    else:
        raise ValueError(
            "Invalid type for group, must be one of List, Processgroup, DeviceMesh or (DeviceMesh, int)."
        )

    return (tag, rankset, group_size)


def _resolve_group(
    group: RANK_TYPES, tag: str = ""
) -> dist.ProcessGroup | c10d.GroupName:
    """
    Given group in RANK_TYPES, return a ProcessGroup or group name.
    """
    # `tag` will be deprecated. See details in:
    # https://github.com/pytorch/pytorch/issues/93173#issuecomment-1907095208
    if isinstance(group, dist.ProcessGroup):
        return group
    elif isinstance(group, str):
        # In some cases Dynamo doesn't like tracing through NewType constructors
        # - so use a cast instead (the actual newtype representation is
        # literally the underlying type so this is fine). I haven't been able to
        # reproduce it in isolation (see T247631668).
        # pyrefly: ignore [redundant-cast]
        group_name = cast(c10d.GroupName, group)  # c10d.GroupName(group)
        return group_name
    elif isinstance(group, DeviceMesh):
        if group.ndim != 1:
            raise AssertionError(
                "Only 1D mesh is supported, pass in (DeviceMesh, int) together if mesh > 1D"
            )
        if torch.compiler.config.compile_on_one_rank:
            return torch.ops._dtensor.mesh_get_process_group(group, 0)
        return group._dim_group_names[0]
    elif isinstance(group, tuple):
        if (
            len(group) == 2
            and isinstance(group[0], DeviceMesh)
            and isinstance(group[1], int)
        ):
            dmesh = group[0]
            dim = group[1]
            if torch.compiler.config.compile_on_one_rank:
                return torch.ops._dtensor.mesh_get_process_group(dmesh, dim)
            return dmesh._dim_group_names[dim]
        else:
            raise ValueError(
                f"Invalid tuple for group must be (DeviceMesh, int). Instead got {(type(group[0]), type(group[1]))}"
            )
    elif isinstance(group, list):
        if not is_torchdynamo_compiling():
            warnings.warn(
                "The combination of ranks + tag as process group "
                "identifier has been deprecated. Please switch to "
                "using ProcessGroup, DeviceMesh, or group name instead.",
                FutureWarning,
                stacklevel=3,
            )
        return c10d._resolve_group_name_by_ranks_and_tag(
            # pyrefly: ignore [redundant-cast]
            cast(list[int], group),
            tag,
        )
    else:
        raise ValueError(f"Unsupported group type: {type(group)}, {group}")


def _resolve_group_name(group: RANK_TYPES, tag: str = "") -> c10d.GroupName:
    """
    Given group in RANK_TYPES, return the group name.
    """
    group = _resolve_group(group, tag)
    if isinstance(group, str):
        return c10d.GroupName(group)
    else:
        return group.group_name


@torch.library.custom_op(
    "_c10d_functional::_wrap_tensor_autograd",
    mutates_args=(),
    schema="(Tensor input) -> Tensor",
)
def _wrap_tensor_autograd(input: torch.Tensor) -> torch.Tensor:
    """
    Custom op that allows autograd to propagate
    from a normal Tensor to an AsyncCollectiveTensor.

    This is the low-level implementation. Users should call _maybe_wrap_tensor directly.

    Args:
        input: Input tensor to wrap in AsyncCollectiveTensor

    Returns:
        AsyncCollectiveTensor wrapping the input (or wait_tensor result if tracing)
    """
    return AsyncCollectiveTensor(input)


@_wrap_tensor_autograd.register_fake
def _(input: torch.Tensor) -> torch.Tensor:
    """
    Meta kernel for _wrap_tensor_autograd.
    """
    return torch.empty_like(input)


def _wrap_tensor_autograd_backward(ctx, grad_output: torch.Tensor):
    """
    Backward for _wrap_tensor_autograd: identity (no-op).

    The wrapping is just for async optimization, gradients flow through unchanged.

    Args:
        ctx: Context object (unused)
        grad_output: Gradient from downstream operations

    Returns:
        Gradient unchanged (identity)
    """
    return grad_output


def _wrap_tensor_autograd_setup_context(ctx, inputs, output):
    """
    Setup context for _wrap_tensor_autograd backward.

    Args:
        ctx: Context object to save state for backward (nothing to save)
        inputs: Tuple of (input,)
        output: Output from forward pass
    """
    return


_wrap_tensor_autograd.register_autograd(
    _wrap_tensor_autograd_backward,
    setup_context=_wrap_tensor_autograd_setup_context,
)


def _are_we_tracing() -> bool:
    if is_torchdynamo_compiling():
        return True
    # If fake mode is turned on, we are almost definitely compiling/tracing.
    if torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.FAKE) is not None:
        return True
    # See Note [enable_python_dispatcher in dynamo]
    if torch._C._dispatch_tls_is_dispatch_key_included(
        torch._C.DispatchKey.PythonDispatcher
    ):
        return True
    return get_proxy_mode() is not None


def _maybe_wrap_tensor(self) -> torch.Tensor:
    if _are_we_tracing():
        return wait_tensor(self)
    return _wrap_tensor_autograd(self)


@contextlib.contextmanager
def allow_inflight_collective_as_graph_input_ctx(value: bool = True):
    """
    Context manager to temporarily set whether inflight collectives are allowed as torch.compile graph inputs.
    Common use case is when the collective is issued in eager (with `async_op=True`) but waited in compiled region:
    ```
    def all_reduce_eager(x):
        y = x * x
        req = dist.all_reduce(y, op=dist.ReduceOp.SUM, async_op=True)
        return y


    @torch.compile(fullgraph=True)
    def all_reduce_wait_compiled(y):
        torch.ops.c10d_functional.wait_tensor(y)
        return y * y


    x = torch.ones(1280, 1280, device="cuda") + self.rank
    # the context manager ensures that `wait_tensor(y)` will wait on the correct work object
    with allow_inflight_collective_as_graph_input_ctx():
        y = all_reduce_eager(x)
        z = all_reduce_wait_compiled(y)
    ```
    With this context manager, when a collective is called, under the hood the work object of the collective
    will be registered in the work registry, and the wait_tensor() in compiled region called on
    the output tensor of the collective will wait on the correct work object.
    """
    previous = torch._C._distributed_c10d._allow_inflight_collective_as_graph_input()

    try:
        torch._C._distributed_c10d._set_allow_inflight_collective_as_graph_input(value)
        yield
    finally:
        torch._C._distributed_c10d._set_allow_inflight_collective_as_graph_input(
            previous
        )


def _make_all_gather_out_tensor(input, group_size):
    out_size = list(input.size())
    if len(out_size) == 0:
        out_size.append(group_size)
    else:
        out_size[0] *= group_size
    out_tensor = input.new_empty(out_size)
    return out_tensor


def _all_gather_into_tensor_coalesced_meta(self, tag, rankset, group_size):
    return [_make_all_gather_out_tensor(t, group_size) for t in self]


# We now register meta kernels to deal with tracing
def _broadcast_meta(self, *args):
    return torch.empty_like(self)


def _all_reduce_meta(self, *args):
    return torch.empty_like(self, memory_format=torch.contiguous_format)


def _wait_tensor_meta(self, *args):
    return torch.empty_like(self)


def _isend_meta(self, *args):
    return torch.empty(0, dtype=self.dtype, device=self.device)


def _irecv_meta(self, *args):
    return torch.empty_like(self)


def _batch_p2p_ops_meta(op_list, peer_list, tag_list, tensors, group_name):
    return [
        t if op == "irecv" else torch.empty(0, dtype=t.dtype, device=t.device)
        for op, t in zip(op_list, tensors)
    ]


def _all_gather_into_tensor_meta(shard, tag, rankset, group_size):
    return _make_all_gather_out_tensor(shard, group_size)


def _reduce_scatter_tensor_meta(input, reduce_op, tag, rankset, group_size):
    out_size = list(input.size())
    out_size[0] //= group_size
    return input.new_empty(out_size)


def _all_reduce_coalesced_meta(self, *args):
    return [torch.empty_like(t) for t in self]


def _all_reduce__meta(inp, *args):
    return inp


def _broadcast__meta(inp, *args):
    return inp


def _all_reduce_coalesced__meta(inputs, *args):
    return inputs


def _reduce_scatter_tensor_coalesced_meta(inputs, reduceOp, tag, rankset, group_size):
    def mk_out_tensor(input):
        out_size = list(input.size())
        out_size[0] //= group_size
        out_tensor = input.new_empty(out_size)
        return out_tensor

    return [mk_out_tensor(t) for t in inputs]


# NB: We often say all_to_all has dynamic output size, but this is not
# technically true: instead, what typically happens is you manually
# communicate the output_split_sizes ahead of time (which is dynamic),
# but then you pass those sizes explicitly, and the all to all itself
# isn't dynamic, it just follows the specified output splits
def _all_to_all_single_meta(
    input, output_split_sizes, input_split_sizes, *args, **kwargs
):
    if output_split_sizes is None:
        return input.new_empty(input.size())
    else:
        for s in output_split_sizes:
            torch._check(s >= 0)
        out_size = list(input.size())
        out_size[0] = sum(output_split_sizes)
        return input.new_empty(out_size)


def _all_gather_into_tensor_out_native_meta(input, group_size, group_name, *, out):
    return _make_all_gather_out_tensor(input, group_size)


def _all_gather_into_tensor_native_meta(input, group_size, group_name):
    return _make_all_gather_out_tensor(input, group_size)


def _all_gather_into_tensor_coalesced_native_meta(inputs, group_size, group_name):
    return [
        _all_gather_into_tensor_native_meta(input, group_size, group_name)
        for input in inputs
    ]


def _reduce_scatter_tensor_native_meta(inp, reduce_op, group_size, group_name):
    shape = list(inp.size())
    shape[0] //= group_size
    return inp.new_empty(shape)


def _reduce_scatter_tensor_out_native_meta(
    inp, reduce_op, group_size, group_name, *, out
):
    shape = list(inp.size())
    shape[0] //= group_size
    return inp.new_empty(shape)


def _reduce_scatter_tensor_coalesced_native_meta(
    inputs, reduce_op, group_size, group_name
):
    return [
        _reduce_scatter_tensor_native_meta(inp, reduce_op, group_size, group_name)
        for inp in inputs
    ]


# Library MUST be defined at module scope or it doesn't work
lib_impl = torch.library.Library("_c10d_functional", "IMPL")
lib_impl.impl("all_reduce", _all_reduce_meta, "Meta")
lib_impl.impl("all_reduce_", _all_reduce__meta, "Meta")
lib_impl.impl("all_reduce_coalesced", _all_reduce_coalesced_meta, "Meta")
lib_impl.impl("all_reduce_coalesced_", _all_reduce_coalesced__meta, "Meta")
lib_impl.impl("wait_tensor", _wait_tensor_meta, "Meta")
lib_impl.impl("isend", _isend_meta, "Meta")
lib_impl.impl("irecv", _irecv_meta, "Meta")
lib_impl.impl("batch_p2p_ops", _batch_p2p_ops_meta, "Meta")

lib_impl.impl(
    "all_gather_into_tensor_out", _all_gather_into_tensor_out_native_meta, "Meta"
)
lib_impl.impl("all_gather_into_tensor", _all_gather_into_tensor_native_meta, "Meta")
lib_impl.impl(
    "all_gather_into_tensor_coalesced",
    _all_gather_into_tensor_coalesced_native_meta,
    "Meta",
)
lib_impl.impl("reduce_scatter_tensor", _reduce_scatter_tensor_native_meta, "Meta")
lib_impl.impl(
    "reduce_scatter_tensor_out", _reduce_scatter_tensor_out_native_meta, "Meta"
)
lib_impl.impl(
    "reduce_scatter_tensor_coalesced",
    _reduce_scatter_tensor_coalesced_native_meta,
    "Meta",
)
lib_impl.impl("all_to_all_single", _all_to_all_single_meta, "Meta")
lib_impl.impl("broadcast", _broadcast_meta, "Meta")
lib_impl.impl("broadcast_", _broadcast__meta, "Meta")

# Backward compatibility: register Meta impls for _c10d_functional_autograd ops
# These ops redirect to _c10d_functional but need their own Meta registration so
# torch.compile / FakeTensor mode can dispatch them.
lib_impl_autograd = torch.library.Library("_c10d_functional_autograd", "IMPL")
lib_impl_autograd.impl(
    "all_gather_into_tensor", _all_gather_into_tensor_native_meta, "Meta"
)
lib_impl_autograd.impl(
    "reduce_scatter_tensor", _reduce_scatter_tensor_native_meta, "Meta"
)
lib_impl_autograd.impl("all_to_all_single", _all_to_all_single_meta, "Meta")

# Mark these ops as side effectful so that DCE does not remove communication
# whose result tensors are ignored by user code.
torch.fx.node.has_side_effect(torch.ops._c10d_functional.wait_tensor.default)  # type: ignore[has-type]
torch.fx.node.has_side_effect(torch.ops._c10d_functional.wait_tensor)  # type: ignore[has-type]
torch.fx.node.has_side_effect(torch.ops._c10d_functional.isend.default)  # type: ignore[has-type]
torch.fx.node.has_side_effect(torch.ops._c10d_functional.isend)  # type: ignore[has-type]
torch.fx.node.has_side_effect(torch.ops._c10d_functional.irecv.default)  # type: ignore[has-type]
torch.fx.node.has_side_effect(torch.ops._c10d_functional.irecv)  # type: ignore[has-type]
torch.fx.node.has_side_effect(torch.ops._c10d_functional.batch_p2p_ops.default)  # type: ignore[has-type]
torch.fx.node.has_side_effect(torch.ops._c10d_functional.batch_p2p_ops)  # type: ignore[has-type]


# Register legacy ops for backward compatibility
# TODO(yifu): remove these in functional collective beta release
legacy_lib = torch.library.Library("c10d_functional", "DEF")
legacy_lib_impl = torch.library.Library("c10d_functional", "IMPL")
ops_defs = [
    "broadcast(Tensor self, int src, str tag, int[] ranks, int group_size) -> Tensor",
    "all_reduce(Tensor self, str reduceOp, str tag, int[] ranks, int group_size) -> Tensor",
    "all_reduce_coalesced(Tensor[] self, str reduceOp, str tag, int[] ranks, int group_size) -> Tensor[]",
    "wait_tensor(Tensor self) -> Tensor",
    "isend(Tensor self, int dst, int tag, str group_name) -> Tensor",
    "irecv(Tensor self, int src, int tag, str group_name) -> Tensor",
    "batch_p2p_ops(str[] op_list, int[] peer_list, int[] tag_list, Tensor[] tensors, str group_name) -> Tensor[]",
    "all_gather_into_tensor(Tensor shard, str tag, int[] ranks, int group_size) -> Tensor",
    "all_gather_into_tensor_coalesced(Tensor[] input, str tag, int[] ranks, int group_size) -> Tensor[]",
    "reduce_scatter_tensor(Tensor input, str reduceOp, str tag, int[] ranks, int group_size) -> Tensor",
    "reduce_scatter_tensor_coalesced(Tensor[] inputs, str reduceOp, str tag, int[] ranks, int group_size) -> Tensor[]",
    "all_to_all_single(Tensor input, SymInt[]? output_split_sizes, SymInt[]? input_split_sizes, str tag, int[] ranks, int group_size) -> Tensor",
]

my_module = sys.modules[__name__]
for op_def in ops_defs:
    op_name = op_def[0 : op_def.index("(")]
    backend_impl = getattr(fun_col_impl, f"_{op_name}")
    legacy_lib.define(op_def, tags=torch.Tag.pt2_compliant_tag)
    legacy_lib_impl.impl(op_name, backend_impl, "CompositeImplicitAutograd")


"""
Dynamo Remappings allow seamless translation from non-functional collectives of supportable form into
functional collective calls followed by inplace copy ops, allowing them to be traced into a functional graph.

We implement this by writing a decomposition and teaching dynamo how to associate it to a corresponding op via
the mapping dict below.

These schemas intentionally match torch.distributed.distributed_c10d.* ops that we are trying to remap from
"""


def all_gather_tensor_inplace(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    group=None,  # TODO add a type,
    async_op: bool = False,
    tag: str = "",
    gather_dim: int = 0,
):
    if async_op:
        raise AssertionError(
            "Can't remap async version of inplace op to functional collective"
        )

    group = group or dist.group.WORLD
    if group is None:
        raise AssertionError("group cannot be None")

    return output_tensor.copy_(all_gather_single(input_tensor, gather_dim, group, tag))


def reduce_scatter_tensor_inplace(
    output: torch.Tensor,
    input: torch.Tensor,
    op: str = "sum",  # TODO type is actually c10d ReduceOp. is this ok?
    group=None,  # TODO add a type
    async_op: bool = False,
    scatter_dim: int = 0,
    tag: str = "",
):
    if async_op:
        raise AssertionError(
            "Can't remap async version of inplace op to functional collective"
        )

    group = group or dist.group.WORLD
    if group is None:
        raise AssertionError("group cannot be None")

    return output.copy_(reduce_scatter_single(input, op, scatter_dim, group, tag))


REDUCE_OP_TO_STR = {
    dist.ReduceOp.SUM: "sum",
    dist.ReduceOp.AVG: "avg",
    dist.ReduceOp.PRODUCT: "product",
    dist.ReduceOp.MIN: "min",
    dist.ReduceOp.MAX: "max",
    dist.ReduceOp.BAND: "band",
    dist.ReduceOp.BOR: "bor",
    dist.ReduceOp.BXOR: "bxor",
}


def all_reduce_inplace(
    tensor: torch.Tensor,
    op: str = "sum",
    group=None,
    async_op: bool = False,
    tag: str = "",
):
    if async_op:
        raise AssertionError(
            "Can't remap async version of inplace op to functional collective"
        )

    group = group or dist.group.WORLD
    if group is None:
        raise AssertionError("group cannot be None")

    return tensor.copy_(all_reduce(tensor, op, group, tag))


def all_to_all_inplace(
    output: torch.Tensor,
    input: torch.Tensor,
    output_split_sizes=None,
    input_split_sizes=None,
    group=None,
    async_op=False,
    tag: str = "",
):
    if async_op:
        raise AssertionError(
            "Can't remap async version of inplace op to functional collective"
        )

    group = group or dist.group.WORLD
    if group is None:
        raise AssertionError("group cannot be None")

    return output.copy_(
        all_to_all_single(
            input,
            output_split_sizes,
            input_split_sizes,
            group,
            tag,
        )
    )


def all_gather_inplace(
    tensor_list: list[torch.Tensor],
    tensor: torch.Tensor,
    group=None,
    async_op=False,
    tag: str = "",
):
    if async_op:
        raise AssertionError(
            "Can't remap async version of inplace op to functional collective"
        )
    if tensor.dim() != 0 and not all(t.size(0) == tensor.size(0) for t in tensor_list):
        raise AssertionError("Remapping variable size all_gather is not yet supported")

    group = group or dist.group.WORLD
    if group is None:
        raise AssertionError("group cannot be None")

    output = all_gather_single(tensor, 0, group, tag)

    # Use aten.slice instead of aten.split because the latter causes
    # tensor.shape(0) to be unnecessarily baked in when it's a SymInt.
    output_splits = []
    offset = 0
    for t in tensor_list:
        is_scalar = t.dim() == 0
        t_offset = 1 if is_scalar else t.size(0)

        out = output[offset] if is_scalar else output[offset : offset + t_offset]
        output_splits.append(out)

        offset += t_offset
    for dst, src in zip(tensor_list, output_splits):
        dst.copy_(src)
    return tensor_list


def reduce_scatter_inplace(
    output: torch.Tensor,
    input_list: list[torch.Tensor],
    op: str = "sum",
    group: dist.ProcessGroup | None = None,
    async_op: bool = False,
    tag: str = "",
):
    if async_op:
        raise AssertionError(
            "Can't remap async version of inplace op to functional collective"
        )
    if not all(t.size() == output.size() for t in input_list):
        raise AssertionError(
            "reduce_scatter requires every input_list element to have the same size as output"
        )

    group = group or dist.group.WORLD
    if group is None:
        raise AssertionError("group cannot be None")

    if output.dim() == 0:
        # scalars have no dim to scatter along; stack into 1-D then reshape back
        result = reduce_scatter_single(torch.stack(input_list), op, 0, group, tag)
        return output.copy_(result.reshape(output.shape))
    return output.copy_(reduce_scatter_single(torch.cat(input_list), op, 0, group, tag))


def isend_inplace(
    tensor: torch.Tensor,
    dst: int | None = None,
    tag: int = 0,
    group: dist.ProcessGroup | None = None,
    group_dst: int = -1,
):
    if group is None:
        group = dist.group.WORLD
    if group is None:
        raise AssertionError("group cannot be None")
    if group_dst != -1:
        if dst is not None:
            raise ValueError(
                "Cannot specify both 'dst' and 'group_dst' args as per eager impl"
            )
        local_dst = group_dst
    elif dst is not None:
        local_dst = c10d.get_group_rank(group, dst)
    else:
        raise ValueError("Must specify either 'dst' or 'group_dst'")

    group_name = _resolve_group_name(group)
    tensor = torch.ops._c10d_functional.isend(tensor, local_dst, tag, group_name)
    if _are_we_tracing():
        return tensor
    return _maybe_wrap_tensor(tensor)


def irecv_inplace(
    tensor: torch.Tensor,
    src: int | None = None,
    tag: int = 0,
    group: dist.ProcessGroup | None = None,
    group_src: int = -1,
):
    if group is None:
        group = dist.group.WORLD
    if group is None:
        raise AssertionError("group cannot be None")
    if group_src != -1:
        if src is not None:
            raise ValueError(
                "Cannot specify both 'src' and 'group_src' args as per eager impl"
            )
        local_src = group_src
    elif src is not None:
        local_src = c10d.get_group_rank(group, src)
    else:
        raise ValueError("Must specify either 'src' or 'group_src'")
    group_name = _resolve_group_name(group)
    tensor = torch.ops._c10d_functional.irecv(tensor, local_src, tag, group_name)
    return _maybe_wrap_tensor(tensor)


def batch_p2p_ops_inplace(
    op_list: list[str],
    peer_list: list[int],
    tag_list: list[int],
    tensors: list[torch.Tensor],
    group_name: RANK_TYPES,
):
    if not dist.is_initialized():
        raise AssertionError("torch.distributed must be initialized")
    if group_name is None or group_name == "":
        group_name = c10d._get_default_group()
    resolved = _resolve_group(group_name)
    group_name = resolved if isinstance(resolved, str) else resolved.group_name
    tensors = torch.ops._c10d_functional.batch_p2p_ops(
        op_list, peer_list, tag_list, tensors, group_name
    )
    if _are_we_tracing():
        return [
            _maybe_wrap_tensor(t) if op == "irecv" else t
            for op, t in zip(op_list, tensors)
        ]
    return list(map(_maybe_wrap_tensor, tensors))


def _group_or_group_name(
    group: dist.ProcessGroup | c10d.GroupName,
) -> dist.ProcessGroup | c10d.GroupName:
    if isinstance(group, str):
        return group
    elif torch.compiler.config.compile_on_one_rank:
        return group
    else:
        return group.group_name


from torch.distributed.distributed_c10d import (  # pyrefly: ignore  # deprecated; pyrefly: ignore [deprecated]
    _all_gather_base as legacy_all_gather_base,
    _reduce_scatter_base as legacy_reduce_scatter_base,
    all_gather as legacy_all_gather,
    all_gather_into_tensor as legacy_allgather,
    all_gather_single as legacy_allgather_single,
    all_reduce as legacy_allreduce,
    all_to_all_single as legacy_all_to_all_single,
    batch_isend_irecv as legacy_batch_p2p_ops,
    irecv as legacy_irecv,
    isend as legacy_isend,
    reduce_scatter as legacy_reduce_scatter,
    reduce_scatter_single as legacy_reducescatter_single,
    reduce_scatter_tensor as legacy_reducescatter,
)


# Dynamo remaps dist.* collectives to these wrappers via traceable_collective_remaps.
# Each wrapper calls the in-place functional collective and returns None,
# matching the return type of the original dist.* APIs when async_op=False.
# async_op=True already graph-breaks in CollectiveFunctionRewriteVariable.
# These must be module-level def statements (not closures from a decorator factory)
# because _traceable_collectives_source resolves Dynamo guard sources by looking up
# fn.__name__ as a module attribute — a def's __name__ matches its variable name
# automatically, whereas a closure's would not.
def _remapped_allgather(*args, **kwargs):
    if not _are_we_tracing():
        raise AssertionError("_remapped_allgather should only be called during tracing")
    all_gather_tensor_inplace(*args, **kwargs)


def _remapped_reducescatter(*args, **kwargs):
    if not _are_we_tracing():
        raise AssertionError(
            "_remapped_reducescatter should only be called during tracing"
        )
    reduce_scatter_tensor_inplace(*args, **kwargs)


def _remapped_allreduce(*args, **kwargs):
    if not _are_we_tracing():
        raise AssertionError("_remapped_allreduce should only be called during tracing")
    all_reduce_inplace(*args, **kwargs)


def _remapped_all_to_all_single(*args, **kwargs):
    if not _are_we_tracing():
        raise AssertionError(
            "_remapped_all_to_all_single should only be called during tracing"
        )
    all_to_all_inplace(*args, **kwargs)


def _remapped_all_gather(*args, **kwargs):
    if not _are_we_tracing():
        raise AssertionError(
            "_remapped_all_gather should only be called during tracing"
        )
    all_gather_inplace(*args, **kwargs)


def _remapped_reduce_scatter(*args, **kwargs):
    if not _are_we_tracing():
        raise AssertionError(
            "_remapped_reduce_scatter should only be called during tracing"
        )
    reduce_scatter_inplace(*args, **kwargs)


def _remapped_isend(*args, **kwargs):
    if not _are_we_tracing():
        raise AssertionError("_remapped_isend should only be called during tracing")
    return isend_inplace(*args, **kwargs)


def _remapped_irecv(*args, **kwargs):
    if not _are_we_tracing():
        raise AssertionError("_remapped_irecv should only be called during tracing")
    return irecv_inplace(*args, **kwargs)


def _remapped_batch_p2p_ops(*args, **kwargs):
    if not _are_we_tracing():
        raise AssertionError(
            "_remapped_batch_p2p_ops should only be called during tracing"
        )
    return batch_p2p_ops_inplace(*args, **kwargs)


# This dict should contain sets of functions that dynamo is allowed to remap.
# Functions in this set should accept the same args/kwargs 1:1 as their mapping.
traceable_collective_remaps = {
    legacy_allgather: _remapped_allgather,
    legacy_allgather_single: _remapped_allgather,
    legacy_reducescatter: _remapped_reducescatter,
    legacy_reducescatter_single: _remapped_reducescatter,
    legacy_allreduce: _remapped_allreduce,
    legacy_all_to_all_single: _remapped_all_to_all_single,
    legacy_all_gather: _remapped_all_gather,
    legacy_reduce_scatter: _remapped_reduce_scatter,
    legacy_reduce_scatter_base: _remapped_reducescatter,
    legacy_all_gather_base: _remapped_allgather,
    legacy_isend: _remapped_isend,
    legacy_irecv: _remapped_irecv,
    legacy_batch_p2p_ops: _remapped_batch_p2p_ops,
}


def _remap_traceable_collective(
    func: Callable[..., Any],
    args: Sequence[Any],
    kwargs: dict[str, Any] | None,
) -> tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]] | None:
    """Redirect a legacy ``torch.distributed`` collective to its functional
    equivalent, following Dynamo's CollectiveFunctionRewriteVariable.

    Returns ``(mapped_func, (), kwargs)`` if ``func`` is a remappable collective,
    else None. Shared by the make_fx compile_on_one_rank mode below and
    non-strict export's _NonStrictTorchFunctionHandler.
    """
    if func not in traceable_collective_remaps:
        return None
    import inspect

    mapped_func = traceable_collective_remaps[func]
    bound = dict(inspect.signature(func).bind(*args, **(kwargs or {})).arguments)
    if func in (
        torch.distributed.all_reduce,
        torch.distributed.reduce_scatter,
        torch.distributed.reduce_scatter_single,
        torch.distributed.reduce_scatter_tensor,
        torch.distributed._reduce_scatter_base,
    ):
        if "op" in bound:
            bound["op"] = REDUCE_OP_TO_STR[bound["op"]]
    return mapped_func, (), bound


class _LegacyToFunctionalCollectiveMode(torch.overrides.TorchFunctionMode):
    """Redirect legacy ``torch.distributed`` collectives to functional collectives
    while tracing, mirroring Dynamo and non-strict export.

    Used by ``make_fx`` under ``compile_on_one_rank``: the in-place ``c10d.*`` ops
    bind the ProcessGroup as a torchbind constant that cannot be serialized,
    whereas the functional collectives take the group as an op argument, so the
    group (e.g. a ``DeviceMesh.get_group()`` -> ``mesh_get_process_group`` output)
    flows into the graph instead of being baked in.
    """

    def __torch_function__(
        self,
        func: Callable[..., Any],
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        remapped = _remap_traceable_collective(func, args, kwargs)
        if remapped is not None:
            mapped_func, mapped_args, mapped_kwargs = remapped
            return mapped_func(*mapped_args, **mapped_kwargs)
        return func(*args, **(kwargs or {}))
<|endoftext|># mypy: allow-untyped-defs
import dataclasses
import traceback
from collections import OrderedDict
from collections.abc import Callable, Container
from typing import Any, Optional, overload, TypeVar

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.utils.rnn import PackedSequence


__all__ = []  # type: ignore[var-annotated]


def _pack_kwargs(*args: Any, **kwargs: Any) -> tuple[tuple[Any, ...], tuple[str, ...]]:
    """
    Turn argument list into separate key list and value list (unpack_kwargs does the opposite).

    Inspiration: https://github.com/facebookresearch/fairscale/blob/eeb6684/fairscale/internal/containers.py#L70
    Usage::

        kwarg_keys, flat_args = pack_kwargs(1, 2, a=3, b=4)
        assert kwarg_keys == ("a", "b")
        assert flat_args == (1, 2, 3, 4)
        args, kwargs = unpack_kwargs(kwarg_keys, flat_args)
        assert args == (1, 2)
        assert kwargs == {"a": 3, "b": 4}
    Returns:
        Tuple[Tuple[Any, ...], Tuple[str, ...]]: The first tuple element gives
        gives both positional args and kwarg values, where the positional args
        proceed kwarg values and kwarg values are ordered consistently with the
        kwarg keys. The second tuple element gives the kwarg keys.
        The second tuple element's length is at most the first tuple element's length.
    """
    kwarg_keys: list[str] = []
    flat_args: list[Any] = list(args)
    for k, v in kwargs.items():
        kwarg_keys.append(k)
        flat_args.append(v)

    return tuple(flat_args), tuple(kwarg_keys)


def _cast_forward_inputs(
    dtype: torch.dtype | None,
    *args: Any,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """
    Cast floating point tensors in ``args`` and ``kwargs`` to ``input_dtype``.

    This respects the existing ``requires_grad`` on the tensors.
    """
    if dtype is None:
        return args, kwargs

    def cast_fn(x: torch.Tensor) -> torch.Tensor:
        if not torch.is_floating_point(x) or x.dtype == dtype:
            return x

        return x.to(dtype)

    return (_apply_to_tensors(cast_fn, args), _apply_to_tensors(cast_fn, kwargs))


def _unpack_kwargs(
    flat_args: tuple[Any, ...], kwarg_keys: tuple[str, ...]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """See _pack_kwargs."""
    if len(kwarg_keys) > len(flat_args):
        raise AssertionError(f"too many keys {len(kwarg_keys)} vs. {len(flat_args)}")
    if len(kwarg_keys) == 0:
        return flat_args, {}
    args = flat_args[: -len(kwarg_keys)]
    kwargs = dict(zip(kwarg_keys, flat_args[-len(kwarg_keys) :]))
    return args, kwargs


S = TypeVar("S", dict, list, tuple)
T = TypeVar("T", torch.Tensor, PackedSequence)


@overload
def _recursive_to(
    inputs: S, target_device: torch.device, use_side_stream_for_tensor_copies: bool
) -> list[S]: ...


@overload
def _recursive_to(
    inputs: T, target_device: torch.device, use_side_stream_for_tensor_copies: bool
) -> tuple[T]: ...


def _recursive_to(inputs, target_device, use_side_stream_for_tensor_copies):
    r"""Recursively moves input to the target_device."""

    def to_map(obj):
        if isinstance(obj, (torch.Tensor, PackedSequence)):
            device = obj.data.device if isinstance(obj, PackedSequence) else obj.device
            if device == target_device:
                return (obj,)
            if not use_side_stream_for_tensor_copies:
                return (obj.to(target_device),)
            else:
                # If the custom module is not registered to torch, stream is not used for acceleration
                if device.type == "cpu":
                    return (obj.to(target_device),)

                from torch.nn.parallel._functions import _get_stream

                # Perform CPU -> target_device copies in a background stream. This code is
                # motivated from similar logic in torch/nn/parallel/_functions.py
                stream = _get_stream(target_device)
                with stream:
                    output = obj.to(target_device)
                # synchronize with the copy stream
                with torch.accelerator.device_index(target_device.index):
                    current_stream = torch.accelerator.current_stream()
                    # Sync the current stream with the copy stream
                    current_stream.wait_stream(stream)
                    # Ensure tensor memory is not reused until work on
                    # main stream is complete
                    if isinstance(obj, PackedSequence):
                        output.data.record_stream(current_stream)  # type: ignore[arg-type]
                    else:
                        if not isinstance(output, torch.Tensor):
                            raise AssertionError("output must be a torch.Tensor")
                        output.record_stream(current_stream)  # type: ignore[arg-type]
                return (output,)

        from torch.nn.parallel.scatter_gather import _is_namedtuple

        def _handle_container(obj, elements, make_container):
            """Handle container types with object identity preservation."""
            # pyrefly: ignore [bad-argument-type]
            mapped = list(map(to_map, elements))
            # Preserve object identity when all elements are unchanged (single-device case)
            if all(len(m) == 1 for m in mapped):
                transformed = [m[0] for m in mapped]
                if all(t is o for t, o in zip(transformed, elements)):
                    return [obj]
                # pyrefly: ignore [no-matching-overload]
                return [make_container(transformed)]
            # pyrefly: ignore [no-matching-overload]
            return [make_container(args) for args in zip(*mapped)]

        if _is_namedtuple(obj):
            return _handle_container(obj, obj, lambda x: type(obj)(*x))
        if isinstance(obj, tuple) and len(obj) > 0:
            return _handle_container(obj, obj, tuple)
        if isinstance(obj, list) and len(obj) > 0:
            return _handle_container(obj, obj, list)
        if isinstance(obj, dict) and len(obj) > 0:
            keys = list(obj.keys())
            return _handle_container(
                obj, obj.values(), lambda v: type(obj)(zip(keys, v))
            )
        return [obj]

    # Avoid reference cycle
    try:
        res = to_map(inputs)
    finally:
        to_map = None  # type: ignore[assignment]
    return res


def _p_assert(cond: Any, s: str, raise_assertion_error: bool = True) -> None:
    """Alternate to ``assert`` when in the backward context to print the error message ``s`` since otherwise, it is swallowed."""
    if not cond:
        print(s)
        traceback.print_stack()
        if raise_assertion_error:
            raise AssertionError(s)


def _alloc_storage(tensor: torch.Tensor, size: torch.Size) -> None:
    """
    Allocate storage for ``tensor`` with the given size.

    Returns:
        bool: ``True`` if this method allocated storage and ``False`` if the
        storage was already allocated.
    """
    with torch.no_grad():
        if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
            already_allocated = tensor._typed_storage()._size() == size.numel()
            if not already_allocated:
                tensor_storage_size = tensor._typed_storage()._size()
                _p_assert(
                    tensor_storage_size == 0,
                    "Tensor storage should have been resized to be 0 but got PLACEHOLDER",
                )
                tensor._typed_storage()._resize_(size.numel())


def _free_storage(tensor: torch.Tensor):
    """
    Frees the underlying storage of ``tensor``.

    Returns:
        bool: ``True`` if the method freed the storage and ``False`` if the
        storage was already freed.
    """
    with torch.no_grad():
        if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
            already_freed = tensor._typed_storage()._size() == 0
            if not already_freed:
                _p_assert(
                    tensor.storage_offset() == 0,
                    "Freeing a tensor's storage is unsafe when it is not the sole occupant\n"
                    f"storage offset: {tensor.storage_offset()}\n"
                    f"storage size: {tensor._typed_storage()._size()}\n"
                    f"tensor shape: {tensor.shape}",
                )
                tensor._typed_storage()._resize_(0)


Q = TypeVar("Q")
R = TypeVar("R", dict, list, tuple, set, OrderedDict, PackedSequence, Any)


@overload
def _apply_to_tensors(
    fn: Callable[[torch.Tensor], Q], container: torch.Tensor
) -> Q: ...


@overload
def _apply_to_tensors(fn: Callable[[torch.Tensor], Any], container: R) -> R: ...


def _apply_to_tensors(fn, container):
    """Recursively apply to all tensor in different kinds of container types."""

    def apply(x):
        from torch.nn.parallel.scatter_gather import _is_namedtuple

        if isinstance(x, torch.Tensor):
            return fn(x)
        elif hasattr(x, "__dataclass_fields__"):
            dc = dataclasses.replace(x)
            changes = {
                f.name: apply(getattr(dc, f.name)) for f in dataclasses.fields(dc)
            }
            return dataclasses.replace(dc, **changes)
        elif isinstance(x, OrderedDict):
            od = x.__class__()
            for key, value in x.items():
                od[key] = apply(value)
            return od
        elif isinstance(x, PackedSequence):
            apply(x.data)
            return x
        elif isinstance(x, dict):
            return {key: apply(value) for key, value in x.items()}
        elif _is_namedtuple(x):
            res = (apply(el) for el in x)
            return type(x)(*res)
        elif isinstance(x, (list, tuple, set)):
            return type(x)(apply(el) for el in x)
        else:
            return x

    return apply(container)


def _to_kwargs(
    inputs: tuple[Any, ...],
    kwargs: dict[str, Any] | None,
    target_device: torch.device,
    use_side_stream_for_tensor_copies: bool,
) -> tuple[tuple[Any, ...], tuple[dict[str, Any], ...]]:
    moved_inputs = (
        _recursive_to(inputs, target_device, use_side_stream_for_tensor_copies)
        if inputs
        else []
    )
    moved_kwargs = (
        _recursive_to(kwargs, target_device, use_side_stream_for_tensor_copies)
        if kwargs
        else []
    )
    if len(moved_inputs) < len(moved_kwargs):
        moved_inputs.extend([() for _ in range(len(moved_kwargs) - len(inputs))])
    elif len(moved_kwargs) < len(moved_inputs):
        moved_kwargs.extend([{} for _ in range(len(moved_inputs) - len(moved_kwargs))])
    return tuple(moved_inputs), tuple(moved_kwargs)


def _verify_param_shape_across_processes(
    process_group: dist.ProcessGroup,
    tensors: list[torch.Tensor],
    logger: Optional["dist.Logger"] = None,
):
    return dist._verify_params_across_processes(process_group, tensors, logger)


def _sync_module_states(
    module: nn.Module,
    process_group: dist.ProcessGroup,
    broadcast_bucket_size: int,
    src: int,
    params_and_buffers_to_ignore: Container[str],
    broadcast_buffers: bool = True,
) -> None:
    """
    Sync ``module``'s parameters and buffers state.

    Syncs ``module``'s parameters and buffers state so that all ranks contain
    the same module state across all ranks. Note that this API assumes that all
    parameter shapes are consistent before running the synchronization. This can
    be checked with ``_verify_param_shape_across_processes``.
    """
    module_states: list[torch.Tensor] = []
    for name, param in module.named_parameters():
        if name not in params_and_buffers_to_ignore:
            module_states.append(param.detach())

    if broadcast_buffers:
        for name, buffer in module.named_buffers():
            if name not in params_and_buffers_to_ignore:
                module_states.append(buffer.detach())

    _sync_params_and_buffers(process_group, module_states, broadcast_bucket_size, src)


def _sync_params_and_buffers(
    process_group: dist.ProcessGroup,
    module_states: list[torch.Tensor],
    broadcast_bucket_size: int,
    src: int,
) -> None:
    """Synchronize ``module_states`` (list of tensors) across all processes by broadcasting them from rank 0."""
    if len(module_states) > 0:
        dist._broadcast_coalesced(
            process_group, module_states, broadcast_bucket_size, src
        )


def _replace_by_prefix(
    state_dict: dict[str, Any],
    old_prefix: str,
    new_prefix: str,
) -> None:
    """
    Replace all keys that match a given old_prefix with a new_prefix (in-place).

    Usage::

        state_dict = {"layer.xyz": torch.tensor(1)}
        replace_by_prefix_(state_dict, "layer.", "module.layer.")
        assert state_dict == {"module.layer.xyz": torch.tensor(1)}
    """
    if old_prefix == new_prefix:
        raise ValueError("old_prefix and new_prefix must be distinct")
    for key in list(state_dict.keys()):
        if not key.startswith(old_prefix):
            continue
        new_key = new_prefix + key[len(old_prefix) :]
        state_dict[new_key] = state_dict[key]
        del state_dict[key]


def _data_ptr_allocated(tensor: torch.Tensor) -> bool:
    return tensor.untyped_storage().data_ptr() > 0


def _get_root_modules(modules: list[nn.Module]) -> list[nn.Module]:
    """
    Returns the modules in ``modules`` that are root modules (i.e.
    parent-less) with respect to the set ``modules``. In other words, these
    are the modules in ``modules`` that are the not child of any other
    module in ``modules``.
    """
    root_modules: list[nn.Module] = []
    module_to_modules: dict[nn.Module, set[nn.Module]] = {
        module: set(module.modules()) for module in modules
    }
    for candidate_module in modules:
        is_root_module = True
        for module, _modules in module_to_modules.items():
            is_child_module = (
                candidate_module is not module and candidate_module in _modules
            )
            if is_child_module:
                is_root_module = False
                break
        if is_root_module:
            root_modules.append(candidate_module)
    return root_modules
<|endoftext|>import random

import torch
from torch._C._distributed_c10d import FakeWork


used_ids: set[int] = set()


def generate_unique_id() -> int:
    while True:
        new_id = random.randint(1, 10**9)
        if new_id not in used_ids:
            used_ids.add(new_id)
            return new_id


# Function to create and return FakeWork object
def create_fakework(args, return_first_arg=True):  # type: ignore[no-untyped-def]
    work = FakeWork()
    work.seq_id = generate_unique_id()
    fakework_script_obj = work.boxed()
    return (args[0], fakework_script_obj) if return_first_arg else fakework_script_obj


# Dictionary mapping collective operations to their meta functions
# All 20 ops from torch.csrc.distributed.c10d.Ops.cpp are included
# _DEPRECATED_META_FUNCTIONS = {
#     "allreduce_coalesced_": lambda *args: create_fakework(args, return_first_arg=False),
#     "allgather_coalesced_": lambda *args: create_fakework(args, return_first_arg=False),
#     "allgather_into_tensor_coalesced_": lambda *args: create_fakework(args, return_first_arg=False),
#     "reduce_scatter_tensor_coalesced_": lambda *args: create_fakework(args, return_first_arg=False),
# }
_META_FUNCTIONS = {
    "broadcast_": lambda *args: create_fakework(args),
    "allreduce_": lambda *args: create_fakework(args),
    "allgather_": lambda *args: create_fakework(args),
    "_allgather_base_": lambda *args: create_fakework(args),
    "reduce_scatter_": lambda *args: create_fakework(args),
    "_reduce_scatter_base_": lambda *args: create_fakework(args),
    "reduce_": lambda *args: create_fakework(args, return_first_arg=False),
    "gather_": lambda *args: create_fakework(args, return_first_arg=False),
    "scatter_": lambda *args: create_fakework(args),
    "alltoall_": lambda *args: create_fakework(args),
    "alltoall_base_": lambda *args: create_fakework(args, return_first_arg=False),
    "barrier": lambda *args: create_fakework(args, return_first_arg=False),
    "monitored_barrier_": lambda *args: None,
    "send": lambda *args: create_fakework(args, return_first_arg=False),
    "recv_": lambda *args: create_fakework(args, return_first_arg=False),
    "recv_any_source_": lambda *args: create_fakework(args, return_first_arg=False),
}

lib_impl = torch.library.Library("c10d", "IMPL")
for op, meta_func in _META_FUNCTIONS.items():
    lib_impl.impl(op, meta_func, "Meta")
<|endoftext|>from datetime import timedelta

from torch._C._distributed_c10d import _DEFAULT_PG_TIMEOUT


__all__ = ["default_pg_timeout", "default_pg_nccl_timeout"]

# Default process group wide timeout, if applicable.
# This only applies to the non-nccl backends
# To make an attempt at backwards compatibility with THD, we use an
# extraordinarily high default timeout, given that THD did not have timeouts.
default_pg_timeout: timedelta = _DEFAULT_PG_TIMEOUT
# Separate timeout for PGNCCL mainly because it's always been that way in the C++ layer, but until recently
# there was one default that applied across all backends in the python layer.
# Later, we could consider merging them back together at the c++ layer if we can align on a same value.
# (only if TORCH_NCCL_BLOCKING_WAIT or TORCH_NCCL_ASYNC_ERROR_HANDLING is set to 1).

try:
    from torch._C._distributed_c10d import _DEFAULT_PG_NCCL_TIMEOUT

    default_pg_nccl_timeout: timedelta | None = _DEFAULT_PG_NCCL_TIMEOUT
except ImportError:
    # if C++ NCCL support is not compiled, we don't have access to the default nccl value.
    # if anyone is actually trying to use nccl in this state, it should error.
    default_pg_nccl_timeout = None
<|endoftext|># Copyright (c) Meta Platforms, Inc. and affiliates

"""
Global configuration flags for torch.distributed
"""

import sys
from typing import TYPE_CHECKING

from torch.utils._config_module import Config, install_config_module


__all__ = ["compile_on_one_rank", "use_torchcomms", "pipeline_per_direction_p2p"]

# Deprecated alias. The canonical flag now lives in torch.compiler.config -- it is read
# across the compiler stack (make_fx, inductor) not just by distributed. Kept here for
# back-compat (reads, writes, and .patch forward to the canonical flag).
compile_on_one_rank: bool = Config(
    alias="torch.compiler.config.compile_on_one_rank",
    deprecated=True,
    deprecation_message="use torch.compiler.config.compile_on_one_rank instead",
)

# When enabled, uses TorchComms for communication backend instead of the
# traditional ProcessGroup backends (NCCL, Gloo, etc.).
use_torchcomms: bool = Config(
    default=False,
    env_name_default="TORCH_DISTRIBUTED_USE_TORCHCOMMS",
)

# When enabled, pipeline stages carry downstream (r -> r+1, forward activations)
# and upstream (r -> r-1, backward gradients) P2P on two separate communicators
# instead of sharing one. A single PP communicator serializes all send/recv in one
# FIFO: coalescing makes a single mixed batch deadlock-free, but across batches
# (pipeline skew, looped / V schedules, skip connections) the shared FIFO can
# still form a dependency cycle and deadlock. Splitting by direction removes that
# hazard and restores full-duplex bandwidth. Requires a device-bound default
# process group.
#
# This flag force-enables the behavior; it is auto-enabled when TorchComms is in
# use regardless of this flag (see PipelineStage), so it mainly matters for the
# non-TorchComms backends.
pipeline_per_direction_p2p: bool = Config(
    default=False,
    env_name_default="TORCH_DISTRIBUTED_PIPELINE_PER_DIRECTION_P2P",
)


if TYPE_CHECKING:
    from torch.utils._config_typing import *  # noqa: F403


# adds patch, save_config, invalid config checks, etc
install_config_module(sys.modules[__name__])
<|endoftext|># mypy: allow-untyped-defs
import contextlib
import logging
import sys
import traceback
import typing
from datetime import timedelta

import torch


RankType = int | torch.SymInt


log = logging.getLogger(__name__)


def is_available() -> bool:
    """
    Return ``True`` if the distributed package is available.

    Otherwise,
    ``torch.distributed`` does not expose any other APIs. Currently,
    ``torch.distributed`` is available on Linux, MacOS and Windows. Set
    ``USE_DISTRIBUTED=1`` to enable it when building PyTorch from source.
    Currently, the default value is ``USE_DISTRIBUTED=1`` for Linux and Windows,
    ``USE_DISTRIBUTED=0`` for MacOS.
    """
    return hasattr(torch._C, "_c10d_init")


_spmd_types_available: bool | None = None


def _is_spmd_types_available() -> bool:
    """Check if the spmd_types package is installed."""
    import importlib.util

    global _spmd_types_available
    if _spmd_types_available is None:
        _spmd_types_available = importlib.util.find_spec("spmd_types") is not None
    return _spmd_types_available


def _spmd_no_typecheck():
    """
    Return a spmd_types no_typecheck context, or a no-op if not installed.
    """
    if _is_spmd_types_available():
        import spmd_types

        return spmd_types.no_typecheck()

    @contextlib.contextmanager
    def no_typecheck():
        yield

    return no_typecheck()


if is_available() and not torch._C._c10d_init():
    raise RuntimeError("Failed to initialize torch.distributed")

# Custom Runtime Errors thrown from the distributed package
DistError = torch._C._DistError
DistBackendError = torch._C._DistBackendError
DistNetworkError = torch._C._DistNetworkError
DistStoreError = torch._C._DistStoreError
QueueEmptyError = torch._C._DistQueueEmptyError

if is_available():
    from torch._C._distributed_c10d import (
        _broadcast_coalesced,
        _compute_bucket_assignment_by_size,
        _DEFAULT_FIRST_BUCKET_BYTES,
        _make_nccl_premul_sum,
        _register_builtin_comm_hook,
        _register_comm_hook,
        _test_python_store,
        _verify_params_across_processes,
        Backend as _Backend,
        BuiltinCommHookType,
        DebugLevel,
        FileStore,
        get_debug_level,
        GradBucket,
        Logger,
        PrefixStore,
        ProcessGroup as ProcessGroup,
        Reducer,
        set_debug_level,
        set_debug_level_from_env,
        Store,
        TCPStore,
        Work as _Work,
    )

    def _make_distributed_pdb():
        """
        Supports using PDB from inside a multiprocessing child process.

        Usage:
        _make_distributed_pdb().set_trace()
        """

        # Lazy import pdb only if we set breakpoints.
        import pdb

        class _DistributedPdb(pdb.Pdb):
            def interaction(self, *args, **kwargs):
                _stdin = sys.stdin
                try:
                    with open("/dev/stdin") as sys.stdin:
                        pdb.Pdb.interaction(self, *args, **kwargs)
                finally:
                    sys.stdin = _stdin

        return _DistributedPdb()

    _breakpoint_cache: dict[int, typing.Any] = {}

    def breakpoint(rank: int = 0, skip: int = 0, timeout_s=3600):
        """
        Set a breakpoint, but only on a single rank.  All other ranks will wait for you to be
        done with the breakpoint before continuing.

        Args:
            rank (int): Which rank to break on.  Default: ``0``
            skip (int): Skip the first ``skip`` calls to this breakpoint. Default: ``0``.
        """
        if skip > 0:
            key = hash(str(traceback.format_exc()))
            counter = _breakpoint_cache.get(key, 0) + 1
            _breakpoint_cache[key] = counter
            if counter <= skip:
                log.warning("Skip the breakpoint, counter=%d", counter)
                return

        # avoid having the default timeout (if short) interrupt your debug session
        if timeout_s is not None:
            for group in torch.distributed.distributed_c10d._pg_map:
                torch.distributed.distributed_c10d.set_timeout(
                    timedelta(seconds=timeout_s), group
                )

        if get_rank() == rank:
            pdb = _make_distributed_pdb()
            pdb.message(
                "\n!!! ATTENTION !!!\n\n"
                f"Type 'up' to get to the frame that called dist.breakpoint(rank={rank})\n"
            )
            pdb.set_trace()
        # If Meta/Python keys are in the TLS, we want to make sure that we ignore them
        # and hit the (default) CPU/CUDA implementation of barrier.
        meta_in_tls = torch._C._meta_in_tls_dispatch_include()
        guard = torch._C._DisableTorchDispatch()  # type: ignore[attr-defined]
        torch._C._set_meta_in_tls_dispatch_include(False)
        try:
            barrier()
        finally:
            torch._C._set_meta_in_tls_dispatch_include(meta_in_tls)
            del guard

    if sys.platform != "win32":
        from torch._C._distributed_c10d import HashStore

    from .device_mesh import DeviceMesh, init_device_mesh

    # Variables prefixed with underscore are not auto imported
    # See the comment in `distributed_c10d.py` above `_backend` on why we expose
    # this.
    from .distributed_c10d import *  # noqa: F403
    from .distributed_c10d import (
        _all_gather_base,
        _coalescing_manager,
        _CoalescingManager,
        _create_process_group_wrapper,
        _get_process_group_name,
        _get_reconfigure_handle,
        _new_window,
        _rank_not_in_group,
        _reconfigure,
        _reduce_scatter_base,
        _supports_reconfigure,
        _supports_window,
        _time_estimator,
        get_node_local_rank,
    )
    from .remote_device import _remote_device
    from .rendezvous import (
        _create_store_from_options,
        register_rendezvous_handler,
        rendezvous,
    )

    set_debug_level_from_env()

else:
    # This stub is sufficient to get
    #   python test/test_public_bindings.py -k test_correct_module_names
    # working even when USE_DISTRIBUTED=0.  Feel free to add more
    # stubs as necessary.
    # We cannot define stubs directly because they confuse pyre

    class _Stub:
        pass

    sys.modules["torch.distributed"].GroupName = _Stub  # type: ignore[attr-defined]
    sys.modules["torch.distributed"].ProcessGroup = _Stub  # type: ignore[attr-defined]
<|endoftext|># mypy: allow-untyped-defs

import torch


class _remote_device:
    """
    Represents a device on a remote worker.

    Args:
        remote_device (str or torch.device): Represents a device on a remote worker.
            The string format should be one of the following:

                1. "<workername>/<device>", where the device field can be parsed as torch.device type.
                   E.g., "trainer0/cpu", "trainer0", "ps0/cuda:0".
                   In addition, the device field can be optional and the default value is "cpu".
                2. "rank:<rank>/<device>", where <rank> is the rank of the
                   process and device can be parsed as torch.device type.
                   E.g., "rank:0/cpu", "rank:0", "rank:0/cuda:0"
                3. <workername> and <rank> are optional and formats like "cpu"
                    and "cuda:1", just represent local devices.
    """

    def __init__(self, remote_device: str | torch.device):
        PARSE_ERROR = (
            f"Could not parse remote_device: {remote_device}. The valid format is "
            "'<workername>/<device>' or 'rank:<rank>/<device>' or '<device>'"
        )
        self._worker_name = None
        self._rank = None
        self._device: str | int | torch.device | None = None

        if isinstance(remote_device, torch.device):
            self._device = remote_device
        elif isinstance(remote_device, str):
            fields = remote_device.split("/")
            if len(fields) == 2:
                self._worker_name, self._device = fields
            elif len(fields) == 1:
                # Check if this is a valid device.
                if _remote_device._is_valid_local_device(fields[0]):
                    self._device = fields[0]
                else:
                    self._worker_name = fields[0]
                    self._device = "cpu"
            else:
                raise ValueError(PARSE_ERROR)
        else:
            raise TypeError(f"Invalid type for remote_device: {type(remote_device)}")

        # Do some basic sanity check (no empty string)
        if self._worker_name is not None and not self._worker_name:
            raise ValueError(PARSE_ERROR)

        # Validate the device.
        self._device = torch.device(self._device)

        # Check for rank based format.
        if self._worker_name is not None:
            fields = self._worker_name.split(":")
            if len(fields) == 2:
                # rank:<rank>/device format, extract rank
                if fields[0] == "rank" and fields[1].isdigit():
                    self._rank = int(fields[1])  # type: ignore[assignment]
                    self._worker_name = None
                else:
                    raise ValueError(PARSE_ERROR)
            elif len(fields) > 2:
                raise ValueError(PARSE_ERROR)

    @staticmethod
    def _is_valid_local_device(device):
        # Check for torch.device
        try:
            torch.device(device)
            return True
        except Exception:
            return False

    def worker_name(self) -> str | None:
        """Return the name of remote worker representing the remote device and ``None`` if no worker name is available."""
        return self._worker_name

    def rank(self) -> int | None:
        """
        Returns the rank of remote worker representing the remote device.
        Returns ``None`` if no rank is available.
        """
        return self._rank

    def device(self) -> torch.device:
        """Return the local device on the remote worker."""
        return self._device  # type: ignore[return-value]

    def __repr__(self):
        if self._device is not None:
            if self._worker_name is not None:
                return f"{self._worker_name}/{self._device}"
            elif self._rank is not None:
                return f"rank:{self._rank}/{self._device}"
            else:
                return str(self._device)
        else:
            if self._worker_name is not None:
                return f"{self._worker_name}"
            elif self._rank is not None:
                return f"{self._rank}"
            else:
                raise RuntimeError("Invalid state!")

    def __eq__(self, other):
        return isinstance(other, _remote_device) and (
            self._worker_name == other._worker_name
            and self._device == other._device
            and self._rank == other._rank
        )

    def __hash__(self):
        return hash(self._worker_name) ^ hash(self._device) ^ hash(self._rank)
<|endoftext|>import pickle
from dataclasses import dataclass
from io import BufferedIOBase
from typing import Any

import torch
import torch._weights_only_unpickler as _weights_only_unpickler
from torch.serialization import _load, _save, DEFAULT_PROTOCOL, MAP_LOCATION


__all__: list[str] = []


@dataclass
class _Entry:
    key: str
    is_storage: bool
    length: int


_weights_only_unpickler._add_safe_globals([_Entry])


class _PseudoZipFile:
    def __init__(self) -> None:
        self.records: dict[str, tuple[object, int]] = {}

    def write_record(self, key: str, data: object, length: int) -> None:
        self.records[key] = (data, length)

    def write_to(self, f: BufferedIOBase) -> None:
        entries = []
        for key, (data, length) in self.records.items():
            entries.append(
                _Entry(
                    key=key,
                    is_storage=isinstance(data, torch.UntypedStorage),
                    length=length,
                )
            )

        pickle.dump(entries, f, protocol=DEFAULT_PROTOCOL)

        for data, _ in self.records.values():
            if isinstance(data, bytes):
                f.write(data)
            elif isinstance(data, str):
                f.write(data.encode("utf-8"))
            elif isinstance(data, torch.UntypedStorage):
                data._write_file(f, False, False, 1)
            else:
                raise TypeError(f"unknown type: {type(data)}")

    def read_from(self, f: BufferedIOBase) -> None:
        entries = _weights_only_unpickler.load(f)

        for entry in entries:
            data = f.read(entry.length)
            if entry.is_storage:
                if entry.length == 0:
                    storage = torch.UntypedStorage(0)
                else:
                    storage = torch.frombuffer(
                        data,
                        dtype=torch.uint8,
                    ).untyped_storage()

                self.records[entry.key] = (
                    storage,
                    entry.length,
                )
            else:
                self.records[entry.key] = (data, entry.length)

    def has_record(self, key: str) -> bool:
        return key in self.records

    def get_record(self, key: str) -> object:
        return self.records[key][0]

    def get_storage_from_record(
        self, key: str, _length: int, _type: int
    ) -> torch.Tensor:
        return torch.tensor(self.records[key][0], dtype=torch.uint8)

    def serialization_id(self) -> str:
        return "torchft"


def _streaming_save(
    obj: object,
    f: BufferedIOBase,
    pickle_module: Any = pickle,
    pickle_protocol: int = DEFAULT_PROTOCOL,
) -> None:
    """
    Save the object to a file-like object in a streaming fashion compatible with
    network sockets.

    This behaves similarly to :func:`torch.save` with a few notable differences:

    * A non-seekable file like object can be used when loading.
    * No forwards/backwards compatibility is provided for the serialization
      format. This is only intended to be used with a single version of PyTorch
      with transient storage (i.e. sockets or temp files).
    * mmap is not supported

    See :func:`torch.save` for more details on specific arguments.
    """

    zip_file = _PseudoZipFile()
    _save(
        obj,
        zip_file=zip_file,
        pickle_module=pickle_module,
        pickle_protocol=pickle_protocol,
        _disable_byteorder_record=False,
    )
    zip_file.write_to(f)


def _streaming_load(
    f: BufferedIOBase,
    map_location: MAP_LOCATION = None,
    pickle_module: Any = None,
    *,
    weights_only: bool = True,
    **pickle_load_args: Any,
) -> object:
    """
    Load the object from a file-like object in a streaming fashion compatible with
    network sockets.

    See :func:`_streaming_save` for more details about the streaming behavior.

    See :func:`torch.load` for more details on specific arguments.
    """
    if weights_only:
        if pickle_module is not None:
            raise RuntimeError(
                "Can not safely load weights when explicit pickle_module is specified"
            )
        pickle_module = _weights_only_unpickler
    else:
        if pickle_module is None:
            pickle_module = pickle

    if "encoding" not in pickle_load_args:
        pickle_load_args["encoding"] = "utf-8"

    zip_file = _PseudoZipFile()
    zip_file.read_from(f)
    return _load(
        zip_file=zip_file,
        map_location=map_location,
        pickle_module=pickle_module,
        **pickle_load_args,
    )
<|endoftext|># Copyright (c) Meta Platforms, Inc. and affiliates
from typing_extensions import Protocol, runtime_checkable

import torch


@runtime_checkable
class _Checkpointable(Protocol):  # noqa: PYI046
    """
    Interface for checkpointable objects.
    Implemented as a protocol, implicit subtyping is supported so subclasses do not need to inherit this explicitly.
    This is to allow arbitrary objects/tensor subclasses to hook into DCP seamlessly through implementing the interface.
    """

    def __create_write_items__(self, fqn: str, object: object) -> list[object]:
        """
        Return a list of WriteItems based on object's contents.
        """
        raise NotImplementedError(
            "_Checkpointable._create_write_items is not implemented"
        )

    def __create_chunk_list__(self) -> list[object]:
        """
        Return a list of `ChunkStorageMetadata` based on object's contents.
        """
        raise NotImplementedError(
            "_Checkpointable._create_chunk_list is not implemented"
        )

    def __get_tensor_shard__(self, index: int) -> torch.Tensor:
        """
        Return a 'torch.Tensor' shard based on 'MetadataIndex'.
        """
        raise NotImplementedError(
            "_Checkpointable._get_tensor_shard is not implemented"
        )
<|endoftext|>#!/usr/bin/env python3
# mypy: allow-untyped-defs

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import logging
from collections.abc import Callable
from typing import Any, TypeVar
from typing_extensions import ParamSpec

import torch
import torch.distributed as dist
from torch.distributed.logging_handlers import _log_handlers
from torch.monitor import _WaitCounter


__all__: list[str] = []

_DEFAULT_DESTINATION = "default"


def _get_or_create_logger(destination: str = _DEFAULT_DESTINATION) -> logging.Logger:
    logging_handler, log_handler_name = _get_logging_handler(destination)
    logger = logging.getLogger(f"c10d-{log_handler_name}")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s %(filename)s:%(lineno)s %(levelname)s p:%(processName)s t:%(threadName)s: %(message)s"
    )
    logging_handler.setFormatter(formatter)
    logger.propagate = False
    logger.addHandler(logging_handler)
    return logger


def _get_logging_handler(
    destination: str = _DEFAULT_DESTINATION,
) -> tuple[logging.Handler, str]:
    log_handler = _log_handlers[destination]
    log_handler_name = f"{type(log_handler).__name__}-{destination}"
    return (log_handler, log_handler_name)


# pyrefly: ignore [unknown-name]
global _c10d_logger
_c10d_logger = _get_or_create_logger()


def _get_msg_dict(func_name, *args, **kwargs) -> dict[str, Any]:
    if dist.is_initialized():
        group = kwargs.get("group") or kwargs.get("process_group")
        msg_dict = {
            "func_name": f"{func_name}",
            "pg_name": f"{dist._get_process_group_name(kwargs.get('pg'))}",  # type: ignore[arg-type]
            "backend": f"{dist.get_backend(group)}",
            "world_size": f"{dist.get_world_size()}",
            "group_size": f"{dist.get_world_size(group)}",
            "global_rank": f"{dist.get_rank()}",
            "local_rank": f"{dist.get_rank(group)}",
        }
        if msg_dict["backend"] == "nccl":
            nccl_version = torch.cuda.nccl.version()
            msg_dict["nccl_version"] = ".".join(str(v) for v in nccl_version)
    else:
        msg_dict = {
            "func_name": f"{func_name}",
        }
    return msg_dict


_T = TypeVar("_T")
_P = ParamSpec("_P")


def _exception_logger(func: Callable[_P, _T]) -> Callable[_P, _T]:
    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        try:
            return func(*args, **kwargs)
        except Exception as error:
            msg_dict = _get_msg_dict(func.__name__, *args, **kwargs)
            msg_dict["error"] = f"{error}"
            _c10d_logger.debug(msg_dict)
            raise

    return wrapper


def _time_logger(func: Callable[_P, _T]) -> Callable[_P, _T]:
    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        with _WaitCounter(f"pytorch.wait_counter.c10d.{func.__name__}").guard():
            func_return = func(*args, **kwargs)
        return func_return

    return wrapper
<|endoftext|># mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
import os
import threading
import warnings
from collections.abc import Callable, Iterator
from itertools import zip_longest
from typing import Any, TYPE_CHECKING

import torch
from torch._custom_class_base import CustomClassBase
from torch.distributed import is_available
from torch.distributed._mesh_layout import _FlatLayout, _MeshLayout
from torch.types import IntLikeType
from torch.utils._typing_utils import not_none


__all__ = ["init_device_mesh", "DeviceMesh"]


if not is_available():
    import sys

    # We need to create the stubs when distributed is not available.
    # Otherwise, we would fail the doc tests (```./.ci/pytorch/docs-test.sh```),
    # since it would try to import ``torch.distributed.device_mesh`` or
    # ``torch.distributed.init_device_mesh`` but cannot find them.

    class _DeviceMeshStub:
        pass

    def _init_device_mesh_stub():
        pass

    sys.modules["torch.distributed.device_mesh"].DeviceMesh = _DeviceMeshStub  # type: ignore[attr-defined]
    # pyrefly: ignore [missing-attribute]
    sys.modules[
        "torch.distributed.device_mesh"
    ].init_device_mesh = _init_device_mesh_stub  # type: ignore[attr-defined]


else:
    from torch._C._distributed_c10d import Backend as C10dBackend
    from torch.distributed import config as dist_config
    from torch.distributed.distributed_c10d import (
        _get_default_group,
        _register_process_group_opaque_type,
        _resolve_process_group,
        get_backend,
        get_process_group_ranks,
        get_rank,
        get_world_size,
        GroupMember,
        GroupName,
        init_process_group,
        is_initialized,
        new_group,
        ProcessGroup,
        split_group,
    )

    logger = logging.getLogger(__name__)

    # only import numpy typing when type checking
    if TYPE_CHECKING:
        try:
            from numpy.typing import ArrayLike
        except ImportError:
            logger.warning(
                "DeviceMesh requires numpy >= 1.21 to be installed for type checking"
            )

    BackendConfig = tuple[str | None, C10dBackend.Options | None]
    torch.serialization.add_safe_globals([_FlatLayout, _MeshLayout])

    def _get_pg_from_name(mesh: "DeviceMesh", name: str) -> ProcessGroup:
        """
        This method allows us to torch.compile through DeviceMesh and lift its
        PGs as inputs to the graph since all PGs will have a source from the
        DeviceMesh through the `_pg_registry`.
        This will be moved to the DeviceMesh backend object once we separate
        DeviceMesh into the frontend and backend.
        """
        if torch.compiler.is_compiling():
            pg = mesh._pg_registry.get(name, None)
            if pg is None:
                raise RuntimeError(
                    f"PG {name} was not found while torch.compile tracing "
                    "This is probably because we pickle/unpickled a device mesh "
                    "before the PGs were created."
                )
            return pg
        else:
            return _resolve_process_group(name)  # pyrefly: ignore[bad-argument-type]

    def _maybe_traced_group(mesh: "DeviceMesh", mesh_dim: int) -> ProcessGroup | None:
        """Resolve the group via an in-graph op when tracing under CooR.

        Under compile_on_one_rank and an active make_fx/proxy trace, return the
        group as the output of _dtensor.mesh_get_process_group so the
        ProcessGroup is extracted from the (graph-input) mesh at runtime instead
        of being baked into the graph as an unserializable torchbind constant.
        Returns None to take the eager path otherwise. Mirrors
        _functional_collectives._resolve_group, but also covers eager collectives
        (dist.all_reduce -> c10d.allreduce_) that bind the ProcessGroup directly.

        This is a module-level helper rather than a DeviceMesh method so that
        Dynamo does not reject it as an unregistered opaque-object member.
        """
        if not torch.compiler.config.compile_on_one_rank:
            return None
        # Under torch.compile, Dynamo lifts the ProcessGroup from the mesh's
        # _pg_registry itself (see _get_pg_from_name); the op-emitting path here
        # is only for make_fx tracing. is_compiling() folds to a constant under
        # Dynamo, so this returns cleanly without tracing the op below.
        if torch.compiler.is_compiling():
            return None
        from torch.fx.experimental.proxy_tensor import get_proxy_mode

        # get_proxy_mode() is None in eager and inside the op's own fake/eager
        # impl (the proxy mode is popped while dispatching), so this does not
        # recurse.
        if get_proxy_mode() is None:
            return None
        # Ensure the op is registered without a module-level circular import.
        from torch.distributed.tensor import _collective_utils  # noqa: F401

        return torch.ops._dtensor.mesh_get_process_group(mesh, mesh_dim)

    class _MeshEnv(threading.local):
        def __init__(self) -> None:
            self.mesh_stack: list[DeviceMesh] = []

        def get_current_mesh(self) -> "DeviceMesh":
            if len(self.mesh_stack) == 0:
                raise RuntimeError("No device mesh is currently active!")
            return self.mesh_stack[-1]

        # TODO: to remove it once we move all use cases into new API.
        def get_root_mesh(self, device_mesh: "DeviceMesh") -> "DeviceMesh":
            # If a mesh could not be found in the child_to_root_mapping, it is a root mesh itself.
            # A root mesh is not created through slicing.
            # We considers the root mesh of a root mesh is itself.
            # We keep this function for backward compatibility.
            warnings.warn(
                "This get_root_mesh API will be deprecated soon."
                "Please use `get_root_mesh` inside DeviceMesh instead.",
                stacklevel=2,
            )
            if not device_mesh:
                return device_mesh
            return device_mesh._get_root_mesh()

        @staticmethod
        def num_devices_per_host(device_type: str) -> int:
            return _get_device_handle(device_type).device_count()

        @staticmethod
        def num_hosts(device_type: str) -> int:
            # ProcessGroup can't tell us this info so we have to infer it, assume
            # homogeneous hardware for now
            return get_world_size() // _MeshEnv.num_devices_per_host(device_type)

        # TODO: to remove it once we move all use cases into new API.
        # We keep this API for backward compatibility.
        def _get_all_submeshes(
            self, device_mesh: "DeviceMesh", mesh_dim_name: str
        ) -> list["DeviceMesh"]:
            warnings.warn(
                "This _get_all_submeshes API will be deprecated soon."
                "Please use `_get_all_submeshes` inside DeviceMesh instead.",
                stacklevel=2,
            )
            return device_mesh._get_all_submeshes(mesh_dim_name)

    _mesh_resources: _MeshEnv = _MeshEnv()

    def _get_device_handle(device_type: str = "cuda"):
        """
        Get the module corresponding to the device_type which is cuda or cuda-like device.
        For example, when the device_type is cuda, the module `torch.cuda` is returned.
        Return None when there is no corresponding module for device_type, otherwise
        return the corresponding module.
        """
        return getattr(torch, device_type, None)

    class DeviceMesh(CustomClassBase):
        """
        DeviceMesh represents a mesh of devices, where layout of devices could be
        represented as a n-d dimension array, and each value of the n-d dimensional
        array is the global id of the default process group ranks.

        DeviceMesh could be used to setup the N dimensional device connections across the cluster,
        and manage the ProcessGroups for N dimensional parallelisms. Communications could happen on
        each dimension of the DeviceMesh separately. DeviceMesh respects the device that user selects
        already (i.e. if user call `torch.cuda.set_device` before the DeviceMesh initialization),
        and will select/set the device for the current process if user does not set the device
        beforehand. Note that manual device selection should happen BEFORE the DeviceMesh initialization.

        DeviceMesh can also be used as a context manager when using together with DTensor APIs.

        .. note::
            DeviceMesh follows SPMD programming model, which means the same PyTorch Python program
            is running on all processes/ranks in the cluster. Therefore, users need to make sure the
            `mesh` array (which describes the layout of devices) should be identical across all ranks.
            Inconsistent `mesh` will lead to silent hang.

        Args:
            device_type (str): The device type of the mesh. Currently supports: "cpu", "cuda/cuda-like".
            mesh (ndarray): A multi-dimensional array or an integer tensor describing the layout
                of devices, where the IDs are global IDs of the default process group.
            _rank (int): (experimental/internal)
                The global rank of the current process. If not provided, it will
                be inferred from the default process group.

        Returns:
            DeviceMesh: A :class:`DeviceMesh` object representing the device layout.

        The following program runs on each process/rank in an SPMD manner. In this example, we have 2
        hosts with 4 GPUs each.
        A reduction over the first dimension of mesh will reduce across
        columns (0, 4), .. and (3, 7), a reduction over the second dimension
        of mesh reduces across rows (0, 1, 2, 3) and (4, 5, 6, 7).

        Example::

            >>> # xdoctest: +SKIP("no rank")
            >>> from torch.distributed.device_mesh import DeviceMesh
            >>>
            >>> # Initialize device mesh as (2, 4) to represent the topology
            >>> # of cross-host(dim 0), and within-host (dim 1).
            >>> mesh = DeviceMesh(device_type="cuda", mesh=[[0, 1, 2, 3],[4, 5, 6, 7]])
        """

        _rank: int
        _device_type: str
        _rank_map: torch.Tensor
        _mesh_dim_names: tuple[str, ...] | None
        _layout: _MeshLayout
        _root_mesh: "DeviceMesh | None" = None
        _thread_id: int | None
        # Record flatten mesh name to its flattened mesh in root mesh.
        _flatten_mapping: dict[str, "DeviceMesh"]
        # Registry mapping group names to ProcessGroup objects (to avoid C++ lookup)
        _pg_registry: dict[str, ProcessGroup]

        def __init__(
            self,
            device_type: str,
            mesh: "torch.Tensor | ArrayLike | None" = None,
            *,
            mesh_dim_names: tuple[str, ...] | None = None,
            backend_override: tuple[BackendConfig, ...] | None = None,
            _init_backend: bool = True,
            _rank: int | None = None,
            _layout: _MeshLayout | None = None,
            _rank_map: torch.Tensor | None = None,
            _root_mesh: "DeviceMesh | None" = None,
        ) -> None:
            # no-op in OSS, logs API usage metrics in meta-internal runs
            torch._C._log_api_usage_once(
                "torch.distributed.device_mesh.DeviceMesh.__init__"
            )
            if mesh is not None:
                if _layout is not None or _rank_map is not None:
                    raise TypeError(
                        "Cannot provide _layout and/or _rank_map if passing explicit mesh"
                    )
                if isinstance(mesh, torch.Tensor) and mesh.device.type != "cpu":
                    mesh = mesh.to("cpu")
                mesh_tensor = (
                    mesh.detach().to(dtype=torch.int).contiguous()
                    if isinstance(mesh, torch.Tensor)
                    else torch.tensor(mesh, device="cpu", dtype=torch.int)
                )
                _layout = _MeshLayout.from_sizes_strides(
                    tuple(mesh_tensor.size()), tuple(mesh_tensor.stride())
                )
                _rank_map = mesh_tensor.flatten()
            else:
                if _layout is None or _rank_map is None:
                    raise TypeError(
                        "The mesh argument is required except for PRIVATE USAGE ONLY!"
                    )

            if not _layout.collapse().check_orthogonal():
                raise AssertionError(
                    "Please use a non-overlapping layout when creating a DeviceMesh."
                )
            if _rank_map.ndim != 1:
                raise AssertionError("The rank map must be 1-dimensional")
            if not _rank_map.is_contiguous():
                raise AssertionError("The rank map must be contiguous")
            if _rank_map.numel() < _layout.cosize():
                raise AssertionError(
                    f"The rank map contains {_rank_map.numel()} element, "
                    f"which isn't large enough for layout {_layout}"
                )

            self._device_type = device_type
            self._layout = _layout
            self._rank_map = _rank_map
            self._mesh_dim_names = tuple(mesh_dim_names) if mesh_dim_names else None
            self._root_mesh = _root_mesh

            if backend_override is None:
                backend_override = ((None, None),) * len(self._layout)
            elif len(backend_override) != len(self._layout):
                raise ValueError(
                    f"backend_override should have the same length as the number of mesh dimensions, "
                    f"but got {len(backend_override)} and {len(self._layout)}."
                )
            # Internal bookkeeping for the device mesh.
            self._layout = (
                _layout
                if _layout
                else _MeshLayout.from_sizes_strides(
                    tuple(self.mesh.size()), tuple(self.mesh.stride())
                )
            )
            if not self._layout.collapse().check_orthogonal():
                raise AssertionError(
                    "Please use a non-overlapping layout when creating a DeviceMesh."
                )
            # Because we still need to support slicing of flattened dim from root mesh, so we don't check stride here.
            if self._layout.numel() != self.mesh.numel():
                raise AssertionError(
                    "Please use a valid layout when creating a DeviceMesh."
                    f"The layout {self._layout} is not consistent with the mesh size {self.mesh.size()}."
                )

            # private field to pre-generate DeviceMesh's hash
            self._flatten_rank_map = tuple(self._rank_map.tolist())
            self._thread_id = None
            # Initialize instance-specific flatten mapping
            self._flatten_mapping = {}
            # Initialize process group registry
            self._pg_registry = {}

            # Skip process group initialization if xla device or init backend is False
            # TODO(yeounoh) implement DeviceMesh backend and register XLA backend.
            if device_type != "xla":
                # always try to create default (world) pg, even if it is not initialized
                # already. The world pg is used for device mesh identity (rank) on each
                # process (we need to know if the current global rank is in the mesh or not).
                if _init_backend:
                    self._setup_world_group_and_device()
                    self._dim_group_names = self._init_process_groups(
                        self._layout,
                        self._rank_map,
                        self._mesh_dim_names,
                        backend_override,
                    )
                    # Populate the process group registry
                    # If we have a root mesh, add to root's registry for lookups
                    target_registry = (
                        self._root_mesh._pg_registry
                        if self._root_mesh is not None
                        else self._pg_registry
                    )
                    for name in self._dim_group_names:
                        pg = _resolve_process_group(name)
                        if pg is not None:
                            target_registry[name] = pg

                if is_initialized() and get_backend() == "threaded":
                    self._thread_id = threading.get_ident()

                # Now that the process group is initialized, we can get the rank
                if _rank is None:
                    self._rank = get_rank()
                else:
                    self._rank = _rank

                self._coordinate_on_dim = self._compute_coordinate_on_dim()

            self._hash: int | None = None

        @staticmethod
        def _compute_coordinates_from_mesh(
            mesh_tensor: torch.Tensor,
            rank: int,
        ) -> tuple[int, ...] | None:
            """
            Compute the coordinates of a rank within a mesh tensor.

            Args:
                mesh_tensor: The mesh tensor to search in
                rank: The rank to find coordinates for

            Returns:
                A tuple of coordinates if the rank is found in the mesh, None otherwise

            Raises:
                AssertionError: If the rank appears more than once in the mesh
            """
            rank_coords = (mesh_tensor == rank).nonzero()
            if rank_coords.size(0) not in (0, 1):
                raise AssertionError(
                    f"rank_coords.size(0) must be 0 or 1, got {rank_coords.size(0)}"
                )

            if rank_coords.size(0) == 0:
                return None

            coords = rank_coords[0].tolist()
            return tuple(coords)

        def _compute_coordinate_on_dim(self) -> tuple[int, ...] | None:
            # calculate the coordinates of the current global rank on the mesh
            return self._compute_coordinates_from_mesh(self.mesh, self._rank)

        def __getstate__(self) -> dict:
            # Exclude _pg_registry from pickle since ProcessGroup objects can't be pickled
            state = self.__dict__.copy()
            state.pop("_pg_registry", None)
            return state

        def __setstate__(self, state: dict) -> None:
            self.__dict__.update(state)
            # Reconstruct _pg_registry from _dim_group_names
            self._pg_registry = {}
            if hasattr(self, "_dim_group_names"):
                for name in self._dim_group_names:
                    try:
                        pg = _resolve_process_group(name)
                        if pg is not None:
                            self._pg_registry[name] = pg
                    except RuntimeError:
                        # Note: process groups may not exist if loading in a different process
                        logger.warning(
                            "It seems like pickling/unpickling of the DeviceMesh "
                            "occurred before the PGs were created. This will cause PG "
                            "lookup to fail when torch.compile is enabled"
                        )

        @property
        def device_type(self) -> str:
            """Returns the device type of the mesh."""
            return self._device_type

        @staticmethod
        def _get_mesh_tensor_from_full_mesh(
            full_mesh: torch.Tensor,
            current_rank: int | None = None,
        ) -> torch.Tensor:
            if full_mesh.size(0) == 1:
                return full_mesh[0]

            if current_rank is None:
                current_rank = get_rank()

            my_coords = (full_mesh == current_rank).nonzero()
            if my_coords.size(0) > 0:
                return full_mesh[my_coords[0, 0]]
            raise RuntimeError(
                "In order to get the mesh Tensor of a DeviceMesh it needs to "
                "either have all its original dimensions (e.g., no slicing) "
                "or it needs to contain the local rank"
            )

        @property
        def mesh(self) -> torch.Tensor:
            """Returns the tensor representing the layout of devices."""
            full_mesh = self._layout.remap_to_tensor(self._rank_map)
            return self._get_mesh_tensor_from_full_mesh(full_mesh)

        @property
        def mesh_dim_names(self) -> tuple[str, ...] | None:
            """Returns the names of mesh dimensions."""
            return self._mesh_dim_names

        def _setup_world_group_and_device(self):
            default_initialized = is_initialized()
            if not default_initialized:
                init_process_group()

            world_size = get_world_size()
            if self._layout.numel() > world_size:
                raise RuntimeError(
                    f"Mesh should not be bigger than default world size {world_size}, but found {self._layout.numel()} ranks!"
                )

            # Skip device setup for fake backend (cross-compilation mode).
            # The fake backend is used to simulate distributed training on a
            # single process without actual devices, enabling compilation of
            # GPU programs on CPU-only machines.
            backend = get_backend()
            if backend == "fake":
                return _get_default_group()

            # ONLY set the device if the current device is not initialized, if user already
            # set the device before DeviceMesh init, we respect the user's choice.
            device_handle = _get_device_handle(self._device_type)
            if device_handle and not device_handle.is_initialized():
                # auto set the cuda/cuda-like device only if user has not set it, if there's LOCAL_RANK
                # env variable from launchers, we use it to set the device.
                if "LOCAL_RANK" in os.environ:
                    local_rank = int(os.environ["LOCAL_RANK"])
                    logger.info(
                        "Setting default device for the current process based on LOCAL_RANK=%s",
                        local_rank,
                    )
                    device_handle.set_device(local_rank)
                else:
                    # heuristic to set the current cuda/cuda-like device base on num of gpu devices available in each host
                    # NOTE: This device selection would only work for homogeneous hardware.
                    num_devices_per_host = device_handle.device_count()
                    # Skip device setup if no devices are available (cross-compilation mode)
                    if num_devices_per_host == 0:
                        return _get_default_group()
                    warnings.warn(
                        "It seems like you did not set/select the default device for the current process before the DeviceMesh "
                        "initialization or use a launcher (i.e. torchrun) which populates `LOCAL_RANK` environment variable. "
                        "It is recommended to set the current device for the process BEFORE the DeviceMesh initialization so that "
                        "the underlying communicator (i.e. NCCL) can be initialized properly. "
                        "Given that the current process has no default device selected, DeviceMesh will use a heuristic to set the "
                        "device_id via `global_rank % num_devices_per_host`, assuming homogeneous hardware cluster. ",
                        stacklevel=2,
                    )
                    if (
                        world_size > num_devices_per_host
                        and world_size % num_devices_per_host != 0
                    ):
                        raise RuntimeError(
                            f"DeviceMesh only support homogeneous hardware, but found "
                            f"{world_size} ranks and {num_devices_per_host} {self._device_type} devices!"
                        )
                    device_handle.set_device(get_rank() % num_devices_per_host)

            return _get_default_group()

        @staticmethod
        def _init_one_process_group(
            sub_layout: _FlatLayout,
            rank_map: torch.Tensor,
            dim_name: str,
            backend_override: BackendConfig,
        ) -> GroupName | None:
            # Generate a 2D global mesh tensor for the current dim for PG creation.
            pg_ranks_by_dim = _MeshLayout([sub_layout]).remap_to_tensor(rank_map)
            backend, pg_options = backend_override
            # We need to explicitly pass in timeout when specified in option, otherwise
            # the default timeout will be used to override the timeout set in option.
            # TODO: remove this once we have fixed inside c10d level.
            timeout = pg_options._timeout if pg_options else None

            # If we have a 2D mesh with mesh_dim_names ("dp", "tp"), the group description
            # of the subgroups would be `mesh_dim_dp` and `mesh_name_tp`.
            # If the mesh doesn't have a mesh_dim_names, then the group description of the
            # subgroup would be `mesh_dim_0` and `mesh_dim_1`.
            group_desc = f"mesh_{dim_name}"

            dim_group = None
            default_group = _get_default_group()

            # Early return if there is only one sub_layout in the mesh layout.
            if sub_layout.numel() == get_world_size() and backend_override == (
                None,
                None,
            ):
                # Append the default pg to the first dim groups only if the default pg is compatible with `self._device_type`.
                # Otherwise, create new pg.
                ranks = list(range(get_world_size()))
                dim_group = (
                    new_group(
                        backend=backend,
                        ranks=ranks,
                        group_desc="mesh_default",
                    )
                    if torch.cuda.is_available()
                    and get_backend(default_group) == "gloo"
                    else default_group
                )
                return dim_group.group_name  # type: ignore[union-attr]

            # If bound_device_id exists, it means the nccl communicator has been eagerly initialized
            # so that we can use `split_group` to create subgroups through `ncclCommSplit`.
            # In this case, we only need to make one API call (`split_group``) for the subgroup creation
            # for each mesh dimension. In a 2 * 4 mesh, we only need to make two API calls per ranks to create
            # all the subgroups.
            # Otherwise, we need to make more than one API call (`new_group`) for subgroup creations. The
            # numbers of API calls are equal to the number of subgroups for each mesh dimension. In a 2 * 4
            # mesh, we need to make two API calls per ranks to create all the subgroups.
            if (
                (
                    getattr(default_group, "bound_device_id", None) is not None
                    or dist_config.use_torchcomms
                )
                and torch.accelerator.is_available()
                and (
                    backend is None
                    or default_group._get_backend(
                        torch.accelerator.current_accelerator()  # pyrefly: ignore[bad-argument-type]
                    ).name()
                    == backend
                )
            ):
                dim_group = split_group(
                    parent_pg=default_group,
                    timeout=timeout,
                    pg_options=pg_options,
                    split_ranks=pg_ranks_by_dim.tolist(),
                    group_desc=group_desc,
                )
                if dim_group == GroupMember.NON_GROUP_MEMBER:
                    return None
                return dim_group.group_name

            # If the subgroup has been already created through `split_group`, we simply loop over `pg_ranks_by_dim`
            # and append the `group_name` to the `dim_group_names` list when the current rank is in the subgroup.
            # Otherwise, we use `new_group` instead of `split_group` to create subgroups by looping over `pg_ranks_by_dim`
            # along with appending information to the `dim_group_names` list whenever necessary.
            # When torchcomms is enabled with a fake backend (e.g. disabled mesh dimensions), use hashed PG names so they
            # stay consistent with the hash-based names produced by split_group for real backends. Sequential integer names
            # from new_group are not resolvable from compiled code when mixed with split_group hash names.
            use_hashed = dist_config.use_torchcomms and backend == "fake"
            pg_name = None
            for dim_mesh in pg_ranks_by_dim:
                subgroup_ranks = dim_mesh.tolist()
                dim_group = new_group(
                    ranks=subgroup_ranks,
                    timeout=timeout,
                    backend=backend,
                    pg_options=pg_options,
                    group_desc=group_desc,
                    use_local_synchronization=use_hashed,
                )

                # only add to dim_groups if the current rank in the subgroup
                if get_rank() in subgroup_ranks:
                    if dim_group == GroupMember.NON_GROUP_MEMBER:
                        raise AssertionError(
                            f"Rank {get_rank()} was not included in process group {subgroup_ranks}"
                        )
                    if pg_name is not None:
                        raise RuntimeError(
                            f"Each device mesh dimension should get only one process group, but got {get_rank()} "
                            f"in {subgroup_ranks}!"
                        )
                    pg_name = dim_group.group_name
            return pg_name

        @staticmethod
        def _init_process_groups(
            layout: _MeshLayout,
            rank_map: torch.Tensor,
            mesh_dim_names: tuple[str, ...] | None,
            backend_override: tuple[BackendConfig, ...],
        ) -> list[GroupName]:
            # group_name associated with each mesh dimension, each
            # mesh dimension should have one sub-group per rank
            dim_group_names: list[GroupName | None] = []
            # create sub pgs base on the mesh argument specified
            for dim in range(len(layout)):
                dim_name = mesh_dim_names[dim] if mesh_dim_names else f"dim_{dim}"
                dim_group_names.append(
                    DeviceMesh._init_one_process_group(
                        layout[dim],
                        rank_map,
                        dim_name,
                        backend_override[dim],
                    )
                )
            # Filter out None values. If any are None then they should all be None.
            dim_non_none_group_names = [n for n in dim_group_names if n is not None]
            if dim_non_none_group_names and len(dim_non_none_group_names) != len(
                dim_group_names
            ):
                raise AssertionError
            return dim_non_none_group_names

        def _get_root_mesh(self) -> "DeviceMesh":
            return self._root_mesh if self._root_mesh else self

        def __enter__(self) -> "DeviceMesh":
            # set this mesh as the current mesh in mesh env
            _mesh_resources.mesh_stack.append(self)
            return self

        # pyre-fixme[2]: Parameter must be annotated.
        def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
            # pop this mesh from mesh env
            _mesh_resources.mesh_stack.pop()

        def __repr__(self) -> str:
            device_mesh_repr = (
                f"({', '.join(f'{k}={v}' for k, v in zip(self._mesh_dim_names, self._layout.top_level_sizes))})"
                if self._mesh_dim_names
                else f"{self._layout.top_level_sizes}"
            )
            stride = tuple(
                axis.stride[0] if len(axis.stride) == 1 else axis.stride
                for axis in self._layout
            )
            device_mesh_repr = (
                f"DeviceMesh({device_mesh_repr}, '{self.device_type}', stride={stride}"
            )
            # We only print the mesh tensor if the debug mode is turned on.
            if os.environ.get("TORCH_DISTRIBUTED_DEBUG", "") == "DETAIL":
                device_mesh_repr += f", Mesh: {self.mesh.tolist()}"
            return f"{device_mesh_repr})"

        def _hash_key(self) -> tuple[Any, ...]:
            """Return the tuple used for hashing. Used by both __hash__ and _stable_hash."""
            return (
                self._flatten_rank_map,
                self._layout,
                self._device_type,
                self._mesh_dim_names,
                self._thread_id,
            )

        def __hash__(self):
            # lazily compute hash
            self._hash = getattr(self, "_hash", None)
            if not self._hash:
                self._hash = hash(self._hash_key())
            return self._hash

        def __eq__(self, other: object) -> bool:
            if self is other:
                return True
            if not isinstance(other, DeviceMesh):
                return False
            return (
                self._flatten_rank_map == other._flatten_rank_map
                and self._layout == other._layout
                and self._device_type == other._device_type
                and self._mesh_dim_names == other._mesh_dim_names
                and self._thread_id == other._thread_id
            )

        def _stable_hash(self) -> str:
            """
            Return a stable hash for AOT autograd caching.
            [See note: Tensor subclass stable hashing for AOT autograd cache]
            """
            import hashlib

            return hashlib.blake2b(
                repr(self._hash_key()).encode(), digest_size=16
            ).hexdigest()

        def __getitem__(self, mesh_dim_names: str | tuple[str, ...]) -> "DeviceMesh":
            """
            Slice the current DeviceMesh based on the mesh_dim_names given to create a submesh.
            The submesh created consists of the dimensions and the communicators indicated by
            ``mesh_dim_names``

            Args:
                mesh_dim_names (Union[str, tuple[str, ...]]): the name or the tuple of names of the
                mesh dimension of the DeviceMesh to create the submesh for.
            Returns:
                A :class:`DeviceMesh` object

            The following program runs on each process/rank in an SPMD manner in a world size of 8.
            In the first example:
                Calling mesh_2d["tp"] on rank 0, 1, 2, 3 returns a 1D submesh of DeviceMesh:([0, 1, 2, 3]).
                Calling mesh_2d["tp"] on rank 4, 5, 6, 7 returns a 1D submesh of  DeviceMesh:([4, 5, 6, 7]).
                Calling mesh_2d["dp"] on rank 0, 4 returns a 1D submesh of  DeviceMesh:([0, 4]).
                Calling mesh_2d["dp"] on rank 1, 5 returns a 1D submesh of  DeviceMesh:([1, 5]).
                Calling mesh_2d["dp"] on rank 2, 6 returns a 1D submesh of  DeviceMesh:([2, 6]).
                Calling mesh_2d["dp"] on rank 3, 7 returns a 1D submesh of  DeviceMesh:([3, 7]).

            In the second example:
                Calling mesh_3d["dp", "cp"] on rank 0, 1, 4, 5 returns a 2D submesh of DeviceMesh:([[0, 1], [4, 5]]).
                Calling mesh_3d["dp", "cp"] on rank 2, 3, 6, 7 returns a 2D submesh of DeviceMesh:([[2, 3], [6, 7]]).
                Calling mesh_3d["cp", "dp"] on rank 0, 1, 4, 5 returns a 2D submesh of DeviceMesh:([[0, 4], [1, 5]]).
                Calling mesh_3d["cp", "dp"] on rank 2, 3, 6, 7 returns a 2D submesh of DeviceMesh:([[2, 6], [3, 7]]).

            Example::

                >>> # xdoctest: +SKIP("no rank")
                >>> from torch.distributed.device_mesh import DeviceMesh
                >>>
                >>> # Initialize a 2D device mesh as (2, 4) to represent the topology
                >>> # of cross-host(dim 0), and within-host (dim 1).
                >>> mesh_2d = init_device_mesh(device_type="cuda", (2,4), mesh_dim_names=("dp", "tp"))
                >>> tp_mesh = mesh_2d["tp"]
                >>> dp_mesh = mesh_2d["dp"]
                >>>
                >>> # Initialize a 3D mesh.
                >>> mesh_3d = init_device_mesh(device_type="cuda", (2,2,2), mesh_dim_names=("dp", "pp", "cp"))
                >>> # The order of the mesh_dim_names provided determines the order of dimensions in the submesh.
                >>> dp_cp_mesh = mesh_3d["dp", "cp"]
                >>> cp_dp_mesh = mesh_3d["cp", "dp"]
            """
            if not self._mesh_dim_names:
                raise RuntimeError("Cannot slice a DeviceMesh without mesh_dim_names!")

            mesh_dim_names = (
                (mesh_dim_names,) if isinstance(mesh_dim_names, str) else mesh_dim_names
            )

            if mesh_dim_names == self._mesh_dim_names:
                return self
            else:
                sliced_mesh_layout = self._get_slice_mesh_layout(mesh_dim_names)
                # When using FakeTensorMode to trace the model, `_create_sub_mesh()` will
                # fail as it will require a real tensor to manipulate.
                # `unset_fake_temporarily()` and `disable_proxy_modes_tracing()`
                # will allow us to materialize the tensors within
                # `_create_sub_mesh`, which should not affect modeling.
                #
                # Note that this should be orthogonal to torch.compile(). But whether
                # we can compile device_mesh `slicing` (no graph break) is not verified
                # yet and need a follow-up,
                # TODO: compiler + device_mesh slicing.
                from torch.fx.experimental.proxy_tensor import (
                    disable_proxy_modes_tracing,
                )

                with (
                    torch._subclasses.fake_tensor.unset_fake_temporarily(),
                    disable_proxy_modes_tracing(),
                ):
                    submesh = self._create_sub_mesh(sliced_mesh_layout, mesh_dim_names)
                return submesh

        def get_group(self, mesh_dim: int | str | None = None) -> ProcessGroup:
            """
            Returns the single ProcessGroup specified by mesh_dim, or, if mesh_dim is not specified and the
            DeviceMesh is 1-dimensional, returns the only ProcessGroup in the mesh.

            Args:
                mesh_dim (str/int, optional): it can be the name of the mesh dimension or the index
                of the mesh dimension. Default is None.

            Returns:
                A :class:`ProcessGroup` object.
            """
            if not hasattr(self, "_dim_group_names"):
                raise RuntimeError("DeviceMesh process groups not initialized!")

            if len(self._layout) > 1 and mesh_dim is None:
                raise RuntimeError(
                    f"Found the DeviceMesh have {len(self._layout)} dimensions",
                    "Optional kwarg `mesh_dim` needs to be specified when device_mesh.ndim > 1.",
                    "If you want to get the list of all the ProcessGroups in the DeviceMesh,"
                    "please use `get_all_groups()` instead.",
                )

            root_mesh = self._get_root_mesh()

            # Quick return if the current device_mesh is a 1D mesh.
            if len(self._layout) == 1 and mesh_dim is None:
                traced = _maybe_traced_group(self, 0)
                if traced is not None:
                    return traced
                return not_none(_get_pg_from_name(root_mesh, self._dim_group_names[0]))

            root_to_flatten_mapping = root_mesh._flatten_mapping
            if root_to_flatten_mapping and mesh_dim in root_to_flatten_mapping:
                dim_group_name = root_to_flatten_mapping[
                    mesh_dim  # type: ignore[index]
                ]._dim_group_names[0]
                return not_none(_get_pg_from_name(root_mesh, dim_group_name))
            else:
                mesh_dim = (
                    self._get_mesh_dim_by_name(mesh_dim)
                    if isinstance(mesh_dim, str)
                    else mesh_dim
                )
                if not isinstance(mesh_dim, int):
                    raise AssertionError(
                        f"mesh_dim must be an int, got {type(mesh_dim)}"
                    )
                traced = _maybe_traced_group(self, mesh_dim)
                if traced is not None:
                    return traced
                return not_none(
                    _get_pg_from_name(root_mesh, self._dim_group_names[mesh_dim])
                )

        def get_all_groups(self) -> list[ProcessGroup]:
            """
            Returns a list of ProcessGroups for all mesh dimensions.

            Returns:
                A list of :class:`ProcessGroup` object.
            """
            return [self.get_group(i) for i in range(len(self._layout))]

        def _create_sub_mesh(
            self,
            layout: _MeshLayout,
            submesh_dim_names: tuple[str, ...],
        ) -> "DeviceMesh":
            with torch._dynamo.disable_nested_graph_breaks():
                root_mesh = self._get_root_mesh()
                slice_dim_group_name = []
                dim_names = not_none(self._mesh_dim_names)
                if len(self._dim_group_names) > 0:
                    if len(self._dim_group_names) != len(dim_names):
                        raise AssertionError(
                            "The number of dim_group_names and mesh_dim_names "
                            "should have the same length if the rank is in the mesh."
                        )
                    for name in submesh_dim_names:
                        if name in dim_names:
                            slice_dim_group_name.append(
                                self._dim_group_names[dim_names.index(name)]
                            )
                        else:
                            # If device_mesh is not root_mesh, we already throw error in _get_slice_mesh_layout
                            # Since we will deprecate the slicing of flattened dim_name from root mesh soon,
                            # we don't want to optimize the code furthermore.
                            flatten_mesh = self._flatten_mapping[name]
                            slice_dim_group_name.append(
                                flatten_mesh._dim_group_names[
                                    not_none(flatten_mesh._mesh_dim_names).index(name)
                                ]
                            )
                res_submesh = DeviceMesh(
                    self._device_type,
                    _layout=layout,
                    _rank_map=root_mesh._rank_map,
                    mesh_dim_names=submesh_dim_names,
                    _root_mesh=root_mesh,
                    _init_backend=False,
                )
                res_submesh._dim_group_names = slice_dim_group_name
                return res_submesh

        def _create_flatten_mesh(
            self,
            mesh_dim_name: str,
            backend_override: BackendConfig = (None, None),
        ) -> None:
            """
            Creates a flattened mesh and stores it in the root mesh's _flatten_mapping.
            This is a helper method for _flatten() that performs the actual mesh creation.
            """
            root_mesh = self._get_root_mesh()

            # Check whether the mesh_dim_name for flattened mesh is valid.
            invalid_dim_names = not_none(root_mesh._mesh_dim_names)
            if mesh_dim_name in invalid_dim_names:
                raise ValueError(
                    f"{mesh_dim_name} already exists for submesh of the {root_mesh}. ",
                    f"The mesh_dim_names of submesh and flattened mesh are {invalid_dim_names}. "
                    f"Please specify another valid mesh_dim_name.",
                )

            flattened_mesh_layout = _MeshLayout([self._layout.collapse()])

            # Check if the flatten mesh has been created before.
            if mesh_dim_name in root_mesh._flatten_mapping:
                if (
                    flattened_mesh_layout
                    == root_mesh._flatten_mapping[mesh_dim_name]._layout
                ):
                    # Already exists with same layout, nothing to do
                    return
                else:
                    raise ValueError(
                        f"Flatten mesh with mesh_dim_name {mesh_dim_name} has been created before, "
                        f"Please specify another valid mesh_dim_name."
                    )

            res_flattened_mesh = DeviceMesh(
                root_mesh._device_type,
                _layout=flattened_mesh_layout,
                _rank_map=root_mesh._rank_map,
                mesh_dim_names=(mesh_dim_name,),
                _root_mesh=root_mesh,
                backend_override=(backend_override,),
            )
            root_mesh._flatten_mapping[mesh_dim_name] = res_flattened_mesh

        def _get_root_mesh_dim(self) -> int | None:
            """
            Returns the index of the mesh dim in the root mesh.
            The device_mesh passed in needs to be sliced out from the root mesh
            or submesh of the root mesh.
            """
            root_mesh = self._get_root_mesh()
            child_mesh_dim_names = self._mesh_dim_names
            if root_mesh and child_mesh_dim_names:
                if len(child_mesh_dim_names) != 1:
                    raise AssertionError("The submesh can only be a 1D mesh.")
                child_mesh_dim_name = child_mesh_dim_names[0]
                return root_mesh._get_mesh_dim_by_name(child_mesh_dim_name)
            return None

        def _get_mesh_dim_by_name(self, mesh_dim_name: str) -> int:
            if self._mesh_dim_names is None or len(self._mesh_dim_names) == 0:
                raise KeyError(
                    "No `mesh_dim_names` found.",
                )
            if mesh_dim_name not in self._mesh_dim_names:
                raise KeyError(
                    f"Mesh dimension '{mesh_dim_name}' does not exist.",
                    f"Available mesh dimensions are: mesh_dim_names={self._mesh_dim_names}",
                )
            return not_none(self._mesh_dim_names.index(mesh_dim_name))

        def _get_slice_mesh_layout(
            self, mesh_dim_names: tuple[str, ...]
        ) -> _MeshLayout:
            """
            Validate whether the mesh_dim_names is valid for slicing the given device_mesh.
            If valid, return dim indexes of the slice mesh in the device mesh.
            """
            slice_from_root = True
            if self != self._get_root_mesh():
                slice_from_root = False

            # The slice mesh_dim_names should consist either the current device_mesh's mesh_dim_names
            # or its flattened mesh's mesh_dim_names if it's root_mesh.
            flatten_name_to_root_layout: dict[str, _FlatLayout] = (
                {
                    key: mesh._layout[0]  # Extract the single axis from flattened mesh
                    for key, mesh in self._get_root_mesh()._flatten_mapping.items()
                }
                if slice_from_root
                else {}
            )
            valid_mesh_dim_names = [
                *not_none(self._mesh_dim_names),
                *flatten_name_to_root_layout,
            ]

            if not all(
                mesh_dim_name in valid_mesh_dim_names
                for mesh_dim_name in mesh_dim_names
            ):
                raise KeyError(
                    f"Invalid mesh_dim_names {mesh_dim_names} specified. "
                    f"Valid mesh_dim_names are {valid_mesh_dim_names}."
                )

            layout_sliced: list[_FlatLayout] = []
            for name in mesh_dim_names:
                if name in not_none(self._mesh_dim_names):
                    layout_sliced.append(
                        self._layout[not_none(self._mesh_dim_names).index(name)]
                    )
                elif name in flatten_name_to_root_layout:
                    warnings.warn(
                        "Slicing a flattened dim from root mesh will be deprecated in PT 2.11. "
                        "Users need to bookkeep the flattened mesh directly. ",
                        stacklevel=2,
                    )
                    layout_sliced.append(flatten_name_to_root_layout[name])
            result_layout = _MeshLayout(layout_sliced)

            # The check below is from DeviceMesh's implementation before adopting CuTe layout for internal
            # bookkeeping and it can be removed but we need to define what is the expected behavior.
            # TODO: Remove the below check and define the expected behavior.
            # Validate the order of the slice mesh dim indices.
            # This needs to be in ascending order.
            if not result_layout.collapse().check_sorted():
                raise KeyError(
                    f"Invalid mesh_dim_names {mesh_dim_names} specified. "
                    "Mesh dim indices should be in ascending order."
                )

            # When users sliced dim_names outside from current mesh, we will check whether
            # there is layout overlap.
            # TODO: Eventually we will just directly throw error here because
            # we will deprecate the slicing of flattened dim_name from root mesh.
            if not result_layout.collapse().check_orthogonal():
                raise RuntimeError(
                    f"Slicing overlapping dim_names {mesh_dim_names} is not allowed."
                )

            return result_layout

        # TODO: to make this use case by other components public API in the future.
        def _get_all_submeshes(self, mesh_dim_name: str) -> list["DeviceMesh"]:
            """
            Return all the submeshes of a given mesh dimension of the device mesh.
            """
            mesh_dim = self._get_mesh_dim_by_name(mesh_dim_name)
            layout = self._layout[mesh_dim]
            pg_ranks_by_dim = _MeshLayout([layout]).remap_to_tensor(self._rank_map)
            cur_rank = self.get_rank()
            res_submeshes = []
            for mesh_1d in pg_ranks_by_dim:
                submesh = DeviceMesh(
                    self._device_type,
                    mesh_1d,
                    mesh_dim_names=(mesh_dim_name,),
                    _init_backend=False,
                )
                submesh._dim_group_names = (  # type: ignore[has-type]
                    [self._dim_group_names[mesh_dim]]  # type: ignore[has-type]
                    if cur_rank in mesh_1d
                    else []
                )
                res_submeshes.append(submesh)

            return res_submeshes

        @staticmethod
        def from_group(
            group: ProcessGroup | list[ProcessGroup],
            device_type: str,
            mesh: "torch.Tensor | ArrayLike | None" = None,
            *,
            mesh_dim_names: tuple[str, ...] | None = None,
        ) -> "DeviceMesh":
            """
            Constructs a :class:`DeviceMesh` with ``device_type`` from an
            existing :class:`ProcessGroup` or a list of existing :class:`ProcessGroup`.

            The constructed device mesh has number of dimensions equal to the
            number of groups passed. For example, if a single process group is passed in,
            the resulted DeviceMesh is a 1D mesh. If a list of 2 process groups is passed in,
            the resulted DeviceMesh is a 2D mesh.

            If more than one group is passed, then the ``mesh`` and ``mesh_dim_names`` arguments
            are required. The order of the process groups passed in determines the topology of
            the mesh. For example, the first process group will be the 0th dimension of the DeviceMesh.
            The `mesh` tensor passed in must have the same number of dimensions as the number of process
            groups passed in, and the order of the dimensions in the `mesh` tensor must match the order
            in the process groups passed in.

            Args:
                group (ProcessGroup or list[ProcessGroup]): the existing ProcessGroup
                    or a list of existing ProcessGroups.
                device_type (str): The device type of the mesh. Currently supports: "cpu",
                    "cuda/cuda-like". Passing in a device type with a GPU index, such as "cuda:0",
                    is not allowed.
                mesh (torch.Tensor or ArrayLike, optional): A multi-dimensional array or an
                    integer tensor describing the layout of devices, where the IDs are global IDs
                    of the default process group. Default is None.
                mesh_dim_names (tuple[str, ...], optional): A tuple of mesh dimension names to assign
                    to each dimension of the multi-dimensional array describing the layout of devices.
                    Its length must match the length of `mesh_shape`. Each string in `mesh_dim_names`
                    must be unique. Default is None.

            Returns:
                DeviceMesh: A :class:`DeviceMesh` object representing the device layout.
            """

            # 1D scenario
            if isinstance(group, ProcessGroup):
                group_ranks = get_process_group_ranks(group)
                if (
                    isinstance(mesh, torch.Tensor) and mesh.tolist() != group_ranks
                ) or (
                    mesh is not None
                    and not isinstance(mesh, torch.Tensor)
                    and mesh != group_ranks
                ):
                    raise ValueError(
                        f"Invalid mesh {str(mesh)} for ProcessGroup with ranks {group_ranks}"
                    )
                mesh = torch.tensor(group_ranks, device="cpu", dtype=torch.int)
                device_mesh = DeviceMesh(
                    device_type,
                    mesh,
                    mesh_dim_names=mesh_dim_names,
                    _init_backend=False,
                )
                device_mesh._dim_group_names = [group.group_name]
                device_mesh._pg_registry[group.group_name] = group
                return device_mesh

            # nD scenario
            groups = list(group)
            if len(groups) == 0:
                raise ValueError("Expects at least one ProcessGroup to be passed")
            if mesh is None:
                raise ValueError("Must pass mesh if passing multiple ProcessGroups")
            if mesh_dim_names is None:
                raise ValueError(
                    "Must pass mesh_dim_names if passing multiple ProcessGroups"
                )
            # When init a DeviceMesh with multiple ProcessGroups directly, we need to make sure
            # the mesh tensor is contiguous. Otherwise, the layout we inferred from the mesh tensor
            # will have larger span than the actual tensor. This is just internal implementation detail
            # and does not affect user facing behavior.
            mesh = (
                mesh.detach().to(dtype=torch.int, device="cpu")
                if isinstance(mesh, torch.Tensor)
                else torch.tensor(mesh, device="cpu", dtype=torch.int)
            )
            if mesh.ndim != len(groups):
                raise ValueError(
                    "Expects mesh with ndim equal to number of ProcessGroups but got "
                    f"mesh {mesh.tolist()} and {len(groups)} ProcessGroups"
                )
            device_mesh = DeviceMesh(
                device_type, mesh, mesh_dim_names=mesh_dim_names, _init_backend=False
            )
            device_mesh._dim_group_names = [group.group_name for group in groups]
            for group in groups:
                device_mesh._pg_registry[group.group_name] = group
            return device_mesh

        def size(self, mesh_dim: int | None = None) -> int:
            if mesh_dim is not None:
                return self._layout[mesh_dim].numel()
            return self._layout.numel()

        @property
        def ndim(self) -> int:
            return len(self._layout)

        @property
        def shape(self) -> tuple[int, ...]:
            return self._layout.top_level_sizes

        def get_rank(self) -> int:
            """
            Returns the current global rank.
            """
            return get_rank()

        def get_local_rank(self, mesh_dim: int | str | None = None) -> int:
            """
            Returns the local rank of the given mesh_dim of the DeviceMesh.

            Args:
                mesh_dim (str/int, optional): it can be the name of the mesh dimension or the index
                of the mesh dimension. Default is None.

            Returns:
                An integer denotes the local rank.

            The following program runs on each process/rank in an SPMD manner. In this example, we have 2
            hosts with 4 GPUs each.
            Calling mesh_2d.get_local_rank(mesh_dim=0) on rank 0, 1, 2, 3 would return 0.
            Calling mesh_2d.get_local_rank(mesh_dim=0) on rank 4, 5, 6, 7 would return 1.
            Calling mesh_2d.get_local_rank(mesh_dim=1) on rank 0, 4 would return 0.
            Calling mesh_2d.get_local_rank(mesh_dim=1) on rank 1, 5 would return 1.
            Calling mesh_2d.get_local_rank(mesh_dim=1) on rank 2, 6 would return 2.
            Calling mesh_2d.get_local_rank(mesh_dim=1) on rank 3, 7 would return 3.

            Example::

                >>> # xdoctest: +SKIP("no rank")
                >>> from torch.distributed.device_mesh import DeviceMesh
                >>>
                >>> # Initialize device mesh as (2, 4) to represent the topology
                >>> # of cross-host(dim 0), and within-host (dim 1).
                >>> mesh = DeviceMesh(device_type="cuda", mesh=[[0, 1, 2, 3],[4, 5, 6, 7]])
            """
            if self.ndim > 1 and mesh_dim is None:
                raise RuntimeError(
                    f"Found the DeviceMesh have {len(self._layout)} dimensions",
                    "Optional kwarg `mesh_dim` needs to be specified when device_mesh.ndim > 1.",
                )
            elif mesh_dim is None:
                mesh_dim = 0

            mesh_dim_group = not_none(self.get_group(mesh_dim))
            if not isinstance(mesh_dim_group, ProcessGroup):
                raise AssertionError(
                    "We expect ProcessGroup before calling `get_rank`!"
                )
            return not_none(get_rank(mesh_dim_group))

        def _is_current_rank_part_of_mesh(self) -> bool:
            """
            Return True if the current rank is part of this mesh.

            When a DeviceMesh is created with a subset of ranks (a sub-mesh),
            ranks not included in the mesh are "non-participating ranks". These
            ranks:

            - Return None from get_coordinate()
            - Hold empty tensors as their local DTensor representation
            - Skip computation during DTensor dispatch (returning default values)
            - Skip collective operations during redistribute
            - Return 0 cost for redistribute cost calculations

            This allows DTensor operations to execute correctly across the entire
            process group while only performing actual work on participating ranks.
            """
            return self._coordinate_on_dim is not None

        def get_coordinate(self) -> tuple[int, ...] | None:
            """
            Return the relative indices of this rank relative to all
            dimensions of the mesh. If this rank is not part of the mesh, return None.
            """
            return self._coordinate_on_dim

        def _sym_get_coordinate(self, index: int) -> IntLikeType:
            import torch.compiler.config as config
            from torch._guards import detect_fake_mode

            if (
                not config.compile_on_one_rank
                or not (fake_mode := detect_fake_mode())
                or not fake_mode.shape_env
            ):
                # This is only valid when the current rank is part of the mesh.
                if self._coordinate_on_dim is None:
                    raise AssertionError
                return self._coordinate_on_dim[index]

            # This will cause the ops to be registered - so don't let RUFF
            # delete this import because it thinks it's unused...
            from ._ops import device_mesh  # noqa: F401

            # Temporarily turn off tracing while we lift the constant
            # rank_map to a list so it can be a constant in the graph.
            with torch._subclasses.fake_tensor.unset_fake_temporarily():
                rank_map_list = self._rank_map.tolist()
            rank_map = torch.tensor(rank_map_list, device="cpu", dtype=torch.int)
            full_mesh = self._layout.remap_to_tensor(rank_map)

            return torch.ops.device_mesh._runtime_compute_coordinate_on_dim(
                full_mesh, index
            )

        def _flatten(
            self,
            mesh_dim_name: str | None = None,
            backend_override: str
            | C10dBackend.Options
            | tuple[str, C10dBackend.Options]
            | None = None,
        ) -> "DeviceMesh":
            """
            Returns a 1D DeviceMesh by flattening the current DeviceMesh.

            If no mesh_dim_name is provided, the default is a string concatenating the mesh_dim_names of the
            given submesh with each mesh_dim_name separated by "_". For example, if we have a 3D mesh
            DeviceMesh([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], mesh_dim_names=("dp", "cp", "tp")), calling
            mesh_3d["dp", "cp"]._flatten() will create a 1D submesh DeviceMesh([0, 2, 4, 6], mesh_dim_names=("dp_cp",))
            on rank 0, 2, 4, 6 and a 1D submesh DeviceMesh([1, 3, 5, 7], mesh_dim_names=("dp_cp",)) on rank 1, 3, 5, 7.

            After the flattened dimension is created, to access the flattened dimension in mesh_3d, one can use the
            existing slicing method to obtain the flattened mesh through calling mesh_3d["dp_cp"].
            """
            if not self._mesh_dim_names:
                raise RuntimeError(
                    "Cannot flatten a DeviceMesh without mesh_dim_names!"
                )

            # Compute mesh_dim_name first if not provided
            if not mesh_dim_name:
                mesh_dim_name = "_".join(not_none(self._mesh_dim_names))

            # Flatten a 1D device mesh into its original mesh_dim_name will return itself.
            if self.ndim == 1 and mesh_dim_name in not_none(self._mesh_dim_names):
                return self

            if backend_override is not None:
                (backend_override_tuple,) = _normalize_backend_override(
                    {0: backend_override}, 1
                )
            else:
                backend_override_tuple = (None, None)

            self._create_flatten_mesh(mesh_dim_name, backend_override_tuple)

            return self._get_root_mesh()._flatten_mapping[mesh_dim_name]

        def _create_unflatten_mesh(
            self,
            dim: int,
            mesh_sizes: tuple[int, ...],
            mesh_dim_names: tuple[str, ...],
            backend_override: tuple[
                tuple[str | None, C10dBackend.Options | None], ...
            ] = ((None, None),),
        ) -> "DeviceMesh":
            inner_layout = _MeshLayout.from_sizes_strides(tuple(mesh_sizes))

            if inner_layout.numel() != self._layout[dim].numel():
                raise ValueError(
                    f"The product of {mesh_sizes=} is {inner_layout.numel()}, "
                    f"but the original dimension at dim={dim} has size {self._layout[dim].numel()}. "
                    f"These must be equal for unflatten to work correctly."
                )

            partial_layout = self._layout[dim].composition(inner_layout)
            unflattened_layout = self._layout.splice(dim, dim + 1, partial_layout)
            unflattened_mesh_dim_names = list(not_none(self.mesh_dim_names))
            unflattened_mesh_dim_names[dim : dim + 1] = list(mesh_dim_names)

            root_mesh = self._get_root_mesh()
            res_mesh = DeviceMesh(
                self.device_type,
                _layout=unflattened_layout,
                _rank_map=root_mesh._rank_map,
                mesh_dim_names=tuple(unflattened_mesh_dim_names),
                _root_mesh=root_mesh,
                _init_backend=False,
            )

            # If original mesh has initiated its backend, we need to initialize the backend
            # of unflatten dims as well.
            # TODO: To make backend init more efficient with cute layout representation and support
            # per dim backend init.
            if hasattr(self, "_dim_group_names"):
                dim_group_names = self._dim_group_names.copy()
                new_group_names = self._init_process_groups(
                    partial_layout,
                    root_mesh._rank_map,
                    mesh_dim_names,
                    backend_override,
                )
                dim_group_names[dim : dim + 1] = new_group_names
                res_mesh._dim_group_names = dim_group_names
                # Populate root mesh's pg registry with new groups
                for name in new_group_names:
                    pg = _resolve_process_group(name)
                    if pg is not None:
                        root_mesh._pg_registry[name] = pg

            return res_mesh

        def _unflatten(
            self,
            dim: int | str,
            mesh_sizes: tuple[int, ...],
            mesh_dim_names: tuple[str, ...],
            backend_override: dict[
                str, str | C10dBackend.Options | tuple[str, C10dBackend.Options]
            ]
            | None = None,
        ) -> "DeviceMesh":
            """
            Returns a DeviceMesh by unflatten the current DeviceMesh.

            This api can be used to unflatten a N-D DeviceMesh into N-1+len(mesh_sizes)-D meshes or submeshes.
            The dim is the dimension to be unflattened which can be either a string or an integer.

            The mesh_sizes is a tuple which specifies the shape of the mesh unflatten into for the given dim.
            The mesh_dim_names is a list of strings which specifies the names of the dimensions of the mesh unflatten into.
            Its length must match the length of mesh_sizes.

            For example, if we have a 1D mesh DeviceMesh([0, 1, 2, 3, 4, 5, 6, 7], mesh_dim_names=("world")),
            calling mesh_1d._unflatten(0, (2, 2, 4), ["dp", "pp", "tp"]) will create a 3D mesh
            DeviceMesh([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], mesh_dim_names=("dp", "cp", "tp")).

            Note that after calling the unflatten, there is no access to the unflattened dimension in mesh_1d, one can only
            use the newly unflattened mesh to slice out the unflattened mesh dims.
            """
            if isinstance(dim, int) and dim >= self.ndim:
                raise ValueError(
                    f"dim {dim} specified in `_unflatten` is out of range {self.ndim}"
                )
            elif isinstance(dim, str) and dim not in not_none(self.mesh_dim_names):
                raise ValueError(
                    f"dim {dim} specified in `_unflatten` is not in {self.mesh_dim_names}"
                )

            if len(mesh_sizes) != len(mesh_dim_names):
                raise RuntimeError(
                    "mesh_dim_names must have same length as mesh_sizes in _unflatten!"
                )

            if isinstance(dim, str):
                dim = not_none(self.mesh_dim_names).index(dim)

            if backend_override is not None:
                backend_override_tuple = tuple(
                    _normalize_backend_override(
                        backend_override,  # type: ignore[arg-type]
                        len(mesh_sizes),
                        mesh_dim_names,
                    )
                )
            else:
                backend_override_tuple = ((None, None),) * len(mesh_dim_names)

            return self._create_unflatten_mesh(
                dim,
                mesh_sizes,
                mesh_dim_names,
                backend_override_tuple,
            )

        @staticmethod
        def _concatenate(device_mesh_list: list["DeviceMesh"]) -> "DeviceMesh":
            concat_dim_names: list[str] = []
            concat_axes: list[_FlatLayout] = []
            concat_dim_group_name: list[GroupName] = []
            flatten_rank_map = device_mesh_list[0]._flatten_rank_map
            for dm in device_mesh_list:
                concat_axes.extend(dm._layout)
                concat_dim_names.extend(not_none(dm.mesh_dim_names))
                concat_dim_group_name.extend(not_none(dm._dim_group_names))
                # Concatenate device mesh having different root mesh tensors are meaningless
                # because the concatenated indices should be indexed by the same root mesh tensor.
                if dm._flatten_rank_map != flatten_rank_map:
                    raise RuntimeError(
                        "Cannot concatenate DeviceMeshes derived from different device meshes"
                    )
            concat_mesh_layout = _MeshLayout(concat_axes)
            if not concat_mesh_layout.collapse().check_orthogonal():
                raise RuntimeError(
                    f"Cannot concatenate overlapping meshes: {device_mesh_list}"
                )
            res_mesh = DeviceMesh(
                device_mesh_list[0].device_type,
                _layout=concat_mesh_layout,
                _rank_map=device_mesh_list[0]._rank_map,
                mesh_dim_names=tuple(concat_dim_names),
                _root_mesh=device_mesh_list[0]._get_root_mesh(),
                _init_backend=False,
            )
            res_mesh._dim_group_names = concat_dim_group_name
            return res_mesh

    def _normalize_backend_override(
        backend_override: dict[
            int | str,
            str | C10dBackend.Options | tuple[str, C10dBackend.Options],
        ],
        ndim: int,
        mesh_dim_names: tuple[str, ...] | None = None,
    ) -> Iterator[BackendConfig]:
        if mesh_dim_names is None:
            mesh_dim_names = ()
        for dim_idx, dim_name in zip_longest(range(ndim), mesh_dim_names):
            if dim_name is not None and dim_name in backend_override:
                if dim_idx in backend_override:
                    raise RuntimeError(
                        f"Found redundant dim index {dim_idx} and "
                        f"name {dim_name} in backend_override"
                    )
                val = backend_override.pop(dim_name)
            elif dim_idx in backend_override:
                val = backend_override.pop(dim_idx)
            else:
                yield (None, None)
                continue

            if isinstance(val, str):
                yield (val, None)
            elif isinstance(val, C10dBackend.Options):
                yield (None, val)
            else:
                yield val

        if backend_override:
            raise RuntimeError(
                f"Found invalid keys in backend_override: got {list(backend_override.keys())}, "
                f"expected integers in range [0, {ndim}) or one of {mesh_dim_names}"
            )

    def init_device_mesh(
        device_type: str,
        mesh_shape: tuple[int, ...],
        *,
        mesh_dim_names: tuple[str, ...] | None = None,
        backend_override: dict[
            int | str, str | C10dBackend.Options | tuple[str, C10dBackend.Options]
        ]
        | None = None,
    ) -> DeviceMesh:
        """
        Initializes a `DeviceMesh` based on `device_type`, `mesh_shape`, and `mesh_dim_names` parameters.

        This creates a DeviceMesh with an n-dimensional array layout, where `n` is the length of `mesh_shape`.
        If `mesh_dim_names` is provided, each dimension is labeled as `mesh_dim_names[i]`.

        .. note::
            `init_device_mesh` follows SPMD programming model, meaning the same PyTorch Python program
            runs on all processes/ranks in the cluster. Ensure `mesh_shape` (the dimensions of the nD array
            describing device layout) is identical across all ranks. Inconsistent `mesh_shape` may lead to hanging.

        .. note::
            If no process group is found, init_device_mesh will initialize distributed process group/groups
            required for distributed communications behind the scene.

        Args:
            device_type (str): The device type of the mesh. Currently supports: "cpu", "cuda/cuda-like", "xpu".
                Passing in a device type with a GPU index, such as "cuda:0", is not allowed.
            mesh_shape (Tuple[int]): A tuple defining the dimensions of the multi-dimensional array
                describing the layout of devices.
            mesh_dim_names (tuple[str, ...], optional): A tuple of mesh dimension names to assign to each dimension
                of the multi-dimensional array describing the layout of devices. Its length must match the length
                of `mesh_shape`. Each string in `mesh_dim_names` must be unique.
            backend_override (Dict[int | str, tuple[str, Options] | str | Options], optional): Overrides for some or all of
                the ProcessGroups that will be created for each mesh dimension. Each key can be either the index of a
                dimension or its name (if mesh_dim_names is provided). Each value can be a tuple containing the name
                of the backend and its options, or just one of these two components (in which case the other will be
                set to its default value).

        Returns:
            DeviceMesh: A :class:`DeviceMesh` object representing the device layout.

        Example::

            >>> # xdoctest: +SKIP("no rank")
            >>> from torch.distributed.device_mesh import init_device_mesh
            >>>
            >>> mesh_1d = init_device_mesh("cuda", mesh_shape=(8,))
            >>> mesh_2d = init_device_mesh("cuda", mesh_shape=(2, 8), mesh_dim_names=("dp", "tp"))

        """
        if mesh_dim_names is not None:
            if len(set(mesh_dim_names)) != len(mesh_dim_names):
                raise RuntimeError(
                    "Each mesh_dim_name must be unique. "
                    f"Found repeated mesh_dim_name in mesh_dim_names {mesh_dim_names}"
                )

            if len(mesh_shape) != len(mesh_dim_names):
                raise RuntimeError(
                    "mesh_shape and mesh_dim_names should have same length! "
                    f"Found len(mesh_dim_names): {len(mesh_dim_names)} and len(mesh_shape):{len(mesh_shape)}."
                )

        if backend_override is not None:
            backend_override_tuple = tuple(
                _normalize_backend_override(
                    backend_override, len(mesh_shape), mesh_dim_names
                )
            )
        else:
            backend_override_tuple = None

        # assume valid device types are all letters
        if device_type and not device_type.isalpha():
            raise RuntimeError(
                f"Device type with index is not supported but got {device_type}. ",
                "If you maintained a 'torch.device' object, it's recommended to pass in 'device.type'.",
            )

        layout = _MeshLayout.from_sizes_strides(tuple(mesh_shape))
        # Always initialize the (identity) rank map on CPU, regardless of what the
        # external device type has been set to be (e.g. meta)
        with torch.device("cpu"):
            rank_map = torch.arange(layout.numel(), dtype=torch.int)
        device_mesh = DeviceMesh(
            device_type=device_type,
            _layout=layout,
            _rank_map=rank_map,
            mesh_dim_names=mesh_dim_names,
            backend_override=backend_override_tuple,
        )

        return device_mesh


_distributed_opaque_types_registered = False


def _device_mesh_reconstruct_fn(
    mesh: "CustomClassBase",
    get_tracked_proxy: Callable[["CustomClassBase"], "torch.fx.Proxy | None"],
    tracer: Any,
) -> "torch.fx.Proxy | None":
    """Reconstruct a DeviceMesh submesh from a tracked ancestor mesh.

    Called by PythonKeyTracer when make_fx encounters a DeviceMesh that isn't
    tracked (e.g. a submesh captured by a backward closure). Looks for any
    tracked mesh that shares the same root and contains the target dim names,
    then emits a call_function node that derives the submesh via _get_submesh.
    """
    if not isinstance(mesh, DeviceMesh):
        raise AssertionError("DeviceMesh expected")

    root_mesh = mesh._get_root_mesh()

    # Only submeshes can be reconstructed; root meshes must already be tracked.
    if mesh is root_mesh:
        return None

    dim_names = mesh._mesh_dim_names
    if dim_names is None:
        return None

    # Ensure the custom ops are registered
    from torch.distributed._ops import device_mesh as _dm_ops  # noqa: F401

    # Try the root mesh first (original path).
    ancestor_proxy = get_tracked_proxy(root_mesh)
    ancestor_dim_names = root_mesh._mesh_dim_names

    # If root isn't tracked, search for any tracked DeviceMesh that shares
    # the same root AND contains all our dim names. This handles the case
    # where e.g. a concatenated (fsdp, tp) mesh is a graph input (from
    # DTensor.__tensor_flatten__) but neither root nor the individual
    # submeshes are tracked directly.
    if ancestor_proxy is None:
        from torch._library.fake_class_registry import FakeScriptObject

        for tracked_obj, proxy in tracer.opaque_tracker.items():
            real_obj = (
                tracked_obj.real_obj
                if isinstance(tracked_obj, FakeScriptObject)
                else tracked_obj
            )
            if not isinstance(real_obj, DeviceMesh) or real_obj is mesh:
                continue
            if real_obj._get_root_mesh() is not root_mesh:
                continue
            tracked_dim_names = real_obj._mesh_dim_names
            if tracked_dim_names is None:
                continue
            if all(n in tracked_dim_names for n in dim_names):
                ancestor_proxy = proxy
                ancestor_dim_names = tracked_dim_names
                break

    if ancestor_proxy is None or ancestor_dim_names is None:
        return None

    # Convert our dim names to indices into the ancestor mesh's dim names
    mesh_dims = [ancestor_dim_names.index(n) for n in dim_names]

    # Dispatch through the custom op with proxy mode active so that
    # meta["val"] is set and the result is tracked in opaque_tracker.
    return torch.ops.device_mesh._get_submesh(ancestor_proxy, mesh_dims)


def _register_distributed_opaque_types():
    """
    Register DeviceMesh as an opaque type for torch.compile.
    This must happen before any custom ops that use DeviceMesh in their schema.
    Called lazily to avoid circular import issues.
    """
    global _distributed_opaque_types_registered
    if _distributed_opaque_types_registered:
        return
    _distributed_opaque_types_registered = True

    from torch._library.opaque_object import MemberType, register_custom_class

    _register_process_group_opaque_type()

    register_custom_class(
        DeviceMesh,
        typ="symbolic",
        reconstruct_fn=_device_mesh_reconstruct_fn,
        guard_fn=lambda obj: [
            obj._flatten_rank_map,
            obj._layout,
            obj._device_type,
            obj._mesh_dim_names,
            obj._thread_id,
        ],
        members={
            # USE_REAL: Evaluate these with the real object at compile time
            # and bake the result as a constant
            "_flatten_rank_map": MemberType.USE_REAL,
            "_layout": MemberType.USE_REAL,
            "_device_type": MemberType.USE_REAL,
            "_mesh_dim_names": MemberType.USE_REAL,
            "_thread_id": MemberType.USE_REAL,
            "get_rank": MemberType.USE_REAL,
            "size": MemberType.USE_REAL,
            "get_coordinate": MemberType.USE_REAL,
            "get_local_rank": MemberType.USE_REAL,
            "__eq__": MemberType.USE_REAL,
            "__ne__": MemberType.USE_REAL,
            "ndim": MemberType.USE_REAL,
            "shape": MemberType.USE_REAL,
            "mesh_dim_names": MemberType.USE_REAL,
            "get_group": MemberType.INLINED,
            "_pg_registry": MemberType.INLINED,
            "_hash": MemberType.USE_REAL,
            "_create_flatten_mesh": MemberType.USE_REAL,
            "_coordinate_on_dim": MemberType.USE_REAL,
            "_dim_group_names": MemberType.USE_REAL,
            "_flatten_mapping": MemberType.USE_REAL,
            "_rank_map": MemberType.USE_REAL,
            "_root_mesh": MemberType.USE_REAL,
            "device_type": MemberType.USE_REAL,
            "mesh": MemberType.USE_REAL,
            "_flatten": MemberType.INLINED,
            "_unflatten": MemberType.USE_REAL,
            "_is_current_rank_part_of_mesh": MemberType.USE_REAL,
            "_sym_get_coordinate": MemberType.USE_REAL,
            "_get_mesh_dim_by_name": MemberType.USE_REAL,
            "_get_root_mesh": MemberType.INLINED,
            "__getitem__": MemberType.INLINED,
            "_get_slice_mesh_layout": MemberType.INLINED,
            "_create_sub_mesh": MemberType.INLINED,
        },
    )
<|endoftext|>import weakref
from typing import cast

import torch.nn as nn


class _State:
    pass


_module_state_mapping: weakref.WeakKeyDictionary[
    nn.Module, weakref.ReferenceType[_State]
] = weakref.WeakKeyDictionary()


def _insert_module_state(module: nn.Module, state: _State) -> None:
    global _module_state_mapping
    if module in _module_state_mapping:
        raise AssertionError(f"Inserting {module} more than once.")
    _module_state_mapping[module] = weakref.ref(state)


def _get_module_state(module: nn.Module) -> _State | None:
    """
    Return the ``_State`` in ``model``.

    Given a ``module``, this API finds out if the module is also a ``_State``
    instance or if the module is managed by a composable API. If the module
    is also a ``_State``, ``module`` will be casted to ``_State` and returned.
    If it is managed by a composable API, the corresponding ``_State`` will
    be returned.
    """
    global _module_state_mapping
    if isinstance(module, _State):
        return cast(_State, module)
    else:
        # https://github.com/pytorch/pytorch/issues/107054
        if module in _module_state_mapping:
            state_ref = _module_state_mapping[module]
            state = state_ref()
            if state is None:
                raise AssertionError("State has already been garbage collected")
            return state
        else:
            return None
<|endoftext|># mypy: allow-untyped-defs
from __future__ import annotations

import abc
import ctypes
import importlib.util
import os
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from torch.distributed.distributed_c10d import ProcessGroup


# Keeps preloaded shared libraries resident for the process lifetime.
_NCCL_EP_KEEPALIVE: list[ctypes.CDLL] = []


def _find_pkg_dir(name: str) -> str | None:
    # Locate an installed package's directory without importing it. find_spec
    # raises (rather than returning None) when a dotted name's parent namespace
    # is absent, so guard that.
    try:
        spec = importlib.util.find_spec(name)
    except ModuleNotFoundError:
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return spec.submodule_search_locations[0]


def _prepare_nccl4py() -> None:
    # Dynamic (USE_SYSTEM_NCCL=ON / wheel) build only: the extension NEEDED-links
    # libnccl_ep.so, which the nccl4py wheel provides at runtime. Point nccl-ep's
    # JIT at nccl4py's EP headers (NCCL_EP_HOME) and the nvidia.nccl wheel's NCCL
    # headers (NCCL_HOME), and make libnccl_ep resolvable, before importing the
    # extension (its libnccl dependency is already loaded by torch). The static
    # build bakes all of this in and never reaches here.
    nccl_pkg = _find_pkg_dir("nccl")
    if nccl_pkg is None:
        raise ImportError(
            "TokenSwitchNCCL needs the 'nccl4py' package for this build's "
            "libnccl_ep.so and runtime JIT headers. Install it with "
            "`pip install nccl4py==0.3.1`."
        )
    ep_dir = os.path.join(nccl_pkg, "ep")
    if "NCCL_EP_HOME" not in os.environ:
        ep_headers = os.path.join(ep_dir, "include", "nccl_ep")
        if not os.path.isdir(ep_headers):
            raise ImportError(
                f"nccl4py at {nccl_pkg} is missing the EP JIT headers expected "
                f"at {ep_headers}; reinstall nccl4py or set NCCL_EP_HOME."
            )
        os.environ["NCCL_EP_HOME"] = ep_dir

    # Point the JIT at NCCL headers. libnccl.so itself needs no preload here: a
    # USE_SYSTEM_NCCL=ON torch (the only build that reaches this path) already
    # dynamically loaded the system libnccl, so libnccl_ep.so's NEEDED
    # libnccl.so.2 resolves to that already-loaded copy.
    nvidia_nccl = _find_pkg_dir("nvidia.nccl")
    if nvidia_nccl is not None and os.path.isdir(os.path.join(nvidia_nccl, "include")):
        os.environ.setdefault("NCCL_HOME", nvidia_nccl)

    # nccl4py ships libnccl_ep unversioned (libnccl_ep.so) while the extension's
