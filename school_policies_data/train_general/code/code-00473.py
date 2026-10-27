            idx = i
            break

    if idx is None:
        if dim is None:
            dim = 0
        raise TypeError(f"tensor does not contain dimension {dim}")

    # Calculate split indices
    indices = []
    total_size = 0
    unbound = []

    for i, size_dim in enumerate(sizes):
        if size_dim.is_bound:
            indices.append(size_dim.size)
            total_size += indices[-1]
        else:
            indices.append(0)
            unbound.append(i)

    if self_info.tensor is None:
        raise AssertionError("Cannot get tensor size on None tensor")
    tensor_size = self_info.tensor.size(idx)

    # Handle unbound dimensions
    if unbound:
        if total_size > tensor_size:
            raise TypeError(
                f"sizes of target dimensions add up to more ({total_size}) than source dim ({tensor_size})"
            )
        remaining_size = tensor_size - total_size
        chunk_size = (remaining_size + len(unbound) - 1) // len(unbound)
        for u in unbound:
            sz = min(chunk_size, remaining_size)
            sizes[u].size = sz
            indices[u] = sz
            remaining_size -= sz
    elif tensor_size != total_size:
        raise TypeError(
            f"sum of sizes of target dimensions ({total_size}) do not match the source dim ({tensor_size})"
        )

    # Perform the split
    result_tensors = self_info.tensor.split_with_sizes(indices, idx)

    # Create result with new levels
    result = []
    new_levels = list(self_info.levels)

    for i, (result_tensor, size_dim) in enumerate(zip(result_tensors, sizes)):
        new_levels[idx] = DimEntry(size_dim)
        result.append(
            Tensor.from_positional(
                result_tensor, list(new_levels), self_info.has_device
            )
        )

    return tuple(result)


def cat(tensors: Any, dim: Any, new_dim: Any) -> _Tensor:
    n = dims(1)  # Get single Dim instead of tuple
    return stack(tensors, n, dim).index([n, dim], new_dim)  # type: ignore[list-item]


class DotPart:
    """
    Helper class for organizing dimensions in dot products.
    """

    def __init__(self) -> None:
        self.dims: list[DimEntry] = []
        self.total_size = 1

    def append(self, dim_entry: Any) -> None:
        """Add a dimension entry to this part."""
        self.dims.append(dim_entry)
        if not dim_entry.is_positional():
            self.total_size *= dim_entry.dim().size


def dot_prepare(parts: list[DotPart], tensor_info: TensorInfo) -> torch.Tensor:
    """
    Prepare tensor for dot product by matching levels and reshaping.
    """
    new_levels = []
    needs_reshape = False

    for part in parts:
        if len(part.dims) != 1:
            needs_reshape = True
        new_levels.extend(part.dims)

    if tensor_info.tensor is None:
        raise RuntimeError("Cannot perform dot product on None tensor")
    result = _match_levels(tensor_info.tensor, tensor_info.levels, new_levels)

    if not needs_reshape:
        return result

    # Reshape for matrix operations
    view = [part.total_size for part in parts]
    return result.reshape(view)


def dot_finish(parts: list[DotPart], result_tensor: torch.Tensor) -> Tensor:
    """
    Finish dot product by reshaping result and creating Tensor.
    """
    result_levels = []
    needs_reshape = False

    for part in parts:
        if len(part.dims) != 1:
            needs_reshape = True
        result_levels.extend(part.dims)

    if needs_reshape:
        new_size = []
        for level in result_levels:
            new_size.append(level.dim().size)
        result_tensor = result_tensor.reshape(new_size)

    tensor_result = Tensor.from_positional(result_tensor, result_levels, True)
    return tensor_result  # type: ignore[return-value]


def dot(lhs: Any, rhs: Any, sum_dims: Any) -> _Tensor | torch.Tensor:
    """
    Perform dot product between two tensors along specified dimensions.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side tensor
        sum_dims: Dimensions to sum over (contract)

    Returns:
        Result of dot product
    """
    # Get tensor info
    lhs_info = TensorInfo.create(lhs, ensure_batched=False, ensure_present=False)
    rhs_info = TensorInfo.create(rhs, ensure_batched=False, ensure_present=False)

    if not (lhs_info and rhs_info):
        # Fall back to regular operations
        return torch.matmul(lhs, rhs)

    if lhs_info.tensor is None or rhs_info.tensor is None:
        raise AssertionError("Cannot perform dot product on None tensors")

    lhs_strides = lhs_info.tensor.stride()
    rhs_strides = rhs_info.tensor.stride()

    # Create dot parts for different dimension categories
    lro_dims = DotPart()  # Left-right-output (batch dims)
    lo_dims = DotPart()  # Left-output only
    ro_dims = DotPart()  # Right-output only
    lr_dims = DotPart()  # Left-right (contracted dims)

    def insert_dim(d: Any, lhs_idx: Any, rhs_idx: Any) -> None:
        """Insert dimension into appropriate part based on stride pattern."""
        reduced = d in sum_dims
        lhs_stride = lhs_strides[lhs_idx] if lhs_idx is not None else 0
        rhs_stride = rhs_strides[rhs_idx] if rhs_idx is not None else 0

        if reduced:
            lr_dims.append(d)
        else:
            if (lhs_stride == 0) == (rhs_stride == 0):
                lro_dims.append(d)  # Both have or both lack this dim
            elif lhs_stride != 0:
                lo_dims.append(d)  # Only lhs has this dim
            else:
                ro_dims.append(d)  # Only rhs has this dim

    # Track which rhs dimensions we've seen
    rhs_seen = [False] * len(rhs_info.levels)

    # Process lhs dimensions
    for i, lhs_level in enumerate(lhs_info.levels):
        rhs_idx = None
        for j, rhs_level in enumerate(rhs_info.levels):
            if lhs_level == rhs_level:
                rhs_idx = j
                rhs_seen[j] = True
                break

        insert_dim(lhs_level, i, rhs_idx)

    # Process remaining rhs dimensions
    for i, rhs_level in enumerate(rhs_info.levels):
        if not rhs_seen[i]:
            insert_dim(rhs_level, None, i)

    # Validate sum dimensions exist
    if len(lr_dims.dims) != len(sum_dims):
        for d in sum_dims:
            if d not in lhs_info.levels and d not in rhs_info.levels:
                raise ValueError(f"summing over non-existent dimension {d}")

    # Prepare tensors and perform matrix multiplication
    if len(lro_dims.dims) != 0:
        # Batched matrix multiply
        lhs_tensor = dot_prepare([lro_dims, lo_dims, lr_dims], lhs_info)
        rhs_tensor = dot_prepare([lro_dims, lr_dims, ro_dims], rhs_info)
        result = torch.bmm(lhs_tensor, rhs_tensor)
        return dot_finish([lro_dims, lo_dims, ro_dims], result)
    else:
        # Regular matrix multiply
        lhs_tensor = dot_prepare([lo_dims, lr_dims], lhs_info)
        rhs_tensor = dot_prepare([lr_dims, ro_dims], rhs_info)
        result = torch.mm(lhs_tensor, rhs_tensor)
        return dot_finish([lo_dims, ro_dims], result)


from functorch.dim._wrap import _wrap
from functorch.dim.wrap_type import wrap_type


wrap_type(_Tensor, torch.Tensor, _Tensor.__torch_function__)
del _Tensor.ndim


def index(self: Any, positions: Any, dims: Any) -> _Tensor:
    """
    Index a regular tensor by binding specified positions to dims.

    This converts a regular tensor to a first-class tensor by binding
    the specified positional dimensions to Dim objects.

    Args:
        positions: Tuple of dimension positions to bind
        dims: Dim objects or tuple of Dim objects to bind to

    Returns:
        First-class tensor with specified dimensions bound
    """
    # If this is already a first-class tensor (_Tensor), call its index method directly
    if isinstance(self, _Tensor):
        return _Tensor.index(self, positions, dims)

    # Convert regular tensor to first-class tensor
    info = TensorInfo.create(self, ensure_batched=False, ensure_present=False)

    # Create the first-class tensor
    if info.tensor is None:
        raise AssertionError("Cannot index None tensor")
    result = Tensor.from_positional(info.tensor, info.levels, info.has_device)

    # Now call the index method on the first-class tensor
    # Cast result to _Tensor for the method call
    return _Tensor.index(result, positions, dims)  # type: ignore[arg-type]


def _def(name: str, *args: Any, **kwargs: Any) -> None:
    orig = getattr(torch.Tensor, name)
    setattr(_Tensor, name, _wrap(orig, *args, **kwargs))


_def("mean")
_def("sum")
_def("all")
_def("amax")
_def("amin")
_def("aminmax")
_def("any")
_def("count_nonzero")
_def("logsumexp")
_def("nanmean")
_def("nansum")
_def("prod")
_def("std", keepdim_offset=2)
_def("var", keepdim_offset=2)
_def("max", single_dim=True)
_def("min", single_dim=True)
_def("argmax", single_dim=True)
_def("argmin", single_dim=True)
_def("kthvalue", single_dim=True)
_def("median", single_dim=True)
_def("nanmedian", single_dim=True)
_def("mode", single_dim=True)
_def("sort", reduce=False)
_def("argsort", reduce=False)
_def("unbind", single_dim=True)
_def("chunk", dim_offset=1, reduce=False)
_def("cummax", single_dim=True, reduce=False)
_def("cummin", single_dim=True, reduce=False)
_def("cumprod", single_dim=True, reduce=False)
_def("cumprod_", single_dim=True, reduce=False)
_def("cumsum", single_dim=True, reduce=False)
_def("cumsum_", single_dim=True, reduce=False)
_def("logcumsumexp", single_dim=True, reduce=False)
_def("renorm", dim_offset=1, single_dim=True, reduce=False)
_def("softmax", single_dim=True, reduce=False)
softmax = _wrap(torch.nn.functional.softmax, single_dim=True, reduce=False)

# stuff to handle in the future, because they require special
# binding logic for dims
# cross
# diag_embed
# diagonal
# diagonal_scatter
# diff
# nanquantile
# quantile
# roll
# rot90
# topk (new dimes on output)
# should these all be subsumed by inplace indexing?
# index_add_
# index_add
# index_copy
# index_copy_
# index_fill
# index_fill_
# index_select
# scatter
# scatter_
# scatter_add
# scatter_add_
# scatter_reduce
<|endoftext|>"""
Python implementation of function wrapping functionality for functorch.dim.
"""

from __future__ import annotations

import functools
from typing import Any, TYPE_CHECKING

import torch
from torch.utils._pytree import tree_map

from ._dim_entry import DimEntry
from ._enable_all_layers import EnableAllLayers
from ._tensor_info import TensorInfo


if TYPE_CHECKING:
    from collections.abc import Callable


def handle_from_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Handle tensor conversion for torch function integration."""
    return tensor


class WrappedOperator:
    """
    This class wraps PyTorch operations to support first-class dimensions.
    """

    def __init__(
        self, orig: Callable, wrapper_implementation: Callable, dim_name: str = "dim"
    ):
        self.orig = orig
        self.wrapper_implementation = wrapper_implementation
        self.name = getattr(orig, "__name__", "")
        self.doc = getattr(orig, "__doc__", None)
        self.dim_name = dim_name

        self.is_pointwise = False
        self.dim_offset = 0
        self.keepdim_offset = 1
        self.single_dim = False
        self.reduce = True

        # Update docstring if we have a dim_name
        if self.doc and self.dim_name:
            self.doc = f"{self.doc}\nArgument '{self.dim_name}' can be either an integer or a torchdim.Dim object.\n"

    def function(self) -> Callable:
        """Create a wrapped function that calls our wrapper implementation."""

        def wrapped_func(*args: Any, **kwargs: Any) -> Any:
            return self.wrapper_implementation(self, *args, **kwargs)

        # Copy metadata using functools.update_wrapper for just __name__ and __doc__
        functools.update_wrapper(
            wrapped_func, self.orig, assigned=("__name__",), updated=()
        )
        wrapped_func.__doc__ = self.doc

        return wrapped_func


def _wrap_dim(dim: Any, ndim: int, keepdim: bool = False) -> DimEntry:
    """Convert single dimension specification to DimEntry object."""
    from . import Dim

    if isinstance(dim, Dim):
        if keepdim:
            raise ValueError("cannot preserve first-class dimensions with keepdim=True")
        return DimEntry(dim)
    elif isinstance(dim, int):
        i = dim
        while i >= 0:
            i -= ndim
        return DimEntry(i)
    else:
        return DimEntry()


def _wrap_dims(dim: Any, ndim: int, keepdim: bool = False) -> list[DimEntry]:
    """Convert dimension specification to list of DimEntry objects."""
    de = _wrap_dim(dim, ndim, keepdim)
    result = []
    if not de.is_none():
        result.append(de)
    else:
        for d in dim:
            result.append(_wrap_dim(d, ndim, keepdim))
    return result


def patched_dim_method(wrapper: WrappedOperator, *args: Any, **kwargs: Any) -> Any:
    """
    This is the core method that handles dimension-aware operations.
    """
    if not args:
        raise ValueError("Expected at least one argument (self)")

    # Get dimension argument
    dim_arg = kwargs.get(wrapper.dim_name)
    if dim_arg is None and wrapper.dim_offset < len(args):
        # Try to get dim from positional args (accounting for self at index 0)
        dim_idx = wrapper.dim_offset + 1
        if dim_idx < len(args):
            dim_arg = args[dim_idx]

    # If no dimension argument provided, fall back to standard functorch handling
    if dim_arg is None:
        info = TensorInfo.create(args[0], ensure_batched=True, ensure_present=False)
        if not info:
            return wrapper.orig(*args, **kwargs)

        with EnableAllLayers(info.levels) as guard:
            if info.batchedtensor is None:
                raise AssertionError("Expected batchedtensor to be non-None")
            guard.inplace_update_layers(info.batchedtensor, info.levels)
            new_args = list(args)
            new_args[0] = handle_from_tensor(info.batchedtensor)
            result = wrapper.orig(*new_args, **kwargs)
            return guard.from_batched(result, info.has_device)

    # Handle dimension-aware operation
    info = TensorInfo.create(args[0])
    if not info:
        return wrapper.orig(*args, **kwargs)

    # Check for keepdim parameter
    keepdim = False
    if wrapper.reduce:
        keepdim_arg = kwargs.get("keepdim")
        if keepdim_arg is None and wrapper.keepdim_offset < len(args):
            keepdim_idx = wrapper.keepdim_offset + 1
            if keepdim_idx < len(args):
                keepdim_arg = args[keepdim_idx]
        if keepdim_arg is not None:
            keepdim = bool(keepdim_arg)

    # Wrap dimensions
    ndim = info.ndim()
    dims = _wrap_dims(dim_arg, ndim, keepdim)

    # Convert dimensions to indices and validate
    dim_indices: list[int] = []
    seen = [False] * len(info.levels)

    for d in dims:
        midx = None
        for i, level in enumerate(info.levels):
            if level == d:
                midx = i
                break

        if midx is None:
            # Try to match by position/name more flexibly
            for i, level in enumerate(info.levels):
                if hasattr(level, "matches") and level.matches(d):
                    midx = i
                    break

            if midx is None:
                level_strs = [str(level) for level in info.levels]
                raise ValueError(
                    f"Tensor with dimensions {level_strs} does not contain {d}"
                )

        seen[midx] = True
        dim_indices.append(midx)

    # Determine new levels after reduction
    new_levels = []
    if wrapper.reduce and not keepdim:
        for i, level in enumerate(info.levels):
            if not seen[i]:
                new_levels.append(level)
    else:
        new_levels = info.levels[:]

    # Create dimension indices for the original function
    if len(dim_indices) == 1:
        py_indices: Any = dim_indices[0]
    else:
        py_indices = tuple(dim_indices)

    # Update arguments
    new_args = list(args)
    new_kwargs = kwargs.copy()
    if info.tensor is None:
        raise AssertionError("Expected tensor to be non-None")
    new_args[0] = handle_from_tensor(info.tensor)

    # Update dimension argument
    if wrapper.dim_name in new_kwargs:
        new_kwargs[wrapper.dim_name] = py_indices
    else:
        dim_idx = wrapper.dim_offset + 1
        if dim_idx < len(new_args):
            new_args = list(new_args)
            new_args[dim_idx] = py_indices

    # Call original function
    result = wrapper.orig(*new_args, **new_kwargs)

    # Wrap results
    def wrap_result(obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            from . import Tensor

            return Tensor.from_positional(obj, new_levels, info.has_device)
        return obj

    return tree_map(wrap_result, result)


def _wrap(
    orig: Callable,
    dim_offset: int | None = None,
    keepdim_offset: int | None = None,
    dim_name: str | None = None,
    single_dim: bool | None = None,
    reduce: bool | None = None,
) -> Callable:
    """
    Wrap a PyTorch function to support first-class dimensions.

    Args:
        orig: Original function to wrap
        dim_offset: Offset for dimension argument (default: 0)
        keepdim_offset: Offset for keepdim argument (default: 1)
        dim_name: Name of dimension parameter (default: "dim")
        single_dim: Whether function takes single dimension (default: False)
        reduce: Whether function reduces dimensions (default: True)
    """
    dim_name = dim_name or "dim"

    wrapper = WrappedOperator(orig, patched_dim_method, dim_name)

    if dim_offset is not None:
        wrapper.dim_offset = dim_offset
    if keepdim_offset is not None:
        wrapper.keepdim_offset = keepdim_offset
    if single_dim is not None:
        wrapper.single_dim = single_dim
    if reduce is not None:
        wrapper.reduce = reduce

    return wrapper.function()


def call_torch_function(
    wrapper: WrappedOperator,
    func: Callable,
    types: tuple,
    args: tuple = (),
    kwargs: dict | None = None,
) -> Any:
    """
    Handle __torch_function__ calls for wrapped operators.
    """
    if kwargs is None:
        kwargs = {}

    # Import here to avoid circular imports
    from . import _Tensor

    # Use the torch function mechanism from _Tensor
    return _Tensor.__torch_function__(func, types, args, kwargs)
<|endoftext|>from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Sequence

    from . import Dim

import torch  # noqa: TC002


# NB: The old code represented dimension was from as negative number, so we
# follow this convention even though it shouldn't be necessary now
class DimEntry:
    # The dimension this is from the rhs, or a FCD
    data: Dim | int

    def __init__(self, data: Dim | int | None = None) -> None:
        from . import Dim

        if type(data) is int:
            if data >= 0:
                raise AssertionError(f"Expected negative int, got {data}")
        elif data is None:
            data = 0
        else:
            if not isinstance(data, Dim):
                raise AssertionError(f"Expected Dim, got {type(data)}")
        self.data = data

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DimEntry):
            return False
        # Use 'is' for Dim objects to avoid triggering __torch_function__
        # Use '==' only for positional (int) comparisons
        if self.is_positional() and other.is_positional():
            # Both are positional (ints)
            return self.data == other.data
        elif not self.is_positional() and not other.is_positional():
            # Both are Dim objects - use 'is' to avoid __eq__
            return self.data is other.data
        else:
            # One is positional, one is Dim - they can't be equal
            return False

    def is_positional(self) -> bool:
        return type(self.data) is int and self.data < 0

    def is_none(self) -> bool:
        # Use isinstance to check for Dim objects, avoid triggering __torch_function__
        from . import Dim

        if isinstance(self.data, Dim):
            # This is a Dim object, it can't be "none" (which is represented by 0)
            return False
        else:
            # This is an int or other type
            return self.data == 0

    def position(self) -> int:
        if not isinstance(self.data, int):
            raise AssertionError(f"Expected int, got {type(self.data)}")
        return self.data

    def dim(self) -> Dim:
        if isinstance(self.data, int):
            raise AssertionError("Expected Dim, got int")
        return self.data

    def __repr__(self) -> str:
        return repr(self.data)


def ndim_of_levels(levels: Sequence[DimEntry]) -> int:
    r = 0
    for l in levels:
        if l.is_positional():
            r += 1
    return r


def _match_levels(
    tensor: torch.Tensor,
    from_levels: list[DimEntry],
    to_levels: list[DimEntry],
    drop_levels: bool = False,
) -> torch.Tensor:
    """
    Reshape a tensor to match target levels using as_strided.

    Args:
        tensor: Input tensor to reshape
        from_levels: Current levels of the tensor
        to_levels: Target levels to match
        drop_levels: If True, missing dimensions are assumed to have stride 0

    Returns:
        Reshaped tensor
    """
    if from_levels == to_levels:
        return tensor

    sizes = tensor.size()
    strides = tensor.stride()

    if not drop_levels:
        if len(from_levels) > len(to_levels):
            raise AssertionError("Cannot expand dimensions without drop_levels")

    new_sizes = []
    new_strides = []

    for level in to_levels:
        # Find index of this level in from_levels
        try:
            idx = from_levels.index(level)
        except ValueError:
            # Level not found in from_levels
            if level.is_positional():
                new_sizes.append(1)
            else:
                new_sizes.append(level.dim().size)
            new_strides.append(0)
        else:
            new_sizes.append(sizes[idx])
            new_strides.append(strides[idx])

    return tensor.as_strided(new_sizes, new_strides, tensor.storage_offset())
<|endoftext|>from __future__ import annotations

from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Sequence

import torch  # noqa: TC002

from ._dim_entry import _match_levels, DimEntry, ndim_of_levels


def _wrap_dim(arg: Any, orig_ndim: int, allow_none: bool = True) -> DimEntry:
    """
    Convert various dimension representations to DimEntry.

    Args:
        arg: The argument to convert (Dim, int, or other)
        orig_ndim: Original number of dimensions
        allow_none: Whether to allow None values

    Returns:
        DimEntry representation of the dimension
    """
    from . import Dim

    if arg is None and allow_none:
        return DimEntry()  # None entry
    elif isinstance(arg, Dim):
        return DimEntry(arg)
    elif isinstance(arg, int):
        if arg < 0:
            pos = arg
        else:
            pos = arg - orig_ndim
        return DimEntry(pos)
    else:
        return DimEntry()


def order(
    tensor_or_dim: torch.Tensor | Any, *dims: Any | Sequence[Any]
) -> torch.Tensor:
    """
    Reorder the dimensions of a tensor or create a tensor from a dimension.

    It allows reordering tensor dimensions using first-class dimensions and
    positional indices.

    Args:
        tensor_or_dim: Input tensor with first-class dimensions, or a Dim object
        *dims: Dimensions or sequences of dimensions specifying the new order

    Returns:
        Tensor with reordered dimensions

    Examples:
        >>> import torch
        >>> from functorch.dim import dims
        >>> batch, channel, height, width = dims(4)
        >>> x = torch.randn(2, 3, 4, 5)[batch, channel, height, width]
        >>> # Reorder to [height, width, batch, channel]
        >>> y = order(x, height, width, batch, channel)
    """
    from . import Dim, DimList, Tensor

    # Handle first argument - tensor or dimension
    if isinstance(tensor_or_dim, Tensor):
        # First-class tensor
        orig_levels = tensor_or_dim._levels[:]
        data = tensor_or_dim._tensor
        has_device = tensor_or_dim._has_device
    elif isinstance(tensor_or_dim, Dim):
        # Single dimension - create range tensor
        orig_levels = [DimEntry(tensor_or_dim)]
        data = tensor_or_dim._get_range()
        has_device = False
    else:
        raise ValueError("First argument must be a Tensor or Dim object")

    flat_positional_dims = []
    to_flatten = []  # List of (start_index, length) pairs for flattening
    levels = orig_levels[:]

    orig_ndim = ndim_of_levels(levels)

    def append_dim(d: DimEntry) -> None:
        """Add a dimension to the reordering, removing it from available levels."""
        try:
            idx = levels.index(d)
        except ValueError:
            idx = None
        if idx is None:
            if d.is_positional():
                raise ValueError(
                    f"tensor has {orig_ndim} positional dimensions, but {d.position() + orig_ndim} specified, "
                    f"or it was specified twice"
                )
            else:
                raise ValueError(
                    f"tensor does not contain dim {d.dim()} or it was specified twice"
                )

        levels[idx] = DimEntry()
        flat_positional_dims.append(d)

    n_new_positional = 0

    # Process each dimension argument
    for arg in dims:
        entry = _wrap_dim(arg, orig_ndim, False)
        if not entry.is_none():
            append_dim(entry)
            n_new_positional += 1
        elif isinstance(arg, DimList):
            # Handle DimList
            for dim in arg._dims:
                append_dim(DimEntry(dim))
                n_new_positional += 1
        else:
            # Handle sequences of dimensions for flattening
            n_new_positional += 1
            if not hasattr(arg, "__iter__"):
                raise ValueError("expected a Dim, List[Dim], or Sequence[Dim]")

            # Convert to list to get length
            seq = list(arg)
            to_flatten.append((len(flat_positional_dims), len(seq)))

            for item in seq:
                entry = _wrap_dim(item, orig_ndim, False)
                if entry.is_none():
                    raise ValueError("expected a Dim or int")
                append_dim(entry)

    # Build new level ordering
    insert_point = -1
    new_levels: list[DimEntry] = []

    # Add remaining (non-reordered) levels, finding insertion point for new dimensions
    for level in levels:
        if level.is_none():
            continue
        if level.is_positional():
            if insert_point == -1:
                insert_point = len(new_levels)
                new_levels.extend(flat_positional_dims)
        new_levels.append(level)

    # If no positional dimensions found, append new dims at the end
    if insert_point == -1:
        insert_point = len(new_levels)
        new_levels.extend(flat_positional_dims)

    # Match tensor to new level structure
    if data is None:
        raise AssertionError("Cannot reorder None tensor")
    ndata = _match_levels(data, orig_levels, new_levels)

    # Handle dimension flattening if requested
    if to_flatten:
        # Now build the reshape target
        view_shape = []
        sizes = ndata.size()

        # Add dimensions before the reordered ones
        for i in range(insert_point):
            view_shape.append(sizes[i])

        # Process flattening groups
        i = 0
        for start_idx, length in to_flatten:
            # Add individual dims before this flattening group
            while i < start_idx:
                view_shape.append(sizes[insert_point + i])
                i += 1

            # Flatten the group
            new_size = 1
            for j in range(length):
                new_size *= sizes[insert_point + i + j]
            view_shape.append(new_size)
            i += length

        # Add remaining individual dims
        while i < len(flat_positional_dims):
            view_shape.append(sizes[insert_point + i])
            i += 1

        # Add dimensions after the reordered ones
        for i in range(insert_point + len(flat_positional_dims), len(levels)):
            view_shape.append(sizes[i])

        # Update levels by removing flattened dimensions
        n_to_remove = len(flat_positional_dims) - n_new_positional
        if n_to_remove > 0:
            # Remove flattened levels
            new_levels = (
                new_levels[:insert_point] + new_levels[insert_point + n_to_remove :]
            )

        ndata = ndata.reshape(view_shape)

    # Renumber positional dimensions (negative indexing from the right)
    seen = 0
    for i in range(len(new_levels) - 1, -1, -1):
        if new_levels[i].is_positional() or (
            i >= insert_point and i < insert_point + n_new_positional
        ):
            seen -= 1
            new_levels[i] = DimEntry(seen)

    result = Tensor.from_positional(ndata, new_levels, has_device)
    return result  # type: ignore[return-value]
<|endoftext|>from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from ._dim_entry import DimEntry


@dataclass
class TensorInfo:
    tensor: torch.Tensor | None
    levels: list[DimEntry]
    has_device: bool
    batchedtensor: torch.Tensor | None

    def __post_init__(self) -> None:
        from ._dim_entry import DimEntry

        if not all(isinstance(l, DimEntry) for l in self.levels):
            raise AssertionError("All levels must be DimEntry instances")

    def ndim(self) -> int:
        from ._dim_entry import ndim_of_levels

        return ndim_of_levels(self.levels)

    def __bool__(self) -> bool:
        return self.tensor is not None

    @staticmethod
    def create(
        h: Any, ensure_batched: bool = True, ensure_present: bool = True
    ) -> TensorInfo:
        from . import Dim, DimEntry, Tensor

        if Tensor.check_exact(h):
            # functorch Tensor with first-class dimensions
            return TensorInfo(
                h._get_tensor(),
                h._get_levels(),
                h._get_has_device(),
                h._get_batchtensor() if ensure_batched else None,
            )
        elif Dim.check_exact(h):
            # For Dim objects, only get range/batchtensor if needed and dimension is bound
            tensor = h._get_range() if h.is_bound else None
            batchtensor = (
                h._get_batchtensor() if ensure_batched and h.is_bound else None
            )
            return TensorInfo(
                tensor,
                [DimEntry(h)],
                False,
                batchtensor,
            )
        elif isinstance(h, torch.Tensor):
            # Plain torch tensor - create positional levels
            levels = []
            for i in range(-h.dim(), 0):
                levels.append(DimEntry(i))
            return TensorInfo(h, levels, True, h)
        else:
            if ensure_present:
                raise ValueError("expected a tensor object")
            return TensorInfo(None, [], False, None)
<|endoftext|># Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import signal
import subprocess
from collections.abc import Generator
from contextlib import contextmanager


@contextmanager
def magic_trace(
    output: str = "trace.fxt", magic_trace_cache: str = "/tmp/magic-trace"
) -> Generator[None, None, None]:
    pid = os.getpid()
    if not os.path.exists(magic_trace_cache):
        print(f"Downloading magic_trace to: {magic_trace_cache}")
        subprocess.run(
            [
                "wget",
                "-O",
                magic_trace_cache,
                "-q",
                "https://github.com/janestreet/magic-trace/releases/download/v1.0.2/magic-trace",
            ]
        )
        subprocess.run(["chmod", "+x", magic_trace_cache])
    args = [magic_trace_cache, "attach", "-pid", str(pid), "-o", output]
    p = subprocess.Popen(args, stderr=subprocess.PIPE, encoding="utf-8")
    if p.stderr is None:
        raise AssertionError("Expected stderr to be non-None")
    while True:
        x = p.stderr.readline()
        print(x)
        if "Attached" in x:
            break
    try:
        yield
    finally:
        p.send_signal(signal.SIGINT)
        r = p.wait()
        if p.stderr is not None:
            print(p.stderr.read())
            p.stderr.close()
        if r != 0:
            raise ValueError(f"magic_trace exited abnormally: {r}")
<|endoftext|>from __future__ import annotations

import collections.abc as c
import inspect
import typing as t
from weakref import ref
from weakref import WeakMethod

T = t.TypeVar("T")


class Symbol:
    """A constant symbol, nicer than ``object()``. Repeated calls return the
    same instance.

    >>> Symbol('foo') is Symbol('foo')
    True
    >>> Symbol('foo')
    foo
    """

    symbols: t.ClassVar[dict[str, Symbol]] = {}

    def __new__(cls, name: str) -> Symbol:
        if name in cls.symbols:
            return cls.symbols[name]

        obj = super().__new__(cls)
        cls.symbols[name] = obj
        return obj

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return self.name

    def __getnewargs__(self) -> tuple[t.Any, ...]:
        return (self.name,)


def make_id(obj: object) -> c.Hashable:
    """Get a stable identifier for a receiver or sender, to be used as a dict
    key or in a set.
    """
    if inspect.ismethod(obj):
        # The id of a bound method is not stable, but the id of the unbound
        # function and instance are.
        return id(obj.__func__), id(obj.__self__)

    if isinstance(obj, (str, int)):
        # Instances with the same value always compare equal and have the same
        # hash, even if the id may change.
        return obj

    # Assume other types are not hashable but will always be the same instance.
    return id(obj)


def make_ref(obj: T, callback: c.Callable[[ref[T]], None] | None = None) -> ref[T]:
    if inspect.ismethod(obj):
        return WeakMethod(obj, callback)  # type: ignore[arg-type, return-value]

    return ref(obj, callback)
<|endoftext|>from __future__ import annotations

from .base import ANY
from .base import default_namespace
from .base import NamedSignal
from .base import Namespace
from .base import Signal
from .base import signal

__all__ = [
    "ANY",
    "default_namespace",
    "NamedSignal",
    "Namespace",
    "Signal",
    "signal",
]
<|endoftext|>from __future__ import annotations

import collections.abc as c
import sys
import typing as t
import weakref
from collections import defaultdict
from contextlib import contextmanager
from functools import cached_property
from inspect import iscoroutinefunction

from ._utilities import make_id
from ._utilities import make_ref
from ._utilities import Symbol

F = t.TypeVar("F", bound=c.Callable[..., t.Any])

ANY = Symbol("ANY")
"""Symbol for "any sender"."""

ANY_ID = 0


class Signal:
    """A notification emitter.

    :param doc: The docstring for the signal.
    """

    ANY = ANY
    """An alias for the :data:`~blinker.ANY` sender symbol."""

    set_class: type[set[t.Any]] = set
    """The set class to use for tracking connected receivers and senders.
    Python's ``set`` is unordered. If receivers must be dispatched in the order
    they were connected, an ordered set implementation can be used.

    .. versionadded:: 1.7
    """

    @cached_property
    def receiver_connected(self) -> Signal:
        """Emitted at the end of each :meth:`connect` call.

        The signal sender is the signal instance, and the :meth:`connect`
        arguments are passed through: ``receiver``, ``sender``, and ``weak``.

        .. versionadded:: 1.2
        """
        return Signal(doc="Emitted after a receiver connects.")

    @cached_property
    def receiver_disconnected(self) -> Signal:
        """Emitted at the end of each :meth:`disconnect` call.

        The sender is the signal instance, and the :meth:`disconnect` arguments
        are passed through: ``receiver`` and ``sender``.

        This signal is emitted **only** when :meth:`disconnect` is called
        explicitly. This signal cannot be emitted by an automatic disconnect
        when a weakly referenced receiver or sender goes out of scope, as the
        instance is no longer be available to be used as the sender for this
        signal.

        An alternative approach is available by subscribing to
        :attr:`receiver_connected` and setting up a custom weakref cleanup
        callback on weak receivers and senders.

        .. versionadded:: 1.2
        """
        return Signal(doc="Emitted after a receiver disconnects.")

    def __init__(self, doc: str | None = None) -> None:
        if doc:
            self.__doc__ = doc

        self.receivers: dict[
            t.Any, weakref.ref[c.Callable[..., t.Any]] | c.Callable[..., t.Any]
        ] = {}
        """The map of connected receivers. Useful to quickly check if any
        receivers are connected to the signal: ``if s.receivers:``. The
        structure and data is not part of the public API, but checking its
        boolean value is.
        """

        self.is_muted: bool = False
        self._by_receiver: dict[t.Any, set[t.Any]] = defaultdict(self.set_class)
        self._by_sender: dict[t.Any, set[t.Any]] = defaultdict(self.set_class)
        self._weak_senders: dict[t.Any, weakref.ref[t.Any]] = {}

    def connect(self, receiver: F, sender: t.Any = ANY, weak: bool = True) -> F:
        """Connect ``receiver`` to be called when the signal is sent by
        ``sender``.

        :param receiver: The callable to call when :meth:`send` is called with
            the given ``sender``, passing ``sender`` as a positional argument
            along with any extra keyword arguments.
        :param sender: Any object or :data:`ANY`. ``receiver`` will only be
            called when :meth:`send` is called with this sender. If ``ANY``, the
            receiver will be called for any sender. A receiver may be connected
            to multiple senders by calling :meth:`connect` multiple times.
        :param weak: Track the receiver with a :mod:`weakref`. The receiver will
            be automatically disconnected when it is garbage collected. When
            connecting a receiver defined within a function, set to ``False``,
            otherwise it will be disconnected when the function scope ends.
        """
        receiver_id = make_id(receiver)
        sender_id = ANY_ID if sender is ANY else make_id(sender)

        if weak:
            self.receivers[receiver_id] = make_ref(
                receiver, self._make_cleanup_receiver(receiver_id)
            )
        else:
            self.receivers[receiver_id] = receiver

        self._by_sender[sender_id].add(receiver_id)
        self._by_receiver[receiver_id].add(sender_id)

        if sender is not ANY and sender_id not in self._weak_senders:
            # store a cleanup for weakref-able senders
            try:
                self._weak_senders[sender_id] = make_ref(
                    sender, self._make_cleanup_sender(sender_id)
                )
            except TypeError:
                pass

        if "receiver_connected" in self.__dict__ and self.receiver_connected.receivers:
            try:
                self.receiver_connected.send(
                    self, receiver=receiver, sender=sender, weak=weak
                )
            except TypeError:
                # TODO no explanation or test for this
                self.disconnect(receiver, sender)
                raise

        return receiver

    def connect_via(self, sender: t.Any, weak: bool = False) -> c.Callable[[F], F]:
        """Connect the decorated function to be called when the signal is sent
        by ``sender``.

        The decorated function will be called when :meth:`send` is called with
        the given ``sender``, passing ``sender`` as a positional argument along
        with any extra keyword arguments.

        :param sender: Any object or :data:`ANY`. ``receiver`` will only be
            called when :meth:`send` is called with this sender. If ``ANY``, the
            receiver will be called for any sender. A receiver may be connected
            to multiple senders by calling :meth:`connect` multiple times.
        :param weak: Track the receiver with a :mod:`weakref`. The receiver will
            be automatically disconnected when it is garbage collected. When
            connecting a receiver defined within a function, set to ``False``,
            otherwise it will be disconnected when the function scope ends.=

        .. versionadded:: 1.1
        """

        def decorator(fn: F) -> F:
            self.connect(fn, sender, weak)
            return fn

        return decorator

    @contextmanager
    def connected_to(
        self, receiver: c.Callable[..., t.Any], sender: t.Any = ANY
    ) -> c.Generator[None, None, None]:
        """A context manager that temporarily connects ``receiver`` to the
        signal while a ``with`` block executes. When the block exits, the
        receiver is disconnected. Useful for tests.

        :param receiver: The callable to call when :meth:`send` is called with
            the given ``sender``, passing ``sender`` as a positional argument
            along with any extra keyword arguments.
        :param sender: Any object or :data:`ANY`. ``receiver`` will only be
            called when :meth:`send` is called with this sender. If ``ANY``, the
            receiver will be called for any sender.

        .. versionadded:: 1.1
        """
        self.connect(receiver, sender=sender, weak=False)

        try:
            yield None
        finally:
            self.disconnect(receiver)

    @contextmanager
    def muted(self) -> c.Generator[None, None, None]:
        """A context manager that temporarily disables the signal. No receivers
        will be called if the signal is sent, until the ``with`` block exits.
        Useful for tests.
        """
        self.is_muted = True

        try:
            yield None
        finally:
            self.is_muted = False

    def send(
        self,
        sender: t.Any | None = None,
        /,
        *,
        _async_wrapper: c.Callable[
            [c.Callable[..., c.Coroutine[t.Any, t.Any, t.Any]]], c.Callable[..., t.Any]
        ]
        | None = None,
        **kwargs: t.Any,
    ) -> list[tuple[c.Callable[..., t.Any], t.Any]]:
        """Call all receivers that are connected to the given ``sender``
        or :data:`ANY`. Each receiver is called with ``sender`` as a positional
        argument along with any extra keyword arguments. Return a list of
        ``(receiver, return value)`` tuples.

        The order receivers are called is undefined, but can be influenced by
        setting :attr:`set_class`.

        If a receiver raises an exception, that exception will propagate up.
        This makes debugging straightforward, with an assumption that correctly
        implemented receivers will not raise.

        :param sender: Call receivers connected to this sender, in addition to
            those connected to :data:`ANY`.
        :param _async_wrapper: Will be called on any receivers that are async
            coroutines to turn them into sync callables. For example, could run
            the receiver with an event loop.
        :param kwargs: Extra keyword arguments to pass to each receiver.

        .. versionchanged:: 1.7
            Added the ``_async_wrapper`` argument.
        """
        if self.is_muted:
            return []

        results = []

        for receiver in self.receivers_for(sender):
            if iscoroutinefunction(receiver):
                if _async_wrapper is None:
                    raise RuntimeError("Cannot send to a coroutine function.")

                result = _async_wrapper(receiver)(sender, **kwargs)
            else:
                result = receiver(sender, **kwargs)

            results.append((receiver, result))

        return results

    async def send_async(
        self,
        sender: t.Any | None = None,
        /,
        *,
        _sync_wrapper: c.Callable[
            [c.Callable[..., t.Any]], c.Callable[..., c.Coroutine[t.Any, t.Any, t.Any]]
        ]
        | None = None,
        **kwargs: t.Any,
    ) -> list[tuple[c.Callable[..., t.Any], t.Any]]:
        """Await all receivers that are connected to the given ``sender``
        or :data:`ANY`. Each receiver is called with ``sender`` as a positional
        argument along with any extra keyword arguments. Return a list of
        ``(receiver, return value)`` tuples.

        The order receivers are called is undefined, but can be influenced by
        setting :attr:`set_class`.

        If a receiver raises an exception, that exception will propagate up.
        This makes debugging straightforward, with an assumption that correctly
        implemented receivers will not raise.

        :param sender: Call receivers connected to this sender, in addition to
            those connected to :data:`ANY`.
        :param _sync_wrapper: Will be called on any receivers that are sync
            callables to turn them into async coroutines. For example,
            could call the receiver in a thread.
        :param kwargs: Extra keyword arguments to pass to each receiver.

        .. versionadded:: 1.7
        """
        if self.is_muted:
            return []

        results = []

        for receiver in self.receivers_for(sender):
            if not iscoroutinefunction(receiver):
                if _sync_wrapper is None:
                    raise RuntimeError("Cannot send to a non-coroutine function.")

                result = await _sync_wrapper(receiver)(sender, **kwargs)
            else:
                result = await receiver(sender, **kwargs)

            results.append((receiver, result))

        return results

    def has_receivers_for(self, sender: t.Any) -> bool:
        """Check if there is at least one receiver that will be called with the
        given ``sender``. A receiver connected to :data:`ANY` will always be
        called, regardless of sender. Does not check if weakly referenced
        receivers are still live. See :meth:`receivers_for` for a stronger
        search.

        :param sender: Check for receivers connected to this sender, in addition
            to those connected to :data:`ANY`.
        """
        if not self.receivers:
            return False

        if self._by_sender[ANY_ID]:
            return True

        if sender is ANY:
            return False

        return make_id(sender) in self._by_sender

    def receivers_for(
        self, sender: t.Any
    ) -> c.Generator[c.Callable[..., t.Any], None, None]:
        """Yield each receiver to be called for ``sender``, in addition to those
        to be called for :data:`ANY`. Weakly referenced receivers that are not
        live will be disconnected and skipped.

        :param sender: Yield receivers connected to this sender, in addition
            to those connected to :data:`ANY`.
        """
        # TODO: test receivers_for(ANY)
        if not self.receivers:
            return

        sender_id = make_id(sender)

        if sender_id in self._by_sender:
            ids = self._by_sender[ANY_ID] | self._by_sender[sender_id]
        else:
            ids = self._by_sender[ANY_ID].copy()

        for receiver_id in ids:
            receiver = self.receivers.get(receiver_id)

            if receiver is None:
                continue

            if isinstance(receiver, weakref.ref):
                strong = receiver()

                if strong is None:
                    self._disconnect(receiver_id, ANY_ID)
                    continue

                yield strong
            else:
                yield receiver

    def disconnect(self, receiver: c.Callable[..., t.Any], sender: t.Any = ANY) -> None:
        """Disconnect ``receiver`` from being called when the signal is sent by
        ``sender``.

        :param receiver: A connected receiver callable.
        :param sender: Disconnect from only this sender. By default, disconnect
            from all senders.
        """
        sender_id: c.Hashable

        if sender is ANY:
            sender_id = ANY_ID
        else:
            sender_id = make_id(sender)

        receiver_id = make_id(receiver)
        self._disconnect(receiver_id, sender_id)

        if (
            "receiver_disconnected" in self.__dict__
            and self.receiver_disconnected.receivers
        ):
            self.receiver_disconnected.send(self, receiver=receiver, sender=sender)

    def _disconnect(self, receiver_id: c.Hashable, sender_id: c.Hashable) -> None:
        if sender_id == ANY_ID:
            if self._by_receiver.pop(receiver_id, None) is not None:
                for bucket in self._by_sender.values():
                    bucket.discard(receiver_id)

            self.receivers.pop(receiver_id, None)
        else:
            self._by_sender[sender_id].discard(receiver_id)
            self._by_receiver[receiver_id].discard(sender_id)

    def _make_cleanup_receiver(
        self, receiver_id: c.Hashable
    ) -> c.Callable[[weakref.ref[c.Callable[..., t.Any]]], None]:
        """Create a callback function to disconnect a weakly referenced
        receiver when it is garbage collected.
        """

        def cleanup(ref: weakref.ref[c.Callable[..., t.Any]]) -> None:
            # If the interpreter is shutting down, disconnecting can result in a
            # weird ignored exception. Don't call it in that case.
            if not sys.is_finalizing():
                self._disconnect(receiver_id, ANY_ID)

        return cleanup

    def _make_cleanup_sender(
        self, sender_id: c.Hashable
    ) -> c.Callable[[weakref.ref[t.Any]], None]:
        """Create a callback function to disconnect all receivers for a weakly
        referenced sender when it is garbage collected.
        """
        assert sender_id != ANY_ID

        def cleanup(ref: weakref.ref[t.Any]) -> None:
            self._weak_senders.pop(sender_id, None)

            for receiver_id in self._by_sender.pop(sender_id, ()):
                self._by_receiver[receiver_id].discard(sender_id)

        return cleanup

    def _cleanup_bookkeeping(self) -> None:
        """Prune unused sender/receiver bookkeeping. Not threadsafe.

        Connecting & disconnecting leaves behind a small amount of bookkeeping
        data. Typical workloads using Blinker, for example in most web apps,
        Flask, CLI scripts, etc., are not adversely affected by this
        bookkeeping.

        With a long-running process performing dynamic signal routing with high
        volume, e.g. connecting to function closures, senders are all unique
        object instances. Doing all of this over and over may cause memory usage
        to grow due to extraneous bookkeeping. (An empty ``set`` for each stale
        sender/receiver pair.)

        This method will prune that bookkeeping away, with the caveat that such
        pruning is not threadsafe. The risk is that cleanup of a fully
        disconnected receiver/sender pair occurs while another thread is
        connecting that same pair. If you are in the highly dynamic, unique
        receiver/sender situation that has lead you to this method, that failure
        mode is perhaps not a big deal for you.
        """
        for mapping in (self._by_sender, self._by_receiver):
            for ident, bucket in list(mapping.items()):
                if not bucket:
                    mapping.pop(ident, None)

    def _clear_state(self) -> None:
        """Disconnect all receivers and senders. Useful for tests."""
        self._weak_senders.clear()
        self.receivers.clear()
        self._by_sender.clear()
        self._by_receiver.clear()


class NamedSignal(Signal):
    """A named generic notification emitter. The name is not used by the signal
    itself, but matches the key in the :class:`Namespace` that it belongs to.

    :param name: The name of the signal within the namespace.
    :param doc: The docstring for the signal.
    """

    def __init__(self, name: str, doc: str | None = None) -> None:
        super().__init__(doc)

        #: The name of this signal.
        self.name: str = name

    def __repr__(self) -> str:
        base = super().__repr__()
        return f"{base[:-1]}; {self.name!r}>"  # noqa: E702


class Namespace(dict[str, NamedSignal]):
    """A dict mapping names to signals."""

    def signal(self, name: str, doc: str | None = None) -> NamedSignal:
        """Return the :class:`NamedSignal` for the given ``name``, creating it
        if required. Repeated calls with the same name return the same signal.

        :param name: The name of the signal.
        :param doc: The docstring of the signal.
        """
        if name not in self:
            self[name] = NamedSignal(name, doc)

        return self[name]


class _PNamespaceSignal(t.Protocol):
    def __call__(self, name: str, doc: str | None = None) -> NamedSignal: ...


default_namespace: Namespace = Namespace()
"""A default :class:`Namespace` for creating named signals. :func:`signal`
creates a :class:`NamedSignal` in this namespace.
"""

signal: _PNamespaceSignal = default_namespace.signal
"""Return a :class:`NamedSignal` in :data:`default_namespace` with the given
``name``, creating it if required. Repeated calls with the same name return the
same signal.
"""
<|endoftext|>from __future__ import annotations

import inspect
import sys
from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractContextManager
from types import TracebackType
from typing import TYPE_CHECKING, Any

if sys.version_info < (3, 11):
    from ._exceptions import BaseExceptionGroup

if TYPE_CHECKING:
    _Handler = Callable[[BaseExceptionGroup[Any]], Any]


class _Catcher:
    def __init__(self, handler_map: Mapping[tuple[type[BaseException], ...], _Handler]):
        self._handler_map = handler_map

    def __enter__(self) -> None:
        pass

    def __exit__(
        self,
        etype: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        if exc is not None:
            unhandled = self.handle_exception(exc)
            if unhandled is exc:
                return False
            elif unhandled is None:
                return True
            else:
                if isinstance(exc, BaseExceptionGroup):
                    try:
                        raise unhandled from exc.__cause__
                    except BaseExceptionGroup:
                        # Change __context__ to __cause__ because Python 3.11 does this
                        # too
                        unhandled.__context__ = exc.__cause__
                        raise

                raise unhandled from exc

        return False

    def handle_exception(self, exc: BaseException) -> BaseException | None:
        excgroup: BaseExceptionGroup | None
        if isinstance(exc, BaseExceptionGroup):
            excgroup = exc
        else:
            excgroup = BaseExceptionGroup("", [exc])

        new_exceptions: list[BaseException] = []
        for exc_types, handler in self._handler_map.items():
            matched, excgroup = excgroup.split(exc_types)
            if matched:
                try:
                    try:
                        raise matched
                    except BaseExceptionGroup:
                        result = handler(matched)
                except BaseExceptionGroup as new_exc:
                    if new_exc is matched:
                        new_exceptions.append(new_exc)
                    else:
                        new_exceptions.extend(new_exc.exceptions)
                except BaseException as new_exc:
                    new_exceptions.append(new_exc)
                else:
                    if inspect.iscoroutine(result):
                        raise TypeError(
                            f"Error trying to handle {matched!r} with {handler!r}. "
                            "Exception handler must be a sync function."
                        ) from exc

            if not excgroup:
                break

        if new_exceptions:
            if len(new_exceptions) == 1:
                return new_exceptions[0]

            return BaseExceptionGroup("", new_exceptions)
        elif (
            excgroup and len(excgroup.exceptions) == 1 and excgroup.exceptions[0] is exc
        ):
            return exc
        else:
            return excgroup


def catch(
    __handlers: Mapping[type[BaseException] | Iterable[type[BaseException]], _Handler],
) -> AbstractContextManager[None]:
    if not isinstance(__handlers, Mapping):
        raise TypeError("the argument must be a mapping")

    handler_map: dict[
        tuple[type[BaseException], ...], Callable[[BaseExceptionGroup]]
    ] = {}
    for type_or_iterable, handler in __handlers.items():
        iterable: tuple[type[BaseException]]
        if isinstance(type_or_iterable, type) and issubclass(
            type_or_iterable, BaseException
        ):
            iterable = (type_or_iterable,)
        elif isinstance(type_or_iterable, Iterable):
            iterable = tuple(type_or_iterable)
        else:
            raise TypeError(
                "each key must be either an exception classes or an iterable thereof"
            )

        if not callable(handler):
            raise TypeError("handlers must be callable")

        for exc_type in iterable:
            if not isinstance(exc_type, type) or not issubclass(
                exc_type, BaseException
            ):
                raise TypeError(
                    "each key must be either an exception classes or an iterable "
                    "thereof"
                )

            if issubclass(exc_type, BaseExceptionGroup):
                raise TypeError(
                    "catching ExceptionGroup with catch() is not allowed. "
                    "Use except instead."
                )

        handler_map[iterable] = handler

    return _Catcher(handler_map)
<|endoftext|>__all__ = [
    "BaseExceptionGroup",
    "ExceptionGroup",
    "catch",
    "format_exception",
    "format_exception_only",
    "print_exception",
    "print_exc",
    "suppress",
]

import os
import sys

from ._catch import catch
from ._version import version as __version__  # noqa: F401

if sys.version_info < (3, 11):
    from ._exceptions import BaseExceptionGroup, ExceptionGroup
    from ._formatting import (
        format_exception,
        format_exception_only,
        print_exc,
        print_exception,
    )

    if os.getenv("EXCEPTIONGROUP_NO_PATCH") != "1":
        from . import _formatting  # noqa: F401

    BaseExceptionGroup.__module__ = __name__
    ExceptionGroup.__module__ = __name__
else:
    from traceback import (
        format_exception,
        format_exception_only,
        print_exc,
        print_exception,
    )

    BaseExceptionGroup = BaseExceptionGroup
    ExceptionGroup = ExceptionGroup

if sys.version_info < (3, 12, 1):
    from ._suppress import suppress
else:
    from contextlib import suppress
<|endoftext|>from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from functools import partial
from inspect import getmro, isclass
from typing import TYPE_CHECKING, Generic, Type, TypeVar, cast, overload

if sys.version_info < (3, 13):
    from typing_extensions import TypeVar

_BaseExceptionT_co = TypeVar(
    "_BaseExceptionT_co", bound=BaseException, covariant=True, default=BaseException
)
_BaseExceptionT = TypeVar("_BaseExceptionT", bound=BaseException)
_ExceptionT_co = TypeVar(
    "_ExceptionT_co", bound=Exception, covariant=True, default=Exception
)
_ExceptionT = TypeVar("_ExceptionT", bound=Exception)
# using typing.Self would require a typing_extensions dependency on py<3.11
_ExceptionGroupSelf = TypeVar("_ExceptionGroupSelf", bound="ExceptionGroup")
_BaseExceptionGroupSelf = TypeVar("_BaseExceptionGroupSelf", bound="BaseExceptionGroup")


def check_direct_subclass(
    exc: BaseException, parents: tuple[type[BaseException]]
) -> bool:
    for cls in getmro(exc.__class__)[:-1]:
        if cls in parents:
            return True

    return False


def get_condition_filter(
    condition: type[_BaseExceptionT]
    | tuple[type[_BaseExceptionT], ...]
    | Callable[[_BaseExceptionT_co], bool],
) -> Callable[[_BaseExceptionT_co], bool]:
    if isclass(condition) and issubclass(
        cast(Type[BaseException], condition), BaseException
    ):
        return partial(check_direct_subclass, parents=(condition,))
    elif isinstance(condition, tuple):
        if all(isclass(x) and issubclass(x, BaseException) for x in condition):
            return partial(check_direct_subclass, parents=condition)
    elif callable(condition):
        return cast("Callable[[BaseException], bool]", condition)

    raise TypeError("expected a function, exception type or tuple of exception types")


def _derive_and_copy_attributes(self, excs):
    eg = self.derive(excs)
    eg.__cause__ = self.__cause__
    eg.__context__ = self.__context__
    eg.__traceback__ = self.__traceback__
    if hasattr(self, "__notes__"):
        # Create a new list so that add_note() only affects one exceptiongroup
        eg.__notes__ = list(self.__notes__)
    return eg


class BaseExceptionGroup(BaseException, Generic[_BaseExceptionT_co]):
    """A combination of multiple unrelated exceptions."""

    def __new__(
        cls: type[_BaseExceptionGroupSelf],
        __message: str,
        __exceptions: Sequence[_BaseExceptionT_co],
    ) -> _BaseExceptionGroupSelf:
        if not isinstance(__message, str):
            raise TypeError(f"argument 1 must be str, not {type(__message)}")
        if not isinstance(__exceptions, Sequence):
            raise TypeError("second argument (exceptions) must be a sequence")
        if not __exceptions:
            raise ValueError(
                "second argument (exceptions) must be a non-empty sequence"
            )

        for i, exc in enumerate(__exceptions):
            if not isinstance(exc, BaseException):
                raise ValueError(
                    f"Item {i} of second argument (exceptions) is not an exception"
                )

        if cls is BaseExceptionGroup:
            if all(isinstance(exc, Exception) for exc in __exceptions):
                cls = ExceptionGroup

        if issubclass(cls, Exception):
            for exc in __exceptions:
                if not isinstance(exc, Exception):
                    if cls is ExceptionGroup:
                        raise TypeError(
                            "Cannot nest BaseExceptions in an ExceptionGroup"
                        )
                    else:
                        raise TypeError(
                            f"Cannot nest BaseExceptions in {cls.__name__!r}"
                        )

        instance = super().__new__(cls, __message, __exceptions)
        instance._exceptions = tuple(__exceptions)
        return instance

    def __init__(
        self,
        __message: str,
        __exceptions: Sequence[_BaseExceptionT_co],
        *args: object,
    ) -> None:
        BaseException.__init__(self, __message, __exceptions, *args)

    def add_note(self, note: str) -> None:
        if not isinstance(note, str):
            raise TypeError(
                f"Expected a string, got note={note!r} (type {type(note).__name__})"
            )

        if not hasattr(self, "__notes__"):
            self.__notes__: list[str] = []

        self.__notes__.append(note)

    @property
    def message(self) -> str:
        return self.args[0]

    @property
    def exceptions(
        self,
    ) -> tuple[_BaseExceptionT_co | BaseExceptionGroup[_BaseExceptionT_co], ...]:
        return tuple(self._exceptions)

    @overload
    def subgroup(
        self, __condition: type[_ExceptionT] | tuple[type[_ExceptionT], ...]
    ) -> ExceptionGroup[_ExceptionT] | None: ...

    @overload
    def subgroup(
        self, __condition: type[_BaseExceptionT] | tuple[type[_BaseExceptionT], ...]
    ) -> BaseExceptionGroup[_BaseExceptionT] | None: ...

    @overload
    def subgroup(
        self,
        __condition: Callable[[_BaseExceptionT_co | _BaseExceptionGroupSelf], bool],
    ) -> BaseExceptionGroup[_BaseExceptionT_co] | None: ...

    def subgroup(
        self,
        __condition: type[_BaseExceptionT]
        | tuple[type[_BaseExceptionT], ...]
        | Callable[[_BaseExceptionT_co | _BaseExceptionGroupSelf], bool],
    ) -> BaseExceptionGroup[_BaseExceptionT] | None:
        condition = get_condition_filter(__condition)
        modified = False
        if condition(self):
            return self

        exceptions: list[BaseException] = []
        for exc in self.exceptions:
            if isinstance(exc, BaseExceptionGroup):
                subgroup = exc.subgroup(__condition)
                if subgroup is not None:
                    exceptions.append(subgroup)

                if subgroup is not exc:
                    modified = True
            elif condition(exc):
                exceptions.append(exc)
            else:
                modified = True

        if not modified:
            return self
        elif exceptions:
            group = _derive_and_copy_attributes(self, exceptions)
            return group
        else:
            return None

    @overload
    def split(
        self, __condition: type[_ExceptionT] | tuple[type[_ExceptionT], ...]
    ) -> tuple[
        ExceptionGroup[_ExceptionT] | None,
        BaseExceptionGroup[_BaseExceptionT_co] | None,
    ]: ...

    @overload
    def split(
        self, __condition: type[_BaseExceptionT] | tuple[type[_BaseExceptionT], ...]
    ) -> tuple[
        BaseExceptionGroup[_BaseExceptionT] | None,
        BaseExceptionGroup[_BaseExceptionT_co] | None,
    ]: ...

    @overload
    def split(
        self,
        __condition: Callable[[_BaseExceptionT_co | _BaseExceptionGroupSelf], bool],
    ) -> tuple[
        BaseExceptionGroup[_BaseExceptionT_co] | None,
        BaseExceptionGroup[_BaseExceptionT_co] | None,
    ]: ...

    def split(
        self,
        __condition: type[_BaseExceptionT]
        | tuple[type[_BaseExceptionT], ...]
        | Callable[[_BaseExceptionT_co], bool],
    ) -> (
        tuple[
            ExceptionGroup[_ExceptionT] | None,
            BaseExceptionGroup[_BaseExceptionT_co] | None,
        ]
        | tuple[
            BaseExceptionGroup[_BaseExceptionT] | None,
            BaseExceptionGroup[_BaseExceptionT_co] | None,
        ]
        | tuple[
            BaseExceptionGroup[_BaseExceptionT_co] | None,
            BaseExceptionGroup[_BaseExceptionT_co] | None,
        ]
    ):
        condition = get_condition_filter(__condition)
        if condition(self):
            return self, None

        matching_exceptions: list[BaseException] = []
        nonmatching_exceptions: list[BaseException] = []
        for exc in self.exceptions:
            if isinstance(exc, BaseExceptionGroup):
                matching, nonmatching = exc.split(condition)
                if matching is not None:
                    matching_exceptions.append(matching)

                if nonmatching is not None:
                    nonmatching_exceptions.append(nonmatching)
            elif condition(exc):
                matching_exceptions.append(exc)
            else:
                nonmatching_exceptions.append(exc)

        matching_group: _BaseExceptionGroupSelf | None = None
        if matching_exceptions:
            matching_group = _derive_and_copy_attributes(self, matching_exceptions)

        nonmatching_group: _BaseExceptionGroupSelf | None = None
        if nonmatching_exceptions:
            nonmatching_group = _derive_and_copy_attributes(
                self, nonmatching_exceptions
            )

        return matching_group, nonmatching_group

    @overload
    def derive(self, __excs: Sequence[_ExceptionT]) -> ExceptionGroup[_ExceptionT]: ...

    @overload
    def derive(
        self, __excs: Sequence[_BaseExceptionT]
    ) -> BaseExceptionGroup[_BaseExceptionT]: ...

    def derive(
        self, __excs: Sequence[_BaseExceptionT]
    ) -> BaseExceptionGroup[_BaseExceptionT]:
        return BaseExceptionGroup(self.message, __excs)

    def __str__(self) -> str:
        suffix = "" if len(self._exceptions) == 1 else "s"
        return f"{self.message} ({len(self._exceptions)} sub-exception{suffix})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.args[0]!r}, {self.args[1]!r})"


class ExceptionGroup(BaseExceptionGroup[_ExceptionT_co], Exception):
    def __new__(
        cls: type[_ExceptionGroupSelf],
        __message: str,
        __exceptions: Sequence[_ExceptionT_co],
    ) -> _ExceptionGroupSelf:
        return super().__new__(cls, __message, __exceptions)

    if TYPE_CHECKING:

        @property
        def exceptions(
            self,
        ) -> tuple[_ExceptionT_co | ExceptionGroup[_ExceptionT_co], ...]: ...

        @overload  # type: ignore[override]
        def subgroup(
            self, __condition: type[_ExceptionT] | tuple[type[_ExceptionT], ...]
        ) -> ExceptionGroup[_ExceptionT] | None: ...

        @overload
        def subgroup(
            self, __condition: Callable[[_ExceptionT_co | _ExceptionGroupSelf], bool]
        ) -> ExceptionGroup[_ExceptionT_co] | None: ...

        def subgroup(
            self,
            __condition: type[_ExceptionT]
            | tuple[type[_ExceptionT], ...]
            | Callable[[_ExceptionT_co], bool],
        ) -> ExceptionGroup[_ExceptionT] | None:
            return super().subgroup(__condition)

        @overload
        def split(
            self, __condition: type[_ExceptionT] | tuple[type[_ExceptionT], ...]
        ) -> tuple[
            ExceptionGroup[_ExceptionT] | None, ExceptionGroup[_ExceptionT_co] | None
        ]: ...

        @overload
        def split(
            self, __condition: Callable[[_ExceptionT_co | _ExceptionGroupSelf], bool]
        ) -> tuple[
            ExceptionGroup[_ExceptionT_co] | None, ExceptionGroup[_ExceptionT_co] | None
        ]: ...

        def split(
            self: _ExceptionGroupSelf,
            __condition: type[_ExceptionT]
            | tuple[type[_ExceptionT], ...]
            | Callable[[_ExceptionT_co], bool],
        ) -> tuple[
            ExceptionGroup[_ExceptionT_co] | None, ExceptionGroup[_ExceptionT_co] | None
        ]:
            return super().split(__condition)
<|endoftext|>from __future__ import annotations

import sys
from contextlib import AbstractContextManager
from types import TracebackType
from typing import TYPE_CHECKING, Optional, Type, cast

if sys.version_info < (3, 11):
    from ._exceptions import BaseExceptionGroup

if TYPE_CHECKING:
    # requires python 3.9
    BaseClass = AbstractContextManager[None]
else:
    BaseClass = AbstractContextManager


class suppress(BaseClass):
    """Backport of :class:`contextlib.suppress` from Python 3.12.1."""

    def __init__(self, *exceptions: type[BaseException]):
        self._exceptions = exceptions

    def __enter__(self) -> None:
        pass

    def __exit__(
        self,
        exctype: Optional[Type[BaseException]],
        excinst: Optional[BaseException],
        exctb: Optional[TracebackType],
    ) -> bool:
        # Unlike isinstance and issubclass, CPython exception handling
        # currently only looks at the concrete type hierarchy (ignoring
        # the instance and subclass checking hooks). While Guido considers
        # that a bug rather than a feature, it's a fairly hard one to fix
        # due to various internal implementation details. suppress provides
        # the simpler issubclass based semantics, rather than trying to
        # exactly reproduce the limitations of the CPython interpreter.
        #
        # See http://bugs.python.org/issue12029 for more details
        if exctype is None:
            return False

        if issubclass(exctype, self._exceptions):
            return True

        if issubclass(exctype, BaseExceptionGroup):
            match, rest = cast(BaseExceptionGroup, excinst).split(self._exceptions)
            if rest is None:
                return True

            raise rest

        return False
<|endoftext|># traceback_exception_init() adapted from trio
#
# _ExceptionPrintContext and traceback_exception_format() copied from the standard
# library
from __future__ import annotations

import collections.abc
import sys
import textwrap
import traceback
from functools import singledispatch
from types import TracebackType
from typing import Any, List, Optional

from ._exceptions import BaseExceptionGroup

max_group_width = 15
max_group_depth = 10
_cause_message = (
    "\nThe above exception was the direct cause of the following exception:\n\n"
)

_context_message = (
    "\nDuring handling of the above exception, another exception occurred:\n\n"
)


def _format_final_exc_line(etype, value):
    valuestr = _safe_string(value, "exception")
    if value is None or not valuestr:
        line = f"{etype}\n"
    else:
        line = f"{etype}: {valuestr}\n"

    return line


def _safe_string(value, what, func=str):
    try:
        return func(value)
    except BaseException:
        return f"<{what} {func.__name__}() failed>"


class _ExceptionPrintContext:
    def __init__(self):
        self.seen = set()
        self.exception_group_depth = 0
        self.need_close = False

    def indent(self):
        return " " * (2 * self.exception_group_depth)

    def emit(self, text_gen, margin_char=None):
        if margin_char is None:
            margin_char = "|"
        indent_str = self.indent()
        if self.exception_group_depth:
            indent_str += margin_char + " "

        if isinstance(text_gen, str):
            yield textwrap.indent(text_gen, indent_str, lambda line: True)
        else:
            for text in text_gen:
                yield textwrap.indent(text, indent_str, lambda line: True)


def exceptiongroup_excepthook(
    etype: type[BaseException], value: BaseException, tb: TracebackType | None
) -> None:
    sys.stderr.write("".join(traceback.format_exception(etype, value, tb)))


class PatchedTracebackException(traceback.TracebackException):
    def __init__(
        self,
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_traceback: TracebackType | None,
        *,
        limit: int | None = None,
        lookup_lines: bool = True,
        capture_locals: bool = False,
        compact: bool = False,
        _seen: set[int] | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {}
        if sys.version_info >= (3, 10):
            kwargs["compact"] = compact

        is_recursive_call = _seen is not None
        if _seen is None:
            _seen = set()
        _seen.add(id(exc_value))

        self.stack = traceback.StackSummary.extract(
            traceback.walk_tb(exc_traceback),
            limit=limit,
            lookup_lines=lookup_lines,
            capture_locals=capture_locals,
        )
        self.exc_type = exc_type
        # Capture now to permit freeing resources: only complication is in the
        # unofficial API _format_final_exc_line
        self._str = _safe_string(exc_value, "exception")
        try:
            self.__notes__ = getattr(exc_value, "__notes__", None)
        except KeyError:
            # Workaround for https://github.com/python/cpython/issues/98778 on Python
            # <= 3.9, and some 3.10 and 3.11 patch versions.
            HTTPError = getattr(sys.modules.get("urllib.error", None), "HTTPError", ())
            if sys.version_info[:2] <= (3, 11) and isinstance(exc_value, HTTPError):
                self.__notes__ = None
            else:
                raise

        if exc_type and issubclass(exc_type, SyntaxError):
            # Handle SyntaxError's specially
            self.filename = exc_value.filename
            lno = exc_value.lineno
            self.lineno = str(lno) if lno is not None else None
            self.text = exc_value.text
            self.offset = exc_value.offset
            self.msg = exc_value.msg
            if sys.version_info >= (3, 10):
                end_lno = exc_value.end_lineno
                self.end_lineno = str(end_lno) if end_lno is not None else None
                self.end_offset = exc_value.end_offset
        elif (
            exc_type
            and issubclass(exc_type, (NameError, AttributeError))
            and getattr(exc_value, "name", None) is not None
        ):
            suggestion = _compute_suggestion_error(exc_value, exc_traceback)
            if suggestion:
                self._str += f". Did you mean: '{suggestion}'?"

        if lookup_lines:
            # Force all lines in the stack to be loaded
            for frame in self.stack:
                frame.line

        self.__suppress_context__ = (
            exc_value.__suppress_context__ if exc_value is not None else False
        )

        # Convert __cause__ and __context__ to `TracebackExceptions`s, use a
        # queue to avoid recursion (only the top-level call gets _seen == None)
        if not is_recursive_call:
            queue = [(self, exc_value)]
            while queue:
                te, e = queue.pop()

                if e and e.__cause__ is not None and id(e.__cause__) not in _seen:
                    cause = PatchedTracebackException(
                        type(e.__cause__),
                        e.__cause__,
                        e.__cause__.__traceback__,
                        limit=limit,
                        lookup_lines=lookup_lines,
                        capture_locals=capture_locals,
                        _seen=_seen,
                    )
                else:
                    cause = None

                if compact:
                    need_context = (
                        cause is None and e is not None and not e.__suppress_context__
                    )
                else:
                    need_context = True
                if (
                    e
                    and e.__context__ is not None
                    and need_context
                    and id(e.__context__) not in _seen
                ):
                    context = PatchedTracebackException(
                        type(e.__context__),
                        e.__context__,
                        e.__context__.__traceback__,
                        limit=limit,
                        lookup_lines=lookup_lines,
                        capture_locals=capture_locals,
                        _seen=_seen,
                    )
                else:
                    context = None

                # Capture each of the exceptions in the ExceptionGroup along with each
                # of their causes and contexts
                if e and isinstance(e, BaseExceptionGroup):
                    exceptions = []
                    for exc in e.exceptions:
                        texc = PatchedTracebackException(
                            type(exc),
                            exc,
                            exc.__traceback__,
                            lookup_lines=lookup_lines,
                            capture_locals=capture_locals,
                            _seen=_seen,
                        )
                        exceptions.append(texc)
                else:
                    exceptions = None

                te.__cause__ = cause
                te.__context__ = context
                te.exceptions = exceptions
                if cause:
                    queue.append((te.__cause__, e.__cause__))
                if context:
                    queue.append((te.__context__, e.__context__))
                if exceptions:
                    queue.extend(zip(te.exceptions, e.exceptions))

    def format(self, *, chain=True, _ctx=None, **kwargs):
        if _ctx is None:
            _ctx = _ExceptionPrintContext()

        output = []
        exc = self
        if chain:
            while exc:
                if exc.__cause__ is not None:
                    chained_msg = _cause_message
                    chained_exc = exc.__cause__
                elif exc.__context__ is not None and not exc.__suppress_context__:
                    chained_msg = _context_message
                    chained_exc = exc.__context__
                else:
                    chained_msg = None
                    chained_exc = None

                output.append((chained_msg, exc))
                exc = chained_exc
        else:
            output.append((None, exc))

        for msg, exc in reversed(output):
            if msg is not None:
                yield from _ctx.emit(msg)
            if getattr(exc, "exceptions", None) is None:
                if exc.stack:
                    yield from _ctx.emit("Traceback (most recent call last):\n")
                    yield from _ctx.emit(exc.stack.format())
                yield from _ctx.emit(exc.format_exception_only())
            elif _ctx.exception_group_depth > max_group_depth:
                # exception group, but depth exceeds limit
                yield from _ctx.emit(f"... (max_group_depth is {max_group_depth})\n")
            else:
                # format exception group
                is_toplevel = _ctx.exception_group_depth == 0
                if is_toplevel:
                    _ctx.exception_group_depth += 1

                if exc.stack:
                    yield from _ctx.emit(
                        "Exception Group Traceback (most recent call last):\n",
                        margin_char="+" if is_toplevel else None,
                    )
                    yield from _ctx.emit(exc.stack.format())

                yield from _ctx.emit(exc.format_exception_only())
                num_excs = len(exc.exceptions)
                if num_excs <= max_group_width:
                    n = num_excs
                else:
                    n = max_group_width + 1
                _ctx.need_close = False
                for i in range(n):
                    last_exc = i == n - 1
                    if last_exc:
                        # The closing frame may be added by a recursive call
                        _ctx.need_close = True

                    if max_group_width is not None:
                        truncated = i >= max_group_width
                    else:
                        truncated = False
                    title = f"{i + 1}" if not truncated else "..."
                    yield (
                        _ctx.indent()
                        + ("+-" if i == 0 else "  ")
                        + f"+---------------- {title} ----------------\n"
                    )
                    _ctx.exception_group_depth += 1
                    if not truncated:
                        yield from exc.exceptions[i].format(chain=chain, _ctx=_ctx)
                    else:
                        remaining = num_excs - max_group_width
                        plural = "s" if remaining > 1 else ""
                        yield from _ctx.emit(
                            f"and {remaining} more exception{plural}\n"
                        )

                    if last_exc and _ctx.need_close:
                        yield _ctx.indent() + "+------------------------------------\n"
                        _ctx.need_close = False
                    _ctx.exception_group_depth -= 1

                if is_toplevel:
                    assert _ctx.exception_group_depth == 1
                    _ctx.exception_group_depth = 0

    def format_exception_only(self, **kwargs):
        """Format the exception part of the traceback.
        The return value is a generator of strings, each ending in a newline.
        Normally, the generator emits a single string; however, for
        SyntaxError exceptions, it emits several lines that (when
        printed) display detailed information about where the syntax
        error occurred.
        The message indicating which exception occurred is always the last
        string in the output.
        """
        if self.exc_type is None:
            yield traceback._format_final_exc_line(None, self._str)
            return

        stype = self.exc_type.__qualname__
        smod = self.exc_type.__module__
        if smod not in ("__main__", "builtins"):
            if not isinstance(smod, str):
                smod = "<unknown>"
            stype = smod + "." + stype

        if not issubclass(self.exc_type, SyntaxError):
            yield _format_final_exc_line(stype, self._str)
        elif traceback_exception_format_syntax_error is not None:
            yield from traceback_exception_format_syntax_error(self, stype)
        else:
            yield from traceback_exception_original_format_exception_only(self)

        notes = getattr(self, "__notes__", None)
        if isinstance(notes, collections.abc.Sequence):
            for note in notes:
                note = _safe_string(note, "note")
                yield from [line + "\n" for line in note.split("\n")]
        elif notes is not None:
            yield _safe_string(notes, "__notes__", func=repr)


traceback_exception_original_format = traceback.TracebackException.format
traceback_exception_original_format_exception_only = (
    traceback.TracebackException.format_exception_only
)
traceback_exception_format_syntax_error = getattr(
    traceback.TracebackException, "_format_syntax_error", None
)
if sys.excepthook is sys.__excepthook__:
    traceback.TracebackException.__init__ = (  # type: ignore[assignment]
        PatchedTracebackException.__init__
    )
    traceback.TracebackException.format = (  # type: ignore[assignment]
        PatchedTracebackException.format
    )
    traceback.TracebackException.format_exception_only = (  # type: ignore[assignment]
        PatchedTracebackException.format_exception_only
    )
    sys.excepthook = exceptiongroup_excepthook

# Ubuntu's system Python has a sitecustomize.py file that imports
# apport_python_hook and replaces sys.excepthook.
#
# The custom hook captures the error for crash reporting, and then calls
# sys.__excepthook__ to actually print the error.
#
# We don't mind it capturing the error for crash reporting, but we want to
# take over printing the error. So we monkeypatch the apport_python_hook
# module so that instead of calling sys.__excepthook__, it calls our custom
# hook.
#
# More details: https://github.com/python-trio/trio/issues/1065
if getattr(sys.excepthook, "__name__", None) in (
    "apport_excepthook",
    # on ubuntu 22.10 the hook was renamed to partial_apport_excepthook
    "partial_apport_excepthook",
):
    # patch traceback like above
    traceback.TracebackException.__init__ = (  # type: ignore[assignment]
        PatchedTracebackException.__init__
    )
    traceback.TracebackException.format = (  # type: ignore[assignment]
        PatchedTracebackException.format
    )
    traceback.TracebackException.format_exception_only = (  # type: ignore[assignment]
        PatchedTracebackException.format_exception_only
    )

    from types import ModuleType

    import apport_python_hook

    # monkeypatch the sys module that apport has imported
    fake_sys = ModuleType("exceptiongroup_fake_sys")
    fake_sys.__dict__.update(sys.__dict__)
    fake_sys.__excepthook__ = exceptiongroup_excepthook
    apport_python_hook.sys = fake_sys


@singledispatch
def format_exception_only(__exc: BaseException, **kwargs: Any) -> List[str]:
    return list(
        PatchedTracebackException(
            type(__exc), __exc, None, compact=True
        ).format_exception_only()
    )


@format_exception_only.register
def _(__exc: type, value: BaseException, **kwargs: Any) -> List[str]:
    return format_exception_only(value)


@singledispatch
def format_exception(
    __exc: BaseException, limit: Optional[int] = None, chain: bool = True, **kwargs: Any
) -> List[str]:
    return list(
        PatchedTracebackException(
            type(__exc), __exc, __exc.__traceback__, limit=limit, compact=True
        ).format(chain=chain)
    )


@format_exception.register
def _(
    __exc: type,
    value: BaseException,
    tb: TracebackType,
    limit: Optional[int] = None,
    chain: bool = True,
    **kwargs: Any,
) -> List[str]:
    return format_exception(value, limit, chain)


@singledispatch
def print_exception(
    __exc: BaseException,
    limit: Optional[int] = None,
    file: Any = None,
    chain: bool = True,
    **kwargs: Any,
) -> None:
    if file is None:
        file = sys.stderr

    for line in PatchedTracebackException(
        type(__exc), __exc, __exc.__traceback__, limit=limit
    ).format(chain=chain):
        print(line, file=file, end="")


@print_exception.register
def _(
    __exc: type,
    value: BaseException,
    tb: TracebackType,
    limit: Optional[int] = None,
    file: Any = None,
    chain: bool = True,
) -> None:
    print_exception(value, limit, file, chain)


def print_exc(
    limit: Optional[int] = None,
    file: Any | None = None,
    chain: bool = True,
) -> None:
    value = sys.exc_info()[1]
    print_exception(value, limit, file, chain)


# Python levenshtein edit distance code for NameError/AttributeError
# suggestions, backported from 3.12

_MAX_CANDIDATE_ITEMS = 750
_MAX_STRING_SIZE = 40
_MOVE_COST = 2
_CASE_COST = 1
_SENTINEL = object()


def _substitution_cost(ch_a, ch_b):
    if ch_a == ch_b:
        return 0
    if ch_a.lower() == ch_b.lower():
        return _CASE_COST
    return _MOVE_COST


def _compute_suggestion_error(exc_value, tb):
    wrong_name = getattr(exc_value, "name", None)
    if wrong_name is None or not isinstance(wrong_name, str):
        return None
    if isinstance(exc_value, AttributeError):
        obj = getattr(exc_value, "obj", _SENTINEL)
        if obj is _SENTINEL:
            return None
        obj = exc_value.obj
        try:
            d = dir(obj)
        except Exception:
            return None
    else:
        assert isinstance(exc_value, NameError)
        # find most recent frame
        if tb is None:
            return None
        while tb.tb_next is not None:
            tb = tb.tb_next
        frame = tb.tb_frame

        d = list(frame.f_locals) + list(frame.f_globals) + list(frame.f_builtins)
    if len(d) > _MAX_CANDIDATE_ITEMS:
        return None
    wrong_name_len = len(wrong_name)
    if wrong_name_len > _MAX_STRING_SIZE:
        return None
    best_distance = wrong_name_len
    suggestion = None
    for possible_name in d:
        if possible_name == wrong_name:
            # A missing attribute is "found". Don't suggest it (see GH-88821).
            continue
        # No more than 1/3 of the involved characters should need changed.
        max_distance = (len(possible_name) + wrong_name_len + 3) * _MOVE_COST // 6
        # Don't take matches we've already beaten.
        max_distance = min(max_distance, best_distance - 1)
        current_distance = _levenshtein_distance(
            wrong_name, possible_name, max_distance
        )
        if current_distance > max_distance:
            continue
        if not suggestion or current_distance < best_distance:
            suggestion = possible_name
            best_distance = current_distance
    return suggestion


def _levenshtein_distance(a, b, max_cost):
    # A Python implementation of Python/suggestions.c:levenshtein_distance.

    # Both strings are the same
    if a == b:
        return 0

    # Trim away common affixes
    pre = 0
    while a[pre:] and b[pre:] and a[pre] == b[pre]:
        pre += 1
    a = a[pre:]
    b = b[pre:]
    post = 0
    while a[: post or None] and b[: post or None] and a[post - 1] == b[post - 1]:
        post -= 1
    a = a[: post or None]
    b = b[: post or None]
    if not a or not b:
        return _MOVE_COST * (len(a) + len(b))
    if len(a) > _MAX_STRING_SIZE or len(b) > _MAX_STRING_SIZE:
        return max_cost + 1

    # Prefer shorter buffer
    if len(b) < len(a):
        a, b = b, a

    # Quick fail when a match is impossible
    if (len(b) - len(a)) * _MOVE_COST > max_cost:
        return max_cost + 1

    # Instead of producing the whole traditional len(a)-by-len(b)
    # matrix, we can update just one row in place.
    # Initialize the buffer row
    row = list(range(_MOVE_COST, _MOVE_COST * (len(a) + 1), _MOVE_COST))

    result = 0
    for bindex in range(len(b)):
        bchar = b[bindex]
        distance = result = bindex * _MOVE_COST
        minimum = sys.maxsize
        for index in range(len(a)):
            # 1) Previous distance in this row is cost(b[:b_index], a[:index])
            substitute = distance + _substitution_cost(bchar, a[index])
            # 2) cost(b[:b_index], a[:index+1]) from previous row
            distance = row[index]
            # 3) existing result is cost(b[:b_index+1], a[index])

            insert_delete = min(result, distance) + _MOVE_COST
            result = min(insert_delete, substitute)

            # cost(b[:b_index+1], a[:index+1])
            row[index] = result
            if result < minimum:
                minimum = result
        if minimum > max_cost:
            # Everything in this row is too big, so bail early.
            return max_cost + 1
    return result
<|endoftext|># file generated by setuptools-scm
# don't change, don't track in version control

__all__ = [
    "__version__",
    "__version_tuple__",
    "version",
    "version_tuple",
    "__commit_id__",
    "commit_id",
]

TYPE_CHECKING = False
if TYPE_CHECKING:
    from typing import Tuple
    from typing import Union

    VERSION_TUPLE = Tuple[Union[int, str], ...]
    COMMIT_ID = Union[str, None]
else:
    VERSION_TUPLE = object
    COMMIT_ID = object

version: str
__version__: str
__version_tuple__: VERSION_TUPLE
version_tuple: VERSION_TUPLE
commit_id: COMMIT_ID
__commit_id__: COMMIT_ID

__version__ = version = '1.3.1'
__version_tuple__ = version_tuple = (1, 3, 1)

__commit_id__ = commit_id = None
<|endoftext|># Copyright 2020 The HuggingFace Datasets Authors and the TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""DatasetBuilder base class."""

import abc
import contextlib
import copy
import inspect
import os
import posixpath
import shutil
import time
import urllib
from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union
from unittest.mock import patch

import fsspec
import pyarrow as pa
from fsspec.core import url_to_fs
from multiprocess import Pool
from tqdm.contrib.concurrent import thread_map

from . import config, utils
from .arrow_dataset import Dataset
from .arrow_reader import (
    ArrowReader,
    ReadInstruction,
)
from .arrow_writer import ArrowWriter, ParquetWriter, SchemaInferenceError
from .data_files import DataFilesDict, DataFilesPatternsDict, sanitize_patterns
from .dataset_dict import DatasetDict, IterableDatasetDict
from .download.download_config import DownloadConfig
from .download.download_manager import DownloadManager, DownloadMode
from .download.streaming_download_manager import StreamingDownloadManager, xjoin
from .exceptions import DatasetGenerationCastError, DatasetGenerationError, FileFormatError
from .features import Features
from .filesystems import (
    is_remote_filesystem,
    rename,
)
from .fingerprint import Hasher
from .info import DatasetInfo
from .iterable_dataset import (
    ArrowExamplesIterable,
    ExamplesIterable,
    IterableDataset,
    _concatenate_iterable_datasets,
)
from .naming import INVALID_WINDOWS_CHARACTERS_IN_PATH, camelcase_to_snakecase
from .splits import Split, SplitDict, SplitGenerator, SplitInfo
from .streaming import extend_dataset_builder_for_streaming
from .table import CastError
from .utils import logging
from .utils import tqdm as hf_tqdm
from .utils._filelock import FileLock
from .utils.file_utils import is_remote_url
from .utils.info_utils import VerificationMode, verify_checksums, verify_splits
from .utils.py_utils import (
    classproperty,
    convert_file_size_to_int,
    has_sufficient_disk_space,
    iflatmap_unordered,
    map_nested,
    memoize,
    size_str,
    temporary_assignment,
)
from .utils.sharding import _number_of_shards_in_gen_kwargs, _split_gen_kwargs
from .utils.track import tracked_list


if TYPE_CHECKING:
    from .load import DatasetModule


logger = logging.get_logger(__name__)


class InvalidConfigName(ValueError):
    pass


@dataclass
class BuilderConfig:
    """Base class for `DatasetBuilder` data configuration.

    `DatasetBuilder` subclasses with data configuration options should subclass
    `BuilderConfig` and add their own properties.

    Attributes:
        name (`str`, defaults to `default`):
            The name of the configuration.
        version (`Version` or `str`, defaults to `0.0.0`):
            The version of the configuration.
        data_dir (`str`, *optional*):
            Path to the directory containing the source data.
        data_files (`str` or `Sequence` or `Mapping`, *optional*):
            Path(s) to source data file(s).
        description (`str`, *optional*):
            A human description of the configuration.
    """

    name: str = "default"
    version: Optional[Union[utils.Version, str]] = utils.Version("0.0.0")
    data_dir: Optional[str] = None
    data_files: Optional[Union[DataFilesDict, DataFilesPatternsDict]] = None
    description: Optional[str] = None

    def __post_init__(self):
        # The config name is used to name the cache directory.
        for invalid_char in INVALID_WINDOWS_CHARACTERS_IN_PATH:
            if invalid_char in self.name:
                raise InvalidConfigName(
                    f"Bad characters from black list '{INVALID_WINDOWS_CHARACTERS_IN_PATH}' found in '{self.name}'. "
                    f"They could create issues when creating a directory for this config on Windows filesystem."
                )
        if self.data_files is not None and not isinstance(self.data_files, (DataFilesDict, DataFilesPatternsDict)):
            raise ValueError(f"Expected a DataFilesDict in data_files but got {self.data_files}")

    def __eq__(self, o):
        # we need to override the default dataclass __eq__ since it doesn't check for
        # other attributes that the ones of the signature.
        if set(self.__dict__.keys()) != set(o.__dict__.keys()):
            return False
        return all((k, getattr(self, k)) == (k, getattr(o, k)) for k in self.__dict__.keys())

    def create_config_id(
        self,
        config_kwargs: dict,
        custom_features: Optional[Features] = None,
    ) -> str:
        """
        The config id is used to build the cache directory.
        By default it is equal to the config name.
        However the name of a config is not sufficient to have a unique identifier for the dataset being generated
        since it doesn't take into account:
        - the config kwargs that can be used to overwrite attributes
        - the custom features used to write the dataset
        - the data_files for json/text/csv/pandas datasets

        Therefore the config id is just the config name with an optional suffix based on these.
        """
        # Possibly add a suffix to the name to handle custom features/data_files/config_kwargs
        suffix: Optional[str] = None
        config_kwargs_to_add_to_suffix = config_kwargs.copy()
        # name and version are already used to build the cache directory
        config_kwargs_to_add_to_suffix.pop("name", None)
        config_kwargs_to_add_to_suffix.pop("version", None)
        # data dir handling (when specified it points to the manually downloaded data):
        # it was previously ignored before the introduction of config id because we didn't want
        # to change the config name. Now it's fine to take it into account for the config id.
        # config_kwargs_to_add_to_suffix.pop("data_dir", None)
        if "data_dir" in config_kwargs_to_add_to_suffix:
            if config_kwargs_to_add_to_suffix["data_dir"] is None:
                config_kwargs_to_add_to_suffix.pop("data_dir", None)
            else:
                # canonicalize the data dir to avoid two paths to the same location having different
                # hashes
                data_dir = config_kwargs_to_add_to_suffix["data_dir"]
                data_dir = os.path.normpath(data_dir)
                config_kwargs_to_add_to_suffix["data_dir"] = data_dir
        if config_kwargs_to_add_to_suffix:
            # we don't care about the order of the kwargs
            config_kwargs_to_add_to_suffix = {
                k: config_kwargs_to_add_to_suffix[k] for k in sorted(config_kwargs_to_add_to_suffix)
            }
            if all(isinstance(v, (str, bool, int, float)) for v in config_kwargs_to_add_to_suffix.values()):
                suffix = ",".join(
                    str(k) + "=" + urllib.parse.quote_plus(str(v)) for k, v in config_kwargs_to_add_to_suffix.items()
                )
                if len(suffix) > 32:  # hash if too long
                    suffix = Hasher.hash(config_kwargs_to_add_to_suffix)
            else:
                suffix = Hasher.hash(config_kwargs_to_add_to_suffix)

        if custom_features is not None:
            m = Hasher()
            if suffix:
                m.update(suffix)
            m.update(custom_features)
            suffix = m.hexdigest()

        if suffix:
            config_id = self.name + "-" + suffix
            if len(config_id) > config.MAX_DATASET_CONFIG_ID_READABLE_LENGTH:
                config_id = self.name + "-" + Hasher.hash(suffix)
            return config_id
        else:
            return self.name

    def _resolve_data_files(self, base_path: str, download_config: DownloadConfig) -> None:
        if isinstance(self.data_files, DataFilesPatternsDict):
            base_path = xjoin(base_path, self.data_dir) if self.data_dir else base_path
            self.data_files = self.data_files.resolve(base_path, download_config)


class DatasetBuilder:
    """Abstract base class for all datasets.

    `DatasetBuilder` has 3 key methods:

        - [`DatasetBuilder.info`]: Documents the dataset, including feature
          names, types, shapes, version, splits, citation, etc.
        - [`DatasetBuilder.download_and_prepare`]: Downloads the source data
          and writes it to disk.
        - [`DatasetBuilder.as_dataset`]: Generates a [`Dataset`].

    Some `DatasetBuilder`s expose multiple variants of the
    dataset by defining a [`BuilderConfig`] subclass and accepting a
    config object (or name) on construction. Configurable datasets expose a
    pre-defined set of configurations in [`DatasetBuilder.builder_configs`].

    Args:
        cache_dir (`str`, *optional*):
            Directory to cache data. Defaults to `"~/.cache/huggingface/datasets"`.
        dataset_name (`str`, *optional*):
            Name of the dataset, if different from the builder name. Useful for packaged builders
            like csv, imagefolder, audiofolder, etc. to reflect the difference between datasets
            that use the same packaged builder.
        config_name (`str`, *optional*):
            Name of the dataset configuration.
            It affects the data generated on disk. Different configurations will have their own subdirectories and
            versions.
            If not provided, the default configuration is used (if it exists).

            <Added version="2.3.0">

            Parameter `name` was renamed to `config_name`.

            </Added>
        hash (`str`, *optional*):
            Hash specific to the dataset builder code. Used to update the caching directory when the
            dataset builder code is updated (to avoid reusing old data).
            The typical caching directory (defined in `self._relative_data_dir`) is `name/version/hash/`.
        base_path (`str`, *optional*):
            Base path for relative paths that are used to download files.
            This can be a remote URL.
        features ([`Features`], *optional*):
            Features types to use with this dataset.
            It can be used to change the [`Features`] types of a dataset, for example.
        token (`str` or `bool`, *optional*):
            String or boolean to use as Bearer token for remote files on the
            Datasets Hub. If `True`, will get token from `"~/.huggingface"`.
        repo_id (`str`, *optional*):
            ID of the dataset repository.
            Used to distinguish builders with the same name but not coming from the same namespace, for example "rajpurkar/squad"
            and "lhoestq/squad" repo IDs. In the latter, the builder name would be "lhoestq___squad".
        data_files (`str` or `Sequence` or `Mapping`, *optional*):
            Path(s) to source data file(s).
            For builders like "csv" or "json" that need the user to specify data files. They can be either
            local or remote files. For convenience, you can use a `DataFilesDict`.
        data_dir (`str`, *optional*):
            Path to directory containing source data file(s).
            Use only if `data_files` is not passed, in which case it is equivalent to passing
            `os.path.join(data_dir, "**")` as `data_files`.
            For builders that require manual download, it must be the path to the local directory containing the
            manually downloaded data.
        storage_options (`dict`, *optional*):
            Key/value pairs to be passed on to the dataset file-system backend, if any.
        writer_batch_size (`int`, *optional*):
            Batch size used by the ArrowWriter.
            It defines the number of samples that are kept in memory before writing them
            and also the length of the arrow chunks.
            None means that the ArrowWriter will use its default value.
        **config_kwargs (additional keyword arguments): Keyword arguments to be passed to the corresponding builder
            configuration class, set on the class attribute [`DatasetBuilder.BUILDER_CONFIG_CLASS`]. The builder
            configuration class is [`BuilderConfig`] or a subclass of it.
    """

    # Default version
    VERSION = None  # Default version set in BuilderConfig

    # Class for the builder config.
    BUILDER_CONFIG_CLASS = BuilderConfig

    # Named configurations that modify the data generated by download_and_prepare.
    BUILDER_CONFIGS = []

    # Optional default config name to be used when name is None
    DEFAULT_CONFIG_NAME = None

    # Default batch size used by the ArrowWriter
    # It defines the number of samples that are kept in memory before writing them
    # and also the length of the arrow chunks
    # None means that the ArrowWriter will use its default value
    DEFAULT_WRITER_BATCH_SIZE = None

    # Useful to make sure PyArrow threads in c++ have time to shut down before gargabe collection
    SLEEP_ON_THREADS_SHUTDOWNS = False

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        dataset_name: Optional[str] = None,
        config_name: Optional[str] = None,
        hash: Optional[str] = None,
        base_path: Optional[str] = None,
        info: Optional[DatasetInfo] = None,
        features: Optional[Features] = None,
        token: Optional[Union[bool, str]] = None,
        repo_id: Optional[str] = None,
        data_files: Optional[Union[str, list, dict, DataFilesDict]] = None,
        data_dir: Optional[str] = None,
        storage_options: Optional[dict] = None,
        writer_batch_size: Optional[int] = None,
        config_id: Optional[str] = None,
        **config_kwargs,
    ):
        # DatasetBuilder name
        self.name: str = camelcase_to_snakecase(self.__module__.split(".")[-1])
        self.hash: Optional[str] = hash
        self.base_path = base_path
        self.token = token
        self.repo_id = repo_id
        self.storage_options = storage_options or {}
        self.dataset_name = camelcase_to_snakecase(dataset_name) if dataset_name else self.name
        self._writer_batch_size = writer_batch_size or self.DEFAULT_WRITER_BATCH_SIZE

        if data_files is not None and not isinstance(data_files, DataFilesDict):
            data_files = DataFilesDict.from_patterns(
                sanitize_patterns(data_files),
                base_path=base_path,
                download_config=DownloadConfig(token=token, storage_options=self.storage_options),
            )

        # Prepare config: DatasetConfig contains name, version and description but can be extended by each dataset
        if "features" in inspect.signature(self.BUILDER_CONFIG_CLASS.__init__).parameters and features is not None:
            config_kwargs["features"] = features
        if data_files is not None:
            config_kwargs["data_files"] = data_files
        if data_dir is not None:
            config_kwargs["data_dir"] = data_dir
        self.config_kwargs = config_kwargs
        self.config, self.config_id = self._create_builder_config(
            config_name=config_name,
            custom_features=features,
            config_id=config_id,
            **config_kwargs,
        )

        # Ensure files are in the repo
        if repo_id is not None and self.config.data_files is not None:
            for split in self.config.data_files:
                for data_file in self.config.data_files[split]:
                    if not posixpath.relpath(data_file, start="hf://").startswith(f"datasets/{repo_id}@"):
                        raise ValueError(
                            f"Data files don't belong to {repo_id}. "
                            "Make sure the dataset `data_files` (e.g. in the config README.md) are valid. "
                            "They should be relative paths to the dataset repository root."
                        )

        # prepare info: DatasetInfo are a standardized dataclass across all datasets
        # Prefill datasetinfo
        if info is None:
            info = self._info()
        info.builder_name = self.name
        info.dataset_name = self.dataset_name
        info.config_name = self.config.name
        info.version = self.config.version
        self.info = info
        # update info with user specified infos
        if features is not None:
            self.info.features = features

        # Prepare data dirs:
        # cache_dir can be a remote bucket on GCS or S3
        self._cache_dir_root = str(cache_dir or config.HF_DATASETS_CACHE)
        self._cache_dir_root = (
            self._cache_dir_root if is_remote_url(self._cache_dir_root) else os.path.expanduser(self._cache_dir_root)
        )
        self._cache_downloaded_dir = (
            posixpath.join(self._cache_dir_root, config.DOWNLOADED_DATASETS_DIR)
            if cache_dir
            else str(config.DOWNLOADED_DATASETS_PATH)
        )
        self._cache_downloaded_dir = (
            self._cache_downloaded_dir
            if is_remote_url(self._cache_downloaded_dir)
            else os.path.expanduser(self._cache_downloaded_dir)
        )

        # In case there exists a legacy cache directory
        self._legacy_relative_data_dir = None

        self._cache_dir = self._build_cache_dir()
        if not is_remote_url(self._cache_dir_root):
            os.makedirs(self._cache_dir_root, exist_ok=True)
            lock_path = os.path.join(
                self._cache_dir_root, Path(self._cache_dir).as_posix().replace("/", "_") + ".lock"
            )
            with FileLock(lock_path):
                if os.path.exists(self._cache_dir):  # check if data exist
                    if len(os.listdir(self._cache_dir)) > 0:
                        if os.path.exists(os.path.join(self._cache_dir, config.DATASET_INFO_FILENAME)):
                            logger.debug("Overwrite dataset info from restored data version if exists.")
                            self.info = DatasetInfo.from_directory(self._cache_dir)
                    else:  # dir exists but no data, remove the empty dir as data aren't available anymore
                        logger.warning(
                            f"Old caching folder {self._cache_dir} for dataset {self.dataset_name} exists but no data were found. Removing it. "
                        )
                        os.rmdir(self._cache_dir)

        # Store in the cache by default unless the user specifies a custom output_dir to download_and_prepare
        self._output_dir = self._cache_dir
        self._fs: fsspec.AbstractFileSystem = fsspec.filesystem("file")

        # Set download manager
        self.dl_manager = None

        # Set to True by "datasets-cli test" to generate file checksums for (deprecated) dataset_infos.json independently of verification_mode value.
        self._record_checksums = False

        # Set in `.download_and_prepare` once the format of the generated dataset is known
        self._file_format = None

        # Enable streaming (e.g. it patches "open" to work with remote files)
        extend_dataset_builder_for_streaming(self)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__ = d
        # Re-enable streaming, since patched functions are not kept when pickling
        extend_dataset_builder_for_streaming(self)

    def _check_legacy_cache(self) -> Optional[str]:
        """Check for the old cache directory template {cache_dir}/{namespace}___{builder_name} from 2.13"""
        if (
            self.__module__.startswith("datasets.")
            and not is_remote_url(self._cache_dir_root)
            and self.config.name == "default"
        ):
            from .packaged_modules import _PACKAGED_DATASETS_MODULES

            namespace = self.repo_id.split("/")[0] if self.repo_id and self.repo_id.count("/") > 0 else None
            config_name = self.repo_id.replace("/", "--") if self.repo_id is not None else self.dataset_name
            config_id = config_name + self.config_id[len(self.config.name) :]
            hash = _PACKAGED_DATASETS_MODULES.get(self.name, "missing")[1]
            legacy_relative_data_dir = posixpath.join(
                self.dataset_name if namespace is None else f"{namespace}___{self.dataset_name}",
                config_id,
                "0.0.0",
                hash,
            )
            legacy_cache_dir = posixpath.join(self._cache_dir_root, legacy_relative_data_dir)
            if os.path.isdir(legacy_cache_dir):
                return legacy_relative_data_dir

    def _check_legacy_cache2(self, dataset_module: "DatasetModule") -> Optional[str]:
        """Check for the old cache directory template {cache_dir}/{namespace}___{dataset_name}/{config_name}-xxx from 2.14 and 2.15"""
        if (
            self.__module__.startswith("datasets.")
            and not is_remote_url(self._cache_dir_root)
            and not (set(self.config_kwargs) - {"data_files", "data_dir"})
        ):
            from .packaged_modules import _PACKAGED_DATASETS_MODULES_2_15_HASHES
            from .utils._dill import Pickler

            def update_hash_with_config_parameters(hash: str, config_parameters: dict) -> str:
                """
                Used to update hash of packaged modules which is used for creating unique cache directories to reflect
                different config parameters which are passed in metadata from readme.
                """
                params_to_exclude = {"config_name", "version", "description"}
                params_to_add_to_hash = {
                    param: value
                    for param, value in sorted(config_parameters.items())
                    if param not in params_to_exclude
                }
                m = Hasher()
                m.update(hash)
                m.update(params_to_add_to_hash)
                return m.hexdigest()

            namespace = self.repo_id.split("/")[0] if self.repo_id and self.repo_id.count("/") > 0 else None
            with patch.object(Pickler, "_legacy_no_dict_keys_sorting", True):
                config_id = self.config.name + "-" + Hasher.hash({"data_files": self.config.data_files})
            hash = _PACKAGED_DATASETS_MODULES_2_15_HASHES.get(self.name, "missing")
            if (
                dataset_module.builder_configs_parameters.metadata_configs
                and self.config.name in dataset_module.builder_configs_parameters.metadata_configs
            ):
                hash = update_hash_with_config_parameters(
                    hash, dataset_module.builder_configs_parameters.metadata_configs[self.config.name]
                )
            legacy_relative_data_dir = posixpath.join(
                self.dataset_name if namespace is None else f"{namespace}___{self.dataset_name}",
                config_id,
                "0.0.0",
                hash,
            )
            legacy_cache_dir = posixpath.join(self._cache_dir_root, legacy_relative_data_dir)
            if os.path.isdir(legacy_cache_dir):
                return legacy_relative_data_dir

    def _create_builder_config(
        self, config_name=None, custom_features=None, config_id=None, **config_kwargs
    ) -> tuple[BuilderConfig, str]:
        """Create and validate BuilderConfig object as well as a unique config id for this config.
        Raises ValueError if there are multiple builder configs and config_name and DEFAULT_CONFIG_NAME are None.
        config_kwargs override the defaults kwargs in config
        """
        builder_config = None

        # try default config
        if config_name is None and self.BUILDER_CONFIGS:
            if self.DEFAULT_CONFIG_NAME is not None:
                builder_config = self.builder_configs.get(self.DEFAULT_CONFIG_NAME)
                logger.info(f"No config specified, defaulting to: {self.dataset_name}/{builder_config.name}")
            else:
                if len(self.BUILDER_CONFIGS) > 1:
                    if not config_kwargs:
                        example_of_usage = (
                            f"load_dataset('{self.repo_id or self.dataset_name}', '{self.BUILDER_CONFIGS[0].name}')"
                        )
                        raise ValueError(
                            "Config name is missing."
                            f"\nPlease pick one among the available configs: {list(self.builder_configs.keys())}"
                            + f"\nExample of usage:\n\t`{example_of_usage}`"
                        )
                else:
                    builder_config = self.BUILDER_CONFIGS[0]
                    logger.info(
                        f"No config specified, defaulting to the single config: {self.dataset_name}/{builder_config.name}"
                    )

        # try to get config by name
        if isinstance(config_name, str):
            builder_config = self.builder_configs.get(config_name)
            if builder_config is None and self.BUILDER_CONFIGS:
                raise ValueError(
                    f"BuilderConfig '{config_name}' not found. Available: {list(self.builder_configs.keys())}"
                )

        # if not using an existing config, then create a new config on the fly
        if not builder_config:
            if config_name is not None:
                config_kwargs["name"] = config_name
            elif self.DEFAULT_CONFIG_NAME and not config_kwargs:
                # Use DEFAULT_CONFIG_NAME only if no config_kwargs are passed
                config_kwargs["name"] = self.DEFAULT_CONFIG_NAME
            if "version" not in config_kwargs and hasattr(self, "VERSION") and self.VERSION:
                config_kwargs["version"] = self.VERSION
            builder_config = self.BUILDER_CONFIG_CLASS(**config_kwargs)

        # otherwise use the config_kwargs to overwrite the attributes
        else:
            builder_config = copy.deepcopy(builder_config) if config_kwargs else builder_config
            for key, value in config_kwargs.items():
                if value is not None:
                    if not hasattr(builder_config, key):
                        raise ValueError(f"BuilderConfig {builder_config} doesn't have a '{key}' key.")
                    setattr(builder_config, key, value)

        if not builder_config.name:
            raise ValueError(f"BuilderConfig must have a name, got {builder_config.name}")

        # resolve data files if needed
        builder_config._resolve_data_files(
            base_path=self.base_path,
            download_config=DownloadConfig(token=self.token, storage_options=self.storage_options),
        )

        # compute the config id that is going to be used for caching
        if config_id is None:
            config_id = builder_config.create_config_id(
                config_kwargs,
                custom_features=custom_features,
            )
        is_custom = (config_id not in self.builder_configs) and config_id != "default"
        if is_custom:
            logger.info(f"Using custom data configuration {config_id}")
        else:
            if (
                builder_config.name in self.builder_configs
                and builder_config != self.builder_configs[builder_config.name]
            ):
                raise ValueError(
                    "Cannot name a custom BuilderConfig the same as an available "
                    f"BuilderConfig. Change the name. Available BuilderConfigs: {list(self.builder_configs.keys())}"
                )
            if not builder_config.version:
                raise ValueError(f"BuilderConfig {builder_config.name} must have a version")

        return builder_config, config_id

    @classproperty
    @classmethod
    @memoize()
    def builder_configs(cls) -> dict[str, BuilderConfig]:
        """Dictionary of pre-defined configurations for this builder class."""
        configs = {config.name: config for config in cls.BUILDER_CONFIGS}
        if len(configs) != len(cls.BUILDER_CONFIGS):
            names = [config.name for config in cls.BUILDER_CONFIGS]
            raise ValueError(f"Names in BUILDER_CONFIGS must not be duplicated. Got {names}")
        return configs

    @property
    def cache_dir(self):
        return self._cache_dir

    def _use_legacy_cache_dir_if_possible(self, dataset_module: "DatasetModule"):
        # Check for the legacy cache directory template (datasets<3.0.0)
        self._legacy_relative_data_dir = (
            self._check_legacy_cache2(dataset_module) or self._check_legacy_cache() or None
        )
        self._cache_dir = self._build_cache_dir()
        self._output_dir = self._cache_dir

    def _relative_data_dir(self, with_version=True, with_hash=True) -> str:
        """Relative path of this dataset in cache_dir:
        Will be:
            self.dataset_name/self.config.version/self.hash/
        or if a repo_id with a namespace has been specified:
            self.namespace___self.dataset_name/self.config.version/self.hash/
        If any of these element is missing or if ``with_version=False`` the corresponding subfolders are dropped.
        """
        if self._legacy_relative_data_dir is not None and with_version and with_hash:
            return self._legacy_relative_data_dir

        namespace = self.repo_id.split("/")[0] if self.repo_id and self.repo_id.count("/") > 0 else None
        builder_data_dir = self.dataset_name if namespace is None else f"{namespace}___{self.dataset_name}"
        builder_data_dir = posixpath.join(builder_data_dir, self.config_id)
        if with_version:
            builder_data_dir = posixpath.join(builder_data_dir, str(self.config.version))
        if with_hash and self.hash and isinstance(self.hash, str):
            builder_data_dir = posixpath.join(builder_data_dir, self.hash)
        return builder_data_dir

    def _build_cache_dir(self):
        """Return the data directory for the current version."""
        builder_data_dir = posixpath.join(self._cache_dir_root, self._relative_data_dir(with_version=False))
        version_data_dir = posixpath.join(self._cache_dir_root, self._relative_data_dir(with_version=True))

        def _other_versions_on_disk():
            """Returns previous versions on disk."""
            if not os.path.exists(builder_data_dir):
                return []

            version_dirnames = []
            for dir_name in os.listdir(builder_data_dir):
                try:
                    version_dirnames.append((utils.Version(dir_name), dir_name))
                except ValueError:  # Invalid version (ex: incomplete data dir)
                    pass
            version_dirnames.sort(reverse=True)
            return version_dirnames

        # Check and warn if other versions exist
        if not is_remote_url(builder_data_dir):
            version_dirs = _other_versions_on_disk()
            if version_dirs:
                other_version = version_dirs[0][0]
                if other_version != self.config.version:
                    warn_msg = (
                        f"Found a different version {str(other_version)} of dataset {self.dataset_name} in "
                        f"cache_dir {self._cache_dir_root}. Using currently defined version "
                        f"{str(self.config.version)}."
                    )
                    logger.warning(warn_msg)

        return version_data_dir

    @abc.abstractmethod
    def _info(self) -> DatasetInfo:
        """Construct the DatasetInfo object. See `DatasetInfo` for details.

        Warning: This function is only called once and the result is cached for all
        following .info() calls.

        Returns:
            info: (DatasetInfo) The dataset information
        """
        raise NotImplementedError

    @classmethod
    def get_imported_module_dir(cls):
        """Return the path of the module of this class or subclass."""
        return os.path.dirname(inspect.getfile(inspect.getmodule(cls)))

    def _rename(self, src: str, dst: str):
        rename(self._fs, src, dst)

    def download_and_prepare(
        self,
        output_dir: Optional[str] = None,
        download_config: Optional[DownloadConfig] = None,
        download_mode: Optional[Union[DownloadMode, str]] = None,
        verification_mode: Optional[Union[VerificationMode, str]] = None,
        dl_manager: Optional[DownloadManager] = None,
        base_path: Optional[str] = None,
        file_format: str = "arrow",
        max_shard_size: Optional[Union[int, str]] = None,
        num_proc: Optional[int] = None,
        storage_options: Optional[dict] = None,
        **download_and_prepare_kwargs,
    ):
        """Downloads and prepares dataset for reading.

        Args:
            output_dir (`str`, *optional*):
                Output directory for the dataset.
                Default to this builder's `cache_dir`, which is inside `~/.cache/huggingface/datasets` by default.

                <Added version="2.5.0"/>
            download_config (`DownloadConfig`, *optional*):
                Specific download configuration parameters.
            download_mode ([`DownloadMode`] or `str`, *optional*):
                Select the download/generate mode, default to `REUSE_DATASET_IF_EXISTS`.
            verification_mode ([`VerificationMode`] or `str`, defaults to `BASIC_CHECKS`):
                Verification mode determining the checks to run on the downloaded/processed dataset information (checksums/size/splits/...).

                <Added version="2.9.1"/>
            dl_manager (`DownloadManager`, *optional*):
                Specific `DownloadManger` to use.
            base_path (`str`, *optional*):
                Base path for relative paths that are used to download files. This can be a remote url.
                If not specified, the value of the `base_path` attribute (`self.base_path`) will be used instead.
            file_format (`str`, *optional*):
                Format of the data files in which the dataset will be written.
                Supported formats: "arrow", "parquet". Default to "arrow" format.
                If the format is "parquet", then image and audio data are embedded into the Parquet files instead of pointing to local files.

                <Added version="2.5.0"/>
            max_shard_size (`Union[str, int]`, *optional*):
                Maximum number of bytes written per shard, default is "500MB".
                The size is based on uncompressed data size, so in practice your shard files may be smaller than
                `max_shard_size` thanks to Parquet compression for example.

                <Added version="2.5.0"/>
            num_proc (`int`, *optional*, defaults to `None`):
                Number of processes when downloading and generating the dataset locally.
                Multiprocessing is disabled by default.

                <Added version="2.7.0"/>
            storage_options (`dict`, *optional*):
                Key/value pairs to be passed on to the caching file-system backend, if any.

                <Added version="2.5.0"/>
            **download_and_prepare_kwargs (additional keyword arguments): Keyword arguments.

        Example:

        Download and prepare the dataset as Arrow files that can be loaded as a Dataset using `builder.as_dataset()`:

        ```py
        >>> from datasets import load_dataset_builder
        >>> builder = load_dataset_builder("cornell-movie-review-data/rotten_tomatoes")
        >>> builder.download_and_prepare()
        ```

        Download and prepare the dataset as sharded Parquet files locally:

        ```py
        >>> from datasets import load_dataset_builder
        >>> builder = load_dataset_builder("cornell-movie-review-data/rotten_tomatoes")
        >>> builder.download_and_prepare("./output_dir", file_format="parquet")
        ```

        Download and prepare the dataset as sharded Parquet files in a cloud storage:

        ```py
        >>> from datasets import load_dataset_builder
        >>> storage_options = {"key": aws_access_key_id, "secret": aws_secret_access_key}
        >>> builder = load_dataset_builder("cornell-movie-review-data/rotten_tomatoes")
        >>> builder.download_and_prepare("s3://my-bucket/my_rotten_tomatoes", storage_options=storage_options, file_format="parquet")
        ```
        """
        output_dir = output_dir if output_dir is not None else self._cache_dir
        # output_dir can be a remote bucket on GCS or S3
        fs, output_dir = url_to_fs(output_dir, **(storage_options or {}))
        self._fs = fs
        self._output_dir = output_dir if not is_remote_filesystem(self._fs) else self._fs.unstrip_protocol(output_dir)

        download_mode = DownloadMode(download_mode or DownloadMode.REUSE_DATASET_IF_EXISTS)
        verification_mode = VerificationMode(verification_mode or VerificationMode.BASIC_CHECKS)
        base_path = base_path if base_path is not None else self.base_path

        if file_format is not None and file_format not in ["arrow", "parquet"]:
            raise ValueError(f"Unsupported file_format: {file_format}. Expected 'arrow' or 'parquet'")
        self._file_format = file_format

        if self._fs._strip_protocol(self._output_dir) == "":
            # We don't support the root directory, because it has no dirname,
            # and we need a dirname to use a <dirname>.incomplete directory
            # when the dataset is being written
            raise RuntimeError(
                f"Unable to download and prepare the dataset at the root {self._output_dir}. "
                f"Please specify a subdirectory, e.g. '{self._output_dir + self.dataset_name}'"
            )

        if dl_manager is None:
            if download_config is None:
                download_config = DownloadConfig(
                    cache_dir=self._cache_downloaded_dir,
                    force_download=download_mode == DownloadMode.FORCE_REDOWNLOAD,
                    force_extract=download_mode == DownloadMode.FORCE_REDOWNLOAD,
                    use_etag=False,
                    num_proc=num_proc,
                    token=self.token,
                    storage_options=self.storage_options,
                )  # We don't use etag for data files to speed up the process

            dl_manager = DownloadManager(
                dataset_name=self.dataset_name,
                download_config=download_config,
                data_dir=self.config.data_dir,
                base_path=base_path,
                record_checksums=self._record_checksums,
            )

        is_local = not is_remote_filesystem(self._fs)
        self.dl_manager = dl_manager

        # Prevent parallel local disk operations
        if is_local:
            # Create parent directory of the output_dir to put the lock file in there
            Path(self._output_dir).parent.mkdir(parents=True, exist_ok=True)
            lock_path = self._output_dir + "_builder.lock"

        # File locking only with local paths; no file locking on GCS or S3
        with FileLock(lock_path) if is_local else contextlib.nullcontext():
            # Check if the data already exists
            data_exists = self._fs.exists(posixpath.join(self._output_dir, config.DATASET_INFO_FILENAME))
            if data_exists and download_mode == DownloadMode.REUSE_DATASET_IF_EXISTS:
                logger.info(f"Found cached dataset {self.dataset_name} ({self._output_dir})")
                # We need to update the info in case some splits were added in the meantime
                # for example when calling load_dataset from multiple workers.
                self.info = self._load_info()
                return

            logger.info(f"Generating dataset {self.dataset_name} ({self._output_dir})")
            if is_local:  # if cache dir is local, check for available space
                if not has_sufficient_disk_space(
                    self.info.size_in_bytes or 0, directory=Path(self._output_dir).parent
                ):
                    raise OSError(
                        f"Not enough disk space. Needed: {size_str(self.info.size_in_bytes or 0)} (download: {size_str(self.info.download_size or 0)}, generated: {size_str(self.info.dataset_size or 0)}"
                    )

            @contextlib.contextmanager
            def incomplete_dir(dirname):
                """Create temporary dir for dirname and rename on exit."""
                if not is_local:
                    self._fs.makedirs(dirname, exist_ok=True)
                    yield dirname
                else:
                    tmp_dir = dirname + ".incomplete"
                    os.makedirs(tmp_dir, exist_ok=True)
                    try:
                        yield tmp_dir
                        if os.path.isdir(dirname):
                            shutil.rmtree(dirname)
                        # LocalFileSystem.mv does copy + rm, it is more efficient to simply rename a local directory
                        shutil.move(tmp_dir, dirname)
                    finally:
                        if os.path.exists(tmp_dir):
                            shutil.rmtree(tmp_dir)

            # Print is intentional: we want this to always go to stdout so user has
            # information needed to cancel download/preparation if needed.
            # This comes right before the progress bar.
            if self.info.size_in_bytes:
                logger.info(
                    f"Downloading and preparing dataset {self.dataset_name}/{self.config.name} "
                    f"(download: {size_str(self.info.download_size)}, generated: {size_str(self.info.dataset_size)}, "
                    f"total: {size_str(self.info.size_in_bytes)}) to {self._output_dir}..."
                )
            else:
                _dest = self._fs._strip_protocol(self._output_dir) if is_local else self._output_dir
                logger.info(f"Downloading and preparing dataset {self.dataset_name}/{self.config.name} to {_dest}...")

            # Create a tmp dir and rename to self._output_dir on successful exit.
            with incomplete_dir(self._output_dir) as tmp_output_dir:
                # Temporarily assign _output_dir to tmp_data_dir to avoid having to forward
                # it to every sub function.
                with temporary_assignment(self, "_output_dir", tmp_output_dir):
                    prepare_split_kwargs = {"file_format": file_format}
                    if max_shard_size is not None:
                        prepare_split_kwargs["max_shard_size"] = max_shard_size
                    if num_proc is not None:
                        prepare_split_kwargs["num_proc"] = num_proc
                    self._download_and_prepare(
                        dl_manager=dl_manager,
                        verification_mode=verification_mode,
                        **prepare_split_kwargs,
                        **download_and_prepare_kwargs,
                    )
                    # Sync info
                    self.info.dataset_size = sum(split.num_bytes for split in self.info.splits.values())
                    if dl_manager.record_checksums:
                        self.info.download_checksums = dl_manager.get_recorded_sizes_checksums()
                    if self.info.download_size is not None:
                        self.info.size_in_bytes = self.info.dataset_size + self.info.download_size
                    # Save info
                    self._save_info()

            logger.info(
                f"Dataset {self.dataset_name} downloaded and prepared to {self._output_dir}. "
                f"Subsequent calls will reuse this data."
            )

    def _download_and_prepare(self, dl_manager, verification_mode, **prepare_split_kwargs):
        """Downloads and prepares dataset for reading.

        This is the internal implementation to overwrite called when user calls
        `download_and_prepare`. It should download all required data and generate
        the pre-processed datasets files.

        Args:
            dl_manager ([`DownloadManager`]):
                `DownloadManager` used to download and cache data.
            verification_mode ([`VerificationMode`]):
                if `ALL_CHECKS`, perform all the verifications including checksums.
                if `BASIC_CHECKS`, do not perform checksums, only perform split tests.
                if `NO_CHECKS`, do not perform any verification.
            prepare_split_kwargs: Additional options, such as `file_format`, `max_shard_size`
        """
        # Generating data for all splits
        split_dict = SplitDict(dataset_name=self.dataset_name)
        split_generators_kwargs = self._make_split_generators_kwargs(prepare_split_kwargs)
        split_generators: list[SplitGenerator] = self._split_generators(dl_manager, **split_generators_kwargs)

        # Checksums verification
        if verification_mode == VerificationMode.ALL_CHECKS and dl_manager.record_checksums:
            verify_checksums(
                self.info.download_checksums, dl_manager.get_recorded_sizes_checksums(), "dataset source files"
            )

        # Build splits
        for split_generator in split_generators:
            if str(split_generator.split_info.name).lower() == "all":
                raise ValueError(
                    "`all` is a special split keyword corresponding to the "
                    "union of all splits, so cannot be used as key in "
                    "._split_generator()."
                )

            logger.info(f"Generating {split_generator.split_info.name} split")
            split_dict.add(split_generator.split_info)

            try:
                # Prepare split will record examples associated to the split
                self._prepare_split(split_generator, **prepare_split_kwargs)
            except OSError as e:
                raise OSError("Cannot find data file. " + "\nOriginal error:\n" + str(e)) from None
            dl_manager.manage_extracted_files()

        if verification_mode == VerificationMode.BASIC_CHECKS or verification_mode == VerificationMode.ALL_CHECKS:
            verify_splits(self.info.splits, split_dict)

        # Update the info object with the splits.
        self.info.splits = split_dict
        self.info.download_size = dl_manager.downloaded_size

    def _load_info(self) -> DatasetInfo:
        return DatasetInfo.from_directory(self._output_dir, storage_options=self._fs.storage_options)

    def _save_info(self):
        file_lock = (
            FileLock(self._output_dir + "_info.lock")
            if not is_remote_filesystem(self._fs)
            else contextlib.nullcontext()
        )
        with file_lock:
            self.info.write_to_directory(self._output_dir, storage_options=self._fs.storage_options)

    def _make_split_generators_kwargs(self, prepare_split_kwargs):
        """Get kwargs for `self._split_generators()` from `prepare_split_kwargs`."""
        del prepare_split_kwargs
        return {}

    def as_dataset(
        self,
        split: Optional[Union[str, Split, list[str], list[Split]]] = None,
        in_memory=False,
    ) -> Union[Dataset, DatasetDict]:
        """Return a Dataset for the specified split.

        Args:
            split (`datasets.Split`):
                Which subset of the data to return.
            verification_mode ([`VerificationMode`] or `str`, defaults to `BASIC_CHECKS`):
                Verification mode determining the checks to run on the
                downloaded/processed dataset information (checksums/size/splits/...).

                <Added version="2.9.1"/>
            in_memory (`bool`, defaults to `False`):
                Whether to copy the data in-memory.

        Returns:
            datasets.Dataset

        Example:

        ```py
        >>> from datasets import load_dataset_builder
        >>> builder = load_dataset_builder('cornell-movie-review-data/rotten_tomatoes')
        >>> builder.download_and_prepare()
        >>> ds = builder.as_dataset(split='train')
        >>> ds
        Dataset({
            features: ['text', 'label'],
            num_rows: 8530
        })
        ```
        """
        if self._file_format is not None and self._file_format != "arrow":
            raise FileFormatError('Loading a dataset not written in the "arrow" format is not supported.')
        if is_remote_filesystem(self._fs):
            raise NotImplementedError(f"Loading a dataset cached in a {type(self._fs).__name__} is not supported.")
        if not os.path.exists(self._output_dir):
            raise FileNotFoundError(
                f"Dataset {self.dataset_name}: could not find data in {self._output_dir}. Please make sure to call "
                "builder.download_and_prepare(), or use "
                "datasets.load_dataset() before trying to access the Dataset object."
            )

        logger.debug(f"Constructing Dataset for split {split or ', '.join(self.info.splits)}, from {self._output_dir}")

        # By default, return all splits
        if split is None:
            split = {s: s for s in self.info.splits}

        # Create a dataset for each of the given splits
        datasets = map_nested(
            partial(
                self._build_single_dataset,
                in_memory=in_memory,
            ),
            split,
            map_tuple=True,
            disable_tqdm=True,
        )
        if isinstance(datasets, dict):
            datasets = DatasetDict(datasets)
        return datasets

    def _build_single_dataset(
        self,
        split: Union[str, ReadInstruction, Split],
        in_memory: bool = False,
    ):
        """as_dataset for a single split."""
        if not isinstance(split, ReadInstruction):
            split = str(split)
            if split == "all":
                split = "+".join(self.info.splits.keys())
            split = Split(split)

        # Build base dataset
        ds = self._as_dataset(
            split=split,
            in_memory=in_memory,
        )
        return ds

    def _as_dataset(self, split: Union[ReadInstruction, Split] = Split.TRAIN, in_memory: bool = False) -> Dataset:
        """Constructs a `Dataset`.

        This is the internal implementation to overwrite called when user calls
        `as_dataset`. It should read the pre-processed datasets files and generate
        the `Dataset` object.

        Args:
            split (`datasets.Split`):
                which subset of the data to read.
            in_memory (`bool`, defaults to `False`):
                Whether to copy the data in-memory.

        Returns:
            `Dataset`
        """
        cache_dir = self._fs._strip_protocol(self._output_dir)
        dataset_name = self.dataset_name
        if self._check_legacy_cache():
            dataset_name = self.name
        dataset_kwargs = ArrowReader(cache_dir, self.info).read(
            name=dataset_name,
            instructions=split,
            split_infos=self.info.splits.values(),
            in_memory=in_memory,
        )
        fingerprint = self._get_dataset_fingerprint(split)
        return Dataset(fingerprint=fingerprint, **dataset_kwargs)

    def _get_dataset_fingerprint(self, split: Union[ReadInstruction, Split]) -> str:
        """The dataset fingerprint is the hash of the relative directory dataset_name/config_name/version/hash, as well as the split specs."""
        hasher = Hasher()
        hasher.update(Path(self._relative_data_dir()).as_posix())
        hasher.update(str(split))  # for example: train, train+test, train[:10%], test[:33%](pct1_dropremainder)
        fingerprint = hasher.hexdigest()
        return fingerprint

    def as_streaming_dataset(
        self,
        split: Optional[str] = None,
        base_path: Optional[str] = None,
    ) -> Union[dict[str, IterableDataset], IterableDataset]:
        if is_remote_filesystem(self._fs):
            raise NotImplementedError(
                f"Loading a streaming dataset cached in a {type(self._fs).__name__} is not supported yet."
            )

        dl_manager = StreamingDownloadManager(
            base_path=base_path or self.base_path,
            download_config=DownloadConfig(token=self.token, storage_options=self.storage_options),
            dataset_name=self.dataset_name,
            data_dir=self.config.data_dir,
        )
        splits_generators = {sg.name: sg for sg in self._split_generators(dl_manager)}
        # By default, return all splits
        if split is None:
            splits_generator = splits_generators
        elif str(split) in splits_generators:
            splits_generator = splits_generators[str(split)]
        elif hasattr(split, "get_read_instruction"):
            split_dict = SplitDict({name: sg.split_info for name, sg in splits_generators.items()})
            split_instruction = split.get_read_instruction(split_dict)
            split_infos = split_instruction.get_list_sliced_split_info()
            if any(split_info.slice_value is not None for split_info in split_infos):
                raise ValueError(f"Bad split: {split}. Available splits: {list(splits_generators)}")
            return _concatenate_iterable_datasets(
                [
                    self._as_streaming_dataset_single(splits_generators[split_info.split_info.name])
                    for split_info in split_infos
                ],
                info=self.info,
                split=split,
            )
        elif str(split) == str(Split.ALL):
            return _concatenate_iterable_datasets(
                [self._as_streaming_dataset_single(sg) for sg in splits_generators.values()],
                info=self.info,
                split=split,
            )
        elif "+" in str(split) and "[" not in str(split) and "]" not in str(split):
            split_names = [split_name.strip() for split_name in str(split).split("+")]
            if not all(split_name in splits_generators for split_name in split_names):
                raise ValueError(f"Bad split: {split}. Available splits: {list(splits_generators)}")
            return _concatenate_iterable_datasets(
                [self._as_streaming_dataset_single(splits_generators[split_name]) for split_name in split_names],
                info=self.info,
                split=split,
            )
        else:
            raise ValueError(f"Bad split: {split}. Available splits: {list(splits_generators)}")

        # Create a dataset for each of the given splits
        datasets = map_nested(
            self._as_streaming_dataset_single,
            splits_generator,
            map_tuple=True,
        )
        if isinstance(datasets, dict):
            datasets = IterableDatasetDict(datasets)
        return datasets

    def _as_streaming_dataset_single(
        self,
        splits_generator,
    ) -> IterableDataset:
        ex_iterable = self._get_examples_iterable_for_split(splits_generator)
        # add auth to be able to access and decode audio/image files from private repositories.
        token_per_repo_id = {self.repo_id: self.token} if self.repo_id else {}
        return IterableDataset(
            ex_iterable, info=self.info, split=splits_generator.name, token_per_repo_id=token_per_repo_id
        )

    @abc.abstractmethod
    def _split_generators(self, dl_manager: Union[DownloadManager, StreamingDownloadManager]):
        """Specify feature dictionary generators and dataset splits.

        This function returns a list of `SplitGenerator`s defining how to generate
        data and what splits to use.

        Example:

            return [
                    datasets.SplitGenerator(
                            name=datasets.Split.TRAIN,
                            gen_kwargs={'file': 'train_data.zip'},
                    ),
                    datasets.SplitGenerator(
                            name=datasets.Split.TEST,
                            gen_kwargs={'file': 'test_data.zip'},
                    ),
            ]

        The above code will first call `_generate_examples(file='train_data.zip')`
        to write the train data, then `_generate_examples(file='test_data.zip')` to
        write the test data.

        Datasets are typically split into different subsets to be used at various
        stages of training and evaluation.

        Note that for datasets without a `VALIDATION` split, you can use a
        fraction of the `TRAIN` data for evaluation as you iterate on your model
        so as not to overfit to the `TEST` data.

        For downloads and extractions, use the given `download_manager`.
        Note that the `DownloadManager` caches downloads, so it is fine to have each
        generator attempt to download the source data.

        A good practice is to download all data in this function, and then
        distribute the relevant parts to each split with the `gen_kwargs` argument

        Args:
            dl_manager (`Union[DownloadManager, StreamingDownloadManager]`):
                Download manager to download the data

        Returns:
            `list<SplitGenerator>`.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _prepare_split(
        self,
        split_generator: SplitGenerator,
        file_format: str = "arrow",
        max_shard_size: Optional[Union[str, int]] = None,
        num_proc: Optional[int] = None,
        **kwargs,
    ):
        """Generate the examples and record them on disk.

        Args:
            split_generator (`SplitGenerator`):
                Split generator to process
            file_format (`str`, *optional*):
                format of the data files in which the dataset will be written.
                Supported formats: "arrow", "parquet". Default to "arrow" format.
            max_shard_size (`Union[str, int]`, *optional*):
                Maximum number of bytes written per shard, default is "500MB".
                The size is based on uncompressed data size, so in practice your shard files may be smaller than
                `max_shard_size` thanks to Parquet compression for example.
            num_proc (`int`, *optional*, defaults to `None`):
                Number of processes when downloading and generating the dataset locally.
                Multiprocessing is disabled by default.

                <Added version="2.7.0"/>
            **kwargs: Additional kwargs forwarded from _download_and_prepare
        """
        raise NotImplementedError()

    def _get_examples_iterable_for_split(self, split_generator: SplitGenerator) -> ExamplesIterable:
        """Generate the examples on the fly.

        Args:
            split_generator (`SplitGenerator`):
                Split generator to process
        """
        raise NotImplementedError()


@dataclass
class Key:
    original_shard_id: int
    item_or_batch_id: int

    def __str__(self):
        return str((self.original_shard_id, self.item_or_batch_id))


class GeneratorBasedBuilder(DatasetBuilder):
    """Base class for datasets with data generation based on dict generators.

    `GeneratorBasedBuilder` is a convenience class that abstracts away much
    of the data writing and reading of `DatasetBuilder`. It expects subclasses to
    implement generators of feature dictionaries across the dataset splits
    (`_split_generators`). See the method docstrings for details.
    """

    def _generate_shards(self, **kwargs) -> Iterator[Union[str, dict[str, Any]]]:
        """Default function generating shards paths for each `SplitGenerator`.

        This function is useful to list the original shards from where the data
        comes from and is either converted to Arrow or streamed to an IterableDataset.

        This is optional and only used for certain utilities, but not in Dataset
        nor IterableDataset. E.g. it's used to map original shard files to Parquet
        files in the Dataset Viewer after conversion.

        Args:
            **kwargs (additional keyword arguments):
                Arguments forwarded from the SplitGenerator.gen_kwargs

        Yields:
            shard: generally a string representing the shard path, or a dict
                representing the shard in case of shards spanning intra or inter-files.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _generate_examples(self, **kwargs) -> Iterator[tuple[Key, dict[str, Any]]]:
        """Default function generating examples for each `SplitGenerator`.

        This function preprocess the examples from the raw data to the preprocessed
        dataset files.
        This function is called once for each `SplitGenerator` defined in
        `_split_generators`. The examples yielded here will be written on
        disk.

        Args:
            **kwargs (additional keyword arguments):
                Arguments forwarded from the SplitGenerator.gen_kwargs

        Yields:
            key: `str` or `int`, a unique deterministic example identification key.
                * Unique: An error will be raised if two examples are yield with the
                    same key.
                * Deterministic: When generating the dataset twice, the same example
                    should have the same key.
                Good keys can be the image id, or line number if examples are extracted
                from a text file.
                The key will be hashed and sorted to shuffle examples deterministically,
                such as generating the dataset multiple times keep examples in the
                same order.
            example: `dict<str feature_name, feature_value>`, a feature dictionary
                ready to be encoded and written to disk. The example will be
                encoded with `self.info.features.encode_example({...})`.
        """
        raise NotImplementedError()

    def _prepare_split(
        self,
        split_generator: SplitGenerator,
        file_format="arrow",
        num_proc: Optional[int] = None,
        max_shard_size: Optional[Union[int, str]] = None,
    ):
        max_shard_size = convert_file_size_to_int(max_shard_size or config.MAX_SHARD_SIZE)

        if self.info.splits is not None:
            split_info = self.info.splits[split_generator.name]
        else:
            split_info = split_generator.split_info

        SUFFIX = "-JJJJJ-SSSSS-of-NNNNN"
        fname = f"{self.dataset_name}-{split_generator.name}{SUFFIX}.{file_format}"
        fpath = posixpath.join(self._output_dir, fname)

        if num_proc and num_proc > 1:
            num_original_shards = _number_of_shards_in_gen_kwargs(split_generator.gen_kwargs)
            if num_original_shards <= 1:
                logger.warning(
                    f"Setting num_proc from {num_proc} back to 1 for the {split_info.name} split to disable multiprocessing as it only contains one shard."
                )
                num_proc = 1
            elif num_original_shards < num_proc:
                logger.warning(
                    f"Setting num_proc from {num_proc} to {num_original_shards} for the {split_info.name} split as it only contains {num_original_shards} shards."
                )
                num_proc = num_original_shards

        pbar = hf_tqdm(
            unit=" examples",
            total=split_info.num_examples,
            desc=f"Generating {split_info.name} split",
        )

        _prepare_split_args = {
            "fpath": fpath,
            "file_format": file_format,
            "max_shard_size": max_shard_size,
            "split_info": split_info,
        }

        if num_proc is None or num_proc == 1:
            result = None
            gen_kwargs = split_generator.gen_kwargs
            job_id = 0
            with pbar:
                for job_id, done, content in self._prepare_split_single(
                    gen_kwargs=gen_kwargs, job_id=job_id, **_prepare_split_args
                ):
                    if done:
                        result = content
                    else:
                        pbar.update(content)
            # wrapping everything into lists for consistency with the multiprocessed code path
            assert result is not None, "Failed to retrieve results from prepare_split"
            (
                examples_per_job,
                bytes_per_job,
                features_per_job,
                shards_per_job,
                shard_lengths_per_job,
                original_shards_per_job,
                original_shard_lengths_per_job,
            ) = ([item] for item in result)
        else:
            kwargs_per_job = [
                {"gen_kwargs": gen_kwargs, "job_id": job_id, **_prepare_split_args}
                for job_id, gen_kwargs in enumerate(
                    _split_gen_kwargs(split_generator.gen_kwargs, max_num_jobs=num_proc)
                )
            ]
            num_jobs = len(kwargs_per_job)

            examples_per_job = [None] * num_jobs
            bytes_per_job = [None] * num_jobs
            features_per_job = [None] * num_jobs
            shards_per_job = [None] * num_jobs
            shard_lengths_per_job = [None] * num_jobs
            original_shards_per_job = [None] * num_jobs
            original_shard_lengths_per_job = [None] * num_jobs

            with Pool(num_proc) as pool:
                with pbar:
                    for job_id, done, content in iflatmap_unordered(
                        pool, self._prepare_split_single, kwargs_iterable=kwargs_per_job
                    ):
                        if done:
                            # the content is the result of the job
                            (
                                examples_per_job[job_id],
                                bytes_per_job[job_id],
                                features_per_job[job_id],
                                shards_per_job[job_id],
                                shard_lengths_per_job[job_id],
                                original_shards_per_job[job_id],
                                original_shard_lengths_per_job[job_id],
                            ) = content
                        else:
                            # the content is the number of examples progress update
                            pbar.update(content)

            assert None not in examples_per_job, (
                f"Failed to retrieve results from prepare_split: result list {examples_per_job} still contains None - at least one worker failed to return its results"
            )

        total_shards = sum(shards_per_job)
        total_original_shards = sum(original_shards_per_job)
        total_num_examples = sum(examples_per_job)
        total_num_bytes = sum(bytes_per_job)
        features = features_per_job[0]

        split_generator.split_info.num_examples = total_num_examples
        split_generator.split_info.num_bytes = total_num_bytes

        # should rename everything at the end
        logger.debug(f"Renaming {total_shards} shards.")
        if total_shards > 1:
            # use the -SSSSS-of-NNNNN pattern

            def _rename_shard(shard_and_job: tuple[int]):
                shard_id, job_id = shard_and_job
                global_shard_id = sum(shards_per_job[:job_id]) + shard_id
                self._rename(
                    fpath.replace("SSSSS", f"{shard_id:05d}").replace("JJJJJ", f"{job_id:05d}"),
                    fpath.replace("JJJJJ-SSSSS", f"{global_shard_id:05d}").replace("NNNNN", f"{total_shards:05d}"),
                )

            shards_and_jobs = [
                (shard_id, job_id)
                for job_id, num_shards in enumerate(shards_per_job)
                for shard_id in range(num_shards)
            ]
            thread_map(_rename_shard, shards_and_jobs, disable=True, max_workers=64)

            split_generator.split_info.shard_lengths = [
                shard_length for shard_lengths in shard_lengths_per_job for shard_length in shard_lengths
            ]
        else:
            # don't use any pattern
            shard_id, job_id = 0, 0
            self._rename(
                fpath.replace("SSSSS", f"{shard_id:05d}").replace("JJJJJ", f"{job_id:05d}"),
                fpath.replace(SUFFIX, ""),
            )

        if total_original_shards > 1 and config.SAVE_ORIGINAL_SHARD_LENGTHS:
            split_generator.split_info.original_shard_lengths = [
                original_shard_length
                for original_shard_lengths in original_shard_lengths_per_job
                for original_shard_length in original_shard_lengths
            ]

        if self.info.features is None:
            self.info.features = features

    def _prepare_split_single(
        self,
        gen_kwargs: dict,
        fpath: str,
        file_format: str,
        max_shard_size: int,
        split_info: SplitInfo,
        job_id: int,
    ) -> Iterator[tuple[int, bool, tuple[int, int, Features, int, int, int]]]:
        generator = self._generate_examples(**gen_kwargs)
        writer_class = ParquetWriter if file_format == "parquet" else ArrowWriter
        embed_local_files = file_format == "parquet"
        shard_lengths = []
        original_shard_lengths = []
        total_num_examples, total_num_bytes = 0, 0

        shard_id = 0
        original_shard_id = 0
        num_examples_progress_update = 0
        try:
            writer = writer_class(
                features=self.info.features,
                path=fpath.replace("SSSSS", f"{shard_id:05d}").replace("JJJJJ", f"{job_id:05d}"),
                writer_batch_size=self._writer_batch_size,
                storage_options=self._fs.storage_options,
                embed_local_files=embed_local_files,
            )
            try:
                _time = time.time()
                for key, record in generator:
                    if isinstance(key, Key):  # old custom builders may not use Key
                        original_shard_id = key.original_shard_id
                    if max_shard_size is not None and writer._num_bytes > max_shard_size:
                        num_examples, num_bytes = writer.finalize()
                        writer.close()
                        shard_lengths.append(num_examples)
                        total_num_examples += num_examples
                        total_num_bytes += num_bytes
                        shard_id += 1
                        writer = writer_class(
                            features=writer._features,
                            path=fpath.replace("SSSSS", f"{shard_id:05d}").replace("JJJJJ", f"{job_id:05d}"),
                            writer_batch_size=self._writer_batch_size,
                            storage_options=self._fs.storage_options,
                            embed_local_files=embed_local_files,
                        )
                    example = self.info.features.encode_example(record) if self.info.features is not None else record
                    writer.write(example)
                    if len(original_shard_lengths) <= original_shard_id:
                        original_shard_lengths.extend([0] * (1 + original_shard_id - len(original_shard_lengths)))
                    original_shard_lengths[original_shard_id] += 1
                    num_examples_progress_update += 1
                    if time.time() > _time + config.PBAR_REFRESH_TIME_INTERVAL:
                        _time = time.time()
                        yield job_id, False, num_examples_progress_update
                        num_examples_progress_update = 0
            finally:
                yield job_id, False, num_examples_progress_update
                num_shards = shard_id + 1
                num_original_shards = original_shard_id + 1
                num_examples, num_bytes = writer.finalize()
                writer.close()
                shard_lengths.append(num_examples)
                total_num_examples += num_examples
                total_num_bytes += num_bytes
        except Exception as e:
            # Ignore the writer's error for no examples written to the file if this error was caused by the error in _generate_examples before the first example was yielded
            if isinstance(e, SchemaInferenceError) and e.__context__ is not None:
                e = e.__context__
            raise DatasetGenerationError("An error occurred while generating the dataset") from e

        yield (
            job_id,
            True,
            (
                total_num_examples,
                total_num_bytes,
                writer._features,
                num_shards,
                shard_lengths,
                num_original_shards,
                original_shard_lengths,
            ),
        )

    def _download_and_prepare(self, dl_manager, verification_mode, **prepare_splits_kwargs):
        super()._download_and_prepare(
            dl_manager,
            verification_mode,
            **prepare_splits_kwargs,
        )

    def _get_examples_iterable_for_split(self, split_generator: SplitGenerator) -> ExamplesIterable:
        return ExamplesIterable(
            self._generate_examples,
            split_generator.gen_kwargs,
            generate_more_kwargs_fn=getattr(self, "_generate_more_gen_kwargs", None),
        )


class ArrowBasedBuilder(DatasetBuilder):
    """Base class for datasets with data generation based on Arrow loading functions (CSV/JSON/Parquet)."""

    def _generate_shards(self, **kwargs) -> Iterator[Union[str, dict[str, Any]]]:
        """Default function generating shards paths for each `SplitGenerator`.

        This function is useful to list the original shards from where the data
        comes from and is either converted to Arrow or streamed to an IterableDataset.

        This is optional and only used for certain utilities, but not in Dataset
        nor IterableDataset. E.g. it's used to map original shard files to Parquet
        files in the Dataset Viewer after conversion.

        Args:
            **kwargs (additional keyword arguments):
                Arguments forwarded from the SplitGenerator.gen_kwargs

        Yields:
            shard: generally a string representing the shard path, or a dict
                representing the shard in case of shards spanning intra or inter-files.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _generate_tables(self, **kwargs) -> Iterator[tuple[Key, pa.Table]]:
        """Default function generating examples for each `SplitGenerator`.

        This function preprocess the examples from the raw data to the preprocessed
        dataset files.
        This function is called once for each `SplitGenerator` defined in
        `_split_generators`. The examples yielded here will be written on
        disk.

        Args:
            **kwargs (additional keyword arguments):
                Arguments forwarded from the SplitGenerator.gen_kwargs

        Yields:
            key: tuple[int, int] original_shard_id and table_idx within that shard
            example: `pyarrow.Table`, a feature table
                ready to be encoded and written to disk.
        """
        raise NotImplementedError()

    def _prepare_split(
        self,
        split_generator: SplitGenerator,
        file_format: str = "arrow",
        num_proc: Optional[int] = None,
        max_shard_size: Optional[Union[str, int]] = None,
    ):
        max_shard_size = convert_file_size_to_int(max_shard_size or config.MAX_SHARD_SIZE)

        try:
            split_info = self.info.splits[split_generator.name]
        except Exception:
            split_info = split_generator.split_info

        SUFFIX = "-JJJJJ-SSSSS-of-NNNNN"
        fname = f"{self.dataset_name}-{split_generator.name}{SUFFIX}.{file_format}"
        fpath = posixpath.join(self._output_dir, fname)

        if num_proc and num_proc > 1:
            num_original_shards = _number_of_shards_in_gen_kwargs(split_generator.gen_kwargs)
            if num_original_shards <= 1:
                logger.warning(
                    f"Setting num_proc from {num_proc} back to 1 for the {split_info.name} split to disable multiprocessing as it only contains one shard."
                )
                num_proc = 1
            elif num_original_shards < num_proc:
                logger.warning(
                    f"Setting num_proc from {num_proc} to {num_original_shards} for the {split_info.name} split as it only contains {num_original_shards} shards."
                )
                num_proc = num_original_shards

        pbar = hf_tqdm(
            unit=" examples",
            total=split_info.num_examples,
            desc=f"Generating {split_info.name} split",
        )

        _prepare_split_args = {
            "fpath": fpath,
            "file_format": file_format,
            "max_shard_size": max_shard_size,
        }

        if num_proc is None or num_proc == 1:
            result = None
            gen_kwargs = split_generator.gen_kwargs
            job_id = 0
            with pbar:
                for job_id, done, content in self._prepare_split_single(
                    gen_kwargs=gen_kwargs, job_id=job_id, **_prepare_split_args
                ):
                    if done:
                        result = content
                    else:
                        pbar.update(content)
            # wrapping everything into lists for consistency with the multiprocessed code path
            assert result is not None, "Failed to retrieve results from prepare_split"
            (
                examples_per_job,
                bytes_per_job,
                features_per_job,
                shards_per_job,
                shard_lengths_per_job,
                original_shards_per_job,
                original_shard_lengths_per_job,
            ) = ([item] for item in result)
        else:
            kwargs_per_job = [
                {"gen_kwargs": gen_kwargs, "job_id": job_id, **_prepare_split_args}
                for job_id, gen_kwargs in enumerate(
                    _split_gen_kwargs(split_generator.gen_kwargs, max_num_jobs=num_proc)
                )
            ]
            num_jobs = len(kwargs_per_job)

            examples_per_job = [None] * num_jobs
            bytes_per_job = [None] * num_jobs
            features_per_job = [None] * num_jobs
            shards_per_job = [None] * num_jobs
            shard_lengths_per_job = [None] * num_jobs
            original_shards_per_job = [None] * num_jobs
            original_shard_lengths_per_job = [None] * num_jobs

            with Pool(num_proc) as pool:
                with pbar:
                    for job_id, done, content in iflatmap_unordered(
                        pool, self._prepare_split_single, kwargs_iterable=kwargs_per_job
                    ):
                        if done:
                            # the content is the result of the job
                            (
                                examples_per_job[job_id],
                                bytes_per_job[job_id],
                                features_per_job[job_id],
                                shards_per_job[job_id],
                                shard_lengths_per_job[job_id],
                                original_shards_per_job[job_id],
                                original_shard_lengths_per_job[job_id],
                            ) = content
                        else:
                            # the content is the number of examples progress update
                            pbar.update(content)

            assert None not in examples_per_job, (
                f"Failed to retrieve results from prepare_split: result list {examples_per_job} still contains None - at least one worker failed to return its results"
            )

        total_shards = sum(shards_per_job)
        total_original_shards = sum(original_shards_per_job)
        total_num_examples = sum(examples_per_job)
        total_num_bytes = sum(bytes_per_job)
        features = features_per_job[0]

        split_generator.split_info.num_examples = total_num_examples
        split_generator.split_info.num_bytes = total_num_bytes

        # should rename everything at the end
        logger.debug(f"Renaming {total_shards} shards.")
        if total_shards > 1:
            # use the -SSSSS-of-NNNNN pattern

            def _rename_shard(shard_id_and_job: tuple[int]):
                shard_id, job_id = shard_id_and_job
                global_shard_id = sum(shards_per_job[:job_id]) + shard_id
                self._rename(
                    fpath.replace("SSSSS", f"{shard_id:05d}").replace("JJJJJ", f"{job_id:05d}"),
                    fpath.replace("JJJJJ-SSSSS", f"{global_shard_id:05d}").replace("NNNNN", f"{total_shards:05d}"),
                )

            shard_ids_and_jobs = [
                (shard_id, job_id)
                for job_id, num_shards in enumerate(shards_per_job)
                for shard_id in range(num_shards)
            ]
            thread_map(_rename_shard, shard_ids_and_jobs, disable=True, max_workers=64)

            split_generator.split_info.shard_lengths = [
                shard_length for shard_lengths in shard_lengths_per_job for shard_length in shard_lengths
            ]
        else:
            # don't use any pattern
            shard_id, job_id = 0, 0
            self._rename(
                fpath.replace("SSSSS", f"{shard_id:05d}").replace("JJJJJ", f"{job_id:05d}"),
                fpath.replace(SUFFIX, ""),
            )

        if total_original_shards > 1 and config.SAVE_ORIGINAL_SHARD_LENGTHS:
            split_generator.split_info.original_shard_lengths = [
                original_shard_length
                for original_shard_lengths in original_shard_lengths_per_job
                for original_shard_length in original_shard_lengths
            ]

        if self.info.features is None:
            self.info.features = features

    def _prepare_split_single(
        self, gen_kwargs: dict, fpath: str, file_format: str, max_shard_size: int, job_id: int
    ) -> Iterator[tuple[int, bool, tuple[int, int, Features, int, int, int]]]:
        gen_kwargs = {k: tracked_list(v) if isinstance(v, list) else v for k, v in gen_kwargs.items()}
        generator = self._generate_tables(**gen_kwargs)
        writer_class = ParquetWriter if file_format == "parquet" else ArrowWriter
        embed_local_files = file_format == "parquet"
        shard_lengths = []
        original_shard_lengths = []
        total_num_examples, total_num_bytes = 0, 0

        shard_id = 0
        original_shard_id = 0
        num_examples_progress_update = 0
        try:
            writer = writer_class(
                features=self.info.features,
                path=fpath.replace("SSSSS", f"{shard_id:05d}").replace("JJJJJ", f"{job_id:05d}"),
                writer_batch_size=self._writer_batch_size,
                storage_options=self._fs.storage_options,
                embed_local_files=embed_local_files,
            )
            try:
                _time = time.time()
                for key, table in generator:
                    if isinstance(key, Key):  # old custom builders may not use Key
                        original_shard_id = key.original_shard_id
                    if max_shard_size is not None and writer._num_bytes > max_shard_size:
                        num_examples, num_bytes = writer.finalize()
                        writer.close()
                        shard_lengths.append(num_examples)
                        total_num_examples += num_examples
                        total_num_bytes += num_bytes
                        shard_id += 1
                        writer = writer_class(
                            features=writer._features,
                            path=fpath.replace("SSSSS", f"{shard_id:05d}").replace("JJJJJ", f"{job_id:05d}"),
                            writer_batch_size=self._writer_batch_size,
                            storage_options=self._fs.storage_options,
                            embed_local_files=embed_local_files,
                        )
                    try:
                        if len(table) == 1:
                            writer.write_row(table)
                        else:
                            writer.write_table(table)
                    except CastError as cast_error:
                        raise DatasetGenerationCastError.from_cast_error(
                            cast_error=cast_error,
                            builder_name=self.info.builder_name,
                            gen_kwargs=gen_kwargs,
                            token=self.token,
                        )
                    if len(original_shard_lengths) <= original_shard_id:
                        original_shard_lengths.extend([0] * (1 + original_shard_id - len(original_shard_lengths)))
                    original_shard_lengths[original_shard_id] += len(table)
                    num_examples_progress_update += len(table)
                    if time.time() > _time + config.PBAR_REFRESH_TIME_INTERVAL:
                        _time = time.time()
                        yield job_id, False, num_examples_progress_update
                        num_examples_progress_update = 0
            finally:
                yield job_id, False, num_examples_progress_update
                num_shards = shard_id + 1
                num_original_shards = original_shard_id + 1
                num_examples, num_bytes = writer.finalize()
                writer.close()
                shard_lengths.append(num_examples)
                total_num_examples += num_examples
                total_num_bytes += num_bytes
        except Exception as e:
            # Ignore the writer's error for no examples written to the file if this error was caused by the error in _generate_examples before the first example was yielded
            if isinstance(e, SchemaInferenceError) and e.__context__ is not None:
                e = e.__context__
            if isinstance(e, DatasetGenerationError):
                raise
            raise DatasetGenerationError("An error occurred while generating the dataset") from e

        yield (
            job_id,
            True,
            (
                total_num_examples,
                total_num_bytes,
                writer._features,
                num_shards,
                shard_lengths,
                num_original_shards,
                original_shard_lengths,
            ),
        )

    def _get_examples_iterable_for_split(self, split_generator: SplitGenerator) -> ExamplesIterable:
        return ArrowExamplesIterable(
            self._generate_tables,
            kwargs=split_generator.gen_kwargs,
            generate_more_kwargs_fn=getattr(self, "_generate_more_gen_kwargs", None),
            sleep_on_threads_shutdown=self.SLEEP_ON_THREADS_SHUTDOWNS,
        )


class _CountableBuilderMixin(DatasetBuilder):
    @abc.abstractmethod
    def _generate_num_examples(self, **kwargs) -> Iterator[int]:
        raise NotImplementedError()

    def count_examples(self, dl_manager: DownloadManager) -> dict[str, int]:
        split_generators_kwargs = self._make_split_generators_kwargs({})
        split_generators: list[SplitGenerator] = self._split_generators(dl_manager, **split_generators_kwargs)
        return {split_generator.name: self._count_examples(split_generator) for split_generator in split_generators}

    def _count_examples(self, split_generator: SplitGenerator) -> int:
        max_workers = min(32, os.cpu_count() + 4)
        return sum(
            thread_map(
                self._count_examples_single,
                _split_gen_kwargs(split_generator.gen_kwargs, max_workers),
                delay=5,
                desc=f"Counting rows for split={split_generator.name}",
            )
        )

    def _count_examples_single(self, gen_kwargs: dict[str, Any]) -> int:
        return sum(self._generate_num_examples(**gen_kwargs))
<|endoftext|># Copyright 2020 The HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""List and inspect datasets."""

import os
from collections.abc import Mapping, Sequence
from typing import Optional, Union

from .download.download_config import DownloadConfig
from .download.download_manager import DownloadMode
from .download.streaming_download_manager import StreamingDownloadManager
from .info import DatasetInfo
from .load import (
    dataset_module_factory,
    get_dataset_builder_class,
    load_dataset_builder,
)
from .utils.logging import get_logger
from .utils.version import Version


logger = get_logger(__name__)


class SplitsNotFoundError(ValueError):
    pass


def get_dataset_infos(
    path: str,
    data_files: Optional[Union[dict, list, str]] = None,
    download_config: Optional[DownloadConfig] = None,
    download_mode: Optional[Union[DownloadMode, str]] = None,
    revision: Optional[Union[str, Version]] = None,
    token: Optional[Union[bool, str]] = None,
    **config_kwargs,
):
    """Get the meta information about a dataset, returned as a dict mapping config name to DatasetInfoDict.

    Args:
        path (`str`): path to the dataset repository. Can be either:

            - a local path to the dataset directory containing the data files,
                e.g. `'./dataset/squad'`
            - a dataset identifier on the Hugging Face Hub (list all available datasets and ids with [`huggingface_hub.list_datasets`]),
                e.g. `'rajpurkar/squad'`, `'nyu-mll/glue'` or``'openai/webtext'`
        revision (`Union[str, datasets.Version]`, *optional*):
            If specified, the dataset module will be loaded from the datasets repository at this version.
            By default:
            - it is set to the local version of the lib.
            - it will also try to load it from the main branch if it's not available at the local version of the lib.
            Specifying a version that is different from your local version of the lib might cause compatibility issues.
        download_config ([`DownloadConfig`], *optional*):
            Specific download configuration parameters.
        download_mode ([`DownloadMode`] or `str`, defaults to `REUSE_DATASET_IF_EXISTS`):
            Download/generate mode.
        data_files (`Union[Dict, List, str]`, *optional*):
            Defining the data_files of the dataset configuration.
        token (`str` or `bool`, *optional*):
            Optional string or boolean to use as Bearer token for remote files on the Datasets Hub.
            If `True`, or not specified, will get token from `"~/.huggingface"`.
        **config_kwargs (additional keyword arguments):
            Optional attributes for builder class which will override the attributes if supplied.

    Example:

    ```py
    >>> from datasets import get_dataset_infos
    >>> get_dataset_infos('cornell-movie-review-data/rotten_tomatoes')
    {'default': DatasetInfo(description="Movie Review Dataset.\nThis is a dataset of containing 5,331 positive and 5,331 negative processed\nsentences from Rotten Tomatoes movie reviews...), ...}
    ```
    """
    config_names = get_dataset_config_names(
        path=path,
        revision=revision,
        download_config=download_config,
        download_mode=download_mode,
        data_files=data_files,
        token=token,
    )
    return {
        config_name: get_dataset_config_info(
            path=path,
            config_name=config_name,
            data_files=data_files,
            download_config=download_config,
            download_mode=download_mode,
            revision=revision,
            token=token,
            **config_kwargs,
        )
        for config_name in config_names
    }


def get_dataset_config_names(
    path: str,
    revision: Optional[Union[str, Version]] = None,
    download_config: Optional[DownloadConfig] = None,
    download_mode: Optional[Union[DownloadMode, str]] = None,
    data_files: Optional[Union[dict, list, str]] = None,
    **download_kwargs,
):
    """Get the list of available config names for a particular dataset.

    Args:
        path (`str`): path to the dataset repository. Can be either:

            - a local path to the dataset directory containing the data files,
                e.g. `'./dataset/squad'`
            - a dataset identifier on the Hugging Face Hub (list all available datasets and ids with [`huggingface_hub.list_datasets`]),
                e.g. `'rajpurkar/squad'`, `'nyu-mll/glue'` or``'openai/webtext'`
        revision (`Union[str, datasets.Version]`, *optional*):
            If specified, the dataset module will be loaded from the datasets repository at this version.
            By default:
            - it is set to the local version of the lib.
            - it will also try to load it from the main branch if it's not available at the local version of the lib.
            Specifying a version that is different from your local version of the lib might cause compatibility issues.
        download_config ([`DownloadConfig`], *optional*):
            Specific download configuration parameters.
        download_mode ([`DownloadMode`] or `str`, defaults to `REUSE_DATASET_IF_EXISTS`):
            Download/generate mode.
        data_files (`Union[Dict, List, str]`, *optional*):
            Defining the data_files of the dataset configuration.
        **download_kwargs (additional keyword arguments):
            Optional attributes for [`DownloadConfig`] which will override the attributes in `download_config` if supplied,
            for example `token`.

    Example:

    ```py
    >>> from datasets import get_dataset_config_names
    >>> get_dataset_config_names("nyu-mll/glue")
    ['cola',
     'sst2',
     'mrpc',
     'qqp',
     'stsb',
     'mnli',
     'mnli_mismatched',
     'mnli_matched',
     'qnli',
     'rte',
     'wnli',
     'ax']
    ```
    """
    dataset_module = dataset_module_factory(
        path,
        revision=revision,
        download_config=download_config,
        download_mode=download_mode,
        data_files=data_files,
        **download_kwargs,
    )
    builder_cls = get_dataset_builder_class(dataset_module, dataset_name=os.path.basename(path))
    return list(builder_cls.builder_configs.keys()) or [
        dataset_module.builder_kwargs.get("config_name", builder_cls.DEFAULT_CONFIG_NAME or "default")
    ]


def get_dataset_default_config_name(
    path: str,
    revision: Optional[Union[str, Version]] = None,
    download_config: Optional[DownloadConfig] = None,
    download_mode: Optional[Union[DownloadMode, str]] = None,
    data_files: Optional[Union[dict, list, str]] = None,
    **download_kwargs,
) -> Optional[str]:
    """Get the default config name for a particular dataset.
    Can return None only if the dataset has multiple configurations and no default configuration.

    Args:
        path (`str`): path to the dataset repository. Can be either:

            - a local path to the dataset directory containing the data files,
                e.g. `'./dataset/squad'`
            - a dataset identifier on the Hugging Face Hub (list all available datasets and ids with [`huggingface_hub.list_datasets`]),
                e.g. `'rajpurkar/squad'`, `'nyu-mll/glue'` or``'openai/webtext'`
        revision (`Union[str, datasets.Version]`, *optional*):
            If specified, the dataset module will be loaded from the datasets repository at this version.
            By default:
            - it is set to the local version of the lib.
            - it will also try to load it from the main branch if it's not available at the local version of the lib.
            Specifying a version that is different from your local version of the lib might cause compatibility issues.
        download_config ([`DownloadConfig`], *optional*):
            Specific download configuration parameters.
        download_mode ([`DownloadMode`] or `str`, defaults to `REUSE_DATASET_IF_EXISTS`):
            Download/generate mode.
        data_files (`Union[Dict, List, str]`, *optional*):
            Defining the data_files of the dataset configuration.
        **download_kwargs (additional keyword arguments):
            Optional attributes for [`DownloadConfig`] which will override the attributes in `download_config` if supplied,
            for example `token`.

    Returns:
        Optional[str]: the default config name if there is one

    Example:

    ```py
    >>> from datasets import get_dataset_default_config_name
    >>> get_dataset_default_config_name("openbookqa")
    'main'
    ```
    """
    dataset_module = dataset_module_factory(
        path,
        revision=revision,
        download_config=download_config,
        download_mode=download_mode,
        data_files=data_files,
        **download_kwargs,
    )
    builder_cls = get_dataset_builder_class(dataset_module, dataset_name=os.path.basename(path))
    builder_configs = list(builder_cls.builder_configs.keys())
    if builder_configs:
        default_config_name = builder_configs[0] if len(builder_configs) == 1 else None
    else:
        default_config_name = "default"
    return builder_cls.DEFAULT_CONFIG_NAME or default_config_name


def get_dataset_config_info(
    path: str,
    config_name: Optional[str] = None,
    data_files: Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]] = None,
    download_config: Optional[DownloadConfig] = None,
    download_mode: Optional[Union[DownloadMode, str]] = None,
    revision: Optional[Union[str, Version]] = None,
    token: Optional[Union[bool, str]] = None,
    **config_kwargs,
) -> DatasetInfo:
    """Get the meta information (DatasetInfo) about a dataset for a particular config

    Args:
        path (`str`): path to the dataset repository. Can be either:

            - a local path to the dataset directory containing the data files,
                e.g. `'./dataset/squad'`
            - a dataset identifier on the Hugging Face Hub (list all available datasets and ids with [`huggingface_hub.list_datasets`]),
                e.g. `'rajpurkar/squad'`, `'nyu-mll/glue'` or``'openai/webtext'`
        config_name (:obj:`str`, optional): Defining the name of the dataset configuration.
        data_files (:obj:`str` or :obj:`Sequence` or :obj:`Mapping`, optional): Path(s) to source data file(s).
        download_config (:class:`~download.DownloadConfig`, optional): Specific download configuration parameters.
        download_mode (:class:`DownloadMode` or :obj:`str`, default ``REUSE_DATASET_IF_EXISTS``): Download/generate mode.
        revision (:class:`~utils.Version` or :obj:`str`, optional): Version of the dataset to load.
            As datasets have their own git repository on the Datasets Hub, the default version "main" corresponds to their "main" branch.
            You can specify a different version than the default "main" by using a commit SHA or a git tag of the dataset repository.
        token (``str`` or :obj:`bool`, optional): Optional string or boolean to use as Bearer token for remote files on the Datasets Hub.
            If True, or not specified, will get token from `"~/.huggingface"`.
        **config_kwargs (additional keyword arguments): optional attributes for builder class which will override the attributes if supplied.

    """
    builder = load_dataset_builder(
        path,
        name=config_name,
        data_files=data_files,
        download_config=download_config,
        download_mode=download_mode,
        revision=revision,
        token=token,
        **config_kwargs,
    )
    info = builder.info
    if info.splits is None:
        download_config = download_config.copy() if download_config else DownloadConfig()
        if token is not None:
            download_config.token = token
        try:
            info.splits = {
                split_generator.name: {"name": split_generator.name, "dataset_name": path}
                for split_generator in builder._split_generators(
                    StreamingDownloadManager(base_path=builder.base_path, download_config=download_config)
                )
            }
        except Exception as err:
            raise SplitsNotFoundError("The split names could not be parsed from the dataset config.") from err
    return info


def get_dataset_split_names(
    path: str,
    config_name: Optional[str] = None,
    data_files: Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]] = None,
    download_config: Optional[DownloadConfig] = None,
    download_mode: Optional[Union[DownloadMode, str]] = None,
    revision: Optional[Union[str, Version]] = None,
    token: Optional[Union[bool, str]] = None,
    **config_kwargs,
):
    """Get the list of available splits for a particular config and dataset.

    Args:
        path (`str`): path to the dataset repository. Can be either:

            - a local path to the dataset directory containing the data files,
                e.g. `'./dataset/squad'`
            - a dataset identifier on the Hugging Face Hub (list all available datasets and ids with [`huggingface_hub.list_datasets`]),
                e.g. `'rajpurkar/squad'`, `'nyu-mll/glue'` or``'openai/webtext'`
        config_name (`str`, *optional*):
            Defining the name of the dataset configuration.
        data_files (`str` or `Sequence` or `Mapping`, *optional*):
            Path(s) to source data file(s).
        download_config ([`DownloadConfig`], *optional*):
            Specific download configuration parameters.
        download_mode ([`DownloadMode`] or `str`, defaults to `REUSE_DATASET_IF_EXISTS`):
            Download/generate mode.
        revision ([`Version`] or `str`, *optional*):
            Version of the dataset to load.
            As datasets have their own git repository on the Datasets Hub, the default version "main" corresponds to their "main" branch.
            You can specify a different version than the default "main" by using a commit SHA or a git tag of the dataset repository.
        token (`str` or `bool`, *optional*):
            Optional string or boolean to use as Bearer token for remote files on the Datasets Hub.
            If `True`, or not specified, will get token from `"~/.huggingface"`.
        **config_kwargs (additional keyword arguments):
            Optional attributes for builder class which will override the attributes if supplied.

    Example:

    ```py
    >>> from datasets import get_dataset_split_names
    >>> get_dataset_split_names('cornell-movie-review-data/rotten_tomatoes')
    ['train', 'validation', 'test']
    ```
    """
    info = get_dataset_config_info(
        path,
        config_name=config_name,
        data_files=data_files,
        download_config=download_config,
        download_mode=download_mode,
        revision=revision,
        token=token,
        **config_kwargs,
    )
    return list(info.splits.keys())
<|endoftext|>import importlib
from functools import wraps
from typing import TYPE_CHECKING, Optional

from .download.download_config import DownloadConfig
from .utils.file_utils import (
    xbasename,
    xdirname,
    xet_parse,
    xexists,
    xgetsize,
    xglob,
    xgzip_open,
    xisdir,
    xisfile,
    xjoin,
    xlistdir,
    xnumpy_load,
    xopen,
    xpandas_read_csv,
    xpandas_read_excel,
    xPath,
    xpyarrow_parquet_read_table,
    xrelpath,
    xsio_loadmat,
    xsplit,
    xsplitext,
    xwalk,
    xxml_dom_minidom_parse,
)
from .utils.logging import get_logger
from .utils.patching import patch_submodule


logger = get_logger(__name__)


if TYPE_CHECKING:
    from .builder import DatasetBuilder


