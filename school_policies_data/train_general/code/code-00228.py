            if self.compressed_swizzled_bitmask is not None
            else None
        ),
        fuse_transpose_cusparselt=args[0].fuse_transpose_cusparselt,
        alg_id_cusparselt=args[0].alg_id_cusparselt,
    )


def semi_sparse_view(func, types, args=(), kwargs=None) -> torch.Tensor:
    if len(args) != 2:
        raise AssertionError(f"expected 2 args, got {len(args)}")
    self, shape = args
    if tuple(shape) != self.shape:
        raise NotImplementedError(
            f"`view` is not implemented for SparseSemiStructuredTensor, except for the dummy case (shape={shape})"
        )
    return self


def semi_sparse_detach(func, types, args, kwargs) -> torch.Tensor:
    if len(args) != 1:
        raise AssertionError(f"expected 1 arg, got {len(args)}")
    self = args[0]
    return self.__class__(
        shape=self.shape,
        packed=self.packed,
        meta=self.meta,
        packed_t=self.packed_t,
        meta_t=self.meta_t,
        compressed_swizzled_bitmask=self.compressed_swizzled_bitmask,
        fuse_transpose_cusparselt=self.fuse_transpose_cusparselt,
        alg_id_cusparselt=self.alg_id_cusparselt,
        requires_grad=False,
    )


def semi_sparse_mm(func, types, args=(), kwargs=None) -> torch.Tensor:
    if len(args) != 2:
        raise AssertionError(f"expected 2 args, got {len(args)}")
    A, B = args
    if A.ndim != 2 or B.ndim != 2:
        raise NotImplementedError(
            "`SparseSemiStructuredTensor` matmul: Broadcasting is not implemented"
        )
    if isinstance(A, torch.sparse.SparseSemiStructuredTensor):
        return A._mm(B)
    else:
        B_t = B.t()
        if not isinstance(B_t, torch.sparse.SparseSemiStructuredTensor):
            raise AssertionError(
                f"expected SparseSemiStructuredTensor, got {type(B_t).__name__}"
            )
        return B_t._mm(A, should_transpose_dense=True)


def semi_sparse_addmm(func, types, args=(), kwargs=None) -> torch.Tensor:
    if len(args) != 3:
        raise AssertionError(f"expected 3 args, got {len(args)}")
    bias, A, B = args
    if A.ndim != 2 or B.ndim != 2:
        raise NotImplementedError(
            "`SparseSemiStructuredTensor` matmul: Broadcasting is not implemented"
        )
    if bias.ndim != 1:
        raise NotImplementedError(
            f"`SparseSemiStructuredTensor` matmul: only bias dim=1 supported. Shape={bias.shape}"
        )
    if isinstance(A, torch.sparse.SparseSemiStructuredTensor):
        raise NotImplementedError(
            "`SparseSemiStructuredTensor` matmul: only operand B of `addmm` can be sparse"
        )
    B_t = B.t()
    if not isinstance(B_t, torch.sparse.SparseSemiStructuredTensor):
        raise AssertionError(
            f"expected SparseSemiStructuredTensor, got {type(B_t).__name__}"
        )
    row, _col = A.shape
    return B_t._mm(A, bias=bias, should_transpose_dense=True)


def semi_sparse_linear(func, types, args=(), kwargs=None) -> torch.Tensor:
    if len(args) not in [2, 3]:
        raise AssertionError(f"expected 2 or 3 args, got {len(args)}")
    A, B = args[:2]
    bias = args[2] if len(args) == 3 else None

    shape = A.shape
    A_2d = A.view(-1, shape[-1])
    if bias is None:
        res = A_2d @ B.t()
    else:
        res = semi_sparse_addmm(
            func=None,
            types=None,
            args=[bias, A_2d, B.t()],
        )
    return res.view(*shape[:-1], -1)


def semi_sparse_scaled_mm(func, types, args=(), kwargs=None) -> torch.Tensor:
    # pull all args, excluding use_fast_accum flag if set.
    A, B, A_scale, B_scale, bias, scale_result, out_dtype = args[:7]

    if A.dtype != torch.float8_e4m3fn:
        raise AssertionError(f"expected A.dtype float8_e4m3fn, got {A.dtype}")
    if B.dtype != torch.float8_e4m3fn:
        raise AssertionError(f"expected B.dtype float8_e4m3fn, got {B.dtype}")
    # only cuSPARSELt supports float8_e4m3fn currently
    if not isinstance(A, torch.sparse.SparseSemiStructuredTensorCUSPARSELT):
        raise AssertionError(
            f"expected SparseSemiStructuredTensorCUSPARSELT, got {type(A).__name__}"
        )
    if A.packed is None:
        raise AssertionError("A.packed must not be None")
    # Currently we only support per-tensor scaling, with float32 scales
    if A_scale.numel() != 1 or B_scale.numel() != 1:
        raise AssertionError(
            f"expected A_scale and B_scale to have numel 1, got {A_scale.numel()} and {B_scale.numel()}"
        )
    if A_scale.dtype != torch.float32 or B_scale.dtype != torch.float32:
        raise AssertionError(
            f"expected A_scale and B_scale dtype float32, got {A_scale.dtype} and {B_scale.dtype}"
        )

    # cuSPARSELt lacks the A and B operand scaling support, so instead we use alpha to scale the result.
    # Note that this limits us to per-tensor scaling only.
    sparse_result = torch._cslt_sparse_mm(
        A.packed,
        B,
        alpha=A_scale * B_scale,
        out_dtype=out_dtype,
    )
    return sparse_result


def semi_sparse_clone(func, types, args=(), kwargs=None) -> torch.Tensor:
    if len(args) != 1:
        raise AssertionError(f"expected 1 arg, got {len(args)}")

    self = args[0]
    if not isinstance(self, torch.sparse.SparseSemiStructuredTensor):
        raise AssertionError(
            f"expected SparseSemiStructuredTensor, got {type(self).__name__}"
        )

    # pyrefly: ignore [no-matching-overload]
    return self.__class__(
        shape=self.shape,
        packed=None if self.packed is None else self.packed.clone(),
        meta=None if self.meta is None else self.meta.clone(),
        packed_t=None if self.packed_t is None else self.packed_t.clone(),
        meta_t=None if self.meta_t is None else self.meta_t.clone(),
        compressed_swizzled_bitmask=(
            None
            if self.compressed_swizzled_bitmask is None
            else self.compressed_swizzled_bitmask.clone()
        ),
        fuse_transpose_cusparselt=self.fuse_transpose_cusparselt,
        alg_id_cusparselt=self.alg_id_cusparselt,
        requires_grad=self.requires_grad,
    )


def semi_sparse_to_copy(func, types, args, kwargs=None) -> torch.Tensor:
    self = args[0]
    kwargs = kwargs or {}

    device = kwargs.get("device", None)

    if device is not None and torch.device(device).type == "cpu":
        dense = self.to_dense()
        return func(dense, **kwargs)

    raise NotImplementedError(
        f"`_to_copy()` with kwargs={kwargs} is not implemented "
        "for SparseSemiStructuredTensor. Only converting to CPU is supported currently."
    )


def semi_sparse_to(func, types, args, kwargs=None) -> torch.Tensor:
    self = args[0]
    remaining_args = args[1:]
    kwargs = kwargs or {}

    # Determine the target device from args/kwargs
    device = None
    if remaining_args:
        first_arg = remaining_args[0]
        if isinstance(first_arg, (torch.device, str)):
            try:
                device = torch.device(first_arg)
            except RuntimeError:
                pass
    if "device" in kwargs:
        device = torch.device(kwargs["device"])

    if device is not None and device.type == "cpu":
        dense = self.to_dense()
        return func(dense, *remaining_args, **kwargs)

    raise NotImplementedError(
        f"`to()` with args={remaining_args}, kwargs={kwargs} is not implemented "
        "for SparseSemiStructuredTensor. Only `to('cpu')` is supported currently."
    )
<|endoftext|># mypy: allow-untyped-defs
import warnings
from collections import namedtuple
from collections.abc import Callable
from typing import Any

import torch
from torch.sparse._semi_structured_conversions import (
    sparse_semi_structured_from_dense_cutlass,
    sparse_semi_structured_to_dense_cutlass,
)
from torch.sparse._semi_structured_ops import (
    fallback_dispatcher,
    semi_sparse_addmm,
    semi_sparse_clone,
    semi_sparse_detach,
    semi_sparse_indices,
    semi_sparse_linear,
    semi_sparse_mm,
    semi_sparse_scaled_mm,
    semi_sparse_t,
    semi_sparse_to,
    semi_sparse_to_copy,
    semi_sparse_values,
    semi_sparse_view,
)


__all__ = [
    "SparseSemiStructuredTensor",
    "SparseSemiStructuredTensorCUTLASS",
    "SparseSemiStructuredTensorCUSPARSELT",
    "to_sparse_semi_structured",
]

_SEMI_STRUCTURED_SPARSE_CONFIG = namedtuple(
    "_SEMI_STRUCTURED_SPARSE_CONFIG",
    "sparse_min_rows sparse_min_cols dense_min_rows dense_min_cols",
)


class SparseSemiStructuredTensor(torch.Tensor):
    """
    This class implements semi-structured sparsity as a Tensor subclass.

    Semi-structured sparsity describes a sparsity pattern where n in every 2n elements are sparse,
    depending on the datatype. It is also referred to as 2:4 sparsity or fine-grained
    structured sparsity.

    There are two backends available for semi_structured sparsity, either cuSPARSELt or CUTLASS.
    This class is meant to serve as a base class for both implementations. SparseSemiStructuredCUTLASS
    and SparseSemiStructuredCUSPARSELT both inherit from this class and define three backend-specific items.
    Note that as such, this class cannot be instantiated directly.

    -`_DTYPE_SHAPE_CONSTRAINTS` - A dictionary holding backend specific dense/sparse min shape constraints
    - `def from_dense()` - backend specific compression routines
    - `def _mm()` - backend specific mm op (either torch._cslt_sparse_mm or torch._sparse_semi_structured_(mm|addmm))
    """

    _DEFAULT_ALG_ID: int = 0
    _DTYPE_SHAPE_CONSTRAINTS: dict[torch.dtype, _SEMI_STRUCTURED_SPARSE_CONFIG]
    _FORCE_CUTLASS: bool = False
    _FUSE_TRANSPOSE: bool = False
    _PROTOTYPE_WARNING_SHOWN: bool = False

    BACKEND: str
    SPARSE_DISPATCH: dict[Callable, Callable]

    packed: torch.Tensor | None
    meta: torch.Tensor | None
    packed_t: torch.Tensor | None
    meta_t: torch.Tensor | None
    compressed_swizzled_bitmask: torch.Tensor | None
    fuse_transpose_cusparselt: bool
    alg_id_cusparselt: int

    __slots__ = ["packed", "meta", "packed_t", "meta_t", "compressed_swizzled_bitmask"]

    @staticmethod
    def __new__(
        cls,
        shape: torch.Size,
        packed: torch.Tensor | None,
        meta: torch.Tensor | None,
        packed_t: torch.Tensor | None,
        meta_t: torch.Tensor | None,
        compressed_swizzled_bitmask: torch.Tensor | None,
        fuse_transpose_cusparselt: bool = False,
        alg_id_cusparselt: int = 0,
        requires_grad: bool = False,
    ):
        """
        Create a new instance of the tensor subclass from the compressed sparse representation.

        We have the option to create the subclass with the compressed representations of both X and X', for training.
        For inference, we only need a single representation (either X or X'), while the corresponding other set will be None.

        Depending on the backend selected, certain fields will be set to None. (CUSPARSELT vs CUTLASS)

        Args:
            shape: The shape of the original dense tensor
            packed: The compressed representation of the original dense tensor
            meta: The metadata of the original dense tensor, if it is stored separately
            packed_t: The compressed representation of the transposed original dense tensor
            meta_t: The metadata of the transposed original dense tensor, if it is stored separately
            compressed_swizzled_bitmask: The masks used by the CUTLASS backend to determine which threads should
                                         participate in the computation. Used for pointwise ops.
            fuse_transpose_cusparselt: When running with cuSPARSELt, we have the option to fuse a transposition
                                       with a matmul, which is useful in the case of 2:4 sparse training.
            alg_id_cusparselt: The algorithm id to use when using cuSPARSELT, will have an effect on performance

        Returns:
            torch.Tensor: A torch.Tensor wrapper subclass.

        Raises:
            ValueError: If all of the tensor arguments are None.
        """
        if not cls._PROTOTYPE_WARNING_SHOWN:
            warnings.warn(
                (
                    "The PyTorch API of SparseSemiStructuredTensor is in prototype stage "
                    "and will change in the near future. Please open a Github issue "
                    "for features requests and see our documentation on the torch.sparse "
                    "module for further information about the project."
                ),
                UserWarning,
                stacklevel=2,
            )
            cls._PROTOTYPE_WARNING_SHOWN = True

            # Because this only runs once, we also load the dispatch table here as well.
            # We can't define the dispatch table explicitly because of torch.ops import errors, so we do this instead
            # But this is useful since it allows users to overload the dispatch table for debugging / testing.
            cls._load_dispatch_table()

            # we can also register the classes with dynamo when the warning is shown.
            torch._dynamo.allow_in_graph(cls)

        if packed is not None:
            previous_tensor = packed
        elif packed_t is not None:
            previous_tensor = packed_t
        else:
            raise ValueError("At least one of packed or packed_t must be provided")

        tensor = torch.Tensor._make_wrapper_subclass(
            cls,
            shape,
            device=previous_tensor.device,
            dtype=previous_tensor.dtype,
            layout=previous_tensor.layout,
            requires_grad=requires_grad,
        )

        tensor.packed = packed
        tensor.meta = meta
        tensor.packed_t = packed_t
        tensor.meta_t = meta_t
        tensor.compressed_swizzled_bitmask = compressed_swizzled_bitmask
        tensor.fuse_transpose_cusparselt = fuse_transpose_cusparselt
        tensor.alg_id_cusparselt = alg_id_cusparselt
        return tensor

    def __repr__(self) -> str:  # type: ignore[override]
        if not hasattr(self, "shape"):
            raise AssertionError("tensor has no shape attribute")
        return f"{self.__class__.__name__}(shape={self.shape})"

    def __tensor_flatten__(
        self,
    ) -> tuple[list[str], tuple[torch.Size, bool, int, bool]]:
        inner_tensors = list(
            filter(lambda x: getattr(self, x) is not None, self.__slots__)
        )
        tensor_meta = (
            self.shape,
            self.fuse_transpose_cusparselt,
            self.alg_id_cusparselt,
            self.requires_grad,
        )
        return inner_tensors, tensor_meta

    @classmethod
    def __tensor_unflatten__(
        cls,
        inner_tensors,
        tensor_meta: tuple[torch.Size, bool, int, bool],
        outer_size,
        outer_stride,
    ) -> torch.Tensor:
        shape, fuse_transpose_cusparselt, alg_id_cusparselt, requires_grad = tensor_meta
        # pyrefly: ignore [no-matching-overload]
        return cls(
            shape=shape,
            packed=inner_tensors.get("packed", None),
            meta=inner_tensors.get("meta", None),
            packed_t=inner_tensors.get("packed_t", None),
            meta_t=inner_tensors.get("meta_t", None),
            compressed_swizzled_bitmask=inner_tensors.get(
                "compressed_swizzled_bitmask", None
            ),
            fuse_transpose_cusparselt=fuse_transpose_cusparselt,
            alg_id_cusparselt=alg_id_cusparselt,
            requires_grad=requires_grad,
        )

    __torch_function__ = torch._C._disabled_torch_function_impl  # type: ignore[assignment]

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs) -> Any:  # type: ignore[override]
        if func._overloadpacket not in cls.SPARSE_DISPATCH:
            raise NotImplementedError(
                f"{cls.__name__} only supports a specific set of operations, "
                f"can't perform requested op ({func.__name__})"
            )
        return cls.SPARSE_DISPATCH[func._overloadpacket](func, types, args, kwargs)

    @classmethod
    def _load_dispatch_table(cls, custom_dispatch_table=None) -> None:
        """
        Loads the op overload sparse dispatch table for the current class.
        """
        if getattr(cls, "SPARSE_DISPATCH", None) is None:
            cls.SPARSE_DISPATCH = {
                torch.ops.aten.values: semi_sparse_values,
                torch.ops.aten.indices: semi_sparse_indices,
                torch.ops.aten.is_same_size: fallback_dispatcher,
                torch.ops.aten.detach_: fallback_dispatcher,
                torch.ops.aten.detach: semi_sparse_detach,
                torch.ops.aten.t: semi_sparse_t,
                torch.ops.aten.view: semi_sparse_view,
                torch.ops.aten.mm: semi_sparse_mm,
                torch.ops.aten.matmul: semi_sparse_mm,
                torch.ops.aten.addmm: semi_sparse_addmm,
                torch.ops.aten.linear: semi_sparse_linear,
                torch.ops.aten._to_copy: semi_sparse_to_copy,
                torch.ops.aten._scaled_mm: semi_sparse_scaled_mm,
                torch.ops.aten.clone: semi_sparse_clone,
                torch.ops.aten.to: semi_sparse_to,
            }
            if custom_dispatch_table is not None:
                cls.SPARSE_DISPATCH.update(custom_dispatch_table)

    @classmethod
    def _validate_device_dim_dtype_shape(cls, original_tensor: torch.Tensor) -> None:
        """
        Assert that the given tensor is valid for semi-structured sparse compression.
        """
        # check device
        if not original_tensor.is_cuda:
            raise RuntimeError(
                f"Error original_tensor.device= {original_tensor.device} is not supported! "
                "Only CUDA tensors are currently supported."
            )

        # check dim
        if original_tensor.dim() != 2:
            raise RuntimeError(
                f"Error original_tensor.dim = {original_tensor.dim()} is not supported! "
                "Only 2d tensors are currently supported."
            )

        # check contiguous
        if not original_tensor.is_contiguous():
            raise RuntimeError(
                "Error original_tensor is not contiguous!"
                "Only contiguous tensors are currently supported."
            )

        # check dtype
        if original_tensor.dtype not in cls._DTYPE_SHAPE_CONSTRAINTS:
            raise RuntimeError(
                f"Error original_tensor.dtype {original_tensor.dtype} is not a supported dtype for {cls}!"
            )

        # check shape
        m, n = original_tensor.shape
        min_rows = cls._DTYPE_SHAPE_CONSTRAINTS[original_tensor.dtype].sparse_min_rows
        min_cols = cls._DTYPE_SHAPE_CONSTRAINTS[original_tensor.dtype].sparse_min_cols
        if m < min_rows or m % min_rows or n < min_cols or n % min_cols:
            # TODO in the future we can add in padding to support sparse dimensions that aren't perfect multiples
            raise RuntimeError(
                f"Error original_tensor.shape {original_tensor.shape} is not supported! "
                f"Both dimensions must be larger than or equal to and a multiple of ({min_rows}, {min_cols})"
            )

    def to_dense(self):  # type:ignore[override]
        col = self.shape[-1]
        return torch.mm(self, torch.eye(col, dtype=self.dtype, device=self.device))

    @classmethod
    def from_dense(
        cls,
        original_tensor: torch.Tensor,
        alg_id: int = _DEFAULT_ALG_ID,
    ) -> "SparseSemiStructuredTensor":
        raise NotImplementedError

    def _mm(
        self,
        B: torch.Tensor,
        *,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError


def to_sparse_semi_structured(
    original_tensor: torch.Tensor,
    transposed: bool = False,
    alg_id: int = SparseSemiStructuredTensor._DEFAULT_ALG_ID,
) -> SparseSemiStructuredTensor:
    """
    This function converts a dense tensor into a sparse semi-structured tensor.
    It will return a SparseSemiStructuredTensor, a subclass of torch.Tensor.

    This function will check to ensure the dense tensor has the right dtype, size, dims, and device.
    We currently only support semi-structured sparse tensors for 2d CUDA tensors.
    Additionally, your tensor must be a positive multiple of the minimum sparse block size, given in
    `_DTYPE_TO_SHAPE_CONSTRAINTS` for each dtype (float32, float16, bfloat16, int8).

    Args:
        original_tensor (Tensor): the dense tensor to convert
        transposed (bool, optional): deprecated arg to be removed in another release. Do not use.
        alg_id (int, optional): the algorithm id to use for cuSPARSELt matmul. Defaults to 0.
            Can be obtained via ``torch._cslt_sparse_mm_search``.
    Returns:
        SparseSemiStructuredTensor: A sparse semi-structured tensor created from the given original_tensor
    Raises:
        None
    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> A = torch.Tensor([0, 0, 1, 1]).tile((128, 32)).half().cuda()
        tensor([[0., 0., 1.,  ..., 0., 1., 1.],
                [0., 0., 1.,  ..., 0., 1., 1.],
                [0., 0., 1.,  ..., 0., 1., 1.],
                ...,
                [0., 0., 1.,  ..., 0., 1., 1.],
                [0., 0., 1.,  ..., 0., 1., 1.],
                [0., 0., 1.,  ..., 0., 1., 1.]], device='cuda:0', dtype=torch.float16)
        >>> A_sparse = to_sparse_semi_structured(A)
        SparseSemiStructuredTensor(shape=torch.Size([128, 128]))
        >>> A_sparse.values()
        tensor([[1., 1., 1.,  ..., 1., 1., 1.],
                [1., 1., 1.,  ..., 1., 1., 1.],
                [1., 1., 1.,  ..., 1., 1., 1.],
                ...,
                [1., 1., 1.,  ..., 1., 1., 1.],
                [1., 1., 1.,  ..., 1., 1., 1.],
                [1., 1., 1.,  ..., 1., 1., 1.]], device='cuda:0', dtype=torch.float16),
        >>> A_sparse.indices()
        tensor([[-4370, -4370, -4370,  ..., -4370, -4370, -4370],
                [-4370, -4370, -4370,  ..., -4370, -4370, -4370],
                [-4370, -4370, -4370,  ..., -4370, -4370, -4370],
                ...,
                [-4370, -4370, -4370,  ..., -4370, -4370, -4370],
                [-4370, -4370, -4370,  ..., -4370, -4370, -4370],
                [-4370, -4370, -4370,  ..., -4370, -4370, -4370]], device='cuda:0', dtype=torch.int16))
    """
    if transposed:
        warnings.warn(
            "Setting transpose from `to_sparse_semi_structured` is deprecated "
            "and will be removed in a future release. "
            "`SparseSemiStructuredTensor` only support contiguous input tensors.",
            FutureWarning,
            stacklevel=2,
        )

    # set from _FORCE_CUTLASS flag
    SPARSE_SUBCLASS = (
        torch.sparse.SparseSemiStructuredTensorCUTLASS
        if SparseSemiStructuredTensor._FORCE_CUTLASS
        else torch.sparse.SparseSemiStructuredTensorCUSPARSELT
    )

    return SPARSE_SUBCLASS.from_dense(original_tensor, alg_id=alg_id)


class SparseSemiStructuredTensorCUTLASS(SparseSemiStructuredTensor):
    """
    This class implements semi-structured sparsity for the CUTLASS backend.


    In this implementation, the specified elements and metadata are stored separately,
    in packed and meta respectively.

    When _FORCE_CUTLASS is set, or when cuSPARSELt is not available, this subclass calls into _sparse_semi_structured_(mm|addmm) and
    sparse_semi_structured_from_dense for conversion to the compressed format.
    """

    BACKEND = "cutlass"
    _DTYPE_SHAPE_CONSTRAINTS = {
        torch.int8: _SEMI_STRUCTURED_SPARSE_CONFIG(16, 128, 16, 16),
        torch.float16: _SEMI_STRUCTURED_SPARSE_CONFIG(32, 64, 8, 8),
        torch.bfloat16: _SEMI_STRUCTURED_SPARSE_CONFIG(32, 64, 8, 8),
        torch.float32: _SEMI_STRUCTURED_SPARSE_CONFIG(32, 32, 4, 4),
    }

    @classmethod
    def from_dense(
        cls,
        original_tensor: torch.Tensor,
        alg_id: int = SparseSemiStructuredTensor._DEFAULT_ALG_ID,
    ) -> "SparseSemiStructuredTensorCUTLASS":
        cls._validate_device_dim_dtype_shape(original_tensor)
        (
            sparse_tensor_cutlass,
            meta_tensor_cutlass,
        ) = sparse_semi_structured_from_dense_cutlass(original_tensor)
        # pyrefly: ignore [no-matching-overload]
        return cls(
            original_tensor.shape,
            packed=sparse_tensor_cutlass,
            meta=meta_tensor_cutlass,
            packed_t=None,
            meta_t=None,
            compressed_swizzled_bitmask=None,
            requires_grad=original_tensor.requires_grad,
        )

    def to_dense(self):  # type: ignore[override]
        if self.meta is None or self.packed is None:
            raise AssertionError("meta and packed must not be None")
        return (
            sparse_semi_structured_to_dense_cutlass(
                self.packed,
                self.meta,
            )
            if self.meta.ndim == 2
            else super().to_dense()
        )

    @classmethod
    def prune_dense_static_sort(
        cls, original_tensor: torch.Tensor, algorithm=""
    ) -> "SparseSemiStructuredTensor":
        """
        This function takes in a unpruned dense tensor and runs a (branchless) static sort across a 4x4 tile.

        It greedily picks the largest values in the tile, upholding the 2:4 sparsity constraint across both rows and columns.
        The algorithm used to prune the matrix is implemented in `_sparse_semi_structured_tile`.

        Then it creates the packed and meta tensors for the compressed sparse representation of the pruned dense tensor.
        It also calculates the packed_t and meta_t tensors for the compressed sparse representation of the transposed
        pruned dense tensor.
        Since we cannot transpose the compressed representations, we store both for the fw/bw pass respectively.

        Finally, this function also computes a compressed swizzled bitmask that encodes the sparsity pattern.
        This can be used in the backward pass to mask the gradients.

        ::

            [9 1 7 4]                       [9 0 7 0]
            [1 2 3 0]                       [0 2 0 0]
            [8 3 5 4] -> prune 4x4 tile  -> [8 0 0 4] -> pack to CUTLASS semi-structured -> packed
            [1 2 6 2]                       [0 0 6 2]                                    -> metadata

                                                      -> pack to transposed CUTLASS      -> packed_t
                                                         semi-structured representation  -> metadata_t

                                                      -> compute swizzled bitmask        -> compressed_swizzled_bitmask

        The equivalent PyTorch code to create the same five outputs from the dense tensor can be found below::

            from torch.sparse import SparseSemiStructuredTensorCUTLASS
            from torch.sparse._semi_structured_conversions import (
                _sparse_semi_structured_tile,
                _compute_compressed_swizzled_bitmask,
            )

            pruned = _sparse_semi_structured_tile(dense)
            packed_cutlass, meta_cutlass = sparse_semi_structured_from_dense_cutlass(
                pruned
            )
            packed_t_cutlass, meta_t_cutlass = (
                sparse_semi_structured_from_dense_cutlass(pruned.t().contiguous())
            )
            bitmask = _compute_compressed_swizzled_bitmask(pruned)

            SparseSemiStructuredTensorCUTLASS(
                dense.shape,
                packed_cutlass,
                meta_cutlass,
                packed_t_cutlass,
                meta_t_cutlass,
                bitmask,
            )
        """
        # We can either pack to the CUTLASS or cuSPARSELt representation, depending on the use_cutlass flag.
        (
            packed,
            meta,
            packed_t,
            meta_t,
            compressed_swizzled_bitmask,
        ) = torch._sparse_semi_structured_tile(
            original_tensor, algorithm=algorithm, use_cutlass=True
        )

        # pyrefly: ignore [no-matching-overload]
        return cls(
            original_tensor.shape,
            packed=packed,
            meta=meta,
            packed_t=packed_t,
            meta_t=meta_t,
            compressed_swizzled_bitmask=compressed_swizzled_bitmask,
            requires_grad=False,
        )

    def _mm(
        self,
        B: torch.Tensor,
        *,
        bias: torch.Tensor | None = None,
        should_transpose_dense: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        if isinstance(B, SparseSemiStructuredTensor):
            raise ValueError(
                "`SparseSemiStructuredTensor @ SparseSemiStructuredTensor` is not supported by the hardware"
            )
        cls_name = self.__class__.__name__
        if self.ndim != 2 or B.ndim != 2:
            raise NotImplementedError(
                f"`{cls_name}` matmul: Broadcasting is not implemented"
            )
        if self.packed is None or self.meta is None:
            raise NotImplementedError(
                f"`{cls_name}` matmul: operation is not supported"
            )
        else:
            _ensure_cutlass_mm_registered()
            constraints = self._DTYPE_SHAPE_CONSTRAINTS[B.dtype]
            return torch.ops.semi_structured.cutlass_mm(
                B,
                self.packed,
                self.meta,
                bias,
                self.shape[0],
                constraints.dense_min_rows,
                constraints.dense_min_cols,
                should_transpose_dense,
            )


class SparseSemiStructuredTensorCUSPARSELT(SparseSemiStructuredTensor):
    """
    The cuSPARSELt backend expects the specified elements and the metadata to be stored in a single tensor:
    packed = [ specified elements of original tensor | metadata ]
    For an original tensor of size (m, k) we expect the first m * k // 2 elements to be the kept elements
    The rest of the tensor is metadata. Since there is only one tensor, we only use the packed and packed_t
    attributes respectively.

    cuSPARSELt also supports transposition fusion, which is necessary for performant 2:4 sparse training, as well
    as specifying alg_id, a config that affects the performance of the matmul depending on matmul sizes.
    """

    BACKEND = "cusparselt"
    _DTYPE_SHAPE_CONSTRAINTS = {
        torch.float8_e4m3fn: _SEMI_STRUCTURED_SPARSE_CONFIG(32, 32, 16, 16),
        torch.int8: _SEMI_STRUCTURED_SPARSE_CONFIG(32, 32, 16, 16),
        torch.float16: _SEMI_STRUCTURED_SPARSE_CONFIG(16, 16, 8, 8),
        torch.bfloat16: _SEMI_STRUCTURED_SPARSE_CONFIG(16, 16, 8, 8),
    }

    @classmethod
    def from_dense(
        cls,
        original_tensor: torch.Tensor,
        alg_id: int = SparseSemiStructuredTensor._DEFAULT_ALG_ID,
    ) -> "SparseSemiStructuredTensorCUSPARSELT":
        cls._validate_device_dim_dtype_shape(original_tensor)
        # pyrefly: ignore [no-matching-overload]
        return cls(
            shape=original_tensor.shape,
            packed=torch._cslt_compress(original_tensor),
            meta=None,
            packed_t=None,
            meta_t=None,
            compressed_swizzled_bitmask=None,
            fuse_transpose_cusparselt=SparseSemiStructuredTensor._FUSE_TRANSPOSE,
            alg_id_cusparselt=alg_id,
            requires_grad=original_tensor.requires_grad,
        )

    @classmethod
    def prune_dense_static_sort(
        cls, original_tensor: torch.Tensor, algorithm=""
    ) -> "SparseSemiStructuredTensor":
        """
        This function does the same thing as described in SparseSemiStructuredCUTLASS, but uses the cuSPARSELt metadata
        layout and sparse matmul.

        The only functional difference is that cuSPARSELt stores `metadata` and `packed` together into a single tensor.

        ::

            [9 1 7 4]                       [9 0 7 0]
            [1 2 3 0]                       [0 2 0 0]
            [8 3 5 4] -> prune 4x4 tile  -> [8 0 0 4] -> pack to cuSPARSELT semi-structured -> packed
            [1 2 6 2]                       [0 0 6 2]

                                                      -> pack to transposed cuSPARSELt      -> packed_t
                                                         semi-structured representation

                                                      -> compute swizzled bitmask           -> compressed_swizzled_bitmask

        The equivalent PyTorch code to create the same three outputs from the dense tensor can be found below::

            from torch.sparse import SparseSemiStructuredTensorCUSPARSELT
            from torch.sparse._semi_structured_conversions import (
                _sparse_semi_structured_tile,
                _compute_compressed_swizzled_bitmask,
            )

            pruned = _sparse_semi_structured_tile(dense)
            packed_cusparselt = torch._cslt_compress(pruned)
            packed_t_cusparselt = torch._cslt_compress(pruned.t().contiguous())
            bitmask = _compute_compressed_swizzled_bitmask(pruned)

            SparseSemiStructuredTensorCUSPARSELT(
                dense.shape, packed_cutlass, None, packed_t_cutlass, None, bitmask
            )
        """
        (
            packed,
            meta,
            packed_t,
            meta_t,
            compressed_swizzled_bitmask,
        ) = torch._sparse_semi_structured_tile(
            original_tensor, algorithm=algorithm, use_cutlass=False
        )

        # Map this two 2-dim view of packed data.
        # TODO: is this proper cuSPARSELt metadata?
        packed = packed.view(original_tensor.shape[0], -1)
        packed_t = packed_t.view(original_tensor.shape[1], -1)

        # pyrefly: ignore [no-matching-overload]
        return cls(
            original_tensor.shape,
            packed=packed,
            meta=meta,
            packed_t=packed_t,
            meta_t=meta_t,
            compressed_swizzled_bitmask=compressed_swizzled_bitmask,
            requires_grad=False,
        )

    def _mm(
        self,
        B: torch.Tensor,
        *,
        bias: torch.Tensor | None = None,
        should_transpose_dense: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        if isinstance(B, SparseSemiStructuredTensor):
            raise ValueError(
                "`SparseSemiStructuredTensor @ SparseSemiStructuredTensor` is not supported by the hardware"
            )
        if self.ndim != 2 or B.ndim != 2:
            raise NotImplementedError(
                f"`{self.__class__.__name__}` matmul: Broadcasting is not implemented"
            )
        if B.dtype != self.dtype:
            raise NotImplementedError(
                f"`{self.__class__.__name__}` matmul: trying to do `A={tuple(self.shape)} @ B={tuple(B.shape)}`, "
                f"with A.dtype={self.dtype} and B.dtype={B.dtype}. "
                "This operation is only supported when A and B have the same data type."
            )
        if bias is not None and bias.dtype != self.dtype:
            raise NotImplementedError(
                f"`{self.__class__.__name__}` matmul: trying to do `A={tuple(self.shape)} @ B={tuple(B.shape)} + C`, "
                f"with A.dtype=B.dtype={self.dtype} and C.dtype={B.dtype}. "
                "This operation is only supported when A, B and C have the same data type."
            )
        # Force fp8 mm to error to be consistent with torch
        if self.dtype == torch.float8_e4m3fn:
            raise NotImplementedError(
                f"`{self.__class__.__name__}` matmul: trying to do `A={tuple(self.shape)} @ B={tuple(B.shape)}`, "
                f"with A.dtype=B.dtype={self.dtype}. "
                "mm is not supported for float8_e4m3fn, please use `torch._scaled_mm` instead."
            )
        if self.packed is None:
            raise NotImplementedError(
                f"`{self.__class__.__name__}` matmul: operation is not supported"
            )
        else:
            _ensure_cusparselt_mm_registered()
            constraints = self._DTYPE_SHAPE_CONSTRAINTS[B.dtype]
            return torch.ops.semi_structured.cusparselt_mm(
                B,
                self.packed,
                bias,
                self.shape[0],
                constraints.dense_min_rows,
                constraints.dense_min_cols,
                should_transpose_dense,
                self.alg_id_cusparselt,
                should_transpose_dense,
            )


_cutlass_mm_registered = False


def _ensure_cutlass_mm_registered():
    """Lazily register the cutlass_mm custom op.

    Registration is deferred to avoid importing torch.library at module load
    time, since torch.sparse is imported early during ``import torch``.
    """
    global _cutlass_mm_registered
    if _cutlass_mm_registered:
        return
    _cutlass_mm_registered = True

    from torch.library import custom_op

    def _cutlass_mm_impl(
        dense: torch.Tensor,
        packed: torch.Tensor,
        meta: torch.Tensor,
        bias: torch.Tensor | None,
        out_features: int,
        min_rows: int,
        min_cols: int,
        should_transpose_dense: bool,
    ) -> torch.Tensor:
        m, n = dense.shape
        to_pad_m = (-m) % min_rows
        to_pad_n = (-n) % min_cols
        need_pad = to_pad_m != 0 or to_pad_n != 0
        dense_padded = dense
        if need_pad:
            dense_padded = torch.nn.functional.pad(dense, (0, to_pad_n, 0, to_pad_m))
        mm_input = dense_padded.t() if should_transpose_dense else dense_padded
        if bias is None:
            res = torch._sparse_semi_structured_mm(packed, meta, mm_input)
        else:
            res = torch._sparse_semi_structured_addmm(bias, packed, meta, mm_input)
        if need_pad:
            out_cols = m if should_transpose_dense else n
            res = res[:out_features].narrow(1, 0, out_cols)
            return res.t().contiguous() if should_transpose_dense else res.contiguous()
        return res.t().contiguous() if should_transpose_dense else res

    cutlass_mm = custom_op("semi_structured::cutlass_mm", mutates_args=())(
        _cutlass_mm_impl
    )

    @cutlass_mm.register_fake
    def _cutlass_mm_fake(
        dense: torch.Tensor,
        packed: torch.Tensor,
        meta: torch.Tensor,
        bias: torch.Tensor | None,
        out_features: int,
        min_rows: int,
        min_cols: int,
        transpose_dense: bool,
    ) -> torch.Tensor:
        out_cols = dense.shape[0] if transpose_dense else dense.shape[1]
        output_shape = (
            (out_cols, out_features) if transpose_dense else (out_features, out_cols)
        )
        return torch.empty(
            output_shape,
            dtype=dense.dtype,
            device=dense.device,
        )

    from torch._subclasses.functional_tensor import (
        _has_unrecognized_tensor_types,
        FunctionalTensorMode,
    )

    # Unlike triton custom ops (which guard with custom_triton_ops_decomposition_disabled
    # to keep the opaque op in exported programs), we decompose unconditionally here.
    # The decomposition targets (aten._sparse_semi_structured_mm, pad, narrow) are real
    # ATen ops with proper meta registrations and are fully serializable, so export
    # consumers benefit from seeing the underlying ops rather than an opaque custom op.
    def _cutlass_mm_decomp(mode, _op, types, args, kwargs):
        if _has_unrecognized_tensor_types(types):
            return NotImplemented
        with mode:
            return _cutlass_mm_impl(*args, **kwargs)

    cutlass_mm.register_torch_dispatch(FunctionalTensorMode, _cutlass_mm_decomp)


_cusparselt_mm_registered = False


def _ensure_cusparselt_mm_registered():
    """Lazily register the cusparselt_mm custom op."""
    global _cusparselt_mm_registered
    if _cusparselt_mm_registered:
        return
    _cusparselt_mm_registered = True

    from torch.library import custom_op

    @custom_op("semi_structured::cusparselt_mm", mutates_args=())
    def cusparselt_mm(
        dense: torch.Tensor,
        packed: torch.Tensor,
        bias: torch.Tensor | None,
        out_features: int,
        min_rows: int,
        min_cols: int,
        fuse_transpose: bool,
        alg_id: int,
        should_transpose_dense: bool = False,
    ) -> torch.Tensor:
        m, n = dense.shape
        to_pad_m = (-m) % min_rows
        to_pad_n = (-n) % min_cols
        need_pad = to_pad_m != 0 or to_pad_n != 0
        dense_padded = dense
        if need_pad:
            dense_padded = torch.nn.functional.pad(dense, (0, to_pad_n, 0, to_pad_m))
        mm_input = dense_padded.t() if should_transpose_dense else dense_padded
        res = torch._cslt_sparse_mm(
            packed,
            mm_input,
            bias=bias,
            transpose_result=fuse_transpose,
            alg_id=alg_id,
        )
        if need_pad:
            out_cols = m if should_transpose_dense else n
            if fuse_transpose:
                return res.narrow(0, 0, out_cols).contiguous()
            else:
                return res.narrow(1, 0, out_cols).contiguous()
        return res

    @cusparselt_mm.register_fake
    def _cusparselt_mm_fake(
        dense: torch.Tensor,
        packed: torch.Tensor,
        bias: torch.Tensor | None,
        out_features: int,
        min_rows: int,
        min_cols: int,
        fuse_transpose: bool,
        alg_id: int,
        should_transpose_dense: bool,
    ) -> torch.Tensor:
        out_cols = dense.shape[0] if should_transpose_dense else dense.shape[1]
        output_shape = (
            (out_cols, out_features) if fuse_transpose else (out_features, out_cols)
        )
        return torch.empty(
            output_shape,
            dtype=dense.dtype,
            device=dense.device,
        )
<|endoftext|># mypy: disallow-untyped-defs

import json
import logging
import os
import re
import subprocess
from datetime import datetime
from socket import gethostname
from typing import Any

from torch._strobelight.cli_function_profiler import StrobelightCLIFunctionProfiler


logger = logging.getLogger("strobelight_compile_time_profiler")

console_handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(name)s, line %(lineno)d, %(asctime)s, %(levelname)s: %(message)s"
)
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.setLevel(logging.INFO)
logger.propagate = False


def get_fburl(url: str) -> str:
    short_url = url
    # Attempt to shorten the URL
    try:
        result = subprocess.run(
            ["fburl", url], capture_output=True, stdin=subprocess.DEVNULL
        )
        if result.returncode == 0:
            short_url = result.stdout.decode("utf-8")
    except Exception as e:
        logger.warning("URL shortening failed: %s, using long URL", repr(e))
    return short_url


def get_strobelight_url(identifier: str) -> str:
    scuba_json = {
        "aggregateList": [],
        "aggregation_field": "async_stack_complete",
        "b_constraints": [[]],
        "c_constraints": [[]],
        "cols": ["namespace_id", "namespace_process_id"],
        "compare": "none",
        "constraints": [
            [{"column": "sample_tags", "op": "all", "value": [f'["{identifier}"]']}]
        ],
        "derivedCols": [],
        "end": "now",
        "enumCols": [],
        "filterMode": "DEFAULT",
        "hideEmptyColumns": "false",
        "ignoreGroupByInComparison": "false",
        "is_timeseries": "false",
        "mappedCols": [],
        "metric": "count",
        "modifiers": [],
        "order": "weight",
        "order_desc": "true",
        "param_dimensions": [
            {"dim": "py_async_stack", "op": "edge", "param": "0", "anchor": "0"}
        ],
        "purposes": [],
        "return_remainder": "false",
        "samplingRatio": "1",
        "should_pivot": "false",
        "start": "-30 days",
        "timezone": "America/Los_Angeles",
        "top": 10000,
    }
    scuba_url_prefix = "https://www.internalfb.com/intern/scuba/query/?dataset=pyperf_experimental/on_demand&drillstate="
    scuba_url_suff = "&view=GraphProfilerView&&normalized=1726332703&pool=uber"
    long_url = scuba_url_prefix + json.dumps(scuba_json) + scuba_url_suff
    return get_fburl(long_url)


class StrobelightCompileTimeProfiler:
    success_profile_count: int = 0
    failed_profile_count: int = 0
    ignored_profile_runs: int = 0
    inside_profile_compile_time: bool = False
    enabled: bool = False

    # A regex that can be used to filter out what frames to profile. ex: "1/.*"
    frame_id_filter: str | None = os.environ.get("COMPILE_STROBELIGHT_FRAME_FILTER")

    # A unique identifier that is used as the run_user_name in the strobelight profile to
    # associate all compile time profiles together.
    identifier: str | None = None

    current_phase: str | None = None

    profiler: Any | None = None

    max_stack_length: int = int(
        os.environ.get("COMPILE_STROBELIGHT_MAX_STACK_LENGTH", 500)
    )
    max_profile_time: int = int(
        os.environ.get("COMPILE_STROBELIGHT_MAX_PROFILE_TIME", 60 * 30)
    )
    # Collect sample each x cycles.
    sample_each: int = int(
        float(os.environ.get("COMPILE_STROBELIGHT_SAMPLE_RATE", 1e7))
    )

    @classmethod
    def get_frame(cls) -> str:
        from torch._guards import CompileContext

        return (str)(CompileContext.current_trace_id())

    @classmethod
    def enable(cls, profiler_class: Any = StrobelightCLIFunctionProfiler) -> None:
        if cls.enabled:
            logger.info("compile time strobelight profiling already enabled")
            return

        logger.info("compile time strobelight profiling enabled")

        if profiler_class is StrobelightCLIFunctionProfiler:
            import shutil

            if not shutil.which("strobeclient"):
                logger.info(
                    "strobeclient not found, can't enable compile time strobelight profiling, seems"
                    "like you are not on a FB machine."
                )
                return

        cls.enabled = True
        cls._cls_init()
        # profiler_class should have public API similar to that of StrobelightCLIFunctionProfiler.
        # we have pass different functionProfilerClass for meta-internal fbcode targets.
        # NB: the actual implementation in Meta is at
        # fbcode/caffe2/fb/strobelight/function_profiler.py
        cls.profiler = profiler_class(
            sample_each=cls.sample_each,
            max_profile_duration_sec=cls.max_profile_time,
            stack_max_len=cls.max_stack_length,
            async_stack_max_len=cls.max_stack_length,
            run_user_name="pt2-profiler/"
            + os.environ.get("USER", os.environ.get("USERNAME", "")),
            sample_tags={cls.identifier},
        )

    @classmethod
    def _cls_init(cls) -> None:
        cls.identifier = "{date}{pid}{hostname}".format(
            date=datetime.now().strftime("%Y-%m-%d-%H:%M:%S"),
            pid=os.getpid(),
            hostname=gethostname(),
        )

        logger.info("Unique sample tag for this run is: %s", cls.identifier)
        logger.info(
            "URL to access the strobelight profile at the end of the run: %s",
            get_strobelight_url(cls.identifier),
        )

    @classmethod
    def _log_stats(cls) -> None:
        logger.info(
            "%s strobelight success runs out of %s non-recursive compilation events.",
            cls.success_profile_count,
            cls.success_profile_count + cls.failed_profile_count,
        )

    # TODO use threadlevel meta data to tags to record phases.
    @classmethod
    def profile_compile_time(
        cls, func: Any, phase_name: str, *args: Any, **kwargs: Any
    ) -> Any:
        def skip() -> Any:
            return func(*args, **kwargs)

        if not cls.enabled:
            return skip()

        if cls.profiler is None:
            logger.error("profiler is not set")
            return

        frame_id = cls.get_frame()

        if cls.inside_profile_compile_time:
            cls.ignored_profile_runs += 1
            logger.info(
                "profile_compile_time is requested for phase: %s, frame %s, while already in running phase: %s,"
                "frame %s, recursive call ignored",
                phase_name,
                frame_id,
                cls.current_phase,
                frame_id,
            )
            return skip()

        if cls.frame_id_filter is not None:
            should_run = re.match(cls.frame_id_filter, frame_id) is not None
            if not should_run:
                logger.info(
                    "profiling frame %s is skipped due to frame_id_filter %s",
                    frame_id,
                    cls.frame_id_filter,
                )
                return skip()

        cls.inside_profile_compile_time = True
        cls.current_phase = phase_name
        logger.info("profiling frame %s", frame_id)
        work_result = cls.profiler.profile(func, *args, **kwargs)

        if cls.profiler.profile_result is not None:
            cls.success_profile_count += 1
        else:
            cls.failed_profile_count += 1

        cls._log_stats()
        cls.inside_profile_compile_time = False
        return work_result
<|endoftext|># mypy: disallow-untyped-defs

import functools
import logging
import os
import re
import subprocess
import time
from collections.abc import Callable, Sequence
from threading import Lock
from timeit import default_timer as timer
from typing import Any, TypeVar
from typing_extensions import ParamSpec


logger = logging.getLogger("strobelight_function_profiler")

console_handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(name)s, line %(lineno)d, %(asctime)s, %(levelname)s: %(message)s"
)
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.setLevel(logging.INFO)
logger.propagate = False

_P = ParamSpec("_P")
_R = TypeVar("_R")


class StrobelightCLIProfilerError(Exception):
    """
    Raised when an error happens during strobelight profiling
    """


def _pid_namespace_link(pid: int | None = None) -> str:
    """Returns the link to the process's namespace, example: pid:[4026531836]"""
    PID_NAMESPACE_PATH = "/proc/{}/ns/pid"
    pid = pid or os.getpid()
    return os.readlink(PID_NAMESPACE_PATH.format(pid))


def _pid_namespace(pid: int | None = None) -> int:
    """Returns the process's namespace id"""
    pid = pid or os.getpid()
    link = _pid_namespace_link(pid)
    return int(link[link.find("[") + 1 : -1])


def _command_to_string(command: Sequence[str]) -> str:
    return " ".join(command)


class StrobelightCLIFunctionProfiler:
    """
    Note: this is a Meta only tool.

    StrobelightCLIFunctionProfiler can be used to profile a python function and
    generate a strobelight link with the results. It works on meta servers but
    does not require an fbcode target.
    When stop_at_error is false(default), error during profiling does not prevent
    the work function from running.

    Check function_profiler_example.py for an example.
    """

    # This lock is used to make sure only one thread is running the profiler at any point.
    _lock = Lock()

    def __init__(
        self,
        *,
        stop_at_error: bool = False,
        max_profile_duration_sec: int = 60 * 10,
        sample_each: float = 1e7,  # sample each sample_each cycles.
        run_user_name: str = "pytorch-strobelight-ondemand",
        timeout_wait_for_running_sec: int = 60,
        timeout_wait_for_finished_sec: int = 60,
        recorded_env_variables: list[str] | None = None,
        sample_tags: list[str] | None = None,
        stack_max_len: int = 127,
        async_stack_max_len: int = 127,
    ):
        self.stop_at_error = stop_at_error
        self.max_profile_duration_sec = max_profile_duration_sec
        self.sample_each = sample_each
        self.run_user_name = run_user_name
        self.timeout_wait_for_running_sec = timeout_wait_for_running_sec
        self.timeout_wait_for_finished_sec = timeout_wait_for_finished_sec
        # Results of the most recent run.
        # Tracks the strobelight run id of the most recent run
        self.current_run_id: int | None = None
        self.profile_result: list[str] | None = None
        self.sample_tags = sample_tags

    def _run_async(self) -> None:
        processId = os.getpid()
        namespace = _pid_namespace(processId)
        command = [
            "strobeclient",
            "run",
            "--profiler",
            "pyperf",
            "--event",
            "cycles",
            "--async",
            "--sample-interval",
            f"{int(self.sample_each)}",
            "--duration-ms",
            f"{int(self.max_profile_duration_sec * 1000)}",
            "--pid",
            f"{namespace}:{processId}",
        ]

        if self.sample_tags:
            command.append("--sample-tags")
            command.append(",".join(self.sample_tags))

        logger.debug("running command: %s", _command_to_string(command))
        result = subprocess.run(command, capture_output=True)
        output = result.stderr.decode("utf-8")
        logger.debug("output:\n{%s}", output)

        if result.returncode != 0:
            raise StrobelightCLIProfilerError(
                f"failed to start strobelight profiling, error in run_async:{output}"
            )

        if match := re.search(r"INFO Run Id: (-?\d+)", output):
            self.current_run_id = int(match.group(1))
            return

        raise StrobelightCLIProfilerError(
            f"failed to start strobelight profiling, unexpected result {output}"
        )

    def _wait_for_running(self, counter: int = 0) -> None:
        if counter > 20:
            raise StrobelightCLIProfilerError(
                "wait_for_running called more than 20 times"
            )

        command = ["strobeclient", "getRunStatus", "--run-id", f"{self.current_run_id}"]
        logger.debug("running command: %s", _command_to_string(command))
        result = subprocess.run(command, capture_output=True)
        output = result.stderr.decode("utf-8")
        logger.debug("output:\n{%s}", output)

        if result.returncode != 0:
            raise StrobelightCLIProfilerError(
                f"failed to start strobelight profiling, error in wait_for_running:{output}"
            )

        if match := re.search("Profile run status: (.*)", output):
            current_status = match.group(1)
            if current_status == "RUNNING":
                return
            elif current_status == "PREPARING":
                time.sleep(10)
                self._wait_for_running(counter + 1)
                return
            else:
                raise StrobelightCLIProfilerError(f"unexpected {current_status} phase")

        raise StrobelightCLIProfilerError(f"unexpected output\n: {output} ")

    def _stop_run(self) -> None:
        command = ["strobeclient", "stopRun", "--run-id", str(self.current_run_id)]
        logger.debug("running command: %s", _command_to_string(command))
        result = subprocess.run(command, capture_output=True)
        output = result.stderr.decode("utf-8")
        logger.debug("output:\n{%s}", output)

        if result.returncode != 0:
            raise StrobelightCLIProfilerError(
                f"failed to stop strobelight profiling, return code is not 0 :{output}"
            )

        if match := re.search("INFO ::1:(.*)", output):
            current_status = match.group(1)
            if current_status.__contains__("Success!"):
                return
            else:
                raise StrobelightCLIProfilerError(
                    f"failed to stop strobelight profiling, got {current_status} result"
                )

        raise StrobelightCLIProfilerError(f"unexpected output\n: {output} ")

    def _get_results(self) -> None:
        command = ["strobeclient", "getRunStatus", "--run-id", str(self.current_run_id)]
        logger.debug("running command: %s", _command_to_string(command))
        result = subprocess.run(command, capture_output=True)
        output = result.stderr.decode("utf-8")
        logger.debug("output:\n{%s}", output)

        if result.returncode != 0:
            raise StrobelightCLIProfilerError(
                f"failed to extract profiling results, return code is not 0 : {output}"
            )

        if match := re.search("INFO ::1:(.*)", output):
            current_status = match.group(1)
            if current_status.__contains__("Profile run status: PROCESSING"):
                time.sleep(10)
                self._get_results()
                return
            elif not current_status.__contains__("Profile run finished with SUCCESS"):
                raise StrobelightCLIProfilerError(
                    f"failed to extract profiling results, unexpected response {output}"
                )

        self.profile_result = []
        for item in re.findall(
            r"(Total samples(.*)|GraphProfiler(.*)|Icicle view \(python stack\)(.*))",
            output,
        ):
            self.profile_result += item[0]
            logger.info(item[0])

    def _stop_strobelight_no_throw(
        self,
        collect_results: bool,
    ) -> None:
        try:
            # call stop run
            self._stop_run()
            logger.info("strobelight profiling stopped")

            logger.debug("collection stopped")

            if not collect_results:
                return

            self._get_results()
        except Exception:
            logger.warning("error during stop_strobelight", exc_info=True)

    # Return true if strobelight started and is running. Never throw.
    def _start_strobelight(self) -> bool:
        strobelight_started = False
        try:
            self._run_async()
            strobelight_started = True
            logger.info("strobelight run id is: %s", self.current_run_id)
            self._wait_for_running()
            logger.info("strobelight profiling running")
            return True

        except Exception:
            logger.warning("error during start_strobelight:", exc_info=True)
            if strobelight_started:
                self._stop_strobelight_no_throw(collect_results=False)
            return False

    def profile(
        self, work_function: Callable[_P, _R], *args: _P.args, **kwargs: _P.kwargs
    ) -> _R | None:
        self.current_run_id = None
        self.profile_result = None

        if locked := StrobelightCLIFunctionProfiler._lock.acquire(False):
            if not locked:
                if self.stop_at_error:
                    raise StrobelightCLIProfilerError("concurrent runs not supported")

                logger.warning("concurrent runs not supported")
                return work_function(*args, **kwargs)

            started = self._start_strobelight()
            if not started:
                if self.stop_at_error:
                    StrobelightCLIFunctionProfiler._lock.release()
                    raise StrobelightCLIProfilerError(
                        "failed to start strobelight profiling"
                    )
                result = work_function(*args, **kwargs)
                StrobelightCLIFunctionProfiler._lock.release()
                return result

            try:
                logger.debug("collection started")
                start = timer()
                result = work_function(*args, **kwargs)
                end = timer()
                total_time = end - start  # Time in seconds, e.g. 5.38091952400282
                logger.info("work function took %s seconds", total_time)
                self._stop_strobelight_no_throw(collect_results=True)
                StrobelightCLIFunctionProfiler._lock.release()
                return result
            except Exception as error:
                logger.warning("work function throw exception", exc_info=True)
                self._stop_strobelight_no_throw(collect_results=False)
                StrobelightCLIFunctionProfiler._lock.release()
                raise error
        return None


# A function decorator that wraps profile, if no profiler is provided one with
# default args is created. A function can be annotated as:
# @strobelight()
# @strobelight(profiler = StrobelightFunctionProfiler(stop_at_error=True,..))
# @strobelight(stop_at_error=True,...)
def strobelight(
    profiler: StrobelightCLIFunctionProfiler | None = None, **kwargs: Any
) -> Callable[[Callable[_P, _R]], Callable[_P, _R | None]]:
    if not profiler:
        profiler = StrobelightCLIFunctionProfiler(**kwargs)

    def strobelight_inner(
        work_function: Callable[_P, _R],
    ) -> Callable[_P, _R | None]:
        @functools.wraps(work_function)
        def wrapper_function(*args: _P.args, **kwargs: _P.kwargs) -> _R | None:
            # pyrefly: ignore [bad-argument-type]
            return profiler.profile(work_function, *args, **kwargs)

        return wrapper_function

    return strobelight_inner
<|endoftext|>import textwrap
from collections.abc import Callable
from typing import Any, TypeVar


_BACK_COMPAT_OBJECTS: dict[Any, None] = {}
_MARKED_WITH_COMPATIBILITY: dict[Any, None] = {}


_T = TypeVar("_T")


def compatibility(is_backward_compatible: bool) -> Callable[[_T], _T]:
    if is_backward_compatible:

        def mark_back_compat(fn: _T) -> _T:
            docstring = textwrap.dedent(getattr(fn, "__doc__", None) or "")
            docstring += """

.. note::
    Backwards-compatibility for this API is guaranteed.
"""
            fn.__doc__ = docstring
            _BACK_COMPAT_OBJECTS.setdefault(fn)
            _MARKED_WITH_COMPATIBILITY.setdefault(fn)
            return fn

        return mark_back_compat
    else:

        def mark_not_back_compat(fn: _T) -> _T:
            docstring = textwrap.dedent(getattr(fn, "__doc__", None) or "")
            docstring += """

.. warning::
    This API is experimental and is *NOT* backward-compatible.
"""
            fn.__doc__ = docstring
            _MARKED_WITH_COMPATIBILITY.setdefault(fn)
            return fn

        return mark_not_back_compat
<|endoftext|>from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, TYPE_CHECKING

from torch.fx.graph_module import (
    _format_import_block,
    GraphModule,
    reduce_graph_module,
    reduce_package_graph_module,
)
from torch.package import PackageExporter, sys_importer

from ._compatibility import compatibility


if TYPE_CHECKING:
    from torch.fx.graph import PythonCode


_use_lazy_graph_module_flag = False
_force_skip_lazy_graph_module_flag = False


@compatibility(is_backward_compatible=False)
@contextmanager
def _force_skip_lazy_graph_module() -> Iterator[None]:
    """
    Skip using lazy graph module disregarding the setting of _use_lazy_graph_module.
    Use to skip _LazyGraphModule when testing inductor torchscript related backend.

    torch.jit.script a _LazyGraphModule results in following error:
        https://gist.github.com/shunting314/5143654c8084aed84ecd19b818258a69
    """
    try:
        global _force_skip_lazy_graph_module_flag
        prior = _force_skip_lazy_graph_module_flag
        _force_skip_lazy_graph_module_flag = True
        yield
    finally:
        _force_skip_lazy_graph_module_flag = prior


@compatibility(is_backward_compatible=False)
@contextmanager
def _use_lazy_graph_module(should_use: bool) -> Iterator[None]:
    try:
        global _use_lazy_graph_module_flag
        prior = _use_lazy_graph_module_flag
        _use_lazy_graph_module_flag = (
            should_use and not _force_skip_lazy_graph_module_flag
        )
        yield
    finally:
        _use_lazy_graph_module_flag = prior


@compatibility(is_backward_compatible=False)
def _get_graph_module_cls() -> type[GraphModule]:
    return _LazyGraphModule if _use_lazy_graph_module_flag else GraphModule


def _make_graph_module(
    *args: Any, graph_module_cls: type[GraphModule] | None = None, **kwargs: Any
) -> GraphModule:
    if graph_module_cls is None:
        graph_module_cls = _get_graph_module_cls()

    return graph_module_cls(*args, **kwargs)


@compatibility(is_backward_compatible=False)
def _unwrap_lazy_graph_module(gm: GraphModule) -> GraphModule:
    # torch.jit.script cannot script _LazyGraphModule, so rebuild a plain one
    if isinstance(gm, _LazyGraphModule):
        return GraphModule(gm, gm.graph)
    return gm


@compatibility(is_backward_compatible=False)
class _LazyGraphModule(GraphModule):
    """
    The main difference between _LazyGraphModule and GraphModule is how recompile happens.
    GraphModule will do a 'recompile' call to generate python code and the forward method when it's
    constructed. Later on if the graph get updated, recompile method can be called again to refresh
    the saved python code and forward method.

    However in some cases especially in inductor, the recompilation can be a waste since we never
    check the python code for the graph module or call its forward method. A few more concreate
    examples regarding pattern matching fx passes in inductor:
    1. some passes will update the graph to be compiled and then call recompile on the GraphModule.
    2. some passes will trace small pattern function to search it in the graph being compiled and
       replace the match with the traced graph of a replacement function. The pattern graph and
       replacement graph are quite small but there are large amount of them. Doing GraphModule.recompile
       for them in GraphModule.__init__ is also a waste of time.

    However simply skip calling GraphModule.recompile in these scenarios is also dangeruous.
    People may want to check the python code or call the GraphModule's forward method for debugging purposes.

    The way _LazyGraphModule solves it is, we override the recompile method to just mark the
    need for recompilation but does not do the actual recompilation. Later on if people really
    access the compiled python code or call the GraphModule's forward method, we do the real
    recompilation.
    """

    @classmethod
    def from_graphmodule(cls, gm: GraphModule) -> GraphModule:
        if isinstance(gm, _LazyGraphModule):
            return gm
        else:
            return _LazyGraphModule(gm, gm.graph)

    @staticmethod
    def force_recompile(gm: GraphModule) -> None:
        """
        Sometimes we need force a recompile as a workaround
        - we want to do the real recompilation before symbolic_trace to avoid error:
            https://gist.github.com/shunting314/75549c2e82ae07ac1139c94a3583d259
        """
        if isinstance(gm, _LazyGraphModule):
            gm.real_recompile()

    def real_recompile(self) -> None:
        if self._needs_recompile():
            self._real_recompile()

    @classmethod
    def _needs_recompile(cls) -> bool:
        return cls.forward is cls._lazy_forward

    def _lazy_forward(self, *args: Any, **kwargs: Any) -> Any:
        # Call self.real_recompile() rather than self._real_recompile() here.
        # The _lazy_forward method may be saved and call repeatedly.
        # Calling self.real_recompile can make sure we skip recompilation if
        # we have already done so.
        self.real_recompile()
        if self._needs_recompile():
            raise AssertionError("Recompilation required after real_recompile()")

        # call `__call__` rather than 'forward' since recompilation may
        # install a wrapper for `__call__` to provide a customized error
        # message.
        return self(*args, **kwargs)

    forward = _lazy_forward

    def __reduce_package__(
        self, exporter: PackageExporter
    ) -> tuple[Any, tuple[Any, str]]:
        """
        Follow GraphModule.__reduce__ but call 'self._real_recompile' rather
        than 'self.recompile' since for a _LazyGraphModule, self.recompile just
        mark the need of recompilation and does not return the PythonCode object.
        """
        python_code = self._real_recompile()
        dict_without_graph = self.__dict__.copy()
        dict_without_graph["_graphmodule_cls_name"] = self.__class__.__name__
        del dict_without_graph["_graph"]

        generated_module_name = f"fx-generated._{exporter.get_unique_id()}"
        import_block = _format_import_block(python_code.globals, exporter.importer)
        module_code = import_block + self.code
        exporter.save_source_string(generated_module_name, module_code)
        return (
            reduce_package_graph_module,
            (dict_without_graph, generated_module_name),
        )

    def __reduce__(self) -> tuple[Any, tuple[Any, str]]:
        """
        Follow GraphModule.__reduce__ but call 'self._real_recompile' rather
        than 'self.recompile' since for a _LazyGraphModule, self.recompile just
        mark the need of recompilation and does not return the PythonCode object.
        """
        python_code = self._real_recompile()
        dict_without_graph = self.__dict__.copy()
        import_block = _format_import_block(python_code.globals, sys_importer)
        del dict_without_graph["_graph"]
        return (reduce_graph_module, (dict_without_graph, import_block))

    def _real_recompile(self) -> "PythonCode":
        return super().recompile()

    @classmethod
    def recompile(cls) -> None:  # pyrefly: ignore[bad-override]
        cls.forward = cls._lazy_forward

    @property
    def code(self) -> str:
        self.real_recompile()
        return super().code

    def __str__(self) -> str:
        """
        str(GraphModule) will access the _code attribute. Make sure recompile
        happens so _code attribute is available.
        """
        self.real_recompile()
        return super().__str__()
<|endoftext|>from __future__ import annotations

from typing import Any, TYPE_CHECKING

from torch.fx.experimental.unification import Var  # type: ignore[attr-defined]

from ._compatibility import compatibility


if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch.fx.experimental.migrate_gradual_types.constraint import DVar


__all__ = ["Dyn", "TensorType", "is_consistent", "is_more_precise"]


@compatibility(is_backward_compatible=False)
class TensorType:
    """
    TensorType defines a type for tensors, which consists of a list of dimensions.
    Example:
        class M(torch.nn.Module):
            def forward(self, x:TensorType((1,2,3, Dyn)), y:TensorType((1,2,3, Dyn))):
                return torch.add(x, y)
    """

    def __init__(self, dim: Sequence[Any]) -> None:
        self.__origin__ = TensorType
        self.__args__ = dim

    @property
    def dims(self) -> Sequence[DVar | int | _DynType]:
        return self.__args__

    def __repr__(self) -> str:
        return f"TensorType[{self.dims}]"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return list(self.dims) == list(other.dims)
        else:
            return False

    @staticmethod
    def __class_getitem__(*args: object) -> TensorType:
        if len(args) == 1 and isinstance(args[0], tuple):
            args = args[0]
        return TensorType(tuple(args))


class _DynType:
    """
    _DynType defines a type which stands for the absence of type information.
    """

    def __init__(self) -> None:
        self.__name__ = "_DynType"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def __str__(self) -> str:
        return "Dyn"

    def __repr__(self) -> str:
        return "Dyn"


Dyn = _DynType()


@compatibility(is_backward_compatible=False)
def is_consistent(t1: object, t2: object) -> bool:
    """
    A binary relation denoted by ~ that determines if t1 is consistent with t2.
    The relation is reflexive, symmetric but not transitive.
    returns True if t1 and t2 are consistent and False otherwise.
    Example:
        Dyn ~ TensorType((1,2,3))
        int ~ Dyn
        int ~ int
        TensorType((1,Dyn,3)) ~ TensorType((1,2,3))
    """

    if t1 == t2:
        return True

    if t1 == Dyn or t2 == Dyn or isinstance(t1, Var) or isinstance(t2, Var):
        return True

    if isinstance(t1, TensorType) and isinstance(t2, TensorType):
        return len(t1.dims) == len(t2.dims) and all(
            is_consistent(elem1, elem2) for elem1, elem2 in zip(t1.dims, t2.dims)
        )
    else:
        return False


@compatibility(is_backward_compatible=False)
def is_more_precise(t1: object, t2: object) -> bool:
    """
    A binary relation denoted by <= that determines if t1 is more precise than t2.
    The relation is reflexive and transitive.
    returns True if t1 is more precise than t2 and False otherwise.
    Example:
        Dyn >= TensorType((1,2,3))
        int >= Dyn
        int >= int
        TensorType((1,Dyn,3)) <= TensorType((1,2,3))
    """
    if t1 == t2:
        return True

    if isinstance(t2, _DynType):
        return True

    if isinstance(t1, TensorType) and isinstance(t2, TensorType):
        return len(t1.dims) == len(t2.dims) and all(
            is_more_precise(elem1, elem2) for elem1, elem2 in zip(t1.dims, t2.dims)
        )

    else:
        return False
<|endoftext|>import inspect
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
import torch.fx.traceback as fx_traceback
from torch._logging import LazyString, trace_structured
from torch.hub import tqdm

from . import config
from ._compatibility import compatibility
from ._lazy_graph_module import _make_graph_module
from ._symbolic_trace import Tracer
from .graph import Graph
from .graph_module import GraphModule
from .node import Argument, map_aggregate, map_arg, Node, Target
from .proxy import Proxy


log = logging.getLogger(__name__)

__all__ = ["Interpreter", "Transformer"]


def _format_fx_node(n: Node) -> str:
    """
    Format a torch.fx.Node into a human-readable string for debug logging.

    Args:
        n (torch.fx.Node): The FX node being executed.

    Returns:
        str: A formatted string describing the node operation, including its
        name, target, positional arguments, and keyword arguments.
    """
    module_prefix = getattr(n.target, "__module__", "")
    module_prefix = f"{module_prefix}." if module_prefix else ""

    # Handle positional and keyword arguments
    args = ", ".join(map(str, n.args))
    kwargs = ", ".join(f"{k}={v}" for k, v in n.kwargs.items())
    joined = ", ".join(filter(None, [args, kwargs]))

    return (
        f"{n.name} = {module_prefix}{getattr(n.target, '__name__', n.target)}({joined})"
    )


@compatibility(is_backward_compatible=True)
class Interpreter:
    """
    An Interpreter executes an FX graph Node-by-Node. This pattern
    can be useful for many things, including writing code
    transformations as well as analysis passes.

    Methods in the Interpreter class can be overridden to customize
    the behavior of execution. The map of overridable methods
    in terms of call hierarchy::

        run()
            +-- run_node
                +-- placeholder()
                +-- get_attr()
                +-- call_function()
                +-- call_method()
                +-- call_module()
                +-- output()

    Example:

        Suppose we want to swap all instances of ``torch.neg`` with
        ``torch.sigmoid`` and vice versa (including their ``Tensor``
        method equivalents). We could subclass Interpreter like so::

            class NegSigmSwapInterpreter(Interpreter):
                def call_function(
                    self, target: Target, args: Tuple, kwargs: Dict
                ) -> Any:
                    if target is torch.sigmoid:
                        return torch.neg(*args, **kwargs)
                    return super().call_function(target, args, kwargs)

                def call_method(self, target: Target, args: Tuple, kwargs: Dict) -> Any:
                    if target == "neg":
                        call_self, *args_tail = args
                        return call_self.sigmoid(*args_tail, **kwargs)
                    return super().call_method(target, args, kwargs)


            def fn(x):
                return torch.sigmoid(x).neg()


            gm = torch.fx.symbolic_trace(fn)
            input = torch.randn(3, 4)
            result = NegSigmSwapInterpreter(gm).run(input)
            torch.testing.assert_close(result, torch.neg(input).sigmoid())

    Args:
        module (torch.nn.Module): The module to be executed
        garbage_collect_values (bool): Whether to delete values after their last
            use within the Module's execution. This ensures optimal memory usage during
            execution. This can be disabled to, for example, examine all of the intermediate
            values in the execution by looking at the ``Interpreter.env`` attribute.
        graph (Optional[Graph]): If passed, the interpreter will execute this
            graph instead of `module.graph`, using the provided `module`
            argument to satisfy any requests for state.
    """

    @compatibility(is_backward_compatible=True)
    def __init__(
        self,
        module: torch.nn.Module,
        garbage_collect_values: bool = True,
        graph: Graph | None = None,
    ) -> None:
        self.module = module
        self.submodules = dict(self.module.named_modules())
        if graph is not None:
            self.graph = graph
        else:
            self.graph = self.module.graph  # type: ignore[assignment]
        self.env: dict[Node, Any] = {}
        self.name = "Interpreter"
        self.garbage_collect_values = garbage_collect_values
        self.extra_traceback = True

        if self.garbage_collect_values:
            # Run through reverse nodes and record the first instance of a use
            # of a given node. This represents the *last* use of the node in the
            # execution order of the program, which we will use to free unused
            # values
            node_to_last_use: dict[Node, Node] = {}
            self.user_to_last_uses: dict[Node, list[Node]] = {}

            def register_last_uses(n: Node, user: Node) -> None:
                if n not in node_to_last_use:
                    node_to_last_use[n] = user
                    self.user_to_last_uses.setdefault(user, []).append(n)

            for node in reversed(self.graph.nodes):
                for n in node._input_nodes:
                    register_last_uses(n, node)

    @compatibility(is_backward_compatible=True)
    def run(
        self,
        *args: Any,
        initial_env: dict[Node, Any] | None = None,
        enable_io_processing: bool = True,
    ) -> Any:
        """
        Run `module` via interpretation and return the result.

        Args:
            *args: The arguments to the Module to run, in positional order
            initial_env (Optional[Dict[Node, Any]]): An optional starting environment for execution.
                This is a dict mapping `Node` to any value. This can be used, for example, to
                pre-populate results for certain `Nodes` so as to do only partial evaluation within
                the interpreter.
            enable_io_processing (bool): If true, we process the inputs and outputs with graph's process_inputs and
                process_outputs function first before using them.

        Returns:
            Any: The value returned from executing the Module
        """
        self.env = initial_env if initial_env is not None else {}

        # Positional function args are consumed left-to-right by
        # `placeholder` nodes. Use an iterator to keep track of
        # position and extract those values.
        if enable_io_processing:
            args = self.graph.process_inputs(*args)
        self.args_iter: Iterator[Any] = iter(args)
        pbar = tqdm(
            total=len(self.graph.nodes),
            desc=f"{self.name}: {str(list(self.graph.nodes)) if config.verbose_progress else ''}",
            initial=0,
            position=0,
            leave=True,
            disable=config.disable_progress,
            delay=0,
        )

        for node in self.graph.nodes:
            pbar.update(1)
            if node in self.env:
                # Short circuit if we have this value. This could
                # be used, for example, for partial evaluation
                # where the caller has pre-populated `env` with
                # values for a subset of the program.
                continue

            try:
                self.env[node] = self.run_node(node)
            except Exception as e:
                if self.extra_traceback:
                    msg = f"While executing {node.format_node()}"
                    msg = f"{e.args[0]}\n\n{msg}" if e.args else str(msg)
                    msg += f"\nOriginal traceback:\n{node.stack_trace}"
                    if (
                        isinstance(self.module, GraphModule)
                        and self.module.graph is not None
                        and isinstance(self.module.graph, torch.fx.Graph)
                    ):
                        trace_structured(
                            "artifact",
                            metadata_fn=lambda: {
                                "name": "fx_interpreter_error",
                                "encoding": "string",
                            },
                            payload_fn=lambda: (
                                f"{msg}\nGraphModule: "
                                f"{self.module.print_readable(print_output=False, include_stride=True)}"  # type: ignore[operator]
                            ),
                        )

                    msg += "\nUse tlparse to see full graph. "
                    msg += "(https://github.com/pytorch/tlparse?tab=readme-ov-file#tlparse-parse-structured-pt2-logs)"
                    e.args = (msg,) + e.args[1:]
                    if isinstance(e, KeyError):
                        raise RuntimeError(*e.args) from e
                raise

            if self.garbage_collect_values:
                for to_delete in self.user_to_last_uses.get(node, []):
                    del self.env[to_delete]

            if node.op == "output":
                output_val = self.env[node]
                return (
                    self.graph.process_outputs(output_val)
                    if enable_io_processing
                    else output_val
                )

    @compatibility(is_backward_compatible=True)
    def boxed_run(self, args_list: list[Any]) -> Any:
        """
        Run `module` via interpretation and return the result.  This uses the "boxed"
        calling convention, where you pass a list of arguments, which will be cleared
        by the interpreter.  This ensures that input tensors are promptly deallocated.
        """
        # Collect placeholder nodes first
        placeholder_nodes = [n for n in self.graph.nodes if n.op == "placeholder"]

        # Check argument count
        if len(args_list) != len(placeholder_nodes):
            detail = (
                "extra arguments"
                if len(args_list) > len(placeholder_nodes)
                else "missing arguments"
            )
            raise RuntimeError(
                f"Interpreter.boxed_run expected {len(placeholder_nodes)} arguments for placeholders "
                f"but received {len(args_list)} ({detail})"
            )

        # Assign arguments to placeholders
        env = dict(zip(placeholder_nodes, args_list))
        args_list.clear()
        return self.run(initial_env=env)

    @contextmanager
    def _set_current_node(self, node: Node) -> Iterator[None]:
        with fx_traceback.set_current_meta(
            node, f"Interpreter_{self.__class__.__name__}"
        ):
            yield

    @compatibility(is_backward_compatible=True)
    def run_node(self, n: Node) -> Any:
        """
        Run a specific node ``n`` and return the result.
        Calls into placeholder, get_attr, call_function,
        call_method, call_module, or output depending
        on ``node.op``

        Args:
            n (Node): The Node to execute

        Returns:
            Any: The result of executing ``n``
        """
        log.debug("run_node %s", LazyString(lambda: _format_fx_node(n)))
        with self._set_current_node(n):
            args, kwargs = self.fetch_args_kwargs_from_env(n)
            if not isinstance(args, tuple):
                raise AssertionError(f"Expected args to be tuple, got {type(args)}")
            if not isinstance(kwargs, dict):
                raise AssertionError(f"Expected kwargs to be dict, got {type(kwargs)}")
            return getattr(self, n.op)(n.target, args, kwargs)

    # Main Node running APIs
    @compatibility(is_backward_compatible=True)
    def placeholder(
        self, target: "Target", args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any:
        """
        Execute a ``placeholder`` node. Note that this is stateful:
        ``Interpreter`` maintains an internal iterator over
        arguments passed to ``run`` and this method returns
        next() on that iterator.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/main/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Returns:
            Any: The argument value that was retrieved.
        """
        if not isinstance(target, str):
            raise AssertionError(f"Expected target to be str, got {type(target)}")
        if target.startswith("*"):
            # For a starred parameter e.g. `*args`, retrieve all
            # remaining values from the args list.
            return list(self.args_iter)
        else:
            try:
                return next(self.args_iter)
            except StopIteration as si:
                if len(args) > 0:
                    return args[0]
                else:
                    raise RuntimeError(
                        f"Expected positional argument for parameter {target}, but one was not passed in!"
                    ) from si

    @compatibility(is_backward_compatible=True)
    def get_attr(
        self, target: "Target", args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any:
        """
        Execute a ``get_attr`` node. Will retrieve an attribute
        value from the ``Module`` hierarchy of ``self.module``.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/main/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return:
            Any: The value of the attribute that was retrieved
        """
        if not isinstance(target, str):
            raise AssertionError(f"Expected target to be str, got {type(target)}")
        return self.fetch_attr(target)

    @compatibility(is_backward_compatible=True)
    def call_function(
        self, target: "Target", args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any:
        """
        Execute a ``call_function`` node and return the result.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/main/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return
            Any: The value returned by the function invocation
        """
        if isinstance(target, str):
            raise AssertionError("target should not be a string for call_function")

        # Execute the function and return the result
        return target(*args, **kwargs)

    @compatibility(is_backward_compatible=True)
    def call_method(
        self, target: "Target", args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any:
        """
        Execute a ``call_method`` node and return the result.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/main/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return
            Any: The value returned by the method invocation
        """
        # args[0] is the `self` object for this method call
        self_obj, *args_tail = args

        # Execute the method and return the result
        if not isinstance(target, str):
            raise AssertionError(f"Expected target to be str, got {type(target)}")
        return getattr(self_obj, target)(*args_tail, **kwargs)

    @compatibility(is_backward_compatible=True)
    def call_module(
        self, target: "Target", args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any:
        """
        Execute a ``call_module`` node and return the result.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/main/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return
            Any: The value returned by the module invocation
        """
        # Retrieve executed args and kwargs values from the environment

        # Execute the method and return the result
        if not isinstance(target, str):
            raise AssertionError(f"Expected target to be str, got {type(target)}")
        submod = self.fetch_attr(target)

        return submod(*args, **kwargs)

    @compatibility(is_backward_compatible=True)
    def output(
        self, target: "Target", args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any:
        """
        Execute an ``output`` node. This really just retrieves
        the value referenced by the ``output`` node and returns it.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/main/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation

        Return:
            Any: The return value referenced by the output node
        """
        return args[0]

    # Helper methods
    @compatibility(is_backward_compatible=True)
    def fetch_attr(self, target: str) -> Any:
        """
        Fetch an attribute from the ``Module`` hierarchy of ``self.module``.

        Args:
            target (str): The fully-qualified name of the attribute to fetch

        Return:
            Any: The value of the attribute.
        """
        target_atoms = target.split(".")
        attr_itr = self.module
        for i, atom in enumerate(target_atoms):
            if not hasattr(attr_itr, atom):
                raise RuntimeError(
                    f"Node referenced nonexistent target {'.'.join(target_atoms[: i + 1])}"
                )
            attr_itr = getattr(attr_itr, atom)
        return attr_itr

    @compatibility(is_backward_compatible=True)
    def fetch_args_kwargs_from_env(
        self, n: Node
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """
        Fetch the concrete values of ``args`` and ``kwargs`` of node ``n``
        from the current execution environment.

        Args:
            n (Node): The node for which ``args`` and ``kwargs`` should be fetched.

        Return:
            Tuple[Tuple, Dict]: ``args`` and ``kwargs`` with concrete values for ``n``.
        """
        args = self.map_nodes_to_values(n.args, n)
        if not isinstance(args, tuple):
            raise AssertionError(f"Expected args to be tuple, got {type(args)}")
        kwargs = self.map_nodes_to_values(n.kwargs, n)
        if not isinstance(kwargs, dict):
            raise AssertionError(f"Expected kwargs to be dict, got {type(kwargs)}")
        return args, kwargs

    @compatibility(is_backward_compatible=True)
    def map_nodes_to_values(self, args: Argument, n: Node) -> Argument:
        """
        Recursively descend through ``args`` and look up the concrete value
        for each ``Node`` in the current execution environment.

        Args:
            args (Argument): Data structure within which to look up concrete values

            n (Node): Node to which ``args`` belongs. This is only used for error reporting.
        """

        def load_arg(n_arg: Node) -> Any:
            if n_arg not in self.env:
                raise RuntimeError(
                    f"Node {n} referenced nonexistent value {n_arg}! Run Graph.lint() "
                    f"to diagnose such issues"
                )
            return self.env[n_arg]

        return map_arg(args, load_arg)


@compatibility(is_backward_compatible=True)
class Transformer(Interpreter):
    """
    ``Transformer`` is a special type of interpreter that produces a
    new ``Module``. It exposes a ``transform()`` method that returns
    the transformed ``Module``. ``Transformer`` does not require
    arguments to run, as ``Interpreter`` does. ``Transformer`` works
    entirely symbolically.

    Example:

        Suppose we want to swap all instances of ``torch.neg`` with
        ``torch.sigmoid`` and vice versa (including their ``Tensor``
        method equivalents). We could subclass ``Transformer`` like so::

            class NegSigmSwapXformer(Transformer):
                def call_function(
                    self,
                    target: "Target",
                    args: Tuple[Argument, ...],
                    kwargs: Dict[str, Any],
                ) -> Any:
                    if target is torch.sigmoid:
                        return torch.neg(*args, **kwargs)
                    return super().call_function(target, args, kwargs)

                def call_method(
                    self,
                    target: "Target",
                    args: Tuple[Argument, ...],
                    kwargs: Dict[str, Any],
                ) -> Any:
                    if target == "neg":
                        call_self, *args_tail = args
                        return call_self.sigmoid(*args_tail, **kwargs)
                    return super().call_method(target, args, kwargs)


            def fn(x):
                return torch.sigmoid(x).neg()


            gm = torch.fx.symbolic_trace(fn)

            transformed: torch.nn.Module = NegSigmSwapXformer(gm).transform()
            input = torch.randn(3, 4)
            torch.testing.assert_close(transformed(input), torch.neg(input).sigmoid())

    Args:
        module (GraphModule): The ``Module`` to be transformed.
    """

    @compatibility(is_backward_compatible=True)
    def __init__(self, module: GraphModule) -> None:
        super().__init__(module)
        self.new_graph = Graph()
        self.new_graph.set_codegen(module.graph._codegen)

        class TransformerTracer(Tracer):
            def __init__(self, graph: Graph) -> None:
                super().__init__()
                self.graph = graph
                self.tensor_attrs: dict[torch.Tensor, str] = {}  # type: ignore[assignment]

            def is_leaf_module(self, _: torch.nn.Module, __: str) -> bool:
                return True

        self.tracer = TransformerTracer(self.new_graph)
        self.tracer.root = module

    @compatibility(is_backward_compatible=True)
    def placeholder(
        self, target: "Target", args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Proxy:
        """
        Execute a ``placeholder`` node. In ``Transformer``, this is
        overridden to insert a new ``placeholder`` into the output
        graph.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/main/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation
        """
        if not isinstance(target, str):
            raise AssertionError(f"Expected target to be str, got {type(target)}")
        default_value = next(iter(args)) if args else inspect.Signature.empty
        return Proxy(
            self.new_graph.placeholder(target, default_value=default_value), self.tracer
        )

    @compatibility(is_backward_compatible=True)
    def get_attr(
        self, target: "Target", args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Proxy:
        """
        Execute a ``get_attr`` node. In ``Transformer``, this is
        overridden to insert a new ``get_attr`` node into the output
        graph.

        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/main/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation
        """
        if not isinstance(target, str):
            raise AssertionError(f"Expected target to be str, got {type(target)}")
        return self.tracer.create_proxy("get_attr", target, args, kwargs)

    @compatibility(is_backward_compatible=True)
    def call_module(
        self, target: "Target", args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any:
        # Override so that the leaf module policy from `self.tracer` is respected.
        if not isinstance(target, str):
            raise AssertionError(f"Expected target to be str, got {type(target)}")
        submod = self.fetch_attr(target)
        return self.tracer.call_module(submod, submod.forward, args, kwargs)

    @compatibility(is_backward_compatible=True)
    def call_function(
        self, target: "Target", args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any:
        # Override so that functions that were wrapped are still wrapped.
        return self.tracer.create_proxy("call_function", target, args, kwargs)

    @compatibility(is_backward_compatible=True)
    def transform(self) -> GraphModule:
        """
        Transform ``self.module`` and return the transformed
        ``GraphModule``.
        """
        with fx_traceback.preserve_node_meta():
            result = super().run(enable_io_processing=False)
        if result is not None:

            def strip_proxy(a: Argument | Proxy) -> Any:
                return a.node if isinstance(a, Proxy) else a

            new_output_node = self.new_graph.output(map_aggregate(result, strip_proxy))
            # also preserve the metadata from the old output node, if it exists
            old_output_node = list(self.graph.nodes)[-1]
            if old_output_node.op != "output":
                raise AssertionError(
                    f"Expected output node, got op={old_output_node.op}"
                )
            for k, v in old_output_node.meta.items():
                new_output_node.meta[k] = v

        return _make_graph_module(self.module, self.new_graph)
<|endoftext|>import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple, TYPE_CHECKING

import torch

from ._compatibility import compatibility
from ._symbolic_trace import symbolic_trace
from .graph import Graph
from .graph_module import GraphModule
from .node import Node


if TYPE_CHECKING:
    from .passes.utils.matcher_with_name_node_map_utils import InternalMatch

__all__ = [
    "Match",
    "replace_pattern",
    "replace_pattern_with_filters",
    "ReplacedPatterns",
]


@compatibility(is_backward_compatible=True)
class Match(NamedTuple):
    # Node from which the match was found
    anchor: Node
    # Maps nodes in the pattern subgraph to nodes in the larger graph
    nodes_map: dict[Node, Node]


@compatibility(is_backward_compatible=False)
@dataclass
class ReplacedPatterns:
    # Node from which the match was found
    anchor: Node
    # Maps nodes in the pattern subgraph to nodes in the larger graph
    nodes_map: dict[Node, Node]
    # List of nodes that were added into the graph
    replacements: list[Node]


def _replace_attributes(gm: GraphModule, replacement: torch.nn.Module) -> None:
    gm.delete_all_unused_submodules()

    if isinstance(replacement, GraphModule):
        replacement.graph.lint()

    def try_get_attr(gm: torch.nn.Module, target: str) -> Any | None:
        module_path, _, attr_name = target.rpartition(".")
        try:
            mod: torch.nn.Module = gm.get_submodule(module_path)
        except AttributeError:
            return None
        attr = getattr(mod, attr_name, None)
        return attr

    for node in gm.graph.nodes:
        if node.op == "call_module" or node.op == "get_attr":
            gm_attr = try_get_attr(gm, node.target)
            replacement_attr = try_get_attr(replacement, node.target)

            # CASE 1: This target already exists as an attribute in our
            # result GraphModule. Whether or not it exists in
            # `replacement`, the existing submodule takes precedence.
            if gm_attr is not None:
                continue

            # CASE 2: The target exists as an attribute in `replacement`
            # only, so we need to copy it over.
            elif replacement_attr is not None:
                new_attr = copy.deepcopy(replacement_attr)
                if isinstance(replacement_attr, torch.nn.Module):
                    gm.add_submodule(node.target, new_attr)
                else:
                    setattr(gm, node.target, new_attr)

            # CASE 3: The target doesn't exist as an attribute in `gm`
            # or `replacement`
            else:
                raise RuntimeError(
                    'Attempted to create a "',
                    node.op,
                    '" node during subgraph rewriting '
                    f"with target {node.target}, but "
                    "the referenced attribute does not "
                    "exist in the replacement GraphModule",
                )

    gm.graph.lint()


@compatibility(is_backward_compatible=True)
def replace_pattern(
    gm: GraphModule,
    pattern: Callable[..., Any] | GraphModule,
    replacement: Callable[..., Any] | GraphModule,
) -> list[Match]:
    """
    Matches all possible non-overlapping sets of operators and their
    data dependencies (``pattern``) in the Graph of a GraphModule
    (``gm``), then replaces each of these matched subgraphs with another
    subgraph (``replacement``).

    Args:
        ``gm``: The GraphModule that wraps the Graph to operate on
        ``pattern``: The subgraph to match in ``gm`` for replacement
        ``replacement``: The subgraph to replace ``pattern`` with

    Returns:
        List[Match]: A list of ``Match`` objects representing the places
        in the original graph that ``pattern`` was matched to. The list
        is empty if there are no matches. ``Match`` is defined as:

        .. code-block:: python

            class Match(NamedTuple):
                # Node from which the match was found
                anchor: Node
                # Maps nodes in the pattern subgraph to nodes in the larger graph
                nodes_map: Dict[Node, Node]

    Examples:

    .. code-block:: python

        import torch
        from torch.fx import symbolic_trace, subgraph_rewriter


        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, w1, w2):
                m1 = torch.cat([w1, w2]).sum()
                m2 = torch.cat([w1, w2]).sum()
                return x + torch.max(m1) + torch.max(m2)


        def pattern(w1, w2):
            return torch.cat([w1, w2])


        def replacement(w1, w2):
            return torch.stack([w1, w2])


        traced_module = symbolic_trace(M())

        subgraph_rewriter.replace_pattern(traced_module, pattern, replacement)

    The above code will first match ``pattern`` in the ``forward``
    method of ``traced_module``. Pattern-matching is done based on
    use-def relationships, not node names. For example, if you had
    ``p = torch.cat([a, b])`` in ``pattern``, you could match
    ``m = torch.cat([a, b])`` in the original ``forward`` function,
    despite the variable names being different (``p`` vs ``m``).

    The ``return`` statement in ``pattern`` is matched based on its
    value only; it may or may not match to the ``return`` statement in
    the larger graph. In other words, the pattern doesn't have to extend
    to the end of the larger graph.

    When the pattern is matched, it will be removed from the larger
    function and replaced by ``replacement``. If there are multiple
    matches for ``pattern`` in the larger function, each non-overlapping
    match will be replaced. In the case of a match overlap, the first
    found match in the set of overlapping matches will be replaced.
    ("First" here being defined as the first in a topological ordering
    of the Nodes' use-def relationships. In most cases, the first Node
    is the parameter that appears directly after ``self``, while the
    last Node is whatever the function returns.)

    One important thing to note is that the parameters of the
    ``pattern`` Callable must be used in the Callable itself,
    and the parameters of the ``replacement`` Callable must match
    the pattern. The first rule is why, in the above code block, the
    ``forward`` function has parameters ``x, w1, w2``, but the
    ``pattern`` function only has parameters ``w1, w2``. ``pattern``
    doesn't use ``x``, so it shouldn't specify ``x`` as a parameter.
    As an example of the second rule, consider replacing

    .. code-block:: python

        def pattern(x, y):
            return torch.neg(x) + torch.relu(y)

    with

    .. code-block:: python

        def replacement(x, y):
            return torch.relu(x)

    In this case, ``replacement`` needs the same number of parameters
    as ``pattern`` (both ``x`` and ``y``), even though the parameter
    ``y`` isn't used in ``replacement``.

    After calling ``subgraph_rewriter.replace_pattern``, the generated
    Python code looks like this:

    .. code-block:: python

        def forward(self, x, w1, w2):
            stack_1 = torch.stack([w1, w2])
            sum_1 = stack_1.sum()
            stack_2 = torch.stack([w1, w2])
            sum_2 = stack_2.sum()
            max_1 = torch.max(sum_1)
            add_1 = x + max_1
            max_2 = torch.max(sum_2)
            add_2 = add_1 + max_2
            return add_2
    """
    match_and_replacements = _replace_pattern(gm, pattern, replacement)
    return [
        Match(anchor=m.anchor, nodes_map=m.nodes_map) for m in match_and_replacements
    ]


# Experimental API, not backward compatible
@compatibility(is_backward_compatible=False)
def replace_pattern_with_filters(
    gm: GraphModule,
    pattern: Callable[..., Any] | Graph | GraphModule,
    replacement: Callable[..., Any] | Graph | GraphModule | None = None,
    match_filters: list[Callable[["InternalMatch", Graph, Graph], bool]] | None = None,
    ignore_literals: bool = False,
    # Placed at the end to avoid breaking backward compatibility
    replacement_callback: Callable[["InternalMatch", Graph, Graph], Graph]
    | None = None,
    node_name_match: str = "",
) -> list[ReplacedPatterns]:
    """
    See replace_pattern for documentation. This function is an overload with an additional match_filter argument.

    Args:
        ``match_filters``: A list of functions that take in
            (match: InternalMatch, original_graph: Graph, pattern_graph: Graph) and return a boolean indicating
            whether the match satisfies the condition.
            See matcher_utils.py for definition of InternalMatch.
        ``replacement_callback``: A function that takes in a match and returns a
            Graph to be used as the replacement. This allows you to construct a
            replacement graph based on the match.
        ``node_name_match``: Node name to match. If not empty, it will try to match the node name.
    """

    return _replace_pattern(
        gm,
        pattern,
        replacement,
        match_filters,
        ignore_literals,
        replacement_callback,
        node_name_match,
    )


def _replace_pattern(
    gm: GraphModule,
    pattern: Callable[..., Any] | Graph | GraphModule,
    replacement: Callable[..., Any] | Graph | GraphModule | None = None,
    match_filters: list[Callable[["InternalMatch", Graph, Graph], bool]] | None = None,
    ignore_literals: bool = False,
    # Placed at the end to avoid breaking backward compatibility
    replacement_callback: Callable[["InternalMatch", Graph, Graph], Graph]
    | None = None,
    node_name_match: str = "",
) -> list[ReplacedPatterns]:
    from torch.fx.passes.utils.matcher_utils import InternalMatch, SubgraphMatcher

    if match_filters is None:
        match_filters = []

    # Get the graphs for `gm`, `pattern`, `replacement`
    original_graph: Graph = gm.graph

    if isinstance(pattern, GraphModule):
        pattern_graph = pattern.graph
    elif isinstance(pattern, Graph):
        pattern_graph = pattern
    else:
        pattern_graph = symbolic_trace(pattern).graph  # type: ignore[arg-type]

    matcher = SubgraphMatcher(
        pattern_graph,
        match_output=False,
        match_placeholder=False,
        remove_overlapping_matches=True,
        ignore_literals=ignore_literals,
    )
    _matches: list[InternalMatch] = matcher.match(
        original_graph, node_name_match=node_name_match
    )

    # Filter out matches that don't match the filter
    _matches = [
        m
        for m in _matches
        if all(
            match_filter(m, original_graph, pattern_graph)
            for match_filter in match_filters
        )
    ]

    if isinstance(replacement, GraphModule):
        common_replacement_graph = replacement.graph
    elif isinstance(replacement, Graph):
        common_replacement_graph = replacement
    elif callable(replacement):
        common_replacement_graph = symbolic_trace(replacement).graph
    else:
        if replacement_callback is None:
            raise AssertionError(
                "Must provide either a replacement GraphModule or a replacement callback"
            )
        common_replacement_graph = None  # type: ignore[assignment]

    # As we progressively replace nodes, we'll need to keep track of how the match results should change
    match_changed_node: dict[Node, Node] = {}

    match_and_replacements = []
    for match in _matches:
        if replacement_callback is not None:
            replacement_graph = replacement_callback(
                match, original_graph, pattern_graph
            )
        else:
            if common_replacement_graph is None:
                raise AssertionError(
                    "Must provide either a replacement GraphModule or a replacement callback"
                )
            replacement_graph = common_replacement_graph
        replacement_placeholders = [
            n for n in replacement_graph.nodes if n.op == "placeholder"
        ]

        # Build connecting between replacement graph's input and original graph input producer node

        # Initialize `val_map` with mappings from placeholder nodes in
        # `replacement` to their corresponding node in `original_graph`
        if len(match.placeholder_nodes) != len(replacement_placeholders):
            raise AssertionError(
                f"Placeholder count mismatch: {len(match.placeholder_nodes)} vs "
                f"{len(replacement_placeholders)}"
            )
        val_map: dict[Node, Node] = {}
        for rn, gn in zip(replacement_placeholders, match.placeholder_nodes):
            if isinstance(gn, Node):
                val_map[rn] = match_changed_node.get(gn, gn)
                if gn != val_map[rn]:
                    # Update match.placeholder_nodes and match.nodes_map with the node that replaced gn
                    gn_ind = match.placeholder_nodes.index(gn)
                    match.placeholder_nodes[gn_ind] = match_changed_node[gn]
                    map_key = list(match.nodes_map.keys())[
                        list(match.nodes_map.values()).index(gn)
                    ]
                    match.nodes_map[map_key] = match_changed_node[gn]
            else:
                val_map[rn] = gn

        # Copy the replacement graph over
        user_nodes: set[Node] = set()
        for n in match.returning_nodes:
            user_nodes.update(n.users)

        first_user_node = None
        if len(user_nodes) == 0:
            first_user_node = None
        elif len(user_nodes) == 1:
            first_user_node = next(iter(user_nodes))
        else:
            # If there are multiple user nodes, we need to find the first user node
            # in the current execution order of the `original_graph`
            for n in original_graph.nodes:
                if n in user_nodes:
                    first_user_node = n
                    break

        first_next_node = None
        if first_user_node is None:
            # no users, so we insert the replacement graph before the first next
            # node of returning nodes
            next_node = None
            for n in reversed(original_graph.nodes):
                if n in match.returning_nodes:
                    first_next_node = next_node
                    break
                else:
                    next_node = n
        insert_point = (
            first_user_node if first_user_node is not None else first_next_node
        )
        if insert_point is None:
            raise AssertionError("The insert point can't be None")
        with original_graph.inserting_before(insert_point):
            copied_returning_nodes = original_graph.graph_copy(
                replacement_graph, val_map
            )

        if isinstance(copied_returning_nodes, Node):
            copied_returning_nodes = (copied_returning_nodes,)

        # Get a list of nodes that have been replaced into the graph
        replacement_nodes: list[Node] = [
            v for v in val_map.values() if v not in match.placeholder_nodes
        ]

        # Hook the output Node of the replacement subgraph in to the
        # original Graph at the correct location
        if len(match.returning_nodes) != len(copied_returning_nodes):  # type: ignore[arg-type]
            raise AssertionError(
                f"Returning nodes count mismatch: {len(match.returning_nodes)} vs "
                f"{len(copied_returning_nodes)}"  # pyrefly: ignore [bad-argument-type]
            )
        for gn, copied_node in zip(match.returning_nodes, copied_returning_nodes):  # type: ignore[arg-type]
            # pyrefly: ignore [bad-argument-type]
            gn.replace_all_uses_with(copied_node)
            # pyrefly: ignore [unsupported-operation]
            match_changed_node[gn] = copied_node
        # Remove the original nodes
        for node in reversed(pattern_graph.nodes):
            if node.op != "placeholder" and node.op != "output":
                gn = match.nodes_map[node]
                gm.graph.erase_node(gn)

        match_and_replacements.append(
            ReplacedPatterns(
                anchor=match.anchors[0],
                nodes_map=match.nodes_map,
                replacements=replacement_nodes,
            )
        )

    # Update the passed-in GraphModule to reflect the new state of
    # `original_graph`
    gm.recompile()

    # If `replacement` was an nn.Module, we'll need to make sure that
    # all the submodules have been copied over correctly
    if isinstance(replacement, torch.nn.Module):
        _replace_attributes(gm, replacement)

    return match_and_replacements
<|endoftext|># Nodes represent a definition of a value in our graph of operators.
import builtins
import inspect
import logging
import operator
import types
import typing
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Optional, TYPE_CHECKING, TypeAlias, Union
from typing_extensions import ParamSpec, TypeVar

import torch
from torch._C import _fx_map_aggregate, _fx_map_arg, _NodeBase
from torch.fx.operator_schemas import (
    ArgsKwargsPair,
    normalize_function,
    normalize_module,
)
from torch.utils._dtype_abbrs import dtype_abbrs

from .._ops import ops as _ops
from ._compatibility import compatibility


if TYPE_CHECKING:
    from .graph import Graph

__all__ = ["Node", "map_arg", "map_aggregate", "has_side_effect"]

log = logging.getLogger(__name__)

BaseArgumentTypes = Union[  # noqa: UP007
    str,
    int,
    float,
    bool,
    complex,
    torch.dtype,
    torch.Tensor,
    torch.device,
    torch.memory_format,
    torch.layout,
    torch._ops.OpOverload,
    torch.SymInt,
    torch.SymBool,
    torch.SymFloat,
]
base_types = typing.get_args(BaseArgumentTypes)

Target: TypeAlias = Callable[..., Any] | str

Argument = Optional[  # noqa: UP045
    Union[
        tuple["Argument", ...],
        Sequence["Argument"],
        Mapping[str, "Argument"],
        slice,  # Slice[Argument, Argument, Argument], but slice is not a templated type in typing
        range,
        "Node",
        BaseArgumentTypes,
    ]
]
# pyrefly: ignore [invalid-annotation]
ArgumentT = TypeVar("ArgumentT", bound=Argument)
_P = ParamSpec("_P")
_R = TypeVar("_R")

_legal_ops = dict.fromkeys(
    [
        "placeholder",
        "call_method",
        "call_module",
        "call_function",
        "get_attr",
        "output",
        "root",
    ]
)

# Dynamo is unable to trace global set[Callable].__contains__.
# See https://github.com/pytorch/pytorch/issues/145761. Since we only have
# a handful of ops so switch to list of callables.
_side_effectful_need_to_be_preserved_pre_dispatch: list[Callable[..., Any]] = [
    torch._C._set_grad_enabled,
    torch.amp._enter_autocast,
    torch.amp._exit_autocast,
]

# TODO: Either refactor this into 2 functions 1 dce for functional graphs and 1 dce for all graphs,
# or add logic to correctly mark all inplace ops as side effectful.
#
# NOTE: For new operators, please do not add to this set!
# Instead, consider using the effects system via
# torch.library._register_effectful_op() for operators.
#
# This _side_effectful_functions set is only for:
# - Legacy functions that aren't operators (e.g., profiler ops, asserts)
# - Things that cannot be marked via the normal effects system
_side_effectful_functions: set[Callable[..., Any]] = {
    torch._assert,
    torch._assert_async,
    _ops.aten._assert_async.msg,
    _ops.aten._assert_scalar.default,
    _ops.aten._assert_tensor_metadata.default,
    _ops.aten.sym_constrain_range.default,
    _ops.aten.sym_constrain_range_for_size.default,
    _ops.profiler._record_function_enter,
    _ops.profiler._record_function_enter.default,
    _ops.profiler._record_function_enter_new,
    _ops.profiler._record_function_enter_new.default,
    _ops.profiler._record_function_exit,
    _ops.profiler._record_function_exit._RecordFunction,
    _ops.inductor.accumulate_grad_.default,
    operator.setitem,
    *_side_effectful_need_to_be_preserved_pre_dispatch,
}

if hasattr(_ops.inductor, "resize_storage_bytes_"):
    _side_effectful_functions.add(_ops.inductor.resize_storage_bytes_.default)


def _device_annotation(device: torch.device) -> str:
    # Render a tensor's device for graph printing. Under compile-on-one-rank the device
    # index is rank-specific; drop it when it is the current default device for the
    # accelerator (e.g. "cuda:3" -> "cuda") so the printed graph is byte-identical across
    # ranks. A non-current index keeps its index, preserving debuggability.
    import torch.compiler.config as compiler_config

    if compiler_config.compile_on_one_rank and device.index is not None:
        acc = torch.accelerator.current_accelerator()
        if (
            acc is not None
            and device.type == acc.type
            and device.index == torch.accelerator.current_device_index()
        ):
            return device.type
    return str(device)


@compatibility(is_backward_compatible=False)
def has_side_effect(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    """
    Registers a function to not be dead code eliminated by
    fx.graph.eliminate_dead_code

    NOTE: For new operators, please do not add to this set!
    Instead, consider using the effects system via
    torch.library._register_effectful_op() for operators.

    This _side_effectful_functions set is only for:
    - Legacy functions that aren't operators (e.g., profiler ops, asserts)
    - Things that cannot be marked via the normal effects system
    """
    _side_effectful_functions.add(fn)
    return fn


# this is fixed on master, WAR for 1.5
def _find_module_of_method(orig_method: Callable[..., Any]) -> str:
    name = orig_method.__name__
    module = orig_method.__module__
    if module is not None:
        return module
    for guess in [torch, torch.nn.functional]:
        if getattr(guess, name, None) is orig_method:
            return guess.__name__
    raise RuntimeError(f"cannot find module for {orig_method}")


# Borrowed from CPython typing module
# https://github.com/python/cpython/blob/f90dc36c15d7fee0efaf6d39e97be0bdf2683e93/Lib/typing.py#L156
def _type_repr(obj: object) -> str:
    """Return the repr() of an object, special-casing types (internal helper).
    If obj is a type, we return a shorter version than the default
    type.__repr__, based on the module and qualified name, which is
    typically enough to uniquely identify a type.  For everything
    else, we fall back on repr(obj).
    """
    # Extension: If we don't ignore GenericAlias then `list[int]` will print
    # simply "list".
    if isinstance(obj, type) and not isinstance(obj, types.GenericAlias):
        if obj.__module__ == "builtins":
            return obj.__qualname__
        return f"{obj.__module__}.{obj.__qualname__}"
    if obj is ...:
        return "..."
    if isinstance(obj, types.FunctionType):
        return obj.__name__
    return repr(obj)


def _get_qualified_name(func: Callable[..., Any]) -> str:
    # things like getattr just appear in builtins
    if getattr(builtins, func.__name__, None) is func:
        return func.__name__
    # torch.Tensor.{fn}
    if (
        isinstance(func, (types.MethodDescriptorType, types.WrapperDescriptorType))
        and func is getattr(torch.Tensor, func.__name__, None)
    ) or (
        func.__module__ == torch._tensor.__name__
        and func.__qualname__ == f"Tensor.{func.__name__}"
    ):
        return f"torch.Tensor.{func.__name__}"
    name = func.__name__

    if name == "<lambda>":
        # For lambdas, try to get their defining name in the module
        try:
            name = inspect.getsource(func).split("=")[0].strip()
        except Exception as e:
            raise RuntimeError("Unable to represent lambda") from e
    module = _find_module_of_method(func)
    module = module.replace(
        "torch._ops", "torch.ops"
    )  # WAR for bug in how torch.ops assigns module
    # Fixup segment_reduce mismatch
    if module == "torch" and name == "segment_reduce":
        name = "_" + name
    if module == "torch.nn.functional" and name in ("_ScalingType", "_SwizzleType"):
        name = name.removeprefix("_")
    return f"{module}.{name}"


def _format_arg(arg: object, max_list_len: float = float("inf")) -> str:
    if hasattr(arg, "_custom_fx_repr_fn"):
        return arg._custom_fx_repr_fn()
    elif isinstance(arg, list):
        items = ", ".join(
            _format_arg(a) for idx, a in enumerate(arg) if idx < max_list_len
        )
        maybe_len = (
            "" if len(arg) < max_list_len + 1 else f", ...[total_len={len(arg)}]"
        )
        return f"[{items}{maybe_len}]"
    elif isinstance(arg, tuple):
        items = ", ".join(
            _format_arg(a) for idx, a in enumerate(arg) if idx < max_list_len
        )
        maybe_len = (
            "" if len(arg) < max_list_len + 1 else f", ...[total_len={len(arg)}]"
        )
        maybe_comma = "," if len(arg) == 1 else ""
        return f"({items}{maybe_comma}{maybe_len})"
    elif isinstance(arg, dict):
        items_str = ", ".join(f"{k}: {_format_arg(v)}" for k, v in arg.items())
        return f"{{{items_str}}}"

    if isinstance(arg, Node):
        return "%" + str(arg)
    elif isinstance(arg, torch.device):
        return _device_annotation(arg)
    else:
        return str(arg)


@compatibility(is_backward_compatible=True)
class Node(_NodeBase):
    """
    ``Node`` is the data structure that represents individual operations within
    a ``Graph``. For the most part, Nodes represent callsites to various entities,
    such as operators, methods, and Modules (some exceptions include nodes that
    specify function inputs and outputs). Each ``Node`` has a function specified
    by its ``op`` property. The ``Node`` semantics for each value of ``op`` are as follows:

    - ``placeholder`` represents a function input. The ``name`` attribute specifies the name this value will take on.
      ``target`` is similarly the name of the argument. ``args`` holds either: 1) nothing, or 2) a single argument
      denoting the default parameter of the function input. ``kwargs`` is don't-care. Placeholders correspond to
      the function parameters (e.g. ``x``) in the graph printout.
    - ``get_attr`` retrieves a parameter from the module hierarchy. ``name`` is similarly the name the result of the
      fetch is assigned to. ``target`` is the fully-qualified name of the parameter's position in the module hierarchy.
      ``args`` and ``kwargs`` are don't-care
    - ``call_function`` applies a free function to some values. ``name`` is similarly the name of the value to assign
      to. ``target`` is the function to be applied. ``args`` and ``kwargs`` represent the arguments to the function,
      following the Python calling convention
    - ``call_module`` applies a module in the module hierarchy's ``forward()`` method to given arguments. ``name`` is
      as previous. ``target`` is the fully-qualified name of the module in the module hierarchy to call.
      ``args`` and ``kwargs`` represent the arguments to invoke the module on, *excluding the self argument*.
    - ``call_method`` calls a method on a value. ``name`` is as similar. ``target`` is the string name of the method
      to apply to the ``self`` argument. ``args`` and ``kwargs`` represent the arguments to invoke the module on,
      *including the self argument*
    - ``output`` contains the output of the traced function in its ``args[0]`` attribute. This corresponds to the "return" statement
      in the Graph printout.
    """

    _args: tuple["Argument", ...]
    _kwargs: dict[str, "Argument"]
    graph: "Graph"
    # unique name of value being created
    name: str
    # the kind of operation = placeholder|call_method|call_module|call_function|get_attr
    op: str
    # for method/module/function, the name of the method/module/function/attr
    # being invoked, e.g add, layer1, or torch.add
    target: "Target"
    # All `Node`-valued inputs. Key is the Node, value is don't-care.
    # The public API for this is `all_input_nodes`, this private attribute
    # should not be accessed directly.
    _input_nodes: dict["Node", None]
    # All of the nodes that use the value produced by this Node
    # Note one user may correspond to several uses, e.g. the node for ``x + x``
    # would appear once here, but represents two uses.
    # Is a dict to act as an "ordered set". Keys are significant, value don't-care
    users: dict["Node", None]
    # Type expression representing the output value of this node.
    # This should contain the same class of Type objects that would appear
    # as type annotations for function inputs/outputs.
    #
    # For placeholder nodes, this value will be used to type-annotate the
    # generated function parameters.
    # For the return node, this value will be used to type-annotate the
    # generated function return type. (Note this is a special case. ``return``
    # does not produce a value, it's more of a notation. Thus, this value
    # describes the type of args[0] in the ``return`` node.
    # TODO: narrow this to TensorType | _DynType | None
    type: Any | None
    _sort_key: Any
    # If set, use this fn to print this node
    _repr_fn: Callable[["Node"], str] | None
    # Dictionary to store metadata passes need to do their
    # transformations. This metadata is preserved across node copies
    meta: dict[str, Any]

    @compatibility(is_backward_compatible=True)
    def __init__(
        self,
        graph: "Graph",
        name: str,
        op: str,
        target: "Target",
        args: tuple["Argument", ...],
        kwargs: dict[str, "Argument"],
        return_type: Any | None = None,
    ) -> None:
        """
        Instantiate an instance of ``Node``. Note: most often, you want to use the
        Graph APIs, i.e. ``Graph.call_module``, ``Graph.call_method``, etc. rather
        than instantiating a ``Node`` directly.

        Args:
            graph (Graph): The ``Graph`` to which this ``Node`` should belong.

            name (str): The name to which the output of this ``Node`` should be assigned

            op (str): The opcode for this ``Node``. Can be one of 'placeholder',
                'call_method', 'call_module', 'call_function', 'get_attr',
                'output'

            target ('Target'): The target this op should call. See the broader
                ``Node`` docstring for more details.

            args (Tuple['Argument']): The args to be passed to ``target``

            kwargs (Dict[str, 'Argument']): The kwargs to be passed to ``target``

            return_type (Optional[Any]): The python type expression representing the
                type of the output of this node. This field can be used for
                annotation of values in the generated code or for other types
                of analyses.
        """
        if op == "call_function":
            if not callable(target):
                raise ValueError(
                    f"Node [graph = {graph}, name = '{name}'] target {target} has type {torch.typename(target)} "
                    "but a Callable is expected"
                )
        else:
            if op not in _legal_ops:
                raise AssertionError(f"op '{op}' is not in _legal_ops")
            if not isinstance(target, str):
                raise ValueError(
                    f"Node [graph = {graph}, name = '{name}'] target {target} has type {torch.typename(target)} "
                    "but a str is expected"
                )
        super().__init__(graph, name, op, target, return_type)
        self._update_args_kwargs(args, kwargs)

    def __getstate__(self) -> dict[str, Any]:
        return {
            **self.__dict__,
            "graph": self.graph,
            "name": self.name,
            "op": self.op,
            "target": self.target,
            "type": self.type,
            "_sort_key": self._sort_key,
            "_args": self._args,
            "_kwargs": self._kwargs,
            "_erased": self._erased,
            "_prev": self._prev,
            "_next": self._next,
            "_input_nodes": self._input_nodes,
            "users": self.users,
            "_repr_fn": self._repr_fn,
            "meta": self.meta,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        for k, v in state.items():
            setattr(self, k, v)

    @property
    def next(self) -> "Node":
        """
        Returns the next ``Node`` in the linked list of Nodes.

        Returns:

            The next ``Node`` in the linked list of Nodes.
        """
        return self._next

    @property
    def prev(self) -> "Node":
        """
        Returns the previous ``Node`` in the linked list of Nodes.

        Returns:

            The previous ``Node`` in the linked list of Nodes.
        """
        return self._prev

    @compatibility(is_backward_compatible=True)
    def prepend(self, x: "Node") -> None:
        """
        Insert x before this node in the list of nodes in the graph. Example::

            Before: p -> self
                    bx -> x -> ax
            After:  p -> x -> self
                    bx -> ax

        Args:
            x (Node): The node to put before this node. Must be a member of the same graph.
        """

        self._prepend(x)

    @compatibility(is_backward_compatible=True)
    def append(self, x: "Node") -> None:
        """
        Insert ``x`` after this node in the list of nodes in the graph.
        Equivalent to ``self.next.prepend(x)``

        Args:
            x (Node): The node to put after this node. Must be a member of the same graph.
        """

        self._next._prepend(x)

    @property
    def args(self) -> tuple[Argument, ...]:
        """
        The tuple of arguments to this ``Node``. The interpretation of arguments
        depends on the node's opcode. See the :class:`Node` docstring for more
        information.

        Assignment to this property is allowed. All accounting of uses and users
        is updated automatically on assignment.
        """
        return self._args

    @args.setter
    def args(self, a: tuple[Argument, ...]) -> None:
        """
        Set the tuple of arguments to this Node. The interpretation of arguments
        depends on the node's opcode. See the ``fx.Graph`` docstring for more
        information.
        """
        # DO NOT CALL `_update_args_kwargs` directly. The correct way to
        # set `args` is via direct assignment, i.e. `node.args = new_args`
        self._update_args_kwargs(a, self._kwargs)

    @property
    def kwargs(self) -> dict[str, Argument]:
        """
        The dict of keyword arguments to this ``Node``. The interpretation of arguments
        depends on the node's opcode. See the :class:`Node` docstring for more
        information.

        Assignment to this property is allowed. All accounting of uses and users
        is updated automatically on assignment.
        """
        return self._kwargs

    @kwargs.setter
    def kwargs(self, k: dict[str, Argument]) -> None:
        """
        Set the dict of kwargs to this Node. The interpretation of arguments
        depends on the node's opcode. See the ``fx.Graph`` docstring for more
        information.
        """
        # DO NOT CALL `_update_args_kwargs` directly. The correct way to
        # set `args` is via direct assignment, i.e. `node.kwargs = new_kwargs`
        self._update_args_kwargs(self._args, k)

    @property
    def all_input_nodes(self) -> list["Node"]:
        """
        Return all Nodes that are inputs to this Node. This is equivalent to
        iterating over ``args`` and ``kwargs`` and only collecting the values that
        are Nodes.

        Returns:

            List of ``Nodes`` that appear in the ``args`` and ``kwargs`` of this
            ``Node``, in that order.
        """
        return list(self._input_nodes.keys())

    @compatibility(is_backward_compatible=True)
    def update_arg(self, idx: int, arg: Argument) -> None:
        """
        Update an existing positional argument to contain the new value
        ``arg``. After calling, ``self.args[idx] == arg``.

        Args:

            idx (int): The index into ``self.args`` of the element to update
            arg (Argument): The new argument value to write into ``args``
        """
        args = list(self.args)
        args[idx] = arg
        self.args = tuple(args)

    @compatibility(is_backward_compatible=True)
    def insert_arg(self, idx: int, arg: Argument) -> None:
        """
        Insert a positional argument to the argument list with given index.

        Args:

            idx (int): The index of the element in ``self.args`` to be inserted before.
            arg (Argument): The new argument value to insert into ``args``
        """
        if not (0 <= idx <= len(self.args)):
            raise AssertionError(
                f"insert_args index must be between 0 and len(self.args), got {idx}"
            )
        args_left = self.args[:idx]
        args_right = self.args[idx:]

        self._args = args_left + (arg,) + args_right

        _new_input_nodes: dict[Node, None] = {}
        _fx_map_arg(arg, _new_input_nodes.setdefault)

        for new_use in _new_input_nodes:
            if new_use not in self._input_nodes:
                self._input_nodes.setdefault(new_use)
                new_use.users.setdefault(self)

    @compatibility(is_backward_compatible=True)
    def update_kwarg(self, key: str, arg: Argument) -> None:
        """
        Update an existing keyword argument to contain the new value
        ``arg``. After calling, ``self.kwargs[key] == arg``.

        Args:

            key (str): The key in ``self.kwargs`` of the element to update
            arg (Argument): The new argument value to write into ``kwargs``
        """
        self.kwargs = {**self.kwargs, key: arg}

    @property
    def stack_trace(self) -> str | None:
        """
        Return the Python stack trace that was recorded during tracing, if any.
        When traced with fx.Tracer, this property is usually populated by
        `Tracer.create_proxy`. To record stack traces during tracing for debug purposes,
        set `record_stack_traces = True` on the `Tracer` instance.
        When traced with dynamo, this property will be populated by default by
        `OutputGraph.create_proxy`.

        stack_trace would have the innermost frame at the end of the string.
        """
        return self.meta.get("stack_trace", None)

    @stack_trace.setter
    def stack_trace(self, trace: str | None) -> None:
        self.meta["stack_trace"] = trace

    def __repr__(self) -> str:
        if self._repr_fn:
            return self._repr_fn(self)
        return self.name

    @staticmethod
    def _pretty_print_target(target: object) -> str:
        """
        Make target printouts more user-friendly.
        1) builtins will be printed as `builtins.xyz`
        2) operators will be printed as `operator.xyz`
        3) other callables will be printed with qualified name, e.g. torch.add
        """
        if isinstance(target, str):
            return target
        if hasattr(target, "__module__"):
            name = getattr(target, "__name__", None)
            if name is None:
                # Just to be defensive, if we don't have `__name__`, get the
                # qualname. Not sure if this happens for any members of `operator`
                # or `builtins`. This fallback path is not as good, since e.g.
                # things in `operator` have `_operator` as their __module__.
                # TODO: THIS IS BROKEN: _get_qualified_name calls `__name__`
                return _get_qualified_name(target)  # type: ignore[arg-type]
            if target.__module__ == "builtins":
                return f"builtins.{name}"
            elif target.__module__ == "_operator":
                return f"operator.{name}"
        return _get_qualified_name(target)  # type: ignore[arg-type]

    @compatibility(is_backward_compatible=True)
    def format_node(
        self,
        placeholder_names: list[str] | None = None,
        maybe_return_typename: list[str] | None = None,
        *,
        include_tensor_metadata: bool = False,
    ) -> str | None:
        """
        Return a descriptive string representation of ``self``.

        This method can be used with no arguments as a debugging
        utility.

        This function is also used internally in the ``__str__`` method
        of ``Graph``. Together, the strings in ``placeholder_names``
        and ``maybe_return_typename`` make up the signature of the
        autogenerated ``forward`` function in this Graph's surrounding
        GraphModule. ``placeholder_names`` and ``maybe_return_typename``
        should not be used otherwise.

        Args:
            placeholder_names: A list that will store formatted strings
                representing the placeholders in the generated
                ``forward`` function. Internal use only.
            maybe_return_typename: A single-element list that will store
                a formatted string representing the output of the
                generated ``forward`` function. Internal use only.
            include_tensor_metadata: Whether to include tensor metadata

        Returns:
            str: If 1) we're using ``format_node`` as an internal helper
                in the ``__str__`` method of ``Graph``, and 2) ``self``
                is a placeholder Node, return ``None``. Otherwise,
                return a  descriptive string representation of the
                current Node.
        """
        if self.op == "placeholder":
            if not isinstance(self.target, str):
                raise AssertionError(
                    f"Expected target to be str for placeholder, got {type(self.target)}"
                )
            arg_str = self.target
            arg_str += arg_str + f": {_type_repr(self.type)}" if self.type else ""
            if placeholder_names:
                placeholder_names.append(arg_str)
                return None
            maybe_typename = f"{_type_repr(self.type)} " if self.type else ""
            default_val = "(default=" + str(self.args[0]) + ")" if self.args else ""
            return f"%{self.name} : {maybe_typename}[num_users={len(self.users)}] = {self.op}[target={self.target}]{default_val}"
        elif self.op == "get_attr":
            maybe_typename = (
                f"{_type_repr(self.type)} " if self.type is not None else ""
            )
            return (
                f"%{self.name} : {maybe_typename}[num_users={len(self.users)}] = "
                f"{self.op}[target={self._pretty_print_target(self.target)}]"
            )
        elif self.op == "output":
            if self.type and maybe_return_typename:
                maybe_return_typename[0] = f" -> {_type_repr(self.type)}"
            return f"return {self.args[0]}"
        else:

            def stringify_shape(shape: Iterable[Any]) -> str:
                return f"[{', '.join([str(x) for x in shape])}]"

            meta_val = self.meta.get(
                "val",
                self.meta.get("tensor_meta", self.meta.get("example_value", None)),
            )
            type_annotation = ""
            if (
                include_tensor_metadata
                and isinstance(meta_val, torch.Tensor)
                and meta_val.layout
                not in (
                    torch.sparse_csc,
                    torch.sparse_csr,
                )
            ):
                stride_annotation = f"{stringify_shape(meta_val.stride())}"
                device_annotation = _device_annotation(meta_val.device)
                type_annotation = (
                    f'Tensor "{dtype_abbrs[meta_val.dtype]}{stringify_shape(meta_val.shape)}'
                    f'{stride_annotation}{device_annotation}"'
                )
            else:
                type_annotation = (
                    f"{_type_repr(self.type)} " if self.type is not None else ""
                )
            return (
                f"%{self.name} : {type_annotation}[num_users={len(self.users)}] = "
                f"{self.op}[target={self._pretty_print_target(self.target)}]("
                f"args = {_format_arg(self.args)}, kwargs = {_format_arg(self.kwargs)})"
            )

    @compatibility(is_backward_compatible=True)
    def replace_all_uses_with(
        self,
        replace_with: "Node",
        delete_user_cb: Callable[["Node"], bool] | None = None,
        *,
        propagate_meta: bool = False,
    ) -> list["Node"]:
        """
        Replace all uses of ``self`` in the Graph with the Node ``replace_with``.

        Args:

            replace_with (Node): The node to replace all uses of ``self`` with.
            delete_user_cb (Callable): Callback that is called to determine
              whether a given user of the self node should be removed.
            propagate_meta (bool): Whether or not to copy all properties
              on the .meta field of the original node onto the replacement node.
              For safety, this is only valid to do if the replacement node
              doesn't already have an existing .meta field.

        Returns:

            The list of Nodes on which this change was made.
        """
        if propagate_meta:
            if len(replace_with.meta) != 0:
                raise AssertionError(
                    "Called node.replace_all_uses_with(replace_with, propagate_meta=True), "
                    "but replace_with already has .meta keys"
                )
            for k, v in self.meta.items():
                replace_with.meta[k] = v
        to_process = [*self.users]
        replace_hooks = getattr(self.graph.owning_module, "_replace_hooks", None)
        result = []
        for use_node in to_process:
            if delete_user_cb is not None and not delete_user_cb(use_node):
                continue
            result.append(use_node)
            if replace_hooks:
                for replace_hook in replace_hooks:
                    replace_hook(old=self, new=replace_with.name, user=use_node)

            use_node._replace_input_with(self, replace_with)
        return result

    @compatibility(is_backward_compatible=False)
    def is_impure(self, impure_random: bool = True) -> bool:
        """
        Returns whether this op is impure, i.e. if its op is a placeholder or
        output, or if a call_function or call_module which is impure.

        Args:
            impure_random (bool): Whether to treat rand op as impure.

        Returns:

            bool: If the op is impure or not.
        """
        # Placeholders and outputs are always impure for DCE purposes
        if self.op in {"placeholder", "output"}:
            return True

        # Check if an impure module.
        if self.op == "call_module":
            if self.graph.owning_module is None:
                raise AssertionError(
                    "self.graph.owning_module not set for purity check"
                )
            if not isinstance(self.target, str):
                raise AssertionError(
                    f"Expected str target for call_module, got {type(self.target)}"
                )
            target_mod = self.graph.owning_module.get_submodule(self.target)
            if target_mod is None:
                raise AssertionError(
                    f"Did not find expected submodule target {self.target}"
                )
            # NOTE: here we can end up considering GraphModule submodules pure,
            # even if they contain impure ops. It may not be safe to change
            # because this function is used by graph.eliminate_dead_code,
            # and some users depend on current elimination behavior.
            return getattr(target_mod, "_is_impure", False)

        # For call_function, delegate to the unified has_side_effects function
        if self.op == "call_function":
            from torch._library.utils import is_impure

            return is_impure(
                self.target,  # pyrefly: ignore[bad-argument-type]
                args=self.args,
                kwargs=self.kwargs,
                impure_random=impure_random,
            )

        return False

    @compatibility(is_backward_compatible=False)
    def normalized_arguments(
        self,
        root: torch.nn.Module,
        arg_types: tuple[Any] | None = None,
        kwarg_types: dict[str, Any] | None = None,
        normalize_to_only_use_kwargs: bool = False,
    ) -> ArgsKwargsPair | None:
        """
        Returns normalized arguments to Python targets. This means that
        `args/kwargs` will be matched up to the module/functional's
        signature and return exclusively kwargs in positional order
        if `normalize_to_only_use_kwargs` is true.
        Also populates default values. Does not support positional-only
        parameters or varargs parameters.

        Supports module calls.

        May require `arg_types` and `kwarg_types` in order to disambiguate overloads.

        Args:
            root (torch.nn.Module): Module upon which to resolve module targets.
            arg_types (Optional[Tuple[Any]]): Tuple of arg types for the args
            kwarg_types (Optional[Dict[str, Any]]): Dict of arg types for the kwargs
            normalize_to_only_use_kwargs (bool): Whether to normalize to only use kwargs.

        Returns:

            Returns NamedTuple ArgsKwargsPair, or `None` if not successful.
        """
        if self.op == "call_function":
            if not callable(self.target):
                raise AssertionError(
                    f"Expected callable target, got {type(self.target)}"
                )
            return normalize_function(
                self.target,
                self.args,  # type: ignore[arg-type]
                self.kwargs,
                arg_types,
                kwarg_types,
                normalize_to_only_use_kwargs=normalize_to_only_use_kwargs,
            )
        elif self.op == "call_module":
            if not isinstance(self.target, str):
                raise AssertionError(
                    f"Expected str target for call_module, got {type(self.target)}"
                )
            return normalize_module(
                root,
                self.target,
                self.args,  # type: ignore[arg-type]
                self.kwargs,
                normalize_to_only_use_kwargs=normalize_to_only_use_kwargs,
            )

        return None

    @compatibility(is_backward_compatible=True)
    def replace_input_with(self, old_input: "Node", new_input: "Node") -> None:
        """
        Loop through input nodes of ``self``, and replace all instances of
        ``old_input`` with ``new_input``.

        Args:

            old_input (Node): The old input node to be replaced.
            new_input (Node): The new input node to replace ``old_input``.
        """

        m = self.graph.owning_module
        if getattr(m, "_replace_hooks", None):
            for replace_hook in m._replace_hooks:
                replace_hook(old=old_input, new=new_input.name, user=self)

        self._replace_input_with(old_input, new_input)

    def _rename(self, candidate: str) -> None:
        if candidate == self.name:
            return
        name = self.graph._graph_namespace.create_name(candidate, None)
        self.name = name
        self.graph._graph_namespace._rename_object(self, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "name" and hasattr(self, "name"):
            m = self.graph.owning_module
            if getattr(m, "_replace_hooks", None):
                if not isinstance(value, str):
                    raise AssertionError(f"Expected value to be str, got {type(value)}")
                for user in self.users:
                    for replace_hook in m._replace_hooks:
                        replace_hook(old=self, new=value, user=user)
        update = False
        if (
            hasattr(self, name)
            and hasattr(self.graph, "_find_nodes_lookup_table")
            and self in self.graph._find_nodes_lookup_table
        ):
            update = True
            self.graph._find_nodes_lookup_table.remove(self)
        object.__setattr__(self, name, value)
        if update:
            self.graph._find_nodes_lookup_table.insert(self)


@compatibility(is_backward_compatible=True)
def map_arg(a: ArgumentT, fn: Callable[[Node], Argument]) -> ArgumentT:
    """
    Apply fn recursively to each Node appearing in arg.

    arg may be a list, tuple, slice, or dict with string keys: the return value will
    have the same type and structure.
    """
    if not callable(fn):
        raise AssertionError("torch.fx.map_arg(a, fn): fn must be a callable")
    return _fx_map_arg(a, fn)


@compatibility(is_backward_compatible=True)
def map_aggregate(a: ArgumentT, fn: Callable[[Argument], Argument]) -> ArgumentT:
    """
    Apply fn recursively to each object appearing in arg.

    arg may be a list, tuple, slice, or dict with string keys: the return value will
    have the same type and structure.
    """
    return _fx_map_aggregate(a, fn)
<|endoftext|>from __future__ import annotations

import builtins
import contextlib
import copy
import enum
import functools
import inspect
import keyword
import logging
import math
import os
import pprint
import re
import types
import typing
import warnings
from collections import defaultdict
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, TYPE_CHECKING

import torch
import torch.utils._pytree as pytree
from torch._C import _fx_map_arg as map_arg, _NodeIter
from torch._library.opaque_object import get_opaque_obj_repr, is_opaque_constant_type
from torch.types import py_sym_types
from torch.utils._dtype_abbrs import dtype_abbrs

from . import _pytree as fx_pytree
from ._compatibility import compatibility
from .immutable_collections import immutable_dict
from .node import (
    _device_annotation,
    _get_qualified_name,
    _type_repr,
    Argument,
    Node,
    Target,
)
from .tensor_type import TensorType


log = logging.getLogger(__name__)

__all__ = ["PythonCode", "CodeGen", "Graph"]


if TYPE_CHECKING:
    import sympy

    from ._symbolic_trace import Tracer
    from .graph_module import GraphModule


# Mapping of builtins to their `typing` equivalent.
# (PEP585: See D68459095 test plan)
_origin_type_map = {
    list: typing.List,  # noqa: UP006
    dict: typing.Dict,  # noqa: UP006
    set: typing.Set,  # noqa: UP006
    frozenset: typing.FrozenSet,  # noqa: UP006
    tuple: typing.Tuple,  # noqa: UP006
}

_legal_ops = dict.fromkeys(
    ["call_function", "call_method", "get_attr", "call_module", "placeholder", "output"]
)


# Signature for functions that transform the body (`list[str]`) of the
# generated code
TransformCodeFunc = Callable[[list[str]], list[str]]


class _CustomBuiltin(NamedTuple):
    """Additional objs that we add to every graph's globals.

    The repr() for some standard library objects is not valid Python code without
    an import. For common objects of this sort, we bundle them in the globals of
    every FX graph.
    """

    # How to import this object from the standard library.
    import_str: str
    # The actual object, produced from that import string.
    obj: object


# Combined dict of disallowed variable names so we can check with one lookup
_illegal_names = {k: object() for k in keyword.kwlist}
_illegal_names.update(builtins.__dict__)  # can't shadow a builtin name

_custom_builtins: dict[str, _CustomBuiltin] = {}


def _register_custom_builtin(name: str, import_str: str, obj: object) -> None:
    _custom_builtins[name] = _CustomBuiltin(import_str, obj)
    _illegal_names[name] = obj


_register_custom_builtin("inf", "from math import inf", math.inf)
_register_custom_builtin("nan", "from math import nan", math.nan)
_register_custom_builtin("NoneType", "NoneType = type(None)", type(None))
_register_custom_builtin("torch", "import torch", torch)
_register_custom_builtin("device", "from torch import device", torch.device)
_register_custom_builtin("fx_pytree", "import torch.fx._pytree as fx_pytree", fx_pytree)
_register_custom_builtin("pytree", "import torch.utils._pytree as pytree", pytree)


def _is_magic(x: str) -> bool:
    return x.startswith("__") and x.endswith("__")


def _snake_case(s: str) -> str:
    """
    Transforms the given string ``s`` to a Python-style variable name

    Examples:
        ``mod.snake_case`` -> ``mod.snake_case``
        ``mod.pascalCase``-> ``mod.pascal_case``
        ``mod.ALL_CAPS`` -> ``mod.all_caps``
    """
    return _snake_case_sub(s).lower()


# Replace occurrences where a lowercase letter is followed by an uppercase letter
_snake_case_sub = functools.partial(re.compile(r"(?<=[a-z])([A-Z])").sub, r"_\1")

# Find chars that can't be in a Python identifier
_illegal_char_regex = re.compile("[^0-9a-zA-Z_]+")

# Combined check for variable names:
# 1) Checks name is not empty
# 2) Checks first character is not a digit
# 3) Checks name has no illegal characters (_illegal_char_regex)
# 3) Splits off the number suffix (if present)
_name_regex = re.compile(r"^([a-zA-Z_][0-9a-zA-Z_]*?)(?:_(\d+))?$")

# starts with torch but does not start with torch._dynamo. or torch._inductor.
_torch_but_not_dynamo = re.compile(
    r"^torch(?:\.(?!_dynamo\.|_inductor\.)[^.]+)*$"
).fullmatch


def _is_from_torch(obj: object) -> bool:
    module_name = getattr(obj, "__module__", None)
    if module_name is not None:
        return _torch_but_not_dynamo(module_name) is not None

    name = getattr(obj, "__name__", None)
    # exclude torch because torch.torch.torch.torch works. idk mang
    if name is not None and name != "torch":
        for guess in [torch, torch.nn.functional]:
            if getattr(guess, name, None) is obj:
                return True

    return False


class _Namespace:
    """A context for associating names uniquely with objects.

    The following invariants are enforced:
    - Each object gets a single name.
    - Each name is unique within a given namespace.
    - Names generated do not shadow builtins, unless the object is indeed that builtin.
    """

    def __init__(self) -> None:
        self._obj_to_name: dict[object, str] = {}
        self._used_names: set[str] = set()
        self._base_count: dict[str, int] = {}

    def create_name(self, candidate: str, obj: object | None) -> str:
        """Create a unique name.

        Arguments:
            candidate: used as the basis for the unique name, relevant to the user.
            obj: If not None, an object that will be associated with the unique name.
        """
        if obj is not None and obj in self._obj_to_name:
            return self._obj_to_name[obj]

        # optimistically check if candidate is already a valid name
        match = _name_regex.match(candidate)
        if match is None:
            # delete all characters that are illegal in a Python identifier
            candidate = _illegal_char_regex.sub("_", candidate)

            if not candidate:
                candidate = "_unnamed"

            if candidate[0].isdigit():
                candidate = f"_{candidate}"

            match = _name_regex.match(candidate)
            if match is None:
                raise AssertionError(
                    f"Name regex failed to match candidate: {candidate}"
                )

        base, num = match.group(1, 2)
        if num is None or candidate in self._used_names:
            # Look up `base` to match the key used in the store on line below;
            # using `candidate` misses when it has a numeric suffix, making
            # the while-loop quadratic.
            num = self._base_count.get(base, 0)
            if _illegal_names.get(candidate, obj) is not obj:
                num += 1
                candidate = f"{base}_{num}"
                # assume illegal names don't end in _\d so no need to check again
        else:
            num = int(num)

        while candidate in self._used_names:
            num += 1
            candidate = f"{base}_{num}"

        self._used_names.add(candidate)
        self._base_count[base] = num
        if obj is not None:
            self._obj_to_name[obj] = candidate
        return candidate

    def associate_name_with_obj(self, name: str, obj: object) -> None:
        """Associate a unique name with an object.

        Neither `name` nor `obj` should be associated already.
        """
        maybe_existing = self._obj_to_name.setdefault(obj, name)
        if maybe_existing is not name:
            raise AssertionError("obj is already associated")

    def _rename_object(self, obj: object, name: str) -> None:
        if obj not in self._obj_to_name:
            raise AssertionError(f"Object {obj} is not in _obj_to_name")
        self._obj_to_name[obj] = name
        self._used_names.add(name)


@compatibility(is_backward_compatible=True)
@dataclass
class PythonCode:
    """
    Represents all the information necessary to exec or save a graph as Python code.
    """

    # Python source code for the forward function definition.
    src: str
    # Values in global scope during execution of `src_def`.
    globals: dict[str, Any]
    # Optional mapping from the forward function's line number to
    # node index. Line number starts at the prologue (i.e. forward()).
    _lineno_map: dict[int, int | None] | None
    # The line number of prologue in fn_code
    _prologue_start: int = 0


def _format_target(base: str, target: str) -> str:
    elems = target.split(".")
    r = base
    for e in elems:
        if not e.isidentifier():
            r = f'getattr({r}, "{e}")'
        else:
            r = f"{r}.{e}"
    return r


class _InsertPoint:
    def __init__(self, graph: Graph, new_insert: Callable[..., None]) -> None:
        self.graph = graph
        self.orig_insert, graph._insert = graph._insert, new_insert

    def __enter__(self) -> None:
        pass

    def __exit__(
        self,
        type: type[BaseException] | None,
        value: BaseException | None,
        tb: types.TracebackType | None,
    ) -> None:
        self.graph._insert = self.orig_insert


class _node_list:
    def __init__(
        self, graph: Graph, direction: Literal["_prev", "_next"] = "_next"
    ) -> None:
        if direction not in ("_next", "_prev"):
            raise AssertionError(
                f"direction must be '_next' or '_prev', got {direction}"
            )
        self.graph = graph
        self.direction = direction

    def __len__(self) -> int:
        return self.graph._len

    # TODO: These should return Iterator[Node], but doing so causes ~350
    # downstream pyrefly errors because Node.target is typed as
    # Callable[..., Any] | str and pyrefly can't narrow it based on
    # node.op checks (e.g. `if node.op == "call_module": node.target`
    # should be str but pyrefly doesn't support that narrowing).
    def __iter__(self) -> Iterator[Any]:
        return _NodeIter(self.graph._root, self.direction == "_prev")

    def __reversed__(self) -> Iterator[Any]:
        return _NodeIter(self.graph._root, self.direction == "_next")


class _PyTreeInfo(NamedTuple):
    """
    Contains extra info stored when we're using Pytrees
    """

    orig_args: list[str]
    in_spec: pytree.TreeSpec
    out_spec: pytree.TreeSpec | None


@dataclass(frozen=True)
class _ParsedStackTrace:
    """
    Represents the top-most frame of a parsed stack trace
    """

    file: str
    lineno: str
    name: str
    code: str

    def get_summary_str(self) -> str:
        return f"File: {self.file}:{self.lineno} in {self.name}, code: {self.code}"


# get File:lineno code from stack_trace
def _parse_stack_trace(
    stack_trace: str, filter_fn: Callable[[str, str, str], bool] | None = None
) -> _ParsedStackTrace | None:
    if stack_trace is None:
        return None
    pattern = re.compile(r"^File \"(.+)\", line (\d+), in (.+)$")
    lines = stack_trace.strip().split("\n")
    # stacktrace should have innermost frame last, so we
    # iterate backwards to find the first line that starts
    # with 'File '
    for idx in range(len(lines) - 2, -1, -1):
        line = lines[idx].strip()
        matches = pattern.match(line)
        if matches:
            file = matches.group(1)
            lineno = matches.group(2)
            name = matches.group(3)
            # next line should be the code
            code = lines[idx + 1].strip()
            if filter_fn and not filter_fn(file, name, code):
                continue
            return _ParsedStackTrace(file, lineno, name, code)
    return None


@compatibility(is_backward_compatible=False)
class CodeGen:
    # This is an override hook so we can customize the SymNode printer.
    _sym_repr: Callable[[torch.types.PySymType], str] = lambda x: repr(x)

    def __init__(self) -> None:
        self._body_transformer: TransformCodeFunc | None = None
        self._func_name: str = "forward"

    def _format_multiline_args(self, args: list[str]) -> str:
        """Helper to format function arguments in expanded multiline format."""
        return "".join(self._format_single_arg(arg) for arg in args)

    def _format_single_arg(self, arg: str) -> str:
        """Helper to format a single argument with optional comment."""
        if "#" in arg:
            arg_part, comment_part = arg.split("#", 1)
            return f"    {arg_part.rstrip()},  # {comment_part.lstrip()}\n"
        else:
            return f"    {arg},\n"

    def _get_delimiters(self, container: Sequence[object]) -> tuple[str, str]:
        """Helper to get opening and closing delimiters for containers."""
        return ("(", ")") if isinstance(container, tuple) else ("[", "]")

    def _format_multiline_container(
        self,
        items: Sequence[object],
        descs: Sequence[str] | None = None,
        prefix: str = "",
        repr_fn: Callable[[object], str] | None = None,
    ) -> str:
        """Helper to format containers (lists/tuples) in multiline format."""
        ldelim, rdelim = self._get_delimiters(items)
        desc_trailers = self._get_desc_trailers(items, descs)
        if repr_fn is None:
            repr_fn = repr

        return (
            f"{prefix}{ldelim}\n"
            + "".join(
                f"    {repr_fn(item)},{trailer}\n"
                for item, trailer in zip(items, desc_trailers)
            )
            + f"{rdelim}"
        )

    def _get_desc_trailers(
        self, items: Sequence[object], descs: Sequence[str] | None
    ) -> list[str]:
        """Helper to generate description trailers for items."""
        if descs is None:
            return [""] * len(items)
        return [f"  # {desc}" for desc in descs]

    def _call_method_with_signature_check(
        self, method: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Helper to call a method with optional parameters based on signature."""
        sig = inspect.signature(method)
        # Filter kwargs to only include parameters that exist in the method signature
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return method(*args, **filtered_kwargs)

    def gen_fn_def(
        self,
        free_vars: list[str],
        maybe_return_annotation: str,
        *,
        expanded_def: bool = False,
    ) -> str:
        """
        Given the free variables and a return annotation, generates the beginning of the FX function.
        By default, `gen_fn_def(['a', 'b'], '') == 'def {self._func_name}(a, b):'`
        """
        # If the original function didn't have self as its first argument, we
        # would have added it.
        if len(free_vars) == 0 or free_vars[0] != "self":
            free_vars.insert(0, "self")

        if expanded_def:
            args_formatted = self._format_multiline_args(free_vars)
            return (
                f"def {self._func_name}(\n{args_formatted}){maybe_return_annotation}:"
            )
        else:
            return f"def {self._func_name}({', '.join(free_vars)}){maybe_return_annotation}:"

    def generate_output(
        self,
        output_args: Argument,
        *,
        descs: Sequence[str] | None = None,
        repr_fn: Callable[[object], str] | None = None,
    ) -> str:
        """
        Given the output arguments, generates the return statement of the FX function.
        Note: The returned statement should not be indented.
        """
        if repr_fn is None:
            repr_fn = repr
        if descs is not None and isinstance(output_args, (list, tuple)):
            return self._format_multiline_container(
                output_args, descs, "return ", repr_fn=repr_fn
            )
        else:
            return f"return {repr_fn(output_args)}"

    def process_inputs(self, *args: Any) -> Any:
        """
        Transforms the inputs so that the graph can take them as arguments, as
        non-default codegen may result in the inputs to the function being
        different from the inputs to the graph.

        If the graph was directly runnable, this invariant should hold true
        `f.graph.process_outputs(f.graph(*f.graph.process_inputs(*inputs))) == f(*inputs)`
        """
        return args

    def process_outputs(self, outputs: Any) -> Any:
        """
        Transforms the outputs of the graph to be identical to the codegen.

        See ``process_inputs`` for more details.
        """
        return outputs

    def additional_globals(self) -> list[tuple[str, Any]]:
        """
        If your codegen uses extra global values, add tuples of (identifier,reference to the value) here.
        For example, return ['List', typing.List] if you need ``List`` in the global context.
        """
        return []

    def _gen_python_code(
        self,
        nodes: _node_list,
        root_module: str,
        namespace: _Namespace,
        *,
        verbose: bool = False,
        include_stride: bool = False,
        include_device: bool = False,
        colored: bool = False,
        # Render each argument on its own line
        expanded_def: bool = False,
        record_func: bool = False,
        additional_meta: list[str] | None = None,
    ) -> PythonCode:
        free_vars: list[str] = []
        body: list[str] = []
        globals_: dict[str, Any] = {}
        wrapped_fns: dict[str, None] = {}

        # Wrap string in list to pass by reference
        maybe_return_annotation: list[str] = [""]
        include_stride = include_stride or (
            os.environ.get("FX_GRAPH_SHOW_STRIDE", "0") == "1"
        )
        include_device = include_device or (
            os.environ.get("FX_GRAPH_SHOW_DEVICE", "0") == "1"
        )
        include_meta = os.environ.get("FX_GRAPH_SHOW_META", "0") == "1"

        def add_global(name_hint: str, obj: Any) -> str:
            """Add an obj to be tracked as a global.

            We call this for names that reference objects external to the
            Graph, like functions or types.

            Returns: the global name that should be used to reference 'obj' in generated source.
            """
            if (
                _is_from_torch(obj) and obj != torch.device
            ):  # to support registering torch.device
                # HACK: workaround for how torch custom ops are registered. We
                # can't import them like normal modules so they must retain their
                # fully qualified name.
                return _get_qualified_name(obj)

            # normalize the name hint to get a proper identifier
            global_name = namespace.create_name(name_hint, obj)

            if global_name in globals_:
                if globals_[global_name] != obj:
                    raise AssertionError(
                        f"Global name {global_name} already assigned to different object"
                    )
                return global_name
            globals_[global_name] = obj
            return global_name

        # Pre-fill the globals table with registered builtins.
        for name, (_, obj) in _custom_builtins.items():
            add_global(name, obj)

        def type_repr(o: object) -> str:
            if o == ():
                # Empty tuple is used for empty tuple type annotation Tuple[()]
                return "()"

            typename = _type_repr(o)
            if isinstance(o, types.UnionType) and "|" in typename:
                # TorchScript's PEP604 parser does not resolve generated globals
                # for nested aliases such as typing_Dict.
                origin_typename = add_global(_type_repr(typing.Union), typing.Union)
                args = [type_repr(arg) for arg in typing.get_args(o)]
                return f"{origin_typename}[{','.join(args)}]"

            if origin_type := getattr(o, "__origin__", None):
                # list[...], typing.List[...], TensorType[...]

                if isinstance(o, typing._GenericAlias):  # type: ignore[attr-defined]
                    # This is a generic pre-PEP585 type, e.g. typing.List[torch.Tensor]
                    origin_type = _origin_type_map.get(origin_type, origin_type)

                origin_typename = add_global(_type_repr(origin_type), origin_type)

                if isinstance(o, TensorType):
                    type_args = o.dims
                else:
                    type_args = typing.get_args(o)
                if type_args:
                    args = [type_repr(arg) for arg in type_args]
                    return f"{origin_typename}[{','.join(args)}]"
                else:
                    return origin_typename

            # Common case: this is a regular module name like 'foo.bar.baz'
            return add_global(typename, o)

        if colored:
            red = _color_fns["red"]
            dim_green = _color_fns["dim_green"]
            dim = _color_fns["dim"]
            dim_blue = _color_fns["dim_blue"]
            blue = _color_fns["blue"]
        else:
            red = _identity
            dim_green = _identity
            dim = _identity
            dim_blue = _identity
            blue = _identity

        def _get_repr(arg: object) -> str:
            if isinstance(arg, Node):  # first because common
                return repr(arg)
            elif isinstance(arg, tuple) and hasattr(arg, "_fields"):
                # Handle NamedTuples (if it has `_fields`) via add_global.
                qualified_name = _get_qualified_name(type(arg))
                global_name = add_global(qualified_name, type(arg))
                return f"{global_name}{repr(tuple(arg))}"
            elif isinstance(
                arg, (torch._ops.OpOverload, torch._ops.HigherOrderOperator)
            ):
                qualified_name = _get_qualified_name(arg)
                global_name = add_global(qualified_name, arg)
                return f"{global_name}"
            elif isinstance(arg, enum.Enum):
                cls = arg.__class__
                clsname = add_global(cls.__name__, cls)
                return f"{clsname}.{arg.name}"
            elif isinstance(arg, complex):
                if (
                    arg.real == 0.0
                    or arg.imag == 0.0
                    or not math.isfinite(arg.real)
                    or not math.isfinite(arg.imag)
                ):
                    # complex.__repr__ is not a safe source representation for
                    # signed zero components, e.g. eval("(-0-1j)") loses the sign.
                    # It's also unsafe for nan/inf imaginary parts: repr produces
                    # "nanj"/"infj" which Python parses as a single identifier.
                    return f"complex({_get_repr(arg.real)}, {_get_repr(arg.imag)})"
                return blue(repr(arg))
            elif isinstance(arg, torch.Tensor):
                size = list(arg.size())
                dtype = str(arg.dtype).split(".")[-1]
                return f"torch.Tensor(size={size}, dtype={dtype})"
            elif isinstance(arg, tuple):
                if len(arg) == 1:
                    return f"({_get_repr(arg[0])},)"
                else:
                    return "(" + ", ".join(_get_repr(a) for a in arg) + ")"
            elif isinstance(arg, list):
                return "[" + ", ".join(_get_repr(a) for a in arg) + "]"
            elif isinstance(arg, slice):
                return f"slice({_get_repr(arg.start)}, {_get_repr(arg.stop)}, {_get_repr(arg.step)})"
            elif is_opaque_constant_type(type(arg)):
                obj_repr, opaque_types = get_opaque_obj_repr(arg)
                for n, t in opaque_types.items():
                    add_global(n, t)
                return obj_repr
            else:
                return blue(repr(arg))

        def _format_args(
            args: tuple[Argument, ...], kwargs: dict[str, Argument]
        ) -> str:
            res = [_get_repr(a) for a in args]
            res.extend([f"{k} = {_get_repr(v)}" for k, v in kwargs.items()])
            return ", ".join(res)

        # Run through reverse nodes and record the first instance of a use
        # of a given node. This represents the *last* use of the node in the
        # execution order of the program, which we will use to free unused
        # values
        node_to_last_use: dict[Node, Node] = {}
        user_to_last_uses: dict[Node, list[Node]] = {}

        def register_last_uses(n: Node, user: Node) -> None:
            if n not in node_to_last_use:
                node_to_last_use[n] = user
                user_to_last_uses.setdefault(user, []).append(n)

        for node in reversed(nodes):
            for input_node in node._input_nodes:
                register_last_uses(input_node, node)

        def delete_unused_values(user: Node) -> None:
            """
            Delete values after their last use. This ensures that values that are
            not used in the remainder of the code are freed and the memory usage
            of the code is optimal.
            """
            if user.op == "placeholder":
                return
            if user.op == "output":
                body.append("\n")
                return
            nodes_to_delete = user_to_last_uses.get(user, [])

            if len(user.users.keys()) == 0:
                # This node is not used by any others. however it's also not
                # removed by DCE since side-effect. We want to free it's outputs
                # right after its execution done to save memory.
                nodes_to_delete.append(user)

            if len(nodes_to_delete):
                to_delete_str = " = ".join(
                    [repr(n) for n in nodes_to_delete] + ["None"]
                )
                body.append(f";  {dim(to_delete_str)}\n")
            else:
                body.append("\n")

        prev_summary_str = None

        def append_stacktrace_summary(node: Node) -> None:
            """
            Append a summary of the stacktrace to the generated code. This is
            useful for debugging.
            """
            nonlocal prev_summary_str

            if node.op not in {"placeholder", "output"}:
                additional_meta_str = ""
                if additional_meta:
                    parts: list[str] = []
                    for key in additional_meta:
                        if key in node.meta:
                            parts.append(f"{key}: {node.meta[key]}")
                    if parts:
                        additional_meta_str = f"# {', '.join(parts)} "

                annotation_str = ""
                annotation = node.meta.get("custom", {})
                annotation_trunc = {}
                if annotation:
                    for key, value in annotation.items():
                        value_str = str(value)
                        if len(value_str) > 40:
                            annotation_trunc[key] = value_str[:40] + "..."
                        else:
                            annotation_trunc[key] = value
                    annotation_str = f" Annotation: {annotation_trunc}"

                stack_trace_str = "No stacktrace found for following nodes"
                if stack_trace := node.stack_trace:
                    if parsed_stack_trace := _parse_stack_trace(stack_trace):
                        stack_trace_str = parsed_stack_trace.get_summary_str()
                        if node.meta.get("autograd_backward", False):
                            stack_trace_str = (
                                f"Backward of forward node: {stack_trace_str}"
                            )

                maybe_recompute_info = ""
                if hasattr(node, "meta") and node.meta:
                    # recompute tags are generated by torch.compile and put in the joint graph.
                    # These tags are load bearing enough that we want them to show up by default
                    # in tlparse, when you run torch.compile.
                    recompute = node.meta.get("recompute", None)
                    ac_graph_id = node.meta.get("ac_graph_id", None)

                    if recompute is not None and ac_graph_id is not None:
                        maybe_recompute_info = (
                            f" ac_graph_id: {str(ac_graph_id)} - {str(recompute.name)}"
                        )
                    elif recompute is not None:
                        maybe_recompute_info = f" recompute: {str(recompute.name)}"
                    elif ac_graph_id is not None:
                        maybe_recompute_info = f" ac_graph_id: {str(ac_graph_id)}"

                summary_str = f"\n{dim(f'{additional_meta_str}#{annotation_str}{maybe_recompute_info} {stack_trace_str}')}\n"

                if summary_str != prev_summary_str:
                    prev_summary_str = summary_str
                    body.append(summary_str)

        def stringify_shape(shape: Iterable[object]) -> str:
            return f"[{', '.join([str(x) for x in shape])}]"

        def emit_node(node: Node) -> None:
            maybe_type_annotation = (
                "" if node.type is None else f" : {type_repr(node.type)}"
            )
            maybe_comment = ""

            if verbose:
                # override annotation with more detailed information
                try:
                    from torch.distributed.tensor._api import DTensor, DTensorSpec

                    dtensorspec_format_shard_order_str = (
                        DTensorSpec.format_shard_order_str
                    )
                except ModuleNotFoundError:
                    DTensor = None  # type: ignore[assignment,misc]
                    dtensorspec_format_shard_order_str = None
                from torch.fx.experimental.proxy_tensor import py_sym_types
                from torch.fx.passes.shape_prop import TensorMetadata

                meta_val = node.meta.get(
                    "val",
                    node.meta.get("tensor_meta", node.meta.get("example_value", None)),
                )

                def _tensor_annotation(t: torch.Tensor) -> str:
                    stride = stringify_shape(t.stride()) if include_stride else ""
                    device = _device_annotation(t.device) if include_device else ""
                    return (
                        f"{red(dtype_abbrs[t.dtype])}"
                        f"{blue(stringify_shape(t.shape))}"
                        f"{dim_blue(stride)}"
                        f"{dim_green(device)}"
                    )

                # use string as annotation, to make it valid python code
                if isinstance(meta_val, torch.Tensor) and meta_val.layout not in (
                    torch.sparse_csc,
                    torch.sparse_csr,
                ):
                    # Fake tensors cause tests to wobble, so do not custom print them.
                    is_plain = type(meta_val) is torch.Tensor or isinstance(
                        meta_val, torch._subclasses.FakeTensor
                    )
                    core = _tensor_annotation(meta_val)
                    if is_plain:
                        maybe_type_annotation = f': "{core}"'
                    elif type(meta_val) is DTensor:
                        if dtensorspec_format_shard_order_str is None:
                            raise AssertionError(
                                "dtensorspec_format_shard_order_str is None for DTensor"
                            )
                        dtensor_meta = dtensorspec_format_shard_order_str(
                            meta_val._spec.placements,  # type: ignore[attr-defined]
                            meta_val._spec.shard_order,  # type: ignore[attr-defined]
                        )
                        cls = meta_val.__class__.__name__
                        maybe_type_annotation = (
                            f': "{cls}({core}, {dim_green(dtensor_meta)})"'
                        )
                    else:
                        cls = meta_val.__class__.__name__
                        maybe_type_annotation = f': "{cls}({core})"'

                elif isinstance(meta_val, py_sym_types):
                    val_str = CodeGen._sym_repr(meta_val)
                    maybe_type_annotation = f': "Sym({val_str})"'

                elif isinstance(meta_val, TensorMetadata):
                    maybe_type_annotation = f': "{dtype_abbrs[meta_val.dtype]}{stringify_shape(meta_val.shape)}"'

            desc = None
            if expanded_def:
                desc = node.meta.get("desc", None)
                if desc is not None and node.op == "placeholder":
                    maybe_comment += f"  # {desc}"
                # output is handled specially

            if include_meta and hasattr(node, "meta") and node.meta:
                body.append('"""\n')
                for k, v in node.meta.items():
                    # use str over repr since repr is susceptible to sympy
                    # errors such as "cannot determine truth value of Relational"
                    # Pretty print the high-level dict with str() for values
                    body.append(
                        f"{k}: {pprint.pformat(str(v), width=80, compact=True)}\n"
                    )
                body.append('"""\n')

            if node.op == "placeholder":
                if not isinstance(node.target, str):
                    raise AssertionError(
                        f"Expected node.target to be str, got {type(node.target)}"
                    )
                maybe_default_arg = (
                    "" if not node.args else f" = {_get_repr(node.args[0])}"
                )
                free_vars.append(
                    f"{node.target}{maybe_type_annotation}{maybe_default_arg}{maybe_comment}"
                )
                raw_name = node.target.replace("*", "")
                if raw_name != repr(node):
                    body.append(f"{repr(node)} = {raw_name}\n")
                return
            elif node.op == "call_method":
                if not isinstance(node.target, str):
                    raise AssertionError(
                        f"Expected node.target to be str for call_method, got {type(node.target)}"
                    )
                body.append(
                    f"{repr(node)}{maybe_type_annotation} = {_format_target(_get_repr(node.args[0]), node.target)}"
                    f"({_format_args(node.args[1:], node.kwargs)})"
                )
                return
            elif node.op == "call_function":
                if not callable(node.target):
                    raise AssertionError(
                        f"Expected node.target to be callable, got {type(node.target)}"
                    )
                # pretty print operators
                if (
                    getattr(node.target, "__module__", "") == "_operator"
                    and node.target.__name__ in magic_methods
                ):
                    if not isinstance(node.args, tuple):
                        raise AssertionError(
                            f"Expected node.args to be tuple, got {type(node.args)}"
                        )
                    body.append(
                        f"{repr(node)}{maybe_type_annotation} = "
                        f"{magic_methods[node.target.__name__].format(*(_get_repr(a) for a in node.args))}"
                    )
                    return

                # pretty print inplace operators; required for jit.script to work properly
                # not currently supported in normal FX graphs, but generated by torchdynamo
                if (
                    getattr(node.target, "__module__", "") == "_operator"
                    and node.target.__name__ in inplace_methods
                ):
                    body.append(
                        f"{inplace_methods[node.target.__name__].format(*(_get_repr(a) for a in node.args))};  "
                        f"{repr(node)}{maybe_type_annotation} = {_get_repr(node.args[0])}"
                    )
                    return

                qualified_name = _get_qualified_name(node.target)
                global_name = add_global(qualified_name, node.target)
                # special case for getattr: node.args could be 2-argument or 3-argument
                # 2-argument: attribute access; 3-argument: fall through to attrib function call with default value
                if (
                    global_name == "getattr"
                    and isinstance(node.args, tuple)
                    and isinstance(node.args[1], str)
                    and node.args[1].isidentifier()
                    and len(node.args) == 2
                ):
                    body.append(
                        f"{repr(node)}{maybe_type_annotation} = {_format_target(_get_repr(node.args[0]), node.args[1])}"
                    )
                    return
                meta = node.meta
                boxed_arg_indices = getattr(node.target, "_boxed_arg_indices", ())
                boxed_arg_indices = meta.get("boxed_arg_indices", boxed_arg_indices)
                boxed_arg_names: dict[int, str] = {}
                for i in boxed_arg_indices:
                    name = f"{node.name}_boxed_arg_{i}"
                    boxed_arg_names[i] = namespace.create_name(name, None)

                if boxed_arg_names:
                    # Generate the boxed arguments on separate lines so the
                    # original node locals can be cleared before the call.
                    boxed_assignments = []
                    for i, name in boxed_arg_names.items():
                        boxed_assignments.append(f"{name} = {_get_repr(node.args[i])}")
                    body.append("\n".join(boxed_assignments))

                    # If this call is the last use of nodes inside the boxed
                    # args, clear those node locals after creating the boxed
                    # args and before emitting the call.
                    boxed_nodes: set[Node] = set()
                    unboxed_nodes: set[Node] = set()
                    boxed_args = tuple(node.args[i] for i in boxed_arg_names)
                    unboxed_args = []
                    for i, arg in enumerate(node.args):
                        if i not in boxed_arg_names:
                            unboxed_args.append(arg)
                    map_arg(boxed_args, boxed_nodes.add)
                    map_arg((unboxed_args, node.kwargs), unboxed_nodes.add)

                    last_uses = user_to_last_uses.get(node, [])
                    only_in_boxes = boxed_nodes - unboxed_nodes
                    nodes_to_delete = [n for n in last_uses if n in only_in_boxes]
                    if nodes_to_delete:
                        last_uses = [n for n in last_uses if n not in nodes_to_delete]
                        user_to_last_uses[node] = last_uses
                        to_delete_str = " = ".join(
                            [repr(n) for n in nodes_to_delete] + ["None"]
                        )
                        body.append(f";  {dim(to_delete_str)}")
                    body.append("\n")

                    # Rewrite the generated call to use the boxed arg locals
                    # instead of rebuilding the boxed args inline.
                    kwargs = node.kwargs.items()
                    call_args = [_get_repr(arg) for arg in node.args]
                    for i, name in boxed_arg_names.items():
                        call_args[i] = name
                    call_args.extend(f"{k} = {_get_repr(v)}" for k, v in kwargs)
                    formatted_args_str = ", ".join(call_args)
                else:
                    formatted_args_str = _format_args(node.args, node.kwargs)

                lhs = f"{repr(node)}{maybe_type_annotation}"
                rhs = f"{global_name}({formatted_args_str})"
                body.append(f"{lhs} = {rhs}")

                if boxed_arg_names:
                    # Clear the generated boxed arg locals after the call. The
                    # call is their only generated use, and these locals are not
                    # FX nodes tracked by normal last-use cleanup.
                    boxed_names_str = " = ".join([*boxed_arg_names.values(), "None"])
                    body.append(f";  {dim(boxed_names_str)}")
                if node.meta.get("is_wrapped", False):
                    wrapped_fns.setdefault(global_name)
                return
            elif node.op == "call_module":
                if not isinstance(node.target, str):
                    raise AssertionError(
                        f"Expected node.target to be str for call_module, got {type(node.target)}"
                    )
                body.append(
                    f"{repr(node)}{maybe_type_annotation} = "
                    f"{_format_target(root_module, node.target)}({_format_args(node.args, node.kwargs)})"
                )
                return
            elif node.op == "get_attr":
                if not isinstance(node.target, str):
                    raise AssertionError(
                        f"Expected node.target to be str for get_attr, got {type(node.target)}"
                    )
                body.append(
                    f"{repr(node)}{maybe_type_annotation} = {_format_target(root_module, node.target)}"
                )
                return
            elif node.op == "output":
                if node.type is not None:
                    maybe_return_annotation[0] = f" -> {type_repr(node.type)}"
                body.append(
                    self._call_method_with_signature_check(
                        self.generate_output,
                        node.args[0],
                        descs=desc if expanded_def else None,
                        repr_fn=_get_repr,
                    )
                )
                return
            raise NotImplementedError(f"node: {node.op} {node.target}")

        if record_func:
            body.append(
                "_rf = torch._C._profiler._RecordFunctionFast('## ENTER_GRAPH_PLACEHOLDER_KEY ##'); _rf.__enter__()\n"
            )
        for i, node in enumerate(nodes):
            # NOTE: emit_node does not emit a string with newline. It depends
            # on delete_unused_values to append one
            if verbose:
                append_stacktrace_summary(node)
            # emit a counter comment to keep track of
            # node index, which will be deleted later
            # after going through _body_transformer
            body.append(f"# COUNTER: {i}\n")
            do_record = record_func and node.op in (
                "call_function",
                "call_method",
                "call_module",
            )
            if do_record:
                # The double hash ## convention is used by post-processing to find the fx markers
                body.append(
                    f"_rf_{node.name} = torch._C._profiler._RecordFunctionFast('## {i} ##'); _rf_{node.name}.__enter__()\n"
                )
            emit_node(node)
            delete_unused_values(node)
            if do_record:
                body.append(f"_rf_{node.name}.__exit__(None, None, None)\n")
        if record_func:
            body.append("_rf.__exit__(None, None, None)\n")

        if len(body) == 0:
            # If the Graph has no non-placeholder nodes, no lines for the body
            # have been emitted. To continue to have valid Python code, emit a
            # single pass statement
            body.append("pass\n")

        if len(wrapped_fns) > 0:
            wrap_name = add_global("wrap", torch.fx.wrap)
            wrap_stmts = "\n".join([f'{wrap_name}("{name}")' for name in wrapped_fns])
        else:
            wrap_stmts = ""

        if self._body_transformer:
            body = self._body_transformer(body)

        for name, value in self.additional_globals():
            add_global(name, value)

        prologue = self._call_method_with_signature_check(
            self.gen_fn_def,
            free_vars,
            maybe_return_annotation[0],
            expanded_def=expanded_def,
        )

        # remove counter and generate lineno to node index mapping
        lineno_map: dict[int, int | None] = {}
        prologue_len = prologue.count("\n") + 1
        new_lines: list[str] = []
        cur_idx = None
        for line in "".join(body).split("\n"):
            counter = _counter_regexp.search(line)
            if counter is not None:
                cur_idx = int(counter.group(1))
            else:
                lineno_map[len(new_lines) + prologue_len] = cur_idx
                new_lines.append(line)

        code = "\n".join(new_lines).lstrip("\n")
        code = "\n".join("    " + line for line in code.split("\n"))

        fn_code = f"""
{wrap_stmts}

{prologue}
{code}"""
        # The +4 accounts for the empty lines before prologue in fn_code
        prologue_start = wrap_stmts.count("\n") + 4
        return PythonCode(
            fn_code,
            globals_,
            _lineno_map=lineno_map,
            _prologue_start=prologue_start,
        )


# Ideally, we'd like to refactor all of the pytree logic into this codegen
# class. Unfortunately, there are 3 areas we currently need extra logic in FX.
# 1. In the initial symbolic trace, the pytree logic is tied up with `concrete_args`.
# 2. In the FX graph, we need to access 2 attributes - in_spec and out_spec.
#    Since we can't access .graph within the FX forward, we need to copy the attribute to the module.
# 3. We currently can't register the pytree imports with `add_global` - not sure why.
class _BoxedCodeGen(CodeGen):
    """
    CodeGen subclass that generates code using the "boxed" calling convention.

    The boxed calling convention takes a single list argument and clears it
    after extracting the arguments, which allows for early deallocation of
    input tensors.
    """

    def gen_fn_def(
        self,
        free_vars: list[str],
        maybe_return_annotation: str,
        *,
        expanded_def: bool = False,
    ) -> str:
        """
        Generate function definition for boxed calling convention.

