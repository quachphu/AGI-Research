    _score_mod_signature,
    BlockMask,
    noop_mask,
)


__all__ = ["PagedAttention"]


def _cdiv(x: int | float | torch.Tensor, multiple: int | float | torch.Tensor):
    return (x + multiple - 1) // multiple


class PagedAttention:
    """
    PagedAttention supports flex attention inference with a large batch size.
    With PagedAttention, a batch of key/value tensors with varying kv length
    is split into tensor blocks of fixed length and cached in a compact way.
    Thus we can avoid redundant memory consumption due to varying kv length and
    support a larger batch size.
    """

    def __init__(
        self,
        n_pages: int,
        page_size: int,
        max_batch_size: int,
        device: str = "cuda",
    ) -> None:
        # number of pages
        self.n_pages = n_pages

        # number of tokens per page
        self.page_size = page_size

        # page table: [batch, logical_block_idx] -> physical_page_idx
        self.page_table = -torch.ones(
            (max_batch_size, self.n_pages), dtype=torch.int64, device=device
        )

        # capacity: batch_idx -> allocated sequence length
        self.capacity = torch.zeros(max_batch_size, dtype=torch.int64, device=device)

        # index of empty pages that is available for allocation
        self.empty_pages = list(range(n_pages - 1, -1, -1))

        # mapping from physical page index to logical page index
        self.physical_to_logical = -torch.ones(
            (max_batch_size, n_pages), dtype=torch.int64, device=device
        )

    def reserve(self, batch_idx: torch.Tensor, seq_len: torch.Tensor) -> None:
        """
        Requests the capacity of a given batch to be at least enough to
        hold `seq_len` elements.

        Args:
            batch_idx (Tensor): batch index to be reserved; shape :math:`(1)`.
            seq_len (Tensor): minimum capacity for the given batch; shape :math:`(1)`.
        """

        if seq_len <= self.capacity[batch_idx]:
            return

        num_pages_to_allocate = _cdiv(
            seq_len - self.capacity[batch_idx], self.page_size
        )

        if len(self.empty_pages) < num_pages_to_allocate:
            raise AssertionError(
                f"requested {num_pages_to_allocate.item()} pages "
                f"but there are only {len(self.empty_pages)} empty pages"
            )

        start_page_idx = self.capacity[batch_idx] // self.page_size
        end_page_idx = start_page_idx + num_pages_to_allocate

        # find empty physical pages
        allocated_pages = torch.tensor(
            self.empty_pages[-num_pages_to_allocate:],
            device=num_pages_to_allocate.device,
        )
        self.empty_pages = self.empty_pages[:-num_pages_to_allocate]

        # update page table
        self.page_table[
            batch_idx,
            start_page_idx:end_page_idx,
        ] = allocated_pages

        # update metadata
        self.physical_to_logical[batch_idx, allocated_pages] = torch.arange(
            start_page_idx.item(),
            end_page_idx.item(),
            device=num_pages_to_allocate.device,
        )
        self.capacity[batch_idx] += num_pages_to_allocate * self.page_size

    def erase(self, batch_idx: torch.Tensor) -> None:
        """
        Removes a single batch from paged attention.

        Args:
            batch_idx (Tensor): batch index to be removed; shape :math:`(1)`.
        """

        # find allocated pages
        allocated_page_idx = self.page_table[batch_idx] != -1
        allocated_pages = self.page_table[batch_idx][allocated_page_idx]

        # clean metadata
        self.capacity[batch_idx] = 0
        self.empty_pages += allocated_pages.tolist()
        self.physical_to_logical[batch_idx][:, allocated_pages] = -1
        self.page_table[batch_idx] = -1

    def assign(
        self,
        batch_idx: torch.Tensor,
        input_pos: torch.Tensor,
        k_val: torch.Tensor,
        v_val: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> None:
        """
        Assigns new contents `val` to the storage `cache` at the location
        `batch_idx` and `input_pos`.

        Args:
            batch_idx (Tensor): batch index; shape :math:`(B)`.
            input_pos (Tensor): input positions to be assigned for the given batch; shape :math:`(B, S)`.
            val (Tensor): value to be assigned; shape :math:`(B, H, S, D)`
            cache (Tensor): the cache to store the values; shape:`(1, H, MAX_S, D)`
        """
        if k_val.requires_grad:
            raise RuntimeError("val must not require gradient")

        B, H, S, K_D = k_val.shape
        V_D = v_val.shape[3]
        if B != batch_idx.shape[0]:
            raise RuntimeError(
                f"Expect val and batch_idx have the same batch size "
                f"but got B={B} and B={batch_idx.shape[0]}."
            )
        if H != k_cache.shape[1]:
            raise RuntimeError(
                f"Expect val and cache has the same number of heads "
                f"but got H={H} and H={k_cache.shape[1]}."
            )
        if S != input_pos.shape[1]:
            raise RuntimeError(
                f"Expect val and input_pos has the same length "
                f"but got S={S} and S={input_pos.shape[0]}."
            )
        if K_D != k_cache.shape[3]:
            raise RuntimeError(
                f"Expect k_val and k_cache has the same hidden dim "
                f"but got D={K_D} and D={k_cache.shape[3]}."
            )
        if V_D != v_cache.shape[3]:
            raise RuntimeError(
                f"Expect v_val and v_cache has the same hidden dim "
                f"but got D={V_D} and D={v_cache.shape[3]}."
            )

        # find address
        logical_block_idx = input_pos // self.page_size  # [B, S]
        logical_block_offset = input_pos % self.page_size  # [B, S]
        physical_block_idx = torch.gather(
            self.page_table[batch_idx], 1, logical_block_idx.to(torch.int64)
        ).to(torch.int32)  # [B, S]

        addr = (physical_block_idx * self.page_size + logical_block_offset).view(
            -1
        )  # [B*S]

        k_val = k_val.permute(1, 0, 2, 3).contiguous().view(1, H, B * S, K_D)
        v_val = v_val.permute(1, 0, 2, 3).contiguous().view(1, H, B * S, V_D)

        k_cache[:, :, addr, :] = k_val
        v_cache[:, :, addr, :] = v_val

    def convert_logical_block_mask(
        self,
        block_mask: BlockMask,
        batch_idx: torch.Tensor | None = None,
        kv_len: torch.Tensor | None = None,
    ) -> BlockMask:
        """
        Converts a logical block mask by mapping its logical kv indices to the corresponding
        physical kv indices.

        Args:
            block_mask (BlockMask): logical block mask;
                kv_indices shape :math:`(B, H, ROWS, MAX_BLOCKS_IN_COL)`.
            batch_idx (Tensor): batch index corresponding to the block_mask
                batch dimension. This provides flexibility to convert a
                block mask with smaller batch size than the page table;
                shape :math:`(B)`.
            kv_len (Optional[Tensor]): actual KV sequence length for upper bound check;
                shape :math:`(B,)` to handle multiple batches.
        """
        B, H, ROWS, MAX_BLOCKS_IN_COL = block_mask.kv_indices.shape

        if block_mask.BLOCK_SIZE[1] != self.page_size:
            raise RuntimeError(
                f"Expect block_mask has the same column block size as page_size"
                f"but got size={block_mask.BLOCK_SIZE[1]} and size={self.page_size}"
            )

        # Increase the num columns of converted block mask from logical block mask's
        # num columns to n_pages, since a) the converted block mask
        # may have larger indices values; and b) `_ordered_to_dense` realizes
        # a dense tensor with these converted indices. There would be an IndexError
        # if using the logical block mask's num columns.

        device = block_mask.kv_num_blocks.device

        if batch_idx is None:
            batch_idx = torch.arange(B, device=device)
        page_table = self.page_table[batch_idx]

        new_kv_num_blocks = block_mask.kv_num_blocks.clone()

        new_kv_indices = torch.zeros(
            (B, H, ROWS, self.n_pages), dtype=torch.int32, device=device
        )
        new_kv_indices[:, :, :, :MAX_BLOCKS_IN_COL] = (
            torch.gather(
                page_table, 1, block_mask.kv_indices.view(B, -1).to(torch.int64)
            )
            .view(block_mask.kv_indices.shape)
            .to(torch.int32)
        )

        new_full_kv_indices, new_full_kv_num_blocks = None, None
        if block_mask.full_kv_num_blocks is not None:
            if block_mask.full_kv_indices is None:
                raise AssertionError(
                    "block_mask.full_kv_indices must not be None when full_kv_num_blocks is not None"
                )
            new_full_kv_num_blocks = block_mask.full_kv_num_blocks.clone()
            new_full_kv_indices = torch.zeros(
                (B, H, ROWS, self.n_pages), dtype=torch.int32, device=device
            )
            new_full_kv_indices[:, :, :, :MAX_BLOCKS_IN_COL] = (
                torch.gather(
                    page_table,
                    1,
                    block_mask.full_kv_indices.view(B, -1).to(torch.int64),
                )
                .view(block_mask.full_kv_indices.shape)
                .to(torch.int32)
            )

        new_mask_mod = self.get_mask_mod(block_mask.mask_mod, kv_len)

        seq_lengths = (block_mask.seq_lengths[0], self.n_pages * self.page_size)
        return BlockMask.from_kv_blocks(
            new_kv_num_blocks,
            new_kv_indices,
            new_full_kv_num_blocks,
            new_full_kv_indices,
            block_mask.BLOCK_SIZE,
            new_mask_mod,
            seq_lengths=seq_lengths,
        )

    def get_mask_mod(
        self,
        mask_mod: _mask_mod_signature | None,
        kv_len: torch.Tensor | None = None,
    ) -> _mask_mod_signature:
        """
        Converts a mask_mod based on mapping from the physical block index to the logical
        block index.

        Args:
            mask_mod (_mask_mod_signature): mask_mod based on the logical block index.
            kv_len (Optional[torch.Tensor]): actual KV sequence length for upper bound check.
        """
        if mask_mod is None:
            mask_mod = noop_mask

        def new_mask_mod(
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            physical_kv_idx: torch.Tensor,
        ):
            physical_kv_block = physical_kv_idx // self.page_size
            physical_kv_offset = physical_kv_idx % self.page_size
            logical_block_idx = self.physical_to_logical[b, physical_kv_block]
            logical_kv_idx = logical_block_idx * self.page_size + physical_kv_offset
            live_block = logical_block_idx >= 0
            within_upper_bound = (
                logical_kv_idx < kv_len[b] if kv_len is not None else True
            )
            within_lower_bound = logical_kv_idx >= 0
            is_valid = live_block & within_upper_bound & within_lower_bound

            return torch.where(is_valid, mask_mod(b, h, q_idx, logical_kv_idx), False)

        return new_mask_mod

    def get_score_mod(
        self,
        score_mod: _score_mod_signature | None,
        kv_len: torch.Tensor | None = None,
    ) -> _score_mod_signature:
        """
        Converts a score_mod based on mapping from the physical block index to the logical
        block index.

        Args:
            score_mod (_score_mod_signature): score_mod based on the logical block index.
            `kv_len (Optional[torch.Tensor]): actual KV sequence length for upper bound check.

        """
        if score_mod is None:
            score_mod = _identity

        def new_score_mod(
            score: torch.Tensor,
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            physical_kv_idx: torch.Tensor,
        ):
            physical_kv_block = physical_kv_idx // self.page_size
            physical_kv_offset = physical_kv_idx % self.page_size
            logical_block_idx = self.physical_to_logical[b, physical_kv_block]
            logical_kv_idx = logical_block_idx * self.page_size + physical_kv_offset
            live_block = logical_block_idx >= 0
            within_upper_bound = (
                logical_kv_idx < kv_len[b] if kv_len is not None else True
            )
            within_lower_bound = logical_kv_idx >= 0
            is_valid = live_block & within_upper_bound & within_lower_bound

            return torch.where(
                is_valid,
                score_mod(score, b, h, q_idx, logical_kv_idx),
                float("-inf"),
            )

        return new_score_mod
<|endoftext|># mypy: allow-untyped-defs
"""
This operator implements FP8 scaled dot product attention using Flash Attention 3.
This operator is experimental and subject to change.
"""

import warnings
from enum import IntEnum

import torch
from torch import Tensor


class DescaleType(IntEnum):
    """Describes the scaling granularity for FP8 descale tensors.

    Used with _scaled_dot_product_attention_quantized to explicitly specify
    how the descale factors are applied to the quantized inputs.

    .. warning::
        This enum is experimental and subject to change.
    """

    PER_HEAD = 0
    """Per-head descaling. Descale tensor shape: (batch_size, num_kv_heads)."""


def _validate_descale(
    descale: Tensor | None,
    name: str,
    query: Tensor,
    key: Tensor,
    descale_type: DescaleType,
) -> None:
    """Validate descale tensor for the specified scaling type.

    Args:
        descale: The descale tensor to validate (may be None)
        name: Name of the descale tensor ("q", "k", or "v") for error messages
        query: Query tensor to get batch size
        key: Key tensor to get num_kv_heads
        descale_type: The scaling granularity being used

    Raises:
        ValueError: If the descale tensor has invalid dtype, device, or shape

    Note:
        All descale tensors (q, k, v) use num_kv_heads for the head dimension.
        For GQA/MQA where num_query_heads > num_kv_heads, q_descale is broadcast
        from (B, H_kv) to match the query heads internally.
    """
    if descale is None:
        return

    # Check dtype
    if descale.dtype != torch.float32:
        raise ValueError(f"{name}_descale must have dtype float32, got {descale.dtype}")

    # Check device
    if not descale.is_cuda:
        raise ValueError(f"{name}_descale must be a CUDA tensor")

    # Check shape based on descale type
    if descale_type == DescaleType.PER_HEAD:
        batch_size = query.size(0)
        # All descale tensors use num_kv_heads, even q_descale (broadcast internally)
        # For BHSD layout, num_kv_heads is at dim 1 of key
        num_kv_heads = key.size(1)

        if descale.dim() != 2:
            raise ValueError(
                f"{name}_descale must be a 2D tensor with shape (batch_size, num_kv_heads) "
                f"for PER_HEAD descaling, got {descale.dim()}D tensor"
            )

        if descale.size(0) != batch_size:
            raise ValueError(
                f"{name}_descale batch dimension must match query batch size, "
                f"expected {batch_size}, got {descale.size(0)}"
            )

        if descale.size(1) != num_kv_heads:
            raise ValueError(
                f"{name}_descale head dimension must match num_kv_heads, "
                f"expected {num_kv_heads}, got {descale.size(1)}"
            )


def _scaled_dot_product_attention_quantized(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    is_causal: bool = False,
    scale: float | None = None,
    q_descale: Tensor | None = None,
    k_descale: Tensor | None = None,
    v_descale: Tensor | None = None,
    q_descale_type: DescaleType = DescaleType.PER_HEAD,
    k_descale_type: DescaleType = DescaleType.PER_HEAD,
    v_descale_type: DescaleType = DescaleType.PER_HEAD,
) -> Tensor:
    r"""Scaled dot product attention for FP8 inputs.

    This is a specialized version of scaled_dot_product_attention that supports
    FP8 quantized inputs (float8_e4m3fn) with per-head descaling. Requires the
    Flash Attention 3 backend to be activated.

    .. warning::
        This function is experimental and only supports forward pass.

    Args:
        query (Tensor): Query tensor; shape :math:`(N, H_q, L, E)` dtype float8_e4m3fn
        key (Tensor): Key tensor; shape :math:`(N, H, S, E)` dtype float8_e4m3fn
        value (Tensor): Value tensor; shape :math:`(N, H, S, E_v)` dtype float8_e4m3fn
        is_causal (bool): Apply causal attention mask
        scale (float, optional): Scaling factor for attention weights
        q_descale (Tensor, optional): Query descale tensor; shape :math:`(N, H)` for PER_HEAD
        k_descale (Tensor, optional): Key descale tensor; shape :math:`(N, H)` for PER_HEAD
        v_descale (Tensor, optional): Value descale tensor; shape :math:`(N, H)` for PER_HEAD
        q_descale_type (DescaleType): Specifies the descaling granularity for query. Default: PER_HEAD
        k_descale_type (DescaleType): Specifies the descaling granularity for key. Default: PER_HEAD
        v_descale_type (DescaleType): Specifies the descaling granularity for value. Default: PER_HEAD

    Returns:
        Tensor: Attention output; shape :math:`(N, H_q, L, E_v)` dtype bfloat16
    """
    # Validate descale tensors
    _validate_descale(q_descale, "q", query, key, q_descale_type)
    _validate_descale(k_descale, "k", query, key, k_descale_type)
    _validate_descale(v_descale, "v", query, key, v_descale_type)

    if torch.is_grad_enabled() and (
        query.requires_grad or key.requires_grad or value.requires_grad
    ):
        warnings.warn(
            "_scaled_dot_product_attention_quantized does not support backward pass. "
            "Gradients will not be computed for query, key, or value.",
            UserWarning,
        )
    # Directly call the internal flash attention operator which has descale support
    # NOTE: This should be torch._scaled_dot_product_flash_attention, but it does not work with torch.compile
    result = torch.ops.aten._scaled_dot_product_flash_attention.quantized(
        query,
        key,
        value,
        q_descale,
        k_descale,
        v_descale,
        0.0,
        is_causal,
        False,
        scale=scale,
    )
    return result[0]  # Return the output tensor, mirroring scaled_dot_product_attention
<|endoftext|># mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import functools
import types
import typing
import warnings
from collections.abc import Callable
from typing import cast, TypeAlias, TypeVar
from typing_extensions import deprecated, ParamSpec

import torch
from torch import Tensor
from torch.utils._foreach_utils import (
    _device_has_foreach_support,
    _group_tensors_by_device_and_dtype,
    _has_foreach_support,
)


__all__: list[str] = [
    "clip_grad_norm",
    "clip_grad_norm_",
    "clip_grad_value_",
]


_tensor_or_tensors: TypeAlias = torch.Tensor | typing.Iterable[torch.Tensor]  # noqa: PYI042

_P = ParamSpec("_P")
_R = TypeVar("_R")


def _no_grad(func: Callable[_P, _R]) -> Callable[_P, _R]:
    """
    This wrapper is needed to avoid a circular import when using @torch.no_grad on the exposed functions
    clip_grad_norm_ and clip_grad_value_ themselves.
    """

    def _no_grad_wrapper(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)

    functools.update_wrapper(_no_grad_wrapper, func)
    # pyrefly: ignore [bad-return]
    return _no_grad_wrapper


@_no_grad
def _get_total_norm(
    tensors: _tensor_or_tensors,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
) -> torch.Tensor:
    r"""Compute the norm of an iterable of tensors.

    The norm is computed over the norms of the individual tensors, as if the norms of
    the individual tensors were concatenated into a single vector.

    Args:
        tensors (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will be normalized
        norm_type (float): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of :attr:`tensors` is ``nan``, ``inf``, or ``-inf``.
            Default: ``False``
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``

    Returns:
        Total norm of the tensors (viewed as a single vector).
    """
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    else:
        tensors = list(tensors)
    norm_type = float(norm_type)
    if len(tensors) == 0:
        return torch.tensor(0.0)
    first_device = tensors[0].device
    grouped_tensors: dict[
        tuple[torch.device, torch.dtype], tuple[list[list[Tensor]], list[int]]
    ] = _group_tensors_by_device_and_dtype(  # pyrefly: ignore [bad-assignment]
        [tensors]  # type: ignore[list-item]
    )  # type: ignore[assignment]

    norms: list[Tensor] = []
    for (device, _), ([device_tensors], _) in grouped_tensors.items():
        if (foreach is None and _has_foreach_support(device_tensors, device)) or (
            foreach and _device_has_foreach_support(device)
        ):
            norms.extend(torch._foreach_norm(device_tensors, norm_type))
        elif foreach:
            raise RuntimeError(
                f"foreach=True was passed, but can't use the foreach API on {device.type} tensors"
            )
        else:
            norms.extend(
                [torch.linalg.vector_norm(g, norm_type) for g in device_tensors]
            )

    total_norm = torch.linalg.vector_norm(
        torch.stack([norm.to(first_device) for norm in norms]), norm_type
    )

    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from "
            "`parameters` is non-finite, so it cannot be clipped. To disable "
            "this error and scale the gradients by the non-finite norm anyway, "
            "set `error_if_nonfinite=False`"
        )
    return total_norm


@_no_grad
def _clip_grads_with_norm_(
    parameters: _tensor_or_tensors,
    max_norm: float,
    total_norm: torch.Tensor,
    foreach: bool | None = None,
) -> None:
    r"""Scale the gradients of an iterable of parameters given a pre-calculated total norm and desired max norm.

    The gradients will be scaled by the following calculation

    .. math::
        grad = grad * \min(\frac{max\_norm}{total\_norm + 1e-6}, 1)

    Gradients are modified in-place.

    Note: The scale coefficient is clamped to a maximum of 1.0 to prevent gradient amplification.
    This ensures that gradients are only scaled down when the total norm exceeds max_norm.

    This function is equivalent to :func:`torch.nn.utils.clip_grad_norm_` with a pre-calculated
    total norm.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float): max norm of the gradients
        total_norm (Tensor): total norm of the gradients to use for clipping
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``

    Returns:
        None
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    if len(grads) == 0:
        return
    grouped_grads: dict[
        tuple[torch.device, torch.dtype], tuple[list[list[Tensor]], list[int]]
    ] = _group_tensors_by_device_and_dtype([grads])  # type: ignore[assignment]

    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for (device, _), ([device_grads], _) in grouped_grads.items():
        if (foreach is None and _has_foreach_support(device_grads, device)) or (
            foreach and _device_has_foreach_support(device)
        ):
            torch._foreach_mul_(device_grads, clip_coef_clamped.to(device))
        elif foreach:
            raise RuntimeError(
                f"foreach=True was passed, but can't use the foreach API on {device.type} tensors"
            )
        else:
            clip_coef_clamped_device = clip_coef_clamped.to(device)
            for g in device_grads:
                g.mul_(clip_coef_clamped_device)


@_no_grad
def clip_grad_norm_(
    parameters: _tensor_or_tensors,
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
) -> torch.Tensor:
    r"""Clip the gradient norm of an iterable of parameters.

    The norm is computed over the norms of the individual gradients of all parameters,
    as if the norms of the individual gradients were concatenated into a single vector.
    Gradients are modified in-place.

    This function is equivalent to :func:`torch.nn.utils.get_total_norm` followed by
    :func:`torch.nn.utils.clip_grads_with_norm_` with the ``total_norm`` returned by ``get_total_norm``.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float): max norm of the gradients
        norm_type (float, optional): type of the used p-norm. Can be ``'inf'`` for
            infinity norm. Default: 2.0
        error_if_nonfinite (bool, optional): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False
        foreach (bool, optional): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        is_generator = isinstance(parameters, types.GeneratorType)
        # prevent generators from being exhausted
        parameters = list(parameters)
        if is_generator and len(parameters) == 0:
            warnings.warn(
                "`parameters` is an empty generator, no gradient clipping will occur.",
                stacklevel=3,
            )
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = _get_total_norm(grads, norm_type, error_if_nonfinite, foreach)
    _clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
    return total_norm


@deprecated(
    "`torch.nn.utils.clip_grad_norm` is now deprecated "
    "in favor of `torch.nn.utils.clip_grad_norm_`.",
    category=FutureWarning,
)
def clip_grad_norm(
    parameters: _tensor_or_tensors,
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
) -> torch.Tensor:
    r"""Clip the gradient norm of an iterable of parameters.

    .. warning::
        This method is now deprecated in favor of
        :func:`torch.nn.utils.clip_grad_norm_`.
    """
    return clip_grad_norm_(parameters, max_norm, norm_type, error_if_nonfinite, foreach)


@_no_grad
def clip_grad_value_(
    parameters: _tensor_or_tensors,
    clip_value: float,
    foreach: bool | None = None,
) -> None:
    r"""Clip the gradients of an iterable of parameters at specified value.

    Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        clip_value (float): maximum allowed value of the gradients.
            The gradients are clipped in the range
            :math:`\left[\text{-clip\_value}, \text{clip\_value}\right]`
        foreach (bool, optional): use the faster foreach-based implementation
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and
            silently fall back to the slow implementation for other device types.
            Default: ``None``
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    clip_value = float(clip_value)

    grads = [p.grad for p in parameters if p.grad is not None]
    # pyrefly: ignore [bad-argument-type]
    grouped_grads = _group_tensors_by_device_and_dtype([grads])

    for (device, _), ([grads], _) in grouped_grads.items():
        if (
            foreach is None
            and _has_foreach_support(cast(list[Tensor], grads), device=device)
        ) or (foreach and _device_has_foreach_support(device)):
            torch._foreach_clamp_min_(cast(list[Tensor], grads), -clip_value)
            torch._foreach_clamp_max_(cast(list[Tensor], grads), clip_value)
        elif foreach:
            raise RuntimeError(
                f"foreach=True was passed, but can't use the foreach API on {device.type} tensors"
            )
        else:
            for grad in grads:
                cast(Tensor, grad).clamp_(min=-clip_value, max=clip_value)
<|endoftext|># mypy: allow-untyped-defs
from enum import auto, Enum

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules import Module
from torch.nn.utils import parametrize


__all__ = ["orthogonal", "spectral_norm", "weight_norm"]


def _is_orthogonal(Q, eps=None):
    n, k = Q.size(-2), Q.size(-1)
    Id = torch.eye(k, dtype=Q.dtype, device=Q.device)
    # A reasonable eps, but not too large
    eps = 10.0 * n * torch.finfo(Q.dtype).eps
    return torch.allclose(Q.mH @ Q, Id, atol=eps)


def _make_orthogonal(A):
    """Assume that A is a tall matrix.

    Compute the Q factor s.t. A = QR (A may be complex) and diag(R) is real and non-negative.
    """
    X, tau = torch.geqrf(A)
    Q = torch.linalg.householder_product(X, tau)
    # The diagonal of X is the diagonal of R (which is always real) so we normalise by its signs
    Q *= X.diagonal(dim1=-2, dim2=-1).sgn().unsqueeze(-2)
    return Q


class _OrthMaps(Enum):
    matrix_exp = auto()
    cayley = auto()
    householder = auto()


class _Orthogonal(Module):
    base: Tensor

    def __init__(
        self, weight, orthogonal_map: _OrthMaps, *, use_trivialization=True
    ) -> None:
        super().__init__()

        # Note [Householder complex]
        # For complex tensors, it is not possible to compute the tensor `tau` necessary for
        # linalg.householder_product from the reflectors.
        # To see this, note that the reflectors have a shape like:
        # 0 0 0
        # * 0 0
        # * * 0
        # which, for complex matrices, give n(n-1) (real) parameters. Now, you need n^2 parameters
        # to parametrize the unitary matrices. Saving tau on its own does not work either, because
        # not every combination of `(A, tau)` gives a unitary matrix, meaning that if we optimise
        # them as independent tensors we would not maintain the constraint
        # An equivalent reasoning holds for rectangular matrices
        if weight.is_complex() and orthogonal_map == _OrthMaps.householder:
            raise ValueError(
                "The householder parametrization does not support complex tensors."
            )

        self.shape = weight.shape
        self.orthogonal_map = orthogonal_map
        if use_trivialization:
            self.register_buffer("base", None)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        n, k = X.size(-2), X.size(-1)
        transposed = n < k
        if transposed:
            X = X.mT
            n, k = k, n
        # Here n > k and X is a tall matrix
        if (
            self.orthogonal_map == _OrthMaps.matrix_exp
            or self.orthogonal_map == _OrthMaps.cayley
        ):
            # We just need n x k - k(k-1)/2 parameters
            X = X.tril()
            if n != k:
                # Embed into a square matrix
                X = torch.cat(
                    [X, X.new_zeros(n, n - k).expand(*X.shape[:-2], -1, -1)], dim=-1
                )
            A = X - X.mH
            # A is skew-symmetric (or skew-hermitian)
            if self.orthogonal_map == _OrthMaps.matrix_exp:
                Q = torch.matrix_exp(A)
            elif self.orthogonal_map == _OrthMaps.cayley:
                # Computes the Cayley retraction (I+A/2)(I-A/2)^{-1}
                Id = torch.eye(n, dtype=A.dtype, device=A.device)
                Q = torch.linalg.solve(
                    torch.add(Id, A, alpha=-0.5), torch.add(Id, A, alpha=0.5)
                )
            # Q is now orthogonal (or unitary) of size (..., n, n)
            if n != k:
                # pyrefly: ignore [unbound-name]
                Q = Q[..., :k]
            # Q is now the size of the X (albeit perhaps transposed)
        else:
            # X is real here, as we do not support householder with complex numbers
            A = X.tril(diagonal=-1)
            tau = 2.0 / (1.0 + (A * A).sum(dim=-2))
            Q = torch.linalg.householder_product(A, tau)
            # The diagonal of X is 1's and -1's
            # We do not want to differentiate through this or update the diagonal of X hence the casting
            Q = Q * X.diagonal(dim1=-2, dim2=-1).int().unsqueeze(-2)

        if hasattr(self, "base"):
            # pyrefly: ignore [unbound-name]
            Q = self.base @ Q
        if transposed:
            # pyrefly: ignore [unbound-name]
            Q = Q.mT
        return Q  # type: ignore[possibly-undefined]

    @torch.autograd.no_grad()
    def right_inverse(self, Q: torch.Tensor) -> torch.Tensor:
        if Q.shape != self.shape:
            raise ValueError(
                f"Expected a matrix or batch of matrices of shape {self.shape}. "
                f"Got a tensor of shape {Q.shape}."
            )

        Q_init = Q
        n, k = Q.size(-2), Q.size(-1)
        transpose = n < k
        if transpose:
            Q = Q.mT
            n, k = k, n

        # We always make sure to copy Q in every path
        if not hasattr(self, "base"):
            # Note [right_inverse expm cayley]
            # If we do not have use_trivialization=True, we just implement the inverse of the forward
            # map for the Householder. To see why, think that for the Cayley map,
            # we would need to find the matrix X \in R^{n x k} such that:
            # Y = torch.cat([X.tril(), X.new_zeros(n, n - k).expand(*X.shape[:-2], -1, -1)], dim=-1)
            # A = Y - Y.mH
            # cayley(A)[:, :k]
            # gives the original tensor. It is not clear how to do this.
            # Perhaps via some algebraic manipulation involving the QR like that of
            # Corollary 2.2 in Edelman, Arias and Smith?
            if (
                self.orthogonal_map == _OrthMaps.cayley
                or self.orthogonal_map == _OrthMaps.matrix_exp
            ):
                raise NotImplementedError(
                    "It is not possible to assign to the matrix exponential "
                    "or the Cayley parametrizations when use_trivialization=False."
                )

            # If parametrization == _OrthMaps.householder, make Q orthogonal via the QR decomposition.
            # Here Q is always real because we do not support householder and complex matrices.
            # See note [Householder complex]
            A, tau = torch.geqrf(Q)
            # We want to have a decomposition X = QR with diag(R) > 0, as otherwise we could
            # decompose an orthogonal matrix Q as Q = (-Q)@(-Id), which is a valid QR decomposition
            # The diagonal of Q is the diagonal of R from the qr decomposition
            A.diagonal(dim1=-2, dim2=-1).sign_()
            # Equality with zero is ok because LAPACK returns exactly zero when it does not want
            # to use a particular reflection
            A.diagonal(dim1=-2, dim2=-1)[tau == 0.0] *= -1
            return A.mT if transpose else A
        else:
            if n == k:
                # We check whether Q is orthogonal
                if not _is_orthogonal(Q):
                    Q = _make_orthogonal(Q)
                else:  # Is orthogonal
                    Q = Q.clone()
            else:
                # Complete Q into a full n x n orthogonal matrix
                N = torch.randn(
                    *(Q.size()[:-2] + (n, n - k)), dtype=Q.dtype, device=Q.device
                )
                Q = torch.cat([Q, N], dim=-1)
                Q = _make_orthogonal(Q)
            self.base = Q

            # It is necessary to return the -Id, as we use the diagonal for the
            # Householder parametrization. Using -Id makes:
            # householder(torch.zeros(m,n)) == torch.eye(m,n)
            # Poor man's version of eye_like
            neg_Id = torch.zeros_like(Q_init)
            neg_Id.diagonal(dim1=-2, dim2=-1).fill_(-1.0)
            return neg_Id


def orthogonal(
    module: Module,
    name: str = "weight",
    orthogonal_map: str | None = None,
    *,
    use_trivialization: bool = True,
) -> Module:
    r"""Apply an orthogonal or unitary parametrization to a matrix or a batch of matrices.

    Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`, the parametrized
    matrix :math:`Q \in \mathbb{K}^{m \times n}` is **orthogonal** as

    .. math::

        \begin{align*}
            Q^{\text{H}}Q &= \mathrm{I}_n \mathrlap{\qquad \text{if }m \geq n}\\
            QQ^{\text{H}} &= \mathrm{I}_m \mathrlap{\qquad \text{if }m < n}
        \end{align*}

    where :math:`Q^{\text{H}}` is the conjugate transpose when :math:`Q` is complex
    and the transpose when :math:`Q` is real-valued, and
    :math:`\mathrm{I}_n` is the `n`-dimensional identity matrix.
    In plain words, :math:`Q` will have orthonormal columns whenever :math:`m \geq n`
    and orthonormal rows otherwise.

    If the tensor has more than two dimensions, we consider it as a batch of matrices of shape `(..., m, n)`.

    The matrix :math:`Q` may be parametrized via three different ``orthogonal_map`` in terms of the original tensor:

    - ``"matrix_exp"``/``"cayley"``:
      the :func:`~torch.matrix_exp` :math:`Q = \exp(A)` and the `Cayley map`_
      :math:`Q = (\mathrm{I}_n + A/2)(\mathrm{I}_n - A/2)^{-1}` are applied to a skew-symmetric
      :math:`A` to give an orthogonal matrix.
    - ``"householder"``: computes a product of Householder reflectors
      (:func:`~torch.linalg.householder_product`).

    ``"matrix_exp"``/``"cayley"`` often make the parametrized weight converge faster than
    ``"householder"``, but they are slower to compute for very thin or very wide matrices.

    If ``use_trivialization=True`` (default), the parametrization implements the "Dynamic Trivialization Framework",
    where an extra matrix :math:`B \in \mathbb{K}^{n \times n}` is stored under
    ``module.parametrizations.weight[0].base``. This helps the
    convergence of the parametrized layer at the expense of some extra memory use.
    See `Trivializations for Gradient-Based Optimization on Manifolds`_ .

    Initial value of :math:`Q`:
    If the original tensor is not parametrized and ``use_trivialization=True`` (default), the initial value
    of :math:`Q` is that of the original tensor if it is orthogonal (or unitary in the complex case)
    and it is orthogonalized via the QR decomposition otherwise (see :func:`torch.linalg.qr`).
    Same happens when it is not parametrized and ``orthogonal_map="householder"`` even when ``use_trivialization=False``.
    Otherwise, the initial value is the result of the composition of all the registered
    parametrizations applied to the original tensor.

    .. note::
        This function is implemented using the parametrization functionality
        in :func:`~torch.nn.utils.parametrize.register_parametrization`.


    .. _`Cayley map`: https://en.wikipedia.org/wiki/Cayley_transform#Matrix_map
    .. _`Trivializations for Gradient-Based Optimization on Manifolds`: https://arxiv.org/abs/1909.09501

    Args:
        module (nn.Module): module on which to register the parametrization.
        name (str, optional): name of the tensor to make orthogonal. Default: ``"weight"``.
        orthogonal_map (str, optional): One of the following: ``"matrix_exp"``, ``"cayley"``, ``"householder"``.
            Default: ``"matrix_exp"`` if the matrix is square or complex, ``"householder"`` otherwise.
        use_trivialization (bool, optional): whether to use the dynamic trivialization framework.
            Default: ``True``.

    Returns:
        The original module with an orthogonal parametrization registered to the specified
        weight

    Example::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_LAPACK)
        >>> orth_linear = orthogonal(nn.Linear(20, 40))
        >>> orth_linear
        ParametrizedLinear(
        in_features=20, out_features=40, bias=True
        (parametrizations): ModuleDict(
            (weight): ParametrizationList(
            (0): _Orthogonal()
            )
        )
        )
        >>> # xdoctest: +IGNORE_WANT
        >>> Q = orth_linear.weight
        >>> torch.dist(Q.T @ Q, torch.eye(20))
        tensor(4.9332e-07)
    """
    weight = getattr(module, name, None)
    if not isinstance(weight, Tensor):
        raise ValueError(
            f"Module '{module}' has no parameter or buffer with name '{name}'"
        )

    # We could implement this for 1-dim tensors as the maps on the sphere
    # but I believe it'd bite more people than it'd help
    if weight.ndim < 2:
        raise ValueError(
            "Expected a matrix or batch of matrices. "
            f"Got a tensor of {weight.ndim} dimensions."
        )

    if orthogonal_map is None:
        orthogonal_map = (
            "matrix_exp"
            if weight.size(-2) == weight.size(-1) or weight.is_complex()
            else "householder"
        )

    orth_enum = getattr(_OrthMaps, orthogonal_map, None)
    if orth_enum is None:
        raise ValueError(
            'orthogonal_map has to be one of "matrix_exp", "cayley", "householder". '
            f"Got: {orthogonal_map}"
        )
    orth = _Orthogonal(weight, orth_enum, use_trivialization=use_trivialization)
    parametrize.register_parametrization(module, name, orth, unsafe=True)
    return module


class _WeightNorm(Module):
    def __init__(
        self,
        dim: int | None = 0,
    ) -> None:
        super().__init__()
        if dim is None:
            dim = -1
        self.dim = dim

    def forward(self, weight_g, weight_v):
        return torch._weight_norm(weight_v, weight_g, self.dim)

    def right_inverse(self, weight):
        weight_g = torch.norm_except_dim(weight, 2, self.dim)
        weight_v = weight

        return weight_g, weight_v


def weight_norm(module: Module, name: str = "weight", dim: int = 0):
    r"""Apply weight normalization to a parameter in the given module.

    .. math::
         \mathbf{w} = g \dfrac{\mathbf{v}}{\|\mathbf{v}\|}

    Weight normalization is a reparameterization that decouples the magnitude
    of a weight tensor from its direction. This replaces the parameter specified
    by :attr:`name` with two parameters: one specifying the magnitude
    and one specifying the direction.

    By default, with ``dim=0``, the norm is computed independently per output
    channel/plane. To compute a norm over the entire weight tensor, use
    ``dim=None``.

    See https://arxiv.org/abs/1602.07868

    Args:
        module (Module): containing module
        name (str, optional): name of weight parameter
        dim (int, optional): dimension over which to compute the norm

    Returns:
        The original module with the weight norm hook

    Example::

        >>> m = weight_norm(nn.Linear(20, 40), name='weight')
        >>> m
        ParametrizedLinear(
          in_features=20, out_features=40, bias=True
          (parametrizations): ModuleDict(
            (weight): ParametrizationList(
              (0): _WeightNorm()
            )
          )
        )
        >>> m.parametrizations.weight.original0.size()
        torch.Size([40, 1])
        >>> m.parametrizations.weight.original1.size()
        torch.Size([40, 20])

    """
    _weight_norm = _WeightNorm(dim)
    parametrize.register_parametrization(module, name, _weight_norm, unsafe=True)

    def _weight_norm_compat_hook(
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        g_key = f"{prefix}{name}_g"
        v_key = f"{prefix}{name}_v"
        if g_key in state_dict and v_key in state_dict:
            original0 = state_dict.pop(g_key)
            original1 = state_dict.pop(v_key)
            state_dict[f"{prefix}parametrizations.{name}.original0"] = original0
            state_dict[f"{prefix}parametrizations.{name}.original1"] = original1

    module._register_load_state_dict_pre_hook(_weight_norm_compat_hook)
    return module


class _SpectralNorm(Module):
    def __init__(
        self,
        weight: torch.Tensor,
        n_power_iterations: int = 1,
        dim: int = 0,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        ndim = weight.ndim
        if dim >= ndim or dim < -ndim:
            raise IndexError(
                "Dimension out of range (expected to be in range of "
                f"[-{ndim}, {ndim - 1}] but got {dim})"
            )

        if n_power_iterations <= 0:
            raise ValueError(
                "Expected n_power_iterations to be positive, but "
                f"got n_power_iterations={n_power_iterations}"
            )
        self.dim = dim if dim >= 0 else dim + ndim
        self.eps = eps
        if ndim > 1:
            # For ndim == 1 we do not need to approximate anything (see _SpectralNorm.forward)
            self.n_power_iterations = n_power_iterations
            weight_mat = self._reshape_weight_to_matrix(weight)
            h, w = weight_mat.size()

            u = weight_mat.new_empty(h).normal_(0, 1)
            v = weight_mat.new_empty(w).normal_(0, 1)
            self.register_buffer("_u", F.normalize(u, dim=0, eps=self.eps))
            self.register_buffer("_v", F.normalize(v, dim=0, eps=self.eps))

            # Start with u, v initialized to some reasonable values by performing a number
            # of iterations of the power method
            self._power_method(weight_mat, 15)

    def _reshape_weight_to_matrix(self, weight: torch.Tensor) -> torch.Tensor:
        # Precondition
        if weight.ndim <= 1:
            raise AssertionError(
                f"Expected weight to have more than 1 dimension, got {weight.ndim}"
            )

        if self.dim != 0:
            # permute dim to front
            weight = weight.permute(
                self.dim, *(d for d in range(weight.dim()) if d != self.dim)
            )

        return weight.flatten(1)

    @torch.autograd.no_grad()
    def _power_method(self, weight_mat: torch.Tensor, n_power_iterations: int) -> None:
        # See original note at torch/nn/utils/spectral_norm.py
        # NB: If `do_power_iteration` is set, the `u` and `v` vectors are
        #     updated in power iteration **in-place**. This is very important
        #     because in `DataParallel` forward, the vectors (being buffers) are
        #     broadcast from the parallelized module to each module replica,
        #     which is a new module object created on the fly. And each replica
        #     runs its own spectral norm power iteration. So simply assigning
        #     the updated vectors to the module this function runs on will cause
        #     the update to be lost forever. And the next time the parallelized
        #     module is replicated, the same randomly initialized vectors are
        #     broadcast and used!
        #
        #     Therefore, to make the change propagate back, we rely on two
        #     important behaviors (also enforced via tests):
        #       1. `DataParallel` doesn't clone storage if the broadcast tensor
        #          is already on correct device; and it makes sure that the
        #          parallelized module is already on `device[0]`.
        #       2. If the out tensor in `out=` kwarg has correct shape, it will
        #          just fill in the values.
        #     Therefore, since the same power iteration is performed on all
        #     devices, simply updating the tensors in-place will make sure that
        #     the module replica on `device[0]` will update the _u vector on the
        #     parallelized module (by shared storage).
        #
        #    However, after we update `u` and `v` in-place, we need to **clone**
        #    them before using them to normalize the weight. This is to support
        #    backpropagating through two forward passes, e.g., the common pattern in
        #    GAN training: loss = D(real) - D(fake). Otherwise, engine will
        #    complain that variables needed to do backward for the first forward
        #    (i.e., the `u` and `v` vectors) are changed in the second forward.

        # Precondition
        if weight_mat.ndim <= 1:
            raise AssertionError(
                f"Expected weight_mat to have more than 1 dimension, got {weight_mat.ndim}"
            )

        for _ in range(n_power_iterations):
            # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
            # are the first left and right singular vectors.
            # This power iteration produces approximations of `u` and `v`.
            self._u = F.normalize(
                torch.mv(weight_mat, self._v),  # type: ignore[has-type]
                dim=0,
                eps=self.eps,
                out=self._u,  # type: ignore[has-type]
            )
            self._v = F.normalize(
                torch.mv(weight_mat.H, self._u),  # type: ignore[has-type]
                dim=0,
                eps=self.eps,
                out=self._v,  # type: ignore[has-type]
            )

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if weight.ndim == 1:
            # Faster and more exact path, no need to approximate anything
            return F.normalize(weight, dim=0, eps=self.eps)
        else:
            weight_mat = self._reshape_weight_to_matrix(weight)
            if self.training:
                self._power_method(weight_mat, self.n_power_iterations)
            # See above on why we need to clone
            u = self._u.clone(memory_format=torch.contiguous_format)
            v = self._v.clone(memory_format=torch.contiguous_format)
            # The proper way of computing this should be through F.bilinear, but
            # it seems to have some efficiency issues:
            # https://github.com/pytorch/pytorch/issues/58093
            sigma = torch.vdot(u, torch.mv(weight_mat, v))
            return weight / sigma

    def right_inverse(self, value: torch.Tensor) -> torch.Tensor:
        # we may want to assert here that the passed value already
        # satisfies constraints
        return value


def spectral_norm(
    module: Module,
    name: str = "weight",
    n_power_iterations: int = 1,
    eps: float = 1e-12,
    dim: int | None = None,
) -> Module:
    r"""Apply spectral normalization to a parameter in the given module.

    .. math::
        \mathbf{W}_{SN} = \dfrac{\mathbf{W}}{\sigma(\mathbf{W})},
        \sigma(\mathbf{W}) = \max_{\mathbf{h}: \mathbf{h} \ne 0} \dfrac{\|\mathbf{W} \mathbf{h}\|_2}{\|\mathbf{h}\|_2}

    When applied on a vector, it simplifies to

    .. math::
        \mathbf{x}_{SN} = \dfrac{\mathbf{x}}{\|\mathbf{x}\|_2}

    Spectral normalization stabilizes the training of discriminators (critics)
    in Generative Adversarial Networks (GANs) by reducing the Lipschitz constant
    of the model. :math:`\sigma` is approximated performing one iteration of the
    `power method`_ every time the weight is accessed. If the dimension of the
    weight tensor is greater than 2, it is reshaped to 2D in power iteration
    method to get spectral norm.


    See `Spectral Normalization for Generative Adversarial Networks`_ .

    .. _`power method`: https://en.wikipedia.org/wiki/Power_iteration
    .. _`Spectral Normalization for Generative Adversarial Networks`: https://arxiv.org/abs/1802.05957

    .. note::
        This function is implemented using the parametrization functionality
        in :func:`~torch.nn.utils.parametrize.register_parametrization`. It is a
        reimplementation of :func:`torch.nn.utils.spectral_norm`.

    .. note::
        When this constraint is registered, the singular vectors associated to the largest
        singular value are estimated rather than sampled at random. These are then updated
        performing :attr:`n_power_iterations` of the `power method`_ whenever the tensor
        is accessed with the module on `training` mode.

    .. note::
        If the `_SpectralNorm` module, i.e., `module.parametrization.weight[idx]`,
        is in training mode on removal, it will perform another power iteration.
        If you'd like to avoid this iteration, set the module to eval mode
        before its removal.

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter. Default: ``"weight"``.
        n_power_iterations (int, optional): number of power iterations to
            calculate spectral norm. Default: ``1``.
        eps (float, optional): epsilon for numerical stability in
            calculating norms. Default: ``1e-12``.
        dim (int, optional): dimension corresponding to number of outputs.
            Default: ``0``, except for modules that are instances of
            ConvTranspose{1,2,3}d, when it is ``1``

    Returns:
        The original module with a new parametrization registered to the specified
        weight

    Example::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_LAPACK)
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> snm = spectral_norm(nn.Linear(20, 40))
        >>> snm
        ParametrizedLinear(
          in_features=20, out_features=40, bias=True
          (parametrizations): ModuleDict(
            (weight): ParametrizationList(
              (0): _SpectralNorm()
            )
          )
        )
        >>> torch.linalg.matrix_norm(snm.weight, 2)
        tensor(1.0081, grad_fn=<AmaxBackward0>)
    """
    weight = getattr(module, name, None)
    if not isinstance(weight, Tensor):
        raise ValueError(
            f"Module '{module}' has no parameter or buffer with name '{name}'"
        )

    if dim is None:
        if isinstance(
            module,
            (
                torch.nn.ConvTranspose1d,
                torch.nn.ConvTranspose2d,
                torch.nn.ConvTranspose3d,
            ),
        ):
            dim = 1
        else:
            dim = 0
    parametrize.register_parametrization(
        module, name, _SpectralNorm(weight, n_power_iterations, dim, eps)
    )
    return module
<|endoftext|>from __future__ import annotations

from typing import TypeVar

import torch


_M = TypeVar("_M", bound="torch.nn.Module")


def convert_conv2d_weight_memory_format(
    module: _M, memory_format: torch.memory_format
) -> _M:
    r"""Convert ``memory_format`` of ``nn.Conv2d.weight`` to ``memory_format``.

    The conversion recursively applies to nested ``nn.Module``, including ``module``.
    Note that it only changes the memory_format, but not the semantics of each dimensions.
    This function is used to facilitate the computation to adopt NHWC kernels, which
    provides considerable speed up for fp16 data on CUDA devices with compute capability >= 7.0

    .. note::
        Calling ``model.to(memory_format=torch.channels_last)`` is more aggressive
        than the utility function ``convert_conv2d_weight_memory_format``. Any
        layer with 4d weight will be affected by ``model.to``, which does not
        necessarily benefit from conversion to specified ``memory_format``.
        One place we are confident in is that NHWC(channels_last) conversion for
        convolution in cuDNN, as it is beneficial to run convolution in NHWC,
        even in cases where we have to apply permutation to input tensors.

        Hence our strategy here is to convert only the weight of convolution to
        channels_last. This ensures that;
        1. Fast convolution kernels will be used, the benefit of which could
        outweigh overhead of permutation (if input is not in the same format).
        2. No unnecessary permutations are applied on layers that do not benefit
        from memory_format conversion.

        The optimal case is that, layers between convolution layers are channels
        last compatible. Input tensor would be permuted to channels last when it
        encounters the first convolution layer and stay in that memory format.
        Hence following convolutions will not need to permute its input tensor.

        In case where a channels last incompatible layer is between convolution
        layers, we need to permute the input tensor back to contiguous format
        for that layer. The input tensor will go through the remaining layers in
        contiguous format and be permuted to channels last when it encounters
        another convolution layer. There's no point in propagating that
        permutation to an earlier layer, as most layers are quite agnostic to
        ``memory_format``.

        This claim might change when PyTorch supports fusion of permutation, as
        there might have been a better spot to fuse the permutation other than
        immediately before a convolution.

    Args:
        module (nn.Module): ``nn.Conv2d`` & ``nn.ConvTranspose2d`` or container
                            ``nn.Module``
        memory_format: user specified ``memory_format``,
            e.g. ``torch.channels_last`` or ``torch.contiguous_format``

    Returns:
        The original module with updated ``nn.Conv2d``

    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> # xdoctest: +REQUIRES(env:CUBLAS_WORKSPACE_CONFIG)
        >>> input = torch.randint(
        ...     1, 10, (2, 8, 4, 4), dtype=torch.float16, device="cuda"
        ... )
        >>> model = nn.Sequential(
        >>>     nn.Conv2d(8, 4, 3)).cuda().half()
        >>> # This is identical to:
        >>> # nn.utils.convert_conv2d_weight_memory_format(model, torch.channels_last)
        >>> model = nn.utils.convert_conv2d_weight_memory_format(
        ...     model, torch.channels_last
        ... )
        >>> out = model(input)
    """
    # TODO: expand this to `_ConvNd` when channels_last support is extended
    # beyond only 4d tensors.
    if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
        weight_data = module.weight.detach().clone(memory_format=memory_format)
        module.weight.data = weight_data.resize_(
            weight_data.size(), memory_format=memory_format
        )
    for child in module.children():
        convert_conv2d_weight_memory_format(child, memory_format)

    return module


def convert_conv3d_weight_memory_format(
    module: _M, memory_format: torch.memory_format
) -> _M:
    r"""Convert ``memory_format`` of ``nn.Conv3d.weight`` to ``memory_format``
    The conversion recursively applies to nested ``nn.Module``, including ``module``.
    Note that it only changes the memory_format, but not the semantics of each dimensions.
    This function is used to facilitate the computation to adopt NHWC kernels, which
    provides considerable speed up for fp16 data on CUDA devices with compute capability >= 7.0

    .. note::
        Calling ``model.to(memory_format=torch.channels_last_3d)`` is more aggressive
        than the utility function ``convert_conv3d_weight_memory_format``. Any
        layer with 4d weight will be affected by ``model.to``, which does not
        necessarily benefit from conversion to specified ``memory_format``.
        One place we are confident in is that NDHWC(channels_last_3d) conversion for
        convolution in cuDNN, as it is beneficial to run convolution in NDHWC,
        even in cases where we have to apply permutation to input tensors.

        Hence our strategy here is to convert only the weight of convolution to
        channels_last_3d. This ensures that;
        1. Fast convolution kernels will be used, the benefit of which could
        outweigh overhead of permutation (if input is not in the same format).
        2. No unnecessary permutations are applied on layers that do not benefit
        from memory_format conversion.

        The optimal case is that, layers between convolution layers are channels
        last compatible. Input tensor would be permuted to channels last when it
        encounters the first convolution layer and stay in that memory format.
        Hence following convolutions will not need to permute its input tensor.

        In case where a channels last incompatible layer is between convolution
        layers, we need to permute the input tensor back to contiguous format
        for that layer. The input tensor will go through the remaining layers in
        contiguous format and be permuted to channels last when it encounters
        another convolution layer. There's no point in propagating that
        permutation to an earlier layer, as most layers are quite agnostic to
        ``memory_format``.

        This claim might change when PyTorch supports fusion of permutation, as
        there might have been a better spot to fuse the permutation other than
        immediately before a convolution.

    Args:
        module (nn.Module): ``nn.Conv3d`` & ``nn.ConvTranspose3d`` or container
                            ``nn.Module``
        memory_format: user specified ``memory_format``,
            e.g. ``torch.channels_last`` or ``torch.contiguous_format``

    Returns:
        The original module with updated ``nn.Conv3d``

    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> # xdoctest: +REQUIRES(env:CUBLAS_WORKSPACE_CONFIG)
        >>> input = torch.randint(
        ...     1, 10, (2, 8, 4, 4, 4), dtype=torch.float16, device="cuda"
        ... )
        >>> model = nn.Sequential(
        >>>     nn.Conv3d(8, 4, 3)).cuda().half()
        >>> # This is identical to:
        >>> # nn.utils.convert_conv3d_weight_memory_format(model, torch.channels_last_3d)
        >>> model = nn.utils.convert_conv3d_weight_memory_format(
        ...     model, torch.channels_last_3d
        ... )
        >>> out = model(input)
    """

    # TODO: expand this to `_ConvNd` when channels_last support is extended
    # beyond only 4d tensors.
    if isinstance(module, (torch.nn.Conv3d, torch.nn.ConvTranspose3d)):
        weight_data = module.weight.detach().clone(memory_format=memory_format)
        module.weight.data = weight_data.resize_(
            weight_data.size(), memory_format=memory_format
        )
    for child in module.children():
        convert_conv3d_weight_memory_format(child, memory_format)

    return module


__all__ = [
    "convert_conv2d_weight_memory_format",
    "convert_conv3d_weight_memory_format",
]
<|endoftext|># mypy: allow-untyped-defs
r"""Pruning methods."""

import numbers
from abc import ABC, abstractmethod
from collections.abc import Iterable

import torch


class BasePruningMethod(ABC):
    r"""Abstract base class for creation of new pruning techniques.

    Provides a skeleton for customization requiring the overriding of methods
    such as :meth:`compute_mask` and :meth:`apply`.
    """

    _tensor_name: str

    def __call__(self, module, inputs):
        r"""Multiply the mask into original tensor and store the result.

        Multiplies the mask (stored in ``module[name + '_mask']``)
        into the original tensor (stored in ``module[name + '_orig']``)
        and stores the result into ``module[name]`` by using :meth:`apply_mask`.

        Args:
            module (nn.Module): module containing the tensor to prune
            inputs: not used.
        """
        setattr(module, self._tensor_name, self.apply_mask(module))

    @abstractmethod
    def compute_mask(self, t, default_mask):
        r"""Compute and returns a mask for the input tensor ``t``.

        Starting from a base ``default_mask`` (which should be a mask of ones
        if the tensor has not been pruned yet), generate a random mask to
        apply on top of the ``default_mask`` according to the specific pruning
        method recipe.

        Args:
            t (torch.Tensor): tensor representing the importance scores of the
            parameter to prune.
            default_mask (torch.Tensor): Base mask from previous pruning
            iterations, that need to be respected after the new mask is
            applied. Same dims as ``t``.

        Returns:
            mask (torch.Tensor): mask to apply to ``t``, of same dims as ``t``
        """

    def apply_mask(self, module):
        r"""Simply handles the multiplication between the parameter being pruned and the generated mask.

        Fetches the mask and the original tensor from the module
        and returns the pruned version of the tensor.

        Args:
            module (nn.Module): module containing the tensor to prune

        Returns:
            pruned_tensor (torch.Tensor): pruned version of the input tensor
        """
        # to carry out the multiplication, the mask needs to have been computed,
        # so the pruning method must know what tensor it's operating on
        if self._tensor_name is None:
            raise AssertionError(
                f"Module {module} has to be pruned"
            )  # this gets set in apply()
        mask = getattr(module, self._tensor_name + "_mask")
        orig = getattr(module, self._tensor_name + "_orig")
        pruned_tensor = mask.to(dtype=orig.dtype) * orig
        return pruned_tensor

    @classmethod
    def apply(cls, module, name, *args, importance_scores=None, **kwargs):
        r"""Add pruning on the fly and reparameterization of a tensor.

        Adds the forward pre-hook that enables pruning on the fly and
        the reparameterization of a tensor in terms of the original tensor
        and the pruning mask.

        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
            args: arguments passed on to a subclass of
                :class:`BasePruningMethod`
            importance_scores (torch.Tensor): tensor of importance scores (of
                same shape as module parameter) used to compute mask for pruning.
                The values in this tensor indicate the importance of the
                corresponding elements in the parameter being pruned.
                If unspecified or None, the parameter will be used in its place.
            kwargs: keyword arguments passed on to a subclass of a
                :class:`BasePruningMethod`
        """

        def _get_composite_method(cls, module, name, *args, **kwargs):
            # Check if a pruning method has already been applied to
            # `module[name]`. If so, store that in `old_method`.
            old_method = None
            found = 0
            # there should technically be only 1 hook with hook.name == name
            # assert this using `found`
            hooks_to_remove = []
            for k, hook in module._forward_pre_hooks.items():
                # if it exists, take existing thing, remove hook, then
                # go through normal thing
                if isinstance(hook, BasePruningMethod) and hook._tensor_name == name:
                    old_method = hook
                    hooks_to_remove.append(k)
                    found += 1
            if found > 1:
                raise AssertionError(
                    f"Avoid adding multiple pruning hooks to the "
                    f"same tensor {name} of module {module}. Use a PruningContainer."
                )

            for k in hooks_to_remove:
                del module._forward_pre_hooks[k]

            # Apply the new pruning method, either from scratch or on top of
            # the previous one.
            method = cls(*args, **kwargs)  # new pruning
            # Have the pruning method remember what tensor it's been applied to
            method._tensor_name = name

            # combine `methods` with `old_method`, if `old_method` exists
            if old_method is not None:  # meaning that there was a hook
                # if the hook is already a pruning container, just add the
                # new pruning method to the container
                if isinstance(old_method, PruningContainer):
                    old_method.add_pruning_method(method)
                    method = old_method  # rename old_method --> method

                # if the hook is simply a single pruning method, create a
                # container, add the old pruning method and the new one
                elif isinstance(old_method, BasePruningMethod):
                    container = PruningContainer(old_method)
                    # Have the pruning method remember the name of its tensor
                    # setattr(container, '_tensor_name', name)
                    container.add_pruning_method(method)
                    method = container  # rename container --> method
            return method

        method = _get_composite_method(cls, module, name, *args, **kwargs)
        # at this point we have no forward_pre_hooks but we could have an
        # active reparameterization of the tensor if another pruning method
        # had been applied (in which case `method` would be a PruningContainer
        # and not a simple pruning method).

        # Pruning is to be applied to the module's tensor named `name`,
        # starting from the state it is found in prior to this iteration of
        # pruning. The pruning mask is calculated based on importances scores.

        orig = getattr(module, name)
        if importance_scores is not None:
            if importance_scores.shape != orig.shape:
                raise AssertionError(
                    f"importance_scores should have the same shape as parameter "
                    f"{name} of {module}, got {importance_scores.shape} vs {orig.shape}"
                )
        else:
            importance_scores = orig

        # If this is the first time pruning is applied, take care of moving
        # the original tensor to a new parameter called name + '_orig'
        # and deleting the original parameter
        if not isinstance(method, PruningContainer):
            # copy `module[name]` to `module[name + '_orig']`
            module.register_parameter(name + "_orig", orig)
            # temporarily delete `module[name]`
            del module._parameters[name]
            default_mask = torch.ones_like(orig)  # temp
        # If this is not the first time pruning is applied, all of the above
        # has been done before in a previous pruning iteration, so we're good
        # to go
        else:
            default_mask = (
                getattr(module, name + "_mask")
                .detach()
                .clone(memory_format=torch.contiguous_format)
            )

        # Use try/except because if anything goes wrong with the mask
        # computation etc., you'd want to roll back.
        try:
            # get the final mask, computed according to the specific method
            mask = method.compute_mask(importance_scores, default_mask=default_mask)
            # reparameterize by saving mask to `module[name + '_mask']`...
            module.register_buffer(name + "_mask", mask)
            # ... and the new pruned tensor to `module[name]`
            setattr(module, name, method.apply_mask(module))
            # associate the pruning method to the module via a hook to
            # compute the function before every forward() (compile by run)
            module.register_forward_pre_hook(method)

        except Exception as e:
            if not isinstance(method, PruningContainer):
                orig = getattr(module, name + "_orig")
                module.register_parameter(name, orig)
                del module._parameters[name + "_orig"]
            raise e

        return method

    def prune(self, t, default_mask=None, importance_scores=None):
        r"""Compute and returns a pruned version of input tensor ``t``.

        According to the pruning rule specified in :meth:`compute_mask`.

        Args:
            t (torch.Tensor): tensor to prune (of same dimensions as
                ``default_mask``).
            importance_scores (torch.Tensor): tensor of importance scores (of
                same shape as ``t``) used to compute mask for pruning ``t``.
                The values in this tensor indicate the importance of the
                corresponding elements in the ``t`` that is being pruned.
                If unspecified or None, the tensor ``t`` will be used in its place.
            default_mask (torch.Tensor, optional): mask from previous pruning
                iteration, if any. To be considered when determining what
                portion of the tensor that pruning should act on. If None,
                default to a mask of ones.

        Returns:
            pruned version of tensor ``t``.
        """
        if importance_scores is not None:
            if importance_scores.shape != t.shape:
                raise AssertionError(
                    f"importance_scores should have the same shape as tensor t, "
                    f"got {importance_scores.shape} vs {t.shape}"
                )
        else:
            importance_scores = t
        default_mask = default_mask if default_mask is not None else torch.ones_like(t)
        return t * self.compute_mask(importance_scores, default_mask=default_mask)

    def remove(self, module) -> None:
        r"""Remove the pruning reparameterization from a module.

        The pruned parameter named ``name`` remains permanently pruned,
        and the parameter named ``name+'_orig'`` is removed from the parameter list.
        Similarly, the buffer named ``name+'_mask'`` is removed from the buffers.

        Note:
            Pruning itself is NOT undone or reversed!
        """
        # before removing pruning from a tensor, it has to have been applied
        if self._tensor_name is None:
            raise AssertionError(
                f"Module {module} has to be pruned before pruning can be removed"
            )  # this gets set in apply()

        # to update module[name] to latest trained weights
        weight = self.apply_mask(module)  # masked weights

        # delete and reset
        if hasattr(module, self._tensor_name):
            delattr(module, self._tensor_name)
        orig = module._parameters[self._tensor_name + "_orig"]
        orig.data = weight.data
        del module._parameters[self._tensor_name + "_orig"]
        del module._buffers[self._tensor_name + "_mask"]
        setattr(module, self._tensor_name, orig)


class PruningContainer(BasePruningMethod):
    """Container holding a sequence of pruning methods for iterative pruning.

    Keeps track of the order in which pruning methods are applied and handles
    combining successive pruning calls.

    Accepts as argument an instance of a BasePruningMethod or an iterable of
    them.
    """

    def __init__(self, *args) -> None:
        self._pruning_methods: tuple[BasePruningMethod, ...] = ()
        if not isinstance(args, Iterable):  # only 1 item
            self._tensor_name = args._tensor_name
            self.add_pruning_method(args)

        elif len(args) == 1:  # only 1 item in a tuple
            self._tensor_name = args[0]._tensor_name

            self.add_pruning_method(args[0])
        else:  # manual construction from list or other iterable (or no args)
            for method in args:
                self.add_pruning_method(method)

    def add_pruning_method(self, method) -> None:
        r"""Add a child pruning ``method`` to the container.

        Args:
            method (subclass of BasePruningMethod): child pruning method
                to be added to the container.
        """
        # check that we're adding a pruning method to the container
        if not isinstance(method, BasePruningMethod) and method is not None:
            raise TypeError(f"{type(method)} is not a BasePruningMethod subclass")
        elif method is not None and self._tensor_name != method._tensor_name:
            raise ValueError(
                "Can only add pruning methods acting on "
                f"the parameter named '{self._tensor_name}' to PruningContainer {self}."
                + f" Found '{method._tensor_name}'"
            )
        # if all checks passed, add to _pruning_methods tuple
        self._pruning_methods += (method,)  # type: ignore[operator]

    def __len__(self) -> int:
        return len(self._pruning_methods)

    def __iter__(self):
        return iter(self._pruning_methods)

    def __getitem__(self, idx):
        return self._pruning_methods[idx]

    def compute_mask(self, t, default_mask):
        r"""Apply the latest ``method`` by computing the new partial masks and returning its combination with the ``default_mask``.

        The new partial mask should be computed on the entries or channels
        that were not zeroed out by the ``default_mask``.
        Which portions of the tensor ``t`` the new mask will be calculated from
        depends on the ``PRUNING_TYPE`` (handled by the type handler):

        * for 'unstructured', the mask will be computed from the raveled
          list of nonmasked entries;

        * for 'structured', the mask will be computed from the nonmasked
          channels in the tensor;

        * for 'global', the mask will be computed across all entries.

        Args:
            t (torch.Tensor): tensor representing the parameter to prune
                (of same dimensions as ``default_mask``).
            default_mask (torch.Tensor): mask from previous pruning iteration.

        Returns:
            mask (torch.Tensor): new mask that combines the effects
            of the ``default_mask`` and the new mask from the current
            pruning ``method`` (of same dimensions as ``default_mask`` and
            ``t``).
        """

        def _combine_masks(method, t, mask):
            r"""Combine the masks from all pruning methods and returns a new mask.

            Args:
                method (a BasePruningMethod subclass): pruning method
                    currently being applied.
                t (torch.Tensor): tensor representing the parameter to prune
                    (of same dimensions as mask).
                mask (torch.Tensor): mask from previous pruning iteration

            Returns:
                new_mask (torch.Tensor): new mask that combines the effects
                    of the old mask and the new mask from the current
                    pruning method (of same dimensions as mask and t).
            """
            new_mask = mask  # start off from existing mask
            new_mask = new_mask.to(dtype=t.dtype)

            # compute a slice of t onto which the new pruning method will operate
            if method.PRUNING_TYPE == "unstructured":
                # prune entries of t where the mask is 1
                slc = mask == 1

            # for struct pruning, exclude channels that have already been
            # entirely pruned
            elif method.PRUNING_TYPE == "structured":
                if not hasattr(method, "dim"):
                    raise AttributeError(
                        "Pruning methods of PRUNING_TYPE "
                        '"structured" need to have the attribute `dim` defined.'
                    )

                # find the channels to keep by removing the ones that have been
                # zeroed out already (i.e. where sum(entries) == 0)
                n_dims = t.dim()  # "is this a 2D tensor? 3D? ..."
                dim = method.dim
                # convert negative indexing
                if dim < 0:
                    dim = n_dims + dim
                # if dim is still negative after subtracting it from n_dims
                if dim < 0:
                    raise IndexError(
                        f"Index is out of bounds for tensor with dimensions {n_dims}"
                    )
                # find channels along dim = dim that aren't already tots 0ed out
                keep_channel = mask.sum(dim=[d for d in range(n_dims) if d != dim]) != 0
                # create slice to identify what to prune
                slc = [slice(None)] * n_dims
                slc[dim] = keep_channel

            elif method.PRUNING_TYPE == "global":
                n_dims = len(t.shape)  # "is this a 2D tensor? 3D? ..."
                slc = [slice(None)] * n_dims

            else:
                raise ValueError(f"Unrecognized PRUNING_TYPE {method.PRUNING_TYPE}")

            # compute the new mask on the unpruned slice of the tensor t
            if isinstance(slc, list):
                slc = tuple(slc)
            partial_mask = method.compute_mask(t[slc], default_mask=mask[slc])
            new_mask[slc] = partial_mask.to(dtype=new_mask.dtype)

            return new_mask

        method = self._pruning_methods[-1]
        mask = _combine_masks(method, t, default_mask)
        return mask


class Identity(BasePruningMethod):
    r"""Utility pruning method that does not prune any units but generates the pruning parametrization with a mask of ones."""

    PRUNING_TYPE = "unstructured"

    def compute_mask(self, t, default_mask):
        mask = default_mask
        return mask

    @classmethod
    def apply(cls, module, name):  # type: ignore[override]
        r"""Add pruning on the fly and reparameterization of a tensor.

        Adds the forward pre-hook that enables pruning on the fly and
        the reparameterization of a tensor in terms of the original tensor
        and the pruning mask.

        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
        """
        return super().apply(module, name)


class RandomUnstructured(BasePruningMethod):
    r"""Prune (currently unpruned) units in a tensor at random.

    Args:
        name (str): parameter name within ``module`` on which pruning
            will act.
        amount (int or float): quantity of parameters to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.
    """

    PRUNING_TYPE = "unstructured"

    def __init__(self, amount) -> None:
        # Check range of validity of pruning amount
        _validate_pruning_amount_init(amount)
        self.amount = amount

    def compute_mask(self, t, default_mask):
        # Check that the amount of units to prune is not > than the number of
        # parameters in t
        tensor_size = t.nelement()
        # Compute number of units to prune: amount if int,
        # else amount * tensor_size
        nparams_toprune = _compute_nparams_toprune(self.amount, tensor_size)
        # This should raise an error if the number of units to prune is larger
        # than the number of units in the tensor
        _validate_pruning_amount(nparams_toprune, tensor_size)

        mask = default_mask.clone(memory_format=torch.contiguous_format)

        if nparams_toprune != 0:  # k=0 not supported by torch.kthvalue
            prob = torch.rand_like(t)
            topk = torch.topk(prob.view(-1), k=nparams_toprune)
            mask.view(-1)[topk.indices] = 0

        return mask

    @classmethod
    def apply(cls, module, name, amount):  # type: ignore[override]
        r"""Add pruning on the fly and reparameterization of a tensor.

        Adds the forward pre-hook that enables pruning on the fly and
        the reparameterization of a tensor in terms of the original tensor
        and the pruning mask.

        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
            amount (int or float): quantity of parameters to prune.
                If ``float``, should be between 0.0 and 1.0 and represent the
                fraction of parameters to prune. If ``int``, it represents the
                absolute number of parameters to prune.
        """
        return super().apply(module, name, amount=amount)


class L1Unstructured(BasePruningMethod):
    r"""Prune (currently unpruned) units in a tensor by zeroing out the ones with the lowest L1-norm.

    Args:
        amount (int or float): quantity of parameters to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.
    """

    PRUNING_TYPE = "unstructured"

    def __init__(self, amount) -> None:
        # Check range of validity of pruning amount
        _validate_pruning_amount_init(amount)
        self.amount = amount

    def compute_mask(self, t, default_mask):
        # Check that the amount of units to prune is not > than the number of
        # parameters in t
        tensor_size = t.nelement()
        # Compute number of units to prune: amount if int,
        # else amount * tensor_size
        nparams_toprune = _compute_nparams_toprune(self.amount, tensor_size)
        # This should raise an error if the number of units to prune is larger
        # than the number of units in the tensor
        _validate_pruning_amount(nparams_toprune, tensor_size)

        mask = default_mask.clone(memory_format=torch.contiguous_format)

        if nparams_toprune != 0:  # k=0 not supported by torch.kthvalue
            # largest=True --> top k; largest=False --> bottom k
            # Prune the smallest k
            topk = torch.topk(torch.abs(t).view(-1), k=nparams_toprune, largest=False)
            # topk will have .indices and .values
            mask.view(-1)[topk.indices] = 0

        return mask

    @classmethod
    def apply(cls, module, name, amount, importance_scores=None):  # type: ignore[override]
        r"""Add pruning on the fly and reparameterization of a tensor.

        Adds the forward pre-hook that enables pruning on the fly and
        the reparameterization of a tensor in terms of the original tensor
        and the pruning mask.

        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
            amount (int or float): quantity of parameters to prune.
                If ``float``, should be between 0.0 and 1.0 and represent the
                fraction of parameters to prune. If ``int``, it represents the
                absolute number of parameters to prune.
            importance_scores (torch.Tensor): tensor of importance scores (of same
                shape as module parameter) used to compute mask for pruning.
                The values in this tensor indicate the importance of the corresponding
                elements in the parameter being pruned.
                If unspecified or None, the module parameter will be used in its place.
        """
        return super().apply(
            module, name, amount=amount, importance_scores=importance_scores
        )


class RandomStructured(BasePruningMethod):
    r"""Prune entire (currently unpruned) channels in a tensor at random.

    Args:
        amount (int or float): quantity of parameters to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.
        dim (int, optional): index of the dim along which we define
            channels to prune. Default: -1.
    """

    PRUNING_TYPE = "structured"

    def __init__(self, amount, dim=-1) -> None:
        # Check range of validity of amount
        _validate_pruning_amount_init(amount)
        self.amount = amount
        self.dim = dim

    def compute_mask(self, t, default_mask):
        r"""Compute and returns a mask for the input tensor ``t``.

        Starting from a base ``default_mask`` (which should be a mask of ones
        if the tensor has not been pruned yet), generate a random mask to
        apply on top of the ``default_mask`` by randomly zeroing out channels
        along the specified dim of the tensor.

        Args:
            t (torch.Tensor): tensor representing the parameter to prune
            default_mask (torch.Tensor): Base mask from previous pruning
                iterations, that need to be respected after the new mask is
                applied. Same dims as ``t``.

        Returns:
            mask (torch.Tensor): mask to apply to ``t``, of same dims as ``t``

        Raises:
            IndexError: if ``self.dim >= len(t.shape)``
        """
        # Check that tensor has structure (i.e. more than 1 dimension) such
        # that the concept of "channels" makes sense
        _validate_structured_pruning(t)

        # Check that self.dim is a valid dim to index t, else raise IndexError
        _validate_pruning_dim(t, self.dim)

        # Check that the amount of channels to prune is not > than the number of
        # channels in t along the dim to prune
        tensor_size = t.shape[self.dim]
        # Compute number of units to prune: amount if int,
        # else amount * tensor_size
        nparams_toprune = _compute_nparams_toprune(self.amount, tensor_size)
        # This should raise an error if the number of units to prune is larger
        # than the number of units in the tensor
        _validate_pruning_amount(nparams_toprune, tensor_size)

        # Compute binary mask by initializing it to all 0s and then filling in
        # 1s wherever topk.indices indicates, along self.dim.
        # mask has the same shape as tensor t
        def make_mask(t, dim, nchannels, nchannels_toprune):
            # generate a random number in [0, 1] to associate to each channel
            prob = torch.rand(nchannels)
            # generate mask for each channel by 0ing out the channels that
            # got assigned the k = nchannels_toprune lowest values in prob
            threshold = torch.kthvalue(prob, k=nchannels_toprune).values
            channel_mask = prob > threshold

            mask = torch.zeros_like(t)
            slc = [slice(None)] * len(t.shape)
            slc[dim] = channel_mask
            slc = tuple(slc)
            mask[slc] = 1
            return mask

        if nparams_toprune == 0:  # k=0 not supported by torch.kthvalue
            mask = default_mask
        else:
            # apply the new structured mask on top of prior (potentially
            # unstructured) mask
            mask = make_mask(t, self.dim, tensor_size, nparams_toprune)
            mask *= default_mask.to(dtype=mask.dtype)
        return mask

    @classmethod
    def apply(cls, module, name, amount, dim=-1):  # type: ignore[override]
        r"""Add pruning on the fly and reparameterization of a tensor.

        Adds the forward pre-hook that enables pruning on the fly and
        the reparameterization of a tensor in terms of the original tensor
        and the pruning mask.

        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
            amount (int or float): quantity of parameters to prune.
                If ``float``, should be between 0.0 and 1.0 and represent the
                fraction of parameters to prune. If ``int``, it represents the
                absolute number of parameters to prune.
            dim (int, optional): index of the dim along which we define
                channels to prune. Default: -1.
        """
        return super().apply(module, name, amount=amount, dim=dim)


class LnStructured(BasePruningMethod):
    r"""Prune entire (currently unpruned) channels in a tensor based on their L\ ``n``-norm.

    Args:
        amount (int or float): quantity of channels to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.
        n (int, float, inf, -inf, 'fro', 'nuc'): See documentation of valid
            entries for argument ``p`` in :func:`torch.norm`.
        dim (int, optional): index of the dim along which we define
            channels to prune. Default: -1.
    """

    PRUNING_TYPE = "structured"

    def __init__(self, amount, n, dim=-1) -> None:
        # Check range of validity of amount
        _validate_pruning_amount_init(amount)
        self.amount = amount
        self.n = n
        self.dim = dim

    def compute_mask(self, t, default_mask):
        r"""Compute and returns a mask for the input tensor ``t``.

        Starting from a base ``default_mask`` (which should be a mask of ones
        if the tensor has not been pruned yet), generate a mask to apply on
        top of the ``default_mask`` by zeroing out the channels along the
        specified dim with the lowest L\ ``n``-norm.

        Args:
            t (torch.Tensor): tensor representing the parameter to prune
            default_mask (torch.Tensor): Base mask from previous pruning
                iterations, that need to be respected after the new mask is
                applied.  Same dims as ``t``.

        Returns:
            mask (torch.Tensor): mask to apply to ``t``, of same dims as ``t``

        Raises:
            IndexError: if ``self.dim >= len(t.shape)``
        """
        # Check that tensor has structure (i.e. more than 1 dimension) such
        # that the concept of "channels" makes sense
        _validate_structured_pruning(t)
        # Check that self.dim is a valid dim to index t, else raise IndexError
        _validate_pruning_dim(t, self.dim)

        # Check that the amount of channels to prune is not > than the number of
        # channels in t along the dim to prune
        tensor_size = t.shape[self.dim]
        # Compute number of units to prune: amount if int,
        # else amount * tensor_size
        nparams_toprune = _compute_nparams_toprune(self.amount, tensor_size)
        nparams_tokeep = tensor_size - nparams_toprune
        # This should raise an error if the number of units to prune is larger
        # than the number of units in the tensor
        _validate_pruning_amount(nparams_toprune, tensor_size)

        # Structured pruning prunes entire channels so we need to know the
        # L_n norm along each channel to then find the topk based on this
        # metric
        norm = _compute_norm(t, self.n, self.dim)
        # largest=True --> top k; largest=False --> bottom k
        # Keep the largest k channels along dim=self.dim
        topk = torch.topk(norm, k=nparams_tokeep, largest=True)
        # topk will have .indices and .values

        # Compute binary mask by initializing it to all 0s and then filling in
        # 1s wherever topk.indices indicates, along self.dim.
        # mask has the same shape as tensor t
        def make_mask(t, dim, indices):
            # init mask to 0
            mask = torch.zeros_like(t)
            # e.g.: slc = [None, None, None], if len(t.shape) = 3
            slc = [slice(None)] * len(t.shape)
            # replace a None at position=dim with indices
            # e.g.: slc = [None, None, [0, 2, 3]] if dim=2 & indices=[0,2,3]
            slc[dim] = indices
            slc = tuple(slc)
            # use slc to slice mask and replace all its entries with 1s
            # e.g.: mask[:, :, [0, 2, 3]] = 1
            mask[slc] = 1
            return mask

        if nparams_toprune == 0:  # k=0 not supported by torch.kthvalue
            mask = default_mask
        else:
            mask = make_mask(t, self.dim, topk.indices)
            mask *= default_mask.to(dtype=mask.dtype)

        return mask

    @classmethod
    def apply(cls, module, name, amount, n, dim, importance_scores=None):  # type: ignore[override]
        r"""Add pruning on the fly and reparameterization of a tensor.

        Adds the forward pre-hook that enables pruning on the fly and
        the reparameterization of a tensor in terms of the original tensor
        and the pruning mask.

        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
            amount (int or float): quantity of parameters to prune.
                If ``float``, should be between 0.0 and 1.0 and represent the
                fraction of parameters to prune. If ``int``, it represents the
                absolute number of parameters to prune.
            n (int, float, inf, -inf, 'fro', 'nuc'): See documentation of valid
                entries for argument ``p`` in :func:`torch.norm`.
            dim (int): index of the dim along which we define channels to
                prune.
            importance_scores (torch.Tensor): tensor of importance scores (of same
                shape as module parameter) used to compute mask for pruning.
                The values in this tensor indicate the importance of the corresponding
                elements in the parameter being pruned.
                If unspecified or None, the module parameter will be used in its place.
        """
        return super().apply(
            module,
            name,
            amount=amount,
            n=n,
            dim=dim,
            importance_scores=importance_scores,
        )


class CustomFromMask(BasePruningMethod):
    PRUNING_TYPE = "global"

    def __init__(self, mask) -> None:
        self.mask = mask

    def compute_mask(self, t, default_mask):
        if default_mask.shape != self.mask.shape:
            raise AssertionError(
                f"default_mask shape {default_mask.shape} must match "
                f"self.mask shape {self.mask.shape}"
            )
        mask = default_mask * self.mask.to(dtype=default_mask.dtype)
        return mask

    @classmethod
    def apply(cls, module, name, mask):  # type: ignore[override]
        r"""Add pruning on the fly and reparameterization of a tensor.

        Adds the forward pre-hook that enables pruning on the fly and
        the reparameterization of a tensor in terms of the original tensor
        and the pruning mask.

        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
        """
        return super().apply(module, name, mask=mask)


def identity(module, name):
    r"""Apply pruning reparameterization without pruning any units.

    Applies pruning reparameterization to the tensor corresponding to the
    parameter called ``name`` in ``module`` without actually pruning any
    units. Modifies module in place (and also return the modified module)
    by:

    1) adding a named buffer called ``name+'_mask'`` corresponding to the
       binary mask applied to the parameter ``name`` by the pruning method.
    2) replacing the parameter ``name`` by its pruned version, while the
       original (unpruned) parameter is stored in a new parameter named
       ``name+'_orig'``.

    Note:
        The mask is a tensor of ones.

    Args:
        module (nn.Module): module containing the tensor to prune.
        name (str): parameter name within ``module`` on which pruning
                will act.

    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input module

    Examples:
        >>> # xdoctest: +SKIP
        >>> m = prune.identity(nn.Linear(2, 3), "bias")
        >>> print(m.bias_mask)
        tensor([1., 1., 1.])
    """
    Identity.apply(module, name)
    return module


def random_unstructured(module, name, amount):
    r"""Prune tensor by removing random (currently unpruned) units.

    Prunes tensor corresponding to parameter called ``name`` in ``module``
    by removing the specified ``amount`` of (currently unpruned) units
    selected at random.
    Modifies module in place (and also return the modified module) by:

    1) adding a named buffer called ``name+'_mask'`` corresponding to the
       binary mask applied to the parameter ``name`` by the pruning method.
    2) replacing the parameter ``name`` by its pruned version, while the
       original (unpruned) parameter is stored in a new parameter named
       ``name+'_orig'``.

    Args:
        module (nn.Module): module containing the tensor to prune
        name (str): parameter name within ``module`` on which pruning
                will act.
        amount (int or float): quantity of parameters to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.

    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input module

    Examples:
        >>> # xdoctest: +SKIP
        >>> m = prune.random_unstructured(nn.Linear(2, 3), "weight", amount=1)
        >>> torch.sum(m.weight_mask == 0)
        tensor(1)

    """
    RandomUnstructured.apply(module, name, amount)
    return module


def l1_unstructured(module, name, amount, importance_scores=None):
    r"""Prune tensor by removing units with the lowest L1-norm.

    Prunes tensor corresponding to parameter called ``name`` in ``module``
    by removing the specified `amount` of (currently unpruned) units with the
    lowest L1-norm.
    Modifies module in place (and also return the modified module)
    by:

    1) adding a named buffer called ``name+'_mask'`` corresponding to the
       binary mask applied to the parameter ``name`` by the pruning method.
    2) replacing the parameter ``name`` by its pruned version, while the
       original (unpruned) parameter is stored in a new parameter named
       ``name+'_orig'``.

    Args:
        module (nn.Module): module containing the tensor to prune
        name (str): parameter name within ``module`` on which pruning
                will act.
        amount (int or float): quantity of parameters to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.
        importance_scores (torch.Tensor): tensor of importance scores (of same
            shape as module parameter) used to compute mask for pruning.
            The values in this tensor indicate the importance of the corresponding
            elements in the parameter being pruned.
            If unspecified or None, the module parameter will be used in its place.

    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input module

    Examples:
        >>> # xdoctest: +SKIP
        >>> m = prune.l1_unstructured(nn.Linear(2, 3), "weight", amount=0.2)
        >>> m.state_dict().keys()
        odict_keys(['bias', 'weight_orig', 'weight_mask'])
    """
    L1Unstructured.apply(
        module, name, amount=amount, importance_scores=importance_scores
    )
    return module


def random_structured(module, name, amount, dim):
    r"""Prune tensor by removing random channels along the specified dimension.

    Prunes tensor corresponding to parameter called ``name`` in ``module``
    by removing the specified ``amount`` of (currently unpruned) channels
    along the specified ``dim`` selected at random.
    Modifies module in place (and also return the modified module)
    by:

    1) adding a named buffer called ``name+'_mask'`` corresponding to the
       binary mask applied to the parameter ``name`` by the pruning method.
    2) replacing the parameter ``name`` by its pruned version, while the
       original (unpruned) parameter is stored in a new parameter named
       ``name+'_orig'``.

    Args:
        module (nn.Module): module containing the tensor to prune
        name (str): parameter name within ``module`` on which pruning
                will act.
        amount (int or float): quantity of parameters to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.
        dim (int): index of the dim along which we define channels to prune.

    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input module

    Examples:
        >>> # xdoctest: +SKIP
        >>> m = prune.random_structured(nn.Linear(5, 3), "weight", amount=3, dim=1)
        >>> columns_pruned = int(sum(torch.sum(m.weight, dim=0) == 0))
        >>> print(columns_pruned)
        3
    """
    RandomStructured.apply(module, name, amount, dim)
    return module


def ln_structured(module, name, amount, n, dim, importance_scores=None):
    r"""Prune tensor by removing channels with the lowest L\ ``n``-norm along the specified dimension.

    Prunes tensor corresponding to parameter called ``name`` in ``module``
    by removing the specified ``amount`` of (currently unpruned) channels
    along the specified ``dim`` with the lowest L\ ``n``-norm.
    Modifies module in place (and also return the modified module)
    by:

    1) adding a named buffer called ``name+'_mask'`` corresponding to the
       binary mask applied to the parameter ``name`` by the pruning method.
    2) replacing the parameter ``name`` by its pruned version, while the
       original (unpruned) parameter is stored in a new parameter named
       ``name+'_orig'``.

    Args:
        module (nn.Module): module containing the tensor to prune
        name (str): parameter name within ``module`` on which pruning
                will act.
        amount (int or float): quantity of parameters to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.
        n (int, float, inf, -inf, 'fro', 'nuc'): See documentation of valid
            entries for argument ``p`` in :func:`torch.norm`.
        dim (int): index of the dim along which we define channels to prune.
        importance_scores (torch.Tensor): tensor of importance scores (of same
            shape as module parameter) used to compute mask for pruning.
            The values in this tensor indicate the importance of the corresponding
            elements in the parameter being pruned.
            If unspecified or None, the module parameter will be used in its place.

    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input module

    Examples:
        >>> from torch.nn.utils import prune
        >>> m = prune.ln_structured(
        ...     nn.Conv2d(5, 3, 2), "weight", amount=0.3, dim=1, n=float("-inf")
        ... )
    """
    LnStructured.apply(
        module, name, amount, n, dim, importance_scores=importance_scores
    )
    return module


def global_unstructured(
    parameters, pruning_method, importance_scores=None, **kwargs
) -> None:
    r"""
    Globally prunes tensors corresponding to all parameters in ``parameters`` by applying the specified ``pruning_method``.

    Modifies modules in place by:

    1) adding a named buffer called ``name+'_mask'`` corresponding to the
       binary mask applied to the parameter ``name`` by the pruning method.
    2) replacing the parameter ``name`` by its pruned version, while the
       original (unpruned) parameter is stored in a new parameter named
       ``name+'_orig'``.

    Args:
        parameters (Iterable of (module, name) tuples): parameters of
            the model to prune in a global fashion, i.e. by aggregating all
            weights prior to deciding which ones to prune. module must be of
            type :class:`nn.Module`, and name must be a string.
        pruning_method (function): a valid pruning function from this module,
            or a custom one implemented by the user that satisfies the
            implementation guidelines and has ``PRUNING_TYPE='unstructured'``.
        importance_scores (dict): a dictionary mapping (module, name) tuples to
            the corresponding parameter's importance scores tensor. The tensor
            should be the same shape as the parameter, and is used for computing
            mask for pruning.
            If unspecified or None, the parameter will be used in place of its
            importance scores.
        kwargs: other keyword arguments such as:
            amount (int or float): quantity of parameters to prune across the
            specified parameters.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.

    Raises:
        TypeError: if ``PRUNING_TYPE != 'unstructured'``

    Note:
        Since global structured pruning doesn't make much sense unless the
        norm is normalized by the size of the parameter, we now limit the
        scope of global pruning to unstructured methods.

    Examples:
        >>> from torch.nn.utils import prune
        >>> from collections import OrderedDict
        >>> net = nn.Sequential(
        ...     OrderedDict(
        ...         [
        ...             ("first", nn.Linear(10, 4)),
        ...             ("second", nn.Linear(4, 1)),
        ...         ]
        ...     )
        ... )
        >>> parameters_to_prune = (
        ...     (net.first, "weight"),
        ...     (net.second, "weight"),
        ... )
        >>> prune.global_unstructured(
        ...     parameters_to_prune,
        ...     pruning_method=prune.L1Unstructured,
        ...     amount=10,
        ... )
        >>> print(sum(torch.nn.utils.parameters_to_vector(net.buffers()) == 0))
        tensor(10)

    """
    # ensure parameters is a list or generator of tuples
    if not isinstance(parameters, Iterable):
        raise TypeError("global_unstructured(): parameters is not an Iterable")

    importance_scores = importance_scores if importance_scores is not None else {}
    if not isinstance(importance_scores, dict):
        raise TypeError("global_unstructured(): importance_scores must be of type dict")

    # flatten importance scores to consider them all at once in global pruning
    relevant_importance_scores = torch.nn.utils.parameters_to_vector(
        # pyrefly: ignore [bad-argument-type]
        [
            importance_scores.get((module, name), getattr(module, name))
            for (module, name) in parameters
        ]
    )
    # similarly, flatten the masks (if they exist), or use a flattened vector
    # of 1s of the same dimensions as t
    default_mask = torch.nn.utils.parameters_to_vector(
        [
            getattr(module, name + "_mask", torch.ones_like(getattr(module, name)))
            for (module, name) in parameters
        ]
    )

    # use the canonical pruning methods to compute the new mask, even if the
    # parameter is now a flattened out version of `parameters`
    container = PruningContainer()
    container._tensor_name = "temp"  # to make it match that of `method`
    method = pruning_method(**kwargs)
    method._tensor_name = "temp"  # to make it match that of `container`
    if method.PRUNING_TYPE != "unstructured":
        raise TypeError(
            'Only "unstructured" PRUNING_TYPE supported for '
            f"the `pruning_method`. Found method {pruning_method} of type {method.PRUNING_TYPE}"
        )

    container.add_pruning_method(method)

    # use the `compute_mask` method from `PruningContainer` to combine the
    # mask computed by the new method with the pre-existing mask
    final_mask = container.compute_mask(relevant_importance_scores, default_mask)

    # Pointer for slicing the mask to match the shape of each parameter
    pointer = 0
    for module, name in parameters:
        param = getattr(module, name)
        # The length of the parameter
        num_param = param.numel()
        # Slice the mask, reshape it
        param_mask = final_mask[pointer : pointer + num_param].view_as(param)
        # Assign the correct pre-computed mask to each parameter and add it
        # to the forward_pre_hooks like any other pruning method
        custom_from_mask(module, name, mask=param_mask)

        # Increment the pointer to continue slicing the final_mask
        pointer += num_param


def custom_from_mask(module, name, mask):
    r"""Prune tensor corresponding to parameter called ``name`` in ``module`` by applying the pre-computed mask in ``mask``.

    Modifies module in place (and also return the modified module) by:

    1) adding a named buffer called ``name+'_mask'`` corresponding to the
       binary mask applied to the parameter ``name`` by the pruning method.
    2) replacing the parameter ``name`` by its pruned version, while the
       original (unpruned) parameter is stored in a new parameter named
       ``name+'_orig'``.

    Args:
        module (nn.Module): module containing the tensor to prune
        name (str): parameter name within ``module`` on which pruning
            will act.
        mask (Tensor): binary mask to be applied to the parameter.

    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input module

    Examples:
        >>> from torch.nn.utils import prune
        >>> m = prune.custom_from_mask(
        ...     nn.Linear(5, 3), name="bias", mask=torch.tensor([0, 1, 0])
        ... )
        >>> print(m.bias_mask)
        tensor([0., 1., 0.])

    """
    CustomFromMask.apply(module, name, mask)
    return module


def remove(module, name):
    r"""Remove the pruning reparameterization from a module and the pruning method from the forward hook.

    The pruned parameter named ``name`` remains permanently pruned, and the parameter
    named ``name+'_orig'`` is removed from the parameter list. Similarly,
    the buffer named ``name+'_mask'`` is removed from the buffers.

    Note:
        Pruning itself is NOT undone or reversed!

    Args:
        module (nn.Module): module containing the tensor to prune
        name (str): parameter name within ``module`` on which pruning
            will act.

    Examples:
        >>> m = random_unstructured(nn.Linear(5, 7), name="weight", amount=0.2)
        >>> m = remove(m, name="weight")
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, BasePruningMethod) and hook._tensor_name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError(
        f"Parameter '{name}' of module {module} has to be pruned before pruning can be removed"
    )


def is_pruned(module) -> bool:
    r"""Check if a module is pruned by looking for pruning pre-hooks.

    Check whether ``module`` is pruned by looking for
    ``forward_pre_hooks`` in its modules that inherit from the
    :class:`BasePruningMethod`.

    Args:
        module (nn.Module): object that is either pruned or unpruned

    Returns:
        binary answer to whether ``module`` is pruned.

    Examples:
        >>> from torch.nn.utils import prune
        >>> m = nn.Linear(5, 7)
        >>> print(prune.is_pruned(m))
        False
        >>> prune.random_unstructured(m, name="weight", amount=0.2)
        >>> print(prune.is_pruned(m))
        True
    """
    for _, submodule in module.named_modules():
        for hook in submodule._forward_pre_hooks.values():
            if isinstance(hook, BasePruningMethod):
                return True
    return False


def _validate_pruning_amount_init(amount) -> None:
    r"""Validate helper to check the range of amount at init.

    Args:
        amount (int or float): quantity of parameters to prune.
            If float, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If int, it represents the
            absolute number of parameters to prune.

    Raises:
        ValueError: if amount is a float not in [0, 1], or if it's a negative
            integer.
        TypeError: if amount is neither a float nor an integer.

    Note:
        This does not take into account the number of parameters in the
        tensor to be pruned, which is known only at prune.
    """
    if not isinstance(amount, numbers.Real):
        raise TypeError(f"Invalid type for amount: {amount}. Must be int or float.")

    if (isinstance(amount, numbers.Integral) and amount < 0) or (
        not isinstance(amount, numbers.Integral)  # so it's a float
        and (float(amount) > 1.0 or float(amount) < 0.0)
    ):
        raise ValueError(
            f"amount={amount} should either be a float in the range [0, 1] or a non-negative integer"
        )


def _validate_pruning_amount(amount, tensor_size) -> None:
    r"""Validate that the pruning amount is meaningful wrt to the size of the data.

    Validation helper to check that the amount of parameters to prune
    is meaningful wrt to the size of the data (`tensor_size`).

    Args:
        amount (int or float): quantity of parameters to prune.
            If float, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If int, it represents the
            absolute number of parameters to prune.
        tensor_size (int): absolute number of parameters in the tensor
            to prune.
    """
    # TODO: consider removing this check and allowing users to specify
    # a number of units to prune that is greater than the number of units
    # left to prune. In this case, the tensor will just be fully pruned.

    if isinstance(amount, numbers.Integral) and amount > tensor_size:
        raise ValueError(
            f"amount={amount} should be smaller than the number of parameters to prune={tensor_size}"
        )


def _validate_structured_pruning(t) -> None:
    r"""Validate that the tensor to be pruned is at least 2-Dimensional.

    Validation helper to check that the tensor to be pruned is multi-
    dimensional, such that the concept of "channels" is well-defined.

    Args:
        t (torch.Tensor): tensor representing the parameter to prune

    Raises:
        ValueError: if the tensor `t` is not at least 2D.
    """
    shape = t.shape
    if len(shape) <= 1:
        raise ValueError(
            "Structured pruning can only be applied to "
            "multidimensional tensors. Found tensor of shape "
            f"{shape} with {len(shape)} dims"
        )


def _compute_nparams_toprune(amount, tensor_size):
    r"""Convert the pruning amount from a percentage to absolute value.

    Since amount can be expressed either in absolute value or as a
    percentage of the number of units/channels in a tensor, this utility
    function converts the percentage to absolute value to standardize
    the handling of pruning.

    Args:
        amount (int or float): quantity of parameters to prune.
            If float, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If int, it represents the
            absolute number of parameters to prune.
        tensor_size (int): absolute number of parameters in the tensor
            to prune.

    Returns:
        int: the number of units to prune in the tensor
    """
    # incorrect type already checked in _validate_pruning_amount_init
    if isinstance(amount, numbers.Integral):
        return amount
    else:
        return round(amount * tensor_size)


def _validate_pruning_dim(t, dim) -> None:
    r"""Validate that the pruning dimension is within the bounds of the tensor dimension.

    Args:
        t (torch.Tensor): tensor representing the parameter to prune
        dim (int): index of the dim along which we define channels to prune
    """
    if dim >= t.dim():
        raise IndexError(f"Invalid index {dim} for tensor of size {t.shape}")


def _compute_norm(t, n, dim):
    r"""Compute the L_n-norm of a tensor along all dimensions except for the specified dimension.

    The L_n-norm will be computed across all entries in tensor `t` along all dimension
    except for the one identified by dim.
    Example: if `t` is of shape, say, 3x2x4 and dim=2 (the last dim),
    then norm will have Size [4], and each entry will represent the
    `L_n`-norm computed using the 3x2=6 entries for each of the 4 channels.

    Args:
        t (torch.Tensor): tensor representing the parameter to prune
        n (int, float, inf, -inf, 'fro', 'nuc'): See documentation of valid
            entries for argument p in torch.norm
        dim (int): dim identifying the channels to prune

    Returns:
        norm (torch.Tensor): L_n norm computed across all dimensions except
            for `dim`. By construction, `norm.shape = t.shape[-1]`.
    """
    # dims = all axes, except for the one identified by `dim`
    dims = list(range(t.dim()))
    # convert negative indexing
    if dim < 0:
        dim = dims[dim]
    dims.remove(dim)

    # torch.norm: 'nuc' and 'fro' over a matrix dim are matrix norms;
    # everything else (numeric n, 'fro' elsewhere) is the flat p-norm.
    is_matrix_dim = len(dims) == 2
    if n == "nuc" or (n == "fro" and is_matrix_dim):
        return torch.linalg.matrix_norm(
            t, ord=n, dim=dims if is_matrix_dim else (-2, -1)
        )
    return torch.linalg.vector_norm(t, ord=2 if n == "fro" else n, dim=dims)
<|endoftext|># mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import collections
import copyreg
from collections.abc import Sequence
from contextlib import contextmanager
from copy import deepcopy

import torch
from torch import Tensor
from torch.__future__ import get_swap_module_params_on_conversion
from torch._custom_class_base import CustomClassBase
from torch._library.opaque_object import is_opaque_symbolic_type
from torch.nn.modules.container import Module, ModuleDict, ModuleList
from torch.nn.parameter import Parameter
from torch.utils._python_dispatch import is_traceable_wrapper_subclass


__all__ = [
    "cached",
    "ParametrizationList",
    "register_parametrization",
    "is_parametrized",
    "remove_parametrizations",
    "type_before_parametrizations",
    "transfer_parametrizations_and_params",
]

_cache_enabled = 0
_cache: dict[tuple[int, str], Tensor | None] = {}


@contextmanager
def cached():
    r"""Context manager that enables the caching system within parametrizations registered with :func:`register_parametrization`.

    The value of the parametrized objects is computed and cached the first time
    they are required when this context manager is active. The cached values are
    discarded when leaving the context manager.

    This is useful when using a parametrized parameter more than once in the forward pass.
    An example of this is when parametrizing the recurrent kernel of an RNN or when
    sharing weights.

    The simplest way to activate the cache is by wrapping the forward pass of the neural network

    .. code-block:: python

        import torch.nn.utils.parametrize as P

        ...
        with P.cached():
            output = model(inputs)

    in training and evaluation. One may also wrap the parts of the modules that use
    several times the parametrized tensors. For example, the loop of an RNN with a
    parametrized recurrent kernel:

    .. code-block:: python

        with P.cached():
            for x in xs:
                out_rnn = self.rnn_cell(x, out_rnn)
    """
    global _cache
    global _cache_enabled
    _cache_enabled += 1
    try:
        yield
    finally:
        _cache_enabled -= 1
        if not _cache_enabled:
            _cache = {}


def _register_parameter_or_buffer(module, name, X) -> None:
    if isinstance(X, Parameter):
        module.register_parameter(name, X)
    else:
        module.register_buffer(name, X)


def _maybe_set(dest: Tensor, src: Tensor) -> None:
    should_swap = (
        get_swap_module_params_on_conversion() or is_traceable_wrapper_subclass(dest)
    )
    if should_swap:
        if isinstance(dest, Parameter) and not isinstance(src, Parameter):
            src = Parameter(src, requires_grad=dest.requires_grad)
        torch.utils.swap_tensors(dest, src)
    else:
        dest.set_(src)  # type: ignore[call-overload]


class ParametrizationList(ModuleList):
    r"""A sequential container that holds and manages the original parameters or buffers of a parametrized :class:`torch.nn.Module`.

    It is the type of ``module.parametrizations[tensor_name]`` when ``module[tensor_name]``
    has been parametrized with :func:`register_parametrization`.

    If the first registered parametrization has a ``right_inverse`` that returns one tensor or
    does not have a ``right_inverse`` (in which case we assume that ``right_inverse`` is the identity),
    it will hold the tensor under the name ``original``.
    If it has a ``right_inverse`` that returns more than one tensor, these will be registered as
    ``original0``, ``original1``, ...

    .. warning::
        This class is used internally by :func:`register_parametrization`. It is documented
        here for completeness. It shall not be instantiated by the user.

    Args:
        modules (sequence): sequence of modules representing the parametrizations
        original (Parameter or Tensor): parameter or buffer that is parametrized
        unsafe (bool): a boolean flag that denotes whether the parametrization
            may change the dtype and shape of the tensor. Default: `False`
            Warning: the parametrization is not checked for consistency upon registration.
            Enable this flag at your own risk.
    """

    original: Tensor
    unsafe: bool

    def __init__(
        self,
        modules: Sequence[Module],
        original: Tensor | Parameter,
        unsafe: bool = False,
    ) -> None:
        # We require this because we need to treat differently the first parametrization
        # This should never throw, unless this class is used from the outside
        if len(modules) == 0:
            raise ValueError("ParametrizationList requires one or more modules.")

        super().__init__(modules)
        self.unsafe = unsafe

        # In plain words:
        # module.weight must keep its dtype and shape.
        # Furthermore, if there is no right_inverse or the right_inverse returns a tensor,
        # this should be of the same dtype as the original tensor
        #
        # We check that the following invariants hold:
        #    X = module.weight
        #    Y = param.right_inverse(X)
        #    assert isinstance(Y, Tensor) or
        #           (isinstance(Y, collections.abc.Sequence) and all(isinstance(t, Tensor) for t in Y))
        #    Z = param(Y) if isinstance(Y, Tensor) else param(*Y)
        #    # Consistency checks
        #    assert X.dtype == Z.dtype and X.shape == Z.shape
        #    # If it has one input, this allows to be able to use set_ to be able to
        #    # move data to/from the original tensor without changing its id (which is what the
        #    # optimizer uses to track parameters)
        #    if isinstance(Y, Tensor)
        #      assert X.dtype == Y.dtype
        # Below we use original = X, new = Y

        original_shape = original.shape
        original_dtype = original.dtype

        # Compute new
        with torch.no_grad():
            new = original
            for module in reversed(self):  # type: ignore[call-overload]
                if hasattr(module, "right_inverse"):
                    try:
                        new = module.right_inverse(new)  # type: ignore[operator]
                    except NotImplementedError:
                        pass
                # else, or if it throws, we assume that right_inverse is the identity

        if not isinstance(new, Tensor) and not isinstance(new, Sequence):
            raise ValueError(
                "'right_inverse' must return a Tensor or a Sequence of tensors (list, tuple...). "
                f"Got {type(new).__name__}"
            )

        # Set the number of original tensors
        self.is_tensor = isinstance(new, Tensor)
        self.ntensors = 1 if self.is_tensor else len(new)

        # Register the tensor(s)
        if self.is_tensor:
            # pyrefly: ignore [missing-attribute]
            if original.dtype != new.dtype:
                raise ValueError(
                    "When `right_inverse` outputs one tensor, it may not change the dtype.\n"
                    f"original.dtype: {original.dtype}\n"
                    # pyrefly: ignore [missing-attribute]
                    f"right_inverse(original).dtype: {new.dtype}"
                )

            # pyrefly: ignore [missing-attribute]
            if original.device != new.device:
                raise ValueError(
                    "When `right_inverse` outputs one tensor, it may not change the device.\n"
                    f"original.device: {original.device}\n"
                    # pyrefly: ignore [missing-attribute]
                    f"right_inverse(original).device: {new.device}"
                )

            # Set the original to original so that the user does not need to re-register the parameter
            # manually in the optimiser
            with torch.no_grad():
                # pyrefly: ignore [bad-argument-type]
                _maybe_set(original, new)
            _register_parameter_or_buffer(self, "original", original)
        else:
            for i, originali in enumerate(new):
                match originali:
                    case CustomClassBase():
                        if not is_opaque_symbolic_type(type(originali)):
                            raise ValueError(
                                f"'right_inverse' must return a Tensor or a reference-type "
                                f"opaque. Got element {i} of the sequence with type "
                                f"{type(originali).__name__}."
                            )
                        setattr(self, f"original{i}", originali)
                    case Tensor():
                        # If the original tensor was a Parameter that required grad, we expect the user to
                        # add the new parameters to the optimizer after registering the parametrization
                        # (this is documented)
                        if isinstance(original, Parameter):
                            originali = Parameter(originali, original.requires_grad)
                        originali.requires_grad_(original.requires_grad)
                        _register_parameter_or_buffer(self, f"original{i}", originali)
                    case _:
                        raise ValueError(
                            "'right_inverse' must return a Tensor or a Sequence of tensors "
                            "(list, tuple...). "
                            f"Got element {i} of the sequence with type {type(originali).__name__}."
                        )

        if not self.unsafe:
            # Consistency checks:
            # Since f : A -> B, right_inverse : B -> A, Z and original should live in B
            # Z = forward(right_inverse(original))
            Z = self()
            if not isinstance(Z, Tensor):
                raise ValueError(
                    f"A parametrization must return a tensor. Got {type(Z).__name__}."
                )
            if Z.dtype != original_dtype:
                raise ValueError(
                    "Registering a parametrization may not change the dtype of the tensor, unless `unsafe` flag is enabled.\n"
                    f"unparametrized dtype: {original_dtype}\n"
                    f"parametrized dtype: {Z.dtype}"
                )
            if Z.shape != original_shape:
                raise ValueError(
                    "Registering a parametrization may not change the shape of the tensor, unless `unsafe` flag is enabled.\n"
                    f"unparametrized shape: {original_shape}\n"
                    f"parametrized shape: {Z.shape}"
                )

    def right_inverse(self, value: Tensor) -> None:
        r"""Call the ``right_inverse`` methods of the parametrizations in the inverse registration order.

        Then, it stores the result in ``self.original`` if ``right_inverse`` outputs one tensor
        or in ``self.original0``, ``self.original1``, ... if it outputs several.

        Args:
            value (Tensor): Value to which initialize the module
        """
        # All the exceptions in this function should almost never throw.
        # They could throw if, for example, right_inverse function returns a different
        # dtype when given a different input, which should most likely be caused by a
        # bug in the user's code

        with torch.no_grad():
            # See https://github.com/pytorch/pytorch/issues/53103
            for module in reversed(self):  # type: ignore[call-overload]
                if hasattr(module, "right_inverse"):
                    value = module.right_inverse(value)  # type: ignore[operator]
                else:
                    raise RuntimeError(
                        f"parametrization {type(module).__name__} does not implement "
                        "right_inverse."
                    )
            if self.is_tensor:
                # These exceptions should only throw when a right_inverse function does not
                # return the same dtype for every input, which should most likely be caused by a bug
                if not isinstance(value, Tensor):
                    raise ValueError(
                        f"`right_inverse` should return a tensor. Got {type(value).__name__}"
                    )
                if value.dtype != self.original.dtype:
                    raise ValueError(
                        f"The tensor returned by `right_inverse` has dtype {value.dtype} "
                        f"while `original` has dtype {self.original.dtype}"
                    )
                # We know that the result is going to have the same dtype
                _maybe_set(self.original, value)
            else:
                if not isinstance(value, collections.abc.Sequence):
                    raise ValueError(
                        "'right_inverse' must return a sequence of tensors. "
                        f"Got {type(value).__name__}."
                    )
                if len(value) != self.ntensors:
                    raise ValueError(
                        "'right_inverse' must return a sequence of tensors of length "
                        f"{self.ntensors}. Got a sequence of length {len(value)}."
                    )
                for i, tensor in enumerate(value):
                    original_i = getattr(self, f"original{i}")
                    match tensor:
                        case CustomClassBase():
                            if is_opaque_symbolic_type(type(tensor)):
                                setattr(self, f"original{i}", tensor)
                                continue
                            # Fall-through
                        case Tensor():
                            if original_i.dtype != tensor.dtype:
                                raise ValueError(
                                    f"Tensor {i} returned by `right_inverse` has dtype {tensor.dtype} "
                                    f"while `original{i}` has dtype {original_i.dtype}"
                                )
                            _maybe_set(original_i, tensor)
                            continue
                    raise ValueError(
                        f"'right_inverse' must return a sequence of tensors "
                        f"or reference-type opaques. Got element {i} of type "
                        f"{type(tensor).__name__}."
                    )

    def forward(self) -> Tensor:
        if torch.jit.is_scripting():
            raise RuntimeError("Parametrization is not working with scripting.")
        # Unpack the originals for the first parametrization
        if self.is_tensor:
            x = self[0](self.original)
        else:
            originals = (getattr(self, f"original{i}") for i in range(self.ntensors))
            x = self[0](*originals)
        # It's not possible to call self[1:] here, so we have to be a bit more cryptic
        # Also we want to skip all non-integer keys
        curr_idx = 1
        while hasattr(self, str(curr_idx)):
            x = self[curr_idx](x)
            curr_idx += 1
        return x


def _inject_new_class(module: Module) -> None:
    r"""Set up a module to be parametrized.

    This works by substituting the class of the module by a class
    that extends it to be able to inject a property

    Args:
        module (nn.Module): module into which to inject the property
    """
    cls = module.__class__

    def default_deepcopy(self, memo):
        # Just emulate a standard deepcopy procedure when __deepcopy__ doesn't exist in the current class.
        obj = memo.get(id(self), None)
        if obj is not None:
            return obj
        replica = self.__new__(self.__class__)
        memo[id(self)] = replica
        replica.__dict__ = deepcopy(self.__dict__, memo)
        # Also save all slots if they exist.
        slots_to_save = copyreg._slotnames(self.__class__)  # type: ignore[attr-defined]
        for slot in slots_to_save:
            if hasattr(self, slot):
                setattr(replica, slot, deepcopy(getattr(self, slot), memo))
        return replica

    def getstate(self):
        raise RuntimeError(
            "Serialization of parametrized modules is only "
            "supported through state_dict(). See:\n"
            "https://pytorch.org/tutorials/beginner/saving_loading_models.html"
            "#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training"
        )

    dct = {"__getstate__": getstate}
    # We don't allow serialization of parametrized modules but should still allow deepcopying.
    # Default 'deepcopy' function invokes __deepcopy__ method instead of __getstate__ when it exists.
    if not hasattr(cls, "__deepcopy__"):
        dct["__deepcopy__"] = default_deepcopy  # type: ignore[assignment]

    param_cls = type(
        f"Parametrized{cls.__name__}",
        (cls,),
        dct,
    )

    module.__class__ = param_cls


def _inject_property(module: Module, tensor_name: str) -> None:
    r"""Injects a property into module[tensor_name].

    It assumes that the class in the module has already been modified from its
    original one using _inject_new_class and that the tensor under :attr:`tensor_name`
    has already been moved out

    Args:
        module (nn.Module): module into which to inject the property
        tensor_name (str): name of the name of the property to create
    """
    # We check the precondition.
    # This should never fire if register_parametrization is correctly implemented
    if hasattr(module, tensor_name):
        raise AssertionError(f"Module already has an attribute named '{tensor_name}'")

    @torch.jit.unused
    def get_cached_parametrization(parametrization) -> Tensor:
        global _cache
        key = (id(module), tensor_name)
        tensor = _cache.get(key)
        if tensor is None:
            tensor = parametrization()
            _cache[key] = tensor
        return tensor

    def get_parametrized(self) -> Tensor:
        if torch.jit.is_scripting():
            raise RuntimeError("Parametrization is not working with scripting.")
        parametrization = self.parametrizations[tensor_name]
        # pyrefly: ignore [redundant-condition]
        if _cache_enabled:
            if torch.jit.is_scripting():
                # Scripting
                raise RuntimeError(
                    "Caching is not implemented for scripting. "
                    "Either disable caching or avoid scripting."
                )
            elif torch._C._get_tracing_state() is not None:
                # Tracing
                raise RuntimeError(
                    "Cannot trace a model while caching parametrizations."
                )
            else:
                return get_cached_parametrization(parametrization)
        else:
            # If caching is not active, this function just evaluates the parametrization
            return parametrization()

    def set_original(self, value: Tensor) -> None:
        if torch.jit.is_scripting():
            raise RuntimeError("Parametrization is not working with scripting.")
        self.parametrizations[tensor_name].right_inverse(value)

    setattr(module.__class__, tensor_name, property(get_parametrized, set_original))


def register_parametrization(
    module: Module,
    tensor_name: str,
    parametrization: Module,
    *,
    unsafe: bool = False,
) -> Module:
    r"""Register a parametrization to a tensor in a module.

    Assume that ``tensor_name="weight"`` for simplicity. When accessing ``module.weight``,
    the module will return the parametrized version ``parametrization(module.weight)``.
    If the original tensor requires a gradient, the backward pass will differentiate
    through :attr:`parametrization`, and the optimizer will update the tensor accordingly.

    The first time that a module registers a parametrization, this function will add an attribute
    ``parametrizations`` to the module of type :class:`~ParametrizationList`.

    The list of parametrizations on the tensor ``weight`` will be accessible under
    ``module.parametrizations.weight``.

    The original tensor will be accessible under
    ``module.parametrizations.weight.original``.

    Parametrizations may be concatenated by registering several parametrizations
    on the same attribute.

    The training mode of a registered parametrization is updated on registration
    to match the training mode of the host module

    Parametrized parameters and buffers have an inbuilt caching system that can be activated
    using the context manager :func:`cached`.

    A :attr:`parametrization` may optionally implement a method with signature

    .. code-block:: python

        def right_inverse(self, X: Tensor) -> Union[Tensor, Sequence[Tensor]]

    This method is called on the unparametrized tensor when the first parametrization
    is registered to compute the initial value of the original tensor.
    If this method is not implemented, the original tensor will be just the unparametrized tensor.

    If all the parametrizations registered on a tensor implement `right_inverse` it is possible
    to initialize a parametrized tensor by assigning to it, as shown in the example below.

    It is possible for the first parametrization to depend on several inputs.
    This may be implemented returning a tuple of tensors from ``right_inverse``
    (see the example implementation of a ``RankOne`` parametrization below).

    In this case, the unconstrained tensors are also located under ``module.parametrizations.weight``
    with names ``original0``, ``original1``,...

    .. note::

        If unsafe=False (default) both the forward and right_inverse methods will be called
        once to perform a number of consistency checks.
        If unsafe=True, then right_inverse will be called if the tensor is not parametrized,
        and nothing will be called otherwise.

    .. note::

        In most situations, ``right_inverse`` will be a function such that
        ``forward(right_inverse(X)) == X`` (see
        `right inverse <https://en.wikipedia.org/wiki/Inverse_function#Right_inverses>`_).
        Sometimes, when the parametrization is not surjective, it may be reasonable
        to relax this.

    .. warning::

        If a parametrization depends on several inputs, :func:`~register_parametrization`
        will register a number of new parameters. If such parametrization is registered
        after the optimizer is created, these new parameters will need to be added manually
        to the optimizer. See :meth:`torch.Optimizer.add_param_group`.

    Args:
        module (nn.Module): module on which to register the parametrization
        tensor_name (str): name of the parameter or buffer on which to register
            the parametrization
        parametrization (nn.Module): the parametrization to register
    Keyword args:
        unsafe (bool): a boolean flag that denotes whether the parametrization
            may change the dtype and shape of the tensor. Default: `False`
            Warning: the parametrization is not checked for consistency upon registration.
            Enable this flag at your own risk.

    Raises:
        ValueError: if the module does not have a parameter or a buffer named :attr:`tensor_name`

    Examples:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_LAPACK)
        >>> import torch
        >>> import torch.nn as nn
        >>> import torch.nn.utils.parametrize as P
        >>>
        >>> class Symmetric(nn.Module):
        >>>     def forward(self, X):
        >>>         return X.triu() + X.triu(1).T  # Return a symmetric matrix
        >>>
        >>>     def right_inverse(self, A):
        >>>         return A.triu()
        >>>
        >>> m = nn.Linear(5, 5)
        >>> P.register_parametrization(m, "weight", Symmetric())
        >>> print(torch.allclose(m.weight, m.weight.T))  # m.weight is now symmetric
        True
        >>> A = torch.rand(5, 5)
        >>> A = A + A.T  # A is now symmetric
        >>> m.weight = A  # Initialize the weight to be the symmetric matrix A
        >>> print(torch.allclose(m.weight, A))
        True

        >>> class RankOne(nn.Module):
        >>>     def forward(self, x, y):
        >>> # Form a rank 1 matrix multiplying two vectors
        >>>         return x.unsqueeze(-1) @ y.unsqueeze(-2)
        >>>
        >>>     def right_inverse(self, Z):
        >>> # Project Z onto the rank 1 matrices
        >>>         U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
        >>> # Return rescaled singular vectors
        >>>         s0_sqrt = S[0].sqrt().unsqueeze(-1)
        >>>         return U[..., :, 0] * s0_sqrt, Vh[..., 0, :] * s0_sqrt
        >>>
        >>> linear_rank_one = P.register_parametrization(
        ...     nn.Linear(4, 4), "weight", RankOne()
        ... )
        >>> print(torch.linalg.matrix_rank(linear_rank_one.weight).item())
        1

    """
    parametrization.train(module.training)
    if is_parametrized(module, tensor_name):
        # Correctness checks.
        # If A is the space of tensors with shape and dtype equal to module.weight
        # we check that parametrization.forward and parametrization.right_inverse are
        # functions from A to A
        if not unsafe:
            Y = getattr(module, tensor_name)
            X = parametrization(Y)
            if not isinstance(X, Tensor):
                raise ValueError(
                    f"A parametrization must return a tensor. Got {type(X).__name__}."
                )
            if X.dtype != Y.dtype:
                raise ValueError(
                    "Registering a parametrization may not change the dtype of the tensor, unless the `unsafe` flag is enabled.\n"
                    f"module.{tensor_name}.dtype: {Y.dtype}\n"
                    f"parametrization(module.{tensor_name}).dtype: {X.dtype}"
                )
            if X.shape != Y.shape:
                raise ValueError(
                    "Registering a parametrization may not change the shape of the tensor, unless the `unsafe` flag is enabled.\n"
                    f"module.{tensor_name}.shape: {Y.shape}\n"
                    f"parametrization(module.{tensor_name}).shape: {X.shape}"
                )
            if hasattr(parametrization, "right_inverse"):
                try:
                    Z = parametrization.right_inverse(X)  # type: ignore[operator]
                except NotImplementedError:
                    pass
                else:
                    if not isinstance(Z, Tensor):
                        raise ValueError(
                            f"parametrization.right_inverse must return a tensor. Got: {type(Z).__name__}"
                        )
                    if Z.dtype != Y.dtype:
                        raise ValueError(
                            "The tensor returned by parametrization.right_inverse must have the same dtype "
                            f"as module.{tensor_name}, unless the `unsafe` flag is enabled.\n"
                            f"module.{tensor_name}.dtype: {Y.dtype}\n"
                            f"returned dtype: {Z.dtype}"
                        )
                    if Z.shape != Y.shape:
                        raise ValueError(
                            "The tensor returned by parametrization.right_inverse must have the same shape "
                            f"as module.{tensor_name}, unless the `unsafe` flag is enabled.\n"
                            f"module.{tensor_name}.shape: {Y.shape}\n"
                            f"returned shape: {Z.shape}"
                        )
            # else right_inverse is assumed to be the identity

        # add the new parametrization to the parametrization list
        if not isinstance(module.parametrizations, ModuleDict):
            raise AssertionError(
                f"Expected module.parametrizations to be a ModuleDict, "
                f"got {type(module.parametrizations).__name__}"
            )
        module.parametrizations[tensor_name].append(parametrization)  # type: ignore[operator]
        # If unsafe was True in previous parametrization, keep it enabled
        module.parametrizations[tensor_name].unsafe |= unsafe  # type: ignore[index, union-attr, operator]
    elif tensor_name in module._buffers or tensor_name in module._parameters:
        # Set the parametrization mechanism
        # Fetch the original buffer or parameter
        original = getattr(module, tensor_name)
        # We create this early to check for possible errors
        parametrizations = ParametrizationList(
            [parametrization], original, unsafe=unsafe
        )
        # Delete the previous parameter or buffer
        delattr(module, tensor_name)
        # If this is the first parametrization registered on the module,
        # we prepare the module to inject the property
        if not is_parametrized(module):
            # Change the class
            _inject_new_class(module)
            # Inject a ``ModuleDict`` into the instance under module.parametrizations
            module.parametrizations = ModuleDict()
        # Add a property into the class
        _inject_property(module, tensor_name)
        # Add a ParametrizationList
        if not isinstance(module.parametrizations, ModuleDict):
            raise AssertionError(
                f"Expected module.parametrizations to be a ModuleDict, "
                f"got {type(module.parametrizations).__name__}"
            )
        module.parametrizations[tensor_name] = parametrizations
    else:
        raise ValueError(
            f"Module '{module}' does not have a parameter, a buffer, or a "
            f"parametrized element with name '{tensor_name}'"
        )
    return module


def is_parametrized(module: Module, tensor_name: str | None = None) -> bool:
    r"""Determine if a module has a parametrization.

    Args:
        module (nn.Module): module to query
        tensor_name (str, optional): name of the parameter in the module
            Default: ``None``
    Returns:
        ``True`` if :attr:`module` has a parametrization for the parameter named :attr:`tensor_name`,
        or if it has any parametrization when :attr:`tensor_name` is ``None``;
        otherwise ``False``
    """
    parametrizations = getattr(module, "parametrizations", None)
    if parametrizations is None or not isinstance(parametrizations, ModuleDict):
        return False
    if tensor_name is None:
        # Check that there is at least one parametrized buffer or Parameter
        return len(parametrizations) > 0
    else:
        return tensor_name in parametrizations


def remove_parametrizations(
    module: Module,
    tensor_name: str,
    leave_parametrized: bool = True,
) -> Module:
    r"""Remove the parametrizations on a tensor in a module.

    - If ``leave_parametrized=True``, ``module[tensor_name]`` will be set to
      its current output. In this case, the parametrization shall not change the ``dtype``
      of the tensor.
    - If ``leave_parametrized=False``, ``module[tensor_name]`` will be set to
      the unparametrised tensor in ``module.parametrizations[tensor_name].original``.
      This is only possible when the parametrization depends on just one tensor.

    Args:
        module (nn.Module): module from which remove the parametrization
        tensor_name (str): name of the parametrization to be removed
        leave_parametrized (bool, optional): leave the attribute :attr:`tensor_name` parametrized.
            Default: ``True``

    Returns:
        Module: module

    Raises:
        ValueError: if ``module[tensor_name]`` is not parametrized
        ValueError: if ``leave_parametrized=False`` and the parametrization depends on several tensors
    """
    if not is_parametrized(module, tensor_name):
        raise ValueError(
            f"Module {module} does not have a parametrization on {tensor_name}"
        )

    # Fetch the original tensor
    if not isinstance(module.parametrizations, ModuleDict):
        raise AssertionError(
            f"Expected module.parametrizations to be a ModuleDict, "
            f"got {type(module.parametrizations).__name__}"
        )
    parametrizations = module.parametrizations[tensor_name]

    if parametrizations.is_tensor:
        original = parametrizations.original
        if not isinstance(original, torch.Tensor):
            raise AssertionError(
                f"Expected original to be a Tensor (is_tensor promised us a Tensor), "
                f"got {type(original).__name__}"
            )
        if leave_parametrized:
            with torch.no_grad():
                t = getattr(module, tensor_name)
            # We know they have the same dtype because we have checked this when registering the
            # parametrizations. As such, we can use set_
            # We do this so that the parameter does not to change the id()
            # This way the user does not need to update the optimizer
            with torch.no_grad():
                if type(original) is torch.Tensor:
                    _maybe_set(original, t)
                else:
                    try:
                        _maybe_set(original, t)
                    except RuntimeError as e:
                        # TODO: Fix this for tensor subclasses that are parameters:
                        # RuntimeError: set_storage is not allowed on a Tensor created from .data or .detach().
                        raise RuntimeError(
                            "Calling remove_parametrizations() with leave_parametrized=True "
                            "for a parameter that is an instance of a tensor subclass requires "
                            "set_() to be implemented correctly for the tensor subclass."
                            "Alternatively, one can opt into the swap_tensors path"
                            "Either set leave_parametrized=False or provide a working implementation"
                            "for set_() in the tensor subclass or set "
                            "torch.__future__.set_swap_module_params_on_conversion(True)."
                        ) from e
    else:
        if leave_parametrized:
            # We cannot use no_grad because we need to know whether one or more
            # original tensors required grad
            t = getattr(module, tensor_name)
            # We'll have to trust the user to add it to the optimizer
            original = Parameter(t) if t.requires_grad else t
        else:
            raise ValueError(
                "Cannot leave unparametrized (`leave_parametrized=False`) a tensor "
                "that is parametrized in terms of a sequence of tensors."
            )

    # Delete the property that manages the parametrization
    delattr(module.__class__, tensor_name)
    # Delete the ParametrizationList
    del module.parametrizations[tensor_name]

    # Restore the parameter / buffer into the main class
    _register_parameter_or_buffer(module, tensor_name, original)

    # Roll back the parametrized class if no other buffer or parameter
    # is currently parametrized in this class
    if not is_parametrized(module):
        delattr(module, "parametrizations")
        # Restore class
        orig_cls = module.__class__.__bases__[0]
        module.__class__ = orig_cls
    return module


def type_before_parametrizations(module: Module) -> type:
    r"""Return the module type before parametrizations were applied and if not, then it returns the module type.

    Args:
        module (nn.Module): module to get type of
    """
    if is_parametrized(module):
        return module.__class__.__bases__[0]
    else:
        return type(module)


def transfer_parametrizations_and_params(
    from_module: Module,
    to_module: Module,
    tensor_name: str | None = None,
) -> Module:
    r"""Transfer parametrizations and the parameters they parametrize from :attr:`from_module` to :attr:`to_module`.

    If :attr:`tensor_name` is specified, only transfers the specified parameter, otherwise
    transfers all parametrized parameters. If those parameters do not exist in to_module, it will create them.
    Does nothing if from_module is not parametrized.

    Args:
        from_module (nn.Module): module to transfer from
        to_module (nn.Module): module to transfer to
        tensor_name (str, optional): parameter to transfer

    Returns:
        Module: to_module
    """
    if is_parametrized(from_module):
        if not isinstance(from_module.parametrizations, ModuleDict):
            raise AssertionError(
                f"Expected from_module.parametrizations to be a ModuleDict, "
                f"got {type(from_module.parametrizations).__name__}"
            )

        # get list of all params or the single param to transfer
        parameters_to_transfer: list | ModuleDict = (
            from_module.parametrizations if tensor_name is None else [tensor_name]
        )

        if not hasattr(parameters_to_transfer, "__iter__"):
            raise AssertionError(
                f"Expected parameters_to_transfer to be iterable, "
                f"got {type(parameters_to_transfer).__name__}"
            )
        for parameter_name in parameters_to_transfer:
            # initialize the to-be-transferred param in to_module if it doesn't exist already
            if not hasattr(to_module, parameter_name):
                setattr(
                    to_module,
                    parameter_name,
                    Parameter(getattr(from_module, parameter_name)),
                )

            # apply the params's parametrizations to to_module
            for param_func in from_module.parametrizations[  # type: ignore[attr-defined]
                parameter_name
            ]:
                register_parametrization(to_module, parameter_name, param_func)
            if not isinstance(to_module.parametrizations, ModuleDict):
                raise AssertionError(
                    f"Expected to_module.parametrizations to be a ModuleDict, "
                    f"got {type(to_module.parametrizations).__name__}"
                )

            # make values match, original values can be stored in either original or
            # original0, original1..., need to check both cases
            if hasattr(from_module.parametrizations[parameter_name], "original"):
                to_module.parametrizations[
                    parameter_name
                ].original = from_module.parametrizations[parameter_name].original
            else:
                num = 0
                orig_num = "original" + str(num)
                # loop through each original# until all values have been set
                while hasattr(from_module.parametrizations[parameter_name], orig_num):
                    setattr(
                        to_module.parametrizations[parameter_name],
                        orig_num,
                        getattr(from_module.parametrizations[parameter_name], orig_num),
                    )
                    num = num + 1
                    orig_num = "original" + str(num)

    return to_module
<|endoftext|># mypy: allow-untyped-defs
"""Spectral Normalization from https://arxiv.org/abs/1802.05957."""

from typing import Any, TypeVar

import torch
import torch.nn.functional as F
from torch.nn.modules import Module


__all__ = [
    "SpectralNorm",
    "SpectralNormLoadStateDictPreHook",
    "SpectralNormStateDictHook",
    "spectral_norm",
    "remove_spectral_norm",
]


class SpectralNorm:
    # Invariant before and after each forward call:
    #   u = F.normalize(W @ v)
    # NB: At initialization, this invariant is not enforced

    _version: int = 1
    # At version 1:
    #   made  `W` not a buffer,
    #   added `v` as a buffer, and
    #   made eval mode use `W = u @ W_orig @ v` rather than the stored `W`.
    name: str
    dim: int
    n_power_iterations: int
    eps: float

    def __init__(
        self,
        name: str = "weight",
        n_power_iterations: int = 1,
        dim: int = 0,
        eps: float = 1e-12,
    ) -> None:
        self.name = name
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError(
                "Expected n_power_iterations to be positive, but "
                f"got n_power_iterations={n_power_iterations}"
            )
        self.n_power_iterations = n_power_iterations
        self.eps = eps

    def reshape_weight_to_matrix(self, weight: torch.Tensor) -> torch.Tensor:
        weight_mat = weight
        if self.dim != 0:
            # permute dim to front
            weight_mat = weight_mat.permute(
                self.dim, *[d for d in range(weight_mat.dim()) if d != self.dim]
            )
        height = weight_mat.size(0)
        return weight_mat.reshape(height, -1)

    def compute_weight(self, module: Module, do_power_iteration: bool) -> torch.Tensor:
        # NB: If `do_power_iteration` is set, the `u` and `v` vectors are
        #     updated in power iteration **in-place**. This is very important
        #     because in `DataParallel` forward, the vectors (being buffers) are
        #     broadcast from the parallelized module to each module replica,
        #     which is a new module object created on the fly. And each replica
        #     runs its own spectral norm power iteration. So simply assigning
        #     the updated vectors to the module this function runs on will cause
        #     the update to be lost forever. And the next time the parallelized
        #     module is replicated, the same randomly initialized vectors are
        #     broadcast and used!
        #
        #     Therefore, to make the change propagate back, we rely on two
        #     important behaviors (also enforced via tests):
        #       1. `DataParallel` doesn't clone storage if the broadcast tensor
        #          is already on correct device; and it makes sure that the
        #          parallelized module is already on `device[0]`.
        #       2. If the out tensor in `out=` kwarg has correct shape, it will
        #          just fill in the values.
        #     Therefore, since the same power iteration is performed on all
        #     devices, simply updating the tensors in-place will make sure that
        #     the module replica on `device[0]` will update the _u vector on the
        #     parallelized module (by shared storage).
        #
        #    However, after we update `u` and `v` in-place, we need to **clone**
        #    them before using them to normalize the weight. This is to support
        #    backpropagating through two forward passes, e.g., the common pattern in
        #    GAN training: loss = D(real) - D(fake). Otherwise, engine will
        #    complain that variables needed to do backward for the first forward
        #    (i.e., the `u` and `v` vectors) are changed in the second forward.
        weight = getattr(module, self.name + "_orig")
        u = getattr(module, self.name + "_u")
        v = getattr(module, self.name + "_v")
        weight_mat = self.reshape_weight_to_matrix(weight)

        if do_power_iteration:
            with torch.no_grad():
                for _ in range(self.n_power_iterations):
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    v = F.normalize(
                        torch.mv(weight_mat.t(), u), dim=0, eps=self.eps, out=v
                    )
                    u = F.normalize(torch.mv(weight_mat, v), dim=0, eps=self.eps, out=u)
                if self.n_power_iterations > 0:
                    # See above on why we need to clone
                    u = u.clone(memory_format=torch.contiguous_format)
                    v = v.clone(memory_format=torch.contiguous_format)

        sigma = torch.dot(u, torch.mv(weight_mat, v))
        weight = weight / sigma
        return weight

    def remove(self, module: Module) -> None:
        with torch.no_grad():
            weight = self.compute_weight(module, do_power_iteration=False)
        delattr(module, self.name)
        delattr(module, self.name + "_u")
        delattr(module, self.name + "_v")
        delattr(module, self.name + "_orig")
        module.register_parameter(self.name, torch.nn.Parameter(weight.detach()))

    def __call__(self, module: Module, inputs: Any) -> None:
        setattr(
            module,
            self.name,
            self.compute_weight(module, do_power_iteration=module.training),
        )

    def _solve_v_and_rescale(self, weight_mat, u, target_sigma):
        # Tries to return a vector `v` s.t. `u = F.normalize(W @ v)`
        # (the invariant at top of this class) and `u @ W @ v = sigma`.
        # This uses pinverse in case W^T W is not invertible.
        v = torch.linalg.multi_dot(
            [weight_mat.t().mm(weight_mat).pinverse(), weight_mat.t(), u.unsqueeze(1)]
        ).squeeze(1)
        return v.mul_(target_sigma / torch.dot(u, torch.mv(weight_mat, v)))

    @staticmethod
    def apply(
        module: Module, name: str, n_power_iterations: int, dim: int, eps: float
    ) -> "SpectralNorm":
        for hook in module._forward_pre_hooks.values():
            if isinstance(hook, SpectralNorm) and hook.name == name:
                raise RuntimeError(
                    f"Cannot register two spectral_norm hooks on the same parameter {name}"
                )

        fn = SpectralNorm(name, n_power_iterations, dim, eps)
        weight = module._parameters[name]
        if weight is None:
            raise ValueError(
                f"`SpectralNorm` cannot be applied as parameter `{name}` is None"
            )
        if isinstance(weight, torch.nn.parameter.UninitializedParameter):
            raise ValueError(
                "The module passed to `SpectralNorm` can't have uninitialized parameters. "
                "Make sure to run the dummy forward before applying spectral normalization"
            )

        with torch.no_grad():
            weight_mat = fn.reshape_weight_to_matrix(weight)

            h, w = weight_mat.size()
            # randomly initialize `u` and `v`
            u = F.normalize(weight.new_empty(h).normal_(0, 1), dim=0, eps=fn.eps)
            v = F.normalize(weight.new_empty(w).normal_(0, 1), dim=0, eps=fn.eps)

        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        # We still need to assign weight back as fn.name because all sorts of
        # things may assume that it exists, e.g., when initializing weights.
        # However, we can't directly assign as it could be an nn.Parameter and
        # gets added as a parameter. Instead, we register weight.data as a plain
        # attribute.
        setattr(module, fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)
        module.register_buffer(fn.name + "_v", v)

        module.register_forward_pre_hook(fn)
        module._register_state_dict_hook(SpectralNormStateDictHook(fn))
        module._register_load_state_dict_pre_hook(SpectralNormLoadStateDictPreHook(fn))
        return fn


class SpectralNormLoadStateDictPreHook:
    # See docstring of SpectralNorm._version on the changes to spectral_norm.
    def __init__(self, fn) -> None:
        self.fn = fn

    # For state_dict with version None, (assuming that it has gone through at
    # least one training forward), we have
    #
    #    u = F.normalize(W_orig @ v)
    #    W = W_orig / sigma, where sigma = u @ W_orig @ v
    #
    # To compute `v`, we solve `W_orig @ x = u`, and let
    #    v = x / (u @ W_orig @ x) * (W / W_orig).
    def __call__(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        fn = self.fn
        version = local_metadata.get("spectral_norm", {}).get(
            fn.name + ".version", None
        )
        if version is None or version < 1:
            weight_key = prefix + fn.name
            if (
                version is None
                and all(weight_key + s in state_dict for s in ("_orig", "_u", "_v"))
                and weight_key not in state_dict
            ):
                # Detect if it is the updated state dict and just missing metadata.
                # This could happen if the users are crafting a state dict themselves,
                # so we just pretend that this is the newest.
                return
            has_missing_keys = False
            for suffix in ("_orig", "", "_u"):
                key = weight_key + suffix
                if key not in state_dict:
                    has_missing_keys = True
                    if strict:
                        missing_keys.append(key)
            if has_missing_keys:
                return
            with torch.no_grad():
                weight_orig = state_dict[weight_key + "_orig"]
                weight = state_dict.pop(weight_key)
                sigma = (weight_orig / weight).mean()
                weight_mat = fn.reshape_weight_to_matrix(weight_orig)
                u = state_dict[weight_key + "_u"]
                v = fn._solve_v_and_rescale(weight_mat, u, sigma)
                state_dict[weight_key + "_v"] = v


class SpectralNormStateDictHook:
    # See docstring of SpectralNorm._version on the changes to spectral_norm.
    def __init__(self, fn) -> None:
        self.fn = fn

    def __call__(self, module, state_dict, prefix, local_metadata) -> None:
        if "spectral_norm" not in local_metadata:
            local_metadata["spectral_norm"] = {}
        key = self.fn.name + ".version"
        if key in local_metadata["spectral_norm"]:
            raise RuntimeError(f"Unexpected key in metadata['spectral_norm']: {key}")
        local_metadata["spectral_norm"][key] = self.fn._version


T_module = TypeVar("T_module", bound=Module)


def spectral_norm(
    module: T_module,
    name: str = "weight",
    n_power_iterations: int = 1,
    eps: float = 1e-12,
    dim: int | None = None,
) -> T_module:
    r"""Apply spectral normalization to a parameter in the given module.

    .. math::
        \mathbf{W}_{SN} = \dfrac{\mathbf{W}}{\sigma(\mathbf{W})},
        \sigma(\mathbf{W}) = \max_{\mathbf{h}: \mathbf{h} \ne 0} \dfrac{\|\mathbf{W} \mathbf{h}\|_2}{\|\mathbf{h}\|_2}

    Spectral normalization stabilizes the training of discriminators (critics)
    in Generative Adversarial Networks (GANs) by rescaling the weight tensor
    with spectral norm :math:`\sigma` of the weight matrix calculated using
    power iteration method. If the dimension of the weight tensor is greater
    than 2, it is reshaped to 2D in power iteration method to get spectral
    norm. This is implemented via a hook that calculates spectral norm and
    rescales weight before every :meth:`~Module.forward` call.

    See `Spectral Normalization for Generative Adversarial Networks`_ .

    .. _`Spectral Normalization for Generative Adversarial Networks`: https://arxiv.org/abs/1802.05957

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
        n_power_iterations (int, optional): number of power iterations to
            calculate spectral norm
        eps (float, optional): epsilon for numerical stability in
            calculating norms
        dim (int, optional): dimension corresponding to number of outputs,
            the default is ``0``, except for modules that are instances of
            ConvTranspose{1,2,3}d, when it is ``1``

    Returns:
        The original module with the spectral norm hook

    .. note::
        This function has been reimplemented as
        :func:`torch.nn.utils.parametrizations.spectral_norm` using the new
        parametrization functionality in
        :func:`torch.nn.utils.parametrize.register_parametrization`. Please use
        the newer version. This function will be deprecated in a future version
        of PyTorch.

    Example::

        >>> m = spectral_norm(nn.Linear(20, 40))
        >>> m
        Linear(in_features=20, out_features=40, bias=True)
        >>> m.weight_u.size()
        torch.Size([40])

    """
    if dim is None:
        if isinstance(
            module,
            (
                torch.nn.ConvTranspose1d,
                torch.nn.ConvTranspose2d,
                torch.nn.ConvTranspose3d,
            ),
        ):
            dim = 1
        else:
            dim = 0
    SpectralNorm.apply(module, name, n_power_iterations, dim, eps)

    return module


def remove_spectral_norm(module: T_module, name: str = "weight") -> T_module:
    r"""Remove the spectral normalization reparameterization from a module.

    Args:
        module (Module): containing module
        name (str, optional): name of weight parameter

    Example:
        >>> m = spectral_norm(nn.Linear(40, 10))
        >>> remove_spectral_norm(m)
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, SpectralNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            break
    else:
        raise ValueError(f"spectral_norm of '{name}' not found in {module}")

    for k, hook in module._state_dict_hooks.items():
        if isinstance(hook, SpectralNormStateDictHook) and hook.fn.name == name:
            del module._state_dict_hooks[k]
            break

    for k, hook in module._load_state_dict_pre_hooks.items():
        if isinstance(hook, SpectralNormLoadStateDictPreHook) and hook.fn.name == name:
            del module._load_state_dict_pre_hooks[k]
            break

    return module
<|endoftext|># mypy: allow-untyped-defs
import inspect

import torch


def skip_init(module_cls, *args, **kwargs):
    r"""
    Given a module class object and args / kwargs, instantiate the module without initializing parameters / buffers.

    This can be useful if initialization is slow or if custom initialization will
    be performed, making the default initialization unnecessary. There are some caveats to this, due to
    the way this function is implemented:

    1. The module must accept a `device` arg in its constructor that is passed to any parameters
    or buffers created during construction.

    2. The module must not perform any computation on parameters in its constructor except
    initialization (i.e. functions from :mod:`torch.nn.init`).

    If these conditions are satisfied, the module can be instantiated with parameter / buffer values
    uninitialized, as if having been created using :func:`torch.empty`.

    Args:
        module_cls: Class object; should be a subclass of :class:`torch.nn.Module`
        args: args to pass to the module's constructor
        kwargs: kwargs to pass to the module's constructor

    Returns:
        Instantiated module with uninitialized parameters / buffers

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> import torch
        >>> m = torch.nn.utils.skip_init(torch.nn.Linear, 5, 1)
        >>> m.weight
        Parameter containing:
        tensor([[0.0000e+00, 1.5846e+29, 7.8307e+00, 2.5250e-29, 1.1210e-44]],
               requires_grad=True)
        >>> m2 = torch.nn.utils.skip_init(torch.nn.Linear, in_features=6, out_features=1)
        >>> m2.weight
        Parameter containing:
        tensor([[-1.4677e+24,  4.5915e-41,  1.4013e-45,  0.0000e+00, -1.4677e+24,
                  4.5915e-41]], requires_grad=True)

    """
    if not issubclass(module_cls, torch.nn.Module):
        raise RuntimeError(f"Expected a Module; got {module_cls}")
    if "device" not in inspect.signature(module_cls).parameters:
        raise RuntimeError("Module must support a 'device' arg to skip initialization")

    final_device = kwargs.pop("device", "cpu")
    kwargs["device"] = "meta"
    return module_cls(*args, **kwargs).to_empty(device=final_device)
<|endoftext|>from collections.abc import Iterable

import torch


def parameters_to_vector(parameters: Iterable[torch.Tensor]) -> torch.Tensor:
    r"""Flatten an iterable of parameters into a single vector.

    Args:
        parameters (Iterable[Tensor]): an iterable of Tensors that are the
            parameters of a model.

    Returns:
        The parameters represented by a single vector
    """
    # Flag for the device where the parameter is located
    param_device = None

    vec = []
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        vec.append(param.view(-1))
    return torch.cat(vec)


def vector_to_parameters(vec: torch.Tensor, parameters: Iterable[torch.Tensor]) -> None:
    r"""Copy slices of a vector into an iterable of parameters.

    Args:
        vec (Tensor): a single vector representing the parameters of a model.
        parameters (Iterable[Tensor]): an iterable of Tensors that are the
            parameters of a model.
    """
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError(f"expected torch.Tensor, but got: {torch.typename(vec)}")
    # Flag for the device where the parameter is located
    param_device = None

    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        param.data = vec[pointer : pointer + num_param].view_as(param).data

        # Increment the pointer
        pointer += num_param


def _check_param_device(param: torch.Tensor, old_param_device: int | None) -> int:
    r"""Check if the parameters are located on the same device.

    Currently, the conversion between model parameters and single vector form is not supported
    for multiple allocations, e.g. parameters in different GPUs/PrivateUse1s, or mixture of CPU/GPU/PrivateUse1.

    Args:
        param ([Tensor]): a Tensor of a parameter of a model
        old_param_device (int): the device where the first parameter of a
                                model is allocated.

    Returns:
        old_param_device (int): report device for the first time
    """
    # Meet the first parameter
    support_device_types = ["cuda", torch._C._get_privateuse1_backend_name()]
    if old_param_device is None:
        old_param_device = (
            param.get_device() if param.device.type in support_device_types else -1
        )
    else:
        warn = False
        if (
            param.device.type in support_device_types
        ):  # Check if in same GPU/PrivateUse1
            warn = param.get_device() != old_param_device
        else:  # Check if in CPU
            warn = old_param_device != -1
        if warn:
            raise TypeError(
                "Found two parameters on different devices, "
                "this is currently not supported."
            )
    return old_param_device
<|endoftext|>from . import parametrizations, parametrize, rnn, stateless
from .clip_grad import (
    _clip_grads_with_norm_ as clip_grads_with_norm_,
    _get_total_norm as get_total_norm,
    clip_grad_norm,
    clip_grad_norm_,
    clip_grad_value_,
)
from .convert_parameters import parameters_to_vector, vector_to_parameters
from .fusion import (
    fuse_conv_bn_eval,
    fuse_conv_bn_weights,
    fuse_linear_bn_eval,
    fuse_linear_bn_weights,
)
from .init import skip_init
from .memory_format import (
    convert_conv2d_weight_memory_format,
    convert_conv3d_weight_memory_format,
)
from .spectral_norm import remove_spectral_norm, spectral_norm
from .weight_norm import remove_weight_norm, weight_norm


__all__ = [
    "clip_grad_norm",
    "clip_grad_norm_",
    "clip_grads_with_norm_",
    "clip_grad_value_",
    "convert_conv2d_weight_memory_format",
    "convert_conv3d_weight_memory_format",
    "fuse_conv_bn_eval",
    "fuse_conv_bn_weights",
    "fuse_linear_bn_eval",
    "fuse_linear_bn_weights",
    "get_total_norm",
    "parameters_to_vector",
    "parametrizations",
    "parametrize",
    "remove_spectral_norm",
    "remove_weight_norm",
    "rnn",
    "skip_init",
    "spectral_norm",
    "stateless",
    "vector_to_parameters",
    "weight_norm",
]
<|endoftext|>import importlib
import warnings
from collections.abc import Callable


_MESSAGE_TEMPLATE = (
    r"Usage of '{old_location}' is deprecated; please use '{new_location}' instead."
)


def lazy_deprecated_import(
    all: list[str],
    old_module: str,
    new_module: str,
) -> Callable:
    r"""Import utility to lazily import deprecated packages / modules / functional.

    The old_module and new_module are also used in the deprecation warning defined
    by the `_MESSAGE_TEMPLATE`.

    Args:
        all: The list of the functions that are imported. Generally, the module's
            __all__ list of the module.
        old_module: Old module location
        new_module: New module location / Migrated location

    Returns:
        Callable to assign to the `__getattr__`

    Usage:

        # In the `torch/nn/quantized/functional.py`
        from torch.nn.utils._deprecation_utils import lazy_deprecated_import
        _MIGRATED_TO = "torch.ao.nn.quantized.functional"
        __getattr__ = lazy_deprecated_import(
            all=__all__,
            old_module=__name__,
            new_module=_MIGRATED_TO)
    """
    warning_message = _MESSAGE_TEMPLATE.format(
        old_location=old_module, new_location=new_module
    )

    def getattr_dunder(name: str) -> None:
        if name in all:
            # We are using the "RuntimeWarning" to make sure it is not
            # ignored by default.
            warnings.warn(warning_message, RuntimeWarning, stacklevel=2)
            package = importlib.import_module(new_module)
            return getattr(package, name)
        raise AttributeError(f"Module {new_module!r} has no attribute {name!r}.")

    return getattr_dunder
<|endoftext|># mypy: allow-untyped-defs
r"""Weight Normalization from https://arxiv.org/abs/1602.07868."""

from typing import Any, TypeVar
from typing_extensions import deprecated

from torch import _weight_norm, norm_except_dim
from torch.nn.modules import Module
from torch.nn.parameter import Parameter, UninitializedParameter


__all__ = ["WeightNorm", "weight_norm", "remove_weight_norm"]


class WeightNorm:
    name: str
    dim: int

    def __init__(self, name: str, dim: int) -> None:
        if dim is None:
            dim = -1
        self.name = name
        self.dim = dim

    # TODO Make return type more specific
    def compute_weight(self, module: Module) -> Any:
        g = getattr(module, self.name + "_g")
        v = getattr(module, self.name + "_v")
        return _weight_norm(v, g, self.dim)

    @staticmethod
    @deprecated(
        "`torch.nn.utils.weight_norm` is deprecated "
        "in favor of `torch.nn.utils.parametrizations.weight_norm`.",
        category=FutureWarning,
    )
    def apply(module, name: str, dim: int) -> "WeightNorm":
        for hook in module._forward_pre_hooks.values():
            if isinstance(hook, WeightNorm) and hook.name == name:
                raise RuntimeError(
                    f"Cannot register two weight_norm hooks on the same parameter {name}"
                )

        if dim is None:
            dim = -1

        fn = WeightNorm(name, dim)

        weight = getattr(module, name)
        if isinstance(weight, UninitializedParameter):
            raise ValueError(
                "The module passed to `WeightNorm` can't have uninitialized parameters. "
                "Make sure to run the dummy forward before applying weight normalization"
            )
        # remove w from parameter list
        del module._parameters[name]

        # add g and v as new parameters and express w as g/||v|| * v
        module.register_parameter(
            name + "_g", Parameter(norm_except_dim(weight, 2, dim).data)
        )
        module.register_parameter(name + "_v", Parameter(weight.data))
        setattr(module, name, fn.compute_weight(module))

        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)

        return fn

    def remove(self, module: Module) -> None:
        weight = self.compute_weight(module)
        delattr(module, self.name)
        del module._parameters[self.name + "_g"]
        del module._parameters[self.name + "_v"]
        setattr(module, self.name, Parameter(weight.data))

    def __call__(self, module: Module, inputs: Any) -> None:
        setattr(module, self.name, self.compute_weight(module))


T_module = TypeVar("T_module", bound=Module)


def weight_norm(module: T_module, name: str = "weight", dim: int = 0) -> T_module:
    r"""Apply weight normalization to a parameter in the given module.

    .. math::
         \mathbf{w} = g \dfrac{\mathbf{v}}{\|\mathbf{v}\|}

    Weight normalization is a reparameterization that decouples the magnitude
    of a weight tensor from its direction. This replaces the parameter specified
    by :attr:`name` (e.g. ``'weight'``) with two parameters: one specifying the magnitude
    (e.g. ``'weight_g'``) and one specifying the direction (e.g. ``'weight_v'``).
    Weight normalization is implemented via a hook that recomputes the weight
    tensor from the magnitude and direction before every :meth:`~Module.forward`
    call.

    By default, with ``dim=0``, the norm is computed independently per output
    channel/plane. To compute a norm over the entire weight tensor, use
    ``dim=None``.

    See https://arxiv.org/abs/1602.07868

    .. warning::

        This function is deprecated.  Use :func:`torch.nn.utils.parametrizations.weight_norm`
        which uses the modern parametrization API.  The new ``weight_norm`` is compatible
        with ``state_dict`` generated from old ``weight_norm``.

        Migration guide:

        * The magnitude (``weight_g``) and direction (``weight_v``) are now expressed
          as ``parametrizations.weight.original0`` and ``parametrizations.weight.original1``
          respectively.  If this is bothering you, please comment on
          https://github.com/pytorch/pytorch/issues/102999

        * To remove the weight normalization reparameterization, use
          :func:`torch.nn.utils.parametrize.remove_parametrizations`.

        * The weight is no longer recomputed once at module forward; instead, it will
          be recomputed on every access.  To restore the old behavior, use
          :func:`torch.nn.utils.parametrize.cached` before invoking the module
          in question.

    Args:
        module (Module): containing module
        name (str, optional): name of weight parameter
        dim (int, optional): dimension over which to compute the norm

    Returns:
        The original module with the weight norm hook

    Example::

        >>> warnings.filterwarnings("ignore", message=".*torch.nn.utils.weight_norm")  # docs: hide
        >>> m = weight_norm(nn.Linear(20, 40), name='weight')
        >>> m
        Linear(in_features=20, out_features=40, bias=True)
        >>> m.weight_g.size()
        torch.Size([40, 1])
        >>> m.weight_v.size()
        torch.Size([40, 20])

    """
    WeightNorm.apply(module, name, dim)
    return module


def remove_weight_norm(module: T_module, name: str = "weight") -> T_module:
    r"""Remove the weight normalization reparameterization from a module.

    Args:
        module (Module): containing module
        name (str, optional): name of weight parameter

    Example:
        >>> warnings.filterwarnings(
        ...     "ignore", message=".*torch.nn.utils.weight_norm"
        ... )  # docs: hide
        >>> m = weight_norm(nn.Linear(20, 40))
        >>> remove_weight_norm(m)
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, WeightNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError(f"weight_norm of '{name}' not found in {module}")
<|endoftext|># mypy: allow-untyped-defs
import functools

import torch
from torch.nn.utils._expanded_weights.expanded_weights_impl import ExpandedWeight
from torch.utils import _pytree as pytree


# dependency on `functional_call` means that this can't be exposed in utils
# without creating circular dependency
def call_for_per_sample_grads(
    module,
    *,
    batch_size=None,
    loss_reduction="sum",
    batch_first=True,
):
    r"""
    Return a forward function for a module, populating grad_sample with per sample gradients on backward invocation.

    Args:
        module: The ``nn.Module`` to get per sample gradients with respect to. All trainable
          parameters will compute per sample gradients, located in a ``grad_sample``
          field when ``backward`` is invoked
        batch_size: The batch size of the input. If None is passed, all tensor arguments in args and kwargs must have
          the same batch size, which is the size of the first dimension. Otherwise, it must be passed manually.
          Default: None
        loss_reduction: Indicates if the loss reduction (for aggregating the gradients) is a sum or a mean operation. If
          "mean", per sample gradients will be scaled by the batch size to offset the crossbatch interaction from
          running mean across a batch. Must be "mean" or "sum". Default: "sum"
        batch_first: Indicates if the batch dimension is the first dimension. If True, the batch dimension is the first
          dimension. If False, it's the second dimension. Default: True.

    Examples::
        >>> # xdoctest: +SKIP
        >>> model = nn.Linear(4, 3)
        >>> batched_input = torch.randn(5, 4)  # batch size of 5
        >>> res = call_for_per_sample_grads(model)(batched_input).sum()
        >>> res.backward()
        >>> assert model.weight.shape == (3, 4)
        >>> assert model.weight.grad_sample.shape == (5, 3, 4)
        >>> assert model.weight.grad is None
        >>> assert model.bias.shape == (3,)
        >>> assert model.bias.grad_sample.shape == (5, 3)
        >>> assert model.bias.grad is None

    An example using "mean" loss reduction. The grad_sample fields will be scaled by batch_size from what they would be
    if we ran the same code with loss_reduction="sum". This is because the mean at the end will scale all
    grad_outputs by 1 / batch_size from cross batch interaction.
        >>> model = nn.Linear(4, 3)
        >>> batched_input = torch.randn(5, 4)  # batch size of 5
        >>> res = call_for_per_sample_grads(model, 5, loss_reduction="mean")(
        ...     batched_input
        ... ).mean()
        >>> res.backward()

    Note::
        Does not work with any `nn.RNN`, including `nn.GRU` or `nn.LSTM`. Please use custom
        rewrites that wrap an `nn.Linear` module. See Opacus for an example
    """

    def maybe_build_expanded_weight(og_tensor, batch_size):
        if og_tensor.requires_grad:
            return ExpandedWeight(og_tensor, batch_size, loss_reduction)
        else:
            return og_tensor

    def compute_batch_size(*args, **kwargs):
        args_and_kwargs = pytree.arg_tree_leaves(*args, **kwargs)
        batch_size = None
        for arg in args_and_kwargs:
            if not isinstance(arg, torch.Tensor):
                continue

            arg_batch_size = arg.shape[0] if batch_first else arg.shape[1]
            if batch_size is not None and batch_size != arg_batch_size:
                raise RuntimeError(
                    "When computing batch size, found at least one input with batch size "
                    f"{batch_size} and one with batch size {arg_batch_size}. Please specify it "
                    "explicitly using the batch size kwarg in call_for_per_sample_grads"
                )
            batch_size = arg_batch_size
        if batch_size is None:
            raise RuntimeError(
                "Unable to find a tensor in the passed args and kwargs. They may not be pytree-able "
                "and so ExpandedWeights cannot compute the batch size from the inputs. Please specify "
                "it explicitly"
            )
        return batch_size

    if loss_reduction not in ["sum", "mean"]:
        raise RuntimeError(
            f"Expected loss_reduction argument to be sum or mean, got {loss_reduction}"
        )

    if not isinstance(module, torch.nn.Module):
        raise RuntimeError(
            f"Module passed must be nn.Module, got {type(module).__name__}"
        )
    if not (batch_size is None or isinstance(batch_size, int)):
        raise RuntimeError(
            f"Batch size passed must be None or an integer, got {type(batch_size).__name__}"
        )
    if batch_size is not None and batch_size < 1:
        raise RuntimeError(f"Batch size must be positive, got {batch_size}")
    for weight in module.parameters():
        if hasattr(weight, "grad_sample") and weight.grad_sample is not None:  # type: ignore[attr-defined]
            raise RuntimeError(
                "Current Expanded Weights accumulates the gradients, which will be incorrect for multiple "
                f"calls without clearing gradients. Please clear out the grad_sample parameter of {weight} or "
                "post an issue to pytorch/pytorch to prioritize correct behavior"
            )

    @functools.wraps(module.forward)
    def wrapper(*args, **kwargs):
        wrapper_batch_size = batch_size
        if wrapper_batch_size is None:
            wrapper_batch_size = compute_batch_size(*args, **kwargs)

        params = {
            name: maybe_build_expanded_weight(value, wrapper_batch_size)
            for (name, value) in module.named_parameters()
        }
        return torch.func.functional_call(module, params, args, kwargs)

    return wrapper
<|endoftext|># mypy: allow-untyped-defs
import contextlib
from typing import Any
from typing_extensions import deprecated

import torch
from torch import Tensor
from torch.nn.utils._named_member_accessor import NamedMemberAccessor


__all__ = ["functional_call"]


def _untie_named_tensors_map(
    module: "torch.nn.Module",
    parameters_and_buffers: dict[str, Tensor],
) -> dict[str, Tensor]:
    """
    Unties all tied tensors in the module to parameters_and_buffers.

    This function returns a new untied_parameters_and_buffers dictionary and leave the original
    untied_parameters_and_buffers dictionary unchanged. It adds new (missing) keys for tied tensors
    in the module to untied_parameters_and_buffers. The value of the new key is the user-given value
    in the original parameters_and_buffers dictionary.

    If there are more than one user-given values for the same tied tensor, it will raise an error.

    For example, if the module has two tied weights self.foo and self.tied_foo and the user passes
    {'foo': foo_value, ...}, this will return {'foo': foo_value, 'tied_foo': foo_value, ...}. If the
    user passes {'foo': foo_value, 'tied_foo': tied_foo_value, ...}, it will raise an error. If the
    user passes {'foo': foo_value, 'tied_foo': foo_value, ...}, it will not raise an error.

    Args:
        module (torch.nn.Module): the module to determine which tensors are tied.
        parameters_and_buffers (Dict[str, Tensor]): a map of {name: tensor} for reparameterizing the module.

    Returns:
        A new untied version of the parameters_and_buffers dictionary.

    Raises:
        ValueError: if there are more than one user-given values for the same tied tensor.
    """
    # A map of {name: tensor} for all tensors (including tied ones) in the module.
    all_named_tensors: dict[str, Tensor] = {}
    all_named_tensors.update(module.named_parameters(remove_duplicate=False))
