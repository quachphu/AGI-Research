import torch
import torch.nn
import torch.nn.functional as F
from torch.backends.cuda import (
    can_use_cudnn_attention,
    can_use_efficient_attention,
    can_use_flash_attention,
    cudnn_sdp_enabled,
    flash_sdp_enabled,
    math_sdp_enabled,
    mem_efficient_sdp_enabled,
    SDPAParams,
)
from torch.nn.attention import SDPBackend

from .nested_tensor import NestedTensor


log = logging.getLogger(__name__)


def _validate_sdpa_input(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
) -> None:
    if (
        not isinstance(query, NestedTensor)
        or not isinstance(key, NestedTensor)
        or not isinstance(value, NestedTensor)
    ):
        raise ValueError(
            f"Expected query, key, and value to be nested tensors, "
            f"but got query.is_nested: {query.is_nested}, key.is_nested: {key.is_nested}, "
            f"and value.is_nested: {value.is_nested} instead."
        )
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError(
            f"Expected query, key, and value to have the same dtype, "
            f"but got query.dtype: {query.dtype}, key.dtype: {key.dtype}, "
            f"and value.dtype: {value.dtype} instead."
        )
    if query.device != key.device or query.device != value.device:
        raise ValueError(
            f"Expected query, key, and value to have the same device type, "
            f"but got query.device: {query.device}, key.device: {key.device}, "
            f"and value.device: {value.device} instead."
        )
    if query.dim() < 3 or key.dim() < 3 or value.dim() < 3:
        raise ValueError(
            f"Expected query, key, and value to all be at least 3 dimensional, but got query.dim: "
            f"{query.dim()}, key.dim: {key.dim()} and value.dim: {value.dim()} instead."
        )
    if query._ragged_idx != key._ragged_idx or query._ragged_idx != value._ragged_idx:
        raise ValueError(
            f"Expected query, key, and value to all be ragged on the same dimension, but got ragged "
            f"dims {query._ragged_idx}, {key._ragged_idx}, and {value._ragged_idx}, respectively."
        )
    if attn_mask is not None:
        # TODO: Figure out whether masks are actually supported for this layout or not
        raise ValueError("Masks are not yet supported!")
        if attn_mask.dtype != torch.bool and attn_mask.dtype != query.dtype:
            raise ValueError(
                f"Expected attn_mask dtype to be bool or to match query dtype, but got attn_mask.dtype: "
                f"{attn_mask.dtype}, and query.dtype: {query.dtype} instead."
            )


def _check_batch_size_nested(params: SDPAParams, debug=False) -> bool:
    # This is expected to be called after check_tensor_shapes ensuring that the
    # size() calls won't error since the inputs are all 4 dimensional
    q_batch_size = params.query.size(0)
    k_batch_size = params.key.size(0)
    v_batch_size = params.value.size(0)

    # num_heads logic for nested input is checked in
    # check_for_seq_len_0_nested_tensor as there is handling there to make sure
    # num_heads is not ragged
    return q_batch_size == k_batch_size and q_batch_size == v_batch_size


def _check_head_dim_size_flash_nested(params: SDPAParams, debug=False) -> bool:
    max_size = 256
    query_size_last = params.query.size(-1)
    key_size_last = params.key.size(-1)
    value_size_last = params.value.size(-1)
    same_head_dim_size = (
        query_size_last == key_size_last and query_size_last == value_size_last
    )
    if not (
        same_head_dim_size
        and (query_size_last % 8 == 0)
        and (query_size_last <= max_size)
    ):
        if debug:
            log.warning(
                "For NestedTensor inputs, Flash attention requires q,k,v to have the same "
                "last dimension and to be a multiple of 8 and less than or equal to 256. "
                "Got Query.size(-1): %d, Key.size(-1): %d, Value.size(-1): %d instead.",
                query_size_last,
                key_size_last,
                value_size_last,
            )
        return False
    return True


def _check_head_dim_size_cudnn_nested(params: SDPAParams, debug=False) -> bool:
    max_size = 128
    query_size_last = params.query.size(-1)
    key_size_last = params.key.size(-1)
    value_size_last = params.value.size(-1)
    same_head_dim_size = (
        query_size_last == key_size_last and query_size_last == value_size_last
    )
    if not (
        same_head_dim_size
        and (query_size_last % 8 == 0)
        and (query_size_last <= max_size)
    ):
        if debug:
            log.warning(
                "For NestedTensor inputs, cuDNN attention requires q,k,v to have the same "
                "last dimension and to be a multiple of 8 and less than or equal to 128. "
                "Got Query.size(-1): %d, Key.size(-1): %d, Value.size(-1): %d instead.",
                query_size_last,
                key_size_last,
                value_size_last,
            )
        return False
    return True


def _check_for_seq_len_0_and_consistent_head_dim_nested_helper(
    param: torch.Tensor, param_name: str, debug=False
) -> bool:
    if not isinstance(param, NestedTensor):
        raise AssertionError("param should be a jagged NT")

    if param._ragged_idx == 1:
        # num_head_dims is ragged
        if debug:
            log.warning(
                "Fused kernels do not support ragged num_head_dims, %s has a ragged num_heads.",
                param_name,
            )
        return False

    # This is being called inside sdp with shape [batch, heads, {seq_len}, dim]
    if param._get_min_seqlen() == 0:
        if debug:
            log.warning(
                "Fused kernels do not support seq_len == 0, %s has a seq len of 0.",
                param_name,
            )
        return False

    return True


def _try_broadcast_param_size(q_size, k_size, v_size, param_name, debug=False) -> bool:
    max_size = max(q_size, k_size, v_size)
    if (
        (q_size != max_size and q_size != 1)
        or (k_size != max_size and k_size != 1)
        or (v_size != max_size and v_size != 1)
    ):
        if debug:
            log.warning(
                "Both fused kernels require query, key and value to have broadcastable %s, "
                "got Query %s %d, Key %s %d, Value %s %d instead.",
                param_name,
                param_name,
                q_size,
                param_name,
                k_size,
                param_name,
                v_size,
            )
        return False
    return True


def _check_for_seq_len_0_nested(params: SDPAParams, debug=False) -> bool:
    # When this function is called we are assured that the nt is dim==4
    q_is_safe = (
        _check_for_seq_len_0_and_consistent_head_dim_nested_helper(
            params.query, "query", debug
        )
        if params.query.is_nested
        else True
    )
    # short circuit if any is unsafe
    if not q_is_safe:
        return False

    k_is_safe = (
        _check_for_seq_len_0_and_consistent_head_dim_nested_helper(
            params.key, "key", debug
        )
        if params.key.is_nested
        else True
    )
    # short circuit if any is unsafe
    if not k_is_safe:
        return False

    v_is_safe = (
        _check_for_seq_len_0_and_consistent_head_dim_nested_helper(
            params.value, "value", debug
        )
        if params.value.is_nested
        else True
    )
    # short circuit if any is unsafe
    if not v_is_safe:
        return False

    # We now know none of the inputs have ragged num_heads, so we can safely
    # access .size(1)
    q_num_heads = params.query.size(1)
    k_num_heads = params.key.size(1)
    v_num_heads = params.value.size(1)
    same_num_heads = q_num_heads == k_num_heads and q_num_heads == v_num_heads

    if not same_num_heads:
        if (
            params.query.requires_grad
            or params.key.requires_grad
            or params.value.requires_grad
        ):
            if debug:
                log.warning(
                    "Both fused kernels do not support training with broadcasted NT inputs."
                )
            return False
        return _try_broadcast_param_size(
            q_num_heads, k_num_heads, v_num_heads, "num heads", debug
        )
    return True


def _check_query_seq_len_cudnn_nested(params: SDPAParams, debug=False) -> bool:
    query = params.query
    if not isinstance(query, NestedTensor):
        raise AssertionError("query should be a jagged NT")

    if query._get_max_seqlen() <= 128:
        if debug:
            log.warning(
                "cuDNN attention does not support nested tensor query sequence "
                "length <= 128."
            )
        return False
    return True


def _can_use_flash_sdpa_jagged(params: SDPAParams, debug=False) -> bool:
    constraints = (
        _check_batch_size_nested,
        _check_head_dim_size_flash_nested,
        _check_for_seq_len_0_nested,
    )
    for constraint in constraints:
        if not constraint(params, debug):
            return False
    return True


def _can_use_efficient_sdpa_jagged(params: SDPAParams, debug=False) -> bool:
    constraints = (
        _check_batch_size_nested,
        _check_for_seq_len_0_nested,
    )
    for constraint in constraints:
        if not constraint(params, debug):
            return False
    return True


def _can_use_cudnn_sdpa_jagged(params: SDPAParams, debug=False) -> bool:
    constraints = (
        _check_batch_size_nested,
        _check_head_dim_size_cudnn_nested,
        _check_for_seq_len_0_nested,
        _check_query_seq_len_cudnn_nested,
    )
    for constraint in constraints:
        if not constraint(params, debug):
            return False
    return True


def _can_use_math_sdpa_jagged(params: SDPAParams, debug=False) -> bool:
    if (
        not params.query.transpose(1, 2).is_contiguous()
        or not params.key.transpose(1, 2).is_contiguous()
        or not params.value.transpose(1, 2).is_contiguous()
    ):
        if debug:
            log.warning(
                "If inputs are nested tensors they must be contiguous after transposing."
            )
        return False
    if params.is_causal:
        if debug:
            log.warning(
                "Nested tensors for query / key are not supported when is_causal=True."
            )
        return False
    return True


def _select_sdp_backend(query, key, value, attn_mask, dropout, is_causal, enable_gqa):
    if (
        not flash_sdp_enabled()
        and not mem_efficient_sdp_enabled()
        and not math_sdp_enabled()
        and not cudnn_sdp_enabled()
    ):
        return SDPBackend.ERROR

    ordering = (
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.MATH,
        SDPBackend.CUDNN_ATTENTION,
    )

    params = SDPAParams(query, key, value, attn_mask, dropout, is_causal, enable_gqa)

    for backend in ordering:
        if backend == SDPBackend.CUDNN_ATTENTION:
            if can_use_cudnn_attention(params) and _can_use_cudnn_sdpa_jagged(params):
                return SDPBackend.CUDNN_ATTENTION
        if backend == SDPBackend.FLASH_ATTENTION:
            if can_use_flash_attention(params) and _can_use_flash_sdpa_jagged(params):
                return SDPBackend.FLASH_ATTENTION
        if backend == SDPBackend.EFFICIENT_ATTENTION:
            if can_use_efficient_attention(params) and _can_use_efficient_sdpa_jagged(
                params
            ):
                return SDPBackend.EFFICIENT_ATTENTION
        if backend == SDPBackend.MATH:
            if math_sdp_enabled() and _can_use_math_sdpa_jagged(params):
                return SDPBackend.MATH

    log.warning("Memory efficient kernel not used because:")
    can_use_efficient_attention(params, debug=True)
    _can_use_efficient_sdpa_jagged(params, debug=True)
    log.warning("Flash attention kernel not used because:")
    can_use_flash_attention(params, debug=True)
    _can_use_flash_sdpa_jagged(params, debug=True)
    log.warning("Math attention kernel not used because:")
    _can_use_math_sdpa_jagged(params, debug=True)
    log.warning("cuDNN attention kernel not used because:")
    can_use_cudnn_attention(params, debug=True)
    _can_use_cudnn_sdpa_jagged(params, debug=True)
    return SDPBackend.ERROR


def _cumulative_and_max_seq_len_nnz(qkv: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    # This function is used to calculate two pieces of metadata that are needed
    # for use with flash-attention and efficient_attention kernels. They are the
    # cumulative sequence_length over a batch of sequences and the maximum
    # sequence length.

    # It returns a tuple of cumulative sequence lengths and the maximum sequence
    # length, and the last element in the cumulative_sequence_lengths
    if not isinstance(qkv, NestedTensor):
        raise ValueError("QKV must be nested for flash cumulative_seq_len calculation.")

    if qkv.lengths() is None:
        # TODO: Explore performance impact of copying
        cumulative_seqlen = qkv.offsets().to(dtype=torch.int32, device=qkv.device)
        max_seqlen = qkv._get_max_seqlen()
        n_elem = qkv.values().shape[0]
    else:
        # TODO: Explore performance impact of copying
        cumulative_seqlen = (
            qkv.lengths().cumsum(0).to(dtype=torch.int32, device=qkv.device)
        )
        max_seqlen = qkv._get_max_seqlen()
        # TODO: Explore performance impact when compiling
        n_elem = int(cumulative_seqlen[-1].item())
    return cumulative_seqlen, max_seqlen, n_elem


def _is_safe_to_get_storage_as_tensor(tensor: torch.Tensor) -> bool:
    # This function checks if a nested tensor is valid for
    # use with the flash-attention and efficient_attention kernels without
    # needing to call contiguous on the nested tensor input.
    # It checks that the storage offsets' adjacent_differences are a constant
    # multiple of the previous tensor in the nested tensor and that the strides
    # are monotonically decreasing. This check is done after calling transpose on
    # the nested tensor resulting in a Nt of shape [bsz, {seq_len}, num_heads, dim]

    # Returns a boolean indicating if contiguous needs to be called for input
    if not isinstance(tensor, NestedTensor):
        raise AssertionError("tensor must be a NestedTensor")
    offsets = tensor.offsets()
    strides = tensor._strides

    n_tensors = offsets.size(0) - 1
    if n_tensors <= 1:
        return True

    # Check initially that the tensor strides are in strictly descending order
    prev_stride = strides[1]
    for stride in strides[2:]:
        if prev_stride <= stride:
            # This would mean that the last stride is greater than the seq_len
            # stride
            return False
        prev_stride = stride

    # Congrats you made it!
    return True


def _view_as_dense(
    tensor: torch.Tensor, Nnz: int, num_heads: int, head_dim: int
) -> torch.Tensor:
    if tensor.is_nested:
        return tensor.values()
    return tensor.view(Nnz, num_heads, head_dim)


# TODO: Next iteration should add test cases and check it works
# def _sdpa_nested_preprocessing_with_broadcast(query, key, value):
#     # Query (Batch x Num_heads x {Q_seq_len}  x Dim_per_head)
#     # Key   (Batch x Num_heads x {KV_seq_len} x Dim_per_head)
#     # Value (Batch x Num_heads x {KV_seq_len} x Dim_per_head)
#     q_batch_size = query.size(0)
#     k_batch_size = key.size(0)
#     v_batch_size = value.size(0)

#     output_batch_size = max(q_batch_size, k_batch_size, v_batch_size)

#     q_num_heads = query.size(1)
#     k_num_heads = key.size(1)
#     v_num_heads = value.size(1)

#     output_num_heads = max(q_num_heads, k_num_heads, v_num_heads)

#     head_dim_qk = query.size(3)
#     head_dim_v = value.size(3)

#     q_t = query.transpose(1, 2)
#     k_t = key.transpose(1, 2)
#     v_t = value.transpose(1, 2)

#     # Checks in sdp_utils ensure that if {*}_batch_size/{*}_num_heads !=
#     # output_batch_size/num_heads then they are 1
#     q_batch_size_needs_broadcast = q_batch_size != output_batch_size
#     k_batch_size_needs_broadcast = k_batch_size != output_batch_size
#     v_batch_size_needs_broadcast = v_batch_size != output_batch_size

#     # If {*}_batch_size_needs_broadcast, then
#     # (1) max_seqlen_batch_{*} is given by {*}_t.size(1)
#     #     this is because needs_broadcast indicates that the batch_size is 1
#     #     and hence there is only 1 value for seq_len
#     # (2) The cum_seq_lens are given by [0, {*}_t.size(1), 2 * {*}_t.size(1),
#     # ..., output_batch_size * {*}_t.size(1)]
#     # (3) Nnz_{*} is given by output_batch_size * {*}_t.size(1)

#     if q_batch_size_needs_broadcast or not q_t.is_nested:
#         max_seqlen_batch_q = q_t.size(1)
#         cumulative_sequence_length_q = torch.arange(
#             0,
#             (output_batch_size + 1) * max_seqlen_batch_q,
#             max_seqlen_batch_q,
#             device=q_t.device,
#             dtype=torch.int32,
#         )
#         Nnz_q = output_batch_size * max_seqlen_batch_q
#     else:
#         (
#             cumulative_sequence_length_q,
#             max_seqlen_batch_q,
#             Nnz_q,
#         ) = _cumulative_and_max_seq_len_nnz(q_t)

#     if k_batch_size_needs_broadcast and v_batch_size_needs_broadcast:
#         assert k_t.size(1) == v_t.size(1)
#         max_seqlen_batch_kv = k_t.size(1)
#         cumulative_sequence_length_kv = torch.arange(
#             0,
#             (output_batch_size + 1) * max_seqlen_batch_kv,
#             max_seqlen_batch_kv,
#             device=k_t.device,
#             dtype=torch.int32,
#         )
#         Nnz_kv = output_batch_size * max_seqlen_batch_kv
#     else:
#         cumulative_sequence_length_kv, max_seqlen_batch_kv, Nnz_kv = (
#             _cumulative_and_max_seq_len_nnz(v_t)
#             if k_batch_size_needs_broadcast
#             else _cumulative_and_max_seq_len_nnz(k_t)
#         )

#     q_num_heads_needs_broadcast = q_num_heads != output_num_heads
#     k_num_heads_needs_broadcast = k_num_heads != output_num_heads
#     v_num_heads_needs_broadcast = v_num_heads != output_num_heads

#     if not q_t.is_nested:
#         query_buffer_reshaped = q_t.expand(
#             output_batch_size, q_t.size(1), output_num_heads, head_dim_qk
#         )
#         query_buffer_reshaped = query_buffer_reshaped.reshape(
#             Nnz_q, output_num_heads, head_dim_qk
#         )
#     else:
#         if not q_t.is_contiguous() and not _is_safe_to_get_storage_as_tensor(q_t):
#             q_t = q_t.contiguous()
#         # If we are broadcasting then Nnz_q will be the output_batch_size since
#         # seq_len is 1
#         effective_batch_size_q = (
#             output_batch_size if q_batch_size_needs_broadcast else Nnz_q
#         )
#         query_buffer_reshaped = _view_as_dense(
#             q_t, effective_batch_size_q, output_num_heads, head_dim_qk
#         )

#     # If the physical layout of the NestedTensor's storage
#     # is not: batch, {seq_len}, num_heads, head_dim then we need
#     # to call contiguous
#     if not k_t.is_contiguous() and not _is_safe_to_get_storage_as_tensor(k_t):
#         k_t = k_t.contiguous()
#     if not v_t.is_contiguous() and not _is_safe_to_get_storage_as_tensor(v_t):
#         v_t = v_t.contiguous()

#     effective_batch_size_k = (
#         output_batch_size if k_batch_size_needs_broadcast else Nnz_kv
#     )
#     key_buffer_reshaped = _view_as_dense(
#         k_t, effective_batch_size_k, output_num_heads, head_dim_qk
#     )

#     effective_batch_size_v = (
#         output_batch_size if v_batch_size_needs_broadcast else Nnz_kv
#     )
#     value_buffer_reshaped = _view_as_dense(
#         v_t, effective_batch_size_v, output_num_heads, head_dim_v
#     )

#     if not q_batch_size_needs_broadcast:
#         output_shape = q_t._size
#         if head_dim_v != head_dim_qk:
#             output_shape[-1] = head_dim_v
#         if q_num_heads_needs_broadcast:
#             output_shape[1] = output_num_heads
#     else:
#         output_shape = torch.empty(3, dtype=torch.int64, device=torch.device("cpu"))
#         output_shape[0] = q_t.size(1)
#         output_shape[1] = output_num_heads
#         output_shape[2] = head_dim_v

#     return (
#         query_buffer_reshaped,
#         key_buffer_reshaped,
#         value_buffer_reshaped,
#         cumulative_sequence_length_q,
#         cumulative_sequence_length_kv,
#         max_seqlen_batch_q,
#         max_seqlen_batch_kv,
#         output_shape,
#     )


def _sdpa_nested_preprocessing(query, key, value):
    # Query (Batch x Num_heads x {Q_seq_len}  x Dim_per_head)
    # Key   (Batch x Num_heads x {KV_seq_len} x Dim_per_head)
    # Value (Batch x Num_heads x {KV_seq_len} x Dim_per_head)
    q_batch_size = query.size(0)
    k_batch_size = key.size(0)
    v_batch_size = value.size(0)

    q_num_heads = query.size(1)
    k_num_heads = key.size(1)
    v_num_heads = value.size(1)

    if not (q_batch_size == k_batch_size and q_batch_size == v_batch_size) or not (
        q_num_heads == k_num_heads and k_num_heads == v_num_heads
    ):
        raise RuntimeError(
            "This path is currently not implemented for jagged layout NT."
        )
        # return _sdpa_nested_preprocessing_with_broadcast(query, key, value)

    num_heads = query.size(1)
    head_dim_qk = query.size(3)
    head_dim_v = value.size(3)
    q_t = query.transpose(1, 2)
    k_t = key.transpose(1, 2)
    v_t = value.transpose(1, 2)

    (
        cumulative_sequence_length_q,
        max_seqlen_batch_q,
        Nnz_q,
    ) = _cumulative_and_max_seq_len_nnz(q_t)
    (
        cumulative_sequence_length_kv,
        max_seqlen_batch_kv,
        Nnz_kv,
    ) = _cumulative_and_max_seq_len_nnz(k_t)

    # [TODO] K and V have to have the same Nnz, should probably torch_check
    # assume in order to not iterate over v

    # If the physical layout of the NestedTensor's storage
    # is not: batch, {seq_len}, num_heads, head_dim then we need
    # to call contiguous
    if not q_t.is_contiguous() and not _is_safe_to_get_storage_as_tensor(q_t):
        q_t = q_t.contiguous()
    if not k_t.is_contiguous() and not _is_safe_to_get_storage_as_tensor(k_t):
        k_t = k_t.contiguous()
    if not v_t.is_contiguous() and not _is_safe_to_get_storage_as_tensor(v_t):
        v_t = v_t.contiguous()

    query_buffer_reshaped = _view_as_dense(q_t, Nnz_q, num_heads, head_dim_qk)
    key_buffer_reshaped = _view_as_dense(k_t, Nnz_kv, num_heads, head_dim_qk)
    value_buffer_reshaped = _view_as_dense(v_t, Nnz_kv, num_heads, head_dim_v)

    output_nt_info = {
        "offsets": q_t.offsets(),
        "lengths": q_t.lengths(),
        "max_seqlen": q_t._get_max_seqlen(),
        "min_seqlen": q_t._get_min_seqlen(),
    }

    return (
        query_buffer_reshaped,
        key_buffer_reshaped,
        value_buffer_reshaped,
        cumulative_sequence_length_q,
        cumulative_sequence_length_kv,
        max_seqlen_batch_q,
        max_seqlen_batch_kv,
        output_nt_info,
    )


def _pad_last_dim(
    tensor: torch.Tensor, alignment_size: int, slice: bool
) -> torch.Tensor:
    # FlashAttentionV2 requires that head dimension be a multiple of 8
    # This was previously done within the kernel, however
    # This causes the kernel to maybe alias query, key, value
    # So instead we pad the head_dimensions to be a multiple of 8
    # in the composite region
    last_dim_size = tensor.size(-1)
    if last_dim_size % alignment_size == 0:
        return tensor
    pad_count = alignment_size - (last_dim_size % alignment_size)
    tensor = torch.nn.functional.pad(tensor, [0, pad_count])
    if slice:
        return tensor[..., 0:last_dim_size]
    return tensor


# TODO: coalesce with torch/nn/utils/attention.py
def _calculate_scale(query, scale):
    # TODO: Investigate why math.sqrt() isn't properly handled by Dynamo?
    softmax_scale = scale if scale is not None else torch.sym_sqrt(1.0 / query.size(-1))
    return softmax_scale


def _post_process_flash_output(out: torch.Tensor, og_size):
    if not out.is_nested and out.size(-1) != og_size:
        out = out[..., 0:og_size]
    return out


def _is_computing_meta_flops(x):
    # Note: there's a use case of using meta tensors & the dispatch-based flop counter.
    # We can use this function to check for this scenario in order to handle it specially.
    if not torch.jit.is_scripting() and x.device.type == "meta":
        torch_dispatch_mode_stack = (
            torch.utils._python_dispatch._get_current_dispatch_mode_stack()
        )
        return any(
            type(x) is torch.utils.flop_counter._FlopCounterMode
            for x in torch_dispatch_mode_stack
        )
    return False


def _autocast(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    [Autocasting SDPA for NJT]

    Normal autocasting doesn't work for NJT+SDPA right now:
    * NJT intercepts the __torch_function__ call for scaled_dot_product_attention, which happens
      before we get to any aten ops or dispatcher logic; then the torch_function logic calls into
      efficient attention or flash attention. So, autocasting on the scaled_dot_product_attention
      op won't work because we never see that aten op.
    * If we put autocasting on `_flash_attention_forward`, then we'll get autocasting to run, but
      the kernel selection logic in torch_function handling (ie. jagged_scaled_dot_product_attention)
      won't work correctly: the kernel selection logic will run before autocasting, and choose
      a kernel based on the un-autocasted dtypes; but then autocasting will run and the actual
      attention computation will happen in a different dtype.

    An alternative is to just change the backend selection logic for SDPA+NJT to be autocast-aware
    and rely on autocasting to do the actual conversions for flash attention / efficient attention.
    However, by manually doing the actual autocast before the backend selection, we ensure that the
    autocast handling for backend selection doesn't diverge from the autocast handling for the
    actual dtype conversions.
    """
    device_type = query.device.type
    # meta device is not supported by autocast, so break early for it
    if _is_computing_meta_flops(query) or not torch.is_autocast_enabled(device_type):
        return query, key, value, attn_mask

    def cvt(x):
        if x is None:
            return x
        target_dtype = torch.get_autocast_dtype(device_type)
        if (
            (not x.dtype.is_floating_point)
            or x.dtype == target_dtype
            or x.dtype == torch.float64
        ):
            return x
        return x.to(target_dtype)

    return cvt(query), cvt(key), cvt(value), cvt(attn_mask)


def jagged_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    query, key, value, attn_mask = _autocast(query, key, value, attn_mask)
    _validate_sdpa_input(query, key, value, attn_mask, dropout_p, is_causal, scale)
    # for mypy, ugh
    if not (
        isinstance(query, NestedTensor)
        and isinstance(key, NestedTensor)
        and isinstance(value, NestedTensor)
    ):
        raise AssertionError("query, key, and value must all be NestedTensor instances")
    from torch.nested._internal.nested_tensor import (
        nested_view_from_values_offsets_lengths,
    )

    # Special path for non-ragged sequence length (e.g. for SAM where we have a ragged
    # second batch dim instead). For this case, we can just send the dense buffers through
    # vanilla SDPA.
    if query.dim() > 3 and key.dim() > 3 and value.dim() > 3 and query._ragged_idx == 1:
        output = F.scaled_dot_product_attention(
            query.values(),
            key.values(),
            value.values(),
            attn_mask=(
                attn_mask.values() if isinstance(attn_mask, NestedTensor) else attn_mask
            ),
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )
        return nested_view_from_values_offsets_lengths(
            output,
            query.offsets(),
            query.lengths(),
            min_seqlen=query._maybe_min_seqlen,  # type: ignore[attr-defined]
            max_seqlen=query._maybe_max_seqlen,  # type: ignore[attr-defined]
        )

    compute_logsumexp = query.requires_grad or key.requires_grad or value.requires_grad

    backend_choice = _select_sdp_backend(
        query, key, value, attn_mask, dropout_p, is_causal, enable_gqa
    )

    if _is_computing_meta_flops(query):
        # Backend choice will probably not be correct if we have a meta device,
        # because backend choice is device-aware. In this case, we mostly just
        # want to avoid using math backend (which does a .item() call).
        # Arbitrarily choose flash attention.
        backend_choice = SDPBackend.FLASH_ATTENTION

    if backend_choice == SDPBackend.FLASH_ATTENTION:
        og_size = query.size(-1)
        query_padded = _pad_last_dim(query, 8, False)
        key_padded = _pad_last_dim(key, 8, False)
        value_padded = _pad_last_dim(value, 8, False)
        # We need to calculate the scale based off the OG head dim size
        og_scale = _calculate_scale(query, scale)
        (
            query_buffer_reshaped,
            key_buffer_reshaped,
            value_buffer_reshaped,
            cumulative_sequence_length_q,
            cumulative_sequence_length_kv,
            max_seqlen_batch_q,
            max_seqlen_batch_kv,
            output_nt_info,
        ) = _sdpa_nested_preprocessing(query_padded, key_padded, value_padded)
        (
            attention,
            _logsumexp,
            _philox_seed,
            _philox_offset,
            _debug_attn_mask,
        ) = torch.ops.aten._flash_attention_forward(
            query_buffer_reshaped,
            key_buffer_reshaped,
            value_buffer_reshaped,
            cumulative_sequence_length_q,
            cumulative_sequence_length_kv,
            max_seqlen_batch_q,
            max_seqlen_batch_kv,
            dropout_p,
            is_causal,
            False,
            scale=og_scale,
        )
        # Reshape output to convert nnz to batch_size and seq_len
        attention = nested_view_from_values_offsets_lengths(
            attention,  # output from flash_attn is [total_q, num_heads, head_size_og]
            **output_nt_info,
        ).transpose(1, 2)
        return _post_process_flash_output(attention, og_size)
    elif backend_choice == SDPBackend.EFFICIENT_ATTENTION:
        (
            query_reshaped,
            key_reshaped,
            value_reshaped,
            cumulative_sequence_length_q,
            cumulative_sequence_length_kv,
            max_seqlen_batch_q,
            max_seqlen_batch_kv,
            output_nt_info,
        ) = _sdpa_nested_preprocessing(query, key, value)
        (
            attention,
            log_sumexp,
            seed,
            offset,
            max_seqlen_q,
            max_seqlen_batch_kv,
        ) = torch.ops.aten._efficient_attention_forward(
            query_reshaped.unsqueeze(0),
            key_reshaped.unsqueeze(0),
            value_reshaped.unsqueeze(0),
            None,
            cumulative_sequence_length_q,
            cumulative_sequence_length_kv,
            max_seqlen_batch_q,
            max_seqlen_batch_kv,
            dropout_p,
            int(is_causal),
            compute_logsumexp,
            scale=scale,
        )
        # Reshape output to convert nnz to batch_size and seq_len
        return nested_view_from_values_offsets_lengths(
            attention.squeeze(0),
            **output_nt_info,
        ).transpose(1, 2)
    elif backend_choice == SDPBackend.CUDNN_ATTENTION:
        (
            query_reshaped,
            key_reshaped,
            value_reshaped,
            cumulative_sequence_length_q,
            cumulative_sequence_length_kv,
            max_seqlen_batch_q,
            max_seqlen_batch_kv,
            output_nt_info,
        ) = _sdpa_nested_preprocessing(query, key, value)
        (
            attention,
            logsumexp,
            cum_seqlen_q,
            cum_seqlen_kv,
            max_seqlen_q,
            max_seqlen_kv,
            seed,
            offset,
            _,
        ) = torch.ops.aten._cudnn_attention_forward(
            query_reshaped,
            key_reshaped,
            value_reshaped,
            attn_mask,
            cumulative_sequence_length_q,
            cumulative_sequence_length_kv,
            max_seqlen_batch_q,
            max_seqlen_batch_kv,
            compute_logsumexp,
            dropout_p,
            is_causal,
            False,
            scale=scale,
        )
        return nested_view_from_values_offsets_lengths(
            attention,
            **output_nt_info,
        ).transpose(1, 2)
    elif backend_choice == SDPBackend.MATH:
        # save the offsets and shape of the inputs, so we can reshape the final output
        # query @ key = attn: [B, D1, j0, D'] @ [B, D1, D' j1] = [B, D1, j0, j1]
        # attn @ value = out: [B, D1, j0, j1] @ [B, D1, j1, D2] = [B, D1, j0, D2]
        offsets = query.offsets()
        q_lengths = query.lengths()
        min_seqlen = query._maybe_min_seqlen
        max_seqlen = query._maybe_max_seqlen
        d1 = query._size[1]
        d2 = value._size[-1]

        # convert jagged layout Nested Tensor to strided layout Nested Tensor
        # which support the math implementation of SDPA
        def get_strided_layout_nested_tensor(jagged_layout_nt):
            lengths = jagged_layout_nt._offsets[1:] - jagged_layout_nt._offsets[:-1]
            transpose = torch.transpose(jagged_layout_nt, 1, 2)
            tensor_list = transpose.values().split(list(lengths), dim=0)
            strided_nt = torch.nested.as_nested_tensor(list(tensor_list))
            strided_nt = strided_nt.transpose(1, 2).contiguous()
            return strided_nt

        query = get_strided_layout_nested_tensor(query)
        key = get_strided_layout_nested_tensor(key)
        value = get_strided_layout_nested_tensor(value)

        attn_out = torch._scaled_dot_product_attention_math(
            query, key, value, attn_mask, dropout_p, is_causal, scale=scale
        )[0]

        # convert strided layout Nested Tensor back to jagged layout Nested Tensor
        attn_out = attn_out.transpose(1, 2).contiguous().values()
        attn_out = attn_out.view(-1, d1, d2)
        attn_out = nested_view_from_values_offsets_lengths(
            attn_out,
            offsets,
            lengths=q_lengths,
            min_seqlen=min_seqlen,
            max_seqlen=max_seqlen,
        ).transpose(1, 2)

        return attn_out
    else:
        raise RuntimeError(
            "No viable backend for scaled_dot_product_attention was found."
        )
<|endoftext|># mypy: allow-untyped-defs
import functools
import operator
from typing import *  # noqa: F403

import torch
import torch.nn.functional as F
from torch.fx.operator_schemas import normalize_function
from torch.nested._internal.sdpa import jagged_scaled_dot_product_attention

from .nested_tensor import _jagged_numel, NestedTensor


__all__: list[Any] = []

JAGGED_OPS_TABLE: Dict[Any, Any] = {}


def _get_padding_value(dtype, padding_type):
    if dtype.is_floating_point:
        return (
            torch.finfo(dtype).max if padding_type == "max" else torch.finfo(dtype).min
        )
    else:
        # For integer dtypes, use infinity sentinels which the C++ implementation
        # clamps to dtype min/max, avoiding precision loss through double.
        return float("inf") if padding_type == "max" else float("-inf")


def _outer_to_inner_dim(ndim, dim, ragged_dim, canonicalize=False):
    from torch._prims_common import canonicalize_dims

    if isinstance(dim, (tuple, list)):
        output = type(dim)(_outer_to_inner_dim(ndim, d, ragged_dim) for d in dim)
        # ensure no duplicates, which can result from both batch and ragged mapping to 0
        return type(output)(dict.fromkeys(output))

    if canonicalize:
        dim = canonicalize_dims(ndim, dim)

    if not (dim >= 0 and dim < ndim):  # pyrefly: ignore [unsupported-operation]
        raise AssertionError(f"dim {dim} out of range for ndim {ndim}")

    # Map dim=0 (AKA batch dim) -> packed dim i.e. outer ragged dim - 1.
    # For other dims, subtract 1 to convert to inner space.
    return (
        # pyrefly: ignore [unsupported-operation]
        ragged_dim - 1 if dim == 0 else dim - 1
    )


def _wrap_jagged_dim(
    ndim,
    dim,
    ragged_dim,
    op_name,
    convert_to_inner_dim=True,
    allow_ragged_dim=False,
    allow_batch_dim=False,
):
    from torch._prims_common import canonicalize_dims

    wrapped = canonicalize_dims(ndim, dim)
    if wrapped == ragged_dim and not allow_ragged_dim:
        raise RuntimeError(f"{op_name}(): not supported for NestedTensor on ragged dim")
    elif wrapped == 0 and not allow_batch_dim:
        raise RuntimeError(f"{op_name}(): not supported for NestedTensor on dim=0")
    ret = (
        _outer_to_inner_dim(ndim, wrapped, ragged_dim)
        if convert_to_inner_dim
        else wrapped
    )
    if allow_batch_dim:
        # Need to disambiguate whether we're operating on the batch dim or not.
        # Operating on dim=1 -> dim=0 after the inner dim conversion.
        operating_on_batch = wrapped == 0
        return (ret, operating_on_batch)
    return ret


def _wrap_jagged_dims(ndim, dims, op_name, ragged_idx=1):
    """
    For NestedTensor operators,
    wraps dimensions to non-negative values,
    and returns metadata related to reduction dimension(s).
    """
    from torch._prims_common import canonicalize_dims

    if not isinstance(dims, (tuple, list)):
        raise AssertionError(
            f"_wrap_jagged_dims(): cannot iterate over dimensions of type {type(dims)}"
        )

    wrapped_dims = [
        canonicalize_dims(ndim, d) for d in dims
    ]  # convert all indices to non-negative values

    operate_on_batch = 0 in wrapped_dims
    operate_on_ragged = ragged_idx in wrapped_dims
    operate_on_non_batch = any(d != 0 and d != ragged_idx for d in wrapped_dims)

    # ensure no duplicates, which can result from both batch and ragged mapping to 0
    outer_to_inner_dim = tuple(
        dict.fromkeys(_outer_to_inner_dim(ndim, d, ragged_idx) for d in wrapped_dims)
    )

    return outer_to_inner_dim, operate_on_batch, operate_on_ragged, operate_on_non_batch


def check_schema(schema_str: str, func, *args, **kwargs) -> None:
    named_arg_types = schema_str.split(", ")
    num_optional_args = [x.endswith("?") for x in named_arg_types].count(True)
    min_args = len(named_arg_types) - num_optional_args

    # special case: ellipses allows for any number of unchecked args at the end
    if named_arg_types[-1] == "...":
        named_arg_types = named_arg_types[:-1]
    else:
        if not (len(args) >= min_args and len(args) <= len(named_arg_types)):
            raise ValueError(
                f"NestedTensor {func.__name__}({schema_str}): expected at least {min_args} "
                f"arguments and at most {len(named_arg_types)} arguments, but got: "
                f"{len(args)} arguments"
            )

    arg_type_check_fns = {
        "t": lambda x: isinstance(x, torch.Tensor) and not isinstance(x, NestedTensor),
        "jt": lambda x: isinstance(x, NestedTensor)
        and x._lengths is None
        and x._ragged_idx == 1,  # ops with "jt" require contiguous JT only
        "jt_all": lambda x: isinstance(
            x, NestedTensor
        ),  # ops with "jt_all" can accept all kinds of JT
        "any": lambda x: True,
    }
    for i, named_arg_type in enumerate(named_arg_types):
        name, arg_type = named_arg_type.split(": ")
        is_optional = arg_type.endswith("?")
        normalized_arg_type = arg_type[:-1] if is_optional else arg_type
        if normalized_arg_type not in arg_type_check_fns:
            raise AssertionError(f"Unknown arg type: {normalized_arg_type}")

        if i >= len(args):
            if not is_optional:
                raise ValueError(
                    f"NestedTensor {func.__name__}({schema_str}) "
                    f"missing required argument: {name}"
                )
            continue

        _check_fn = arg_type_check_fns[normalized_arg_type]

        def check_fn(x, is_optional=is_optional):
            if is_optional:
                return x is None or _check_fn(x)
            else:
                return _check_fn(x)

        if not check_fn(args[i]):
            type_to_desc = {
                "t": "tensor",
                "t?": "optional tensor",
                "jt": "contiguous jagged layout NestedTensor",
                "jt_all": "jagged layout NestedTensor",
                "any": "<any type>",
            }

            raise ValueError(
                f"NestedTensor {func.__name__}({schema_str}): expected {name} to be a "
                f"{type_to_desc[arg_type]}"
            )


def check_ragged_dim_same(
    func, a: NestedTensor, a_name: str, b: NestedTensor, b_name: str
) -> None:
    # Calling into .shape here
    if a._size[a._ragged_idx] != b._size[b._ragged_idx]:
        raise RuntimeError(
            f"NestedTensor {func.__name__}: expected {a_name} and {b_name} to have the "
            "same exact offsets tensor."
        )


# returns True if the raggedness-relevant portions of the NT shape
# match those of the specified size
def raggedness_matches(nt, size):
    end = nt._ragged_idx + 1
    nt_ragged = nt._size[:end]
    size_ragged = size[:end]
    return len(nt_ragged) == len(size_ragged) and (
        all(ns == s or s == -1 for ns, s in zip(nt_ragged, size_ragged))
    )


def squeeze_leading_ones(t):
    # Note: [ Squeezing leading ones ]
    #
    # Squeeze leading ones from t.
    #
    # We want:
    #   (B, j0, ?, ?) + (1, 1, ?, ?) -> (B, j0, ?, ?)
    #   (B, j0, ?, ?) + (1, 1, 1, ?, ?) -> (1, B, j0, ?, ?)  (not yet supported)
    #
    # 1) Squeeze extra ones and grab values from NT
    #   (1, 1, ?, ?) -> (?, ?)   and   (sum(*), ?, ?) -> (B, j0, ?, ?)
    # 2) Do dense broadcasting:
    #   (sum(*), ?, ?) + (?, ?) -> (sum(*), ?, ?)
    # 3) Construct nested tensor
    #   (sum(*), ?, ?) -> (B, j0, ?, ?)
    #
    # If unsqueezing on the 0th dim becomes supported, we would unsqueeze
    # at step (4) and we would need to update this function to record how
    # many ones we unsqueezed.
    while t.dim() > 0 and t.shape[0] == 1:
        t = t.squeeze(0)
    return t


def register_func(tables, aten_ops, schema_str):
    if not isinstance(aten_ops, list):
        aten_ops = [aten_ops]
    if not isinstance(tables, list):
        tables = [tables]

    def wrapper(func):
        for aten_op in aten_ops:

            def get_inner(aten_op):
                def inner(*args, **kwargs):
                    check_schema(schema_str, func, *args, **kwargs)
                    return func(aten_op, *args, **kwargs)

                return inner

            for table in tables:
                table[aten_op] = get_inner(aten_op)
        return func

    return wrapper


register_jagged_func = functools.partial(register_func, JAGGED_OPS_TABLE)


def lookup_jagged(func, *args, **kwargs) -> Callable | None:
    dispatch_func = JAGGED_OPS_TABLE.get(func, None)
    if dispatch_func is not None:
        return dispatch_func

    # Handle pointwise fallbacks
    if torch.Tag.pointwise in func.tags:
        from torch.fx.experimental.symbolic_shapes import is_nested_int

        # No pointwise ops legitimately accept nested int inputs. Without this check,
        # they will be incorrectly interpreted as tensors.
        # See https://github.com/pytorch/pytorch/issues/138496
        for arg in args:
            if is_nested_int(arg):
                raise RuntimeError(
                    f"NestedTensor {func.__name__}: invalid argument {arg}"
                )

        # Assume there aren't additional tensors that aren't the "unary/binary" args
        num_tensor_args = sum(isinstance(x, torch.Tensor) for x in args)
        if num_tensor_args == 1:
            # Build up the check schema string. The first tensor arg is assumed to be
            # an NJT and other args are sent through as-is.
            schema_parts = []
            for arg in func._schema.arguments:
                if isinstance(arg.type, torch.TensorType):
                    schema_parts.append(f"{arg.name}: jt_all")
                    break
                else:
                    schema_parts.append(f"{arg.name}: any")
            schema_parts.append("...")
            check_schema_str = ", ".join(schema_parts)
            check_schema(check_schema_str, func, *args, **kwargs)
            return functools.partial(jagged_unary_pointwise, func)
        elif num_tensor_args == 2:
            check_schema("lhs: any, rhs: any, ...", func, *args, **kwargs)
            return functools.partial(jagged_binary_pointwise, func)

    return None


def extract_kwargs(arg):
    kwargs = {
        "offsets": arg.offsets(),
        "lengths": arg.lengths(),
        "_metadata_cache": arg._metadata_cache,
        "_ragged_idx": arg._ragged_idx,
    }
    return kwargs


def jagged_unary_pointwise(func, *args, **kwargs):
    # assume if we get here that there is a single NJT input in the args
    njt = next(arg for arg in args if isinstance(arg, NestedTensor))
    return NestedTensor(
        func(*(arg._values if arg is njt else arg for arg in args), **kwargs),
        **extract_kwargs(njt),
    )


def jagged_binary_pointwise(func, *args, **kwargs):
    a, b = args[0], args[1]
    if not (isinstance(a, NestedTensor) or isinstance(b, NestedTensor)):
        raise AssertionError("At least one of the arguments must be a NestedTensor")

    mismatch_error_msg = (
        "cannot call binary pointwise function {} with inputs of shapes {} and {}"
    )
    # a is NT, b is NT
    if isinstance(a, NestedTensor) and isinstance(b, NestedTensor):
        # ex: (B, j0, D) + (B, j0, D)
        # ex: (B, j0, D) + (B, j0, 1)
        if raggedness_matches(a, b._size):
            return NestedTensor(
                func(a._values, b._values, *args[2:], **kwargs), **extract_kwargs(a)
            )
        raise RuntimeError(mismatch_error_msg.format(func.__name__, a._size, b._size))
    # either a is NT or b is NT at this point
    a_is_nt = isinstance(a, NestedTensor)
    extracted_kwargs = extract_kwargs(a) if a_is_nt else extract_kwargs(b)

    # === Handle broadcasting across the batch / ragged dims ===

    # Easy case: take advantage of pre-existing broadcasting logic
    # ex: (B, j0, ?, ?) + (?) -> (B, j0, ?, ?)
    # ex: (B, j0, ?, ?) + (?, ?) -> (B, j0, ?, ?)
    # ex: (B, j0, ?, ?) + (1, 1, ?, ?) -> (B, j0, ?, ?)
    nt, t = (a, b) if a_is_nt else (b, a)
    # See Note: [ Squeezing leading ones ]
    if t.dim() > nt.dim():
        raise NotImplementedError("NYI: broadcasting NT with T with larger dim")
    t_squeezed = squeeze_leading_ones(t)
    if nt.dim() >= t_squeezed.dim() + 2:
        lhs, rhs = (nt._values, t_squeezed) if a_is_nt else (t_squeezed, nt._values)
        return NestedTensor(func(lhs, rhs, *args[2:], **kwargs), **extracted_kwargs)

    # Harder case: do manual broadcasting when NT dim == non-NT dim
    # ex: (B, j0, D_0, D_1) + (B, 1, D_0, D_1) -> (B, j0, D_0, D_1)
    if a.dim() == b.dim():
        # ex: (B, j0, D_0, D_1) + (1, 1, D_0, D_1) -> should
        # be (B, j0, D_0, D_1) but not yet supported
        if a.shape[0] != b.shape[0]:
            raise RuntimeError(
                mismatch_error_msg.format(func.__name__, a.shape, b.shape)
            )

        from .nested_tensor import nested_from_padded

        # handle broadcasting via padded dense -> jagged conversion
        min_seqlen = nt._maybe_min_seqlen
        max_seqlen = nt._maybe_max_seqlen
        padded_max_S = max_seqlen
        total_L = nt._values.shape[nt._ragged_idx - 1]
        if padded_max_S is None:
            # use upper bound on max seqlen if it's not present
            padded_max_S = total_L

        # convert dense tensor -> jagged
        t = t.expand(
            [x if i != nt._ragged_idx else padded_max_S for i, x in enumerate(t.shape)]
        )
        t_as_nt = nested_from_padded(
            t,
            offsets=nt._offsets,
            ragged_idx=nt._ragged_idx,
            sum_S=total_L,
            min_seqlen=min_seqlen,
            max_seqlen=max_seqlen,
        )

        # function call with two NJTs
        lhs, rhs = (nt, t_as_nt) if a_is_nt else (t_as_nt, nt)
        return func(lhs, rhs, *args[2:], **kwargs)

    # ex: (B, j0, D_0, D_1) + (A, B, 1, D_0, D_1) -> error because this breaks the invariant
    # that ragged dim is wrt left-most batch dim
    raise RuntimeError(mismatch_error_msg.format(func.__name__, a.shape, b.shape))


def jagged_torch_function(func, *args, **kwargs):
    # SDPA has special kernels that handle nested tensors.
    # Dispatch to the correct implementation here
    if func is torch._C._nn.scaled_dot_product_attention:
        return jagged_scaled_dot_product_attention(*args, **kwargs)

    if func.__name__ == "apply_":
        func(args[0]._values, *args[1:], **kwargs)
        return args[0]

    # Handle flatten() here because it's CompositeImplicit.
    if func.__name__ == "flatten":

        def _flatten_sig(input, start_dim=0, end_dim=-1) -> None:
            pass

        _, new_kwargs = normalize_function(  # type: ignore[misc]
            _flatten_sig, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
        )

        inp = new_kwargs.pop("input")

        # NB: stay in outer dim space because we're going to redispatch on a NT input
        start_dim = _wrap_jagged_dim(
            inp.dim(),
            new_kwargs["start_dim"],
            inp._ragged_idx,
            "flatten",
            convert_to_inner_dim=False,
        )
        end_dim = _wrap_jagged_dim(
            inp.dim(),
            new_kwargs["end_dim"],
            inp._ragged_idx,
            "flatten",
            convert_to_inner_dim=False,
        )

        if start_dim == end_dim:
            return inp

        product = functools.reduce(operator.mul, inp.shape[start_dim : end_dim + 1])
        new_shape = (*inp.shape[:start_dim], product, *inp.shape[end_dim + 1 :])

        return inp.reshape(*new_shape)

    # Handle NestedTensor share_memory_.
    if func.__name__ == "share_memory_":
        nt = args[0]

        if not nt.is_cpu and not nt.is_meta:
            return nt

        names, _ = nt.__tensor_flatten__()
        with torch._C.DisableTorchFunctionSubclass():
            for name in names:
                component = getattr(nt, name, None)
                if component is not None:
                    component.share_memory_()
        return nt

    # Handle NestedTensor is_shared.
    if func.__name__ == "is_shared":
        nt = args[0]

        if not nt.is_cpu and not nt.is_meta:
            return True

        names, _ = nt.__tensor_flatten__()
        if not names:
            return False
        return all(
            getattr(nt, name) is not None and getattr(nt, name).is_shared()
            for name in names
        )

    # Handle nested-specific input validation for CompositeImplicit rms_norm
    if func.__name__ == "rms_norm":

        def _rms_norm_sig(input, normalized_shape, weight=None, eps=None) -> None:
            pass

        _, new_kwargs = normalize_function(  # type: ignore[misc]
            _rms_norm_sig, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
        )

        inp = new_kwargs.pop("input")
        normalized_shape = new_kwargs.pop("normalized_shape")

        # can't normalize over the ragged dim (yet)
        max_normalizable = inp.dim() - inp._ragged_idx - 1
        if len(normalized_shape) > max_normalizable:
            raise ValueError(
                "rms_norm(): Normalization over the ragged dim not supported for nested tensors"
            )

        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    raise NotImplementedError(func)


@register_jagged_func(
    [
        torch.ops.aten.is_non_overlapping_and_dense.default,
        torch.ops.aten.sym_size.default,
        torch.ops.aten.dim.default,
        torch.ops.aten.numel.default,
        torch.ops.aten.sym_numel.default,
        torch.ops.aten.sym_stride.default,
        torch.ops.aten.sym_storage_offset.default,
    ],
    "self: jt_all",
)
def tensor_attr_supported_getter(func, *args, **kwargs):
    if func is torch.ops.aten.is_non_overlapping_and_dense.default:
        return False

    if func is torch.ops.aten.sym_size.default:
        return args[0]._size

    if func is torch.ops.aten.dim.default:
        return len(args[0]._size)

    if func in (torch.ops.aten.sym_numel.default, torch.ops.aten.numel.default):
        return _jagged_numel(args[0], func)

    if func is torch.ops.aten.sym_stride.default:
        return args[0]._strides

    if func is torch.ops.aten.sym_storage_offset.default:
        return args[0]._values.storage_offset()


@register_jagged_func(torch.ops.prim.layout.default, "self: jt_all")
def prim_layout_default(func, *args, **kwargs):
    return torch.jagged


@register_jagged_func(
    [torch.ops.aten.size.default],
    "self: jt_all",
)
def tensor_attr_unsupported_getter(func, *args, **kwargs) -> None:
    if func is torch.ops.aten.size.default:
        raise RuntimeError(
            "NestedTensor does not support directly calling torch.ops.aten.size; "
            "please use `nested_tensor.size()` instead."
        )


@register_jagged_func(torch.ops.aten.is_contiguous.default, "self: jt_all")
def is_contiguous_general(func, *args, **kwargs):
    from torch._prims_common import is_contiguous_for_memory_format

    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )
    inp = new_kwargs.pop("input")

    # If created from narrow() check for lengths
    if inp.lengths() is not None:
        return False

    new_kwargs["memory_format"] = new_kwargs.get(
        "memory_format", torch.contiguous_format
    )
    if new_kwargs["memory_format"] == torch.preserve_format:
        return True
    return is_contiguous_for_memory_format(inp._values, **new_kwargs)


register_jagged_func(
    torch.ops.aten.is_contiguous.memory_format, "self: jt_all, memory_format: any?"
)(is_contiguous_general)


@register_jagged_func(
    torch.ops.aten.sym_is_contiguous.default, "self: jt_all, memory_format: any?"
)
def sym_is_contiguous_general(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )
    inp = new_kwargs.pop("input")

    # If created from narrow() check for lengths
    if inp.lengths() is not None:
        return False

    new_kwargs["memory_format"] = new_kwargs.get(
        "memory_format", torch.contiguous_format
    )

    if new_kwargs["memory_format"] == torch.preserve_format:
        return True

    return torch.ops.aten.sym_is_contiguous.default(inp._values, **new_kwargs)


@register_jagged_func(
    torch.ops.aten.clone.default, "input: jt_all, memory_format: any?"
)
def clone_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    new_meta = extract_kwargs(inp)

    if inp._lengths is not None:
        if new_kwargs["memory_format"] == torch.contiguous_format:
            # need to copy to remove "holes" non-contiguity / lengths metadata
            # TODO: write a kernel for this
            from .nested_tensor import jagged_from_list

            # TODO: We probably want the output to have the same ragged structure / nested int.
            if inp._ragged_idx != 1:
                raise AssertionError(
                    "NJT with ragged_idx != 1 not supported for contiguous clone"
                )
            contig, _ = jagged_from_list(inp.unbind(), offsets=None)
            return contig

    return NestedTensor(func(inp._values, **new_kwargs), **new_meta)


@register_jagged_func(torch.ops.aten.linear.default, "input: jt, weight: t, bias: t?")
def linear_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))


@register_jagged_func(
    torch.ops.aten.linear_backward.default,
    "self: jt, grad_output: jt, weight: t, output_mask: any",
)
def linear_backward_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    grad_output = new_kwargs.pop("grad_output")
    weight = new_kwargs.pop("weight")
    output_mask = new_kwargs.pop("output_mask")

    ds, dw, db = None, None, None
    check_ragged_dim_same(func, inp, "self", grad_output, "grad_output")
    if output_mask[0]:
        ds = NestedTensor(
            torch.matmul(grad_output._values, weight), **extract_kwargs(grad_output)
        )
    if output_mask[1]:
        # NB: Fold dims of values for input and grad_output to treat them as 2D. This
        # trick avoids materializing large intermediates and immediately reducing over
        # them via sum(). This is equivalent to computing:
        #     torch.matmul(grad_output._values.transpose(-2, -1), inp._values)
        # and then summing over the leading dimensions to get a 2D weight grad.
        grad_2d = grad_output._values.reshape(-1, weight.size(0))
        input_2d = inp._values.reshape(-1, weight.size(1))
        dw = torch.matmul(grad_2d.t(), input_2d)
    if output_mask[2]:
        # Sum over all but the last dim to get a 1D bias grad. We cannot
        # rely on the autograd engine to reduce for us, because returning a
        # tensor aliasing the input would violate the aten signature annotation
        reduce_dims = tuple(range(grad_output._values.ndim - 1))
        if reduce_dims == ():
            db = grad_output._values.clone()
        else:
            db = torch.sum(grad_output._values, reduce_dims, keepdim=False)
    return (ds, dw, db)


@register_jagged_func(torch.ops.aten.to.dtype, "input: jt_all, dtype: any")
def to_dtype(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))


@register_jagged_func(torch.ops.aten._to_copy.default, "self: jt_all")
def to_copy_default(func, *args, **kwargs):
    from .nested_tensor import _tensor_symint_registry

    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    # don't change layout
    new_kwargs.pop("layout")

    new_values = func(inp._values, **new_kwargs)
    new_offsets = inp._offsets.to(device=new_values.device)
    new_lengths = None
    if inp._lengths is not None:
        new_lengths = inp._lengths.to(device=new_values.device)

    from torch._subclasses.fake_tensor import is_fake_tensor
    from torch._subclasses.functional_tensor import (
        FunctionalTensor,
        mb_unwrap_functional_tensor,
    )

    ragged_source = inp._offsets if inp._lengths is None else inp._lengths
    new_thing = new_offsets if new_lengths is None else new_lengths
    if is_fake_tensor(new_thing) or isinstance(new_thing, FunctionalTensor):
        # Temporary hack until we have the union find
        tgt = mb_unwrap_functional_tensor(new_thing)
        src = mb_unwrap_functional_tensor(ragged_source)
        # pyrefly: ignore[missing-attribute]
        tgt.nested_int_memo = src.nested_int_memo
    else:
        _tensor_symint_registry[new_thing] = _tensor_symint_registry[ragged_source]
    inp_kwargs = extract_kwargs(inp)
    inp_kwargs["offsets"] = new_offsets
    inp_kwargs["lengths"] = new_lengths

    output = NestedTensor(new_values, **inp_kwargs)
    return output


@register_jagged_func(
    torch.ops.aten.copy_.default, "self: jt_all, src: jt_all, non_blocking: any?"
)
def copy_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )
    inp = new_kwargs.pop("input")
    src = new_kwargs.pop("src")
    if inp._size != src._size:
        # try to recursively copy_ on unbound components to get around nested int mismatch
        # TODO: eventually do a direct copy when this is possible
        inp_comps = inp.unbind()
        inp_comp_shapes = [c.shape for c in inp_comps]
        src_comps = src.unbind()
        src_comp_shapes = [c.shape for c in src_comps]
        if inp_comp_shapes != src_comp_shapes:
            raise RuntimeError(
                "copy_(): expected compatible input and src shapes, but got: "
                f"{inp.shape} and {src.shape}"
            )
        for inp_comp, src_comp in zip(inp_comps, src_comps):
            inp_comp.copy_(src_comp)

    # AOTD allows mutations of inputs only, (not views of the inputs).
    # NJT.values() returns _values.detach() to workaround some issues.
    # To keep mutation in the graph, AOTD manually calls copy_ on the input (NJT).
    # Here we directly mutate self._values to not emit .detach() in the graph, which would make it non-compilable.
    inp._values.copy_(src._values)
    return inp


register_jagged_func(torch.ops.aten.detach.default, "self: jt_all")(
    jagged_unary_pointwise
)


@register_jagged_func(
    [
        torch.ops.aten.empty_like.default,
        torch.ops.aten.ones_like.default,
        torch.ops.aten.zeros_like.default,
        torch.ops.aten.rand_like.default,
        torch.ops.aten.randn_like.default,
    ],
    "self: jt_all",
)
def like_factory_default(func, *args, **kwargs):
    from torch._subclasses.fake_tensor import is_fake_tensor

    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    # Default layout is technically torch.strided but only jagged is supported here.
    # Rather than force users to specify the layout, assume jagged.
    # This should be set to strided for redispatching on values.
    new_kwargs["layout"] = torch.strided

    new_values = func(inp._values, **new_kwargs)
    new_offsets = inp._offsets.to(device=new_values.device)
    new_lengths = None
    if inp._lengths is not None:
        new_lengths = inp._lengths.to(device=new_values.device)
    output_kwargs = extract_kwargs(inp)
    if "offsets" in output_kwargs:
        output_kwargs["offsets"] = new_offsets
    if "lengths" in output_kwargs:
        output_kwargs["lengths"] = new_lengths

    if inp.device != new_values.device:
        # Update the nested int registry to indicate that the ragged structure is the same
        # between the two offsets / lengths on different devices.
        from torch._subclasses.functional_tensor import (
            FunctionalTensor,
            mb_unwrap_functional_tensor,
        )

        from .nested_tensor import _tensor_symint_registry

        ragged_source = inp._offsets if inp._lengths is None else inp._lengths
        new_thing = new_offsets if new_lengths is None else new_lengths
        if is_fake_tensor(new_thing) or isinstance(new_thing, FunctionalTensor):
            # Temporary hack until we have the union find
            tgt = mb_unwrap_functional_tensor(new_thing)
            src = mb_unwrap_functional_tensor(ragged_source)
            # pyrefly: ignore[missing-attribute]
            tgt.nested_int_memo = src.nested_int_memo
        else:
            _tensor_symint_registry[new_thing] = _tensor_symint_registry[ragged_source]

    return NestedTensor(new_values, **output_kwargs)


register_jagged_func(torch.ops.aten.full_like.default, "self: jt_all, fill_value: any")(
    like_factory_default
)

register_jagged_func(torch.ops.aten.randint_like.default, "self: jt_all, high: any")(
    like_factory_default
)

register_jagged_func(
    torch.ops.aten.randint_like.low_dtype, "self: jt_all, low: any, high: any"
)(like_factory_default)


@register_jagged_func(torch.ops.aten.zero_.default, "self: jt_all")
def zero__default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    func(inp._values)
    return inp


@register_jagged_func(
    torch.ops.aten._softmax.default, "self: jt_all, dim: any, half_to_float: any"
)
def _softmax_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    if isinstance(new_kwargs["dim"], tuple):
        raise RuntimeError(
            "softmax(): not supported for dimensions of type 'tuple' for NestedTensor"
        )

    inp = new_kwargs.pop("input")

    (
        new_kwargs["dim"],
        reduce_on_batch,
        reduce_on_ragged,
        _reduce_on_non_batch,
    ) = _wrap_jagged_dims(
        inp.dim(),
        (new_kwargs["dim"],),
        "softmax",
        inp._ragged_idx,
    )

    if reduce_on_batch:
        raise RuntimeError(
            "softmax(): not supported when reducing across the batch dimension for NestedTensor"
        )

    if reduce_on_ragged and inp._ragged_idx > 1:
        raise RuntimeError(
            "softmax(): not supported when reducing along the ragged dimension for ragged_idx > 1 for NestedTensor"
        )

    if reduce_on_ragged and inp._lengths is not None:
        raise RuntimeError(
            "softmax(): not supported where lengths is not None "
            + "if reducing across the ragged dimension for NestedTensor"
        )

    new_kwargs["dim"] = new_kwargs["dim"][
        0
    ]  # torch.softmax takes in the reduction dimension as an integer

    if reduce_on_ragged:
        padded_softmax_values = torch.nn.functional.softmax(
            torch.ops.aten._jagged_to_padded_dense_forward(
                inp._values.reshape(
                    inp._values.shape[0], -1
                ),  # values are required to be 2D tensors for j2pd
                [inp._offsets],
                max_lengths=[inp._max_seqlen],  # max length of ragged dimension
                padding_value=float("-inf"),  # e^-inf = 0
            ),
            dim=inp._ragged_idx,
        )

        softmax_values = torch.ops.aten._padded_dense_to_jagged_forward(
            padded_softmax_values,
            [inp._offsets],
            total_L=inp._values.shape[
                0
            ],  # providing this parameter helps avoid a GPU/CPU sync
        ).reshape(
            -1, *inp._values.shape[1:]
        )  # expand softmax_values back to original shape (inp._values.shape)

        return NestedTensor(softmax_values, **extract_kwargs(inp))

    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))


@register_jagged_func(
    torch.ops.aten._log_softmax.default, "self: jt_all, dim: any, half_to_float: any"
)
def _log_softmax_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    if isinstance(new_kwargs["dim"], tuple):
        raise RuntimeError(
            "log_softmax(): not supported for dimensions of type 'tuple' for NestedTensor"
        )

    inp = new_kwargs.pop("input")

    (
        new_kwargs["dim"],
        reduce_on_batch,
        reduce_on_ragged,
        _reduce_on_non_batch,
    ) = _wrap_jagged_dims(
        inp.dim(), (new_kwargs["dim"],), "log_softmax", inp._ragged_idx
    )

    if reduce_on_batch:
        raise RuntimeError(
            "log_softmax(): not supported when reducing across the batch dimension for NestedTensor"
        )

    if reduce_on_ragged:
        raise RuntimeError(
            "log_softmax(): not supported when reducing along the ragged dimension for NestedTensor"
        )

    # torch.log_softmax takes in the reduction dimension as an integer
    new_kwargs["dim"] = new_kwargs["dim"][0]

    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))


@register_jagged_func(
    torch.ops.aten._softmax_backward_data.default,
    "grad_output: jt, output: jt, dim: any, input_dtype: any",
)
def _softmax_backward(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )
    grad_out = new_kwargs.pop("grad_output")
    output = new_kwargs.pop("output")
    return NestedTensor(
        func(grad_out._values, output._values, **new_kwargs), **extract_kwargs(grad_out)
    )


@register_jagged_func(
    torch.ops.aten.native_dropout.default, "self: jt, float: any, train: any?"
)
def native_dropout_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    out1, out2 = func(inp._values, **new_kwargs)
    return (
        NestedTensor(out1, **extract_kwargs(inp)),
        NestedTensor(out2, **extract_kwargs(inp)),
    )


@register_jagged_func(
    torch.ops.aten.native_dropout_backward.default,
    "grad_output: jt, mask: jt, scale: any",
)
def native_dropout_backward_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )
    grad_output = new_kwargs.pop("grad_output")
    mask = new_kwargs.pop("mask")
    return NestedTensor(
        func(grad_output._values, mask._values, **new_kwargs),
        **extract_kwargs(grad_output),
    )


@register_jagged_func(
    torch.ops.aten.prod.dim_int,
    "self: jt_all, dim: any, keepdim: any?, dtype: any?",
)
def prod_dim_int(func, *args, **kwargs):
    return _apply_reduction(func, "prod", 1, *args, **kwargs)


@register_jagged_func(torch.ops.aten.prod.default, "self: jt_all, dtype: any?")
def prod_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    return func(inp._values, **new_kwargs)


@register_jagged_func(
    torch.ops.aten.split.Tensor, "self: jt, split_size: any, dim: any?"
)
def split_tensor(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    new_kwargs["dim"] = _wrap_jagged_dim(
        inp.dim(), new_kwargs["dim"], inp._ragged_idx, "split"
    )

    return tuple(
        NestedTensor(values=x, **extract_kwargs(inp))
        for x in func(inp._values, **new_kwargs)
    )


@register_jagged_func(
    torch.ops.aten.split_with_sizes.default, "self: jt, split_sizes: any, dim: any?"
)
def split_with_sizes_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    new_kwargs["dim"] = _wrap_jagged_dim(
        inp.dim(), new_kwargs["dim"], inp._ragged_idx, "split_with_sizes"
    )

    return [
        NestedTensor(values=x, **extract_kwargs(inp))
        for x in func(inp._values, **new_kwargs)
    ]


@register_jagged_func(
    torch.ops.aten.narrow.default, "self: jt, dim: any, start: any, length: any"
)
def narrow(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )
    inp = new_kwargs.pop("input")

    dim = _wrap_jagged_dim(inp.dim(), new_kwargs["dim"], inp._ragged_idx, "narrow")
    values = func(
        inp._values,
        dim=dim,
        start=new_kwargs["start"],
        length=new_kwargs["length"],
    )
    return NestedTensor(values, **extract_kwargs(inp))


@register_jagged_func(torch.ops.aten.chunk.default, "self: jt, chunks: any, dim: any?")
def chunk_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    new_kwargs["dim"], operating_on_batch = _wrap_jagged_dim(
        inp.dim(), new_kwargs["dim"], inp._ragged_idx, "chunk", allow_batch_dim=True
    )

    if operating_on_batch:
        chunks = new_kwargs["chunks"]

        # get _offsets of the chunks
        lengths = inp._offsets.diff()
        chunked_lengths = lengths.chunk(chunks)
        chunked_offsets = [torch.cumsum(x, dim=0) for x in chunked_lengths]
        chunked_offsets = [F.pad(x, (1, 0), value=0) for x in chunked_offsets]  # type: ignore[arg-type]
        nested_kwargs = [
            {"offsets": per_offsets, "_ragged_idx": inp._ragged_idx}
            for per_offsets in chunked_offsets
        ]

        # get _values of the chunks
        split_sizes = [x.sum().item() for x in chunked_lengths]
        chunk_values = inp._values.split(split_sizes)

        # Note that the actual number of chunks returned is not necessarily the same as
        # the input number; it can be counter-intuitive, but it matches dense behavior.
        return [
            NestedTensor(values=chunk_values[i], **(nested_kwargs[i]))
            for i in range(len(chunk_values))
        ]
    else:
        return [
            NestedTensor(values=x, **extract_kwargs(inp))
            for x in func(inp._values, **new_kwargs)
        ]


@register_jagged_func(torch.ops.aten.unbind.int, "self: jt_all, dim: any?")
def unbind_int(func, *args, **kwargs):
    # Note that this specializes on the length of the offsets
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    dim = new_kwargs["dim"]
    if dim != 0:
        raise RuntimeError("unbind(): only supported for NestedTensor on dim=0")

    inp = new_kwargs.pop("input")
    values = inp.values()
    offsets = inp.offsets()
    lengths = inp.lengths()
    ragged_idx = inp._ragged_idx

    def _torch_check(_lengths: list[int], _offsets: list[int] | None = None) -> None:
        # This torch._check are needed for torch.compile
        # symbolic shapes processing.
        # offsets and lengths are symbolic variables during compilation,
        # we guarantee the correct offsets/lengths correspondence:
        # sum of lengths <= total ragged_dim_size
        # every length and offset are size-like variable (allows sym shapes to reason it as [2, inf))
        # offset[i] + length[i] <= ragged_dim_size, for unbind and split dim correctness
        # offsets[i] <= ragged_dim_size

        lengths_sum = 0
        ragged_dim_size = values.shape[ragged_idx - 1]
        for i in range(len(_lengths)):
            torch._check(_lengths[i] >= 0)
            torch._check(_lengths[i] <= ragged_dim_size)

            lengths_sum += _lengths[i]
            if _offsets is not None:
                torch._check(
                    _offsets[i] + _lengths[i] <= ragged_dim_size,
                    lambda: "unbind(): nested tensor offsets and lengths do not match ragged_idx dimension",
                )
        torch._check(lengths_sum <= ragged_dim_size)

        if _offsets is not None:
            for i in range(len(_offsets)):
                torch._check(_offsets[i] >= 0)
                torch._check(_offsets[i] <= ragged_dim_size)

    if lengths is None:
        lengths_scalars = offsets.diff().tolist()
        _torch_check(lengths_scalars)

        return torch.split(values, lengths_scalars, dim=(ragged_idx - 1))

    if ragged_idx <= 0:
        raise RuntimeError(
            "unbind(): nested tensor ragged_idx out of bounds (should be >= 1)"
        )

    lengths_scalars = lengths.tolist()
    offsets_scalars = offsets.tolist()

    _torch_check(lengths_scalars, offsets_scalars)

    return [
        torch.narrow(
            values,
            dim=(ragged_idx - 1),
            start=offsets_scalars[i],
            length=lengths_scalars[i],
        )
        for i in range(lengths.shape[0])
    ]


@register_jagged_func(torch.ops.aten.squeeze.dim, "self: jt, dim: any")
def squeeze_dim(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    values = inp._values

    new_kwargs["dim"] = _wrap_jagged_dim(
        len(inp._size), new_kwargs["dim"], inp._ragged_idx, "squeeze"
    )
    return NestedTensor(func(values, **new_kwargs), **extract_kwargs(inp))


@register_jagged_func(torch.ops.aten.unsqueeze.default, "self: jt_all, dim: any")
def unsqueeze_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    values = inp._values

    # Account for collapsed jagged dim
    dim = new_kwargs["dim"]
    new_kwargs["dim"] = _wrap_jagged_dim(
        len(inp._size) + 1, dim, inp._ragged_idx, "unsqueeze", allow_ragged_dim=True
    )

    # ragged_idx changes if a dimension is added before it
    output_kwargs = extract_kwargs(inp)
    if new_kwargs["dim"] <= inp._ragged_idx - 1:
        output_kwargs["_ragged_idx"] += 1

    return NestedTensor(func(values, **new_kwargs), **output_kwargs)


@register_jagged_func(torch.ops.aten.cat.default, "tensors: any, dim: any?")
def cat_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    tensors = new_kwargs.pop("tensors")

    # Convert any non-nested to nested
    nested = [t for t in tensors if t.is_nested]
    if len(nested) == 0:
        raise AssertionError("At least one tensor must be nested")
    first = nested[0]
    tensors = [t if t.is_nested else t.expand_as(first) for t in tensors]

    # Account for collapsed jagged dim
    dim = new_kwargs["dim"]
    new_kwargs["dim"] = _wrap_jagged_dim(
        len(first.shape), dim, first._ragged_idx, "cat"
    )

    return NestedTensor(
        func([t._values for t in tensors], **new_kwargs), **extract_kwargs(tensors[0])
    )


@register_jagged_func(torch.ops.aten.matmul.default, "self: any, other: any")
def matmul_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    other = new_kwargs.pop("other")

    def _unbind_impl(a, b):
        return [
            func(a_comp, b_comp) for (a_comp, b_comp) in zip(a.unbind(), b.unbind())
        ]

    def _padded_impl(a, b):
        if a.is_nested:
            nt = a
        else:
            nt = b

        from .nested_tensor import nested_from_padded

        min_seqlen = nt._maybe_min_seqlen
        max_seqlen = nt._maybe_max_seqlen
        padded_max_S = max_seqlen
        total_L = nt._values.shape[nt._ragged_idx - 1]
        if padded_max_S is None:
            # use upper bound on max seqlen if it's not present
            padded_max_S = total_L

        padded_shape = (
            *nt.shape[: nt._ragged_idx],
            padded_max_S,
            *nt.shape[nt._ragged_idx + 1 :],
        )
        padded_nt = nt.to_padded_tensor(0.0, output_size=padded_shape)
        if a.is_nested:
            padded_t = func(padded_nt, b)
        else:
            padded_t = func(a, padded_nt)
        return nested_from_padded(
            padded_t,
            offsets=nt._offsets,
            ragged_idx=nt._ragged_idx,
            sum_S=total_L,
            min_seqlen=min_seqlen,
            max_seqlen=max_seqlen,
        )

    # TODO: Back these with proper kernels (e.g. grouped GEMM)
    # NJT x dense
    if inp.is_nested and not other.is_nested:
        # (B, j1, D) x (B, D, E) => (B, j1, E)
        if (
            inp.dim() >= 3
            and inp.dim() == other.dim()
            and inp._ragged_idx < inp.dim() - 1
        ):
            # convert to padded for this
            return _padded_impl(inp, other)
        # Support broadcasting the dense:
        # (B, j1, D) x (D, E) => (B, j1, E)
        # (B, j1, D, E) x (E, F) => (B, j1, D, F)
        # etc.
        elif (
            other.dim() == 2
            and inp.dim() > other.dim()
            and inp._ragged_idx < inp.dim() - 1
        ):
            return NestedTensor(
                func(inp._values, other, **new_kwargs), **extract_kwargs(inp)
            )
    # Dense x NJT
    elif not inp.is_nested and other.is_nested:
        # (B, D, E) x (B, E, j1) => (B, E, j1)
        if other.dim() >= 3 and other.dim() == inp.dim() and other._ragged_idx >= 2:
            # convert to padded for this
            return _padded_impl(inp, other)
        # Support broadcasting the dense:
        # (D, E) x (B, E, j1) => (B, D, j1)
        # (D, E) x (B, E, j1, F) => (B, D, j1, F)
        # etc.
        elif inp.dim() == 2 and other.dim() > inp.dim() and other._ragged_idx >= 2:
            return NestedTensor(
                func(inp, other._values, **new_kwargs), **extract_kwargs(other)
            )

    # NJT x NJT
    elif inp.is_nested and other.is_nested:
        # Support ragged batch dim:
        # (B, j1, D, E) x (B, j1, E, F) => (B, j1, D, F), etc.
        if inp.dim() > 3 and other.dim() > 3 and raggedness_matches(inp, other._size):
            return NestedTensor(func(inp._values, other._values), **extract_kwargs(inp))
        # Support reducing over ragged with dense output:
        # (B, D, j1) x (B, j1, E) => (B, D, E)
        elif (
            inp.dim() == 3
            and other.dim() == 3
            and inp._ragged_idx == 2
            and other._ragged_idx == 1
            and inp.size(inp._ragged_idx) == other.size(other._ragged_idx)
        ):
            # do unbind for this; can't use padded conversion due to j1 in last dim
            return torch.stack(_unbind_impl(inp, other))

    raise RuntimeError(
        f"matmul(): not supported between inputs of shapes {inp._size} and {other.shape}"
    )


@register_jagged_func(torch.ops.aten.bmm.default, "self: jt_all, mat2: any")
def bmm_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    other = new_kwargs.pop("mat2")

    if inp.dim() != 3:
        raise ValueError("bmm(): input must be 3D")
    if other.dim() != 3:
        raise ValueError("bmm(): mat2 must be 3D")

    return matmul_default(torch.ops.aten.matmul.default, inp, other)


@register_jagged_func(
    torch.ops.aten.expand.default, "self: jt_all, size: any, implicit: any?"
)
def expand_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    size = new_kwargs["size"]

    if "implicit" in new_kwargs and new_kwargs.pop("implicit"):
        raise AssertionError("implicit expand is not supported")
    if not raggedness_matches(inp, size):
        raise RuntimeError(f"expand(): cannot expand shape {inp._size} -> {size}")

    expand_arg = [-1 if d == inp._ragged_idx else size[d] for d in range(1, inp.dim())]
    return NestedTensor(func(inp._values, expand_arg), **extract_kwargs(inp))


@register_jagged_func(torch.ops.aten.expand_as.default, "self: t, other: jt")
def expand_as_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    other = new_kwargs.pop("other")

    return NestedTensor(func(inp, other._values), **extract_kwargs(other))


@register_jagged_func(torch.ops.aten.broadcast_to.default, "self: jt_all, size: any")
def broadcast_to(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    size = new_kwargs.pop("size")

    if len(size) <= inp.dim():
        return inp.expand([*(1 for _ in range(inp.dim() - len(size))), *size])

    raise ValueError(
        "broadcast_to(): broadcasting to a higher-dim shape is currently not supported "
        "for nested tensors with the jagged layout"
    )


@register_jagged_func(torch.ops.aten.broadcast_tensors.default, "tensors: any")
def broadcast_tensors(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    tensors = new_kwargs.pop("tensors")
    if len(tensors) == 0:
        raise ValueError("broadcast_tensors(): expected at least one tensor input")
    if len(tensors) == 1:
        return tensors[0]

    outs = []
    broadcast_shape = torch.broadcast_shapes(*(t.shape for t in tensors))
    # Pull out the first NJT. If broadcast_shapes() worked, the nested ints are compatible.
    njt = next(t for t in tensors if isinstance(t, NestedTensor))
    for t in tensors:
        if t.is_nested:
            outs.append(t.broadcast_to(broadcast_shape))
        elif t.dim() < len(broadcast_shape):
            outs.append(
                NestedTensor(t.broadcast_to(njt._values.shape), **extract_kwargs(njt))
            )
        else:
            raise ValueError(
                "broadcast_tensors(): broadcasting nested tensors with dense tensors of equal "
                "or higher dim is not currently supported"
            )

    return tuple(outs)


@register_jagged_func(
    torch.ops.aten.where.self, "condition: jt_all, self: any, other: any"
)
def where_self(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    condition = new_kwargs.pop("condition")
    inp = new_kwargs.pop("input")
    other = new_kwargs.pop("other")

    # if the tensors aren't compatible, broadcast_tensors() will let us know
    condition, inp, other = torch.broadcast_tensors(condition, inp, other)

    return NestedTensor(
        func(condition._values, inp._values, other._values, **new_kwargs),
        **extract_kwargs(condition),
    )


@register_jagged_func(torch.ops.aten._pin_memory.default, "self: jt, device: any?")
def _pin_memory_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))


@register_jagged_func(torch.ops.aten.is_pinned.default, "self: jt, device: any?")
def is_pinned_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    return func(inp._values, **new_kwargs)


@register_jagged_func(
    torch.ops.aten.is_same_size.default, "self: jt_all, other: jt_all"
)
def is_same_size_default(func, *args, **kwargs):
    return args[0]._size == args[1]._size


def _apply_reduction(func, func_name, identity_element, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    # some ops use dim=None to indicate a full reduction; some use an empty dim list
    full_reduction = new_kwargs["dim"] is None or (
        isinstance(new_kwargs["dim"], (tuple, list)) and len(new_kwargs["dim"]) == 0
    )
    if full_reduction:
        out = func(inp._values, **new_kwargs)
        if new_kwargs.get("keepdim", False):
            if isinstance(out, (tuple, list)):
                # some ops return multiple things; unsqueeze all of them
                out = type(out)(o.unsqueeze(inp._ragged_idx) for o in out)
            else:
                out = out.unsqueeze(inp._ragged_idx)
        return out

    # some ops support lists of dims; some don't
    dim_to_convert = new_kwargs["dim"]
    is_dimlist = isinstance(new_kwargs["dim"], (tuple, list))
    if not is_dimlist:
        dim_to_convert = [dim_to_convert]

    (
        converted_dim,
        reduce_on_batch,
        reduce_on_ragged,
        reduce_on_non_batch,
    ) = _wrap_jagged_dims(
        inp.dim(),
        dim_to_convert,
        f"{func_name}",
        inp._ragged_idx,
    )

    if not is_dimlist:
        # convert back from list
        converted_dim = converted_dim[0]
    new_kwargs["dim"] = converted_dim

    if reduce_on_ragged and inp._lengths is not None:
        raise RuntimeError(
            f"{func_name}(): reducing across the ragged dimension is not supported "
            "for non-contiguous nested tensors with holes"
        )

    from torch.utils._pytree import tree_map

    # raggedness reduced away --> return dense tensor
    if reduce_on_ragged:
        # reduction cases: (batch, ragged), (batch, ragged, non-batch), etc.
        if reduce_on_batch:
            # no need to read offsets --> apply sum directly on values
            out = func(inp._values, **new_kwargs)
            if new_kwargs.get("keepdim", False):
                # some ops return multiple things; unsqueeze all of them
                out = tree_map(lambda o: o.unsqueeze(0), out)
            return out
        else:
            # invalid reduction cases: (ragged, non-batch), etc.
            if reduce_on_non_batch:
                raise RuntimeError(
                    f"{func_name}(): reducing along a ragged and non-batch dimension "
                    "is not supported for nested tensors"
                )

            # reduction cases: (ragged)
            # convert to padded dense and reduce
            new_kwargs.pop("dim")
            dim_to_pass = [inp._ragged_idx] if is_dimlist else inp._ragged_idx
            return func(
                inp.to_padded_tensor(identity_element), dim=dim_to_pass, **new_kwargs
            )
    # raggedness preserved --> return nested tensor
    else:
        # invalid reduction cases: (batch), (batch, non-batch), etc.
        if reduce_on_batch:
            raise RuntimeError(
                f"{func_name}(): reducing along the batch dimension but not "
                "the ragged dimension is not supported for nested tensors"
            )

        # reduction cases: (non-batch), (non-batch, non-batch), etc.
        # apply sum directly on values
        out = func(inp._values, **new_kwargs)
        out_kwargs = extract_kwargs(inp)
        if not new_kwargs.get("keepdim", False):
            # dims are reduced away -> ragged_idx of output needs to be reevaluated
            dimlist = (
                new_kwargs["dim"]
                if isinstance(new_kwargs["dim"], (tuple, list))
                else [new_kwargs["dim"]]
            )
            for d in dimlist:
                # adjust for all dims reduced before the ragged dim
                if d < inp._ragged_idx - 1:
                    out_kwargs["_ragged_idx"] -= 1

        # some ops return multiple things; wrap each of them as an NJT
        return tree_map(lambda o: NestedTensor(o, **out_kwargs), out)


@register_jagged_func(torch.ops.aten.sum.default, "self: jt_all, dtype: any?")
def sum_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    return func(inp._values, **new_kwargs)


@register_jagged_func(
    torch.ops.aten.sum.dim_IntList,
    "self: jt_all, dim: any?, keepdim: any?, dtype: any?",
)
def sum_dim_IntList(func, *args, **kwargs):
    return _apply_reduction(func, "sum", 0, *args, **kwargs)


@register_jagged_func(
    torch.ops.aten.transpose.int, "self: jt_all, dim0: any, dim1: any"
)
def transpose_int(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    from torch._prims_common import canonicalize_dims

    inp = new_kwargs.pop("input")
    dim0, dim1 = canonicalize_dims(inp.dim(), (new_kwargs["dim0"], new_kwargs["dim1"]))

    # To support the SDPA API, inputs need to have the ragged idx transposed to dim 2
    # instead of 1, although the internal Flash and mem-effn implementations will
    # use the inputs with raggedness in dim 1.
    if dim0 == inp._ragged_idx or dim1 == inp._ragged_idx:
        if dim0 == 0 or dim1 == 0:
            raise ValueError(
                "Transpose is not supported on the batch dimension for jagged NT"
            )
        if dim0 == inp._ragged_idx:
            to_dim = dim1
        else:
            to_dim = dim0
        inp_kwargs = extract_kwargs(inp)
        inp_kwargs["_ragged_idx"] = to_dim
        return NestedTensor(
            inp.values().transpose(
                _outer_to_inner_dim(len(inp._size), dim0, inp._ragged_idx),
                _outer_to_inner_dim(len(inp._size), dim1, inp._ragged_idx),
            ),
            **inp_kwargs,
        )

    new_kwargs["dim0"] = _wrap_jagged_dim(
        inp.dim(), new_kwargs["dim0"], inp._ragged_idx, "transpose"
    )
    new_kwargs["dim1"] = _wrap_jagged_dim(
        inp.dim(), new_kwargs["dim1"], inp._ragged_idx, "transpose"
    )

    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))


@register_jagged_func(torch.ops.aten.permute.default, "self: jt_all, dims: any")
def permute_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )
    inp = new_kwargs.pop("input")
    dims = new_kwargs.pop("dims")
    inp_kwargs = extract_kwargs(inp)
    inp_dim = len(inp._size)

    # The first two checks are the same as the checks in the normal permute implementation
    if inp_dim != len(dims):
        raise ValueError(
            f"permute(): number of dimensions in the tensor input ({inp_dim}) "
            + f"does not match the length of the desired ordering of dimensions ({len(dims)}).",
        )

    from torch._prims_common import canonicalize_dims

    canonicalized_dims = canonicalize_dims(inp_dim, dims)

    if len(canonicalized_dims) != len(set(canonicalized_dims)):
        raise ValueError("permute(): duplicate dims are not allowed.")

    if inp._lengths is not None:
        raise ValueError(
            "permute(): not supported on jagged layout nested tensor with holes"
        )
    if canonicalized_dims[0] != 0:
        raise ValueError(
            "Permute is not supported on the batch dimension for jagged NT"
        )
    inp_kwargs["_ragged_idx"] = canonicalized_dims.index(inp._ragged_idx)
    inner_dims = [
        _outer_to_inner_dim(inp_dim, dim, inp._ragged_idx)
        for dim in canonicalized_dims[1:]
    ]
    new_kwargs["dims"] = inner_dims
    return NestedTensor(func(inp._values, **new_kwargs), **inp_kwargs)


@register_jagged_func(
    [torch.ops.aten.view.default, torch.ops.aten._unsafe_view.default],
    "self: jt_all, size: any",
)
def view_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    size = new_kwargs.pop("size")

    if inp._ragged_idx != 1 and tuple(inp._size) != tuple(size):
        raise RuntimeError(
            f"view(): does not support ragged_idx != 1 except when inp._size == size. "
            f"inp._size is ({inp._size}) and size is ({size})."
        )

    # Ensure specified size still includes batch and ragged dims
    if len(size) < 3 or not raggedness_matches(inp, size):
        raise RuntimeError(f"view(): cannot view shape {inp._size} as {size}")

    # outer size: the size of the NT, e.g. [3, j0, 10]
    # inner size: the size of the values, e.g. [8, 10] (e.g. for offsets = [0, 3, 5, 8])
    # this function gets inner_size[inner_idx] for a given inner_idx.
    #
    # example: for outer size [a, b, c, j0, d, e, f]
    #                         assume that j0 is ragged, other are concrete integers
    #                         and ragged_idx=3
    # inner size will be      [b, c, inp._values.size(ragged_idx), d, e, f]
    # therefore:
    #    inner_size[0] = outer_size[1]
    #    inner_size[1] = outer_size[2]
    #    inner_size[0] = inp._values.size(ragged_idx - 1)
    #    inner_size[3] = outer_size[4]
    #    inner_size[4] = outer_size[5]
    def get_inner_size(inner_idx):
        nonlocal inp, size
        if inner_idx == inp._ragged_idx - 1:
            return inp._values.size(inner_idx)
        else:
            return size[inner_idx + 1]

    inner_size = [get_inner_size(i) for i in range(len(size) - 1)]

    # Preserve inference-mode-ness of input.
    # TODO: Do this for all other views!
    with torch.inference_mode(inp.is_inference()):
        return NestedTensor(func(inp._values, inner_size), **extract_kwargs(inp))


@register_jagged_func(
    torch.ops.aten.native_layer_norm.default,
    "input: jt_all, normalized_shape: any, weight: any?, bias: any?, eps: any",
)
def native_layer_norm_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    if inp.dim() <= 2:
        raise RuntimeError(
            "layer_norm(): not supported for NestedTensor objects with 2 or fewer dimensions"
        )

    normalized_shape = new_kwargs["normalized_shape"]
    ragged_size = inp.shape[inp._ragged_idx]

    num_dims_not_normalized = inp.dim() - len(normalized_shape)

    if (
        num_dims_not_normalized == 0
    ):  # error if trying to normalize over the batch dimension
        raise RuntimeError(
            "layer_norm(): not supported when normalizing over the batch dimension for NestedTensor"
        )

    if ragged_size in normalized_shape and inp._lengths is not None:
        raise RuntimeError(
            "layer_norm(): not supported where lengths is not None if operating on the ragged dimension for NestedTensor"
        )

    if (
        ragged_size in normalized_shape
    ):  # special handling for normalizing over the ragged dimension
        padded_input = torch.ops.aten._jagged_to_padded_dense_forward(
            inp._values.flatten(
                start_dim=inp._ragged_idx
            ),  # _jagged_to_padded_dense_forward requires values to be a 2D tensor
            [inp._offsets],
            max_lengths=[inp._max_seqlen],  # max length of ragged dimension
        )

        padded_mask = torch.ops.aten._jagged_to_padded_dense_forward(
            torch.ones((inp._values.shape[0], 1), device=inp.device, dtype=inp.dtype),
            [inp._offsets],
            max_lengths=[inp._max_seqlen],  # max length of ragged dimension
        ).expand(
            padded_input.shape
        )  # mask elements outside of the ragged dimension and expand to the same shape as padded input (3D dense tensor)

        ragged_lengths = (
            inp._offsets.diff().unsqueeze(1).unsqueeze(1) * padded_input.shape[2]
        )  # ragged dim * inner dim, since we sum over dims (1, 2) (the layer on which we normalize)

        mean = (
            torch.sum(
                padded_input,
                dim=(1, 2),
                keepdim=True,
            )
            / ragged_lengths
        )  # a sum over (1, 2) ensures layer norm, whereas a sum over (1) would be an instance norm

        padded_normalized = (
            (padded_input - mean) * padded_mask
        )  # mask elements outside of the ragged dimension size for correct variance calculation

        variance = (
            torch.sum(
                torch.square(padded_normalized),
                dim=(1, 2),
                keepdim=True,
            )
            / ragged_lengths
        )  # a sum over (1, 2) ensures layer norm, whereas a sum over (1) would be an instance norm

        std = torch.sqrt(variance + new_kwargs["eps"])
        padded_layer_norm = padded_normalized / std

        jagged_layer_norm_values = torch.ops.aten._padded_dense_to_jagged_forward(
            padded_layer_norm,
            [inp._offsets],
            total_L=inp._values.shape[
                0
            ],  # providing this parameter helps avoid a GPU/CPU sync
        ).unflatten(
            -1, inp.shape[inp._ragged_idx + 1 :]
        )  # unflatten last dimension back into original nested tensor shape, e.g. (B, *, WH) --> (B, *, W, H)

        return (
            NestedTensor(jagged_layer_norm_values, **extract_kwargs(inp)),
            mean,
            std,
        )

    output, mean, std = func(inp._values, **new_kwargs)
    return (NestedTensor(output, **extract_kwargs(inp)), mean, std)


@register_jagged_func(
    torch.ops.aten.native_layer_norm_backward.default,
    "grad_out: jt, input: jt, normalized_shape: any, mean: any, rstd: any, weight: any?, bias: any?, output_mask: any",
)
def native_layer_norm_backward_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )
    grad_out = new_kwargs.pop("grad_out")
    inp = new_kwargs.pop("input")
    d_input, d_gamma, d_beta = func(grad_out._values, inp._values, **new_kwargs)
    if d_input is None:
        return (None, d_gamma, d_beta)

    return (NestedTensor(d_input, **extract_kwargs(inp)), d_gamma, d_beta)


@register_jagged_func(torch.ops.aten.select.int, "self: jt_all, dim: any, index: any")
def select_int(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    new_kwargs["dim"], operating_on_batch = _wrap_jagged_dim(
        inp.dim(), new_kwargs["dim"], inp._ragged_idx, "select", allow_batch_dim=True
    )

    # handle batch dim slicing via unbind() for now
    # TODO: make this more efficient
    if operating_on_batch:
        return inp.unbind()[new_kwargs["index"]]

    if inp._lengths is not None:
        raise ValueError(
            "select(): not yet supported on dim != 0 for non-contiguous nested tensor with holes"
        )

    # if selecting before the ragged dim, adjust output ragged_idx
    out_kwargs = extract_kwargs(inp)
    if new_kwargs["dim"] < inp._ragged_idx - 1:
        out_kwargs["_ragged_idx"] -= 1

    return NestedTensor(func(inp._values, **new_kwargs), **out_kwargs)


@register_jagged_func(
    torch.ops.aten.slice.Tensor,
    "self: jt, dim: any?, start: any?, end: any?, step: any?",
)
def slice_tensor(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    new_kwargs["dim"] = _wrap_jagged_dim(
        inp.dim(), new_kwargs["dim"], inp._ragged_idx, "slice"
    )

    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))


@register_jagged_func(
    torch.ops.aten.index_put.default,
    "input: jt_all, indices: any, values: t, accumulate: any?",
)
@register_jagged_func(
    torch.ops.aten.index_put_.default,
    "input: jt_all, indices: any, values: t, accumulate: any?",
)
def index_put_(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp: NestedTensor = new_kwargs.pop("input")

    # For index_put_ to work, we add together the indices of the ragged dimension
    # and the batch dimension, adding the offsets of each ragged dimension to its
    # indices

    indices = new_kwargs.pop("indices")

    if len(indices) > inp.dim():
        raise AssertionError(
            f"Too many indices: got {len(indices)} but tensor has {inp.dim()} dimensions"
        )

    if len(indices) < inp._ragged_idx + 1:
        if not inp.is_contiguous():
            raise RuntimeError(
                "index_put(): If ragged dimension is not part of indices, this only works on contiguous NJTs"
            )
        # Ragged dim is NOT part of indices, we need to pad the nested tensor to apply func
        from .nested_tensor import nested_from_padded

        min_seqlen = inp._maybe_min_seqlen
        max_seqlen = inp._maybe_max_seqlen
        padded_max_S = max_seqlen
        total_L = inp._values.shape[inp._ragged_idx - 1]
        if padded_max_S is None:
            # use upper bound on max seqlen if it's not present
            padded_max_S = total_L

        padded_shape = (
            *inp.shape[: inp._ragged_idx],
            padded_max_S,
            *inp.shape[inp._ragged_idx + 1 :],
        )
        padded_inp = inp.to_padded_tensor(0.0, output_size=padded_shape)
        new_njt = nested_from_padded(
            func(padded_inp, indices, **new_kwargs),
            offsets=inp._offsets,
            ragged_idx=inp._ragged_idx,
            sum_S=total_L,
            min_seqlen=min_seqlen,
            max_seqlen=max_seqlen,
        )

        if func is torch.ops.aten.index_put_.default:
            inp._values.copy_(new_njt.values())
            return inp
        return new_njt

    # We can run on the underlying values directly

    # Validate indices
    if inp.lengths() is None:
        lengths = inp.offsets().diff()
    else:
        lengths = inp.lengths()
    torch._assert_async(
        # pyrefly: ignore [no-matching-overload]
        torch.all(indices[inp._ragged_idx] < lengths),
        "Some indices in the ragged dimension are out of bounds!",
    )

    # Recompute indices for _values
    ragged_indices = inp.offsets()[indices[0]] + indices[inp._ragged_idx]
    func_indices = (
        # before ragged dim
        indices[1 : inp._ragged_idx]
        # ragged dim (combined with batch)
        + [ragged_indices]
        # after ragged dim
        + indices[inp._ragged_idx + 1 :]
    )

    if func is torch.ops.aten.index_put_.default:
        inp._values = func(inp._values, func_indices, **new_kwargs)
        return inp

    return NestedTensor(
        func(inp._values, func_indices, **new_kwargs),
        **extract_kwargs(inp),
    )


@register_jagged_func(
    torch.ops.aten.convolution.default,
    "input: jt, weight: t, bias: t?, stride: any, padding: any, "
    "dilation: any, transposed: any, output_padding: any, groups: any",
)
def convolution_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))


@register_jagged_func(
    torch.ops.aten.mean.dim, "self: jt_all, dim: any?, keepdim: any?, dtype: any?"
)
def mean_dim(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs["input"]
    (_, reduce_on_batch, reduce_on_ragged, reduce_on_non_batch) = _wrap_jagged_dims(
        inp.dim(),
        new_kwargs["dim"],
        "mean",
        inp._ragged_idx,
    )

    if reduce_on_ragged and not reduce_on_batch:
        if reduce_on_non_batch:
            raise AssertionError(
                "Cannot reduce on both ragged and non-batch dimensions without also reducing on batch"
            )
        # calculate an intermediate sum and leave the dim in for normalization purposes
        keepdim = new_kwargs["keepdim"]
        new_kwargs["keepdim"] = True
        intermediate_sum = _apply_reduction(
            torch.ops.aten.sum.dim_IntList, "mean", 0, **new_kwargs
        )

        # normalize by sequence lengths
        lengths = inp._lengths if inp._lengths is not None else inp._offsets.diff()
        for _ in range(intermediate_sum.dim() - 1):
            lengths = lengths.unsqueeze(-1)
        out = intermediate_sum / lengths
        if not keepdim:
            out = out.squeeze(inp._ragged_idx)
        return out

    # at this point, we're just redispatching on the values buffer
    # since we expect it to be unused, specify a weird intermediate value to
    # hopefully make errors obvious
    intermediate_value = 0.42
    return _apply_reduction(func, "mean", intermediate_value, **new_kwargs)


@register_jagged_func(torch.ops.aten.mean.default, "self: jt_all, dtype: any?")
def mean_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    return func(inp._values, **new_kwargs)


@register_jagged_func(torch.ops.aten.any.dims, "self: jt_all, dim: any?, keepdim: any?")
def any_dims(func, *args, **kwargs):
    return _apply_reduction(func, "any", False, *args, **kwargs)


@register_jagged_func(torch.ops.aten.any.dim, "self: jt_all, dim: any, keepdim: any?")
def any_dim(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    # wrap dim in list to redispatch to dims overload
    new_kwargs["dim"] = [new_kwargs["dim"]]
    return any_dims(torch.ops.aten.any.dims, **new_kwargs)


@register_jagged_func(torch.ops.aten.all.dims, "self: jt_all, dim: any?, keepdim: any?")
def all_dims(func, *args, **kwargs):
    return _apply_reduction(func, "all", True, *args, **kwargs)


@register_jagged_func(torch.ops.aten.all.dim, "self: jt_all, dim: any, keepdim: any?")
def all_dim(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    # wrap dim in list to redispatch to dims overload
    new_kwargs["dim"] = [new_kwargs["dim"]]
    return all_dims(torch.ops.aten.all.dims, **new_kwargs)


@register_jagged_func(
    [
        torch.ops.aten.all.default,
        torch.ops.aten.any.default,
        torch.ops.aten.max.default,
        torch.ops.aten.min.default,
    ],
    "self: jt_all",
)
def all_any_max_min_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    return func(inp._values, **new_kwargs)


@register_jagged_func(
    [torch.ops.aten._is_all_true.default, torch.ops.aten._is_any_true.default],
    "self: jt_all",
)
def _is_true_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    return func(inp._values)


@register_jagged_func(torch.ops.aten.min.dim, "self: jt_all, dim: any, keepdim: any?")
def min_dim(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    dtype = new_kwargs["input"].dtype
    dtype_max = _get_padding_value(dtype, "max")
    return _apply_reduction(func, "min", dtype_max, *args, **kwargs)


@register_jagged_func(torch.ops.aten.max.dim, "self: jt_all, dim: any, keepdim: any?")
def max_dim(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    dtype = new_kwargs["input"].dtype
    dtype_min = _get_padding_value(dtype, "min")
    return _apply_reduction(func, "max", dtype_min, *args, **kwargs)


@register_jagged_func(
    torch.ops.aten.amin.default, "self: jt_all, dim: any?, keepdim: any?"
)
def amin_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    dtype = new_kwargs["input"].dtype
    dtype_max = _get_padding_value(dtype, "max")
    return _apply_reduction(func, "amin", dtype_max, *args, **kwargs)


@register_jagged_func(
    torch.ops.aten.amax.default, "self: jt_all, dim: any?, keepdim: any?"
)
def amax_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    dtype = new_kwargs["input"].dtype
    dtype_min = _get_padding_value(dtype, "min")
    return _apply_reduction(func, "amax", dtype_min, *args, **kwargs)


@register_jagged_func(
    torch.ops.aten.argmin.default, "self: jt_all, dim: any?, keepdim: any?"
)
def argmin_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    dtype = new_kwargs["input"].dtype
    dtype_max = _get_padding_value(dtype, "max")
    return _apply_reduction(func, "argmin", dtype_max, *args, **kwargs)


@register_jagged_func(
    torch.ops.aten.argmax.default, "self: jt_all, dim: any?, keepdim: any?"
)
def argmax_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    dtype = new_kwargs["input"].dtype
    dtype_min = _get_padding_value(dtype, "min")
    return _apply_reduction(func, "argmax", dtype_min, *args, **kwargs)


@register_jagged_func(
    torch.ops.aten.value_selecting_reduction_backward.default,
    "grad: jt_all, dim: any, indices: jt_all, sizes: any, keepdim: any",
)
def value_selecting_reduction_backward_default(func, *args, **kwargs):
    from torch.fx.experimental.symbolic_shapes import is_nested_int

    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    grad = new_kwargs.pop("grad")
    new_kwargs["grad"] = grad._values
    indices = new_kwargs.pop("indices")
    new_kwargs["indices"] = indices._values
    # should always succeed; sizes should contain a nested int
    ragged_idx = next(i for i, s in enumerate(new_kwargs["sizes"]) if is_nested_int(s))
    # convert dim -> values-space dim
    new_kwargs["dim"] = _wrap_jagged_dim(
        len(new_kwargs["sizes"]),
        new_kwargs["dim"],
        ragged_idx,
        "value_selecting_reduction_backward",
    )
    # convert saved NJT sizes -> values-space sizes
    sizes = new_kwargs.pop("sizes")
    sizes[ragged_idx] = indices._values.size(indices._ragged_idx - 1)
    sizes = sizes[1:]
    new_kwargs["sizes"] = sizes

    output_kwargs = extract_kwargs(indices)
    output_kwargs["_ragged_idx"] = ragged_idx

    return NestedTensor(func(**new_kwargs), **output_kwargs)


@register_jagged_func(torch.ops.aten.stack.default, "tensors: any, dim: any?")
def stack_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    # guaranteed this is non-empty if we got here
    tensors = new_kwargs.pop("tensors")
    for t in tensors:
        if not isinstance(t, NestedTensor):
            raise RuntimeError("stack(): expected all nested tensors inputs")

        if t.dim() != tensors[0].dim():
            raise RuntimeError(
                "stack(): expected all nested tensors to have the same dim"
            )

        if not raggedness_matches(t, tensors[0].shape):
            raise RuntimeError(
                "stack(): expected all nested tensors to have the same nested structure"
            )

    new_kwargs["dim"] = _wrap_jagged_dim(
        tensors[0].dim() + 1, new_kwargs["dim"], tensors[0]._ragged_idx, "stack"
    )

    return NestedTensor(
        func([t._values for t in tensors], **new_kwargs), **extract_kwargs(tensors[0])
    )


@register_jagged_func(
    torch.ops.aten.embedding.default,
    "weight: t, indices: jt, padding_idx: any?, scale_grad_by_freq: any?, sparse: any?",
)
def embedding_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    # guaranteed this is non-empty if we got here
    indices = new_kwargs.pop("indices")
    weight = new_kwargs.pop("weight")

    return NestedTensor(
        func(weight, indices._values, **new_kwargs), **extract_kwargs(indices)
    )


@register_jagged_func(
    torch.ops.aten.embedding_dense_backward.default,
    "grad_output: jt, indices: jt, num_weights: any, padding_idx: any, scale_grad_by_freq: any",
)
def embedding_dense_backward_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    indices = new_kwargs.pop("indices")
    grad_output = new_kwargs.pop("grad_output")
    return func(grad_output._values, indices._values, **new_kwargs)


@register_jagged_func(
    [
        torch.ops.aten.values.default,
        torch.ops.aten._nested_get_values.default,
    ],
    "self: jt_all",
)
def values_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    # TODO: Handle inference mode properly.
    # See https://github.com/pytorch/pytorch/issues/112024#issuecomment-1779554292
    return inp._values.detach()


@register_jagged_func(torch.ops.aten.all.default, "self: jt_all")
def all_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    return func(inp._values)


@register_jagged_func(
    torch.ops.aten.to_padded_tensor.default,
    "self: jt_all, padding: any, output_size: any?",
)
def to_padded_tensor_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    if inp._lengths is not None:
        raise RuntimeError(
            "to_padded_tensor(): not supported for nested tensors with holes"
        )

    # TODO: Handle the rest of output_size
    output_size = new_kwargs["output_size"]
    if output_size is not None:
        max_seq_len = output_size[inp._ragged_idx]
    else:
        max_seq_len = (
            inp._max_seqlen
            if inp._max_seqlen_tensor is not None
            else inp._values.size(0)
        )

    # only 2D values with ragged packed dim=0 is supported by the underlying FBGEMM
    # kernel so do shape gymnastics if needed
    values = inp.values()
    if inp._ragged_idx > 1:
        values = values.transpose(inp._ragged_idx - 1, 0)
    values_shape = values.shape
    if values.dim() > 2:
        values = values.flatten(start_dim=1)
    elif values.dim() == 1:
        values = values.unsqueeze(-1)

    # NB: The CUDA kernel for jagged -> padded dense conversion does not support
    # integer / bool types; work around this by casting to half.
    is_bool = values.dtype is torch.bool
    if is_bool and values.is_cuda:
        values = values.to(torch.half)
    padded_out = torch.ops.aten._jagged_to_padded_dense_forward(
        values,
        [inp._offsets],
        [max_seq_len],
        new_kwargs["padding"],
    )
    if is_bool and padded_out.is_cuda:
        padded_out = padded_out.to(torch.bool)

    # shape gymnastics part 2
    if len(values_shape) > 2:
        padded_out = padded_out.unflatten(-1, values_shape[1:])
    elif len(values_shape) == 1:
        padded_out = padded_out.squeeze(-1)
    if inp._ragged_idx > 1:
        padded_out = padded_out.transpose(inp._ragged_idx, 1)

    return padded_out


@register_jagged_func(
    torch.ops.aten._nested_from_padded_tensor.default,
    "padded: t, offsets: t, dummy: jt, ragged_idx: any?, min_seqlen: any?, max_seqlen: any?, sum_S: any?",
)
def _nested_from_padded_tensor_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    padded, offsets = new_kwargs["padded"], new_kwargs["offsets"]
    ragged_idx = new_kwargs.get("ragged_idx", 1)

    # only 3D padded with ragged packed dim=0 is supported by the underlying FBGEMM
    # kernel so do shape gymnastics
    if ragged_idx > 1:
        padded = padded.transpose(ragged_idx, 1)
    padded_ragged_dim1_shape = padded.shape
    if padded.dim() > 3:
        padded = padded.flatten(start_dim=2)
    elif padded.dim() < 3:
        padded = padded.unsqueeze(-1)

    # NB: The CUDA kernel for padded dense -> jagged conversion does not support
    # integer / bool types; work around this by casting to half.
    is_bool = padded.dtype is torch.bool
    if is_bool and padded.is_cuda:
        padded = padded.to(torch.half)
    values = torch.ops.aten._padded_dense_to_jagged_forward(
        padded, [offsets], new_kwargs["sum_S"]
    )
    if is_bool and values.is_cuda:
        values = values.to(torch.bool)

    # shape gymnastics part 2
    if len(padded_ragged_dim1_shape) > 3:
        values = values.unflatten(-1, padded_ragged_dim1_shape[2:])
    elif len(padded_ragged_dim1_shape) < 3:
        values = values.squeeze(-1)
    if ragged_idx > 1:
        values = values.transpose(ragged_idx - 1, 0)

    min_seqlen = new_kwargs["min_seqlen"]
    max_seqlen = new_kwargs["max_seqlen"]
    metadata_cache = {}
    if min_seqlen is not None:
        metadata_cache["min_seqlen"] = min_seqlen
    if max_seqlen is not None:
        metadata_cache["max_seqlen"] = max_seqlen

    return NestedTensor(
        values,
        offsets,
        _ragged_idx=ragged_idx,
        _metadata_cache=metadata_cache,
    )


@register_jagged_func(
    torch.ops.aten._nested_view_from_jagged.default,
    "values: t, offsets: t, dummy: jt_all, lengths: t?, ragged_idx: any?, min_seqlen: t?, max_seqlen: t?",
)
def _nested_view_from_jagged_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    values, offsets, lengths = (
        new_kwargs["input"],
        new_kwargs["offsets"],
        new_kwargs["lengths"],
    )
    ragged_idx = new_kwargs["ragged_idx"]
    min_seqlen = new_kwargs["min_seqlen"]
    max_seqlen = new_kwargs["max_seqlen"]
    metadata_cache = {}
    if min_seqlen is not None:
        metadata_cache["min_seqlen"] = min_seqlen
    if max_seqlen is not None:
        metadata_cache["max_seqlen"] = max_seqlen

    return NestedTensor(
        values,
        offsets,
        lengths=lengths,
        _ragged_idx=ragged_idx,
        _metadata_cache=metadata_cache,
    )


@register_jagged_func(torch.ops.aten._nested_get_offsets.default, "self: jt_all")
def _nested_get_offsets(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    return inp._offsets


@register_jagged_func(torch.ops.aten._nested_get_lengths.default, "self: jt_all")
def _nested_get_lengths(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    return inp._lengths


@register_jagged_func(torch.ops.aten._nested_get_ragged_idx.default, "self: jt_all")
def _nested_get_ragged_idx(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    return inp._ragged_idx


@register_jagged_func(torch.ops.aten._nested_get_min_seqlen.default, "self: jt_all")
def _nested_get_min_seqlen(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    return inp._metadata_cache.get("min_seqlen", None)


@register_jagged_func(torch.ops.aten._nested_get_max_seqlen.default, "self: jt_all")
def _nested_get_max_seqlen(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    return inp._metadata_cache.get("max_seqlen", None)


# If a section of the Nested Tensor is fully masked out we still retain the section with a length of 0
@register_jagged_func(torch.ops.aten.masked_select.default, "self: jt, mask: any")
def masked_select_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )
    inp = new_kwargs.pop("input")
    mask = new_kwargs.pop("mask")

    if inp.ndim > 2:
        raise RuntimeError("masked_select only support 2-D selections currently")
    elif inp.shape != mask.shape:
        raise RuntimeError(
            f"Mask with shape {mask.shape} is not compatible with input's shape {inp.shape}"
        )
    res_values = inp._values.masked_select(mask.values())
    mask_cumsum = F.pad(mask.values().cumsum(dim=0), (1, 0))  # type: ignore[arg-type]

    args = extract_kwargs(inp)
    args["offsets"] = mask_cumsum[inp._offsets]
    return NestedTensor(
        values=res_values,
        **args,
    )


@register_jagged_func(
    torch.ops.aten._nested_select_backward.default,
    "grad_output: t, self: jt_all, dim: any, index: any",
)
def _nested_select_backward_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    grad_output = new_kwargs.pop("grad_output")

    grad_input = torch.zeros_like(inp, dtype=grad_output.dtype)
    grad_input.select(new_kwargs["dim"], new_kwargs["index"]).copy_(grad_output)

    return grad_input


@register_jagged_func(torch.ops.aten.record_stream.default, "self: jt_all, s: any")
def record_stream_default(func, *args, **kwargs) -> None:
    inp = args[0]
    stream = args[1]
    # ensure all components live until stream computation completes
    func(inp._values, stream)
    func(inp._offsets, stream)
    if inp._lengths is not None:
        func(inp._lengths, stream)


@register_jagged_func(
    [
        torch.ops.aten.new_empty.default,
        torch.ops.aten.new_zeros.default,
        torch.ops.aten.new_ones.default,
    ],
    "self: jt_all, size: any, dtype: any?, layout: any?, device: any?, pin_memory: any?",
)
def new_empty_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    if len(new_kwargs["size"]) == 0:
        return func(inp._values, **new_kwargs)

    raise RuntimeError("new_empty() not supported for NJT with shape != ()")


@register_jagged_func(
    [
        torch.ops.aten.elu_backward.default,
        torch.ops.aten.hardshrink_backward.default,
        torch.ops.aten.hardsigmoid_backward.default,
        torch.ops.aten.hardtanh_backward.default,
        torch.ops.aten.softplus_backward.default,
        torch.ops.aten.softshrink_backward.default,
    ],
    "self: jt_all, ...",
)
def activation_backward(func, *args, **kwargs):
    # first NJT arg is expected to be grad_output
    grad_output = next(arg for arg in args if isinstance(arg, NestedTensor))
    return NestedTensor(
        func(
            *(arg._values if isinstance(arg, NestedTensor) else arg for arg in args),
            **kwargs,
        ),
        **extract_kwargs(grad_output),
    )


@register_jagged_func(torch.ops.aten.fill.Scalar, "self: jt_all, value: any")
def fill_Scalar(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))


@register_jagged_func(torch.ops.aten.fill_.Scalar, "self: jt_all, value: any")
def fill__Scalar(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    func(inp._values, **new_kwargs)
    return inp


@register_jagged_func(torch.ops.aten.frexp.Tensor, "self: jt_all")
def frexp_Tensor(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    output_kwargs = extract_kwargs(inp)

    mantissa, exponent = func(inp._values)
    return NestedTensor(mantissa, **output_kwargs), NestedTensor(
        exponent, **output_kwargs
    )


@register_jagged_func(
    torch.ops.aten.matmul_backward.default,
    "grad: any, self: any, other: any, mask: any",
)
def matmul_backward_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    grad = new_kwargs.pop("grad")
    inp = new_kwargs.pop("input")
    other = new_kwargs.pop("other")
    grad_input_mask = new_kwargs.pop("mask")

    if grad is None:
        return (None, None)

    grad_self = None
    if grad_input_mask[0]:
        grad_self = torch.matmul(grad, other.transpose(-1, -2))

    grad_other = None
    if grad_input_mask[1]:
        grad_other = torch.matmul(inp.transpose(-1, -2), grad)

    return (grad_self, grad_other)


# Make the dummy available on the C++ side.
@register_jagged_func(torch.ops.aten._nested_get_jagged_dummy.default, "self: any")
def _nested_get_jagged_dummy(func, *args, **kwargs):
    from torch.nested._internal.nested_tensor import _nt_view_dummy

    return _nt_view_dummy()


with torch.library._scoped_library("aten", "IMPL") as aten:
    aten.impl("_nested_get_jagged_dummy", _nested_get_jagged_dummy, "CPU")
    aten.impl("_nested_get_jagged_dummy", _nested_get_jagged_dummy, "CUDA")
    aten.impl("_nested_get_jagged_dummy", _nested_get_jagged_dummy, "Meta")
<|endoftext|># mypy: allow-untyped-defs
import math
from typing import *  # noqa: F403

import torch
from torch._C import DispatchKey, DispatchKeySet
from torch._prims_common import canonicalize_dim, is_expandable_to
from torch.nested._internal.nested_int import NestedIntNode
from torch.utils.weak import WeakTensorKeyDictionary


_tensor_id_counter = 0
_tensor_symint_registry = WeakTensorKeyDictionary()


def get_tensor_symint(tensor, *, coeff=1):
    from torch._subclasses.fake_tensor import FakeTensor
    from torch._subclasses.functional_tensor import mb_unwrap_functional_tensor

    # NB: Only FakeTensor is associated with a memo
    tensor = mb_unwrap_functional_tensor(tensor)
    if isinstance(tensor, FakeTensor):  # noqa: ISINSTANCE_FAKE_TENSOR
        return tensor.get_nested_int(coeff=coeff)

    global _tensor_id_counter

    tensor_symint = _tensor_symint_registry.get(tensor)
    if tensor_symint is None:
        tensor_symint = torch.SymInt(NestedIntNode(_tensor_id_counter, coeff))
        _tensor_id_counter += 1
        _tensor_symint_registry[tensor] = tensor_symint
    return tensor_symint


# SDPA metadata; max / min seqlens are needed for e.g. flash
def _get_sdpa_extreme_seqlen(func, tensor):
    return int(func(tensor).item())


def _store_val_in_tensor(val) -> torch.Tensor:
    # hack to get dynamic shapes support: store in a (val, 0) shaped tensor
    return torch.zeros(val, 0)


def _load_val_from_tensor(t: torch.Tensor):
    return t.shape[0]


def _jagged_numel(inp, func):
    if inp._lengths is None:
        return inp._values.numel()

    fixed_shape = (*inp._size[1 : inp._ragged_idx], *inp._size[inp._ragged_idx + 1 :])
    fixed_numel = math.prod(fixed_shape)

    from torch._subclasses.fake_tensor import DynamicOutputShapeException, is_fake
    from torch._subclasses.functional_tensor import mb_unwrap_functional_tensor

    lengths = mb_unwrap_functional_tensor(inp._lengths)
    if is_fake(lengths):
        raise DynamicOutputShapeException(func)

    return int(inp._lengths.sum().item()) * fixed_numel


# serialization function must be defined at top level
def _rebuild_njt(constructor_kwargs):
    return NestedTensor(**constructor_kwargs)


class NestedTensor(torch.Tensor):
    _values: torch.Tensor  # type: ignore[assignment]
    _offsets: torch.Tensor
    _lengths: Optional[torch.Tensor]
    # NOTE [ Nested ints for ragged sizes and strides ]
    #
    # Jagged layout tensors are tensors that represent a n-dim tensor with a
    # ragged dimension, but are backed by an (n-1)-dim tensor underneath, e.g.,
    # a jagged tensor with outer shape [B, x, D] is represented internally by a
    # tensor with shape [sum(x), D] where we introduce what we call a nested int
    # denoted as "x" here (but sometimes denoted with "*" to
    # represent the ragged dimension, and sum(x) represents the dim of the inner
    # tensor or equivalently the sum of all the sizes of the constituent
    # tensors' varying lengths.
    #
    # We also use nested ints to represent the strides of this tensor.
    # For example, a jagged tensor with shape [B, x, D] can be strided in two
    # ways: [xD, D, 1] and [x, 1, sum(x)], where xD represents x multiplied by D
    _size: tuple[int, ...]
    _strides: tuple[int, ...]
    # Indicates that the nth dimension is ragged
    _ragged_idx: int
    _metadata_cache: Dict[str, Any]

    @staticmethod
    def __new__(
        cls,
        values,
        offsets,
        *,
        lengths=None,
        **kwargs,
    ):
        ks = DispatchKeySet(DispatchKey.NestedTensor)
        ks = ks.add(DispatchKey.AutogradNestedTensor)

        # Only support jagged for now.
        if offsets is None:
            raise AssertionError("offsets must not be None")
        if offsets.ndim != 1:
            raise AssertionError(f"offsets must be 1D, but got {offsets.ndim}D")
        if isinstance(values, NestedTensor):
            raise AssertionError("values must not be a NestedTensor")
        if values.device != offsets.device:
            raise AssertionError(
                f"values and offsets must be on the same device, but got "
                f"values.device={values.device} and offsets.device={offsets.device}"
            )

        # Query cache for the symint associated with offsets or lengths
        # (create a new one if needed).
        ragged_source = offsets if lengths is None else lengths
        ragged_size = get_tensor_symint(ragged_source, coeff=1)
        _ragged_idx = kwargs.get("_ragged_idx", 1)
        B = offsets.shape[0] - 1
        if lengths is not None:
            if B != lengths.shape[0]:
                raise AssertionError(
                    f"offsets and lengths batch sizes must match: "
                    f"offsets.shape[0] - 1 = {B}, lengths.shape[0] = {lengths.shape[0]}"
                )

        # subtract 1 to convert to values dim space
        r = _ragged_idx - 1
        _size = (B, *values.shape[:r], ragged_size, *values.shape[r + 1 :])
        stride = values.stride()
        _strides = (ragged_size * stride[r], *stride)

        r = torch.Tensor._make_wrapper_subclass(
            cls,
            _size,
            _strides,
            0,
            torch.contiguous_format,
            values.dtype,
            torch.jagged,
            values.device,
            False,
            kwargs.get("requires_grad", False),
            "sizes",
            False,
            True,  # dispatch_layout
            ks,
            # don't try to calculate storage based on non-zero size
            storage_size=values.untyped_storage().size(),
        )
        r._ragged_idx = _ragged_idx
        r._size = _size
        r._strides = _strides

        return r

    def __init__(self, values, offsets, *, lengths=None, **kwargs) -> None:
        super().__init__()

        self._values = values
        self._offsets = offsets
        self._lengths = lengths

        # holds properties that are computed lazily
        self._metadata_cache = kwargs.get("_metadata_cache") or {}

        # Collapsed ragged dim must always be dynamic.
        # Use _dynamo_propagated_dynamic_indices (unguarded) rather than
        # maybe_mark_dynamic (which sets _dynamo_weak_dynamic_indices, a guarded
        # attribute) to avoid spurious guard failures when the same tensor is
        # reused across multiple compile calls. The builder reads both attributes,
        # so dynamism propagation still works.
        # See [Note: Dimension Marking Guards] in torch/_dynamo/guards.py.
        # Inlined here to avoid circular import from runtime_wrappers.
        self._dynamo_propagated_dynamic_indices = {self._ragged_idx}  # type: ignore[attr-defined]
        self._values._dynamo_propagated_dynamic_indices = {self._ragged_idx - 1}  # type: ignore[attr-defined]

        # min / max sequence length should be dynamic if present
        max_seqlen_tensor = self._metadata_cache.get("max_seqlen", None)
        if max_seqlen_tensor is not None:
            torch._dynamo.mark_dynamic(max_seqlen_tensor, 0)
        min_seqlen_tensor = self._metadata_cache.get("min_seqlen", None)
        if min_seqlen_tensor is not None:
            torch._dynamo.mark_dynamic(min_seqlen_tensor, 0)

    def values(self):
        # dispatch to get proper view relationship
        return torch._nested_get_values(self)  # type: ignore[attr-defined]

    def offsets(self):
        return self._offsets

    def lengths(self):
        return self._lengths

    # Private accessor functions for min / max sequence length. They're
    # purposefully not @properties because those don't work with PT2 (yet).
    # These compute / cache if not present.
    # TODO: Revisit this when @properties are better supported by PT2. I think the ideal
    # state would be to have public @properties for min / max sequence length that compile
    # (including setters).
    def _get_max_seqlen(self):
        max_seqlen_tensor = self._max_seqlen_tensor
        if max_seqlen_tensor is None:
            # compute & cache
            max_val = _get_sdpa_extreme_seqlen(
                torch.max,
                self._offsets.diff() if self._lengths is None else self._lengths,
            )
            max_seqlen_tensor = _store_val_in_tensor(max_val)
            self._metadata_cache["max_seqlen"] = max_seqlen_tensor
        return _load_val_from_tensor(max_seqlen_tensor)

    def _get_min_seqlen(self):
        min_seqlen_tensor = self._min_seqlen_tensor
        if min_seqlen_tensor is None:
            # compute & cache
            min_val = _get_sdpa_extreme_seqlen(
                torch.min,
                self._offsets.diff() if self._lengths is None else self._lengths,
            )
            min_seqlen_tensor = _store_val_in_tensor(min_val)
            self._metadata_cache["min_seqlen"] = min_seqlen_tensor
        return _load_val_from_tensor(min_seqlen_tensor)

    # Private accessors used for treating min / max seqlen as inner tensors for
    # flatten / unflatten. These must be properties to work with the traceable wrapper
    # subclass logic. These do not compute / cache if not present.
    @property
    def _max_seqlen_tensor(self) -> Optional[torch.Tensor]:
        return self._metadata_cache.get("max_seqlen", None)

    @_max_seqlen_tensor.setter
    def _max_seqlen_tensor(self, val: Optional[torch.Tensor]) -> None:
        self._metadata_cache["max_seqlen"] = val

    @property
    def _min_seqlen_tensor(self) -> Optional[torch.Tensor]:
        return self._metadata_cache.get("min_seqlen", None)

    @_min_seqlen_tensor.setter
    def _min_seqlen_tensor(self, val: Optional[torch.Tensor]) -> None:
        self._metadata_cache["min_seqlen"] = val

    # These are old private @property accessors that are kept around for internal BC
    # reasons. TODO: Remove these!
    @property
    def _max_seqlen(self):
        return self._get_max_seqlen()

    @property
    def _min_seqlen(self):
        return self._get_min_seqlen()

    # Convenience accessors that return a min / max seqlen if one is present and do NOT
    # compute / cache them if they're not.
    @property
    def _maybe_max_seqlen(self) -> int | torch.SymInt | None:
        mt = self._max_seqlen_tensor
        return None if mt is None else _load_val_from_tensor(mt)

    @property
    def _maybe_min_seqlen(self) -> int | torch.SymInt | None:
        mt = self._min_seqlen_tensor
        return None if mt is None else _load_val_from_tensor(mt)

    def _is_contiguous_or_false(self):
        if self.lengths() is not None:
            return False
        from torch._prims_common import is_contiguous_for_memory_format_or_false

        return is_contiguous_for_memory_format_or_false(
            self._values, memory_format=torch.contiguous_format
        )

    def __repr__(self) -> str:  # type: ignore[override]
        # We should implement this in torch/_tensor_str.py instead
        grad_fn_str = (
            f", requires_grad={self.requires_grad}" if self.requires_grad else ""
        )

        if self.grad_fn:
            grad_fn_str = f", grad_fn={self.grad_fn}"

        return f"NestedTensor(size={self._size}, offsets={self._offsets}{grad_fn_str}, contiguous={self._is_contiguous_or_false()})"

    # TODO: Remove this in favor of the default tensor subclass serialization logic.
    # We don't do this today because of https://github.com/pytorch/pytorch/issues/125622.
    def __reduce_ex__(self, proto):
        state = torch._utils._get_obj_state(self)

        # Cached PyCapsules for sizes / strides are not serializable.
        # See Note [Tensor Subclass custom size/stride caching strategy]
        self._clear_non_serializable_cached_data()
        # SymNodes are not serializable
        if "_size" not in state or "_strides" not in state:
            raise AssertionError("state must contain '_size' and '_strides'")
        state = dict(state)
        del state["_size"]
        del state["_strides"]

        func = _rebuild_njt
        constructor_kwargs = {
            "values": self._values,
            "offsets": self._offsets,
            "lengths": self._lengths,
            "_ragged_idx": self._ragged_idx,
            "_metadata_cache": self._metadata_cache,
            "requires_grad": self.requires_grad,
        }
        args = (constructor_kwargs,)
        return (torch._tensor._rebuild_from_type_v2, (func, type(self), args, state))

    def __tensor_flatten__(self):
        ctx = {
            "requires_grad": self.requires_grad,
            "ragged_idx": self._ragged_idx,
        }
        inner_tensors = ["_values", "_offsets"]
        if self._lengths is not None:
            inner_tensors.append("_lengths")
        if self._min_seqlen_tensor is not None:
            inner_tensors.append("_min_seqlen_tensor")
        if self._max_seqlen_tensor is not None:
            inner_tensors.append("_max_seqlen_tensor")
        return inner_tensors, ctx

    @staticmethod
    def __tensor_unflatten__(inner_tensors: Dict, meta, outer_size, outer_stride):
        from torch._subclasses.fake_tensor import FakeTensor

        # inner tensors: _values, _offsets, [_lengths], [_min_seqlen], [_max_seqlen]
        if not (len(inner_tensors) >= 2 and len(inner_tensors) <= 5):
            raise AssertionError(
                f"Expected 2-5 inner tensors, but got {len(inner_tensors)}"
            )
        values = inner_tensors["_values"]
        offsets = inner_tensors["_offsets"]
        lengths = inner_tensors.get("_lengths", None)
        min_seqlen_tensor = inner_tensors.get("_min_seqlen_tensor", None)
        max_seqlen_tensor = inner_tensors.get("_max_seqlen_tensor", None)

        metadata_cache = {}
        if min_seqlen_tensor is not None:
            metadata_cache["min_seqlen"] = min_seqlen_tensor
        if max_seqlen_tensor is not None:
            metadata_cache["max_seqlen"] = max_seqlen_tensor
        ragged_idx = meta["ragged_idx"]

        # Alternatively, we could make it the caller's responsibility to
        # cache it. But this heuristic seems simple enough.
        ragged_source = offsets if lengths is None else lengths
        if isinstance(ragged_source, FakeTensor):  # noqa: ISINSTANCE_FAKE_TENSOR
            ragged_size = outer_size[ragged_idx]
            ragged_source.nested_int_memo = ragged_size

        return NestedTensor(
            values,
            offsets=offsets,
            lengths=lengths,
            requires_grad=meta["requires_grad"],
            _ragged_idx=ragged_idx,
            _metadata_cache=metadata_cache,
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):  # type: ignore[override]
        # If you're wondering why there's a nested tensor with one of its
        # size = -1, see note: [NJT outer_size in AOTDispatcher]
        kwargs = {} if kwargs is None else kwargs

        if args and isinstance(args[0], NestedTensor) and len(args) == 1 and not kwargs:
            inp = args[0]
            if func is torch.ops.aten.is_non_overlapping_and_dense.default:
                return False
            if func is torch.ops.aten.sym_size.default:
                return inp._size
            if func is torch.ops.aten.dim.default:
                return len(inp._size)
            if func in (torch.ops.aten.sym_numel.default, torch.ops.aten.numel.default):
                return _jagged_numel(inp, func)
            if func is torch.ops.aten.sym_stride.default:
                return inp._strides
            if func is torch.ops.aten.sym_storage_offset.default:
                return inp._values.storage_offset()
            if func is torch.ops.prim.layout.default:
                return torch.jagged

        # Lazy import to avoid circular dependency
        from .ops import lookup_jagged

        fn = lookup_jagged(func, *args, **kwargs)
        if fn is not None:
            return fn(*args, **kwargs)

        # Poor man's redispatch for composite ops. This becomes relevant under inference
        # mode, where disabling autograd key dispatch prevents decomposition.
        all_dks = (
            # We want to handle both the cases where NestedTensor overrides the
            # composite implicit autograd kernel, and the case where it doesn't.
            # Prioritize calling into NestedTensor's kernel if it exists.
            torch._C.DispatchKey.CompositeImplicitAutogradNestedTensor,
            torch._C.DispatchKey.CompositeImplicitAutograd,
        )
        for dk in all_dks:
            if torch._C._dispatch_has_kernel_for_dispatch_key(func.name(), dk):
                with torch.overrides.enable_reentrant_dispatch():
                    return func._op_dk(dk, *args, **kwargs)

        raise NotImplementedError(func)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if args and isinstance(args[0], NestedTensor):
            inp = args[0]
            if (
                (func is torch.Tensor.size or func is torch.Tensor.stride)
                and len(args) <= 2
                and (not kwargs or set(kwargs) == {"dim"})
            ):
                dim = kwargs.get("dim", args[1] if len(args) == 2 else None)
                data = inp._size if func is torch.Tensor.size else inp._strides
                if dim is None:
                    return torch.Size(data) if func is torch.Tensor.size else data
                return data[canonicalize_dim(len(data), dim)]
            if func is torch.Tensor.dim and len(args) == 1 and not kwargs:
                return len(inp._size)
            if (
                getattr(func, "__name__", None) == "__get__"
                and len(args) == 1
                and not kwargs
            ):
                descriptor = getattr(func, "__self__", None)
                if descriptor is torch.Tensor.shape:
                    return torch.Size(inp._size)
                if descriptor is torch.Tensor.ndim:
                    return len(inp._size)

        from torch.fx.experimental.proxy_tensor import maybe_enable_thunkify

        from .ops import jagged_torch_function

        # This should be removed after
        # https://github.com/pytorch/pytorch/pull/125941/ lands
        with maybe_enable_thunkify():
            try:
                return jagged_torch_function(func, *args, **kwargs)
            except NotImplementedError:
                pass
            with torch._C.DisableTorchFunctionSubclass():
                return func(*args, **kwargs)


# NB: These fake view autograd.Functions are superseded by real view ops. Don't use them!
# TODO: Remove ViewBufferFromNested, ViewNestedFromBuffer, and buffer_from_jagged once the
# internal BC period has passed.


# Not actually a view!
class ViewBufferFromNested(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: NestedTensor):  # type: ignore[override]
        ctx.save_for_backward(x.offsets())
        ctx.metadata_cache = x._metadata_cache
        ctx.ragged_idx = x._ragged_idx
        return x._values

    @staticmethod
    def backward(ctx, gO: torch.Tensor):  # type: ignore[override]
        (offsets,) = ctx.saved_tensors
        return NestedTensor(
            gO,
            offsets=offsets,
            _metadata_cache=ctx.metadata_cache,
            _ragged_idx=ctx.ragged_idx,
        )


# Not actually a view!
class ViewNestedFromBuffer(torch.autograd.Function):
    @staticmethod
    def forward(  # pyrefly: ignore  # bad-override
        ctx,
        values: torch.Tensor,
        offsets: torch.Tensor,
        metadata_cache: Optional[Dict[str, Any]] = None,
    ):  # type: ignore[override]
        # maintain BC with this usages of this where the seqlens are stuffed
        # directly into the metadata cache as non-Tensors / ints
        if metadata_cache is not None:
            min_seqlen = metadata_cache.get("min_seqlen", None)
            max_seqlen = metadata_cache.get("max_seqlen", None)
            if min_seqlen is not None and not isinstance(min_seqlen, torch.Tensor):
                metadata_cache["min_seqlen"] = _store_val_in_tensor(min_seqlen)
            if max_seqlen is not None and not isinstance(max_seqlen, torch.Tensor):
                metadata_cache["max_seqlen"] = _store_val_in_tensor(max_seqlen)
        return NestedTensor(
            values.detach(),
            offsets=offsets,
            _metadata_cache=metadata_cache,
        )

    @staticmethod
    def backward(ctx, gO: NestedTensor):  # type: ignore[override]
        return gO._values, None, None


def buffer_from_jagged(jagged):
    return ViewBufferFromNested.apply(jagged)


# Need to make it obvious that users should be passing in offsets
def jagged_from_list(
    tensors: List[torch.Tensor],
    offsets: Optional[torch.Tensor],
    dtype=None,
    device=None,
) -> tuple[NestedTensor, torch.Tensor]:
    """Constructs a NestedTensor backed by jagged layout from a list of tensors"""

    if len(tensors) == 0:
        raise RuntimeError("Cannot construct a nested tensor from an empty tensor list")
    if not len(set(t.dtype for t in tensors)) == 1:  # noqa: C401
        raise RuntimeError(
            "When constructing a nested tensor, all tensors in list must have the same dtype"
        )
    if not len(set(t.device for t in tensors)) == 1:  # noqa: C401
        raise RuntimeError(
            "When constructing a nested tensor, all tensors in list must be on the same device"
        )
    if not len(set(t.dim() for t in tensors)) == 1:  # noqa: C401
        raise RuntimeError(
            "When constructing a nested tensor, all tensors in list must have the same dim"
        )
    component_dim = tensors[0].dim()
    if component_dim == 0:
        raise RuntimeError(
            "Cannot construct a nested tensor from a list of zero-dim tensors"
        )

    # Check that the NT is representable by the jagged layout, which
    # allows for a single ragged dimension after the batch dim.
    # e.g. (B, *, D_0, ..., D_N), (B, D_0, *, ..., D_N), etc.
    sizes = [t.shape for t in tensors]
    ragged_idx = None
    for d in range(component_dim):
        dim_is_ragged = any(size[d] != sizes[0][d] for size in sizes)
        if dim_is_ragged:
            if ragged_idx is None:
                # add 1 to convert to outer NJT dim space
                ragged_idx = d + 1
            else:
                raise RuntimeError(
                    "Cannot represent given tensor list as a nested tensor with the jagged layout. "
                    "Note that the jagged layout only allows for a single ragged dimension. "
                    "For example: (B, *, D_0, D_1, ..., D_N), with ragged * dim."
                )

    # allow for a rectangular NJT and default the ragged dim next to the batch dim
    if ragged_idx is None:
        ragged_idx = 1

    # Set properties appropriately.
    values = torch.cat(tensors, dim=(ragged_idx - 1))
    to_kwargs = {}
    if device is not None:
        to_kwargs["device"] = device
    if dtype is not None:
        to_kwargs["dtype"] = dtype
    values = values.to(**to_kwargs)

    # Calculate jagged offsets if not provided.
    if offsets is None:
        # Jagged layout specifies that offsets are stored as int64 on the same device as values.
        # TODO: An alternative way to construct offsets is to use F.pad. This avoids creating
        # an extra leaf tensor during the forward, potentially resolving compatibility issues.
        offsets = torch.cat(
            [
                torch.zeros(1, dtype=torch.int64, device=values.device),
                torch.tensor(
                    [s[ragged_idx - 1] for s in sizes], device=values.device
                ).cumsum(dim=0),
            ]
        )

    # compute this now since it's easy
    min_seqlen = min(t.shape[ragged_idx - 1] for t in tensors)
    max_seqlen = max(t.shape[ragged_idx - 1] for t in tensors)
    ret_nt = nested_view_from_values_offsets(
        values,
        offsets,
        min_seqlen=min_seqlen,
        max_seqlen=max_seqlen,
        ragged_idx=ragged_idx,
    )
    return (ret_nt, offsets)  # type: ignore[return-value]


def jagged_from_tensor_and_lengths(
    tensor: torch.Tensor, starts: torch.Tensor, lengths: torch.Tensor
) -> tuple[NestedTensor, torch.Tensor, Optional[torch.Tensor]]:
    """Constructs a NestedTensor backed by jagged layout from a tensor, starts of sequences, and sequence lengths"""
    batch_size = tensor.shape[0]
    if is_expandable_to(starts.shape, (batch_size,)) and is_expandable_to(
        lengths.shape, (batch_size,)
    ):
        start_list = starts.expand(batch_size)
        length_list = lengths.expand(batch_size)
    else:
        raise RuntimeError(
            "When constructing a jagged nested tensor using narrow(), "
            "your start and length must be Tensors that broadcast to input.shape[0]"
        )

    # Calculate jagged offsets
    if len(tensor.shape) < 2:
        raise AssertionError(
            "tensor must at least be 2D for the nested narrow op to work"
        )
    max_seq_len = tensor.shape[1]
    offset_lengths = max_seq_len * torch.arange(
        0, batch_size, dtype=torch.int64, device=tensor.device
    )
    # Jagged layout specifies that offsets are stored as int64 on the same device as values.
    offsets = torch.cat(
        [
            start_list + offset_lengths,
            (start_list[-1] + offset_lengths[-1] + length_list[-1]).unsqueeze(0),
        ]
    )

    # Reshape buffer to flatten the 1st and 2nd dimension (view used to enforce non-copy)
    if len(tensor.shape) > 2:
        values = tensor.view(-1, *tensor.shape[2:])
    else:
        values = tensor.view(-1)

    # Check if offsets and lengths make it possibly contiguous and return a regular NT
    is_contiguous = True
    orig_dim = tensor.shape[1]
    if torch.any(length_list[1:-1].ne(orig_dim)):
        is_contiguous = False
    if torch.any(offsets[1:-2].diff().ne(orig_dim)):
        is_contiguous = False
    if offsets[0] + length_list[0] != orig_dim:
        is_contiguous = False

    actual_max_seqlen = int(torch.max(lengths).item())
    min_seqlen = int(torch.min(lengths).item())

    if is_contiguous:
        ret_nt = nested_view_from_values_offsets(
            values[offsets[0] : offsets[-1]],
            offsets - offsets[0],
            min_seqlen=min_seqlen,
            max_seqlen=actual_max_seqlen,
        )
    else:
        ret_nt = nested_view_from_values_offsets_lengths(
            values,
            offsets,
            length_list,
            min_seqlen=min_seqlen,
            max_seqlen=actual_max_seqlen,
        )

    return (ret_nt, offsets, None if is_contiguous else length_list)


# NB: A dummy arg is required so that NestedTensor.__torch_dispatch__() is invoked
# for _nested_view_from_values_offsets(). Sizes don't matter much, but they shouldn't be
# 0/1 because the dummy can be fake-ified and we want to avoid specializing.
# This arg is otherwise unused.
_dummy_instance: Optional[torch.Tensor] = None


def _nt_view_dummy() -> torch.Tensor:
    global _dummy_instance
    if _dummy_instance is None:
        _dummy_instance = NestedTensor(
            values=torch.zeros(3, 3, device="meta"),
            offsets=torch.zeros(3, device="meta", dtype=torch.int64),
        ).detach()
    return _dummy_instance


def nested_view_from_values_offsets(
    values, offsets, ragged_idx=1, min_seqlen=None, max_seqlen=None
):
    min_seqlen_tensor = None
    if min_seqlen is not None:
        min_seqlen_tensor = _store_val_in_tensor(min_seqlen)

    max_seqlen_tensor = None
    if max_seqlen is not None:
        max_seqlen_tensor = _store_val_in_tensor(max_seqlen)

    return torch._nested_view_from_jagged(  # type: ignore[attr-defined]
        values,
        offsets,
        _nt_view_dummy(),
        None,
        ragged_idx,
        min_seqlen_tensor,
        max_seqlen_tensor,
    )  # type: ignore[return-value]


def nested_view_from_values_offsets_lengths(
    values, offsets, lengths, ragged_idx=1, min_seqlen=None, max_seqlen=None
):
    min_seqlen_tensor = None
    if min_seqlen is not None:
        min_seqlen_tensor = _store_val_in_tensor(min_seqlen)

    max_seqlen_tensor = None
    if max_seqlen is not None:
        max_seqlen_tensor = _store_val_in_tensor(max_seqlen)

    return torch._nested_view_from_jagged(  # type: ignore[attr-defined]
        values,
        offsets,
        _nt_view_dummy(),
        lengths,
        ragged_idx,
        min_seqlen_tensor,
        max_seqlen_tensor,
    )  # type: ignore[return-value]


def nested_from_padded(
    padded, offsets, ragged_idx=1, min_seqlen=None, max_seqlen=None, sum_S=None
):
    min_seqlen_tensor = None
    if min_seqlen is not None:
        min_seqlen_tensor = _store_val_in_tensor(min_seqlen)

    max_seqlen_tensor = None
    if max_seqlen is not None:
        max_seqlen_tensor = _store_val_in_tensor(max_seqlen)

    return torch._nested_from_padded_tensor(
        padded,
        offsets,
        _nt_view_dummy(),
        ragged_idx,
        min_seqlen_tensor,
        max_seqlen_tensor,
        sum_S,
    )
<|endoftext|>import functools
import logging
import sys
from typing import cast

from torch._vendor.packaging.version import Version

from ..backends import cuda as _cuda
from .common_utils import (
    _available_version,
    _unavailable_reason,
    check_native_jit_disabled,
    check_native_version_skip,
)
from .dsl_registry import dsl_registry, DSLModuleProtocol
from .registry import (
    _OpCondFn,
    _OpImplFn,
    deregister_op_overrides as _deregister_op_overrides_impl,
    register_op_override as _register_op_override_impl,
)


log = logging.getLogger(__name__)


_HELION_DSL_NAME = "helion"
# Minimum supported Helion: the latest release verified with the native DSL
# kernels (older releases carry the same APIs but are untested).
_HELION_MINIMUM_VERSION = Version("1.2.0")
# HELION_BACKEND name -> (pip package, import module); extend to add more,
# e.g. "cute": ("nvidia-cutlass-dsl", "cutlass").
_HELION_BACKENDS: dict[str, tuple[str, str]] = {"triton": ("triton", "triton")}
_DEFAULT_HELION_BACKEND = "triton"


def _chosen_backend() -> str:
    """Selected Helion lowering backend (HELION_BACKEND env, default triton)."""
    import os

    return os.getenv("HELION_BACKEND", _DEFAULT_HELION_BACKEND)


@functools.cache
def _check_runtime_available() -> tuple[bool, Version | None]:
    if not _cuda.is_built():
        return (False, None)

    import torch

    if torch.version.hip is not None:
        return (False, None)

    backend = _chosen_backend()
    dep = _HELION_BACKENDS.get(backend)
    if dep is None:
        log.info(
            "Helion native DSL ops support backends %s; HELION_BACKEND=%s",
            tuple(_HELION_BACKENDS),
            backend,
        )
        return (False, None)

    reason = _unavailable_reason([("helion", "helion"), dep])
    if reason is not None:
        log.info(
            "Helion native DSL ops require optional packages `helion` and `%s`; %s",
            dep[0],
            reason,
        )
        return (False, None)
    return (True, _available_version("helion"))


def runtime_available() -> bool:
    # Package presence only, like triton/cutedsl; version gated in register_op_override.
    available, _ = _check_runtime_available()
    return available


def runtime_version() -> Version | None:
    _, version = _check_runtime_available()
    return version


@functools.cache
def _version_is_sufficient() -> bool:
    _, version = _check_runtime_available()
    if version is not None and (
        version >= _HELION_MINIMUM_VERSION or check_native_version_skip()
    ):
        return True

    log.info(
        "helion version %s is not sufficient (>= %s); "
        "set TORCH_NATIVE_SKIP_VERSION_CHECK=1 to override",
        version,
        _HELION_MINIMUM_VERSION,
    )
    return False


def deregister_op_overrides() -> None:
    _deregister_op_overrides_impl(disable_dsl_names=_HELION_DSL_NAME)


def register_op_override(
    lib_symbol: str,
    op_symbol: str,
    dispatch_key: str,
    cond: _OpCondFn | None,
    impl: _OpImplFn,
    *,
    allow_multiple_override: bool = False,
    unconditional_override: bool = False,
) -> None:
    available, _ = _check_runtime_available()
    if (not available) or check_native_jit_disabled():
        return

    if not _version_is_sufficient():
        return

    _register_op_override_impl(
        _HELION_DSL_NAME,
        lib_symbol,
        op_symbol,
        dispatch_key,
        cond,
        impl,
        allow_multiple_override=allow_multiple_override,
        unconditional_override=unconditional_override,
    )


dsl_registry.register_dsl(
    _HELION_DSL_NAME, cast(DSLModuleProtocol, sys.modules[__name__])
)
<|endoftext|>import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import ParamSpec, TypeVar

import torch.library


__all__ = [
    "UserOrderingFn",
    "register_op_override",
    "reorder_graphs_from_user_function",
    "reenable_op_overrides",
    "deregister_op_overrides",
    "get_dsl_operations",
    "native_decomp_table",
]

log = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")

_OpCondFn = Callable[P, bool]
_OpImplFn = Callable[P, R]


@dataclass
class _OverrideNode:
    """Track function override data."""

    dsl_name: str
    op_symbol: str
    dispatch_key: str
    cond_fn: _OpCondFn
    impl_fn: _OpImplFn
    # Identifier of the opaque `_native::<node_id>` op that carries this
    # override's impl. Assigned once at registration time and never reused --
    # reorder / deregister / reenable do not change it, so the cached
    # `_defined_native_ops` entry remains valid.
    node_id: str
    unconditional_override: bool = False
    active: bool = True


UserOrderingFn = Callable[[str, str, list[_OverrideNode]], list[_OverrideNode]]


@dataclass
class _FilterState:
    """Manages filtering state for override nodes."""

    _dsl_names: set[str] = field(default_factory=set)
    _op_symbols: set[str] = field(default_factory=set)
    _dispatch_keys: set[str] = field(default_factory=set)

    def check_enabled(self, node: _OverrideNode) -> bool:
        """
        Check if a node is enabled based on current filter state.

        Args:
            node: The override node to check

        Returns:
            bool: True if the node should be enabled, False if filtered out
        """
        if node.dsl_name in self._dsl_names:
            return False

        if node.op_symbol in self._op_symbols:
            return False

        if node.dispatch_key in self._dispatch_keys:
            return False

        return True

    def update(
        self,
        dsl_names: str | Iterable[str] | None,
        op_symbols: str | Iterable[str] | None,
        dispatch_keys: str | Iterable[str] | None,
        remove_keys: bool = False,
    ) -> None:
        """
        Update filter sets as (current | new) or (current ~ new).

        Args:
            dsl_names: DSL names to add/remove from filter
            op_symbols: Operation symbols to add/remove from filter
            dispatch_keys: Dispatch keys to add/remove from filter
            remove_keys: If True, remove keys from filter; if False, add them

        Note:
            Uses set.discard as it doesn't raise an exception if the element
            wasn't in the set to begin with.
        """
        if remove_keys:
            self._dsl_names -= set(_resolve_iterable(dsl_names))
            self._op_symbols -= set(_resolve_iterable(op_symbols))
            self._dispatch_keys -= set(_resolve_iterable(dispatch_keys))
        else:
            self._dsl_names |= set(_resolve_iterable(dsl_names))
            self._op_symbols |= set(_resolve_iterable(op_symbols))
            self._dispatch_keys |= set(_resolve_iterable(dispatch_keys))

    def build_disable_key_set(self) -> set[tuple[str, str]]:
        """
        Build a set of dictionary keys based on the current filter state.

        Returns:
            set[tuple[str, str]]: Set of (op_symbol, dispatch_key) tuples
        """
        return _build_key_set(
            self._dsl_names,
            self._op_symbols,
            self._dispatch_keys,
        )

    def __str__(self) -> str:
        """Return string representation of filter state."""
        s = ""
        s += "Filter State:\n"
        s += "  === DSL: ===\n"
        for i, dsl in enumerate(self._dsl_names):
            s += f"    {i}: {dsl}\n"
        s += "  === OP SYMBOL: ===\n"
        for i, op in enumerate(self._op_symbols):
            s += f"    {i}: {op}\n"
        s += "  === DISPATCH KEYS: ===\n"
        for i, key in enumerate(self._dispatch_keys):
            s += f"    {i}: {key}\n"

        return s


# Store the global override filtering state
_filter_state: _FilterState = _FilterState()

# Store torch.library.Library instances
_libs: dict[tuple[str, str], torch.library.Library] = {}

# Decomposition-style routers for the compile / export path, keyed by aten
# OpOverload. Built in `_register_overrides_from_graph` alongside the eager
# `Library.impl` routers.
#
# This dict is *not* written into any global compile/export decomp table (in
# particular, not into torch._inductor.decomposition.decompositions). Users
# opt into the overrides explicitly by passing `native_decomp_table()` to
# `ExportedProgram.run_decompositions(...)`, or by threading it through
# Dynamo/Inductor config for `torch.compile`. Reasons:
#
#   1. torch._inductor.decomposition.decompositions is an Inductor internal
#      with no stability guarantee; writing to it couples us to Inductor's
#      refactors.
#   2. Global writes would affect *every* consumer of that table (tests,
#      third-party backends, etc.), with no scoping.
#   3. Import-time writes would force `import torch._native` to also pull in
#      inductor → dynamo → triton, breaking
#      `test_no_dsl_imports_after_import_torch`.
#
# Keeping the table registry-local means `import torch._native` stays cheap
# and consumers control exactly when and where overrides take effect.
_native_decomp_overrides: dict[object, Callable] = {}


def _has_cow_tensor(*args, **kwargs) -> bool:
    def is_cow_tensor(arg) -> bool:
        if isinstance(arg, torch.Tensor):
            return torch._C._is_cow_tensor(arg)  # pyrefly: ignore[missing-attribute]
        if isinstance(arg, (list, tuple)):
            return any(is_cow_tensor(item) for item in arg)
        if isinstance(arg, dict):
            return any(is_cow_tensor(item) for item in arg.values())
        return False

    return any(is_cow_tensor(arg) for arg in args) or any(
        is_cow_tensor(arg) for arg in kwargs.values()
    )


# store graph structures
_GraphsType = dict[tuple[str, str], list[_OverrideNode]]
_graphs: _GraphsType = {}

_MappingType = dict[str, list[tuple[str, str]]]

# map a {dsl, op, dispatch_key} to keys to all graphs that contain it
_dsl_name_to_lib_graph: _MappingType = {}
_dispatch_key_to_lib_graph: _MappingType = {}
_op_symbol_to_lib_graph: _MappingType = {}

# Monotonic counter used to mint stable, unique `_native::<id>` op names.
# Must never decrease across the process, since `_defined_native_ops`
# pins a fake kernel for each minted id.
_node_id_counter: int = 0

# Dispatch keys where installing an override would break the registry's
# assumptions. The eager router is wired at a backend key so aten's
# higher-priority Autograd/Autocast kernels handle those layers, and the
# fake kernel for `_native::<id>` redispatches to the aten op's meta -- if
# the override lived at Meta or CompositeImplicitAutograd, that
# redispatch would loop back into the router.
_DISALLOWED_DISPATCH_KEYS: frozenset[str] = frozenset(
    {"Meta", "CompositeImplicitAutograd", "CompositeExplicitAutograd"}
)


def _build_key_set(
    dsl_names: str | Iterable[str] | None,
    op_symbols: str | Iterable[str] | None,
    dispatch_keys: str | Iterable[str] | None,
) -> set[tuple[str, str]]:
    """
    Build a set of dictionary keys based on filter criteria.

    Args:
        dsl_names: DSL names to include in key set
        op_symbols: Operation symbols to include in key set
        dispatch_keys: Dispatch keys to include in key set

    Returns:
        set[tuple[str, str]]: Set of (op_symbol, dispatch_key) tuples
    """
    key_set: set[tuple[str, str]] = set()

    def _append_to_set(
        entries: str | Iterable[str] | None, graph_lib_dict: _MappingType
    ) -> None:
        """Helper to add matching keys from graph_lib_dict to key_set."""
        resolved_entries = _resolve_iterable(entries)

        for entry in resolved_entries:
            if entry in graph_lib_dict:
                for key in graph_lib_dict[entry]:
                    key_set.add(key)

    _append_to_set(dsl_names, _dsl_name_to_lib_graph)
    _append_to_set(op_symbols, _op_symbol_to_lib_graph)
    _append_to_set(dispatch_keys, _dispatch_key_to_lib_graph)

    return key_set


def _print_override_graphs(*, print_inactive: bool = False) -> None:
    """
    Print all override graphs for debugging purposes.

    Args:
        print_inactive: Whether to print inactive nodes
    """
    for (op, key), node_list in _graphs.items():
        print(f"{op=}, {key=}")

        for i, node in enumerate(node_list):
            if node.active or print_inactive:
                s: str = f"    {i}: {node.dsl_name=}, {node.unconditional_override=}"
                if print_inactive:
                    s += f" {node.active=}"

                print(s)


# One DEF/FRAGMENT library per namespace -- only one is allowed per process.
_def_libs: dict[str, torch.library.Library] = {}
# Ops that have already been `define`d on the _native namespace.
_defined_native_ops: set[str] = set()
# IMPL libraries on the aten namespace, keyed by (op_symbol, dispatch_key).
# One library per op/key pair so we can call `_destroy()` on it to tear down
# just that one override without affecting other ops at the same dispatch
# key. Torch's Library has no per-kernel removal API -- `_destroy` on the
# library is the only teardown mechanism.
_aten_override_libs: dict[tuple[str, str], torch.library.Library] = {}


def _get_def_library(namespace: str) -> torch.library.Library:
    if namespace not in _def_libs:
        _def_libs[namespace] = torch.library.Library(namespace, "FRAGMENT")
    return _def_libs[namespace]


def _get_or_create_library(dispatch_key: str) -> torch.library.Library:
    """
    Get or create the _native IMPL library for a given dispatch key.

    One library per dispatch key is shared across all overridden ops.
    """
    global _libs

    key = ("_native", dispatch_key)
    if key not in _libs:
        _libs[key] = torch.library.Library("_native", "IMPL", dispatch_key)

    return _libs[key]


def _install_aten_override(op_symbol: str, dispatch_key: str, kernel: Callable) -> None:
    """
    Install (or replace) an aten kernel at (op_symbol, dispatch_key).

    Creates a fresh Library per (op, key) so we can tear down just this
    one override via `_destroy_aten_override` without affecting any other
    override at the same dispatch key.
    """
    key = (op_symbol, dispatch_key)
    # Destroy any existing library so its kernel is fully removed before
    # we install the new one. `_destroy` calls into the dispatcher to
    # unregister all kernels on the library.
    existing = _aten_override_libs.pop(key, None)
    if existing is not None:
        existing._destroy()

    lib = torch.library.Library("aten", "IMPL", dispatch_key)
    lib.impl(op_symbol, kernel, dispatch_key, with_keyset=True)
    _aten_override_libs[key] = lib


def _destroy_aten_override(op_symbol: str, dispatch_key: str) -> None:
    """Tear down the aten override at (op_symbol, dispatch_key), if any."""
    lib = _aten_override_libs.pop((op_symbol, dispatch_key), None)
    if lib is not None:
        lib._destroy()


def _resolve_aten_overload(op_symbol: str) -> "torch._ops.OpOverload | None":
    """
    Resolve `op_symbol` to a concrete OpOverload on `torch.ops.aten`.

    Accepts bare names ("bmm" → aten.bmm.default) and overload-qualified
    names ("add_.Tensor" → aten.add_.Tensor). Returns None if the op is not
    registered (e.g. a test-only op_symbol that never hit the C++ dispatcher).
    """
    name, _, overload_name = op_symbol.partition(".")
    overload_name = overload_name or "default"
    try:
        packet = getattr(torch.ops.aten, name)
        return getattr(packet, overload_name)
    except AttributeError:
        return None


def _aten_schema_tail(op_symbol: str) -> str:
    """Return the schema of at::<op_symbol> with the `aten::<name>` prefix stripped.

    Accepts bare names ("bmm" → aten.bmm.default) and overload-qualified
    names ("add_.Tensor" → aten.add_.Tensor).
    """
    overload = _resolve_aten_overload(op_symbol)
    if overload is None:
        raise AttributeError(f"aten op not found for op_symbol={op_symbol!r}")
    s = str(overload._schema)
    # "aten::bmm(Tensor self, Tensor mat2) -> Tensor" -> "(Tensor self, Tensor mat2) -> Tensor"
    _, rest = s.split("::", 1)
    _, args = rest.split("(", 1)
    return f"({args}"


def _define_native_op_once(name: str, op_symbol: str) -> None:
    # Invariant: callers must only install the eager router at a real backend
    # dispatch key (CPU, CUDA, XPU, ...). The fake kernel below redispatches
    # to the aten op, and if the router were installed at Meta /
    # CompositeImplicitAutograd that redispatch would re-enter the router.
    # `register_op_override` enforces this via `_DISALLOWED_DISPATCH_KEYS`.
    if name in _defined_native_ops:
        return
    _get_def_library("_native").define(f"{name}{_aten_schema_tail(op_symbol)}")
    # Fake/meta kernel: required so export / dynamo / AOTAutograd can shape-infer
    # through the opaque _native op. Reusing the aten op's meta is safe because
    # the schema (and therefore shape rules) is cloned from it.
    aten_overload = _resolve_aten_overload(op_symbol)
    torch.library.register_fake(f"_native::{name}")(
        lambda *args, _aten_overload=aten_overload, **kwargs: _aten_overload(
            *args, **kwargs
        )
    )
    # No autograd or autocast kernels are registered on _native::<id>. Because
    # the router is registered at the backend dispatch key (e.g. CUDA), aten's
    # own higher-priority kernels on aten::<op> handle both:
    #   - Autograd: AutogradCUDA sees aten::<op> in the autograd graph and
    #     uses aten's built-in derivative formula.
    #   - Autocast: AutocastCUDA casts inputs to the autocast dtype before
    #     redispatching down to our CUDA-level router.
    _defined_native_ops.add(name)


def _register_node_impl(
    lib: torch.library.Library, node: _OverrideNode, dispatch_key: str
) -> None:
    """
    Register a single node implementation with the library.

    Args:
        lib: The torch.library.Library instance
        node: The override node to register
        dispatch_key: The dispatch key for registration
    """
    if not node.node_id:
        raise ValueError(
            f"_OverrideNode must have a non-empty node_id before registration "
            f"(dsl_name={node.dsl_name!r}, op_symbol={node.op_symbol!r})"
        )
    _define_native_op_once(node.node_id, node.op_symbol)
    lib.impl(
        node.node_id,
        node.impl_fn,
        dispatch_key,
        with_keyset=False,
        allow_override=True,
    )


def _resolve_iterable(iterable: str | Iterable[str] | None) -> Iterable[str]:
    """
    Resolve various input types to a consistent iterable of strings.

    Args:
        iterable: String, iterable of strings, or None

    Returns:
        Iterable[str]: Consistent iterable output
    """
    if iterable is None:
        return []

    if not isinstance(iterable, Iterable) or isinstance(iterable, str):
        return (iterable,)

    return iterable


def reenable_op_overrides(
    *,
    enable_dsl_names: str | list[str] | None = None,
    enable_op_symbols: str | list[str] | None = None,
    enable_dispatch_keys: str | list[str] | None = None,
) -> None:
    """
    Re-enable overrides by removing them from filter state and reregistering.

    Args:
        enable_dsl_names: DSL names to re-enable
        enable_op_symbols: Operation symbols to re-enable
        enable_dispatch_keys: Dispatch keys to re-enable

    Note:
        This function uses reverse filter state management (removing from
        filters to enable).
    """
    log.info(
        "Re-registering ops by dsl: %s, op_symbol: %s, dispatch_key: %s",
        enable_dsl_names,
        enable_op_symbols,
        enable_dispatch_keys,
    )

    # Update the filters - note `remove_keys=True` because
    # we are removing keys from the filters (vs. adding them)
    _filter_state.update(
        enable_dsl_names,
        enable_op_symbols,
        enable_dispatch_keys,
        remove_keys=True,
    )

    # Get the set of keys that need to be reprocessed
    key_set: set[tuple[str, str]] = _build_key_set(
        enable_dsl_names,
        enable_op_symbols,
        enable_dispatch_keys,
    )

    # Process each affected graph with updated filter state
    for key in key_set:
        op_symbol, dispatch_key = key

        if key in _graphs:
            # Note: We don't need to cleanup and recreate the library here
            # since we're just updating the registration with new filter state
            _register_overrides_from_graph(
                op_symbol, dispatch_key, _graphs[key], filter_state=_filter_state
            )


def deregister_op_overrides(
    *,
    disable_dsl_names: str | list[str] | None = None,
    disable_op_symbols: str | list[str] | None = None,
    disable_dispatch_keys: str | list[str] | None = None,
) -> None:
    """
    De-register overrides by updating filter state and reregistering graphs.

    Args:
        disable_dsl_names: DSL names to disable
        disable_op_symbols: Operation symbols to disable
        disable_dispatch_keys: Dispatch keys to disable

    Note:
        This function uses filter state management to selectively disable
        operations.
    """
    log.info(
        "De-registering ops by dsl: %s, op_symbol: %s, dispatch_key: %s",
        disable_dsl_names,
        disable_op_symbols,
        disable_dispatch_keys,
    )

    # Update filter state to disable specified entries
    _filter_state.update(disable_dsl_names, disable_op_symbols, disable_dispatch_keys)

    # Get the set of keys that need to be reprocessed
    key_set: set[tuple[str, str]] = _filter_state.build_disable_key_set()

    # Process each affected graph with filter state
    for key in key_set:
        op_symbol, dispatch_key = key

        if key in _graphs:
            _cleanup_and_reregister_graph(
                op_symbol,
                dispatch_key,
                _graphs[key],
                filter_state=_filter_state,
            )


def get_dsl_operations(dsl_name: str) -> list[str]:
    """Get list of operations registered by a specific DSL.

    Args:
        dsl_name: Name of the DSL to query.

    Returns:
        Sorted list of operation names registered by the DSL.
    """
    operations = set()
    for (op_symbol, _), nodes in _graphs.items():
        for node in nodes:
            if node.dsl_name == dsl_name:
                operations.add(op_symbol)
                break
    return sorted(operations)


def _update_registration_maps(
    dsl_name: str,
    op_symbol: str,
    dispatch_key: str,
    key: tuple[str, str],
) -> None:
    """
    Update the registration mapping dictionaries.

    Args:
        dsl_name: The DSL name
        op_symbol: The operation symbol
        dispatch_key: The dispatch key
        key: The dictionary key tuple
    """
    global _dsl_name_to_lib_graph
    global _op_symbol_to_lib_graph
    global _dispatch_key_to_lib_graph

    def _get_new_entry_or_append(
        registration: dict[str, list[tuple[str, str]]],
        symbol: str,
        key: tuple[str, str],
    ) -> None:
        """Helper to add key to registration list or create new entry."""
        entry_list = registration.get(symbol)

        if entry_list is None:
            entry_list = [key]
            registration[symbol] = entry_list
        else:
            entry_list.append(key)

    _get_new_entry_or_append(_dsl_name_to_lib_graph, dsl_name, key)
    _get_new_entry_or_append(_op_symbol_to_lib_graph, op_symbol, key)
    _get_new_entry_or_append(_dispatch_key_to_lib_graph, dispatch_key, key)


def _always_true(*args: object, **kwargs: object) -> bool:
    return True


def register_op_override(
    backend: str,
    lib_symbol: str,
    op_symbol: str,
    dispatch_key: str,
    cond: _OpCondFn | None,
    impl: _OpImplFn,
    *,
    allow_multiple_override: bool = False,
    unconditional_override: bool = False,
) -> None:
    """
    Register a passed override function to the dispatcher.

    Actually a graph-building operation; real registration happens later.

    Args:
        backend: The backend name (DSL name)
        lib_symbol: Library you're overriding symbols in (must be "aten")
        op_symbol: Name of the operation you're overriding
        dispatch_key: Dispatch key to override
        cond: Predicate choosing whether `impl` applies to a given call. May
            be None if `unconditional_override=True`.
        impl: Implementation function for the override
        allow_multiple_override: Allow overriding an existing override
        unconditional_override: Implementation doesn't have a fallback and
            doesn't require torch.DispatchKeySet as the first argument. When
            True, a trivially-True predicate is supplied for the router if
            `cond` is None.

    Raises:
        ValueError: If lib_symbol is not "aten", if dispatch_key is in
            _DISALLOWED_DISPATCH_KEYS (Meta / CompositeImplicitAutograd /
            CompositeExplicitAutograd), or if cond is None without
            unconditional_override=True.
    """
    if lib_symbol != "aten":
        raise ValueError(f'Unsupported lib_symbol (must be "aten", got: "{lib_symbol}"')

    if dispatch_key in _DISALLOWED_DISPATCH_KEYS:
        raise ValueError(
            f"dispatch_key={dispatch_key!r} is not supported. Overrides must be "
            f"installed at a backend key (e.g. CPU, CUDA, XPU); the router's fake "
            f"kernel redispatches to aten and would recurse otherwise."
        )

    if cond is None:
        if not unconditional_override:
            raise ValueError("cond must be provided unless unconditional_override=True")
        cond = _always_true

    key = (op_symbol, dispatch_key)

    global _graphs, _node_id_counter
    op_graph = _graphs.get(key, [])

    # Mint a stable id for the opaque `_native::<node_id>` op. `_sanitized`
    # strips dots from overload-qualified names so the result is a valid op
    # name. The monotonic counter guarantees uniqueness even if the same
    # (op_symbol, dsl_name) is registered, deregistered, and re-registered.
    _sanitized = op_symbol.replace(".", "_")
    node_id = f"{_sanitized}_{backend}_{_node_id_counter}"
    _node_id_counter += 1

    op_graph.append(
        _OverrideNode(
            dsl_name=backend,
            op_symbol=op_symbol,
            dispatch_key=dispatch_key,
            cond_fn=cond,
            impl_fn=impl,
            unconditional_override=unconditional_override,
            node_id=node_id,
        )
    )
    _graphs[key] = op_graph
    # Build additional maps helpful for de-registration
    _update_registration_maps(backend, op_symbol, dispatch_key, key=key)


def _should_reregister_graph(
    original_graph: list[_OverrideNode],
    new_graph: list[_OverrideNode],
    *,
    force_reregister: bool = False,
) -> bool:
    """
    Determine if a graph needs reregistration based on changes.

    Args:
        original_graph: The original graph before modification
        new_graph: The graph after modification
        force_reregister: If True, always reregister regardless of changes

    Returns:
        bool: True if reregistration is needed
    """
    if force_reregister:
        return True

    # Check if the graph structure has changed
    return original_graph != new_graph


def _cleanup_and_reregister_graph(
    op_symbol: str,
    dispatch_key: str,
    graph: list[_OverrideNode],
    *,
    filter_state: _FilterState | None = None,
) -> None:
    """
    Reregister a graph's routes from scratch.

    Used by reorder / deregister / reenable. Libraries are intentionally
    long-lived singletons; we rebuild the per-op router closure here.

    Args:
        op_symbol: The operation symbol
        dispatch_key: The dispatch key
        graph: The graph to register
        filter_state: Optional filter state for conditional registration
    """
    _register_overrides_from_graph(
        op_symbol,
        dispatch_key,
        graph,
        filter_state=filter_state,
    )


def _apply_graph_transformation(
    transformation_fn: UserOrderingFn,
    *,
    keys_to_process: set[tuple[str, str]] | None = None,
    reregister_overrides: bool = False,
    filter_state: _FilterState | None = None,
) -> None:
    """
    Apply a transformation function to graphs and optionally reregister.

    This is the core pattern used by reorder_graphs_from_user_function and
    can be reused for other graph transformation operations.

    Args:
        transformation_fn: Function to transform each graph
        keys_to_process: Keys to process, or None for all graphs
        reregister_overrides: Whether to reregister changed graphs
        filter_state: Optional filter state for conditional registration

    Note:
        If transformation_fn raises an exception for a specific graph, that graph
        will be skipped and processing will continue with remaining graphs.
    """
    global _graphs

    # Determine which graphs to process
    target_keys = (
        keys_to_process if keys_to_process is not None else set(_graphs.keys())
    )

    # Process each graph
    for op_symbol, dispatch_key in list(target_keys):
        if (op_symbol, dispatch_key) not in _graphs:
            continue  # Skip if graph doesn't exist

        original_graph = list(_graphs[(op_symbol, dispatch_key)])

        # Apply the transformation with error handling
        try:
            new_graph = transformation_fn(op_symbol, dispatch_key, original_graph)
        except (TypeError, ValueError, AttributeError, RuntimeError):
            log.warning(
                "Graph transformation failed for %s/%s. Preserving original graph.",
                op_symbol,
                dispatch_key,
                exc_info=True,
            )
            continue
        except Exception:
            log.exception(
                "Unexpected error in graph transformation for %s/%s. Preserving original graph.",
                op_symbol,
                dispatch_key,
            )
            continue

        # Validate that the transformation returned a valid result
        if not isinstance(new_graph, list):
            log.warning(
                "Graph transformation returned invalid type %s for %s/%s. Expected list. Preserving original graph.",
                type(new_graph).__name__,
                op_symbol,
                dispatch_key,
            )
            continue

        # Update the graph
        _graphs[(op_symbol, dispatch_key)] = new_graph

        # Reregister if needed
        if reregister_overrides and _should_reregister_graph(
            original_graph, new_graph, force_reregister=False
        ):
            _cleanup_and_reregister_graph(
                op_symbol,
                dispatch_key,
                new_graph,
                filter_state=filter_state,
            )


def native_decomp_table(
    overrides_only: bool = False,
) -> dict[object, Callable]:
    """
    Return a decomposition table suitable for passing to
    ``ExportedProgram.run_decompositions`` or to Inductor / Dynamo as a
    decomposition set for ``torch.compile``.

    This is the canonical way to apply native overrides outside of eager.
    Callers opt in explicitly -- the registry does **not** install into any
    global compile/export decomp table on its own. This preserves three
    properties:

      * ``import torch._native`` doesn't transitively import inductor /
        dynamo / triton.
      * No other consumer (ONNX, tests, third-party backends) accidentally
        picks up our overrides.
      * Scoping is up to the caller: pass the table only where routing is
        desired.

    By default, the returned table contains
    ``torch.export.default_decompositions()`` with native-registered
    overrides layered on top -- matching typical "run the usual
    decompositions, plus my overrides" intent. The override entries win
    over any same-op default, because they're merged last.

    Example:

        ep = torch.export.export(model, args)
        ep = ep.run_decompositions(
            torch._native.registry.native_decomp_table()
        )

    Args:
        overrides_only: If True, return only the native-registered overrides
            and no default aten decompositions. Useful for inspection /
            debugging, or when composing tables manually.

    Returns:
        A dict mapping ``OpOverload`` to decomposition callable.
    """
    if overrides_only:
        table: dict[object, Callable] = {}
    else:
        # Local import: keeps `import torch._native` from pulling in
        # torch.export (and its transitive imports) at module load time.
        from torch.export import default_decompositions

        table = dict(default_decompositions())
    # Merge overrides last so they win on conflicts.
    table.update(_native_decomp_overrides)
    return table


def _register_overrides_from_graph(
    op_symbol: str,
    dispatch_key: str,
    graph: list[_OverrideNode],
    *,
    filter_state: _FilterState | None = None,
) -> None:
    """
    Register all overrides in a single graph.

    Args:
        op_symbol: The operation symbol
        dispatch_key: The dispatch key
        graph: List of override nodes to register
        filter_state: Optional filter state for conditional registration
    """
    lib = _get_or_create_library(dispatch_key)

    cond_impl: list[tuple[_OpCondFn, str]] = []

    # node.node_id is minted once at `register_op_override` time and is
    # stable across reorder / deregister / reenable -- never regenerate it
    # here, since `_defined_native_ops` pins a fake kernel against each id.
    for node in graph:
        enable = True
        if filter_state:
            enable = filter_state.check_enabled(node)

        if enable:
            _register_node_impl(lib, node, dispatch_key)
            cond_impl.append((node.cond_fn, node.node_id))
            node.active = True
        else:
            node.active = False

    overload = _resolve_aten_overload(op_symbol)

    # Tear down any existing aten override so either (a) the op cleanly
    # reverts to native aten (empty cond_impl), or (b) the subsequent
    # `get_kernel` call returns the native kernel rather than a stale
    # previously-installed router.
    _destroy_aten_override(op_symbol, dispatch_key)

    # If no active conds remain for this (op, key), leave the native op
    # behavior intact and drop any decomp table entry.
    if not cond_impl:
        if overload is not None:
            _native_decomp_overrides.pop(overload, None)
        return

    # Capture the prior kernel at this (op, dispatch_key) *before* we install
    # our override. The fallback path calls it via `call_boxed`, which
    # re-enters the dispatcher with the original kernel handle -- bypassing
    # our just-registered router and avoiding the recursion that would
    # otherwise appear in aten's backward formulas (e.g. bmm's backward
    # calls bmm, which would route back to us).
    fallback_kernel = torch.library.get_kernel(f"aten::{op_symbol}", dispatch_key)

    # Build the router closures. Both share a first-match-wins loop over
    # `cond_impl`; they differ only in
    #   (a) whether cond exceptions fail loudly or silently, and
    #   (b) what to do when no cond matches.
    #
    # Eager routers run on real tensors where cond exceptions indicate a
    # genuine bug; missing a match falls back to the captured native kernel.
    #
    # Compile/export routers run under FakeTensor where some predicates are
    # undefined (e.g. _is_cow_tensor), so we swallow cond exceptions and
    # treat them as non-matches. On no-match we return NotImplemented so
    # Inductor reuses the default lowering rather than recursing.
    _NO_MATCH = object()  # sentinel; impl return values of None would be valid outputs

    def _dispatch(args, kwargs, swallow_cond_exceptions: bool):
        for cond, impl_name in cond_impl:
            try:
                matched = cond(*args, **kwargs)
            except Exception:
                if not swallow_cond_exceptions:
                    raise
                continue
            if matched:
                return getattr(torch.ops._native, impl_name)(*args, **kwargs)
        return _NO_MATCH

    def eager_router(
        keyset, *args, _fallback=fallback_kernel, _aten_overload=overload, **kwargs
    ):
        # This branch is only safe while Dynamo is actively tracing this Python
        # router. The broader compile-session flag can be true when this router
        # executes eagerly; redispatching to aten there would re-enter us.
        #
        # COW state is guarded by Dynamo's _is_cow_tensor handler but is not
        # modeled in the compiled graph. If a COW input reaches this router,
        # keep the existing eager path so COW-preserving fallback semantics are
        # maintained instead of compiling through aten and materializing it.
        if (
            torch.compiler.is_dynamo_compiling()
            and _aten_overload is not None
            and not _has_cow_tensor(*args, **kwargs)
        ):
            return _aten_overload(*args, **kwargs)

        result = _dispatch(args, kwargs, swallow_cond_exceptions=False)
        if result is _NO_MATCH:
            return _fallback.call_boxed(keyset, *args, **kwargs)
        return result

    def compile_router(*args, **kwargs):
        result = _dispatch(args, kwargs, swallow_cond_exceptions=True)
        if result is _NO_MATCH:
            return NotImplemented
        return result

    # Eager path: install a fresh aten override for this (op, key).
    _install_aten_override(op_symbol, dispatch_key, eager_router)

    # Compile / export path: record the router in our own decomp table.
    # Callers opt in via `native_decomp_table()` -- we deliberately do NOT
    # write into torch._inductor.decomposition.decompositions (see the
