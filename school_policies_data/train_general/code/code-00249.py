    # NEEDED entry is its SONAME libnccl_ep.so.0; load it by path to register
    # that SONAME for the import below.
    lib = os.path.join(ep_dir, "lib", "libnccl_ep.so")
    if os.path.isfile(lib):
        _NCCL_EP_KEEPALIVE.append(ctypes.CDLL(lib, mode=ctypes.RTLD_LOCAL))


def _import_nccl_ep() -> Any:
    # The EP bindings live in the optional torch._nccl_ep extension (USE_NCCL_EP).
    # A USE_SYSTEM_NCCL=OFF build links libnccl_ep statically and bakes its JIT
    # header paths, so the extension imports directly -- self-contained, no
    # nccl4py. A USE_SYSTEM_NCCL=ON build NEEDED-links libnccl_ep from the nccl4py
    # wheel, so the first import fails until nccl4py is set up; fall back to that
    # and retry.
    try:
        # pyrefly: ignore [missing-import]  # built only with USE_NCCL_EP
        import torch._nccl_ep as _ep

        return _ep
    except ImportError:
        pass

    _prepare_nccl4py()
    try:
        # pyrefly: ignore [missing-import]  # built only with USE_NCCL_EP
        import torch._nccl_ep as _ep
    except ImportError as e:
        raise ImportError(
            "torch._nccl_ep is unavailable; this PyTorch was not built with "
            "USE_NCCL_EP (or, for a USE_SYSTEM_NCCL=ON build, the nccl4py wheel "
            "is missing)."
        ) from e

    return _ep


@dataclass(frozen=True, slots=True)
class Routing:
    handle: object
    topk_idx: torch.Tensor


class _DispatchAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        ts: TokenSwitch,
        routing: Routing,
        tokens: torch.Tensor,
        topk_weights: torch.Tensor,
        max_recv_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _N, H = tokens.shape
        K = topk_weights.shape[1]
        out_tokens = tokens.new_zeros(max_recv_tokens, H)
        out_topk_weights = topk_weights.new_zeros(max_recv_tokens, K)
        out_topk_idx = routing.topk_idx.new_zeros(max_recv_tokens, K)
        ts._dispatch(
            routing, tokens, topk_weights, out_tokens, out_topk_weights, out_topk_idx
        )
        ctx.ts = ts
        ctx.routing = routing
        ctx.tokens_shape = tokens.shape
        return out_tokens, out_topk_weights, out_topk_idx

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(
        ctx: Any,
        grad_out_tokens: torch.Tensor,
        grad_out_topk_weights: torch.Tensor,
        grad_out_topk_idx: torch.Tensor,
    ) -> tuple[None, None, torch.Tensor, None, None]:
        grad_tokens = grad_out_tokens.new_zeros(ctx.tokens_shape)
        ctx.ts._combine(ctx.routing, grad_out_tokens.contiguous(), grad_tokens)
        return None, None, grad_tokens, None, None


class _CombineAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        ts: TokenSwitch,
        routing: Routing,
        expert_tokens: torch.Tensor,
    ) -> torch.Tensor:
        N = routing.topk_idx.shape[0]
        H = expert_tokens.shape[1]
        out_tokens = expert_tokens.new_zeros(N, H)
        ts._combine(routing, expert_tokens, out_tokens)
        ctx.ts = ts
        ctx.routing = routing
        ctx.expert_shape = expert_tokens.shape
        ctx.expert_dtype = expert_tokens.dtype
        ctx.top_k = routing.topk_idx.shape[1]
        return out_tokens

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(
        ctx: Any, grad_out_tokens: torch.Tensor
    ) -> tuple[None, None, torch.Tensor]:
        M, H = ctx.expert_shape
        N = grad_out_tokens.shape[0]
        K = ctx.top_k
        dtype = ctx.expert_dtype
        # ncclEpDispatch requires the output buffer sized to the group's
        # max_recv_tokens_per_rank, regardless of what shape expert_tokens had
        # in forward (often a slice like out_tokens[:M]). Allocate full-size,
        # run dispatch, then slice to ctx.expert_shape so the returned grad
        # matches the input that produced it.
        max_recv = ctx.ts._max_recv_tokens_per_rank
        grad_expert_full = grad_out_tokens.new_zeros(max_recv, H).to(dtype)
        dummy_weights = grad_out_tokens.new_zeros(N, K, dtype=torch.float32)
        dummy_out_weights = grad_out_tokens.new_zeros(max_recv, K, dtype=torch.float32)
        dummy_out_idx = ctx.routing.topk_idx.new_zeros(max_recv, K)
        ctx.ts._dispatch(
            ctx.routing,
            grad_out_tokens.to(dtype).contiguous(),
            dummy_weights,
            grad_expert_full,
            dummy_out_weights,
            dummy_out_idx,
        )
        return None, None, grad_expert_full[:M].contiguous()


class TokenSwitch(abc.ABC):
    """Abstract token routing switch (e.g. expert-parallel dispatch / combine).

    Typical usage: :meth:`create_routing`, then :meth:`dispatch` / :meth:`combine`.
    """

    @abc.abstractmethod
    def create_routing(
        self,
        topk_idx: torch.Tensor,
        per_expert_token_counts: torch.Tensor | None = None,
    ) -> Routing:
        """Create expert routing for the current phase (e.g. top-k indices).

        ``per_expert_token_counts`` is optional 1D int32, length >= local experts:
        output buffer for per-expert receive counts (NCCL EP ``RECV_EXPERT_COUNTER``).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _dispatch(
        self,
        routing: Routing,
        tokens: torch.Tensor,
        topk_weights: torch.Tensor,
        out_tokens: torch.Tensor,
        out_topk_weights: torch.Tensor,
        out_topk_idx: torch.Tensor,
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def _combine(
        self,
        routing: Routing,
        expert_tokens: torch.Tensor,
        out_tokens: torch.Tensor,
    ) -> None:
        raise NotImplementedError

    def dispatch(
        self,
        routing: Routing,
        tokens: torch.Tensor,
        topk_weights: torch.Tensor,
        max_recv_tokens: int | None = None,
        *,
        out: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route tokens to experts.

        With ``out=(out_tokens, out_topk_weights, out_topk_idx)``: writes to the provided
        buffers and returns them; no autograd support.
        Without ``out``: allocates output buffers and returns
        ``(out_tokens, out_topk_weights, out_topk_idx)`` with autograd support.
        ``max_recv_tokens`` is required when ``out`` is not provided.
        ``topk_weights`` receives no gradient (routing metadata).
        """
        if out is not None:
            self._dispatch(routing, tokens, topk_weights, *out)
            return out
        if max_recv_tokens is None:
            raise ValueError("max_recv_tokens is required when out= is not provided")
        return _DispatchAutograd.apply(
            self, routing, tokens, topk_weights, max_recv_tokens
        )  # type: ignore[return-value]

    def combine(
        self,
        routing: Routing,
        expert_tokens: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Gather expert outputs back to token order.

        With ``out=out_tokens``: writes to the provided buffer and returns it;
        no autograd support.
        Without ``out``: allocates an output buffer and returns it with autograd support.
        """
        if out is not None:
            self._combine(routing, expert_tokens, out)
            return out
        return _CombineAutograd.apply(self, routing, expert_tokens)  # type: ignore[return-value]


class TokenSwitchNCCL(TokenSwitch):
    """Token switch backed by NCCL EP (:func:`ncclEpCreateGroup` / dispatch / combine)."""

    def __init__(
        self,
        process_group: ProcessGroup,
        num_experts: int,
        max_dispatch_tokens_per_rank: int,
        max_recv_tokens_per_rank: int,
        max_token_bytes: int,
    ) -> None:
        self._ep = _import_nccl_ep()
        self._max_recv_tokens_per_rank = max_recv_tokens_per_rank
        self._group = self._ep._NcclEpGroup.create(
            process_group,
            num_experts,
            max_dispatch_tokens_per_rank,
            max_recv_tokens_per_rank,
            max_token_bytes,
        )

    def create_routing(
        self,
        topk_idx: torch.Tensor,
        per_expert_token_counts: torch.Tensor | None = None,
    ) -> Routing:
        """Create expert routing for this phase; pass to :meth:`dispatch` / :meth:`combine`."""
        handle = self._ep._NcclEpHandle.create(
            self._group,
            topk_idx,
            per_expert_token_counts,
        )
        return Routing(handle=handle, topk_idx=topk_idx)

    def _dispatch(
        self,
        routing: Routing,
        tokens: torch.Tensor,
        topk_weights: torch.Tensor,
        out_tokens: torch.Tensor,
        out_topk_weights: torch.Tensor,
        out_topk_idx: torch.Tensor,
    ) -> None:
        self._ep._nccl_ep_dispatch(
            routing.handle,
            tokens,
            topk_weights,
            out_tokens,
            out_topk_weights,
            out_topk_idx,
        )

    def _combine(
        self,
        routing: Routing,
        expert_tokens: torch.Tensor,
        out_tokens: torch.Tensor,
    ) -> None:
        self._ep._nccl_ep_combine(
            routing.handle,
            expert_tokens,
            out_tokens,
        )
<|endoftext|>"""
Definition of CuTe inspired Layouts for DeviceMesh internal bookkeeping and functions to manipulate them
"""

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import product
from typing import NoReturn, overload

import torch
from torch.distributed._pycute import (
    as_tuple,
    coalesce,
    complement,
    composition,
    flatten,
    IntTuple,
    is_int,
    is_tuple,
    Layout,
    make_layout,
    match_structure,
    suffix_product,
)


@dataclass(frozen=True)
class _FlatLayout:
    """
    A canonical CuTe layout for a single dimension of a DeviceMesh

    Utility class for representing an integer layout by borrowing ideas from CuTe Layout Algebra.
    See https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/02_layout_algebra.md for more details.

    Each layout is represented as a list of sizes and strides. We use it as a way for mechanical bookkeeping
    of the integers such as ranks in a SPMD mesh, and the transformation on top of it.

    Lots of methods of layout like coalesce, composition, complement, etc. are borrowed from pycute.
    https://github.com/NVIDIA/cutlass/blob/6dd13d42784ee5bfa232d2441e6b9a021c5c6290/python/pycute/layout.py#L137,L257

    Note this is a CuTe-inspired layout, because CuTe uses co-lexicographic way in linearization while PyTorch
    is using lexicographic. So even though the CuTe documentation can still be referenced, the implementation will be
    different from that of PyCute's.

    This layout is _not_ itself subdivided into multiple dimensions. It might
    internally sometimes use multidimensional tuple to represent "irregular"
    layouts (e.g., flattening non-adjacent dims), but this should be considered
    an opaque implementation detail.

    This class guarantees that all equivalent layouts are encoded as the same
    normalized representation, and thus compare equal. This is achieved by
    flattening and coalescing compatible adjacent dimensions (which includes
    removing all dimensions of size 1).

    """

    shape: tuple[int, ...]
    stride: tuple[int, ...]

    def __init__(self, shape: IntTuple, stride: IntTuple | None = None) -> None:
        if not is_tuple(shape) and not is_int(shape):
            raise TypeError(f"shape must be a tuple or int, got {type(shape)}")
        stride = stride if stride is not None else suffix_product(shape)
        if not is_tuple(stride) and not is_int(stride):
            raise TypeError(f"stride must be a tuple or int, got {type(stride)}")
        if not match_structure(shape, stride):
            raise ValueError(f"sizes {shape} and strides {stride} don't match")

        coalesced_layout = coalesce(Layout(shape, stride))
        flat_shape = flatten(coalesced_layout.shape)
        flat_stride = flatten(coalesced_layout.stride)

        # pycute will preserve a size=1 dim if it's the only remaining dim, but
        # we prefer to stick to the Tensor convention and make it 0-dimensional
        if flat_shape == (1,) and flat_stride == (0,):
            flat_shape = ()
            flat_stride = ()

        # Set attributes using object.__setattr__ since frozen=True
        object.__setattr__(self, "shape", flat_shape)
        object.__setattr__(self, "stride", flat_stride)

    def __len__(self) -> NoReturn:
        raise RuntimeError(
            "You should never need to know the length of the internal representation of a FlatLayout"
        )

    def __getitem__(self, i: int) -> NoReturn:
        raise RuntimeError(
            "You should never need to index into the internal representation of a FlatLayout"
        )

    def to_pycute(self) -> Layout:
        if not self.shape:
            return Layout(1, 0)
        return Layout(self.shape, self.stride)

    def numel(self) -> int:
        return math.prod(self.shape)

    def composition(self, layout: "_MeshLayout") -> "_MeshLayout":
        """
        By-dimension composition allows one layout to "select from" or "filter through" another layout.
        Think of it as function composition: (self ∘ layout)(input) = self(layout(input))
        between two layouts. This function is a wrapper of pycute's composition.

        Mental model about how to understand the composition logic:
        - The LEFT layout (self) defines the "output space" - what indices are possible
        - The RIGHT layout (layout parameter) acts as a "selector" - which specific indices to pick
        - The composition only generates indices that the left layout could originally produce,
          but the right layout determines which indices to be picked.
        - The stride of the composition layout will not be smaller than the stride of the right layout,
          because when picking the indices the composition will at least follow the right layout's stride
          to move forward.

        Example:
          self = (6,2):(2,1)      # sizes=(6,2), strides=(2,1)
          layout = (3:2)          # sizes=(3,), stride=(2,)
          self o layout = (3:2)

        Returns:
          Layout being composed.
        """
        result = composition(self.to_pycute(), layout.to_pycute())
        result_axes = [
            _FlatLayout(shape, stride)
            for shape, stride in zip(as_tuple(result.shape), as_tuple(result.stride))
        ]
        return _MeshLayout(result_axes)

    def complement(self, world_size: int) -> "_FlatLayout":
        """
        Compute the "complement layout" relative to a given world_size.
        A complement layout fills in the "missing" factor so that: self repeat a layout of complement(self, world_size)
        will get a complete world_size. We use ⊗ to denote the repeat operation.

        Example:
          self = (4:1)   # size=4, stride=1
          world_size = 8
          Then:
            complete needed factor = 8 / 4 = 2
            complement(self, 8) = (2:1)

          Together they form:
            (4:1) ⊗ (2:1) = (4,2):(2,1)
          which has world_size = 4 * 2 = 8, as required.

        In distributed terms, complement() is often used to derive the "other"
        rank grouping when splitting processes into 2D meshes.

        For a visualized explanation, see https://x.com/ezyang/status/1962364978393981433/
        """
        result = complement(self.to_pycute(), world_size)
        return _FlatLayout(result.shape, result.stride)

    def all_ranks_from_zero(self) -> list[int]:
        """
        This function computes the all ranks specified by the layout starting from zero.

        How it works:
        1. we enumerate every possible coordinate (like a nested for-loop).
        If sizes = (2, 3), we get the following coordinates:
            (0,0), (0,1), (0,2), (1,0), (1,1), (1,2)

        2. For each coordinate, we compute a linear rank index as:
            all_ranks_from_zero = sum(coord[i] * strides[i] for i in range(ndim))

        Example A:
        sizes = (2, 3)        # 2 rows, 3 cols
        strides = (3, 1)        # row-major layout
        coords = (0,0) -> 0*3 + 0*1 = 0
                 (0,1) -> 0*3 + 1*1 = 1
                 (0,2) -> 0*3 + 2*1 = 2
                 (1,0) -> 1*3 + 0*1 = 3
                 (1,1) -> 1*3 + 1*1 = 4
                 (1,2) -> 1*3 + 2*1 = 5
        result = [0, 1, 2, 3, 4, 5]

        Example B:
        sizes = (2, 3)
        strides = (1, 2)        # non-standard / strided layout
        coords = (0,0) -> 0*1 + 0*2 = 0
                 (0,1) -> 0*1 + 1*2 = 2
                 (0,2) -> 0*1 + 2*2 = 4
                 (1,0) -> 1*1 + 0*2 = 1
                 (1,1) -> 1*1 + 1*2 = 3
                 (1,2) -> 1*1 + 2*2 = 5
        result = [0, 2, 4, 1, 3, 5]
        """
        return [
            sum(c * s for c, s in zip(coord, self.stride))
            for coord in product(*(range(s) for s in self.shape))
        ]

    def global_ranks(self, world_size: int) -> list[list[int]]:
        """
        Build global ranks specified by the layout via two-level ranks composition.

        The nested list forms the Cartesian product of all ranks for one layout and offset
        regarding filling up the world_size with the layout.
        The final global ranks are the addition of these two. The result is a
        list of lists: one sublist per layout. This rank list will be used to build
        the communicator underlying the layout and the given `world_size`.

        Example:
        world_size = 16
        self.size = 4
        self.stride = 1
        ranks = [0, 1, 2, 3]
        offsets = [0, 4, 8, 12]
        result = [
            [0+0, 0+1, 0+2, 0+3],  # → [0, 1, 2, 3]
            [4+0, 4+1, 4+2, 4+3],  # → [4, 5, 6, 7]
            [8+0, 8+1, 8+2, 8+3],  # → [8, 9, 10,11]
            [12+0, 12+1, 12+2, 12+3],  # → [12,13,14,15]
        ]
        """
        return [
            [offset + rank for rank in self.all_ranks_from_zero()]
            for offset in self.complement(world_size).all_ranks_from_zero()
        ]

    def check_sorted(self) -> bool:
        return tuple(sorted(self.stride, reverse=True)) == self.stride

    def check_orthogonal(self) -> bool:
        """
        Check if the layout has any overlap between the ranks it generates. If there is overlap,
        we return False, otherwise True.

        The layout is supposed to be injective i.e, aside from indice 0, indices from each
        dim of the layout must be non-overlapping.

        Example 1 - Valid (no overlap):
        Layout: sizes=(2,3), strides=(6,1)
        - Dim 1: stride=1, span=3*1=3, covers indices [0,1,2]
        - Dim 0: stride=6, span=2*6=12, covers indices [0,6]
        → No overlap since 6 > 3

        Example 2 - Invalid (overlap):
        Layout: sizes=(2,3), strides=(2,1)
        - Dim 1: stride=1, span=3*1=3, covers indices [0,1,2]
        - Dim 0: stride=2, span=2*2=4, covers indices [0,2]
        → Overlap! stride=2 < span=3, so indices [0,2] are duplicated

        Example 3 - Invalid (overlap):
        Layout: sizes=(4,2), strides=(1,1)
        - Dim 1: stride=1, span=4, covers indices [0,1,2,3]
        - Dim 0: stride=1, span=2, covers indices [0,1]
        → Overlap! stride is same for two dims, so indices [0,2] are duplicated

        Returns:
            bool: True if no overlap, False if overlap detected
        """
        if len(self.shape) < 2:
            return True
        stride, shape = zip(*sorted(zip(self.stride, self.shape), reverse=True))
        return all(
            stride[i] % (stride[i + 1] * shape[i + 1]) == 0
            for i in range(len(stride) - 1)
        )

    @property
    def sizes_and_strides(self) -> Iterator[tuple[int, int]]:
        """Iterate over (size, stride) pairs for each dimension."""
        return zip(self.shape, self.stride)


@dataclass(frozen=True)
class _MeshLayout(Sequence[_FlatLayout]):
    """
    A multi-dimensional structure consisting of a series of dimension-less layouts

    This class represents the layout of a full DeviceMesh, where the overall
    top-level ndim and "logical" shape are well defined, but each individual
    mesh axis is squashed and normalized into a canonical _FlatLayout.

    It only contains methods that need to make use of this multi-dimensional
    structure (i.e., which access the ndim or the top-level sizes). Everything
    else should go on _FlatLayout and accessed by first calling .collapse().

    """

    axes: tuple[_FlatLayout, ...]

    def __init__(self, axes: Sequence[_FlatLayout]) -> None:
        object.__setattr__(self, "axes", tuple(axes))

    @classmethod
    def from_sizes_strides(
        cls, sizes: tuple[int, ...], strides: tuple[int, ...] | None = None
    ) -> "_MeshLayout":
        if strides is None:
            strides = flatten(suffix_product(sizes))
        if len(sizes) != len(strides):
            raise ValueError(
                f"sizes and strides must have the same length, got {len(sizes)} and {len(strides)}"
            )
        axes = tuple(_FlatLayout((s,), (d,)) for s, d in zip(sizes, strides))
        return cls(axes)

    def __len__(self) -> int:
        return len(self.axes)

    @overload
    def __getitem__(self, i: int) -> _FlatLayout: ...

    @overload
    def __getitem__(self, i: slice) -> "_MeshLayout": ...

    def __getitem__(self, i: int | slice) -> "_FlatLayout | _MeshLayout":
        if isinstance(i, slice):
            return _MeshLayout(self.axes[i])
        return self.axes[i]

    def __iter__(self) -> Iterator[_FlatLayout]:
        return iter(self.axes)

    def to_pycute(self) -> Layout:
        if len(self.axes) == 0:
            return Layout(1, 0)
        return make_layout(*(axis.to_pycute() for axis in self.axes))

    @property
    def top_level_sizes(self) -> tuple[int, ...]:
        return tuple(axis.numel() for axis in self.axes)

    def numel(self) -> int:
        return math.prod(self.top_level_sizes)

    def cosize(self) -> int:
        return self.to_pycute().cosize()

    def collapse(self) -> _FlatLayout:
        """
        Merge all axes into a single _FlatLayout.

        This is used to "forget" the multi-dimensional structure of this object
        and recover a "flat" (and coalesced) representation.
        """
        shapes = tuple(axis.shape for axis in self.axes)
        strides = tuple(axis.stride for axis in self.axes)
        return _FlatLayout(shapes, strides)

    def splice(self, start: int, end: int, layout: "_MeshLayout") -> "_MeshLayout":
        """
        Replace (out-of-place) the start:end slice with the given list of layouts

        Returns the concatenation of self[:start] + layout + self[end:].
        """
        new_axes = list(self.axes)
        new_axes[start:end] = list(layout.axes)
        return _MeshLayout(new_axes)

    def remap_to_tensor(self, rank_map: torch.Tensor) -> torch.Tensor:
        """
        Leverage layout as an index for mesh tensor that re-maps the indexes after layout
        transformation to actual device ranks.

        With this method, the cute layout serves as the backend of indices bookkeeping for the
        mesh tensor when it comes to flatten, unflatten and slicing operations. The actual mesh
        tensor still represents the actual device assignment and ranks. We need this function
        to specify device allocation and create backend for a mesh. Although any transform of mesh tensors
        can be treated as a view or subset of mesh tensor, we do need to use the actual view or
        sub-tensor for DeviceMesh and its backend creation.

        The shape of the `rank_map` must be 1D and contiguous.

        Examples:

        Case 1 - Consecutive ranks, full world:
            original_mesh_tensor = [[0,1],[2,3]]  # 2x2 mesh, ranks 0-3
            world_size = 4
            layout = Layout(2:2)
            Return: [[0,2],[1,3]]

        Case 2 - Non-consecutive ranks:
            original_mesh_tensor = [[10,20],[30,40]]  # custom rank assignment
            world_size = 4
            layout = Layout(2:2)
            Return: [[[10,30],[20,40]]]

        Args:
            rank_map: The concrete mesh tensor with actual device ranks

        Returns:
            torch.Tensor: A tensor representing the actual device allocation from rank_map
        """
        if rank_map.ndim != 1:
            raise AssertionError
        if not rank_map.is_contiguous():
            raise AssertionError
        if rank_map.numel() < self.cosize():
            raise AssertionError

        self_layout = self.collapse()
        complement_layout = self_layout.complement(rank_map.numel())

        return rank_map.as_strided(
            complement_layout.shape + self_layout.shape,
            complement_layout.stride + self_layout.stride,
        ).reshape(-1, *self.top_level_sizes)
<|endoftext|>#!/usr/bin/env python3
# mypy: allow-untyped-defs

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Module ``torch.distributed.run``.

``torch.distributed.run`` is a module that spawns up multiple distributed
training processes on each of the training nodes.

``torchrun`` is a python
`console script <https://packaging.python.org/en/latest/specifications/entry-points/#use-for-scripts>`_
to the main module
`torch.distributed.run <https://github.com/pytorch/pytorch/blob/master/torch/distributed/run.py>`_
declared in the ``entry_points`` configuration in
`setup.py <https://github.com/pytorch/pytorch/blob/master/setup.py>`_.
It is equivalent to invoking ``python -m torch.distributed.run``.

``torchrun`` can be used for single-node distributed training, in which one or
more processes per node will be spawned. It can be used for either
CPU training or GPU training. If it is used for GPU training,
each distributed process will be operating on a single GPU. This can achieve
well-improved single-node training performance. ``torchrun`` can also be used in
multi-node distributed training, by spawning up multiple processes on each node
for well-improved multi-node distributed training performance as well.
This will especially be beneficial for systems with multiple Infiniband
interfaces that have direct-GPU support, since all of them can be utilized for
aggregated communication bandwidth.

In both cases of single-node distributed training or multi-node distributed
training, ``torchrun`` will launch the given number of processes per node
(``--nproc-per-node``). If used for GPU training, this number needs to be less
or equal to the number of GPUs on the current system (``nproc_per_node``),
and each process will be operating on a single GPU from *GPU 0 to
GPU (nproc_per_node - 1)*.

.. versionchanged:: 2.0.0

    ``torchrun`` will pass the ``--local-rank=<rank>`` argument to your script.
    From PyTorch 2.0.0 onwards, the dashed ``--local-rank`` is preferred over the
    previously used underscored ``--local_rank``.

    For backward compatibility, it may be necessary for users to handle both
    cases in their argument parsing code. This means including both ``"--local-rank"``
    and ``"--local_rank"`` in the argument parser. If only ``"--local_rank"`` is
    provided, ``torchrun`` will trigger an error: "error: unrecognized arguments:
    --local-rank=<rank>". For training code that only supports PyTorch 2.0.0+,
    including ``"--local-rank"`` should be sufficient.

    ::

        >>> # xdoctest: +SKIP
        >>> import argparse
        >>> parser = argparse.ArgumentParser()
        >>> parser.add_argument("--local-rank", "--local_rank", type=int)
        >>> args = parser.parse_args()

Usage
-----

Single-node multi-worker
++++++++++++++++++++++++

::

    torchrun
        --standalone
        --nnodes=1
        --nproc-per-node=$NUM_TRAINERS
        YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)

.. note:: ``--nproc-per-node`` may be
          ``"gpu"`` (spawn one process per GPU),
          ``"cpu"`` (spawn one process per CPU),
          ``"xpu"`` (spawn one process per XPU),
          ``"auto"`` (equivalent to ``"gpu"`` if CUDA is available,
          else equivalent to ``"xpu"`` if XPU is available,
          else equivalent to ``"cpu"``),
          or an integer specifying the number of processes.
          See `torch.distributed.run.determine_local_world_size
          <https://github.com/pytorch/pytorch/blob/0a94bb432ed75cc2d950d81b2921363218a7e459/torch/distributed/run.py#L673-L716>`_
          for more details.

Stacked single-node multi-worker
++++++++++++++++++++++++++++++++

To run multiple instances (separate jobs) of single-node, multi-worker on the
same host, we need to make sure that each instance (job) is
setup on different ports to avoid port conflicts (or worse, two jobs being merged
as a single job). To do this you have to run with ``--rdzv-backend=c10d``
and specify a different port by setting ``--rdzv-endpoint=localhost:$PORT_k``.
For ``--nodes=1``, its often convenient to let ``torchrun`` pick a free random
port automatically instead of manually assigning different ports for each run.

::

    torchrun
        --rdzv-backend=c10d
        --rdzv-endpoint=localhost:0
        --nnodes=1
        --nproc-per-node=$NUM_TRAINERS
        YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)


Fault tolerant (fixed sized number of workers, no elasticity, tolerates 3 failures)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

::

    torchrun
        --nnodes=$NUM_NODES
        --nproc-per-node=$NUM_TRAINERS
        --max-restarts=3
        --rdzv-id=$JOB_ID
        --rdzv-backend=c10d
        --rdzv-endpoint=$HOST_NODE_ADDR
        YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)

``HOST_NODE_ADDR``, in form <host>[:<port>] (e.g. node1.example.com:29400), specifies the node and
the port on which the C10d rendezvous backend should be instantiated and hosted. It can be any
node in your training cluster, but ideally you should pick a node that has a high bandwidth.

.. note::
   If no port number is specified ``HOST_NODE_ADDR`` defaults to 29400.

Elastic (``min=1``, ``max=4``, tolerates up to 3 membership changes or failures)
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

::

    torchrun
        --nnodes=1:4
        --nproc-per-node=$NUM_TRAINERS
        --max-restarts=3
        --rdzv-id=$JOB_ID
        --rdzv-backend=c10d
        --rdzv-endpoint=$HOST_NODE_ADDR
        YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)

``HOST_NODE_ADDR``, in form <host>[:<port>] (e.g. node1.example.com:29400), specifies the node and
the port on which the C10d rendezvous backend should be instantiated and hosted. It can be any
node in your training cluster, but ideally you should pick a node that has a high bandwidth.

.. note::
   If no port number is specified ``HOST_NODE_ADDR`` defaults to 29400.

Shell completion
----------------

``torchrun`` can emit a completion script for ``bash``, ``zsh`` or ``tcsh``. The
script is generated from the argument parser, so it stays in sync with the
options above. This requires the optional `shtab <https://tqdm.github.io/shtab>`_
package (``pip install shtab``); torchrun only imports it when the flag is used.

::

    # zsh
    torchrun --print-completion zsh > ~/.zsh/completions/_torchrun

    # bash
    torchrun --print-completion bash > ~/.local/share/bash-completion/completions/torchrun

Refer to your shell's documentation for the directory it loads completions from;
the paths above are the common defaults.

Note on rendezvous backend
--------------------------

For multi-node training you need to specify:

1. ``--rdzv-id``: A unique job id (shared by all nodes participating in the job)
2. ``--rdzv-backend``: An implementation of
   :py:class:`torch.distributed.elastic.rendezvous.RendezvousHandler`
3. ``--rdzv-endpoint``: The endpoint where the rendezvous backend is running; usually in form
   ``host:port``.

Currently ``c10d`` (recommended), ``etcd-v2``, and ``etcd`` (legacy)  rendezvous backends are
supported out of the box. To use ``etcd-v2`` or ``etcd``, setup an etcd server with the ``v2`` api
enabled (e.g. ``--enable-v2``).

.. warning::
   ``etcd-v2`` and ``etcd`` rendezvous use etcd API v2. You MUST enable the v2 API on the etcd
   server. Our tests use etcd v3.4.3.

.. warning::
   For etcd-based rendezvous we recommend using ``etcd-v2`` over ``etcd`` which is functionally
   equivalent, but uses a revised implementation. ``etcd`` is in maintenance mode and will be
   removed in a future version.

Definitions
-----------

1. ``Node`` - A physical instance or a container; maps to the unit that the job manager works with.

2. ``Worker`` - A worker in the context of distributed training.

3. ``WorkerGroup`` - The set of workers that execute the same function (e.g. trainers).

4. ``LocalWorkerGroup`` - A subset of the workers in the worker group running on the same node.

5. ``RANK`` - The rank of the worker within a worker group.

6. ``WORLD_SIZE`` - The total number of workers in a worker group.

7. ``LOCAL_RANK`` - The rank of the worker within a local worker group.

8. ``LOCAL_WORLD_SIZE`` - The size of the local worker group.

9. ``rdzv_id`` - A user-defined id that uniquely identifies the worker group for a job. This id is
   used by each node to join as a member of a particular worker group.

9. ``rdzv_backend`` - The backend of the rendezvous (e.g. ``c10d``). This is typically a strongly
   consistent key-value store.

10. ``rdzv_endpoint`` - The rendezvous backend endpoint; usually in form ``<host>:<port>``.

A ``Node`` runs ``LOCAL_WORLD_SIZE`` workers which comprise a ``LocalWorkerGroup``. The union of
all ``LocalWorkerGroups`` in the nodes in the job comprise the ``WorkerGroup``.

Environment Variables
---------------------

The following environment variables are made available to you in your script:

1. ``LOCAL_RANK`` -  The local rank.

2. ``RANK`` -  The global rank.

3. ``GROUP_RANK`` - The rank of the worker group. A number between 0 and ``max_nnodes``. When
   running a single worker group per node, this is the rank of the node.

4. ``ROLE_RANK`` -  The rank of the worker across all the workers that have the same role. The role
   of the worker is specified in the ``WorkerSpec``.

5. ``LOCAL_WORLD_SIZE`` - The local world size (e.g. number of workers running locally); equals to
   ``--nproc-per-node`` specified on ``torchrun``.

6. ``WORLD_SIZE`` - The world size (total number of workers in the job).

7. ``ROLE_WORLD_SIZE`` - The total number of workers that was launched with the same role specified
   in ``WorkerSpec``.

8. ``MASTER_ADDR`` - The FQDN of the host that is running worker with rank 0; used to initialize
   the Torch Distributed backend.

9. ``MASTER_PORT`` - The port on the ``MASTER_ADDR`` that can be used to host the C10d TCP store.

10. ``TORCHELASTIC_RESTART_COUNT`` - The number of worker group restarts so far.

11. ``TORCHELASTIC_MAX_RESTARTS`` - The configured maximum number of restarts.

12. ``TORCHELASTIC_RUN_ID`` - Equal to the rendezvous ``run_id`` (e.g. unique job id).

13. ``PYTHON_EXEC`` - System executable override. If provided, the python user script will
    use the value of ``PYTHON_EXEC`` as executable. The `sys.executable` is used by default.

Logging
-------

By default each worker's ``stdout``/``stderr`` go to the console unchanged, which
interleaves the output of every rank with no way to tell them apart.

``--redirects`` writes the streams to log files under ``--log-dir`` instead of the
console; ``--tee`` writes them to log files *and* echoes them to the console. Both
take the same format: a single value applies to all workers (``3`` for both
streams, ``1`` for stdout, ``2`` for stderr), or a per-local-rank mapping such as
``0:1,1:2``. For example ``--tee 3`` tees both streams for every worker.

Tee'd console lines are prefixed with ``[${role_name}${local_rank}]:`` (e.g.
``[default3]: foobar``). Use ``--log-line-prefix-template`` to change that; the
macros ``${role_name}``, ``${local_rank}``, ``${rank}`` and ``${hostname}`` are
substituted per worker. ``${hostname}`` is the name of the node the worker runs
on, which is what identifies the offending host in a multi-node job::

    torchrun --nnodes 2 --nproc-per-node 8 --tee 3 \
             --log-line-prefix-template "${hostname}:${rank}: " train.py

    r12i0n8:3: python: src/psm2_nccl_net.c:756: Assertion `r->used' failed.
    r12i0n8:3: Fatal Python error: Segmentation fault

The template may also be set with the ``TORCHELASTIC_LOG_LINE_PREFIX_TEMPLATE``
environment variable; the command line option takes precedence.
``--local-ranks-filter`` restricts which ranks reach the console, without
affecting the log files written by ``--redirects``/``--tee``.

Deployment
----------

1. (Not needed for the C10d backend) Start the rendezvous backend server and get the endpoint (to be
   passed as ``--rdzv-endpoint`` to ``torchrun``)

2. Single-node multi-worker: Start ``torchrun`` on the host to start the agent process which
   creates and monitors a local worker group.

3. Multi-node multi-worker: Start ``torchrun`` with the same arguments on all the nodes
   participating in training.

When using a job/cluster manager, the entry point command to the multi-node job should be ``torchrun``.

Failure Modes
-------------

1. Worker failure: For a training job with ``n`` workers, if ``k<=n`` workers fail all workers
   are stopped and restarted up to ``max_restarts``.

2. Agent failure: An agent failure results in a local worker group failure. It is up to the job
   manager to fail the entire job (gang semantics) or attempt to replace the node. Both behaviors
   are supported by the agent.

3. Node failure: Same as agent failure.

Membership Changes
------------------

1. Node departure (scale-down): The agent is notified of the departure, all existing workers are
   stopped, a new ``WorkerGroup`` is formed, and all workers are started with a new ``RANK`` and
   ``WORLD_SIZE``.

2. Node arrival (scale-up): The new node is admitted to the job, all existing workers are stopped,
   a new ``WorkerGroup`` is formed, and all workers are started with a new ``RANK`` and
   ``WORLD_SIZE``.

NUMA Binding
------------

On multi-GPU systems with NUMA (Non-Uniform Memory Access) architecture, you can improve
performance by binding worker processes to CPUs near their assigned GPUs. Use the
``--numa-binding`` flag:

::

    torchrun --numa-binding=node --nproc-per-node=8 train.py

See :ref:`numa-api` for more details.

Important Notices
-----------------

1. This utility and multi-process distributed (single-node or
   multi-node) GPU training currently only achieves the best performance using
   the NCCL distributed backend. Thus NCCL backend is the recommended backend to
   use for GPU training.

2. The environment variables necessary to initialize a Torch process group are provided to you by
   this module, no need for you to pass ``RANK`` manually.  To initialize a process group in your
   training script, simply run:

::

    >>> # xdoctest: +SKIP("stub")
    >>> import torch.distributed as dist
    >>> dist.init_process_group(backend="gloo|nccl")

3. In your training program, you can either use regular distributed functions
   or use :func:`torch.nn.parallel.DistributedDataParallel` module. If your
   training program uses GPUs for training and you would like to use
   :func:`torch.nn.parallel.DistributedDataParallel` module,
   here is how to configure it.

::

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank
    )

Please ensure that ``device_ids`` argument is set to be the only GPU device id
that your code will be operating on. This is generally the local rank of the
process. In other words, the ``device_ids`` needs to be ``[int(os.environ("LOCAL_RANK"))]``,
and ``output_device`` needs to be ``int(os.environ("LOCAL_RANK"))`` in order to use this
utility


4. On failures or membership changes ALL surviving workers are killed immediately. Make sure to
   checkpoint your progress. The frequency of checkpoints should depend on your job's tolerance
   for lost work.

5. This module only supports homogeneous ``LOCAL_WORLD_SIZE``. That is, it is assumed that all
   nodes run the same number of local workers (per role).

6. ``RANK`` is NOT stable. Between restarts, the local workers on a node can be assigned a
   different range of ranks than before. NEVER hard code any assumptions about the stable-ness of
   ranks or some correlation between ``RANK`` and ``LOCAL_RANK``.

7. When using elasticity (``min_size!=max_size``) DO NOT hard code assumptions about
   ``WORLD_SIZE`` as the world size can change as nodes are allowed to leave and join.

8. It is recommended for your script to have the following structure:

::

    def main():
        load_checkpoint(checkpoint_path)
        initialize()
        train()


    def train():
        for batch in iter(dataset):
            train_step(batch)

            if should_checkpoint:
                save_checkpoint(checkpoint_path)

9. (Recommended) On worker errors, this tool will summarize the details of the error
   (e.g. time, rank, host, pid, traceback, etc). On each node, the first error (by timestamp)
   is heuristically reported as the "Root Cause" error. To get tracebacks as part of this
   error summary print out, you must decorate your main entrypoint function in your
   training script as shown in the example below. If not decorated, then the summary
   will not include the traceback of the exception and will only contain the exitcode.
   For details on torchelastic error handling see: https://pytorch.org/docs/stable/elastic/errors.html

::

    from torch.distributed.elastic.multiprocessing.errors import record


    @record
    def main():
        # do train
        pass


    if __name__ == "__main__":
        main()
"""

import os
import sys
import uuid
from argparse import Action as _Action, ArgumentParser, REMAINDER
from collections.abc import Callable
from importlib import metadata

import torch
from torch.distributed.argparse_util import check_env, env
from torch.distributed.elastic.multiprocessing import DefaultLogsSpecs, LogsSpecs, Std
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.elastic.rendezvous.utils import _parse_rendezvous_config
from torch.distributed.elastic.utils import macros
from torch.distributed.elastic.utils.logging import get_logger
from torch.distributed.launcher.api import elastic_launch, LaunchConfig
from torch.numa.binding import (
    AffinityMode as _AffinityMode,  # Signify as private with _
    NumaOptions as _NumaOptions,
)
from torch.utils.backend_registration import _get_custom_mod_func


logger = get_logger(__name__)


class _PrintCompletionAction(_Action):
    """Print a shell completion script for ``torchrun`` and exit.

    ``shtab`` derives the script from this parser, so completions stay in sync
    with the options defined below. It is an optional dependency: torchrun only
    imports it when this flag is used.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        try:
            import shtab
        except ImportError:
            parser.exit(
                1,
                f"{option_string} requires the 'shtab' package: pip install shtab\n",
            )
        else:
            # Completions are for the `torchrun` console script. parser.prog is
            # `__main__.py` when invoked as `python -m torch.distributed.run`,
            # which would bind the completion to the wrong command.
            parser.prog = "torchrun"
            print(shtab.complete(parser, shell=values))
            parser.exit()


def get_args_parser() -> ArgumentParser:
    """Parse the command line options."""
    parser = ArgumentParser(description="Torch Distributed Elastic Training Launcher")

    def comma_separated_list(value):
        placeholder = "<COMMA_PLACEHOLDER>"
        value = value.replace(",,", placeholder)
        items = value.split(",")
        items = [item.replace(placeholder, ",") for item in items]
        return items

    #
    # Worker/node size related arguments.
    #

    parser.add_argument(
        "--nnodes",
        action=env,
        type=str,
        default="1:1",
        help="Number of nodes, or the range of nodes in form <minimum_nodes>:<maximum_nodes>.",
    )
    parser.add_argument(
        "--nproc-per-node",
        "--nproc_per_node",
        action=env,
        type=str,
        default="1",
        help="Number of workers per node; supported values: [auto, cpu, gpu, xpu, int].",
    )

    #
    # Rendezvous related arguments
    #

    parser.add_argument(
        "--rdzv-backend",
        "--rdzv_backend",
        action=env,
        type=str,
        default="static",
        help="Rendezvous backend.",
    )
    parser.add_argument(
        "--rdzv-endpoint",
        "--rdzv_endpoint",
        action=env,
        type=str,
        default="",
        help="Rendezvous backend endpoint; usually in form <host>:<port>.",
    )
    parser.add_argument(
        "--rdzv-id",
        "--rdzv_id",
        action=env,
        type=str,
        default="none",
        help="User-defined group id.",
    )
    parser.add_argument(
        "--rdzv-conf",
        "--rdzv_conf",
        action=env,
        type=str,
        default="",
        help="Additional rendezvous configuration (<key1>=<value1>,<key2>=<value2>,...).",
    )
    parser.add_argument(
        "--standalone",
        action=check_env,
        help="Start a local standalone rendezvous backend that is represented by a C10d TCP store "
        "on a free port. Useful when launching single-node, multi-worker job. If specified "
        "--rdzv-backend, --rdzv-endpoint, --rdzv-id are auto-assigned and any explicitly set values "
        "are ignored.",
    )

    #
    # User-code launch related arguments.
    #

    parser.add_argument(
        "--max-restarts",
        "--max_restarts",
        action=env,
        type=int,
        default=0,
        help="Maximum number of worker group restarts before failing.",
    )
    parser.add_argument(
        "--monitor-interval",
        "--monitor_interval",
        action=env,
        type=float,
        default=0.1,
        help="Interval, in seconds, to monitor the state of workers.",
    )
    parser.add_argument(
        "--start-method",
        "--start_method",
        action=env,
        type=str,
        default="spawn",
        choices=["spawn", "fork", "forkserver"],
        help="Multiprocessing start method to use when creating workers.",
    )
    parser.add_argument(
        "--event-log-handler",
        "--event_log_handler",
        action=env,
        type=str,
        default="null",
        help="name of a registered event logging handler (see: https://docs.pytorch.org/docs/stable/elastic/events.html)",
    )
    parser.add_argument(
        "--role",
        action=env,
        type=str,
        default="default",
        help="User-defined role for the workers.",
    )
    parser.add_argument(
        "-m",
        "--module",
        action=check_env,
        help="Change each process to interpret the launch script as a Python module, executing "
        "with the same behavior as 'python -m'.",
    )
    parser.add_argument(
        "--no-python",
        "--no_python",
        action=check_env,
        help="Skip prepending the training script with 'python' - just execute it directly. Useful "
        "when the script is not a Python script.",
    )

    parser.add_argument(
        "--run-path",
        "--run_path",
        action=check_env,
        help="Run the training script with runpy.run_path in the same interpreter."
        " Script must be provided as an abs path (e.g. /abs/path/script.py)."
        " Takes precedence over --no-python.",
    )
    parser.add_argument(
        "--log-dir",
        "--log_dir",
        action=env,
        type=str,
        default=None,
        help="Base directory to use for log files (e.g. /var/log/torch/elastic). The same "
        "directory is reused for multiple runs (a unique job-level sub-directory is created with "
        "rdzv_id as the prefix).",
    )
    parser.add_argument(
        "-r",
        "--redirects",
        action=env,
        type=str,
        default="0",
        help="Redirect std streams into a log file in the log directory (e.g. [-r 3] redirects "
        "both stdout+stderr for all workers, [-r 0:1,1:2] redirects stdout for local rank 0 and "
        "stderr for local rank 1).",
    )
    parser.add_argument(
        "-t",
        "--tee",
        action=env,
        type=str,
        default="0",
        help="Tee std streams into a log file and also to console (see --redirects for format).",
    )

    parser.add_argument(
        "--log-line-prefix-template",
        "--log_line_prefix_template",
        action=env,
        type=str,
        default="",
        help="Template for the prefix prepended to each console log line of a tee'd stream "
        "(e.g. [--log-line-prefix-template '${hostname}:${rank}: '] renders as 'r12i0n8:3: '). "
        "Available macros: ${role_name}, ${local_rank}, ${rank}, ${hostname}. Defaults to "
        "'[${role_name}${local_rank}]:'.",
    )

    parser.add_argument(
        "--local-ranks-filter",
        "--local_ranks_filter",
        action=env,
        type=str,
        default="",
        help="Only show logs from specified ranks in console (e.g. [--local_ranks_filter=0,1,2] will "
        "only show logs from rank 0, 1 and 2). This will only apply to stdout and stderr, not to"
        "log files saved via --redirect or --tee",
    )

    parser.add_argument(
        "--duplicate-stdout-filters",
        "--duplicate_stdout_filters",
        action=env,
        type=comma_separated_list,
        default=[],
        help="Duplicates logs streamed to stdout to another specified file with a list of filters (e.g. "
        "[--duplicate_stdout_filters 'apple,orange'] will duplicate log lines matching 'apple' "
        "OR 'orange'. An empty filters list won't duplicate any lines. Use double comma to escape a comma) ",
    )

    parser.add_argument(
        "--duplicate-stderr-filters",
        "--duplicate_stderr_filters",
        action=env,
        type=comma_separated_list,
        default=[],
        help="Duplicates logs streamed to stderr to another specified file with a list of filters (e.g. "
        "[--duplicate_stdout_filters 'apple,orange'] will duplicate log lines matching 'apple' "
        "OR 'orange'. An empty filters list won't duplicate any lines. Use double comma to escape a comma) ",
    )

    #
    # Backwards compatible parameters with caffe2.distributed.launch.
    #

    parser.add_argument(
        "--node-rank",
        "--node_rank",
        type=int,
        action=env,
        default=0,
        help="Rank of the node for multi-node distributed training. It is only used for static "
        "rendezvous (i.e., when ``--rdzv-backend=static``).",
    )
    parser.add_argument(
        "--master-addr",
        "--master_addr",
        default="127.0.0.1",
        type=str,
        action=env,
        help="Address of the master node (rank 0) that only used for static rendezvous. It should "
        "be either the IP address or the hostname of rank 0. For single node multi-proc training "
        "the --master-addr can simply be 127.0.0.1; IPv6 should have the pattern "
        "`[0:0:0:0:0:0:0:1]`.",
    )
    parser.add_argument(
        "--master-port",
        "--master_port",
        default=None,
        type=int,
        action=env,
        help="Port on the master node (rank 0) to be used for communication during distributed "
        "training. It is only used for static rendezvous. Defaults to 29500.",
    )
    parser.add_argument(
        "--local-addr",
        "--local_addr",
        default=None,
        type=str,
        action=env,
        help="Address of the local node. If specified, will use the given address for connection. "
        "Else, will look up the local node address instead. Else, it will be default to local "
        "machine's FQDN.",
    )

    parser.add_argument(
        "--logs-specs",
        "--logs_specs",
        default=None,
        type=str,
        help="torchrun.logs_specs group entrypoint name, value must be type of LogsSpecs. "
        "Can be used to override custom logging behavior.",
    )

    parser.add_argument(
        "--numa-binding",
        "--numa_binding",
        type=str,
        choices=[mode.value for mode in _AffinityMode],
        default=None,
        help="Bind worker processes to CPUs near their assigned GPUs for better performance. "
        "See torch/numa/binding.py for available modes and details.",
    )

    parser.add_argument(
        "--signals-to-handle",
        "--signals_to_handle",
        action=env,
        type=str,
        default="SIGTERM,SIGINT,SIGHUP,SIGQUIT",
        help="Comma-separated list of signals to handle and forward to subprocesses. "
        "Default: SIGTERM,SIGINT,SIGHUP,SIGQUIT. "
        "Common additional signals: SIGUSR1,SIGUSR2 (used in SLURM environments).",
    )

    parser.add_argument(
        "--shutdown-timeout",
        "--shutdown_timeout",
        action=env,
        type=int,
        default=None,
        help="Time in seconds to wait for graceful shutdown of worker processes before "
        "sending SIGKILL. If not specified, uses TORCH_ELASTIC_SHUTDOWN_TIMEOUT environment "
        "variable or defaults to 30 seconds.",
    )

    parser.add_argument(
        "--virtual-local-rank",
        "--virtual_local_rank",
        action=check_env,
        help="Enable virtual local rank mode for workers. When enabled, LOCAL_RANK is set to 0 "
        "for all workers and CUDA_VISIBLE_DEVICES is adjusted so each worker accesses its "
        "assigned GPU at device index 0.",
    )

    #
    # Shell completion.
    #

    try:
        import shtab
    except ImportError:
        # shtab is optional; fall back to the shells it supports today so the
        # flag still validates its argument when shtab is not installed
        choices = ["bash", "zsh", "tcsh"]
    else:
        choices = shtab.SUPPORTED_SHELLS

    parser.add_argument(
        "--print-completion",
        "--print_completion",
        action=_PrintCompletionAction,
        choices=choices,
        help="Print a shell completion script for torchrun to stdout and exit "
        "(e.g. [--print-completion zsh > ~/.zsh/completions/_torchrun]). "
        "Requires the optional 'shtab' package.",
    )

    #
    # Positional arguments.
    #

    parser.add_argument(
        "training_script",
        type=str,
        help="Full path to the (single GPU) training program/script to be launched in parallel, "
        "followed by all the arguments for the training script.",
    )

    # Rest from the training program.
    parser.add_argument("training_script_args", nargs=REMAINDER)

    return parser


def parse_args(args):
    parser = get_args_parser()
    return parser.parse_args(args)


def parse_min_max_nnodes(nnodes: str):
    arr = nnodes.split(":")

    if len(arr) == 1:
        min_nodes = max_nodes = int(arr[0])
    elif len(arr) == 2:
        min_nodes = int(arr[0])
        max_nodes = int(arr[1])
    else:
        raise RuntimeError(f'nnodes={nnodes} is not in "MIN:MAX" format')  # noqa: E231

    return min_nodes, max_nodes


def determine_local_world_size(nproc_per_node: str):
    try:
        logger.info("Using nproc_per_node=%s.", nproc_per_node)
        return int(nproc_per_node)
    except ValueError as e:
        if nproc_per_node == "cpu":
            num_proc = torch._utils.cpu_count()
            device_type = "cpu"
        elif nproc_per_node == "gpu":
            if not torch.cuda.is_available():
                raise ValueError("Cuda is not available.") from e
            device_type = "gpu"
            num_proc = torch.cuda.device_count()
        elif nproc_per_node == "xpu":
            if not torch.xpu.is_available():
                raise ValueError("Xpu is not available.") from e
            device_type = "xpu"
            num_proc = torch.xpu.device_count()
        elif nproc_per_node == torch._C._get_privateuse1_backend_name():
            if not _get_custom_mod_func("is_available")():
                raise ValueError(f"{nproc_per_node} is not available.") from e
            device_type = nproc_per_node
            num_proc = _get_custom_mod_func("device_count")()
        elif nproc_per_node == "auto":
            if torch.accelerator.is_available():
                num_proc = torch.accelerator.device_count()
                device_type = torch.accelerator.current_accelerator().type  # type: ignore[union-attr]
            else:
                num_proc = torch._utils.cpu_count()
                device_type = "cpu"
        else:
            raise ValueError(
                f"Unsupported nproc_per_node value: {nproc_per_node}"
            ) from e

        logger.info(
            "Using nproc_per_node=%s, setting nproc_per_node to %s since the instance has %s %s",
            nproc_per_node,
            num_proc,
            num_proc,
            device_type,
        )
        return num_proc


def get_rdzv_endpoint(args):
    if args.rdzv_backend == "static" and not args.rdzv_endpoint:
        return f"{args.master_addr}:{args.master_port}"  # noqa: E231
    return args.rdzv_endpoint


def get_use_env(args) -> bool:
    """
    Retrieve ``use_env`` from the args.

    ``use_env`` is a legacy argument, if ``use_env`` is False, the
    ``--node-rank`` argument will be transferred to all worker processes.
    ``use_env`` is only used by the ``torch.distributed.launch`` and will
    be deprecated in future releases.
    """
    if not hasattr(args, "use_env"):
        return True
    return args.use_env


def _get_logs_specs_class(logs_specs_name: str | None) -> type[LogsSpecs]:
    """
    Attempts to load `torchrun.logs_spec` entrypoint with key of `logs_specs_name` param.
    Provides plugin mechanism to provide custom implementation of LogsSpecs.

    Returns `DefaultLogsSpecs` when logs_spec_name is None.
    Raises ValueError when entrypoint for `logs_spec_name` can't be found in entrypoints.
    """
    logs_specs_cls = None
    if logs_specs_name is not None:
        eps = metadata.entry_points()
        group = eps.select(group="torchrun.logs_specs")
        if group.select(name=logs_specs_name):
            # pyrefly: ignore [bad-index]
            logs_specs_cls = group[logs_specs_name].load()

        if logs_specs_cls is None:
            raise ValueError(
                f"Could not find entrypoint under 'torchrun.logs_specs[{logs_specs_name}]' key"
            )

        logger.info(
            "Using logs_spec '%s' mapped to %s", logs_specs_name, logs_specs_cls
        )
    else:
        logs_specs_cls = DefaultLogsSpecs

    return logs_specs_cls


def config_from_args(args) -> tuple[LaunchConfig, Callable | str, list[str]]:
    # If ``args`` not passed, defaults to ``sys.argv[:1]``
    min_nodes, max_nodes = parse_min_max_nnodes(args.nnodes)
    if not (0 < min_nodes <= max_nodes):
        raise AssertionError(
            f"min_nodes must be > 0 and <= max_nodes, got min_nodes={min_nodes}, max_nodes={max_nodes}"
        )
    if args.max_restarts < 0:
        raise AssertionError("max_restarts must be >= 0")

    if (
        hasattr(args, "master_addr")
        and args.rdzv_backend != "static"
        and not args.rdzv_endpoint
    ):
        logger.warning(
            "master_addr is only used for static rdzv_backend and when rdzv_endpoint "
            "is not specified."
        )

    if (
        hasattr(args, "node_rank")
        and args.node_rank != 0
        and args.rdzv_backend != "static"
    ):
        logger.warning(
            "node_rank is only used for static rdzv_backend. It will be ignored "
            "for rdzv_backend=%s.",
            args.rdzv_backend,
        )

    nproc_per_node = determine_local_world_size(args.nproc_per_node)
    if "OMP_NUM_THREADS" not in os.environ and nproc_per_node > 1:
        omp_num_threads = 1
        logger.warning(
            "\n*****************************************\n"
            "Setting OMP_NUM_THREADS environment variable for each process to be "
            "%s in default, to avoid your system being overloaded, "
            "please further tune the variable for optimal performance in "
            "your application as needed. \n"
            "*****************************************",
            omp_num_threads,
        )
        # This env variable will be passed down to the subprocesses
        os.environ["OMP_NUM_THREADS"] = str(omp_num_threads)

    log_line_prefix_template = args.log_line_prefix_template or os.getenv(
        "TORCHELASTIC_LOG_LINE_PREFIX_TEMPLATE"
    )

    rdzv_configs = _parse_rendezvous_config(args.rdzv_conf)

    if args.rdzv_backend == "static":
        rdzv_configs["rank"] = args.node_rank

    rdzv_endpoint = get_rdzv_endpoint(args)

    ranks: set[int] | None = None
    if args.local_ranks_filter:
        try:
            ranks = set(map(int, args.local_ranks_filter.split(",")))
            if not ranks:
                raise AssertionError("ranks set cannot be empty")
        except Exception as e:
            raise ValueError(
                "--local_ranks_filter must be a comma-separated list of integers e.g. --local_ranks_filter=0,1,2"
            ) from e

    logs_specs_cls: type[LogsSpecs] = _get_logs_specs_class(args.logs_specs)

    logs_specs = logs_specs_cls(
        log_dir=args.log_dir,
        redirects=Std.from_str(args.redirects),
        tee=Std.from_str(args.tee),
        local_ranks_filter=ranks,
    )
    numa_options = (
        None
        if args.numa_binding is None
        else _NumaOptions(affinity_mode=_AffinityMode(args.numa_binding))
    )

    config = LaunchConfig(
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        nproc_per_node=nproc_per_node,
        run_id=args.rdzv_id,
        role=args.role,
        rdzv_endpoint=rdzv_endpoint,
        rdzv_backend=args.rdzv_backend,
        rdzv_configs=rdzv_configs,
        max_restarts=args.max_restarts,
        monitor_interval=args.monitor_interval,
        start_method=args.start_method,
        log_line_prefix_template=log_line_prefix_template,
        local_addr=args.local_addr,
        logs_specs=logs_specs,
        event_log_handler=args.event_log_handler,
        numa_options=numa_options,
        signals_to_handle=args.signals_to_handle,
        duplicate_stdout_filters=args.duplicate_stdout_filters,
        duplicate_stderr_filters=args.duplicate_stderr_filters,
        virtual_local_rank=args.virtual_local_rank,
        shutdown_timeout=args.shutdown_timeout,
    )

    with_python = not args.no_python
    cmd: Callable | str
    cmd_args = []
    use_env = get_use_env(args)
    if args.run_path:
        cmd = run_script_path
        cmd_args.append(args.training_script)
    else:
        if with_python:
            cmd = os.getenv("PYTHON_EXEC", sys.executable)
            cmd_args.append("-u")
            if args.module:
                cmd_args.append("-m")
            cmd_args.append(args.training_script)
        else:
            if args.module:
                raise ValueError(
                    "Don't use both the '--no-python' flag"
                    " and the '--module' flag at the same time."
                )
            cmd = args.training_script
    if not use_env:
        cmd_args.append(f"--local-rank={macros.local_rank}")
    cmd_args.extend(args.training_script_args)

    return config, cmd, cmd_args


def run_script_path(training_script: str, *training_script_args: str):
    """
    Run the provided `training_script` from within this interpreter.

    Usage: `script_as_function("/abs/path/to/script.py", "--arg1", "val1")`
    """
    import runpy
    import sys

    sys.argv = [training_script] + [*training_script_args]
    runpy.run_path(sys.argv[0], run_name="__main__")


def run(args):
    torch.multiprocessing._set_thread_name("pt_elastic")

    if args.standalone:
        args.rdzv_backend = "c10d"
        args.rdzv_endpoint = "localhost:0"
        args.rdzv_id = str(uuid.uuid4())
        logger.info(
            "\n**************************************\n"
            "Rendezvous info:\n"
            "--rdzv-backend=%s "
            "--rdzv-endpoint=%s "
            "--rdzv-id=%s\n"
            "**************************************\n",
            args.rdzv_backend,
            args.rdzv_endpoint,
            args.rdzv_id,
        )
    elif (
        args.rdzv_backend == "static"
        and not args.rdzv_endpoint
        and args.master_port is None
    ):
        _, max_nodes = parse_min_max_nnodes(args.nnodes)
        if max_nodes == 1:
            args.rdzv_backend = "c10d"
            args.rdzv_endpoint = "localhost:0"
            args.rdzv_id = str(uuid.uuid4())

    # master_port is only used for the static rendezvous backend, not c10d
    if args.master_port is None:
        args.master_port = 29500

    config, cmd, cmd_args = config_from_args(args)
    elastic_launch(
        config=config,
        entrypoint=cmd,
    )(*cmd_args)


@record
def main(args=None):
    args = parse_args(args)
    run(args)


if __name__ == "__main__":
    main()
<|endoftext|># mypy: allow-untyped-defs
try:
    from urllib.parse import urlparse, urlunparse
except ImportError as e:
    raise ImportError(
        "urllib cannot be found, urlparse from python2 is no longer supported."
    ) from e

import numbers
import os
import sys
from collections.abc import Callable, Iterator
from datetime import timedelta

from torch.distributed import FileStore, Store, TCPStore

from .constants import default_pg_timeout


_rendezvous_handlers: dict[str, Callable[..., Iterator[tuple[Store, int, int]]]] = {}

__all__ = ["register_rendezvous_handler", "rendezvous"]


def register_rendezvous_handler(scheme, handler):
    """
    Register a new rendezvous handler.

    Before we can run collective algorithms, participating processes
    need to find each other and exchange information to be able to
    communicate. We call this process rendezvous.

    The outcome of the rendezvous process is a triplet containing a
    shared key/value store, the rank of the process, and the total
    number of participating processes.

    If none of the bundled rendezvous methods apply to your execution
    environment you can opt to register your own rendezvous handler.
    Pick a unique name and use the URL scheme to identify it when
    calling the `rendezvous()` function.

    Args:
        scheme (str): URL scheme to identify your rendezvous handler.
        handler (function): Handler that is invoked when the
            `rendezvous()` function is called with a URL that uses
            the corresponding scheme. It must be a generator function
            that yields the triplet.
    """
    global _rendezvous_handlers
    if scheme in _rendezvous_handlers:
        raise RuntimeError(f"Rendezvous handler for {scheme}:// already registered")
    _rendezvous_handlers[scheme] = handler


# Query will have format "rank=0&world_size=1" and is
# converted into {"rank": 0, "world_size": 1}
def _query_to_dict(query: str) -> dict[str, str]:
    return {
        pair[0]: pair[1]
        for pair in (pair.split("=") for pair in filter(None, query.split("&")))
    }


def _get_use_libuv_from_query_dict(query_dict: dict[str, str]) -> bool:
    # libuv is the default backend for TCPStore. To enable the non-libuv backend,
    # user can explicitly specify ``use_libuv=0`` in the URL parameter.
    if sys.platform == "win32":
        #  PyTorch is built without libuv support on windows, so default to 0
        return query_dict.get("use_libuv", os.environ.get("USE_LIBUV", "0")) == "1"
    return query_dict.get("use_libuv", os.environ.get("USE_LIBUV", "1")) == "1"


def _rendezvous_helper(url: str, rank: int, world_size_opt: int | None, **kwargs):
    result = urlparse(url)
    if world_size_opt is None:
        world_size = -1
        if result.scheme == "env":
            rank = int(os.environ.get("RANK", rank))
            # If the world_size env variable is not present then it is a dynamic group
            world_size = int(os.environ.get("WORLD_SIZE", world_size))
    else:
        world_size = world_size_opt
    if rank != -1 or world_size != -1 or world_size_opt is None:
        query_dict = _query_to_dict(result.query)
        if "rank" in query_dict or "world_size" in query_dict:
            raise AssertionError(
                f"The url: {url} has node-specific arguments(rank, world_size) already."
            )
        if rank != -1:
            query_dict["rank"] = str(rank)
        if world_size != -1 or world_size_opt is None:
            query_dict["world_size"] = str(world_size)
        result = result._replace(
            query=f"{'&'.join([f'{k}={v}' for k, v in query_dict.items()])}"
        )
        # pyrefly: ignore [bad-assignment]
        url = urlunparse(result)

    if result.scheme not in _rendezvous_handlers:
        raise RuntimeError(f"No rendezvous handler for {result.scheme}://")
    return _rendezvous_handlers[result.scheme](url, **kwargs)


def rendezvous(url: str, rank: int = -1, world_size: int = -1, **kwargs):
    if not isinstance(url, (str, bytes)):
        raise RuntimeError(f"`url` must be a string. {type(url)}: {url}")

    if not isinstance(rank, numbers.Integral):
        raise RuntimeError(f"`rank` must be an integer. {rank}")

    if not isinstance(world_size, numbers.Integral):
        raise RuntimeError(f"`world_size` must be an integer. {world_size}")

    return _rendezvous_helper(url, rank, world_size, **kwargs)


def _create_store_from_options(backend_options, rank):
    store, _, _ = next(_rendezvous_helper(backend_options.init_method, rank, None))
    return store


def _rendezvous_error(msg):
    return ValueError("Error initializing torch.distributed using " + msg)


def _file_rendezvous_handler(url: str, **kwargs):
    def _error(msg):
        return _rendezvous_error("file:// rendezvous: " + msg)

    result = urlparse(url)
    path = result.path
    if sys.platform == "win32":
        import urllib.request

        full_path = result.netloc + result.path
        path = urllib.request.url2pathname(full_path)
        if path:
            # Normalizing an empty string produces ".", which is not expected.
            path = os.path.normpath(path)

    if not path:
        raise _error("path missing")
    query_dict = _query_to_dict(result.query)
    if "rank" not in query_dict:
        raise _error("rank parameter missing")
    if "world_size" not in query_dict:
        raise _error("world size parameter missing")

    rank = int(query_dict["rank"])
    world_size = int(query_dict["world_size"])
    store = FileStore(path, world_size)
    yield (store, rank, world_size)

    # If this configuration is invalidated, there is nothing we can do about it
    raise RuntimeError("Unable to perform rerendezvous using file:// method")


def _torchelastic_use_agent_store() -> bool:
    return os.environ.get("TORCHELASTIC_USE_AGENT_STORE", None) == str(True)


def _agent_store_serves(hostname, port) -> bool:
    """Whether the elastic agent is the one serving a store at ``hostname:port``.

    ``TORCHELASTIC_USE_AGENT_STORE`` only promises a server at the agent's own
    ``MASTER_ADDR:MASTER_PORT``, which the agent exports alongside it. Honoring
    the flag for any other endpoint makes every rank a client of a store nobody
    hosts. ``TCPStore``'s C++ constructor applies the same address check before
    it forgives a failed server bind.
    """
    return (
        _torchelastic_use_agent_store()
        and os.environ.get("MASTER_ADDR") == hostname
        and os.environ.get("MASTER_PORT") == str(port)
    )


def _create_c10d_store(
    hostname, port, rank, world_size, timeout, use_libuv=True
) -> Store:
    """
    Smartly creates a c10d Store object on ``rank`` based on whether we need to reuse agent store.

    The TCPStore server is assumed to be hosted
    on ``hostname:port``.

    By default, the TCPStore server uses the asynchronous implementation
    ``LibUVStoreDaemon`` which utilizes libuv.

    If ``torchelastic_use_agent_store()`` is ``True`` *and* ``hostname:port`` is
    the agent's own ``MASTER_ADDR:MASTER_PORT``, then it is assumed that the
    agent leader (node rank 0) hosts the TCPStore server there. Hence
    ALL ranks will create and return a TCPStore client (e.g. ``start_daemon=False``).

    Otherwise rank 0 will host the TCPStore (with multi-tenancy) and it is
    assumed that rank 0's hostname and port are correctly passed via
    ``hostname`` and ``port``. All non-zero ranks will create and return a
    TCPStore client.
    """
    # check if port is uint16_t
    if not 0 <= port < 2**16:
        raise ValueError(f"port must have value from 0 to 65535 but was {port}.")

    if _agent_store_serves(hostname, port):
        # We create a new TCPStore for every retry so no need to add prefix for each attempt.
        return TCPStore(
            host_name=hostname,
            port=port,
            world_size=world_size,
            is_master=False,
            timeout=timeout,
        )
    else:
        start_daemon = rank == 0
        return TCPStore(
            host_name=hostname,
            port=port,
            world_size=world_size,
            is_master=start_daemon,
            timeout=timeout,
            multi_tenant=True,
            use_libuv=use_libuv,
        )


def _tcp_rendezvous_handler(
    url: str, timeout: timedelta = default_pg_timeout, **kwargs
):
    def _error(msg):
        return _rendezvous_error("tcp:// rendezvous: " + msg)

    result = urlparse(url)
    if result.port is None:
        raise _error("port number missing")
    query_dict = _query_to_dict(result.query)
    if "rank" not in query_dict:
        raise _error("rank parameter missing")
    if "world_size" not in query_dict:
        raise _error("world size parameter missing")

    rank = int(query_dict["rank"])
    world_size = int(query_dict["world_size"])
    use_libuv = _get_use_libuv_from_query_dict(query_dict)

    if result.hostname is None:
        raise AssertionError("hostname cannot be None")

    store = _create_c10d_store(
        result.hostname, result.port, rank, world_size, timeout, use_libuv
    )

    yield (store, rank, world_size)

    # If this configuration is invalidated, there is nothing we can do about it
    raise RuntimeError("Unable to perform re-rendezvous using tcp:// method")


def _env_rendezvous_handler(
    url: str, timeout: timedelta = default_pg_timeout, **kwargs
):
    def _error(msg):
        return _rendezvous_error("env:// rendezvous: " + msg)

    def _env_error(var):
        return _error(f"environment variable {var} expected, but not set")

    def _get_env_or_raise(env_var: str) -> str:
        env_val = os.environ.get(env_var, None)
        if not env_val:
            raise _env_error(env_var)
        else:
            return env_val

    result = urlparse(url)
    query_dict = _query_to_dict(result.query)

    rank: int
    world_size: int
    master_port: int
    master_addr: str

    if "rank" in query_dict:
        rank = int(query_dict["rank"])
    else:
        rank = int(_get_env_or_raise("RANK"))

    if "world_size" in query_dict:
        world_size = int(query_dict["world_size"])
    else:
        world_size = int(_get_env_or_raise("WORLD_SIZE"))

    master_addr = _get_env_or_raise("MASTER_ADDR")
    master_port = int(_get_env_or_raise("MASTER_PORT"))
    use_libuv = _get_use_libuv_from_query_dict(query_dict)

    store = _create_c10d_store(
        master_addr, master_port, rank, world_size, timeout, use_libuv
    )

    yield (store, rank, world_size)

    # If this configuration is invalidated, there is nothing we can do about it
    raise RuntimeError("Unable to perform re-rendezvous using env:// method")


register_rendezvous_handler("tcp", _tcp_rendezvous_handler)
register_rendezvous_handler("env", _env_rendezvous_handler)
register_rendezvous_handler("file", _file_rendezvous_handler)
<|endoftext|># mypy: allow-untyped-defs
import torch

from . import _is_tensorpipe_available, constants as rpc_contants


DeviceType = int | str | torch.device

__all__ = ["TensorPipeRpcBackendOptions"]


def _to_device(device: DeviceType) -> torch.device:
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(
            "`set_devices` expect a list of CUDA devices, but got "
            f"device type {device.type}."
        )
    return device


def _to_device_map(
    device_map: dict[DeviceType, DeviceType],
) -> dict[torch.device, torch.device]:
    full_device_map: dict[torch.device, torch.device] = {}
    reverse_map: dict[torch.device, torch.device] = {}
    for k, v in device_map.items():
        k, v = torch.device(k), torch.device(v)
        if v in reverse_map:
            raise ValueError(
                "`device_map` only supports 1-to-1 mapping, "
                f"trying to map {k} and {reverse_map[v]} to {v}"
            )
        full_device_map[k] = v
        reverse_map[v] = k
    return full_device_map


def _to_device_list(devices: list[DeviceType]) -> list[torch.device]:
    return list(map(_to_device, devices))


if _is_tensorpipe_available:  # type: ignore[has-type]
    from torch._C._distributed_rpc import _TensorPipeRpcBackendOptionsBase
else:
    _TensorPipeRpcBackendOptionsBase = object  # type: ignore[assignment, misc]


# pyrefly: ignore [invalid-inheritance]
class TensorPipeRpcBackendOptions(_TensorPipeRpcBackendOptionsBase):
    r"""
    The backend options for
    :class:`~torch.distributed.rpc.TensorPipeAgent`, derived from
    :class:`~torch.distributed.rpc.RpcBackendOptions`.

    Args:
        num_worker_threads (int, optional): The number of threads in the
            thread-pool used by
            :class:`~torch.distributed.rpc.TensorPipeAgent` to execute
            requests (default: 16).
        rpc_timeout (float, optional): The default timeout, in seconds,
            for RPC requests (default: 60 seconds). If the RPC has not
            completed in this timeframe, an exception indicating so will
            be raised. Callers can override this timeout for individual
            RPCs in :meth:`~torch.distributed.rpc.rpc_sync` and
            :meth:`~torch.distributed.rpc.rpc_async` if necessary.
        init_method (str, optional): The URL to initialize the distributed
            store used for rendezvous. It takes any value accepted for the
            same argument of :meth:`~torch.distributed.init_process_group`
            (default: ``env://``).
        device_maps (Dict[str, Dict], optional): Device placement mappings from
            this worker to the callee. Key is the callee worker name and value
            the dictionary (``Dict`` of ``int``, ``str``, or ``torch.device``)
            that maps this worker's devices to the callee worker's devices.
            (default: ``None``)
        devices (List[int, str, or ``torch.device``], optional): all local
            CUDA devices used by RPC agent. By Default, it will be initialized
            to all local devices from its own ``device_maps`` and corresponding
            devices from its peers' ``device_maps``. When processing CUDA RPC
            requests, the agent will properly synchronize CUDA streams for
            all devices in this ``List``.
    """

    def __init__(
        self,
        *,
        num_worker_threads: int = rpc_contants.DEFAULT_NUM_WORKER_THREADS,
        rpc_timeout: float = rpc_contants.DEFAULT_RPC_TIMEOUT_SEC,
        init_method: str = rpc_contants.DEFAULT_INIT_METHOD,
        device_maps: dict[str, dict[DeviceType, DeviceType]] | None = None,
        devices: list[DeviceType] | None = None,
        _transports: list | None = None,
        _channels: list | None = None,
    ):
        full_device_maps = (
            {}
            if device_maps is None
            else {k: _to_device_map(v) for k, v in device_maps.items()}
        )
        full_device_list = [] if devices is None else _to_device_list(devices)
        super().__init__(
            num_worker_threads,
            _transports,
            _channels,
            rpc_timeout,
            init_method,
            full_device_maps,
            full_device_list,
        )

    def set_device_map(self, to: str, device_map: dict[DeviceType, DeviceType]):
        r"""
        Set device mapping between each RPC caller and callee pair. This
        function can be called multiple times to incrementally add
        device placement configurations.

        Args:
            to (str): Callee name.
            device_map (Dict of int, str, or torch.device): Device placement
                mappings from this worker to the callee. This map must be
                invertible.

        Example:
            >>> # xdoctest: +SKIP("distributed")
            >>> # both workers
            >>> def add(x, y):
            >>>     print(x)  # tensor([1., 1.], device='cuda:1')
            >>>     return x + y, (x + y).to(2)
            >>>
            >>> # on worker 0
            >>> options = TensorPipeRpcBackendOptions(
            >>>     num_worker_threads=8,
            >>>     device_maps={"worker1": {0: 1}}
            >>> # maps worker0's cuda:0 to worker1's cuda:1
            >>> )
            >>> options.set_device_map("worker1", {1: 2})
            >>> # maps worker0's cuda:1 to worker1's cuda:2
            >>>
            >>> rpc.init_rpc(
            >>>     "worker0",
            >>>     rank=0,
            >>>     world_size=2,
            >>>     backend=rpc.BackendType.TENSORPIPE,
            >>>     rpc_backend_options=options
            >>> )
            >>>
            >>> x = torch.ones(2)
            >>> rets = rpc.rpc_sync("worker1", add, args=(x.to(0), 1))
            >>> # The first argument will be moved to cuda:1 on worker1. When
            >>> # sending the return value back, it will follow the invert of
            >>> # the device map, and hence will be moved back to cuda:0 and
            >>> # cuda:1 on worker0
            >>> print(rets[0])  # tensor([2., 2.], device='cuda:0')
            >>> print(rets[1])  # tensor([2., 2.], device='cuda:1')
        """
        full_device_map = _to_device_map(device_map)
        curr_device_maps = super().device_maps

        if to in curr_device_maps:
            for k, v in full_device_map.items():
                if k in curr_device_maps[to] and v != curr_device_maps[to][k]:
                    raise ValueError(
                        "`set_device_map` only supports 1-to-1 mapping, trying"
                        f" to map {k} to {v} and {curr_device_maps[to][k]}"
                    )

        super()._set_device_map(to, full_device_map)

    def set_devices(self, devices: list[DeviceType]):
        r"""
        Set local devices used by the TensorPipe RPC agent. When processing
        CUDA RPC requests, the TensorPipe RPC agent will properly synchronize
        CUDA streams for all devices in this ``List``.

        Args:
            devices (List of int, str, or torch.device): local devices used by
                the TensorPipe RPC agent.
        """
        self.devices = _to_device_list(devices)
<|endoftext|># mypy: allow-untyped-defs
from functools import partial

import torch
from torch.futures import Future

from . import functions, rpc_async
from .constants import UNSET_RPC_TIMEOUT


def _local_invoke(rref, func_name, args, kwargs):
    return getattr(rref.local_value(), func_name)(*args, **kwargs)


@functions.async_execution
def _local_invoke_async_execution(rref, func_name, args, kwargs):
    return getattr(rref.local_value(), func_name)(*args, **kwargs)


def _invoke_rpc(rref, rpc_api, func_name, timeout, *args, **kwargs):
    def _rref_type_cont(rref_fut):
        rref_type = rref_fut.value()

        _invoke_func = _local_invoke
        # Bypass ScriptModules when checking for async function attribute.
        bypass_type = issubclass(rref_type, torch.jit.ScriptModule) or issubclass(
            rref_type, torch._C.ScriptModule
        )
        if not bypass_type:
            func = getattr(rref_type, func_name)
            if hasattr(func, "_wrapped_async_rpc_function"):
                _invoke_func = _local_invoke_async_execution

        return rpc_api(
            rref.owner(),
            _invoke_func,
            args=(rref, func_name, args, kwargs),
            timeout=timeout,
        )

    rref_fut = rref._get_type(timeout=timeout, blocking=False)

    if rpc_api is not rpc_async:
        rref_fut.wait()
        return _rref_type_cont(rref_fut)
    else:
        # A little explanation on this.
        # rpc_async returns a Future pointing to the return value of `func_name`, it returns a `Future[T]`
        # Calling _rref_type_cont from the `then` lambda causes Future wrapping. IOW, `then` returns a `Future[Future[T]]`
        # To address that, we return a Future that is completed with the result of the async call.
        result: Future = Future()

        def _wrap_rref_type_cont(fut):
            try:
                _rref_type_cont(fut).then(_complete_op)
            except BaseException as ex:
                result.set_exception(ex)

        def _complete_op(fut):
            try:
                result.set_result(fut.value())
            except BaseException as ex:
                result.set_exception(ex)

        rref_fut.then(_wrap_rref_type_cont)
        return result


# This class manages proxied RPC API calls for RRefs. It is entirely used from
# C++ (see python_rpc_handler.cpp).
class RRefProxy:
    def __init__(self, rref, rpc_api, timeout=UNSET_RPC_TIMEOUT):
        self.rref = rref
        self.rpc_api = rpc_api
        self.rpc_timeout = timeout

    def __getattr__(self, func_name):
        return partial(
            _invoke_rpc, self.rref, self.rpc_api, func_name, self.rpc_timeout
        )
<|endoftext|># mypy: allow-untyped-defs
import logging
import os
import threading
import warnings
from collections.abc import Generator
from datetime import timedelta
from urllib.parse import urlparse

import torch
import torch.distributed as dist


__all__ = ["is_available"]


logger = logging.getLogger(__name__)


_init_counter = 0
_init_counter_lock = threading.Lock()


def is_available() -> bool:
    return hasattr(torch._C, "_rpc_init")


if is_available() and not torch._C._rpc_init():
    raise RuntimeError("Failed to initialize torch.distributed.rpc")


if is_available():
    _is_tensorpipe_available = hasattr(
        torch._C._distributed_rpc, "_TensorPipeRpcBackendOptionsBase"
    )

    import numbers

    import torch.distributed.autograd as dist_autograd
    from torch._C._distributed_c10d import Store
    from torch._C._distributed_rpc import (
        _cleanup_python_rpc_handler,
        _DEFAULT_INIT_METHOD,
        _DEFAULT_RPC_TIMEOUT_SEC,
        _delete_all_user_and_unforked_owner_rrefs,
        _destroy_rref_context,
        _disable_jit_rref_pickle,
        _disable_server_process_global_profiler,
        _enable_jit_rref_pickle,
        _enable_server_process_global_profiler,
        _get_current_rpc_agent,
        _invoke_remote_builtin,
        _invoke_remote_python_udf,
        _invoke_remote_torchscript,
        _invoke_rpc_builtin,
        _invoke_rpc_python_udf,
        _invoke_rpc_torchscript,
        _is_current_rpc_agent_set,
        _reset_current_rpc_agent,
        _rref_context_get_debug_info,
        _set_and_start_rpc_agent,
        _set_profiler_node_id,
        _set_rpc_timeout,
        _UNSET_RPC_TIMEOUT,
        enable_gil_profiling,
        get_rpc_timeout,
        PyRRef,
        RemoteProfilerManager,
        RpcAgent,
        RpcBackendOptions,
        WorkerInfo,
    )

    if _is_tensorpipe_available:
        from torch._C._distributed_rpc import (
            _DEFAULT_NUM_WORKER_THREADS,
            _TensorPipeRpcBackendOptionsBase,
            TensorPipeAgent,
        )

    from . import api, backend_registry, functions
    from .api import *  # noqa: F403
    from .backend_registry import BackendType
    from .options import TensorPipeRpcBackendOptions
    from .server_process_global_profiler import _server_process_global_profile

    rendezvous_iterator: Generator[tuple[Store, int, int], None, None]

    __all__ += ["init_rpc", "BackendType", "TensorPipeRpcBackendOptions"]
    # pyrefly: ignore [unresolvable-dunder-all]
    __all__ = __all__ + api.__all__ + backend_registry.__all__

    def init_rpc(
        name,
        backend=None,
        rank=-1,
        world_size=None,
        rpc_backend_options=None,
    ):
        r"""
        Initializes RPC primitives such as the local RPC agent
        and distributed autograd, which immediately makes the current
        process ready to send and receive RPCs.

        Args:
            name (str): a globally unique name of this node. (e.g.,
                ``Trainer3``, ``ParameterServer2``, ``Master``, ``Worker1``)
                Name can only contain number, alphabet, underscore, colon,
                and/or dash, and must be shorter than 128 characters.
            backend (BackendType, optional): The type of RPC backend
                implementation. Supported values is
                ``BackendType.TENSORPIPE`` (the default).
                See :ref:`rpc-backends` for more information.
            rank (int): a globally unique id/rank of this node.
            world_size (int): The number of workers in the group.
            rpc_backend_options (RpcBackendOptions, optional): The options
                passed to the RpcAgent constructor. It must be an agent-specific
                subclass of :class:`~torch.distributed.rpc.RpcBackendOptions`
                and contains agent-specific initialization configurations. By
                default, for all agents, it sets the default timeout to 60
                seconds and performs the rendezvous with an underlying process
                group initialized using ``init_method = "env://"``,
                meaning that environment variables ``MASTER_ADDR`` and
                ``MASTER_PORT`` need to be set properly. See
                :ref:`rpc-backends` for more information and find which options
                are available.
        """
        torch._C._log_api_usage_once("torch.distributed.init_rpc")
        if backend is not None and not isinstance(
            backend, backend_registry.BackendType
        ):
            raise TypeError("Argument backend must be a member of BackendType")

        if rpc_backend_options is not None and not isinstance(
            rpc_backend_options, RpcBackendOptions
        ):
            raise TypeError(
                "Argument rpc_backend_options must be an instance of RpcBackendOptions"
            )

        # Try to detect the backend from the options
        if backend is None and rpc_backend_options is not None:
            for candidate_backend in BackendType:
                if isinstance(
                    rpc_backend_options,
                    type(
                        backend_registry.construct_rpc_backend_options(
                            candidate_backend
                        )
                    ),
                ):
                    backend = candidate_backend
                    break
            else:
                raise TypeError(
                    f"Could not infer backend for options {rpc_backend_options}"
                )
            # Ignore type error because mypy doesn't handle dynamically generated type objects (#4865)
            if backend != BackendType.TENSORPIPE:  # type: ignore[attr-defined]
                logger.warning(
                    "RPC was initialized with no explicit backend but with options "  # type: ignore[attr-defined]
                    "corresponding to %(backend)s, hence that backend will be used "
                    "instead of the default BackendType.TENSORPIPE. To silence this "
                    "warning pass `backend=%(backend)s` explicitly.",
                    {"backend": backend},
                )

        if backend is None:
            backend = BackendType.TENSORPIPE  # type: ignore[attr-defined]

        if rpc_backend_options is None:
            # default construct a set of RPC backend options.
            rpc_backend_options = backend_registry.construct_rpc_backend_options(
                backend
            )

        # Create store, performs rendezvous for static RPC group.
        if not world_size:
            # If world_size is not set in construction and also not set in environment variables
            # The store will be created for the dynamic group setting
            store = dist._create_store_from_options(rpc_backend_options, rank)
        else:
            # This rendezvous state sometimes is destroyed before all processes
            # finishing handshaking. To avoid that issue, we make it global to
            # keep it alive.
            global rendezvous_iterator
            rendezvous_iterator = dist.rendezvous(
                rpc_backend_options.init_method, rank=rank, world_size=world_size
            )
            store, _, _ = next(rendezvous_iterator)
        # Use same timeout as RPC.
        store.set_timeout(timedelta(seconds=rpc_backend_options.rpc_timeout))

        # Use a PrefixStore to distinguish multiple invocations.
        with _init_counter_lock:
            global _init_counter
            store = dist.PrefixStore(str(f"rpc_prefix_{_init_counter}"), store)
            _init_counter += 1

        # Initialize autograd before RPC since _init_rpc_backend guarantees all
        # processes sync via the store. If we initialize autograd after RPC,
        # there could be a race where some nodes might have initialized autograd
        # and others might not have. As a result, a node calling
        # torch.distributed.autograd.backward() would run into errors since
        # other nodes might not have been initialized.
        dist_autograd._init(rank)

        _set_profiler_node_id(rank)
        # Initialize RPC.
        _init_rpc_backend(backend, store, name, rank, world_size, rpc_backend_options)

    def _validate_rpc_args(backend, store, name, rank, world_size, rpc_backend_options):
        type_mapping = {
            backend: backend_registry.BackendType,
            store: dist.Store,
            name: str,
            rank: numbers.Integral,
            # world_size can be None for a dynamic group
            world_size: (numbers.Integral, type(None)),
            rpc_backend_options: RpcBackendOptions,
        }
        for arg, arg_type in type_mapping.items():
            if not isinstance(arg, arg_type):  # type: ignore[arg-type]
                raise RuntimeError(
                    f"Argument {arg} must be of type {arg_type} but got type {type(arg)}"
                )

    def _init_rpc_backend(
        backend=BackendType.TENSORPIPE,  # type: ignore[attr-defined]
        store=None,
        name=None,
        rank=-1,
        world_size=None,
        rpc_backend_options=None,
    ):
        _validate_rpc_args(backend, store, name, rank, world_size, rpc_backend_options)

        if _is_current_rpc_agent_set():
            raise RuntimeError("RPC is already initialized")

        # Initialize RPC.
        rpc_agent = backend_registry.init_backend(
            backend,
            store=store,
            name=name,
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )

        api._init_rpc_states(rpc_agent)

    @api._require_initialized
    def _get_debug_info():
        info = _rref_context_get_debug_info()
        info.update(api._get_current_rpc_agent().get_debug_info())
        info.update(dist_autograd._get_debug_info())
        return info
<|endoftext|>#!/usr/bin/python3
# mypy: allow-untyped-defs

import itertools

import torch

# pyrefly: ignore [deprecated]
from torch.autograd.profiler_legacy import profile

from . import (
    _disable_server_process_global_profiler,
    _enable_server_process_global_profiler,
)


__all__: list[str] = []


class _server_process_global_profile(profile):
    """
    It has the same API as ``torch.autograd.profiler.profile`` class,
    except that it enables profiling on all threads running RPC server request callbacks.

    Context manager that manages autograd profiler state and holds a summary of results.
    Under the hood it just records events of functions being executed in C++ and
    exposes those events to Python. You can wrap any code into it and it will
    only report runtime of PyTorch functions.
    Note: profiler is thread local and is automatically propagated into the async tasks

    Args:
        enabled (bool, optional): Setting this to False makes this context manager a no-op.
            Default: ``True``.

        use_cuda (bool, optional): Enables timing of CUDA events as well using the cudaEvent API.
            Adds approximately 4us of overhead to each tensor operation.
            Default: ``False``

        record_shapes (bool, optional): If shapes recording is set, information
            about input dimensions will be collected. This allows one to see which
            dimensions have been used under the hood and further group by them
            using prof.key_averages(group_by_input_shape=True). Please note that
            shape recording might skew your profiling data. It is recommended to
            use separate runs with and without shape recording to validate the timing.
            Most likely the skew will be negligible for bottom most events (in a case
            of nested function calls). But for higher level functions the total
            self cpu time might be artificially increased because of the shape
            collection.

        profile_memory (bool, optional): Whether to report memory usage, default: ``False``

    .. warning::
        Enabling memory profiling incurs additional profiler overhead

    .. warning::
        Due to some CUDA multiprocessing limitations (see :ref:`multiprocessing-cuda-note`),
        one cannot use the profiler with ``use_cuda = True`` to benchmark
        DataLoaders with ``num_workers > 0``. If you wish to benchmark data loading,
        please use ``use_cuda = False`` or ``num_workers = 0``.

    Example:
        >>> # xdoctest: +SKIP
        >>> # On worker 0:
        >>> import torch
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker0", rank=0, world_size=2)
        >>> x, y = torch.tensor(1), torch.tensor(2)
        >>> outer_profile_rref = rpc.remote(
        ...     dst_worker_name, rpc._server_process_global_profile
        ... )
        >>> outer_profile_rref.rpc_sync().__enter__()
        >>> rpc.rpc_sync(dst_worker_name, torch.add, (x, y))
        >>> inner_profile_rref = rpc.remote(
        ...     dst_worker_name, rpc._server_process_global_profile
        ... )
        >>> inner_profile_rref.rpc_sync().__enter__()
        >>> rpc.rpc_sync(dst_worker_name, torch.sub, (x, y))
        >>> inner_profile_rref.rpc_sync().__exit__(None, None, None)
        >>> outer_profile_rref.rpc_sync().__exit__(None, None, None)
        >>> print(inner_profile_rref.rpc_sync().key_averages())
        ---------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------
        Name       Self CPU total %  Self CPU total   CPU total %      CPU total        CPU time avg     Number of Calls
        ---------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------
        sub        85.06%           76.275us         100.00%          89.667us         89.667us         1
        empty      14.94%           13.392us         14.94%           13.392us         13.392us         1
        ---------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------
        Self CPU time total: 89.667us
        >>> print(outer_profile_rref.rpc_sync().key_averages())
        ---------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------
        Name       Self CPU total %  Self CPU total   CPU total %      CPU total        CPU time avg     Number of Calls
        ---------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------
        sub        35.65%           76.275us         41.91%           89.667us         89.667us         1
        empty      12.67%           27.101us         12.67%           27.101us         13.551us         2
        add        51.68%           110.550us        58.09%           124.259us        124.259us        1
        ---------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------
        Self CPU time total: 213.926us
        >>> rpc.shutdown()

        >>> # On worker 1:
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker1", rank=1, world_size=2)
        >>> # wait for worker 0 to finish work, and then shutdown.
        >>> rpc.shutdown()
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __enter__(self):
        """
        Turn on server-side process-global profiling.
        This enables thread-local profiler on all RPC threads running server-side request callbacks.
        """
        if not self.enabled:
            return

        if self.entered:  # type: ignore[has-type]
            raise RuntimeError("autograd profiler traces are not reentrant")
        self.entered = True

        profiler_kind = (
            torch.autograd.ProfilerState.CUDA
            if self.use_cuda
            else torch.autograd.ProfilerState.CPU
        )
        profiler_config = torch.autograd.ProfilerConfig(
            profiler_kind,
            self.record_shapes,
            self.profile_memory,
            False,
            False,
            False,
            torch.profiler._ExperimentalConfig(),
        )
        _enable_server_process_global_profiler(profiler_config)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Turn off server-side process-global profiling.
        Aggregate all profiling events recorded by RPC threads.

        These attributes are assigned on exiting context.

        Attributes:
            function_events (torch.autograd.profiler.EventList).  It's a list that has helper
            methods, like 1) show record items in a pretty-print table.
            2) do averaging by grouping on keys. 3) and more.

            process_global_function_events (List[torch.autograd.profiler.FunctionEvent]).
            It's a list of ``FunctionEvent`` elements. Every element is a profiling result
            of an RPC request handling within the profiling range.
        """
        if not self.enabled:
            return

        process_global_events = _disable_server_process_global_profiler()

        # Every element in this list is a thread profiling result from an RPC request handling.
        process_global_function_events = []
        for thread_local_events in process_global_events:
            # Parse from ``Event``s to ``FunctionEvent``s.
            thread_local_function_events = (
                torch.autograd.profiler_legacy._parse_legacy_records(
                    thread_local_events
                )
            )
            thread_local_function_events.sort(
                key=lambda function_event: [
                    function_event.time_range.start,
                    -(function_event.time_range.end),
                ]
            )
            process_global_function_events.append(thread_local_function_events)

        flattened_function_events = list(
            itertools.chain.from_iterable(process_global_function_events)
        )
        self.function_events = torch.autograd.profiler_util.EventList(
            flattened_function_events,
            use_device="cuda" if self.use_cuda else None,
            profile_memory=self.profile_memory,
        )
        self.function_events._build_tree()

        self.process_global_function_events = process_global_function_events

        return False
<|endoftext|># mypy: allow-untyped-defs
import collections
import copyreg
import io
import pickle
import sys
import threading
import traceback
from enum import Enum

import torch
import torch.distributed as dist
from torch._C._distributed_rpc import _get_current_rpc_agent


__all__ = ["RPCExecMode", "serialize", "deserialize", "PythonUDF", "RemoteException"]

# Thread local tensor tables to store tensors while pickling torch.Tensor
# objects
_thread_local_tensor_tables = threading.local()
_pickler = pickle.Pickler
_unpickler = pickle.Unpickler


class RPCExecMode(Enum):
    SYNC = "sync"
    ASYNC = "async"
    ASYNC_JIT = "async_jit"
    REMOTE = "remote"


class _InternalRPCPickler:
    r"""
    This class provides serialize() and deserialize() interfaces to serialize
    data to be "binary string + tensor table" format
    So for RPC python UDF function and args, non tensor data will be serialized
    into regular binary string, tensor data will be put into thread local tensor
    tables, this serialization format is consistent with builtin operator and args
    using JIT pickler. This format will make tensor handling in C++ much easier,
    e.g. attach tensor to distributed autograd graph in C++
    """

    def __init__(self):
        # Ignore type error because dispatch_table is defined in third-party package
        self._dispatch_table = copyreg.dispatch_table.copy()  # type: ignore[attr-defined]
        self._dispatch_table[torch.Tensor] = self._tensor_reducer
        # Used for registering customized picklers.
        self._class_reducer_dict = {}

    def _register_reducer(self, obj_class, reducer):
        # For the same class, only register the reducer once.
        if obj_class not in self._class_reducer_dict:
            self._class_reducer_dict[obj_class] = reducer

    @classmethod
    def _tensor_receiver(cls, tensor_index):
        global _thread_local_tensor_tables
        return _thread_local_tensor_tables.recv_tables[tensor_index]

    def _tensor_reducer(self, tensor):
        global _thread_local_tensor_tables
        _thread_local_tensor_tables.send_tables.append(tensor)
        tensor_index = len(_thread_local_tensor_tables.send_tables) - 1
        return (_InternalRPCPickler._tensor_receiver, (tensor_index,))

    @classmethod
    def _py_rref_receiver(cls, rref_fork_data):
        return dist.rpc.PyRRef._deserialize(rref_fork_data)

    def _py_rref_reducer(self, py_rref):
        rref_fork_data = py_rref._serialize()
        return (_InternalRPCPickler._py_rref_receiver, (rref_fork_data,))

    def _rref_reducer(self, rref):
        return self._py_rref_reducer(rref)

    @classmethod
    def _script_module_receiver(cls, script_module_serialized):
        """
        Given a serialized representation of a ScriptModule created with torch.jit.save,
        loads and returns the ScriptModule.
        """
        f = io.BytesIO(script_module_serialized)
        m = torch.jit.load(f)
        return m

    def _script_module_reducer(self, script_module):
        """
        Serializes a ScriptModule.
        """
        f = io.BytesIO()
        torch.jit.save(script_module, f)
        return (_InternalRPCPickler._script_module_receiver, (f.getvalue(),))

    def serialize(self, obj):
        r"""
        Serialize non tensor data into binary string, tensor data into
        tensor table
        """
        f = io.BytesIO()
        p = _pickler(f)
        p.dispatch_table = self._dispatch_table

        # rpc api could accept user picklers inheriting from _InternalRPCPickler to serialize rref,
        # user picklers could have different initialization function from _InternalRPCPickler,
        # but all the user picklers should call serialize() and use _rref_reducer to pickle rref
        # in python. also, when _internal_rpc_pickler is imported to rpc/api.py, rpc.RRef is not
        # compiled yet, it is not good place to access rpc.RRef inside _InternalRPCPickler constructor,
        # so putting rref's dispatch table here
        #
        # The return value of a `rpc.remote(..)` call is type of `rpc.PyRRef`.
        # The deserialized RRef object on an RPC receiver side is type of `rpc.PyRRef`.
        # Ignore type error because dispatch_table is defined in third-party package
        p.dispatch_table[dist.rpc.PyRRef] = self._py_rref_reducer  # type: ignore[index]
        # An RRef created locally by RRef Python constructor is type of `rpc.RRef`.
        # Ignore type error because dispatch_table is defined in third-party package
        p.dispatch_table[dist.rpc.RRef] = self._rref_reducer  # type: ignore[index]

        # Add dispatch pickling for ScriptModule or its subclass.
        if isinstance(obj, torch.jit.ScriptModule):
            # Ignore type error because dispatch_table is defined in third-party package
            p.dispatch_table[obj.__class__] = self._script_module_reducer  # type: ignore[index]

        # Install customized picklers.
        for class_name in self._class_reducer_dict:
            p.dispatch_table[class_name] = self._class_reducer_dict[class_name]  # type: ignore[index]

        # save _thread_local_tensor_tables.send_tables if it is in nested call
        global _thread_local_tensor_tables
        if hasattr(_thread_local_tensor_tables, "send_tables"):
            old_send_tables = _thread_local_tensor_tables.send_tables
        else:
            old_send_tables = None
        _thread_local_tensor_tables.send_tables = []

        p.dump(obj)

        # restore _thread_local_tensor_tables.send_tables if return
        # from nested call, otherwise clean up the table
        tensors = _thread_local_tensor_tables.send_tables
        if old_send_tables is not None:
            _thread_local_tensor_tables.send_tables = old_send_tables
        else:
            del _thread_local_tensor_tables.send_tables

        return (f.getvalue(), tensors)

    def deserialize(self, binary_data, tensor_table):
        r"""
        Deserialize binary string + tensor table to original obj
        """
        # save _thread_local_tensor_tables.recv_tables if it is in nested call
        global _thread_local_tensor_tables
        if hasattr(_thread_local_tensor_tables, "recv_tables"):
            old_recv_tables = _thread_local_tensor_tables.recv_tables
        else:
            old_recv_tables = None
        _thread_local_tensor_tables.recv_tables = tensor_table

        try:
            unpickler = _unpickler(io.BytesIO(binary_data))
            ret = unpickler.load()
        except AttributeError as e:
            # Occurs when function is not found on module/class during
            # unpickling.
            except_str = (
                str(e)
                + """ Default RPC pickler does not serialize
            function code. Ensure that UDFs are defined on both caller and
            callee modules."""
            )
            ret = AttributeError(except_str)
            # Ensure the stack trace gets preserved
            ret.__cause__ = e

        # restore _thread_local_tensor_tables.recv_tables if return
        # from nested call, otherwise clean up the table
        if old_recv_tables is not None:
            _thread_local_tensor_tables.recv_tables = old_recv_tables
        else:
            del _thread_local_tensor_tables.recv_tables

        return ret


# Create _internal_rpc_pickler only once to initialize _dispatch_table only once
_internal_rpc_pickler = _InternalRPCPickler()


def serialize(obj):
    return _internal_rpc_pickler.serialize(obj)


def deserialize(binary_data, tensor_table):
    return _internal_rpc_pickler.deserialize(binary_data, tensor_table)


def _run_function(python_udf):
    r"""
    This function is exclusively called from C++.
    See ``torch/csrc/distributed/rpc/python_rpc_handler.cpp``.

    Runs a Python UDF and returns its return value.
    Wraps any exception in ``RemoteException`` if the function raises.
    """
    try:
        if isinstance(python_udf, AttributeError):
            raise python_udf
        result = python_udf.func(*python_udf.args, **python_udf.kwargs)
    except Exception as e:
        # except str = exception info + traceback string
        except_str = (
            f"On {_get_current_rpc_agent().get_worker_info()}:\n"
            f"{repr(e)}\n{traceback.format_exc()}"
        )
        print(except_str, file=sys.stderr)
        result = RemoteException(except_str, type(e))
    return result


def _handle_exception(result):
    if isinstance(result, RemoteException):
        exception_msg = result.msg.encode("utf-8").decode("unicode_escape")
        # We wrap exception re-creation here in case some exception classes
        # cannot be constructed directly from a string.
        exc = None
        try:
            exc = result.exception_type(exception_msg)
        except BaseException as e:
            raise RuntimeError(
                f"Failed to create original exception type. Error msg was {str(e)}"
                f" Original exception on remote side was {exception_msg}"
            ) from e

        if exc is not None:
            raise exc


def _build_rpc_profiling_key(
    exec_type, func_name, current_worker_name, dst_worker_name
):
    """
    Builds the key that RPC calls are profiled with using the autograd profiler.
    This will be the name of the corresponding Event recorded in the profiler.

    Args:
        exec_type (RPCExecMode): Type of RPC/RRef call
        func_name (str): Name of function being profiled.
        current_worker_name (str): Name of current worker.
        dst_worker_name (str): Name of the destination worker.

    Returns:
        String representing profiling key
    """
    profile_key = (
        f"rpc_{exec_type.value}#{func_name}({current_worker_name} -> {dst_worker_name})"
    )
    return profile_key


def _start_record_function(exec_type, func_name, current_worker_name, dest_worker_name):
    """
    This function should be called from RPC/RRef functions to create a
    RecordFunction object for profiling. This function also runs the before
    callbacks that start the profiling, though the user is responsible for
    running the appropriate callbacks when the function to be profiled finishes.

    Args:
        exec_type (RPCExecMode): Type of RPC/RRef call
        func_name (str): Name of function being profiled.
        current_worker_name (str): Name of current worker.
        dest_worker_name (str): Name of the destination worker.

    Returns:
        An instance of `torch.autograd._RecordFunction`.
    """
    if not torch.autograd._profiler_enabled():
        raise AssertionError("Autograd profiler should be enabled.")
    profile_key = f"rpc_{exec_type.value}#{str(func_name)}({current_worker_name} -> {dest_worker_name})"
    rf = torch.autograd._RecordFunction()  # type: ignore[attr-defined]
    torch.autograd._run_before_callbacks(rf, profile_key)  # type: ignore[attr-defined]
    return rf


PythonUDF = collections.namedtuple("PythonUDF", ["func", "args", "kwargs"])
RemoteException = collections.namedtuple("RemoteException", ["msg", "exception_type"])
<|endoftext|># mypy: allow-untyped-defs
import logging
from contextlib import contextmanager
from typing import cast


logger = logging.getLogger(__name__)


@contextmanager
def _group_membership_management(store, name, is_join):
    token_key = "RpcGroupManagementToken"
    join_or_leave = "join" if is_join else "leave"
    my_token = f"Token_for_{name}_{join_or_leave}"
    while True:
        # Retrieve token from store to signal start of rank join/leave critical section
        returned = store.compare_set(token_key, "", my_token).decode()
        if returned == my_token:
            # Yield to the function this context manager wraps
            yield
            # Finished, now exit and release token
            # Update from store to signal end of rank join/leave critical section
            store.set(token_key, "")
            # Other will wait for this token to be set before they execute
            store.set(my_token, "Done")
            break
        else:
            # Store will wait for the token to be released
            try:
                store.wait([returned])
            except RuntimeError:
                logger.error(
                    "Group membership token %s timed out waiting for %s to be released.",
                    my_token,
                    returned,
                )
                raise


def _update_group_membership(worker_info, my_devices, reverse_device_map, is_join):
    from . import api, TensorPipeAgent

    agent = cast(TensorPipeAgent, api._get_current_rpc_agent())
    ret = agent._update_group_membership(
        worker_info, my_devices, reverse_device_map, is_join
    )
    return ret
<|endoftext|># mypy: allow-untyped-decorators
# mypy: allow-untyped-defs

import collections
import contextlib
import functools
import inspect
import logging
import threading
from collections.abc import Callable
from typing import Any, Generic, TYPE_CHECKING, TypeVar
from typing_extensions import ParamSpec

import torch
from torch._C._distributed_rpc import (
    _cleanup_python_rpc_handler,
    _delete_all_user_and_unforked_owner_rrefs,
    _destroy_rref_context,
    _get_current_rpc_agent,
    _invoke_remote_builtin,
    _invoke_remote_python_udf,
    _invoke_remote_torchscript,
    _invoke_rpc_builtin,
    _invoke_rpc_python_udf,
    _invoke_rpc_torchscript,
    _is_current_rpc_agent_set,
    _reset_current_rpc_agent,
    _set_and_start_rpc_agent,
    get_rpc_timeout,
    PyRRef,
    RemoteProfilerManager,
    WorkerInfo,
)
from torch.futures import Future

from ._utils import _group_membership_management, _update_group_membership
from .constants import DEFAULT_SHUTDOWN_TIMEOUT, UNSET_RPC_TIMEOUT
from .internal import (
    _build_rpc_profiling_key,
    _internal_rpc_pickler,
    PythonUDF,
    RPCExecMode,
)


_P = ParamSpec("_P")
_R = TypeVar("_R")


__all__ = [
    "shutdown",
    "get_worker_info",
    "remote",
    "rpc_sync",
    "rpc_async",
    "RRef",
    "AllGatherStates",
    "method_factory",
    "new_method",
]


logger = logging.getLogger(__name__)

# NB: Ignoring RRef leaks during shutdown. Without this, applications have to
# make sure there is no references to any RRef in the application code and
# Python GC has done its job to delete those RRefs. This is could result in bad
# debugging experiences especially when for large applications. Therefore, by
# default, we are going to ignore RRef leaks during shutdown. This is usually
# fine as shutdown means applications have done training and no longer care
# about states.
#
# To enable RRef leak checking, set this _ignore_rref_leak to False
_ignore_rref_leak = True
_default_pickler = _internal_rpc_pickler


@contextlib.contextmanager
def _use_rpc_pickler(rpc_pickler):
    r"""
    rpc_pickler: (.internal._InternalRPCPickler) Overrides the default RPC pickler
    """
    global _default_pickler
    _default_pickler = rpc_pickler
    try:
        yield
    finally:
        _default_pickler = _internal_rpc_pickler


def _require_initialized(func: Callable[_P, _R]) -> Callable[_P, _R]:
    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        if not _is_current_rpc_agent_set():
            raise RuntimeError(
                "RPC has not been initialized. Call "
                "torch.distributed.rpc.init_rpc first."
            )
        return func(*args, **kwargs)

    return wrapper


class AllGatherStates:
    def __init__(self):
        # Each `gathered_objects` is an empty dict at beginning.
        # The leader worker is elected as the first worker in a sorted worker
        # name list. Whenever there is a worker entering `_all_gather()`, it
        # runs `_gather_to_leader()` on the leader to add its own name and
        # data obj to this dict. The leader also adds itself's name to the dict
        # on calling `_all_gather()`.
        # Once `set(gathered_objects.keys()) == _ALL_WORKER_NAMES`, the leader
        # will broadcast the gathered dict to all follower workers and set their
        # `gathered_objects` field and the `proceed_signal` field.
        self.gathered_objects = {}
        # All workers wait on this signal until it receives all gathered
        # objects.
        self.proceed_signal = threading.Event()


# States used by `def _all_gather()`.
# `_ALL_WORKER_NAMES` is initialized on initializing RPC layer.
_ALL_WORKER_NAMES: set[Any] = set()
_all_gather_dict_lock = threading.RLock()
_all_gather_sequence_id: dict[str, int] = {}
_all_gather_sequence_id_to_states: collections.defaultdict = collections.defaultdict(
    AllGatherStates
)


def _init_rpc_states(agent):
    worker_infos = agent.get_worker_infos()
    global _ALL_WORKER_NAMES
    _ALL_WORKER_NAMES = {worker_info.name for worker_info in worker_infos}

    # NB: backend implementation might have already set the rpc_agent.
    if not _is_current_rpc_agent_set():
        _set_and_start_rpc_agent(agent)


def _gather_to_leader(sequence_id, worker_name, obj, worker_names=None):
    with _all_gather_dict_lock:
        if not worker_names:
            worker_names = _ALL_WORKER_NAMES
            if worker_name not in worker_names:
                raise AssertionError(f"{worker_name} is not expected by leader.")
        states = _all_gather_sequence_id_to_states[sequence_id]
        if worker_name in states.gathered_objects:
            raise AssertionError(
                f"{worker_name} reported intent sequence id {sequence_id} twice. "
            )
        states.gathered_objects[worker_name] = obj
        if worker_names == set(states.gathered_objects.keys()):
            states.proceed_signal.set()


def _broadcast_to_followers(sequence_id, objects_map):
    with _all_gather_dict_lock:
        states = _all_gather_sequence_id_to_states[sequence_id]

    if states.proceed_signal.is_set():
        raise AssertionError(
            f"Termination signal sequence id {sequence_id} got set twice."
        )
    states.gathered_objects = objects_map
    states.proceed_signal.set()


_thread_local_var = threading.local()


@contextlib.contextmanager
def _wait_all():
    r"""
    A context manager that collects all futures returned by ``rpc_async`` and
    waits them on the context manager's exit; relieving the user of needing
    to explicitly call wait.


    Example::
        >>> # xdoctest: +SKIP("distributed")
        >>> # On worker 0:
        >>> import torch
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker0", rank=0, world_size=2)
        >>> with rpc._wait_all():
        >>>    fut_1 = rpc.rpc_async(dst, torch.add, (torch.ones(2, 2), 1))
        >>>    fut_2 = rpc.rpc_async(dst, torch.add, (torch.ones(2, 2), 1))
        >>> #fut_1 and fut_2 are waited on
    """
    _thread_local_var.future_list = []
    try:
        yield
    finally:
        try:
            torch.futures.wait_all(_thread_local_var.future_list)
        finally:
            del _thread_local_var.future_list


@_require_initialized
def _all_gather(obj, worker_names=None, timeout: float = UNSET_RPC_TIMEOUT):
    r"""
    This is similar to torch.distributed.all_gather(), but is using RPC. It
    picks the worker with the smallest name (alphabetic order) as the leader.
    Then all followers send their data ``obj`` to the leader. After the leader
    has received all, it will broadcast the results back to all followers. This
    function blocks until all workers have received the gathered results.
    """
    if not worker_names:
        if _ALL_WORKER_NAMES is None:
            raise AssertionError(
                "`_ALL_WORKER_NAMES` is not initialized for `def _all_gather`."
            )
        worker_names = _ALL_WORKER_NAMES
    leader_name = min(worker_names)

    self_name = _get_current_rpc_agent().get_worker_info().name

    with _all_gather_dict_lock:
        concat_names = "".join(sorted(worker_names))
        sequence_num = _all_gather_sequence_id.get(concat_names, 0)
        _all_gather_sequence_id[concat_names] = sequence_num + 1
        sequence_id = concat_names + str(sequence_num)

    is_leader = leader_name == self_name

    if timeout == UNSET_RPC_TIMEOUT:
        # Timeout is specified by agent for RPC calls
        rpc_timeout = get_rpc_timeout()
        # No timeout for signal
        signal_timeout = None
    elif timeout == DEFAULT_SHUTDOWN_TIMEOUT:
        # No timeout for RPC
        rpc_timeout = timeout
        # No timeout for signal
        signal_timeout = None
    else:
        # Signal and RPC timeout use the same timeout
        signal_timeout = rpc_timeout = timeout

    # Phase 1: Followers send it's object to the leader
    if is_leader:
        _gather_to_leader(sequence_id, self_name, obj, worker_names)
    else:
        rpc_sync(
            leader_name,
            _gather_to_leader,
            args=(sequence_id, self_name, obj, worker_names),
            timeout=rpc_timeout,
        )

    with _all_gather_dict_lock:
        states = _all_gather_sequence_id_to_states[sequence_id]

    # Timeout is either set by function parameter or None (which is indefinite)
    states.proceed_signal.wait(timeout=signal_timeout)

    # Phase 2: Leader broadcast gathered results to all followers
    # Leader's signal is the first to be unblocked, after receiving all
    # followers' data objects.
    if is_leader:
        worker_name_to_response_future_dict = {}
        for follower_name in worker_names - {leader_name}:
            fut = rpc_async(
                follower_name,
                _broadcast_to_followers,
                args=(sequence_id, states.gathered_objects),
                timeout=rpc_timeout,
            )
            worker_name_to_response_future_dict[follower_name] = fut

        errors = []
        for follower_name, fut in worker_name_to_response_future_dict.items():
            try:
                fut.wait()
            except RuntimeError as ex:
                errors.append((follower_name, ex))

        if errors:
            raise RuntimeError(
                f"Followers {[e[0] for e in errors]} timed out in _all_gather "
                f"after {rpc_timeout:.2f} seconds. The first exception is {errors[0][1]}"
            )

    # Clean up for the states using the sequence_id
    with _all_gather_dict_lock:
        states = _all_gather_sequence_id_to_states.pop(sequence_id)
    return states.gathered_objects


@_require_initialized
def _barrier(worker_names):
    r"""
    Synchronizes local and remote RPC processes.

    This will block until all local and remote RPC processes specified under worker_names
    reach this method to wait for all outstanding work to complete.

    Args:
        worker_names (List[str]): The set of workers to synchronize.

    """
    try:
        _all_gather(None, set(worker_names))
    except RuntimeError:
        logger.exception("Failed to complete barrier")


@_require_initialized
def _wait_all_workers(timeout=DEFAULT_SHUTDOWN_TIMEOUT):
    r"""
    Block until all local and remote RPC processes reach this method and wait
    for all outstanding work to complete. Every RPC process must call this
    method before exit to perform a graceful shutdown. This should be used to
    terminate the RPC framework, and there is no guarantee that the RPC
    framework will work after this method returns.
    """
    try:
        _all_gather(None, timeout=timeout)
    except RuntimeError as ex:
        logger.exception("Failed to respond to 'Shutdown Proceed' in time")
        raise ex


@_require_initialized
def shutdown(graceful=True, timeout=DEFAULT_SHUTDOWN_TIMEOUT):
    r"""
    Perform a shutdown of the RPC agent, and then destroy the RPC agent. This
    stops the local agent from accepting outstanding requests, and shuts
    down the RPC framework by terminating all RPC threads. If ``graceful=True``,
    this will block until all local and remote RPC processes reach this method
    and wait for all outstanding work to complete. Otherwise, if
    ``graceful=False``, this is a local shutdown, and it does not wait for other
    RPC processes to reach this method.

    .. warning::
        For :class:`~torch.futures.Future` objects returned by
        :meth:`~torch.distributed.rpc.rpc_async`, ``future.wait()`` should not
        be called after ``shutdown()``.

    Args:
        graceful (bool): Whether to do a graceful shutdown or not. If True,
                         this will 1) wait until there is no pending system
                         messages for ``UserRRefs`` and delete them; 2) block
                         until all local and remote RPC processes have reached
                         this method and wait for all outstanding work to
                         complete.

    Example::
        Make sure that ``MASTER_ADDR`` and ``MASTER_PORT`` are set properly
        on both workers. Refer to :meth:`~torch.distributed.init_process_group`
        API for more details. For example,

        export MASTER_ADDR=localhost
        export MASTER_PORT=5678

        Then run the following code in two different processes:

        >>> # xdoctest: +SKIP
        >>> # On worker 0:
        >>> import torch
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker0", rank=0, world_size=2)
        >>> # do some work
        >>> result = rpc.rpc_sync("worker1", torch.add, args=(torch.ones(1), 1))
        >>> # ready to shutdown
        >>> rpc.shutdown()

        >>> # On worker 1:
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker1", rank=1, world_size=2)
        >>> # wait for worker 0 to finish work, and then shutdown.
        >>> rpc.shutdown()
    """
    if graceful:
        try:
            agent = _get_current_rpc_agent()
            from torch._C._distributed_rpc import TensorPipeAgent

            if not isinstance(agent, TensorPipeAgent) or agent.is_static_group:
                _wait_all_workers(timeout)
                _delete_all_user_and_unforked_owner_rrefs()
                agent.join(shutdown=True, timeout=timeout)
            else:
                # This is a dynamic group so we need to grab the token for the operation
                my_worker_info = agent.get_worker_info()
                my_name = my_worker_info.name
                with _group_membership_management(agent.store, my_name, False):
                    all_worker_infos = agent.get_worker_infos()
                    for worker in all_worker_infos:
                        if worker.name != my_name:
                            rpc_sync(
                                worker.name,
                                _update_group_membership,
                                args=(my_worker_info, [], {}, False),
                            )
                    agent.join(shutdown=True, timeout=timeout)
        finally:
            # In case of errors, continue to complete the local shutdown.
            _finalize_shutdown()
    else:
        _finalize_shutdown()


def _finalize_shutdown():
    try:
        # This raises a `TORCH_CHECK()` exception on RRef leak detected.
        _destroy_rref_context(_ignore_rref_leak)
    finally:
        _get_current_rpc_agent().shutdown()
        # clean up python rpc handler in shutdown(), see comments in
        # PythonRpcHandler::cleanup(), call it in python API because the
        # cleanup() function has python dependency, it assumes python
        # interpreter exists.
        # No matter if RRef leak exception is raised, this clean-up code
        # must run to avoid destruction segfault in Python 3.5.
        #
        # future.wait() should not be called after shutdown().
        # pythonRpcHandler is cleaned up in shutdown(), after
        # shutdown(), python objects returned from rpc python call can not be
        # resolved.
        _cleanup_python_rpc_handler()
        _reset_current_rpc_agent()


@_require_initialized
def get_worker_info(worker_name=None):
    r"""
    Get :class:`~torch.distributed.rpc.WorkerInfo` of a given worker name.
    Use this :class:`~torch.distributed.rpc.WorkerInfo` to avoid passing an
    expensive string on every invocation.

    Args:
        worker_name (str): the string name of a worker. If ``None``, return the
                           id of the current worker. (default ``None``)

    Returns:
        :class:`~torch.distributed.rpc.WorkerInfo` instance for the given
        ``worker_name`` or :class:`~torch.distributed.rpc.WorkerInfo` of the
        current worker if ``worker_name`` is ``None``.
    """
    if worker_name is not None:
        return _get_current_rpc_agent().get_worker_info(worker_name)
    else:
        return _get_current_rpc_agent().get_worker_info()


def _to_worker_info(to):
    if isinstance(to, WorkerInfo):
        return to
    elif isinstance(to, (str, int)):
        return get_worker_info(to)
    else:
        raise ValueError(f"Cannot get WorkerInfo from name {to}")


def _rref_typeof_on_owner(rref, blocking: bool = True):
    rref_type = type(rref.local_value())
    if blocking:
        return rref_type
    else:
        # Wrap result into a completed Future. This is so that if blocking=`False`
        # is specified, we return a future regardless of if this call is on user
        # or owner.
        future = Future[type]()
        future.set_result(rref_type)
        return future


def _rref_typeof_on_user(
    rref, timeout: float = UNSET_RPC_TIMEOUT, blocking: bool = True
):
    fut = rpc_async(rref.owner(), _rref_typeof_on_owner, args=(rref,), timeout=timeout)
    if blocking:
        return fut.wait()
    else:
        return fut


T = TypeVar("T")
# pyrefly: ignore [invalid-annotation]
GenericWithOneTypeVar = Generic[T]


if TYPE_CHECKING:

    class RRef(PyRRef[T], Generic[T]):
        pass

else:
    try:
        # Combine the implementation class and the type class.
        class RRef(PyRRef, Generic[T]):
            pass

    except TypeError:
        # TypeError: metaclass conflict: the metaclass of a derived class
        # must be a (non-strict) subclass of the metaclasses of all its bases
        # Mypy doesn't understand __class__ (mypy bug #4177)
        class RRefMeta(PyRRef.__class__, GenericWithOneTypeVar.__class__):  # type: ignore[name-defined, misc, valid-type]
            pass

        # Combine the implementation class and the type class.
        # Types for classes expecting a certain generic parameter (mypy bug #7791)
        class RRef(PyRRef, GenericWithOneTypeVar, metaclass=RRefMeta):  # type: ignore[misc, no-redef, valid-type]
            pass


# Install docstrings from `PyRRef` to `RRef`.
#
# This is for the fact that pybind11 generates the parameter
# `self` as type `rpc.PyRRef`, so a `:inherited-members:`
# under `.. autoclass:: RRef` does not work.
# we have to do the following process to replace `rpc.PyRRef` with `rpc.RRef`.
#
def method_factory(method_name, docstring):
    def method(self, *args, **kwargs):
        return getattr(super(RRef, self), method_name)(*args, **kwargs)

    if method.__doc__:
        method.__doc__ = docstring
    return method


for method_name, method in inspect.getmembers(PyRRef):
    # Ignore magic methods, except "__str__".
    if method_name.startswith("_") and method_name != "__str__":
        continue

    # Get pybind11 generated docstring.
    # It's like,
    """
    to_here(self: torch.distributed.rpc.PyRRef, timeout: float=-1.0) -> object

        Blocking call that copies the value of the RRef from the owner
        to the local node and returns it. If the current node is the
        owner, returns a reference to the local value.
    """
    docstring = getattr(method, "__doc__", None)
    if docstring is None:
        raise AssertionError("RRef user-facing methods should all have docstrings.")

    # Do surgery on pybind11 generated docstrings.
    docstring = docstring.replace(
        "torch.distributed.rpc.PyRRef", "torch.distributed.rpc.RRef"
    )

    # Attach user-facing RRef method with modified docstring.
    new_method = method_factory(method_name, docstring)
    setattr(RRef, method_name, new_method)


@_require_initialized
def remote(to, func, args=None, kwargs=None, timeout=UNSET_RPC_TIMEOUT):
    r"""
    Make a remote call to run ``func`` on worker ``to`` and return an
    :class:`~torch.distributed.rpc.RRef` to the result value immediately.
    Worker ``to`` will be the owner of the returned
    :class:`~torch.distributed.rpc.RRef`, and the worker calling ``remote`` is
    a user. The owner manages the global reference count of its
    :class:`~torch.distributed.rpc.RRef`, and the owner
    :class:`~torch.distributed.rpc.RRef` is only destructed when globally there
    are no living references to it.

    Args:
        to (str or WorkerInfo or int): name/rank/``WorkerInfo`` of the destination worker.
        func (Callable): a callable function, such as Python callables, builtin
                         operators (e.g. :meth:`~torch.add`) and annotated
                         TorchScript functions.
        args (tuple): the argument tuple for the ``func`` invocation.
        kwargs (dict): is a dictionary of keyword arguments for the ``func``
                       invocation.

        timeout (float, optional): timeout in seconds for this remote call. If the
                                   creation of this
                                   :class:`~torch.distributed.rpc.RRef` on worker
                                   ``to`` is not successfully processed on this
                                   worker within this timeout, then the next time
                                   there is an attempt to use the RRef (such as
                                   ``to_here()``), a timeout will be raised
                                   indicating this failure. A value of 0 indicates
                                   an infinite timeout, i.e. a timeout error will
                                   never be raised. If not provided, the default
                                   value set during initialization or with
                                   ``_set_rpc_timeout`` is used.

    Returns:
        A user :class:`~torch.distributed.rpc.RRef` instance to the result
        value. Use the blocking API :meth:`torch.distributed.rpc.RRef.to_here`
        to retrieve the result value locally.

    .. warning ::
        The ``remote`` API does not copy storages of argument tensors until
        sending them over the wire, which could be done by a different thread
        depending on the RPC backend type. The caller should make sure that the
        contents of those tensors stay intact until the returned RRef is
        confirmed by the owner, which can be checked using the
        :meth:`torch.distributed.rpc.RRef.confirmed_by_owner` API.

    .. warning ::
        Errors such as timeouts for the ``remote`` API are handled on a
        best-effort basis. This means that when remote calls initiated by
        ``remote`` fail, such as with a timeout error, we take a best-effort
        approach to error handling. This means that errors are handled and set
        on the resulting RRef on an asynchronous basis. If the RRef has not been
        used by the application before this handling (such as ``to_here`` or
        fork call), then future uses of the ``RRef`` will appropriately raise
        errors. However, it is possible that the user application will use the
        ``RRef`` before the errors are handled. In this case, errors may not be
        raised as they have not yet been handled.

    Example::

        Make sure that ``MASTER_ADDR`` and ``MASTER_PORT`` are set properly
        on both workers. Refer to :meth:`~torch.distributed.init_process_group`
        API for more details. For example,

        export MASTER_ADDR=localhost
        export MASTER_PORT=5678

        Then run the following code in two different processes:

        >>> # xdoctest: +SKIP
        >>> # On worker 0:
        >>> import torch
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker0", rank=0, world_size=2)
        >>> rref1 = rpc.remote("worker1", torch.add, args=(torch.ones(2), 3))
        >>> rref2 = rpc.remote("worker1", torch.add, args=(torch.ones(2), 1))
        >>> x = rref1.to_here() + rref2.to_here()
        >>> rpc.shutdown()

        >>> # On worker 1:
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker1", rank=1, world_size=2)
        >>> rpc.shutdown()

        Below is an example of running a TorchScript function using RPC.

        >>> # On both workers:
        >>> @torch.jit.script
        >>> def my_script_add(tensor: torch.Tensor, scalar: int):
        >>>    return torch.add(tensor, scalar)

        >>> # On worker 0:
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker0", rank=0, world_size=2)
        >>> rref = rpc.remote("worker1", my_script_add, args=(torch.ones(2), 3))
        >>> rref.to_here()
        >>> rpc.shutdown()

        >>> # On worker 1:
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker1", rank=1, world_size=2)
        >>> rpc.shutdown()
    """
    torch._C._log_api_usage_once("torch.distributed.rpc_remote")
    qualified_name = torch.jit._builtins._find_builtin(func)
    dst_worker_info = _to_worker_info(to)
    should_profile = _get_should_profile()

    ctx_manager = _enable_rpc_profiler(
        should_profile, qualified_name, func, RPCExecMode.REMOTE, dst_worker_info
    )

    with ctx_manager as rf:
        args = args if args else ()
        kwargs = kwargs if kwargs else {}

        is_async_exec = hasattr(func, "_wrapped_async_rpc_function")

        if is_async_exec:
            wrapped = func._wrapped_async_rpc_function
            if isinstance(wrapped, torch.jit.ScriptFunction):
                func = wrapped

        if qualified_name is not None:
            rref = _invoke_remote_builtin(
                dst_worker_info, qualified_name, timeout, *args, **kwargs
            )
        elif isinstance(func, torch.jit.ScriptFunction):
            rref = _invoke_remote_torchscript(
                dst_worker_info.name,
                torch._jit_internal._qualified_name(func),
                timeout,
                is_async_exec,
                *args,
                **kwargs,
            )
        else:
            (pickled_python_udf, tensors) = _default_pickler.serialize(
                PythonUDF(func, args, kwargs)
            )
            rref = _invoke_remote_python_udf(
                dst_worker_info, pickled_python_udf, tensors, timeout, is_async_exec
            )
        # attach profiling information
        if should_profile:
            if not torch.autograd._profiler_enabled():
                raise AssertionError
            if rf is None:
                raise AssertionError
            fut = rf._call_end_callbacks_on_future(rref._get_future())
            rref._set_profiling_future(fut)

    return rref


def _invoke_rpc(
    to, func, rpc_type, args=None, kwargs=None, rpc_timeout: float = UNSET_RPC_TIMEOUT
):
    if not callable(func):
        raise TypeError("function should be callable.")

    qualified_name = torch.jit._builtins._find_builtin(func)
    dst_worker_info = _to_worker_info(to)

    should_profile = _get_should_profile()

    ctx_manager = _enable_rpc_profiler(
        should_profile, qualified_name, func, rpc_type, dst_worker_info
    )

    with ctx_manager as rf:
        args = args if args else ()
        kwargs = kwargs if kwargs else {}

        is_async_exec = hasattr(func, "_wrapped_async_rpc_function")

        if is_async_exec:
            # pyrefly: ignore [missing-attribute]
            wrapped = func._wrapped_async_rpc_function
            if isinstance(wrapped, torch.jit.ScriptFunction):
                func = wrapped

        if qualified_name is not None:
            fut = _invoke_rpc_builtin(
                dst_worker_info, qualified_name, rpc_timeout, *args, **kwargs
            )
        elif isinstance(func, torch.jit.ScriptFunction):
            fut = _invoke_rpc_torchscript(
                dst_worker_info.name,
                torch._jit_internal._qualified_name(func),
                args,
                kwargs,
                rpc_timeout,
                is_async_exec,
            )
        else:
            (pickled_python_udf, tensors) = _default_pickler.serialize(
                PythonUDF(func, args, kwargs)
            )
            fut = _invoke_rpc_python_udf(
                dst_worker_info, pickled_python_udf, tensors, rpc_timeout, is_async_exec
            )
        if should_profile:
            if not torch.autograd._profiler_enabled():
                raise AssertionError
            if rf is None:
                raise AssertionError
            # Schedule profiling callbacks to run when the future completes.
            # This returns a future that is completed when the original future
            # completes and the profiling callbacks have been completed as well,
            # to guarantee that fut.wait() completes the profiling. This new
            # future will contain the same value as the original future.
            fut = rf._call_end_callbacks_on_future(fut)
    return fut


@_require_initialized
def rpc_sync(to, func, args=None, kwargs=None, timeout: float = UNSET_RPC_TIMEOUT):
    r"""
    Make a blocking RPC call to run function ``func`` on worker ``to``. RPC
    messages are sent and received in parallel to execution of Python code. This
    method is thread-safe.

    Args:
        to (str or WorkerInfo or int): name/rank/``WorkerInfo`` of the destination worker.
        func (Callable): a callable function, such as Python callables, builtin
                         operators (e.g. :meth:`~torch.add`) and annotated
                         TorchScript functions.
        args (tuple): the argument tuple for the ``func`` invocation.
        kwargs (dict): is a dictionary of keyword arguments for the ``func``
                       invocation.
        timeout (float, optional): timeout in seconds to use for this RPC. If
                                   the RPC does not complete in this amount of
                                   time, an exception indicating it has
                                   timed out will be raised. A value of 0
                                   indicates an infinite timeout, i.e. a timeout
                                   error will never be raised. If not provided,
                                   the default value set during initialization
                                   or with ``_set_rpc_timeout`` is used.

    Returns:
        Returns the result of running ``func`` with ``args`` and ``kwargs``.

    Example::
        Make sure that ``MASTER_ADDR`` and ``MASTER_PORT`` are set properly
        on both workers. Refer to :meth:`~torch.distributed.init_process_group`
        API for more details. For example,

        export MASTER_ADDR=localhost
        export MASTER_PORT=5678

        Then run the following code in two different processes:

        >>> # xdoctest: +SKIP
        >>> # On worker 0:
        >>> import torch
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker0", rank=0, world_size=2)
        >>> ret = rpc.rpc_sync("worker1", torch.add, args=(torch.ones(2), 3))
        >>> rpc.shutdown()

        >>> # On worker 1:
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker1", rank=1, world_size=2)
        >>> rpc.shutdown()

        Below is an example of running a TorchScript function using RPC.

        >>> # On both workers:
        >>> @torch.jit.script
        >>> def my_script_add(tensor: torch.Tensor, scalar: int):
        >>>    return torch.add(tensor, scalar)

        >>> # On worker 0:
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker0", rank=0, world_size=2)
        >>> ret = rpc.rpc_sync("worker1", my_script_add, args=(torch.ones(2), 3))
        >>> rpc.shutdown()

        >>> # On worker 1:
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker1", rank=1, world_size=2)
        >>> rpc.shutdown()

    """
    torch._C._log_api_usage_once("torch.distributed.rpc_sync")
    fut = _invoke_rpc(to, func, RPCExecMode.SYNC, args, kwargs, timeout)
    return fut.wait()


@_require_initialized
def rpc_async(to, func, args=None, kwargs=None, timeout=UNSET_RPC_TIMEOUT):
    r"""
    Make a non-blocking RPC call to run function ``func`` on worker ``to``. RPC
    messages are sent and received in parallel to execution of Python code. This
    method is thread-safe. This method will immediately return a
    :class:`~torch.futures.Future` that can be awaited on.

    Args:
        to (str or WorkerInfo or int): name/rank/``WorkerInfo`` of the destination worker.
        func (Callable): a callable function, such as Python callables, builtin
                         operators (e.g. :meth:`~torch.add`) and annotated
                         TorchScript functions.
        args (tuple): the argument tuple for the ``func`` invocation.
        kwargs (dict): is a dictionary of keyword arguments for the ``func``
                       invocation.
        timeout (float, optional): timeout in seconds to use for this RPC. If
                                   the RPC does not complete in this amount of
                                   time, an exception indicating it has
                                   timed out will be raised. A value of 0
                                   indicates an infinite timeout, i.e. a timeout
                                   error will never be raised. If not provided,
                                   the default value set during initialization
                                   or with ``_set_rpc_timeout`` is used.


    Returns:
        Returns a :class:`~torch.futures.Future` object that can be waited
        on. When completed, the return value of ``func`` on ``args`` and
        ``kwargs`` can be retrieved from the :class:`~torch.futures.Future`
        object.

    .. warning ::
        Using GPU tensors as arguments or return values of ``func`` is not
        supported since we don't support sending GPU tensors over the wire. You
        need to explicitly copy GPU tensors to CPU before using them as
        arguments or return values of ``func``.

    .. warning ::
        The ``rpc_async`` API does not copy storages of argument tensors until
        sending them over the wire, which could be done by a different thread
        depending on the RPC backend type. The caller should make sure that the
        contents of those tensors stay intact until the returned
        :class:`~torch.futures.Future` completes.

    Example::
        Make sure that ``MASTER_ADDR`` and ``MASTER_PORT`` are set properly
        on both workers. Refer to :meth:`~torch.distributed.init_process_group`
        API for more details. For example,

        export MASTER_ADDR=localhost
        export MASTER_PORT=5678

        Then run the following code in two different processes:

        >>> # xdoctest: +SKIP
        >>> # On worker 0:
        >>> import torch
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker0", rank=0, world_size=2)
        >>> fut1 = rpc.rpc_async("worker1", torch.add, args=(torch.ones(2), 3))
        >>> fut2 = rpc.rpc_async("worker1", min, args=(1, 2))
        >>> result = fut1.wait() + fut2.wait()
        >>> rpc.shutdown()

        >>> # On worker 1:
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker1", rank=1, world_size=2)
        >>> rpc.shutdown()

        Below is an example of running a TorchScript function using RPC.

        >>> # On both workers:
        >>> @torch.jit.script
        >>> def my_script_add(tensor: torch.Tensor, scalar: int):
        >>>    return torch.add(tensor, scalar)

        >>> # On worker 0:
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker0", rank=0, world_size=2)
        >>> fut = rpc.rpc_async("worker1", my_script_add, args=(torch.ones(2), 3))
        >>> ret = fut.wait()
        >>> rpc.shutdown()

        >>> # On worker 1:
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker1", rank=1, world_size=2)
        >>> rpc.shutdown()
    """
    torch._C._log_api_usage_once("torch.distributed.rpc_async")
    fut = _invoke_rpc(to, func, RPCExecMode.ASYNC, args, kwargs, timeout)
    if hasattr(_thread_local_var, "future_list"):
        _thread_local_var.future_list.append(fut)
    return fut


def _get_should_profile():
    # Legacy profiler should be enabled. RPC profiling is not supported with
    # Kineto profiler.
    ActiveProfilerType = torch._C._profiler.ActiveProfilerType
    return (
        torch.autograd._profiler_enabled()
        and torch._C._autograd._profiler_type() == ActiveProfilerType.LEGACY  # type: ignore[attr-defined]
    )


def _enable_rpc_profiler(
    should_profile, qualified_name, func, rpc_type, dst_worker_info
):
    ctx_manager = contextlib.nullcontext()

    if should_profile:
        # Create appropriate string representation based on type of func
        # (builtin, script, python)
        if qualified_name is None:
            func_name = (
                torch._jit_internal._qualified_name(func)
                if isinstance(func, torch.jit.ScriptFunction)
                else func.__qualname__
            )
        else:
            func_name = qualified_name
        # Build RPC profiling key.
        rpc_profiling_key = _build_rpc_profiling_key(
            rpc_type,
            func_name,
            get_worker_info().name,
            dst_worker_info.name,
        )
        RemoteProfilerManager.set_current_profiling_key(rpc_profiling_key)
        # Mypy doesn't support re-def of a variable not in the same block (#1174)
        ctx_manager = torch.autograd.profiler.record_function(rpc_profiling_key)  # type: ignore[assignment]

    return ctx_manager
<|endoftext|># mypy: allow-untyped-defs
import functools


def async_execution(fn):
    r"""
    A decorator for a function indicating that the return value of the function
    is guaranteed to be a :class:`~torch.futures.Future` object and this
    function can run asynchronously on the RPC callee. More specifically, the
    callee extracts the :class:`~torch.futures.Future` returned by the wrapped
    function and installs subsequent processing steps as a callback to that
    :class:`~torch.futures.Future`. The installed callback will read the value
    from the :class:`~torch.futures.Future` when completed and send the
    value back as the RPC response. That also means the returned
    :class:`~torch.futures.Future` only exists on the callee side and is never
    sent through RPC. This decorator is useful when the wrapped function's
    (``fn``) execution needs to pause and resume due to, e.g., containing
    :meth:`~torch.distributed.rpc.rpc_async` or waiting for other signals.

    .. note:: To enable asynchronous execution, applications must pass the
        function object returned by this decorator to RPC APIs. If RPC detected
        attributes installed by this decorator, it knows that this function
        returns a ``Future`` object and will handle that accordingly.
        However, this does not mean this decorator has to be outmost one when
        defining a function. For example, when combined with ``@staticmethod``
        or ``@classmethod``, ``@rpc.functions.async_execution`` needs to be the
        inner decorator to allow the target function be recognized as a static
        or class function. This target function can still execute asynchronously
        because, when accessed, the static or class method preserves attributes
        installed by ``@rpc.functions.async_execution``.


    Example::
        The returned :class:`~torch.futures.Future` object can come from
        :meth:`~torch.distributed.rpc.rpc_async`,
        :meth:`~torch.futures.Future.then`, or :class:`~torch.futures.Future`
        constructor. The example below shows directly using the
        :class:`~torch.futures.Future` returned by
        :meth:`~torch.futures.Future.then`.

        >>> from torch.distributed import rpc
        >>>
        >>> # omitting setup and shutdown RPC
        >>>
        >>> # On all workers
        >>> @rpc.functions.async_execution
        >>> def async_add_chained(to, x, y, z):
        >>>     # This function runs on "worker1" and returns immediately when
        >>>     # the callback is installed through the `then(cb)` API. In the
        >>>     # mean time, the `rpc_async` to "worker2" can run concurrently.
        >>>     # When the return value of that `rpc_async` arrives at
        >>>     # "worker1", "worker1" will run the lambda function accordingly
        >>>     # and set the value for the previously returned `Future`, which
        >>>     # will then trigger RPC to send the result back to "worker0".
        >>>     return rpc.rpc_async(to, torch.add, args=(x, y)).then(
        >>>         lambda fut: fut.wait() + z
        >>>     )
        >>>
        >>> # On worker0
        >>> # xdoctest: +SKIP
        >>> ret = rpc.rpc_sync(
        >>>     "worker1",
        >>>     async_add_chained,
        >>>     args=("worker2", torch.ones(2), 1, 1)
        >>> )
        >>> print(ret)  # prints tensor([3., 3.])

        When combined with TorchScript decorators, this decorator must be the
        outmost one.

        >>> from torch import Tensor
        >>> from torch.futures import Future
        >>> from torch.distributed import rpc
        >>>
        >>> # omitting setup and shutdown RPC
        >>>
        >>> # On all workers
        >>> @torch.jit.script
        >>> def script_add(x: Tensor, y: Tensor) -> Tensor:
        >>>     return x + y
        >>>
        >>> @rpc.functions.async_execution
        >>> @torch.jit.script
        >>> def async_add(to: str, x: Tensor, y: Tensor) -> Future[Tensor]:
        >>>     return rpc.rpc_async(to, script_add, (x, y))
        >>>
        >>> # On worker0
        >>> ret = rpc.rpc_sync(
        >>>     "worker1",
        >>>     async_add,
        >>>     args=("worker2", torch.ones(2), 1)
        >>> )
        >>> print(ret)  # prints tensor([2., 2.])

        When combined with static or class method, this decorator must be the
        inner one.

        >>> from torch.distributed import rpc
        >>>
        >>> # omitting setup and shutdown RPC
        >>>
        >>> # On all workers
        >>> class AsyncExecutionClass:
        >>>
        >>>     @staticmethod
        >>>     @rpc.functions.async_execution
        >>>     def static_async_add(to, x, y, z):
        >>>         return rpc.rpc_async(to, torch.add, args=(x, y)).then(
        >>>             lambda fut: fut.wait() + z
        >>>         )
        >>>
        >>>     @classmethod
        >>>     @rpc.functions.async_execution
        >>>     def class_async_add(cls, to, x, y, z):
        >>>         ret_fut = torch.futures.Future()
        >>>         rpc.rpc_async(to, torch.add, args=(x, y)).then(
        >>>             lambda fut: ret_fut.set_result(fut.wait() + z)
        >>>         )
        >>>         return ret_fut
        >>>
        >>>     @rpc.functions.async_execution
        >>>     def bound_async_add(self, to, x, y, z):
        >>>         return rpc.rpc_async(to, torch.add, args=(x, y)).then(
        >>>             lambda fut: fut.wait() + z
        >>>         )
        >>>
        >>> # On worker0
        >>> ret = rpc.rpc_sync(
        >>>     "worker1",
        >>>     AsyncExecutionClass.static_async_add,
        >>>     args=("worker2", torch.ones(2), 1, 2)
        >>> )
        >>> print(ret)  # prints tensor([4., 4.])
        >>>
        >>> ret = rpc.rpc_sync(
        >>>     "worker1",
        >>>     AsyncExecutionClass.class_async_add,
        >>>     args=("worker2", torch.ones(2), 1, 2)
        >>> )
        >>> print(ret)  # prints tensor([4., 4.])

        This decorator also works with RRef helpers, i.e., .
        :meth:`torch.distributed.rpc.RRef.rpc_sync`,
        :meth:`torch.distributed.rpc.RRef.rpc_async`, and
        :meth:`torch.distributed.rpc.RRef.remote`.

        >>> from torch.distributed import rpc
        >>>
        >>> # reuse the AsyncExecutionClass class above
        >>> rref = rpc.remote("worker1", AsyncExecutionClass)
        >>> ret = rref.rpc_sync().static_async_add("worker2", torch.ones(2), 1, 2)
        >>> print(ret)  # prints tensor([4., 4.])
        >>>
        >>> rref = rpc.remote("worker1", AsyncExecutionClass)
        >>> ret = rref.rpc_async().static_async_add("worker2", torch.ones(2), 1, 2).wait()
        >>> print(ret)  # prints tensor([4., 4.])
        >>>
        >>> rref = rpc.remote("worker1", AsyncExecutionClass)
        >>> ret = rref.remote().static_async_add("worker2", torch.ones(2), 1, 2).to_here()
        >>> print(ret)  # prints tensor([4., 4.])
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    # Can't declare and use attributes of function objects (mypy#2087)
    wrapper._wrapped_async_rpc_function = fn  # type: ignore[attr-defined]
    return wrapper
<|endoftext|># mypy: allow-untyped-defs


import collections
import enum
from typing import cast

import torch
import torch.distributed as dist

from . import api, constants as rpc_constants
from ._utils import _group_membership_management, _update_group_membership


__all__ = [
    "backend_registered",
    "register_backend",
    "construct_rpc_backend_options",
    "init_backend",
    "BackendValue",
    "BackendType",
]

BackendValue = collections.namedtuple(
    "BackendValue", ["construct_rpc_backend_options_handler", "init_backend_handler"]
)


def _backend_type_repr(self):
    return "BackendType." + self.name


_backend_type_doc = """
    An enum class of available backends.

    PyTorch ships with a builtin ``BackendType.TENSORPIPE`` backend.
    Additional ones can be registered using the
    :func:`~torch.distributed.rpc.backend_registry.register_backend` function.
"""

# Create an enum type, `BackendType`, with empty members.
# Can't handle Function Enum API (mypy bug #9079)
BackendType = enum.Enum(value="BackendType", names={})  # type: ignore[misc]
# Unable to assign a function a method (mypy bug #2427)
BackendType.__repr__ = _backend_type_repr  # type: ignore[assignment]

if BackendType.__doc__:
    BackendType.__doc__ = _backend_type_doc


def backend_registered(backend_name):
    """
    Checks if backend_name is registered as an RPC backend.

    Args:
        backend_name (str): string to identify the RPC backend.
    Returns:
        True if the backend has been registered with ``register_backend``, else
        False.
    """
    return backend_name in BackendType.__members__


def register_backend(
    backend_name, construct_rpc_backend_options_handler, init_backend_handler
):
    """Registers a new RPC backend.

    Args:
        backend_name (str): backend string to identify the handler.
        construct_rpc_backend_options_handler (function):
            Handler that is invoked when
            ``rpc_backend.construct_rpc_backend_options(**dict)`` is called.
        init_backend_handler (function): Handler that is invoked when the
            `_init_rpc_backend()` function is called with a backend.
            This returns the agent.
    """
    global BackendType
    if backend_registered(backend_name):
        raise RuntimeError(f"RPC backend {backend_name}: already registered")
    # Create a new enum type, `BackendType`, with extended members.
    existing_enum_dict = {member.name: member.value for member in BackendType}
    extended_enum_dict = dict(
        {
            backend_name: BackendValue(
                construct_rpc_backend_options_handler=construct_rpc_backend_options_handler,
                init_backend_handler=init_backend_handler,
            )
        },
        **existing_enum_dict,
    )
    # Can't handle Function Enum API (mypy bug #9079)
    BackendType = enum.Enum(value="BackendType", names=extended_enum_dict)  # type: ignore[misc]
    # Unable to assign a function a method (mypy bug #2427)
    BackendType.__repr__ = _backend_type_repr  # type: ignore[assignment]
    if BackendType.__doc__:
        BackendType.__doc__ = _backend_type_doc

    return BackendType[backend_name]


def construct_rpc_backend_options(
    backend,
    rpc_timeout=rpc_constants.DEFAULT_RPC_TIMEOUT_SEC,
    init_method=rpc_constants.DEFAULT_INIT_METHOD,
    **kwargs,
):
    return backend.value.construct_rpc_backend_options_handler(
        rpc_timeout, init_method, **kwargs
    )


def init_backend(backend, *args, **kwargs):
    return backend.value.init_backend_handler(*args, **kwargs)


def _init_process_group(store, rank, world_size):
    # Initialize ProcessGroup.
    process_group_timeout = rpc_constants.DEFAULT_PROCESS_GROUP_TIMEOUT

    # We're using a bunch of private APIs here since `new_group` requires the
    # default group to be initialized.
    group = dist.ProcessGroupGloo(store, rank, world_size, process_group_timeout)

    if group is None:
        raise AssertionError("Failed to initialize default ProcessGroup.")

    if (rank != -1) and (rank != group.rank()):
        raise RuntimeError(f"rank argument {rank} doesn't match pg rank {group.rank()}")
    if (world_size != -1) and (world_size != group.size()):
        raise RuntimeError(
            f"world_size argument {world_size} doesn't match pg size {group.size()}"
        )
    return group


def _tensorpipe_construct_rpc_backend_options_handler(
    rpc_timeout,
    init_method,
    num_worker_threads=rpc_constants.DEFAULT_NUM_WORKER_THREADS,
    _transports=None,
    _channels=None,
    **kwargs,
):
    from . import TensorPipeRpcBackendOptions

    return TensorPipeRpcBackendOptions(
        rpc_timeout=rpc_timeout,
        init_method=init_method,
        num_worker_threads=num_worker_threads,
        _transports=_transports,
        _channels=_channels,
    )


def _tensorpipe_validate_devices(devices, device_count):
    device_type = (
        acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
    )
    return all(
        d.type == "cpu" or (d.type == device_type and 0 <= d.index < device_count)
        for d in devices
    )


# detect if any worker has invalid device_map configurations, and return
# reverse device maps
def _tensorpipe_exchange_and_check_all_device_maps(
    my_name, my_device_count, my_device_maps, my_devices, group
):
    gathered: list[
        tuple[str, int, dict[str, dict[torch.device, torch.device]], list[torch.device]]
    ] = [("", 0, {}, []) for _ in range(group.size())]
    dist.all_gather_object(
        gathered, (my_name, my_device_count, my_device_maps, my_devices), group
    )
    all_names = [name for name, _, _, _ in gathered]
    all_device_counts = {name: count for name, count, _, _ in gathered}
    all_device_maps = {name: map_ for name, _, map_, _ in gathered}
    all_devices = {name: devices for name, _, _, devices in gathered}

    _validate_device_maps(all_names, all_device_counts, all_device_maps, all_devices)

    # passed all checked, construct reverse mapping and get list of devices handled by this agent
    reverse_device_maps = _create_reverse_mapping(my_name, all_names, all_device_maps)
    my_devices = _create_device_list(my_devices, my_device_maps, reverse_device_maps)
    return reverse_device_maps, my_devices


def _validate_device_maps(
    all_names, all_device_counts, all_device_maps, all_devices, is_static_group=True
):
    for node in all_names:
        devices = all_devices[node]
        if len(set(devices)) != len(devices):
            raise ValueError(f"Node {node} has duplicated devices\ndevices = {devices}")
        if not _tensorpipe_validate_devices(devices, all_device_counts[node]):
            raise ValueError(
                f"Node {node} has devices with invalid indices\n"
                f"devices = {devices}\n"
                f"device count = {all_device_counts[node]}"
            )

    for source_node in all_names:
        # For dynamic group (non-static) do not check the target node name since it may not have joined yet
        if is_static_group and not set(all_device_maps[source_node].keys()).issubset(
            all_names
        ):
            raise ValueError(
                f"Node {source_node} has invalid target node names in its device maps\n"
                f"device maps = {all_device_maps[source_node].keys()}\n"
                f"node names = {all_names}"
            )
        for target_node, map_ in all_device_maps[source_node].items():
            if len(set(map_.values())) != len(map_):
                raise ValueError(
                    f"Node {source_node} has duplicated target devices "
                    f"in its device map for {target_node}\n"
                    f"device map = {map_}"
                )
            if all_devices[source_node]:
                if not set(map_.keys()).issubset(all_devices[source_node]):
                    raise ValueError(
                        f"Node {source_node} has unexpected source devices "
                        f"in its device map for {target_node}\n"
                        f"device map = {map_}\n"
                        f"devices = {all_devices[source_node]}"
                    )
            elif not _tensorpipe_validate_devices(
                map_.keys(), all_device_counts[source_node]
            ):
                raise ValueError(
                    f"Node {source_node} has source devices with invalid indices "
                    f"in its device map for {target_node}\n"
                    f"device map = {map_}\n"
                    f"device count = {all_device_counts[source_node]}"
                )
            if all_devices.get(target_node, []):
                if not set(map_.values()).issubset(all_devices[target_node]):
                    raise ValueError(
                        f"Node {source_node} has unexpected target devices "
                        f"in its device map for {target_node}\n"
                        f"device map = {map_}\n"
                        f"devices = {all_devices[target_node]}"
                    )
            elif target_node in all_device_counts and not _tensorpipe_validate_devices(
                map_.values(), all_device_counts[target_node]
            ):
                raise ValueError(
                    f"Node {source_node} has target devices with invalid indices "
                    f"in its device map for {target_node}\n"
                    f"device map = {map_}\n"
                    f"device count = {all_device_counts[target_node]}"
                )


def _create_device_list(my_devices, my_device_maps, reverse_device_maps):
    if not my_devices:
        devices_set: set[torch.device] = set()
        for map_ in my_device_maps.values():
            devices_set.update(map_.keys())
        for map_ in reverse_device_maps.values():
            devices_set.update(map_.keys())
        devices_set.discard(torch.device("cpu"))
        my_devices = list(devices_set)
    my_devices = sorted(my_devices, key=lambda d: d.index)
    return my_devices


def _create_reverse_mapping(my_name, all_names, all_device_maps):
    reverse_device_maps: dict[str, dict[torch.device, torch.device]] = {}
    for node in all_names:
        if my_name in all_device_maps[node]:
            reverse_device_maps[node] = {
                v: k for k, v in all_device_maps[node][my_name].items()
            }
    return reverse_device_maps


def _get_device_infos():
    from . import TensorPipeAgent

    agent = cast(TensorPipeAgent, api._get_current_rpc_agent())
    opts = agent._get_backend_options()
    device_count = torch.accelerator.device_count()
    if torch.accelerator.is_available() and opts.devices:
        mod = torch.get_device_module(torch.accelerator.current_accelerator())
        mod.init()
    return device_count, opts.device_maps, opts.devices


def _set_devices_and_reverse_device_map(agent):
    from . import TensorPipeAgent

    agent = cast(TensorPipeAgent, agent)
    # Group state is retrieved from local agent
    # On initialization, tensorpipe agent retrieves information from all existing workers, so group state is valid
    my_worker_info = agent.get_worker_info()
    my_name = my_worker_info.name
    all_worker_infos = agent.get_worker_infos()
    # One round to get device_maps of all workers and construct reverse device maps
    all_device_counts, all_device_maps, all_devices, all_names = {}, {}, {}, []
    for worker_info in all_worker_infos:
        worker_name = worker_info.name
        if worker_name != my_name:
            # TODO: make async?
            device_count, device_map, devices = api.rpc_sync(
                worker_name, _get_device_infos
            )
        else:
            opts = agent._get_backend_options()
            device_count, device_map, devices = (
                torch.accelerator.device_count(),
                opts.device_maps,
                opts.devices,
            )
        all_device_counts[worker_name] = device_count
        all_device_maps[worker_name] = device_map
        all_devices[worker_name] = devices
        all_names.append(worker_name)

    _validate_device_maps(
        all_names,
        all_device_counts,
        all_device_maps,
        all_devices,
        is_static_group=False,
    )
    reverse_device_maps = _create_reverse_mapping(my_name, all_names, all_device_maps)

    # Perform RPC call to all workers, including itself, to include newly joined worker information and device maps
    for worker_name in all_names:
        # Set device list for each worker
        all_devices[worker_name] = _create_device_list(
            all_devices[worker_name], all_device_maps[worker_name], reverse_device_maps
        )
        api.rpc_sync(
            worker_name,
            _update_group_membership,
            args=(my_worker_info, all_devices[worker_name], reverse_device_maps, True),
        )


def _tensorpipe_init_backend_handler(
    store, name, rank, world_size, rpc_backend_options
):
    from . import TensorPipeAgent, TensorPipeRpcBackendOptions

    if not isinstance(store, dist.Store):
        raise TypeError(f"`store` must be a c10d::Store. {store}")

    if not isinstance(rpc_backend_options, TensorPipeRpcBackendOptions):
        raise TypeError(
            f"`rpc_backend_options` must be a `TensorPipeRpcBackendOptions`. {rpc_backend_options}"
        )

    device_count = torch.accelerator.device_count()

    is_static_group = bool(world_size)
    # world_size is specified so this is a static group (ranks cannot join and leave)
    if is_static_group:
        # The agent's join method is required to behave like a barrier and perform
        # collective operations, for which it relies on a process group, instead of
        # re-implementing this on top of RPCs.
        group = _init_process_group(store, rank, world_size)

        reverse_device_maps, devices = _tensorpipe_exchange_and_check_all_device_maps(
            name,
            device_count,
            rpc_backend_options.device_maps,
            rpc_backend_options.devices,
            group,
        )

        if torch.accelerator.is_available() and devices:
            # It's necessary to initialize PyTorch CUDA states here (e.g.,
            # CUDACachingAllocator). If this is missing, we could hit errors like
            # "allocator not initialized", because other processes might send
            # CUDA-related RPC request to this process before user code in this
            # process initializes its PyTorch CUDA states.
            mod = torch.get_device_module(torch.accelerator.current_accelerator())
            mod.init()

        # TODO: add try-except and destroy _agent in all processes if any fails.
        agent = TensorPipeAgent(
            store,
            name,
            rank,
            world_size,
            rpc_backend_options,
            reverse_device_maps,
            devices,
        )

        api._init_rpc_states(agent)

        # Run one dummy round of RPC to initialize channels/transports. Without
        # this, it's easy to hit timeout in rpc.shutdown() if there is no other RPC
        # on that process before rpc.shutdown(), as the agent initialization can
        # take longer than 5s.
        api._all_gather(None, timeout=rpc_backend_options.rpc_timeout)
        # Need a barrier here to make sure no peers leave before the rank0 finishes
        # _all_gather
        group.barrier().wait()

        return agent
    # initialization for dynamic rpc (ranks can join and leave)
    else:
        with _group_membership_management(store, name, True):
            # Construct TPAgent with empty reverse_device_map and devices
            # these properties will be updated after initialization
            agent = TensorPipeAgent(
                store,
                name,
                rank,
                world_size,
                rpc_backend_options,
                {},
                [],
            )
            api._init_rpc_states(agent)

            try:
                # Notify all workers in group this rank has joined and set devices and reverse_device_map
                # This is a synchronous operation that completes once all existing ranks are updated
                _set_devices_and_reverse_device_map(agent)
            except Exception:
                api.shutdown()
                raise
            return agent


register_backend(
    "TENSORPIPE",
    _tensorpipe_construct_rpc_backend_options_handler,
    _tensorpipe_init_backend_handler,
)
<|endoftext|>#!/usr/bin/env python3
# mypy: allow-untyped-defs

import torch.distributed as dist
import torch.distributed.rpc as rpc


def _faulty_tensorpipe_construct_rpc_backend_options_handler(
    rpc_timeout,
    init_method,
    num_worker_threads,
    messages_to_fail,
    messages_to_delay,
    num_fail_sends,
    **kwargs,
):
    from . import FaultyTensorPipeRpcBackendOptions

    return FaultyTensorPipeRpcBackendOptions(
        num_worker_threads=num_worker_threads,
        rpc_timeout=rpc_timeout,
        init_method=init_method,
        messages_to_fail=messages_to_fail,
        messages_to_delay=messages_to_delay,
        num_fail_sends=num_fail_sends,
    )


def _faulty_tensorpipe_init_backend_handler(
    store, name, rank, world_size, rpc_backend_options
):
    from torch.distributed.rpc import api

    from . import FaultyTensorPipeAgent, FaultyTensorPipeRpcBackendOptions

    if not isinstance(store, dist.Store):
        raise TypeError(f"`store` must be a c10d::Store. {store}")

    if not isinstance(rpc_backend_options, FaultyTensorPipeRpcBackendOptions):
        raise TypeError(
            f"`rpc_backend_options` must be a `FaultyTensorPipeRpcBackendOptions`. {rpc_backend_options}"
        )

    agent = FaultyTensorPipeAgent(
        store,
        name,
        rank,
        world_size,
        rpc_backend_options,
        {},  # reverse_device_map
        [],  # devices
    )
    api._init_rpc_states(agent)

    return agent


rpc.backend_registry.register_backend(
    "FAULTY_TENSORPIPE",
    _faulty_tensorpipe_construct_rpc_backend_options_handler,
    _faulty_tensorpipe_init_backend_handler,
)
<|endoftext|>import torch


def is_available() -> bool:
    return hasattr(torch._C, "_faulty_agent_init")


if is_available() and not torch._C._faulty_agent_init():
    raise RuntimeError("Failed to initialize torch.distributed.rpc._testing")

if is_available():
    # Registers FAULTY_TENSORPIPE RPC backend.
    from torch._C._distributed_rpc_testing import (
        FaultyTensorPipeAgent,
        FaultyTensorPipeRpcBackendOptions,
    )

    from . import faulty_agent_backend_registry
<|endoftext|># mypy: allow-untyped-defs

import torch
from torch.utils import _pytree as pytree


def _basic_validation(op, args=(), kwargs=None):
    """
    Common validation across all ops go in here.
    """
    from torch.distributed._shard.sharded_tensor import ShardedTensor

    if len(args) == 0 and (kwargs is None or len(kwargs) == 0):
        raise ValueError(f" No input for '{op.__name__}'!")

    # Validate types
    has_distributed_tensor = False

    def is_distributed_tensor(e):
        nonlocal has_distributed_tensor
        if isinstance(e, ShardedTensor):
            has_distributed_tensor = True

    pytree.tree_map_(is_distributed_tensor, args)
    pytree.tree_map_(is_distributed_tensor, kwargs)

    if not has_distributed_tensor:
        raise TypeError(
            f"torch function '{op.__name__}', with args: {args} and "
            f"kwargs: {kwargs} are called without any distributed tensor!"
        )

    # Validate all distributed tensors use the same PG.
    cur_pg: torch.distributed.ProcessGroup | None = None

    def validate_pg(e):
        nonlocal cur_pg
        if isinstance(e, ShardedTensor):
            if cur_pg is not None and e._process_group is not cur_pg:
                raise RuntimeError(
                    "All distributed tensors should use the "
                    "same ProcessGroup if used together in an op."
                )
            cur_pg = e._process_group

    pytree.tree_map_(validate_pg, args)
    pytree.tree_map_(validate_pg, kwargs)


def _register_default_op(op, decorator):
    @decorator(op)
    def tensor_default_op(types, args=(), kwargs=None, pg=None):
        """
        Handles ``__torch_function__`` dispatch for the default tensor ops that
        behave the same as ``torch.Tensor`` such as ``torch.Tensor.shape`` or
        ``torch.Tensor.dtype``. We simply lower to the real op call with
        DisableTorchFunctionSubclass context like ``torch.Tensor.__torch_function__``
        to avoid recursions.
        """
        if kwargs is None:
            kwargs = {}

        with torch._C.DisableTorchFunctionSubclass():
            return op(*args, **kwargs)
<|endoftext|># mypy: allow-untyped-defs
import functools
from inspect import signature

from .common_op_utils import _basic_validation


"""
Common utilities to register ops on ShardedTensor
and PartialTensor.
"""


def _register_op(op, func, op_table):
    """
    Performs basic validation and registers the provided op in the given
    op_table.
    """
    if len(signature(func).parameters) != 4:
        raise TypeError(
            f"Custom sharded op function expects signature: "
            f"(types, args, kwargs, process_group), but received "
            f"signature: {signature(func)}"
        )

    op_table[op] = func


def _decorator_func(wrapped_func, op, op_table):
    """
    Decorator function to register the given ``op`` in the provided
    ``op_table``
    """

    @functools.wraps(wrapped_func)
    def wrapper(types, args, kwargs, process_group):
        _basic_validation(op, args, kwargs)
        return wrapped_func(types, args, kwargs, process_group)

    _register_op(op, wrapper, op_table)
    return wrapper
<|endoftext|>from collections.abc import Sequence

import torch
from torch.distributed._shard.metadata import ShardMetadata


DEPRECATE_MSG = "Please use DTensor instead and we are deprecating ShardedTensor."


def narrow_tensor_by_index(
    tensor: torch.Tensor,
    offsets: Sequence[int],
    sizes: Sequence[int],
) -> torch.Tensor:
    """
    Narrow the tensor according to ``offsets`` and ``sizes``.
    """
    narrowed_tensor = tensor
    for idx, (offset, size) in enumerate(zip(offsets, sizes)):
        if size < tensor.size(idx):
            # Reshape to get shard for this rank and we don't want autograd
            # recording here for the narrow op and 'local_shard' should be a
            # leaf variable in the autograd graph.
            narrowed_tensor = narrowed_tensor.narrow(idx, offset, size)
    return narrowed_tensor


def narrow_tensor(tensor: torch.Tensor, metadata: ShardMetadata) -> torch.Tensor:
    """
    Narrow the tensor according to the metadata
    """
    return narrow_tensor_by_index(tensor, metadata.shard_offsets, metadata.shard_sizes)
<|endoftext|># mypy: allow-untyped-defs
from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import distributed_c10d
from torch.distributed._shard.sharded_tensor import ShardedTensor

from .sharder import Sharder
from .sharding_plan import ShardingPlan
from .sharding_spec import ChunkShardingSpec, ShardingSpec


def _shard_tensor(
    tensor: torch.Tensor, sharding_spec: ShardingSpec, src_rank=0, process_group=None
) -> ShardedTensor:
    """
    Given a :class:`torch.Tensor`, it shards that tensor according to the provided
    ``sharding_spec``. ``src_rank`` denotes the source rank which would be
    used as the ground truth of the data which would be scattered as shards
    across the rest of the ranks.

    Args:
        tensor (:class:`torch.Tensor`): Tensor needs to be sharded.
        sharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The specification
            describing how to shard the Tensor.

    Keyword args:
        src_rank (int, optional): The source rank which is used as the ground truth of
            the data for the parameter that would be sharded and scattered
            across the rest of the ranks.
            Default: 0.
        process_group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.

    Returns:
        A :class:`ShardedTensor` sharded from the given tensor.

    .. warning::
        Only :class:`torch.distributed._shard.sharding_spec.ChunkShardingSpec` is
        currently supported as the ``sharding_spec``.
    """
    if not tensor.is_contiguous():
        raise ValueError("input tensor is not a contiguous Tensor")

    pg = (
        process_group
        if process_group is not None
        else distributed_c10d._get_default_group()
    )
    world_size = dist.get_world_size(pg)
    current_rank = dist.get_rank(pg)

    # Validate src_rank and sharding_spec are same across all ranks.
    gathered_list = [None] * world_size
    dist.all_gather_object(gathered_list, (src_rank, sharding_spec), group=pg)

    for idx, entry in enumerate(gathered_list):
        if src_rank != entry[0]:  # type: ignore[index]
            raise ValueError(
                f"src_rank={src_rank} on rank: {current_rank} does not "  # type: ignore[index]
                f"match with src_rank={entry[0]} on rank: {idx}"  # type: ignore[index]
            )
        if sharding_spec != entry[1]:  # type: ignore[index]
            raise ValueError(
                f"sharding_spec={sharding_spec} on rank: {current_rank} does not "  # type: ignore[index]
                f"match with sharding_spec={entry[1]} on rank: {idx}"  # type: ignore[index]
            )

    st = sharding_spec.shard(tensor, src_rank=src_rank, process_group=pg)

    return st


def shard_parameter(
    module: torch.nn.Module,
    param_name: str,
    sharding_spec: ShardingSpec,
    src_rank=0,
    process_group=None,
):
    """
    Given a :class:`torch.nn.Module`, a ``param_name`` for a parameter in that
    module, it shards that parameter according to the provided
    ``sharding_spec``. ``src_rank`` denotes the source rank which would be
    used as the ground truth of the data which would be scattered as shards
    across the rest of the ranks.

    This method replaces ``module.param_name`` with a
    :class:`torch.distributed._sharded_tensor.ShardedTensor`

    Args:
        module (:class:`torch.nn.Module`): Module whose parameter needs to be sharded.
        param_name (str): Name of the parameter of ``module`` that needs to be sharded.
        sharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The specification
            describing how to shard the Tensor.

    Keyword args:
        src_rank (int, optional): The source rank which is used as the ground truth of
            the data for the parameter that would be sharded and scattered
            across the rest of the ranks.
            Default: 0.
        process_group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.

    .. warning::
        Only :class:`torch.distributed._shard.sharding_spec.ChunkShardingSpec` is
        currently supported as the ``sharding_spec``.
    """
    # Perform some validation first.
    if not hasattr(module, param_name):
        raise AttributeError(f"{module._get_name()} has no attribute `{param_name}`")

    tensor = getattr(module, param_name)
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(
            f"Expected {type(module).__name__}.{param_name} to be a Tensor, but found {type(tensor).__name__}"
        )

    if not tensor.is_contiguous():
        raise ValueError(f"param: {param_name} is not a contiguous Tensor")

    st = _shard_tensor(tensor, sharding_spec, src_rank, process_group)

    # Replace param with ShardedTensor.
    module.register_parameter(param_name, nn.Parameter(st))


# Tracks the current process group in the load context manager.
_CURRENT_PROCESS_GROUP: dist.ProcessGroup | None = None


@contextmanager
def load_with_process_group(process_group):
    """
    Context manager to set the process group with which to load a ShardedTensor.
    """
    global _CURRENT_PROCESS_GROUP
    if _CURRENT_PROCESS_GROUP is not None:
        raise RuntimeError(
            'ProcessGroup already set by previous "load_with_process_group" '
            "context manager"
        )
    _CURRENT_PROCESS_GROUP = process_group
    try:
        yield process_group
    finally:
        _CURRENT_PROCESS_GROUP = None


def _get_current_process_group():
    """
    Retrieves the current process group set by ``load_with_process_group``.
    If not set, it just returns the default group.
    """
    global _CURRENT_PROCESS_GROUP
    if _CURRENT_PROCESS_GROUP is None:
        return distributed_c10d._get_default_group()
    else:
        return _CURRENT_PROCESS_GROUP


def _reshard_output(
    module: torch.nn.Module, resharding_spec: ShardingSpec
) -> torch.nn.Module:
    """
    Hook a module with output resharding in the forward pass according
    to the given ``resharding_spec``.

    Args:
        module (:class:`torch.nn.Module`): Module whose output needs to be resharded.
        resharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`):
            The specification describing how the output of the module will be resharded.

    Returns:
        A :class:`torch.nn.Module` object with reshard API hooked.
    """

    def hook_func(_module, _input, output):
        if isinstance(output, ShardedTensor):
            return output.reshard(resharding_spec)
        return output

    module.register_forward_hook(hook_func)
    return module


def _collect_local_shard(module: torch.nn.Module) -> torch.nn.Module:
    """
    Hook a module with local shards collection in the forward pass.

    This API is typically used to convert a sharded representation back to data parallel
    representation. In particular, it returns the local tensor for this Shard. If the
    size along the sharding dimension for the local tensor is 1, this dimension is removed
    from the final result. For example a [4, 16] ShardedTensor across 4 ranks is typically
    a local Tensor of size [16] across each rank and not [1, 16] across each rank.

    Args:
        module (:class:`torch.nn.Module`): Module whose output is ShardedTensor and the
            local tensor value needs to be returned.

    Returns:
        A :class:`torch.nn.Module` object with collection API hooked.
    """

    def hook_func(_module, _input, output):
        if isinstance(output, ShardedTensor):
            local_tensor = output.local_tensor()
            # Squeeze the # of dimensions manually, only applicable to ChunkShardingSpec
            sharding_spec = output._sharding_spec
            if (
                isinstance(sharding_spec, ChunkShardingSpec)
                and local_tensor.size(sharding_spec.dim) == 1  # type: ignore[attr-defined, arg-type]
            ):
                local_tensor = local_tensor.squeeze(
                    output._sharding_spec.dim  # type: ignore[attr-defined]
                )
            return local_tensor

    module.register_forward_hook(hook_func)
    return module


def shard_module(module: nn.Module, plan: ShardingPlan, src_rank=0, process_group=None):
    """
    Shards a given module according to the provided sharding `plan`. This method
    first shards all the parameters according to the given sharding `plan`. Then if
    `output_plan` and `return_local_tensor` are specified in the sharding `plan`, it
    will tag the output of modules according `output_plan`, convert the module's
    output back to data parallel according to `return_local_tensor`.

    Needs to be called on all ranks in an SPMD fashion.

    Args:
        module (:class:`torch.nn.Module`): The module to apply sharding to
        plan (:class:`torch.distributed._shard.sharding_plan.ShardingPlan`):
            The ShardingPlan which specified param name to ShardingSpec to apply to
            each parameter.

    Keyword args:
         src_rank (int, optional): The source rank which is used as the ground truth of
            the data for the module that would be sharded and scattered across the rest
            of the ranks.
            Default: 0.
        process_group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
    """
    # record Sharder paths for sanity check on the plan to ensure items in the plan
    # does not conflict with the submodule tree that the Sharder is working with
    sharder_paths = []
    for name, spec in plan.plan.items():
        if isinstance(spec, Sharder):
            sharder_paths.append(name)

    # shard the parameter according to the ShardingPlan
    for name, spec in plan.plan.items():
        if isinstance(spec, ShardingSpec):
            # if found a sharding spec, try to shard the parameter
            module_path, _, param_name = name.rpartition(".")

            for sharder_path in sharder_paths:
                if module_path.startswith(sharder_path):
                    raise RuntimeError(
                        f"ShardingPlan is in-valid, trying to shard a parameter: {name},"
                        f" but there's already a Sharder entry for module {sharder_path},"
                        f" parameter sharding should not conflict with the submodule tree"
                        f" that a Sharder is working with!"
                    )

            mod = module.get_submodule(module_path)
            shard_parameter(
                mod, param_name, spec, src_rank=src_rank, process_group=process_group
            )
        elif isinstance(spec, Sharder):
            parent_mod_path, _, _mod_name = name.rpartition(".")
            if name == "":
                raise KeyError("Module path must not be empty for custom sharder!")
            mod = module.get_submodule(name)
            parent_mod = module.get_submodule(parent_mod_path)
            sharded_mod = spec.shard(mod)
            # swap this submodule with the sharded module
            parent_mod.mod_name = sharded_mod
        else:
            raise TypeError(
                f"Only `ShardingSpec` and `Sharder` are supported to shard '{name}'"
            )

    # reshard output if there's an entry in `reshard_output` for this module
    if plan.output_plan is not None:
        for module_path, output_spec in plan.output_plan.items():
            if isinstance(output_spec, ShardingSpec):
                mod = module.get_submodule(module_path)
                _reshard_output(mod, output_spec)
            else:
                raise TypeError(
                    f"Only `ShardingSpec` is supported as output_plan for '{module_path}'"
                )
    # convert the output back to data parallel for the modules appears in
    # `return_local_tensor` of the plan, we will call `_collect_local_shard`
    # to collect the local tensor for output of modules
    if plan.return_local_tensor is not None:
        for module_path in plan.return_local_tensor:
            mod = module.get_submodule(module_path)
            _collect_local_shard(mod)
<|endoftext|># mypy: allow-untyped-defs
from dataclasses import dataclass
from functools import reduce

from torch.distributed.remote_device import _remote_device


@dataclass
class ShardMetadata:
    """
    Represents a shard of the overall Tensor including its
    offsets, lengths and device placement.

    Args:
        shard_offsets(List[int]): Offsets in the original tensor indicating
            the start offsets for this shard. Should have the same rank as
            the original tensor.
        shard_sizes(List[int]): Integers indicating the size of each
            dimension for this shard. Should have the same rank as the
            original tensor.
        placement(:class:`torch.distributed._remote_device`):
            Specifies the placement of this shard.
    """

    __slots__ = ["shard_offsets", "shard_sizes", "placement"]

    shard_offsets: list[int]
    shard_sizes: list[int]
    placement: _remote_device | None

    def __init__(
        self,
        shard_offsets: list[int],
        shard_sizes: list[int],
        placement: str | _remote_device | None = None,
    ):
        self.shard_offsets = shard_offsets
        self.shard_sizes = shard_sizes
        if isinstance(placement, str):
            self.placement = _remote_device(placement)
        else:
            self.placement = placement
        if len(self.shard_offsets) != len(self.shard_sizes):
            raise ValueError(
                f"shard_offsets and shard_sizes should have "
