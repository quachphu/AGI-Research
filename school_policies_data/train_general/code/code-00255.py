    def __tensor_unflatten__(inner_tensors, flatten_spec, outer_size, outer_stride):
        if flatten_spec is None:
            raise AssertionError(
                "Expecting spec to be not None from `__tensor_flatten__` return value!"
            )
        local_tensor = inner_tensors["_local_tensor"]
        mesh = inner_tensors["device_mesh"]
        placements, old_tensor_meta, shard_order, requires_grad = flatten_spec
        unflatten_tensor_meta = TensorMeta(
            shape=outer_size,
            stride=outer_stride,
            dtype=old_tensor_meta.dtype,
        )
        unflatten_spec = DTensorSpec(
            mesh,
            placements,
            tensor_meta=unflatten_tensor_meta,
            shard_order=shard_order,
        )
        # pyrefly: ignore [bad-argument-type]
        return DTensor(
            # pyrefly: ignore [bad-argument-count]
            local_tensor,
            unflatten_spec,
            # pyrefly: ignore [unexpected-keyword]
            requires_grad=requires_grad,
        )

    def _stable_hash_for_caching(self) -> str:
        """
        Return a stable hash for AOT autograd caching.
        [See note: Tensor subclass stable hashing for AOT autograd cache]
        """
        # Include local tensor device so different ranks produce distinct keys.
        # AOTAutogradCachePickler has no device_id_agnostic normalization (unlike
        # FxGraphCachePickler), so ranks sharing a disk cache need separate entries.
        device = str(self._local_tensor.device)
        cache_data = self._spec._stable_hash() + str(self.requires_grad) + device
        return hashlib.blake2b(cache_data.encode(), digest_size=16).hexdigest()

    def __coerce_tangent_metadata__(self):
        if not any(isinstance(p, Partial) for p in self.placements):
            return self
        placements = [
            Replicate() if isinstance(p, Partial) else p for p in self.placements
        ]
        return self.redistribute(device_mesh=self.device_mesh, placements=placements)

    def __coerce_same_metadata_as_tangent__(self, flatten_spec, expected_type=None):
        if expected_type is not None:
            return None

        (placements, _, _, _) = flatten_spec
        return self.redistribute(
            device_mesh=self.device_mesh,
            placements=placements,
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):  # type: ignore[override]
        # Base DTensor is normally dispatched via a C++ fast path (see #167051)
        # and never reaches here. This implementation exists so that DTensor
        # subclasses can delegate back via super().__torch_dispatch__().
        # It unwraps subclass instances to base DTensor and re-calls the op,
        # which re-enters dispatch and hits the C++ fast path.
        def unwrap(t):
            if isinstance(t, DTensor) and type(t) is not DTensor:
                # pyrefly: ignore [bad-argument-type]
                return DTensor(
                    # pyrefly: ignore [bad-argument-count]
                    t._local_tensor,
                    t._spec,
                    # pyrefly: ignore [unexpected-keyword]
                    requires_grad=t.requires_grad,
                )
            return t

        args = torch.utils._pytree.tree_map(unwrap, args)
        kwargs = torch.utils._pytree.tree_map(unwrap, kwargs or {})
        return func(*args, **kwargs)

    @staticmethod
    def from_local(
        local_tensor: torch.Tensor,
        device_mesh: DeviceMesh | None = None,
        placements: Sequence[Placement] | None = None,
        *,
        run_check: bool = False,
        shape: torch.Size | None = None,
        stride: tuple[int, ...] | None = None,
        grad_placements: Sequence[Placement] | None = None,
    ) -> "DTensor":
        """
        Create a :class:`DTensor` from a local torch.Tensor on each rank
        according to the ``device_mesh`` and ``placements`` specified.

        Args:
            local_tensor (torch.Tensor): local torch.Tensor on each rank.
            device_mesh (:class:`DeviceMesh`, optional): DeviceMesh to place the
                tensor, if not specified, must be called under a DeviceMesh
                context manager, default: None
            placements (List[:class:`Placement`], optional): the placements that
                describes how to place the local torch.Tensor on DeviceMesh, must
                have the same number of elements as ``device_mesh.ndim``.

        Keyword args:
            run_check (bool, optional): at a cost of extra communications, perform
                sanity check across ranks to check each local tensor's meta information
                to ensure correctness. If have :class:`Replicate` in ``placements``, the
                data on first rank of the device mesh dimension will be broadcasted
                to other ranks. default: False
            shape (torch.Size, optional): A List of int which specifies the size of
                DTensor which build on top of `local_tensor`. Note this needs to be
                provided if the shape of ``local_tensor`` are different across the ranks.
                If not provided, ``shape`` will be computed assuming the given distributed
                tensor is evenly sharded across ranks. default: None
            stride (tuple, optional): A List of int which specifies the stride of DTensor.
                If not provided, ``stride`` will be computed assuming the given distributed
                tensor is evenly sharded across ranks. default: None
            grad_placements (List[:class:`Placement`], optional): specifies the expected
                input gradient placements. The input gradient (a plain tensor) will be
                redistributed to this placement before exiting DTensor. If not
                specified, follows the default placement guarantees below. default: None

        Returns:
            A :class:`DTensor` object

        Raises:
            ValueError: If ``placements`` contains mixed :class:`Partial` reduce types
                (e.g., both ``Partial("sum")`` and ``Partial("max")``). All Partial
                placements must use the same reduce operation.

        .. note:: When ``run_check=False``, it is the user's responsibility to ensure the
            local tensor passed in is correct across ranks (i.e. the tensor is sharded for
            the ``Shard(dim)`` placement or replicated for the ``Replicate()`` placement).
            If not, the behavior of the created DTensor is undefined.

        .. note:: ``from_local`` is differentiable, the `requires_grad` of the created
            `DTensor` object will depend on if `local_tensor` requires_grad or not.

        .. note:: During backward, ``from_local`` provides the following gradient placement
            guarantees. For each mesh dimension, the gradient placement maps as follows:

            +---------------------+--------------------+
            | Forward Placement   | Gradient Placement |
            +=====================+====================+
            | ``Shard``           | ``Shard``          |
            +---------------------+--------------------+
            | ``Replicate``       | ``Replicate``      |
            +---------------------+--------------------+
            | ``Partial``         | ``Replicate``      |
            +---------------------+--------------------+

            When the forward placement is :class:`Partial`, we always redistribute the gradient
            to :class:`Replicate` instead of keeping it :class:`Partial`. This may not be the most
            efficient option, but it avoids ambiguity and provides clearer gradient semantics to users.
        """
        # `local_tensor` argument cannot be DTensor
        if isinstance(local_tensor, DTensor):
            raise RuntimeError(
                f"the local_tensor argument only accepts torch.Tensor but got {type(local_tensor)} value."
            )

        # if same shape/dtype, no need to run_check, if not, must allgather
        # the metadatas to check the size/dtype across ranks
        # There should be no data communication unless there's replication
        # strategy, where we broadcast the replication from the first rank
        # in the mesh dimension
        device_mesh = device_mesh or _mesh_resources.get_current_mesh()
        device_type = device_mesh.device_type

        # convert the local tensor to desired device base on device mesh's device_type
        if device_type != local_tensor.device.type and not local_tensor.is_meta:
            local_tensor = local_tensor.to(device_type)

        # set default placements to replicated if not specified
        if placements is None:
            placements = [Replicate() for _ in range(device_mesh.ndim)]
        else:
            placements = list(placements)
            for idx, placement in enumerate(placements):
                # normalize shard dim to be positive
                if isinstance(placement, Shard | _StridedShard):
                    if placement.dim < 0:
                        normalized_dim = placement.dim + local_tensor.ndim
                        if type(placement) is _StridedShard:
                            placements[idx] = _StridedShard(
                                normalized_dim, split_factor=placement.split_factor
                            )
                        elif type(placement) is Shard:
                            placements[idx] = Shard(normalized_dim)

        # Validate that placements don't contain mixed Partial reduce types
        assert_no_mixed_partial_types(placements)

        # `from_local` is differentiable, and the gradient of the dist tensor this function
        # created should flow back the gradients to the local_tensor, so we call an autograd
        # function to construct the dist tensor instead.
        return _FromTorchTensor.apply(  # pyre-ignore[16]: autograd func
            local_tensor,
            device_mesh,
            tuple(placements),
            run_check,
            shape,
            stride,
            tuple(grad_placements) if grad_placements is not None else None,
        )

    def to_local(
        self, *, grad_placements: Sequence[Placement] | None = None
    ) -> torch.Tensor:
        """
        Get the local tensor of this DTensor on its current rank. For sharding it returns
        a local shard of the logical tensor view, for replication it returns the replica on
        its current rank.

        Keyword args:
            grad_placements (List[:class:`Placement`], optional): the placements describes
                the future layout of any gradient layout of the Tensor returned from this
                function.
                `to_local` converts DTensor to local tensor and the returned local tensor
                might not be used as the original DTensor layout later in the code. This
                argument is the hint that user can give to autograd in case the gradient
                layout of the returned tensor does not match the original DTensor layout.
                If not specified, we will assume the gradient layout remains the same
                as the original DTensor and use that for gradient computation.

        Returns:
            A :class:`torch.Tensor` or ``AsyncCollectiveTensor`` object. it represents the
            local tensor on its current rank. When an ``AsyncCollectiveTensor`` object is returned,
            it means the local tensor is not ready yet (i.e. communication is not finished). In this
            case, user needs to call ``wait`` to wait the local tensor to be ready.

        .. note:: ``to_local`` is differentiable, the ``requires_grad`` of the local tensor returned
            will depend on if the `DTensor` requires_grad or not.
        """
        if not torch.is_grad_enabled():
            result = self._local_tensor
        else:
            if grad_placements is not None and not isinstance(grad_placements, tuple):
                grad_placements = tuple(grad_placements)
            result = _ToTorchTensor.apply(
                self, grad_placements
            )  # pyre-ignore[16]: autograd func

        # Preserve nn.Parameter-ness: if self is an nn.Parameter (i.e. has the
        # _is_param flag, which is how Parameter is represented for custom
        # tensor subclasses like DTensor), make the returned local tensor also
        # satisfy isinstance(result, nn.Parameter). See gh-166156.
        #
        # BC: this branch only runs when self has `_is_param` set, i.e. self
        # is already a Parameter(DTensor). All raw-DTensor call sites are
        # byte-for-byte unchanged, including the historical identity guarantee
        # `to_local() is self._local_tensor` in the no-grad path. The only
        # observable change is the targeted bug fix: Parameter(DTensor) inputs
        # no longer return a plain Tensor.
        #
        # We probe `_is_param` via getattr rather than isinstance(self,
        # nn.Parameter) for two reasons: (1) it avoids importing torch.nn at
        # this layer, and (2) it matches the exact mechanism that
        # nn.parameter._ParameterMeta.__instancecheck__ uses to recognize
        # Parameter-ness on custom tensor subclasses.
        if getattr(self, "_is_param", False) and not getattr(
            result, "_is_param", False
        ):
            if result is self._local_tensor:
                # Avoid mutating the internal storage of this DTensor by
                # returning a fresh view to attach the flag to.
                result = result.view_as(result)
            result._is_param = True

        return result

    def redistribute(
        self,
        device_mesh: DeviceMesh | None = None,
        placements: Sequence[Placement] | None = None,
        *,
        async_op: bool = False,
        forward_dtype: torch.dtype | None = None,
        backward_dtype: torch.dtype | None = None,
    ) -> "DTensor":
        """
        ``redistribute`` performs necessary collective operations that redistribute the current
        DTensor from its current placements to a new placements, or from its current DeviceMesh
        to a new DeviceMesh. i.e. we can turn a Sharded DTensor to a Replicated DTensor by
        specifying a Replicate placement for each dimension of the DeviceMesh.

        When redistributing from current to the new placements on one device mesh dimension, we
        will perform the following operations including communication collective or local operation:

        1. ``Shard(dim)`` -> ``Replicate()``: ``all_gather``
        2. ``Shard(src_dim)`` -> ``Shard(dst_dim)``: ``all_to_all``
        3. ``Replicate()`` -> ``Shard(dim)``: local chunking (i.e. ``torch.chunk``)
        4. ``Partial()`` -> ``Replicate()``: ``all_reduce``
        5. ``Partial()`` -> ``Shard(dim)``: ``reduce_scatter``
        6. ``Shard(dim)`` -> ``Partial("sum")``: local zero-filling. This
           explicit-only conversion materializes a global-shape local tensor
           on every rank and does not communicate.

        Of conversions to ``Partial``, this public API supports only
        ``Shard(dim)`` -> ``Partial("sum")``. Each rank places its local shard
        at the corresponding offset in a zero-filled logical-shape tensor, so
        summing across ranks reconstructs the original logical tensor.

        ``redistribute`` would correctly figure out the necessary redistribute steps for DTensors
        that are created either on 1-D or N-D DeviceMesh.

        Args:
            device_mesh (:class:`DeviceMesh`, optional): DeviceMesh to place the
                DTensor. If not specified, it would use the current DTensor's DeviceMesh.
                default: None
            placements (List[:class:`Placement`], optional): the new placements that
                describes how to place the DTensor into the DeviceMesh, must
                have the same number of elements as ``device_mesh.ndim``.
                default: replicate on all mesh dimensions

        Keyword args:
            async_op (bool, optional): whether to perform the DTensor redistribute operation
                asynchronously or not. Default: False
            forward_dtype (torch.dtype, optional): cast the local tensor to
                ``forward_dtype`` before running the forward collective.
                The result DTensor will be in ``forward_dtype``. If ``None``,
                no conversion is performed. Default: None
            backward_dtype (torch.dtype, optional): cast the gradient to
                ``backward_dtype`` before running the backward collective.
                After the collective the gradient is always cast back to the
                input DTensor's dtype. If ``None``, the backward collective
                runs at the input DTensor's dtype (not ``forward_dtype``).
                Default: None

        Returns:
            A :class:`DTensor` object

        .. note:: ``redistribute`` is twice-differentiable, which means user do not need to worry about
            the backward formula of the redistribute operation, or its compatibility with autograd for
            second-order gradients. Higher-order differentiation has not been tested (but may work).

        .. note:: ``redistribute`` currently only supports redistributing DTensor on the same DeviceMesh,
            Please file an issue if you need to redistribute DTensor to different DeviceMesh.
        """
        # NOTE: This redistribute API currently only supports out
        # of place redistribution, i.e. it always create a new
        # DTensor object and leave the original one unchanged.

        # if device_mesh is not specified, use the current device_mesh
        device_mesh = device_mesh or self.device_mesh
        # raise error if new placements not specified
        if placements is None:
            raise RuntimeError("placements is needed for redistribute!")

        placements = list(placements)
        for i, placement in enumerate(placements):
            if placement.is_partial() and self.placements[i] != placement:
                source = self.placements[i]
                if not (
                    isinstance(source, Shard)
                    and type(placement) is Partial
                    and placement.reduce_op == "sum"
                ):
                    raise RuntimeError(
                        f"Can not redistribute from {source} to {placement}, "
                        "only Shard to Partial(sum) redistribution is supported!"
                    )
            elif isinstance(placement, Shard) and placement.dim < 0:
                # normalize shard dim to be positive
                placements[i] = Shard(placement.dim + self.ndim)
            elif isinstance(placement, _StridedShard) and placement.dim < 0:
                placements[i] = _StridedShard(
                    placement.dim + self.ndim, split_factor=placement.split_factor
                )
        placements = tuple(placements)

        input_dtype = self._local_tensor.dtype
        forward_dtype = forward_dtype or input_dtype
        # pyre-fixme[16]: `Redistribute` has no attribute `apply`.
        return Redistribute.apply(
            self,
            device_mesh,
            placements,
            async_op,
            {
                "op_dtype": forward_dtype,
                "out_dtype": forward_dtype,
                "backward_options": {
                    # Absent backward_dtype means the backward collective runs
                    # at the input's storage dtype (matching pre-fix behavior).
                    "op_dtype": backward_dtype or input_dtype,
                    "out_dtype": input_dtype,
                },
            },
        )

    def full_tensor(
        self, *, grad_placements: Sequence[Placement] | None = None
    ) -> torch.Tensor:
        """
        Return the full tensor of this DTensor. It will perform necessary collectives
        to gather the local tensors from other ranks in its DeviceMesh and concatenate
        them together. It's a syntactic sugar of the following code:

        ``dtensor.redistribute(placements=[Replicate()] * mesh.ndim).to_local()``

        Keyword args:
            grad_placements (List[:class:`Placement`], optional): the placements describes
                the future layout of any gradient layout of the full Tensor returned from this
                function.
                `full_tensor` converts DTensor to a full torch.Tensor and the returned torch.tensor
                might not be used as the original replicated DTensor layout later in the code. This
                argument is the hint that user can give to autograd in case the gradient
                layout of the returned tensor does not match the original replicated DTensor layout.
                If not specified, we will assume the gradient layout of the full tensor be replicated.

        Returns:
            A :class:`torch.Tensor` object that represents the full tensor of this DTensor.

        .. note:: ``full_tensor`` is differentiable.
        """

        redist_res = self.redistribute(
            placements=[Replicate()] * self.device_mesh.ndim, async_op=False
        )
        return _ToTorchTensor.apply(redist_res, grad_placements)

    @property
    def device_mesh(self) -> DeviceMesh:
        """
        The :class:`DeviceMesh` attribute that associates with this DTensor object.

        .. note:: ``device_mesh`` is a read-only property, it can not be set.
        """
        return self._spec.mesh

    @property
    def placements(self) -> tuple[Placement, ...]:
        """
        The placements attribute of this DTensor that describes the layout of this
        DTensor on the its DeviceMesh.

        .. note:: ``placements`` is a read-only property, it can not be set.
        """
        return self._spec.placements

    def _raise_if_contains_partial_placements(self) -> None:
        """
        Raise an error if the DTensor contains partial placements.
        """
        for placement in self._spec.placements:
            if not isinstance(placement, Partial):
                continue

            raise ValueError(
                "Any checkpointing related operations are not supported for "
                "DTensor with partial placements!"
            )

    def __create_write_items__(self, fqn: str, object: Any):
        self._raise_if_contains_partial_placements()
        from torch.distributed.checkpoint.planner_helpers import (
            _create_write_items_for_dtensor,
        )

        if hasattr(self._local_tensor, "__create_write_items__"):
            return self._local_tensor.__create_write_items__(fqn, object)  # type: ignore[attr-defined]
        elif isinstance(self._local_tensor, torch.Tensor):
            return [_create_write_items_for_dtensor(fqn, object)]
        else:
            raise RuntimeError("Unsupported tensor type!")

    def __create_chunk_list__(self):
        """
        Return a list of ChunkStorageMetadata, which is a dataclass that describes the size/offset of the local shard/replica
        on current rank. For DTensor, each rank will have a single local shard/replica, so the returned list usually only
        has one element.

        This dunder method is primariy used for distributed checkpoint purpose.

        Returns:
            A List[:class:`ChunkStorageMetadata`] object that represents the shard size/offset on the current rank.
        """
        self._raise_if_contains_partial_placements()
        from torch.distributed.checkpoint.planner_helpers import (
            _create_chunk_from_dtensor,
        )

        if hasattr(self._local_tensor, "__create_chunk_list__"):
            return self._local_tensor.__create_chunk_list__()  # type: ignore[attr-defined]
        elif isinstance(self._local_tensor, torch.Tensor):
            return [_create_chunk_from_dtensor(self)]
        else:
            raise RuntimeError("Unsupported tensor type!")

    def __get_tensor_shard__(self, index):
        self._raise_if_contains_partial_placements()
        if hasattr(self._local_tensor, "__get_tensor_shard__"):
            return self._local_tensor.__get_tensor_shard__(index)  # type: ignore[attr-defined]
        elif isinstance(self._local_tensor, torch.Tensor):
            return self.to_local()
        else:
            raise RuntimeError("Unsupported tensor type!")


def distribute_tensor(
    tensor: torch.Tensor,
    device_mesh: DeviceMesh | None = None,
    placements: Sequence[Placement] | None = None,
    *,
    src_data_rank: int | None = 0,
) -> DTensor:
    """
    Distribute a leaf ``torch.Tensor`` (i.e. nn.Parameter/buffers) to the ``device_mesh`` according
    to the ``placements`` specified. The rank of ``device_mesh`` and ``placements`` must be the
    same. The ``tensor`` to distribute is the logical or "global" tensor, and the API would use
    the ``tensor`` from first rank of the DeviceMesh dimension as the source of truth to preserve
    the single-device semantic. If you want to construct a DTensor in the middle of the Autograd
    computation, please use :meth:`DTensor.from_local` instead.

    Args:
        tensor (torch.Tensor): torch.Tensor to be distributed. Note that if you
            want to shard a tensor on a dimension that is not evenly divisible by
            the number of devices in that mesh dimension, we use ``torch.chunk``
            semantic to shard the tensor and scatter the shards. The uneven sharding
            behavior is experimental and subject to change.
        device_mesh (:class:`DeviceMesh`, optional): DeviceMesh to distribute the
            tensor, if not specified, must be called under a DeviceMesh context
            manager, default: None
        placements (List[:class:`Placement`], optional): the placements that
            describes how to place the tensor on DeviceMesh, must have the same
            number of elements as ``device_mesh.ndim``. If not specified, we will
            by default replicate the tensor across the ``device_mesh`` from the
            first rank of each dimension of the `device_mesh`.

    Keyword args:
        src_data_rank (int, optional): the rank of the source data for the logical/global tensor, it is
            used by :meth:`distribute_tensor` to scatter/broadcast the shards/replicas to other ranks.
            By default, we use ``group_rank=0`` on each DeviceMesh dimension as the source data to preserve
            the single-device semantic. If passing ``None`` explicitly, :meth:`distribute_tensor` simply uses
            its local data instead of trying to preserve the single-device semantic via scatter/broadcast.
            Default: 0

    Returns:
        A :class:`DTensor` or ``XLAShardedTensor`` object.

    Raises:
        ValueError: If ``placements`` contains mixed :class:`Partial` reduce types
            (e.g., both ``Partial("sum")`` and ``Partial("max")``). All Partial
            placements must use the same reduce operation.

    .. note::
        When initialize the DeviceMesh with the ``xla`` device_type, ``distribute_tensor``
        return `XLAShardedTensor` instead. see `this issue <https://github.com/pytorch/pytorch/issues/92909>`__
        for more details. The XLA integration is experimental and subject to change.
    """

    torch._C._log_api_usage_once("torch.dtensor.distribute_tensor")

    # get default device mesh if there's nothing specified
    device_mesh = device_mesh or _mesh_resources.get_current_mesh()
    device_type = device_mesh.device_type
    if device_type == "xla":
        try:
            # call PyTorch/XLA SPMD for `xla` backend type device mesh.
            # This returns XLAShardedTensor
            from torch_xla.distributed.spmd import (  # type:ignore[import]
                xla_distribute_tensor,
            )

            return xla_distribute_tensor(tensor, device_mesh, placements)  # type:ignore[return-value]
        except ImportError as e:
            msg = "To use DTensor API with xla, you must install the torch_xla package!"
            raise ImportError(msg) from e

    if not tensor.is_leaf:
        raise RuntimeError(
            "`distribute_tensor` should be used to distribute leaf tensors! but found non-leaf tensor!"
        )

    # convert tensor to the corresponding device type if it's not in that device type
    if device_type != tensor.device.type and not tensor.is_meta:
        tensor = tensor.to(device_type)

    # set default placements to replicated if not specified
    if placements is None:
        placements = [Replicate() for _ in range(device_mesh.ndim)]

    if len(placements) != device_mesh.ndim:
        raise ValueError(
            f"`placements` must have the same length as `device_mesh.ndim`! "
            f"Found placements length: {len(placements)}, and device_mesh.ndim: {device_mesh.ndim}."
        )

    # Validate that placements don't contain mixed Partial reduce types
    assert_no_mixed_partial_types(placements)
    if isinstance(tensor, DTensor):
        # if the tensor is already a DTensor, we need to check:
        # 1. if we can further shard this DTensor if the two device mesh belong to
        #   the same parent mesh and further sharding is possible.
        # 2. check if device mesh and placements are the same
        if tensor.device_mesh != device_mesh:
            raise ValueError(
                f"Cannot distribute a DTensor with device mesh {tensor.device_mesh} "
                f"to a different device mesh {device_mesh}."
            )
        if tensor.placements != tuple(placements):
            raise ValueError(
                f"Cannot distribute a DTensor with placements {tensor.placements} "
                f"to a different placements {placements}. do you want to call "
                f"`redistribute` instead?"
            )
        return tensor

    local_tensor = tensor.detach()

    # TODO(xilun): address sharding order
    # distribute the tensor according to the placements.
    placements = list(placements)
    for idx, placement in enumerate(placements):
        if isinstance(placement, Shard | _StridedShard):
            placement_dim = (
                placement.dim + tensor.ndim if placement.dim < 0 else placement.dim
            )
            if isinstance(placement, Shard):
                local_tensor = Shard._make_shard_tensor(
                    placement_dim, local_tensor, device_mesh, idx, src_data_rank
                )
                placements[idx] = Shard(placement_dim)
            else:
                local_tensor = _StridedShard._make_shard_tensor(
                    placement_dim,
                    local_tensor,
                    device_mesh,
                    idx,
                    src_data_rank,
                    split_factor=placement.split_factor,
                )
                placements[idx] = _StridedShard(
                    placement_dim, split_factor=placement.split_factor
                )
        elif isinstance(placement, Replicate):
            local_tensor = Replicate._make_replicate_tensor(
                local_tensor, device_mesh, idx, src_data_rank
            )
        elif isinstance(placement, Partial):
            local_tensor = Replicate._make_replicate_tensor(
                local_tensor, device_mesh, idx, src_data_rank
            )
            local_tensor = placement._partition_value(local_tensor, device_mesh, idx)
        else:
            raise RuntimeError(
                f"Trying to distribute tensor with unsupported placements {placement} on device mesh dimension {idx}!"
            )
    placements = tuple(placements)

    if local_tensor is None:
        raise AssertionError("distributing a tensor should not be None")
    # detach the local tensor passed to DTensor since after the construction
    # of DTensor, autograd would work on top of DTensor instead of local tensor
    spec = DTensorSpec(
        mesh=device_mesh,
        placements=placements,
        tensor_meta=TensorMeta(
            shape=tensor.size(),
            stride=tensor.stride(),
            dtype=tensor.dtype,
        ),
    )
    # pyrefly: ignore [bad-argument-type]
    return DTensor(
        # pyrefly: ignore [bad-argument-count]
        local_tensor.requires_grad_(tensor.requires_grad),
        spec,
        # pyrefly: ignore [unexpected-keyword]
        requires_grad=tensor.requires_grad,
    )


@deprecated("Please use `distribute_tensor` with `src_data_rank=None` instead.")
def _shard_tensor(
    full_tensor: torch.Tensor,
    placements: Sequence[Shard],
    device_mesh: DeviceMesh | None = None,
) -> "DTensor":
    """
    Locally shards a full tensor based on indicated sharding arrangement, and
    returns a DTensor containing the local shard.

    .. warning:: This is a private API that is subject to change. It skips the
        communication otherwise required by `distribute_tensor`. It is only
        applicable to cases where all ranks have the same `full_tensor`. For
        example, in distributed inference all ranks load from the same
        checkpoint. This API will not check for data equality between ranks, it
        is thus user's responsibility to ensure the `full_tensor` is the same
        across ranks.

    Args:
        full_tensor (torch.Tensor): the full tensor to be sharded.
        placements (Sequence[:class:`Shard`]): the placements that
            describes how to place the local tensor on DeviceMesh.
        device_mesh (:class:`DeviceMesh`, optional): DeviceMesh to place the
            DTensor.  Must have same dimension as the number of placements.
            If not specified, would be retrieve from current context.

    Returns:
        A :class:`DTensor` object with the shard as its local tensor.

    Examples:
        >>> # xdoctest: +SKIP("need world_size and rank")
        >>> device_mesh = dist.init_device_mesh("cuda", (world_size,))
        >>> full_tensor = torch.arange(world_size, device=f"cuda:{rank}")
        >>> dtensor = _shard_tensor(full_tensor, [Shard(1)], device_mesh)
    """
    return distribute_tensor(full_tensor, device_mesh, placements, src_data_rank=None)


def distribute_module(
    module: nn.Module,
    device_mesh: DeviceMesh | None = None,
    partition_fn: Callable[[str, nn.Module, DeviceMesh], None] | None = None,
    input_fn: Callable[[nn.Module, Any, DeviceMesh], None] | None = None,
    output_fn: Callable[[nn.Module, Any, DeviceMesh], None] | None = None,
) -> nn.Module:
    """
    Convert a module's parameters/buffers to :class:`DTensor` s, and optionally
    convert its inputs/outputs at runtime, using the three user-provided callbacks.

    The behavior of the three callbacks:

    1. ``partition_fn`` shards the module's parameters/buffers *before* runtime
       execution (i.e. it converts plain ``torch.Tensor`` parameters into
       :class:`DTensor` parameters). Any parameter/buffer not converted by
       ``partition_fn`` is replicated across ``device_mesh`` afterwards. If
       ``partition_fn`` is ``None``, *all* parameters/buffers are replicated.
    2. ``input_fn`` / ``output_fn`` convert the module's inputs/outputs *during*
       runtime execution (e.g. convert plain tensor inputs into :class:`DTensor`,
       or convert :class:`DTensor` outputs back into plain ``torch.Tensor``).

    Args:
        module (:class:`nn.Module`): user module to be partitioned.
        device_mesh (:class:`DeviceMesh`): the device mesh to place the module. If
            ``None``, the current mesh from the ambient mesh context is used.
        partition_fn (Callable): a function with signature
            ``partition_fn(name: str, module: nn.Module, device_mesh: DeviceMesh) -> None``.
            It is called once for *every* submodule (including ``module`` itself),
            where ``name`` is the submodule's fully-qualified name (``""`` for the
            top-level module) as returned by :meth:`nn.Module.named_modules`. The
            function is expected to mutate ``module`` in place, typically by calling
            :func:`distribute_tensor` and :meth:`register_parameter` to replace
            selected parameters with :class:`DTensor` parameters. Its return value
            is ignored. If not specified, all module parameters are replicated
            across ``device_mesh``.
        input_fn (Callable): a function with signature
            ``input_fn(module: nn.Module, inputs: Inputs, device_mesh: DeviceMesh) -> Inputs``,
            where ``Inputs`` is the tuple of positional arguments to ``forward`` (one
            entry per positional argument, e.g. ``(x,)`` for ``forward(self, x)``;
            keyword arguments are not forwarded to ``input_fn``). Installed on the
            top-level ``module`` as a ``forward_pre_hook``, it returns the new
            argument tuple -- the same structure as ``inputs``, typically with each
            ``torch.Tensor`` entry converted to a :class:`DTensor` -- which replaces
            ``inputs`` for the actual ``forward`` call.
        output_fn (Callable): a function with signature
            ``output_fn(module: nn.Module, outputs: Outputs, device_mesh: DeviceMesh) -> Outputs``,
            where ``Outputs`` is the return type of ``module.forward`` (a single
            tensor, or any nested structure of tensors such as a tuple/dict), with the
            tensors typically being :class:`DTensor` s. Installed on the top-level
            ``module`` as a ``forward_hook``, it returns the replacement output of the
            same structure -- typically with each :class:`DTensor` converted back to a
            plain ``torch.Tensor`` via :meth:`DTensor.to_local` or
            :meth:`DTensor.full_tensor`, or redistributed to different placements.

    Returns:
        A module whose parameters/buffers are all :class:`DTensor` s (the same
        ``module`` object, mutated in place).

    .. warning::
        ``input_fn`` and ``output_fn`` previously took two arguments
        ``(inputs, device_mesh)`` / ``(outputs, device_mesh)``. That form is
        deprecated; use the three-argument form documented above.

    Example::

        import torch.nn as nn
        from torch.distributed.tensor import (
            DTensor,
            Replicate,
            Shard,
            distribute_module,
            distribute_tensor,
            init_device_mesh,
        )


        class MyModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(8, 8)

            def forward(self, x):
                return self.fc(x)


        mesh = init_device_mesh("cuda", (4,))


        def shard_params(name, mod, mesh):
            if isinstance(mod, nn.Linear):
                for pname, param in mod.named_parameters(recurse=False):
                    mod.register_parameter(
                        pname, nn.Parameter(distribute_tensor(param, mesh, [Shard(0)]))
                    )


        # Replicate the input across the mesh before forward.
        def to_replicated(mod, inputs, mesh):
            return tuple(
                distribute_tensor(x, mesh, [Replicate()])
                if isinstance(x, torch.Tensor)
                else x
                for x in inputs
            )


        # Convert the DTensor output back to a plain torch.Tensor after forward.
        def to_local(mod, outputs, mesh):
            return outputs.full_tensor() if isinstance(outputs, DTensor) else outputs


        sharded = distribute_module(
            MyModule(),
            mesh,
            partition_fn=shard_params,
            input_fn=to_replicated,
            output_fn=to_local,
        )

    .. note::
        When initializing the ``DeviceMesh`` with the ``xla`` device_type,
        ``distribute_module`` returns an ``nn.Module`` with PyTorch/XLA SPMD
        annotated parameters. See
        `this issue <https://github.com/pytorch/pytorch/issues/92909>`__
        for more details. The XLA integration is experimental and subject to change.

    """

    torch._C._log_api_usage_once("torch.dtensor.distribute_module")

    already_distributed = getattr(module, "_distribute_module_applied", False)
    if already_distributed:
        raise RuntimeError(
            "distribute_module should only be called once on a module, "
            "but it has already been called on this module!"
        )

    device_mesh = device_mesh or _mesh_resources.get_current_mesh()
    device_type = device_mesh.device_type
    if device_type == "xla":
        try:
            # This function annotates all module parameters for auto-partitioning with
            # PyTorch/XLA SPMD or explicitly partition to :class:`XLAShardedTensor` parameters
            # according to the `partition_fn` specified.
            from torch_xla.distributed.spmd import (  # type:ignore[import]
                xla_distribute_module,
            )

            return xla_distribute_module(
                module, device_mesh, partition_fn, input_fn, output_fn
            )  # type:ignore[return-value]
        except ImportError as e:
            msg = "To use DTensor API with xla, you must install the torch_xla package!"
            raise ImportError(msg) from e

    def replicate_module_params_buffers(m: nn.Module, mesh: DeviceMesh) -> None:
        # This function loop over the immediate module parameters and
        # buffers, replicate all non DTensor params/buffers to DTensor
        # parameters/buffers, if they have not been partitioned in the
        # partition_fn, we can't easily use `module._apply` here
        # because we don't know what happened inside partition_fn as
        # user could do anything, i.e. install hooks, and we want to
        # preserve those.
        full_replicate = [Replicate()] * mesh.ndim
        for key, param in m._parameters.items():
            if param is not None and not isinstance(param, DTensor):
                m.register_parameter(
                    key,
                    nn.Parameter(
                        distribute_tensor(param.data, mesh, full_replicate),
                        requires_grad=param.requires_grad,
                    ),
                )
        for key, buffer in m._buffers.items():
            if buffer is not None and not isinstance(buffer, DTensor):
                m._buffers[key] = distribute_tensor(buffer, mesh, full_replicate)

    if partition_fn is None:
        # if partition_fn not specified, we by default replicate
        # all module params/buffers
        for submod in module.modules():
            replicate_module_params_buffers(submod, device_mesh)
    else:
        # apply partition_fun to submodules
        for name, submod in module.named_modules():
            partition_fn(name, submod, device_mesh)
            replicate_module_params_buffers(submod, device_mesh)

    # register input_fn as module forward pre hook
    if input_fn is not None:
        # check the input_fn signature
        num_args = len(inspect.signature(input_fn).parameters)
        if num_args == 2:
            # input_fn only takes in inputs and device mesh
            warnings.warn(
                "Deprecating input_fn that takes two arguments (inputs, device_mesh), "
                "please use input_fn that takes in (module, inputs, device_mesh) instead!",
                FutureWarning,
                stacklevel=2,
            )
            module.register_forward_pre_hook(
                lambda _, inputs: input_fn(inputs, device_mesh)  # type: ignore[call-arg]
            )
        elif num_args == 3:
            # input_fn takes in module, inputs, device mesh
            module.register_forward_pre_hook(
                lambda mod, inputs: input_fn(mod, inputs, device_mesh)
            )
        else:
            raise ValueError(
                f"input_fn should take in 3 arguments, but got {num_args} arguments!"
            )
    # register output_fn as module forward hook
    if output_fn is not None:
        num_args = len(inspect.signature(output_fn).parameters)
        if num_args == 2:
            # output_fn only takes in outputs and device mesh
            warnings.warn(
                "Deprecating output_fn that takes two arguments (inputs, device_mesh), "
                "please use output_fn that takes in (module, inputs, device_mesh) instead!",
                FutureWarning,
                stacklevel=2,
            )
            module.register_forward_hook(
                lambda mod, inputs, outputs: output_fn(outputs, device_mesh)  # type: ignore[call-arg]
            )
        elif num_args == 3:
            module.register_forward_hook(
                lambda mod, inputs, outputs: output_fn(mod, outputs, device_mesh)
            )
        else:
            raise ValueError(
                f"output_fn should take in 3 arguments, but got {num_args} arguments!"
            )

    module._distribute_module_applied = True  # type: ignore[assignment]
    return module


# Below are tensor factory function APIs, which are used to create a DTensor directly. We need
# to make separate factory function APIs because tensor subclass could not override the tensor
# factory methods, and we need user to call the factory functions with user intended device_mesh
# and placements to create a proper DTensor.


@maybe_run_for_local_tensor
def _linspace_local_tensor(
    init_op, start, end, total_steps, local_steps, offset, **kwargs
):
    local_start = start
    local_end = end
    if local_steps > 0 and total_steps > 1:
        step = (end - start) / (total_steps - 1)
        local_start = start + offset * step
        local_end = local_start + step * (local_steps - 1)
    elif local_steps > 0:
        local_end = start
    return init_op(local_start, local_end, steps=local_steps, **kwargs)


@maybe_run_for_local_tensor
def _logspace_local_tensor(
    init_op, start, end, base, total_steps, local_steps, offset, **kwargs
):
    local_start = start
    local_end = end
    if local_steps > 0 and total_steps > 1:
        step = (end - start) / (total_steps - 1)
        local_start = start + offset * step
        local_end = local_start + step * (local_steps - 1)
    elif local_steps > 0:
        local_end = start
    return init_op(local_start, local_end, steps=local_steps, base=base, **kwargs)


def _dtensor_init_helper(  # type: ignore[no-untyped-def]
    init_op,
    size: torch.Size,
    device_mesh: DeviceMesh | None = None,
    placements: Sequence[Placement] | None = None,
    **kwargs,
) -> DTensor:
    # if device_mesh is None, use the one from mesh resources
    device_mesh = device_mesh or _mesh_resources.get_current_mesh()
    kwargs["device"] = device_mesh.device_type

    # set default placements to replicated if not specified
    placements = placements or tuple(Replicate() for _ in range(device_mesh.ndim))

    # check device_mesh against placements
    if device_mesh.ndim != len(placements):
        raise AssertionError("mesh dimension does not match the length of placements")

    if kwargs["layout"] != torch.strided:
        raise AssertionError("layout value not supported!")
    torch_stride = torch._prims_common.make_contiguous_strides_for(size)

    # get local tensor shape
    local_shape, global_offset = compute_local_shape_and_global_offset(
        size,
        device_mesh,
        placements,
        skip_offset=init_op not in (torch.linspace, torch.logspace),
    )

    # initialize the local tensor
    if init_op is torch.full:
        fill_value = kwargs.pop("fill_value")
        local_tensor = init_op(local_shape, fill_value, **kwargs)
    elif init_op is torch.linspace:
        start = kwargs.pop("start")
        end = kwargs.pop("end")
        local_tensor = _linspace_local_tensor(
            init_op,
            start,
            end,
            size[0],
            local_shape[0],
            global_offset[0],
            **kwargs,
        )
    elif init_op is torch.logspace:
        start = kwargs.pop("start")
        end = kwargs.pop("end")
        base = kwargs.pop("base")
        local_tensor = _logspace_local_tensor(
            init_op,
            start,
            end,
            base,
            size[0],
            local_shape[0],
            global_offset[0],
            **kwargs,
        )
    elif init_op is torch.rand or init_op is torch.randn:
        # this tensor meta is not used except `shape`
        dtype = kwargs.get("dtype", torch.get_default_dtype())

        tensor_meta = TensorMeta(size, torch_stride, dtype)
        spec = DTensorSpec(device_mesh, tuple(placements), tensor_meta=tensor_meta)

        if random.is_rng_supported_mesh(device_mesh) and not random._rng_tracker:
            random._rng_tracker = random.OffsetBasedRNGTracker(device_mesh)

        if random._rng_tracker is None:
            raise AssertionError
        with random._rng_tracker._distribute_region(spec):
            local_tensor = init_op(local_shape, **kwargs)
    else:
        local_tensor = init_op(local_shape, **kwargs)

    spec = DTensorSpec(
        device_mesh,
        tuple(placements),
        tensor_meta=TensorMeta(
            size,
            torch_stride,
            local_tensor.dtype,
        ),
    )

    # pyrefly: ignore [bad-argument-type]
    return DTensor(
        # pyrefly: ignore [bad-argument-count]
        local_tensor,
        spec,
        # pyrefly: ignore [unexpected-keyword]
        requires_grad=kwargs["requires_grad"],
    )


def ones(  # type: ignore[no-untyped-def]
    *size,
    dtype: torch.dtype | None = None,
    layout: torch.layout = torch.strided,
    requires_grad: bool = False,
    device_mesh: DeviceMesh | None = None,
    placements: Sequence[Placement] | None = None,
) -> DTensor:
    """
    Returns a :class:`DTensor` filled with the scalar value 1, with the shape defined
    by the variable argument ``size``.

    Args:
        size (int...): a sequence of integers defining the shape of the output :class:`DTensor`.
            Can be a variable number of arguments or a collection like a list or tuple.
            E.g.: ones(1,2,3..) or ones([1,2,3..]) or ones((1,2,3..))

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned :class:`DTensor`.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).
        layout (:class:`torch.layout`, optional): the desired layout of returned DTensor.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned :class:`DTensor`. Default: ``False``.
        device_mesh: :class:`DeviceMesh` type, contains the mesh info of ranks
        placements: a sequence of :class:`Placement` type: ``Shard``, ``Replicate``

    Returns:
        A :class:`DTensor` object on each rank
    """
    torch_size = normalize_to_torch_size(size)

    return _dtensor_init_helper(
        torch.ones,
        torch_size,
        dtype=dtype,
        layout=layout,
        requires_grad=requires_grad,
        device_mesh=device_mesh,
        placements=placements,
    )


def empty(  # type: ignore[no-untyped-def]
    *size,
    dtype: torch.dtype | None = None,
    layout: torch.layout = torch.strided,
    requires_grad: bool = False,
    device_mesh: DeviceMesh | None = None,
    placements: Sequence[Placement] | None = None,
) -> DTensor:
    """
    Returns a :class:`DTensor` filled with uninitialized data. The shape of the :class:`DTensor`
    is defined by the variable argument ``size``.

    Args:
        size (int...): a sequence of integers defining the shape of the output :class:`DTensor`.
            Can be a variable number of arguments or a collection like a list or tuple.
            E.g.: empty(1,2,3..) or empty([1,2,3..]) or empty((1,2,3..))

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned :class:`DTensor`.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).\
        layout (:class:`torch.layout`, optional): the desired layout of returned :class:`DTensor`.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned :class:`DTensor`. Default: ``False``.
        device_mesh: :class:`DeviceMesh` type, contains the mesh info of ranks
        placements: a sequence of :class:`Placement` type: ``Shard``, ``Replicate``

    Returns:
        A :class:`DTensor` object on each rank
    """
    torch_size = normalize_to_torch_size(size)

    return _dtensor_init_helper(
        torch.empty,
        torch_size,
        dtype=dtype,
        layout=layout,
        requires_grad=requires_grad,
        device_mesh=device_mesh,
        placements=placements,
    )


def full(  # type: ignore[no-untyped-def]
    size,
    fill_value,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout = torch.strided,
    requires_grad: bool = False,
    device_mesh: DeviceMesh | None = None,
    placements: Sequence[Placement] | None = None,
) -> DTensor:
    """
    Returns a :class:`DTensor` filled with ``fill_value`` according to ``device_mesh`` and
    ``placements``, with the shape defined by the argument ``size``.

    Args:
        size (int...): a sequence of integers defining the shape of the output :class:`DTensor`.
            Can be a variable number of arguments or a collection like a list or tuple.
            E.g.: ones(1,2,3..) or ones([1,2,3..]) or ones((1,2,3..))
        fill_value(Scalar): the value to fill the output tensor with.

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned :class:`DTensor`.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).
        layout (:class:`torch.layout`, optional): the desired layout of returned DTensor.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned :class:`DTensor`. Default: ``False``.
        device_mesh: :class:`DeviceMesh` type, contains the mesh info of ranks.
        placements: a sequence of :class:`Placement` type: ``Shard``, ``Replicate``

    Returns:
        A :class:`DTensor` object on each rank
    """
    torch_size = normalize_to_torch_size(size)

    return _dtensor_init_helper(
        torch.full,
        torch_size,
        fill_value=fill_value,
        dtype=dtype,
        layout=layout,
        requires_grad=requires_grad,
        device_mesh=device_mesh,
        placements=placements,
    )


def linspace(
    start,
    end,
    steps,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout = torch.strided,
    requires_grad: bool = False,
    device_mesh: DeviceMesh | None = None,
    placements: Sequence[Placement] | None = None,
) -> DTensor:
    """
    Returns a :class:`DTensor` of size ``steps`` whose values are evenly spaced from
    `start` to `end`.

    Args:
        start (float or :class:`DTensor`): the starting value for the set of points. If
           :class:`DTensor`, it must be 0-dimensional
        end (float or :class:`DTensor`): the ending value for the set of points. If
           :class:`DTensor`, it must be 0-dimensional
        steps (int): size of the constructed :class:`DTensor`

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the data type to perform the computation
            in. Default: if ``None``, uses the global default dtype
            (see :func:`torch.set_default_dtype`) when both `start` and `end` are real,
            and corresponding complex dtype when either is complex.
        layout (:class:`torch.layout`, optional): the desired layout of returned
            :class:`DTensor`. Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned :class:`DTensor`. Default: ``False``.
        device_mesh: :class:`DeviceMesh` type, contains the mesh info of ranks
        placements: a sequence of :class:`Placement` type: ``Shard``, ``Replicate``

    Returns:
        A :class:`DTensor` object on each rank
    """
    if placements is not None and any(isinstance(p, _StridedShard) for p in placements):
        raise AssertionError("linspace does not support _StridedShard placements")

    if isinstance(start, torch.Tensor):
        torch._check(
            start.dim() == 0,
            lambda: "linspace only supports 0-dimensional start and end tensors",
        )
        start = start.item()

    if isinstance(end, torch.Tensor):
        torch._check(
            end.dim() == 0,
            lambda: "linspace only supports 0-dimensional start and end tensors",
        )
        end = end.item()

    torch_size = normalize_to_torch_size((steps,))

    return _dtensor_init_helper(
        torch.linspace,
        torch_size,
        start=start,
        end=end,
        dtype=dtype,
        layout=layout,
        requires_grad=requires_grad,
        device_mesh=device_mesh,
        placements=placements,
    )


def logspace(
    start,
    end,
    steps,
    base=10.0,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout = torch.strided,
    requires_grad: bool = False,
    device_mesh: DeviceMesh | None = None,
    placements: Sequence[Placement] | None = None,
) -> DTensor:
    """
    Returns a :class:`DTensor` of size ``steps`` whose values are evenly spaced from
    ``base`` :sup:`start` to ``base`` :sup:`end`, inclusive, on a logarithmic scale
    with base ``base``.

    Args:
        start (float or :class:`DTensor`): the starting value for the set of points. If
           :class:`DTensor`, it must be 0-dimensional
        end (float or :class:`DTensor`): the ending value for the set of points. If
           :class:`DTensor`, it must be 0-dimensional
        steps (int): size of the constructed :class:`DTensor`

    Keyword args:
        base (float, optional): base of the logarithm function. Default: ``10.0``.
        dtype (:class:`torch.dtype`, optional): the data type to perform the computation
            in. Default: if ``None``, uses the global default dtype
            (see :func:`torch.set_default_dtype`) when both `start` and `end` are real,
            and corresponding complex dtype when either is complex.
        layout (:class:`torch.layout`, optional): the desired layout of returned
            :class:`DTensor`. Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned :class:`DTensor`. Default: ``False``.
        device_mesh: :class:`DeviceMesh` type, contains the mesh info of ranks
        placements: a sequence of :class:`Placement` type: ``Shard``, ``Replicate``

    Returns:
        A :class:`DTensor` object on each rank
    """
    if placements is not None and any(isinstance(p, _StridedShard) for p in placements):
        raise AssertionError("logspace does not support _StridedShard placements")

    if isinstance(start, torch.Tensor):
        torch._check(
            start.dim() == 0,
            lambda: "logspace only supports 0-dimensional start and end tensors",
        )
        start = start.item()

    if isinstance(end, torch.Tensor):
        torch._check(
            end.dim() == 0,
            lambda: "logspace only supports 0-dimensional start and end tensors",
        )
        end = end.item()

    torch_size = normalize_to_torch_size((steps,))

    return _dtensor_init_helper(
        torch.logspace,
        torch_size,
        start=start,
        end=end,
        base=base,
        dtype=dtype,
        layout=layout,
        requires_grad=requires_grad,
        device_mesh=device_mesh,
        placements=placements,
    )


def rand(  # type: ignore[no-untyped-def]
    *size,
    requires_grad: bool = False,
    dtype: torch.dtype | None = None,
    layout: torch.layout = torch.strided,
    device_mesh: DeviceMesh | None = None,
    placements: Sequence[Placement] | None = None,
) -> DTensor:
    """
    Returns a :class:`DTensor` filled with random numbers from a uniform distribution
    on the interval ``[0, 1)``. The shape of the tensor is defined by the variable
    argument ``size``.

    Args:
        size (int...): a sequence of integers defining the shape of the output :class:`DTensor`.
            Can be a variable number of arguments or a collection like a list or tuple.
            E.g.: ones(1,2,3..) or ones([1,2,3..]) or ones((1,2,3..))

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned :class:`DTensor`.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).
        layout (:class:`torch.layout`, optional): the desired layout of returned DTensor.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned :class:`DTensor`. Default: ``False``.
        device_mesh: :class:`DeviceMesh` type, contains the mesh info of ranks.
        placements: a sequence of :class:`Placement` type: ``Shard``, ``Replicate``

    Returns:
        A :class:`DTensor` object on each rank
    """
    torch_size = normalize_to_torch_size(size)

    return _dtensor_init_helper(
        torch.rand,
        torch_size,
        dtype=dtype,
        layout=layout,
        requires_grad=requires_grad,
        device_mesh=device_mesh,
        placements=placements,
    )


def randn(  # type: ignore[no-untyped-def]
    *size,
    requires_grad: bool = False,
    dtype: torch.dtype | None = None,
    layout: torch.layout = torch.strided,
    device_mesh: DeviceMesh | None = None,
    placements: Sequence[Placement] | None = None,
) -> DTensor:
    """
    Returns a :class:`DTensor` filled with random numbers from a normal distribution
    with mean 0 and variance 1. The shape of the tensor is defined by the variable
    argument ``size``.

    Args:
        size (int...): a sequence of integers defining the shape of the output :class:`DTensor`.
            Can be a variable number of arguments or a collection like a list or tuple.
            E.g.: ones(1,2,3..) or ones([1,2,3..]) or ones((1,2,3..))

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned :class:`DTensor`.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).
        layout (:class:`torch.layout`, optional): the desired layout of returned DTensor.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned :class:`DTensor`. Default: ``False``.
        device_mesh: :class:`DeviceMesh` type, contains the mesh info of ranks.
        placements: a sequence of :class:`Placement` type: ``Shard``, ``Replicate``

    Returns:
        A :class:`DTensor` object on each rank
    """
    torch_size = normalize_to_torch_size(size)

    return _dtensor_init_helper(
        torch.randn,
        torch_size,
        dtype=dtype,
        layout=layout,
        requires_grad=requires_grad,
        device_mesh=device_mesh,
        placements=placements,
    )


def zeros(  # type: ignore[no-untyped-def]
    *size,
    requires_grad: bool = False,
    dtype: torch.dtype | None = None,
    layout: torch.layout = torch.strided,
    device_mesh: DeviceMesh | None = None,
    placements: Sequence[Placement] | None = None,
) -> DTensor:
    """
    Returns a :class:`DTensor` filled with the scalar value 0.

    Args:
        size (int...): a sequence of integers defining the shape of the output :class:`DTensor`.
            Can be a variable number of arguments or a collection like a list or tuple.
            E.g.: zeros(1,2,3..) or zeros([1,2,3..]) or zeros((1,2,3..))
    Keyword args:
        requires_grad (bool, optional): If autograd should record operations on the
            returned :class:`DTensor`. Default: ``False``.
        dtype (:class:`torch.dtype`, optional): the desired data type of returned :class:`DTensor`.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).
        layout (:class:`torch.layout`, optional): the desired layout of returned :class:`DTensor`.
            Default: ``torch.strided``.
        device_mesh: :class:`DeviceMesh` type, contains the mesh info of ranks
        placements: a sequence of :class:`Placement` type: ``Shard``, ``Replicate``

    Returns:
        A :class:`DTensor` object on each rank
    """
    torch_size = normalize_to_torch_size(size)

    return _dtensor_init_helper(
        torch.zeros,
        torch_size,
        dtype=dtype,
        layout=layout,
        requires_grad=requires_grad,
        device_mesh=device_mesh,
        placements=placements,
    )


# Module-level alias for DTensor.from_local to support FX code generation.
# FX uses __name__ instead of __qualname__ when generating code, which loses
# the class context for static methods. This alias allows generated code like
# `torch.distributed.tensor._api.from_local(...)` to work correctly.
from_local = DTensor.from_local

_register_distributed_opaque_types()
<|endoftext|># mypy: allow-untyped-defs
import logging
import threading
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from functools import lru_cache
from itertools import chain
from typing import cast

import torch
from torch._guards import detect_fake_mode
from torch._logging import LazyString
from torch._ops import OpOverload
from torch._subclasses import FakeTensorMode
from torch.distributed._functional_collectives import _are_we_tracing
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._decompositions import DecompShardingStrategy
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import (
    OpInfo,
    OpSchema,
    OpSpec,
    OpStrategy,
    OutputSharding,
    OutputSpecType,
    RuntimeSchemaInfo,
    StrategyType,
    TupleStrategy,
)
from torch.distributed.tensor._ops.single_dim_strategy import (
    _expand_single_dim_strategy_to_mesh,
    _SingleDimStrategyInfo,
)
from torch.distributed.tensor._utils import (
    compute_local_shape_and_global_offset,
    compute_local_stride,
    try_find_mesh_from_args,
)
from torch.distributed.tensor.placement_types import _StridedShard, Shard
from torch.utils._pytree import tree_map


aten = torch.ops.aten

log = logging.getLogger(__name__)


def _propagate_use_strided_shard_flag(
    op_strategy: OpStrategy,
    op_schema: OpSchema,
) -> None:
    """Propagate use_strided_shard_as_shard_order from input specs to output specs.

    When inputs carry _StridedShard with an explicit flag, all output (and input)
    DTensorSpecs in the strategy that also contain _StridedShard must agree.
    Strategy functions may forget to propagate the flag; this function fixes
    them up centrally after the strategy is produced.
    """
    _use_strided: bool | None = None
    for spec in op_schema.args_spec:
        if any(isinstance(p, _StridedShard) for p in spec.placements):
            val = spec.use_strided_shard_as_shard_order
            if _use_strided is not None and _use_strided != val:
                _use_strided = True
            else:
                _use_strided = val

    if _use_strided is None:
        return

    def _fixup(spec: DTensorSpec) -> None:
        if not any(isinstance(p, _StridedShard) for p in spec.placements):
            return
        if spec.use_strided_shard_as_shard_order == _use_strided:
            return
        spec.use_strided_shard_as_shard_order = _use_strided
        if _use_strided:
            spec.shard_order = None  # pyrefly: ignore[bad-assignment]
        else:
            spec.shard_order = DTensorSpec.compute_default_shard_order(spec.placements)

    for op_spec in op_strategy.strategies:
        out = op_spec.output_specs
        if out is not None:
            if isinstance(out, DTensorSpec):
                _fixup(out)
            else:
                for s in out:
                    if s is not None:
                        _fixup(s)
        if op_spec.input_specs is not None:
            for s in op_spec.input_specs:
                _fixup(s)


def _length(obj) -> int:
    if obj is None:
        return 0
    if not isinstance(obj, Sequence):
        return 1
    return len(obj)


def _get_expected_num_tensor_outputs(op: OpOverload) -> int | None:
    """
    Get the expected number of tensor outputs for an operator based on its schema.

    Returns:
        The number of tensor outputs expected. Returns 0 for ops that don't return tensors
        (e.g., _linalg_check_errors). Returns 1 for single tensor return, and >1 for
        tuple returns where each element is a tensor. Returns None for List[Tensor]
        returns where the length is unknown at schema time.
    """
    return_types = op._schema.returns
    if len(return_types) == 0:
        return 0

    first_return = return_types[0]
    if isinstance(first_return.type, torch.TensorType):
        # Could be single tensor or tuple of tensors
        return len(return_types)
    elif isinstance(first_return.type, torch.ListType):
        # List[Tensor] - we don't know the length at schema time
        return None
    else:
        # Not a tensor return type
        return 0


def _validate_tensor_meta_count(
    op_schema: OpSchema,
    tensor_meta: TensorMeta | Sequence[TensorMeta | None] | None,
) -> None:
    """
    Validate that the tensor_meta matches the expected number of outputs for the op.

    Raises AssertionError if the count doesn't match, providing a helpful error message.
    """
    expected_outputs = _get_expected_num_tensor_outputs(op_schema.op)

    # Compute actual count:
    # - None means 0 outputs
    # - TensorMeta (single instance) means 1 output
    # - Sequence of TensorMeta means len(sequence) outputs
    # Note: TensorMeta is a NamedTuple (subclass of tuple), so we must check for it first
    if tensor_meta is None:
        actual_outputs = 0
    elif isinstance(tensor_meta, TensorMeta):
        actual_outputs = 1
    else:
        actual_outputs = len(tensor_meta)

    if expected_outputs is None:
        # List[Tensor] return type: length unknown at schema time, but
        # tensor_meta must be a list of TensorMeta.
        if not isinstance(tensor_meta, list):
            raise AssertionError(
                f"Tensor meta for {op_schema.op} should be a list[TensorMeta] "
                f"(op returns List[Tensor]), but got {type(tensor_meta).__name__}"
            )
        return

    if actual_outputs != expected_outputs:
        raise AssertionError(
            f"Tensor meta count mismatch for {op_schema.op}: "
            f"expected {expected_outputs} tensor output(s) based on op schema, "
            f"but _propagate_tensor_meta returned {actual_outputs}. "
            f"This usually indicates a bug in fake tensor propagation for this op."
        )


class LocalLRUCache(threading.local):
    def __init__(self, user_function: Callable) -> None:
        self.cache = lru_cache(None)(user_function)

    def __call__(self, *args, **kwargs) -> object:
        # Fast path: log.handlers check is very cheap (just checking if list is non-empty)
        # Only do the more expensive isEnabledFor check if handlers exist
        if log.handlers and log.isEnabledFor(logging.DEBUG):
            info_before = self.cache.cache_info()
            result = self.cache(*args, **kwargs)
            info_after = self.cache.cache_info()
            cache_hit = info_after.hits > info_before.hits
            op_schema = args[0] if args else None
            output_spec = getattr(result, "output_spec", None)
            log.debug(
                "sharding_prop python cache %s: %s -> %s",
                "HIT" if cache_hit else "MISS",
                op_schema,
                output_spec,
            )
            return result
        return self.cache(*args, **kwargs)

    def cache_info(self):
        return self.cache.cache_info()

    def cache_clear(self):
        return self.cache.cache_clear()


def _format_unbacked_hinting_log(
    op_schema: OpSchema,
    strategies: list[OpSpec],
    strategy_index: int,
    replacements: dict,
) -> str:
    """Format log message for unbacked hinting strategy selection (only called if debug logging enabled)."""
    args_spec = tuple(str(spec) for spec in op_schema.args_schema)
    strat = strategies[strategy_index]
    if strat.input_specs is None:
        placements_in = None
    else:
        placements_in = tuple(
            spec.format_shard_order_str(spec.placements, spec.shard_order)
            for spec in strat.input_specs
        )
    placements_out = tree_map(
        lambda spec: spec.format_shard_order_str(spec.placements, spec.shard_order),
        strat.output_specs,
        is_leaf=lambda x: isinstance(x, DTensorSpec),
    )
    return (
        f"Selected strategy {placements_in} -> {placements_out} "
        f"for {op_schema.op} with input {args_spec}, using unbacked hints: {replacements}"
    )


def _select_min_redistribute_cost(
    costs: list[torch.types.FloatLikeType],
    strategies: list[OpSpec],
    op_schema: OpSchema | None = None,
) -> int:
    """
    Given a list of costs and corresponding op strategies, selects the minimum cost strategy, returning the index.
    If unbacked symbols are involved, replaces them with known upper-bound values, falling back to hardcoded values.
    """
    from torch.fx.experimental.symbolic_shapes import (
        free_unbacked_symbols,
        is_concrete_float,
    )
    from torch.utils._sympy.interp import sympy_interp
    from torch.utils._sympy.numbers import int_oo
    from torch.utils._sympy.reference import PythonReferenceAnalysis

    int_fallback = 8192
    free_unbacked = list(set(chain(*[free_unbacked_symbols(cost) for cost in costs])))

    # Easy path: no unbacked shapes involved, choose min cost strategy.
    # Doing the hard path for backed could also make sense?
    if all(is_concrete_float(c) for c in costs) or not free_unbacked:
        return costs.index(min(costs))

    # Figure out heuristic hints for unbacked shapes.
    # If available, use shape upper bound. If not, fallback to some integer (inductor size-hinting style).
    shape_env = next(iter(x for x in costs if not is_concrete_float(x))).node.shape_env  # type: ignore[arg-type]
    replacements = {}
    for sym in free_unbacked:
        # TODO(laithsakka): unify with optimization_hint API
        if (hint := shape_env.var_to_hint_override.get(sym)) is not None:
            replacements[sym] = hint
        elif (upper := shape_env.bound_sympy(sym).upper) is not int_oo:
            replacements[sym] = upper
        else:
            replacements[sym] = int_fallback

    # Use replacements for redistribute cost hints
    proxy_costs = [
        float(cost)
        if is_concrete_float(cost)
        else sympy_interp(
            PythonReferenceAnalysis,
            replacements,
            cost.node.expr.xreplace(replacements),  # type: ignore[arg-type]
        )
        for cost in costs
    ]
    min_cost = min(proxy_costs)
    strategy_index = proxy_costs.index(min_cost)

    if op_schema:
        log.debug(
            "%s",
            LazyString(
                _format_unbacked_hinting_log,
                op_schema,
                strategies,
                strategy_index,
                replacements,
            ),
        )
    return strategy_index


def _select_min_cost_strategy(
    strategy: OpStrategy, op_schema: OpSchema | None = None
) -> OpSpec:
    from torch.fx.experimental.symbolic_shapes import guard_or_false

    if len(strategy.strategies) == 1:
        # short cut with only one possible OpSpec
        return strategy.strategies[0]

    op_spec_costs: list[torch.types.FloatLikeType] = []
    no_redistribute_strategy_index: int = -1
    negative_cost_index: int = -1
    zero_cost_index: int = -1
    for strategy_idx, op_spec in enumerate(strategy.strategies):
        if op_spec.redistribute_cost is None:
            raise AssertionError("must set redistribute cost each OpSpec!")
        redistribute_cost = sum(chain.from_iterable(op_spec.redistribute_cost))
        op_spec_costs.append(redistribute_cost)

        # If there are strategies with negative/zero/no redistribute cost,
        # we record those indices.
        # TODO: Currently this only applies to OpStrategy selection. Requires extra
        # logic to make it work for TupleStrategy, if needed.
        if op_schema is not None:
            if guard_or_false(redistribute_cost < 0):
                if (
                    negative_cost_index == -1
                    or redistribute_cost < op_spec_costs[negative_cost_index]
                ):
                    negative_cost_index = strategy_idx
            elif guard_or_false(redistribute_cost == 0):
                needs_redistribute = False
                for spec_idx, input_spec in enumerate(op_schema.args_spec):
                    desired_spec = (
                        op_spec.output_spec
                        if op_spec.input_specs is None
                        else op_spec.input_specs[spec_idx]
                    )
                    if input_spec.placements != desired_spec.placements:
                        needs_redistribute = True
                        break

                if not needs_redistribute:
                    no_redistribute_strategy_index = strategy_idx
                elif zero_cost_index == -1:
                    zero_cost_index = strategy_idx

    # prioritize negative/zero/no redistribute cost strategies
    if negative_cost_index != -1:
        # If there's negative cost, we select the one with the minimal cost,
        # even if this means we need to redistribute, e.g. via local chunking.
        # E.g. this can happen for ops in self.op_to_shape_and_stride_idx
        # when the inputs / outputs are sharded.
        selected_strategy_index = negative_cost_index
    elif no_redistribute_strategy_index != -1:
        selected_strategy_index = no_redistribute_strategy_index
    elif zero_cost_index != -1:
        selected_strategy_index = zero_cost_index
    else:
        # default to choosing minimal redistribute cost
        selected_strategy_index = _select_min_redistribute_cost(
            op_spec_costs, strategy.strategies, op_schema
        )

    return strategy.strategies[selected_strategy_index]


class ShardingPropagator:
    # Lock to protect FakeTensorMode context during tensor meta propagation.
    # By default this is a no-op (nullcontext) for performance. Multi-threaded
    # tests should set this to threading.Lock() to prevent race conditions
    # when multiple threads enter different FakeTensorMode contexts.
    _fake_mode_lock = nullcontext()

    def __init__(self) -> None:
        self.op_to_rules: dict[OpOverload, Callable[[OpSchema], OutputSharding]] = {}
        self.op_strategy_funcs: dict[
            OpOverload,
            Callable[[OpSchema], StrategyType],
        ] = {}
        self.op_single_dim_strategy_funcs: dict[
            OpOverload,
            _SingleDimStrategyInfo,
        ] = {}
        # op map to save static argnum to decide to reuse sharding prop cache or
        # re-run sharding prop
        self.op_to_schema_info: dict[OpOverload, RuntimeSchemaInfo] = {}
        self.op_to_schema_info_for_single_dim_strategy: dict[
            OpOverload, RuntimeSchemaInfo
        ] = {}
        self.propagate_op_sharding = LocalLRUCache(
            self.propagate_op_sharding_non_cached
        )
        self.decomp_strategy = DecompShardingStrategy(self)
        # op map to save indices of shape (and stride) args which may need to be
        # modified in sharding prop
        self.op_to_shape_and_stride_idx: dict[OpOverload, int | tuple[int, int]] = {
            # new factory ops
            aten.new_empty.default: 1,
            aten.new_full.default: 1,
            aten.new_ones.default: 1,
            aten.new_zeros.default: 1,
            aten.new_empty_strided.default: (1, 2),
            # view ops
            aten.expand.default: 1,
            aten.expand_copy.default: 1,
            aten.reshape.default: 1,
            aten.view.default: 1,
            aten.view_copy.default: 1,
            aten._unsafe_view.default: 1,
            aten.select_backward.default: 1,
            aten.slice_backward.default: 1,
            # upsample backward ops: input_size (arg 2) is the full [N,C,...] shape
            aten.upsample_nearest1d_backward.default: 2,
            aten.upsample_nearest2d_backward.default: 2,
            aten.upsample_nearest3d_backward.default: 2,
            aten._upsample_nearest_exact1d_backward.default: 2,
            aten._upsample_nearest_exact2d_backward.default: 2,
            aten._upsample_nearest_exact3d_backward.default: 2,
            aten._upsample_bilinear2d_aa_backward.default: 2,
            aten._upsample_bicubic2d_aa_backward.default: 2,
            aten._upsample_lanczos2d_aa_backward.default: 2,
            aten.upsample_bicubic2d_backward.default: 2,
            aten.upsample_bilinear2d_backward.default: 2,
            aten.upsample_linear1d_backward.default: 2,
            aten.upsample_trilinear3d_backward.default: 2,
        }
        # ops with individual scalar shape args that need local adjustment
        # maps op -> callable(input_specs, schema) -> adjusted schema
        # populated by op modules (e.g. _math_ops.py) at registration time
        self.op_to_scalar_shape_adjuster: dict[
            OpOverload,
            Callable[[list[DTensorSpec], OpSchema], OpSchema],
        ] = {}
        # squeeze ops that need dim arg rewritten to only globally-singleton dims
        self.squeeze_op_to_dims_variant: dict[OpOverload, OpOverload] = {
            aten.squeeze.default: aten.squeeze.dims,
            aten.squeeze.dim: aten.squeeze.dims,
            aten.squeeze.dims: aten.squeeze.dims,
            aten.squeeze_.default: aten.squeeze_.dims,
            aten.squeeze_.dim: aten.squeeze_.dims,
            aten.squeeze_.dims: aten.squeeze_.dims,
        }

    def register_sharding_prop_rule(
        self,
        op_overload: OpOverload,
        rule_func: Callable[[OpSchema], OutputSharding],
        schema_info: RuntimeSchemaInfo | None = None,
    ):
        """
        Register a sharding propagation rule for an operator.
        """
        self.op_to_rules[op_overload] = rule_func
        if schema_info is not None:
            self.op_to_schema_info[op_overload] = schema_info

    def register_single_dim_op_strategy(
        self,
        op_overload: OpOverload,
        strategy_info: _SingleDimStrategyInfo,
        schema_info: RuntimeSchemaInfo | None = None,
    ):
        """
        Register a strategy over a single mesh-dim, relying on infra to automatically expand to the full mesh.
        """
        self.op_single_dim_strategy_funcs[op_overload] = strategy_info
        if schema_info is not None:
            self.op_to_schema_info_for_single_dim_strategy[op_overload] = schema_info

    def register_op_strategy(
        self,
        op_overload: OpOverload,
        strategy_func: Callable[[OpSchema], StrategyType],
        schema_info: RuntimeSchemaInfo | None = None,
    ):
        """
        Register a :class:`OpStrategy` generator for an operator.

        During the sharding propagation, DTensor wants to enumerate all
        acceptable sharding specs (:class:`OpSpec`) for an operator,
        and by "acceptable" we mean that the operator can be executed on
        the ``_local_tensor`` of DTensor args/kwargs (with ``OpSpec.input_specs``)
        and the output(s) constitute valid DTensor(s) (with ``OpSpec.output_specs``).

        ``strategy_func`` is the function that enumerates such acceptable specs
        for the operator ``op_overload``. One general approach to write ``strategy_func``
        is, if the operator has simple arguments structure (e.g. mm, bmm), first enumerating
        all sharding specs for the operands, and then filtering out the ones that
        are not valid. For example, for ``mm``, the operands are two 2D tensors, and
        if both ``input`` and ``mat2`` have sharding placements ``[Shard(0)]``, then this
        is not an acceptable ``input_specs``.

        Once we have a way to enumerate all acceptable sharding specs, we can use each
        of them to construct a :class:`OpSpec`. The ``OpSpec.input_specs`` directly comes
        from the sharding spec, and the ``OpSpec.output_specs`` is therefore determined
        (e.g. ``[Shard(1)]`` @ ``[Shard(0)]`` yields ``[Partial()]``). In addition,
        :class:`OpSpec` also contains ``redistribute_cost`` which records the redistribution
        cost from each :class:`OpSpec` in the source :class:`OpStrategy.strategies` to
        the target sharding spec, for each operand.

        The ``strategy_func`` should return a :class:`OpStrategy` which contains a list of
        all the :class:`OpSpec`s generated in the above.

        The optional ``schema_info`` tells which non-DTensor args/kwargs could affect the
        cache and whether ``pytree`` is needed to flatten the nested args. ``static_argnum``
        marks the starting index of the non-DTensor args that should be hashed into the
        sharding propagation hash key, and ``static_kwargkey`` marks the keys of the
        non-DTensor kwargs that should be hashed. ``needs_pytree`` should be used when
        the input arg has :class:`list` or :class:`dict` structure.

        For example, ``aten.cat.default`` op has a ``List[Tensor]`` argument ``tensors``
        and an ``int`` argument ``dim``. Because ``dim`` affects the sharding propagation
        result, we want to pass ``RuntimeSchemaInfo(static_argnum=1)`` because the argument
        index of ``dim`` is 1. Besides, we also want to set ``needs_pytree=True`` because
        ``tensors`` needs be flattened in sharding propagation. Another example is
        ``aten.histc.default``. ``histc`` has 4 arguments (self, bins, min, max) and the
        last two would affect sharding propagation along with the :class:`DTensor` argument
        ``self``. Since the argument index of ``min`` is 2, the `schema_info` should be
        `RuntimeSchemaInfo(static_argnum=2)`.
        """
        self.op_strategy_funcs[op_overload] = strategy_func
        if schema_info is not None:
            self.op_to_schema_info[op_overload] = schema_info

    def _propagate_tensor_meta_non_cached(
        self, op_schema: OpSchema
    ) -> TensorMeta | Sequence[TensorMeta | None] | None:
        """
        Propagate the tensor metadata, it could either return a TensorMeta
        or a list/tuple of TensorMetas
        """
        if op_schema.op == aten.equal.default:
            # data dependent ops can't be used for fake propagation
            return None

        # NOTE: We must call the tracing in fake tensor mode so that it avoids
        # materializing memory.
        # NOTE: Use _fake_mode_lock to serialize access when running in
        # multi-threaded tests (lock must be set to threading.Lock()).
        # This is a nullcontext by default.
        # NOTE: disable_proxy_modes_tracing() prevents make_fx from tracing
        # into shard propagation. This op runs purely to derive output
        # metadata (global shape/stride/dtype); it is not part of the real
        # (local) computation. Without disabling the proxy tracer, the
        # gen_fake_args() factory calls and op_schema.op(...) get recorded
        # into the enclosing graph as dead code -- global-shape "shadow" nodes
        # on empty_strided placeholders that no real output consumes.
        from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing

        with ShardingPropagator._fake_mode_lock:
            fake_mode = detect_fake_mode() or FakeTensorMode()
            with fake_mode, disable_proxy_modes_tracing():
                fake_args = op_schema.gen_fake_args()
                fake_kwargs = op_schema.gen_fake_kwargs()
                fake_out = op_schema.op(*fake_args, **fake_kwargs)

        if isinstance(fake_out, torch.Tensor):
            return TensorMeta(
                shape=fake_out.shape, stride=fake_out.stride(), dtype=fake_out.dtype
            )

        elif isinstance(fake_out, (tuple, list)):
            tensor_meta_list: list[TensorMeta | None] = []
            for fake_out_item in fake_out:
                if isinstance(fake_out_item, torch.Tensor):
                    tensor_meta_list.append(
                        TensorMeta(
                            shape=fake_out_item.shape,
                            stride=fake_out_item.stride(),
                            dtype=fake_out_item.dtype,
                        )
                    )
                else:
                    tensor_meta_list.append(None)
            return (
                tuple(tensor_meta_list)
                if isinstance(fake_out, tuple)
                else tensor_meta_list
            )
        else:
            # if fake is not a tensor or tuple of tensor, return as none
            return None

    @lru_cache  # noqa: B019
    def _propagate_tensor_meta_cached(
        self, op_schema: OpSchema
    ) -> TensorMeta | Sequence[TensorMeta | None] | None:
        """
        Cached version of _propagate_tensor_meta_non_cached
        Use _propagate_tensor_meta instead to handle dynamic shapes.
        """
        return self._propagate_tensor_meta_non_cached(op_schema)

    def _propagate_tensor_meta(
        self, op_schema: OpSchema
    ) -> TensorMeta | Sequence[TensorMeta | None] | None:
        """
        Propagate the tensor metadata, it could either return a TensorMeta
        or a list/tuple of TensorMetas. Uses the cached version if not
        actively tracing. Use this method instead of _propagate_tensor_meta_non_cached
        """
        if _are_we_tracing():
            return self._propagate_tensor_meta_non_cached(op_schema)
        else:
            return self._propagate_tensor_meta_cached(op_schema)

    def _create_output_spec_with_new_tensor_meta(
        self,
        op: OpOverload,
        output_specs: OutputSpecType,
        output_tensor_meta: TensorMeta | Sequence[TensorMeta | None] | None,
    ) -> OutputSpecType:
        """
        Wrap the output_specs with the tensor metadata from the output.
        """

        if isinstance(output_specs, DTensorSpec):
            if not isinstance(output_tensor_meta, TensorMeta):
                # Either error due to ShardingPropagator or due to incorrect OutputSpec
                if not isinstance(output_tensor_meta, (tuple, list)):
                    raise ValueError(
                        "ShardingPropagator error: output does not have an associated "
                        "TensorMeta"
                    )
                raise ValueError(
                    f"For the op {op.name()}, `output_specs` has 1 output which does "
                    "not equal the "
                    f"number of op outputs: {len(output_tensor_meta)}."
                )
            return output_specs.shallow_copy_with_tensor_meta(output_tensor_meta)
        elif isinstance(output_specs, (tuple, list)):
            new_specs: list[DTensorSpec | None] = []
            if not isinstance(output_tensor_meta, (tuple, list)) or len(
                output_specs
            ) != len(output_tensor_meta):
                raise ValueError(
                    f"For the op {op.name()}, `output_specs` has {len(output_specs)} "
                    "outputs which does not equal the "
                    f"number of op outputs {_length(output_tensor_meta)}."
                )

            # pyrefly: ignore [bad-argument-type]
            for i, spec in enumerate(output_specs):
                if isinstance(spec, DTensorSpec):
                    output_tensor_meta_i = output_tensor_meta[i]
                    if not isinstance(output_tensor_meta_i, TensorMeta):
                        # Some ops (e.g. convolution_backward, native_layer_norm_backward,
                        # _fused_rms_norm_backward) have an output_mask parameter that
                        # controls which outputs are computed. When output_mask[i] is
                        # False, the output at position i is None and has no TensorMeta.
                        if output_tensor_meta_i is None:
                            new_specs.append(None)
                            continue
                        else:
                            raise ValueError(
                                f"ShardingPropagator error: output {i} of {op.name()} "
                                "does not have an associated TensorMeta"
                            )

                    new_specs.append(
                        spec.shallow_copy_with_tensor_meta(output_tensor_meta_i)
                    )
                else:
                    new_specs.append(spec)

            return tuple(new_specs)
        else:
            if output_specs is not None:
                raise AssertionError
            return output_specs

    def _wrap_with_op_strategy(self, op_schema: OpSchema) -> OpSchema:
        """
        wrap a op_schema that contains DTensorSpec to another op_schema that contains
        OpStrategy/TupleStrategy, the returned op_schema is then used for sharding
        strategy propagation on pytorch operators.
        """

        def spec_to_strategy(spec: object) -> object:
            if isinstance(spec, DTensorSpec):
                return OpStrategy([OpSpec(spec)])
            elif isinstance(spec, (list, tuple)) and len(spec) > 0:
                if all(isinstance(s, DTensorSpec) for s in spec):
                    # tensor list create tuple strategy
                    tuple_strategy = [spec_to_strategy(s) for s in spec]
                    tuple_strategy = cast(Sequence[StrategyType], tuple_strategy)
                    return TupleStrategy(
                        tuple(tuple_strategy)
                        if isinstance(spec, tuple)
                        else tuple_strategy
                    )
                elif any(isinstance(s, DTensorSpec) for s in spec):
                    # mixed list (e.g. [DTensorSpec, None, DTensorSpec]) for
                    # ops like aten.index.Tensor; keep as list so pytree
                    # flattening can extract OpStrategy items
                    return [spec_to_strategy(s) for s in spec]
                else:
                    return spec
            else:
                return spec

        args_op_strategy = [spec_to_strategy(i) for i in op_schema.args_schema]

        kwargs_op_strategy = {
            k: spec_to_strategy(v) for k, v in op_schema.kwargs_schema.items()
        }

        return OpSchema(
            op=op_schema.op,
            args_schema=tuple(args_op_strategy),
            kwargs_schema=kwargs_op_strategy,
            schema_info=op_schema.schema_info,
        )

    def propagate(self, op_info: OpInfo) -> None:
        # NB: The logic here is duplicated in _propagate_op_sharding_dispatch_slow_path.
        # Ideally, this function would be deleted, but there are a handful of
        # one off call sites here that aren't cleaned up.

        # NOTE: schema should always be populated when calling this function,
        # as it's only called after unwrap_to_op_info (create_schema=True).
        if op_info.schema is None:
            raise AssertionError(
                "op_info.schema should not be None in propagate. "
                "This function should only be called after unwrap_to_op_info."
            )

        # We cannot use an lru cache if we know that inputs will have dynamic shapes,
        # because SymInts are not hashable.
        # This is generally ok because this only happens during tracing in torch.compile,
        # and tracing does not need to be as fast as eagermode DTensor usages.
        if _are_we_tracing():
            output_sharding = self.propagate_op_sharding_non_cached(op_info.schema)
        else:
            output_sharding = cast(
                OutputSharding, self.propagate_op_sharding(op_info.schema)
            )
        op_info.output_sharding = output_sharding

    def propagate_op_sharding_non_cached(self, op_schema: OpSchema) -> OutputSharding:
        """
        Propagate the sharding for an operator given the op_schema.
        """
        # no-op in OSS, logs API usage metrics in meta-internal runs
        torch._C._log_api_usage_once(
            "torch.distributed.tensor._sharding_prop.ShardingPropagator.propogate_op_sharding_non_cached"
        )
        # special case op, we don't need to propagate for local
        # scalar. TODO: figure out a better way to handle this
        if op_schema.op is aten._local_scalar_dense.default:
            return OutputSharding(None, op_schema)

        out_tensor_meta = self._propagate_tensor_meta_non_cached(op_schema)

        single_dim_strategy_info = self.op_single_dim_strategy_funcs.get(op_schema.op)
        op_strategy_func = self.op_strategy_funcs.get(op_schema.op)
        decomp_exception = None
        if single_dim_strategy_info is not None or op_strategy_func is not None:
            # Validate that tensor_meta count matches expected outputs from op schema.
            # This catches bugs in fake tensor propagation early.
            if single_dim_strategy_info is not None:
                _validate_tensor_meta_count(op_schema, out_tensor_meta)
            """
            Given the single_dim_strategy, which is just a minimal set of valid input-output placement specifications
            for the operator over a single mesh dimension,

            And the OpSchema, which includes information about the runtime input tensor placements, and the mesh,

            Combine single_dim_strategies across mesh dims, also expanding placeholders (ShardPlaceholder) to any real
            sharding types in op_schema, and find the lowest cost redistribution of inputs to match a valid strategy
            combination.
            """
            # wrap the op_schema with op strategy for sharding strategy propagation
            strategy_schema = self._wrap_with_op_strategy(op_schema)

            if single_dim_strategy_info is not None:
                mesh = try_find_mesh_from_args(
                    op_schema.op,
                    (*op_schema.args_schema, *op_schema.kwargs_schema.values()),
                )
                if not isinstance(mesh, DeviceMesh):
                    raise AssertionError("Expected to find a valid mesh")
                # expand to generate the full set of strategy combinations, each one
                # with a redistribute cost, and then find the min strategy over those costs.
                _expanded_strategy_fn = _expand_single_dim_strategy_to_mesh(
                    mesh, strategy_schema, single_dim_strategy_info, out_tensor_meta
                )
                op_strategy = _expanded_strategy_fn(
                    op_schema.op, strategy_schema.args_meta, strategy_schema.kwargs_meta
                )
            else:
                if op_strategy_func is None:
                    raise AssertionError
                op_strategy = op_strategy_func(strategy_schema)

        else:
            # try operator decomposition path

            op_strategy = None
            if DecompShardingStrategy.has_decomp(op_schema.op):
                # Ensure schema_info is registered for proper cache key computation
                self.decomp_strategy.ensure_schema_info(op_schema.op)
                try:
                    op_strategy = self.decomp_strategy.propagate_strategy(
                        op_schema,
                        out_tensor_meta,
                    )
                except Exception as e:
                    decomp_exception = e

        if op_strategy is not None:
            if isinstance(op_strategy, OpStrategy):
                _propagate_use_strided_shard_flag(op_strategy, op_schema)
                # single Op strategy
                output_strategy = _select_min_cost_strategy(op_strategy, op_schema)

                # check if we need to redistribute the input
                needs_redistribute = False
                # check if we want to use args value from redistribute_schema
                use_val_from_redistribute_schema = False
                expected_input_specs: list[DTensorSpec] = []

                # in case where the op does not specify input_specs and output_specs
                # is a DTensorSpec, we use output_specs as the spec for each DTensor
                # input arg.
                if output_strategy.input_specs is None:
                    if not isinstance(output_strategy.output_specs, DTensorSpec):
                        raise AssertionError

                for idx, input_spec in enumerate(op_schema.args_spec):
                    desired_spec = (
                        output_strategy.output_spec
                        if output_strategy.input_specs is None
                        else output_strategy.input_specs[idx]
                    )
                    expected_input_specs.append(
                        desired_spec.shallow_copy_with_tensor_meta(
                            input_spec.tensor_meta
                        )
                    )
                    if input_spec.placements != desired_spec.placements:
                        needs_redistribute = True

                suggestion_schema = None
                if needs_redistribute:
                    suggestion_schema = OpSchema(
                        op_schema.op, tuple(expected_input_specs), {}
                    )
                    suggestion_schema._inplace_rewrap_schema_suggestion(op_schema)

                # Rewrite shape/stride args from global to local when sharded.
                # Covers view ops, new factory ops, and upsample backward ops
                # whose shape args refer to global tensor dimensions.
                if op_schema.op in self.op_to_shape_and_stride_idx:
                    if not isinstance(output_strategy.output_spec, DTensorSpec):
                        raise AssertionError
                    if any(
                        isinstance(p, Shard | _StridedShard)
                        for p in output_strategy.output_spec.placements
                    ):
                        schema = suggestion_schema or op_schema
                        if not isinstance(out_tensor_meta, TensorMeta):
                            raise AssertionError
                        suggestion_schema = self._adjust_shape_and_stride_args(
                            out_tensor_meta, schema, output_strategy.output_spec
                        )
                        needs_redistribute = True
                        use_val_from_redistribute_schema = True

                # adjust individual scalar shape args (e.g. N, C, HxW in group_norm)
                if op_schema.op in self.op_to_scalar_shape_adjuster:
                    if any(
                        isinstance(p, Shard | _StridedShard)
                        for spec in expected_input_specs
                        for p in spec.placements
                    ):
                        schema = suggestion_schema or op_schema
                        adjuster = self.op_to_scalar_shape_adjuster[op_schema.op]
                        suggestion_schema = adjuster(expected_input_specs, schema)
                        needs_redistribute = True
                        use_val_from_redistribute_schema = True

                # rewrite squeeze to use only globally-singleton dims
                if op_schema.op in self.squeeze_op_to_dims_variant:
                    schema = suggestion_schema or op_schema
                    adjusted = self._adjust_squeeze_to_global_singletons(schema)
                    if adjusted is not None:
                        suggestion_schema = adjusted
                        needs_redistribute = True
                        use_val_from_redistribute_schema = True

                # construct output spec for the op
                if op_schema.return_type_tuple_tensor_like():
                    # for ops that return multiple tensors and the output_specs is not
                    # a tuple, we use a tuple of that single output spec as the new
                    # output_specs
                    output_specs: OutputSpecType = output_strategy.output_specs
                    if isinstance(output_specs, DTensorSpec):
                        output_specs = tuple(
                            # create a new DTensorSpec with the same placement as the
                            # output_specs in output_strategy
                            DTensorSpec(
                                mesh=output_specs.mesh,
                                placements=output_specs.placements,
                                tensor_meta=output_specs.tensor_meta,
                                use_strided_shard_as_shard_order=output_specs.use_strided_shard_as_shard_order,
                            )
                            for _ in range(len(op_schema.op._schema.returns))
                        )
                elif (
                    op_schema.return_type_tensor()
                    or op_schema.return_type_list_tensor_like()
                ):
                    output_specs = output_strategy.output_specs
                else:
                    output_specs = None

                output_sharding = OutputSharding(
                    output_specs,
                    suggestion_schema,
                    needs_redistribute=needs_redistribute,
                    use_val_from_redistribute_schema=use_val_from_redistribute_schema,
                )
            elif isinstance(op_strategy, TupleStrategy):
                # tuple strategy output sharding processing
                # runtime select OpSpec for each TupleStrategy input arg
                selected_strategies: list[OpSpec] = []
                out_spec_list: list[DTensorSpec] = []
                for strategy in op_strategy.children:
                    if not isinstance(strategy, OpStrategy):
                        raise AssertionError
                    _propagate_use_strided_shard_flag(strategy, op_schema)
                    selected_strategy = _select_min_cost_strategy(strategy)
                    selected_strategies.append(selected_strategy)
                    if selected_strategy.output_specs is not None:
                        out_spec_list.append(selected_strategy.output_spec)

                needs_redistribute = False
                suggestion_args: list[object] = []
                tensor_or_list_tensor_arg_idx = 0

                for arg in op_schema.args_schema:
                    if (
                        arg
                        and isinstance(arg, (list, tuple))
                        and isinstance(arg[0], DTensorSpec)
                    ):
                        expected_input_spec_list: list[DTensorSpec] = []
                        for idx, arg_spec in enumerate(arg):
                            expected_input_spec = selected_strategies[idx].input_spec(
                                tensor_or_list_tensor_arg_idx
                            )
                            expected_input_spec = (
                                expected_input_spec.shallow_copy_with_tensor_meta(
                                    arg_spec.tensor_meta
                                )
                            )
                            if arg_spec.placements != expected_input_spec.placements:
                                needs_redistribute = True
                            expected_input_spec_list.append(expected_input_spec)
                        suggestion_args.append(
                            tuple(expected_input_spec_list)
                            if isinstance(arg, tuple)
                            else expected_input_spec_list
                        )
                        tensor_or_list_tensor_arg_idx += 1

                    elif isinstance(arg, DTensorSpec):
                        expected_input_spec = selected_strategies[0].input_spec(
                            tensor_or_list_tensor_arg_idx
                        )
                        expected_input_spec = (
                            expected_input_spec.shallow_copy_with_tensor_meta(
                                arg.tensor_meta
                            )
                        )
                        if arg.placements != expected_input_spec.placements:
                            needs_redistribute = True
                        suggestion_args.append(expected_input_spec)
                        tensor_or_list_tensor_arg_idx += 1
                    else:
                        suggestion_args.append(arg)

                suggestion_schema = None
                if needs_redistribute:
                    suggestion_schema = OpSchema(
                        op_schema.op, tuple(suggestion_args), op_schema.kwargs_schema
                    )

                output_sharding = OutputSharding(
                    tuple(out_spec_list) if out_tensor_meta is not None else None,
                    suggestion_schema,
                    needs_redistribute=needs_redistribute,
                    use_val_from_redistribute_schema=False,
                )
            else:
                raise ValueError("Unsupported op strategy type")

            # associate the output sharding with the output tensor metadata
            new_output_spec = self._create_output_spec_with_new_tensor_meta(
                op_schema.op, output_sharding.output_spec, out_tensor_meta
            )
            output_sharding.output_spec = new_output_spec
            return output_sharding
        elif op_schema.op in self.op_to_rules:
            # propagate the sharding with rule
            sharding_prop_func = self.op_to_rules[op_schema.op]

            # step 1. there's sharding propagation rule, run
            # sharding propagation to get the output sharding
            try:
                output_sharding = sharding_prop_func(op_schema)
            except NotImplementedError as e:
                raise e
            except Exception as e:
                raise RuntimeError(
                    f"Sharding propagation failed on op {op_schema}.\nError: {e}"
                ) from e

            # step 2. if can't get output_spec from sharding
            # propagation (i.e. no rules apply for input
            # placements), we return the output sharding
            # with schema suggestions, which can be used to
            # decide how to do redistribute on inputs
            if output_sharding.output_spec is None:
                if output_sharding.redistribute_schema is None:
                    raise RuntimeError(
                        f"Sharding propagation failed on op {op_schema}!"
                    )
                else:
                    # we do auto redistribute on inputs if necessary
                    # run sharding propagation again with suggested schema
                    propagation_res = sharding_prop_func(
                        output_sharding.redistribute_schema
                    )
                    # we set the output sharding with the new propagation result
                    # so that dispatching know both output_spec and redistribute_schema
                    # exist, which indicates a reshard is needed
                    output_sharding.output_spec = propagation_res.output_spec
                    output_sharding.needs_redistribute = True

            # associate the output sharding with the output tensor metadata
            new_output_spec = self._create_output_spec_with_new_tensor_meta(
                op_schema.op, output_sharding.output_spec, out_tensor_meta
            )
            output_sharding.output_spec = new_output_spec

            return output_sharding
        else:
            raise NotImplementedError(
                f"Operator {op_schema.op} does not have a sharding strategy registered."
            ) from decomp_exception

    def _adjust_shape_and_stride_args(
        self,
        out_tensor_meta: TensorMeta,
        schema: OpSchema,
        spec: DTensorSpec,
    ) -> OpSchema:
        shape_stride_idx = self.op_to_shape_and_stride_idx[schema.op]
        if isinstance(shape_stride_idx, tuple):
            shape_idx, stride_idx = shape_stride_idx
        else:
            shape_idx = shape_stride_idx
            stride_idx = None

        expected_input_schema = list(schema.args_schema)
        # rewrite args[shape_idx] from the global output shape to the local output shape
        local_shape, _ = compute_local_shape_and_global_offset(
            out_tensor_meta.shape, spec.mesh, spec.placements, skip_offset=True
        )
        expected_input_schema[shape_idx] = local_shape

        # adjust the stride arg for aten.new_empty_strided.default
        if stride_idx:
            expected_input_schema[stride_idx] = compute_local_stride(
                out_tensor_meta.stride, local_shape
            )

        return OpSchema(schema.op, tuple(expected_input_schema), schema.kwargs_schema)

    def _adjust_squeeze_to_global_singletons(self, schema: OpSchema) -> OpSchema | None:
        """
        Rewrite squeeze ops to squeeze.dims with only globally-singleton dims.
        Fixes bug where sharded dims with local size 1 get incorrectly squeezed.
        Returns None if no rewrite is needed (already squeeze.dims with correct args).
        """
        from torch.fx.experimental.symbolic_shapes import guard_or_false

        input_spec = cast(DTensorSpec, schema.args_schema[0])
        tensor_meta = input_spec.tensor_meta
        if tensor_meta is None:
            raise RuntimeError("squeeze requires tensor metadata")
        global_shape = tensor_meta.shape
        ndim = len(global_shape)

        def normalize(d: int) -> int:
            return d if d >= 0 else d + ndim

        def is_singleton(d: int) -> bool:
            nd = normalize(d)
            return 0 <= nd < ndim and guard_or_false(global_shape[nd] == 1)

        # guard_or_false: conservatively keep dims when size is symbolic/unknown
        if schema.op in (aten.squeeze.default, aten.squeeze_.default):
            target_dims = tuple(
                i for i, s in enumerate(global_shape) if guard_or_false(s == 1)
            )
        elif schema.op in (aten.squeeze.dim, aten.squeeze_.dim):
            dim = normalize(schema.args_schema[1])  # type: ignore[arg-type]
            target_dims = (dim,) if is_singleton(dim) else ()
        else:
            dims = cast(Sequence[int], schema.args_schema[1])
            target_dims = tuple(  # type: ignore[union-attr]
                normalize(d) for d in dims if is_singleton(d)
            )

        dims_variant = self.squeeze_op_to_dims_variant[schema.op]
        # Skip rewrite if already targeting the right op with the same dims
        if schema.op == dims_variant and len(schema.args_schema) > 1:
            existing_dims = schema.args_schema[1]
            if existing_dims == target_dims:
                return None
        return OpSchema(dims_variant, (input_spec, target_dims), {})
<|endoftext|># mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
import contextlib
from typing import cast

import torch
import torch._prims_common as utils
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
from torch import Tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._ops._embedding_ops import _MaskPartial
from torch.distributed.tensor._ops._math_ops import (
    _skip_dim,
    Reduction,
    replicate_reduction_dims,
)
from torch.distributed.tensor._ops.utils import normalize_dim
from torch.distributed.tensor.placement_types import Placement


aten = torch.ops.aten


__all__ = ["loss_parallel"]


@contextlib.contextmanager
def loss_parallel():
    """
    A context manager that enables loss parallelism, where efficient parallelized loss computation
    can be performed when the input is sharded on the class dimension. Currently only the cross-entropy
    loss is supported.

    Within this context manager, one can use :func:`~torch.nn.functional.cross_entropy` or
    :class:`~torch.nn.CrossEntropyLoss` as usual, with the following assumptions on the input parameters.
    The corresponding ``backward()`` call, if any, also needs to happen under this context manager.

    Both one-dimensional and multi-dimensional ``DeviceMesh`` are supported. On a
    multi-dimensional mesh, exactly one mesh dim must shard the class dimension —
    that dim is treated as the "TP" dim and can be in any position (not necessarily
    last). The remaining mesh dims may be ``Shard`` (e.g. batch-sharded for DP/CP)
    or ``Replicate``.

    Args:
        input (:class:`DTensor`):
            Input logits. Assumed to be sharded on the class dimension on exactly
            one mesh dim.
        target (Union[:class:`torch.Tensor`, :class:`DTensor`]):
            Must be ground truth class indices (class probabilities currently not
            supported). Its placements are derived from ``input``'s placements by
            (1) replacing the TP dim's ``Shard(class_dim)`` with ``Replicate`` and
            (2) shifting any other ``Shard(d)`` with ``d > class_dim`` down by one
            (since the class tensor dim is removed from the output)::

                input tensor shape       input placements             target shape      target placements
                ----------------------   --------------------------   ---------------   ------------------------
                (batch, class)           (Shard(0), Shard(1))         (batch,)          (Shard(0), Replicate())
                (batch, class)           (Replicate(), Shard(1))      (batch,)          (Replicate(), Replicate())
                (batch, class, seq)      (Shard(2), Shard(1))         (batch, seq)      (Shard(1), Replicate())
                (batch, class, seq)      (Shard(0), Shard(1))         (batch, seq)      (Shard(0), Replicate())

            A plain :class:`torch.Tensor` is only accepted when the derived target
            placements are all ``Replicate`` (e.g. on a one-dimensional mesh);
            otherwise a :class:`DTensor` must be passed explicitly so that the
            intended sharding is unambiguous.
        weight (Union[:class:`torch.Tensor`, :class:`DTensor`], optional):
            If given, assumed to be replicated across the ``DeviceMesh``.
        label_smoothing:
            Currently not supported.

    Returns:
        A :class:`DTensor` whose placements depend on ``reduction``:

        - ``reduction="none"`` — per-sample loss; inherits the ``target``
          placements (``Shard`` on batch-parallel dims, ``Replicate`` on TP).
        - ``reduction="sum"`` — scalar; ``Replicate`` on the TP dim (the custom
          all-reduce makes each rank's local value globally correct over TP),
          ``Partial("sum")`` on every other ``Shard`` mesh dim so that the
          cross-rank reduction happens lazily on materialization/redistribution,
          and ``Replicate`` on ``Replicate`` mesh dims.
        - ``reduction="mean"`` — only supported on a one-dimensional mesh; returns
          a fully replicated :class:`DTensor`.

        On a one-dimensional mesh all three cases simplify to a fully replicated
        :class:`DTensor` (the original behavior).

    .. note::
        ``reduction="mean"`` is only supported on a one-dimensional ``DeviceMesh``.
        On a multi-dimensional mesh the per-rank division by ``total_weight`` uses a
        local count, so aggregating across non-TP dims does not yield the correct
        global mean; use ``reduction="sum"`` or ``"none"`` instead and divide by the
        global count yourself if needed.

    Example:
        A sharded DTensor is manually created here to showcase the usage.
        In practice, it is usually the output of a TP module.

        >>> # xdoctest: +SKIP("distributed")
        >>> from torch.distributed.tensor.parallel import loss_parallel
        >>> from torch.distributed.device_mesh import init_device_mesh
        >>> ...
        >>> device_mesh = init_device_mesh("cuda", (8,))
        >>> input = torch.randn(4, 16, device="cuda", requires_grad=True)
        >>> dist_input = distribute_tensor(input, device_mesh, placements=[Shard(1)])
        >>> target = torch.randint(16, (4,), device="cuda")
        >>> with loss_parallel():
        >>>     loss = F.cross_entropy(dist_input, target, reduction="mean")
        >>>     loss.backward()
        >>> ...
    """
    _enable_custom_loss_ops()

    yield

    _disable_custom_loss_ops()


# Return the mesh_dim whose placement is Shard(dim). Exactly one such mesh dim
# (the "TP" dim) is required; other mesh dims must be Shard on a different
# tensor dim or Replicate. Partial on a non-TP dim is rejected so that
# reduction="none" cannot silently accept garbage (replicate_reduction_dims
# would otherwise rewrite Partial -> Replicate in target_placements).
def _find_all_reduce_mesh_dim(placements: tuple[Placement, ...], dim: int) -> int:
    shard_mesh_dims = [i for i, p in enumerate(placements) if p.is_shard(dim)]
    if len(shard_mesh_dims) != 1:
        raise ValueError(
            f"loss_parallel() requires exactly one mesh dim to shard tensor "
            f"dimension {dim}, but got placements {placements} with "
            f"{len(shard_mesh_dims)} such mesh dim(s)."
        )
    mesh_dim = shard_mesh_dims[0]
    for i, p in enumerate(placements):
        if i == mesh_dim:
            continue
        if not (p.is_shard() or p.is_replicate()):
            raise ValueError(
                f"loss_parallel() expects non-TP mesh dims to be "
                f"Shard or Replicate, got {p} at dim {i}."
            )
    return mesh_dim


def _cast_to_dtensor(
    tensor, placements: tuple[Placement, ...], mesh: DeviceMesh
) -> DTensor:
    if isinstance(tensor, DTensor):
        if tensor.placements == placements:
            return tensor
        else:
            raise RuntimeError(f"Expected {placements} but got {tensor.placements}.")
    elif isinstance(tensor, torch.Tensor):
        if any(p.is_shard() for p in placements):
            # A plain torch.Tensor with Shard placements is ambiguous: it could
            # be a full global tensor the user expects us to shard, or an
            # already-local slice. Require a DTensor so the intent is explicit.
            raise ValueError(
                f"loss_parallel() requires a DTensor (not a plain torch.Tensor) "
                f"when the derived placements {placements} contain Shard — this "
                f"happens on a multi-dimensional mesh with a batch-sharded non-TP "
                f"dim. Wrap the tensor with DTensor.from_local or distribute_tensor "
                f"to make the sharding explicit."
            )
        return DTensor.from_local(
            tensor, device_mesh=mesh, placements=placements, run_check=False
        )
    else:
        raise TypeError(f"Unsupported type {type(tensor)}")


def _propagate_tensor_meta(
    op_call: torch._ops.OpOverload,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> TensorMeta:
    op_info = DTensor._op_dispatcher.unwrap_to_op_info(op_call, args, kwargs)
    if op_info.schema is None:
        raise AssertionError(
            "op_info.schema should not be None after unwrap_to_op_info"
        )
    tensor_meta = DTensor._op_dispatcher.sharding_propagator._propagate_tensor_meta(
        op_info.schema
    )
    if isinstance(tensor_meta, TensorMeta):
        return tensor_meta
    elif isinstance(tensor_meta, tuple):
        # pyrefly: ignore [bad-return]
        return tensor_meta[0]
    else:
        raise RuntimeError(f"Unexpected tensor meta type: {type(tensor_meta)}.")


# NOTE: The implementation follows torch._decomp.decomposition._log_softmax,
# with all_reduce manually inserted to perform distributed computation.
def _log_softmax(x, dim, half_to_float, mesh, mesh_dim):
    if half_to_float:
        if x.dtype != torch.half:
            raise AssertionError
    computation_dtype, result_dtype = utils.elementwise_dtypes(
        x, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )
    x = x.to(dtype=computation_dtype, memory_format=torch.contiguous_format)
    if x.numel() == 0:
        shifted = x
    else:
        x_max = torch.amax(x, dim, keepdim=True)
        x_max = funcol.all_reduce(
            x_max, reduceOp=c10d.ReduceOp.MAX.name, group=(mesh, mesh_dim)
        )
        shifted = x - x_max
    shifted_sumexp = torch.sum(torch.exp(shifted), dim, keepdim=True)
    shifted_sumexp = funcol.all_reduce(
        shifted_sumexp, reduceOp=c10d.ReduceOp.SUM.name, group=(mesh, mesh_dim)
    )
    shifted_logsumexp = torch.log(shifted_sumexp)
    result = shifted - shifted_logsumexp
    if not half_to_float:
        result = result.to(result_dtype)
    return result


def _log_softmax_handler(
    op_call: torch._ops.OpOverload,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    x = cast(DTensor, args[0])
    dim = cast(int, args[1])
    half_to_float = cast(bool, args[2])

    spec = x._spec
    dim = normalize_dim(dim, x.dim())
    mesh_dim = _find_all_reduce_mesh_dim(spec.placements, dim)

    output_tensor_meta = _propagate_tensor_meta(op_call, args, kwargs)

    res = _log_softmax(x._local_tensor, dim, half_to_float, spec.mesh, mesh_dim)

    res_spec = DTensorSpec(
        spec.mesh,
        spec.placements,
        tensor_meta=output_tensor_meta,
    )

    # pyrefly: ignore [bad-argument-type]
    return DTensor(
        # pyrefly: ignore [bad-argument-count]
        res,
        res_spec,
        # pyrefly: ignore [unexpected-keyword]
        requires_grad=res.requires_grad,
    )


# NOTE: As explained below at _nll_loss_and_log_softmax_backward, the
# _log_softmax_backward_handler does not actually do any computation.
def _log_softmax_backward_handler(
    op_call: torch._ops.OpOverload,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    grad_output = cast(DTensor, args[0])
    input_dtype = cast(torch.dtype, args[3])
    return grad_output.to(input_dtype)


# NOTE: The implementation follows torch._decomp.decomposition._nll_loss_forward,
# with customized communication inserted to perform distributed computation.
def _nll_loss_forward(
    x: Tensor,
    target: Tensor,
    weight: Tensor | None,
    local_weight: Tensor | None,
    reduction: int,
    ignore_index: int,
    input_shape: torch.Size,
    channel_dim: int,
    mesh: DeviceMesh,
    mesh_dim: int,
) -> tuple[Tensor, Tensor]:
    n_dims = x.dim()
    channel_dim = 1
    if n_dims < 2:
        channel_dim = 0

    def _weight_view(weight: Tensor) -> Tensor:
        if n_dims > 1:
            shape = [
                1,
            ] * n_dims
            shape[channel_dim] = weight.shape[0]
            w = weight.view(shape)
        else:
            w = weight
        return w

    if weight is not None:
        w = _weight_view(weight)
        if local_weight is None:
            raise AssertionError
        local_w = _weight_view(local_weight)
        x = x * local_w
    safe_target = torch.where(target != ignore_index, target, 0)
    safe_target_ = safe_target.unsqueeze(channel_dim)

    # The following code block is a distributed version of
    # result = -torch.gather(self, channel_dim, safe_target_).squeeze(channel_dim)
    partial_placement = _MaskPartial(offset_shape=input_shape, offset_dim=channel_dim)
    safe_target_partial_ = partial_placement._partition_value(
        safe_target_, mesh, mesh_dim
    )
    result_partial = torch.gather(x, channel_dim, safe_target_partial_)
    # an all_reduce happens here
    result_reduced = partial_placement._reduce_value(result_partial, mesh, mesh_dim)
    result = -result_reduced.squeeze(channel_dim)

    result = torch.where(target != ignore_index, result, 0)

    if reduction == Reduction.NONE.value and n_dims > 1:
        total_weight = x.new_full((), 0.0)
        return result, total_weight

    if weight is not None:
        new_shape = list(x.shape)
        new_shape[channel_dim] = -1
        # pyrefly: ignore [unbound-name]
        w = w.expand(new_shape)
        wsum = torch.gather(w, channel_dim, safe_target_).squeeze(channel_dim)
        wsum = torch.where(target != ignore_index, wsum, 0)
        total_weight = wsum.sum()
    else:
        total_weight = (target != ignore_index).sum().to(x)

    # On an N-D mesh, result.sum() is a per-rank sum over the local batch;
    # the caller emits Partial("sum") on non-TP Shard dims so the cross-rank
    # reduction happens when the output is later materialized/redistributed.
    # reduction="mean" is rejected in the handler for N-D mesh because
    # total_weight is a per-rank local count.
    if reduction == Reduction.SUM.value:
        result = result.sum()
    elif reduction == Reduction.MEAN.value:
        result = result.sum() / total_weight

    return result, total_weight


def _nll_loss_forward_handler(
    op_call: torch._ops.OpOverload,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    x = cast(DTensor, args[0])
    target = args[1]
    weight = args[2]
    reduction = cast(int, args[3])
    ignore_index = cast(int, args[4])

    channel_dim = 1 if x.dim() >= 2 else 0
    spec = x._spec
    mesh_dim = _find_all_reduce_mesh_dim(spec.placements, channel_dim)

    # Check user input: if target and weight are not DTensors, convert them to DTensors;
    # if they are DTensors, check that they have the desired placements.
    target_placements = _skip_dim(
        replicate_reduction_dims(spec.placements, [channel_dim]), channel_dim
    )
    all_replicate_placements = (Replicate(),) * spec.mesh.ndim
    target = _cast_to_dtensor(target, target_placements, spec.mesh)
    local_weight = None
    if weight is not None:
        weight = _cast_to_dtensor(weight, all_replicate_placements, spec.mesh)
        # For local computation, both (replicated) weight and (sharded) local_weight
        # are needed in _nll_loss_forward(). local_weight is generated here using
        # DTensor API, without incurring any communication.
        sharded_placements = [
            Shard(0) if i == mesh_dim else Replicate() for i in range(spec.mesh.ndim)
        ]
        local_weight = weight.redistribute(spec.mesh, sharded_placements)._local_tensor
        if local_weight.shape[0] != x._local_tensor.shape[channel_dim]:
            raise AssertionError

    if reduction == Reduction.NONE.value:
        output_placements = target_placements
    else:
        # reduction="mean" is not yet correct on multi-dimensional mesh: the
        # local division by total_weight uses a per-rank count, so summing
        # across non-TP dims does not yield the global mean. See the NOTE in
        # _nll_loss_forward.
        if reduction == Reduction.MEAN.value and spec.mesh.ndim > 1:
            raise NotImplementedError(
                "loss_parallel() with reduction='mean' is only supported on "
                "one-dimensional DeviceMesh; got mesh with ndim="
                f"{spec.mesh.ndim}. Use reduction='sum' or 'none' instead."
            )
        # On the shard dim (TP), the custom all-reduce produces a globally
        # correct result -> Replicate. On other dims (e.g. DP/CP), each rank
        # only reduced its local batch -> Partial(sum).
        out_placements_list: list[Placement] = []
        for i, p in enumerate(spec.placements):
            if i == mesh_dim:
                out_placements_list.append(Replicate())
            elif p.is_shard():
                out_placements_list.append(Partial())
            else:
                out_placements_list.append(p)
        output_placements = tuple(out_placements_list)

    # tensor inputs to _propagate_tensor_meta need to be DTensors
    # pyrefly: ignore [bad-assignment]
    args = list(args)
    # pyrefly: ignore [unsupported-operation]
    args[1], args[2] = target, weight
    output_tensor_meta = _propagate_tensor_meta(op_call, tuple(args), kwargs)

    result, total_weight = _nll_loss_forward(
        x._local_tensor,
        target._local_tensor,
        weight._local_tensor if weight is not None else None,
        local_weight,
        reduction,
        ignore_index,
        x.shape,
        channel_dim,
        spec.mesh,
        mesh_dim,
    )
    out_spec = DTensorSpec(spec.mesh, output_placements, tensor_meta=output_tensor_meta)

    return (
        # pyrefly: ignore [bad-argument-type]
        DTensor(
            # pyrefly: ignore [bad-argument-count]
            result,
            out_spec,
            # pyrefly: ignore [unexpected-keyword]
            requires_grad=result.requires_grad,
        ),
        total_weight,
    )


# NOTE: The backward computation of cross_entropy goes through two steps:
# backward for nll_loss and then backward for log_softmax. In loss parallel,
# the two steps are fused into the following function (called by _nll_loss_backward_handler)
# to avoid communication when target contains class indices not class probabilities.
# Also note that the _log_softmax_backward_handler does not perform computation.
# The implementation resembles _nll_loss_backward and _log_softmax_backward_data
# from torch._decomp.decomposition.
def _nll_loss_and_log_softmax_backward(
    grad_output: Tensor,
    x: Tensor,
    target: Tensor,
    weight: Tensor | None,
    reduction: int,
    ignore_index: int,
    total_weight: Tensor,
    input_shape: torch.Size,
    channel_dim: int,
    mesh: DeviceMesh,
    mesh_dim: int,
) -> Tensor:
    channel_dim = 0 if x.dim() < 2 else 1
    if reduction == Reduction.MEAN.value:
        grad_output = grad_output / total_weight

    target = target.unsqueeze(channel_dim)
    safe_target = torch.where(target != ignore_index, target, 0)
    grad_input = torch.zeros_like(x)

    # The following code block is a distributed version of
    # grad_input = torch.scatter(grad_input, channel_dim, safe_target, -1.0)
    partial_placement = _MaskPartial(offset_shape=input_shape, offset_dim=channel_dim)
    safe_target = safe_target.squeeze(channel_dim).flatten()
    masked_safe_target = partial_placement._partition_value(safe_target, mesh, mesh_dim)
    # only update grad_input to -1 if not masked
    if partial_placement.mask_buffer.data is None:
        raise AssertionError
    grad_update = partial_placement.mask_buffer.data.to(grad_input.dtype) - 1.0
    arange_1d = torch.arange(
        masked_safe_target.shape[0], device=masked_safe_target.device
    )
    # The first two cases with x.dim() <= 2 are for aten.nll_loss_backward.default;
    # the last case is for aten.nll_loss2d_backward.default.
    if x.dim() == 1:
        grad_input[masked_safe_target] = grad_update
    elif x.dim() == 2:
        grad_input[arange_1d, masked_safe_target] = grad_update
    else:
        grad_input_t = grad_input.transpose(channel_dim, -1)
        intermidate_shape = grad_input_t.shape
        grad_input_2d = grad_input_t.reshape(-1, x.shape[channel_dim])
        grad_input_2d[arange_1d, masked_safe_target] = grad_update
        grad_input = grad_input_2d.view(intermidate_shape).transpose(channel_dim, -1)

    if grad_input.dim() > grad_output.dim() > 0:
        grad_output = grad_output.unsqueeze(channel_dim)

    if weight is not None:
        new_shape = [1 for _ in range(x.dim())]
        new_shape[channel_dim] = weight.shape[0]
        weight = weight.reshape(new_shape)
        # In order for fused computation to work, the following line is rewritten.
        # grad_output = grad_output * weight
        new_shape = list(x.shape)
        new_shape[channel_dim] = -1
        w = weight.expand(new_shape)
        w_target = torch.gather(w, channel_dim, target)
        grad_output = grad_output * w_target

    grad_output = torch.where(target != ignore_index, grad_output, 0)

    # NOTE: Instead of directly returning the grad_input as grad_output for log_softmax,
    # here we perform backward computation for log_softmax altogether to avoid the
    # otherwise extra all_gather communication.
    # return grad_input * grad_output
    return (grad_input + torch.exp(x)) * grad_output


def _nll_loss_backward_handler(
    op_call: torch._ops.OpOverload,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    grad_output = cast(DTensor, args[0])
    x = cast(DTensor, args[1])
    target = args[2]
    weight = args[3]
    reduction = cast(int, args[4])
    ignore_index = cast(int, args[5])
    total_weight = cast(Tensor, args[6])

    channel_dim = 1 if x.dim() >= 2 else 0
    spec = x._spec
    mesh_dim = _find_all_reduce_mesh_dim(spec.placements, channel_dim)

    # if target and weight are not DTensors, convert them to DTensors
    target_placements = _skip_dim(
        replicate_reduction_dims(spec.placements, [channel_dim]), channel_dim
    )
    all_replicate_placements = (Replicate(),) * spec.mesh.ndim
    target = _cast_to_dtensor(target, target_placements, spec.mesh)
    if weight is not None:
        weight = _cast_to_dtensor(weight, all_replicate_placements, spec.mesh)

    # For reduction="none", autograd may deliver grad_output with different
    # placements than the forward output (e.g. Replicate on a batch-sharded dim),
    # causing a shape mismatch with the local batch. Redistribute it to match
    # target_placements — which is what the forward output uses for reduction="none".
    # For reduction="sum"/"mean" the forward output is a 0-D scalar, so a placement
    # mismatch (e.g. Replicate vs. Partial("sum")) cannot produce a shape mismatch
    # against x._local_tensor and no redistribute is needed here.
    if reduction == Reduction.NONE.value:
        grad_output = grad_output.redistribute(spec.mesh, target_placements)

    # tensor inputs to _propagate_tensor_meta need to be DTensors
    # pyrefly: ignore [bad-assignment]
    args = list(args)
    # pyrefly: ignore [unsupported-operation]
    args[0] = grad_output
    # pyrefly: ignore [unsupported-operation]
    args[2], args[3] = target, weight
    # pyrefly: ignore [unsupported-operation]
    args[6] = _cast_to_dtensor(total_weight, all_replicate_placements, spec.mesh)
    output_tensor_meta = _propagate_tensor_meta(op_call, tuple(args), kwargs)

    result = _nll_loss_and_log_softmax_backward(
        grad_output._local_tensor,
        x._local_tensor,
        target._local_tensor,
        weight._local_tensor if weight is not None else None,
        reduction,
        ignore_index,
        total_weight,
        x.shape,
        channel_dim,
        spec.mesh,
        mesh_dim,
    )
    # the output sharding is the same as input sharding: Shard(channel_dim) on mesh_dim
    out_spec = DTensorSpec(
        spec.mesh,
        spec.placements,
        tensor_meta=output_tensor_meta,
    )

    # pyrefly: ignore [bad-argument-type]
    return DTensor(
        # pyrefly: ignore [bad-argument-count]
        result,
        out_spec,
        # pyrefly: ignore [unexpected-keyword]
        requires_grad=result.requires_grad,
    )


customized_loss_ops = {
    aten._log_softmax.default: _log_softmax_handler,
    aten._log_softmax_backward_data.default: _log_softmax_backward_handler,
    aten.nll_loss_forward.default: _nll_loss_forward_handler,
    aten.nll_loss2d_forward.default: _nll_loss_forward_handler,
    aten.nll_loss_backward.default: _nll_loss_backward_handler,
    aten.nll_loss2d_backward.default: _nll_loss_backward_handler,
}


def _enable_custom_loss_ops():
    DTensor._op_dispatcher._custom_op_handlers.update(customized_loss_ops)


def _disable_custom_loss_ops():
    for custom_op in customized_loss_ops:
        DTensor._op_dispatcher._custom_op_handlers.pop(custom_op)
<|endoftext|># mypy: allow-untyped-defs
import copy
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.distributed._shard.sharding_spec as shard_spec
import torch.distributed.distributed_c10d as c10d
from torch.distributed._shard.sharded_tensor import (
    Shard,
    ShardedTensor,
    ShardedTensorMetadata,
    TensorProperties,
)
from torch.distributed._shard.sharding_spec import ShardMetadata
from torch.distributed._shard.sharding_spec.chunk_sharding_spec import ChunkShardingSpec
from torch.distributed.fsdp._common_utils import _set_fsdp_flattened
from torch.distributed.fsdp._fsdp_extensions import FSDPExtensions
from torch.distributed.fsdp._shard_utils import _create_chunk_sharded_tensor
from torch.distributed.remote_device import _remote_device
from torch.distributed.tensor import DeviceMesh, DTensor, Replicate, Shard as DShard
from torch.distributed.tensor.parallel._data_parallel_utils import (
    _flatten_tensor,
    _unflatten_tensor,
)


__all__ = ["DTensorExtensions"]


def _get_box(tensor: DTensor) -> tuple[torch.Size, torch.Size]:
    device_mesh = tensor.device_mesh
    if device_mesh.ndim != 1:
        raise AssertionError("Only 1D DeviceMeshes currently handled")

    placement = tensor.placements[0]
    offsets = [0] * len(tensor.size())
    num_chunks = device_mesh.size(mesh_dim=0)

    # NOTE: is_shard() does not match _StridedShard; see _is_shard_like().
    if tensor.placements[0].is_shard():
        shard_dim = cast(DShard, placement).dim
        chunk_size = tensor.size(shard_dim) // num_chunks
        offsets[shard_dim] = chunk_size

    return (torch.Size(offsets), tensor._local_tensor.size())


def _get_box_for(tensor: DTensor, idx: int) -> tuple[torch.Size, torch.Size]:
    offsets, size = _get_box(tensor)
    return (torch.Size([val * idx for val in offsets]), size)


def _get_local_box(tensor: DTensor) -> tuple[torch.Size, torch.Size]:
    device_mesh = tensor.device_mesh
    coord = device_mesh.get_coordinate()
    if coord is None:
        raise AssertionError
    return _get_box_for(tensor, coord[0])


def _create_shard_md_from_dt(dt: DTensor, current_rank: int) -> ShardMetadata:
    mesh = dt.device_mesh
    if mesh.ndim != 1:
        raise AssertionError("Only 1D DeviceMeshes currently handled")

    offsets, sizes = _get_local_box(dt)
    return ShardMetadata(
        shard_offsets=list(offsets),
        shard_sizes=list(sizes),
        placement=f"rank:{current_rank}/{dt._local_tensor.device}",
    )


def _create_sharded_tensor_md_from_dt(
    dt: DTensor, dt_pg: c10d.ProcessGroup
) -> ShardedTensorMetadata:
    # This is where it gets tricky, we have to produce a ShardedTensor that has full coverage
    # and yet has only one valid shard for the current rank.

    shards_md = []
    my_rank = dist.get_rank(dt_pg)
    scapegoat_rank = 0 if my_rank > 0 else 1

    # NOTE: is_shard() does not match _StridedShard; see _is_shard_like().
    if dt.placements[0].is_shard():
        shard_count = dt_pg.size()
    else:
        shard_count = 1

    for i in range(shard_count):
        offsets, sizes = _get_box_for(dt, i)
        shards_md.append(
            ShardMetadata(
                shard_offsets=list(offsets),
                shard_sizes=list(sizes),
                placement=(
                    f"rank:{scapegoat_rank if i > 0 else my_rank}/{dt._local_tensor.device}"
                ),
            )
        )

    return ShardedTensorMetadata(
        shards_metadata=shards_md,
        size=dt.size(),
        tensor_properties=TensorProperties(
            dtype=dt.dtype,
            layout=dt.layout,
            requires_grad=dt.requires_grad,
            # ignore memory_format and pin_memory as those are not supported by DT
        ),
    )


def _get_dt_pg(dt: DTensor) -> c10d.ProcessGroup:
    mesh = dt.device_mesh
    if mesh.ndim != 1:
        raise AssertionError("Only 1D DeviceMeshes currently handled")
    return mesh.get_group()


def _rewrite_spec_if_needed(
    spec: shard_spec.ShardingSpec, tensor: torch.Tensor, rank: int
) -> shard_spec.ShardingSpec:
    """
    Rewrite ``spec`` to match the device of ``tensor``.

    FSDP.sharded_optim_state_dict sneakily ships optimizer state to CPU so if the original ShardingSpec
    produces CUDA metadata, ST construction bombs.
    """
    if not isinstance(spec, ChunkShardingSpec):
        return spec

    # let's see if we need
    rewrite = False
    for p in spec.placements:
        p = cast(_remote_device, p)
        if p.rank() == rank and p.device() != tensor.device:
            rewrite = True
            break
    if rewrite:
        spec = copy.deepcopy(spec)
        # pyrefly: ignore [missing-attribute]
        for i, placement in enumerate(spec.placements):
            placement = cast(_remote_device, placement)
            if placement.rank() == rank and placement.device() != tensor.device:
                # pyrefly: ignore [missing-attribute]
                spec.placements[i] = _remote_device(f"rank:{rank}/{tensor.device}")

    return spec


def _chunk_tensor(
    tensor: torch.Tensor,
    rank: int,
    world_size: int,
    num_devices_per_node: int,
    pg: dist.ProcessGroup,
) -> torch.Tensor:
    if type(tensor) is ShardedTensor:
        if len(tensor.local_shards()) != 1:
            raise AssertionError

        inner_param = tensor.local_tensor()
        inner_st = _create_chunk_sharded_tensor(
            inner_param,
            rank,
            world_size,
            num_devices_per_node,
            pg,
        )

        outer_local_shard = tensor.local_shards()[0]
        shards: list[Shard] = [
            Shard(inner_st, copy.deepcopy(outer_local_shard.metadata))
        ]
        st_meta = copy.deepcopy(tensor.metadata())
        st_meta.tensor_properties.requires_grad = False

        st_outer = ShardedTensor._init_from_local_shards_and_global_metadata(
            shards,
            sharded_tensor_metadata=st_meta,
            process_group=tensor._process_group,
            init_rrefs=False,
        )
        return st_outer
    elif type(tensor) is DTensor:
        device_mesh = tensor.device_mesh
        if device_mesh.ndim != 1:
            raise AssertionError("Only 1D DeviceMeshes currently handled")

        inner_param = tensor._local_tensor

        inner_st = _create_chunk_sharded_tensor(
            inner_param,
            rank,
            world_size,
            torch.accelerator.device_count(),
            pg,
        )

        dt_pg = _get_dt_pg(tensor)
        # We do this differently here, we create a ST with no local shards then patch it
        shards = [
            Shard(inner_st, _create_shard_md_from_dt(tensor, dist.get_rank(dt_pg)))
        ]

        st_meta = _create_sharded_tensor_md_from_dt(tensor, dt_pg)
        st_meta.tensor_properties.requires_grad = False

        st_outer = ShardedTensor._init_from_local_shards_and_global_metadata(
            shards,
            sharded_tensor_metadata=st_meta,
            process_group=dt_pg,
            init_rrefs=False,
        )

        return st_outer
    else:
        return _create_chunk_sharded_tensor(
            tensor,
            rank,
            world_size,
            num_devices_per_node,
            pg,
        )


def _chunk_dtensor(
    tensor: torch.Tensor,
    rank: int,
    device_mesh: DeviceMesh,
) -> DTensor:
    """
    Shard a tensor to chunks along the first dimension.

    The local rank will gets its corresponding chunk as the local tensor to create a DTensor.
    """
    root_mesh = device_mesh._get_root_mesh() if device_mesh is not None else None
    if root_mesh is None:
        raise RuntimeError("No parent device_mesh is found for FSDP device_mesh.")
    if root_mesh.ndim < 2:
        raise RuntimeError(
            f"Found parent device_mesh of ndim={root_mesh.ndim},",
            "but meshes must be at least 2D.",
        )

    # We need to explicitly call .detach() to return a new tensor detached from the current graph.
    tensor = tensor.detach().clone()

    # When a layer is not involved in TP, then the tensor will not be a DTensor.
    # e.g. When a layer is not specified in the parallelize_plan, TP will have no effect on the layer.
    # e.g. When you do PairwiseParallel on a 3 layer model, TP will have no effect on the third layer.
    if isinstance(tensor, torch.Tensor) and not isinstance(tensor, DTensor):
        # For tensors, it is replicated across tp dimension and sharded across FSDP dimension.
        # TP is the inner dimension and FSDP is the outer dimension.
        # Therefore, shard placements for tensor is (Shard(0), Replicate()).
        replicate_placements = [Replicate() for _ in range(root_mesh.ndim)]
        shard_placements = [Replicate() for _ in range(root_mesh.ndim)]
        shard_placements[0] = DShard(0)  # type: ignore[call-overload]

        return DTensor.from_local(
            tensor, root_mesh, replicate_placements, run_check=False
        ).redistribute(
            device_mesh=root_mesh,
            placements=shard_placements,
        )

    else:
        tp_placements = tensor.placements
        tp_placement = tp_placements[0]

        tensor = tensor.to_local()

        # For DTensors, it is sharded across tp dimension first and then sharded across FSDP dimension.
        # TP is the inner dimension and FSDP is the outer dimension.
        # Therefore, shard placements for tensor is (Shard(0), tp_placement).
        # For higher dimensional meshes, it is replicated across other dimensions. For example, with
        # HSDP the shard placements for tensor is (Replicate, Shard(0), tp_placement).
        replicate_placements = [Replicate() for _ in range(root_mesh.ndim)]
        replicate_placements[-1] = tp_placement  # type: ignore[call-overload]
        shard_placements = [Replicate() for i in range(root_mesh.ndim)]  # type: ignore[misc]
        shard_placements[-2] = DShard(0)  # type: ignore[call-overload]
        shard_placements[-1] = tp_placement  # type: ignore[call-overload]

        return DTensor.from_local(
            tensor, root_mesh, replicate_placements, run_check=False
        ).redistribute(
            device_mesh=root_mesh,
            placements=shard_placements,
        )


def _pre_load_state_dict(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, list[Shard]]:
    shards = cast(ShardedTensor, tensor).local_shards()
    if len(shards) == 1 and type(shards[0].tensor) is ShardedTensor:
        inner_tensor = shards[0].tensor
        shards = inner_tensor.local_shards()  # pyre-ignore[16]
        tensor = inner_tensor

    return (tensor, shards if len(shards) > 0 else [])


def _all_gather_dtensor(
    tensor: DTensor,
    parent_mesh: DeviceMesh | None,
) -> torch.Tensor:
    """All gather a DTensor in its FSDP dimension and return the local tensor."""
    if parent_mesh != tensor.device_mesh:
        raise AssertionError

    placements = list(copy.deepcopy(tensor.placements))
    # FSDP + TP: [Shard(0), tp_placement] -> [Replicate(), tp_placement]
    # HSDP + TP: [Replicate(), Shard(0), tp_placement] -> [Replicate(), Replicate(), tp_placement]
    for i in range(len(placements) - 1):
        placements[i] = Replicate()
    tensor = tensor.redistribute(
        device_mesh=tensor.device_mesh,
        placements=placements,
    )

    return tensor.to_local()


class DTensorExtensions(FSDPExtensions):
    """
    DTensorExtension is the TensorFlattener extension needed for 2D FSDP + TP.

    This is the implementation for FSDPExtensions defined in
    https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/_fsdp_extensions.py
    """

    def __init__(self, device_handle) -> None:
        super().__init__()
        self.compute_stream = None
        self.device_handle = device_handle
        # we have to use the dynamo disable this way to disable dynamo as the decorator way would
        # trigger build failure with torch deploy...
        self.post_unflatten_transform = torch._dynamo.disable(  # type: ignore[method-assign]
            self.post_unflatten_transform
        )

    def pre_flatten_transform(
        self,
        tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, Any | None]:
        return _flatten_tensor(tensor)

    def post_unflatten_transform(
        self, tensor: torch.Tensor, param_extension: Any
    ) -> torch.Tensor:
        stream = self.compute_stream or self.device_handle.current_stream()
        with self.device_handle.stream(stream):
            # runtime we put the unflattened tensor call on the compute stream since
            # the unflattened tensor might contain computations in fwd/bwd where we
            # need to sync properly.
            # TODO: this is a short term fix and we should make the get_unflat_views
            # directly happen in the compute stream.
            result = _unflatten_tensor(
                tensor,
                param_extension,
                device_handle=self.device_handle,
                compute_stream=self.compute_stream,
            )
            _set_fsdp_flattened(result)
            return result

    def chunk_tensor(
        self,
        tensor: torch.Tensor,
        rank: int,
        world_size: int,
        num_devices_per_node: int,
        pg: dist.ProcessGroup,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        return _chunk_tensor(tensor, rank, world_size, num_devices_per_node, pg)

    def chunk_dtensor(
        self,
        tensor: torch.Tensor,
        rank: int,
        device_mesh: DeviceMesh,
    ) -> torch.Tensor:
        return _chunk_dtensor(tensor, rank, device_mesh)

    def pre_load_state_dict_transform(
        self,
        tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, list[Shard]]:
        return _pre_load_state_dict(tensor)

    def all_gather_dtensor(
        self,
        tensor: DTensor,
        parent_mesh: DeviceMesh | None,
    ) -> torch.Tensor:
        return _all_gather_dtensor(tensor, parent_mesh)
<|endoftext|># Copyright (c) Meta Platforms, Inc. and affiliates
from functools import partial
from typing import Any

import torch
from torch.distributed.tensor import DeviceMesh, DTensor, Replicate, Shard
from torch.distributed.tensor.placement_types import _is_shard_like


__all__ = [
    "input_reshard",
]


def input_reshard(
    module: torch.nn.Module,
    tp_device_mesh: DeviceMesh,
    input_reshard_dim: int | None = None,
) -> torch.nn.Module:
    """
    Register hooks to an nn.Module for input resharding, enabling sharding and restoration during backward computation.

    Register hooks to an nn.Module with input resharding so that we can shard
    per the given `tp_device_mesh` and `input_reshard_dim` and restore the
    input back when recomputing the activations in the backward. The reason
    why we can do this is that for Tensor Parallel(TP), the input are same
    across all TP ranks.

    Args:
        module (:class:`nn.Module`):
            Module to be registered with input resharding.
        tp_device_mesh (:class:`DeviceMesh`):
            Object which describes the mesh topology
            of devices for Tensor Parallel.
        input_reshard_dim (Optional[int]):
            The dimension of where we perform the sharding
            of input. If set None, there is no sharding of input.
            Default: None

    Return:
        A :class:`nn.Module` object registered with TP input resharding.
    """
    if input_reshard_dim is None:
        return module

    cx: torch.autograd.graph.saved_tensors_hooks | None = None

    def input_reshard_forward_pre_hook(_: torch.nn.Module, _i: tuple[Any, ...]) -> None:
        saved_tensor_hooks = torch.autograd.graph.saved_tensors_hooks(
            partial(_pack_hook_tp, tp_device_mesh, input_reshard_dim),
            partial(_unpack_hook_tp, tp_device_mesh, input_reshard_dim),
        )
        saved_tensor_hooks.__enter__()
        nonlocal cx
        cx = saved_tensor_hooks  # type: ignore[name-defined]

    def input_reshard_backward_hook(
        _: torch.nn.Module, _i: tuple[Any, ...], _o: Any
    ) -> Any:
        nonlocal cx
        cx.__exit__()  # type: ignore[name-defined, union-attr]

    module.register_forward_pre_hook(input_reshard_forward_pre_hook)
    module.register_forward_hook(input_reshard_backward_hook)
    return module


def _pack_hook_tp(mesh: DeviceMesh, input_reshard_dim: int, x: torch.Tensor) -> Any:
    """Hook function called after FWD to shard input."""
    if isinstance(x, DTensor) and all(p.is_replicate() for p in x._spec.placements):
        return x.redistribute(device_mesh=mesh, placements=[Shard(input_reshard_dim)])
    elif (
        not isinstance(x, DTensor)
        and isinstance(x, torch.Tensor)
        and x.numel() >= mesh.size()
    ):
        return (
            DTensor.from_local(x, device_mesh=mesh)
            .redistribute(device_mesh=mesh, placements=[Shard(input_reshard_dim)])
            .to_local()
        )
    else:
        return x


def _unpack_hook_tp(mesh: DeviceMesh, input_reshard_dim: int, x: Any) -> torch.Tensor:
    """Hook function called before activation recomputing in BWD to restore input."""
    if (
        isinstance(x, DTensor)
        and len(x._spec.placements) == 1
        and _is_shard_like(x._spec.placements[0])
    ):
        return x.redistribute(device_mesh=mesh, placements=[Replicate()])
    elif (
        not isinstance(x, DTensor)
        and isinstance(x, torch.Tensor)
        and x.numel() >= mesh.size()
    ):
        return (
            DTensor.from_local(
                x, device_mesh=mesh, placements=[Shard(input_reshard_dim)]
            )
            .redistribute(device_mesh=mesh, placements=[Replicate()])
            .to_local()
        )
    else:
        return x
<|endoftext|># mypy: allow-untyped-defs
from typing import Any

import torch.nn as nn
from torch.distributed.tensor.parallel._data_parallel_utils import (
    _flatten_tensor,
    _unflatten_tensor,
)


__all__ = []  # type: ignore[var-annotated]


def _get_submodule_n_params(module: nn.Module, path: str):
    """
    Get submodule and the direct path of parameter from the module
    """
    if "." in path:
        path_list = path.split(".")
        parent_module_path = ".".join(path_list[:-1])
        module = module.get_submodule(parent_module_path)
        path = path_list[-1]
    return module, path


def _update_module_param(param_list: list[tuple[nn.Module, str, nn.Parameter]]):
    """
    Update parameters within the module
    """
    for item in param_list:
        parent_module, module_path, t = item
        if not hasattr(parent_module, module_path):
            raise AssertionError
        delattr(parent_module, module_path)
        setattr(parent_module, module_path, t)


def _reconstruct_dtensor(module: nn.Module, _input: Any):
    """
    Reconstruct DTensor parameters from local tensors
    """
    param_list = []
    # TODO: To add perf optimizations to this iterations
    for name, t in module.named_parameters():
        if hasattr(t, "_st_info"):
            dtensor = _unflatten_tensor(t, t._st_info)
            param_list.append((*_get_submodule_n_params(module, name), dtensor))
    _update_module_param(param_list)  # type: ignore[arg-type]


def _localize_dtensor(
    module: nn.Module, *_: Any, ignored_params: set[nn.Parameter] | None = None
):
    """
    Convert DTensor parameters to local tensors
    """
    if ignored_params is None:
        ignored_params = set()
    param_list = []
    for name, param in module.named_parameters():
        if param in ignored_params:
            continue
        t, sharding_info = _flatten_tensor(param)
        if sharding_info is not None:
            t = nn.Parameter(t)
            t._st_info = sharding_info  # type: ignore[attr-defined]
            param_list.append((*_get_submodule_n_params(module, name), t))
    _update_module_param(param_list)  # type: ignore[arg-type]


def _pre_dp_module_transform(module: nn.Module):
    """
    Enable the composability between Tensor Parallelism (TP) and Data
    Parallelism(DP) in PyTorch when using DDP. We need to convert Parameters which
    are DTensors to local tensors before wrapping with data parallelism API.
    We then register two hooks, one for converting local tensors back to DTensor
    preforward and one to convert DTensors back to tensors after Forward. By
    integrating this way, we avoid any special handling of DTensor parameters by DDP
    and get DTensor's gradients propagated back to DP, e.g. gradient buckets of DDP.

    For now, this API only works with ``DistributedDataParallel``. It will later support
    other DP methods such as FSDP.

    Args:
        module (:class:`nn.Module`):
            Module which has been applied TP on.

    Example::
        >>> # xdoctest: +SKIP("distributed")
        >>> from torch.distributed.tensor.parallel import parallelize_module, PairwiseParallel
        >>> from torch.nn.parallel import DistributedDataParallel as DDP
        >>> from torch.distributed.tensor.parallel.ddp import pre_dp_module_transform
        >>>
        >>> # Define the module.
        >>> m = module(...)
        >>> parallelize_module(m, PairwiseParallel())
        >>> m = pre_dp_module_transform(m)
        >>> m = DDP(m)
        >>>
    """

    _localize_dtensor(module, None, None)
    # TODO: To add test cases and ensure that it works for nested modules
    module.register_forward_pre_hook(_reconstruct_dtensor)
    module.register_forward_hook(_localize_dtensor)
<|endoftext|>from functools import partial
from typing import no_type_check

import torch
from torch.distributed._functional_collectives import AsyncCollectiveTensor
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._dtensor_spec import DTensorSpec


@no_type_check
def sync_grad_hook(grad, *, device_handle=None, compute_stream=None):
    if isinstance(grad, AsyncCollectiveTensor):
        if compute_stream is not None:
            with device_handle.stream(compute_stream):
                grad = grad.wait()
        else:
            grad = grad.wait()

    return grad


def _flatten_tensor(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, DTensorSpec | None]:
    if isinstance(tensor, DTensor):
        tensor._local_tensor.requires_grad_()
        return tensor._local_tensor, tensor._spec
    return tensor, None


@no_type_check
def _unflatten_tensor(tensor, spec, *, device_handle=None, compute_stream=None):
    # unflatten would mainly be called every time FSDP allgather parameters.
    result = DTensor.from_local(
        tensor,
        spec.mesh,
        spec.placements,
        run_check=False,
        shape=spec.shape,
        stride=spec.stride,
    )
    if tensor.requires_grad:
        # only register the hook if the tensor requires grad
        tensor.register_hook(
            partial(
                sync_grad_hook,
                device_handle=device_handle,
                compute_stream=compute_stream,
            )
        )
    return result
<|endoftext|># Copyright (c) Meta Platforms, Inc. and affiliates
from torch.distributed.tensor.parallel.api import parallelize_module
from torch.distributed.tensor.parallel.loss import loss_parallel
from torch.distributed.tensor.parallel.style import (
    ColwiseParallel,
    ParallelStyle,
    PrepareModuleInput,
    PrepareModuleInputOutput,
    PrepareModuleOutput,
    RowwiseParallel,
    SequenceParallel,
)


__all__ = [
    "ColwiseParallel",
    "ParallelStyle",
    "PrepareModuleInput",
    "PrepareModuleInputOutput",
    "PrepareModuleOutput",
    "RowwiseParallel",
    "SequenceParallel",
    "parallelize_module",
    "loss_parallel",
]
<|endoftext|># Copyright (c) Meta Platforms, Inc. and affiliates
import warnings
from fnmatch import fnmatch

import torch
import torch.nn as nn
from torch.distributed.device_mesh import _mesh_resources, DeviceMesh
from torch.distributed.tensor.parallel.style import ParallelStyle


__all__ = ["parallelize_module"]


def parallelize_module(  # type: ignore[return]
    module: nn.Module,
    device_mesh: DeviceMesh | None = None,
    parallelize_plan: ParallelStyle | dict[str, ParallelStyle] | None = None,
    *,
    src_data_rank: int | None = 0,
) -> nn.Module:
    """
    Apply Tensor Parallelism in PyTorch by parallelizing modules or sub-modules based on a user-specified plan.

    We parallelize module or sub_modules based on a parallelize_plan. The parallelize_plan contains
    :class:`ParallelStyle`, which indicates how user wants the module or sub_module
    to be parallelized.

    User can also specify different parallel style per module fully qualified name (FQN).

    Note that ``parallelize_module`` only accepts a 1-D :class:`DeviceMesh`, if you have a 2-D or N-D :class:`DeviceMesh`,
    slice the DeviceMesh to a 1-D sub DeviceMesh first then pass to this API(i.e. ``device_mesh[\"tp\"]``)

    Args:
        module (:class:`nn.Module`):
            Module to be parallelized.
        device_mesh (:class:`DeviceMesh`, optional):
            Object which describes the mesh topology of devices for the DTensor.
            If not specified, the call must be under a DeviceMesh context.
        parallelize_plan (Union[:class:`ParallelStyle`, Dict[str, :class:`ParallelStyle`]], optional):
            The plan used to parallelize the module. It can be either a
            :class:`ParallelStyle` object which contains how we prepare
            input/output for Tensor Parallelism or it can be a dict of module
            FQN and its corresponding :class:`ParallelStyle` object. If not
            specified, the call will do nothing at the moment.
    Keyword args:
        src_data_rank (int, optional): the rank of the source data for the logical/global tensor, it is used by
            :meth:`distribute_tensor` to scatter/broadcast the shards/replicas to other ranks. By default,
            we use ``group_rank=0`` on each DeviceMesh dimension as the source data to preserve the single-device
            semantic. If passing ``None`` explicitly, :meth:`parallelize_module` simply uses its local data instead
            of trying to preserve the single-device semantic via scatter/broadcast. Default: 0
    Return:
        A :class:`nn.Module` object parallelized.

    Example::
        >>> # xdoctest: +SKIP("distributed")
        >>> from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel
        >>> from torch.distributed.device_mesh import init_device_mesh
        >>>
        >>> # Define the module.
        >>> m = Model(...)
        >>> tp_mesh = init_device_mesh("cuda", (8,))
        >>> m = parallelize_module(m, tp_mesh, {"w1": ColwiseParallel(), "w2": RowwiseParallel()})
        >>>

    .. note:: For complex module architecture like Attention, MLP layers, we recommend composing
        different ParallelStyles together (i.e. ``ColwiseParallel`` and ``RowwiseParallel``) and pass
        as a parallelize_plan, to achieve the desired sharding computation.
    """
    torch._C._log_api_usage_once("torch.distributed.tensor.parallel.parallelize_module")

    device_mesh = device_mesh or _mesh_resources.get_current_mesh()

    if parallelize_plan is None:
        warnings.warn(
            "No parallelize_plan is provided and auto-parallel is not supported "
            "at the moment, so this parallelize_module call will do nothing.",
            stacklevel=2,
        )
        return module

    # note: The RNG tracker will be initialized in distribute_tensor() call if it hasn't
    # been initialized.

    if isinstance(parallelize_plan, ParallelStyle):
        parallelize_plan.src_data_rank = src_data_rank
        return parallelize_plan._apply(module, device_mesh)
    elif isinstance(parallelize_plan, dict):
        for module_path, parallelize_style in parallelize_plan.items():
            if module_path == "":
                # shortcut: empty string means to apply the plan to the current module
                parallelize_module(module, device_mesh, parallelize_style)
                continue

            path_splits = module_path.split(".")
            # Instead of blindly popping tokens, first check the match,
            # we only consume/pop the token if we found a match.
            token = path_splits[0]

            matched_children = list(
                filter(
                    # `t[0]` is child name
                    lambda t: fnmatch(t[0], token),
                    module.named_children(),
                )
            )
            if not matched_children:
                # No match at this level. Log a warning and process next plan entry.
                warnings.warn(
                    f"Parallelize plan key '{module_path}' could not be resolved: "
                    f"no submodule matching token '{token}' in module {module}, "
                    f"skipping this plan entry.",
                    stacklevel=2,
                )
                continue

            # Now that we have a match, we can consume the token.
            path_splits.pop(0)
            # apply the plan to all matched submodules
            for _, submodule in matched_children:
                if path_splits:
                    # we haven't reached the leaf, apply in dict style
                    leaf_path = ".".join(path_splits)  # rest of the path after `token`
                    parallelize_module(
                        submodule,
                        device_mesh,
                        {leaf_path: parallelize_style},
                        src_data_rank=src_data_rank,
                    )
                else:
                    # otherwise, directly apply style to this submodule
                    parallelize_module(
                        submodule,
                        device_mesh,
                        parallelize_style,
                        src_data_rank=src_data_rank,
                    )
        return module
    else:
        raise TypeError(  # pyre-ignore[7]
            "Expect Union[ParallelStyle, Dict[str, ParallelStyle]] for"
            f" parallelize_plan, {type(parallelize_plan)} found!"
        )
<|endoftext|># mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
from abc import ABC, abstractmethod
from functools import partial
from typing import Any

import torch
import torch.nn as nn
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_module,
    distribute_tensor,
    DTensor,
    Replicate,
    Shard,
)
from torch.distributed.tensor.placement_types import Placement


__all__ = [
    "ParallelStyle",
    "RowwiseParallel",
    "SequenceParallel",
    "ColwiseParallel",
    "PrepareModuleInput",
    "PrepareModuleInputOutput",
    "PrepareModuleOutput",
]


class ParallelStyle(ABC):
    """
    The parallel style contract defines how the module or submodule should be parallelized.

    It only defines the ``apply`` method for ``parallelize_module`` to use, this allows maximum
    flexibility for different kind of style implementations.
    """

    src_data_rank: int | None = 0

    @abstractmethod
    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module: ...


class ColwiseParallel(ParallelStyle):
    """
    Partition a compatible nn.Module in a column-wise fashion. Currently supports nn.Linear and nn.Embedding.
    Users can compose it together with RowwiseParallel to achieve the sharding of more complicated modules.
    (i.e. MLP, Attention)

    Keyword Args:
        input_layouts (Placement, optional):
            The DTensor layout of input tensor for the nn.Module, this is used to annotate the input tensor to
            become a DTensor. If not specified, we assume the input tensor to be replicated.
        output_layouts (Placement, optional):
            The DTensor layout of the output for the nn.Module, this is used to ensure the output of the nn.Module
            with the user desired layout. If not specified, the output tensor is sharded on the last dimension.
        use_local_output (bool, optional):
            Whether to use local :class:`torch.Tensor` instead of :class:`DTensor` for the module output, default: True.
    Returns:
        A :class:`ParallelStyle` object that represents Colwise sharding of the nn.Module.

    Example::
        >>> # xdoctest: +SKIP(failing)
        >>> from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel
        >>> from torch.distributed.device_mesh import init_device_mesh
        >>> ...
        >>> m = Model(...)  # m is a nn.Module that contains a "w1" nn.Linear submodule
        >>> tp_mesh = init_device_mesh("cuda", (8,))
        >>>
        >>> # By default, the input of the "w1" Linear will be converted to Replicated DTensor
        >>> # and the output of "w1" will return :class:`torch.Tensor` that shards on the last dim.
        >>>
        >>> sharded_mod = parallelize_module(m, tp_mesh, {"w1": ColwiseParallel()})
        >>> ...

    .. note:: By default ``ColwiseParallel`` output is sharded on the last dimension if the ``output_layouts`` not
        specified, if there're operators that require specific tensor shape (i.e. before the paired ``RowwiseParallel``),
        keep in mind that if the output is sharded the operator might need to be adjusted to the sharded size.
    """

    def __init__(
        self,
        *,
        input_layouts: Placement | None = None,
        output_layouts: Placement | None = None,
        use_local_output: bool = True,
    ):
        super().__init__()
        self.input_layouts = (input_layouts or Replicate(),)
        self.output_layouts = (output_layouts or Shard(-1),)
        # colwise linear runtime sharding (desired sharding):
        # 1. requires replicate input
        # 2. shard output on last dim
        self.desired_input_layouts = (Replicate(),)
        self.use_local_output = use_local_output

    @staticmethod
    def _prepare_input_fn(
        input_layouts, desired_input_layouts, mod, inputs, device_mesh
    ):
        # TODO: figure out dynamo support for instance method and switch this to instance method

        # annotate module input placements/sharding with input_layouts
        input_tensor = inputs[0]
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(
                input_tensor,
                device_mesh,
                input_layouts,
                run_check=False,
            )

        # transform the input layouts to the desired layouts of ColwiseParallel
        if input_layouts != desired_input_layouts:
            input_tensor = input_tensor.redistribute(
                placements=desired_input_layouts, async_op=True
            )
        return input_tensor

    def _partition_linear_fn(self, name, module, device_mesh):
        # colwise shard weight/bias to Shard(0), weight be Shard(0)
        # means Colwise as Linear is input * weight^T + bias, where
        # weight would become Shard(1)
        for name, param in module.named_parameters():
            dist_param = nn.Parameter(
                distribute_tensor(
                    param, device_mesh, [Shard(0)], src_data_rank=self.src_data_rank
                ),
                requires_grad=param.requires_grad,
            )
            module.register_parameter(name, dist_param)

    def _partition_embedding_fn(self, name, module, device_mesh):
        # colwise shard embedding.weight is straight forward as Shard(1)
        for name, param in module.named_parameters():
            dist_param = nn.Parameter(
                distribute_tensor(
                    param, device_mesh, [Shard(1)], src_data_rank=self.src_data_rank
                ),
                requires_grad=param.requires_grad,
            )
            module.register_parameter(name, dist_param)

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        # outputs is a shard on last dimension DTensor, i.e. Shard(-1)
        if outputs.placements != output_layouts:
            outputs = outputs.redistribute(placements=output_layouts, async_op=True)
        # back to local tensor
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        if isinstance(module, nn.Linear):
            partition_fn = self._partition_linear_fn
        elif isinstance(module, nn.Embedding):
            partition_fn = self._partition_embedding_fn
        else:
            raise NotImplementedError(
                "ColwiseParallel currently only support nn.Linear and nn.Embedding!"
            )

        return distribute_module(
            module,
            device_mesh,
            partition_fn,
            partial(
                self._prepare_input_fn,
                self.input_layouts,
                self.desired_input_layouts,
            ),
            partial(
                self._prepare_output_fn, self.output_layouts, self.use_local_output
            ),
        )

    def __repr__(self) -> str:
        tmpstr = self.__class__.__name__ + "("
        tmpstr += f"input_layouts={self.input_layouts}, "
        tmpstr += f"output_layouts={self.output_layouts}, "
        tmpstr += f"use_local_output={self.use_local_output}"
        tmpstr += ")"
        return tmpstr


class RowwiseParallel(ParallelStyle):
    """
    Partition a compatible nn.Module in a row-wise fashion. Currently supports nn.Linear and nn.Embedding.
    Users can compose it with ColwiseParallel to achieve the sharding of more complicated modules.
    (i.e. MLP, Attention)

    Keyword Args:
        input_layouts (Placement, optional):
            The DTensor layout of input tensor for the nn.Module, this is used to annotate the input tensor to
            become a DTensor. If not specified, we assume the input tensor to be sharded on the last dimension.
        output_layouts (Placement, optional):
            The DTensor layout of the output for the nn.Module, this is used to ensure the output of the nn.Module
            with the user desired layout. If not specified, the output tensor is replicated.
        use_local_output (bool, optional):
            Whether to use local :class:`torch.Tensor` instead of :class:`DTensor` for the module output, default: True.
    Returns:
        A :class:`ParallelStyle` object that represents Rowwise sharding of the nn.Module.

    Example::
        >>> # xdoctest: +SKIP(failing)
        >>> from torch.distributed.tensor.parallel import parallelize_module, RowwiseParallel
        >>> from torch.distributed.device_mesh import init_device_mesh
        >>> ...
        >>> m = Model(...)  # m is a nn.Module that contains a "w2" nn.Linear submodule
        >>> tp_mesh = init_device_mesh("cuda", (8,))
        >>>
        >>> # By default, the input of the "w2" Linear will be converted to DTensor that shards on the last dim
        >>> # and the output of "w2" will return a replicated :class:`torch.Tensor`.
        >>>
        >>> sharded_mod = parallelize_module(m, tp_mesh, {"w2": RowwiseParallel()}),
        >>> ...
    """

    def __init__(
        self,
        *,
        input_layouts: Placement | None = None,
        output_layouts: Placement | None = None,
        use_local_output: bool = True,
    ):
        super().__init__()
        self.input_layouts = (input_layouts or Shard(-1),)
        self.output_layouts = (output_layouts or Replicate(),)
        self.use_local_output = use_local_output

    @staticmethod
    def _prepare_input_fn(
        input_layouts, desired_input_layouts, mod, inputs, device_mesh
    ):
        input_tensor = inputs[0]
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(
                input_tensor,
                device_mesh,
                input_layouts,
                run_check=False,
            )

        if input_layouts != desired_input_layouts:
            input_tensor = input_tensor.redistribute(
                placements=desired_input_layouts, async_op=True
            )
        return input_tensor

    def _partition_linear_fn(self, name, module, device_mesh):
        # Rowwise shard weight to Shard(1), bias to Replicate(), weight be Shard(1)
        # means Rowwise as nn.Linear is input * weight^T + bias, where
        # weight would become Shard(0)
        module.register_parameter(
            "weight",
            nn.Parameter(
                distribute_tensor(
                    module.weight,
                    device_mesh,
                    [Shard(1)],
                    src_data_rank=self.src_data_rank,
                ),
                requires_grad=module.weight.requires_grad,
            ),
        )
        if getattr(module, "bias", None) is not None:
            # The Linear module has bias
            module.register_parameter(
                "bias",
                nn.Parameter(
                    distribute_tensor(
                        module.bias,
                        device_mesh,
                        [Replicate()],
                        src_data_rank=self.src_data_rank,
                    ),
                    requires_grad=module.bias.requires_grad,
                ),
            )

    def _partition_embedding_fn(self, name, module, device_mesh):
        # rowwise shard embedding.weight is Shard(0)
        for name, param in module.named_parameters():
            dist_param = nn.Parameter(
                distribute_tensor(
                    param, device_mesh, [Shard(0)], src_data_rank=self.src_data_rank
                ),
                requires_grad=param.requires_grad,
            )
            module.register_parameter(name, dist_param)

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        # Rowwise sharding produces partial output, depending on output layouts:
        # 1. to replicate -> allreduce
        # 2. to shard -> reduce_scatter
        if outputs.placements != output_layouts:
            outputs = outputs.redistribute(placements=output_layouts, async_op=True)
        # back to local tensor if use_local_output is True
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        if isinstance(module, nn.Linear):
            partition_fn = self._partition_linear_fn
            # rowwise linear runtime sharding requires input tensor shard on last dim
            self.desired_input_layouts: tuple[Placement, ...] = (Shard(-1),)
        elif isinstance(module, nn.Embedding):
            partition_fn = self._partition_embedding_fn
            # rowwise embedding runtime sharding requires input tensor replicated
            self.desired_input_layouts = (Replicate(),)
        else:
            raise NotImplementedError(
                "RowwiseParallel currently only support nn.Linear and nn.Embedding!"
            )

        return distribute_module(
            module,
            device_mesh,
            partition_fn,
            partial(
                self._prepare_input_fn,
                self.input_layouts,
                self.desired_input_layouts,
            ),
            partial(
                self._prepare_output_fn, self.output_layouts, self.use_local_output
            ),
        )

    def __repr__(self) -> str:
        tmpstr = self.__class__.__name__ + "("
        tmpstr += f"input_layouts={self.input_layouts}, "
        tmpstr += f"output_layouts={self.output_layouts}, "
        tmpstr += f"use_local_output={self.use_local_output}"
        tmpstr += ")"
        return tmpstr


class SequenceParallel(ParallelStyle):
    """
    SequenceParallel replicates a compatible ``nn.Module`` parameters and runs the sharded computation with
    input sharded on the sequence dimension. This currently supports ``nn.LayerNorm``, ``nn.Dropout``, and the
    `RMSNorm python implementation <https://github.com/facebookresearch/llama/blob/main/llama/model.py#L34>`__

    This style implements the operation that is described in the paper
    `Reducing Activation Recomputation in Large Transformer Models <https://arxiv.org/abs/2205.05198>`__

    If the input passed in to this ``nn.Module`` is a :class:`torch.Tensor`, it assumes that the input is already sharded
    on the sequence dimension and converts the input to a :class:`DTensor` sharded on the sequence dimension. If the input
    passed in to this ``nn.Module`` is already a :class:`DTensor` but is not sharded on the sequence dimension, it would
    redistribute the input to be sharded on the sequence dimension.

    The output of the ``nn.Module`` will be sharded on the sequence dimension.

    Keyword Args:
        sequence_dim (int, optional):
            The sequence dimension of the input tensor for the ``nn.Module``, this is used to annotate the input tensor to
            become a DTensor that is sharded on the sequence dimension, default: 1.
        use_local_output (bool, optional):
            Whether to use local :class:`torch.Tensor` instead of :class:`DTensor` for the module output, default: False.
    Returns:
        A :class:`ParallelStyle` object that represents Sequence Parallel of the ``nn.Module``.

    Example::
        >>> # xdoctest: +SKIP(failing)
        >>> from torch.distributed.tensor.parallel import parallelize_module, SequenceParallel
        >>> from torch.distributed.device_mesh import init_device_mesh
        >>> ...
        >>> m = Model(...)  # m is a nn.Module that contains a "norm" nn.LayerNorm submodule
        >>> tp_mesh = init_device_mesh("cuda", (8,))
        >>>
        >>> # By default, the input of the "norm" will be converted to DTensor that shards on the sequence dim
        >>> # and the output of "norm" will return a sharded on sequence dimension :class:`DTensor`.
        >>>
        >>> sharded_mod = parallelize_module(m, tp_mesh, {"norm": SequenceParallel()}),
        >>> ...

    .. note:: SequenceParallel style assumes ones initialization if there are weights in the nn.Module (i.e.
        ``nn.LayerNorm`` or ``RMSNorm``, and they by default have ones initialization). If you have custom
        inits for the weights on those modules, you need to broadcast the weights before/after parallelizing
        to ensure that they are replicated.
    """

    def __init__(self, *, sequence_dim: int = 1, use_local_output: bool = False):
        super().__init__()
        self.sequence_sharding = (Shard(sequence_dim),)
        self.use_local_output = use_local_output

    def _replicate_module_fn(
        self, name: str, module: nn.Module, device_mesh: DeviceMesh
    ):
        for p_name, param in module.named_parameters():
            # simple replication with fixed ones_ init from LayerNorm/RMSNorm, which allow
            # us to simply just use from_local
            replicated_param = torch.nn.Parameter(
                DTensor.from_local(param, device_mesh, [Replicate()], run_check=False)
            )
            module.register_parameter(p_name, replicated_param)

    @staticmethod
    def _prepare_input_fn(sequence_sharding, mod, inputs, device_mesh):
        input_tensor = inputs[0]
        if isinstance(input_tensor, DTensor):
            # if the passed in input DTensor is not sharded on the sequence dim, we need to redistribute it
            if input_tensor.placements != sequence_sharding:
                input_tensor = input_tensor.redistribute(
                    placements=sequence_sharding, async_op=True
                )
            return input_tensor
        elif isinstance(input_tensor, torch.Tensor):
            # assume the input passed in already sharded on the sequence dim and create the DTensor
            return DTensor.from_local(
                input_tensor, device_mesh, sequence_sharding, run_check=False
            )
        else:
            raise ValueError(
                f"expecting input of {mod} to be a torch.Tensor or DTensor, but got {input_tensor}"
            )

    @staticmethod
    def _prepare_output_fn(use_local_output, mod, outputs, device_mesh):
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            self._replicate_module_fn,
            partial(self._prepare_input_fn, self.sequence_sharding),
            partial(self._prepare_output_fn, self.use_local_output),
        )

    def __repr__(self) -> str:
        tmpstr = self.__class__.__name__ + "("
        if len(self.sequence_sharding) == 1:
            tmpstr += f"sequence_dim={self.sequence_sharding[0].dim}, "
        tmpstr += f"use_local_output={self.use_local_output}"
        tmpstr += ")"
        return tmpstr


class PrepareModuleInput(ParallelStyle):
    """
    Configure the nn.Module's inputs to convert the input tensors of the nn.Module to DTensors at runtime according to
    ``input_layouts``, and perform layout redistribution according to the ``desired_input_layouts``.

    Keyword Args:
        input_layouts (Union[Placement, Tuple[Optional[Placement]]]):
            The DTensor layouts of input tensors for the nn.Module, this is used to convert the input tensors to
            DTensors. If some inputs are not torch.Tensor or no need to convert to DTensors, ``None`` need to be specified
            as a placeholder. default: None.
        desired_input_layouts (Union[Placement, Tuple[Optional[Placement]]]):
            The desired DTensor layout of input tensors for the nn.Module, this is used to ensure the inputs of the nn.Module
            have the desired DTensor layouts. This argument needs to have the same length with ``input_layouts``. default: None.
        input_kwarg_layouts (Dict[str, Placement]):
            The DTensor layouts of input kwargs for the nn.Module, this is used to convert the input kwarg tensors to DTensors.
            default: None
        desired_input_kwarg_layouts: (Dict[str, Placement]):
            The desired DTensor layout of input kwargs for the nn.Module, this is used to ensure the inputs of the nn.Module
            have the desired DTensor layouts. default: None.
        use_local_output (bool, optional):
            Whether to use local :class:`torch.Tensor` instead of :class:`DTensor` for the module inputs, default: False.
    Returns:
        A :class:`ParallelStyle` object that prepares the sharding layouts of the nn.Module's inputs.

    Example::
        >>> # xdoctest: +SKIP(failing)
        >>> from torch.distributed.tensor.parallel import parallelize_module, PrepareModuleInput
        >>> from torch.distributed.device_mesh import init_device_mesh
        >>> ...
        >>> block = TransformerBlock(...)  # block is a nn.Module that contains an "attn" Attention submodule
        >>> tp_mesh = init_device_mesh("cuda", (8,))
        >>>
        >>> # According to the style specified below, the first input of attn will be annotated to Sharded DTensor
        >>> # and then redistributed to Replicated DTensor.
        >>> parallelize_module(
        >>>     block, # this can be a submodule or module
        >>>     tp_mesh,
        >>>     parallelize_plan={
        >>>         "attn": PrepareModuleInput(
        >>>             input_layouts=(Shard(0), None, None, ...),
        >>>             desired_input_layouts=(Replicate(), None, None, ...)
        >>>         ),
        >>>     }
        >>> )
    """

    def __init__(
        self,
        *,
        input_layouts: Placement | tuple[Placement | None, ...] | None = None,
        desired_input_layouts: Placement | tuple[Placement | None, ...] | None = None,
        input_kwarg_layouts: dict[str, Placement] | None = None,
        desired_input_kwarg_layouts: dict[str, Placement] | None = None,
        use_local_output: bool = False,
    ):
        self.input_layouts = (
            (input_layouts,) if isinstance(input_layouts, Placement) else input_layouts
        )
        self.desired_input_layouts = (
            (desired_input_layouts,)
            if isinstance(desired_input_layouts, Placement)
            else desired_input_layouts
        )
        self.use_local_output = use_local_output
        if self.input_layouts is not None:
            if self.desired_input_layouts is None:
                raise AssertionError("desired module inputs should not be None!")
            if len(self.input_layouts) != len(self.desired_input_layouts):
                raise AssertionError(
                    "input_layouts and desired_input_layouts should have same length!"
                )
        self.with_kwargs = input_kwarg_layouts is not None
        self.input_kwarg_layouts = input_kwarg_layouts or {}
        self.desired_input_kwarg_layouts = desired_input_kwarg_layouts or {}
        if self.with_kwargs:
            if len(self.input_kwarg_layouts) != len(self.desired_input_kwarg_layouts):
                raise AssertionError(
                    "input_kwarg_layouts and desired_input_kwarg_layouts should have same length!"
                )

    def _prepare_input_arg(
        self,
        input: Any,
        mesh: DeviceMesh,
        input_layout: Placement | None,
        desired_layout: Placement | None,
    ):
        if input_layout is not None:
            if isinstance(input, DTensor):
                # TODO: re-enable the check once we fix the compile path
                # assert inp.placements[0] == input_layout
                dt_inp = input
            else:
                if not isinstance(input, torch.Tensor):
                    raise AssertionError("expecting input to be a torch.Tensor!")
                dt_inp = DTensor.from_local(
                    input, mesh, (input_layout,), run_check=False
                )

            if desired_layout is not None and input_layout != desired_layout:
                dt_inp = dt_inp.redistribute(placements=(desired_layout,))

            return dt_inp.to_local() if self.use_local_output else dt_inp
        else:
            return input

    def _prepare_input_fn(self, inputs, device_mesh):
        if self.input_layouts is None:
            return inputs
        prepared_inputs = []
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        if len(inputs) != len(self.input_layouts):
            raise ValueError("module inputs and input_layouts should have same length!")

        if self.desired_input_layouts is None:
            raise AssertionError("desired module inputs should not be None!")

        for inp, input_layout, desired_layout in zip(
            inputs, self.input_layouts, self.desired_input_layouts
        ):
            prepared_inputs.append(
                self._prepare_input_arg(inp, device_mesh, input_layout, desired_layout)
            )
        return tuple(prepared_inputs)

    def _prepare_input_kwarg_fn(self, inputs, kwarg_inputs, device_mesh):
        prepared_arg_inputs = self._prepare_input_fn(inputs, device_mesh)
        prepared_kwarg_inputs = {}
        for kwarg_key in kwarg_inputs:
            kwarg_val = kwarg_inputs[kwarg_key]
            input_layout = self.input_kwarg_layouts.get(kwarg_key)
            desired_input_layout = self.desired_input_kwarg_layouts.get(kwarg_key)

            prepared_kwarg_inputs[kwarg_key] = self._prepare_input_arg(
                kwarg_val, device_mesh, input_layout, desired_input_layout
            )

        return (prepared_arg_inputs, prepared_kwarg_inputs)

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        if self.with_kwargs:
            module.register_forward_pre_hook(
                lambda _, inputs, kwargs: self._prepare_input_kwarg_fn(
                    inputs, kwargs, device_mesh
                ),
                with_kwargs=True,
            )  # type: ignore[misc]
        else:
            module.register_forward_pre_hook(
                lambda _, inputs: self._prepare_input_fn(inputs, device_mesh)
            )  # type: ignore[misc, call-arg]
        return module

    def __repr__(self) -> str:
        tmpstr = self.__class__.__name__ + "("
        tmpstr += f"input_layouts={self.input_layouts}, "
        tmpstr += f"desired_input_layouts={self.desired_input_layouts}, "
        tmpstr += f"input_kwarg_layouts={self.input_kwarg_layouts}, "
        tmpstr += f"desired_input_kwarg_layouts={self.desired_input_kwarg_layouts}, "
        tmpstr += f"use_local_output={self.use_local_output}"
        tmpstr += ")"
        return tmpstr


class PrepareModuleOutput(ParallelStyle):
    """
    Configure the nn.Module's outputs to convert the output tensors of the nn.Module to DTensors at runtime according to
    ``output_layouts``, and perform layout redistribution according to the ``desired_output_layouts``.

    Keyword Args:
        output_layouts (Union[Placement, Tuple[Placement]]):
            The DTensor layouts of output tensors for the nn.Module, this is used to convert the output tensors to
            DTensors if they are :class:`torch.Tensor`. If some outputs are not torch.Tensor or no need to convert to DTensors,
            ``None`` need to be specified as a placeholder.
        desired_output_layouts (Union[Placement, Tuple[Placement]]):
            The desired DTensor layouts of output tensors for the nn.Module, this is used to ensure the outputs of the nn.Module
            have the desired DTensor layouts.
        use_local_output (bool, optional):
            Whether to use local :class:`torch.Tensor` instead of :class:`DTensor` for the module outputs, default: True.
    Returns:
        A ParallelStyle object that prepares the sharding layouts of the nn.Module's outputs.

    Example::
        >>> # xdoctest: +SKIP(failing)
        >>> from torch.distributed.tensor.parallel import parallelize_module, PrepareModuleOutput
        >>> from torch.distributed.device_mesh import init_device_mesh
        >>> ...
        >>> block = TransformerBlock(...)  # block is a nn.Module that contains an "attn" Attention submodule
        >>> tp_mesh = init_device_mesh("cuda", (8,))
        >>>
        >>> # According to the style specified below, the output of the TransformerBlock will be converted to Replicated DTensor
        >>> # and then redistributed to Sharded DTensor.
        >>> parallelize_module(
        >>>     block, # this can be a submodule or module
        >>>     tp_mesh,
        >>>     parallelize_plan = PrepareModuleOutput(
        >>>         output_layouts=Replicate(),
        >>>         desired_output_layouts=Shard(0)
        >>>     )
        >>> )
    """

    def __init__(
