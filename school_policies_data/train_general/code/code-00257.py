        guard_or_true,
    )

    from_nelem = prod(from_size)
    to_size = infer_size(from_nelem, normalize_sizes(to_size))

    torch._check(
        from_nelem == prod(to_size),
        lambda: "Total view shape does not add up",
    )

    from_idx = 0
    to_idx = 0
    from_len = len(from_size)
    to_len = len(to_size)

    result_pp: list[DimSpec] = []

    while from_idx < from_len or to_idx < to_len:
        from_group_dim, to_group_shape = [], []

        if from_idx >= from_len:
            f = 1
        else:
            f = from_size[from_idx]
            from_group_dim.append(from_idx)
            from_idx += 1

        if to_idx >= to_len:
            t = 1
        else:
            t = to_size[to_idx]
            to_group_shape.append(t)
            to_idx += 1

        # if any of the groups is singleton, great, we need to backtrack though
        if guard_or_false(f == 1) and guard_or_true(t != 1):
            # produces ([1], [])
            to_idx -= 1
            to_group_shape = []
        elif guard_or_true(f != 1) and guard_or_false(t == 1):
            # produces ([], [1])
            from_idx -= 1
            from_group_dim = []
        else:
            # produces ([1], [1]),  ([2], [2]), ([2,3], [6])
            while not guard_or_false(f == t):
                if from_idx < from_len and (
                    to_idx >= to_len
                    or guard_or_false(t % f == 0)
                    or guard_or_false(t > f)
                ):  # for easier symbolic comparisons, e.g. u0*u1 > u0
                    nf = from_size[from_idx]
                    from_group_dim.append(from_idx)
                    from_idx += 1
                    f *= nf
                elif to_idx < to_len and (
                    from_idx >= from_len
                    or guard_or_false(f % t == 0)
                    or guard_or_false(f > t)
                    or (free_symbols(f) and not free_symbols(t))
                ):
                    nt = to_size[to_idx]
                    to_group_shape.append(nt)
                    to_idx += 1
                    t *= nt
                elif from_idx < from_len and free_symbols(t) and not free_symbols(f):
                    nf = from_size[from_idx]
                    from_group_dim.append(from_idx)
                    from_idx += 1
                    f *= nf
                elif to_idx < to_len and from_idx < from_len:
                    # With unbacked dimensions, neither divisibility nor
                    # ordering may be provable without a data-dependent guard.
                    # Fall back to one conservative reshape group over the
                    # remaining contiguous dims; sharding propagation can then
                    # replicate if the exact split is not statically known.
                    while from_idx < from_len:
                        nf = from_size[from_idx]
                        from_group_dim.append(from_idx)
                        from_idx += 1
                        f *= nf
                    while to_idx < to_len:
                        nt = to_size[to_idx]
                        to_group_shape.append(nt)
                        to_idx += 1
                        t *= nt
                    break
                else:
                    break

        if len(to_group_shape) > 0:
            flattened = Flatten.new(
                tuple(
                    InputDim(fi)
                    for fi in from_group_dim
                    if guard_or_true(from_size[fi] >= 1)
                )
            )
            result_pp += [
                Split.new(flattened, tuple(to_group_shape), i)
                for i in range(len(to_group_shape))
            ]

    return tuple(result_pp)


def dim_tile(ndim: int, dims: tuple[int, ...]) -> DimMap:
    if len(dims) < ndim:
        dims = (1,) * (ndim - len(dims)) + dims
    return dim_repeat(ndim, dims)


def dim_transpose(ndim: int, dim1: int, dim2: int) -> DimMap:
    dim1 = normalize_dim(dim1, ndim)
    dim2 = normalize_dim(dim2, ndim)
    if not dim1 < ndim:
        raise AssertionError(f"Expected dim1 < ndim, got {dim1} >= {ndim}")
    if not dim2 < ndim:
        raise AssertionError(f"Expected dim2 < ndim, got {dim2} >= {ndim}")
    dimmap = [InputDim(i) for i in range(ndim)]
    swapdim = dimmap[dim1]
    dimmap[dim1] = dimmap[dim2]
    dimmap[dim2] = swapdim
    return tuple(dimmap)


def dim_squeeze(shape: Shape, dim: DimsType | None = None) -> DimMap:
    # Operates on local shape; sharding_prop rewrites squeeze ops to squeeze.dims
    # with only globally-singleton dims before this is called.
    from torch.fx.experimental.symbolic_shapes import guard_or_true

    ndim = len(shape)
    if dim is None:
        target_dims = set(range(ndim))
    elif isinstance(dim, int):
        target_dims = {normalize_dim(dim, ndim)}
    else:
        target_dims = set(normalize_dims(dim, ndim))
    return tuple(
        InputDim(i)
        for i, s in enumerate(shape)
        if guard_or_true(s > 1) or i not in target_dims
    )


def dim_unsqueeze(ndim: int, dim: int) -> DimMap:
    dims = tuple(InputDim(i) for i in range(ndim))
    if dim < 0:
        dim += ndim + 1
    return dims[:dim] + (Singleton(),) + dims[dim:]


def dim_view_as_real(shape: Shape) -> DimMap:
    ndim = len(shape)
    results: list[DimSpec] = [InputDim(i) for i in range(ndim - 1)]
    # each complex number is split into two real numbers,
    # resulting in one more dimension of size 2
    results.append(Split(InputDim(ndim - 1), (shape[-1], 2), 0))
    results.append(Split(InputDim(ndim - 1), (shape[-1], 2), 1))
    return tuple(results)


def dim_reduction(ndim: int, dim_or_dims: DimsType | None, keepdim: bool) -> DimMap:
    """
    General fallback for reduction ops where Partial() does not apply.

    This will cause incoming tensor to be replicated on the reducing dimensions.
    """
    if dim_or_dims is None:
        dim_or_dims = tuple(range(ndim))
    if isinstance(dim_or_dims, int):
        dim_or_dims = (dim_or_dims,)
    dim_or_dims = tuple(d if d >= 0 else d + ndim for d in dim_or_dims)
    return tuple(
        InputDim(i) if i not in dim_or_dims else Singleton()
        for i in range(ndim)
        if i not in dim_or_dims or keepdim
    )


dim_maps: dict[Callable[..., torch.Tensor], Callable[..., DimMap]] = {
    torch.atleast_1d: lambda x: dim_pad_left(x.ndim, 1),
    torch.atleast_2d: lambda x: dim_pad_left(x.ndim, 2),
    torch.atleast_3d: lambda x: dim_atleast_3d(x.ndim),
    torch.broadcast_to: lambda input, shape: expand(input.shape, shape),
    Tensor.expand: lambda self, *sizes: expand(self.shape, normalize_sizes(sizes)),
    torch.flatten: lambda tensor: dim_flatten(tensor.ndim),
    torch.movedim: lambda input, source, destination: dim_movedim(
        input.ndim, source, destination
    ),
    torch.permute: lambda input, dims: tuple(
        InputDim(i) for i in normalize_dims(dims, input.ndim)
    ),
    torch.ravel: lambda tensor: dim_flatten(tensor.ndim),
    Tensor.repeat: lambda self, *sizes: dim_repeat(self.ndim, sizes),
    torch.reshape: lambda input, shape: view_groups(input.shape, shape),
    torch.squeeze: lambda input, dim=None: dim_squeeze(input.shape, dim),
    torch.tile: lambda input, dims: dim_tile(input.ndim, dims),
    torch.transpose: lambda input, dim0, dim1: dim_transpose(input.ndim, dim0, dim1),
    torch.unsqueeze: lambda input, dim: dim_unsqueeze(input.ndim, dim),
    Tensor.view: lambda input, *shape: view_groups(input.shape, shape),
    torch.view_as_complex: lambda input: dim_flatten(input.ndim, input.ndim - 2),
    torch.view_as_real: lambda input: dim_view_as_real(input.shape),
}


def propagate_shape_and_sharding(
    input_src_placements: Sequence[Placement],
    global_input_shape: Shape,
    rule: DimMap,
    mesh_sizes: Shape,
    strict_view: bool = False,
) -> tuple[Sequence[Placement], Sequence[Placement]]:
    """
    Determine input target sharding and output sharding based on
    given global tensor shape and input source sharding.

    Sharding propagation follows mapped dimensions:
    - An output dimension that maps directly to an input dimension is sharded equally
    - An output dimension that is a flattened set of input dimensions can be sharded:
      the first sharded dim stays as Shard, non-first sharded dims become _StridedShard
    - An output dimension that is a split of the input dimension can only be sharded
      if the leftmost split size is divisible by the mesh dimension
    """
    propagator = _ViewShardingPropagator(
        input_src_placements, global_input_shape, rule, mesh_sizes, strict_view
    )
    input_tgt_placements, input_to_output_tensor_dims = propagator.analyze()
    output_placements = propagator.rewrite_output_placements(
        input_tgt_placements, input_to_output_tensor_dims
    )
    return input_tgt_placements, output_placements


class _ViewShardingPropagator:
    """Two-phase sharding propagator for view ops.

    Phase 1 — ``analyze()``:
      Walks the DimMap rule and returns:
      - ``input_tgt_placements``: input placements with unshardable dims
        demoted to Replicate.
      - ``input_to_output_tensor_dims``: maps each input tensor dim to its
        output dim(s).  Cardinality encodes the op type: 1→1 for InputDim,
        N→1 for Flatten, 1→N for Split/unflatten.

    Phase 2 — ``rewrite_output_placements()``:
      Consumes both Phase 1 outputs.  Iterates mesh dims 0..n-1, maintaining:
      - ``strided_shard_claimed_dims``: (input_dim, output_dim) pairs already assigned
        to a mesh dim by _StridedShard rewriting.
      - ``local_tensor_shapes``: global shape progressively divided by each
        mesh dim's shard size.
      For each surviving Shard/_StridedShard, looks up the output dim(s) and
      produces the final output placement.
    """

    def __init__(
        self,
        input_src_placements: Sequence[Placement],
        global_input_shape: Shape,
        rule: DimMap,
        mesh_sizes: Shape,
        strict_view: bool,
    ) -> None:
        self.input_src_placements = input_src_placements
        self.global_input_shape = global_input_shape
        self.rule = rule
        self.mesh_sizes = mesh_sizes
        self.strict_view = strict_view
        self.mesh_ndim = len(mesh_sizes)

        # shard_allowed[input_dim][mesh_dim]: whether input_dim can stay
        # sharded on mesh_dim.  Populated by _analyze_dim and its helpers.
        self.shard_allowed: dict[int, list[bool]] = {}
        # Mesh dims whose _StridedShard has already been matched to an output dim.
        # Populated by _analyze_split.
        self.matched_strided_mesh_dims: set[int] = set()
        self.strict_replicate_fallback: set[tuple[int, int]] = set()

    # ------------------------------------------------------------------
    # Public API: analyze → rewrite_output_placements
    # ------------------------------------------------------------------

    def analyze(
        self,
    ) -> tuple[Sequence[Placement], dict[int, list[int]]]:
        """Phase 1: walk the DimMap rule, return (input_tgt_placements, input_to_output_tensor_dims)."""
        input_dims_in_rule = self._input_dims_in_rule(self.rule)

        # Default: shardable if the dim appears in the rule. Refined by _analyze_*.
        for dim in range(len(self.global_input_shape)):
            self.shard_allowed[dim] = [dim in input_dims_in_rule] * self.mesh_ndim

        # Walk the rule to refine shard_allowed and build input_to_output_tensor_dims.
        #
        # Flatten example: view([2, 3, 4], [6, 4])
        #   rule = (Flatten(InputDim(0), InputDim(1)), InputDim(2))
        #   output_dim=0 (Flatten): hits the isinstance(cmd, Flatten) branch.
        #     Maps input dims 0 and 1 to output dim 0.  Result: {0: [0], 1: [0]}
        #   output_dim=1 (InputDim(2)): hits the len(in_dims) > 0 branch.
        #     Maps input dim 2 to output dim 1.  Result: {0: [0], 1: [0], 2: [1]}
        #
        # Split example: view([6], [2, 3])
        #   rule = (Split(InputDim(0), (2,3), 0), Split(InputDim(0), (2,3), 1))
        #   output_dim=0 (split_id=0): hits the len(in_dims) > 0 branch.
        #     Maps input dim 0 to output dim 0.  Result: {0: [0]}
        #   output_dim=1 (split_id=1): hits the isinstance(cmd, Split) branch
        #     because _analyze_split returns [] for split_id>0.  Chases root
        #     InputDim(0) and appends output dim 1.  Result: {0: [0, 1]}
        input_to_output_tensor_dims: dict[int, list[int]] = {}
        for output_dim, cmd in enumerate(self.rule):
            in_dims = self._analyze_dim(cmd)
            if isinstance(cmd, Flatten):
                for in_dim in in_dims:
                    if in_dim.input_dim in input_to_output_tensor_dims:
                        raise AssertionError(
                            f"Input dim {in_dim.input_dim} already mapped to output dims "
                            f"{input_to_output_tensor_dims[in_dim.input_dim]}"
                        )
                    input_to_output_tensor_dims[in_dim.input_dim] = [output_dim]
            elif len(in_dims) > 0:
                # InputDim (identity) or Split(split_id=0).
                in_dim = in_dims[0]
                if in_dim.input_dim not in input_to_output_tensor_dims:
                    input_to_output_tensor_dims[in_dim.input_dim] = [output_dim]
                else:
                    input_to_output_tensor_dims[in_dim.input_dim].append(output_dim)
            elif isinstance(cmd, Split):
                # Split(split_id>0): _analyze_split returned [], so chase the
                # root input dim and append this output dim to its existing entry.
                #
                # Flatten+Split example: view([2, 3], [3, 2])
                #   rule = (Split(Flatten(InputDim(0), InputDim(1)), (3,2), 0),
                #           Split(Flatten(InputDim(0), InputDim(1)), (3,2), 1))
                #   output_dim=0 (split_id=0): same as Split example above.
                #     Result: {0: [0]}
                #   output_dim=1 (split_id=1): same as Split example, but
                #     the chase unwraps the inner Flatten to find InputDim(0).
                #     Result: {0: [0, 1]}
                root_spec = cmd.input_dim
                while isinstance(root_spec, (Flatten, Split)):
                    if isinstance(root_spec, Flatten):
                        # _analyze_flatten always returns input_dims[0] as
                        # the first element (either as the only shardable dim
                        # in non-strict mode, or as the fallback when nothing
                        # is sharded), so split_id=0 uses it as the key in
                        # input_to_output_tensor_dims. Use [0] here to match.
                        root_spec = root_spec.input_dims[0]
                    else:
                        root_spec = root_spec.input_dim
                root = root_spec if isinstance(root_spec, InputDim) else None
                if root is not None and root.input_dim in input_to_output_tensor_dims:
                    input_to_output_tensor_dims[root.input_dim].append(output_dim)

        input_tgt_placements: list[Placement] = []
        for mesh_dim, p in enumerate(self.input_src_placements):
            if (
                isinstance(p, Shard | _StridedShard)
                and not self.shard_allowed[p.dim][mesh_dim]
            ):
                if (
                    self.strict_view
                    and (p.dim, mesh_dim) not in self.strict_replicate_fallback
                ):
                    raise RuntimeError(
                        f"This operation would remove or reshape sharded "
                        f"dimension {p.dim}, which requires redistribution. "
                        f"Please redistribute the input first."
                    )
                input_tgt_placements.append(Replicate())
            else:
                input_tgt_placements.append(p)
        return input_tgt_placements, input_to_output_tensor_dims

    def rewrite_output_placements(
        self,
        input_tgt_placements: Sequence[Placement],
        input_to_output_tensor_dims: dict[int, list[int]],
    ) -> list[Placement]:
        """Phase 2: consume analyze() outputs, return final output placements."""
        # (input_dim, output_dim) pairs claimed by earlier mesh dims
        # (via _rewrite_strided_shard), to avoid double-assignment.
        strided_shard_claimed_dims: set[ClaimedDim] = set()
        # Starts as global_input_shape; each mesh dim divides its sharded dim.
        local_tensor_shapes: list[int] = list(self.global_input_shape)

        output_placements: list[Placement] = []
        # Process mesh dims in order; _rewrite_*_shard relies on this for
        # truncating division safety in local_tensor_shapes.
        for mesh_dim, p in enumerate(input_tgt_placements):
            if isinstance(p, Shard):
                placement, local_tensor_shapes = self._rewrite_plain_shard(
                    p,
                    mesh_dim,
                    input_tgt_placements,
                    strided_shard_claimed_dims,
                    local_tensor_shapes,
                    input_to_output_tensor_dims,
                )
                output_placements.append(placement)
            elif isinstance(p, _StridedShard):
                placement, local_tensor_shapes = self._rewrite_strided_shard(
                    p,
                    mesh_dim,
                    input_tgt_placements,
                    strided_shard_claimed_dims,
                    local_tensor_shapes,
                    input_to_output_tensor_dims,
                )
                output_placements.append(placement)
            else:
                output_placements.append(p)
        return output_placements

    # ------------------------------------------------------------------
    # Analysis phase helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _input_dims_in_rule(rule: DimMap) -> set[int]:
        """Walk the DimMap rule tree and return all input dim indices that appear in it."""
        seen: set[int] = set()

        def _walk(cmd: DimSpec) -> None:
            if isinstance(cmd, InputDim):
                seen.add(cmd.input_dim)
            for inp in cmd.inputs():
                _walk(inp)

        for cmd in rule:
            _walk(cmd)
        return seen

    def _find_plain_shard(
        self, input_dim: InputDim
    ) -> tuple[int | None, Shard | _StridedShard | None]:
        """Find the mesh dim with a plain Shard on ``input_dim``.

        Only matches Shard, not _StridedShard.  Used by both _analyze_flatten
        and _analyze_split.  _find_shard_for_split is the counterpart that
        also matches _StridedShard with split_factor validation.
        """
        for mesh_dim, placement in enumerate(self.input_src_placements):
            if isinstance(placement, Shard) and placement.dim == input_dim.input_dim:
                return mesh_dim, placement
        return None, None

    def _find_shard_for_split(
        self,
        current_dim: int,
        cmd: Split,
        placements: Sequence[Placement],
    ) -> tuple[int | None, Shard | _StridedShard | None]:
        """Find the mesh dim and placement for an input dim in Split ops.

        Matches both Shard and _StridedShard:
        - Shard: plain unflatten, e.g. [6] Shard(0) → [2, 3].
        - _StridedShard: unflatten after a prior flatten that produced
          _StridedShard, e.g. [2,3,4] Shard(1) → flatten → [6,4]
          _StridedShard(0,sf=2) → unflatten → [2,3,4].  Validates that
          the split_factor matches the expected value for this split_id.
        """
        for mesh_dim, placement in enumerate(placements):
            if not isinstance(placement, Shard | _StridedShard):
                continue
            if placement.dim != current_dim:
                continue
            if mesh_dim in self.matched_strided_mesh_dims:
                continue

            if isinstance(placement, _StridedShard):
                expected_sf = self._expected_split_factor(
                    cmd, current_dim, mesh_dim, placements
                )
                if _split_factor_matches(expected_sf, placement.split_factor):
                    return mesh_dim, placement
            else:
                return mesh_dim, placement
        return None, None

    def _analyze_flatten(self, cmd: Flatten) -> list[InputDim]:
        """Fill self.shard_allowed for Flatten; return sharded input dims."""
        from torch.fx.experimental.symbolic_shapes import guard_or_true

        sharded_dims: list[InputDim] = []
        num_input_dims = len(cmd.input_dims)
        for i, dim in enumerate(cmd.input_dims):
            if not isinstance(dim, InputDim):
                raise AssertionError(f"Expected InputDim, got {type(dim)}")
            shard_mesh_dim, shard_placement = self._find_plain_shard(dim)
            if shard_mesh_dim is None or shard_placement is None:
                continue  # default from analyze() already covers this
            tensor_dim_size = self.global_input_shape[shard_placement.dim]
            mesh_dim_size = self.mesh_sizes[shard_mesh_dim]
            can_shard_dim = True
            if self.strict_view:
                is_last_input_dim = i == num_input_dims - 1
                if not is_last_input_dim and guard_or_true(
                    tensor_dim_size % mesh_dim_size != 0
                ):
                    raise RuntimeError(
                        f"Cannot flatten unevenly sharded tensor: "
                        f"dimension {dim.input_dim} (size {tensor_dim_size}) "
                        f"is not evenly divisible by mesh dimension "
                        f"{shard_mesh_dim} (size {mesh_dim_size}). "
                        f"Please redistribute the tensor before this operation."
                    )
                sharded_dims.append(dim)
            else:
                # TODO: non-strict (reshape) should allow can_shard_dim = True
                # for non-first flatten dims, since strict_view already does.
                # Currently forces redistribution because the rewrite phase
                # wasn't originally implemented for this case.
                if i == 0:
                    sharded_dims.append(dim)
                    if guard_or_true(tensor_dim_size % mesh_dim_size != 0):
                        can_shard_dim = False
                else:
                    can_shard_dim = False
            self.shard_allowed[dim.input_dim] = [can_shard_dim] * self.mesh_ndim

        if len(sharded_dims) > 0:
            return sharded_dims
        # No sharded dims: e.g. Flatten([InputDim(0), InputDim(1)]) where
        # neither dim is sharded.  Return the first input dim so that
        # input_to_output_tensor_dims is populated for identity rewrites.
        if not isinstance(cmd.input_dims[0], InputDim):
            raise AssertionError(f"Expected InputDim, got {type(cmd.input_dims[0])}")
        return [cmd.input_dims[0]]

    def _analyze_split(self, cmd: Split) -> list[InputDim]:
        """Fill self.shard_allowed for Split; return shardable input dims."""
        from torch.fx.experimental.symbolic_shapes import guard_or_false, guard_or_true

        in_dims = self._analyze_dim(cmd.input_dim)
        if len(in_dims) == 0:
            return []
        in_dim = in_dims[0]
        if isinstance(cmd.input_dim, Flatten):
            # A plain Shard on a non-first flattened input dim would need an
            # intermediate _StridedShard before this Split. The combined
            # Flatten+Split rewrite does not preserve that placement exactly,
            # so redistribute that mesh dim instead of raising later.
            for flat_dim in cmd.input_dim.input_dims[1:]:
                if not isinstance(flat_dim, InputDim):
                    raise AssertionError(f"Expected InputDim, got {type(flat_dim)}")
                for mesh_dim, placement in enumerate(self.input_src_placements):
                    if (
                        isinstance(placement, Shard)
                        and placement.dim == flat_dim.input_dim
                    ):
                        self.shard_allowed[flat_dim.input_dim][mesh_dim] = False
                        self.strict_replicate_fallback.add(
                            (flat_dim.input_dim, mesh_dim)
                        )
        out_size = cmd.group_shape[cmd.split_id]
        shard_mesh_dim, input_src_placement = self._find_shard_for_split(
            in_dim.input_dim, cmd, self.input_src_placements
        )
        # split_id == 0 sets the base shard_allowed for this input dim.
        # Later split_ids (processed in subsequent rule iterations) refine
        # individual mesh_dim entries via the _StridedShard branch below.
        if cmd.split_id == 0:
            self.shard_allowed[in_dim.input_dim] = [
                guard_or_false(out_size % mesh_dim_size == 0)
                for mesh_dim_size in self.mesh_sizes
            ]
            plain_mesh_dim, _ = self._find_plain_shard(in_dim)
            # Non-strict silently redistributes via shard_allowed=False above;
            # strict raises so the user knows to redistribute before view().
            if self.strict_view and plain_mesh_dim is not None:
                if not self.shard_allowed[in_dim.input_dim][plain_mesh_dim]:
                    raise RuntimeError(
                        f"Cannot unflatten unevenly sharded tensor: "
                        f"output dimension {cmd.split_id} (size {out_size}) "
                        f"is not evenly divisible by mesh dimension "
                        f"{plain_mesh_dim} (size {self.mesh_sizes[plain_mesh_dim]}). "
                        f"Please redistribute the tensor before this operation."
                    )
        if shard_mesh_dim is not None and isinstance(
            input_src_placement, _StridedShard
        ):
            # The last split dim doesn't require even divisibility because
            # its local size is inferred: local_last = local_flat / product
            # of earlier dims, and DTensor handles uneven local sizes.
            # Non-last dims must be evenly divisible because they appear as
            # fixed sizes in the local reshape — uneven division would make
            # the stride pattern inconsistent across devices.
            # E.g. [12] → [3, 4], _StridedShard targeting dim 1 (last),
            # mesh=3: 4%3≠0, but local shapes [3,2],[3,1],[3,1] are valid.
            is_last_split_dim = cmd.split_id == len(cmd.group_shape) - 1
            if (
                self.strict_view
                and not is_last_split_dim
                and guard_or_true(out_size % self.mesh_sizes[shard_mesh_dim] != 0)
            ):
                raise RuntimeError(
                    f"Cannot unflatten unevenly sharded tensor: "
                    f"output dimension {cmd.split_id} (size {out_size}) "
                    f"is not evenly divisible by mesh dimension {shard_mesh_dim} "
                    f"(size {self.mesh_sizes[shard_mesh_dim]}). "
                    f"Please redistribute the tensor before this operation."
                )
            # Prevents _find_shard_for_split from matching this mesh dim
            # again for a later split_id of the same Split group.
            self.matched_strided_mesh_dims.add(shard_mesh_dim)
            if in_dim.input_dim in self.shard_allowed:
                self.shard_allowed[in_dim.input_dim][shard_mesh_dim] = (
                    guard_or_false(out_size % self.mesh_sizes[shard_mesh_dim] == 0)
                    or is_last_split_dim
                )
        # Only split_id==0 returns the input dim for input_to_output_tensor_dims.
        # Later split_ids refine shard_allowed above but return [] — their
        # output dims are linked via the root-input-dim chase in analyze().
        return [in_dim] if cmd.split_id == 0 else []

    def _analyze_dim(self, cmd: DimSpec) -> list[InputDim]:
        """Dispatch one DimSpec: update self.shard_allowed, return input dim(s) to shard on."""
        if isinstance(cmd, InputDim):
            return [cmd]
        elif isinstance(cmd, Flatten):
            return self._analyze_flatten(cmd)
        elif isinstance(cmd, Split):
            return self._analyze_split(cmd)
        elif isinstance(cmd, Repeat):
            in_dims = self._analyze_dim(cmd.input_dim)
            for d in in_dims:
                self.shard_allowed[d.input_dim] = [False] * self.mesh_ndim
            return []
        else:
            return []

    # ------------------------------------------------------------------
    # Rewrite phase helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_last_shard_in_flatten_range(
        mesh_dim: int,
        placements: Sequence[Placement],
        flatten_start: int,
        flatten_end: int,
    ) -> bool:
        """Check if no later mesh dim shards a dim within the flatten range at or above this one.

        Uneven sharding on dim d breaks stride computation for all earlier dims
        that flatten together with d. Only dims within [flatten_start, flatten_end)
        matter; shards on dims outside the flatten range are independent.

        Requires: placements[mesh_dim] must be Shard or _StridedShard.
        """
        p = placements[mesh_dim]
        if not isinstance(p, (Shard, _StridedShard)):
            raise AssertionError(
                f"Expected Shard or _StridedShard at mesh_dim {mesh_dim}, got {type(p)}"
            )
        tensor_dim = p.dim
        return not any(
            isinstance(other_p, (Shard, _StridedShard))
            and flatten_start <= other_p.dim < flatten_end
            and other_p.dim >= tensor_dim
            for other_p in placements[mesh_dim + 1 :]
        )

    def _expected_split_factor(
        self,
        cmd: Split,
        sharded_dim: int,
        mesh_dim: int,
        placements: Sequence[Placement],
    ) -> int | torch.SymInt | None:
        """Compute the residual split factor for ``cmd`` after earlier mesh dims.

        Starts from ``math.prod(cmd.group_shape[:cmd.split_id])`` and divides
        out each earlier mesh dim that shards the same input dim.  Returns
        ``None`` if any earlier mesh size doesn't divide evenly.
        """
        sf = math.prod(cmd.group_shape[: cmd.split_id])
        for m in range(mesh_dim):
            other_p = placements[m]
            if (
                isinstance(other_p, (_StridedShard, Shard))
                and other_p.dim == sharded_dim
            ):
                if sf % self.mesh_sizes[m] != 0:
                    return None
                sf //= self.mesh_sizes[m]
        return sf

    def _find_keep_ss_dim(
        self,
        tgt_shard_dims: list[int],
        p: _StridedShard,
        mesh_dim: int,
    ) -> int | None:
        """Find an output dim where SS stays as SS.

        Returns the first output dim whose Split can accommodate the combined
        sharding (mesh_size * split_factor), or ``None`` if no dim fits.
        """
        total_shard = cast(Any, self.mesh_sizes[mesh_dim]) * p.split_factor
        input_dim_size = cast(Any, self.global_input_shape[p.dim])
        if input_dim_size % total_shard != 0:
            return None
        shard_size = input_dim_size // total_shard
        for candidate_dim in tgt_shard_dims:
            cmd = self.rule[candidate_dim]
            if isinstance(cmd, Split):
                inner_size = math.prod(cmd.group_shape[cmd.split_id + 1 :])
                # When a Split wraps a Flatten, the per-shard chunk covers
                # the sharded dim plus trailing dims flattened together.
                trailing_size = 1
                if isinstance(cmd.input_dim, Flatten):
                    found = False
                    for flat_dim in cmd.input_dim.input_dims:
                        if not isinstance(flat_dim, InputDim):
                            raise AssertionError(
                                f"Expected InputDim, got {type(flat_dim)}"
                            )
                        if flat_dim.input_dim == p.dim:
                            found = True
                        elif found:
                            trailing_size *= self.global_input_shape[flat_dim.input_dim]
                flattened_shard_size = shard_size * trailing_size
                if (
                    flattened_shard_size >= inner_size
                    and flattened_shard_size % inner_size == 0
                ):
                    return candidate_dim
        return None

    def _rewrite_plain_shard(
        self,
        p: Shard,
        mesh_dim: int,
        placements: Sequence[Placement],
        strided_shard_claimed_dims: set[ClaimedDim],
        local_tensor_shapes: list[int],
        input_to_output_tensor_dims: dict[int, list[int]],
    ) -> tuple[Placement, list[int]]:
        """Given a plain Shard(dim=X) input placement on a specific mesh dim,
        determine what output placement it maps to after the view op.

        For identity and unflatten, produces Shard on the output dim.  For
        flatten, Shard on the first flattened dim stays Shard, while Shard on
        a non-first dim produces _StridedShard (consumed later by
        _rewrite_strided_shard).

        Returns the output placement and a new local_tensor_shapes with this
        mesh dim's division applied.
        """
        # Output dims that input dim p.dim maps to, filtering out any
        # already claimed by _StridedShard rewriting on earlier mesh dims.
        tgt_shard_dims = [
            d
            for d in input_to_output_tensor_dims[p.dim]
            if ClaimedDim(p.dim, d) not in strided_shard_claimed_dims
        ]
        if len(tgt_shard_dims) == 0:
            raise AssertionError(
                f"No output dim available for Shard(dim={p.dim}) on mesh dim "
                f"{mesh_dim}. All output dims already claimed by earlier mesh dims."
            )
        if len(tgt_shard_dims) == 1:
            tgt_shard_dim = tgt_shard_dims[0]
        else:
            # Unflatten: one input dim maps to multiple output dims
            # (e.g. (24,) → (2, 3, 4) gives 3 splits). Plain Shard
            # always targets the split_id=0 output dim.
            tgt_shard_dim = next(
                (
                    d
                    for d in tgt_shard_dims
                    if isinstance(self.rule[d], Split)
                    and cast(Split, self.rule[d]).split_id == 0
                ),
                None,
            )
            if tgt_shard_dim is None:
                raise AssertionError(
                    f"No Split(split_id=0) found among unclaimed output dims "
                    f"{tgt_shard_dims} for Shard(dim={p.dim}) on mesh dim {mesh_dim}."
                )
        cmd = self.rule[tgt_shard_dim]
        if isinstance(cmd, Split) and isinstance(cmd.input_dim, Flatten):
            first_dim = cmd.input_dim.input_dims[0]
            if isinstance(first_dim, InputDim) and p.dim != first_dim.input_dim:
                raise AssertionError(
                    f"Shard(dim={p.dim}) should have been demoted before "
                    f"Split(Flatten(...), {cmd.group_shape}) rewrite."
                )
        if isinstance(cmd, (Split, InputDim)):
            # Split/InputDim: 1:1 dim mapping, sharding transfers directly.
            # Flatten needs stride computation below (multiple dims merge).
            new_shapes = list(local_tensor_shapes)
            new_shapes[p.dim] //= self.mesh_sizes[mesh_dim]
            return Shard(tgt_shard_dim), new_shapes
        if not isinstance(cmd, Flatten):
            raise AssertionError(f"Expected Flatten, got {type(cmd)}")
        first_dim = cmd.input_dims[0]
        last_dim = cmd.input_dims[-1]
        if not isinstance(first_dim, InputDim):
            raise AssertionError(f"Expected InputDim, got {type(first_dim)}")
        if not isinstance(last_dim, InputDim):
            raise AssertionError(f"Expected InputDim, got {type(last_dim)}")
        input_start_idx = first_dim.input_dim
        if p.dim == input_start_idx:
            output_placement: Placement = Shard(tgt_shard_dim)
        else:
            split_factor = math.prod(local_tensor_shapes[input_start_idx : p.dim])
            if isinstance(split_factor, torch.SymInt):
                if _explicit_unbacked_hint(split_factor) is None:
                    raise RuntimeError(
                        "Cannot propagate strict view sharding through a symbolic "
                        "_StridedShard split_factor without an explicit unbacked "
                        f"hint override: split_factor={split_factor}, "
                        f"input_shape={self.global_input_shape}, rule={self.rule}."
                    )
            output_placement = _StridedShard(tgt_shard_dim, split_factor=split_factor)
        # Uneven sharding on a non-last flatten dim breaks _StridedShard:
        # split_factor (number of groups) must be the same on all devices,
        # but uneven division of a non-last dim makes group count vary.
        # E.g. [3,4]→[12] Shard(0) mesh=2: device 0 has 2 groups of 4,
        # device 1 has 1 group of 4 — no consistent split_factor.
        # The last dim is exempt: only group *size* varies, not count.
        flatten_end = last_dim.input_dim + 1
        if local_tensor_shapes[p.dim] % self.mesh_sizes[
            mesh_dim
        ] != 0 and not self._is_last_shard_in_flatten_range(
            mesh_dim, placements, input_start_idx, flatten_end
        ):
            raise RuntimeError(
                f"Cannot shard unevenly distributed tensor: "
                f"dimension {p.dim} (size {local_tensor_shapes[p.dim]}) "
                f"is not evenly divisible by mesh dimension "
                f"{mesh_dim} (size {self.mesh_sizes[mesh_dim]}). "
                f"Please redistribute the tensor before this operation."
            )
        new_shapes = list(local_tensor_shapes)
        new_shapes[p.dim] //= self.mesh_sizes[mesh_dim]
        return output_placement, new_shapes

    def _rewrite_strided_shard(
        self,
        p: _StridedShard,
        mesh_dim: int,
        placements: Sequence[Placement],
        strided_shard_claimed_dims: set[ClaimedDim],
        local_tensor_shapes: list[int],
        input_to_output_tensor_dims: dict[int, list[int]],
    ) -> tuple[Placement, list[int]]:
        """Rewrite _StridedShard placement to target the correct output dim.

        _StridedShard inputs arise from a prior flatten on a non-first dim
        (produced by _rewrite_plain_shard above).  The interesting case is
        unflatten (Split rule): the split_factor may resolve to contiguous
        sharding (producing Shard) or stay as _StridedShard.  For
        identity/flatten rules, falls through to the fallback and keeps the
        placement as-is.

        Returns the output placement and a new local_tensor_shapes with this
        mesh dim's division applied.
        """
        tgt_shard_dims = [
            d
            for d in input_to_output_tensor_dims[p.dim]
            if ClaimedDim(p.dim, d) not in strided_shard_claimed_dims
        ]
        # Phase 1: resolve SS → Shard.  If an output dim's Split has a
        # group_shape prefix matching the split_factor, the strided pattern
        # is fully captured by the Split, so SS simplifies to Shard.
        # E.g. unflatten (6, 4) → (2, 3, 4) with SS(0, sf=2) on mesh (3):
        # sf=2 means 2 groups of contiguous data in dim 0.  Split into
        # (2, 3, 4) gives group_shape=(2, 3); prod(group_shape[:1])=2==sf,
        # so the strided pattern lands exactly on output dim 1 → Shard(1).
        for candidate_dim in tgt_shard_dims:
            cmd = self.rule[candidate_dim]
            if isinstance(cmd, Split):
                expected_sf = self._expected_split_factor(
                    cmd, p.dim, mesh_dim, placements
                )
                if not _split_factor_matches(expected_sf, p.split_factor):
                    continue
                strided_shard_claimed_dims.add(ClaimedDim(p.dim, candidate_dim))
                new_shapes = list(local_tensor_shapes)
                new_shapes[p.dim] //= self.mesh_sizes[mesh_dim]
                return Shard(candidate_dim), new_shapes

        # Phase 2: keep SS as SS.  Phase 1 is tried first because we prefer
        # resolving to the simpler Shard when possible.
        tgt_shard_dim = self._find_keep_ss_dim(tgt_shard_dims, p, mesh_dim)

        if tgt_shard_dim is None:
            if self.strict_view and any(
                isinstance(self.rule[d], Split) for d in tgt_shard_dims
            ):
                raise RuntimeError(
                    f"Cannot unflatten tensor with _StridedShard placement: "
                    f"split_factor={p.split_factor} does not match any output "
                    f"dimension. This typically means the _StridedShard placement "
                    f"was constructed with a split_factor that is incompatible "
                    f"with the unflatten shape. Please redistribute the tensor "
                    f"before this operation."
                )
            if len(tgt_shard_dims) == 0:
                raise AssertionError(
                    f"No unclaimed output dims for _StridedShard(dim={p.dim}) "
                    f"on mesh dim {mesh_dim}."
                )
            # Fallback for identity/flatten: tgt_shard_dims has exactly one
            # element, so [0] is correct.  For Split rules this is unreachable
            # in practice — the analysis phase rejects mismatched split_factors
            # via shard_allowed, forcing redistribution before we get here.
            tgt_shard_dim = tgt_shard_dims[0]
        new_shapes = list(local_tensor_shapes)
        new_shapes[p.dim] //= self.mesh_sizes[mesh_dim]
        return _StridedShard(tgt_shard_dim, split_factor=p.split_factor), new_shapes


def register_op_strategy_map(
    aten_op_overload: torch._ops.OpOverload,
    local_op_name: Callable[..., torch.Tensor],
    schema_info: RuntimeSchemaInfo | None = None,
    strict_view: bool = False,
) -> None:
    """
    Helper that registers strategies for view-like operators that follow a pattern:
      (1) define the way input dims are split/combined to form output dims (dim_maps)
      (2) register a strategy for the op schema that uses the dim_map as a sharding prop rule

    strict_view: if True, we will error out if the view-operation would require resharding the input.
       Currently, this should be set to 'true' for any "view" ops.
       We could diverge behavior for "reshape" ops which could perform a redistribute implicitly.
    """
    dim_map: Callable[..., DimMap] = dim_maps[local_op_name]

    @register_op_strategy(aten_op_overload, schema_info=schema_info)
    def reshape_strategy(op_schema: OpSchema) -> StrategyType:
        rules = dim_map(*op_schema.args_schema, **op_schema.kwargs_schema)
        input_strategy = cast(OpStrategy, op_schema.args_schema[0])
        mesh = op_schema.get_mesh_from_args(validate=False)

        global_in_shape = input_strategy.shape
        if global_in_shape is None:
            raise AssertionError("Shape required.")

        output_strategy = OpStrategy([])
        for input_placement_strategy in input_strategy.strategies:
            input_src_spec = input_placement_strategy.output_spec

            input_tgt_placements, output_placements = propagate_shape_and_sharding(
                input_src_spec.placements,
                tuple(global_in_shape),
                rules,
                mesh.shape,
                strict_view,
            )

            # TODO: optimize this. we shouldn't simply blindly replicate
            #       unshardable dims ...
            # FIXME: this can be wrong for situations where we have
            #        [Shard(0), Shard(0)]
            input_tgt_spec = DTensorSpec(
                placements=tuple(input_tgt_placements),
                mesh=mesh,
                tensor_meta=input_src_spec.tensor_meta,
                use_strided_shard_as_shard_order=False,
            )
            redistribute_costs: list[list[float]] = [
                generate_redistribute_costs(input_strategy, input_tgt_spec)
            ]

            output_spec = DTensorSpec(
                mesh=mesh,
                placements=tuple(output_placements),
                use_strided_shard_as_shard_order=False,
            )
            output_strategy.strategies.append(
                OpSpec(
                    output_specs=output_spec,
                    input_specs=(input_tgt_spec,),
                    redistribute_cost=redistribute_costs,
                )
            )

        return output_strategy


register_op_strategy_map(aten.squeeze.default, torch.squeeze, strict_view=True)
register_op_strategy_map(aten.squeeze_.default, torch.squeeze, strict_view=True)
register_op_strategy_map(
    aten.squeeze_.dim,
    torch.squeeze,
    schema_info=RuntimeSchemaInfo(1),
    strict_view=True,
)
register_op_strategy_map(
    aten.squeeze.dim,
    torch.squeeze,
    schema_info=RuntimeSchemaInfo(1),
    strict_view=True,
)
register_op_strategy_map(
    aten.squeeze.dims,
    torch.squeeze,
    schema_info=RuntimeSchemaInfo(1),
    strict_view=True,
)
register_op_strategy_map(
    aten.squeeze_.dims,
    torch.squeeze,
    schema_info=RuntimeSchemaInfo(1),
    strict_view=True,
)
register_op_strategy_map(
    aten.view.default,
    Tensor.view,
    schema_info=RuntimeSchemaInfo(1),
    strict_view=True,
)
register_op_strategy_map(
    aten.view_copy.default,
    Tensor.view,
    schema_info=RuntimeSchemaInfo(1),
)
register_op_strategy_map(
    aten.reshape.default, torch.reshape, schema_info=RuntimeSchemaInfo(1)
)
register_op_strategy_map(
    aten._unsafe_view.default,
    Tensor.view,
    schema_info=RuntimeSchemaInfo(1),
    strict_view=True,
)
register_op_strategy_map(
    aten.unsqueeze.default, torch.unsqueeze, schema_info=RuntimeSchemaInfo(1)
)
register_op_strategy_map(
    aten.expand.default, Tensor.expand, schema_info=RuntimeSchemaInfo(1)
)
register_op_strategy_map(
    aten.expand_copy.default, Tensor.expand, schema_info=RuntimeSchemaInfo(1)
)
register_op_strategy_map(
    aten.permute.default, torch.permute, schema_info=RuntimeSchemaInfo(1)
)
register_op_strategy_map(
    aten.repeat.default, Tensor.repeat, schema_info=RuntimeSchemaInfo(1)
)
register_op_strategy_map(
    aten.transpose.int, torch.transpose, schema_info=RuntimeSchemaInfo(1)
)


@register_single_dim_strategy(aten.view_as_complex.default)
def view_as_complex_single_dim_strategy(op, args_schema, kwargs_schema):
    # view_as_complex: float [..., 2] -> complex [...]
    # Dims 0..ndim-2 map 1:1; last dim (real/imag pair) is consumed.
    # P(max)/P(min) invalid: complex numbers have no total ordering.
    input_meta = args_schema[0]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    ndim = len(input_meta.shape)
    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for d in range(ndim - 1):
        strategies.append([_ShardingPlaceholder(d), _ShardingPlaceholder(d)])
    strategies.append([Partial("sum"), Partial("sum")])
    strategies.append([Partial("avg"), Partial("avg")])
    return strategies


register_op_strategy_map(aten.view_as_real.default, torch.view_as_real)
<|endoftext|># mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
import functools
import itertools
import operator
from collections.abc import Callable, Iterable, Sequence
from typing import Any, cast, TypeAlias, TypeVar

import torch
from torch._prims_common import DimsSequenceType, DimsType
from torch.distributed.tensor._api import DTensor
from torch.distributed.tensor._collective_utils import redistribute_cost
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import (
    OpSchema,
    OpSpec,
    OpStrategy,
    OutputSharding,
    PlacementList,
    RuntimeSchemaInfo,
    StrategyType,
)
from torch.distributed.tensor.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import (
    _is_shard_like,
    _StridedShard,
    Partial,
    Placement,
    Replicate,
    Shard,
)


def _get_registration_wrapper(
    registration_fn,
    op: torch._ops.OpOverload | list[torch._ops.OpOverload],
    schema_info: RuntimeSchemaInfo | None,
    arg_names_that_require_specializing_cache_strategy: list[str] | None,
):
    def wrapper(impl):
        overloads = op if isinstance(op, list) else [op]
        for overload in overloads:
            curr_schema_info = None
            if (
                schema_info is None
                and arg_names_that_require_specializing_cache_strategy is not None
            ):
                specialized_args = [
                    a.name
                    for a in overload._schema.arguments
                    if a.name in arg_names_that_require_specializing_cache_strategy
                ]
                if any(specialized_args):
                    curr_schema_info = RuntimeSchemaInfo(
                        static_kwargkey=specialized_args
                    )
            else:
                curr_schema_info = schema_info
            registration_fn(overload, impl, curr_schema_info)
        return impl

    return wrapper


# convenient wrapper to register sharding propagation rules
def register_prop_rule(
    op: torch._ops.OpOverload | list[torch._ops.OpOverload],
    schema_info: RuntimeSchemaInfo | None = None,
) -> Callable[
    [Callable[[OpSchema], OutputSharding]], Callable[[OpSchema], OutputSharding]
]:
    return _get_registration_wrapper(
        DTensor._op_dispatcher.sharding_propagator.register_sharding_prop_rule,
        op,
        schema_info,
        arg_names_that_require_specializing_cache_strategy=None,
    )


# Note:
# using TypeVar here allows the registration decorator to preserve the specific type info of the wrapped strategy,
# while hardcoding the typing on the wrapper (e.g. Callable[[OpSchema], StrategyType]) would mean mypy would treat
# the return value of the wrapped strategy as always being a `StrategyType` even if it were a derived class like
# MyStrategyType(StrategyType).
_OpSchemaT = TypeVar("_OpSchemaT", bound=OpSchema)
_StrategyTypeT = TypeVar("_StrategyTypeT", bound=StrategyType)
_ShardingStrategyFunc: TypeAlias = Callable[[_OpSchemaT], _StrategyTypeT]


def register_op_strategy(
    op: torch._ops.OpOverload | list[torch._ops.OpOverload],
    schema_info: RuntimeSchemaInfo | None = None,
) -> Callable[[_ShardingStrategyFunc], _ShardingStrategyFunc]:
    # For every ATen op that accepts any args in this list,
    # the arg itself can impact the strides (and potentially the sharding strategy)
    # of the output tensor.
    # thus, we will detect ATen schemas with any of these args and ensure
    # that they get specialized here.
    arg_names_that_require_specializing_cache_strategy = [
        "memory_format",
    ]
    return _get_registration_wrapper(
        DTensor._op_dispatcher.sharding_propagator.register_op_strategy,
        op,
        schema_info,
        arg_names_that_require_specializing_cache_strategy,
    )


def replicate_op_strategy(op_schema: OpSchema) -> StrategyType:
    """
    Fallback strategy all use Replication()
    """
    args_strategy = op_schema.args_strategy
    kwargs_strategy = op_schema.kwargs_strategy
    inputs_strategy = args_strategy + kwargs_strategy

    output_type = [str(ret.type) for ret in op_schema.op._schema.returns]
    output_len = output_type.count("Tensor")
    # TODO(zpcore): Confirm if view op can be handle properly or not. Prevent
    # handling view ops until confirmed.
    if op_schema.op.is_view:
        raise RuntimeError(
            "fallback strategy is unable to handle view ops until confirmed"
        )
    if "List[Tensor]" in output_type:
        raise RuntimeError(
            "fallback strategy is unable to handle ops with List[Tensor] output "
            "because size of the list may depend on the op's input value"
        )

    mesh = inputs_strategy[0].mesh

    dim_sharding: PlacementList = [Replicate()] * (output_len + len(inputs_strategy))
    single_dim_placement = [dim_sharding]
    return expand_to_full_mesh_op_strategy(
        mesh, op_schema, single_dim_placement, input_index=output_len
    )


def as_list(
    x: list[object] | object,
    # pyre-fixme[11]: Annotation `immutable_list` is not defined as a type.
) -> list[object] | torch.fx.immutable_collections.immutable_list:  # type: ignore[valid-type]
    # During tracing, `aten.sum.dim_IntList` uses `immutable_list` for its args,
    # which is an object but treated as a list by the tracer. Therefore, keep
    # `immutable_list` intact here as well.
    if type(x) is list or isinstance(x, torch.fx.immutable_collections.immutable_list):
        return x
    else:
        return [x]


def normalize_dim(dim: int, ndim: int) -> int:
    return dim if dim >= 0 else dim + ndim


def normalize_dims(dims: DimsType, ndim: int) -> DimsSequenceType:
    """Normalize a dim or a sequence of dims, so that they are all positive."""
    if isinstance(dims, int):
        dims = (normalize_dim(dims, ndim),)
    elif isinstance(dims, list):
        dims = [normalize_dim(dim, ndim) for dim in dims]
    elif isinstance(dims, tuple):
        dims = tuple(normalize_dim(dim, ndim) for dim in dims)
    return dims


def prod(xs: Iterable[int]) -> int:
    return functools.reduce(operator.mul, xs, 1)


def is_tensor_shardable(
    shape: Sequence[int],
    spec: DTensorSpec,
    allow_unbacked_sharding: bool | None = None,
) -> bool:
    """
    Check if the shape is shardable according to the spec.

    This function handles both `Shard` and `_StridedShard` placements:
    - For `Shard`: checks if the tensor dimension size >= number of shards
    - For `_StridedShard`: additionally checks if the dimension is shardable after
      splitting with the placement's `split_factor`

    allow_unbacked_sharding: determines the fallback value if unbacked shapes are involved,
    and the queried shape properties are not statically known.

    e.g. when asking if u0 is shardable on num_shards, and u0 has generic bounds [0, inf],
    the behavior of allow_unbacked_sharding is:

        None: will data-dependent error
        True: assumes shardability; we return True, allowing zero-size shards at runtime when u0 < num_shards.
        False: returns False, and lower-bounding u0, e.g. torch._check(u0 >= num_shards), is needed to enable sharding.
    """
    from torch.fx.experimental.symbolic_shapes import (
        free_unbacked_symbols,
        guard_or_false,
        guard_or_true,
        optimization_hint,
    )

    if allow_unbacked_sharding not in [None, True, False]:
        raise AssertionError
    guard_fn = {
        None: bool,
        True: guard_or_false,
        False: guard_or_true,
    }[allow_unbacked_sharding]

    def hint_proves_at_least(size, lower_bound: int) -> bool:
        if guard_or_false(size >= lower_bound):
            return True
        if not (
            isinstance(size, torch.SymInt)
            and (shape_env := getattr(size.node, "shape_env", None)) is not None
            and (unbacked_symbols := free_unbacked_symbols(size.node.expr))
            and unbacked_symbols.issubset(shape_env.var_to_hint_override.keys())
            and optimization_hint(size, fallback=None) >= lower_bound
        ):
            return False

        # Shardability is a correctness property, not just an optimization.
        # A user-provided unbacked hint can select the shardable branch, but
        # the runtime extent must still be checked before the strategy is used.
        torch._check(size >= lower_bound)
        return True

    # number of shards in each tensor dimension
    num_shards = [1] * len(shape)
    for i, placement in enumerate(spec.placements):
        if _is_shard_like(placement):
            shard_dim = placement.dim
            if shard_dim >= len(shape):
                return False
            num_shards[shard_dim] *= spec.mesh.size(i)
            if isinstance(placement, _StridedShard):
                # make sure tensor dim `shard_dim` is shardable after splitting
                # with split_factor
                lower_bound = cast(Any, num_shards[shard_dim]) * placement.split_factor
                if not hint_proves_at_least(shape[shard_dim], lower_bound) and guard_fn(
                    shape[shard_dim] < lower_bound
                ):
                    return False
            else:
                lower_bound = num_shards[shard_dim]
                if not hint_proves_at_least(shape[shard_dim], lower_bound) and guard_fn(
                    shape[shard_dim] < lower_bound
                ):
                    return False

    return True


def is_tensor_evenly_shardable(shape: Sequence[int], spec: DTensorSpec) -> bool:
    """Check if the shape is evenly shardable according to the spec."""
    # number of shards in each tensor dimension
    num_shards = [1] * len(shape)
    for i, placement in enumerate(spec.placements):
        if _is_shard_like(placement):
            shard_dim = placement.dim
            if shard_dim >= len(shape):
                return False
            num_shards[shard_dim] *= spec.mesh.size(i)
            if isinstance(placement, _StridedShard):
                if (
                    cast(Any, shape[shard_dim])
                    % (placement.split_factor * num_shards[shard_dim])
                    != 0
                ):
                    return False
            else:
                if shape[shard_dim] % num_shards[shard_dim] != 0:
                    return False
    return True


def is_tensor_evenly_shardable_on_dim(
    shape: Sequence[int], spec: DTensorSpec, dim: int
) -> bool:
    """Check if the shape is evenly shardable according to the spec on dim."""
    dim = normalize_dim(dim, len(shape))

    num_shards = 1
    for i, placement in enumerate(spec.placements):
        if _is_shard_like(placement) and placement.dim == dim:
            num_shards *= spec.mesh.size(i)
            if isinstance(placement, _StridedShard):
                # _StridedShard._split_tensor first chunks into split_factor
                # groups, then into num_shards within each group, so the dim
                # must be divisible by the product of both.  This is stricter
                # than the final num_shards check and implies it.  Note:
                # num_shards already includes spec.mesh.size(i) from this
                # iteration, so the check covers the full shard count.
                if cast(Any, shape[dim]) % (placement.split_factor * num_shards) != 0:
                    return False

    return shape[dim] % num_shards == 0


def is_tensor_dim_sharded(spec: DTensorSpec, dim: int) -> bool:
    """Return True if tensor dim is sharded."""
    return any(_is_shard_like(p) and p.dim == dim for p in spec.placements)


def is_tensor_partial(spec: DTensorSpec) -> bool:
    """Return True if tensor is partial on the mesh."""
    return any(p.is_partial() for p in spec.placements)


def infer_broadcast_dims_map(
    common_shape: torch.Size, input_shape: torch.Size
) -> list[int]:
    # infer the broadcast dims map, where it maps from the common shape dim to the input shape dim
    # this is aligned with the broadcast semantics
    # e.g. if common_shape = [1, 2, 3, 4] and input_shape = [2, 3, 4],
    # broadcast_dims_map will be [-1, 0, 1, 2]
    # meaning that dim 0 in the output has no mapping to the input, and dim 1 in the output maps to dim 0 in the input
    from torch.fx.experimental.symbolic_shapes import guard_or_false

    common_ndim = len(common_shape)
    input_ndim = len(input_shape)
    broadcast_dims_map = [-1] * common_ndim
    for idx in range(-1, -1 - input_ndim, -1):
        if guard_or_false(input_shape[idx] == common_shape[idx]):
            broadcast_dims_map[common_ndim + idx] = input_ndim + idx
    return broadcast_dims_map


def map_placements_after_broadcast(
    placements: tuple[Placement, ...],
    shape: torch.Size,
    broadcast_dims_map: list[int],
    partial_to_replicate: bool = False,
) -> tuple[Placement, ...]:
    """Map each placement based on the output shape after broadcast."""
    new_placements: list[Placement] = []
    for placement in placements:
        if isinstance(placement, Partial):
            if partial_to_replicate:
                # map the partial placement to replicate
                new_placements.append(Replicate())
            else:
                new_placements.append(placement)
        elif isinstance(placement, Replicate):
            new_placements.append(placement)
        else:
            if not _is_shard_like(placement):
                raise AssertionError
            shard_dim = normalize_dim(placement.dim, len(shape))
            new_shard_dim = broadcast_dims_map[shard_dim]
            if new_shard_dim != -1:
                # there's a map from the common shape shard dim to
                # the input shape shard dim before broadcasting,
                # use that instead
                if isinstance(placement, _StridedShard):
                    new_placements.append(
                        _StridedShard(
                            new_shard_dim, split_factor=placement.split_factor
                        )
                    )
                else:
                    new_placements.append(Shard(new_shard_dim))
            else:
                # there's no map between common shape shard dim and
                # the input shape shard dim before broadcasting,
                # in this case it means implicit broadcasting happen
                # in this dim, so we can just mark it as replicate
                # and implicit broadcast will broadcast automatically
                # to the sharded shape
                new_placements.append(Replicate())

    return tuple(new_placements)


def generate_redistribute_costs(
    src_strategy: OpStrategy, dst_spec: DTensorSpec
) -> list[float]:
    """Generates one row in the 'redistribute_costs' matrix in an OpSpec
    The length of the returned list will match the number of strategies in 'src_strategy'.

    Each value in the row is the cost of redistributing from a particular src_strategy to dst_spec.
    """
    redistribute_costs: list[float] = [
        redistribute_cost(strat.output_spec, dst_spec)
        for strat in src_strategy.strategies
    ]

    return redistribute_costs


def expand_to_full_mesh_op_strategy(
    mesh: DeviceMesh,
    op_schema: OpSchema,
    single_mesh_dim_strategies: list[PlacementList],
    *,
    output_tensor_meta: TensorMeta | Sequence[TensorMeta | None] | None = None,
    input_index: int = 1,
    inplace_op: bool = False,
    allow_unbacked_sharding: bool | None = None,
    allow_uneven_sharding: bool = False,
    full_mesh_strategy_filter: Callable[
        [list[DTensorSpec], DTensorSpec | tuple[DTensorSpec | None, ...]], bool
    ]
    | None = None,
    different_mesh_args: list[int] | None = None,
) -> OpStrategy:
    """
    Convenience function to allow writing a sharding strategy considering only a single mesh dimension,
    and have it expanded combinatorially to all mesh dimensions.

    Args:
        mesh (DeviceMesh): the device mesh to expand the strategy to
        op_schema (OpSchema): the op schema
        single_mesh_dim_strategies (list[PlacementList]): the sharding strategies to expand. The outer list is over
            different strategies.  The inner PlacementList is over the outputs and inputs of the op. If input_index is 1,
            a PlacementList looks like [output_placement, input_placement1, input_placement2, ...].
        output_tensor_meta: tensor metadata for the output(s), used to populate DTensorSpec.tensor_meta field
        input_index: the number of outputs of the op, defaults to 1
        inplace_op: whether the op is inplace or not, defaults to False
        full_mesh_strategy_filter: callback to filter invalid full-mesh strategy
            candidates, defaults to None.

    Example: Let's say `my_op(tensor_x, tensor_y) - > output_tensor`  can support sharding or replicating tensor_x,
    but always requires tensor_y to be replicated.  We can specify these valid combinations ignoring mesh dims.
    Then, we can rely on `expand_to_full_mesh_op_strategy` to create every possible combination of these shardings
    over multiple mesh dimensions, filtering out any combinations that are invalid based on the actual mesh dim size.

        single_mesh_dim_strategies = [
            # first strategy: return output sharded on first dim, shard tensor_x on its first dim, replicate tensor_y
            [Shard(0), Shard(0), Replicate()]
            # second strategy: replicate output, and both inputs
            [Replicate(), Replicate(), Replicate()]
        ]
    """
    # Expand the single_mesh_dim_strategies to full mesh dim strategies.
    all_mesh_dim_strategies = [single_mesh_dim_strategies] * mesh.ndim

    strategy_combs = itertools.product(*all_mesh_dim_strategies)

    args_strategy = op_schema.args_strategy
    kwargs_strategy = op_schema.kwargs_strategy
    input_args_strategy = args_strategy + kwargs_strategy

    # Propagate use_strided_shard_as_shard_order from inputs so that
    # strategy specs with _StridedShard get the correct flag (and thus
    # correct shard_order) at construction time, avoiding shard_order
    # mismatches in redistribute_cost computation.
    _input_use_strided: bool | None = None
    for input_strat in input_args_strategy:
        input_spec = input_strat.strategies[0].output_spec
        if any(isinstance(p, _StridedShard) for p in input_spec.placements):
            _input_use_strided = input_spec.use_strided_shard_as_shard_order
            break

    all_strategies = []
    # Track input placements if we skip strategies due to inplace placement mismatch
    blocking_inplace_input_placements: tuple[Placement, ...] | None = None
    for strategy_comb in strategy_combs:
        spec_list: list[DTensorSpec | None] = []
        # Track how many non-None output specs we've seen (for output_tensor_meta indexing).
        # This is needed because output_tensor_meta may contain only non-None entries,
        # so we can't use position directly when there are None entries in the output.
        output_spec_count = 0
        # Track input args separately since not all tensor inputs have OpStrategy
        # (e.g., philox_seed/offset in SDPA are scalar tensors without OpStrategy)
        input_strategy_counter = 0
        for position, specs in enumerate(zip(*strategy_comb, strict=True)):
            if any(spec is None for spec in specs):
                if any(spec is not None for spec in specs):
                    raise AssertionError(
                        f"{op_schema.op}: strategy position {position} mixes None "
                        f"with placements across mesh dims: {specs}"
                    )
                spec_list.append(None)
            else:
                # Populate tensor_meta field for both output and input specs,
                # including for tuple output cases
                tensor_meta = None
                # Use position to determine output vs input territory
                # (position includes None entries, unlike the old spec_index)
                if position < input_index:
                    # This is an output position
                    if output_tensor_meta is not None:
                        if isinstance(output_tensor_meta, TensorMeta):
                            tensor_meta = output_tensor_meta
                        elif isinstance(output_tensor_meta, (tuple, list)):
                            if output_spec_count < len(output_tensor_meta):
                                tensor_meta = output_tensor_meta[output_spec_count]
                    output_spec_count += 1
                else:
                    # This is an input position
                    # Only get tensor_meta if we have a corresponding input_args_strategy entry
                    if input_strategy_counter < len(input_args_strategy):
                        tensor_meta = input_args_strategy[
                            input_strategy_counter
                        ].tensor_meta
                        input_strategy_counter += 1

                # pyrefly: ignore [bad-argument-type]
                use_strided = (
                    _input_use_strided
                    if _input_use_strided is not None
                    and any(isinstance(p, _StridedShard) for p in specs)
                    else None
                )
                spec_list.append(
                    DTensorSpec(
                        mesh,
                        specs,
                        tensor_meta=tensor_meta,
                        use_strided_shard_as_shard_order=use_strided,
                    )
                )

        # Skip strategy combinations that would create mixed partial types
        # (except sum+avg which commute with each other).
        # We check (type, reduce_op) pairs rather than just reduce_op because
        # Partial subclasses like _MaskPartial have different reduction semantics
        # even when they share the same reduce_op string.
        has_mixed_partial = False
        for spec in spec_list:
            if spec is not None:
                partial_kinds = {
                    (type(p), p.reduce_op)
                    for p in spec.placements
                    if isinstance(p, Partial)
                }
                if len(partial_kinds) > 1:
                    reduce_ops = {ro for _, ro in partial_kinds}
                    types = {t for t, _ in partial_kinds}
                    if not (len(types) == 1 and reduce_ops == {"sum", "avg"}):
                        has_mixed_partial = True
                        break
        if has_mixed_partial:
            continue

        input_specs: list[DTensorSpec] = [
            s for s in spec_list[input_index:] if isinstance(s, DTensorSpec)
        ]

        if len(input_specs) != len(input_args_strategy):
            raise AssertionError(
                f"input_specs({len(input_specs)}) != strategies({len(input_args_strategy)}: "
                f"{len(args_strategy)} args + {len(kwargs_strategy)} kwargs)"
            )

        # Note [Multi-mesh args]
        #
        # Some ops accept args whose DTensor lives on a different DeviceMesh
        # than the op's primary compute mesh.  We call these "multi-mesh
        # args".  They arise in fused optimizer ops (e.g. _fused_adam_)
        # where *state_steps* is a per-rank scalar counter allocated on a
        # smaller sub-mesh (e.g. 1-D DP) while params and grads live on a
        # larger mesh (e.g. 2-D DP × TP).
        #
        # Why must these args be Replicate?
        #   Sharding implies a specific partitioning of a tensor's data
        #   across the ranks of a mesh.  If a tensor doesn't even *exist*
        #   on the compute mesh, there is no meaningful way to interpret a
        #   Shard placement for it.  Replicate, on the other hand, is
        #   mesh-agnostic: every rank already holds the full data, so the
        #   op can simply read the value regardless of which mesh owns it.
        #
        # What we do here:
        #   We preserve the original mesh and Replicate placement for these
        #   args so the propagator does not try to redistribute them onto
        #   the compute mesh (which would fail or produce wrong results).
        #
        # This is distinct from the *element_mesh* handling in
        # single_dim_strategy.py, which deals with foreach ops where
        # different *elements* in a tensor list may live on different
        # sub-meshes (e.g. param group A on 2-D mesh, param group B on
        # 1-D mesh).
        # TODO: refactor fused_ops handling so that there are no longer
        # args on different meshes
        if different_mesh_args is not None:
            for idx in different_mesh_args:
                if idx < len(input_args_strategy):
                    cross_mesh_input = input_args_strategy[idx]
                    original_spec = cross_mesh_input.strategies[0].output_spec
                    if original_spec.mesh != mesh:
                        if not all(p == Replicate() for p in original_spec.placements):
                            raise RuntimeError(
                                f"Cross-mesh input at index {idx} must be Replicate, "
                                f"but got {original_spec.placements}"
                            )
                        input_specs[idx] = DTensorSpec(
                            mesh=original_spec.mesh,
                            placements=original_spec.placements,
                            tensor_meta=original_spec.tensor_meta,
                        )
        self_spec = input_args_strategy[0].strategies[0].output_spec

        redistribute_input = self_spec.placements != input_specs[0].placements
        mismatching_input_output = (
            spec_list[0] is not None and spec_list[0].placements != self_spec.placements
        )
        if inplace_op and (redistribute_input or mismatching_input_output):
            # For inplace ops, both the proposed input[0] and the output must
            # match self's runtime placement: input[0] because self can't be
            # redistributed, output because the result IS self.
            if blocking_inplace_input_placements is None:
                blocking_inplace_input_placements = self_spec.placements
            continue

        # For out= variant ops, output placement must match each out kwarg's
        # placement. Kwargs are intentionally not redistributed in dispatch.
        if op_schema.is_out_variant_op():
            out_arg_idx = 0
            mismatched_out_arg = False
            for argument in op_schema.op._schema.arguments:
                if not argument.is_out:
                    continue
                out_arg = op_schema.kwargs_schema.get(argument.name)
                if isinstance(out_arg, OpStrategy):
                    output_spec = spec_list[out_arg_idx]
                    out_arg_spec = out_arg.strategies[0].output_spec
                    if (
                        output_spec is None
                        or output_spec.placements != out_arg_spec.placements
                    ):
                        mismatched_out_arg = True
                        break
                out_arg_idx += 1
            if mismatched_out_arg:
                continue

        output_specs: tuple[DTensorSpec | None, ...] | DTensorSpec | None
        has_container_output_meta = isinstance(
            output_tensor_meta, Sequence
        ) and not isinstance(output_tensor_meta, TensorMeta)
        if op_schema.return_type_list_tensor_like() and has_container_output_meta:
            # Tensor[] outputs must stay container-shaped even when there is
            # only one output spec, so they are distinguishable from Tensor.
            output_specs = tuple(spec_list[:input_index])
        elif input_index == 0:
            # No outputs (e.g., _linalg_check_errors)
            output_specs = None
        elif input_index > 1:
            output_specs = tuple(spec_list[:input_index])
        else:
            if spec_list[0] is not None:
                output_specs = spec_list[0]
            else:
                raise RuntimeError("output spec is None")

        # check all inputs are shardable
        if not all(
            is_tensor_shardable(
                inp.shape, s, allow_unbacked_sharding=allow_unbacked_sharding
            )
            or (
                allow_uneven_sharding
                and inp.strategies[0].output_spec.placements == s.placements
            )
            for inp, s in zip(input_args_strategy, input_specs)
        ):
            continue

        # perform additional op-specific filtering
        # Skip callback for no-output ops (output_specs is None)
        if full_mesh_strategy_filter is not None and output_specs is not None:
            if not full_mesh_strategy_filter(input_specs, output_specs):
                continue

        redistribute_cost = [
            generate_redistribute_costs(input_strategy, input_spec)
            for input_strategy, input_spec in zip(input_args_strategy, input_specs)
        ]

        strategy = OpSpec(
            output_specs=output_specs,
            input_specs=input_specs,
            redistribute_cost=redistribute_cost,
        )
        all_strategies.append(strategy)

    # If all strategies were filtered out due to inplace placement mismatch,
    # raise a clear error message instead of returning an empty OpStrategy
    # (which would later cause a cryptic "min() arg is an empty sequence" error)
    if not all_strategies and blocking_inplace_input_placements is not None:
        raise RuntimeError(
            f"{op_schema.op}: in-place operations that require placement changes "
            f"are not supported. The input has placement {blocking_inplace_input_placements}, "
            f"but no valid strategy preserves this placement. "
            f"Please use the out-of-place version of this operation instead."
        )
    if not all_strategies:
        raise RuntimeError(
            f"{op_schema.op}: no valid sharding strategy for the input placements."
        )

    return OpStrategy(all_strategies)


def shift_shard_dims_after_insert(
    placements: Sequence[Placement], insert_dim: int = 0
) -> Sequence[Placement]:
    normalized_placements: list[Placement] = []
    for placement in placements:
        if isinstance(placement, _StridedShard) and placement.dim >= insert_dim:
            normalized_placements.append(
                _StridedShard(placement.dim + 1, split_factor=placement.split_factor)
            )
        elif isinstance(placement, Shard) and placement.dim >= insert_dim:
            normalized_placements.append(Shard(placement.dim + 1))
        else:
            normalized_placements.append(placement)
    return normalized_placements


def shift_shard_dims_after_remove(
    placements: Sequence[Placement], remove_dim: int = 0
) -> Sequence[Placement]:
    normalized_placements: list[Placement] = []
    for placement in placements:
        if isinstance(placement, _StridedShard) and placement.dim > remove_dim:
            normalized_placements.append(
                _StridedShard(placement.dim - 1, split_factor=placement.split_factor)
            )
        elif isinstance(placement, Shard) and placement.dim > remove_dim:
            normalized_placements.append(Shard(placement.dim - 1))
        else:
            normalized_placements.append(placement)
    return normalized_placements
<|endoftext|># mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
from collections.abc import Callable, Sequence, Sized
from typing import cast

import torch
from torch._ops import OpOverload
from torch._prims_common import IntLike
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._op_schema import (
    ArgsType,
    KwargsType,
    OpSchema,
    OpSpec,
    OpStrategy,
    RuntimeSchemaInfo,
    StrategyType,
    TensorMeta,
    TupleStrategy,
)
from torch.distributed.tensor._ops.single_dim_strategy import (
    _ShardingPlaceholder,
    register_single_dim_strategy,
)
from torch.distributed.tensor._ops.utils import (
    generate_redistribute_costs,
    is_tensor_dim_sharded,
    normalize_dim,
    register_op_strategy,
    shift_shard_dims_after_insert,
    shift_shard_dims_after_remove,
)
from torch.distributed.tensor.placement_types import (
    _is_shard_like,
    _MaskPartial,
    Partial,
    Placement,
    Replicate,
    Shard,
)
from torch.fx.experimental.symbolic_shapes import statically_known_true


aten = torch.ops.aten
prims = torch.ops.prims

_PARTIAL_PASS_THROUGH_REDUCE_OPS = ("sum", "avg", "min", "max")


@register_single_dim_strategy(
    [
        aten.clone.default,
        aten.contiguous.default,
        aten.detach.default,
        aten.detach_.default,
        aten.alias.default,
        aten.fill_.Scalar,
        aten.view.dtype,
        aten.zero_.default,
        prims.view_of.default,
    ],
    allow_unbacked_sharding=True,
    allow_uneven_sharding=True,
)
def propagate_single_input_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Propagate placements for ops with one tensor input.

    This is the single-dim form of the old one-to-one OpStrategy: a sharded
    input dim produces the same sharded output dim. Supported Partial
    placements are also forwarded with the same pending reduction.
    """
    input_meta = cast(TensorMeta, args_schema[0])
    strategies: list[list[Placement | _ShardingPlaceholder]] = [
        [_ShardingPlaceholder(dim), _ShardingPlaceholder(dim)]
        for dim in range(len(input_meta.shape))
    ]
    for reduce_op in _PARTIAL_PASS_THROUGH_REDUCE_OPS:
        strategies.append([Partial(reduce_op), Partial(reduce_op)])
    return strategies


def _partial_needs_reduce_for_dtype_cast(
    reduce_op: str,
    src_dtype: torch.dtype,
    target_dtype: torch.dtype | None,
) -> bool:
    """Return True when reduce_op does not commute with the dtype cast."""
    if target_dtype is None or src_dtype == target_dtype:
        return False
    if target_dtype == torch.bool:
        return True
    if reduce_op in ("max", "min"):
        return False
    return src_dtype.is_floating_point and not target_dtype.is_floating_point


@register_single_dim_strategy(
    aten._to_copy.default,
    schema_info=RuntimeSchemaInfo(static_kwargkey=["dtype"]),
    allow_unbacked_sharding=True,
    allow_uneven_sharding=True,
)
def _to_copy_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    input_meta = cast(TensorMeta, args_schema[0])
    src_dtype = input_meta.dtype
    target_dtype = cast(torch.dtype | None, kwargs_schema.get("dtype", None))

    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for dim in range(len(input_meta.shape)):
        strategies.append([_ShardingPlaceholder(dim), _ShardingPlaceholder(dim)])
    for reduce_op in _PARTIAL_PASS_THROUGH_REDUCE_OPS:
        if not _partial_needs_reduce_for_dtype_cast(reduce_op, src_dtype, target_dtype):
            strategies.append([Partial(reduce_op), Partial(reduce_op)])
    return strategies


@register_single_dim_strategy(
    [
        aten.equal.default,
        aten.is_same_size.default,
    ],
    allow_unbacked_sharding=True,
    allow_uneven_sharding=True,
)
def equal_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Require both tensor operands to use the same non-partial layout.

    equal/is_same_size compare corresponding local shards, so mismatched
    shardings would compare different global elements. Partial inputs must use
    the implicit Replicate fallback because comparing unreduced partial values
    would be invalid. If either operand is scalar, no sharded rows are
    produced and the implicit Replicate rule is used.
    """
    self_meta = cast(TensorMeta, args_schema[0])
    other_meta = cast(TensorMeta, args_schema[1])
    return [
        [_ShardingPlaceholder(dim), _ShardingPlaceholder(dim)]
        for dim in range(min(len(self_meta.shape), len(other_meta.shape)))
    ]


register_single_dim_strategy(
    aten.empty_like.default,
    schema_info=RuntimeSchemaInfo(1, ["dtype"]),
    allow_unbacked_sharding=True,
    allow_uneven_sharding=True,
)(propagate_single_input_single_dim_strategy)


@register_single_dim_strategy(
    [
        aten.ones_like.default,
        aten.rand_like.default,
        aten.randn_like.default,
        aten.zeros_like.default,
    ],
    schema_info=RuntimeSchemaInfo(1, ["dtype"]),
    allow_unbacked_sharding=True,
    allow_uneven_sharding=True,
)
@register_single_dim_strategy(
    [aten.full_like.default],
    schema_info=RuntimeSchemaInfo(2, ["dtype"]),
    allow_unbacked_sharding=True,
    allow_uneven_sharding=True,
)
@register_single_dim_strategy(
    [
        aten.randint_like.default,
        aten.randint_like.low_dtype,
        aten.randint_like.low_dtype_out,
    ],
    schema_info=RuntimeSchemaInfo(3, ["dtype"]),
    allow_unbacked_sharding=True,
    allow_uneven_sharding=True,
)
def create_like_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    # These factory-like ops create tensors with the input shape but with
    # content independent of the input. Sharding can follow the input, but a
    # Partial input maps to a Replicate output so generated values are not
    # reduced as if they were accumulated partials.
    input_meta = cast(TensorMeta, args_schema[0])
    kw_names = [k for k, v in kwargs_schema.items() if isinstance(v, TensorMeta)]
    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for dim in range(len(input_meta.shape)):
        placement = _ShardingPlaceholder(dim)
        strategies.append(
            [placement, placement]
            + [placement if name == "out" else Replicate() for name in kw_names]
        )
    for reduce_op in Partial.ALL_REDUCE_OPS:
        strategies.append(
            [Replicate(), Partial(reduce_op)] + [Replicate() for _ in kw_names]
        )
    return strategies


@register_single_dim_strategy(
    [
        aten.new_empty.default,
        aten.new_full.default,
        aten.new_ones.default,
        aten.new_zeros.default,
        aten.new_empty_strided.default,
    ],
    schema_info=RuntimeSchemaInfo(1, ["dtype"]),
    allow_unbacked_sharding=True,
    allow_uneven_sharding=True,
)
def new_factory_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    # Replicated output is always legal and is inserted by the single-dim
    # infra. If input and output have the same shape, sharded inputs can also
    # propagate to the new tensor.
    input_meta = cast(TensorMeta, args_schema[0])
    output_shape = cast(Sequence[object], args_schema[1])
    same_shape = tuple(input_meta.shape) == tuple(output_shape)

    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for dim in range(len(input_meta.shape)):
        placement = _ShardingPlaceholder(dim)
        if same_shape:
            strategies.append([placement, placement])
        else:
            strategies.append([Replicate(), placement])

    # Uninitialized factories (new_empty*) can propagate Partial when the shape
    # matches because the storage is expected to be overwritten later, e.g.
    # autograd's new_empty_strided + copy_ path. Initialized factories keep a
    # Replicate output for Partial inputs to avoid reducing generated constants.
    if op in (aten.new_empty.default, aten.new_empty_strided.default):
        for reduce_op in _PARTIAL_PASS_THROUGH_REDUCE_OPS:
            if same_shape:
                strategies.append([Partial(reduce_op), Partial(reduce_op)])
            else:
                strategies.append([Replicate(), Partial(reduce_op)])
    else:
        for reduce_op in _PARTIAL_PASS_THROUGH_REDUCE_OPS:
            strategies.append([Replicate(), Partial(reduce_op)])

    return strategies


@register_single_dim_strategy(aten.bucketize.Tensor)
def bucketize_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Bucketize returns indices into a sorted boundary tensor.

    Three families of strategies:
    1. Shard the input (and output) on any dim, keep boundaries replicated.
    2. Shard boundaries on dim 0, replicate input, output is Partial("sum").
       Each rank counts how many of its local boundary values each input
       element exceeds; summing across ranks gives the correct global index.
    3. Partial("max") or Partial("min") input with replicated boundaries.
       Bucketize is monotonically non-decreasing in its input, so reducing
       local bucket indices with max (or min) across ranks gives the same
       result as bucketizing the reduced input values.
    """
    input_meta, _boundaries_meta = args_schema
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for dim in range(len(input_meta.shape)):
        strategies.append(
            [_ShardingPlaceholder(dim), _ShardingPlaceholder(dim), Replicate()]
        )
    strategies.append([Partial("sum"), Replicate(), _ShardingPlaceholder(0)])
    for reduce_op in ("max", "min"):
        strategies.append([Partial(reduce_op), Partial(reduce_op), Replicate()])
    return strategies


@register_single_dim_strategy(
    aten.select.int,
    schema_info=RuntimeSchemaInfo(1),
    allow_unbacked_sharding=True,
    allow_uneven_sharding=True,
)
def select_int_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Forward all shardings except the selected dimension.

    select removes selected_dim. A shard before that dim keeps its output dim;
    a shard after it shifts left by one. If selected_dim is sharded, the
    strategy search must pick a replicated/resharded fallback.
    """
    input_meta = cast(TensorMeta, args_schema[0])
    selected_dim = normalize_dim(cast(int, args_schema[1]), len(input_meta.shape))

    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for dim in range(len(input_meta.shape)):
        if dim == selected_dim:
            continue
        output_dim = dim if dim < selected_dim else dim - 1
        strategies.append([_ShardingPlaceholder(output_dim), _ShardingPlaceholder(dim)])
    for reduce_op in _PARTIAL_PASS_THROUGH_REDUCE_OPS:
        strategies.append([Partial(reduce_op), Partial(reduce_op)])
    return strategies


@register_single_dim_strategy(
    aten.select_backward.default,
    schema_info=RuntimeSchemaInfo(1),
    allow_unbacked_sharding=True,
    allow_uneven_sharding=True,
)
def select_backward_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Inverse of select: insert the selected dimension into the output layout."""
    grad_meta = cast(TensorMeta, args_schema[0])
    input_sizes = cast(Sequence[object], args_schema[1])
    dim = normalize_dim(cast(int, args_schema[2]), len(input_sizes))

    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for grad_dim in range(len(grad_meta.shape)):
        output_dim = grad_dim if grad_dim < dim else grad_dim + 1
        strategies.append(
            [_ShardingPlaceholder(output_dim), _ShardingPlaceholder(grad_dim)]
        )
    for reduce_op in _PARTIAL_PASS_THROUGH_REDUCE_OPS:
        strategies.append([Partial(reduce_op), Partial(reduce_op)])
    return strategies


@register_single_dim_strategy(
    aten.slice.Tensor,
    schema_info=RuntimeSchemaInfo(1),
    allow_unbacked_sharding=True,
    allow_uneven_sharding=True,
)
def slice_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Forward shardings except on the sliced dimension.

    A nontrivial slice changes offsets along slice_dim, so a local shard on
    that dimension cannot be forwarded without extra global index handling. A
    statically full slice is just a view and can keep that sharding.
    """
    defaults = (None, 0, None, None, 1)
    input_meta, dim, start, end, step = args_schema + defaults[len(args_schema) :]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    if not isinstance(dim, int):
        raise AssertionError(f"Expected int, got {type(dim)}")
    input_shape = input_meta.shape

    if start is None:
        start = 0
    if not isinstance(start, IntLike):
        raise AssertionError(f"Expected IntLike, got {type(start)}")
    if end is not None and not isinstance(end, IntLike):
        raise AssertionError(f"Expected IntLike, got {type(end)}")
    if not isinstance(step, IntLike):
        raise AssertionError(f"Expected IntLike, got {type(step)}")
    if end is None or statically_known_true(end > input_shape[dim]):
        end = input_shape[dim]

    slice_dim = normalize_dim(dim, len(input_shape))
    start = start + input_shape[dim] if statically_known_true(start < 0) else start
    end = end + input_shape[dim] if statically_known_true(end < 0) else end

    statically_redundant_slice = (
        statically_known_true(start == 0)
        and statically_known_true(end == input_shape[dim])
        and statically_known_true(step == 1)
    )

    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for dim in range(len(input_shape)):
        if dim != slice_dim or statically_redundant_slice:
            strategies.append([_ShardingPlaceholder(dim), _ShardingPlaceholder(dim)])
    for reduce_op in _PARTIAL_PASS_THROUGH_REDUCE_OPS:
        strategies.append([Partial(reduce_op), Partial(reduce_op)])
    return strategies


@register_single_dim_strategy(
    aten.slice_backward.default,
    schema_info=RuntimeSchemaInfo(1),
    allow_unbacked_sharding=True,
    allow_uneven_sharding=True,
)
def slice_backward_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Forward grad shardings except along the original sliced dimension."""
    grad_output_meta = args_schema[0]
    input_sizes = args_schema[1]
    dim = args_schema[2]
    if not isinstance(grad_output_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(grad_output_meta)}")
    if not isinstance(input_sizes, Sequence):
        raise AssertionError(f"Expected Sequence, got {type(input_sizes)}")
    if not isinstance(dim, int):
        raise AssertionError(f"Expected int, got {type(dim)}")
    slice_dim = normalize_dim(dim, len(input_sizes))

    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for dim in range(len(grad_output_meta.shape)):
        if dim != slice_dim:
            strategies.append([_ShardingPlaceholder(dim), _ShardingPlaceholder(dim)])
    for reduce_op in _PARTIAL_PASS_THROUGH_REDUCE_OPS:
        strategies.append([Partial(reduce_op), Partial(reduce_op)])
    return strategies


@register_single_dim_strategy(
    aten.slice_scatter.default,
    schema_info=RuntimeSchemaInfo(2),
    allow_unbacked_sharding=True,
    allow_uneven_sharding=True,
)
def slice_scatter_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Let input, src, and output follow each other off the scatter dimension.

    slice_scatter replaces a slice of self with src. Non-slice dimensions have
    matching extents and can be sharded locally; the slice dimension stays
    replicated via fallback because local slice coordinates would otherwise
    depend on global offsets.
    """
    defaults = (None, None, 0)
    input_meta, src_meta, dim = args_schema[:3] + defaults[len(args_schema[:3]) :]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    if not isinstance(src_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(src_meta)}")
    if not isinstance(dim, int):
        raise AssertionError(f"Expected int, got {type(dim)}")
    slice_dim = normalize_dim(dim, len(input_meta.shape))

    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for dim in range(len(input_meta.shape)):
        if dim != slice_dim:
            strategies.append(
                [
                    _ShardingPlaceholder(dim),
                    _ShardingPlaceholder(dim),
                    _ShardingPlaceholder(dim),
                ]
            )
    # Replacement is pointwise by position, so matching Partial placements pass
    # through for the supported reductions.
    for reduce_op in _PARTIAL_PASS_THROUGH_REDUCE_OPS:
        strategies.append(
            [
                Partial(reduce_op),
                Partial(reduce_op),
                Partial(reduce_op),
            ]
        )
    return strategies


@register_single_dim_strategy(
    [aten.select_scatter.default],
    schema_info=RuntimeSchemaInfo(1),
)
def select_scatter_single_dim_strategy(
    op: OpOverload,
    args_schema: ArgsType,
    kwargs_schema: KwargsType,
) -> list[list[Placement | _ShardingPlaceholder]]:
    input_meta = args_schema[0]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    ndim = len(input_meta.shape)
    dim = normalize_dim(cast(int, args_schema[2]), ndim)
    # [output, self, src] — src has the select dim removed
    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for d in range(ndim):
        if d == dim:
            continue
        strategies.append(
            [
                _ShardingPlaceholder(d),
                _ShardingPlaceholder(d),
                _ShardingPlaceholder(d if d < dim else d - 1),
            ]
        )
    return strategies


@register_single_dim_strategy(
    [aten.diagonal_scatter.default],
    schema_info=RuntimeSchemaInfo(1),
)
def diagonal_scatter_single_dim_strategy(
    op: OpOverload,
    args_schema: ArgsType,
    kwargs_schema: KwargsType,
) -> list[list[Placement | _ShardingPlaceholder]]:
    input_meta = args_schema[0]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    ndim = len(input_meta.shape)
    # schema: (self, src, offset=0, dim1=0, dim2=1)
    dim1 = cast(int, args_schema[3]) if len(args_schema) > 3 else 0
    dim2 = cast(int, args_schema[4]) if len(args_schema) > 4 else 1
    dim1 = normalize_dim(dim1, ndim)
    dim2 = normalize_dim(dim2, ndim)
    min_d, max_d = min(dim1, dim2), max(dim1, dim2)
    # [output, self, src] — src has dim1/dim2 removed and diagonal appended
    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for d in range(ndim):
        if d in (dim1, dim2):
            continue
        removed = (1 if d > min_d else 0) + (1 if d > max_d else 0)
        strategies.append(
            [
                _ShardingPlaceholder(d),
                _ShardingPlaceholder(d),
                _ShardingPlaceholder(d - removed),
            ]
        )
    return strategies


@register_single_dim_strategy(aten._local_scalar_dense.default)
def local_scalar_dense_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Only allow replication on the input; the output is a local scalar."""
    return []


@register_single_dim_strategy(
    [
        aten.scatter_.value,
        aten.scatter.value,
        aten.scatter_.src,
        aten.scatter.src,
    ],
    schema_info=RuntimeSchemaInfo(1),
)
def scatter_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Sharding strategy for scatter/scatter_ value and src overloads.

    scatter writes along the dim argument, so that tensor axis must stay
    replicated. Any other axis can be sharded only if every tensor operand has
    the same size on it; then each rank scatters into its local slice with no
    communication.

    Rows are [output, input, index] for scalar value overloads and
    [output, input, index, src] when src is a tensor. The all-Replicate
    fallback is inserted by the single-dim strategy infra.
    """
    input_meta, dim, index_meta = args_schema[:3]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    if not isinstance(index_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(index_meta)}")
    if not isinstance(dim, int):
        raise AssertionError(f"Expected int, got {type(dim)}")

    input_shape = input_meta.shape
    index_shape = index_meta.shape
    scatter_dim = normalize_dim(dim, len(input_shape))

    # The scalar value overloads carry no src tensor. scatter.src can also
    # receive a Python number, so only TensorMeta creates a src placement slot.
    src_meta = args_schema[3] if len(args_schema) > 3 else None
    has_tensor_src = isinstance(src_meta, TensorMeta)
    src_shape = src_meta.shape if has_tensor_src else None
    num_specs = 4 if has_tensor_src else 3

    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    if len(input_shape) == len(index_shape):
        for d in range(len(input_shape)):
            # Shard a non-scatter axis only when input/index/src shards align.
            if d == scatter_dim or input_shape[d] != index_shape[d]:
                continue
            if src_shape is not None and (
                len(src_shape) != len(index_shape) or src_shape[d] != index_shape[d]
            ):
                continue
            placement = _ShardingPlaceholder(d)
            strategies.append([placement] * num_specs)
    return strategies


@register_single_dim_strategy(
    aten.scatter_add.default, schema_info=RuntimeSchemaInfo(1)
)
def scatter_add_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Shard scatter_add only on non-scatter axes with aligned sizes.

    Placement rows are [output, input, index, src]. The scatter axis stays
    replicated because indices address the full extent of that dimension.
    """
    input_meta, dim, index_meta = args_schema[:3]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    if not isinstance(index_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(index_meta)}")
    if not isinstance(dim, int):
        raise AssertionError(f"Expected int, got {type(dim)}")
    input_shape = input_meta.shape
    index_shape = index_meta.shape
    scatter_dim = normalize_dim(dim, len(input_shape))

    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    if len(input_shape) == len(index_shape):
        for d in range(len(input_shape)):
            if d != scatter_dim and input_shape[d] == index_shape[d]:
                placement = _ShardingPlaceholder(d)
                strategies.append([placement, placement, placement, placement])
    return strategies


@register_single_dim_strategy(aten.gather.default, schema_info=RuntimeSchemaInfo(1))
def gather_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Sharding strategy for gather with rows [output, input, index]."""
    input_meta, dim, index_meta = args_schema[:3]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    if not isinstance(index_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(index_meta)}")
    if not isinstance(dim, int):
        raise AssertionError(f"Expected int, got {type(dim)}")
    input_shape = input_meta.shape
    index_shape = index_meta.shape
    gather_dim = normalize_dim(dim, len(input_shape))

    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    # Input sharded on gather_dim: only valid when index has size 1 on that
    # dim. Index/output use a mask partial so reducing combines the local
    # gather results into the requested global value.
    if (
        len(input_shape) > 0
        and gather_dim < len(index_shape)
        and index_shape[gather_dim] == 1
    ):
        mask_partial = _MaskPartial(offset_shape=input_shape, offset_dim=gather_dim)
        strategies.append([mask_partial, Shard(gather_dim), mask_partial])

    # Index sharded on gather_dim: input must be replicated because each rank
    # may request values from the full input; output follows index.
    if len(input_shape) > 0 and gather_dim < len(index_shape):
        strategies.append([Shard(gather_dim), Replicate(), Shard(gather_dim)])

    # Any non-gather dimension can be sharded across all tensors.
    if len(input_shape) == len(index_shape):
        for d in range(len(input_shape)):
            if d != gather_dim:
                placement = _ShardingPlaceholder(d)
                strategies.append([placement, placement, placement])

    return strategies


def _derive_follow_placements_from_tuple_strategy(
    op: torch._ops.OpOverload,
    tuple_strategy: TupleStrategy,
) -> Sequence[Placement]:
    """
    derive the placements to follow from the tuple strategy, mainly used by
    aten.stack, where each operand have the same shape, and correspondingly
    expecting the same sharding
    """

    def merge_placement(
        cur_placement: Placement, new_placement: Placement
    ) -> Placement:
        # semantic if we already have a follow placement, we
        # check each placement for the current arg placement
        # to see if we want to merge/adjust the placement to follow
        # the priority: Partial -> Shard -> Replicate
        # _StridedShard.__eq__ compares both dim and split_factor,
        # so two _StridedShard with different split_factor won't match here.
        if cur_placement == new_placement:
            return cur_placement

        if cur_placement.is_partial():
            if _is_shard_like(new_placement):
                # follow new placement
                return new_placement
            elif new_placement.is_partial():
                # different partial types, we can't merge and have to replicate all here
                return Replicate()
            else:
                # follow partial
                return cur_placement
        elif _is_shard_like(cur_placement):
            if _is_shard_like(new_placement):
                # cur/new placement are different sharding (i.e. different shard dim)
                # currently fallback to replicate all args
                return Replicate()
            else:
                # for partial/replicate, follow the current shard placement
                return cur_placement
        else:
            # current replicate, just follow new placement
            return new_placement

    follow_placements: list[Placement] | None = None
    mesh = tuple_strategy.child_mesh(0)
    for arg_strategy in tuple_strategy.children:
        if not isinstance(arg_strategy, OpStrategy):
            raise AssertionError(f"Expected OpStrategy, got {type(arg_strategy)}")
        if arg_strategy.mesh != mesh:
            raise ValueError(
                f"All operands in {op} must have the same mesh, "
                f"but got {arg_strategy.mesh} and {mesh}."
            )

        for placement_strategy in arg_strategy.strategies:
            arg_placements = placement_strategy.output_spec.placements
            if follow_placements is None:
                follow_placements = list(arg_placements)
                continue
            if follow_placements is None:
                raise AssertionError(
                    "follow_placements should not be None at this point"
                )
            for mesh_idx in range(mesh.ndim):
                # merge placements with the priority
                follow_placements[mesh_idx] = merge_placement(
                    follow_placements[mesh_idx], arg_placements[mesh_idx]
                )
    if follow_placements is None:
        raise AssertionError("follow placements should not be None!")
    return follow_placements


@register_op_strategy(aten.stack.default, RuntimeSchemaInfo(1, needs_pytree=True))
def stack_strategy(op_schema: OpSchema) -> StrategyType:
    # We keep stack as an OpStrategy because .stack is used w/ NormPartial, and its
    # impossible to include every NormPartial rules as there are infinite possibilities
    # with ord. Future changes to infra could be made, but we prefer to deprecate
    # NormPartial at some point.

    args_schema = op_schema.args_schema
    input_tuple_strategy = args_schema[0]
    if not isinstance(input_tuple_strategy, TupleStrategy):
        raise AssertionError(f"Expected TupleStrategy, got {input_tuple_strategy}")
    input_strategies: list[OpStrategy] = []
    for child in input_tuple_strategy.children:
        if not isinstance(child, OpStrategy):
            raise AssertionError(f"Expected OpStrategy, got {child}")
        input_strategies.append(child)
    first_input_strategy = input_strategies[0]
    common_input_ndim = first_input_strategy.ndim
    dim = cast(int, args_schema[1]) if len(args_schema) > 1 else 0
    # normalize the dim to be within the output ndim (input ndim + 1),
    # since stack inserts a new dimension
    dim = normalize_dim(dim, common_input_ndim + 1)

    mesh = first_input_strategy.mesh

    follow_placements = _derive_follow_placements_from_tuple_strategy(
        op_schema.op, input_tuple_strategy
    )

    # create op strategy base on the follow placements
    op_strategy = OpStrategy([])

    input_specs = tuple(
        DTensorSpec(mesh, tuple(follow_placements))
        for _ in range(len(input_tuple_strategy.children))
    )

    # stack op would "insert" new dim, so all sharded dim >= the inserted dim need to
    # be normalized with the new Shard placement
    follow_placements = shift_shard_dims_after_insert(follow_placements, dim)
    output_spec = DTensorSpec(mesh, tuple(follow_placements))
    redistribute_cost = [
        generate_redistribute_costs(input_strategies[i], input_specs[i])
        for i in range(len(input_specs))
    ]
    op_strategy.strategies.append(
        OpSpec(
            output_specs=output_spec,
            input_specs=input_specs,
            redistribute_cost=redistribute_cost,
        )
    )
    return op_strategy


@register_single_dim_strategy(
    aten.cat.default,
    RuntimeSchemaInfo(1, needs_pytree=True),
    allow_unbacked_sharding=False,
)
def cat_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Forward cat input placements except along the concatenation dimension.

    Concatenation changes only cat_dim; sharding there needs redistribution, so
    this rule offers non-cat dims and leaves the replicate fallback to the
    single-dim infra. Legacy 1-D empty tensors have no real cat layout, so they
    stay Replicate while non-empty tensors follow the output placement.
    """
    input_list = args_schema[0]
    # Unfortunate naming, but yes it's a TensorList input, and we represent it
    # as a tuple of TensorMeta.
    if not isinstance(input_list, (tuple, list)):
        raise AssertionError(type(input_list))
    if not all(isinstance(tm, TensorMeta) for tm in input_list):
        raise AssertionError

    if isinstance(input_list, list):
        input_list = tuple(input_list)

    def is_legacy_cat_empty(meta: TensorMeta) -> bool:
        return len(meta.shape) == 1 and statically_known_true(meta.shape[0] == 0)

    num_inputs = len(input_list)
    if num_inputs == 0:
        raise AssertionError("Expected cat inputs to be non-empty list of tensors")
    legacy_empty_inputs = tuple(is_legacy_cat_empty(meta) for meta in input_list)
    real_ndim_set = {
        len(meta.shape)
        for meta, legacy_empty in zip(input_list, legacy_empty_inputs, strict=True)
        if not legacy_empty
    }
    if len(real_ndim_set) > 1:
        raise AssertionError("Expected all non-empty cat inputs to be the same ndim")
    common_ndim = next(iter(real_ndim_set), 1)
    cat_dim = cast(int, args_schema[1]) if len(args_schema) > 1 else 0
    cat_dim = normalize_dim(cat_dim, common_ndim)
    single_dim_strategies = []
    for i in range(common_ndim):
        if i != cat_dim:
            single_dim_strategies.append(
                [_ShardingPlaceholder(i)]
                + [
                    Replicate() if legacy_empty else _ShardingPlaceholder(i)
                    for legacy_empty in legacy_empty_inputs
                ]
            )
    for reduce_op in _PARTIAL_PASS_THROUGH_REDUCE_OPS:
        single_dim_strategies.append(
            [Partial(reduce_op)]  # pyrefly: ignore [bad-argument-type]
            + [
                Replicate() if legacy_empty else Partial(reduce_op)
                for legacy_empty in legacy_empty_inputs
            ]
        )
    # pyrefly: ignore [bad-return]
    return single_dim_strategies


@register_single_dim_strategy(
    aten.index_select.default, schema_info=RuntimeSchemaInfo(1)
)
def index_select_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    values_meta, dim, index_meta = args_schema
    if not isinstance(values_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(values_meta)}")
    if not isinstance(dim, int):
        raise AssertionError(f"Expected int, got {type(dim)}")
    dim = normalize_dim(dim, len(values_meta.shape))

    strategies: list[list[Placement | _ShardingPlaceholder]] = []

    # Shard values on any non-indexed dim (output has same ndim)
    for d in range(len(values_meta.shape)):
        if d == dim:
            continue
        strategies.append(
            [_ShardingPlaceholder(d), _ShardingPlaceholder(d), Replicate()]
        )

    # Shard index → output sharded on the indexed dim
    strategies.append([_ShardingPlaceholder(dim), Replicate(), _ShardingPlaceholder(0)])

    # Partial passthrough from values
    for reduce_op in _PARTIAL_PASS_THROUGH_REDUCE_OPS:
        strategies.append([Partial(reduce_op), Partial(reduce_op), Replicate()])

    return strategies


@register_single_dim_strategy(
    aten.index.Tensor, schema_info=RuntimeSchemaInfo(needs_pytree=True)
)
def index_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    values_meta, multi_indices_meta = args_schema
    if not isinstance(values_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(values_meta)}")
    if not isinstance(multi_indices_meta, (list, tuple)):
        raise AssertionError(f"Expected list or tuple, got {type(multi_indices_meta)}")

    indexed_dims = [i for i, idx in enumerate(multi_indices_meta) if idx is not None]
    non_indexed_dims = [
        i for i in range(len(values_meta.shape)) if i not in set(indexed_dims)
    ]

    index_metas = [idx for idx in multi_indices_meta if idx is not None]
    if not all(isinstance(m, TensorMeta) for m in index_metas):
        raise AssertionError("Expected all index metas to be TensorMeta")
    broadcast_ndim = max(len(m.shape) for m in index_metas)
    num_indices = len(indexed_dims)

    # Determine where index output dims are inserted in the result
    all_consecutive = all(
        indexed_dims[i + 1] - indexed_dims[i] == 1 for i in range(len(indexed_dims) - 1)
    )
    insert_dim = indexed_dims[0] if all_consecutive else 0

    def values_dim_to_output_dim(d: int) -> int:
        if d < insert_dim:
            return d
        return d + broadcast_ndim - sum(1 for idx_dim in indexed_dims if d > idx_dim)

    strategies: list[list[Placement | _ShardingPlaceholder]] = []

    # Shard values on a non-indexed dim, all indices replicated
    for d in non_indexed_dims:
        out_dim = values_dim_to_output_dim(d)
        rule: list[Placement | _ShardingPlaceholder] = [_ShardingPlaceholder(out_dim)]
        rule.append(_ShardingPlaceholder(d))
        rule.extend([Replicate()] * num_indices)
        strategies.append(rule)

    # Shard indices on the same broadcast dim.  Each index tensor may
    # have a different ndim, so we map broadcast dim → tensor dim via
    # left-padding.  Tensors with size 1 on that dim are replicated
    # (broadcast semantics).
    for bd in range(broadcast_ndim):
        per_tensor: list[tuple[int, int]] = []  # (tensor_dim, size)
        for m in index_metas:
            offset = broadcast_ndim - len(m.shape)
            if bd < offset:
                per_tensor.append((-1, 1))  # implicit broadcast
            else:
                td = bd - offset
                per_tensor.append((td, m.shape[td]))
        if all(s == 1 for _, s in per_tensor):
            continue  # all broadcast-only, skip
        out_dim = bd + insert_dim
        rule: list[Placement | _ShardingPlaceholder] = [_ShardingPlaceholder(out_dim)]
        rule.append(Replicate())
        for td, s in per_tensor:
            if s > 1:
                rule.append(_ShardingPlaceholder(td))
            else:
                rule.append(Replicate())
        strategies.append(rule)

    # Partial passthrough from values
    for reduce_op in Partial.LINEAR_REDUCE_OPS:
        rule: list[Placement | _ShardingPlaceholder] = [
            Partial(reduce_op),
            Partial(reduce_op),
        ]
        rule.extend([Replicate()] * num_indices)
        strategies.append(rule)

    return strategies


@register_single_dim_strategy(
    [aten.index_put.default, aten.index_put_.default, aten._index_put_impl_.default],
    schema_info=RuntimeSchemaInfo(needs_pytree=True),
)
def index_put_single_dim_strategy(
    op: OpOverload, args: ArgsType, kwargs: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Single-dim sharding strategy for index_put(self, indices, values).

    Strategy format: [output, input, *indices, value]

    How index_put works:

      indices is a tuple of index tensors and Nones:
      - an index tensor at entry i means self is indexed on dim i.
      - a None at entry i means all elements along dim i are selected (like :).
      - any trailing dims (if self.ndim > len(indices)) are also not indexed
        (i.e. implicit trailing Nones).

      All non-None index tensors are broadcast together to produce a
      broadcasted indexing shape. Each position in this broadcasted shape
      serves as an indexing coordinate into self. Each coordinate selects a
      tensor element, or a slice (if non-indexed dims exist).

      values is a tensor broadcastable to the indexing output shape.
      When indexed dims are consecutive starting at dim k, this shape is
      (*self[:k], *broadcast_shape, *self[k+n_indexed:]). When indexed
      dims are non-consecutive, it is (*broadcast_shape, *non_indexed_dims).

    Sharding rules (possibly conservative and incomplete):
      - Index tensors: always Replicate (every rank needs all coordinates).
      - Self cannot be sharded on indexed dims (local position != global position).
      - Self and values CAN be sharded on non-indexed dims.
        The exception is broadcasted value dimensions (size 1) - we require Replicate, but can shard self.
      - Additionally, we allow the full Partial rule on non-indexing tensors.

    """
    self_meta = cast(TensorMeta, args[0])
    indices_meta = cast(tuple[TensorMeta | None, ...], args[1])
    values_meta = cast(TensorMeta, args[2])

    # Determine indexed vs non-indexed dims of self.
    indexed_dims = {i for i, idx in enumerate(indices_meta) if idx is not None}
    non_indexed_dims = [d for d in range(len(self_meta.shape)) if d not in indexed_dims]
    n_indexed = len(indexed_dims)
    values_ndim = len(values_meta.shape)

    # Explicitly compute the broadcast shape of the index tensors.
    index_shapes = [idx.shape for idx in indices_meta if idx is not None]
    broadcast_ndim = len(torch.broadcast_shapes(*index_shapes)) if index_shapes else 0

    # Strategy format: [output, input, *indices, value]
    # The infra flattens the indices list and drops None entries, so only
    # non-None index tensors get a placement slot (all Replicate).
    #
    # Values dim mapping depends on whether indexed dims are contiguous:
    #   Contiguous (e.g., (None, idx0, idx1)): broadcast replaces indexed block in-place.
    #     values shape = (*non_indexed_before, *broadcast_shape, *non_indexed_after)
    #   Non-contiguous (e.g., (idx0, None, idx1)): broadcast goes to front.
    #     values shape = (*broadcast_shape, *non_indexed_dim_sizes)
    indexed_dims_sorted = sorted(indexed_dims)
    contiguous_indexed = len(indexed_dims_sorted) <= 1 or (
        indexed_dims_sorted[-1] - indexed_dims_sorted[0] + 1 == len(indexed_dims_sorted)
    )

    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for i, self_dim in enumerate(non_indexed_dims):
        if contiguous_indexed and indexed_dims_sorted:
            # Broadcast replaces the indexed block in-place.
            first_indexed = indexed_dims_sorted[0]
            if self_dim < first_indexed:
                values_dim = self_dim
            else:
                values_dim = self_dim - n_indexed + broadcast_ndim
        else:
            # Broadcast goes to front (non-contiguous or no indexed dims).
            values_dim = broadcast_ndim + i

        # values_dim is the position in the result tensor, but values may
        # have fewer dims (right-aligned broadcasting). Convert to the
        # actual values tensor dimension.
        result_ndim = broadcast_ndim + len(non_indexed_dims)
        values_tensor_dim = values_dim - (result_ndim - values_ndim)

        if values_tensor_dim < 0:
            values_placement: Placement | _ShardingPlaceholder = Replicate()
        elif values_meta.shape[values_tensor_dim] == 1:
            values_placement = Replicate()
        else:
            values_placement = _ShardingPlaceholder(values_tensor_dim)

        strategies.append(
            [
                _ShardingPlaceholder(self_dim),
                _ShardingPlaceholder(self_dim),
                *([Replicate()] * n_indexed),
                values_placement,
            ]
        )

    # full-partial rule on non-indexing tensors
    strategies.append(
        [
            Partial(),
            Partial(),
            *([Replicate()] * n_indexed),
            Partial(),
        ]
    )
    return strategies


def _index_dim_strategy(
    args_schema: ArgsType,
    shard_row: Callable[[int], list[Placement | _ShardingPlaceholder]],
    partial_rules: list[list[Placement | _ShardingPlaceholder]] | None = None,
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Common strategy for index ops that shard on all dims except the indexed dim.

    Args:
        shard_row: given a dim d, returns the strategy row for sharding on that dim.
        partial_rules: additional Partial passthrough strategies.
    """
    self_meta = cast(TensorMeta, args_schema[0])
    ndim = len(self_meta.shape)
    dim = normalize_dim(cast(int, args_schema[1]), ndim)
    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for d in range(ndim):
        if d != dim:
            strategies.append(shard_row(d))
    if partial_rules:
        strategies.extend(partial_rules)
    return strategies


@register_single_dim_strategy(
    [aten.index_fill.int_Scalar, aten.index_fill_.int_Scalar],
    schema_info=RuntimeSchemaInfo(1),
)
def index_fill_scalar_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    # index_fill(self, dim, index, value) — fills self[..., index, ...] with scalar value.
    # Partial rules: each rank fills with the same scalar v, then reduces.
    # Only idempotent reduces work: avg(v,v,...,v)=v, max(v,v,...,v)=v, min(v,v,...,v)=v.
    # sum and product fail: sum(v,v,...,v)=nv, product(v,v,...,v)=v^n.
    return _index_dim_strategy(
        args_schema,
        lambda d: [
            _ShardingPlaceholder(d),  # result
            _ShardingPlaceholder(d),  # self
            Replicate(),  # value (scalar, same on all ranks)
        ],
        [[Partial(op), Partial(op), Replicate()] for op in ("avg", "max", "min")],
    )


@register_single_dim_strategy(
    [aten.index_fill.int_Tensor, aten.index_fill_.int_Tensor],
    schema_info=RuntimeSchemaInfo(1),
)
def index_fill_tensor_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    # index_fill(self, dim, index, value) — fills self[..., index, ...] with 0-d tensor value.
    # Partial rules: each rank fills with its partial value v_i, then reduces.
    # All reduce ops work because reduce(v_0, ..., v_{n-1}) = V (the global value)
    # regardless of op, since fill is a pure replacement (no mixing with self).
    return _index_dim_strategy(
        args_schema,
        lambda d: [
            _ShardingPlaceholder(d),  # result
            _ShardingPlaceholder(d),  # self
            Replicate(),  # index
            Replicate(),  # value
        ],
        [
            [Partial(op), Partial(op), Replicate(), Partial(op)]
            for op in _PARTIAL_PASS_THROUGH_REDUCE_OPS
        ],
    )


@register_single_dim_strategy(
    [aten.index_reduce.default, aten.index_reduce_.default],
    schema_info=RuntimeSchemaInfo(1),
)
def index_reduce_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    # index_reduce(self, dim, index, source, reduce) — reduces source into self at index positions.
    # No partial rules: reduce ops are "mean"/"amax"/"amin"/"prod", which don't match
    # any Partial reduce op names ("avg"/"max"/"min"/"product"/"sum").
    return _index_dim_strategy(
        args_schema,
        lambda d: [
            _ShardingPlaceholder(d),  # result
            _ShardingPlaceholder(d),  # self
            Replicate(),  # index
            _ShardingPlaceholder(d),  # source
        ],
    )


@register_single_dim_strategy(
    [
        aten.split.Tensor,
        aten.split_with_sizes.default,
        aten.split_with_sizes_copy.default,
    ],
    RuntimeSchemaInfo(1),
    allow_unbacked_sharding=False,
)
def split_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Forward all shardings except the split dimension.

    Splitting a sharded split_dim would require each output chunk to know
    global chunk boundaries, so the single-dim rule excludes that dim and
    relies on the redistribute-to-replicate fallback.
    """
    input_meta = args_schema[0]
    split_size_or_sections = args_schema[1]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    input_ndim = len(input_meta.shape)
    split_dim = cast(int, args_schema[2]) if len(args_schema) > 2 else 0
    dim = normalize_dim(split_dim, input_ndim)

    def size_split(N, i) -> list:
        # Last chunk will be smaller if the tensor size N
        # along the given dimension dim is not divisible by i.
        if not i > 0:
            raise AssertionError(f"Split size must be positive, got {i}")
        return [i] * (N // i) + ([N % i] if N % i != 0 else [])

    output_size_list = (
        size_split(input_meta.shape[dim], split_size_or_sections)
        if isinstance(split_size_or_sections, IntLike)
        else split_size_or_sections
    )
    if not isinstance(output_size_list, Sized):
        raise AssertionError(f"Expected Sized, got {type(output_size_list)}")
    num_outputs = len(output_size_list)

    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for d in range(input_ndim):
        if d != dim:
            placement = _ShardingPlaceholder(d)
            # pyrefly: ignore [bad-argument-type]
            strategies.append([placement] * num_outputs + [placement])
    for reduce_op in _PARTIAL_PASS_THROUGH_REDUCE_OPS:
        strategies.append([Partial(reduce_op)] * (num_outputs + 1))
    return strategies


# TODO: fix remaining failures in xfail("unbind") in test_dtensor_ops.py
#       and remove this xfail item
@register_op_strategy(aten.unbind.int, schema_info=RuntimeSchemaInfo(1))
def gen_unbind_strategy(op_schema: OpSchema) -> StrategyType:
    """Forward all shardings except the unbind dimension."""
    input_strategy = op_schema.args_schema[0]
    if not isinstance(input_strategy, OpStrategy):
        raise AssertionError(f"Expected OpStrategy, got {type(input_strategy)}")
    input_ndim = input_strategy.ndim
    input_shape = input_strategy.shape
    unbind_dim = (
        cast(int, op_schema.args_schema[1]) if len(op_schema.args_schema) > 1 else 0
    )
    unbind_dim = normalize_dim(unbind_dim, input_ndim)

    mesh = input_strategy.mesh
    unbind_strategy = OpStrategy([])
    for arg_strategy in input_strategy.strategies:
        arg_spec = arg_strategy.output_spec
        if is_tensor_dim_sharded(arg_spec, dim=unbind_dim):
            raise RuntimeError(
                f"Attempted to unbind along the sharded dimension {unbind_dim}. ",
                "It cannot be performed without redistribution, which is disallowed "
                "by the current operator.",
            )
        # Only add the strategy if the unbind dim is not sharded.
        output_placements = shift_shard_dims_after_remove(
            arg_spec.placements, unbind_dim
        )
        output_specs = tuple(
            DTensorSpec(mesh, tuple(output_placements))
            for _ in range(input_shape[unbind_dim])
        )
        unbind_strategy.strategies.append(
            OpSpec(
                output_specs=output_specs,
                input_specs=(arg_spec,),
                redistribute_cost=[[0.0] * len(input_strategy.strategies)],
            )
        )
    return unbind_strategy


@register_single_dim_strategy(
    aten.eye.m_out,
    schema_info=RuntimeSchemaInfo(static_kwargkey=["out"]),
    allow_unbacked_sharding=True,
    allow_uneven_sharding=True,
)
def eye_out_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Strategy for torch.eye(..., out=...).

    Sharded eye needs global index offsets to place the diagonal correctly.
    The local out= kernel does not have that context, so only the implicit
    all-Replicate strategy is valid.
    """
    out_meta = kwargs_schema["out"]
    if not isinstance(out_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta for out, got {type(out_meta)}")

    return []


def _pass_through_partials(
    num_inputs: int = 1,
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Pass-through strategies for all supported reduce ops."""
    return [[Partial(op)] * (1 + num_inputs) for op in ("sum", "avg", "max", "min")]


def _shard_inactive_dims(
    ndim: int, active_dims: set[int], num_inputs: int = 1
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Single-dim strategies: shard on dims the op doesn't touch."""
    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for d in range(ndim):
        if d not in active_dims:
            strategies.append([_ShardingPlaceholder(d)] * (1 + num_inputs))
    return strategies


@register_single_dim_strategy(aten.roll.default, schema_info=RuntimeSchemaInfo(1))
def roll_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    input_meta = args_schema[0]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    ndim = len(input_meta.shape)
    raw_dims = cast(list[int], args_schema[2]) if len(args_schema) > 2 else []
    # When dims is empty, roll flattens the tensor — all dims are active
    if not raw_dims:
        raw_dims = list(range(ndim))
    active_dims = {normalize_dim(d, ndim) for d in raw_dims}
    return _shard_inactive_dims(ndim, active_dims) + _pass_through_partials()


@register_single_dim_strategy(aten.flip.default, schema_info=RuntimeSchemaInfo(1))
def flip_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    input_meta = args_schema[0]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    ndim = len(input_meta.shape)
    raw_dims = cast(list[int], args_schema[1])
    active_dims = {normalize_dim(d, ndim) for d in raw_dims}
    return _shard_inactive_dims(ndim, active_dims) + _pass_through_partials()


@register_single_dim_strategy(
    [aten._fft_c2c.default, aten._fft_r2c.default, aten._fft_c2r.default],
    schema_info=RuntimeSchemaInfo(1),
)
def fft_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    input_meta = args_schema[0]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    ndim = len(input_meta.shape)
    raw_dims = cast(list[int], args_schema[1])
    active_dims = {normalize_dim(d, ndim) for d in raw_dims}
    return _shard_inactive_dims(ndim, active_dims) + _pass_through_partials()
<|endoftext|># Copyright (c) Meta Platforms, Inc. and affiliates
from ._conv_ops import *  # noqa: F403
from ._embedding_ops import *  # noqa: F403
from ._math_ops import *  # noqa: F403
from ._matrix_ops import *  # noqa: F403
from ._pointwise_ops import *  # noqa: F403
from ._random_ops import *  # noqa: F403
from ._tensor_ops import *  # noqa: F403
from ._view_ops import *  # noqa: F403
from .autogen import auto_register_op_variants


auto_register_op_variants()
<|endoftext|># mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import (
    OpSchema,
    OpSpec,
    OpStrategy,
    RuntimeSchemaInfo,
)
from torch.distributed.tensor._ops.single_dim_strategy import (
    _ShardingPlaceholder,
    register_single_dim_strategy,
)
from torch.distributed.tensor._ops.utils import (
    as_list,
    generate_redistribute_costs,
    is_tensor_evenly_shardable_on_dim,
    normalize_dim,
    normalize_dims,
)
from torch.distributed.tensor._utils import (
    compute_local_shape_and_global_offset,
    normalize_to_torch_size,
)
from torch.distributed.tensor.placement_types import (
    _is_shard_like,
    _StridedShard,
    Partial,
    Placement,
    Replicate,
    Shard,
)
from torch.fx.experimental.symbolic_shapes import guard_or_false


aten = torch.ops.aten
prims = torch.ops.prims


class Reduction(Enum):
    NONE = 0
    MEAN = 1
    SUM = 2


@dataclass(frozen=True)
class NormReduction:
    norm_type: int | float


ReductionOpType = NormReduction | str


@dataclass(frozen=True)
class _NormPartial(Partial):
    """
    This placement is used for partial p-norm (p not in {inf, -inf, 0}).

    For p-norms, the p-norm over n elements computes (sum_i x_i^p)^(1/p).
    For example, consider 2 ranks, a (4,) tensor sharded on dim-0, and 2-norm:
        Rank 0: [t1, t2] | Rank 1: [t3, t4]
    After computing 2-norm per gradient (partial placement):
        Rank 0: [sqrt(t1^2 + t2^2)] | Rank 1: [sqrt(t3^2 + t4^2)]
    Converting from partial to replicate wants to ultimately get:
        Rank 0/1: [sqrt(t1^2 + t2^2 + t3^2 + t4^2)]
    This is achieved by: x^p -> allreduce sum -> x^(1/p).
    """

    norm_type: int | float = 2

    def __init__(self, norm_type: int | float = 2):
        super().__init__("sum")
        object.__setattr__(self, "norm_type", norm_type)

    def _partition_value(
        self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int
    ) -> torch.Tensor:
        return tensor / math.pow(mesh.size(mesh_dim), 1 / self.norm_type)

    def _reduce_shard_value(
        self,
        tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        shard_spec: Placement,
    ) -> torch.Tensor:
        if not isinstance(shard_spec, Shard):
            raise AssertionError(f"Expected Shard, got {type(shard_spec)}")
        tensor = self._pre_reduce_transform(tensor)
        reduced_tensor = super()._reduce_shard_value(tensor, mesh, mesh_dim, shard_spec)
        return self._post_reduce_transform(reduced_tensor)

    def _reduce_value(
        self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int
    ) -> torch.Tensor:
        tensor = self._pre_reduce_transform(tensor)
        reduced_tensor = super()._reduce_value(tensor, mesh, mesh_dim)
        return self._post_reduce_transform(reduced_tensor)

    def _pre_reduce_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor**self.norm_type

    def _post_reduce_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor ** (1.0 / self.norm_type)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _NormPartial):
            return False
        return self.norm_type == other.norm_type

    def __hash__(self) -> int:
        return 1 + hash(self.norm_type)

    def __repr__(self) -> str:
        return f"_NormPartial({self.norm_type})"

    def __str__(self) -> str:
        return f"_NormP({self.norm_type})"


def _infer_reduction_dims(dims_arg: object, ndim: int) -> list[int] | None:
    if dims_arg is None:
        return None
    dims = cast(list[int], as_list(dims_arg))
    dims = cast(list[int], normalize_dims(dims, ndim))
    empty_dims = [[0], [-1], []]
    if ndim == 0 and dims_arg in empty_dims:
        return None
    return dims


def _infer_reduce_dims_map(
    reduction_dims: list[int], input_ndim: int, keep_dim=False
) -> list[int]:
    reduction_dims_map = []
    new_dim_count = 0
    for input_dim in range(input_ndim):
        if input_dim in reduction_dims and not keep_dim:
            # if input dim in reduction dims, mark it as -1
            reduction_dims_map.append(-1)
        else:
            # otherwise mark it as the new dim
            reduction_dims_map.append(new_dim_count)
            new_dim_count += 1

    return reduction_dims_map


def _replicate_dims_start_at(
    placements: Sequence[Placement], start_dim: int = 0
) -> tuple[Placement, ...]:
    new_placements: list[Placement] = []
    for p in placements:
        if p.is_partial() or (_is_shard_like(p) and p.dim >= start_dim):
            new_placements.append(Replicate())  # make it replicate
        else:
            new_placements.append(p)  # keep the placement
    return tuple(new_placements)


# return new_placements which align with placements but skip the skipped_dim
# Precondition: no shard-like placement on skipped_dim (callers must
# replicate it first via replicate_reduction_dims).
def _skip_dim(
    placements: tuple[Placement, ...], skipped_dim: int
) -> tuple[Placement, ...]:
    new_placements: list[Placement] = []
    for p in placements:
        if isinstance(p, _StridedShard) and p.dim >= skipped_dim:
            new_placements.append(_StridedShard(p.dim - 1, split_factor=p.split_factor))
        elif isinstance(p, Shard) and p.dim >= skipped_dim:
            new_placements.append(Shard(p.dim - 1))
        else:
            new_placements.append(p)
    return tuple(new_placements)


def replicate_reduction_dims(
    placements: tuple[Placement, ...], reduction_dims: list[int]
) -> tuple[Placement, ...]:
    # replicate the reduction dims if not reduction_linear
    new_placements: list[Placement] = []

    for p in placements:
        if p.is_partial():
            new_placements.append(Replicate())
        elif _is_shard_like(p) and p.dim in reduction_dims:
            new_placements.append(Replicate())
        else:
            new_placements.append(p)

    return tuple(new_placements)


def map_placements_after_reduction(
    placements: tuple[Placement, ...],
    reduction_dims: list[int],
    reduction_dims_map: list[int],
    reduction_op: ReductionOpType,
) -> tuple[Placement, ...]:
    """
    Map each placement based on the output shape after reduction.
    """
    new_placements: list[Placement] = []
    for placement in placements:
        if isinstance(placement, (Replicate, Partial)):
            new_placements.append(placement)
        else:
            if not _is_shard_like(placement):
                raise AssertionError(
                    f"Expected Shard/_StridedShard, got {type(placement)}"
                )
            shard_dim = placement.dim
            new_shard_dim = reduction_dims_map[shard_dim]
            if new_shard_dim == -1 or shard_dim in reduction_dims:
                # if new_shard_dim collapsed or its in the reduction dims
                # (i.e. for the case where keepdims=True), we generate partial
                new_placements.append(get_placement_from_reduction_op(reduction_op))
            else:
                if isinstance(placement, _StridedShard):
                    new_placements.append(
                        _StridedShard(
                            new_shard_dim, split_factor=placement.split_factor
                        )
                    )
                elif isinstance(placement, Shard):
                    new_placements.append(Shard(new_shard_dim))
    return tuple(new_placements)


def get_placement_from_reduction_op(reduction_op: ReductionOpType) -> Placement:
    if isinstance(reduction_op, NormReduction):
        if reduction_op.norm_type == 0:
            # return P(sum) for easier reduction_linear handling.
            return Partial("sum")
        return _NormPartial(norm_type=reduction_op.norm_type)
    return Partial(reduction_op)


def common_reduction_strategy(
    input_strategy: OpStrategy,
    reduce_dims: list[int],
    keep_dim: bool = False,
    reduction_linear: bool = True,
    reduction_op: ReductionOpType = "sum",
) -> OpStrategy:
    """
    reduction_linear means that the reduction `f` follows this rule:
        f([f(a), f(b)]) = f([a, b])

    reduction linear should be super set of linearity.
    """
    reduction_strategy = OpStrategy([])

    for op_spec in input_strategy.strategies:
        is_reduction_linear = reduction_linear
        if reduction_op == "avg":
            output_spec = op_spec.output_spec
            local_shape = list(output_spec.tensor_meta.shape)  # type:ignore[union-attr]
            for dim in reduce_dims:
                if not is_tensor_evenly_shardable_on_dim(local_shape, output_spec, dim):
                    is_reduction_linear = False
                    break

        for p in op_spec.output_spec.placements:
            if isinstance(p, Partial) and p.reduce_op != reduction_op:
                is_reduction_linear = False
                break

        if not is_reduction_linear:
            input_placements = replicate_reduction_dims(
                op_spec.output_spec.placements, reduce_dims
            )
        else:
            input_placements = op_spec.output_spec.placements

        input_spec = DTensorSpec(
            mesh=input_strategy.mesh,
            placements=input_placements,
            tensor_meta=op_spec.output_spec.tensor_meta,
        )

        reduce_dims_map = _infer_reduce_dims_map(reduce_dims, input_spec.ndim, keep_dim)
        out_placements = map_placements_after_reduction(
            input_spec.placements, reduce_dims, reduce_dims_map, reduction_op
        )
        reduction_strategy.strategies.append(
            OpSpec(
                output_specs=DTensorSpec(
                    mesh=input_strategy.mesh,
                    placements=out_placements,
                ),
                input_specs=(input_spec,),
                redistribute_cost=[
                    generate_redistribute_costs(input_strategy, input_spec)
                ],
            )
        )

    return reduction_strategy


def _reduction_single_dim_strategy(
    args_schema: tuple[Any, ...],
    reduction_dims: list[int] | None,
    keep_dim: bool,
    reduction_linear: bool,
    reduction_op: ReductionOpType,
    extra_partial_rules: list[list[Placement | _ShardingPlaceholder]] | None = None,
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Build single-dim strategies for reduction ops.

    For non-reduction dims: shard input and output on that dim (with dim remapping).
    For reduction dims (if reduction_linear): shard input on reduction dim, output is
    Partial. Also adds partial propagation rules for linear reductions.

    "avg" reductions do not shard reduced dims because averaging uneven local
    averages does not produce the global average.
    """
    input_meta = args_schema[0]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    ndim = len(input_meta.shape)

    reduce_dims = list(range(ndim)) if reduction_dims is None else reduction_dims

    strategies: list[list[Placement | _ShardingPlaceholder]] = []

    for d in range(ndim):
        if d in reduce_dims:
            if reduction_linear and reduction_op != "avg":
                # Shard on reduction dim -> Partial output.
                strategies.append(
                    [
                        get_placement_from_reduction_op(reduction_op),
                        _ShardingPlaceholder(d),
                    ]
                )
        else:
            # Shard on non-reduction dim -> shard output (with dim remapping)
            if keep_dim:
                out_d = d
            else:
                out_d = d - sum(1 for rd in reduce_dims if rd < d)
            strategies.append([_ShardingPlaceholder(out_d), _ShardingPlaceholder(d)])

    # Partial propagation: Partial(reduction_op) input -> Partial(reduction_op) output.
    # Skip for NormReduction: the old OpStrategy path did not propagate
    # _NormPartial through, so keep that behavior.
    if reduction_linear and not isinstance(reduction_op, NormReduction):
        partial_placement = get_placement_from_reduction_op(reduction_op)
        strategies.append([partial_placement, partial_placement])

    if extra_partial_rules:
        strategies.extend(extra_partial_rules)

    return strategies


# Category C: Linear reductions
# all/any are listed here because the remaining reduction helpers share this map,
# but they are registered separately with reduction_linear=False.
LINEAR_REDUCTION_OP_MAP = {
    aten.all.default: "product",
    aten.all.dim: "product",
    aten.any.default: "sum",
    aten.any.dim: "sum",
    aten.any.dims: "sum",
    aten.any.out: "sum",
    aten.sum.default: "sum",
    aten.sum.dim_IntList: "sum",
    prims.sum.default: "sum",
    # These are only valid when there is no padding
    aten.prod.default: "product",
    aten.prod.dim_int: "product",
    aten.prod.int_out: "product",
    prims.prod.default: "product",
    aten.max.default: "max",
    aten.max.out: "max",
    aten.min.default: "min",
    aten.min.out: "min",
    aten.amax.default: "max",
    aten.amax.out: "max",
    prims.amax.default: "max",
    aten.amin.default: "min",
    aten.amin.out: "min",
    prims.amin.default: "min",
    aten.nansum.default: "sum",
}

FOREACH_LINEAR_REDUCTION_OP_MAP = {
    aten._foreach_max.default: "max",
}

FOREACH_LINEAR_REDUCTION_OPS = list(FOREACH_LINEAR_REDUCTION_OP_MAP)

# all/any are NOT linear reductions: validation showed S(reduction_dim)->P(...)
# produces incorrect results. They only support sharding on non-reduction dims.
NON_LINEAR_BOOL_REDUCTION_OPS = [
    aten.all.default,
    aten.all.dim,
    aten.any.default,
    aten.any.dim,
    aten.any.dims,
    aten.any.out,
]

MEAN_REDUCTION_OPS = [
    aten.mean.default,
    aten.mean.dim,
    aten.mean.out,
]

ARGMAX_ARGMIN_OPS = {
    aten.argmax.default: "max",
    aten.argmin.default: "min",
}


LINEAR_REDUCTION_OPS = [
    op for op in LINEAR_REDUCTION_OP_MAP if op not in NON_LINEAR_BOOL_REDUCTION_OPS
]


@register_single_dim_strategy(
    FOREACH_LINEAR_REDUCTION_OPS,
    schema_info=RuntimeSchemaInfo(1, needs_pytree=True),
    allow_uneven_sharding=True,
    allow_unbacked_sharding=False,
)
@register_single_dim_strategy(
    LINEAR_REDUCTION_OPS,
    schema_info=RuntimeSchemaInfo(1),
    allow_uneven_sharding=True,
    allow_unbacked_sharding=False,
)
def linear_reduction_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    input_meta = args_schema[0]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    ndim = len(input_meta.shape)

    dims = None
    if len(args_schema) > 1:
        dims = _infer_reduction_dims(args_schema[1], ndim)

    keep_dim = len(args_schema) > 2 and bool(args_schema[2])
    reduction_op = (
        FOREACH_LINEAR_REDUCTION_OP_MAP[op]
        if op in FOREACH_LINEAR_REDUCTION_OP_MAP
        else LINEAR_REDUCTION_OP_MAP[op]
    )
    return _reduction_single_dim_strategy(
        args_schema,
        reduction_dims=dims,
        keep_dim=keep_dim,
        reduction_linear=True,
        reduction_op=reduction_op,
    )


def _is_spec_evenly_sharded_on_dim(spec: DTensorSpec, dim: int) -> bool:
    shape = spec.shape
    dim = normalize_dim(dim, len(shape))

    num_shards = 1
    for mesh_dim, placement in enumerate(spec.placements):
        if _is_shard_like(placement) and placement.dim == dim:
            num_shards *= spec.mesh.size(mesh_dim)
            if isinstance(placement, _StridedShard):
                num_shards *= placement.split_factor

    return guard_or_false(cast(Any, shape[dim]) % num_shards == 0)


def _mean_full_mesh_strategy_filter(
    mesh: DeviceMesh,
    op_schema: OpSchema,
    input_specs: list[DTensorSpec],
    output_specs: DTensorSpec | tuple[DTensorSpec | None, ...],
) -> bool:
    """Reject mean strategies that would average uneven local means.

    mean uses Partial("avg") for shards on reduction dims. That is only correct
    when every rank contributes the same number of elements; otherwise averaging
    the per-rank means weights small and large shards equally. The single-dim
    rule cannot know the final product of mesh dims sharding a tensor dim, so
    this filter checks the materialized full-mesh input spec.
    """
    input_spec = input_specs[0]
    dims = None
    if len(op_schema.args_schema) > 1:
        dims = _infer_reduction_dims(op_schema.args_schema[1], input_spec.ndim)
    reduce_dims = list(range(input_spec.ndim)) if dims is None else dims

    return all(_is_spec_evenly_sharded_on_dim(input_spec, dim) for dim in reduce_dims)


@register_single_dim_strategy(
    MEAN_REDUCTION_OPS,
    schema_info=RuntimeSchemaInfo(1),
    full_mesh_strategy_filter=_mean_full_mesh_strategy_filter,
)
def mean_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    input_meta = args_schema[0]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    ndim = len(input_meta.shape)

    dims = None
    if len(args_schema) > 1:
        dims = _infer_reduction_dims(args_schema[1], ndim)

    reduce_dims = list(range(ndim)) if dims is None else dims
    keep_dim = len(args_schema) > 2 and bool(args_schema[2])

    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for d in range(ndim):
        if d in reduce_dims:
            strategies.append([Partial("avg"), _ShardingPlaceholder(d)])
        else:
            if keep_dim:
                out_d = d
            else:
                out_d = d - sum(1 for rd in reduce_dims if rd < d)
            strategies.append([_ShardingPlaceholder(out_d), _ShardingPlaceholder(d)])

    strategies.append([Partial("avg"), Partial("avg")])
    return strategies


@register_single_dim_strategy(
    NON_LINEAR_BOOL_REDUCTION_OPS,
    schema_info=RuntimeSchemaInfo(1),
    allow_uneven_sharding=True,
)
def bool_reduction_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    """all/any: only shard on non-reduction dims (no partial propagation)."""
    input_meta = args_schema[0]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    ndim = len(input_meta.shape)

    dims = None
    if len(args_schema) > 1:
        dims = _infer_reduction_dims(args_schema[1], ndim)

    keep_dim = len(args_schema) > 2 and bool(args_schema[2])
    return _reduction_single_dim_strategy(
        args_schema,
        reduction_dims=dims,
        keep_dim=keep_dim,
        reduction_linear=False,
        reduction_op="sum",
    )


def _shard_non_reduction_dim(
    args_schema: tuple[Any, ...],
    dim: int,
    keep_dim: bool,
    n_outputs: int,
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Shared logic for ops that shard on non-reduction dims with dim remapping.

    Used by dim-reduction ops returning (values, indices) and argmax/argmin.

    Args:
        args_schema: Op args with TensorMeta at position 0.
        dim: The reduction dimension.
        keep_dim: Whether the reduction dim is kept.
        n_outputs: Number of output tensors (1 for argmax, 2 for max.dim).
    """
    input_meta = args_schema[0]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")

    ndim = len(input_meta.shape)
    dim = normalize_dim(dim, ndim)

    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for d in range(ndim):
        if d == dim:
            continue
        out_d = d if keep_dim or d < dim else d - 1
        strategies.append(
            [_ShardingPlaceholder(out_d)]  # pyrefly: ignore[bad-argument-type]
            * n_outputs
            + [_ShardingPlaceholder(d)]
        )
    return strategies


# Category B: Dim-reduction ops returning (values, indices)
# max.dim/min.dim return (values, indices). Indices are local to each shard
# and cannot be combined across ranks, so we force Replicate on reduction dims
# (same approach as argmax/argmin).
@register_single_dim_strategy(
    [aten.max.dim, aten.min.dim],
    schema_info=RuntimeSchemaInfo(1),
    allow_uneven_sharding=True,
)
def max_min_dim_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    dim = cast(int, args_schema[1])
    keep_dim = len(args_schema) > 2 and bool(args_schema[2])
    return _shard_non_reduction_dim(args_schema, dim, keep_dim, n_outputs=2)


def _argmax_argmin_reduction_dims(
    args_schema: tuple[Any, ...], kwargs_schema: dict[str, Any], ndim: int
) -> list[int]:
    if "dim" in kwargs_schema:
        dims_arg = kwargs_schema["dim"]
    elif len(args_schema) > 1:
        dims_arg = args_schema[1]
    else:
        dims_arg = None
    dims = _infer_reduction_dims(dims_arg, ndim)
    return list(range(ndim)) if dims is None else dims


def _argmax_argmin_full_mesh_strategy_filter(
    _mesh: DeviceMesh,
    op_schema: OpSchema,
    input_specs: list[DTensorSpec],
    _output_specs: DTensorSpec | tuple[DTensorSpec | None, ...],
) -> bool:
    """
    Filter strategies to match the custom nonlinear-redux handler.

    argminmax_handler computes from the runtime DTensor placements instead of
    applying the normal redistribute_schema. It only materializes Partial inputs
    and reduction-dim shards to Replicate, so sharding propagation must not pick
    a target input spec that would require retargeting those placements to a
    non-reduction shard.
    """
    args_schema = op_schema.args_schema
    input_strategy = args_schema[0]
    if not isinstance(input_strategy, OpStrategy):
        raise AssertionError(f"Expected OpStrategy, got {type(input_strategy)}")

    input_meta = input_strategy.tensor_meta
    if input_meta is None:
        raise AssertionError("Expected input tensor meta")

    reduce_dims = _argmax_argmin_reduction_dims(
        args_schema, op_schema.kwargs_schema, len(input_meta.shape)
    )

    input_spec = input_strategy.strategies[0].output_spec
    target_input_spec = input_specs[0]
    for input_placement, target_placement in zip(
        input_spec.placements, target_input_spec.placements
    ):
        if input_placement.is_partial() or (
            _is_shard_like(input_placement) and input_placement.dim in reduce_dims
        ):
            if not target_placement.is_replicate():
                return False
    return True


@register_single_dim_strategy(
    list(ARGMAX_ARGMIN_OPS.keys()),
    schema_info=RuntimeSchemaInfo(1, ["dim", "keepdim"]),
    allow_uneven_sharding=True,
    allow_unbacked_sharding=False,
    full_mesh_strategy_filter=_argmax_argmin_full_mesh_strategy_filter,
)
def argmax_argmin_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    """argmax/argmin return indices only. Shard on non-reduction dims."""
    input_meta = args_schema[0]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    ndim = len(input_meta.shape)

    reduce_dims = _argmax_argmin_reduction_dims(args_schema, kwargs_schema, ndim)
    keep_dim = (
        bool(kwargs_schema["keepdim"])
        if "keepdim" in kwargs_schema
        else len(args_schema) > 2 and bool(args_schema[2])
    )

    if len(reduce_dims) == ndim:
        return []

    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for d in range(ndim):
        if d in reduce_dims:
            continue
        out_d = d
        if not keep_dim:
            out_d = d - sum(1 for rd in reduce_dims if rd < d)
        strategies.append([_ShardingPlaceholder(out_d), _ShardingPlaceholder(d)])
    return strategies


def _shard_except_dim_strategy(
    n_placements: int,
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
    active_dim: int,
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Build single-dim strategies that shard on every dim except an active dim.

    Used by sort-like ops (sort, topk, cummax, cummin), scan ops (cumsum, cumprod,
    logcumsumexp), and softmax-like ops. All outputs and inputs get the same sharding
    placeholder.

    Args:
        n_placements: Total number of placements per rule (outputs + inputs).
        active_dim: The dim to exclude from sharding (e.g. sort dim, softmax dim).
    """
    input_meta = args_schema[0]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    ndim = len(input_meta.shape)
    active_dim = normalize_dim(active_dim, ndim)
    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for d in range(ndim):
        if d != active_dim:
            strategies.append([_ShardingPlaceholder(d)] * n_placements)
    return strategies


# Category A: Sort-like and scan ops
# Scan: 1 output + 1 input = 2 placements
_SCAN_N_PLACEMENTS = 2
# Sort-like: 2 outputs (values, indices) + 1 input = 3 placements
_SORT_LIKE_N_PLACEMENTS = 3
# Softmax forward: 1 output + 1 input = 2 placements
_SOFTMAX_FWD_N_PLACEMENTS = 2
# Softmax backward: 1 output + 2 inputs = 3 placements
_SOFTMAX_BWD_N_PLACEMENTS = 3


@register_single_dim_strategy(
    [aten.cumsum.default, aten.cumprod.default, aten.logcumsumexp.default],
    schema_info=RuntimeSchemaInfo(1),
    allow_uneven_sharding=True,
    allow_unbacked_sharding=False,
)
def scan_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    dim = args_schema[1]
    if not isinstance(dim, int):
        raise AssertionError(f"Expected int, got {type(dim)}")
    strategies = _shard_except_dim_strategy(
        _SCAN_N_PLACEMENTS, op, args_schema, kwargs_schema, active_dim=dim
    )
    # cumsum is linear for sum/avg: partial sums/avgs propagate through
    if op == aten.cumsum.default:
        strategies.append([Partial("sum"), Partial("sum")])
        strategies.append([Partial("avg"), Partial("avg")])
    return strategies


@register_single_dim_strategy(
    [aten.median.default, aten.nanmedian.default],
    schema_info=RuntimeSchemaInfo(1),
    allow_uneven_sharding=True,
)
def global_median_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Global median reduces all dims — only Replicate is valid."""
    return []


@register_single_dim_strategy(
    [aten.median.dim, aten.nanmedian.dim, aten.mode.default],
    schema_info=RuntimeSchemaInfo(1),
    allow_uneven_sharding=True,
)
def dim_reduction_with_indices_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    dim = cast(int, args_schema[1]) if len(args_schema) > 1 else -1
    keep_dim = len(args_schema) > 2 and bool(args_schema[2])
    return _shard_non_reduction_dim(args_schema, dim, keep_dim, n_outputs=2)


@register_single_dim_strategy(
    [aten.kthvalue.default],
    schema_info=RuntimeSchemaInfo(2),
    allow_uneven_sharding=True,
)
def kthvalue_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    dim = cast(int, args_schema[2]) if len(args_schema) > 2 else -1
    keep_dim = len(args_schema) > 3 and bool(args_schema[3])
    return _shard_non_reduction_dim(args_schema, dim, keep_dim, n_outputs=2)


@register_single_dim_strategy(
    [aten.cummax.default, aten.cummin.default],
    schema_info=RuntimeSchemaInfo(1),
    allow_uneven_sharding=True,
)
def cummax_cummin_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    dim = cast(int, args_schema[1])
    return _shard_except_dim_strategy(
        _SORT_LIKE_N_PLACEMENTS, op, args_schema, kwargs_schema, active_dim=dim
    )


_STD_VAR_OPS = [
    aten.std.correction,
    aten.std.correction_out,
    aten.var.correction,
    aten.var.correction_out,
    aten.var_mean.correction,
    aten.var_mean.correction_out,
    prims.var.default,
]


@register_single_dim_strategy(
    _STD_VAR_OPS,
    schema_info=RuntimeSchemaInfo(1, ["keepdim"]),
    allow_uneven_sharding=True,
    allow_unbacked_sharding=False,
)
def std_var_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    input_meta = args_schema[0]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    ndim = len(input_meta.shape)

    dims = None
    if len(args_schema) > 1:
        dims = _infer_reduction_dims(args_schema[1], ndim)

    keep_dim = cast(bool, kwargs_schema.get("keepdim", False))
    strategies = _reduction_single_dim_strategy(
        args_schema,
        reduction_dims=dims,
        keep_dim=keep_dim,
        reduction_linear=False,
        reduction_op="sum",
    )
    num_outputs = len(op._schema.returns)
    if num_outputs == 1 and not any(
        isinstance(arg, TensorMeta) for arg in kwargs_schema.values()
    ):
        return strategies

    expanded_strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for rule in strategies:
        output_placements = [rule[0]] * num_outputs
        input_placements = rule[1:]
        kwarg_placements: list[Placement | _ShardingPlaceholder] = []
        for arg in kwargs_schema.values():
            if isinstance(arg, TensorMeta):
                kwarg_placements.append(output_placements[len(kwarg_placements)])
        expanded_strategies.append(
            output_placements + input_placements + kwarg_placements
        )
    return expanded_strategies


def _get_norm_reduction_op(norm_type: int | float | str) -> ReductionOpType:
    """Get the reduction op for vector/foreach norm based on norm_type.

    For inf/-inf norms, returns simple reduction ops ("max", "min").
    For other norms (including 0), returns NormReduction which produces the
    appropriate Partial placement via get_placement_from_reduction_op.
    """
    if norm_type in (float("inf"), "inf"):
        return "max"
    elif norm_type in (float("-inf"), "-inf"):
        return "min"
    else:
        if not isinstance(norm_type, (int, float)):
            raise AssertionError
        return NormReduction(norm_type)


@register_single_dim_strategy(
    [aten.linalg_vector_norm.default, aten.norm.Scalar],
    schema_info=RuntimeSchemaInfo(1),
    allow_uneven_sharding=True,
    allow_unbacked_sharding=False,
)
def vector_norm_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Vector norm: linear reduction with NormPartial.

    S(reduction_dim)->_NormPartial(ord) rules appear as false positives in
    strategy_validation because the validator doesn't enumerate _NormPartial
    as an output placement. The rules are correct at runtime.
    TODO: remove this comment when _NormPartial is deprecated.
    """
    input_meta = args_schema[0]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    ndim = len(input_meta.shape)

    norm_type = args_schema[1] if len(args_schema) > 1 else 2
    if not isinstance(norm_type, (int, float, str)):
        raise AssertionError(f"Expected int, float, or str, got {type(norm_type)}")
    dim = args_schema[2] if len(args_schema) > 2 else None
    keepdim = args_schema[3] if len(args_schema) > 3 else False
    dims = _infer_reduction_dims(dim, ndim)

    return _reduction_single_dim_strategy(
        args_schema,
        reduction_dims=dims,
        keep_dim=cast(bool, keepdim),
        reduction_linear=True,
        reduction_op=_get_norm_reduction_op(norm_type),
    )


register_single_dim_strategy(
    [aten._foreach_norm.Scalar],
    schema_info=RuntimeSchemaInfo(1, needs_pytree=True),
    allow_uneven_sharding=True,
    allow_unbacked_sharding=False,
)(vector_norm_single_dim_strategy)


@register_single_dim_strategy(
    [aten.linalg__powsum.default],
    schema_info=RuntimeSchemaInfo(1),
    allow_uneven_sharding=True,
    allow_unbacked_sharding=False,
)
def powsum_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    """linalg__powsum: computes sum(|x|^ord), always reducible with Partial("sum")."""
    input_meta = args_schema[0]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    ndim = len(input_meta.shape)

    dim = args_schema[2] if len(args_schema) > 2 else None
    keepdim = args_schema[3] if len(args_schema) > 3 else False
    dims = _infer_reduction_dims(dim, ndim)

    return _reduction_single_dim_strategy(
        args_schema,
        reduction_dims=dims,
        keep_dim=cast(bool, keepdim),
        reduction_linear=True,
        reduction_op="sum",
    )


register_single_dim_strategy(
    [aten._foreach_powsum.Scalar],
    schema_info=RuntimeSchemaInfo(1, needs_pytree=True),
    allow_uneven_sharding=True,
    allow_unbacked_sharding=False,
)(powsum_single_dim_strategy)


_REPLICATE_ONLY_OPS = [
    aten._linalg_svd.default,
    aten.linalg_qr.default,
    # TODO: The diagonal ops can have an improved sharding strategy for
    # shard placements that does not require redistributing to replicate.
    aten.diagonal_copy.default,
    aten.diag_embed.default,
    aten.diag.default,
    aten.diagonal.default,
    aten.tril.default,
    aten.triu.default,
    aten._linalg_eigh.default,
]


@register_single_dim_strategy(
    _REPLICATE_ONLY_OPS,
    schema_info=RuntimeSchemaInfo(1),
)
def linalg_replicate_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Replicate-only ops: only the all-Replicate strategy is valid."""
    return []


# Maps each pooling op to its spatial rank (number of spatial dimensions).
# Batched inputs have layout (N, C, *spatial) with ndim = spatial_rank + 2;
# unbatched inputs drop the batch dim giving ndim = spatial_rank + 1.
POOL_SPATIAL_RANK: dict[torch._ops.OpOverload, int] = {
    aten.avg_pool1d.default: 1,
    aten.avg_pool2d.default: 2,
    aten.avg_pool3d.default: 3,
    aten.adaptive_avg_pool1d.default: 1,
    aten._adaptive_avg_pool2d.default: 2,
    aten._adaptive_avg_pool3d.default: 3,
    aten.adaptive_max_pool1d.default: 1,
    aten.adaptive_max_pool2d.default: 2,
    aten.adaptive_max_pool3d.default: 3,
    aten.fractional_max_pool2d.default: 2,
    aten.fractional_max_pool3d.default: 3,
    aten.max_pool1d_with_indices.default: 1,
    aten.max_pool2d_with_indices.default: 2,
    aten.max_pool3d_with_indices.default: 3,
}

AVG_POOL_OPS = [
    aten.avg_pool1d.default,
    aten.avg_pool2d.default,
    aten.avg_pool3d.default,
    aten.adaptive_avg_pool1d.default,
    aten._adaptive_avg_pool2d.default,
    aten._adaptive_avg_pool3d.default,
]

MAX_POOL_OPS = [
    aten.adaptive_max_pool1d.default,
    aten.adaptive_max_pool2d.default,
    aten.adaptive_max_pool3d.default,
    aten.fractional_max_pool2d.default,
    aten.fractional_max_pool3d.default,
    aten.max_pool1d_with_indices.default,
    aten.max_pool2d_with_indices.default,
    aten.max_pool3d_with_indices.default,
]


@register_single_dim_strategy(
    AVG_POOL_OPS + MAX_POOL_OPS,
    schema_info=RuntimeSchemaInfo(1),
    allow_uneven_sharding=True,
)
def pooling_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    input_meta = args_schema[0]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")

    ndim = len(input_meta.shape)
    num_outputs = sum(1 for r in op._schema.returns if "Tensor" in str(r.type))
    num_inputs = sum(1 for a in args_schema if isinstance(a, TensorMeta))
    n = num_outputs + num_inputs

    strategies: list[list[Placement | _ShardingPlaceholder]] = []

    # S(0) for batch dim — always valid
    strategies.append([_ShardingPlaceholder(0)] * n)

    # avg_pool is linear: Partial(sum) and Partial(avg) pass through unchanged.
    if op in AVG_POOL_OPS:
        strategies.append([Partial("sum")] * n)
        strategies.append([Partial("avg")] * n)

    # S(1) is safe when dim 1 is the channel dim (pooling never touches it).
    # Batched inputs have layout (N, C, *spatial) with ndim = spatial_rank + 2.
    spatial_rank = POOL_SPATIAL_RANK[op]
    is_batched = ndim >= spatial_rank + 2
    if is_batched:
        strategies.append([_ShardingPlaceholder(1)] * n)

    return strategies


# Category F: Softmax-like ops
@register_single_dim_strategy(
    [aten._log_softmax.default, aten._softmax.default, aten._safe_softmax.default],
    schema_info=RuntimeSchemaInfo(1),
    allow_uneven_sharding=True,
    allow_unbacked_sharding=False,
)
def softmax_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    softmax_dim = cast(int, args_schema[1])
    return _shard_except_dim_strategy(
        _SOFTMAX_FWD_N_PLACEMENTS,
        op,
        args_schema,
        kwargs_schema,
        active_dim=softmax_dim,
    )


@register_single_dim_strategy(
    [
        aten._log_softmax_backward_data.default,
        aten._softmax_backward_data.default,
    ],
    schema_info=RuntimeSchemaInfo(2),
    allow_uneven_sharding=True,
    allow_unbacked_sharding=False,
)
def softmax_backward_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    softmax_dim = cast(int, args_schema[2])
    return _shard_except_dim_strategy(
        _SOFTMAX_BWD_N_PLACEMENTS,
        op,
        args_schema,
        kwargs_schema,
        active_dim=softmax_dim,
    )


@register_single_dim_strategy(
    [aten.nll_loss_forward.default, aten.nll_loss2d_forward.default],
    schema_info=RuntimeSchemaInfo(3),
)
def nll_loss_forward_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    if len(args_schema) != 5:
        raise AssertionError(f"Expected 5 args, got {len(args_schema)}")

    (
        input_meta,
        target_meta,
        weight_meta,
        reduction,
        _,
    ) = args_schema
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    if not isinstance(target_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(target_meta)}")
    if weight_meta is not None and not isinstance(weight_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(weight_meta)}")
    reduction = cast(int, reduction)

    input_shape = input_meta.shape
    channel_dim = 1 if len(input_shape) >= 2 else 0
    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    # weight, if given, has size input_shape[channel_dim], so replicate it.
    weight_placement = [Replicate()] if weight_meta is not None else []

    for input_dim in range(len(input_shape)):
        # Input must be replicated along the channel dim.
        if input_dim == channel_dim:
            continue
        # Target has no channel dim and follows input on the other dims.
        target_dim = input_dim if input_dim < channel_dim else input_dim - 1
        input_shard = _ShardingPlaceholder(input_dim)
        target_shard = _ShardingPlaceholder(target_dim)
        if reduction == Reduction.NONE.value:
            # With reduction="none", output follows target and total_weight is
            # a replicated scalar.
            strategies.append(
                [
                    target_shard,
                    Replicate(),
                    input_shard,
                    target_shard,
                    *weight_placement,
                ]
            )
        elif reduction == Reduction.SUM.value:
            # The output reduces across all target dims. total_weight is summed
            # across ranks if not replicated.
            strategies.append(
                [
                    Partial("sum"),
                    Partial("sum"),
                    input_shard,
                    target_shard,
                    *weight_placement,
                ]
            )

    return strategies


@register_single_dim_strategy(
    [aten.nll_loss_backward.default, aten.nll_loss2d_backward.default],
    schema_info=RuntimeSchemaInfo(4),
)
def nll_loss_backward_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    if len(args_schema) != 7:
        raise AssertionError(f"Expected 7 args, got {len(args_schema)}")
    (
        grad_out_meta,
        input_meta,
        target_meta,
        weight_meta,
        reduction,
        _,
        total_weight_meta,
    ) = args_schema
    if not isinstance(grad_out_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(grad_out_meta)}")
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    if not isinstance(target_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(target_meta)}")
    if weight_meta is not None and not isinstance(weight_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(weight_meta)}")
    if not isinstance(total_weight_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(total_weight_meta)}")
    reduction = cast(int, reduction)

    input_shape = input_meta.shape
    channel_dim = 1 if len(input_shape) >= 2 else 0
    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    # weight, if given, has size input_shape[channel_dim], so replicate it.
    weight_placement = [Replicate()] if weight_meta is not None else []

    for input_dim in range(len(input_shape)):
        # Input must be replicated along the channel dim.
        if input_dim == channel_dim:
            continue
        # Target has no channel dim and follows input on the other dims.
        target_dim = input_dim if input_dim < channel_dim else input_dim - 1
        input_shard = _ShardingPlaceholder(input_dim)
        target_shard = _ShardingPlaceholder(target_dim)

        if reduction == Reduction.NONE.value:
            # grad_out follows target when there is no reduction; total_weight
            # is unused by the kernel, so keep it replicated here.
            strategies.append(
                [
                    input_shard,
                    target_shard,
                    input_shard,
                    target_shard,
                    *weight_placement,
                    Replicate(),
                ]
            )
        else:
            # grad_out is a replicated scalar for reduced outputs.
            total_weight_placements: list[Placement] = [Replicate()]
            if reduction != Reduction.MEAN.value:
                # total_weight is only used for reduction="mean". For sum it is
                # unused, so a partial placement is also valid and avoids
                # unnecessary redistribution.
                total_weight_placements.append(Partial("sum"))

            for total_weight_placement in total_weight_placements:
                strategies.append(
                    [
                        input_shard,
                        Replicate(),
                        input_shard,
                        target_shard,
                        *weight_placement,
                        total_weight_placement,
                    ]
                )

    if reduction != Reduction.MEAN.value:
        # When total_weight is unused, also allow a replicated input/output
        # strategy that preserves Partial("sum") total_weight.
        strategies.append(
            [
                Replicate(),
                Replicate(),
                Replicate(),
                Replicate(),
                *weight_placement,
                Partial("sum"),
            ]
        )

    return strategies


@register_single_dim_strategy(
    [aten.native_layer_norm.default],
    schema_info=RuntimeSchemaInfo(1),
    allow_uneven_sharding=True,
)
def layer_norm_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    input_meta = args_schema[0]
    normalized_shape = args_schema[1]
    weight_meta = args_schema[2]
    bias_meta = args_schema[3]

    axis = len(input_meta.shape) - len(normalize_to_torch_size(normalized_shape))

    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for dim in range(axis):
        # [out, mean, rstd, input, weight?, bias?]
        rule: list[Placement | _ShardingPlaceholder] = [
            _ShardingPlaceholder(dim),  # out
            _ShardingPlaceholder(dim),  # mean
            _ShardingPlaceholder(dim),  # rstd
            _ShardingPlaceholder(dim),  # input
        ]
        if weight_meta is not None:
            rule.append(Replicate())
        if bias_meta is not None:
            rule.append(Replicate())
        strategies.append(rule)
    return strategies


@register_single_dim_strategy(
    [aten._fused_rms_norm.default],
    schema_info=RuntimeSchemaInfo(1),
    allow_uneven_sharding=True,
)
def rms_norm_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    input_meta = args_schema[0]
    normalized_shape = args_schema[1]
    weight_meta = args_schema[2]

    axis = len(input_meta.shape) - len(normalize_to_torch_size(normalized_shape))

    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for dim in range(axis):
        # [out, rrms, input, weight?]
        rule: list[Placement | _ShardingPlaceholder] = [
            _ShardingPlaceholder(dim),  # out
            _ShardingPlaceholder(dim),  # rrms
            _ShardingPlaceholder(dim),  # input
        ]
        if weight_meta is not None:
            rule.append(Replicate())
        strategies.append(rule)
    return strategies


@register_single_dim_strategy(
    [aten.native_layer_norm_backward.default],
    schema_info=RuntimeSchemaInfo(2),
)
def layer_norm_bwd_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder | None]]:
    input_meta = args_schema[1]
    normalized_shape = args_schema[2]
    # mean = args_schema[3], rstd = args_schema[4]
    weight_meta = args_schema[5]
    bias_meta = args_schema[6]
    output_mask = args_schema[7]
    compute_d_input = output_mask[0]
    compute_d_weight = weight_meta is not None and output_mask[1]
    compute_d_bias = bias_meta is not None and output_mask[2]

    axis = len(input_meta.shape) - len(normalize_to_torch_size(normalized_shape))

    strategies: list[list[Placement | _ShardingPlaceholder | None]] = []
    for dim in range(axis):
        # outputs: [d_input, d_weight, d_bias] — always 3 per schema
        # Masked-off outputs use None even if the corresponding input exists.
        rule: list[Placement | _ShardingPlaceholder | None] = [
            _ShardingPlaceholder(dim) if compute_d_input else None,
            Partial("sum") if compute_d_weight else None,
            Partial("sum") if compute_d_bias else None,
        ]
        # inputs: [grad_out, input, mean, rstd, weight?, bias?]
        rule.extend(
            [
                _ShardingPlaceholder(dim),  # grad_out
                _ShardingPlaceholder(dim),  # input
                _ShardingPlaceholder(dim),  # mean
                _ShardingPlaceholder(dim),  # rstd
            ]
        )
        if weight_meta is not None:
            rule.append(Replicate())
        if bias_meta is not None:
            rule.append(Replicate())
        strategies.append(rule)

    return strategies


@register_single_dim_strategy(
    [aten._fused_rms_norm_backward.default],
    schema_info=RuntimeSchemaInfo(2),
)
def rms_norm_bwd_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder | None]]:
    input_meta = args_schema[1]
    normalized_shape = args_schema[2]
    # rstd = args_schema[3]
    weight_meta = args_schema[4]
    output_mask = args_schema[5]
    compute_d_input = output_mask[0]
    compute_d_weight = weight_meta is not None and output_mask[1]

    axis = len(input_meta.shape) - len(normalize_to_torch_size(normalized_shape))

    strategies: list[list[Placement | _ShardingPlaceholder | None]] = []
    for dim in range(axis):
        # outputs: [d_input, d_weight] — always 2 per schema
        # Masked-off outputs use None even if the corresponding input exists.
        # inputs: [grad_out, input, rstd, weight?]
        rule: list[Placement | _ShardingPlaceholder | None] = [
            _ShardingPlaceholder(dim) if compute_d_input else None,
            Partial("sum") if compute_d_weight else None,
            _ShardingPlaceholder(dim),  # grad_out
            _ShardingPlaceholder(dim),  # input
            _ShardingPlaceholder(dim),  # rstd
        ]
        if weight_meta is not None:
            rule.append(Replicate())
        strategies.append(rule)

    return strategies


@register_single_dim_strategy(
    [aten.topk.default],
    schema_info=RuntimeSchemaInfo(2),
    allow_uneven_sharding=True,
)
def topk_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    topk_dim = cast(int, args_schema[2]) if len(args_schema) > 2 else -1
    return _shard_except_dim_strategy(
        _SORT_LIKE_N_PLACEMENTS, op, args_schema, kwargs_schema, active_dim=topk_dim
    )


@register_single_dim_strategy(
    aten.sort.default,
    schema_info=RuntimeSchemaInfo(1),
    allow_uneven_sharding=True,
)
def sort_default_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    sort_dim = cast(int, args_schema[1]) if len(args_schema) > 1 else -1
    return _shard_except_dim_strategy(
        _SORT_LIKE_N_PLACEMENTS, op, args_schema, kwargs_schema, active_dim=sort_dim
    )


@register_single_dim_strategy(
    aten.sort.stable,
    schema_info=RuntimeSchemaInfo(
        1,
        static_kwargkey=["dim", "descending", "stable"],
    ),
    allow_uneven_sharding=True,
)
def sort_stable_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    sort_dim = cast(int, kwargs_schema.get("dim", -1))
    return _shard_except_dim_strategy(
        _SORT_LIKE_N_PLACEMENTS, op, args_schema, kwargs_schema, active_dim=sort_dim
    )


@register_single_dim_strategy(
    [aten.histc.default],
    schema_info=RuntimeSchemaInfo(2),
    allow_uneven_sharding=True,
)
def histc_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    input_meta = args_schema[0]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    ndim = len(input_meta.shape)

    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    # histc supports sharded input -> partial output only when min/max are specified
    if len(args_schema) == 4:
        for dim in range(ndim):
            strategies.append([Partial(), _ShardingPlaceholder(dim)])
    return strategies


@register_single_dim_strategy(
    [aten.logsumexp.default],
    schema_info=RuntimeSchemaInfo(
        static_argnum=1,
        static_kwargkey=["keepdim"],
    ),
    allow_uneven_sharding=True,
)
def logsumexp_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    input_meta = args_schema[0]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    ndim = len(input_meta.shape)

    dims = _infer_reduction_dims(args_schema[1], ndim)
    keep_dim = cast(bool, kwargs_schema.get("keepdim", False))
    return _reduction_single_dim_strategy(
        args_schema,
        reduction_dims=dims,
        keep_dim=keep_dim,
        reduction_linear=False,
        reduction_op="sum",
    )


_LINALG_NUM_PLACEMENTS = {
    # 1 in 1 out
    aten.cholesky_inverse.default: 2,
    aten.linalg_matrix_exp.default: 2,
    # 2 in 1 out
    aten.cholesky_solve.default: 3,
    aten.linalg_householder_product.default: 3,
    aten.linalg_solve_triangular.default: 3,
    # 3 in 1 out
    aten.linalg_ldl_solve.default: 4,
    aten.linalg_lu_solve.default: 4,
    aten.ormqr.default: 4,
    # 1 in 2 out
    aten.geqrf.default: 3,
    aten.linalg_cholesky_ex.default: 3,
    aten.linalg_eig.default: 3,
    aten.linalg_inv_ex.default: 3,
    # 2 in 2 out
    aten.triangular_solve.default: 4,
    # 1 in 3 out
    aten._linalg_det.default: 4,
    aten.linalg_ldl_factor_ex.default: 4,
    aten.linalg_lu.default: 4,
    aten.linalg_lu_factor_ex.default: 4,
    # 2 in 3 out
    aten.lu_unpack.default: 5,
    # 1 in 4 out
    aten._linalg_slogdet.default: 5,
    # 2 in 4 out
    aten._linalg_solve_ex.default: 6,
    # 1 in
    aten._linalg_check_errors.default: 1,
}


def _linalg_batch_dim_strategies(
    ndim: int, n_placements: int
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Build single-dim strategies for linalg ops that operate on the last 1-2 dims.

    Returns sharding on each batch dim (all dims except the last 2), with all
    outputs and inputs sharded on the same dim.
    """
    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for dim in range(ndim - 2):
        strategies.append([_ShardingPlaceholder(dim)] * n_placements)
    return strategies


def _get_ndim(tensor_meta: Any) -> int:
    if not isinstance(tensor_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(tensor_meta)}")
    return len(tensor_meta.shape)


@register_single_dim_strategy(
    [
        aten.cholesky_inverse.default,
        aten.linalg_matrix_exp.default,
        aten.cholesky_solve.default,
        aten.linalg_householder_product.default,
        aten.linalg_solve_triangular.default,
        aten.linalg_ldl_solve.default,
        aten.linalg_lu_solve.default,
        aten.ormqr.default,
        aten.geqrf.default,
        aten.linalg_cholesky_ex.default,
        aten.linalg_eig.default,
        aten.linalg_inv_ex.default,
        aten.triangular_solve.default,
        aten._linalg_det.default,
        aten.linalg_ldl_factor_ex.default,
        aten.linalg_lu.default,
        aten.linalg_lu_factor_ex.default,
        aten.lu_unpack.default,
        aten._linalg_slogdet.default,
        aten._linalg_solve_ex.default,
        aten._linalg_check_errors.default,
    ],
    schema_info=RuntimeSchemaInfo(1),
    allow_uneven_sharding=True,
)
def linalg_batch_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    ndim = _get_ndim(args_schema[0])
    if op not in _LINALG_NUM_PLACEMENTS:
        raise AssertionError(f"Expected op in _LINALG_NUM_PLACEMENTS, got {op}")

    n_placements = _LINALG_NUM_PLACEMENTS[op]
    strategies = _linalg_batch_dim_strategies(ndim, n_placements=n_placements)

    if op == aten.linalg_solve_triangular.default:
        # solve_triangular(A, B) -> result: linear in B
        strategies.append([Partial(), Replicate(), Partial()])
        strategies.append([Partial("avg"), Replicate(), Partial("avg")])
        # A replicated, B sharded on batch dims (B may have more batch dims than A)
        ndim_b = _get_ndim(args_schema[1])
        for dim in range(ndim_b - 2):
            strategies.append(
                [_ShardingPlaceholder(dim), Replicate(), _ShardingPlaceholder(dim)]
            )
    elif op == aten.cholesky_solve.default:
        # cholesky_solve(B, A) -> result  (B is arg0)
        strategies.append([Partial(), Partial(), Replicate()])
    elif op == aten.linalg_lu_solve.default:
        # linalg_lu_solve(LU, pivots, B) -> result
        strategies.append([Partial(), Replicate(), Replicate(), Partial()])
    elif op == aten.linalg_ldl_solve.default:
        # linalg_ldl_solve(LD, pivots, B) -> result
        strategies.append([Partial(), Replicate(), Replicate(), Partial()])
    elif op == aten.ormqr.default:
        # ormqr(a, tau, C) -> result  (linear in C)
        strategies.append([Partial(), Replicate(), Replicate(), Partial()])
    elif op == aten._linalg_solve_ex.default:
        # _linalg_solve_ex(A, B) -> (result, LU, pivots, info)
        strategies.append(
            [Partial(), Replicate(), Replicate(), Replicate(), Replicate(), Partial()]
        )

    return strategies


# linalg_pinv has optional tensor kwargs atol, rtol (scalar tensors when present).
# Schema: (Tensor self, *, Tensor? atol=None, Tensor? rtol=None, bool hermitian) -> Tensor
# When atol/rtol are None, num_inputs=1; when present, they add to num_inputs.
@register_single_dim_strategy(
    [aten.linalg_pinv.atol_rtol_tensor],
    schema_info=RuntimeSchemaInfo(1),
    allow_uneven_sharding=True,
)
def linalg_pinv_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    ndim = _get_ndim(args_schema[0])
    # Count optional tensor kwargs that are actually present
    extra_tensors = sum(
        isinstance(kwargs_schema.get(k), TensorMeta) for k in ("atol", "rtol")
    )
    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for dim in range(ndim - 2):
        s: list[Placement | _ShardingPlaceholder] = [
            _ShardingPlaceholder(dim),
            _ShardingPlaceholder(dim),
        ]
        # atol, rtol are scalar tensors — always Replicate
        s.extend([Replicate()] * extra_tensors)
        strategies.append(s)
    return strategies


# linalg_cross is pointwise on every dim except the cross-product dim (which
# must be size 3).  Shard on any other dim.
@register_single_dim_strategy(
    [aten.linalg_cross.default],
    schema_info=RuntimeSchemaInfo(1),
    allow_uneven_sharding=True,
)
def linalg_cross_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    ndim = _get_ndim(args_schema[0])
    cross_dim = kwargs_schema.get("dim", -1) % ndim
    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for dim in range(ndim):
        if dim == cross_dim:
            continue
        strategies.append([_ShardingPlaceholder(dim)] * 3)
    return strategies


# ---------------------------------------------------------------------------
# Interpolation / upsample / pooling ops
#
# These ops operate on spatial dims and are safely shardable on batch (dim 0)
# and channel (dim 1). grid_sampler is batch-only because the grid tensor has
# no channel dimension.
# ---------------------------------------------------------------------------


@register_single_dim_strategy(
    [
        # Forward ops
        aten.upsample_nearest1d.default,
        aten.upsample_nearest2d.default,
        aten.upsample_nearest3d.default,
        aten._upsample_nearest_exact1d.default,
        aten._upsample_nearest_exact2d.default,
        aten._upsample_nearest_exact3d.default,
        aten._upsample_bilinear2d_aa.default,
        aten._upsample_bicubic2d_aa.default,
        aten._upsample_lanczos2d_aa.default,
        aten.upsample_bicubic2d.default,
        aten.upsample_bilinear2d.default,
        aten.upsample_linear1d.default,
        aten.upsample_trilinear3d.default,
        # Backward ops
        aten.upsample_nearest1d_backward.default,
        aten.upsample_nearest2d_backward.default,
        aten.upsample_nearest3d_backward.default,
        aten._upsample_nearest_exact1d_backward.default,
        aten._upsample_nearest_exact2d_backward.default,
        aten._upsample_nearest_exact3d_backward.default,
        aten._upsample_bilinear2d_aa_backward.default,
        aten._upsample_bicubic2d_aa_backward.default,
        aten._upsample_lanczos2d_aa_backward.default,
        aten.upsample_bicubic2d_backward.default,
        aten.upsample_bilinear2d_backward.default,
        aten.upsample_linear1d_backward.default,
        aten.upsample_trilinear3d_backward.default,
    ],
    schema_info=RuntimeSchemaInfo(1),
)
def interp_upsample_1out_1in_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    # 1 output + 1 input = 2 placements; shard on batch (0) and channel (1)
    # Upsample is a linear transformation so Partial(sum/avg) is valid.
    return [
        [_ShardingPlaceholder(0)] * 2,
        [_ShardingPlaceholder(1)] * 2,
        [Partial("sum"), Partial("sum")],
        [Partial("avg"), Partial("avg")],
    ]


@register_single_dim_strategy(
    [
        aten.max_unpool2d.default,
        aten.max_unpool3d.default,
        aten._adaptive_avg_pool2d_backward.default,
    ],
    schema_info=RuntimeSchemaInfo(1),
)
def interp_pool_1out_2in_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    # 1 output + 2 inputs = 3 placements; shard on batch (0) and channel (1)
    return [
        [_ShardingPlaceholder(0)] * 3,
        [_ShardingPlaceholder(1)] * 3,
    ]


@register_single_dim_strategy(
    [aten.max_pool2d_with_indices_backward.default],
    schema_info=RuntimeSchemaInfo(1),
)
def pool_backward_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    # max_pool2d_with_indices_backward(grad_output, self, ..., indices) -> grad_input
    # 1 output + 3 tensor inputs = 4 placements
    # Order: [output, grad_output, self, indices]
    input_meta = cast(TensorMeta, args_schema[0])
    strategies: list[list[Placement | _ShardingPlaceholder]] = [
        [_ShardingPlaceholder(0)] * 4,
    ]
    if len(input_meta.shape) >= 4:  # batched: (N, C, H, W)
        strategies.append([_ShardingPlaceholder(1)] * 4)
    # The backward is linear in grad_output, so P(sum/avg) pass through.
    # indices must be replicated (integer positions, not reducible).
    # self is only used for shape, so replicate it too.
    r = Replicate()
    for reduce_op in ("sum", "avg"):
