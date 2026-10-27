            return [self] if inplace else [self.copy()]

        if is_re(to_replace) and self.dtype not in [object, "string"]:
            # only object or string dtype can hold strings, and a regex object
            # will only match strings
            return [self.copy(deep=False)]

        if not (
            self._can_hold_element(value) or (self.dtype == "string" and is_re(value))
        ):
            block = self.astype(np.dtype(object))
        else:
            block = self._maybe_copy(using_cow, inplace)

        rx = re.compile(to_replace)

        replace_regex(block.values, rx, value, mask)

        if (
            inplace
            and warn_copy_on_write()
            and already_warned is not None
            and not already_warned.warned_already
        ):
            if self.refs.has_reference():
                warnings.warn(
                    COW_WARNING_GENERAL_MSG,
                    FutureWarning,
                    stacklevel=find_stack_level(),
                )
                already_warned.warned_already = True

        nbs = block.convert(
            copy=False,
            using_cow=using_cow,
            convert_string=convert_string or self.dtype == "string",
        )
        opt = get_option("future.no_silent_downcasting")
        if (
            len(nbs) > 1
            or (
                nbs[0].dtype != block.dtype
                and not (self.dtype == "string" and nbs[0].dtype == "string")
            )
        ) and not opt:
            warnings.warn(
                # GH#54710
                "Downcasting behavior in `replace` is deprecated and "
                "will be removed in a future version. To retain the old "
                "behavior, explicitly call `result.infer_objects(copy=False)`. "
                "To opt-in to the future "
                "behavior, set "
                "`pd.set_option('future.no_silent_downcasting', True)`",
                FutureWarning,
                stacklevel=find_stack_level(),
            )
        return nbs

    @final
    def replace_list(
        self,
        src_list: Iterable[Any],
        dest_list: Sequence[Any],
        inplace: bool = False,
        regex: bool = False,
        using_cow: bool = False,
        already_warned=None,
    ) -> list[Block]:
        """
        See BlockManager.replace_list docstring.
        """
        values = self.values

        if isinstance(values, Categorical):
            # TODO: avoid special-casing
            # GH49404
            blk = self._maybe_copy(using_cow, inplace)
            values = cast(Categorical, blk.values)
            values._replace(to_replace=src_list, value=dest_list, inplace=True)
            return [blk]

        convert_string = self.dtype == "string"

        # Exclude anything that we know we won't contain
        pairs = [
            (x, y)
            for x, y in zip(src_list, dest_list)
            if (self._can_hold_element(x) or (self.dtype == "string" and is_re(x)))
        ]
        if not len(pairs):
            if using_cow:
                return [self.copy(deep=False)]
            # shortcut, nothing to replace
            return [self] if inplace else [self.copy()]

        src_len = len(pairs) - 1

        if is_string_dtype(values.dtype):
            # Calculate the mask once, prior to the call of comp
            # in order to avoid repeating the same computations
            na_mask = ~isna(values)
            masks: Iterable[npt.NDArray[np.bool_]] = (
                extract_bool_array(
                    cast(
                        ArrayLike,
                        compare_or_regex_search(
                            values, s[0], regex=regex, mask=na_mask
                        ),
                    )
                )
                for s in pairs
            )
        else:
            # GH#38086 faster if we know we dont need to check for regex
            masks = (missing.mask_missing(values, s[0]) for s in pairs)
        # Materialize if inplace = True, since the masks can change
        # as we replace
        if inplace:
            masks = list(masks)

        if using_cow:
            # Don't set up refs here, otherwise we will think that we have
            # references when we check again later
            rb = [self]
        else:
            rb = [self if inplace else self.copy()]

        if (
            inplace
            and warn_copy_on_write()
            and already_warned is not None
            and not already_warned.warned_already
        ):
            if self.refs.has_reference():
                warnings.warn(
                    COW_WARNING_GENERAL_MSG,
                    FutureWarning,
                    stacklevel=find_stack_level(),
                )
                already_warned.warned_already = True

        opt = get_option("future.no_silent_downcasting")
        for i, ((src, dest), mask) in enumerate(zip(pairs, masks)):
            convert = i == src_len  # only convert once at the end
            new_rb: list[Block] = []

            # GH-39338: _replace_coerce can split a block into
            # single-column blocks, so track the index so we know
            # where to index into the mask
            for blk_num, blk in enumerate(rb):
                if len(rb) == 1:
                    m = mask
                else:
                    mib = mask
                    assert not isinstance(mib, bool)
                    m = mib[blk_num : blk_num + 1]

                # error: Argument "mask" to "_replace_coerce" of "Block" has
                # incompatible type "Union[ExtensionArray, ndarray[Any, Any], bool]";
                # expected "ndarray[Any, dtype[bool_]]"
                result = blk._replace_coerce(
                    to_replace=src,
                    value=dest,
                    mask=m,
                    inplace=inplace,
                    regex=regex,
                    using_cow=using_cow,
                    convert_string=convert_string,
                )

                if using_cow and i != src_len:
                    # This is ugly, but we have to get rid of intermediate refs
                    # that did not go out of scope yet, otherwise we will trigger
                    # many unnecessary copies
                    for b in result:
                        ref = weakref.ref(b)
                        b.refs.referenced_blocks.pop(
                            b.refs.referenced_blocks.index(ref)
                        )

                if (
                    not opt
                    and convert
                    and blk.is_object
                    and not all(x is None for x in dest_list)
                ):
                    # GH#44498 avoid unwanted cast-back
                    nbs = []
                    for res_blk in result:
                        converted = res_blk.convert(
                            copy=True and not using_cow,
                            using_cow=using_cow,
                            convert_string=convert_string,
                        )
                        if len(converted) > 1 or converted[0].dtype != res_blk.dtype:
                            warnings.warn(
                                # GH#54710
                                "Downcasting behavior in `replace` is deprecated "
                                "and will be removed in a future version. To "
                                "retain the old behavior, explicitly call "
                                "`result.infer_objects(copy=False)`. "
                                "To opt-in to the future "
                                "behavior, set "
                                "`pd.set_option('future.no_silent_downcasting', True)`",
                                FutureWarning,
                                stacklevel=find_stack_level(),
                            )
                        nbs.extend(converted)
                    result = nbs
                new_rb.extend(result)
            rb = new_rb
        return rb

    @final
    def _replace_coerce(
        self,
        to_replace,
        value,
        mask: npt.NDArray[np.bool_],
        inplace: bool = True,
        regex: bool = False,
        using_cow: bool = False,
        convert_string: bool = True,
    ) -> list[Block]:
        """
        Replace value corresponding to the given boolean array with another
        value.

        Parameters
        ----------
        to_replace : object or pattern
            Scalar to replace or regular expression to match.
        value : object
            Replacement object.
        mask : np.ndarray[bool]
            True indicate corresponding element is ignored.
        inplace : bool, default True
            Perform inplace modification.
        regex : bool, default False
            If true, perform regular expression substitution.

        Returns
        -------
        List[Block]
        """
        if should_use_regex(regex, to_replace):
            return self._replace_regex(
                to_replace,
                value,
                inplace=inplace,
                mask=mask,
                using_cow=using_cow,
                convert_string=convert_string,
            )
        else:
            if value is None:
                # gh-45601, gh-45836, gh-46634
                if mask.any():
                    has_ref = self.refs.has_reference()
                    nb = self.astype(np.dtype(object), copy=False, using_cow=using_cow)
                    if (nb is self or using_cow) and not inplace:
                        nb = nb.copy()
                    elif inplace and has_ref and nb.refs.has_reference() and using_cow:
                        # no copy in astype and we had refs before
                        nb = nb.copy()
                    putmask_inplace(nb.values, mask, value)
                    return [nb]
                if using_cow:
                    return [self.copy(deep=False)]
                return [self] if inplace else [self.copy()]
            return self.replace(
                to_replace=to_replace,
                value=value,
                inplace=inplace,
                mask=mask,
                using_cow=using_cow,
                convert_string=convert_string,
            )

    # ---------------------------------------------------------------------
    # 2D Methods - Shared by NumpyBlock and NDArrayBackedExtensionBlock
    #  but not ExtensionBlock

    def _maybe_squeeze_arg(self, arg: np.ndarray) -> np.ndarray:
        """
        For compatibility with 1D-only ExtensionArrays.
        """
        return arg

    def _unwrap_setitem_indexer(self, indexer):
        """
        For compatibility with 1D-only ExtensionArrays.
        """
        return indexer

    # NB: this cannot be made cache_readonly because in mgr.set_values we pin
    #  new .values that can have different shape GH#42631
    @property
    def shape(self) -> Shape:
        return self.values.shape

    def iget(self, i: int | tuple[int, int] | tuple[slice, int]) -> np.ndarray:
        # In the case where we have a tuple[slice, int], the slice will always
        #  be slice(None)
        # Note: only reached with self.ndim == 2
        # Invalid index type "Union[int, Tuple[int, int], Tuple[slice, int]]"
        # for "Union[ndarray[Any, Any], ExtensionArray]"; expected type
        # "Union[int, integer[Any]]"
        return self.values[i]  # type: ignore[index]

    def _slice(
        self, slicer: slice | npt.NDArray[np.bool_] | npt.NDArray[np.intp]
    ) -> ArrayLike:
        """return a slice of my values"""

        return self.values[slicer]

    def set_inplace(self, locs, values: ArrayLike, copy: bool = False) -> None:
        """
        Modify block values in-place with new item value.

        If copy=True, first copy the underlying values in place before modifying
        (for Copy-on-Write).

        Notes
        -----
        `set_inplace` never creates a new array or new Block, whereas `setitem`
        _may_ create a new array and always creates a new Block.

        Caller is responsible for checking values.dtype == self.dtype.
        """
        if copy:
            self.values = self.values.copy()
        self.values[locs] = values

    @final
    def take_nd(
        self,
        indexer: npt.NDArray[np.intp],
        axis: AxisInt,
        new_mgr_locs: BlockPlacement | None = None,
        fill_value=lib.no_default,
    ) -> Block:
        """
        Take values according to indexer and return them as a block.
        """
        values = self.values

        if fill_value is lib.no_default:
            fill_value = self.fill_value
            allow_fill = False
        else:
            allow_fill = True

        # Note: algos.take_nd has upcast logic similar to coerce_to_target_dtype
        new_values = algos.take_nd(
            values, indexer, axis=axis, allow_fill=allow_fill, fill_value=fill_value
        )

        # Called from three places in managers, all of which satisfy
        #  these assertions
        if isinstance(self, ExtensionBlock):
            # NB: in this case, the 'axis' kwarg will be ignored in the
            #  algos.take_nd call above.
            assert not (self.ndim == 1 and new_mgr_locs is None)
        assert not (axis == 0 and new_mgr_locs is None)

        if new_mgr_locs is None:
            new_mgr_locs = self._mgr_locs

        if new_values.dtype != self.dtype:
            return self.make_block(new_values, new_mgr_locs)
        else:
            return self.make_block_same_class(new_values, new_mgr_locs)

    def _unstack(
        self,
        unstacker,
        fill_value,
        new_placement: npt.NDArray[np.intp],
        needs_masking: npt.NDArray[np.bool_],
    ):
        """
        Return a list of unstacked blocks of self

        Parameters
        ----------
        unstacker : reshape._Unstacker
        fill_value : int
            Only used in ExtensionBlock._unstack
        new_placement : np.ndarray[np.intp]
        allow_fill : bool
        needs_masking : np.ndarray[bool]

        Returns
        -------
        blocks : list of Block
            New blocks of unstacked values.
        mask : array-like of bool
            The mask of columns of `blocks` we should keep.
        """
        new_values, mask = unstacker.get_new_values(
            self.values.T, fill_value=fill_value
        )

        mask = mask.any(0)
        # TODO: in all tests we have mask.all(); can we rely on that?

        # Note: these next two lines ensure that
        #  mask.sum() == sum(len(nb.mgr_locs) for nb in blocks)
        #  which the calling function needs in order to pass verify_integrity=False
        #  to the BlockManager constructor
        new_values = new_values.T[mask]
        new_placement = new_placement[mask]

        bp = BlockPlacement(new_placement)
        blocks = [new_block_2d(new_values, placement=bp)]
        return blocks, mask

    # ---------------------------------------------------------------------

    def setitem(self, indexer, value, using_cow: bool = False) -> Block:
        """
        Attempt self.values[indexer] = value, possibly creating a new array.

        Parameters
        ----------
        indexer : tuple, list-like, array-like, slice, int
            The subset of self.values to set
        value : object
            The value being set
        using_cow: bool, default False
            Signaling if CoW is used.

        Returns
        -------
        Block

        Notes
        -----
        `indexer` is a direct slice/positional indexer. `value` must
        be a compatible shape.
        """

        value = self._standardize_fill_value(value)

        values = cast(np.ndarray, self.values)
        if self.ndim == 2:
            values = values.T

        # length checking
        check_setitem_lengths(indexer, value, values)

        if self.dtype != _dtype_obj:
            # GH48933: extract_array would convert a pd.Series value to np.ndarray
            value = extract_array(value, extract_numpy=True)
        try:
            casted = np_can_hold_element(values.dtype, value)
        except LossySetitemError:
            # current dtype cannot store value, coerce to common dtype
            nb = self.coerce_to_target_dtype(value, warn_on_upcast=True)
            return nb.setitem(indexer, value)
        else:
            if self.dtype == _dtype_obj:
                # TODO: avoid having to construct values[indexer]
                vi = values[indexer]
                if lib.is_list_like(vi):
                    # checking lib.is_scalar here fails on
                    #  test_iloc_setitem_custom_object
                    casted = setitem_datetimelike_compat(values, len(vi), casted)

            self = self._maybe_copy(using_cow, inplace=True)
            values = cast(np.ndarray, self.values.T)
            if isinstance(casted, np.ndarray) and casted.ndim == 1 and len(casted) == 1:
                # NumPy 1.25 deprecation: https://github.com/numpy/numpy/pull/10615
                casted = casted[0, ...]
            try:
                values[indexer] = casted
            except (TypeError, ValueError) as err:
                if is_list_like(casted):
                    raise ValueError(
                        "setting an array element with a sequence."
                    ) from err
                raise
        return self

    def putmask(
        self, mask, new, using_cow: bool = False, already_warned=None
    ) -> list[Block]:
        """
        putmask the data to the block; it is possible that we may create a
        new dtype of block

        Return the resulting block(s).

        Parameters
        ----------
        mask : np.ndarray[bool], SparseArray[bool], or BooleanArray
        new : a ndarray/object
        using_cow: bool, default False

        Returns
        -------
        List[Block]
        """
        orig_mask = mask
        values = cast(np.ndarray, self.values)
        mask, noop = validate_putmask(values.T, mask)
        assert not isinstance(new, (ABCIndex, ABCSeries, ABCDataFrame))

        if new is lib.no_default:
            new = self.fill_value

        new = self._standardize_fill_value(new)
        new = extract_array(new, extract_numpy=True)

        if noop:
            if using_cow:
                return [self.copy(deep=False)]
            return [self]

        if (
            warn_copy_on_write()
            and already_warned is not None
            and not already_warned.warned_already
        ):
            if self.refs.has_reference():
                warnings.warn(
                    COW_WARNING_GENERAL_MSG,
                    FutureWarning,
                    stacklevel=find_stack_level(),
                )
                already_warned.warned_already = True

        try:
            casted = np_can_hold_element(values.dtype, new)

            self = self._maybe_copy(using_cow, inplace=True)
            values = cast(np.ndarray, self.values)

            putmask_without_repeat(values.T, mask, casted)
            return [self]
        except LossySetitemError:
            if self.ndim == 1 or self.shape[0] == 1:
                # no need to split columns

                if not is_list_like(new):
                    # using just new[indexer] can't save us the need to cast
                    return self.coerce_to_target_dtype(
                        new, warn_on_upcast=True
                    ).putmask(mask, new)
                else:
                    indexer = mask.nonzero()[0]
                    nb = self.setitem(indexer, new[indexer], using_cow=using_cow)
                    return [nb]

            else:
                is_array = isinstance(new, np.ndarray)

                res_blocks = []
                nbs = self._split()
                for i, nb in enumerate(nbs):
                    n = new
                    if is_array:
                        # we have a different value per-column
                        n = new[:, i : i + 1]

                    submask = orig_mask[:, i : i + 1]
                    rbs = nb.putmask(submask, n, using_cow=using_cow)
                    res_blocks.extend(rbs)
                return res_blocks

    def where(
        self, other, cond, _downcast: str | bool = "infer", using_cow: bool = False
    ) -> list[Block]:
        """
        evaluate the block; return result block(s) from the result

        Parameters
        ----------
        other : a ndarray/object
        cond : np.ndarray[bool], SparseArray[bool], or BooleanArray
        _downcast : str or None, default "infer"
            Private because we only specify it when calling from fillna.

        Returns
        -------
        List[Block]
        """
        assert cond.ndim == self.ndim
        assert not isinstance(other, (ABCIndex, ABCSeries, ABCDataFrame))

        transpose = self.ndim == 2

        cond = extract_bool_array(cond)

        # EABlocks override where
        values = cast(np.ndarray, self.values)
        orig_other = other
        if transpose:
            values = values.T

        icond, noop = validate_putmask(values, ~cond)
        if noop:
            # GH-39595: Always return a copy; short-circuit up/downcasting
            if using_cow:
                return [self.copy(deep=False)]
            return [self.copy()]

        if other is lib.no_default:
            other = self.fill_value

        other = self._standardize_fill_value(other)

        try:
            # try/except here is equivalent to a self._can_hold_element check,
            #  but this gets us back 'casted' which we will reuse below;
            #  without using 'casted', expressions.where may do unwanted upcasts.
            casted = np_can_hold_element(values.dtype, other)
        except (ValueError, TypeError, LossySetitemError):
            # we cannot coerce, return a compat dtype

            if self.ndim == 1 or self.shape[0] == 1:
                # no need to split columns

                block = self.coerce_to_target_dtype(other)
                blocks = block.where(orig_other, cond, using_cow=using_cow)
                return self._maybe_downcast(
                    blocks, downcast=_downcast, using_cow=using_cow, caller="where"
                )

            else:
                # since _maybe_downcast would split blocks anyway, we
                #  can avoid some potential upcast/downcast by splitting
                #  on the front end.
                is_array = isinstance(other, (np.ndarray, ExtensionArray))

                res_blocks = []
                nbs = self._split()
                for i, nb in enumerate(nbs):
                    oth = other
                    if is_array:
                        # we have a different value per-column
                        oth = other[:, i : i + 1]

                    submask = cond[:, i : i + 1]
                    rbs = nb.where(
                        oth, submask, _downcast=_downcast, using_cow=using_cow
                    )
                    res_blocks.extend(rbs)
                return res_blocks

        else:
            other = casted
            alt = setitem_datetimelike_compat(values, icond.sum(), other)
            if alt is not other:
                if is_list_like(other) and len(other) < len(values):
                    # call np.where with other to get the appropriate ValueError
                    np.where(~icond, values, other)
                    raise NotImplementedError(
                        "This should not be reached; call to np.where above is "
                        "expected to raise ValueError. Please report a bug at "
                        "github.com/pandas-dev/pandas"
                    )
                result = values.copy()
                np.putmask(result, icond, alt)
            else:
                # By the time we get here, we should have all Series/Index
                #  args extracted to ndarray
                if (
                    is_list_like(other)
                    and not isinstance(other, np.ndarray)
                    and len(other) == self.shape[-1]
                ):
                    # If we don't do this broadcasting here, then expressions.where
                    #  will broadcast a 1D other to be row-like instead of
                    #  column-like.
                    other = np.array(other).reshape(values.shape)
                    # If lengths don't match (or len(other)==1), we will raise
                    #  inside expressions.where, see test_series_where

                # Note: expressions.where may upcast.
                result = expressions.where(~icond, values, other)
                # The np_can_hold_element check _should_ ensure that we always
                #  have result.dtype == self.dtype here.

        if transpose:
            result = result.T

        return [self.make_block(result)]

    def fillna(
        self,
        value,
        limit: int | None = None,
        inplace: bool = False,
        downcast=None,
        using_cow: bool = False,
        already_warned=None,
    ) -> list[Block]:
        """
        fillna on the block with the value. If we fail, then convert to
        block to hold objects instead and try again
        """
        # Caller is responsible for validating limit; if int it is strictly positive
        inplace = validate_bool_kwarg(inplace, "inplace")

        if not self._can_hold_na:
            # can short-circuit the isna call
            noop = True
        else:
            mask = isna(self.values)
            mask, noop = validate_putmask(self.values, mask)

        if noop:
            # we can't process the value, but nothing to do
            if inplace:
                if using_cow:
                    return [self.copy(deep=False)]
                # Arbitrarily imposing the convention that we ignore downcast
                #  on no-op when inplace=True
                return [self]
            else:
                # GH#45423 consistent downcasting on no-ops.
                nb = self.copy(deep=not using_cow)
                nbs = nb._maybe_downcast(
                    [nb], downcast=downcast, using_cow=using_cow, caller="fillna"
                )
                return nbs

        if limit is not None:
            mask[mask.cumsum(self.values.ndim - 1) > limit] = False

        if inplace:
            nbs = self.putmask(
                mask.T, value, using_cow=using_cow, already_warned=already_warned
            )
        else:
            # without _downcast, we would break
            #  test_fillna_dtype_conversion_equiv_replace
            nbs = self.where(value, ~mask.T, _downcast=False)

        # Note: blk._maybe_downcast vs self._maybe_downcast(nbs)
        #  makes a difference bc blk may have object dtype, which has
        #  different behavior in _maybe_downcast.
        return extend_blocks(
            [
                blk._maybe_downcast(
                    [blk], downcast=downcast, using_cow=using_cow, caller="fillna"
                )
                for blk in nbs
            ]
        )

    def pad_or_backfill(
        self,
        *,
        method: FillnaOptions,
        axis: AxisInt = 0,
        inplace: bool = False,
        limit: int | None = None,
        limit_area: Literal["inside", "outside"] | None = None,
        downcast: Literal["infer"] | None = None,
        using_cow: bool = False,
        already_warned=None,
    ) -> list[Block]:
        if not self._can_hold_na:
            # If there are no NAs, then interpolate is a no-op
            if using_cow:
                return [self.copy(deep=False)]
            return [self] if inplace else [self.copy()]

        copy, refs = self._get_refs_and_copy(using_cow, inplace)

        # Dispatch to the NumpyExtensionArray method.
        # We know self.array_values is a NumpyExtensionArray bc EABlock overrides
        vals = cast(NumpyExtensionArray, self.array_values)
        if axis == 1:
            vals = vals.T
        new_values = vals._pad_or_backfill(
            method=method,
            limit=limit,
            limit_area=limit_area,
            copy=copy,
        )
        if (
            not copy
            and warn_copy_on_write()
            and already_warned is not None
            and not already_warned.warned_already
        ):
            if self.refs.has_reference():
                warnings.warn(
                    COW_WARNING_GENERAL_MSG,
                    FutureWarning,
                    stacklevel=find_stack_level(),
                )
                already_warned.warned_already = True
        if axis == 1:
            new_values = new_values.T

        data = extract_array(new_values, extract_numpy=True)

        nb = self.make_block_same_class(data, refs=refs)
        return nb._maybe_downcast([nb], downcast, using_cow, caller="fillna")

    @final
    def interpolate(
        self,
        *,
        method: InterpolateOptions,
        index: Index,
        inplace: bool = False,
        limit: int | None = None,
        limit_direction: Literal["forward", "backward", "both"] = "forward",
        limit_area: Literal["inside", "outside"] | None = None,
        downcast: Literal["infer"] | None = None,
        using_cow: bool = False,
        already_warned=None,
        **kwargs,
    ) -> list[Block]:
        inplace = validate_bool_kwarg(inplace, "inplace")
        # error: Non-overlapping equality check [...]
        if method == "asfreq":  # type: ignore[comparison-overlap]
            # clean_fill_method used to allow this
            missing.clean_fill_method(method)

        if not self._can_hold_na:
            # If there are no NAs, then interpolate is a no-op
            if using_cow:
                return [self.copy(deep=False)]
            return [self] if inplace else [self.copy()]

        # TODO(3.0): this case will not be reachable once GH#53638 is enforced
        if self.dtype == _dtype_obj:
            # only deal with floats
            # bc we already checked that can_hold_na, we don't have int dtype here
            # test_interp_basic checks that we make a copy here
            if using_cow:
                return [self.copy(deep=False)]
            return [self] if inplace else [self.copy()]

        copy, refs = self._get_refs_and_copy(using_cow, inplace)

        # Dispatch to the EA method.
        new_values = self.array_values.interpolate(
            method=method,
            axis=self.ndim - 1,
            index=index,
            limit=limit,
            limit_direction=limit_direction,
            limit_area=limit_area,
            copy=copy,
            **kwargs,
        )
        data = extract_array(new_values, extract_numpy=True)

        if (
            not copy
            and warn_copy_on_write()
            and already_warned is not None
            and not already_warned.warned_already
        ):
            if self.refs.has_reference():
                warnings.warn(
                    COW_WARNING_GENERAL_MSG,
                    FutureWarning,
                    stacklevel=find_stack_level(),
                )
                already_warned.warned_already = True

        nb = self.make_block_same_class(data, refs=refs)
        return nb._maybe_downcast([nb], downcast, using_cow, caller="interpolate")

    @final
    def diff(self, n: int) -> list[Block]:
        """return block for the diff of the values"""
        # only reached with ndim == 2
        # TODO(EA2D): transpose will be unnecessary with 2D EAs
        new_values = algos.diff(self.values.T, n, axis=0).T
        return [self.make_block(values=new_values)]

    def shift(self, periods: int, fill_value: Any = None) -> list[Block]:
        """shift the block by periods, possibly upcast"""
        # convert integer to float if necessary. need to do a lot more than
        # that, handle boolean etc also
        axis = self.ndim - 1

        # Note: periods is never 0 here, as that is handled at the top of
        #  NDFrame.shift.  If that ever changes, we can do a check for periods=0
        #  and possibly avoid coercing.

        if not lib.is_scalar(fill_value) and self.dtype != _dtype_obj:
            # with object dtype there is nothing to promote, and the user can
            #  pass pretty much any weird fill_value they like
            # see test_shift_object_non_scalar_fill
            raise ValueError("fill_value must be a scalar")

        fill_value = self._standardize_fill_value(fill_value)

        try:
            # error: Argument 1 to "np_can_hold_element" has incompatible type
            # "Union[dtype[Any], ExtensionDtype]"; expected "dtype[Any]"
            casted = np_can_hold_element(
                self.dtype, fill_value  # type: ignore[arg-type]
            )
        except LossySetitemError:
            nb = self.coerce_to_target_dtype(fill_value)
            return nb.shift(periods, fill_value=fill_value)

        else:
            values = cast(np.ndarray, self.values)
            new_values = shift(values, periods, axis, casted)
            return [self.make_block_same_class(new_values)]

    @final
    def quantile(
        self,
        qs: Index,  # with dtype float64
        interpolation: QuantileInterpolation = "linear",
    ) -> Block:
        """
        compute the quantiles of the

        Parameters
        ----------
        qs : Index
            The quantiles to be computed in float64.
        interpolation : str, default 'linear'
            Type of interpolation.

        Returns
        -------
        Block
        """
        # We should always have ndim == 2 because Series dispatches to DataFrame
        assert self.ndim == 2
        assert is_list_like(qs)  # caller is responsible for this

        result = quantile_compat(self.values, np.asarray(qs._values), interpolation)
        # ensure_block_shape needed for cases where we start with EA and result
        #  is ndarray, e.g. IntegerArray, SparseArray
        result = ensure_block_shape(result, ndim=2)
        return new_block_2d(result, placement=self._mgr_locs)

    @final
    def round(self, decimals: int, using_cow: bool = False) -> Self:
        """
        Rounds the values.
        If the block is not of an integer or float dtype, nothing happens.
        This is consistent with DataFrame.round behavivor.
        (Note: Series.round would raise)

        Parameters
        ----------
        decimals: int,
            Number of decimal places to round to.
            Caller is responsible for validating this
        using_cow: bool,
            Whether Copy on Write is enabled right now
        """
        if not self.is_numeric or self.is_bool:
            return self.copy(deep=not using_cow)
        refs = None
        # TODO: round only defined on BaseMaskedArray
        # Series also does this, so would need to fix both places
        # error: Item "ExtensionArray" of "Union[ndarray[Any, Any], ExtensionArray]"
        # has no attribute "round"
        values = self.values.round(decimals)  # type: ignore[union-attr]
        if values is self.values:
            if not using_cow:
                # Normally would need to do this before, but
                # numpy only returns same array when round operation
                # is no-op
                # https://github.com/numpy/numpy/blob/486878b37fc7439a3b2b87747f50db9b62fea8eb/numpy/core/src/multiarray/calculation.c#L625-L636
                values = values.copy()
            else:
                refs = self.refs
        return self.make_block_same_class(values, refs=refs)

    # ---------------------------------------------------------------------
    # Abstract Methods Overridden By EABackedBlock and NumpyBlock

    def delete(self, loc) -> list[Block]:
        """Deletes the locs from the block.

        We split the block to avoid copying the underlying data. We create new
        blocks for every connected segment of the initial block that is not deleted.
        The new blocks point to the initial array.
        """
        if not is_list_like(loc):
            loc = [loc]

        if self.ndim == 1:
            values = cast(np.ndarray, self.values)
            values = np.delete(values, loc)
            mgr_locs = self._mgr_locs.delete(loc)
            return [type(self)(values, placement=mgr_locs, ndim=self.ndim)]

        if np.max(loc) >= self.values.shape[0]:
            raise IndexError

        # Add one out-of-bounds indexer as maximum to collect
        # all columns after our last indexer if any
        loc = np.concatenate([loc, [self.values.shape[0]]])
        mgr_locs_arr = self._mgr_locs.as_array
        new_blocks: list[Block] = []

        previous_loc = -1
        # TODO(CoW): This is tricky, if parent block goes out of scope
        # all split blocks are referencing each other even though they
        # don't share data
        refs = self.refs if self.refs.has_reference() else None
        for idx in loc:
            if idx == previous_loc + 1:
                # There is no column between current and last idx
                pass
            else:
                # No overload variant of "__getitem__" of "ExtensionArray" matches
                # argument type "Tuple[slice, slice]"
                values = self.values[previous_loc + 1 : idx, :]  # type: ignore[call-overload]
                locs = mgr_locs_arr[previous_loc + 1 : idx]
                nb = type(self)(
                    values, placement=BlockPlacement(locs), ndim=self.ndim, refs=refs
                )
                new_blocks.append(nb)

            previous_loc = idx

        return new_blocks

    @property
    def is_view(self) -> bool:
        """return a boolean if I am possibly a view"""
        raise AbstractMethodError(self)

    @property
    def array_values(self) -> ExtensionArray:
        """
        The array that Series.array returns. Always an ExtensionArray.
        """
        raise AbstractMethodError(self)

    def get_values(self, dtype: DtypeObj | None = None) -> np.ndarray:
        """
        return an internal format, currently just the ndarray
        this is often overridden to handle to_dense like operations
        """
        raise AbstractMethodError(self)


class EABackedBlock(Block):
    """
    Mixin for Block subclasses backed by ExtensionArray.
    """

    values: ExtensionArray

    @final
    def shift(self, periods: int, fill_value: Any = None) -> list[Block]:
        """
        Shift the block by `periods`.

        Dispatches to underlying ExtensionArray and re-boxes in an
        ExtensionBlock.
        """
        # Transpose since EA.shift is always along axis=0, while we want to shift
        #  along rows.
        new_values = self.values.T.shift(periods=periods, fill_value=fill_value).T
        return [self.make_block_same_class(new_values)]

    @final
    def setitem(self, indexer, value, using_cow: bool = False):
        """
        Attempt self.values[indexer] = value, possibly creating a new array.

        This differs from Block.setitem by not allowing setitem to change
        the dtype of the Block.

        Parameters
        ----------
        indexer : tuple, list-like, array-like, slice, int
            The subset of self.values to set
        value : object
            The value being set
        using_cow: bool, default False
            Signaling if CoW is used.

        Returns
        -------
        Block

        Notes
        -----
        `indexer` is a direct slice/positional indexer. `value` must
        be a compatible shape.
        """
        orig_indexer = indexer
        orig_value = value

        indexer = self._unwrap_setitem_indexer(indexer)
        value = self._maybe_squeeze_arg(value)

        values = self.values
        if values.ndim == 2:
            # TODO(GH#45419): string[pyarrow] tests break if we transpose
            #  unconditionally
            values = values.T
        check_setitem_lengths(indexer, value, values)

        try:
            values[indexer] = value
        except (ValueError, TypeError):
            if isinstance(self.dtype, IntervalDtype):
                # see TestSetitemFloatIntervalWithIntIntervalValues
                nb = self.coerce_to_target_dtype(orig_value, warn_on_upcast=True)
                return nb.setitem(orig_indexer, orig_value)

            elif isinstance(self, NDArrayBackedExtensionBlock):
                nb = self.coerce_to_target_dtype(orig_value, warn_on_upcast=True)
                return nb.setitem(orig_indexer, orig_value)

            else:
                raise

        else:
            return self

    @final
    def where(
        self, other, cond, _downcast: str | bool = "infer", using_cow: bool = False
    ) -> list[Block]:
        # _downcast private bc we only specify it when calling from fillna
        arr = self.values.T

        cond = extract_bool_array(cond)

        orig_other = other
        orig_cond = cond
        other = self._maybe_squeeze_arg(other)
        cond = self._maybe_squeeze_arg(cond)

        if other is lib.no_default:
            other = self.fill_value

        icond, noop = validate_putmask(arr, ~cond)
        if noop:
            # GH#44181, GH#45135
            # Avoid a) raising for Interval/PeriodDtype and b) unnecessary object upcast
            if using_cow:
                return [self.copy(deep=False)]
            return [self.copy()]

        try:
            res_values = arr._where(cond, other).T
        except (ValueError, TypeError):
            if self.ndim == 1 or self.shape[0] == 1:
                if isinstance(self.dtype, (IntervalDtype, StringDtype)):
                    # TestSetitemFloatIntervalWithIntIntervalValues
                    blk = self.coerce_to_target_dtype(orig_other)
                    if (
                        self.ndim == 2
                        and isinstance(orig_cond, np.ndarray)
                        and orig_cond.ndim == 1
                        and not is_1d_only_ea_dtype(blk.dtype)
                    ):
                        orig_cond = orig_cond[:, None]
                    nbs = blk.where(orig_other, orig_cond, using_cow=using_cow)
                    return self._maybe_downcast(
                        nbs, downcast=_downcast, using_cow=using_cow, caller="where"
                    )

                elif isinstance(self, NDArrayBackedExtensionBlock):
                    # NB: not (yet) the same as
                    #  isinstance(values, NDArrayBackedExtensionArray)
                    blk = self.coerce_to_target_dtype(orig_other)
                    nbs = blk.where(orig_other, orig_cond, using_cow=using_cow)
                    return self._maybe_downcast(
                        nbs, downcast=_downcast, using_cow=using_cow, caller="where"
                    )

                else:
                    raise

            else:
                # Same pattern we use in Block.putmask
                is_array = isinstance(orig_other, (np.ndarray, ExtensionArray))

                res_blocks = []
                nbs = self._split()
                for i, nb in enumerate(nbs):
                    n = orig_other
                    if is_array:
                        # we have a different value per-column
                        n = orig_other[:, i : i + 1]

                    submask = orig_cond[:, i : i + 1]
                    rbs = nb.where(n, submask, using_cow=using_cow)
                    res_blocks.extend(rbs)
                return res_blocks

        nb = self.make_block_same_class(res_values)
        return [nb]

    @final
    def putmask(
        self, mask, new, using_cow: bool = False, already_warned=None
    ) -> list[Block]:
        """
        See Block.putmask.__doc__
        """
        mask = extract_bool_array(mask)
        if new is lib.no_default:
            new = self.fill_value

        orig_new = new
        orig_mask = mask
        new = self._maybe_squeeze_arg(new)
        mask = self._maybe_squeeze_arg(mask)

        if not mask.any():
            if using_cow:
                return [self.copy(deep=False)]
            return [self]

        if (
            warn_copy_on_write()
            and already_warned is not None
            and not already_warned.warned_already
        ):
            if self.refs.has_reference():
                warnings.warn(
                    COW_WARNING_GENERAL_MSG,
                    FutureWarning,
                    stacklevel=find_stack_level(),
                )
                already_warned.warned_already = True

        self = self._maybe_copy(using_cow, inplace=True)
        values = self.values
        if values.ndim == 2:
            values = values.T

        try:
            # Caller is responsible for ensuring matching lengths
            values._putmask(mask, new)
        except (TypeError, ValueError):
            if self.ndim == 1 or self.shape[0] == 1:
                if isinstance(self.dtype, IntervalDtype):
                    # Discussion about what we want to support in the general
                    #  case GH#39584
                    blk = self.coerce_to_target_dtype(orig_new, warn_on_upcast=True)
                    return blk.putmask(orig_mask, orig_new)

                elif isinstance(self, NDArrayBackedExtensionBlock):
                    # NB: not (yet) the same as
                    #  isinstance(values, NDArrayBackedExtensionArray)
                    blk = self.coerce_to_target_dtype(orig_new, warn_on_upcast=True)
                    return blk.putmask(orig_mask, orig_new)

                else:
                    raise

            else:
                # Same pattern we use in Block.putmask
                is_array = isinstance(orig_new, (np.ndarray, ExtensionArray))

                res_blocks = []
                nbs = self._split()
                for i, nb in enumerate(nbs):
                    n = orig_new
                    if is_array:
                        # we have a different value per-column
                        n = orig_new[:, i : i + 1]

                    submask = orig_mask[:, i : i + 1]
                    rbs = nb.putmask(submask, n)
                    res_blocks.extend(rbs)
                return res_blocks

        return [self]

    @final
    def delete(self, loc) -> list[Block]:
        # This will be unnecessary if/when __array_function__ is implemented
        if self.ndim == 1:
            values = self.values.delete(loc)
            mgr_locs = self._mgr_locs.delete(loc)
            return [type(self)(values, placement=mgr_locs, ndim=self.ndim)]
        elif self.values.ndim == 1:
            # We get here through to_stata
            return []
        return super().delete(loc)

    @final
    @cache_readonly
    def array_values(self) -> ExtensionArray:
        return self.values

    @final
    def get_values(self, dtype: DtypeObj | None = None) -> np.ndarray:
        """
        return object dtype as boxed values, such as Timestamps/Timedelta
        """
        values: ArrayLike = self.values
        if dtype == _dtype_obj:
            values = values.astype(object)
        # TODO(EA2D): reshape not needed with 2D EAs
        return np.asarray(values).reshape(self.shape)

    @final
    def pad_or_backfill(
        self,
        *,
        method: FillnaOptions,
        axis: AxisInt = 0,
        inplace: bool = False,
        limit: int | None = None,
        limit_area: Literal["inside", "outside"] | None = None,
        downcast: Literal["infer"] | None = None,
        using_cow: bool = False,
        already_warned=None,
    ) -> list[Block]:
        values = self.values

        kwargs: dict[str, Any] = {"method": method, "limit": limit}
        if "limit_area" in inspect.signature(values._pad_or_backfill).parameters:
            kwargs["limit_area"] = limit_area
        elif limit_area is not None:
            raise NotImplementedError(
                f"{type(values).__name__} does not implement limit_area "
                "(added in pandas 2.2). 3rd-party ExtnsionArray authors "
                "need to add this argument to _pad_or_backfill."
            )

        if values.ndim == 2 and axis == 1:
            # NDArrayBackedExtensionArray.fillna assumes axis=0
            new_values = values.T._pad_or_backfill(**kwargs).T
        else:
            new_values = values._pad_or_backfill(**kwargs)
        return [self.make_block_same_class(new_values)]


class ExtensionBlock(EABackedBlock):
    """
    Block for holding extension types.

    Notes
    -----
    This holds all 3rd-party extension array types. It's also the immediate
    parent class for our internal extension types' blocks.

    ExtensionArrays are limited to 1-D.
    """

    values: ExtensionArray

    def fillna(
        self,
        value,
        limit: int | None = None,
        inplace: bool = False,
        downcast=None,
        using_cow: bool = False,
        already_warned=None,
    ) -> list[Block]:
        if isinstance(self.dtype, (IntervalDtype, StringDtype)):
            # Block.fillna handles coercion (test_fillna_interval)
            return super().fillna(
                value=value,
                limit=limit,
                inplace=inplace,
                downcast=downcast,
                using_cow=using_cow,
                already_warned=already_warned,
            )
        if using_cow and self._can_hold_na and not self.values._hasna:
            refs = self.refs
            new_values = self.values
        else:
            copy, refs = self._get_refs_and_copy(using_cow, inplace)

            try:
                new_values = self.values.fillna(
                    value=value, method=None, limit=limit, copy=copy
                )
            except TypeError:
                # 3rd party EA that has not implemented copy keyword yet
                refs = None
                new_values = self.values.fillna(value=value, method=None, limit=limit)
                # issue the warning *after* retrying, in case the TypeError
                #  was caused by an invalid fill_value
                warnings.warn(
                    # GH#53278
                    "ExtensionArray.fillna added a 'copy' keyword in pandas "
                    "2.1.0. In a future version, ExtensionArray subclasses will "
                    "need to implement this keyword or an exception will be "
                    "raised. In the interim, the keyword is ignored by "
                    f"{type(self.values).__name__}.",
                    DeprecationWarning,
                    stacklevel=find_stack_level(),
                )
            else:
                if (
                    not copy
                    and warn_copy_on_write()
                    and already_warned is not None
                    and not already_warned.warned_already
                ):
                    if self.refs.has_reference():
                        warnings.warn(
                            COW_WARNING_GENERAL_MSG,
                            FutureWarning,
                            stacklevel=find_stack_level(),
                        )
                        already_warned.warned_already = True

        nb = self.make_block_same_class(new_values, refs=refs)
        return nb._maybe_downcast([nb], downcast, using_cow=using_cow, caller="fillna")

    @cache_readonly
    def shape(self) -> Shape:
        # TODO(EA2D): override unnecessary with 2D EAs
        if self.ndim == 1:
            return (len(self.values),)
        return len(self._mgr_locs), len(self.values)

    def iget(self, i: int | tuple[int, int] | tuple[slice, int]):
        # In the case where we have a tuple[slice, int], the slice will always
        #  be slice(None)
        # We _could_ make the annotation more specific, but mypy would
        #  complain about override mismatch:
        #  Literal[0] | tuple[Literal[0], int] | tuple[slice, int]

        # Note: only reached with self.ndim == 2

        if isinstance(i, tuple):
            # TODO(EA2D): unnecessary with 2D EAs
            col, loc = i
            if not com.is_null_slice(col) and col != 0:
                raise IndexError(f"{self} only contains one item")
            if isinstance(col, slice):
                # the is_null_slice check above assures that col is slice(None)
                #  so what we want is a view on all our columns and row loc
                if loc < 0:
                    loc += len(self.values)
                # Note: loc:loc+1 vs [[loc]] makes a difference when called
                #  from fast_xs because we want to get a view back.
                return self.values[loc : loc + 1]
            return self.values[loc]
        else:
            if i != 0:
                raise IndexError(f"{self} only contains one item")
            return self.values

    def set_inplace(self, locs, values: ArrayLike, copy: bool = False) -> None:
        # When an ndarray, we should have locs.tolist() == [0]
        # When a BlockPlacement we should have list(locs) == [0]
        if copy:
            self.values = self.values.copy()
        self.values[:] = values

    def _maybe_squeeze_arg(self, arg):
        """
        If necessary, squeeze a (N, 1) ndarray to (N,)
        """
        # e.g. if we are passed a 2D mask for putmask
        if (
            isinstance(arg, (np.ndarray, ExtensionArray))
            and arg.ndim == self.values.ndim + 1
        ):
            # TODO(EA2D): unnecessary with 2D EAs
            assert arg.shape[1] == 1
            # error: No overload variant of "__getitem__" of "ExtensionArray"
            # matches argument type "Tuple[slice, int]"
            arg = arg[:, 0]  # type: ignore[call-overload]
        elif isinstance(arg, ABCDataFrame):
            # 2022-01-06 only reached for setitem
            # TODO: should we avoid getting here with DataFrame?
            assert arg.shape[1] == 1
            arg = arg._ixs(0, axis=1)._values

        return arg

    def _unwrap_setitem_indexer(self, indexer):
        """
        Adapt a 2D-indexer to our 1D values.

        This is intended for 'setitem', not 'iget' or '_slice'.
        """
        # TODO: ATM this doesn't work for iget/_slice, can we change that?

        if isinstance(indexer, tuple) and len(indexer) == 2:
            # TODO(EA2D): not needed with 2D EAs
            #  Should never have length > 2.  Caller is responsible for checking.
            #  Length 1 is reached vis setitem_single_block and setitem_single_column
            #  each of which pass indexer=(pi,)
            if all(isinstance(x, np.ndarray) and x.ndim == 2 for x in indexer):
                # GH#44703 went through indexing.maybe_convert_ix
                first, second = indexer
                if not (
                    second.size == 1 and (second == 0).all() and first.shape[1] == 1
                ):
                    raise NotImplementedError(
                        "This should not be reached. Please report a bug at "
                        "github.com/pandas-dev/pandas/"
                    )
                indexer = first[:, 0]

            elif lib.is_integer(indexer[1]) and indexer[1] == 0:
                # reached via setitem_single_block passing the whole indexer
                indexer = indexer[0]

            elif com.is_null_slice(indexer[1]):
                indexer = indexer[0]

            elif is_list_like(indexer[1]) and indexer[1][0] == 0:
                indexer = indexer[0]

            else:
                raise NotImplementedError(
                    "This should not be reached. Please report a bug at "
                    "github.com/pandas-dev/pandas/"
                )
        return indexer

    @property
    def is_view(self) -> bool:
        """Extension arrays are never treated as views."""
        return False

    # error: Cannot override writeable attribute with read-only property
    @cache_readonly
    def is_numeric(self) -> bool:  # type: ignore[override]
        return self.values.dtype._is_numeric

    def _slice(
        self, slicer: slice | npt.NDArray[np.bool_] | npt.NDArray[np.intp]
    ) -> ExtensionArray:
        """
        Return a slice of my values.

        Parameters
        ----------
        slicer : slice, ndarray[int], or ndarray[bool]
            Valid (non-reducing) indexer for self.values.

        Returns
        -------
        ExtensionArray
        """
        # Notes: ndarray[bool] is only reachable when via get_rows_with_mask, which
        #  is only for Series, i.e. self.ndim == 1.

        # return same dims as we currently have
        if self.ndim == 2:
            # reached via getitem_block via _slice_take_blocks_ax0
            # TODO(EA2D): won't be necessary with 2D EAs

            if not isinstance(slicer, slice):
                raise AssertionError(
                    "invalid slicing for a 1-ndim ExtensionArray", slicer
                )
            # GH#32959 only full-slicers along fake-dim0 are valid
            # TODO(EA2D): won't be necessary with 2D EAs
            # range(1) instead of self._mgr_locs to avoid exception on [::-1]
            #  see test_iloc_getitem_slice_negative_step_ea_block
            new_locs = range(1)[slicer]
            if not len(new_locs):
                raise AssertionError(
                    "invalid slicing for a 1-ndim ExtensionArray", slicer
                )
            slicer = slice(None)

        return self.values[slicer]

    @final
    def slice_block_rows(self, slicer: slice) -> Self:
        """
        Perform __getitem__-like specialized to slicing along index.
        """
        # GH#42787 in principle this is equivalent to values[..., slicer], but we don't
        # require subclasses of ExtensionArray to support that form (for now).
        new_values = self.values[slicer]
        return type(self)(new_values, self._mgr_locs, ndim=self.ndim, refs=self.refs)

    def _unstack(
        self,
        unstacker,
        fill_value,
        new_placement: npt.NDArray[np.intp],
        needs_masking: npt.NDArray[np.bool_],
    ):
        # ExtensionArray-safe unstack.
        # We override Block._unstack, which unstacks directly on the
        # values of the array. For EA-backed blocks, this would require
        # converting to a 2-D ndarray of objects.
        # Instead, we unstack an ndarray of integer positions, followed by
        # a `take` on the actual values.

        # Caller is responsible for ensuring self.shape[-1] == len(unstacker.index)
        new_values, mask = unstacker.arange_result

        # Note: these next two lines ensure that
        #  mask.sum() == sum(len(nb.mgr_locs) for nb in blocks)
        #  which the calling function needs in order to pass verify_integrity=False
        #  to the BlockManager constructor
        new_values = new_values.T[mask]
        new_placement = new_placement[mask]

        # needs_masking[i] calculated once in BlockManager.unstack tells
        #  us if there are any -1s in the relevant indices.  When False,
        #  that allows us to go through a faster path in 'take', among
        #  other things avoiding e.g. Categorical._validate_scalar.
        blocks = [
            # TODO: could cast to object depending on fill_value?
            type(self)(
                self.values.take(
                    indices, allow_fill=needs_masking[i], fill_value=fill_value
                ),
                BlockPlacement(place),
                ndim=2,
            )
            for i, (indices, place) in enumerate(zip(new_values, new_placement))
        ]
        return blocks, mask


class NumpyBlock(Block):
    values: np.ndarray
    __slots__ = ()

    @property
    def is_view(self) -> bool:
        """return a boolean if I am possibly a view"""
        return self.values.base is not None

    @property
    def array_values(self) -> ExtensionArray:
        return NumpyExtensionArray(self.values)

    def get_values(self, dtype: DtypeObj | None = None) -> np.ndarray:
        if dtype == _dtype_obj:
            return self.values.astype(_dtype_obj)
        return self.values

    @cache_readonly
    def is_numeric(self) -> bool:  # type: ignore[override]
        dtype = self.values.dtype
        kind = dtype.kind

        return kind in "fciub"


class NumericBlock(NumpyBlock):
    # this Block type is kept for backwards-compatibility
    # TODO(3.0): delete and remove deprecation in __init__.py.
    __slots__ = ()


class ObjectBlock(NumpyBlock):
    # this Block type is kept for backwards-compatibility
    # TODO(3.0): delete and remove deprecation in __init__.py.
    __slots__ = ()


class NDArrayBackedExtensionBlock(EABackedBlock):
    """
    Block backed by an NDArrayBackedExtensionArray
    """

    values: NDArrayBackedExtensionArray

    @property
    def is_view(self) -> bool:
        """return a boolean if I am possibly a view"""
        # check the ndarray values of the DatetimeIndex values
        return self.values._ndarray.base is not None


class DatetimeLikeBlock(NDArrayBackedExtensionBlock):
    """Block for datetime64[ns], timedelta64[ns]."""

    __slots__ = ()
    is_numeric = False
    values: DatetimeArray | TimedeltaArray


class DatetimeTZBlock(DatetimeLikeBlock):
    """implement a datetime64 block with a tz attribute"""

    values: DatetimeArray

    __slots__ = ()


# -----------------------------------------------------------------
# Constructor Helpers


def maybe_coerce_values(values: ArrayLike) -> ArrayLike:
    """
    Input validation for values passed to __init__. Ensure that
    any datetime64/timedelta64 dtypes are in nanoseconds.  Ensure
    that we do not have string dtypes.

    Parameters
    ----------
    values : np.ndarray or ExtensionArray

    Returns
    -------
    values : np.ndarray or ExtensionArray
    """
    # Caller is responsible for ensuring NumpyExtensionArray is already extracted.

    if isinstance(values, np.ndarray):
        values = ensure_wrapped_if_datetimelike(values)

        if issubclass(values.dtype.type, str):
            values = np.array(values, dtype=object)

    if isinstance(values, (DatetimeArray, TimedeltaArray)) and values.freq is not None:
        # freq is only stored in DatetimeIndex/TimedeltaIndex, not in Series/DataFrame
        values = values._with_freq(None)

    return values


def get_block_type(dtype: DtypeObj) -> type[Block]:
    """
    Find the appropriate Block subclass to use for the given values and dtype.

    Parameters
    ----------
    dtype : numpy or pandas dtype

    Returns
    -------
    cls : class, subclass of Block
    """
    if isinstance(dtype, DatetimeTZDtype):
        return DatetimeTZBlock
    elif isinstance(dtype, PeriodDtype):
        return NDArrayBackedExtensionBlock
    elif isinstance(dtype, ExtensionDtype):
        # Note: need to be sure NumpyExtensionArray is unwrapped before we get here
        return ExtensionBlock

    # We use kind checks because it is much more performant
    #  than is_foo_dtype
    kind = dtype.kind
    if kind in "Mm":
        return DatetimeLikeBlock

    return NumpyBlock


def new_block_2d(
    values: ArrayLike, placement: BlockPlacement, refs: BlockValuesRefs | None = None
):
    # new_block specialized to case with
    #  ndim=2
    #  isinstance(placement, BlockPlacement)
    #  check_ndim/ensure_block_shape already checked
    klass = get_block_type(values.dtype)

    values = maybe_coerce_values(values)
    return klass(values, ndim=2, placement=placement, refs=refs)


def new_block(
    values,
    placement: BlockPlacement,
    *,
    ndim: int,
    refs: BlockValuesRefs | None = None,
) -> Block:
    # caller is responsible for ensuring:
    # - values is NOT a NumpyExtensionArray
    # - check_ndim/ensure_block_shape already checked
    # - maybe_coerce_values already called/unnecessary
    klass = get_block_type(values.dtype)
    return klass(values, ndim=ndim, placement=placement, refs=refs)


def check_ndim(values, placement: BlockPlacement, ndim: int) -> None:
    """
    ndim inference and validation.

    Validates that values.ndim and ndim are consistent.
    Validates that len(values) and len(placement) are consistent.

    Parameters
    ----------
    values : array-like
    placement : BlockPlacement
    ndim : int

    Raises
    ------
    ValueError : the number of dimensions do not match
    """

    if values.ndim > ndim:
        # Check for both np.ndarray and ExtensionArray
        raise ValueError(
            "Wrong number of dimensions. "
            f"values.ndim > ndim [{values.ndim} > {ndim}]"
        )

    if not is_1d_only_ea_dtype(values.dtype):
        # TODO(EA2D): special case not needed with 2D EAs
        if values.ndim != ndim:
            raise ValueError(
                "Wrong number of dimensions. "
                f"values.ndim != ndim [{values.ndim} != {ndim}]"
            )
        if len(placement) != len(values):
            raise ValueError(
                f"Wrong number of items passed {len(values)}, "
                f"placement implies {len(placement)}"
            )
    elif ndim == 2 and len(placement) != 1:
        # TODO(EA2D): special case unnecessary with 2D EAs
        raise ValueError("need to split")


def extract_pandas_array(
    values: ArrayLike, dtype: DtypeObj | None, ndim: int
) -> tuple[ArrayLike, DtypeObj | None]:
    """
    Ensure that we don't allow NumpyExtensionArray / NumpyEADtype in internals.
    """
    # For now, blocks should be backed by ndarrays when possible.
    if isinstance(values, ABCNumpyExtensionArray):
        values = values.to_numpy()
        if ndim and ndim > 1:
            # TODO(EA2D): special case not needed with 2D EAs
            values = np.atleast_2d(values)

    if isinstance(dtype, NumpyEADtype):
        dtype = dtype.numpy_dtype

    return values, dtype


# -----------------------------------------------------------------


def extend_blocks(result, blocks=None) -> list[Block]:
    """return a new extended blocks, given the result"""
    if blocks is None:
        blocks = []
    if isinstance(result, list):
        for r in result:
            if isinstance(r, list):
                blocks.extend(r)
            else:
                blocks.append(r)
    else:
        assert isinstance(result, Block), type(result)
        blocks.append(result)
    return blocks


def ensure_block_shape(values: ArrayLike, ndim: int = 1) -> ArrayLike:
    """
    Reshape if possible to have values.ndim == ndim.
    """

    if values.ndim < ndim:
        if not is_1d_only_ea_dtype(values.dtype):
            # TODO(EA2D): https://github.com/pandas-dev/pandas/issues/23023
            # block.shape is incorrect for "2D" ExtensionArrays
            # We can't, and don't need to, reshape.
            values = cast("np.ndarray | DatetimeArray | TimedeltaArray", values)
            values = values.reshape(1, -1)

    return values


def external_values(values: ArrayLike) -> ArrayLike:
    """
    The array that Series.values returns (public attribute).

    This has some historical constraints, and is overridden in block
    subclasses to return the correct array (e.g. period returns
    object ndarray and datetimetz a datetime64[ns] ndarray instead of
    proper extension array).
    """
    if isinstance(values, (PeriodArray, IntervalArray)):
        return values.astype(object)
    elif isinstance(values, (DatetimeArray, TimedeltaArray)):
        # NB: for datetime64tz this is different from np.asarray(values), since
        #  that returns an object-dtype ndarray of Timestamps.
        # Avoid raising in .astype in casting from dt64tz to dt64
        values = values._ndarray

    if isinstance(values, np.ndarray) and using_copy_on_write():
        values = values.view()
        values.flags.writeable = False

    # TODO(CoW) we should also mark our ExtensionArrays as read-only

    return values
<|endoftext|>"""
This is a pseudo-public API for downstream libraries.  We ask that downstream
authors

1) Try to avoid using internals directly altogether, and failing that,
2) Use only functions exposed here (or in core.internals)

"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pandas._libs.internals import BlockPlacement

from pandas.core.dtypes.common import pandas_dtype
from pandas.core.dtypes.dtypes import (
    DatetimeTZDtype,
    PeriodDtype,
)

from pandas.core.arrays import DatetimeArray
from pandas.core.construction import extract_array
from pandas.core.internals.blocks import (
    check_ndim,
    ensure_block_shape,
    extract_pandas_array,
    get_block_type,
    maybe_coerce_values,
)

if TYPE_CHECKING:
    from pandas._typing import Dtype

    from pandas.core.internals.blocks import Block


def make_block(
    values, placement, klass=None, ndim=None, dtype: Dtype | None = None
) -> Block:
    """
    This is a pseudo-public analogue to blocks.new_block.

    We ask that downstream libraries use this rather than any fully-internal
    APIs, including but not limited to:

    - core.internals.blocks.make_block
    - Block.make_block
    - Block.make_block_same_class
    - Block.__init__
    """
    if dtype is not None:
        dtype = pandas_dtype(dtype)

    values, dtype = extract_pandas_array(values, dtype, ndim)

    from pandas.core.internals.blocks import (
        DatetimeTZBlock,
        ExtensionBlock,
    )

    if klass is ExtensionBlock and isinstance(values.dtype, PeriodDtype):
        # GH-44681 changed PeriodArray to be stored in the 2D
        # NDArrayBackedExtensionBlock instead of ExtensionBlock
        # -> still allow ExtensionBlock to be passed in this case for back compat
        klass = None

    if klass is None:
        dtype = dtype or values.dtype
        klass = get_block_type(dtype)

    elif klass is DatetimeTZBlock and not isinstance(values.dtype, DatetimeTZDtype):
        # pyarrow calls get here
        values = DatetimeArray._simple_new(
            # error: Argument "dtype" to "_simple_new" of "DatetimeArray" has
            # incompatible type "Union[ExtensionDtype, dtype[Any], None]";
            # expected "Union[dtype[datetime64], DatetimeTZDtype]"
            values,
            dtype=dtype,  # type: ignore[arg-type]
        )

    if not isinstance(placement, BlockPlacement):
        placement = BlockPlacement(placement)

    ndim = maybe_infer_ndim(values, placement, ndim)
    if isinstance(values.dtype, (PeriodDtype, DatetimeTZDtype)):
        # GH#41168 ensure we can pass 1D dt64tz values
        # More generally, any EA dtype that isn't is_1d_only_ea_dtype
        values = extract_array(values, extract_numpy=True)
        values = ensure_block_shape(values, ndim)

    check_ndim(values, placement, ndim)
    values = maybe_coerce_values(values)
    return klass(values, ndim=ndim, placement=placement)


def maybe_infer_ndim(values, placement: BlockPlacement, ndim: int | None) -> int:
    """
    If `ndim` is not provided, infer it from placement and values.
    """
    if ndim is None:
        # GH#38134 Block constructor now assumes ndim is not None
        if not isinstance(values.dtype, np.dtype):
            if len(placement) != 1:
                ndim = 1
            else:
                ndim = 2
        else:
            ndim = values.ndim
    return ndim


def __getattr__(name: str):
    # GH#55139
    import warnings

    if name in [
        "Block",
        "ExtensionBlock",
        "DatetimeTZBlock",
        "create_block_manager_from_blocks",
    ]:
        # GH#33892
        warnings.warn(
            f"{name} is deprecated and will be removed in a future version. "
            "Use public APIs instead.",
            DeprecationWarning,
            # https://github.com/pandas-dev/pandas/pull/55139#pullrequestreview-1720690758
            # on hard-coding stacklevel
            stacklevel=2,
        )

        if name == "create_block_manager_from_blocks":
            from pandas.core.internals.managers import create_block_manager_from_blocks

            return create_block_manager_from_blocks

        elif name == "Block":
            from pandas.core.internals.blocks import Block

            return Block

        elif name == "DatetimeTZBlock":
            from pandas.core.internals.blocks import DatetimeTZBlock

            return DatetimeTZBlock

        elif name == "ExtensionBlock":
            from pandas.core.internals.blocks import ExtensionBlock

            return ExtensionBlock

    raise AttributeError(
        f"module 'pandas.core.internals.api' has no attribute '{name}'"
    )
<|endoftext|>"""
Experimental manager based on storing a collection of 1D arrays
"""
from __future__ import annotations

import itertools
from typing import (
    TYPE_CHECKING,
    Callable,
    Literal,
)

import numpy as np

from pandas._libs import (
    NaT,
    lib,
)

from pandas.core.dtypes.astype import (
    astype_array,
    astype_array_safe,
)
from pandas.core.dtypes.cast import (
    ensure_dtype_can_hold_na,
    find_common_type,
    infer_dtype_from_scalar,
    np_find_common_type,
)
from pandas.core.dtypes.common import (
    ensure_platform_int,
    is_datetime64_ns_dtype,
    is_integer,
    is_numeric_dtype,
    is_object_dtype,
    is_timedelta64_ns_dtype,
)
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCSeries,
)
from pandas.core.dtypes.missing import (
    array_equals,
    isna,
    na_value_for_dtype,
)

import pandas.core.algorithms as algos
from pandas.core.array_algos.quantile import quantile_compat
from pandas.core.array_algos.take import take_1d
from pandas.core.arrays import (
    DatetimeArray,
    ExtensionArray,
    NumpyExtensionArray,
    TimedeltaArray,
)
from pandas.core.construction import (
    ensure_wrapped_if_datetimelike,
    extract_array,
    sanitize_array,
)
from pandas.core.indexers import (
    maybe_convert_indices,
    validate_indices,
)
from pandas.core.indexes.api import (
    Index,
    ensure_index,
)
from pandas.core.indexes.base import get_values_for_csv
from pandas.core.internals.base import (
    DataManager,
    SingleDataManager,
    ensure_np_dtype,
    interleaved_dtype,
)
from pandas.core.internals.blocks import (
    BlockPlacement,
    ensure_block_shape,
    external_values,
    extract_pandas_array,
    maybe_coerce_values,
    new_block,
)
from pandas.core.internals.managers import make_na_array

if TYPE_CHECKING:
    from collections.abc import Hashable

    from pandas._typing import (
        ArrayLike,
        AxisInt,
        DtypeObj,
        QuantileInterpolation,
        Self,
        npt,
    )


class BaseArrayManager(DataManager):
    """
    Core internal data structure to implement DataFrame and Series.

    Alternative to the BlockManager, storing a list of 1D arrays instead of
    Blocks.

    This is *not* a public API class

    Parameters
    ----------
    arrays : Sequence of arrays
    axes : Sequence of Index
    verify_integrity : bool, default True

    """

    __slots__ = [
        "_axes",  # private attribute, because 'axes' has different order, see below
        "arrays",
    ]

    arrays: list[np.ndarray | ExtensionArray]
    _axes: list[Index]

    def __init__(
        self,
        arrays: list[np.ndarray | ExtensionArray],
        axes: list[Index],
        verify_integrity: bool = True,
    ) -> None:
        raise NotImplementedError

    def make_empty(self, axes=None) -> Self:
        """Return an empty ArrayManager with the items axis of len 0 (no columns)"""
        if axes is None:
            axes = [self.axes[1:], Index([])]

        arrays: list[np.ndarray | ExtensionArray] = []
        return type(self)(arrays, axes)

    @property
    def items(self) -> Index:
        return self._axes[-1]

    @property
    # error: Signature of "axes" incompatible with supertype "DataManager"
    def axes(self) -> list[Index]:  # type: ignore[override]
        # mypy doesn't work to override attribute with property
        # see https://github.com/python/mypy/issues/4125
        """Axes is BlockManager-compatible order (columns, rows)"""
        return [self._axes[1], self._axes[0]]

    @property
    def shape_proper(self) -> tuple[int, ...]:
        # this returns (n_rows, n_columns)
        return tuple(len(ax) for ax in self._axes)

    @staticmethod
    def _normalize_axis(axis: AxisInt) -> int:
        # switch axis
        axis = 1 if axis == 0 else 0
        return axis

    def set_axis(self, axis: AxisInt, new_labels: Index) -> None:
        # Caller is responsible for ensuring we have an Index object.
        self._validate_set_axis(axis, new_labels)
        axis = self._normalize_axis(axis)
        self._axes[axis] = new_labels

    def get_dtypes(self) -> npt.NDArray[np.object_]:
        return np.array([arr.dtype for arr in self.arrays], dtype="object")

    def add_references(self, mgr: BaseArrayManager) -> None:
        """
        Only implemented on the BlockManager level
        """
        return

    def __getstate__(self):
        return self.arrays, self._axes

    def __setstate__(self, state) -> None:
        self.arrays = state[0]
        self._axes = state[1]

    def __repr__(self) -> str:
        output = type(self).__name__
        output += f"\nIndex: {self._axes[0]}"
        if self.ndim == 2:
            output += f"\nColumns: {self._axes[1]}"
        output += f"\n{len(self.arrays)} arrays:"
        for arr in self.arrays:
            output += f"\n{arr.dtype}"
        return output

    def apply(
        self,
        f,
        align_keys: list[str] | None = None,
        **kwargs,
    ) -> Self:
        """
        Iterate over the arrays, collect and create a new ArrayManager.

        Parameters
        ----------
        f : str or callable
            Name of the Array method to apply.
        align_keys: List[str] or None, default None
        **kwargs
            Keywords to pass to `f`

        Returns
        -------
        ArrayManager
        """
        assert "filter" not in kwargs

        align_keys = align_keys or []
        result_arrays: list[ArrayLike] = []
        # fillna: Series/DataFrame is responsible for making sure value is aligned

        aligned_args = {k: kwargs[k] for k in align_keys}

        if f == "apply":
            f = kwargs.pop("func")

        for i, arr in enumerate(self.arrays):
            if aligned_args:
                for k, obj in aligned_args.items():
                    if isinstance(obj, (ABCSeries, ABCDataFrame)):
                        # The caller is responsible for ensuring that
                        #  obj.axes[-1].equals(self.items)
                        if obj.ndim == 1:
                            kwargs[k] = obj.iloc[i]
                        else:
                            kwargs[k] = obj.iloc[:, i]._values
                    else:
                        # otherwise we have an array-like
                        kwargs[k] = obj[i]

            if callable(f):
                applied = f(arr, **kwargs)
            else:
                applied = getattr(arr, f)(**kwargs)

            result_arrays.append(applied)

        new_axes = self._axes
        return type(self)(result_arrays, new_axes)

    def apply_with_block(self, f, align_keys=None, **kwargs) -> Self:
        # switch axis to follow BlockManager logic
        swap_axis = True
        if f == "interpolate":
            swap_axis = False
        if swap_axis and "axis" in kwargs and self.ndim == 2:
            kwargs["axis"] = 1 if kwargs["axis"] == 0 else 0

        align_keys = align_keys or []
        aligned_args = {k: kwargs[k] for k in align_keys}

        result_arrays = []

        for i, arr in enumerate(self.arrays):
            if aligned_args:
                for k, obj in aligned_args.items():
                    if isinstance(obj, (ABCSeries, ABCDataFrame)):
                        # The caller is responsible for ensuring that
                        #  obj.axes[-1].equals(self.items)
                        if obj.ndim == 1:
                            if self.ndim == 2:
                                kwargs[k] = obj.iloc[slice(i, i + 1)]._values
                            else:
                                kwargs[k] = obj.iloc[:]._values
                        else:
                            kwargs[k] = obj.iloc[:, [i]]._values
                    else:
                        # otherwise we have an ndarray
                        if obj.ndim == 2:
                            kwargs[k] = obj[[i]]

            if isinstance(arr.dtype, np.dtype) and not isinstance(arr, np.ndarray):
                # i.e. TimedeltaArray, DatetimeArray with tz=None. Need to
                #  convert for the Block constructors.
                arr = np.asarray(arr)

            arr = maybe_coerce_values(arr)
            if self.ndim == 2:
                arr = ensure_block_shape(arr, 2)
                bp = BlockPlacement(slice(0, 1, 1))
                block = new_block(arr, placement=bp, ndim=2)
            else:
                bp = BlockPlacement(slice(0, len(self), 1))
                block = new_block(arr, placement=bp, ndim=1)

            applied = getattr(block, f)(**kwargs)
            if isinstance(applied, list):
                applied = applied[0]
            arr = applied.values
            if self.ndim == 2 and arr.ndim == 2:
                # 2D for np.ndarray or DatetimeArray/TimedeltaArray
                assert len(arr) == 1
                # error: No overload variant of "__getitem__" of "ExtensionArray"
                # matches argument type "Tuple[int, slice]"
                arr = arr[0, :]  # type: ignore[call-overload]
            result_arrays.append(arr)

        return type(self)(result_arrays, self._axes)

    def setitem(self, indexer, value, warn: bool = True) -> Self:
        return self.apply_with_block("setitem", indexer=indexer, value=value)

    def diff(self, n: int) -> Self:
        assert self.ndim == 2  # caller ensures
        return self.apply(algos.diff, n=n)

    def astype(self, dtype, copy: bool | None = False, errors: str = "raise") -> Self:
        if copy is None:
            copy = True

        return self.apply(astype_array_safe, dtype=dtype, copy=copy, errors=errors)

    def convert(self, copy: bool | None) -> Self:
        if copy is None:
            copy = True

        def _convert(arr):
            if is_object_dtype(arr.dtype):
                # extract NumpyExtensionArray for tests that patch
                #  NumpyExtensionArray._typ
                arr = np.asarray(arr)
                result = lib.maybe_convert_objects(
                    arr,
                    convert_non_numeric=True,
                )
                if result is arr and copy:
                    return arr.copy()
                return result
            else:
                return arr.copy() if copy else arr

        return self.apply(_convert)

    def get_values_for_csv(
        self, *, float_format, date_format, decimal, na_rep: str = "nan", quoting=None
    ) -> Self:
        return self.apply(
            get_values_for_csv,
            na_rep=na_rep,
            quoting=quoting,
            float_format=float_format,
            date_format=date_format,
            decimal=decimal,
        )

    @property
    def any_extension_types(self) -> bool:
        """Whether any of the blocks in this manager are extension blocks"""
        return False  # any(block.is_extension for block in self.blocks)

    @property
    def is_view(self) -> bool:
        """return a boolean if we are a single block and are a view"""
        # TODO what is this used for?
        return False

    @property
    def is_single_block(self) -> bool:
        return len(self.arrays) == 1

    def _get_data_subset(self, predicate: Callable) -> Self:
        indices = [i for i, arr in enumerate(self.arrays) if predicate(arr)]
        arrays = [self.arrays[i] for i in indices]
        # TODO copy?
        # Note: using Index.take ensures we can retain e.g. DatetimeIndex.freq,
        #  see test_describe_datetime_columns
        taker = np.array(indices, dtype="intp")
        new_cols = self._axes[1].take(taker)
        new_axes = [self._axes[0], new_cols]
        return type(self)(arrays, new_axes, verify_integrity=False)

    def get_bool_data(self, copy: bool = False) -> Self:
        """
        Select columns that are bool-dtype and object-dtype columns that are all-bool.

        Parameters
        ----------
        copy : bool, default False
            Whether to copy the blocks
        """
        return self._get_data_subset(lambda x: x.dtype == np.dtype(bool))

    def get_numeric_data(self, copy: bool = False) -> Self:
        """
        Select columns that have a numeric dtype.

        Parameters
        ----------
        copy : bool, default False
            Whether to copy the blocks
        """
        return self._get_data_subset(
            lambda arr: is_numeric_dtype(arr.dtype)
            or getattr(arr.dtype, "_is_numeric", False)
        )

    def copy(self, deep: bool | Literal["all"] | None = True) -> Self:
        """
        Make deep or shallow copy of ArrayManager

        Parameters
        ----------
        deep : bool or string, default True
            If False, return shallow copy (do not copy data)
            If 'all', copy data and a deep copy of the index

        Returns
        -------
        BlockManager
        """
        if deep is None:
            # ArrayManager does not yet support CoW, so deep=None always means
            # deep=True for now
            deep = True

        # this preserves the notion of view copying of axes
        if deep:
            # hit in e.g. tests.io.json.test_pandas

            def copy_func(ax):
                return ax.copy(deep=True) if deep == "all" else ax.view()

            new_axes = [copy_func(ax) for ax in self._axes]
        else:
            new_axes = list(self._axes)

        if deep:
            new_arrays = [arr.copy() for arr in self.arrays]
        else:
            new_arrays = list(self.arrays)
        return type(self)(new_arrays, new_axes, verify_integrity=False)

    def reindex_indexer(
        self,
        new_axis,
        indexer,
        axis: AxisInt,
        fill_value=None,
        allow_dups: bool = False,
        copy: bool | None = True,
        # ignored keywords
        only_slice: bool = False,
        # ArrayManager specific keywords
        use_na_proxy: bool = False,
    ) -> Self:
        axis = self._normalize_axis(axis)
        return self._reindex_indexer(
            new_axis,
            indexer,
            axis,
            fill_value,
            allow_dups,
            copy,
            use_na_proxy,
        )

    def _reindex_indexer(
        self,
        new_axis,
        indexer: npt.NDArray[np.intp] | None,
        axis: AxisInt,
        fill_value=None,
        allow_dups: bool = False,
        copy: bool | None = True,
        use_na_proxy: bool = False,
    ) -> Self:
        """
        Parameters
        ----------
        new_axis : Index
        indexer : ndarray[intp] or None
        axis : int
        fill_value : object, default None
        allow_dups : bool, default False
        copy : bool, default True


        pandas-indexer with -1's only.
        """
        if copy is None:
            # ArrayManager does not yet support CoW, so deep=None always means
            # deep=True for now
            copy = True

        if indexer is None:
            if new_axis is self._axes[axis] and not copy:
                return self

            result = self.copy(deep=copy)
            result._axes = list(self._axes)
            result._axes[axis] = new_axis
            return result

        # some axes don't allow reindexing with dups
        if not allow_dups:
            self._axes[axis]._validate_can_reindex(indexer)

        if axis >= self.ndim:
            raise IndexError("Requested axis not found in manager")

        if axis == 1:
            new_arrays = []
            for i in indexer:
                if i == -1:
                    arr = self._make_na_array(
                        fill_value=fill_value, use_na_proxy=use_na_proxy
                    )
                else:
                    arr = self.arrays[i]
                    if copy:
                        arr = arr.copy()
                new_arrays.append(arr)

        else:
            validate_indices(indexer, len(self._axes[0]))
            indexer = ensure_platform_int(indexer)
            mask = indexer == -1
            needs_masking = mask.any()
            new_arrays = [
                take_1d(
                    arr,
                    indexer,
                    allow_fill=needs_masking,
                    fill_value=fill_value,
                    mask=mask,
                    # if fill_value is not None else blk.fill_value
                )
                for arr in self.arrays
            ]

        new_axes = list(self._axes)
        new_axes[axis] = new_axis

        return type(self)(new_arrays, new_axes, verify_integrity=False)

    def take(
        self,
        indexer: npt.NDArray[np.intp],
        axis: AxisInt = 1,
        verify: bool = True,
    ) -> Self:
        """
        Take items along any axis.
        """
        assert isinstance(indexer, np.ndarray), type(indexer)
        assert indexer.dtype == np.intp, indexer.dtype

        axis = self._normalize_axis(axis)

        if not indexer.ndim == 1:
            raise ValueError("indexer should be 1-dimensional")

        n = self.shape_proper[axis]
        indexer = maybe_convert_indices(indexer, n, verify=verify)

        new_labels = self._axes[axis].take(indexer)
        return self._reindex_indexer(
            new_axis=new_labels, indexer=indexer, axis=axis, allow_dups=True
        )

    def _make_na_array(self, fill_value=None, use_na_proxy: bool = False):
        if use_na_proxy:
            assert fill_value is None
            return NullArrayProxy(self.shape_proper[0])

        if fill_value is None:
            fill_value = np.nan

        dtype, fill_value = infer_dtype_from_scalar(fill_value)
        array_values = make_na_array(dtype, self.shape_proper[:1], fill_value)
        return array_values

    def _equal_values(self, other) -> bool:
        """
        Used in .equals defined in base class. Only check the column values
        assuming shape and indexes have already been checked.
        """
        for left, right in zip(self.arrays, other.arrays):
            if not array_equals(left, right):
                return False
        return True

    # TODO
    # to_dict


class ArrayManager(BaseArrayManager):
    @property
    def ndim(self) -> Literal[2]:
        return 2

    def __init__(
        self,
        arrays: list[np.ndarray | ExtensionArray],
        axes: list[Index],
        verify_integrity: bool = True,
    ) -> None:
        # Note: we are storing the axes in "_axes" in the (row, columns) order
        # which contrasts the order how it is stored in BlockManager
        self._axes = axes
        self.arrays = arrays

        if verify_integrity:
            self._axes = [ensure_index(ax) for ax in axes]
            arrays = [extract_pandas_array(x, None, 1)[0] for x in arrays]
            self.arrays = [maybe_coerce_values(arr) for arr in arrays]
            self._verify_integrity()

    def _verify_integrity(self) -> None:
        n_rows, n_columns = self.shape_proper
        if not len(self.arrays) == n_columns:
            raise ValueError(
                "Number of passed arrays must equal the size of the column Index: "
                f"{len(self.arrays)} arrays vs {n_columns} columns."
            )
        for arr in self.arrays:
            if not len(arr) == n_rows:
                raise ValueError(
                    "Passed arrays should have the same length as the rows Index: "
                    f"{len(arr)} vs {n_rows} rows"
                )
            if not isinstance(arr, (np.ndarray, ExtensionArray)):
                raise ValueError(
                    "Passed arrays should be np.ndarray or ExtensionArray instances, "
                    f"got {type(arr)} instead"
                )
            if not arr.ndim == 1:
                raise ValueError(
                    "Passed arrays should be 1-dimensional, got array with "
                    f"{arr.ndim} dimensions instead."
                )

    # --------------------------------------------------------------------
    # Indexing

    def fast_xs(self, loc: int) -> SingleArrayManager:
        """
        Return the array corresponding to `frame.iloc[loc]`.

        Parameters
        ----------
        loc : int

        Returns
        -------
        np.ndarray or ExtensionArray
        """
        dtype = interleaved_dtype([arr.dtype for arr in self.arrays])

        values = [arr[loc] for arr in self.arrays]
        if isinstance(dtype, ExtensionDtype):
            result = dtype.construct_array_type()._from_sequence(values, dtype=dtype)
        # for datetime64/timedelta64, the np.ndarray constructor cannot handle pd.NaT
        elif is_datetime64_ns_dtype(dtype):
            result = DatetimeArray._from_sequence(values, dtype=dtype)._ndarray
        elif is_timedelta64_ns_dtype(dtype):
            result = TimedeltaArray._from_sequence(values, dtype=dtype)._ndarray
        else:
            result = np.array(values, dtype=dtype)
        return SingleArrayManager([result], [self._axes[1]])

    def get_slice(self, slobj: slice, axis: AxisInt = 0) -> ArrayManager:
        axis = self._normalize_axis(axis)

        if axis == 0:
            arrays = [arr[slobj] for arr in self.arrays]
        elif axis == 1:
            arrays = self.arrays[slobj]

        new_axes = list(self._axes)
        new_axes[axis] = new_axes[axis]._getitem_slice(slobj)

        return type(self)(arrays, new_axes, verify_integrity=False)

    def iget(self, i: int) -> SingleArrayManager:
        """
        Return the data as a SingleArrayManager.
        """
        values = self.arrays[i]
        return SingleArrayManager([values], [self._axes[0]])

    def iget_values(self, i: int) -> ArrayLike:
        """
        Return the data for column i as the values (ndarray or ExtensionArray).
        """
        return self.arrays[i]

    @property
    def column_arrays(self) -> list[ArrayLike]:
        """
        Used in the JSON C code to access column arrays.
        """

        return [np.asarray(arr) for arr in self.arrays]

    def iset(
        self,
        loc: int | slice | np.ndarray,
        value: ArrayLike,
        inplace: bool = False,
        refs=None,
    ) -> None:
        """
        Set new column(s).

        This changes the ArrayManager in-place, but replaces (an) existing
        column(s), not changing column values in-place).

        Parameters
        ----------
        loc : integer, slice or boolean mask
            Positional location (already bounds checked)
        value : np.ndarray or ExtensionArray
        inplace : bool, default False
            Whether overwrite existing array as opposed to replacing it.
        """
        # single column -> single integer index
        if lib.is_integer(loc):
            # TODO can we avoid needing to unpack this here? That means converting
            # DataFrame into 1D array when loc is an integer
            if isinstance(value, np.ndarray) and value.ndim == 2:
                assert value.shape[1] == 1
                value = value[:, 0]

            # TODO we receive a datetime/timedelta64 ndarray from DataFrame._iset_item
            # but we should avoid that and pass directly the proper array
            value = maybe_coerce_values(value)

            assert isinstance(value, (np.ndarray, ExtensionArray))
            assert value.ndim == 1
            assert len(value) == len(self._axes[0])
            self.arrays[loc] = value
            return

        # multiple columns -> convert slice or array to integer indices
        elif isinstance(loc, slice):
            indices: range | np.ndarray = range(
                loc.start if loc.start is not None else 0,
                loc.stop if loc.stop is not None else self.shape_proper[1],
                loc.step if loc.step is not None else 1,
            )
        else:
            assert isinstance(loc, np.ndarray)
            assert loc.dtype == "bool"
            indices = np.nonzero(loc)[0]

        assert value.ndim == 2
        assert value.shape[0] == len(self._axes[0])

        for value_idx, mgr_idx in enumerate(indices):
            # error: No overload variant of "__getitem__" of "ExtensionArray" matches
            # argument type "Tuple[slice, int]"
            value_arr = value[:, value_idx]  # type: ignore[call-overload]
            self.arrays[mgr_idx] = value_arr
        return

    def column_setitem(
        self, loc: int, idx: int | slice | np.ndarray, value, inplace_only: bool = False
    ) -> None:
        """
        Set values ("setitem") into a single column (not setting the full column).

        This is a method on the ArrayManager level, to avoid creating an
        intermediate Series at the DataFrame level (`s = df[loc]; s[idx] = value`)
        """
        if not is_integer(loc):
            raise TypeError("The column index should be an integer")
        arr = self.arrays[loc]
        mgr = SingleArrayManager([arr], [self._axes[0]])
        if inplace_only:
            mgr.setitem_inplace(idx, value)
        else:
            new_mgr = mgr.setitem((idx,), value)
            # update existing ArrayManager in-place
            self.arrays[loc] = new_mgr.arrays[0]

    def insert(self, loc: int, item: Hashable, value: ArrayLike, refs=None) -> None:
        """
        Insert item at selected position.

        Parameters
        ----------
        loc : int
        item : hashable
        value : np.ndarray or ExtensionArray
        """
        # insert to the axis; this could possibly raise a TypeError
        new_axis = self.items.insert(loc, item)

        value = extract_array(value, extract_numpy=True)
        if value.ndim == 2:
            if value.shape[0] == 1:
                # error: No overload variant of "__getitem__" of "ExtensionArray"
                # matches argument type "Tuple[int, slice]"
                value = value[0, :]  # type: ignore[call-overload]
            else:
                raise ValueError(
                    f"Expected a 1D array, got an array with shape {value.shape}"
                )
        value = maybe_coerce_values(value)

        # TODO self.arrays can be empty
        # assert len(value) == len(self.arrays[0])

        # TODO is this copy needed?
        arrays = self.arrays.copy()
        arrays.insert(loc, value)

        self.arrays = arrays
        self._axes[1] = new_axis

    def idelete(self, indexer) -> ArrayManager:
        """
        Delete selected locations in-place (new block and array, same BlockManager)
        """
        to_keep = np.ones(self.shape[0], dtype=np.bool_)
        to_keep[indexer] = False

        self.arrays = [self.arrays[i] for i in np.nonzero(to_keep)[0]]
        self._axes = [self._axes[0], self._axes[1][to_keep]]
        return self

    # --------------------------------------------------------------------
    # Array-wise Operation

    def grouped_reduce(self, func: Callable) -> Self:
        """
        Apply grouped reduction function columnwise, returning a new ArrayManager.

        Parameters
        ----------
        func : grouped reduction function

        Returns
        -------
        ArrayManager
        """
        result_arrays: list[np.ndarray] = []
        result_indices: list[int] = []

        for i, arr in enumerate(self.arrays):
            # grouped_reduce functions all expect 2D arrays
            arr = ensure_block_shape(arr, ndim=2)
            res = func(arr)
            if res.ndim == 2:
                # reverse of ensure_block_shape
                assert res.shape[0] == 1
                res = res[0]

            result_arrays.append(res)
            result_indices.append(i)

        if len(result_arrays) == 0:
            nrows = 0
        else:
            nrows = result_arrays[0].shape[0]
        index = Index(range(nrows))

        columns = self.items

        # error: Argument 1 to "ArrayManager" has incompatible type "List[ndarray]";
        # expected "List[Union[ndarray, ExtensionArray]]"
        return type(self)(result_arrays, [index, columns])  # type: ignore[arg-type]

    def reduce(self, func: Callable) -> Self:
        """
        Apply reduction function column-wise, returning a single-row ArrayManager.

        Parameters
        ----------
        func : reduction function

        Returns
        -------
        ArrayManager
        """
        result_arrays: list[np.ndarray] = []
        for i, arr in enumerate(self.arrays):
            res = func(arr, axis=0)

            # TODO NaT doesn't preserve dtype, so we need to ensure to create
            # a timedelta result array if original was timedelta
            # what if datetime results in timedelta? (eg std)
            dtype = arr.dtype if res is NaT else None
            result_arrays.append(
                sanitize_array([res], None, dtype=dtype)  # type: ignore[arg-type]
            )

        index = Index._simple_new(np.array([None], dtype=object))  # placeholder
        columns = self.items

        # error: Argument 1 to "ArrayManager" has incompatible type "List[ndarray]";
        # expected "List[Union[ndarray, ExtensionArray]]"
        new_mgr = type(self)(result_arrays, [index, columns])  # type: ignore[arg-type]
        return new_mgr

    def operate_blockwise(self, other: ArrayManager, array_op) -> ArrayManager:
        """
        Apply array_op blockwise with another (aligned) BlockManager.
        """
        # TODO what if `other` is BlockManager ?
        left_arrays = self.arrays
        right_arrays = other.arrays
        result_arrays = [
            array_op(left, right) for left, right in zip(left_arrays, right_arrays)
        ]
        return type(self)(result_arrays, self._axes)

    def quantile(
        self,
        *,
        qs: Index,  # with dtype float64
        transposed: bool = False,
        interpolation: QuantileInterpolation = "linear",
    ) -> ArrayManager:
        arrs = [ensure_block_shape(x, 2) for x in self.arrays]
        new_arrs = [
            quantile_compat(x, np.asarray(qs._values), interpolation) for x in arrs
        ]
        for i, arr in enumerate(new_arrs):
            if arr.ndim == 2:
                assert arr.shape[0] == 1, arr.shape
                new_arrs[i] = arr[0]

        axes = [qs, self._axes[1]]
        return type(self)(new_arrs, axes)

    # ----------------------------------------------------------------

    def unstack(self, unstacker, fill_value) -> ArrayManager:
        """
        Return a BlockManager with all blocks unstacked.

        Parameters
        ----------
        unstacker : reshape._Unstacker
        fill_value : Any
            fill_value for newly introduced missing values.

        Returns
        -------
        unstacked : BlockManager
        """
        indexer, _ = unstacker._indexer_and_to_sort
        if unstacker.mask.all():
            new_indexer = indexer
            allow_fill = False
            new_mask2D = None
            needs_masking = None
        else:
            new_indexer = np.full(unstacker.mask.shape, -1)
            new_indexer[unstacker.mask] = indexer
            allow_fill = True
            # calculating the full mask once and passing it to take_1d is faster
            # than letting take_1d calculate it in each repeated call
            new_mask2D = (~unstacker.mask).reshape(*unstacker.full_shape)
            needs_masking = new_mask2D.any(axis=0)
        new_indexer2D = new_indexer.reshape(*unstacker.full_shape)
        new_indexer2D = ensure_platform_int(new_indexer2D)

        new_arrays = []
        for arr in self.arrays:
            for i in range(unstacker.full_shape[1]):
                if allow_fill:
                    # error: Value of type "Optional[Any]" is not indexable  [index]
                    new_arr = take_1d(
                        arr,
                        new_indexer2D[:, i],
                        allow_fill=needs_masking[i],  # type: ignore[index]
                        fill_value=fill_value,
                        mask=new_mask2D[:, i],  # type: ignore[index]
                    )
                else:
                    new_arr = take_1d(arr, new_indexer2D[:, i], allow_fill=False)
                new_arrays.append(new_arr)

        new_index = unstacker.new_index
        new_columns = unstacker.get_new_columns(self._axes[1])
        new_axes = [new_index, new_columns]

        return type(self)(new_arrays, new_axes, verify_integrity=False)

    def as_array(
        self,
        dtype=None,
        copy: bool = False,
        na_value: object = lib.no_default,
    ) -> np.ndarray:
        """
        Convert the blockmanager data into an numpy array.

        Parameters
        ----------
        dtype : object, default None
            Data type of the return array.
        copy : bool, default False
            If True then guarantee that a copy is returned. A value of
            False does not guarantee that the underlying data is not
            copied.
        na_value : object, default lib.no_default
            Value to be used as the missing value sentinel.

        Returns
        -------
        arr : ndarray
        """
        if len(self.arrays) == 0:
            empty_arr = np.empty(self.shape, dtype=float)
            return empty_arr.transpose()

        # We want to copy when na_value is provided to avoid
        # mutating the original object
        copy = copy or na_value is not lib.no_default

        if not dtype:
            dtype = interleaved_dtype([arr.dtype for arr in self.arrays])

        dtype = ensure_np_dtype(dtype)

        result = np.empty(self.shape_proper, dtype=dtype)

        for i, arr in enumerate(self.arrays):
            arr = arr.astype(dtype, copy=copy)
            result[:, i] = arr

        if na_value is not lib.no_default:
            result[isna(result)] = na_value

        return result

    @classmethod
    def concat_horizontal(cls, mgrs: list[Self], axes: list[Index]) -> Self:
        """
        Concatenate uniformly-indexed ArrayManagers horizontally.
        """
        # concatting along the columns -> combine reindexed arrays in a single manager
        arrays = list(itertools.chain.from_iterable([mgr.arrays for mgr in mgrs]))
        new_mgr = cls(arrays, [axes[1], axes[0]], verify_integrity=False)
        return new_mgr

    @classmethod
    def concat_vertical(cls, mgrs: list[Self], axes: list[Index]) -> Self:
        """
        Concatenate uniformly-indexed ArrayManagers vertically.
        """
        # concatting along the rows -> concat the reindexed arrays
        # TODO(ArrayManager) doesn't yet preserve the correct dtype
        arrays = [
            concat_arrays([mgrs[i].arrays[j] for i in range(len(mgrs))])
            for j in range(len(mgrs[0].arrays))
        ]
        new_mgr = cls(arrays, [axes[1], axes[0]], verify_integrity=False)
        return new_mgr


class SingleArrayManager(BaseArrayManager, SingleDataManager):
    __slots__ = [
        "_axes",  # private attribute, because 'axes' has different order, see below
        "arrays",
    ]

    arrays: list[np.ndarray | ExtensionArray]
    _axes: list[Index]

    @property
    def ndim(self) -> Literal[1]:
        return 1

    def __init__(
        self,
        arrays: list[np.ndarray | ExtensionArray],
        axes: list[Index],
        verify_integrity: bool = True,
    ) -> None:
        self._axes = axes
        self.arrays = arrays

        if verify_integrity:
            assert len(axes) == 1
            assert len(arrays) == 1
            self._axes = [ensure_index(ax) for ax in self._axes]
            arr = arrays[0]
            arr = maybe_coerce_values(arr)
            arr = extract_pandas_array(arr, None, 1)[0]
            self.arrays = [arr]
            self._verify_integrity()

    def _verify_integrity(self) -> None:
        (n_rows,) = self.shape
        assert len(self.arrays) == 1
        arr = self.arrays[0]
        assert len(arr) == n_rows
        if not arr.ndim == 1:
            raise ValueError(
                "Passed array should be 1-dimensional, got array with "
                f"{arr.ndim} dimensions instead."
            )

    @staticmethod
    def _normalize_axis(axis):
        return axis

    def make_empty(self, axes=None) -> Self:
        """Return an empty ArrayManager with index/array of length 0"""
        if axes is None:
            axes = [Index([], dtype=object)]
        array: np.ndarray = np.array([], dtype=self.dtype)
        return type(self)([array], axes)

    @classmethod
    def from_array(cls, array, index) -> SingleArrayManager:
        return cls([array], [index])

    # error: Cannot override writeable attribute with read-only property
    @property
    def axes(self) -> list[Index]:  # type: ignore[override]
        return self._axes

    @property
    def index(self) -> Index:
        return self._axes[0]

    @property
    def dtype(self):
        return self.array.dtype

    def external_values(self):
        """The array that Series.values returns"""
        return external_values(self.array)

    def internal_values(self):
        """The array that Series._values returns"""
        return self.array

    def array_values(self):
        """The array that Series.array returns"""
        arr = self.array
        if isinstance(arr, np.ndarray):
            arr = NumpyExtensionArray(arr)
        return arr

    @property
    def _can_hold_na(self) -> bool:
        if isinstance(self.array, np.ndarray):
            return self.array.dtype.kind not in "iub"
        else:
            # ExtensionArray
            return self.array._can_hold_na

    @property
    def is_single_block(self) -> bool:
        return True

    def fast_xs(self, loc: int) -> SingleArrayManager:
        raise NotImplementedError("Use series._values[loc] instead")

    def get_slice(self, slobj: slice, axis: AxisInt = 0) -> SingleArrayManager:
        if axis >= self.ndim:
            raise IndexError("Requested axis not found in manager")

        new_array = self.array[slobj]
        new_index = self.index._getitem_slice(slobj)
        return type(self)([new_array], [new_index], verify_integrity=False)

    def get_rows_with_mask(self, indexer: npt.NDArray[np.bool_]) -> SingleArrayManager:
        new_array = self.array[indexer]
        new_index = self.index[indexer]
        return type(self)([new_array], [new_index])

    # error: Signature of "apply" incompatible with supertype "BaseArrayManager"
    def apply(self, func, **kwargs) -> Self:  # type: ignore[override]
        if callable(func):
            new_array = func(self.array, **kwargs)
        else:
            new_array = getattr(self.array, func)(**kwargs)
        return type(self)([new_array], self._axes)

    def setitem(self, indexer, value, warn: bool = True) -> SingleArrayManager:
        """
        Set values with indexer.

        For SingleArrayManager, this backs s[indexer] = value

        See `setitem_inplace` for a version that works inplace and doesn't
        return a new Manager.
        """
        if isinstance(indexer, np.ndarray) and indexer.ndim > self.ndim:
            raise ValueError(f"Cannot set values with ndim > {self.ndim}")
        return self.apply_with_block("setitem", indexer=indexer, value=value)

    def idelete(self, indexer) -> SingleArrayManager:
        """
        Delete selected locations in-place (new array, same ArrayManager)
        """
        to_keep = np.ones(self.shape[0], dtype=np.bool_)
        to_keep[indexer] = False

        self.arrays = [self.arrays[0][to_keep]]
        self._axes = [self._axes[0][to_keep]]
        return self

    def _get_data_subset(self, predicate: Callable) -> SingleArrayManager:
        # used in get_numeric_data / get_bool_data
        if predicate(self.array):
            return type(self)(self.arrays, self._axes, verify_integrity=False)
        else:
            return self.make_empty()

    def set_values(self, values: ArrayLike) -> None:
        """
        Set (replace) the values of the SingleArrayManager in place.

        Use at your own risk! This does not check if the passed values are
        valid for the current SingleArrayManager (length, dtype, etc).
        """
        self.arrays[0] = values

    def to_2d_mgr(self, columns: Index) -> ArrayManager:
        """
        Manager analogue of Series.to_frame
        """
        arrays = [self.arrays[0]]
        axes = [self.axes[0], columns]

        return ArrayManager(arrays, axes, verify_integrity=False)


class NullArrayProxy:
    """
    Proxy object for an all-NA array.

    Only stores the length of the array, and not the dtype. The dtype
    will only be known when actually concatenating (after determining the
    common dtype, for which this proxy is ignored).
    Using this object avoids that the internals/concat.py needs to determine
    the proper dtype and array type.
    """

    ndim = 1

    def __init__(self, n: int) -> None:
        self.n = n

    @property
    def shape(self) -> tuple[int]:
        return (self.n,)

    def to_array(self, dtype: DtypeObj) -> ArrayLike:
        """
        Helper function to create the actual all-NA array from the NullArrayProxy
        object.

        Parameters
        ----------
        arr : NullArrayProxy
        dtype : the dtype for the resulting array

        Returns
        -------
        np.ndarray or ExtensionArray
        """
        if isinstance(dtype, ExtensionDtype):
            empty = dtype.construct_array_type()._from_sequence([], dtype=dtype)
            indexer = -np.ones(self.n, dtype=np.intp)
            return empty.take(indexer, allow_fill=True)
        else:
            # when introducing missing values, int becomes float, bool becomes object
            dtype = ensure_dtype_can_hold_na(dtype)
            fill_value = na_value_for_dtype(dtype)
            arr = np.empty(self.n, dtype=dtype)
            arr.fill(fill_value)
            return ensure_wrapped_if_datetimelike(arr)


def concat_arrays(to_concat: list) -> ArrayLike:
    """
    Alternative for concat_compat but specialized for use in the ArrayManager.

    Differences: only deals with 1D arrays (no axis keyword), assumes
    ensure_wrapped_if_datetimelike and does not skip empty arrays to determine
    the dtype.
    In addition ensures that all NullArrayProxies get replaced with actual
    arrays.

    Parameters
    ----------
    to_concat : list of arrays

    Returns
    -------
    np.ndarray or ExtensionArray
    """
    # ignore the all-NA proxies to determine the resulting dtype
    to_concat_no_proxy = [x for x in to_concat if not isinstance(x, NullArrayProxy)]

    dtypes = {x.dtype for x in to_concat_no_proxy}
    single_dtype = len(dtypes) == 1

    if single_dtype:
        target_dtype = to_concat_no_proxy[0].dtype
    elif all(lib.is_np_dtype(x, "iub") for x in dtypes):
        # GH#42092
        target_dtype = np_find_common_type(*dtypes)
    else:
        target_dtype = find_common_type([arr.dtype for arr in to_concat_no_proxy])

    to_concat = [
        arr.to_array(target_dtype)
        if isinstance(arr, NullArrayProxy)
        else astype_array(arr, target_dtype, copy=False)
        for arr in to_concat
    ]

    if isinstance(to_concat[0], ExtensionArray):
        cls = type(to_concat[0])
        return cls._concat_same_type(to_concat)

    result = np.concatenate(to_concat)

    # TODO decide on exact behaviour (we shouldn't do this only for empty result)
    # see https://github.com/pandas-dev/pandas/issues/39817
    if len(result) == 0:
        # all empties -> check for bool to not coerce to float
        kinds = {obj.dtype.kind for obj in to_concat_no_proxy}
        if len(kinds) != 1:
            if "b" in kinds:
                result = result.astype(object)
    return result
<|endoftext|>from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    NamedTuple,
)

from pandas.core.dtypes.common import is_1d_only_ea_dtype

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pandas._libs.internals import BlockPlacement
    from pandas._typing import ArrayLike

    from pandas.core.internals.blocks import Block
    from pandas.core.internals.managers import BlockManager


class BlockPairInfo(NamedTuple):
    lvals: ArrayLike
    rvals: ArrayLike
    locs: BlockPlacement
    left_ea: bool
    right_ea: bool
    rblk: Block


def _iter_block_pairs(
    left: BlockManager, right: BlockManager
) -> Iterator[BlockPairInfo]:
    # At this point we have already checked the parent DataFrames for
    #  assert rframe._indexed_same(lframe)

    for blk in left.blocks:
        locs = blk.mgr_locs
        blk_vals = blk.values

        left_ea = blk_vals.ndim == 1

        rblks = right._slice_take_blocks_ax0(locs.indexer, only_slice=True)

        # Assertions are disabled for performance, but should hold:
        # if left_ea:
        #    assert len(locs) == 1, locs
        #    assert len(rblks) == 1, rblks
        #    assert rblks[0].shape[0] == 1, rblks[0].shape

        for rblk in rblks:
            right_ea = rblk.values.ndim == 1

            lvals, rvals = _get_same_shape_values(blk, rblk, left_ea, right_ea)
            info = BlockPairInfo(lvals, rvals, locs, left_ea, right_ea, rblk)
            yield info


def operate_blockwise(
    left: BlockManager, right: BlockManager, array_op
) -> BlockManager:
    # At this point we have already checked the parent DataFrames for
    #  assert rframe._indexed_same(lframe)

    res_blks: list[Block] = []
    for lvals, rvals, locs, left_ea, right_ea, rblk in _iter_block_pairs(left, right):
        res_values = array_op(lvals, rvals)
        if (
            left_ea
            and not right_ea
            and hasattr(res_values, "reshape")
            and not is_1d_only_ea_dtype(res_values.dtype)
        ):
            res_values = res_values.reshape(1, -1)
        nbs = rblk._split_op_result(res_values)

        # Assertions are disabled for performance, but should hold:
        # if right_ea or left_ea:
        #    assert len(nbs) == 1
        # else:
        #    assert res_values.shape == lvals.shape, (res_values.shape, lvals.shape)

        _reset_block_mgr_locs(nbs, locs)

        res_blks.extend(nbs)

    # Assertions are disabled for performance, but should hold:
    #  slocs = {y for nb in res_blks for y in nb.mgr_locs.as_array}
    #  nlocs = sum(len(nb.mgr_locs.as_array) for nb in res_blks)
    #  assert nlocs == len(left.items), (nlocs, len(left.items))
    #  assert len(slocs) == nlocs, (len(slocs), nlocs)
    #  assert slocs == set(range(nlocs)), slocs

    new_mgr = type(right)(tuple(res_blks), axes=right.axes, verify_integrity=False)
    return new_mgr


def _reset_block_mgr_locs(nbs: list[Block], locs) -> None:
    """
    Reset mgr_locs to correspond to our original DataFrame.
    """
    for nb in nbs:
        nblocs = locs[nb.mgr_locs.indexer]
        nb.mgr_locs = nblocs
        # Assertions are disabled for performance, but should hold:
        #  assert len(nblocs) == nb.shape[0], (len(nblocs), nb.shape)
        #  assert all(x in locs.as_array for x in nb.mgr_locs.as_array)


def _get_same_shape_values(
    lblk: Block, rblk: Block, left_ea: bool, right_ea: bool
) -> tuple[ArrayLike, ArrayLike]:
    """
    Slice lblk.values to align with rblk.  Squeeze if we have EAs.
    """
    lvals = lblk.values
    rvals = rblk.values

    # Require that the indexing into lvals be slice-like
    assert rblk.mgr_locs.is_slice_like, rblk.mgr_locs

    # TODO(EA2D): with 2D EAs only this first clause would be needed
    if not (left_ea or right_ea):
        # error: No overload variant of "__getitem__" of "ExtensionArray" matches
        # argument type "Tuple[Union[ndarray, slice], slice]"
        lvals = lvals[rblk.mgr_locs.indexer, :]  # type: ignore[call-overload]
        assert lvals.shape == rvals.shape, (lvals.shape, rvals.shape)
    elif left_ea and right_ea:
        assert lvals.shape == rvals.shape, (lvals.shape, rvals.shape)
    elif right_ea:
        # lvals are 2D, rvals are 1D

        # error: No overload variant of "__getitem__" of "ExtensionArray" matches
        # argument type "Tuple[Union[ndarray, slice], slice]"
        lvals = lvals[rblk.mgr_locs.indexer, :]  # type: ignore[call-overload]
        assert lvals.shape[0] == 1, lvals.shape
        lvals = lvals[0, :]
    else:
        # lvals are 1D, rvals are 2D
        assert rvals.shape[0] == 1, rvals.shape
        # error: No overload variant of "__getitem__" of "ExtensionArray" matches
        # argument type "Tuple[int, slice]"
        rvals = rvals[0, :]  # type: ignore[call-overload]

    return lvals, rvals


def blockwise_all(left: BlockManager, right: BlockManager, op) -> bool:
    """
    Blockwise `all` reduction.
    """
    for info in _iter_block_pairs(left, right):
        res = op(info.lvals, info.rvals)
        if not res:
            return False
    return True
<|endoftext|>"""
Base class for the internal managers. Both BlockManager and ArrayManager
inherit from this class.
"""
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    cast,
    final,
)

import numpy as np

from pandas._config import (
    using_copy_on_write,
    warn_copy_on_write,
)

from pandas._libs import (
    algos as libalgos,
    lib,
)
from pandas.errors import AbstractMethodError
from pandas.util._validators import validate_bool_kwarg

from pandas.core.dtypes.cast import (
    find_common_type,
    np_can_hold_element,
)
from pandas.core.dtypes.dtypes import (
    ExtensionDtype,
    SparseDtype,
)

from pandas.core.base import PandasObject
from pandas.core.construction import extract_array
from pandas.core.indexes.api import (
    Index,
    default_index,
)

if TYPE_CHECKING:
    from pandas._typing import (
        ArrayLike,
        AxisInt,
        DtypeObj,
        Self,
        Shape,
    )


class _AlreadyWarned:
    def __init__(self):
        # This class is used on the manager level to the block level to
        # ensure that we warn only once. The block method can update the
        # warned_already option without returning a value to keep the
        # interface consistent. This is only a temporary solution for
        # CoW warnings.
        self.warned_already = False


class DataManager(PandasObject):
    # TODO share more methods/attributes

    axes: list[Index]

    @property
    def items(self) -> Index:
        raise AbstractMethodError(self)

    @final
    def __len__(self) -> int:
        return len(self.items)

    @property
    def ndim(self) -> int:
        return len(self.axes)

    @property
    def shape(self) -> Shape:
        return tuple(len(ax) for ax in self.axes)

    @final
    def _validate_set_axis(self, axis: AxisInt, new_labels: Index) -> None:
        # Caller is responsible for ensuring we have an Index object.
        old_len = len(self.axes[axis])
        new_len = len(new_labels)

        if axis == 1 and len(self.items) == 0:
            # If we are setting the index on a DataFrame with no columns,
            #  it is OK to change the length.
            pass

        elif new_len != old_len:
            raise ValueError(
                f"Length mismatch: Expected axis has {old_len} elements, new "
                f"values have {new_len} elements"
            )

    def reindex_indexer(
        self,
        new_axis,
        indexer,
        axis: AxisInt,
        fill_value=None,
        allow_dups: bool = False,
        copy: bool = True,
        only_slice: bool = False,
    ) -> Self:
        raise AbstractMethodError(self)

    @final
    def reindex_axis(
        self,
        new_index: Index,
        axis: AxisInt,
        fill_value=None,
        only_slice: bool = False,
    ) -> Self:
        """
        Conform data manager to new index.
        """
        new_index, indexer = self.axes[axis].reindex(new_index)

        return self.reindex_indexer(
            new_index,
            indexer,
            axis=axis,
            fill_value=fill_value,
            copy=False,
            only_slice=only_slice,
        )

    def _equal_values(self, other: Self) -> bool:
        """
        To be implemented by the subclasses. Only check the column values
        assuming shape and indexes have already been checked.
        """
        raise AbstractMethodError(self)

    @final
    def equals(self, other: object) -> bool:
        """
        Implementation for DataFrame.equals
        """
        if not isinstance(other, type(self)):
            return False

        self_axes, other_axes = self.axes, other.axes
        if len(self_axes) != len(other_axes):
            return False
        if not all(ax1.equals(ax2) for ax1, ax2 in zip(self_axes, other_axes)):
            return False

        return self._equal_values(other)

    def apply(
        self,
        f,
        align_keys: list[str] | None = None,
        **kwargs,
    ) -> Self:
        raise AbstractMethodError(self)

    def apply_with_block(
        self,
        f,
        align_keys: list[str] | None = None,
        **kwargs,
    ) -> Self:
        raise AbstractMethodError(self)

    @final
    def isna(self, func) -> Self:
        return self.apply("apply", func=func)

    @final
    def fillna(self, value, limit: int | None, inplace: bool, downcast) -> Self:
        if limit is not None:
            # Do this validation even if we go through one of the no-op paths
            limit = libalgos.validate_limit(None, limit=limit)

        return self.apply_with_block(
            "fillna",
            value=value,
            limit=limit,
            inplace=inplace,
            downcast=downcast,
            using_cow=using_copy_on_write(),
            already_warned=_AlreadyWarned(),
        )

    @final
    def where(self, other, cond, align: bool) -> Self:
        if align:
            align_keys = ["other", "cond"]
        else:
            align_keys = ["cond"]
            other = extract_array(other, extract_numpy=True)

        return self.apply_with_block(
            "where",
            align_keys=align_keys,
            other=other,
            cond=cond,
            using_cow=using_copy_on_write(),
        )

    @final
    def putmask(self, mask, new, align: bool = True, warn: bool = True) -> Self:
        if align:
            align_keys = ["new", "mask"]
        else:
            align_keys = ["mask"]
            new = extract_array(new, extract_numpy=True)

        already_warned = None
        if warn_copy_on_write():
            already_warned = _AlreadyWarned()
            if not warn:
                already_warned.warned_already = True

        return self.apply_with_block(
            "putmask",
            align_keys=align_keys,
            mask=mask,
            new=new,
            using_cow=using_copy_on_write(),
            already_warned=already_warned,
        )

    @final
    def round(self, decimals: int, using_cow: bool = False) -> Self:
        return self.apply_with_block(
            "round",
            decimals=decimals,
            using_cow=using_cow,
        )

    @final
    def replace(self, to_replace, value, inplace: bool) -> Self:
        inplace = validate_bool_kwarg(inplace, "inplace")
        # NDFrame.replace ensures the not-is_list_likes here
        assert not lib.is_list_like(to_replace)
        assert not lib.is_list_like(value)
        return self.apply_with_block(
            "replace",
            to_replace=to_replace,
            value=value,
            inplace=inplace,
            using_cow=using_copy_on_write(),
            already_warned=_AlreadyWarned(),
        )

    @final
    def replace_regex(self, **kwargs) -> Self:
        return self.apply_with_block(
            "_replace_regex",
            **kwargs,
            using_cow=using_copy_on_write(),
            already_warned=_AlreadyWarned(),
        )

    @final
    def replace_list(
        self,
        src_list: list[Any],
        dest_list: list[Any],
        inplace: bool = False,
        regex: bool = False,
    ) -> Self:
        """do a list replace"""
        inplace = validate_bool_kwarg(inplace, "inplace")

        bm = self.apply_with_block(
            "replace_list",
            src_list=src_list,
            dest_list=dest_list,
            inplace=inplace,
            regex=regex,
            using_cow=using_copy_on_write(),
            already_warned=_AlreadyWarned(),
        )
        bm._consolidate_inplace()
        return bm

    def interpolate(self, inplace: bool, **kwargs) -> Self:
        return self.apply_with_block(
            "interpolate",
            inplace=inplace,
            **kwargs,
            using_cow=using_copy_on_write(),
            already_warned=_AlreadyWarned(),
        )

    def pad_or_backfill(self, inplace: bool, **kwargs) -> Self:
        return self.apply_with_block(
            "pad_or_backfill",
            inplace=inplace,
            **kwargs,
            using_cow=using_copy_on_write(),
            already_warned=_AlreadyWarned(),
        )

    def shift(self, periods: int, fill_value) -> Self:
        if fill_value is lib.no_default:
            fill_value = None

        return self.apply_with_block("shift", periods=periods, fill_value=fill_value)

    # --------------------------------------------------------------------
    # Consolidation: No-ops for all but BlockManager

    def is_consolidated(self) -> bool:
        return True

    def consolidate(self) -> Self:
        return self

    def _consolidate_inplace(self) -> None:
        return


class SingleDataManager(DataManager):
    @property
    def ndim(self) -> Literal[1]:
        return 1

    @final
    @property
    def array(self) -> ArrayLike:
        """
        Quick access to the backing array of the Block or SingleArrayManager.
        """
        # error: "SingleDataManager" has no attribute "arrays"; maybe "array"
        return self.arrays[0]  # type: ignore[attr-defined]

    def setitem_inplace(self, indexer, value, warn: bool = True) -> None:
        """
        Set values with indexer.

        For Single[Block/Array]Manager, this backs s[indexer] = value

        This is an inplace version of `setitem()`, mutating the manager/values
        in place, not returning a new Manager (and Block), and thus never changing
        the dtype.
        """
        arr = self.array

        # EAs will do this validation in their own __setitem__ methods.
        if isinstance(arr, np.ndarray):
            # Note: checking for ndarray instead of np.dtype means we exclude
            #  dt64/td64, which do their own validation.
            value = np_can_hold_element(arr.dtype, value)

        if isinstance(value, np.ndarray) and value.ndim == 1 and len(value) == 1:
            # NumPy 1.25 deprecation: https://github.com/numpy/numpy/pull/10615
            value = value[0, ...]

        arr[indexer] = value

    def grouped_reduce(self, func):
        arr = self.array
        res = func(arr)
        index = default_index(len(res))

        mgr = type(self).from_array(res, index)
        return mgr

    @classmethod
    def from_array(cls, arr: ArrayLike, index: Index):
        raise AbstractMethodError(cls)


def interleaved_dtype(dtypes: list[DtypeObj]) -> DtypeObj | None:
    """
    Find the common dtype for `blocks`.

    Parameters
    ----------
    blocks : List[DtypeObj]

    Returns
    -------
    dtype : np.dtype, ExtensionDtype, or None
        None is returned when `blocks` is empty.
    """
    if not len(dtypes):
        return None

    return find_common_type(dtypes)


def ensure_np_dtype(dtype: DtypeObj) -> np.dtype:
    # TODO: https://github.com/pandas-dev/pandas/issues/22791
    # Give EAs some input on what happens here. Sparse needs this.
    if isinstance(dtype, SparseDtype):
        dtype = dtype.subtype
        dtype = cast(np.dtype, dtype)
    elif isinstance(dtype, ExtensionDtype):
        dtype = np.dtype("object")
    elif dtype == np.dtype(str):
        dtype = np.dtype("object")
    return dtype
<|endoftext|>""" manage PyTables query interface via Expressions """
from __future__ import annotations

import ast
from decimal import (
    Decimal,
    InvalidOperation,
)
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
)

import numpy as np

from pandas._libs.tslibs import (
    Timedelta,
    Timestamp,
)
from pandas.errors import UndefinedVariableError

from pandas.core.dtypes.common import is_list_like

import pandas.core.common as com
from pandas.core.computation import (
    expr,
    ops,
    scope as _scope,
)
from pandas.core.computation.common import ensure_decoded
from pandas.core.computation.expr import BaseExprVisitor
from pandas.core.computation.ops import is_term
from pandas.core.construction import extract_array
from pandas.core.indexes.base import Index

from pandas.io.formats.printing import (
    pprint_thing,
    pprint_thing_encoded,
)

if TYPE_CHECKING:
    from pandas._typing import (
        Self,
        npt,
    )


class PyTablesScope(_scope.Scope):
    __slots__ = ("queryables",)

    queryables: dict[str, Any]

    def __init__(
        self,
        level: int,
        global_dict=None,
        local_dict=None,
        queryables: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(level + 1, global_dict=global_dict, local_dict=local_dict)
        self.queryables = queryables or {}


class Term(ops.Term):
    env: PyTablesScope

    def __new__(cls, name, env, side=None, encoding=None):
        if isinstance(name, str):
            klass = cls
        else:
            klass = Constant
        return object.__new__(klass)

    def __init__(self, name, env: PyTablesScope, side=None, encoding=None) -> None:
        super().__init__(name, env, side=side, encoding=encoding)

    def _resolve_name(self):
        # must be a queryables
        if self.side == "left":
            # Note: The behavior of __new__ ensures that self.name is a str here
            if self.name not in self.env.queryables:
                raise NameError(f"name {repr(self.name)} is not defined")
            return self.name

        # resolve the rhs (and allow it to be None)
        try:
            return self.env.resolve(self.name, is_local=False)
        except UndefinedVariableError:
            return self.name

    # read-only property overwriting read/write property
    @property  # type: ignore[misc]
    def value(self):
        return self._value


class Constant(Term):
    def __init__(self, name, env: PyTablesScope, side=None, encoding=None) -> None:
        assert isinstance(env, PyTablesScope), type(env)
        super().__init__(name, env, side=side, encoding=encoding)

    def _resolve_name(self):
        return self._name


class BinOp(ops.BinOp):
    _max_selectors = 31

    op: str
    queryables: dict[str, Any]
    condition: str | None

    def __init__(self, op: str, lhs, rhs, queryables: dict[str, Any], encoding) -> None:
        super().__init__(op, lhs, rhs)
        self.queryables = queryables
        self.encoding = encoding
        self.condition = None

    def _disallow_scalar_only_bool_ops(self) -> None:
        pass

    def prune(self, klass):
        def pr(left, right):
            """create and return a new specialized BinOp from myself"""
            if left is None:
                return right
            elif right is None:
                return left

            k = klass
            if isinstance(left, ConditionBinOp):
                if isinstance(right, ConditionBinOp):
                    k = JointConditionBinOp
                elif isinstance(left, k):
                    return left
                elif isinstance(right, k):
                    return right

            elif isinstance(left, FilterBinOp):
                if isinstance(right, FilterBinOp):
                    k = JointFilterBinOp
                elif isinstance(left, k):
                    return left
                elif isinstance(right, k):
                    return right

            return k(
                self.op, left, right, queryables=self.queryables, encoding=self.encoding
            ).evaluate()

        left, right = self.lhs, self.rhs

        if is_term(left) and is_term(right):
            res = pr(left.value, right.value)
        elif not is_term(left) and is_term(right):
            res = pr(left.prune(klass), right.value)
        elif is_term(left) and not is_term(right):
            res = pr(left.value, right.prune(klass))
        elif not (is_term(left) or is_term(right)):
            res = pr(left.prune(klass), right.prune(klass))

        return res

    def conform(self, rhs):
        """inplace conform rhs"""
        if not is_list_like(rhs):
            rhs = [rhs]
        if isinstance(rhs, np.ndarray):
            rhs = rhs.ravel()
        return rhs

    @property
    def is_valid(self) -> bool:
        """return True if this is a valid field"""
        return self.lhs in self.queryables

    @property
    def is_in_table(self) -> bool:
        """
        return True if this is a valid column name for generation (e.g. an
        actual column in the table)
        """
        return self.queryables.get(self.lhs) is not None

    @property
    def kind(self):
        """the kind of my field"""
        return getattr(self.queryables.get(self.lhs), "kind", None)

    @property
    def meta(self):
        """the meta of my field"""
        return getattr(self.queryables.get(self.lhs), "meta", None)

    @property
    def metadata(self):
        """the metadata of my field"""
        return getattr(self.queryables.get(self.lhs), "metadata", None)

    def generate(self, v) -> str:
        """create and return the op string for this TermValue"""
        val = v.tostring(self.encoding)
        return f"({self.lhs} {self.op} {val})"

    def convert_value(self, v) -> TermValue:
        """
        convert the expression that is in the term to something that is
        accepted by pytables
        """

        def stringify(value):
            if self.encoding is not None:
                return pprint_thing_encoded(value, encoding=self.encoding)
            return pprint_thing(value)

        kind = ensure_decoded(self.kind)
        meta = ensure_decoded(self.meta)
        if kind == "datetime" or (kind and kind.startswith("datetime64")):
            if isinstance(v, (int, float)):
                v = stringify(v)
            v = ensure_decoded(v)
            v = Timestamp(v).as_unit("ns")
            if v.tz is not None:
                v = v.tz_convert("UTC")
            return TermValue(v, v._value, kind)
        elif kind in ("timedelta64", "timedelta"):
            if isinstance(v, str):
                v = Timedelta(v)
            else:
                v = Timedelta(v, unit="s")
            v = v.as_unit("ns")._value
            return TermValue(int(v), v, kind)
        elif meta == "category":
            metadata = extract_array(self.metadata, extract_numpy=True)
            result: npt.NDArray[np.intp] | np.intp | int
            if v not in metadata:
                result = -1
            else:
                result = metadata.searchsorted(v, side="left")
            return TermValue(result, result, "integer")
        elif kind == "integer":
            try:
                v_dec = Decimal(v)
            except InvalidOperation:
                # GH 54186
                # convert v to float to raise float's ValueError
                float(v)
            else:
                v = int(v_dec.to_integral_exact(rounding="ROUND_HALF_EVEN"))
            return TermValue(v, v, kind)
        elif kind == "float":
            v = float(v)
            return TermValue(v, v, kind)
        elif kind == "bool":
            if isinstance(v, str):
                v = v.strip().lower() not in [
                    "false",
                    "f",
                    "no",
                    "n",
                    "none",
                    "0",
                    "[]",
                    "{}",
                    "",
                ]
            else:
                v = bool(v)
            return TermValue(v, v, kind)
        elif isinstance(v, str):
            # string quoting
            return TermValue(v, stringify(v), "string")
        else:
            raise TypeError(f"Cannot compare {v} of type {type(v)} to {kind} column")

    def convert_values(self) -> None:
        pass


class FilterBinOp(BinOp):
    filter: tuple[Any, Any, Index] | None = None

    def __repr__(self) -> str:
        if self.filter is None:
            return "Filter: Not Initialized"
        return pprint_thing(f"[Filter : [{self.filter[0]}] -> [{self.filter[1]}]")

    def invert(self) -> Self:
        """invert the filter"""
        if self.filter is not None:
            self.filter = (
                self.filter[0],
                self.generate_filter_op(invert=True),
                self.filter[2],
            )
        return self

    def format(self):
        """return the actual filter format"""
        return [self.filter]

    # error: Signature of "evaluate" incompatible with supertype "BinOp"
    def evaluate(self) -> Self | None:  # type: ignore[override]
        if not self.is_valid:
            raise ValueError(f"query term is not valid [{self}]")

        rhs = self.conform(self.rhs)
        values = list(rhs)

        if self.is_in_table:
            # if too many values to create the expression, use a filter instead
            if self.op in ["==", "!="] and len(values) > self._max_selectors:
                filter_op = self.generate_filter_op()
                self.filter = (self.lhs, filter_op, Index(values))

                return self
            return None

        # equality conditions
        if self.op in ["==", "!="]:
            filter_op = self.generate_filter_op()
            self.filter = (self.lhs, filter_op, Index(values))

        else:
            raise TypeError(
                f"passing a filterable condition to a non-table indexer [{self}]"
            )

        return self

    def generate_filter_op(self, invert: bool = False):
        if (self.op == "!=" and not invert) or (self.op == "==" and invert):
            return lambda axis, vals: ~axis.isin(vals)
        else:
            return lambda axis, vals: axis.isin(vals)


class JointFilterBinOp(FilterBinOp):
    def format(self):
        raise NotImplementedError("unable to collapse Joint Filters")

    # error: Signature of "evaluate" incompatible with supertype "BinOp"
    def evaluate(self) -> Self:  # type: ignore[override]
        return self


class ConditionBinOp(BinOp):
    def __repr__(self) -> str:
        return pprint_thing(f"[Condition : [{self.condition}]]")

    def invert(self):
        """invert the condition"""
        # if self.condition is not None:
        #    self.condition = "~(%s)" % self.condition
        # return self
        raise NotImplementedError(
            "cannot use an invert condition when passing to numexpr"
        )

    def format(self):
        """return the actual ne format"""
        return self.condition

    # error: Signature of "evaluate" incompatible with supertype "BinOp"
    def evaluate(self) -> Self | None:  # type: ignore[override]
        if not self.is_valid:
            raise ValueError(f"query term is not valid [{self}]")

        # convert values if we are in the table
        if not self.is_in_table:
            return None

        rhs = self.conform(self.rhs)
        values = [self.convert_value(v) for v in rhs]

        # equality conditions
        if self.op in ["==", "!="]:
            # too many values to create the expression?
            if len(values) <= self._max_selectors:
                vs = [self.generate(v) for v in values]
                self.condition = f"({' | '.join(vs)})"

            # use a filter after reading
            else:
                return None
        else:
            self.condition = self.generate(values[0])

        return self


class JointConditionBinOp(ConditionBinOp):
    # error: Signature of "evaluate" incompatible with supertype "BinOp"
    def evaluate(self) -> Self:  # type: ignore[override]
        self.condition = f"({self.lhs.condition} {self.op} {self.rhs.condition})"
        return self


class UnaryOp(ops.UnaryOp):
    def prune(self, klass):
        if self.op != "~":
            raise NotImplementedError("UnaryOp only support invert type ops")

        operand = self.operand
        operand = operand.prune(klass)

        if operand is not None and (
            issubclass(klass, ConditionBinOp)
            and operand.condition is not None
            or not issubclass(klass, ConditionBinOp)
            and issubclass(klass, FilterBinOp)
            and operand.filter is not None
        ):
            return operand.invert()
        return None


class PyTablesExprVisitor(BaseExprVisitor):
    const_type: ClassVar[type[ops.Term]] = Constant
    term_type: ClassVar[type[Term]] = Term

    def __init__(self, env, engine, parser, **kwargs) -> None:
        super().__init__(env, engine, parser)
        for bin_op in self.binary_ops:
            bin_node = self.binary_op_nodes_map[bin_op]
            setattr(
                self,
                f"visit_{bin_node}",
                lambda node, bin_op=bin_op: partial(BinOp, bin_op, **kwargs),
            )

    def visit_UnaryOp(self, node, **kwargs) -> ops.Term | UnaryOp | None:
        if isinstance(node.op, (ast.Not, ast.Invert)):
            return UnaryOp("~", self.visit(node.operand))
        elif isinstance(node.op, ast.USub):
            return self.const_type(-self.visit(node.operand).value, self.env)
        elif isinstance(node.op, ast.UAdd):
            raise NotImplementedError("Unary addition not supported")
        # TODO: return None might never be reached
        return None

    def visit_Index(self, node, **kwargs):
        return self.visit(node.value).value

    def visit_Assign(self, node, **kwargs):
        cmpr = ast.Compare(
            ops=[ast.Eq()], left=node.targets[0], comparators=[node.value]
        )
        return self.visit(cmpr)

    def visit_Subscript(self, node, **kwargs) -> ops.Term:
        # only allow simple subscripts

        value = self.visit(node.value)
        slobj = self.visit(node.slice)
        try:
            value = value.value
        except AttributeError:
            pass

        if isinstance(slobj, Term):
            # In py39 np.ndarray lookups with Term containing int raise
            slobj = slobj.value

        try:
            return self.const_type(value[slobj], self.env)
        except TypeError as err:
            raise ValueError(
                f"cannot subscript {repr(value)} with {repr(slobj)}"
            ) from err

    def visit_Attribute(self, node, **kwargs):
        attr = node.attr
        value = node.value

        ctx = type(node.ctx)
        if ctx == ast.Load:
            # resolve the value
            resolved = self.visit(value)

            # try to get the value to see if we are another expression
            try:
                resolved = resolved.value
            except AttributeError:
                pass

            try:
                return self.term_type(getattr(resolved, attr), self.env)
            except AttributeError:
                # something like datetime.datetime where scope is overridden
                if isinstance(value, ast.Name) and value.id == attr:
                    return resolved

        raise ValueError(f"Invalid Attribute context {ctx.__name__}")

    def translate_In(self, op):
        return ast.Eq() if isinstance(op, ast.In) else op

    def _rewrite_membership_op(self, node, left, right):
        return self.visit(node.op), node.op, left, right


def _validate_where(w):
    """
    Validate that the where statement is of the right type.

    The type may either be String, Expr, or list-like of Exprs.

    Parameters
    ----------
    w : String term expression, Expr, or list-like of Exprs.

    Returns
    -------
    where : The original where clause if the check was successful.

    Raises
    ------
    TypeError : An invalid data type was passed in for w (e.g. dict).
    """
    if not (isinstance(w, (PyTablesExpr, str)) or is_list_like(w)):
        raise TypeError(
            "where must be passed as a string, PyTablesExpr, "
            "or list-like of PyTablesExpr"
        )

    return w


class PyTablesExpr(expr.Expr):
    """
    Hold a pytables-like expression, comprised of possibly multiple 'terms'.

    Parameters
    ----------
    where : string term expression, PyTablesExpr, or list-like of PyTablesExprs
    queryables : a "kinds" map (dict of column name -> kind), or None if column
        is non-indexable
    encoding : an encoding that will encode the query terms

    Returns
    -------
    a PyTablesExpr object

    Examples
    --------
    'index>=date'
    "columns=['A', 'D']"
    'columns=A'
    'columns==A'
    "~(columns=['A','B'])"
    'index>df.index[3] & string="bar"'
    '(index>df.index[3] & index<=df.index[6]) | string="bar"'
    "ts>=Timestamp('2012-02-01')"
    "major_axis>=20130101"
    """

    _visitor: PyTablesExprVisitor | None
    env: PyTablesScope
    expr: str

    def __init__(
        self,
        where,
        queryables: dict[str, Any] | None = None,
        encoding=None,
        scope_level: int = 0,
    ) -> None:
        where = _validate_where(where)

        self.encoding = encoding
        self.condition = None
        self.filter = None
        self.terms = None
        self._visitor = None

        # capture the environment if needed
        local_dict: _scope.DeepChainMap[Any, Any] | None = None

        if isinstance(where, PyTablesExpr):
            local_dict = where.env.scope
            _where = where.expr

        elif is_list_like(where):
            where = list(where)
            for idx, w in enumerate(where):
                if isinstance(w, PyTablesExpr):
                    local_dict = w.env.scope
                else:
                    where[idx] = _validate_where(w)
            _where = " & ".join([f"({w})" for w in com.flatten(where)])
        else:
            # _validate_where ensures we otherwise have a string
            _where = where

        self.expr = _where
        self.env = PyTablesScope(scope_level + 1, local_dict=local_dict)

        if queryables is not None and isinstance(self.expr, str):
            self.env.queryables.update(queryables)
            self._visitor = PyTablesExprVisitor(
                self.env,
                queryables=queryables,
                parser="pytables",
                engine="pytables",
                encoding=encoding,
            )
            self.terms = self.parse()

    def __repr__(self) -> str:
        if self.terms is not None:
            return pprint_thing(self.terms)
        return pprint_thing(self.expr)

    def evaluate(self):
        """create and return the numexpr condition and filter"""
        try:
            self.condition = self.terms.prune(ConditionBinOp)
        except AttributeError as err:
            raise ValueError(
                f"cannot process expression [{self.expr}], [{self}] "
                "is not a valid condition"
            ) from err
        try:
            self.filter = self.terms.prune(FilterBinOp)
        except AttributeError as err:
            raise ValueError(
                f"cannot process expression [{self.expr}], [{self}] "
                "is not a valid filter"
            ) from err

        return self.condition, self.filter


class TermValue:
    """hold a term value the we use to construct a condition/filter"""

    def __init__(self, value, converted, kind: str) -> None:
        assert isinstance(kind, str), kind
        self.value = value
        self.converted = converted
        self.kind = kind

    def tostring(self, encoding) -> str:
        """quote the string if not encoded else encode and return"""
        if self.kind == "string":
            if encoding is not None:
                return str(self.converted)
            return f'"{self.converted}"'
        elif self.kind == "float":
            # python 2 str(float) is not always
            # round-trippable so use repr()
            return repr(self.converted)
        return str(self.converted)


def maybe_expression(s) -> bool:
    """loose checking if s is a pytables-acceptable expression"""
    if not isinstance(s, str):
        return False
    operations = PyTablesExprVisitor.binary_ops + PyTablesExprVisitor.unary_ops + ("=",)

    # make sure we have an op at least
    return any(op in s for op in operations)
<|endoftext|>"""
Core eval alignment algorithms.
"""
from __future__ import annotations

from functools import (
    partial,
    wraps,
)
from typing import (
    TYPE_CHECKING,
    Callable,
)
import warnings

import numpy as np

from pandas.errors import PerformanceWarning
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCSeries,
)

from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.computation.common import result_type_many

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pandas._typing import F

    from pandas.core.generic import NDFrame
    from pandas.core.indexes.api import Index


def _align_core_single_unary_op(
    term,
) -> tuple[partial | type[NDFrame], dict[str, Index] | None]:
    typ: partial | type[NDFrame]
    axes: dict[str, Index] | None = None

    if isinstance(term.value, np.ndarray):
        typ = partial(np.asanyarray, dtype=term.value.dtype)
    else:
        typ = type(term.value)
        if hasattr(term.value, "axes"):
            axes = _zip_axes_from_type(typ, term.value.axes)

    return typ, axes


def _zip_axes_from_type(
    typ: type[NDFrame], new_axes: Sequence[Index]
) -> dict[str, Index]:
    return {name: new_axes[i] for i, name in enumerate(typ._AXIS_ORDERS)}


def _any_pandas_objects(terms) -> bool:
    """
    Check a sequence of terms for instances of PandasObject.
    """
    return any(isinstance(term.value, PandasObject) for term in terms)


def _filter_special_cases(f) -> Callable[[F], F]:
    @wraps(f)
    def wrapper(terms):
        # single unary operand
        if len(terms) == 1:
            return _align_core_single_unary_op(terms[0])

        term_values = (term.value for term in terms)

        # we don't have any pandas objects
        if not _any_pandas_objects(terms):
            return result_type_many(*term_values), None

        return f(terms)

    return wrapper


@_filter_special_cases
def _align_core(terms):
    term_index = [i for i, term in enumerate(terms) if hasattr(term.value, "axes")]
    term_dims = [terms[i].value.ndim for i in term_index]

    from pandas import Series

    ndims = Series(dict(zip(term_index, term_dims)))

    # initial axes are the axes of the largest-axis'd term
    biggest = terms[ndims.idxmax()].value
    typ = biggest._constructor
    axes = biggest.axes
    naxes = len(axes)
    gt_than_one_axis = naxes > 1

    for value in (terms[i].value for i in term_index):
        is_series = isinstance(value, ABCSeries)
        is_series_and_gt_one_axis = is_series and gt_than_one_axis

        for axis, items in enumerate(value.axes):
            if is_series_and_gt_one_axis:
                ax, itm = naxes - 1, value.index
            else:
                ax, itm = axis, items

            if not axes[ax].is_(itm):
                axes[ax] = axes[ax].union(itm)

    for i, ndim in ndims.items():
        for axis, items in zip(range(ndim), axes):
            ti = terms[i].value

            if hasattr(ti, "reindex"):
                transpose = isinstance(ti, ABCSeries) and naxes > 1
                reindexer = axes[naxes - 1] if transpose else items

                term_axis_size = len(ti.axes[axis])
                reindexer_size = len(reindexer)

                ordm = np.log10(max(1, abs(reindexer_size - term_axis_size)))
                if ordm >= 1 and reindexer_size >= 10000:
                    w = (
                        f"Alignment difference on axis {axis} is larger "
                        f"than an order of magnitude on term {repr(terms[i].name)}, "
                        f"by more than {ordm:.4g}; performance may suffer."
                    )
                    warnings.warn(
                        w, category=PerformanceWarning, stacklevel=find_stack_level()
                    )

                obj = ti.reindex(reindexer, axis=axis, copy=False)
                terms[i].update(obj)

        terms[i].update(terms[i].value.values)

    return typ, _zip_axes_from_type(typ, axes)


def align_terms(terms):
    """
    Align a set of terms.
    """
    try:
        # flatten the parse tree (a nested list, really)
        terms = list(com.flatten(terms))
    except TypeError:
        # can't iterate so it must just be a constant or single variable
        if isinstance(terms.value, (ABCSeries, ABCDataFrame)):
            typ = type(terms.value)
            return typ, _zip_axes_from_type(typ, terms.value.axes)
        return np.result_type(terms.type), None

    # if all resolved variables are numeric scalars
    if all(term.is_scalar for term in terms):
        return result_type_many(*(term.value for term in terms)).type, None

    # perform the main alignment
    typ, axes = _align_core(terms)
    return typ, axes


def reconstruct_object(typ, obj, axes, dtype):
    """
    Reconstruct an object given its type, raw value, and possibly empty
    (None) axes.

    Parameters
    ----------
    typ : object
        A type
    obj : object
        The value to use in the type constructor
    axes : dict
        The axes to use to construct the resulting pandas object

    Returns
    -------
    ret : typ
        An object of type ``typ`` with the value `obj` and possible axes
        `axes`.
    """
    try:
        typ = typ.type
    except AttributeError:
        pass

    res_t = np.result_type(obj.dtype, dtype)

    if not isinstance(typ, partial) and issubclass(typ, PandasObject):
        return typ(obj, dtype=res_t, **axes)

    # special case for pathological things like ~True/~False
    if hasattr(res_t, "type") and typ == np.bool_ and res_t != np.bool_:
        ret_value = res_t.type(obj)
    else:
        ret_value = typ(obj).astype(res_t)
        # The condition is to distinguish 0-dim array (returned in case of
        # scalar) and 1 element array
        # e.g. np.array(0) and np.array([0])
        if (
            len(obj.shape) == 1
            and len(obj) == 1
            and not isinstance(ret_value, np.ndarray)
        ):
            ret_value = np.array([ret_value]).astype(res_t)

    return ret_value
<|endoftext|>"""
:func:`~pandas.eval` parsers.
"""
from __future__ import annotations

import ast
from functools import (
    partial,
    reduce,
)
from keyword import iskeyword
import tokenize
from typing import (
    Callable,
    ClassVar,
    TypeVar,
)

import numpy as np

from pandas.errors import UndefinedVariableError

from pandas.core.dtypes.common import is_string_dtype

import pandas.core.common as com
from pandas.core.computation.ops import (
    ARITH_OPS_SYMS,
    BOOL_OPS_SYMS,
    CMP_OPS_SYMS,
    LOCAL_TAG,
    MATHOPS,
    REDUCTIONS,
    UNARY_OPS_SYMS,
    BinOp,
    Constant,
    FuncNode,
    Op,
    Term,
    UnaryOp,
    is_term,
)
from pandas.core.computation.parsing import (
    clean_backtick_quoted_toks,
    tokenize_string,
)
from pandas.core.computation.scope import Scope

from pandas.io.formats import printing


def _rewrite_assign(tok: tuple[int, str]) -> tuple[int, str]:
    """
    Rewrite the assignment operator for PyTables expressions that use ``=``
    as a substitute for ``==``.

    Parameters
    ----------
    tok : tuple of int, str
        ints correspond to the all caps constants in the tokenize module

    Returns
    -------
    tuple of int, str
        Either the input or token or the replacement values
    """
    toknum, tokval = tok
    return toknum, "==" if tokval == "=" else tokval


def _replace_booleans(tok: tuple[int, str]) -> tuple[int, str]:
    """
    Replace ``&`` with ``and`` and ``|`` with ``or`` so that bitwise
    precedence is changed to boolean precedence.

    Parameters
    ----------
    tok : tuple of int, str
        ints correspond to the all caps constants in the tokenize module

    Returns
    -------
    tuple of int, str
        Either the input or token or the replacement values
    """
    toknum, tokval = tok
    if toknum == tokenize.OP:
        if tokval == "&":
            return tokenize.NAME, "and"
        elif tokval == "|":
            return tokenize.NAME, "or"
        return toknum, tokval
    return toknum, tokval


def _replace_locals(tok: tuple[int, str]) -> tuple[int, str]:
    """
    Replace local variables with a syntactically valid name.

    Parameters
    ----------
    tok : tuple of int, str
        ints correspond to the all caps constants in the tokenize module

    Returns
    -------
    tuple of int, str
        Either the input or token or the replacement values

    Notes
    -----
    This is somewhat of a hack in that we rewrite a string such as ``'@a'`` as
    ``'__pd_eval_local_a'`` by telling the tokenizer that ``__pd_eval_local_``
    is a ``tokenize.OP`` and to replace the ``'@'`` symbol with it.
    """
    toknum, tokval = tok
    if toknum == tokenize.OP and tokval == "@":
        return tokenize.OP, LOCAL_TAG
    return toknum, tokval


def _compose2(f, g):
    """
    Compose 2 callables.
    """
    return lambda *args, **kwargs: f(g(*args, **kwargs))


def _compose(*funcs):
    """
    Compose 2 or more callables.
    """
    assert len(funcs) > 1, "At least 2 callables must be passed to compose"
    return reduce(_compose2, funcs)


def _preparse(
    source: str,
    f=_compose(
        _replace_locals, _replace_booleans, _rewrite_assign, clean_backtick_quoted_toks
    ),
) -> str:
    """
    Compose a collection of tokenization functions.

    Parameters
    ----------
    source : str
        A Python source code string
    f : callable
        This takes a tuple of (toknum, tokval) as its argument and returns a
        tuple with the same structure but possibly different elements. Defaults
        to the composition of ``_rewrite_assign``, ``_replace_booleans``, and
        ``_replace_locals``.

    Returns
    -------
    str
        Valid Python source code

    Notes
    -----
    The `f` parameter can be any callable that takes *and* returns input of the
    form ``(toknum, tokval)``, where ``toknum`` is one of the constants from
    the ``tokenize`` module and ``tokval`` is a string.
    """
    assert callable(f), "f must be callable"
    return tokenize.untokenize(f(x) for x in tokenize_string(source))


def _is_type(t):
    """
    Factory for a type checking function of type ``t`` or tuple of types.
    """
    return lambda x: isinstance(x.value, t)


_is_list = _is_type(list)
_is_str = _is_type(str)


# partition all AST nodes
_all_nodes = frozenset(
    node
    for node in (getattr(ast, name) for name in dir(ast))
    if isinstance(node, type) and issubclass(node, ast.AST)
)


def _filter_nodes(superclass, all_nodes=_all_nodes):
    """
    Filter out AST nodes that are subclasses of ``superclass``.
    """
    node_names = (node.__name__ for node in all_nodes if issubclass(node, superclass))
    return frozenset(node_names)


_all_node_names = frozenset(x.__name__ for x in _all_nodes)
_mod_nodes = _filter_nodes(ast.mod)
_stmt_nodes = _filter_nodes(ast.stmt)
_expr_nodes = _filter_nodes(ast.expr)
_expr_context_nodes = _filter_nodes(ast.expr_context)
_boolop_nodes = _filter_nodes(ast.boolop)
_operator_nodes = _filter_nodes(ast.operator)
_unary_op_nodes = _filter_nodes(ast.unaryop)
_cmp_op_nodes = _filter_nodes(ast.cmpop)
_comprehension_nodes = _filter_nodes(ast.comprehension)
_handler_nodes = _filter_nodes(ast.excepthandler)
_arguments_nodes = _filter_nodes(ast.arguments)
_keyword_nodes = _filter_nodes(ast.keyword)
_alias_nodes = _filter_nodes(ast.alias)


# nodes that we don't support directly but are needed for parsing
_hacked_nodes = frozenset(["Assign", "Module", "Expr"])


_unsupported_expr_nodes = frozenset(
    [
        "Yield",
        "GeneratorExp",
        "IfExp",
        "DictComp",
        "SetComp",
        "Repr",
        "Lambda",
        "Set",
        "AST",
        "Is",
        "IsNot",
    ]
)

# these nodes are low priority or won't ever be supported (e.g., AST)
_unsupported_nodes = (
    _stmt_nodes
    | _mod_nodes
    | _handler_nodes
    | _arguments_nodes
    | _keyword_nodes
    | _alias_nodes
    | _expr_context_nodes
    | _unsupported_expr_nodes
) - _hacked_nodes

# we're adding a different assignment in some cases to be equality comparison
# and we don't want `stmt` and friends in their so get only the class whose
# names are capitalized
_base_supported_nodes = (_all_node_names - _unsupported_nodes) | _hacked_nodes
intersection = _unsupported_nodes & _base_supported_nodes
_msg = f"cannot both support and not support {intersection}"
assert not intersection, _msg


def _node_not_implemented(node_name: str) -> Callable[..., None]:
    """
    Return a function that raises a NotImplementedError with a passed node name.
    """

    def f(self, *args, **kwargs):
        raise NotImplementedError(f"'{node_name}' nodes are not implemented")

    return f


# should be bound by BaseExprVisitor but that creates a circular dependency:
# _T is used in disallow, but disallow is used to define BaseExprVisitor
# https://github.com/microsoft/pyright/issues/2315
_T = TypeVar("_T")


def disallow(nodes: set[str]) -> Callable[[type[_T]], type[_T]]:
    """
    Decorator to disallow certain nodes from parsing. Raises a
    NotImplementedError instead.

    Returns
    -------
    callable
    """

    def disallowed(cls: type[_T]) -> type[_T]:
        # error: "Type[_T]" has no attribute "unsupported_nodes"
        cls.unsupported_nodes = ()  # type: ignore[attr-defined]
        for node in nodes:
            new_method = _node_not_implemented(node)
            name = f"visit_{node}"
            # error: "Type[_T]" has no attribute "unsupported_nodes"
            cls.unsupported_nodes += (name,)  # type: ignore[attr-defined]
            setattr(cls, name, new_method)
        return cls

    return disallowed


def _op_maker(op_class, op_symbol):
    """
    Return a function to create an op class with its symbol already passed.

    Returns
    -------
    callable
    """

    def f(self, node, *args, **kwargs):
        """
        Return a partial function with an Op subclass with an operator already passed.

        Returns
        -------
        callable
        """
        return partial(op_class, op_symbol, *args, **kwargs)

    return f


_op_classes = {"binary": BinOp, "unary": UnaryOp}


def add_ops(op_classes):
    """
    Decorator to add default implementation of ops.
    """

    def f(cls):
        for op_attr_name, op_class in op_classes.items():
            ops = getattr(cls, f"{op_attr_name}_ops")
            ops_map = getattr(cls, f"{op_attr_name}_op_nodes_map")
            for op in ops:
                op_node = ops_map[op]
                if op_node is not None:
                    made_op = _op_maker(op_class, op)
                    setattr(cls, f"visit_{op_node}", made_op)
        return cls

    return f


@disallow(_unsupported_nodes)
@add_ops(_op_classes)
class BaseExprVisitor(ast.NodeVisitor):
    """
    Custom ast walker. Parsers of other engines should subclass this class
    if necessary.

    Parameters
    ----------
    env : Scope
    engine : str
    parser : str
    preparser : callable
    """

    const_type: ClassVar[type[Term]] = Constant
    term_type: ClassVar[type[Term]] = Term

    binary_ops = CMP_OPS_SYMS + BOOL_OPS_SYMS + ARITH_OPS_SYMS
    binary_op_nodes = (
        "Gt",
        "Lt",
        "GtE",
        "LtE",
        "Eq",
        "NotEq",
        "In",
        "NotIn",
        "BitAnd",
        "BitOr",
        "And",
        "Or",
        "Add",
        "Sub",
        "Mult",
        "Div",
        "Pow",
        "FloorDiv",
        "Mod",
    )
    binary_op_nodes_map = dict(zip(binary_ops, binary_op_nodes))

    unary_ops = UNARY_OPS_SYMS
    unary_op_nodes = "UAdd", "USub", "Invert", "Not"
    unary_op_nodes_map = dict(zip(unary_ops, unary_op_nodes))

    rewrite_map = {
        ast.Eq: ast.In,
        ast.NotEq: ast.NotIn,
        ast.In: ast.In,
        ast.NotIn: ast.NotIn,
    }

    unsupported_nodes: tuple[str, ...]

    def __init__(self, env, engine, parser, preparser=_preparse) -> None:
        self.env = env
        self.engine = engine
        self.parser = parser
        self.preparser = preparser
        self.assigner = None

    def visit(self, node, **kwargs):
        if isinstance(node, str):
            clean = self.preparser(node)
            try:
                node = ast.fix_missing_locations(ast.parse(clean))
            except SyntaxError as e:
                if any(iskeyword(x) for x in clean.split()):
                    e.msg = "Python keyword not valid identifier in numexpr query"
                raise e

        method = f"visit_{type(node).__name__}"
        visitor = getattr(self, method)
        return visitor(node, **kwargs)

    def visit_Module(self, node, **kwargs):
        if len(node.body) != 1:
            raise SyntaxError("only a single expression is allowed")
        expr = node.body[0]
        return self.visit(expr, **kwargs)

    def visit_Expr(self, node, **kwargs):
        return self.visit(node.value, **kwargs)

    def _rewrite_membership_op(self, node, left, right):
        # the kind of the operator (is actually an instance)
        op_instance = node.op
        op_type = type(op_instance)

        # must be two terms and the comparison operator must be ==/!=/in/not in
        if is_term(left) and is_term(right) and op_type in self.rewrite_map:
            left_list, right_list = map(_is_list, (left, right))
            left_str, right_str = map(_is_str, (left, right))

            # if there are any strings or lists in the expression
            if left_list or right_list or left_str or right_str:
                op_instance = self.rewrite_map[op_type]()

            # pop the string variable out of locals and replace it with a list
            # of one string, kind of a hack
            if right_str:
                name = self.env.add_tmp([right.value])
                right = self.term_type(name, self.env)

            if left_str:
                name = self.env.add_tmp([left.value])
                left = self.term_type(name, self.env)

        op = self.visit(op_instance)
        return op, op_instance, left, right

    def _maybe_transform_eq_ne(self, node, left=None, right=None):
        if left is None:
            left = self.visit(node.left, side="left")
        if right is None:
            right = self.visit(node.right, side="right")
        op, op_class, left, right = self._rewrite_membership_op(node, left, right)
        return op, op_class, left, right

    def _maybe_downcast_constants(self, left, right):
        f32 = np.dtype(np.float32)
        if (
            left.is_scalar
            and hasattr(left, "value")
            and not right.is_scalar
            and right.return_type == f32
        ):
            # right is a float32 array, left is a scalar
            name = self.env.add_tmp(np.float32(left.value))
            left = self.term_type(name, self.env)
        if (
            right.is_scalar
            and hasattr(right, "value")
            and not left.is_scalar
            and left.return_type == f32
        ):
            # left is a float32 array, right is a scalar
            name = self.env.add_tmp(np.float32(right.value))
            right = self.term_type(name, self.env)

        return left, right

    def _maybe_eval(self, binop, eval_in_python):
        # eval `in` and `not in` (for now) in "partial" python space
        # things that can be evaluated in "eval" space will be turned into
        # temporary variables. for example,
        # [1,2] in a + 2 * b
        # in that case a + 2 * b will be evaluated using numexpr, and the "in"
        # call will be evaluated using isin (in python space)
        return binop.evaluate(
            self.env, self.engine, self.parser, self.term_type, eval_in_python
        )

    def _maybe_evaluate_binop(
        self,
        op,
        op_class,
        lhs,
        rhs,
        eval_in_python=("in", "not in"),
        maybe_eval_in_python=("==", "!=", "<", ">", "<=", ">="),
    ):
        res = op(lhs, rhs)

        if res.has_invalid_return_type:
            raise TypeError(
                f"unsupported operand type(s) for {res.op}: "
                f"'{lhs.type}' and '{rhs.type}'"
            )

        if self.engine != "pytables" and (
            res.op in CMP_OPS_SYMS
            and getattr(lhs, "is_datetime", False)
            or getattr(rhs, "is_datetime", False)
        ):
            # all date ops must be done in python bc numexpr doesn't work
            # well with NaT
            return self._maybe_eval(res, self.binary_ops)

        if res.op in eval_in_python:
            # "in"/"not in" ops are always evaluated in python
            return self._maybe_eval(res, eval_in_python)
        elif self.engine != "pytables":
            if (
                getattr(lhs, "return_type", None) == object
                or is_string_dtype(getattr(lhs, "return_type", None))
                or getattr(rhs, "return_type", None) == object
                or is_string_dtype(getattr(rhs, "return_type", None))
            ):
                # evaluate "==" and "!=" in python if either of our operands
                # has an object or string return type
                return self._maybe_eval(res, eval_in_python + maybe_eval_in_python)
        return res

    def visit_BinOp(self, node, **kwargs):
        op, op_class, left, right = self._maybe_transform_eq_ne(node)
        left, right = self._maybe_downcast_constants(left, right)
        return self._maybe_evaluate_binop(op, op_class, left, right)

    def visit_UnaryOp(self, node, **kwargs):
        op = self.visit(node.op)
        operand = self.visit(node.operand)
        return op(operand)

    def visit_Name(self, node, **kwargs) -> Term:
        return self.term_type(node.id, self.env, **kwargs)

    # TODO(py314): deprecated since Python 3.8. Remove after Python 3.14 is min
    def visit_NameConstant(self, node, **kwargs) -> Term:
        return self.const_type(node.value, self.env)

    # TODO(py314): deprecated since Python 3.8. Remove after Python 3.14 is min
    def visit_Num(self, node, **kwargs) -> Term:
        return self.const_type(node.value, self.env)

    def visit_Constant(self, node, **kwargs) -> Term:
        return self.const_type(node.value, self.env)

    # TODO(py314): deprecated since Python 3.8. Remove after Python 3.14 is min
    def visit_Str(self, node, **kwargs) -> Term:
        name = self.env.add_tmp(node.s)
        return self.term_type(name, self.env)

    def visit_List(self, node, **kwargs) -> Term:
        name = self.env.add_tmp([self.visit(e)(self.env) for e in node.elts])
        return self.term_type(name, self.env)

    visit_Tuple = visit_List

    def visit_Index(self, node, **kwargs):
        """df.index[4]"""
        return self.visit(node.value)

    def visit_Subscript(self, node, **kwargs) -> Term:
        from pandas import eval as pd_eval

        value = self.visit(node.value)
        slobj = self.visit(node.slice)
        result = pd_eval(
            slobj, local_dict=self.env, engine=self.engine, parser=self.parser
        )
        try:
            # a Term instance
            v = value.value[result]
        except AttributeError:
            # an Op instance
            lhs = pd_eval(
                value, local_dict=self.env, engine=self.engine, parser=self.parser
            )
            v = lhs[result]
        name = self.env.add_tmp(v)
        return self.term_type(name, env=self.env)

    def visit_Slice(self, node, **kwargs) -> slice:
        """df.index[slice(4,6)]"""
        lower = node.lower
        if lower is not None:
            lower = self.visit(lower).value
        upper = node.upper
        if upper is not None:
            upper = self.visit(upper).value
        step = node.step
        if step is not None:
            step = self.visit(step).value

        return slice(lower, upper, step)

    def visit_Assign(self, node, **kwargs):
        """
        support a single assignment node, like

        c = a + b

        set the assigner at the top level, must be a Name node which
        might or might not exist in the resolvers

        """
        if len(node.targets) != 1:
            raise SyntaxError("can only assign a single expression")
        if not isinstance(node.targets[0], ast.Name):
            raise SyntaxError("left hand side of an assignment must be a single name")
        if self.env.target is None:
            raise ValueError("cannot assign without a target object")

        try:
            assigner = self.visit(node.targets[0], **kwargs)
        except UndefinedVariableError:
            assigner = node.targets[0].id

        self.assigner = getattr(assigner, "name", assigner)
        if self.assigner is None:
            raise SyntaxError(
                "left hand side of an assignment must be a single resolvable name"
            )

        return self.visit(node.value, **kwargs)

    def visit_Attribute(self, node, **kwargs):
        attr = node.attr
        value = node.value

        ctx = node.ctx
        if isinstance(ctx, ast.Load):
            # resolve the value
            resolved = self.visit(value).value
            try:
                v = getattr(resolved, attr)
                name = self.env.add_tmp(v)
                return self.term_type(name, self.env)
            except AttributeError:
                # something like datetime.datetime where scope is overridden
                if isinstance(value, ast.Name) and value.id == attr:
                    return resolved
                raise

        raise ValueError(f"Invalid Attribute context {type(ctx).__name__}")

    def visit_Call(self, node, side=None, **kwargs):
        if isinstance(node.func, ast.Attribute) and node.func.attr != "__call__":
            res = self.visit_Attribute(node.func)
        elif not isinstance(node.func, ast.Name):
            raise TypeError("Only named functions are supported")
        else:
            try:
                res = self.visit(node.func)
            except UndefinedVariableError:
                # Check if this is a supported function name
                try:
                    res = FuncNode(node.func.id)
                except ValueError:
                    # Raise original error
                    raise

        if res is None:
            # error: "expr" has no attribute "id"
            raise ValueError(
                f"Invalid function call {node.func.id}"  # type: ignore[attr-defined]
            )
        if hasattr(res, "value"):
            res = res.value

        if isinstance(res, FuncNode):
            new_args = [self.visit(arg) for arg in node.args]

            if node.keywords:
                raise TypeError(
                    f'Function "{res.name}" does not support keyword arguments'
                )

            return res(*new_args)

        else:
            new_args = [self.visit(arg)(self.env) for arg in node.args]

            for key in node.keywords:
                if not isinstance(key, ast.keyword):
                    # error: "expr" has no attribute "id"
                    raise ValueError(
                        "keyword error in function call "
                        f"'{node.func.id}'"  # type: ignore[attr-defined]
                    )

                if key.arg:
                    kwargs[key.arg] = self.visit(key.value)(self.env)

            name = self.env.add_tmp(res(*new_args, **kwargs))
            return self.term_type(name=name, env=self.env)

    def translate_In(self, op):
        return op

    def visit_Compare(self, node, **kwargs):
        ops = node.ops
        comps = node.comparators

        # base case: we have something like a CMP b
        if len(comps) == 1:
            op = self.translate_In(ops[0])
            binop = ast.BinOp(op=op, left=node.left, right=comps[0])
            return self.visit(binop)

        # recursive case: we have a chained comparison, a CMP b CMP c, etc.
        left = node.left
        values = []
        for op, comp in zip(ops, comps):
            new_node = self.visit(
                ast.Compare(comparators=[comp], left=left, ops=[self.translate_In(op)])
            )
            left = comp
            values.append(new_node)
        return self.visit(ast.BoolOp(op=ast.And(), values=values))

    def _try_visit_binop(self, bop):
        if isinstance(bop, (Op, Term)):
            return bop
        return self.visit(bop)

    def visit_BoolOp(self, node, **kwargs):
        def visitor(x, y):
            lhs = self._try_visit_binop(x)
            rhs = self._try_visit_binop(y)

            op, op_class, lhs, rhs = self._maybe_transform_eq_ne(node, lhs, rhs)
            return self._maybe_evaluate_binop(op, node.op, lhs, rhs)

        operands = node.values
        return reduce(visitor, operands)


_python_not_supported = frozenset(["Dict", "BoolOp", "In", "NotIn"])
_numexpr_supported_calls = frozenset(REDUCTIONS + MATHOPS)


@disallow(
    (_unsupported_nodes | _python_not_supported)
    - (_boolop_nodes | frozenset(["BoolOp", "Attribute", "In", "NotIn", "Tuple"]))
)
class PandasExprVisitor(BaseExprVisitor):
    def __init__(
        self,
        env,
        engine,
        parser,
        preparser=partial(
            _preparse,
            f=_compose(_replace_locals, _replace_booleans, clean_backtick_quoted_toks),
        ),
    ) -> None:
        super().__init__(env, engine, parser, preparser)


@disallow(_unsupported_nodes | _python_not_supported | frozenset(["Not"]))
class PythonExprVisitor(BaseExprVisitor):
    def __init__(
        self, env, engine, parser, preparser=lambda source, f=None: source
    ) -> None:
        super().__init__(env, engine, parser, preparser=preparser)


class Expr:
    """
    Object encapsulating an expression.

    Parameters
    ----------
    expr : str
    engine : str, optional, default 'numexpr'
    parser : str, optional, default 'pandas'
    env : Scope, optional, default None
    level : int, optional, default 2
    """

    env: Scope
    engine: str
    parser: str

    def __init__(
        self,
        expr,
        engine: str = "numexpr",
        parser: str = "pandas",
        env: Scope | None = None,
        level: int = 0,
    ) -> None:
        self.expr = expr
        self.env = env or Scope(level=level + 1)
        self.engine = engine
        self.parser = parser
        self._visitor = PARSERS[parser](self.env, self.engine, self.parser)
        self.terms = self.parse()

    @property
    def assigner(self):
        return getattr(self._visitor, "assigner", None)

    def __call__(self):
        return self.terms(self.env)

    def __repr__(self) -> str:
        return printing.pprint_thing(self.terms)

    def __len__(self) -> int:
        return len(self.expr)

    def parse(self):
        """
        Parse an expression.
        """
        return self._visitor.visit(self.expr)

    @property
    def names(self):
        """
        Get the names in an expression.
        """
        if is_term(self.terms):
            return frozenset([self.terms.name])
        return frozenset(term.name for term in com.flatten(self.terms))


PARSERS = {"python": PythonExprVisitor, "pandas": PandasExprVisitor}
<|endoftext|>from __future__ import annotations

from functools import reduce

import numpy as np

from pandas._config import get_option


def ensure_decoded(s) -> str:
    """
    If we have bytes, decode them to unicode.
    """
    if isinstance(s, (np.bytes_, bytes)):
        s = s.decode(get_option("display.encoding"))
    return s


def result_type_many(*arrays_and_dtypes):
    """
    Wrapper around numpy.result_type which overcomes the NPY_MAXARGS (32)
    argument limit.
    """
    try:
        return np.result_type(*arrays_and_dtypes)
    except ValueError:
        # we have > NPY_MAXARGS terms in our expression
        return reduce(np.result_type, arrays_and_dtypes)
    except TypeError:
        from pandas.core.dtypes.cast import find_common_type
        from pandas.core.dtypes.common import is_extension_array_dtype

        arr_and_dtypes = list(arrays_and_dtypes)
        ea_dtypes, non_ea_dtypes = [], []
        for arr_or_dtype in arr_and_dtypes:
            if is_extension_array_dtype(arr_or_dtype):
                ea_dtypes.append(arr_or_dtype)
            else:
                non_ea_dtypes.append(arr_or_dtype)

        if non_ea_dtypes:
            try:
                np_dtype = np.result_type(*non_ea_dtypes)
            except ValueError:
                np_dtype = reduce(np.result_type, arrays_and_dtypes)
            return find_common_type(ea_dtypes + [np_dtype])

        return find_common_type(ea_dtypes)
<|endoftext|>"""
Engine classes for :func:`~pandas.eval`
"""
from __future__ import annotations

import abc
from typing import TYPE_CHECKING

from pandas.errors import NumExprClobberingError

from pandas.core.computation.align import (
    align_terms,
    reconstruct_object,
)
from pandas.core.computation.ops import (
    MATHOPS,
    REDUCTIONS,
)

from pandas.io.formats import printing

if TYPE_CHECKING:
    from pandas.core.computation.expr import Expr

_ne_builtins = frozenset(MATHOPS + REDUCTIONS)


def _check_ne_builtin_clash(expr: Expr) -> None:
    """
    Attempt to prevent foot-shooting in a helpful way.

    Parameters
    ----------
    expr : Expr
        Terms can contain
    """
    names = expr.names
    overlap = names & _ne_builtins

    if overlap:
        s = ", ".join([repr(x) for x in overlap])
        raise NumExprClobberingError(
            f'Variables in expression "{expr}" overlap with builtins: ({s})'
        )


class AbstractEngine(metaclass=abc.ABCMeta):
    """Object serving as a base class for all engines."""

    has_neg_frac = False

    def __init__(self, expr) -> None:
        self.expr = expr
        self.aligned_axes = None
        self.result_type = None

    def convert(self) -> str:
        """
        Convert an expression for evaluation.

        Defaults to return the expression as a string.
        """
        return printing.pprint_thing(self.expr)

    def evaluate(self) -> object:
        """
        Run the engine on the expression.

        This method performs alignment which is necessary no matter what engine
        is being used, thus its implementation is in the base class.

        Returns
        -------
        object
            The result of the passed expression.
        """
        if not self._is_aligned:
            self.result_type, self.aligned_axes = align_terms(self.expr.terms)

        # make sure no names in resolvers and locals/globals clash
        res = self._evaluate()
        return reconstruct_object(
            self.result_type, res, self.aligned_axes, self.expr.terms.return_type
        )

    @property
    def _is_aligned(self) -> bool:
        return self.aligned_axes is not None and self.result_type is not None

    @abc.abstractmethod
    def _evaluate(self):
        """
        Return an evaluated expression.

        Parameters
        ----------
        env : Scope
            The local and global environment in which to evaluate an
            expression.

        Notes
        -----
        Must be implemented by subclasses.
        """


class NumExprEngine(AbstractEngine):
    """NumExpr engine class"""

    has_neg_frac = True

    def _evaluate(self):
        import numexpr as ne

        # convert the expression to a valid numexpr expression
        s = self.convert()

        env = self.expr.env
        scope = env.full_scope
        _check_ne_builtin_clash(self.expr)
        return ne.evaluate(s, local_dict=scope)


class PythonEngine(AbstractEngine):
    """
    Evaluate an expression in Python space.

    Mostly for testing purposes.
    """

    has_neg_frac = False

    def evaluate(self):
        return self.expr()

    def _evaluate(self) -> None:
        pass


ENGINES: dict[str, type[AbstractEngine]] = {
    "numexpr": NumExprEngine,
    "python": PythonEngine,
}
<|endoftext|>"""
Module for scope operations
"""
from __future__ import annotations

from collections import ChainMap
import datetime
import inspect
from io import StringIO
import itertools
import pprint
import struct
import sys
from typing import TypeVar

import numpy as np

from pandas._libs.tslibs import Timestamp
from pandas.errors import UndefinedVariableError

_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


# https://docs.python.org/3/library/collections.html#chainmap-examples-and-recipes
class DeepChainMap(ChainMap[_KT, _VT]):
    """
    Variant of ChainMap that allows direct updates to inner scopes.

    Only works when all passed mapping are mutable.
    """

    def __setitem__(self, key: _KT, value: _VT) -> None:
        for mapping in self.maps:
            if key in mapping:
                mapping[key] = value
                return
        self.maps[0][key] = value

    def __delitem__(self, key: _KT) -> None:
        """
        Raises
        ------
        KeyError
            If `key` doesn't exist.
        """
        for mapping in self.maps:
            if key in mapping:
                del mapping[key]
                return
        raise KeyError(key)


def ensure_scope(
    level: int, global_dict=None, local_dict=None, resolvers=(), target=None
) -> Scope:
    """Ensure that we are grabbing the correct scope."""
    return Scope(
        level + 1,
        global_dict=global_dict,
        local_dict=local_dict,
        resolvers=resolvers,
        target=target,
    )


def _replacer(x) -> str:
    """
    Replace a number with its hexadecimal representation. Used to tag
    temporary variables with their calling scope's id.
    """
    # get the hex repr of the binary char and remove 0x and pad by pad_size
    # zeros
    try:
        hexin = ord(x)
    except TypeError:
        # bytes literals masquerade as ints when iterating in py3
        hexin = x

    return hex(hexin)


def _raw_hex_id(obj) -> str:
    """Return the padded hexadecimal id of ``obj``."""
    # interpret as a pointer since that's what really what id returns
    packed = struct.pack("@P", id(obj))
    return "".join([_replacer(x) for x in packed])


DEFAULT_GLOBALS = {
    "Timestamp": Timestamp,
    "datetime": datetime.datetime,
    "True": True,
    "False": False,
    "list": list,
    "tuple": tuple,
    "inf": np.inf,
    "Inf": np.inf,
}


def _get_pretty_string(obj) -> str:
    """
    Return a prettier version of obj.

    Parameters
    ----------
    obj : object
        Object to pretty print

    Returns
    -------
    str
        Pretty print object repr
    """
    sio = StringIO()
    pprint.pprint(obj, stream=sio)
    return sio.getvalue()


class Scope:
    """
    Object to hold scope, with a few bells to deal with some custom syntax
    and contexts added by pandas.

    Parameters
    ----------
    level : int
    global_dict : dict or None, optional, default None
    local_dict : dict or Scope or None, optional, default None
    resolvers : list-like or None, optional, default None
    target : object

    Attributes
    ----------
    level : int
    scope : DeepChainMap
    target : object
    temps : dict
    """

    __slots__ = ["level", "scope", "target", "resolvers", "temps"]
    level: int
    scope: DeepChainMap
    resolvers: DeepChainMap
    temps: dict

    def __init__(
        self, level: int, global_dict=None, local_dict=None, resolvers=(), target=None
    ) -> None:
        self.level = level + 1

        # shallow copy because we don't want to keep filling this up with what
        # was there before if there are multiple calls to Scope/_ensure_scope
        self.scope = DeepChainMap(DEFAULT_GLOBALS.copy())
        self.target = target

        if isinstance(local_dict, Scope):
            self.scope.update(local_dict.scope)
            if local_dict.target is not None:
                self.target = local_dict.target
            self._update(local_dict.level)

        frame = sys._getframe(self.level)

        try:
            # shallow copy here because we don't want to replace what's in
            # scope when we align terms (alignment accesses the underlying
            # numpy array of pandas objects)
            scope_global = self.scope.new_child(
                (global_dict if global_dict is not None else frame.f_globals).copy()
            )
            self.scope = DeepChainMap(scope_global)
            if not isinstance(local_dict, Scope):
                scope_local = self.scope.new_child(
                    (local_dict if local_dict is not None else frame.f_locals).copy()
                )
                self.scope = DeepChainMap(scope_local)
        finally:
            del frame

        # assumes that resolvers are going from outermost scope to inner
        if isinstance(local_dict, Scope):
            resolvers += tuple(local_dict.resolvers.maps)
        self.resolvers = DeepChainMap(*resolvers)
        self.temps = {}

    def __repr__(self) -> str:
        scope_keys = _get_pretty_string(list(self.scope.keys()))
        res_keys = _get_pretty_string(list(self.resolvers.keys()))
        return f"{type(self).__name__}(scope={scope_keys}, resolvers={res_keys})"

    @property
    def has_resolvers(self) -> bool:
        """
        Return whether we have any extra scope.

        For example, DataFrames pass Their columns as resolvers during calls to
        ``DataFrame.eval()`` and ``DataFrame.query()``.

        Returns
        -------
        hr : bool
        """
        return bool(len(self.resolvers))

    def resolve(self, key: str, is_local: bool):
        """
        Resolve a variable name in a possibly local context.

        Parameters
        ----------
        key : str
            A variable name
        is_local : bool
            Flag indicating whether the variable is local or not (prefixed with
            the '@' symbol)

        Returns
        -------
        value : object
            The value of a particular variable
        """
        try:
            # only look for locals in outer scope
            if is_local:
                return self.scope[key]

            # not a local variable so check in resolvers if we have them
            if self.has_resolvers:
                return self.resolvers[key]

            # if we're here that means that we have no locals and we also have
            # no resolvers
            assert not is_local and not self.has_resolvers
            return self.scope[key]
        except KeyError:
            try:
                # last ditch effort we look in temporaries
                # these are created when parsing indexing expressions
                # e.g., df[df > 0]
                return self.temps[key]
            except KeyError as err:
                raise UndefinedVariableError(key, is_local) from err

    def swapkey(self, old_key: str, new_key: str, new_value=None) -> None:
        """
        Replace a variable name, with a potentially new value.

        Parameters
        ----------
        old_key : str
            Current variable name to replace
        new_key : str
            New variable name to replace `old_key` with
        new_value : object
            Value to be replaced along with the possible renaming
        """
        if self.has_resolvers:
            maps = self.resolvers.maps + self.scope.maps
        else:
            maps = self.scope.maps

        maps.append(self.temps)

        for mapping in maps:
            if old_key in mapping:
                mapping[new_key] = new_value
                return

    def _get_vars(self, stack, scopes: list[str]) -> None:
        """
        Get specifically scoped variables from a list of stack frames.

        Parameters
        ----------
        stack : list
            A list of stack frames as returned by ``inspect.stack()``
        scopes : sequence of strings
            A sequence containing valid stack frame attribute names that
            evaluate to a dictionary. For example, ('locals', 'globals')
        """
        variables = itertools.product(scopes, stack)
        for scope, (frame, _, _, _, _, _) in variables:
            try:
                d = getattr(frame, f"f_{scope}")
                self.scope = DeepChainMap(self.scope.new_child(d))
            finally:
                # won't remove it, but DECREF it
                # in Py3 this probably isn't necessary since frame won't be
                # scope after the loop
                del frame

    def _update(self, level: int) -> None:
        """
        Update the current scope by going back `level` levels.

        Parameters
        ----------
        level : int
        """
        sl = level + 1

        # add sl frames to the scope starting with the
        # most distant and overwriting with more current
        # makes sure that we can capture variable scope
        stack = inspect.stack()

        try:
            self._get_vars(stack[:sl], scopes=["locals"])
        finally:
            del stack[:], stack

    def add_tmp(self, value) -> str:
        """
        Add a temporary variable to the scope.

        Parameters
        ----------
        value : object
            An arbitrary object to be assigned to a temporary variable.

        Returns
        -------
        str
            The name of the temporary variable created.
        """
        name = f"{type(value).__name__}_{self.ntemps}_{_raw_hex_id(self)}"

        # add to inner most scope
        assert name not in self.temps
        self.temps[name] = value
        assert name in self.temps

        # only increment if the variable gets put in the scope
        return name

    @property
    def ntemps(self) -> int:
        """The number of temporary variables in this scope"""
