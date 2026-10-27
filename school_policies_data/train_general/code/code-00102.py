        else:
            raise ValueError(
                "indicator option can only accept boolean or string arguments"
            )

    @final
    def _indicator_pre_merge(
        self, left: DataFrame, right: DataFrame
    ) -> tuple[DataFrame, DataFrame]:
        columns = left.columns.union(right.columns)

        for i in ["_left_indicator", "_right_indicator"]:
            if i in columns:
                raise ValueError(
                    "Cannot use `indicator=True` option when "
                    f"data contains a column named {i}"
                )
        if self._indicator_name in columns:
            raise ValueError(
                "Cannot use name of an existing column for indicator column"
            )

        left = left.copy()
        right = right.copy()

        left["_left_indicator"] = 1
        left["_left_indicator"] = left["_left_indicator"].astype("int8")

        right["_right_indicator"] = 2
        right["_right_indicator"] = right["_right_indicator"].astype("int8")

        return left, right

    @final
    def _indicator_post_merge(self, result: DataFrame) -> DataFrame:
        result["_left_indicator"] = result["_left_indicator"].fillna(0)
        result["_right_indicator"] = result["_right_indicator"].fillna(0)

        result[self._indicator_name] = Categorical(
            (result["_left_indicator"] + result["_right_indicator"]),
            categories=[1, 2, 3],
        )
        result[self._indicator_name] = result[
            self._indicator_name
        ].cat.rename_categories(["left_only", "right_only", "both"])

        result = result.drop(labels=["_left_indicator", "_right_indicator"], axis=1)
        return result

    @final
    def _maybe_restore_index_levels(self, result: DataFrame) -> None:
        """
        Restore index levels specified as `on` parameters

        Here we check for cases where `self.left_on` and `self.right_on` pairs
        each reference an index level in their respective DataFrames. The
        joined columns corresponding to these pairs are then restored to the
        index of `result`.

        **Note:** This method has side effects. It modifies `result` in-place

        Parameters
        ----------
        result: DataFrame
            merge result

        Returns
        -------
        None
        """
        names_to_restore = []
        for name, left_key, right_key in zip(
            self.join_names, self.left_on, self.right_on
        ):
            if (
                # Argument 1 to "_is_level_reference" of "NDFrame" has incompatible
                # type "Union[Hashable, ExtensionArray, Index, Series]"; expected
                # "Hashable"
                self.orig_left._is_level_reference(left_key)  # type: ignore[arg-type]
                # Argument 1 to "_is_level_reference" of "NDFrame" has incompatible
                # type "Union[Hashable, ExtensionArray, Index, Series]"; expected
                # "Hashable"
                and self.orig_right._is_level_reference(
                    right_key  # type: ignore[arg-type]
                )
                and left_key == right_key
                and name not in result.index.names
            ):
                names_to_restore.append(name)

        if names_to_restore:
            result.set_index(names_to_restore, inplace=True)

    @final
    def _maybe_add_join_keys(
        self,
        result: DataFrame,
        left_indexer: npt.NDArray[np.intp] | None,
        right_indexer: npt.NDArray[np.intp] | None,
    ) -> None:
        left_has_missing = None
        right_has_missing = None

        assert all(isinstance(x, _known) for x in self.left_join_keys)

        keys = zip(self.join_names, self.left_on, self.right_on)
        for i, (name, lname, rname) in enumerate(keys):
            if not _should_fill(lname, rname):
                continue

            take_left, take_right = None, None

            if name in result:
                if left_indexer is not None or right_indexer is not None:
                    if name in self.left:
                        if left_has_missing is None:
                            left_has_missing = (
                                False
                                if left_indexer is None
                                else (left_indexer == -1).any()
                            )

                        if left_has_missing:
                            take_right = self.right_join_keys[i]

                            if result[name].dtype != self.left[name].dtype:
                                take_left = self.left[name]._values

                    elif name in self.right:
                        if right_has_missing is None:
                            right_has_missing = (
                                False
                                if right_indexer is None
                                else (right_indexer == -1).any()
                            )

                        if right_has_missing:
                            take_left = self.left_join_keys[i]

                            if result[name].dtype != self.right[name].dtype:
                                take_right = self.right[name]._values

            else:
                take_left = self.left_join_keys[i]
                take_right = self.right_join_keys[i]

            if take_left is not None or take_right is not None:
                if take_left is None:
                    lvals = result[name]._values
                elif left_indexer is None:
                    lvals = take_left
                else:
                    # TODO: can we pin down take_left's type earlier?
                    take_left = extract_array(take_left, extract_numpy=True)
                    lfill = na_value_for_dtype(take_left.dtype)
                    lvals = algos.take_nd(take_left, left_indexer, fill_value=lfill)

                if take_right is None:
                    rvals = result[name]._values
                elif right_indexer is None:
                    rvals = take_right
                else:
                    # TODO: can we pin down take_right's type earlier?
                    taker = extract_array(take_right, extract_numpy=True)
                    rfill = na_value_for_dtype(taker.dtype)
                    rvals = algos.take_nd(taker, right_indexer, fill_value=rfill)

                # if we have an all missing left_indexer
                # make sure to just use the right values or vice-versa
                if left_indexer is not None and (left_indexer == -1).all():
                    key_col = Index(rvals, dtype=rvals.dtype, copy=False)
                    result_dtype = rvals.dtype
                elif right_indexer is not None and (right_indexer == -1).all():
                    key_col = Index(lvals, dtype=lvals.dtype, copy=False)
                    result_dtype = lvals.dtype
                else:
                    key_col = Index(lvals, dtype=lvals.dtype, copy=False)
                    if left_indexer is not None:
                        mask_left = left_indexer == -1
                        key_col = key_col.where(~mask_left, rvals)
                    result_dtype = find_common_type([lvals.dtype, rvals.dtype])
                    if (
                        lvals.dtype.kind == "M"
                        and rvals.dtype.kind == "M"
                        and result_dtype.kind == "O"
                    ):
                        # TODO(non-nano) Workaround for common_type not dealing
                        # with different resolutions
                        result_dtype = key_col.dtype

                if result._is_label_reference(name):
                    result[name] = result._constructor_sliced(
                        key_col, dtype=result_dtype, index=result.index
                    )
                elif result._is_level_reference(name):
                    if isinstance(result.index, MultiIndex):
                        key_col.name = name
                        idx_list = [
                            result.index.get_level_values(level_name)
                            if level_name != name
                            else key_col
                            for level_name in result.index.names
                        ]

                        result.set_index(idx_list, inplace=True)
                    else:
                        key_col.name = name
                        result.index = key_col
                else:
                    result.insert(i, name or f"key_{i}", key_col)

    def _get_join_indexers(
        self,
    ) -> tuple[npt.NDArray[np.intp] | None, npt.NDArray[np.intp] | None]:
        """return the join indexers"""
        # make mypy happy
        assert self.how != "asof"
        return get_join_indexers(
            self.left_join_keys, self.right_join_keys, sort=self.sort, how=self.how
        )

    @final
    def _get_join_info(
        self,
    ) -> tuple[Index, npt.NDArray[np.intp] | None, npt.NDArray[np.intp] | None]:
        left_ax = self.left.index
        right_ax = self.right.index

        if self.left_index and self.right_index and self.how != "asof":
            join_index, left_indexer, right_indexer = left_ax.join(
                right_ax, how=self.how, return_indexers=True, sort=self.sort
            )

        elif self.right_index and self.how == "left":
            join_index, left_indexer, right_indexer = _left_join_on_index(
                left_ax, right_ax, self.left_join_keys, sort=self.sort
            )

        elif self.left_index and self.how == "right":
            join_index, right_indexer, left_indexer = _left_join_on_index(
                right_ax, left_ax, self.right_join_keys, sort=self.sort
            )
        else:
            (left_indexer, right_indexer) = self._get_join_indexers()

            if self.right_index:
                if len(self.left) > 0:
                    join_index = self._create_join_index(
                        left_ax,
                        right_ax,
                        left_indexer,
                        how="right",
                    )
                elif right_indexer is None:
                    join_index = right_ax.copy()
                else:
                    join_index = right_ax.take(right_indexer)
            elif self.left_index:
                if self.how == "asof":
                    # GH#33463 asof should always behave like a left merge
                    join_index = self._create_join_index(
                        left_ax,
                        right_ax,
                        left_indexer,
                        how="left",
                    )

                elif len(self.right) > 0:
                    join_index = self._create_join_index(
                        right_ax,
                        left_ax,
                        right_indexer,
                        how="left",
                    )
                elif left_indexer is None:
                    join_index = left_ax.copy()
                else:
                    join_index = left_ax.take(left_indexer)
            else:
                n = len(left_ax) if left_indexer is None else len(left_indexer)
                join_index = default_index(n)

        return join_index, left_indexer, right_indexer

    @final
    def _create_join_index(
        self,
        index: Index,
        other_index: Index,
        indexer: npt.NDArray[np.intp] | None,
        how: JoinHow = "left",
    ) -> Index:
        """
        Create a join index by rearranging one index to match another

        Parameters
        ----------
        index : Index
            index being rearranged
        other_index : Index
            used to supply values not found in index
        indexer : np.ndarray[np.intp] or None
            how to rearrange index
        how : str
            Replacement is only necessary if indexer based on other_index.

        Returns
        -------
        Index
        """
        if self.how in (how, "outer") and not isinstance(other_index, MultiIndex):
            # if final index requires values in other_index but not target
            # index, indexer may hold missing (-1) values, causing Index.take
            # to take the final value in target index. So, we set the last
            # element to be the desired fill value. We do not use allow_fill
            # and fill_value because it throws a ValueError on integer indices
            mask = indexer == -1
            if np.any(mask):
                fill_value = na_value_for_dtype(index.dtype, compat=False)
                index = index.append(Index([fill_value]))
        if indexer is None:
            return index.copy()
        return index.take(indexer)

    @final
    def _get_merge_keys(
        self,
    ) -> tuple[
        list[ArrayLike],
        list[ArrayLike],
        list[Hashable],
        list[Hashable],
        list[Hashable],
    ]:
        """
        Returns
        -------
        left_keys, right_keys, join_names, left_drop, right_drop
        """
        left_keys: list[ArrayLike] = []
        right_keys: list[ArrayLike] = []
        join_names: list[Hashable] = []
        right_drop: list[Hashable] = []
        left_drop: list[Hashable] = []

        left, right = self.left, self.right

        is_lkey = lambda x: isinstance(x, _known) and len(x) == len(left)
        is_rkey = lambda x: isinstance(x, _known) and len(x) == len(right)

        # Note that pd.merge_asof() has separate 'on' and 'by' parameters. A
        # user could, for example, request 'left_index' and 'left_by'. In a
        # regular pd.merge(), users cannot specify both 'left_index' and
        # 'left_on'. (Instead, users have a MultiIndex). That means the
        # self.left_on in this function is always empty in a pd.merge(), but
        # a pd.merge_asof(left_index=True, left_by=...) will result in a
        # self.left_on array with a None in the middle of it. This requires
        # a work-around as designated in the code below.
        # See _validate_left_right_on() for where this happens.

        # ugh, spaghetti re #733
        if _any(self.left_on) and _any(self.right_on):
            for lk, rk in zip(self.left_on, self.right_on):
                lk = extract_array(lk, extract_numpy=True)
                rk = extract_array(rk, extract_numpy=True)
                if is_lkey(lk):
                    lk = cast(ArrayLike, lk)
                    left_keys.append(lk)
                    if is_rkey(rk):
                        rk = cast(ArrayLike, rk)
                        right_keys.append(rk)
                        join_names.append(None)  # what to do?
                    else:
                        # Then we're either Hashable or a wrong-length arraylike,
                        #  the latter of which will raise
                        rk = cast(Hashable, rk)
                        if rk is not None:
                            right_keys.append(right._get_label_or_level_values(rk))
                            join_names.append(rk)
                        else:
                            # work-around for merge_asof(right_index=True)
                            right_keys.append(right.index._values)
                            join_names.append(right.index.name)
                else:
                    if not is_rkey(rk):
                        # Then we're either Hashable or a wrong-length arraylike,
                        #  the latter of which will raise
                        rk = cast(Hashable, rk)
                        if rk is not None:
                            right_keys.append(right._get_label_or_level_values(rk))
                        else:
                            # work-around for merge_asof(right_index=True)
                            right_keys.append(right.index._values)
                        if lk is not None and lk == rk:  # FIXME: what about other NAs?
                            right_drop.append(rk)
                    else:
                        rk = cast(ArrayLike, rk)
                        right_keys.append(rk)
                    if lk is not None:
                        # Then we're either Hashable or a wrong-length arraylike,
                        #  the latter of which will raise
                        lk = cast(Hashable, lk)
                        left_keys.append(left._get_label_or_level_values(lk))
                        join_names.append(lk)
                    else:
                        # work-around for merge_asof(left_index=True)
                        left_keys.append(left.index._values)
                        join_names.append(left.index.name)
        elif _any(self.left_on):
            for k in self.left_on:
                if is_lkey(k):
                    k = extract_array(k, extract_numpy=True)
                    k = cast(ArrayLike, k)
                    left_keys.append(k)
                    join_names.append(None)
                else:
                    # Then we're either Hashable or a wrong-length arraylike,
                    #  the latter of which will raise
                    k = cast(Hashable, k)
                    left_keys.append(left._get_label_or_level_values(k))
                    join_names.append(k)
            if isinstance(self.right.index, MultiIndex):
                right_keys = [
                    lev._values.take(lev_codes)
                    for lev, lev_codes in zip(
                        self.right.index.levels, self.right.index.codes
                    )
                ]
            else:
                right_keys = [self.right.index._values]
        elif _any(self.right_on):
            for k in self.right_on:
                k = extract_array(k, extract_numpy=True)
                if is_rkey(k):
                    k = cast(ArrayLike, k)
                    right_keys.append(k)
                    join_names.append(None)
                else:
                    # Then we're either Hashable or a wrong-length arraylike,
                    #  the latter of which will raise
                    k = cast(Hashable, k)
                    right_keys.append(right._get_label_or_level_values(k))
                    join_names.append(k)
            if isinstance(self.left.index, MultiIndex):
                left_keys = [
                    lev._values.take(lev_codes)
                    for lev, lev_codes in zip(
                        self.left.index.levels, self.left.index.codes
                    )
                ]
            else:
                left_keys = [self.left.index._values]

        return left_keys, right_keys, join_names, left_drop, right_drop

    @final
    def _maybe_coerce_merge_keys(self) -> None:
        # we have valid merges but we may have to further
        # coerce these if they are originally incompatible types
        #
        # for example if these are categorical, but are not dtype_equal
        # or if we have object and integer dtypes

        for lk, rk, name in zip(
            self.left_join_keys, self.right_join_keys, self.join_names
        ):
            if (len(lk) and not len(rk)) or (not len(lk) and len(rk)):
                continue

            lk = extract_array(lk, extract_numpy=True)
            rk = extract_array(rk, extract_numpy=True)

            lk_is_cat = isinstance(lk.dtype, CategoricalDtype)
            rk_is_cat = isinstance(rk.dtype, CategoricalDtype)
            lk_is_object_or_string = is_object_dtype(lk.dtype) or is_string_dtype(
                lk.dtype
            )
            rk_is_object_or_string = is_object_dtype(rk.dtype) or is_string_dtype(
                rk.dtype
            )

            # if either left or right is a categorical
            # then the must match exactly in categories & ordered
            if lk_is_cat and rk_is_cat:
                lk = cast(Categorical, lk)
                rk = cast(Categorical, rk)
                if lk._categories_match_up_to_permutation(rk):
                    continue

            elif lk_is_cat or rk_is_cat:
                pass

            elif lk.dtype == rk.dtype:
                continue

            msg = (
                f"You are trying to merge on {lk.dtype} and {rk.dtype} columns "
                f"for key '{name}'. If you wish to proceed you should use pd.concat"
            )

            # if we are numeric, then allow differing
            # kinds to proceed, eg. int64 and int8, int and float
            # further if we are object, but we infer to
            # the same, then proceed
            if is_numeric_dtype(lk.dtype) and is_numeric_dtype(rk.dtype):
                if lk.dtype.kind == rk.dtype.kind:
                    continue

                if isinstance(lk.dtype, ExtensionDtype) and not isinstance(
                    rk.dtype, ExtensionDtype
                ):
                    ct = find_common_type([lk.dtype, rk.dtype])
                    if isinstance(ct, ExtensionDtype):
                        com_cls = ct.construct_array_type()
                        rk = com_cls._from_sequence(rk, dtype=ct, copy=False)
                    else:
                        rk = rk.astype(ct)
                elif isinstance(rk.dtype, ExtensionDtype):
                    ct = find_common_type([lk.dtype, rk.dtype])
                    if isinstance(ct, ExtensionDtype):
                        com_cls = ct.construct_array_type()
                        lk = com_cls._from_sequence(lk, dtype=ct, copy=False)
                    else:
                        lk = lk.astype(ct)

                # check whether ints and floats
                if is_integer_dtype(rk.dtype) and is_float_dtype(lk.dtype):
                    # GH 47391 numpy > 1.24 will raise a RuntimeError for nan -> int
                    with np.errstate(invalid="ignore"):
                        # error: Argument 1 to "astype" of "ndarray" has incompatible
                        # type "Union[ExtensionDtype, Any, dtype[Any]]"; expected
                        # "Union[dtype[Any], Type[Any], _SupportsDType[dtype[Any]]]"
                        casted = lk.astype(rk.dtype)  # type: ignore[arg-type]

                    mask = ~np.isnan(lk)
                    match = lk == casted
                    if not match[mask].all():
                        warnings.warn(
                            "You are merging on int and float "
                            "columns where the float values "
                            "are not equal to their int representation.",
                            UserWarning,
                            stacklevel=find_stack_level(),
                        )
                    continue

                if is_float_dtype(rk.dtype) and is_integer_dtype(lk.dtype):
                    # GH 47391 numpy > 1.24 will raise a RuntimeError for nan -> int
                    with np.errstate(invalid="ignore"):
                        # error: Argument 1 to "astype" of "ndarray" has incompatible
                        # type "Union[ExtensionDtype, Any, dtype[Any]]"; expected
                        # "Union[dtype[Any], Type[Any], _SupportsDType[dtype[Any]]]"
                        casted = rk.astype(lk.dtype)  # type: ignore[arg-type]

                    mask = ~np.isnan(rk)
                    match = rk == casted
                    if not match[mask].all():
                        warnings.warn(
                            "You are merging on int and float "
                            "columns where the float values "
                            "are not equal to their int representation.",
                            UserWarning,
                            stacklevel=find_stack_level(),
                        )
                    continue

                # let's infer and see if we are ok
                if lib.infer_dtype(lk, skipna=False) == lib.infer_dtype(
                    rk, skipna=False
                ):
                    continue

            # Check if we are trying to merge on obviously
            # incompatible dtypes GH 9780, GH 15800

            # bool values are coerced to object
            elif (lk_is_object_or_string and is_bool_dtype(rk.dtype)) or (
                is_bool_dtype(lk.dtype) and rk_is_object_or_string
            ):
                pass

            # object values are allowed to be merged
            elif (lk_is_object_or_string and is_numeric_dtype(rk.dtype)) or (
                is_numeric_dtype(lk.dtype) and rk_is_object_or_string
            ):
                inferred_left = lib.infer_dtype(lk, skipna=False)
                inferred_right = lib.infer_dtype(rk, skipna=False)
                bool_types = ["integer", "mixed-integer", "boolean", "empty"]
                string_types = ["string", "unicode", "mixed", "bytes", "empty"]

                # inferred bool
                if inferred_left in bool_types and inferred_right in bool_types:
                    pass

                # unless we are merging non-string-like with string-like
                elif (
                    inferred_left in string_types and inferred_right not in string_types
                ) or (
                    inferred_right in string_types and inferred_left not in string_types
                ):
                    raise ValueError(msg)

            # datetimelikes must match exactly
            elif needs_i8_conversion(lk.dtype) and not needs_i8_conversion(rk.dtype):
                raise ValueError(msg)
            elif not needs_i8_conversion(lk.dtype) and needs_i8_conversion(rk.dtype):
                raise ValueError(msg)
            elif isinstance(lk.dtype, DatetimeTZDtype) and not isinstance(
                rk.dtype, DatetimeTZDtype
            ):
                raise ValueError(msg)
            elif not isinstance(lk.dtype, DatetimeTZDtype) and isinstance(
                rk.dtype, DatetimeTZDtype
            ):
                raise ValueError(msg)
            elif (
                isinstance(lk.dtype, DatetimeTZDtype)
                and isinstance(rk.dtype, DatetimeTZDtype)
            ) or (lk.dtype.kind == "M" and rk.dtype.kind == "M"):
                # allows datetime with different resolutions
                continue
            # datetime and timedelta not allowed
            elif lk.dtype.kind == "M" and rk.dtype.kind == "m":
                raise ValueError(msg)
            elif lk.dtype.kind == "m" and rk.dtype.kind == "M":
                raise ValueError(msg)

            elif is_object_dtype(lk.dtype) and is_object_dtype(rk.dtype):
                continue

            # Houston, we have a problem!
            # let's coerce to object if the dtypes aren't
            # categorical, otherwise coerce to the category
            # dtype. If we coerced categories to object,
            # then we would lose type information on some
            # columns, and end up trying to merge
            # incompatible dtypes. See GH 16900.
            if name in self.left.columns:
                typ = cast(Categorical, lk).categories.dtype if lk_is_cat else object
                self.left = self.left.copy()
                self.left[name] = self.left[name].astype(typ)
            if name in self.right.columns:
                typ = cast(Categorical, rk).categories.dtype if rk_is_cat else object
                self.right = self.right.copy()
                self.right[name] = self.right[name].astype(typ)

    def _validate_left_right_on(self, left_on, right_on):
        left_on = com.maybe_make_list(left_on)
        right_on = com.maybe_make_list(right_on)

        # Hm, any way to make this logic less complicated??
        if self.on is None and left_on is None and right_on is None:
            if self.left_index and self.right_index:
                left_on, right_on = (), ()
            elif self.left_index:
                raise MergeError("Must pass right_on or right_index=True")
            elif self.right_index:
                raise MergeError("Must pass left_on or left_index=True")
            else:
                # use the common columns
                left_cols = self.left.columns
                right_cols = self.right.columns
                common_cols = left_cols.intersection(right_cols)
                if len(common_cols) == 0:
                    raise MergeError(
                        "No common columns to perform merge on. "
                        f"Merge options: left_on={left_on}, "
                        f"right_on={right_on}, "
                        f"left_index={self.left_index}, "
                        f"right_index={self.right_index}"
                    )
                if (
                    not left_cols.join(common_cols, how="inner").is_unique
                    or not right_cols.join(common_cols, how="inner").is_unique
                ):
                    raise MergeError(f"Data columns not unique: {repr(common_cols)}")
                left_on = right_on = common_cols
        elif self.on is not None:
            if left_on is not None or right_on is not None:
                raise MergeError(
                    'Can only pass argument "on" OR "left_on" '
                    'and "right_on", not a combination of both.'
                )
            if self.left_index or self.right_index:
                raise MergeError(
                    'Can only pass argument "on" OR "left_index" '
                    'and "right_index", not a combination of both.'
                )
            left_on = right_on = self.on
        elif left_on is not None:
            if self.left_index:
                raise MergeError(
                    'Can only pass argument "left_on" OR "left_index" not both.'
                )
            if not self.right_index and right_on is None:
                raise MergeError('Must pass "right_on" OR "right_index".')
            n = len(left_on)
            if self.right_index:
                if len(left_on) != self.right.index.nlevels:
                    raise ValueError(
                        "len(left_on) must equal the number "
                        'of levels in the index of "right"'
                    )
                right_on = [None] * n
        elif right_on is not None:
            if self.right_index:
                raise MergeError(
                    'Can only pass argument "right_on" OR "right_index" not both.'
                )
            if not self.left_index and left_on is None:
                raise MergeError('Must pass "left_on" OR "left_index".')
            n = len(right_on)
            if self.left_index:
                if len(right_on) != self.left.index.nlevels:
                    raise ValueError(
                        "len(right_on) must equal the number "
                        'of levels in the index of "left"'
                    )
                left_on = [None] * n
        if len(right_on) != len(left_on):
            raise ValueError("len(right_on) must equal len(left_on)")

        return left_on, right_on

    @final
    def _validate_validate_kwd(self, validate: str) -> None:
        # Check uniqueness of each
        if self.left_index:
            left_unique = self.orig_left.index.is_unique
        else:
            left_unique = MultiIndex.from_arrays(self.left_join_keys).is_unique

        if self.right_index:
            right_unique = self.orig_right.index.is_unique
        else:
            right_unique = MultiIndex.from_arrays(self.right_join_keys).is_unique

        # Check data integrity
        if validate in ["one_to_one", "1:1"]:
            if not left_unique and not right_unique:
                raise MergeError(
                    "Merge keys are not unique in either left "
                    "or right dataset; not a one-to-one merge"
                )
            if not left_unique:
                raise MergeError(
                    "Merge keys are not unique in left dataset; not a one-to-one merge"
                )
            if not right_unique:
                raise MergeError(
                    "Merge keys are not unique in right dataset; not a one-to-one merge"
                )

        elif validate in ["one_to_many", "1:m"]:
            if not left_unique:
                raise MergeError(
                    "Merge keys are not unique in left dataset; not a one-to-many merge"
                )

        elif validate in ["many_to_one", "m:1"]:
            if not right_unique:
                raise MergeError(
                    "Merge keys are not unique in right dataset; "
                    "not a many-to-one merge"
                )

        elif validate in ["many_to_many", "m:m"]:
            pass

        else:
            raise ValueError(
                f'"{validate}" is not a valid argument. '
                "Valid arguments are:\n"
                '- "1:1"\n'
                '- "1:m"\n'
                '- "m:1"\n'
                '- "m:m"\n'
                '- "one_to_one"\n'
                '- "one_to_many"\n'
                '- "many_to_one"\n'
                '- "many_to_many"'
            )


def get_join_indexers(
    left_keys: list[ArrayLike],
    right_keys: list[ArrayLike],
    sort: bool = False,
    how: JoinHow = "inner",
) -> tuple[npt.NDArray[np.intp] | None, npt.NDArray[np.intp] | None]:
    """

    Parameters
    ----------
    left_keys : list[ndarray, ExtensionArray, Index, Series]
    right_keys : list[ndarray, ExtensionArray, Index, Series]
    sort : bool, default False
    how : {'inner', 'outer', 'left', 'right'}, default 'inner'

    Returns
    -------
    np.ndarray[np.intp] or None
        Indexer into the left_keys.
    np.ndarray[np.intp] or None
        Indexer into the right_keys.
    """
    assert len(left_keys) == len(
        right_keys
    ), "left_keys and right_keys must be the same length"

    # fast-path for empty left/right
    left_n = len(left_keys[0])
    right_n = len(right_keys[0])
    if left_n == 0:
        if how in ["left", "inner"]:
            return _get_empty_indexer()
        elif not sort and how in ["right", "outer"]:
            return _get_no_sort_one_missing_indexer(right_n, True)
    elif right_n == 0:
        if how in ["right", "inner"]:
            return _get_empty_indexer()
        elif not sort and how in ["left", "outer"]:
            return _get_no_sort_one_missing_indexer(left_n, False)

    lkey: ArrayLike
    rkey: ArrayLike
    if len(left_keys) > 1:
        # get left & right join labels and num. of levels at each location
        mapped = (
            _factorize_keys(left_keys[n], right_keys[n], sort=sort)
            for n in range(len(left_keys))
        )
        zipped = zip(*mapped)
        llab, rlab, shape = (list(x) for x in zipped)

        # get flat i8 keys from label lists
        lkey, rkey = _get_join_keys(llab, rlab, tuple(shape), sort)
    else:
        lkey = left_keys[0]
        rkey = right_keys[0]

    left = Index(lkey)
    right = Index(rkey)

    if (
        left.is_monotonic_increasing
        and right.is_monotonic_increasing
        and (left.is_unique or right.is_unique)
    ):
        _, lidx, ridx = left.join(right, how=how, return_indexers=True, sort=sort)
    else:
        lidx, ridx = get_join_indexers_non_unique(
            left._values, right._values, sort, how
        )

    if lidx is not None and is_range_indexer(lidx, len(left)):
        lidx = None
    if ridx is not None and is_range_indexer(ridx, len(right)):
        ridx = None
    return lidx, ridx


def get_join_indexers_non_unique(
    left: ArrayLike,
    right: ArrayLike,
    sort: bool = False,
    how: JoinHow = "inner",
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """
    Get join indexers for left and right.

    Parameters
    ----------
    left : ArrayLike
    right : ArrayLike
    sort : bool, default False
    how : {'inner', 'outer', 'left', 'right'}, default 'inner'

    Returns
    -------
    np.ndarray[np.intp]
        Indexer into left.
    np.ndarray[np.intp]
        Indexer into right.
    """
    lkey, rkey, count = _factorize_keys(left, right, sort=sort)
    if how == "left":
        lidx, ridx = libjoin.left_outer_join(lkey, rkey, count, sort=sort)
    elif how == "right":
        ridx, lidx = libjoin.left_outer_join(rkey, lkey, count, sort=sort)
    elif how == "inner":
        lidx, ridx = libjoin.inner_join(lkey, rkey, count, sort=sort)
    elif how == "outer":
        lidx, ridx = libjoin.full_outer_join(lkey, rkey, count)
    return lidx, ridx


def restore_dropped_levels_multijoin(
    left: MultiIndex,
    right: MultiIndex,
    dropped_level_names,
    join_index: Index,
    lindexer: npt.NDArray[np.intp],
    rindexer: npt.NDArray[np.intp],
) -> tuple[FrozenList, FrozenList, FrozenList]:
    """
    *this is an internal non-public method*

    Returns the levels, labels and names of a multi-index to multi-index join.
    Depending on the type of join, this method restores the appropriate
    dropped levels of the joined multi-index.
    The method relies on lindexer, rindexer which hold the index positions of
    left and right, where a join was feasible

    Parameters
    ----------
    left : MultiIndex
        left index
    right : MultiIndex
        right index
    dropped_level_names : str array
        list of non-common level names
    join_index : Index
        the index of the join between the
        common levels of left and right
    lindexer : np.ndarray[np.intp]
        left indexer
    rindexer : np.ndarray[np.intp]
        right indexer

    Returns
    -------
    levels : list of Index
        levels of combined multiindexes
    labels : np.ndarray[np.intp]
        labels of combined multiindexes
    names : List[Hashable]
        names of combined multiindex levels

    """

    def _convert_to_multiindex(index: Index) -> MultiIndex:
        if isinstance(index, MultiIndex):
            return index
        else:
            return MultiIndex.from_arrays([index._values], names=[index.name])

    # For multi-multi joins with one overlapping level,
    # the returned index if of type Index
    # Assure that join_index is of type MultiIndex
    # so that dropped levels can be appended
    join_index = _convert_to_multiindex(join_index)

    join_levels = join_index.levels
    join_codes = join_index.codes
    join_names = join_index.names

    # Iterate through the levels that must be restored
    for dropped_level_name in dropped_level_names:
        if dropped_level_name in left.names:
            idx = left
            indexer = lindexer
        else:
            idx = right
            indexer = rindexer

        # The index of the level name to be restored
        name_idx = idx.names.index(dropped_level_name)

        restore_levels = idx.levels[name_idx]
        # Inject -1 in the codes list where a join was not possible
        # IOW indexer[i]=-1
        codes = idx.codes[name_idx]
        if indexer is None:
            restore_codes = codes
        else:
            restore_codes = algos.take_nd(codes, indexer, fill_value=-1)

        # error: Cannot determine type of "__add__"
        join_levels = join_levels + [restore_levels]  # type: ignore[has-type]
        join_codes = join_codes + [restore_codes]  # type: ignore[has-type]
        join_names = join_names + [dropped_level_name]

    return join_levels, join_codes, join_names


class _OrderedMerge(_MergeOperation):
    _merge_type = "ordered_merge"

    def __init__(
        self,
        left: DataFrame | Series,
        right: DataFrame | Series,
        on: IndexLabel | None = None,
        left_on: IndexLabel | None = None,
        right_on: IndexLabel | None = None,
        left_index: bool = False,
        right_index: bool = False,
        suffixes: Suffixes = ("_x", "_y"),
        fill_method: str | None = None,
        how: JoinHow | Literal["asof"] = "outer",
    ) -> None:
        self.fill_method = fill_method
        _MergeOperation.__init__(
            self,
            left,
            right,
            on=on,
            left_on=left_on,
            left_index=left_index,
            right_index=right_index,
            right_on=right_on,
            how=how,
            suffixes=suffixes,
            sort=True,  # factorize sorts
        )

    def get_result(self, copy: bool | None = True) -> DataFrame:
        join_index, left_indexer, right_indexer = self._get_join_info()

        left_join_indexer: npt.NDArray[np.intp] | None
        right_join_indexer: npt.NDArray[np.intp] | None

        if self.fill_method == "ffill":
            if left_indexer is None:
                left_join_indexer = None
            else:
                left_join_indexer = libjoin.ffill_indexer(left_indexer)
            if right_indexer is None:
                right_join_indexer = None
            else:
                right_join_indexer = libjoin.ffill_indexer(right_indexer)
        elif self.fill_method is None:
            left_join_indexer = left_indexer
            right_join_indexer = right_indexer
        else:
            raise ValueError("fill_method must be 'ffill' or None")

        result = self._reindex_and_concat(
            join_index, left_join_indexer, right_join_indexer, copy=copy
        )
        self._maybe_add_join_keys(result, left_indexer, right_indexer)

        return result


def _asof_by_function(direction: str):
    name = f"asof_join_{direction}_on_X_by_Y"
    return getattr(libjoin, name, None)


class _AsOfMerge(_OrderedMerge):
    _merge_type = "asof_merge"

    def __init__(
        self,
        left: DataFrame | Series,
        right: DataFrame | Series,
        on: IndexLabel | None = None,
        left_on: IndexLabel | None = None,
        right_on: IndexLabel | None = None,
        left_index: bool = False,
        right_index: bool = False,
        by=None,
        left_by=None,
        right_by=None,
        suffixes: Suffixes = ("_x", "_y"),
        how: Literal["asof"] = "asof",
        tolerance=None,
        allow_exact_matches: bool = True,
        direction: str = "backward",
    ) -> None:
        self.by = by
        self.left_by = left_by
        self.right_by = right_by
        self.tolerance = tolerance
        self.allow_exact_matches = allow_exact_matches
        self.direction = direction

        # check 'direction' is valid
        if self.direction not in ["backward", "forward", "nearest"]:
            raise MergeError(f"direction invalid: {self.direction}")

        # validate allow_exact_matches
        if not is_bool(self.allow_exact_matches):
            msg = (
                "allow_exact_matches must be boolean, "
                f"passed {self.allow_exact_matches}"
            )
            raise MergeError(msg)

        _OrderedMerge.__init__(
            self,
            left,
            right,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
            how=how,
            suffixes=suffixes,
            fill_method=None,
        )

    def _validate_left_right_on(self, left_on, right_on):
        left_on, right_on = super()._validate_left_right_on(left_on, right_on)

        # we only allow on to be a single item for on
        if len(left_on) != 1 and not self.left_index:
            raise MergeError("can only asof on a key for left")

        if len(right_on) != 1 and not self.right_index:
            raise MergeError("can only asof on a key for right")

        if self.left_index and isinstance(self.left.index, MultiIndex):
            raise MergeError("left can only have one index")

        if self.right_index and isinstance(self.right.index, MultiIndex):
            raise MergeError("right can only have one index")

        # set 'by' columns
        if self.by is not None:
            if self.left_by is not None or self.right_by is not None:
                raise MergeError("Can only pass by OR left_by and right_by")
            self.left_by = self.right_by = self.by
        if self.left_by is None and self.right_by is not None:
            raise MergeError("missing left_by")
        if self.left_by is not None and self.right_by is None:
            raise MergeError("missing right_by")

        # GH#29130 Check that merge keys do not have dtype object
        if not self.left_index:
            left_on_0 = left_on[0]
            if isinstance(left_on_0, _known):
                lo_dtype = left_on_0.dtype
            else:
                lo_dtype = (
                    self.left._get_label_or_level_values(left_on_0).dtype
                    if left_on_0 in self.left.columns
                    else self.left.index.get_level_values(left_on_0)
                )
        else:
            lo_dtype = self.left.index.dtype

        if not self.right_index:
            right_on_0 = right_on[0]
            if isinstance(right_on_0, _known):
                ro_dtype = right_on_0.dtype
            else:
                ro_dtype = (
                    self.right._get_label_or_level_values(right_on_0).dtype
                    if right_on_0 in self.right.columns
                    else self.right.index.get_level_values(right_on_0)
                )
        else:
            ro_dtype = self.right.index.dtype

        if (
            is_object_dtype(lo_dtype)
            or is_object_dtype(ro_dtype)
            or is_string_dtype(lo_dtype)
            or is_string_dtype(ro_dtype)
        ):
            raise MergeError(
                f"Incompatible merge dtype, {repr(ro_dtype)} and "
                f"{repr(lo_dtype)}, both sides must have numeric dtype"
            )

        # add 'by' to our key-list so we can have it in the
        # output as a key
        if self.left_by is not None:
            if not is_list_like(self.left_by):
                self.left_by = [self.left_by]
            if not is_list_like(self.right_by):
                self.right_by = [self.right_by]

            if len(self.left_by) != len(self.right_by):
                raise MergeError("left_by and right_by must be the same length")

            left_on = self.left_by + list(left_on)
            right_on = self.right_by + list(right_on)

        return left_on, right_on

    def _maybe_require_matching_dtypes(
        self, left_join_keys: list[ArrayLike], right_join_keys: list[ArrayLike]
    ) -> None:
        # TODO: why do we do this for AsOfMerge but not the others?

        def _check_dtype_match(left: ArrayLike, right: ArrayLike, i: int):
            if left.dtype != right.dtype:
                if isinstance(left.dtype, CategoricalDtype) and isinstance(
                    right.dtype, CategoricalDtype
                ):
                    # The generic error message is confusing for categoricals.
                    #
                    # In this function, the join keys include both the original
                    # ones of the merge_asof() call, and also the keys passed
                    # to its by= argument. Unordered but equal categories
                    # are not supported for the former, but will fail
                    # later with a ValueError, so we don't *need* to check
                    # for them here.
                    msg = (
                        f"incompatible merge keys [{i}] {repr(left.dtype)} and "
                        f"{repr(right.dtype)}, both sides category, but not equal ones"
                    )
                else:
                    msg = (
                        f"incompatible merge keys [{i}] {repr(left.dtype)} and "
                        f"{repr(right.dtype)}, must be the same type"
                    )
                raise MergeError(msg)

        # validate index types are the same
        for i, (lk, rk) in enumerate(zip(left_join_keys, right_join_keys)):
            _check_dtype_match(lk, rk, i)

        if self.left_index:
            lt = self.left.index._values
        else:
            lt = left_join_keys[-1]

        if self.right_index:
            rt = self.right.index._values
        else:
            rt = right_join_keys[-1]

        _check_dtype_match(lt, rt, 0)

    def _validate_tolerance(self, left_join_keys: list[ArrayLike]) -> None:
        # validate tolerance; datetime.timedelta or Timedelta if we have a DTI
        if self.tolerance is not None:
            if self.left_index:
                lt = self.left.index._values
            else:
                lt = left_join_keys[-1]

            msg = (
                f"incompatible tolerance {self.tolerance}, must be compat "
                f"with type {repr(lt.dtype)}"
            )

            if needs_i8_conversion(lt.dtype) or (
                isinstance(lt, ArrowExtensionArray) and lt.dtype.kind in "mM"
            ):
                if not isinstance(self.tolerance, datetime.timedelta):
                    raise MergeError(msg)
                if self.tolerance < Timedelta(0):
                    raise MergeError("tolerance must be positive")

            elif is_integer_dtype(lt.dtype):
                if not is_integer(self.tolerance):
                    raise MergeError(msg)
                if self.tolerance < 0:
                    raise MergeError("tolerance must be positive")

            elif is_float_dtype(lt.dtype):
                if not is_number(self.tolerance):
                    raise MergeError(msg)
                # error: Unsupported operand types for > ("int" and "Number")
                if self.tolerance < 0:  # type: ignore[operator]
                    raise MergeError("tolerance must be positive")

            else:
                raise MergeError("key must be integer, timestamp or float")

    def _convert_values_for_libjoin(
        self, values: AnyArrayLike, side: str
    ) -> np.ndarray:
        # we require sortedness and non-null values in the join keys
        if not Index(values).is_monotonic_increasing:
            if isna(values).any():
                raise ValueError(f"Merge keys contain null values on {side} side")
            raise ValueError(f"{side} keys must be sorted")

        if isinstance(values, ArrowExtensionArray):
            values = values._maybe_convert_datelike_array()

        if needs_i8_conversion(values.dtype):
            values = values.view("i8")

        elif isinstance(values, BaseMaskedArray):
            # we've verified above that no nulls exist
            values = values._data
        elif isinstance(values, ExtensionArray):
            values = values.to_numpy()

        # error: Incompatible return value type (got "Union[ExtensionArray,
        # Any, ndarray[Any, Any], ndarray[Any, dtype[Any]], Index, Series]",
        # expected "ndarray[Any, Any]")
        return values  # type: ignore[return-value]

    def _get_join_indexers(self) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        """return the join indexers"""

        # values to compare
        left_values = (
            self.left.index._values if self.left_index else self.left_join_keys[-1]
        )
        right_values = (
            self.right.index._values if self.right_index else self.right_join_keys[-1]
        )

        # _maybe_require_matching_dtypes already checked for dtype matching
        assert left_values.dtype == right_values.dtype

        tolerance = self.tolerance
        if tolerance is not None:
            # TODO: can we reuse a tolerance-conversion function from
            #  e.g. TimedeltaIndex?
            if needs_i8_conversion(left_values.dtype) or (
                isinstance(left_values, ArrowExtensionArray)
                and left_values.dtype.kind in "mM"
            ):
                tolerance = Timedelta(tolerance)
                # TODO: we have no test cases with PeriodDtype here; probably
                #  need to adjust tolerance for that case.
                if left_values.dtype.kind in "mM":
                    # Make sure the i8 representation for tolerance
                    #  matches that for left_values/right_values.
                    if isinstance(left_values, ArrowExtensionArray):
                        unit = left_values.dtype.pyarrow_dtype.unit
                    else:
                        unit = ensure_wrapped_if_datetimelike(left_values).unit
                    tolerance = tolerance.as_unit(unit)

                tolerance = tolerance._value

        # initial type conversion as needed
        left_values = self._convert_values_for_libjoin(left_values, "left")
        right_values = self._convert_values_for_libjoin(right_values, "right")

        # a "by" parameter requires special handling
        if self.left_by is not None:
            # remove 'on' parameter from values if one existed
            if self.left_index and self.right_index:
                left_join_keys = self.left_join_keys
                right_join_keys = self.right_join_keys
            else:
                left_join_keys = self.left_join_keys[0:-1]
                right_join_keys = self.right_join_keys[0:-1]

            mapped = [
                _factorize_keys(
                    left_join_keys[n],
                    right_join_keys[n],
                    sort=False,
                )
                for n in range(len(left_join_keys))
            ]

            if len(left_join_keys) == 1:
                left_by_values = mapped[0][0]
                right_by_values = mapped[0][1]
            else:
                arrs = [np.concatenate(m[:2]) for m in mapped]
                shape = tuple(m[2] for m in mapped)
                group_index = get_group_index(
                    arrs, shape=shape, sort=False, xnull=False
                )
                left_len = len(left_join_keys[0])
                left_by_values = group_index[:left_len]
                right_by_values = group_index[left_len:]

            left_by_values = ensure_int64(left_by_values)
            right_by_values = ensure_int64(right_by_values)

            # choose appropriate function by type
            func = _asof_by_function(self.direction)
            return func(
                left_values,
                right_values,
                left_by_values,
                right_by_values,
                self.allow_exact_matches,
                tolerance,
            )
        else:
            # choose appropriate function by type
            func = _asof_by_function(self.direction)
            return func(
                left_values,
                right_values,
                None,
                None,
                self.allow_exact_matches,
                tolerance,
                False,
            )


def _get_multiindex_indexer(
    join_keys: list[ArrayLike], index: MultiIndex, sort: bool
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    # left & right join labels and num. of levels at each location
    mapped = (
        _factorize_keys(index.levels[n]._values, join_keys[n], sort=sort)
        for n in range(index.nlevels)
    )
    zipped = zip(*mapped)
    rcodes, lcodes, shape = (list(x) for x in zipped)
    if sort:
        rcodes = list(map(np.take, rcodes, index.codes))
    else:
        i8copy = lambda a: a.astype("i8", subok=False, copy=True)
        rcodes = list(map(i8copy, index.codes))

    # fix right labels if there were any nulls
    for i, join_key in enumerate(join_keys):
        mask = index.codes[i] == -1
        if mask.any():
            # check if there already was any nulls at this location
            # if there was, it is factorized to `shape[i] - 1`
            a = join_key[lcodes[i] == shape[i] - 1]
            if a.size == 0 or not a[0] != a[0]:
                shape[i] += 1

            rcodes[i][mask] = shape[i] - 1

    # get flat i8 join keys
    lkey, rkey = _get_join_keys(lcodes, rcodes, tuple(shape), sort)
    return lkey, rkey


def _get_empty_indexer() -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """Return empty join indexers."""
    return (
        np.array([], dtype=np.intp),
        np.array([], dtype=np.intp),
    )


def _get_no_sort_one_missing_indexer(
    n: int, left_missing: bool
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """
    Return join indexers where all of one side is selected without sorting
    and none of the other side is selected.

    Parameters
    ----------
    n : int
        Length of indexers to create.
    left_missing : bool
        If True, the left indexer will contain only -1's.
        If False, the right indexer will contain only -1's.

    Returns
    -------
    np.ndarray[np.intp]
        Left indexer
    np.ndarray[np.intp]
        Right indexer
    """
    idx = np.arange(n, dtype=np.intp)
    idx_missing = np.full(shape=n, fill_value=-1, dtype=np.intp)
    if left_missing:
        return idx_missing, idx
    return idx, idx_missing


def _left_join_on_index(
    left_ax: Index, right_ax: Index, join_keys: list[ArrayLike], sort: bool = False
) -> tuple[Index, npt.NDArray[np.intp] | None, npt.NDArray[np.intp]]:
    if isinstance(right_ax, MultiIndex):
        lkey, rkey = _get_multiindex_indexer(join_keys, right_ax, sort=sort)
    else:
        # error: Incompatible types in assignment (expression has type
        # "Union[Union[ExtensionArray, ndarray[Any, Any]], Index, Series]",
        # variable has type "ndarray[Any, dtype[signedinteger[Any]]]")
        lkey = join_keys[0]  # type: ignore[assignment]
        # error: Incompatible types in assignment (expression has type "Index",
        # variable has type "ndarray[Any, dtype[signedinteger[Any]]]")
        rkey = right_ax._values  # type: ignore[assignment]

    left_key, right_key, count = _factorize_keys(lkey, rkey, sort=sort)
    left_indexer, right_indexer = libjoin.left_outer_join(
        left_key, right_key, count, sort=sort
    )

    if sort or len(left_ax) != len(left_indexer):
        # if asked to sort or there are 1-to-many matches
        join_index = left_ax.take(left_indexer)
        return join_index, left_indexer, right_indexer

    # left frame preserves order & length of its index
    return left_ax, None, right_indexer


def _factorize_keys(
    lk: ArrayLike, rk: ArrayLike, sort: bool = True
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], int]:
    """
    Encode left and right keys as enumerated types.

    This is used to get the join indexers to be used when merging DataFrames.

    Parameters
    ----------
    lk : ndarray, ExtensionArray
        Left key.
    rk : ndarray, ExtensionArray
        Right key.
    sort : bool, defaults to True
        If True, the encoding is done such that the unique elements in the
        keys are sorted.

    Returns
    -------
    np.ndarray[np.intp]
        Left (resp. right if called with `key='right'`) labels, as enumerated type.
    np.ndarray[np.intp]
        Right (resp. left if called with `key='right'`) labels, as enumerated type.
    int
        Number of unique elements in union of left and right labels.

    See Also
    --------
    merge : Merge DataFrame or named Series objects
        with a database-style join.
    algorithms.factorize : Encode the object as an enumerated type
        or categorical variable.

    Examples
    --------
    >>> lk = np.array(["a", "c", "b"])
    >>> rk = np.array(["a", "c"])

    Here, the unique values are `'a', 'b', 'c'`. With the default
    `sort=True`, the encoding will be `{0: 'a', 1: 'b', 2: 'c'}`:

    >>> pd.core.reshape.merge._factorize_keys(lk, rk)
    (array([0, 2, 1]), array([0, 2]), 3)

    With the `sort=False`, the encoding will correspond to the order
    in which the unique elements first appear: `{0: 'a', 1: 'c', 2: 'b'}`:

    >>> pd.core.reshape.merge._factorize_keys(lk, rk, sort=False)
    (array([0, 1, 2]), array([0, 1]), 3)
    """
    # TODO: if either is a RangeIndex, we can likely factorize more efficiently?

    if (
        isinstance(lk.dtype, DatetimeTZDtype) and isinstance(rk.dtype, DatetimeTZDtype)
    ) or (lib.is_np_dtype(lk.dtype, "M") and lib.is_np_dtype(rk.dtype, "M")):
        # Extract the ndarray (UTC-localized) values
        # Note: we dont need the dtypes to match, as these can still be compared
        lk, rk = cast("DatetimeArray", lk)._ensure_matching_resos(rk)
        lk = cast("DatetimeArray", lk)._ndarray
        rk = cast("DatetimeArray", rk)._ndarray

    elif (
        isinstance(lk.dtype, CategoricalDtype)
        and isinstance(rk.dtype, CategoricalDtype)
        and lk.dtype == rk.dtype
    ):
        assert isinstance(lk, Categorical)
        assert isinstance(rk, Categorical)
        # Cast rk to encoding so we can compare codes with lk

        rk = lk._encode_with_my_categories(rk)

        lk = ensure_int64(lk.codes)
        rk = ensure_int64(rk.codes)

    elif isinstance(lk, ExtensionArray) and lk.dtype == rk.dtype:
        if (isinstance(lk.dtype, ArrowDtype) and is_string_dtype(lk.dtype)) or (
            isinstance(lk.dtype, StringDtype) and lk.dtype.storage == "pyarrow"
        ):
            import pyarrow as pa
            import pyarrow.compute as pc

            len_lk = len(lk)
            lk = lk._pa_array  # type: ignore[attr-defined]
            rk = rk._pa_array  # type: ignore[union-attr]
            dc = (
                pa.chunked_array(lk.chunks + rk.chunks)  # type: ignore[union-attr]
                .combine_chunks()
                .dictionary_encode()
            )

            llab, rlab, count = (
                pc.fill_null(dc.indices[slice(len_lk)], -1)
                .to_numpy()
                .astype(np.intp, copy=False),
                pc.fill_null(dc.indices[slice(len_lk, None)], -1)
                .to_numpy()
                .astype(np.intp, copy=False),
                len(dc.dictionary),
            )

            if sort:
                uniques = dc.dictionary.to_numpy(zero_copy_only=False)
                llab, rlab = _sort_labels(uniques, llab, rlab)

            if dc.null_count > 0:
                lmask = llab == -1
                lany = lmask.any()
                rmask = rlab == -1
                rany = rmask.any()
                if lany:
                    np.putmask(llab, lmask, count)
                if rany:
                    np.putmask(rlab, rmask, count)
                count += 1
            return llab, rlab, count

        if not isinstance(lk, BaseMaskedArray) and not (
            # exclude arrow dtypes that would get cast to object
            isinstance(lk.dtype, ArrowDtype)
            and (
                is_numeric_dtype(lk.dtype.numpy_dtype)
                or is_string_dtype(lk.dtype)
                and not sort
            )
        ):
            lk, _ = lk._values_for_factorize()

            # error: Item "ndarray" of "Union[Any, ndarray]" has no attribute
            # "_values_for_factorize"
            rk, _ = rk._values_for_factorize()  # type: ignore[union-attr]

    if needs_i8_conversion(lk.dtype) and lk.dtype == rk.dtype:
        # GH#23917 TODO: Needs tests for non-matching dtypes
        # GH#23917 TODO: needs tests for case where lk is integer-dtype
        #  and rk is datetime-dtype
        lk = np.asarray(lk, dtype=np.int64)
        rk = np.asarray(rk, dtype=np.int64)

    klass, lk, rk = _convert_arrays_and_get_rizer_klass(lk, rk)

    rizer = klass(max(len(lk), len(rk)))

    if isinstance(lk, BaseMaskedArray):
        assert isinstance(rk, BaseMaskedArray)
        llab = rizer.factorize(lk._data, mask=lk._mask)
        rlab = rizer.factorize(rk._data, mask=rk._mask)
    elif isinstance(lk, ArrowExtensionArray):
        assert isinstance(rk, ArrowExtensionArray)
        # we can only get here with numeric dtypes
        # TODO: Remove when we have a Factorizer for Arrow
        llab = rizer.factorize(
            lk.to_numpy(na_value=1, dtype=lk.dtype.numpy_dtype), mask=lk.isna()
        )
        rlab = rizer.factorize(
            rk.to_numpy(na_value=1, dtype=lk.dtype.numpy_dtype), mask=rk.isna()
        )
    else:
        # Argument 1 to "factorize" of "ObjectFactorizer" has incompatible type
        # "Union[ndarray[Any, dtype[signedinteger[_64Bit]]],
        # ndarray[Any, dtype[object_]]]"; expected "ndarray[Any, dtype[object_]]"
        llab = rizer.factorize(lk)  # type: ignore[arg-type]
        rlab = rizer.factorize(rk)  # type: ignore[arg-type]
    assert llab.dtype == np.dtype(np.intp), llab.dtype
    assert rlab.dtype == np.dtype(np.intp), rlab.dtype

    count = rizer.get_count()

    if sort:
        uniques = rizer.uniques.to_array()
        llab, rlab = _sort_labels(uniques, llab, rlab)

    # NA group
    lmask = llab == -1
    lany = lmask.any()
    rmask = rlab == -1
    rany = rmask.any()

    if lany or rany:
        if lany:
            np.putmask(llab, lmask, count)
        if rany:
            np.putmask(rlab, rmask, count)
        count += 1

    return llab, rlab, count


def _convert_arrays_and_get_rizer_klass(
    lk: ArrayLike, rk: ArrayLike
) -> tuple[type[libhashtable.Factorizer], ArrayLike, ArrayLike]:
    klass: type[libhashtable.Factorizer]
    if is_numeric_dtype(lk.dtype):
        if lk.dtype != rk.dtype:
            dtype = find_common_type([lk.dtype, rk.dtype])
            if isinstance(dtype, ExtensionDtype):
                cls = dtype.construct_array_type()
                if not isinstance(lk, ExtensionArray):
                    lk = cls._from_sequence(lk, dtype=dtype, copy=False)
                else:
                    lk = lk.astype(dtype, copy=False)

                if not isinstance(rk, ExtensionArray):
                    rk = cls._from_sequence(rk, dtype=dtype, copy=False)
                else:
                    rk = rk.astype(dtype, copy=False)
            else:
                lk = lk.astype(dtype, copy=False)
                rk = rk.astype(dtype, copy=False)
        if isinstance(lk, BaseMaskedArray):
            #  Invalid index type "type" for "Dict[Type[object], Type[Factorizer]]";
            #  expected type "Type[object]"
            klass = _factorizers[lk.dtype.type]  # type: ignore[index]
        elif isinstance(lk.dtype, ArrowDtype):
            klass = _factorizers[lk.dtype.numpy_dtype.type]
        else:
            klass = _factorizers[lk.dtype.type]

    else:
        klass = libhashtable.ObjectFactorizer
        lk = ensure_object(lk)
        rk = ensure_object(rk)
    return klass, lk, rk


def _sort_labels(
    uniques: np.ndarray, left: npt.NDArray[np.intp], right: npt.NDArray[np.intp]
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    llength = len(left)
    labels = np.concatenate([left, right])

    _, new_labels = algos.safe_sort(uniques, labels, use_na_sentinel=True)
    new_left, new_right = new_labels[:llength], new_labels[llength:]

    return new_left, new_right


def _get_join_keys(
    llab: list[npt.NDArray[np.int64 | np.intp]],
    rlab: list[npt.NDArray[np.int64 | np.intp]],
    shape: Shape,
    sort: bool,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    # how many levels can be done without overflow
    nlev = next(
        lev
        for lev in range(len(shape), 0, -1)
        if not is_int64_overflow_possible(shape[:lev])
    )

    # get keys for the first `nlev` levels
    stride = np.prod(shape[1:nlev], dtype="i8")
    lkey = stride * llab[0].astype("i8", subok=False, copy=False)
    rkey = stride * rlab[0].astype("i8", subok=False, copy=False)

    for i in range(1, nlev):
        with np.errstate(divide="ignore"):
            stride //= shape[i]
        lkey += llab[i] * stride
        rkey += rlab[i] * stride

    if nlev == len(shape):  # all done!
        return lkey, rkey

    # densify current keys to avoid overflow
    lkey, rkey, count = _factorize_keys(lkey, rkey, sort=sort)

    llab = [lkey] + llab[nlev:]
    rlab = [rkey] + rlab[nlev:]
    shape = (count,) + shape[nlev:]

    return _get_join_keys(llab, rlab, shape, sort)


def _should_fill(lname, rname) -> bool:
    if not isinstance(lname, str) or not isinstance(rname, str):
        return True
    return lname == rname


def _any(x) -> bool:
    return x is not None and com.any_not_none(*x)


def _validate_operand(obj: DataFrame | Series) -> DataFrame:
    if isinstance(obj, ABCDataFrame):
        return obj
    elif isinstance(obj, ABCSeries):
        if obj.name is None:
            raise ValueError("Cannot merge a Series without a name")
        return obj.to_frame()
    else:
        raise TypeError(
            f"Can only merge Series or DataFrame objects, a {type(obj)} was passed"
        )


def _items_overlap_with_suffix(
    left: Index, right: Index, suffixes: Suffixes
) -> tuple[Index, Index]:
    """
    Suffixes type validation.

    If two indices overlap, add suffixes to overlapping entries.

    If corresponding suffix is empty, the entry is simply converted to string.

    """
    if not is_list_like(suffixes, allow_sets=False) or isinstance(suffixes, dict):
        raise TypeError(
            f"Passing 'suffixes' as a {type(suffixes)}, is not supported. "
            "Provide 'suffixes' as a tuple instead."
        )

    to_rename = left.intersection(right)
    if len(to_rename) == 0:
        return left, right

    lsuffix, rsuffix = suffixes

    if not lsuffix and not rsuffix:
        raise ValueError(f"columns overlap but no suffix specified: {to_rename}")

    def renamer(x, suffix: str | None):
        """
        Rename the left and right indices.

        If there is overlap, and suffix is not None, add
        suffix, otherwise, leave it as-is.

        Parameters
        ----------
        x : original column name
        suffix : str or None

        Returns
        -------
        x : renamed column name
        """
        if x in to_rename and suffix is not None:
            return f"{x}{suffix}"
        return x

    lrenamer = partial(renamer, suffix=lsuffix)
    rrenamer = partial(renamer, suffix=rsuffix)

    llabels = left._transform_index(lrenamer)
    rlabels = right._transform_index(rrenamer)

    dups = []
    if not llabels.is_unique:
        # Only warn when duplicates are caused because of suffixes, already duplicated
        # columns in origin should not warn
        dups = llabels[(llabels.duplicated()) & (~left.duplicated())].tolist()
    if not rlabels.is_unique:
        dups.extend(rlabels[(rlabels.duplicated()) & (~right.duplicated())].tolist())
    if dups:
        raise MergeError(
            f"Passing 'suffixes' which cause duplicate columns {set(dups)} is "
            f"not allowed.",
        )

    return llabels, rlabels
<|endoftext|>"""
Quantilization functions and related stuff
"""
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
)

import numpy as np

from pandas._libs import (
    Timedelta,
    Timestamp,
    lib,
)

from pandas.core.dtypes.common import (
    ensure_platform_int,
    is_bool_dtype,
    is_integer,
    is_list_like,
    is_numeric_dtype,
    is_scalar,
)
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    DatetimeTZDtype,
    ExtensionDtype,
)
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.dtypes.missing import isna

from pandas import (
    Categorical,
    Index,
    IntervalIndex,
)
import pandas.core.algorithms as algos
from pandas.core.arrays.datetimelike import dtype_to_unit

if TYPE_CHECKING:
    from pandas._typing import (
        DtypeObj,
        IntervalLeftRight,
    )


def cut(
    x,
    bins,
    right: bool = True,
    labels=None,
    retbins: bool = False,
    precision: int = 3,
    include_lowest: bool = False,
    duplicates: str = "raise",
    ordered: bool = True,
):
    """
    Bin values into discrete intervals.

    Use `cut` when you need to segment and sort data values into bins. This
    function is also useful for going from a continuous variable to a
    categorical variable. For example, `cut` could convert ages to groups of
    age ranges. Supports binning into an equal number of bins, or a
    pre-specified array of bins.

    Parameters
    ----------
    x : array-like
        The input array to be binned. Must be 1-dimensional.
    bins : int, sequence of scalars, or IntervalIndex
        The criteria to bin by.

        * int : Defines the number of equal-width bins in the range of `x`. The
          range of `x` is extended by .1% on each side to include the minimum
          and maximum values of `x`.
        * sequence of scalars : Defines the bin edges allowing for non-uniform
          width. No extension of the range of `x` is done.
        * IntervalIndex : Defines the exact bins to be used. Note that
          IntervalIndex for `bins` must be non-overlapping.

    right : bool, default True
        Indicates whether `bins` includes the rightmost edge or not. If
        ``right == True`` (the default), then the `bins` ``[1, 2, 3, 4]``
        indicate (1,2], (2,3], (3,4]. This argument is ignored when
        `bins` is an IntervalIndex.
    labels : array or False, default None
        Specifies the labels for the returned bins. Must be the same length as
        the resulting bins. If False, returns only integer indicators of the
        bins. This affects the type of the output container (see below).
        This argument is ignored when `bins` is an IntervalIndex. If True,
        raises an error. When `ordered=False`, labels must be provided.
    retbins : bool, default False
        Whether to return the bins or not. Useful when bins is provided
        as a scalar.
    precision : int, default 3
        The precision at which to store and display the bins labels.
    include_lowest : bool, default False
        Whether the first interval should be left-inclusive or not.
    duplicates : {default 'raise', 'drop'}, optional
        If bin edges are not unique, raise ValueError or drop non-uniques.
    ordered : bool, default True
        Whether the labels are ordered or not. Applies to returned types
        Categorical and Series (with Categorical dtype). If True,
        the resulting categorical will be ordered. If False, the resulting
        categorical will be unordered (labels must be provided).

    Returns
    -------
    out : Categorical, Series, or ndarray
        An array-like object representing the respective bin for each value
        of `x`. The type depends on the value of `labels`.

        * None (default) : returns a Series for Series `x` or a
          Categorical for all other inputs. The values stored within
          are Interval dtype.

        * sequence of scalars : returns a Series for Series `x` or a
          Categorical for all other inputs. The values stored within
          are whatever the type in the sequence is.

        * False : returns an ndarray of integers.

    bins : numpy.ndarray or IntervalIndex.
        The computed or specified bins. Only returned when `retbins=True`.
        For scalar or sequence `bins`, this is an ndarray with the computed
        bins. If set `duplicates=drop`, `bins` will drop non-unique bin. For
        an IntervalIndex `bins`, this is equal to `bins`.

    See Also
    --------
    qcut : Discretize variable into equal-sized buckets based on rank
        or based on sample quantiles.
    Categorical : Array type for storing data that come from a
        fixed set of values.
    Series : One-dimensional array with axis labels (including time series).
    IntervalIndex : Immutable Index implementing an ordered, sliceable set.

    Notes
    -----
    Any NA values will be NA in the result. Out of bounds values will be NA in
    the resulting Series or Categorical object.

    Reference :ref:`the user guide <reshaping.tile.cut>` for more examples.

    Examples
    --------
    Discretize into three equal-sized bins.

    >>> pd.cut(np.array([1, 7, 5, 4, 6, 3]), 3)
    ... # doctest: +ELLIPSIS
    [(0.994, 3.0], (5.0, 7.0], (3.0, 5.0], (3.0, 5.0], (5.0, 7.0], ...
    Categories (3, interval[float64, right]): [(0.994, 3.0] < (3.0, 5.0] ...

    >>> pd.cut(np.array([1, 7, 5, 4, 6, 3]), 3, retbins=True)
    ... # doctest: +ELLIPSIS
    ([(0.994, 3.0], (5.0, 7.0], (3.0, 5.0], (3.0, 5.0], (5.0, 7.0], ...
    Categories (3, interval[float64, right]): [(0.994, 3.0] < (3.0, 5.0] ...
    array([0.994, 3.   , 5.   , 7.   ]))

    Discovers the same bins, but assign them specific labels. Notice that
    the returned Categorical's categories are `labels` and is ordered.

    >>> pd.cut(np.array([1, 7, 5, 4, 6, 3]),
    ...        3, labels=["bad", "medium", "good"])
    ['bad', 'good', 'medium', 'medium', 'good', 'bad']
    Categories (3, object): ['bad' < 'medium' < 'good']

    ``ordered=False`` will result in unordered categories when labels are passed.
    This parameter can be used to allow non-unique labels:

    >>> pd.cut(np.array([1, 7, 5, 4, 6, 3]), 3,
    ...        labels=["B", "A", "B"], ordered=False)
    ['B', 'B', 'A', 'A', 'B', 'B']
    Categories (2, object): ['A', 'B']

    ``labels=False`` implies you just want the bins back.

    >>> pd.cut([0, 1, 1, 2], bins=4, labels=False)
    array([0, 1, 1, 3])

    Passing a Series as an input returns a Series with categorical dtype:

    >>> s = pd.Series(np.array([2, 4, 6, 8, 10]),
    ...               index=['a', 'b', 'c', 'd', 'e'])
    >>> pd.cut(s, 3)
    ... # doctest: +ELLIPSIS
    a    (1.992, 4.667]
    b    (1.992, 4.667]
    c    (4.667, 7.333]
    d     (7.333, 10.0]
    e     (7.333, 10.0]
    dtype: category
    Categories (3, interval[float64, right]): [(1.992, 4.667] < (4.667, ...

    Passing a Series as an input returns a Series with mapping value.
    It is used to map numerically to intervals based on bins.

    >>> s = pd.Series(np.array([2, 4, 6, 8, 10]),
    ...               index=['a', 'b', 'c', 'd', 'e'])
    >>> pd.cut(s, [0, 2, 4, 6, 8, 10], labels=False, retbins=True, right=False)
    ... # doctest: +ELLIPSIS
    (a    1.0
     b    2.0
     c    3.0
     d    4.0
     e    NaN
     dtype: float64,
     array([ 0,  2,  4,  6,  8, 10]))

    Use `drop` optional when bins is not unique

    >>> pd.cut(s, [0, 2, 4, 6, 10, 10], labels=False, retbins=True,
    ...        right=False, duplicates='drop')
    ... # doctest: +ELLIPSIS
    (a    1.0
     b    2.0
     c    3.0
     d    3.0
     e    NaN
     dtype: float64,
     array([ 0,  2,  4,  6, 10]))

    Passing an IntervalIndex for `bins` results in those categories exactly.
    Notice that values not covered by the IntervalIndex are set to NaN. 0
    is to the left of the first bin (which is closed on the right), and 1.5
    falls between two bins.

    >>> bins = pd.IntervalIndex.from_tuples([(0, 1), (2, 3), (4, 5)])
    >>> pd.cut([0, 0.5, 1.5, 2.5, 4.5], bins)
    [NaN, (0.0, 1.0], NaN, (2.0, 3.0], (4.0, 5.0]]
    Categories (3, interval[int64, right]): [(0, 1] < (2, 3] < (4, 5]]
    """
    # NOTE: this binning code is changed a bit from histogram for var(x) == 0

    original = x
    x_idx = _preprocess_for_cut(x)
    x_idx, _ = _coerce_to_type(x_idx)

    if not np.iterable(bins):
        bins = _nbins_to_bins(x_idx, bins, right)

    elif isinstance(bins, IntervalIndex):
        if bins.is_overlapping:
            raise ValueError("Overlapping IntervalIndex is not accepted.")

    else:
        bins = Index(bins)
        if not bins.is_monotonic_increasing:
            raise ValueError("bins must increase monotonically.")

    fac, bins = _bins_to_cuts(
        x_idx,
        bins,
        right=right,
        labels=labels,
        precision=precision,
        include_lowest=include_lowest,
        duplicates=duplicates,
        ordered=ordered,
    )

    return _postprocess_for_cut(fac, bins, retbins, original)


def qcut(
    x,
    q,
    labels=None,
    retbins: bool = False,
    precision: int = 3,
    duplicates: str = "raise",
):
    """
    Quantile-based discretization function.

    Discretize variable into equal-sized buckets based on rank or based
    on sample quantiles. For example 1000 values for 10 quantiles would
    produce a Categorical object indicating quantile membership for each data point.

    Parameters
    ----------
    x : 1d ndarray or Series
    q : int or list-like of float
        Number of quantiles. 10 for deciles, 4 for quartiles, etc. Alternately
        array of quantiles, e.g. [0, .25, .5, .75, 1.] for quartiles.
    labels : array or False, default None
        Used as labels for the resulting bins. Must be of the same length as
        the resulting bins. If False, return only integer indicators of the
        bins. If True, raises an error.
    retbins : bool, optional
        Whether to return the (bins, labels) or not. Can be useful if bins
        is given as a scalar.
    precision : int, optional
        The precision at which to store and display the bins labels.
    duplicates : {default 'raise', 'drop'}, optional
        If bin edges are not unique, raise ValueError or drop non-uniques.

    Returns
    -------
    out : Categorical or Series or array of integers if labels is False
        The return type (Categorical or Series) depends on the input: a Series
        of type category if input is a Series else Categorical. Bins are
        represented as categories when categorical data is returned.
    bins : ndarray of floats
        Returned only if `retbins` is True.

    Notes
    -----
    Out of bounds values will be NA in the resulting Categorical object

    Examples
    --------
    >>> pd.qcut(range(5), 4)
    ... # doctest: +ELLIPSIS
    [(-0.001, 1.0], (-0.001, 1.0], (1.0, 2.0], (2.0, 3.0], (3.0, 4.0]]
    Categories (4, interval[float64, right]): [(-0.001, 1.0] < (1.0, 2.0] ...

    >>> pd.qcut(range(5), 3, labels=["good", "medium", "bad"])
    ... # doctest: +SKIP
    [good, good, medium, bad, bad]
    Categories (3, object): [good < medium < bad]

    >>> pd.qcut(range(5), 4, labels=False)
    array([0, 0, 1, 2, 3])
    """
    original = x
    x_idx = _preprocess_for_cut(x)
    x_idx, _ = _coerce_to_type(x_idx)

    quantiles = np.linspace(0, 1, q + 1) if is_integer(q) else q

    bins = x_idx.to_series().dropna().quantile(quantiles)

    fac, bins = _bins_to_cuts(
        x_idx,
        Index(bins),
        labels=labels,
        precision=precision,
        include_lowest=True,
        duplicates=duplicates,
    )

    return _postprocess_for_cut(fac, bins, retbins, original)


def _nbins_to_bins(x_idx: Index, nbins: int, right: bool) -> Index:
    """
    If a user passed an integer N for bins, convert this to a sequence of N
    equal(ish)-sized bins.
    """
    if is_scalar(nbins) and nbins < 1:
        raise ValueError("`bins` should be a positive integer.")

    if x_idx.size == 0:
        raise ValueError("Cannot cut empty array")

    rng = (x_idx.min(), x_idx.max())
    mn, mx = rng

    if is_numeric_dtype(x_idx.dtype) and (np.isinf(mn) or np.isinf(mx)):
        # GH#24314
        raise ValueError(
            "cannot specify integer `bins` when input data contains infinity"
        )

    if mn == mx:  # adjust end points before binning
        if _is_dt_or_td(x_idx.dtype):
            # using seconds=1 is pretty arbitrary here
            # error: Argument 1 to "dtype_to_unit" has incompatible type
            # "dtype[Any] | ExtensionDtype"; expected "DatetimeTZDtype | dtype[Any]"
            unit = dtype_to_unit(x_idx.dtype)  # type: ignore[arg-type]
            td = Timedelta(seconds=1).as_unit(unit)
            # Use DatetimeArray/TimedeltaArray method instead of linspace
            # error: Item "ExtensionArray" of "ExtensionArray | ndarray[Any, Any]"
            # has no attribute "_generate_range"
            bins = x_idx._values._generate_range(  # type: ignore[union-attr]
                start=mn - td, end=mx + td, periods=nbins + 1, freq=None, unit=unit
            )
        else:
            mn -= 0.001 * abs(mn) if mn != 0 else 0.001
            mx += 0.001 * abs(mx) if mx != 0 else 0.001

            bins = np.linspace(mn, mx, nbins + 1, endpoint=True)
    else:  # adjust end points after binning
        if _is_dt_or_td(x_idx.dtype):
            # Use DatetimeArray/TimedeltaArray method instead of linspace

            # error: Argument 1 to "dtype_to_unit" has incompatible type
            # "dtype[Any] | ExtensionDtype"; expected "DatetimeTZDtype | dtype[Any]"
            unit = dtype_to_unit(x_idx.dtype)  # type: ignore[arg-type]
            # error: Item "ExtensionArray" of "ExtensionArray | ndarray[Any, Any]"
            # has no attribute "_generate_range"
            bins = x_idx._values._generate_range(  # type: ignore[union-attr]
                start=mn, end=mx, periods=nbins + 1, freq=None, unit=unit
            )
        else:
            bins = np.linspace(mn, mx, nbins + 1, endpoint=True)
        adj = (mx - mn) * 0.001  # 0.1% of the range
        if right:
            bins[0] -= adj
        else:
            bins[-1] += adj

    return Index(bins)


def _bins_to_cuts(
    x_idx: Index,
    bins: Index,
    right: bool = True,
    labels=None,
    precision: int = 3,
    include_lowest: bool = False,
    duplicates: str = "raise",
    ordered: bool = True,
):
    if not ordered and labels is None:
        raise ValueError("'labels' must be provided if 'ordered = False'")

    if duplicates not in ["raise", "drop"]:
        raise ValueError(
            "invalid value for 'duplicates' parameter, valid options are: raise, drop"
        )

    result: Categorical | np.ndarray

    if isinstance(bins, IntervalIndex):
        # we have a fast-path here
        ids = bins.get_indexer(x_idx)
        cat_dtype = CategoricalDtype(bins, ordered=True)
        result = Categorical.from_codes(ids, dtype=cat_dtype, validate=False)
        return result, bins

    unique_bins = algos.unique(bins)
    if len(unique_bins) < len(bins) and len(bins) != 2:
        if duplicates == "raise":
            raise ValueError(
                f"Bin edges must be unique: {repr(bins)}.\n"
                f"You can drop duplicate edges by setting the 'duplicates' kwarg"
            )
        bins = unique_bins

    side: Literal["left", "right"] = "left" if right else "right"

    try:
        ids = bins.searchsorted(x_idx, side=side)
    except TypeError as err:
        # e.g. test_datetime_nan_error if bins are DatetimeArray and x_idx
        #  is integers
        if x_idx.dtype.kind == "m":
            raise ValueError("bins must be of timedelta64 dtype") from err
        elif x_idx.dtype.kind == bins.dtype.kind == "M":
            raise ValueError(
                "Cannot use timezone-naive bins with timezone-aware values, "
                "or vice-versa"
            ) from err
        elif x_idx.dtype.kind == "M":
            raise ValueError("bins must be of datetime64 dtype") from err
        else:
            raise
    ids = ensure_platform_int(ids)

    if include_lowest:
        ids[x_idx == bins[0]] = 1

    na_mask = isna(x_idx) | (ids == len(bins)) | (ids == 0)
    has_nas = na_mask.any()

    if labels is not False:
        if not (labels is None or is_list_like(labels)):
            raise ValueError(
                "Bin labels must either be False, None or passed in as a "
                "list-like argument"
            )

        if labels is None:
            labels = _format_labels(
                bins, precision, right=right, include_lowest=include_lowest
            )
        elif ordered and len(set(labels)) != len(labels):
            raise ValueError(
                "labels must be unique if ordered=True; pass ordered=False "
                "for duplicate labels"
            )
        else:
            if len(labels) != len(bins) - 1:
                raise ValueError(
                    "Bin labels must be one fewer than the number of bin edges"
                )

        if not isinstance(getattr(labels, "dtype", None), CategoricalDtype):
            labels = Categorical(
                labels,
                categories=labels if len(set(labels)) == len(labels) else None,
                ordered=ordered,
            )
        # TODO: handle mismatch between categorical label order and pandas.cut order.
        np.putmask(ids, na_mask, 0)
        result = algos.take_nd(labels, ids - 1)

    else:
        result = ids - 1
        if has_nas:
            result = result.astype(np.float64)
            np.putmask(result, na_mask, np.nan)

    return result, bins


def _coerce_to_type(x: Index) -> tuple[Index, DtypeObj | None]:
    """
    if the passed data is of datetime/timedelta, bool or nullable int type,
    this method converts it to numeric so that cut or qcut method can
    handle it
    """
    dtype: DtypeObj | None = None

    if _is_dt_or_td(x.dtype):
        dtype = x.dtype
    elif is_bool_dtype(x.dtype):
        # GH 20303
        x = x.astype(np.int64)
    # To support cut and qcut for IntegerArray we convert to float dtype.
    # Will properly support in the future.
    # https://github.com/pandas-dev/pandas/pull/31290
    # https://github.com/pandas-dev/pandas/issues/31389
    elif isinstance(x.dtype, ExtensionDtype) and is_numeric_dtype(x.dtype):
        x_arr = x.to_numpy(dtype=np.float64, na_value=np.nan)
        x = Index(x_arr)

    return Index(x), dtype


def _is_dt_or_td(dtype: DtypeObj) -> bool:
    # Note: the dtype here comes from an Index.dtype, so we know that that any
    #  dt64/td64 dtype is of a supported unit.
    return isinstance(dtype, DatetimeTZDtype) or lib.is_np_dtype(dtype, "mM")


def _format_labels(
    bins: Index,
    precision: int,
    right: bool = True,
    include_lowest: bool = False,
):
    """based on the dtype, return our labels"""
    closed: IntervalLeftRight = "right" if right else "left"

    formatter: Callable[[Any], Timestamp] | Callable[[Any], Timedelta]

    if _is_dt_or_td(bins.dtype):
        # error: Argument 1 to "dtype_to_unit" has incompatible type
        # "dtype[Any] | ExtensionDtype"; expected "DatetimeTZDtype | dtype[Any]"
        unit = dtype_to_unit(bins.dtype)  # type: ignore[arg-type]
        formatter = lambda x: x
        adjust = lambda x: x - Timedelta(1, unit=unit).as_unit(unit)
    else:
        precision = _infer_precision(precision, bins)
        formatter = lambda x: _round_frac(x, precision)
        adjust = lambda x: x - 10 ** (-precision)

    breaks = [formatter(b) for b in bins]
    if right and include_lowest:
        # adjust lhs of first interval by precision to account for being right closed
        breaks[0] = adjust(breaks[0])

    if _is_dt_or_td(bins.dtype):
        # error: "Index" has no attribute "as_unit"
        breaks = type(bins)(breaks).as_unit(unit)  # type: ignore[attr-defined]

    return IntervalIndex.from_breaks(breaks, closed=closed)


def _preprocess_for_cut(x) -> Index:
    """
    handles preprocessing for cut where we convert passed
    input to array, strip the index information and store it
    separately
    """
    # Check that the passed array is a Pandas or Numpy object
    # We don't want to strip away a Pandas data-type here (e.g. datetimetz)
    ndim = getattr(x, "ndim", None)
    if ndim is None:
        x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("Input array must be 1 dimensional")

    return Index(x)


def _postprocess_for_cut(fac, bins, retbins: bool, original):
    """
    handles post processing for the cut method where
    we combine the index information if the originally passed
    datatype was a series
    """
    if isinstance(original, ABCSeries):
        fac = original._constructor(fac, index=original.index, name=original.name)

    if not retbins:
        return fac

    if isinstance(bins, Index) and is_numeric_dtype(bins.dtype):
        bins = bins._values

    return fac, bins


def _round_frac(x, precision: int):
    """
    Round the fractional part of the given number
    """
    if not np.isfinite(x) or x == 0:
        return x
    else:
        frac, whole = np.modf(x)
        if whole == 0:
            digits = -int(np.floor(np.log10(abs(frac)))) - 1 + precision
        else:
            digits = precision
        return np.around(x, digits)


def _infer_precision(base_precision: int, bins: Index) -> int:
    """
    Infer an appropriate precision for _round_frac
    """
    for precision in range(base_precision, 20):
        levels = np.asarray([_round_frac(b, precision) for b in bins])
        if algos.unique(levels).size == bins.size:
            return precision
    return base_precision  # default
<|endoftext|>from __future__ import annotations

import itertools
from typing import (
    TYPE_CHECKING,
    cast,
)
import warnings

import numpy as np

import pandas._libs.reshape as libreshape
from pandas.errors import PerformanceWarning
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.cast import (
    find_common_type,
    maybe_promote,
)
from pandas.core.dtypes.common import (
    ensure_platform_int,
    is_1d_only_ea_dtype,
    is_integer,
    needs_i8_conversion,
)
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.missing import notna

import pandas.core.algorithms as algos
from pandas.core.algorithms import (
    factorize,
    unique,
)
from pandas.core.arrays.categorical import factorize_from_iterable
from pandas.core.construction import ensure_wrapped_if_datetimelike
from pandas.core.frame import DataFrame
from pandas.core.indexes.api import (
    Index,
    MultiIndex,
    RangeIndex,
)
from pandas.core.reshape.concat import concat
from pandas.core.series import Series
from pandas.core.sorting import (
    compress_group_index,
    decons_obs_group_ids,
    get_compressed_ids,
    get_group_index,
    get_group_index_sorter,
)

if TYPE_CHECKING:
    from pandas._typing import (
        ArrayLike,
        Level,
        npt,
    )

    from pandas.core.arrays import ExtensionArray
    from pandas.core.indexes.frozen import FrozenList


class _Unstacker:
    """
    Helper class to unstack data / pivot with multi-level index

    Parameters
    ----------
    index : MultiIndex
    level : int or str, default last level
        Level to "unstack". Accepts a name for the level.
    fill_value : scalar, optional
        Default value to fill in missing values if subgroups do not have the
        same set of labels. By default, missing values will be replaced with
        the default fill value for that data type, NaN for float, NaT for
        datetimelike, etc. For integer types, by default data will converted to
        float and missing values will be set to NaN.
    constructor : object
        Pandas ``DataFrame`` or subclass used to create unstacked
        response.  If None, DataFrame will be used.

    Examples
    --------
    >>> index = pd.MultiIndex.from_tuples([('one', 'a'), ('one', 'b'),
    ...                                    ('two', 'a'), ('two', 'b')])
    >>> s = pd.Series(np.arange(1, 5, dtype=np.int64), index=index)
    >>> s
    one  a    1
         b    2
    two  a    3
         b    4
    dtype: int64

    >>> s.unstack(level=-1)
         a  b
    one  1  2
    two  3  4

    >>> s.unstack(level=0)
       one  two
    a    1    3
    b    2    4

    Returns
    -------
    unstacked : DataFrame
    """

    def __init__(
        self, index: MultiIndex, level: Level, constructor, sort: bool = True
    ) -> None:
        self.constructor = constructor
        self.sort = sort

        self.index = index.remove_unused_levels()

        self.level = self.index._get_level_number(level)

        # when index includes `nan`, need to lift levels/strides by 1
        self.lift = 1 if -1 in self.index.codes[self.level] else 0

        # Note: the "pop" below alters these in-place.
        self.new_index_levels = list(self.index.levels)
        self.new_index_names = list(self.index.names)

        self.removed_name = self.new_index_names.pop(self.level)
        self.removed_level = self.new_index_levels.pop(self.level)
        self.removed_level_full = index.levels[self.level]
        if not self.sort:
            unique_codes = unique(self.index.codes[self.level])
            self.removed_level = self.removed_level.take(unique_codes)
            self.removed_level_full = self.removed_level_full.take(unique_codes)

        # Bug fix GH 20601
        # If the data frame is too big, the number of unique index combination
        # will cause int32 overflow on windows environments.
        # We want to check and raise an warning before this happens
        num_rows = np.max([index_level.size for index_level in self.new_index_levels])
        num_columns = self.removed_level.size

        # GH20601: This forces an overflow if the number of cells is too high.
        num_cells = num_rows * num_columns

        # GH 26314: Previous ValueError raised was too restrictive for many users.
        if num_cells > np.iinfo(np.int32).max:
            warnings.warn(
                f"The following operation may generate {num_cells} cells "
                f"in the resulting pandas object.",
                PerformanceWarning,
                stacklevel=find_stack_level(),
            )

        self._make_selectors()

    @cache_readonly
    def _indexer_and_to_sort(
        self,
    ) -> tuple[
        npt.NDArray[np.intp],
        list[np.ndarray],  # each has _some_ signed integer dtype
    ]:
        v = self.level

        codes = list(self.index.codes)
        levs = list(self.index.levels)
        to_sort = codes[:v] + codes[v + 1 :] + [codes[v]]
        sizes = tuple(len(x) for x in levs[:v] + levs[v + 1 :] + [levs[v]])

        comp_index, obs_ids = get_compressed_ids(to_sort, sizes)
        ngroups = len(obs_ids)

        indexer = get_group_index_sorter(comp_index, ngroups)
        return indexer, to_sort

    @cache_readonly
    def sorted_labels(self) -> list[np.ndarray]:
        indexer, to_sort = self._indexer_and_to_sort
        if self.sort:
            return [line.take(indexer) for line in to_sort]
        return to_sort

    def _make_sorted_values(self, values: np.ndarray) -> np.ndarray:
        if self.sort:
            indexer, _ = self._indexer_and_to_sort

            sorted_values = algos.take_nd(values, indexer, axis=0)
            return sorted_values
        return values

    def _make_selectors(self):
        new_levels = self.new_index_levels

        # make the mask
        remaining_labels = self.sorted_labels[:-1]
        level_sizes = tuple(len(x) for x in new_levels)

        comp_index, obs_ids = get_compressed_ids(remaining_labels, level_sizes)
        ngroups = len(obs_ids)

        comp_index = ensure_platform_int(comp_index)
        stride = self.index.levshape[self.level] + self.lift
        self.full_shape = ngroups, stride

        selector = self.sorted_labels[-1] + stride * comp_index + self.lift
        mask = np.zeros(np.prod(self.full_shape), dtype=bool)
        mask.put(selector, True)

        if mask.sum() < len(self.index):
            raise ValueError("Index contains duplicate entries, cannot reshape")

        self.group_index = comp_index
        self.mask = mask
        if self.sort:
            self.compressor = comp_index.searchsorted(np.arange(ngroups))
        else:
            self.compressor = np.sort(np.unique(comp_index, return_index=True)[1])

    @cache_readonly
    def mask_all(self) -> bool:
        return bool(self.mask.all())

    @cache_readonly
    def arange_result(self) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.bool_]]:
        # We cache this for reuse in ExtensionBlock._unstack
        dummy_arr = np.arange(len(self.index), dtype=np.intp)
        new_values, mask = self.get_new_values(dummy_arr, fill_value=-1)
        return new_values, mask.any(0)
        # TODO: in all tests we have mask.any(0).all(); can we rely on that?

    def get_result(self, values, value_columns, fill_value) -> DataFrame:
        if values.ndim == 1:
            values = values[:, np.newaxis]

        if value_columns is None and values.shape[1] != 1:  # pragma: no cover
            raise ValueError("must pass column labels for multi-column data")

        values, _ = self.get_new_values(values, fill_value)
        columns = self.get_new_columns(value_columns)
        index = self.new_index

        return self.constructor(
            values, index=index, columns=columns, dtype=values.dtype
        )

    def get_new_values(self, values, fill_value=None):
        if values.ndim == 1:
            values = values[:, np.newaxis]

        sorted_values = self._make_sorted_values(values)

        # place the values
        length, width = self.full_shape
        stride = values.shape[1]
        result_width = width * stride
        result_shape = (length, result_width)
        mask = self.mask
        mask_all = self.mask_all

        # we can simply reshape if we don't have a mask
        if mask_all and len(values):
            # TODO: Under what circumstances can we rely on sorted_values
            #  matching values?  When that holds, we can slice instead
            #  of take (in particular for EAs)
            new_values = (
                sorted_values.reshape(length, width, stride)
                .swapaxes(1, 2)
                .reshape(result_shape)
            )
            new_mask = np.ones(result_shape, dtype=bool)
            return new_values, new_mask

        dtype = values.dtype

        # if our mask is all True, then we can use our existing dtype
        if mask_all:
            dtype = values.dtype
            new_values = np.empty(result_shape, dtype=dtype)
        else:
            if isinstance(dtype, ExtensionDtype):
                # GH#41875
                # We are assuming that fill_value can be held by this dtype,
                #  unlike the non-EA case that promotes.
                cls = dtype.construct_array_type()
                new_values = cls._empty(result_shape, dtype=dtype)
                new_values[:] = fill_value
            else:
                dtype, fill_value = maybe_promote(dtype, fill_value)
                new_values = np.empty(result_shape, dtype=dtype)
                new_values.fill(fill_value)

        name = dtype.name
        new_mask = np.zeros(result_shape, dtype=bool)

        # we need to convert to a basic dtype
        # and possibly coerce an input to our output dtype
        # e.g. ints -> floats
        if needs_i8_conversion(values.dtype):
            sorted_values = sorted_values.view("i8")
            new_values = new_values.view("i8")
        else:
            sorted_values = sorted_values.astype(name, copy=False)

        # fill in our values & mask
        libreshape.unstack(
            sorted_values,
            mask.view("u1"),
            stride,
            length,
            width,
            new_values,
            new_mask.view("u1"),
        )

        # reconstruct dtype if needed
        if needs_i8_conversion(values.dtype):
            # view as datetime64 so we can wrap in DatetimeArray and use
            #  DTA's view method
            new_values = new_values.view("M8[ns]")
            new_values = ensure_wrapped_if_datetimelike(new_values)
            new_values = new_values.view(values.dtype)

        return new_values, new_mask

    def get_new_columns(self, value_columns: Index | None):
        if value_columns is None:
            if self.lift == 0:
                return self.removed_level._rename(name=self.removed_name)

            lev = self.removed_level.insert(0, item=self.removed_level._na_value)
            return lev.rename(self.removed_name)

        stride = len(self.removed_level) + self.lift
        width = len(value_columns)
        propagator = np.repeat(np.arange(width), stride)

        new_levels: FrozenList | list[Index]

        if isinstance(value_columns, MultiIndex):
            # error: Cannot determine type of "__add__"  [has-type]
            new_levels = value_columns.levels + (  # type: ignore[has-type]
                self.removed_level_full,
            )
            new_names = value_columns.names + (self.removed_name,)

            new_codes = [lab.take(propagator) for lab in value_columns.codes]
        else:
            new_levels = [
                value_columns,
                self.removed_level_full,
            ]
            new_names = [value_columns.name, self.removed_name]
            new_codes = [propagator]

        repeater = self._repeater

        # The entire level is then just a repetition of the single chunk:
        new_codes.append(np.tile(repeater, width))
        return MultiIndex(
            levels=new_levels, codes=new_codes, names=new_names, verify_integrity=False
        )

    @cache_readonly
    def _repeater(self) -> np.ndarray:
        # The two indices differ only if the unstacked level had unused items:
        if len(self.removed_level_full) != len(self.removed_level):
            # In this case, we remap the new codes to the original level:
            repeater = self.removed_level_full.get_indexer(self.removed_level)
            if self.lift:
                repeater = np.insert(repeater, 0, -1)
        else:
            # Otherwise, we just use each level item exactly once:
            stride = len(self.removed_level) + self.lift
            repeater = np.arange(stride) - self.lift

        return repeater

    @cache_readonly
    def new_index(self) -> MultiIndex:
        # Does not depend on values or value_columns
        result_codes = [lab.take(self.compressor) for lab in self.sorted_labels[:-1]]

        # construct the new index
        if len(self.new_index_levels) == 1:
            level, level_codes = self.new_index_levels[0], result_codes[0]
            if (level_codes == -1).any():
                level = level.insert(len(level), level._na_value)
            return level.take(level_codes).rename(self.new_index_names[0])

        return MultiIndex(
            levels=self.new_index_levels,
            codes=result_codes,
            names=self.new_index_names,
            verify_integrity=False,
        )


def _unstack_multiple(
    data: Series | DataFrame, clocs, fill_value=None, sort: bool = True
):
    if len(clocs) == 0:
        return data

    # NOTE: This doesn't deal with hierarchical columns yet

    index = data.index
    index = cast(MultiIndex, index)  # caller is responsible for checking

    # GH 19966 Make sure if MultiIndexed index has tuple name, they will be
    # recognised as a whole
    if clocs in index.names:
        clocs = [clocs]
    clocs = [index._get_level_number(i) for i in clocs]

    rlocs = [i for i in range(index.nlevels) if i not in clocs]

    clevels = [index.levels[i] for i in clocs]
    ccodes = [index.codes[i] for i in clocs]
    cnames = [index.names[i] for i in clocs]
    rlevels = [index.levels[i] for i in rlocs]
    rcodes = [index.codes[i] for i in rlocs]
    rnames = [index.names[i] for i in rlocs]

    shape = tuple(len(x) for x in clevels)
    group_index = get_group_index(ccodes, shape, sort=False, xnull=False)

    comp_ids, obs_ids = compress_group_index(group_index, sort=False)
    recons_codes = decons_obs_group_ids(comp_ids, obs_ids, shape, ccodes, xnull=False)

    if not rlocs:
        # Everything is in clocs, so the dummy df has a regular index
        dummy_index = Index(obs_ids, name="__placeholder__")
    else:
        dummy_index = MultiIndex(
            levels=rlevels + [obs_ids],
            codes=rcodes + [comp_ids],
            names=rnames + ["__placeholder__"],
            verify_integrity=False,
        )

    if isinstance(data, Series):
        dummy = data.copy()
        dummy.index = dummy_index

        unstacked = dummy.unstack("__placeholder__", fill_value=fill_value, sort=sort)
        new_levels = clevels
        new_names = cnames
        new_codes = recons_codes
    else:
        if isinstance(data.columns, MultiIndex):
            result = data
            while clocs:
                val = clocs.pop(0)
                result = result.unstack(val, fill_value=fill_value, sort=sort)
                clocs = [v if v < val else v - 1 for v in clocs]

            return result

        # GH#42579 deep=False to avoid consolidating
        dummy_df = data.copy(deep=False)
        dummy_df.index = dummy_index

        unstacked = dummy_df.unstack(
            "__placeholder__", fill_value=fill_value, sort=sort
        )
        if isinstance(unstacked, Series):
            unstcols = unstacked.index
        else:
            unstcols = unstacked.columns
        assert isinstance(unstcols, MultiIndex)  # for mypy
        new_levels = [unstcols.levels[0]] + clevels
        new_names = [data.columns.name] + cnames

        new_codes = [unstcols.codes[0]]
        new_codes.extend(rec.take(unstcols.codes[-1]) for rec in recons_codes)

    new_columns = MultiIndex(
        levels=new_levels, codes=new_codes, names=new_names, verify_integrity=False
    )

    if isinstance(unstacked, Series):
        unstacked.index = new_columns
    else:
        unstacked.columns = new_columns

    return unstacked


def unstack(obj: Series | DataFrame, level, fill_value=None, sort: bool = True):
    if isinstance(level, (tuple, list)):
        if len(level) != 1:
            # _unstack_multiple only handles MultiIndexes,
            # and isn't needed for a single level
            return _unstack_multiple(obj, level, fill_value=fill_value, sort=sort)
        else:
            level = level[0]

    if not is_integer(level) and not level == "__placeholder__":
        # check if level is valid in case of regular index
        obj.index._get_level_number(level)

    if isinstance(obj, DataFrame):
        if isinstance(obj.index, MultiIndex):
            return _unstack_frame(obj, level, fill_value=fill_value, sort=sort)
        else:
            return obj.T.stack(future_stack=True)
    elif not isinstance(obj.index, MultiIndex):
        # GH 36113
        # Give nicer error messages when unstack a Series whose
        # Index is not a MultiIndex.
        raise ValueError(
            f"index must be a MultiIndex to unstack, {type(obj.index)} was passed"
        )
    else:
        if is_1d_only_ea_dtype(obj.dtype):
            return _unstack_extension_series(obj, level, fill_value, sort=sort)
        unstacker = _Unstacker(
            obj.index, level=level, constructor=obj._constructor_expanddim, sort=sort
        )
        return unstacker.get_result(
            obj._values, value_columns=None, fill_value=fill_value
        )


def _unstack_frame(
    obj: DataFrame, level, fill_value=None, sort: bool = True
) -> DataFrame:
    assert isinstance(obj.index, MultiIndex)  # checked by caller
    unstacker = _Unstacker(
        obj.index, level=level, constructor=obj._constructor, sort=sort
    )

    if not obj._can_fast_transpose:
        mgr = obj._mgr.unstack(unstacker, fill_value=fill_value)
        return obj._constructor_from_mgr(mgr, axes=mgr.axes)
    else:
        return unstacker.get_result(
            obj._values, value_columns=obj.columns, fill_value=fill_value
        )


def _unstack_extension_series(
    series: Series, level, fill_value, sort: bool
) -> DataFrame:
    """
    Unstack an ExtensionArray-backed Series.

    The ExtensionDtype is preserved.

    Parameters
    ----------
    series : Series
        A Series with an ExtensionArray for values
    level : Any
        The level name or number.
    fill_value : Any
        The user-level (not physical storage) fill value to use for
        missing values introduced by the reshape. Passed to
        ``series.values.take``.
    sort : bool
        Whether to sort the resulting MuliIndex levels

    Returns
    -------
    DataFrame
        Each column of the DataFrame will have the same dtype as
        the input Series.
    """
    # Defer to the logic in ExtensionBlock._unstack
    df = series.to_frame()
    result = df.unstack(level=level, fill_value=fill_value, sort=sort)

    # equiv: result.droplevel(level=0, axis=1)
    #  but this avoids an extra copy
    result.columns = result.columns._drop_level_numbers([0])
    return result


def stack(frame: DataFrame, level=-1, dropna: bool = True, sort: bool = True):
    """
    Convert DataFrame to Series with multi-level Index. Columns become the
    second level of the resulting hierarchical index

    Returns
    -------
    stacked : Series or DataFrame
    """

    def stack_factorize(index):
        if index.is_unique:
            return index, np.arange(len(index))
        codes, categories = factorize_from_iterable(index)
        return categories, codes

    N, K = frame.shape

    # Will also convert negative level numbers and check if out of bounds.
    level_num = frame.columns._get_level_number(level)

    if isinstance(frame.columns, MultiIndex):
        return _stack_multi_columns(
            frame, level_num=level_num, dropna=dropna, sort=sort
        )
    elif isinstance(frame.index, MultiIndex):
        new_levels = list(frame.index.levels)
        new_codes = [lab.repeat(K) for lab in frame.index.codes]

        clev, clab = stack_factorize(frame.columns)
        new_levels.append(clev)
        new_codes.append(np.tile(clab, N).ravel())

        new_names = list(frame.index.names)
        new_names.append(frame.columns.name)
        new_index = MultiIndex(
            levels=new_levels, codes=new_codes, names=new_names, verify_integrity=False
        )
    else:
        levels, (ilab, clab) = zip(*map(stack_factorize, (frame.index, frame.columns)))
        codes = ilab.repeat(K), np.tile(clab, N).ravel()
        new_index = MultiIndex(
            levels=levels,
            codes=codes,
            names=[frame.index.name, frame.columns.name],
            verify_integrity=False,
        )

    new_values: ArrayLike
    if not frame.empty and frame._is_homogeneous_type:
        # For homogeneous EAs, frame._values will coerce to object. So
        # we concatenate instead.
        dtypes = list(frame.dtypes._values)
        dtype = dtypes[0]

        if isinstance(dtype, ExtensionDtype):
            arr = dtype.construct_array_type()
            new_values = arr._concat_same_type(
                [col._values for _, col in frame.items()]
            )
            new_values = _reorder_for_extension_array_stack(new_values, N, K)
        else:
            # homogeneous, non-EA
            new_values = frame._values.ravel()

    else:
        # non-homogeneous
        new_values = frame._values.ravel()

    if dropna:
        mask = notna(new_values)
        new_values = new_values[mask]
        new_index = new_index[mask]

    return frame._constructor_sliced(new_values, index=new_index)


def stack_multiple(frame: DataFrame, level, dropna: bool = True, sort: bool = True):
    # If all passed levels match up to column names, no
    # ambiguity about what to do
    if all(lev in frame.columns.names for lev in level):
        result = frame
        for lev in level:
            result = stack(result, lev, dropna=dropna, sort=sort)

    # Otherwise, level numbers may change as each successive level is stacked
    elif all(isinstance(lev, int) for lev in level):
        # As each stack is done, the level numbers decrease, so we need
        #  to account for that when level is a sequence of ints
        result = frame
        # _get_level_number() checks level numbers are in range and converts
        # negative numbers to positive
        level = [frame.columns._get_level_number(lev) for lev in level]

        while level:
            lev = level.pop(0)
            result = stack(result, lev, dropna=dropna, sort=sort)
            # Decrement all level numbers greater than current, as these
            # have now shifted down by one
            level = [v if v <= lev else v - 1 for v in level]

    else:
        raise ValueError(
            "level should contain all level names or all level "
            "numbers, not a mixture of the two."
        )

    return result


def _stack_multi_column_index(columns: MultiIndex) -> MultiIndex:
    """Creates a MultiIndex from the first N-1 levels of this MultiIndex."""
    if len(columns.levels) <= 2:
        return columns.levels[0]._rename(name=columns.names[0])

    levs = [
        [lev[c] if c >= 0 else None for c in codes]
        for lev, codes in zip(columns.levels[:-1], columns.codes[:-1])
    ]

    # Remove duplicate tuples in the MultiIndex.
    tuples = zip(*levs)
    unique_tuples = (key for key, _ in itertools.groupby(tuples))
    new_levs = zip(*unique_tuples)

    # The dtype of each level must be explicitly set to avoid inferring the wrong type.
    # See GH-36991.
    return MultiIndex.from_arrays(
        [
            # Not all indices can accept None values.
            Index(new_lev, dtype=lev.dtype) if None not in new_lev else new_lev
            for new_lev, lev in zip(new_levs, columns.levels)
        ],
        names=columns.names[:-1],
    )


def _stack_multi_columns(
    frame: DataFrame, level_num: int = -1, dropna: bool = True, sort: bool = True
) -> DataFrame:
    def _convert_level_number(level_num: int, columns: Index):
        """
        Logic for converting the level number to something we can safely pass
        to swaplevel.

        If `level_num` matches a column name return the name from
        position `level_num`, otherwise return `level_num`.
        """
        if level_num in columns.names:
            return columns.names[level_num]

        return level_num

    this = frame.copy(deep=False)
    mi_cols = this.columns  # cast(MultiIndex, this.columns)
    assert isinstance(mi_cols, MultiIndex)  # caller is responsible

    # this makes life much simpler
    if level_num != mi_cols.nlevels - 1:
        # roll levels to put selected level at end
        roll_columns = mi_cols
        for i in range(level_num, mi_cols.nlevels - 1):
            # Need to check if the ints conflict with level names
            lev1 = _convert_level_number(i, roll_columns)
            lev2 = _convert_level_number(i + 1, roll_columns)
            roll_columns = roll_columns.swaplevel(lev1, lev2)
        this.columns = mi_cols = roll_columns

    if not mi_cols._is_lexsorted() and sort:
        # Workaround the edge case where 0 is one of the column names,
        # which interferes with trying to sort based on the first
        # level
        level_to_sort = _convert_level_number(0, mi_cols)
        this = this.sort_index(level=level_to_sort, axis=1)
        mi_cols = this.columns

    mi_cols = cast(MultiIndex, mi_cols)
    new_columns = _stack_multi_column_index(mi_cols)

    # time to ravel the values
    new_data = {}
    level_vals = mi_cols.levels[-1]
    level_codes = unique(mi_cols.codes[-1])
    if sort:
        level_codes = np.sort(level_codes)
    level_vals_nan = level_vals.insert(len(level_vals), None)

    level_vals_used = np.take(level_vals_nan, level_codes)
    levsize = len(level_codes)
    drop_cols = []
    for key in new_columns:
        try:
            loc = this.columns.get_loc(key)
        except KeyError:
            drop_cols.append(key)
            continue

        # can make more efficient?
        # we almost always return a slice
        # but if unsorted can get a boolean
        # indexer
        if not isinstance(loc, slice):
            slice_len = len(loc)
        else:
            slice_len = loc.stop - loc.start

        if slice_len != levsize:
            chunk = this.loc[:, this.columns[loc]]
            chunk.columns = level_vals_nan.take(chunk.columns.codes[-1])
            value_slice = chunk.reindex(columns=level_vals_used).values
        else:
            subset = this.iloc[:, loc]
            dtype = find_common_type(subset.dtypes.tolist())
            if isinstance(dtype, ExtensionDtype):
                # TODO(EA2D): won't need special case, can go through .values
                #  paths below (might change to ._values)
                value_slice = dtype.construct_array_type()._concat_same_type(
                    [x._values.astype(dtype, copy=False) for _, x in subset.items()]
                )
                N, K = subset.shape
                idx = np.arange(N * K).reshape(K, N).T.ravel()
                value_slice = value_slice.take(idx)
            else:
                value_slice = subset.values

        if value_slice.ndim > 1:
            # i.e. not extension
            value_slice = value_slice.ravel()

        new_data[key] = value_slice

    if len(drop_cols) > 0:
        new_columns = new_columns.difference(drop_cols)

    N = len(this)

    if isinstance(this.index, MultiIndex):
        new_levels = list(this.index.levels)
        new_names = list(this.index.names)
        new_codes = [lab.repeat(levsize) for lab in this.index.codes]
    else:
        old_codes, old_levels = factorize_from_iterable(this.index)
        new_levels = [old_levels]
        new_codes = [old_codes.repeat(levsize)]
        new_names = [this.index.name]  # something better?

    new_levels.append(level_vals)
    new_codes.append(np.tile(level_codes, N))
    new_names.append(frame.columns.names[level_num])

    new_index = MultiIndex(
        levels=new_levels, codes=new_codes, names=new_names, verify_integrity=False
    )

    result = frame._constructor(new_data, index=new_index, columns=new_columns)

    if frame.columns.nlevels > 1:
        desired_columns = frame.columns._drop_level_numbers([level_num]).unique()
        if not result.columns.equals(desired_columns):
            result = result[desired_columns]

    # more efficient way to go about this? can do the whole masking biz but
    # will only save a small amount of time...
    if dropna:
        result = result.dropna(axis=0, how="all")

    return result


def _reorder_for_extension_array_stack(
    arr: ExtensionArray, n_rows: int, n_columns: int
) -> ExtensionArray:
    """
    Re-orders the values when stacking multiple extension-arrays.

    The indirect stacking method used for EAs requires a followup
    take to get the order correct.

    Parameters
    ----------
    arr : ExtensionArray
    n_rows, n_columns : int
        The number of rows and columns in the original DataFrame.

    Returns
    -------
    taken : ExtensionArray
        The original `arr` with elements re-ordered appropriately

    Examples
    --------
    >>> arr = np.array(['a', 'b', 'c', 'd', 'e', 'f'])
    >>> _reorder_for_extension_array_stack(arr, 2, 3)
    array(['a', 'c', 'e', 'b', 'd', 'f'], dtype='<U1')

    >>> _reorder_for_extension_array_stack(arr, 3, 2)
    array(['a', 'd', 'b', 'e', 'c', 'f'], dtype='<U1')
    """
    # final take to get the order correct.
    # idx is an indexer like
    # [c0r0, c1r0, c2r0, ...,
    #  c0r1, c1r1, c2r1, ...]
    idx = np.arange(n_rows * n_columns).reshape(n_columns, n_rows).T.ravel()
    return arr.take(idx)


def stack_v3(frame: DataFrame, level: list[int]) -> Series | DataFrame:
    if frame.columns.nunique() != len(frame.columns):
        raise ValueError("Columns with duplicate values are not supported in stack")

    # If we need to drop `level` from columns, it needs to be in descending order
    drop_levnums = sorted(level, reverse=True)
    stack_cols = frame.columns._drop_level_numbers(
        [k for k in range(frame.columns.nlevels) if k not in level][::-1]
    )
    if len(level) > 1:
        # Arrange columns in the order we want to take them, e.g. level=[2, 0, 1]
        sorter = np.argsort(level)
        ordered_stack_cols = stack_cols._reorder_ilevels(sorter)
    else:
        ordered_stack_cols = stack_cols

    stack_cols_unique = stack_cols.unique()
    ordered_stack_cols_unique = ordered_stack_cols.unique()

    # Grab data for each unique index to be stacked
    buf = []
    for idx in stack_cols_unique:
        if len(frame.columns) == 1:
            data = frame.copy()
        else:
            # Take the data from frame corresponding to this idx value
            if len(level) == 1:
                idx = (idx,)
            gen = iter(idx)
            column_indexer = tuple(
                next(gen) if k in level else slice(None)
                for k in range(frame.columns.nlevels)
            )
            data = frame.loc[:, column_indexer]

        if len(level) < frame.columns.nlevels:
            data.columns = data.columns._drop_level_numbers(drop_levnums)
        elif stack_cols.nlevels == 1:
            if data.ndim == 1:
                data.name = 0
            else:
                data.columns = RangeIndex(len(data.columns))
        buf.append(data)

    result: Series | DataFrame
    if len(buf) > 0 and not frame.empty:
        result = concat(buf)
        ratio = len(result) // len(frame)
    else:
        # input is empty
        if len(level) < frame.columns.nlevels:
            # concat column order may be different from dropping the levels
            new_columns = frame.columns._drop_level_numbers(drop_levnums).unique()
        else:
            new_columns = [0]
        result = DataFrame(columns=new_columns, dtype=frame._values.dtype)
        ratio = 0

    if len(level) < frame.columns.nlevels:
        # concat column order may be different from dropping the levels
        desired_columns = frame.columns._drop_level_numbers(drop_levnums).unique()
        if not result.columns.equals(desired_columns):
            result = result[desired_columns]

    # Construct the correct MultiIndex by combining the frame's index and
    # stacked columns.
    index_levels: list | FrozenList
    if isinstance(frame.index, MultiIndex):
        index_levels = frame.index.levels
        index_codes = list(np.tile(frame.index.codes, (1, ratio)))
    else:
        codes, uniques = factorize(frame.index, use_na_sentinel=False)
        index_levels = [uniques]
        index_codes = list(np.tile(codes, (1, ratio)))
    if isinstance(stack_cols, MultiIndex):
        column_levels = ordered_stack_cols.levels
        column_codes = ordered_stack_cols.drop_duplicates().codes
    else:
        column_levels = [ordered_stack_cols.unique()]
        column_codes = [factorize(ordered_stack_cols_unique, use_na_sentinel=False)[0]]
    column_codes = [np.repeat(codes, len(frame)) for codes in column_codes]
    result.index = MultiIndex(
        levels=index_levels + column_levels,
        codes=index_codes + column_codes,
        names=frame.index.names + list(ordered_stack_cols.names),
        verify_integrity=False,
    )

    # sort result, but faster than calling sort_index since we know the order we need
    len_df = len(frame)
    n_uniques = len(ordered_stack_cols_unique)
    indexer = np.arange(n_uniques)
    idxs = np.tile(len_df * indexer, len_df) + np.repeat(np.arange(len_df), n_uniques)
    result = result.take(idxs)

    # Reshape/rename if needed and dropna
    if result.ndim == 2 and frame.columns.nlevels == len(level):
        if len(result.columns) == 0:
            result = Series(index=result.index)
        else:
            result = result.iloc[:, 0]
    if result.ndim == 1:
        result.name = None

    return result
<|endoftext|>from __future__ import annotations

from collections.abc import (
    Hashable,
    Sequence,
)
from typing import (
    TYPE_CHECKING,
    Callable,
    Literal,
    cast,
)
import warnings

import numpy as np

from pandas._libs import lib
from pandas.util._decorators import (
    Appender,
    Substitution,
)
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.cast import maybe_downcast_to_dtype
from pandas.core.dtypes.common import (
    is_list_like,
    is_nested_list_like,
    is_scalar,
)
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCSeries,
)

import pandas.core.common as com
from pandas.core.frame import _shared_docs
from pandas.core.groupby import Grouper
from pandas.core.indexes.api import (
    Index,
    MultiIndex,
    get_objs_combined_axis,
)
from pandas.core.reshape.concat import concat
from pandas.core.reshape.util import cartesian_product
from pandas.core.series import Series

if TYPE_CHECKING:
    from pandas._typing import (
        AggFuncType,
        AggFuncTypeBase,
        AggFuncTypeDict,
        IndexLabel,
    )

    from pandas import DataFrame


# Note: We need to make sure `frame` is imported before `pivot`, otherwise
# _shared_docs['pivot_table'] will not yet exist.  TODO: Fix this dependency
@Substitution("\ndata : DataFrame")
@Appender(_shared_docs["pivot_table"], indents=1)
def pivot_table(
    data: DataFrame,
    values=None,
    index=None,
    columns=None,
    aggfunc: AggFuncType = "mean",
    fill_value=None,
    margins: bool = False,
    dropna: bool = True,
    margins_name: Hashable = "All",
    observed: bool | lib.NoDefault = lib.no_default,
    sort: bool = True,
) -> DataFrame:
    index = _convert_by(index)
    columns = _convert_by(columns)

    if isinstance(aggfunc, list):
        pieces: list[DataFrame] = []
        keys = []
        for func in aggfunc:
            _table = __internal_pivot_table(
                data,
                values=values,
                index=index,
                columns=columns,
                fill_value=fill_value,
                aggfunc=func,
                margins=margins,
                dropna=dropna,
                margins_name=margins_name,
                observed=observed,
                sort=sort,
            )
            pieces.append(_table)
            keys.append(getattr(func, "__name__", func))

        table = concat(pieces, keys=keys, axis=1)
        return table.__finalize__(data, method="pivot_table")

    table = __internal_pivot_table(
        data,
        values,
        index,
        columns,
        aggfunc,
        fill_value,
        margins,
        dropna,
        margins_name,
        observed,
        sort,
    )
    return table.__finalize__(data, method="pivot_table")


def __internal_pivot_table(
    data: DataFrame,
    values,
    index,
    columns,
    aggfunc: AggFuncTypeBase | AggFuncTypeDict,
    fill_value,
    margins: bool,
    dropna: bool,
    margins_name: Hashable,
    observed: bool | lib.NoDefault,
    sort: bool,
) -> DataFrame:
    """
    Helper of :func:`pandas.pivot_table` for any non-list ``aggfunc``.
    """
    keys = index + columns

    values_passed = values is not None
    if values_passed:
        if is_list_like(values):
            values_multi = True
            values = list(values)
        else:
            values_multi = False
            values = [values]

        # GH14938 Make sure value labels are in data
        for i in values:
            if i not in data:
                raise KeyError(i)

        to_filter = []
        for x in keys + values:
            if isinstance(x, Grouper):
                x = x.key
            try:
                if x in data:
                    to_filter.append(x)
            except TypeError:
                pass
        if len(to_filter) < len(data.columns):
            data = data[to_filter]

    else:
        values = data.columns
        for key in keys:
            try:
                values = values.drop(key)
            except (TypeError, ValueError, KeyError):
                pass
        values = list(values)

    observed_bool = False if observed is lib.no_default else observed
    grouped = data.groupby(keys, observed=observed_bool, sort=sort, dropna=dropna)
    if observed is lib.no_default and any(
        ping._passed_categorical for ping in grouped._grouper.groupings
    ):
        warnings.warn(
            "The default value of observed=False is deprecated and will change "
            "to observed=True in a future version of pandas. Specify "
            "observed=False to silence this warning and retain the current behavior",
            category=FutureWarning,
            stacklevel=find_stack_level(),
        )
    agged = grouped.agg(aggfunc)

    if dropna and isinstance(agged, ABCDataFrame) and len(agged.columns):
        agged = agged.dropna(how="all")

    table = agged

    # GH17038, this check should only happen if index is defined (not None)
    if table.index.nlevels > 1 and index:
        # Related GH #17123
        # If index_names are integers, determine whether the integers refer
        # to the level position or name.
        index_names = agged.index.names[: len(index)]
        to_unstack = []
        for i in range(len(index), len(keys)):
            name = agged.index.names[i]
            if name is None or name in index_names:
                to_unstack.append(i)
            else:
                to_unstack.append(name)
        table = agged.unstack(to_unstack, fill_value=fill_value)

    if not dropna:
        if isinstance(table.index, MultiIndex):
            m = MultiIndex.from_arrays(
                cartesian_product(table.index.levels), names=table.index.names
            )
            table = table.reindex(m, axis=0, fill_value=fill_value)

        if isinstance(table.columns, MultiIndex):
            m = MultiIndex.from_arrays(
                cartesian_product(table.columns.levels), names=table.columns.names
            )
            table = table.reindex(m, axis=1, fill_value=fill_value)

    if sort is True and isinstance(table, ABCDataFrame):
        table = table.sort_index(axis=1)

    if fill_value is not None:
        table = table.fillna(fill_value)
        if aggfunc is len and not observed and lib.is_integer(fill_value):
            # TODO: can we avoid this?  this used to be handled by
            #  downcast="infer" in fillna
            table = table.astype(np.int64)

    if margins:
        if dropna:
            data = data[data.notna().all(axis=1)]
        table = _add_margins(
            table,
            data,
            values,
            rows=index,
            cols=columns,
            aggfunc=aggfunc,
            observed=dropna,
            margins_name=margins_name,
            fill_value=fill_value,
        )

    # discard the top level
    if values_passed and not values_multi and table.columns.nlevels > 1:
        table.columns = table.columns.droplevel(0)
    if len(index) == 0 and len(columns) > 0:
        table = table.T

    # GH 15193 Make sure empty columns are removed if dropna=True
    if isinstance(table, ABCDataFrame) and dropna:
        table = table.dropna(how="all", axis=1)

    return table


def _add_margins(
    table: DataFrame | Series,
    data: DataFrame,
    values,
    rows,
    cols,
    aggfunc,
    observed: bool,
    margins_name: Hashable = "All",
    fill_value=None,
):
    if not isinstance(margins_name, str):
        raise ValueError("margins_name argument must be a string")

    msg = f'Conflicting name "{margins_name}" in margins'
    for level in table.index.names:
        if margins_name in table.index.get_level_values(level):
            raise ValueError(msg)

    grand_margin = _compute_grand_margin(data, values, aggfunc, margins_name)

    if table.ndim == 2:
        # i.e. DataFrame
        for level in table.columns.names[1:]:
            if margins_name in table.columns.get_level_values(level):
                raise ValueError(msg)

    key: str | tuple[str, ...]
    if len(rows) > 1:
        key = (margins_name,) + ("",) * (len(rows) - 1)
    else:
        key = margins_name

    if not values and isinstance(table, ABCSeries):
        # If there are no values and the table is a series, then there is only
        # one column in the data. Compute grand margin and return it.
        return table._append(table._constructor({key: grand_margin[margins_name]}))

    elif values:
        marginal_result_set = _generate_marginal_results(
            table, data, values, rows, cols, aggfunc, observed, margins_name
        )
        if not isinstance(marginal_result_set, tuple):
            return marginal_result_set
        result, margin_keys, row_margin = marginal_result_set
    else:
        # no values, and table is a DataFrame
        assert isinstance(table, ABCDataFrame)
        marginal_result_set = _generate_marginal_results_without_values(
            table, data, rows, cols, aggfunc, observed, margins_name
        )
        if not isinstance(marginal_result_set, tuple):
            return marginal_result_set
        result, margin_keys, row_margin = marginal_result_set

    row_margin = row_margin.reindex(result.columns, fill_value=fill_value)
    # populate grand margin
    for k in margin_keys:
        if isinstance(k, str):
            row_margin[k] = grand_margin[k]
        else:
            row_margin[k] = grand_margin[k[0]]

    from pandas import DataFrame

    margin_dummy = DataFrame(row_margin, columns=Index([key])).T

    row_names = result.index.names
    # check the result column and leave floats

    for dtype in set(result.dtypes):
        if isinstance(dtype, ExtensionDtype):
            # Can hold NA already
            continue

        cols = result.select_dtypes([dtype]).columns
        margin_dummy[cols] = margin_dummy[cols].apply(
            maybe_downcast_to_dtype, args=(dtype,)
        )
    result = result._append(margin_dummy)
    result.index.names = row_names

    return result


def _compute_grand_margin(
    data: DataFrame, values, aggfunc, margins_name: Hashable = "All"
):
    if values:
        grand_margin = {}
        for k, v in data[values].items():
            try:
                if isinstance(aggfunc, str):
                    grand_margin[k] = getattr(v, aggfunc)()
                elif isinstance(aggfunc, dict):
                    if isinstance(aggfunc[k], str):
                        grand_margin[k] = getattr(v, aggfunc[k])()
                    else:
                        grand_margin[k] = aggfunc[k](v)
                else:
                    grand_margin[k] = aggfunc(v)
            except TypeError:
                pass
        return grand_margin
    else:
        return {margins_name: aggfunc(data.index)}


def _generate_marginal_results(
    table,
    data: DataFrame,
    values,
    rows,
    cols,
    aggfunc,
    observed: bool,
    margins_name: Hashable = "All",
):
    margin_keys: list | Index
    if len(cols) > 0:
        # need to "interleave" the margins
        table_pieces = []
        margin_keys = []

        def _all_key(key):
            return (key, margins_name) + ("",) * (len(cols) - 1)

        if len(rows) > 0:
            margin = data[rows + values].groupby(rows, observed=observed).agg(aggfunc)
            cat_axis = 1

            for key, piece in table.T.groupby(level=0, observed=observed):
                piece = piece.T
                all_key = _all_key(key)

                # we are going to mutate this, so need to copy!
                piece = piece.copy()
                piece[all_key] = margin[key]

                table_pieces.append(piece)
                margin_keys.append(all_key)
        else:
            from pandas import DataFrame

            cat_axis = 0
            for key, piece in table.groupby(level=0, observed=observed):
                if len(cols) > 1:
                    all_key = _all_key(key)
                else:
                    all_key = margins_name
                table_pieces.append(piece)
                # GH31016 this is to calculate margin for each group, and assign
                # corresponded key as index
                transformed_piece = DataFrame(piece.apply(aggfunc)).T
                if isinstance(piece.index, MultiIndex):
                    # We are adding an empty level
                    transformed_piece.index = MultiIndex.from_tuples(
                        [all_key], names=piece.index.names + [None]
                    )
                else:
                    transformed_piece.index = Index([all_key], name=piece.index.name)

                # append piece for margin into table_piece
                table_pieces.append(transformed_piece)
                margin_keys.append(all_key)

        if not table_pieces:
            # GH 49240
            return table
        else:
            result = concat(table_pieces, axis=cat_axis)

        if len(rows) == 0:
            return result
    else:
        result = table
        margin_keys = table.columns

    if len(cols) > 0:
        row_margin = data[cols + values].groupby(cols, observed=observed).agg(aggfunc)
        row_margin = row_margin.stack(future_stack=True)

        # GH#26568. Use names instead of indices in case of numeric names
        new_order_indices = [len(cols)] + list(range(len(cols)))
        new_order_names = [row_margin.index.names[i] for i in new_order_indices]
        row_margin.index = row_margin.index.reorder_levels(new_order_names)
    else:
        row_margin = data._constructor_sliced(np.nan, index=result.columns)

    return result, margin_keys, row_margin


def _generate_marginal_results_without_values(
    table: DataFrame,
    data: DataFrame,
    rows,
    cols,
    aggfunc,
    observed: bool,
    margins_name: Hashable = "All",
):
    margin_keys: list | Index
    if len(cols) > 0:
        # need to "interleave" the margins
        margin_keys = []

        def _all_key():
            if len(cols) == 1:
                return margins_name
            return (margins_name,) + ("",) * (len(cols) - 1)

        if len(rows) > 0:
            margin = data.groupby(rows, observed=observed)[rows].apply(aggfunc)
            all_key = _all_key()
            table[all_key] = margin
            result = table
            margin_keys.append(all_key)

        else:
            margin = data.groupby(level=0, axis=0, observed=observed).apply(aggfunc)
            all_key = _all_key()
            table[all_key] = margin
            result = table
            margin_keys.append(all_key)
            return result
    else:
        result = table
        margin_keys = table.columns

    if len(cols):
        row_margin = data.groupby(cols, observed=observed)[cols].apply(aggfunc)
    else:
        row_margin = Series(np.nan, index=result.columns)

    return result, margin_keys, row_margin


def _convert_by(by):
    if by is None:
        by = []
    elif (
        is_scalar(by)
        or isinstance(by, (np.ndarray, Index, ABCSeries, Grouper))
        or callable(by)
    ):
        by = [by]
    else:
        by = list(by)
    return by


@Substitution("\ndata : DataFrame")
@Appender(_shared_docs["pivot"], indents=1)
def pivot(
    data: DataFrame,
    *,
    columns: IndexLabel,
    index: IndexLabel | lib.NoDefault = lib.no_default,
    values: IndexLabel | lib.NoDefault = lib.no_default,
) -> DataFrame:
    columns_listlike = com.convert_to_list_like(columns)

    # If columns is None we will create a MultiIndex level with None as name
    # which might cause duplicated names because None is the default for
    # level names
    data = data.copy(deep=False)
    data.index = data.index.copy()
    data.index.names = [
        name if name is not None else lib.no_default for name in data.index.names
    ]

    indexed: DataFrame | Series
    if values is lib.no_default:
        if index is not lib.no_default:
            cols = com.convert_to_list_like(index)
        else:
            cols = []

        append = index is lib.no_default
        # error: Unsupported operand types for + ("List[Any]" and "ExtensionArray")
        # error: Unsupported left operand type for + ("ExtensionArray")
        indexed = data.set_index(
            cols + columns_listlike, append=append  # type: ignore[operator]
        )
    else:
        index_list: list[Index] | list[Series]
        if index is lib.no_default:
            if isinstance(data.index, MultiIndex):
                # GH 23955
                index_list = [
                    data.index.get_level_values(i) for i in range(data.index.nlevels)
                ]
            else:
                index_list = [
                    data._constructor_sliced(data.index, name=data.index.name)
                ]
        else:
            index_list = [data[idx] for idx in com.convert_to_list_like(index)]

        data_columns = [data[col] for col in columns_listlike]
        index_list.extend(data_columns)
        multiindex = MultiIndex.from_arrays(index_list)

        if is_list_like(values) and not isinstance(values, tuple):
            # Exclude tuple because it is seen as a single column name
            values = cast(Sequence[Hashable], values)
            indexed = data._constructor(
                data[values]._values, index=multiindex, columns=values
            )
        else:
            indexed = data._constructor_sliced(data[values]._values, index=multiindex)
    # error: Argument 1 to "unstack" of "DataFrame" has incompatible type "Union
    # [List[Any], ExtensionArray, ndarray[Any, Any], Index, Series]"; expected
    # "Hashable"
    result = indexed.unstack(columns_listlike)  # type: ignore[arg-type]
    result.index.names = [
        name if name is not lib.no_default else None for name in result.index.names
    ]

    return result


def crosstab(
    index,
    columns,
    values=None,
    rownames=None,
    colnames=None,
    aggfunc=None,
    margins: bool = False,
    margins_name: Hashable = "All",
    dropna: bool = True,
    normalize: bool | Literal[0, 1, "all", "index", "columns"] = False,
) -> DataFrame:
    """
    Compute a simple cross tabulation of two (or more) factors.

    By default, computes a frequency table of the factors unless an
    array of values and an aggregation function are passed.

    Parameters
    ----------
    index : array-like, Series, or list of arrays/Series
        Values to group by in the rows.
    columns : array-like, Series, or list of arrays/Series
        Values to group by in the columns.
    values : array-like, optional
        Array of values to aggregate according to the factors.
        Requires `aggfunc` be specified.
    rownames : sequence, default None
        If passed, must match number of row arrays passed.
    colnames : sequence, default None
        If passed, must match number of column arrays passed.
    aggfunc : function, optional
        If specified, requires `values` be specified as well.
    margins : bool, default False
        Add row/column margins (subtotals).
    margins_name : str, default 'All'
        Name of the row/column that will contain the totals
        when margins is True.
    dropna : bool, default True
        Do not include columns whose entries are all NaN.
    normalize : bool, {'all', 'index', 'columns'}, or {0,1}, default False
        Normalize by dividing all values by the sum of values.

        - If passed 'all' or `True`, will normalize over all values.
        - If passed 'index' will normalize over each row.
        - If passed 'columns' will normalize over each column.
        - If margins is `True`, will also normalize margin values.

    Returns
    -------
    DataFrame
        Cross tabulation of the data.

    See Also
    --------
    DataFrame.pivot : Reshape data based on column values.
    pivot_table : Create a pivot table as a DataFrame.

    Notes
    -----
    Any Series passed will have their name attributes used unless row or column
    names for the cross-tabulation are specified.

    Any input passed containing Categorical data will have **all** of its
    categories included in the cross-tabulation, even if the actual data does
    not contain any instances of a particular category.

    In the event that there aren't overlapping indexes an empty DataFrame will
    be returned.

    Reference :ref:`the user guide <reshaping.crosstabulations>` for more examples.

    Examples
    --------
    >>> a = np.array(["foo", "foo", "foo", "foo", "bar", "bar",
    ...               "bar", "bar", "foo", "foo", "foo"], dtype=object)
    >>> b = np.array(["one", "one", "one", "two", "one", "one",
    ...               "one", "two", "two", "two", "one"], dtype=object)
    >>> c = np.array(["dull", "dull", "shiny", "dull", "dull", "shiny",
    ...               "shiny", "dull", "shiny", "shiny", "shiny"],
    ...              dtype=object)
    >>> pd.crosstab(a, [b, c], rownames=['a'], colnames=['b', 'c'])
    b   one        two
    c   dull shiny dull shiny
    a
    bar    1     2    1     0
    foo    2     2    1     2

    Here 'c' and 'f' are not represented in the data and will not be
    shown in the output because dropna is True by default. Set
    dropna=False to preserve categories with no data.

    >>> foo = pd.Categorical(['a', 'b'], categories=['a', 'b', 'c'])
    >>> bar = pd.Categorical(['d', 'e'], categories=['d', 'e', 'f'])
    >>> pd.crosstab(foo, bar)
    col_0  d  e
    row_0
    a      1  0
    b      0  1
    >>> pd.crosstab(foo, bar, dropna=False)
    col_0  d  e  f
    row_0
    a      1  0  0
    b      0  1  0
    c      0  0  0
    """
    if values is None and aggfunc is not None:
        raise ValueError("aggfunc cannot be used without values.")

    if values is not None and aggfunc is None:
        raise ValueError("values cannot be used without an aggfunc.")

    if not is_nested_list_like(index):
        index = [index]
    if not is_nested_list_like(columns):
        columns = [columns]

    common_idx = None
    pass_objs = [x for x in index + columns if isinstance(x, (ABCSeries, ABCDataFrame))]
    if pass_objs:
        common_idx = get_objs_combined_axis(pass_objs, intersect=True, sort=False)

    rownames = _get_names(index, rownames, prefix="row")
    colnames = _get_names(columns, colnames, prefix="col")

    # duplicate names mapped to unique names for pivot op
    (
        rownames_mapper,
        unique_rownames,
        colnames_mapper,
        unique_colnames,
    ) = _build_names_mapper(rownames, colnames)

    from pandas import DataFrame

    data = {
        **dict(zip(unique_rownames, index)),
        **dict(zip(unique_colnames, columns)),
    }
    df = DataFrame(data, index=common_idx)

    if values is None:
        df["__dummy__"] = 0
        kwargs = {"aggfunc": len, "fill_value": 0}
    else:
        df["__dummy__"] = values
        kwargs = {"aggfunc": aggfunc}

    # error: Argument 7 to "pivot_table" of "DataFrame" has incompatible type
    # "**Dict[str, object]"; expected "Union[...]"
    table = df.pivot_table(
        "__dummy__",
        index=unique_rownames,
        columns=unique_colnames,
        margins=margins,
        margins_name=margins_name,
        dropna=dropna,
        observed=False,
        **kwargs,  # type: ignore[arg-type]
    )

    # Post-process
    if normalize is not False:
        table = _normalize(
            table, normalize=normalize, margins=margins, margins_name=margins_name
        )

    table = table.rename_axis(index=rownames_mapper, axis=0)
    table = table.rename_axis(columns=colnames_mapper, axis=1)

    return table


def _normalize(
    table: DataFrame, normalize, margins: bool, margins_name: Hashable = "All"
) -> DataFrame:
    if not isinstance(normalize, (bool, str)):
        axis_subs = {0: "index", 1: "columns"}
        try:
            normalize = axis_subs[normalize]
        except KeyError as err:
            raise ValueError("Not a valid normalize argument") from err

    if margins is False:
        # Actual Normalizations
        normalizers: dict[bool | str, Callable] = {
            "all": lambda x: x / x.sum(axis=1).sum(axis=0),
            "columns": lambda x: x / x.sum(),
            "index": lambda x: x.div(x.sum(axis=1), axis=0),
        }

        normalizers[True] = normalizers["all"]

        try:
            f = normalizers[normalize]
        except KeyError as err:
            raise ValueError("Not a valid normalize argument") from err

        table = f(table)
        table = table.fillna(0)

    elif margins is True:
        # keep index and column of pivoted table
        table_index = table.index
        table_columns = table.columns
        last_ind_or_col = table.iloc[-1, :].name

        # check if margin name is not in (for MI cases) and not equal to last
        # index/column and save the column and index margin
        if (margins_name not in last_ind_or_col) & (margins_name != last_ind_or_col):
            raise ValueError(f"{margins_name} not in pivoted DataFrame")
        column_margin = table.iloc[:-1, -1]
        index_margin = table.iloc[-1, :-1]

        # keep the core table
        table = table.iloc[:-1, :-1]

        # Normalize core
        table = _normalize(table, normalize=normalize, margins=False)

        # Fix Margins
        if normalize == "columns":
            column_margin = column_margin / column_margin.sum()
            table = concat([table, column_margin], axis=1)
            table = table.fillna(0)
            table.columns = table_columns

        elif normalize == "index":
            index_margin = index_margin / index_margin.sum()
            table = table._append(index_margin)
            table = table.fillna(0)
            table.index = table_index

        elif normalize == "all" or normalize is True:
            column_margin = column_margin / column_margin.sum()
            index_margin = index_margin / index_margin.sum()
            index_margin.loc[margins_name] = 1
            table = concat([table, column_margin], axis=1)
            table = table._append(index_margin)

            table = table.fillna(0)
            table.index = table_index
            table.columns = table_columns

        else:
            raise ValueError("Not a valid normalize argument")

    else:
        raise ValueError("Not a valid margins argument")

    return table


def _get_names(arrs, names, prefix: str = "row"):
    if names is None:
        names = []
        for i, arr in enumerate(arrs):
            if isinstance(arr, ABCSeries) and arr.name is not None:
                names.append(arr.name)
            else:
                names.append(f"{prefix}_{i}")
    else:
        if len(names) != len(arrs):
            raise AssertionError("arrays and names must have the same length")
        if not isinstance(names, list):
            names = list(names)

    return names


def _build_names_mapper(
    rownames: list[str], colnames: list[str]
) -> tuple[dict[str, str], list[str], dict[str, str], list[str]]:
    """
    Given the names of a DataFrame's rows and columns, returns a set of unique row
    and column names and mappers that convert to original names.

    A row or column name is replaced if it is duplicate among the rows of the inputs,
    among the columns of the inputs or between the rows and the columns.

    Parameters
    ----------
    rownames: list[str]
    colnames: list[str]

    Returns
    -------
    Tuple(Dict[str, str], List[str], Dict[str, str], List[str])

    rownames_mapper: dict[str, str]
        a dictionary with new row names as keys and original rownames as values
    unique_rownames: list[str]
        a list of rownames with duplicate names replaced by dummy names
    colnames_mapper: dict[str, str]
        a dictionary with new column names as keys and original column names as values
    unique_colnames: list[str]
        a list of column names with duplicate names replaced by dummy names

    """

    def get_duplicates(names):
        seen: set = set()
        return {name for name in names if name not in seen}

    shared_names = set(rownames).intersection(set(colnames))
    dup_names = get_duplicates(rownames) | get_duplicates(colnames) | shared_names

    rownames_mapper = {
        f"row_{i}": name for i, name in enumerate(rownames) if name in dup_names
    }
    unique_rownames = [
        f"row_{i}" if name in dup_names else name for i, name in enumerate(rownames)
    ]

    colnames_mapper = {
        f"col_{i}": name for i, name in enumerate(colnames) if name in dup_names
    }
    unique_colnames = [
        f"col_{i}" if name in dup_names else name for i, name in enumerate(colnames)
    ]

    return rownames_mapper, unique_rownames, colnames_mapper, unique_colnames
<|endoftext|>from pandas.core.reshape.concat import concat
from pandas.core.reshape.encoding import (
    from_dummies,
    get_dummies,
)
from pandas.core.reshape.melt import (
    lreshape,
    melt,
    wide_to_long,
)
from pandas.core.reshape.merge import (
    merge,
    merge_asof,
    merge_ordered,
)
from pandas.core.reshape.pivot import (
    crosstab,
    pivot,
    pivot_table,
)
from pandas.core.reshape.tile import (
    cut,
    qcut,
)

__all__ = [
    "concat",
    "crosstab",
    "cut",
    "from_dummies",
    "get_dummies",
    "lreshape",
    "melt",
    "merge",
    "merge_asof",
    "merge_ordered",
    "pivot",
    "pivot_table",
    "qcut",
    "wide_to_long",
]
<|endoftext|>from __future__ import annotations

import operator
from operator import (
    le,
    lt,
)
import textwrap
from typing import (
    TYPE_CHECKING,
    Literal,
    Union,
    overload,
)
import warnings

import numpy as np

from pandas._libs import lib
from pandas._libs.interval import (
    VALID_CLOSED,
    Interval,
    IntervalMixin,
    intervals_to_interval_bounds,
)
from pandas._libs.missing import NA
from pandas._typing import (
    ArrayLike,
    AxisInt,
    Dtype,
    FillnaOptions,
    IntervalClosedType,
    NpDtype,
    PositionalIndexer,
    ScalarIndexer,
    Self,
    SequenceIndexer,
    SortKind,
    TimeArrayLike,
    npt,
)
from pandas.compat.numpy import function as nv
from pandas.errors import IntCastingNaNError
from pandas.util._decorators import Appender
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.cast import (
    LossySetitemError,
    maybe_upcast_numeric_to_64bit,
)
from pandas.core.dtypes.common import (
    is_float_dtype,
    is_integer_dtype,
    is_list_like,
    is_object_dtype,
    is_scalar,
    is_string_dtype,
    needs_i8_conversion,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    IntervalDtype,
)
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCDatetimeIndex,
    ABCIntervalIndex,
    ABCPeriodIndex,
)
from pandas.core.dtypes.missing import (
    is_valid_na_for_dtype,
    isna,
    notna,
)

from pandas.core.algorithms import (
    isin,
    take,
    unique,
    value_counts_internal as value_counts,
)
from pandas.core.arrays import ArrowExtensionArray
from pandas.core.arrays.base import (
    ExtensionArray,
    _extension_array_shared_docs,
)
from pandas.core.arrays.datetimes import DatetimeArray
from pandas.core.arrays.timedeltas import TimedeltaArray
import pandas.core.common as com
from pandas.core.construction import (
    array as pd_array,
    ensure_wrapped_if_datetimelike,
    extract_array,
)
from pandas.core.indexers import check_array_indexer
from pandas.core.ops import (
    invalid_comparison,
    unpack_zerodim_and_defer,
)

if TYPE_CHECKING:
    from collections.abc import (
        Iterator,
        Sequence,
    )

    from pandas import (
        Index,
        Series,
    )


IntervalSide = Union[TimeArrayLike, np.ndarray]
IntervalOrNA = Union[Interval, float]

_interval_shared_docs: dict[str, str] = {}

_shared_docs_kwargs = {
    "klass": "IntervalArray",
    "qualname": "arrays.IntervalArray",
    "name": "",
}


_interval_shared_docs[
    "class"
] = """
%(summary)s

Parameters
----------
data : array-like (1-dimensional)
    Array-like (ndarray, :class:`DateTimeArray`, :class:`TimeDeltaArray`) containing
    Interval objects from which to build the %(klass)s.
closed : {'left', 'right', 'both', 'neither'}, default 'right'
    Whether the intervals are closed on the left-side, right-side, both or
    neither.
dtype : dtype or None, default None
    If None, dtype will be inferred.
copy : bool, default False
    Copy the input data.
%(name)s\
verify_integrity : bool, default True
    Verify that the %(klass)s is valid.

Attributes
----------
left
right
closed
mid
length
is_empty
is_non_overlapping_monotonic
%(extra_attributes)s\

Methods
-------
from_arrays
from_tuples
from_breaks
contains
overlaps
set_closed
to_tuples
%(extra_methods)s\

See Also
--------
Index : The base pandas Index type.
Interval : A bounded slice-like interval; the elements of an %(klass)s.
interval_range : Function to create a fixed frequency IntervalIndex.
cut : Bin values into discrete Intervals.
qcut : Bin values into equal-sized Intervals based on rank or sample quantiles.

Notes
-----
See the `user guide
<https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html#intervalindex>`__
for more.

%(examples)s\
"""


@Appender(
    _interval_shared_docs["class"]
    % {
        "klass": "IntervalArray",
        "summary": "Pandas array for interval data that are closed on the same side.",
        "name": "",
        "extra_attributes": "",
        "extra_methods": "",
        "examples": textwrap.dedent(
            """\
    Examples
    --------
    A new ``IntervalArray`` can be constructed directly from an array-like of
    ``Interval`` objects:

    >>> pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(1, 5)])
    <IntervalArray>
    [(0, 1], (1, 5]]
    Length: 2, dtype: interval[int64, right]

    It may also be constructed using one of the constructor
    methods: :meth:`IntervalArray.from_arrays`,
    :meth:`IntervalArray.from_breaks`, and :meth:`IntervalArray.from_tuples`.
    """
        ),
    }
)
class IntervalArray(IntervalMixin, ExtensionArray):
    can_hold_na = True
    _na_value = _fill_value = np.nan

    @property
    def ndim(self) -> Literal[1]:
        return 1

    # To make mypy recognize the fields
    _left: IntervalSide
    _right: IntervalSide
    _dtype: IntervalDtype

    # ---------------------------------------------------------------------
    # Constructors

    def __new__(
        cls,
        data,
        closed: IntervalClosedType | None = None,
        dtype: Dtype | None = None,
        copy: bool = False,
        verify_integrity: bool = True,
    ):
        data = extract_array(data, extract_numpy=True)

        if isinstance(data, cls):
            left: IntervalSide = data._left
            right: IntervalSide = data._right
            closed = closed or data.closed
            dtype = IntervalDtype(left.dtype, closed=closed)
        else:
            # don't allow scalars
            if is_scalar(data):
                msg = (
                    f"{cls.__name__}(...) must be called with a collection "
                    f"of some kind, {data} was passed"
                )
                raise TypeError(msg)

            # might need to convert empty or purely na data
            data = _maybe_convert_platform_interval(data)
            left, right, infer_closed = intervals_to_interval_bounds(
                data, validate_closed=closed is None
            )
            if left.dtype == object:
                left = lib.maybe_convert_objects(left)
                right = lib.maybe_convert_objects(right)
            closed = closed or infer_closed

            left, right, dtype = cls._ensure_simple_new_inputs(
                left,
                right,
                closed=closed,
                copy=copy,
                dtype=dtype,
            )

        if verify_integrity:
            cls._validate(left, right, dtype=dtype)

        return cls._simple_new(
            left,
            right,
            dtype=dtype,
        )

    @classmethod
    def _simple_new(
        cls,
        left: IntervalSide,
        right: IntervalSide,
        dtype: IntervalDtype,
    ) -> Self:
        result = IntervalMixin.__new__(cls)
        result._left = left
        result._right = right
        result._dtype = dtype

        return result

    @classmethod
    def _ensure_simple_new_inputs(
        cls,
        left,
        right,
        closed: IntervalClosedType | None = None,
        copy: bool = False,
        dtype: Dtype | None = None,
    ) -> tuple[IntervalSide, IntervalSide, IntervalDtype]:
        """Ensure correctness of input parameters for cls._simple_new."""
        from pandas.core.indexes.base import ensure_index

        left = ensure_index(left, copy=copy)
        left = maybe_upcast_numeric_to_64bit(left)

        right = ensure_index(right, copy=copy)
        right = maybe_upcast_numeric_to_64bit(right)

        if closed is None and isinstance(dtype, IntervalDtype):
            closed = dtype.closed

        closed = closed or "right"

        if dtype is not None:
            # GH 19262: dtype must be an IntervalDtype to override inferred
            dtype = pandas_dtype(dtype)
            if isinstance(dtype, IntervalDtype):
                if dtype.subtype is not None:
                    left = left.astype(dtype.subtype)
                    right = right.astype(dtype.subtype)
            else:
                msg = f"dtype must be an IntervalDtype, got {dtype}"
                raise TypeError(msg)

            if dtype.closed is None:
                # possibly loading an old pickle
                dtype = IntervalDtype(dtype.subtype, closed)
            elif closed != dtype.closed:
                raise ValueError("closed keyword does not match dtype.closed")

        # coerce dtypes to match if needed
        if is_float_dtype(left.dtype) and is_integer_dtype(right.dtype):
            right = right.astype(left.dtype)
        elif is_float_dtype(right.dtype) and is_integer_dtype(left.dtype):
            left = left.astype(right.dtype)

        if type(left) != type(right):
            msg = (
                f"must not have differing left [{type(left).__name__}] and "
                f"right [{type(right).__name__}] types"
            )
            raise ValueError(msg)
        if isinstance(left.dtype, CategoricalDtype) or is_string_dtype(left.dtype):
            # GH 19016
            msg = (
                "category, object, and string subtypes are not supported "
                "for IntervalArray"
            )
            raise TypeError(msg)
        if isinstance(left, ABCPeriodIndex):
            msg = "Period dtypes are not supported, use a PeriodIndex instead"
            raise ValueError(msg)
        if isinstance(left, ABCDatetimeIndex) and str(left.tz) != str(right.tz):
            msg = (
                "left and right must have the same time zone, got "
                f"'{left.tz}' and '{right.tz}'"
            )
            raise ValueError(msg)
        elif needs_i8_conversion(left.dtype) and left.unit != right.unit:
            # e.g. m8[s] vs m8[ms], try to cast to a common dtype GH#55714
            left_arr, right_arr = left._data._ensure_matching_resos(right._data)
            left = ensure_index(left_arr)
            right = ensure_index(right_arr)

        # For dt64/td64 we want DatetimeArray/TimedeltaArray instead of ndarray
        left = ensure_wrapped_if_datetimelike(left)
        left = extract_array(left, extract_numpy=True)
        right = ensure_wrapped_if_datetimelike(right)
        right = extract_array(right, extract_numpy=True)

        if isinstance(left, ArrowExtensionArray) or isinstance(
            right, ArrowExtensionArray
        ):
            pass
        else:
            lbase = getattr(left, "_ndarray", left)
            lbase = getattr(lbase, "_data", lbase).base
            rbase = getattr(right, "_ndarray", right)
            rbase = getattr(rbase, "_data", rbase).base
            if lbase is not None and lbase is rbase:
                # If these share data, then setitem could corrupt our IA
                right = right.copy()

        dtype = IntervalDtype(left.dtype, closed=closed)

        return left, right, dtype

    @classmethod
    def _from_sequence(
        cls,
        scalars,
        *,
        dtype: Dtype | None = None,
        copy: bool = False,
    ) -> Self:
        return cls(scalars, dtype=dtype, copy=copy)

    @classmethod
    def _from_factorized(cls, values: np.ndarray, original: IntervalArray) -> Self:
        return cls._from_sequence(values, dtype=original.dtype)

    _interval_shared_docs["from_breaks"] = textwrap.dedent(
        """
        Construct an %(klass)s from an array of splits.

        Parameters
        ----------
        breaks : array-like (1-dimensional)
            Left and right bounds for each interval.
        closed : {'left', 'right', 'both', 'neither'}, default 'right'
            Whether the intervals are closed on the left-side, right-side, both
            or neither.\
        %(name)s
        copy : bool, default False
            Copy the data.
        dtype : dtype or None, default None
            If None, dtype will be inferred.

        Returns
        -------
        %(klass)s

        See Also
        --------
        interval_range : Function to create a fixed frequency IntervalIndex.
        %(klass)s.from_arrays : Construct from a left and right array.
        %(klass)s.from_tuples : Construct from a sequence of tuples.

        %(examples)s\
        """
    )

    @classmethod
    @Appender(
        _interval_shared_docs["from_breaks"]
        % {
            "klass": "IntervalArray",
            "name": "",
            "examples": textwrap.dedent(
                """\
        Examples
        --------
        >>> pd.arrays.IntervalArray.from_breaks([0, 1, 2, 3])
        <IntervalArray>
        [(0, 1], (1, 2], (2, 3]]
        Length: 3, dtype: interval[int64, right]
        """
            ),
        }
    )
    def from_breaks(
        cls,
        breaks,
        closed: IntervalClosedType | None = "right",
        copy: bool = False,
        dtype: Dtype | None = None,
    ) -> Self:
        breaks = _maybe_convert_platform_interval(breaks)

        return cls.from_arrays(breaks[:-1], breaks[1:], closed, copy=copy, dtype=dtype)

    _interval_shared_docs["from_arrays"] = textwrap.dedent(
        """
        Construct from two arrays defining the left and right bounds.

        Parameters
        ----------
        left : array-like (1-dimensional)
            Left bounds for each interval.
        right : array-like (1-dimensional)
            Right bounds for each interval.
        closed : {'left', 'right', 'both', 'neither'}, default 'right'
            Whether the intervals are closed on the left-side, right-side, both
            or neither.\
        %(name)s
        copy : bool, default False
            Copy the data.
        dtype : dtype, optional
            If None, dtype will be inferred.

        Returns
        -------
        %(klass)s

        Raises
        ------
        ValueError
            When a value is missing in only one of `left` or `right`.
            When a value in `left` is greater than the corresponding value
            in `right`.

        See Also
        --------
        interval_range : Function to create a fixed frequency IntervalIndex.
        %(klass)s.from_breaks : Construct an %(klass)s from an array of
            splits.
        %(klass)s.from_tuples : Construct an %(klass)s from an
            array-like of tuples.

        Notes
        -----
        Each element of `left` must be less than or equal to the `right`
        element at the same position. If an element is missing, it must be
        missing in both `left` and `right`. A TypeError is raised when
        using an unsupported type for `left` or `right`. At the moment,
        'category', 'object', and 'string' subtypes are not supported.

        %(examples)s\
        """
    )

    @classmethod
    @Appender(
        _interval_shared_docs["from_arrays"]
        % {
            "klass": "IntervalArray",
            "name": "",
            "examples": textwrap.dedent(
                """\
        Examples
        --------
        >>> pd.arrays.IntervalArray.from_arrays([0, 1, 2], [1, 2, 3])
        <IntervalArray>
        [(0, 1], (1, 2], (2, 3]]
        Length: 3, dtype: interval[int64, right]
        """
            ),
        }
    )
    def from_arrays(
        cls,
        left,
        right,
        closed: IntervalClosedType | None = "right",
        copy: bool = False,
        dtype: Dtype | None = None,
    ) -> Self:
        left = _maybe_convert_platform_interval(left)
        right = _maybe_convert_platform_interval(right)

        left, right, dtype = cls._ensure_simple_new_inputs(
            left,
            right,
            closed=closed,
            copy=copy,
            dtype=dtype,
        )
        cls._validate(left, right, dtype=dtype)

        return cls._simple_new(left, right, dtype=dtype)

    _interval_shared_docs["from_tuples"] = textwrap.dedent(
        """
        Construct an %(klass)s from an array-like of tuples.

        Parameters
        ----------
        data : array-like (1-dimensional)
            Array of tuples.
        closed : {'left', 'right', 'both', 'neither'}, default 'right'
            Whether the intervals are closed on the left-side, right-side, both
            or neither.\
        %(name)s
        copy : bool, default False
            By-default copy the data, this is compat only and ignored.
        dtype : dtype or None, default None
            If None, dtype will be inferred.

        Returns
        -------
        %(klass)s

        See Also
        --------
        interval_range : Function to create a fixed frequency IntervalIndex.
        %(klass)s.from_arrays : Construct an %(klass)s from a left and
                                    right array.
        %(klass)s.from_breaks : Construct an %(klass)s from an array of
                                    splits.

        %(examples)s\
        """
    )

    @classmethod
    @Appender(
        _interval_shared_docs["from_tuples"]
        % {
            "klass": "IntervalArray",
            "name": "",
            "examples": textwrap.dedent(
                """\
        Examples
        --------
        >>> pd.arrays.IntervalArray.from_tuples([(0, 1), (1, 2)])
        <IntervalArray>
        [(0, 1], (1, 2]]
        Length: 2, dtype: interval[int64, right]
        """
            ),
        }
    )
    def from_tuples(
        cls,
        data,
        closed: IntervalClosedType | None = "right",
        copy: bool = False,
        dtype: Dtype | None = None,
    ) -> Self:
        if len(data):
            left, right = [], []
        else:
            # ensure that empty data keeps input dtype
            left = right = data

        for d in data:
            if not isinstance(d, tuple) and isna(d):
                lhs = rhs = np.nan
            else:
                name = cls.__name__
                try:
                    # need list of length 2 tuples, e.g. [(0, 1), (1, 2), ...]
                    lhs, rhs = d
                except ValueError as err:
                    msg = f"{name}.from_tuples requires tuples of length 2, got {d}"
                    raise ValueError(msg) from err
                except TypeError as err:
                    msg = f"{name}.from_tuples received an invalid item, {d}"
                    raise TypeError(msg) from err
            left.append(lhs)
            right.append(rhs)

        return cls.from_arrays(left, right, closed, copy=False, dtype=dtype)

    @classmethod
    def _validate(cls, left, right, dtype: IntervalDtype) -> None:
        """
        Verify that the IntervalArray is valid.

        Checks that

        * dtype is correct
        * left and right match lengths
        * left and right have the same missing values
        * left is always below right
        """
        if not isinstance(dtype, IntervalDtype):
            msg = f"invalid dtype: {dtype}"
            raise ValueError(msg)
        if len(left) != len(right):
            msg = "left and right must have the same length"
            raise ValueError(msg)
        left_mask = notna(left)
        right_mask = notna(right)
        if not (left_mask == right_mask).all():
            msg = (
                "missing values must be missing in the same "
                "location both left and right sides"
            )
            raise ValueError(msg)
        if not (left[left_mask] <= right[left_mask]).all():
            msg = "left side of interval must be <= right side"
            raise ValueError(msg)

    def _shallow_copy(self, left, right) -> Self:
        """
        Return a new IntervalArray with the replacement attributes

        Parameters
        ----------
        left : Index
            Values to be used for the left-side of the intervals.
        right : Index
            Values to be used for the right-side of the intervals.
        """
        dtype = IntervalDtype(left.dtype, closed=self.closed)
        left, right, dtype = self._ensure_simple_new_inputs(left, right, dtype=dtype)

        return self._simple_new(left, right, dtype=dtype)

    # ---------------------------------------------------------------------
    # Descriptive

    @property
    def dtype(self) -> IntervalDtype:
        return self._dtype

    @property
    def nbytes(self) -> int:
        return self.left.nbytes + self.right.nbytes

    @property
    def size(self) -> int:
        # Avoid materializing self.values
        return self.left.size

    # ---------------------------------------------------------------------
    # EA Interface

    def __iter__(self) -> Iterator:
        return iter(np.asarray(self))

    def __len__(self) -> int:
        return len(self._left)

    @overload
    def __getitem__(self, key: ScalarIndexer) -> IntervalOrNA:
        ...

    @overload
    def __getitem__(self, key: SequenceIndexer) -> Self:
        ...

    def __getitem__(self, key: PositionalIndexer) -> Self | IntervalOrNA:
        key = check_array_indexer(self, key)
        left = self._left[key]
        right = self._right[key]

        if not isinstance(left, (np.ndarray, ExtensionArray)):
            # scalar
            if is_scalar(left) and isna(left):
                return self._fill_value
            return Interval(left, right, self.closed)
        if np.ndim(left) > 1:
            # GH#30588 multi-dimensional indexer disallowed
            raise ValueError("multi-dimensional indexing not allowed")
        # Argument 2 to "_simple_new" of "IntervalArray" has incompatible type
        # "Union[Period, Timestamp, Timedelta, NaTType, DatetimeArray, TimedeltaArray,
        # ndarray[Any, Any]]"; expected "Union[Union[DatetimeArray, TimedeltaArray],
        # ndarray[Any, Any]]"
        return self._simple_new(left, right, dtype=self.dtype)  # type: ignore[arg-type]

    def __setitem__(self, key, value) -> None:
        value_left, value_right = self._validate_setitem_value(value)
        key = check_array_indexer(self, key)

        self._left[key] = value_left
        self._right[key] = value_right

    def _cmp_method(self, other, op):
        # ensure pandas array for list-like and eliminate non-interval scalars
        if is_list_like(other):
            if len(self) != len(other):
                raise ValueError("Lengths must match to compare")
            other = pd_array(other)
        elif not isinstance(other, Interval):
            # non-interval scalar -> no matches
            if other is NA:
                # GH#31882
                from pandas.core.arrays import BooleanArray

                arr = np.empty(self.shape, dtype=bool)
                mask = np.ones(self.shape, dtype=bool)
                return BooleanArray(arr, mask)
            return invalid_comparison(self, other, op)

        # determine the dtype of the elements we want to compare
        if isinstance(other, Interval):
            other_dtype = pandas_dtype("interval")
        elif not isinstance(other.dtype, CategoricalDtype):
            other_dtype = other.dtype
        else:
            # for categorical defer to categories for dtype
            other_dtype = other.categories.dtype

            # extract intervals if we have interval categories with matching closed
            if isinstance(other_dtype, IntervalDtype):
                if self.closed != other.categories.closed:
                    return invalid_comparison(self, other, op)

                other = other.categories._values.take(
                    other.codes, allow_fill=True, fill_value=other.categories._na_value
                )

        # interval-like -> need same closed and matching endpoints
        if isinstance(other_dtype, IntervalDtype):
            if self.closed != other.closed:
                return invalid_comparison(self, other, op)
            elif not isinstance(other, Interval):
                other = type(self)(other)

            if op is operator.eq:
                return (self._left == other.left) & (self._right == other.right)
            elif op is operator.ne:
                return (self._left != other.left) | (self._right != other.right)
            elif op is operator.gt:
                return (self._left > other.left) | (
                    (self._left == other.left) & (self._right > other.right)
                )
            elif op is operator.ge:
                return (self == other) | (self > other)
            elif op is operator.lt:
                return (self._left < other.left) | (
                    (self._left == other.left) & (self._right < other.right)
                )
            else:
                # operator.lt
                return (self == other) | (self < other)

        # non-interval/non-object dtype -> no matches
        if not is_object_dtype(other_dtype):
            return invalid_comparison(self, other, op)

        # object dtype -> iteratively check for intervals
        result = np.zeros(len(self), dtype=bool)
        for i, obj in enumerate(other):
            try:
                result[i] = op(self[i], obj)
            except TypeError:
                if obj is NA:
                    # comparison with np.nan returns NA
                    # github.com/pandas-dev/pandas/pull/37124#discussion_r509095092
                    result = result.astype(object)
                    result[i] = NA
                else:
                    raise
        return result

    @unpack_zerodim_and_defer("__eq__")
    def __eq__(self, other):
        return self._cmp_method(other, operator.eq)

    @unpack_zerodim_and_defer("__ne__")
    def __ne__(self, other):
        return self._cmp_method(other, operator.ne)

    @unpack_zerodim_and_defer("__gt__")
    def __gt__(self, other):
        return self._cmp_method(other, operator.gt)

    @unpack_zerodim_and_defer("__ge__")
    def __ge__(self, other):
        return self._cmp_method(other, operator.ge)

    @unpack_zerodim_and_defer("__lt__")
    def __lt__(self, other):
        return self._cmp_method(other, operator.lt)

    @unpack_zerodim_and_defer("__le__")
    def __le__(self, other):
        return self._cmp_method(other, operator.le)

    def argsort(
        self,
        *,
        ascending: bool = True,
        kind: SortKind = "quicksort",
        na_position: str = "last",
        **kwargs,
    ) -> np.ndarray:
        ascending = nv.validate_argsort_with_ascending(ascending, (), kwargs)

        if ascending and kind == "quicksort" and na_position == "last":
            # TODO: in an IntervalIndex we can reuse the cached
            #  IntervalTree.left_sorter
            return np.lexsort((self.right, self.left))

        # TODO: other cases we can use lexsort for?  much more performant.
        return super().argsort(
            ascending=ascending, kind=kind, na_position=na_position, **kwargs
        )

    def min(self, *, axis: AxisInt | None = None, skipna: bool = True) -> IntervalOrNA:
        nv.validate_minmax_axis(axis, self.ndim)

        if not len(self):
            return self._na_value

        mask = self.isna()
        if mask.any():
            if not skipna:
                return self._na_value
            obj = self[~mask]
        else:
            obj = self

        indexer = obj.argsort()[0]
        return obj[indexer]

    def max(self, *, axis: AxisInt | None = None, skipna: bool = True) -> IntervalOrNA:
        nv.validate_minmax_axis(axis, self.ndim)

        if not len(self):
            return self._na_value

        mask = self.isna()
        if mask.any():
            if not skipna:
                return self._na_value
            obj = self[~mask]
        else:
            obj = self

        indexer = obj.argsort()[-1]
        return obj[indexer]

    def _pad_or_backfill(  # pylint: disable=useless-parent-delegation
        self,
        *,
        method: FillnaOptions,
        limit: int | None = None,
        limit_area: Literal["inside", "outside"] | None = None,
        copy: bool = True,
    ) -> Self:
        # TODO(3.0): after EA.fillna 'method' deprecation is enforced, we can remove
        #  this method entirely.
        return super()._pad_or_backfill(
            method=method, limit=limit, limit_area=limit_area, copy=copy
        )

    def fillna(
        self, value=None, method=None, limit: int | None = None, copy: bool = True
    ) -> Self:
        """
        Fill NA/NaN values using the specified method.

        Parameters
        ----------
        value : scalar, dict, Series
            If a scalar value is passed it is used to fill all missing values.
            Alternatively, a Series or dict can be used to fill in different
            values for each index. The value should not be a list. The
            value(s) passed should be either Interval objects or NA/NaN.
        method : {'backfill', 'bfill', 'pad', 'ffill', None}, default None
            (Not implemented yet for IntervalArray)
            Method to use for filling holes in reindexed Series
        limit : int, default None
            (Not implemented yet for IntervalArray)
            If method is specified, this is the maximum number of consecutive
            NaN values to forward/backward fill. In other words, if there is
            a gap with more than this number of consecutive NaNs, it will only
            be partially filled. If method is not specified, this is the
            maximum number of entries along the entire axis where NaNs will be
            filled.
        copy : bool, default True
            Whether to make a copy of the data before filling. If False, then
            the original should be modified and no new memory should be allocated.
            For ExtensionArray subclasses that cannot do this, it is at the
            author's discretion whether to ignore "copy=False" or to raise.

        Returns
        -------
        filled : IntervalArray with NA/NaN filled
        """
        if copy is False:
            raise NotImplementedError
        if method is not None:
            return super().fillna(value=value, method=method, limit=limit)

        value_left, value_right = self._validate_scalar(value)

        left = self.left.fillna(value=value_left)
        right = self.right.fillna(value=value_right)
        return self._shallow_copy(left, right)

    def astype(self, dtype, copy: bool = True):
        """
        Cast to an ExtensionArray or NumPy array with dtype 'dtype'.

        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to which the array is cast.

        copy : bool, default True
            Whether to copy the data, even if not necessary. If False,
            a copy is made only if the old dtype does not match the
            new dtype.

        Returns
        -------
        array : ExtensionArray or ndarray
            ExtensionArray or NumPy ndarray with 'dtype' for its dtype.
        """
        from pandas import Index

        if dtype is not None:
            dtype = pandas_dtype(dtype)

        if isinstance(dtype, IntervalDtype):
            if dtype == self.dtype:
                return self.copy() if copy else self

            if is_float_dtype(self.dtype.subtype) and needs_i8_conversion(
                dtype.subtype
            ):
                # This is allowed on the Index.astype but we disallow it here
                msg = (
                    f"Cannot convert {self.dtype} to {dtype}; subtypes are incompatible"
                )
                raise TypeError(msg)

            # need to cast to different subtype
            try:
                # We need to use Index rules for astype to prevent casting
                #  np.nan entries to int subtypes
                new_left = Index(self._left, copy=False).astype(dtype.subtype)
                new_right = Index(self._right, copy=False).astype(dtype.subtype)
            except IntCastingNaNError:
                # e.g test_subtype_integer
                raise
            except (TypeError, ValueError) as err:
                # e.g. test_subtype_integer_errors f8->u8 can be lossy
                #  and raises ValueError
                msg = (
                    f"Cannot convert {self.dtype} to {dtype}; subtypes are incompatible"
                )
                raise TypeError(msg) from err
            return self._shallow_copy(new_left, new_right)
        else:
            try:
                return super().astype(dtype, copy=copy)
            except (TypeError, ValueError) as err:
                msg = f"Cannot cast {type(self).__name__} to dtype {dtype}"
                raise TypeError(msg) from err

    def equals(self, other) -> bool:
        if type(self) != type(other):
            return False

        return bool(
            self.closed == other.closed
            and self.left.equals(other.left)
            and self.right.equals(other.right)
        )

    @classmethod
    def _concat_same_type(cls, to_concat: Sequence[IntervalArray]) -> Self:
        """
        Concatenate multiple IntervalArray

        Parameters
        ----------
        to_concat : sequence of IntervalArray

        Returns
        -------
        IntervalArray
        """
        closed_set = {interval.closed for interval in to_concat}
        if len(closed_set) != 1:
            raise ValueError("Intervals must all be closed on the same side.")
        closed = closed_set.pop()

        left: IntervalSide = np.concatenate([interval.left for interval in to_concat])
        right: IntervalSide = np.concatenate([interval.right for interval in to_concat])

        left, right, dtype = cls._ensure_simple_new_inputs(left, right, closed=closed)

        return cls._simple_new(left, right, dtype=dtype)

    def copy(self) -> Self:
        """
        Return a copy of the array.

        Returns
        -------
        IntervalArray
        """
        left = self._left.copy()
        right = self._right.copy()
        dtype = self.dtype
        return self._simple_new(left, right, dtype=dtype)

    def isna(self) -> np.ndarray:
        return isna(self._left)

    def shift(self, periods: int = 1, fill_value: object = None) -> IntervalArray:
        if not len(self) or periods == 0:
            return self.copy()

        self._validate_scalar(fill_value)

        # ExtensionArray.shift doesn't work for two reasons
        # 1. IntervalArray.dtype.na_value may not be correct for the dtype.
        # 2. IntervalArray._from_sequence only accepts NaN for missing values,
        #    not other values like NaT

        empty_len = min(abs(periods), len(self))
        if isna(fill_value):
            from pandas import Index

            fill_value = Index(self._left, copy=False)._na_value
            empty = IntervalArray.from_breaks([fill_value] * (empty_len + 1))
        else:
            empty = self._from_sequence([fill_value] * empty_len, dtype=self.dtype)

        if periods > 0:
            a = empty
            b = self[:-periods]
        else:
            a = self[abs(periods) :]
            b = empty
        return self._concat_same_type([a, b])

    def take(
        self,
        indices,
        *,
        allow_fill: bool = False,
        fill_value=None,
        axis=None,
        **kwargs,
    ) -> Self:
        """
        Take elements from the IntervalArray.

        Parameters
        ----------
        indices : sequence of integers
            Indices to be taken.

        allow_fill : bool, default False
            How to handle negative values in `indices`.

            * False: negative values in `indices` indicate positional indices
              from the right (the default). This is similar to
              :func:`numpy.take`.

            * True: negative values in `indices` indicate
              missing values. These values are set to `fill_value`. Any other
              other negative values raise a ``ValueError``.

        fill_value : Interval or NA, optional
            Fill value to use for NA-indices when `allow_fill` is True.
            This may be ``None``, in which case the default NA value for
            the type, ``self.dtype.na_value``, is used.

            For many ExtensionArrays, there will be two representations of
            `fill_value`: a user-facing "boxed" scalar, and a low-level
            physical NA value. `fill_value` should be the user-facing version,
            and the implementation should handle translating that to the
            physical version for processing the take if necessary.

        axis : any, default None
            Present for compat with IntervalIndex; does nothing.

        Returns
        -------
        IntervalArray

        Raises
        ------
        IndexError
            When the indices are out of bounds for the array.
        ValueError
            When `indices` contains negative values other than ``-1``
            and `allow_fill` is True.
        """
        nv.validate_take((), kwargs)

        fill_left = fill_right = fill_value
        if allow_fill:
            fill_left, fill_right = self._validate_scalar(fill_value)

        left_take = take(
            self._left, indices, allow_fill=allow_fill, fill_value=fill_left
        )
        right_take = take(
            self._right, indices, allow_fill=allow_fill, fill_value=fill_right
        )

        return self._shallow_copy(left_take, right_take)

    def _validate_listlike(self, value):
        # list-like of intervals
        try:
            array = IntervalArray(value)
            self._check_closed_matches(array, name="value")
            value_left, value_right = array.left, array.right
        except TypeError as err:
            # wrong type: not interval or NA
            msg = f"'value' should be an interval type, got {type(value)} instead."
            raise TypeError(msg) from err

        try:
            self.left._validate_fill_value(value_left)
        except (LossySetitemError, TypeError) as err:
            msg = (
                "'value' should be a compatible interval type, "
                f"got {type(value)} instead."
            )
            raise TypeError(msg) from err

        return value_left, value_right

    def _validate_scalar(self, value):
        if isinstance(value, Interval):
            self._check_closed_matches(value, name="value")
            left, right = value.left, value.right
            # TODO: check subdtype match like _validate_setitem_value?
        elif is_valid_na_for_dtype(value, self.left.dtype):
            # GH#18295
            left = right = self.left._na_value
        else:
            raise TypeError(
                "can only insert Interval objects and NA into an IntervalArray"
            )
        return left, right

    def _validate_setitem_value(self, value):
        if is_valid_na_for_dtype(value, self.left.dtype):
            # na value: need special casing to set directly on numpy arrays
            value = self.left._na_value
            if is_integer_dtype(self.dtype.subtype):
                # can't set NaN on a numpy integer array
                # GH#45484 TypeError, not ValueError, matches what we get with
                #  non-NA un-holdable value.
                raise TypeError("Cannot set float NaN to integer-backed IntervalArray")
            value_left, value_right = value, value

        elif isinstance(value, Interval):
            # scalar interval
            self._check_closed_matches(value, name="value")
            value_left, value_right = value.left, value.right
            self.left._validate_fill_value(value_left)
            self.left._validate_fill_value(value_right)

        else:
            return self._validate_listlike(value)

        return value_left, value_right

    def value_counts(self, dropna: bool = True) -> Series:
        """
        Returns a Series containing counts of each interval.

        Parameters
        ----------
        dropna : bool, default True
            Don't include counts of NaN.

        Returns
        -------
        counts : Series

        See Also
        --------
        Series.value_counts
        """
        # TODO: implement this is a non-naive way!
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                "The behavior of value_counts with object-dtype is deprecated",
                category=FutureWarning,
            )
            result = value_counts(np.asarray(self), dropna=dropna)
            # Once the deprecation is enforced, we will need to do
            #  `result.index = result.index.astype(self.dtype)`
        return result

    # ---------------------------------------------------------------------
    # Rendering Methods

    def _formatter(self, boxed: bool = False):
        # returning 'str' here causes us to render as e.g. "(0, 1]" instead of
        #  "Interval(0, 1, closed='right')"
        return str

    # ---------------------------------------------------------------------
    # Vectorized Interval Properties/Attributes

    @property
    def left(self) -> Index:
        """
        Return the left endpoints of each Interval in the IntervalArray as an Index.

        Examples
        --------

        >>> interv_arr = pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(2, 5)])
        >>> interv_arr
        <IntervalArray>
        [(0, 1], (2, 5]]
        Length: 2, dtype: interval[int64, right]
        >>> interv_arr.left
        Index([0, 2], dtype='int64')
        """
        from pandas import Index

        return Index(self._left, copy=False)

    @property
    def right(self) -> Index:
        """
        Return the right endpoints of each Interval in the IntervalArray as an Index.

        Examples
        --------

        >>> interv_arr = pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(2, 5)])
        >>> interv_arr
        <IntervalArray>
        [(0, 1], (2, 5]]
        Length: 2, dtype: interval[int64, right]
        >>> interv_arr.right
        Index([1, 5], dtype='int64')
        """
        from pandas import Index

        return Index(self._right, copy=False)

    @property
    def length(self) -> Index:
        """
        Return an Index with entries denoting the length of each Interval.

        Examples
        --------

        >>> interv_arr = pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(1, 5)])
        >>> interv_arr
        <IntervalArray>
        [(0, 1], (1, 5]]
        Length: 2, dtype: interval[int64, right]
        >>> interv_arr.length
        Index([1, 4], dtype='int64')
        """
        return self.right - self.left

    @property
    def mid(self) -> Index:
        """
        Return the midpoint of each Interval in the IntervalArray as an Index.

        Examples
        --------

        >>> interv_arr = pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(1, 5)])
        >>> interv_arr
        <IntervalArray>
        [(0, 1], (1, 5]]
        Length: 2, dtype: interval[int64, right]
        >>> interv_arr.mid
