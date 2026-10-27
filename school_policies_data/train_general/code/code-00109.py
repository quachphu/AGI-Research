        else:
            idx_names = self._get_level_names()

        if not allow_duplicates and len(set(idx_names)) != len(idx_names):
            raise ValueError(
                "Cannot create duplicate column labels if allow_duplicates is False"
            )

        # Guarantee resulting column order - PY36+ dict maintains insertion order
        result = DataFrame(
            {level: self._get_level_values(level) for level in range(len(self.levels))},
            copy=False,
        )
        result.columns = idx_names

        if index:
            result.index = self
        return result

    # error: Return type "Index" of "to_flat_index" incompatible with return type
    # "MultiIndex" in supertype "Index"
    def to_flat_index(self) -> Index:  # type: ignore[override]
        """
        Convert a MultiIndex to an Index of Tuples containing the level values.

        Returns
        -------
        pd.Index
            Index with the MultiIndex data represented in Tuples.

        See Also
        --------
        MultiIndex.from_tuples : Convert flat index back to MultiIndex.

        Notes
        -----
        This method will simply return the caller if called by anything other
        than a MultiIndex.

        Examples
        --------
        >>> index = pd.MultiIndex.from_product(
        ...     [['foo', 'bar'], ['baz', 'qux']],
        ...     names=['a', 'b'])
        >>> index.to_flat_index()
        Index([('foo', 'baz'), ('foo', 'qux'),
               ('bar', 'baz'), ('bar', 'qux')],
              dtype='object')
        """
        return Index(self._values, tupleize_cols=False)

    def _is_lexsorted(self) -> bool:
        """
        Return True if the codes are lexicographically sorted.

        Returns
        -------
        bool

        Examples
        --------
        In the below examples, the first level of the MultiIndex is sorted because
        a<b<c, so there is no need to look at the next level.

        >>> pd.MultiIndex.from_arrays([['a', 'b', 'c'],
        ...                            ['d', 'e', 'f']])._is_lexsorted()
        True
        >>> pd.MultiIndex.from_arrays([['a', 'b', 'c'],
        ...                            ['d', 'f', 'e']])._is_lexsorted()
        True

        In case there is a tie, the lexicographical sorting looks
        at the next level of the MultiIndex.

        >>> pd.MultiIndex.from_arrays([[0, 1, 1], ['a', 'b', 'c']])._is_lexsorted()
        True
        >>> pd.MultiIndex.from_arrays([[0, 1, 1], ['a', 'c', 'b']])._is_lexsorted()
        False
        >>> pd.MultiIndex.from_arrays([['a', 'a', 'b', 'b'],
        ...                            ['aa', 'bb', 'aa', 'bb']])._is_lexsorted()
        True
        >>> pd.MultiIndex.from_arrays([['a', 'a', 'b', 'b'],
        ...                            ['bb', 'aa', 'aa', 'bb']])._is_lexsorted()
        False
        """
        return self._lexsort_depth == self.nlevels

    @cache_readonly
    def _lexsort_depth(self) -> int:
        """
        Compute and return the lexsort_depth, the number of levels of the
        MultiIndex that are sorted lexically

        Returns
        -------
        int
        """
        if self.sortorder is not None:
            return self.sortorder
        return _lexsort_depth(self.codes, self.nlevels)

    def _sort_levels_monotonic(self, raise_if_incomparable: bool = False) -> MultiIndex:
        """
        This is an *internal* function.

        Create a new MultiIndex from the current to monotonically sorted
        items IN the levels. This does not actually make the entire MultiIndex
        monotonic, JUST the levels.

        The resulting MultiIndex will have the same outward
        appearance, meaning the same .values and ordering. It will also
        be .equals() to the original.

        Returns
        -------
        MultiIndex

        Examples
        --------
        >>> mi = pd.MultiIndex(levels=[['a', 'b'], ['bb', 'aa']],
        ...                    codes=[[0, 0, 1, 1], [0, 1, 0, 1]])
        >>> mi
        MultiIndex([('a', 'bb'),
                    ('a', 'aa'),
                    ('b', 'bb'),
                    ('b', 'aa')],
                   )

        >>> mi.sort_values()
        MultiIndex([('a', 'aa'),
                    ('a', 'bb'),
                    ('b', 'aa'),
                    ('b', 'bb')],
                   )
        """
        if self._is_lexsorted() and self.is_monotonic_increasing:
            return self

        new_levels = []
        new_codes = []

        for lev, level_codes in zip(self.levels, self.codes):
            if not lev.is_monotonic_increasing:
                try:
                    # indexer to reorder the levels
                    indexer = lev.argsort()
                except TypeError:
                    if raise_if_incomparable:
                        raise
                else:
                    lev = lev.take(indexer)

                    # indexer to reorder the level codes
                    indexer = ensure_platform_int(indexer)
                    ri = lib.get_reverse_indexer(indexer, len(indexer))
                    level_codes = algos.take_nd(ri, level_codes, fill_value=-1)

            new_levels.append(lev)
            new_codes.append(level_codes)

        return MultiIndex(
            new_levels,
            new_codes,
            names=self.names,
            sortorder=self.sortorder,
            verify_integrity=False,
        )

    def remove_unused_levels(self) -> MultiIndex:
        """
        Create new MultiIndex from current that removes unused levels.

        Unused level(s) means levels that are not expressed in the
        labels. The resulting MultiIndex will have the same outward
        appearance, meaning the same .values and ordering. It will
        also be .equals() to the original.

        Returns
        -------
        MultiIndex

        Examples
        --------
        >>> mi = pd.MultiIndex.from_product([range(2), list('ab')])
        >>> mi
        MultiIndex([(0, 'a'),
                    (0, 'b'),
                    (1, 'a'),
                    (1, 'b')],
                   )

        >>> mi[2:]
        MultiIndex([(1, 'a'),
                    (1, 'b')],
                   )

        The 0 from the first level is not represented
        and can be removed

        >>> mi2 = mi[2:].remove_unused_levels()
        >>> mi2.levels
        FrozenList([[1], ['a', 'b']])
        """
        new_levels = []
        new_codes = []

        changed = False
        for lev, level_codes in zip(self.levels, self.codes):
            # Since few levels are typically unused, bincount() is more
            # efficient than unique() - however it only accepts positive values
            # (and drops order):
            uniques = np.where(np.bincount(level_codes + 1) > 0)[0] - 1
            has_na = int(len(uniques) and (uniques[0] == -1))

            if len(uniques) != len(lev) + has_na:
                if lev.isna().any() and len(uniques) == len(lev):
                    break
                # We have unused levels
                changed = True

                # Recalculate uniques, now preserving order.
                # Can easily be cythonized by exploiting the already existing
                # "uniques" and stop parsing "level_codes" when all items
                # are found:
                uniques = algos.unique(level_codes)
                if has_na:
                    na_idx = np.where(uniques == -1)[0]
                    # Just ensure that -1 is in first position:
                    uniques[[0, na_idx[0]]] = uniques[[na_idx[0], 0]]

                # codes get mapped from uniques to 0:len(uniques)
                # -1 (if present) is mapped to last position
                code_mapping = np.zeros(len(lev) + has_na)
                # ... and reassigned value -1:
                code_mapping[uniques] = np.arange(len(uniques)) - has_na

                level_codes = code_mapping[level_codes]

                # new levels are simple
                lev = lev.take(uniques[has_na:])

            new_levels.append(lev)
            new_codes.append(level_codes)

        result = self.view()

        if changed:
            result._reset_identity()
            result._set_levels(new_levels, validate=False)
            result._set_codes(new_codes, validate=False)

        return result

    # --------------------------------------------------------------------
    # Pickling Methods

    def __reduce__(self):
        """Necessary for making this object picklable"""
        d = {
            "levels": list(self.levels),
            "codes": list(self.codes),
            "sortorder": self.sortorder,
            "names": list(self.names),
        }
        return ibase._new_Index, (type(self), d), None

    # --------------------------------------------------------------------

    def __getitem__(self, key):
        if is_scalar(key):
            key = com.cast_scalar_indexer(key)

            retval = []
            for lev, level_codes in zip(self.levels, self.codes):
                if level_codes[key] == -1:
                    retval.append(np.nan)
                else:
                    retval.append(lev[level_codes[key]])

            return tuple(retval)
        else:
            # in general cannot be sure whether the result will be sorted
            sortorder = None
            if com.is_bool_indexer(key):
                key = np.asarray(key, dtype=bool)
                sortorder = self.sortorder
            elif isinstance(key, slice):
                if key.step is None or key.step > 0:
                    sortorder = self.sortorder
            elif isinstance(key, Index):
                key = np.asarray(key)

            new_codes = [level_codes[key] for level_codes in self.codes]

            return MultiIndex(
                levels=self.levels,
                codes=new_codes,
                names=self.names,
                sortorder=sortorder,
                verify_integrity=False,
            )

    def _getitem_slice(self: MultiIndex, slobj: slice) -> MultiIndex:
        """
        Fastpath for __getitem__ when we know we have a slice.
        """
        sortorder = None
        if slobj.step is None or slobj.step > 0:
            sortorder = self.sortorder

        new_codes = [level_codes[slobj] for level_codes in self.codes]

        return type(self)(
            levels=self.levels,
            codes=new_codes,
            names=self._names,
            sortorder=sortorder,
            verify_integrity=False,
        )

    @Appender(_index_shared_docs["take"] % _index_doc_kwargs)
    def take(
        self: MultiIndex,
        indices,
        axis: Axis = 0,
        allow_fill: bool = True,
        fill_value=None,
        **kwargs,
    ) -> MultiIndex:
        nv.validate_take((), kwargs)
        indices = ensure_platform_int(indices)

        # only fill if we are passing a non-None fill_value
        allow_fill = self._maybe_disallow_fill(allow_fill, fill_value, indices)

        na_value = -1

        taken = [lab.take(indices) for lab in self.codes]
        if allow_fill:
            mask = indices == -1
            if mask.any():
                masked = []
                for new_label in taken:
                    label_values = new_label
                    label_values[mask] = na_value
                    masked.append(np.asarray(label_values))
                taken = masked

        return MultiIndex(
            levels=self.levels, codes=taken, names=self.names, verify_integrity=False
        )

    def append(self, other):
        """
        Append a collection of Index options together.

        Parameters
        ----------
        other : Index or list/tuple of indices

        Returns
        -------
        Index
            The combined index.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([['a'], ['b']])
        >>> mi
        MultiIndex([('a', 'b')],
                   )
        >>> mi.append(mi)
        MultiIndex([('a', 'b'), ('a', 'b')],
                   )
        """
        if not isinstance(other, (list, tuple)):
            other = [other]

        if all(
            (isinstance(o, MultiIndex) and o.nlevels >= self.nlevels) for o in other
        ):
            codes = []
            levels = []
            names = []
            for i in range(self.nlevels):
                level_values = self.levels[i]
                for mi in other:
                    level_values = level_values.union(mi.levels[i])
                level_codes = [
                    recode_for_categories(
                        mi.codes[i], mi.levels[i], level_values, copy=False
                    )
                    for mi in ([self, *other])
                ]
                level_name = self.names[i]
                if any(mi.names[i] != level_name for mi in other):
                    level_name = None
                codes.append(np.concatenate(level_codes))
                levels.append(level_values)
                names.append(level_name)
            return MultiIndex(
                codes=codes, levels=levels, names=names, verify_integrity=False
            )

        to_concat = (self._values,) + tuple(k._values for k in other)
        new_tuples = np.concatenate(to_concat)

        # if all(isinstance(x, MultiIndex) for x in other):
        try:
            # We only get here if other contains at least one index with tuples,
            # setting names to None automatically
            return MultiIndex.from_tuples(new_tuples)
        except (TypeError, IndexError):
            return Index(new_tuples)

    def argsort(
        self, *args, na_position: str = "last", **kwargs
    ) -> npt.NDArray[np.intp]:
        target = self._sort_levels_monotonic(raise_if_incomparable=True)
        keys = [lev.codes for lev in target._get_codes_for_sorting()]
        return lexsort_indexer(keys, na_position=na_position, codes_given=True)

    @Appender(_index_shared_docs["repeat"] % _index_doc_kwargs)
    def repeat(self, repeats: int, axis=None) -> MultiIndex:
        nv.validate_repeat((), {"axis": axis})
        # error: Incompatible types in assignment (expression has type "ndarray",
        # variable has type "int")
        repeats = ensure_platform_int(repeats)  # type: ignore[assignment]
        return MultiIndex(
            levels=self.levels,
            codes=[
                level_codes.view(np.ndarray).astype(np.intp, copy=False).repeat(repeats)
                for level_codes in self.codes
            ],
            names=self.names,
            sortorder=self.sortorder,
            verify_integrity=False,
        )

    # error: Signature of "drop" incompatible with supertype "Index"
    def drop(  # type: ignore[override]
        self,
        codes,
        level: Index | np.ndarray | Iterable[Hashable] | None = None,
        errors: IgnoreRaise = "raise",
    ) -> MultiIndex:
        """
        Make a new :class:`pandas.MultiIndex` with the passed list of codes deleted.

        Parameters
        ----------
        codes : array-like
            Must be a list of tuples when ``level`` is not specified.
        level : int or level name, default None
        errors : str, default 'raise'

        Returns
        -------
        MultiIndex

        Examples
        --------
        >>> idx = pd.MultiIndex.from_product([(0, 1, 2), ('green', 'purple')],
        ...                                  names=["number", "color"])
        >>> idx
        MultiIndex([(0,  'green'),
                    (0, 'purple'),
                    (1,  'green'),
                    (1, 'purple'),
                    (2,  'green'),
                    (2, 'purple')],
                   names=['number', 'color'])
        >>> idx.drop([(1, 'green'), (2, 'purple')])
        MultiIndex([(0,  'green'),
                    (0, 'purple'),
                    (1, 'purple'),
                    (2,  'green')],
                   names=['number', 'color'])

        We can also drop from a specific level.

        >>> idx.drop('green', level='color')
        MultiIndex([(0, 'purple'),
                    (1, 'purple'),
                    (2, 'purple')],
                   names=['number', 'color'])

        >>> idx.drop([1, 2], level=0)
        MultiIndex([(0,  'green'),
                    (0, 'purple')],
                   names=['number', 'color'])
        """
        if level is not None:
            return self._drop_from_level(codes, level, errors)

        if not isinstance(codes, (np.ndarray, Index)):
            try:
                codes = com.index_labels_to_array(codes, dtype=np.dtype("object"))
            except ValueError:
                pass

        inds = []
        for level_codes in codes:
            try:
                loc = self.get_loc(level_codes)
                # get_loc returns either an integer, a slice, or a boolean
                # mask
                if isinstance(loc, int):
                    inds.append(loc)
                elif isinstance(loc, slice):
                    step = loc.step if loc.step is not None else 1
                    inds.extend(range(loc.start, loc.stop, step))
                elif com.is_bool_indexer(loc):
                    if self._lexsort_depth == 0:
                        warnings.warn(
                            "dropping on a non-lexsorted multi-index "
                            "without a level parameter may impact performance.",
                            PerformanceWarning,
                            stacklevel=find_stack_level(),
                        )
                    loc = loc.nonzero()[0]
                    inds.extend(loc)
                else:
                    msg = f"unsupported indexer of type {type(loc)}"
                    raise AssertionError(msg)
            except KeyError:
                if errors != "ignore":
                    raise

        return self.delete(inds)

    def _drop_from_level(
        self, codes, level, errors: IgnoreRaise = "raise"
    ) -> MultiIndex:
        codes = com.index_labels_to_array(codes)
        i = self._get_level_number(level)
        index = self.levels[i]
        values = index.get_indexer(codes)
        # If nan should be dropped it will equal -1 here. We have to check which values
        # are not nan and equal -1, this means they are missing in the index
        nan_codes = isna(codes)
        values[(np.equal(nan_codes, False)) & (values == -1)] = -2
        if index.shape[0] == self.shape[0]:
            values[np.equal(nan_codes, True)] = -2

        not_found = codes[values == -2]
        if len(not_found) != 0 and errors != "ignore":
            raise KeyError(f"labels {not_found} not found in level")
        mask = ~algos.isin(self.codes[i], values)

        return self[mask]

    def swaplevel(self, i=-2, j=-1) -> MultiIndex:
        """
        Swap level i with level j.

        Calling this method does not change the ordering of the values.

        Parameters
        ----------
        i : int, str, default -2
            First level of index to be swapped. Can pass level name as string.
            Type of parameters can be mixed.
        j : int, str, default -1
            Second level of index to be swapped. Can pass level name as string.
            Type of parameters can be mixed.

        Returns
        -------
        MultiIndex
            A new MultiIndex.

        See Also
        --------
        Series.swaplevel : Swap levels i and j in a MultiIndex.
        DataFrame.swaplevel : Swap levels i and j in a MultiIndex on a
            particular axis.

        Examples
        --------
        >>> mi = pd.MultiIndex(levels=[['a', 'b'], ['bb', 'aa']],
        ...                    codes=[[0, 0, 1, 1], [0, 1, 0, 1]])
        >>> mi
        MultiIndex([('a', 'bb'),
                    ('a', 'aa'),
                    ('b', 'bb'),
                    ('b', 'aa')],
                   )
        >>> mi.swaplevel(0, 1)
        MultiIndex([('bb', 'a'),
                    ('aa', 'a'),
                    ('bb', 'b'),
                    ('aa', 'b')],
                   )
        """
        new_levels = list(self.levels)
        new_codes = list(self.codes)
        new_names = list(self.names)

        i = self._get_level_number(i)
        j = self._get_level_number(j)

        new_levels[i], new_levels[j] = new_levels[j], new_levels[i]
        new_codes[i], new_codes[j] = new_codes[j], new_codes[i]
        new_names[i], new_names[j] = new_names[j], new_names[i]

        return MultiIndex(
            levels=new_levels, codes=new_codes, names=new_names, verify_integrity=False
        )

    def reorder_levels(self, order) -> MultiIndex:
        """
        Rearrange levels using input order. May not drop or duplicate levels.

        Parameters
        ----------
        order : list of int or list of str
            List representing new level order. Reference level by number
            (position) or by key (label).

        Returns
        -------
        MultiIndex

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([[1, 2], [3, 4]], names=['x', 'y'])
        >>> mi
        MultiIndex([(1, 3),
                    (2, 4)],
                   names=['x', 'y'])

        >>> mi.reorder_levels(order=[1, 0])
        MultiIndex([(3, 1),
                    (4, 2)],
                   names=['y', 'x'])

        >>> mi.reorder_levels(order=['y', 'x'])
        MultiIndex([(3, 1),
                    (4, 2)],
                   names=['y', 'x'])
        """
        order = [self._get_level_number(i) for i in order]
        result = self._reorder_ilevels(order)
        return result

    def _reorder_ilevels(self, order) -> MultiIndex:
        if len(order) != self.nlevels:
            raise AssertionError(
                f"Length of order must be same as number of levels ({self.nlevels}), "
                f"got {len(order)}"
            )
        new_levels = [self.levels[i] for i in order]
        new_codes = [self.codes[i] for i in order]
        new_names = [self.names[i] for i in order]

        return MultiIndex(
            levels=new_levels, codes=new_codes, names=new_names, verify_integrity=False
        )

    def _recode_for_new_levels(
        self, new_levels, copy: bool = True
    ) -> Generator[np.ndarray, None, None]:
        if len(new_levels) > self.nlevels:
            raise AssertionError(
                f"Length of new_levels ({len(new_levels)}) "
                f"must be <= self.nlevels ({self.nlevels})"
            )
        for i in range(len(new_levels)):
            yield recode_for_categories(
                self.codes[i], self.levels[i], new_levels[i], copy=copy
            )

    def _get_codes_for_sorting(self) -> list[Categorical]:
        """
        we are categorizing our codes by using the
        available categories (all, not just observed)
        excluding any missing ones (-1); this is in preparation
        for sorting, where we need to disambiguate that -1 is not
        a valid valid
        """

        def cats(level_codes):
            return np.arange(
                np.array(level_codes).max() + 1 if len(level_codes) else 0,
                dtype=level_codes.dtype,
            )

        return [
            Categorical.from_codes(level_codes, cats(level_codes), True, validate=False)
            for level_codes in self.codes
        ]

    def sortlevel(
        self,
        level: IndexLabel = 0,
        ascending: bool | list[bool] = True,
        sort_remaining: bool = True,
        na_position: str = "first",
    ) -> tuple[MultiIndex, npt.NDArray[np.intp]]:
        """
        Sort MultiIndex at the requested level.

        The result will respect the original ordering of the associated
        factor at that level.

        Parameters
        ----------
        level : list-like, int or str, default 0
            If a string is given, must be a name of the level.
            If list-like must be names or ints of levels.
        ascending : bool, default True
            False to sort in descending order.
            Can also be a list to specify a directed ordering.
        sort_remaining : sort by the remaining levels after level
        na_position : {'first' or 'last'}, default 'first'
            Argument 'first' puts NaNs at the beginning, 'last' puts NaNs at
            the end.

            .. versionadded:: 2.1.0

        Returns
        -------
        sorted_index : pd.MultiIndex
            Resulting index.
        indexer : np.ndarray[np.intp]
            Indices of output values in original index.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([[0, 0], [2, 1]])
        >>> mi
        MultiIndex([(0, 2),
                    (0, 1)],
                   )

        >>> mi.sortlevel()
        (MultiIndex([(0, 1),
                    (0, 2)],
                   ), array([1, 0]))

        >>> mi.sortlevel(sort_remaining=False)
        (MultiIndex([(0, 2),
                    (0, 1)],
                   ), array([0, 1]))

        >>> mi.sortlevel(1)
        (MultiIndex([(0, 1),
                    (0, 2)],
                   ), array([1, 0]))

        >>> mi.sortlevel(1, ascending=False)
        (MultiIndex([(0, 2),
                    (0, 1)],
                   ), array([0, 1]))
        """
        if not is_list_like(level):
            level = [level]
        # error: Item "Hashable" of "Union[Hashable, Sequence[Hashable]]" has
        # no attribute "__iter__" (not iterable)
        level = [
            self._get_level_number(lev) for lev in level  # type: ignore[union-attr]
        ]
        sortorder = None

        codes = [self.codes[lev] for lev in level]
        # we have a directed ordering via ascending
        if isinstance(ascending, list):
            if not len(level) == len(ascending):
                raise ValueError("level must have same length as ascending")
        elif sort_remaining:
            codes.extend(
                [self.codes[lev] for lev in range(len(self.levels)) if lev not in level]
            )
        else:
            sortorder = level[0]

        indexer = lexsort_indexer(
            codes, orders=ascending, na_position=na_position, codes_given=True
        )

        indexer = ensure_platform_int(indexer)
        new_codes = [level_codes.take(indexer) for level_codes in self.codes]

        new_index = MultiIndex(
            codes=new_codes,
            levels=self.levels,
            names=self.names,
            sortorder=sortorder,
            verify_integrity=False,
        )

        return new_index, indexer

    def _wrap_reindex_result(self, target, indexer, preserve_names: bool):
        if not isinstance(target, MultiIndex):
            if indexer is None:
                target = self
            elif (indexer >= 0).all():
                target = self.take(indexer)
            else:
                try:
                    target = MultiIndex.from_tuples(target)
                except TypeError:
                    # not all tuples, see test_constructor_dict_multiindex_reindex_flat
                    return target

        target = self._maybe_preserve_names(target, preserve_names)
        return target

    def _maybe_preserve_names(self, target: Index, preserve_names: bool) -> Index:
        if (
            preserve_names
            and target.nlevels == self.nlevels
            and target.names != self.names
        ):
            target = target.copy(deep=False)
            target.names = self.names
        return target

    # --------------------------------------------------------------------
    # Indexing Methods

    def _check_indexing_error(self, key) -> None:
        if not is_hashable(key) or is_iterator(key):
            # We allow tuples if they are hashable, whereas other Index
            #  subclasses require scalar.
            # We have to explicitly exclude generators, as these are hashable.
            raise InvalidIndexError(key)

    @cache_readonly
    def _should_fallback_to_positional(self) -> bool:
        """
        Should integer key(s) be treated as positional?
        """
        # GH#33355
        return self.levels[0]._should_fallback_to_positional

    def _get_indexer_strict(
        self, key, axis_name: str
    ) -> tuple[Index, npt.NDArray[np.intp]]:
        keyarr = key
        if not isinstance(keyarr, Index):
            keyarr = com.asarray_tuplesafe(keyarr)

        if len(keyarr) and not isinstance(keyarr[0], tuple):
            indexer = self._get_indexer_level_0(keyarr)

            self._raise_if_missing(key, indexer, axis_name)
            return self[indexer], indexer

        return super()._get_indexer_strict(key, axis_name)

    def _raise_if_missing(self, key, indexer, axis_name: str) -> None:
        keyarr = key
        if not isinstance(key, Index):
            keyarr = com.asarray_tuplesafe(key)

        if len(keyarr) and not isinstance(keyarr[0], tuple):
            # i.e. same condition for special case in MultiIndex._get_indexer_strict

            mask = indexer == -1
            if mask.any():
                check = self.levels[0].get_indexer(keyarr)
                cmask = check == -1
                if cmask.any():
                    raise KeyError(f"{keyarr[cmask]} not in index")
                # We get here when levels still contain values which are not
                # actually in Index anymore
                raise KeyError(f"{keyarr} not in index")
        else:
            return super()._raise_if_missing(key, indexer, axis_name)

    def _get_indexer_level_0(self, target) -> npt.NDArray[np.intp]:
        """
        Optimized equivalent to `self.get_level_values(0).get_indexer_for(target)`.
        """
        lev = self.levels[0]
        codes = self._codes[0]
        cat = Categorical.from_codes(codes=codes, categories=lev, validate=False)
        ci = Index(cat)
        return ci.get_indexer_for(target)

    def get_slice_bound(
        self,
        label: Hashable | Sequence[Hashable],
        side: Literal["left", "right"],
    ) -> int:
        """
        For an ordered MultiIndex, compute slice bound
        that corresponds to given label.

        Returns leftmost (one-past-the-rightmost if `side=='right') position
        of given label.

        Parameters
        ----------
        label : object or tuple of objects
        side : {'left', 'right'}

        Returns
        -------
        int
            Index of label.

        Notes
        -----
        This method only works if level 0 index of the MultiIndex is lexsorted.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([list('abbc'), list('gefd')])

        Get the locations from the leftmost 'b' in the first level
        until the end of the multiindex:

        >>> mi.get_slice_bound('b', side="left")
        1

        Like above, but if you get the locations from the rightmost
        'b' in the first level and 'f' in the second level:

        >>> mi.get_slice_bound(('b','f'), side="right")
        3

        See Also
        --------
        MultiIndex.get_loc : Get location for a label or a tuple of labels.
        MultiIndex.get_locs : Get location for a label/slice/list/mask or a
                              sequence of such.
        """
        if not isinstance(label, tuple):
            label = (label,)
        return self._partial_tup_index(label, side=side)

    # pylint: disable-next=useless-parent-delegation
    def slice_locs(self, start=None, end=None, step=None) -> tuple[int, int]:
        """
        For an ordered MultiIndex, compute the slice locations for input
        labels.

        The input labels can be tuples representing partial levels, e.g. for a
        MultiIndex with 3 levels, you can pass a single value (corresponding to
        the first level), or a 1-, 2-, or 3-tuple.

        Parameters
        ----------
        start : label or tuple, default None
            If None, defaults to the beginning
        end : label or tuple
            If None, defaults to the end
        step : int or None
            Slice step

        Returns
        -------
        (start, end) : (int, int)

        Notes
        -----
        This method only works if the MultiIndex is properly lexsorted. So,
        if only the first 2 levels of a 3-level MultiIndex are lexsorted,
        you can only pass two levels to ``.slice_locs``.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([list('abbd'), list('deff')],
        ...                                names=['A', 'B'])

        Get the slice locations from the beginning of 'b' in the first level
        until the end of the multiindex:

        >>> mi.slice_locs(start='b')
        (1, 4)

        Like above, but stop at the end of 'b' in the first level and 'f' in
        the second level:

        >>> mi.slice_locs(start='b', end=('b', 'f'))
        (1, 3)

        See Also
        --------
        MultiIndex.get_loc : Get location for a label or a tuple of labels.
        MultiIndex.get_locs : Get location for a label/slice/list/mask or a
                              sequence of such.
        """
        # This function adds nothing to its parent implementation (the magic
        # happens in get_slice_bound method), but it adds meaningful doc.
        return super().slice_locs(start, end, step)

    def _partial_tup_index(self, tup: tuple, side: Literal["left", "right"] = "left"):
        if len(tup) > self._lexsort_depth:
            raise UnsortedIndexError(
                f"Key length ({len(tup)}) was greater than MultiIndex lexsort depth "
                f"({self._lexsort_depth})"
            )

        n = len(tup)
        start, end = 0, len(self)
        zipped = zip(tup, self.levels, self.codes)
        for k, (lab, lev, level_codes) in enumerate(zipped):
            section = level_codes[start:end]

            loc: npt.NDArray[np.intp] | np.intp | int
            if lab not in lev and not isna(lab):
                # short circuit
                try:
                    loc = algos.searchsorted(lev, lab, side=side)
                except TypeError as err:
                    # non-comparable e.g. test_slice_locs_with_type_mismatch
                    raise TypeError(f"Level type mismatch: {lab}") from err
                if not is_integer(loc):
                    # non-comparable level, e.g. test_groupby_example
                    raise TypeError(f"Level type mismatch: {lab}")
                if side == "right" and loc >= 0:
                    loc -= 1
                return start + algos.searchsorted(section, loc, side=side)

            idx = self._get_loc_single_level_index(lev, lab)
            if isinstance(idx, slice) and k < n - 1:
                # Get start and end value from slice, necessary when a non-integer
                # interval is given as input GH#37707
                start = idx.start
                end = idx.stop
            elif k < n - 1:
                # error: Incompatible types in assignment (expression has type
                # "Union[ndarray[Any, dtype[signedinteger[Any]]]
                end = start + algos.searchsorted(  # type: ignore[assignment]
                    section, idx, side="right"
                )
                # error: Incompatible types in assignment (expression has type
                # "Union[ndarray[Any, dtype[signedinteger[Any]]]
                start = start + algos.searchsorted(  # type: ignore[assignment]
                    section, idx, side="left"
                )
            elif isinstance(idx, slice):
                idx = idx.start
                return start + algos.searchsorted(section, idx, side=side)
            else:
                return start + algos.searchsorted(section, idx, side=side)

    def _get_loc_single_level_index(self, level_index: Index, key: Hashable) -> int:
        """
        If key is NA value, location of index unify as -1.

        Parameters
        ----------
        level_index: Index
        key : label

        Returns
        -------
        loc : int
            If key is NA value, loc is -1
            Else, location of key in index.

        See Also
        --------
        Index.get_loc : The get_loc method for (single-level) index.
        """
        if is_scalar(key) and isna(key):
            # TODO: need is_valid_na_for_dtype(key, level_index.dtype)
            return -1
        else:
            return level_index.get_loc(key)

    def get_loc(self, key):
        """
        Get location for a label or a tuple of labels.

        The location is returned as an integer/slice or boolean
        mask.

        Parameters
        ----------
        key : label or tuple of labels (one for each level)

        Returns
        -------
        int, slice object or boolean mask
            If the key is past the lexsort depth, the return may be a
            boolean mask array, otherwise it is always a slice or int.

        See Also
        --------
        Index.get_loc : The get_loc method for (single-level) index.
        MultiIndex.slice_locs : Get slice location given start label(s) and
                                end label(s).
        MultiIndex.get_locs : Get location for a label/slice/list/mask or a
                              sequence of such.

        Notes
        -----
        The key cannot be a slice, list of same-level labels, a boolean mask,
        or a sequence of such. If you want to use those, use
        :meth:`MultiIndex.get_locs` instead.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([list('abb'), list('def')])

        >>> mi.get_loc('b')
        slice(1, 3, None)

        >>> mi.get_loc(('b', 'e'))
        1
        """
        self._check_indexing_error(key)

        def _maybe_to_slice(loc):
            """convert integer indexer to boolean mask or slice if possible"""
            if not isinstance(loc, np.ndarray) or loc.dtype != np.intp:
                return loc

            loc = lib.maybe_indices_to_slice(loc, len(self))
            if isinstance(loc, slice):
                return loc

            mask = np.empty(len(self), dtype="bool")
            mask.fill(False)
            mask[loc] = True
            return mask

        if not isinstance(key, tuple):
            loc = self._get_level_indexer(key, level=0)
            return _maybe_to_slice(loc)

        keylen = len(key)
        if self.nlevels < keylen:
            raise KeyError(
                f"Key length ({keylen}) exceeds index depth ({self.nlevels})"
            )

        if keylen == self.nlevels and self.is_unique:
            # TODO: what if we have an IntervalIndex level?
            #  i.e. do we need _index_as_unique on that level?
            try:
                return self._engine.get_loc(key)
            except KeyError as err:
                raise KeyError(key) from err
            except TypeError:
                # e.g. test_partial_slicing_with_multiindex partial string slicing
                loc, _ = self.get_loc_level(key, list(range(self.nlevels)))
                return loc

        # -- partial selection or non-unique index
        # break the key into 2 parts based on the lexsort_depth of the index;
        # the first part returns a continuous slice of the index; the 2nd part
        # needs linear search within the slice
        i = self._lexsort_depth
        lead_key, follow_key = key[:i], key[i:]

        if not lead_key:
            start = 0
            stop = len(self)
        else:
            try:
                start, stop = self.slice_locs(lead_key, lead_key)
            except TypeError as err:
                # e.g. test_groupby_example key = ((0, 0, 1, 2), "new_col")
                #  when self has 5 integer levels
                raise KeyError(key) from err

        if start == stop:
            raise KeyError(key)

        if not follow_key:
            return slice(start, stop)

        warnings.warn(
            "indexing past lexsort depth may impact performance.",
            PerformanceWarning,
            stacklevel=find_stack_level(),
        )

        loc = np.arange(start, stop, dtype=np.intp)

        for i, k in enumerate(follow_key, len(lead_key)):
            mask = self.codes[i][loc] == self._get_loc_single_level_index(
                self.levels[i], k
            )
            if not mask.all():
                loc = loc[mask]
            if not len(loc):
                raise KeyError(key)

        return _maybe_to_slice(loc) if len(loc) != stop - start else slice(start, stop)

    def get_loc_level(self, key, level: IndexLabel = 0, drop_level: bool = True):
        """
        Get location and sliced index for requested label(s)/level(s).

        Parameters
        ----------
        key : label or sequence of labels
        level : int/level name or list thereof, optional
        drop_level : bool, default True
            If ``False``, the resulting index will not drop any level.

        Returns
        -------
        tuple
            A 2-tuple where the elements :

            Element 0: int, slice object or boolean array.

            Element 1: The resulting sliced multiindex/index. If the key
            contains all levels, this will be ``None``.

        See Also
        --------
        MultiIndex.get_loc  : Get location for a label or a tuple of labels.
        MultiIndex.get_locs : Get location for a label/slice/list/mask or a
                              sequence of such.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([list('abb'), list('def')],
        ...                                names=['A', 'B'])

        >>> mi.get_loc_level('b')
        (slice(1, 3, None), Index(['e', 'f'], dtype='object', name='B'))

        >>> mi.get_loc_level('e', level='B')
        (array([False,  True, False]), Index(['b'], dtype='object', name='A'))

        >>> mi.get_loc_level(['b', 'e'])
        (1, None)
        """
        if not isinstance(level, (list, tuple)):
            level = self._get_level_number(level)
        else:
            level = [self._get_level_number(lev) for lev in level]

        loc, mi = self._get_loc_level(key, level=level)
        if not drop_level:
            if lib.is_integer(loc):
                # Slice index must be an integer or None
                mi = self[loc : loc + 1]
            else:
                mi = self[loc]
        return loc, mi

    def _get_loc_level(self, key, level: int | list[int] = 0):
        """
        get_loc_level but with `level` known to be positional, not name-based.
        """

        # different name to distinguish from maybe_droplevels
        def maybe_mi_droplevels(indexer, levels):
            """
            If level does not exist or all levels were dropped, the exception
            has to be handled outside.
            """
            new_index = self[indexer]

            for i in sorted(levels, reverse=True):
                new_index = new_index._drop_level_numbers([i])

            return new_index

        if isinstance(level, (tuple, list)):
            if len(key) != len(level):
                raise AssertionError(
                    "Key for location must have same length as number of levels"
                )
            result = None
            for lev, k in zip(level, key):
                loc, new_index = self._get_loc_level(k, level=lev)
                if isinstance(loc, slice):
                    mask = np.zeros(len(self), dtype=bool)
                    mask[loc] = True
                    loc = mask
                result = loc if result is None else result & loc

            try:
                # FIXME: we should be only dropping levels on which we are
                #  scalar-indexing
                mi = maybe_mi_droplevels(result, level)
            except ValueError:
                # droplevel failed because we tried to drop all levels,
                #  i.e. len(level) == self.nlevels
                mi = self[result]

            return result, mi

        # kludge for #1796
        if isinstance(key, list):
            key = tuple(key)

        if isinstance(key, tuple) and level == 0:
            try:
                # Check if this tuple is a single key in our first level
                if key in self.levels[0]:
                    indexer = self._get_level_indexer(key, level=level)
                    new_index = maybe_mi_droplevels(indexer, [0])
                    return indexer, new_index
            except (TypeError, InvalidIndexError):
                pass

            if not any(isinstance(k, slice) for k in key):
                if len(key) == self.nlevels and self.is_unique:
                    # Complete key in unique index -> standard get_loc
                    try:
                        return (self._engine.get_loc(key), None)
                    except KeyError as err:
                        raise KeyError(key) from err
                    except TypeError:
                        # e.g. partial string indexing
                        #  test_partial_string_timestamp_multiindex
                        pass

                # partial selection
                indexer = self.get_loc(key)
                ilevels = [i for i in range(len(key)) if key[i] != slice(None, None)]
                if len(ilevels) == self.nlevels:
                    if is_integer(indexer):
                        # we are dropping all levels
                        return indexer, None

                    # TODO: in some cases we still need to drop some levels,
                    #  e.g. test_multiindex_perf_warn
                    # test_partial_string_timestamp_multiindex
                    ilevels = [
                        i
                        for i in range(len(key))
                        if (
                            not isinstance(key[i], str)
                            or not self.levels[i]._supports_partial_string_indexing
                        )
                        and key[i] != slice(None, None)
                    ]
                    if len(ilevels) == self.nlevels:
                        # TODO: why?
                        ilevels = []
                return indexer, maybe_mi_droplevels(indexer, ilevels)

            else:
                indexer = None
                for i, k in enumerate(key):
                    if not isinstance(k, slice):
                        loc_level = self._get_level_indexer(k, level=i)
                        if isinstance(loc_level, slice):
                            if com.is_null_slice(loc_level) or com.is_full_slice(
                                loc_level, len(self)
                            ):
                                # everything
                                continue

                            # e.g. test_xs_IndexSlice_argument_not_implemented
                            k_index = np.zeros(len(self), dtype=bool)
                            k_index[loc_level] = True

                        else:
                            k_index = loc_level

                    elif com.is_null_slice(k):
                        # taking everything, does not affect `indexer` below
                        continue

                    else:
                        # FIXME: this message can be inaccurate, e.g.
                        #  test_series_varied_multiindex_alignment
                        raise TypeError(f"Expected label or tuple of labels, got {key}")

                    if indexer is None:
                        indexer = k_index
                    else:
                        indexer &= k_index
                if indexer is None:
                    indexer = slice(None, None)
                ilevels = [i for i in range(len(key)) if key[i] != slice(None, None)]
                return indexer, maybe_mi_droplevels(indexer, ilevels)
        else:
            indexer = self._get_level_indexer(key, level=level)
            if (
                isinstance(key, str)
                and self.levels[level]._supports_partial_string_indexing
            ):
                # check to see if we did an exact lookup vs sliced
                check = self.levels[level].get_loc(key)
                if not is_integer(check):
                    # e.g. test_partial_string_timestamp_multiindex
                    return indexer, self[indexer]

            try:
                result_index = maybe_mi_droplevels(indexer, [level])
            except ValueError:
                result_index = self[indexer]

            return indexer, result_index

    def _get_level_indexer(
        self, key, level: int = 0, indexer: npt.NDArray[np.bool_] | None = None
    ):
        # `level` kwarg is _always_ positional, never name
        # return a boolean array or slice showing where the key is
        # in the totality of values
        # if the indexer is provided, then use this

        level_index = self.levels[level]
        level_codes = self.codes[level]

        def convert_indexer(start, stop, step, indexer=indexer, codes=level_codes):
            # Compute a bool indexer to identify the positions to take.
            # If we have an existing indexer, we only need to examine the
            # subset of positions where the existing indexer is True.
            if indexer is not None:
                # we only need to look at the subset of codes where the
                # existing indexer equals True
                codes = codes[indexer]

            if step is None or step == 1:
                new_indexer = (codes >= start) & (codes < stop)
            else:
                r = np.arange(start, stop, step, dtype=codes.dtype)
                new_indexer = algos.isin(codes, r)

            if indexer is None:
                return new_indexer

            indexer = indexer.copy()
            indexer[indexer] = new_indexer
            return indexer

        if isinstance(key, slice):
            # handle a slice, returning a slice if we can
            # otherwise a boolean indexer
            step = key.step
            is_negative_step = step is not None and step < 0

            try:
                if key.start is not None:
                    start = level_index.get_loc(key.start)
                elif is_negative_step:
                    start = len(level_index) - 1
                else:
                    start = 0

                if key.stop is not None:
                    stop = level_index.get_loc(key.stop)
                elif is_negative_step:
                    stop = 0
                elif isinstance(start, slice):
                    stop = len(level_index)
                else:
                    stop = len(level_index) - 1
            except KeyError:
                # we have a partial slice (like looking up a partial date
                # string)
                start = stop = level_index.slice_indexer(key.start, key.stop, key.step)
                step = start.step

            if isinstance(start, slice) or isinstance(stop, slice):
                # we have a slice for start and/or stop
                # a partial date slicer on a DatetimeIndex generates a slice
                # note that the stop ALREADY includes the stopped point (if
                # it was a string sliced)
                start = getattr(start, "start", start)
                stop = getattr(stop, "stop", stop)
                return convert_indexer(start, stop, step)

            elif level > 0 or self._lexsort_depth == 0 or step is not None:
                # need to have like semantics here to right
                # searching as when we are using a slice
                # so adjust the stop by 1 (so we include stop)
                stop = (stop - 1) if is_negative_step else (stop + 1)
                return convert_indexer(start, stop, step)
            else:
                # sorted, so can return slice object -> view
                i = algos.searchsorted(level_codes, start, side="left")
                j = algos.searchsorted(level_codes, stop, side="right")
                return slice(i, j, step)

        else:
            idx = self._get_loc_single_level_index(level_index, key)

            if level > 0 or self._lexsort_depth == 0:
                # Desired level is not sorted
                if isinstance(idx, slice):
                    # test_get_loc_partial_timestamp_multiindex
                    locs = (level_codes >= idx.start) & (level_codes < idx.stop)
                    return locs

                locs = np.asarray(level_codes == idx, dtype=bool)

                if not locs.any():
                    # The label is present in self.levels[level] but unused:
                    raise KeyError(key)
                return locs

            if isinstance(idx, slice):
                # e.g. test_partial_string_timestamp_multiindex
                start = algos.searchsorted(level_codes, idx.start, side="left")
                # NB: "left" here bc of slice semantics
                end = algos.searchsorted(level_codes, idx.stop, side="left")
            else:
                start = algos.searchsorted(level_codes, idx, side="left")
                end = algos.searchsorted(level_codes, idx, side="right")

            if start == end:
                # The label is present in self.levels[level] but unused:
                raise KeyError(key)
            return slice(start, end)

    def get_locs(self, seq) -> npt.NDArray[np.intp]:
        """
        Get location for a sequence of labels.

        Parameters
        ----------
        seq : label, slice, list, mask or a sequence of such
           You should use one of the above for each level.
           If a level should not be used, set it to ``slice(None)``.

        Returns
        -------
        numpy.ndarray
            NumPy array of integers suitable for passing to iloc.

        See Also
        --------
        MultiIndex.get_loc : Get location for a label or a tuple of labels.
        MultiIndex.slice_locs : Get slice location given start label(s) and
                                end label(s).

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([list('abb'), list('def')])

        >>> mi.get_locs('b')  # doctest: +SKIP
        array([1, 2], dtype=int64)

        >>> mi.get_locs([slice(None), ['e', 'f']])  # doctest: +SKIP
        array([1, 2], dtype=int64)

        >>> mi.get_locs([[True, False, True], slice('e', 'f')])  # doctest: +SKIP
        array([2], dtype=int64)
        """

        # must be lexsorted to at least as many levels
        true_slices = [i for (i, s) in enumerate(com.is_true_slices(seq)) if s]
        if true_slices and true_slices[-1] >= self._lexsort_depth:
            raise UnsortedIndexError(
                "MultiIndex slicing requires the index to be lexsorted: slicing "
                f"on levels {true_slices}, lexsort depth {self._lexsort_depth}"
            )

        if any(x is Ellipsis for x in seq):
            raise NotImplementedError(
                "MultiIndex does not support indexing with Ellipsis"
            )

        n = len(self)

        def _to_bool_indexer(indexer) -> npt.NDArray[np.bool_]:
            if isinstance(indexer, slice):
                new_indexer = np.zeros(n, dtype=np.bool_)
                new_indexer[indexer] = True
                return new_indexer
            return indexer

        # a bool indexer for the positions we want to take
        indexer: npt.NDArray[np.bool_] | None = None

        for i, k in enumerate(seq):
            lvl_indexer: npt.NDArray[np.bool_] | slice | None = None

            if com.is_bool_indexer(k):
                if len(k) != n:
                    raise ValueError(
                        "cannot index with a boolean indexer that "
                        "is not the same length as the index"
                    )
                lvl_indexer = np.asarray(k)
                if indexer is None:
                    lvl_indexer = lvl_indexer.copy()

            elif is_list_like(k):
                # a collection of labels to include from this level (these are or'd)

                # GH#27591 check if this is a single tuple key in the level
                try:
                    lvl_indexer = self._get_level_indexer(k, level=i, indexer=indexer)
                except (InvalidIndexError, TypeError, KeyError) as err:
                    # InvalidIndexError e.g. non-hashable, fall back to treating
                    #  this as a sequence of labels
                    # KeyError it can be ambiguous if this is a label or sequence
                    #  of labels
                    #  github.com/pandas-dev/pandas/issues/39424#issuecomment-871626708
                    for x in k:
                        if not is_hashable(x):
                            # e.g. slice
                            raise err
                        # GH 39424: Ignore not founds
                        # GH 42351: No longer ignore not founds & enforced in 2.0
                        # TODO: how to handle IntervalIndex level? (no test cases)
                        item_indexer = self._get_level_indexer(
                            x, level=i, indexer=indexer
                        )
                        if lvl_indexer is None:
                            lvl_indexer = _to_bool_indexer(item_indexer)
                        elif isinstance(item_indexer, slice):
                            lvl_indexer[item_indexer] = True  # type: ignore[index]
                        else:
                            lvl_indexer |= item_indexer

                if lvl_indexer is None:
                    # no matches we are done
                    # test_loc_getitem_duplicates_multiindex_empty_indexer
                    return np.array([], dtype=np.intp)

            elif com.is_null_slice(k):
                # empty slice
                if indexer is None and i == len(seq) - 1:
                    return np.arange(n, dtype=np.intp)
                continue

            else:
                # a slice or a single label
                lvl_indexer = self._get_level_indexer(k, level=i, indexer=indexer)

            # update indexer
            lvl_indexer = _to_bool_indexer(lvl_indexer)
            if indexer is None:
                indexer = lvl_indexer
            else:
                indexer &= lvl_indexer
                if not np.any(indexer) and np.any(lvl_indexer):
                    raise KeyError(seq)

        # empty indexer
        if indexer is None:
            return np.array([], dtype=np.intp)

        pos_indexer = indexer.nonzero()[0]
        return self._reorder_indexer(seq, pos_indexer)

    # --------------------------------------------------------------------

    def _reorder_indexer(
        self,
        seq: tuple[Scalar | Iterable | AnyArrayLike, ...],
        indexer: npt.NDArray[np.intp],
    ) -> npt.NDArray[np.intp]:
        """
        Reorder an indexer of a MultiIndex (self) so that the labels are in the
        same order as given in seq

        Parameters
        ----------
        seq : label/slice/list/mask or a sequence of such
        indexer: a position indexer of self

        Returns
        -------
        indexer : a sorted position indexer of self ordered as seq
        """

        # check if sorting is necessary
        need_sort = False
        for i, k in enumerate(seq):
            if com.is_null_slice(k) or com.is_bool_indexer(k) or is_scalar(k):
                pass
            elif is_list_like(k):
                if len(k) <= 1:  # type: ignore[arg-type]
                    pass
                elif self._is_lexsorted():
                    # If the index is lexsorted and the list_like label
                    # in seq are sorted then we do not need to sort
                    k_codes = self.levels[i].get_indexer(k)
                    k_codes = k_codes[k_codes >= 0]  # Filter absent keys
                    # True if the given codes are not ordered
                    need_sort = (k_codes[:-1] > k_codes[1:]).any()
                else:
                    need_sort = True
            elif isinstance(k, slice):
                if self._is_lexsorted():
                    need_sort = k.step is not None and k.step < 0
                else:
                    need_sort = True
            else:
                need_sort = True
            if need_sort:
                break
        if not need_sort:
            return indexer

        n = len(self)
        keys: tuple[np.ndarray, ...] = ()
        # For each level of the sequence in seq, map the level codes with the
        # order they appears in a list-like sequence
        # This mapping is then use to reorder the indexer
        for i, k in enumerate(seq):
            if is_scalar(k):
                # GH#34603 we want to treat a scalar the same as an all equal list
                k = [k]
            if com.is_bool_indexer(k):
                new_order = np.arange(n)[indexer]
            elif is_list_like(k):
                # Generate a map with all level codes as sorted initially
                if not isinstance(k, (np.ndarray, ExtensionArray, Index, ABCSeries)):
                    k = sanitize_array(k, None)
                k = algos.unique(k)
                key_order_map = np.ones(len(self.levels[i]), dtype=np.uint64) * len(
                    self.levels[i]
                )
                # Set order as given in the indexer list
                level_indexer = self.levels[i].get_indexer(k)
                level_indexer = level_indexer[level_indexer >= 0]  # Filter absent keys
                key_order_map[level_indexer] = np.arange(len(level_indexer))

                new_order = key_order_map[self.codes[i][indexer]]
            elif isinstance(k, slice) and k.step is not None and k.step < 0:
                # flip order for negative step
                new_order = np.arange(n)[::-1][indexer]
            elif isinstance(k, slice) and k.start is None and k.stop is None:
                # slice(None) should not determine order GH#31330
                new_order = np.ones((n,), dtype=np.intp)[indexer]
            else:
                # For all other case, use the same order as the level
                new_order = np.arange(n)[indexer]
            keys = (new_order,) + keys

        # Find the reordering using lexsort on the keys mapping
        ind = np.lexsort(keys)
        return indexer[ind]

    def truncate(self, before=None, after=None) -> MultiIndex:
        """
        Slice index between two labels / tuples, return new MultiIndex.

        Parameters
        ----------
        before : label or tuple, can be partial. Default None
            None defaults to start.
        after : label or tuple, can be partial. Default None
            None defaults to end.

        Returns
        -------
        MultiIndex
            The truncated MultiIndex.

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays([['a', 'b', 'c'], ['x', 'y', 'z']])
        >>> mi
        MultiIndex([('a', 'x'), ('b', 'y'), ('c', 'z')],
                   )
        >>> mi.truncate(before='a', after='b')
        MultiIndex([('a', 'x'), ('b', 'y')],
                   )
        """
        if after and before and after < before:
            raise ValueError("after < before")

        i, j = self.levels[0].slice_locs(before, after)
        left, right = self.slice_locs(before, after)

        new_levels = list(self.levels)
        new_levels[0] = new_levels[0][i:j]

        new_codes = [level_codes[left:right] for level_codes in self.codes]
        new_codes[0] = new_codes[0] - i

        return MultiIndex(
            levels=new_levels,
            codes=new_codes,
            names=self._names,
            verify_integrity=False,
        )

    def equals(self, other: object) -> bool:
        """
        Determines if two MultiIndex objects have the same labeling information
        (the levels themselves do not necessarily have to be the same)

        See Also
        --------
        equal_levels
        """
        if self.is_(other):
            return True

        if not isinstance(other, Index):
            return False

        if len(self) != len(other):
            return False

        if not isinstance(other, MultiIndex):
            # d-level MultiIndex can equal d-tuple Index
            if not self._should_compare(other):
                # object Index or Categorical[object] may contain tuples
                return False
            return array_equivalent(self._values, other._values)

        if self.nlevels != other.nlevels:
            return False

        for i in range(self.nlevels):
            self_codes = self.codes[i]
            other_codes = other.codes[i]
            self_mask = self_codes == -1
            other_mask = other_codes == -1
            if not np.array_equal(self_mask, other_mask):
                return False
            self_codes = self_codes[~self_mask]
            self_values = self.levels[i]._values.take(self_codes)

            other_codes = other_codes[~other_mask]
            other_values = other.levels[i]._values.take(other_codes)

            # since we use NaT both datetime64 and timedelta64 we can have a
            # situation where a level is typed say timedelta64 in self (IOW it
            # has other values than NaT) but types datetime64 in other (where
            # its all NaT) but these are equivalent
            if len(self_values) == 0 and len(other_values) == 0:
                continue

            if not isinstance(self_values, np.ndarray):
                # i.e. ExtensionArray
                if not self_values.equals(other_values):
                    return False
            elif not isinstance(other_values, np.ndarray):
                # i.e. other is ExtensionArray
                if not other_values.equals(self_values):
                    return False
            else:
                if not array_equivalent(self_values, other_values):
                    return False

        return True

    def equal_levels(self, other: MultiIndex) -> bool:
        """
        Return True if the levels of both MultiIndex objects are the same

        """
        if self.nlevels != other.nlevels:
            return False

        for i in range(self.nlevels):
            if not self.levels[i].equals(other.levels[i]):
                return False
        return True

    # --------------------------------------------------------------------
    # Set Methods

    def _union(self, other, sort) -> MultiIndex:
        other, result_names = self._convert_can_do_setop(other)
        if other.has_duplicates:
            # This is only necessary if other has dupes,
            # otherwise difference is faster
            result = super()._union(other, sort)

            if isinstance(result, MultiIndex):
                return result
            return MultiIndex.from_arrays(
                zip(*result), sortorder=None, names=result_names
            )

        else:
            right_missing = other.difference(self, sort=False)
            if len(right_missing):
                result = self.append(right_missing)
            else:
                result = self._get_reconciled_name_object(other)

            if sort is not False:
                try:
                    result = result.sort_values()
                except TypeError:
                    if sort is True:
                        raise
                    warnings.warn(
                        "The values in the array are unorderable. "
                        "Pass `sort=False` to suppress this warning.",
                        RuntimeWarning,
                        stacklevel=find_stack_level(),
                    )
            return result

    def _is_comparable_dtype(self, dtype: DtypeObj) -> bool:
        return is_object_dtype(dtype)

    def _get_reconciled_name_object(self, other) -> MultiIndex:
        """
        If the result of a set operation will be self,
        return self, unless the names change, in which
        case make a shallow copy of self.
        """
        names = self._maybe_match_names(other)
        if self.names != names:
            # error: Cannot determine type of "rename"
            return self.rename(names)  # type: ignore[has-type]
        return self

    def _maybe_match_names(self, other):
        """
        Try to find common names to attach to the result of an operation between
        a and b. Return a consensus list of names if they match at least partly
        or list of None if they have completely different names.
        """
        if len(self.names) != len(other.names):
            return [None] * len(self.names)
        names = []
        for a_name, b_name in zip(self.names, other.names):
            if a_name == b_name:
                names.append(a_name)
            else:
                # TODO: what if they both have np.nan for their names?
                names.append(None)
        return names

    def _wrap_intersection_result(self, other, result) -> MultiIndex:
        _, result_names = self._convert_can_do_setop(other)
        return result.set_names(result_names)

    def _wrap_difference_result(self, other, result: MultiIndex) -> MultiIndex:
        _, result_names = self._convert_can_do_setop(other)

        if len(result) == 0:
            return result.remove_unused_levels().set_names(result_names)
        else:
            return result.set_names(result_names)

    def _convert_can_do_setop(self, other):
        result_names = self.names

        if not isinstance(other, Index):
            if len(other) == 0:
                return self[:0], self.names
            else:
                msg = "other must be a MultiIndex or a list of tuples"
                try:
                    other = MultiIndex.from_tuples(other, names=self.names)
                except (ValueError, TypeError) as err:
                    # ValueError raised by tuples_to_object_array if we
                    #  have non-object dtype
                    raise TypeError(msg) from err
        else:
            result_names = get_unanimous_names(self, other)

        return other, result_names

    # --------------------------------------------------------------------

    @doc(Index.astype)
    def astype(self, dtype, copy: bool = True):
        dtype = pandas_dtype(dtype)
        if isinstance(dtype, CategoricalDtype):
            msg = "> 1 ndim Categorical are not supported at this time"
            raise NotImplementedError(msg)
        if not is_object_dtype(dtype):
            raise TypeError(
                "Setting a MultiIndex dtype to anything other than object "
                "is not supported"
            )
        if copy is True:
            return self._view()
        return self

    def _validate_fill_value(self, item):
        if isinstance(item, MultiIndex):
            # GH#43212
            if item.nlevels != self.nlevels:
                raise ValueError("Item must have length equal to number of levels.")
            return item._values
        elif not isinstance(item, tuple):
            # Pad the key with empty strings if lower levels of the key
            # aren't specified:
            item = (item,) + ("",) * (self.nlevels - 1)
        elif len(item) != self.nlevels:
            raise ValueError("Item must have length equal to number of levels.")
        return item

    def putmask(self, mask, value: MultiIndex) -> MultiIndex:
        """
        Return a new MultiIndex of the values set with the mask.

        Parameters
        ----------
        mask : array like
        value : MultiIndex
            Must either be the same length as self or length one

        Returns
        -------
        MultiIndex
        """
        mask, noop = validate_putmask(self, mask)
        if noop:
            return self.copy()

        if len(mask) == len(value):
            subset = value[mask].remove_unused_levels()
        else:
            subset = value.remove_unused_levels()

        new_levels = []
        new_codes = []

        for i, (value_level, level, level_codes) in enumerate(
            zip(subset.levels, self.levels, self.codes)
        ):
            new_level = level.union(value_level, sort=False)
            value_codes = new_level.get_indexer_for(subset.get_level_values(i))
            new_code = ensure_int64(level_codes)
            new_code[mask] = value_codes
            new_levels.append(new_level)
            new_codes.append(new_code)

        return MultiIndex(
            levels=new_levels, codes=new_codes, names=self.names, verify_integrity=False
        )

    def insert(self, loc: int, item) -> MultiIndex:
        """
        Make new MultiIndex inserting new item at location

        Parameters
        ----------
        loc : int
        item : tuple
            Must be same length as number of levels in the MultiIndex

        Returns
        -------
        new_index : Index
        """
        item = self._validate_fill_value(item)

        new_levels = []
        new_codes = []
        for k, level, level_codes in zip(item, self.levels, self.codes):
            if k not in level:
                # have to insert into level
                # must insert at end otherwise you have to recompute all the
                # other codes
                lev_loc = len(level)
                level = level.insert(lev_loc, k)
            else:
                lev_loc = level.get_loc(k)

            new_levels.append(level)
            new_codes.append(np.insert(ensure_int64(level_codes), loc, lev_loc))

        return MultiIndex(
            levels=new_levels, codes=new_codes, names=self.names, verify_integrity=False
        )

    def delete(self, loc) -> MultiIndex:
        """
        Make new index with passed location deleted

        Returns
        -------
        new_index : MultiIndex
        """
        new_codes = [np.delete(level_codes, loc) for level_codes in self.codes]
        return MultiIndex(
            levels=self.levels,
            codes=new_codes,
            names=self.names,
            verify_integrity=False,
        )

    @doc(Index.isin)
    def isin(self, values, level=None) -> npt.NDArray[np.bool_]:
        if isinstance(values, Generator):
            values = list(values)

        if level is None:
            if len(values) == 0:
                return np.zeros((len(self),), dtype=np.bool_)
            if not isinstance(values, MultiIndex):
                values = MultiIndex.from_tuples(values)
            return values.unique().get_indexer_for(self) != -1
        else:
            num = self._get_level_number(level)
            levs = self.get_level_values(num)

            if levs.size == 0:
                return np.zeros(len(levs), dtype=np.bool_)
            return levs.isin(values)

    # error: Incompatible types in assignment (expression has type overloaded function,
    # base class "Index" defined the type as "Callable[[Index, Any, bool], Any]")
    rename = Index.set_names  # type: ignore[assignment]

    # ---------------------------------------------------------------
    # Arithmetic/Numeric Methods - Disabled

    __add__ = make_invalid_op("__add__")
    __radd__ = make_invalid_op("__radd__")
    __iadd__ = make_invalid_op("__iadd__")
    __sub__ = make_invalid_op("__sub__")
    __rsub__ = make_invalid_op("__rsub__")
    __isub__ = make_invalid_op("__isub__")
    __pow__ = make_invalid_op("__pow__")
    __rpow__ = make_invalid_op("__rpow__")
    __mul__ = make_invalid_op("__mul__")
    __rmul__ = make_invalid_op("__rmul__")
    __floordiv__ = make_invalid_op("__floordiv__")
    __rfloordiv__ = make_invalid_op("__rfloordiv__")
    __truediv__ = make_invalid_op("__truediv__")
    __rtruediv__ = make_invalid_op("__rtruediv__")
    __mod__ = make_invalid_op("__mod__")
    __rmod__ = make_invalid_op("__rmod__")
    __divmod__ = make_invalid_op("__divmod__")
    __rdivmod__ = make_invalid_op("__rdivmod__")
    # Unary methods disabled
    __neg__ = make_invalid_op("__neg__")
    __pos__ = make_invalid_op("__pos__")
    __abs__ = make_invalid_op("__abs__")
    __invert__ = make_invalid_op("__invert__")


def _lexsort_depth(codes: list[np.ndarray], nlevels: int) -> int:
    """Count depth (up to a maximum of `nlevels`) with which codes are lexsorted."""
    int64_codes = [ensure_int64(level_codes) for level_codes in codes]
    for k in range(nlevels, 0, -1):
        if libalgos.is_lexsorted(int64_codes[:k]):
            return k
    return 0


def sparsify_labels(label_list, start: int = 0, sentinel: object = ""):
    pivoted = list(zip(*label_list))
    k = len(label_list)

    result = pivoted[: start + 1]
    prev = pivoted[start]

    for cur in pivoted[start + 1 :]:
        sparse_cur = []

        for i, (p, t) in enumerate(zip(prev, cur)):
            if i == k - 1:
                sparse_cur.append(t)
                # error: Argument 1 to "append" of "list" has incompatible
                # type "list[Any]"; expected "tuple[Any, ...]"
                result.append(sparse_cur)  # type: ignore[arg-type]
                break

            if p == t:
                sparse_cur.append(sentinel)
            else:
                sparse_cur.extend(cur[i:])
                # error: Argument 1 to "append" of "list" has incompatible
                # type "list[Any]"; expected "tuple[Any, ...]"
                result.append(sparse_cur)  # type: ignore[arg-type]
                break

        prev = cur

    return list(zip(*result))


def _get_na_rep(dtype: DtypeObj) -> str:
    if isinstance(dtype, ExtensionDtype):
        return f"{dtype.na_value}"
    else:
        dtype_type = dtype.type

    return {np.datetime64: "NaT", np.timedelta64: "NaT"}.get(dtype_type, "NaN")


def maybe_droplevels(index: Index, key) -> Index:
    """
    Attempt to drop level or levels from the given index.

    Parameters
    ----------
    index: Index
    key : scalar or tuple

    Returns
    -------
    Index
    """
    # drop levels
    original_index = index
    if isinstance(key, tuple):
        # Caller is responsible for ensuring the key is not an entry in the first
        #  level of the MultiIndex.
        for _ in key:
            try:
                index = index._drop_level_numbers([0])
            except ValueError:
                # we have dropped too much, so back out
                return original_index
    else:
        try:
            index = index._drop_level_numbers([0])
        except ValueError:
            pass

    return index


def _coerce_indexer_frozen(array_like, categories, copy: bool = False) -> np.ndarray:
    """
    Coerce the array-like indexer to the smallest integer dtype that can encode all
    of the given categories.

    Parameters
    ----------
    array_like : array-like
    categories : array-like
    copy : bool

    Returns
    -------
    np.ndarray
        Non-writeable.
    """
    array_like = coerce_indexer_dtype(array_like, categories)
    if copy:
        array_like = array_like.copy()
    array_like.flags.writeable = False
    return array_like


def _require_listlike(level, arr, arrname: str):
    """
    Ensure that level is either None or listlike, and arr is list-of-listlike.
    """
    if level is not None and not is_list_like(level):
        if not is_list_like(arr):
            raise TypeError(f"{arrname} must be list-like")
        if len(arr) > 0 and is_list_like(arr[0]):
            raise TypeError(f"{arrname} must be list-like")
        level = [level]
        arr = [arr]
    elif level is None or is_list_like(level):
        if not is_list_like(arr) or not is_list_like(arr[0]):
            raise TypeError(f"{arrname} must be list of lists-like")
    return level, arr
<|endoftext|>"""
datetimelike delegation
"""
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    cast,
)
import warnings

import numpy as np

from pandas._libs import lib
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import (
    is_integer_dtype,
    is_list_like,
)
from pandas.core.dtypes.dtypes import (
    ArrowDtype,
    CategoricalDtype,
    DatetimeTZDtype,
    PeriodDtype,
)
from pandas.core.dtypes.generic import ABCSeries

from pandas.core.accessor import (
    PandasDelegate,
    delegate_names,
)
from pandas.core.arrays import (
    DatetimeArray,
    PeriodArray,
    TimedeltaArray,
)
from pandas.core.arrays.arrow.array import ArrowExtensionArray
from pandas.core.base import (
    NoNewAttributesMixin,
    PandasObject,
)
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex

if TYPE_CHECKING:
    from pandas import (
        DataFrame,
        Series,
    )


class Properties(PandasDelegate, PandasObject, NoNewAttributesMixin):
    _hidden_attrs = PandasObject._hidden_attrs | {
        "orig",
        "name",
    }

    def __init__(self, data: Series, orig) -> None:
        if not isinstance(data, ABCSeries):
            raise TypeError(
                f"cannot convert an object of type {type(data)} to a datetimelike index"
            )

        self._parent = data
        self.orig = orig
        self.name = getattr(data, "name", None)
        self._freeze()

    def _get_values(self):
        data = self._parent
        if lib.is_np_dtype(data.dtype, "M"):
            return DatetimeIndex(data, copy=False, name=self.name)

        elif isinstance(data.dtype, DatetimeTZDtype):
            return DatetimeIndex(data, copy=False, name=self.name)

        elif lib.is_np_dtype(data.dtype, "m"):
            return TimedeltaIndex(data, copy=False, name=self.name)

        elif isinstance(data.dtype, PeriodDtype):
            return PeriodArray(data, copy=False)

        raise TypeError(
            f"cannot convert an object of type {type(data)} to a datetimelike index"
        )

    def _delegate_property_get(self, name: str):
        from pandas import Series

        values = self._get_values()

        result = getattr(values, name)

        # maybe need to upcast (ints)
        if isinstance(result, np.ndarray):
            if is_integer_dtype(result):
                result = result.astype("int64")
        elif not is_list_like(result):
            return result

        result = np.asarray(result)

        if self.orig is not None:
            index = self.orig.index
        else:
            index = self._parent.index
        # return the result as a Series
        result = Series(result, index=index, name=self.name).__finalize__(self._parent)

        # setting this object will show a SettingWithCopyWarning/Error
        result._is_copy = (
            "modifications to a property of a datetimelike "
            "object are not supported and are discarded. "
            "Change values on the original."
        )

        return result

    def _delegate_property_set(self, name: str, value, *args, **kwargs):
        raise ValueError(
            "modifications to a property of a datetimelike object are not supported. "
            "Change values on the original."
        )

    def _delegate_method(self, name: str, *args, **kwargs):
        from pandas import Series

        values = self._get_values()

        method = getattr(values, name)
        result = method(*args, **kwargs)

        if not is_list_like(result):
            return result

        result = Series(result, index=self._parent.index, name=self.name).__finalize__(
            self._parent
        )

        # setting this object will show a SettingWithCopyWarning/Error
        result._is_copy = (
            "modifications to a method of a datetimelike "
            "object are not supported and are discarded. "
            "Change values on the original."
        )

        return result


@delegate_names(
    delegate=ArrowExtensionArray,
    accessors=TimedeltaArray._datetimelike_ops,
    typ="property",
    accessor_mapping=lambda x: f"_dt_{x}",
    raise_on_missing=False,
)
@delegate_names(
    delegate=ArrowExtensionArray,
    accessors=TimedeltaArray._datetimelike_methods,
    typ="method",
    accessor_mapping=lambda x: f"_dt_{x}",
    raise_on_missing=False,
)
@delegate_names(
    delegate=ArrowExtensionArray,
    accessors=DatetimeArray._datetimelike_ops,
    typ="property",
    accessor_mapping=lambda x: f"_dt_{x}",
    raise_on_missing=False,
)
@delegate_names(
    delegate=ArrowExtensionArray,
    accessors=DatetimeArray._datetimelike_methods,
    typ="method",
    accessor_mapping=lambda x: f"_dt_{x}",
    raise_on_missing=False,
)
class ArrowTemporalProperties(PandasDelegate, PandasObject, NoNewAttributesMixin):
    def __init__(self, data: Series, orig) -> None:
        if not isinstance(data, ABCSeries):
            raise TypeError(
                f"cannot convert an object of type {type(data)} to a datetimelike index"
            )

        self._parent = data
        self._orig = orig
        self._freeze()

    def _delegate_property_get(self, name: str):
        if not hasattr(self._parent.array, f"_dt_{name}"):
            raise NotImplementedError(
                f"dt.{name} is not supported for {self._parent.dtype}"
            )
        result = getattr(self._parent.array, f"_dt_{name}")

        if not is_list_like(result):
            return result

        if self._orig is not None:
            index = self._orig.index
        else:
            index = self._parent.index
        # return the result as a Series, which is by definition a copy
        result = type(self._parent)(
            result, index=index, name=self._parent.name
        ).__finalize__(self._parent)

        return result

    def _delegate_method(self, name: str, *args, **kwargs):
        if not hasattr(self._parent.array, f"_dt_{name}"):
            raise NotImplementedError(
                f"dt.{name} is not supported for {self._parent.dtype}"
            )

        result = getattr(self._parent.array, f"_dt_{name}")(*args, **kwargs)

        if self._orig is not None:
            index = self._orig.index
        else:
            index = self._parent.index
        # return the result as a Series, which is by definition a copy
        result = type(self._parent)(
            result, index=index, name=self._parent.name
        ).__finalize__(self._parent)

        return result

    def to_pytimedelta(self):
        return cast(ArrowExtensionArray, self._parent.array)._dt_to_pytimedelta()

    def to_pydatetime(self):
        # GH#20306
        warnings.warn(
            f"The behavior of {type(self).__name__}.to_pydatetime is deprecated, "
            "in a future version this will return a Series containing python "
            "datetime objects instead of an ndarray. To retain the old behavior, "
            "call `np.array` on the result",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
        return cast(ArrowExtensionArray, self._parent.array)._dt_to_pydatetime()

    def isocalendar(self) -> DataFrame:
        from pandas import DataFrame

        result = (
            cast(ArrowExtensionArray, self._parent.array)
            ._dt_isocalendar()
            ._pa_array.combine_chunks()
        )
        iso_calendar_df = DataFrame(
            {
                col: type(self._parent.array)(result.field(i))  # type: ignore[call-arg]
                for i, col in enumerate(["year", "week", "day"])
            }
        )
        return iso_calendar_df

    @property
    def components(self) -> DataFrame:
        from pandas import DataFrame

        components_df = DataFrame(
            {
                col: getattr(self._parent.array, f"_dt_{col}")
                for col in [
                    "days",
                    "hours",
                    "minutes",
                    "seconds",
                    "milliseconds",
                    "microseconds",
                    "nanoseconds",
                ]
            }
        )
        return components_df


@delegate_names(
    delegate=DatetimeArray,
    accessors=DatetimeArray._datetimelike_ops + ["unit"],
    typ="property",
)
@delegate_names(
    delegate=DatetimeArray,
    accessors=DatetimeArray._datetimelike_methods + ["as_unit"],
    typ="method",
)
class DatetimeProperties(Properties):
    """
    Accessor object for datetimelike properties of the Series values.

    Examples
    --------
    >>> seconds_series = pd.Series(pd.date_range("2000-01-01", periods=3, freq="s"))
    >>> seconds_series
    0   2000-01-01 00:00:00
    1   2000-01-01 00:00:01
    2   2000-01-01 00:00:02
    dtype: datetime64[ns]
    >>> seconds_series.dt.second
    0    0
    1    1
    2    2
    dtype: int32

    >>> hours_series = pd.Series(pd.date_range("2000-01-01", periods=3, freq="h"))
    >>> hours_series
    0   2000-01-01 00:00:00
    1   2000-01-01 01:00:00
    2   2000-01-01 02:00:00
    dtype: datetime64[ns]
    >>> hours_series.dt.hour
    0    0
    1    1
    2    2
    dtype: int32

    >>> quarters_series = pd.Series(pd.date_range("2000-01-01", periods=3, freq="QE"))
    >>> quarters_series
    0   2000-03-31
    1   2000-06-30
    2   2000-09-30
    dtype: datetime64[ns]
    >>> quarters_series.dt.quarter
    0    1
    1    2
    2    3
    dtype: int32

    Returns a Series indexed like the original Series.
    Raises TypeError if the Series does not contain datetimelike values.
    """

    def to_pydatetime(self) -> np.ndarray:
        """
        Return the data as an array of :class:`datetime.datetime` objects.

        .. deprecated:: 2.1.0

            The current behavior of dt.to_pydatetime is deprecated.
            In a future version this will return a Series containing python
            datetime objects instead of a ndarray.

        Timezone information is retained if present.

        .. warning::

           Python's datetime uses microsecond resolution, which is lower than
           pandas (nanosecond). The values are truncated.

        Returns
        -------
        numpy.ndarray
            Object dtype array containing native Python datetime objects.

        See Also
        --------
        datetime.datetime : Standard library value for a datetime.

        Examples
        --------
        >>> s = pd.Series(pd.date_range('20180310', periods=2))
        >>> s
        0   2018-03-10
        1   2018-03-11
        dtype: datetime64[ns]

        >>> s.dt.to_pydatetime()
        array([datetime.datetime(2018, 3, 10, 0, 0),
               datetime.datetime(2018, 3, 11, 0, 0)], dtype=object)

        pandas' nanosecond precision is truncated to microseconds.

        >>> s = pd.Series(pd.date_range('20180310', periods=2, freq='ns'))
        >>> s
        0   2018-03-10 00:00:00.000000000
        1   2018-03-10 00:00:00.000000001
        dtype: datetime64[ns]

        >>> s.dt.to_pydatetime()
        array([datetime.datetime(2018, 3, 10, 0, 0),
               datetime.datetime(2018, 3, 10, 0, 0)], dtype=object)
        """
        # GH#20306
        warnings.warn(
            f"The behavior of {type(self).__name__}.to_pydatetime is deprecated, "
            "in a future version this will return a Series containing python "
            "datetime objects instead of an ndarray. To retain the old behavior, "
            "call `np.array` on the result",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
        return self._get_values().to_pydatetime()

    @property
    def freq(self):
        return self._get_values().inferred_freq

    def isocalendar(self) -> DataFrame:
        """
        Calculate year, week, and day according to the ISO 8601 standard.

        Returns
        -------
        DataFrame
            With columns year, week and day.

        See Also
        --------
        Timestamp.isocalendar : Function return a 3-tuple containing ISO year,
            week number, and weekday for the given Timestamp object.
        datetime.date.isocalendar : Return a named tuple object with
            three components: year, week and weekday.

        Examples
        --------
        >>> ser = pd.to_datetime(pd.Series(["2010-01-01", pd.NaT]))
        >>> ser.dt.isocalendar()
           year  week  day
        0  2009    53     5
        1  <NA>  <NA>  <NA>
        >>> ser.dt.isocalendar().week
        0      53
        1    <NA>
        Name: week, dtype: UInt32
        """
        return self._get_values().isocalendar().set_index(self._parent.index)


@delegate_names(
    delegate=TimedeltaArray, accessors=TimedeltaArray._datetimelike_ops, typ="property"
)
@delegate_names(
    delegate=TimedeltaArray,
    accessors=TimedeltaArray._datetimelike_methods,
    typ="method",
)
class TimedeltaProperties(Properties):
    """
    Accessor object for datetimelike properties of the Series values.

    Returns a Series indexed like the original Series.
    Raises TypeError if the Series does not contain datetimelike values.

    Examples
    --------
    >>> seconds_series = pd.Series(
    ...     pd.timedelta_range(start="1 second", periods=3, freq="s")
    ... )
    >>> seconds_series
    0   0 days 00:00:01
    1   0 days 00:00:02
    2   0 days 00:00:03
    dtype: timedelta64[ns]
    >>> seconds_series.dt.seconds
    0    1
    1    2
    2    3
    dtype: int32
    """

    def to_pytimedelta(self) -> np.ndarray:
        """
        Return an array of native :class:`datetime.timedelta` objects.

        Python's standard `datetime` library uses a different representation
        timedelta's. This method converts a Series of pandas Timedeltas
        to `datetime.timedelta` format with the same length as the original
        Series.

        Returns
        -------
        numpy.ndarray
            Array of 1D containing data with `datetime.timedelta` type.

        See Also
        --------
        datetime.timedelta : A duration expressing the difference
            between two date, time, or datetime.

        Examples
        --------
        >>> s = pd.Series(pd.to_timedelta(np.arange(5), unit="d"))
        >>> s
        0   0 days
        1   1 days
        2   2 days
        3   3 days
        4   4 days
        dtype: timedelta64[ns]

        >>> s.dt.to_pytimedelta()
        array([datetime.timedelta(0), datetime.timedelta(days=1),
        datetime.timedelta(days=2), datetime.timedelta(days=3),
        datetime.timedelta(days=4)], dtype=object)
        """
        return self._get_values().to_pytimedelta()

    @property
    def components(self):
        """
        Return a Dataframe of the components of the Timedeltas.

        Returns
        -------
        DataFrame

        Examples
        --------
        >>> s = pd.Series(pd.to_timedelta(np.arange(5), unit='s'))
        >>> s
        0   0 days 00:00:00
        1   0 days 00:00:01
        2   0 days 00:00:02
        3   0 days 00:00:03
        4   0 days 00:00:04
        dtype: timedelta64[ns]
        >>> s.dt.components
           days  hours  minutes  seconds  milliseconds  microseconds  nanoseconds
        0     0      0        0        0             0             0            0
        1     0      0        0        1             0             0            0
        2     0      0        0        2             0             0            0
        3     0      0        0        3             0             0            0
        4     0      0        0        4             0             0            0
        """
        return (
            self._get_values()
            .components.set_index(self._parent.index)
            .__finalize__(self._parent)
        )

    @property
    def freq(self):
        return self._get_values().inferred_freq


@delegate_names(
    delegate=PeriodArray, accessors=PeriodArray._datetimelike_ops, typ="property"
)
@delegate_names(
    delegate=PeriodArray, accessors=PeriodArray._datetimelike_methods, typ="method"
)
class PeriodProperties(Properties):
    """
    Accessor object for datetimelike properties of the Series values.

    Returns a Series indexed like the original Series.
    Raises TypeError if the Series does not contain datetimelike values.

    Examples
    --------
    >>> seconds_series = pd.Series(
    ...     pd.period_range(
    ...         start="2000-01-01 00:00:00", end="2000-01-01 00:00:03", freq="s"
    ...     )
    ... )
    >>> seconds_series
    0    2000-01-01 00:00:00
    1    2000-01-01 00:00:01
    2    2000-01-01 00:00:02
    3    2000-01-01 00:00:03
    dtype: period[s]
    >>> seconds_series.dt.second
    0    0
    1    1
    2    2
    3    3
    dtype: int64

    >>> hours_series = pd.Series(
    ...     pd.period_range(start="2000-01-01 00:00", end="2000-01-01 03:00", freq="h")
    ... )
    >>> hours_series
    0    2000-01-01 00:00
    1    2000-01-01 01:00
    2    2000-01-01 02:00
    3    2000-01-01 03:00
    dtype: period[h]
    >>> hours_series.dt.hour
    0    0
    1    1
    2    2
    3    3
    dtype: int64

    >>> quarters_series = pd.Series(
    ...     pd.period_range(start="2000-01-01", end="2000-12-31", freq="Q-DEC")
    ... )
    >>> quarters_series
    0    2000Q1
    1    2000Q2
    2    2000Q3
    3    2000Q4
    dtype: period[Q-DEC]
    >>> quarters_series.dt.quarter
    0    1
    1    2
    2    3
    3    4
    dtype: int64
    """


class CombinedDatetimelikeProperties(
    DatetimeProperties, TimedeltaProperties, PeriodProperties
):
    def __new__(cls, data: Series):  # pyright: ignore[reportInconsistentConstructor]
        # CombinedDatetimelikeProperties isn't really instantiated. Instead
        # we need to choose which parent (datetime or timedelta) is
        # appropriate. Since we're checking the dtypes anyway, we'll just
        # do all the validation here.

        if not isinstance(data, ABCSeries):
            raise TypeError(
                f"cannot convert an object of type {type(data)} to a datetimelike index"
            )

        orig = data if isinstance(data.dtype, CategoricalDtype) else None
        if orig is not None:
            data = data._constructor(
                orig.array,
                name=orig.name,
                copy=False,
                dtype=orig._values.categories.dtype,
                index=orig.index,
            )

        if isinstance(data.dtype, ArrowDtype) and data.dtype.kind in "Mm":
            return ArrowTemporalProperties(data, orig)
        if lib.is_np_dtype(data.dtype, "M"):
            return DatetimeProperties(data, orig)
        elif isinstance(data.dtype, DatetimeTZDtype):
            return DatetimeProperties(data, orig)
        elif lib.is_np_dtype(data.dtype, "m"):
            return TimedeltaProperties(data, orig)
        elif isinstance(data.dtype, PeriodDtype):
            return PeriodProperties(data, orig)

        raise AttributeError("Can only use .dt accessor with datetimelike values")
<|endoftext|>from __future__ import annotations

import datetime as dt
import operator
from typing import TYPE_CHECKING
import warnings

import numpy as np
import pytz

from pandas._libs import (
    NaT,
    Period,
    Timestamp,
    index as libindex,
    lib,
)
from pandas._libs.tslibs import (
    Resolution,
    Tick,
    Timedelta,
    periods_per_day,
    timezones,
    to_offset,
)
from pandas._libs.tslibs.offsets import prefix_mapping
from pandas.util._decorators import (
    cache_readonly,
    doc,
)
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import is_scalar
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.dtypes.missing import is_valid_na_for_dtype

from pandas.core.arrays.datetimes import (
    DatetimeArray,
    tz_to_dtype,
)
import pandas.core.common as com
from pandas.core.indexes.base import (
    Index,
    maybe_extract_name,
)
from pandas.core.indexes.datetimelike import DatetimeTimedeltaMixin
from pandas.core.indexes.extension import inherit_names
from pandas.core.tools.times import to_time

if TYPE_CHECKING:
    from collections.abc import Hashable

    from pandas._typing import (
        Dtype,
        DtypeObj,
        Frequency,
        IntervalClosedType,
        Self,
        TimeAmbiguous,
        TimeNonexistent,
        npt,
    )

    from pandas.core.api import (
        DataFrame,
        PeriodIndex,
    )

from pandas._libs.tslibs.dtypes import OFFSET_TO_PERIOD_FREQSTR


def _new_DatetimeIndex(cls, d):
    """
    This is called upon unpickling, rather than the default which doesn't
    have arguments and breaks __new__
    """
    if "data" in d and not isinstance(d["data"], DatetimeIndex):
        # Avoid need to verify integrity by calling simple_new directly
        data = d.pop("data")
        if not isinstance(data, DatetimeArray):
            # For backward compat with older pickles, we may need to construct
            #  a DatetimeArray to adapt to the newer _simple_new signature
            tz = d.pop("tz")
            freq = d.pop("freq")
            dta = DatetimeArray._simple_new(data, dtype=tz_to_dtype(tz), freq=freq)
        else:
            dta = data
            for key in ["tz", "freq"]:
                # These are already stored in our DatetimeArray; if they are
                #  also in the pickle and don't match, we have a problem.
                if key in d:
                    assert d[key] == getattr(dta, key)
                    d.pop(key)
        result = cls._simple_new(dta, **d)
    else:
        with warnings.catch_warnings():
            # TODO: If we knew what was going in to **d, we might be able to
            #  go through _simple_new instead
            warnings.simplefilter("ignore")
            result = cls.__new__(cls, **d)

    return result


@inherit_names(
    DatetimeArray._field_ops
    + [
        method
        for method in DatetimeArray._datetimelike_methods
        if method not in ("tz_localize", "tz_convert", "strftime")
    ],
    DatetimeArray,
    wrap=True,
)
@inherit_names(["is_normalized"], DatetimeArray, cache=True)
@inherit_names(
    [
        "tz",
        "tzinfo",
        "dtype",
        "to_pydatetime",
        "date",
        "time",
        "timetz",
        "std",
    ]
    + DatetimeArray._bool_ops,
    DatetimeArray,
)
class DatetimeIndex(DatetimeTimedeltaMixin):
    """
    Immutable ndarray-like of datetime64 data.

    Represented internally as int64, and which can be boxed to Timestamp objects
    that are subclasses of datetime and carry metadata.

    .. versionchanged:: 2.0.0
        The various numeric date/time attributes (:attr:`~DatetimeIndex.day`,
        :attr:`~DatetimeIndex.month`, :attr:`~DatetimeIndex.year` etc.) now have dtype
        ``int32``. Previously they had dtype ``int64``.

    Parameters
    ----------
    data : array-like (1-dimensional)
        Datetime-like data to construct index with.
    freq : str or pandas offset object, optional
        One of pandas date offset strings or corresponding objects. The string
        'infer' can be passed in order to set the frequency of the index as the
        inferred frequency upon creation.
    tz : pytz.timezone or dateutil.tz.tzfile or datetime.tzinfo or str
        Set the Timezone of the data.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.

        .. deprecated:: 2.1.0

    closed : {'left', 'right'}, optional
        Set whether to include `start` and `end` that are on the
        boundary. The default includes boundary points on either end.

        .. deprecated:: 2.1.0

    ambiguous : 'infer', bool-ndarray, 'NaT', default 'raise'
        When clocks moved backward due to DST, ambiguous times may arise.
        For example in Central European Time (UTC+01), when going from 03:00
        DST to 02:00 non-DST, 02:30:00 local time occurs both at 00:30:00 UTC
        and at 01:30:00 UTC. In such a situation, the `ambiguous` parameter
        dictates how ambiguous times should be handled.

        - 'infer' will attempt to infer fall dst-transition hours based on
          order
        - bool-ndarray where True signifies a DST time, False signifies a
          non-DST time (note that this flag is only applicable for ambiguous
          times)
        - 'NaT' will return NaT where there are ambiguous times
        - 'raise' will raise an AmbiguousTimeError if there are ambiguous times.
    dayfirst : bool, default False
        If True, parse dates in `data` with the day first order.
    yearfirst : bool, default False
        If True parse dates in `data` with the year first order.
    dtype : numpy.dtype or DatetimeTZDtype or str, default None
        Note that the only NumPy dtype allowed is `datetime64[ns]`.
    copy : bool, default False
        Make a copy of input ndarray.
    name : label, default None
        Name to be stored in the index.

    Attributes
    ----------
    year
    month
    day
    hour
    minute
    second
    microsecond
    nanosecond
    date
    time
    timetz
    dayofyear
    day_of_year
    dayofweek
    day_of_week
    weekday
    quarter
    tz
    freq
    freqstr
    is_month_start
    is_month_end
    is_quarter_start
    is_quarter_end
    is_year_start
    is_year_end
    is_leap_year
    inferred_freq

    Methods
    -------
    normalize
    strftime
    snap
    tz_convert
    tz_localize
    round
    floor
    ceil
    to_period
    to_pydatetime
    to_series
    to_frame
    month_name
    day_name
    mean
    std

    See Also
    --------
    Index : The base pandas Index type.
    TimedeltaIndex : Index of timedelta64 data.
    PeriodIndex : Index of Period data.
    to_datetime : Convert argument to datetime.
    date_range : Create a fixed-frequency DatetimeIndex.

    Notes
    -----
    To learn more about the frequency strings, please see `this link
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__.

    Examples
    --------
    >>> idx = pd.DatetimeIndex(["1/1/2020 10:00:00+00:00", "2/1/2020 11:00:00+00:00"])
    >>> idx
    DatetimeIndex(['2020-01-01 10:00:00+00:00', '2020-02-01 11:00:00+00:00'],
    dtype='datetime64[ns, UTC]', freq=None)
    """

    _typ = "datetimeindex"

    _data_cls = DatetimeArray
    _supports_partial_string_indexing = True

    @property
    def _engine_type(self) -> type[libindex.DatetimeEngine]:
        return libindex.DatetimeEngine

    _data: DatetimeArray
    _values: DatetimeArray
    tz: dt.tzinfo | None

    # --------------------------------------------------------------------
    # methods that dispatch to DatetimeArray and wrap result

    @doc(DatetimeArray.strftime)
    def strftime(self, date_format) -> Index:
        arr = self._data.strftime(date_format)
        return Index(arr, name=self.name, dtype=arr.dtype)

    @doc(DatetimeArray.tz_convert)
    def tz_convert(self, tz) -> Self:
        arr = self._data.tz_convert(tz)
        return type(self)._simple_new(arr, name=self.name, refs=self._references)

    @doc(DatetimeArray.tz_localize)
    def tz_localize(
        self,
        tz,
        ambiguous: TimeAmbiguous = "raise",
        nonexistent: TimeNonexistent = "raise",
    ) -> Self:
        arr = self._data.tz_localize(tz, ambiguous, nonexistent)
        return type(self)._simple_new(arr, name=self.name)

    @doc(DatetimeArray.to_period)
    def to_period(self, freq=None) -> PeriodIndex:
        from pandas.core.indexes.api import PeriodIndex

        arr = self._data.to_period(freq)
        return PeriodIndex._simple_new(arr, name=self.name)

    @doc(DatetimeArray.to_julian_date)
    def to_julian_date(self) -> Index:
        arr = self._data.to_julian_date()
        return Index._simple_new(arr, name=self.name)

    @doc(DatetimeArray.isocalendar)
    def isocalendar(self) -> DataFrame:
        df = self._data.isocalendar()
        return df.set_index(self)

    @cache_readonly
    def _resolution_obj(self) -> Resolution:
        return self._data._resolution_obj

    # --------------------------------------------------------------------
    # Constructors

    def __new__(
        cls,
        data=None,
        freq: Frequency | lib.NoDefault = lib.no_default,
        tz=lib.no_default,
        normalize: bool | lib.NoDefault = lib.no_default,
        closed=lib.no_default,
        ambiguous: TimeAmbiguous = "raise",
        dayfirst: bool = False,
        yearfirst: bool = False,
        dtype: Dtype | None = None,
        copy: bool = False,
        name: Hashable | None = None,
    ) -> Self:
        if closed is not lib.no_default:
            # GH#52628
            warnings.warn(
                f"The 'closed' keyword in {cls.__name__} construction is "
                "deprecated and will be removed in a future version.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )
        if normalize is not lib.no_default:
            # GH#52628
            warnings.warn(
                f"The 'normalize' keyword in {cls.__name__} construction is "
                "deprecated and will be removed in a future version.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )

        if is_scalar(data):
            cls._raise_scalar_data_error(data)

        # - Cases checked above all return/raise before reaching here - #

        name = maybe_extract_name(name, data, cls)

        if (
            isinstance(data, DatetimeArray)
            and freq is lib.no_default
            and tz is lib.no_default
            and dtype is None
        ):
            # fastpath, similar logic in TimedeltaIndex.__new__;
            # Note in this particular case we retain non-nano.
            if copy:
                data = data.copy()
            return cls._simple_new(data, name=name)

        dtarr = DatetimeArray._from_sequence_not_strict(
            data,
            dtype=dtype,
            copy=copy,
            tz=tz,
            freq=freq,
            dayfirst=dayfirst,
            yearfirst=yearfirst,
            ambiguous=ambiguous,
        )
        refs = None
        if not copy and isinstance(data, (Index, ABCSeries)):
            refs = data._references

        subarr = cls._simple_new(dtarr, name=name, refs=refs)
        return subarr

    # --------------------------------------------------------------------

    @cache_readonly
    def _is_dates_only(self) -> bool:
        """
        Return a boolean if we are only dates (and don't have a timezone)

        Returns
        -------
        bool
        """
        if isinstance(self.freq, Tick):
            delta = Timedelta(self.freq)

            if delta % dt.timedelta(days=1) != dt.timedelta(days=0):
                return False

        return self._values._is_dates_only

    def __reduce__(self):
        d = {"data": self._data, "name": self.name}
        return _new_DatetimeIndex, (type(self), d), None

    def _is_comparable_dtype(self, dtype: DtypeObj) -> bool:
        """
        Can we compare values of the given dtype to our own?
        """
        if self.tz is not None:
            # If we have tz, we can compare to tzaware
            return isinstance(dtype, DatetimeTZDtype)
        # if we dont have tz, we can only compare to tznaive
        return lib.is_np_dtype(dtype, "M")

    # --------------------------------------------------------------------
    # Rendering Methods

    @cache_readonly
    def _formatter_func(self):
        # Note this is equivalent to the DatetimeIndexOpsMixin method but
        #  uses the maybe-cached self._is_dates_only instead of re-computing it.
        from pandas.io.formats.format import get_format_datetime64

        formatter = get_format_datetime64(is_dates_only=self._is_dates_only)
        return lambda x: f"'{formatter(x)}'"

    # --------------------------------------------------------------------
    # Set Operation Methods

    def _can_range_setop(self, other) -> bool:
        # GH 46702: If self or other have non-UTC tzs, DST transitions prevent
        # range representation due to no singular step
        if (
            self.tz is not None
            and not timezones.is_utc(self.tz)
            and not timezones.is_fixed_offset(self.tz)
        ):
            return False
        if (
            other.tz is not None
            and not timezones.is_utc(other.tz)
            and not timezones.is_fixed_offset(other.tz)
        ):
            return False
        return super()._can_range_setop(other)

    # --------------------------------------------------------------------

    def _get_time_micros(self) -> npt.NDArray[np.int64]:
        """
        Return the number of microseconds since midnight.

        Returns
        -------
        ndarray[int64_t]
        """
        values = self._data._local_timestamps()

        ppd = periods_per_day(self._data._creso)

        frac = values % ppd
        if self.unit == "ns":
            micros = frac // 1000
        elif self.unit == "us":
            micros = frac
        elif self.unit == "ms":
            micros = frac * 1000
        elif self.unit == "s":
            micros = frac * 1_000_000
        else:  # pragma: no cover
            raise NotImplementedError(self.unit)

        micros[self._isnan] = -1
        return micros

    def snap(self, freq: Frequency = "S") -> DatetimeIndex:
        """
        Snap time stamps to nearest occurring frequency.

        Returns
        -------
        DatetimeIndex

        Examples
        --------
        >>> idx = pd.DatetimeIndex(['2023-01-01', '2023-01-02',
        ...                        '2023-02-01', '2023-02-02'])
        >>> idx
        DatetimeIndex(['2023-01-01', '2023-01-02', '2023-02-01', '2023-02-02'],
        dtype='datetime64[ns]', freq=None)
        >>> idx.snap('MS')
        DatetimeIndex(['2023-01-01', '2023-01-01', '2023-02-01', '2023-02-01'],
        dtype='datetime64[ns]', freq=None)
        """
        # Superdumb, punting on any optimizing
        freq = to_offset(freq)

        dta = self._data.copy()

        for i, v in enumerate(self):
            s = v
            if not freq.is_on_offset(s):
                t0 = freq.rollback(s)
                t1 = freq.rollforward(s)
                if abs(s - t0) < abs(t1 - s):
                    s = t0
                else:
                    s = t1
            dta[i] = s

        return DatetimeIndex._simple_new(dta, name=self.name)

    # --------------------------------------------------------------------
    # Indexing Methods

    def _parsed_string_to_bounds(self, reso: Resolution, parsed: dt.datetime):
        """
        Calculate datetime bounds for parsed time string and its resolution.

        Parameters
        ----------
        reso : Resolution
            Resolution provided by parsed string.
        parsed : datetime
            Datetime from parsed string.

        Returns
        -------
        lower, upper: pd.Timestamp
        """
        freq = OFFSET_TO_PERIOD_FREQSTR.get(reso.attr_abbrev, reso.attr_abbrev)
        per = Period(parsed, freq=freq)
        start, end = per.start_time, per.end_time

        # GH 24076
        # If an incoming date string contained a UTC offset, need to localize
        # the parsed date to this offset first before aligning with the index's
        # timezone
        start = start.tz_localize(parsed.tzinfo)
        end = end.tz_localize(parsed.tzinfo)

        if parsed.tzinfo is not None:
            if self.tz is None:
                raise ValueError(
                    "The index must be timezone aware when indexing "
                    "with a date string with a UTC offset"
                )
        # The flipped case with parsed.tz is None and self.tz is not None
        #  is ruled out bc parsed and reso are produced by _parse_with_reso,
        #  which localizes parsed.
        return start, end

    def _parse_with_reso(self, label: str):
        parsed, reso = super()._parse_with_reso(label)

        parsed = Timestamp(parsed)

        if self.tz is not None and parsed.tzinfo is None:
            # we special-case timezone-naive strings and timezone-aware
            #  DatetimeIndex
            # https://github.com/pandas-dev/pandas/pull/36148#issuecomment-687883081
            parsed = parsed.tz_localize(self.tz)

        return parsed, reso

    def _disallow_mismatched_indexing(self, key) -> None:
        """
        Check for mismatched-tzawareness indexing and re-raise as KeyError.
        """
        # we get here with isinstance(key, self._data._recognized_scalars)
        try:
            # GH#36148
            self._data._assert_tzawareness_compat(key)
        except TypeError as err:
            raise KeyError(key) from err

    def get_loc(self, key):
        """
        Get integer location for requested label

        Returns
        -------
        loc : int
        """
        self._check_indexing_error(key)

        orig_key = key
        if is_valid_na_for_dtype(key, self.dtype):
            key = NaT

        if isinstance(key, self._data._recognized_scalars):
            # needed to localize naive datetimes
            self._disallow_mismatched_indexing(key)
            key = Timestamp(key)

        elif isinstance(key, str):
            try:
                parsed, reso = self._parse_with_reso(key)
            except (ValueError, pytz.NonExistentTimeError) as err:
                raise KeyError(key) from err
            self._disallow_mismatched_indexing(parsed)

            if self._can_partial_date_slice(reso):
                try:
                    return self._partial_date_slice(reso, parsed)
                except KeyError as err:
                    raise KeyError(key) from err

            key = parsed

        elif isinstance(key, dt.timedelta):
            # GH#20464
            raise TypeError(
                f"Cannot index {type(self).__name__} with {type(key).__name__}"
            )

        elif isinstance(key, dt.time):
            return self.indexer_at_time(key)

        else:
            # unrecognized type
            raise KeyError(key)

        try:
            return Index.get_loc(self, key)
        except KeyError as err:
            raise KeyError(orig_key) from err

    @doc(DatetimeTimedeltaMixin._maybe_cast_slice_bound)
    def _maybe_cast_slice_bound(self, label, side: str):
        # GH#42855 handle date here instead of get_slice_bound
        if isinstance(label, dt.date) and not isinstance(label, dt.datetime):
            # Pandas supports slicing with dates, treated as datetimes at midnight.
            # https://github.com/pandas-dev/pandas/issues/31501
            label = Timestamp(label).to_pydatetime()

        label = super()._maybe_cast_slice_bound(label, side)
        self._data._assert_tzawareness_compat(label)
        return Timestamp(label)

    def slice_indexer(self, start=None, end=None, step=None):
        """
        Return indexer for specified label slice.
        Index.slice_indexer, customized to handle time slicing.

        In addition to functionality provided by Index.slice_indexer, does the
        following:

        - if both `start` and `end` are instances of `datetime.time`, it
          invokes `indexer_between_time`
        - if `start` and `end` are both either string or None perform
          value-based selection in non-monotonic cases.

        """
        # For historical reasons DatetimeIndex supports slices between two
        # instances of datetime.time as if it were applying a slice mask to
        # an array of (self.hour, self.minute, self.seconds, self.microsecond).
        if isinstance(start, dt.time) and isinstance(end, dt.time):
            if step is not None and step != 1:
                raise ValueError("Must have step size of 1 with time slices")
            return self.indexer_between_time(start, end)

        if isinstance(start, dt.time) or isinstance(end, dt.time):
            raise KeyError("Cannot mix time and non-time slice keys")

        def check_str_or_none(point) -> bool:
            return point is not None and not isinstance(point, str)

        # GH#33146 if start and end are combinations of str and None and Index is not
        # monotonic, we can not use Index.slice_indexer because it does not honor the
        # actual elements, is only searching for start and end
        if (
            check_str_or_none(start)
            or check_str_or_none(end)
            or self.is_monotonic_increasing
        ):
            return Index.slice_indexer(self, start, end, step)

        mask = np.array(True)
        in_index = True
        if start is not None:
            start_casted = self._maybe_cast_slice_bound(start, "left")
            mask = start_casted <= self
            in_index &= (start_casted == self).any()

        if end is not None:
            end_casted = self._maybe_cast_slice_bound(end, "right")
            mask = (self <= end_casted) & mask
            in_index &= (end_casted == self).any()

        if not in_index:
            raise KeyError(
                "Value based partial slicing on non-monotonic DatetimeIndexes "
                "with non-existing keys is not allowed.",
            )
        indexer = mask.nonzero()[0][::step]
        if len(indexer) == len(self):
            return slice(None)
        else:
            return indexer

    # --------------------------------------------------------------------

    @property
    def inferred_type(self) -> str:
        # b/c datetime is represented as microseconds since the epoch, make
        # sure we can't have ambiguous indexing
        return "datetime64"

    def indexer_at_time(self, time, asof: bool = False) -> npt.NDArray[np.intp]:
        """
        Return index locations of values at particular time of day.

        Parameters
        ----------
        time : datetime.time or str
            Time passed in either as object (datetime.time) or as string in
            appropriate format ("%H:%M", "%H%M", "%I:%M%p", "%I%M%p",
            "%H:%M:%S", "%H%M%S", "%I:%M:%S%p", "%I%M%S%p").

        Returns
        -------
        np.ndarray[np.intp]

        See Also
        --------
        indexer_between_time : Get index locations of values between particular
            times of day.
        DataFrame.at_time : Select values at particular time of day.

        Examples
        --------
        >>> idx = pd.DatetimeIndex(["1/1/2020 10:00", "2/1/2020 11:00",
        ...                         "3/1/2020 10:00"])
        >>> idx.indexer_at_time("10:00")
        array([0, 2])
        """
        if asof:
            raise NotImplementedError("'asof' argument is not supported")

        if isinstance(time, str):
            from dateutil.parser import parse

            time = parse(time).time()

        if time.tzinfo:
            if self.tz is None:
                raise ValueError("Index must be timezone aware.")
            time_micros = self.tz_convert(time.tzinfo)._get_time_micros()
        else:
            time_micros = self._get_time_micros()
        micros = _time_to_micros(time)
        return (time_micros == micros).nonzero()[0]

    def indexer_between_time(
        self, start_time, end_time, include_start: bool = True, include_end: bool = True
    ) -> npt.NDArray[np.intp]:
        """
        Return index locations of values between particular times of day.

        Parameters
        ----------
        start_time, end_time : datetime.time, str
            Time passed either as object (datetime.time) or as string in
            appropriate format ("%H:%M", "%H%M", "%I:%M%p", "%I%M%p",
            "%H:%M:%S", "%H%M%S", "%I:%M:%S%p","%I%M%S%p").
        include_start : bool, default True
        include_end : bool, default True

        Returns
        -------
        np.ndarray[np.intp]

        See Also
        --------
        indexer_at_time : Get index locations of values at particular time of day.
        DataFrame.between_time : Select values between particular times of day.

        Examples
        --------
        >>> idx = pd.date_range("2023-01-01", periods=4, freq="h")
        >>> idx
        DatetimeIndex(['2023-01-01 00:00:00', '2023-01-01 01:00:00',
                           '2023-01-01 02:00:00', '2023-01-01 03:00:00'],
                          dtype='datetime64[ns]', freq='h')
        >>> idx.indexer_between_time("00:00", "2:00", include_end=False)
        array([0, 1])
        """
        start_time = to_time(start_time)
        end_time = to_time(end_time)
        time_micros = self._get_time_micros()
        start_micros = _time_to_micros(start_time)
        end_micros = _time_to_micros(end_time)

        if include_start and include_end:
            lop = rop = operator.le
        elif include_start:
            lop = operator.le
            rop = operator.lt
        elif include_end:
            lop = operator.lt
            rop = operator.le
        else:
            lop = rop = operator.lt

        if start_time <= end_time:
            join_op = operator.and_
        else:
            join_op = operator.or_

        mask = join_op(lop(start_micros, time_micros), rop(time_micros, end_micros))

        return mask.nonzero()[0]


def date_range(
    start=None,
    end=None,
    periods=None,
    freq=None,
    tz=None,
    normalize: bool = False,
    name: Hashable | None = None,
    inclusive: IntervalClosedType = "both",
    *,
    unit: str | None = None,
    **kwargs,
) -> DatetimeIndex:
    """
    Return a fixed frequency DatetimeIndex.

    Returns the range of equally spaced time points (where the difference between any
    two adjacent points is specified by the given frequency) such that they all
    satisfy `start <[=] x <[=] end`, where the first one and the last one are, resp.,
    the first and last time points in that range that fall on the boundary of ``freq``
    (if given as a frequency string) or that are valid for ``freq`` (if given as a
    :class:`pandas.tseries.offsets.DateOffset`). (If exactly one of ``start``,
    ``end``, or ``freq`` is *not* specified, this missing parameter can be computed
    given ``periods``, the number of timesteps in the range. See the note below.)

    Parameters
    ----------
    start : str or datetime-like, optional
        Left bound for generating dates.
    end : str or datetime-like, optional
        Right bound for generating dates.
    periods : int, optional
        Number of periods to generate.
    freq : str, Timedelta, datetime.timedelta, or DateOffset, default 'D'
        Frequency strings can have multiples, e.g. '5h'. See
        :ref:`here <timeseries.offset_aliases>` for a list of
        frequency aliases.
    tz : str or tzinfo, optional
        Time zone name for returning localized DatetimeIndex, for example
        'Asia/Hong_Kong'. By default, the resulting DatetimeIndex is
        timezone-naive unless timezone-aware datetime-likes are passed.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    name : str, default None
        Name of the resulting DatetimeIndex.
    inclusive : {"both", "neither", "left", "right"}, default "both"
        Include boundaries; Whether to set each bound as closed or open.

        .. versionadded:: 1.4.0
    unit : str, default None
        Specify the desired resolution of the result.

        .. versionadded:: 2.0.0
    **kwargs
        For compatibility. Has no effect on the result.

    Returns
    -------
    DatetimeIndex

    See Also
    --------
    DatetimeIndex : An immutable container for datetimes.
    timedelta_range : Return a fixed frequency TimedeltaIndex.
    period_range : Return a fixed frequency PeriodIndex.
    interval_range : Return a fixed frequency IntervalIndex.

    Notes
    -----
    Of the four parameters ``start``, ``end``, ``periods``, and ``freq``,
    exactly three must be specified. If ``freq`` is omitted, the resulting
    ``DatetimeIndex`` will have ``periods`` linearly spaced elements between
    ``start`` and ``end`` (closed on both sides).

    To learn more about the frequency strings, please see `this link
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__.

    Examples
    --------
    **Specifying the values**

    The next four examples generate the same `DatetimeIndex`, but vary
    the combination of `start`, `end` and `periods`.

    Specify `start` and `end`, with the default daily frequency.

    >>> pd.date_range(start='1/1/2018', end='1/08/2018')
    DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04',
                   '2018-01-05', '2018-01-06', '2018-01-07', '2018-01-08'],
                  dtype='datetime64[ns]', freq='D')

    Specify timezone-aware `start` and `end`, with the default daily frequency.

    >>> pd.date_range(
    ...     start=pd.to_datetime("1/1/2018").tz_localize("Europe/Berlin"),
    ...     end=pd.to_datetime("1/08/2018").tz_localize("Europe/Berlin"),
    ... )
    DatetimeIndex(['2018-01-01 00:00:00+01:00', '2018-01-02 00:00:00+01:00',
                   '2018-01-03 00:00:00+01:00', '2018-01-04 00:00:00+01:00',
                   '2018-01-05 00:00:00+01:00', '2018-01-06 00:00:00+01:00',
                   '2018-01-07 00:00:00+01:00', '2018-01-08 00:00:00+01:00'],
                  dtype='datetime64[ns, Europe/Berlin]', freq='D')

    Specify `start` and `periods`, the number of periods (days).

    >>> pd.date_range(start='1/1/2018', periods=8)
    DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04',
                   '2018-01-05', '2018-01-06', '2018-01-07', '2018-01-08'],
                  dtype='datetime64[ns]', freq='D')

    Specify `end` and `periods`, the number of periods (days).

    >>> pd.date_range(end='1/1/2018', periods=8)
    DatetimeIndex(['2017-12-25', '2017-12-26', '2017-12-27', '2017-12-28',
                   '2017-12-29', '2017-12-30', '2017-12-31', '2018-01-01'],
                  dtype='datetime64[ns]', freq='D')

    Specify `start`, `end`, and `periods`; the frequency is generated
    automatically (linearly spaced).

    >>> pd.date_range(start='2018-04-24', end='2018-04-27', periods=3)
    DatetimeIndex(['2018-04-24 00:00:00', '2018-04-25 12:00:00',
                   '2018-04-27 00:00:00'],
                  dtype='datetime64[ns]', freq=None)

    **Other Parameters**

    Changed the `freq` (frequency) to ``'ME'`` (month end frequency).

    >>> pd.date_range(start='1/1/2018', periods=5, freq='ME')
    DatetimeIndex(['2018-01-31', '2018-02-28', '2018-03-31', '2018-04-30',
                   '2018-05-31'],
                  dtype='datetime64[ns]', freq='ME')

    Multiples are allowed

    >>> pd.date_range(start='1/1/2018', periods=5, freq='3ME')
    DatetimeIndex(['2018-01-31', '2018-04-30', '2018-07-31', '2018-10-31',
                   '2019-01-31'],
                  dtype='datetime64[ns]', freq='3ME')

    `freq` can also be specified as an Offset object.

    >>> pd.date_range(start='1/1/2018', periods=5, freq=pd.offsets.MonthEnd(3))
    DatetimeIndex(['2018-01-31', '2018-04-30', '2018-07-31', '2018-10-31',
                   '2019-01-31'],
                  dtype='datetime64[ns]', freq='3ME')

    Specify `tz` to set the timezone.

    >>> pd.date_range(start='1/1/2018', periods=5, tz='Asia/Tokyo')
    DatetimeIndex(['2018-01-01 00:00:00+09:00', '2018-01-02 00:00:00+09:00',
                   '2018-01-03 00:00:00+09:00', '2018-01-04 00:00:00+09:00',
                   '2018-01-05 00:00:00+09:00'],
                  dtype='datetime64[ns, Asia/Tokyo]', freq='D')

    `inclusive` controls whether to include `start` and `end` that are on the
    boundary. The default, "both", includes boundary points on either end.

    >>> pd.date_range(start='2017-01-01', end='2017-01-04', inclusive="both")
    DatetimeIndex(['2017-01-01', '2017-01-02', '2017-01-03', '2017-01-04'],
                  dtype='datetime64[ns]', freq='D')

    Use ``inclusive='left'`` to exclude `end` if it falls on the boundary.

    >>> pd.date_range(start='2017-01-01', end='2017-01-04', inclusive='left')
    DatetimeIndex(['2017-01-01', '2017-01-02', '2017-01-03'],
                  dtype='datetime64[ns]', freq='D')

    Use ``inclusive='right'`` to exclude `start` if it falls on the boundary, and
    similarly ``inclusive='neither'`` will exclude both `start` and `end`.

    >>> pd.date_range(start='2017-01-01', end='2017-01-04', inclusive='right')
    DatetimeIndex(['2017-01-02', '2017-01-03', '2017-01-04'],
                  dtype='datetime64[ns]', freq='D')

    **Specify a unit**

    >>> pd.date_range(start="2017-01-01", periods=10, freq="100YS", unit="s")
    DatetimeIndex(['2017-01-01', '2117-01-01', '2217-01-01', '2317-01-01',
                   '2417-01-01', '2517-01-01', '2617-01-01', '2717-01-01',
                   '2817-01-01', '2917-01-01'],
                  dtype='datetime64[s]', freq='100YS-JAN')
    """
    if freq is None and com.any_none(periods, start, end):
        freq = "D"

    dtarr = DatetimeArray._generate_range(
        start=start,
        end=end,
        periods=periods,
        freq=freq,
        tz=tz,
        normalize=normalize,
        inclusive=inclusive,
        unit=unit,
        **kwargs,
    )
    return DatetimeIndex._simple_new(dtarr, name=name)


def bdate_range(
    start=None,
    end=None,
    periods: int | None = None,
    freq: Frequency | dt.timedelta = "B",
    tz=None,
    normalize: bool = True,
    name: Hashable | None = None,
    weekmask=None,
    holidays=None,
    inclusive: IntervalClosedType = "both",
    **kwargs,
) -> DatetimeIndex:
    """
    Return a fixed frequency DatetimeIndex with business day as the default.

    Parameters
    ----------
    start : str or datetime-like, default None
        Left bound for generating dates.
    end : str or datetime-like, default None
        Right bound for generating dates.
    periods : int, default None
        Number of periods to generate.
    freq : str, Timedelta, datetime.timedelta, or DateOffset, default 'B'
        Frequency strings can have multiples, e.g. '5h'. The default is
        business daily ('B').
    tz : str or None
        Time zone name for returning localized DatetimeIndex, for example
        Asia/Beijing.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    name : str, default None
        Name of the resulting DatetimeIndex.
    weekmask : str or None, default None
        Weekmask of valid business days, passed to ``numpy.busdaycalendar``,
        only used when custom frequency strings are passed.  The default
        value None is equivalent to 'Mon Tue Wed Thu Fri'.
    holidays : list-like or None, default None
        Dates to exclude from the set of valid business days, passed to
        ``numpy.busdaycalendar``, only used when custom frequency strings
        are passed.
    inclusive : {"both", "neither", "left", "right"}, default "both"
        Include boundaries; Whether to set each bound as closed or open.

        .. versionadded:: 1.4.0
    **kwargs
        For compatibility. Has no effect on the result.

    Returns
    -------
    DatetimeIndex

    Notes
    -----
    Of the four parameters: ``start``, ``end``, ``periods``, and ``freq``,
    exactly three must be specified.  Specifying ``freq`` is a requirement
    for ``bdate_range``.  Use ``date_range`` if specifying ``freq`` is not
    desired.

    To learn more about the frequency strings, please see `this link
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__.

    Examples
    --------
    Note how the two weekend days are skipped in the result.

    >>> pd.bdate_range(start='1/1/2018', end='1/08/2018')
    DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04',
               '2018-01-05', '2018-01-08'],
              dtype='datetime64[ns]', freq='B')
    """
    if freq is None:
        msg = "freq must be specified for bdate_range; use date_range instead"
        raise TypeError(msg)

    if isinstance(freq, str) and freq.startswith("C"):
        try:
            weekmask = weekmask or "Mon Tue Wed Thu Fri"
            freq = prefix_mapping[freq](holidays=holidays, weekmask=weekmask)
        except (KeyError, TypeError) as err:
            msg = f"invalid custom frequency string: {freq}"
            raise ValueError(msg) from err
    elif holidays or weekmask:
        msg = (
            "a custom frequency string is required when holidays or "
            f"weekmask are passed, got frequency {freq}"
        )
        raise ValueError(msg)

    return date_range(
        start=start,
        end=end,
        periods=periods,
        freq=freq,
        tz=tz,
        normalize=normalize,
        name=name,
        inclusive=inclusive,
        **kwargs,
    )


def _time_to_micros(time_obj: dt.time) -> int:
    seconds = time_obj.hour * 60 * 60 + 60 * time_obj.minute + time_obj.second
    return 1_000_000 * seconds + time_obj.microsecond
<|endoftext|>from __future__ import annotations

import textwrap
from typing import (
    TYPE_CHECKING,
    cast,
)

import numpy as np

from pandas._libs import (
    NaT,
    lib,
)
from pandas.errors import InvalidIndexError

from pandas.core.dtypes.cast import find_common_type

from pandas.core.algorithms import safe_sort
from pandas.core.indexes.base import (
    Index,
    _new_Index,
    ensure_index,
    ensure_index_from_sequences,
    get_unanimous_names,
)
from pandas.core.indexes.category import CategoricalIndex
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.interval import IntervalIndex
from pandas.core.indexes.multi import MultiIndex
from pandas.core.indexes.period import PeriodIndex
from pandas.core.indexes.range import RangeIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex

if TYPE_CHECKING:
    from pandas._typing import Axis
_sort_msg = textwrap.dedent(
    """\
Sorting because non-concatenation axis is not aligned. A future version
of pandas will change to not sort by default.

To accept the future behavior, pass 'sort=False'.

To retain the current behavior and silence the warning, pass 'sort=True'.
"""
)


__all__ = [
    "Index",
    "MultiIndex",
    "CategoricalIndex",
    "IntervalIndex",
    "RangeIndex",
    "InvalidIndexError",
    "TimedeltaIndex",
    "PeriodIndex",
    "DatetimeIndex",
    "_new_Index",
    "NaT",
    "ensure_index",
    "ensure_index_from_sequences",
    "get_objs_combined_axis",
    "union_indexes",
    "get_unanimous_names",
    "all_indexes_same",
    "default_index",
    "safe_sort_index",
]


def get_objs_combined_axis(
    objs,
    intersect: bool = False,
    axis: Axis = 0,
    sort: bool = True,
    copy: bool = False,
) -> Index:
    """
    Extract combined index: return intersection or union (depending on the
    value of "intersect") of indexes on given axis, or None if all objects
    lack indexes (e.g. they are numpy arrays).

    Parameters
    ----------
    objs : list
        Series or DataFrame objects, may be mix of the two.
    intersect : bool, default False
        If True, calculate the intersection between indexes. Otherwise,
        calculate the union.
    axis : {0 or 'index', 1 or 'outer'}, default 0
        The axis to extract indexes from.
    sort : bool, default True
        Whether the result index should come out sorted or not.
    copy : bool, default False
        If True, return a copy of the combined index.

    Returns
    -------
    Index
    """
    obs_idxes = [obj._get_axis(axis) for obj in objs]
    return _get_combined_index(obs_idxes, intersect=intersect, sort=sort, copy=copy)


def _get_distinct_objs(objs: list[Index]) -> list[Index]:
    """
    Return a list with distinct elements of "objs" (different ids).
    Preserves order.
    """
    ids: set[int] = set()
    res = []
    for obj in objs:
        if id(obj) not in ids:
            ids.add(id(obj))
            res.append(obj)
    return res


def _get_combined_index(
    indexes: list[Index],
    intersect: bool = False,
    sort: bool = False,
    copy: bool = False,
) -> Index:
    """
    Return the union or intersection of indexes.

    Parameters
    ----------
    indexes : list of Index or list objects
        When intersect=True, do not accept list of lists.
    intersect : bool, default False
        If True, calculate the intersection between indexes. Otherwise,
        calculate the union.
    sort : bool, default False
        Whether the result index should come out sorted or not.
    copy : bool, default False
        If True, return a copy of the combined index.

    Returns
    -------
    Index
    """
    # TODO: handle index names!
    indexes = _get_distinct_objs(indexes)
    if len(indexes) == 0:
        index = Index([])
    elif len(indexes) == 1:
        index = indexes[0]
    elif intersect:
        index = indexes[0]
        for other in indexes[1:]:
            index = index.intersection(other)
    else:
        index = union_indexes(indexes, sort=False)
        index = ensure_index(index)

    if sort:
        index = safe_sort_index(index)
    # GH 29879
    if copy:
        index = index.copy()

    return index


def safe_sort_index(index: Index) -> Index:
    """
    Returns the sorted index

    We keep the dtypes and the name attributes.

    Parameters
    ----------
    index : an Index

    Returns
    -------
    Index
    """
    if index.is_monotonic_increasing:
        return index

    try:
        array_sorted = safe_sort(index)
    except TypeError:
        pass
    else:
        if isinstance(array_sorted, Index):
            return array_sorted

        array_sorted = cast(np.ndarray, array_sorted)
        if isinstance(index, MultiIndex):
            index = MultiIndex.from_tuples(array_sorted, names=index.names)
        else:
            index = Index(array_sorted, name=index.name, dtype=index.dtype)

    return index


def union_indexes(indexes, sort: bool | None = True) -> Index:
    """
    Return the union of indexes.

    The behavior of sort and names is not consistent.

    Parameters
    ----------
    indexes : list of Index or list objects
    sort : bool, default True
        Whether the result index should come out sorted or not.

    Returns
    -------
    Index
    """
    if len(indexes) == 0:
        raise AssertionError("Must have at least 1 Index to union")
    if len(indexes) == 1:
        result = indexes[0]
        if isinstance(result, list):
            if not sort:
                result = Index(result)
            else:
                result = Index(sorted(result))
        return result

    indexes, kind = _sanitize_and_check(indexes)

    def _unique_indices(inds, dtype) -> Index:
        """
        Concatenate indices and remove duplicates.

        Parameters
        ----------
        inds : list of Index or list objects
        dtype : dtype to set for the resulting Index

        Returns
        -------
        Index
        """
        if all(isinstance(ind, Index) for ind in inds):
            inds = [ind.astype(dtype, copy=False) for ind in inds]
            result = inds[0].unique()
            other = inds[1].append(inds[2:])
            diff = other[result.get_indexer_for(other) == -1]
            if len(diff):
                result = result.append(diff.unique())
            if sort:
                result = result.sort_values()
            return result

        def conv(i):
            if isinstance(i, Index):
                i = i.tolist()
            return i

        return Index(
            lib.fast_unique_multiple_list([conv(i) for i in inds], sort=sort),
            dtype=dtype,
        )

    def _find_common_index_dtype(inds):
        """
        Finds a common type for the indexes to pass through to resulting index.

        Parameters
        ----------
        inds: list of Index or list objects

        Returns
        -------
        The common type or None if no indexes were given
        """
        dtypes = [idx.dtype for idx in indexes if isinstance(idx, Index)]
        if dtypes:
            dtype = find_common_type(dtypes)
        else:
            dtype = None

        return dtype

    if kind == "special":
        result = indexes[0]

        dtis = [x for x in indexes if isinstance(x, DatetimeIndex)]
        dti_tzs = [x for x in dtis if x.tz is not None]
        if len(dti_tzs) not in [0, len(dtis)]:
            # TODO: this behavior is not tested (so may not be desired),
            #  but is kept in order to keep behavior the same when
            #  deprecating union_many
            # test_frame_from_dict_with_mixed_indexes
            raise TypeError("Cannot join tz-naive with tz-aware DatetimeIndex")

        if len(dtis) == len(indexes):
            sort = True
            result = indexes[0]

        elif len(dtis) > 1:
            # If we have mixed timezones, our casting behavior may depend on
            #  the order of indexes, which we don't want.
            sort = False

            # TODO: what about Categorical[dt64]?
            # test_frame_from_dict_with_mixed_indexes
            indexes = [x.astype(object, copy=False) for x in indexes]
            result = indexes[0]

        for other in indexes[1:]:
            result = result.union(other, sort=None if sort else False)
        return result

    elif kind == "array":
        dtype = _find_common_index_dtype(indexes)
        index = indexes[0]
        if not all(index.equals(other) for other in indexes[1:]):
            index = _unique_indices(indexes, dtype)

        name = get_unanimous_names(*indexes)[0]
        if name != index.name:
            index = index.rename(name)
        return index
    else:  # kind='list'
        dtype = _find_common_index_dtype(indexes)
        return _unique_indices(indexes, dtype)


def _sanitize_and_check(indexes):
    """
    Verify the type of indexes and convert lists to Index.

    Cases:

    - [list, list, ...]: Return ([list, list, ...], 'list')
    - [list, Index, ...]: Return _sanitize_and_check([Index, Index, ...])
        Lists are sorted and converted to Index.
    - [Index, Index, ...]: Return ([Index, Index, ...], TYPE)
        TYPE = 'special' if at least one special type, 'array' otherwise.

    Parameters
    ----------
    indexes : list of Index or list objects

    Returns
    -------
    sanitized_indexes : list of Index or list objects
    type : {'list', 'array', 'special'}
    """
    kinds = list({type(index) for index in indexes})

    if list in kinds:
        if len(kinds) > 1:
            indexes = [
                Index(list(x)) if not isinstance(x, Index) else x for x in indexes
            ]
            kinds.remove(list)
        else:
            return indexes, "list"

    if len(kinds) > 1 or Index not in kinds:
        return indexes, "special"
    else:
        return indexes, "array"


def all_indexes_same(indexes) -> bool:
    """
    Determine if all indexes contain the same elements.

    Parameters
    ----------
    indexes : iterable of Index objects

    Returns
    -------
    bool
        True if all indexes contain the same elements, False otherwise.
    """
    itr = iter(indexes)
    first = next(itr)
    return all(first.equals(index) for index in itr)


def default_index(n: int) -> RangeIndex:
    rng = range(n)
    return RangeIndex._simple_new(rng, name=None)
<|endoftext|>from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    cast,
)

import numpy as np

from pandas._libs import index as libindex
from pandas.util._decorators import (
    cache_readonly,
    doc,
)

from pandas.core.dtypes.common import is_scalar
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas.core.dtypes.missing import (
    is_valid_na_for_dtype,
    isna,
)

from pandas.core.arrays.categorical import (
    Categorical,
    contains,
)
from pandas.core.construction import extract_array
from pandas.core.indexes.base import (
    Index,
    maybe_extract_name,
)
from pandas.core.indexes.extension import (
    NDArrayBackedExtensionIndex,
    inherit_names,
)

if TYPE_CHECKING:
    from collections.abc import Hashable

    from pandas._typing import (
        Dtype,
        DtypeObj,
        Self,
        npt,
    )


@inherit_names(
    [
        "argsort",
        "tolist",
        "codes",
        "categories",
        "ordered",
        "_reverse_indexer",
        "searchsorted",
        "min",
        "max",
    ],
    Categorical,
)
@inherit_names(
    [
        "rename_categories",
        "reorder_categories",
        "add_categories",
        "remove_categories",
        "remove_unused_categories",
        "set_categories",
        "as_ordered",
        "as_unordered",
    ],
    Categorical,
    wrap=True,
)
class CategoricalIndex(NDArrayBackedExtensionIndex):
    """
    Index based on an underlying :class:`Categorical`.

    CategoricalIndex, like Categorical, can only take on a limited,
    and usually fixed, number of possible values (`categories`). Also,
    like Categorical, it might have an order, but numerical operations
    (additions, divisions, ...) are not possible.

    Parameters
    ----------
    data : array-like (1-dimensional)
        The values of the categorical. If `categories` are given, values not in
        `categories` will be replaced with NaN.
    categories : index-like, optional
        The categories for the categorical. Items need to be unique.
        If the categories are not given here (and also not in `dtype`), they
        will be inferred from the `data`.
    ordered : bool, optional
        Whether or not this categorical is treated as an ordered
        categorical. If not given here or in `dtype`, the resulting
        categorical will be unordered.
    dtype : CategoricalDtype or "category", optional
        If :class:`CategoricalDtype`, cannot be used together with
        `categories` or `ordered`.
    copy : bool, default False
        Make a copy of input ndarray.
    name : object, optional
        Name to be stored in the index.

    Attributes
    ----------
    codes
    categories
    ordered

    Methods
    -------
    rename_categories
    reorder_categories
    add_categories
    remove_categories
    remove_unused_categories
    set_categories
    as_ordered
    as_unordered
    map

    Raises
    ------
    ValueError
        If the categories do not validate.
    TypeError
        If an explicit ``ordered=True`` is given but no `categories` and the
        `values` are not sortable.

    See Also
    --------
    Index : The base pandas Index type.
    Categorical : A categorical array.
    CategoricalDtype : Type for categorical data.

    Notes
    -----
    See the `user guide
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html#categoricalindex>`__
    for more.

    Examples
    --------
    >>> pd.CategoricalIndex(["a", "b", "c", "a", "b", "c"])
    CategoricalIndex(['a', 'b', 'c', 'a', 'b', 'c'],
                     categories=['a', 'b', 'c'], ordered=False, dtype='category')

    ``CategoricalIndex`` can also be instantiated from a ``Categorical``:

    >>> c = pd.Categorical(["a", "b", "c", "a", "b", "c"])
    >>> pd.CategoricalIndex(c)
    CategoricalIndex(['a', 'b', 'c', 'a', 'b', 'c'],
                     categories=['a', 'b', 'c'], ordered=False, dtype='category')

    Ordered ``CategoricalIndex`` can have a min and max value.

    >>> ci = pd.CategoricalIndex(
    ...     ["a", "b", "c", "a", "b", "c"], ordered=True, categories=["c", "b", "a"]
    ... )
    >>> ci
    CategoricalIndex(['a', 'b', 'c', 'a', 'b', 'c'],
                     categories=['c', 'b', 'a'], ordered=True, dtype='category')
    >>> ci.min()
    'c'
    """

    _typ = "categoricalindex"
    _data_cls = Categorical

    @property
    def _can_hold_strings(self):
        return self.categories._can_hold_strings

    @cache_readonly
    def _should_fallback_to_positional(self) -> bool:
        return self.categories._should_fallback_to_positional

    codes: np.ndarray
    categories: Index
    ordered: bool | None
    _data: Categorical
    _values: Categorical

    @property
    def _engine_type(self) -> type[libindex.IndexEngine]:
        # self.codes can have dtype int8, int16, int32 or int64, so we need
        # to return the corresponding engine type (libindex.Int8Engine, etc.).
        return {
            np.int8: libindex.Int8Engine,
            np.int16: libindex.Int16Engine,
            np.int32: libindex.Int32Engine,
            np.int64: libindex.Int64Engine,
        }[self.codes.dtype.type]

    # --------------------------------------------------------------------
    # Constructors

    def __new__(
        cls,
        data=None,
        categories=None,
        ordered=None,
        dtype: Dtype | None = None,
        copy: bool = False,
        name: Hashable | None = None,
    ) -> Self:
        name = maybe_extract_name(name, data, cls)

        if is_scalar(data):
            # GH#38944 include None here, which pre-2.0 subbed in []
            cls._raise_scalar_data_error(data)

        data = Categorical(
            data, categories=categories, ordered=ordered, dtype=dtype, copy=copy
        )

        return cls._simple_new(data, name=name)

    # --------------------------------------------------------------------

    def _is_dtype_compat(self, other: Index) -> Categorical:
        """
        *this is an internal non-public method*

        provide a comparison between the dtype of self and other (coercing if
        needed)

        Parameters
        ----------
        other : Index

        Returns
        -------
        Categorical

        Raises
        ------
        TypeError if the dtypes are not compatible
        """
        if isinstance(other.dtype, CategoricalDtype):
            cat = extract_array(other)
            cat = cast(Categorical, cat)
            if not cat._categories_match_up_to_permutation(self._values):
                raise TypeError(
                    "categories must match existing categories when appending"
                )

        elif other._is_multi:
            # preempt raising NotImplementedError in isna call
            raise TypeError("MultiIndex is not dtype-compatible with CategoricalIndex")
        else:
            values = other

            cat = Categorical(other, dtype=self.dtype)
            other = CategoricalIndex(cat)
            if not other.isin(values).all():
                raise TypeError(
                    "cannot append a non-category item to a CategoricalIndex"
                )
            cat = other._values

            if not ((cat == values) | (isna(cat) & isna(values))).all():
                # GH#37667 see test_equals_non_category
                raise TypeError(
                    "categories must match existing categories when appending"
                )

        return cat

    def equals(self, other: object) -> bool:
        """
        Determine if two CategoricalIndex objects contain the same elements.

        Returns
        -------
        bool
            ``True`` if two :class:`pandas.CategoricalIndex` objects have equal
            elements, ``False`` otherwise.

        Examples
        --------
        >>> ci = pd.CategoricalIndex(['a', 'b', 'c', 'a', 'b', 'c'])
        >>> ci2 = pd.CategoricalIndex(pd.Categorical(['a', 'b', 'c', 'a', 'b', 'c']))
        >>> ci.equals(ci2)
        True

        The order of elements matters.

        >>> ci3 = pd.CategoricalIndex(['c', 'b', 'a', 'a', 'b', 'c'])
        >>> ci.equals(ci3)
        False

        The orderedness also matters.

        >>> ci4 = ci.as_ordered()
        >>> ci.equals(ci4)
        False

        The categories matter, but the order of the categories matters only when
        ``ordered=True``.

        >>> ci5 = ci.set_categories(['a', 'b', 'c', 'd'])
        >>> ci.equals(ci5)
        False

        >>> ci6 = ci.set_categories(['b', 'c', 'a'])
        >>> ci.equals(ci6)
        True
        >>> ci_ordered = pd.CategoricalIndex(['a', 'b', 'c', 'a', 'b', 'c'],
        ...                                  ordered=True)
        >>> ci2_ordered = ci_ordered.set_categories(['b', 'c', 'a'])
        >>> ci_ordered.equals(ci2_ordered)
        False
        """
        if self.is_(other):
            return True

        if not isinstance(other, Index):
            return False

        try:
            other = self._is_dtype_compat(other)
        except (TypeError, ValueError):
            return False

        return self._data.equals(other)

    # --------------------------------------------------------------------
    # Rendering Methods

    @property
    def _formatter_func(self):
        return self.categories._formatter_func

    def _format_attrs(self):
        """
        Return a list of tuples of the (attr,formatted_value)
        """
        attrs: list[tuple[str, str | int | bool | None]]

        attrs = [
            (
                "categories",
                f"[{', '.join(self._data._repr_categories())}]",
            ),
            ("ordered", self.ordered),
        ]
        extra = super()._format_attrs()
        return attrs + extra

    # --------------------------------------------------------------------

    @property
    def inferred_type(self) -> str:
        return "categorical"

    @doc(Index.__contains__)
    def __contains__(self, key: Any) -> bool:
        # if key is a NaN, check if any NaN is in self.
        if is_valid_na_for_dtype(key, self.categories.dtype):
            return self.hasnans

        return contains(self, key, container=self._engine)

    def reindex(
        self, target, method=None, level=None, limit: int | None = None, tolerance=None
    ) -> tuple[Index, npt.NDArray[np.intp] | None]:
        """
        Create index with target's values (move/add/delete values as necessary)

        Returns
        -------
        new_index : pd.Index
            Resulting index
        indexer : np.ndarray[np.intp] or None
            Indices of output values in original index

        """
        if method is not None:
            raise NotImplementedError(
                "argument method is not implemented for CategoricalIndex.reindex"
            )
        if level is not None:
            raise NotImplementedError(
                "argument level is not implemented for CategoricalIndex.reindex"
            )
        if limit is not None:
            raise NotImplementedError(
                "argument limit is not implemented for CategoricalIndex.reindex"
            )
        return super().reindex(target)

    # --------------------------------------------------------------------
    # Indexing Methods

    def _maybe_cast_indexer(self, key) -> int:
        # GH#41933: we have to do this instead of self._data._validate_scalar
        #  because this will correctly get partial-indexing on Interval categories
        try:
            return self._data._unbox_scalar(key)
        except KeyError:
            if is_valid_na_for_dtype(key, self.categories.dtype):
                return -1
            raise

    def _maybe_cast_listlike_indexer(self, values) -> CategoricalIndex:
        if isinstance(values, CategoricalIndex):
            values = values._data
        if isinstance(values, Categorical):
            # Indexing on codes is more efficient if categories are the same,
            #  so we can apply some optimizations based on the degree of
            #  dtype-matching.
            cat = self._data._encode_with_my_categories(values)
            codes = cat._codes
        else:
            codes = self.categories.get_indexer(values)
            codes = codes.astype(self.codes.dtype, copy=False)
            cat = self._data._from_backing_data(codes)
        return type(self)._simple_new(cat)

    # --------------------------------------------------------------------

    def _is_comparable_dtype(self, dtype: DtypeObj) -> bool:
        return self.categories._is_comparable_dtype(dtype)

    def map(self, mapper, na_action: Literal["ignore"] | None = None):
        """
        Map values using input an input mapping or function.

        Maps the values (their categories, not the codes) of the index to new
        categories. If the mapping correspondence is one-to-one the result is a
        :class:`~pandas.CategoricalIndex` which has the same order property as
        the original, otherwise an :class:`~pandas.Index` is returned.

        If a `dict` or :class:`~pandas.Series` is used any unmapped category is
        mapped to `NaN`. Note that if this happens an :class:`~pandas.Index`
        will be returned.

        Parameters
        ----------
        mapper : function, dict, or Series
            Mapping correspondence.

        Returns
        -------
        pandas.CategoricalIndex or pandas.Index
            Mapped index.

        See Also
        --------
        Index.map : Apply a mapping correspondence on an
            :class:`~pandas.Index`.
        Series.map : Apply a mapping correspondence on a
            :class:`~pandas.Series`.
        Series.apply : Apply more complex functions on a
            :class:`~pandas.Series`.

        Examples
        --------
        >>> idx = pd.CategoricalIndex(['a', 'b', 'c'])
        >>> idx
        CategoricalIndex(['a', 'b', 'c'], categories=['a', 'b', 'c'],
                          ordered=False, dtype='category')
        >>> idx.map(lambda x: x.upper())
        CategoricalIndex(['A', 'B', 'C'], categories=['A', 'B', 'C'],
                         ordered=False, dtype='category')
        >>> idx.map({'a': 'first', 'b': 'second', 'c': 'third'})
        CategoricalIndex(['first', 'second', 'third'], categories=['first',
                         'second', 'third'], ordered=False, dtype='category')

        If the mapping is one-to-one the ordering of the categories is
        preserved:

        >>> idx = pd.CategoricalIndex(['a', 'b', 'c'], ordered=True)
        >>> idx
        CategoricalIndex(['a', 'b', 'c'], categories=['a', 'b', 'c'],
                         ordered=True, dtype='category')
        >>> idx.map({'a': 3, 'b': 2, 'c': 1})
        CategoricalIndex([3, 2, 1], categories=[3, 2, 1], ordered=True,
                         dtype='category')

        If the mapping is not one-to-one an :class:`~pandas.Index` is returned:

        >>> idx.map({'a': 'first', 'b': 'second', 'c': 'first'})
        Index(['first', 'second', 'first'], dtype='object')

        If a `dict` is used, all unmapped categories are mapped to `NaN` and
        the result is an :class:`~pandas.Index`:

        >>> idx.map({'a': 'first', 'b': 'second'})
        Index(['first', 'second', nan], dtype='object')
        """
        mapped = self._values.map(mapper, na_action=na_action)
        return Index(mapped, name=self.name)

    def _concat(self, to_concat: list[Index], name: Hashable) -> Index:
        # if calling index is category, don't check dtype of others
        try:
            cat = Categorical._concat_same_type(
                [self._is_dtype_compat(c) for c in to_concat]
            )
        except TypeError:
            # not all to_concat elements are among our categories (or NA)

            res = concat_compat([x._values for x in to_concat])
            return Index(res, name=name)
        else:
            return type(self)._simple_new(cat, name=name)
<|endoftext|>"""
frozen (immutable) data structures to support MultiIndexing

These are used for:

- .names (FrozenList)

"""
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    NoReturn,
)

from pandas.core.base import PandasObject

from pandas.io.formats.printing import pprint_thing

if TYPE_CHECKING:
    from pandas._typing import Self


class FrozenList(PandasObject, list):
    """
    Container that doesn't allow setting item *but*
    because it's technically hashable, will be used
    for lookups, appropriately, etc.
    """

    # Side note: This has to be of type list. Otherwise,
    #            it messes up PyTables type checks.

    def union(self, other) -> FrozenList:
        """
        Returns a FrozenList with other concatenated to the end of self.

        Parameters
        ----------
        other : array-like
            The array-like whose elements we are concatenating.

        Returns
        -------
        FrozenList
            The collection difference between self and other.
        """
        if isinstance(other, tuple):
            other = list(other)
        return type(self)(super().__add__(other))

    def difference(self, other) -> FrozenList:
        """
        Returns a FrozenList with elements from other removed from self.

        Parameters
        ----------
        other : array-like
            The array-like whose elements we are removing self.

        Returns
        -------
        FrozenList
            The collection difference between self and other.
        """
        other = set(other)
        temp = [x for x in self if x not in other]
        return type(self)(temp)

    # TODO: Consider deprecating these in favor of `union` (xref gh-15506)
    # error: Incompatible types in assignment (expression has type
    # "Callable[[FrozenList, Any], FrozenList]", base class "list" defined the
    # type as overloaded function)
    __add__ = __iadd__ = union  # type: ignore[assignment]

    def __getitem__(self, n):
        if isinstance(n, slice):
            return type(self)(super().__getitem__(n))
        return super().__getitem__(n)

    def __radd__(self, other) -> Self:
        if isinstance(other, tuple):
            other = list(other)
        return type(self)(other + list(self))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (tuple, FrozenList)):
            other = list(other)
        return super().__eq__(other)

    __req__ = __eq__

    def __mul__(self, other) -> Self:
        return type(self)(super().__mul__(other))

    __imul__ = __mul__

    def __reduce__(self):
        return type(self), (list(self),)

    # error: Signature of "__hash__" incompatible with supertype "list"
    def __hash__(self) -> int:  # type: ignore[override]
        return hash(tuple(self))

    def _disabled(self, *args, **kwargs) -> NoReturn:
        """
        This method will not function because object is immutable.
        """
        raise TypeError(f"'{type(self).__name__}' does not support mutable operations.")

    def __str__(self) -> str:
        return pprint_thing(self, quote_strings=True, escape_chars=("\t", "\r", "\n"))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({str(self)})"

    __setitem__ = __setslice__ = _disabled  # type: ignore[assignment]
    __delitem__ = __delslice__ = _disabled
    pop = append = extend = _disabled
    remove = sort = insert = _disabled  # type: ignore[assignment]
<|endoftext|>""" implement the TimedeltaIndex """
from __future__ import annotations

from typing import TYPE_CHECKING
import warnings

from pandas._libs import (
    index as libindex,
    lib,
)
from pandas._libs.tslibs import (
    Resolution,
    Timedelta,
    to_offset,
)
from pandas._libs.tslibs.timedeltas import disallow_ambiguous_unit
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import (
    is_scalar,
    pandas_dtype,
)
from pandas.core.dtypes.generic import ABCSeries

from pandas.core.arrays.timedeltas import TimedeltaArray
import pandas.core.common as com
from pandas.core.indexes.base import (
    Index,
    maybe_extract_name,
)
from pandas.core.indexes.datetimelike import DatetimeTimedeltaMixin
from pandas.core.indexes.extension import inherit_names

if TYPE_CHECKING:
    from pandas._typing import DtypeObj


@inherit_names(
    ["__neg__", "__pos__", "__abs__", "total_seconds", "round", "floor", "ceil"]
    + TimedeltaArray._field_ops,
    TimedeltaArray,
    wrap=True,
)
@inherit_names(
    [
        "components",
        "to_pytimedelta",
        "sum",
        "std",
        "median",
    ],
    TimedeltaArray,
)
class TimedeltaIndex(DatetimeTimedeltaMixin):
    """
    Immutable Index of timedelta64 data.

    Represented internally as int64, and scalars returned Timedelta objects.

    Parameters
    ----------
    data : array-like (1-dimensional), optional
        Optional timedelta-like data to construct index with.
    unit : {'D', 'h', 'm', 's', 'ms', 'us', 'ns'}, optional
        The unit of ``data``.

        .. deprecated:: 2.2.0
         Use ``pd.to_timedelta`` instead.

    freq : str or pandas offset object, optional
        One of pandas date offset strings or corresponding objects. The string
        ``'infer'`` can be passed in order to set the frequency of the index as
        the inferred frequency upon creation.
    dtype : numpy.dtype or str, default None
        Valid ``numpy`` dtypes are ``timedelta64[ns]``, ``timedelta64[us]``,
        ``timedelta64[ms]``, and ``timedelta64[s]``.
    copy : bool
        Make a copy of input array.
    name : object
        Name to be stored in the index.

    Attributes
    ----------
    days
    seconds
    microseconds
    nanoseconds
    components
    inferred_freq

    Methods
    -------
    to_pytimedelta
    to_series
    round
    floor
    ceil
    to_frame
    mean

    See Also
    --------
    Index : The base pandas Index type.
    Timedelta : Represents a duration between two dates or times.
    DatetimeIndex : Index of datetime64 data.
    PeriodIndex : Index of Period data.
    timedelta_range : Create a fixed-frequency TimedeltaIndex.

    Notes
    -----
    To learn more about the frequency strings, please see `this link
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__.

    Examples
    --------
    >>> pd.TimedeltaIndex(['0 days', '1 days', '2 days', '3 days', '4 days'])
    TimedeltaIndex(['0 days', '1 days', '2 days', '3 days', '4 days'],
                   dtype='timedelta64[ns]', freq=None)

    We can also let pandas infer the frequency when possible.

    >>> pd.TimedeltaIndex(np.arange(5) * 24 * 3600 * 1e9, freq='infer')
    TimedeltaIndex(['0 days', '1 days', '2 days', '3 days', '4 days'],
                   dtype='timedelta64[ns]', freq='D')
    """

    _typ = "timedeltaindex"

    _data_cls = TimedeltaArray

    @property
    def _engine_type(self) -> type[libindex.TimedeltaEngine]:
        return libindex.TimedeltaEngine

    _data: TimedeltaArray

    # Use base class method instead of DatetimeTimedeltaMixin._get_string_slice
    _get_string_slice = Index._get_string_slice

    # error: Signature of "_resolution_obj" incompatible with supertype
    # "DatetimeIndexOpsMixin"
    @property
    def _resolution_obj(self) -> Resolution | None:  # type: ignore[override]
        return self._data._resolution_obj

    # -------------------------------------------------------------------
    # Constructors

    def __new__(
        cls,
        data=None,
        unit=lib.no_default,
        freq=lib.no_default,
        closed=lib.no_default,
        dtype=None,
        copy: bool = False,
        name=None,
    ):
        if closed is not lib.no_default:
            # GH#52628
            warnings.warn(
                f"The 'closed' keyword in {cls.__name__} construction is "
                "deprecated and will be removed in a future version.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )

        if unit is not lib.no_default:
            # GH#55499
            warnings.warn(
                f"The 'unit' keyword in {cls.__name__} construction is "
                "deprecated and will be removed in a future version. "
                "Use pd.to_timedelta instead.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )
        else:
            unit = None

        name = maybe_extract_name(name, data, cls)

        if is_scalar(data):
            cls._raise_scalar_data_error(data)

        disallow_ambiguous_unit(unit)
        if dtype is not None:
            dtype = pandas_dtype(dtype)

        if (
            isinstance(data, TimedeltaArray)
            and freq is lib.no_default
            and (dtype is None or dtype == data.dtype)
        ):
            if copy:
                data = data.copy()
            return cls._simple_new(data, name=name)

        if (
            isinstance(data, TimedeltaIndex)
            and freq is lib.no_default
            and name is None
            and (dtype is None or dtype == data.dtype)
        ):
            if copy:
                return data.copy()
            else:
                return data._view()

        # - Cases checked above all return/raise before reaching here - #

        tdarr = TimedeltaArray._from_sequence_not_strict(
            data, freq=freq, unit=unit, dtype=dtype, copy=copy
        )
        refs = None
        if not copy and isinstance(data, (ABCSeries, Index)):
            refs = data._references

        return cls._simple_new(tdarr, name=name, refs=refs)

    # -------------------------------------------------------------------

    def _is_comparable_dtype(self, dtype: DtypeObj) -> bool:
        """
        Can we compare values of the given dtype to our own?
        """
        return lib.is_np_dtype(dtype, "m")  # aka self._data._is_recognized_dtype

    # -------------------------------------------------------------------
    # Indexing Methods

    def get_loc(self, key):
        """
        Get integer location for requested label

        Returns
        -------
        loc : int, slice, or ndarray[int]
        """
        self._check_indexing_error(key)

        try:
            key = self._data._validate_scalar(key, unbox=False)
        except TypeError as err:
            raise KeyError(key) from err

        return Index.get_loc(self, key)

    def _parse_with_reso(self, label: str):
        # the "with_reso" is a no-op for TimedeltaIndex
        parsed = Timedelta(label)
        return parsed, None

    def _parsed_string_to_bounds(self, reso, parsed: Timedelta):
        # reso is unused, included to match signature of DTI/PI
        lbound = parsed.round(parsed.resolution_string)
        rbound = lbound + to_offset(parsed.resolution_string) - Timedelta(1, "ns")
        return lbound, rbound

    # -------------------------------------------------------------------

    @property
    def inferred_type(self) -> str:
        return "timedelta64"


def timedelta_range(
    start=None,
    end=None,
    periods: int | None = None,
    freq=None,
    name=None,
    closed=None,
    *,
    unit: str | None = None,
) -> TimedeltaIndex:
    """
    Return a fixed frequency TimedeltaIndex with day as the default.

    Parameters
    ----------
    start : str or timedelta-like, default None
        Left bound for generating timedeltas.
    end : str or timedelta-like, default None
        Right bound for generating timedeltas.
    periods : int, default None
        Number of periods to generate.
    freq : str, Timedelta, datetime.timedelta, or DateOffset, default 'D'
        Frequency strings can have multiples, e.g. '5h'.
    name : str, default None
        Name of the resulting TimedeltaIndex.
    closed : str, default None
        Make the interval closed with respect to the given frequency to
        the 'left', 'right', or both sides (None).
    unit : str, default None
        Specify the desired resolution of the result.

        .. versionadded:: 2.0.0

    Returns
    -------
    TimedeltaIndex

    Notes
    -----
    Of the four parameters ``start``, ``end``, ``periods``, and ``freq``,
    exactly three must be specified. If ``freq`` is omitted, the resulting
    ``TimedeltaIndex`` will have ``periods`` linearly spaced elements between
    ``start`` and ``end`` (closed on both sides).

    To learn more about the frequency strings, please see `this link
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__.

    Examples
    --------
    >>> pd.timedelta_range(start='1 day', periods=4)
    TimedeltaIndex(['1 days', '2 days', '3 days', '4 days'],
                   dtype='timedelta64[ns]', freq='D')

    The ``closed`` parameter specifies which endpoint is included.  The default
    behavior is to include both endpoints.

    >>> pd.timedelta_range(start='1 day', periods=4, closed='right')
    TimedeltaIndex(['2 days', '3 days', '4 days'],
                   dtype='timedelta64[ns]', freq='D')

    The ``freq`` parameter specifies the frequency of the TimedeltaIndex.
    Only fixed frequencies can be passed, non-fixed frequencies such as
    'M' (month end) will raise.

    >>> pd.timedelta_range(start='1 day', end='2 days', freq='6h')
    TimedeltaIndex(['1 days 00:00:00', '1 days 06:00:00', '1 days 12:00:00',
                    '1 days 18:00:00', '2 days 00:00:00'],
                   dtype='timedelta64[ns]', freq='6h')

    Specify ``start``, ``end``, and ``periods``; the frequency is generated
    automatically (linearly spaced).

    >>> pd.timedelta_range(start='1 day', end='5 days', periods=4)
    TimedeltaIndex(['1 days 00:00:00', '2 days 08:00:00', '3 days 16:00:00',
                    '5 days 00:00:00'],
                   dtype='timedelta64[ns]', freq=None)

    **Specify a unit**

    >>> pd.timedelta_range("1 Day", periods=3, freq="100000D", unit="s")
    TimedeltaIndex(['1 days', '100001 days', '200001 days'],
                   dtype='timedelta64[s]', freq='100000D')
    """
    if freq is None and com.any_none(periods, start, end):
        freq = "D"

    freq = to_offset(freq)
    tdarr = TimedeltaArray._generate_range(
        start, end, periods, freq, closed=closed, unit=unit
    )
    return TimedeltaIndex._simple_new(tdarr, name=name)
<|endoftext|>from __future__ import annotations

import operator
from typing import (
    TYPE_CHECKING,
    Literal,
    NoReturn,
    cast,
)

import numpy as np

from pandas._libs import lib
from pandas._libs.missing import is_matching_na
from pandas._libs.sparse import SparseIndex
import pandas._libs.testing as _testing
from pandas._libs.tslibs.np_datetime import compare_mismatched_resolutions

from pandas.core.dtypes.common import (
    is_bool,
    is_float_dtype,
    is_integer_dtype,
    is_number,
    is_numeric_dtype,
    needs_i8_conversion,
)
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    DatetimeTZDtype,
    ExtensionDtype,
    NumpyEADtype,
)
from pandas.core.dtypes.missing import array_equivalent

import pandas as pd
from pandas import (
    Categorical,
    DataFrame,
    DatetimeIndex,
    Index,
    IntervalDtype,
    IntervalIndex,
    MultiIndex,
    PeriodIndex,
    RangeIndex,
    Series,
    TimedeltaIndex,
)
from pandas.core.arrays import (
    DatetimeArray,
    ExtensionArray,
    IntervalArray,
    PeriodArray,
    TimedeltaArray,
)
from pandas.core.arrays.datetimelike import DatetimeLikeArrayMixin
from pandas.core.arrays.string_ import StringDtype
from pandas.core.indexes.api import safe_sort_index

from pandas.io.formats.printing import pprint_thing

if TYPE_CHECKING:
    from pandas._typing import DtypeObj


def assert_almost_equal(
    left,
    right,
    check_dtype: bool | Literal["equiv"] = "equiv",
    rtol: float = 1.0e-5,
    atol: float = 1.0e-8,
    **kwargs,
) -> None:
    """
    Check that the left and right objects are approximately equal.

    By approximately equal, we refer to objects that are numbers or that
    contain numbers which may be equivalent to specific levels of precision.

    Parameters
    ----------
    left : object
    right : object
    check_dtype : bool or {'equiv'}, default 'equiv'
        Check dtype if both a and b are the same type. If 'equiv' is passed in,
        then `RangeIndex` and `Index` with int64 dtype are also considered
        equivalent when doing type checking.
    rtol : float, default 1e-5
        Relative tolerance.
    atol : float, default 1e-8
        Absolute tolerance.
    """
    if isinstance(left, Index):
        assert_index_equal(
            left,
            right,
            check_exact=False,
            exact=check_dtype,
            rtol=rtol,
            atol=atol,
            **kwargs,
        )

    elif isinstance(left, Series):
        assert_series_equal(
            left,
            right,
            check_exact=False,
            check_dtype=check_dtype,
            rtol=rtol,
            atol=atol,
            **kwargs,
        )

    elif isinstance(left, DataFrame):
        assert_frame_equal(
            left,
            right,
            check_exact=False,
            check_dtype=check_dtype,
            rtol=rtol,
            atol=atol,
            **kwargs,
        )

    else:
        # Other sequences.
        if check_dtype:
            if is_number(left) and is_number(right):
                # Do not compare numeric classes, like np.float64 and float.
                pass
            elif is_bool(left) and is_bool(right):
                # Do not compare bool classes, like np.bool_ and bool.
                pass
            else:
                if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
                    obj = "numpy array"
                else:
                    obj = "Input"
                assert_class_equal(left, right, obj=obj)

        # if we have "equiv", this becomes True
        _testing.assert_almost_equal(
            left, right, check_dtype=bool(check_dtype), rtol=rtol, atol=atol, **kwargs
        )


def _check_isinstance(left, right, cls) -> None:
    """
    Helper method for our assert_* methods that ensures that
    the two objects being compared have the right type before
    proceeding with the comparison.

    Parameters
    ----------
    left : The first object being compared.
    right : The second object being compared.
    cls : The class type to check against.

    Raises
    ------
    AssertionError : Either `left` or `right` is not an instance of `cls`.
    """
    cls_name = cls.__name__

    if not isinstance(left, cls):
        raise AssertionError(
            f"{cls_name} Expected type {cls}, found {type(left)} instead"
        )
    if not isinstance(right, cls):
        raise AssertionError(
            f"{cls_name} Expected type {cls}, found {type(right)} instead"
        )


def assert_dict_equal(left, right, compare_keys: bool = True) -> None:
    _check_isinstance(left, right, dict)
    _testing.assert_dict_equal(left, right, compare_keys=compare_keys)


def assert_index_equal(
    left: Index,
    right: Index,
    exact: bool | str = "equiv",
    check_names: bool = True,
    check_exact: bool = True,
    check_categorical: bool = True,
    check_order: bool = True,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-8,
    obj: str = "Index",
) -> None:
    """
    Check that left and right Index are equal.

    Parameters
    ----------
    left : Index
    right : Index
    exact : bool or {'equiv'}, default 'equiv'
        Whether to check the Index class, dtype and inferred_type
        are identical. If 'equiv', then RangeIndex can be substituted for
        Index with an int64 dtype as well.
    check_names : bool, default True
        Whether to check the names attribute.
    check_exact : bool, default True
        Whether to compare number exactly.
    check_categorical : bool, default True
        Whether to compare internal Categorical exactly.
    check_order : bool, default True
        Whether to compare the order of index entries as well as their values.
        If True, both indexes must contain the same elements, in the same order.
        If False, both indexes must contain the same elements, but in any order.
    rtol : float, default 1e-5
        Relative tolerance. Only used when check_exact is False.
    atol : float, default 1e-8
        Absolute tolerance. Only used when check_exact is False.
    obj : str, default 'Index'
        Specify object name being compared, internally used to show appropriate
        assertion message.

    Examples
    --------
    >>> from pandas import testing as tm
    >>> a = pd.Index([1, 2, 3])
    >>> b = pd.Index([1, 2, 3])
    >>> tm.assert_index_equal(a, b)
    """
    __tracebackhide__ = True

    def _check_types(left, right, obj: str = "Index") -> None:
        if not exact:
            return

        assert_class_equal(left, right, exact=exact, obj=obj)
        assert_attr_equal("inferred_type", left, right, obj=obj)

        # Skip exact dtype checking when `check_categorical` is False
        if isinstance(left.dtype, CategoricalDtype) and isinstance(
            right.dtype, CategoricalDtype
        ):
            if check_categorical:
                assert_attr_equal("dtype", left, right, obj=obj)
                assert_index_equal(left.categories, right.categories, exact=exact)
            return

        assert_attr_equal("dtype", left, right, obj=obj)

    # instance validation
    _check_isinstance(left, right, Index)

    # class / dtype comparison
    _check_types(left, right, obj=obj)

    # level comparison
    if left.nlevels != right.nlevels:
        msg1 = f"{obj} levels are different"
        msg2 = f"{left.nlevels}, {left}"
        msg3 = f"{right.nlevels}, {right}"
        raise_assert_detail(obj, msg1, msg2, msg3)

    # length comparison
    if len(left) != len(right):
        msg1 = f"{obj} length are different"
        msg2 = f"{len(left)}, {left}"
        msg3 = f"{len(right)}, {right}"
        raise_assert_detail(obj, msg1, msg2, msg3)

    # If order doesn't matter then sort the index entries
    if not check_order:
        left = safe_sort_index(left)
        right = safe_sort_index(right)

    # MultiIndex special comparison for little-friendly error messages
    if isinstance(left, MultiIndex):
        right = cast(MultiIndex, right)

        for level in range(left.nlevels):
            lobj = f"MultiIndex level [{level}]"
            try:
                # try comparison on levels/codes to avoid densifying MultiIndex
                assert_index_equal(
                    left.levels[level],
                    right.levels[level],
                    exact=exact,
                    check_names=check_names,
                    check_exact=check_exact,
                    check_categorical=check_categorical,
                    rtol=rtol,
                    atol=atol,
                    obj=lobj,
                )
                assert_numpy_array_equal(left.codes[level], right.codes[level])
            except AssertionError:
                llevel = left.get_level_values(level)
                rlevel = right.get_level_values(level)

                assert_index_equal(
                    llevel,
                    rlevel,
                    exact=exact,
                    check_names=check_names,
                    check_exact=check_exact,
                    check_categorical=check_categorical,
                    rtol=rtol,
                    atol=atol,
                    obj=lobj,
                )
            # get_level_values may change dtype
            _check_types(left.levels[level], right.levels[level], obj=obj)

    # skip exact index checking when `check_categorical` is False
    elif check_exact and check_categorical:
        if not left.equals(right):
            mismatch = left._values != right._values

            if not isinstance(mismatch, np.ndarray):
                mismatch = cast("ExtensionArray", mismatch).fillna(True)

            diff = np.sum(mismatch.astype(int)) * 100.0 / len(left)
            msg = f"{obj} values are different ({np.round(diff, 5)} %)"
            raise_assert_detail(obj, msg, left, right)
    else:
        # if we have "equiv", this becomes True
        exact_bool = bool(exact)
        _testing.assert_almost_equal(
            left.values,
            right.values,
            rtol=rtol,
            atol=atol,
            check_dtype=exact_bool,
            obj=obj,
            lobj=left,
            robj=right,
        )

    # metadata comparison
    if check_names:
        assert_attr_equal("names", left, right, obj=obj)
    if isinstance(left, PeriodIndex) or isinstance(right, PeriodIndex):
        assert_attr_equal("dtype", left, right, obj=obj)
    if isinstance(left, IntervalIndex) or isinstance(right, IntervalIndex):
        assert_interval_array_equal(left._values, right._values)

    if check_categorical:
        if isinstance(left.dtype, CategoricalDtype) or isinstance(
            right.dtype, CategoricalDtype
        ):
            assert_categorical_equal(left._values, right._values, obj=f"{obj} category")


def assert_class_equal(
    left, right, exact: bool | str = True, obj: str = "Input"
) -> None:
    """
    Checks classes are equal.
    """
    __tracebackhide__ = True

    def repr_class(x):
        if isinstance(x, Index):
            # return Index as it is to include values in the error message
            return x

        return type(x).__name__

    def is_class_equiv(idx: Index) -> bool:
        """Classes that are a RangeIndex (sub-)instance or exactly an `Index` .

        This only checks class equivalence. There is a separate check that the
        dtype is int64.
        """
        return type(idx) is Index or isinstance(idx, RangeIndex)

    if type(left) == type(right):
        return

    if exact == "equiv":
        if is_class_equiv(left) and is_class_equiv(right):
            return

    msg = f"{obj} classes are different"
    raise_assert_detail(obj, msg, repr_class(left), repr_class(right))


def assert_attr_equal(attr: str, left, right, obj: str = "Attributes") -> None:
    """
    Check attributes are equal. Both objects must have attribute.

    Parameters
    ----------
    attr : str
        Attribute name being compared.
    left : object
    right : object
    obj : str, default 'Attributes'
        Specify object name being compared, internally used to show appropriate
        assertion message
    """
    __tracebackhide__ = True

    left_attr = getattr(left, attr)
    right_attr = getattr(right, attr)

    if left_attr is right_attr or is_matching_na(left_attr, right_attr):
        # e.g. both np.nan, both NaT, both pd.NA, ...
        return None

    try:
        result = left_attr == right_attr
    except TypeError:
        # datetimetz on rhs may raise TypeError
        result = False
    if (left_attr is pd.NA) ^ (right_attr is pd.NA):
        result = False
    elif not isinstance(result, bool):
        result = result.all()

    if not result:
        msg = f'Attribute "{attr}" are different'
        raise_assert_detail(obj, msg, left_attr, right_attr)
    return None


def assert_is_valid_plot_return_object(objs) -> None:
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes

    if isinstance(objs, (Series, np.ndarray)):
        if isinstance(objs, Series):
            objs = objs._values
        for el in objs.ravel():
            msg = (
                "one of 'objs' is not a matplotlib Axes instance, "
                f"type encountered {repr(type(el).__name__)}"
            )
            assert isinstance(el, (Axes, dict)), msg
    else:
        msg = (
            "objs is neither an ndarray of Artist instances nor a single "
            "ArtistArtist instance, tuple, or dict, 'objs' is a "
            f"{repr(type(objs).__name__)}"
        )
        assert isinstance(objs, (Artist, tuple, dict)), msg


def assert_is_sorted(seq) -> None:
    """Assert that the sequence is sorted."""
    if isinstance(seq, (Index, Series)):
        seq = seq.values
    # sorting does not change precisions
    if isinstance(seq, np.ndarray):
        assert_numpy_array_equal(seq, np.sort(np.array(seq)))
    else:
        assert_extension_array_equal(seq, seq[seq.argsort()])


def assert_categorical_equal(
    left,
    right,
    check_dtype: bool = True,
    check_category_order: bool = True,
    obj: str = "Categorical",
) -> None:
    """
    Test that Categoricals are equivalent.

    Parameters
    ----------
    left : Categorical
    right : Categorical
    check_dtype : bool, default True
        Check that integer dtype of the codes are the same.
    check_category_order : bool, default True
        Whether the order of the categories should be compared, which
        implies identical integer codes.  If False, only the resulting
        values are compared.  The ordered attribute is
        checked regardless.
    obj : str, default 'Categorical'
        Specify object name being compared, internally used to show appropriate
        assertion message.
    """
    _check_isinstance(left, right, Categorical)

    exact: bool | str
    if isinstance(left.categories, RangeIndex) or isinstance(
        right.categories, RangeIndex
    ):
        exact = "equiv"
    else:
        # We still want to require exact matches for Index
        exact = True

    if check_category_order:
        assert_index_equal(
            left.categories, right.categories, obj=f"{obj}.categories", exact=exact
        )
        assert_numpy_array_equal(
            left.codes, right.codes, check_dtype=check_dtype, obj=f"{obj}.codes"
        )
    else:
        try:
            lc = left.categories.sort_values()
            rc = right.categories.sort_values()
        except TypeError:
            # e.g. '<' not supported between instances of 'int' and 'str'
            lc, rc = left.categories, right.categories
        assert_index_equal(lc, rc, obj=f"{obj}.categories", exact=exact)
        assert_index_equal(
            left.categories.take(left.codes),
            right.categories.take(right.codes),
            obj=f"{obj}.values",
            exact=exact,
        )

    assert_attr_equal("ordered", left, right, obj=obj)


def assert_interval_array_equal(
    left, right, exact: bool | Literal["equiv"] = "equiv", obj: str = "IntervalArray"
) -> None:
    """
    Test that two IntervalArrays are equivalent.

    Parameters
    ----------
    left, right : IntervalArray
        The IntervalArrays to compare.
    exact : bool or {'equiv'}, default 'equiv'
        Whether to check the Index class, dtype and inferred_type
        are identical. If 'equiv', then RangeIndex can be substituted for
        Index with an int64 dtype as well.
    obj : str, default 'IntervalArray'
        Specify object name being compared, internally used to show appropriate
        assertion message
    """
    _check_isinstance(left, right, IntervalArray)

    kwargs = {}
    if left._left.dtype.kind in "mM":
        # We have a DatetimeArray or TimedeltaArray
        kwargs["check_freq"] = False

    assert_equal(left._left, right._left, obj=f"{obj}.left", **kwargs)
    assert_equal(left._right, right._right, obj=f"{obj}.left", **kwargs)

    assert_attr_equal("closed", left, right, obj=obj)


def assert_period_array_equal(left, right, obj: str = "PeriodArray") -> None:
    _check_isinstance(left, right, PeriodArray)

    assert_numpy_array_equal(left._ndarray, right._ndarray, obj=f"{obj}._ndarray")
    assert_attr_equal("dtype", left, right, obj=obj)


def assert_datetime_array_equal(
    left, right, obj: str = "DatetimeArray", check_freq: bool = True
) -> None:
    __tracebackhide__ = True
    _check_isinstance(left, right, DatetimeArray)

    assert_numpy_array_equal(left._ndarray, right._ndarray, obj=f"{obj}._ndarray")
    if check_freq:
        assert_attr_equal("freq", left, right, obj=obj)
    assert_attr_equal("tz", left, right, obj=obj)


def assert_timedelta_array_equal(
    left, right, obj: str = "TimedeltaArray", check_freq: bool = True
) -> None:
    __tracebackhide__ = True
    _check_isinstance(left, right, TimedeltaArray)
    assert_numpy_array_equal(left._ndarray, right._ndarray, obj=f"{obj}._ndarray")
    if check_freq:
        assert_attr_equal("freq", left, right, obj=obj)


def raise_assert_detail(
    obj, message, left, right, diff=None, first_diff=None, index_values=None
) -> NoReturn:
    __tracebackhide__ = True

    msg = f"""{obj} are different

{message}"""

    if isinstance(index_values, Index):
        index_values = np.asarray(index_values)

    if isinstance(index_values, np.ndarray):
        msg += f"\n[index]: {pprint_thing(index_values)}"

    if isinstance(left, np.ndarray):
        left = pprint_thing(left)
    elif isinstance(left, (CategoricalDtype, NumpyEADtype)):
        left = repr(left)
    elif isinstance(left, StringDtype):
        # TODO(infer_string) this special case could be avoided if we have
        # a more informative repr https://github.com/pandas-dev/pandas/issues/59342
        left = f"StringDtype(storage={left.storage}, na_value={left.na_value})"

    if isinstance(right, np.ndarray):
        right = pprint_thing(right)
    elif isinstance(right, (CategoricalDtype, NumpyEADtype)):
        right = repr(right)
    elif isinstance(right, StringDtype):
        right = f"StringDtype(storage={right.storage}, na_value={right.na_value})"

    msg += f"""
[left]:  {left}
[right]: {right}"""

    if diff is not None:
        msg += f"\n[diff]: {diff}"

    if first_diff is not None:
        msg += f"\n{first_diff}"

    raise AssertionError(msg)


def assert_numpy_array_equal(
    left,
    right,
    strict_nan: bool = False,
    check_dtype: bool | Literal["equiv"] = True,
    err_msg=None,
    check_same=None,
    obj: str = "numpy array",
    index_values=None,
) -> None:
    """
    Check that 'np.ndarray' is equivalent.

    Parameters
    ----------
    left, right : numpy.ndarray or iterable
        The two arrays to be compared.
    strict_nan : bool, default False
        If True, consider NaN and None to be different.
    check_dtype : bool, default True
        Check dtype if both a and b are np.ndarray.
    err_msg : str, default None
        If provided, used as assertion message.
    check_same : None|'copy'|'same', default None
        Ensure left and right refer/do not refer to the same memory area.
    obj : str, default 'numpy array'
        Specify object name being compared, internally used to show appropriate
        assertion message.
    index_values : Index | numpy.ndarray, default None
        optional index (shared by both left and right), used in output.
    """
    __tracebackhide__ = True

    # instance validation
    # Show a detailed error message when classes are different
    assert_class_equal(left, right, obj=obj)
    # both classes must be an np.ndarray
    _check_isinstance(left, right, np.ndarray)

    def _get_base(obj):
        return obj.base if getattr(obj, "base", None) is not None else obj

    left_base = _get_base(left)
    right_base = _get_base(right)

    if check_same == "same":
        if left_base is not right_base:
            raise AssertionError(f"{repr(left_base)} is not {repr(right_base)}")
    elif check_same == "copy":
        if left_base is right_base:
            raise AssertionError(f"{repr(left_base)} is {repr(right_base)}")

