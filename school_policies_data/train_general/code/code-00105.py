    # If a KeyError, the key isn't in categories, so logically
    #  can't be in container either.
    try:
        loc = cat.categories.get_loc(key)
    except (KeyError, TypeError):
        return False

    # loc is the location of key in categories, but also the *value*
    # for key in container. So, `key` may be in categories,
    # but still not in `container`. Example ('b' in categories,
    # but not in values):
    # 'b' in Categorical(['a'], categories=['a', 'b'])  # False
    if is_scalar(loc):
        return loc in container
    else:
        # if categories is an IntervalIndex, loc is an array.
        return any(loc_ in container for loc_ in loc)


class Categorical(NDArrayBackedExtensionArray, PandasObject, ObjectStringArrayMixin):
    """
    Represent a categorical variable in classic R / S-plus fashion.

    `Categoricals` can only take on a limited, and usually fixed, number
    of possible values (`categories`). In contrast to statistical categorical
    variables, a `Categorical` might have an order, but numerical operations
    (additions, divisions, ...) are not possible.

    All values of the `Categorical` are either in `categories` or `np.nan`.
    Assigning values outside of `categories` will raise a `ValueError`. Order
    is defined by the order of the `categories`, not lexical order of the
    values.

    Parameters
    ----------
    values : list-like
        The values of the categorical. If categories are given, values not in
        categories will be replaced with NaN.
    categories : Index-like (unique), optional
        The unique categories for this categorical. If not given, the
        categories are assumed to be the unique values of `values` (sorted, if
        possible, otherwise in the order in which they appear).
    ordered : bool, default False
        Whether or not this categorical is treated as a ordered categorical.
        If True, the resulting categorical will be ordered.
        An ordered categorical respects, when sorted, the order of its
        `categories` attribute (which in turn is the `categories` argument, if
        provided).
    dtype : CategoricalDtype
        An instance of ``CategoricalDtype`` to use for this categorical.

    Attributes
    ----------
    categories : Index
        The categories of this categorical.
    codes : ndarray
        The codes (integer positions, which point to the categories) of this
        categorical, read only.
    ordered : bool
        Whether or not this Categorical is ordered.
    dtype : CategoricalDtype
        The instance of ``CategoricalDtype`` storing the ``categories``
        and ``ordered``.

    Methods
    -------
    from_codes
    __array__

    Raises
    ------
    ValueError
        If the categories do not validate.
    TypeError
        If an explicit ``ordered=True`` is given but no `categories` and the
        `values` are not sortable.

    See Also
    --------
    CategoricalDtype : Type for categorical data.
    CategoricalIndex : An Index with an underlying ``Categorical``.

    Notes
    -----
    See the `user guide
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html>`__
    for more.

    Examples
    --------
    >>> pd.Categorical([1, 2, 3, 1, 2, 3])
    [1, 2, 3, 1, 2, 3]
    Categories (3, int64): [1, 2, 3]

    >>> pd.Categorical(['a', 'b', 'c', 'a', 'b', 'c'])
    ['a', 'b', 'c', 'a', 'b', 'c']
    Categories (3, object): ['a', 'b', 'c']

    Missing values are not included as a category.

    >>> c = pd.Categorical([1, 2, 3, 1, 2, 3, np.nan])
    >>> c
    [1, 2, 3, 1, 2, 3, NaN]
    Categories (3, int64): [1, 2, 3]

    However, their presence is indicated in the `codes` attribute
    by code `-1`.

    >>> c.codes
    array([ 0,  1,  2,  0,  1,  2, -1], dtype=int8)

    Ordered `Categoricals` can be sorted according to the custom order
    of the categories and can have a min and max value.

    >>> c = pd.Categorical(['a', 'b', 'c', 'a', 'b', 'c'], ordered=True,
    ...                    categories=['c', 'b', 'a'])
    >>> c
    ['a', 'b', 'c', 'a', 'b', 'c']
    Categories (3, object): ['c' < 'b' < 'a']
    >>> c.min()
    'c'
    """

    # For comparisons, so that numpy uses our implementation if the compare
    # ops, which raise
    __array_priority__ = 1000
    # tolist is not actually deprecated, just suppressed in the __dir__
    _hidden_attrs = PandasObject._hidden_attrs | frozenset(["tolist"])
    _typ = "categorical"

    _dtype: CategoricalDtype

    @classmethod
    # error: Argument 2 of "_simple_new" is incompatible with supertype
    # "NDArrayBacked"; supertype defines the argument type as
    # "Union[dtype[Any], ExtensionDtype]"
    def _simple_new(  # type: ignore[override]
        cls, codes: np.ndarray, dtype: CategoricalDtype
    ) -> Self:
        # NB: This is not _quite_ as simple as the "usual" _simple_new
        codes = coerce_indexer_dtype(codes, dtype.categories)
        dtype = CategoricalDtype(ordered=False).update_dtype(dtype)
        return super()._simple_new(codes, dtype)

    def __init__(
        self,
        values,
        categories=None,
        ordered=None,
        dtype: Dtype | None = None,
        fastpath: bool | lib.NoDefault = lib.no_default,
        copy: bool = True,
    ) -> None:
        if fastpath is not lib.no_default:
            # GH#20110
            warnings.warn(
                "The 'fastpath' keyword in Categorical is deprecated and will "
                "be removed in a future version. Use Categorical.from_codes instead",
                DeprecationWarning,
                stacklevel=find_stack_level(),
            )
        else:
            fastpath = False

        dtype = CategoricalDtype._from_values_or_dtype(
            values, categories, ordered, dtype
        )
        # At this point, dtype is always a CategoricalDtype, but
        # we may have dtype.categories be None, and we need to
        # infer categories in a factorization step further below

        if fastpath:
            codes = coerce_indexer_dtype(values, dtype.categories)
            dtype = CategoricalDtype(ordered=False).update_dtype(dtype)
            super().__init__(codes, dtype)
            return

        if not is_list_like(values):
            # GH#38433
            raise TypeError("Categorical input must be list-like")

        # null_mask indicates missing values we want to exclude from inference.
        # This means: only missing values in list-likes (not arrays/ndframes).
        null_mask = np.array(False)

        # sanitize input
        vdtype = getattr(values, "dtype", None)
        if isinstance(vdtype, CategoricalDtype):
            if dtype.categories is None:
                dtype = CategoricalDtype(values.categories, dtype.ordered)
        elif not isinstance(values, (ABCIndex, ABCSeries, ExtensionArray)):
            values = com.convert_to_list_like(values)
            if isinstance(values, list) and len(values) == 0:
                # By convention, empty lists result in object dtype:
                values = np.array([], dtype=object)
            elif isinstance(values, np.ndarray):
                if values.ndim > 1:
                    # preempt sanitize_array from raising ValueError
                    raise NotImplementedError(
                        "> 1 ndim Categorical are not supported at this time"
                    )
                values = sanitize_array(values, None)
            else:
                # i.e. must be a list
                arr = sanitize_array(values, None)
                null_mask = isna(arr)
                if null_mask.any():
                    # We remove null values here, then below will re-insert
                    #  them, grep "full_codes"
                    arr_list = [values[idx] for idx in np.where(~null_mask)[0]]

                    # GH#44900 Do not cast to float if we have only missing values
                    if arr_list or arr.dtype == "object":
                        sanitize_dtype = None
                    else:
                        sanitize_dtype = arr.dtype

                    arr = sanitize_array(arr_list, None, dtype=sanitize_dtype)
                values = arr

        if dtype.categories is None:
            if isinstance(values.dtype, ArrowDtype) and issubclass(
                values.dtype.type, CategoricalDtypeType
            ):
                arr = values._pa_array.combine_chunks()
                categories = arr.dictionary.to_pandas(types_mapper=ArrowDtype)
                codes = arr.indices.to_numpy()
                dtype = CategoricalDtype(categories, values.dtype.pyarrow_dtype.ordered)
            else:
                if not isinstance(values, ABCIndex):
                    # in particular RangeIndex xref test_index_equal_range_categories
                    values = sanitize_array(values, None)
                try:
                    codes, categories = factorize(values, sort=True)
                except TypeError as err:
                    codes, categories = factorize(values, sort=False)
                    if dtype.ordered:
                        # raise, as we don't have a sortable data structure and so
                        # the user should give us one by specifying categories
                        raise TypeError(
                            "'values' is not ordered, please "
                            "explicitly specify the categories order "
                            "by passing in a categories argument."
                        ) from err

                # we're inferring from values
                dtype = CategoricalDtype(categories, dtype.ordered)

        elif isinstance(values.dtype, CategoricalDtype):
            old_codes = extract_array(values)._codes
            codes = recode_for_categories(
                old_codes, values.dtype.categories, dtype.categories, copy=copy
            )

        else:
            codes = _get_codes_for_values(values, dtype.categories)

        if null_mask.any():
            # Reinsert -1 placeholders for previously removed missing values
            full_codes = -np.ones(null_mask.shape, dtype=codes.dtype)
            full_codes[~null_mask] = codes
            codes = full_codes

        dtype = CategoricalDtype(ordered=False).update_dtype(dtype)
        arr = coerce_indexer_dtype(codes, dtype.categories)
        super().__init__(arr, dtype)

    @property
    def dtype(self) -> CategoricalDtype:
        """
        The :class:`~pandas.api.types.CategoricalDtype` for this instance.

        Examples
        --------
        >>> cat = pd.Categorical(['a', 'b'], ordered=True)
        >>> cat
        ['a', 'b']
        Categories (2, object): ['a' < 'b']
        >>> cat.dtype
        CategoricalDtype(categories=['a', 'b'], ordered=True, categories_dtype=object)
        """
        return self._dtype

    @property
    def _internal_fill_value(self) -> int:
        # using the specific numpy integer instead of python int to get
        #  the correct dtype back from _quantile in the all-NA case
        dtype = self._ndarray.dtype
        return dtype.type(-1)

    @classmethod
    def _from_sequence(
        cls, scalars, *, dtype: Dtype | None = None, copy: bool = False
    ) -> Self:
        return cls(scalars, dtype=dtype, copy=copy)

    @classmethod
    def _from_scalars(cls, scalars, *, dtype: DtypeObj) -> Self:
        if dtype is None:
            # The _from_scalars strictness doesn't make much sense in this case.
            raise NotImplementedError

        res = cls._from_sequence(scalars, dtype=dtype)

        # if there are any non-category elements in scalars, these will be
        #  converted to NAs in res.
        mask = isna(scalars)
        if not (mask == res.isna()).all():
            # Some non-category element in scalars got converted to NA in res.
            raise ValueError
        return res

    @overload
    def astype(self, dtype: npt.DTypeLike, copy: bool = ...) -> np.ndarray:
        ...

    @overload
    def astype(self, dtype: ExtensionDtype, copy: bool = ...) -> ExtensionArray:
        ...

    @overload
    def astype(self, dtype: AstypeArg, copy: bool = ...) -> ArrayLike:
        ...

    def astype(self, dtype: AstypeArg, copy: bool = True) -> ArrayLike:
        """
        Coerce this type to another dtype

        Parameters
        ----------
        dtype : numpy dtype or pandas type
        copy : bool, default True
            By default, astype always returns a newly allocated object.
            If copy is set to False and dtype is categorical, the original
            object is returned.
        """
        dtype = pandas_dtype(dtype)
        if self.dtype is dtype:
            result = self.copy() if copy else self

        elif isinstance(dtype, CategoricalDtype):
            # GH 10696/18593/18630
            dtype = self.dtype.update_dtype(dtype)
            self = self.copy() if copy else self
            result = self._set_dtype(dtype)

        elif isinstance(dtype, ExtensionDtype):
            return super().astype(dtype, copy=copy)

        elif dtype.kind in "iu" and self.isna().any():
            raise ValueError("Cannot convert float NaN to integer")

        elif len(self.codes) == 0 or len(self.categories) == 0:
            # For NumPy 1.x compatibility we cannot use copy=None.  And
            # `copy=False` has the meaning of `copy=None` here:
            if not copy:
                result = np.asarray(self, dtype=dtype)
            else:
                result = np.array(self, dtype=dtype)

        else:
            # GH8628 (PERF): astype category codes instead of astyping array
            new_cats = self.categories._values

            try:
                new_cats = new_cats.astype(dtype=dtype, copy=copy)
                fill_value = self.categories._na_value
                if not is_valid_na_for_dtype(fill_value, dtype):
                    fill_value = lib.item_from_zerodim(
                        np.array(self.categories._na_value).astype(dtype)
                    )
            except (
                TypeError,  # downstream error msg for CategoricalIndex is misleading
                ValueError,
            ):
                msg = f"Cannot cast {self.categories.dtype} dtype to {dtype}"
                raise ValueError(msg)

            result = take_nd(
                new_cats, ensure_platform_int(self._codes), fill_value=fill_value
            )

        return result

    def to_list(self):
        """
        Alias for tolist.
        """
        # GH#51254
        warnings.warn(
            "Categorical.to_list is deprecated and will be removed in a future "
            "version. Use obj.tolist() instead",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
        return self.tolist()

    @classmethod
    def _from_inferred_categories(
        cls, inferred_categories, inferred_codes, dtype, true_values=None
    ) -> Self:
        """
        Construct a Categorical from inferred values.

        For inferred categories (`dtype` is None) the categories are sorted.
        For explicit `dtype`, the `inferred_categories` are cast to the
        appropriate type.

        Parameters
        ----------
        inferred_categories : Index
        inferred_codes : Index
        dtype : CategoricalDtype or 'category'
        true_values : list, optional
            If none are provided, the default ones are
            "True", "TRUE", and "true."

        Returns
        -------
        Categorical
        """
        from pandas import (
            Index,
            to_datetime,
            to_numeric,
            to_timedelta,
        )

        cats = Index(inferred_categories)
        known_categories = (
            isinstance(dtype, CategoricalDtype) and dtype.categories is not None
        )

        if known_categories:
            # Convert to a specialized type with `dtype` if specified.
            if is_any_real_numeric_dtype(dtype.categories.dtype):
                cats = to_numeric(inferred_categories, errors="coerce")
            elif lib.is_np_dtype(dtype.categories.dtype, "M"):
                cats = to_datetime(inferred_categories, errors="coerce")
            elif lib.is_np_dtype(dtype.categories.dtype, "m"):
                cats = to_timedelta(inferred_categories, errors="coerce")
            elif is_bool_dtype(dtype.categories.dtype):
                if true_values is None:
                    true_values = ["True", "TRUE", "true"]

                # error: Incompatible types in assignment (expression has type
                # "ndarray", variable has type "Index")
                cats = cats.isin(true_values)  # type: ignore[assignment]

        if known_categories:
            # Recode from observation order to dtype.categories order.
            categories = dtype.categories
            codes = recode_for_categories(inferred_codes, cats, categories)
        elif not cats.is_monotonic_increasing:
            # Sort categories and recode for unknown categories.
            unsorted = cats.copy()
            categories = cats.sort_values()

            codes = recode_for_categories(inferred_codes, unsorted, categories)
            dtype = CategoricalDtype(categories, ordered=False)
        else:
            dtype = CategoricalDtype(cats, ordered=False)
            codes = inferred_codes

        return cls._simple_new(codes, dtype=dtype)

    @classmethod
    def from_codes(
        cls,
        codes,
        categories=None,
        ordered=None,
        dtype: Dtype | None = None,
        validate: bool = True,
    ) -> Self:
        """
        Make a Categorical type from codes and categories or dtype.

        This constructor is useful if you already have codes and
        categories/dtype and so do not need the (computation intensive)
        factorization step, which is usually done on the constructor.

        If your data does not follow this convention, please use the normal
        constructor.

        Parameters
        ----------
        codes : array-like of int
            An integer array, where each integer points to a category in
            categories or dtype.categories, or else is -1 for NaN.
        categories : index-like, optional
            The categories for the categorical. Items need to be unique.
            If the categories are not given here, then they must be provided
            in `dtype`.
        ordered : bool, optional
            Whether or not this categorical is treated as an ordered
            categorical. If not given here or in `dtype`, the resulting
            categorical will be unordered.
        dtype : CategoricalDtype or "category", optional
            If :class:`CategoricalDtype`, cannot be used together with
            `categories` or `ordered`.
        validate : bool, default True
            If True, validate that the codes are valid for the dtype.
            If False, don't validate that the codes are valid. Be careful about skipping
            validation, as invalid codes can lead to severe problems, such as segfaults.

            .. versionadded:: 2.1.0

        Returns
        -------
        Categorical

        Examples
        --------
        >>> dtype = pd.CategoricalDtype(['a', 'b'], ordered=True)
        >>> pd.Categorical.from_codes(codes=[0, 1, 0, 1], dtype=dtype)
        ['a', 'b', 'a', 'b']
        Categories (2, object): ['a' < 'b']
        """
        dtype = CategoricalDtype._from_values_or_dtype(
            categories=categories, ordered=ordered, dtype=dtype
        )
        if dtype.categories is None:
            msg = (
                "The categories must be provided in 'categories' or "
                "'dtype'. Both were None."
            )
            raise ValueError(msg)

        if validate:
            # beware: non-valid codes may segfault
            codes = cls._validate_codes_for_dtype(codes, dtype=dtype)

        return cls._simple_new(codes, dtype=dtype)

    # ------------------------------------------------------------------
    # Categories/Codes/Ordered

    @property
    def categories(self) -> Index:
        """
        The categories of this categorical.

        Setting assigns new values to each category (effectively a rename of
        each individual category).

        The assigned value has to be a list-like object. All items must be
        unique and the number of items in the new categories must be the same
        as the number of items in the old categories.

        Raises
        ------
        ValueError
            If the new categories do not validate as categories or if the
            number of new categories is unequal the number of old categories

        See Also
        --------
        rename_categories : Rename categories.
        reorder_categories : Reorder categories.
        add_categories : Add new categories.
        remove_categories : Remove the specified categories.
        remove_unused_categories : Remove categories which are not used.
        set_categories : Set the categories to the specified ones.

        Examples
        --------
        For :class:`pandas.Series`:

        >>> ser = pd.Series(['a', 'b', 'c', 'a'], dtype='category')
        >>> ser.cat.categories
        Index(['a', 'b', 'c'], dtype='object')

        >>> raw_cat = pd.Categorical(['a', 'b', 'c', 'a'], categories=['b', 'c', 'd'])
        >>> ser = pd.Series(raw_cat)
        >>> ser.cat.categories
        Index(['b', 'c', 'd'], dtype='object')

        For :class:`pandas.Categorical`:

        >>> cat = pd.Categorical(['a', 'b'], ordered=True)
        >>> cat.categories
        Index(['a', 'b'], dtype='object')

        For :class:`pandas.CategoricalIndex`:

        >>> ci = pd.CategoricalIndex(['a', 'c', 'b', 'a', 'c', 'b'])
        >>> ci.categories
        Index(['a', 'b', 'c'], dtype='object')

        >>> ci = pd.CategoricalIndex(['a', 'c'], categories=['c', 'b', 'a'])
        >>> ci.categories
        Index(['c', 'b', 'a'], dtype='object')
        """
        return self.dtype.categories

    @property
    def ordered(self) -> Ordered:
        """
        Whether the categories have an ordered relationship.

        Examples
        --------
        For :class:`pandas.Series`:

        >>> ser = pd.Series(['a', 'b', 'c', 'a'], dtype='category')
        >>> ser.cat.ordered
        False

        >>> raw_cat = pd.Categorical(['a', 'b', 'c', 'a'], ordered=True)
        >>> ser = pd.Series(raw_cat)
        >>> ser.cat.ordered
        True

        For :class:`pandas.Categorical`:

        >>> cat = pd.Categorical(['a', 'b'], ordered=True)
        >>> cat.ordered
        True

        >>> cat = pd.Categorical(['a', 'b'], ordered=False)
        >>> cat.ordered
        False

        For :class:`pandas.CategoricalIndex`:

        >>> ci = pd.CategoricalIndex(['a', 'b'], ordered=True)
        >>> ci.ordered
        True

        >>> ci = pd.CategoricalIndex(['a', 'b'], ordered=False)
        >>> ci.ordered
        False
        """
        return self.dtype.ordered

    @property
    def codes(self) -> np.ndarray:
        """
        The category codes of this categorical index.

        Codes are an array of integers which are the positions of the actual
        values in the categories array.

        There is no setter, use the other categorical methods and the normal item
        setter to change values in the categorical.

        Returns
        -------
        ndarray[int]
            A non-writable view of the ``codes`` array.

        Examples
        --------
        For :class:`pandas.Categorical`:

        >>> cat = pd.Categorical(['a', 'b'], ordered=True)
        >>> cat.codes
        array([0, 1], dtype=int8)

        For :class:`pandas.CategoricalIndex`:

        >>> ci = pd.CategoricalIndex(['a', 'b', 'c', 'a', 'b', 'c'])
        >>> ci.codes
        array([0, 1, 2, 0, 1, 2], dtype=int8)

        >>> ci = pd.CategoricalIndex(['a', 'c'], categories=['c', 'b', 'a'])
        >>> ci.codes
        array([2, 0], dtype=int8)
        """
        v = self._codes.view()
        v.flags.writeable = False
        return v

    def _set_categories(self, categories, fastpath: bool = False) -> None:
        """
        Sets new categories inplace

        Parameters
        ----------
        fastpath : bool, default False
           Don't perform validation of the categories for uniqueness or nulls

        Examples
        --------
        >>> c = pd.Categorical(['a', 'b'])
        >>> c
        ['a', 'b']
        Categories (2, object): ['a', 'b']

        >>> c._set_categories(pd.Index(['a', 'c']))
        >>> c
        ['a', 'c']
        Categories (2, object): ['a', 'c']
        """
        if fastpath:
            new_dtype = CategoricalDtype._from_fastpath(categories, self.ordered)
        else:
            new_dtype = CategoricalDtype(categories, ordered=self.ordered)
        if (
            not fastpath
            and self.dtype.categories is not None
            and len(new_dtype.categories) != len(self.dtype.categories)
        ):
            raise ValueError(
                "new categories need to have the same number of "
                "items as the old categories!"
            )

        super().__init__(self._ndarray, new_dtype)

    def _set_dtype(self, dtype: CategoricalDtype) -> Self:
        """
        Internal method for directly updating the CategoricalDtype

        Parameters
        ----------
        dtype : CategoricalDtype

        Notes
        -----
        We don't do any validation here. It's assumed that the dtype is
        a (valid) instance of `CategoricalDtype`.
        """
        codes = recode_for_categories(self.codes, self.categories, dtype.categories)
        return type(self)._simple_new(codes, dtype=dtype)

    def set_ordered(self, value: bool) -> Self:
        """
        Set the ordered attribute to the boolean value.

        Parameters
        ----------
        value : bool
           Set whether this categorical is ordered (True) or not (False).
        """
        new_dtype = CategoricalDtype(self.categories, ordered=value)
        cat = self.copy()
        NDArrayBacked.__init__(cat, cat._ndarray, new_dtype)
        return cat

    def as_ordered(self) -> Self:
        """
        Set the Categorical to be ordered.

        Returns
        -------
        Categorical
            Ordered Categorical.

        Examples
        --------
        For :class:`pandas.Series`:

        >>> ser = pd.Series(['a', 'b', 'c', 'a'], dtype='category')
        >>> ser.cat.ordered
        False
        >>> ser = ser.cat.as_ordered()
        >>> ser.cat.ordered
        True

        For :class:`pandas.CategoricalIndex`:

        >>> ci = pd.CategoricalIndex(['a', 'b', 'c', 'a'])
        >>> ci.ordered
        False
        >>> ci = ci.as_ordered()
        >>> ci.ordered
        True
        """
        return self.set_ordered(True)

    def as_unordered(self) -> Self:
        """
        Set the Categorical to be unordered.

        Returns
        -------
        Categorical
            Unordered Categorical.

        Examples
        --------
        For :class:`pandas.Series`:

        >>> raw_cat = pd.Categorical(['a', 'b', 'c', 'a'], ordered=True)
        >>> ser = pd.Series(raw_cat)
        >>> ser.cat.ordered
        True
        >>> ser = ser.cat.as_unordered()
        >>> ser.cat.ordered
        False

        For :class:`pandas.CategoricalIndex`:

        >>> ci = pd.CategoricalIndex(['a', 'b', 'c', 'a'], ordered=True)
        >>> ci.ordered
        True
        >>> ci = ci.as_unordered()
        >>> ci.ordered
        False
        """
        return self.set_ordered(False)

    def set_categories(self, new_categories, ordered=None, rename: bool = False):
        """
        Set the categories to the specified new categories.

        ``new_categories`` can include new categories (which will result in
        unused categories) or remove old categories (which results in values
        set to ``NaN``). If ``rename=True``, the categories will simply be renamed
        (less or more items than in old categories will result in values set to
        ``NaN`` or in unused categories respectively).

        This method can be used to perform more than one action of adding,
        removing, and reordering simultaneously and is therefore faster than
        performing the individual steps via the more specialised methods.

        On the other hand this methods does not do checks (e.g., whether the
        old categories are included in the new categories on a reorder), which
        can result in surprising changes, for example when using special string
        dtypes, which does not considers a S1 string equal to a single char
        python string.

        Parameters
        ----------
        new_categories : Index-like
           The categories in new order.
        ordered : bool, default False
           Whether or not the categorical is treated as a ordered categorical.
           If not given, do not change the ordered information.
        rename : bool, default False
           Whether or not the new_categories should be considered as a rename
           of the old categories or as reordered categories.

        Returns
        -------
        Categorical with reordered categories.

        Raises
        ------
        ValueError
            If new_categories does not validate as categories

        See Also
        --------
        rename_categories : Rename categories.
        reorder_categories : Reorder categories.
        add_categories : Add new categories.
        remove_categories : Remove the specified categories.
        remove_unused_categories : Remove categories which are not used.

        Examples
        --------
        For :class:`pandas.Series`:

        >>> raw_cat = pd.Categorical(['a', 'b', 'c', 'A'],
        ...                           categories=['a', 'b', 'c'], ordered=True)
        >>> ser = pd.Series(raw_cat)
        >>> ser
        0   a
        1   b
        2   c
        3   NaN
        dtype: category
        Categories (3, object): ['a' < 'b' < 'c']

        >>> ser.cat.set_categories(['A', 'B', 'C'], rename=True)
        0   A
        1   B
        2   C
        3   NaN
        dtype: category
        Categories (3, object): ['A' < 'B' < 'C']

        For :class:`pandas.CategoricalIndex`:

        >>> ci = pd.CategoricalIndex(['a', 'b', 'c', 'A'],
        ...                          categories=['a', 'b', 'c'], ordered=True)
        >>> ci
        CategoricalIndex(['a', 'b', 'c', nan], categories=['a', 'b', 'c'],
                         ordered=True, dtype='category')

        >>> ci.set_categories(['A', 'b', 'c'])
        CategoricalIndex([nan, 'b', 'c', nan], categories=['A', 'b', 'c'],
                         ordered=True, dtype='category')
        >>> ci.set_categories(['A', 'b', 'c'], rename=True)
        CategoricalIndex(['A', 'b', 'c', nan], categories=['A', 'b', 'c'],
                         ordered=True, dtype='category')
        """

        if ordered is None:
            ordered = self.dtype.ordered
        new_dtype = CategoricalDtype(new_categories, ordered=ordered)

        cat = self.copy()
        if rename:
            if cat.dtype.categories is not None and len(new_dtype.categories) < len(
                cat.dtype.categories
            ):
                # remove all _codes which are larger and set to -1/NaN
                cat._codes[cat._codes >= len(new_dtype.categories)] = -1
            codes = cat._codes
        else:
            codes = recode_for_categories(
                cat.codes, cat.categories, new_dtype.categories
            )
        NDArrayBacked.__init__(cat, codes, new_dtype)
        return cat

    def rename_categories(self, new_categories) -> Self:
        """
        Rename categories.

        Parameters
        ----------
        new_categories : list-like, dict-like or callable

            New categories which will replace old categories.

            * list-like: all items must be unique and the number of items in
              the new categories must match the existing number of categories.

            * dict-like: specifies a mapping from
              old categories to new. Categories not contained in the mapping
              are passed through and extra categories in the mapping are
              ignored.

            * callable : a callable that is called on all items in the old
              categories and whose return values comprise the new categories.

        Returns
        -------
        Categorical
            Categorical with renamed categories.

        Raises
        ------
        ValueError
            If new categories are list-like and do not have the same number of
            items than the current categories or do not validate as categories

        See Also
        --------
        reorder_categories : Reorder categories.
        add_categories : Add new categories.
        remove_categories : Remove the specified categories.
        remove_unused_categories : Remove categories which are not used.
        set_categories : Set the categories to the specified ones.

        Examples
        --------
        >>> c = pd.Categorical(['a', 'a', 'b'])
        >>> c.rename_categories([0, 1])
        [0, 0, 1]
        Categories (2, int64): [0, 1]

        For dict-like ``new_categories``, extra keys are ignored and
        categories not in the dictionary are passed through

        >>> c.rename_categories({'a': 'A', 'c': 'C'})
        ['A', 'A', 'b']
        Categories (2, object): ['A', 'b']

        You may also provide a callable to create the new categories

        >>> c.rename_categories(lambda x: x.upper())
        ['A', 'A', 'B']
        Categories (2, object): ['A', 'B']
        """

        if is_dict_like(new_categories):
            new_categories = [
                new_categories.get(item, item) for item in self.categories
            ]
        elif callable(new_categories):
            new_categories = [new_categories(item) for item in self.categories]

        cat = self.copy()
        cat._set_categories(new_categories)
        return cat

    def reorder_categories(self, new_categories, ordered=None) -> Self:
        """
        Reorder categories as specified in new_categories.

        ``new_categories`` need to include all old categories and no new category
        items.

        Parameters
        ----------
        new_categories : Index-like
           The categories in new order.
        ordered : bool, optional
           Whether or not the categorical is treated as a ordered categorical.
           If not given, do not change the ordered information.

        Returns
        -------
        Categorical
            Categorical with reordered categories.

        Raises
        ------
        ValueError
            If the new categories do not contain all old category items or any
            new ones

        See Also
        --------
        rename_categories : Rename categories.
        add_categories : Add new categories.
        remove_categories : Remove the specified categories.
        remove_unused_categories : Remove categories which are not used.
        set_categories : Set the categories to the specified ones.

        Examples
        --------
        For :class:`pandas.Series`:

        >>> ser = pd.Series(['a', 'b', 'c', 'a'], dtype='category')
        >>> ser = ser.cat.reorder_categories(['c', 'b', 'a'], ordered=True)
        >>> ser
        0   a
        1   b
        2   c
        3   a
        dtype: category
        Categories (3, object): ['c' < 'b' < 'a']

        >>> ser.sort_values()
        2   c
        1   b
        0   a
        3   a
        dtype: category
        Categories (3, object): ['c' < 'b' < 'a']

        For :class:`pandas.CategoricalIndex`:

        >>> ci = pd.CategoricalIndex(['a', 'b', 'c', 'a'])
        >>> ci
        CategoricalIndex(['a', 'b', 'c', 'a'], categories=['a', 'b', 'c'],
                         ordered=False, dtype='category')
        >>> ci.reorder_categories(['c', 'b', 'a'], ordered=True)
        CategoricalIndex(['a', 'b', 'c', 'a'], categories=['c', 'b', 'a'],
                         ordered=True, dtype='category')
        """
        if (
            len(self.categories) != len(new_categories)
            or not self.categories.difference(new_categories).empty
        ):
            raise ValueError(
                "items in new_categories are not the same as in old categories"
            )
        return self.set_categories(new_categories, ordered=ordered)

    def add_categories(self, new_categories) -> Self:
        """
        Add new categories.

        `new_categories` will be included at the last/highest place in the
        categories and will be unused directly after this call.

        Parameters
        ----------
        new_categories : category or list-like of category
           The new categories to be included.

        Returns
        -------
        Categorical
            Categorical with new categories added.

        Raises
        ------
        ValueError
            If the new categories include old categories or do not validate as
            categories

        See Also
        --------
        rename_categories : Rename categories.
        reorder_categories : Reorder categories.
        remove_categories : Remove the specified categories.
        remove_unused_categories : Remove categories which are not used.
        set_categories : Set the categories to the specified ones.

        Examples
        --------
        >>> c = pd.Categorical(['c', 'b', 'c'])
        >>> c
        ['c', 'b', 'c']
        Categories (2, object): ['b', 'c']

        >>> c.add_categories(['d', 'a'])
        ['c', 'b', 'c']
        Categories (4, object): ['b', 'c', 'd', 'a']
        """

        if not is_list_like(new_categories):
            new_categories = [new_categories]
        already_included = set(new_categories) & set(self.dtype.categories)
        if len(already_included) != 0:
            raise ValueError(
                f"new categories must not include old categories: {already_included}"
            )

        if hasattr(new_categories, "dtype"):
            from pandas import Series

            dtype = find_common_type(
                [self.dtype.categories.dtype, new_categories.dtype]
            )
            new_categories = Series(
                list(self.dtype.categories) + list(new_categories), dtype=dtype
            )
        else:
            new_categories = list(self.dtype.categories) + list(new_categories)

        new_dtype = CategoricalDtype(new_categories, self.ordered)
        cat = self.copy()
        codes = coerce_indexer_dtype(cat._ndarray, new_dtype.categories)
        NDArrayBacked.__init__(cat, codes, new_dtype)
        return cat

    def remove_categories(self, removals) -> Self:
        """
        Remove the specified categories.

        `removals` must be included in the old categories. Values which were in
        the removed categories will be set to NaN

        Parameters
        ----------
        removals : category or list of categories
           The categories which should be removed.

        Returns
        -------
        Categorical
            Categorical with removed categories.

        Raises
        ------
        ValueError
            If the removals are not contained in the categories

        See Also
        --------
        rename_categories : Rename categories.
        reorder_categories : Reorder categories.
        add_categories : Add new categories.
        remove_unused_categories : Remove categories which are not used.
        set_categories : Set the categories to the specified ones.

        Examples
        --------
        >>> c = pd.Categorical(['a', 'c', 'b', 'c', 'd'])
        >>> c
        ['a', 'c', 'b', 'c', 'd']
        Categories (4, object): ['a', 'b', 'c', 'd']

        >>> c.remove_categories(['d', 'a'])
        [NaN, 'c', 'b', 'c', NaN]
        Categories (2, object): ['b', 'c']
        """
        from pandas import Index

        if not is_list_like(removals):
            removals = [removals]

        removals = Index(removals).unique().dropna()
        new_categories = (
            self.dtype.categories.difference(removals, sort=False)
            if self.dtype.ordered is True
            else self.dtype.categories.difference(removals)
        )
        not_included = removals.difference(self.dtype.categories)

        if len(not_included) != 0:
            not_included = set(not_included)
            raise ValueError(f"removals must all be in old categories: {not_included}")

        return self.set_categories(new_categories, ordered=self.ordered, rename=False)

    def remove_unused_categories(self) -> Self:
        """
        Remove categories which are not used.

        Returns
        -------
        Categorical
            Categorical with unused categories dropped.

        See Also
        --------
        rename_categories : Rename categories.
        reorder_categories : Reorder categories.
        add_categories : Add new categories.
        remove_categories : Remove the specified categories.
        set_categories : Set the categories to the specified ones.

        Examples
        --------
        >>> c = pd.Categorical(['a', 'c', 'b', 'c', 'd'])
        >>> c
        ['a', 'c', 'b', 'c', 'd']
        Categories (4, object): ['a', 'b', 'c', 'd']

        >>> c[2] = 'a'
        >>> c[4] = 'c'
        >>> c
        ['a', 'c', 'a', 'c', 'c']
        Categories (4, object): ['a', 'b', 'c', 'd']

        >>> c.remove_unused_categories()
        ['a', 'c', 'a', 'c', 'c']
        Categories (2, object): ['a', 'c']
        """
        idx, inv = np.unique(self._codes, return_inverse=True)

        if idx.size != 0 and idx[0] == -1:  # na sentinel
            idx, inv = idx[1:], inv - 1

        new_categories = self.dtype.categories.take(idx)
        new_dtype = CategoricalDtype._from_fastpath(
            new_categories, ordered=self.ordered
        )
        new_codes = coerce_indexer_dtype(inv, new_dtype.categories)

        cat = self.copy()
        NDArrayBacked.__init__(cat, new_codes, new_dtype)
        return cat

    # ------------------------------------------------------------------

    def map(
        self,
        mapper,
        na_action: Literal["ignore"] | None | lib.NoDefault = lib.no_default,
    ):
        """
        Map categories using an input mapping or function.

        Maps the categories to new categories. If the mapping correspondence is
        one-to-one the result is a :class:`~pandas.Categorical` which has the
        same order property as the original, otherwise a :class:`~pandas.Index`
        is returned. NaN values are unaffected.

        If a `dict` or :class:`~pandas.Series` is used any unmapped category is
        mapped to `NaN`. Note that if this happens an :class:`~pandas.Index`
        will be returned.

        Parameters
        ----------
        mapper : function, dict, or Series
            Mapping correspondence.
        na_action : {None, 'ignore'}, default 'ignore'
            If 'ignore', propagate NaN values, without passing them to the
            mapping correspondence.

            .. deprecated:: 2.1.0

               The default value of 'ignore' has been deprecated and will be changed to
               None in the future.

        Returns
        -------
        pandas.Categorical or pandas.Index
            Mapped categorical.

        See Also
        --------
        CategoricalIndex.map : Apply a mapping correspondence on a
            :class:`~pandas.CategoricalIndex`.
        Index.map : Apply a mapping correspondence on an
            :class:`~pandas.Index`.
        Series.map : Apply a mapping correspondence on a
            :class:`~pandas.Series`.
        Series.apply : Apply more complex functions on a
            :class:`~pandas.Series`.

        Examples
        --------
        >>> cat = pd.Categorical(['a', 'b', 'c'])
        >>> cat
        ['a', 'b', 'c']
        Categories (3, object): ['a', 'b', 'c']
        >>> cat.map(lambda x: x.upper(), na_action=None)
        ['A', 'B', 'C']
        Categories (3, object): ['A', 'B', 'C']
        >>> cat.map({'a': 'first', 'b': 'second', 'c': 'third'}, na_action=None)
        ['first', 'second', 'third']
        Categories (3, object): ['first', 'second', 'third']

        If the mapping is one-to-one the ordering of the categories is
        preserved:

        >>> cat = pd.Categorical(['a', 'b', 'c'], ordered=True)
        >>> cat
        ['a', 'b', 'c']
        Categories (3, object): ['a' < 'b' < 'c']
        >>> cat.map({'a': 3, 'b': 2, 'c': 1}, na_action=None)
        [3, 2, 1]
        Categories (3, int64): [3 < 2 < 1]

        If the mapping is not one-to-one an :class:`~pandas.Index` is returned:

        >>> cat.map({'a': 'first', 'b': 'second', 'c': 'first'}, na_action=None)
        Index(['first', 'second', 'first'], dtype='object')

        If a `dict` is used, all unmapped categories are mapped to `NaN` and
        the result is an :class:`~pandas.Index`:

        >>> cat.map({'a': 'first', 'b': 'second'}, na_action=None)
        Index(['first', 'second', nan], dtype='object')
        """
        if na_action is lib.no_default:
            warnings.warn(
                "The default value of 'ignore' for the `na_action` parameter in "
                "pandas.Categorical.map is deprecated and will be "
                "changed to 'None' in a future version. Please set na_action to the "
                "desired value to avoid seeing this warning",
                FutureWarning,
                stacklevel=find_stack_level(),
            )
            na_action = "ignore"

        assert callable(mapper) or is_dict_like(mapper)

        new_categories = self.categories.map(mapper)

        has_nans = np.any(self._codes == -1)

        na_val = np.nan
        if na_action is None and has_nans:
            na_val = mapper(np.nan) if callable(mapper) else mapper.get(np.nan, np.nan)

        if new_categories.is_unique and not new_categories.hasnans and na_val is np.nan:
            new_dtype = CategoricalDtype(new_categories, ordered=self.ordered)
            return self.from_codes(self._codes.copy(), dtype=new_dtype, validate=False)

        if has_nans:
            new_categories = new_categories.insert(len(new_categories), na_val)

        return np.take(new_categories, self._codes)

    __eq__ = _cat_compare_op(operator.eq)
    __ne__ = _cat_compare_op(operator.ne)
    __lt__ = _cat_compare_op(operator.lt)
    __gt__ = _cat_compare_op(operator.gt)
    __le__ = _cat_compare_op(operator.le)
    __ge__ = _cat_compare_op(operator.ge)

    # -------------------------------------------------------------
    # Validators; ideally these can be de-duplicated

    def _validate_setitem_value(self, value):
        if not is_hashable(value):
            # wrap scalars and hashable-listlikes in list
            return self._validate_listlike(value)
        else:
            return self._validate_scalar(value)

    def _validate_scalar(self, fill_value):
        """
        Convert a user-facing fill_value to a representation to use with our
        underlying ndarray, raising TypeError if this is not possible.

        Parameters
        ----------
        fill_value : object

        Returns
        -------
        fill_value : int

        Raises
        ------
        TypeError
        """

        if is_valid_na_for_dtype(fill_value, self.categories.dtype):
            fill_value = -1
        elif fill_value in self.categories:
            fill_value = self._unbox_scalar(fill_value)
        else:
            raise TypeError(
                "Cannot setitem on a Categorical with a new "
                f"category ({fill_value}), set the categories first"
            ) from None
        return fill_value

    @classmethod
    def _validate_codes_for_dtype(cls, codes, *, dtype: CategoricalDtype) -> np.ndarray:
        if isinstance(codes, ExtensionArray) and is_integer_dtype(codes.dtype):
            # Avoid the implicit conversion of Int to object
            if isna(codes).any():
                raise ValueError("codes cannot contain NA values")
            codes = codes.to_numpy(dtype=np.int64)
        else:
            codes = np.asarray(codes)
        if len(codes) and codes.dtype.kind not in "iu":
            raise ValueError("codes need to be array-like integers")

        if len(codes) and (codes.max() >= len(dtype.categories) or codes.min() < -1):
            raise ValueError("codes need to be between -1 and len(categories)-1")
        return codes

    # -------------------------------------------------------------

    @ravel_compat
    def __array__(
        self, dtype: NpDtype | None = None, copy: bool | None = None
    ) -> np.ndarray:
        """
        The numpy array interface.

        Users should not call this directly. Rather, it is invoked by
        :func:`numpy.array` and :func:`numpy.asarray`.

        Parameters
        ----------
        dtype : np.dtype or None
            Specifies the the dtype for the array.

        copy : bool or None, optional
            See :func:`numpy.asarray`.

        Returns
        -------
        numpy.array
            A numpy array of either the specified dtype or,
            if dtype==None (default), the same dtype as
            categorical.categories.dtype.

        Examples
        --------

        >>> cat = pd.Categorical(['a', 'b'], ordered=True)

        The following calls ``cat.__array__``

        >>> np.asarray(cat)
        array(['a', 'b'], dtype=object)
        """
        if copy is False:
            warnings.warn(
                "Starting with NumPy 2.0, the behavior of the 'copy' keyword has "
                "changed and passing 'copy=False' raises an error when returning "
                "a zero-copy NumPy array is not possible. pandas will follow "
                "this behavior starting with pandas 3.0.\nThis conversion to "
                "NumPy requires a copy, but 'copy=False' was passed. Consider "
                "using 'np.asarray(..)' instead.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )

        ret = take_nd(self.categories._values, self._codes)
        # When we're a Categorical[ExtensionArray], like Interval,
        # we need to ensure __array__ gets all the way to an
        # ndarray.

        # `take_nd` should already make a copy, so don't force again.
        return np.asarray(ret, dtype=dtype)

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
        # for binary ops, use our custom dunder methods
        result = arraylike.maybe_dispatch_ufunc_to_dunder_op(
            self, ufunc, method, *inputs, **kwargs
        )
        if result is not NotImplemented:
            return result

        if "out" in kwargs:
            # e.g. test_numpy_ufuncs_out
            return arraylike.dispatch_ufunc_with_out(
                self, ufunc, method, *inputs, **kwargs
            )

        if method == "reduce":
            # e.g. TestCategoricalAnalytics::test_min_max_ordered
            result = arraylike.dispatch_reduction_ufunc(
                self, ufunc, method, *inputs, **kwargs
            )
            if result is not NotImplemented:
                return result

        # for all other cases, raise for now (similarly as what happens in
        # Series.__array_prepare__)
        raise TypeError(
            f"Object with dtype {self.dtype} cannot perform "
            f"the numpy op {ufunc.__name__}"
        )

    def __setstate__(self, state) -> None:
        """Necessary for making this object picklable"""
        if not isinstance(state, dict):
            return super().__setstate__(state)

        if "_dtype" not in state:
            state["_dtype"] = CategoricalDtype(state["_categories"], state["_ordered"])

        if "_codes" in state and "_ndarray" not in state:
            # backward compat, changed what is property vs attribute
            state["_ndarray"] = state.pop("_codes")

        super().__setstate__(state)

    @property
    def nbytes(self) -> int:
        return self._codes.nbytes + self.dtype.categories.values.nbytes

    def memory_usage(self, deep: bool = False) -> int:
        """
        Memory usage of my values

        Parameters
        ----------
        deep : bool
            Introspect the data deeply, interrogate
            `object` dtypes for system-level memory consumption

        Returns
        -------
        bytes used

        Notes
        -----
        Memory usage does not include memory consumed by elements that
        are not components of the array if deep=False

        See Also
        --------
        numpy.ndarray.nbytes
        """
        return self._codes.nbytes + self.dtype.categories.memory_usage(deep=deep)

    def isna(self) -> npt.NDArray[np.bool_]:
        """
        Detect missing values

        Missing values (-1 in .codes) are detected.

        Returns
        -------
        np.ndarray[bool] of whether my values are null

        See Also
        --------
        isna : Top-level isna.
        isnull : Alias of isna.
        Categorical.notna : Boolean inverse of Categorical.isna.

        """
        return self._codes == -1

    isnull = isna

    def notna(self) -> npt.NDArray[np.bool_]:
        """
        Inverse of isna

        Both missing values (-1 in .codes) and NA as a category are detected as
        null.

        Returns
        -------
        np.ndarray[bool] of whether my values are not null

        See Also
        --------
        notna : Top-level notna.
        notnull : Alias of notna.
        Categorical.isna : Boolean inverse of Categorical.notna.

        """
        return ~self.isna()

    notnull = notna

    def value_counts(self, dropna: bool = True) -> Series:
        """
        Return a Series containing counts of each category.

        Every category will have an entry, even those with a count of 0.

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
        from pandas import (
            CategoricalIndex,
            Series,
        )

        code, cat = self._codes, self.categories
        ncat, mask = (len(cat), code >= 0)
        ix, clean = np.arange(ncat), mask.all()

        if dropna or clean:
            obs = code if clean else code[mask]
            count = np.bincount(obs, minlength=ncat or 0)
        else:
            count = np.bincount(np.where(mask, code, ncat))
            ix = np.append(ix, -1)

        ix = coerce_indexer_dtype(ix, self.dtype.categories)
        ix = self._from_backing_data(ix)

        return Series(
            count, index=CategoricalIndex(ix), dtype="int64", name="count", copy=False
        )

    # error: Argument 2 of "_empty" is incompatible with supertype
    # "NDArrayBackedExtensionArray"; supertype defines the argument type as
    # "ExtensionDtype"
    @classmethod
    def _empty(  # type: ignore[override]
        cls, shape: Shape, dtype: CategoricalDtype
    ) -> Self:
        """
        Analogous to np.empty(shape, dtype=dtype)

        Parameters
        ----------
        shape : tuple[int]
        dtype : CategoricalDtype
        """
        arr = cls._from_sequence([], dtype=dtype)

        # We have to use np.zeros instead of np.empty otherwise the resulting
        #  ndarray may contain codes not supported by this dtype, in which
        #  case repr(result) could segfault.
        backing = np.zeros(shape, dtype=arr._ndarray.dtype)

        return arr._from_backing_data(backing)

    def _internal_get_values(self) -> ArrayLike:
        """
        Return the values.

        For internal compatibility with pandas formatting.

        Returns
        -------
        np.ndarray or ExtensionArray
            A numpy array or ExtensionArray of the same dtype as
            categorical.categories.dtype.
        """
        # if we are a datetime and period index, return Index to keep metadata
        if needs_i8_conversion(self.categories.dtype):
            return self.categories.take(self._codes, fill_value=NaT)._values
        elif is_integer_dtype(self.categories.dtype) and -1 in self._codes:
            return (
                self.categories.astype("object")
                .take(self._codes, fill_value=np.nan)
                ._values
            )
        return np.array(self)

    def check_for_ordered(self, op) -> None:
        """assert that we are ordered"""
        if not self.ordered:
            raise TypeError(
                f"Categorical is not ordered for operation {op}\n"
                "you can use .as_ordered() to change the "
                "Categorical to an ordered one\n"
            )

    def argsort(
        self, *, ascending: bool = True, kind: SortKind = "quicksort", **kwargs
    ):
        """
        Return the indices that would sort the Categorical.

        Missing values are sorted at the end.

        Parameters
        ----------
        ascending : bool, default True
            Whether the indices should result in an ascending
            or descending sort.
        kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
            Sorting algorithm.
        **kwargs:
            passed through to :func:`numpy.argsort`.

        Returns
        -------
        np.ndarray[np.intp]

        See Also
        --------
        numpy.ndarray.argsort

        Notes
        -----
        While an ordering is applied to the category values, arg-sorting
        in this context refers more to organizing and grouping together
        based on matching category values. Thus, this function can be
        called on an unordered Categorical instance unlike the functions
        'Categorical.min' and 'Categorical.max'.

        Examples
        --------
        >>> pd.Categorical(['b', 'b', 'a', 'c']).argsort()
        array([2, 0, 1, 3])

        >>> cat = pd.Categorical(['b', 'b', 'a', 'c'],
        ...                      categories=['c', 'b', 'a'],
        ...                      ordered=True)
        >>> cat.argsort()
        array([3, 0, 1, 2])

        Missing values are placed at the end

        >>> cat = pd.Categorical([2, None, 1])
        >>> cat.argsort()
        array([2, 0, 1])
        """
        return super().argsort(ascending=ascending, kind=kind, **kwargs)

    @overload
    def sort_values(
        self,
        *,
        inplace: Literal[False] = ...,
        ascending: bool = ...,
        na_position: str = ...,
    ) -> Self:
        ...

    @overload
    def sort_values(
        self, *, inplace: Literal[True], ascending: bool = ..., na_position: str = ...
    ) -> None:
        ...

    def sort_values(
        self,
        *,
        inplace: bool = False,
        ascending: bool = True,
        na_position: str = "last",
    ) -> Self | None:
        """
        Sort the Categorical by category value returning a new
        Categorical by default.

        While an ordering is applied to the category values, sorting in this
        context refers more to organizing and grouping together based on
        matching category values. Thus, this function can be called on an
        unordered Categorical instance unlike the functions 'Categorical.min'
        and 'Categorical.max'.

        Parameters
        ----------
        inplace : bool, default False
            Do operation in place.
        ascending : bool, default True
            Order ascending. Passing False orders descending. The
            ordering parameter provides the method by which the
            category values are organized.
        na_position : {'first', 'last'} (optional, default='last')
            'first' puts NaNs at the beginning
            'last' puts NaNs at the end

        Returns
        -------
        Categorical or None

        See Also
        --------
        Categorical.sort
        Series.sort_values

        Examples
        --------
        >>> c = pd.Categorical([1, 2, 2, 1, 5])
        >>> c
        [1, 2, 2, 1, 5]
        Categories (3, int64): [1, 2, 5]
        >>> c.sort_values()
        [1, 1, 2, 2, 5]
        Categories (3, int64): [1, 2, 5]
        >>> c.sort_values(ascending=False)
        [5, 2, 2, 1, 1]
        Categories (3, int64): [1, 2, 5]

        >>> c = pd.Categorical([1, 2, 2, 1, 5])

        'sort_values' behaviour with NaNs. Note that 'na_position'
        is independent of the 'ascending' parameter:

        >>> c = pd.Categorical([np.nan, 2, 2, np.nan, 5])
        >>> c
        [NaN, 2, 2, NaN, 5]
        Categories (2, int64): [2, 5]
        >>> c.sort_values()
        [2, 2, 5, NaN, NaN]
        Categories (2, int64): [2, 5]
        >>> c.sort_values(ascending=False)
        [5, 2, 2, NaN, NaN]
        Categories (2, int64): [2, 5]
        >>> c.sort_values(na_position='first')
        [NaN, NaN, 2, 2, 5]
        Categories (2, int64): [2, 5]
        >>> c.sort_values(ascending=False, na_position='first')
        [NaN, NaN, 5, 2, 2]
        Categories (2, int64): [2, 5]
        """
        inplace = validate_bool_kwarg(inplace, "inplace")
        if na_position not in ["last", "first"]:
            raise ValueError(f"invalid na_position: {repr(na_position)}")

        sorted_idx = nargsort(self, ascending=ascending, na_position=na_position)

        if not inplace:
            codes = self._codes[sorted_idx]
            return self._from_backing_data(codes)
        self._codes[:] = self._codes[sorted_idx]
        return None

    def _rank(
        self,
        *,
        axis: AxisInt = 0,
        method: str = "average",
        na_option: str = "keep",
        ascending: bool = True,
        pct: bool = False,
    ):
        """
        See Series.rank.__doc__.
        """
        if axis != 0:
            raise NotImplementedError
        vff = self._values_for_rank()
        return algorithms.rank(
            vff,
            axis=axis,
            method=method,
            na_option=na_option,
            ascending=ascending,
            pct=pct,
        )

    def _values_for_rank(self) -> np.ndarray:
        """
        For correctly ranking ordered categorical data. See GH#15420

        Ordered categorical data should be ranked on the basis of
        codes with -1 translated to NaN.

        Returns
        -------
        numpy.array

        """
        from pandas import Series

        if self.ordered:
            values = self.codes
            mask = values == -1
            if mask.any():
                values = values.astype("float64")
                values[mask] = np.nan
        elif is_any_real_numeric_dtype(self.categories.dtype):
            values = np.array(self)
        else:
            #  reorder the categories (so rank can use the float codes)
            #  instead of passing an object array to rank
            values = np.array(
                self.rename_categories(
                    Series(self.categories, copy=False).rank().values
                )
            )
        return values

    def _hash_pandas_object(
        self, *, encoding: str, hash_key: str, categorize: bool
    ) -> npt.NDArray[np.uint64]:
        """
        Hash a Categorical by hashing its categories, and then mapping the codes
        to the hashes.

        Parameters
        ----------
        encoding : str
        hash_key : str
        categorize : bool
            Ignored for Categorical.

        Returns
        -------
        np.ndarray[uint64]
        """
        # Note we ignore categorize, as we are already Categorical.
        from pandas.core.util.hashing import hash_array

        # Convert ExtensionArrays to ndarrays
        values = np.asarray(self.categories._values)
        hashed = hash_array(values, encoding, hash_key, categorize=False)

        # we have uint64, as we don't directly support missing values
        # we don't want to use take_nd which will coerce to float
        # instead, directly construct the result with a
        # max(np.uint64) as the missing value indicator
        #
        # TODO: GH#15362

        mask = self.isna()
        if len(hashed):
            result = hashed.take(self._codes)
        else:
            result = np.zeros(len(mask), dtype="uint64")

        if mask.any():
            result[mask] = lib.u8max

        return result

    # ------------------------------------------------------------------
    # NDArrayBackedExtensionArray compat

    @property
    def _codes(self) -> np.ndarray:
        return self._ndarray

    def _box_func(self, i: int):
        if i == -1:
            return np.nan
        return self.categories[i]

    def _unbox_scalar(self, key) -> int:
        # searchsorted is very performance sensitive. By converting codes
        # to same dtype as self.codes, we get much faster performance.
        code = self.categories.get_loc(key)
        code = self._ndarray.dtype.type(code)
        return code

    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator:
        """
        Returns an Iterator over the values of this Categorical.
        """
        if self.ndim == 1:
            return iter(self._internal_get_values().tolist())
        else:
            return (self[n] for n in range(len(self)))

    def __contains__(self, key) -> bool:
        """
        Returns True if `key` is in this Categorical.
        """
        # if key is a NaN, check if any NaN is in self.
        if is_valid_na_for_dtype(key, self.categories.dtype):
            return bool(self.isna().any())

        return contains(self, key, container=self._codes)

    # ------------------------------------------------------------------
    # Rendering Methods

    def _formatter(self, boxed: bool = False):
        # Returning None here will cause format_array to do inference.
        return None

    def _repr_categories(self) -> list[str]:
        """
        return the base repr for the categories
        """
        max_categories = (
            10
            if get_option("display.max_categories") == 0
            else get_option("display.max_categories")
        )
        from pandas.io.formats import format as fmt

        formatter = None
        if self.categories.dtype == "str":
            # the extension array formatter defaults to boxed=True in format_array
            # override here to boxed=False to be consistent with QUOTE_NONNUMERIC
            formatter = cast(ExtensionArray, self.categories._values)._formatter(
                boxed=False
            )

        format_array = partial(
            fmt.format_array, formatter=formatter, quoting=QUOTE_NONNUMERIC
        )
        if len(self.categories) > max_categories:
            num = max_categories // 2
            head = format_array(self.categories[:num]._values)
            tail = format_array(self.categories[-num:]._values)
            category_strs = head + ["..."] + tail
        else:
            category_strs = format_array(self.categories._values)

        # Strip all leading spaces, which format_array adds for columns...
        category_strs = [x.strip() for x in category_strs]
        return category_strs

    def _get_repr_footer(self) -> str:
        """
        Returns a string representation of the footer.
        """
        category_strs = self._repr_categories()
        dtype = str(self.categories.dtype)
        levheader = f"Categories ({len(self.categories)}, {dtype}): "
        width, _ = get_terminal_size()
        max_width = get_option("display.width") or width
        if console.in_ipython_frontend():
            # 0 = no breaks
            max_width = 0
        levstring = ""
        start = True
        cur_col_len = len(levheader)  # header
        sep_len, sep = (3, " < ") if self.ordered else (2, ", ")
        linesep = f"{sep.rstrip()}\n"  # remove whitespace
        for val in category_strs:
            if max_width != 0 and cur_col_len + sep_len + len(val) > max_width:
                levstring += linesep + (" " * (len(levheader) + 1))
                cur_col_len = len(levheader) + 1  # header + a whitespace
            elif not start:
                levstring += sep
                cur_col_len += len(val)
            levstring += val
            start = False
        # replace to simple save space by
        return f"{levheader}[{levstring.replace(' < ... < ', ' ... ')}]"

    def _get_values_repr(self) -> str:
        from pandas.io.formats import format as fmt

        assert len(self) > 0

        vals = self._internal_get_values()
        fmt_values = fmt.format_array(
            vals,
            None,
            float_format=None,
            na_rep="NaN",
            quoting=QUOTE_NONNUMERIC,
        )

        fmt_values = [i.strip() for i in fmt_values]
        joined = ", ".join(fmt_values)
        result = "[" + joined + "]"
        return result

    def __repr__(self) -> str:
        """
        String representation.
        """
        footer = self._get_repr_footer()
        length = len(self)
        max_len = 10
        if length > max_len:
            # In long cases we do not display all entries, so we add Length
            #  information to the __repr__.
            num = max_len // 2
            head = self[:num]._get_values_repr()
            tail = self[-(max_len - num) :]._get_values_repr()
            body = f"{head[:-1]}, ..., {tail[1:]}"
            length_info = f"Length: {len(self)}"
            result = f"{body}\n{length_info}\n{footer}"
        elif length > 0:
            body = self._get_values_repr()
            result = f"{body}\n{footer}"
        else:
            # In the empty case we use a comma instead of newline to get
            #  a more compact __repr__
            body = "[]"
            result = f"{body}, {footer}"

        return result

    # ------------------------------------------------------------------

    def _validate_listlike(self, value):
        # NB: here we assume scalar-like tuples have already been excluded
        value = extract_array(value, extract_numpy=True)

        # require identical categories set
        if isinstance(value, Categorical):
            if self.dtype != value.dtype:
                raise TypeError(
                    "Cannot set a Categorical with another, "
                    "without identical categories"
                )
            # dtype equality implies categories_match_up_to_permutation
            value = self._encode_with_my_categories(value)
            return value._codes

        from pandas import Index

        # tupleize_cols=False for e.g. test_fillna_iterable_category GH#41914
        to_add = Index._with_infer(value, tupleize_cols=False).difference(
            self.categories
        )

        # no assignments of values not in categories, but it's always ok to set
        # something to np.nan
        if len(to_add) and not isna(to_add).all():
            raise TypeError(
                "Cannot setitem on a Categorical with a new "
                "category, set the categories first"
            )

        codes = self.categories.get_indexer(value)
        return codes.astype(self._ndarray.dtype, copy=False)

    def _reverse_indexer(self) -> dict[Hashable, npt.NDArray[np.intp]]:
        """
        Compute the inverse of a categorical, returning
        a dict of categories -> indexers.

        *This is an internal function*

        Returns
        -------
        Dict[Hashable, np.ndarray[np.intp]]
            dict of categories -> indexers

        Examples
        --------
        >>> c = pd.Categorical(list('aabca'))
        >>> c
        ['a', 'a', 'b', 'c', 'a']
        Categories (3, object): ['a', 'b', 'c']
        >>> c.categories
        Index(['a', 'b', 'c'], dtype='object')
        >>> c.codes
        array([0, 0, 1, 2, 0], dtype=int8)
        >>> c._reverse_indexer()
        {'a': array([0, 1, 4]), 'b': array([2]), 'c': array([3])}

        """
        categories = self.categories
        r, counts = libalgos.groupsort_indexer(
            ensure_platform_int(self.codes), categories.size
        )
        counts = ensure_int64(counts).cumsum()
        _result = (r[start:end] for start, end in zip(counts, counts[1:]))
        return dict(zip(categories, _result))

    # ------------------------------------------------------------------
    # Reductions

    def _reduce(
        self, name: str, *, skipna: bool = True, keepdims: bool = False, **kwargs
    ):
        result = super()._reduce(name, skipna=skipna, keepdims=keepdims, **kwargs)
        if name in ["argmax", "argmin"]:
            # don't wrap in Categorical!
            return result
        if keepdims:
            return type(self)(result, dtype=self.dtype)
        else:
            return result

    def min(self, *, skipna: bool = True, **kwargs):
        """
        The minimum value of the object.

        Only ordered `Categoricals` have a minimum!

        Raises
        ------
        TypeError
            If the `Categorical` is not `ordered`.

        Returns
        -------
        min : the minimum of this `Categorical`, NA value if empty
        """
        nv.validate_minmax_axis(kwargs.get("axis", 0))
        nv.validate_min((), kwargs)
        self.check_for_ordered("min")

        if not len(self._codes):
            return self.dtype.na_value

        good = self._codes != -1
        if not good.all():
            if skipna and good.any():
                pointer = self._codes[good].min()
            else:
                return np.nan
        else:
            pointer = self._codes.min()
        return self._wrap_reduction_result(None, pointer)

    def max(self, *, skipna: bool = True, **kwargs):
        """
        The maximum value of the object.

        Only ordered `Categoricals` have a maximum!

        Raises
        ------
        TypeError
            If the `Categorical` is not `ordered`.

        Returns
        -------
        max : the maximum of this `Categorical`, NA if array is empty
        """
        nv.validate_minmax_axis(kwargs.get("axis", 0))
        nv.validate_max((), kwargs)
        self.check_for_ordered("max")

        if not len(self._codes):
            return self.dtype.na_value

        good = self._codes != -1
        if not good.all():
            if skipna and good.any():
                pointer = self._codes[good].max()
            else:
                return np.nan
        else:
            pointer = self._codes.max()
        return self._wrap_reduction_result(None, pointer)

    def _mode(self, dropna: bool = True) -> Categorical:
        codes = self._codes
        mask = None
        if dropna:
            mask = self.isna()

        res_codes = algorithms.mode(codes, mask=mask)
        res_codes = cast(np.ndarray, res_codes)
        assert res_codes.dtype == codes.dtype
        res = self._from_backing_data(res_codes)
        return res

    # ------------------------------------------------------------------
    # ExtensionArray Interface

    def unique(self) -> Self:
        """
        Return the ``Categorical`` which ``categories`` and ``codes`` are
        unique.

        .. versionchanged:: 1.3.0

            Previously, unused categories were dropped from the new categories.

        Returns
        -------
        Categorical

        See Also
        --------
        pandas.unique
        CategoricalIndex.unique
        Series.unique : Return unique values of Series object.

        Examples
        --------
        >>> pd.Categorical(list("baabc")).unique()
        ['b', 'a', 'c']
        Categories (3, object): ['a', 'b', 'c']
        >>> pd.Categorical(list("baab"), categories=list("abc"), ordered=True).unique()
        ['b', 'a']
        Categories (3, object): ['a' < 'b' < 'c']
        """
        # pylint: disable=useless-parent-delegation
        return super().unique()

    def equals(self, other: object) -> bool:
        """
        Returns True if categorical arrays are equal.

        Parameters
        ----------
        other : `Categorical`

        Returns
        -------
        bool
        """
        if not isinstance(other, Categorical):
            return False
        elif self._categories_match_up_to_permutation(other):
            other = self._encode_with_my_categories(other)
            return np.array_equal(self._codes, other._codes)
        return False

    @classmethod
    def _concat_same_type(cls, to_concat: Sequence[Self], axis: AxisInt = 0) -> Self:
        from pandas.core.dtypes.concat import union_categoricals

        first = to_concat[0]
        if axis >= first.ndim:
            raise ValueError(
                f"axis {axis} is out of bounds for array of dimension {first.ndim}"
            )

        if axis == 1:
            # Flatten, concatenate then reshape
            if not all(x.ndim == 2 for x in to_concat):
                raise ValueError

            # pass correctly-shaped to union_categoricals
            tc_flat = []
            for obj in to_concat:
                tc_flat.extend([obj[:, i] for i in range(obj.shape[1])])

            res_flat = cls._concat_same_type(tc_flat, axis=0)

            result = res_flat.reshape(len(first), -1, order="F")
            return result

        result = union_categoricals(to_concat)
        return result

    # ------------------------------------------------------------------

    def _encode_with_my_categories(self, other: Categorical) -> Categorical:
        """
        Re-encode another categorical using this Categorical's categories.

        Notes
        -----
        This assumes we have already checked
        self._categories_match_up_to_permutation(other).
        """
        # Indexing on codes is more efficient if categories are the same,
        #  so we can apply some optimizations based on the degree of
        #  dtype-matching.
        codes = recode_for_categories(
            other.codes, other.categories, self.categories, copy=False
        )
        return self._from_backing_data(codes)

    def _categories_match_up_to_permutation(self, other: Categorical) -> bool:
        """
        Returns True if categoricals are the same dtype
          same categories, and same ordered

        Parameters
        ----------
        other : Categorical

        Returns
        -------
        bool
        """
        return hash(self.dtype) == hash(other.dtype)

    def describe(self) -> DataFrame:
        """
        Describes this Categorical

        Returns
        -------
        description: `DataFrame`
            A dataframe with frequency and counts by category.
        """
        counts = self.value_counts(dropna=False)
        freqs = counts / counts.sum()

        from pandas import Index
        from pandas.core.reshape.concat import concat

        result = concat([counts, freqs], axis=1)
        result.columns = Index(["counts", "freqs"])
        result.index.name = "categories"

        return result

    def isin(self, values: ArrayLike) -> npt.NDArray[np.bool_]:
        """
        Check whether `values` are contained in Categorical.

        Return a boolean NumPy Array showing whether each element in
        the Categorical matches an element in the passed sequence of
        `values` exactly.

        Parameters
        ----------
        values : np.ndarray or ExtensionArray
            The sequence of values to test. Passing in a single string will
            raise a ``TypeError``. Instead, turn a single string into a
            list of one element.

        Returns
        -------
        np.ndarray[bool]

        Raises
        ------
        TypeError
          * If `values` is not a set or list-like

        See Also
        --------
        pandas.Series.isin : Equivalent method on Series.

        Examples
        --------
        >>> s = pd.Categorical(['lama', 'cow', 'lama', 'beetle', 'lama',
        ...                'hippo'])
        >>> s.isin(['cow', 'lama'])
        array([ True,  True,  True, False,  True, False])

        Passing a single string as ``s.isin('lama')`` will raise an error. Use
        a list of one element instead:

        >>> s.isin(['lama'])
        array([ True, False,  True, False,  True, False])
        """
        null_mask = np.asarray(isna(values))
        code_values = self.categories.get_indexer_for(values)
        code_values = code_values[null_mask | (code_values >= 0)]
        return algorithms.isin(self.codes, code_values)

    def _replace(self, *, to_replace, value, inplace: bool = False):
        from pandas import Index

        orig_dtype = self.dtype

        inplace = validate_bool_kwarg(inplace, "inplace")
        cat = self if inplace else self.copy()

        mask = isna(np.asarray(value))
        if mask.any():
            removals = np.asarray(to_replace)[mask]
            removals = cat.categories[cat.categories.isin(removals)]
            new_cat = cat.remove_categories(removals)
            NDArrayBacked.__init__(cat, new_cat.codes, new_cat.dtype)

        ser = cat.categories.to_series()
        ser = ser.replace(to_replace=to_replace, value=value)

        all_values = Index(ser)

        # GH51016: maintain order of existing categories
        idxr = cat.categories.get_indexer_for(all_values)
        locs = np.arange(len(ser))
        locs = np.where(idxr == -1, locs, idxr)
        locs = locs.argsort()

        new_categories = ser.take(locs)
        new_categories = new_categories.drop_duplicates(keep="first")
        new_categories = Index(new_categories)
        new_codes = recode_for_categories(
            cat._codes, all_values, new_categories, copy=False
        )
        new_dtype = CategoricalDtype(new_categories, ordered=self.dtype.ordered)
        NDArrayBacked.__init__(cat, new_codes, new_dtype)

        if new_dtype != orig_dtype:
            warnings.warn(
                # GH#55147
                "The behavior of Series.replace (and DataFrame.replace) with "
                "CategoricalDtype is deprecated. In a future version, replace "
                "will only be used for cases that preserve the categories. "
                "To change the categories, use ser.cat.rename_categories "
                "instead.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )
        if not inplace:
            return cat

    # ------------------------------------------------------------------------
    # String methods interface
    def _str_map(
        self, f, na_value=lib.no_default, dtype=np.dtype("object"), convert: bool = True
    ):
        # Optimization to apply the callable `f` to the categories once
        # and rebuild the result by `take`ing from the result with the codes.
        # Returns the same type as the object-dtype implementation though.
        categories = self.categories
        codes = self.codes
        if categories.dtype == "string":
            result = categories.array._str_map(f, na_value, dtype)  # type: ignore[attr-defined]
            if (
                categories.dtype.na_value is np.nan  # type: ignore[union-attr]
                and is_bool_dtype(dtype)
                and (na_value is lib.no_default or isna(na_value))
            ):
                # NaN propagates as False for functions with boolean return type
                na_value = False
        else:
            from pandas.core.arrays import NumpyExtensionArray

            result = NumpyExtensionArray(categories.to_numpy())._str_map(
                f, na_value, dtype
            )
        return take_nd(result, codes, fill_value=na_value)

    def _str_get_dummies(self, sep: str = "|"):
        # sep may not be in categories. Just bail on this.
        from pandas.core.arrays import NumpyExtensionArray

        return NumpyExtensionArray(self.to_numpy(str, na_value="NaN"))._str_get_dummies(
            sep
        )

    # ------------------------------------------------------------------------
    # GroupBy Methods

    def _groupby_op(
        self,
        *,
        how: str,
        has_dropped_na: bool,
        min_count: int,
        ngroups: int,
        ids: npt.NDArray[np.intp],
        **kwargs,
    ):
        from pandas.core.groupby.ops import WrappedCythonOp

        kind = WrappedCythonOp.get_kind_from_how(how)
        op = WrappedCythonOp(how=how, kind=kind, has_dropped_na=has_dropped_na)

        dtype = self.dtype
        if how in ["sum", "prod", "cumsum", "cumprod", "skew"]:
            raise TypeError(f"{dtype} type does not support {how} operations")
        if how in ["min", "max", "rank", "idxmin", "idxmax"] and not dtype.ordered:
            # raise TypeError instead of NotImplementedError to ensure we
            #  don't go down a group-by-group path, since in the empty-groups
            #  case that would fail to raise
            raise TypeError(f"Cannot perform {how} with non-ordered Categorical")
        if how not in [
            "rank",
            "any",
            "all",
            "first",
            "last",
            "min",
            "max",
            "idxmin",
            "idxmax",
        ]:
            if kind == "transform":
                raise TypeError(f"{dtype} type does not support {how} operations")
            raise TypeError(f"{dtype} dtype does not support aggregation '{how}'")

        result_mask = None
        mask = self.isna()
        if how == "rank":
            assert self.ordered  # checked earlier
            npvalues = self._ndarray
        elif how in ["first", "last", "min", "max", "idxmin", "idxmax"]:
            npvalues = self._ndarray
            result_mask = np.zeros(ngroups, dtype=bool)
        else:
            # any/all
            npvalues = self.astype(bool)

        res_values = op._cython_op_ndim_compat(
            npvalues,
            min_count=min_count,
            ngroups=ngroups,
            comp_ids=ids,
            mask=mask,
            result_mask=result_mask,
            **kwargs,
        )

        if how in op.cast_blocklist:
            return res_values
        elif how in ["first", "last", "min", "max"]:
            res_values[result_mask == 1] = -1
        return self._from_backing_data(res_values)


# The Series.cat accessor


@delegate_names(
    delegate=Categorical, accessors=["categories", "ordered"], typ="property"
)
@delegate_names(
    delegate=Categorical,
    accessors=[
        "rename_categories",
        "reorder_categories",
        "add_categories",
        "remove_categories",
        "remove_unused_categories",
        "set_categories",
        "as_ordered",
        "as_unordered",
    ],
    typ="method",
)
class CategoricalAccessor(PandasDelegate, PandasObject, NoNewAttributesMixin):
    """
    Accessor object for categorical properties of the Series values.

    Parameters
    ----------
    data : Series or CategoricalIndex

    Examples
    --------
    >>> s = pd.Series(list("abbccc")).astype("category")
    >>> s
    0    a
    1    b
    2    b
    3    c
    4    c
    5    c
    dtype: category
    Categories (3, object): ['a', 'b', 'c']

    >>> s.cat.categories
    Index(['a', 'b', 'c'], dtype='object')

    >>> s.cat.rename_categories(list("cba"))
    0    c
    1    b
    2    b
    3    a
    4    a
    5    a
    dtype: category
    Categories (3, object): ['c', 'b', 'a']

    >>> s.cat.reorder_categories(list("cba"))
    0    a
    1    b
    2    b
    3    c
    4    c
    5    c
    dtype: category
    Categories (3, object): ['c', 'b', 'a']

    >>> s.cat.add_categories(["d", "e"])
    0    a
    1    b
    2    b
    3    c
    4    c
    5    c
    dtype: category
    Categories (5, object): ['a', 'b', 'c', 'd', 'e']

    >>> s.cat.remove_categories(["a", "c"])
    0    NaN
    1      b
    2      b
    3    NaN
    4    NaN
    5    NaN
    dtype: category
    Categories (1, object): ['b']

    >>> s1 = s.cat.add_categories(["d", "e"])
    >>> s1.cat.remove_unused_categories()
    0    a
    1    b
    2    b
    3    c
    4    c
    5    c
    dtype: category
    Categories (3, object): ['a', 'b', 'c']

    >>> s.cat.set_categories(list("abcde"))
    0    a
    1    b
    2    b
    3    c
    4    c
    5    c
    dtype: category
    Categories (5, object): ['a', 'b', 'c', 'd', 'e']

    >>> s.cat.as_ordered()
    0    a
    1    b
    2    b
    3    c
    4    c
    5    c
    dtype: category
    Categories (3, object): ['a' < 'b' < 'c']

    >>> s.cat.as_unordered()
    0    a
    1    b
    2    b
    3    c
    4    c
    5    c
    dtype: category
    Categories (3, object): ['a', 'b', 'c']
    """

    def __init__(self, data) -> None:
        self._validate(data)
        self._parent = data.values
        self._index = data.index
        self._name = data.name
        self._freeze()

    @staticmethod
    def _validate(data):
        if not isinstance(data.dtype, CategoricalDtype):
            raise AttributeError("Can only use .cat accessor with a 'category' dtype")

    def _delegate_property_get(self, name: str):
        return getattr(self._parent, name)

    # error: Signature of "_delegate_property_set" incompatible with supertype
    # "PandasDelegate"
    def _delegate_property_set(self, name: str, new_values):  # type: ignore[override]
        return setattr(self._parent, name, new_values)

    @property
    def codes(self) -> Series:
        """
        Return Series of codes as well as the index.

        Examples
        --------
        >>> raw_cate = pd.Categorical(["a", "b", "c", "a"], categories=["a", "b"])
        >>> ser = pd.Series(raw_cate)
        >>> ser.cat.codes
        0   0
        1   1
        2  -1
        3   0
        dtype: int8
        """
        from pandas import Series

        return Series(self._parent.codes, index=self._index)

    def _delegate_method(self, name: str, *args, **kwargs):
        from pandas import Series

        method = getattr(self._parent, name)
        res = method(*args, **kwargs)
        if res is not None:
            return Series(res, index=self._index, name=self._name)


# utility routines


def _get_codes_for_values(
    values: Index | Series | ExtensionArray | np.ndarray,
    categories: Index,
) -> np.ndarray:
    """
    utility routine to turn values into codes given the specified categories

    If `values` is known to be a Categorical, use recode_for_categories instead.
    """
    codes = categories.get_indexer_for(values)
    return coerce_indexer_dtype(codes, categories)


def recode_for_categories(
    codes: np.ndarray, old_categories, new_categories, copy: bool = True
) -> np.ndarray:
    """
    Convert a set of codes for to a new set of categories

    Parameters
    ----------
    codes : np.ndarray
    old_categories, new_categories : Index
    copy: bool, default True
        Whether to copy if the codes are unchanged.

    Returns
    -------
    new_codes : np.ndarray[np.int64]

    Examples
    --------
    >>> old_cat = pd.Index(['b', 'a', 'c'])
    >>> new_cat = pd.Index(['a', 'b'])
    >>> codes = np.array([0, 1, 1, 2])
    >>> recode_for_categories(codes, old_cat, new_cat)
    array([ 1,  0,  0, -1], dtype=int8)
    """
    if len(old_categories) == 0:
        # All null anyway, so just retain the nulls
        if copy:
            return codes.copy()
        return codes
    elif new_categories.equals(old_categories):
        # Same categories, so no need to actually recode
        if copy:
            return codes.copy()
        return codes

    indexer = coerce_indexer_dtype(
        new_categories.get_indexer_for(old_categories), new_categories
    )
    new_codes = take_nd(indexer, codes, fill_value=-1)
    return new_codes


def factorize_from_iterable(values) -> tuple[np.ndarray, Index]:
    """
    Factorize an input `values` into `categories` and `codes`. Preserves
    categorical dtype in `categories`.

    Parameters
    ----------
    values : list-like

    Returns
    -------
    codes : ndarray
    categories : Index
        If `values` has a categorical dtype, then `categories` is
        a CategoricalIndex keeping the categories and order of `values`.
    """
    from pandas import CategoricalIndex

    if not is_list_like(values):
        raise TypeError("Input must be list-like")

    categories: Index

    vdtype = getattr(values, "dtype", None)
    if isinstance(vdtype, CategoricalDtype):
        values = extract_array(values)
        # The Categorical we want to build has the same categories
        # as values but its codes are by def [0, ..., len(n_categories) - 1]
        cat_codes = np.arange(len(values.categories), dtype=values.codes.dtype)
        cat = Categorical.from_codes(cat_codes, dtype=values.dtype, validate=False)

        categories = CategoricalIndex(cat)
        codes = values.codes
    else:
        # The value of ordered is irrelevant since we don't use cat as such,
        # but only the resulting categories, the order of which is independent
        # from ordered. Set ordered to False as default. See GH #15457
        cat = Categorical(values, ordered=False)
        categories = cat.categories
        codes = cat.codes
    return codes, categories


def factorize_from_iterables(iterables) -> tuple[list[np.ndarray], list[Index]]:
    """
    A higher-level wrapper over `factorize_from_iterable`.

    Parameters
    ----------
    iterables : list-like of list-likes

    Returns
    -------
    codes : list of ndarrays
    categories : list of Indexes

    Notes
    -----
    See `factorize_from_iterable` for more info.
    """
    if len(iterables) == 0:
        # For consistency, it should return two empty lists.
        return [], []

    codes, categories = zip(*(factorize_from_iterable(it) for it in iterables))
    return list(codes), list(categories)
<|endoftext|>from __future__ import annotations

from functools import partial
import re
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
)

import numpy as np

from pandas._libs import lib
from pandas.compat import (
    pa_version_under10p1,
    pa_version_under11p0,
    pa_version_under13p0,
    pa_version_under17p0,
    pa_version_under21p0,
)

if not pa_version_under10p1:
    import pyarrow as pa
    import pyarrow.compute as pc

if TYPE_CHECKING:
    from collections.abc import Callable

    from pandas._typing import (
        Scalar,
        Self,
    )


class ArrowStringArrayMixin:
    _pa_array: pa.ChunkedArray

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def _convert_bool_result(self, result, na=lib.no_default, method_name=None):
        # Convert a bool-dtype result to the appropriate result type
        raise NotImplementedError

    def _convert_int_result(self, result):
        # Convert an integer-dtype result to the appropriate result type
        raise NotImplementedError

    def _apply_elementwise(self, func: Callable) -> list[list[Any]]:
        raise NotImplementedError

    def _str_len(self):
        result = pc.utf8_length(self._pa_array)
        return self._convert_int_result(result)

    def _str_lower(self) -> Self:
        return type(self)(pc.utf8_lower(self._pa_array))

    def _str_upper(self) -> Self:
        return type(self)(pc.utf8_upper(self._pa_array))

    def _str_strip(self, to_strip=None) -> Self:
        if to_strip is None:
            result = pc.utf8_trim_whitespace(self._pa_array)
        else:
            result = pc.utf8_trim(self._pa_array, characters=to_strip)
        return type(self)(result)

    def _str_lstrip(self, to_strip=None) -> Self:
        if to_strip is None:
            result = pc.utf8_ltrim_whitespace(self._pa_array)
        else:
            result = pc.utf8_ltrim(self._pa_array, characters=to_strip)
        return type(self)(result)

    def _str_rstrip(self, to_strip=None) -> Self:
        if to_strip is None:
            result = pc.utf8_rtrim_whitespace(self._pa_array)
        else:
            result = pc.utf8_rtrim(self._pa_array, characters=to_strip)
        return type(self)(result)

    def _str_pad(
        self,
        width: int,
        side: Literal["left", "right", "both"] = "left",
        fillchar: str = " ",
    ):
        if side == "left":
            pa_pad = pc.utf8_lpad
        elif side == "right":
            pa_pad = pc.utf8_rpad
        elif side == "both":
            if pa_version_under17p0:
                # GH#59624 fall back to object dtype
                from pandas import array as pd_array

                obj_arr = self.astype(object, copy=False)  # type: ignore[attr-defined]
                obj = pd_array(obj_arr, dtype=object)
                result = obj._str_pad(width, side, fillchar)  # type: ignore[attr-defined]
                return type(self)._from_sequence(result, dtype=self.dtype)  # type: ignore[attr-defined]
            else:
                # GH#54792
                # https://github.com/apache/arrow/issues/15053#issuecomment-2317032347
                lean_left = (width % 2) == 0
                pa_pad = partial(pc.utf8_center, lean_left_on_odd_padding=lean_left)
        else:
            raise ValueError(
                f"Invalid side: {side}. Side must be one of 'left', 'right', 'both'"
            )
        return type(self)(pa_pad(self._pa_array, width=width, padding=fillchar))

    def _str_get(self, i: int):
        lengths = pc.utf8_length(self._pa_array)
        if i >= 0:
            out_of_bounds = pc.greater_equal(i, lengths)
            start = i
            stop = i + 1
            step = 1
        else:
            out_of_bounds = pc.greater(-i, lengths)
            start = i
            stop = i - 1
            step = -1
        not_out_of_bounds = pc.invert(out_of_bounds.fill_null(True))
        selected = pc.utf8_slice_codeunits(
            self._pa_array, start=start, stop=stop, step=step
        )
        null_value = pa.scalar(None, type=self._pa_array.type)
        result = pc.if_else(not_out_of_bounds, selected, null_value)
        return type(self)(result)

    def _str_slice(
        self, start: int | None = None, stop: int | None = None, step: int | None = None
    ):
        if pa_version_under11p0:
            # GH#59724
            result = self._apply_elementwise(lambda val: val[start:stop:step])
            return type(self)(pa.chunked_array(result, type=self._pa_array.type))
        if start is None:
            if step is not None and step < 0:
                # GH#59710
                start = -1
            else:
                start = 0
        if step is None:
            step = 1
        return type(self)(
            pc.utf8_slice_codeunits(self._pa_array, start=start, stop=stop, step=step)
        )

    def _str_slice_replace(
        self, start: int | None = None, stop: int | None = None, repl: str | None = None
    ):
        if repl is None:
            repl = ""
        if start is None:
            start = 0
        if stop is None:
            stop = np.iinfo(np.int64).max
        return type(self)(pc.utf8_replace_slice(self._pa_array, start, stop, repl))

    def _str_replace(
        self,
        pat: str | re.Pattern,
        repl: str | Callable,
        n: int = -1,
        case: bool = True,
        flags: int = 0,
        regex: bool = True,
    ) -> Self:
        if (
            isinstance(pat, re.Pattern)
            or callable(repl)
            or not case
            or flags
            or (
                isinstance(repl, str)
                and (r"\g<" in repl or re.search(r"\\\d", repl) is not None)
            )
        ):
            raise NotImplementedError(
                "replace is not supported with a re.Pattern, callable repl, "
                "case=False, flags!=0, or when the replacement string contains "
                "named group references (\\g<...>, \\d+)"
            )

        func = pc.replace_substring_regex if regex else pc.replace_substring
        # https://github.com/apache/arrow/issues/39149
        # GH 56404, unexpected behavior with negative max_replacements with pyarrow.
        pa_max_replacements = None if n < 0 else n
        result = func(
            self._pa_array,
            pattern=pat,
            replacement=repl,
            max_replacements=pa_max_replacements,
        )
        return type(self)(result)

    def _str_capitalize(self) -> Self:
        return type(self)(pc.utf8_capitalize(self._pa_array))

    def _str_title(self):
        return type(self)(pc.utf8_title(self._pa_array))

    def _str_swapcase(self):
        return type(self)(pc.utf8_swapcase(self._pa_array))

    def _str_removeprefix(self, prefix: str):
        if not pa_version_under13p0:
            starts_with = pc.starts_with(self._pa_array, pattern=prefix)
            removed = pc.utf8_slice_codeunits(self._pa_array, len(prefix))
            result = pc.if_else(starts_with, removed, self._pa_array)
            return type(self)(result)
        predicate = lambda val: val.removeprefix(prefix)
        result = self._apply_elementwise(predicate)
        return type(self)(pa.chunked_array(result))

    def _str_removesuffix(self, suffix: str):
        ends_with = pc.ends_with(self._pa_array, pattern=suffix)
        removed = pc.utf8_slice_codeunits(self._pa_array, 0, stop=-len(suffix))
        result = pc.if_else(ends_with, removed, self._pa_array)
        return type(self)(result)

    def _str_startswith(
        self, pat: str | tuple[str, ...], na: Scalar | lib.NoDefault = lib.no_default
    ):
        if isinstance(pat, str):
            result = pc.starts_with(self._pa_array, pattern=pat)
        else:
            if len(pat) == 0:
                # For empty tuple we return null for missing values and False
                #  for valid values.
                result = pc.if_else(pc.is_null(self._pa_array), None, False)
            else:
                result = pc.starts_with(self._pa_array, pattern=pat[0])

                for p in pat[1:]:
                    result = pc.or_(result, pc.starts_with(self._pa_array, pattern=p))
        return self._convert_bool_result(result, na=na, method_name="startswith")

    def _str_endswith(
        self, pat: str | tuple[str, ...], na: Scalar | lib.NoDefault = lib.no_default
    ):
        if isinstance(pat, str):
            result = pc.ends_with(self._pa_array, pattern=pat)
        else:
            if len(pat) == 0:
                # For empty tuple we return null for missing values and False
                #  for valid values.
                result = pc.if_else(pc.is_null(self._pa_array), None, False)
            else:
                result = pc.ends_with(self._pa_array, pattern=pat[0])

                for p in pat[1:]:
                    result = pc.or_(result, pc.ends_with(self._pa_array, pattern=p))
        return self._convert_bool_result(result, na=na, method_name="endswith")

    def _str_isalnum(self):
        result = pc.utf8_is_alnum(self._pa_array)
        return self._convert_bool_result(result)

    def _str_isalpha(self):
        result = pc.utf8_is_alpha(self._pa_array)
        return self._convert_bool_result(result)

    def _str_isdecimal(self):
        result = pc.utf8_is_decimal(self._pa_array)
        return self._convert_bool_result(result)

    def _str_isdigit(self):
        if pa_version_under21p0:
            # https://github.com/pandas-dev/pandas/issues/61466
            res_list = self._apply_elementwise(str.isdigit)
            return self._convert_bool_result(
                pa.chunked_array(res_list, type=pa.bool_())
            )
        result = pc.utf8_is_digit(self._pa_array)
        return self._convert_bool_result(result)

    def _str_islower(self):
        result = pc.utf8_is_lower(self._pa_array)
        return self._convert_bool_result(result)

    def _str_isnumeric(self):
        result = pc.utf8_is_numeric(self._pa_array)
        return self._convert_bool_result(result)

    def _str_isspace(self):
        result = pc.utf8_is_space(self._pa_array)
        return self._convert_bool_result(result)

    def _str_istitle(self):
        result = pc.utf8_is_title(self._pa_array)
        return self._convert_bool_result(result)

    def _str_isupper(self):
        result = pc.utf8_is_upper(self._pa_array)
        return self._convert_bool_result(result)

    def _str_contains(
        self,
        pat,
        case: bool = True,
        flags: int = 0,
        na: Scalar | lib.NoDefault = lib.no_default,
        regex: bool = True,
    ):
        if flags:
            raise NotImplementedError(f"contains not implemented with {flags=}")

        if regex:
            pa_contains = pc.match_substring_regex
        else:
            pa_contains = pc.match_substring
        result = pa_contains(self._pa_array, pat, ignore_case=not case)
        return self._convert_bool_result(result, na=na, method_name="contains")

    def _str_match(
        self,
        pat: str,
        case: bool = True,
        flags: int = 0,
        na: Scalar | lib.NoDefault = lib.no_default,
    ):
        if not pat.startswith("^"):
            pat = f"^({pat})"
        return self._str_contains(pat, case, flags, na, regex=True)

    def _str_fullmatch(
        self,
        pat: str,
        case: bool = True,
        flags: int = 0,
        na: Scalar | lib.NoDefault = lib.no_default,
    ):
        if (not pat.endswith("$") or pat.endswith("\\$")) and not pat.startswith("^"):
            pat = f"^({pat})$"
        elif not pat.endswith("$") or pat.endswith("\\$"):
            pat = f"^({pat[1:]})$"
        elif not pat.startswith("^"):
            pat = f"^({pat[0:-1]})$"
        return self._str_match(pat, case, flags, na)

    def _str_find(self, sub: str, start: int = 0, end: int | None = None):
        if (
            pa_version_under13p0
            and not (start != 0 and end is not None)
            and not (start == 0 and end is None)
        ):
            # GH#59562
            res_list = self._apply_elementwise(lambda val: val.find(sub, start, end))
            return self._convert_int_result(pa.chunked_array(res_list))

        if (start == 0 or start is None) and end is None:
            result = pc.find_substring(self._pa_array, sub)
        else:
            if sub == "":
                # GH#56792
                res_list = self._apply_elementwise(
                    lambda val: val.find(sub, start, end)
                )
                return self._convert_int_result(pa.chunked_array(res_list))
            if start is None:
                start_offset = 0
                start = 0
            elif start < 0:
                start_offset = pc.add(start, pc.utf8_length(self._pa_array))
                start_offset = pc.if_else(pc.less(start_offset, 0), 0, start_offset)
            else:
                start_offset = start
            slices = pc.utf8_slice_codeunits(self._pa_array, start, stop=end)
            result = pc.find_substring(slices, sub)
            found = pc.not_equal(result, pa.scalar(-1, type=result.type))
            offset_result = pc.add(result, start_offset)
            result = pc.if_else(found, offset_result, -1)
        return self._convert_int_result(result)
<|endoftext|>"""
Helper functions to generate range-like data for DatetimeArray
(and possibly TimedeltaArray/PeriodArray)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pandas._libs.lib import i8max
from pandas._libs.tslibs import (
    BaseOffset,
    OutOfBoundsDatetime,
    Timedelta,
    Timestamp,
    iNaT,
)

if TYPE_CHECKING:
    from pandas._typing import npt


def generate_regular_range(
    start: Timestamp | Timedelta | None,
    end: Timestamp | Timedelta | None,
    periods: int | None,
    freq: BaseOffset,
    unit: str = "ns",
) -> npt.NDArray[np.intp]:
    """
    Generate a range of dates or timestamps with the spans between dates
    described by the given `freq` DateOffset.

    Parameters
    ----------
    start : Timedelta, Timestamp or None
        First point of produced date range.
    end : Timedelta, Timestamp or None
        Last point of produced date range.
    periods : int or None
        Number of periods in produced date range.
    freq : Tick
        Describes space between dates in produced date range.
    unit : str, default "ns"
        The resolution the output is meant to represent.

    Returns
    -------
    ndarray[np.int64]
        Representing the given resolution.
    """
    istart = start._value if start is not None else None
    iend = end._value if end is not None else None
    freq.nanos  # raises if non-fixed frequency
    td = Timedelta(freq)
    b: int
    e: int
    try:
        td = td.as_unit(unit, round_ok=False)
    except ValueError as err:
        raise ValueError(
            f"freq={freq} is incompatible with unit={unit}. "
            "Use a lower freq or a higher unit instead."
        ) from err
    stride = int(td._value)

    if periods is None and istart is not None and iend is not None:
        b = istart
        # cannot just use e = Timestamp(end) + 1 because arange breaks when
        # stride is too large, see GH10887
        e = b + (iend - b) // stride * stride + stride // 2 + 1
    elif istart is not None and periods is not None:
        b = istart
        e = _generate_range_overflow_safe(b, periods, stride, side="start")
    elif iend is not None and periods is not None:
        e = iend + stride
        b = _generate_range_overflow_safe(e, periods, stride, side="end")
    else:
        raise ValueError(
            "at least 'start' or 'end' should be specified if a 'period' is given."
        )

    with np.errstate(over="raise"):
        # If the range is sufficiently large, np.arange may overflow
        #  and incorrectly return an empty array if not caught.
        try:
            values = np.arange(b, e, stride, dtype=np.int64)
        except FloatingPointError:
            xdr = [b]
            while xdr[-1] != e:
                xdr.append(xdr[-1] + stride)
            values = np.array(xdr[:-1], dtype=np.int64)
    return values


def _generate_range_overflow_safe(
    endpoint: int, periods: int, stride: int, side: str = "start"
) -> int:
    """
    Calculate the second endpoint for passing to np.arange, checking
    to avoid an integer overflow.  Catch OverflowError and re-raise
    as OutOfBoundsDatetime.

    Parameters
    ----------
    endpoint : int
        nanosecond timestamp of the known endpoint of the desired range
    periods : int
        number of periods in the desired range
    stride : int
        nanoseconds between periods in the desired range
    side : {'start', 'end'}
        which end of the range `endpoint` refers to

    Returns
    -------
    other_end : int

    Raises
    ------
    OutOfBoundsDatetime
    """
    # GH#14187 raise instead of incorrectly wrapping around
    assert side in ["start", "end"]

    i64max = np.uint64(i8max)
    msg = f"Cannot generate range with {side}={endpoint} and periods={periods}"

    with np.errstate(over="raise"):
        # if periods * strides cannot be multiplied within the *uint64* bounds,
        #  we cannot salvage the operation by recursing, so raise
        try:
            addend = np.uint64(periods) * np.uint64(np.abs(stride))
        except FloatingPointError as err:
            raise OutOfBoundsDatetime(msg) from err

    if np.abs(addend) <= i64max:
        # relatively easy case without casting concerns
        return _generate_range_overflow_safe_signed(endpoint, periods, stride, side)

    elif (endpoint > 0 and side == "start" and stride > 0) or (
        endpoint < 0 < stride and side == "end"
    ):
        # no chance of not-overflowing
        raise OutOfBoundsDatetime(msg)

    elif side == "end" and endpoint - stride <= i64max < endpoint:
        # in _generate_regular_range we added `stride` thereby overflowing
        #  the bounds.  Adjust to fix this.
        return _generate_range_overflow_safe(
            endpoint - stride, periods - 1, stride, side
        )

    # split into smaller pieces
    mid_periods = periods // 2
    remaining = periods - mid_periods
    assert 0 < remaining < periods, (remaining, periods, endpoint, stride)

    midpoint = int(_generate_range_overflow_safe(endpoint, mid_periods, stride, side))
    return _generate_range_overflow_safe(midpoint, remaining, stride, side)


def _generate_range_overflow_safe_signed(
    endpoint: int, periods: int, stride: int, side: str
) -> int:
    """
    A special case for _generate_range_overflow_safe where `periods * stride`
    can be calculated without overflowing int64 bounds.
    """
    assert side in ["start", "end"]
    if side == "end":
        stride *= -1

    with np.errstate(over="raise"):
        addend = np.int64(periods) * np.int64(stride)
        try:
            # easy case with no overflows
            result = np.int64(endpoint) + addend
            if result == iNaT:
                # Putting this into a DatetimeArray/TimedeltaArray
                #  would incorrectly be interpreted as NaT
                raise OverflowError
            return int(result)
        except (FloatingPointError, OverflowError):
            # with endpoint negative and addend positive we risk
            #  FloatingPointError; with reversed signed we risk OverflowError
            pass

        # if stride and endpoint had opposite signs, then endpoint + addend
        #  should never overflow.  so they must have the same signs
        assert (stride > 0 and endpoint >= 0) or (stride < 0 and endpoint <= 0)

        if stride > 0:
            # watch out for very special case in which we just slightly
            #  exceed implementation bounds, but when passing the result to
            #  np.arange will get a result slightly within the bounds

            uresult = np.uint64(endpoint) + np.uint64(addend)
            i64max = np.uint64(i8max)
            assert uresult > i64max
            if uresult <= i64max + np.uint64(stride):
                return int(uresult)

    raise OutOfBoundsDatetime(
        f"Cannot generate range with {side}={endpoint} and periods={periods}"
    )
<|endoftext|>from __future__ import annotations

from functools import partial
import operator
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    cast,
)
import warnings

import numpy as np

from pandas._config import (
    get_option,
    using_string_dtype,
)

from pandas._libs import (
    lib,
    missing as libmissing,
)
from pandas._libs.arrays import NDArrayBacked
from pandas._libs.lib import ensure_string_array
from pandas.compat import (
    HAS_PYARROW,
    pa_version_under10p1,
)
from pandas.compat.numpy import function as nv
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.base import (
    ExtensionDtype,
    StorageExtensionDtype,
    register_extension_dtype,
)
from pandas.core.dtypes.common import (
    is_array_like,
    is_bool_dtype,
    is_integer_dtype,
    is_object_dtype,
    is_string_dtype,
    pandas_dtype,
)

from pandas.core import (
    missing,
    nanops,
    ops,
    roperator,
)
from pandas.core.algorithms import isin
from pandas.core.array_algos import masked_reductions
from pandas.core.arrays.base import ExtensionArray
from pandas.core.arrays.floating import (
    FloatingArray,
    FloatingDtype,
)
from pandas.core.arrays.integer import (
    IntegerArray,
    IntegerDtype,
)
from pandas.core.arrays.numpy_ import NumpyExtensionArray
from pandas.core.construction import extract_array
from pandas.core.indexers import check_array_indexer
from pandas.core.missing import isna

from pandas.io.formats import printing

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    import pyarrow

    from pandas._typing import (
        ArrayLike,
        AxisInt,
        Dtype,
        DtypeObj,
        NumpySorter,
        NumpyValueArrayLike,
        Scalar,
        Self,
        npt,
        type_t,
    )

    from pandas import Series


@register_extension_dtype
class StringDtype(StorageExtensionDtype):
    """
    Extension dtype for string data.

    .. warning::

       StringDtype is considered experimental. The implementation and
       parts of the API may change without warning.

    Parameters
    ----------
    storage : {"python", "pyarrow"}, optional
        If not given, the value of ``pd.options.mode.string_storage``.
    na_value : {np.nan, pd.NA}, default pd.NA
        Whether the dtype follows NaN or NA missing value semantics.

    Attributes
    ----------
    None

    Methods
    -------
    None

    Examples
    --------
    >>> pd.StringDtype()
    string[python]

    >>> pd.StringDtype(storage="pyarrow")
    string[pyarrow]
    """

    @property
    def name(self) -> str:  # type: ignore[override]
        if self._na_value is libmissing.NA:
            return "string"
        else:
            return "str"

    #: StringDtype().na_value uses pandas.NA except the implementation that
    # follows NumPy semantics, which uses nan.
    @property
    def na_value(self) -> libmissing.NAType | float:  # type: ignore[override]
        return self._na_value

    _metadata = ("storage", "_na_value")  # type: ignore[assignment]

    def __init__(
        self,
        storage: str | None = None,
        na_value: libmissing.NAType | float = libmissing.NA,
    ) -> None:
        # infer defaults
        if storage is None:
            if na_value is not libmissing.NA:
                storage = get_option("mode.string_storage")
                if storage == "auto":
                    if HAS_PYARROW:
                        storage = "pyarrow"
                    else:
                        storage = "python"
            else:
                storage = get_option("mode.string_storage")
                if storage == "auto":
                    storage = "python"

        if storage == "pyarrow_numpy":
            warnings.warn(
                "The 'pyarrow_numpy' storage option name is deprecated and will be "
                'removed in pandas 3.0. Use \'pd.StringDtype(storage="pyarrow", '
                "na_value-np.nan)' to construct the same dtype.\nOr enable the "
                "'pd.options.future.infer_string = True' option globally and use "
                'the "str" alias as a shorthand notation to specify a dtype '
                '(instead of "string[pyarrow_numpy]").',
                FutureWarning,
                stacklevel=find_stack_level(),
            )
            storage = "pyarrow"
            na_value = np.nan

        # validate options
        if storage not in {"python", "pyarrow"}:
            raise ValueError(
                f"Storage must be 'python' or 'pyarrow'. Got {storage} instead."
            )
        if storage == "pyarrow" and pa_version_under10p1:
            raise ImportError(
                "pyarrow>=10.0.1 is required for PyArrow backed StringArray."
            )

        if isinstance(na_value, float) and np.isnan(na_value):
            # when passed a NaN value, always set to np.nan to ensure we use
            # a consistent NaN value (and we can use `dtype.na_value is np.nan`)
            na_value = np.nan
        elif na_value is not libmissing.NA:
            raise ValueError(f"'na_value' must be np.nan or pd.NA, got {na_value}")

        self.storage = cast(str, storage)
        self._na_value = na_value

    def __repr__(self) -> str:
        if self._na_value is libmissing.NA:
            return f"{self.name}[{self.storage}]"
        else:
            storage = "" if self.storage == "pyarrow" else "storage='python', "
            return f"<StringDtype({storage}na_value={self._na_value})>"

    def __eq__(self, other: object) -> bool:
        # we need to override the base class __eq__ because na_value (NA or NaN)
        # cannot be checked with normal `==`
        if isinstance(other, str):
            # TODO should dtype == "string" work for the NaN variant?
            if other == "string" or other == self.name:  # noqa: PLR1714
                return True
            try:
                other = self.construct_from_string(other)
            except (TypeError, ImportError):
                # TypeError if `other` is not a valid string for StringDtype
                # ImportError if pyarrow is not installed for "string[pyarrow]"
                return False
        if isinstance(other, type(self)):
            return self.storage == other.storage and self.na_value is other.na_value
        return False

    def __setstate__(self, state: MutableMapping[str, Any]) -> None:
        # back-compat for pandas < 2.3, where na_value did not yet exist
        self.storage = state.pop("storage", "python")
        self._na_value = state.pop("_na_value", libmissing.NA)

    def __hash__(self) -> int:
        # need to override __hash__ as well because of overriding __eq__
        return super().__hash__()

    def __reduce__(self):
        return StringDtype, (self.storage, self.na_value)

    @property
    def type(self) -> type[str]:
        return str

    @classmethod
    def construct_from_string(cls, string) -> Self:
        """
        Construct a StringDtype from a string.

        Parameters
        ----------
        string : str
            The type of the name. The storage type will be taking from `string`.
            Valid options and their storage types are

            ========================== ==============================================
            string                     result storage
            ========================== ==============================================
            ``'string'``               pd.options.mode.string_storage, default python
            ``'string[python]'``       python
            ``'string[pyarrow]'``      pyarrow
            ========================== ==============================================

        Returns
        -------
        StringDtype

        Raise
        -----
        TypeError
            If the string is not a valid option.
        """
        if not isinstance(string, str):
            raise TypeError(
                f"'construct_from_string' expects a string, got {type(string)}"
            )
        if string == "string":
            return cls()
        elif string == "str" and using_string_dtype():
            return cls(na_value=np.nan)
        elif string == "string[python]":
            return cls(storage="python")
        elif string == "string[pyarrow]":
            return cls(storage="pyarrow")
        elif string == "string[pyarrow_numpy]":
            # this is deprecated in the dtype __init__, remove this in pandas 3.0
            return cls(storage="pyarrow_numpy")
        else:
            raise TypeError(f"Cannot construct a '{cls.__name__}' from '{string}'")

    # https://github.com/pandas-dev/pandas/issues/36126
    # error: Signature of "construct_array_type" incompatible with supertype
    # "ExtensionDtype"
    def construct_array_type(  # type: ignore[override]
        self,
    ) -> type_t[BaseStringArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        from pandas.core.arrays.string_arrow import (
            ArrowStringArray,
            ArrowStringArrayNumpySemantics,
        )

        if self.storage == "python" and self._na_value is libmissing.NA:
            return StringArray
        elif self.storage == "pyarrow" and self._na_value is libmissing.NA:
            return ArrowStringArray
        elif self.storage == "python":
            return StringArrayNumpySemantics
        else:
            return ArrowStringArrayNumpySemantics

    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None:
        storages = set()
        na_values = set()

        for dtype in dtypes:
            if isinstance(dtype, StringDtype):
                storages.add(dtype.storage)
                na_values.add(dtype.na_value)
            elif isinstance(dtype, np.dtype) and dtype.kind in ("U", "T"):
                continue
            else:
                return None

        if len(storages) == 2:
            # if both python and pyarrow storage -> priority to pyarrow
            storage = "pyarrow"
        else:
            storage = next(iter(storages))  # type: ignore[assignment]

        na_value: libmissing.NAType | float
        if len(na_values) == 2:
            # if both NaN and NA -> priority to NA
            na_value = libmissing.NA
        else:
            na_value = next(iter(na_values))

        return StringDtype(storage=storage, na_value=na_value)

    def __from_arrow__(
        self, array: pyarrow.Array | pyarrow.ChunkedArray
    ) -> BaseStringArray:
        """
        Construct StringArray from pyarrow Array/ChunkedArray.
        """
        if self.storage == "pyarrow":
            if self._na_value is libmissing.NA:
                from pandas.core.arrays.string_arrow import ArrowStringArray

                return ArrowStringArray(array)
            else:
                from pandas.core.arrays.string_arrow import (
                    ArrowStringArrayNumpySemantics,
                )

                return ArrowStringArrayNumpySemantics(array)

        else:
            import pyarrow

            if isinstance(array, pyarrow.Array):
                chunks = [array]
            else:
                # pyarrow.ChunkedArray
                chunks = array.chunks

            results = []
            for arr in chunks:
                # convert chunk by chunk to numpy and concatenate then, to avoid
                # overflow for large string data when concatenating the pyarrow arrays
                arr = arr.to_numpy(zero_copy_only=False)
                arr = ensure_string_array(arr, na_value=self.na_value)
                results.append(arr)

        if len(chunks) == 0:
            arr = np.array([], dtype=object)
        else:
            arr = np.concatenate(results)

        # Bypass validation inside StringArray constructor, see GH#47781
        new_string_array = StringArray.__new__(StringArray)
        NDArrayBacked.__init__(new_string_array, arr, self)
        return new_string_array


class BaseStringArray(ExtensionArray):
    """
    Mixin class for StringArray, ArrowStringArray.
    """

    dtype: StringDtype

    # TODO(4.0): Once the deprecation here is enforced, this method can be
    #  removed and we use the parent class method instead.
    def _logical_method(self, other, op):
        if (
            op in (roperator.ror_, roperator.rand_, roperator.rxor)
            and isinstance(other, np.ndarray)
            and other.dtype == bool
        ):
            # GH#60234 backward compatibility for the move to StringDtype in 3.0
            op_name = op.__name__[1:].strip("_")
            warnings.warn(
                f"'{op_name}' operations between boolean dtype and {self.dtype} are "
                "deprecated and will raise in a future version. Explicitly "
                "cast the strings to a boolean dtype before operating instead.",
                DeprecationWarning,
                stacklevel=find_stack_level(),
            )
            return op(other, self.astype(bool))
        return NotImplemented

    @doc(ExtensionArray.tolist)
    def tolist(self):
        if self.ndim > 1:
            return [x.tolist() for x in self]
        return list(self.to_numpy())

    @classmethod
    def _from_scalars(cls, scalars, dtype: DtypeObj) -> Self:
        if lib.infer_dtype(scalars, skipna=True) not in ["string", "empty"]:
            # TODO: require any NAs be valid-for-string
            raise ValueError
        return cls._from_sequence(scalars, dtype=dtype)

    def _formatter(self, boxed: bool = False):
        formatter = partial(
            printing.pprint_thing,
            escape_chars=("\t", "\r", "\n"),
            quote_strings=not boxed,
        )
        return formatter

    def _str_map(
        self,
        f,
        na_value=lib.no_default,
        dtype: Dtype | None = None,
        convert: bool = True,
    ):
        if self.dtype.na_value is np.nan:
            return self._str_map_nan_semantics(
                f, na_value=na_value, dtype=dtype, convert=convert
            )

        from pandas.arrays import BooleanArray

        if dtype is None:
            dtype = self.dtype
        if na_value is lib.no_default:
            na_value = self.dtype.na_value

        mask = isna(self)
        arr = np.asarray(self)

        if is_integer_dtype(dtype) or is_bool_dtype(dtype):
            constructor: type[IntegerArray | BooleanArray]
            if is_integer_dtype(dtype):
                constructor = IntegerArray
            else:
                constructor = BooleanArray

            na_value_is_na = isna(na_value)
            if na_value_is_na:
                na_value = 1
            elif dtype == np.dtype("bool"):
                # GH#55736
                na_value = bool(na_value)
            result = lib.map_infer_mask(
                arr,
                f,
                mask.view("uint8"),
                convert=False,
                na_value=na_value,
                # error: Argument 1 to "dtype" has incompatible type
                # "Union[ExtensionDtype, str, dtype[Any], Type[object]]"; expected
                # "Type[object]"
                dtype=np.dtype(cast(type, dtype)),
            )

            if not na_value_is_na:
                mask[:] = False

            return constructor(result, mask)

        else:
            return self._str_map_str_or_object(dtype, na_value, arr, f, mask)

    def _str_map_str_or_object(
        self,
        dtype,
        na_value,
        arr: np.ndarray,
        f,
        mask: npt.NDArray[np.bool_],
    ):
        # _str_map helper for case where dtype is either string dtype or object
        if is_string_dtype(dtype) and not is_object_dtype(dtype):
            # i.e. StringDtype
            result = lib.map_infer_mask(
                arr, f, mask.view("uint8"), convert=False, na_value=na_value
            )
            if self.dtype.storage == "pyarrow":
                import pyarrow as pa

                result = pa.array(
                    result, mask=mask, type=pa.large_string(), from_pandas=True
                )
            # error: Too many arguments for "BaseStringArray"
            return type(self)(result)  # type: ignore[call-arg]

        else:
            # This is when the result type is object. We reach this when
            # -> We know the result type is truly object (e.g. .encode returns bytes
            #    or .findall returns a list).
            # -> We don't know the result type. E.g. `.get` can return anything.
            return lib.map_infer_mask(arr, f, mask.view("uint8"))

    def _str_map_nan_semantics(
        self,
        f,
        na_value=lib.no_default,
        dtype: Dtype | None = None,
        convert: bool = True,
    ):
        if dtype is None:
            dtype = self.dtype
        if na_value is lib.no_default:
            if is_bool_dtype(dtype):
                # NaN propagates as False
                na_value = False
            else:
                na_value = self.dtype.na_value

        mask = isna(self)
        arr = np.asarray(self)

        if is_integer_dtype(dtype) or is_bool_dtype(dtype):
            na_value_is_na = isna(na_value)
            if na_value_is_na:
                if is_integer_dtype(dtype):
                    na_value = 0
                else:
                    # NaN propagates as False
                    na_value = False

            result = lib.map_infer_mask(
                arr,
                f,
                mask.view("uint8"),
                convert=False,
                na_value=na_value,
                dtype=np.dtype(cast(type, dtype)),
            )
            if na_value_is_na and is_integer_dtype(dtype) and mask.any():
                # TODO: we could alternatively do this check before map_infer_mask
                #  and adjust the dtype/na_value we pass there. Which is more
                #  performant?
                result = result.astype("float64")
                result[mask] = np.nan

            return result

        else:
            return self._str_map_str_or_object(dtype, na_value, arr, f, mask)

    def view(self, dtype: Dtype | None = None) -> ArrayLike:
        if dtype is not None:
            raise TypeError("Cannot change data-type for string array.")
        return super().view(dtype=dtype)


# error: Definition of "_concat_same_type" in base class "NDArrayBacked" is
# incompatible with definition in base class "ExtensionArray"
class StringArray(BaseStringArray, NumpyExtensionArray):  # type: ignore[misc]
    """
    Extension array for string data.

    .. warning::

       StringArray is considered experimental. The implementation and
       parts of the API may change without warning.

    Parameters
    ----------
    values : array-like
        The array of data.

        .. warning::

           Currently, this expects an object-dtype ndarray
           where the elements are Python strings
           or nan-likes (``None``, ``np.nan``, ``NA``).
           This may change without warning in the future. Use
           :meth:`pandas.array` with ``dtype="string"`` for a stable way of
           creating a `StringArray` from any sequence.

        .. versionchanged:: 1.5.0

           StringArray now accepts array-likes containing
           nan-likes(``None``, ``np.nan``) for the ``values`` parameter
           in addition to strings and :attr:`pandas.NA`

    copy : bool, default False
        Whether to copy the array of data.

    Attributes
    ----------
    None

    Methods
    -------
    None

    See Also
    --------
    :func:`pandas.array`
        The recommended function for creating a StringArray.
    Series.str
        The string methods are available on Series backed by
        a StringArray.

    Notes
    -----
    StringArray returns a BooleanArray for comparison methods.

    Examples
    --------
    >>> pd.array(['This is', 'some text', None, 'data.'], dtype="string")
    <StringArray>
    ['This is', 'some text', <NA>, 'data.']
    Length: 4, dtype: string

    Unlike arrays instantiated with ``dtype="object"``, ``StringArray``
    will convert the values to strings.

    >>> pd.array(['1', 1], dtype="object")
    <NumpyExtensionArray>
    ['1', 1]
    Length: 2, dtype: object
    >>> pd.array(['1', 1], dtype="string")
    <StringArray>
    ['1', '1']
    Length: 2, dtype: string

    However, instantiating StringArrays directly with non-strings will raise an error.

    For comparison methods, `StringArray` returns a :class:`pandas.BooleanArray`:

    >>> pd.array(["a", None, "c"], dtype="string") == "a"
    <BooleanArray>
    [True, <NA>, False]
    Length: 3, dtype: boolean
    """

    # undo the NumpyExtensionArray hack
    _typ = "extension"
    _storage = "python"
    _na_value: libmissing.NAType | float = libmissing.NA

    def __init__(self, values, copy: bool = False) -> None:
        values = extract_array(values)

        super().__init__(values, copy=copy)
        if not isinstance(values, type(self)):
            self._validate()
        NDArrayBacked.__init__(
            self,
            self._ndarray,
            StringDtype(storage=self._storage, na_value=self._na_value),
        )

    def _validate(self):
        """Validate that we only store NA or strings."""
        if len(self._ndarray) and not lib.is_string_array(self._ndarray, skipna=True):
            raise ValueError("StringArray requires a sequence of strings or pandas.NA")
        if self._ndarray.dtype != "object":
            raise ValueError(
                "StringArray requires a sequence of strings or pandas.NA. Got "
                f"'{self._ndarray.dtype}' dtype instead."
            )
        # Check to see if need to convert Na values to pd.NA
        if self._ndarray.ndim > 2:
            # Ravel if ndims > 2 b/c no cythonized version available
            lib.convert_nans_to_NA(self._ndarray.ravel("K"))
        else:
            lib.convert_nans_to_NA(self._ndarray)

    def _validate_scalar(self, value):
        # used by NDArrayBackedExtensionIndex.insert
        if isna(value):
            return self.dtype.na_value
        elif not isinstance(value, str):
            raise TypeError(
                f"Invalid value '{value}' for dtype '{self.dtype}'. Value should be a "
                f"string or missing value, got '{type(value).__name__}' instead."
            )
        return value

    @classmethod
    def _from_sequence(cls, scalars, *, dtype: Dtype | None = None, copy: bool = False):
        if dtype and not (isinstance(dtype, str) and dtype == "string"):
            dtype = pandas_dtype(dtype)
            assert isinstance(dtype, StringDtype) and dtype.storage == "python"
        else:
            if using_string_dtype():
                dtype = StringDtype(storage="python", na_value=np.nan)
            else:
                dtype = StringDtype(storage="python")

        from pandas.core.arrays.masked import BaseMaskedArray

        na_value = dtype.na_value
        if isinstance(scalars, BaseMaskedArray):
            # avoid costly conversion to object dtype
            na_values = scalars._mask
            result = scalars._data
            result = lib.ensure_string_array(result, copy=copy, convert_na_value=False)
            result[na_values] = na_value

        else:
            if lib.is_pyarrow_array(scalars):
                # pyarrow array; we cannot rely on the "to_numpy" check in
                #  ensure_string_array because calling scalars.to_numpy would set
                #  zero_copy_only to True which caused problems see GH#52076
                scalars = np.array(scalars)
            # convert non-na-likes to str, and nan-likes to StringDtype().na_value
            result = lib.ensure_string_array(scalars, na_value=na_value, copy=copy)

        # Manually creating new array avoids the validation step in the __init__, so is
        # faster. Refactor need for validation?
        new_string_array = cls.__new__(cls)
        NDArrayBacked.__init__(new_string_array, result, dtype)

        return new_string_array

    @classmethod
    def _from_sequence_of_strings(
        cls, strings, *, dtype: Dtype | None = None, copy: bool = False
    ):
        return cls._from_sequence(strings, dtype=dtype, copy=copy)

    @classmethod
    def _empty(cls, shape, dtype) -> StringArray:
        values = np.empty(shape, dtype=object)
        values[:] = libmissing.NA
        return cls(values).astype(dtype, copy=False)

    def __arrow_array__(self, type=None):
        """
        Convert myself into a pyarrow Array.
        """
        import pyarrow as pa

        if type is None:
            type = pa.string()

        values = self._ndarray.copy()
        values[self.isna()] = None
        return pa.array(values, type=type, from_pandas=True)

    def _values_for_factorize(self) -> tuple[np.ndarray, libmissing.NAType | float]:  # type: ignore[override]
        arr = self._ndarray.copy()

        return arr, self.dtype.na_value

    def _maybe_convert_setitem_value(self, value):
        """Maybe convert value to be pyarrow compatible."""
        if lib.is_scalar(value):
            if isna(value):
                value = self.dtype.na_value
            elif not isinstance(value, str):
                raise TypeError(
                    f"Invalid value '{value}' for dtype '{self.dtype}'. Value should "
                    f"be a string or missing value, got '{type(value).__name__}' "
                    "instead."
                )
        else:
            value = extract_array(value, extract_numpy=True)
            if not is_array_like(value):
                value = np.asarray(value, dtype=object)
            elif isinstance(value.dtype, type(self.dtype)):
                return value
            else:
                # cast categories and friends to arrays to see if values are
                # compatible, compatibility with arrow backed strings
                value = np.asarray(value)
            if len(value) and not lib.is_string_array(value, skipna=True):
                raise TypeError(
                    "Invalid value for dtype 'str'. Value should be a "
                    "string or missing value (or array of those)."
                )
        return value

    def __setitem__(self, key, value) -> None:
        value = self._maybe_convert_setitem_value(value)

        key = check_array_indexer(self, key)
        scalar_key = lib.is_scalar(key)
        scalar_value = lib.is_scalar(value)
        if scalar_key and not scalar_value:
            raise ValueError("setting an array element with a sequence.")

        if not scalar_value:
            if value.dtype == self.dtype:
                value = value._ndarray
            else:
                value = np.asarray(value)
                mask = isna(value)
                if mask.any():
                    value = value.copy()
                    value[isna(value)] = self.dtype.na_value

        super().__setitem__(key, value)

    def _putmask(self, mask: npt.NDArray[np.bool_], value) -> None:
        # the super() method NDArrayBackedExtensionArray._putmask uses
        # np.putmask which doesn't properly handle None/pd.NA, so using the
        # base class implementation that uses __setitem__
        ExtensionArray._putmask(self, mask, value)

    def _where(self, mask: npt.NDArray[np.bool_], value) -> Self:
        # the super() method NDArrayBackedExtensionArray._where uses
        # np.putmask which doesn't properly handle None/pd.NA, so using the
        # base class implementation that uses __setitem__
        return ExtensionArray._where(self, mask, value)

    def isin(self, values: ArrayLike) -> npt.NDArray[np.bool_]:
        if isinstance(values, BaseStringArray) or (
            isinstance(values, ExtensionArray) and is_string_dtype(values.dtype)
        ):
            values = values.astype(self.dtype, copy=False)
        else:
            if not lib.is_string_array(np.asarray(values), skipna=True):
                values = np.array(
                    [val for val in values if isinstance(val, str) or isna(val)],
                    dtype=object,
                )
                if not len(values):
                    return np.zeros(self.shape, dtype=bool)

            values = self._from_sequence(values, dtype=self.dtype)

        return isin(np.asarray(self), np.asarray(values))

    def astype(self, dtype, copy: bool = True):
        dtype = pandas_dtype(dtype)

        if dtype == self.dtype:
            if copy:
                return self.copy()
            return self

        elif isinstance(dtype, IntegerDtype):
            arr = self._ndarray.copy()
            mask = self.isna()
            arr[mask] = 0
            values = arr.astype(dtype.numpy_dtype)
            return IntegerArray(values, mask, copy=False)
        elif isinstance(dtype, FloatingDtype):
            arr = self.copy()
            mask = self.isna()
            arr[mask] = "0"
            values = arr.astype(dtype.numpy_dtype)
            return FloatingArray(values, mask, copy=False)
        elif isinstance(dtype, ExtensionDtype):
            # Skip the NumpyExtensionArray.astype method
            return ExtensionArray.astype(self, dtype, copy)
        elif np.issubdtype(dtype, np.floating):
            arr = self._ndarray.copy()
            mask = self.isna()
            arr[mask] = 0
            values = arr.astype(dtype)
            values[mask] = np.nan
            return values

        return super().astype(dtype, copy)

    def _reduce(
        self,
        name: str,
        *,
        skipna: bool = True,
        keepdims: bool = False,
        axis: AxisInt | None = 0,
        **kwargs,
    ):
        if self.dtype.na_value is np.nan and name in ["any", "all"]:
            if name == "any":
                return nanops.nanany(self._ndarray, skipna=skipna)
            else:
                return nanops.nanall(self._ndarray, skipna=skipna)

        if name in ["min", "max", "argmin", "argmax", "sum"]:
            result = getattr(self, name)(skipna=skipna, axis=axis, **kwargs)
            if keepdims:
                return self._from_sequence([result], dtype=self.dtype)
            return result
        raise TypeError(f"Cannot perform reduction '{name}' with string dtype")

    def _accumulate(self, name: str, *, skipna: bool = True, **kwargs) -> StringArray:
        """
        Return an ExtensionArray performing an accumulation operation.

        The underlying data type might change.

        Parameters
        ----------
        name : str
            Name of the function, supported values are:
            - cummin
            - cummax
            - cumsum
            - cumprod
        skipna : bool, default True
            If True, skip NA values.
        **kwargs
            Additional keyword arguments passed to the accumulation function.
            Currently, there is no supported kwarg.

        Returns
        -------
        array

        Raises
        ------
        NotImplementedError : subclass does not define accumulations
        """
        if name == "cumprod":
            msg = f"operation '{name}' not supported for dtype '{self.dtype}'"
            raise TypeError(msg)

        # We may need to strip out trailing NA values
        tail: np.ndarray | None = None
        na_mask: np.ndarray | None = None
        ndarray = self._ndarray
        np_func = {
            "cumsum": np.cumsum,
            "cummin": np.minimum.accumulate,
            "cummax": np.maximum.accumulate,
        }[name]

        if self._hasna:
            na_mask = cast("npt.NDArray[np.bool_]", isna(ndarray))
            if np.all(na_mask):
                return type(self)(ndarray)
            if skipna:
                if name == "cumsum":
                    ndarray = np.where(na_mask, "", ndarray)
                else:
                    # We can retain the running min/max by forward/backward filling.
                    ndarray = ndarray.copy()
                    missing.pad_or_backfill_inplace(
                        ndarray,
                        method="pad",
                        axis=0,
                    )
                    missing.pad_or_backfill_inplace(
                        ndarray,
                        method="backfill",
                        axis=0,
                    )
            else:
                # When not skipping NA values, the result should be null from
                # the first NA value onward.
                idx = np.argmax(na_mask)
                tail = np.empty(len(ndarray) - idx, dtype="object")
                tail[:] = self.dtype.na_value
                ndarray = ndarray[:idx]

        # mypy: Cannot call function of unknown type
        np_result = np_func(ndarray)  # type: ignore[operator]

        if tail is not None:
            np_result = np.hstack((np_result, tail))
        elif na_mask is not None:
            # Argument 2 to "where" has incompatible type "NAType | float"
            np_result = np.where(na_mask, self.dtype.na_value, np_result)  # type: ignore[arg-type]

        result = type(self)(np_result)
        return result

    def _wrap_reduction_result(self, axis: AxisInt | None, result) -> Any:
        if self.dtype.na_value is np.nan and result is libmissing.NA:
            # the masked_reductions use pd.NA -> convert to np.nan
            return np.nan
        return super()._wrap_reduction_result(axis, result)

    def min(self, axis=None, skipna: bool = True, **kwargs) -> Scalar:
        nv.validate_min((), kwargs)
        result = masked_reductions.min(
            values=self.to_numpy(), mask=self.isna(), skipna=skipna
        )
        return self._wrap_reduction_result(axis, result)

    def max(self, axis=None, skipna: bool = True, **kwargs) -> Scalar:
        nv.validate_max((), kwargs)
        result = masked_reductions.max(
            values=self.to_numpy(), mask=self.isna(), skipna=skipna
        )
        return self._wrap_reduction_result(axis, result)

    def sum(
        self,
        *,
        axis: AxisInt | None = None,
        skipna: bool = True,
        min_count: int = 0,
        **kwargs,
    ) -> Scalar:
        nv.validate_sum((), kwargs)
        result = masked_reductions.sum(
            values=self._ndarray, mask=self.isna(), skipna=skipna
        )
        return self._wrap_reduction_result(axis, result)

    def value_counts(self, dropna: bool = True) -> Series:
        from pandas.core.algorithms import value_counts_internal as value_counts

        result = value_counts(self._ndarray, dropna=dropna).astype("Int64")
        result = value_counts(self._ndarray, sort=False, dropna=dropna)
        result.index = result.index.astype(self.dtype)

        if self.dtype.na_value is libmissing.NA:
            result = result.astype("Int64")
        return result

    def memory_usage(self, deep: bool = False) -> int:
        result = self._ndarray.nbytes
        if deep:
            return result + lib.memory_usage_of_objects(self._ndarray)
        return result

    @doc(ExtensionArray.searchsorted)
    def searchsorted(
        self,
        value: NumpyValueArrayLike | ExtensionArray,
        side: Literal["left", "right"] = "left",
        sorter: NumpySorter | None = None,
    ) -> npt.NDArray[np.intp] | np.intp:
        if self._hasna:
            raise ValueError(
                "searchsorted requires array to be sorted, which is impossible "
                "with NAs present."
            )
        return super().searchsorted(value=value, side=side, sorter=sorter)

    def _cmp_method(self, other, op):
        from pandas.arrays import (
            ArrowExtensionArray,
            BooleanArray,
        )

        if (
            isinstance(other, BaseStringArray)
            and self.dtype.na_value is not libmissing.NA
            and other.dtype.na_value is libmissing.NA
        ):
            # NA has priority of NaN semantics
            return NotImplemented

        if isinstance(other, ArrowExtensionArray):
            if isinstance(other, BaseStringArray):
                # pyarrow storage has priority over python storage
                # (except if we have NA semantics and other not)
                if not (
                    self.dtype.na_value is libmissing.NA
                    and other.dtype.na_value is not libmissing.NA
                ):
                    return NotImplemented
            else:
                return NotImplemented

        if isinstance(other, StringArray):
            other = other._ndarray

        mask = isna(self) | isna(other)
        valid = ~mask

        if lib.is_list_like(other):
            if len(other) != len(self):
                # prevent improper broadcasting when other is 2D
                raise ValueError(
                    f"Lengths of operands do not match: {len(self)} != {len(other)}"
                )

            # for array-likes, first filter out NAs before converting to numpy
            if not is_array_like(other):
                other = np.asarray(other)
            other = other[valid]

        if op.__name__ in ops.ARITHMETIC_BINOPS:
            result = np.empty_like(self._ndarray, dtype="object")
            result[mask] = self.dtype.na_value
            result[valid] = op(self._ndarray[valid], other)
            if isinstance(other, Path):
                # GH#61940
                return result
            return self._from_backing_data(result)
        else:
            # logical
            result = np.zeros(len(self._ndarray), dtype="bool")
            result[valid] = op(self._ndarray[valid], other)
            res_arr = BooleanArray(result, mask)
            if self.dtype.na_value is np.nan:
                if op == operator.ne:
                    return res_arr.to_numpy(np.bool_, na_value=True)
                else:
                    return res_arr.to_numpy(np.bool_, na_value=False)
            return res_arr

    _arith_method = _cmp_method


class StringArrayNumpySemantics(StringArray):
    _storage = "python"
    _na_value = np.nan

    def _validate(self) -> None:
        """Validate that we only store NaN or strings."""
        if len(self._ndarray) and not lib.is_string_array(self._ndarray, skipna=True):
            raise ValueError(
                "StringArrayNumpySemantics requires a sequence of strings or NaN"
            )
        if self._ndarray.dtype != "object":
            raise ValueError(
                "StringArrayNumpySemantics requires a sequence of strings or NaN. Got "
                f"'{self._ndarray.dtype}' dtype instead."
            )
        # TODO validate or force NA/None to NaN

    @classmethod
    def _from_sequence(
        cls, scalars, *, dtype: Dtype | None = None, copy: bool = False
    ) -> Self:
        if dtype is None:
            dtype = StringDtype(storage="python", na_value=np.nan)
        return super()._from_sequence(scalars, dtype=dtype, copy=copy)
<|endoftext|>"""
An interface for extending pandas with custom arrays.

.. warning::

   This is an experimental API and subject to breaking changes
   without warning.
"""
from __future__ import annotations

import operator
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Literal,
    cast,
    overload,
)
import warnings

import numpy as np

from pandas._libs import (
    algos as libalgos,
    lib,
)
from pandas.compat import set_function_name
from pandas.compat.numpy import function as nv
from pandas.errors import AbstractMethodError
from pandas.util._decorators import (
    Appender,
    Substitution,
    cache_readonly,
)
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import (
    validate_bool_kwarg,
    validate_fillna_kwargs,
    validate_insert_loc,
)

from pandas.core.dtypes.cast import maybe_cast_pointwise_result
from pandas.core.dtypes.common import (
    is_list_like,
    is_scalar,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCIndex,
    ABCSeries,
)
from pandas.core.dtypes.missing import isna

from pandas.core import (
    arraylike,
    missing,
    roperator,
)
from pandas.core.algorithms import (
    duplicated,
    factorize_array,
    isin,
    map_array,
    mode,
    rank,
    unique,
)
from pandas.core.array_algos.quantile import quantile_with_mask
from pandas.core.missing import _fill_limit_area_1d
from pandas.core.sorting import (
    nargminmax,
    nargsort,
)

if TYPE_CHECKING:
    from collections.abc import (
        Iterator,
        Sequence,
    )

    from pandas._typing import (
        ArrayLike,
        AstypeArg,
        AxisInt,
        Dtype,
        DtypeObj,
        FillnaOptions,
        InterpolateOptions,
        NumpySorter,
        NumpyValueArrayLike,
        PositionalIndexer,
        ScalarIndexer,
        Self,
        SequenceIndexer,
        Shape,
        SortKind,
        TakeIndexer,
        npt,
    )

    from pandas import Index

_extension_array_shared_docs: dict[str, str] = {}


class ExtensionArray:
    """
    Abstract base class for custom 1-D array types.

    pandas will recognize instances of this class as proper arrays
    with a custom type and will not attempt to coerce them to objects. They
    may be stored directly inside a :class:`DataFrame` or :class:`Series`.

    Attributes
    ----------
    dtype
    nbytes
    ndim
    shape

    Methods
    -------
    argsort
    astype
    copy
    dropna
    duplicated
    factorize
    fillna
    equals
    insert
    interpolate
    isin
    isna
    ravel
    repeat
    searchsorted
    shift
    take
    tolist
    unique
    view
    _accumulate
    _concat_same_type
    _explode
    _formatter
    _from_factorized
    _from_sequence
    _from_sequence_of_strings
    _hash_pandas_object
    _pad_or_backfill
    _reduce
    _values_for_argsort
    _values_for_factorize

    Notes
    -----
    The interface includes the following abstract methods that must be
    implemented by subclasses:

    * _from_sequence
    * _from_factorized
    * __getitem__
    * __len__
    * __eq__
    * dtype
    * nbytes
    * isna
    * take
    * copy
    * _concat_same_type
    * interpolate

    A default repr displaying the type, (truncated) data, length,
    and dtype is provided. It can be customized or replaced by
    by overriding:

    * __repr__ : A default repr for the ExtensionArray.
    * _formatter : Print scalars inside a Series or DataFrame.

    Some methods require casting the ExtensionArray to an ndarray of Python
    objects with ``self.astype(object)``, which may be expensive. When
    performance is a concern, we highly recommend overriding the following
    methods:

    * fillna
    * _pad_or_backfill
    * dropna
    * unique
    * factorize / _values_for_factorize
    * argsort, argmax, argmin / _values_for_argsort
    * searchsorted
    * map

    The remaining methods implemented on this class should be performant,
    as they only compose abstract methods. Still, a more efficient
    implementation may be available, and these methods can be overridden.

    One can implement methods to handle array accumulations or reductions.

    * _accumulate
    * _reduce

    One can implement methods to handle parsing from strings that will be used
    in methods such as ``pandas.io.parsers.read_csv``.

    * _from_sequence_of_strings

    This class does not inherit from 'abc.ABCMeta' for performance reasons.
    Methods and properties required by the interface raise
    ``pandas.errors.AbstractMethodError`` and no ``register`` method is
    provided for registering virtual subclasses.

    ExtensionArrays are limited to 1 dimension.

    They may be backed by none, one, or many NumPy arrays. For example,
    ``pandas.Categorical`` is an extension array backed by two arrays,
    one for codes and one for categories. An array of IPv6 address may
    be backed by a NumPy structured array with two fields, one for the
    lower 64 bits and one for the upper 64 bits. Or they may be backed
    by some other storage type, like Python lists. Pandas makes no
    assumptions on how the data are stored, just that it can be converted
    to a NumPy array.
    The ExtensionArray interface does not impose any rules on how this data
    is stored. However, currently, the backing data cannot be stored in
    attributes called ``.values`` or ``._values`` to ensure full compatibility
    with pandas internals. But other names as ``.data``, ``._data``,
    ``._items``, ... can be freely used.

    If implementing NumPy's ``__array_ufunc__`` interface, pandas expects
    that

    1. You defer by returning ``NotImplemented`` when any Series are present
       in `inputs`. Pandas will extract the arrays and call the ufunc again.
    2. You define a ``_HANDLED_TYPES`` tuple as an attribute on the class.
       Pandas inspect this to determine whether the ufunc is valid for the
       types present.

    See :ref:`extending.extension.ufunc` for more.

    By default, ExtensionArrays are not hashable.  Immutable subclasses may
    override this behavior.

    Examples
    --------
    Please see the following:

    https://github.com/pandas-dev/pandas/blob/main/pandas/tests/extension/list/array.py
    """

    # '_typ' is for pandas.core.dtypes.generic.ABCExtensionArray.
    # Don't override this.
    _typ = "extension"

    # similar to __array_priority__, positions ExtensionArray after Index,
    #  Series, and DataFrame.  EA subclasses may override to choose which EA
    #  subclass takes priority. If overriding, the value should always be
    #  strictly less than 2000 to be below Index.__pandas_priority__.
    __pandas_priority__ = 1000

    # ------------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------------

    @classmethod
    def _from_sequence(cls, scalars, *, dtype: Dtype | None = None, copy: bool = False):
        """
        Construct a new ExtensionArray from a sequence of scalars.

        Parameters
        ----------
        scalars : Sequence
            Each element will be an instance of the scalar type for this
            array, ``cls.dtype.type`` or be converted into this type in this method.
        dtype : dtype, optional
            Construct for this particular dtype. This should be a Dtype
            compatible with the ExtensionArray.
        copy : bool, default False
            If True, copy the underlying data.

        Returns
        -------
        ExtensionArray

        Examples
        --------
        >>> pd.arrays.IntegerArray._from_sequence([4, 5])
        <IntegerArray>
        [4, 5]
        Length: 2, dtype: Int64
        """
        raise AbstractMethodError(cls)

    @classmethod
    def _from_scalars(cls, scalars, *, dtype: DtypeObj) -> Self:
        """
        Strict analogue to _from_sequence, allowing only sequences of scalars
        that should be specifically inferred to the given dtype.

        Parameters
        ----------
        scalars : sequence
        dtype : ExtensionDtype

        Raises
        ------
        TypeError or ValueError

        Notes
        -----
        This is called in a try/except block when casting the result of a
        pointwise operation.
        """
        try:
            return cls._from_sequence(scalars, dtype=dtype, copy=False)
        except (ValueError, TypeError):
            raise
        except Exception:
            warnings.warn(
                "_from_scalars should only raise ValueError or TypeError. "
                "Consider overriding _from_scalars where appropriate.",
                stacklevel=find_stack_level(),
            )
            raise

    @classmethod
    def _from_sequence_of_strings(
        cls, strings, *, dtype: Dtype | None = None, copy: bool = False
    ):
        """
        Construct a new ExtensionArray from a sequence of strings.

        Parameters
        ----------
        strings : Sequence
            Each element will be an instance of the scalar type for this
            array, ``cls.dtype.type``.
        dtype : dtype, optional
            Construct for this particular dtype. This should be a Dtype
            compatible with the ExtensionArray.
        copy : bool, default False
            If True, copy the underlying data.

        Returns
        -------
        ExtensionArray

        Examples
        --------
        >>> pd.arrays.IntegerArray._from_sequence_of_strings(["1", "2", "3"])
        <IntegerArray>
        [1, 2, 3]
        Length: 3, dtype: Int64
        """
        raise AbstractMethodError(cls)

    @classmethod
    def _from_factorized(cls, values, original):
        """
        Reconstruct an ExtensionArray after factorization.

        Parameters
        ----------
        values : ndarray
            An integer ndarray with the factorized values.
        original : ExtensionArray
            The original ExtensionArray that factorize was called on.

        See Also
        --------
        factorize : Top-level factorize method that dispatches here.
        ExtensionArray.factorize : Encode the extension array as an enumerated type.

        Examples
        --------
        >>> interv_arr = pd.arrays.IntervalArray([pd.Interval(0, 1),
        ...                                      pd.Interval(1, 5), pd.Interval(1, 5)])
        >>> codes, uniques = pd.factorize(interv_arr)
        >>> pd.arrays.IntervalArray._from_factorized(uniques, interv_arr)
        <IntervalArray>
        [(0, 1], (1, 5]]
        Length: 2, dtype: interval[int64, right]
        """
        raise AbstractMethodError(cls)

    # ------------------------------------------------------------------------
    # Must be a Sequence
    # ------------------------------------------------------------------------
    @overload
    def __getitem__(self, item: ScalarIndexer) -> Any:
        ...

    @overload
    def __getitem__(self, item: SequenceIndexer) -> Self:
        ...

    def __getitem__(self, item: PositionalIndexer) -> Self | Any:
        """
        Select a subset of self.

        Parameters
        ----------
        item : int, slice, or ndarray
            * int: The position in 'self' to get.

            * slice: A slice object, where 'start', 'stop', and 'step' are
              integers or None

            * ndarray: A 1-d boolean NumPy ndarray the same length as 'self'

            * list[int]:  A list of int

        Returns
        -------
        item : scalar or ExtensionArray

        Notes
        -----
        For scalar ``item``, return a scalar value suitable for the array's
        type. This should be an instance of ``self.dtype.type``.

        For slice ``key``, return an instance of ``ExtensionArray``, even
        if the slice is length 0 or 1.

        For a boolean mask, return an instance of ``ExtensionArray``, filtered
        to the values where ``item`` is True.
        """
        raise AbstractMethodError(self)

    def __setitem__(self, key, value) -> None:
        """
        Set one or more values inplace.

        This method is not required to satisfy the pandas extension array
        interface.

        Parameters
        ----------
        key : int, ndarray, or slice
            When called from, e.g. ``Series.__setitem__``, ``key`` will be
            one of

            * scalar int
            * ndarray of integers.
            * boolean ndarray
            * slice object

        value : ExtensionDtype.type, Sequence[ExtensionDtype.type], or object
            value or values to be set of ``key``.

        Returns
        -------
        None
        """
        # Some notes to the ExtensionArray implementer who may have ended up
        # here. While this method is not required for the interface, if you
        # *do* choose to implement __setitem__, then some semantics should be
        # observed:
        #
        # * Setting multiple values : ExtensionArrays should support setting
        #   multiple values at once, 'key' will be a sequence of integers and
        #  'value' will be a same-length sequence.
        #
        # * Broadcasting : For a sequence 'key' and a scalar 'value',
        #   each position in 'key' should be set to 'value'.
        #
        # * Coercion : Most users will expect basic coercion to work. For
        #   example, a string like '2018-01-01' is coerced to a datetime
        #   when setting on a datetime64ns array. In general, if the
        #   __init__ method coerces that value, then so should __setitem__
        # Note, also, that Series/DataFrame.where internally use __setitem__
        # on a copy of the data.
        raise NotImplementedError(f"{type(self)} does not implement __setitem__.")

    def __len__(self) -> int:
        """
        Length of this array

        Returns
        -------
        length : int
        """
        raise AbstractMethodError(self)

    def __iter__(self) -> Iterator[Any]:
        """
        Iterate over elements of the array.
        """
        # This needs to be implemented so that pandas recognizes extension
        # arrays as list-like. The default implementation makes successive
        # calls to ``__getitem__``, which may be slower than necessary.
        for i in range(len(self)):
            yield self[i]

    def __contains__(self, item: object) -> bool | np.bool_:
        """
        Return for `item in self`.
        """
        # GH37867
        # comparisons of any item to pd.NA always return pd.NA, so e.g. "a" in [pd.NA]
        # would raise a TypeError. The implementation below works around that.
        if is_scalar(item) and isna(item):
            if not self._can_hold_na:
                return False
            elif item is self.dtype.na_value or isinstance(item, self.dtype.type):
                return self._hasna
            else:
                return False
        else:
            # error: Item "ExtensionArray" of "Union[ExtensionArray, ndarray]" has no
            # attribute "any"
            return (item == self).any()  # type: ignore[union-attr]

    # error: Signature of "__eq__" incompatible with supertype "object"
    def __eq__(self, other: object) -> ArrayLike:  # type: ignore[override]
        """
        Return for `self == other` (element-wise equality).
        """
        # Implementer note: this should return a boolean numpy ndarray or
        # a boolean ExtensionArray.
        # When `other` is one of Series, Index, or DataFrame, this method should
        # return NotImplemented (to ensure that those objects are responsible for
        # first unpacking the arrays, and then dispatch the operation to the
        # underlying arrays)
        raise AbstractMethodError(self)

    # error: Signature of "__ne__" incompatible with supertype "object"
    def __ne__(self, other: object) -> ArrayLike:  # type: ignore[override]
        """
        Return for `self != other` (element-wise in-equality).
        """
        # error: Unsupported operand type for ~ ("ExtensionArray")
        return ~(self == other)  # type: ignore[operator]

    def to_numpy(
        self,
        dtype: npt.DTypeLike | None = None,
        copy: bool = False,
        na_value: object = lib.no_default,
    ) -> np.ndarray:
        """
        Convert to a NumPy ndarray.

        This is similar to :meth:`numpy.asarray`, but may provide additional control
        over how the conversion is done.

        Parameters
        ----------
        dtype : str or numpy.dtype, optional
            The dtype to pass to :meth:`numpy.asarray`.
        copy : bool, default False
            Whether to ensure that the returned value is a not a view on
            another array. Note that ``copy=False`` does not *ensure* that
            ``to_numpy()`` is no-copy. Rather, ``copy=True`` ensure that
            a copy is made, even if not strictly necessary.
        na_value : Any, optional
            The value to use for missing values. The default value depends
            on `dtype` and the type of the array.

        Returns
        -------
        numpy.ndarray
        """
        result = np.asarray(self, dtype=dtype)
        if copy or na_value is not lib.no_default:
            result = result.copy()
        if na_value is not lib.no_default:
            result[self.isna()] = na_value
        return result

    # ------------------------------------------------------------------------
    # Required attributes
    # ------------------------------------------------------------------------

    @property
    def dtype(self) -> ExtensionDtype:
        """
        An instance of ExtensionDtype.

        Examples
        --------
        >>> pd.array([1, 2, 3]).dtype
        Int64Dtype()
        """
        raise AbstractMethodError(self)

    @property
    def shape(self) -> Shape:
        """
        Return a tuple of the array dimensions.

        Examples
        --------
        >>> arr = pd.array([1, 2, 3])
        >>> arr.shape
        (3,)
        """
        return (len(self),)

    @property
    def size(self) -> int:
        """
        The number of elements in the array.
        """
        # error: Incompatible return value type (got "signedinteger[_64Bit]",
        # expected "int")  [return-value]
        return np.prod(self.shape)  # type: ignore[return-value]

    @property
    def ndim(self) -> int:
        """
        Extension Arrays are only allowed to be 1-dimensional.

        Examples
        --------
        >>> arr = pd.array([1, 2, 3])
        >>> arr.ndim
        1
        """
        return 1

    @property
    def nbytes(self) -> int:
        """
        The number of bytes needed to store this object in memory.

        Examples
        --------
        >>> pd.array([1, 2, 3]).nbytes
        27
        """
        # If this is expensive to compute, return an approximate lower bound
        # on the number of bytes needed.
        raise AbstractMethodError(self)

    # ------------------------------------------------------------------------
    # Additional Methods
    # ------------------------------------------------------------------------

    @overload
    def astype(self, dtype: npt.DTypeLike, copy: bool = ...) -> np.ndarray:
        ...

    @overload
    def astype(self, dtype: ExtensionDtype, copy: bool = ...) -> ExtensionArray:
        ...

    @overload
    def astype(self, dtype: AstypeArg, copy: bool = ...) -> ArrayLike:
        ...

    def astype(self, dtype: AstypeArg, copy: bool = True) -> ArrayLike:
        """
        Cast to a NumPy array or ExtensionArray with 'dtype'.

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
        np.ndarray or pandas.api.extensions.ExtensionArray
            An ``ExtensionArray`` if ``dtype`` is ``ExtensionDtype``,
            otherwise a Numpy ndarray with ``dtype`` for its dtype.

        Examples
        --------
        >>> arr = pd.array([1, 2, 3])
        >>> arr
        <IntegerArray>
        [1, 2, 3]
        Length: 3, dtype: Int64

        Casting to another ``ExtensionDtype`` returns an ``ExtensionArray``:

        >>> arr1 = arr.astype('Float64')
        >>> arr1
        <FloatingArray>
        [1.0, 2.0, 3.0]
        Length: 3, dtype: Float64
        >>> arr1.dtype
        Float64Dtype()

        Otherwise, we will get a Numpy ndarray:

        >>> arr2 = arr.astype('float64')
        >>> arr2
        array([1., 2., 3.])
        >>> arr2.dtype
        dtype('float64')
        """
        dtype = pandas_dtype(dtype)
        if dtype == self.dtype:
            if not copy:
                return self
            else:
                return self.copy()

        if isinstance(dtype, ExtensionDtype):
            cls = dtype.construct_array_type()
            return cls._from_sequence(self, dtype=dtype, copy=copy)

        elif lib.is_np_dtype(dtype, "M"):
            from pandas.core.arrays import DatetimeArray

            return DatetimeArray._from_sequence(self, dtype=dtype, copy=copy)

        elif lib.is_np_dtype(dtype, "m"):
            from pandas.core.arrays import TimedeltaArray

            return TimedeltaArray._from_sequence(self, dtype=dtype, copy=copy)

        if not copy:
            return np.asarray(self, dtype=dtype)
        else:
            return np.array(self, dtype=dtype, copy=copy)

    def isna(self) -> np.ndarray | ExtensionArraySupportsAnyAll:
        """
        A 1-D array indicating if each value is missing.

        Returns
        -------
        numpy.ndarray or pandas.api.extensions.ExtensionArray
            In most cases, this should return a NumPy ndarray. For
            exceptional cases like ``SparseArray``, where returning
            an ndarray would be expensive, an ExtensionArray may be
            returned.

        Notes
        -----
        If returning an ExtensionArray, then

        * ``na_values._is_boolean`` should be True
        * `na_values` should implement :func:`ExtensionArray._reduce`
        * ``na_values.any`` and ``na_values.all`` should be implemented

        Examples
        --------
        >>> arr = pd.array([1, 2, np.nan, np.nan])
        >>> arr.isna()
        array([False, False,  True,  True])
        """
        raise AbstractMethodError(self)

    @property
    def _hasna(self) -> bool:
        # GH#22680
        """
        Equivalent to `self.isna().any()`.

        Some ExtensionArray subclasses may be able to optimize this check.
        """
        return bool(self.isna().any())

    def _values_for_argsort(self) -> np.ndarray:
        """
        Return values for sorting.

        Returns
        -------
        ndarray
            The transformed values should maintain the ordering between values
            within the array.

        See Also
        --------
        ExtensionArray.argsort : Return the indices that would sort this array.

        Notes
        -----
        The caller is responsible for *not* modifying these values in-place, so
        it is safe for implementers to give views on ``self``.

        Functions that use this (e.g. ``ExtensionArray.argsort``) should ignore
        entries with missing values in the original array (according to
        ``self.isna()``). This means that the corresponding entries in the returned
        array don't need to be modified to sort correctly.

        Examples
        --------
        In most cases, this is the underlying Numpy array of the ``ExtensionArray``:

        >>> arr = pd.array([1, 2, 3])
        >>> arr._values_for_argsort()
        array([1, 2, 3])
        """
        # Note: this is used in `ExtensionArray.argsort/argmin/argmax`.
        return np.array(self)

    def argsort(
        self,
        *,
        ascending: bool = True,
        kind: SortKind = "quicksort",
        na_position: str = "last",
        **kwargs,
    ) -> np.ndarray:
        """
        Return the indices that would sort this array.

        Parameters
        ----------
        ascending : bool, default True
            Whether the indices should result in an ascending
            or descending sort.
        kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
            Sorting algorithm.
        na_position : {'first', 'last'}, default 'last'
            If ``'first'``, put ``NaN`` values at the beginning.
            If ``'last'``, put ``NaN`` values at the end.
        *args, **kwargs:
            Passed through to :func:`numpy.argsort`.

        Returns
        -------
        np.ndarray[np.intp]
            Array of indices that sort ``self``. If NaN values are contained,
            NaN values are placed at the end.

        See Also
        --------
        numpy.argsort : Sorting implementation used internally.

        Examples
        --------
        >>> arr = pd.array([3, 1, 2, 5, 4])
        >>> arr.argsort()
        array([1, 2, 0, 4, 3])
        """
        # Implementer note: You have two places to override the behavior of
        # argsort.
        # 1. _values_for_argsort : construct the values passed to np.argsort
        # 2. argsort : total control over sorting. In case of overriding this,
        #    it is recommended to also override argmax/argmin
        ascending = nv.validate_argsort_with_ascending(ascending, (), kwargs)

        values = self._values_for_argsort()
        return nargsort(
            values,
            kind=kind,
            ascending=ascending,
            na_position=na_position,
            mask=np.asarray(self.isna()),
        )

    def argmin(self, skipna: bool = True) -> int:
        """
        Return the index of minimum value.

        In case of multiple occurrences of the minimum value, the index
        corresponding to the first occurrence is returned.

        Parameters
        ----------
        skipna : bool, default True

        Returns
        -------
        int

        See Also
        --------
        ExtensionArray.argmax : Return the index of the maximum value.

        Examples
        --------
        >>> arr = pd.array([3, 1, 2, 5, 4])
        >>> arr.argmin()
        1
        """
        # Implementer note: You have two places to override the behavior of
        # argmin.
        # 1. _values_for_argsort : construct the values used in nargminmax
        # 2. argmin itself : total control over sorting.
        validate_bool_kwarg(skipna, "skipna")
        if not skipna and self._hasna:
            raise NotImplementedError
        return nargminmax(self, "argmin")

    def argmax(self, skipna: bool = True) -> int:
        """
        Return the index of maximum value.

        In case of multiple occurrences of the maximum value, the index
        corresponding to the first occurrence is returned.

        Parameters
        ----------
        skipna : bool, default True

        Returns
        -------
        int

        See Also
        --------
        ExtensionArray.argmin : Return the index of the minimum value.

        Examples
        --------
        >>> arr = pd.array([3, 1, 2, 5, 4])
        >>> arr.argmax()
        3
        """
        # Implementer note: You have two places to override the behavior of
        # argmax.
        # 1. _values_for_argsort : construct the values used in nargminmax
        # 2. argmax itself : total control over sorting.
        validate_bool_kwarg(skipna, "skipna")
        if not skipna and self._hasna:
            raise NotImplementedError
        return nargminmax(self, "argmax")

    def interpolate(
        self,
        *,
        method: InterpolateOptions,
        axis: int,
        index: Index,
        limit,
        limit_direction,
        limit_area,
        copy: bool,
        **kwargs,
    ) -> Self:
        """
        See DataFrame.interpolate.__doc__.

        Examples
        --------
        >>> arr = pd.arrays.NumpyExtensionArray(np.array([0, 1, np.nan, 3]))
        >>> arr.interpolate(method="linear",
        ...                 limit=3,
        ...                 limit_direction="forward",
        ...                 index=pd.Index([1, 2, 3, 4]),
        ...                 fill_value=1,
        ...                 copy=False,
        ...                 axis=0,
        ...                 limit_area="inside"
        ...                 )
        <NumpyExtensionArray>
        [0.0, 1.0, 2.0, 3.0]
        Length: 4, dtype: float64
        """
        # NB: we return type(self) even if copy=False
        raise NotImplementedError(
            f"{type(self).__name__} does not implement interpolate"
        )

    def _pad_or_backfill(
        self,
        *,
        method: FillnaOptions,
        limit: int | None = None,
        limit_area: Literal["inside", "outside"] | None = None,
        copy: bool = True,
    ) -> Self:
        """
        Pad or backfill values, used by Series/DataFrame ffill and bfill.

        Parameters
        ----------
        method : {'backfill', 'bfill', 'pad', 'ffill'}
            Method to use for filling holes in reindexed Series:

            * pad / ffill: propagate last valid observation forward to next valid.
            * backfill / bfill: use NEXT valid observation to fill gap.

        limit : int, default None
            This is the maximum number of consecutive
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
            The base class implementation ignores the keyword if any NAs are
            present.

        Returns
        -------
        Same type as self

        Examples
        --------
        >>> arr = pd.array([np.nan, np.nan, 2, 3, np.nan, np.nan])
        >>> arr._pad_or_backfill(method="backfill", limit=1)
        <IntegerArray>
        [<NA>, 2, 2, 3, <NA>, <NA>]
        Length: 6, dtype: Int64
        """

        # If a 3rd-party EA has implemented this functionality in fillna,
        #  we warn that they need to implement _pad_or_backfill instead.
        if (
            type(self).fillna is not ExtensionArray.fillna
            and type(self)._pad_or_backfill is ExtensionArray._pad_or_backfill
        ):
            # Check for _pad_or_backfill here allows us to call
            #  super()._pad_or_backfill without getting this warning
            warnings.warn(
                "ExtensionArray.fillna 'method' keyword is deprecated. "
                "In a future version. arr._pad_or_backfill will be called "
                "instead. 3rd-party ExtensionArray authors need to implement "
                "_pad_or_backfill.",
                DeprecationWarning,
                stacklevel=find_stack_level(),
            )
            if limit_area is not None:
                raise NotImplementedError(
                    f"{type(self).__name__} does not implement limit_area "
                    "(added in pandas 2.2). 3rd-party ExtnsionArray authors "
                    "need to add this argument to _pad_or_backfill."
                )
            return self.fillna(method=method, limit=limit)

        mask = self.isna()

        if mask.any():
            # NB: the base class does not respect the "copy" keyword
            meth = missing.clean_fill_method(method)

            npmask = np.asarray(mask)
            if limit_area is not None and not npmask.all():
                _fill_limit_area_1d(npmask, limit_area)
            if meth == "pad":
                indexer = libalgos.get_fill_indexer(npmask, limit=limit)
                return self.take(indexer, allow_fill=True)
            else:
                # i.e. meth == "backfill"
                indexer = libalgos.get_fill_indexer(npmask[::-1], limit=limit)[::-1]
                return self[::-1].take(indexer, allow_fill=True)

        else:
            if not copy:
                return self
            new_values = self.copy()
        return new_values

    def fillna(
        self,
        value: object | ArrayLike | None = None,
        method: FillnaOptions | None = None,
        limit: int | None = None,
        copy: bool = True,
    ) -> Self:
        """
        Fill NA/NaN values using the specified method.

        Parameters
        ----------
        value : scalar, array-like
            If a scalar value is passed it is used to fill all missing values.
            Alternatively, an array-like "value" can be given. It's expected
            that the array-like have the same length as 'self'.
        method : {'backfill', 'bfill', 'pad', 'ffill', None}, default None
            Method to use for filling holes in reindexed Series:

            * pad / ffill: propagate last valid observation forward to next valid.
            * backfill / bfill: use NEXT valid observation to fill gap.

            .. deprecated:: 2.1.0

        limit : int, default None
            If method is specified, this is the maximum number of consecutive
            NaN values to forward/backward fill. In other words, if there is
            a gap with more than this number of consecutive NaNs, it will only
            be partially filled. If method is not specified, this is the
            maximum number of entries along the entire axis where NaNs will be
            filled.

            .. deprecated:: 2.1.0

        copy : bool, default True
            Whether to make a copy of the data before filling. If False, then
            the original should be modified and no new memory should be allocated.
            For ExtensionArray subclasses that cannot do this, it is at the
            author's discretion whether to ignore "copy=False" or to raise.
            The base class implementation ignores the keyword in pad/backfill
            cases.

        Returns
        -------
        ExtensionArray
            With NA/NaN filled.

        Examples
        --------
        >>> arr = pd.array([np.nan, np.nan, 2, 3, np.nan, np.nan])
        >>> arr.fillna(0)
        <IntegerArray>
        [0, 0, 2, 3, 0, 0]
        Length: 6, dtype: Int64
        """
        if method is not None:
            warnings.warn(
                f"The 'method' keyword in {type(self).__name__}.fillna is "
                "deprecated and will be removed in a future version.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )

        value, method = validate_fillna_kwargs(value, method)

        mask = self.isna()
        # error: Argument 2 to "check_value_size" has incompatible type
        # "ExtensionArray"; expected "ndarray"
        value = missing.check_value_size(
            value, mask, len(self)  # type: ignore[arg-type]
        )

        if mask.any():
            if method is not None:
                meth = missing.clean_fill_method(method)

                npmask = np.asarray(mask)
                if meth == "pad":
                    indexer = libalgos.get_fill_indexer(npmask, limit=limit)
                    return self.take(indexer, allow_fill=True)
                else:
                    # i.e. meth == "backfill"
                    indexer = libalgos.get_fill_indexer(npmask[::-1], limit=limit)[::-1]
                    return self[::-1].take(indexer, allow_fill=True)
            else:
                # fill with value
                if not copy:
                    new_values = self[:]
                else:
                    new_values = self.copy()
                new_values[mask] = value
        else:
            if not copy:
                new_values = self[:]
            else:
                new_values = self.copy()
        return new_values

    def dropna(self) -> Self:
        """
        Return ExtensionArray without NA values.

        Returns
        -------

        Examples
        --------
        >>> pd.array([1, 2, np.nan]).dropna()
        <IntegerArray>
        [1, 2]
        Length: 2, dtype: Int64
        """
        # error: Unsupported operand type for ~ ("ExtensionArray")
        return self[~self.isna()]  # type: ignore[operator]

    def duplicated(
        self, keep: Literal["first", "last", False] = "first"
    ) -> npt.NDArray[np.bool_]:
        """
        Return boolean ndarray denoting duplicate values.

        Parameters
        ----------
        keep : {'first', 'last', False}, default 'first'
            - ``first`` : Mark duplicates as ``True`` except for the first occurrence.
            - ``last`` : Mark duplicates as ``True`` except for the last occurrence.
            - False : Mark all duplicates as ``True``.

        Returns
        -------
        ndarray[bool]

        Examples
        --------
        >>> pd.array([1, 1, 2, 3, 3], dtype="Int64").duplicated()
        array([False,  True, False, False,  True])
        """
        mask = self.isna().astype(np.bool_, copy=False)
        return duplicated(values=self, keep=keep, mask=mask)

    def shift(self, periods: int = 1, fill_value: object = None) -> ExtensionArray:
        """
        Shift values by desired number.

        Newly introduced missing values are filled with
        ``self.dtype.na_value``.

        Parameters
        ----------
        periods : int, default 1
            The number of periods to shift. Negative values are allowed
            for shifting backwards.

        fill_value : object, optional
            The scalar value to use for newly introduced missing values.
            The default is ``self.dtype.na_value``.

        Returns
        -------
        ExtensionArray
            Shifted.

        Notes
        -----
        If ``self`` is empty or ``periods`` is 0, a copy of ``self`` is
        returned.

        If ``periods > len(self)``, then an array of size
        len(self) is returned, with all values filled with
        ``self.dtype.na_value``.

        For 2-dimensional ExtensionArrays, we are always shifting along axis=0.

        Examples
        --------
        >>> arr = pd.array([1, 2, 3])
        >>> arr.shift(2)
        <IntegerArray>
        [<NA>, <NA>, 1]
        Length: 3, dtype: Int64
        """
        # Note: this implementation assumes that `self.dtype.na_value` can be
        # stored in an instance of your ExtensionArray with `self.dtype`.
        if not len(self) or periods == 0:
            return self.copy()

        if isna(fill_value):
            fill_value = self.dtype.na_value

        empty = self._from_sequence(
            [fill_value] * min(abs(periods), len(self)), dtype=self.dtype
        )
        if periods > 0:
            a = empty
            b = self[:-periods]
        else:
            a = self[abs(periods) :]
            b = empty
        return self._concat_same_type([a, b])

    def unique(self) -> Self:
        """
        Compute the ExtensionArray of unique values.

        Returns
        -------
        pandas.api.extensions.ExtensionArray

        Examples
        --------
        >>> arr = pd.array([1, 2, 3, 1, 2, 3])
        >>> arr.unique()
        <IntegerArray>
        [1, 2, 3]
        Length: 3, dtype: Int64
        """
        uniques = unique(self.astype(object))
        return self._from_sequence(uniques, dtype=self.dtype)

    def searchsorted(
        self,
        value: NumpyValueArrayLike | ExtensionArray,
        side: Literal["left", "right"] = "left",
        sorter: NumpySorter | None = None,
    ) -> npt.NDArray[np.intp] | np.intp:
        """
        Find indices where elements should be inserted to maintain order.

        Find the indices into a sorted array `self` (a) such that, if the
        corresponding elements in `value` were inserted before the indices,
        the order of `self` would be preserved.

        Assuming that `self` is sorted:

        ======  ================================
        `side`  returned index `i` satisfies
        ======  ================================
        left    ``self[i-1] < value <= self[i]``
        right   ``self[i-1] <= value < self[i]``
        ======  ================================

        Parameters
        ----------
        value : array-like, list or scalar
            Value(s) to insert into `self`.
        side : {'left', 'right'}, optional
            If 'left', the index of the first suitable location found is given.
            If 'right', return the last such index.  If there is no suitable
            index, return either 0 or N (where N is the length of `self`).
        sorter : 1-D array-like, optional
            Optional array of integer indices that sort array a into ascending
            order. They are typically the result of argsort.

        Returns
        -------
        array of ints or int
            If value is array-like, array of insertion points.
            If value is scalar, a single integer.

        See Also
        --------
        numpy.searchsorted : Similar method from NumPy.

        Examples
        --------
        >>> arr = pd.array([1, 2, 3, 5])
        >>> arr.searchsorted([4])
        array([3])
        """
        # Note: the base tests provided by pandas only test the basics.
        # We do not test
        # 1. Values outside the range of the `data_for_sorting` fixture
        # 2. Values between the values in the `data_for_sorting` fixture
        # 3. Missing values.
        arr = self.astype(object)
        if isinstance(value, ExtensionArray):
            value = value.astype(object)
        return arr.searchsorted(value, side=side, sorter=sorter)

    def equals(self, other: object) -> bool:
        """
        Return if another array is equivalent to this array.

        Equivalent means that both arrays have the same shape and dtype, and
        all values compare equal. Missing values in the same location are
        considered equal (in contrast with normal equality).

        Parameters
        ----------
        other : ExtensionArray
            Array to compare to this Array.

        Returns
        -------
        boolean
            Whether the arrays are equivalent.

        Examples
        --------
        >>> arr1 = pd.array([1, 2, np.nan])
        >>> arr2 = pd.array([1, 2, np.nan])
        >>> arr1.equals(arr2)
        True
        """
        if type(self) != type(other):
            return False
        other = cast(ExtensionArray, other)
        if self.dtype != other.dtype:
            return False
        elif len(self) != len(other):
            return False
        else:
            equal_values = self == other
            if isinstance(equal_values, ExtensionArray):
                # boolean array with NA -> fill with False
                equal_values = equal_values.fillna(False)
            # error: Unsupported left operand type for & ("ExtensionArray")
            equal_na = self.isna() & other.isna()  # type: ignore[operator]
            return bool((equal_values | equal_na).all())

    def isin(self, values: ArrayLike) -> npt.NDArray[np.bool_]:
        """
        Pointwise comparison for set containment in the given values.

        Roughly equivalent to `np.array([x in values for x in self])`

        Parameters
        ----------
        values : np.ndarray or ExtensionArray

        Returns
        -------
        np.ndarray[bool]

        Examples
        --------
        >>> arr = pd.array([1, 2, 3])
        >>> arr.isin([1])
        <BooleanArray>
        [True, False, False]
        Length: 3, dtype: boolean
        """
        return isin(np.asarray(self), values)

    def _values_for_factorize(self) -> tuple[np.ndarray, Any]:
        """
        Return an array and missing value suitable for factorization.

        Returns
        -------
        values : ndarray
            An array suitable for factorization. This should maintain order
            and be a supported dtype (Float64, Int64, UInt64, String, Object).
            By default, the extension array is cast to object dtype.
        na_value : object
            The value in `values` to consider missing. This will be treated
            as NA in the factorization routines, so it will be coded as
            `-1` and not included in `uniques`. By default,
            ``np.nan`` is used.

        Notes
        -----
        The values returned by this method are also used in
        :func:`pandas.util.hash_pandas_object`. If needed, this can be
        overridden in the ``self._hash_pandas_object()`` method.

        Examples
        --------
        >>> pd.array([1, 2, 3])._values_for_factorize()
        (array([1, 2, 3], dtype=object), nan)
        """
        return self.astype(object), np.nan

    def factorize(
        self,
        use_na_sentinel: bool = True,
    ) -> tuple[np.ndarray, ExtensionArray]:
        """
        Encode the extension array as an enumerated type.

        Parameters
        ----------
        use_na_sentinel : bool, default True
            If True, the sentinel -1 will be used for NaN values. If False,
