            NaN values will be encoded as non-negative integers and will not drop the
            NaN from the uniques of the values.

            .. versionadded:: 1.5.0

        Returns
        -------
        codes : ndarray
            An integer NumPy array that's an indexer into the original
            ExtensionArray.
        uniques : ExtensionArray
            An ExtensionArray containing the unique values of `self`.

            .. note::

               uniques will *not* contain an entry for the NA value of
               the ExtensionArray if there are any missing values present
               in `self`.

        See Also
        --------
        factorize : Top-level factorize method that dispatches here.

        Notes
        -----
        :meth:`pandas.factorize` offers a `sort` keyword as well.

        Examples
        --------
        >>> idx1 = pd.PeriodIndex(["2014-01", "2014-01", "2014-02", "2014-02",
        ...                       "2014-03", "2014-03"], freq="M")
        >>> arr, idx = idx1.factorize()
        >>> arr
        array([0, 0, 1, 1, 2, 2])
        >>> idx
        PeriodIndex(['2014-01', '2014-02', '2014-03'], dtype='period[M]')
        """
        # Implementer note: There are two ways to override the behavior of
        # pandas.factorize
        # 1. _values_for_factorize and _from_factorize.
        #    Specify the values passed to pandas' internal factorization
        #    routines, and how to convert from those values back to the
        #    original ExtensionArray.
        # 2. ExtensionArray.factorize.
        #    Complete control over factorization.
        arr, na_value = self._values_for_factorize()

        codes, uniques = factorize_array(
            arr, use_na_sentinel=use_na_sentinel, na_value=na_value
        )

        uniques_ea = self._from_factorized(uniques, self)
        return codes, uniques_ea

    _extension_array_shared_docs[
        "repeat"
    ] = """
        Repeat elements of a %(klass)s.

        Returns a new %(klass)s where each element of the current %(klass)s
        is repeated consecutively a given number of times.

        Parameters
        ----------
        repeats : int or array of ints
            The number of repetitions for each element. This should be a
            non-negative integer. Repeating 0 times will return an empty
            %(klass)s.
        axis : None
            Must be ``None``. Has no effect but is accepted for compatibility
            with numpy.

        Returns
        -------
        %(klass)s
            Newly created %(klass)s with repeated elements.

        See Also
        --------
        Series.repeat : Equivalent function for Series.
        Index.repeat : Equivalent function for Index.
        numpy.repeat : Similar method for :class:`numpy.ndarray`.
        ExtensionArray.take : Take arbitrary positions.

        Examples
        --------
        >>> cat = pd.Categorical(['a', 'b', 'c'])
        >>> cat
        ['a', 'b', 'c']
        Categories (3, object): ['a', 'b', 'c']
        >>> cat.repeat(2)
        ['a', 'a', 'b', 'b', 'c', 'c']
        Categories (3, object): ['a', 'b', 'c']
        >>> cat.repeat([1, 2, 3])
        ['a', 'b', 'b', 'c', 'c', 'c']
        Categories (3, object): ['a', 'b', 'c']
        """

    @Substitution(klass="ExtensionArray")
    @Appender(_extension_array_shared_docs["repeat"])
    def repeat(self, repeats: int | Sequence[int], axis: AxisInt | None = None) -> Self:
        nv.validate_repeat((), {"axis": axis})
        ind = np.arange(len(self)).repeat(repeats)
        return self.take(ind)

    # ------------------------------------------------------------------------
    # Indexing methods
    # ------------------------------------------------------------------------

    def take(
        self,
        indices: TakeIndexer,
        *,
        allow_fill: bool = False,
        fill_value: Any = None,
    ) -> Self:
        """
        Take elements from an array.

        Parameters
        ----------
        indices : sequence of int or one-dimensional np.ndarray of int
            Indices to be taken.
        allow_fill : bool, default False
            How to handle negative values in `indices`.

            * False: negative values in `indices` indicate positional indices
              from the right (the default). This is similar to
              :func:`numpy.take`.

            * True: negative values in `indices` indicate
              missing values. These values are set to `fill_value`. Any other
              other negative values raise a ``ValueError``.

        fill_value : any, optional
            Fill value to use for NA-indices when `allow_fill` is True.
            This may be ``None``, in which case the default NA value for
            the type, ``self.dtype.na_value``, is used.

            For many ExtensionArrays, there will be two representations of
            `fill_value`: a user-facing "boxed" scalar, and a low-level
            physical NA value. `fill_value` should be the user-facing version,
            and the implementation should handle translating that to the
            physical version for processing the take if necessary.

        Returns
        -------
        ExtensionArray

        Raises
        ------
        IndexError
            When the indices are out of bounds for the array.
        ValueError
            When `indices` contains negative values other than ``-1``
            and `allow_fill` is True.

        See Also
        --------
        numpy.take : Take elements from an array along an axis.
        api.extensions.take : Take elements from an array.

        Notes
        -----
        ExtensionArray.take is called by ``Series.__getitem__``, ``.loc``,
        ``iloc``, when `indices` is a sequence of values. Additionally,
        it's called by :meth:`Series.reindex`, or any other method
        that causes realignment, with a `fill_value`.

        Examples
        --------
        Here's an example implementation, which relies on casting the
        extension array to object dtype. This uses the helper method
        :func:`pandas.api.extensions.take`.

        .. code-block:: python

           def take(self, indices, allow_fill=False, fill_value=None):
               from pandas.core.algorithms import take

               # If the ExtensionArray is backed by an ndarray, then
               # just pass that here instead of coercing to object.
               data = self.astype(object)

               if allow_fill and fill_value is None:
                   fill_value = self.dtype.na_value

               # fill value should always be translated from the scalar
               # type for the array, to the physical storage type for
               # the data, before passing to take.

               result = take(data, indices, fill_value=fill_value,
                             allow_fill=allow_fill)
               return self._from_sequence(result, dtype=self.dtype)
        """
        # Implementer note: The `fill_value` parameter should be a user-facing
        # value, an instance of self.dtype.type. When passed `fill_value=None`,
        # the default of `self.dtype.na_value` should be used.
        # This may differ from the physical storage type your ExtensionArray
        # uses. In this case, your implementation is responsible for casting
        # the user-facing type to the storage type, before using
        # pandas.api.extensions.take
        raise AbstractMethodError(self)

    def copy(self) -> Self:
        """
        Return a copy of the array.

        Returns
        -------
        ExtensionArray

        Examples
        --------
        >>> arr = pd.array([1, 2, 3])
        >>> arr2 = arr.copy()
        >>> arr[0] = 2
        >>> arr2
        <IntegerArray>
        [1, 2, 3]
        Length: 3, dtype: Int64
        """
        raise AbstractMethodError(self)

    def view(self, dtype: Dtype | None = None) -> ArrayLike:
        """
        Return a view on the array.

        Parameters
        ----------
        dtype : str, np.dtype, or ExtensionDtype, optional
            Default None.

        Returns
        -------
        ExtensionArray or np.ndarray
            A view on the :class:`ExtensionArray`'s data.

        Examples
        --------
        This gives view on the underlying data of an ``ExtensionArray`` and is not a
        copy. Modifications on either the view or the original ``ExtensionArray``
        will be reflectd on the underlying data:

        >>> arr = pd.array([1, 2, 3])
        >>> arr2 = arr.view()
        >>> arr[0] = 2
        >>> arr2
        <IntegerArray>
        [2, 2, 3]
        Length: 3, dtype: Int64
        """
        # NB:
        # - This must return a *new* object referencing the same data, not self.
        # - The only case that *must* be implemented is with dtype=None,
        #   giving a view with the same dtype as self.
        if dtype is not None:
            raise NotImplementedError(dtype)
        return self[:]

    # ------------------------------------------------------------------------
    # Printing
    # ------------------------------------------------------------------------

    def __repr__(self) -> str:
        if self.ndim > 1:
            return self._repr_2d()

        from pandas.io.formats.printing import format_object_summary

        # the short repr has no trailing newline, while the truncated
        # repr does. So we include a newline in our template, and strip
        # any trailing newlines from format_object_summary
        data = format_object_summary(
            self, self._formatter(), indent_for_name=False
        ).rstrip(", \n")
        class_name = f"<{type(self).__name__}>\n"
        footer = self._get_repr_footer()
        return f"{class_name}{data}\n{footer}"

    def _get_repr_footer(self) -> str:
        # GH#24278
        if self.ndim > 1:
            return f"Shape: {self.shape}, dtype: {self.dtype}"
        return f"Length: {len(self)}, dtype: {self.dtype}"

    def _repr_2d(self) -> str:
        from pandas.io.formats.printing import format_object_summary

        # the short repr has no trailing newline, while the truncated
        # repr does. So we include a newline in our template, and strip
        # any trailing newlines from format_object_summary
        lines = [
            format_object_summary(x, self._formatter(), indent_for_name=False).rstrip(
                ", \n"
            )
            for x in self
        ]
        data = ",\n".join(lines)
        class_name = f"<{type(self).__name__}>"
        footer = self._get_repr_footer()
        return f"{class_name}\n[\n{data}\n]\n{footer}"

    def _formatter(self, boxed: bool = False) -> Callable[[Any], str | None]:
        """
        Formatting function for scalar values.

        This is used in the default '__repr__'. The returned formatting
        function receives instances of your scalar type.

        Parameters
        ----------
        boxed : bool, default False
            An indicated for whether or not your array is being printed
            within a Series, DataFrame, or Index (True), or just by
            itself (False). This may be useful if you want scalar values
            to appear differently within a Series versus on its own (e.g.
            quoted or not).

        Returns
        -------
        Callable[[Any], str]
            A callable that gets instances of the scalar type and
            returns a string. By default, :func:`repr` is used
            when ``boxed=False`` and :func:`str` is used when
            ``boxed=True``.

        Examples
        --------
        >>> class MyExtensionArray(pd.arrays.NumpyExtensionArray):
        ...     def _formatter(self, boxed=False):
        ...         return lambda x: '*' + str(x) + '*' if boxed else repr(x) + '*'
        >>> MyExtensionArray(np.array([1, 2, 3, 4]))
        <MyExtensionArray>
        [1*, 2*, 3*, 4*]
        Length: 4, dtype: int64
        """
        if boxed:
            return str
        return repr

    # ------------------------------------------------------------------------
    # Reshaping
    # ------------------------------------------------------------------------

    def transpose(self, *axes: int) -> ExtensionArray:
        """
        Return a transposed view on this array.

        Because ExtensionArrays are always 1D, this is a no-op.  It is included
        for compatibility with np.ndarray.

        Returns
        -------
        ExtensionArray

        Examples
        --------
        >>> pd.array([1, 2, 3]).transpose()
        <IntegerArray>
        [1, 2, 3]
        Length: 3, dtype: Int64
        """
        return self[:]

    @property
    def T(self) -> ExtensionArray:
        return self.transpose()

    def ravel(self, order: Literal["C", "F", "A", "K"] | None = "C") -> ExtensionArray:
        """
        Return a flattened view on this array.

        Parameters
        ----------
        order : {None, 'C', 'F', 'A', 'K'}, default 'C'

        Returns
        -------
        ExtensionArray

        Notes
        -----
        - Because ExtensionArrays are 1D-only, this is a no-op.
        - The "order" argument is ignored, is for compatibility with NumPy.

        Examples
        --------
        >>> pd.array([1, 2, 3]).ravel()
        <IntegerArray>
        [1, 2, 3]
        Length: 3, dtype: Int64
        """
        return self

    @classmethod
    def _concat_same_type(cls, to_concat: Sequence[Self]) -> Self:
        """
        Concatenate multiple array of this dtype.

        Parameters
        ----------
        to_concat : sequence of this type

        Returns
        -------
        ExtensionArray

        Examples
        --------
        >>> arr1 = pd.array([1, 2, 3])
        >>> arr2 = pd.array([4, 5, 6])
        >>> pd.arrays.IntegerArray._concat_same_type([arr1, arr2])
        <IntegerArray>
        [1, 2, 3, 4, 5, 6]
        Length: 6, dtype: Int64
        """
        # Implementer note: this method will only be called with a sequence of
        # ExtensionArrays of this class and with the same dtype as self. This
        # should allow "easy" concatenation (no upcasting needed), and result
        # in a new ExtensionArray of the same dtype.
        # Note: this strict behaviour is only guaranteed starting with pandas 1.1
        raise AbstractMethodError(cls)

    # The _can_hold_na attribute is set to True so that pandas internals
    # will use the ExtensionDtype.na_value as the NA value in operations
    # such as take(), reindex(), shift(), etc.  In addition, those results
    # will then be of the ExtensionArray subclass rather than an array
    # of objects
    @cache_readonly
    def _can_hold_na(self) -> bool:
        return self.dtype._can_hold_na

    def _accumulate(
        self, name: str, *, skipna: bool = True, **kwargs
    ) -> ExtensionArray:
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

        Examples
        --------
        >>> arr = pd.array([1, 2, 3])
        >>> arr._accumulate(name='cumsum')
        <IntegerArray>
        [1, 3, 6]
        Length: 3, dtype: Int64
        """
        raise NotImplementedError(f"cannot perform {name} with type {self.dtype}")

    def _reduce(
        self, name: str, *, skipna: bool = True, keepdims: bool = False, **kwargs
    ):
        """
        Return a scalar result of performing the reduction operation.

        Parameters
        ----------
        name : str
            Name of the function, supported values are:
            { any, all, min, max, sum, mean, median, prod,
            std, var, sem, kurt, skew }.
        skipna : bool, default True
            If True, skip NaN values.
        keepdims : bool, default False
            If False, a scalar is returned.
            If True, the result has dimension with size one along the reduced axis.

            .. versionadded:: 2.1

               This parameter is not required in the _reduce signature to keep backward
               compatibility, but will become required in the future. If the parameter
               is not found in the method signature, a FutureWarning will be emitted.
        **kwargs
            Additional keyword arguments passed to the reduction function.
            Currently, `ddof` is the only supported kwarg.

        Returns
        -------
        scalar

        Raises
        ------
        TypeError : subclass does not define reductions

        Examples
        --------
        >>> pd.array([1, 2, 3])._reduce("min")
        1
        """
        meth = getattr(self, name, None)
        if meth is None:
            raise TypeError(
                f"'{type(self).__name__}' with dtype {self.dtype} "
                f"does not support reduction '{name}'"
            )
        result = meth(skipna=skipna, **kwargs)
        if keepdims:
            result = np.array([result])

        return result

    # https://github.com/python/typeshed/issues/2148#issuecomment-520783318
    # Incompatible types in assignment (expression has type "None", base class
    # "object" defined the type as "Callable[[object], int]")
    __hash__: ClassVar[None]  # type: ignore[assignment]

    # ------------------------------------------------------------------------
    # Non-Optimized Default Methods; in the case of the private methods here,
    #  these are not guaranteed to be stable across pandas versions.

    def _values_for_json(self) -> np.ndarray:
        """
        Specify how to render our entries in to_json.

        Notes
        -----
        The dtype on the returned ndarray is not restricted, but for non-native
        types that are not specifically handled in objToJSON.c, to_json is
        liable to raise. In these cases, it may be safer to return an ndarray
        of strings.
        """
        return np.asarray(self)

    def _hash_pandas_object(
        self, *, encoding: str, hash_key: str, categorize: bool
    ) -> npt.NDArray[np.uint64]:
        """
        Hook for hash_pandas_object.

        Default is to use the values returned by _values_for_factorize.

        Parameters
        ----------
        encoding : str
            Encoding for data & key when strings.
        hash_key : str
            Hash_key for string key to encode.
        categorize : bool
            Whether to first categorize object arrays before hashing. This is more
            efficient when the array contains duplicate values.

        Returns
        -------
        np.ndarray[uint64]

        Examples
        --------
        >>> pd.array([1, 2])._hash_pandas_object(encoding='utf-8',
        ...                                      hash_key="1000000000000000",
        ...                                      categorize=False
        ...                                      )
        array([ 6238072747940578789, 15839785061582574730], dtype=uint64)
        """
        from pandas.core.util.hashing import hash_array

        values, _ = self._values_for_factorize()
        return hash_array(
            values, encoding=encoding, hash_key=hash_key, categorize=categorize
        )

    def _explode(self) -> tuple[Self, npt.NDArray[np.uint64]]:
        """
        Transform each element of list-like to a row.

        For arrays that do not contain list-like elements the default
        implementation of this method just returns a copy and an array
        of ones (unchanged index).

        Returns
        -------
        ExtensionArray
            Array with the exploded values.
        np.ndarray[uint64]
            The original lengths of each list-like for determining the
            resulting index.

        See Also
        --------
        Series.explode : The method on the ``Series`` object that this
            extension array method is meant to support.

        Examples
        --------
        >>> import pyarrow as pa
        >>> a = pd.array([[1, 2, 3], [4], [5, 6]],
        ...              dtype=pd.ArrowDtype(pa.list_(pa.int64())))
        >>> a._explode()
        (<ArrowExtensionArray>
        [1, 2, 3, 4, 5, 6]
        Length: 6, dtype: int64[pyarrow], array([3, 1, 2], dtype=int32))
        """
        values = self.copy()
        counts = np.ones(shape=(len(self),), dtype=np.uint64)
        return values, counts

    def tolist(self) -> list:
        """
        Return a list of the values.

        These are each a scalar type, which is a Python scalar
        (for str, int, float) or a pandas scalar
        (for Timestamp/Timedelta/Interval/Period)

        Returns
        -------
        list

        Examples
        --------
        >>> arr = pd.array([1, 2, 3])
        >>> arr.tolist()
        [1, 2, 3]
        """
        if self.ndim > 1:
            return [x.tolist() for x in self]
        return list(self)

    def delete(self, loc: PositionalIndexer) -> Self:
        indexer = np.delete(np.arange(len(self)), loc)
        return self.take(indexer)

    def insert(self, loc: int, item) -> Self:
        """
        Insert an item at the given position.

        Parameters
        ----------
        loc : int
        item : scalar-like

        Returns
        -------
        same type as self

        Notes
        -----
        This method should be both type and dtype-preserving.  If the item
        cannot be held in an array of this type/dtype, either ValueError or
        TypeError should be raised.

        The default implementation relies on _from_sequence to raise on invalid
        items.

        Examples
        --------
        >>> arr = pd.array([1, 2, 3])
        >>> arr.insert(2, -1)
        <IntegerArray>
        [1, 2, -1, 3]
        Length: 4, dtype: Int64
        """
        loc = validate_insert_loc(loc, len(self))

        item_arr = type(self)._from_sequence([item], dtype=self.dtype)

        return type(self)._concat_same_type([self[:loc], item_arr, self[loc:]])

    def _putmask(self, mask: npt.NDArray[np.bool_], value) -> None:
        """
        Analogue to np.putmask(self, mask, value)

        Parameters
        ----------
        mask : np.ndarray[bool]
        value : scalar or listlike
            If listlike, must be arraylike with same length as self.

        Returns
        -------
        None

        Notes
        -----
        Unlike np.putmask, we do not repeat listlike values with mismatched length.
        'value' should either be a scalar or an arraylike with the same length
        as self.
        """
        if is_list_like(value):
            val = value[mask]
        else:
            val = value

        self[mask] = val

    def _where(self, mask: npt.NDArray[np.bool_], value) -> Self:
        """
        Analogue to np.where(mask, self, value)

        Parameters
        ----------
        mask : np.ndarray[bool]
        value : scalar or listlike

        Returns
        -------
        same type as self
        """
        result = self.copy()

        if is_list_like(value):
            val = value[~mask]
        else:
            val = value

        result[~mask] = val
        return result

    # TODO(3.0): this can be removed once GH#33302 deprecation is enforced
    def _fill_mask_inplace(
        self, method: str, limit: int | None, mask: npt.NDArray[np.bool_]
    ) -> None:
        """
        Replace values in locations specified by 'mask' using pad or backfill.

        See also
        --------
        ExtensionArray.fillna
        """
        func = missing.get_fill_func(method)
        npvalues = self.astype(object)
        # NB: if we don't copy mask here, it may be altered inplace, which
        #  would mess up the `self[mask] = ...` below.
        func(npvalues, limit=limit, mask=mask.copy())
        new_values = self._from_sequence(npvalues, dtype=self.dtype)
        self[mask] = new_values[mask]

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

        return rank(
            self._values_for_argsort(),
            axis=axis,
            method=method,
            na_option=na_option,
            ascending=ascending,
            pct=pct,
        )

    @classmethod
    def _empty(cls, shape: Shape, dtype: ExtensionDtype):
        """
        Create an ExtensionArray with the given shape and dtype.

        See also
        --------
        ExtensionDtype.empty
            ExtensionDtype.empty is the 'official' public version of this API.
        """
        # Implementer note: while ExtensionDtype.empty is the public way to
        # call this method, it is still required to implement this `_empty`
        # method as well (it is called internally in pandas)
        obj = cls._from_sequence([], dtype=dtype)

        taker = np.broadcast_to(np.intp(-1), shape)
        result = obj.take(taker, allow_fill=True)
        if not isinstance(result, cls) or dtype != result.dtype:
            raise NotImplementedError(
                f"Default 'empty' implementation is invalid for dtype='{dtype}'"
            )
        return result

    def _quantile(self, qs: npt.NDArray[np.float64], interpolation: str) -> Self:
        """
        Compute the quantiles of self for each quantile in `qs`.

        Parameters
        ----------
        qs : np.ndarray[float64]
        interpolation: str

        Returns
        -------
        same type as self
        """
        mask = np.asarray(self.isna())
        arr = np.asarray(self)
        fill_value = np.nan

        res_values = quantile_with_mask(arr, mask, fill_value, qs, interpolation)
        return type(self)._from_sequence(res_values)

    def _mode(self, dropna: bool = True) -> Self:
        """
        Returns the mode(s) of the ExtensionArray.

        Always returns `ExtensionArray` even if only one value.

        Parameters
        ----------
        dropna : bool, default True
            Don't consider counts of NA values.

        Returns
        -------
        same type as self
            Sorted, if possible.
        """
        # error: Incompatible return value type (got "Union[ExtensionArray,
        # ndarray[Any, Any]]", expected "Self")
        return mode(self, dropna=dropna)  # type: ignore[return-value]

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
        if any(
            isinstance(other, (ABCSeries, ABCIndex, ABCDataFrame)) for other in inputs
        ):
            return NotImplemented

        result = arraylike.maybe_dispatch_ufunc_to_dunder_op(
            self, ufunc, method, *inputs, **kwargs
        )
        if result is not NotImplemented:
            return result

        if "out" in kwargs:
            return arraylike.dispatch_ufunc_with_out(
                self, ufunc, method, *inputs, **kwargs
            )

        if method == "reduce":
            result = arraylike.dispatch_reduction_ufunc(
                self, ufunc, method, *inputs, **kwargs
            )
            if result is not NotImplemented:
                return result

        return arraylike.default_array_ufunc(self, ufunc, method, *inputs, **kwargs)

    def map(self, mapper, na_action=None):
        """
        Map values using an input mapping or function.

        Parameters
        ----------
        mapper : function, dict, or Series
            Mapping correspondence.
        na_action : {None, 'ignore'}, default None
            If 'ignore', propagate NA values, without passing them to the
            mapping correspondence. If 'ignore' is not supported, a
            ``NotImplementedError`` should be raised.

        Returns
        -------
        Union[ndarray, Index, ExtensionArray]
            The output of the mapping function applied to the array.
            If the function returns a tuple with more than one element
            a MultiIndex will be returned.
        """
        return map_array(self, mapper, na_action=na_action)

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
    ) -> ArrayLike:
        """
        Dispatch GroupBy reduction or transformation operation.

        This is an *experimental* API to allow ExtensionArray authors to implement
        reductions and transformations. The API is subject to change.

        Parameters
        ----------
        how : {'any', 'all', 'sum', 'prod', 'min', 'max', 'mean', 'median',
               'median', 'var', 'std', 'sem', 'nth', 'last', 'ohlc',
               'cumprod', 'cumsum', 'cummin', 'cummax', 'rank'}
        has_dropped_na : bool
        min_count : int
        ngroups : int
        ids : np.ndarray[np.intp]
            ids[i] gives the integer label for the group that self[i] belongs to.
        **kwargs : operation-specific
            'any', 'all' -> ['skipna']
            'var', 'std', 'sem' -> ['ddof']
            'cumprod', 'cumsum', 'cummin', 'cummax' -> ['skipna']
            'rank' -> ['ties_method', 'ascending', 'na_option', 'pct']

        Returns
        -------
        np.ndarray or ExtensionArray
        """
        from pandas.core.arrays.string_ import StringDtype
        from pandas.core.groupby.ops import WrappedCythonOp

        kind = WrappedCythonOp.get_kind_from_how(how)
        op = WrappedCythonOp(how=how, kind=kind, has_dropped_na=has_dropped_na)

        initial: Any = 0
        # GH#43682
        if isinstance(self.dtype, StringDtype):
            # StringArray
            if op.how in [
                "prod",
                "mean",
                "median",
                "cumsum",
                "cumprod",
                "std",
                "sem",
                "var",
                "skew",
            ]:
                raise TypeError(
                    f"dtype '{self.dtype}' does not support operation '{how}'"
                )
            if op.how not in ["any", "all"]:
                # Fail early to avoid conversion to object
                op._get_cython_function(op.kind, op.how, np.dtype(object), False)

            arr = self
            if op.how == "sum":
                initial = ""
                # https://github.com/pandas-dev/pandas/issues/60229
                # All NA should result in the empty string.
                if min_count == 0:
                    arr = arr.fillna("")
            npvalues = arr.to_numpy(object, na_value=np.nan)
        else:
            raise NotImplementedError(
                f"function is not implemented for this dtype: {self.dtype}"
            )

        res_values = op._cython_op_ndim_compat(
            npvalues,
            min_count=min_count,
            ngroups=ngroups,
            comp_ids=ids,
            mask=None,
            initial=initial,
            **kwargs,
        )

        if op.how in op.cast_blocklist:
            # i.e. how in ["rank"], since other cast_blocklist methods don't go
            #  through cython_operation
            return res_values

        if isinstance(self.dtype, StringDtype):
            dtype = self.dtype
            string_array_cls = dtype.construct_array_type()
            return string_array_cls._from_sequence(res_values, dtype=dtype)

        else:
            raise NotImplementedError


class ExtensionArraySupportsAnyAll(ExtensionArray):
    def any(self, *, skipna: bool = True) -> bool:
        raise AbstractMethodError(self)

    def all(self, *, skipna: bool = True) -> bool:
        raise AbstractMethodError(self)


class ExtensionOpsMixin:
    """
    A base class for linking the operators to their dunder names.

    .. note::

       You may want to set ``__array_priority__`` if you want your
       implementation to be called when involved in binary operations
       with NumPy arrays.
    """

    @classmethod
    def _create_arithmetic_method(cls, op):
        raise AbstractMethodError(cls)

    @classmethod
    def _add_arithmetic_ops(cls) -> None:
        setattr(cls, "__add__", cls._create_arithmetic_method(operator.add))
        setattr(cls, "__radd__", cls._create_arithmetic_method(roperator.radd))
        setattr(cls, "__sub__", cls._create_arithmetic_method(operator.sub))
        setattr(cls, "__rsub__", cls._create_arithmetic_method(roperator.rsub))
        setattr(cls, "__mul__", cls._create_arithmetic_method(operator.mul))
        setattr(cls, "__rmul__", cls._create_arithmetic_method(roperator.rmul))
        setattr(cls, "__pow__", cls._create_arithmetic_method(operator.pow))
        setattr(cls, "__rpow__", cls._create_arithmetic_method(roperator.rpow))
        setattr(cls, "__mod__", cls._create_arithmetic_method(operator.mod))
        setattr(cls, "__rmod__", cls._create_arithmetic_method(roperator.rmod))
        setattr(cls, "__floordiv__", cls._create_arithmetic_method(operator.floordiv))
        setattr(
            cls, "__rfloordiv__", cls._create_arithmetic_method(roperator.rfloordiv)
        )
        setattr(cls, "__truediv__", cls._create_arithmetic_method(operator.truediv))
        setattr(cls, "__rtruediv__", cls._create_arithmetic_method(roperator.rtruediv))
        setattr(cls, "__divmod__", cls._create_arithmetic_method(divmod))
        setattr(cls, "__rdivmod__", cls._create_arithmetic_method(roperator.rdivmod))

    @classmethod
    def _create_comparison_method(cls, op):
        raise AbstractMethodError(cls)

    @classmethod
    def _add_comparison_ops(cls) -> None:
        setattr(cls, "__eq__", cls._create_comparison_method(operator.eq))
        setattr(cls, "__ne__", cls._create_comparison_method(operator.ne))
        setattr(cls, "__lt__", cls._create_comparison_method(operator.lt))
        setattr(cls, "__gt__", cls._create_comparison_method(operator.gt))
        setattr(cls, "__le__", cls._create_comparison_method(operator.le))
        setattr(cls, "__ge__", cls._create_comparison_method(operator.ge))

    @classmethod
    def _create_logical_method(cls, op):
        raise AbstractMethodError(cls)

    @classmethod
    def _add_logical_ops(cls) -> None:
        setattr(cls, "__and__", cls._create_logical_method(operator.and_))
        setattr(cls, "__rand__", cls._create_logical_method(roperator.rand_))
        setattr(cls, "__or__", cls._create_logical_method(operator.or_))
        setattr(cls, "__ror__", cls._create_logical_method(roperator.ror_))
        setattr(cls, "__xor__", cls._create_logical_method(operator.xor))
        setattr(cls, "__rxor__", cls._create_logical_method(roperator.rxor))


class ExtensionScalarOpsMixin(ExtensionOpsMixin):
    """
    A mixin for defining ops on an ExtensionArray.

    It is assumed that the underlying scalar objects have the operators
    already defined.

    Notes
    -----
    If you have defined a subclass MyExtensionArray(ExtensionArray), then
    use MyExtensionArray(ExtensionArray, ExtensionScalarOpsMixin) to
    get the arithmetic operators.  After the definition of MyExtensionArray,
    insert the lines

    MyExtensionArray._add_arithmetic_ops()
    MyExtensionArray._add_comparison_ops()

    to link the operators to your class.

    .. note::

       You may want to set ``__array_priority__`` if you want your
       implementation to be called when involved in binary operations
       with NumPy arrays.
    """

    @classmethod
    def _create_method(cls, op, coerce_to_dtype: bool = True, result_dtype=None):
        """
        A class method that returns a method that will correspond to an
        operator for an ExtensionArray subclass, by dispatching to the
        relevant operator defined on the individual elements of the
        ExtensionArray.

        Parameters
        ----------
        op : function
            An operator that takes arguments op(a, b)
        coerce_to_dtype : bool, default True
            boolean indicating whether to attempt to convert
            the result to the underlying ExtensionArray dtype.
            If it's not possible to create a new ExtensionArray with the
            values, an ndarray is returned instead.

        Returns
        -------
        Callable[[Any, Any], Union[ndarray, ExtensionArray]]
            A method that can be bound to a class. When used, the method
            receives the two arguments, one of which is the instance of
            this class, and should return an ExtensionArray or an ndarray.

            Returning an ndarray may be necessary when the result of the
            `op` cannot be stored in the ExtensionArray. The dtype of the
            ndarray uses NumPy's normal inference rules.

        Examples
        --------
        Given an ExtensionArray subclass called MyExtensionArray, use

            __add__ = cls._create_method(operator.add)

        in the class definition of MyExtensionArray to create the operator
        for addition, that will be based on the operator implementation
        of the underlying elements of the ExtensionArray
        """

        def _binop(self, other):
            def convert_values(param):
                if isinstance(param, ExtensionArray) or is_list_like(param):
                    ovalues = param
                else:  # Assume its an object
                    ovalues = [param] * len(self)
                return ovalues

            if isinstance(other, (ABCSeries, ABCIndex, ABCDataFrame)):
                # rely on pandas to unbox and dispatch to us
                return NotImplemented

            lvalues = self
            rvalues = convert_values(other)

            # If the operator is not defined for the underlying objects,
            # a TypeError should be raised
            res = [op(a, b) for (a, b) in zip(lvalues, rvalues)]

            def _maybe_convert(arr):
                if coerce_to_dtype:
                    # https://github.com/pandas-dev/pandas/issues/22850
                    # We catch all regular exceptions here, and fall back
                    # to an ndarray.
                    res = maybe_cast_pointwise_result(arr, self.dtype, same_dtype=False)
                    if not isinstance(res, type(self)):
                        # exception raised in _from_sequence; ensure we have ndarray
                        res = np.asarray(arr)
                else:
                    res = np.asarray(arr, dtype=result_dtype)
                return res

            if op.__name__ in {"divmod", "rdivmod"}:
                a, b = zip(*res)
                return _maybe_convert(a), _maybe_convert(b)

            return _maybe_convert(res)

        op_name = f"__{op.__name__}__"
        return set_function_name(_binop, op_name, cls)

    @classmethod
    def _create_arithmetic_method(cls, op):
        return cls._create_method(op)

    @classmethod
    def _create_comparison_method(cls, op):
        return cls._create_method(op, coerce_to_dtype=False, result_dtype=bool)
<|endoftext|>from __future__ import annotations

from datetime import timedelta
import operator
from typing import (
    TYPE_CHECKING,
    cast,
)

import numpy as np

from pandas._libs import (
    lib,
    tslibs,
)
from pandas._libs.tslibs import (
    NaT,
    NaTType,
    Tick,
    Timedelta,
    astype_overflowsafe,
    get_supported_dtype,
    iNaT,
    is_supported_dtype,
    periods_per_second,
)
from pandas._libs.tslibs.conversion import cast_from_unit_vectorized
from pandas._libs.tslibs.fields import (
    get_timedelta_days,
    get_timedelta_field,
)
from pandas._libs.tslibs.timedeltas import (
    array_to_timedelta64,
    floordiv_object_array,
    ints_to_pytimedelta,
    parse_timedelta_unit,
    truediv_object_array,
)
from pandas.compat.numpy import function as nv
from pandas.util._validators import validate_endpoints

from pandas.core.dtypes.common import (
    TD64NS_DTYPE,
    is_float_dtype,
    is_integer_dtype,
    is_object_dtype,
    is_scalar,
    is_string_dtype,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.missing import isna

from pandas.core import (
    nanops,
    roperator,
)
from pandas.core.array_algos import datetimelike_accumulations
from pandas.core.arrays import datetimelike as dtl
from pandas.core.arrays._ranges import generate_regular_range
import pandas.core.common as com
from pandas.core.ops.common import unpack_zerodim_and_defer

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pandas._typing import (
        AxisInt,
        DateTimeErrorChoices,
        DtypeObj,
        NpDtype,
        Self,
        npt,
    )

    from pandas import DataFrame

import textwrap


def _field_accessor(name: str, alias: str, docstring: str):
    def f(self) -> np.ndarray:
        values = self.asi8
        if alias == "days":
            result = get_timedelta_days(values, reso=self._creso)
        else:
            # error: Incompatible types in assignment (
            # expression has type "ndarray[Any, dtype[signedinteger[_32Bit]]]",
            # variable has type "ndarray[Any, dtype[signedinteger[_64Bit]]]
            result = get_timedelta_field(values, alias, reso=self._creso)  # type: ignore[assignment]
        if self._hasna:
            result = self._maybe_mask_results(
                result, fill_value=None, convert="float64"
            )

        return result

    f.__name__ = name
    f.__doc__ = f"\n{docstring}\n"
    return property(f)


class TimedeltaArray(dtl.TimelikeOps):
    """
    Pandas ExtensionArray for timedelta data.

    .. warning::

       TimedeltaArray is currently experimental, and its API may change
       without warning. In particular, :attr:`TimedeltaArray.dtype` is
       expected to change to be an instance of an ``ExtensionDtype``
       subclass.

    Parameters
    ----------
    values : array-like
        The timedelta data.

    dtype : numpy.dtype
        Currently, only ``numpy.dtype("timedelta64[ns]")`` is accepted.
    freq : Offset, optional
    copy : bool, default False
        Whether to copy the underlying array of data.

    Attributes
    ----------
    None

    Methods
    -------
    None

    Examples
    --------
    >>> pd.arrays.TimedeltaArray._from_sequence(pd.TimedeltaIndex(['1h', '2h']))
    <TimedeltaArray>
    ['0 days 01:00:00', '0 days 02:00:00']
    Length: 2, dtype: timedelta64[ns]
    """

    _typ = "timedeltaarray"
    _internal_fill_value = np.timedelta64("NaT", "ns")
    _recognized_scalars = (timedelta, np.timedelta64, Tick)
    _is_recognized_dtype = lambda x: lib.is_np_dtype(x, "m")
    _infer_matches = ("timedelta", "timedelta64")

    @property
    def _scalar_type(self) -> type[Timedelta]:
        return Timedelta

    __array_priority__ = 1000
    # define my properties & methods for delegation
    _other_ops: list[str] = []
    _bool_ops: list[str] = []
    _object_ops: list[str] = ["freq"]
    _field_ops: list[str] = ["days", "seconds", "microseconds", "nanoseconds"]
    _datetimelike_ops: list[str] = _field_ops + _object_ops + _bool_ops + ["unit"]
    _datetimelike_methods: list[str] = [
        "to_pytimedelta",
        "total_seconds",
        "round",
        "floor",
        "ceil",
        "as_unit",
    ]

    # Note: ndim must be defined to ensure NaT.__richcmp__(TimedeltaArray)
    #  operates pointwise.

    def _box_func(self, x: np.timedelta64) -> Timedelta | NaTType:
        y = x.view("i8")
        if y == NaT._value:
            return NaT
        return Timedelta._from_value_and_reso(y, reso=self._creso)

    @property
    # error: Return type "dtype" of "dtype" incompatible with return type
    # "ExtensionDtype" in supertype "ExtensionArray"
    def dtype(self) -> np.dtype[np.timedelta64]:  # type: ignore[override]
        """
        The dtype for the TimedeltaArray.

        .. warning::

           A future version of pandas will change dtype to be an instance
           of a :class:`pandas.api.extensions.ExtensionDtype` subclass,
           not a ``numpy.dtype``.

        Returns
        -------
        numpy.dtype
        """
        return self._ndarray.dtype

    # ----------------------------------------------------------------
    # Constructors

    _freq = None
    _default_dtype = TD64NS_DTYPE  # used in TimeLikeOps.__init__

    @classmethod
    def _validate_dtype(cls, values, dtype):
        # used in TimeLikeOps.__init__
        dtype = _validate_td64_dtype(dtype)
        _validate_td64_dtype(values.dtype)
        if dtype != values.dtype:
            raise ValueError("Values resolution does not match dtype.")
        return dtype

    # error: Signature of "_simple_new" incompatible with supertype "NDArrayBacked"
    @classmethod
    def _simple_new(  # type: ignore[override]
        cls,
        values: npt.NDArray[np.timedelta64],
        freq: Tick | None = None,
        dtype: np.dtype[np.timedelta64] = TD64NS_DTYPE,
    ) -> Self:
        # Require td64 dtype, not unit-less, matching values.dtype
        assert lib.is_np_dtype(dtype, "m")
        assert not tslibs.is_unitless(dtype)
        assert isinstance(values, np.ndarray), type(values)
        assert dtype == values.dtype
        assert freq is None or isinstance(freq, Tick)

        result = super()._simple_new(values=values, dtype=dtype)
        result._freq = freq
        return result

    @classmethod
    def _from_sequence(cls, data, *, dtype=None, copy: bool = False) -> Self:
        if dtype:
            dtype = _validate_td64_dtype(dtype)

        data, freq = sequence_to_td64ns(data, copy=copy, unit=None)

        if dtype is not None:
            data = astype_overflowsafe(data, dtype=dtype, copy=False)

        return cls._simple_new(data, dtype=data.dtype, freq=freq)

    @classmethod
    def _from_sequence_not_strict(
        cls,
        data,
        *,
        dtype=None,
        copy: bool = False,
        freq=lib.no_default,
        unit=None,
    ) -> Self:
        """
        _from_sequence_not_strict but without responsibility for finding the
        result's `freq`.
        """
        if dtype:
            dtype = _validate_td64_dtype(dtype)

        assert unit not in ["Y", "y", "M"]  # caller is responsible for checking

        data, inferred_freq = sequence_to_td64ns(data, copy=copy, unit=unit)

        if dtype is not None:
            data = astype_overflowsafe(data, dtype=dtype, copy=False)

        result = cls._simple_new(data, dtype=data.dtype, freq=inferred_freq)

        result._maybe_pin_freq(freq, {})
        return result

    @classmethod
    def _generate_range(
        cls, start, end, periods, freq, closed=None, *, unit: str | None = None
    ) -> Self:
        periods = dtl.validate_periods(periods)
        if freq is None and any(x is None for x in [periods, start, end]):
            raise ValueError("Must provide freq argument if no data is supplied")

        if com.count_not_none(start, end, periods, freq) != 3:
            raise ValueError(
                "Of the four parameters: start, end, periods, "
                "and freq, exactly three must be specified"
            )

        if start is not None:
            start = Timedelta(start).as_unit("ns")

        if end is not None:
            end = Timedelta(end).as_unit("ns")

        if unit is not None:
            if unit not in ["s", "ms", "us", "ns"]:
                raise ValueError("'unit' must be one of 's', 'ms', 'us', 'ns'")
        else:
            unit = "ns"

        if start is not None and unit is not None:
            start = start.as_unit(unit, round_ok=False)
        if end is not None and unit is not None:
            end = end.as_unit(unit, round_ok=False)

        left_closed, right_closed = validate_endpoints(closed)

        if freq is not None:
            index = generate_regular_range(start, end, periods, freq, unit=unit)
        else:
            index = np.linspace(start._value, end._value, periods).astype("i8")

        if not left_closed:
            index = index[1:]
        if not right_closed:
            index = index[:-1]

        td64values = index.view(f"m8[{unit}]")
        return cls._simple_new(td64values, dtype=td64values.dtype, freq=freq)

    # ----------------------------------------------------------------
    # DatetimeLike Interface

    def _unbox_scalar(self, value) -> np.timedelta64:
        if not isinstance(value, self._scalar_type) and value is not NaT:
            raise ValueError("'value' should be a Timedelta.")
        self._check_compatible_with(value)
        if value is NaT:
            return np.timedelta64(value._value, self.unit)
        else:
            return value.as_unit(self.unit).asm8

    def _scalar_from_string(self, value) -> Timedelta | NaTType:
        return Timedelta(value)

    def _check_compatible_with(self, other) -> None:
        # we don't have anything to validate.
        pass

    # ----------------------------------------------------------------
    # Array-Like / EA-Interface Methods

    def astype(self, dtype, copy: bool = True):
        # We handle
        #   --> timedelta64[ns]
        #   --> timedelta64
        # DatetimeLikeArrayMixin super call handles other cases
        dtype = pandas_dtype(dtype)

        if lib.is_np_dtype(dtype, "m"):
            if dtype == self.dtype:
                if copy:
                    return self.copy()
                return self

            if is_supported_dtype(dtype):
                # unit conversion e.g. timedelta64[s]
                res_values = astype_overflowsafe(self._ndarray, dtype, copy=False)
                return type(self)._simple_new(
                    res_values, dtype=res_values.dtype, freq=self.freq
                )
            else:
                raise ValueError(
                    f"Cannot convert from {self.dtype} to {dtype}. "
                    "Supported resolutions are 's', 'ms', 'us', 'ns'"
                )

        return dtl.DatetimeLikeArrayMixin.astype(self, dtype, copy=copy)

    def __iter__(self) -> Iterator:
        if self.ndim > 1:
            for i in range(len(self)):
                yield self[i]
        else:
            # convert in chunks of 10k for efficiency
            data = self._ndarray
            length = len(self)
            chunksize = 10000
            chunks = (length // chunksize) + 1
            for i in range(chunks):
                start_i = i * chunksize
                end_i = min((i + 1) * chunksize, length)
                converted = ints_to_pytimedelta(data[start_i:end_i], box=True)
                yield from converted

    # ----------------------------------------------------------------
    # Reductions

    def sum(
        self,
        *,
        axis: AxisInt | None = None,
        dtype: NpDtype | None = None,
        out=None,
        keepdims: bool = False,
        initial=None,
        skipna: bool = True,
        min_count: int = 0,
    ):
        nv.validate_sum(
            (), {"dtype": dtype, "out": out, "keepdims": keepdims, "initial": initial}
        )

        result = nanops.nansum(
            self._ndarray, axis=axis, skipna=skipna, min_count=min_count
        )
        return self._wrap_reduction_result(axis, result)

    def std(
        self,
        *,
        axis: AxisInt | None = None,
        dtype: NpDtype | None = None,
        out=None,
        ddof: int = 1,
        keepdims: bool = False,
        skipna: bool = True,
    ):
        nv.validate_stat_ddof_func(
            (), {"dtype": dtype, "out": out, "keepdims": keepdims}, fname="std"
        )

        result = nanops.nanstd(self._ndarray, axis=axis, skipna=skipna, ddof=ddof)
        if axis is None or self.ndim == 1:
            return self._box_func(result)
        return self._from_backing_data(result)

    # ----------------------------------------------------------------
    # Accumulations

    def _accumulate(self, name: str, *, skipna: bool = True, **kwargs):
        if name == "cumsum":
            op = getattr(datetimelike_accumulations, name)
            result = op(self._ndarray.copy(), skipna=skipna, **kwargs)

            return type(self)._simple_new(result, freq=None, dtype=self.dtype)
        elif name == "cumprod":
            raise TypeError("cumprod not supported for Timedelta.")

        else:
            return super()._accumulate(name, skipna=skipna, **kwargs)

    # ----------------------------------------------------------------
    # Rendering Methods

    def _formatter(self, boxed: bool = False):
        from pandas.io.formats.format import get_format_timedelta64

        return get_format_timedelta64(self, box=True)

    def _format_native_types(
        self, *, na_rep: str | float = "NaT", date_format=None, **kwargs
    ) -> npt.NDArray[np.object_]:
        from pandas.io.formats.format import get_format_timedelta64

        # Relies on TimeDelta._repr_base
        formatter = get_format_timedelta64(self, na_rep)
        # equiv: np.array([formatter(x) for x in self._ndarray])
        #  but independent of dimension
        return np.frompyfunc(formatter, 1, 1)(self._ndarray)

    # ----------------------------------------------------------------
    # Arithmetic Methods

    def _add_offset(self, other):
        assert not isinstance(other, Tick)
        raise TypeError(
            f"cannot add the type {type(other).__name__} to a {type(self).__name__}"
        )

    @unpack_zerodim_and_defer("__mul__")
    def __mul__(self, other) -> Self:
        if is_scalar(other):
            # numpy will accept float and int, raise TypeError for others
            result = self._ndarray * other
            if result.dtype.kind != "m":
                # numpy >= 2.1 may not raise a TypeError
                # and seems to dispatch to others.__rmul__?
                raise TypeError(f"Cannot multiply with {type(other).__name__}")
            freq = None
            if self.freq is not None and not isna(other):
                freq = self.freq * other
                if freq.n == 0:
                    # GH#51575 Better to have no freq than an incorrect one
                    freq = None
            return type(self)._simple_new(result, dtype=result.dtype, freq=freq)

        if not hasattr(other, "dtype"):
            # list, tuple
            other = np.array(other)
        if len(other) != len(self) and not lib.is_np_dtype(other.dtype, "m"):
            # Exclude timedelta64 here so we correctly raise TypeError
            #  for that instead of ValueError
            raise ValueError("Cannot multiply with unequal lengths")

        if is_object_dtype(other.dtype):
            # this multiplication will succeed only if all elements of other
            #  are int or float scalars, so we will end up with
            #  timedelta64[ns]-dtyped result
            arr = self._ndarray
            result = [arr[n] * other[n] for n in range(len(self))]
            result = np.array(result)
            return type(self)._simple_new(result, dtype=result.dtype)

        # numpy will accept float or int dtype, raise TypeError for others
        result = self._ndarray * other
        if result.dtype.kind != "m":
            # numpy >= 2.1 may not raise a TypeError
            # and seems to dispatch to others.__rmul__?
            raise TypeError(f"Cannot multiply with {type(other).__name__}")
        return type(self)._simple_new(result, dtype=result.dtype)

    __rmul__ = __mul__

    def _scalar_divlike_op(self, other, op):
        """
        Shared logic for __truediv__, __rtruediv__, __floordiv__, __rfloordiv__
        with scalar 'other'.
        """
        if isinstance(other, self._recognized_scalars):
            other = Timedelta(other)
            # mypy assumes that __new__ returns an instance of the class
            # github.com/python/mypy/issues/1020
            if cast("Timedelta | NaTType", other) is NaT:
                # specifically timedelta64-NaT
                res = np.empty(self.shape, dtype=np.float64)
                res.fill(np.nan)
                return res

            # otherwise, dispatch to Timedelta implementation
            return op(self._ndarray, other)

        else:
            # caller is responsible for checking lib.is_scalar(other)
            # assume other is numeric, otherwise numpy will raise

            if op in [roperator.rtruediv, roperator.rfloordiv]:
                raise TypeError(
                    f"Cannot divide {type(other).__name__} by {type(self).__name__}"
                )

            result = op(self._ndarray, other)
            freq = None

            if self.freq is not None:
                # Note: freq gets division, not floor-division, even if op
                #  is floordiv.
                freq = self.freq / other
                if freq.nanos == 0 and self.freq.nanos != 0:
                    # e.g. if self.freq is Nano(1) then dividing by 2
                    #  rounds down to zero
                    freq = None

            return type(self)._simple_new(result, dtype=result.dtype, freq=freq)

    def _cast_divlike_op(self, other):
        if not hasattr(other, "dtype"):
            # e.g. list, tuple
            other = np.array(other)

        if len(other) != len(self):
            raise ValueError("Cannot divide vectors with unequal lengths")
        return other

    def _vector_divlike_op(self, other, op) -> np.ndarray | Self:
        """
        Shared logic for __truediv__, __floordiv__, and their reversed versions
        with timedelta64-dtype ndarray other.
        """
        # Let numpy handle it
        result = op(self._ndarray, np.asarray(other))

        if (is_integer_dtype(other.dtype) or is_float_dtype(other.dtype)) and op in [
            operator.truediv,
            operator.floordiv,
        ]:
            return type(self)._simple_new(result, dtype=result.dtype)

        if op in [operator.floordiv, roperator.rfloordiv]:
            mask = self.isna() | isna(other)
            if mask.any():
                result = result.astype(np.float64)
                np.putmask(result, mask, np.nan)

        return result

    @unpack_zerodim_and_defer("__truediv__")
    def __truediv__(self, other):
        # timedelta / X is well-defined for timedelta-like or numeric X
        op = operator.truediv
        if is_scalar(other):
            return self._scalar_divlike_op(other, op)

        other = self._cast_divlike_op(other)
        if (
            lib.is_np_dtype(other.dtype, "m")
            or is_integer_dtype(other.dtype)
            or is_float_dtype(other.dtype)
        ):
            return self._vector_divlike_op(other, op)

        if is_object_dtype(other.dtype):
            other = np.asarray(other)
            if self.ndim > 1:
                res_cols = [left / right for left, right in zip(self, other)]
                res_cols2 = [x.reshape(1, -1) for x in res_cols]
                result = np.concatenate(res_cols2, axis=0)
            else:
                result = truediv_object_array(self._ndarray, other)

            return result

        else:
            return NotImplemented

    @unpack_zerodim_and_defer("__rtruediv__")
    def __rtruediv__(self, other):
        # X / timedelta is defined only for timedelta-like X
        op = roperator.rtruediv
        if is_scalar(other):
            return self._scalar_divlike_op(other, op)

        other = self._cast_divlike_op(other)
        if lib.is_np_dtype(other.dtype, "m"):
            return self._vector_divlike_op(other, op)

        elif is_object_dtype(other.dtype):
            # Note: unlike in __truediv__, we do not _need_ to do type
            #  inference on the result.  It does not raise, a numeric array
            #  is returned.  GH#23829
            result_list = [other[n] / self[n] for n in range(len(self))]
            return np.array(result_list)

        else:
            return NotImplemented

    @unpack_zerodim_and_defer("__floordiv__")
    def __floordiv__(self, other):
        op = operator.floordiv
        if is_scalar(other):
            return self._scalar_divlike_op(other, op)

        other = self._cast_divlike_op(other)
        if (
            lib.is_np_dtype(other.dtype, "m")
            or is_integer_dtype(other.dtype)
            or is_float_dtype(other.dtype)
        ):
            return self._vector_divlike_op(other, op)

        elif is_object_dtype(other.dtype):
            other = np.asarray(other)
            if self.ndim > 1:
                res_cols = [left // right for left, right in zip(self, other)]
                res_cols2 = [x.reshape(1, -1) for x in res_cols]
                result = np.concatenate(res_cols2, axis=0)
            else:
                result = floordiv_object_array(self._ndarray, other)

            assert result.dtype == object
            return result

        else:
            return NotImplemented

    @unpack_zerodim_and_defer("__rfloordiv__")
    def __rfloordiv__(self, other):
        op = roperator.rfloordiv
        if is_scalar(other):
            return self._scalar_divlike_op(other, op)

        other = self._cast_divlike_op(other)
        if lib.is_np_dtype(other.dtype, "m"):
            return self._vector_divlike_op(other, op)

        elif is_object_dtype(other.dtype):
            result_list = [other[n] // self[n] for n in range(len(self))]
            result = np.array(result_list)
            return result

        else:
            return NotImplemented

    @unpack_zerodim_and_defer("__mod__")
    def __mod__(self, other):
        # Note: This is a naive implementation, can likely be optimized
        if isinstance(other, self._recognized_scalars):
            other = Timedelta(other)
        return self - (self // other) * other

    @unpack_zerodim_and_defer("__rmod__")
    def __rmod__(self, other):
        # Note: This is a naive implementation, can likely be optimized
        if isinstance(other, self._recognized_scalars):
            other = Timedelta(other)
        return other - (other // self) * self

    @unpack_zerodim_and_defer("__divmod__")
    def __divmod__(self, other):
        # Note: This is a naive implementation, can likely be optimized
        if isinstance(other, self._recognized_scalars):
            other = Timedelta(other)

        res1 = self // other
        res2 = self - res1 * other
        return res1, res2

    @unpack_zerodim_and_defer("__rdivmod__")
    def __rdivmod__(self, other):
        # Note: This is a naive implementation, can likely be optimized
        if isinstance(other, self._recognized_scalars):
            other = Timedelta(other)

        res1 = other // self
        res2 = other - res1 * self
        return res1, res2

    def __neg__(self) -> TimedeltaArray:
        freq = None
        if self.freq is not None:
            freq = -self.freq
        return type(self)._simple_new(-self._ndarray, dtype=self.dtype, freq=freq)

    def __pos__(self) -> TimedeltaArray:
        return type(self)._simple_new(
            self._ndarray.copy(), dtype=self.dtype, freq=self.freq
        )

    def __abs__(self) -> TimedeltaArray:
        # Note: freq is not preserved
        return type(self)._simple_new(np.abs(self._ndarray), dtype=self.dtype)

    # ----------------------------------------------------------------
    # Conversion Methods - Vectorized analogues of Timedelta methods

    def total_seconds(self) -> npt.NDArray[np.float64]:
        """
        Return total duration of each element expressed in seconds.

        This method is available directly on TimedeltaArray, TimedeltaIndex
        and on Series containing timedelta values under the ``.dt`` namespace.

        Returns
        -------
        ndarray, Index or Series
            When the calling object is a TimedeltaArray, the return type
            is ndarray.  When the calling object is a TimedeltaIndex,
            the return type is an Index with a float64 dtype. When the calling object
            is a Series, the return type is Series of type `float64` whose
            index is the same as the original.

        See Also
        --------
        datetime.timedelta.total_seconds : Standard library version
            of this method.
        TimedeltaIndex.components : Return a DataFrame with components of
            each Timedelta.

        Examples
        --------
        **Series**

        >>> s = pd.Series(pd.to_timedelta(np.arange(5), unit='d'))
        >>> s
        0   0 days
        1   1 days
        2   2 days
        3   3 days
        4   4 days
        dtype: timedelta64[ns]

        >>> s.dt.total_seconds()
        0         0.0
        1     86400.0
        2    172800.0
        3    259200.0
        4    345600.0
        dtype: float64

        **TimedeltaIndex**

        >>> idx = pd.to_timedelta(np.arange(5), unit='d')
        >>> idx
        TimedeltaIndex(['0 days', '1 days', '2 days', '3 days', '4 days'],
                       dtype='timedelta64[ns]', freq=None)

        >>> idx.total_seconds()
        Index([0.0, 86400.0, 172800.0, 259200.0, 345600.0], dtype='float64')
        """
        pps = periods_per_second(self._creso)
        return self._maybe_mask_results(self.asi8 / pps, fill_value=None)

    def to_pytimedelta(self) -> npt.NDArray[np.object_]:
        """
        Return an ndarray of datetime.timedelta objects.

        Returns
        -------
        numpy.ndarray

        Examples
        --------
        >>> tdelta_idx = pd.to_timedelta([1, 2, 3], unit='D')
        >>> tdelta_idx
        TimedeltaIndex(['1 days', '2 days', '3 days'],
                        dtype='timedelta64[ns]', freq=None)
        >>> tdelta_idx.to_pytimedelta()
        array([datetime.timedelta(days=1), datetime.timedelta(days=2),
               datetime.timedelta(days=3)], dtype=object)
        """
        return ints_to_pytimedelta(self._ndarray)

    days_docstring = textwrap.dedent(
        """Number of days for each element.

    Examples
    --------
    For Series:

    >>> ser = pd.Series(pd.to_timedelta([1, 2, 3], unit='d'))
    >>> ser
    0   1 days
    1   2 days
    2   3 days
    dtype: timedelta64[ns]
    >>> ser.dt.days
    0    1
    1    2
    2    3
    dtype: int64

    For TimedeltaIndex:

    >>> tdelta_idx = pd.to_timedelta(["0 days", "10 days", "20 days"])
    >>> tdelta_idx
    TimedeltaIndex(['0 days', '10 days', '20 days'],
                    dtype='timedelta64[ns]', freq=None)
    >>> tdelta_idx.days
    Index([0, 10, 20], dtype='int64')"""
    )
    days = _field_accessor("days", "days", days_docstring)

    seconds_docstring = textwrap.dedent(
        """Number of seconds (>= 0 and less than 1 day) for each element.

    Examples
    --------
    For Series:

    >>> ser = pd.Series(pd.to_timedelta([1, 2, 3], unit='s'))
    >>> ser
    0   0 days 00:00:01
    1   0 days 00:00:02
    2   0 days 00:00:03
    dtype: timedelta64[ns]
    >>> ser.dt.seconds
    0    1
    1    2
    2    3
    dtype: int32

    For TimedeltaIndex:

    >>> tdelta_idx = pd.to_timedelta([1, 2, 3], unit='s')
    >>> tdelta_idx
    TimedeltaIndex(['0 days 00:00:01', '0 days 00:00:02', '0 days 00:00:03'],
                   dtype='timedelta64[ns]', freq=None)
    >>> tdelta_idx.seconds
    Index([1, 2, 3], dtype='int32')"""
    )
    seconds = _field_accessor(
        "seconds",
        "seconds",
        seconds_docstring,
    )

    microseconds_docstring = textwrap.dedent(
        """Number of microseconds (>= 0 and less than 1 second) for each element.

    Examples
    --------
    For Series:

    >>> ser = pd.Series(pd.to_timedelta([1, 2, 3], unit='us'))
    >>> ser
    0   0 days 00:00:00.000001
    1   0 days 00:00:00.000002
    2   0 days 00:00:00.000003
    dtype: timedelta64[ns]
    >>> ser.dt.microseconds
    0    1
    1    2
    2    3
    dtype: int32

    For TimedeltaIndex:

    >>> tdelta_idx = pd.to_timedelta([1, 2, 3], unit='us')
    >>> tdelta_idx
    TimedeltaIndex(['0 days 00:00:00.000001', '0 days 00:00:00.000002',
                    '0 days 00:00:00.000003'],
                   dtype='timedelta64[ns]', freq=None)
    >>> tdelta_idx.microseconds
    Index([1, 2, 3], dtype='int32')"""
    )
    microseconds = _field_accessor(
        "microseconds",
        "microseconds",
        microseconds_docstring,
    )

    nanoseconds_docstring = textwrap.dedent(
        """Number of nanoseconds (>= 0 and less than 1 microsecond) for each element.

    Examples
    --------
    For Series:

    >>> ser = pd.Series(pd.to_timedelta([1, 2, 3], unit='ns'))
    >>> ser
    0   0 days 00:00:00.000000001
    1   0 days 00:00:00.000000002
    2   0 days 00:00:00.000000003
    dtype: timedelta64[ns]
    >>> ser.dt.nanoseconds
    0    1
    1    2
    2    3
    dtype: int32

    For TimedeltaIndex:

    >>> tdelta_idx = pd.to_timedelta([1, 2, 3], unit='ns')
    >>> tdelta_idx
    TimedeltaIndex(['0 days 00:00:00.000000001', '0 days 00:00:00.000000002',
                    '0 days 00:00:00.000000003'],
                   dtype='timedelta64[ns]', freq=None)
    >>> tdelta_idx.nanoseconds
    Index([1, 2, 3], dtype='int32')"""
    )
    nanoseconds = _field_accessor(
        "nanoseconds",
        "nanoseconds",
        nanoseconds_docstring,
    )

    @property
    def components(self) -> DataFrame:
        """
        Return a DataFrame of the individual resolution components of the Timedeltas.

        The components (days, hours, minutes seconds, milliseconds, microseconds,
        nanoseconds) are returned as columns in a DataFrame.

        Returns
        -------
        DataFrame

        Examples
        --------
        >>> tdelta_idx = pd.to_timedelta(['1 day 3 min 2 us 42 ns'])
        >>> tdelta_idx
        TimedeltaIndex(['1 days 00:03:00.000002042'],
                       dtype='timedelta64[ns]', freq=None)
        >>> tdelta_idx.components
           days  hours  minutes  seconds  milliseconds  microseconds  nanoseconds
        0     1      0        3        0             0             2           42
        """
        from pandas import DataFrame

        columns = [
            "days",
            "hours",
            "minutes",
            "seconds",
            "milliseconds",
            "microseconds",
            "nanoseconds",
        ]
        hasnans = self._hasna
        if hasnans:

            def f(x):
                if isna(x):
                    return [np.nan] * len(columns)
                return x.components

        else:

            def f(x):
                return x.components

        result = DataFrame([f(x) for x in self], columns=columns)
        if not hasnans:
            result = result.astype("int64")
        return result


# ---------------------------------------------------------------------
# Constructor Helpers


def sequence_to_td64ns(
    data,
    copy: bool = False,
    unit=None,
    errors: DateTimeErrorChoices = "raise",
) -> tuple[np.ndarray, Tick | None]:
    """
    Parameters
    ----------
    data : list-like
    copy : bool, default False
    unit : str, optional
        The timedelta unit to treat integers as multiples of. For numeric
        data this defaults to ``'ns'``.
        Must be un-specified if the data contains a str and ``errors=="raise"``.
    errors : {"raise", "coerce", "ignore"}, default "raise"
        How to handle elements that cannot be converted to timedelta64[ns].
        See ``pandas.to_timedelta`` for details.

    Returns
    -------
    converted : numpy.ndarray
        The sequence converted to a numpy array with dtype ``timedelta64[ns]``.
    inferred_freq : Tick or None
        The inferred frequency of the sequence.

    Raises
    ------
    ValueError : Data cannot be converted to timedelta64[ns].

    Notes
    -----
    Unlike `pandas.to_timedelta`, if setting ``errors=ignore`` will not cause
    errors to be ignored; they are caught and subsequently ignored at a
    higher level.
    """
    assert unit not in ["Y", "y", "M"]  # caller is responsible for checking

    inferred_freq = None
    if unit is not None:
        unit = parse_timedelta_unit(unit)

    data, copy = dtl.ensure_arraylike_for_datetimelike(
        data, copy, cls_name="TimedeltaArray"
    )

    if isinstance(data, TimedeltaArray):
        inferred_freq = data.freq

    # Convert whatever we have into timedelta64[ns] dtype
    if data.dtype == object or is_string_dtype(data.dtype):
        # no need to make a copy, need to convert if string-dtyped
        data = _objects_to_td64ns(data, unit=unit, errors=errors)
        copy = False

    elif is_integer_dtype(data.dtype):
        # treat as multiples of the given unit
        data, copy_made = _ints_to_td64ns(data, unit=unit)
        copy = copy and not copy_made

    elif is_float_dtype(data.dtype):
        # cast the unit, multiply base/frac separately
        # to avoid precision issues from float -> int
        if isinstance(data.dtype, ExtensionDtype):
            mask = data._mask
            data = data._data
        else:
            mask = np.isnan(data)

        data = cast_from_unit_vectorized(data, unit or "ns")
        data[mask] = iNaT
        data = data.view("m8[ns]")
        copy = False

    elif lib.is_np_dtype(data.dtype, "m"):
        if not is_supported_dtype(data.dtype):
            # cast to closest supported unit, i.e. s or ns
            new_dtype = get_supported_dtype(data.dtype)
            data = astype_overflowsafe(data, dtype=new_dtype, copy=False)
            copy = False

    else:
        # This includes datetime64-dtype, see GH#23539, GH#29794
        raise TypeError(f"dtype {data.dtype} cannot be converted to timedelta64[ns]")

    if not copy:
        data = np.asarray(data)
    else:
        data = np.array(data, copy=copy)

    assert data.dtype.kind == "m"
    assert data.dtype != "m8"  # i.e. not unit-less

    return data, inferred_freq


def _ints_to_td64ns(data, unit: str = "ns"):
    """
    Convert an ndarray with integer-dtype to timedelta64[ns] dtype, treating
    the integers as multiples of the given timedelta unit.

    Parameters
    ----------
    data : numpy.ndarray with integer-dtype
    unit : str, default "ns"
        The timedelta unit to treat integers as multiples of.

    Returns
    -------
    numpy.ndarray : timedelta64[ns] array converted from data
    bool : whether a copy was made
    """
    copy_made = False
    unit = unit if unit is not None else "ns"

    if data.dtype != np.int64:
        # converting to int64 makes a copy, so we can avoid
        # re-copying later
        data = data.astype(np.int64)
        copy_made = True

    if unit != "ns":
        dtype_str = f"timedelta64[{unit}]"
        data = data.view(dtype_str)

        data = astype_overflowsafe(data, dtype=TD64NS_DTYPE)

        # the astype conversion makes a copy, so we can avoid re-copying later
        copy_made = True

    else:
        data = data.view("timedelta64[ns]")

    return data, copy_made


def _objects_to_td64ns(data, unit=None, errors: DateTimeErrorChoices = "raise"):
    """
    Convert a object-dtyped or string-dtyped array into an
    timedelta64[ns]-dtyped array.

    Parameters
    ----------
    data : ndarray or Index
    unit : str, default "ns"
        The timedelta unit to treat integers as multiples of.
        Must not be specified if the data contains a str.
    errors : {"raise", "coerce", "ignore"}, default "raise"
        How to handle elements that cannot be converted to timedelta64[ns].
        See ``pandas.to_timedelta`` for details.

    Returns
    -------
    numpy.ndarray : timedelta64[ns] array converted from data

    Raises
    ------
    ValueError : Data cannot be converted to timedelta64[ns].

    Notes
    -----
    Unlike `pandas.to_timedelta`, if setting `errors=ignore` will not cause
    errors to be ignored; they are caught and subsequently ignored at a
    higher level.
    """
    # coerce Index to np.ndarray, converting string-dtype if necessary
    values = np.asarray(data, dtype=np.object_)

    result = array_to_timedelta64(values, unit=unit, errors=errors)
    return result.view("timedelta64[ns]")


def _validate_td64_dtype(dtype) -> DtypeObj:
    dtype = pandas_dtype(dtype)
    if dtype == np.dtype("m8"):
        # no precision disallowed GH#24806
        msg = (
            "Passing in 'timedelta' dtype with no precision is not allowed. "
            "Please pass in 'timedelta64[ns]' instead."
        )
        raise ValueError(msg)

    if not lib.is_np_dtype(dtype, "m"):
        raise ValueError(f"dtype '{dtype}' is invalid, should be np.timedelta64 dtype")
    elif not is_supported_dtype(dtype):
        raise ValueError("Supported timedelta64 resolutions are 's', 'ms', 'us', 'ns'")

    return dtype
<|endoftext|>"""
Interaction with scipy.sparse matrices.

Currently only includes to_coo helpers.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from pandas._libs import lib

from pandas.core.dtypes.missing import notna

from pandas.core.algorithms import factorize
from pandas.core.indexes.api import MultiIndex
from pandas.core.series import Series

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy as np
    import scipy.sparse

    from pandas._typing import (
        IndexLabel,
        npt,
    )


def _check_is_partition(parts: Iterable, whole: Iterable):
    whole = set(whole)
    parts = [set(x) for x in parts]
    if set.intersection(*parts) != set():
        raise ValueError("Is not a partition because intersection is not null.")
    if set.union(*parts) != whole:
        raise ValueError("Is not a partition because union is not the whole.")


def _levels_to_axis(
    ss,
    levels: tuple[int] | list[int],
    valid_ilocs: npt.NDArray[np.intp],
    sort_labels: bool = False,
) -> tuple[npt.NDArray[np.intp], list[IndexLabel]]:
    """
    For a MultiIndexed sparse Series `ss`, return `ax_coords` and `ax_labels`,
    where `ax_coords` are the coordinates along one of the two axes of the
    destination sparse matrix, and `ax_labels` are the labels from `ss`' Index
    which correspond to these coordinates.

    Parameters
    ----------
    ss : Series
    levels : tuple/list
    valid_ilocs : numpy.ndarray
        Array of integer positions of valid values for the sparse matrix in ss.
    sort_labels : bool, default False
        Sort the axis labels before forming the sparse matrix. When `levels`
        refers to a single level, set to True for a faster execution.

    Returns
    -------
    ax_coords : numpy.ndarray (axis coordinates)
    ax_labels : list (axis labels)
    """
    # Since the labels are sorted in `Index.levels`, when we wish to sort and
    # there is only one level of the MultiIndex for this axis, the desired
    # output can be obtained in the following simpler, more efficient way.
    if sort_labels and len(levels) == 1:
        ax_coords = ss.index.codes[levels[0]][valid_ilocs]
        ax_labels = ss.index.levels[levels[0]]

    else:
        levels_values = lib.fast_zip(
            [ss.index.get_level_values(lvl).to_numpy() for lvl in levels]
        )
        codes, ax_labels = factorize(levels_values, sort=sort_labels)
        ax_coords = codes[valid_ilocs]

    ax_labels = ax_labels.tolist()
    return ax_coords, ax_labels


def _to_ijv(
    ss,
    row_levels: tuple[int] | list[int] = (0,),
    column_levels: tuple[int] | list[int] = (1,),
    sort_labels: bool = False,
) -> tuple[
    np.ndarray,
    npt.NDArray[np.intp],
    npt.NDArray[np.intp],
    list[IndexLabel],
    list[IndexLabel],
]:
    """
    For an arbitrary MultiIndexed sparse Series return (v, i, j, ilabels,
    jlabels) where (v, (i, j)) is suitable for passing to scipy.sparse.coo
    constructor, and ilabels and jlabels are the row and column labels
    respectively.

    Parameters
    ----------
    ss : Series
    row_levels : tuple/list
    column_levels : tuple/list
    sort_labels : bool, default False
        Sort the row and column labels before forming the sparse matrix.
        When `row_levels` and/or `column_levels` refer to a single level,
        set to `True` for a faster execution.

    Returns
    -------
    values : numpy.ndarray
        Valid values to populate a sparse matrix, extracted from
        ss.
    i_coords : numpy.ndarray (row coordinates of the values)
    j_coords : numpy.ndarray (column coordinates of the values)
    i_labels : list (row labels)
    j_labels : list (column labels)
    """
    # index and column levels must be a partition of the index
    _check_is_partition([row_levels, column_levels], range(ss.index.nlevels))
    # From the sparse Series, get the integer indices and data for valid sparse
    # entries.
    sp_vals = ss.array.sp_values
    na_mask = notna(sp_vals)
    values = sp_vals[na_mask]
    valid_ilocs = ss.array.sp_index.indices[na_mask]

    i_coords, i_labels = _levels_to_axis(
        ss, row_levels, valid_ilocs, sort_labels=sort_labels
    )

    j_coords, j_labels = _levels_to_axis(
        ss, column_levels, valid_ilocs, sort_labels=sort_labels
    )

    return values, i_coords, j_coords, i_labels, j_labels


def sparse_series_to_coo(
    ss: Series,
    row_levels: Iterable[int] = (0,),
    column_levels: Iterable[int] = (1,),
    sort_labels: bool = False,
) -> tuple[scipy.sparse.coo_matrix, list[IndexLabel], list[IndexLabel]]:
    """
    Convert a sparse Series to a scipy.sparse.coo_matrix using index
    levels row_levels, column_levels as the row and column
    labels respectively. Returns the sparse_matrix, row and column labels.
    """
    import scipy.sparse

    if ss.index.nlevels < 2:
        raise ValueError("to_coo requires MultiIndex with nlevels >= 2.")
    if not ss.index.is_unique:
        raise ValueError(
            "Duplicate index entries are not allowed in to_coo transformation."
        )

    # to keep things simple, only rely on integer indexing (not labels)
    row_levels = [ss.index._get_level_number(x) for x in row_levels]
    column_levels = [ss.index._get_level_number(x) for x in column_levels]

    v, i, j, rows, columns = _to_ijv(
        ss, row_levels=row_levels, column_levels=column_levels, sort_labels=sort_labels
    )
    sparse_matrix = scipy.sparse.coo_matrix(
        (v, (i, j)), shape=(len(rows), len(columns))
    )
    return sparse_matrix, rows, columns


def coo_to_sparse_series(
    A: scipy.sparse.coo_matrix, dense_index: bool = False
) -> Series:
    """
    Convert a scipy.sparse.coo_matrix to a Series with type sparse.

    Parameters
    ----------
    A : scipy.sparse.coo_matrix
    dense_index : bool, default False

    Returns
    -------
    Series

    Raises
    ------
    TypeError if A is not a coo_matrix
    """
    from pandas import SparseDtype

    try:
        ser = Series(A.data, MultiIndex.from_arrays((A.row, A.col)), copy=False)
    except AttributeError as err:
        raise TypeError(
            f"Expected coo_matrix. Got {type(A).__name__} instead."
        ) from err
    ser = ser.sort_index()
    ser = ser.astype(SparseDtype(ser.dtype))
    if dense_index:
        ind = MultiIndex.from_product([A.row, A.col])
        ser = ser.reindex(ind)
    return ser
<|endoftext|>"""Sparse accessor"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pandas.compat._optional import import_optional_dependency

from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.dtypes import SparseDtype

from pandas.core.accessor import (
    PandasDelegate,
    delegate_names,
)
from pandas.core.arrays.sparse.array import SparseArray

if TYPE_CHECKING:
    from pandas import (
        DataFrame,
        Series,
    )


class BaseAccessor:
    _validation_msg = "Can only use the '.sparse' accessor with Sparse data."

    def __init__(self, data=None) -> None:
        self._parent = data
        self._validate(data)

    def _validate(self, data):
        raise NotImplementedError


@delegate_names(
    SparseArray, ["npoints", "density", "fill_value", "sp_values"], typ="property"
)
class SparseAccessor(BaseAccessor, PandasDelegate):
    """
    Accessor for SparseSparse from other sparse matrix data types.

    Examples
    --------
    >>> ser = pd.Series([0, 0, 2, 2, 2], dtype="Sparse[int]")
    >>> ser.sparse.density
    0.6
    >>> ser.sparse.sp_values
    array([2, 2, 2])
    """

    def _validate(self, data):
        if not isinstance(data.dtype, SparseDtype):
            raise AttributeError(self._validation_msg)

    def _delegate_property_get(self, name: str, *args, **kwargs):
        return getattr(self._parent.array, name)

    def _delegate_method(self, name: str, *args, **kwargs):
        if name == "from_coo":
            return self.from_coo(*args, **kwargs)
        elif name == "to_coo":
            return self.to_coo(*args, **kwargs)
        else:
            raise ValueError

    @classmethod
    def from_coo(cls, A, dense_index: bool = False) -> Series:
        """
        Create a Series with sparse values from a scipy.sparse.coo_matrix.

        Parameters
        ----------
        A : scipy.sparse.coo_matrix
        dense_index : bool, default False
            If False (default), the index consists of only the
            coords of the non-null entries of the original coo_matrix.
            If True, the index consists of the full sorted
            (row, col) coordinates of the coo_matrix.

        Returns
        -------
        s : Series
            A Series with sparse values.

        Examples
        --------
        >>> from scipy import sparse

        >>> A = sparse.coo_matrix(
        ...     ([3.0, 1.0, 2.0], ([1, 0, 0], [0, 2, 3])), shape=(3, 4)
        ... )
        >>> A
        <COOrdinate sparse matrix of dtype 'float64'
            with 3 stored elements and shape (3, 4)>

        >>> A.todense()
        matrix([[0., 0., 1., 2.],
        [3., 0., 0., 0.],
        [0., 0., 0., 0.]])

        >>> ss = pd.Series.sparse.from_coo(A)
        >>> ss
        0  2    1.0
           3    2.0
        1  0    3.0
        dtype: Sparse[float64, nan]
        """
        from pandas import Series
        from pandas.core.arrays.sparse.scipy_sparse import coo_to_sparse_series

        result = coo_to_sparse_series(A, dense_index=dense_index)
        result = Series(result.array, index=result.index, copy=False)

        return result

    def to_coo(self, row_levels=(0,), column_levels=(1,), sort_labels: bool = False):
        """
        Create a scipy.sparse.coo_matrix from a Series with MultiIndex.

        Use row_levels and column_levels to determine the row and column
        coordinates respectively. row_levels and column_levels are the names
        (labels) or numbers of the levels. {row_levels, column_levels} must be
        a partition of the MultiIndex level names (or numbers).

        Parameters
        ----------
        row_levels : tuple/list
        column_levels : tuple/list
        sort_labels : bool, default False
            Sort the row and column labels before forming the sparse matrix.
            When `row_levels` and/or `column_levels` refer to a single level,
            set to `True` for a faster execution.

        Returns
        -------
        y : scipy.sparse.coo_matrix
        rows : list (row labels)
        columns : list (column labels)

        Examples
        --------
        >>> s = pd.Series([3.0, np.nan, 1.0, 3.0, np.nan, np.nan])
        >>> s.index = pd.MultiIndex.from_tuples(
        ...     [
        ...         (1, 2, "a", 0),
        ...         (1, 2, "a", 1),
        ...         (1, 1, "b", 0),
        ...         (1, 1, "b", 1),
        ...         (2, 1, "b", 0),
        ...         (2, 1, "b", 1)
        ...     ],
        ...     names=["A", "B", "C", "D"],
        ... )
        >>> s
        A  B  C  D
        1  2  a  0    3.0
                 1    NaN
           1  b  0    1.0
                 1    3.0
        2  1  b  0    NaN
                 1    NaN
        dtype: float64

        >>> ss = s.astype("Sparse")
        >>> ss
        A  B  C  D
        1  2  a  0    3.0
                 1    NaN
           1  b  0    1.0
                 1    3.0
        2  1  b  0    NaN
                 1    NaN
        dtype: Sparse[float64, nan]

        >>> A, rows, columns = ss.sparse.to_coo(
        ...     row_levels=["A", "B"], column_levels=["C", "D"], sort_labels=True
        ... )
        >>> A
        <COOrdinate sparse matrix of dtype 'float64'
            with 3 stored elements and shape (3, 4)>
        >>> A.todense()
        matrix([[0., 0., 1., 3.],
        [3., 0., 0., 0.],
        [0., 0., 0., 0.]])

        >>> rows
        [(1, 1), (1, 2), (2, 1)]
        >>> columns
        [('a', 0), ('a', 1), ('b', 0), ('b', 1)]
        """
        from pandas.core.arrays.sparse.scipy_sparse import sparse_series_to_coo

        A, rows, columns = sparse_series_to_coo(
            self._parent, row_levels, column_levels, sort_labels=sort_labels
        )
        return A, rows, columns

    def to_dense(self) -> Series:
        """
        Convert a Series from sparse values to dense.

        Returns
        -------
        Series:
            A Series with the same values, stored as a dense array.

        Examples
        --------
        >>> series = pd.Series(pd.arrays.SparseArray([0, 1, 0]))
        >>> series
        0    0
        1    1
        2    0
        dtype: Sparse[int64, 0]

        >>> series.sparse.to_dense()
        0    0
        1    1
        2    0
        dtype: int64
        """
        from pandas import Series

        return Series(
            self._parent.array.to_dense(),
            index=self._parent.index,
            name=self._parent.name,
            copy=False,
        )


class SparseFrameAccessor(BaseAccessor, PandasDelegate):
    """
    DataFrame accessor for sparse data.

    Examples
    --------
    >>> df = pd.DataFrame({"a": [1, 2, 0, 0],
    ...                   "b": [3, 0, 0, 4]}, dtype="Sparse[int]")
    >>> df.sparse.density
    0.5
    """

    def _validate(self, data):
        dtypes = data.dtypes
        if not all(isinstance(t, SparseDtype) for t in dtypes):
            raise AttributeError(self._validation_msg)

    @classmethod
    def from_spmatrix(cls, data, index=None, columns=None) -> DataFrame:
        """
        Create a new DataFrame from a scipy sparse matrix.

        Parameters
        ----------
        data : scipy.sparse.spmatrix
            Must be convertible to csc format.
        index, columns : Index, optional
            Row and column labels to use for the resulting DataFrame.
            Defaults to a RangeIndex.

        Returns
        -------
        DataFrame
            Each column of the DataFrame is stored as a
            :class:`arrays.SparseArray`.

        Examples
        --------
        >>> import scipy.sparse
        >>> mat = scipy.sparse.eye(3, dtype=float)
        >>> pd.DataFrame.sparse.from_spmatrix(mat)
             0    1    2
        0  1.0    0    0
        1    0  1.0    0
        2    0    0  1.0
        """
        from pandas._libs.sparse import IntIndex

        from pandas import DataFrame

        data = data.tocsc()
        index, columns = cls._prep_index(data, index, columns)
        n_rows, n_columns = data.shape
        # We need to make sure indices are sorted, as we create
        # IntIndex with no input validation (i.e. check_integrity=False ).
        # Indices may already be sorted in scipy in which case this adds
        # a small overhead.
        data.sort_indices()
        indices = data.indices
        indptr = data.indptr
        array_data = data.data
        dtype = SparseDtype(array_data.dtype, 0)
        arrays = []
        for i in range(n_columns):
            sl = slice(indptr[i], indptr[i + 1])
            idx = IntIndex(n_rows, indices[sl], check_integrity=False)
            arr = SparseArray._simple_new(array_data[sl], idx, dtype)
            arrays.append(arr)
        return DataFrame._from_arrays(
            arrays, columns=columns, index=index, verify_integrity=False
        )

    def to_dense(self) -> DataFrame:
        """
        Convert a DataFrame with sparse values to dense.

        Returns
        -------
        DataFrame
            A DataFrame with the same values stored as dense arrays.

        Examples
        --------
        >>> df = pd.DataFrame({"A": pd.arrays.SparseArray([0, 1, 0])})
        >>> df.sparse.to_dense()
           A
        0  0
        1  1
        2  0
        """
        from pandas import DataFrame

        data = {k: v.array.to_dense() for k, v in self._parent.items()}
        return DataFrame(data, index=self._parent.index, columns=self._parent.columns)

    def to_coo(self):
        """
        Return the contents of the frame as a sparse SciPy COO matrix.

        Returns
        -------
        scipy.sparse.spmatrix
            If the caller is heterogeneous and contains booleans or objects,
            the result will be of dtype=object. See Notes.

        Notes
        -----
        The dtype will be the lowest-common-denominator type (implicit
        upcasting); that is to say if the dtypes (even of numeric types)
        are mixed, the one that accommodates all will be chosen.

        e.g. If the dtypes are float16 and float32, dtype will be upcast to
        float32. By numpy.find_common_type convention, mixing int64 and
        and uint64 will result in a float64 dtype.

        Examples
        --------
        >>> df = pd.DataFrame({"A": pd.arrays.SparseArray([0, 1, 0, 1])})
        >>> df.sparse.to_coo()
        <COOrdinate sparse matrix of dtype 'int64'
            with 2 stored elements and shape (4, 1)>
        """
        import_optional_dependency("scipy")
        from scipy.sparse import coo_matrix

        dtype = find_common_type(self._parent.dtypes.to_list())
        if isinstance(dtype, SparseDtype):
            dtype = dtype.subtype

        cols, rows, data = [], [], []
        for col, (_, ser) in enumerate(self._parent.items()):
            sp_arr = ser.array
            if sp_arr.fill_value != 0:
                raise ValueError("fill value must be 0 when converting to COO matrix")

            row = sp_arr.sp_index.indices
            cols.append(np.repeat(col, len(row)))
            rows.append(row)
            data.append(sp_arr.sp_values.astype(dtype, copy=False))

        cols = np.concatenate(cols)
        rows = np.concatenate(rows)
        data = np.concatenate(data)
        return coo_matrix((data, (rows, cols)), shape=self._parent.shape)

    @property
    def density(self) -> float:
        """
        Ratio of non-sparse points to total (dense) data points.

        Examples
        --------
        >>> df = pd.DataFrame({"A": pd.arrays.SparseArray([0, 1, 0, 1])})
        >>> df.sparse.density
        0.5
        """
        tmp = np.mean([column.array.density for _, column in self._parent.items()])
        return tmp

    @staticmethod
    def _prep_index(data, index, columns):
        from pandas.core.indexes.api import (
            default_index,
            ensure_index,
        )

        N, K = data.shape
        if index is None:
            index = default_index(N)
        else:
            index = ensure_index(index)
        if columns is None:
            columns = default_index(K)
        else:
            columns = ensure_index(columns)

        if len(columns) != K:
            raise ValueError(f"Column length mismatch: {len(columns)} vs. {K}")
        if len(index) != N:
            raise ValueError(f"Index length mismatch: {len(index)} vs. {N}")
        return index, columns
<|endoftext|>"""
SparseArray data structure
"""
from __future__ import annotations

from collections import abc
import numbers
import operator
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    cast,
    overload,
)
import warnings

import numpy as np

from pandas._libs import lib
import pandas._libs.sparse as splib
from pandas._libs.sparse import (
    BlockIndex,
    IntIndex,
    SparseIndex,
)
from pandas._libs.tslibs import NaT
from pandas.compat.numpy import function as nv
from pandas.errors import PerformanceWarning
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import (
    validate_bool_kwarg,
    validate_insert_loc,
)

from pandas.core.dtypes.astype import astype_array
from pandas.core.dtypes.cast import (
    construct_1d_arraylike_from_scalar,
    find_common_type,
    maybe_box_datetimelike,
)
from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_integer,
    is_list_like,
    is_object_dtype,
    is_scalar,
    is_string_dtype,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import (
    DatetimeTZDtype,
    SparseDtype,
)
from pandas.core.dtypes.generic import (
    ABCIndex,
    ABCSeries,
)
from pandas.core.dtypes.missing import (
    isna,
    na_value_for_dtype,
    notna,
)

from pandas.core import arraylike
import pandas.core.algorithms as algos
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays import ExtensionArray
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.construction import (
    ensure_wrapped_if_datetimelike,
    extract_array,
    sanitize_array,
)
from pandas.core.indexers import (
    check_array_indexer,
    unpack_tuple_and_ellipses,
)
from pandas.core.nanops import check_below_min_count

from pandas.io.formats import printing

# See https://github.com/python/typing/issues/684
if TYPE_CHECKING:
    from collections.abc import Sequence
    from enum import Enum

    class ellipsis(Enum):
        Ellipsis = "..."

    Ellipsis = ellipsis.Ellipsis

    from scipy.sparse import spmatrix

    from pandas._typing import (
        FillnaOptions,
        NumpySorter,
    )

    SparseIndexKind = Literal["integer", "block"]

    from pandas._typing import (
        ArrayLike,
        AstypeArg,
        Axis,
        AxisInt,
        Dtype,
        NpDtype,
        PositionalIndexer,
        Scalar,
        ScalarIndexer,
        Self,
        SequenceIndexer,
        npt,
    )

    from pandas import Series

else:
    ellipsis = type(Ellipsis)


# ----------------------------------------------------------------------------
# Array

_sparray_doc_kwargs = {"klass": "SparseArray"}


def _get_fill(arr: SparseArray) -> np.ndarray:
    """
    Create a 0-dim ndarray containing the fill value

    Parameters
    ----------
    arr : SparseArray

    Returns
    -------
    fill_value : ndarray
        0-dim ndarray with just the fill value.

    Notes
    -----
    coerce fill_value to arr dtype if possible
    int64 SparseArray can have NaN as fill_value if there is no missing
    """
    try:
        return np.asarray(arr.fill_value, dtype=arr.dtype.subtype)
    except ValueError:
        return np.asarray(arr.fill_value)


def _sparse_array_op(
    left: SparseArray, right: SparseArray, op: Callable, name: str
) -> SparseArray:
    """
    Perform a binary operation between two arrays.

    Parameters
    ----------
    left : Union[SparseArray, ndarray]
    right : Union[SparseArray, ndarray]
    op : Callable
        The binary operation to perform
    name str
        Name of the callable.

    Returns
    -------
    SparseArray
    """
    if name.startswith("__"):
        # For lookups in _libs.sparse we need non-dunder op name
        name = name[2:-2]

    # dtype used to find corresponding sparse method
    ltype = left.dtype.subtype
    rtype = right.dtype.subtype

    if ltype != rtype:
        subtype = find_common_type([ltype, rtype])
        ltype = SparseDtype(subtype, left.fill_value)
        rtype = SparseDtype(subtype, right.fill_value)

        left = left.astype(ltype, copy=False)
        right = right.astype(rtype, copy=False)
        dtype = ltype.subtype
    else:
        dtype = ltype

    # dtype the result must have
    result_dtype = None

    if left.sp_index.ngaps == 0 or right.sp_index.ngaps == 0:
        with np.errstate(all="ignore"):
            result = op(left.to_dense(), right.to_dense())
            fill = op(_get_fill(left), _get_fill(right))

        if left.sp_index.ngaps == 0:
            index = left.sp_index
        else:
            index = right.sp_index
    elif left.sp_index.equals(right.sp_index):
        with np.errstate(all="ignore"):
            result = op(left.sp_values, right.sp_values)
            fill = op(_get_fill(left), _get_fill(right))
        index = left.sp_index
    else:
        if name[0] == "r":
            left, right = right, left
            name = name[1:]

        if name in ("and", "or", "xor") and dtype == "bool":
            opname = f"sparse_{name}_uint8"
            # to make template simple, cast here
            left_sp_values = left.sp_values.view(np.uint8)
            right_sp_values = right.sp_values.view(np.uint8)
            result_dtype = bool
        else:
            opname = f"sparse_{name}_{dtype}"
            left_sp_values = left.sp_values
            right_sp_values = right.sp_values

        if (
            name in ["floordiv", "mod"]
            and (right == 0).any()
            and left.dtype.kind in "iu"
        ):
            # Match the non-Sparse Series behavior
            opname = f"sparse_{name}_float64"
            left_sp_values = left_sp_values.astype("float64")
            right_sp_values = right_sp_values.astype("float64")

        sparse_op = getattr(splib, opname)

        with np.errstate(all="ignore"):
            result, index, fill = sparse_op(
                left_sp_values,
                left.sp_index,
                left.fill_value,
                right_sp_values,
                right.sp_index,
                right.fill_value,
            )

    if name == "divmod":
        # result is a 2-tuple
        # error: Incompatible return value type (got "Tuple[SparseArray,
        # SparseArray]", expected "SparseArray")
        return (  # type: ignore[return-value]
            _wrap_result(name, result[0], index, fill[0], dtype=result_dtype),
            _wrap_result(name, result[1], index, fill[1], dtype=result_dtype),
        )

    if result_dtype is None:
        result_dtype = result.dtype

    return _wrap_result(name, result, index, fill, dtype=result_dtype)


def _wrap_result(
    name: str, data, sparse_index, fill_value, dtype: Dtype | None = None
) -> SparseArray:
    """
    wrap op result to have correct dtype
    """
    if name.startswith("__"):
        # e.g. __eq__ --> eq
        name = name[2:-2]

    if name in ("eq", "ne", "lt", "gt", "le", "ge"):
        dtype = bool

    fill_value = lib.item_from_zerodim(fill_value)

    if is_bool_dtype(dtype):
        # fill_value may be np.bool_
        fill_value = bool(fill_value)
    return SparseArray(
        data, sparse_index=sparse_index, fill_value=fill_value, dtype=dtype
    )


class SparseArray(OpsMixin, PandasObject, ExtensionArray):
    """
    An ExtensionArray for storing sparse data.

    Parameters
    ----------
    data : array-like or scalar
        A dense array of values to store in the SparseArray. This may contain
        `fill_value`.
    sparse_index : SparseIndex, optional
    fill_value : scalar, optional
        Elements in data that are ``fill_value`` are not stored in the
        SparseArray. For memory savings, this should be the most common value
        in `data`. By default, `fill_value` depends on the dtype of `data`:

        =========== ==========
        data.dtype  na_value
        =========== ==========
        float       ``np.nan``
        int         ``0``
        bool        False
        datetime64  ``pd.NaT``
        timedelta64 ``pd.NaT``
        =========== ==========

        The fill value is potentially specified in three ways. In order of
        precedence, these are

        1. The `fill_value` argument
        2. ``dtype.fill_value`` if `fill_value` is None and `dtype` is
           a ``SparseDtype``
        3. ``data.dtype.fill_value`` if `fill_value` is None and `dtype`
           is not a ``SparseDtype`` and `data` is a ``SparseArray``.

    kind : str
        Can be 'integer' or 'block', default is 'integer'.
        The type of storage for sparse locations.

        * 'block': Stores a `block` and `block_length` for each
          contiguous *span* of sparse values. This is best when
          sparse data tends to be clumped together, with large
          regions of ``fill-value`` values between sparse values.
        * 'integer': uses an integer to store the location of
          each sparse value.

    dtype : np.dtype or SparseDtype, optional
        The dtype to use for the SparseArray. For numpy dtypes, this
        determines the dtype of ``self.sp_values``. For SparseDtype,
        this determines ``self.sp_values`` and ``self.fill_value``.
    copy : bool, default False
        Whether to explicitly copy the incoming `data` array.

    Attributes
    ----------
    None

    Methods
    -------
    None

    Examples
    --------
    >>> from pandas.arrays import SparseArray
    >>> arr = SparseArray([0, 0, 1, 2])
    >>> arr
    [0, 0, 1, 2]
    Fill: 0
    IntIndex
    Indices: array([2, 3], dtype=int32)
    """

    _subtyp = "sparse_array"  # register ABCSparseArray
    _hidden_attrs = PandasObject._hidden_attrs | frozenset([])
    _sparse_index: SparseIndex
    _sparse_values: np.ndarray
    _dtype: SparseDtype

    def __init__(
        self,
        data,
        sparse_index=None,
        fill_value=None,
        kind: SparseIndexKind = "integer",
        dtype: Dtype | None = None,
        copy: bool = False,
    ) -> None:
        if fill_value is None and isinstance(dtype, SparseDtype):
            fill_value = dtype.fill_value

        if isinstance(data, type(self)):
            # disable normal inference on dtype, sparse_index, & fill_value
            if sparse_index is None:
                sparse_index = data.sp_index
            if fill_value is None:
                fill_value = data.fill_value
            if dtype is None:
                dtype = data.dtype
            # TODO: make kind=None, and use data.kind?
            data = data.sp_values

        # Handle use-provided dtype
        if isinstance(dtype, str):
            # Two options: dtype='int', regular numpy dtype
            # or dtype='Sparse[int]', a sparse dtype
            try:
                dtype = SparseDtype.construct_from_string(dtype)
            except TypeError:
                dtype = pandas_dtype(dtype)

        if isinstance(dtype, SparseDtype):
            if fill_value is None:
                fill_value = dtype.fill_value
            dtype = dtype.subtype

        if is_scalar(data):
            warnings.warn(
                f"Constructing {type(self).__name__} with scalar data is deprecated "
                "and will raise in a future version. Pass a sequence instead.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )
            if sparse_index is None:
                npoints = 1
            else:
                npoints = sparse_index.length

            data = construct_1d_arraylike_from_scalar(data, npoints, dtype=None)
            dtype = data.dtype

        if dtype is not None:
            dtype = pandas_dtype(dtype)

        # TODO: disentangle the fill_value dtype inference from
        # dtype inference
        if data is None:
            # TODO: What should the empty dtype be? Object or float?

            # error: Argument "dtype" to "array" has incompatible type
            # "Union[ExtensionDtype, dtype[Any], None]"; expected "Union[dtype[Any],
            # None, type, _SupportsDType, str, Union[Tuple[Any, int], Tuple[Any,
            # Union[int, Sequence[int]]], List[Any], _DTypeDict, Tuple[Any, Any]]]"
            data = np.array([], dtype=dtype)  # type: ignore[arg-type]

        try:
            data = sanitize_array(data, index=None)
        except ValueError:
            # NumPy may raise a ValueError on data like [1, []]
            # we retry with object dtype here.
            if dtype is None:
                dtype = np.dtype(object)
                data = np.atleast_1d(np.asarray(data, dtype=dtype))
            else:
                raise

        if copy:
            # TODO: avoid double copy when dtype forces cast.
            data = data.copy()

        if fill_value is None:
            fill_value_dtype = data.dtype if dtype is None else dtype
            if fill_value_dtype is None:
                fill_value = np.nan
            else:
                fill_value = na_value_for_dtype(fill_value_dtype)

        if isinstance(data, type(self)) and sparse_index is None:
            sparse_index = data._sparse_index
            # error: Argument "dtype" to "asarray" has incompatible type
            # "Union[ExtensionDtype, dtype[Any], None]"; expected "None"
            sparse_values = np.asarray(
                data.sp_values, dtype=dtype  # type: ignore[arg-type]
            )
        elif sparse_index is None:
            data = extract_array(data, extract_numpy=True)
            if not isinstance(data, np.ndarray):
                # EA
                if isinstance(data.dtype, DatetimeTZDtype):
                    warnings.warn(
                        f"Creating SparseArray from {data.dtype} data "
                        "loses timezone information. Cast to object before "
                        "sparse to retain timezone information.",
                        UserWarning,
                        stacklevel=find_stack_level(),
                    )
                    data = np.asarray(data, dtype="datetime64[ns]")
                    if fill_value is NaT:
                        fill_value = np.datetime64("NaT", "ns")
                data = np.asarray(data)
            sparse_values, sparse_index, fill_value = _make_sparse(
                # error: Argument "dtype" to "_make_sparse" has incompatible type
                # "Union[ExtensionDtype, dtype[Any], None]"; expected
                # "Optional[dtype[Any]]"
                data,
                kind=kind,
                fill_value=fill_value,
                dtype=dtype,  # type: ignore[arg-type]
            )
        else:
            # error: Argument "dtype" to "asarray" has incompatible type
            # "Union[ExtensionDtype, dtype[Any], None]"; expected "None"
            sparse_values = np.asarray(data, dtype=dtype)  # type: ignore[arg-type]
            if len(sparse_values) != sparse_index.npoints:
                raise AssertionError(
                    f"Non array-like type {type(sparse_values)} must "
                    "have the same length as the index"
                )
        self._sparse_index = sparse_index
        self._sparse_values = sparse_values
        self._dtype = SparseDtype(sparse_values.dtype, fill_value)

    @classmethod
    def _simple_new(
        cls,
        sparse_array: np.ndarray,
        sparse_index: SparseIndex,
        dtype: SparseDtype,
    ) -> Self:
        new = object.__new__(cls)
        new._sparse_index = sparse_index
        new._sparse_values = sparse_array
        new._dtype = dtype
        return new

    @classmethod
    def from_spmatrix(cls, data: spmatrix) -> Self:
        """
        Create a SparseArray from a scipy.sparse matrix.

        Parameters
        ----------
        data : scipy.sparse.sp_matrix
            This should be a SciPy sparse matrix where the size
            of the second dimension is 1. In other words, a
            sparse matrix with a single column.

        Returns
        -------
        SparseArray

        Examples
        --------
        >>> import scipy.sparse
        >>> mat = scipy.sparse.coo_matrix((4, 1))
        >>> pd.arrays.SparseArray.from_spmatrix(mat)
        [0.0, 0.0, 0.0, 0.0]
        Fill: 0.0
        IntIndex
        Indices: array([], dtype=int32)
        """
        length, ncol = data.shape

        if ncol != 1:
            raise ValueError(f"'data' must have a single column, not '{ncol}'")

        # our sparse index classes require that the positions be strictly
        # increasing. So we need to sort loc, and arr accordingly.
        data = data.tocsc()
        data.sort_indices()
        arr = data.data
        idx = data.indices

        zero = np.array(0, dtype=arr.dtype).item()
        dtype = SparseDtype(arr.dtype, zero)
        index = IntIndex(length, idx)

        return cls._simple_new(arr, index, dtype)

    def __array__(
        self, dtype: NpDtype | None = None, copy: bool | None = None
    ) -> np.ndarray:
        if self.sp_index.ngaps == 0:
            # Compat for na dtype and int values.
            if copy is True:
                return np.array(self.sp_values)
            else:
                return self.sp_values

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

        fill_value = self.fill_value

        if dtype is None:
            # Can NumPy represent this type?
            # If not, `np.result_type` will raise. We catch that
            # and return object.
            if self.sp_values.dtype.kind == "M":
                # However, we *do* special-case the common case of
                # a datetime64 with pandas NaT.
                if fill_value is NaT:
                    # Can't put pd.NaT in a datetime64[ns]
                    fill_value = np.datetime64("NaT")
            try:
                dtype = np.result_type(self.sp_values.dtype, type(fill_value))
            except TypeError:
                dtype = object

        out = np.full(self.shape, fill_value, dtype=dtype)
        out[self.sp_index.indices] = self.sp_values
        return out

    def __setitem__(self, key, value) -> None:
        # I suppose we could allow setting of non-fill_value elements.
        # TODO(SparseArray.__setitem__): remove special cases in
        # ExtensionBlock.where
        msg = "SparseArray does not support item assignment via setitem"
        raise TypeError(msg)

    @classmethod
    def _from_sequence(cls, scalars, *, dtype: Dtype | None = None, copy: bool = False):
        return cls(scalars, dtype=dtype)

    @classmethod
    def _from_factorized(cls, values, original):
        return cls(values, dtype=original.dtype)

    # ------------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------------
    @property
    def sp_index(self) -> SparseIndex:
        """
        The SparseIndex containing the location of non- ``fill_value`` points.
        """
        return self._sparse_index

    @property
    def sp_values(self) -> np.ndarray:
        """
        An ndarray containing the non- ``fill_value`` values.

        Examples
        --------
        >>> from pandas.arrays import SparseArray
        >>> s = SparseArray([0, 0, 1, 0, 2], fill_value=0)
        >>> s.sp_values
        array([1, 2])
        """
        return self._sparse_values

    @property
    def dtype(self) -> SparseDtype:
        return self._dtype

    @property
    def fill_value(self):
        """
        Elements in `data` that are `fill_value` are not stored.

        For memory savings, this should be the most common value in the array.

        Examples
        --------
        >>> ser = pd.Series([0, 0, 2, 2, 2], dtype="Sparse[int]")
        >>> ser.sparse.fill_value
        0
        >>> spa_dtype = pd.SparseDtype(dtype=np.int32, fill_value=2)
        >>> ser = pd.Series([0, 0, 2, 2, 2], dtype=spa_dtype)
        >>> ser.sparse.fill_value
        2
        """
        return self.dtype.fill_value

    @fill_value.setter
    def fill_value(self, value) -> None:
        self._dtype = SparseDtype(self.dtype.subtype, value)

    @property
    def kind(self) -> SparseIndexKind:
        """
        The kind of sparse index for this array. One of {'integer', 'block'}.
        """
        if isinstance(self.sp_index, IntIndex):
            return "integer"
        else:
            return "block"

    @property
    def _valid_sp_values(self) -> np.ndarray:
        sp_vals = self.sp_values
        mask = notna(sp_vals)
        return sp_vals[mask]

    def __len__(self) -> int:
        return self.sp_index.length

    @property
    def _null_fill_value(self) -> bool:
        return self._dtype._is_na_fill_value

    def _fill_value_matches(self, fill_value) -> bool:
        if self._null_fill_value:
            return isna(fill_value)
        else:
            return self.fill_value == fill_value

    @property
    def nbytes(self) -> int:
        return self.sp_values.nbytes + self.sp_index.nbytes

    @property
    def density(self) -> float:
        """
        The percent of non- ``fill_value`` points, as decimal.

        Examples
        --------
        >>> from pandas.arrays import SparseArray
        >>> s = SparseArray([0, 0, 1, 1, 1], fill_value=0)
        >>> s.density
        0.6
        """
        return self.sp_index.npoints / self.sp_index.length

    @property
    def npoints(self) -> int:
        """
        The number of non- ``fill_value`` points.

        Examples
        --------
        >>> from pandas.arrays import SparseArray
        >>> s = SparseArray([0, 0, 1, 1, 1], fill_value=0)
        >>> s.npoints
        3
        """
        return self.sp_index.npoints

    # error: Return type "SparseArray" of "isna" incompatible with return type
    # "ndarray[Any, Any] | ExtensionArraySupportsAnyAll" in supertype "ExtensionArray"
    def isna(self) -> Self:  # type: ignore[override]
        # If null fill value, we want SparseDtype[bool, true]
        # to preserve the same memory usage.
        dtype = SparseDtype(bool, self._null_fill_value)
        if self._null_fill_value:
            return type(self)._simple_new(isna(self.sp_values), self.sp_index, dtype)
        mask = np.full(len(self), False, dtype=np.bool_)
        mask[self.sp_index.indices] = isna(self.sp_values)
        return type(self)(mask, fill_value=False, dtype=dtype)

    def _pad_or_backfill(  # pylint: disable=useless-parent-delegation
        self,
        *,
        method: FillnaOptions,
        limit: int | None = None,
        limit_area: Literal["inside", "outside"] | None = None,
        copy: bool = True,
    ) -> Self:
        # TODO(3.0): We can remove this method once deprecation for fillna method
        #  keyword is enforced.
        return super()._pad_or_backfill(
            method=method, limit=limit, limit_area=limit_area, copy=copy
        )

    def fillna(
        self,
        value=None,
        method: FillnaOptions | None = None,
        limit: int | None = None,
        copy: bool = True,
    ) -> Self:
        """
        Fill missing values with `value`.

        Parameters
        ----------
        value : scalar, optional
        method : str, optional

            .. warning::

               Using 'method' will result in high memory use,
               as all `fill_value` methods will be converted to
               an in-memory ndarray

        limit : int, optional

        copy: bool, default True
            Ignored for SparseArray.

        Returns
        -------
        SparseArray

        Notes
        -----
        When `value` is specified, the result's ``fill_value`` depends on
        ``self.fill_value``. The goal is to maintain low-memory use.

        If ``self.fill_value`` is NA, the result dtype will be
        ``SparseDtype(self.dtype, fill_value=value)``. This will preserve
        amount of memory used before and after filling.

        When ``self.fill_value`` is not NA, the result dtype will be
        ``self.dtype``. Again, this preserves the amount of memory used.
        """
        if (method is None and value is None) or (
            method is not None and value is not None
        ):
            raise ValueError("Must specify one of 'method' or 'value'.")

        if method is not None:
            return super().fillna(method=method, limit=limit)

        else:
            new_values = np.where(isna(self.sp_values), value, self.sp_values)

            if self._null_fill_value:
                # This is essentially just updating the dtype.
                new_dtype = SparseDtype(self.dtype.subtype, fill_value=value)
            else:
                new_dtype = self.dtype

        return self._simple_new(new_values, self._sparse_index, new_dtype)

    def shift(self, periods: int = 1, fill_value=None) -> Self:
        if not len(self) or periods == 0:
            return self.copy()

        if isna(fill_value):
            fill_value = self.dtype.na_value

        subtype = np.result_type(fill_value, self.dtype.subtype)

        if subtype != self.dtype.subtype:
            # just coerce up front
            arr = self.astype(SparseDtype(subtype, self.fill_value))
        else:
            arr = self

        empty = self._from_sequence(
            [fill_value] * min(abs(periods), len(self)), dtype=arr.dtype
        )

        if periods > 0:
            a = empty
            b = arr[:-periods]
        else:
            a = arr[abs(periods) :]
            b = empty
        return arr._concat_same_type([a, b])

    def _first_fill_value_loc(self):
        """
        Get the location of the first fill value.

        Returns
        -------
        int
        """
        if len(self) == 0 or self.sp_index.npoints == len(self):
            return -1

        indices = self.sp_index.indices
        if not len(indices) or indices[0] > 0:
            return 0

        # a number larger than 1 should be appended to
        # the last in case of fill value only appears
        # in the tail of array
        diff = np.r_[np.diff(indices), 2]
        return indices[(diff > 1).argmax()] + 1

    @doc(ExtensionArray.duplicated)
    def duplicated(
        self, keep: Literal["first", "last", False] = "first"
    ) -> npt.NDArray[np.bool_]:
        values = np.asarray(self)
        mask = np.asarray(self.isna())
        return algos.duplicated(values, keep=keep, mask=mask)

    def unique(self) -> Self:
        uniques = algos.unique(self.sp_values)
        if len(self.sp_values) != len(self):
            fill_loc = self._first_fill_value_loc()
            # Inorder to align the behavior of pd.unique or
            # pd.Series.unique, we should keep the original
            # order, here we use unique again to find the
            # insertion place. Since the length of sp_values
            # is not large, maybe minor performance hurt
            # is worthwhile to the correctness.
            insert_loc = len(algos.unique(self.sp_values[:fill_loc]))
            uniques = np.insert(uniques, insert_loc, self.fill_value)
        return type(self)._from_sequence(uniques, dtype=self.dtype)

    def _values_for_factorize(self):
        # Still override this for hash_pandas_object
        return np.asarray(self), self.fill_value

    def factorize(
        self,
        use_na_sentinel: bool = True,
    ) -> tuple[np.ndarray, SparseArray]:
        # Currently, ExtensionArray.factorize -> Tuple[ndarray, EA]
        # The sparsity on this is backwards from what Sparse would want. Want
        # ExtensionArray.factorize -> Tuple[EA, EA]
        # Given that we have to return a dense array of codes, why bother
        # implementing an efficient factorize?
        codes, uniques = algos.factorize(
            np.asarray(self), use_na_sentinel=use_na_sentinel
        )
        uniques_sp = SparseArray(uniques, dtype=self.dtype)
        return codes, uniques_sp

    def value_counts(self, dropna: bool = True) -> Series:
        """
        Returns a Series containing counts of unique values.

        Parameters
        ----------
        dropna : bool, default True
            Don't include counts of NaN, even if NaN is in sp_values.

        Returns
        -------
        counts : Series
        """
        from pandas import (
            Index,
            Series,
        )

        keys, counts, _ = algos.value_counts_arraylike(self.sp_values, dropna=dropna)
        fcounts = self.sp_index.ngaps
        if fcounts > 0 and (not self._null_fill_value or not dropna):
            mask = isna(keys) if self._null_fill_value else keys == self.fill_value
            if mask.any():
                counts[mask] += fcounts
            else:
                # error: Argument 1 to "insert" has incompatible type "Union[
                # ExtensionArray,ndarray[Any, Any]]"; expected "Union[
                # _SupportsArray[dtype[Any]], Sequence[_SupportsArray[dtype
                # [Any]]], Sequence[Sequence[_SupportsArray[dtype[Any]]]],
                # Sequence[Sequence[Sequence[_SupportsArray[dtype[Any]]]]], Sequence
                # [Sequence[Sequence[Sequence[_SupportsArray[dtype[Any]]]]]]]"
                keys = np.insert(keys, 0, self.fill_value)  # type: ignore[arg-type]
                counts = np.insert(counts, 0, fcounts)

        if not isinstance(keys, ABCIndex):
            index = Index(keys)
        else:
            index = keys
        return Series(counts, index=index, copy=False)

    # --------
    # Indexing
    # --------
    @overload
    def __getitem__(self, key: ScalarIndexer) -> Any:
        ...

    @overload
    def __getitem__(
        self,
        key: SequenceIndexer | tuple[int | ellipsis, ...],
    ) -> Self:
        ...

    def __getitem__(
        self,
        key: PositionalIndexer | tuple[int | ellipsis, ...],
    ) -> Self | Any:
        if isinstance(key, tuple):
            key = unpack_tuple_and_ellipses(key)
            if key is Ellipsis:
                raise ValueError("Cannot slice with Ellipsis")

        if is_integer(key):
            return self._get_val_at(key)
        elif isinstance(key, tuple):
            # error: Invalid index type "Tuple[Union[int, ellipsis], ...]"
            # for "ndarray[Any, Any]"; expected type
            # "Union[SupportsIndex, _SupportsArray[dtype[Union[bool_,
            # integer[Any]]]], _NestedSequence[_SupportsArray[dtype[
            # Union[bool_, integer[Any]]]]], _NestedSequence[Union[
            # bool, int]], Tuple[Union[SupportsIndex, _SupportsArray[
            # dtype[Union[bool_, integer[Any]]]], _NestedSequence[
            # _SupportsArray[dtype[Union[bool_, integer[Any]]]]],
            # _NestedSequence[Union[bool, int]]], ...]]"
            data_slice = self.to_dense()[key]  # type: ignore[index]
        elif isinstance(key, slice):
            # Avoid densifying when handling contiguous slices
            if key.step is None or key.step == 1:
                start = 0 if key.start is None else key.start
                if start < 0:
                    start += len(self)

                end = len(self) if key.stop is None else key.stop
                if end < 0:
                    end += len(self)

                indices = self.sp_index.indices
                keep_inds = np.flatnonzero((indices >= start) & (indices < end))
                sp_vals = self.sp_values[keep_inds]

                sp_index = indices[keep_inds].copy()

                # If we've sliced to not include the start of the array, all our indices
                # should be shifted. NB: here we are careful to also not shift by a
                # negative value for a case like [0, 1][-100:] where the start index
                # should be treated like 0
                if start > 0:
                    sp_index -= start

                # Length of our result should match applying this slice to a range
                # of the length of our original array
                new_len = len(range(len(self))[key])
                new_sp_index = make_sparse_index(new_len, sp_index, self.kind)
                return type(self)._simple_new(sp_vals, new_sp_index, self.dtype)
            else:
                indices = np.arange(len(self), dtype=np.int32)[key]
                return self.take(indices)

        elif not is_list_like(key):
            # e.g. "foo" or 2.5
            # exception message copied from numpy
            raise IndexError(
                r"only integers, slices (`:`), ellipsis (`...`), numpy.newaxis "
                r"(`None`) and integer or boolean arrays are valid indices"
            )

        else:
            if isinstance(key, SparseArray):
                # NOTE: If we guarantee that SparseDType(bool)
                # has only fill_value - true, false or nan
                # (see GH PR 44955)
                # we can apply mask very fast:
                if is_bool_dtype(key):
                    if isna(key.fill_value):
                        return self.take(key.sp_index.indices[key.sp_values])
                    if not key.fill_value:
                        return self.take(key.sp_index.indices)
                    n = len(self)
                    mask = np.full(n, True, dtype=np.bool_)
                    mask[key.sp_index.indices] = False
                    return self.take(np.arange(n)[mask])
                else:
                    key = np.asarray(key)

            key = check_array_indexer(self, key)

            if com.is_bool_indexer(key):
                # mypy doesn't know we have an array here
                key = cast(np.ndarray, key)
                return self.take(np.arange(len(key), dtype=np.int32)[key])
            elif hasattr(key, "__len__"):
                return self.take(key)
            else:
                raise ValueError(f"Cannot slice with '{key}'")

        return type(self)(data_slice, kind=self.kind)

    def _get_val_at(self, loc):
        loc = validate_insert_loc(loc, len(self))

        sp_loc = self.sp_index.lookup(loc)
        if sp_loc == -1:
            return self.fill_value
        else:
            val = self.sp_values[sp_loc]
            val = maybe_box_datetimelike(val, self.sp_values.dtype)
            return val

    def take(self, indices, *, allow_fill: bool = False, fill_value=None) -> Self:
        if is_scalar(indices):
            raise ValueError(f"'indices' must be an array, not a scalar '{indices}'.")
        indices = np.asarray(indices, dtype=np.int32)

        dtype = None
        if indices.size == 0:
            result = np.array([], dtype="object")
            dtype = self.dtype
        elif allow_fill:
            result = self._take_with_fill(indices, fill_value=fill_value)
        else:
            return self._take_without_fill(indices)

        return type(self)(
            result, fill_value=self.fill_value, kind=self.kind, dtype=dtype
        )

    def _take_with_fill(self, indices, fill_value=None) -> np.ndarray:
        if fill_value is None:
            fill_value = self.dtype.na_value

        if indices.min() < -1:
            raise ValueError(
                "Invalid value in 'indices'. Must be between -1 "
                "and the length of the array."
            )

        if indices.max() >= len(self):
            raise IndexError("out of bounds value in 'indices'.")

        if len(self) == 0:
            # Empty... Allow taking only if all empty
            if (indices == -1).all():
                dtype = np.result_type(self.sp_values, type(fill_value))
                taken = np.empty_like(indices, dtype=dtype)
                taken.fill(fill_value)
                return taken
            else:
                raise IndexError("cannot do a non-empty take from an empty axes.")

        # sp_indexer may be -1 for two reasons
        # 1.) we took for an index of -1 (new)
        # 2.) we took a value that was self.fill_value (old)
        sp_indexer = self.sp_index.lookup_array(indices)
        new_fill_indices = indices == -1
        old_fill_indices = (sp_indexer == -1) & ~new_fill_indices

        if self.sp_index.npoints == 0 and old_fill_indices.all():
            # We've looked up all valid points on an all-sparse array.
            taken = np.full(
                sp_indexer.shape, fill_value=self.fill_value, dtype=self.dtype.subtype
            )

        elif self.sp_index.npoints == 0:
            # Use the old fill_value unless we took for an index of -1
            _dtype = np.result_type(self.dtype.subtype, type(fill_value))
            taken = np.full(sp_indexer.shape, fill_value=fill_value, dtype=_dtype)
            taken[old_fill_indices] = self.fill_value
        else:
            taken = self.sp_values.take(sp_indexer)

            # Fill in two steps.
            # Old fill values
            # New fill values
            # potentially coercing to a new dtype at each stage.

            m0 = sp_indexer[old_fill_indices] < 0
            m1 = sp_indexer[new_fill_indices] < 0

            result_type = taken.dtype

            if m0.any():
                result_type = np.result_type(result_type, type(self.fill_value))
                taken = taken.astype(result_type)
                taken[old_fill_indices] = self.fill_value

            if m1.any():
                result_type = np.result_type(result_type, type(fill_value))
                taken = taken.astype(result_type)
                taken[new_fill_indices] = fill_value

        return taken

    def _take_without_fill(self, indices) -> Self:
        to_shift = indices < 0

        n = len(self)

        if (indices.max() >= n) or (indices.min() < -n):
            if n == 0:
                raise IndexError("cannot do a non-empty take from an empty axes.")
            raise IndexError("out of bounds value in 'indices'.")

        if to_shift.any():
            indices = indices.copy()
            indices[to_shift] += n

        sp_indexer = self.sp_index.lookup_array(indices)
        value_mask = sp_indexer != -1
        new_sp_values = self.sp_values[sp_indexer[value_mask]]

        value_indices = np.flatnonzero(value_mask).astype(np.int32, copy=False)

        new_sp_index = make_sparse_index(len(indices), value_indices, kind=self.kind)
        return type(self)._simple_new(new_sp_values, new_sp_index, dtype=self.dtype)

    def searchsorted(
        self,
        v: ArrayLike | object,
        side: Literal["left", "right"] = "left",
        sorter: NumpySorter | None = None,
    ) -> npt.NDArray[np.intp] | np.intp:
        msg = "searchsorted requires high memory usage."
        warnings.warn(msg, PerformanceWarning, stacklevel=find_stack_level())
        v = np.asarray(v)
        return np.asarray(self, dtype=self.dtype.subtype).searchsorted(v, side, sorter)

    def copy(self) -> Self:
        values = self.sp_values.copy()
        return self._simple_new(values, self.sp_index, self.dtype)

    @classmethod
    def _concat_same_type(cls, to_concat: Sequence[Self]) -> Self:
        fill_value = to_concat[0].fill_value

        values = []
        length = 0

        if to_concat:
            sp_kind = to_concat[0].kind
        else:
            sp_kind = "integer"

        sp_index: SparseIndex
        if sp_kind == "integer":
            indices = []

            for arr in to_concat:
                int_idx = arr.sp_index.indices.copy()
                int_idx += length  # TODO: wraparound
                length += arr.sp_index.length

                values.append(arr.sp_values)
                indices.append(int_idx)

            data = np.concatenate(values)
            indices_arr = np.concatenate(indices)
            # error: Argument 2 to "IntIndex" has incompatible type
            # "ndarray[Any, dtype[signedinteger[_32Bit]]]";
            # expected "Sequence[int]"
            sp_index = IntIndex(length, indices_arr)  # type: ignore[arg-type]

        else:
            # when concatenating block indices, we don't claim that you'll
            # get an identical index as concatenating the values and then
            # creating a new index. We don't want to spend the time trying
            # to merge blocks across arrays in `to_concat`, so the resulting
            # BlockIndex may have more blocks.
            blengths = []
            blocs = []

            for arr in to_concat:
                block_idx = arr.sp_index.to_block_index()

                values.append(arr.sp_values)
                blocs.append(block_idx.blocs.copy() + length)
                blengths.append(block_idx.blengths)
                length += arr.sp_index.length

            data = np.concatenate(values)
            blocs_arr = np.concatenate(blocs)
            blengths_arr = np.concatenate(blengths)

            sp_index = BlockIndex(length, blocs_arr, blengths_arr)

        return cls(data, sparse_index=sp_index, fill_value=fill_value)

    def astype(self, dtype: AstypeArg | None = None, copy: bool = True):
        """
        Change the dtype of a SparseArray.

        The output will always be a SparseArray. To convert to a dense
        ndarray with a certain dtype, use :meth:`numpy.asarray`.

        Parameters
        ----------
        dtype : np.dtype or ExtensionDtype
            For SparseDtype, this changes the dtype of
            ``self.sp_values`` and the ``self.fill_value``.

            For other dtypes, this only changes the dtype of
            ``self.sp_values``.

        copy : bool, default True
            Whether to ensure a copy is made, even if not necessary.

        Returns
        -------
        SparseArray

        Examples
        --------
        >>> arr = pd.arrays.SparseArray([0, 0, 1, 2])
        >>> arr
        [0, 0, 1, 2]
        Fill: 0
        IntIndex
        Indices: array([2, 3], dtype=int32)

        >>> arr.astype(SparseDtype(np.dtype('int32')))
        [0, 0, 1, 2]
        Fill: 0
        IntIndex
        Indices: array([2, 3], dtype=int32)

        Using a NumPy dtype with a different kind (e.g. float) will coerce
        just ``self.sp_values``.

        >>> arr.astype(SparseDtype(np.dtype('float64')))
        ... # doctest: +NORMALIZE_WHITESPACE
        [nan, nan, 1.0, 2.0]
        Fill: nan
        IntIndex
        Indices: array([2, 3], dtype=int32)

        Using a SparseDtype, you can also change the fill value as well.

        >>> arr.astype(SparseDtype("float64", fill_value=0.0))
        ... # doctest: +NORMALIZE_WHITESPACE
        [0.0, 0.0, 1.0, 2.0]
        Fill: 0.0
        IntIndex
        Indices: array([2, 3], dtype=int32)
        """
        if dtype == self._dtype:
            if not copy:
                return self
            else:
                return self.copy()

        future_dtype = pandas_dtype(dtype)
        if not isinstance(future_dtype, SparseDtype):
            # GH#34457
            values = np.asarray(self)
            values = ensure_wrapped_if_datetimelike(values)
            return astype_array(values, dtype=future_dtype, copy=False)

        dtype = self.dtype.update_dtype(dtype)
        subtype = pandas_dtype(dtype._subtype_with_str)
        subtype = cast(np.dtype, subtype)  # ensured by update_dtype
        values = ensure_wrapped_if_datetimelike(self.sp_values)
        sp_values = astype_array(values, subtype, copy=copy)
        sp_values = np.asarray(sp_values)

        return self._simple_new(sp_values, self.sp_index, dtype)

    def map(self, mapper, na_action=None) -> Self:
        """
        Map categories using an input mapping or function.

        Parameters
        ----------
        mapper : dict, Series, callable
            The correspondence from old values to new.
        na_action : {None, 'ignore'}, default None
            If 'ignore', propagate NA values, without passing them to the
            mapping correspondence.

        Returns
        -------
        SparseArray
            The output array will have the same density as the input.
            The output fill value will be the result of applying the
            mapping to ``self.fill_value``

        Examples
        --------
        >>> arr = pd.arrays.SparseArray([0, 1, 2])
        >>> arr.map(lambda x: x + 10)
        [10, 11, 12]
        Fill: 10
        IntIndex
        Indices: array([1, 2], dtype=int32)

        >>> arr.map({0: 10, 1: 11, 2: 12})
        [10, 11, 12]
        Fill: 10
        IntIndex
        Indices: array([1, 2], dtype=int32)

        >>> arr.map(pd.Series([10, 11, 12], index=[0, 1, 2]))
        [10, 11, 12]
        Fill: 10
        IntIndex
        Indices: array([1, 2], dtype=int32)
        """
        is_map = isinstance(mapper, (abc.Mapping, ABCSeries))

        fill_val = self.fill_value

        if na_action is None or notna(fill_val):
            fill_val = mapper.get(fill_val, fill_val) if is_map else mapper(fill_val)

        def func(sp_val):
            new_sp_val = mapper.get(sp_val, None) if is_map else mapper(sp_val)
            # check identity and equality because nans are not equal to each other
            if new_sp_val is fill_val or new_sp_val == fill_val:
                msg = "fill value in the sparse values not supported"
                raise ValueError(msg)
            return new_sp_val

        sp_values = [func(x) for x in self.sp_values]

        return type(self)(sp_values, sparse_index=self.sp_index, fill_value=fill_val)

    def to_dense(self) -> np.ndarray:
        """
        Convert SparseArray to a NumPy array.

        Returns
        -------
        arr : NumPy array
        """
        return np.asarray(self, dtype=self.sp_values.dtype)

    def _where(self, mask, value):
        # NB: may not preserve dtype, e.g. result may be Sparse[float64]
        #  while self is Sparse[int64]
        naive_implementation = np.where(mask, self, value)
        dtype = SparseDtype(naive_implementation.dtype, fill_value=self.fill_value)
        result = type(self)._from_sequence(naive_implementation, dtype=dtype)
        return result

    # ------------------------------------------------------------------------
    # IO
    # ------------------------------------------------------------------------
    def __setstate__(self, state) -> None:
        """Necessary for making this object picklable"""
        if isinstance(state, tuple):
            # Compat for pandas < 0.24.0
            nd_state, (fill_value, sp_index) = state
            sparse_values = np.array([])
            sparse_values.__setstate__(nd_state)

            self._sparse_values = sparse_values
            self._sparse_index = sp_index
            self._dtype = SparseDtype(sparse_values.dtype, fill_value)
        else:
            self.__dict__.update(state)

    def nonzero(self) -> tuple[npt.NDArray[np.int32]]:
        if self.fill_value == 0:
            return (self.sp_index.indices,)
        else:
            return (self.sp_index.indices[self.sp_values != 0],)

    # ------------------------------------------------------------------------
    # Reductions
    # ------------------------------------------------------------------------

    def _reduce(
        self, name: str, *, skipna: bool = True, keepdims: bool = False, **kwargs
    ):
        method = getattr(self, name, None)

        if method is None:
            raise TypeError(f"cannot perform {name} with type {self.dtype}")

        if skipna:
            arr = self
        else:
            arr = self.dropna()

        result = getattr(arr, name)(**kwargs)

        if keepdims:
            return type(self)([result], dtype=self.dtype)
        else:
            return result

    def all(self, axis=None, *args, **kwargs):
        """
        Tests whether all elements evaluate True

        Returns
        -------
        all : bool

        See Also
        --------
        numpy.all
        """
        nv.validate_all(args, kwargs)

        values = self.sp_values

        if len(values) != len(self) and not np.all(self.fill_value):
            return False

        return values.all()

    def any(self, axis: AxisInt = 0, *args, **kwargs) -> bool:
        """
        Tests whether at least one of elements evaluate True

        Returns
        -------
        any : bool

        See Also
        --------
        numpy.any
        """
        nv.validate_any(args, kwargs)

        values = self.sp_values

        if len(values) != len(self) and np.any(self.fill_value):
            return True

        return values.any().item()

    def sum(
        self,
        axis: AxisInt = 0,
        min_count: int = 0,
        skipna: bool = True,
        *args,
        **kwargs,
    ) -> Scalar:
        """
        Sum of non-NA/null values

        Parameters
        ----------
        axis : int, default 0
            Not Used. NumPy compatibility.
        min_count : int, default 0
            The required number of valid values to perform the summation. If fewer
            than ``min_count`` valid values are present, the result will be the missing
            value indicator for subarray type.
        *args, **kwargs
            Not Used. NumPy compatibility.

        Returns
        -------
        scalar
        """
        nv.validate_sum(args, kwargs)
        valid_vals = self._valid_sp_values
        sp_sum = valid_vals.sum()
        has_na = self.sp_index.ngaps > 0 and not self._null_fill_value

        if has_na and not skipna:
            return na_value_for_dtype(self.dtype.subtype, compat=False)

        if self._null_fill_value:
            if check_below_min_count(valid_vals.shape, None, min_count):
                return na_value_for_dtype(self.dtype.subtype, compat=False)
            return sp_sum
        else:
            nsparse = self.sp_index.ngaps
            if check_below_min_count(valid_vals.shape, None, min_count - nsparse):
                return na_value_for_dtype(self.dtype.subtype, compat=False)
            return sp_sum + self.fill_value * nsparse

    def cumsum(self, axis: AxisInt = 0, *args, **kwargs) -> SparseArray:
        """
        Cumulative sum of non-NA/null values.

        When performing the cumulative summation, any non-NA/null values will
        be skipped. The resulting SparseArray will preserve the locations of
        NaN values, but the fill value will be `np.nan` regardless.

        Parameters
        ----------
        axis : int or None
            Axis over which to perform the cumulative summation. If None,
            perform cumulative summation over flattened array.

        Returns
        -------
        cumsum : SparseArray
        """
        nv.validate_cumsum(args, kwargs)

        if axis is not None and axis >= self.ndim:  # Mimic ndarray behaviour.
            raise ValueError(f"axis(={axis}) out of bounds")

        if not self._null_fill_value:
            return SparseArray(self.to_dense()).cumsum()

        return SparseArray(
            self.sp_values.cumsum(),
            sparse_index=self.sp_index,
            fill_value=self.fill_value,
        )

    def mean(self, axis: Axis = 0, *args, **kwargs):
        """
        Mean of non-NA/null values

        Returns
        -------
        mean : float
        """
        nv.validate_mean(args, kwargs)
        valid_vals = self._valid_sp_values
        sp_sum = valid_vals.sum()
        ct = len(valid_vals)

        if self._null_fill_value:
            return sp_sum / ct
        else:
            nsparse = self.sp_index.ngaps
            return (sp_sum + self.fill_value * nsparse) / (ct + nsparse)

    def max(self, *, axis: AxisInt | None = None, skipna: bool = True):
        """
        Max of array values, ignoring NA values if specified.

        Parameters
        ----------
        axis : int, default 0
            Not Used. NumPy compatibility.
        skipna : bool, default True
            Whether to ignore NA values.

        Returns
        -------
        scalar
        """
        nv.validate_minmax_axis(axis, self.ndim)
        return self._min_max("max", skipna=skipna)

    def min(self, *, axis: AxisInt | None = None, skipna: bool = True):
        """
        Min of array values, ignoring NA values if specified.

        Parameters
        ----------
        axis : int, default 0
            Not Used. NumPy compatibility.
        skipna : bool, default True
            Whether to ignore NA values.

        Returns
        -------
        scalar
        """
        nv.validate_minmax_axis(axis, self.ndim)
        return self._min_max("min", skipna=skipna)

    def _min_max(self, kind: Literal["min", "max"], skipna: bool) -> Scalar:
        """
        Min/max of non-NA/null values

        Parameters
        ----------
        kind : {"min", "max"}
        skipna : bool

        Returns
        -------
        scalar
        """
        valid_vals = self._valid_sp_values
        has_nonnull_fill_vals = not self._null_fill_value and self.sp_index.ngaps > 0

        if len(valid_vals) > 0:
            sp_min_max = getattr(valid_vals, kind)()

            # If a non-null fill value is currently present, it might be the min/max
            if has_nonnull_fill_vals:
                func = max if kind == "max" else min
                return func(sp_min_max, self.fill_value)
            elif skipna:
                return sp_min_max
            elif self.sp_index.ngaps == 0:
                # No NAs present
                return sp_min_max
            else:
                return na_value_for_dtype(self.dtype.subtype, compat=False)
        elif has_nonnull_fill_vals:
            return self.fill_value
        else:
            return na_value_for_dtype(self.dtype.subtype, compat=False)

    def _argmin_argmax(self, kind: Literal["argmin", "argmax"]) -> int:
        values = self._sparse_values
        index = self._sparse_index.indices
        mask = np.asarray(isna(values))
        func = np.argmax if kind == "argmax" else np.argmin

        idx = np.arange(values.shape[0])
        non_nans = values[~mask]
        non_nan_idx = idx[~mask]

        _candidate = non_nan_idx[func(non_nans)]
        candidate = index[_candidate]

        if isna(self.fill_value):
            return candidate
        if kind == "argmin" and self[candidate] < self.fill_value:
            return candidate
        if kind == "argmax" and self[candidate] > self.fill_value:
            return candidate
        _loc = self._first_fill_value_loc()
        if _loc == -1:
            # fill_value doesn't exist
            return candidate
        else:
            return _loc

    def argmax(self, skipna: bool = True) -> int:
        validate_bool_kwarg(skipna, "skipna")
        if not skipna and self._hasna:
            raise NotImplementedError
        return self._argmin_argmax("argmax")

    def argmin(self, skipna: bool = True) -> int:
        validate_bool_kwarg(skipna, "skipna")
        if not skipna and self._hasna:
            raise NotImplementedError
        return self._argmin_argmax("argmin")

    # ------------------------------------------------------------------------
    # Ufuncs
    # ------------------------------------------------------------------------

    _HANDLED_TYPES = (np.ndarray, numbers.Number)

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
        out = kwargs.get("out", ())

        for x in inputs + out:
            if not isinstance(x, self._HANDLED_TYPES + (SparseArray,)):
                return NotImplemented

        # for binary ops, use our custom dunder methods
        result = arraylike.maybe_dispatch_ufunc_to_dunder_op(
            self, ufunc, method, *inputs, **kwargs
        )
        if result is not NotImplemented:
            return result

        if "out" in kwargs:
            # e.g. tests.arrays.sparse.test_arithmetics.test_ndarray_inplace
            res = arraylike.dispatch_ufunc_with_out(
                self, ufunc, method, *inputs, **kwargs
            )
            return res

        if method == "reduce":
            result = arraylike.dispatch_reduction_ufunc(
                self, ufunc, method, *inputs, **kwargs
            )
            if result is not NotImplemented:
                # e.g. tests.series.test_ufunc.TestNumpyReductions
                return result

        if len(inputs) == 1:
            # No alignment necessary.
            sp_values = getattr(ufunc, method)(self.sp_values, **kwargs)
            fill_value = getattr(ufunc, method)(self.fill_value, **kwargs)

            if ufunc.nout > 1:
                # multiple outputs. e.g. modf
                arrays = tuple(
                    self._simple_new(
                        sp_value, self.sp_index, SparseDtype(sp_value.dtype, fv)
                    )
                    for sp_value, fv in zip(sp_values, fill_value)
                )
                return arrays
            elif method == "reduce":
                # e.g. reductions
                return sp_values

            return self._simple_new(
                sp_values, self.sp_index, SparseDtype(sp_values.dtype, fill_value)
            )

        new_inputs = tuple(np.asarray(x) for x in inputs)
        result = getattr(ufunc, method)(*new_inputs, **kwargs)
        if out:
            if len(out) == 1:
                out = out[0]
            return out

        if ufunc.nout > 1:
            return tuple(type(self)(x) for x in result)
        elif method == "at":
            # no return value
            return None
        else:
            return type(self)(result)

    # ------------------------------------------------------------------------
    # Ops
    # ------------------------------------------------------------------------

    def _arith_method(self, other, op):
        op_name = op.__name__

        if isinstance(other, SparseArray):
            return _sparse_array_op(self, other, op, op_name)

        elif is_scalar(other):
            with np.errstate(all="ignore"):
                fill = op(_get_fill(self), np.asarray(other))
                result = op(self.sp_values, other)

            if op_name == "divmod":
                left, right = result
                lfill, rfill = fill
                return (
                    _wrap_result(op_name, left, self.sp_index, lfill),
                    _wrap_result(op_name, right, self.sp_index, rfill),
                )

            return _wrap_result(op_name, result, self.sp_index, fill)

        else:
            other = np.asarray(other)
            with np.errstate(all="ignore"):
                if len(self) != len(other):
                    raise AssertionError(
                        f"length mismatch: {len(self)} vs. {len(other)}"
                    )
                if not isinstance(other, SparseArray):
                    dtype = getattr(other, "dtype", None)
                    other = SparseArray(other, fill_value=self.fill_value, dtype=dtype)
                return _sparse_array_op(self, other, op, op_name)

    def _cmp_method(self, other, op) -> SparseArray:
        if not is_scalar(other) and not isinstance(other, type(self)):
            # convert list-like to ndarray
            other = np.asarray(other)

        if isinstance(other, np.ndarray):
            # TODO: make this more flexible than just ndarray...
            other = SparseArray(other, fill_value=self.fill_value)

        if isinstance(other, SparseArray):
            if len(self) != len(other):
                raise ValueError(
                    f"operands have mismatched length {len(self)} and {len(other)}"
                )

            op_name = op.__name__.strip("_")
            return _sparse_array_op(self, other, op, op_name)
        else:
            # scalar
            fill_value = op(self.fill_value, other)
            result = np.full(len(self), fill_value, dtype=np.bool_)
            result[self.sp_index.indices] = op(self.sp_values, other)

            return type(self)(
                result,
                fill_value=fill_value,
                dtype=np.bool_,
            )

    _logical_method = _cmp_method

    def _unary_method(self, op) -> SparseArray:
        fill_value = op(np.array(self.fill_value)).item()
        dtype = SparseDtype(self.dtype.subtype, fill_value)
        # NOTE: if fill_value doesn't change
        # we just have to apply op to sp_values
        if isna(self.fill_value) or fill_value == self.fill_value:
            values = op(self.sp_values)
            return type(self)._simple_new(values, self.sp_index, self.dtype)
        # In the other case we have to recalc indexes
        return type(self)(op(self.to_dense()), dtype=dtype)

    def __pos__(self) -> SparseArray:
        return self._unary_method(operator.pos)

    def __neg__(self) -> SparseArray:
        return self._unary_method(operator.neg)

    def __invert__(self) -> SparseArray:
        return self._unary_method(operator.invert)

    def __abs__(self) -> SparseArray:
        return self._unary_method(operator.abs)

    # ----------
    # Formatting
    # -----------
    def __repr__(self) -> str:
        pp_str = printing.pprint_thing(self)
        pp_fill = printing.pprint_thing(self.fill_value)
        pp_index = printing.pprint_thing(self.sp_index)
        return f"{pp_str}\nFill: {pp_fill}\n{pp_index}"

    def _formatter(self, boxed: bool = False):
        # Defer to the formatter from the GenericArrayFormatter calling us.
        # This will infer the correct formatter from the dtype of the values.
        return None


def _make_sparse(
    arr: np.ndarray,
    kind: SparseIndexKind = "block",
    fill_value=None,
    dtype: np.dtype | None = None,
):
    """
    Convert ndarray to sparse format

    Parameters
    ----------
    arr : ndarray
    kind : {'block', 'integer'}
    fill_value : NaN or another value
    dtype : np.dtype, optional
    copy : bool, default False

    Returns
    -------
    (sparse_values, index, fill_value) : (ndarray, SparseIndex, Scalar)
    """
    assert isinstance(arr, np.ndarray)

    if arr.ndim > 1:
        raise TypeError("expected dimension <= 1 data")

    if fill_value is None:
        fill_value = na_value_for_dtype(arr.dtype)

    if isna(fill_value):
        mask = notna(arr)
    else:
        # cast to object comparison to be safe
        if is_string_dtype(arr.dtype):
            arr = arr.astype(object)

        if is_object_dtype(arr.dtype):
            # element-wise equality check method in numpy doesn't treat
            # each element type, eg. 0, 0.0, and False are treated as
            # same. So we have to check the both of its type and value.
            mask = splib.make_mask_object_ndarray(arr, fill_value)
        else:
            mask = arr != fill_value

    length = len(arr)
    if length != len(mask):
        # the arr is a SparseArray
        indices = mask.sp_index.indices
    else:
        indices = mask.nonzero()[0].astype(np.int32)

    index = make_sparse_index(length, indices, kind)
    sparsified_values = arr[mask]
    if dtype is not None:
        sparsified_values = ensure_wrapped_if_datetimelike(sparsified_values)
        sparsified_values = astype_array(sparsified_values, dtype=dtype)
        sparsified_values = np.asarray(sparsified_values)

    # TODO: copy
    return sparsified_values, index, fill_value


@overload
def make_sparse_index(length: int, indices, kind: Literal["block"]) -> BlockIndex:
    ...


@overload
def make_sparse_index(length: int, indices, kind: Literal["integer"]) -> IntIndex:
    ...


def make_sparse_index(length: int, indices, kind: SparseIndexKind) -> SparseIndex:
    index: SparseIndex
    if kind == "block":
        locs, lens = splib.get_blocks(indices)
        index = BlockIndex(length, locs, lens)
    elif kind == "integer":
        index = IntIndex(length, indices)
    else:  # pragma: no cover
        raise ValueError("must be block or integer type")
    return index
<|endoftext|>from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pyarrow

from pandas.compat import pa_version_under14p1

from pandas.core.dtypes.dtypes import (
    IntervalDtype,
    PeriodDtype,
)

from pandas.core.arrays.interval import VALID_CLOSED

if TYPE_CHECKING:
    from pandas._typing import IntervalClosedType


class ArrowPeriodType(pyarrow.ExtensionType):
    def __init__(self, freq) -> None:
        # attributes need to be set first before calling
        # super init (as that calls serialize)
        self._freq = freq
        pyarrow.ExtensionType.__init__(self, pyarrow.int64(), "pandas.period")

    @property
    def freq(self):
        return self._freq

    def __arrow_ext_serialize__(self) -> bytes:
        metadata = {"freq": self.freq}
        return json.dumps(metadata).encode()

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized) -> ArrowPeriodType:
        metadata = json.loads(serialized.decode())
        return ArrowPeriodType(metadata["freq"])

    def __eq__(self, other):
        if isinstance(other, pyarrow.BaseExtensionType):
            return type(self) == type(other) and self.freq == other.freq
        else:
            return NotImplemented

    def __ne__(self, other) -> bool:
        return not self == other

    def __hash__(self) -> int:
        return hash((str(self), self.freq))

    def to_pandas_dtype(self) -> PeriodDtype:
        return PeriodDtype(freq=self.freq)


# register the type with a dummy instance
_period_type = ArrowPeriodType("D")
pyarrow.register_extension_type(_period_type)


class ArrowIntervalType(pyarrow.ExtensionType):
    def __init__(self, subtype, closed: IntervalClosedType) -> None:
        # attributes need to be set first before calling
        # super init (as that calls serialize)
        assert closed in VALID_CLOSED
        self._closed: IntervalClosedType = closed
        if not isinstance(subtype, pyarrow.DataType):
            subtype = pyarrow.type_for_alias(str(subtype))
        self._subtype = subtype

        storage_type = pyarrow.struct([("left", subtype), ("right", subtype)])
        pyarrow.ExtensionType.__init__(self, storage_type, "pandas.interval")

    @property
    def subtype(self):
        return self._subtype

    @property
    def closed(self) -> IntervalClosedType:
        return self._closed

    def __arrow_ext_serialize__(self) -> bytes:
        metadata = {"subtype": str(self.subtype), "closed": self.closed}
        return json.dumps(metadata).encode()

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized) -> ArrowIntervalType:
        metadata = json.loads(serialized.decode())
        subtype = pyarrow.type_for_alias(metadata["subtype"])
        closed = metadata["closed"]
        return ArrowIntervalType(subtype, closed)

    def __eq__(self, other):
        if isinstance(other, pyarrow.BaseExtensionType):
            return (
                type(self) == type(other)
                and self.subtype == other.subtype
                and self.closed == other.closed
            )
        else:
            return NotImplemented

    def __ne__(self, other) -> bool:
        return not self == other

    def __hash__(self) -> int:
        return hash((str(self), str(self.subtype), self.closed))

    def to_pandas_dtype(self) -> IntervalDtype:
        return IntervalDtype(self.subtype.to_pandas_dtype(), self.closed)


# register the type with a dummy instance
_interval_type = ArrowIntervalType(pyarrow.int64(), "left")
pyarrow.register_extension_type(_interval_type)


_ERROR_MSG = """\
Disallowed deserialization of 'arrow.py_extension_type':
storage_type = {storage_type}
serialized = {serialized}
pickle disassembly:\n{pickle_disassembly}

Reading of untrusted Parquet or Feather files with a PyExtensionType column
allows arbitrary code execution.
If you trust this file, you can enable reading the extension type by one of:

- upgrading to pyarrow >= 14.0.1, and call `pa.PyExtensionType.set_auto_load(True)`
- install pyarrow-hotfix (`pip install pyarrow-hotfix`) and disable it by running
  `import pyarrow_hotfix; pyarrow_hotfix.uninstall()`

We strongly recommend updating your Parquet/Feather files to use extension types
derived from `pyarrow.ExtensionType` instead, and register this type explicitly.
"""


def patch_pyarrow():
    # starting from pyarrow 14.0.1, it has its own mechanism
    if not pa_version_under14p1:
        return

    # if https://github.com/pitrou/pyarrow-hotfix was installed and enabled
    if getattr(pyarrow, "_hotfix_installed", False):
        return

    class ForbiddenExtensionType(pyarrow.ExtensionType):
        def __arrow_ext_serialize__(self):
            return b""

        @classmethod
        def __arrow_ext_deserialize__(cls, storage_type, serialized):
            import io
            import pickletools

            out = io.StringIO()
            pickletools.dis(serialized, out)
            raise RuntimeError(
                _ERROR_MSG.format(
                    storage_type=storage_type,
                    serialized=serialized,
                    pickle_disassembly=out.getvalue(),
                )
            )

    pyarrow.unregister_extension_type("arrow.py_extension_type")
    pyarrow.register_extension_type(
        ForbiddenExtensionType(pyarrow.null(), "arrow.py_extension_type")
    )

    pyarrow._hotfix_installed = True


patch_pyarrow()
<|endoftext|>from __future__ import annotations

import numpy as np
import pyarrow


def pyarrow_array_to_numpy_and_mask(
    arr, dtype: np.dtype
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a primitive pyarrow.Array to a numpy array and boolean mask based
    on the buffers of the Array.

    At the moment pyarrow.BooleanArray is not supported.

    Parameters
    ----------
    arr : pyarrow.Array
    dtype : numpy.dtype

    Returns
    -------
    (data, mask)
        Tuple of two numpy arrays with the raw data (with specified dtype) and
        a boolean mask (validity mask, so False means missing)
    """
    dtype = np.dtype(dtype)

    if pyarrow.types.is_null(arr.type):
        # No initialization of data is needed since everything is null
        data = np.empty(len(arr), dtype=dtype)
        mask = np.zeros(len(arr), dtype=bool)
        return data, mask
    buflist = arr.buffers()
    # Since Arrow buffers might contain padding and the data might be offset,
    # the buffer gets sliced here before handing it to numpy.
    # See also https://github.com/pandas-dev/pandas/issues/40896
    offset = arr.offset * dtype.itemsize
    length = len(arr) * dtype.itemsize
    data_buf = buflist[1][offset : offset + length]
    data = np.frombuffer(data_buf, dtype=dtype)
    bitmask = buflist[0]
    if bitmask is not None:
        mask = pyarrow.BooleanArray.from_buffers(
            pyarrow.bool_(), len(arr), [None, bitmask], offset=arr.offset
        )
        mask = np.asarray(mask)
    else:
        mask = np.ones(len(arr), dtype=bool)
    return data, mask
<|endoftext|>"""Accessors for arrow-backed data."""

from __future__ import annotations

from abc import (
    ABCMeta,
    abstractmethod,
)
from typing import (
    TYPE_CHECKING,
    cast,
)

from pandas.compat import (
    pa_version_under10p1,
    pa_version_under11p0,
)

from pandas.core.dtypes.common import is_list_like

if not pa_version_under10p1:
    import pyarrow as pa
    import pyarrow.compute as pc

    from pandas.core.dtypes.dtypes import ArrowDtype

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pandas import (
        DataFrame,
        Series,
    )


class ArrowAccessor(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, data, validation_msg: str) -> None:
        self._data = data
        self._validation_msg = validation_msg
        self._validate(data)

    @abstractmethod
    def _is_valid_pyarrow_dtype(self, pyarrow_dtype) -> bool:
        pass

    def _validate(self, data):
        dtype = data.dtype
        if pa_version_under10p1 or not isinstance(dtype, ArrowDtype):
            # Raise AttributeError so that inspect can handle non-struct Series.
            raise AttributeError(self._validation_msg.format(dtype=dtype))

        if not self._is_valid_pyarrow_dtype(dtype.pyarrow_dtype):
            # Raise AttributeError so that inspect can handle invalid Series.
            raise AttributeError(self._validation_msg.format(dtype=dtype))

    @property
    def _pa_array(self):
        return self._data.array._pa_array


class ListAccessor(ArrowAccessor):
    """
    Accessor object for list data properties of the Series values.

    Parameters
    ----------
    data : Series
        Series containing Arrow list data.
    """

    def __init__(self, data=None) -> None:
        super().__init__(
            data,
            validation_msg="Can only use the '.list' accessor with "
            "'list[pyarrow]' dtype, not {dtype}.",
        )

    def _is_valid_pyarrow_dtype(self, pyarrow_dtype) -> bool:
        return (
            pa.types.is_list(pyarrow_dtype)
            or pa.types.is_fixed_size_list(pyarrow_dtype)
            or pa.types.is_large_list(pyarrow_dtype)
        )

    def len(self) -> Series:
        """
        Return the length of each list in the Series.

        Returns
        -------
        pandas.Series
            The length of each list.

        Examples
        --------
        >>> import pyarrow as pa
        >>> s = pd.Series(
        ...     [
        ...         [1, 2, 3],
        ...         [3],
        ...     ],
        ...     dtype=pd.ArrowDtype(pa.list_(
        ...         pa.int64()
        ...     ))
        ... )
        >>> s.list.len()
        0    3
        1    1
        dtype: int32[pyarrow]
        """
        from pandas import Series

        value_lengths = pc.list_value_length(self._pa_array)
        return Series(value_lengths, dtype=ArrowDtype(value_lengths.type))

    def __getitem__(self, key: int | slice) -> Series:
        """
        Index or slice lists in the Series.

        Parameters
        ----------
        key : int | slice
            Index or slice of indices to access from each list.

        Returns
        -------
        pandas.Series
            The list at requested index.

        Examples
        --------
        >>> import pyarrow as pa
        >>> s = pd.Series(
        ...     [
        ...         [1, 2, 3],
        ...         [3],
        ...     ],
        ...     dtype=pd.ArrowDtype(pa.list_(
        ...         pa.int64()
        ...     ))
        ... )
        >>> s.list[0]
        0    1
        1    3
        dtype: int64[pyarrow]
        """
        from pandas import Series

        if isinstance(key, int):
            # TODO: Support negative key but pyarrow does not allow
            # element index to be an array.
            # if key < 0:
            #     key = pc.add(key, pc.list_value_length(self._pa_array))
            element = pc.list_element(self._pa_array, key)
            return Series(element, dtype=ArrowDtype(element.type))
        elif isinstance(key, slice):
            if pa_version_under11p0:
                raise NotImplementedError(
                    f"List slice not supported by pyarrow {pa.__version__}."
                )

            # TODO: Support negative start/stop/step, ideally this would be added
            # upstream in pyarrow.
            start, stop, step = key.start, key.stop, key.step
            if start is None:
                # TODO: When adding negative step support
                #  this should be setto last element of array
                # when step is negative.
                start = 0
            if step is None:
                step = 1
            sliced = pc.list_slice(self._pa_array, start, stop, step)
            return Series(sliced, dtype=ArrowDtype(sliced.type))
        else:
            raise ValueError(f"key must be an int or slice, got {type(key).__name__}")

    def __iter__(self) -> Iterator:
        raise TypeError(f"'{type(self).__name__}' object is not iterable")

    def flatten(self) -> Series:
        """
        Flatten list values.

        Returns
        -------
        pandas.Series
            The data from all lists in the series flattened.

        Examples
        --------
        >>> import pyarrow as pa
        >>> s = pd.Series(
        ...     [
        ...         [1, 2, 3],
        ...         [3],
        ...     ],
        ...     dtype=pd.ArrowDtype(pa.list_(
        ...         pa.int64()
        ...     ))
        ... )
        >>> s.list.flatten()
        0    1
        1    2
        2    3
        3    3
        dtype: int64[pyarrow]
        """
        from pandas import Series

        flattened = pc.list_flatten(self._pa_array)
        return Series(flattened, dtype=ArrowDtype(flattened.type))


class StructAccessor(ArrowAccessor):
    """
    Accessor object for structured data properties of the Series values.

    Parameters
    ----------
    data : Series
        Series containing Arrow struct data.
    """

    def __init__(self, data=None) -> None:
        super().__init__(
            data,
            validation_msg=(
                "Can only use the '.struct' accessor with 'struct[pyarrow]' "
                "dtype, not {dtype}."
            ),
        )

    def _is_valid_pyarrow_dtype(self, pyarrow_dtype) -> bool:
        return pa.types.is_struct(pyarrow_dtype)

    @property
    def dtypes(self) -> Series:
        """
        Return the dtype object of each child field of the struct.

        Returns
        -------
        pandas.Series
            The data type of each child field.

        Examples
        --------
        >>> import pyarrow as pa
        >>> s = pd.Series(
        ...     [
        ...         {"version": 1, "project": "pandas"},
        ...         {"version": 2, "project": "pandas"},
        ...         {"version": 1, "project": "numpy"},
        ...     ],
        ...     dtype=pd.ArrowDtype(pa.struct(
        ...         [("version", pa.int64()), ("project", pa.string())]
        ...     ))
        ... )
        >>> s.struct.dtypes
        version     int64[pyarrow]
        project    string[pyarrow]
        dtype: object
        """
        from pandas import (
            Index,
            Series,
        )

        pa_type = self._data.dtype.pyarrow_dtype
        types = [ArrowDtype(struct.type) for struct in pa_type]
        names = [struct.name for struct in pa_type]
        return Series(types, index=Index(names))

    def field(
        self,
        name_or_index: list[str]
        | list[bytes]
        | list[int]
        | pc.Expression
        | bytes
        | str
        | int,
    ) -> Series:
        """
        Extract a child field of a struct as a Series.

        Parameters
        ----------
        name_or_index : str | bytes | int | expression | list
            Name or index of the child field to extract.

            For list-like inputs, this will index into a nested
            struct.

        Returns
        -------
        pandas.Series
            The data corresponding to the selected child field.

        See Also
        --------
        Series.struct.explode : Return all child fields as a DataFrame.

        Notes
        -----
        The name of the resulting Series will be set using the following
        rules:

        - For string, bytes, or integer `name_or_index` (or a list of these, for
          a nested selection), the Series name is set to the selected
          field's name.
        - For a :class:`pyarrow.compute.Expression`, this is set to
          the string form of the expression.
        - For list-like `name_or_index`, the name will be set to the
          name of the final field selected.

        Examples
        --------
        >>> import pyarrow as pa
        >>> s = pd.Series(
        ...     [
        ...         {"version": 1, "project": "pandas"},
        ...         {"version": 2, "project": "pandas"},
        ...         {"version": 1, "project": "numpy"},
        ...     ],
        ...     dtype=pd.ArrowDtype(pa.struct(
        ...         [("version", pa.int64()), ("project", pa.string())]
        ...     ))
        ... )

        Extract by field name.

        >>> s.struct.field("project")
        0    pandas
        1    pandas
        2     numpy
        Name: project, dtype: string[pyarrow]

        Extract by field index.

        >>> s.struct.field(0)
        0    1
        1    2
        2    1
        Name: version, dtype: int64[pyarrow]

        Or an expression

        >>> import pyarrow.compute as pc
        >>> s.struct.field(pc.field("project"))
        0    pandas
        1    pandas
        2     numpy
        Name: project, dtype: string[pyarrow]

        For nested struct types, you can pass a list of values to index
        multiple levels:

        >>> version_type = pa.struct([
        ...     ("major", pa.int64()),
        ...     ("minor", pa.int64()),
        ... ])
        >>> s = pd.Series(
        ...     [
        ...         {"version": {"major": 1, "minor": 5}, "project": "pandas"},
        ...         {"version": {"major": 2, "minor": 1}, "project": "pandas"},
        ...         {"version": {"major": 1, "minor": 26}, "project": "numpy"},
        ...     ],
        ...     dtype=pd.ArrowDtype(pa.struct(
        ...         [("version", version_type), ("project", pa.string())]
        ...     ))
        ... )
        >>> s.struct.field(["version", "minor"])
        0     5
        1     1
        2    26
        Name: minor, dtype: int64[pyarrow]
        >>> s.struct.field([0, 0])
        0    1
        1    2
        2    1
        Name: major, dtype: int64[pyarrow]
        """
        from pandas import Series

        def get_name(
            level_name_or_index: list[str]
            | list[bytes]
            | list[int]
            | pc.Expression
            | bytes
            | str
            | int,
            data: pa.ChunkedArray,
        ):
            if isinstance(level_name_or_index, int):
                name = data.type.field(level_name_or_index).name
            elif isinstance(level_name_or_index, (str, bytes)):
                name = level_name_or_index
            elif isinstance(level_name_or_index, pc.Expression):
                name = str(level_name_or_index)
            elif is_list_like(level_name_or_index):
                # For nested input like [2, 1, 2]
                # iteratively get the struct and field name. The last
                # one is used for the name of the index.
                level_name_or_index = list(reversed(level_name_or_index))
                selected = data
                while level_name_or_index:
                    # we need the cast, otherwise mypy complains about
                    # getting ints, bytes, or str here, which isn't possible.
                    level_name_or_index = cast(list, level_name_or_index)
                    name_or_index = level_name_or_index.pop()
                    name = get_name(name_or_index, selected)
                    selected = selected.type.field(selected.type.get_field_index(name))
                    name = selected.name
            else:
                raise ValueError(
                    "name_or_index must be an int, str, bytes, "
                    "pyarrow.compute.Expression, or list of those"
                )
            return name

        pa_arr = self._data.array._pa_array
        name = get_name(name_or_index, pa_arr)
        field_arr = pc.struct_field(pa_arr, name_or_index)

        return Series(
            field_arr,
            dtype=ArrowDtype(field_arr.type),
            index=self._data.index,
            name=name,
        )

    def explode(self) -> DataFrame:
        """
        Extract all child fields of a struct as a DataFrame.

        Returns
        -------
        pandas.DataFrame
            The data corresponding to all child fields.

        See Also
        --------
        Series.struct.field : Return a single child field as a Series.

        Examples
        --------
        >>> import pyarrow as pa
        >>> s = pd.Series(
        ...     [
        ...         {"version": 1, "project": "pandas"},
        ...         {"version": 2, "project": "pandas"},
        ...         {"version": 1, "project": "numpy"},
        ...     ],
        ...     dtype=pd.ArrowDtype(pa.struct(
        ...         [("version", pa.int64()), ("project", pa.string())]
        ...     ))
        ... )

        >>> s.struct.explode()
           version project
        0        1  pandas
        1        2  pandas
        2        1   numpy
        """
        from pandas import concat

        pa_type = self._pa_array.type
        return concat(
            [self.field(i) for i in range(pa_type.num_fields)], axis="columns"
        )
<|endoftext|>from __future__ import annotations

import functools
import operator
from pathlib import Path
import re
import textwrap
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    cast,
)
import unicodedata
import warnings

import numpy as np

from pandas._libs import lib
from pandas._libs.tslibs import (
    NaT,
    Timedelta,
    Timestamp,
    timezones,
)
from pandas.compat import (
    pa_version_under10p1,
    pa_version_under11p0,
    pa_version_under13p0,
)
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import validate_fillna_kwargs

from pandas.core.dtypes.cast import (
    can_hold_element,
    infer_dtype_from_scalar,
)
from pandas.core.dtypes.common import (
    is_array_like,
    is_bool_dtype,
    is_float_dtype,
    is_integer,
    is_list_like,
    is_numeric_dtype,
    is_scalar,
    is_string_dtype,
)
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.missing import isna

from pandas.core import (
    algorithms as algos,
    missing,
    ops,
    roperator,
)
from pandas.core.algorithms import map_array
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays._arrow_string_mixins import ArrowStringArrayMixin
from pandas.core.arrays._utils import to_numpy_dtype_inference
from pandas.core.arrays.base import (
    ExtensionArray,
    ExtensionArraySupportsAnyAll,
)
from pandas.core.arrays.masked import BaseMaskedArray
from pandas.core.arrays.string_ import StringDtype
import pandas.core.common as com
from pandas.core.indexers import (
    check_array_indexer,
    unpack_tuple_and_ellipses,
    validate_indices,
)
from pandas.core.nanops import check_below_min_count
from pandas.core.strings.base import BaseStringArrayMethods

from pandas.io._util import _arrow_dtype_mapping
from pandas.tseries.frequencies import to_offset

if not pa_version_under10p1:
    import pyarrow as pa
    import pyarrow.compute as pc

    from pandas.core.dtypes.dtypes import ArrowDtype

    ARROW_CMP_FUNCS = {
        "eq": pc.equal,
        "ne": pc.not_equal,
        "lt": pc.less,
        "gt": pc.greater,
        "le": pc.less_equal,
        "ge": pc.greater_equal,
    }

    ARROW_LOGICAL_FUNCS = {
        "and_": pc.and_kleene,
        "rand_": lambda x, y: pc.and_kleene(y, x),
        "or_": pc.or_kleene,
        "ror_": lambda x, y: pc.or_kleene(y, x),
        "xor": pc.xor,
        "rxor": lambda x, y: pc.xor(y, x),
    }

    ARROW_BIT_WISE_FUNCS = {
        "and_": pc.bit_wise_and,
        "rand_": lambda x, y: pc.bit_wise_and(y, x),
        "or_": pc.bit_wise_or,
        "ror_": lambda x, y: pc.bit_wise_or(y, x),
        "xor": pc.bit_wise_xor,
        "rxor": lambda x, y: pc.bit_wise_xor(y, x),
    }

    def cast_for_truediv(
        arrow_array: pa.ChunkedArray, pa_object: pa.Array | pa.Scalar
    ) -> tuple[pa.ChunkedArray, pa.Array | pa.Scalar]:
        # Ensure int / int -> float mirroring Python/Numpy behavior
        # as pc.divide_checked(int, int) -> int
        if pa.types.is_integer(arrow_array.type) and pa.types.is_integer(
            pa_object.type
        ):
            # GH: 56645.
            # https://github.com/apache/arrow/issues/35563
            return pc.cast(arrow_array, pa.float64(), safe=False), pc.cast(
                pa_object, pa.float64(), safe=False
            )

        return arrow_array, pa_object

    def floordiv_compat(
        left: pa.ChunkedArray | pa.Array | pa.Scalar,
        right: pa.ChunkedArray | pa.Array | pa.Scalar,
    ) -> pa.ChunkedArray:
        # TODO: Replace with pyarrow floordiv kernel.
        # https://github.com/apache/arrow/issues/39386
        if pa.types.is_integer(left.type) and pa.types.is_integer(right.type):
            divided = pc.divide_checked(left, right)
            if pa.types.is_signed_integer(divided.type):
                # GH 56676
                has_remainder = pc.not_equal(pc.multiply(divided, right), left)
                has_one_negative_operand = pc.less(
                    pc.bit_wise_xor(left, right),
                    pa.scalar(0, type=divided.type),
                )
                result = pc.if_else(
                    pc.and_(
                        has_remainder,
                        has_one_negative_operand,
                    ),
                    # GH: 55561
                    pc.subtract(divided, pa.scalar(1, type=divided.type)),
                    divided,
                )
            else:
                result = divided
            result = result.cast(left.type)
        else:
            divided = pc.divide(left, right)
            result = pc.floor(divided)
        return result

    ARROW_ARITHMETIC_FUNCS = {
        "add": pc.add_checked,
        "radd": lambda x, y: pc.add_checked(y, x),
        "sub": pc.subtract_checked,
        "rsub": lambda x, y: pc.subtract_checked(y, x),
        "mul": pc.multiply_checked,
        "rmul": lambda x, y: pc.multiply_checked(y, x),
        "truediv": lambda x, y: pc.divide(*cast_for_truediv(x, y)),
        "rtruediv": lambda x, y: pc.divide(*cast_for_truediv(y, x)),
        "floordiv": lambda x, y: floordiv_compat(x, y),
        "rfloordiv": lambda x, y: floordiv_compat(y, x),
        "mod": NotImplemented,
        "rmod": NotImplemented,
        "divmod": NotImplemented,
        "rdivmod": NotImplemented,
        "pow": pc.power_checked,
        "rpow": lambda x, y: pc.power_checked(y, x),
    }

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pandas._typing import (
        ArrayLike,
        AxisInt,
        Dtype,
        FillnaOptions,
        InterpolateOptions,
        Iterator,
        NpDtype,
        NumpySorter,
        NumpyValueArrayLike,
        PositionalIndexer,
        Scalar,
        Self,
        SortKind,
        TakeIndexer,
        TimeAmbiguous,
        TimeNonexistent,
        npt,
    )

    from pandas import Series
    from pandas.core.arrays.datetimes import DatetimeArray
    from pandas.core.arrays.timedeltas import TimedeltaArray


def get_unit_from_pa_dtype(pa_dtype):
    # https://github.com/pandas-dev/pandas/pull/50998#discussion_r1100344804
    if pa_version_under11p0:
        unit = str(pa_dtype).split("[", 1)[-1][:-1]
        if unit not in ["s", "ms", "us", "ns"]:
            raise ValueError(pa_dtype)
        return unit
    return pa_dtype.unit


def to_pyarrow_type(
    dtype: ArrowDtype | pa.DataType | Dtype | None,
) -> pa.DataType | None:
    """
    Convert dtype to a pyarrow type instance.
    """
    if isinstance(dtype, ArrowDtype):
        return dtype.pyarrow_dtype
    elif isinstance(dtype, pa.DataType):
        return dtype
    elif isinstance(dtype, DatetimeTZDtype):
        return pa.timestamp(dtype.unit, dtype.tz)
    elif dtype:
        try:
            # Accepts python types too
            # Doesn't handle all numpy types
            return pa.from_numpy_dtype(dtype)
        except pa.ArrowNotImplementedError:
            pass
    return None


class ArrowExtensionArray(  # type: ignore[misc]
    OpsMixin,
    ExtensionArraySupportsAnyAll,
    ArrowStringArrayMixin,
    BaseStringArrayMethods,
):
    """
    Pandas ExtensionArray backed by a PyArrow ChunkedArray.

    .. warning::

       ArrowExtensionArray is considered experimental. The implementation and
       parts of the API may change without warning.

    Parameters
    ----------
    values : pyarrow.Array or pyarrow.ChunkedArray

    Attributes
    ----------
    None

    Methods
    -------
    None

    Returns
    -------
    ArrowExtensionArray

    Notes
    -----
    Most methods are implemented using `pyarrow compute functions. <https://arrow.apache.org/docs/python/api/compute.html>`__
    Some methods may either raise an exception or raise a ``PerformanceWarning`` if an
    associated compute function is not available based on the installed version of PyArrow.

    Please install the latest version of PyArrow to enable the best functionality and avoid
    potential bugs in prior versions of PyArrow.

    Examples
    --------
    Create an ArrowExtensionArray with :func:`pandas.array`:

    >>> pd.array([1, 1, None], dtype="int64[pyarrow]")
    <ArrowExtensionArray>
    [1, 1, <NA>]
    Length: 3, dtype: int64[pyarrow]
    """  # noqa: E501 (http link too long)

    _pa_array: pa.ChunkedArray
    _dtype: ArrowDtype

    def __init__(self, values: pa.Array | pa.ChunkedArray) -> None:
        if pa_version_under10p1:
            msg = "pyarrow>=10.0.1 is required for PyArrow backed ArrowExtensionArray."
            raise ImportError(msg)
        if isinstance(values, pa.Array):
            self._pa_array = pa.chunked_array([values])
        elif isinstance(values, pa.ChunkedArray):
            self._pa_array = values
        else:
            raise ValueError(
                f"Unsupported type '{type(values)}' for ArrowExtensionArray"
            )
        self._dtype = ArrowDtype(self._pa_array.type)

    @classmethod
    def _from_sequence(cls, scalars, *, dtype: Dtype | None = None, copy: bool = False):
        """
        Construct a new ExtensionArray from a sequence of scalars.
        """
        pa_type = to_pyarrow_type(dtype)
        pa_array = cls._box_pa_array(scalars, pa_type=pa_type, copy=copy)
        arr = cls(pa_array)
        return arr

    @classmethod
    def _from_sequence_of_strings(
        cls, strings, *, dtype: Dtype | None = None, copy: bool = False
    ):
        """
        Construct a new ExtensionArray from a sequence of strings.
        """
        pa_type = to_pyarrow_type(dtype)
        if (
            pa_type is None
            or pa.types.is_binary(pa_type)
            or pa.types.is_string(pa_type)
            or pa.types.is_large_string(pa_type)
        ):
            # pa_type is None: Let pa.array infer
            # pa_type is string/binary: scalars already correct type
            scalars = strings
        elif pa.types.is_timestamp(pa_type):
            from pandas.core.tools.datetimes import to_datetime

            scalars = to_datetime(strings, errors="raise")
        elif pa.types.is_date(pa_type):
            from pandas.core.tools.datetimes import to_datetime

            scalars = to_datetime(strings, errors="raise").date
        elif pa.types.is_duration(pa_type):
            from pandas.core.tools.timedeltas import to_timedelta

            scalars = to_timedelta(strings, errors="raise")
            if pa_type.unit != "ns":
                # GH51175: test_from_sequence_of_strings_pa_array
                # attempt to parse as int64 reflecting pyarrow's
                # duration to string casting behavior
                mask = isna(scalars)
                if not isinstance(strings, (pa.Array, pa.ChunkedArray)):
                    strings = pa.array(strings, type=pa.string(), from_pandas=True)
                strings = pc.if_else(mask, None, strings)
                try:
                    scalars = strings.cast(pa.int64())
                except pa.ArrowInvalid:
                    pass
        elif pa.types.is_time(pa_type):
            from pandas.core.tools.times import to_time

            # "coerce" to allow "null times" (None) to not raise
            scalars = to_time(strings, errors="coerce")
        elif pa.types.is_boolean(pa_type):
            # pyarrow string->bool casting is case-insensitive:
            #   "true" or "1" -> True
            #   "false" or "0" -> False
            # Note: BooleanArray was previously used to parse these strings
            #   and allows "1.0" and "0.0". Pyarrow casting does not support
            #   this, but we allow it here.
            if isinstance(strings, (pa.Array, pa.ChunkedArray)):
                scalars = strings
            else:
                scalars = pa.array(strings, type=pa.string(), from_pandas=True)
            scalars = pc.if_else(pc.equal(scalars, "1.0"), "1", scalars)
            scalars = pc.if_else(pc.equal(scalars, "0.0"), "0", scalars)
            scalars = scalars.cast(pa.bool_())
        elif (
            pa.types.is_integer(pa_type)
            or pa.types.is_floating(pa_type)
            or pa.types.is_decimal(pa_type)
        ):
            from pandas.core.tools.numeric import to_numeric

            scalars = to_numeric(strings, errors="raise")
        else:
            raise NotImplementedError(
                f"Converting strings to {pa_type} is not implemented."
            )
        return cls._from_sequence(scalars, dtype=pa_type, copy=copy)

    @classmethod
    def _box_pa(
        cls, value, pa_type: pa.DataType | None = None
    ) -> pa.Array | pa.ChunkedArray | pa.Scalar:
        """
        Box value into a pyarrow Array, ChunkedArray or Scalar.

        Parameters
        ----------
        value : any
        pa_type : pa.DataType | None

        Returns
        -------
        pa.Array or pa.ChunkedArray or pa.Scalar
        """
        if isinstance(value, pa.Scalar) or not is_list_like(value):
            return cls._box_pa_scalar(value, pa_type)
        return cls._box_pa_array(value, pa_type)

    @classmethod
    def _box_pa_scalar(cls, value, pa_type: pa.DataType | None = None) -> pa.Scalar:
        """
        Box value into a pyarrow Scalar.

        Parameters
        ----------
        value : any
        pa_type : pa.DataType | None

        Returns
        -------
        pa.Scalar
        """
        if isinstance(value, pa.Scalar):
            pa_scalar = value
        elif isna(value):
            pa_scalar = pa.scalar(None, type=pa_type)
        else:
            # Workaround https://github.com/apache/arrow/issues/37291
            if isinstance(value, Timedelta):
                if pa_type is None:
                    pa_type = pa.duration(value.unit)
                elif value.unit != pa_type.unit:
                    value = value.as_unit(pa_type.unit)
                value = value._value
            elif isinstance(value, Timestamp):
                if pa_type is None:
                    pa_type = pa.timestamp(value.unit, tz=value.tz)
                elif value.unit != pa_type.unit:
                    value = value.as_unit(pa_type.unit)
                value = value._value

            pa_scalar = pa.scalar(value, type=pa_type, from_pandas=True)

        if pa_type is not None and pa_scalar.type != pa_type:
            pa_scalar = pa_scalar.cast(pa_type)

        return pa_scalar

    @classmethod
    def _box_pa_array(
        cls, value, pa_type: pa.DataType | None = None, copy: bool = False
    ) -> pa.Array | pa.ChunkedArray:
        """
        Box value into a pyarrow Array or ChunkedArray.

        Parameters
        ----------
        value : Sequence
        pa_type : pa.DataType | None

        Returns
        -------
        pa.Array or pa.ChunkedArray
        """
        if isinstance(value, cls):
            pa_array = value._pa_array
        elif isinstance(value, (pa.Array, pa.ChunkedArray)):
            pa_array = value
        elif isinstance(value, BaseMaskedArray):
            # GH 52625
            if copy:
                value = value.copy()
            pa_array = value.__arrow_array__()
        else:
            if (
                isinstance(value, np.ndarray)
                and pa_type is not None
                and (
                    pa.types.is_large_binary(pa_type)
                    or pa.types.is_large_string(pa_type)
                )
            ):
                # See https://github.com/apache/arrow/issues/35289
                value = value.tolist()
            elif copy and is_array_like(value):
                # pa array should not get updated when numpy array is updated
                value = value.copy()

            if (
                pa_type is not None
                and pa.types.is_duration(pa_type)
                and (not isinstance(value, np.ndarray) or value.dtype.kind not in "mi")
            ):
                # Workaround https://github.com/apache/arrow/issues/37291
                from pandas.core.tools.timedeltas import to_timedelta

                value = to_timedelta(value, unit=pa_type.unit).as_unit(pa_type.unit)
                value = value.to_numpy()

            try:
                pa_array = pa.array(value, type=pa_type, from_pandas=True)
            except (pa.ArrowInvalid, pa.ArrowTypeError):
                # GH50430: let pyarrow infer type, then cast
                pa_array = pa.array(value, from_pandas=True)

            if pa_type is None and pa.types.is_duration(pa_array.type):
                # Workaround https://github.com/apache/arrow/issues/37291
                from pandas.core.tools.timedeltas import to_timedelta

                value = to_timedelta(value)
                value = value.to_numpy()
                pa_array = pa.array(value, type=pa_type, from_pandas=True)

            if pa.types.is_duration(pa_array.type) and pa_array.null_count > 0:
                # GH52843: upstream bug for duration types when originally
                # constructed with data containing numpy NaT.
                # https://github.com/apache/arrow/issues/35088
                arr = cls(pa_array)
                arr = arr.fillna(arr.dtype.na_value)
                pa_array = arr._pa_array

        if pa_type is not None and pa_array.type != pa_type:
            if pa.types.is_dictionary(pa_type):
                pa_array = pa_array.dictionary_encode()
            else:
                try:
                    pa_array = pa_array.cast(pa_type)
                except (
                    pa.ArrowInvalid,
                    pa.ArrowTypeError,
