                    pa.ArrowNotImplementedError,
                ):
                    if pa.types.is_string(pa_array.type) or pa.types.is_large_string(
                        pa_array.type
                    ):
                        # TODO: Move logic in _from_sequence_of_strings into
                        # _box_pa_array
                        return cls._from_sequence_of_strings(
                            value, dtype=pa_type
                        )._pa_array
                    else:
                        raise

        return pa_array

    def __getitem__(self, item: PositionalIndexer):
        """Select a subset of self.

        Parameters
        ----------
        item : int, slice, or ndarray
            * int: The position in 'self' to get.
            * slice: A slice object, where 'start', 'stop', and 'step' are
              integers or None
            * ndarray: A 1-d boolean NumPy ndarray the same length as 'self'

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
        item = check_array_indexer(self, item)

        if isinstance(item, np.ndarray):
            if not len(item):
                # Removable once we migrate StringDtype[pyarrow] to ArrowDtype[string]
                if (
                    isinstance(self._dtype, StringDtype)
                    and self._dtype.storage == "pyarrow"
                ):
                    # TODO(infer_string) should this be large_string?
                    pa_dtype = pa.string()
                else:
                    pa_dtype = self._dtype.pyarrow_dtype
                return type(self)(pa.chunked_array([], type=pa_dtype))
            elif item.dtype.kind in "iu":
                return self.take(item)
            elif item.dtype.kind == "b":
                return type(self)(self._pa_array.filter(item))
            else:
                raise IndexError(
                    "Only integers, slices and integer or "
                    "boolean arrays are valid indices."
                )
        elif isinstance(item, tuple):
            item = unpack_tuple_and_ellipses(item)

        if item is Ellipsis:
            # TODO: should be handled by pyarrow?
            item = slice(None)

        if is_scalar(item) and not is_integer(item):
            # e.g. "foo" or 2.5
            # exception message copied from numpy
            raise IndexError(
                r"only integers, slices (`:`), ellipsis (`...`), numpy.newaxis "
                r"(`None`) and integer or boolean arrays are valid indices"
            )
        # We are not an array indexer, so maybe e.g. a slice or integer
        # indexer. We dispatch to pyarrow.
        if isinstance(item, slice):
            # Arrow bug https://github.com/apache/arrow/issues/38768
            if item.start == item.stop:
                pass
            elif (
                item.stop is not None
                and item.stop < -len(self)
                and item.step is not None
                and item.step < 0
            ):
                item = slice(item.start, None, item.step)

        value = self._pa_array[item]
        if isinstance(value, pa.ChunkedArray):
            return type(self)(value)
        else:
            pa_type = self._pa_array.type
            scalar = value.as_py()
            if scalar is None:
                return self._dtype.na_value
            elif pa.types.is_timestamp(pa_type) and pa_type.unit != "ns":
                # GH 53326
                return Timestamp(scalar).as_unit(pa_type.unit)
            elif pa.types.is_duration(pa_type) and pa_type.unit != "ns":
                # GH 53326
                return Timedelta(scalar).as_unit(pa_type.unit)
            else:
                return scalar

    def __iter__(self) -> Iterator[Any]:
        """
        Iterate over elements of the array.
        """
        na_value = self._dtype.na_value
        # GH 53326
        pa_type = self._pa_array.type
        box_timestamp = pa.types.is_timestamp(pa_type) and pa_type.unit != "ns"
        box_timedelta = pa.types.is_duration(pa_type) and pa_type.unit != "ns"
        for value in self._pa_array:
            val = value.as_py()
            if val is None:
                yield na_value
            elif box_timestamp:
                yield Timestamp(val).as_unit(pa_type.unit)
            elif box_timedelta:
                yield Timedelta(val).as_unit(pa_type.unit)
            else:
                yield val

    def __arrow_array__(self, type=None):
        """Convert myself to a pyarrow ChunkedArray."""
        return self._pa_array

    def __array__(
        self, dtype: NpDtype | None = None, copy: bool | None = None
    ) -> np.ndarray:
        """Correctly construct numpy arrays when passed to `np.asarray()`."""
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
        elif copy is None:
            # `to_numpy(copy=False)` has the meaning of NumPy `copy=None`.
            copy = False

        return self.to_numpy(dtype=dtype, copy=copy)

    def __invert__(self) -> Self:
        # This is a bit wise op for integer types
        if pa.types.is_integer(self._pa_array.type):
            return type(self)(pc.bit_wise_not(self._pa_array))
        elif pa.types.is_string(self._pa_array.type) or pa.types.is_large_string(
            self._pa_array.type
        ):
            # Raise TypeError instead of pa.ArrowNotImplementedError
            raise TypeError("__invert__ is not supported for string dtypes")
        else:
            return type(self)(pc.invert(self._pa_array))

    def __neg__(self) -> Self:
        try:
            return type(self)(pc.negate_checked(self._pa_array))
        except pa.ArrowNotImplementedError as err:
            raise TypeError(
                f"unary '-' not supported for dtype '{self.dtype}'"
            ) from err

    def __pos__(self) -> Self:
        return type(self)(self._pa_array)

    def __abs__(self) -> Self:
        return type(self)(pc.abs_checked(self._pa_array))

    # GH 42600: __getstate__/__setstate__ not necessary once
    # https://issues.apache.org/jira/browse/ARROW-10739 is addressed
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_pa_array"] = self._pa_array.combine_chunks()
        return state

    def __setstate__(self, state) -> None:
        if "_data" in state:
            data = state.pop("_data")
        else:
            data = state["_pa_array"]
        state["_pa_array"] = pa.chunked_array(data)
        self.__dict__.update(state)

    def _cmp_method(self, other, op):
        pc_func = ARROW_CMP_FUNCS[op.__name__]
        if isinstance(other, (ExtensionArray, np.ndarray, list)):
            try:
                boxed = self._box_pa(other)
            except pa.lib.ArrowInvalid:
                # e.g. GH#60228 [1, "b"] we have to operate pointwise
                res_values = [op(x, y) for x, y in zip(self, other)]
                result = pa.array(res_values, type=pa.bool_(), from_pandas=True)
            else:
                try:
                    result = pc_func(self._pa_array, boxed)
                except pa.ArrowNotImplementedError:
                    result = ops.invalid_comparison(self, other, op)
                    result = pa.array(result, type=pa.bool_())

        elif is_scalar(other):
            try:
                result = pc_func(self._pa_array, self._box_pa(other))
            except (pa.lib.ArrowNotImplementedError, pa.lib.ArrowInvalid):
                mask = isna(self) | isna(other)
                valid = ~mask
                result = np.zeros(len(self), dtype="bool")
                np_array = np.array(self)
                try:
                    result[valid] = op(np_array[valid], other)
                except TypeError:
                    result = ops.invalid_comparison(self, other, op)
                result = pa.array(result, type=pa.bool_())
                result = pc.if_else(valid, result, None)
        else:
            raise NotImplementedError(
                f"{op.__name__} not implemented for {type(other)}"
            )
        return ArrowExtensionArray(result)

    def _op_method_error_message(self, other, op) -> str:
        if hasattr(other, "dtype"):
            other_type = f"dtype '{other.dtype}'"
        else:
            other_type = f"object of type {type(other)}"
        return (
            f"operation '{op.__name__}' not supported for "
            f"dtype '{self.dtype}' with {other_type}"
        )

    def _evaluate_op_method(self, other, op, arrow_funcs) -> Self:
        pa_type = self._pa_array.type
        other_original = other
        other = self._box_pa(other)

        if (
            pa.types.is_string(pa_type)
            or pa.types.is_large_string(pa_type)
            or pa.types.is_binary(pa_type)
        ):
            if op in [operator.add, roperator.radd]:
                sep = pa.scalar("", type=pa_type)
                try:
                    if op is operator.add:
                        result = pc.binary_join_element_wise(self._pa_array, other, sep)
                    elif op is roperator.radd:
                        result = pc.binary_join_element_wise(other, self._pa_array, sep)
                except pa.ArrowNotImplementedError as err:
                    raise TypeError(
                        self._op_method_error_message(other_original, op)
                    ) from err
                return type(self)(result)
            elif op in [operator.mul, roperator.rmul]:
                binary = self._pa_array
                integral = other
                if not pa.types.is_integer(integral.type):
                    raise TypeError("Can only string multiply by an integer.")
                pa_integral = pc.if_else(pc.less(integral, 0), 0, integral)
                result = pc.binary_repeat(binary, pa_integral)
                return type(self)(result)
        elif (
            pa.types.is_string(other.type)
            or pa.types.is_binary(other.type)
            or pa.types.is_large_string(other.type)
        ) and op in [operator.mul, roperator.rmul]:
            binary = other
            integral = self._pa_array
            if not pa.types.is_integer(integral.type):
                raise TypeError("Can only string multiply by an integer.")
            pa_integral = pc.if_else(pc.less(integral, 0), 0, integral)
            result = pc.binary_repeat(binary, pa_integral)
            return type(self)(result)
        if (
            isinstance(other, pa.Scalar)
            and pc.is_null(other).as_py()
            and op.__name__ in ARROW_LOGICAL_FUNCS
        ):
            # pyarrow kleene ops require null to be typed
            other = other.cast(pa_type)

        pc_func = arrow_funcs[op.__name__]
        if pc_func is NotImplemented:
            if pa.types.is_string(pa_type) or pa.types.is_large_string(pa_type):
                raise TypeError(self._op_method_error_message(other_original, op))
            raise NotImplementedError(f"{op.__name__} not implemented.")

        try:
            result = pc_func(self._pa_array, other)
        except pa.ArrowNotImplementedError as err:
            raise TypeError(self._op_method_error_message(other_original, op)) from err
        return type(self)(result)

    def _logical_method(self, other, op):
        # For integer types `^`, `|`, `&` are bitwise operators and return
        # integer types. Otherwise these are boolean ops.
        if pa.types.is_integer(self._pa_array.type):
            return self._evaluate_op_method(other, op, ARROW_BIT_WISE_FUNCS)
        elif (
            (
                pa.types.is_string(self._pa_array.type)
                or pa.types.is_large_string(self._pa_array.type)
            )
            and op in (roperator.ror_, roperator.rand_, roperator.rxor)
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
        else:
            return self._evaluate_op_method(other, op, ARROW_LOGICAL_FUNCS)

    def _arith_method(self, other, op) -> Self | npt.NDArray[np.object_]:
        if (
            op in [operator.truediv, roperator.rtruediv]
            and isinstance(other, Path)
            and (
                pa.types.is_string(self._pa_array.type)
                or pa.types.is_large_string(self._pa_array.type)
            )
        ):
            # GH#61940
            return np.array(
                [
                    op(x, other) if isinstance(x, str) else self.dtype.na_value
                    for x in self
                ],
                dtype=object,
            )
        return self._evaluate_op_method(other, op, ARROW_ARITHMETIC_FUNCS)

    def equals(self, other) -> bool:
        if not isinstance(other, ArrowExtensionArray):
            return False
        # I'm told that pyarrow makes __eq__ behave like pandas' equals;
        #  TODO: is this documented somewhere?
        return self._pa_array == other._pa_array

    @property
    def dtype(self) -> ArrowDtype:
        """
        An instance of 'ExtensionDtype'.
        """
        return self._dtype

    @property
    def nbytes(self) -> int:
        """
        The number of bytes needed to store this object in memory.
        """
        return self._pa_array.nbytes

    def __len__(self) -> int:
        """
        Length of this array.

        Returns
        -------
        length : int
        """
        return len(self._pa_array)

    def __contains__(self, key) -> bool:
        # https://github.com/pandas-dev/pandas/pull/51307#issuecomment-1426372604
        if isna(key) and key is not self.dtype.na_value:
            if self.dtype.kind == "f" and lib.is_float(key):
                return pc.any(pc.is_nan(self._pa_array)).as_py()

            # e.g. date or timestamp types we do not allow None here to match pd.NA
            return False
            # TODO: maybe complex? object?

        return bool(super().__contains__(key))

    @property
    def _hasna(self) -> bool:
        return self._pa_array.null_count > 0

    def isna(self) -> npt.NDArray[np.bool_]:
        """
        Boolean NumPy array indicating if each value is missing.

        This should return a 1-D array the same length as 'self'.
        """
        # GH51630: fast paths
        null_count = self._pa_array.null_count
        if null_count == 0:
            return np.zeros(len(self), dtype=np.bool_)
        elif null_count == len(self):
            return np.ones(len(self), dtype=np.bool_)

        return self._pa_array.is_null().to_numpy()

    def any(self, *, skipna: bool = True, **kwargs):
        """
        Return whether any element is truthy.

        Returns False unless there is at least one element that is truthy.
        By default, NAs are skipped. If ``skipna=False`` is specified and
        missing values are present, similar :ref:`Kleene logic <boolean.kleene>`
        is used as for logical operations.

        Parameters
        ----------
        skipna : bool, default True
            Exclude NA values. If the entire array is NA and `skipna` is
            True, then the result will be False, as for an empty array.
            If `skipna` is False, the result will still be True if there is
            at least one element that is truthy, otherwise NA will be returned
            if there are NA's present.

        Returns
        -------
        bool or :attr:`pandas.NA`

        See Also
        --------
        ArrowExtensionArray.all : Return whether all elements are truthy.

        Examples
        --------
        The result indicates whether any element is truthy (and by default
        skips NAs):

        >>> pd.array([True, False, True], dtype="boolean[pyarrow]").any()
        True
        >>> pd.array([True, False, pd.NA], dtype="boolean[pyarrow]").any()
        True
        >>> pd.array([False, False, pd.NA], dtype="boolean[pyarrow]").any()
        False
        >>> pd.array([], dtype="boolean[pyarrow]").any()
        False
        >>> pd.array([pd.NA], dtype="boolean[pyarrow]").any()
        False
        >>> pd.array([pd.NA], dtype="float64[pyarrow]").any()
        False

        With ``skipna=False``, the result can be NA if this is logically
        required (whether ``pd.NA`` is True or False influences the result):

        >>> pd.array([True, False, pd.NA], dtype="boolean[pyarrow]").any(skipna=False)
        True
        >>> pd.array([1, 0, pd.NA], dtype="boolean[pyarrow]").any(skipna=False)
        True
        >>> pd.array([False, False, pd.NA], dtype="boolean[pyarrow]").any(skipna=False)
        <NA>
        >>> pd.array([0, 0, pd.NA], dtype="boolean[pyarrow]").any(skipna=False)
        <NA>
        """
        return self._reduce("any", skipna=skipna, **kwargs)

    def all(self, *, skipna: bool = True, **kwargs):
        """
        Return whether all elements are truthy.

        Returns True unless there is at least one element that is falsey.
        By default, NAs are skipped. If ``skipna=False`` is specified and
        missing values are present, similar :ref:`Kleene logic <boolean.kleene>`
        is used as for logical operations.

        Parameters
        ----------
        skipna : bool, default True
            Exclude NA values. If the entire array is NA and `skipna` is
            True, then the result will be True, as for an empty array.
            If `skipna` is False, the result will still be False if there is
            at least one element that is falsey, otherwise NA will be returned
            if there are NA's present.

        Returns
        -------
        bool or :attr:`pandas.NA`

        See Also
        --------
        ArrowExtensionArray.any : Return whether any element is truthy.

        Examples
        --------
        The result indicates whether all elements are truthy (and by default
        skips NAs):

        >>> pd.array([True, True, pd.NA], dtype="boolean[pyarrow]").all()
        True
        >>> pd.array([1, 1, pd.NA], dtype="boolean[pyarrow]").all()
        True
        >>> pd.array([True, False, pd.NA], dtype="boolean[pyarrow]").all()
        False
        >>> pd.array([], dtype="boolean[pyarrow]").all()
        True
        >>> pd.array([pd.NA], dtype="boolean[pyarrow]").all()
        True
        >>> pd.array([pd.NA], dtype="float64[pyarrow]").all()
        True

        With ``skipna=False``, the result can be NA if this is logically
        required (whether ``pd.NA`` is True or False influences the result):

        >>> pd.array([True, True, pd.NA], dtype="boolean[pyarrow]").all(skipna=False)
        <NA>
        >>> pd.array([1, 1, pd.NA], dtype="boolean[pyarrow]").all(skipna=False)
        <NA>
        >>> pd.array([True, False, pd.NA], dtype="boolean[pyarrow]").all(skipna=False)
        False
        >>> pd.array([1, 0, pd.NA], dtype="boolean[pyarrow]").all(skipna=False)
        False
        """
        return self._reduce("all", skipna=skipna, **kwargs)

    def argsort(
        self,
        *,
        ascending: bool = True,
        kind: SortKind = "quicksort",
        na_position: str = "last",
        **kwargs,
    ) -> np.ndarray:
        order = "ascending" if ascending else "descending"
        null_placement = {"last": "at_end", "first": "at_start"}.get(na_position, None)
        if null_placement is None:
            raise ValueError(f"invalid na_position: {na_position}")

        result = pc.array_sort_indices(
            self._pa_array, order=order, null_placement=null_placement
        )
        np_result = result.to_numpy()
        return np_result.astype(np.intp, copy=False)

    def _argmin_max(self, skipna: bool, method: str) -> int:
        if self._pa_array.length() in (0, self._pa_array.null_count) or (
            self._hasna and not skipna
        ):
            # For empty or all null, pyarrow returns -1 but pandas expects TypeError
            # For skipna=False and data w/ null, pandas expects NotImplementedError
            # let ExtensionArray.arg{max|min} raise
            return getattr(super(), f"arg{method}")(skipna=skipna)

        data = self._pa_array
        if pa.types.is_duration(data.type):
            data = data.cast(pa.int64())

        value = getattr(pc, method)(data, skip_nulls=skipna)
        return pc.index(data, value).as_py()

    def argmin(self, skipna: bool = True) -> int:
        return self._argmin_max(skipna, "min")

    def argmax(self, skipna: bool = True) -> int:
        return self._argmin_max(skipna, "max")

    def copy(self) -> Self:
        """
        Return a shallow copy of the array.

        Underlying ChunkedArray is immutable, so a deep copy is unnecessary.

        Returns
        -------
        type(self)
        """
        return type(self)(self._pa_array)

    def dropna(self) -> Self:
        """
        Return ArrowExtensionArray without NA values.

        Returns
        -------
        ArrowExtensionArray
        """
        return type(self)(pc.drop_null(self._pa_array))

    def _pad_or_backfill(
        self,
        *,
        method: FillnaOptions,
        limit: int | None = None,
        limit_area: Literal["inside", "outside"] | None = None,
        copy: bool = True,
    ) -> Self:
        if not self._hasna:
            # TODO(CoW): Not necessary anymore when CoW is the default
            return self.copy()

        if limit is None and limit_area is None:
            method = missing.clean_fill_method(method)
            try:
                if method == "pad":
                    return type(self)(pc.fill_null_forward(self._pa_array))
                elif method == "backfill":
                    return type(self)(pc.fill_null_backward(self._pa_array))
            except pa.ArrowNotImplementedError:
                # ArrowNotImplementedError: Function 'coalesce' has no kernel
                #   matching input types (duration[ns], duration[ns])
                # TODO: remove try/except wrapper if/when pyarrow implements
                #   a kernel for duration types.
                pass

        # TODO(3.0): after EA.fillna 'method' deprecation is enforced, we can remove
        #  this method entirely.
        return super()._pad_or_backfill(
            method=method, limit=limit, limit_area=limit_area, copy=copy
        )

    @doc(ExtensionArray.fillna)
    def fillna(
        self,
        value: object | ArrayLike | None = None,
        method: FillnaOptions | None = None,
        limit: int | None = None,
        copy: bool = True,
    ) -> Self:
        value, method = validate_fillna_kwargs(value, method)

        if not self._hasna:
            # TODO(CoW): Not necessary anymore when CoW is the default
            return self.copy()

        if limit is not None:
            return super().fillna(value=value, method=method, limit=limit, copy=copy)

        if method is not None:
            return super().fillna(method=method, limit=limit, copy=copy)

        if isinstance(value, (np.ndarray, ExtensionArray)):
            # Similar to check_value_size, but we do not mask here since we may
            #  end up passing it to the super() method.
            if len(value) != len(self):
                raise ValueError(
                    f"Length of 'value' does not match. Got ({len(value)}) "
                    f" expected {len(self)}"
                )

        try:
            fill_value = self._box_pa(value, pa_type=self._pa_array.type)
        except pa.ArrowTypeError as err:
            msg = f"Invalid value '{value!s}' for dtype '{self.dtype}'"
            raise TypeError(msg) from err

        try:
            return type(self)(pc.fill_null(self._pa_array, fill_value=fill_value))
        except pa.ArrowNotImplementedError:
            # ArrowNotImplementedError: Function 'coalesce' has no kernel
            #   matching input types (duration[ns], duration[ns])
            # TODO: remove try/except wrapper if/when pyarrow implements
            #   a kernel for duration types.
            pass

        return super().fillna(value=value, method=method, limit=limit, copy=copy)

    def isin(self, values: ArrayLike) -> npt.NDArray[np.bool_]:
        # short-circuit to return all False array.
        if not len(values):
            return np.zeros(len(self), dtype=bool)

        result = pc.is_in(self._pa_array, value_set=pa.array(values, from_pandas=True))
        # pyarrow 2.0.0 returned nulls, so we explicitly specify dtype to convert nulls
        # to False
        return np.array(result, dtype=np.bool_)

    def _values_for_factorize(self) -> tuple[np.ndarray, Any]:
        """
        Return an array and missing value suitable for factorization.

        Returns
        -------
        values : ndarray
        na_value : pd.NA

        Notes
        -----
        The values returned by this method are also used in
        :func:`pandas.util.hash_pandas_object`.
        """
        values = self._pa_array.to_numpy()
        return values, self.dtype.na_value

    @doc(ExtensionArray.factorize)
    def factorize(
        self,
        use_na_sentinel: bool = True,
    ) -> tuple[np.ndarray, ExtensionArray]:
        null_encoding = "mask" if use_na_sentinel else "encode"

        data = self._pa_array
        pa_type = data.type
        if pa_version_under11p0 and pa.types.is_duration(pa_type):
            # https://github.com/apache/arrow/issues/15226#issuecomment-1376578323
            data = data.cast(pa.int64())

        if pa.types.is_dictionary(data.type):
            encoded = data
        else:
            encoded = data.dictionary_encode(null_encoding=null_encoding)
        if encoded.length() == 0:
            indices = np.array([], dtype=np.intp)
            uniques = type(self)(pa.chunked_array([], type=encoded.type.value_type))
        else:
            # GH 54844
            combined = encoded.combine_chunks()
            pa_indices = combined.indices
            if pa_indices.null_count > 0:
                pa_indices = pc.fill_null(pa_indices, -1)
            indices = pa_indices.to_numpy(zero_copy_only=False, writable=True).astype(
                np.intp, copy=False
            )
            uniques = type(self)(combined.dictionary)

        if pa_version_under11p0 and pa.types.is_duration(pa_type):
            uniques = cast(ArrowExtensionArray, uniques.astype(self.dtype))
        return indices, uniques

    def reshape(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self)} does not support reshape "
            f"as backed by a 1D pyarrow.ChunkedArray."
        )

    def round(self, decimals: int = 0, *args, **kwargs) -> Self:
        """
        Round each value in the array a to the given number of decimals.

        Parameters
        ----------
        decimals : int, default 0
            Number of decimal places to round to. If decimals is negative,
            it specifies the number of positions to the left of the decimal point.
        *args, **kwargs
            Additional arguments and keywords have no effect.

        Returns
        -------
        ArrowExtensionArray
            Rounded values of the ArrowExtensionArray.

        See Also
        --------
        DataFrame.round : Round values of a DataFrame.
        Series.round : Round values of a Series.
        """
        return type(self)(pc.round(self._pa_array, ndigits=decimals))

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
        if isinstance(value, ExtensionArray):
            value = value.astype(object)
        # Base class searchsorted would cast to object, which is *much* slower.
        dtype = None
        if isinstance(self.dtype, ArrowDtype):
            pa_dtype = self.dtype.pyarrow_dtype
            if (
                pa.types.is_timestamp(pa_dtype) or pa.types.is_duration(pa_dtype)
            ) and pa_dtype.unit == "ns":
                # np.array[datetime/timedelta].searchsorted(datetime/timedelta)
                # erroneously fails when numpy type resolution is nanoseconds
                dtype = object
        return self.to_numpy(dtype=dtype).searchsorted(value, side=side, sorter=sorter)

    def take(
        self,
        indices: TakeIndexer,
        allow_fill: bool = False,
        fill_value: Any = None,
    ) -> ArrowExtensionArray:
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
        numpy.take
        api.extensions.take

        Notes
        -----
        ExtensionArray.take is called by ``Series.__getitem__``, ``.loc``,
        ``iloc``, when `indices` is a sequence of values. Additionally,
        it's called by :meth:`Series.reindex`, or any other method
        that causes realignment, with a `fill_value`.
        """
        indices_array = np.asanyarray(indices)

        if len(self._pa_array) == 0 and (indices_array >= 0).any():
            raise IndexError("cannot do a non-empty take")
        if indices_array.size > 0 and indices_array.max() >= len(self._pa_array):
            raise IndexError("out of bounds value in 'indices'.")

        if allow_fill:
            fill_mask = indices_array < 0
            if fill_mask.any():
                validate_indices(indices_array, len(self._pa_array))
                # TODO(ARROW-9433): Treat negative indices as NULL
                indices_array = pa.array(indices_array, mask=fill_mask)
                result = self._pa_array.take(indices_array)
                if isna(fill_value):
                    return type(self)(result)
                # TODO: ArrowNotImplementedError: Function fill_null has no
                # kernel matching input types (array[string], scalar[string])
                result = type(self)(result)
                result[fill_mask] = fill_value
                return result
                # return type(self)(pc.fill_null(result, pa.scalar(fill_value)))
            else:
                # Nothing to fill
                return type(self)(self._pa_array.take(indices))
        else:  # allow_fill=False
            # TODO(ARROW-9432): Treat negative indices as indices from the right.
            if (indices_array < 0).any():
                # Don't modify in-place
                indices_array = np.copy(indices_array)
                indices_array[indices_array < 0] += len(self._pa_array)
            return type(self)(self._pa_array.take(indices_array))

    def _maybe_convert_datelike_array(self):
        """Maybe convert to a datelike array."""
        pa_type = self._pa_array.type
        if pa.types.is_timestamp(pa_type):
            return self._to_datetimearray()
        elif pa.types.is_duration(pa_type):
            return self._to_timedeltaarray()
        return self

    def _to_datetimearray(self) -> DatetimeArray:
        """Convert a pyarrow timestamp typed array to a DatetimeArray."""
        from pandas.core.arrays.datetimes import (
            DatetimeArray,
            tz_to_dtype,
        )

        pa_type = self._pa_array.type
        assert pa.types.is_timestamp(pa_type)
        np_dtype = np.dtype(f"M8[{pa_type.unit}]")
        dtype = tz_to_dtype(pa_type.tz, pa_type.unit)
        np_array = self._pa_array.to_numpy()
        np_array = np_array.astype(np_dtype)
        return DatetimeArray._simple_new(np_array, dtype=dtype)

    def _to_timedeltaarray(self) -> TimedeltaArray:
        """Convert a pyarrow duration typed array to a TimedeltaArray."""
        from pandas.core.arrays.timedeltas import TimedeltaArray

        pa_type = self._pa_array.type
        assert pa.types.is_duration(pa_type)
        np_dtype = np.dtype(f"m8[{pa_type.unit}]")
        np_array = self._pa_array.to_numpy()
        np_array = np_array.astype(np_dtype)
        return TimedeltaArray._simple_new(np_array, dtype=np_dtype)

    def _values_for_json(self) -> np.ndarray:
        if is_numeric_dtype(self.dtype):
            return np.asarray(self, dtype=object)
        return super()._values_for_json()

    @doc(ExtensionArray.to_numpy)
    def to_numpy(
        self,
        dtype: npt.DTypeLike | None = None,
        copy: bool = False,
        na_value: object = lib.no_default,
    ) -> np.ndarray:
        original_na_value = na_value
        dtype, na_value = to_numpy_dtype_inference(self, dtype, na_value, self._hasna)
        pa_type = self._pa_array.type
        if not self._hasna or isna(na_value) or pa.types.is_null(pa_type):
            data = self
        else:
            data = self.fillna(na_value)
            copy = False

        if pa.types.is_timestamp(pa_type) or pa.types.is_duration(pa_type):
            # GH 55997
            if dtype != object and na_value is self.dtype.na_value:
                na_value = lib.no_default
            result = data._maybe_convert_datelike_array().to_numpy(
                dtype=dtype, na_value=na_value
            )
        elif pa.types.is_time(pa_type) or pa.types.is_date(pa_type):
            # convert to list of python datetime.time objects before
            # wrapping in ndarray
            result = np.array(list(data), dtype=dtype)
            if data._hasna:
                result[data.isna()] = na_value
        elif pa.types.is_null(pa_type):
            if dtype is not None and isna(na_value):
                na_value = None
            result = np.full(len(data), fill_value=na_value, dtype=dtype)
        elif not data._hasna or (
            pa.types.is_floating(pa_type)
            and (
                na_value is np.nan
                or original_na_value is lib.no_default
                and is_float_dtype(dtype)
            )
        ):
            result = data._pa_array.to_numpy()
            if dtype is not None:
                result = result.astype(dtype, copy=False)
            if copy:
                result = result.copy()
        else:
            if dtype is None:
                empty = pa.array([], type=pa_type).to_numpy(zero_copy_only=False)
                if can_hold_element(empty, na_value):
                    dtype = empty.dtype
                else:
                    dtype = np.object_
            result = np.empty(len(data), dtype=dtype)
            mask = data.isna()
            result[mask] = na_value
            result[~mask] = data[~mask]._pa_array.to_numpy()
        return result

    def map(self, mapper, na_action=None):
        if is_numeric_dtype(self.dtype):
            return map_array(self.to_numpy(), mapper, na_action=na_action)
        else:
            return super().map(mapper, na_action)

    @doc(ExtensionArray.duplicated)
    def duplicated(
        self, keep: Literal["first", "last", False] = "first"
    ) -> npt.NDArray[np.bool_]:
        pa_type = self._pa_array.type
        if pa.types.is_floating(pa_type) or pa.types.is_integer(pa_type):
            values = self.to_numpy(na_value=0)
        elif pa.types.is_boolean(pa_type):
            values = self.to_numpy(na_value=False)
        elif pa.types.is_temporal(pa_type):
            if pa_type.bit_width == 32:
                pa_type = pa.int32()
            else:
                pa_type = pa.int64()
            arr = self.astype(ArrowDtype(pa_type))
            values = arr.to_numpy(na_value=0)
        else:
            # factorize the values to avoid the performance penalty of
            # converting to object dtype
            values = self.factorize()[0]

        mask = self.isna() if self._hasna else None
        return algos.duplicated(values, keep=keep, mask=mask)

    def unique(self) -> Self:
        """
        Compute the ArrowExtensionArray of unique values.

        Returns
        -------
        ArrowExtensionArray
        """
        pa_type = self._pa_array.type

        if pa_version_under11p0 and pa.types.is_duration(pa_type):
            # https://github.com/apache/arrow/issues/15226#issuecomment-1376578323
            data = self._pa_array.cast(pa.int64())
        else:
            data = self._pa_array

        pa_result = pc.unique(data)

        if pa_version_under11p0 and pa.types.is_duration(pa_type):
            pa_result = pa_result.cast(pa_type)

        return type(self)(pa_result)

    def value_counts(self, dropna: bool = True) -> Series:
        """
        Return a Series containing counts of each unique value.

        Parameters
        ----------
        dropna : bool, default True
            Don't include counts of missing values.

        Returns
        -------
        counts : Series

        See Also
        --------
        Series.value_counts
        """
        pa_type = self._pa_array.type
        if pa_version_under11p0 and pa.types.is_duration(pa_type):
            # https://github.com/apache/arrow/issues/15226#issuecomment-1376578323
            data = self._pa_array.cast(pa.int64())
        else:
            data = self._pa_array

        from pandas import (
            Index,
            Series,
        )

        vc = data.value_counts()

        values = vc.field(0)
        counts = vc.field(1)
        if dropna and data.null_count > 0:
            mask = values.is_valid()
            values = values.filter(mask)
            counts = counts.filter(mask)

        if pa_version_under11p0 and pa.types.is_duration(pa_type):
            values = values.cast(pa_type)

        counts = ArrowExtensionArray(counts)

        index = Index(type(self)(values))

        return Series(counts, index=index, name="count", copy=False)

    @classmethod
    def _concat_same_type(cls, to_concat) -> Self:
        """
        Concatenate multiple ArrowExtensionArrays.

        Parameters
        ----------
        to_concat : sequence of ArrowExtensionArrays

        Returns
        -------
        ArrowExtensionArray
        """
        chunks = [array for ea in to_concat for array in ea._pa_array.iterchunks()]
        if to_concat[0].dtype == "string":
            # StringDtype has no attribute pyarrow_dtype
            pa_dtype = pa.large_string()
        else:
            pa_dtype = to_concat[0].dtype.pyarrow_dtype
        arr = pa.chunked_array(chunks, type=pa_dtype)
        return cls(arr)

    def _accumulate(
        self, name: str, *, skipna: bool = True, **kwargs
    ) -> ArrowExtensionArray | ExtensionArray:
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
        if is_string_dtype(self):
            return self._str_accumulate(name=name, skipna=skipna, **kwargs)

        pyarrow_name = {
            "cummax": "cumulative_max",
            "cummin": "cumulative_min",
            "cumprod": "cumulative_prod_checked",
            "cumsum": "cumulative_sum_checked",
        }.get(name, name)
        pyarrow_meth = getattr(pc, pyarrow_name, None)
        if pyarrow_meth is None:
            return super()._accumulate(name, skipna=skipna, **kwargs)

        data_to_accum = self._pa_array

        pa_dtype = data_to_accum.type

        convert_to_int = (
            pa.types.is_temporal(pa_dtype) and name in ["cummax", "cummin"]
        ) or (pa.types.is_duration(pa_dtype) and name == "cumsum")

        if convert_to_int:
            if pa_dtype.bit_width == 32:
                data_to_accum = data_to_accum.cast(pa.int32())
            else:
                data_to_accum = data_to_accum.cast(pa.int64())

        try:
            result = pyarrow_meth(data_to_accum, skip_nulls=skipna, **kwargs)
        except pa.ArrowNotImplementedError as err:
            msg = f"operation '{name}' not supported for dtype '{self.dtype}'"
            raise TypeError(msg) from err

        if convert_to_int:
            result = result.cast(pa_dtype)

        return type(self)(result)

    def _str_accumulate(
        self, name: str, *, skipna: bool = True, **kwargs
    ) -> ArrowExtensionArray | ExtensionArray:
        """
        Accumulate implementation for strings, see `_accumulate` docstring for details.

        pyarrow.compute does not implement these methods for strings.
        """
        if name == "cumprod":
            msg = f"operation '{name}' not supported for dtype '{self.dtype}'"
            raise TypeError(msg)

        # We may need to strip out trailing NA values
        tail: pa.array | None = None
        na_mask: pa.array | None = None
        pa_array = self._pa_array
        np_func = {
            "cumsum": np.cumsum,
            "cummin": np.minimum.accumulate,
            "cummax": np.maximum.accumulate,
        }[name]

        if self._hasna:
            na_mask = pc.is_null(pa_array)
            if pc.all(na_mask) == pa.scalar(True):
                return type(self)(pa_array)
            if skipna:
                if name == "cumsum":
                    pa_array = pc.fill_null(pa_array, "")
                else:
                    # We can retain the running min/max by forward/backward filling.
                    pa_array = pc.fill_null_forward(pa_array)
                    pa_array = pc.fill_null_backward(pa_array)
            else:
                # When not skipping NA values, the result should be null from
                # the first NA value onward.
                idx = pc.index(na_mask, True).as_py()
                tail = pa.nulls(len(pa_array) - idx, type=pa_array.type)
                pa_array = pa_array[:idx]

        # error: Cannot call function of unknown type
        pa_result = pa.array(np_func(pa_array), type=pa_array.type)  # type: ignore[operator]

        if tail is not None:
            pa_result = pa.concat_arrays([pa_result, tail])
        elif na_mask is not None:
            pa_result = pc.if_else(na_mask, None, pa_result)

        result = type(self)(pa_result)
        return result

    def _reduce_pyarrow(self, name: str, *, skipna: bool = True, **kwargs) -> pa.Scalar:
        """
        Return a pyarrow scalar result of performing the reduction operation.

        Parameters
        ----------
        name : str
            Name of the function, supported values are:
            { any, all, min, max, sum, mean, median, prod,
            std, var, sem, kurt, skew }.
        skipna : bool, default True
            If True, skip NaN values.
        **kwargs
            Additional keyword arguments passed to the reduction function.
            Currently, `ddof` is the only supported kwarg.

        Returns
        -------
        pyarrow scalar

        Raises
        ------
        TypeError : subclass does not define reductions
        """
        pa_type = self._pa_array.type

        data_to_reduce = self._pa_array

        cast_kwargs = {} if pa_version_under13p0 else {"safe": False}

        if name in ["any", "all"] and (
            pa.types.is_integer(pa_type)
            or pa.types.is_floating(pa_type)
            or pa.types.is_duration(pa_type)
            or pa.types.is_decimal(pa_type)
        ):
            # pyarrow only supports any/all for boolean dtype, we allow
            #  for other dtypes, matching our non-pyarrow behavior

            if pa.types.is_duration(pa_type):
                data_to_cmp = self._pa_array.cast(pa.int64())
            else:
                data_to_cmp = self._pa_array

            not_eq = pc.not_equal(data_to_cmp, 0)
            data_to_reduce = not_eq

        elif name in ["min", "max", "sum"] and pa.types.is_duration(pa_type):
            data_to_reduce = self._pa_array.cast(pa.int64())

        elif name in ["median", "mean", "std", "sem"] and pa.types.is_temporal(pa_type):
            nbits = pa_type.bit_width
            if nbits == 32:
                data_to_reduce = self._pa_array.cast(pa.int32())
            else:
                data_to_reduce = self._pa_array.cast(pa.int64())

        if name == "sem":

            def pyarrow_meth(data, skip_nulls, **kwargs):
                numerator = pc.stddev(data, skip_nulls=skip_nulls, **kwargs)
                denominator = pc.sqrt_checked(pc.count(self._pa_array))
                return pc.divide_checked(numerator, denominator)

        elif name == "sum" and (
            pa.types.is_string(pa_type) or pa.types.is_large_string(pa_type)
        ):

            def pyarrow_meth(data, skip_nulls, min_count=0):  # type: ignore[misc]
                mask = pc.is_null(data) if data.null_count > 0 else None
                if skip_nulls:
                    if min_count > 0 and check_below_min_count(
                        (len(data),),
                        None if mask is None else mask.to_numpy(),
                        min_count,
                    ):
                        return pa.scalar(None, type=data.type)
                    if data.null_count > 0:
                        # binary_join returns null if there is any null ->
                        # have to filter out any nulls
                        data = data.filter(pc.invert(mask))
                else:
                    if mask is not None or check_below_min_count(
                        (len(data),), None, min_count
                    ):
                        return pa.scalar(None, type=data.type)

                if pa.types.is_large_string(data.type):
                    # binary_join only supports string, not large_string
                    data = data.cast(pa.string())
                data_list = pa.ListArray.from_arrays(
                    [0, len(data)], data.combine_chunks()
                )[0]
                return pc.binary_join(data_list, "")

        else:
            pyarrow_name = {
                "median": "quantile",
                "prod": "product",
                "std": "stddev",
                "var": "variance",
            }.get(name, name)
            # error: Incompatible types in assignment
            # (expression has type "Optional[Any]", variable has type
            # "Callable[[Any, Any, KwArg(Any)], Any]")
            pyarrow_meth = getattr(pc, pyarrow_name, None)  # type: ignore[assignment]
            if pyarrow_meth is None:
                # Let ExtensionArray._reduce raise the TypeError
                return super()._reduce(name, skipna=skipna, **kwargs)

        # GH51624: pyarrow defaults to min_count=1, pandas behavior is min_count=0
        if name in ["any", "all"] and "min_count" not in kwargs:
            kwargs["min_count"] = 0
        elif name == "median":
            # GH 52679: Use quantile instead of approximate_median
            kwargs["q"] = 0.5

        try:
            result = pyarrow_meth(data_to_reduce, skip_nulls=skipna, **kwargs)
        except (AttributeError, NotImplementedError, TypeError) as err:
            msg = (
                f"'{type(self).__name__}' with dtype {self.dtype} "
                f"does not support reduction '{name}' with pyarrow "
                f"version {pa.__version__}. '{name}' may be supported by "
                f"upgrading pyarrow."
            )
            raise TypeError(msg) from err
        if name == "median":
            # GH 52679: Use quantile instead of approximate_median; returns array
            result = result[0]
        if pc.is_null(result).as_py():
            return result

        if name in ["min", "max", "sum"] and pa.types.is_duration(pa_type):
            result = result.cast(pa_type)
        if name in ["median", "mean"] and pa.types.is_temporal(pa_type):
            if not pa_version_under13p0:
                nbits = pa_type.bit_width
                if nbits == 32:
                    result = result.cast(pa.int32(), **cast_kwargs)
                else:
                    result = result.cast(pa.int64(), **cast_kwargs)
            result = result.cast(pa_type)
        if name in ["std", "sem"] and pa.types.is_temporal(pa_type):
            result = result.cast(pa.int64(), **cast_kwargs)
            if pa.types.is_duration(pa_type):
                result = result.cast(pa_type)
            elif pa.types.is_time(pa_type):
                unit = get_unit_from_pa_dtype(pa_type)
                result = result.cast(pa.duration(unit))
            elif pa.types.is_date(pa_type):
                # go with closest available unit, i.e. "s"
                result = result.cast(pa.duration("s"))
            else:
                # i.e. timestamp
                result = result.cast(pa.duration(pa_type.unit))

        return result

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
        **kwargs
            Additional keyword arguments passed to the reduction function.
            Currently, `ddof` is the only supported kwarg.

        Returns
        -------
        scalar

        Raises
        ------
        TypeError : subclass does not define reductions
        """
        result = self._reduce_calc(name, skipna=skipna, keepdims=keepdims, **kwargs)
        if isinstance(result, pa.Array):
            return type(self)(result)
        else:
            return result

    def _reduce_calc(
        self, name: str, *, skipna: bool = True, keepdims: bool = False, **kwargs
    ):
        pa_result = self._reduce_pyarrow(name, skipna=skipna, **kwargs)

        if keepdims:
            if isinstance(pa_result, pa.Scalar):
                result = pa.array([pa_result.as_py()], type=pa_result.type)
            else:
                result = pa.array(
                    [pa_result],
                    type=to_pyarrow_type(infer_dtype_from_scalar(pa_result)[0]),
                )
            return result

        if pc.is_null(pa_result).as_py():
            return self.dtype.na_value
        elif isinstance(pa_result, pa.Scalar):
            return pa_result.as_py()
        else:
            return pa_result

    def _explode(self):
        """
        See Series.explode.__doc__.
        """
        # child class explode method supports only list types; return
        # default implementation for non list types.
        if not hasattr(self.dtype, "pyarrow_dtype") or (
            not pa.types.is_list(self.dtype.pyarrow_dtype)
        ):
            return super()._explode()
        values = self
        counts = pa.compute.list_value_length(values._pa_array)
        counts = counts.fill_null(1).to_numpy()
        fill_value = pa.scalar([None], type=self._pa_array.type)
        mask = counts == 0
        if mask.any():
            values = values.copy()
            values[mask] = fill_value
            counts = counts.copy()
            counts[mask] = 1
        values = values.fillna(fill_value)
        values = type(self)(pa.compute.list_flatten(values._pa_array))
        return values, counts

    def __setitem__(self, key, value) -> None:
        """Set one or more values inplace.

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
        # GH50085: unwrap 1D indexers
        if isinstance(key, tuple) and len(key) == 1:
            key = key[0]

        key = check_array_indexer(self, key)
        value = self._maybe_convert_setitem_value(value)

        if com.is_null_slice(key):
            # fast path (GH50248)
            data = self._if_else(True, value, self._pa_array)

        elif is_integer(key):
            # fast path
            key = cast(int, key)
            n = len(self)
            if key < 0:
                key += n
            if not 0 <= key < n:
                raise IndexError(
                    f"index {key} is out of bounds for axis 0 with size {n}"
                )
            if isinstance(value, pa.Scalar):
                value = value.as_py()
            elif is_list_like(value):
                raise ValueError("Length of indexer and values mismatch")
            chunks = [
                *self._pa_array[:key].chunks,
                pa.array([value], type=self._pa_array.type, from_pandas=True),
                *self._pa_array[key + 1 :].chunks,
            ]
            data = pa.chunked_array(chunks).combine_chunks()

        elif is_bool_dtype(key):
            key = np.asarray(key, dtype=np.bool_)
            data = self._replace_with_mask(self._pa_array, key, value)

        elif is_scalar(value) or isinstance(value, pa.Scalar):
            mask = np.zeros(len(self), dtype=np.bool_)
            mask[key] = True
            data = self._if_else(mask, value, self._pa_array)

        else:
            indices = np.arange(len(self))[key]
            if len(indices) != len(value):
                raise ValueError("Length of indexer and values mismatch")
            if len(indices) == 0:
                return
            argsort = np.argsort(indices)
            indices = indices[argsort]
            value = value.take(argsort)
            mask = np.zeros(len(self), dtype=np.bool_)
            mask[indices] = True
            data = self._replace_with_mask(self._pa_array, mask, value)

        if isinstance(data, pa.Array):
            data = pa.chunked_array([data])
        self._pa_array = data

    def _rank_calc(
        self,
        *,
        axis: AxisInt = 0,
        method: str = "average",
        na_option: str = "keep",
        ascending: bool = True,
        pct: bool = False,
    ):
        if axis != 0:
            ranked = super()._rank(
                axis=axis,
                method=method,
                na_option=na_option,
                ascending=ascending,
                pct=pct,
            )
            # keep dtypes consistent with the implementation below
            if method == "average" or pct:
                pa_type = pa.float64()
            else:
                pa_type = pa.uint64()
            result = pa.array(ranked, type=pa_type, from_pandas=True)
            return result

        data = self._pa_array.combine_chunks()
        sort_keys = "ascending" if ascending else "descending"
        null_placement = "at_start" if na_option == "top" else "at_end"
        tiebreaker = "min" if method == "average" else method

        result = pc.rank(
            data,
            sort_keys=sort_keys,
            null_placement=null_placement,
            tiebreaker=tiebreaker,
        )

        if na_option == "keep":
            mask = pc.is_null(self._pa_array)
            null = pa.scalar(None, type=result.type)
            result = pc.if_else(mask, null, result)

        if method == "average":
            result_max = pc.rank(
                data,
                sort_keys=sort_keys,
                null_placement=null_placement,
                tiebreaker="max",
            )
            result_max = result_max.cast(pa.float64())
            result_min = result.cast(pa.float64())
            result = pc.divide(pc.add(result_min, result_max), 2)

        if pct:
            if not pa.types.is_floating(result.type):
                result = result.cast(pa.float64())
            if method == "dense":
                divisor = pc.max(result)
            else:
                divisor = pc.count(result)
            result = pc.divide(result, divisor)

        return result

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
        return self._convert_rank_result(
            self._rank_calc(
                axis=axis,
                method=method,
                na_option=na_option,
                ascending=ascending,
                pct=pct,
            )
        )

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
        pa_dtype = self._pa_array.type

        data = self._pa_array
        if pa.types.is_temporal(pa_dtype):
            # https://github.com/apache/arrow/issues/33769 in these cases
            #  we can cast to ints and back
            nbits = pa_dtype.bit_width
            if nbits == 32:
                data = data.cast(pa.int32())
            else:
                data = data.cast(pa.int64())

        result = pc.quantile(data, q=qs, interpolation=interpolation)

        if pa.types.is_temporal(pa_dtype):
            if pa.types.is_floating(result.type):
                result = pc.floor(result)
            nbits = pa_dtype.bit_width
            if nbits == 32:
                result = result.cast(pa.int32())
            else:
                result = result.cast(pa.int64())
            result = result.cast(pa_dtype)

        return type(self)(result)

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
        pa_type = self._pa_array.type
        if pa.types.is_temporal(pa_type):
            nbits = pa_type.bit_width
            if nbits == 32:
                data = self._pa_array.cast(pa.int32())
            elif nbits == 64:
                data = self._pa_array.cast(pa.int64())
            else:
                raise NotImplementedError(pa_type)
        else:
            data = self._pa_array

        if dropna:
            data = data.drop_null()

        res = pc.value_counts(data)
        most_common = res.field("values").filter(
            pc.equal(res.field("counts"), pc.max(res.field("counts")))
        )

        if pa.types.is_temporal(pa_type):
            most_common = most_common.cast(pa_type)

        most_common = most_common.take(pc.array_sort_indices(most_common))
        return type(self)(most_common)

    def _maybe_convert_setitem_value(self, value):
        """Maybe convert value to be pyarrow compatible."""
        try:
            value = self._box_pa(value, self._pa_array.type)
        except pa.ArrowTypeError as err:
            msg = f"Invalid value '{value!s}' for dtype '{self.dtype}'"
            raise TypeError(msg) from err
        return value

    def interpolate(
        self,
        *,
        method: InterpolateOptions,
        axis: int,
        index,
        limit,
        limit_direction,
        limit_area,
        copy: bool,
        **kwargs,
    ) -> Self:
        """
        See NDFrame.interpolate.__doc__.
        """
        # NB: we return type(self) even if copy=False
        if not self.dtype._is_numeric:
            raise TypeError(f"Cannot interpolate with {self.dtype} dtype")

        mask = self.isna()
        if self.dtype.kind == "f":
            data = self._pa_array.to_numpy()
        elif self.dtype.kind in "iu":
            data = self.to_numpy(dtype="f8", na_value=0.0)
        else:
            raise NotImplementedError(
                f"interpolate is not implemented for dtype={self.dtype}"
            )

        missing.interpolate_2d_inplace(
            data,
            method=method,
            axis=0,
            index=index,
            limit=limit,
            limit_direction=limit_direction,
            limit_area=limit_area,
            mask=mask,
            **kwargs,
        )
        return type(self)(self._box_pa_array(pa.array(data, mask=mask)))

    @classmethod
    def _if_else(
        cls,
        cond: npt.NDArray[np.bool_] | bool,
        left: ArrayLike | Scalar,
        right: ArrayLike | Scalar,
    ):
        """
        Choose values based on a condition.

        Analogous to pyarrow.compute.if_else, with logic
        to fallback to numpy for unsupported types.

        Parameters
        ----------
        cond : npt.NDArray[np.bool_] or bool
        left : ArrayLike | Scalar
        right : ArrayLike | Scalar

        Returns
        -------
        pa.Array
        """
        try:
            return pc.if_else(cond, left, right)
        except pa.ArrowNotImplementedError:
            pass

        def _to_numpy_and_type(value) -> tuple[np.ndarray, pa.DataType | None]:
            if isinstance(value, (pa.Array, pa.ChunkedArray)):
                pa_type = value.type
            elif isinstance(value, pa.Scalar):
                pa_type = value.type
                value = value.as_py()
            else:
                pa_type = None
            return np.array(value, dtype=object), pa_type

        left, left_type = _to_numpy_and_type(left)
        right, right_type = _to_numpy_and_type(right)
        pa_type = left_type or right_type
        result = np.where(cond, left, right)
        return pa.array(result, type=pa_type, from_pandas=True)

    @classmethod
    def _replace_with_mask(
        cls,
        values: pa.Array | pa.ChunkedArray,
        mask: npt.NDArray[np.bool_] | bool,
        replacements: ArrayLike | Scalar,
    ):
        """
        Replace items selected with a mask.

        Analogous to pyarrow.compute.replace_with_mask, with logic
        to fallback to numpy for unsupported types.

        Parameters
        ----------
        values : pa.Array or pa.ChunkedArray
        mask : npt.NDArray[np.bool_] or bool
        replacements : ArrayLike or Scalar
            Replacement value(s)

        Returns
        -------
        pa.Array or pa.ChunkedArray
        """
        if isinstance(replacements, pa.ChunkedArray):
            # replacements must be array or scalar, not ChunkedArray
            replacements = replacements.combine_chunks()
        if isinstance(values, pa.ChunkedArray) and pa.types.is_boolean(values.type):
            # GH#52059 replace_with_mask segfaults for chunked array
            # https://github.com/apache/arrow/issues/34634
            values = values.combine_chunks()
        try:
            return pc.replace_with_mask(values, mask, replacements)
        except pa.ArrowNotImplementedError:
            pass
        if isinstance(replacements, pa.Array):
            replacements = np.array(replacements, dtype=object)
        elif isinstance(replacements, pa.Scalar):
            replacements = replacements.as_py()
        result = np.array(values, dtype=object)
        result[mask] = replacements
        return pa.array(result, type=values.type, from_pandas=True)

    # ------------------------------------------------------------------
    # GroupBy Methods

    def _to_masked(self):
        pa_dtype = self._pa_array.type

        if pa.types.is_floating(pa_dtype) or pa.types.is_integer(pa_dtype):
            na_value = 1
        elif pa.types.is_boolean(pa_dtype):
            na_value = True
        else:
            raise NotImplementedError

        dtype = _arrow_dtype_mapping()[pa_dtype]
        mask = self.isna()
        arr = self.to_numpy(dtype=dtype.numpy_dtype, na_value=na_value)
        return dtype.construct_array_type()(arr, mask)

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
        if isinstance(self.dtype, StringDtype):
            if how in [
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
            return super()._groupby_op(
                how=how,
                has_dropped_na=has_dropped_na,
                min_count=min_count,
                ngroups=ngroups,
                ids=ids,
                **kwargs,
            )

        # maybe convert to a compatible dtype optimized for groupby
        values: ExtensionArray
        pa_type = self._pa_array.type
        if pa.types.is_timestamp(pa_type):
            values = self._to_datetimearray()
        elif pa.types.is_duration(pa_type):
            values = self._to_timedeltaarray()
        else:
            values = self._to_masked()

        result = values._groupby_op(
            how=how,
            has_dropped_na=has_dropped_na,
            min_count=min_count,
            ngroups=ngroups,
            ids=ids,
            **kwargs,
        )
        if isinstance(result, np.ndarray):
            return result
        return type(self)._from_sequence(result, copy=False)

    def _apply_elementwise(self, func: Callable) -> list[list[Any]]:
        """Apply a callable to each element while maintaining the chunking structure."""
        return [
            [
                None if val is None else func(val)
                for val in chunk.to_numpy(zero_copy_only=False)
            ]
            for chunk in self._pa_array.iterchunks()
        ]

    def _convert_bool_result(self, result, na=lib.no_default, method_name=None):
        if na is not lib.no_default and not isna(
            na
        ):  # pyright: ignore [reportGeneralTypeIssues]
            result = result.fill_null(na)
        return type(self)(result)

    def _convert_int_result(self, result):
        return type(self)(result)

    def _convert_rank_result(self, result):
        return type(self)(result)

    def _str_count(self, pat: str, flags: int = 0):
        if flags:
            raise NotImplementedError(f"count not implemented with {flags=}")
        return type(self)(pc.count_substring_regex(self._pa_array, pat))

    def _str_repeat(self, repeats: int | Sequence[int]):
        if not isinstance(repeats, int):
            raise NotImplementedError(
                f"repeat is not implemented when repeats is {type(repeats).__name__}"
            )
        else:
            return type(self)(pc.binary_repeat(self._pa_array, repeats))

    def _str_join(self, sep: str):
        if pa.types.is_string(self._pa_array.type) or pa.types.is_large_string(
            self._pa_array.type
        ):
            result = self._apply_elementwise(list)
            result = pa.chunked_array(result, type=pa.list_(pa.string()))
        else:
            result = self._pa_array
        return type(self)(pc.binary_join(result, sep))

    def _str_partition(self, sep: str, expand: bool):
        predicate = lambda val: val.partition(sep)
        result = self._apply_elementwise(predicate)
        return type(self)(pa.chunked_array(result))

    def _str_rpartition(self, sep: str, expand: bool):
        predicate = lambda val: val.rpartition(sep)
        result = self._apply_elementwise(predicate)
        return type(self)(pa.chunked_array(result))

    def _str_casefold(self):
        predicate = lambda val: val.casefold()
        result = self._apply_elementwise(predicate)
        return type(self)(pa.chunked_array(result))

    def _str_encode(self, encoding: str, errors: str = "strict"):
        predicate = lambda val: val.encode(encoding, errors)
        result = self._apply_elementwise(predicate)
        return type(self)(pa.chunked_array(result))

    def _str_extract(self, pat: str, flags: int = 0, expand: bool = True):
        if flags:
            raise NotImplementedError("Only flags=0 is implemented.")
        groups = re.compile(pat).groupindex.keys()
        if len(groups) == 0:
            raise ValueError(f"{pat=} must contain a symbolic group name.")
        result = pc.extract_regex(self._pa_array, pat)
        if expand:
            return {
                col: type(self)(pc.struct_field(result, [i]))
                for col, i in zip(groups, range(result.type.num_fields))
            }
        else:
            return type(self)(pc.struct_field(result, [0]))

    def _str_findall(self, pat: str, flags: int = 0):
        regex = re.compile(pat, flags=flags)
        predicate = lambda val: regex.findall(val)
        result = self._apply_elementwise(predicate)
        return type(self)(pa.chunked_array(result))

    def _str_get_dummies(self, sep: str = "|"):
        split = pc.split_pattern(self._pa_array, sep)
        flattened_values = pc.list_flatten(split)
        uniques = flattened_values.unique()
        uniques_sorted = uniques.take(pa.compute.array_sort_indices(uniques))
        lengths = pc.list_value_length(split).fill_null(0).to_numpy()
        n_rows = len(self)
        n_cols = len(uniques)
        indices = pc.index_in(flattened_values, uniques_sorted).to_numpy()
        indices = indices + np.arange(n_rows).repeat(lengths) * n_cols
        dummies = np.zeros(n_rows * n_cols, dtype=np.bool_)
        dummies[indices] = True
        dummies = dummies.reshape((n_rows, n_cols))
        result = type(self)(pa.array(list(dummies)))
        return result, uniques_sorted.to_pylist()

    def _str_index(self, sub: str, start: int = 0, end: int | None = None):
        predicate = lambda val: val.index(sub, start, end)
        result = self._apply_elementwise(predicate)
        return type(self)(pa.chunked_array(result))

    def _str_rindex(self, sub: str, start: int = 0, end: int | None = None):
        predicate = lambda val: val.rindex(sub, start, end)
        result = self._apply_elementwise(predicate)
        return type(self)(pa.chunked_array(result))

    def _str_normalize(self, form: str):
        predicate = lambda val: unicodedata.normalize(form, val)
        result = self._apply_elementwise(predicate)
        return type(self)(pa.chunked_array(result))

    def _str_rfind(self, sub: str, start: int = 0, end=None):
        predicate = lambda val: val.rfind(sub, start, end)
        result = self._apply_elementwise(predicate)
        return type(self)(pa.chunked_array(result))

    def _str_split(
        self,
        pat: str | None = None,
        n: int | None = -1,
        expand: bool = False,
        regex: bool | None = None,
    ):
        if n in {-1, 0}:
            n = None
        if pat is None:
            split_func = pc.utf8_split_whitespace
        elif regex:
            split_func = functools.partial(pc.split_pattern_regex, pattern=pat)
        else:
            split_func = functools.partial(pc.split_pattern, pattern=pat)
        return type(self)(split_func(self._pa_array, max_splits=n))

    def _str_rsplit(self, pat: str | None = None, n: int | None = -1):
        if n in {-1, 0}:
            n = None
        if pat is None:
            return type(self)(
                pc.utf8_split_whitespace(self._pa_array, max_splits=n, reverse=True)
            )
        else:
            return type(self)(
                pc.split_pattern(self._pa_array, pat, max_splits=n, reverse=True)
            )

    def _str_translate(self, table: dict[int, str]):
        predicate = lambda val: val.translate(table)
        result = self._apply_elementwise(predicate)
        return type(self)(pa.chunked_array(result))

    def _str_wrap(self, width: int, **kwargs):
        kwargs["width"] = width
        tw = textwrap.TextWrapper(**kwargs)
        predicate = lambda val: "\n".join(tw.wrap(val))
        result = self._apply_elementwise(predicate)
        return type(self)(pa.chunked_array(result))

    @property
    def _dt_days(self):
        return type(self)(
            pa.array(self._to_timedeltaarray().days, from_pandas=True, type=pa.int32())
        )

    @property
    def _dt_hours(self):
        return type(self)(
            pa.array(
                [
                    td.components.hours if td is not NaT else None
                    for td in self._to_timedeltaarray()
                ],
                type=pa.int32(),
            )
        )

    @property
    def _dt_minutes(self):
        return type(self)(
            pa.array(
                [
                    td.components.minutes if td is not NaT else None
                    for td in self._to_timedeltaarray()
                ],
                type=pa.int32(),
            )
        )

    @property
    def _dt_seconds(self):
        return type(self)(
            pa.array(
                self._to_timedeltaarray().seconds, from_pandas=True, type=pa.int32()
            )
        )

    @property
    def _dt_milliseconds(self):
        return type(self)(
            pa.array(
                [
                    td.components.milliseconds if td is not NaT else None
                    for td in self._to_timedeltaarray()
                ],
                type=pa.int32(),
            )
        )

    @property
    def _dt_microseconds(self):
        return type(self)(
            pa.array(
                self._to_timedeltaarray().microseconds,
                from_pandas=True,
                type=pa.int32(),
            )
        )

    @property
    def _dt_nanoseconds(self):
        return type(self)(
            pa.array(
                self._to_timedeltaarray().nanoseconds, from_pandas=True, type=pa.int32()
            )
        )

    def _dt_to_pytimedelta(self):
        data = self._pa_array.to_pylist()
        if self._dtype.pyarrow_dtype.unit == "ns":
            data = [None if ts is None else ts.to_pytimedelta() for ts in data]
        return np.array(data, dtype=object)

    def _dt_total_seconds(self):
        return type(self)(
            pa.array(self._to_timedeltaarray().total_seconds(), from_pandas=True)
        )

    def _dt_as_unit(self, unit: str):
        if pa.types.is_date(self.dtype.pyarrow_dtype):
            raise NotImplementedError("as_unit not implemented for date types")
        pd_array = self._maybe_convert_datelike_array()
        # Don't just cast _pa_array in order to follow pandas unit conversion rules
        return type(self)(pa.array(pd_array.as_unit(unit), from_pandas=True))

    @property
    def _dt_year(self):
        return type(self)(pc.year(self._pa_array))

    @property
    def _dt_day(self):
        return type(self)(pc.day(self._pa_array))

    @property
    def _dt_day_of_week(self):
        return type(self)(pc.day_of_week(self._pa_array))

    _dt_dayofweek = _dt_day_of_week
    _dt_weekday = _dt_day_of_week

    @property
    def _dt_day_of_year(self):
        return type(self)(pc.day_of_year(self._pa_array))

    _dt_dayofyear = _dt_day_of_year

    @property
    def _dt_hour(self):
        return type(self)(pc.hour(self._pa_array))

    def _dt_isocalendar(self):
        return type(self)(pc.iso_calendar(self._pa_array))

    @property
    def _dt_is_leap_year(self):
        return type(self)(pc.is_leap_year(self._pa_array))

    @property
    def _dt_is_month_start(self):
        return type(self)(pc.equal(pc.day(self._pa_array), 1))

    @property
    def _dt_is_month_end(self):
        result = pc.equal(
            pc.days_between(
                pc.floor_temporal(self._pa_array, unit="day"),
                pc.ceil_temporal(self._pa_array, unit="month"),
            ),
            1,
        )
        return type(self)(result)

    @property
    def _dt_is_year_start(self):
        return type(self)(
            pc.and_(
                pc.equal(pc.month(self._pa_array), 1),
                pc.equal(pc.day(self._pa_array), 1),
            )
        )

    @property
    def _dt_is_year_end(self):
        return type(self)(
            pc.and_(
                pc.equal(pc.month(self._pa_array), 12),
                pc.equal(pc.day(self._pa_array), 31),
            )
        )

    @property
    def _dt_is_quarter_start(self):
        result = pc.equal(
            pc.floor_temporal(self._pa_array, unit="quarter"),
            pc.floor_temporal(self._pa_array, unit="day"),
        )
        return type(self)(result)

    @property
    def _dt_is_quarter_end(self):
        result = pc.equal(
            pc.days_between(
                pc.floor_temporal(self._pa_array, unit="day"),
                pc.ceil_temporal(self._pa_array, unit="quarter"),
            ),
            1,
        )
        return type(self)(result)

    @property
    def _dt_days_in_month(self):
        result = pc.days_between(
            pc.floor_temporal(self._pa_array, unit="month"),
            pc.ceil_temporal(self._pa_array, unit="month"),
        )
        return type(self)(result)

    _dt_daysinmonth = _dt_days_in_month

    @property
    def _dt_microsecond(self):
        return type(self)(pc.microsecond(self._pa_array))

    @property
    def _dt_minute(self):
        return type(self)(pc.minute(self._pa_array))

    @property
    def _dt_month(self):
        return type(self)(pc.month(self._pa_array))

    @property
    def _dt_nanosecond(self):
        return type(self)(pc.nanosecond(self._pa_array))

    @property
    def _dt_quarter(self):
        return type(self)(pc.quarter(self._pa_array))

    @property
    def _dt_second(self):
        return type(self)(pc.second(self._pa_array))

    @property
    def _dt_date(self):
        return type(self)(self._pa_array.cast(pa.date32()))

    @property
    def _dt_time(self):
        unit = (
            self.dtype.pyarrow_dtype.unit
            if self.dtype.pyarrow_dtype.unit in {"us", "ns"}
            else "ns"
        )
        return type(self)(self._pa_array.cast(pa.time64(unit)))

    @property
    def _dt_tz(self):
        return timezones.maybe_get_tz(self.dtype.pyarrow_dtype.tz)

    @property
    def _dt_unit(self):
        return self.dtype.pyarrow_dtype.unit

    def _dt_normalize(self):
        return type(self)(pc.floor_temporal(self._pa_array, 1, "day"))

    def _dt_strftime(self, format: str):
        return type(self)(pc.strftime(self._pa_array, format=format))

    def _round_temporally(
        self,
        method: Literal["ceil", "floor", "round"],
        freq,
        ambiguous: TimeAmbiguous = "raise",
        nonexistent: TimeNonexistent = "raise",
    ):
        if ambiguous != "raise":
            raise NotImplementedError("ambiguous is not supported.")
        if nonexistent != "raise":
            raise NotImplementedError("nonexistent is not supported.")
        offset = to_offset(freq)
        if offset is None:
            raise ValueError(f"Must specify a valid frequency: {freq}")
        pa_supported_unit = {
            "Y": "year",
            "YS": "year",
            "Q": "quarter",
            "QS": "quarter",
            "M": "month",
            "MS": "month",
            "W": "week",
            "D": "day",
            "h": "hour",
            "min": "minute",
            "s": "second",
            "ms": "millisecond",
            "us": "microsecond",
            "ns": "nanosecond",
        }
        unit = pa_supported_unit.get(offset._prefix, None)
        if unit is None:
            raise ValueError(f"{freq=} is not supported")
        multiple = offset.n
        rounding_method = getattr(pc, f"{method}_temporal")
        return type(self)(rounding_method(self._pa_array, multiple=multiple, unit=unit))

    def _dt_ceil(
        self,
        freq,
        ambiguous: TimeAmbiguous = "raise",
        nonexistent: TimeNonexistent = "raise",
    ):
        return self._round_temporally("ceil", freq, ambiguous, nonexistent)

    def _dt_floor(
        self,
        freq,
        ambiguous: TimeAmbiguous = "raise",
        nonexistent: TimeNonexistent = "raise",
    ):
        return self._round_temporally("floor", freq, ambiguous, nonexistent)

    def _dt_round(
        self,
        freq,
        ambiguous: TimeAmbiguous = "raise",
        nonexistent: TimeNonexistent = "raise",
    ):
        return self._round_temporally("round", freq, ambiguous, nonexistent)

    def _dt_day_name(self, locale: str | None = None):
        if locale is None:
            locale = "C"
        return type(self)(pc.strftime(self._pa_array, format="%A", locale=locale))

    def _dt_month_name(self, locale: str | None = None):
        if locale is None:
            locale = "C"
        return type(self)(pc.strftime(self._pa_array, format="%B", locale=locale))

    def _dt_to_pydatetime(self):
        if pa.types.is_date(self.dtype.pyarrow_dtype):
            raise ValueError(
                f"to_pydatetime cannot be called with {self.dtype.pyarrow_dtype} type. "
                "Convert to pyarrow timestamp type."
            )
        data = self._pa_array.to_pylist()
        if self._dtype.pyarrow_dtype.unit == "ns":
            data = [None if ts is None else ts.to_pydatetime(warn=False) for ts in data]
        return np.array(data, dtype=object)

    def _dt_tz_localize(
        self,
        tz,
        ambiguous: TimeAmbiguous = "raise",
        nonexistent: TimeNonexistent = "raise",
    ):
        if ambiguous != "raise":
            raise NotImplementedError(f"{ambiguous=} is not supported")
        nonexistent_pa = {
            "raise": "raise",
            "shift_backward": "earliest",
            "shift_forward": "latest",
        }.get(
            nonexistent, None  # type: ignore[arg-type]
        )
        if nonexistent_pa is None:
            raise NotImplementedError(f"{nonexistent=} is not supported")
        if tz is None:
            result = self._pa_array.cast(pa.timestamp(self.dtype.pyarrow_dtype.unit))
        else:
            result = pc.assume_timezone(
                self._pa_array, str(tz), ambiguous=ambiguous, nonexistent=nonexistent_pa
            )
        return type(self)(result)

    def _dt_tz_convert(self, tz):
        if self.dtype.pyarrow_dtype.tz is None:
            raise TypeError(
                "Cannot convert tz-naive timestamps, use tz_localize to localize"
            )
        current_unit = self.dtype.pyarrow_dtype.unit
        result = self._pa_array.cast(pa.timestamp(current_unit, tz))
        return type(self)(result)


def transpose_homogeneous_pyarrow(
    arrays: Sequence[ArrowExtensionArray],
) -> list[ArrowExtensionArray]:
    """Transpose arrow extension arrays in a list, but faster.

    Input should be a list of arrays of equal length and all have the same
    dtype. The caller is responsible for ensuring validity of input data.
    """
    arrays = list(arrays)
    nrows, ncols = len(arrays[0]), len(arrays)
    indices = np.arange(nrows * ncols).reshape(ncols, nrows).T.flatten()
    arr = pa.chunked_array([chunk for arr in arrays for chunk in arr._pa_array.chunks])
    arr = arr.take(indices)
    return [ArrowExtensionArray(arr.slice(i * ncols, ncols)) for i in range(nrows)]
<|endoftext|># Disable type checking for this module since numba's internals
# are not typed, and we use numba's internals via its extension API
# mypy: ignore-errors
"""
Utility classes/functions to let numba recognize
pandas Index/Series/DataFrame

Mostly vendored from https://github.com/numba/numba/blob/main/numba/tests/pdlike_usecase.py
"""

from __future__ import annotations

from contextlib import contextmanager
import operator

import numba
from numba import types
from numba.core import cgutils
from numba.core.datamodel import models
from numba.core.extending import (
    NativeValue,
    box,
    lower_builtin,
    make_attribute_wrapper,
    overload,
    overload_attribute,
    overload_method,
    register_model,
    type_callable,
    typeof_impl,
    unbox,
)
from numba.core.imputils import impl_ret_borrowed
import numpy as np

from pandas._libs import lib

from pandas.core.indexes.base import Index
from pandas.core.indexing import _iLocIndexer
from pandas.core.internals import SingleBlockManager
from pandas.core.series import Series


# Helper function to hack around fact that Index casts numpy string dtype to object
#
# Idea is to set an attribute on a Index called _numba_data
# that is the original data, or the object data casted to numpy string dtype,
# with a context manager that is unset afterwards
@contextmanager
def set_numba_data(index: Index):
    numba_data = index._data
    if numba_data.dtype in (object, "string"):
        numba_data = np.asarray(numba_data)
        if not lib.is_string_array(numba_data):
            raise ValueError(
                "The numba engine only supports using string or numeric column names"
            )
        numba_data = numba_data.astype("U")
    try:
        index._numba_data = numba_data
        yield index
    finally:
        del index._numba_data


# TODO: Range index support
# (this currently lowers OK, but does not round-trip)
class IndexType(types.Type):
    """
    The type class for Index objects.
    """

    def __init__(self, dtype, layout, pyclass: any) -> None:
        self.pyclass = pyclass
        name = f"index({dtype}, {layout})"
        self.dtype = dtype
        self.layout = layout
        super().__init__(name)

    @property
    def key(self):
        return self.pyclass, self.dtype, self.layout

    @property
    def as_array(self):
        return types.Array(self.dtype, 1, self.layout)

    def copy(self, dtype=None, ndim: int = 1, layout=None):
        assert ndim == 1
        if dtype is None:
            dtype = self.dtype
        layout = layout or self.layout
        return type(self)(dtype, layout, self.pyclass)


class SeriesType(types.Type):
    """
    The type class for Series objects.
    """

    def __init__(self, dtype, index, namety) -> None:
        assert isinstance(index, IndexType)
        self.dtype = dtype
        self.index = index
        self.values = types.Array(self.dtype, 1, "C")
        self.namety = namety
        name = f"series({dtype}, {index}, {namety})"
        super().__init__(name)

    @property
    def key(self):
        return self.dtype, self.index, self.namety

    @property
    def as_array(self):
        return self.values

    def copy(self, dtype=None, ndim: int = 1, layout: str = "C"):
        assert ndim == 1
        assert layout == "C"
        if dtype is None:
            dtype = self.dtype
        return type(self)(dtype, self.index, self.namety)


@typeof_impl.register(Index)
def typeof_index(val, c):
    """
    This will assume that only strings are in object dtype
    index.
    (you should check this before this gets lowered down to numba)
    """
    # arrty = typeof_impl(val._data, c)
    arrty = typeof_impl(val._numba_data, c)
    assert arrty.ndim == 1
    return IndexType(arrty.dtype, arrty.layout, type(val))


@typeof_impl.register(Series)
def typeof_series(val, c):
    index = typeof_impl(val.index, c)
    arrty = typeof_impl(val.values, c)
    namety = typeof_impl(val.name, c)
    assert arrty.ndim == 1
    assert arrty.layout == "C"
    return SeriesType(arrty.dtype, index, namety)


@type_callable(Series)
def type_series_constructor(context):
    def typer(data, index, name=None):
        if isinstance(index, IndexType) and isinstance(data, types.Array):
            assert data.ndim == 1
            if name is None:
                name = types.intp
            return SeriesType(data.dtype, index, name)

    return typer


@type_callable(Index)
def type_index_constructor(context):
    def typer(data, hashmap=None):
        if isinstance(data, types.Array):
            assert data.layout == "C"
            assert data.ndim == 1
            assert hashmap is None or isinstance(hashmap, types.DictType)
            return IndexType(data.dtype, layout=data.layout, pyclass=Index)

    return typer


# Backend extensions for Index and Series and Frame
@register_model(IndexType)
class IndexModel(models.StructModel):
    def __init__(self, dmm, fe_type) -> None:
        # We don't want the numpy string scalar type in our hashmap
        members = [
            ("data", fe_type.as_array),
            # This is an attempt to emulate our hashtable code with a numba
            # typed dict
            # It maps from values in the index to their integer positions in the array
            ("hashmap", types.DictType(fe_type.dtype, types.intp)),
            # Pointer to the Index object this was created from, or that it
            # boxes to
            # https://numba.discourse.group/t/qst-how-to-cache-the-boxing-of-an-object/2128/2?u=lithomas1
            ("parent", types.pyobject),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


@register_model(SeriesType)
class SeriesModel(models.StructModel):
    def __init__(self, dmm, fe_type) -> None:
        members = [
            ("index", fe_type.index),
            ("values", fe_type.as_array),
            ("name", fe_type.namety),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(IndexType, "data", "_data")
make_attribute_wrapper(IndexType, "hashmap", "hashmap")

make_attribute_wrapper(SeriesType, "index", "index")
make_attribute_wrapper(SeriesType, "values", "values")
make_attribute_wrapper(SeriesType, "name", "name")


@lower_builtin(Series, types.Array, IndexType)
def pdseries_constructor(context, builder, sig, args):
    data, index = args
    series = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    series.index = index
    series.values = data
    series.name = context.get_constant(types.intp, 0)
    return impl_ret_borrowed(context, builder, sig.return_type, series._getvalue())


@lower_builtin(Series, types.Array, IndexType, types.intp)
@lower_builtin(Series, types.Array, IndexType, types.float64)
@lower_builtin(Series, types.Array, IndexType, types.unicode_type)
def pdseries_constructor_with_name(context, builder, sig, args):
    data, index, name = args
    series = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    series.index = index
    series.values = data
    series.name = name
    return impl_ret_borrowed(context, builder, sig.return_type, series._getvalue())


@lower_builtin(Index, types.Array, types.DictType, types.pyobject)
def index_constructor_2arg(context, builder, sig, args):
    (data, hashmap, parent) = args
    index = cgutils.create_struct_proxy(sig.return_type)(context, builder)

    index.data = data
    index.hashmap = hashmap
    index.parent = parent
    return impl_ret_borrowed(context, builder, sig.return_type, index._getvalue())


@lower_builtin(Index, types.Array, types.DictType)
def index_constructor_2arg_parent(context, builder, sig, args):
    # Basically same as index_constructor_1arg, but also lets you specify the
    # parent object
    (data, hashmap) = args
    index = cgutils.create_struct_proxy(sig.return_type)(context, builder)

    index.data = data
    index.hashmap = hashmap
    return impl_ret_borrowed(context, builder, sig.return_type, index._getvalue())


@lower_builtin(Index, types.Array)
def index_constructor_1arg(context, builder, sig, args):
    from numba.typed import Dict

    key_type = sig.return_type.dtype
    value_type = types.intp

    def index_impl(data):
        return Index(data, Dict.empty(key_type, value_type))

    return context.compile_internal(builder, index_impl, sig, args)


# Helper to convert the unicodecharseq (numpy string scalar) into a unicode_type
# (regular string)
def maybe_cast_str(x):
    # Dummy function that numba can overload
    pass


@overload(maybe_cast_str)
def maybe_cast_str_impl(x):
    """Converts numba UnicodeCharSeq (numpy string scalar) -> unicode type (string).
    Is a no-op for other types."""
    if isinstance(x, types.UnicodeCharSeq):
        return lambda x: str(x)
    else:
        return lambda x: x


@unbox(IndexType)
def unbox_index(typ, obj, c):
    """
    Convert a Index object to a native structure.

    Note: Object dtype is not allowed here
    """
    data_obj = c.pyapi.object_getattr_string(obj, "_numba_data")
    index = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    # If we see an object array, assume its been validated as only containing strings
    # We still need to do the conversion though
    index.data = c.unbox(typ.as_array, data_obj).value
    typed_dict_obj = c.pyapi.unserialize(c.pyapi.serialize_object(numba.typed.Dict))
    # Create an empty typed dict in numba for the hashmap for indexing
    # equiv of numba.typed.Dict.empty(typ.dtype, types.intp)
    arr_type_obj = c.pyapi.unserialize(c.pyapi.serialize_object(typ.dtype))
    intp_type_obj = c.pyapi.unserialize(c.pyapi.serialize_object(types.intp))
    hashmap_obj = c.pyapi.call_method(
        typed_dict_obj, "empty", (arr_type_obj, intp_type_obj)
    )
    index.hashmap = c.unbox(types.DictType(typ.dtype, types.intp), hashmap_obj).value
    # Set the parent for speedy boxing.
    index.parent = obj

    # Decrefs
    c.pyapi.decref(data_obj)
    c.pyapi.decref(arr_type_obj)
    c.pyapi.decref(intp_type_obj)
    c.pyapi.decref(typed_dict_obj)

    return NativeValue(index._getvalue())


@unbox(SeriesType)
def unbox_series(typ, obj, c):
    """
    Convert a Series object to a native structure.
    """
    index_obj = c.pyapi.object_getattr_string(obj, "index")
    values_obj = c.pyapi.object_getattr_string(obj, "values")
    name_obj = c.pyapi.object_getattr_string(obj, "name")

    series = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    series.index = c.unbox(typ.index, index_obj).value
    series.values = c.unbox(typ.values, values_obj).value
    series.name = c.unbox(typ.namety, name_obj).value

    # Decrefs
    c.pyapi.decref(index_obj)
    c.pyapi.decref(values_obj)
    c.pyapi.decref(name_obj)

    return NativeValue(series._getvalue())


@box(IndexType)
def box_index(typ, val, c):
    """
    Convert a native index structure to a Index object.

    If our native index is of a numpy string dtype, we'll cast it to
    object.
    """
    # First build a Numpy array object, then wrap it in a Index
    index = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)

    res = cgutils.alloca_once_value(c.builder, index.parent)

    # Does parent exist?
    # (it means already boxed once, or Index same as original df.index or df.columns)
    # xref https://github.com/numba/numba/blob/596e8a55334cc46854e3192766e643767bd7c934/numba/core/boxing.py#L593C17-L593C17
    with c.builder.if_else(cgutils.is_not_null(c.builder, index.parent)) as (
        has_parent,
        otherwise,
    ):
        with has_parent:
            c.pyapi.incref(index.parent)
        with otherwise:
            # TODO: preserve the original class for the index
            # Also need preserve the name of the Index
            # class_obj = c.pyapi.unserialize(c.pyapi.serialize_object(typ.pyclass))
            class_obj = c.pyapi.unserialize(c.pyapi.serialize_object(Index))
            array_obj = c.box(typ.as_array, index.data)
            if isinstance(typ.dtype, types.UnicodeCharSeq):
                # We converted to numpy string dtype, convert back
                # to object since _simple_new won't do that for uss
                object_str_obj = c.pyapi.unserialize(c.pyapi.serialize_object("object"))
                array_obj = c.pyapi.call_method(array_obj, "astype", (object_str_obj,))
                c.pyapi.decref(object_str_obj)
            # this is basically Index._simple_new(array_obj, name_obj) in python
            index_obj = c.pyapi.call_method(class_obj, "_simple_new", (array_obj,))
            index.parent = index_obj
            c.builder.store(index_obj, res)

            # Decrefs
            c.pyapi.decref(class_obj)
            c.pyapi.decref(array_obj)
    return c.builder.load(res)


@box(SeriesType)
def box_series(typ, val, c):
    """
    Convert a native series structure to a Series object.
    """
    series = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
    series_const_obj = c.pyapi.unserialize(c.pyapi.serialize_object(Series._from_mgr))
    mgr_const_obj = c.pyapi.unserialize(
        c.pyapi.serialize_object(SingleBlockManager.from_array)
    )
    index_obj = c.box(typ.index, series.index)
    array_obj = c.box(typ.as_array, series.values)
    name_obj = c.box(typ.namety, series.name)
    # This is basically equivalent of
    # pd.Series(data=array_obj, index=index_obj)
    # To improve perf, we will construct the Series from a manager
    # object to avoid checks.
    # We'll also set the name attribute manually to avoid validation
    mgr_obj = c.pyapi.call_function_objargs(
        mgr_const_obj,
        (
            array_obj,
            index_obj,
        ),
    )
    mgr_axes_obj = c.pyapi.object_getattr_string(mgr_obj, "axes")
    # Series._constructor_from_mgr(mgr, axes)
    series_obj = c.pyapi.call_function_objargs(
        series_const_obj, (mgr_obj, mgr_axes_obj)
    )
    c.pyapi.object_setattr_string(series_obj, "_name", name_obj)

    # Decrefs
    c.pyapi.decref(series_const_obj)
    c.pyapi.decref(mgr_axes_obj)
    c.pyapi.decref(mgr_obj)
    c.pyapi.decref(mgr_const_obj)
    c.pyapi.decref(index_obj)
    c.pyapi.decref(array_obj)
    c.pyapi.decref(name_obj)

    return series_obj


# Add common series reductions (e.g. mean, sum),
# and also add common binops (e.g. add, sub, mul, div)
def generate_series_reduction(ser_reduction, ser_method):
    @overload_method(SeriesType, ser_reduction)
    def series_reduction(series):
        def series_reduction_impl(series):
            return ser_method(series.values)

        return series_reduction_impl

    return series_reduction


def generate_series_binop(binop):
    @overload(binop)
    def series_binop(series1, value):
        if isinstance(series1, SeriesType):
            if isinstance(value, SeriesType):

                def series_binop_impl(series1, series2):
                    # TODO: Check index matching?
                    return Series(
                        binop(series1.values, series2.values),
                        series1.index,
                        series1.name,
                    )

                return series_binop_impl
            else:

                def series_binop_impl(series1, value):
                    return Series(
                        binop(series1.values, value), series1.index, series1.name
                    )

                return series_binop_impl

    return series_binop


series_reductions = [
    ("sum", np.sum),
    ("mean", np.mean),
    # Disabled due to discrepancies between numba std. dev
    # and pandas std. dev (no way to specify dof)
    # ("std", np.std),
    # ("var", np.var),
    ("min", np.min),
    ("max", np.max),
]
for reduction, reduction_method in series_reductions:
    generate_series_reduction(reduction, reduction_method)

series_binops = [operator.add, operator.sub, operator.mul, operator.truediv]

for ser_binop in series_binops:
    generate_series_binop(ser_binop)


# get_loc on Index
@overload_method(IndexType, "get_loc")
def index_get_loc(index, item):
    def index_get_loc_impl(index, item):
        # Initialize the hash table if not initialized
        if len(index.hashmap) == 0:
            for i, val in enumerate(index._data):
                index.hashmap[val] = i
        return index.hashmap[item]

    return index_get_loc_impl


# Indexing for Series/Index
@overload(operator.getitem)
def series_indexing(series, item):
    if isinstance(series, SeriesType):

        def series_getitem(series, item):
            loc = series.index.get_loc(item)
            return series.iloc[loc]

        return series_getitem


@overload(operator.getitem)
def index_indexing(index, idx):
    if isinstance(index, IndexType):

        def index_getitem(index, idx):
            return index._data[idx]

        return index_getitem


class IlocType(types.Type):
    def __init__(self, obj_type) -> None:
        self.obj_type = obj_type
        name = f"iLocIndexer({obj_type})"
        super().__init__(name=name)

    @property
    def key(self):
        return self.obj_type


@typeof_impl.register(_iLocIndexer)
def typeof_iloc(val, c):
    objtype = typeof_impl(val.obj, c)
    return IlocType(objtype)


@type_callable(_iLocIndexer)
def type_iloc_constructor(context):
    def typer(obj):
        if isinstance(obj, SeriesType):
            return IlocType(obj)

    return typer


@lower_builtin(_iLocIndexer, SeriesType)
def iloc_constructor(context, builder, sig, args):
    (obj,) = args
    iloc_indexer = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    iloc_indexer.obj = obj
    return impl_ret_borrowed(
        context, builder, sig.return_type, iloc_indexer._getvalue()
    )


@register_model(IlocType)
class ILocModel(models.StructModel):
    def __init__(self, dmm, fe_type) -> None:
        members = [("obj", fe_type.obj_type)]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(IlocType, "obj", "obj")


@overload_attribute(SeriesType, "iloc")
def series_iloc(series):
    def get(series):
        return _iLocIndexer(series)

    return get


@overload(operator.getitem)
def iloc_getitem(iloc_indexer, i):
    if isinstance(iloc_indexer, IlocType):

        def getitem_impl(iloc_indexer, i):
            return iloc_indexer.obj.values[i]

        return getitem_impl
<|endoftext|>from __future__ import annotations

import functools
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
)

if TYPE_CHECKING:
    from pandas._typing import Scalar

import numpy as np

from pandas.compat._optional import import_optional_dependency


@functools.cache
def generate_apply_looper(func, nopython=True, nogil=True, parallel=False):
    if TYPE_CHECKING:
        import numba
    else:
        numba = import_optional_dependency("numba")
    nb_compat_func = numba.extending.register_jitable(func)

    @numba.jit(nopython=nopython, nogil=nogil, parallel=parallel)
    def nb_looper(values, axis):
        # Operate on the first row/col in order to get
        # the output shape
        if axis == 0:
            first_elem = values[:, 0]
            dim0 = values.shape[1]
        else:
            first_elem = values[0]
            dim0 = values.shape[0]
        res0 = nb_compat_func(first_elem)
        # Use np.asarray to get shape for
        # https://github.com/numba/numba/issues/4202#issuecomment-1185981507
        buf_shape = (dim0,) + np.atleast_1d(np.asarray(res0)).shape
        if axis == 0:
            buf_shape = buf_shape[::-1]
        buff = np.empty(buf_shape)

        if axis == 1:
            buff[0] = res0
            for i in numba.prange(1, values.shape[0]):
                buff[i] = nb_compat_func(values[i])
        else:
            buff[:, 0] = res0
            for j in numba.prange(1, values.shape[1]):
                buff[:, j] = nb_compat_func(values[:, j])
        return buff

    return nb_looper


@functools.cache
def make_looper(func, result_dtype, is_grouped_kernel, nopython, nogil, parallel):
    if TYPE_CHECKING:
        import numba
    else:
        numba = import_optional_dependency("numba")

    if is_grouped_kernel:

        @numba.jit(nopython=nopython, nogil=nogil, parallel=parallel)
        def column_looper(
            values: np.ndarray,
            labels: np.ndarray,
            ngroups: int,
            min_periods: int,
            *args,
        ):
            result = np.empty((values.shape[0], ngroups), dtype=result_dtype)
            na_positions = {}
            for i in numba.prange(values.shape[0]):
                output, na_pos = func(
                    values[i], result_dtype, labels, ngroups, min_periods, *args
                )
                result[i] = output
                if len(na_pos) > 0:
                    na_positions[i] = np.array(na_pos)
            return result, na_positions

    else:

        @numba.jit(nopython=nopython, nogil=nogil, parallel=parallel)
        def column_looper(
            values: np.ndarray,
            start: np.ndarray,
            end: np.ndarray,
            min_periods: int,
            *args,
        ):
            result = np.empty((values.shape[0], len(start)), dtype=result_dtype)
            na_positions = {}
            for i in numba.prange(values.shape[0]):
                output, na_pos = func(
                    values[i], result_dtype, start, end, min_periods, *args
                )
                result[i] = output
                if len(na_pos) > 0:
                    na_positions[i] = np.array(na_pos)
            return result, na_positions

    return column_looper


default_dtype_mapping: dict[np.dtype, Any] = {
    np.dtype("int8"): np.int64,
    np.dtype("int16"): np.int64,
    np.dtype("int32"): np.int64,
    np.dtype("int64"): np.int64,
    np.dtype("uint8"): np.uint64,
    np.dtype("uint16"): np.uint64,
    np.dtype("uint32"): np.uint64,
    np.dtype("uint64"): np.uint64,
    np.dtype("float32"): np.float64,
    np.dtype("float64"): np.float64,
    np.dtype("complex64"): np.complex128,
    np.dtype("complex128"): np.complex128,
}


# TODO: Preserve complex dtypes

float_dtype_mapping: dict[np.dtype, Any] = {
    np.dtype("int8"): np.float64,
    np.dtype("int16"): np.float64,
    np.dtype("int32"): np.float64,
    np.dtype("int64"): np.float64,
    np.dtype("uint8"): np.float64,
    np.dtype("uint16"): np.float64,
    np.dtype("uint32"): np.float64,
    np.dtype("uint64"): np.float64,
    np.dtype("float32"): np.float64,
    np.dtype("float64"): np.float64,
    np.dtype("complex64"): np.float64,
    np.dtype("complex128"): np.float64,
}

identity_dtype_mapping: dict[np.dtype, Any] = {
    np.dtype("int8"): np.int8,
    np.dtype("int16"): np.int16,
    np.dtype("int32"): np.int32,
    np.dtype("int64"): np.int64,
    np.dtype("uint8"): np.uint8,
    np.dtype("uint16"): np.uint16,
    np.dtype("uint32"): np.uint32,
    np.dtype("uint64"): np.uint64,
    np.dtype("float32"): np.float32,
    np.dtype("float64"): np.float64,
    np.dtype("complex64"): np.complex64,
    np.dtype("complex128"): np.complex128,
}


def generate_shared_aggregator(
    func: Callable[..., Scalar],
    dtype_mapping: dict[np.dtype, np.dtype],
    is_grouped_kernel: bool,
    nopython: bool,
    nogil: bool,
    parallel: bool,
):
    """
    Generate a Numba function that loops over the columns 2D object and applies
    a 1D numba kernel over each column.

    Parameters
    ----------
    func : function
        aggregation function to be applied to each column
    dtype_mapping: dict or None
        If not None, maps a dtype to a result dtype.
        Otherwise, will fall back to default mapping.
    is_grouped_kernel: bool, default False
        Whether func operates using the group labels (True)
        or using starts/ends arrays

        If true, you also need to pass the number of groups to this function
    nopython : bool
        nopython to be passed into numba.jit
    nogil : bool
        nogil to be passed into numba.jit
    parallel : bool
        parallel to be passed into numba.jit

    Returns
    -------
    Numba function
    """

    # A wrapper around the looper function,
    # to dispatch based on dtype since numba is unable to do that in nopython mode

    # It also post-processes the values by inserting nans where number of observations
    # is less than min_periods
    # Cannot do this in numba nopython mode
    # (you'll run into type-unification error when you cast int -> float)
    def looper_wrapper(
        values,
        start=None,
        end=None,
        labels=None,
        ngroups=None,
        min_periods: int = 0,
        **kwargs,
    ):
        result_dtype = dtype_mapping[values.dtype]
        column_looper = make_looper(
            func, result_dtype, is_grouped_kernel, nopython, nogil, parallel
        )
        # Need to unpack kwargs since numba only supports *args
        if is_grouped_kernel:
            result, na_positions = column_looper(
                values, labels, ngroups, min_periods, *kwargs.values()
            )
        else:
            result, na_positions = column_looper(
                values, start, end, min_periods, *kwargs.values()
            )
        if result.dtype.kind == "i":
            # Look if na_positions is not empty
            # If so, convert the whole block
            # This is OK since int dtype cannot hold nan,
            # so if min_periods not satisfied for 1 col, it is not satisfied for
            # all columns at that index
            for na_pos in na_positions.values():
                if len(na_pos) > 0:
                    result = result.astype("float64")
                    break
        # TODO: Optimize this
        for i, na_pos in na_positions.items():
            if len(na_pos) > 0:
                result[i, na_pos] = np.nan
        return result

    return looper_wrapper
<|endoftext|>"""
Numba 1D mean kernels that can be shared by
* Dataframe / Series
* groupby
* rolling / expanding

Mirrors pandas/_libs/window/aggregation.pyx
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numba
import numpy as np

from pandas.core._numba.kernels.shared import is_monotonic_increasing
from pandas.core._numba.kernels.sum_ import grouped_kahan_sum

if TYPE_CHECKING:
    from pandas._typing import npt


@numba.jit(nopython=True, nogil=True, parallel=False)
def add_mean(
    val: float,
    nobs: int,
    sum_x: float,
    neg_ct: int,
    compensation: float,
    num_consecutive_same_value: int,
    prev_value: float,
) -> tuple[int, float, int, float, int, float]:
    if not np.isnan(val):
        nobs += 1
        y = val - compensation
        t = sum_x + y
        compensation = t - sum_x - y
        sum_x = t
        if val < 0:
            neg_ct += 1

        if val == prev_value:
            num_consecutive_same_value += 1
        else:
            num_consecutive_same_value = 1
        prev_value = val

    return nobs, sum_x, neg_ct, compensation, num_consecutive_same_value, prev_value


@numba.jit(nopython=True, nogil=True, parallel=False)
def remove_mean(
    val: float, nobs: int, sum_x: float, neg_ct: int, compensation: float
) -> tuple[int, float, int, float]:
    if not np.isnan(val):
        nobs -= 1
        y = -val - compensation
        t = sum_x + y
        compensation = t - sum_x - y
        sum_x = t
        if val < 0:
            neg_ct -= 1
    return nobs, sum_x, neg_ct, compensation


@numba.jit(nopython=True, nogil=True, parallel=False)
def sliding_mean(
    values: np.ndarray,
    result_dtype: np.dtype,
    start: np.ndarray,
    end: np.ndarray,
    min_periods: int,
) -> tuple[np.ndarray, list[int]]:
    N = len(start)
    nobs = 0
    sum_x = 0.0
    neg_ct = 0
    compensation_add = 0.0
    compensation_remove = 0.0

    is_monotonic_increasing_bounds = is_monotonic_increasing(
        start
    ) and is_monotonic_increasing(end)

    output = np.empty(N, dtype=result_dtype)

    for i in range(N):
        s = start[i]
        e = end[i]
        if i == 0 or not is_monotonic_increasing_bounds:
            prev_value = values[s]
            num_consecutive_same_value = 0

            for j in range(s, e):
                val = values[j]
                (
                    nobs,
                    sum_x,
                    neg_ct,
                    compensation_add,
                    num_consecutive_same_value,
                    prev_value,
                ) = add_mean(
                    val,
                    nobs,
                    sum_x,
                    neg_ct,
                    compensation_add,
                    num_consecutive_same_value,
                    prev_value,  # pyright: ignore[reportGeneralTypeIssues]
                )
        else:
            for j in range(start[i - 1], s):
                val = values[j]
                nobs, sum_x, neg_ct, compensation_remove = remove_mean(
                    val, nobs, sum_x, neg_ct, compensation_remove
                )

            for j in range(end[i - 1], e):
                val = values[j]
                (
                    nobs,
                    sum_x,
                    neg_ct,
                    compensation_add,
                    num_consecutive_same_value,
                    prev_value,
                ) = add_mean(
                    val,
                    nobs,
                    sum_x,
                    neg_ct,
                    compensation_add,
                    num_consecutive_same_value,
                    prev_value,  # pyright: ignore[reportGeneralTypeIssues]
                )

        if nobs >= min_periods and nobs > 0:
            result = sum_x / nobs
            if num_consecutive_same_value >= nobs:
                result = prev_value
            elif neg_ct == 0 and result < 0:
                result = 0
            elif neg_ct == nobs and result > 0:
                result = 0
        else:
            result = np.nan

        output[i] = result

        if not is_monotonic_increasing_bounds:
            nobs = 0
            sum_x = 0.0
            neg_ct = 0
            compensation_remove = 0.0

    # na_position is empty list since float64 can already hold nans
    # Do list comprehension, since numba cannot figure out that na_pos is
    # empty list of ints on its own
    na_pos = [0 for i in range(0)]
    return output, na_pos


@numba.jit(nopython=True, nogil=True, parallel=False)
def grouped_mean(
    values: np.ndarray,
    result_dtype: np.dtype,
    labels: npt.NDArray[np.intp],
    ngroups: int,
    min_periods: int,
) -> tuple[np.ndarray, list[int]]:
    output, nobs_arr, comp_arr, consecutive_counts, prev_vals = grouped_kahan_sum(
        values, result_dtype, labels, ngroups
    )

    # Post-processing, replace sums that don't satisfy min_periods
    for lab in range(ngroups):
        nobs = nobs_arr[lab]
        num_consecutive_same_value = consecutive_counts[lab]
        prev_value = prev_vals[lab]
        sum_x = output[lab]
        if nobs >= min_periods:
            if num_consecutive_same_value >= nobs:
                result = prev_value * nobs
            else:
                result = sum_x
        else:
            result = np.nan
        result /= nobs
        output[lab] = result

    # na_position is empty list since float64 can already hold nans
    # Do list comprehension, since numba cannot figure out that na_pos is
    # empty list of ints on its own
    na_pos = [0 for i in range(0)]
    return output, na_pos
<|endoftext|>from __future__ import annotations

from typing import TYPE_CHECKING

import numba

if TYPE_CHECKING:
    import numpy as np


@numba.jit(
    # error: Any? not callable
    numba.boolean(numba.int64[:]),  # type: ignore[misc]
    nopython=True,
    nogil=True,
    parallel=False,
)
def is_monotonic_increasing(bounds: np.ndarray) -> bool:
    """Check if int64 values are monotonically increasing."""
    n = len(bounds)
    if n < 2:
        return True
    prev = bounds[0]
    for i in range(1, n):
        cur = bounds[i]
        if cur < prev:
            return False
        prev = cur
    return True
<|endoftext|>"""
Numba 1D var kernels that can be shared by
* Dataframe / Series
* groupby
* rolling / expanding

Mirrors pandas/_libs/window/aggregation.pyx
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numba
import numpy as np

if TYPE_CHECKING:
    from pandas._typing import npt

from pandas.core._numba.kernels.shared import is_monotonic_increasing


@numba.jit(nopython=True, nogil=True, parallel=False)
def add_var(
    val: float,
    nobs: int,
    mean_x: float,
    ssqdm_x: float,
    compensation: float,
    num_consecutive_same_value: int,
    prev_value: float,
) -> tuple[int, float, float, float, int, float]:
    if not np.isnan(val):
        if val == prev_value:
            num_consecutive_same_value += 1
        else:
            num_consecutive_same_value = 1
        prev_value = val

        nobs += 1
        prev_mean = mean_x - compensation
        y = val - compensation
        t = y - mean_x
        compensation = t + mean_x - y
        delta = t
        if nobs:
            mean_x += delta / nobs
        else:
            mean_x = 0
        ssqdm_x += (val - prev_mean) * (val - mean_x)
    return nobs, mean_x, ssqdm_x, compensation, num_consecutive_same_value, prev_value


@numba.jit(nopython=True, nogil=True, parallel=False)
def remove_var(
    val: float, nobs: int, mean_x: float, ssqdm_x: float, compensation: float
) -> tuple[int, float, float, float]:
    if not np.isnan(val):
        nobs -= 1
        if nobs:
            prev_mean = mean_x - compensation
            y = val - compensation
            t = y - mean_x
            compensation = t + mean_x - y
            delta = t
            mean_x -= delta / nobs
            ssqdm_x -= (val - prev_mean) * (val - mean_x)
        else:
            mean_x = 0
            ssqdm_x = 0
    return nobs, mean_x, ssqdm_x, compensation


@numba.jit(nopython=True, nogil=True, parallel=False)
def sliding_var(
    values: np.ndarray,
    result_dtype: np.dtype,
    start: np.ndarray,
    end: np.ndarray,
    min_periods: int,
    ddof: int = 1,
) -> tuple[np.ndarray, list[int]]:
    N = len(start)
    nobs = 0
    mean_x = 0.0
    ssqdm_x = 0.0
    compensation_add = 0.0
    compensation_remove = 0.0

    min_periods = max(min_periods, 1)
    is_monotonic_increasing_bounds = is_monotonic_increasing(
        start
    ) and is_monotonic_increasing(end)

    output = np.empty(N, dtype=result_dtype)

    for i in range(N):
        s = start[i]
        e = end[i]
        if i == 0 or not is_monotonic_increasing_bounds:
            prev_value = values[s]
            num_consecutive_same_value = 0

            for j in range(s, e):
                val = values[j]
                (
                    nobs,
                    mean_x,
                    ssqdm_x,
                    compensation_add,
                    num_consecutive_same_value,
                    prev_value,
                ) = add_var(
                    val,
                    nobs,
                    mean_x,
                    ssqdm_x,
                    compensation_add,
                    num_consecutive_same_value,
                    prev_value,
                )
        else:
            for j in range(start[i - 1], s):
                val = values[j]
                nobs, mean_x, ssqdm_x, compensation_remove = remove_var(
                    val, nobs, mean_x, ssqdm_x, compensation_remove
                )

            for j in range(end[i - 1], e):
                val = values[j]
                (
                    nobs,
                    mean_x,
                    ssqdm_x,
                    compensation_add,
                    num_consecutive_same_value,
                    prev_value,
                ) = add_var(
                    val,
                    nobs,
                    mean_x,
                    ssqdm_x,
                    compensation_add,
                    num_consecutive_same_value,
                    prev_value,
                )

        if nobs >= min_periods and nobs > ddof:
            if nobs == 1 or num_consecutive_same_value >= nobs:
                result = 0.0
            else:
                result = ssqdm_x / (nobs - ddof)
        else:
            result = np.nan

        output[i] = result

        if not is_monotonic_increasing_bounds:
            nobs = 0
            mean_x = 0.0
            ssqdm_x = 0.0
            compensation_remove = 0.0

    # na_position is empty list since float64 can already hold nans
    # Do list comprehension, since numba cannot figure out that na_pos is
    # empty list of ints on its own
    na_pos = [0 for i in range(0)]
    return output, na_pos


@numba.jit(nopython=True, nogil=True, parallel=False)
def grouped_var(
    values: np.ndarray,
    result_dtype: np.dtype,
    labels: npt.NDArray[np.intp],
    ngroups: int,
    min_periods: int,
    ddof: int = 1,
) -> tuple[np.ndarray, list[int]]:
    N = len(labels)

    nobs_arr = np.zeros(ngroups, dtype=np.int64)
    comp_arr = np.zeros(ngroups, dtype=values.dtype)
    consecutive_counts = np.zeros(ngroups, dtype=np.int64)
    prev_vals = np.zeros(ngroups, dtype=values.dtype)
    output = np.zeros(ngroups, dtype=result_dtype)
    means = np.zeros(ngroups, dtype=result_dtype)

    for i in range(N):
        lab = labels[i]
        val = values[i]

        if lab < 0:
            continue

        mean_x = means[lab]
        ssqdm_x = output[lab]
        nobs = nobs_arr[lab]
        compensation_add = comp_arr[lab]
        num_consecutive_same_value = consecutive_counts[lab]
        prev_value = prev_vals[lab]

        (
            nobs,
            mean_x,
            ssqdm_x,
            compensation_add,
            num_consecutive_same_value,
            prev_value,
        ) = add_var(
            val,
            nobs,
            mean_x,
            ssqdm_x,
            compensation_add,
            num_consecutive_same_value,
            prev_value,
        )

        output[lab] = ssqdm_x
        means[lab] = mean_x
        consecutive_counts[lab] = num_consecutive_same_value
        prev_vals[lab] = prev_value
        comp_arr[lab] = compensation_add
        nobs_arr[lab] = nobs

    # Post-processing, replace vars that don't satisfy min_periods
    for lab in range(ngroups):
        nobs = nobs_arr[lab]
        num_consecutive_same_value = consecutive_counts[lab]
        ssqdm_x = output[lab]
        if nobs >= min_periods and nobs > ddof:
            if nobs == 1 or num_consecutive_same_value >= nobs:
                result = 0.0
            else:
                result = ssqdm_x / (nobs - ddof)
        else:
            result = np.nan
        output[lab] = result

    # Second pass to get the std.dev
    # na_position is empty list since float64 can already hold nans
    # Do list comprehension, since numba cannot figure out that na_pos is
    # empty list of ints on its own
    na_pos = [0 for i in range(0)]
    return output, na_pos
<|endoftext|>"""
Numba 1D min/max kernels that can be shared by
* Dataframe / Series
* groupby
* rolling / expanding

Mirrors pandas/_libs/window/aggregation.pyx
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numba
import numpy as np

if TYPE_CHECKING:
    from pandas._typing import npt


@numba.jit(nopython=True, nogil=True, parallel=False)
def sliding_min_max(
    values: np.ndarray,
    result_dtype: np.dtype,
    start: np.ndarray,
    end: np.ndarray,
    min_periods: int,
    is_max: bool,
) -> tuple[np.ndarray, list[int]]:
    N = len(start)
    nobs = 0
    output = np.empty(N, dtype=result_dtype)
    na_pos = []
    # Use deque once numba supports it
    # https://github.com/numba/numba/issues/7417
    Q: list = []
    W: list = []
    for i in range(N):
        curr_win_size = end[i] - start[i]
        if i == 0:
            st = start[i]
        else:
            st = end[i - 1]

        for k in range(st, end[i]):
            ai = values[k]
            if not np.isnan(ai):
                nobs += 1
            elif is_max:
                ai = -np.inf
            else:
                ai = np.inf
            # Discard previous entries if we find new min or max
            if is_max:
                while Q and ((ai >= values[Q[-1]]) or values[Q[-1]] != values[Q[-1]]):
                    Q.pop()
            else:
                while Q and ((ai <= values[Q[-1]]) or values[Q[-1]] != values[Q[-1]]):
                    Q.pop()
            Q.append(k)
            W.append(k)

        # Discard entries outside and left of current window
        while Q and Q[0] <= start[i] - 1:
            Q.pop(0)
        while W and W[0] <= start[i] - 1:
            if not np.isnan(values[W[0]]):
                nobs -= 1
            W.pop(0)

        # Save output based on index in input value array
        if Q and curr_win_size > 0 and nobs >= min_periods:
            output[i] = values[Q[0]]
        else:
            if values.dtype.kind != "i":
                output[i] = np.nan
            else:
                na_pos.append(i)

    return output, na_pos


@numba.jit(nopython=True, nogil=True, parallel=False)
def grouped_min_max(
    values: np.ndarray,
    result_dtype: np.dtype,
    labels: npt.NDArray[np.intp],
    ngroups: int,
    min_periods: int,
    is_max: bool,
) -> tuple[np.ndarray, list[int]]:
    N = len(labels)
    nobs = np.zeros(ngroups, dtype=np.int64)
    na_pos = []
    output = np.empty(ngroups, dtype=result_dtype)

    for i in range(N):
        lab = labels[i]
        val = values[i]
        if lab < 0:
            continue

        if values.dtype.kind == "i" or not np.isnan(val):
            nobs[lab] += 1
        else:
            # NaN value cannot be a min/max value
            continue

        if nobs[lab] == 1:
            # First element in group, set output equal to this
            output[lab] = val
            continue

        if is_max:
            if val > output[lab]:
                output[lab] = val
        else:
            if val < output[lab]:
                output[lab] = val

    # Set labels that don't satisfy min_periods as np.nan
    for lab, count in enumerate(nobs):
        if count < min_periods:
            na_pos.append(lab)

    return output, na_pos
<|endoftext|>from pandas.core._numba.kernels.mean_ import (
    grouped_mean,
    sliding_mean,
)
from pandas.core._numba.kernels.min_max_ import (
    grouped_min_max,
    sliding_min_max,
)
from pandas.core._numba.kernels.sum_ import (
    grouped_sum,
    sliding_sum,
)
from pandas.core._numba.kernels.var_ import (
    grouped_var,
    sliding_var,
)

__all__ = [
    "sliding_mean",
    "grouped_mean",
    "sliding_sum",
    "grouped_sum",
    "sliding_var",
    "grouped_var",
    "sliding_min_max",
    "grouped_min_max",
]
<|endoftext|>"""
Numba 1D sum kernels that can be shared by
* Dataframe / Series
* groupby
* rolling / expanding

Mirrors pandas/_libs/window/aggregation.pyx
"""
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)

import numba
from numba.extending import register_jitable
import numpy as np

if TYPE_CHECKING:
    from pandas._typing import npt

from pandas.core._numba.kernels.shared import is_monotonic_increasing


@numba.jit(nopython=True, nogil=True, parallel=False)
def add_sum(
    val: Any,
    nobs: int,
    sum_x: Any,
    compensation: Any,
    num_consecutive_same_value: int,
    prev_value: Any,
) -> tuple[int, Any, Any, int, Any]:
    if not np.isnan(val):
        nobs += 1
        y = val - compensation
        t = sum_x + y
        compensation = t - sum_x - y
        sum_x = t

        if val == prev_value:
            num_consecutive_same_value += 1
        else:
            num_consecutive_same_value = 1
        prev_value = val

    return nobs, sum_x, compensation, num_consecutive_same_value, prev_value


@numba.jit(nopython=True, nogil=True, parallel=False)
def remove_sum(
    val: Any, nobs: int, sum_x: Any, compensation: Any
) -> tuple[int, Any, Any]:
    if not np.isnan(val):
        nobs -= 1
        y = -val - compensation
        t = sum_x + y
        compensation = t - sum_x - y
        sum_x = t
    return nobs, sum_x, compensation


@numba.jit(nopython=True, nogil=True, parallel=False)
def sliding_sum(
    values: np.ndarray,
    result_dtype: np.dtype,
    start: np.ndarray,
    end: np.ndarray,
    min_periods: int,
) -> tuple[np.ndarray, list[int]]:
    dtype = values.dtype

    na_val: object = np.nan
    if dtype.kind == "i":
        na_val = 0

    N = len(start)
    nobs = 0
    sum_x = 0
    compensation_add = 0
    compensation_remove = 0
    na_pos = []

    is_monotonic_increasing_bounds = is_monotonic_increasing(
        start
    ) and is_monotonic_increasing(end)

    output = np.empty(N, dtype=result_dtype)

    for i in range(N):
        s = start[i]
        e = end[i]
        if i == 0 or not is_monotonic_increasing_bounds:
            prev_value = values[s]
            num_consecutive_same_value = 0

            for j in range(s, e):
                val = values[j]
                (
                    nobs,
                    sum_x,
                    compensation_add,
                    num_consecutive_same_value,
                    prev_value,
                ) = add_sum(
                    val,
                    nobs,
                    sum_x,
                    compensation_add,
                    num_consecutive_same_value,
                    prev_value,
                )
        else:
            for j in range(start[i - 1], s):
                val = values[j]
                nobs, sum_x, compensation_remove = remove_sum(
                    val, nobs, sum_x, compensation_remove
                )

            for j in range(end[i - 1], e):
                val = values[j]
                (
                    nobs,
                    sum_x,
                    compensation_add,
                    num_consecutive_same_value,
                    prev_value,
                ) = add_sum(
                    val,
                    nobs,
                    sum_x,
                    compensation_add,
                    num_consecutive_same_value,
                    prev_value,
                )

        if nobs == 0 == min_periods:
            result: object = 0
        elif nobs >= min_periods:
            if num_consecutive_same_value >= nobs:
                result = prev_value * nobs
            else:
                result = sum_x
        else:
            result = na_val
            if dtype.kind == "i":
                na_pos.append(i)

        output[i] = result

        if not is_monotonic_increasing_bounds:
            nobs = 0
            sum_x = 0
            compensation_remove = 0

    return output, na_pos


# Mypy/pyright don't like the fact that the decorator is untyped
@register_jitable  # type: ignore[misc]
def grouped_kahan_sum(
    values: np.ndarray,
    result_dtype: np.dtype,
    labels: npt.NDArray[np.intp],
    ngroups: int,
) -> tuple[
    np.ndarray, npt.NDArray[np.int64], np.ndarray, npt.NDArray[np.int64], np.ndarray
]:
    N = len(labels)

    nobs_arr = np.zeros(ngroups, dtype=np.int64)
    comp_arr = np.zeros(ngroups, dtype=values.dtype)
    consecutive_counts = np.zeros(ngroups, dtype=np.int64)
    prev_vals = np.zeros(ngroups, dtype=values.dtype)
    output = np.zeros(ngroups, dtype=result_dtype)

    for i in range(N):
        lab = labels[i]
        val = values[i]

        if lab < 0:
            continue

        sum_x = output[lab]
        nobs = nobs_arr[lab]
        compensation_add = comp_arr[lab]
        num_consecutive_same_value = consecutive_counts[lab]
        prev_value = prev_vals[lab]

        (
            nobs,
            sum_x,
            compensation_add,
            num_consecutive_same_value,
            prev_value,
        ) = add_sum(
            val,
            nobs,
            sum_x,
            compensation_add,
            num_consecutive_same_value,
            prev_value,
        )

        output[lab] = sum_x
        consecutive_counts[lab] = num_consecutive_same_value
        prev_vals[lab] = prev_value
        comp_arr[lab] = compensation_add
        nobs_arr[lab] = nobs
    return output, nobs_arr, comp_arr, consecutive_counts, prev_vals


@numba.jit(nopython=True, nogil=True, parallel=False)
def grouped_sum(
    values: np.ndarray,
    result_dtype: np.dtype,
    labels: npt.NDArray[np.intp],
    ngroups: int,
    min_periods: int,
) -> tuple[np.ndarray, list[int]]:
    na_pos = []

    output, nobs_arr, comp_arr, consecutive_counts, prev_vals = grouped_kahan_sum(
        values, result_dtype, labels, ngroups
    )

    # Post-processing, replace sums that don't satisfy min_periods
    for lab in range(ngroups):
        nobs = nobs_arr[lab]
        num_consecutive_same_value = consecutive_counts[lab]
        prev_value = prev_vals[lab]
        sum_x = output[lab]
        if nobs >= min_periods:
            if num_consecutive_same_value >= nobs:
                result = prev_value * nobs
            else:
                result = sum_x
        else:
            result = sum_x  # Don't change val, will be replaced by nan later
            na_pos.append(lab)
        output[lab] = result

    return output, na_pos
<|endoftext|>from __future__ import annotations

from datetime import (
    datetime,
    time,
)
from typing import TYPE_CHECKING
import warnings

import numpy as np

from pandas._libs.lib import is_list_like
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.generic import (
    ABCIndex,
    ABCSeries,
)
from pandas.core.dtypes.missing import notna

if TYPE_CHECKING:
    from pandas._typing import DateTimeErrorChoices


def to_time(
    arg,
    format: str | None = None,
    infer_time_format: bool = False,
    errors: DateTimeErrorChoices = "raise",
):
    """
    Parse time strings to time objects using fixed strptime formats ("%H:%M",
    "%H%M", "%I:%M%p", "%I%M%p", "%H:%M:%S", "%H%M%S", "%I:%M:%S%p",
    "%I%M%S%p")

    Use infer_time_format if all the strings are in the same format to speed
    up conversion.

    Parameters
    ----------
    arg : string in time format, datetime.time, list, tuple, 1-d array,  Series
    format : str, default None
        Format used to convert arg into a time object.  If None, fixed formats
        are used.
    infer_time_format: bool, default False
        Infer the time format based on the first non-NaN element.  If all
        strings are in the same format, this will speed up conversion.
    errors : {'ignore', 'raise', 'coerce'}, default 'raise'
        - If 'raise', then invalid parsing will raise an exception
        - If 'coerce', then invalid parsing will be set as None
        - If 'ignore', then invalid parsing will return the input

    Returns
    -------
    datetime.time
    """
    if errors == "ignore":
        # GH#54467
        warnings.warn(
            "errors='ignore' is deprecated and will raise in a future version. "
            "Use to_time without passing `errors` and catch exceptions "
            "explicitly instead",
            FutureWarning,
            stacklevel=find_stack_level(),
        )

    def _convert_listlike(arg, format):
        if isinstance(arg, (list, tuple)):
            arg = np.array(arg, dtype="O")

        elif getattr(arg, "ndim", 1) > 1:
            raise TypeError(
                "arg must be a string, datetime, list, tuple, 1-d array, or Series"
            )

        arg = np.asarray(arg, dtype="O")

        if infer_time_format and format is None:
            format = _guess_time_format_for_array(arg)

        times: list[time | None] = []
        if format is not None:
            for element in arg:
                try:
                    times.append(datetime.strptime(element, format).time())
                except (ValueError, TypeError) as err:
                    if errors == "raise":
                        msg = (
                            f"Cannot convert {element} to a time with given "
                            f"format {format}"
                        )
                        raise ValueError(msg) from err
                    if errors == "ignore":
                        return arg
                    else:
                        times.append(None)
        else:
            formats = _time_formats[:]
            format_found = False
            for element in arg:
                time_object = None
                try:
                    time_object = time.fromisoformat(element)
                except (ValueError, TypeError):
                    for time_format in formats:
                        try:
                            time_object = datetime.strptime(element, time_format).time()
                            if not format_found:
                                # Put the found format in front
                                fmt = formats.pop(formats.index(time_format))
                                formats.insert(0, fmt)
                                format_found = True
                            break
                        except (ValueError, TypeError):
                            continue

                if time_object is not None:
                    times.append(time_object)
                elif errors == "raise":
                    raise ValueError(f"Cannot convert arg {arg} to a time")
                elif errors == "ignore":
                    return arg
                else:
                    times.append(None)

        return times

    if arg is None:
        return arg
    elif isinstance(arg, time):
        return arg
    elif isinstance(arg, ABCSeries):
        values = _convert_listlike(arg._values, format)
        return arg._constructor(values, index=arg.index, name=arg.name)
    elif isinstance(arg, ABCIndex):
        return _convert_listlike(arg, format)
    elif is_list_like(arg):
        return _convert_listlike(arg, format)

    return _convert_listlike(np.array([arg]), format)[0]


# Fixed time formats for time parsing
_time_formats = [
    "%H:%M",
    "%H%M",
    "%I:%M%p",
    "%I%M%p",
    "%H:%M:%S",
    "%H%M%S",
    "%I:%M:%S%p",
    "%I%M%S%p",
]


def _guess_time_format_for_array(arr):
    # Try to guess the format based on the first non-NaN element
    non_nan_elements = notna(arr).nonzero()[0]
    if len(non_nan_elements):
        element = arr[non_nan_elements[0]]
        for time_format in _time_formats:
            try:
                datetime.strptime(element, time_format)
                return time_format
            except ValueError:
                pass

    return None
<|endoftext|>from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Literal,
)
import warnings

import numpy as np

from pandas._libs import (
    lib,
    missing as libmissing,
)
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend

from pandas.core.dtypes.cast import maybe_downcast_numeric
from pandas.core.dtypes.common import (
    ensure_object,
    is_bool_dtype,
    is_decimal,
    is_integer_dtype,
    is_number,
    is_numeric_dtype,
    is_scalar,
    is_string_dtype,
    needs_i8_conversion,
)
from pandas.core.dtypes.dtypes import ArrowDtype
from pandas.core.dtypes.generic import (
    ABCIndex,
    ABCSeries,
)

from pandas.core.arrays import BaseMaskedArray
from pandas.core.arrays.string_ import StringDtype

if TYPE_CHECKING:
    from pandas._typing import (
        DateTimeErrorChoices,
        DtypeBackend,
        npt,
    )


def to_numeric(
    arg,
    errors: DateTimeErrorChoices = "raise",
    downcast: Literal["integer", "signed", "unsigned", "float"] | None = None,
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
):
    """
    Convert argument to a numeric type.

    The default return dtype is `float64` or `int64`
    depending on the data supplied. Use the `downcast` parameter
    to obtain other dtypes.

    Please note that precision loss may occur if really large numbers
    are passed in. Due to the internal limitations of `ndarray`, if
    numbers smaller than `-9223372036854775808` (np.iinfo(np.int64).min)
    or larger than `18446744073709551615` (np.iinfo(np.uint64).max) are
    passed in, it is very likely they will be converted to float so that
    they can be stored in an `ndarray`. These warnings apply similarly to
    `Series` since it internally leverages `ndarray`.

    Parameters
    ----------
    arg : scalar, list, tuple, 1-d array, or Series
        Argument to be converted.
    errors : {'ignore', 'raise', 'coerce'}, default 'raise'
        - If 'raise', then invalid parsing will raise an exception.
        - If 'coerce', then invalid parsing will be set as NaN.
        - If 'ignore', then invalid parsing will return the input.

        .. versionchanged:: 2.2

        "ignore" is deprecated. Catch exceptions explicitly instead.

    downcast : str, default None
        Can be 'integer', 'signed', 'unsigned', or 'float'.
        If not None, and if the data has been successfully cast to a
        numerical dtype (or if the data was numeric to begin with),
        downcast that resulting data to the smallest numerical dtype
        possible according to the following rules:

        - 'integer' or 'signed': smallest signed int dtype (min.: np.int8)
        - 'unsigned': smallest unsigned int dtype (min.: np.uint8)
        - 'float': smallest float dtype (min.: np.float32)

        As this behaviour is separate from the core conversion to
        numeric values, any errors raised during the downcasting
        will be surfaced regardless of the value of the 'errors' input.

        In addition, downcasting will only occur if the size
        of the resulting data's dtype is strictly larger than
        the dtype it is to be cast to, so if none of the dtypes
        checked satisfy that specification, no downcasting will be
        performed on the data.
    dtype_backend : {'numpy_nullable', 'pyarrow'}, default 'numpy_nullable'
        Back-end data type applied to the resultant :class:`DataFrame`
        (still experimental). Behaviour is as follows:

        * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
          (default).
        * ``"pyarrow"``: returns pyarrow-backed nullable :class:`ArrowDtype`
          DataFrame.

        .. versionadded:: 2.0

    Returns
    -------
    ret
        Numeric if parsing succeeded.
        Return type depends on input.  Series if Series, otherwise ndarray.

    See Also
    --------
    DataFrame.astype : Cast argument to a specified dtype.
    to_datetime : Convert argument to datetime.
    to_timedelta : Convert argument to timedelta.
    numpy.ndarray.astype : Cast a numpy array to a specified type.
    DataFrame.convert_dtypes : Convert dtypes.

    Examples
    --------
    Take separate series and convert to numeric, coercing when told to

    >>> s = pd.Series(['1.0', '2', -3])
    >>> pd.to_numeric(s)
    0    1.0
    1    2.0
    2   -3.0
    dtype: float64
    >>> pd.to_numeric(s, downcast='float')
    0    1.0
    1    2.0
    2   -3.0
    dtype: float32
    >>> pd.to_numeric(s, downcast='signed')
    0    1
    1    2
    2   -3
    dtype: int8
    >>> s = pd.Series(['apple', '1.0', '2', -3])
    >>> pd.to_numeric(s, errors='coerce')
    0    NaN
    1    1.0
    2    2.0
    3   -3.0
    dtype: float64

    Downcasting of nullable integer and floating dtypes is supported:

    >>> s = pd.Series([1, 2, 3], dtype="Int64")
    >>> pd.to_numeric(s, downcast="integer")
    0    1
    1    2
    2    3
    dtype: Int8
    >>> s = pd.Series([1.0, 2.1, 3.0], dtype="Float64")
    >>> pd.to_numeric(s, downcast="float")
    0    1.0
    1    2.1
    2    3.0
    dtype: Float32
    """
    if downcast not in (None, "integer", "signed", "unsigned", "float"):
        raise ValueError("invalid downcasting method provided")

    if errors not in ("ignore", "raise", "coerce"):
        raise ValueError("invalid error value specified")
    if errors == "ignore":
        # GH#54467
        warnings.warn(
            "errors='ignore' is deprecated and will raise in a future version. "
            "Use to_numeric without passing `errors` and catch exceptions "
            "explicitly instead",
            FutureWarning,
            stacklevel=find_stack_level(),
        )

    check_dtype_backend(dtype_backend)

    is_series = False
    is_index = False
    is_scalars = False

    if isinstance(arg, ABCSeries):
        is_series = True
        values = arg.values
    elif isinstance(arg, ABCIndex):
        is_index = True
        if needs_i8_conversion(arg.dtype):
            values = arg.view("i8")
        else:
            values = arg.values
    elif isinstance(arg, (list, tuple)):
        values = np.array(arg, dtype="O")
    elif is_scalar(arg):
        if is_decimal(arg):
            return float(arg)
        if is_number(arg):
            return arg
        is_scalars = True
        values = np.array([arg], dtype="O")
    elif getattr(arg, "ndim", 1) > 1:
        raise TypeError("arg must be a list, tuple, 1-d array, or Series")
    else:
        values = arg

    orig_values = values

    # GH33013: for IntegerArray & FloatingArray extract non-null values for casting
    # save mask to reconstruct the full array after casting
    mask: npt.NDArray[np.bool_] | None = None
    if isinstance(values, BaseMaskedArray):
        mask = values._mask
        values = values._data[~mask]

    values_dtype = getattr(values, "dtype", None)
    if isinstance(values_dtype, ArrowDtype):
        mask = values.isna()
        values = values.dropna().to_numpy()
    new_mask: np.ndarray | None = None
    if is_numeric_dtype(values_dtype):
        pass
    elif lib.is_np_dtype(values_dtype, "mM"):
        values = values.view(np.int64)
    else:
        values = ensure_object(values)
        coerce_numeric = errors not in ("ignore", "raise")
        try:
            values, new_mask = lib.maybe_convert_numeric(  # type: ignore[call-overload]
                values,
                set(),
                coerce_numeric=coerce_numeric,
                convert_to_masked_nullable=dtype_backend is not lib.no_default
                or isinstance(values_dtype, StringDtype)
                and values_dtype.na_value is libmissing.NA,
            )
        except (ValueError, TypeError):
            if errors == "raise":
                raise
            values = orig_values

    if new_mask is not None:
        # Remove unnecessary values, is expected later anyway and enables
        # downcasting
        values = values[~new_mask]
    elif (
        dtype_backend is not lib.no_default
        and new_mask is None
        or isinstance(values_dtype, StringDtype)
        and values_dtype.na_value is libmissing.NA
    ):
        new_mask = np.zeros(values.shape, dtype=np.bool_)

    # attempt downcast only if the data has been successfully converted
    # to a numerical dtype and if a downcast method has been specified
    if downcast is not None and is_numeric_dtype(values.dtype):
        typecodes: str | None = None

        if downcast in ("integer", "signed"):
            typecodes = np.typecodes["Integer"]
        elif downcast == "unsigned" and (not len(values) or np.min(values) >= 0):
            typecodes = np.typecodes["UnsignedInteger"]
        elif downcast == "float":
            typecodes = np.typecodes["Float"]

            # pandas support goes only to np.float32,
            # as float dtypes smaller than that are
            # extremely rare and not well supported
            float_32_char = np.dtype(np.float32).char
            float_32_ind = typecodes.index(float_32_char)
            typecodes = typecodes[float_32_ind:]

        if typecodes is not None:
            # from smallest to largest
            for typecode in typecodes:
                dtype = np.dtype(typecode)
                if dtype.itemsize <= values.dtype.itemsize:
                    values = maybe_downcast_numeric(values, dtype)

                    # successful conversion
                    if values.dtype == dtype:
                        break

    # GH33013: for IntegerArray, BooleanArray & FloatingArray need to reconstruct
    # masked array
    if (mask is not None or new_mask is not None) and not is_string_dtype(values.dtype):
        if mask is None or (new_mask is not None and new_mask.shape == mask.shape):
            # GH 52588
            mask = new_mask
        else:
            mask = mask.copy()
        assert isinstance(mask, np.ndarray)
        data = np.zeros(mask.shape, dtype=values.dtype)
        data[~mask] = values

        from pandas.core.arrays import (
            ArrowExtensionArray,
            BooleanArray,
            FloatingArray,
            IntegerArray,
        )

        klass: type[IntegerArray | BooleanArray | FloatingArray]
        if is_integer_dtype(data.dtype):
            klass = IntegerArray
        elif is_bool_dtype(data.dtype):
            klass = BooleanArray
        else:
            klass = FloatingArray
        values = klass(data, mask)

        if dtype_backend == "pyarrow" or isinstance(values_dtype, ArrowDtype):
            values = ArrowExtensionArray(values.__arrow_array__())

    if is_series:
        return arg._constructor(values, index=arg.index, name=arg.name)
    elif is_index:
        # because we want to coerce to numeric if possible,
        # do not use _shallow_copy
        from pandas import Index

        return Index(values, name=arg.name)
    elif is_scalars:
        return values[0]
    else:
        return values
<|endoftext|>from __future__ import annotations

from collections import abc
from datetime import date
from functools import partial
from itertools import islice
from typing import (
    TYPE_CHECKING,
    Callable,
    TypedDict,
    Union,
    cast,
    overload,
)
import warnings

import numpy as np

from pandas._config import using_string_dtype

from pandas._libs import (
    lib,
    tslib,
)
from pandas._libs.tslibs import (
    OutOfBoundsDatetime,
    Timedelta,
    Timestamp,
    astype_overflowsafe,
    is_supported_dtype,
    timezones as libtimezones,
)
from pandas._libs.tslibs.conversion import cast_from_unit_vectorized
from pandas._libs.tslibs.parsing import (
    DateParseError,
    guess_datetime_format,
)
from pandas._libs.tslibs.strptime import array_strptime
from pandas._typing import (
    AnyArrayLike,
    ArrayLike,
    DateTimeErrorChoices,
)
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import (
    ensure_object,
    is_float,
    is_integer,
    is_integer_dtype,
    is_list_like,
    is_numeric_dtype,
)
from pandas.core.dtypes.dtypes import (
    ArrowDtype,
    DatetimeTZDtype,
)
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCSeries,
)

from pandas.arrays import (
    DatetimeArray,
    IntegerArray,
    NumpyExtensionArray,
)
from pandas.core.algorithms import unique
from pandas.core.arrays import ArrowExtensionArray
from pandas.core.arrays.base import ExtensionArray
from pandas.core.arrays.datetimes import (
    maybe_convert_dtype,
    objects_to_datetime64,
    tz_to_dtype,
)
from pandas.core.construction import extract_array
from pandas.core.indexes.base import Index
from pandas.core.indexes.datetimes import DatetimeIndex

if TYPE_CHECKING:
    from collections.abc import Hashable

    from pandas._libs.tslibs.nattype import NaTType
    from pandas._libs.tslibs.timedeltas import UnitChoices

    from pandas import (
        DataFrame,
        Series,
    )

# ---------------------------------------------------------------------
# types used in annotations

ArrayConvertible = Union[list, tuple, AnyArrayLike]
Scalar = Union[float, str]
DatetimeScalar = Union[Scalar, date, np.datetime64]

DatetimeScalarOrArrayConvertible = Union[DatetimeScalar, ArrayConvertible]

DatetimeDictArg = Union[list[Scalar], tuple[Scalar, ...], AnyArrayLike]


class YearMonthDayDict(TypedDict, total=True):
    year: DatetimeDictArg
    month: DatetimeDictArg
    day: DatetimeDictArg


class FulldatetimeDict(YearMonthDayDict, total=False):
    hour: DatetimeDictArg
    hours: DatetimeDictArg
    minute: DatetimeDictArg
    minutes: DatetimeDictArg
    second: DatetimeDictArg
    seconds: DatetimeDictArg
    ms: DatetimeDictArg
    us: DatetimeDictArg
    ns: DatetimeDictArg


DictConvertible = Union[FulldatetimeDict, "DataFrame"]
start_caching_at = 50


# ---------------------------------------------------------------------


def _guess_datetime_format_for_array(arr, dayfirst: bool | None = False) -> str | None:
    # Try to guess the format based on the first non-NaN element, return None if can't
    if (first_non_null := tslib.first_non_null(arr)) != -1:
        if type(first_non_nan_element := arr[first_non_null]) is str:  # noqa: E721
            # GH#32264 np.str_ object
            guessed_format = guess_datetime_format(
                first_non_nan_element, dayfirst=dayfirst
            )
            if guessed_format is not None:
                return guessed_format
            # If there are multiple non-null elements, warn about
            # how parsing might not be consistent
            if tslib.first_non_null(arr[first_non_null + 1 :]) != -1:
                warnings.warn(
                    "Could not infer format, so each element will be parsed "
                    "individually, falling back to `dateutil`. To ensure parsing is "
                    "consistent and as-expected, please specify a format.",
                    UserWarning,
                    stacklevel=find_stack_level(),
                )
    return None


def should_cache(
    arg: ArrayConvertible, unique_share: float = 0.7, check_count: int | None = None
) -> bool:
    """
    Decides whether to do caching.

    If the percent of unique elements among `check_count` elements less
    than `unique_share * 100` then we can do caching.

    Parameters
    ----------
    arg: listlike, tuple, 1-d array, Series
    unique_share: float, default=0.7, optional
        0 < unique_share < 1
    check_count: int, optional
        0 <= check_count <= len(arg)

    Returns
    -------
    do_caching: bool

    Notes
    -----
    By default for a sequence of less than 50 items in size, we don't do
    caching; for the number of elements less than 5000, we take ten percent of
    all elements to check for a uniqueness share; if the sequence size is more
    than 5000, then we check only the first 500 elements.
    All constants were chosen empirically by.
    """
    do_caching = True

    # default realization
    if check_count is None:
        # in this case, the gain from caching is negligible
        if len(arg) <= start_caching_at:
            return False

        if len(arg) <= 5000:
            check_count = len(arg) // 10
        else:
            check_count = 500
    else:
        assert (
            0 <= check_count <= len(arg)
        ), "check_count must be in next bounds: [0; len(arg)]"
        if check_count == 0:
            return False

    assert 0 < unique_share < 1, "unique_share must be in next bounds: (0; 1)"

    try:
        # We can't cache if the items are not hashable.
        unique_elements = set(islice(arg, check_count))
    except TypeError:
        return False
    if len(unique_elements) > check_count * unique_share:
        do_caching = False
    return do_caching


def _maybe_cache(
    arg: ArrayConvertible,
    format: str | None,
    cache: bool,
    convert_listlike: Callable,
) -> Series:
    """
    Create a cache of unique dates from an array of dates

    Parameters
    ----------
    arg : listlike, tuple, 1-d array, Series
    format : string
        Strftime format to parse time
    cache : bool
        True attempts to create a cache of converted values
    convert_listlike : function
        Conversion function to apply on dates

    Returns
    -------
    cache_array : Series
        Cache of converted, unique dates. Can be empty
    """
    from pandas import Series

    cache_array = Series(dtype=object)

    if cache:
        # Perform a quicker unique check
        if not should_cache(arg):
            return cache_array

        if not isinstance(arg, (np.ndarray, ExtensionArray, Index, ABCSeries)):
            arg = np.array(arg)

        unique_dates = unique(arg)
        if len(unique_dates) < len(arg):
            cache_dates = convert_listlike(unique_dates, format)
            # GH#45319
            try:
                cache_array = Series(cache_dates, index=unique_dates, copy=False)
            except OutOfBoundsDatetime:
                return cache_array
            # GH#39882 and GH#35888 in case of None and NaT we get duplicates
            if not cache_array.index.is_unique:
                cache_array = cache_array[~cache_array.index.duplicated()]
    return cache_array


def _box_as_indexlike(
    dt_array: ArrayLike, utc: bool = False, name: Hashable | None = None
) -> Index:
    """
    Properly boxes the ndarray of datetimes to DatetimeIndex
    if it is possible or to generic Index instead

    Parameters
    ----------
    dt_array: 1-d array
        Array of datetimes to be wrapped in an Index.
    utc : bool
        Whether to convert/localize timestamps to UTC.
    name : string, default None
        Name for a resulting index

    Returns
    -------
    result : datetime of converted dates
        - DatetimeIndex if convertible to sole datetime64 type
        - general Index otherwise
    """

    if lib.is_np_dtype(dt_array.dtype, "M"):
        tz = "utc" if utc else None
        return DatetimeIndex(dt_array, tz=tz, name=name)
    return Index(dt_array, name=name, dtype=dt_array.dtype)


def _convert_and_box_cache(
    arg: DatetimeScalarOrArrayConvertible,
    cache_array: Series,
    name: Hashable | None = None,
) -> Index:
    """
    Convert array of dates with a cache and wrap the result in an Index.

    Parameters
    ----------
    arg : integer, float, string, datetime, list, tuple, 1-d array, Series
    cache_array : Series
        Cache of converted, unique dates
    name : string, default None
        Name for a DatetimeIndex

    Returns
    -------
    result : Index-like of converted dates
    """
    from pandas import Series

    result = Series(arg, dtype=cache_array.index.dtype).map(cache_array)
    return _box_as_indexlike(result._values, utc=False, name=name)


def _convert_listlike_datetimes(
    arg,
    format: str | None,
    name: Hashable | None = None,
    utc: bool = False,
    unit: str | None = None,
    errors: DateTimeErrorChoices = "raise",
    dayfirst: bool | None = None,
    yearfirst: bool | None = None,
    exact: bool = True,
):
    """
    Helper function for to_datetime. Performs the conversions of 1D listlike
    of dates

    Parameters
    ----------
    arg : list, tuple, ndarray, Series, Index
        date to be parsed
    name : object
        None or string for the Index name
    utc : bool
        Whether to convert/localize timestamps to UTC.
    unit : str
        None or string of the frequency of the passed data
    errors : str
        error handing behaviors from to_datetime, 'raise', 'coerce', 'ignore'
    dayfirst : bool
        dayfirst parsing behavior from to_datetime
    yearfirst : bool
        yearfirst parsing behavior from to_datetime
    exact : bool, default True
        exact format matching behavior from to_datetime

    Returns
    -------
    Index-like of parsed dates
    """
    if isinstance(arg, (list, tuple)):
        arg = np.array(arg, dtype="O")
    elif isinstance(arg, NumpyExtensionArray):
        arg = np.array(arg)

    arg_dtype = getattr(arg, "dtype", None)
    # these are shortcutable
    tz = "utc" if utc else None
    if isinstance(arg_dtype, DatetimeTZDtype):
        if not isinstance(arg, (DatetimeArray, DatetimeIndex)):
            return DatetimeIndex(arg, tz=tz, name=name)
        if utc:
            arg = arg.tz_convert(None).tz_localize("utc")
        return arg

    elif isinstance(arg_dtype, ArrowDtype) and arg_dtype.type is Timestamp:
        # TODO: Combine with above if DTI/DTA supports Arrow timestamps
        if utc:
            # pyarrow uses UTC, not lowercase utc
            if isinstance(arg, Index):
                arg_array = cast(ArrowExtensionArray, arg.array)
                if arg_dtype.pyarrow_dtype.tz is not None:
                    arg_array = arg_array._dt_tz_convert("UTC")
                else:
                    arg_array = arg_array._dt_tz_localize("UTC")
                arg = Index(arg_array)
            else:
                # ArrowExtensionArray
                if arg_dtype.pyarrow_dtype.tz is not None:
                    arg = arg._dt_tz_convert("UTC")
                else:
                    arg = arg._dt_tz_localize("UTC")
        return arg

    elif lib.is_np_dtype(arg_dtype, "M"):
        if not is_supported_dtype(arg_dtype):
            # We go to closest supported reso, i.e. "s"
            arg = astype_overflowsafe(
                # TODO: looks like we incorrectly raise with errors=="ignore"
                np.asarray(arg),
                np.dtype("M8[s]"),
                is_coerce=errors == "coerce",
            )

        if not isinstance(arg, (DatetimeArray, DatetimeIndex)):
            return DatetimeIndex(arg, tz=tz, name=name)
        elif utc:
            # DatetimeArray, DatetimeIndex
            return arg.tz_localize("utc")

        return arg

    elif unit is not None:
        if format is not None:
            raise ValueError("cannot specify both format and unit")
        return _to_datetime_with_unit(arg, unit, name, utc, errors)
    elif getattr(arg, "ndim", 1) > 1:
        raise TypeError(
            "arg must be a string, datetime, list, tuple, 1-d array, or Series"
        )

    # warn if passing timedelta64, raise for PeriodDtype
    # NB: this must come after unit transformation
    try:
        arg, _ = maybe_convert_dtype(arg, copy=False, tz=libtimezones.maybe_get_tz(tz))
    except TypeError:
        if errors == "coerce":
            npvalues = np.array(["NaT"], dtype="datetime64[ns]").repeat(len(arg))
            return DatetimeIndex(npvalues, name=name)
        elif errors == "ignore":
            idx = Index(arg, name=name)
            return idx
        raise

    arg = ensure_object(arg)

    if format is None:
        format = _guess_datetime_format_for_array(arg, dayfirst=dayfirst)

    # `format` could be inferred, or user didn't ask for mixed-format parsing.
    if format is not None and format != "mixed":
        return _array_strptime_with_fallback(arg, name, utc, format, exact, errors)

    result, tz_parsed = objects_to_datetime64(
        arg,
        dayfirst=dayfirst,
        yearfirst=yearfirst,
        utc=utc,
        errors=errors,
        allow_object=True,
    )

    if tz_parsed is not None:
        # We can take a shortcut since the datetime64 numpy array
        # is in UTC
        out_unit = np.datetime_data(result.dtype)[0]
        dtype = cast(DatetimeTZDtype, tz_to_dtype(tz_parsed, out_unit))
        dt64_values = result.view(f"M8[{dtype.unit}]")
        dta = DatetimeArray._simple_new(dt64_values, dtype=dtype)
        return DatetimeIndex._simple_new(dta, name=name)

    return _box_as_indexlike(result, utc=utc, name=name)


def _array_strptime_with_fallback(
    arg,
    name,
    utc: bool,
    fmt: str,
    exact: bool,
    errors: str,
) -> Index:
    """
    Call array_strptime, with fallback behavior depending on 'errors'.
    """
    result, tz_out = array_strptime(arg, fmt, exact=exact, errors=errors, utc=utc)
    if tz_out is not None:
        unit = np.datetime_data(result.dtype)[0]
        dtype = DatetimeTZDtype(tz=tz_out, unit=unit)
        dta = DatetimeArray._simple_new(result, dtype=dtype)
        if utc:
            dta = dta.tz_convert("UTC")
        return Index(dta, name=name)
    elif result.dtype != object and utc:
        unit = np.datetime_data(result.dtype)[0]
        res = Index(result, dtype=f"M8[{unit}, UTC]", name=name)
        return res
    elif using_string_dtype() and result.dtype == object:
        if lib.is_string_array(result):
            return Index(result, dtype="str", name=name)
    return Index(result, dtype=result.dtype, name=name)


def _to_datetime_with_unit(arg, unit, name, utc: bool, errors: str) -> Index:
    """
    to_datetime specalized to the case where a 'unit' is passed.
    """
    arg = extract_array(arg, extract_numpy=True)

    # GH#30050 pass an ndarray to tslib.array_with_unit_to_datetime
    # because it expects an ndarray argument
    if isinstance(arg, IntegerArray):
        arr = arg.astype(f"datetime64[{unit}]")
        tz_parsed = None
    else:
        arg = np.asarray(arg)

        if arg.dtype.kind in "iu":
            # Note we can't do "f" here because that could induce unwanted
            #  rounding GH#14156, GH#20445
            arr = arg.astype(f"datetime64[{unit}]", copy=False)
            try:
                arr = astype_overflowsafe(arr, np.dtype("M8[ns]"), copy=False)
            except OutOfBoundsDatetime:
                if errors == "raise":
                    raise
                arg = arg.astype(object)
                return _to_datetime_with_unit(arg, unit, name, utc, errors)
            tz_parsed = None

        elif arg.dtype.kind == "f":
            with np.errstate(over="raise"):
                try:
                    arr = cast_from_unit_vectorized(arg, unit=unit)
                except OutOfBoundsDatetime:
                    if errors != "raise":
                        return _to_datetime_with_unit(
                            arg.astype(object), unit, name, utc, errors
                        )
                    raise OutOfBoundsDatetime(
                        f"cannot convert input with unit '{unit}'"
                    )

            arr = arr.view("M8[ns]")
            tz_parsed = None
        else:
            arg = arg.astype(object, copy=False)
            arr, tz_parsed = tslib.array_with_unit_to_datetime(arg, unit, errors=errors)

    if errors == "ignore":
        # Index constructor _may_ infer to DatetimeIndex
        result = Index._with_infer(arr, name=name)
    else:
        result = DatetimeIndex(arr, name=name)

    if not isinstance(result, DatetimeIndex):
        return result

    # GH#23758: We may still need to localize the result with tz
    # GH#25546: Apply tz_parsed first (from arg), then tz (from caller)
    # result will be naive but in UTC
    result = result.tz_localize("UTC").tz_convert(tz_parsed)

    if utc:
        if result.tz is None:
            result = result.tz_localize("utc")
        else:
            result = result.tz_convert("utc")
    return result


def _adjust_to_origin(arg, origin, unit):
    """
    Helper function for to_datetime.
    Adjust input argument to the specified origin

    Parameters
    ----------
    arg : list, tuple, ndarray, Series, Index
        date to be adjusted
    origin : 'julian' or Timestamp
        origin offset for the arg
    unit : str
        passed unit from to_datetime, must be 'D'

    Returns
    -------
    ndarray or scalar of adjusted date(s)
    """
    if origin == "julian":
        original = arg
        j0 = Timestamp(0).to_julian_date()
        if unit != "D":
            raise ValueError("unit must be 'D' for origin='julian'")
        try:
            arg = arg - j0
        except TypeError as err:
            raise ValueError(
                "incompatible 'arg' type for given 'origin'='julian'"
            ) from err

        # preemptively check this for a nice range
        j_max = Timestamp.max.to_julian_date() - j0
        j_min = Timestamp.min.to_julian_date() - j0
        if np.any(arg > j_max) or np.any(arg < j_min):
            raise OutOfBoundsDatetime(
                f"{original} is Out of Bounds for origin='julian'"
            )
    else:
        # arg must be numeric
        if not (
            (is_integer(arg) or is_float(arg)) or is_numeric_dtype(np.asarray(arg))
        ):
            raise ValueError(
                f"'{arg}' is not compatible with origin='{origin}'; "
                "it must be numeric with a unit specified"
            )

        # we are going to offset back to unix / epoch time
        try:
            offset = Timestamp(origin, unit=unit)
        except OutOfBoundsDatetime as err:
            raise OutOfBoundsDatetime(f"origin {origin} is Out of Bounds") from err
        except ValueError as err:
            raise ValueError(
                f"origin {origin} cannot be converted to a Timestamp"
            ) from err

        if offset.tz is not None:
            raise ValueError(f"origin offset {offset} must be tz-naive")
        td_offset = offset - Timestamp(0)

        # convert the offset to the unit of the arg
        # this should be lossless in terms of precision
        ioffset = td_offset // Timedelta(1, unit=unit)

        # scalars & ndarray-like can handle the addition
        if is_list_like(arg) and not isinstance(arg, (ABCSeries, Index, np.ndarray)):
            arg = np.asarray(arg)
        arg = arg + ioffset
    return arg


@overload
def to_datetime(
    arg: DatetimeScalar,
    errors: DateTimeErrorChoices = ...,
    dayfirst: bool = ...,
    yearfirst: bool = ...,
    utc: bool = ...,
    format: str | None = ...,
    exact: bool = ...,
    unit: str | None = ...,
    infer_datetime_format: bool = ...,
    origin=...,
    cache: bool = ...,
) -> Timestamp:
    ...


@overload
def to_datetime(
    arg: Series | DictConvertible,
    errors: DateTimeErrorChoices = ...,
    dayfirst: bool = ...,
    yearfirst: bool = ...,
    utc: bool = ...,
    format: str | None = ...,
    exact: bool = ...,
    unit: str | None = ...,
    infer_datetime_format: bool = ...,
    origin=...,
    cache: bool = ...,
) -> Series:
    ...


@overload
def to_datetime(
    arg: list | tuple | Index | ArrayLike,
    errors: DateTimeErrorChoices = ...,
    dayfirst: bool = ...,
    yearfirst: bool = ...,
    utc: bool = ...,
    format: str | None = ...,
    exact: bool = ...,
    unit: str | None = ...,
    infer_datetime_format: bool = ...,
    origin=...,
    cache: bool = ...,
) -> DatetimeIndex:
    ...


def to_datetime(
    arg: DatetimeScalarOrArrayConvertible | DictConvertible,
    errors: DateTimeErrorChoices = "raise",
    dayfirst: bool = False,
    yearfirst: bool = False,
    utc: bool = False,
    format: str | None = None,
    exact: bool | lib.NoDefault = lib.no_default,
    unit: str | None = None,
    infer_datetime_format: lib.NoDefault | bool = lib.no_default,
    origin: str = "unix",
    cache: bool = True,
) -> DatetimeIndex | Series | DatetimeScalar | NaTType | None:
    """
    Convert argument to datetime.

    This function converts a scalar, array-like, :class:`Series` or
    :class:`DataFrame`/dict-like to a pandas datetime object.

    Parameters
    ----------
    arg : int, float, str, datetime, list, tuple, 1-d array, Series, DataFrame/dict-like
        The object to convert to a datetime. If a :class:`DataFrame` is provided, the
        method expects minimally the following columns: :const:`"year"`,
        :const:`"month"`, :const:`"day"`. The column "year"
        must be specified in 4-digit format.
    errors : {'ignore', 'raise', 'coerce'}, default 'raise'
        - If :const:`'raise'`, then invalid parsing will raise an exception.
        - If :const:`'coerce'`, then invalid parsing will be set as :const:`NaT`.
        - If :const:`'ignore'`, then invalid parsing will return the input.
    dayfirst : bool, default False
        Specify a date parse order if `arg` is str or is list-like.
        If :const:`True`, parses dates with the day first, e.g. :const:`"10/11/12"`
        is parsed as :const:`2012-11-10`.

        .. warning::

            ``dayfirst=True`` is not strict, but will prefer to parse
            with day first.

    yearfirst : bool, default False
        Specify a date parse order if `arg` is str or is list-like.

        - If :const:`True` parses dates with the year first, e.g.
          :const:`"10/11/12"` is parsed as :const:`2010-11-12`.
        - If both `dayfirst` and `yearfirst` are :const:`True`, `yearfirst` is
          preceded (same as :mod:`dateutil`).

        .. warning::

            ``yearfirst=True`` is not strict, but will prefer to parse
            with year first.

    utc : bool, default False
        Control timezone-related parsing, localization and conversion.

        - If :const:`True`, the function *always* returns a timezone-aware
          UTC-localized :class:`Timestamp`, :class:`Series` or
          :class:`DatetimeIndex`. To do this, timezone-naive inputs are
          *localized* as UTC, while timezone-aware inputs are *converted* to UTC.

        - If :const:`False` (default), inputs will not be coerced to UTC.
          Timezone-naive inputs will remain naive, while timezone-aware ones
          will keep their time offsets. Limitations exist for mixed
          offsets (typically, daylight savings), see :ref:`Examples
          <to_datetime_tz_examples>` section for details.

        .. warning::

            In a future version of pandas, parsing datetimes with mixed time
            zones will raise an error unless `utc=True`.
            Please specify `utc=True` to opt in to the new behaviour
            and silence this warning. To create a `Series` with mixed offsets and
            `object` dtype, please use `apply` and `datetime.datetime.strptime`.

        See also: pandas general documentation about `timezone conversion and
        localization
        <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html
        #time-zone-handling>`_.

    format : str, default None
        The strftime to parse time, e.g. :const:`"%d/%m/%Y"`. See
        `strftime documentation
        <https://docs.python.org/3/library/datetime.html
        #strftime-and-strptime-behavior>`_ for more information on choices, though
        note that :const:`"%f"` will parse all the way up to nanoseconds.
        You can also pass:

        - "ISO8601", to parse any `ISO8601 <https://en.wikipedia.org/wiki/ISO_8601>`_
          time string (not necessarily in exactly the same format);
        - "mixed", to infer the format for each element individually. This is risky,
          and you should probably use it along with `dayfirst`.

        .. note::

            If a :class:`DataFrame` is passed, then `format` has no effect.

    exact : bool, default True
        Control how `format` is used:

        - If :const:`True`, require an exact `format` match.
        - If :const:`False`, allow the `format` to match anywhere in the target
          string.

        Cannot be used alongside ``format='ISO8601'`` or ``format='mixed'``.
    unit : str, default 'ns'
        The unit of the arg (D,s,ms,us,ns) denote the unit, which is an
        integer or float number. This will be based off the origin.
        Example, with ``unit='ms'`` and ``origin='unix'``, this would calculate
        the number of milliseconds to the unix epoch start.
    infer_datetime_format : bool, default False
        If :const:`True` and no `format` is given, attempt to infer the format
        of the datetime strings based on the first non-NaN element,
        and if it can be inferred, switch to a faster method of parsing them.
        In some cases this can increase the parsing speed by ~5-10x.

        .. deprecated:: 2.0.0
            A strict version of this argument is now the default, passing it has
            no effect.

    origin : scalar, default 'unix'
        Define the reference date. The numeric values would be parsed as number
        of units (defined by `unit`) since this reference date.

        - If :const:`'unix'` (or POSIX) time; origin is set to 1970-01-01.
        - If :const:`'julian'`, unit must be :const:`'D'`, and origin is set to
          beginning of Julian Calendar. Julian day number :const:`0` is assigned
          to the day starting at noon on January 1, 4713 BC.
        - If Timestamp convertible (Timestamp, dt.datetime, np.datetimt64 or date
          string), origin is set to Timestamp identified by origin.
        - If a float or integer, origin is the difference
          (in units determined by the ``unit`` argument) relative to 1970-01-01.
    cache : bool, default True
        If :const:`True`, use a cache of unique, converted dates to apply the
        datetime conversion. May produce significant speed-up when parsing
        duplicate date strings, especially ones with timezone offsets. The cache
        is only used when there are at least 50 values. The presence of
        out-of-bounds values will render the cache unusable and may slow down
        parsing.

    Returns
    -------
    datetime
        If parsing succeeded.
        Return type depends on input (types in parenthesis correspond to
        fallback in case of unsuccessful timezone or out-of-range timestamp
        parsing):

        - scalar: :class:`Timestamp` (or :class:`datetime.datetime`)
        - array-like: :class:`DatetimeIndex` (or :class:`Series` with
          :class:`object` dtype containing :class:`datetime.datetime`)
        - Series: :class:`Series` of :class:`datetime64` dtype (or
          :class:`Series` of :class:`object` dtype containing
          :class:`datetime.datetime`)
        - DataFrame: :class:`Series` of :class:`datetime64` dtype (or
          :class:`Series` of :class:`object` dtype containing
          :class:`datetime.datetime`)

    Raises
    ------
    ParserError
        When parsing a date from string fails.
    ValueError
        When another datetime conversion error happens. For example when one
        of 'year', 'month', day' columns is missing in a :class:`DataFrame`, or
        when a Timezone-aware :class:`datetime.datetime` is found in an array-like
        of mixed time offsets, and ``utc=False``.

    See Also
    --------
    DataFrame.astype : Cast argument to a specified dtype.
    to_timedelta : Convert argument to timedelta.
    convert_dtypes : Convert dtypes.

    Notes
    -----

    Many input types are supported, and lead to different output types:

    - **scalars** can be int, float, str, datetime object (from stdlib :mod:`datetime`
      module or :mod:`numpy`). They are converted to :class:`Timestamp` when
      possible, otherwise they are converted to :class:`datetime.datetime`.
      None/NaN/null scalars are converted to :const:`NaT`.

    - **array-like** can contain int, float, str, datetime objects. They are
      converted to :class:`DatetimeIndex` when possible, otherwise they are
      converted to :class:`Index` with :class:`object` dtype, containing
      :class:`datetime.datetime`. None/NaN/null entries are converted to
      :const:`NaT` in both cases.

    - **Series** are converted to :class:`Series` with :class:`datetime64`
      dtype when possible, otherwise they are converted to :class:`Series` with
      :class:`object` dtype, containing :class:`datetime.datetime`. None/NaN/null
      entries are converted to :const:`NaT` in both cases.

    - **DataFrame/dict-like** are converted to :class:`Series` with
      :class:`datetime64` dtype. For each row a datetime is created from assembling
      the various dataframe columns. Column keys can be common abbreviations
      like ['year', 'month', 'day', 'minute', 'second', 'ms', 'us', 'ns']) or
      plurals of the same.

    The following causes are responsible for :class:`datetime.datetime` objects
    being returned (possibly inside an :class:`Index` or a :class:`Series` with
    :class:`object` dtype) instead of a proper pandas designated type
    (:class:`Timestamp`, :class:`DatetimeIndex` or :class:`Series`
    with :class:`datetime64` dtype):

    - when any input element is before :const:`Timestamp.min` or after
      :const:`Timestamp.max`, see `timestamp limitations
      <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html
      #timeseries-timestamp-limits>`_.

    - when ``utc=False`` (default) and the input is an array-like or
      :class:`Series` containing mixed naive/aware datetime, or aware with mixed
      time offsets. Note that this happens in the (quite frequent) situation when
      the timezone has a daylight savings policy. In that case you may wish to
      use ``utc=True``.

    Examples
    --------

    **Handling various input formats**

    Assembling a datetime from multiple columns of a :class:`DataFrame`. The keys
    can be common abbreviations like ['year', 'month', 'day', 'minute', 'second',
    'ms', 'us', 'ns']) or plurals of the same

    >>> df = pd.DataFrame({'year': [2015, 2016],
    ...                    'month': [2, 3],
    ...                    'day': [4, 5]})
    >>> pd.to_datetime(df)
    0   2015-02-04
    1   2016-03-05
    dtype: datetime64[ns]

    Using a unix epoch time

    >>> pd.to_datetime(1490195805, unit='s')
    Timestamp('2017-03-22 15:16:45')
    >>> pd.to_datetime(1490195805433502912, unit='ns')
    Timestamp('2017-03-22 15:16:45.433502912')

    .. warning:: For float arg, precision rounding might happen. To prevent
        unexpected behavior use a fixed-width exact type.

    Using a non-unix epoch origin

    >>> pd.to_datetime([1, 2, 3], unit='D',
    ...                origin=pd.Timestamp('1960-01-01'))
    DatetimeIndex(['1960-01-02', '1960-01-03', '1960-01-04'],
                  dtype='datetime64[ns]', freq=None)

    **Differences with strptime behavior**

    :const:`"%f"` will parse all the way up to nanoseconds.

    >>> pd.to_datetime('2018-10-26 12:00:00.0000000011',
    ...                format='%Y-%m-%d %H:%M:%S.%f')
    Timestamp('2018-10-26 12:00:00.000000001')

    **Non-convertible date/times**

    Passing ``errors='coerce'`` will force an out-of-bounds date to :const:`NaT`,
    in addition to forcing non-dates (or non-parseable dates) to :const:`NaT`.

    >>> pd.to_datetime('13000101', format='%Y%m%d', errors='coerce')
    NaT

    .. _to_datetime_tz_examples:

    **Timezones and time offsets**

    The default behaviour (``utc=False``) is as follows:

    - Timezone-naive inputs are converted to timezone-naive :class:`DatetimeIndex`:

    >>> pd.to_datetime(['2018-10-26 12:00:00', '2018-10-26 13:00:15'])
    DatetimeIndex(['2018-10-26 12:00:00', '2018-10-26 13:00:15'],
                  dtype='datetime64[ns]', freq=None)

    - Timezone-aware inputs *with constant time offset* are converted to
      timezone-aware :class:`DatetimeIndex`:

    >>> pd.to_datetime(['2018-10-26 12:00 -0500', '2018-10-26 13:00 -0500'])
    DatetimeIndex(['2018-10-26 12:00:00-05:00', '2018-10-26 13:00:00-05:00'],
                  dtype='datetime64[ns, UTC-05:00]', freq=None)

    - However, timezone-aware inputs *with mixed time offsets* (for example
      issued from a timezone with daylight savings, such as Europe/Paris)
      are **not successfully converted** to a :class:`DatetimeIndex`.
      Parsing datetimes with mixed time zones will show a warning unless
      `utc=True`. If you specify `utc=False` the warning below will be shown
      and a simple :class:`Index` containing :class:`datetime.datetime`
      objects will be returned:

    >>> pd.to_datetime(['2020-10-25 02:00 +0200',
    ...                 '2020-10-25 04:00 +0100'])  # doctest: +SKIP
    FutureWarning: In a future version of pandas, parsing datetimes with mixed
    time zones will raise an error unless `utc=True`. Please specify `utc=True`
    to opt in to the new behaviour and silence this warning. To create a `Series`
    with mixed offsets and `object` dtype, please use `apply` and
    `datetime.datetime.strptime`.
    Index([2020-10-25 02:00:00+02:00, 2020-10-25 04:00:00+01:00],
          dtype='object')

    - A mix of timezone-aware and timezone-naive inputs is also converted to
      a simple :class:`Index` containing :class:`datetime.datetime` objects:

    >>> from datetime import datetime
    >>> pd.to_datetime(["2020-01-01 01:00:00-01:00",
    ...                 datetime(2020, 1, 1, 3, 0)])  # doctest: +SKIP
    FutureWarning: In a future version of pandas, parsing datetimes with mixed
    time zones will raise an error unless `utc=True`. Please specify `utc=True`
    to opt in to the new behaviour and silence this warning. To create a `Series`
    with mixed offsets and `object` dtype, please use `apply` and
    `datetime.datetime.strptime`.
    Index([2020-01-01 01:00:00-01:00, 2020-01-01 03:00:00], dtype='object')

    |

    Setting ``utc=True`` solves most of the above issues:

    - Timezone-naive inputs are *localized* as UTC

    >>> pd.to_datetime(['2018-10-26 12:00', '2018-10-26 13:00'], utc=True)
    DatetimeIndex(['2018-10-26 12:00:00+00:00', '2018-10-26 13:00:00+00:00'],
                  dtype='datetime64[ns, UTC]', freq=None)

    - Timezone-aware inputs are *converted* to UTC (the output represents the
      exact same datetime, but viewed from the UTC time offset `+00:00`).

    >>> pd.to_datetime(['2018-10-26 12:00 -0530', '2018-10-26 12:00 -0500'],
    ...                utc=True)
    DatetimeIndex(['2018-10-26 17:30:00+00:00', '2018-10-26 17:00:00+00:00'],
                  dtype='datetime64[ns, UTC]', freq=None)

    - Inputs can contain both string or datetime, the above
      rules still apply

    >>> pd.to_datetime(['2018-10-26 12:00', datetime(2020, 1, 1, 18)], utc=True)
    DatetimeIndex(['2018-10-26 12:00:00+00:00', '2020-01-01 18:00:00+00:00'],
                  dtype='datetime64[ns, UTC]', freq=None)
    """
    if exact is not lib.no_default and format in {"mixed", "ISO8601"}:
        raise ValueError("Cannot use 'exact' when 'format' is 'mixed' or 'ISO8601'")
    if infer_datetime_format is not lib.no_default:
        warnings.warn(
            "The argument 'infer_datetime_format' is deprecated and will "
            "be removed in a future version. "
            "A strict version of it is now the default, see "
            "https://pandas.pydata.org/pdeps/0004-consistent-to-datetime-parsing.html. "
            "You can safely remove this argument.",
            stacklevel=find_stack_level(),
        )
    if errors == "ignore":
        # GH#54467
        warnings.warn(
            "errors='ignore' is deprecated and will raise in a future version. "
            "Use to_datetime without passing `errors` and catch exceptions "
            "explicitly instead",
            FutureWarning,
            stacklevel=find_stack_level(),
        )

    if arg is None:
        return None

    if origin != "unix":
        arg = _adjust_to_origin(arg, origin, unit)

    convert_listlike = partial(
        _convert_listlike_datetimes,
        utc=utc,
        unit=unit,
        dayfirst=dayfirst,
        yearfirst=yearfirst,
        errors=errors,
        exact=exact,
    )
    # pylint: disable-next=used-before-assignment
    result: Timestamp | NaTType | Series | Index

    if isinstance(arg, Timestamp):
        result = arg
        if utc:
            if arg.tz is not None:
                result = arg.tz_convert("utc")
            else:
                result = arg.tz_localize("utc")
    elif isinstance(arg, ABCSeries):
        cache_array = _maybe_cache(arg, format, cache, convert_listlike)
        if not cache_array.empty:
            result = arg.map(cache_array)
        else:
            values = convert_listlike(arg._values, format)
            result = arg._constructor(values, index=arg.index, name=arg.name)
    elif isinstance(arg, (ABCDataFrame, abc.MutableMapping)):
        result = _assemble_from_unit_mappings(arg, errors, utc)
    elif isinstance(arg, Index):
        cache_array = _maybe_cache(arg, format, cache, convert_listlike)
        if not cache_array.empty:
            result = _convert_and_box_cache(arg, cache_array, name=arg.name)
        else:
            result = convert_listlike(arg, format, name=arg.name)
    elif is_list_like(arg):
        try:
            # error: Argument 1 to "_maybe_cache" has incompatible type
            # "Union[float, str, datetime, List[Any], Tuple[Any, ...], ExtensionArray,
            # ndarray[Any, Any], Series]"; expected "Union[List[Any], Tuple[Any, ...],
            # Union[Union[ExtensionArray, ndarray[Any, Any]], Index, Series], Series]"
            argc = cast(
                Union[list, tuple, ExtensionArray, np.ndarray, "Series", Index], arg
            )
            cache_array = _maybe_cache(argc, format, cache, convert_listlike)
        except OutOfBoundsDatetime:
            # caching attempts to create a DatetimeIndex, which may raise
            # an OOB. If that's the desired behavior, then just reraise...
            if errors == "raise":
                raise
            # ... otherwise, continue without the cache.
            from pandas import Series

            cache_array = Series([], dtype=object)  # just an empty array
        if not cache_array.empty:
            result = _convert_and_box_cache(argc, cache_array)
        else:
            result = convert_listlike(argc, format)
    else:
        result = convert_listlike(np.array([arg]), format)[0]
        if isinstance(arg, bool) and isinstance(result, np.bool_):
            result = bool(result)  # TODO: avoid this kludge.

    #  error: Incompatible return value type (got "Union[Timestamp, NaTType,
    # Series, Index]", expected "Union[DatetimeIndex, Series, float, str,
    # NaTType, None]")
    return result  # type: ignore[return-value]


# mappings for assembling units
_unit_map = {
    "year": "year",
    "years": "year",
    "month": "month",
    "months": "month",
    "day": "day",
    "days": "day",
    "hour": "h",
    "hours": "h",
    "minute": "m",
    "minutes": "m",
    "second": "s",
    "seconds": "s",
    "ms": "ms",
    "millisecond": "ms",
    "milliseconds": "ms",
    "us": "us",
    "microsecond": "us",
    "microseconds": "us",
    "ns": "ns",
    "nanosecond": "ns",
    "nanoseconds": "ns",
}


def _assemble_from_unit_mappings(arg, errors: DateTimeErrorChoices, utc: bool):
    """
    assemble the unit specified fields from the arg (DataFrame)
    Return a Series for actual parsing

    Parameters
    ----------
    arg : DataFrame
    errors : {'ignore', 'raise', 'coerce'}, default 'raise'

        - If :const:`'raise'`, then invalid parsing will raise an exception
        - If :const:`'coerce'`, then invalid parsing will be set as :const:`NaT`
        - If :const:`'ignore'`, then invalid parsing will return the input
    utc : bool
        Whether to convert/localize timestamps to UTC.

    Returns
    -------
    Series
    """
    from pandas import (
        DataFrame,
        to_numeric,
        to_timedelta,
    )

    arg = DataFrame(arg)
    if not arg.columns.is_unique:
        raise ValueError("cannot assemble with duplicate keys")

    # replace passed unit with _unit_map
    def f(value):
        if value in _unit_map:
            return _unit_map[value]

        # m is case significant
        if value.lower() in _unit_map:
            return _unit_map[value.lower()]

        return value

    unit = {k: f(k) for k in arg.keys()}
    unit_rev = {v: k for k, v in unit.items()}

    # we require at least Ymd
    required = ["year", "month", "day"]
    req = sorted(set(required) - set(unit_rev.keys()))
    if len(req):
        _required = ",".join(req)
        raise ValueError(
            "to assemble mappings requires at least that "
            f"[year, month, day] be specified: [{_required}] is missing"
        )

    # keys we don't recognize
    excess = sorted(set(unit_rev.keys()) - set(_unit_map.values()))
    if len(excess):
        _excess = ",".join(excess)
        raise ValueError(
            f"extra keys have been passed to the datetime assemblage: [{_excess}]"
        )

    def coerce(values):
        # we allow coercion to if errors allows
        values = to_numeric(values, errors=errors)

        # prevent overflow in case of int8 or int16
        if is_integer_dtype(values.dtype):
            values = values.astype("int64", copy=False)
        return values

    values = (
        coerce(arg[unit_rev["year"]]) * 10000
        + coerce(arg[unit_rev["month"]]) * 100
        + coerce(arg[unit_rev["day"]])
    )
    try:
        values = to_datetime(values, format="%Y%m%d", errors=errors, utc=utc)
    except (TypeError, ValueError) as err:
        raise ValueError(f"cannot assemble the datetimes: {err}") from err

    units: list[UnitChoices] = ["h", "m", "s", "ms", "us", "ns"]
    for u in units:
        value = unit_rev.get(u)
        if value is not None and value in arg:
            try:
                values += to_timedelta(coerce(arg[value]), unit=u, errors=errors)
            except (TypeError, ValueError) as err:
                raise ValueError(
                    f"cannot assemble the datetimes [{value}]: {err}"
                ) from err
    return values


__all__ = [
    "DateParseError",
    "should_cache",
    "to_datetime",
]
<|endoftext|>"""
timedelta support tools
"""
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    overload,
)
import warnings

import numpy as np

from pandas._libs import lib
from pandas._libs.tslibs import (
    NaT,
    NaTType,
)
from pandas._libs.tslibs.timedeltas import (
    Timedelta,
    disallow_ambiguous_unit,
    parse_timedelta_unit,
)
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import is_list_like
from pandas.core.dtypes.dtypes import ArrowDtype
from pandas.core.dtypes.generic import (
    ABCIndex,
    ABCSeries,
)

from pandas.core.arrays.timedeltas import sequence_to_td64ns

if TYPE_CHECKING:
    from collections.abc import Hashable
    from datetime import timedelta

    from pandas._libs.tslibs.timedeltas import UnitChoices
    from pandas._typing import (
        ArrayLike,
        DateTimeErrorChoices,
    )

    from pandas import (
        Index,
        Series,
        TimedeltaIndex,
    )


@overload
def to_timedelta(
    arg: str | float | timedelta,
    unit: UnitChoices | None = ...,
    errors: DateTimeErrorChoices = ...,
) -> Timedelta:
    ...


@overload
def to_timedelta(
    arg: Series,
    unit: UnitChoices | None = ...,
    errors: DateTimeErrorChoices = ...,
) -> Series:
    ...


@overload
def to_timedelta(
    arg: list | tuple | range | ArrayLike | Index,
    unit: UnitChoices | None = ...,
    errors: DateTimeErrorChoices = ...,
) -> TimedeltaIndex:
    ...


def to_timedelta(
    arg: str
    | int
    | float
    | timedelta
    | list
    | tuple
    | range
    | ArrayLike
    | Index
    | Series,
    unit: UnitChoices | None = None,
    errors: DateTimeErrorChoices = "raise",
) -> Timedelta | TimedeltaIndex | Series:
    """
    Convert argument to timedelta.

    Timedeltas are absolute differences in times, expressed in difference
    units (e.g. days, hours, minutes, seconds). This method converts
    an argument from a recognized timedelta format / value into
    a Timedelta type.

    Parameters
    ----------
    arg : str, timedelta, list-like or Series
        The data to be converted to timedelta.

        .. versionchanged:: 2.0
            Strings with units 'M', 'Y' and 'y' do not represent
            unambiguous timedelta values and will raise an exception.

    unit : str, optional
        Denotes the unit of the arg for numeric `arg`. Defaults to ``"ns"``.

        Possible values:

        * 'W'
        * 'D' / 'days' / 'day'
        * 'hours' / 'hour' / 'hr' / 'h' / 'H'
        * 'm' / 'minute' / 'min' / 'minutes' / 'T'
        * 's' / 'seconds' / 'sec' / 'second' / 'S'
        * 'ms' / 'milliseconds' / 'millisecond' / 'milli' / 'millis' / 'L'
        * 'us' / 'microseconds' / 'microsecond' / 'micro' / 'micros' / 'U'
        * 'ns' / 'nanoseconds' / 'nano' / 'nanos' / 'nanosecond' / 'N'

        Must not be specified when `arg` contains strings and ``errors="raise"``.

        .. deprecated:: 2.2.0
            Units 'H', 'T', 'S', 'L', 'U' and 'N' are deprecated and will be removed
            in a future version. Please use 'h', 'min', 's', 'ms', 'us', and 'ns'
            instead of 'H', 'T', 'S', 'L', 'U' and 'N'.

    errors : {'ignore', 'raise', 'coerce'}, default 'raise'
