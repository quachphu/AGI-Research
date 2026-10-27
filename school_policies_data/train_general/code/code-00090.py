            dayfirst=self.dayfirst,
            cache_dates=self.cache_dates,
        )

        # validate header options for mi
        self.header = kwds.get("header")
        if is_list_like(self.header, allow_sets=False):
            if kwds.get("usecols"):
                raise ValueError(
                    "cannot specify usecols when specifying a multi-index header"
                )
            if kwds.get("names"):
                raise ValueError(
                    "cannot specify names when specifying a multi-index header"
                )

            # validate index_col that only contains integers
            if self.index_col is not None:
                # In this case we can pin down index_col as list[int]
                if is_integer(self.index_col):
                    self.index_col = [self.index_col]
                elif not (
                    is_list_like(self.index_col, allow_sets=False)
                    and all(map(is_integer, self.index_col))
                ):
                    raise ValueError(
                        "index_col must only contain row numbers "
                        "when specifying a multi-index header"
                    )
                else:
                    self.index_col = list(self.index_col)

        self._name_processed = False

        self._first_chunk = True

        self.usecols, self.usecols_dtype = self._validate_usecols_arg(kwds["usecols"])

        # Fallback to error to pass a sketchy test(test_override_set_noconvert_columns)
        # Normally, this arg would get pre-processed earlier on
        self.on_bad_lines = kwds.get("on_bad_lines", self.BadLineHandleMethod.ERROR)

    def _validate_parse_dates_presence(self, columns: Sequence[Hashable]) -> Iterable:
        """
        Check if parse_dates are in columns.

        If user has provided names for parse_dates, check if those columns
        are available.

        Parameters
        ----------
        columns : list
            List of names of the dataframe.

        Returns
        -------
        The names of the columns which will get parsed later if a dict or list
        is given as specification.

        Raises
        ------
        ValueError
            If column to parse_date is not in dataframe.

        """
        cols_needed: Iterable
        if is_dict_like(self.parse_dates):
            cols_needed = itertools.chain(*self.parse_dates.values())
        elif is_list_like(self.parse_dates):
            # a column in parse_dates could be represented
            # ColReference = Union[int, str]
            # DateGroups = List[ColReference]
            # ParseDates = Union[DateGroups, List[DateGroups],
            #     Dict[ColReference, DateGroups]]
            cols_needed = itertools.chain.from_iterable(
                col if is_list_like(col) and not isinstance(col, tuple) else [col]
                for col in self.parse_dates
            )
        else:
            cols_needed = []

        cols_needed = list(cols_needed)

        # get only columns that are references using names (str), not by index
        missing_cols = ", ".join(
            sorted(
                {
                    col
                    for col in cols_needed
                    if isinstance(col, str) and col not in columns
                }
            )
        )
        if missing_cols:
            raise ValueError(
                f"Missing column provided to 'parse_dates': '{missing_cols}'"
            )
        # Convert positions to actual column names
        return [
            col if (isinstance(col, str) or col in columns) else columns[col]
            for col in cols_needed
        ]

    def close(self) -> None:
        pass

    @final
    @property
    def _has_complex_date_col(self) -> bool:
        return isinstance(self.parse_dates, dict) or (
            isinstance(self.parse_dates, list)
            and len(self.parse_dates) > 0
            and isinstance(self.parse_dates[0], list)
        )

    @final
    def _should_parse_dates(self, i: int) -> bool:
        if lib.is_bool(self.parse_dates):
            return bool(self.parse_dates)
        else:
            if self.index_names is not None:
                name = self.index_names[i]
            else:
                name = None
            j = i if self.index_col is None else self.index_col[i]

            return (j in self.parse_dates) or (
                name is not None and name in self.parse_dates
            )

    @final
    def _extract_multi_indexer_columns(
        self,
        header,
        index_names: Sequence[Hashable] | None,
        passed_names: bool = False,
    ) -> tuple[
        Sequence[Hashable], Sequence[Hashable] | None, Sequence[Hashable] | None, bool
    ]:
        """
        Extract and return the names, index_names, col_names if the column
        names are a MultiIndex.

        Parameters
        ----------
        header: list of lists
            The header rows
        index_names: list, optional
            The names of the future index
        passed_names: bool, default False
            A flag specifying if names where passed

        """
        if len(header) < 2:
            return header[0], index_names, None, passed_names

        # the names are the tuples of the header that are not the index cols
        # 0 is the name of the index, assuming index_col is a list of column
        # numbers
        ic = self.index_col
        if ic is None:
            ic = []

        if not isinstance(ic, (list, tuple, np.ndarray)):
            ic = [ic]
        sic = set(ic)

        # clean the index_names
        index_names = header.pop(-1)
        index_names, _, _ = self._clean_index_names(index_names, self.index_col)

        # extract the columns
        field_count = len(header[0])

        # check if header lengths are equal
        if not all(len(header_iter) == field_count for header_iter in header[1:]):
            raise ParserError("Header rows must have an equal number of columns.")

        def extract(r):
            return tuple(r[i] for i in range(field_count) if i not in sic)

        columns = list(zip(*(extract(r) for r in header)))
        names = columns.copy()
        for single_ic in sorted(ic):
            names.insert(single_ic, single_ic)

        # Clean the column names (if we have an index_col).
        if len(ic):
            col_names = [
                r[ic[0]]
                if ((r[ic[0]] is not None) and r[ic[0]] not in self.unnamed_cols)
                else None
                for r in header
            ]
        else:
            col_names = [None] * len(header)

        passed_names = True

        return names, index_names, col_names, passed_names

    @final
    def _maybe_make_multi_index_columns(
        self,
        columns: Sequence[Hashable],
        col_names: Sequence[Hashable] | None = None,
    ) -> Sequence[Hashable] | MultiIndex:
        # possibly create a column mi here
        if is_potential_multi_index(columns):
            list_columns = cast(list[tuple], columns)
            return MultiIndex.from_tuples(list_columns, names=col_names)
        return columns

    @final
    def _make_index(
        self, data, alldata, columns, indexnamerow: list[Scalar] | None = None
    ) -> tuple[Index | None, Sequence[Hashable] | MultiIndex]:
        index: Index | None
        if not is_index_col(self.index_col) or not self.index_col:
            index = None

        elif not self._has_complex_date_col:
            simple_index = self._get_simple_index(alldata, columns)
            index = self._agg_index(simple_index)
        elif self._has_complex_date_col:
            if not self._name_processed:
                (self.index_names, _, self.index_col) = self._clean_index_names(
                    list(columns), self.index_col
                )
                self._name_processed = True
            date_index = self._get_complex_date_index(data, columns)
            index = self._agg_index(date_index, try_parse_dates=False)

        # add names for the index
        if indexnamerow:
            coffset = len(indexnamerow) - len(columns)
            assert index is not None
            index = index.set_names(indexnamerow[:coffset])

        # maybe create a mi on the columns
        columns = self._maybe_make_multi_index_columns(columns, self.col_names)

        return index, columns

    @final
    def _get_simple_index(self, data, columns):
        def ix(col):
            if not isinstance(col, str):
                return col
            raise ValueError(f"Index {col} invalid")

        to_remove = []
        index = []
        for idx in self.index_col:
            i = ix(idx)
            to_remove.append(i)
            index.append(data[i])

        # remove index items from content and columns, don't pop in
        # loop
        for i in sorted(to_remove, reverse=True):
            data.pop(i)
            if not self._implicit_index:
                columns.pop(i)

        return index

    @final
    def _get_complex_date_index(self, data, col_names):
        def _get_name(icol):
            if isinstance(icol, str):
                return icol

            if col_names is None:
                raise ValueError(f"Must supply column order to use {icol!s} as index")

            for i, c in enumerate(col_names):
                if i == icol:
                    return c

        to_remove = []
        index = []
        for idx in self.index_col:
            name = _get_name(idx)
            to_remove.append(name)
            index.append(data[name])

        # remove index items from content and columns, don't pop in
        # loop
        for c in sorted(to_remove, reverse=True):
            data.pop(c)
            col_names.remove(c)

        return index

    @final
    def _clean_mapping(self, mapping):
        """converts col numbers to names"""
        if not isinstance(mapping, dict):
            return mapping
        clean = {}
        # for mypy
        assert self.orig_names is not None

        for col, v in mapping.items():
            if isinstance(col, int) and col not in self.orig_names:
                col = self.orig_names[col]
            clean[col] = v
        if isinstance(mapping, defaultdict):
            remaining_cols = set(self.orig_names) - set(clean.keys())
            clean.update({col: mapping[col] for col in remaining_cols})
        return clean

    @final
    def _agg_index(self, index, try_parse_dates: bool = True) -> Index:
        arrays = []
        converters = self._clean_mapping(self.converters)

        if self.index_names is not None:
            names: Iterable = self.index_names
        else:
            names = itertools.cycle([None])
        for i, (arr, name) in enumerate(zip(index, names)):
            if try_parse_dates and self._should_parse_dates(i):
                arr = self._date_conv(
                    arr,
                    col=self.index_names[i] if self.index_names is not None else None,
                )

            if self.na_filter:
                col_na_values = self.na_values
                col_na_fvalues = self.na_fvalues
            else:
                col_na_values = set()
                col_na_fvalues = set()

            if isinstance(self.na_values, dict):
                assert self.index_names is not None
                col_name = self.index_names[i]
                if col_name is not None:
                    col_na_values, col_na_fvalues = _get_na_values(
                        col_name, self.na_values, self.na_fvalues, self.keep_default_na
                    )

            clean_dtypes = self._clean_mapping(self.dtype)

            cast_type = None
            index_converter = False
            if self.index_names is not None:
                if isinstance(clean_dtypes, dict):
                    cast_type = clean_dtypes.get(self.index_names[i], None)

                if isinstance(converters, dict):
                    index_converter = converters.get(self.index_names[i]) is not None

            try_num_bool = not (
                cast_type and is_string_dtype(cast_type) or index_converter
            )

            arr, _ = self._infer_types(
                arr, col_na_values | col_na_fvalues, cast_type is None, try_num_bool
            )
            if cast_type is not None:
                # Don't perform RangeIndex inference
                idx = Index(arr, name=name, dtype=cast_type)
            else:
                idx = ensure_index_from_sequences([arr], [name])
            arrays.append(idx)

        if len(arrays) == 1:
            return arrays[0]
        else:
            return MultiIndex.from_arrays(arrays)

    @final
    def _convert_to_ndarrays(
        self,
        dct: Mapping,
        na_values,
        na_fvalues,
        verbose: bool = False,
        converters=None,
        dtypes=None,
    ):
        result = {}
        for c, values in dct.items():
            conv_f = None if converters is None else converters.get(c, None)
            if isinstance(dtypes, dict):
                cast_type = dtypes.get(c, None)
            else:
                # single dtype or None
                cast_type = dtypes

            if self.na_filter:
                col_na_values, col_na_fvalues = _get_na_values(
                    c, na_values, na_fvalues, self.keep_default_na
                )
            else:
                col_na_values, col_na_fvalues = set(), set()

            if c in self._parse_date_cols:
                # GH#26203 Do not convert columns which get converted to dates
                # but replace nans to ensure to_datetime works
                mask = algorithms.isin(values, set(col_na_values) | col_na_fvalues)
                np.putmask(values, mask, np.nan)
                result[c] = values
                continue

            if conv_f is not None:
                # conv_f applied to data before inference
                if cast_type is not None:
                    warnings.warn(
                        (
                            "Both a converter and dtype were specified "
                            f"for column {c} - only the converter will be used."
                        ),
                        ParserWarning,
                        stacklevel=find_stack_level(),
                    )

                try:
                    values = lib.map_infer(values, conv_f)
                except ValueError:
                    mask = algorithms.isin(values, list(na_values)).view(np.uint8)
                    values = lib.map_infer_mask(values, conv_f, mask)

                cvals, na_count = self._infer_types(
                    values,
                    set(col_na_values) | col_na_fvalues,
                    cast_type is None,
                    try_num_bool=False,
                )
            else:
                is_ea = is_extension_array_dtype(cast_type)
                is_str_or_ea_dtype = is_ea or is_string_dtype(cast_type)
                # skip inference if specified dtype is object
                # or casting to an EA
                try_num_bool = not (cast_type and is_str_or_ea_dtype)

                # general type inference and conversion
                cvals, na_count = self._infer_types(
                    values,
                    set(col_na_values) | col_na_fvalues,
                    cast_type is None,
                    try_num_bool,
                )

                # type specified in dtype param or cast_type is an EA
                if cast_type is not None:
                    cast_type = pandas_dtype(cast_type)
                if cast_type and (cvals.dtype != cast_type or is_ea):
                    if not is_ea and na_count > 0:
                        if is_bool_dtype(cast_type):
                            raise ValueError(f"Bool column has NA values in column {c}")
                    cvals = self._cast_types(cvals, cast_type, c)

            result[c] = cvals
            if verbose and na_count:
                print(f"Filled {na_count} NA values in column {c!s}")
        return result

    @final
    def _set_noconvert_dtype_columns(
        self, col_indices: list[int], names: Sequence[Hashable]
    ) -> set[int]:
        """
        Set the columns that should not undergo dtype conversions.

        Currently, any column that is involved with date parsing will not
        undergo such conversions. If usecols is specified, the positions of the columns
        not to cast is relative to the usecols not to all columns.

        Parameters
        ----------
        col_indices: The indices specifying order and positions of the columns
        names: The column names which order is corresponding with the order
               of col_indices

        Returns
        -------
        A set of integers containing the positions of the columns not to convert.
        """
        usecols: list[int] | list[str] | None
        noconvert_columns = set()
        if self.usecols_dtype == "integer":
            # A set of integers will be converted to a list in
            # the correct order every single time.
            usecols = sorted(self.usecols)
        elif callable(self.usecols) or self.usecols_dtype not in ("empty", None):
            # The names attribute should have the correct columns
            # in the proper order for indexing with parse_dates.
            usecols = col_indices
        else:
            # Usecols is empty.
            usecols = None

        def _set(x) -> int:
            if usecols is not None and is_integer(x):
                x = usecols[x]

            if not is_integer(x):
                x = col_indices[names.index(x)]

            return x

        if isinstance(self.parse_dates, list):
            for val in self.parse_dates:
                if isinstance(val, list):
                    for k in val:
                        noconvert_columns.add(_set(k))
                else:
                    noconvert_columns.add(_set(val))

        elif isinstance(self.parse_dates, dict):
            for val in self.parse_dates.values():
                if isinstance(val, list):
                    for k in val:
                        noconvert_columns.add(_set(k))
                else:
                    noconvert_columns.add(_set(val))

        elif self.parse_dates:
            if isinstance(self.index_col, list):
                for k in self.index_col:
                    noconvert_columns.add(_set(k))
            elif self.index_col is not None:
                noconvert_columns.add(_set(self.index_col))

        return noconvert_columns

    @final
    def _infer_types(
        self, values, na_values, no_dtype_specified, try_num_bool: bool = True
    ) -> tuple[ArrayLike, int]:
        """
        Infer types of values, possibly casting

        Parameters
        ----------
        values : ndarray
        na_values : set
        no_dtype_specified: Specifies if we want to cast explicitly
        try_num_bool : bool, default try
           try to cast values to numeric (first preference) or boolean

        Returns
        -------
        converted : ndarray or ExtensionArray
        na_count : int
        """
        na_count = 0
        if issubclass(values.dtype.type, (np.number, np.bool_)):
            # If our array has numeric dtype, we don't have to check for strings in isin
            na_values = np.array([val for val in na_values if not isinstance(val, str)])
            mask = algorithms.isin(values, na_values)
            na_count = mask.astype("uint8", copy=False).sum()
            if na_count > 0:
                if is_integer_dtype(values):
                    values = values.astype(np.float64)
                np.putmask(values, mask, np.nan)
            return values, na_count

        dtype_backend = self.dtype_backend
        non_default_dtype_backend = (
            no_dtype_specified and dtype_backend is not lib.no_default
        )
        result: ArrayLike

        if try_num_bool and is_object_dtype(values.dtype):
            # exclude e.g DatetimeIndex here
            try:
                result, result_mask = lib.maybe_convert_numeric(
                    values,
                    na_values,
                    False,
                    convert_to_masked_nullable=non_default_dtype_backend,  # type: ignore[arg-type]
                )
            except (ValueError, TypeError):
                # e.g. encountering datetime string gets ValueError
                #  TypeError can be raised in floatify
                na_count = parsers.sanitize_objects(values, na_values)
                result = values
            else:
                if non_default_dtype_backend:
                    if result_mask is None:
                        result_mask = np.zeros(result.shape, dtype=np.bool_)

                    if result_mask.all():
                        result = IntegerArray(
                            np.ones(result_mask.shape, dtype=np.int64), result_mask
                        )
                    elif is_integer_dtype(result):
                        result = IntegerArray(result, result_mask)
                    elif is_bool_dtype(result):
                        result = BooleanArray(result, result_mask)
                    elif is_float_dtype(result):
                        result = FloatingArray(result, result_mask)

                    na_count = result_mask.sum()
                else:
                    na_count = isna(result).sum()
        else:
            result = values
            if values.dtype == np.object_:
                na_count = parsers.sanitize_objects(values, na_values)

        if result.dtype == np.object_ and try_num_bool:
            result, bool_mask = libops.maybe_convert_bool(
                np.asarray(values),
                true_values=self.true_values,
                false_values=self.false_values,
                convert_to_masked_nullable=non_default_dtype_backend,  # type: ignore[arg-type]
            )
            if result.dtype == np.bool_ and non_default_dtype_backend:
                if bool_mask is None:
                    bool_mask = np.zeros(result.shape, dtype=np.bool_)
                result = BooleanArray(result, bool_mask)
            elif result.dtype == np.object_ and non_default_dtype_backend:
                # read_excel sends array of datetime objects
                if not lib.is_datetime_array(result, skipna=True):
                    dtype = StringDtype()
                    cls = dtype.construct_array_type()
                    result = cls._from_sequence(values, dtype=dtype)

        if dtype_backend == "pyarrow":
            pa = import_optional_dependency("pyarrow")
            if isinstance(result, np.ndarray):
                result = ArrowExtensionArray(pa.array(result, from_pandas=True))
            elif isinstance(result, BaseMaskedArray):
                if result._mask.all():
                    # We want an arrow null array here
                    result = ArrowExtensionArray(pa.array([None] * len(result)))
                else:
                    result = ArrowExtensionArray(
                        pa.array(result._data, mask=result._mask)
                    )
            else:
                result = ArrowExtensionArray(
                    pa.array(result.to_numpy(), from_pandas=True)
                )

        return result, na_count

    @final
    def _cast_types(self, values: ArrayLike, cast_type: DtypeObj, column) -> ArrayLike:
        """
        Cast values to specified type

        Parameters
        ----------
        values : ndarray or ExtensionArray
        cast_type : np.dtype or ExtensionDtype
           dtype to cast values to
        column : string
            column name - used only for error reporting

        Returns
        -------
        converted : ndarray or ExtensionArray
        """
        if isinstance(cast_type, CategoricalDtype):
            known_cats = cast_type.categories is not None

            if not is_object_dtype(values.dtype) and not known_cats:
                # TODO: this is for consistency with
                # c-parser which parses all categories
                # as strings
                values = lib.ensure_string_array(
                    values, skipna=False, convert_na_value=False
                )

            cats = Index(values).unique().dropna()
            values = Categorical._from_inferred_categories(
                cats, cats.get_indexer(values), cast_type, true_values=self.true_values
            )

        # use the EA's implementation of casting
        elif isinstance(cast_type, ExtensionDtype):
            array_type = cast_type.construct_array_type()
            try:
                if isinstance(cast_type, BooleanDtype):
                    # error: Unexpected keyword argument "true_values" for
                    # "_from_sequence_of_strings" of "ExtensionArray"
                    return array_type._from_sequence_of_strings(  # type: ignore[call-arg]
                        values,
                        dtype=cast_type,
                        true_values=self.true_values,
                        false_values=self.false_values,
                    )
                else:
                    return array_type._from_sequence_of_strings(values, dtype=cast_type)
            except NotImplementedError as err:
                raise NotImplementedError(
                    f"Extension Array: {array_type} must implement "
                    "_from_sequence_of_strings in order to be used in parser methods"
                ) from err

        elif isinstance(values, ExtensionArray):
            values = values.astype(cast_type, copy=False)
        elif issubclass(cast_type.type, str):
            # TODO: why skipna=True here and False above? some tests depend
            #  on it here, but nothing fails if we change it above
            #  (as no tests get there as of 2022-12-06)
            values = lib.ensure_string_array(
                values, skipna=True, convert_na_value=False
            )
        else:
            try:
                values = astype_array(values, cast_type, copy=True)
            except ValueError as err:
                raise ValueError(
                    f"Unable to convert column {column} to type {cast_type}"
                ) from err
        return values

    @overload
    def _do_date_conversions(
        self,
        names: Index,
        data: DataFrame,
    ) -> tuple[Sequence[Hashable] | Index, DataFrame]:
        ...

    @overload
    def _do_date_conversions(
        self,
        names: Sequence[Hashable],
        data: Mapping[Hashable, ArrayLike],
    ) -> tuple[Sequence[Hashable], Mapping[Hashable, ArrayLike]]:
        ...

    @final
    def _do_date_conversions(
        self,
        names: Sequence[Hashable] | Index,
        data: Mapping[Hashable, ArrayLike] | DataFrame,
    ) -> tuple[Sequence[Hashable] | Index, Mapping[Hashable, ArrayLike] | DataFrame]:
        # returns data, columns

        if self.parse_dates is not None:
            data, names = _process_date_conversion(
                data,
                self._date_conv,
                self.parse_dates,
                self.index_col,
                self.index_names,
                names,
                keep_date_col=self.keep_date_col,
                dtype_backend=self.dtype_backend,
            )

        return names, data

    @final
    def _check_data_length(
        self,
        columns: Sequence[Hashable],
        data: Sequence[ArrayLike],
    ) -> None:
        """Checks if length of data is equal to length of column names.

        One set of trailing commas is allowed. self.index_col not False
        results in a ParserError previously when lengths do not match.

        Parameters
        ----------
        columns: list of column names
        data: list of array-likes containing the data column-wise.
        """
        if not self.index_col and len(columns) != len(data) and columns:
            empty_str = is_object_dtype(data[-1]) and data[-1] == ""
            # error: No overload variant of "__ror__" of "ndarray" matches
            # argument type "ExtensionArray"
            empty_str_or_na = empty_str | isna(data[-1])  # type: ignore[operator]
            if len(columns) == len(data) - 1 and np.all(empty_str_or_na):
                return
            warnings.warn(
                "Length of header or names does not match length of data. This leads "
                "to a loss of data with index_col=False.",
                ParserWarning,
                stacklevel=find_stack_level(),
            )

    @overload
    def _evaluate_usecols(
        self,
        usecols: set[int] | Callable[[Hashable], object],
        names: Sequence[Hashable],
    ) -> set[int]:
        ...

    @overload
    def _evaluate_usecols(
        self, usecols: set[str], names: Sequence[Hashable]
    ) -> set[str]:
        ...

    @final
    def _evaluate_usecols(
        self,
        usecols: Callable[[Hashable], object] | set[str] | set[int],
        names: Sequence[Hashable],
    ) -> set[str] | set[int]:
        """
        Check whether or not the 'usecols' parameter
        is a callable.  If so, enumerates the 'names'
        parameter and returns a set of indices for
        each entry in 'names' that evaluates to True.
        If not a callable, returns 'usecols'.
        """
        if callable(usecols):
            return {i for i, name in enumerate(names) if usecols(name)}
        return usecols

    @final
    def _validate_usecols_names(self, usecols, names: Sequence):
        """
        Validates that all usecols are present in a given
        list of names. If not, raise a ValueError that
        shows what usecols are missing.

        Parameters
        ----------
        usecols : iterable of usecols
            The columns to validate are present in names.
        names : iterable of names
            The column names to check against.

        Returns
        -------
        usecols : iterable of usecols
            The `usecols` parameter if the validation succeeds.

        Raises
        ------
        ValueError : Columns were missing. Error message will list them.
        """
        missing = [c for c in usecols if c not in names]
        if len(missing) > 0:
            raise ValueError(
                f"Usecols do not match columns, columns expected but not found: "
                f"{missing}"
            )

        return usecols

    @final
    def _validate_usecols_arg(self, usecols):
        """
        Validate the 'usecols' parameter.

        Checks whether or not the 'usecols' parameter contains all integers
        (column selection by index), strings (column by name) or is a callable.
        Raises a ValueError if that is not the case.

        Parameters
        ----------
        usecols : list-like, callable, or None
            List of columns to use when parsing or a callable that can be used
            to filter a list of table columns.

        Returns
        -------
        usecols_tuple : tuple
            A tuple of (verified_usecols, usecols_dtype).

            'verified_usecols' is either a set if an array-like is passed in or
            'usecols' if a callable or None is passed in.

            'usecols_dtype` is the inferred dtype of 'usecols' if an array-like
            is passed in or None if a callable or None is passed in.
        """
        msg = (
            "'usecols' must either be list-like of all strings, all unicode, "
            "all integers or a callable."
        )
        if usecols is not None:
            if callable(usecols):
                return usecols, None

            if not is_list_like(usecols):
                # see gh-20529
                #
                # Ensure it is iterable container but not string.
                raise ValueError(msg)

            usecols_dtype = lib.infer_dtype(usecols, skipna=False)

            if usecols_dtype not in ("empty", "integer", "string"):
                raise ValueError(msg)

            usecols = set(usecols)

            return usecols, usecols_dtype
        return usecols, None

    @final
    def _clean_index_names(self, columns, index_col) -> tuple[list | None, list, list]:
        if not is_index_col(index_col):
            return None, columns, index_col

        columns = list(columns)

        # In case of no rows and multiindex columns we have to set index_names to
        # list of Nones GH#38292
        if not columns:
            return [None] * len(index_col), columns, index_col

        cp_cols = list(columns)
        index_names: list[str | int | None] = []

        # don't mutate
        index_col = list(index_col)

        for i, c in enumerate(index_col):
            if isinstance(c, str):
                index_names.append(c)
                for j, name in enumerate(cp_cols):
                    if name == c:
                        index_col[i] = j
                        columns.remove(name)
                        break
            else:
                name = cp_cols[c]
                columns.remove(name)
                index_names.append(name)

        # Only clean index names that were placeholders.
        for i, name in enumerate(index_names):
            if isinstance(name, str) and name in self.unnamed_cols:
                index_names[i] = None

        return index_names, columns, index_col

    @final
    def _get_empty_meta(self, columns, dtype: DtypeArg | None = None):
        columns = list(columns)

        index_col = self.index_col
        index_names = self.index_names

        # Convert `dtype` to a defaultdict of some kind.
        # This will enable us to write `dtype[col_name]`
        # without worrying about KeyError issues later on.
        dtype_dict: defaultdict[Hashable, Any]
        if not is_dict_like(dtype):
            # if dtype == None, default will be object.
            dtype_dict = defaultdict(lambda: dtype)
        else:
            dtype = cast(dict, dtype)
            dtype_dict = defaultdict(
                lambda: None,
                {columns[k] if is_integer(k) else k: v for k, v in dtype.items()},
            )

        # Even though we have no data, the "index" of the empty DataFrame
        # could for example still be an empty MultiIndex. Thus, we need to
        # check whether we have any index columns specified, via either:
        #
        # 1) index_col (column indices)
        # 2) index_names (column names)
        #
        # Both must be non-null to ensure a successful construction. Otherwise,
        # we have to create a generic empty Index.
        index: Index
        if (index_col is None or index_col is False) or index_names is None:
            index = default_index(0)
        else:
            # TODO: We could return default_index(0) if dtype_dict[name] is None
            data = [
                Index([], name=name, dtype=dtype_dict[name]) for name in index_names
            ]
            if len(data) == 1:
                index = data[0]
            else:
                index = MultiIndex.from_arrays(data)
            index_col.sort()

            for i, n in enumerate(index_col):
                columns.pop(n - i)

        col_dict = {
            col_name: Series([], dtype=dtype_dict[col_name]) for col_name in columns
        }

        return index, columns, col_dict


def _make_date_converter(
    date_parser=lib.no_default,
    dayfirst: bool = False,
    cache_dates: bool = True,
    date_format: dict[Hashable, str] | str | None = None,
):
    if date_parser is not lib.no_default:
        warnings.warn(
            "The argument 'date_parser' is deprecated and will "
            "be removed in a future version. "
            "Please use 'date_format' instead, or read your data in as 'object' dtype "
            "and then call 'to_datetime'.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
    if date_parser is not lib.no_default and date_format is not None:
        raise TypeError("Cannot use both 'date_parser' and 'date_format'")

    def unpack_if_single_element(arg):
        # NumPy 1.25 deprecation: https://github.com/numpy/numpy/pull/10615
        if isinstance(arg, np.ndarray) and arg.ndim == 1 and len(arg) == 1:
            return arg[0]
        return arg

    def converter(*date_cols, col: Hashable):
        if len(date_cols) == 1 and date_cols[0].dtype.kind in "Mm":
            return date_cols[0]

        if date_parser is lib.no_default:
            strs = parsing.concat_date_cols(date_cols)
            date_fmt = (
                date_format.get(col) if isinstance(date_format, dict) else date_format
            )

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    ".*parsing datetimes with mixed time zones will raise an error",
                    category=FutureWarning,
                )
                str_objs = ensure_object(strs)
                try:
                    result = tools.to_datetime(
                        str_objs,
                        format=date_fmt,
                        utc=False,
                        dayfirst=dayfirst,
                        cache=cache_dates,
                    )
                except (ValueError, TypeError):
                    # test_usecols_with_parse_dates4
                    return str_objs

            if isinstance(result, DatetimeIndex):
                arr = result.to_numpy()
                arr.flags.writeable = True
                return arr
            return result._values
        else:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        ".*parsing datetimes with mixed time zones "
                        "will raise an error",
                        category=FutureWarning,
                    )
                    pre_parsed = date_parser(
                        *(unpack_if_single_element(arg) for arg in date_cols)
                    )
                    try:
                        result = tools.to_datetime(
                            pre_parsed,
                            cache=cache_dates,
                        )
                    except (ValueError, TypeError):
                        # test_read_csv_with_custom_date_parser
                        result = pre_parsed
                if isinstance(result, datetime.datetime):
                    raise Exception("scalar parser")
                return result
            except Exception:
                # e.g. test_datetime_fractional_seconds
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        ".*parsing datetimes with mixed time zones "
                        "will raise an error",
                        category=FutureWarning,
                    )
                    pre_parsed = parsing.try_parse_dates(
                        parsing.concat_date_cols(date_cols),
                        parser=date_parser,
                    )
                    try:
                        return tools.to_datetime(pre_parsed)
                    except (ValueError, TypeError):
                        # TODO: not reached in tests 2023-10-27; needed?
                        return pre_parsed

    return converter


parser_defaults = {
    "delimiter": None,
    "escapechar": None,
    "quotechar": '"',
    "quoting": csv.QUOTE_MINIMAL,
    "doublequote": True,
    "skipinitialspace": False,
    "lineterminator": None,
    "header": "infer",
    "index_col": None,
    "names": None,
    "skiprows": None,
    "skipfooter": 0,
    "nrows": None,
    "na_values": None,
    "keep_default_na": True,
    "true_values": None,
    "false_values": None,
    "converters": None,
    "dtype": None,
    "cache_dates": True,
    "thousands": None,
    "comment": None,
    "decimal": ".",
    # 'engine': 'c',
    "parse_dates": False,
    "keep_date_col": False,
    "dayfirst": False,
    "date_parser": lib.no_default,
    "date_format": None,
    "usecols": None,
    # 'iterator': False,
    "chunksize": None,
    "verbose": False,
    "encoding": None,
    "compression": None,
    "skip_blank_lines": True,
    "encoding_errors": "strict",
    "on_bad_lines": ParserBase.BadLineHandleMethod.ERROR,
    "dtype_backend": lib.no_default,
}


def _process_date_conversion(
    data_dict,
    converter: Callable,
    parse_spec,
    index_col,
    index_names,
    columns,
    keep_date_col: bool = False,
    dtype_backend=lib.no_default,
):
    def _isindex(colspec):
        return (isinstance(index_col, list) and colspec in index_col) or (
            isinstance(index_names, list) and colspec in index_names
        )

    new_cols = []
    new_data = {}

    orig_names = columns
    columns = list(columns)

    date_cols = set()

    if parse_spec is None or isinstance(parse_spec, bool):
        return data_dict, columns

    if isinstance(parse_spec, list):
        # list of column lists
        for colspec in parse_spec:
            if is_scalar(colspec) or isinstance(colspec, tuple):
                if isinstance(colspec, int) and colspec not in data_dict:
                    colspec = orig_names[colspec]
                if _isindex(colspec):
                    continue
                elif dtype_backend == "pyarrow":
                    import pyarrow as pa

                    dtype = data_dict[colspec].dtype
                    if isinstance(dtype, ArrowDtype) and (
                        pa.types.is_timestamp(dtype.pyarrow_dtype)
                        or pa.types.is_date(dtype.pyarrow_dtype)
                    ):
                        continue

                # Pyarrow engine returns Series which we need to convert to
                # numpy array before converter, its a no-op for other parsers
                data_dict[colspec] = converter(
                    np.asarray(data_dict[colspec]), col=colspec
                )
            else:
                new_name, col, old_names = _try_convert_dates(
                    converter, colspec, data_dict, orig_names
                )
                if new_name in data_dict:
                    raise ValueError(f"New date column already in dict {new_name}")
                new_data[new_name] = col
                new_cols.append(new_name)
                date_cols.update(old_names)

    elif isinstance(parse_spec, dict):
        # dict of new name to column list
        for new_name, colspec in parse_spec.items():
            if new_name in data_dict:
                raise ValueError(f"Date column {new_name} already in dict")

            _, col, old_names = _try_convert_dates(
                converter,
                colspec,
                data_dict,
                orig_names,
                target_name=new_name,
            )

            new_data[new_name] = col

            # If original column can be converted to date we keep the converted values
            # This can only happen if values are from single column
            if len(colspec) == 1:
                new_data[colspec[0]] = col

            new_cols.append(new_name)
            date_cols.update(old_names)

    if isinstance(data_dict, DataFrame):
        data_dict = concat([DataFrame(new_data), data_dict], axis=1, copy=False)
    else:
        data_dict.update(new_data)
    new_cols.extend(columns)

    if not keep_date_col:
        for c in list(date_cols):
            data_dict.pop(c)
            new_cols.remove(c)

    return data_dict, new_cols


def _try_convert_dates(
    parser: Callable, colspec, data_dict, columns, target_name: str | None = None
):
    colset = set(columns)
    colnames = []

    for c in colspec:
        if c in colset:
            colnames.append(c)
        elif isinstance(c, int) and c not in columns:
            colnames.append(columns[c])
        else:
            colnames.append(c)

    new_name: tuple | str
    if all(isinstance(x, tuple) for x in colnames):
        new_name = tuple(map("_".join, zip(*colnames)))
    else:
        new_name = "_".join([str(x) for x in colnames])
    to_parse = [np.asarray(data_dict[c]) for c in colnames if c in data_dict]

    new_col = parser(*to_parse, col=new_name if target_name is None else target_name)
    return new_name, new_col, colnames


def _get_na_values(col, na_values, na_fvalues, keep_default_na: bool):
    """
    Get the NaN values for a given column.

    Parameters
    ----------
    col : str
        The name of the column.
    na_values : array-like, dict
        The object listing the NaN values as strings.
    na_fvalues : array-like, dict
        The object listing the NaN values as floats.
    keep_default_na : bool
        If `na_values` is a dict, and the column is not mapped in the
        dictionary, whether to return the default NaN values or the empty set.

    Returns
    -------
    nan_tuple : A length-two tuple composed of

        1) na_values : the string NaN values for that column.
        2) na_fvalues : the float NaN values for that column.
    """
    if isinstance(na_values, dict):
        if col in na_values:
            return na_values[col], na_fvalues[col]
        else:
            if keep_default_na:
                return STR_NA_VALUES, set()

            return set(), set()
    else:
        return na_values, na_fvalues


def _validate_parse_dates_arg(parse_dates):
    """
    Check whether or not the 'parse_dates' parameter
    is a non-boolean scalar. Raises a ValueError if
    that is the case.
    """
    msg = (
        "Only booleans, lists, and dictionaries are accepted "
        "for the 'parse_dates' parameter"
    )

    if not (
        parse_dates is None
        or lib.is_bool(parse_dates)
        or isinstance(parse_dates, (list, dict))
    ):
        raise TypeError(msg)

    return parse_dates


def is_index_col(col) -> bool:
    return col is not None and col is not False
<|endoftext|>from __future__ import annotations

from typing import TYPE_CHECKING
import warnings

from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.errors import (
    ParserError,
    ParserWarning,
)
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import pandas_dtype
from pandas.core.dtypes.inference import is_integer

from pandas.io._util import arrow_table_to_pandas
from pandas.io.parsers.base_parser import ParserBase

if TYPE_CHECKING:
    from pandas._typing import ReadBuffer

    from pandas import DataFrame


class ArrowParserWrapper(ParserBase):
    """
    Wrapper for the pyarrow engine for read_csv()
    """

    def __init__(self, src: ReadBuffer[bytes], **kwds) -> None:
        super().__init__(kwds)
        self.kwds = kwds
        self.src = src

        self._parse_kwds()

    def _parse_kwds(self) -> None:
        """
        Validates keywords before passing to pyarrow.
        """
        encoding: str | None = self.kwds.get("encoding")
        self.encoding = "utf-8" if encoding is None else encoding

        na_values = self.kwds["na_values"]
        if isinstance(na_values, dict):
            raise ValueError(
                "The pyarrow engine doesn't support passing a dict for na_values"
            )
        self.na_values = list(self.kwds["na_values"])

    def _get_pyarrow_options(self) -> None:
        """
        Rename some arguments to pass to pyarrow
        """
        mapping = {
            "usecols": "include_columns",
            "na_values": "null_values",
            "escapechar": "escape_char",
            "skip_blank_lines": "ignore_empty_lines",
            "decimal": "decimal_point",
            "quotechar": "quote_char",
        }
        for pandas_name, pyarrow_name in mapping.items():
            if pandas_name in self.kwds and self.kwds.get(pandas_name) is not None:
                self.kwds[pyarrow_name] = self.kwds.pop(pandas_name)

        # Date format handling
        # If we get a string, we need to convert it into a list for pyarrow
        # If we get a dict, we want to parse those separately
        date_format = self.date_format
        if isinstance(date_format, str):
            date_format = [date_format]
        else:
            # In case of dict, we don't want to propagate through, so
            # just set to pyarrow default of None

            # Ideally, in future we disable pyarrow dtype inference (read in as string)
            # to prevent misreads.
            date_format = None
        self.kwds["timestamp_parsers"] = date_format

        self.parse_options = {
            option_name: option_value
            for option_name, option_value in self.kwds.items()
            if option_value is not None
            and option_name
            in ("delimiter", "quote_char", "escape_char", "ignore_empty_lines")
        }

        on_bad_lines = self.kwds.get("on_bad_lines")
        if on_bad_lines is not None:
            if callable(on_bad_lines):
                self.parse_options["invalid_row_handler"] = on_bad_lines
            elif on_bad_lines == ParserBase.BadLineHandleMethod.ERROR:
                self.parse_options[
                    "invalid_row_handler"
                ] = None  # PyArrow raises an exception by default
            elif on_bad_lines == ParserBase.BadLineHandleMethod.WARN:

                def handle_warning(invalid_row) -> str:
                    warnings.warn(
                        f"Expected {invalid_row.expected_columns} columns, but found "
                        f"{invalid_row.actual_columns}: {invalid_row.text}",
                        ParserWarning,
                        stacklevel=find_stack_level(),
                    )
                    return "skip"

                self.parse_options["invalid_row_handler"] = handle_warning
            elif on_bad_lines == ParserBase.BadLineHandleMethod.SKIP:
                self.parse_options["invalid_row_handler"] = lambda _: "skip"

        self.convert_options = {
            option_name: option_value
            for option_name, option_value in self.kwds.items()
            if option_value is not None
            and option_name
            in (
                "include_columns",
                "null_values",
                "true_values",
                "false_values",
                "decimal_point",
                "timestamp_parsers",
            )
        }
        self.convert_options["strings_can_be_null"] = "" in self.kwds["null_values"]
        # autogenerated column names are prefixed with 'f' in pyarrow.csv
        if self.header is None and "include_columns" in self.convert_options:
            self.convert_options["include_columns"] = [
                f"f{n}" for n in self.convert_options["include_columns"]
            ]

        self.read_options = {
            "autogenerate_column_names": self.header is None,
            "skip_rows": self.header
            if self.header is not None
            else self.kwds["skiprows"],
            "encoding": self.encoding,
        }

    def _finalize_pandas_output(self, frame: DataFrame) -> DataFrame:
        """
        Processes data read in based on kwargs.

        Parameters
        ----------
        frame: DataFrame
            The DataFrame to process.

        Returns
        -------
        DataFrame
            The processed DataFrame.
        """
        num_cols = len(frame.columns)
        multi_index_named = True
        if self.header is None:
            if self.names is None:
                if self.header is None:
                    self.names = range(num_cols)
            if len(self.names) != num_cols:
                # usecols is passed through to pyarrow, we only handle index col here
                # The only way self.names is not the same length as number of cols is
                # if we have int index_col. We should just pad the names(they will get
                # removed anyways) to expected length then.
                columns_prefix = [str(x) for x in range(num_cols - len(self.names))]
                self.names = columns_prefix + self.names
                multi_index_named = False
            frame.columns = self.names
        # we only need the frame not the names
        _, frame = self._do_date_conversions(frame.columns, frame)
        if self.index_col is not None:
            index_to_set = self.index_col.copy()
            for i, item in enumerate(self.index_col):
                if is_integer(item):
                    index_to_set[i] = frame.columns[item]
                # String case
                elif item not in frame.columns:
                    raise ValueError(f"Index {item} invalid")

                # Process dtype for index_col and drop from dtypes
                if self.dtype is not None:
                    key, new_dtype = (
                        (item, self.dtype.get(item))
                        if self.dtype.get(item) is not None
                        else (frame.columns[item], self.dtype.get(frame.columns[item]))
                    )
                    if new_dtype is not None:
                        frame[key] = frame[key].astype(new_dtype)
                        del self.dtype[key]

            frame.set_index(index_to_set, drop=True, inplace=True)
            # Clear names if headerless and no name given
            if self.header is None and not multi_index_named:
                frame.index.names = [None] * len(frame.index.names)

        if self.dtype is not None:
            # Ignore non-existent columns from dtype mapping
            # like other parsers do
            if isinstance(self.dtype, dict):
                self.dtype = {
                    k: pandas_dtype(v)
                    for k, v in self.dtype.items()
                    if k in frame.columns
                }
            else:
                self.dtype = pandas_dtype(self.dtype)
            try:
                frame = frame.astype(self.dtype)
            except TypeError as e:
                # GH#44901 reraise to keep api consistent
                raise ValueError(e)
        return frame

    def _validate_usecols(self, usecols) -> None:
        if lib.is_list_like(usecols) and not all(isinstance(x, str) for x in usecols):
            raise ValueError(
                "The pyarrow engine does not allow 'usecols' to be integer "
                "column positions. Pass a list of string column names instead."
            )
        elif callable(usecols):
            raise ValueError(
                "The pyarrow engine does not allow 'usecols' to be a callable."
            )

    def read(self) -> DataFrame:
        """
        Reads the contents of a CSV file into a DataFrame and
        processes it according to the kwargs passed in the
        constructor.

        Returns
        -------
        DataFrame
            The DataFrame created from the CSV file.
        """
        pa = import_optional_dependency("pyarrow")
        pyarrow_csv = import_optional_dependency("pyarrow.csv")
        self._get_pyarrow_options()

        try:
            convert_options = pyarrow_csv.ConvertOptions(**self.convert_options)
        except TypeError:
            include = self.convert_options.get("include_columns", None)
            if include is not None:
                self._validate_usecols(include)

            nulls = self.convert_options.get("null_values", set())
            if not lib.is_list_like(nulls) or not all(
                isinstance(x, str) for x in nulls
            ):
                raise TypeError(
                    "The 'pyarrow' engine requires all na_values to be strings"
                )

            raise

        try:
            table = pyarrow_csv.read_csv(
                self.src,
                read_options=pyarrow_csv.ReadOptions(**self.read_options),
                parse_options=pyarrow_csv.ParseOptions(**self.parse_options),
                convert_options=convert_options,
            )
        except pa.ArrowInvalid as e:
            raise ParserError(e) from e

        dtype_backend = self.kwds["dtype_backend"]

        # Convert all pa.null() cols -> float64 (non nullable)
        # else Int64 (nullable case, see below)
        if dtype_backend is lib.no_default:
            new_schema = table.schema
            new_type = pa.float64()
            for i, arrow_type in enumerate(table.schema.types):
                if pa.types.is_null(arrow_type):
                    new_schema = new_schema.set(
                        i, new_schema.field(i).with_type(new_type)
                    )

            table = table.cast(new_schema)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                "make_block is deprecated",
                DeprecationWarning,
            )
            frame = arrow_table_to_pandas(
                table, dtype_backend=dtype_backend, null_to_int64=True
            )

        return self._finalize_pandas_output(frame)
<|endoftext|>"""
Module contains tools for processing files into DataFrames or other objects

GH#48849 provides a convenient way of deprecating keyword arguments
"""
from __future__ import annotations

from collections import (
    abc,
    defaultdict,
)
import csv
import sys
from textwrap import fill
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    NamedTuple,
    TypedDict,
    overload,
)
import warnings

import numpy as np

from pandas._config import using_copy_on_write

from pandas._libs import lib
from pandas._libs.parsers import STR_NA_VALUES
from pandas.errors import (
    AbstractMethodError,
    ParserWarning,
)
from pandas.util._decorators import Appender
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend

from pandas.core.dtypes.common import (
    is_file_like,
    is_float,
    is_hashable,
    is_integer,
    is_list_like,
    pandas_dtype,
)

from pandas import Series
from pandas.core.frame import DataFrame
from pandas.core.indexes.api import RangeIndex
from pandas.core.shared_docs import _shared_docs

from pandas.io.common import (
    IOHandles,
    get_handle,
    stringify_path,
    validate_header_arg,
)
from pandas.io.parsers.arrow_parser_wrapper import ArrowParserWrapper
from pandas.io.parsers.base_parser import (
    ParserBase,
    is_index_col,
    parser_defaults,
)
from pandas.io.parsers.c_parser_wrapper import CParserWrapper
from pandas.io.parsers.python_parser import (
    FixedWidthFieldParser,
    PythonParser,
)

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Iterable,
        Mapping,
        Sequence,
    )
    from types import TracebackType

    from pandas._typing import (
        CompressionOptions,
        CSVEngine,
        DtypeArg,
        DtypeBackend,
        FilePath,
        IndexLabel,
        ReadCsvBuffer,
        Self,
        StorageOptions,
        UsecolsArgType,
    )
_doc_read_csv_and_table = (
    r"""
{summary}

Also supports optionally iterating or breaking of the file
into chunks.

Additional help can be found in the online docs for
`IO Tools <https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html>`_.

Parameters
----------
filepath_or_buffer : str, path object or file-like object
    Any valid string path is acceptable. The string could be a URL. Valid
    URL schemes include http, ftp, s3, gs, and file. For file URLs, a host is
    expected. A local file could be: file://localhost/path/to/table.csv.

    If you want to pass in a path object, pandas accepts any ``os.PathLike``.

    By file-like object, we refer to objects with a ``read()`` method, such as
    a file handle (e.g. via builtin ``open`` function) or ``StringIO``.
sep : str, default {_default_sep}
    Character or regex pattern to treat as the delimiter. If ``sep=None``, the
    C engine cannot automatically detect
    the separator, but the Python parsing engine can, meaning the latter will
    be used and automatically detect the separator from only the first valid
    row of the file by Python's builtin sniffer tool, ``csv.Sniffer``.
    In addition, separators longer than 1 character and different from
    ``'\s+'`` will be interpreted as regular expressions and will also force
    the use of the Python parsing engine. Note that regex delimiters are prone
    to ignoring quoted data. Regex example: ``'\r\t'``.
delimiter : str, optional
    Alias for ``sep``.
header : int, Sequence of int, 'infer' or None, default 'infer'
    Row number(s) containing column labels and marking the start of the
    data (zero-indexed). Default behavior is to infer the column names: if no ``names``
    are passed the behavior is identical to ``header=0`` and column
    names are inferred from the first line of the file, if column
    names are passed explicitly to ``names`` then the behavior is identical to
    ``header=None``. Explicitly pass ``header=0`` to be able to
    replace existing names. The header can be a list of integers that
    specify row locations for a :class:`~pandas.MultiIndex` on the columns
    e.g. ``[0, 1, 3]``. Intervening rows that are not specified will be
    skipped (e.g. 2 in this example is skipped). Note that this
    parameter ignores commented lines and empty lines if
    ``skip_blank_lines=True``, so ``header=0`` denotes the first line of
    data rather than the first line of the file.
names : Sequence of Hashable, optional
    Sequence of column labels to apply. If the file contains a header row,
    then you should explicitly pass ``header=0`` to override the column names.
    Duplicates in this list are not allowed.
index_col : Hashable, Sequence of Hashable or False, optional
  Column(s) to use as row label(s), denoted either by column labels or column
  indices.  If a sequence of labels or indices is given, :class:`~pandas.MultiIndex`
  will be formed for the row labels.

  Note: ``index_col=False`` can be used to force pandas to *not* use the first
  column as the index, e.g., when you have a malformed file with delimiters at
  the end of each line.
usecols : Sequence of Hashable or Callable, optional
    Subset of columns to select, denoted either by column labels or column indices.
    If list-like, all elements must either
    be positional (i.e. integer indices into the document columns) or strings
    that correspond to column names provided either by the user in ``names`` or
    inferred from the document header row(s). If ``names`` are given, the document
    header row(s) are not taken into account. For example, a valid list-like
    ``usecols`` parameter would be ``[0, 1, 2]`` or ``['foo', 'bar', 'baz']``.
    Element order is ignored, so ``usecols=[0, 1]`` is the same as ``[1, 0]``.
    To instantiate a :class:`~pandas.DataFrame` from ``data`` with element order
    preserved use ``pd.read_csv(data, usecols=['foo', 'bar'])[['foo', 'bar']]``
    for columns in ``['foo', 'bar']`` order or
    ``pd.read_csv(data, usecols=['foo', 'bar'])[['bar', 'foo']]``
    for ``['bar', 'foo']`` order.

    If callable, the callable function will be evaluated against the column
    names, returning names where the callable function evaluates to ``True``. An
    example of a valid callable argument would be ``lambda x: x.upper() in
    ['AAA', 'BBB', 'DDD']``. Using this parameter results in much faster
    parsing time and lower memory usage.
dtype : dtype or dict of {{Hashable : dtype}}, optional
    Data type(s) to apply to either the whole dataset or individual columns.
    E.g., ``{{'a': np.float64, 'b': np.int32, 'c': 'Int64'}}``
    Use ``str`` or ``object`` together with suitable ``na_values`` settings
    to preserve and not interpret ``dtype``.
    If ``converters`` are specified, they will be applied INSTEAD
    of ``dtype`` conversion.

    .. versionadded:: 1.5.0

        Support for ``defaultdict`` was added. Specify a ``defaultdict`` as input where
        the default determines the ``dtype`` of the columns which are not explicitly
        listed.
engine : {{'c', 'python', 'pyarrow'}}, optional
    Parser engine to use. The C and pyarrow engines are faster, while the python engine
    is currently more feature-complete. Multithreading is currently only supported by
    the pyarrow engine.

    .. versionadded:: 1.4.0

        The 'pyarrow' engine was added as an *experimental* engine, and some features
        are unsupported, or may not work correctly, with this engine.
converters : dict of {{Hashable : Callable}}, optional
    Functions for converting values in specified columns. Keys can either
    be column labels or column indices.
true_values : list, optional
    Values to consider as ``True`` in addition to case-insensitive variants of 'True'.
false_values : list, optional
    Values to consider as ``False`` in addition to case-insensitive variants of 'False'.
skipinitialspace : bool, default False
    Skip spaces after delimiter.
skiprows : int, list of int or Callable, optional
    Line numbers to skip (0-indexed) or number of lines to skip (``int``)
    at the start of the file.

    If callable, the callable function will be evaluated against the row
    indices, returning ``True`` if the row should be skipped and ``False`` otherwise.
    An example of a valid callable argument would be ``lambda x: x in [0, 2]``.
skipfooter : int, default 0
    Number of lines at bottom of file to skip (Unsupported with ``engine='c'``).
nrows : int, optional
    Number of rows of file to read. Useful for reading pieces of large files.
na_values : Hashable, Iterable of Hashable or dict of {{Hashable : Iterable}}, optional
    Additional strings to recognize as ``NA``/``NaN``. If ``dict`` passed, specific
    per-column ``NA`` values.  By default the following values are interpreted as
    ``NaN``: " """
    + fill('", "'.join(sorted(STR_NA_VALUES)), 70, subsequent_indent="    ")
    + """ ".

keep_default_na : bool, default True
    Whether or not to include the default ``NaN`` values when parsing the data.
    Depending on whether ``na_values`` is passed in, the behavior is as follows:

    * If ``keep_default_na`` is ``True``, and ``na_values`` are specified, ``na_values``
      is appended to the default ``NaN`` values used for parsing.
    * If ``keep_default_na`` is ``True``, and ``na_values`` are not specified, only
      the default ``NaN`` values are used for parsing.
    * If ``keep_default_na`` is ``False``, and ``na_values`` are specified, only
      the ``NaN`` values specified ``na_values`` are used for parsing.
    * If ``keep_default_na`` is ``False``, and ``na_values`` are not specified, no
      strings will be parsed as ``NaN``.

    Note that if ``na_filter`` is passed in as ``False``, the ``keep_default_na`` and
    ``na_values`` parameters will be ignored.
na_filter : bool, default True
    Detect missing value markers (empty strings and the value of ``na_values``). In
    data without any ``NA`` values, passing ``na_filter=False`` can improve the
    performance of reading a large file.
verbose : bool, default False
    Indicate number of ``NA`` values placed in non-numeric columns.

    .. deprecated:: 2.2.0
skip_blank_lines : bool, default True
    If ``True``, skip over blank lines rather than interpreting as ``NaN`` values.
parse_dates : bool, list of Hashable, list of lists or dict of {{Hashable : list}}, \
default False
    The behavior is as follows:

    * ``bool``. If ``True`` -> try parsing the index. Note: Automatically set to
      ``True`` if ``date_format`` or ``date_parser`` arguments have been passed.
    * ``list`` of ``int`` or names. e.g. If ``[1, 2, 3]`` -> try parsing columns 1, 2, 3
      each as a separate date column.
    * ``list`` of ``list``. e.g.  If ``[[1, 3]]`` -> combine columns 1 and 3 and parse
      as a single date column. Values are joined with a space before parsing.
    * ``dict``, e.g. ``{{'foo' : [1, 3]}}`` -> parse columns 1, 3 as date and call
      result 'foo'. Values are joined with a space before parsing.

    If a column or index cannot be represented as an array of ``datetime``,
    say because of an unparsable value or a mixture of timezones, the column
    or index will be returned unaltered as an ``object`` data type. For
    non-standard ``datetime`` parsing, use :func:`~pandas.to_datetime` after
    :func:`~pandas.read_csv`.

    Note: A fast-path exists for iso8601-formatted dates.
infer_datetime_format : bool, default False
    If ``True`` and ``parse_dates`` is enabled, pandas will attempt to infer the
    format of the ``datetime`` strings in the columns, and if it can be inferred,
    switch to a faster method of parsing them. In some cases this can increase
    the parsing speed by 5-10x.

    .. deprecated:: 2.0.0
        A strict version of this argument is now the default, passing it has no effect.

keep_date_col : bool, default False
    If ``True`` and ``parse_dates`` specifies combining multiple columns then
    keep the original columns.
date_parser : Callable, optional
    Function to use for converting a sequence of string columns to an array of
    ``datetime`` instances. The default uses ``dateutil.parser.parser`` to do the
    conversion. pandas will try to call ``date_parser`` in three different ways,
    advancing to the next if an exception occurs: 1) Pass one or more arrays
    (as defined by ``parse_dates``) as arguments; 2) concatenate (row-wise) the
    string values from the columns defined by ``parse_dates`` into a single array
    and pass that; and 3) call ``date_parser`` once for each row using one or
    more strings (corresponding to the columns defined by ``parse_dates``) as
    arguments.

    .. deprecated:: 2.0.0
       Use ``date_format`` instead, or read in as ``object`` and then apply
       :func:`~pandas.to_datetime` as-needed.
date_format : str or dict of column -> format, optional
    Format to use for parsing dates when used in conjunction with ``parse_dates``.
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

    .. versionadded:: 2.0.0
dayfirst : bool, default False
    DD/MM format dates, international and European format.
cache_dates : bool, default True
    If ``True``, use a cache of unique, converted dates to apply the ``datetime``
    conversion. May produce significant speed-up when parsing duplicate
    date strings, especially ones with timezone offsets.

iterator : bool, default False
    Return ``TextFileReader`` object for iteration or getting chunks with
    ``get_chunk()``.
chunksize : int, optional
    Number of lines to read from the file per chunk. Passing a value will cause the
    function to return a ``TextFileReader`` object for iteration.
    See the `IO Tools docs
    <https://pandas.pydata.org/pandas-docs/stable/io.html#io-chunking>`_
    for more information on ``iterator`` and ``chunksize``.

{decompression_options}

    .. versionchanged:: 1.4.0 Zstandard support.

thousands : str (length 1), optional
    Character acting as the thousands separator in numerical values.
decimal : str (length 1), default '.'
    Character to recognize as decimal point (e.g., use ',' for European data).
lineterminator : str (length 1), optional
    Character used to denote a line break. Only valid with C parser.
quotechar : str (length 1), optional
    Character used to denote the start and end of a quoted item. Quoted
    items can include the ``delimiter`` and it will be ignored.
quoting : {{0 or csv.QUOTE_MINIMAL, 1 or csv.QUOTE_ALL, 2 or csv.QUOTE_NONNUMERIC, \
3 or csv.QUOTE_NONE}}, default csv.QUOTE_MINIMAL
    Control field quoting behavior per ``csv.QUOTE_*`` constants. Default is
    ``csv.QUOTE_MINIMAL`` (i.e., 0) which implies that only fields containing special
    characters are quoted (e.g., characters defined in ``quotechar``, ``delimiter``,
    or ``lineterminator``.
doublequote : bool, default True
   When ``quotechar`` is specified and ``quoting`` is not ``QUOTE_NONE``, indicate
   whether or not to interpret two consecutive ``quotechar`` elements INSIDE a
   field as a single ``quotechar`` element.
escapechar : str (length 1), optional
    Character used to escape other characters.
comment : str (length 1), optional
    Character indicating that the remainder of line should not be parsed.
    If found at the beginning
    of a line, the line will be ignored altogether. This parameter must be a
    single character. Like empty lines (as long as ``skip_blank_lines=True``),
    fully commented lines are ignored by the parameter ``header`` but not by
    ``skiprows``. For example, if ``comment='#'``, parsing
    ``#empty\\na,b,c\\n1,2,3`` with ``header=0`` will result in ``'a,b,c'`` being
    treated as the header.
encoding : str, optional, default 'utf-8'
    Encoding to use for UTF when reading/writing (ex. ``'utf-8'``). `List of Python
    standard encodings
    <https://docs.python.org/3/library/codecs.html#standard-encodings>`_ .

encoding_errors : str, optional, default 'strict'
    How encoding errors are treated. `List of possible values
    <https://docs.python.org/3/library/codecs.html#error-handlers>`_ .

    .. versionadded:: 1.3.0

dialect : str or csv.Dialect, optional
    If provided, this parameter will override values (default or not) for the
    following parameters: ``delimiter``, ``doublequote``, ``escapechar``,
    ``skipinitialspace``, ``quotechar``, and ``quoting``. If it is necessary to
    override values, a ``ParserWarning`` will be issued. See ``csv.Dialect``
    documentation for more details.
on_bad_lines : {{'error', 'warn', 'skip'}} or Callable, default 'error'
    Specifies what to do upon encountering a bad line (a line with too many fields).
    Allowed values are :

    - ``'error'``, raise an Exception when a bad line is encountered.
    - ``'warn'``, raise a warning when a bad line is encountered and skip that line.
    - ``'skip'``, skip bad lines without raising or warning when they are encountered.

    .. versionadded:: 1.3.0

    .. versionadded:: 1.4.0

        - Callable, function with signature
          ``(bad_line: list[str]) -> list[str] | None`` that will process a single
          bad line. ``bad_line`` is a list of strings split by the ``sep``.
          If the function returns ``None``, the bad line will be ignored.
          If the function returns a new ``list`` of strings with more elements than
          expected, a ``ParserWarning`` will be emitted while dropping extra elements.
          Only supported when ``engine='python'``

    .. versionchanged:: 2.2.0

        - Callable, function with signature
          as described in `pyarrow documentation
          <https://arrow.apache.org/docs/python/generated/pyarrow.csv.ParseOptions.html
          #pyarrow.csv.ParseOptions.invalid_row_handler>`_ when ``engine='pyarrow'``

delim_whitespace : bool, default False
    Specifies whether or not whitespace (e.g. ``' '`` or ``'\\t'``) will be
    used as the ``sep`` delimiter. Equivalent to setting ``sep='\\s+'``. If this option
    is set to ``True``, nothing should be passed in for the ``delimiter``
    parameter.

    .. deprecated:: 2.2.0
        Use ``sep="\\s+"`` instead.
low_memory : bool, default True
    Internally process the file in chunks, resulting in lower memory use
    while parsing, but possibly mixed type inference.  To ensure no mixed
    types either set ``False``, or specify the type with the ``dtype`` parameter.
    Note that the entire file is read into a single :class:`~pandas.DataFrame`
    regardless, use the ``chunksize`` or ``iterator`` parameter to return the data in
    chunks. (Only valid with C parser).
memory_map : bool, default False
    If a filepath is provided for ``filepath_or_buffer``, map the file object
    directly onto memory and access the data directly from there. Using this
    option can improve performance because there is no longer any I/O overhead.
float_precision : {{'high', 'legacy', 'round_trip'}}, optional
    Specifies which converter the C engine should use for floating-point
    values. The options are ``None`` or ``'high'`` for the ordinary converter,
    ``'legacy'`` for the original lower precision pandas converter, and
    ``'round_trip'`` for the round-trip converter.

{storage_options}

dtype_backend : {{'numpy_nullable', 'pyarrow'}}, default 'numpy_nullable'
    Back-end data type applied to the resultant :class:`DataFrame`
    (still experimental). Behaviour is as follows:

    * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
      (default).
    * ``"pyarrow"``: returns pyarrow-backed nullable :class:`ArrowDtype`
      DataFrame.

    .. versionadded:: 2.0

Returns
-------
DataFrame or TextFileReader
    A comma-separated values (csv) file is returned as two-dimensional
    data structure with labeled axes.

See Also
--------
DataFrame.to_csv : Write DataFrame to a comma-separated values (csv) file.
{see_also_func_name} : {see_also_func_summary}
read_fwf : Read a table of fixed-width formatted lines into DataFrame.

Examples
--------
>>> pd.{func_name}('data.csv')  # doctest: +SKIP
"""
)


class _C_Parser_Defaults(TypedDict):
    delim_whitespace: Literal[False]
    na_filter: Literal[True]
    low_memory: Literal[True]
    memory_map: Literal[False]
    float_precision: None


_c_parser_defaults: _C_Parser_Defaults = {
    "delim_whitespace": False,
    "na_filter": True,
    "low_memory": True,
    "memory_map": False,
    "float_precision": None,
}


class _Fwf_Defaults(TypedDict):
    colspecs: Literal["infer"]
    infer_nrows: Literal[100]
    widths: None


_fwf_defaults: _Fwf_Defaults = {"colspecs": "infer", "infer_nrows": 100, "widths": None}
_c_unsupported = {"skipfooter"}
_python_unsupported = {"low_memory", "float_precision"}
_pyarrow_unsupported = {
    "skipfooter",
    "float_precision",
    "chunksize",
    "comment",
    "nrows",
    "thousands",
    "memory_map",
    "dialect",
    "delim_whitespace",
    "quoting",
    "lineterminator",
    "converters",
    "iterator",
    "dayfirst",
    "verbose",
    "skipinitialspace",
    "low_memory",
}


class _DeprecationConfig(NamedTuple):
    default_value: Any
    msg: str | None


@overload
def validate_integer(name: str, val: None, min_val: int = ...) -> None:
    ...


@overload
def validate_integer(name: str, val: float, min_val: int = ...) -> int:
    ...


@overload
def validate_integer(name: str, val: int | None, min_val: int = ...) -> int | None:
    ...


def validate_integer(
    name: str, val: int | float | None, min_val: int = 0
) -> int | None:
    """
    Checks whether the 'name' parameter for parsing is either
    an integer OR float that can SAFELY be cast to an integer
    without losing accuracy. Raises a ValueError if that is
    not the case.

    Parameters
    ----------
    name : str
        Parameter name (used for error reporting)
    val : int or float
        The value to check
    min_val : int
        Minimum allowed value (val < min_val will result in a ValueError)
    """
    if val is None:
        return val

    msg = f"'{name:s}' must be an integer >={min_val:d}"
    if is_float(val):
        if int(val) != val:
            raise ValueError(msg)
        val = int(val)
    elif not (is_integer(val) and val >= min_val):
        raise ValueError(msg)

    return int(val)


def _validate_names(names: Sequence[Hashable] | None) -> None:
    """
    Raise ValueError if the `names` parameter contains duplicates or has an
    invalid data type.

    Parameters
    ----------
    names : array-like or None
        An array containing a list of the names used for the output DataFrame.

    Raises
    ------
    ValueError
        If names are not unique or are not ordered (e.g. set).
    """
    if names is not None:
        if len(names) != len(set(names)):
            raise ValueError("Duplicate names are not allowed.")
        if not (
            is_list_like(names, allow_sets=False) or isinstance(names, abc.KeysView)
        ):
            raise ValueError("Names should be an ordered collection.")


def _read(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str], kwds
) -> DataFrame | TextFileReader:
    """Generic reader of line files."""
    # if we pass a date_parser and parse_dates=False, we should not parse the
    # dates GH#44366
    if kwds.get("parse_dates", None) is None:
        if (
            kwds.get("date_parser", lib.no_default) is lib.no_default
            and kwds.get("date_format", None) is None
        ):
            kwds["parse_dates"] = False
        else:
            kwds["parse_dates"] = True

    # Extract some of the arguments (pass chunksize on).
    iterator = kwds.get("iterator", False)
    chunksize = kwds.get("chunksize", None)
    if kwds.get("engine") == "pyarrow":
        if iterator:
            raise ValueError(
                "The 'iterator' option is not supported with the 'pyarrow' engine"
            )

        if chunksize is not None:
            raise ValueError(
                "The 'chunksize' option is not supported with the 'pyarrow' engine"
            )
    else:
        chunksize = validate_integer("chunksize", chunksize, 1)

    nrows = kwds.get("nrows", None)

    # Check for duplicates in names.
    _validate_names(kwds.get("names", None))

    # Create the parser.
    parser = TextFileReader(filepath_or_buffer, **kwds)

    if chunksize or iterator:
        return parser

    with parser:
        return parser.read(nrows)


# iterator=True -> TextFileReader
@overload
def read_csv(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    sep: str | None | lib.NoDefault = ...,
    delimiter: str | None | lib.NoDefault = ...,
    header: int | Sequence[int] | None | Literal["infer"] = ...,
    names: Sequence[Hashable] | None | lib.NoDefault = ...,
    index_col: IndexLabel | Literal[False] | None = ...,
    usecols: UsecolsArgType = ...,
    dtype: DtypeArg | None = ...,
    engine: CSVEngine | None = ...,
    converters: Mapping[Hashable, Callable] | None = ...,
    true_values: list | None = ...,
    false_values: list | None = ...,
    skipinitialspace: bool = ...,
    skiprows: list[int] | int | Callable[[Hashable], bool] | None = ...,
    skipfooter: int = ...,
    nrows: int | None = ...,
    na_values: Hashable
    | Iterable[Hashable]
    | Mapping[Hashable, Iterable[Hashable]]
    | None = ...,
    na_filter: bool = ...,
    verbose: bool | lib.NoDefault = ...,
    skip_blank_lines: bool = ...,
    parse_dates: bool | Sequence[Hashable] | None = ...,
    infer_datetime_format: bool | lib.NoDefault = ...,
    keep_date_col: bool | lib.NoDefault = ...,
    date_parser: Callable | lib.NoDefault = ...,
    date_format: str | dict[Hashable, str] | None = ...,
    dayfirst: bool = ...,
    cache_dates: bool = ...,
    iterator: Literal[True],
    chunksize: int | None = ...,
    compression: CompressionOptions = ...,
    thousands: str | None = ...,
    decimal: str = ...,
    lineterminator: str | None = ...,
    quotechar: str = ...,
    quoting: int = ...,
    doublequote: bool = ...,
    escapechar: str | None = ...,
    comment: str | None = ...,
    encoding: str | None = ...,
    encoding_errors: str | None = ...,
    dialect: str | csv.Dialect | None = ...,
    on_bad_lines=...,
    delim_whitespace: bool | lib.NoDefault = ...,
    low_memory: bool = ...,
    memory_map: bool = ...,
    float_precision: Literal["high", "legacy"] | None = ...,
    storage_options: StorageOptions = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
) -> TextFileReader:
    ...


# chunksize=int -> TextFileReader
@overload
def read_csv(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    sep: str | None | lib.NoDefault = ...,
    delimiter: str | None | lib.NoDefault = ...,
    header: int | Sequence[int] | None | Literal["infer"] = ...,
    names: Sequence[Hashable] | None | lib.NoDefault = ...,
    index_col: IndexLabel | Literal[False] | None = ...,
    usecols: UsecolsArgType = ...,
    dtype: DtypeArg | None = ...,
    engine: CSVEngine | None = ...,
    converters: Mapping[Hashable, Callable] | None = ...,
    true_values: list | None = ...,
    false_values: list | None = ...,
    skipinitialspace: bool = ...,
    skiprows: list[int] | int | Callable[[Hashable], bool] | None = ...,
    skipfooter: int = ...,
    nrows: int | None = ...,
    na_values: Hashable
    | Iterable[Hashable]
    | Mapping[Hashable, Iterable[Hashable]]
    | None = ...,
    keep_default_na: bool = ...,
    na_filter: bool = ...,
    verbose: bool | lib.NoDefault = ...,
    skip_blank_lines: bool = ...,
    parse_dates: bool | Sequence[Hashable] | None = ...,
    infer_datetime_format: bool | lib.NoDefault = ...,
    keep_date_col: bool | lib.NoDefault = ...,
    date_parser: Callable | lib.NoDefault = ...,
    date_format: str | dict[Hashable, str] | None = ...,
    dayfirst: bool = ...,
    cache_dates: bool = ...,
    iterator: bool = ...,
    chunksize: int,
    compression: CompressionOptions = ...,
    thousands: str | None = ...,
    decimal: str = ...,
    lineterminator: str | None = ...,
    quotechar: str = ...,
    quoting: int = ...,
    doublequote: bool = ...,
    escapechar: str | None = ...,
    comment: str | None = ...,
    encoding: str | None = ...,
    encoding_errors: str | None = ...,
    dialect: str | csv.Dialect | None = ...,
    on_bad_lines=...,
    delim_whitespace: bool | lib.NoDefault = ...,
    low_memory: bool = ...,
    memory_map: bool = ...,
    float_precision: Literal["high", "legacy"] | None = ...,
    storage_options: StorageOptions = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
) -> TextFileReader:
    ...


# default case -> DataFrame
@overload
def read_csv(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    sep: str | None | lib.NoDefault = ...,
    delimiter: str | None | lib.NoDefault = ...,
    header: int | Sequence[int] | None | Literal["infer"] = ...,
    names: Sequence[Hashable] | None | lib.NoDefault = ...,
    index_col: IndexLabel | Literal[False] | None = ...,
    usecols: UsecolsArgType = ...,
    dtype: DtypeArg | None = ...,
    engine: CSVEngine | None = ...,
    converters: Mapping[Hashable, Callable] | None = ...,
    true_values: list | None = ...,
    false_values: list | None = ...,
    skipinitialspace: bool = ...,
    skiprows: list[int] | int | Callable[[Hashable], bool] | None = ...,
    skipfooter: int = ...,
    nrows: int | None = ...,
    na_values: Hashable
    | Iterable[Hashable]
    | Mapping[Hashable, Iterable[Hashable]]
    | None = ...,
    keep_default_na: bool = ...,
    na_filter: bool = ...,
    verbose: bool | lib.NoDefault = ...,
    skip_blank_lines: bool = ...,
    parse_dates: bool | Sequence[Hashable] | None = ...,
    infer_datetime_format: bool | lib.NoDefault = ...,
    keep_date_col: bool | lib.NoDefault = ...,
    date_parser: Callable | lib.NoDefault = ...,
    date_format: str | dict[Hashable, str] | None = ...,
    dayfirst: bool = ...,
    cache_dates: bool = ...,
    iterator: Literal[False] = ...,
    chunksize: None = ...,
    compression: CompressionOptions = ...,
    thousands: str | None = ...,
    decimal: str = ...,
    lineterminator: str | None = ...,
    quotechar: str = ...,
    quoting: int = ...,
    doublequote: bool = ...,
    escapechar: str | None = ...,
    comment: str | None = ...,
    encoding: str | None = ...,
    encoding_errors: str | None = ...,
    dialect: str | csv.Dialect | None = ...,
    on_bad_lines=...,
    delim_whitespace: bool | lib.NoDefault = ...,
    low_memory: bool = ...,
    memory_map: bool = ...,
    float_precision: Literal["high", "legacy"] | None = ...,
    storage_options: StorageOptions = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
) -> DataFrame:
    ...


# Unions -> DataFrame | TextFileReader
@overload
def read_csv(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    sep: str | None | lib.NoDefault = ...,
    delimiter: str | None | lib.NoDefault = ...,
    header: int | Sequence[int] | None | Literal["infer"] = ...,
    names: Sequence[Hashable] | None | lib.NoDefault = ...,
    index_col: IndexLabel | Literal[False] | None = ...,
    usecols: UsecolsArgType = ...,
    dtype: DtypeArg | None = ...,
    engine: CSVEngine | None = ...,
    converters: Mapping[Hashable, Callable] | None = ...,
    true_values: list | None = ...,
    false_values: list | None = ...,
    skipinitialspace: bool = ...,
    skiprows: list[int] | int | Callable[[Hashable], bool] | None = ...,
    skipfooter: int = ...,
    nrows: int | None = ...,
    na_values: Hashable
    | Iterable[Hashable]
    | Mapping[Hashable, Iterable[Hashable]]
    | None = ...,
    keep_default_na: bool = ...,
    na_filter: bool = ...,
    verbose: bool | lib.NoDefault = ...,
    skip_blank_lines: bool = ...,
    parse_dates: bool | Sequence[Hashable] | None = ...,
    infer_datetime_format: bool | lib.NoDefault = ...,
    keep_date_col: bool | lib.NoDefault = ...,
    date_parser: Callable | lib.NoDefault = ...,
    date_format: str | dict[Hashable, str] | None = ...,
    dayfirst: bool = ...,
    cache_dates: bool = ...,
    iterator: bool = ...,
    chunksize: int | None = ...,
    compression: CompressionOptions = ...,
    thousands: str | None = ...,
    decimal: str = ...,
    lineterminator: str | None = ...,
    quotechar: str = ...,
    quoting: int = ...,
    doublequote: bool = ...,
    escapechar: str | None = ...,
    comment: str | None = ...,
    encoding: str | None = ...,
    encoding_errors: str | None = ...,
    dialect: str | csv.Dialect | None = ...,
    on_bad_lines=...,
    delim_whitespace: bool | lib.NoDefault = ...,
    low_memory: bool = ...,
    memory_map: bool = ...,
    float_precision: Literal["high", "legacy"] | None = ...,
    storage_options: StorageOptions = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
) -> DataFrame | TextFileReader:
    ...


@Appender(
    _doc_read_csv_and_table.format(
        func_name="read_csv",
        summary="Read a comma-separated values (csv) file into DataFrame.",
        see_also_func_name="read_table",
        see_also_func_summary="Read general delimited file into DataFrame.",
        _default_sep="','",
        storage_options=_shared_docs["storage_options"],
        decompression_options=_shared_docs["decompression_options"]
        % "filepath_or_buffer",
    )
)
def read_csv(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    sep: str | None | lib.NoDefault = lib.no_default,
    delimiter: str | None | lib.NoDefault = None,
    # Column and Index Locations and Names
    header: int | Sequence[int] | None | Literal["infer"] = "infer",
    names: Sequence[Hashable] | None | lib.NoDefault = lib.no_default,
    index_col: IndexLabel | Literal[False] | None = None,
    usecols: UsecolsArgType = None,
    # General Parsing Configuration
    dtype: DtypeArg | None = None,
    engine: CSVEngine | None = None,
    converters: Mapping[Hashable, Callable] | None = None,
    true_values: list | None = None,
    false_values: list | None = None,
    skipinitialspace: bool = False,
    skiprows: list[int] | int | Callable[[Hashable], bool] | None = None,
    skipfooter: int = 0,
    nrows: int | None = None,
    # NA and Missing Data Handling
    na_values: Hashable
    | Iterable[Hashable]
    | Mapping[Hashable, Iterable[Hashable]]
    | None = None,
    keep_default_na: bool = True,
    na_filter: bool = True,
    verbose: bool | lib.NoDefault = lib.no_default,
    skip_blank_lines: bool = True,
    # Datetime Handling
    parse_dates: bool | Sequence[Hashable] | None = None,
    infer_datetime_format: bool | lib.NoDefault = lib.no_default,
    keep_date_col: bool | lib.NoDefault = lib.no_default,
    date_parser: Callable | lib.NoDefault = lib.no_default,
    date_format: str | dict[Hashable, str] | None = None,
    dayfirst: bool = False,
    cache_dates: bool = True,
    # Iteration
    iterator: bool = False,
    chunksize: int | None = None,
    # Quoting, Compression, and File Format
    compression: CompressionOptions = "infer",
    thousands: str | None = None,
    decimal: str = ".",
    lineterminator: str | None = None,
    quotechar: str = '"',
    quoting: int = csv.QUOTE_MINIMAL,
    doublequote: bool = True,
    escapechar: str | None = None,
    comment: str | None = None,
    encoding: str | None = None,
    encoding_errors: str | None = "strict",
    dialect: str | csv.Dialect | None = None,
    # Error Handling
    on_bad_lines: str = "error",
    # Internal
    delim_whitespace: bool | lib.NoDefault = lib.no_default,
    low_memory: bool = _c_parser_defaults["low_memory"],
    memory_map: bool = False,
    float_precision: Literal["high", "legacy"] | None = None,
    storage_options: StorageOptions | None = None,
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
) -> DataFrame | TextFileReader:
    if keep_date_col is not lib.no_default:
        # GH#55569
        warnings.warn(
            "The 'keep_date_col' keyword in pd.read_csv is deprecated and "
            "will be removed in a future version. Explicitly remove unwanted "
            "columns after parsing instead.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
    else:
        keep_date_col = False

    if lib.is_list_like(parse_dates):
        # GH#55569
        depr = False
        # error: Item "bool" of "bool | Sequence[Hashable] | None" has no
        # attribute "__iter__" (not iterable)
        if not all(is_hashable(x) for x in parse_dates):  # type: ignore[union-attr]
            depr = True
        elif isinstance(parse_dates, dict) and any(
            lib.is_list_like(x) for x in parse_dates.values()
        ):
            depr = True
        if depr:
            warnings.warn(
                "Support for nested sequences for 'parse_dates' in pd.read_csv "
                "is deprecated. Combine the desired columns with pd.to_datetime "
                "after parsing instead.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )

    if infer_datetime_format is not lib.no_default:
        warnings.warn(
            "The argument 'infer_datetime_format' is deprecated and will "
            "be removed in a future version. "
            "A strict version of it is now the default, see "
            "https://pandas.pydata.org/pdeps/0004-consistent-to-datetime-parsing.html. "
            "You can safely remove this argument.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )

    if delim_whitespace is not lib.no_default:
        # GH#55569
        warnings.warn(
            "The 'delim_whitespace' keyword in pd.read_csv is deprecated and "
            "will be removed in a future version. Use ``sep='\\s+'`` instead",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
    else:
        delim_whitespace = False

    if verbose is not lib.no_default:
        # GH#55569
        warnings.warn(
            "The 'verbose' keyword in pd.read_csv is deprecated and "
            "will be removed in a future version.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
    else:
        verbose = False

    # locals() should never be modified
    kwds = locals().copy()
    del kwds["filepath_or_buffer"]
    del kwds["sep"]

    kwds_defaults = _refine_defaults_read(
        dialect,
        delimiter,
        delim_whitespace,
        engine,
        sep,
        on_bad_lines,
        names,
        defaults={"delimiter": ","},
        dtype_backend=dtype_backend,
    )
    kwds.update(kwds_defaults)

    return _read(filepath_or_buffer, kwds)


# iterator=True -> TextFileReader
@overload
def read_table(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    sep: str | None | lib.NoDefault = ...,
    delimiter: str | None | lib.NoDefault = ...,
    header: int | Sequence[int] | None | Literal["infer"] = ...,
    names: Sequence[Hashable] | None | lib.NoDefault = ...,
    index_col: IndexLabel | Literal[False] | None = ...,
    usecols: UsecolsArgType = ...,
    dtype: DtypeArg | None = ...,
    engine: CSVEngine | None = ...,
    converters: Mapping[Hashable, Callable] | None = ...,
    true_values: list | None = ...,
    false_values: list | None = ...,
    skipinitialspace: bool = ...,
    skiprows: list[int] | int | Callable[[Hashable], bool] | None = ...,
    skipfooter: int = ...,
    nrows: int | None = ...,
    na_values: Sequence[str] | Mapping[str, Sequence[str]] | None = ...,
    keep_default_na: bool = ...,
    na_filter: bool = ...,
    verbose: bool | lib.NoDefault = ...,
    skip_blank_lines: bool = ...,
    parse_dates: bool | Sequence[Hashable] = ...,
    infer_datetime_format: bool | lib.NoDefault = ...,
    keep_date_col: bool | lib.NoDefault = ...,
    date_parser: Callable | lib.NoDefault = ...,
    date_format: str | dict[Hashable, str] | None = ...,
    dayfirst: bool = ...,
    cache_dates: bool = ...,
    iterator: Literal[True],
    chunksize: int | None = ...,
    compression: CompressionOptions = ...,
    thousands: str | None = ...,
    decimal: str = ...,
    lineterminator: str | None = ...,
    quotechar: str = ...,
    quoting: int = ...,
    doublequote: bool = ...,
    escapechar: str | None = ...,
    comment: str | None = ...,
    encoding: str | None = ...,
    encoding_errors: str | None = ...,
    dialect: str | csv.Dialect | None = ...,
    on_bad_lines=...,
    delim_whitespace: bool = ...,
    low_memory: bool = ...,
    memory_map: bool = ...,
    float_precision: str | None = ...,
    storage_options: StorageOptions = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
) -> TextFileReader:
    ...


# chunksize=int -> TextFileReader
@overload
def read_table(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    sep: str | None | lib.NoDefault = ...,
    delimiter: str | None | lib.NoDefault = ...,
    header: int | Sequence[int] | None | Literal["infer"] = ...,
    names: Sequence[Hashable] | None | lib.NoDefault = ...,
    index_col: IndexLabel | Literal[False] | None = ...,
    usecols: UsecolsArgType = ...,
    dtype: DtypeArg | None = ...,
    engine: CSVEngine | None = ...,
    converters: Mapping[Hashable, Callable] | None = ...,
    true_values: list | None = ...,
    false_values: list | None = ...,
    skipinitialspace: bool = ...,
    skiprows: list[int] | int | Callable[[Hashable], bool] | None = ...,
    skipfooter: int = ...,
    nrows: int | None = ...,
    na_values: Sequence[str] | Mapping[str, Sequence[str]] | None = ...,
    keep_default_na: bool = ...,
    na_filter: bool = ...,
    verbose: bool | lib.NoDefault = ...,
    skip_blank_lines: bool = ...,
    parse_dates: bool | Sequence[Hashable] = ...,
    infer_datetime_format: bool | lib.NoDefault = ...,
    keep_date_col: bool | lib.NoDefault = ...,
    date_parser: Callable | lib.NoDefault = ...,
    date_format: str | dict[Hashable, str] | None = ...,
    dayfirst: bool = ...,
    cache_dates: bool = ...,
    iterator: bool = ...,
    chunksize: int,
    compression: CompressionOptions = ...,
    thousands: str | None = ...,
    decimal: str = ...,
    lineterminator: str | None = ...,
    quotechar: str = ...,
    quoting: int = ...,
    doublequote: bool = ...,
    escapechar: str | None = ...,
    comment: str | None = ...,
    encoding: str | None = ...,
    encoding_errors: str | None = ...,
    dialect: str | csv.Dialect | None = ...,
    on_bad_lines=...,
    delim_whitespace: bool = ...,
    low_memory: bool = ...,
    memory_map: bool = ...,
    float_precision: str | None = ...,
    storage_options: StorageOptions = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
) -> TextFileReader:
    ...


# default -> DataFrame
@overload
def read_table(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    sep: str | None | lib.NoDefault = ...,
    delimiter: str | None | lib.NoDefault = ...,
    header: int | Sequence[int] | None | Literal["infer"] = ...,
    names: Sequence[Hashable] | None | lib.NoDefault = ...,
    index_col: IndexLabel | Literal[False] | None = ...,
    usecols: UsecolsArgType = ...,
    dtype: DtypeArg | None = ...,
    engine: CSVEngine | None = ...,
    converters: Mapping[Hashable, Callable] | None = ...,
    true_values: list | None = ...,
    false_values: list | None = ...,
    skipinitialspace: bool = ...,
    skiprows: list[int] | int | Callable[[Hashable], bool] | None = ...,
    skipfooter: int = ...,
    nrows: int | None = ...,
    na_values: Sequence[str] | Mapping[str, Sequence[str]] | None = ...,
    keep_default_na: bool = ...,
    na_filter: bool = ...,
    verbose: bool | lib.NoDefault = ...,
    skip_blank_lines: bool = ...,
    parse_dates: bool | Sequence[Hashable] = ...,
    infer_datetime_format: bool | lib.NoDefault = ...,
    keep_date_col: bool | lib.NoDefault = ...,
    date_parser: Callable | lib.NoDefault = ...,
    date_format: str | dict[Hashable, str] | None = ...,
    dayfirst: bool = ...,
    cache_dates: bool = ...,
    iterator: Literal[False] = ...,
    chunksize: None = ...,
    compression: CompressionOptions = ...,
    thousands: str | None = ...,
    decimal: str = ...,
    lineterminator: str | None = ...,
    quotechar: str = ...,
    quoting: int = ...,
    doublequote: bool = ...,
    escapechar: str | None = ...,
    comment: str | None = ...,
    encoding: str | None = ...,
    encoding_errors: str | None = ...,
    dialect: str | csv.Dialect | None = ...,
    on_bad_lines=...,
    delim_whitespace: bool = ...,
    low_memory: bool = ...,
    memory_map: bool = ...,
    float_precision: str | None = ...,
    storage_options: StorageOptions = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
) -> DataFrame:
    ...


# Unions -> DataFrame | TextFileReader
@overload
def read_table(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    sep: str | None | lib.NoDefault = ...,
    delimiter: str | None | lib.NoDefault = ...,
    header: int | Sequence[int] | None | Literal["infer"] = ...,
    names: Sequence[Hashable] | None | lib.NoDefault = ...,
    index_col: IndexLabel | Literal[False] | None = ...,
    usecols: UsecolsArgType = ...,
    dtype: DtypeArg | None = ...,
    engine: CSVEngine | None = ...,
    converters: Mapping[Hashable, Callable] | None = ...,
    true_values: list | None = ...,
    false_values: list | None = ...,
    skipinitialspace: bool = ...,
    skiprows: list[int] | int | Callable[[Hashable], bool] | None = ...,
    skipfooter: int = ...,
    nrows: int | None = ...,
    na_values: Sequence[str] | Mapping[str, Sequence[str]] | None = ...,
    keep_default_na: bool = ...,
    na_filter: bool = ...,
    verbose: bool | lib.NoDefault = ...,
    skip_blank_lines: bool = ...,
    parse_dates: bool | Sequence[Hashable] = ...,
    infer_datetime_format: bool | lib.NoDefault = ...,
    keep_date_col: bool | lib.NoDefault = ...,
    date_parser: Callable | lib.NoDefault = ...,
    date_format: str | dict[Hashable, str] | None = ...,
    dayfirst: bool = ...,
    cache_dates: bool = ...,
    iterator: bool = ...,
    chunksize: int | None = ...,
    compression: CompressionOptions = ...,
    thousands: str | None = ...,
    decimal: str = ...,
    lineterminator: str | None = ...,
    quotechar: str = ...,
    quoting: int = ...,
    doublequote: bool = ...,
    escapechar: str | None = ...,
    comment: str | None = ...,
    encoding: str | None = ...,
    encoding_errors: str | None = ...,
    dialect: str | csv.Dialect | None = ...,
    on_bad_lines=...,
    delim_whitespace: bool = ...,
    low_memory: bool = ...,
    memory_map: bool = ...,
    float_precision: str | None = ...,
    storage_options: StorageOptions = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
) -> DataFrame | TextFileReader:
    ...


@Appender(
    _doc_read_csv_and_table.format(
        func_name="read_table",
        summary="Read general delimited file into DataFrame.",
        see_also_func_name="read_csv",
        see_also_func_summary=(
            "Read a comma-separated values (csv) file into DataFrame."
        ),
        _default_sep=r"'\\t' (tab-stop)",
        storage_options=_shared_docs["storage_options"],
        decompression_options=_shared_docs["decompression_options"]
        % "filepath_or_buffer",
    )
)
def read_table(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    sep: str | None | lib.NoDefault = lib.no_default,
    delimiter: str | None | lib.NoDefault = None,
    # Column and Index Locations and Names
    header: int | Sequence[int] | None | Literal["infer"] = "infer",
    names: Sequence[Hashable] | None | lib.NoDefault = lib.no_default,
    index_col: IndexLabel | Literal[False] | None = None,
    usecols: UsecolsArgType = None,
    # General Parsing Configuration
    dtype: DtypeArg | None = None,
    engine: CSVEngine | None = None,
    converters: Mapping[Hashable, Callable] | None = None,
    true_values: list | None = None,
    false_values: list | None = None,
    skipinitialspace: bool = False,
    skiprows: list[int] | int | Callable[[Hashable], bool] | None = None,
    skipfooter: int = 0,
    nrows: int | None = None,
    # NA and Missing Data Handling
    na_values: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
    keep_default_na: bool = True,
    na_filter: bool = True,
    verbose: bool | lib.NoDefault = lib.no_default,
    skip_blank_lines: bool = True,
    # Datetime Handling
    parse_dates: bool | Sequence[Hashable] = False,
    infer_datetime_format: bool | lib.NoDefault = lib.no_default,
    keep_date_col: bool | lib.NoDefault = lib.no_default,
    date_parser: Callable | lib.NoDefault = lib.no_default,
    date_format: str | dict[Hashable, str] | None = None,
    dayfirst: bool = False,
    cache_dates: bool = True,
    # Iteration
    iterator: bool = False,
    chunksize: int | None = None,
    # Quoting, Compression, and File Format
    compression: CompressionOptions = "infer",
    thousands: str | None = None,
    decimal: str = ".",
    lineterminator: str | None = None,
    quotechar: str = '"',
    quoting: int = csv.QUOTE_MINIMAL,
    doublequote: bool = True,
    escapechar: str | None = None,
    comment: str | None = None,
    encoding: str | None = None,
    encoding_errors: str | None = "strict",
    dialect: str | csv.Dialect | None = None,
    # Error Handling
    on_bad_lines: str = "error",
    # Internal
    delim_whitespace: bool | lib.NoDefault = lib.no_default,
    low_memory: bool = _c_parser_defaults["low_memory"],
    memory_map: bool = False,
    float_precision: str | None = None,
    storage_options: StorageOptions | None = None,
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
) -> DataFrame | TextFileReader:
    if keep_date_col is not lib.no_default:
        # GH#55569
        warnings.warn(
            "The 'keep_date_col' keyword in pd.read_table is deprecated and "
            "will be removed in a future version. Explicitly remove unwanted "
            "columns after parsing instead.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
    else:
        keep_date_col = False

    # error: Item "bool" of "bool | Sequence[Hashable]" has no attribute "__iter__"
    if lib.is_list_like(parse_dates) and not all(is_hashable(x) for x in parse_dates):  # type: ignore[union-attr]
        # GH#55569
        warnings.warn(
            "Support for nested sequences for 'parse_dates' in pd.read_table "
            "is deprecated. Combine the desired columns with pd.to_datetime "
            "after parsing instead.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )

    if infer_datetime_format is not lib.no_default:
        warnings.warn(
            "The argument 'infer_datetime_format' is deprecated and will "
            "be removed in a future version. "
            "A strict version of it is now the default, see "
            "https://pandas.pydata.org/pdeps/0004-consistent-to-datetime-parsing.html. "
            "You can safely remove this argument.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )

    if delim_whitespace is not lib.no_default:
        # GH#55569
        warnings.warn(
            "The 'delim_whitespace' keyword in pd.read_table is deprecated and "
            "will be removed in a future version. Use ``sep='\\s+'`` instead",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
    else:
        delim_whitespace = False

    if verbose is not lib.no_default:
        # GH#55569
        warnings.warn(
            "The 'verbose' keyword in pd.read_table is deprecated and "
            "will be removed in a future version.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
    else:
        verbose = False

    # locals() should never be modified
    kwds = locals().copy()
    del kwds["filepath_or_buffer"]
    del kwds["sep"]

    kwds_defaults = _refine_defaults_read(
        dialect,
        delimiter,
        delim_whitespace,
        engine,
        sep,
        on_bad_lines,
        names,
        defaults={"delimiter": "\t"},
        dtype_backend=dtype_backend,
    )
    kwds.update(kwds_defaults)

    return _read(filepath_or_buffer, kwds)


@overload
def read_fwf(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    colspecs: Sequence[tuple[int, int]] | str | None = ...,
    widths: Sequence[int] | None = ...,
    infer_nrows: int = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
    iterator: Literal[True],
    chunksize: int | None = ...,
    **kwds,
) -> TextFileReader:
    ...


@overload
def read_fwf(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    colspecs: Sequence[tuple[int, int]] | str | None = ...,
    widths: Sequence[int] | None = ...,
    infer_nrows: int = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
    iterator: bool = ...,
    chunksize: int,
    **kwds,
) -> TextFileReader:
    ...


@overload
def read_fwf(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    colspecs: Sequence[tuple[int, int]] | str | None = ...,
    widths: Sequence[int] | None = ...,
    infer_nrows: int = ...,
    dtype_backend: DtypeBackend | lib.NoDefault = ...,
    iterator: Literal[False] = ...,
    chunksize: None = ...,
    **kwds,
) -> DataFrame:
    ...


def read_fwf(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    colspecs: Sequence[tuple[int, int]] | str | None = "infer",
    widths: Sequence[int] | None = None,
    infer_nrows: int = 100,
    dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
    iterator: bool = False,
    chunksize: int | None = None,
    **kwds,
) -> DataFrame | TextFileReader:
    r"""
    Read a table of fixed-width formatted lines into DataFrame.

    Also supports optionally iterating or breaking of the file
    into chunks.

    Additional help can be found in the `online docs for IO Tools
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html>`_.

    Parameters
    ----------
    filepath_or_buffer : str, path object, or file-like object
        String, path object (implementing ``os.PathLike[str]``), or file-like
        object implementing a text ``read()`` function.The string could be a URL.
        Valid URL schemes include http, ftp, s3, and file. For file URLs, a host is
        expected. A local file could be:
        ``file://localhost/path/to/table.csv``.
    colspecs : list of tuple (int, int) or 'infer'. optional
        A list of tuples giving the extents of the fixed-width
        fields of each line as half-open intervals (i.e.,  [from, to[ ).
        String value 'infer' can be used to instruct the parser to try
        detecting the column specifications from the first 100 rows of
        the data which are not being skipped via skiprows (default='infer').
    widths : list of int, optional
        A list of field widths which can be used instead of 'colspecs' if
        the intervals are contiguous.
    infer_nrows : int, default 100
        The number of rows to consider when letting the parser determine the
        `colspecs`.
    dtype_backend : {'numpy_nullable', 'pyarrow'}, default 'numpy_nullable'
        Back-end data type applied to the resultant :class:`DataFrame`
        (still experimental). Behaviour is as follows:

        * ``"numpy_nullable"``: returns nullable-dtype-backed :class:`DataFrame`
          (default).
        * ``"pyarrow"``: returns pyarrow-backed nullable :class:`ArrowDtype`
          DataFrame.

        .. versionadded:: 2.0

    **kwds : optional
        Optional keyword arguments can be passed to ``TextFileReader``.

    Returns
    -------
    DataFrame or TextFileReader
        A comma-separated values (csv) file is returned as two-dimensional
        data structure with labeled axes.

    See Also
    --------
    DataFrame.to_csv : Write DataFrame to a comma-separated values (csv) file.
    read_csv : Read a comma-separated values (csv) file into DataFrame.

    Examples
    --------
    >>> pd.read_fwf('data.csv')  # doctest: +SKIP
    """
    # Check input arguments.
    if colspecs is None and widths is None:
        raise ValueError("Must specify either colspecs or widths")
    if colspecs not in (None, "infer") and widths is not None:
        raise ValueError("You must specify only one of 'widths' and 'colspecs'")

    # Compute 'colspecs' from 'widths', if specified.
    if widths is not None:
        colspecs, col = [], 0
        for w in widths:
            colspecs.append((col, col + w))
            col += w

    # for mypy
    assert colspecs is not None

    # GH#40830
    # Ensure length of `colspecs` matches length of `names`
    names = kwds.get("names")
    if names is not None:
        if len(names) != len(colspecs) and colspecs != "infer":
            # need to check len(index_col) as it might contain
            # unnamed indices, in which case it's name is not required
            len_index = 0
            if kwds.get("index_col") is not None:
                index_col: Any = kwds.get("index_col")
                if index_col is not False:
                    if not is_list_like(index_col):
                        len_index = 1
                    else:
                        len_index = len(index_col)
            if kwds.get("usecols") is None and len(names) + len_index != len(colspecs):
                # If usecols is used colspec may be longer than names
                raise ValueError("Length of colspecs must match length of names")

    kwds["colspecs"] = colspecs
    kwds["infer_nrows"] = infer_nrows
    kwds["engine"] = "python-fwf"
    kwds["iterator"] = iterator
    kwds["chunksize"] = chunksize

    check_dtype_backend(dtype_backend)
    kwds["dtype_backend"] = dtype_backend
    return _read(filepath_or_buffer, kwds)


class TextFileReader(abc.Iterator):
    """

    Passed dialect overrides any of the related parser options

    """

    def __init__(
        self,
        f: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str] | list,
        engine: CSVEngine | None = None,
        **kwds,
    ) -> None:
        if engine is not None:
            engine_specified = True
        else:
            engine = "python"
            engine_specified = False
        self.engine = engine
        self._engine_specified = kwds.get("engine_specified", engine_specified)

        _validate_skipfooter(kwds)

        dialect = _extract_dialect(kwds)
        if dialect is not None:
            if engine == "pyarrow":
                raise ValueError(
                    "The 'dialect' option is not supported with the 'pyarrow' engine"
                )
            kwds = _merge_with_dialect_properties(dialect, kwds)

        if kwds.get("header", "infer") == "infer":
            kwds["header"] = 0 if kwds.get("names") is None else None

        self.orig_options = kwds

        # miscellanea
        self._currow = 0

        options = self._get_options_with_defaults(engine)
        options["storage_options"] = kwds.get("storage_options", None)

        self.chunksize = options.pop("chunksize", None)
        self.nrows = options.pop("nrows", None)

        self._check_file_or_buffer(f, engine)
        self.options, self.engine = self._clean_options(options, engine)

        if "has_index_names" in kwds:
            self.options["has_index_names"] = kwds["has_index_names"]

        self.handles: IOHandles | None = None
        self._engine = self._make_engine(f, self.engine)

    def close(self) -> None:
        if self.handles is not None:
            self.handles.close()
        self._engine.close()

    def _get_options_with_defaults(self, engine: CSVEngine) -> dict[str, Any]:
        kwds = self.orig_options

        options = {}
        default: object | None

        for argname, default in parser_defaults.items():
            value = kwds.get(argname, default)

            # see gh-12935
            if (
                engine == "pyarrow"
                and argname in _pyarrow_unsupported
                and value != default
                and value != getattr(value, "value", default)
            ):
                raise ValueError(
                    f"The {repr(argname)} option is not supported with the "
                    f"'pyarrow' engine"
                )
            options[argname] = value

        for argname, default in _c_parser_defaults.items():
            if argname in kwds:
                value = kwds[argname]

                if engine != "c" and value != default:
                    # TODO: Refactor this logic, its pretty convoluted
                    if "python" in engine and argname not in _python_unsupported:
                        pass
                    elif "pyarrow" in engine and argname not in _pyarrow_unsupported:
                        pass
                    else:
                        raise ValueError(
                            f"The {repr(argname)} option is not supported with the "
                            f"{repr(engine)} engine"
                        )
            else:
                value = default
            options[argname] = value

        if engine == "python-fwf":
            for argname, default in _fwf_defaults.items():
                options[argname] = kwds.get(argname, default)

        return options

    def _check_file_or_buffer(self, f, engine: CSVEngine) -> None:
        # see gh-16530
        if is_file_like(f) and engine != "c" and not hasattr(f, "__iter__"):
            # The C engine doesn't need the file-like to have the "__iter__"
            # attribute. However, the Python engine needs "__iter__(...)"
            # when iterating through such an object, meaning it
            # needs to have that attribute
            raise ValueError(
                "The 'python' engine cannot iterate through this file buffer."
            )

    def _clean_options(
        self, options: dict[str, Any], engine: CSVEngine
    ) -> tuple[dict[str, Any], CSVEngine]:
        result = options.copy()

        fallback_reason = None

        # C engine not supported yet
        if engine == "c":
            if options["skipfooter"] > 0:
                fallback_reason = "the 'c' engine does not support skipfooter"
                engine = "python"

        sep = options["delimiter"]
        delim_whitespace = options["delim_whitespace"]

        if sep is None and not delim_whitespace:
            if engine in ("c", "pyarrow"):
                fallback_reason = (
                    f"the '{engine}' engine does not support "
                    "sep=None with delim_whitespace=False"
                )
                engine = "python"
        elif sep is not None and len(sep) > 1:
            if engine == "c" and sep == r"\s+":
                result["delim_whitespace"] = True
                del result["delimiter"]
            elif engine not in ("python", "python-fwf"):
                # wait until regex engine integrated
                fallback_reason = (
                    f"the '{engine}' engine does not support "
                    "regex separators (separators > 1 char and "
                    r"different from '\s+' are interpreted as regex)"
                )
                engine = "python"
        elif delim_whitespace:
            if "python" in engine:
                result["delimiter"] = r"\s+"
        elif sep is not None:
            encodeable = True
            encoding = sys.getfilesystemencoding() or "utf-8"
            try:
                if len(sep.encode(encoding)) > 1:
                    encodeable = False
            except UnicodeDecodeError:
                encodeable = False
            if not encodeable and engine not in ("python", "python-fwf"):
                fallback_reason = (
                    f"the separator encoded in {encoding} "
                    f"is > 1 char long, and the '{engine}' engine "
                    "does not support such separators"
                )
                engine = "python"

        quotechar = options["quotechar"]
        if quotechar is not None and isinstance(quotechar, (str, bytes)):
            if (
                len(quotechar) == 1
                and ord(quotechar) > 127
                and engine not in ("python", "python-fwf")
            ):
                fallback_reason = (
                    "ord(quotechar) > 127, meaning the "
                    "quotechar is larger than one byte, "
                    f"and the '{engine}' engine does not support such quotechars"
                )
                engine = "python"

        if fallback_reason and self._engine_specified:
            raise ValueError(fallback_reason)

        if engine == "c":
            for arg in _c_unsupported:
                del result[arg]

        if "python" in engine:
            for arg in _python_unsupported:
                if fallback_reason and result[arg] != _c_parser_defaults.get(arg):
                    raise ValueError(
                        "Falling back to the 'python' engine because "
                        f"{fallback_reason}, but this causes {repr(arg)} to be "
                        "ignored as it is not supported by the 'python' engine."
                    )
                del result[arg]

        if fallback_reason:
            warnings.warn(
                (
                    "Falling back to the 'python' engine because "
                    f"{fallback_reason}; you can avoid this warning by specifying "
                    "engine='python'."
                ),
                ParserWarning,
                stacklevel=find_stack_level(),
            )

        index_col = options["index_col"]
        names = options["names"]
        converters = options["converters"]
        na_values = options["na_values"]
        skiprows = options["skiprows"]

        validate_header_arg(options["header"])

        if index_col is True:
            raise ValueError("The value of index_col couldn't be 'True'")
        if is_index_col(index_col):
            if not isinstance(index_col, (list, tuple, np.ndarray)):
                index_col = [index_col]
        result["index_col"] = index_col

        names = list(names) if names is not None else names

        # type conversion-related
        if converters is not None:
            if not isinstance(converters, dict):
                raise TypeError(
                    "Type converters must be a dict or subclass, "
                    f"input was a {type(converters).__name__}"
                )
        else:
            converters = {}

        # Converting values to NA
        keep_default_na = options["keep_default_na"]
        floatify = engine != "pyarrow"
        na_values, na_fvalues = _clean_na_values(
            na_values, keep_default_na, floatify=floatify
        )

        # handle skiprows; this is internally handled by the
        # c-engine, so only need for python and pyarrow parsers
        if engine == "pyarrow":
            if not is_integer(skiprows) and skiprows is not None:
                # pyarrow expects skiprows to be passed as an integer
                raise ValueError(
                    "skiprows argument must be an integer when using "
                    "engine='pyarrow'"
                )
        else:
            if is_integer(skiprows):
                skiprows = list(range(skiprows))
            if skiprows is None:
                skiprows = set()
            elif not callable(skiprows):
                skiprows = set(skiprows)

        # put stuff back
        result["names"] = names
        result["converters"] = converters
        result["na_values"] = na_values
        result["na_fvalues"] = na_fvalues
        result["skiprows"] = skiprows

        return result, engine

    def __next__(self) -> DataFrame:
        try:
            return self.get_chunk()
        except StopIteration:
            self.close()
            raise

    def _make_engine(
        self,
        f: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str] | list | IO,
        engine: CSVEngine = "c",
    ) -> ParserBase:
        mapping: dict[str, type[ParserBase]] = {
            "c": CParserWrapper,
            "python": PythonParser,
            "pyarrow": ArrowParserWrapper,
            "python-fwf": FixedWidthFieldParser,
        }
        if engine not in mapping:
            raise ValueError(
                f"Unknown engine: {engine} (valid options are {mapping.keys()})"
            )
        if not isinstance(f, list):
            # open file here
            is_text = True
            mode = "r"
            if engine == "pyarrow":
                is_text = False
                mode = "rb"
            elif (
                engine == "c"
                and self.options.get("encoding", "utf-8") == "utf-8"
                and isinstance(stringify_path(f), str)
            ):
                # c engine can decode utf-8 bytes, adding TextIOWrapper makes
                # the c-engine especially for memory_map=True far slower
                is_text = False
                if "b" not in mode:
                    mode += "b"
            self.handles = get_handle(
                f,
                mode,
                encoding=self.options.get("encoding", None),
                compression=self.options.get("compression", None),
                memory_map=self.options.get("memory_map", False),
                is_text=is_text,
                errors=self.options.get("encoding_errors", "strict"),
                storage_options=self.options.get("storage_options", None),
            )
            assert self.handles is not None
            f = self.handles.handle

        elif engine != "python":
            msg = f"Invalid file path or buffer object type: {type(f)}"
            raise ValueError(msg)

        try:
            return mapping[engine](f, **self.options)
        except Exception:
            if self.handles is not None:
                self.handles.close()
            raise

    def _failover_to_python(self) -> None:
        raise AbstractMethodError(self)

    def read(self, nrows: int | None = None) -> DataFrame:
        if self.engine == "pyarrow":
            try:
                # error: "ParserBase" has no attribute "read"
                df = self._engine.read()  # type: ignore[attr-defined]
            except Exception:
                self.close()
                raise
        else:
            nrows = validate_integer("nrows", nrows)
            try:
                # error: "ParserBase" has no attribute "read"
                (
                    index,
                    columns,
                    col_dict,
                ) = self._engine.read(  # type: ignore[attr-defined]
                    nrows
                )
            except Exception:
                self.close()
                raise

            if index is None:
                if col_dict:
                    # Any column is actually fine:
                    new_rows = len(next(iter(col_dict.values())))
                    index = RangeIndex(self._currow, self._currow + new_rows)
                else:
                    new_rows = 0
            else:
                new_rows = len(index)

            if hasattr(self, "orig_options"):
                dtype_arg = self.orig_options.get("dtype", None)
            else:
                dtype_arg = None

            if isinstance(dtype_arg, dict):
                dtype = defaultdict(lambda: None)  # type: ignore[var-annotated]
                dtype.update(dtype_arg)
            elif dtype_arg is not None and pandas_dtype(dtype_arg) in (
                np.str_,
                np.object_,
            ):
                dtype = defaultdict(lambda: dtype_arg)
            else:
                dtype = None

            if dtype is not None:
                new_col_dict = {}
                for k, v in col_dict.items():
                    d = (
                        dtype[k]
                        if pandas_dtype(dtype[k]) in (np.str_, np.object_)
                        else None
                    )
                    new_col_dict[k] = Series(v, index=index, dtype=d, copy=False)
            else:
                new_col_dict = col_dict

            df = DataFrame(
                new_col_dict,
                columns=columns,
                index=index,
                copy=not using_copy_on_write(),
            )

            self._currow += new_rows
        return df

    def get_chunk(self, size: int | None = None) -> DataFrame:
        if size is None:
            size = self.chunksize
        if self.nrows is not None:
            if self._currow >= self.nrows:
                raise StopIteration
            size = min(size, self.nrows - self._currow)
        return self.read(nrows=size)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def TextParser(*args, **kwds) -> TextFileReader:
    """
    Converts lists of lists/tuples into DataFrames with proper type inference
    and optional (e.g. string to datetime) conversion. Also enables iterating
    lazily over chunks of large files

    Parameters
    ----------
    data : file-like object or list
    delimiter : separator character to use
    dialect : str or csv.Dialect instance, optional
        Ignored if delimiter is longer than 1 character
    names : sequence, default
    header : int, default 0
        Row to use to parse column labels. Defaults to the first row. Prior
        rows will be discarded
    index_col : int or list, optional
        Column or columns to use as the (possibly hierarchical) index
    has_index_names: bool, default False
        True if the cols defined in index_col have an index name and are
        not in the header.
    na_values : scalar, str, list-like, or dict, optional
        Additional strings to recognize as NA/NaN.
    keep_default_na : bool, default True
    thousands : str, optional
        Thousands separator
    comment : str, optional
        Comment out remainder of line
    parse_dates : bool, default False
    keep_date_col : bool, default False
    date_parser : function, optional

        .. deprecated:: 2.0.0
    date_format : str or dict of column -> format, default ``None``

        .. versionadded:: 2.0.0
    skiprows : list of integers
        Row numbers to skip
    skipfooter : int
        Number of line at bottom of file to skip
    converters : dict, optional
        Dict of functions for converting values in certain columns. Keys can
        either be integers or column labels, values are functions that take one
        input argument, the cell (not column) content, and return the
        transformed content.
    encoding : str, optional
        Encoding to use for UTF when reading/writing (ex. 'utf-8')
    float_precision : str, optional
        Specifies which converter the C engine should use for floating-point
        values. The options are `None` or `high` for the ordinary converter,
        `legacy` for the original lower precision pandas converter, and
        `round_trip` for the round-trip converter.
    """
    kwds["engine"] = "python"
    return TextFileReader(*args, **kwds)


def _clean_na_values(na_values, keep_default_na: bool = True, floatify: bool = True):
    na_fvalues: set | dict
    if na_values is None:
        if keep_default_na:
            na_values = STR_NA_VALUES
        else:
            na_values = set()
        na_fvalues = set()
    elif isinstance(na_values, dict):
        old_na_values = na_values.copy()
        na_values = {}  # Prevent aliasing.

        # Convert the values in the na_values dictionary
        # into array-likes for further use. This is also
        # where we append the default NaN values, provided
        # that `keep_default_na=True`.
        for k, v in old_na_values.items():
            if not is_list_like(v):
                v = [v]

            if keep_default_na:
                v = set(v) | STR_NA_VALUES

            na_values[k] = v
        na_fvalues = {k: _floatify_na_values(v) for k, v in na_values.items()}
    else:
        if not is_list_like(na_values):
            na_values = [na_values]
        na_values = _stringify_na_values(na_values, floatify)
        if keep_default_na:
            na_values = na_values | STR_NA_VALUES

        na_fvalues = _floatify_na_values(na_values)

    return na_values, na_fvalues


def _floatify_na_values(na_values):
    # create float versions of the na_values
    result = set()
    for v in na_values:
        try:
            v = float(v)
            if not np.isnan(v):
                result.add(v)
        except (TypeError, ValueError, OverflowError):
            pass
    return result


def _stringify_na_values(na_values, floatify: bool):
    """return a stringified and numeric for these values"""
    result: list[str | float] = []
    for x in na_values:
        result.append(str(x))
        result.append(x)
        try:
            v = float(x)

            # we are like 999 here
            if v == int(v):
                v = int(v)
                result.append(f"{v}.0")
                result.append(str(v))

            if floatify:
                result.append(v)
        except (TypeError, ValueError, OverflowError):
            pass
        if floatify:
            try:
                result.append(int(x))
            except (TypeError, ValueError, OverflowError):
                pass
    return set(result)


def _refine_defaults_read(
    dialect: str | csv.Dialect | None,
    delimiter: str | None | lib.NoDefault,
    delim_whitespace: bool,
    engine: CSVEngine | None,
    sep: str | None | lib.NoDefault,
    on_bad_lines: str | Callable,
    names: Sequence[Hashable] | None | lib.NoDefault,
    defaults: dict[str, Any],
    dtype_backend: DtypeBackend | lib.NoDefault,
):
    """Validate/refine default values of input parameters of read_csv, read_table.

    Parameters
    ----------
    dialect : str or csv.Dialect
        If provided, this parameter will override values (default or not) for the
        following parameters: `delimiter`, `doublequote`, `escapechar`,
        `skipinitialspace`, `quotechar`, and `quoting`. If it is necessary to
        override values, a ParserWarning will be issued. See csv.Dialect
        documentation for more details.
    delimiter : str or object
        Alias for sep.
    delim_whitespace : bool
        Specifies whether or not whitespace (e.g. ``' '`` or ``'\t'``) will be
        used as the sep. Equivalent to setting ``sep='\\s+'``. If this option
        is set to True, nothing should be passed in for the ``delimiter``
        parameter.

        .. deprecated:: 2.2.0
            Use ``sep="\\s+"`` instead.
    engine : {{'c', 'python'}}
        Parser engine to use. The C engine is faster while the python engine is
        currently more feature-complete.
    sep : str or object
        A delimiter provided by the user (str) or a sentinel value, i.e.
        pandas._libs.lib.no_default.
    on_bad_lines : str, callable
        An option for handling bad lines or a sentinel value(None).
    names : array-like, optional
        List of column names to use. If the file contains a header row,
        then you should explicitly pass ``header=0`` to override the column names.
        Duplicates in this list are not allowed.
    defaults: dict
        Default values of input parameters.

    Returns
    -------
    kwds : dict
        Input parameters with correct values.

    Raises
    ------
    ValueError :
        If a delimiter was specified with ``sep`` (or ``delimiter``) and
        ``delim_whitespace=True``.
    """
    # fix types for sep, delimiter to Union(str, Any)
    delim_default = defaults["delimiter"]
    kwds: dict[str, Any] = {}
    # gh-23761
    #
    # When a dialect is passed, it overrides any of the overlapping
    # parameters passed in directly. We don't want to warn if the
    # default parameters were passed in (since it probably means
    # that the user didn't pass them in explicitly in the first place).
    #
    # "delimiter" is the annoying corner case because we alias it to
    # "sep" before doing comparison to the dialect values later on.
    # Thus, we need a flag to indicate that we need to "override"
    # the comparison to dialect values by checking if default values
    # for BOTH "delimiter" and "sep" were provided.
    if dialect is not None:
        kwds["sep_override"] = delimiter is None and (
            sep is lib.no_default or sep == delim_default
        )

    if delimiter and (sep is not lib.no_default):
        raise ValueError("Specified a sep and a delimiter; you can only specify one.")

    kwds["names"] = None if names is lib.no_default else names

    # Alias sep -> delimiter.
    if delimiter is None:
        delimiter = sep

    if delim_whitespace and (delimiter is not lib.no_default):
        raise ValueError(
            "Specified a delimiter with both sep and "
            "delim_whitespace=True; you can only specify one."
        )

    if delimiter == "\n":
        raise ValueError(
            r"Specified \n as separator or delimiter. This forces the python engine "
            "which does not accept a line terminator. Hence it is not allowed to use "
            "the line terminator as separator.",
        )

    if delimiter is lib.no_default:
        # assign default separator value
        kwds["delimiter"] = delim_default
    else:
        kwds["delimiter"] = delimiter

    if engine is not None:
        kwds["engine_specified"] = True
    else:
        kwds["engine"] = "c"
        kwds["engine_specified"] = False

    if on_bad_lines == "error":
        kwds["on_bad_lines"] = ParserBase.BadLineHandleMethod.ERROR
    elif on_bad_lines == "warn":
        kwds["on_bad_lines"] = ParserBase.BadLineHandleMethod.WARN
    elif on_bad_lines == "skip":
        kwds["on_bad_lines"] = ParserBase.BadLineHandleMethod.SKIP
    elif callable(on_bad_lines):
        if engine not in ["python", "pyarrow"]:
            raise ValueError(
                "on_bad_line can only be a callable function "
                "if engine='python' or 'pyarrow'"
            )
        kwds["on_bad_lines"] = on_bad_lines
    else:
        raise ValueError(f"Argument {on_bad_lines} is invalid for on_bad_lines")

    check_dtype_backend(dtype_backend)

    kwds["dtype_backend"] = dtype_backend

    return kwds


def _extract_dialect(kwds: dict[str, Any]) -> csv.Dialect | None:
    """
    Extract concrete csv dialect instance.

    Returns
    -------
    csv.Dialect or None
    """
    if kwds.get("dialect") is None:
        return None

    dialect = kwds["dialect"]
    if dialect in csv.list_dialects():
        dialect = csv.get_dialect(dialect)

    _validate_dialect(dialect)

    return dialect


MANDATORY_DIALECT_ATTRS = (
    "delimiter",
    "doublequote",
    "escapechar",
    "skipinitialspace",
    "quotechar",
    "quoting",
)


def _validate_dialect(dialect: csv.Dialect) -> None:
    """
    Validate csv dialect instance.

    Raises
    ------
    ValueError
        If incorrect dialect is provided.
    """
    for param in MANDATORY_DIALECT_ATTRS:
        if not hasattr(dialect, param):
            raise ValueError(f"Invalid dialect {dialect} provided")


def _merge_with_dialect_properties(
    dialect: csv.Dialect,
    defaults: dict[str, Any],
) -> dict[str, Any]:
    """
    Merge default kwargs in TextFileReader with dialect parameters.

    Parameters
    ----------
    dialect : csv.Dialect
        Concrete csv dialect. See csv.Dialect documentation for more details.
    defaults : dict
        Keyword arguments passed to TextFileReader.

    Returns
    -------
    kwds : dict
        Updated keyword arguments, merged with dialect parameters.
    """
    kwds = defaults.copy()

    for param in MANDATORY_DIALECT_ATTRS:
        dialect_val = getattr(dialect, param)

        parser_default = parser_defaults[param]
        provided = kwds.get(param, parser_default)

        # Messages for conflicting values between the dialect
        # instance and the actual parameters provided.
        conflict_msgs = []

        # Don't warn if the default parameter was passed in,
        # even if it conflicts with the dialect (gh-23761).
        if provided not in (parser_default, dialect_val):
            msg = (
                f"Conflicting values for '{param}': '{provided}' was "
                f"provided, but the dialect specifies '{dialect_val}'. "
                "Using the dialect-specified value."
            )

            # Annoying corner case for not warning about
            # conflicts between dialect and delimiter parameter.
            # Refer to the outer "_read_" function for more info.
            if not (param == "delimiter" and kwds.pop("sep_override", False)):
                conflict_msgs.append(msg)

        if conflict_msgs:
            warnings.warn(
                "\n\n".join(conflict_msgs), ParserWarning, stacklevel=find_stack_level()
            )
        kwds[param] = dialect_val
    return kwds


def _validate_skipfooter(kwds: dict[str, Any]) -> None:
    """
    Check whether skipfooter is compatible with other kwargs in TextFileReader.

    Parameters
    ----------
    kwds : dict
        Keyword arguments passed to TextFileReader.

    Raises
    ------
    ValueError
        If skipfooter is not compatible with other parameters.
    """
    if kwds.get("skipfooter"):
        if kwds.get("iterator") or kwds.get("chunksize"):
            raise ValueError("'skipfooter' not supported for iteration")
        if kwds.get("nrows"):
            raise ValueError("'skipfooter' not supported with 'nrows'")
<|endoftext|>from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING
import warnings

import numpy as np

from pandas._libs import (
    lib,
    parsers,
)
from pandas.compat._optional import import_optional_dependency
from pandas.errors import DtypeWarning
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import pandas_dtype
from pandas.core.dtypes.concat import (
    concat_compat,
    union_categoricals,
)
from pandas.core.dtypes.dtypes import CategoricalDtype

from pandas.core.indexes.api import ensure_index_from_sequences

from pandas.io.common import (
    dedup_names,
    is_potential_multi_index,
)
from pandas.io.parsers.base_parser import (
    ParserBase,
    ParserError,
    is_index_col,
)

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Mapping,
        Sequence,
    )

    from pandas._typing import (
        ArrayLike,
        DtypeArg,
        DtypeObj,
        ReadCsvBuffer,
    )

    from pandas import (
        Index,
        MultiIndex,
    )


class CParserWrapper(ParserBase):
    low_memory: bool
    _reader: parsers.TextReader

    def __init__(self, src: ReadCsvBuffer[str], **kwds) -> None:
        super().__init__(kwds)
        self.kwds = kwds
        kwds = kwds.copy()

        self.low_memory = kwds.pop("low_memory", False)

        # #2442
        # error: Cannot determine type of 'index_col'
        kwds["allow_leading_cols"] = (
            self.index_col is not False  # type: ignore[has-type]
        )

        # GH20529, validate usecol arg before TextReader
        kwds["usecols"] = self.usecols

        # Have to pass int, would break tests using TextReader directly otherwise :(
        kwds["on_bad_lines"] = self.on_bad_lines.value

        for key in (
            "storage_options",
            "encoding",
            "memory_map",
            "compression",
        ):
            kwds.pop(key, None)

        kwds["dtype"] = ensure_dtype_objs(kwds.get("dtype", None))
        if "dtype_backend" not in kwds or kwds["dtype_backend"] is lib.no_default:
            kwds["dtype_backend"] = "numpy"
        if kwds["dtype_backend"] == "pyarrow":
            # Fail here loudly instead of in cython after reading
            import_optional_dependency("pyarrow")
        self._reader = parsers.TextReader(src, **kwds)

        self.unnamed_cols = self._reader.unnamed_cols

        # error: Cannot determine type of 'names'
        passed_names = self.names is None  # type: ignore[has-type]

        if self._reader.header is None:
            self.names = None
        else:
            # error: Cannot determine type of 'names'
            # error: Cannot determine type of 'index_names'
            (
                self.names,  # type: ignore[has-type]
                self.index_names,
                self.col_names,
                passed_names,
            ) = self._extract_multi_indexer_columns(
                self._reader.header,
                self.index_names,  # type: ignore[has-type]
                passed_names,
            )

        # error: Cannot determine type of 'names'
        if self.names is None:  # type: ignore[has-type]
            self.names = list(range(self._reader.table_width))

        # gh-9755
        #
        # need to set orig_names here first
        # so that proper indexing can be done
        # with _set_noconvert_columns
        #
        # once names has been filtered, we will
        # then set orig_names again to names
        # error: Cannot determine type of 'names'
        self.orig_names = self.names[:]  # type: ignore[has-type]

        if self.usecols:
            usecols = self._evaluate_usecols(self.usecols, self.orig_names)

            # GH 14671
            # assert for mypy, orig_names is List or None, None would error in issubset
            assert self.orig_names is not None
            if self.usecols_dtype == "string" and not set(usecols).issubset(
                self.orig_names
            ):
                self._validate_usecols_names(usecols, self.orig_names)

            # error: Cannot determine type of 'names'
            if len(self.names) > len(usecols):  # type: ignore[has-type]
                # error: Cannot determine type of 'names'
                self.names = [  # type: ignore[has-type]
                    n
                    # error: Cannot determine type of 'names'
                    for i, n in enumerate(self.names)  # type: ignore[has-type]
                    if (i in usecols or n in usecols)
                ]

            # error: Cannot determine type of 'names'
            if len(self.names) < len(usecols):  # type: ignore[has-type]
                # error: Cannot determine type of 'names'
                self._validate_usecols_names(
                    usecols,
                    self.names,  # type: ignore[has-type]
                )

        # error: Cannot determine type of 'names'
        self._validate_parse_dates_presence(self.names)  # type: ignore[has-type]
        self._set_noconvert_columns()

        # error: Cannot determine type of 'names'
        self.orig_names = self.names  # type: ignore[has-type]

        if not self._has_complex_date_col:
            # error: Cannot determine type of 'index_col'
            if self._reader.leading_cols == 0 and is_index_col(
                self.index_col  # type: ignore[has-type]
            ):
                self._name_processed = True
                (
                    index_names,
                    # error: Cannot determine type of 'names'
                    self.names,  # type: ignore[has-type]
                    self.index_col,
                ) = self._clean_index_names(
                    # error: Cannot determine type of 'names'
                    self.names,  # type: ignore[has-type]
                    # error: Cannot determine type of 'index_col'
                    self.index_col,  # type: ignore[has-type]
                )

                if self.index_names is None:
                    self.index_names = index_names

            if self._reader.header is None and not passed_names:
                assert self.index_names is not None
                self.index_names = [None] * len(self.index_names)

        self._implicit_index = self._reader.leading_cols > 0

    def close(self) -> None:
        # close handles opened by C parser
        try:
            self._reader.close()
        except ValueError:
            pass

    def _set_noconvert_columns(self) -> None:
        """
        Set the columns that should not undergo dtype conversions.

        Currently, any column that is involved with date parsing will not
        undergo such conversions.
        """
        assert self.orig_names is not None
        # error: Cannot determine type of 'names'

        # much faster than using orig_names.index(x) xref GH#44106
        names_dict = {x: i for i, x in enumerate(self.orig_names)}
        col_indices = [names_dict[x] for x in self.names]  # type: ignore[has-type]
        # error: Cannot determine type of 'names'
        noconvert_columns = self._set_noconvert_dtype_columns(
            col_indices,
            self.names,  # type: ignore[has-type]
        )
        for col in noconvert_columns:
            self._reader.set_noconvert(col)

    def read(
        self,
        nrows: int | None = None,
    ) -> tuple[
        Index | MultiIndex | None,
        Sequence[Hashable] | MultiIndex,
        Mapping[Hashable, ArrayLike],
    ]:
        index: Index | MultiIndex | None
        column_names: Sequence[Hashable] | MultiIndex
        try:
            if self.low_memory:
                chunks = self._reader.read_low_memory(nrows)
                # destructive to chunks
                data = _concatenate_chunks(chunks)

            else:
                data = self._reader.read(nrows)
        except StopIteration:
            if self._first_chunk:
                self._first_chunk = False
                names = dedup_names(
                    self.orig_names,
                    is_potential_multi_index(self.orig_names, self.index_col),
                )
                index, columns, col_dict = self._get_empty_meta(
                    names,
                    dtype=self.dtype,
                )
                columns = self._maybe_make_multi_index_columns(columns, self.col_names)

                if self.usecols is not None:
                    columns = self._filter_usecols(columns)

                col_dict = {k: v for k, v in col_dict.items() if k in columns}

                return index, columns, col_dict

            else:
                self.close()
                raise

        # Done with first read, next time raise StopIteration
        self._first_chunk = False

        # error: Cannot determine type of 'names'
        names = self.names  # type: ignore[has-type]

        if self._reader.leading_cols:
            if self._has_complex_date_col:
                raise NotImplementedError("file structure not yet supported")

            # implicit index, no index names
            arrays = []

            if self.index_col and self._reader.leading_cols != len(self.index_col):
                raise ParserError(
                    "Could not construct index. Requested to use "
                    f"{len(self.index_col)} number of columns, but "
                    f"{self._reader.leading_cols} left to parse."
                )

            for i in range(self._reader.leading_cols):
                if self.index_col is None:
                    values = data.pop(i)
                else:
                    values = data.pop(self.index_col[i])

                values = self._maybe_parse_dates(values, i, try_parse_dates=True)
                arrays.append(values)

            index = ensure_index_from_sequences(arrays)

            if self.usecols is not None:
                names = self._filter_usecols(names)

            names = dedup_names(names, is_potential_multi_index(names, self.index_col))

            # rename dict keys
            data_tups = sorted(data.items())
            data = {k: v for k, (i, v) in zip(names, data_tups)}

            column_names, date_data = self._do_date_conversions(names, data)

            # maybe create a mi on the columns
            column_names = self._maybe_make_multi_index_columns(
                column_names, self.col_names
            )

        else:
            # rename dict keys
            data_tups = sorted(data.items())

            # ugh, mutation

            # assert for mypy, orig_names is List or None, None would error in list(...)
            assert self.orig_names is not None
            names = list(self.orig_names)
            names = dedup_names(names, is_potential_multi_index(names, self.index_col))

            if self.usecols is not None:
                names = self._filter_usecols(names)

            # columns as list
            alldata = [x[1] for x in data_tups]
            if self.usecols is None:
                self._check_data_length(names, alldata)

            data = {k: v for k, (i, v) in zip(names, data_tups)}

            names, date_data = self._do_date_conversions(names, data)
            index, column_names = self._make_index(date_data, alldata, names)

        return index, column_names, date_data

    def _filter_usecols(self, names: Sequence[Hashable]) -> Sequence[Hashable]:
        # hackish
        usecols = self._evaluate_usecols(self.usecols, names)
        if usecols is not None and len(names) != len(usecols):
            names = [
                name for i, name in enumerate(names) if i in usecols or name in usecols
            ]
        return names

    def _maybe_parse_dates(self, values, index: int, try_parse_dates: bool = True):
        if try_parse_dates and self._should_parse_dates(index):
            values = self._date_conv(
                values,
                col=self.index_names[index] if self.index_names is not None else None,
            )
        return values


def _concatenate_chunks(chunks: list[dict[int, ArrayLike]]) -> dict:
    """
    Concatenate chunks of data read with low_memory=True.

    The tricky part is handling Categoricals, where different chunks
    may have different inferred categories.
    """
    names = list(chunks[0].keys())
    warning_columns = []

    result: dict = {}
    for name in names:
        arrs = [chunk.pop(name) for chunk in chunks]
        # Check each arr for consistent types.
        dtypes = {a.dtype for a in arrs}
        non_cat_dtypes = {x for x in dtypes if not isinstance(x, CategoricalDtype)}

        dtype = dtypes.pop()
        if isinstance(dtype, CategoricalDtype):
            result[name] = union_categoricals(arrs, sort_categories=False)
        else:
            result[name] = concat_compat(arrs)
            if len(non_cat_dtypes) > 1 and result[name].dtype == np.dtype(object):
                warning_columns.append(str(name))

    if warning_columns:
        warning_names = ",".join(warning_columns)
        warning_message = " ".join(
            [
                f"Columns ({warning_names}) have mixed types. "
                f"Specify dtype option on import or set low_memory=False."
            ]
        )
        warnings.warn(warning_message, DtypeWarning, stacklevel=find_stack_level())
    return result


def ensure_dtype_objs(
    dtype: DtypeArg | dict[Hashable, DtypeArg] | None
) -> DtypeObj | dict[Hashable, DtypeObj] | None:
    """
    Ensure we have either None, a dtype object, or a dictionary mapping to
    dtype objects.
    """
    if isinstance(dtype, defaultdict):
        # "None" not callable  [misc]
        default_dtype = pandas_dtype(dtype.default_factory())  # type: ignore[misc]
        dtype_converted: defaultdict = defaultdict(lambda: default_dtype)
        for key in dtype.keys():
            dtype_converted[key] = pandas_dtype(dtype[key])
        return dtype_converted
    elif isinstance(dtype, dict):
        return {k: pandas_dtype(dtype[k]) for k in dtype}
    elif dtype is not None:
        return pandas_dtype(dtype)
    return dtype
<|endoftext|>"""
Expose public exceptions & warnings
"""
from __future__ import annotations

import ctypes

from pandas._config.config import OptionError

from pandas._libs.tslibs import (
    OutOfBoundsDatetime,
    OutOfBoundsTimedelta,
)

from pandas.util.version import InvalidVersion


class IntCastingNaNError(ValueError):
    """
    Exception raised when converting (``astype``) an array with NaN to an integer type.

    Examples
    --------
    >>> pd.DataFrame(np.array([[1, np.nan], [2, 3]]), dtype="i8")
    Traceback (most recent call last):
    IntCastingNaNError: Cannot convert non-finite values (NA or inf) to integer
    """


class NullFrequencyError(ValueError):
    """
    Exception raised when a ``freq`` cannot be null.

    Particularly ``DatetimeIndex.shift``, ``TimedeltaIndex.shift``,
    ``PeriodIndex.shift``.

    Examples
    --------
    >>> df = pd.DatetimeIndex(["2011-01-01 10:00", "2011-01-01"], freq=None)
    >>> df.shift(2)
    Traceback (most recent call last):
    NullFrequencyError: Cannot shift with no freq
    """


class PerformanceWarning(Warning):
    """
    Warning raised when there is a possible performance impact.

    Examples
    --------
    >>> df = pd.DataFrame({"jim": [0, 0, 1, 1],
    ...                    "joe": ["x", "x", "z", "y"],
    ...                    "jolie": [1, 2, 3, 4]})
    >>> df = df.set_index(["jim", "joe"])
    >>> df
              jolie
    jim  joe
    0    x    1
         x    2
    1    z    3
         y    4
    >>> df.loc[(1, 'z')]  # doctest: +SKIP
    # PerformanceWarning: indexing past lexsort depth may impact performance.
    df.loc[(1, 'z')]
              jolie
    jim  joe
    1    z        3
    """


class UnsupportedFunctionCall(ValueError):
    """
    Exception raised when attempting to call a unsupported numpy function.

    For example, ``np.cumsum(groupby_object)``.

    Examples
    --------
    >>> df = pd.DataFrame({"A": [0, 0, 1, 1],
    ...                    "B": ["x", "x", "z", "y"],
    ...                    "C": [1, 2, 3, 4]}
    ...                   )
    >>> np.cumsum(df.groupby(["A"]))
    Traceback (most recent call last):
    UnsupportedFunctionCall: numpy operations are not valid with groupby.
    Use .groupby(...).cumsum() instead
    """


class UnsortedIndexError(KeyError):
    """
    Error raised when slicing a MultiIndex which has not been lexsorted.

    Subclass of `KeyError`.

    Examples
    --------
    >>> df = pd.DataFrame({"cat": [0, 0, 1, 1],
    ...                    "color": ["white", "white", "brown", "black"],
    ...                    "lives": [4, 4, 3, 7]},
    ...                   )
    >>> df = df.set_index(["cat", "color"])
    >>> df
                lives
    cat  color
    0    white    4
         white    4
    1    brown    3
         black    7
    >>> df.loc[(0, "black"):(1, "white")]
    Traceback (most recent call last):
    UnsortedIndexError: 'Key length (2) was greater
    than MultiIndex lexsort depth (1)'
    """


class ParserError(ValueError):
    """
    Exception that is raised by an error encountered in parsing file contents.

    This is a generic error raised for errors encountered when functions like
    `read_csv` or `read_html` are parsing contents of a file.

    See Also
    --------
    read_csv : Read CSV (comma-separated) file into a DataFrame.
    read_html : Read HTML table into a DataFrame.

    Examples
    --------
    >>> data = '''a,b,c
    ... cat,foo,bar
    ... dog,foo,"baz'''
    >>> from io import StringIO
    >>> pd.read_csv(StringIO(data), skipfooter=1, engine='python')
    Traceback (most recent call last):
    ParserError: ',' expected after '"'. Error could possibly be due
    to parsing errors in the skipped footer rows
    """


class DtypeWarning(Warning):
    """
    Warning raised when reading different dtypes in a column from a file.

    Raised for a dtype incompatibility. This can happen whenever `read_csv`
    or `read_table` encounter non-uniform dtypes in a column(s) of a given
    CSV file.

    See Also
    --------
    read_csv : Read CSV (comma-separated) file into a DataFrame.
    read_table : Read general delimited file into a DataFrame.

    Notes
    -----
    This warning is issued when dealing with larger files because the dtype
    checking happens per chunk read.

    Despite the warning, the CSV file is read with mixed types in a single
    column which will be an object type. See the examples below to better
    understand this issue.

    Examples
    --------
    This example creates and reads a large CSV file with a column that contains
    `int` and `str`.

    >>> df = pd.DataFrame({'a': (['1'] * 100000 + ['X'] * 100000 +
    ...                          ['1'] * 100000),
    ...                    'b': ['b'] * 300000})  # doctest: +SKIP
    >>> df.to_csv('test.csv', index=False)  # doctest: +SKIP
    >>> df2 = pd.read_csv('test.csv')  # doctest: +SKIP
    ... # DtypeWarning: Columns (0) have mixed types

    Important to notice that ``df2`` will contain both `str` and `int` for the
    same input, '1'.

    >>> df2.iloc[262140, 0]  # doctest: +SKIP
    '1'
    >>> type(df2.iloc[262140, 0])  # doctest: +SKIP
    <class 'str'>
    >>> df2.iloc[262150, 0]  # doctest: +SKIP
    1
    >>> type(df2.iloc[262150, 0])  # doctest: +SKIP
    <class 'int'>

    One way to solve this issue is using the `dtype` parameter in the
    `read_csv` and `read_table` functions to explicit the conversion:

    >>> df2 = pd.read_csv('test.csv', sep=',', dtype={'a': str})  # doctest: +SKIP

    No warning was issued.
    """


class EmptyDataError(ValueError):
    """
    Exception raised in ``pd.read_csv`` when empty data or header is encountered.

    Examples
    --------
    >>> from io import StringIO
    >>> empty = StringIO()
    >>> pd.read_csv(empty)
    Traceback (most recent call last):
    EmptyDataError: No columns to parse from file
    """


class ParserWarning(Warning):
    """
    Warning raised when reading a file that doesn't use the default 'c' parser.

    Raised by `pd.read_csv` and `pd.read_table` when it is necessary to change
    parsers, generally from the default 'c' parser to 'python'.

    It happens due to a lack of support or functionality for parsing a
    particular attribute of a CSV file with the requested engine.

    Currently, 'c' unsupported options include the following parameters:

    1. `sep` other than a single character (e.g. regex separators)
    2. `skipfooter` higher than 0
    3. `sep=None` with `delim_whitespace=False`

    The warning can be avoided by adding `engine='python'` as a parameter in
    `pd.read_csv` and `pd.read_table` methods.

    See Also
    --------
    pd.read_csv : Read CSV (comma-separated) file into DataFrame.
    pd.read_table : Read general delimited file into DataFrame.

    Examples
    --------
    Using a `sep` in `pd.read_csv` other than a single character:

    >>> import io
    >>> csv = '''a;b;c
    ...           1;1,8
    ...           1;2,1'''
    >>> df = pd.read_csv(io.StringIO(csv), sep='[;,]')  # doctest: +SKIP
    ... # ParserWarning: Falling back to the 'python' engine...

    Adding `engine='python'` to `pd.read_csv` removes the Warning:

    >>> df = pd.read_csv(io.StringIO(csv), sep='[;,]', engine='python')
    """


class MergeError(ValueError):
    """
    Exception raised when merging data.

    Subclass of ``ValueError``.

    Examples
    --------
    >>> left = pd.DataFrame({"a": ["a", "b", "b", "d"],
    ...                     "b": ["cat", "dog", "weasel", "horse"]},
    ...                     index=range(4))
    >>> right = pd.DataFrame({"a": ["a", "b", "c", "d"],
    ...                      "c": ["meow", "bark", "chirp", "nay"]},
    ...                      index=range(4)).set_index("a")
    >>> left.join(right, on="a", validate="one_to_one",)
    Traceback (most recent call last):
    MergeError: Merge keys are not unique in left dataset; not a one-to-one merge
    """


class AbstractMethodError(NotImplementedError):
    """
    Raise this error instead of NotImplementedError for abstract methods.

    Examples
    --------
    >>> class Foo:
    ...     @classmethod
    ...     def classmethod(cls):
    ...         raise pd.errors.AbstractMethodError(cls, methodtype="classmethod")
    ...     def method(self):
    ...         raise pd.errors.AbstractMethodError(self)
    >>> test = Foo.classmethod()
    Traceback (most recent call last):
    AbstractMethodError: This classmethod must be defined in the concrete class Foo

    >>> test2 = Foo().method()
    Traceback (most recent call last):
    AbstractMethodError: This classmethod must be defined in the concrete class Foo
    """

    def __init__(self, class_instance, methodtype: str = "method") -> None:
        types = {"method", "classmethod", "staticmethod", "property"}
        if methodtype not in types:
            raise ValueError(
                f"methodtype must be one of {methodtype}, got {types} instead."
            )
        self.methodtype = methodtype
        self.class_instance = class_instance

    def __str__(self) -> str:
        if self.methodtype == "classmethod":
            name = self.class_instance.__name__
        else:
            name = type(self.class_instance).__name__
        return f"This {self.methodtype} must be defined in the concrete class {name}"


class NumbaUtilError(Exception):
    """
    Error raised for unsupported Numba engine routines.

    Examples
    --------
    >>> df = pd.DataFrame({"key": ["a", "a", "b", "b"], "data": [1, 2, 3, 4]},
    ...                   columns=["key", "data"])
    >>> def incorrect_function(x):
    ...     return sum(x) * 2.7
    >>> df.groupby("key").agg(incorrect_function, engine="numba")
    Traceback (most recent call last):
    NumbaUtilError: The first 2 arguments to incorrect_function
    must be ['values', 'index']
    """


class DuplicateLabelError(ValueError):
    """
    Error raised when an operation would introduce duplicate labels.

    Examples
    --------
    >>> s = pd.Series([0, 1, 2], index=['a', 'b', 'c']).set_flags(
    ...     allows_duplicate_labels=False
    ... )
    >>> s.reindex(['a', 'a', 'b'])
    Traceback (most recent call last):
       ...
    DuplicateLabelError: Index has duplicates.
          positions
    label
    a        [0, 1]
    """


class InvalidIndexError(Exception):
    """
    Exception raised when attempting to use an invalid index key.

    Examples
    --------
    >>> idx = pd.MultiIndex.from_product([["x", "y"], [0, 1]])
    >>> df = pd.DataFrame([[1, 1, 2, 2],
    ...                   [3, 3, 4, 4]], columns=idx)
    >>> df
        x       y
        0   1   0   1
    0   1   1   2   2
    1   3   3   4   4
    >>> df[:, 0]
    Traceback (most recent call last):
    InvalidIndexError: (slice(None, None, None), 0)
    """


class DataError(Exception):
    """
    Exceptionn raised when performing an operation on non-numerical data.

    For example, calling ``ohlc`` on a non-numerical column or a function
    on a rolling window.

    Examples
    --------
    >>> ser = pd.Series(['a', 'b', 'c'])
    >>> ser.rolling(2).sum()
    Traceback (most recent call last):
    DataError: No numeric types to aggregate
    """


class SpecificationError(Exception):
    """
    Exception raised by ``agg`` when the functions are ill-specified.

    The exception raised in two scenarios.

    The first way is calling ``agg`` on a
    Dataframe or Series using a nested renamer (dict-of-dict).

    The second way is calling ``agg`` on a Dataframe with duplicated functions
    names without assigning column name.

    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 1, 1, 2, 2],
    ...                    'B': range(5),
    ...                    'C': range(5)})
    >>> df.groupby('A').B.agg({'foo': 'count'}) # doctest: +SKIP
    ... # SpecificationError: nested renamer is not supported

    >>> df.groupby('A').agg({'B': {'foo': ['sum', 'max']}}) # doctest: +SKIP
    ... # SpecificationError: nested renamer is not supported

    >>> df.groupby('A').agg(['min', 'min']) # doctest: +SKIP
    ... # SpecificationError: nested renamer is not supported
    """


class SettingWithCopyError(ValueError):
    """
    Exception raised when trying to set on a copied slice from a ``DataFrame``.

    The ``mode.chained_assignment`` needs to be set to set to 'raise.' This can
    happen unintentionally when chained indexing.

    For more information on evaluation order,
    see :ref:`the user guide<indexing.evaluation_order>`.

    For more information on view vs. copy,
    see :ref:`the user guide<indexing.view_versus_copy>`.

    Examples
    --------
    >>> pd.options.mode.chained_assignment = 'raise'
    >>> df = pd.DataFrame({'A': [1, 1, 1, 2, 2]}, columns=['A'])
    >>> df.loc[0:3]['A'] = 'a' # doctest: +SKIP
    ... # SettingWithCopyError: A value is trying to be set on a copy of a...
    """


class SettingWithCopyWarning(Warning):
    """
    Warning raised when trying to set on a copied slice from a ``DataFrame``.

    The ``mode.chained_assignment`` needs to be set to set to 'warn.'
    'Warn' is the default option. This can happen unintentionally when
    chained indexing.

    For more information on evaluation order,
    see :ref:`the user guide<indexing.evaluation_order>`.

    For more information on view vs. copy,
    see :ref:`the user guide<indexing.view_versus_copy>`.

    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 1, 1, 2, 2]}, columns=['A'])
    >>> df.loc[0:3]['A'] = 'a' # doctest: +SKIP
    ... # SettingWithCopyWarning: A value is trying to be set on a copy of a...
    """


class ChainedAssignmentError(Warning):
    """
    Warning raised when trying to set using chained assignment.

    When the ``mode.copy_on_write`` option is enabled, chained assignment can
    never work. In such a situation, we are always setting into a temporary
    object that is the result of an indexing operation (getitem), which under
    Copy-on-Write always behaves as a copy. Thus, assigning through a chain
    can never update the original Series or DataFrame.

    For more information on view vs. copy,
    see :ref:`the user guide<indexing.view_versus_copy>`.

    Examples
    --------
    >>> pd.options.mode.copy_on_write = True
    >>> df = pd.DataFrame({'A': [1, 1, 1, 2, 2]}, columns=['A'])
    >>> df["A"][0:3] = 10 # doctest: +SKIP
    ... # ChainedAssignmentError: ...
    >>> pd.options.mode.copy_on_write = False
    """


_chained_assignment_msg = (
    "A value is trying to be set on a copy of a DataFrame or Series "
    "through chained assignment.\n"
    "When using the Copy-on-Write mode, such chained assignment never works "
    "to update the original DataFrame or Series, because the intermediate "
    "object on which we are setting values always behaves as a copy.\n\n"
    "Try using '.loc[row_indexer, col_indexer] = value' instead, to perform "
    "the assignment in a single step.\n\n"
    "See the caveats in the documentation: "
    "https://pandas.pydata.org/pandas-docs/stable/user_guide/"
    "indexing.html#returning-a-view-versus-a-copy"
)


_chained_assignment_method_msg = (
    "A value is trying to be set on a copy of a DataFrame or Series "
    "through chained assignment using an inplace method.\n"
    "When using the Copy-on-Write mode, such inplace method never works "
    "to update the original DataFrame or Series, because the intermediate "
    "object on which we are setting values always behaves as a copy.\n\n"
    "For example, when doing 'df[col].method(value, inplace=True)', try "
    "using 'df.method({col: value}, inplace=True)' instead, to perform "
    "the operation inplace on the original object.\n\n"
)


_chained_assignment_warning_msg = (
    "ChainedAssignmentError: behaviour will change in pandas 3.0!\n"
    "You are setting values through chained assignment. Currently this works "
    "in certain cases, but when using Copy-on-Write (which will become the "
    "default behaviour in pandas 3.0) this will never work to update the "
    "original DataFrame or Series, because the intermediate object on which "
    "we are setting values will behave as a copy.\n"
    "A typical example is when you are setting values in a column of a "
    "DataFrame, like:\n\n"
    'df["col"][row_indexer] = value\n\n'
    'Use `df.loc[row_indexer, "col"] = values` instead, to perform the '
    "assignment in a single step and ensure this keeps updating the original `df`.\n\n"
    "See the caveats in the documentation: "
    "https://pandas.pydata.org/pandas-docs/stable/user_guide/"
    "indexing.html#returning-a-view-versus-a-copy\n"
)


_chained_assignment_warning_method_msg = (
    "A value is trying to be set on a copy of a DataFrame or Series "
    "through chained assignment using an inplace method.\n"
    "The behavior will change in pandas 3.0. This inplace method will "
    "never work because the intermediate object on which we are setting "
    "values always behaves as a copy.\n\n"
    "For example, when doing 'df[col].method(value, inplace=True)', try "
    "using 'df.method({col: value}, inplace=True)' or "
    "df[col] = df[col].method(value) instead, to perform "
    "the operation inplace on the original object.\n\n"
)


def _check_cacher(obj):
    # This is a mess, selection paths that return a view set the _cacher attribute
    # on the Series; most of them also set _item_cache which adds 1 to our relevant
    # reference count, but iloc does not, so we have to check if we are actually
    # in the item cache
    if hasattr(obj, "_cacher"):
        parent = obj._cacher[1]()
        # parent could be dead
        if parent is None:
            return False
        if hasattr(parent, "_item_cache"):
            if obj._cacher[0] in parent._item_cache:
                # Check if we are actually the item from item_cache, iloc creates a
                # new object
                return obj is parent._item_cache[obj._cacher[0]]
    return False


class NumExprClobberingError(NameError):
    """
    Exception raised when trying to use a built-in numexpr name as a variable name.

    ``eval`` or ``query`` will throw the error if the engine is set
    to 'numexpr'. 'numexpr' is the default engine value for these methods if the
    numexpr package is installed.

    Examples
    --------
    >>> df = pd.DataFrame({'abs': [1, 1, 1]})
    >>> df.query("abs > 2") # doctest: +SKIP
    ... # NumExprClobberingError: Variables in expression "(abs) > (2)" overlap...
    >>> sin, a = 1, 2
    >>> pd.eval("sin + a", engine='numexpr') # doctest: +SKIP
    ... # NumExprClobberingError: Variables in expression "(sin) + (a)" overlap...
    """


class UndefinedVariableError(NameError):
    """
    Exception raised by ``query`` or ``eval`` when using an undefined variable name.

    It will also specify whether the undefined variable is local or not.

    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 1, 1]})
    >>> df.query("A > x") # doctest: +SKIP
    ... # UndefinedVariableError: name 'x' is not defined
    >>> df.query("A > @y") # doctest: +SKIP
    ... # UndefinedVariableError: local variable 'y' is not defined
    >>> pd.eval('x + 1') # doctest: +SKIP
    ... # UndefinedVariableError: name 'x' is not defined
    """

    def __init__(self, name: str, is_local: bool | None = None) -> None:
        base_msg = f"{repr(name)} is not defined"
        if is_local:
            msg = f"local variable {base_msg}"
        else:
            msg = f"name {base_msg}"
        super().__init__(msg)


class IndexingError(Exception):
    """
    Exception is raised when trying to index and there is a mismatch in dimensions.

    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 1, 1]})
    >>> df.loc[..., ..., 'A'] # doctest: +SKIP
    ... # IndexingError: indexer may only contain one '...' entry
    >>> df = pd.DataFrame({'A': [1, 1, 1]})
    >>> df.loc[1, ..., ...] # doctest: +SKIP
    ... # IndexingError: Too many indexers
    >>> df[pd.Series([True], dtype=bool)] # doctest: +SKIP
    ... # IndexingError: Unalignable boolean Series provided as indexer...
    >>> s = pd.Series(range(2),
    ...               index = pd.MultiIndex.from_product([["a", "b"], ["c"]]))
    >>> s.loc["a", "c", "d"] # doctest: +SKIP
    ... # IndexingError: Too many indexers
    """


class PyperclipException(RuntimeError):
    """
    Exception raised when clipboard functionality is unsupported.

    Raised by ``to_clipboard()`` and ``read_clipboard()``.
    """


class PyperclipWindowsException(PyperclipException):
    """
    Exception raised when clipboard functionality is unsupported by Windows.

    Access to the clipboard handle would be denied due to some other
    window process is accessing it.
    """

    def __init__(self, message: str) -> None:
        # attr only exists on Windows, so typing fails on other platforms
        message += f" ({ctypes.WinError()})"  # type: ignore[attr-defined]
        super().__init__(message)


class CSSWarning(UserWarning):
    """
    Warning is raised when converting css styling fails.

    This can be due to the styling not having an equivalent value or because the
    styling isn't properly formatted.

    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 1, 1]})
    >>> df.style.applymap(
    ...     lambda x: 'background-color: blueGreenRed;'
    ... ).to_excel('styled.xlsx')  # doctest: +SKIP
    CSSWarning: Unhandled color format: 'blueGreenRed'
    >>> df.style.applymap(
    ...     lambda x: 'border: 1px solid red red;'
    ... ).to_excel('styled.xlsx')  # doctest: +SKIP
    CSSWarning: Unhandled color format: 'blueGreenRed'
    """


class PossibleDataLossError(Exception):
    """
    Exception raised when trying to open a HDFStore file when already opened.

    Examples
    --------
    >>> store = pd.HDFStore('my-store', 'a') # doctest: +SKIP
    >>> store.open("w") # doctest: +SKIP
    ... # PossibleDataLossError: Re-opening the file [my-store] with mode [a]...
    """


class ClosedFileError(Exception):
    """
    Exception is raised when trying to perform an operation on a closed HDFStore file.

    Examples
    --------
    >>> store = pd.HDFStore('my-store', 'a') # doctest: +SKIP
    >>> store.close() # doctest: +SKIP
    >>> store.keys() # doctest: +SKIP
    ... # ClosedFileError: my-store file is not open!
    """


class IncompatibilityWarning(Warning):
    """
    Warning raised when trying to use where criteria on an incompatible HDF5 file.
    """


class AttributeConflictWarning(Warning):
    """
    Warning raised when index attributes conflict when using HDFStore.

    Occurs when attempting to append an index with a different
    name than the existing index on an HDFStore or attempting to append an index with a
    different frequency than the existing index on an HDFStore.

    Examples
    --------
    >>> idx1 = pd.Index(['a', 'b'], name='name1')
    >>> df1 = pd.DataFrame([[1, 2], [3, 4]], index=idx1)
    >>> df1.to_hdf('file', 'data', 'w', append=True)  # doctest: +SKIP
    >>> idx2 = pd.Index(['c', 'd'], name='name2')
    >>> df2 = pd.DataFrame([[5, 6], [7, 8]], index=idx2)
    >>> df2.to_hdf('file', 'data', 'a', append=True)  # doctest: +SKIP
    AttributeConflictWarning: the [index_name] attribute of the existing index is
    [name1] which conflicts with the new [name2]...
    """


class DatabaseError(OSError):
    """
    Error is raised when executing sql with bad syntax or sql that throws an error.

    Examples
    --------
    >>> from sqlite3 import connect
    >>> conn = connect(':memory:')
    >>> pd.read_sql('select * test', conn) # doctest: +SKIP
    ... # DatabaseError: Execution failed on sql 'test': near "test": syntax error
    """


class PossiblePrecisionLoss(Warning):
    """
    Warning raised by to_stata on a column with a value outside or equal to int64.

    When the column value is outside or equal to the int64 value the column is
    converted to a float64 dtype.

    Examples
    --------
    >>> df = pd.DataFrame({"s": pd.Series([1, 2**53], dtype=np.int64)})
    >>> df.to_stata('test') # doctest: +SKIP
    ... # PossiblePrecisionLoss: Column converted from int64 to float64...
    """


class ValueLabelTypeMismatch(Warning):
    """
    Warning raised by to_stata on a category column that contains non-string values.

    Examples
    --------
    >>> df = pd.DataFrame({"categories": pd.Series(["a", 2], dtype="category")})
    >>> df.to_stata('test') # doctest: +SKIP
    ... # ValueLabelTypeMismatch: Stata value labels (pandas categories) must be str...
    """


class InvalidColumnName(Warning):
    """
    Warning raised by to_stata the column contains a non-valid stata name.

    Because the column name is an invalid Stata variable, the name needs to be
    converted.

    Examples
    --------
    >>> df = pd.DataFrame({"0categories": pd.Series([2, 2])})
    >>> df.to_stata('test') # doctest: +SKIP
    ... # InvalidColumnName: Not all pandas column names were valid Stata variable...
    """


class CategoricalConversionWarning(Warning):
    """
    Warning is raised when reading a partial labeled Stata file using a iterator.

    Examples
    --------
    >>> from pandas.io.stata import StataReader
    >>> with StataReader('dta_file', chunksize=2) as reader: # doctest: +SKIP
    ...   for i, block in enumerate(reader):
    ...      print(i, block)
    ... # CategoricalConversionWarning: One or more series with value labels...
    """


class LossySetitemError(Exception):
    """
    Raised when trying to do a __setitem__ on an np.ndarray that is not lossless.

    Notes
    -----
    This is an internal error.
    """


class NoBufferPresent(Exception):
    """
    Exception is raised in _get_data_buffer to signal that there is no requested buffer.
    """


class InvalidComparison(Exception):
    """
    Exception is raised by _validate_comparison_value to indicate an invalid comparison.

    Notes
    -----
    This is an internal error.
    """


__all__ = [
    "AbstractMethodError",
    "AttributeConflictWarning",
    "CategoricalConversionWarning",
    "ClosedFileError",
    "CSSWarning",
    "DatabaseError",
    "DataError",
    "DtypeWarning",
    "DuplicateLabelError",
    "EmptyDataError",
    "IncompatibilityWarning",
    "IntCastingNaNError",
    "InvalidColumnName",
    "InvalidComparison",
    "InvalidIndexError",
    "InvalidVersion",
    "IndexingError",
    "LossySetitemError",
    "MergeError",
    "NoBufferPresent",
    "NullFrequencyError",
    "NumbaUtilError",
    "NumExprClobberingError",
    "OptionError",
    "OutOfBoundsDatetime",
    "OutOfBoundsTimedelta",
    "ParserError",
    "ParserWarning",
    "PerformanceWarning",
    "PossibleDataLossError",
    "PossiblePrecisionLoss",
    "PyperclipException",
    "PyperclipWindowsException",
    "SettingWithCopyError",
    "SettingWithCopyWarning",
    "SpecificationError",
    "UndefinedVariableError",
    "UnsortedIndexError",
    "UnsupportedFunctionCall",
    "ValueLabelTypeMismatch",
]
<|endoftext|>__all__ = [
    "NaT",
    "NaTType",
    "OutOfBoundsDatetime",
    "Period",
    "Timedelta",
    "Timestamp",
    "iNaT",
    "Interval",
]


# Below imports needs to happen first to ensure pandas top level
# module gets monkeypatched with the pandas_datetime_CAPI
# see pandas_datetime_exec in pd_datetime.c
import pandas._libs.pandas_parser  # isort: skip # type: ignore[reportUnusedImport]
import pandas._libs.pandas_datetime  # noqa: F401 # isort: skip # type: ignore[reportUnusedImport]
from pandas._libs.interval import Interval
from pandas._libs.tslibs import (
    NaT,
    NaTType,
    OutOfBoundsDatetime,
    Period,
    Timedelta,
    Timestamp,
    iNaT,
)
<|endoftext|>__all__ = [
    "dtypes",
    "localize_pydatetime",
    "NaT",
    "NaTType",
    "iNaT",
    "nat_strings",
    "OutOfBoundsDatetime",
    "OutOfBoundsTimedelta",
    "IncompatibleFrequency",
    "Period",
    "Resolution",
    "Timedelta",
    "normalize_i8_timestamps",
    "is_date_array_normalized",
    "dt64arr_to_periodarr",
    "delta_to_nanoseconds",
    "ints_to_pydatetime",
    "ints_to_pytimedelta",
    "get_resolution",
    "Timestamp",
    "tz_convert_from_utc_single",
    "tz_convert_from_utc",
    "to_offset",
    "Tick",
    "BaseOffset",
    "tz_compare",
    "is_unitless",
    "astype_overflowsafe",
    "get_unit_from_dtype",
    "periods_per_day",
    "periods_per_second",
    "guess_datetime_format",
    "add_overflowsafe",
    "get_supported_dtype",
    "is_supported_dtype",
]

from pandas._libs.tslibs import dtypes  # pylint: disable=import-self
from pandas._libs.tslibs.conversion import localize_pydatetime
from pandas._libs.tslibs.dtypes import (
    Resolution,
    periods_per_day,
    periods_per_second,
)
from pandas._libs.tslibs.nattype import (
    NaT,
    NaTType,
    iNaT,
    nat_strings,
)
from pandas._libs.tslibs.np_datetime import (
    OutOfBoundsDatetime,
    OutOfBoundsTimedelta,
    add_overflowsafe,
    astype_overflowsafe,
    get_supported_dtype,
    is_supported_dtype,
    is_unitless,
    py_get_unit_from_dtype as get_unit_from_dtype,
)
from pandas._libs.tslibs.offsets import (
    BaseOffset,
    Tick,
    to_offset,
)
from pandas._libs.tslibs.parsing import guess_datetime_format
from pandas._libs.tslibs.period import (
    IncompatibleFrequency,
    Period,
)
from pandas._libs.tslibs.timedeltas import (
    Timedelta,
    delta_to_nanoseconds,
    ints_to_pytimedelta,
)
from pandas._libs.tslibs.timestamps import Timestamp
from pandas._libs.tslibs.timezones import tz_compare
from pandas._libs.tslibs.tzconversion import tz_convert_from_utc_single
from pandas._libs.tslibs.vectorized import (
    dt64arr_to_periodarr,
    get_resolution,
    ints_to_pydatetime,
    is_date_array_normalized,
    normalize_i8_timestamps,
    tz_convert_from_utc,
)
<|endoftext|>from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pandas._libs import lib
from pandas._libs.algos import unique_deltas
from pandas._libs.tslibs import (
    Timestamp,
    get_unit_from_dtype,
    periods_per_day,
    tz_convert_from_utc,
)
from pandas._libs.tslibs.ccalendar import (
    DAYS,
    MONTH_ALIASES,
    MONTH_NUMBERS,
    MONTHS,
    int_to_weekday,
)
from pandas._libs.tslibs.dtypes import (
    OFFSET_TO_PERIOD_FREQSTR,
    freq_to_period_freqstr,
)
from pandas._libs.tslibs.fields import (
    build_field_sarray,
    month_position_check,
)
from pandas._libs.tslibs.offsets import (
    DateOffset,
    Day,
    to_offset,
)
from pandas._libs.tslibs.parsing import get_rule_month
from pandas.util._decorators import cache_readonly

from pandas.core.dtypes.common import is_numeric_dtype
from pandas.core.dtypes.dtypes import (
    DatetimeTZDtype,
    PeriodDtype,
)
from pandas.core.dtypes.generic import (
    ABCIndex,
    ABCSeries,
)

from pandas.core.algorithms import unique

if TYPE_CHECKING:
    from pandas._typing import npt

    from pandas import (
        DatetimeIndex,
        Series,
        TimedeltaIndex,
    )
    from pandas.core.arrays.datetimelike import DatetimeLikeArrayMixin
# --------------------------------------------------------------------
# Offset related functions

_need_suffix = ["QS", "BQE", "BQS", "YS", "BYE", "BYS"]

for _prefix in _need_suffix:
    for _m in MONTHS:
        key = f"{_prefix}-{_m}"
        OFFSET_TO_PERIOD_FREQSTR[key] = OFFSET_TO_PERIOD_FREQSTR[_prefix]

for _prefix in ["Y", "Q"]:
    for _m in MONTHS:
        _alias = f"{_prefix}-{_m}"
        OFFSET_TO_PERIOD_FREQSTR[_alias] = _alias

for _d in DAYS:
    OFFSET_TO_PERIOD_FREQSTR[f"W-{_d}"] = f"W-{_d}"


def get_period_alias(offset_str: str) -> str | None:
    """
    Alias to closest period strings BQ->Q etc.
    """
    return OFFSET_TO_PERIOD_FREQSTR.get(offset_str, None)


# ---------------------------------------------------------------------
# Period codes


def infer_freq(
    index: DatetimeIndex | TimedeltaIndex | Series | DatetimeLikeArrayMixin,
) -> str | None:
    """
    Infer the most likely frequency given the input index.

    Parameters
    ----------
    index : DatetimeIndex, TimedeltaIndex, Series or array-like
      If passed a Series will use the values of the series (NOT THE INDEX).

    Returns
    -------
    str or None
        None if no discernible frequency.

    Raises
    ------
    TypeError
        If the index is not datetime-like.
    ValueError
        If there are fewer than three values.

    Examples
    --------
    >>> idx = pd.date_range(start='2020/12/01', end='2020/12/30', periods=30)
    >>> pd.infer_freq(idx)
    'D'
    """
    from pandas.core.api import DatetimeIndex

    if isinstance(index, ABCSeries):
        values = index._values
        if not (
            lib.is_np_dtype(values.dtype, "mM")
            or isinstance(values.dtype, DatetimeTZDtype)
            or values.dtype == object
        ):
            raise TypeError(
                "cannot infer freq from a non-convertible dtype "
                f"on a Series of {index.dtype}"
            )
        index = values

    inferer: _FrequencyInferer

    if not hasattr(index, "dtype"):
        pass
    elif isinstance(index.dtype, PeriodDtype):
        raise TypeError(
            "PeriodIndex given. Check the `freq` attribute "
            "instead of using infer_freq."
        )
    elif lib.is_np_dtype(index.dtype, "m"):
        # Allow TimedeltaIndex and TimedeltaArray
        inferer = _TimedeltaFrequencyInferer(index)
        return inferer.get_freq()

    elif is_numeric_dtype(index.dtype):
        raise TypeError(
            f"cannot infer freq from a non-convertible index of dtype {index.dtype}"
        )

    if not isinstance(index, DatetimeIndex):
        index = DatetimeIndex(index)

    inferer = _FrequencyInferer(index)
    return inferer.get_freq()


class _FrequencyInferer:
    """
    Not sure if I can avoid the state machine here
    """

    def __init__(self, index) -> None:
        self.index = index
        self.i8values = index.asi8

        # For get_unit_from_dtype we need the dtype to the underlying ndarray,
        #  which for tz-aware is not the same as index.dtype
        if isinstance(index, ABCIndex):
            # error: Item "ndarray[Any, Any]" of "Union[ExtensionArray,
            # ndarray[Any, Any]]" has no attribute "_ndarray"
            self._creso = get_unit_from_dtype(
                index._data._ndarray.dtype  # type: ignore[union-attr]
            )
        else:
            # otherwise we have DTA/TDA
            self._creso = get_unit_from_dtype(index._ndarray.dtype)

        # This moves the values, which are implicitly in UTC, to the
        # the timezone so they are in local time
        if hasattr(index, "tz"):
            if index.tz is not None:
                self.i8values = tz_convert_from_utc(
                    self.i8values, index.tz, reso=self._creso
                )

        if len(index) < 3:
            raise ValueError("Need at least 3 dates to infer frequency")

        self.is_monotonic = (
            self.index._is_monotonic_increasing or self.index._is_monotonic_decreasing
        )

    @cache_readonly
    def deltas(self) -> npt.NDArray[np.int64]:
        return unique_deltas(self.i8values)

    @cache_readonly
    def deltas_asi8(self) -> npt.NDArray[np.int64]:
        # NB: we cannot use self.i8values here because we may have converted
        #  the tz in __init__
        return unique_deltas(self.index.asi8)

    @cache_readonly
    def is_unique(self) -> bool:
        return len(self.deltas) == 1

    @cache_readonly
    def is_unique_asi8(self) -> bool:
        return len(self.deltas_asi8) == 1

    def get_freq(self) -> str | None:
        """
        Find the appropriate frequency string to describe the inferred
        frequency of self.i8values

        Returns
        -------
        str or None
        """
        if not self.is_monotonic or not self.index._is_unique:
            return None

        delta = self.deltas[0]
        ppd = periods_per_day(self._creso)
        if delta and _is_multiple(delta, ppd):
            return self._infer_daily_rule()

        # Business hourly, maybe. 17: one day / 65: one weekend
        if self.hour_deltas in ([1, 17], [1, 65], [1, 17, 65]):
            return "bh"

        # Possibly intraday frequency.  Here we use the
        # original .asi8 values as the modified values
        # will not work around DST transitions.  See #8772
        if not self.is_unique_asi8:
            return None

        delta = self.deltas_asi8[0]
        pph = ppd // 24
        ppm = pph // 60
        pps = ppm // 60
        if _is_multiple(delta, pph):
            # Hours
            return _maybe_add_count("h", delta / pph)
        elif _is_multiple(delta, ppm):
            # Minutes
            return _maybe_add_count("min", delta / ppm)
        elif _is_multiple(delta, pps):
            # Seconds
            return _maybe_add_count("s", delta / pps)
        elif _is_multiple(delta, (pps // 1000)):
            # Milliseconds
            return _maybe_add_count("ms", delta / (pps // 1000))
        elif _is_multiple(delta, (pps // 1_000_000)):
            # Microseconds
            return _maybe_add_count("us", delta / (pps // 1_000_000))
        else:
            # Nanoseconds
            return _maybe_add_count("ns", delta)

    @cache_readonly
    def day_deltas(self) -> list[int]:
        ppd = periods_per_day(self._creso)
        return [x / ppd for x in self.deltas]

    @cache_readonly
    def hour_deltas(self) -> list[int]:
        pph = periods_per_day(self._creso) // 24
        return [x / pph for x in self.deltas]

    @cache_readonly
    def fields(self) -> np.ndarray:  # structured array of fields
        return build_field_sarray(self.i8values, reso=self._creso)

    @cache_readonly
    def rep_stamp(self) -> Timestamp:
        return Timestamp(self.i8values[0], unit=self.index.unit)

    def month_position_check(self) -> str | None:
        return month_position_check(self.fields, self.index.dayofweek)

    @cache_readonly
    def mdiffs(self) -> npt.NDArray[np.int64]:
        nmonths = self.fields["Y"] * 12 + self.fields["M"]
        return unique_deltas(nmonths.astype("i8"))

    @cache_readonly
    def ydiffs(self) -> npt.NDArray[np.int64]:
        return unique_deltas(self.fields["Y"].astype("i8"))

    def _infer_daily_rule(self) -> str | None:
        annual_rule = self._get_annual_rule()
        if annual_rule:
            nyears = self.ydiffs[0]
            month = MONTH_ALIASES[self.rep_stamp.month]
            alias = f"{annual_rule}-{month}"
            return _maybe_add_count(alias, nyears)

        quarterly_rule = self._get_quarterly_rule()
        if quarterly_rule:
            nquarters = self.mdiffs[0] / 3
            mod_dict = {0: 12, 2: 11, 1: 10}
            month = MONTH_ALIASES[mod_dict[self.rep_stamp.month % 3]]
            alias = f"{quarterly_rule}-{month}"
            return _maybe_add_count(alias, nquarters)

        monthly_rule = self._get_monthly_rule()
        if monthly_rule:
            return _maybe_add_count(monthly_rule, self.mdiffs[0])

        if self.is_unique:
            return self._get_daily_rule()

        if self._is_business_daily():
            return "B"

        wom_rule = self._get_wom_rule()
        if wom_rule:
            return wom_rule

        return None

    def _get_daily_rule(self) -> str | None:
        ppd = periods_per_day(self._creso)
        days = self.deltas[0] / ppd
        if days % 7 == 0:
            # Weekly
            wd = int_to_weekday[self.rep_stamp.weekday()]
            alias = f"W-{wd}"
            return _maybe_add_count(alias, days / 7)
        else:
            return _maybe_add_count("D", days)

    def _get_annual_rule(self) -> str | None:
        if len(self.ydiffs) > 1:
            return None

        if len(unique(self.fields["M"])) > 1:
            return None

        pos_check = self.month_position_check()

        if pos_check is None:
            return None
        else:
            return {"cs": "YS", "bs": "BYS", "ce": "YE", "be": "BYE"}.get(pos_check)

    def _get_quarterly_rule(self) -> str | None:
        if len(self.mdiffs) > 1:
            return None

        if not self.mdiffs[0] % 3 == 0:
            return None

        pos_check = self.month_position_check()

        if pos_check is None:
            return None
        else:
            return {"cs": "QS", "bs": "BQS", "ce": "QE", "be": "BQE"}.get(pos_check)

    def _get_monthly_rule(self) -> str | None:
        if len(self.mdiffs) > 1:
            return None
        pos_check = self.month_position_check()

        if pos_check is None:
            return None
        else:
            return {"cs": "MS", "bs": "BMS", "ce": "ME", "be": "BME"}.get(pos_check)

    def _is_business_daily(self) -> bool:
        # quick check: cannot be business daily
        if self.day_deltas != [1, 3]:
            return False

