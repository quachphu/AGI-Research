        else:
            freq = None

        if isinstance(scalars, cls):
            validate_dtype_freq(scalars.dtype, freq)
            if copy:
                scalars = scalars.copy()
            return scalars

        periods = np.asarray(scalars, dtype=object)

        freq = freq or libperiod.extract_freq(periods)
        ordinals = libperiod.extract_ordinals(periods, freq)
        dtype = PeriodDtype(freq)
        return cls(ordinals, dtype=dtype)

    @classmethod
    def _from_sequence_of_strings(
        cls, strings, *, dtype: Dtype | None = None, copy: bool = False
    ) -> Self:
        return cls._from_sequence(strings, dtype=dtype, copy=copy)

    @classmethod
    def _from_datetime64(cls, data, freq, tz=None) -> Self:
        """
        Construct a PeriodArray from a datetime64 array

        Parameters
        ----------
        data : ndarray[datetime64[ns], datetime64[ns, tz]]
        freq : str or Tick
        tz : tzinfo, optional

        Returns
        -------
        PeriodArray[freq]
        """
        if isinstance(freq, BaseOffset):
            freq = freq_to_period_freqstr(freq.n, freq.name)
        data, freq = dt64arr_to_periodarr(data, freq, tz)
        dtype = PeriodDtype(freq)
        return cls(data, dtype=dtype)

    @classmethod
    def _generate_range(cls, start, end, periods, freq):
        periods = dtl.validate_periods(periods)

        if freq is not None:
            freq = Period._maybe_convert_freq(freq)

        if start is not None or end is not None:
            subarr, freq = _get_ordinal_range(start, end, periods, freq)
        else:
            raise ValueError("Not enough parameters to construct Period range")

        return subarr, freq

    @classmethod
    def _from_fields(cls, *, fields: dict, freq) -> Self:
        subarr, freq = _range_from_fields(freq=freq, **fields)
        dtype = PeriodDtype(freq)
        return cls._simple_new(subarr, dtype=dtype)

    # -----------------------------------------------------------------
    # DatetimeLike Interface

    # error: Argument 1 of "_unbox_scalar" is incompatible with supertype
    # "DatetimeLikeArrayMixin"; supertype defines the argument type as
    # "Union[Union[Period, Any, Timedelta], NaTType]"
    def _unbox_scalar(  # type: ignore[override]
        self,
        value: Period | NaTType,
    ) -> np.int64:
        if value is NaT:
            # error: Item "Period" of "Union[Period, NaTType]" has no attribute "value"
            return np.int64(value._value)  # type: ignore[union-attr]
        elif isinstance(value, self._scalar_type):
            self._check_compatible_with(value)
            return np.int64(value.ordinal)
        else:
            raise ValueError(f"'value' should be a Period. Got '{value}' instead.")

    def _scalar_from_string(self, value: str) -> Period:
        return Period(value, freq=self.freq)

    # error: Argument 1 of "_check_compatible_with" is incompatible with
    # supertype "DatetimeLikeArrayMixin"; supertype defines the argument type
    # as "Period | Timestamp | Timedelta | NaTType"
    def _check_compatible_with(self, other: Period | NaTType | PeriodArray) -> None:  # type: ignore[override]
        if other is NaT:
            return
        # error: Item "NaTType" of "Period | NaTType | PeriodArray" has no
        # attribute "freq"
        self._require_matching_freq(other.freq)  # type: ignore[union-attr]

    # --------------------------------------------------------------------
    # Data / Attributes

    @cache_readonly
    def dtype(self) -> PeriodDtype:
        return self._dtype

    # error: Cannot override writeable attribute with read-only property
    @property  # type: ignore[override]
    def freq(self) -> BaseOffset:
        """
        Return the frequency object for this PeriodArray.
        """
        return self.dtype.freq

    @property
    def freqstr(self) -> str:
        return freq_to_period_freqstr(self.freq.n, self.freq.name)

    def __array__(
        self, dtype: NpDtype | None = None, copy: bool | None = None
    ) -> np.ndarray:
        if dtype == "i8":
            # For NumPy 1.x compatibility we cannot use copy=None.  And
            # `copy=False` has the meaning of `copy=None` here:
            if not copy:
                return np.asarray(self.asi8, dtype=dtype)
            else:
                return np.array(self.asi8, dtype=dtype)

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

        if dtype == bool:
            return ~self._isnan

        # This will raise TypeError for non-object dtypes
        return np.array(list(self), dtype=object)

    def __arrow_array__(self, type=None):
        """
        Convert myself into a pyarrow Array.
        """
        import pyarrow

        from pandas.core.arrays.arrow.extension_types import ArrowPeriodType

        if type is not None:
            if pyarrow.types.is_integer(type):
                return pyarrow.array(self._ndarray, mask=self.isna(), type=type)
            elif isinstance(type, ArrowPeriodType):
                # ensure we have the same freq
                if self.freqstr != type.freq:
                    raise TypeError(
                        "Not supported to convert PeriodArray to array with different "
                        f"'freq' ({self.freqstr} vs {type.freq})"
                    )
            else:
                raise TypeError(
                    f"Not supported to convert PeriodArray to '{type}' type"
                )

        period_type = ArrowPeriodType(self.freqstr)
        storage_array = pyarrow.array(self._ndarray, mask=self.isna(), type="int64")
        return pyarrow.ExtensionArray.from_storage(period_type, storage_array)

    # --------------------------------------------------------------------
    # Vectorized analogues of Period properties

    year = _field_accessor(
        "year",
        """
        The year of the period.

        Examples
        --------
        >>> idx = pd.PeriodIndex(["2023", "2024", "2025"], freq="Y")
        >>> idx.year
        Index([2023, 2024, 2025], dtype='int64')
        """,
    )
    month = _field_accessor(
        "month",
        """
        The month as January=1, December=12.

        Examples
        --------
        >>> idx = pd.PeriodIndex(["2023-01", "2023-02", "2023-03"], freq="M")
        >>> idx.month
        Index([1, 2, 3], dtype='int64')
        """,
    )
    day = _field_accessor(
        "day",
        """
        The days of the period.

        Examples
        --------
        >>> idx = pd.PeriodIndex(['2020-01-31', '2020-02-28'], freq='D')
        >>> idx.day
        Index([31, 28], dtype='int64')
        """,
    )
    hour = _field_accessor(
        "hour",
        """
        The hour of the period.

        Examples
        --------
        >>> idx = pd.PeriodIndex(["2023-01-01 10:00", "2023-01-01 11:00"], freq='h')
        >>> idx.hour
        Index([10, 11], dtype='int64')
        """,
    )
    minute = _field_accessor(
        "minute",
        """
        The minute of the period.

        Examples
        --------
        >>> idx = pd.PeriodIndex(["2023-01-01 10:30:00",
        ...                       "2023-01-01 11:50:00"], freq='min')
        >>> idx.minute
        Index([30, 50], dtype='int64')
        """,
    )
    second = _field_accessor(
        "second",
        """
        The second of the period.

        Examples
        --------
        >>> idx = pd.PeriodIndex(["2023-01-01 10:00:30",
        ...                       "2023-01-01 10:00:31"], freq='s')
        >>> idx.second
        Index([30, 31], dtype='int64')
        """,
    )
    weekofyear = _field_accessor(
        "week",
        """
        The week ordinal of the year.

        Examples
        --------
        >>> idx = pd.PeriodIndex(["2023-01", "2023-02", "2023-03"], freq="M")
        >>> idx.week  # It can be written `weekofyear`
        Index([5, 9, 13], dtype='int64')
        """,
    )
    week = weekofyear
    day_of_week = _field_accessor(
        "day_of_week",
        """
        The day of the week with Monday=0, Sunday=6.

        Examples
        --------
        >>> idx = pd.PeriodIndex(["2023-01-01", "2023-01-02", "2023-01-03"], freq="D")
        >>> idx.weekday
        Index([6, 0, 1], dtype='int64')
        """,
    )
    dayofweek = day_of_week
    weekday = dayofweek
    dayofyear = day_of_year = _field_accessor(
        "day_of_year",
        """
        The ordinal day of the year.

        Examples
        --------
        >>> idx = pd.PeriodIndex(["2023-01-10", "2023-02-01", "2023-03-01"], freq="D")
        >>> idx.dayofyear
        Index([10, 32, 60], dtype='int64')

        >>> idx = pd.PeriodIndex(["2023", "2024", "2025"], freq="Y")
        >>> idx
        PeriodIndex(['2023', '2024', '2025'], dtype='period[Y-DEC]')
        >>> idx.dayofyear
        Index([365, 366, 365], dtype='int64')
        """,
    )
    quarter = _field_accessor(
        "quarter",
        """
        The quarter of the date.

        Examples
        --------
        >>> idx = pd.PeriodIndex(["2023-01", "2023-02", "2023-03"], freq="M")
        >>> idx.quarter
        Index([1, 1, 1], dtype='int64')
        """,
    )
    qyear = _field_accessor("qyear")
    days_in_month = _field_accessor(
        "days_in_month",
        """
        The number of days in the month.

        Examples
        --------
        For Series:

        >>> period = pd.period_range('2020-1-1 00:00', '2020-3-1 00:00', freq='M')
        >>> s = pd.Series(period)
        >>> s
        0   2020-01
        1   2020-02
        2   2020-03
        dtype: period[M]
        >>> s.dt.days_in_month
        0    31
        1    29
        2    31
        dtype: int64

        For PeriodIndex:

        >>> idx = pd.PeriodIndex(["2023-01", "2023-02", "2023-03"], freq="M")
        >>> idx.days_in_month   # It can be also entered as `daysinmonth`
        Index([31, 28, 31], dtype='int64')
        """,
    )
    daysinmonth = days_in_month

    @property
    def is_leap_year(self) -> npt.NDArray[np.bool_]:
        """
        Logical indicating if the date belongs to a leap year.

        Examples
        --------
        >>> idx = pd.PeriodIndex(["2023", "2024", "2025"], freq="Y")
        >>> idx.is_leap_year
        array([False,  True, False])
        """
        return isleapyear_arr(np.asarray(self.year))

    def to_timestamp(self, freq=None, how: str = "start") -> DatetimeArray:
        """
        Cast to DatetimeArray/Index.

        Parameters
        ----------
        freq : str or DateOffset, optional
            Target frequency. The default is 'D' for week or longer,
            's' otherwise.
        how : {'s', 'e', 'start', 'end'}
            Whether to use the start or end of the time period being converted.

        Returns
        -------
        DatetimeArray/Index

        Examples
        --------
        >>> idx = pd.PeriodIndex(["2023-01", "2023-02", "2023-03"], freq="M")
        >>> idx.to_timestamp()
        DatetimeIndex(['2023-01-01', '2023-02-01', '2023-03-01'],
        dtype='datetime64[ns]', freq='MS')
        """
        from pandas.core.arrays import DatetimeArray

        how = libperiod.validate_end_alias(how)

        end = how == "E"
        if end:
            if freq == "B" or self.freq == "B":
                # roll forward to ensure we land on B date
                adjust = Timedelta(1, "D") - Timedelta(1, "ns")
                return self.to_timestamp(how="start") + adjust
            else:
                adjust = Timedelta(1, "ns")
                return (self + self.freq).to_timestamp(how="start") - adjust

        if freq is None:
            freq_code = self._dtype._get_to_timestamp_base()
            dtype = PeriodDtypeBase(freq_code, 1)
            freq = dtype._freqstr
            base = freq_code
        else:
            freq = Period._maybe_convert_freq(freq)
            base = freq._period_dtype_code

        new_parr = self.asfreq(freq, how=how)

        new_data = libperiod.periodarr_to_dt64arr(new_parr.asi8, base)
        dta = DatetimeArray._from_sequence(new_data)

        if self.freq.name == "B":
            # See if we can retain BDay instead of Day in cases where
            #  len(self) is too small for infer_freq to distinguish between them
            diffs = libalgos.unique_deltas(self.asi8)
            if len(diffs) == 1:
                diff = diffs[0]
                if diff == self.dtype._n:
                    dta._freq = self.freq
                elif diff == 1:
                    dta._freq = self.freq.base
                # TODO: other cases?
            return dta
        else:
            return dta._with_freq("infer")

    # --------------------------------------------------------------------

    def _box_func(self, x) -> Period | NaTType:
        return Period._from_ordinal(ordinal=x, freq=self.freq)

    @doc(**_shared_doc_kwargs, other="PeriodIndex", other_name="PeriodIndex")
    def asfreq(self, freq=None, how: str = "E") -> Self:
        """
        Convert the {klass} to the specified frequency `freq`.

        Equivalent to applying :meth:`pandas.Period.asfreq` with the given arguments
        to each :class:`~pandas.Period` in this {klass}.

        Parameters
        ----------
        freq : str
            A frequency.
        how : str {{'E', 'S'}}, default 'E'
            Whether the elements should be aligned to the end
            or start within pa period.

            * 'E', 'END', or 'FINISH' for end,
            * 'S', 'START', or 'BEGIN' for start.

            January 31st ('END') vs. January 1st ('START') for example.

        Returns
        -------
        {klass}
            The transformed {klass} with the new frequency.

        See Also
        --------
        {other}.asfreq: Convert each Period in a {other_name} to the given frequency.
        Period.asfreq : Convert a :class:`~pandas.Period` object to the given frequency.

        Examples
        --------
        >>> pidx = pd.period_range('2010-01-01', '2015-01-01', freq='Y')
        >>> pidx
        PeriodIndex(['2010', '2011', '2012', '2013', '2014', '2015'],
        dtype='period[Y-DEC]')

        >>> pidx.asfreq('M')
        PeriodIndex(['2010-12', '2011-12', '2012-12', '2013-12', '2014-12',
        '2015-12'], dtype='period[M]')

        >>> pidx.asfreq('M', how='S')
        PeriodIndex(['2010-01', '2011-01', '2012-01', '2013-01', '2014-01',
        '2015-01'], dtype='period[M]')
        """
        how = libperiod.validate_end_alias(how)
        if isinstance(freq, BaseOffset) and hasattr(freq, "_period_dtype_code"):
            freq = PeriodDtype(freq)._freqstr
        freq = Period._maybe_convert_freq(freq)

        base1 = self._dtype._dtype_code
        base2 = freq._period_dtype_code

        asi8 = self.asi8
        # self.freq.n can't be negative or 0
        end = how == "E"
        if end:
            ordinal = asi8 + self.dtype._n - 1
        else:
            ordinal = asi8

        new_data = period_asfreq_arr(ordinal, base1, base2, end)

        if self._hasna:
            new_data[self._isnan] = iNaT

        dtype = PeriodDtype(freq)
        return type(self)(new_data, dtype=dtype)

    # ------------------------------------------------------------------
    # Rendering Methods

    def _formatter(self, boxed: bool = False):
        if boxed:
            return str
        return "'{}'".format

    def _format_native_types(
        self, *, na_rep: str | float = "NaT", date_format=None, **kwargs
    ) -> npt.NDArray[np.object_]:
        """
        actually format my specific types
        """
        return libperiod.period_array_strftime(
            self.asi8, self.dtype._dtype_code, na_rep, date_format
        )

    # ------------------------------------------------------------------

    def astype(self, dtype, copy: bool = True):
        # We handle Period[T] -> Period[U]
        # Our parent handles everything else.
        dtype = pandas_dtype(dtype)
        if dtype == self._dtype:
            if not copy:
                return self
            else:
                return self.copy()
        if isinstance(dtype, PeriodDtype):
            return self.asfreq(dtype.freq)

        if lib.is_np_dtype(dtype, "M") or isinstance(dtype, DatetimeTZDtype):
            # GH#45038 match PeriodIndex behavior.
            tz = getattr(dtype, "tz", None)
            unit = dtl.dtype_to_unit(dtype)
            return self.to_timestamp().tz_localize(tz).as_unit(unit)

        return super().astype(dtype, copy=copy)

    def searchsorted(
        self,
        value: NumpyValueArrayLike | ExtensionArray,
        side: Literal["left", "right"] = "left",
        sorter: NumpySorter | None = None,
    ) -> npt.NDArray[np.intp] | np.intp:
        npvalue = self._validate_setitem_value(value).view("M8[ns]")

        # Cast to M8 to get datetime-like NaT placement,
        #  similar to dtl._period_dispatch
        m8arr = self._ndarray.view("M8[ns]")
        return m8arr.searchsorted(npvalue, side=side, sorter=sorter)

    def _pad_or_backfill(
        self,
        *,
        method: FillnaOptions,
        limit: int | None = None,
        limit_area: Literal["inside", "outside"] | None = None,
        copy: bool = True,
    ) -> Self:
        # view as dt64 so we get treated as timelike in core.missing,
        #  similar to dtl._period_dispatch
        dta = self.view("M8[ns]")
        result = dta._pad_or_backfill(
            method=method, limit=limit, limit_area=limit_area, copy=copy
        )
        if copy:
            return cast("Self", result.view(self.dtype))
        else:
            return self

    def fillna(
        self, value=None, method=None, limit: int | None = None, copy: bool = True
    ) -> Self:
        if method is not None:
            # view as dt64 so we get treated as timelike in core.missing,
            #  similar to dtl._period_dispatch
            dta = self.view("M8[ns]")
            result = dta.fillna(value=value, method=method, limit=limit, copy=copy)
            # error: Incompatible return value type (got "Union[ExtensionArray,
            # ndarray[Any, Any]]", expected "PeriodArray")
            return result.view(self.dtype)  # type: ignore[return-value]
        return super().fillna(value=value, method=method, limit=limit, copy=copy)

    # ------------------------------------------------------------------
    # Arithmetic Methods

    def _addsub_int_array_or_scalar(
        self, other: np.ndarray | int, op: Callable[[Any, Any], Any]
    ) -> Self:
        """
        Add or subtract array of integers.

        Parameters
        ----------
        other : np.ndarray[int64] or int
        op : {operator.add, operator.sub}

        Returns
        -------
        result : PeriodArray
        """
        assert op in [operator.add, operator.sub]
        if op is operator.sub:
            other = -other
        res_values = add_overflowsafe(self.asi8, np.asarray(other, dtype="i8"))
        return type(self)(res_values, dtype=self.dtype)

    def _add_offset(self, other: BaseOffset):
        assert not isinstance(other, Tick)

        self._require_matching_freq(other, base=True)
        return self._addsub_int_array_or_scalar(other.n, operator.add)

    # TODO: can we de-duplicate with Period._add_timedeltalike_scalar?
    def _add_timedeltalike_scalar(self, other):
        """
        Parameters
        ----------
        other : timedelta, Tick, np.timedelta64

        Returns
        -------
        PeriodArray
        """
        if not isinstance(self.freq, Tick):
            # We cannot add timedelta-like to non-tick PeriodArray
            raise raise_on_incompatible(self, other)

        if isna(other):
            # i.e. np.timedelta64("NaT")
            return super()._add_timedeltalike_scalar(other)

        td = np.asarray(Timedelta(other).asm8)
        return self._add_timedelta_arraylike(td)

    def _add_timedelta_arraylike(
        self, other: TimedeltaArray | npt.NDArray[np.timedelta64]
    ) -> Self:
        """
        Parameters
        ----------
        other : TimedeltaArray or ndarray[timedelta64]

        Returns
        -------
        PeriodArray
        """
        if not self.dtype._is_tick_like():
            # We cannot add timedelta-like to non-tick PeriodArray
            raise TypeError(
                f"Cannot add or subtract timedelta64[ns] dtype from {self.dtype}"
            )

        dtype = np.dtype(f"m8[{self.dtype._td64_unit}]")

        # Similar to _check_timedeltalike_freq_compat, but we raise with a
        #  more specific exception message if necessary.
        try:
            delta = astype_overflowsafe(
                np.asarray(other), dtype=dtype, copy=False, round_ok=False
            )
        except ValueError as err:
            # e.g. if we have minutes freq and try to add 30s
            # "Cannot losslessly convert units"
            raise IncompatibleFrequency(
                "Cannot add/subtract timedelta-like from PeriodArray that is "
                "not an integer multiple of the PeriodArray's freq."
            ) from err

        res_values = add_overflowsafe(self.asi8, np.asarray(delta.view("i8")))
        return type(self)(res_values, dtype=self.dtype)

    def _check_timedeltalike_freq_compat(self, other):
        """
        Arithmetic operations with timedelta-like scalars or array `other`
        are only valid if `other` is an integer multiple of `self.freq`.
        If the operation is valid, find that integer multiple.  Otherwise,
        raise because the operation is invalid.

        Parameters
        ----------
        other : timedelta, np.timedelta64, Tick,
                ndarray[timedelta64], TimedeltaArray, TimedeltaIndex

        Returns
        -------
        multiple : int or ndarray[int64]

        Raises
        ------
        IncompatibleFrequency
        """
        assert self.dtype._is_tick_like()  # checked by calling function

        dtype = np.dtype(f"m8[{self.dtype._td64_unit}]")

        if isinstance(other, (timedelta, np.timedelta64, Tick)):
            td = np.asarray(Timedelta(other).asm8)
        else:
            td = np.asarray(other)

        try:
            delta = astype_overflowsafe(td, dtype=dtype, copy=False, round_ok=False)
        except ValueError as err:
            raise raise_on_incompatible(self, other) from err

        delta = delta.view("i8")
        return lib.item_from_zerodim(delta)


def raise_on_incompatible(left, right) -> IncompatibleFrequency:
    """
    Helper function to render a consistent error message when raising
    IncompatibleFrequency.

    Parameters
    ----------
    left : PeriodArray
    right : None, DateOffset, Period, ndarray, or timedelta-like

    Returns
    -------
    IncompatibleFrequency
        Exception to be raised by the caller.
    """
    # GH#24283 error message format depends on whether right is scalar
    if isinstance(right, (np.ndarray, ABCTimedeltaArray)) or right is None:
        other_freq = None
    elif isinstance(right, BaseOffset):
        other_freq = freq_to_period_freqstr(right.n, right.name)
    elif isinstance(right, (ABCPeriodIndex, PeriodArray, Period)):
        other_freq = right.freqstr
    else:
        other_freq = delta_to_tick(Timedelta(right)).freqstr

    own_freq = freq_to_period_freqstr(left.freq.n, left.freq.name)
    msg = DIFFERENT_FREQ.format(
        cls=type(left).__name__, own_freq=own_freq, other_freq=other_freq
    )
    return IncompatibleFrequency(msg)


# -------------------------------------------------------------------
# Constructor Helpers


def period_array(
    data: Sequence[Period | str | None] | AnyArrayLike,
    freq: str | Tick | BaseOffset | None = None,
    copy: bool = False,
) -> PeriodArray:
    """
    Construct a new PeriodArray from a sequence of Period scalars.

    Parameters
    ----------
    data : Sequence of Period objects
        A sequence of Period objects. These are required to all have
        the same ``freq.`` Missing values can be indicated by ``None``
        or ``pandas.NaT``.
    freq : str, Tick, or Offset
        The frequency of every element of the array. This can be specified
        to avoid inferring the `freq` from `data`.
    copy : bool, default False
        Whether to ensure a copy of the data is made.

    Returns
    -------
    PeriodArray

    See Also
    --------
    PeriodArray
    pandas.PeriodIndex

    Examples
    --------
    >>> period_array([pd.Period('2017', freq='Y'),
    ...               pd.Period('2018', freq='Y')])
    <PeriodArray>
    ['2017', '2018']
    Length: 2, dtype: period[Y-DEC]

    >>> period_array([pd.Period('2017', freq='Y'),
    ...               pd.Period('2018', freq='Y'),
    ...               pd.NaT])
    <PeriodArray>
    ['2017', '2018', 'NaT']
    Length: 3, dtype: period[Y-DEC]

    Integers that look like years are handled

    >>> period_array([2000, 2001, 2002], freq='D')
    <PeriodArray>
    ['2000-01-01', '2001-01-01', '2002-01-01']
    Length: 3, dtype: period[D]

    Datetime-like strings may also be passed

    >>> period_array(['2000-Q1', '2000-Q2', '2000-Q3', '2000-Q4'], freq='Q')
    <PeriodArray>
    ['2000Q1', '2000Q2', '2000Q3', '2000Q4']
    Length: 4, dtype: period[Q-DEC]
    """
    data_dtype = getattr(data, "dtype", None)

    if lib.is_np_dtype(data_dtype, "M"):
        return PeriodArray._from_datetime64(data, freq)
    if isinstance(data_dtype, PeriodDtype):
        out = PeriodArray(data)
        if freq is not None:
            if freq == data_dtype.freq:
                return out
            return out.asfreq(freq)
        return out

    # other iterable of some kind
    if not isinstance(data, (np.ndarray, list, tuple, ABCSeries)):
        data = list(data)

    arrdata = np.asarray(data)

    dtype: PeriodDtype | None
    if freq:
        dtype = PeriodDtype(freq)
    else:
        dtype = None

    if arrdata.dtype.kind == "f" and len(arrdata) > 0:
        raise TypeError("PeriodIndex does not allow floating point in construction")

    if arrdata.dtype.kind in "iu":
        arr = arrdata.astype(np.int64, copy=False)
        # error: Argument 2 to "from_ordinals" has incompatible type "Union[str,
        # Tick, None]"; expected "Union[timedelta, BaseOffset, str]"
        ordinals = libperiod.from_ordinals(arr, freq)  # type: ignore[arg-type]
        return PeriodArray(ordinals, dtype=dtype)

    data = ensure_object(arrdata)
    if freq is None:
        freq = libperiod.extract_freq(data)
    dtype = PeriodDtype(freq)
    return PeriodArray._from_sequence(data, dtype=dtype)


@overload
def validate_dtype_freq(dtype, freq: BaseOffsetT) -> BaseOffsetT:
    ...


@overload
def validate_dtype_freq(dtype, freq: timedelta | str | None) -> BaseOffset:
    ...


def validate_dtype_freq(
    dtype, freq: BaseOffsetT | BaseOffset | timedelta | str | None
) -> BaseOffsetT:
    """
    If both a dtype and a freq are available, ensure they match.  If only
    dtype is available, extract the implied freq.

    Parameters
    ----------
    dtype : dtype
    freq : DateOffset or None

    Returns
    -------
    freq : DateOffset

    Raises
    ------
    ValueError : non-period dtype
    IncompatibleFrequency : mismatch between dtype and freq
    """
    if freq is not None:
        freq = to_offset(freq, is_period=True)

    if dtype is not None:
        dtype = pandas_dtype(dtype)
        if not isinstance(dtype, PeriodDtype):
            raise ValueError("dtype must be PeriodDtype")
        if freq is None:
            freq = dtype.freq
        elif freq != dtype.freq:
            raise IncompatibleFrequency("specified freq and dtype are different")
    # error: Incompatible return value type (got "Union[BaseOffset, Any, None]",
    # expected "BaseOffset")
    return freq  # type: ignore[return-value]


def dt64arr_to_periodarr(
    data, freq, tz=None
) -> tuple[npt.NDArray[np.int64], BaseOffset]:
    """
    Convert an datetime-like array to values Period ordinals.

    Parameters
    ----------
    data : Union[Series[datetime64[ns]], DatetimeIndex, ndarray[datetime64ns]]
    freq : Optional[Union[str, Tick]]
        Must match the `freq` on the `data` if `data` is a DatetimeIndex
        or Series.
    tz : Optional[tzinfo]

    Returns
    -------
    ordinals : ndarray[int64]
    freq : Tick
        The frequency extracted from the Series or DatetimeIndex if that's
        used.

    """
    if not isinstance(data.dtype, np.dtype) or data.dtype.kind != "M":
        raise ValueError(f"Wrong dtype: {data.dtype}")

    if freq is None:
        if isinstance(data, ABCIndex):
            data, freq = data._values, data.freq
        elif isinstance(data, ABCSeries):
            data, freq = data._values, data.dt.freq

    elif isinstance(data, (ABCIndex, ABCSeries)):
        data = data._values

    reso = get_unit_from_dtype(data.dtype)
    freq = Period._maybe_convert_freq(freq)
    base = freq._period_dtype_code
    return c_dt64arr_to_periodarr(data.view("i8"), base, tz, reso=reso), freq


def _get_ordinal_range(start, end, periods, freq, mult: int = 1):
    if com.count_not_none(start, end, periods) != 2:
        raise ValueError(
            "Of the three parameters: start, end, and periods, "
            "exactly two must be specified"
        )

    if freq is not None:
        freq = to_offset(freq, is_period=True)
        mult = freq.n

    if start is not None:
        start = Period(start, freq)
    if end is not None:
        end = Period(end, freq)

    is_start_per = isinstance(start, Period)
    is_end_per = isinstance(end, Period)

    if is_start_per and is_end_per and start.freq != end.freq:
        raise ValueError("start and end must have same freq")
    if start is NaT or end is NaT:
        raise ValueError("start and end must not be NaT")

    if freq is None:
        if is_start_per:
            freq = start.freq
        elif is_end_per:
            freq = end.freq
        else:  # pragma: no cover
            raise ValueError("Could not infer freq from start/end")
        mult = freq.n

    if periods is not None:
        periods = periods * mult
        if start is None:
            data = np.arange(
                end.ordinal - periods + mult, end.ordinal + 1, mult, dtype=np.int64
            )
        else:
            data = np.arange(
                start.ordinal, start.ordinal + periods, mult, dtype=np.int64
            )
    else:
        data = np.arange(start.ordinal, end.ordinal + 1, mult, dtype=np.int64)

    return data, freq


def _range_from_fields(
    year=None,
    month=None,
    quarter=None,
    day=None,
    hour=None,
    minute=None,
    second=None,
    freq=None,
) -> tuple[np.ndarray, BaseOffset]:
    if hour is None:
        hour = 0
    if minute is None:
        minute = 0
    if second is None:
        second = 0
    if day is None:
        day = 1

    ordinals = []

    if quarter is not None:
        if freq is None:
            freq = to_offset("Q", is_period=True)
            base = FreqGroup.FR_QTR.value
        else:
            freq = to_offset(freq, is_period=True)
            base = libperiod.freq_to_dtype_code(freq)
            if base != FreqGroup.FR_QTR.value:
                raise AssertionError("base must equal FR_QTR")

        freqstr = freq.freqstr
        year, quarter = _make_field_arrays(year, quarter)
        for y, q in zip(year, quarter):
            calendar_year, calendar_month = parsing.quarter_to_myear(y, q, freqstr)
            val = libperiod.period_ordinal(
                calendar_year, calendar_month, 1, 1, 1, 1, 0, 0, base
            )
            ordinals.append(val)
    else:
        freq = to_offset(freq, is_period=True)
        base = libperiod.freq_to_dtype_code(freq)
        arrays = _make_field_arrays(year, month, day, hour, minute, second)
        for y, mth, d, h, mn, s in zip(*arrays):
            ordinals.append(libperiod.period_ordinal(y, mth, d, h, mn, s, 0, 0, base))

    return np.array(ordinals, dtype=np.int64), freq


def _make_field_arrays(*fields) -> list[np.ndarray]:
    length = None
    for x in fields:
        if isinstance(x, (list, np.ndarray, ABCSeries)):
            if length is not None and len(x) != length:
                raise ValueError("Mismatched Period array lengths")
            if length is None:
                length = len(x)

    # error: Argument 2 to "repeat" has incompatible type "Optional[int]"; expected
    # "Union[Union[int, integer[Any]], Union[bool, bool_], ndarray, Sequence[Union[int,
    # integer[Any]]], Sequence[Union[bool, bool_]], Sequence[Sequence[Any]]]"
    return [
        np.asarray(x)
        if isinstance(x, (np.ndarray, list, ABCSeries))
        else np.repeat(x, length)  # type: ignore[arg-type]
        for x in fields
    ]
<|endoftext|>from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
)

import numpy as np

from pandas._libs import lib
from pandas._libs.tslibs import is_supported_dtype
from pandas.compat.numpy import function as nv

from pandas.core.dtypes.astype import astype_array
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
from pandas.core.dtypes.common import pandas_dtype
from pandas.core.dtypes.dtypes import NumpyEADtype
from pandas.core.dtypes.missing import isna

from pandas.core import (
    arraylike,
    missing,
    nanops,
    ops,
)
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from pandas.core.construction import ensure_wrapped_if_datetimelike
from pandas.core.strings.object_array import ObjectStringArrayMixin

if TYPE_CHECKING:
    from collections.abc import Callable

    from pandas._typing import (
        AxisInt,
        Dtype,
        FillnaOptions,
        InterpolateOptions,
        NpDtype,
        Scalar,
        Self,
        npt,
    )

    from pandas import Index


# error: Definition of "_concat_same_type" in base class "NDArrayBacked" is
# incompatible with definition in base class "ExtensionArray"
class NumpyExtensionArray(  # type: ignore[misc]
    OpsMixin,
    NDArrayBackedExtensionArray,
    ObjectStringArrayMixin,
):
    """
    A pandas ExtensionArray for NumPy data.

    This is mostly for internal compatibility, and is not especially
    useful on its own.

    Parameters
    ----------
    values : ndarray
        The NumPy ndarray to wrap. Must be 1-dimensional.
    copy : bool, default False
        Whether to copy `values`.

    Attributes
    ----------
    None

    Methods
    -------
    None

    Examples
    --------
    >>> pd.arrays.NumpyExtensionArray(np.array([0, 1, 2, 3]))
    <NumpyExtensionArray>
    [0, 1, 2, 3]
    Length: 4, dtype: int64
    """

    # If you're wondering why pd.Series(cls) doesn't put the array in an
    # ExtensionBlock, search for `ABCNumpyExtensionArray`. We check for
    # that _typ to ensure that users don't unnecessarily use EAs inside
    # pandas internals, which turns off things like block consolidation.
    _typ = "npy_extension"
    __array_priority__ = 1000
    _ndarray: np.ndarray
    _dtype: NumpyEADtype
    _internal_fill_value = np.nan

    # ------------------------------------------------------------------------
    # Constructors

    def __init__(
        self, values: np.ndarray | NumpyExtensionArray, copy: bool = False
    ) -> None:
        if isinstance(values, type(self)):
            values = values._ndarray
        if not isinstance(values, np.ndarray):
            raise ValueError(
                f"'values' must be a NumPy array, not {type(values).__name__}"
            )

        if values.ndim == 0:
            # Technically we support 2, but do not advertise that fact.
            raise ValueError("NumpyExtensionArray must be 1-dimensional.")

        if copy:
            values = values.copy()

        dtype = NumpyEADtype(values.dtype)
        super().__init__(values, dtype)

    @classmethod
    def _from_sequence(
        cls, scalars, *, dtype: Dtype | None = None, copy: bool = False
    ) -> NumpyExtensionArray:
        if isinstance(dtype, NumpyEADtype):
            dtype = dtype._dtype

        # error: Argument "dtype" to "asarray" has incompatible type
        # "Union[ExtensionDtype, str, dtype[Any], dtype[floating[_64Bit]], Type[object],
        # None]"; expected "Union[dtype[Any], None, type, _SupportsDType, str,
        # Union[Tuple[Any, int], Tuple[Any, Union[int, Sequence[int]]], List[Any],
        # _DTypeDict, Tuple[Any, Any]]]"
        result = np.asarray(scalars, dtype=dtype)  # type: ignore[arg-type]
        if (
            result.ndim > 1
            and not hasattr(scalars, "dtype")
            and (dtype is None or dtype == object)
        ):
            # e.g. list-of-tuples
            result = construct_1d_object_array_from_listlike(scalars)

        if copy and result is scalars:
            result = result.copy()
        return cls(result)

    # ------------------------------------------------------------------------
    # Data

    @property
    def dtype(self) -> NumpyEADtype:
        return self._dtype

    # ------------------------------------------------------------------------
    # NumPy Array Interface

    def __array__(
        self, dtype: NpDtype | None = None, copy: bool | None = None
    ) -> np.ndarray:
        if copy is not None:
            # Note: branch avoids `copy=None` for NumPy 1.x support
            return np.array(self._ndarray, dtype=dtype, copy=copy)
        return np.asarray(self._ndarray, dtype=dtype)

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
        # Lightly modified version of
        # https://numpy.org/doc/stable/reference/generated/numpy.lib.mixins.NDArrayOperatorsMixin.html
        # The primary modification is not boxing scalar return values
        # in NumpyExtensionArray, since pandas' ExtensionArrays are 1-d.
        out = kwargs.get("out", ())

        result = arraylike.maybe_dispatch_ufunc_to_dunder_op(
            self, ufunc, method, *inputs, **kwargs
        )
        if result is not NotImplemented:
            return result

        if "out" in kwargs:
            # e.g. test_ufunc_unary
            return arraylike.dispatch_ufunc_with_out(
                self, ufunc, method, *inputs, **kwargs
            )

        if method == "reduce":
            result = arraylike.dispatch_reduction_ufunc(
                self, ufunc, method, *inputs, **kwargs
            )
            if result is not NotImplemented:
                # e.g. tests.series.test_ufunc.TestNumpyReductions
                return result

        # Defer to the implementation of the ufunc on unwrapped values.
        inputs = tuple(
            x._ndarray if isinstance(x, NumpyExtensionArray) else x for x in inputs
        )
        if out:
            kwargs["out"] = tuple(
                x._ndarray if isinstance(x, NumpyExtensionArray) else x for x in out
            )
        result = getattr(ufunc, method)(*inputs, **kwargs)

        if ufunc.nout > 1:
            # multiple return values; re-box array-like results
            return tuple(type(self)(x) for x in result)
        elif method == "at":
            # no return value
            return None
        elif method == "reduce":
            if isinstance(result, np.ndarray):
                # e.g. test_np_reduce_2d
                return type(self)(result)

            # e.g. test_np_max_nested_tuples
            return result
        else:
            if self.dtype.type is str:  # type: ignore[comparison-overlap]
                # StringDtype
                try:
                    return type(self)(result)
                except ValueError:
                    # if validation of input fails (no strings)
                    # -> fallback to returning raw numpy array
                    return result
            # one return value; re-box array-like results
            return type(self)(result)

    # ------------------------------------------------------------------------
    # Pandas ExtensionArray Interface

    def astype(self, dtype, copy: bool = True):
        dtype = pandas_dtype(dtype)

        if dtype == self.dtype:
            if copy:
                return self.copy()
            return self

        result = astype_array(self._ndarray, dtype=dtype, copy=copy)
        return result

    def isna(self) -> np.ndarray:
        return isna(self._ndarray)

    def _validate_scalar(self, fill_value):
        if fill_value is None:
            # Primarily for subclasses
            fill_value = self.dtype.na_value
        return fill_value

    def _values_for_factorize(self) -> tuple[np.ndarray, float | None]:
        if self.dtype.kind in "iub":
            fv = None
        else:
            fv = np.nan
        return self._ndarray, fv

    # Base EA class (and all other EA classes) don't have limit_area keyword
    # This can be removed here as well when the interpolate ffill/bfill method
    # deprecation is enforced
    def _pad_or_backfill(
        self,
        *,
        method: FillnaOptions,
        limit: int | None = None,
        limit_area: Literal["inside", "outside"] | None = None,
        copy: bool = True,
    ) -> Self:
        """
        ffill or bfill along axis=0.
        """
        if copy:
            out_data = self._ndarray.copy()
        else:
            out_data = self._ndarray

        meth = missing.clean_fill_method(method)
        missing.pad_or_backfill_inplace(
            out_data.T,
            method=meth,
            axis=0,
            limit=limit,
            limit_area=limit_area,
        )

        if not copy:
            return self
        return type(self)._simple_new(out_data, dtype=self.dtype)

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
        See NDFrame.interpolate.__doc__.
        """
        # NB: we return type(self) even if copy=False
        if not self.dtype._is_numeric:
            raise TypeError(f"Cannot interpolate with {self.dtype} dtype")

        if not copy:
            out_data = self._ndarray
        else:
            out_data = self._ndarray.copy()

        # TODO: assert we have floating dtype?
        missing.interpolate_2d_inplace(
            out_data,
            method=method,
            axis=axis,
            index=index,
            limit=limit,
            limit_direction=limit_direction,
            limit_area=limit_area,
            **kwargs,
        )
        if not copy:
            return self
        return type(self)._simple_new(out_data, dtype=self.dtype)

    # ------------------------------------------------------------------------
    # Reductions

    def any(
        self,
        *,
        axis: AxisInt | None = None,
        out=None,
        keepdims: bool = False,
        skipna: bool = True,
    ):
        nv.validate_any((), {"out": out, "keepdims": keepdims})
        result = nanops.nanany(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def all(
        self,
        *,
        axis: AxisInt | None = None,
        out=None,
        keepdims: bool = False,
        skipna: bool = True,
    ):
        nv.validate_all((), {"out": out, "keepdims": keepdims})
        result = nanops.nanall(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def min(
        self, *, axis: AxisInt | None = None, skipna: bool = True, **kwargs
    ) -> Scalar:
        nv.validate_min((), kwargs)
        result = nanops.nanmin(
            values=self._ndarray, axis=axis, mask=self.isna(), skipna=skipna
        )
        return self._wrap_reduction_result(axis, result)

    def max(
        self, *, axis: AxisInt | None = None, skipna: bool = True, **kwargs
    ) -> Scalar:
        nv.validate_max((), kwargs)
        result = nanops.nanmax(
            values=self._ndarray, axis=axis, mask=self.isna(), skipna=skipna
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
        result = nanops.nansum(
            self._ndarray, axis=axis, skipna=skipna, min_count=min_count
        )
        return self._wrap_reduction_result(axis, result)

    def prod(
        self,
        *,
        axis: AxisInt | None = None,
        skipna: bool = True,
        min_count: int = 0,
        **kwargs,
    ) -> Scalar:
        nv.validate_prod((), kwargs)
        result = nanops.nanprod(
            self._ndarray, axis=axis, skipna=skipna, min_count=min_count
        )
        return self._wrap_reduction_result(axis, result)

    def mean(
        self,
        *,
        axis: AxisInt | None = None,
        dtype: NpDtype | None = None,
        out=None,
        keepdims: bool = False,
        skipna: bool = True,
    ):
        nv.validate_mean((), {"dtype": dtype, "out": out, "keepdims": keepdims})
        result = nanops.nanmean(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def median(
        self,
        *,
        axis: AxisInt | None = None,
        out=None,
        overwrite_input: bool = False,
        keepdims: bool = False,
        skipna: bool = True,
    ):
        nv.validate_median(
            (), {"out": out, "overwrite_input": overwrite_input, "keepdims": keepdims}
        )
        result = nanops.nanmedian(self._ndarray, axis=axis, skipna=skipna)
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
        return self._wrap_reduction_result(axis, result)

    def var(
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
            (), {"dtype": dtype, "out": out, "keepdims": keepdims}, fname="var"
        )
        result = nanops.nanvar(self._ndarray, axis=axis, skipna=skipna, ddof=ddof)
        return self._wrap_reduction_result(axis, result)

    def sem(
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
            (), {"dtype": dtype, "out": out, "keepdims": keepdims}, fname="sem"
        )
        result = nanops.nansem(self._ndarray, axis=axis, skipna=skipna, ddof=ddof)
        return self._wrap_reduction_result(axis, result)

    def kurt(
        self,
        *,
        axis: AxisInt | None = None,
        dtype: NpDtype | None = None,
        out=None,
        keepdims: bool = False,
        skipna: bool = True,
    ):
        nv.validate_stat_ddof_func(
            (), {"dtype": dtype, "out": out, "keepdims": keepdims}, fname="kurt"
        )
        result = nanops.nankurt(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    def skew(
        self,
        *,
        axis: AxisInt | None = None,
        dtype: NpDtype | None = None,
        out=None,
        keepdims: bool = False,
        skipna: bool = True,
    ):
        nv.validate_stat_ddof_func(
            (), {"dtype": dtype, "out": out, "keepdims": keepdims}, fname="skew"
        )
        result = nanops.nanskew(self._ndarray, axis=axis, skipna=skipna)
        return self._wrap_reduction_result(axis, result)

    # ------------------------------------------------------------------------
    # Additional Methods

    def to_numpy(
        self,
        dtype: npt.DTypeLike | None = None,
        copy: bool = False,
        na_value: object = lib.no_default,
    ) -> np.ndarray:
        mask = self.isna()
        if na_value is not lib.no_default and mask.any():
            result = self._ndarray.copy()
            result[mask] = na_value
        else:
            result = self._ndarray

        result = np.asarray(result, dtype=dtype)

        if copy and result is self._ndarray:
            result = result.copy()

        return result

    # ------------------------------------------------------------------------
    # Ops

    def __invert__(self) -> NumpyExtensionArray:
        return type(self)(~self._ndarray)

    def __neg__(self) -> NumpyExtensionArray:
        return type(self)(-self._ndarray)

    def __pos__(self) -> NumpyExtensionArray:
        return type(self)(+self._ndarray)

    def __abs__(self) -> NumpyExtensionArray:
        return type(self)(abs(self._ndarray))

    def _cmp_method(self, other, op):
        if isinstance(other, NumpyExtensionArray):
            other = other._ndarray

        other = ops.maybe_prepare_scalar_for_op(other, (len(self),))
        pd_op = ops.get_array_op(op)
        other = ensure_wrapped_if_datetimelike(other)
        result = pd_op(self._ndarray, other)

        if op is divmod or op is ops.rdivmod:
            a, b = result
            if isinstance(a, np.ndarray):
                # for e.g. op vs TimedeltaArray, we may already
                #  have an ExtensionArray, in which case we do not wrap
                return self._wrap_ndarray_result(a), self._wrap_ndarray_result(b)
            return a, b

        if isinstance(result, np.ndarray):
            # for e.g. multiplication vs TimedeltaArray, we may already
            #  have an ExtensionArray, in which case we do not wrap
            return self._wrap_ndarray_result(result)
        return result

    _arith_method = _cmp_method

    def _wrap_ndarray_result(self, result: np.ndarray):
        # If we have timedelta64[ns] result, return a TimedeltaArray instead
        #  of a NumpyExtensionArray
        if result.dtype.kind == "m" and is_supported_dtype(result.dtype):
            from pandas.core.arrays import TimedeltaArray

            return TimedeltaArray._simple_new(result, dtype=result.dtype)
        return type(self)(result)

    def _formatter(self, boxed: bool = False) -> Callable[[Any], str | None]:
        # NEP 51: https://github.com/numpy/numpy/pull/22449
        if self.dtype.kind in "SU":
            return "'{}'".format
        elif self.dtype == "object":
            return repr
        else:
            return str
<|endoftext|>from __future__ import annotations

import numbers
from typing import (
    TYPE_CHECKING,
    ClassVar,
    cast,
)

import numpy as np

from pandas._libs import (
    lib,
    missing as libmissing,
)

from pandas.core.dtypes.common import is_list_like
from pandas.core.dtypes.dtypes import register_extension_dtype
from pandas.core.dtypes.missing import isna

from pandas.core import ops
from pandas.core.array_algos import masked_accumulations
from pandas.core.arrays.masked import (
    BaseMaskedArray,
    BaseMaskedDtype,
)

if TYPE_CHECKING:
    import pyarrow

    from pandas._typing import (
        Dtype,
        DtypeObj,
        Self,
        npt,
        type_t,
    )


@register_extension_dtype
class BooleanDtype(BaseMaskedDtype):
    """
    Extension dtype for boolean data.

    .. warning::

       BooleanDtype is considered experimental. The implementation and
       parts of the API may change without warning.

    Attributes
    ----------
    None

    Methods
    -------
    None

    Examples
    --------
    >>> pd.BooleanDtype()
    BooleanDtype
    """

    name: ClassVar[str] = "boolean"

    # https://github.com/python/mypy/issues/4125
    # error: Signature of "type" incompatible with supertype "BaseMaskedDtype"
    @property
    def type(self) -> type:  # type: ignore[override]
        return np.bool_

    @property
    def kind(self) -> str:
        return "b"

    @property
    def numpy_dtype(self) -> np.dtype:
        return np.dtype("bool")

    @classmethod
    def construct_array_type(cls) -> type_t[BooleanArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        return BooleanArray

    def __repr__(self) -> str:
        return "BooleanDtype"

    @property
    def _is_boolean(self) -> bool:
        return True

    @property
    def _is_numeric(self) -> bool:
        return True

    def __from_arrow__(
        self, array: pyarrow.Array | pyarrow.ChunkedArray
    ) -> BooleanArray:
        """
        Construct BooleanArray from pyarrow Array/ChunkedArray.
        """
        import pyarrow

        if array.type != pyarrow.bool_() and not pyarrow.types.is_null(array.type):
            raise TypeError(f"Expected array of boolean type, got {array.type} instead")

        if isinstance(array, pyarrow.Array):
            chunks = [array]
            length = len(array)
        else:
            # pyarrow.ChunkedArray
            chunks = array.chunks
            length = array.length()

        if pyarrow.types.is_null(array.type):
            mask = np.ones(length, dtype=bool)
            # No need to init data, since all null
            data = np.empty(length, dtype=bool)
            return BooleanArray(data, mask)

        results = []
        for arr in chunks:
            buflist = arr.buffers()
            data = pyarrow.BooleanArray.from_buffers(
                arr.type, len(arr), [None, buflist[1]], offset=arr.offset
            ).to_numpy(zero_copy_only=False)
            if arr.null_count != 0:
                mask = pyarrow.BooleanArray.from_buffers(
                    arr.type, len(arr), [None, buflist[0]], offset=arr.offset
                ).to_numpy(zero_copy_only=False)
                mask = ~mask
            else:
                mask = np.zeros(len(arr), dtype=bool)

            bool_arr = BooleanArray(data, mask)
            results.append(bool_arr)

        if not results:
            return BooleanArray(
                np.array([], dtype=np.bool_), np.array([], dtype=np.bool_)
            )
        else:
            return BooleanArray._concat_same_type(results)


def coerce_to_array(
    values, mask=None, copy: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Coerce the input values array to numpy arrays with a mask.

    Parameters
    ----------
    values : 1D list-like
    mask : bool 1D array, optional
    copy : bool, default False
        if True, copy the input

    Returns
    -------
    tuple of (values, mask)
    """
    if isinstance(values, BooleanArray):
        if mask is not None:
            raise ValueError("cannot pass mask for BooleanArray input")
        values, mask = values._data, values._mask
        if copy:
            values = values.copy()
            mask = mask.copy()
        return values, mask

    mask_values = None
    if isinstance(values, np.ndarray) and values.dtype == np.bool_:
        if copy:
            values = values.copy()
    elif isinstance(values, np.ndarray) and values.dtype.kind in "iufcb":
        mask_values = isna(values)

        values_bool = np.zeros(len(values), dtype=bool)
        values_bool[~mask_values] = values[~mask_values].astype(bool)

        if not np.all(
            values_bool[~mask_values].astype(values.dtype) == values[~mask_values]
        ):
            raise TypeError("Need to pass bool-like values")

        values = values_bool
    else:
        values_object = np.asarray(values, dtype=object)

        inferred_dtype = lib.infer_dtype(values_object, skipna=True)
        integer_like = ("floating", "integer", "mixed-integer-float")
        if inferred_dtype not in ("boolean", "empty") + integer_like:
            raise TypeError("Need to pass bool-like values")

        # mypy does not narrow the type of mask_values to npt.NDArray[np.bool_]
        # within this branch, it assumes it can also be None
        mask_values = cast("npt.NDArray[np.bool_]", isna(values_object))
        values = np.zeros(len(values), dtype=bool)
        values[~mask_values] = values_object[~mask_values].astype(bool)

        # if the values were integer-like, validate it were actually 0/1's
        if (inferred_dtype in integer_like) and not (
            np.all(
                values[~mask_values].astype(float)
                == values_object[~mask_values].astype(float)
            )
        ):
            raise TypeError("Need to pass bool-like values")

    if mask is None and mask_values is None:
        mask = np.zeros(values.shape, dtype=bool)
    elif mask is None:
        mask = mask_values
    else:
        if isinstance(mask, np.ndarray) and mask.dtype == np.bool_:
            if mask_values is not None:
                mask = mask | mask_values
            else:
                if copy:
                    mask = mask.copy()
        else:
            mask = np.array(mask, dtype=bool)
            if mask_values is not None:
                mask = mask | mask_values

    if values.shape != mask.shape:
        raise ValueError("values.shape and mask.shape must match")

    return values, mask


class BooleanArray(BaseMaskedArray):
    """
    Array of boolean (True/False) data with missing values.

    This is a pandas Extension array for boolean data, under the hood
    represented by 2 numpy arrays: a boolean array with the data and
    a boolean array with the mask (True indicating missing).

    BooleanArray implements Kleene logic (sometimes called three-value
    logic) for logical operations. See :ref:`boolean.kleene` for more.

    To construct an BooleanArray from generic array-like input, use
    :func:`pandas.array` specifying ``dtype="boolean"`` (see examples
    below).

    .. warning::

       BooleanArray is considered experimental. The implementation and
       parts of the API may change without warning.

    Parameters
    ----------
    values : numpy.ndarray
        A 1-d boolean-dtype array with the data.
    mask : numpy.ndarray
        A 1-d boolean-dtype array indicating missing values (True
        indicates missing).
    copy : bool, default False
        Whether to copy the `values` and `mask` arrays.

    Attributes
    ----------
    None

    Methods
    -------
    None

    Returns
    -------
    BooleanArray

    Examples
    --------
    Create an BooleanArray with :func:`pandas.array`:

    >>> pd.array([True, False, None], dtype="boolean")
    <BooleanArray>
    [True, False, <NA>]
    Length: 3, dtype: boolean
    """

    # The value used to fill '_data' to avoid upcasting
    _internal_fill_value = False
    # Fill values used for any/all
    # Incompatible types in assignment (expression has type "bool", base class
    # "BaseMaskedArray" defined the type as "<typing special form>")
    _truthy_value = True  # type: ignore[assignment]
    _falsey_value = False  # type: ignore[assignment]
    _TRUE_VALUES = {"True", "TRUE", "true", "1", "1.0"}
    _FALSE_VALUES = {"False", "FALSE", "false", "0", "0.0"}

    @classmethod
    def _simple_new(cls, values: np.ndarray, mask: npt.NDArray[np.bool_]) -> Self:
        result = super()._simple_new(values, mask)
        result._dtype = BooleanDtype()
        return result

    def __init__(
        self, values: np.ndarray, mask: np.ndarray, copy: bool = False
    ) -> None:
        if not (isinstance(values, np.ndarray) and values.dtype == np.bool_):
            raise TypeError(
                "values should be boolean numpy array. Use "
                "the 'pd.array' function instead"
            )
        self._dtype = BooleanDtype()
        super().__init__(values, mask, copy=copy)

    @property
    def dtype(self) -> BooleanDtype:
        return self._dtype

    @classmethod
    def _from_sequence_of_strings(
        cls,
        strings: list[str],
        *,
        dtype: Dtype | None = None,
        copy: bool = False,
        true_values: list[str] | None = None,
        false_values: list[str] | None = None,
    ) -> BooleanArray:
        true_values_union = cls._TRUE_VALUES.union(true_values or [])
        false_values_union = cls._FALSE_VALUES.union(false_values or [])

        def map_string(s) -> bool:
            if s in true_values_union:
                return True
            elif s in false_values_union:
                return False
            else:
                raise ValueError(f"{s} cannot be cast to bool")

        scalars = np.array(strings, dtype=object)
        mask = isna(scalars)
        scalars[~mask] = list(map(map_string, scalars[~mask]))
        return cls._from_sequence(scalars, dtype=dtype, copy=copy)

    _HANDLED_TYPES = (np.ndarray, numbers.Number, bool, np.bool_)

    @classmethod
    def _coerce_to_array(
        cls, value, *, dtype: DtypeObj, copy: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        if dtype:
            assert dtype == "boolean"
        return coerce_to_array(value, copy=copy)

    def _logical_method(self, other, op):
        assert op.__name__ in {"or_", "ror_", "and_", "rand_", "xor", "rxor"}
        other_is_scalar = lib.is_scalar(other)
        mask = None

        if isinstance(other, BooleanArray):
            other, mask = other._data, other._mask
        elif is_list_like(other):
            other = np.asarray(other, dtype="bool")
            if other.ndim > 1:
                raise NotImplementedError("can only perform ops with 1-d structures")
            other, mask = coerce_to_array(other, copy=False)
        elif isinstance(other, np.bool_):
            other = other.item()

        if other_is_scalar and other is not libmissing.NA and not lib.is_bool(other):
            raise TypeError(
                "'other' should be pandas.NA or a bool. "
                f"Got {type(other).__name__} instead."
            )

        if not other_is_scalar and len(self) != len(other):
            raise ValueError("Lengths must match")

        if op.__name__ in {"or_", "ror_"}:
            result, mask = ops.kleene_or(self._data, other, self._mask, mask)
        elif op.__name__ in {"and_", "rand_"}:
            result, mask = ops.kleene_and(self._data, other, self._mask, mask)
        else:
            # i.e. xor, rxor
            result, mask = ops.kleene_xor(self._data, other, self._mask, mask)

        # i.e. BooleanArray
        return self._maybe_mask_result(result, mask)

    def _accumulate(
        self, name: str, *, skipna: bool = True, **kwargs
    ) -> BaseMaskedArray:
        data = self._data
        mask = self._mask
        if name in ("cummin", "cummax"):
            op = getattr(masked_accumulations, name)
            data, mask = op(data, mask, skipna=skipna, **kwargs)
            return self._simple_new(data, mask)
        else:
            from pandas.core.arrays import IntegerArray

            return IntegerArray(data.astype(int), mask)._accumulate(
                name, skipna=skipna, **kwargs
            )
<|endoftext|>from __future__ import annotations

import operator
import re
from typing import (
    TYPE_CHECKING,
    Callable,
    Union,
)
import warnings

import numpy as np

from pandas._libs import (
    lib,
    missing as libmissing,
)
from pandas.compat import (
    pa_version_under10p1,
    pa_version_under13p0,
    pa_version_under16p0,
)
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import (
    is_scalar,
    pandas_dtype,
)
from pandas.core.dtypes.missing import isna

from pandas.core.arrays._arrow_string_mixins import ArrowStringArrayMixin
from pandas.core.arrays.arrow import ArrowExtensionArray
from pandas.core.arrays.boolean import BooleanDtype
from pandas.core.arrays.floating import Float64Dtype
from pandas.core.arrays.integer import Int64Dtype
from pandas.core.arrays.numeric import NumericDtype
from pandas.core.arrays.string_ import (
    BaseStringArray,
    StringDtype,
)
from pandas.core.strings.object_array import ObjectStringArrayMixin

if not pa_version_under10p1:
    import pyarrow as pa
    import pyarrow.compute as pc


if TYPE_CHECKING:
    from collections.abc import Sequence

    from pandas._typing import (
        ArrayLike,
        Dtype,
        Scalar,
        Self,
        npt,
    )

    from pandas import Series


ArrowStringScalarOrNAT = Union[str, libmissing.NAType]


def _chk_pyarrow_available() -> None:
    if pa_version_under10p1:
        msg = "pyarrow>=10.0.1 is required for PyArrow backed ArrowExtensionArray."
        raise ImportError(msg)


def _is_string_view(typ):
    return not pa_version_under16p0 and pa.types.is_string_view(typ)


# TODO: Inherit directly from BaseStringArrayMethods. Currently we inherit from
# ObjectStringArrayMixin because we want to have the object-dtype based methods as
# fallback for the ones that pyarrow doesn't yet support


class ArrowStringArray(ObjectStringArrayMixin, ArrowExtensionArray, BaseStringArray):
    """
    Extension array for string data in a ``pyarrow.ChunkedArray``.

    .. warning::

       ArrowStringArray is considered experimental. The implementation and
       parts of the API may change without warning.

    Parameters
    ----------
    values : pyarrow.Array or pyarrow.ChunkedArray
        The array of data.

    Attributes
    ----------
    None

    Methods
    -------
    None

    See Also
    --------
    :func:`pandas.array`
        The recommended function for creating a ArrowStringArray.
    Series.str
        The string methods are available on Series backed by
        a ArrowStringArray.

    Notes
    -----
    ArrowStringArray returns a BooleanArray for comparison methods.

    Examples
    --------
    >>> pd.array(['This is', 'some text', None, 'data.'], dtype="string[pyarrow]")
    <ArrowStringArray>
    ['This is', 'some text', <NA>, 'data.']
    Length: 4, dtype: string
    """

    # error: Incompatible types in assignment (expression has type "StringDtype",
    # base class "ArrowExtensionArray" defined the type as "ArrowDtype")
    _dtype: StringDtype  # type: ignore[assignment]
    _storage = "pyarrow"
    _na_value: libmissing.NAType | float = libmissing.NA

    def __init__(self, values) -> None:
        _chk_pyarrow_available()
        if isinstance(values, (pa.Array, pa.ChunkedArray)) and (
            pa.types.is_string(values.type)
            or _is_string_view(values.type)
            or (
                pa.types.is_dictionary(values.type)
                and (
                    pa.types.is_string(values.type.value_type)
                    or pa.types.is_large_string(values.type.value_type)
                    or _is_string_view(values.type.value_type)
                )
            )
        ):
            values = pc.cast(values, pa.large_string())

        super().__init__(values)
        self._dtype = StringDtype(storage=self._storage, na_value=self._na_value)

        if not pa.types.is_large_string(self._pa_array.type):
            raise ValueError(
                "ArrowStringArray requires a PyArrow (chunked) array of "
                "large_string type"
            )

    @classmethod
    def _box_pa_scalar(cls, value, pa_type: pa.DataType | None = None) -> pa.Scalar:
        pa_scalar = super()._box_pa_scalar(value, pa_type)
        if pa.types.is_string(pa_scalar.type) and pa_type is None:
            pa_scalar = pc.cast(pa_scalar, pa.large_string())
        return pa_scalar

    @classmethod
    def _box_pa_array(
        cls, value, pa_type: pa.DataType | None = None, copy: bool = False
    ) -> pa.Array | pa.ChunkedArray:
        pa_array = super()._box_pa_array(value, pa_type)
        if pa.types.is_string(pa_array.type) and pa_type is None:
            pa_array = pc.cast(pa_array, pa.large_string())
        return pa_array

    def __len__(self) -> int:
        """
        Length of this array.

        Returns
        -------
        length : int
        """
        return len(self._pa_array)

    @classmethod
    def _from_sequence(cls, scalars, *, dtype: Dtype | None = None, copy: bool = False):
        from pandas.core.arrays.masked import BaseMaskedArray

        _chk_pyarrow_available()

        if dtype and not (isinstance(dtype, str) and dtype == "string"):
            dtype = pandas_dtype(dtype)
            assert isinstance(dtype, StringDtype) and dtype.storage == "pyarrow"

        if isinstance(scalars, BaseMaskedArray):
            # avoid costly conversion to object dtype in ensure_string_array and
            # numerical issues with Float32Dtype
            na_values = scalars._mask
            result = scalars._data
            result = lib.ensure_string_array(result, copy=copy, convert_na_value=False)
            return cls(pa.array(result, mask=na_values, type=pa.large_string()))
        elif isinstance(scalars, (pa.Array, pa.ChunkedArray)):
            return cls(pc.cast(scalars, pa.large_string()))

        # convert non-na-likes to str
        result = lib.ensure_string_array(scalars, copy=copy)
        return cls(pa.array(result, type=pa.large_string(), from_pandas=True))

    @classmethod
    def _from_sequence_of_strings(
        cls, strings, dtype: Dtype | None = None, copy: bool = False
    ):
        return cls._from_sequence(strings, dtype=dtype, copy=copy)

    @property
    def dtype(self) -> StringDtype:  # type: ignore[override]
        """
        An instance of 'string[pyarrow]'.
        """
        return self._dtype

    def insert(self, loc: int, item) -> ArrowStringArray:
        if self.dtype.na_value is np.nan and item is np.nan:
            item = libmissing.NA
        if not isinstance(item, str) and item is not libmissing.NA:
            raise TypeError(
                f"Invalid value '{item}' for dtype 'str'. Value should be a "
                f"string or missing value, got '{type(item).__name__}' instead."
            )
        return super().insert(loc, item)

    def _convert_bool_result(self, values, na=lib.no_default, method_name=None):
        if na is not lib.no_default and not isna(na) and not isinstance(na, bool):
            # GH#59561
            warnings.warn(
                f"Allowing a non-bool 'na' in obj.str.{method_name} is deprecated "
                "and will raise in a future version.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )
            na = bool(na)

        if self.dtype.na_value is np.nan:
            if na is lib.no_default or isna(na):
                # NaN propagates as False
                values = values.fill_null(False)
            else:
                values = values.fill_null(na)
            return values.to_numpy()
        else:
            if na is not lib.no_default and not isna(
                na
            ):  # pyright: ignore [reportGeneralTypeIssues]
                values = values.fill_null(na)
        return BooleanDtype().__from_arrow__(values)

    def _maybe_convert_setitem_value(self, value):
        """Maybe convert value to be pyarrow compatible."""
        if is_scalar(value):
            if isna(value):
                value = None
            elif not isinstance(value, str):
                raise TypeError(
                    f"Invalid value '{value}' for dtype 'str'. Value should be a "
                    f"string or missing value, got '{type(value).__name__}' instead."
                )
        else:
            value = np.array(value, dtype=object, copy=True)
            value[isna(value)] = None
            for v in value:
                if not (v is None or isinstance(v, str)):
                    raise TypeError(
                        "Invalid value for dtype 'str'. Value should be a "
                        "string or missing value (or array of those)."
                    )
        return super()._maybe_convert_setitem_value(value)

    def isin(self, values: ArrayLike) -> npt.NDArray[np.bool_]:
        value_set = [
            pa_scalar.as_py()
            for pa_scalar in [pa.scalar(value, from_pandas=True) for value in values]
            if pa_scalar.type in (pa.string(), pa.null(), pa.large_string())
        ]

        # short-circuit to return all False array.
        if not len(value_set):
            return np.zeros(len(self), dtype=bool)

        result = pc.is_in(
            self._pa_array, value_set=pa.array(value_set, type=self._pa_array.type)
        )
        # pyarrow 2.0.0 returned nulls, so we explicily specify dtype to convert nulls
        # to False
        return np.array(result, dtype=np.bool_)

    def astype(self, dtype, copy: bool = True):
        dtype = pandas_dtype(dtype)

        if dtype == self.dtype:
            if copy:
                return self.copy()
            return self
        elif isinstance(dtype, NumericDtype):
            data = self._pa_array.cast(pa.from_numpy_dtype(dtype.numpy_dtype))
            return dtype.__from_arrow__(data)
        elif isinstance(dtype, np.dtype) and np.issubdtype(dtype, np.floating):
            return self.to_numpy(dtype=dtype, na_value=np.nan)

        return super().astype(dtype, copy=copy)

    @property
    def _data(self):
        # dask accesses ._data directlys
        warnings.warn(
            f"{type(self).__name__}._data is a deprecated and will be removed "
            "in a future version, use ._pa_array instead",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
        return self._pa_array

    # ------------------------------------------------------------------------
    # String methods interface

    _str_isalnum = ArrowStringArrayMixin._str_isalnum
    _str_isalpha = ArrowStringArrayMixin._str_isalpha
    _str_isdecimal = ArrowStringArrayMixin._str_isdecimal
    _str_isdigit = ArrowStringArrayMixin._str_isdigit
    _str_islower = ArrowStringArrayMixin._str_islower
    _str_isnumeric = ArrowStringArrayMixin._str_isnumeric
    _str_isspace = ArrowStringArrayMixin._str_isspace
    _str_istitle = ArrowStringArrayMixin._str_istitle
    _str_isupper = ArrowStringArrayMixin._str_isupper

    _str_map = BaseStringArray._str_map
    _str_startswith = ArrowStringArrayMixin._str_startswith
    _str_endswith = ArrowStringArrayMixin._str_endswith
    _str_pad = ArrowStringArrayMixin._str_pad
    _str_lower = ArrowStringArrayMixin._str_lower
    _str_upper = ArrowStringArrayMixin._str_upper
    _str_strip = ArrowStringArrayMixin._str_strip
    _str_lstrip = ArrowStringArrayMixin._str_lstrip
    _str_rstrip = ArrowStringArrayMixin._str_rstrip
    _str_removesuffix = ArrowStringArrayMixin._str_removesuffix
    _str_get = ArrowStringArrayMixin._str_get
    _str_capitalize = ArrowStringArrayMixin._str_capitalize
    _str_title = ArrowStringArrayMixin._str_title
    _str_swapcase = ArrowStringArrayMixin._str_swapcase
    _str_slice_replace = ArrowStringArrayMixin._str_slice_replace
    _str_len = ArrowStringArrayMixin._str_len
    _str_slice = ArrowStringArrayMixin._str_slice

    @staticmethod
    def _is_re_pattern_with_flags(pat: str | re.Pattern) -> bool:
        # check if `pat` is a compiled regex pattern with flags that are not
        # supported by pyarrow
        return (
            isinstance(pat, re.Pattern)
            and (pat.flags & ~(re.IGNORECASE | re.UNICODE)) != 0
        )

    @staticmethod
    def _preprocess_re_pattern(pat: re.Pattern, case: bool) -> tuple[str, bool, int]:
        pattern = pat.pattern
        flags = pat.flags
        # flags is not supported by pyarrow, but `case` is -> extract and remove
        if flags & re.IGNORECASE:
            case = False
            flags = flags & ~re.IGNORECASE
        # when creating a pattern with re.compile and a string, it automatically
        # gets a UNICODE flag, while pyarrow assumes unicode for strings anyway
        flags = flags & ~re.UNICODE
        return pattern, case, flags

    def _str_contains(
        self,
        pat,
        case: bool = True,
        flags: int = 0,
        na=lib.no_default,
        regex: bool = True,
    ):
        if flags or self._is_re_pattern_with_flags(pat):
            return super()._str_contains(pat, case, flags, na, regex)
        if isinstance(pat, re.Pattern):
            # TODO flags passed separately by user are ignored
            pat, case, flags = self._preprocess_re_pattern(pat, case)

        return ArrowStringArrayMixin._str_contains(self, pat, case, flags, na, regex)

    def _str_match(
        self,
        pat: str | re.Pattern,
        case: bool = True,
        flags: int = 0,
        na: Scalar | lib.NoDefault = lib.no_default,
    ):
        if flags or self._is_re_pattern_with_flags(pat):
            return super()._str_match(pat, case, flags, na)
        if isinstance(pat, re.Pattern):
            pat, case, flags = self._preprocess_re_pattern(pat, case)

        return ArrowStringArrayMixin._str_match(self, pat, case, flags, na)

    def _str_fullmatch(
        self,
        pat: str | re.Pattern,
        case: bool = True,
        flags: int = 0,
        na: Scalar | lib.NoDefault = lib.no_default,
    ):
        if flags or self._is_re_pattern_with_flags(pat):
            return super()._str_fullmatch(pat, case, flags, na)
        if isinstance(pat, re.Pattern):
            pat, case, flags = self._preprocess_re_pattern(pat, case)

        return ArrowStringArrayMixin._str_fullmatch(self, pat, case, flags, na)

    def _str_replace(
        self,
        pat: str | re.Pattern,
        repl: str | Callable,
        n: int = -1,
        case: bool = True,
        flags: int = 0,
        regex: bool = True,
    ):
        if (
            isinstance(pat, re.Pattern)
            or callable(repl)
            or not case
            or flags
            or (  # substitution contains a named group pattern
                # https://docs.python.org/3/library/re.html
                isinstance(repl, str)
                and (r"\g<" in repl or re.search(r"\\\d", repl) is not None)
            )
        ):
            return super()._str_replace(pat, repl, n, case, flags, regex)

        return ArrowStringArrayMixin._str_replace(
            self, pat, repl, n, case, flags, regex
        )

    def _str_repeat(self, repeats: int | Sequence[int]):
        if not isinstance(repeats, int):
            return super()._str_repeat(repeats)
        else:
            return ArrowExtensionArray._str_repeat(self, repeats=repeats)

    def _str_removeprefix(self, prefix: str):
        if not pa_version_under13p0:
            return ArrowStringArrayMixin._str_removeprefix(self, prefix)
        return super()._str_removeprefix(prefix)

    def _str_count(self, pat: str, flags: int = 0):
        if flags:
            return super()._str_count(pat, flags)
        result = pc.count_substring_regex(self._pa_array, pat)
        return self._convert_int_result(result)

    def _str_find(self, sub: str, start: int = 0, end: int | None = None):
        if (
            pa_version_under13p0
            and not (start != 0 and end is not None)
            and not (start == 0 and end is None)
        ):
            # GH#59562
            return super()._str_find(sub, start, end)
        return ArrowStringArrayMixin._str_find(self, sub, start, end)

    def _str_get_dummies(self, sep: str = "|"):
        dummies_pa, labels = ArrowExtensionArray(self._pa_array)._str_get_dummies(sep)
        if len(labels) == 0:
            return np.empty(shape=(0, 0), dtype=np.int64), labels
        dummies = np.vstack(dummies_pa.to_numpy())
        return dummies.astype(np.int64, copy=False), labels

    def _convert_int_result(self, result):
        if self.dtype.na_value is np.nan:
            if isinstance(result, pa.Array):
                result = result.to_numpy(zero_copy_only=False)
            else:
                result = result.to_numpy()
            if result.dtype == np.int32:
                result = result.astype(np.int64)
            return result

        return Int64Dtype().__from_arrow__(result)

    def _convert_rank_result(self, result):
        if self.dtype.na_value is np.nan:
            if isinstance(result, pa.Array):
                result = result.to_numpy(zero_copy_only=False)
            else:
                result = result.to_numpy()
            return result.astype("float64", copy=False)

        return Float64Dtype().__from_arrow__(result)

    def _reduce(
        self, name: str, *, skipna: bool = True, keepdims: bool = False, **kwargs
    ):
        if self.dtype.na_value is np.nan and name in ["any", "all"]:
            if not skipna:
                nas = pc.is_null(self._pa_array)
                arr = pc.or_kleene(nas, pc.not_equal(self._pa_array, ""))
            else:
                arr = pc.not_equal(self._pa_array, "")
            result = ArrowExtensionArray(arr)._reduce(
                name, skipna=skipna, keepdims=keepdims, **kwargs
            )
            if keepdims:
                # ArrowExtensionArray will return a length-1 bool[pyarrow] array
                return result.astype(np.bool_)
            return result

        if name in ("min", "max", "sum", "argmin", "argmax"):
            result = self._reduce_calc(name, skipna=skipna, keepdims=keepdims, **kwargs)
        else:
            raise TypeError(f"Cannot perform reduction '{name}' with string dtype")

        if name in ("argmin", "argmax") and isinstance(result, pa.Array):
            return self._convert_int_result(result)
        elif isinstance(result, pa.Array):
            return type(self)(result)
        else:
            return result

    def value_counts(self, dropna: bool = True) -> Series:
        result = super().value_counts(dropna=dropna)
        if self.dtype.na_value is np.nan:
            res_values = result._values.to_numpy()
            return result._constructor(
                res_values, index=result.index, name=result.name, copy=False
            )
        return result

    def _cmp_method(self, other, op):
        if (
            isinstance(other, (BaseStringArray, ArrowExtensionArray))
            and self.dtype.na_value is not libmissing.NA
            and other.dtype.na_value is libmissing.NA
        ):
            # NA has priority of NaN semantics
            return NotImplemented

        result = super()._cmp_method(other, op)
        if self.dtype.na_value is np.nan:
            if op == operator.ne:
                return result.to_numpy(np.bool_, na_value=True)
            else:
                return result.to_numpy(np.bool_, na_value=False)
        return result

    def __pos__(self) -> Self:
        raise TypeError(f"bad operand type for unary +: '{self.dtype}'")


class ArrowStringArrayNumpySemantics(ArrowStringArray):
    _na_value = np.nan
<|endoftext|>from __future__ import annotations

import numbers
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
)

import numpy as np

from pandas._libs import (
    lib,
    missing as libmissing,
)
from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly

from pandas.core.dtypes.common import (
    is_integer_dtype,
    is_string_dtype,
    pandas_dtype,
)

from pandas.core.arrays.masked import (
    BaseMaskedArray,
    BaseMaskedDtype,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pyarrow

    from pandas._typing import (
        Dtype,
        DtypeObj,
        Self,
        npt,
    )


class NumericDtype(BaseMaskedDtype):
    _default_np_dtype: np.dtype
    _checker: Callable[[Any], bool]  # is_foo_dtype

    def __repr__(self) -> str:
        return f"{self.name}Dtype()"

    @cache_readonly
    def is_signed_integer(self) -> bool:
        return self.kind == "i"

    @cache_readonly
    def is_unsigned_integer(self) -> bool:
        return self.kind == "u"

    @property
    def _is_numeric(self) -> bool:
        return True

    def __from_arrow__(
        self, array: pyarrow.Array | pyarrow.ChunkedArray
    ) -> BaseMaskedArray:
        """
        Construct IntegerArray/FloatingArray from pyarrow Array/ChunkedArray.
        """
        import pyarrow

        from pandas.core.arrays.arrow._arrow_utils import (
            pyarrow_array_to_numpy_and_mask,
        )

        array_class = self.construct_array_type()

        pyarrow_type = pyarrow.from_numpy_dtype(self.type)
        if not array.type.equals(pyarrow_type) and not pyarrow.types.is_null(
            array.type
        ):
            # test_from_arrow_type_error raise for string, but allow
            #  through itemsize conversion GH#31896
            rt_dtype = pandas_dtype(array.type.to_pandas_dtype())
            if rt_dtype.kind not in "iuf":
                # Could allow "c" or potentially disallow float<->int conversion,
                #  but at the moment we specifically test that uint<->int works
                raise TypeError(
                    f"Expected array of {self} type, got {array.type} instead"
                )

            array = array.cast(pyarrow_type)

        if isinstance(array, pyarrow.ChunkedArray):
            # TODO this "if" can be removed when requiring pyarrow >= 10.0, which fixed
            # combine_chunks for empty arrays https://github.com/apache/arrow/pull/13757
            if array.num_chunks == 0:
                array = pyarrow.array([], type=array.type)
            else:
                array = array.combine_chunks()

        data, mask = pyarrow_array_to_numpy_and_mask(array, dtype=self.numpy_dtype)
        return array_class(data.copy(), ~mask, copy=False)

    @classmethod
    def _get_dtype_mapping(cls) -> Mapping[np.dtype, NumericDtype]:
        raise AbstractMethodError(cls)

    @classmethod
    def _standardize_dtype(cls, dtype: NumericDtype | str | np.dtype) -> NumericDtype:
        """
        Convert a string representation or a numpy dtype to NumericDtype.
        """
        if isinstance(dtype, str) and (dtype.startswith(("Int", "UInt", "Float"))):
            # Avoid DeprecationWarning from NumPy about np.dtype("Int64")
            # https://github.com/numpy/numpy/pull/7476
            dtype = dtype.lower()

        if not isinstance(dtype, NumericDtype):
            mapping = cls._get_dtype_mapping()
            try:
                dtype = mapping[np.dtype(dtype)]
            except KeyError as err:
                raise ValueError(f"invalid dtype specified {dtype}") from err
        return dtype

    @classmethod
    def _safe_cast(cls, values: np.ndarray, dtype: np.dtype, copy: bool) -> np.ndarray:
        """
        Safely cast the values to the given dtype.

        "safe" in this context means the casting is lossless.
        """
        raise AbstractMethodError(cls)


def _coerce_to_data_and_mask(
    values, dtype, copy: bool, dtype_cls: type[NumericDtype], default_dtype: np.dtype
):
    checker = dtype_cls._checker

    mask = None
    inferred_type = None

    if dtype is None and hasattr(values, "dtype"):
        if checker(values.dtype):
            dtype = values.dtype

    if dtype is not None:
        dtype = dtype_cls._standardize_dtype(dtype)

    cls = dtype_cls.construct_array_type()
    if isinstance(values, cls):
        values, mask = values._data, values._mask
        if dtype is not None:
            values = values.astype(dtype.numpy_dtype, copy=False)

        if copy:
            values = values.copy()
            mask = mask.copy()
        return values, mask, dtype, inferred_type

    original = values
    if not copy:
        values = np.asarray(values)
    else:
        values = np.array(values, copy=copy)
    inferred_type = None
    if values.dtype == object or is_string_dtype(values.dtype):
        inferred_type = lib.infer_dtype(values, skipna=True)
        if inferred_type == "boolean" and dtype is None:
            name = dtype_cls.__name__.strip("_")
            raise TypeError(f"{values.dtype} cannot be converted to {name}")

    elif values.dtype.kind == "b" and checker(dtype):
        if not copy:
            values = np.asarray(values, dtype=default_dtype)
        else:
            values = np.array(values, dtype=default_dtype, copy=copy)

    elif values.dtype.kind not in "iuf":
        name = dtype_cls.__name__.strip("_")
        raise TypeError(f"{values.dtype} cannot be converted to {name}")

    if values.ndim != 1:
        raise TypeError("values must be a 1D list-like")

    if mask is None:
        if values.dtype.kind in "iu":
            # fastpath
            mask = np.zeros(len(values), dtype=np.bool_)
        else:
            mask = libmissing.is_numeric_na(values)
    else:
        assert len(mask) == len(values)

    if mask.ndim != 1:
        raise TypeError("mask must be a 1D list-like")

    # infer dtype if needed
    if dtype is None:
        dtype = default_dtype
    else:
        dtype = dtype.numpy_dtype

    if is_integer_dtype(dtype) and values.dtype.kind == "f" and len(values) > 0:
        if mask.all():
            values = np.ones(values.shape, dtype=dtype)
        else:
            idx = np.nanargmax(values)
            if int(values[idx]) != original[idx]:
                # We have ints that lost precision during the cast.
                inferred_type = lib.infer_dtype(original, skipna=True)
                if (
                    inferred_type not in ["floating", "mixed-integer-float"]
                    and not mask.any()
                ):
                    values = np.asarray(original, dtype=dtype)
                else:
                    values = np.asarray(original, dtype="object")

    # we copy as need to coerce here
    if mask.any():
        values = values.copy()
        values[mask] = cls._internal_fill_value
    if inferred_type in ("string", "unicode"):
        # casts from str are always safe since they raise
        # a ValueError if the str cannot be parsed into a float
        values = values.astype(dtype, copy=copy)
    else:
        values = dtype_cls._safe_cast(values, dtype, copy=False)

    return values, mask, dtype, inferred_type


class NumericArray(BaseMaskedArray):
    """
    Base class for IntegerArray and FloatingArray.
    """

    _dtype_cls: type[NumericDtype]

    def __init__(
        self, values: np.ndarray, mask: npt.NDArray[np.bool_], copy: bool = False
    ) -> None:
        checker = self._dtype_cls._checker
        if not (isinstance(values, np.ndarray) and checker(values.dtype)):
            descr = (
                "floating"
                if self._dtype_cls.kind == "f"  # type: ignore[comparison-overlap]
                else "integer"
            )
            raise TypeError(
                f"values should be {descr} numpy array. Use "
                "the 'pd.array' function instead"
            )
        if values.dtype == np.float16:
            # If we don't raise here, then accessing self.dtype would raise
            raise TypeError("FloatingArray does not support np.float16 dtype.")

        super().__init__(values, mask, copy=copy)

    @cache_readonly
    def dtype(self) -> NumericDtype:
        mapping = self._dtype_cls._get_dtype_mapping()
        return mapping[self._data.dtype]

    @classmethod
    def _coerce_to_array(
        cls, value, *, dtype: DtypeObj, copy: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        dtype_cls = cls._dtype_cls
        default_dtype = dtype_cls._default_np_dtype
        values, mask, _, _ = _coerce_to_data_and_mask(
            value, dtype, copy, dtype_cls, default_dtype
        )
        return values, mask

    @classmethod
    def _from_sequence_of_strings(
        cls, strings, *, dtype: Dtype | None = None, copy: bool = False
    ) -> Self:
        from pandas.core.tools.numeric import to_numeric

        scalars = to_numeric(strings, errors="raise", dtype_backend="numpy_nullable")
        return cls._from_sequence(scalars, dtype=dtype, copy=copy)

    _HANDLED_TYPES = (np.ndarray, numbers.Number)
<|endoftext|>from pandas.core.arrays.arrow import ArrowExtensionArray
from pandas.core.arrays.base import (
    ExtensionArray,
    ExtensionOpsMixin,
    ExtensionScalarOpsMixin,
)
from pandas.core.arrays.boolean import BooleanArray
from pandas.core.arrays.categorical import Categorical
from pandas.core.arrays.datetimes import DatetimeArray
from pandas.core.arrays.floating import FloatingArray
from pandas.core.arrays.integer import IntegerArray
from pandas.core.arrays.interval import IntervalArray
from pandas.core.arrays.masked import BaseMaskedArray
from pandas.core.arrays.numpy_ import NumpyExtensionArray
from pandas.core.arrays.period import (
    PeriodArray,
    period_array,
)
from pandas.core.arrays.sparse import SparseArray
from pandas.core.arrays.string_ import StringArray
from pandas.core.arrays.string_arrow import ArrowStringArray
from pandas.core.arrays.timedeltas import TimedeltaArray

__all__ = [
    "ArrowExtensionArray",
    "ExtensionArray",
    "ExtensionOpsMixin",
    "ExtensionScalarOpsMixin",
    "ArrowStringArray",
    "BaseMaskedArray",
    "BooleanArray",
    "Categorical",
    "DatetimeArray",
    "FloatingArray",
    "IntegerArray",
    "IntervalArray",
    "NumpyExtensionArray",
    "PeriodArray",
    "period_array",
    "SparseArray",
    "StringArray",
    "TimedeltaArray",
]
<|endoftext|>from __future__ import annotations

from typing import ClassVar

import numpy as np

from pandas.core.dtypes.base import register_extension_dtype
from pandas.core.dtypes.common import is_float_dtype

from pandas.core.arrays.numeric import (
    NumericArray,
    NumericDtype,
)


class FloatingDtype(NumericDtype):
    """
    An ExtensionDtype to hold a single size of floating dtype.

    These specific implementations are subclasses of the non-public
    FloatingDtype. For example we have Float32Dtype to represent float32.

    The attributes name & type are set when these subclasses are created.
    """

    _default_np_dtype = np.dtype(np.float64)
    _checker = is_float_dtype

    @classmethod
    def construct_array_type(cls) -> type[FloatingArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        return FloatingArray

    @classmethod
    def _get_dtype_mapping(cls) -> dict[np.dtype, FloatingDtype]:
        return NUMPY_FLOAT_TO_DTYPE

    @classmethod
    def _safe_cast(cls, values: np.ndarray, dtype: np.dtype, copy: bool) -> np.ndarray:
        """
        Safely cast the values to the given dtype.

        "safe" in this context means the casting is lossless.
        """
        # This is really only here for compatibility with IntegerDtype
        # Here for compat with IntegerDtype
        return values.astype(dtype, copy=copy)


class FloatingArray(NumericArray):
    """
    Array of floating (optional missing) values.

    .. warning::

       FloatingArray is currently experimental, and its API or internal
       implementation may change without warning. Especially the behaviour
       regarding NaN (distinct from NA missing values) is subject to change.

    We represent a FloatingArray with 2 numpy arrays:

    - data: contains a numpy float array of the appropriate dtype
    - mask: a boolean array holding a mask on the data, True is missing

    To construct an FloatingArray from generic array-like input, use
    :func:`pandas.array` with one of the float dtypes (see examples).

    See :ref:`integer_na` for more.

    Parameters
    ----------
    values : numpy.ndarray
        A 1-d float-dtype array.
    mask : numpy.ndarray
        A 1-d boolean-dtype array indicating missing values.
    copy : bool, default False
        Whether to copy the `values` and `mask`.

    Attributes
    ----------
    None

    Methods
    -------
    None

    Returns
    -------
    FloatingArray

    Examples
    --------
    Create an FloatingArray with :func:`pandas.array`:

    >>> pd.array([0.1, None, 0.3], dtype=pd.Float32Dtype())
    <FloatingArray>
    [0.1, <NA>, 0.3]
    Length: 3, dtype: Float32

    String aliases for the dtypes are also available. They are capitalized.

    >>> pd.array([0.1, None, 0.3], dtype="Float32")
    <FloatingArray>
    [0.1, <NA>, 0.3]
    Length: 3, dtype: Float32
    """

    _dtype_cls = FloatingDtype

    # The value used to fill '_data' to avoid upcasting
    _internal_fill_value = np.nan
    # Fill values used for any/all
    # Incompatible types in assignment (expression has type "float", base class
    # "BaseMaskedArray" defined the type as "<typing special form>")
    _truthy_value = 1.0  # type: ignore[assignment]
    _falsey_value = 0.0  # type: ignore[assignment]


_dtype_docstring = """
An ExtensionDtype for {dtype} data.

This dtype uses ``pd.NA`` as missing value indicator.

Attributes
----------
None

Methods
-------
None

Examples
--------
For Float32Dtype:

>>> ser = pd.Series([2.25, pd.NA], dtype=pd.Float32Dtype())
>>> ser.dtype
Float32Dtype()

For Float64Dtype:

>>> ser = pd.Series([2.25, pd.NA], dtype=pd.Float64Dtype())
>>> ser.dtype
Float64Dtype()
"""

# create the Dtype


@register_extension_dtype
class Float32Dtype(FloatingDtype):
    type = np.float32
    name: ClassVar[str] = "Float32"
    __doc__ = _dtype_docstring.format(dtype="float32")


@register_extension_dtype
class Float64Dtype(FloatingDtype):
    type = np.float64
    name: ClassVar[str] = "Float64"
    __doc__ = _dtype_docstring.format(dtype="float64")


NUMPY_FLOAT_TO_DTYPE: dict[np.dtype, FloatingDtype] = {
    np.dtype(np.float32): Float32Dtype(),
    np.dtype(np.float64): Float64Dtype(),
}
<|endoftext|>from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np

from pandas._libs import lib
from pandas.errors import LossySetitemError

from pandas.core.dtypes.cast import np_can_hold_element
from pandas.core.dtypes.common import is_numeric_dtype

if TYPE_CHECKING:
    from pandas._typing import (
        ArrayLike,
        npt,
    )


def to_numpy_dtype_inference(
    arr: ArrayLike, dtype: npt.DTypeLike | None, na_value, hasna: bool
) -> tuple[npt.DTypeLike, Any]:
    if dtype is None and is_numeric_dtype(arr.dtype):
        dtype_given = False
        if hasna:
            if arr.dtype.kind == "b":
                dtype = np.dtype(np.object_)
            else:
                if arr.dtype.kind in "iu":
                    dtype = np.dtype(np.float64)
                else:
                    dtype = arr.dtype.numpy_dtype  # type: ignore[union-attr]
                if na_value is lib.no_default:
                    na_value = np.nan
        else:
            dtype = arr.dtype.numpy_dtype  # type: ignore[union-attr]
    elif dtype is not None:
        dtype = np.dtype(dtype)
        dtype_given = True
    else:
        dtype_given = True

    if na_value is lib.no_default:
        if dtype is None or not hasna:
            na_value = arr.dtype.na_value
        elif dtype.kind == "f":  # type: ignore[union-attr]
            na_value = np.nan
        elif dtype.kind == "M":  # type: ignore[union-attr]
            na_value = np.datetime64("nat")
        elif dtype.kind == "m":  # type: ignore[union-attr]
            na_value = np.timedelta64("nat")
        else:
            na_value = arr.dtype.na_value

    if not dtype_given and hasna:
        try:
            np_can_hold_element(dtype, na_value)  # type: ignore[arg-type]
        except LossySetitemError:
            dtype = np.dtype(np.object_)
    return dtype, na_value
<|endoftext|>from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
    tzinfo,
)
from typing import (
    TYPE_CHECKING,
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
    BaseOffset,
    NaT,
    NaTType,
    Resolution,
    Timestamp,
    astype_overflowsafe,
    fields,
    get_resolution,
    get_supported_dtype,
    get_unit_from_dtype,
    ints_to_pydatetime,
    is_date_array_normalized,
    is_supported_dtype,
    is_unitless,
    normalize_i8_timestamps,
    timezones,
    to_offset,
    tz_convert_from_utc,
    tzconversion,
)
from pandas._libs.tslibs.dtypes import abbrev_to_npy_unit
from pandas.errors import PerformanceWarning
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import validate_inclusive

from pandas.core.dtypes.common import (
    DT64NS_DTYPE,
    INT64_DTYPE,
    is_bool_dtype,
    is_float_dtype,
    is_string_dtype,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import (
    DatetimeTZDtype,
    ExtensionDtype,
    PeriodDtype,
)
from pandas.core.dtypes.missing import isna

from pandas.core.arrays import datetimelike as dtl
from pandas.core.arrays._ranges import generate_regular_range
import pandas.core.common as com

from pandas.tseries.frequencies import get_period_alias
from pandas.tseries.offsets import (
    Day,
    Tick,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pandas._typing import (
        ArrayLike,
        DateTimeErrorChoices,
        DtypeObj,
        IntervalClosedType,
        Self,
        TimeAmbiguous,
        TimeNonexistent,
        npt,
    )

    from pandas import DataFrame
    from pandas.core.arrays import PeriodArray


_ITER_CHUNKSIZE = 10_000


@overload
def tz_to_dtype(tz: tzinfo, unit: str = ...) -> DatetimeTZDtype:
    ...


@overload
def tz_to_dtype(tz: None, unit: str = ...) -> np.dtype[np.datetime64]:
    ...


def tz_to_dtype(
    tz: tzinfo | None, unit: str = "ns"
) -> np.dtype[np.datetime64] | DatetimeTZDtype:
    """
    Return a datetime64[ns] dtype appropriate for the given timezone.

    Parameters
    ----------
    tz : tzinfo or None
    unit : str, default "ns"

    Returns
    -------
    np.dtype or Datetime64TZDType
    """
    if tz is None:
        return np.dtype(f"M8[{unit}]")
    else:
        return DatetimeTZDtype(tz=tz, unit=unit)


def _field_accessor(name: str, field: str, docstring: str | None = None):
    def f(self):
        values = self._local_timestamps()

        if field in self._bool_ops:
            result: np.ndarray

            if field.endswith(("start", "end")):
                freq = self.freq
                month_kw = 12
                if freq:
                    kwds = freq.kwds
                    month_kw = kwds.get("startingMonth", kwds.get("month", 12))

                result = fields.get_start_end_field(
                    values, field, self.freqstr, month_kw, reso=self._creso
                )
            else:
                result = fields.get_date_field(values, field, reso=self._creso)

            # these return a boolean by-definition
            return result

        if field in self._object_ops:
            result = fields.get_date_name_field(values, field, reso=self._creso)
            result = self._maybe_mask_results(result, fill_value=None)

        else:
            result = fields.get_date_field(values, field, reso=self._creso)
            result = self._maybe_mask_results(
                result, fill_value=None, convert="float64"
            )

        return result

    f.__name__ = name
    f.__doc__ = docstring
    return property(f)


# error: Definition of "_concat_same_type" in base class "NDArrayBacked" is
# incompatible with definition in base class "ExtensionArray"
class DatetimeArray(dtl.TimelikeOps, dtl.DatelikeOps):  # type: ignore[misc]
    """
    Pandas ExtensionArray for tz-naive or tz-aware datetime data.

    .. warning::

       DatetimeArray is currently experimental, and its API may change
       without warning. In particular, :attr:`DatetimeArray.dtype` is
       expected to change to always be an instance of an ``ExtensionDtype``
       subclass.

    Parameters
    ----------
    values : Series, Index, DatetimeArray, ndarray
        The datetime data.

        For DatetimeArray `values` (or a Series or Index boxing one),
        `dtype` and `freq` will be extracted from `values`.

    dtype : numpy.dtype or DatetimeTZDtype
        Note that the only NumPy dtype allowed is 'datetime64[ns]'.
    freq : str or Offset, optional
        The frequency.
    copy : bool, default False
        Whether to copy the underlying array of values.

    Attributes
    ----------
    None

    Methods
    -------
    None

    Examples
    --------
    >>> pd.arrays.DatetimeArray._from_sequence(
    ...    pd.DatetimeIndex(['2023-01-01', '2023-01-02'], freq='D'))
    <DatetimeArray>
    ['2023-01-01 00:00:00', '2023-01-02 00:00:00']
    Length: 2, dtype: datetime64[ns]
    """

    _typ = "datetimearray"
    _internal_fill_value = np.datetime64("NaT", "ns")
    _recognized_scalars = (datetime, np.datetime64)
    _is_recognized_dtype = lambda x: lib.is_np_dtype(x, "M") or isinstance(
        x, DatetimeTZDtype
    )
    _infer_matches = ("datetime", "datetime64", "date")

    @property
    def _scalar_type(self) -> type[Timestamp]:
        return Timestamp

    # define my properties & methods for delegation
    _bool_ops: list[str] = [
        "is_month_start",
        "is_month_end",
        "is_quarter_start",
        "is_quarter_end",
        "is_year_start",
        "is_year_end",
        "is_leap_year",
    ]
    _object_ops: list[str] = ["freq", "tz"]
    _field_ops: list[str] = [
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "weekday",
        "dayofweek",
        "day_of_week",
        "dayofyear",
        "day_of_year",
        "quarter",
        "days_in_month",
        "daysinmonth",
        "microsecond",
        "nanosecond",
    ]
    _other_ops: list[str] = ["date", "time", "timetz"]
    _datetimelike_ops: list[str] = (
        _field_ops + _object_ops + _bool_ops + _other_ops + ["unit"]
    )
    _datetimelike_methods: list[str] = [
        "to_period",
        "tz_localize",
        "tz_convert",
        "normalize",
        "strftime",
        "round",
        "floor",
        "ceil",
        "month_name",
        "day_name",
        "as_unit",
    ]

    # ndim is inherited from ExtensionArray, must exist to ensure
    #  Timestamp.__richcmp__(DateTimeArray) operates pointwise

    # ensure that operations with numpy arrays defer to our implementation
    __array_priority__ = 1000

    # -----------------------------------------------------------------
    # Constructors

    _dtype: np.dtype[np.datetime64] | DatetimeTZDtype
    _freq: BaseOffset | None = None
    _default_dtype = DT64NS_DTYPE  # used in TimeLikeOps.__init__

    @classmethod
    def _from_scalars(cls, scalars, *, dtype: DtypeObj) -> Self:
        if lib.infer_dtype(scalars, skipna=True) not in ["datetime", "datetime64"]:
            # TODO: require any NAs be valid-for-DTA
            # TODO: if dtype is passed, check for tzawareness compat?
            raise ValueError
        return cls._from_sequence(scalars, dtype=dtype)

    @classmethod
    def _validate_dtype(cls, values, dtype):
        # used in TimeLikeOps.__init__
        dtype = _validate_dt64_dtype(dtype)
        _validate_dt64_dtype(values.dtype)
        if isinstance(dtype, np.dtype):
            if values.dtype != dtype:
                raise ValueError("Values resolution does not match dtype.")
        else:
            vunit = np.datetime_data(values.dtype)[0]
            if vunit != dtype.unit:
                raise ValueError("Values resolution does not match dtype.")
        return dtype

    # error: Signature of "_simple_new" incompatible with supertype "NDArrayBacked"
    @classmethod
    def _simple_new(  # type: ignore[override]
        cls,
        values: npt.NDArray[np.datetime64],
        freq: BaseOffset | None = None,
        dtype: np.dtype[np.datetime64] | DatetimeTZDtype = DT64NS_DTYPE,
    ) -> Self:
        assert isinstance(values, np.ndarray)
        assert dtype.kind == "M"
        if isinstance(dtype, np.dtype):
            assert dtype == values.dtype
            assert not is_unitless(dtype)
        else:
            # DatetimeTZDtype. If we have e.g. DatetimeTZDtype[us, UTC],
            #  then values.dtype should be M8[us].
            assert dtype._creso == get_unit_from_dtype(values.dtype)

        result = super()._simple_new(values, dtype)
        result._freq = freq
        return result

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy: bool = False):
        return cls._from_sequence_not_strict(scalars, dtype=dtype, copy=copy)

    @classmethod
    def _from_sequence_not_strict(
        cls,
        data,
        *,
        dtype=None,
        copy: bool = False,
        tz=lib.no_default,
        freq: str | BaseOffset | lib.NoDefault | None = lib.no_default,
        dayfirst: bool = False,
        yearfirst: bool = False,
        ambiguous: TimeAmbiguous = "raise",
    ) -> Self:
        """
        A non-strict version of _from_sequence, called from DatetimeIndex.__new__.
        """

        # if the user either explicitly passes tz=None or a tz-naive dtype, we
        #  disallows inferring a tz.
        explicit_tz_none = tz is None
        if tz is lib.no_default:
            tz = None
        else:
            tz = timezones.maybe_get_tz(tz)

        dtype = _validate_dt64_dtype(dtype)
        # if dtype has an embedded tz, capture it
        tz = _validate_tz_from_dtype(dtype, tz, explicit_tz_none)

        unit = None
        if dtype is not None:
            unit = dtl.dtype_to_unit(dtype)

        data, copy = dtl.ensure_arraylike_for_datetimelike(
            data, copy, cls_name="DatetimeArray"
        )
        inferred_freq = None
        if isinstance(data, DatetimeArray):
            inferred_freq = data.freq

        subarr, tz = _sequence_to_dt64(
            data,
            copy=copy,
            tz=tz,
            dayfirst=dayfirst,
            yearfirst=yearfirst,
            ambiguous=ambiguous,
            out_unit=unit,
        )
        # We have to call this again after possibly inferring a tz above
        _validate_tz_from_dtype(dtype, tz, explicit_tz_none)
        if tz is not None and explicit_tz_none:
            raise ValueError(
                "Passed data is timezone-aware, incompatible with 'tz=None'. "
                "Use obj.tz_localize(None) instead."
            )

        data_unit = np.datetime_data(subarr.dtype)[0]
        data_dtype = tz_to_dtype(tz, data_unit)
        result = cls._simple_new(subarr, freq=inferred_freq, dtype=data_dtype)
        if unit is not None and unit != result.unit:
            # If unit was specified in user-passed dtype, cast to it here
            result = result.as_unit(unit)

        validate_kwds = {"ambiguous": ambiguous}
        result._maybe_pin_freq(freq, validate_kwds)
        return result

    @classmethod
    def _generate_range(
        cls,
        start,
        end,
        periods: int | None,
        freq,
        tz=None,
        normalize: bool = False,
        ambiguous: TimeAmbiguous = "raise",
        nonexistent: TimeNonexistent = "raise",
        inclusive: IntervalClosedType = "both",
        *,
        unit: str | None = None,
    ) -> Self:
        periods = dtl.validate_periods(periods)
        if freq is None and any(x is None for x in [periods, start, end]):
            raise ValueError("Must provide freq argument if no data is supplied")

        if com.count_not_none(start, end, periods, freq) != 3:
            raise ValueError(
                "Of the four parameters: start, end, periods, "
                "and freq, exactly three must be specified"
            )
        freq = to_offset(freq)

        if start is not None:
            start = Timestamp(start)

        if end is not None:
            end = Timestamp(end)

        if start is NaT or end is NaT:
            raise ValueError("Neither `start` nor `end` can be NaT")

        if unit is not None:
            if unit not in ["s", "ms", "us", "ns"]:
                raise ValueError("'unit' must be one of 's', 'ms', 'us', 'ns'")
        else:
            unit = "ns"

        if start is not None:
            start = start.as_unit(unit, round_ok=False)
        if end is not None:
            end = end.as_unit(unit, round_ok=False)

        left_inclusive, right_inclusive = validate_inclusive(inclusive)
        start, end = _maybe_normalize_endpoints(start, end, normalize)
        tz = _infer_tz_from_endpoints(start, end, tz)

        if tz is not None:
            # Localize the start and end arguments
            start = _maybe_localize_point(start, freq, tz, ambiguous, nonexistent)
            end = _maybe_localize_point(end, freq, tz, ambiguous, nonexistent)

        if freq is not None:
            # We break Day arithmetic (fixed 24 hour) here and opt for
            # Day to mean calendar day (23/24/25 hour). Therefore, strip
            # tz info from start and day to avoid DST arithmetic
            if isinstance(freq, Day):
                if start is not None:
                    start = start.tz_localize(None)
                if end is not None:
                    end = end.tz_localize(None)

            if isinstance(freq, Tick):
                i8values = generate_regular_range(start, end, periods, freq, unit=unit)
            else:
                xdr = _generate_range(
                    start=start, end=end, periods=periods, offset=freq, unit=unit
                )
                i8values = np.array([x._value for x in xdr], dtype=np.int64)

            endpoint_tz = start.tz if start is not None else end.tz

            if tz is not None and endpoint_tz is None:
                if not timezones.is_utc(tz):
                    # short-circuit tz_localize_to_utc which would make
                    #  an unnecessary copy with UTC but be a no-op.
                    creso = abbrev_to_npy_unit(unit)
                    i8values = tzconversion.tz_localize_to_utc(
                        i8values,
                        tz,
                        ambiguous=ambiguous,
                        nonexistent=nonexistent,
                        creso=creso,
                    )

                # i8values is localized datetime64 array -> have to convert
                # start/end as well to compare
                if start is not None:
                    start = start.tz_localize(tz, ambiguous, nonexistent)
                if end is not None:
                    end = end.tz_localize(tz, ambiguous, nonexistent)
        else:
            # Create a linearly spaced date_range in local time
            # Nanosecond-granularity timestamps aren't always correctly
            # representable with doubles, so we limit the range that we
            # pass to np.linspace as much as possible
            periods = cast(int, periods)
            i8values = (
                np.linspace(0, end._value - start._value, periods, dtype="int64")
                + start._value
            )
            if i8values.dtype != "i8":
                # 2022-01-09 I (brock) am not sure if it is possible for this
                #  to overflow and cast to e.g. f8, but if it does we need to cast
                i8values = i8values.astype("i8")

        if start == end:
            if not left_inclusive and not right_inclusive:
                i8values = i8values[1:-1]
        else:
            start_i8 = Timestamp(start)._value
            end_i8 = Timestamp(end)._value
            if not left_inclusive or not right_inclusive:
                if not left_inclusive and len(i8values) and i8values[0] == start_i8:
                    i8values = i8values[1:]
                if not right_inclusive and len(i8values) and i8values[-1] == end_i8:
                    i8values = i8values[:-1]

        dt64_values = i8values.view(f"datetime64[{unit}]")
        dtype = tz_to_dtype(tz, unit=unit)
        return cls._simple_new(dt64_values, freq=freq, dtype=dtype)

    # -----------------------------------------------------------------
    # DatetimeLike Interface

    def _unbox_scalar(self, value) -> np.datetime64:
        if not isinstance(value, self._scalar_type) and value is not NaT:
            raise ValueError("'value' should be a Timestamp.")
        self._check_compatible_with(value)
        if value is NaT:
            return np.datetime64(value._value, self.unit)
        else:
            return value.as_unit(self.unit).asm8

    def _scalar_from_string(self, value) -> Timestamp | NaTType:
        return Timestamp(value, tz=self.tz)

    def _check_compatible_with(self, other) -> None:
        if other is NaT:
            return
        self._assert_tzawareness_compat(other)

    # -----------------------------------------------------------------
    # Descriptive Properties

    def _box_func(self, x: np.datetime64) -> Timestamp | NaTType:
        # GH#42228
        value = x.view("i8")
        ts = Timestamp._from_value_and_reso(value, reso=self._creso, tz=self.tz)
        return ts

    @property
    # error: Return type "Union[dtype, DatetimeTZDtype]" of "dtype"
    # incompatible with return type "ExtensionDtype" in supertype
    # "ExtensionArray"
    def dtype(self) -> np.dtype[np.datetime64] | DatetimeTZDtype:  # type: ignore[override]
        """
        The dtype for the DatetimeArray.

        .. warning::

           A future version of pandas will change dtype to never be a
           ``numpy.dtype``. Instead, :attr:`DatetimeArray.dtype` will
           always be an instance of an ``ExtensionDtype`` subclass.

        Returns
        -------
        numpy.dtype or DatetimeTZDtype
            If the values are tz-naive, then ``np.dtype('datetime64[ns]')``
            is returned.

            If the values are tz-aware, then the ``DatetimeTZDtype``
            is returned.
        """
        return self._dtype

    @property
    def tz(self) -> tzinfo | None:
        """
        Return the timezone.

        Returns
        -------
        datetime.tzinfo, pytz.tzinfo.BaseTZInfo, dateutil.tz.tz.tzfile, or None
            Returns None when the array is tz-naive.

        Examples
        --------
        For Series:

        >>> s = pd.Series(["1/1/2020 10:00:00+00:00", "2/1/2020 11:00:00+00:00"])
        >>> s = pd.to_datetime(s)
        >>> s
        0   2020-01-01 10:00:00+00:00
        1   2020-02-01 11:00:00+00:00
        dtype: datetime64[ns, UTC]
        >>> s.dt.tz
        datetime.timezone.utc

        For DatetimeIndex:

        >>> idx = pd.DatetimeIndex(["1/1/2020 10:00:00+00:00",
        ...                         "2/1/2020 11:00:00+00:00"])
        >>> idx.tz
        datetime.timezone.utc
        """
        # GH 18595
        return getattr(self.dtype, "tz", None)

    @tz.setter
    def tz(self, value):
        # GH 3746: Prevent localizing or converting the index by setting tz
        raise AttributeError(
            "Cannot directly set timezone. Use tz_localize() "
            "or tz_convert() as appropriate"
        )

    @property
    def tzinfo(self) -> tzinfo | None:
        """
        Alias for tz attribute
        """
        return self.tz

    @property  # NB: override with cache_readonly in immutable subclasses
    def is_normalized(self) -> bool:
        """
        Returns True if all of the dates are at midnight ("no time")
        """
        return is_date_array_normalized(self.asi8, self.tz, reso=self._creso)

    @property  # NB: override with cache_readonly in immutable subclasses
    def _resolution_obj(self) -> Resolution:
        return get_resolution(self.asi8, self.tz, reso=self._creso)

    # ----------------------------------------------------------------
    # Array-Like / EA-Interface Methods

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        if dtype is None and self.tz:
            # The default for tz-aware is object, to preserve tz info
            dtype = object

        return super().__array__(dtype=dtype, copy=copy)

    def __iter__(self) -> Iterator:
        """
        Return an iterator over the boxed values

        Yields
        ------
        tstamp : Timestamp
        """
        if self.ndim > 1:
            for i in range(len(self)):
                yield self[i]
        else:
            # convert in chunks of 10k for efficiency
            data = self.asi8
            length = len(self)
            chunksize = _ITER_CHUNKSIZE
            chunks = (length // chunksize) + 1

            for i in range(chunks):
                start_i = i * chunksize
                end_i = min((i + 1) * chunksize, length)
                converted = ints_to_pydatetime(
                    data[start_i:end_i],
                    tz=self.tz,
                    box="timestamp",
                    reso=self._creso,
                )
                yield from converted

    def astype(self, dtype, copy: bool = True):
        # We handle
        #   --> datetime
        #   --> period
        # DatetimeLikeArrayMixin Super handles the rest.
        dtype = pandas_dtype(dtype)

        if dtype == self.dtype:
            if copy:
                return self.copy()
            return self

        elif isinstance(dtype, ExtensionDtype):
            if not isinstance(dtype, DatetimeTZDtype):
                # e.g. Sparse[datetime64[ns]]
                return super().astype(dtype, copy=copy)
            elif self.tz is None:
                # pre-2.0 this did self.tz_localize(dtype.tz), which did not match
                #  the Series behavior which did
                #  values.tz_localize("UTC").tz_convert(dtype.tz)
                raise TypeError(
                    "Cannot use .astype to convert from timezone-naive dtype to "
                    "timezone-aware dtype. Use obj.tz_localize instead or "
                    "series.dt.tz_localize instead"
                )
            else:
                # tzaware unit conversion e.g. datetime64[s, UTC]
                np_dtype = np.dtype(dtype.str)
                res_values = astype_overflowsafe(self._ndarray, np_dtype, copy=copy)
                return type(self)._simple_new(res_values, dtype=dtype, freq=self.freq)

        elif (
            self.tz is None
            and lib.is_np_dtype(dtype, "M")
            and not is_unitless(dtype)
            and is_supported_dtype(dtype)
        ):
            # unit conversion e.g. datetime64[s]
            res_values = astype_overflowsafe(self._ndarray, dtype, copy=True)
            return type(self)._simple_new(res_values, dtype=res_values.dtype)
            # TODO: preserve freq?

        elif self.tz is not None and lib.is_np_dtype(dtype, "M"):
            # pre-2.0 behavior for DTA/DTI was
            #  values.tz_convert("UTC").tz_localize(None), which did not match
            #  the Series behavior
            raise TypeError(
                "Cannot use .astype to convert from timezone-aware dtype to "
                "timezone-naive dtype. Use obj.tz_localize(None) or "
                "obj.tz_convert('UTC').tz_localize(None) instead."
            )

        elif (
            self.tz is None
            and lib.is_np_dtype(dtype, "M")
            and dtype != self.dtype
            and is_unitless(dtype)
        ):
            raise TypeError(
                "Casting to unit-less dtype 'datetime64' is not supported. "
                "Pass e.g. 'datetime64[ns]' instead."
            )

        elif isinstance(dtype, PeriodDtype):
            return self.to_period(freq=dtype.freq)
        return dtl.DatetimeLikeArrayMixin.astype(self, dtype, copy)

    # -----------------------------------------------------------------
    # Rendering Methods

    def _format_native_types(
        self, *, na_rep: str | float = "NaT", date_format=None, **kwargs
    ) -> npt.NDArray[np.object_]:
        if date_format is None and self._is_dates_only:
            # Only dates and no timezone: provide a default format
            date_format = "%Y-%m-%d"

        return tslib.format_array_from_datetime(
            self.asi8, tz=self.tz, format=date_format, na_rep=na_rep, reso=self._creso
        )

    # -----------------------------------------------------------------
    # Comparison Methods

    def _has_same_tz(self, other) -> bool:
        # vzone shouldn't be None if value is non-datetime like
        if isinstance(other, np.datetime64):
            # convert to Timestamp as np.datetime64 doesn't have tz attr
            other = Timestamp(other)

        if not hasattr(other, "tzinfo"):
            return False
        other_tz = other.tzinfo
        return timezones.tz_compare(self.tzinfo, other_tz)

    def _assert_tzawareness_compat(self, other) -> None:
        # adapted from _Timestamp._assert_tzawareness_compat
        other_tz = getattr(other, "tzinfo", None)
        other_dtype = getattr(other, "dtype", None)

        if isinstance(other_dtype, DatetimeTZDtype):
            # Get tzinfo from Series dtype
            other_tz = other.dtype.tz
        if other is NaT:
            # pd.NaT quacks both aware and naive
            pass
        elif self.tz is None:
            if other_tz is not None:
                raise TypeError(
                    "Cannot compare tz-naive and tz-aware datetime-like objects."
                )
        elif other_tz is None:
            raise TypeError(
                "Cannot compare tz-naive and tz-aware datetime-like objects"
            )

    # -----------------------------------------------------------------
    # Arithmetic Methods

    def _add_offset(self, offset: BaseOffset) -> Self:
        assert not isinstance(offset, Tick)

        if self.tz is not None:
            values = self.tz_localize(None)
        else:
            values = self

        try:
            res_values = offset._apply_array(values._ndarray)
            if res_values.dtype.kind == "i":
                # error: Argument 1 to "view" of "ndarray" has incompatible type
                # "dtype[datetime64] | DatetimeTZDtype"; expected
                # "dtype[Any] | type[Any] | _SupportsDType[dtype[Any]]"
                res_values = res_values.view(values.dtype)  # type: ignore[arg-type]
        except NotImplementedError:
            warnings.warn(
                "Non-vectorized DateOffset being applied to Series or DatetimeIndex.",
                PerformanceWarning,
                stacklevel=find_stack_level(),
            )
            res_values = self.astype("O") + offset
            # TODO(GH#55564): as_unit will be unnecessary
            result = type(self)._from_sequence(res_values).as_unit(self.unit)
            if not len(self):
                # GH#30336 _from_sequence won't be able to infer self.tz
                return result.tz_localize(self.tz)

        else:
            result = type(self)._simple_new(res_values, dtype=res_values.dtype)
            if offset.normalize:
                result = result.normalize()
                result._freq = None

            if self.tz is not None:
                result = result.tz_localize(self.tz)

        return result

    # -----------------------------------------------------------------
    # Timezone Conversion and Localization Methods

    def _local_timestamps(self) -> npt.NDArray[np.int64]:
        """
        Convert to an i8 (unix-like nanosecond timestamp) representation
        while keeping the local timezone and not using UTC.
        This is used to calculate time-of-day information as if the timestamps
        were timezone-naive.
        """
        if self.tz is None or timezones.is_utc(self.tz):
            # Avoid the copy that would be made in tzconversion
            return self.asi8
        return tz_convert_from_utc(self.asi8, self.tz, reso=self._creso)

    def tz_convert(self, tz) -> Self:
        """
        Convert tz-aware Datetime Array/Index from one time zone to another.

        Parameters
        ----------
        tz : str, pytz.timezone, dateutil.tz.tzfile, datetime.tzinfo or None
            Time zone for time. Corresponding timestamps would be converted
            to this time zone of the Datetime Array/Index. A `tz` of None will
            convert to UTC and remove the timezone information.

        Returns
        -------
        Array or Index

        Raises
        ------
        TypeError
            If Datetime Array/Index is tz-naive.

        See Also
        --------
        DatetimeIndex.tz : A timezone that has a variable offset from UTC.
        DatetimeIndex.tz_localize : Localize tz-naive DatetimeIndex to a
            given time zone, or remove timezone from a tz-aware DatetimeIndex.

        Examples
        --------
        With the `tz` parameter, we can change the DatetimeIndex
        to other time zones:

        >>> dti = pd.date_range(start='2014-08-01 09:00',
        ...                     freq='h', periods=3, tz='Europe/Berlin')

        >>> dti
        DatetimeIndex(['2014-08-01 09:00:00+02:00',
                       '2014-08-01 10:00:00+02:00',
                       '2014-08-01 11:00:00+02:00'],
                      dtype='datetime64[ns, Europe/Berlin]', freq='h')

        >>> dti.tz_convert('US/Central')
        DatetimeIndex(['2014-08-01 02:00:00-05:00',
                       '2014-08-01 03:00:00-05:00',
                       '2014-08-01 04:00:00-05:00'],
                      dtype='datetime64[ns, US/Central]', freq='h')

        With the ``tz=None``, we can remove the timezone (after converting
        to UTC if necessary):

        >>> dti = pd.date_range(start='2014-08-01 09:00', freq='h',
        ...                     periods=3, tz='Europe/Berlin')

        >>> dti
        DatetimeIndex(['2014-08-01 09:00:00+02:00',
                       '2014-08-01 10:00:00+02:00',
                       '2014-08-01 11:00:00+02:00'],
                        dtype='datetime64[ns, Europe/Berlin]', freq='h')

        >>> dti.tz_convert(None)
        DatetimeIndex(['2014-08-01 07:00:00',
                       '2014-08-01 08:00:00',
                       '2014-08-01 09:00:00'],
                        dtype='datetime64[ns]', freq='h')
        """
        tz = timezones.maybe_get_tz(tz)

        if self.tz is None:
            # tz naive, use tz_localize
            raise TypeError(
                "Cannot convert tz-naive timestamps, use tz_localize to localize"
            )

        # No conversion since timestamps are all UTC to begin with
        dtype = tz_to_dtype(tz, unit=self.unit)
        return self._simple_new(self._ndarray, dtype=dtype, freq=self.freq)

    @dtl.ravel_compat
    def tz_localize(
        self,
        tz,
        ambiguous: TimeAmbiguous = "raise",
        nonexistent: TimeNonexistent = "raise",
    ) -> Self:
        """
        Localize tz-naive Datetime Array/Index to tz-aware Datetime Array/Index.

        This method takes a time zone (tz) naive Datetime Array/Index object
        and makes this time zone aware. It does not move the time to another
        time zone.

        This method can also be used to do the inverse -- to create a time
        zone unaware object from an aware object. To that end, pass `tz=None`.

        Parameters
        ----------
        tz : str, pytz.timezone, dateutil.tz.tzfile, datetime.tzinfo or None
            Time zone to convert timestamps to. Passing ``None`` will
            remove the time zone information preserving local time.
        ambiguous : 'infer', 'NaT', bool array, default 'raise'
            When clocks moved backward due to DST, ambiguous times may arise.
            For example in Central European Time (UTC+01), when going from
            03:00 DST to 02:00 non-DST, 02:30:00 local time occurs both at
            00:30:00 UTC and at 01:30:00 UTC. In such a situation, the
            `ambiguous` parameter dictates how ambiguous times should be
            handled.

            - 'infer' will attempt to infer fall dst-transition hours based on
              order
            - bool-ndarray where True signifies a DST time, False signifies a
              non-DST time (note that this flag is only applicable for
              ambiguous times)
            - 'NaT' will return NaT where there are ambiguous times
            - 'raise' will raise an AmbiguousTimeError if there are ambiguous
              times.

        nonexistent : 'shift_forward', 'shift_backward, 'NaT', timedelta, \
default 'raise'
            A nonexistent time does not exist in a particular timezone
            where clocks moved forward due to DST.

            - 'shift_forward' will shift the nonexistent time forward to the
              closest existing time
            - 'shift_backward' will shift the nonexistent time backward to the
              closest existing time
            - 'NaT' will return NaT where there are nonexistent times
            - timedelta objects will shift nonexistent times by the timedelta
            - 'raise' will raise an NonExistentTimeError if there are
              nonexistent times.

        Returns
        -------
        Same type as self
            Array/Index converted to the specified time zone.

        Raises
        ------
        TypeError
            If the Datetime Array/Index is tz-aware and tz is not None.

        See Also
        --------
        DatetimeIndex.tz_convert : Convert tz-aware DatetimeIndex from
            one time zone to another.

        Examples
        --------
        >>> tz_naive = pd.date_range('2018-03-01 09:00', periods=3)
        >>> tz_naive
        DatetimeIndex(['2018-03-01 09:00:00', '2018-03-02 09:00:00',
                       '2018-03-03 09:00:00'],
                      dtype='datetime64[ns]', freq='D')

        Localize DatetimeIndex in US/Eastern time zone:

        >>> tz_aware = tz_naive.tz_localize(tz='US/Eastern')
        >>> tz_aware
        DatetimeIndex(['2018-03-01 09:00:00-05:00',
                       '2018-03-02 09:00:00-05:00',
                       '2018-03-03 09:00:00-05:00'],
                      dtype='datetime64[ns, US/Eastern]', freq=None)

        With the ``tz=None``, we can remove the time zone information
        while keeping the local time (not converted to UTC):

        >>> tz_aware.tz_localize(None)
        DatetimeIndex(['2018-03-01 09:00:00', '2018-03-02 09:00:00',
                       '2018-03-03 09:00:00'],
                      dtype='datetime64[ns]', freq=None)

        Be careful with DST changes. When there is sequential data, pandas can
        infer the DST time:

        >>> s = pd.to_datetime(pd.Series(['2018-10-28 01:30:00',
        ...                               '2018-10-28 02:00:00',
        ...                               '2018-10-28 02:30:00',
        ...                               '2018-10-28 02:00:00',
        ...                               '2018-10-28 02:30:00',
        ...                               '2018-10-28 03:00:00',
        ...                               '2018-10-28 03:30:00']))
        >>> s.dt.tz_localize('CET', ambiguous='infer')
        0   2018-10-28 01:30:00+02:00
        1   2018-10-28 02:00:00+02:00
        2   2018-10-28 02:30:00+02:00
        3   2018-10-28 02:00:00+01:00
        4   2018-10-28 02:30:00+01:00
        5   2018-10-28 03:00:00+01:00
        6   2018-10-28 03:30:00+01:00
        dtype: datetime64[ns, CET]

        In some cases, inferring the DST is impossible. In such cases, you can
        pass an ndarray to the ambiguous parameter to set the DST explicitly

        >>> s = pd.to_datetime(pd.Series(['2018-10-28 01:20:00',
        ...                               '2018-10-28 02:36:00',
        ...                               '2018-10-28 03:46:00']))
        >>> s.dt.tz_localize('CET', ambiguous=np.array([True, True, False]))
        0   2018-10-28 01:20:00+02:00
        1   2018-10-28 02:36:00+02:00
        2   2018-10-28 03:46:00+01:00
        dtype: datetime64[ns, CET]

        If the DST transition causes nonexistent times, you can shift these
        dates forward or backwards with a timedelta object or `'shift_forward'`
        or `'shift_backwards'`.

        >>> s = pd.to_datetime(pd.Series(['2015-03-29 02:30:00',
        ...                               '2015-03-29 03:30:00']))
        >>> s.dt.tz_localize('Europe/Warsaw', nonexistent='shift_forward')
        0   2015-03-29 03:00:00+02:00
        1   2015-03-29 03:30:00+02:00
        dtype: datetime64[ns, Europe/Warsaw]

        >>> s.dt.tz_localize('Europe/Warsaw', nonexistent='shift_backward')
        0   2015-03-29 01:59:59.999999999+01:00
        1   2015-03-29 03:30:00+02:00
        dtype: datetime64[ns, Europe/Warsaw]

        >>> s.dt.tz_localize('Europe/Warsaw', nonexistent=pd.Timedelta('1h'))
        0   2015-03-29 03:30:00+02:00
        1   2015-03-29 03:30:00+02:00
        dtype: datetime64[ns, Europe/Warsaw]
        """
        nonexistent_options = ("raise", "NaT", "shift_forward", "shift_backward")
        if nonexistent not in nonexistent_options and not isinstance(
            nonexistent, timedelta
        ):
            raise ValueError(
                "The nonexistent argument must be one of 'raise', "
                "'NaT', 'shift_forward', 'shift_backward' or "
                "a timedelta object"
            )

        if self.tz is not None:
            if tz is None:
                new_dates = tz_convert_from_utc(self.asi8, self.tz, reso=self._creso)
            else:
                raise TypeError("Already tz-aware, use tz_convert to convert.")
        else:
            tz = timezones.maybe_get_tz(tz)
            # Convert to UTC

            new_dates = tzconversion.tz_localize_to_utc(
                self.asi8,
                tz,
                ambiguous=ambiguous,
                nonexistent=nonexistent,
                creso=self._creso,
            )
        new_dates_dt64 = new_dates.view(f"M8[{self.unit}]")
        dtype = tz_to_dtype(tz, unit=self.unit)

        freq = None
        if timezones.is_utc(tz) or (len(self) == 1 and not isna(new_dates_dt64[0])):
            # we can preserve freq
            # TODO: Also for fixed-offsets
            freq = self.freq
        elif tz is None and self.tz is None:
            # no-op
            freq = self.freq
        return self._simple_new(new_dates_dt64, dtype=dtype, freq=freq)

    # ----------------------------------------------------------------
    # Conversion Methods - Vectorized analogues of Timestamp methods

    def to_pydatetime(self) -> npt.NDArray[np.object_]:
        """
        Return an ndarray of ``datetime.datetime`` objects.

        Returns
        -------
        numpy.ndarray

        Examples
        --------
        >>> idx = pd.date_range('2018-02-27', periods=3)
        >>> idx.to_pydatetime()
        array([datetime.datetime(2018, 2, 27, 0, 0),
               datetime.datetime(2018, 2, 28, 0, 0),
               datetime.datetime(2018, 3, 1, 0, 0)], dtype=object)
        """
        return ints_to_pydatetime(self.asi8, tz=self.tz, reso=self._creso)

    def normalize(self) -> Self:
        """
        Convert times to midnight.

        The time component of the date-time is converted to midnight i.e.
        00:00:00. This is useful in cases, when the time does not matter.
        Length is unaltered. The timezones are unaffected.

        This method is available on Series with datetime values under
        the ``.dt`` accessor, and directly on Datetime Array/Index.

        Returns
        -------
        DatetimeArray, DatetimeIndex or Series
            The same type as the original data. Series will have the same
            name and index. DatetimeIndex will have the same name.

        See Also
        --------
        floor : Floor the datetimes to the specified freq.
        ceil : Ceil the datetimes to the specified freq.
        round : Round the datetimes to the specified freq.

        Examples
        --------
        >>> idx = pd.date_range(start='2014-08-01 10:00', freq='h',
        ...                     periods=3, tz='Asia/Calcutta')
        >>> idx
        DatetimeIndex(['2014-08-01 10:00:00+05:30',
                       '2014-08-01 11:00:00+05:30',
                       '2014-08-01 12:00:00+05:30'],
                        dtype='datetime64[ns, Asia/Calcutta]', freq='h')
        >>> idx.normalize()
        DatetimeIndex(['2014-08-01 00:00:00+05:30',
                       '2014-08-01 00:00:00+05:30',
                       '2014-08-01 00:00:00+05:30'],
                       dtype='datetime64[ns, Asia/Calcutta]', freq=None)
        """
        new_values = normalize_i8_timestamps(self.asi8, self.tz, reso=self._creso)
        dt64_values = new_values.view(self._ndarray.dtype)

        dta = type(self)._simple_new(dt64_values, dtype=dt64_values.dtype)
        dta = dta._with_freq("infer")
        if self.tz is not None:
            dta = dta.tz_localize(self.tz)
        return dta

    def to_period(self, freq=None) -> PeriodArray:
        """
        Cast to PeriodArray/PeriodIndex at a particular frequency.

        Converts DatetimeArray/Index to PeriodArray/PeriodIndex.

        Parameters
        ----------
        freq : str or Period, optional
            One of pandas' :ref:`period aliases <timeseries.period_aliases>`
            or an Period object. Will be inferred by default.

        Returns
        -------
        PeriodArray/PeriodIndex

        Raises
        ------
        ValueError
            When converting a DatetimeArray/Index with non-regular values,
            so that a frequency cannot be inferred.

        See Also
        --------
        PeriodIndex: Immutable ndarray holding ordinal values.
        DatetimeIndex.to_pydatetime: Return DatetimeIndex as object.

        Examples
        --------
        >>> df = pd.DataFrame({"y": [1, 2, 3]},
        ...                   index=pd.to_datetime(["2000-03-31 00:00:00",
        ...                                         "2000-05-31 00:00:00",
        ...                                         "2000-08-31 00:00:00"]))
        >>> df.index.to_period("M")
        PeriodIndex(['2000-03', '2000-05', '2000-08'],
                    dtype='period[M]')

        Infer the daily frequency

        >>> idx = pd.date_range("2017-01-01", periods=2)
        >>> idx.to_period()
        PeriodIndex(['2017-01-01', '2017-01-02'],
                    dtype='period[D]')
        """
        from pandas.core.arrays import PeriodArray

        if self.tz is not None:
            warnings.warn(
                "Converting to PeriodArray/Index representation "
                "will drop timezone information.",
                UserWarning,
                stacklevel=find_stack_level(),
            )

        if freq is None:
            freq = self.freqstr or self.inferred_freq
            if isinstance(self.freq, BaseOffset) and hasattr(
                self.freq, "_period_dtype_code"
            ):
                freq = PeriodDtype(self.freq)._freqstr

            if freq is None:
                raise ValueError(
                    "You must pass a freq argument as current index has none."
                )

            res = get_period_alias(freq)

            #  https://github.com/pandas-dev/pandas/issues/33358
            if res is None:
                res = freq

            freq = res
        return PeriodArray._from_datetime64(self._ndarray, freq, tz=self.tz)

    # -----------------------------------------------------------------
    # Properties - Vectorized Timestamp Properties/Methods

    def month_name(self, locale=None) -> npt.NDArray[np.object_]:
        """
        Return the month names with specified locale.

        Parameters
        ----------
        locale : str, optional
            Locale determining the language in which to return the month name.
            Default is English locale (``'en_US.utf8'``). Use the command
            ``locale -a`` on your terminal on Unix systems to find your locale
            language code.

        Returns
        -------
        Series or Index
            Series or Index of month names.

        Examples
        --------
        >>> s = pd.Series(pd.date_range(start='2018-01', freq='ME', periods=3))
        >>> s
        0   2018-01-31
        1   2018-02-28
        2   2018-03-31
        dtype: datetime64[ns]
        >>> s.dt.month_name()
        0     January
        1    February
        2       March
        dtype: object

        >>> idx = pd.date_range(start='2018-01', freq='ME', periods=3)
        >>> idx
        DatetimeIndex(['2018-01-31', '2018-02-28', '2018-03-31'],
                      dtype='datetime64[ns]', freq='ME')
        >>> idx.month_name()
        Index(['January', 'February', 'March'], dtype='object')

        Using the ``locale`` parameter you can set a different locale language,
        for example: ``idx.month_name(locale='pt_BR.utf8')`` will return month
        names in Brazilian Portuguese language.

        >>> idx = pd.date_range(start='2018-01', freq='ME', periods=3)
        >>> idx
        DatetimeIndex(['2018-01-31', '2018-02-28', '2018-03-31'],
                      dtype='datetime64[ns]', freq='ME')
        >>> idx.month_name(locale='pt_BR.utf8')  # doctest: +SKIP
        Index(['Janeiro', 'Fevereiro', 'Março'], dtype='object')
        """
        values = self._local_timestamps()

        result = fields.get_date_name_field(
            values, "month_name", locale=locale, reso=self._creso
        )
        result = self._maybe_mask_results(result, fill_value=None)
        if using_string_dtype():
            from pandas import (
                StringDtype,
                array as pd_array,
            )

            return pd_array(result, dtype=StringDtype(na_value=np.nan))  # type: ignore[return-value]
        return result

    def day_name(self, locale=None) -> npt.NDArray[np.object_]:
        """
        Return the day names with specified locale.

        Parameters
        ----------
        locale : str, optional
            Locale determining the language in which to return the day name.
            Default is English locale (``'en_US.utf8'``). Use the command
            ``locale -a`` on your terminal on Unix systems to find your locale
            language code.

        Returns
        -------
        Series or Index
            Series or Index of day names.

        Examples
        --------
        >>> s = pd.Series(pd.date_range(start='2018-01-01', freq='D', periods=3))
        >>> s
        0   2018-01-01
        1   2018-01-02
        2   2018-01-03
        dtype: datetime64[ns]
        >>> s.dt.day_name()
        0       Monday
        1      Tuesday
        2    Wednesday
        dtype: object

        >>> idx = pd.date_range(start='2018-01-01', freq='D', periods=3)
        >>> idx
        DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03'],
                      dtype='datetime64[ns]', freq='D')
        >>> idx.day_name()
        Index(['Monday', 'Tuesday', 'Wednesday'], dtype='object')

        Using the ``locale`` parameter you can set a different locale language,
        for example: ``idx.day_name(locale='pt_BR.utf8')`` will return day
        names in Brazilian Portuguese language.

        >>> idx = pd.date_range(start='2018-01-01', freq='D', periods=3)
        >>> idx
        DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03'],
                      dtype='datetime64[ns]', freq='D')
        >>> idx.day_name(locale='pt_BR.utf8') # doctest: +SKIP
        Index(['Segunda', 'Terça', 'Quarta'], dtype='object')
        """
        values = self._local_timestamps()

        result = fields.get_date_name_field(
            values, "day_name", locale=locale, reso=self._creso
        )
        result = self._maybe_mask_results(result, fill_value=None)
        if using_string_dtype():
            # TODO: no tests that check for dtype of result as of 2024-08-15
            from pandas import (
                StringDtype,
                array as pd_array,
            )

            return pd_array(result, dtype=StringDtype(na_value=np.nan))  # type: ignore[return-value]
        return result

    @property
    def time(self) -> npt.NDArray[np.object_]:
        """
        Returns numpy array of :class:`datetime.time` objects.

        The time part of the Timestamps.

        Examples
        --------
        For Series:

        >>> s = pd.Series(["1/1/2020 10:00:00+00:00", "2/1/2020 11:00:00+00:00"])
        >>> s = pd.to_datetime(s)
        >>> s
        0   2020-01-01 10:00:00+00:00
        1   2020-02-01 11:00:00+00:00
        dtype: datetime64[ns, UTC]
        >>> s.dt.time
        0    10:00:00
        1    11:00:00
        dtype: object

        For DatetimeIndex:

        >>> idx = pd.DatetimeIndex(["1/1/2020 10:00:00+00:00",
        ...                         "2/1/2020 11:00:00+00:00"])
        >>> idx.time
        array([datetime.time(10, 0), datetime.time(11, 0)], dtype=object)
        """
        # If the Timestamps have a timezone that is not UTC,
        # convert them into their i8 representation while
        # keeping their timezone and not using UTC
        timestamps = self._local_timestamps()

        return ints_to_pydatetime(timestamps, box="time", reso=self._creso)

    @property
    def timetz(self) -> npt.NDArray[np.object_]:
        """
        Returns numpy array of :class:`datetime.time` objects with timezones.

        The time part of the Timestamps.

        Examples
        --------
        For Series:

        >>> s = pd.Series(["1/1/2020 10:00:00+00:00", "2/1/2020 11:00:00+00:00"])
        >>> s = pd.to_datetime(s)
        >>> s
        0   2020-01-01 10:00:00+00:00
        1   2020-02-01 11:00:00+00:00
        dtype: datetime64[ns, UTC]
        >>> s.dt.timetz
        0    10:00:00+00:00
        1    11:00:00+00:00
        dtype: object

        For DatetimeIndex:

        >>> idx = pd.DatetimeIndex(["1/1/2020 10:00:00+00:00",
        ...                         "2/1/2020 11:00:00+00:00"])
        >>> idx.timetz
        array([datetime.time(10, 0, tzinfo=datetime.timezone.utc),
        datetime.time(11, 0, tzinfo=datetime.timezone.utc)], dtype=object)
        """
        return ints_to_pydatetime(self.asi8, self.tz, box="time", reso=self._creso)

    @property
    def date(self) -> npt.NDArray[np.object_]:
        """
        Returns numpy array of python :class:`datetime.date` objects.

        Namely, the date part of Timestamps without time and
        timezone information.

        Examples
        --------
        For Series:

        >>> s = pd.Series(["1/1/2020 10:00:00+00:00", "2/1/2020 11:00:00+00:00"])
        >>> s = pd.to_datetime(s)
        >>> s
        0   2020-01-01 10:00:00+00:00
        1   2020-02-01 11:00:00+00:00
        dtype: datetime64[ns, UTC]
        >>> s.dt.date
        0    2020-01-01
        1    2020-02-01
        dtype: object

        For DatetimeIndex:

        >>> idx = pd.DatetimeIndex(["1/1/2020 10:00:00+00:00",
        ...                         "2/1/2020 11:00:00+00:00"])
        >>> idx.date
        array([datetime.date(2020, 1, 1), datetime.date(2020, 2, 1)], dtype=object)
        """
        # If the Timestamps have a timezone that is not UTC,
        # convert them into their i8 representation while
        # keeping their timezone and not using UTC
        timestamps = self._local_timestamps()

        return ints_to_pydatetime(timestamps, box="date", reso=self._creso)

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
        >>> idx = pd.date_range(start='2019-12-29', freq='D', periods=4)
        >>> idx.isocalendar()
                    year  week  day
        2019-12-29  2019    52    7
        2019-12-30  2020     1    1
        2019-12-31  2020     1    2
        2020-01-01  2020     1    3
        >>> idx.isocalendar().week
        2019-12-29    52
        2019-12-30     1
        2019-12-31     1
        2020-01-01     1
        Freq: D, Name: week, dtype: UInt32
        """
        from pandas import DataFrame

        values = self._local_timestamps()
        sarray = fields.build_isocalendar_sarray(values, reso=self._creso)
        iso_calendar_df = DataFrame(
            sarray, columns=["year", "week", "day"], dtype="UInt32"
        )
        if self._hasna:
            iso_calendar_df.iloc[self._isnan] = None
        return iso_calendar_df

    year = _field_accessor(
        "year",
        "Y",
        """
        The year of the datetime.

        Examples
        --------
        >>> datetime_series = pd.Series(
        ...     pd.date_range("2000-01-01", periods=3, freq="YE")
        ... )
        >>> datetime_series
        0   2000-12-31
        1   2001-12-31
        2   2002-12-31
        dtype: datetime64[ns]
        >>> datetime_series.dt.year
        0    2000
        1    2001
        2    2002
        dtype: int32
        """,
    )
    month = _field_accessor(
        "month",
        "M",
        """
        The month as January=1, December=12.

        Examples
        --------
        >>> datetime_series = pd.Series(
        ...     pd.date_range("2000-01-01", periods=3, freq="ME")
        ... )
        >>> datetime_series
        0   2000-01-31
        1   2000-02-29
        2   2000-03-31
        dtype: datetime64[ns]
        >>> datetime_series.dt.month
        0    1
        1    2
        2    3
        dtype: int32
        """,
    )
    day = _field_accessor(
        "day",
        "D",
        """
        The day of the datetime.

        Examples
        --------
        >>> datetime_series = pd.Series(
        ...     pd.date_range("2000-01-01", periods=3, freq="D")
        ... )
        >>> datetime_series
        0   2000-01-01
        1   2000-01-02
        2   2000-01-03
        dtype: datetime64[ns]
        >>> datetime_series.dt.day
        0    1
        1    2
        2    3
        dtype: int32
        """,
    )
    hour = _field_accessor(
        "hour",
        "h",
        """
        The hours of the datetime.

        Examples
        --------
        >>> datetime_series = pd.Series(
        ...     pd.date_range("2000-01-01", periods=3, freq="h")
        ... )
        >>> datetime_series
        0   2000-01-01 00:00:00
        1   2000-01-01 01:00:00
        2   2000-01-01 02:00:00
        dtype: datetime64[ns]
        >>> datetime_series.dt.hour
        0    0
        1    1
        2    2
        dtype: int32
        """,
    )
    minute = _field_accessor(
        "minute",
        "m",
        """
        The minutes of the datetime.

        Examples
        --------
        >>> datetime_series = pd.Series(
        ...     pd.date_range("2000-01-01", periods=3, freq="min")
        ... )
        >>> datetime_series
        0   2000-01-01 00:00:00
        1   2000-01-01 00:01:00
        2   2000-01-01 00:02:00
        dtype: datetime64[ns]
        >>> datetime_series.dt.minute
        0    0
        1    1
        2    2
        dtype: int32
        """,
    )
    second = _field_accessor(
        "second",
        "s",
        """
        The seconds of the datetime.

        Examples
        --------
        >>> datetime_series = pd.Series(
        ...     pd.date_range("2000-01-01", periods=3, freq="s")
        ... )
        >>> datetime_series
        0   2000-01-01 00:00:00
        1   2000-01-01 00:00:01
        2   2000-01-01 00:00:02
        dtype: datetime64[ns]
        >>> datetime_series.dt.second
        0    0
        1    1
        2    2
        dtype: int32
        """,
    )
    microsecond = _field_accessor(
        "microsecond",
        "us",
        """
        The microseconds of the datetime.

        Examples
        --------
        >>> datetime_series = pd.Series(
        ...     pd.date_range("2000-01-01", periods=3, freq="us")
        ... )
        >>> datetime_series
        0   2000-01-01 00:00:00.000000
        1   2000-01-01 00:00:00.000001
        2   2000-01-01 00:00:00.000002
        dtype: datetime64[ns]
        >>> datetime_series.dt.microsecond
        0       0
        1       1
        2       2
        dtype: int32
        """,
    )
    nanosecond = _field_accessor(
        "nanosecond",
        "ns",
        """
        The nanoseconds of the datetime.

        Examples
        --------
        >>> datetime_series = pd.Series(
        ...     pd.date_range("2000-01-01", periods=3, freq="ns")
        ... )
        >>> datetime_series
        0   2000-01-01 00:00:00.000000000
        1   2000-01-01 00:00:00.000000001
        2   2000-01-01 00:00:00.000000002
        dtype: datetime64[ns]
        >>> datetime_series.dt.nanosecond
        0       0
        1       1
        2       2
        dtype: int32
        """,
    )
    _dayofweek_doc = """
    The day of the week with Monday=0, Sunday=6.

    Return the day of the week. It is assumed the week starts on
    Monday, which is denoted by 0 and ends on Sunday which is denoted
    by 6. This method is available on both Series with datetime
    values (using the `dt` accessor) or DatetimeIndex.

    Returns
    -------
    Series or Index
        Containing integers indicating the day number.

    See Also
    --------
    Series.dt.dayofweek : Alias.
    Series.dt.weekday : Alias.
    Series.dt.day_name : Returns the name of the day of the week.

    Examples
    --------
    >>> s = pd.date_range('2016-12-31', '2017-01-08', freq='D').to_series()
    >>> s.dt.dayofweek
    2016-12-31    5
    2017-01-01    6
    2017-01-02    0
    2017-01-03    1
    2017-01-04    2
    2017-01-05    3
    2017-01-06    4
    2017-01-07    5
    2017-01-08    6
    Freq: D, dtype: int32
    """
    day_of_week = _field_accessor("day_of_week", "dow", _dayofweek_doc)
    dayofweek = day_of_week
    weekday = day_of_week

    day_of_year = _field_accessor(
        "dayofyear",
        "doy",
        """
        The ordinal day of the year.

        Examples
        --------
        For Series:

        >>> s = pd.Series(["1/1/2020 10:00:00+00:00", "2/1/2020 11:00:00+00:00"])
        >>> s = pd.to_datetime(s)
        >>> s
        0   2020-01-01 10:00:00+00:00
        1   2020-02-01 11:00:00+00:00
        dtype: datetime64[ns, UTC]
        >>> s.dt.dayofyear
        0    1
        1   32
        dtype: int32

        For DatetimeIndex:

        >>> idx = pd.DatetimeIndex(["1/1/2020 10:00:00+00:00",
        ...                         "2/1/2020 11:00:00+00:00"])
        >>> idx.dayofyear
        Index([1, 32], dtype='int32')
        """,
    )
    dayofyear = day_of_year
    quarter = _field_accessor(
        "quarter",
        "q",
        """
        The quarter of the date.

        Examples
        --------
        For Series:

        >>> s = pd.Series(["1/1/2020 10:00:00+00:00", "4/1/2020 11:00:00+00:00"])
        >>> s = pd.to_datetime(s)
        >>> s
        0   2020-01-01 10:00:00+00:00
        1   2020-04-01 11:00:00+00:00
        dtype: datetime64[ns, UTC]
        >>> s.dt.quarter
        0    1
        1    2
        dtype: int32

        For DatetimeIndex:

        >>> idx = pd.DatetimeIndex(["1/1/2020 10:00:00+00:00",
        ...                         "2/1/2020 11:00:00+00:00"])
        >>> idx.quarter
        Index([1, 1], dtype='int32')
        """,
    )
    days_in_month = _field_accessor(
        "days_in_month",
        "dim",
        """
        The number of days in the month.

        Examples
        --------
        >>> s = pd.Series(["1/1/2020 10:00:00+00:00", "2/1/2020 11:00:00+00:00"])
        >>> s = pd.to_datetime(s)
        >>> s
        0   2020-01-01 10:00:00+00:00
        1   2020-02-01 11:00:00+00:00
        dtype: datetime64[ns, UTC]
        >>> s.dt.daysinmonth
        0    31
        1    29
        dtype: int32
        """,
    )
    daysinmonth = days_in_month
    _is_month_doc = """
        Indicates whether the date is the {first_or_last} day of the month.

        Returns
        -------
        Series or array
            For Series, returns a Series with boolean values.
            For DatetimeIndex, returns a boolean array.

        See Also
        --------
        is_month_start : Return a boolean indicating whether the date
            is the first day of the month.
        is_month_end : Return a boolean indicating whether the date
            is the last day of the month.

        Examples
        --------
        This method is available on Series with datetime values under
        the ``.dt`` accessor, and directly on DatetimeIndex.

        >>> s = pd.Series(pd.date_range("2018-02-27", periods=3))
        >>> s
        0   2018-02-27
        1   2018-02-28
        2   2018-03-01
        dtype: datetime64[ns]
        >>> s.dt.is_month_start
        0    False
        1    False
        2    True
        dtype: bool
        >>> s.dt.is_month_end
        0    False
        1    True
        2    False
        dtype: bool

        >>> idx = pd.date_range("2018-02-27", periods=3)
        >>> idx.is_month_start
        array([False, False, True])
        >>> idx.is_month_end
        array([False, True, False])
    """
    is_month_start = _field_accessor(
        "is_month_start", "is_month_start", _is_month_doc.format(first_or_last="first")
    )

    is_month_end = _field_accessor(
        "is_month_end", "is_month_end", _is_month_doc.format(first_or_last="last")
    )

    is_quarter_start = _field_accessor(
        "is_quarter_start",
        "is_quarter_start",
        """
        Indicator for whether the date is the first day of a quarter.

        Returns
        -------
        is_quarter_start : Series or DatetimeIndex
            The same type as the original data with boolean values. Series will
            have the same name and index. DatetimeIndex will have the same
            name.

        See Also
        --------
        quarter : Return the quarter of the date.
        is_quarter_end : Similar property for indicating the quarter end.

        Examples
        --------
        This method is available on Series with datetime values under
        the ``.dt`` accessor, and directly on DatetimeIndex.

        >>> df = pd.DataFrame({'dates': pd.date_range("2017-03-30",
        ...                   periods=4)})
        >>> df.assign(quarter=df.dates.dt.quarter,
        ...           is_quarter_start=df.dates.dt.is_quarter_start)
               dates  quarter  is_quarter_start
        0 2017-03-30        1             False
        1 2017-03-31        1             False
        2 2017-04-01        2              True
        3 2017-04-02        2             False

        >>> idx = pd.date_range('2017-03-30', periods=4)
        >>> idx
        DatetimeIndex(['2017-03-30', '2017-03-31', '2017-04-01', '2017-04-02'],
                      dtype='datetime64[ns]', freq='D')

        >>> idx.is_quarter_start
        array([False, False,  True, False])
        """,
    )
    is_quarter_end = _field_accessor(
        "is_quarter_end",
        "is_quarter_end",
        """
        Indicator for whether the date is the last day of a quarter.

        Returns
        -------
        is_quarter_end : Series or DatetimeIndex
            The same type as the original data with boolean values. Series will
            have the same name and index. DatetimeIndex will have the same
            name.

        See Also
        --------
        quarter : Return the quarter of the date.
        is_quarter_start : Similar property indicating the quarter start.

        Examples
        --------
        This method is available on Series with datetime values under
        the ``.dt`` accessor, and directly on DatetimeIndex.

        >>> df = pd.DataFrame({'dates': pd.date_range("2017-03-30",
        ...                    periods=4)})
        >>> df.assign(quarter=df.dates.dt.quarter,
        ...           is_quarter_end=df.dates.dt.is_quarter_end)
               dates  quarter    is_quarter_end
        0 2017-03-30        1             False
        1 2017-03-31        1              True
        2 2017-04-01        2             False
        3 2017-04-02        2             False

        >>> idx = pd.date_range('2017-03-30', periods=4)
        >>> idx
        DatetimeIndex(['2017-03-30', '2017-03-31', '2017-04-01', '2017-04-02'],
                      dtype='datetime64[ns]', freq='D')

        >>> idx.is_quarter_end
        array([False,  True, False, False])
        """,
    )
    is_year_start = _field_accessor(
        "is_year_start",
        "is_year_start",
        """
        Indicate whether the date is the first day of a year.

        Returns
        -------
        Series or DatetimeIndex
            The same type as the original data with boolean values. Series will
            have the same name and index. DatetimeIndex will have the same
            name.

        See Also
        --------
        is_year_end : Similar property indicating the last day of the year.

        Examples
        --------
        This method is available on Series with datetime values under
        the ``.dt`` accessor, and directly on DatetimeIndex.

        >>> dates = pd.Series(pd.date_range("2017-12-30", periods=3))
        >>> dates
        0   2017-12-30
        1   2017-12-31
        2   2018-01-01
        dtype: datetime64[ns]

        >>> dates.dt.is_year_start
        0    False
        1    False
        2    True
        dtype: bool

        >>> idx = pd.date_range("2017-12-30", periods=3)
        >>> idx
        DatetimeIndex(['2017-12-30', '2017-12-31', '2018-01-01'],
                      dtype='datetime64[ns]', freq='D')

        >>> idx.is_year_start
        array([False, False,  True])
        """,
    )
    is_year_end = _field_accessor(
        "is_year_end",
        "is_year_end",
        """
        Indicate whether the date is the last day of the year.

        Returns
        -------
        Series or DatetimeIndex
            The same type as the original data with boolean values. Series will
            have the same name and index. DatetimeIndex will have the same
            name.

        See Also
        --------
        is_year_start : Similar property indicating the start of the year.

        Examples
        --------
        This method is available on Series with datetime values under
        the ``.dt`` accessor, and directly on DatetimeIndex.

        >>> dates = pd.Series(pd.date_range("2017-12-30", periods=3))
        >>> dates
        0   2017-12-30
        1   2017-12-31
        2   2018-01-01
        dtype: datetime64[ns]

        >>> dates.dt.is_year_end
        0    False
        1     True
        2    False
        dtype: bool

        >>> idx = pd.date_range("2017-12-30", periods=3)
        >>> idx
        DatetimeIndex(['2017-12-30', '2017-12-31', '2018-01-01'],
                      dtype='datetime64[ns]', freq='D')

        >>> idx.is_year_end
        array([False,  True, False])
        """,
    )
    is_leap_year = _field_accessor(
        "is_leap_year",
        "is_leap_year",
        """
        Boolean indicator if the date belongs to a leap year.

        A leap year is a year, which has 366 days (instead of 365) including
        29th of February as an intercalary day.
        Leap years are years which are multiples of four with the exception
        of years divisible by 100 but not by 400.

        Returns
        -------
        Series or ndarray
             Booleans indicating if dates belong to a leap year.

        Examples
        --------
        This method is available on Series with datetime values under
        the ``.dt`` accessor, and directly on DatetimeIndex.

        >>> idx = pd.date_range("2012-01-01", "2015-01-01", freq="YE")
        >>> idx
        DatetimeIndex(['2012-12-31', '2013-12-31', '2014-12-31'],
                      dtype='datetime64[ns]', freq='YE-DEC')
        >>> idx.is_leap_year
        array([ True, False, False])

        >>> dates_series = pd.Series(idx)
        >>> dates_series
        0   2012-12-31
        1   2013-12-31
        2   2014-12-31
        dtype: datetime64[ns]
        >>> dates_series.dt.is_leap_year
        0     True
        1    False
        2    False
        dtype: bool
        """,
    )

    def to_julian_date(self) -> npt.NDArray[np.float64]:
        """
        Convert Datetime Array to float64 ndarray of Julian Dates.
        0 Julian date is noon January 1, 4713 BC.
        https://en.wikipedia.org/wiki/Julian_day
        """

        # http://mysite.verizon.net/aesir_research/date/jdalg2.htm
        year = np.asarray(self.year)
        month = np.asarray(self.month)
        day = np.asarray(self.day)
        testarr = month < 3
        year[testarr] -= 1
        month[testarr] += 12
        return (
            day
            + np.fix((153 * month - 457) / 5)
            + 365 * year
            + np.floor(year / 4)
            - np.floor(year / 100)
            + np.floor(year / 400)
            + 1_721_118.5
            + (
                self.hour
                + self.minute / 60
                + self.second / 3600
                + self.microsecond / 3600 / 10**6
                + self.nanosecond / 3600 / 10**9
            )
            / 24
        )

    # -----------------------------------------------------------------
    # Reductions

    def std(
        self,
        axis=None,
        dtype=None,
        out=None,
        ddof: int = 1,
        keepdims: bool = False,
        skipna: bool = True,
    ):
        """
        Return sample standard deviation over requested axis.

        Normalized by `N-1` by default. This can be changed using ``ddof``.

        Parameters
        ----------
        axis : int, optional
            Axis for the function to be applied on. For :class:`pandas.Series`
            this parameter is unused and defaults to ``None``.
        ddof : int, default 1
            Degrees of Freedom. The divisor used in calculations is `N - ddof`,
            where `N` represents the number of elements.
        skipna : bool, default True
            Exclude NA/null values. If an entire row/column is ``NA``, the result
            will be ``NA``.

        Returns
        -------
        Timedelta

        See Also
        --------
        numpy.ndarray.std : Returns the standard deviation of the array elements
            along given axis.
        Series.std : Return sample standard deviation over requested axis.

        Examples
        --------
        For :class:`pandas.DatetimeIndex`:

        >>> idx = pd.date_range('2001-01-01 00:00', periods=3)
        >>> idx
        DatetimeIndex(['2001-01-01', '2001-01-02', '2001-01-03'],
                      dtype='datetime64[ns]', freq='D')
        >>> idx.std()
        Timedelta('1 days 00:00:00')
        """
        # Because std is translation-invariant, we can get self.std
        #  by calculating (self - Timestamp(0)).std, and we can do it
        #  without creating a copy by using a view on self._ndarray
        from pandas.core.arrays import TimedeltaArray

        # Find the td64 dtype with the same resolution as our dt64 dtype
        dtype_str = self._ndarray.dtype.name.replace("datetime64", "timedelta64")
        dtype = np.dtype(dtype_str)

        tda = TimedeltaArray._simple_new(self._ndarray.view(dtype), dtype=dtype)

        return tda.std(axis=axis, out=out, ddof=ddof, keepdims=keepdims, skipna=skipna)


# -------------------------------------------------------------------
# Constructor Helpers


def _sequence_to_dt64(
    data: ArrayLike,
    *,
    copy: bool = False,
    tz: tzinfo | None = None,
    dayfirst: bool = False,
    yearfirst: bool = False,
    ambiguous: TimeAmbiguous = "raise",
    out_unit: str | None = None,
):
    """
    Parameters
    ----------
    data : np.ndarray or ExtensionArray
        dtl.ensure_arraylike_for_datetimelike has already been called.
    copy : bool, default False
    tz : tzinfo or None, default None
    dayfirst : bool, default False
    yearfirst : bool, default False
    ambiguous : str, bool, or arraylike, default 'raise'
        See pandas._libs.tslibs.tzconversion.tz_localize_to_utc.
    out_unit : str or None, default None
        Desired output resolution.

    Returns
    -------
    result : numpy.ndarray
        The sequence converted to a numpy array with dtype ``datetime64[unit]``.
        Where `unit` is "ns" unless specified otherwise by `out_unit`.
    tz : tzinfo or None
        Either the user-provided tzinfo or one inferred from the data.

    Raises
    ------
    TypeError : PeriodDType data is passed
    """

    # By this point we are assured to have either a numpy array or Index
    data, copy = maybe_convert_dtype(data, copy, tz=tz)
    data_dtype = getattr(data, "dtype", None)

    if out_unit is None:
        out_unit = "ns"
    out_dtype = np.dtype(f"M8[{out_unit}]")

    if data_dtype == object or is_string_dtype(data_dtype):
        # TODO: We do not have tests specific to string-dtypes,
        #  also complex or categorical or other extension
        data = cast(np.ndarray, data)
        copy = False
        if lib.infer_dtype(data, skipna=False) == "integer":
            # Much more performant than going through array_to_datetime
            data = data.astype(np.int64)
        elif tz is not None and ambiguous == "raise":
            obj_data = np.asarray(data, dtype=object)
            result = tslib.array_to_datetime_with_tz(
                obj_data,
                tz=tz,
                dayfirst=dayfirst,
                yearfirst=yearfirst,
                creso=abbrev_to_npy_unit(out_unit),
            )
            return result, tz
        else:
            converted, inferred_tz = objects_to_datetime64(
                data,
                dayfirst=dayfirst,
                yearfirst=yearfirst,
                allow_object=False,
                out_unit=out_unit or "ns",
            )
            copy = False
            if tz and inferred_tz:
                #  two timezones: convert to intended from base UTC repr
                # GH#42505 by convention, these are _already_ UTC
                result = converted

            elif inferred_tz:
                tz = inferred_tz
                result = converted

            else:
                result, _ = _construct_from_dt64_naive(
                    converted, tz=tz, copy=copy, ambiguous=ambiguous
                )
            return result, tz

        data_dtype = data.dtype

    # `data` may have originally been a Categorical[datetime64[ns, tz]],
    # so we need to handle these types.
    if isinstance(data_dtype, DatetimeTZDtype):
        # DatetimeArray -> ndarray
        data = cast(DatetimeArray, data)
        tz = _maybe_infer_tz(tz, data.tz)
        result = data._ndarray

    elif lib.is_np_dtype(data_dtype, "M"):
        # tz-naive DatetimeArray or ndarray[datetime64]
        if isinstance(data, DatetimeArray):
            data = data._ndarray

        data = cast(np.ndarray, data)
        result, copy = _construct_from_dt64_naive(
            data, tz=tz, copy=copy, ambiguous=ambiguous
        )

    else:
        # must be integer dtype otherwise
        # assume this data are epoch timestamps
        if data.dtype != INT64_DTYPE:
            data = data.astype(np.int64, copy=False)
            copy = False
        data = cast(np.ndarray, data)
        result = data.view(out_dtype)

    if copy:
        result = result.copy()

    assert isinstance(result, np.ndarray), type(result)
    assert result.dtype.kind == "M"
    assert result.dtype != "M8"
    assert is_supported_dtype(result.dtype)
    return result, tz


def _construct_from_dt64_naive(
    data: np.ndarray, *, tz: tzinfo | None, copy: bool, ambiguous: TimeAmbiguous
) -> tuple[np.ndarray, bool]:
    """
    Convert datetime64 data to a supported dtype, localizing if necessary.
    """
    # Caller is responsible for ensuring
    #  lib.is_np_dtype(data.dtype)

    new_dtype = data.dtype
    if not is_supported_dtype(new_dtype):
        # Cast to the nearest supported unit, generally "s"
        new_dtype = get_supported_dtype(new_dtype)
        data = astype_overflowsafe(data, dtype=new_dtype, copy=False)
        copy = False

    if data.dtype.byteorder == ">":
        # TODO: better way to handle this?  non-copying alternative?
        #  without this, test_constructor_datetime64_bigendian fails
        data = data.astype(data.dtype.newbyteorder("<"))
        new_dtype = data.dtype
        copy = False

    if tz is not None:
        # Convert tz-naive to UTC
        # TODO: if tz is UTC, are there situations where we *don't* want a
        #  copy?  tz_localize_to_utc always makes one.
        shape = data.shape
        if data.ndim > 1:
            data = data.ravel()

        data_unit = get_unit_from_dtype(new_dtype)
        data = tzconversion.tz_localize_to_utc(
            data.view("i8"), tz, ambiguous=ambiguous, creso=data_unit
        )
        data = data.view(new_dtype)
        data = data.reshape(shape)

    assert data.dtype == new_dtype, data.dtype
    result = data

    return result, copy


def objects_to_datetime64(
    data: np.ndarray,
    dayfirst,
    yearfirst,
    utc: bool = False,
    errors: DateTimeErrorChoices = "raise",
    allow_object: bool = False,
    out_unit: str = "ns",
):
    """
    Convert data to array of timestamps.

    Parameters
    ----------
    data : np.ndarray[object]
    dayfirst : bool
    yearfirst : bool
    utc : bool, default False
        Whether to convert/localize timestamps to UTC.
    errors : {'raise', 'ignore', 'coerce'}
    allow_object : bool
        Whether to return an object-dtype ndarray instead of raising if the
        data contains more than one timezone.
    out_unit : str, default "ns"

    Returns
    -------
    result : ndarray
        np.datetime64[out_unit] if returned values represent wall times or UTC
        timestamps.
        object if mixed timezones
    inferred_tz : tzinfo or None
        If not None, then the datetime64 values in `result` denote UTC timestamps.

    Raises
    ------
    ValueError : if data cannot be converted to datetimes
    TypeError  : When a type cannot be converted to datetime
    """
    assert errors in ["raise", "ignore", "coerce"]

    # if str-dtype, convert
    data = np.asarray(data, dtype=np.object_)

    result, tz_parsed = tslib.array_to_datetime(
        data,
        errors=errors,
        utc=utc,
        dayfirst=dayfirst,
        yearfirst=yearfirst,
        creso=abbrev_to_npy_unit(out_unit),
    )

    if tz_parsed is not None:
        # We can take a shortcut since the datetime64 numpy array
        #  is in UTC
        return result, tz_parsed
    elif result.dtype.kind == "M":
        return result, tz_parsed
    elif result.dtype == object:
        # GH#23675 when called via `pd.to_datetime`, returning an object-dtype
        #  array is allowed.  When called via `pd.DatetimeIndex`, we can
        #  only accept datetime64 dtype, so raise TypeError if object-dtype
        #  is returned, as that indicates the values can be recognized as
        #  datetimes but they have conflicting timezones/awareness
        if allow_object:
            return result, tz_parsed
        raise TypeError("DatetimeIndex has mixed timezones")
    else:  # pragma: no cover
        # GH#23675 this TypeError should never be hit, whereas the TypeError
        #  in the object-dtype branch above is reachable.
        raise TypeError(result)


def maybe_convert_dtype(data, copy: bool, tz: tzinfo | None = None):
    """
    Convert data based on dtype conventions, issuing
    errors where appropriate.

    Parameters
    ----------
    data : np.ndarray or pd.Index
    copy : bool
    tz : tzinfo or None, default None

    Returns
    -------
    data : np.ndarray or pd.Index
    copy : bool

    Raises
    ------
    TypeError : PeriodDType data is passed
    """
    if not hasattr(data, "dtype"):
        # e.g. collections.deque
        return data, copy

    if is_float_dtype(data.dtype):
        # pre-2.0 we treated these as wall-times, inconsistent with ints
        # GH#23675, GH#45573 deprecated to treat symmetrically with integer dtypes.
        # Note: data.astype(np.int64) fails ARM tests, see
        # https://github.com/pandas-dev/pandas/issues/49468.
        data = data.astype(DT64NS_DTYPE).view("i8")
        copy = False

    elif lib.is_np_dtype(data.dtype, "m") or is_bool_dtype(data.dtype):
        # GH#29794 enforcing deprecation introduced in GH#23539
        raise TypeError(f"dtype {data.dtype} cannot be converted to datetime64[ns]")
    elif isinstance(data.dtype, PeriodDtype):
        # Note: without explicitly raising here, PeriodIndex
        #  test_setops.test_join_does_not_recur fails
        raise TypeError(
            "Passing PeriodDtype data is invalid. Use `data.to_timestamp()` instead"
        )

    elif isinstance(data.dtype, ExtensionDtype) and not isinstance(
        data.dtype, DatetimeTZDtype
    ):
        # TODO: We have no tests for these
        data = np.array(data, dtype=np.object_)
        copy = False

    return data, copy


# -------------------------------------------------------------------
# Validation and Inference


def _maybe_infer_tz(tz: tzinfo | None, inferred_tz: tzinfo | None) -> tzinfo | None:
    """
    If a timezone is inferred from data, check that it is compatible with
    the user-provided timezone, if any.

    Parameters
    ----------
    tz : tzinfo or None
    inferred_tz : tzinfo or None

    Returns
    -------
    tz : tzinfo or None

    Raises
    ------
    TypeError : if both timezones are present but do not match
    """
    if tz is None:
        tz = inferred_tz
    elif inferred_tz is None:
        pass
    elif not timezones.tz_compare(tz, inferred_tz):
        raise TypeError(
            f"data is already tz-aware {inferred_tz}, unable to "
            f"set specified tz: {tz}"
        )
    return tz


def _validate_dt64_dtype(dtype):
    """
    Check that a dtype, if passed, represents either a numpy datetime64[ns]
    dtype or a pandas DatetimeTZDtype.

    Parameters
    ----------
    dtype : object

    Returns
    -------
    dtype : None, numpy.dtype, or DatetimeTZDtype

    Raises
    ------
    ValueError : invalid dtype

    Notes
    -----
    Unlike _validate_tz_from_dtype, this does _not_ allow non-existent
    tz errors to go through
    """
    if dtype is not None:
        dtype = pandas_dtype(dtype)
        if dtype == np.dtype("M8"):
            # no precision, disallowed GH#24806
            msg = (
                "Passing in 'datetime64' dtype with no precision is not allowed. "
                "Please pass in 'datetime64[ns]' instead."
            )
            raise ValueError(msg)

        if (
            isinstance(dtype, np.dtype)
            and (dtype.kind != "M" or not is_supported_dtype(dtype))
        ) or not isinstance(dtype, (np.dtype, DatetimeTZDtype)):
            raise ValueError(
                f"Unexpected value for 'dtype': '{dtype}'. "
                "Must be 'datetime64[s]', 'datetime64[ms]', 'datetime64[us]', "
                "'datetime64[ns]' or DatetimeTZDtype'."
            )

        if getattr(dtype, "tz", None):
            # https://github.com/pandas-dev/pandas/issues/18595
            # Ensure that we have a standard timezone for pytz objects.
            # Without this, things like adding an array of timedeltas and
            # a  tz-aware Timestamp (with a tz specific to its datetime) will
            # be incorrect(ish?) for the array as a whole
            dtype = cast(DatetimeTZDtype, dtype)
            dtype = DatetimeTZDtype(
                unit=dtype.unit, tz=timezones.tz_standardize(dtype.tz)
            )

    return dtype


def _validate_tz_from_dtype(
    dtype, tz: tzinfo | None, explicit_tz_none: bool = False
) -> tzinfo | None:
    """
    If the given dtype is a DatetimeTZDtype, extract the implied
    tzinfo object from it and check that it does not conflict with the given
    tz.

    Parameters
    ----------
    dtype : dtype, str
    tz : None, tzinfo
    explicit_tz_none : bool, default False
        Whether tz=None was passed explicitly, as opposed to lib.no_default.

    Returns
    -------
    tz : consensus tzinfo

    Raises
    ------
    ValueError : on tzinfo mismatch
    """
    if dtype is not None:
        if isinstance(dtype, str):
            try:
                dtype = DatetimeTZDtype.construct_from_string(dtype)
            except TypeError:
                # Things like `datetime64[ns]`, which is OK for the
                # constructors, but also nonsense, which should be validated
                # but not by us. We *do* allow non-existent tz errors to
                # go through
                pass
        dtz = getattr(dtype, "tz", None)
        if dtz is not None:
            if tz is not None and not timezones.tz_compare(tz, dtz):
                raise ValueError("cannot supply both a tz and a dtype with a tz")
            if explicit_tz_none:
                raise ValueError("Cannot pass both a timezone-aware dtype and tz=None")
            tz = dtz

        if tz is not None and lib.is_np_dtype(dtype, "M"):
            # We also need to check for the case where the user passed a
            #  tz-naive dtype (i.e. datetime64[ns])
            if tz is not None and not timezones.tz_compare(tz, dtz):
                raise ValueError(
                    "cannot supply both a tz and a "
                    "timezone-naive dtype (i.e. datetime64[ns])"
                )

    return tz


def _infer_tz_from_endpoints(
    start: Timestamp, end: Timestamp, tz: tzinfo | None
) -> tzinfo | None:
    """
    If a timezone is not explicitly given via `tz`, see if one can
    be inferred from the `start` and `end` endpoints.  If more than one
    of these inputs provides a timezone, require that they all agree.

    Parameters
    ----------
    start : Timestamp
    end : Timestamp
    tz : tzinfo or None

    Returns
    -------
    tz : tzinfo or None

    Raises
    ------
    TypeError : if start and end timezones do not agree
    """
    try:
        inferred_tz = timezones.infer_tzinfo(start, end)
    except AssertionError as err:
        # infer_tzinfo raises AssertionError if passed mismatched timezones
        raise TypeError(
            "Start and end cannot both be tz-aware with different timezones"
        ) from err

    inferred_tz = timezones.maybe_get_tz(inferred_tz)
    tz = timezones.maybe_get_tz(tz)

    if tz is not None and inferred_tz is not None:
        if not timezones.tz_compare(inferred_tz, tz):
            raise AssertionError("Inferred time zone not equal to passed time zone")

    elif inferred_tz is not None:
        tz = inferred_tz

    return tz


def _maybe_normalize_endpoints(
    start: Timestamp | None, end: Timestamp | None, normalize: bool
):
    if normalize:
        if start is not None:
            start = start.normalize()

        if end is not None:
            end = end.normalize()

    return start, end


def _maybe_localize_point(
    ts: Timestamp | None, freq, tz, ambiguous, nonexistent
) -> Timestamp | None:
    """
    Localize a start or end Timestamp to the timezone of the corresponding
    start or end Timestamp

    Parameters
    ----------
    ts : start or end Timestamp to potentially localize
    freq : Tick, DateOffset, or None
    tz : str, timezone object or None
    ambiguous: str, localization behavior for ambiguous times
    nonexistent: str, localization behavior for nonexistent times

    Returns
    -------
    ts : Timestamp
    """
    # Make sure start and end are timezone localized if:
    # 1) freq = a Timedelta-like frequency (Tick)
    # 2) freq = None i.e. generating a linspaced range
    if ts is not None and ts.tzinfo is None:
        # Note: We can't ambiguous='infer' a singular ambiguous time; however,
        # we have historically defaulted ambiguous=False
        ambiguous = ambiguous if ambiguous != "infer" else False
        localize_args = {"ambiguous": ambiguous, "nonexistent": nonexistent, "tz": None}
        if isinstance(freq, Tick) or freq is None:
            localize_args["tz"] = tz
        ts = ts.tz_localize(**localize_args)
    return ts


def _generate_range(
    start: Timestamp | None,
    end: Timestamp | None,
    periods: int | None,
    offset: BaseOffset,
    *,
    unit: str,
):
    """
    Generates a sequence of dates corresponding to the specified time
    offset. Similar to dateutil.rrule except uses pandas DateOffset
    objects to represent time increments.

    Parameters
    ----------
    start : Timestamp or None
    end : Timestamp or None
    periods : int or None
    offset : DateOffset
    unit : str

    Notes
    -----
    * This method is faster for generating weekdays than dateutil.rrule
    * At least two of (start, end, periods) must be specified.
    * If both start and end are specified, the returned dates will
    satisfy start <= date <= end.

    Returns
    -------
    dates : generator object
    """
    offset = to_offset(offset)

    # Argument 1 to "Timestamp" has incompatible type "Optional[Timestamp]";
    # expected "Union[integer[Any], float, str, date, datetime64]"
    start = Timestamp(start)  # type: ignore[arg-type]
    if start is not NaT:
        start = start.as_unit(unit)
    else:
        start = None

    # Argument 1 to "Timestamp" has incompatible type "Optional[Timestamp]";
    # expected "Union[integer[Any], float, str, date, datetime64]"
    end = Timestamp(end)  # type: ignore[arg-type]
    if end is not NaT:
        end = end.as_unit(unit)
    else:
        end = None

    if start and not offset.is_on_offset(start):
        # Incompatible types in assignment (expression has type "datetime",
        # variable has type "Optional[Timestamp]")
        start = offset.rollforward(start)  # type: ignore[assignment]

    elif end and not offset.is_on_offset(end):
        # Incompatible types in assignment (expression has type "datetime",
        # variable has type "Optional[Timestamp]")
        end = offset.rollback(end)  # type: ignore[assignment]

    # Unsupported operand types for < ("Timestamp" and "None")
    if periods is None and end < start and offset.n >= 0:  # type: ignore[operator]
        end = None
        periods = 0

    if end is None:
        # error: No overload variant of "__radd__" of "BaseOffset" matches
        # argument type "None"
        end = start + (periods - 1) * offset  # type: ignore[operator]

    if start is None:
        # error: No overload variant of "__radd__" of "BaseOffset" matches
        # argument type "None"
        start = end - (periods - 1) * offset  # type: ignore[operator]

    start = cast(Timestamp, start)
    end = cast(Timestamp, end)

    cur = start
    if offset.n >= 0:
        while cur <= end:
            yield cur

            if cur == end:
                # GH#24252 avoid overflows by not performing the addition
                # in offset.apply unless we have to
                break

            # faster than cur + offset
            next_date = offset._apply(cur)
            next_date = next_date.as_unit(unit)
            if next_date <= cur:
                raise ValueError(f"Offset {offset} did not increment date")
            cur = next_date
    else:
        while cur >= end:
            yield cur

            if cur == end:
                # GH#24252 avoid overflows by not performing the addition
                # in offset.apply unless we have to
                break

            # faster than cur + offset
            next_date = offset._apply(cur)
            next_date = next_date.as_unit(unit)
            if next_date >= cur:
                raise ValueError(f"Offset {offset} did not decrement date")
            cur = next_date
<|endoftext|>from __future__ import annotations

from csv import QUOTE_NONNUMERIC
from functools import partial
import operator
from shutil import get_terminal_size
from typing import (
    TYPE_CHECKING,
    Literal,
    cast,
    overload,
)
import warnings

import numpy as np

from pandas._config import get_option

from pandas._libs import (
    NaT,
    algos as libalgos,
    lib,
)
from pandas._libs.arrays import NDArrayBacked
from pandas.compat.numpy import function as nv
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import validate_bool_kwarg

from pandas.core.dtypes.cast import (
    coerce_indexer_dtype,
    find_common_type,
)
from pandas.core.dtypes.common import (
    ensure_int64,
    ensure_platform_int,
    is_any_real_numeric_dtype,
    is_bool_dtype,
    is_dict_like,
    is_hashable,
    is_integer_dtype,
    is_list_like,
    is_scalar,
    needs_i8_conversion,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import (
    ArrowDtype,
    CategoricalDtype,
    CategoricalDtypeType,
    ExtensionDtype,
)
from pandas.core.dtypes.generic import (
    ABCIndex,
    ABCSeries,
)
from pandas.core.dtypes.missing import (
    is_valid_na_for_dtype,
    isna,
)

from pandas.core import (
    algorithms,
    arraylike,
    ops,
)
from pandas.core.accessor import (
    PandasDelegate,
    delegate_names,
)
from pandas.core.algorithms import (
    factorize,
    take_nd,
)
from pandas.core.arrays._mixins import (
    NDArrayBackedExtensionArray,
    ravel_compat,
)
from pandas.core.base import (
    ExtensionArray,
    NoNewAttributesMixin,
    PandasObject,
)
import pandas.core.common as com
from pandas.core.construction import (
    extract_array,
    sanitize_array,
)
from pandas.core.ops.common import unpack_zerodim_and_defer
from pandas.core.sorting import nargsort
from pandas.core.strings.object_array import ObjectStringArrayMixin

from pandas.io.formats import console

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Iterator,
        Sequence,
    )

    from pandas._typing import (
        ArrayLike,
        AstypeArg,
        AxisInt,
        Dtype,
        DtypeObj,
        NpDtype,
        Ordered,
        Self,
        Shape,
        SortKind,
        npt,
    )

    from pandas import (
        DataFrame,
        Index,
        Series,
    )


def _cat_compare_op(op):
    opname = f"__{op.__name__}__"
    fill_value = op is operator.ne

    @unpack_zerodim_and_defer(opname)
    def func(self, other):
        hashable = is_hashable(other)
        if is_list_like(other) and len(other) != len(self) and not hashable:
            # in hashable case we may have a tuple that is itself a category
            raise ValueError("Lengths must match.")

        if not self.ordered:
            if opname in ["__lt__", "__gt__", "__le__", "__ge__"]:
                raise TypeError(
                    "Unordered Categoricals can only compare equality or not"
                )
        if isinstance(other, Categorical):
            # Two Categoricals can only be compared if the categories are
            # the same (maybe up to ordering, depending on ordered)

            msg = "Categoricals can only be compared if 'categories' are the same."
            if not self._categories_match_up_to_permutation(other):
                raise TypeError(msg)

            if not self.ordered and not self.categories.equals(other.categories):
                # both unordered and different order
                other_codes = recode_for_categories(
                    other.codes, other.categories, self.categories, copy=False
                )
            else:
                other_codes = other._codes

            ret = op(self._codes, other_codes)
            mask = (self._codes == -1) | (other_codes == -1)
            if mask.any():
                ret[mask] = fill_value
            return ret

        if hashable:
            if other in self.categories:
                i = self._unbox_scalar(other)
                ret = op(self._codes, i)

                if opname not in {"__eq__", "__ge__", "__gt__"}:
                    # GH#29820 performance trick; get_loc will always give i>=0,
                    #  so in the cases (__ne__, __le__, __lt__) the setting
                    #  here is a no-op, so can be skipped.
                    mask = self._codes == -1
                    ret[mask] = fill_value
                return ret
            else:
                return ops.invalid_comparison(self, other, op)
        else:
            # allow categorical vs object dtype array comparisons for equality
            # these are only positional comparisons
            if opname not in ["__eq__", "__ne__"]:
                raise TypeError(
                    f"Cannot compare a Categorical for op {opname} with "
                    f"type {type(other)}.\nIf you want to compare values, "
                    "use 'np.asarray(cat) <op> other'."
                )

            if isinstance(other, ExtensionArray) and needs_i8_conversion(other.dtype):
                # We would return NotImplemented here, but that messes up
                #  ExtensionIndex's wrapped methods
                return op(other, self)
            return getattr(np.array(self), opname)(np.array(other))

    func.__name__ = opname

    return func


def contains(cat, key, container) -> bool:
    """
    Helper for membership check for ``key`` in ``cat``.

    This is a helper method for :method:`__contains__`
    and :class:`CategoricalIndex.__contains__`.

    Returns True if ``key`` is in ``cat.categories`` and the
    location of ``key`` in ``categories`` is in ``container``.

    Parameters
    ----------
    cat : :class:`Categorical`or :class:`categoricalIndex`
    key : a hashable object
        The key to check membership for.
    container : Container (e.g. list-like or mapping)
        The container to check for membership in.

    Returns
    -------
    is_in : bool
        True if ``key`` is in ``self.categories`` and location of
        ``key`` in ``categories`` is in ``container``, else False.

    Notes
    -----
    This method does not check for NaN values. Do that separately
    before calling this method.
    """
    hash(key)

    # get location of key in categories.
