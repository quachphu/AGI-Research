        4) A CDT with ordered=True is only equal to another CDT with
           ordered=True and identical categories in the same order
        5) A CDT with ordered={False, None} is only equal to another CDT with
           ordered={False, None} and identical categories, but same order is
           not required. There is no distinction between False/None.
        6) Any other comparison returns False
        """
        if isinstance(other, str):
            return other == self.name
        elif other is self:
            return True
        elif not (hasattr(other, "ordered") and hasattr(other, "categories")):
            return False
        elif self.categories is None or other.categories is None:
            # For non-fully-initialized dtypes, these are only equal to
            #  - the string "category" (handled above)
            #  - other CategoricalDtype with categories=None
            return self.categories is other.categories
        elif self.ordered or other.ordered:
            # At least one has ordered=True; equal if both have ordered=True
            # and the same values for categories in the same order.
            return (self.ordered == other.ordered) and self.categories.equals(
                other.categories
            )
        else:
            # Neither has ordered=True; equal if both have the same categories,
            # but same order is not necessary.  There is no distinction between
            # ordered=False and ordered=None: CDT(., False) and CDT(., None)
            # will be equal if they have the same categories.
            left = self.categories
            right = other.categories

            # GH#36280 the ordering of checks here is for performance
            if not left.dtype == right.dtype:
                return False

            if len(left) != len(right):
                return False

            if self.categories.equals(other.categories):
                # Check and see if they happen to be identical categories
                return True

            if left.dtype != object:
                # Faster than calculating hash
                indexer = left.get_indexer(right)
                # Because left and right have the same length and are unique,
                #  `indexer` not having any -1s implies that there is a
                #  bijection between `left` and `right`.
                return bool((indexer != -1).all())

            # With object-dtype we need a comparison that identifies
            #  e.g. int(2) as distinct from float(2)
            return set(left) == set(right)

    def __repr__(self) -> str_type:
        if self.categories is None:
            data = "None"
            dtype = "None"
        else:
            data = self.categories._format_data(name=type(self).__name__)
            if isinstance(self.categories, ABCRangeIndex):
                data = str(self.categories._range)
            data = data.rstrip(", ")
            dtype = self.categories.dtype

        return (
            f"CategoricalDtype(categories={data}, ordered={self.ordered}, "
            f"categories_dtype={dtype})"
        )

    @cache_readonly
    def _hash_categories(self) -> int:
        from pandas.core.util.hashing import (
            combine_hash_arrays,
            hash_array,
            hash_tuples,
        )

        categories = self.categories
        ordered = self.ordered

        if len(categories) and isinstance(categories[0], tuple):
            # assumes if any individual category is a tuple, then all our. ATM
            # I don't really want to support just some of the categories being
            # tuples.
            cat_list = list(categories)  # breaks if a np.array of categories
            cat_array = hash_tuples(cat_list)
        else:
            if categories.dtype == "O" and len({type(x) for x in categories}) != 1:
                # TODO: hash_array doesn't handle mixed types. It casts
                # everything to a str first, which means we treat
                # {'1', '2'} the same as {'1', 2}
                # find a better solution
                hashed = hash((tuple(categories), ordered))
                return hashed

            if DatetimeTZDtype.is_dtype(categories.dtype):
                # Avoid future warning.
                categories = categories.view("datetime64[ns]")

            cat_array = hash_array(np.asarray(categories), categorize=False)
        if ordered:
            cat_array = np.vstack(
                [cat_array, np.arange(len(cat_array), dtype=cat_array.dtype)]
            )
        else:
            cat_array = np.array([cat_array])
        combined_hashed = combine_hash_arrays(iter(cat_array), num_items=len(cat_array))
        return np.bitwise_xor.reduce(combined_hashed)

    @classmethod
    def construct_array_type(cls) -> type_t[Categorical]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        from pandas import Categorical

        return Categorical

    @staticmethod
    def validate_ordered(ordered: Ordered) -> None:
        """
        Validates that we have a valid ordered parameter. If
        it is not a boolean, a TypeError will be raised.

        Parameters
        ----------
        ordered : object
            The parameter to be verified.

        Raises
        ------
        TypeError
            If 'ordered' is not a boolean.
        """
        if not is_bool(ordered):
            raise TypeError("'ordered' must either be 'True' or 'False'")

    @staticmethod
    def validate_categories(categories, fastpath: bool = False) -> Index:
        """
        Validates that we have good categories

        Parameters
        ----------
        categories : array-like
        fastpath : bool
            Whether to skip nan and uniqueness checks

        Returns
        -------
        categories : Index
        """
        from pandas.core.indexes.base import Index

        if not fastpath and not is_list_like(categories):
            raise TypeError(
                f"Parameter 'categories' must be list-like, was {repr(categories)}"
            )
        if not isinstance(categories, ABCIndex):
            categories = Index._with_infer(categories, tupleize_cols=False)

        if not fastpath:
            if categories.hasnans:
                raise ValueError("Categorical categories cannot be null")

            if not categories.is_unique:
                raise ValueError("Categorical categories must be unique")

        if isinstance(categories, ABCCategoricalIndex):
            categories = categories.categories

        return categories

    def update_dtype(self, dtype: str_type | CategoricalDtype) -> CategoricalDtype:
        """
        Returns a CategoricalDtype with categories and ordered taken from dtype
        if specified, otherwise falling back to self if unspecified

        Parameters
        ----------
        dtype : CategoricalDtype

        Returns
        -------
        new_dtype : CategoricalDtype
        """
        if isinstance(dtype, str) and dtype == "category":
            # dtype='category' should not change anything
            return self
        elif not self.is_dtype(dtype):
            raise ValueError(
                f"a CategoricalDtype must be passed to perform an update, "
                f"got {repr(dtype)}"
            )
        else:
            # from here on, dtype is a CategoricalDtype
            dtype = cast(CategoricalDtype, dtype)

        # update categories/ordered unless they've been explicitly passed as None
        new_categories = (
            dtype.categories if dtype.categories is not None else self.categories
        )
        new_ordered = dtype.ordered if dtype.ordered is not None else self.ordered

        return CategoricalDtype(new_categories, new_ordered)

    @property
    def categories(self) -> Index:
        """
        An ``Index`` containing the unique categories allowed.

        Examples
        --------
        >>> cat_type = pd.CategoricalDtype(categories=['a', 'b'], ordered=True)
        >>> cat_type.categories
        Index(['a', 'b'], dtype='object')
        """
        return self._categories

    @property
    def ordered(self) -> Ordered:
        """
        Whether the categories have an ordered relationship.

        Examples
        --------
        >>> cat_type = pd.CategoricalDtype(categories=['a', 'b'], ordered=True)
        >>> cat_type.ordered
        True

        >>> cat_type = pd.CategoricalDtype(categories=['a', 'b'], ordered=False)
        >>> cat_type.ordered
        False
        """
        return self._ordered

    @property
    def _is_boolean(self) -> bool:
        from pandas.core.dtypes.common import is_bool_dtype

        return is_bool_dtype(self.categories)

    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None:
        # check if we have all categorical dtype with identical categories
        if all(isinstance(x, CategoricalDtype) for x in dtypes):
            first = dtypes[0]
            if all(first == other for other in dtypes[1:]):
                return first

        # special case non-initialized categorical
        # TODO we should figure out the expected return value in general
        non_init_cats = [
            isinstance(x, CategoricalDtype) and x.categories is None for x in dtypes
        ]
        if all(non_init_cats):
            return self
        elif any(non_init_cats):
            return None

        # categorical is aware of Sparse -> extract sparse subdtypes
        dtypes = [x.subtype if isinstance(x, SparseDtype) else x for x in dtypes]
        # extract the categories' dtype
        non_cat_dtypes = [
            x.categories.dtype if isinstance(x, CategoricalDtype) else x for x in dtypes
        ]
        # TODO should categorical always give an answer?
        from pandas.core.dtypes.cast import find_common_type

        return find_common_type(non_cat_dtypes)

    @cache_readonly
    def index_class(self) -> type_t[CategoricalIndex]:
        from pandas import CategoricalIndex

        return CategoricalIndex


@register_extension_dtype
class DatetimeTZDtype(PandasExtensionDtype):
    """
    An ExtensionDtype for timezone-aware datetime data.

    **This is not an actual numpy dtype**, but a duck type.

    Parameters
    ----------
    unit : str, default "ns"
        The precision of the datetime data. Currently limited
        to ``"ns"``.
    tz : str, int, or datetime.tzinfo
        The timezone.

    Attributes
    ----------
    unit
    tz

    Methods
    -------
    None

    Raises
    ------
    ZoneInfoNotFoundError
        When the requested timezone cannot be found.

    Examples
    --------
    >>> from zoneinfo import ZoneInfo
    >>> pd.DatetimeTZDtype(tz=ZoneInfo('UTC'))
    datetime64[ns, UTC]

    >>> pd.DatetimeTZDtype(tz=ZoneInfo('Europe/Paris'))
    datetime64[ns, Europe/Paris]
    """

    type: type[Timestamp] = Timestamp
    kind: str_type = "M"
    num = 101
    _metadata = ("unit", "tz")
    _match = re.compile(r"(datetime64|M8)\[(?P<unit>.+), (?P<tz>.+)\]")
    _cache_dtypes: dict[str_type, PandasExtensionDtype] = {}
    _supports_2d = True
    _can_fast_transpose = True

    @property
    def na_value(self) -> NaTType:
        return NaT

    @cache_readonly
    def base(self) -> DtypeObj:  # type: ignore[override]
        return np.dtype(f"M8[{self.unit}]")

    # error: Signature of "str" incompatible with supertype "PandasExtensionDtype"
    @cache_readonly
    def str(self) -> str:  # type: ignore[override]
        return f"|M8[{self.unit}]"

    def __init__(self, unit: str_type | DatetimeTZDtype = "ns", tz=None) -> None:
        if isinstance(unit, DatetimeTZDtype):
            # error: "str" has no attribute "tz"
            unit, tz = unit.unit, unit.tz  # type: ignore[attr-defined]

        if unit != "ns":
            if isinstance(unit, str) and tz is None:
                # maybe a string like datetime64[ns, tz], which we support for
                # now.
                result = type(self).construct_from_string(unit)
                unit = result.unit
                tz = result.tz
                msg = (
                    f"Passing a dtype alias like 'datetime64[ns, {tz}]' "
                    "to DatetimeTZDtype is no longer supported. Use "
                    "'DatetimeTZDtype.construct_from_string()' instead."
                )
                raise ValueError(msg)
            if unit not in ["s", "ms", "us", "ns"]:
                raise ValueError("DatetimeTZDtype only supports s, ms, us, ns units")

        if tz:
            tz = timezones.maybe_get_tz(tz)
            tz = timezones.tz_standardize(tz)
        elif tz is not None:
            raise pytz.UnknownTimeZoneError(tz)
        if tz is None:
            raise TypeError("A 'tz' is required.")

        self._unit = unit
        self._tz = tz

    @cache_readonly
    def _creso(self) -> int:
        """
        The NPY_DATETIMEUNIT corresponding to this dtype's resolution.
        """
        return abbrev_to_npy_unit(self.unit)

    @property
    def unit(self) -> str_type:
        """
        The precision of the datetime data.

        Examples
        --------
        >>> from zoneinfo import ZoneInfo
        >>> dtype = pd.DatetimeTZDtype(tz=ZoneInfo('America/Los_Angeles'))
        >>> dtype.unit
        'ns'
        """
        return self._unit

    @property
    def tz(self) -> tzinfo:
        """
        The timezone.

        Examples
        --------
        >>> from zoneinfo import ZoneInfo
        >>> dtype = pd.DatetimeTZDtype(tz=ZoneInfo('America/Los_Angeles'))
        >>> dtype.tz
        zoneinfo.ZoneInfo(key='America/Los_Angeles')
        """
        return self._tz

    @classmethod
    def construct_array_type(cls) -> type_t[DatetimeArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        from pandas.core.arrays import DatetimeArray

        return DatetimeArray

    @classmethod
    def construct_from_string(cls, string: str_type) -> DatetimeTZDtype:
        """
        Construct a DatetimeTZDtype from a string.

        Parameters
        ----------
        string : str
            The string alias for this DatetimeTZDtype.
            Should be formatted like ``datetime64[ns, <tz>]``,
            where ``<tz>`` is the timezone name.

        Examples
        --------
        >>> DatetimeTZDtype.construct_from_string('datetime64[ns, UTC]')
        datetime64[ns, UTC]
        """
        if not isinstance(string, str):
            raise TypeError(
                f"'construct_from_string' expects a string, got {type(string)}"
            )

        msg = f"Cannot construct a 'DatetimeTZDtype' from '{string}'"
        match = cls._match.match(string)
        if match:
            d = match.groupdict()
            try:
                return cls(unit=d["unit"], tz=d["tz"])
            except (KeyError, TypeError, ValueError) as err:
                # KeyError if maybe_get_tz tries and fails to get a
                #  pytz timezone (actually pytz.UnknownTimeZoneError).
                # TypeError if we pass a nonsense tz;
                # ValueError if we pass a unit other than "ns"
                raise TypeError(msg) from err
        raise TypeError(msg)

    def __str__(self) -> str_type:
        return f"datetime64[{self.unit}, {self.tz}]"

    @property
    def name(self) -> str_type:
        """A string representation of the dtype."""
        return str(self)

    def __hash__(self) -> int:
        # make myself hashable
        # TODO: update this.
        return hash(str(self))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            if other.startswith("M8["):
                other = f"datetime64[{other[3:]}"
            return other == self.name

        return (
            isinstance(other, DatetimeTZDtype)
            and self.unit == other.unit
            and tz_compare(self.tz, other.tz)
        )

    def __from_arrow__(self, array: pa.Array | pa.ChunkedArray) -> DatetimeArray:
        """
        Construct DatetimeArray from pyarrow Array/ChunkedArray.

        Note: If the units in the pyarrow Array are the same as this
        DatetimeDtype, then values corresponding to the integer representation
        of ``NaT`` (e.g. one nanosecond before :attr:`pandas.Timestamp.min`)
        are converted to ``NaT``, regardless of the null indicator in the
        pyarrow array.

        Parameters
        ----------
        array : pyarrow.Array or pyarrow.ChunkedArray
            The Arrow array to convert to DatetimeArray.

        Returns
        -------
        extension array : DatetimeArray
        """
        import pyarrow

        from pandas.core.arrays import DatetimeArray

        array = array.cast(pyarrow.timestamp(unit=self._unit), safe=True)

        if isinstance(array, pyarrow.Array):
            np_arr = array.to_numpy(zero_copy_only=False)
        else:
            np_arr = array.to_numpy()

        return DatetimeArray._simple_new(np_arr, dtype=self)

    def __setstate__(self, state) -> None:
        # for pickle compat. __get_state__ is defined in the
        # PandasExtensionDtype superclass and uses the public properties to
        # pickle -> need to set the settable private ones here (see GH26067)
        self._tz = state["tz"]
        self._unit = state["unit"]

    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None:
        if all(isinstance(t, DatetimeTZDtype) and t.tz == self.tz for t in dtypes):
            np_dtype = np.max([cast(DatetimeTZDtype, t).base for t in [self, *dtypes]])
            unit = np.datetime_data(np_dtype)[0]
            return type(self)(unit=unit, tz=self.tz)
        return super()._get_common_dtype(dtypes)

    @cache_readonly
    def index_class(self) -> type_t[DatetimeIndex]:
        from pandas import DatetimeIndex

        return DatetimeIndex


@register_extension_dtype
class PeriodDtype(PeriodDtypeBase, PandasExtensionDtype):
    """
    An ExtensionDtype for Period data.

    **This is not an actual numpy dtype**, but a duck type.

    Parameters
    ----------
    freq : str or DateOffset
        The frequency of this PeriodDtype.

    Attributes
    ----------
    freq

    Methods
    -------
    None

    Examples
    --------
    >>> pd.PeriodDtype(freq='D')
    period[D]

    >>> pd.PeriodDtype(freq=pd.offsets.MonthEnd())
    period[M]
    """

    type: type[Period] = Period
    kind: str_type = "O"
    str = "|O08"
    base = np.dtype("O")
    num = 102
    _metadata = ("freq",)
    _match = re.compile(r"(P|p)eriod\[(?P<freq>.+)\]")
    # error: Incompatible types in assignment (expression has type
    # "Dict[int, PandasExtensionDtype]", base class "PandasExtensionDtype"
    # defined the type as "Dict[str, PandasExtensionDtype]")  [assignment]
    _cache_dtypes: dict[BaseOffset, int] = {}  # type: ignore[assignment]
    __hash__ = PeriodDtypeBase.__hash__
    _freq: BaseOffset
    _supports_2d = True
    _can_fast_transpose = True

    def __new__(cls, freq) -> PeriodDtype:  # noqa: PYI034
        """
        Parameters
        ----------
        freq : PeriodDtype, BaseOffset, or string
        """
        if isinstance(freq, PeriodDtype):
            return freq

        if not isinstance(freq, BaseOffset):
            freq = cls._parse_dtype_strict(freq)

        if isinstance(freq, BDay):
            # GH#53446
            # TODO(3.0): enforcing this will close GH#10575
            warnings.warn(
                "PeriodDtype[B] is deprecated and will be removed in a future "
                "version. Use a DatetimeIndex with freq='B' instead",
                FutureWarning,
                stacklevel=find_stack_level(),
            )

        try:
            dtype_code = cls._cache_dtypes[freq]
        except KeyError:
            dtype_code = freq._period_dtype_code
            cls._cache_dtypes[freq] = dtype_code
        u = PeriodDtypeBase.__new__(cls, dtype_code, freq.n)
        u._freq = freq
        return u

    def __reduce__(self) -> tuple[type_t[Self], tuple[str_type]]:
        return type(self), (self.name,)

    @property
    def freq(self) -> BaseOffset:
        """
        The frequency object of this PeriodDtype.

        Examples
        --------
        >>> dtype = pd.PeriodDtype(freq='D')
        >>> dtype.freq
        <Day>
        """
        return self._freq

    @classmethod
    def _parse_dtype_strict(cls, freq: str_type) -> BaseOffset:
        if isinstance(freq, str):  # note: freq is already of type str!
            if freq.startswith(("Period[", "period[")):
                m = cls._match.search(freq)
                if m is not None:
                    freq = m.group("freq")

            freq_offset = to_offset(freq, is_period=True)
            if freq_offset is not None:
                return freq_offset

        raise TypeError(
            "PeriodDtype argument should be string or BaseOffset, "
            f"got {type(freq).__name__}"
        )

    @classmethod
    def construct_from_string(cls, string: str_type) -> PeriodDtype:
        """
        Strict construction from a string, raise a TypeError if not
        possible
        """
        if (
            isinstance(string, str)
            and (string.startswith(("period[", "Period[")))
            or isinstance(string, BaseOffset)
        ):
            # do not parse string like U as period[U]
            # avoid tuple to be regarded as freq
            try:
                return cls(freq=string)
            except ValueError:
                pass
        if isinstance(string, str):
            msg = f"Cannot construct a 'PeriodDtype' from '{string}'"
        else:
            msg = f"'construct_from_string' expects a string, got {type(string)}"
        raise TypeError(msg)

    def __str__(self) -> str_type:
        return self.name

    @property
    def name(self) -> str_type:
        return f"period[{self._freqstr}]"

    @property
    def na_value(self) -> NaTType:
        return NaT

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return other in [self.name, capitalize_first_letter(self.name)]

        return super().__eq__(other)

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    @classmethod
    def is_dtype(cls, dtype: object) -> bool:
        """
        Return a boolean if we if the passed type is an actual dtype that we
        can match (via string or type)
        """
        if isinstance(dtype, str):
            # PeriodDtype can be instantiated from freq string like "U",
            # but doesn't regard freq str like "U" as dtype.
            if dtype.startswith(("period[", "Period[")):
                try:
                    return cls._parse_dtype_strict(dtype) is not None
                except ValueError:
                    return False
            else:
                return False
        return super().is_dtype(dtype)

    @classmethod
    def construct_array_type(cls) -> type_t[PeriodArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        from pandas.core.arrays import PeriodArray

        return PeriodArray

    def __from_arrow__(self, array: pa.Array | pa.ChunkedArray) -> PeriodArray:
        """
        Construct PeriodArray from pyarrow Array/ChunkedArray.
        """
        import pyarrow

        from pandas.core.arrays import PeriodArray
        from pandas.core.arrays.arrow._arrow_utils import (
            pyarrow_array_to_numpy_and_mask,
        )

        if isinstance(array, pyarrow.Array):
            chunks = [array]
        else:
            chunks = array.chunks

        results = []
        for arr in chunks:
            data, mask = pyarrow_array_to_numpy_and_mask(arr, dtype=np.dtype(np.int64))
            parr = PeriodArray(data.copy(), dtype=self, copy=False)
            # error: Invalid index type "ndarray[Any, dtype[bool_]]" for "PeriodArray";
            # expected type "Union[int, Sequence[int], Sequence[bool], slice]"
            parr[~mask] = NaT  # type: ignore[index]
            results.append(parr)

        if not results:
            return PeriodArray(np.array([], dtype="int64"), dtype=self, copy=False)
        return PeriodArray._concat_same_type(results)

    @cache_readonly
    def index_class(self) -> type_t[PeriodIndex]:
        from pandas import PeriodIndex

        return PeriodIndex


@register_extension_dtype
class IntervalDtype(PandasExtensionDtype):
    """
    An ExtensionDtype for Interval data.

    **This is not an actual numpy dtype**, but a duck type.

    Parameters
    ----------
    subtype : str, np.dtype
        The dtype of the Interval bounds.

    Attributes
    ----------
    subtype

    Methods
    -------
    None

    Examples
    --------
    >>> pd.IntervalDtype(subtype='int64', closed='both')
    interval[int64, both]
    """

    name = "interval"
    kind: str_type = "O"
    str = "|O08"
    base = np.dtype("O")
    num = 103
    _metadata = (
        "subtype",
        "closed",
    )

    _match = re.compile(
        r"(I|i)nterval\[(?P<subtype>[^,]+(\[.+\])?)"
        r"(, (?P<closed>(right|left|both|neither)))?\]"
    )

    _cache_dtypes: dict[str_type, PandasExtensionDtype] = {}
    _subtype: None | np.dtype
    _closed: IntervalClosedType | None

    def __init__(self, subtype=None, closed: IntervalClosedType | None = None) -> None:
        from pandas.core.dtypes.common import (
            is_string_dtype,
            pandas_dtype,
        )

        if closed is not None and closed not in {"right", "left", "both", "neither"}:
            raise ValueError("closed must be one of 'right', 'left', 'both', 'neither'")

        if isinstance(subtype, IntervalDtype):
            if closed is not None and closed != subtype.closed:
                raise ValueError(
                    "dtype.closed and 'closed' do not match. "
                    "Try IntervalDtype(dtype.subtype, closed) instead."
                )
            self._subtype = subtype._subtype
            self._closed = subtype._closed
        elif subtype is None:
            # we are called as an empty constructor
            # generally for pickle compat
            self._subtype = None
            self._closed = closed
        elif isinstance(subtype, str) and subtype.lower() == "interval":
            self._subtype = None
            self._closed = closed
        else:
            if isinstance(subtype, str):
                m = IntervalDtype._match.search(subtype)
                if m is not None:
                    gd = m.groupdict()
                    subtype = gd["subtype"]
                    if gd.get("closed", None) is not None:
                        if closed is not None:
                            if closed != gd["closed"]:
                                raise ValueError(
                                    "'closed' keyword does not match value "
                                    "specified in dtype string"
                                )
                        closed = gd["closed"]  # type: ignore[assignment]

            try:
                subtype = pandas_dtype(subtype)
            except TypeError as err:
                raise TypeError("could not construct IntervalDtype") from err
            if CategoricalDtype.is_dtype(subtype) or is_string_dtype(subtype):
                # GH 19016
                msg = (
                    "category, object, and string subtypes are not supported "
                    "for IntervalDtype"
                )
                raise TypeError(msg)
            self._subtype = subtype
            self._closed = closed

    @cache_readonly
    def _can_hold_na(self) -> bool:
        subtype = self._subtype
        if subtype is None:
            # partially-initialized
            raise NotImplementedError(
                "_can_hold_na is not defined for partially-initialized IntervalDtype"
            )
        if subtype.kind in "iu":
            return False
        return True

    @property
    def closed(self) -> IntervalClosedType:
        return self._closed  # type: ignore[return-value]

    @property
    def subtype(self):
        """
        The dtype of the Interval bounds.

        Examples
        --------
        >>> dtype = pd.IntervalDtype(subtype='int64', closed='both')
        >>> dtype.subtype
        dtype('int64')
        """
        return self._subtype

    @classmethod
    def construct_array_type(cls) -> type[IntervalArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        from pandas.core.arrays import IntervalArray

        return IntervalArray

    @classmethod
    def construct_from_string(cls, string: str_type) -> IntervalDtype:
        """
        attempt to construct this type from a string, raise a TypeError
        if its not possible
        """
        if not isinstance(string, str):
            raise TypeError(
                f"'construct_from_string' expects a string, got {type(string)}"
            )

        if string.lower() == "interval" or cls._match.search(string) is not None:
            return cls(string)

        msg = (
            f"Cannot construct a 'IntervalDtype' from '{string}'.\n\n"
            "Incorrectly formatted string passed to constructor. "
            "Valid formats include Interval or Interval[dtype] "
            "where dtype is numeric, datetime, or timedelta"
        )
        raise TypeError(msg)

    @property
    def type(self) -> type[Interval]:
        return Interval

    def __str__(self) -> str_type:
        if self.subtype is None:
            return "interval"
        if self.closed is None:
            # Only partially initialized GH#38394
            return f"interval[{self.subtype}]"
        return f"interval[{self.subtype}, {self.closed}]"

    def __hash__(self) -> int:
        # make myself hashable
        return hash(str(self))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return other.lower() in (self.name.lower(), str(self).lower())
        elif not isinstance(other, IntervalDtype):
            return False
        elif self.subtype is None or other.subtype is None:
            # None should match any subtype
            return True
        elif self.closed != other.closed:
            return False
        else:
            return self.subtype == other.subtype

    def __setstate__(self, state) -> None:
        # for pickle compat. __get_state__ is defined in the
        # PandasExtensionDtype superclass and uses the public properties to
        # pickle -> need to set the settable private ones here (see GH26067)
        self._subtype = state["subtype"]

        # backward-compat older pickles won't have "closed" key
        self._closed = state.pop("closed", None)

    @classmethod
    def is_dtype(cls, dtype: object) -> bool:
        """
        Return a boolean if we if the passed type is an actual dtype that we
        can match (via string or type)
        """
        if isinstance(dtype, str):
            if dtype.lower().startswith("interval"):
                try:
                    return cls.construct_from_string(dtype) is not None
                except (ValueError, TypeError):
                    return False
            else:
                return False
        return super().is_dtype(dtype)

    def __from_arrow__(self, array: pa.Array | pa.ChunkedArray) -> IntervalArray:
        """
        Construct IntervalArray from pyarrow Array/ChunkedArray.
        """
        import pyarrow

        from pandas.core.arrays import IntervalArray

        if isinstance(array, pyarrow.Array):
            chunks = [array]
        else:
            chunks = array.chunks

        results = []
        for arr in chunks:
            if isinstance(arr, pyarrow.ExtensionArray):
                arr = arr.storage
            left = np.asarray(arr.field("left"), dtype=self.subtype)
            right = np.asarray(arr.field("right"), dtype=self.subtype)
            iarr = IntervalArray.from_arrays(left, right, closed=self.closed)
            results.append(iarr)

        if not results:
            return IntervalArray.from_arrays(
                np.array([], dtype=self.subtype),
                np.array([], dtype=self.subtype),
                closed=self.closed,
            )
        return IntervalArray._concat_same_type(results)

    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None:
        if not all(isinstance(x, IntervalDtype) for x in dtypes):
            return None

        closed = cast("IntervalDtype", dtypes[0]).closed
        if not all(cast("IntervalDtype", x).closed == closed for x in dtypes):
            return np.dtype(object)

        from pandas.core.dtypes.cast import find_common_type

        common = find_common_type([cast("IntervalDtype", x).subtype for x in dtypes])
        if common == object:
            return np.dtype(object)
        return IntervalDtype(common, closed=closed)

    @cache_readonly
    def index_class(self) -> type_t[IntervalIndex]:
        from pandas import IntervalIndex

        return IntervalIndex


class NumpyEADtype(ExtensionDtype):
    """
    A Pandas ExtensionDtype for NumPy dtypes.

    This is mostly for internal compatibility, and is not especially
    useful on its own.

    Parameters
    ----------
    dtype : object
        Object to be converted to a NumPy data type object.

    See Also
    --------
    numpy.dtype
    """

    _metadata = ("_dtype",)
    _supports_2d = False
    _can_fast_transpose = False

    def __init__(self, dtype: npt.DTypeLike | NumpyEADtype | None) -> None:
        if isinstance(dtype, NumpyEADtype):
            # make constructor idempotent
            dtype = dtype.numpy_dtype
        self._dtype = np.dtype(dtype)

    def __repr__(self) -> str:
        return f"NumpyEADtype({repr(self.name)})"

    @property
    def numpy_dtype(self) -> np.dtype:
        """
        The NumPy dtype this NumpyEADtype wraps.
        """
        return self._dtype

    @property
    def name(self) -> str:
        """
        A bit-width name for this data-type.
        """
        return self._dtype.name

    @property
    def type(self) -> type[np.generic]:
        """
        The type object used to instantiate a scalar of this NumPy data-type.
        """
        return self._dtype.type

    @property
    def _is_numeric(self) -> bool:
        # exclude object, str, unicode, void.
        return self.kind in set("biufc")

    @property
    def _is_boolean(self) -> bool:
        return self.kind == "b"

    @classmethod
    def construct_from_string(cls, string: str) -> NumpyEADtype:
        try:
            dtype = np.dtype(string)
        except TypeError as err:
            if not isinstance(string, str):
                msg = f"'construct_from_string' expects a string, got {type(string)}"
            else:
                msg = f"Cannot construct a 'NumpyEADtype' from '{string}'"
            raise TypeError(msg) from err
        return cls(dtype)

    @classmethod
    def construct_array_type(cls) -> type_t[NumpyExtensionArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        from pandas.core.arrays import NumpyExtensionArray

        return NumpyExtensionArray

    @property
    def kind(self) -> str:
        """
        A character code (one of 'biufcmMOSUV') identifying the general kind of data.
        """
        return self._dtype.kind

    @property
    def itemsize(self) -> int:
        """
        The element size of this data-type object.
        """
        return self._dtype.itemsize


class BaseMaskedDtype(ExtensionDtype):
    """
    Base class for dtypes for BaseMaskedArray subclasses.
    """

    base = None
    type: type

    @property
    def na_value(self) -> libmissing.NAType:
        return libmissing.NA

    @cache_readonly
    def numpy_dtype(self) -> np.dtype:
        """Return an instance of our numpy dtype"""
        return np.dtype(self.type)

    @cache_readonly
    def kind(self) -> str:
        return self.numpy_dtype.kind

    @cache_readonly
    def itemsize(self) -> int:
        """Return the number of bytes in this dtype"""
        return self.numpy_dtype.itemsize

    @classmethod
    def construct_array_type(cls) -> type_t[BaseMaskedArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        raise NotImplementedError

    @classmethod
    def from_numpy_dtype(cls, dtype: np.dtype) -> BaseMaskedDtype:
        """
        Construct the MaskedDtype corresponding to the given numpy dtype.
        """
        if dtype.kind == "b":
            from pandas.core.arrays.boolean import BooleanDtype

            return BooleanDtype()
        elif dtype.kind in "iu":
            from pandas.core.arrays.integer import NUMPY_INT_TO_DTYPE

            return NUMPY_INT_TO_DTYPE[dtype]
        elif dtype.kind == "f":
            from pandas.core.arrays.floating import NUMPY_FLOAT_TO_DTYPE

            return NUMPY_FLOAT_TO_DTYPE[dtype]
        else:
            raise NotImplementedError(dtype)

    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None:
        # We unwrap any masked dtypes, find the common dtype we would use
        #  for that, then re-mask the result.
        from pandas.core.dtypes.cast import find_common_type

        new_dtype = find_common_type(
            [
                dtype.numpy_dtype if isinstance(dtype, BaseMaskedDtype) else dtype
                for dtype in dtypes
            ]
        )
        if not isinstance(new_dtype, np.dtype):
            # If we ever support e.g. Masked[DatetimeArray] then this will change
            return None
        try:
            return type(self).from_numpy_dtype(new_dtype)
        except (KeyError, NotImplementedError):
            return None


@register_extension_dtype
class SparseDtype(ExtensionDtype):
    """
    Dtype for data stored in :class:`SparseArray`.

    This dtype implements the pandas ExtensionDtype interface.

    Parameters
    ----------
    dtype : str, ExtensionDtype, numpy.dtype, type, default numpy.float64
        The dtype of the underlying array storing the non-fill value values.
    fill_value : scalar, optional
        The scalar value not stored in the SparseArray. By default, this
        depends on `dtype`.

        =========== ==========
        dtype       na_value
        =========== ==========
        float       ``np.nan``
        int         ``0``
        bool        ``False``
        datetime64  ``pd.NaT``
        timedelta64 ``pd.NaT``
        =========== ==========

        The default value may be overridden by specifying a `fill_value`.

    Attributes
    ----------
    None

    Methods
    -------
    None

    Examples
    --------
    >>> ser = pd.Series([1, 0, 0], dtype=pd.SparseDtype(dtype=int, fill_value=0))
    >>> ser
    0    1
    1    0
    2    0
    dtype: Sparse[int64, 0]
    >>> ser.sparse.density
    0.3333333333333333
    """

    _is_immutable = True

    # We include `_is_na_fill_value` in the metadata to avoid hash collisions
    # between SparseDtype(float, 0.0) and SparseDtype(float, nan).
    # Without is_na_fill_value in the comparison, those would be equal since
    # hash(nan) is (sometimes?) 0.
    _metadata = ("_dtype", "_fill_value", "_is_na_fill_value")

    def __init__(self, dtype: Dtype = np.float64, fill_value: Any = None) -> None:
        if isinstance(dtype, type(self)):
            if fill_value is None:
                fill_value = dtype.fill_value
            dtype = dtype.subtype

        from pandas.core.dtypes.common import (
            is_string_dtype,
            pandas_dtype,
        )
        from pandas.core.dtypes.missing import na_value_for_dtype

        dtype = pandas_dtype(dtype)
        if is_string_dtype(dtype):
            dtype = np.dtype("object")
        if not isinstance(dtype, np.dtype):
            # GH#53160
            raise TypeError("SparseDtype subtype must be a numpy dtype")

        if fill_value is None:
            fill_value = na_value_for_dtype(dtype)

        self._dtype = dtype
        self._fill_value = fill_value
        self._check_fill_value()

    def __hash__(self) -> int:
        # Python3 doesn't inherit __hash__ when a base class overrides
        # __eq__, so we explicitly do it here.
        return super().__hash__()

    def __eq__(self, other: object) -> bool:
        # We have to override __eq__ to handle NA values in _metadata.
        # The base class does simple == checks, which fail for NA.
        if isinstance(other, str):
            try:
                other = self.construct_from_string(other)
            except TypeError:
                return False

        if isinstance(other, type(self)):
            subtype = self.subtype == other.subtype
            if self._is_na_fill_value:
                # this case is complicated by two things:
                # SparseDtype(float, float(nan)) == SparseDtype(float, np.nan)
                # SparseDtype(float, np.nan)     != SparseDtype(float, pd.NaT)
                # i.e. we want to treat any floating-point NaN as equal, but
                # not a floating-point NaN and a datetime NaT.
                fill_value = (
                    other._is_na_fill_value
                    and isinstance(self.fill_value, type(other.fill_value))
                    or isinstance(other.fill_value, type(self.fill_value))
                )
            else:
                with warnings.catch_warnings():
                    # Ignore spurious numpy warning
                    warnings.filterwarnings(
                        "ignore",
                        "elementwise comparison failed",
                        category=DeprecationWarning,
                    )

                    fill_value = self.fill_value == other.fill_value

            return subtype and fill_value
        return False

    @property
    def fill_value(self):
        """
        The fill value of the array.

        Converting the SparseArray to a dense ndarray will fill the
        array with this value.

        .. warning::

           It's possible to end up with a SparseArray that has ``fill_value``
           values in ``sp_values``. This can occur, for example, when setting
           ``SparseArray.fill_value`` directly.
        """
        return self._fill_value

    def _check_fill_value(self) -> None:
        if not lib.is_scalar(self._fill_value):
            raise ValueError(
                f"fill_value must be a scalar. Got {self._fill_value} instead"
            )

        from pandas.core.dtypes.cast import can_hold_element
        from pandas.core.dtypes.missing import (
            is_valid_na_for_dtype,
            isna,
        )

        from pandas.core.construction import ensure_wrapped_if_datetimelike

        # GH#23124 require fill_value and subtype to match
        val = self._fill_value
        if isna(val):
            if not is_valid_na_for_dtype(val, self.subtype):
                warnings.warn(
                    "Allowing arbitrary scalar fill_value in SparseDtype is "
                    "deprecated. In a future version, the fill_value must be "
                    "a valid value for the SparseDtype.subtype.",
                    FutureWarning,
                    stacklevel=find_stack_level(),
                )
        else:
            dummy = np.empty(0, dtype=self.subtype)
            dummy = ensure_wrapped_if_datetimelike(dummy)

            if not can_hold_element(dummy, val):
                warnings.warn(
                    "Allowing arbitrary scalar fill_value in SparseDtype is "
                    "deprecated. In a future version, the fill_value must be "
                    "a valid value for the SparseDtype.subtype.",
                    FutureWarning,
                    stacklevel=find_stack_level(),
                )

    @property
    def _is_na_fill_value(self) -> bool:
        from pandas import isna

        return isna(self.fill_value)

    @property
    def _is_numeric(self) -> bool:
        return self.subtype != object

    @property
    def _is_boolean(self) -> bool:
        return self.subtype.kind == "b"

    @property
    def kind(self) -> str:
        """
        The sparse kind. Either 'integer', or 'block'.
        """
        return self.subtype.kind

    @property
    def type(self):
        return self.subtype.type

    @property
    def subtype(self):
        return self._dtype

    @property
    def name(self) -> str:
        return f"Sparse[{self.subtype.name}, {repr(self.fill_value)}]"

    def __repr__(self) -> str:
        return self.name

    @classmethod
    def construct_array_type(cls) -> type_t[SparseArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        from pandas.core.arrays.sparse.array import SparseArray

        return SparseArray

    @classmethod
    def construct_from_string(cls, string: str) -> SparseDtype:
        """
        Construct a SparseDtype from a string form.

        Parameters
        ----------
        string : str
            Can take the following forms.

            string           dtype
            ================ ============================
            'int'            SparseDtype[np.int64, 0]
            'Sparse'         SparseDtype[np.float64, nan]
            'Sparse[int]'    SparseDtype[np.int64, 0]
            'Sparse[int, 0]' SparseDtype[np.int64, 0]
            ================ ============================

            It is not possible to specify non-default fill values
            with a string. An argument like ``'Sparse[int, 1]'``
            will raise a ``TypeError`` because the default fill value
            for integers is 0.

        Returns
        -------
        SparseDtype
        """
        if not isinstance(string, str):
            raise TypeError(
                f"'construct_from_string' expects a string, got {type(string)}"
            )
        msg = f"Cannot construct a 'SparseDtype' from '{string}'"
        if string.startswith("Sparse"):
            try:
                sub_type, has_fill_value = cls._parse_subtype(string)
            except ValueError as err:
                raise TypeError(msg) from err
            else:
                result = SparseDtype(sub_type)
                msg = (
                    f"Cannot construct a 'SparseDtype' from '{string}'.\n\nIt "
                    "looks like the fill_value in the string is not "
                    "the default for the dtype. Non-default fill_values "
                    "are not supported. Use the 'SparseDtype()' "
                    "constructor instead."
                )
                if has_fill_value and str(result) != string:
                    raise TypeError(msg)
                return result
        else:
            raise TypeError(msg)

    @staticmethod
    def _parse_subtype(dtype: str) -> tuple[str, bool]:
        """
        Parse a string to get the subtype

        Parameters
        ----------
        dtype : str
            A string like

            * Sparse[subtype]
            * Sparse[subtype, fill_value]

        Returns
        -------
        subtype : str

        Raises
        ------
        ValueError
            When the subtype cannot be extracted.
        """
        xpr = re.compile(r"Sparse\[(?P<subtype>[^,]*)(, )?(?P<fill_value>.*?)?\]$")
        m = xpr.match(dtype)
        has_fill_value = False
        if m:
            subtype = m.groupdict()["subtype"]
            has_fill_value = bool(m.groupdict()["fill_value"])
        elif dtype == "Sparse":
            subtype = "float64"
        else:
            raise ValueError(f"Cannot parse {dtype}")
        return subtype, has_fill_value

    @classmethod
    def is_dtype(cls, dtype: object) -> bool:
        dtype = getattr(dtype, "dtype", dtype)
        if isinstance(dtype, str) and dtype.startswith("Sparse"):
            sub_type, _ = cls._parse_subtype(dtype)
            dtype = np.dtype(sub_type)
        elif isinstance(dtype, cls):
            return True
        return isinstance(dtype, np.dtype) or dtype == "Sparse"

    def update_dtype(self, dtype) -> SparseDtype:
        """
        Convert the SparseDtype to a new dtype.

        This takes care of converting the ``fill_value``.

        Parameters
        ----------
        dtype : Union[str, numpy.dtype, SparseDtype]
            The new dtype to use.

            * For a SparseDtype, it is simply returned
            * For a NumPy dtype (or str), the current fill value
              is converted to the new dtype, and a SparseDtype
              with `dtype` and the new fill value is returned.

        Returns
        -------
        SparseDtype
            A new SparseDtype with the correct `dtype` and fill value
            for that `dtype`.

        Raises
        ------
        ValueError
            When the current fill value cannot be converted to the
            new `dtype` (e.g. trying to convert ``np.nan`` to an
            integer dtype).


        Examples
        --------
        >>> SparseDtype(int, 0).update_dtype(float)
        Sparse[float64, 0.0]

        >>> SparseDtype(int, 1).update_dtype(SparseDtype(float, np.nan))
        Sparse[float64, nan]
        """
        from pandas.core.dtypes.astype import astype_array
        from pandas.core.dtypes.common import pandas_dtype

        cls = type(self)
        dtype = pandas_dtype(dtype)

        if not isinstance(dtype, cls):
            if not isinstance(dtype, np.dtype):
                raise TypeError("sparse arrays of extension dtypes not supported")

            fv_asarray = np.atleast_1d(np.array(self.fill_value))
            fvarr = astype_array(fv_asarray, dtype)
            # NB: not fv_0d.item(), as that casts dt64->int
            fill_value = fvarr[0]
            dtype = cls(dtype, fill_value=fill_value)

        return dtype

    @property
    def _subtype_with_str(self):
        """
        Whether the SparseDtype's subtype should be considered ``str``.

        Typically, pandas will store string data in an object-dtype array.
        When converting values to a dtype, e.g. in ``.astype``, we need to
        be more specific, we need the actual underlying type.

        Returns
        -------
        >>> SparseDtype(int, 1)._subtype_with_str
        dtype('int64')

        >>> SparseDtype(object, 1)._subtype_with_str
        dtype('O')

        >>> dtype = SparseDtype(str, '')
        >>> dtype.subtype
        dtype('O')

        >>> dtype._subtype_with_str
        <class 'str'>
        """
        if isinstance(self.fill_value, str):
            return type(self.fill_value)
        return self.subtype

    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None:
        # TODO for now only handle SparseDtypes and numpy dtypes => extend
        # with other compatible extension dtypes
        from pandas.core.dtypes.cast import np_find_common_type

        if any(
            isinstance(x, ExtensionDtype) and not isinstance(x, SparseDtype)
            for x in dtypes
        ):
            return None

        fill_values = [x.fill_value for x in dtypes if isinstance(x, SparseDtype)]
        fill_value = fill_values[0]

        from pandas import isna

        # np.nan isn't a singleton, so we may end up with multiple
        # NaNs here, so we ignore the all NA case too.
        if not (len(set(fill_values)) == 1 or isna(fill_values).all()):
            warnings.warn(
                "Concatenating sparse arrays with multiple fill "
                f"values: '{fill_values}'. Picking the first and "
                "converting the rest.",
                PerformanceWarning,
                stacklevel=find_stack_level(),
            )

        np_dtypes = (x.subtype if isinstance(x, SparseDtype) else x for x in dtypes)
        return SparseDtype(np_find_common_type(*np_dtypes), fill_value=fill_value)


@register_extension_dtype
class ArrowDtype(StorageExtensionDtype):
    """
    An ExtensionDtype for PyArrow data types.

    .. warning::

       ArrowDtype is considered experimental. The implementation and
       parts of the API may change without warning.

    While most ``dtype`` arguments can accept the "string"
    constructor, e.g. ``"int64[pyarrow]"``, ArrowDtype is useful
    if the data type contains parameters like ``pyarrow.timestamp``.

    Parameters
    ----------
    pyarrow_dtype : pa.DataType
        An instance of a `pyarrow.DataType <https://arrow.apache.org/docs/python/api/datatypes.html#factory-functions>`__.

    Attributes
    ----------
    pyarrow_dtype

    Methods
    -------
    None

    Returns
    -------
    ArrowDtype

    Examples
    --------
    >>> import pyarrow as pa
    >>> pd.ArrowDtype(pa.int64())
    int64[pyarrow]

    Types with parameters must be constructed with ArrowDtype.

    >>> pd.ArrowDtype(pa.timestamp("s", tz="America/New_York"))
    timestamp[s, tz=America/New_York][pyarrow]
    >>> pd.ArrowDtype(pa.list_(pa.int64()))
    list<item: int64>[pyarrow]
    """

    _metadata = ("storage", "pyarrow_dtype")  # type: ignore[assignment]

    def __init__(self, pyarrow_dtype: pa.DataType) -> None:
        super().__init__("pyarrow")
        if pa_version_under10p1:
            raise ImportError("pyarrow>=10.0.1 is required for ArrowDtype")
        if not isinstance(pyarrow_dtype, pa.DataType):
            raise ValueError(
                f"pyarrow_dtype ({pyarrow_dtype}) must be an instance "
                f"of a pyarrow.DataType. Got {type(pyarrow_dtype)} instead."
            )
        self.pyarrow_dtype = pyarrow_dtype

    def __repr__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        # make myself hashable
        return hash(str(self))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return super().__eq__(other)
        return self.pyarrow_dtype == other.pyarrow_dtype

    @property
    def type(self):
        """
        Returns associated scalar type.
        """
        pa_type = self.pyarrow_dtype
        if pa.types.is_integer(pa_type):
            return int
        elif pa.types.is_floating(pa_type):
            return float
        elif pa.types.is_string(pa_type) or pa.types.is_large_string(pa_type):
            return str
        elif (
            pa.types.is_binary(pa_type)
            or pa.types.is_fixed_size_binary(pa_type)
            or pa.types.is_large_binary(pa_type)
        ):
            return bytes
        elif pa.types.is_boolean(pa_type):
            return bool
        elif pa.types.is_duration(pa_type):
            if pa_type.unit == "ns":
                return Timedelta
            else:
                return timedelta
        elif pa.types.is_timestamp(pa_type):
            if pa_type.unit == "ns":
                return Timestamp
            else:
                return datetime
        elif pa.types.is_date(pa_type):
            return date
        elif pa.types.is_time(pa_type):
            return time
        elif pa.types.is_decimal(pa_type):
            return Decimal
        elif pa.types.is_dictionary(pa_type):
            # TODO: Potentially change this & CategoricalDtype.type to
            #  something more representative of the scalar
            return CategoricalDtypeType
        elif pa.types.is_list(pa_type) or pa.types.is_large_list(pa_type):
            return list
        elif pa.types.is_fixed_size_list(pa_type):
            return list
        elif pa.types.is_map(pa_type):
            return list
        elif pa.types.is_struct(pa_type):
            return dict
        elif pa.types.is_null(pa_type):
            # TODO: None? pd.NA? pa.null?
            return type(pa_type)
        elif isinstance(pa_type, pa.ExtensionType):
            return type(self)(pa_type.storage_type).type
        raise NotImplementedError(pa_type)

    @property
    def name(self) -> str:  # type: ignore[override]
        """
        A string identifying the data type.
        """
        return f"{str(self.pyarrow_dtype)}[{self.storage}]"

    @cache_readonly
    def numpy_dtype(self) -> np.dtype:
        """Return an instance of the related numpy dtype"""
        if pa.types.is_timestamp(self.pyarrow_dtype):
            # pa.timestamp(unit).to_pandas_dtype() returns ns units
            # regardless of the pyarrow timestamp units.
            # This can be removed if/when pyarrow addresses it:
            # https://github.com/apache/arrow/issues/34462
            return np.dtype(f"datetime64[{self.pyarrow_dtype.unit}]")
        if pa.types.is_duration(self.pyarrow_dtype):
            # pa.duration(unit).to_pandas_dtype() returns ns units
            # regardless of the pyarrow duration units
            # This can be removed if/when pyarrow addresses it:
            # https://github.com/apache/arrow/issues/34462
            return np.dtype(f"timedelta64[{self.pyarrow_dtype.unit}]")
        if pa.types.is_string(self.pyarrow_dtype) or pa.types.is_large_string(
            self.pyarrow_dtype
        ):
            # pa.string().to_pandas_dtype() = object which we don't want
            return np.dtype(str)
        try:
            return np.dtype(self.pyarrow_dtype.to_pandas_dtype())
        except (NotImplementedError, TypeError):
            return np.dtype(object)

    @cache_readonly
    def kind(self) -> str:
        if pa.types.is_timestamp(self.pyarrow_dtype):
            # To mirror DatetimeTZDtype
            return "M"
        return self.numpy_dtype.kind

    @cache_readonly
    def itemsize(self) -> int:
        """Return the number of bytes in this dtype"""
        return self.numpy_dtype.itemsize

    @classmethod
    def construct_array_type(cls) -> type_t[ArrowExtensionArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        from pandas.core.arrays.arrow import ArrowExtensionArray

        return ArrowExtensionArray

    @classmethod
    def construct_from_string(cls, string: str) -> ArrowDtype:
        """
        Construct this type from a string.

        Parameters
        ----------
        string : str
            string should follow the format f"{pyarrow_type}[pyarrow]"
            e.g. int64[pyarrow]
        """
        if not isinstance(string, str):
            raise TypeError(
                f"'construct_from_string' expects a string, got {type(string)}"
            )
        if not string.endswith("[pyarrow]"):
            raise TypeError(f"'{string}' must end with '[pyarrow]'")
        if string in ("string[pyarrow]", "str[pyarrow]"):
            # Ensure Registry.find skips ArrowDtype to use StringDtype instead
            raise TypeError("string[pyarrow] should be constructed by StringDtype")

        base_type = string[:-9]  # get rid of "[pyarrow]"
        try:
            pa_dtype = pa.type_for_alias(base_type)
        except ValueError as err:
            has_parameters = re.search(r"[\[\(].*[\]\)]", base_type)
            if has_parameters:
                # Fallback to try common temporal types
                try:
                    return cls._parse_temporal_dtype_string(base_type)
                except (NotImplementedError, ValueError):
                    # Fall through to raise with nice exception message below
                    pass

                raise NotImplementedError(
                    "Passing pyarrow type specific parameters "
                    f"({has_parameters.group()}) in the string is not supported. "
                    "Please construct an ArrowDtype object with a pyarrow_dtype "
                    "instance with specific parameters."
                ) from err
            raise TypeError(f"'{base_type}' is not a valid pyarrow data type.") from err
        return cls(pa_dtype)

    # TODO(arrow#33642): This can be removed once supported by pyarrow
    @classmethod
    def _parse_temporal_dtype_string(cls, string: str) -> ArrowDtype:
        """
        Construct a temporal ArrowDtype from string.
        """
        # we assume
        #  1) "[pyarrow]" has already been stripped from the end of our string.
        #  2) we know "[" is present
        head, tail = string.split("[", 1)

        if not tail.endswith("]"):
            raise ValueError
        tail = tail[:-1]

        if head == "timestamp":
            assert "," in tail  # otherwise type_for_alias should work
            unit, tz = tail.split(",", 1)
            unit = unit.strip()
            tz = tz.strip()
            if tz.startswith("tz="):
                tz = tz[3:]

            pa_type = pa.timestamp(unit, tz=tz)
            dtype = cls(pa_type)
            return dtype

        raise NotImplementedError(string)

    @property
    def _is_numeric(self) -> bool:
        """
        Whether columns with this dtype should be considered numeric.
        """
        # TODO: pa.types.is_boolean?
        return (
            pa.types.is_integer(self.pyarrow_dtype)
            or pa.types.is_floating(self.pyarrow_dtype)
            or pa.types.is_decimal(self.pyarrow_dtype)
        )

    @property
    def _is_boolean(self) -> bool:
        """
        Whether this dtype should be considered boolean.
        """
        return pa.types.is_boolean(self.pyarrow_dtype)

    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None:
        # We unwrap any masked dtypes, find the common dtype we would use
        #  for that, then re-mask the result.
        # Mirrors BaseMaskedDtype
        from pandas.core.dtypes.cast import find_common_type

        null_dtype = type(self)(pa.null())

        new_dtype = find_common_type(
            [
                dtype.numpy_dtype if isinstance(dtype, ArrowDtype) else dtype
                for dtype in dtypes
                if dtype != null_dtype
            ]
        )
        if not isinstance(new_dtype, np.dtype):
            return None
        try:
            pa_dtype = pa.from_numpy_dtype(new_dtype)
            return type(self)(pa_dtype)
        except NotImplementedError:
            return None

    def __from_arrow__(self, array: pa.Array | pa.ChunkedArray):
        """
        Construct IntegerArray/FloatingArray from pyarrow Array/ChunkedArray.
        """
        array_class = self.construct_array_type()
        arr = array.cast(self.pyarrow_dtype, safe=True)
        return array_class(arr)
<|endoftext|>"""
Functions for implementing 'astype' methods according to pandas conventions,
particularly ones that differ from numpy.
"""
from __future__ import annotations

import inspect
from typing import (
    TYPE_CHECKING,
    overload,
)
import warnings

import numpy as np

from pandas._libs import lib
from pandas._libs.tslibs.timedeltas import array_to_timedelta64
from pandas.errors import IntCastingNaNError

from pandas.core.dtypes.common import (
    is_object_dtype,
    is_string_dtype,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import (
    ExtensionDtype,
    NumpyEADtype,
)

if TYPE_CHECKING:
    from pandas._typing import (
        ArrayLike,
        DtypeObj,
        IgnoreRaise,
    )

    from pandas.core.arrays import ExtensionArray

_dtype_obj = np.dtype(object)


@overload
def _astype_nansafe(
    arr: np.ndarray, dtype: np.dtype, copy: bool = ..., skipna: bool = ...
) -> np.ndarray:
    ...


@overload
def _astype_nansafe(
    arr: np.ndarray, dtype: ExtensionDtype, copy: bool = ..., skipna: bool = ...
) -> ExtensionArray:
    ...


def _astype_nansafe(
    arr: np.ndarray, dtype: DtypeObj, copy: bool = True, skipna: bool = False
) -> ArrayLike:
    """
    Cast the elements of an array to a given dtype a nan-safe manner.

    Parameters
    ----------
    arr : ndarray
    dtype : np.dtype or ExtensionDtype
    copy : bool, default True
        If False, a view will be attempted but may fail, if
        e.g. the item sizes don't align.
    skipna: bool, default False
        Whether or not we should skip NaN when casting as a string-type.

    Raises
    ------
    ValueError
        The dtype was a datetime64/timedelta64 dtype, but it had no unit.
    """

    # dispatch on extension dtype if needed
    if isinstance(dtype, ExtensionDtype):
        return dtype.construct_array_type()._from_sequence(arr, dtype=dtype, copy=copy)

    elif not isinstance(dtype, np.dtype):  # pragma: no cover
        raise ValueError("dtype must be np.dtype or ExtensionDtype")

    if arr.dtype.kind in "mM":
        from pandas.core.construction import ensure_wrapped_if_datetimelike

        arr = ensure_wrapped_if_datetimelike(arr)
        res = arr.astype(dtype, copy=copy)
        return np.asarray(res)

    if issubclass(dtype.type, str):
        shape = arr.shape
        if arr.ndim > 1:
            arr = arr.ravel()
        return lib.ensure_string_array(
            arr, skipna=skipna, convert_na_value=False
        ).reshape(shape)

    elif np.issubdtype(arr.dtype, np.floating) and dtype.kind in "iu":
        return _astype_float_to_int_nansafe(arr, dtype, copy)

    elif arr.dtype == object:
        # if we have a datetime/timedelta array of objects
        # then coerce to datetime64[ns] and use DatetimeArray.astype

        if lib.is_np_dtype(dtype, "M"):
            from pandas.core.arrays import DatetimeArray

            dta = DatetimeArray._from_sequence(arr, dtype=dtype)
            return dta._ndarray

        elif lib.is_np_dtype(dtype, "m"):
            from pandas.core.construction import ensure_wrapped_if_datetimelike

            # bc we know arr.dtype == object, this is equivalent to
            #  `np.asarray(to_timedelta(arr))`, but using a lower-level API that
            #  does not require a circular import.
            tdvals = array_to_timedelta64(arr).view("m8[ns]")

            tda = ensure_wrapped_if_datetimelike(tdvals)
            return tda.astype(dtype, copy=False)._ndarray

    if dtype.name in ("datetime64", "timedelta64"):
        msg = (
            f"The '{dtype.name}' dtype has no unit. Please pass in "
            f"'{dtype.name}[ns]' instead."
        )
        raise ValueError(msg)

    if copy or arr.dtype == object or dtype == object:
        # Explicit copy, or required since NumPy can't view from / to object.
        return arr.astype(dtype, copy=True)

    return arr.astype(dtype, copy=copy)


def _astype_float_to_int_nansafe(
    values: np.ndarray, dtype: np.dtype, copy: bool
) -> np.ndarray:
    """
    astype with a check preventing converting NaN to an meaningless integer value.
    """
    if not np.isfinite(values).all():
        raise IntCastingNaNError(
            "Cannot convert non-finite values (NA or inf) to integer"
        )
    if dtype.kind == "u":
        # GH#45151
        if not (values >= 0).all():
            raise ValueError(f"Cannot losslessly cast from {values.dtype} to {dtype}")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        return values.astype(dtype, copy=copy)


def astype_array(values: ArrayLike, dtype: DtypeObj, copy: bool = False) -> ArrayLike:
    """
    Cast array (ndarray or ExtensionArray) to the new dtype.

    Parameters
    ----------
    values : ndarray or ExtensionArray
    dtype : dtype object
    copy : bool, default False
        copy if indicated

    Returns
    -------
    ndarray or ExtensionArray
    """
    if values.dtype == dtype:
        if copy:
            return values.copy()
        return values

    if not isinstance(values, np.ndarray):
        # i.e. ExtensionArray
        values = values.astype(dtype, copy=copy)

    else:
        values = _astype_nansafe(values, dtype, copy=copy)

    # in pandas we don't store numpy str dtypes, so convert to object
    if isinstance(dtype, np.dtype) and issubclass(values.dtype.type, str):
        values = np.array(values, dtype=object)

    return values


def astype_array_safe(
    values: ArrayLike, dtype, copy: bool = False, errors: IgnoreRaise = "raise"
) -> ArrayLike:
    """
    Cast array (ndarray or ExtensionArray) to the new dtype.

    This basically is the implementation for DataFrame/Series.astype and
    includes all custom logic for pandas (NaN-safety, converting str to object,
    not allowing )

    Parameters
    ----------
    values : ndarray or ExtensionArray
    dtype : str, dtype convertible
    copy : bool, default False
        copy if indicated
    errors : str, {'raise', 'ignore'}, default 'raise'
        - ``raise`` : allow exceptions to be raised
        - ``ignore`` : suppress exceptions. On error return original object

    Returns
    -------
    ndarray or ExtensionArray
    """
    errors_legal_values = ("raise", "ignore")

    if errors not in errors_legal_values:
        invalid_arg = (
            "Expected value of kwarg 'errors' to be one of "
            f"{list(errors_legal_values)}. Supplied value is '{errors}'"
        )
        raise ValueError(invalid_arg)

    if inspect.isclass(dtype) and issubclass(dtype, ExtensionDtype):
        msg = (
            f"Expected an instance of {dtype.__name__}, "
            "but got the class instead. Try instantiating 'dtype'."
        )
        raise TypeError(msg)

    dtype = pandas_dtype(dtype)
    if isinstance(dtype, NumpyEADtype):
        # Ensure we don't end up with a NumpyExtensionArray
        dtype = dtype.numpy_dtype

    try:
        new_values = astype_array(values, dtype, copy=copy)
    except (ValueError, TypeError):
        # e.g. _astype_nansafe can fail on object-dtype of strings
        #  trying to convert to float
        if errors == "ignore":
            new_values = values
        else:
            raise

    return new_values


def astype_is_view(dtype: DtypeObj, new_dtype: DtypeObj) -> bool:
    """Checks if astype avoided copying the data.

    Parameters
    ----------
    dtype : Original dtype
    new_dtype : target dtype

    Returns
    -------
    True if new data is a view or not guaranteed to be a copy, False otherwise
    """
    if isinstance(dtype, np.dtype) and not isinstance(new_dtype, np.dtype):
        new_dtype, dtype = dtype, new_dtype

    if dtype == new_dtype:
        return True

    elif isinstance(dtype, np.dtype) and isinstance(new_dtype, np.dtype):
        # Only equal numpy dtypes avoid a copy
        return False

    elif is_string_dtype(dtype) and is_string_dtype(new_dtype):
        # Potentially! a view when converting from object to string
        return True

    elif is_object_dtype(dtype) and new_dtype.kind == "O":
        # When the underlying array has dtype object, we don't have to make a copy
        return True

    elif dtype.kind in "mM" and new_dtype.kind in "mM":
        dtype = getattr(dtype, "numpy_dtype", dtype)
        new_dtype = getattr(new_dtype, "numpy_dtype", new_dtype)
        return getattr(dtype, "unit", None) == getattr(new_dtype, "unit", None)

    numpy_dtype = getattr(dtype, "numpy_dtype", None)
    new_numpy_dtype = getattr(new_dtype, "numpy_dtype", None)

    if numpy_dtype is None and isinstance(dtype, np.dtype):
        numpy_dtype = dtype

    if new_numpy_dtype is None and isinstance(new_dtype, np.dtype):
        new_numpy_dtype = new_dtype

    if numpy_dtype is not None and new_numpy_dtype is not None:
        # if both have NumPy dtype or one of them is a numpy dtype
        # they are only a view when the numpy dtypes are equal, e.g.
        # int64 -> Int64 or int64[pyarrow]
        # int64 -> Int32 copies
        return numpy_dtype == new_numpy_dtype

    # Assume this is a view since we don't know for sure if a copy was made
    return True
<|endoftext|>"""
Common type operations.
"""
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
)
import warnings

import numpy as np

from pandas._config import using_string_dtype

from pandas._libs import (
    Interval,
    Period,
    algos,
    lib,
)
from pandas._libs.tslibs import conversion
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.base import _registry as registry
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    DatetimeTZDtype,
    ExtensionDtype,
    IntervalDtype,
    PeriodDtype,
    SparseDtype,
)
from pandas.core.dtypes.generic import ABCIndex
from pandas.core.dtypes.inference import (
    is_array_like,
    is_bool,
    is_complex,
    is_dataclass,
    is_decimal,
    is_dict_like,
    is_file_like,
    is_float,
    is_hashable,
    is_integer,
    is_interval,
    is_iterator,
    is_list_like,
    is_named_tuple,
    is_nested_list_like,
    is_number,
    is_re,
    is_re_compilable,
    is_scalar,
    is_sequence,
)

if TYPE_CHECKING:
    from pandas._typing import (
        ArrayLike,
        DtypeObj,
    )

DT64NS_DTYPE = conversion.DT64NS_DTYPE
TD64NS_DTYPE = conversion.TD64NS_DTYPE
INT64_DTYPE = np.dtype(np.int64)

# oh the troubles to reduce import time
_is_scipy_sparse = None

ensure_float64 = algos.ensure_float64
ensure_int64 = algos.ensure_int64
ensure_int32 = algos.ensure_int32
ensure_int16 = algos.ensure_int16
ensure_int8 = algos.ensure_int8
ensure_platform_int = algos.ensure_platform_int
ensure_object = algos.ensure_object
ensure_uint64 = algos.ensure_uint64


def ensure_str(value: bytes | Any) -> str:
    """
    Ensure that bytes and non-strings get converted into ``str`` objects.
    """
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    elif not isinstance(value, str):
        value = str(value)
    return value


def ensure_python_int(value: int | np.integer) -> int:
    """
    Ensure that a value is a python int.

    Parameters
    ----------
    value: int or numpy.integer

    Returns
    -------
    int

    Raises
    ------
    TypeError: if the value isn't an int or can't be converted to one.
    """
    if not (is_integer(value) or is_float(value)):
        if not is_scalar(value):
            raise TypeError(
                f"Value needs to be a scalar value, was type {type(value).__name__}"
            )
        raise TypeError(f"Wrong type {type(value)} for value {value}")
    try:
        new_value = int(value)
        assert new_value == value
    except (TypeError, ValueError, AssertionError) as err:
        raise TypeError(f"Wrong type {type(value)} for value {value}") from err
    return new_value


def classes(*klasses) -> Callable:
    """Evaluate if the tipo is a subclass of the klasses."""
    return lambda tipo: issubclass(tipo, klasses)


def _classes_and_not_datetimelike(*klasses) -> Callable:
    """
    Evaluate if the tipo is a subclass of the klasses
    and not a datetimelike.
    """
    return lambda tipo: (
        issubclass(tipo, klasses)
        and not issubclass(tipo, (np.datetime64, np.timedelta64))
    )


def is_object_dtype(arr_or_dtype) -> bool:
    """
    Check whether an array-like or dtype is of the object dtype.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array-like or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array-like or dtype is of the object dtype.

    Examples
    --------
    >>> from pandas.api.types import is_object_dtype
    >>> is_object_dtype(object)
    True
    >>> is_object_dtype(int)
    False
    >>> is_object_dtype(np.array([], dtype=object))
    True
    >>> is_object_dtype(np.array([], dtype=int))
    False
    >>> is_object_dtype([1, 2, 3])
    False
    """
    return _is_dtype_type(arr_or_dtype, classes(np.object_))


def is_sparse(arr) -> bool:
    """
    Check whether an array-like is a 1-D pandas sparse array.

    .. deprecated:: 2.1.0
        Use isinstance(dtype, pd.SparseDtype) instead.

    Check that the one-dimensional array-like is a pandas sparse array.
    Returns True if it is a pandas sparse array, not another type of
    sparse array.

    Parameters
    ----------
    arr : array-like
        Array-like to check.

    Returns
    -------
    bool
        Whether or not the array-like is a pandas sparse array.

    Examples
    --------
    Returns `True` if the parameter is a 1-D pandas sparse array.

    >>> from pandas.api.types import is_sparse
    >>> is_sparse(pd.arrays.SparseArray([0, 0, 1, 0]))
    True
    >>> is_sparse(pd.Series(pd.arrays.SparseArray([0, 0, 1, 0])))
    True

    Returns `False` if the parameter is not sparse.

    >>> is_sparse(np.array([0, 0, 1, 0]))
    False
    >>> is_sparse(pd.Series([0, 1, 0, 0]))
    False

    Returns `False` if the parameter is not a pandas sparse array.

    >>> from scipy.sparse import bsr_matrix
    >>> is_sparse(bsr_matrix([0, 1, 0, 0]))
    False

    Returns `False` if the parameter has more than one dimension.
    """
    warnings.warn(
        "is_sparse is deprecated and will be removed in a future "
        "version. Check `isinstance(dtype, pd.SparseDtype)` instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    dtype = getattr(arr, "dtype", arr)
    return isinstance(dtype, SparseDtype)


def is_scipy_sparse(arr) -> bool:
    """
    Check whether an array-like is a scipy.sparse.spmatrix instance.

    Parameters
    ----------
    arr : array-like
        The array-like to check.

    Returns
    -------
    boolean
        Whether or not the array-like is a scipy.sparse.spmatrix instance.

    Notes
    -----
    If scipy is not installed, this function will always return False.

    Examples
    --------
    >>> from scipy.sparse import bsr_matrix
    >>> is_scipy_sparse(bsr_matrix([1, 2, 3]))
    True
    >>> is_scipy_sparse(pd.arrays.SparseArray([1, 2, 3]))
    False
    """
    global _is_scipy_sparse

    if _is_scipy_sparse is None:  # pylint: disable=used-before-assignment
        try:
            from scipy.sparse import issparse as _is_scipy_sparse
        except ImportError:
            _is_scipy_sparse = lambda _: False

    assert _is_scipy_sparse is not None
    return _is_scipy_sparse(arr)


def is_datetime64_dtype(arr_or_dtype) -> bool:
    """
    Check whether an array-like or dtype is of the datetime64 dtype.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array-like or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array-like or dtype is of the datetime64 dtype.

    Examples
    --------
    >>> from pandas.api.types import is_datetime64_dtype
    >>> is_datetime64_dtype(object)
    False
    >>> is_datetime64_dtype(np.datetime64)
    True
    >>> is_datetime64_dtype(np.array([], dtype=int))
    False
    >>> is_datetime64_dtype(np.array([], dtype=np.datetime64))
    True
    >>> is_datetime64_dtype([1, 2, 3])
    False
    """
    if isinstance(arr_or_dtype, np.dtype):
        # GH#33400 fastpath for dtype object
        return arr_or_dtype.kind == "M"
    return _is_dtype_type(arr_or_dtype, classes(np.datetime64))


def is_datetime64tz_dtype(arr_or_dtype) -> bool:
    """
    Check whether an array-like or dtype is of a DatetimeTZDtype dtype.

    .. deprecated:: 2.1.0
        Use isinstance(dtype, pd.DatetimeTZDtype) instead.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array-like or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array-like or dtype is of a DatetimeTZDtype dtype.

    Examples
    --------
    >>> from pandas.api.types import is_datetime64tz_dtype
    >>> is_datetime64tz_dtype(object)
    False
    >>> is_datetime64tz_dtype([1, 2, 3])
    False
    >>> is_datetime64tz_dtype(pd.DatetimeIndex([1, 2, 3]))  # tz-naive
    False
    >>> is_datetime64tz_dtype(pd.DatetimeIndex([1, 2, 3], tz="US/Eastern"))
    True

    >>> from pandas.core.dtypes.dtypes import DatetimeTZDtype
    >>> dtype = DatetimeTZDtype("ns", tz="US/Eastern")
    >>> s = pd.Series([], dtype=dtype)
    >>> is_datetime64tz_dtype(dtype)
    True
    >>> is_datetime64tz_dtype(s)
    True
    """
    # GH#52607
    warnings.warn(
        "is_datetime64tz_dtype is deprecated and will be removed in a future "
        "version. Check `isinstance(dtype, pd.DatetimeTZDtype)` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if isinstance(arr_or_dtype, DatetimeTZDtype):
        # GH#33400 fastpath for dtype object
        # GH 34986
        return True

    if arr_or_dtype is None:
        return False
    return DatetimeTZDtype.is_dtype(arr_or_dtype)


def is_timedelta64_dtype(arr_or_dtype) -> bool:
    """
    Check whether an array-like or dtype is of the timedelta64 dtype.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array-like or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array-like or dtype is of the timedelta64 dtype.

    Examples
    --------
    >>> from pandas.core.dtypes.common import is_timedelta64_dtype
    >>> is_timedelta64_dtype(object)
    False
    >>> is_timedelta64_dtype(np.timedelta64)
    True
    >>> is_timedelta64_dtype([1, 2, 3])
    False
    >>> is_timedelta64_dtype(pd.Series([], dtype="timedelta64[ns]"))
    True
    >>> is_timedelta64_dtype('0 days')
    False
    """
    if isinstance(arr_or_dtype, np.dtype):
        # GH#33400 fastpath for dtype object
        return arr_or_dtype.kind == "m"

    return _is_dtype_type(arr_or_dtype, classes(np.timedelta64))


def is_period_dtype(arr_or_dtype) -> bool:
    """
    Check whether an array-like or dtype is of the Period dtype.

    .. deprecated:: 2.2.0
        Use isinstance(dtype, pd.Period) instead.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array-like or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array-like or dtype is of the Period dtype.

    Examples
    --------
    >>> from pandas.core.dtypes.common import is_period_dtype
    >>> is_period_dtype(object)
    False
    >>> is_period_dtype(pd.PeriodDtype(freq="D"))
    True
    >>> is_period_dtype([1, 2, 3])
    False
    >>> is_period_dtype(pd.Period("2017-01-01"))
    False
    >>> is_period_dtype(pd.PeriodIndex([], freq="Y"))
    True
    """
    warnings.warn(
        "is_period_dtype is deprecated and will be removed in a future version. "
        "Use `isinstance(dtype, pd.PeriodDtype)` instead",
        DeprecationWarning,
        stacklevel=2,
    )
    if isinstance(arr_or_dtype, ExtensionDtype):
        # GH#33400 fastpath for dtype object
        return arr_or_dtype.type is Period

    if arr_or_dtype is None:
        return False
    return PeriodDtype.is_dtype(arr_or_dtype)


def is_interval_dtype(arr_or_dtype) -> bool:
    """
    Check whether an array-like or dtype is of the Interval dtype.

    .. deprecated:: 2.2.0
        Use isinstance(dtype, pd.IntervalDtype) instead.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array-like or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array-like or dtype is of the Interval dtype.

    Examples
    --------
    >>> from pandas.core.dtypes.common import is_interval_dtype
    >>> is_interval_dtype(object)
    False
    >>> is_interval_dtype(pd.IntervalDtype())
    True
    >>> is_interval_dtype([1, 2, 3])
    False
    >>>
    >>> interval = pd.Interval(1, 2, closed="right")
    >>> is_interval_dtype(interval)
    False
    >>> is_interval_dtype(pd.IntervalIndex([interval]))
    True
    """
    # GH#52607
    warnings.warn(
        "is_interval_dtype is deprecated and will be removed in a future version. "
        "Use `isinstance(dtype, pd.IntervalDtype)` instead",
        DeprecationWarning,
        stacklevel=2,
    )
    if isinstance(arr_or_dtype, ExtensionDtype):
        # GH#33400 fastpath for dtype object
        return arr_or_dtype.type is Interval

    if arr_or_dtype is None:
        return False
    return IntervalDtype.is_dtype(arr_or_dtype)


def is_categorical_dtype(arr_or_dtype) -> bool:
    """
    Check whether an array-like or dtype is of the Categorical dtype.

    .. deprecated:: 2.2.0
        Use isinstance(dtype, pd.CategoricalDtype) instead.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array-like or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array-like or dtype is of the Categorical dtype.

    Examples
    --------
    >>> from pandas.api.types import is_categorical_dtype
    >>> from pandas import CategoricalDtype
    >>> is_categorical_dtype(object)
    False
    >>> is_categorical_dtype(CategoricalDtype())
    True
    >>> is_categorical_dtype([1, 2, 3])
    False
    >>> is_categorical_dtype(pd.Categorical([1, 2, 3]))
    True
    >>> is_categorical_dtype(pd.CategoricalIndex([1, 2, 3]))
    True
    """
    # GH#52527
    warnings.warn(
        "is_categorical_dtype is deprecated and will be removed in a future "
        "version. Use isinstance(dtype, pd.CategoricalDtype) instead",
        DeprecationWarning,
        stacklevel=2,
    )
    if isinstance(arr_or_dtype, ExtensionDtype):
        # GH#33400 fastpath for dtype object
        return arr_or_dtype.name == "category"

    if arr_or_dtype is None:
        return False
    return CategoricalDtype.is_dtype(arr_or_dtype)


def is_string_or_object_np_dtype(dtype: np.dtype) -> bool:
    """
    Faster alternative to is_string_dtype, assumes we have a np.dtype object.
    """
    return dtype == object or dtype.kind in "SU"


def is_string_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of the string dtype.

    If an array is passed with an object dtype, the elements must be
    inferred as strings.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of the string dtype.

    Examples
    --------
    >>> from pandas.api.types import is_string_dtype
    >>> is_string_dtype(str)
    True
    >>> is_string_dtype(object)
    True
    >>> is_string_dtype(int)
    False
    >>> is_string_dtype(np.array(['a', 'b']))
    True
    >>> is_string_dtype(pd.Series([1, 2]))
    False
    >>> is_string_dtype(pd.Series([1, 2], dtype=object))
    False
    """
    if hasattr(arr_or_dtype, "dtype") and _get_dtype(arr_or_dtype).kind == "O":
        return is_all_strings(arr_or_dtype)

    def condition(dtype) -> bool:
        if is_string_or_object_np_dtype(dtype):
            return True
        try:
            return dtype == "string"
        except TypeError:
            return False

    return _is_dtype(arr_or_dtype, condition)


def is_dtype_equal(source, target) -> bool:
    """
    Check if two dtypes are equal.

    Parameters
    ----------
    source : The first dtype to compare
    target : The second dtype to compare

    Returns
    -------
    boolean
        Whether or not the two dtypes are equal.

    Examples
    --------
    >>> is_dtype_equal(int, float)
    False
    >>> is_dtype_equal("int", int)
    True
    >>> is_dtype_equal(object, "category")
    False
    >>> is_dtype_equal(CategoricalDtype(), "category")
    True
    >>> is_dtype_equal(DatetimeTZDtype(tz="UTC"), "datetime64")
    False
    """
    if isinstance(target, str):
        if not isinstance(source, str):
            # GH#38516 ensure we get the same behavior from
            #  is_dtype_equal(CDT, "category") and CDT == "category"
            try:
                src = _get_dtype(source)
                if isinstance(src, ExtensionDtype):
                    return src == target
            except (TypeError, AttributeError, ImportError):
                return False
    elif isinstance(source, str):
        return is_dtype_equal(target, source)

    try:
        source = _get_dtype(source)
        target = _get_dtype(target)
        return source == target
    except (TypeError, AttributeError, ImportError):
        # invalid comparison
        # object == category will hit this
        return False


def is_integer_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of an integer dtype.

    Unlike in `is_any_int_dtype`, timedelta64 instances will return False.

    The nullable Integer dtypes (e.g. pandas.Int64Dtype) are also considered
    as integer by this function.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of an integer dtype and
        not an instance of timedelta64.

    Examples
    --------
    >>> from pandas.api.types import is_integer_dtype
    >>> is_integer_dtype(str)
    False
    >>> is_integer_dtype(int)
    True
    >>> is_integer_dtype(float)
    False
    >>> is_integer_dtype(np.uint64)
    True
    >>> is_integer_dtype('int8')
    True
    >>> is_integer_dtype('Int8')
    True
    >>> is_integer_dtype(pd.Int8Dtype)
    True
    >>> is_integer_dtype(np.datetime64)
    False
    >>> is_integer_dtype(np.timedelta64)
    False
    >>> is_integer_dtype(np.array(['a', 'b']))
    False
    >>> is_integer_dtype(pd.Series([1, 2]))
    True
    >>> is_integer_dtype(np.array([], dtype=np.timedelta64))
    False
    >>> is_integer_dtype(pd.Index([1, 2.]))  # float
    False
    """
    return _is_dtype_type(
        arr_or_dtype, _classes_and_not_datetimelike(np.integer)
    ) or _is_dtype(
        arr_or_dtype, lambda typ: isinstance(typ, ExtensionDtype) and typ.kind in "iu"
    )


def is_signed_integer_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of a signed integer dtype.

    Unlike in `is_any_int_dtype`, timedelta64 instances will return False.

    The nullable Integer dtypes (e.g. pandas.Int64Dtype) are also considered
    as integer by this function.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of a signed integer dtype
        and not an instance of timedelta64.

    Examples
    --------
    >>> from pandas.core.dtypes.common import is_signed_integer_dtype
    >>> is_signed_integer_dtype(str)
    False
    >>> is_signed_integer_dtype(int)
    True
    >>> is_signed_integer_dtype(float)
    False
    >>> is_signed_integer_dtype(np.uint64)  # unsigned
    False
    >>> is_signed_integer_dtype('int8')
    True
    >>> is_signed_integer_dtype('Int8')
    True
    >>> is_signed_integer_dtype(pd.Int8Dtype)
    True
    >>> is_signed_integer_dtype(np.datetime64)
    False
    >>> is_signed_integer_dtype(np.timedelta64)
    False
    >>> is_signed_integer_dtype(np.array(['a', 'b']))
    False
    >>> is_signed_integer_dtype(pd.Series([1, 2]))
    True
    >>> is_signed_integer_dtype(np.array([], dtype=np.timedelta64))
    False
    >>> is_signed_integer_dtype(pd.Index([1, 2.]))  # float
    False
    >>> is_signed_integer_dtype(np.array([1, 2], dtype=np.uint32))  # unsigned
    False
    """
    return _is_dtype_type(
        arr_or_dtype, _classes_and_not_datetimelike(np.signedinteger)
    ) or _is_dtype(
        arr_or_dtype, lambda typ: isinstance(typ, ExtensionDtype) and typ.kind == "i"
    )


def is_unsigned_integer_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of an unsigned integer dtype.

    The nullable Integer dtypes (e.g. pandas.UInt64Dtype) are also
    considered as integer by this function.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of an unsigned integer dtype.

    Examples
    --------
    >>> from pandas.api.types import is_unsigned_integer_dtype
    >>> is_unsigned_integer_dtype(str)
    False
    >>> is_unsigned_integer_dtype(int)  # signed
    False
    >>> is_unsigned_integer_dtype(float)
    False
    >>> is_unsigned_integer_dtype(np.uint64)
    True
    >>> is_unsigned_integer_dtype('uint8')
    True
    >>> is_unsigned_integer_dtype('UInt8')
    True
    >>> is_unsigned_integer_dtype(pd.UInt8Dtype)
    True
    >>> is_unsigned_integer_dtype(np.array(['a', 'b']))
    False
    >>> is_unsigned_integer_dtype(pd.Series([1, 2]))  # signed
    False
    >>> is_unsigned_integer_dtype(pd.Index([1, 2.]))  # float
    False
    >>> is_unsigned_integer_dtype(np.array([1, 2], dtype=np.uint32))
    True
    """
    return _is_dtype_type(
        arr_or_dtype, _classes_and_not_datetimelike(np.unsignedinteger)
    ) or _is_dtype(
        arr_or_dtype, lambda typ: isinstance(typ, ExtensionDtype) and typ.kind == "u"
    )


def is_int64_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of the int64 dtype.

    .. deprecated:: 2.1.0

       is_int64_dtype is deprecated and will be removed in a future
       version. Use dtype == np.int64 instead.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of the int64 dtype.

    Notes
    -----
    Depending on system architecture, the return value of `is_int64_dtype(
    int)` will be True if the OS uses 64-bit integers and False if the OS
    uses 32-bit integers.

    Examples
    --------
    >>> from pandas.api.types import is_int64_dtype
    >>> is_int64_dtype(str)  # doctest: +SKIP
    False
    >>> is_int64_dtype(np.int32)  # doctest: +SKIP
    False
    >>> is_int64_dtype(np.int64)  # doctest: +SKIP
    True
    >>> is_int64_dtype('int8')  # doctest: +SKIP
    False
    >>> is_int64_dtype('Int8')  # doctest: +SKIP
    False
    >>> is_int64_dtype(pd.Int64Dtype)  # doctest: +SKIP
    True
    >>> is_int64_dtype(float)  # doctest: +SKIP
    False
    >>> is_int64_dtype(np.uint64)  # unsigned  # doctest: +SKIP
    False
    >>> is_int64_dtype(np.array(['a', 'b']))  # doctest: +SKIP
    False
    >>> is_int64_dtype(np.array([1, 2], dtype=np.int64))  # doctest: +SKIP
    True
    >>> is_int64_dtype(pd.Index([1, 2.]))  # float  # doctest: +SKIP
    False
    >>> is_int64_dtype(np.array([1, 2], dtype=np.uint32))  # unsigned  # doctest: +SKIP
    False
    """
    # GH#52564
    warnings.warn(
        "is_int64_dtype is deprecated and will be removed in a future "
        "version. Use dtype == np.int64 instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _is_dtype_type(arr_or_dtype, classes(np.int64))


def is_datetime64_any_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of the datetime64 dtype.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array or dtype to check.

    Returns
    -------
    bool
        Whether or not the array or dtype is of the datetime64 dtype.

    Examples
    --------
    >>> from pandas.api.types import is_datetime64_any_dtype
    >>> from pandas.core.dtypes.dtypes import DatetimeTZDtype
    >>> is_datetime64_any_dtype(str)
    False
    >>> is_datetime64_any_dtype(int)
    False
    >>> is_datetime64_any_dtype(np.datetime64)  # can be tz-naive
    True
    >>> is_datetime64_any_dtype(DatetimeTZDtype("ns", "US/Eastern"))
    True
    >>> is_datetime64_any_dtype(np.array(['a', 'b']))
    False
    >>> is_datetime64_any_dtype(np.array([1, 2]))
    False
    >>> is_datetime64_any_dtype(np.array([], dtype="datetime64[ns]"))
    True
    >>> is_datetime64_any_dtype(pd.DatetimeIndex([1, 2, 3], dtype="datetime64[ns]"))
    True
    """
    if isinstance(arr_or_dtype, (np.dtype, ExtensionDtype)):
        # GH#33400 fastpath for dtype object
        return arr_or_dtype.kind == "M"

    if arr_or_dtype is None:
        return False

    try:
        tipo = _get_dtype(arr_or_dtype)
    except TypeError:
        return False
    return lib.is_np_dtype(tipo, "M") or isinstance(tipo, DatetimeTZDtype)


def is_datetime64_ns_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of the datetime64[ns] dtype.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array or dtype to check.

    Returns
    -------
    bool
        Whether or not the array or dtype is of the datetime64[ns] dtype.

    Examples
    --------
    >>> from pandas.api.types import is_datetime64_ns_dtype
    >>> from pandas.core.dtypes.dtypes import DatetimeTZDtype
    >>> is_datetime64_ns_dtype(str)
    False
    >>> is_datetime64_ns_dtype(int)
    False
    >>> is_datetime64_ns_dtype(np.datetime64)  # no unit
    False
    >>> is_datetime64_ns_dtype(DatetimeTZDtype("ns", "US/Eastern"))
    True
    >>> is_datetime64_ns_dtype(np.array(['a', 'b']))
    False
    >>> is_datetime64_ns_dtype(np.array([1, 2]))
    False
    >>> is_datetime64_ns_dtype(np.array([], dtype="datetime64"))  # no unit
    False
    >>> is_datetime64_ns_dtype(np.array([], dtype="datetime64[ps]"))  # wrong unit
    False
    >>> is_datetime64_ns_dtype(pd.DatetimeIndex([1, 2, 3], dtype="datetime64[ns]"))
    True
    """
    if arr_or_dtype is None:
        return False
    try:
        tipo = _get_dtype(arr_or_dtype)
    except TypeError:
        return False
    return tipo == DT64NS_DTYPE or (
        isinstance(tipo, DatetimeTZDtype) and tipo.unit == "ns"
    )


def is_timedelta64_ns_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of the timedelta64[ns] dtype.

    This is a very specific dtype, so generic ones like `np.timedelta64`
    will return False if passed into this function.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of the timedelta64[ns] dtype.

    Examples
    --------
    >>> from pandas.core.dtypes.common import is_timedelta64_ns_dtype
    >>> is_timedelta64_ns_dtype(np.dtype('m8[ns]'))
    True
    >>> is_timedelta64_ns_dtype(np.dtype('m8[ps]'))  # Wrong frequency
    False
    >>> is_timedelta64_ns_dtype(np.array([1, 2], dtype='m8[ns]'))
    True
    >>> is_timedelta64_ns_dtype(np.array([1, 2], dtype=np.timedelta64))
    False
    """
    return _is_dtype(arr_or_dtype, lambda dtype: dtype == TD64NS_DTYPE)


# This exists to silence numpy deprecation warnings, see GH#29553
def is_numeric_v_string_like(a: ArrayLike, b) -> bool:
    """
    Check if we are comparing a string-like object to a numeric ndarray.
    NumPy doesn't like to compare such objects, especially numeric arrays
    and scalar string-likes.

    Parameters
    ----------
    a : array-like, scalar
        The first object to check.
    b : array-like, scalar
        The second object to check.

    Returns
    -------
    boolean
        Whether we return a comparing a string-like object to a numeric array.

    Examples
    --------
    >>> is_numeric_v_string_like(np.array([1]), "foo")
    True
    >>> is_numeric_v_string_like(np.array([1, 2]), np.array(["foo"]))
    True
    >>> is_numeric_v_string_like(np.array(["foo"]), np.array([1, 2]))
    True
    >>> is_numeric_v_string_like(np.array([1]), np.array([2]))
    False
    >>> is_numeric_v_string_like(np.array(["foo"]), np.array(["foo"]))
    False
    """
    is_a_array = isinstance(a, np.ndarray)
    is_b_array = isinstance(b, np.ndarray)

    is_a_numeric_array = is_a_array and a.dtype.kind in ("u", "i", "f", "c", "b")
    is_b_numeric_array = is_b_array and b.dtype.kind in ("u", "i", "f", "c", "b")
    is_a_string_array = is_a_array and a.dtype.kind in ("S", "U")
    is_b_string_array = is_b_array and b.dtype.kind in ("S", "U")

    is_b_scalar_string_like = not is_b_array and isinstance(b, str)

    return (
        (is_a_numeric_array and is_b_scalar_string_like)
        or (is_a_numeric_array and is_b_string_array)
        or (is_b_numeric_array and is_a_string_array)
    )


def needs_i8_conversion(dtype: DtypeObj | None) -> bool:
    """
    Check whether the dtype should be converted to int64.

    Dtype "needs" such a conversion if the dtype is of a datetime-like dtype

    Parameters
    ----------
    dtype : np.dtype, ExtensionDtype, or None

    Returns
    -------
    boolean
        Whether or not the dtype should be converted to int64.

    Examples
    --------
    >>> needs_i8_conversion(str)
    False
    >>> needs_i8_conversion(np.int64)
    False
    >>> needs_i8_conversion(np.datetime64)
    False
    >>> needs_i8_conversion(np.dtype(np.datetime64))
    True
    >>> needs_i8_conversion(np.array(['a', 'b']))
    False
    >>> needs_i8_conversion(pd.Series([1, 2]))
    False
    >>> needs_i8_conversion(pd.Series([], dtype="timedelta64[ns]"))
    False
    >>> needs_i8_conversion(pd.DatetimeIndex([1, 2, 3], tz="US/Eastern"))
    False
    >>> needs_i8_conversion(pd.DatetimeIndex([1, 2, 3], tz="US/Eastern").dtype)
    True
    """
    if isinstance(dtype, np.dtype):
        return dtype.kind in "mM"
    return isinstance(dtype, (PeriodDtype, DatetimeTZDtype))


def is_numeric_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of a numeric dtype.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of a numeric dtype.

    Examples
    --------
    >>> from pandas.api.types import is_numeric_dtype
    >>> is_numeric_dtype(str)
    False
    >>> is_numeric_dtype(int)
    True
    >>> is_numeric_dtype(float)
    True
    >>> is_numeric_dtype(np.uint64)
    True
    >>> is_numeric_dtype(np.datetime64)
    False
    >>> is_numeric_dtype(np.timedelta64)
    False
    >>> is_numeric_dtype(np.array(['a', 'b']))
    False
    >>> is_numeric_dtype(pd.Series([1, 2]))
    True
    >>> is_numeric_dtype(pd.Index([1, 2.]))
    True
    >>> is_numeric_dtype(np.array([], dtype=np.timedelta64))
    False
    """
    return _is_dtype_type(
        arr_or_dtype, _classes_and_not_datetimelike(np.number, np.bool_)
    ) or _is_dtype(
        arr_or_dtype, lambda typ: isinstance(typ, ExtensionDtype) and typ._is_numeric
    )


def is_any_real_numeric_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of a real number dtype.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of a real number dtype.

    Examples
    --------
    >>> from pandas.api.types import is_any_real_numeric_dtype
    >>> is_any_real_numeric_dtype(int)
    True
    >>> is_any_real_numeric_dtype(float)
    True
    >>> is_any_real_numeric_dtype(object)
    False
    >>> is_any_real_numeric_dtype(str)
    False
    >>> is_any_real_numeric_dtype(complex(1, 2))
    False
    >>> is_any_real_numeric_dtype(bool)
    False
    """
    return (
        is_numeric_dtype(arr_or_dtype)
        and not is_complex_dtype(arr_or_dtype)
        and not is_bool_dtype(arr_or_dtype)
    )


def is_float_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of a float dtype.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of a float dtype.

    Examples
    --------
    >>> from pandas.api.types import is_float_dtype
    >>> is_float_dtype(str)
    False
    >>> is_float_dtype(int)
    False
    >>> is_float_dtype(float)
    True
    >>> is_float_dtype(np.array(['a', 'b']))
    False
    >>> is_float_dtype(pd.Series([1, 2]))
    False
    >>> is_float_dtype(pd.Index([1, 2.]))
    True
    """
    return _is_dtype_type(arr_or_dtype, classes(np.floating)) or _is_dtype(
        arr_or_dtype, lambda typ: isinstance(typ, ExtensionDtype) and typ.kind in "f"
    )


def is_bool_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of a boolean dtype.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of a boolean dtype.

    Notes
    -----
    An ExtensionArray is considered boolean when the ``_is_boolean``
    attribute is set to True.

    Examples
    --------
    >>> from pandas.api.types import is_bool_dtype
    >>> is_bool_dtype(str)
    False
    >>> is_bool_dtype(int)
    False
    >>> is_bool_dtype(bool)
    True
    >>> is_bool_dtype(np.bool_)
    True
    >>> is_bool_dtype(np.array(['a', 'b']))
    False
    >>> is_bool_dtype(pd.Series([1, 2]))
    False
    >>> is_bool_dtype(np.array([True, False]))
    True
    >>> is_bool_dtype(pd.Categorical([True, False]))
    True
    >>> is_bool_dtype(pd.arrays.SparseArray([True, False]))
    True
    """
    if arr_or_dtype is None:
        return False
    try:
        dtype = _get_dtype(arr_or_dtype)
    except (TypeError, ValueError):
        return False

    if isinstance(dtype, CategoricalDtype):
        arr_or_dtype = dtype.categories
        # now we use the special definition for Index

    if isinstance(arr_or_dtype, ABCIndex):
        # Allow Index[object] that is all-bools or Index["boolean"]
        if arr_or_dtype.inferred_type == "boolean":
            if not is_bool_dtype(arr_or_dtype.dtype):
                # GH#52680
                warnings.warn(
                    "The behavior of is_bool_dtype with an object-dtype Index "
                    "of bool objects is deprecated. In a future version, "
                    "this will return False. Cast the Index to a bool dtype instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            return True
        return False
    elif isinstance(dtype, ExtensionDtype):
        return getattr(dtype, "_is_boolean", False)

    return issubclass(dtype.type, np.bool_)


def is_1d_only_ea_dtype(dtype: DtypeObj | None) -> bool:
    """
    Analogue to is_extension_array_dtype but excluding DatetimeTZDtype.
    """
    return isinstance(dtype, ExtensionDtype) and not dtype._supports_2d


def is_extension_array_dtype(arr_or_dtype) -> bool:
    """
    Check if an object is a pandas extension array type.

    See the :ref:`Use Guide <extending.extension-types>` for more.

    Parameters
    ----------
    arr_or_dtype : object
        For array-like input, the ``.dtype`` attribute will
        be extracted.

    Returns
    -------
    bool
        Whether the `arr_or_dtype` is an extension array type.

    Notes
    -----
    This checks whether an object implements the pandas extension
    array interface. In pandas, this includes:

    * Categorical
    * Sparse
    * Interval
    * Period
    * DatetimeArray
    * TimedeltaArray

    Third-party libraries may implement arrays or types satisfying
    this interface as well.

    Examples
    --------
    >>> from pandas.api.types import is_extension_array_dtype
    >>> arr = pd.Categorical(['a', 'b'])
    >>> is_extension_array_dtype(arr)
    True
    >>> is_extension_array_dtype(arr.dtype)
    True

    >>> arr = np.array(['a', 'b'])
    >>> is_extension_array_dtype(arr.dtype)
    False
    """
    dtype = getattr(arr_or_dtype, "dtype", arr_or_dtype)
    if isinstance(dtype, ExtensionDtype):
        return True
    elif isinstance(dtype, np.dtype):
        return False
    else:
        try:
            with warnings.catch_warnings():
                # pandas_dtype(..) can raise UserWarning for class input
                warnings.simplefilter("ignore", UserWarning)
                dtype = pandas_dtype(dtype)
        except (TypeError, ValueError):
            # np.dtype(..) can raise ValueError
            return False
        return isinstance(dtype, ExtensionDtype)


def is_ea_or_datetimelike_dtype(dtype: DtypeObj | None) -> bool:
    """
    Check for ExtensionDtype, datetime64 dtype, or timedelta64 dtype.

    Notes
    -----
    Checks only for dtype objects, not dtype-castable strings or types.
    """
    return isinstance(dtype, ExtensionDtype) or (lib.is_np_dtype(dtype, "mM"))


def is_complex_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of a complex dtype.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of a complex dtype.

    Examples
    --------
    >>> from pandas.api.types import is_complex_dtype
    >>> is_complex_dtype(str)
    False
    >>> is_complex_dtype(int)
    False
    >>> is_complex_dtype(np.complex128)
    True
    >>> is_complex_dtype(np.array(['a', 'b']))
    False
    >>> is_complex_dtype(pd.Series([1, 2]))
    False
    >>> is_complex_dtype(np.array([1 + 1j, 5]))
    True
    """
    return _is_dtype_type(arr_or_dtype, classes(np.complexfloating))


def _is_dtype(arr_or_dtype, condition) -> bool:
    """
    Return true if the condition is satisfied for the arr_or_dtype.

    Parameters
    ----------
    arr_or_dtype : array-like, str, np.dtype, or ExtensionArrayType
        The array-like or dtype object whose dtype we want to extract.
    condition : callable[Union[np.dtype, ExtensionDtype]]

    Returns
    -------
    bool

    """
    if arr_or_dtype is None:
        return False
    try:
        dtype = _get_dtype(arr_or_dtype)
    except (TypeError, ValueError):
        return False
    return condition(dtype)


def _get_dtype(arr_or_dtype) -> DtypeObj:
    """
    Get the dtype instance associated with an array
    or dtype object.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array-like or dtype object whose dtype we want to extract.

    Returns
    -------
    obj_dtype : The extract dtype instance from the
                passed in array or dtype object.

    Raises
    ------
    TypeError : The passed in object is None.
    """
    if arr_or_dtype is None:
        raise TypeError("Cannot deduce dtype from null object")

    # fastpath
    if isinstance(arr_or_dtype, np.dtype):
        return arr_or_dtype
    elif isinstance(arr_or_dtype, type):
        return np.dtype(arr_or_dtype)

    # if we have an array-like
    elif hasattr(arr_or_dtype, "dtype"):
        arr_or_dtype = arr_or_dtype.dtype

    return pandas_dtype(arr_or_dtype)


def _is_dtype_type(arr_or_dtype, condition) -> bool:
    """
    Return true if the condition is satisfied for the arr_or_dtype.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array-like or dtype object whose dtype we want to extract.
    condition : callable[Union[np.dtype, ExtensionDtypeType]]

    Returns
    -------
    bool : if the condition is satisfied for the arr_or_dtype
    """
    if arr_or_dtype is None:
        return condition(type(None))

    # fastpath
    if isinstance(arr_or_dtype, np.dtype):
        return condition(arr_or_dtype.type)
    elif isinstance(arr_or_dtype, type):
        if issubclass(arr_or_dtype, ExtensionDtype):
            arr_or_dtype = arr_or_dtype.type
        return condition(np.dtype(arr_or_dtype).type)

    # if we have an array-like
    if hasattr(arr_or_dtype, "dtype"):
        arr_or_dtype = arr_or_dtype.dtype

    # we are not possibly a dtype
    elif is_list_like(arr_or_dtype):
        return condition(type(None))

    try:
        tipo = pandas_dtype(arr_or_dtype).type
    except (TypeError, ValueError):
        if is_scalar(arr_or_dtype):
            return condition(type(None))

        return False

    return condition(tipo)


def infer_dtype_from_object(dtype) -> type:
    """
    Get a numpy dtype.type-style object for a dtype object.

    This methods also includes handling of the datetime64[ns] and
    datetime64[ns, TZ] objects.

    If no dtype can be found, we return ``object``.

    Parameters
    ----------
    dtype : dtype, type
        The dtype object whose numpy dtype.type-style
        object we want to extract.

    Returns
    -------
    type
    """
    if isinstance(dtype, type) and issubclass(dtype, np.generic):
        # Type object from a dtype

        return dtype
    elif isinstance(dtype, (np.dtype, ExtensionDtype)):
        # dtype object
        try:
            _validate_date_like_dtype(dtype)
        except TypeError:
            # Should still pass if we don't have a date-like
            pass
        if hasattr(dtype, "numpy_dtype"):
            # TODO: Implement this properly
            # https://github.com/pandas-dev/pandas/issues/52576
            return dtype.numpy_dtype.type
        return dtype.type

    try:
        dtype = pandas_dtype(dtype)
    except TypeError:
        pass

    if isinstance(dtype, ExtensionDtype):
        return dtype.type
    elif isinstance(dtype, str):
        # TODO(jreback)
        # should deprecate these
        if dtype in ["datetimetz", "datetime64tz"]:
            return DatetimeTZDtype.type
        elif dtype in ["period"]:
            raise NotImplementedError

        if dtype in ["datetime", "timedelta"]:
            dtype += "64"
        try:
            return infer_dtype_from_object(getattr(np, dtype))
        except (AttributeError, TypeError):
            # Handles cases like _get_dtype(int) i.e.,
            # Python objects that are valid dtypes
            # (unlike user-defined types, in general)
            #
            # TypeError handles the float16 type code of 'e'
            # further handle internal types
            pass

    return infer_dtype_from_object(np.dtype(dtype))


def _validate_date_like_dtype(dtype) -> None:
    """
    Check whether the dtype is a date-like dtype. Raises an error if invalid.

    Parameters
    ----------
    dtype : dtype, type
        The dtype to check.

    Raises
    ------
    TypeError : The dtype could not be casted to a date-like dtype.
    ValueError : The dtype is an illegal date-like dtype (e.g. the
                 frequency provided is too specific)
    """
    try:
        typ = np.datetime_data(dtype)[0]
    except ValueError as e:
        raise TypeError(e) from e
    if typ not in ["generic", "ns"]:
        raise ValueError(
            f"{repr(dtype.name)} is too specific of a frequency, "
            f"try passing {repr(dtype.type.__name__)}"
        )


def validate_all_hashable(*args, error_name: str | None = None) -> None:
    """
    Return None if all args are hashable, else raise a TypeError.

    Parameters
    ----------
    *args
        Arguments to validate.
    error_name : str, optional
        The name to use if error

    Raises
    ------
    TypeError : If an argument is not hashable

    Returns
    -------
    None
    """
    if not all(is_hashable(arg) for arg in args):
        if error_name:
            raise TypeError(f"{error_name} must be a hashable type")
        raise TypeError("All elements must be hashable")


def pandas_dtype(dtype) -> DtypeObj:
    """
    Convert input into a pandas only dtype object or a numpy dtype object.

    Parameters
    ----------
    dtype : object to be converted

    Returns
    -------
    np.dtype or a pandas dtype

    Raises
    ------
    TypeError if not a dtype

    Examples
    --------
    >>> pd.api.types.pandas_dtype(int)
    dtype('int64')
    """
    # short-circuit
    if isinstance(dtype, np.ndarray):
        return dtype.dtype
    elif isinstance(dtype, (np.dtype, ExtensionDtype)):
        return dtype

    # builtin aliases
    if dtype is str and using_string_dtype():
        from pandas.core.arrays.string_ import StringDtype

        return StringDtype(na_value=np.nan)

    # registered extension types
    result = registry.find(dtype)
    if result is not None:
        if isinstance(result, type):
            # GH 31356, GH 54592
            warnings.warn(
                f"Instantiating {result.__name__} without any arguments."
                f"Pass a {result.__name__} instance to silence this warning.",
                UserWarning,
                stacklevel=find_stack_level(),
            )
            result = result()
        return result

    # try a numpy dtype
    # raise a consistent TypeError if failed
    try:
        with warnings.catch_warnings():
            # TODO: warnings.catch_warnings can be removed when numpy>2.3.0
            # is the minimum version
            # GH#51523 - Series.astype(np.integer) doesn't show
            # numpy deprecation warning of np.integer
            # Hence enabling DeprecationWarning
            warnings.simplefilter("always", DeprecationWarning)
            npdtype = np.dtype(dtype)
    except SyntaxError as err:
        # np.dtype uses `eval` which can raise SyntaxError
        raise TypeError(f"data type '{dtype}' not understood") from err

    # Any invalid dtype (such as pd.Timestamp) should raise an error.
    # np.dtype(invalid_type).kind = 0 for such objects. However, this will
    # also catch some valid dtypes such as object, np.object_ and 'object'
    # which we safeguard against by catching them earlier and returning
    # np.dtype(valid_dtype) before this condition is evaluated.
    if is_hashable(dtype) and dtype in [
        object,
        np.object_,
        "object",
        "O",
        "object_",
    ]:
        # check hashability to avoid errors/DeprecationWarning when we get
        # here and `dtype` is an array
        return npdtype
    elif npdtype.kind == "O":
        raise TypeError(f"dtype '{dtype}' not understood")

    return npdtype


def is_all_strings(value: ArrayLike) -> bool:
    """
    Check if this is an array of strings that we should try parsing.

    Includes object-dtype ndarray containing all-strings, StringArray,
    and Categorical with all-string categories.
    Does not include numpy string dtypes.
    """
    dtype = value.dtype

    if isinstance(dtype, np.dtype):
        if len(value) == 0:
            return dtype == np.dtype("object")
        else:
            return dtype == np.dtype("object") and lib.is_string_array(
                np.asarray(value), skipna=False
            )
    elif isinstance(dtype, CategoricalDtype):
        return dtype.categories.inferred_type == "string"
    return dtype == "string"


__all__ = [
    "classes",
    "DT64NS_DTYPE",
    "ensure_float64",
    "ensure_python_int",
    "ensure_str",
    "infer_dtype_from_object",
    "INT64_DTYPE",
    "is_1d_only_ea_dtype",
    "is_all_strings",
    "is_any_real_numeric_dtype",
    "is_array_like",
    "is_bool",
    "is_bool_dtype",
    "is_categorical_dtype",
    "is_complex",
    "is_complex_dtype",
    "is_dataclass",
    "is_datetime64_any_dtype",
    "is_datetime64_dtype",
    "is_datetime64_ns_dtype",
    "is_datetime64tz_dtype",
    "is_decimal",
    "is_dict_like",
    "is_dtype_equal",
    "is_ea_or_datetimelike_dtype",
    "is_extension_array_dtype",
    "is_file_like",
    "is_float_dtype",
    "is_int64_dtype",
    "is_integer_dtype",
    "is_interval",
    "is_interval_dtype",
    "is_iterator",
    "is_named_tuple",
    "is_nested_list_like",
    "is_number",
    "is_numeric_dtype",
    "is_object_dtype",
    "is_period_dtype",
    "is_re",
    "is_re_compilable",
    "is_scipy_sparse",
    "is_sequence",
    "is_signed_integer_dtype",
    "is_sparse",
    "is_string_dtype",
    "is_string_or_object_np_dtype",
    "is_timedelta64_dtype",
    "is_timedelta64_ns_dtype",
    "is_unsigned_integer_dtype",
    "needs_i8_conversion",
    "pandas_dtype",
    "TD64NS_DTYPE",
    "validate_all_hashable",
]
<|endoftext|>"""
Utility functions related to concat.
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

from pandas.core.dtypes.astype import astype_array
from pandas.core.dtypes.cast import (
    common_dtype_categorical_compat,
    find_common_type,
    np_find_common_type,
)
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas.core.dtypes.generic import (
    ABCCategoricalIndex,
    ABCSeries,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pandas._typing import (
        ArrayLike,
        AxisInt,
        DtypeObj,
    )

    from pandas.core.arrays import (
        Categorical,
        ExtensionArray,
    )


def _is_nonempty(x, axis) -> bool:
    # filter empty arrays
    # 1-d dtypes always are included here
    if x.ndim <= axis:
        return True
    return x.shape[axis] > 0


def concat_compat(
    to_concat: Sequence[ArrayLike], axis: AxisInt = 0, ea_compat_axis: bool = False
) -> ArrayLike:
    """
    provide concatenation of an array of arrays each of which is a single
    'normalized' dtypes (in that for example, if it's object, then it is a
    non-datetimelike and provide a combined dtype for the resulting array that
    preserves the overall dtype if possible)

    Parameters
    ----------
    to_concat : sequence of arrays
    axis : axis to provide concatenation
    ea_compat_axis : bool, default False
        For ExtensionArray compat, behave as if axis == 1 when determining
        whether to drop empty arrays.

    Returns
    -------
    a single array, preserving the combined dtypes
    """
    if len(to_concat) and lib.dtypes_all_equal([obj.dtype for obj in to_concat]):
        # fastpath!
        obj = to_concat[0]
        if isinstance(obj, np.ndarray):
            to_concat_arrs = cast("Sequence[np.ndarray]", to_concat)
            return np.concatenate(to_concat_arrs, axis=axis)

        to_concat_eas = cast("Sequence[ExtensionArray]", to_concat)
        if ea_compat_axis:
            # We have 1D objects, that don't support axis keyword
            return obj._concat_same_type(to_concat_eas)
        elif axis == 0:
            return obj._concat_same_type(to_concat_eas)
        else:
            # e.g. DatetimeArray
            # NB: We are assuming here that ensure_wrapped_if_arraylike has
            #  been called where relevant.
            return obj._concat_same_type(
                # error: Unexpected keyword argument "axis" for "_concat_same_type"
                # of "ExtensionArray"
                to_concat_eas,
                axis=axis,  # type: ignore[call-arg]
            )

    # If all arrays are empty, there's nothing to convert, just short-cut to
    # the concatenation, #3121.
    #
    # Creating an empty array directly is tempting, but the winnings would be
    # marginal given that it would still require shape & dtype calculation and
    # np.concatenate which has them both implemented is compiled.
    orig = to_concat
    non_empties = [x for x in to_concat if _is_nonempty(x, axis)]
    if non_empties and axis == 0 and not ea_compat_axis:
        # ea_compat_axis see GH#39574
        to_concat = non_empties

    any_ea, kinds, target_dtype = _get_result_dtype(to_concat, non_empties)

    if len(to_concat) < len(orig):
        _, _, alt_dtype = _get_result_dtype(orig, non_empties)
        if alt_dtype != target_dtype:
            # GH#39122
            warnings.warn(
                "The behavior of array concatenation with empty entries is "
                "deprecated. In a future version, this will no longer exclude "
                "empty items when determining the result dtype. "
                "To retain the old behavior, exclude the empty entries before "
                "the concat operation.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )

    if target_dtype is not None:
        to_concat = [astype_array(arr, target_dtype, copy=False) for arr in to_concat]

    if not isinstance(to_concat[0], np.ndarray):
        # i.e. isinstance(to_concat[0], ExtensionArray)
        to_concat_eas = cast("Sequence[ExtensionArray]", to_concat)
        cls = type(to_concat[0])
        # GH#53640: eg. for datetime array, axis=1 but 0 is default
        # However, class method `_concat_same_type()` for some classes
        # may not support the `axis` keyword
        if ea_compat_axis or axis == 0:
            return cls._concat_same_type(to_concat_eas)
        else:
            return cls._concat_same_type(
                to_concat_eas,
                axis=axis,  # type: ignore[call-arg]
            )
    else:
        to_concat_arrs = cast("Sequence[np.ndarray]", to_concat)
        result = np.concatenate(to_concat_arrs, axis=axis)

        if not any_ea and "b" in kinds and result.dtype.kind in "iuf":
            # GH#39817 cast to object instead of casting bools to numeric
            result = result.astype(object, copy=False)
    return result


def _get_result_dtype(
    to_concat: Sequence[ArrayLike], non_empties: Sequence[ArrayLike]
) -> tuple[bool, set[str], DtypeObj | None]:
    target_dtype = None

    dtypes = {obj.dtype for obj in to_concat}
    kinds = {obj.dtype.kind for obj in to_concat}

    any_ea = any(not isinstance(x, np.ndarray) for x in to_concat)
    if any_ea:
        # i.e. any ExtensionArrays

        # we ignore axis here, as internally concatting with EAs is always
        # for axis=0
        if len(dtypes) != 1:
            target_dtype = find_common_type([x.dtype for x in to_concat])
            target_dtype = common_dtype_categorical_compat(to_concat, target_dtype)

    elif not len(non_empties):
        # we have all empties, but may need to coerce the result dtype to
        # object if we have non-numeric type operands (numpy would otherwise
        # cast this to float)
        if len(kinds) != 1:
            if not len(kinds - {"i", "u", "f"}) or not len(kinds - {"b", "i", "u"}):
                # let numpy coerce
                pass
            else:
                # coerce to object
                target_dtype = np.dtype(object)
                kinds = {"o"}
    else:
        # error: Argument 1 to "np_find_common_type" has incompatible type
        # "*Set[Union[ExtensionDtype, Any]]"; expected "dtype[Any]"
        target_dtype = np_find_common_type(*dtypes)  # type: ignore[arg-type]

    return any_ea, kinds, target_dtype


def union_categoricals(
    to_union, sort_categories: bool = False, ignore_order: bool = False
) -> Categorical:
    """
    Combine list-like of Categorical-like, unioning categories.

    All categories must have the same dtype.

    Parameters
    ----------
    to_union : list-like
        Categorical, CategoricalIndex, or Series with dtype='category'.
    sort_categories : bool, default False
        If true, resulting categories will be lexsorted, otherwise
        they will be ordered as they appear in the data.
    ignore_order : bool, default False
        If true, the ordered attribute of the Categoricals will be ignored.
        Results in an unordered categorical.

    Returns
    -------
    Categorical

    Raises
    ------
    TypeError
        - all inputs do not have the same dtype
        - all inputs do not have the same ordered property
        - all inputs are ordered and their categories are not identical
        - sort_categories=True and Categoricals are ordered
    ValueError
        Empty list of categoricals passed

    Notes
    -----
    To learn more about categories, see `link
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html#unioning>`__

    Examples
    --------
    If you want to combine categoricals that do not necessarily have
    the same categories, `union_categoricals` will combine a list-like
    of categoricals. The new categories will be the union of the
    categories being combined.

    >>> a = pd.Categorical(["b", "c"])
    >>> b = pd.Categorical(["a", "b"])
    >>> pd.api.types.union_categoricals([a, b])
    ['b', 'c', 'a', 'b']
    Categories (3, object): ['b', 'c', 'a']

    By default, the resulting categories will be ordered as they appear
    in the `categories` of the data. If you want the categories to be
    lexsorted, use `sort_categories=True` argument.

    >>> pd.api.types.union_categoricals([a, b], sort_categories=True)
    ['b', 'c', 'a', 'b']
    Categories (3, object): ['a', 'b', 'c']

    `union_categoricals` also works with the case of combining two
    categoricals of the same categories and order information (e.g. what
    you could also `append` for).

    >>> a = pd.Categorical(["a", "b"], ordered=True)
    >>> b = pd.Categorical(["a", "b", "a"], ordered=True)
    >>> pd.api.types.union_categoricals([a, b])
    ['a', 'b', 'a', 'b', 'a']
    Categories (2, object): ['a' < 'b']

    Raises `TypeError` because the categories are ordered and not identical.

    >>> a = pd.Categorical(["a", "b"], ordered=True)
    >>> b = pd.Categorical(["a", "b", "c"], ordered=True)
    >>> pd.api.types.union_categoricals([a, b])
    Traceback (most recent call last):
        ...
    TypeError: to union ordered Categoricals, all categories must be the same

    Ordered categoricals with different categories or orderings can be
    combined by using the `ignore_ordered=True` argument.

    >>> a = pd.Categorical(["a", "b", "c"], ordered=True)
    >>> b = pd.Categorical(["c", "b", "a"], ordered=True)
    >>> pd.api.types.union_categoricals([a, b], ignore_order=True)
    ['a', 'b', 'c', 'c', 'b', 'a']
    Categories (3, object): ['a', 'b', 'c']

    `union_categoricals` also works with a `CategoricalIndex`, or `Series`
    containing categorical data, but note that the resulting array will
    always be a plain `Categorical`

    >>> a = pd.Series(["b", "c"], dtype='category')
    >>> b = pd.Series(["a", "b"], dtype='category')
    >>> pd.api.types.union_categoricals([a, b])
    ['b', 'c', 'a', 'b']
    Categories (3, object): ['b', 'c', 'a']
    """
    from pandas import Categorical
    from pandas.core.arrays.categorical import recode_for_categories

    if len(to_union) == 0:
        raise ValueError("No Categoricals to union")

    def _maybe_unwrap(x):
        if isinstance(x, (ABCCategoricalIndex, ABCSeries)):
            return x._values
        elif isinstance(x, Categorical):
            return x
        else:
            raise TypeError("all components to combine must be Categorical")

    to_union = [_maybe_unwrap(x) for x in to_union]
    first = to_union[0]

    if not lib.dtypes_all_equal([obj.categories.dtype for obj in to_union]):
        raise TypeError("dtype of categories must be the same")

    ordered = False
    if all(first._categories_match_up_to_permutation(other) for other in to_union[1:]):
        # identical categories - fastpath
        categories = first.categories
        ordered = first.ordered

        all_codes = [first._encode_with_my_categories(x)._codes for x in to_union]
        new_codes = np.concatenate(all_codes)

        if sort_categories and not ignore_order and ordered:
            raise TypeError("Cannot use sort_categories=True with ordered Categoricals")

        if sort_categories and not categories.is_monotonic_increasing:
            categories = categories.sort_values()
            indexer = categories.get_indexer(first.categories)

            from pandas.core.algorithms import take_nd

            new_codes = take_nd(indexer, new_codes, fill_value=-1)
    elif ignore_order or all(not c.ordered for c in to_union):
        # different categories - union and recode
        cats = first.categories.append([c.categories for c in to_union[1:]])
        categories = cats.unique()
        if sort_categories:
            categories = categories.sort_values()

        new_codes = [
            recode_for_categories(c.codes, c.categories, categories) for c in to_union
        ]
        new_codes = np.concatenate(new_codes)
    else:
        # ordered - to show a proper error message
        if all(c.ordered for c in to_union):
            msg = "to union ordered Categoricals, all categories must be the same"
            raise TypeError(msg)
        raise TypeError("Categorical.ordered must be the same")

    if ignore_order:
        ordered = False

    dtype = CategoricalDtype(categories=categories, ordered=ordered)
    return Categorical._simple_new(new_codes, dtype=dtype)
<|endoftext|>""" basic inference routines """

from __future__ import annotations

from collections import abc
from numbers import Number
import re
from re import Pattern
from typing import TYPE_CHECKING

import numpy as np

from pandas._libs import lib

if TYPE_CHECKING:
    from collections.abc import Hashable

    from pandas._typing import TypeGuard

is_bool = lib.is_bool

is_integer = lib.is_integer

is_float = lib.is_float

is_complex = lib.is_complex

is_scalar = lib.is_scalar

is_decimal = lib.is_decimal

is_interval = lib.is_interval

is_list_like = lib.is_list_like

is_iterator = lib.is_iterator


def is_number(obj) -> TypeGuard[Number | np.number]:
    """
    Check if the object is a number.

    Returns True when the object is a number, and False if is not.

    Parameters
    ----------
    obj : any type
        The object to check if is a number.

    Returns
    -------
    bool
        Whether `obj` is a number or not.

    See Also
    --------
    api.types.is_integer: Checks a subgroup of numbers.

    Examples
    --------
    >>> from pandas.api.types import is_number
    >>> is_number(1)
    True
    >>> is_number(7.15)
    True

    Booleans are valid because they are int subclass.

    >>> is_number(False)
    True

    >>> is_number("foo")
    False
    >>> is_number("5")
    False
    """
    return isinstance(obj, (Number, np.number))


def iterable_not_string(obj) -> bool:
    """
    Check if the object is an iterable but not a string.

    Parameters
    ----------
    obj : The object to check.

    Returns
    -------
    is_iter_not_string : bool
        Whether `obj` is a non-string iterable.

    Examples
    --------
    >>> iterable_not_string([1, 2, 3])
    True
    >>> iterable_not_string("foo")
    False
    >>> iterable_not_string(1)
    False
    """
    return isinstance(obj, abc.Iterable) and not isinstance(obj, str)


def is_file_like(obj) -> bool:
    """
    Check if the object is a file-like object.

    For objects to be considered file-like, they must
    be an iterator AND have either a `read` and/or `write`
    method as an attribute.

    Note: file-like objects must be iterable, but
    iterable objects need not be file-like.

    Parameters
    ----------
    obj : The object to check

    Returns
    -------
    bool
        Whether `obj` has file-like properties.

    Examples
    --------
    >>> import io
    >>> from pandas.api.types import is_file_like
    >>> buffer = io.StringIO("data")
    >>> is_file_like(buffer)
    True
    >>> is_file_like([1, 2, 3])
    False
    """
    if not (hasattr(obj, "read") or hasattr(obj, "write")):
        return False

    return bool(hasattr(obj, "__iter__"))


def is_re(obj) -> TypeGuard[Pattern]:
    """
    Check if the object is a regex pattern instance.

    Parameters
    ----------
    obj : The object to check

    Returns
    -------
    bool
        Whether `obj` is a regex pattern.

    Examples
    --------
    >>> from pandas.api.types import is_re
    >>> import re
    >>> is_re(re.compile(".*"))
    True
    >>> is_re("foo")
    False
    """
    return isinstance(obj, Pattern)


def is_re_compilable(obj) -> bool:
    """
    Check if the object can be compiled into a regex pattern instance.

    Parameters
    ----------
    obj : The object to check

    Returns
    -------
    bool
        Whether `obj` can be compiled as a regex pattern.

    Examples
    --------
    >>> from pandas.api.types import is_re_compilable
    >>> is_re_compilable(".*")
    True
    >>> is_re_compilable(1)
    False
    """
    try:
        re.compile(obj)
    except TypeError:
        return False
    else:
        return True


def is_array_like(obj) -> bool:
    """
    Check if the object is array-like.

    For an object to be considered array-like, it must be list-like and
    have a `dtype` attribute.

    Parameters
    ----------
    obj : The object to check

    Returns
    -------
    is_array_like : bool
        Whether `obj` has array-like properties.

    Examples
    --------
    >>> is_array_like(np.array([1, 2, 3]))
    True
    >>> is_array_like(pd.Series(["a", "b"]))
    True
    >>> is_array_like(pd.Index(["2016-01-01"]))
    True
    >>> is_array_like([1, 2, 3])
    False
    >>> is_array_like(("a", "b"))
    False
    """
    return is_list_like(obj) and hasattr(obj, "dtype")


def is_nested_list_like(obj) -> bool:
    """
    Check if the object is list-like, and that all of its elements
    are also list-like.

    Parameters
    ----------
    obj : The object to check

    Returns
    -------
    is_list_like : bool
        Whether `obj` has list-like properties.

    Examples
    --------
    >>> is_nested_list_like([[1, 2, 3]])
    True
    >>> is_nested_list_like([{1, 2, 3}, {1, 2, 3}])
    True
    >>> is_nested_list_like(["foo"])
    False
    >>> is_nested_list_like([])
    False
    >>> is_nested_list_like([[1, 2, 3], 1])
    False

    Notes
    -----
    This won't reliably detect whether a consumable iterator (e. g.
    a generator) is a nested-list-like without consuming the iterator.
    To avoid consuming it, we always return False if the outer container
    doesn't define `__len__`.

    See Also
    --------
    is_list_like
    """
    return (
        is_list_like(obj)
        and hasattr(obj, "__len__")
        and len(obj) > 0
        and all(is_list_like(item) for item in obj)
    )


def is_dict_like(obj) -> bool:
    """
    Check if the object is dict-like.

    Parameters
    ----------
    obj : The object to check

    Returns
    -------
    bool
        Whether `obj` has dict-like properties.

    Examples
    --------
    >>> from pandas.api.types import is_dict_like
    >>> is_dict_like({1: 2})
    True
    >>> is_dict_like([1, 2, 3])
    False
    >>> is_dict_like(dict)
    False
    >>> is_dict_like(dict())
    True
    """
    dict_like_attrs = ("__getitem__", "keys", "__contains__")
    return (
        all(hasattr(obj, attr) for attr in dict_like_attrs)
        # [GH 25196] exclude classes
        and not isinstance(obj, type)
    )


def is_named_tuple(obj) -> bool:
    """
    Check if the object is a named tuple.

    Parameters
    ----------
    obj : The object to check

    Returns
    -------
    bool
        Whether `obj` is a named tuple.

    Examples
    --------
    >>> from collections import namedtuple
    >>> from pandas.api.types import is_named_tuple
    >>> Point = namedtuple("Point", ["x", "y"])
    >>> p = Point(1, 2)
    >>>
    >>> is_named_tuple(p)
    True
    >>> is_named_tuple((1, 2))
    False
    """
    return isinstance(obj, abc.Sequence) and hasattr(obj, "_fields")


def is_hashable(obj) -> TypeGuard[Hashable]:
    """
    Return True if hash(obj) will succeed, False otherwise.

    Some types will pass a test against collections.abc.Hashable but fail when
    they are actually hashed with hash().

    Distinguish between these and other types by trying the call to hash() and
    seeing if they raise TypeError.

    Returns
    -------
    bool

    Examples
    --------
    >>> import collections
    >>> from pandas.api.types import is_hashable
    >>> a = ([],)
    >>> isinstance(a, collections.abc.Hashable)
    True
    >>> is_hashable(a)
    False
    """
    # Unfortunately, we can't use isinstance(obj, collections.abc.Hashable),
    # which can be faster than calling hash. That is because numpy scalars
    # fail this test.

    # Reconsider this decision once this numpy bug is fixed:
    # https://github.com/numpy/numpy/issues/5562

    try:
        hash(obj)
    except TypeError:
        return False
    else:
        return True


def is_sequence(obj) -> bool:
    """
    Check if the object is a sequence of objects.
    String types are not included as sequences here.

    Parameters
    ----------
    obj : The object to check

    Returns
    -------
    is_sequence : bool
        Whether `obj` is a sequence of objects.

    Examples
    --------
    >>> l = [1, 2, 3]
    >>>
    >>> is_sequence(l)
    True
    >>> is_sequence(iter(l))
    False
    """
    try:
        iter(obj)  # Can iterate over it.
        len(obj)  # Has a length associated with it.
        return not isinstance(obj, (str, bytes))
    except (TypeError, AttributeError):
        return False


def is_dataclass(item) -> bool:
    """
    Checks if the object is a data-class instance

    Parameters
    ----------
    item : object

    Returns
    --------
    is_dataclass : bool
        True if the item is an instance of a data-class,
        will return false if you pass the data class itself

    Examples
    --------
    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class Point:
    ...     x: int
    ...     y: int

    >>> is_dataclass(Point)
    False
    >>> is_dataclass(Point(0,2))
    True

    """
    try:
        import dataclasses

        return dataclasses.is_dataclass(item) and not isinstance(item, type)
    except ImportError:
        return False
<|endoftext|>"""
Extend pandas with custom array types.
"""
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
    cast,
    overload,
)

import numpy as np

from pandas._libs import missing as libmissing
from pandas._libs.hashtable import object_hash
from pandas._libs.properties import cache_readonly
from pandas.errors import AbstractMethodError

from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCIndex,
    ABCSeries,
)

if TYPE_CHECKING:
    from pandas._typing import (
        DtypeObj,
        Self,
        Shape,
        npt,
        type_t,
    )

    from pandas import Index
    from pandas.core.arrays import ExtensionArray

    # To parameterize on same ExtensionDtype
    ExtensionDtypeT = TypeVar("ExtensionDtypeT", bound="ExtensionDtype")


class ExtensionDtype:
    """
    A custom data type, to be paired with an ExtensionArray.

    See Also
    --------
    extensions.register_extension_dtype: Register an ExtensionType
        with pandas as class decorator.
    extensions.ExtensionArray: Abstract base class for custom 1-D array types.

    Notes
    -----
    The interface includes the following abstract methods that must
    be implemented by subclasses:

    * type
    * name
    * construct_array_type

    The following attributes and methods influence the behavior of the dtype in
    pandas operations

    * _is_numeric
    * _is_boolean
    * _get_common_dtype

    The `na_value` class attribute can be used to set the default NA value
    for this type. :attr:`numpy.nan` is used by default.

    ExtensionDtypes are required to be hashable. The base class provides
    a default implementation, which relies on the ``_metadata`` class
    attribute. ``_metadata`` should be a tuple containing the strings
    that define your data type. For example, with ``PeriodDtype`` that's
    the ``freq`` attribute.

    **If you have a parametrized dtype you should set the ``_metadata``
    class property**.

    Ideally, the attributes in ``_metadata`` will match the
    parameters to your ``ExtensionDtype.__init__`` (if any). If any of
    the attributes in ``_metadata`` don't implement the standard
    ``__eq__`` or ``__hash__``, the default implementations here will not
    work.

    Examples
    --------

    For interaction with Apache Arrow (pyarrow), a ``__from_arrow__`` method
    can be implemented: this method receives a pyarrow Array or ChunkedArray
    as only argument and is expected to return the appropriate pandas
    ExtensionArray for this dtype and the passed values:

    >>> import pyarrow
    >>> from pandas.api.extensions import ExtensionArray
    >>> class ExtensionDtype:
    ...     def __from_arrow__(
    ...         self,
    ...         array: pyarrow.Array | pyarrow.ChunkedArray
    ...     ) -> ExtensionArray:
    ...         ...

    This class does not inherit from 'abc.ABCMeta' for performance reasons.
    Methods and properties required by the interface raise
    ``pandas.errors.AbstractMethodError`` and no ``register`` method is
    provided for registering virtual subclasses.
    """

    _metadata: tuple[str, ...] = ()

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        """
        Check whether 'other' is equal to self.

        By default, 'other' is considered equal if either

        * it's a string matching 'self.name'.
        * it's an instance of this type and all of the attributes
          in ``self._metadata`` are equal between `self` and `other`.

        Parameters
        ----------
        other : Any

        Returns
        -------
        bool
        """
        if isinstance(other, str):
            try:
                other = self.construct_from_string(other)
            except TypeError:
                return False
        if isinstance(other, type(self)):
            return all(
                getattr(self, attr) == getattr(other, attr) for attr in self._metadata
            )
        return False

    def __hash__(self) -> int:
        # for python>=3.10, different nan objects have different hashes
        # we need to avoid that and thus use hash function with old behavior
        return object_hash(tuple(getattr(self, attr) for attr in self._metadata))

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    @property
    def na_value(self) -> object:
        """
        Default NA value to use for this type.

        This is used in e.g. ExtensionArray.take. This should be the
        user-facing "boxed" version of the NA value, not the physical NA value
        for storage.  e.g. for JSONArray, this is an empty dictionary.
        """
        return np.nan

    @property
    def type(self) -> type_t[Any]:
        """
        The scalar type for the array, e.g. ``int``

        It's expected ``ExtensionArray[item]`` returns an instance
        of ``ExtensionDtype.type`` for scalar ``item``, assuming
        that value is valid (not NA). NA values do not need to be
        instances of `type`.
        """
        raise AbstractMethodError(self)

    @property
    def kind(self) -> str:
        """
        A character code (one of 'biufcmMOSUV'), default 'O'

        This should match the NumPy dtype used when the array is
        converted to an ndarray, which is probably 'O' for object if
        the extension type cannot be represented as a built-in NumPy
        type.

        See Also
        --------
        numpy.dtype.kind
        """
        return "O"

    @property
    def name(self) -> str:
        """
        A string identifying the data type.

        Will be used for display in, e.g. ``Series.dtype``
        """
        raise AbstractMethodError(self)

    @property
    def names(self) -> list[str] | None:
        """
        Ordered list of field names, or None if there are no fields.

        This is for compatibility with NumPy arrays, and may be removed in the
        future.
        """
        return None

    @classmethod
    def construct_array_type(cls) -> type_t[ExtensionArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        raise AbstractMethodError(cls)

    def empty(self, shape: Shape) -> ExtensionArray:
        """
        Construct an ExtensionArray of this dtype with the given shape.

        Analogous to numpy.empty.

        Parameters
        ----------
        shape : int or tuple[int]

        Returns
        -------
        ExtensionArray
        """
        cls = self.construct_array_type()
        return cls._empty(shape, dtype=self)

    @classmethod
    def construct_from_string(cls, string: str) -> Self:
        r"""
        Construct this type from a string.

        This is useful mainly for data types that accept parameters.
        For example, a period dtype accepts a frequency parameter that
        can be set as ``period[h]`` (where H means hourly frequency).

        By default, in the abstract class, just the name of the type is
        expected. But subclasses can overwrite this method to accept
        parameters.

        Parameters
        ----------
        string : str
            The name of the type, for example ``category``.

        Returns
        -------
        ExtensionDtype
            Instance of the dtype.

        Raises
        ------
        TypeError
            If a class cannot be constructed from this 'string'.

        Examples
        --------
        For extension dtypes with arguments the following may be an
        adequate implementation.

        >>> import re
        >>> @classmethod
        ... def construct_from_string(cls, string):
        ...     pattern = re.compile(r"^my_type\[(?P<arg_name>.+)\]$")
        ...     match = pattern.match(string)
        ...     if match:
        ...         return cls(**match.groupdict())
        ...     else:
        ...         raise TypeError(
        ...             f"Cannot construct a '{cls.__name__}' from '{string}'"
        ...         )
        """
        if not isinstance(string, str):
            raise TypeError(
                f"'construct_from_string' expects a string, got {type(string)}"
            )
        # error: Non-overlapping equality check (left operand type: "str", right
        #  operand type: "Callable[[ExtensionDtype], str]")  [comparison-overlap]
        assert isinstance(cls.name, str), (cls, type(cls.name))
        if string != cls.name:
            raise TypeError(f"Cannot construct a '{cls.__name__}' from '{string}'")
        return cls()

    @classmethod
    def is_dtype(cls, dtype: object) -> bool:
        """
        Check if we match 'dtype'.

        Parameters
        ----------
        dtype : object
            The object to check.

        Returns
        -------
        bool

        Notes
        -----
        The default implementation is True if

        1. ``cls.construct_from_string(dtype)`` is an instance
           of ``cls``.
        2. ``dtype`` is an object and is an instance of ``cls``
        3. ``dtype`` has a ``dtype`` attribute, and any of the above
           conditions is true for ``dtype.dtype``.
        """
        dtype = getattr(dtype, "dtype", dtype)

        if isinstance(dtype, (ABCSeries, ABCIndex, ABCDataFrame, np.dtype)):
            # https://github.com/pandas-dev/pandas/issues/22960
            # avoid passing data to `construct_from_string`. This could
            # cause a FutureWarning from numpy about failing elementwise
            # comparison from, e.g., comparing DataFrame == 'category'.
            return False
        elif dtype is None:
            return False
        elif isinstance(dtype, cls):
            return True
        if isinstance(dtype, str):
            try:
                return cls.construct_from_string(dtype) is not None
            except TypeError:
                return False
        return False

    @property
    def _is_numeric(self) -> bool:
        """
        Whether columns with this dtype should be considered numeric.

        By default ExtensionDtypes are assumed to be non-numeric.
        They'll be excluded from operations that exclude non-numeric
        columns, like (groupby) reductions, plotting, etc.
        """
        return False

    @property
    def _is_boolean(self) -> bool:
        """
        Whether this dtype should be considered boolean.

        By default, ExtensionDtypes are assumed to be non-numeric.
        Setting this to True will affect the behavior of several places,
        e.g.

        * is_bool
        * boolean indexing

        Returns
        -------
        bool
        """
        return False

    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None:
        """
        Return the common dtype, if one exists.

        Used in `find_common_type` implementation. This is for example used
        to determine the resulting dtype in a concat operation.

        If no common dtype exists, return None (which gives the other dtypes
        the chance to determine a common dtype). If all dtypes in the list
        return None, then the common dtype will be "object" dtype (this means
        it is never needed to return "object" dtype from this method itself).

        Parameters
        ----------
        dtypes : list of dtypes
            The dtypes for which to determine a common dtype. This is a list
            of np.dtype or ExtensionDtype instances.

        Returns
        -------
        Common dtype (np.dtype or ExtensionDtype) or None
        """
        if len(set(dtypes)) == 1:
            # only itself
            return self
        else:
            return None

    @property
    def _can_hold_na(self) -> bool:
        """
        Can arrays of this dtype hold NA values?
        """
        return True

    @property
    def _is_immutable(self) -> bool:
        """
        Can arrays with this dtype be modified with __setitem__? If not, return
        True.

        Immutable arrays are expected to raise TypeError on __setitem__ calls.
        """
        return False

    @cache_readonly
    def index_class(self) -> type_t[Index]:
        """
        The Index subclass to return from Index.__new__ when this dtype is
        encountered.
        """
        from pandas import Index

        return Index

    @property
    def _supports_2d(self) -> bool:
        """
        Do ExtensionArrays with this dtype support 2D arrays?

        Historically ExtensionArrays were limited to 1D. By returning True here,
        authors can indicate that their arrays support 2D instances. This can
        improve performance in some cases, particularly operations with `axis=1`.

        Arrays that support 2D values should:

            - implement Array.reshape
            - subclass the Dim2CompatTests in tests.extension.base
            - _concat_same_type should support `axis` keyword
            - _reduce and reductions should support `axis` keyword
        """
        return False

    @property
    def _can_fast_transpose(self) -> bool:
        """
        Is transposing an array with this dtype zero-copy?

        Only relevant for cases where _supports_2d is True.
        """
        return False


class StorageExtensionDtype(ExtensionDtype):
    """ExtensionDtype that may be backed by more than one implementation."""

    name: str
    _metadata = ("storage",)

    def __init__(self, storage: str | None = None) -> None:
        self.storage = storage

    def __repr__(self) -> str:
        return f"{self.name}[{self.storage}]"

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str) and other == self.name:
            return True
        return super().__eq__(other)

    def __hash__(self) -> int:
        # custom __eq__ so have to override __hash__
        return super().__hash__()

    @property
    def na_value(self) -> libmissing.NAType:
        return libmissing.NA


def register_extension_dtype(cls: type_t[ExtensionDtypeT]) -> type_t[ExtensionDtypeT]:
    """
    Register an ExtensionType with pandas as class decorator.

    This enables operations like ``.astype(name)`` for the name
    of the ExtensionDtype.

    Returns
    -------
    callable
        A class decorator.

    Examples
    --------
    >>> from pandas.api.extensions import register_extension_dtype, ExtensionDtype
    >>> @register_extension_dtype
    ... class MyExtensionDtype(ExtensionDtype):
    ...     name = "myextension"
    """
    _registry.register(cls)
    return cls


class Registry:
    """
    Registry for dtype inference.

    The registry allows one to map a string repr of a extension
    dtype to an extension dtype. The string alias can be used in several
    places, including

    * Series and Index constructors
    * :meth:`pandas.array`
    * :meth:`pandas.Series.astype`

    Multiple extension types can be registered.
    These are tried in order.
    """

    def __init__(self) -> None:
        self.dtypes: list[type_t[ExtensionDtype]] = []

    def register(self, dtype: type_t[ExtensionDtype]) -> None:
        """
        Parameters
        ----------
        dtype : ExtensionDtype class
        """
        if not issubclass(dtype, ExtensionDtype):
            raise ValueError("can only register pandas extension dtypes")

        self.dtypes.append(dtype)

    @overload
    def find(self, dtype: type_t[ExtensionDtypeT]) -> type_t[ExtensionDtypeT]:
        ...

    @overload
    def find(self, dtype: ExtensionDtypeT) -> ExtensionDtypeT:
        ...

    @overload
    def find(self, dtype: str) -> ExtensionDtype | None:
        ...

    @overload
    def find(
        self, dtype: npt.DTypeLike
    ) -> type_t[ExtensionDtype] | ExtensionDtype | None:
        ...

    def find(
        self, dtype: type_t[ExtensionDtype] | ExtensionDtype | npt.DTypeLike
    ) -> type_t[ExtensionDtype] | ExtensionDtype | None:
        """
        Parameters
        ----------
        dtype : ExtensionDtype class or instance or str or numpy dtype or python type

        Returns
        -------
        return the first matching dtype, otherwise return None
        """
        if not isinstance(dtype, str):
            dtype_type: type_t
            if not isinstance(dtype, type):
                dtype_type = type(dtype)
            else:
                dtype_type = dtype
            if issubclass(dtype_type, ExtensionDtype):
                # cast needed here as mypy doesn't know we have figured
                # out it is an ExtensionDtype or type_t[ExtensionDtype]
                return cast("ExtensionDtype | type_t[ExtensionDtype]", dtype)

            return None

        for dtype_type in self.dtypes:
            try:
                return dtype_type.construct_from_string(dtype)
            except TypeError:
                pass

        return None


_registry = Registry()
<|endoftext|>"""
missing types & inference
"""
from __future__ import annotations

from decimal import Decimal
from functools import partial
from typing import (
    TYPE_CHECKING,
    overload,
)
import warnings

import numpy as np

from pandas._config import get_option

from pandas._libs import lib
import pandas._libs.missing as libmissing
from pandas._libs.tslibs import (
    NaT,
    iNaT,
)

from pandas.core.dtypes.common import (
    DT64NS_DTYPE,
    TD64NS_DTYPE,
    ensure_object,
    is_scalar,
    is_string_or_object_np_dtype,
)
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    DatetimeTZDtype,
    ExtensionDtype,
    IntervalDtype,
    PeriodDtype,
)
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCExtensionArray,
    ABCIndex,
    ABCMultiIndex,
    ABCSeries,
)
from pandas.core.dtypes.inference import is_list_like

if TYPE_CHECKING:
    from re import Pattern

    from pandas._typing import (
        ArrayLike,
        DtypeObj,
        NDFrame,
        NDFrameT,
        Scalar,
        npt,
    )

    from pandas import Series
    from pandas.core.indexes.base import Index


isposinf_scalar = libmissing.isposinf_scalar
isneginf_scalar = libmissing.isneginf_scalar

nan_checker = np.isnan
INF_AS_NA = False
_dtype_object = np.dtype("object")
_dtype_str = np.dtype(str)


@overload
def isna(obj: Scalar | Pattern) -> bool:
    ...


@overload
def isna(
    obj: ArrayLike | Index | list,
) -> npt.NDArray[np.bool_]:
    ...


@overload
def isna(obj: NDFrameT) -> NDFrameT:
    ...


# handle unions
@overload
def isna(obj: NDFrameT | ArrayLike | Index | list) -> NDFrameT | npt.NDArray[np.bool_]:
    ...


@overload
def isna(obj: object) -> bool | npt.NDArray[np.bool_] | NDFrame:
    ...


def isna(obj: object) -> bool | npt.NDArray[np.bool_] | NDFrame:
    """
    Detect missing values for an array-like object.

    This function takes a scalar or array-like object and indicates
    whether values are missing (``NaN`` in numeric arrays, ``None`` or ``NaN``
    in object arrays, ``NaT`` in datetimelike).

    Parameters
    ----------
    obj : scalar or array-like
        Object to check for null or missing values.

    Returns
    -------
    bool or array-like of bool
        For scalar input, returns a scalar boolean.
        For array input, returns an array of boolean indicating whether each
        corresponding element is missing.

    See Also
    --------
    notna : Boolean inverse of pandas.isna.
    Series.isna : Detect missing values in a Series.
    DataFrame.isna : Detect missing values in a DataFrame.
    Index.isna : Detect missing values in an Index.

    Examples
    --------
    Scalar arguments (including strings) result in a scalar boolean.

    >>> pd.isna('dog')
    False

    >>> pd.isna(pd.NA)
    True

    >>> pd.isna(np.nan)
    True

    ndarrays result in an ndarray of booleans.

    >>> array = np.array([[1, np.nan, 3], [4, 5, np.nan]])
    >>> array
    array([[ 1., nan,  3.],
           [ 4.,  5., nan]])
    >>> pd.isna(array)
    array([[False,  True, False],
           [False, False,  True]])

    For indexes, an ndarray of booleans is returned.

    >>> index = pd.DatetimeIndex(["2017-07-05", "2017-07-06", None,
    ...                           "2017-07-08"])
    >>> index
    DatetimeIndex(['2017-07-05', '2017-07-06', 'NaT', '2017-07-08'],
                  dtype='datetime64[ns]', freq=None)
    >>> pd.isna(index)
    array([False, False,  True, False])

    For Series and DataFrame, the same type is returned, containing booleans.

    >>> df = pd.DataFrame([['ant', 'bee', 'cat'], ['dog', None, 'fly']])
    >>> df
         0     1    2
    0  ant   bee  cat
    1  dog  None  fly
    >>> pd.isna(df)
           0      1      2
    0  False  False  False
    1  False   True  False

    >>> pd.isna(df[1])
    0    False
    1     True
    Name: 1, dtype: bool
    """
    return _isna(obj)


isnull = isna


def _isna(obj, inf_as_na: bool = False):
    """
    Detect missing values, treating None, NaN or NA as null. Infinite
    values will also be treated as null if inf_as_na is True.

    Parameters
    ----------
    obj: ndarray or object value
        Input array or scalar value.
    inf_as_na: bool
        Whether to treat infinity as null.

    Returns
    -------
    boolean ndarray or boolean
    """
    if is_scalar(obj):
        return libmissing.checknull(obj, inf_as_na=inf_as_na)
    elif isinstance(obj, ABCMultiIndex):
        raise NotImplementedError("isna is not defined for MultiIndex")
    elif isinstance(obj, type):
        return False
    elif isinstance(obj, (np.ndarray, ABCExtensionArray)):
        return _isna_array(obj, inf_as_na=inf_as_na)
    elif isinstance(obj, ABCIndex):
        # Try to use cached isna, which also short-circuits for integer dtypes
        #  and avoids materializing RangeIndex._values
        if not obj._can_hold_na:
            return obj.isna()
        return _isna_array(obj._values, inf_as_na=inf_as_na)

    elif isinstance(obj, ABCSeries):
        result = _isna_array(obj._values, inf_as_na=inf_as_na)
        # box
        result = obj._constructor(result, index=obj.index, name=obj.name, copy=False)
        return result
    elif isinstance(obj, ABCDataFrame):
        return obj.isna()
    elif isinstance(obj, list):
        return _isna_array(np.asarray(obj, dtype=object), inf_as_na=inf_as_na)
    elif hasattr(obj, "__array__"):
        return _isna_array(np.asarray(obj), inf_as_na=inf_as_na)
    else:
        return False


def _use_inf_as_na(key) -> None:
    """
    Option change callback for na/inf behaviour.

    Choose which replacement for numpy.isnan / -numpy.isfinite is used.

    Parameters
    ----------
    flag: bool
        True means treat None, NaN, INF, -INF as null (old way),
        False means None and NaN are null, but INF, -INF are not null
        (new way).

    Notes
    -----
    This approach to setting global module values is discussed and
    approved here:

    * https://stackoverflow.com/questions/4859217/
      programmatically-creating-variables-in-python/4859312#4859312
    """
    inf_as_na = get_option(key)
    globals()["_isna"] = partial(_isna, inf_as_na=inf_as_na)
    if inf_as_na:
        globals()["nan_checker"] = lambda x: ~np.isfinite(x)
        globals()["INF_AS_NA"] = True
    else:
        globals()["nan_checker"] = np.isnan
        globals()["INF_AS_NA"] = False


def _isna_array(values: ArrayLike, inf_as_na: bool = False):
    """
    Return an array indicating which values of the input array are NaN / NA.

    Parameters
    ----------
    obj: ndarray or ExtensionArray
        The input array whose elements are to be checked.
    inf_as_na: bool
        Whether or not to treat infinite values as NA.

    Returns
    -------
    array-like
        Array of boolean values denoting the NA status of each element.
    """
    dtype = values.dtype

    if not isinstance(values, np.ndarray):
        # i.e. ExtensionArray
        if inf_as_na and isinstance(dtype, CategoricalDtype):
            result = libmissing.isnaobj(values.to_numpy(), inf_as_na=inf_as_na)
        else:
            # error: Incompatible types in assignment (expression has type
            # "Union[ndarray[Any, Any], ExtensionArraySupportsAnyAll]", variable has
            # type "ndarray[Any, dtype[bool_]]")
            result = values.isna()  # type: ignore[assignment]
    elif isinstance(values, np.rec.recarray):
        # GH 48526
        result = _isna_recarray_dtype(values, inf_as_na=inf_as_na)
    elif is_string_or_object_np_dtype(values.dtype):
        result = _isna_string_dtype(values, inf_as_na=inf_as_na)
    elif dtype.kind in "mM":
        # this is the NaT pattern
        result = values.view("i8") == iNaT
    else:
        if inf_as_na:
            result = ~np.isfinite(values)
        else:
            result = np.isnan(values)

    return result


def _isna_string_dtype(values: np.ndarray, inf_as_na: bool) -> npt.NDArray[np.bool_]:
    # Working around NumPy ticket 1542
    dtype = values.dtype

    if dtype.kind in ("S", "U"):
        result = np.zeros(values.shape, dtype=bool)
    else:
        if values.ndim in {1, 2}:
            result = libmissing.isnaobj(values, inf_as_na=inf_as_na)
        else:
            # 0-D, reached via e.g. mask_missing
            result = libmissing.isnaobj(values.ravel(), inf_as_na=inf_as_na)
            result = result.reshape(values.shape)

    return result


def _has_record_inf_value(record_as_array: np.ndarray) -> np.bool_:
    is_inf_in_record = np.zeros(len(record_as_array), dtype=bool)
    for i, value in enumerate(record_as_array):
        is_element_inf = False
        try:
            is_element_inf = np.isinf(value)
        except TypeError:
            is_element_inf = False
        is_inf_in_record[i] = is_element_inf

    return np.any(is_inf_in_record)


def _isna_recarray_dtype(
    values: np.rec.recarray, inf_as_na: bool
) -> npt.NDArray[np.bool_]:
    result = np.zeros(values.shape, dtype=bool)
    for i, record in enumerate(values):
        record_as_array = np.array(record.tolist())
        does_record_contain_nan = isna_all(record_as_array)
        does_record_contain_inf = False
        if inf_as_na:
            does_record_contain_inf = bool(_has_record_inf_value(record_as_array))
        result[i] = np.any(
            np.logical_or(does_record_contain_nan, does_record_contain_inf)
        )

    return result


@overload
def notna(obj: Scalar) -> bool:
    ...


@overload
def notna(
    obj: ArrayLike | Index | list,
) -> npt.NDArray[np.bool_]:
    ...


@overload
def notna(obj: NDFrameT) -> NDFrameT:
    ...


# handle unions
@overload
def notna(obj: NDFrameT | ArrayLike | Index | list) -> NDFrameT | npt.NDArray[np.bool_]:
    ...


@overload
def notna(obj: object) -> bool | npt.NDArray[np.bool_] | NDFrame:
    ...


def notna(obj: object) -> bool | npt.NDArray[np.bool_] | NDFrame:
    """
    Detect non-missing values for an array-like object.

    This function takes a scalar or array-like object and indicates
    whether values are valid (not missing, which is ``NaN`` in numeric
    arrays, ``None`` or ``NaN`` in object arrays, ``NaT`` in datetimelike).

    Parameters
    ----------
    obj : array-like or object value
        Object to check for *not* null or *non*-missing values.

    Returns
    -------
    bool or array-like of bool
        For scalar input, returns a scalar boolean.
        For array input, returns an array of boolean indicating whether each
        corresponding element is valid.

    See Also
    --------
    isna : Boolean inverse of pandas.notna.
    Series.notna : Detect valid values in a Series.
    DataFrame.notna : Detect valid values in a DataFrame.
    Index.notna : Detect valid values in an Index.

    Examples
    --------
    Scalar arguments (including strings) result in a scalar boolean.

    >>> pd.notna('dog')
    True

    >>> pd.notna(pd.NA)
    False

    >>> pd.notna(np.nan)
    False

    ndarrays result in an ndarray of booleans.

    >>> array = np.array([[1, np.nan, 3], [4, 5, np.nan]])
    >>> array
    array([[ 1., nan,  3.],
           [ 4.,  5., nan]])
    >>> pd.notna(array)
    array([[ True, False,  True],
           [ True,  True, False]])

    For indexes, an ndarray of booleans is returned.

    >>> index = pd.DatetimeIndex(["2017-07-05", "2017-07-06", None,
    ...                          "2017-07-08"])
    >>> index
    DatetimeIndex(['2017-07-05', '2017-07-06', 'NaT', '2017-07-08'],
                  dtype='datetime64[ns]', freq=None)
    >>> pd.notna(index)
    array([ True,  True, False,  True])

    For Series and DataFrame, the same type is returned, containing booleans.

    >>> df = pd.DataFrame([['ant', 'bee', 'cat'], ['dog', None, 'fly']])
    >>> df
         0     1    2
    0  ant   bee  cat
    1  dog  None  fly
    >>> pd.notna(df)
          0      1     2
    0  True   True  True
    1  True  False  True

    >>> pd.notna(df[1])
    0     True
    1    False
    Name: 1, dtype: bool
    """
    res = isna(obj)
    if isinstance(res, bool):
        return not res
    return ~res


notnull = notna


def array_equivalent(
    left,
    right,
    strict_nan: bool = False,
    dtype_equal: bool = False,
) -> bool:
    """
    True if two arrays, left and right, have equal non-NaN elements, and NaNs
    in corresponding locations.  False otherwise. It is assumed that left and
    right are NumPy arrays of the same dtype. The behavior of this function
    (particularly with respect to NaNs) is not defined if the dtypes are
    different.

    Parameters
    ----------
    left, right : ndarrays
    strict_nan : bool, default False
        If True, consider NaN and None to be different.
    dtype_equal : bool, default False
        Whether `left` and `right` are known to have the same dtype
        according to `is_dtype_equal`. Some methods like `BlockManager.equals`.
        require that the dtypes match. Setting this to ``True`` can improve
        performance, but will give different results for arrays that are
        equal but different dtypes.

    Returns
    -------
    b : bool
        Returns True if the arrays are equivalent.

    Examples
    --------
    >>> array_equivalent(
    ...     np.array([1, 2, np.nan]),
    ...     np.array([1, 2, np.nan]))
    True
    >>> array_equivalent(
    ...     np.array([1, np.nan, 2]),
    ...     np.array([1, 2, np.nan]))
    False
    """
    left, right = np.asarray(left), np.asarray(right)

    # shape compat
    if left.shape != right.shape:
        return False

    if dtype_equal:
        # fastpath when we require that the dtypes match (Block.equals)
        if left.dtype.kind in "fc":
            return _array_equivalent_float(left, right)
        elif left.dtype.kind in "mM":
            return _array_equivalent_datetimelike(left, right)
        elif is_string_or_object_np_dtype(left.dtype):
            # TODO: fastpath for pandas' StringDtype
            return _array_equivalent_object(left, right, strict_nan)
        else:
            return np.array_equal(left, right)

    # Slow path when we allow comparing different dtypes.
    # Object arrays can contain None, NaN and NaT.
    # string dtypes must be come to this path for NumPy 1.7.1 compat
    if left.dtype.kind in "OSU" or right.dtype.kind in "OSU":
        # Note: `in "OSU"` is non-trivially faster than `in ["O", "S", "U"]`
        #  or `in ("O", "S", "U")`
        return _array_equivalent_object(left, right, strict_nan)

    # NaNs can occur in float and complex arrays.
    if left.dtype.kind in "fc":
        if not (left.size and right.size):
            return True
        return ((left == right) | (isna(left) & isna(right))).all()

    elif left.dtype.kind in "mM" or right.dtype.kind in "mM":
        # datetime64, timedelta64, Period
        if left.dtype != right.dtype:
            return False

        left = left.view("i8")
        right = right.view("i8")

    # if we have structured dtypes, compare first
    if (
        left.dtype.type is np.void or right.dtype.type is np.void
    ) and left.dtype != right.dtype:
        return False

    return np.array_equal(left, right)


def _array_equivalent_float(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(((left == right) | (np.isnan(left) & np.isnan(right))).all())


def _array_equivalent_datetimelike(left: np.ndarray, right: np.ndarray):
    return np.array_equal(left.view("i8"), right.view("i8"))


def _array_equivalent_object(left: np.ndarray, right: np.ndarray, strict_nan: bool):
    left = ensure_object(left)
    right = ensure_object(right)

    mask: npt.NDArray[np.bool_] | None = None
    if strict_nan:
        mask = isna(left) & isna(right)
        if not mask.any():
            mask = None

    try:
        if mask is None:
            return lib.array_equivalent_object(left, right)
        if not lib.array_equivalent_object(left[~mask], right[~mask]):
            return False
        left_remaining = left[mask]
        right_remaining = right[mask]
    except ValueError:
        # can raise a ValueError if left and right cannot be
        # compared (e.g. nested arrays)
        left_remaining = left
        right_remaining = right

    for left_value, right_value in zip(left_remaining, right_remaining):
        if left_value is NaT and right_value is not NaT:
            return False

        elif left_value is libmissing.NA and right_value is not libmissing.NA:
            return False

        elif isinstance(left_value, float) and np.isnan(left_value):
            if not isinstance(right_value, float) or not np.isnan(right_value):
                return False
        else:
            with warnings.catch_warnings():
                # suppress numpy's "elementwise comparison failed"
                warnings.simplefilter("ignore", DeprecationWarning)
                try:
                    if np.any(np.asarray(left_value != right_value)):
                        return False
                except TypeError as err:
                    if "boolean value of NA is ambiguous" in str(err):
                        return False
                    raise
                except ValueError:
                    # numpy can raise a ValueError if left and right cannot be
                    # compared (e.g. nested arrays)
                    return False
    return True


def array_equals(left: ArrayLike, right: ArrayLike) -> bool:
    """
    ExtensionArray-compatible implementation of array_equivalent.
    """
    if left.dtype != right.dtype:
        return False
    elif isinstance(left, ABCExtensionArray):
        return left.equals(right)
    else:
        return array_equivalent(left, right, dtype_equal=True)


def infer_fill_value(val):
    """
    infer the fill value for the nan/NaT from the provided
    scalar/ndarray/list-like if we are a NaT, return the correct dtyped
    element to provide proper block construction
    """
    if not is_list_like(val):
        val = [val]
    val = np.asarray(val)
    if val.dtype.kind in "mM":
        return np.array("NaT", dtype=val.dtype)
    elif val.dtype == object:
        dtype = lib.infer_dtype(ensure_object(val), skipna=False)
        if dtype in ["datetime", "datetime64"]:
            return np.array("NaT", dtype=DT64NS_DTYPE)
        elif dtype in ["timedelta", "timedelta64"]:
            return np.array("NaT", dtype=TD64NS_DTYPE)
        return np.array(np.nan, dtype=object)
    elif val.dtype.kind == "U":
        return np.array(np.nan, dtype=val.dtype)
    return np.nan


def construct_1d_array_from_inferred_fill_value(
    value: object, length: int
) -> ArrayLike:
    # Find our empty_value dtype by constructing an array
    #  from our value and doing a .take on it
    from pandas.core.algorithms import take_nd
    from pandas.core.construction import sanitize_array
    from pandas.core.indexes.base import Index

    arr = sanitize_array(value, Index(range(1)), copy=False)
    taker = -1 * np.ones(length, dtype=np.intp)
    return take_nd(arr, taker)


def maybe_fill(arr: np.ndarray) -> np.ndarray:
    """
    Fill numpy.ndarray with NaN, unless we have a integer or boolean dtype.
    """
    if arr.dtype.kind not in "iub":
        arr.fill(np.nan)
    return arr


def na_value_for_dtype(dtype: DtypeObj, compat: bool = True):
    """
    Return a dtype compat na value

    Parameters
    ----------
    dtype : string / dtype
    compat : bool, default True

    Returns
    -------
    np.dtype or a pandas dtype

    Examples
    --------
    >>> na_value_for_dtype(np.dtype('int64'))
    0
    >>> na_value_for_dtype(np.dtype('int64'), compat=False)
    nan
    >>> na_value_for_dtype(np.dtype('float64'))
    nan
    >>> na_value_for_dtype(np.dtype('bool'))
    False
    >>> na_value_for_dtype(np.dtype('datetime64[ns]'))
    numpy.datetime64('NaT')
    """

    if isinstance(dtype, ExtensionDtype):
        return dtype.na_value
    elif dtype.kind in "mM":
        unit = np.datetime_data(dtype)[0]
        return dtype.type("NaT", unit)
    elif dtype.kind == "f":
        return np.nan
    elif dtype.kind in "iu":
        if compat:
            return 0
        return np.nan
    elif dtype.kind == "b":
        if compat:
            return False
        return np.nan
    return np.nan


def remove_na_arraylike(arr: Series | Index | np.ndarray):
    """
    Return array-like containing only true/non-NaN values, possibly empty.
    """
    if isinstance(arr.dtype, ExtensionDtype):
        return arr[notna(arr)]
    else:
        return arr[notna(np.asarray(arr))]


def is_valid_na_for_dtype(obj, dtype: DtypeObj) -> bool:
    """
    isna check that excludes incompatible dtypes

    Parameters
    ----------
    obj : object
    dtype : np.datetime64, np.timedelta64, DatetimeTZDtype, or PeriodDtype

    Returns
    -------
    bool
    """
    if not lib.is_scalar(obj) or not isna(obj):
        return False
    elif dtype.kind == "M":
        if isinstance(dtype, np.dtype):
            # i.e. not tzaware
            return not isinstance(obj, (np.timedelta64, Decimal))
        # we have to rule out tznaive dt64("NaT")
        return not isinstance(obj, (np.timedelta64, np.datetime64, Decimal))
    elif dtype.kind == "m":
        return not isinstance(obj, (np.datetime64, Decimal))
    elif dtype.kind in "iufc":
        # Numeric
        return obj is not NaT and not isinstance(obj, (np.datetime64, np.timedelta64))
    elif dtype.kind == "b":
        # We allow pd.NA, None, np.nan in BooleanArray (same as IntervalDtype)
        return lib.is_float(obj) or obj is None or obj is libmissing.NA

    elif dtype == _dtype_str:
        # numpy string dtypes to avoid float np.nan
        return not isinstance(obj, (np.datetime64, np.timedelta64, Decimal, float))

    elif dtype == _dtype_object:
        # This is needed for Categorical, but is kind of weird
        return True

    elif isinstance(dtype, PeriodDtype):
        return not isinstance(obj, (np.datetime64, np.timedelta64, Decimal))

    elif isinstance(dtype, IntervalDtype):
        return lib.is_float(obj) or obj is None or obj is libmissing.NA

    elif isinstance(dtype, CategoricalDtype):
        return is_valid_na_for_dtype(obj, dtype.categories.dtype)

    # fallback, default to allowing NaN, None, NA, NaT
    return not isinstance(obj, (np.datetime64, np.timedelta64, Decimal))


def isna_all(arr: ArrayLike) -> bool:
    """
    Optimized equivalent to isna(arr).all()
    """
    total_len = len(arr)

    # Usually it's enough to check but a small fraction of values to see if
    #  a block is NOT null, chunks should help in such cases.
    #  parameters 1000 and 40 were chosen arbitrarily
    chunk_len = max(total_len // 40, 1000)

    dtype = arr.dtype
    if lib.is_np_dtype(dtype, "f"):
        checker = nan_checker

    elif (lib.is_np_dtype(dtype, "mM")) or isinstance(
        dtype, (DatetimeTZDtype, PeriodDtype)
    ):
        # error: Incompatible types in assignment (expression has type
        # "Callable[[Any], Any]", variable has type "ufunc")
        checker = lambda x: np.asarray(x.view("i8")) == iNaT  # type: ignore[assignment]

    else:
        # error: Incompatible types in assignment (expression has type "Callable[[Any],
        # Any]", variable has type "ufunc")
        checker = lambda x: _isna_array(  # type: ignore[assignment]
            x, inf_as_na=INF_AS_NA
        )

    return all(
        checker(arr[i : i + chunk_len]).all() for i in range(0, total_len, chunk_len)
    )
<|endoftext|>"""Indexer objects for computing start/end window bounds for rolling operations"""
from __future__ import annotations

from datetime import timedelta

import numpy as np

from pandas._libs.tslibs import BaseOffset
from pandas._libs.window.indexers import calculate_variable_window_bounds
from pandas.util._decorators import Appender

from pandas.core.dtypes.common import ensure_platform_int

from pandas.core.indexes.datetimes import DatetimeIndex

from pandas.tseries.offsets import Nano

get_window_bounds_doc = """
Computes the bounds of a window.

Parameters
----------
num_values : int, default 0
    number of values that will be aggregated over
window_size : int, default 0
    the number of rows in a window
min_periods : int, default None
    min_periods passed from the top level rolling API
center : bool, default None
    center passed from the top level rolling API
closed : str, default None
    closed passed from the top level rolling API
step : int, default None
    step passed from the top level rolling API
    .. versionadded:: 1.5
win_type : str, default None
    win_type passed from the top level rolling API

Returns
-------
A tuple of ndarray[int64]s, indicating the boundaries of each
window
"""


class BaseIndexer:
    """
    Base class for window bounds calculations.

    Examples
    --------
    >>> from pandas.api.indexers import BaseIndexer
    >>> class CustomIndexer(BaseIndexer):
    ...     def get_window_bounds(self, num_values, min_periods, center, closed, step):
    ...         start = np.empty(num_values, dtype=np.int64)
    ...         end = np.empty(num_values, dtype=np.int64)
    ...         for i in range(num_values):
    ...             start[i] = i
    ...             end[i] = i + self.window_size
    ...         return start, end
    >>> df = pd.DataFrame({"values": range(5)})
    >>> indexer = CustomIndexer(window_size=2)
    >>> df.rolling(indexer).sum()
        values
    0	1.0
    1	3.0
    2	5.0
    3	7.0
    4	4.0
    """

    def __init__(
        self, index_array: np.ndarray | None = None, window_size: int = 0, **kwargs
    ) -> None:
        self.index_array = index_array
        self.window_size = window_size
        # Set user defined kwargs as attributes that can be used in get_window_bounds
        for key, value in kwargs.items():
            setattr(self, key, value)

    @Appender(get_window_bounds_doc)
    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: int | None = None,
        center: bool | None = None,
        closed: str | None = None,
        step: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class FixedWindowIndexer(BaseIndexer):
    """Creates window boundaries that are of fixed length."""

    @Appender(get_window_bounds_doc)
    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: int | None = None,
        center: bool | None = None,
        closed: str | None = None,
        step: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if center or self.window_size == 0:
            offset = (self.window_size - 1) // 2
        else:
            offset = 0

        end = np.arange(1 + offset, num_values + 1 + offset, step, dtype="int64")
        start = end - self.window_size
        if closed in ["left", "both"]:
            start -= 1
        if closed in ["left", "neither"]:
            end -= 1

        end = np.clip(end, 0, num_values)
        start = np.clip(start, 0, num_values)

        return start, end


class VariableWindowIndexer(BaseIndexer):
    """Creates window boundaries that are of variable length, namely for time series."""

    @Appender(get_window_bounds_doc)
    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: int | None = None,
        center: bool | None = None,
        closed: str | None = None,
        step: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        # error: Argument 4 to "calculate_variable_window_bounds" has incompatible
        # type "Optional[bool]"; expected "bool"
        # error: Argument 6 to "calculate_variable_window_bounds" has incompatible
        # type "Optional[ndarray]"; expected "ndarray"
        return calculate_variable_window_bounds(
            num_values,
            self.window_size,
            min_periods,
            center,  # type: ignore[arg-type]
            closed,
            self.index_array,  # type: ignore[arg-type]
        )


class VariableOffsetWindowIndexer(BaseIndexer):
    """
    Calculate window boundaries based on a non-fixed offset such as a BusinessDay.

    Examples
    --------
    >>> from pandas.api.indexers import VariableOffsetWindowIndexer
    >>> df = pd.DataFrame(range(10), index=pd.date_range("2020", periods=10))
    >>> offset = pd.offsets.BDay(1)
    >>> indexer = VariableOffsetWindowIndexer(index=df.index, offset=offset)
    >>> df
                0
    2020-01-01  0
    2020-01-02  1
    2020-01-03  2
    2020-01-04  3
    2020-01-05  4
    2020-01-06  5
    2020-01-07  6
    2020-01-08  7
    2020-01-09  8
    2020-01-10  9
    >>> df.rolling(indexer).sum()
                   0
    2020-01-01   0.0
    2020-01-02   1.0
    2020-01-03   2.0
    2020-01-04   3.0
    2020-01-05   7.0
    2020-01-06  12.0
    2020-01-07   6.0
    2020-01-08   7.0
    2020-01-09   8.0
    2020-01-10   9.0
    """

    def __init__(
        self,
        index_array: np.ndarray | None = None,
        window_size: int = 0,
        index: DatetimeIndex | None = None,
        offset: BaseOffset | None = None,
        **kwargs,
    ) -> None:
        super().__init__(index_array, window_size, **kwargs)
        if not isinstance(index, DatetimeIndex):
            raise ValueError("index must be a DatetimeIndex.")
        self.index = index
        if not isinstance(offset, BaseOffset):
            raise ValueError("offset must be a DateOffset-like object.")
        self.offset = offset

    @Appender(get_window_bounds_doc)
    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: int | None = None,
        center: bool | None = None,
        closed: str | None = None,
        step: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if step is not None:
            raise NotImplementedError("step not implemented for variable offset window")
        if num_values <= 0:
            return np.empty(0, dtype="int64"), np.empty(0, dtype="int64")

        # if windows is variable, default is 'right', otherwise default is 'both'
        if closed is None:
            closed = "right" if self.index is not None else "both"

        right_closed = closed in ["right", "both"]
        left_closed = closed in ["left", "both"]

        if self.index[num_values - 1] < self.index[0]:
            index_growth_sign = -1
        else:
            index_growth_sign = 1
        offset_diff = index_growth_sign * self.offset

        start = np.empty(num_values, dtype="int64")
        start.fill(-1)
        end = np.empty(num_values, dtype="int64")
        end.fill(-1)

        start[0] = 0

        # right endpoint is closed
        if right_closed:
            end[0] = 1
        # right endpoint is open
        else:
            end[0] = 0

        zero = timedelta(0)
        # start is start of slice interval (including)
        # end is end of slice interval (not including)
        for i in range(1, num_values):
            end_bound = self.index[i]
            start_bound = end_bound - offset_diff

            # left endpoint is closed
            if left_closed:
                start_bound -= Nano(1)

            # advance the start bound until we are
            # within the constraint
            start[i] = i
            for j in range(start[i - 1], i):
                start_diff = (self.index[j] - start_bound) * index_growth_sign
                if start_diff > zero:
                    start[i] = j
                    break

            # end bound is previous end
            # or current index
            end_diff = (self.index[end[i - 1]] - end_bound) * index_growth_sign
            if end_diff == zero and not right_closed:
                end[i] = end[i - 1] + 1
            elif end_diff <= zero:
                end[i] = i + 1
            else:
                end[i] = end[i - 1]

            # right endpoint is open
            if not right_closed:
                end[i] -= 1

        return start, end


class ExpandingIndexer(BaseIndexer):
    """Calculate expanding window bounds, mimicking df.expanding()"""

    @Appender(get_window_bounds_doc)
    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: int | None = None,
        center: bool | None = None,
        closed: str | None = None,
        step: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.zeros(num_values, dtype=np.int64),
            np.arange(1, num_values + 1, dtype=np.int64),
        )


class FixedForwardWindowIndexer(BaseIndexer):
    """
    Creates window boundaries for fixed-length windows that include the current row.

    Examples
    --------
    >>> df = pd.DataFrame({'B': [0, 1, 2, np.nan, 4]})
    >>> df
         B
    0  0.0
    1  1.0
    2  2.0
    3  NaN
    4  4.0

    >>> indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=2)
    >>> df.rolling(window=indexer, min_periods=1).sum()
         B
    0  1.0
    1  3.0
    2  2.0
    3  4.0
    4  4.0
    """

    @Appender(get_window_bounds_doc)
    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: int | None = None,
        center: bool | None = None,
        closed: str | None = None,
        step: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if center:
            raise ValueError("Forward-looking windows can't have center=True")
        if closed is not None:
            raise ValueError(
                "Forward-looking windows don't support setting the closed argument"
            )
        if step is None:
            step = 1

        start = np.arange(0, num_values, step, dtype="int64")
        end = start + self.window_size
        if self.window_size:
            end = np.clip(end, 0, num_values)

        return start, end


class GroupbyIndexer(BaseIndexer):
    """Calculate bounds to compute groupby rolling, mimicking df.groupby().rolling()"""

    def __init__(
        self,
        index_array: np.ndarray | None = None,
        window_size: int | BaseIndexer = 0,
        groupby_indices: dict | None = None,
        window_indexer: type[BaseIndexer] = BaseIndexer,
        indexer_kwargs: dict | None = None,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        index_array : np.ndarray or None
            np.ndarray of the index of the original object that we are performing
            a chained groupby operation over. This index has been pre-sorted relative to
            the groups
        window_size : int or BaseIndexer
            window size during the windowing operation
        groupby_indices : dict or None
            dict of {group label: [positional index of rows belonging to the group]}
        window_indexer : BaseIndexer
            BaseIndexer class determining the start and end bounds of each group
        indexer_kwargs : dict or None
            Custom kwargs to be passed to window_indexer
        **kwargs :
            keyword arguments that will be available when get_window_bounds is called
        """
        self.groupby_indices = groupby_indices or {}
        self.window_indexer = window_indexer
        self.indexer_kwargs = indexer_kwargs.copy() if indexer_kwargs else {}
        super().__init__(
            index_array=index_array,
            window_size=self.indexer_kwargs.pop("window_size", window_size),
            **kwargs,
        )

    @Appender(get_window_bounds_doc)
    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: int | None = None,
        center: bool | None = None,
        closed: str | None = None,
        step: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        # 1) For each group, get the indices that belong to the group
        # 2) Use the indices to calculate the start & end bounds of the window
        # 3) Append the window bounds in group order
        start_arrays = []
        end_arrays = []
        window_indices_start = 0
        for key, indices in self.groupby_indices.items():
            index_array: np.ndarray | None

            if self.index_array is not None:
                index_array = self.index_array.take(ensure_platform_int(indices))
            else:
                index_array = self.index_array
            indexer = self.window_indexer(
                index_array=index_array,
                window_size=self.window_size,
                **self.indexer_kwargs,
            )
            start, end = indexer.get_window_bounds(
                len(indices), min_periods, center, closed, step
            )
            start = start.astype(np.int64)
            end = end.astype(np.int64)
            assert len(start) == len(
                end
            ), "these should be equal in length from get_window_bounds"
            # Cannot use groupby_indices as they might not be monotonic with the object
            # we're rolling over
            window_indices = np.arange(
                window_indices_start, window_indices_start + len(indices)
            )
            window_indices_start += len(indices)
            # Extend as we'll be slicing window like [start, end)
            window_indices = np.append(window_indices, [window_indices[-1] + 1]).astype(
                np.int64, copy=False
            )
            start_arrays.append(window_indices.take(ensure_platform_int(start)))
            end_arrays.append(window_indices.take(ensure_platform_int(end)))
        if len(start_arrays) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        start = np.concatenate(start_arrays)
        end = np.concatenate(end_arrays)
        return start, end


class ExponentialMovingWindowIndexer(BaseIndexer):
    """Calculate ewm window bounds (the entire window)"""

    @Appender(get_window_bounds_doc)
    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: int | None = None,
        center: bool | None = None,
        closed: str | None = None,
        step: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return np.array([0], dtype=np.int64), np.array([num_values], dtype=np.int64)
<|endoftext|>"""
Low-dependency indexing utilities.
"""
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np

from pandas._libs import lib

from pandas.core.dtypes.common import (
    is_array_like,
    is_bool_dtype,
    is_integer,
    is_integer_dtype,
    is_list_like,
)
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
    ABCIndex,
    ABCSeries,
)

if TYPE_CHECKING:
    from pandas._typing import AnyArrayLike

    from pandas.core.frame import DataFrame
    from pandas.core.indexes.base import Index

# -----------------------------------------------------------
# Indexer Identification


def is_valid_positional_slice(slc: slice) -> bool:
    """
    Check if a slice object can be interpreted as a positional indexer.

    Parameters
    ----------
    slc : slice

    Returns
    -------
    bool

    Notes
    -----
    A valid positional slice may also be interpreted as a label-based slice
    depending on the index being sliced.
    """
    return (
        lib.is_int_or_none(slc.start)
        and lib.is_int_or_none(slc.stop)
        and lib.is_int_or_none(slc.step)
    )


def is_list_like_indexer(key) -> bool:
    """
    Check if we have a list-like indexer that is *not* a NamedTuple.

    Parameters
    ----------
    key : object

    Returns
    -------
    bool
    """
    # allow a list_like, but exclude NamedTuples which can be indexers
    return is_list_like(key) and not (isinstance(key, tuple) and type(key) is not tuple)


def is_scalar_indexer(indexer, ndim: int) -> bool:
    """
    Return True if we are all scalar indexers.

    Parameters
    ----------
    indexer : object
    ndim : int
        Number of dimensions in the object being indexed.

    Returns
    -------
    bool
    """
    if ndim == 1 and is_integer(indexer):
        # GH37748: allow indexer to be an integer for Series
        return True
    if isinstance(indexer, tuple) and len(indexer) == ndim:
        return all(is_integer(x) for x in indexer)
    return False


def is_empty_indexer(indexer) -> bool:
    """
    Check if we have an empty indexer.

    Parameters
    ----------
    indexer : object

    Returns
    -------
    bool
    """
    if is_list_like(indexer) and not len(indexer):
        return True
    if not isinstance(indexer, tuple):
        indexer = (indexer,)
    return any(isinstance(idx, np.ndarray) and len(idx) == 0 for idx in indexer)


