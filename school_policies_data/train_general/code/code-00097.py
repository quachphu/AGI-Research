
        Allowed inputs for the index are:

        - An integer valued iterable, e.g. ``range(2, 4)``.
        - A comma separated list of integers and slices, e.g. ``5``, ``2, 4``, ``2:4``.

        The output format is the same as :meth:`~GroupBy.head` and
        :meth:`~GroupBy.tail`, namely
        a subset of the ``DataFrame`` or ``Series`` with the index and order preserved.

        Returns
        -------
        Series
            The filtered subset of the original Series.
        DataFrame
            The filtered subset of the original DataFrame.

        See Also
        --------
        DataFrame.iloc : Purely integer-location based indexing for selection by
            position.
        GroupBy.head : Return first n rows of each group.
        GroupBy.tail : Return last n rows of each group.
        GroupBy.nth : Take the nth row from each group if n is an int, or a
            subset of rows, if n is a list of ints.

        Notes
        -----
        - The slice step cannot be negative.
        - If the index specification results in overlaps, the item is not duplicated.
        - If the index specification changes the order of items, then
          they are returned in their original order.
          By contrast, ``DataFrame.iloc`` can change the row order.
        - ``groupby()`` parameters such as as_index and dropna are ignored.

        The differences between ``_positional_selector[]`` and :meth:`~GroupBy.nth`
        with ``as_index=False`` are:

        - Input to ``_positional_selector`` can include
          one or more slices whereas ``nth``
          just handles an integer or a list of integers.
        - ``_positional_selector`` can  accept a slice relative to the
          last row of each group.
        - ``_positional_selector`` does not have an equivalent to the
          ``nth()`` ``dropna`` parameter.

        Examples
        --------
        >>> df = pd.DataFrame([["a", 1], ["a", 2], ["a", 3], ["b", 4], ["b", 5]],
        ...                   columns=["A", "B"])
        >>> df.groupby("A")._positional_selector[1:2]
           A  B
        1  a  2
        4  b  5

        >>> df.groupby("A")._positional_selector[1, -1]
           A  B
        1  a  2
        2  a  3
        4  b  5
        """
        if TYPE_CHECKING:
            # pylint: disable-next=used-before-assignment
            groupby_self = cast(groupby.GroupBy, self)
        else:
            groupby_self = self

        return GroupByPositionalSelector(groupby_self)

    def _make_mask_from_positional_indexer(
        self,
        arg: PositionalIndexer | tuple,
    ) -> np.ndarray:
        if is_list_like(arg):
            if all(is_integer(i) for i in cast(Iterable, arg)):
                mask = self._make_mask_from_list(cast(Iterable[int], arg))
            else:
                mask = self._make_mask_from_tuple(cast(tuple, arg))

        elif isinstance(arg, slice):
            mask = self._make_mask_from_slice(arg)
        elif is_integer(arg):
            mask = self._make_mask_from_int(cast(int, arg))
        else:
            raise TypeError(
                f"Invalid index {type(arg)}. "
                "Must be integer, list-like, slice or a tuple of "
                "integers and slices"
            )

        if isinstance(mask, bool):
            if mask:
                mask = self._ascending_count >= 0
            else:
                mask = self._ascending_count < 0

        return cast(np.ndarray, mask)

    def _make_mask_from_int(self, arg: int) -> np.ndarray:
        if arg >= 0:
            return self._ascending_count == arg
        else:
            return self._descending_count == (-arg - 1)

    def _make_mask_from_list(self, args: Iterable[int]) -> bool | np.ndarray:
        positive = [arg for arg in args if arg >= 0]
        negative = [-arg - 1 for arg in args if arg < 0]

        mask: bool | np.ndarray = False

        if positive:
            mask |= np.isin(self._ascending_count, positive)

        if negative:
            mask |= np.isin(self._descending_count, negative)

        return mask

    def _make_mask_from_tuple(self, args: tuple) -> bool | np.ndarray:
        mask: bool | np.ndarray = False

        for arg in args:
            if is_integer(arg):
                mask |= self._make_mask_from_int(cast(int, arg))
            elif isinstance(arg, slice):
                mask |= self._make_mask_from_slice(arg)
            else:
                raise ValueError(
                    f"Invalid argument {type(arg)}. Should be int or slice."
                )

        return mask

    def _make_mask_from_slice(self, arg: slice) -> bool | np.ndarray:
        start = arg.start
        stop = arg.stop
        step = arg.step

        if step is not None and step < 0:
            raise ValueError(f"Invalid step {step}. Must be non-negative")

        mask: bool | np.ndarray = True

        if step is None:
            step = 1

        if start is None:
            if step > 1:
                mask &= self._ascending_count % step == 0

        elif start >= 0:
            mask &= self._ascending_count >= start

            if step > 1:
                mask &= (self._ascending_count - start) % step == 0

        else:
            mask &= self._descending_count < -start

            offset_array = self._descending_count + start + 1
            limit_array = (
                self._ascending_count + self._descending_count + (start + 1)
            ) < 0
            offset_array = np.where(limit_array, self._ascending_count, offset_array)

            mask &= offset_array % step == 0

        if stop is not None:
            if stop >= 0:
                mask &= self._ascending_count < stop
            else:
                mask &= self._descending_count >= -stop

        return mask

    @cache_readonly
    def _ascending_count(self) -> np.ndarray:
        if TYPE_CHECKING:
            groupby_self = cast(groupby.GroupBy, self)
        else:
            groupby_self = self

        return groupby_self._cumcount_array()

    @cache_readonly
    def _descending_count(self) -> np.ndarray:
        if TYPE_CHECKING:
            groupby_self = cast(groupby.GroupBy, self)
        else:
            groupby_self = self

        return groupby_self._cumcount_array(ascending=False)


@doc(GroupByIndexingMixin._positional_selector)
class GroupByPositionalSelector:
    def __init__(self, groupby_object: groupby.GroupBy) -> None:
        self.groupby_object = groupby_object

    def __getitem__(self, arg: PositionalIndexer | tuple) -> DataFrame | Series:
        """
        Select by positional index per group.

        Implements GroupBy._positional_selector

        Parameters
        ----------
        arg : PositionalIndexer | tuple
            Allowed values are:
            - int
            - int valued iterable such as list or range
            - slice with step either None or positive
            - tuple of integers and slices

        Returns
        -------
        Series
            The filtered subset of the original groupby Series.
        DataFrame
            The filtered subset of the original groupby DataFrame.

        See Also
        --------
        DataFrame.iloc : Integer-location based indexing for selection by position.
        GroupBy.head : Return first n rows of each group.
        GroupBy.tail : Return last n rows of each group.
        GroupBy._positional_selector : Return positional selection for each group.
        GroupBy.nth : Take the nth row from each group if n is an int, or a
            subset of rows, if n is a list of ints.
        """
        mask = self.groupby_object._make_mask_from_positional_indexer(arg)
        return self.groupby_object._mask_selected_obj(mask)


class GroupByNthSelector:
    """
    Dynamically substituted for GroupBy.nth to enable both call and index
    """

    def __init__(self, groupby_object: groupby.GroupBy) -> None:
        self.groupby_object = groupby_object

    def __call__(
        self,
        n: PositionalIndexer | tuple,
        dropna: Literal["any", "all", None] = None,
    ) -> DataFrame | Series:
        return self.groupby_object._nth(n, dropna)

    def __getitem__(self, n: PositionalIndexer | tuple) -> DataFrame | Series:
        return self.groupby_object._nth(n)
<|endoftext|>from __future__ import annotations

import functools
import re
import textwrap
from typing import (
    TYPE_CHECKING,
    Callable,
    Literal,
    cast,
)
import unicodedata
import warnings

import numpy as np

from pandas._libs import lib
import pandas._libs.missing as libmissing
import pandas._libs.ops as libops
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.missing import isna

from pandas.core.strings.base import BaseStringArrayMethods

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pandas._typing import (
        NpDtype,
        Scalar,
    )

    from pandas import Series


class ObjectStringArrayMixin(BaseStringArrayMethods):
    """
    String Methods operating on object-dtype ndarrays.
    """

    def __len__(self) -> int:
        # For typing, _str_map relies on the object being sized.
        raise NotImplementedError

    def _str_map(
        self,
        f,
        na_value=lib.no_default,
        dtype: NpDtype | None = None,
        convert: bool = True,
    ):
        """
        Map a callable over valid elements of the array.

        Parameters
        ----------
        f : Callable
            A function to call on each non-NA element.
        na_value : Scalar, optional
            The value to set for NA values. Might also be used for the
            fill value if the callable `f` raises an exception.
            This defaults to ``self.dtype.na_value`` which is ``np.nan``
            for object-dtype and Categorical and ``pd.NA`` for StringArray.
        dtype : Dtype, optional
            The dtype of the result array.
        convert : bool, default True
            Whether to call `maybe_convert_objects` on the resulting ndarray
        """
        if dtype is None:
            dtype = np.dtype("object")
        if na_value is lib.no_default:
            na_value = self.dtype.na_value  # type: ignore[attr-defined]

        if not len(self):
            return np.array([], dtype=dtype)

        arr = np.asarray(self, dtype=object)
        mask = isna(arr)
        map_convert = convert and not np.all(mask)
        try:
            result = lib.map_infer_mask(arr, f, mask.view(np.uint8), map_convert)
        except (TypeError, AttributeError) as err:
            # Reraise the exception if callable `f` got wrong number of args.
            # The user may want to be warned by this, instead of getting NaN
            p_err = (
                r"((takes)|(missing)) (?(2)from \d+ to )?\d+ "
                r"(?(3)required )positional arguments?"
            )

            if len(err.args) >= 1 and re.search(p_err, err.args[0]):
                # FIXME: this should be totally avoidable
                raise err

            def g(x):
                # This type of fallback behavior can be removed once
                # we remove object-dtype .str accessor.
                try:
                    return f(x)
                except (TypeError, AttributeError):
                    return na_value

            return self._str_map(g, na_value=na_value, dtype=dtype)
        if not isinstance(result, np.ndarray):
            return result
        if na_value is not np.nan:
            np.putmask(result, mask, na_value)
            if convert and result.dtype == object:
                result = lib.maybe_convert_objects(result)
        return result

    def _str_count(self, pat, flags: int = 0):
        regex = re.compile(pat, flags=flags)
        f = lambda x: len(regex.findall(x))
        return self._str_map(f, dtype="int64")

    def _str_pad(
        self,
        width: int,
        side: Literal["left", "right", "both"] = "left",
        fillchar: str = " ",
    ):
        if side == "left":
            f = lambda x: x.rjust(width, fillchar)
        elif side == "right":
            f = lambda x: x.ljust(width, fillchar)
        elif side == "both":
            f = lambda x: x.center(width, fillchar)
        else:  # pragma: no cover
            raise ValueError("Invalid side")
        return self._str_map(f)

    def _str_contains(
        self,
        pat,
        case: bool = True,
        flags: int = 0,
        na=lib.no_default,
        regex: bool = True,
    ):
        if regex:
            if not case:
                flags |= re.IGNORECASE

            pat = re.compile(pat, flags=flags)

            f = lambda x: pat.search(x) is not None
        else:
            if case:
                f = lambda x: pat in x
            else:
                upper_pat = pat.upper()
                f = lambda x: upper_pat in x.upper()
        if na is not lib.no_default and not isna(na) and not isinstance(na, bool):
            # GH#59561
            warnings.warn(
                "Allowing a non-bool 'na' in obj.str.contains is deprecated "
                "and will raise in a future version.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )
        return self._str_map(f, na, dtype=np.dtype("bool"))

    def _str_startswith(self, pat, na=lib.no_default):
        f = lambda x: x.startswith(pat)
        if na is not lib.no_default and not isna(na) and not isinstance(na, bool):
            # GH#59561
            warnings.warn(
                "Allowing a non-bool 'na' in obj.str.startswith is deprecated "
                "and will raise in a future version.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )
        return self._str_map(f, na_value=na, dtype=np.dtype(bool))

    def _str_endswith(self, pat, na=lib.no_default):
        f = lambda x: x.endswith(pat)
        if na is not lib.no_default and not isna(na) and not isinstance(na, bool):
            # GH#59561
            warnings.warn(
                "Allowing a non-bool 'na' in obj.str.endswith is deprecated "
                "and will raise in a future version.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )
        return self._str_map(f, na_value=na, dtype=np.dtype(bool))

    def _str_replace(
        self,
        pat: str | re.Pattern,
        repl: str | Callable,
        n: int = -1,
        case: bool = True,
        flags: int = 0,
        regex: bool = True,
    ):
        if case is False:
            # add case flag, if provided
            flags |= re.IGNORECASE

        if regex or flags or callable(repl):
            if not isinstance(pat, re.Pattern):
                if regex is False:
                    pat = re.escape(pat)
                pat = re.compile(pat, flags=flags)

            n = n if n >= 0 else 0
            f = lambda x: pat.sub(repl=repl, string=x, count=n)
        else:
            f = lambda x: x.replace(pat, repl, n)

        return self._str_map(f, dtype=str)

    def _str_repeat(self, repeats: int | Sequence[int]):
        if lib.is_integer(repeats):
            rint = cast(int, repeats)

            def scalar_rep(x):
                try:
                    return bytes.__mul__(x, rint)
                except TypeError:
                    return str.__mul__(x, rint)

            return self._str_map(scalar_rep, dtype=str)
        else:
            from pandas.core.arrays.string_ import BaseStringArray

            def rep(x, r):
                if x is libmissing.NA:
                    return x
                try:
                    return bytes.__mul__(x, r)
                except TypeError:
                    return str.__mul__(x, r)

            result = libops.vec_binop(
                np.asarray(self),
                np.asarray(repeats, dtype=object),
                rep,
            )
            if isinstance(self, BaseStringArray):
                # Not going through map, so we have to do this here.
                result = type(self)._from_sequence(result, dtype=self.dtype)
            return result

    def _str_match(
        self,
        pat: str | re.Pattern,
        case: bool = True,
        flags: int = 0,
        na: Scalar | lib.NoDefault = lib.no_default,
    ):
        if not case:
            flags |= re.IGNORECASE

        regex = re.compile(pat, flags=flags)

        f = lambda x: regex.match(x) is not None
        return self._str_map(f, na_value=na, dtype=np.dtype(bool))

    def _str_fullmatch(
        self,
        pat: str | re.Pattern,
        case: bool = True,
        flags: int = 0,
        na: Scalar | lib.NoDefault = lib.no_default,
    ):
        if not case:
            flags |= re.IGNORECASE

        regex = re.compile(pat, flags=flags)

        f = lambda x: regex.fullmatch(x) is not None
        return self._str_map(f, na_value=na, dtype=np.dtype(bool))

    def _str_encode(self, encoding, errors: str = "strict"):
        f = lambda x: x.encode(encoding, errors=errors)
        return self._str_map(f, dtype=object)

    def _str_find(self, sub, start: int = 0, end=None):
        return self._str_find_(sub, start, end, side="left")

    def _str_rfind(self, sub, start: int = 0, end=None):
        return self._str_find_(sub, start, end, side="right")

    def _str_find_(self, sub, start, end, side):
        if side == "left":
            method = "find"
        elif side == "right":
            method = "rfind"
        else:  # pragma: no cover
            raise ValueError("Invalid side")

        if end is None:
            f = lambda x: getattr(x, method)(sub, start)
        else:
            f = lambda x: getattr(x, method)(sub, start, end)
        return self._str_map(f, dtype="int64")

    def _str_findall(self, pat, flags: int = 0):
        regex = re.compile(pat, flags=flags)
        return self._str_map(regex.findall, dtype="object")

    def _str_get(self, i):
        def f(x):
            if isinstance(x, dict):
                return x.get(i)
            elif len(x) > i >= -len(x):
                return x[i]
            return self.dtype.na_value  # type: ignore[attr-defined]

        return self._str_map(f)

    def _str_index(self, sub, start: int = 0, end=None):
        if end:
            f = lambda x: x.index(sub, start, end)
        else:
            f = lambda x: x.index(sub, start, end)
        return self._str_map(f, dtype="int64")

    def _str_rindex(self, sub, start: int = 0, end=None):
        if end:
            f = lambda x: x.rindex(sub, start, end)
        else:
            f = lambda x: x.rindex(sub, start, end)
        return self._str_map(f, dtype="int64")

    def _str_join(self, sep: str):
        return self._str_map(sep.join)

    def _str_partition(self, sep: str, expand):
        result = self._str_map(lambda x: x.partition(sep), dtype="object")
        return result

    def _str_rpartition(self, sep: str, expand):
        return self._str_map(lambda x: x.rpartition(sep), dtype="object")

    def _str_len(self):
        return self._str_map(len, dtype="int64")

    def _str_slice(self, start=None, stop=None, step=None):
        obj = slice(start, stop, step)
        return self._str_map(lambda x: x[obj])

    def _str_slice_replace(self, start=None, stop=None, repl=None):
        if repl is None:
            repl = ""

        def f(x):
            if x[start:stop] == "":
                local_stop = start
            else:
                local_stop = stop
            y = ""
            if start is not None:
                y += x[:start]
            y += repl
            if stop is not None:
                y += x[local_stop:]
            return y

        return self._str_map(f)

    def _str_split(
        self,
        pat: str | re.Pattern | None = None,
        n=-1,
        expand: bool = False,
        regex: bool | None = None,
    ):
        if pat is None:
            if n is None or n == 0:
                n = -1
            f = lambda x: x.split(pat, n)
        else:
            new_pat: str | re.Pattern
            if regex is True or isinstance(pat, re.Pattern):
                new_pat = re.compile(pat)
            elif regex is False:
                new_pat = pat
            # regex is None so link to old behavior #43563
            else:
                if len(pat) == 1:
                    new_pat = pat
                else:
                    new_pat = re.compile(pat)

            if isinstance(new_pat, re.Pattern):
                if n is None or n == -1:
                    n = 0
                f = lambda x: new_pat.split(x, maxsplit=n)
            else:
                if n is None or n == 0:
                    n = -1
                f = lambda x: x.split(pat, n)
        return self._str_map(f, dtype=object)

    def _str_rsplit(self, pat=None, n=-1):
        if n is None or n == 0:
            n = -1
        f = lambda x: x.rsplit(pat, n)
        return self._str_map(f, dtype="object")

    def _str_translate(self, table):
        return self._str_map(lambda x: x.translate(table))

    def _str_wrap(self, width: int, **kwargs):
        kwargs["width"] = width
        tw = textwrap.TextWrapper(**kwargs)
        return self._str_map(lambda s: "\n".join(tw.wrap(s)))

    def _str_get_dummies(self, sep: str = "|"):
        from pandas import Series

        arr = Series(self).fillna("")
        try:
            arr = sep + arr + sep
        except (TypeError, NotImplementedError):
            arr = sep + arr.astype(str) + sep

        tags: set[str] = set()
        for ts in Series(arr, copy=False).str.split(sep):
            tags.update(ts)
        tags2 = sorted(tags - {""})

        dummies = np.empty((len(arr), len(tags2)), dtype=np.int64)

        def _isin(test_elements: str, element: str) -> bool:
            return element in test_elements

        for i, t in enumerate(tags2):
            pat = sep + t + sep
            dummies[:, i] = lib.map_infer(
                arr.to_numpy(), functools.partial(_isin, element=pat)
            )
        return dummies, tags2

    def _str_upper(self):
        return self._str_map(lambda x: x.upper())

    def _str_isalnum(self):
        return self._str_map(str.isalnum, dtype="bool")

    def _str_isalpha(self):
        return self._str_map(str.isalpha, dtype="bool")

    def _str_isdecimal(self):
        return self._str_map(str.isdecimal, dtype="bool")

    def _str_isdigit(self):
        return self._str_map(str.isdigit, dtype="bool")

    def _str_islower(self):
        return self._str_map(str.islower, dtype="bool")

    def _str_isnumeric(self):
        return self._str_map(str.isnumeric, dtype="bool")

    def _str_isspace(self):
        return self._str_map(str.isspace, dtype="bool")

    def _str_istitle(self):
        return self._str_map(str.istitle, dtype="bool")

    def _str_isupper(self):
        return self._str_map(str.isupper, dtype="bool")

    def _str_capitalize(self):
        return self._str_map(str.capitalize)

    def _str_casefold(self):
        return self._str_map(str.casefold)

    def _str_title(self):
        return self._str_map(str.title)

    def _str_swapcase(self):
        return self._str_map(str.swapcase)

    def _str_lower(self):
        return self._str_map(str.lower)

    def _str_normalize(self, form):
        f = lambda x: unicodedata.normalize(form, x)
        return self._str_map(f)

    def _str_strip(self, to_strip=None):
        return self._str_map(lambda x: x.strip(to_strip))

    def _str_lstrip(self, to_strip=None):
        return self._str_map(lambda x: x.lstrip(to_strip))

    def _str_rstrip(self, to_strip=None):
        return self._str_map(lambda x: x.rstrip(to_strip))

    def _str_removeprefix(self, prefix: str) -> Series:
        # outstanding question on whether to use native methods for users on Python 3.9+
        # https://github.com/pandas-dev/pandas/pull/39226#issuecomment-836719770,
        # in which case we could do return self._str_map(str.removeprefix)

        def removeprefix(text: str) -> str:
            if text.startswith(prefix):
                return text[len(prefix) :]
            return text

        return self._str_map(removeprefix)

    def _str_removesuffix(self, suffix: str) -> Series:
        return self._str_map(lambda x: x.removesuffix(suffix))

    def _str_extract(self, pat: str, flags: int = 0, expand: bool = True):
        regex = re.compile(pat, flags=flags)
        na_value = self.dtype.na_value  # type: ignore[attr-defined]

        if not expand:

            def g(x):
                m = regex.search(x)
                return m.groups()[0] if m else na_value

            return self._str_map(g, convert=False)

        empty_row = [na_value] * regex.groups

        def f(x):
            if not isinstance(x, str):
                return empty_row
            m = regex.search(x)
            if m:
                return [na_value if item is None else item for item in m.groups()]
            else:
                return empty_row

        return [f(val) for val in np.asarray(self)]
<|endoftext|>from __future__ import annotations

import codecs
from functools import wraps
import re
from typing import (
    TYPE_CHECKING,
    Callable,
    Literal,
    cast,
)
import warnings

import numpy as np

from pandas._config import get_option

from pandas._libs import lib
from pandas._typing import (
    AlignJoin,
    DtypeObj,
    F,
    Scalar,
    npt,
)
from pandas.util._decorators import Appender
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import (
    ensure_object,
    is_bool_dtype,
    is_integer,
    is_list_like,
    is_object_dtype,
    is_re,
    is_string_dtype,
)
from pandas.core.dtypes.dtypes import (
    ArrowDtype,
    CategoricalDtype,
)
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCIndex,
    ABCMultiIndex,
    ABCSeries,
)
from pandas.core.dtypes.missing import isna

from pandas.core.arrays import ExtensionArray
from pandas.core.base import NoNewAttributesMixin
from pandas.core.construction import extract_array

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Iterator,
    )

    from pandas import (
        DataFrame,
        Index,
        Series,
    )

_shared_docs: dict[str, str] = {}
_cpython_optimized_encoders = (
    "utf-8",
    "utf8",
    "latin-1",
    "latin1",
    "iso-8859-1",
    "mbcs",
    "ascii",
)
_cpython_optimized_decoders = _cpython_optimized_encoders + ("utf-16", "utf-32")


def forbid_nonstring_types(
    forbidden: list[str] | None, name: str | None = None
) -> Callable[[F], F]:
    """
    Decorator to forbid specific types for a method of StringMethods.

    For calling `.str.{method}` on a Series or Index, it is necessary to first
    initialize the :class:`StringMethods` object, and then call the method.
    However, different methods allow different input types, and so this can not
    be checked during :meth:`StringMethods.__init__`, but must be done on a
    per-method basis. This decorator exists to facilitate this process, and
    make it explicit which (inferred) types are disallowed by the method.

    :meth:`StringMethods.__init__` allows the *union* of types its different
    methods allow (after skipping NaNs; see :meth:`StringMethods._validate`),
    namely: ['string', 'empty', 'bytes', 'mixed', 'mixed-integer'].

    The default string types ['string', 'empty'] are allowed for all methods.
    For the additional types ['bytes', 'mixed', 'mixed-integer'], each method
    then needs to forbid the types it is not intended for.

    Parameters
    ----------
    forbidden : list-of-str or None
        List of forbidden non-string types, may be one or more of
        `['bytes', 'mixed', 'mixed-integer']`.
    name : str, default None
        Name of the method to use in the error message. By default, this is
        None, in which case the name from the method being wrapped will be
        copied. However, for working with further wrappers (like _pat_wrapper
        and _noarg_wrapper), it is necessary to specify the name.

    Returns
    -------
    func : wrapper
        The method to which the decorator is applied, with an added check that
        enforces the inferred type to not be in the list of forbidden types.

    Raises
    ------
    TypeError
        If the inferred type of the underlying data is in `forbidden`.
    """
    # deal with None
    forbidden = [] if forbidden is None else forbidden

    allowed_types = {"string", "empty", "bytes", "mixed", "mixed-integer"} - set(
        forbidden
    )

    def _forbid_nonstring_types(func: F) -> F:
        func_name = func.__name__ if name is None else name

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if self._inferred_dtype not in allowed_types:
                msg = (
                    f"Cannot use .str.{func_name} with values of "
                    f"inferred dtype '{self._inferred_dtype}'."
                )
                raise TypeError(msg)
            return func(self, *args, **kwargs)

        wrapper.__name__ = func_name
        return cast(F, wrapper)

    return _forbid_nonstring_types


def _map_and_wrap(name: str | None, docstring: str | None):
    @forbid_nonstring_types(["bytes"], name=name)
    def wrapper(self):
        result = getattr(self._data.array, f"_str_{name}")()
        return self._wrap_result(
            result, returns_string=name not in ("isnumeric", "isdecimal")
        )

    wrapper.__doc__ = docstring
    return wrapper


class StringMethods(NoNewAttributesMixin):
    """
    Vectorized string functions for Series and Index.

    NAs stay NA unless handled otherwise by a particular method.
    Patterned after Python's string methods, with some inspiration from
    R's stringr package.

    Examples
    --------
    >>> s = pd.Series(["A_Str_Series"])
    >>> s
    0    A_Str_Series
    dtype: object

    >>> s.str.split("_")
    0    [A, Str, Series]
    dtype: object

    >>> s.str.replace("_", "")
    0    AStrSeries
    dtype: object
    """

    # Note: see the docstring in pandas.core.strings.__init__
    # for an explanation of the implementation.
    # TODO: Dispatch all the methods
    # Currently the following are not dispatched to the array
    # * cat
    # * extractall

    def __init__(self, data) -> None:
        from pandas.core.arrays.string_ import StringDtype

        self._inferred_dtype = self._validate(data)
        self._is_categorical = isinstance(data.dtype, CategoricalDtype)
        self._is_string = isinstance(data.dtype, StringDtype)
        self._data = data

        self._index = self._name = None
        if isinstance(data, ABCSeries):
            self._index = data.index
            self._name = data.name

        # ._values.categories works for both Series/Index
        self._parent = data._values.categories if self._is_categorical else data
        # save orig to blow up categoricals to the right type
        self._orig = data
        self._freeze()

    @staticmethod
    def _validate(data):
        """
        Auxiliary function for StringMethods, infers and checks dtype of data.

        This is a "first line of defence" at the creation of the StringMethods-
        object, and just checks that the dtype is in the
        *union* of the allowed types over all string methods below; this
        restriction is then refined on a per-method basis using the decorator
        @forbid_nonstring_types (more info in the corresponding docstring).

        This really should exclude all series/index with any non-string values,
        but that isn't practical for performance reasons until we have a str
        dtype (GH 9343 / 13877)

        Parameters
        ----------
        data : The content of the Series

        Returns
        -------
        dtype : inferred dtype of data
        """
        if isinstance(data, ABCMultiIndex):
            raise AttributeError(
                "Can only use .str accessor with Index, not MultiIndex"
            )

        # see _libs/lib.pyx for list of inferred types
        allowed_types = ["string", "empty", "bytes", "mixed", "mixed-integer"]

        data = extract_array(data)

        values = getattr(data, "categories", data)  # categorical / normal

        inferred_dtype = lib.infer_dtype(values, skipna=True)

        if inferred_dtype not in allowed_types:
            raise AttributeError("Can only use .str accessor with string values!")
        return inferred_dtype

    def __getitem__(self, key):
        result = self._data.array._str_getitem(key)
        return self._wrap_result(result)

    def __iter__(self) -> Iterator:
        raise TypeError(f"'{type(self).__name__}' object is not iterable")

    def _wrap_result(
        self,
        result,
        name=None,
        expand: bool | None = None,
        fill_value=np.nan,
        returns_string: bool = True,
        returns_bool: bool = False,
        dtype=None,
    ):
        from pandas import (
            Index,
            MultiIndex,
        )

        if not hasattr(result, "ndim") or not hasattr(result, "dtype"):
            if isinstance(result, ABCDataFrame):
                result = result.__finalize__(self._orig, name="str")
            return result
        assert result.ndim < 3

        # We can be wrapping a string / object / categorical result, in which
        # case we'll want to return the same dtype as the input.
        # Or we can be wrapping a numeric output, in which case we don't want
        # to return a StringArray.
        # Ideally the array method returns the right array type.
        if expand is None:
            # infer from ndim if expand is not specified
            expand = result.ndim != 1
        elif expand is True and not isinstance(self._orig, ABCIndex):
            # required when expand=True is explicitly specified
            # not needed when inferred
            if isinstance(result.dtype, ArrowDtype):
                import pyarrow as pa

                from pandas.compat import pa_version_under11p0

                from pandas.core.arrays.arrow.array import ArrowExtensionArray

                value_lengths = pa.compute.list_value_length(result._pa_array)
                max_len = pa.compute.max(value_lengths).as_py()
                min_len = pa.compute.min(value_lengths).as_py()
                if result._hasna:
                    # ArrowExtensionArray.fillna doesn't work for list scalars
                    result = ArrowExtensionArray(
                        result._pa_array.fill_null([None] * max_len)
                    )
                if min_len < max_len:
                    # append nulls to each scalar list element up to max_len
                    if not pa_version_under11p0:
                        result = ArrowExtensionArray(
                            pa.compute.list_slice(
                                result._pa_array,
                                start=0,
                                stop=max_len,
                                return_fixed_size_list=True,
                            )
                        )
                    else:
                        all_null = np.full(max_len, fill_value=None, dtype=object)
                        values = result.to_numpy()
                        new_values = []
                        for row in values:
                            if len(row) < max_len:
                                nulls = all_null[: max_len - len(row)]
                                row = np.append(row, nulls)
                            new_values.append(row)
                        pa_type = result._pa_array.type
                        result = ArrowExtensionArray(pa.array(new_values, type=pa_type))
                if name is not None:
                    labels = name
                else:
                    labels = range(max_len)
                result = (
                    pa.compute.list_flatten(result._pa_array)
                    .to_numpy()
                    .reshape(len(result), max_len)
                )
                result = {
                    label: ArrowExtensionArray(pa.array(res))
                    for label, res in zip(labels, result.T)
                }
            elif is_object_dtype(result):

                def cons_row(x):
                    if is_list_like(x):
                        return x
                    else:
                        return [x]

                result = [cons_row(x) for x in result]
                if result and not self._is_string:
                    # propagate nan values to match longest sequence (GH 18450)
                    max_len = max(len(x) for x in result)
                    result = [
                        x * max_len if len(x) == 0 or x[0] is np.nan else x
                        for x in result
                    ]

        if not isinstance(expand, bool):
            raise ValueError("expand must be True or False")

        if expand is False:
            # if expand is False, result should have the same name
            # as the original otherwise specified
            if name is None:
                name = getattr(result, "name", None)
            if name is None:
                # do not use logical or, _orig may be a DataFrame
                # which has "name" column
                name = self._orig.name

        # Wait until we are sure result is a Series or Index before
        # checking attributes (GH 12180)
        if isinstance(self._orig, ABCIndex):
            # if result is a boolean np.array, return the np.array
            # instead of wrapping it into a boolean Index (GH 8875)
            if is_bool_dtype(result):
                return result

            if expand:
                result = list(result)
                out: Index = MultiIndex.from_tuples(result, names=name)
                if out.nlevels == 1:
                    # We had all tuples of length-one, which are
                    # better represented as a regular Index.
                    out = out.get_level_values(0)
                return out
            else:
                return Index(result, name=name, dtype=dtype)
        else:
            index = self._orig.index
            # This is a mess.
            _dtype: DtypeObj | str | None = dtype
            vdtype = getattr(result, "dtype", None)
            if _dtype is not None:
                pass
            elif self._is_string:
                if is_bool_dtype(vdtype):
                    _dtype = result.dtype
                elif returns_string:
                    _dtype = self._orig.dtype
                else:
                    _dtype = vdtype
            elif vdtype is not None:
                _dtype = vdtype

            if expand:
                cons = self._orig._constructor_expanddim
                result = cons(result, columns=name, index=index, dtype=_dtype)
            else:
                # Must be a Series
                cons = self._orig._constructor
                result = cons(result, name=name, index=index, dtype=_dtype)
            result = result.__finalize__(self._orig, method="str")
            if name is not None and result.ndim == 1:
                # __finalize__ might copy over the original name, but we may
                # want the new name (e.g. str.extract).
                result.name = name
            return result

    def _get_series_list(self, others):
        """
        Auxiliary function for :meth:`str.cat`. Turn potentially mixed input
        into a list of Series (elements without an index must match the length
        of the calling Series/Index).

        Parameters
        ----------
        others : Series, DataFrame, np.ndarray, list-like or list-like of
            Objects that are either Series, Index or np.ndarray (1-dim).

        Returns
        -------
        list of Series
            Others transformed into list of Series.
        """
        from pandas import (
            DataFrame,
            Series,
        )

        # self._orig is either Series or Index
        idx = self._orig if isinstance(self._orig, ABCIndex) else self._orig.index

        # Generally speaking, all objects without an index inherit the index
        # `idx` of the calling Series/Index - i.e. must have matching length.
        # Objects with an index (i.e. Series/Index/DataFrame) keep their own.
        if isinstance(others, ABCSeries):
            return [others]
        elif isinstance(others, ABCIndex):
            return [Series(others, index=idx, dtype=others.dtype)]
        elif isinstance(others, ABCDataFrame):
            return [others[x] for x in others]
        elif isinstance(others, np.ndarray) and others.ndim == 2:
            others = DataFrame(others, index=idx)
            return [others[x] for x in others]
        elif is_list_like(others, allow_sets=False):
            try:
                others = list(others)  # ensure iterators do not get read twice etc
            except TypeError:
                # e.g. ser.str, raise below
                pass
            else:
                # in case of list-like `others`, all elements must be
                # either Series/Index/np.ndarray (1-dim)...
                if all(
                    isinstance(x, (ABCSeries, ABCIndex, ExtensionArray))
                    or (isinstance(x, np.ndarray) and x.ndim == 1)
                    for x in others
                ):
                    los: list[Series] = []
                    while others:  # iterate through list and append each element
                        los = los + self._get_series_list(others.pop(0))
                    return los
                # ... or just strings
                elif all(not is_list_like(x) for x in others):
                    return [Series(others, index=idx)]
        raise TypeError(
            "others must be Series, Index, DataFrame, np.ndarray "
            "or list-like (either containing only strings or "
            "containing only objects of type Series/Index/"
            "np.ndarray[1-dim])"
        )

    @forbid_nonstring_types(["bytes", "mixed", "mixed-integer"])
    def cat(
        self,
        others=None,
        sep: str | None = None,
        na_rep=None,
        join: AlignJoin = "left",
    ) -> str | Series | Index:
        """
        Concatenate strings in the Series/Index with given separator.

        If `others` is specified, this function concatenates the Series/Index
        and elements of `others` element-wise.
        If `others` is not passed, then all values in the Series/Index are
        concatenated into a single string with a given `sep`.

        Parameters
        ----------
        others : Series, Index, DataFrame, np.ndarray or list-like
            Series, Index, DataFrame, np.ndarray (one- or two-dimensional) and
            other list-likes of strings must have the same length as the
            calling Series/Index, with the exception of indexed objects (i.e.
            Series/Index/DataFrame) if `join` is not None.

            If others is a list-like that contains a combination of Series,
            Index or np.ndarray (1-dim), then all elements will be unpacked and
            must satisfy the above criteria individually.

            If others is None, the method returns the concatenation of all
            strings in the calling Series/Index.
        sep : str, default ''
            The separator between the different elements/columns. By default
            the empty string `''` is used.
        na_rep : str or None, default None
            Representation that is inserted for all missing values:

            - If `na_rep` is None, and `others` is None, missing values in the
              Series/Index are omitted from the result.
            - If `na_rep` is None, and `others` is not None, a row containing a
              missing value in any of the columns (before concatenation) will
              have a missing value in the result.
        join : {'left', 'right', 'outer', 'inner'}, default 'left'
            Determines the join-style between the calling Series/Index and any
            Series/Index/DataFrame in `others` (objects without an index need
            to match the length of the calling Series/Index). To disable
            alignment, use `.values` on any Series/Index/DataFrame in `others`.

        Returns
        -------
        str, Series or Index
            If `others` is None, `str` is returned, otherwise a `Series/Index`
            (same type as caller) of objects is returned.

        See Also
        --------
        split : Split each string in the Series/Index.
        join : Join lists contained as elements in the Series/Index.

        Examples
        --------
        When not passing `others`, all values are concatenated into a single
        string:

        >>> s = pd.Series(['a', 'b', np.nan, 'd'])
        >>> s.str.cat(sep=' ')
        'a b d'

        By default, NA values in the Series are ignored. Using `na_rep`, they
        can be given a representation:

        >>> s.str.cat(sep=' ', na_rep='?')
        'a b ? d'

        If `others` is specified, corresponding values are concatenated with
        the separator. Result will be a Series of strings.

        >>> s.str.cat(['A', 'B', 'C', 'D'], sep=',')
        0    a,A
        1    b,B
        2    NaN
        3    d,D
        dtype: object

        Missing values will remain missing in the result, but can again be
        represented using `na_rep`

        >>> s.str.cat(['A', 'B', 'C', 'D'], sep=',', na_rep='-')
        0    a,A
        1    b,B
        2    -,C
        3    d,D
        dtype: object

        If `sep` is not specified, the values are concatenated without
        separation.

        >>> s.str.cat(['A', 'B', 'C', 'D'], na_rep='-')
        0    aA
        1    bB
        2    -C
        3    dD
        dtype: object

        Series with different indexes can be aligned before concatenation. The
        `join`-keyword works as in other methods.

        >>> t = pd.Series(['d', 'a', 'e', 'c'], index=[3, 0, 4, 2])
        >>> s.str.cat(t, join='left', na_rep='-')
        0    aa
        1    b-
        2    -c
        3    dd
        dtype: object
        >>>
        >>> s.str.cat(t, join='outer', na_rep='-')
        0    aa
        1    b-
        2    -c
        3    dd
        4    -e
        dtype: object
        >>>
        >>> s.str.cat(t, join='inner', na_rep='-')
        0    aa
        2    -c
        3    dd
        dtype: object
        >>>
        >>> s.str.cat(t, join='right', na_rep='-')
        3    dd
        0    aa
        4    -e
        2    -c
        dtype: object

        For more examples, see :ref:`here <text.concatenate>`.
        """
        # TODO: dispatch
        from pandas import (
            Index,
            Series,
            concat,
        )

        if isinstance(others, str):
            raise ValueError("Did you mean to supply a `sep` keyword?")
        if sep is None:
            sep = ""

        if isinstance(self._orig, ABCIndex):
            data = Series(self._orig, index=self._orig, dtype=self._orig.dtype)
        else:  # Series
            data = self._orig

        # concatenate Series/Index with itself if no "others"
        if others is None:
            # error: Incompatible types in assignment (expression has type
            # "ndarray", variable has type "Series")
            data = ensure_object(data)  # type: ignore[assignment]
            na_mask = isna(data)
            if na_rep is None and na_mask.any():
                return sep.join(data[~na_mask])
            elif na_rep is not None and na_mask.any():
                return sep.join(np.where(na_mask, na_rep, data))
            else:
                return sep.join(data)

        try:
            # turn anything in "others" into lists of Series
            others = self._get_series_list(others)
        except ValueError as err:  # do not catch TypeError raised by _get_series_list
            raise ValueError(
                "If `others` contains arrays or lists (or other "
                "list-likes without an index), these must all be "
                "of the same length as the calling Series/Index."
            ) from err

        # align if required
        if any(not data.index.equals(x.index) for x in others):
            # Need to add keys for uniqueness in case of duplicate columns
            others = concat(
                others,
                axis=1,
                join=(join if join == "inner" else "outer"),
                keys=range(len(others)),
                sort=False,
                copy=False,
            )
            data, others = data.align(others, join=join)
            others = [others[x] for x in others]  # again list of Series

        all_cols = [ensure_object(x) for x in [data] + others]
        na_masks = np.array([isna(x) for x in all_cols])
        union_mask = np.logical_or.reduce(na_masks, axis=0)

        if na_rep is None and union_mask.any():
            # no na_rep means NaNs for all rows where any column has a NaN
            # only necessary if there are actually any NaNs
            result = np.empty(len(data), dtype=object)
            np.putmask(result, union_mask, np.nan)

            not_masked = ~union_mask
            result[not_masked] = cat_safe([x[not_masked] for x in all_cols], sep)
        elif na_rep is not None and union_mask.any():
            # fill NaNs with na_rep in case there are actually any NaNs
            all_cols = [
                np.where(nm, na_rep, col) for nm, col in zip(na_masks, all_cols)
            ]
            result = cat_safe(all_cols, sep)
        else:
            # no NaNs - can just concatenate
            result = cat_safe(all_cols, sep)

        out: Index | Series
        if isinstance(self._orig.dtype, CategoricalDtype):
            # We need to infer the new categories.
            dtype = self._orig.dtype.categories.dtype
        else:
            dtype = self._orig.dtype
        if isinstance(self._orig, ABCIndex):
            # add dtype for case that result is all-NA
            if isna(result).all():
                dtype = object  # type: ignore[assignment]

            out = Index(result, dtype=dtype, name=self._orig.name)
        else:  # Series
            res_ser = Series(
                result, dtype=dtype, index=data.index, name=self._orig.name, copy=False
            )
            out = res_ser.__finalize__(self._orig, method="str_cat")
        return out

    _shared_docs[
        "str_split"
    ] = r"""
    Split strings around given separator/delimiter.

    Splits the string in the Series/Index from the %(side)s,
    at the specified delimiter string.

    Parameters
    ----------
    pat : str%(pat_regex)s, optional
        %(pat_description)s.
        If not specified, split on whitespace.
    n : int, default -1 (all)
        Limit number of splits in output.
        ``None``, 0 and -1 will be interpreted as return all splits.
    expand : bool, default False
        Expand the split strings into separate columns.

        - If ``True``, return DataFrame/MultiIndex expanding dimensionality.
        - If ``False``, return Series/Index, containing lists of strings.
    %(regex_argument)s
    Returns
    -------
    Series, Index, DataFrame or MultiIndex
        Type matches caller unless ``expand=True`` (see Notes).
    %(raises_split)s
    See Also
    --------
    Series.str.split : Split strings around given separator/delimiter.
    Series.str.rsplit : Splits string around given separator/delimiter,
        starting from the right.
    Series.str.join : Join lists contained as elements in the Series/Index
        with passed delimiter.
    str.split : Standard library version for split.
    str.rsplit : Standard library version for rsplit.

    Notes
    -----
    The handling of the `n` keyword depends on the number of found splits:

    - If found splits > `n`,  make first `n` splits only
    - If found splits <= `n`, make all splits
    - If for a certain row the number of found splits < `n`,
      append `None` for padding up to `n` if ``expand=True``

    If using ``expand=True``, Series and Index callers return DataFrame and
    MultiIndex objects, respectively.
    %(regex_pat_note)s
    Examples
    --------
    >>> s = pd.Series(
    ...     [
    ...         "this is a regular sentence",
    ...         "https://docs.python.org/3/tutorial/index.html",
    ...         np.nan
    ...     ]
    ... )
    >>> s
    0                       this is a regular sentence
    1    https://docs.python.org/3/tutorial/index.html
    2                                              NaN
    dtype: object

    In the default setting, the string is split by whitespace.

    >>> s.str.split()
    0                   [this, is, a, regular, sentence]
    1    [https://docs.python.org/3/tutorial/index.html]
    2                                                NaN
    dtype: object

    Without the `n` parameter, the outputs of `rsplit` and `split`
    are identical.

    >>> s.str.rsplit()
    0                   [this, is, a, regular, sentence]
    1    [https://docs.python.org/3/tutorial/index.html]
    2                                                NaN
    dtype: object

    The `n` parameter can be used to limit the number of splits on the
    delimiter. The outputs of `split` and `rsplit` are different.

    >>> s.str.split(n=2)
    0                     [this, is, a regular sentence]
    1    [https://docs.python.org/3/tutorial/index.html]
    2                                                NaN
    dtype: object

    >>> s.str.rsplit(n=2)
    0                     [this is a, regular, sentence]
    1    [https://docs.python.org/3/tutorial/index.html]
    2                                                NaN
    dtype: object

    The `pat` parameter can be used to split by other characters.

    >>> s.str.split(pat="/")
    0                         [this is a regular sentence]
    1    [https:, , docs.python.org, 3, tutorial, index...
    2                                                  NaN
    dtype: object

    When using ``expand=True``, the split elements will expand out into
    separate columns. If NaN is present, it is propagated throughout
    the columns during the split.

    >>> s.str.split(expand=True)
                                                   0     1     2        3         4
    0                                           this    is     a  regular  sentence
    1  https://docs.python.org/3/tutorial/index.html  None  None     None      None
    2                                            NaN   NaN   NaN      NaN       NaN

    For slightly more complex use cases like splitting the html document name
    from a url, a combination of parameter settings can be used.

    >>> s.str.rsplit("/", n=1, expand=True)
                                        0           1
    0          this is a regular sentence        None
    1  https://docs.python.org/3/tutorial  index.html
    2                                 NaN         NaN
    %(regex_examples)s"""

    @Appender(
        _shared_docs["str_split"]
        % {
            "side": "beginning",
            "pat_regex": " or compiled regex",
            "pat_description": "String or regular expression to split on",
            "regex_argument": """
    regex : bool, default None
        Determines if the passed-in pattern is a regular expression:

        - If ``True``, assumes the passed-in pattern is a regular expression
        - If ``False``, treats the pattern as a literal string.
        - If ``None`` and `pat` length is 1, treats `pat` as a literal string.
        - If ``None`` and `pat` length is not 1, treats `pat` as a regular expression.
        - Cannot be set to False if `pat` is a compiled regex

        .. versionadded:: 1.4.0
         """,
            "raises_split": """
                      Raises
                      ------
                      ValueError
                          * if `regex` is False and `pat` is a compiled regex
                      """,
            "regex_pat_note": """
    Use of `regex =False` with a `pat` as a compiled regex will raise an error.
            """,
            "method": "split",
            "regex_examples": r"""
    Remember to escape special characters when explicitly using regular expressions.

    >>> s = pd.Series(["foo and bar plus baz"])
    >>> s.str.split(r"and|plus", expand=True)
        0   1   2
    0 foo bar baz

    Regular expressions can be used to handle urls or file names.
    When `pat` is a string and ``regex=None`` (the default), the given `pat` is compiled
    as a regex only if ``len(pat) != 1``.

    >>> s = pd.Series(['foojpgbar.jpg'])
    >>> s.str.split(r".", expand=True)
               0    1
    0  foojpgbar  jpg

    >>> s.str.split(r"\.jpg", expand=True)
               0 1
    0  foojpgbar

    When ``regex=True``, `pat` is interpreted as a regex

    >>> s.str.split(r"\.jpg", regex=True, expand=True)
               0 1
    0  foojpgbar

    A compiled regex can be passed as `pat`

    >>> import re
    >>> s.str.split(re.compile(r"\.jpg"), expand=True)
               0 1
    0  foojpgbar

    When ``regex=False``, `pat` is interpreted as the string itself

    >>> s.str.split(r"\.jpg", regex=False, expand=True)
                   0
    0  foojpgbar.jpg
    """,
        }
    )
    @forbid_nonstring_types(["bytes"])
    def split(
        self,
        pat: str | re.Pattern | None = None,
        *,
        n=-1,
        expand: bool = False,
        regex: bool | None = None,
    ):
        if regex is False and is_re(pat):
            raise ValueError(
                "Cannot use a compiled regex as replacement pattern with regex=False"
            )
        if is_re(pat):
            regex = True
        result = self._data.array._str_split(pat, n, expand, regex)
        if self._data.dtype == "category":
            dtype = self._data.dtype.categories.dtype
        else:
            dtype = object if self._data.dtype == object else None
        return self._wrap_result(
            result, expand=expand, returns_string=expand, dtype=dtype
        )

    @Appender(
        _shared_docs["str_split"]
        % {
            "side": "end",
            "pat_regex": "",
            "pat_description": "String to split on",
            "regex_argument": "",
            "raises_split": "",
            "regex_pat_note": "",
            "method": "rsplit",
            "regex_examples": "",
        }
    )
    @forbid_nonstring_types(["bytes"])
    def rsplit(self, pat=None, *, n=-1, expand: bool = False):
        result = self._data.array._str_rsplit(pat, n=n)
        dtype = object if self._data.dtype == object else None
        return self._wrap_result(
            result, expand=expand, returns_string=expand, dtype=dtype
        )

    _shared_docs[
        "str_partition"
    ] = """
    Split the string at the %(side)s occurrence of `sep`.

    This method splits the string at the %(side)s occurrence of `sep`,
    and returns 3 elements containing the part before the separator,
    the separator itself, and the part after the separator.
    If the separator is not found, return %(return)s.

    Parameters
    ----------
    sep : str, default whitespace
        String to split on.
    expand : bool, default True
        If True, return DataFrame/MultiIndex expanding dimensionality.
        If False, return Series/Index.

    Returns
    -------
    DataFrame/MultiIndex or Series/Index of objects

    See Also
    --------
    %(also)s
    Series.str.split : Split strings around given separators.
    str.partition : Standard library version.

    Examples
    --------

    >>> s = pd.Series(['Linda van der Berg', 'George Pitt-Rivers'])
    >>> s
    0    Linda van der Berg
    1    George Pitt-Rivers
    dtype: object

    >>> s.str.partition()
            0  1             2
    0   Linda     van der Berg
    1  George      Pitt-Rivers

    To partition by the last space instead of the first one:

    >>> s.str.rpartition()
                   0  1            2
    0  Linda van der            Berg
    1         George     Pitt-Rivers

    To partition by something different than a space:

    >>> s.str.partition('-')
                        0  1       2
    0  Linda van der Berg
    1         George Pitt  -  Rivers

    To return a Series containing tuples instead of a DataFrame:

    >>> s.str.partition('-', expand=False)
    0    (Linda van der Berg, , )
    1    (George Pitt, -, Rivers)
    dtype: object

    Also available on indices:

    >>> idx = pd.Index(['X 123', 'Y 999'])
    >>> idx
    Index(['X 123', 'Y 999'], dtype='object')

    Which will create a MultiIndex:

    >>> idx.str.partition()
    MultiIndex([('X', ' ', '123'),
                ('Y', ' ', '999')],
               )

    Or an index with tuples with ``expand=False``:

    >>> idx.str.partition(expand=False)
    Index([('X', ' ', '123'), ('Y', ' ', '999')], dtype='object')
    """

    @Appender(
        _shared_docs["str_partition"]
        % {
            "side": "first",
            "return": "3 elements containing the string itself, followed by two "
            "empty strings",
            "also": "rpartition : Split the string at the last occurrence of `sep`.",
        }
    )
    @forbid_nonstring_types(["bytes"])
    def partition(self, sep: str = " ", expand: bool = True):
        result = self._data.array._str_partition(sep, expand)
        if self._data.dtype == "category":
            dtype = self._data.dtype.categories.dtype
        else:
            dtype = object if self._data.dtype == object else None
        return self._wrap_result(
            result, expand=expand, returns_string=expand, dtype=dtype
        )

    @Appender(
        _shared_docs["str_partition"]
        % {
            "side": "last",
            "return": "3 elements containing two empty strings, followed by the "
            "string itself",
            "also": "partition : Split the string at the first occurrence of `sep`.",
        }
    )
    @forbid_nonstring_types(["bytes"])
    def rpartition(self, sep: str = " ", expand: bool = True):
        result = self._data.array._str_rpartition(sep, expand)
        if self._data.dtype == "category":
            dtype = self._data.dtype.categories.dtype
        else:
            dtype = object if self._data.dtype == object else None
        return self._wrap_result(
            result, expand=expand, returns_string=expand, dtype=dtype
        )

    def get(self, i):
        """
        Extract element from each component at specified position or with specified key.

        Extract element from lists, tuples, dict, or strings in each element in the
        Series/Index.

        Parameters
        ----------
        i : int or hashable dict label
            Position or key of element to extract.

        Returns
        -------
        Series or Index

        Examples
        --------
        >>> s = pd.Series(["String",
        ...               (1, 2, 3),
        ...               ["a", "b", "c"],
        ...               123,
        ...               -456,
        ...               {1: "Hello", "2": "World"}])
        >>> s
        0                        String
        1                     (1, 2, 3)
        2                     [a, b, c]
        3                           123
        4                          -456
        5    {1: 'Hello', '2': 'World'}
        dtype: object

        >>> s.str.get(1)
        0        t
        1        2
        2        b
        3      NaN
        4      NaN
        5    Hello
        dtype: object

        >>> s.str.get(-1)
        0      g
        1      3
        2      c
        3    NaN
        4    NaN
        5    None
        dtype: object

        Return element with given key

        >>> s = pd.Series([{"name": "Hello", "value": "World"},
        ...               {"name": "Goodbye", "value": "Planet"}])
        >>> s.str.get('name')
        0      Hello
        1    Goodbye
        dtype: object
        """
        result = self._data.array._str_get(i)
        return self._wrap_result(result)

    @forbid_nonstring_types(["bytes"])
    def join(self, sep: str):
        """
        Join lists contained as elements in the Series/Index with passed delimiter.

        If the elements of a Series are lists themselves, join the content of these
        lists using the delimiter passed to the function.
        This function is an equivalent to :meth:`str.join`.

        Parameters
        ----------
        sep : str
            Delimiter to use between list entries.

        Returns
        -------
        Series/Index: object
            The list entries concatenated by intervening occurrences of the
            delimiter.

        Raises
        ------
        AttributeError
            If the supplied Series contains neither strings nor lists.

        See Also
        --------
        str.join : Standard library version of this method.
        Series.str.split : Split strings around given separator/delimiter.

        Notes
        -----
        If any of the list items is not a string object, the result of the join
        will be `NaN`.

        Examples
        --------
        Example with a list that contains non-string elements.

        >>> s = pd.Series([['lion', 'elephant', 'zebra'],
        ...                [1.1, 2.2, 3.3],
        ...                ['cat', np.nan, 'dog'],
        ...                ['cow', 4.5, 'goat'],
        ...                ['duck', ['swan', 'fish'], 'guppy']])
        >>> s
        0        [lion, elephant, zebra]
        1                [1.1, 2.2, 3.3]
        2                [cat, nan, dog]
        3               [cow, 4.5, goat]
        4    [duck, [swan, fish], guppy]
        dtype: object

        Join all lists using a '-'. The lists containing object(s) of types other
        than str will produce a NaN.

        >>> s.str.join('-')
        0    lion-elephant-zebra
        1                    NaN
        2                    NaN
        3                    NaN
        4                    NaN
        dtype: object
        """
        result = self._data.array._str_join(sep)
        return self._wrap_result(result)

    @forbid_nonstring_types(["bytes"])
    def contains(
        self,
        pat,
        case: bool = True,
        flags: int = 0,
        na=lib.no_default,
        regex: bool = True,
    ):
        r"""
        Test if pattern or regex is contained within a string of a Series or Index.

        Return boolean Series or Index based on whether a given pattern or regex is
        contained within a string of a Series or Index.

        Parameters
        ----------
        pat : str
            Character sequence or regular expression.
        case : bool, default True
            If True, case sensitive.
        flags : int, default 0 (no flags)
            Flags to pass through to the re module, e.g. re.IGNORECASE.
        na : scalar, optional
            Fill value for missing values. The default depends on dtype of the
            array. For object-dtype, ``numpy.nan`` is used. For the nullable
            ``StringDtype``, ``pandas.NA`` is used. For the ``"str"`` dtype,
            ``False`` is used.
        regex : bool, default True
            If True, assumes the pat is a regular expression.

            If False, treats the pat as a literal string.

        Returns
        -------
        Series or Index of boolean values
            A Series or Index of boolean values indicating whether the
            given pattern is contained within the string of each element
            of the Series or Index.

        See Also
        --------
        match : Analogous, but stricter, relying on re.match instead of re.search.
        Series.str.startswith : Test if the start of each string element matches a
            pattern.
        Series.str.endswith : Same as startswith, but tests the end of string.

        Examples
        --------
        Returning a Series of booleans using only a literal pattern.

        >>> s1 = pd.Series(['Mouse', 'dog', 'house and parrot', '23', np.nan])
        >>> s1.str.contains('og', regex=False)
        0    False
        1     True
        2    False
        3    False
        4      NaN
        dtype: object

        Returning an Index of booleans using only a literal pattern.

        >>> ind = pd.Index(['Mouse', 'dog', 'house and parrot', '23.0', np.nan])
        >>> ind.str.contains('23', regex=False)
        Index([False, False, False, True, nan], dtype='object')

        Specifying case sensitivity using `case`.

        >>> s1.str.contains('oG', case=True, regex=True)
        0    False
        1    False
        2    False
        3    False
        4      NaN
        dtype: object

        Specifying `na` to be `False` instead of `NaN` replaces NaN values
        with `False`. If Series or Index does not contain NaN values
        the resultant dtype will be `bool`, otherwise, an `object` dtype.

        >>> s1.str.contains('og', na=False, regex=True)
        0    False
        1     True
        2    False
        3    False
        4    False
        dtype: bool

        Returning 'house' or 'dog' when either expression occurs in a string.

        >>> s1.str.contains('house|dog', regex=True)
        0    False
        1     True
        2     True
        3    False
        4      NaN
        dtype: object

        Ignoring case sensitivity using `flags` with regex.

        >>> import re
        >>> s1.str.contains('PARROT', flags=re.IGNORECASE, regex=True)
        0    False
        1    False
        2     True
        3    False
        4      NaN
        dtype: object

        Returning any digit using regular expression.

        >>> s1.str.contains('\\d', regex=True)
        0    False
        1    False
        2    False
        3     True
        4      NaN
        dtype: object

        Ensure `pat` is a not a literal pattern when `regex` is set to True.
        Note in the following example one might expect only `s2[1]` and `s2[3]` to
        return `True`. However, '.0' as a regex matches any character
        followed by a 0.

        >>> s2 = pd.Series(['40', '40.0', '41', '41.0', '35'])
        >>> s2.str.contains('.0', regex=True)
        0     True
        1     True
        2    False
        3     True
        4    False
        dtype: bool
        """
        if regex and re.compile(pat).groups:
            warnings.warn(
                "This pattern is interpreted as a regular expression, and has "
                "match groups. To actually get the groups, use str.extract.",
                UserWarning,
                stacklevel=find_stack_level(),
            )

        result = self._data.array._str_contains(pat, case, flags, na, regex)
        return self._wrap_result(result, fill_value=na, returns_string=False)

    @forbid_nonstring_types(["bytes"])
    def match(self, pat: str, case: bool = True, flags: int = 0, na=lib.no_default):
        """
        Determine if each string starts with a match of a regular expression.

        Parameters
        ----------
        pat : str or compiled regex
            Character sequence or regular expression.
        case : bool, default True
            If True, case sensitive.
        flags : int, default 0 (no flags)
            Regex module flags, e.g. re.IGNORECASE.
        na : scalar, optional
            Fill value for missing values. The default depends on dtype of the
            array. For object-dtype, ``numpy.nan`` is used. For the nullable
            ``StringDtype``, ``pandas.NA`` is used. For the ``"str"`` dtype,
            ``False`` is used.

        Returns
        -------
        Series/Index/array of boolean values

        See Also
        --------
        fullmatch : Stricter matching that requires the entire string to match.
        contains : Analogous, but less strict, relying on re.search instead of
            re.match.
        extract : Extract matched groups.

        Examples
        --------
        >>> ser = pd.Series(["horse", "eagle", "donkey"])
        >>> ser.str.match("e")
        0   False
        1   True
        2   False
        dtype: bool
        """
        result = self._data.array._str_match(pat, case=case, flags=flags, na=na)
        return self._wrap_result(result, fill_value=na, returns_string=False)

    @forbid_nonstring_types(["bytes"])
    def fullmatch(self, pat, case: bool = True, flags: int = 0, na=lib.no_default):
        """
        Determine if each string entirely matches a regular expression.

        Parameters
        ----------
        pat : str
            Character sequence or regular expression.
        case : bool, default True
            If True, case sensitive.
        flags : int, default 0 (no flags)
            Regex module flags, e.g. re.IGNORECASE.
        na : scalar, optional
            Fill value for missing values. The default depends on dtype of the
            array. For object-dtype, ``numpy.nan`` is used. For the nullable
            ``StringDtype``, ``pandas.NA`` is used. For the ``"str"`` dtype,
            ``False`` is used.

        Returns
        -------
        Series/Index/array of boolean values

        See Also
        --------
        match : Similar, but also returns `True` when only a *prefix* of the string
            matches the regular expression.
        extract : Extract matched groups.

        Examples
        --------
        >>> ser = pd.Series(["cat", "duck", "dove"])
        >>> ser.str.fullmatch(r'd.+')
        0   False
        1    True
        2    True
        dtype: bool
        """
        result = self._data.array._str_fullmatch(pat, case=case, flags=flags, na=na)
        return self._wrap_result(result, fill_value=na, returns_string=False)

    @forbid_nonstring_types(["bytes"])
    def replace(
        self,
        pat: str | re.Pattern,
        repl: str | Callable,
        n: int = -1,
        case: bool | None = None,
        flags: int = 0,
        regex: bool = False,
    ):
        r"""
        Replace each occurrence of pattern/regex in the Series/Index.

        Equivalent to :meth:`str.replace` or :func:`re.sub`, depending on
        the regex value.

        Parameters
        ----------
        pat : str or compiled regex
            String can be a character sequence or regular expression.
        repl : str or callable
            Replacement string or a callable. The callable is passed the regex
            match object and must return a replacement string to be used.
            See :func:`re.sub`.
        n : int, default -1 (all)
            Number of replacements to make from start.
        case : bool, default None
            Determines if replace is case sensitive:

            - If True, case sensitive (the default if `pat` is a string)
            - Set to False for case insensitive
            - Cannot be set if `pat` is a compiled regex.

        flags : int, default 0 (no flags)
            Regex module flags, e.g. re.IGNORECASE. Cannot be set if `pat` is a compiled
            regex.
        regex : bool, default False
            Determines if the passed-in pattern is a regular expression:

            - If True, assumes the passed-in pattern is a regular expression.
            - If False, treats the pattern as a literal string
            - Cannot be set to False if `pat` is a compiled regex or `repl` is
              a callable.

        Returns
        -------
        Series or Index of object
            A copy of the object with all matching occurrences of `pat` replaced by
            `repl`.

        Raises
        ------
        ValueError
            * if `regex` is False and `repl` is a callable or `pat` is a compiled
              regex
            * if `pat` is a compiled regex and `case` or `flags` is set

        Notes
        -----
        When `pat` is a compiled regex, all flags should be included in the
        compiled regex. Use of `case`, `flags`, or `regex=False` with a compiled
        regex will raise an error.

        Examples
        --------
        When `pat` is a string and `regex` is True, the given `pat`
        is compiled as a regex. When `repl` is a string, it replaces matching
        regex patterns as with :meth:`re.sub`. NaN value(s) in the Series are
        left as is:

        >>> pd.Series(['foo', 'fuz', np.nan]).str.replace('f.', 'ba', regex=True)
        0    bao
        1    baz
        2    NaN
        dtype: object

        When `pat` is a string and `regex` is False, every `pat` is replaced with
        `repl` as with :meth:`str.replace`:

        >>> pd.Series(['f.o', 'fuz', np.nan]).str.replace('f.', 'ba', regex=False)
        0    bao
        1    fuz
        2    NaN
        dtype: object

        When `repl` is a callable, it is called on every `pat` using
        :func:`re.sub`. The callable should expect one positional argument
        (a regex object) and return a string.

        To get the idea:

        >>> pd.Series(['foo', 'fuz', np.nan]).str.replace('f', repr, regex=True)
        0    <re.Match object; span=(0, 1), match='f'>oo
        1    <re.Match object; span=(0, 1), match='f'>uz
        2                                            NaN
        dtype: object

        Reverse every lowercase alphabetic word:

        >>> repl = lambda m: m.group(0)[::-1]
        >>> ser = pd.Series(['foo 123', 'bar baz', np.nan])
        >>> ser.str.replace(r'[a-z]+', repl, regex=True)
        0    oof 123
        1    rab zab
        2        NaN
        dtype: object

        Using regex groups (extract second group and swap case):

        >>> pat = r"(?P<one>\w+) (?P<two>\w+) (?P<three>\w+)"
        >>> repl = lambda m: m.group('two').swapcase()
        >>> ser = pd.Series(['One Two Three', 'Foo Bar Baz'])
        >>> ser.str.replace(pat, repl, regex=True)
        0    tWO
        1    bAR
        dtype: object

        Using a compiled regex with flags

        >>> import re
        >>> regex_pat = re.compile(r'FUZ', flags=re.IGNORECASE)
        >>> pd.Series(['foo', 'fuz', np.nan]).str.replace(regex_pat, 'bar', regex=True)
        0    foo
        1    bar
        2    NaN
        dtype: object
        """
        # Check whether repl is valid (GH 13438, GH 15055)
        if not (isinstance(repl, str) or callable(repl)):
            raise TypeError("repl must be a string or callable")

        is_compiled_re = is_re(pat)
        if regex or regex is None:
            if is_compiled_re and (case is not None or flags != 0):
                raise ValueError(
                    "case and flags cannot be set when pat is a compiled regex"
                )

        elif is_compiled_re:
            raise ValueError(
                "Cannot use a compiled regex as replacement pattern with regex=False"
            )
        elif callable(repl):
            raise ValueError("Cannot use a callable replacement when regex=False")

        if case is None:
            case = True

        result = self._data.array._str_replace(
            pat, repl, n=n, case=case, flags=flags, regex=regex
        )
        return self._wrap_result(result)

    @forbid_nonstring_types(["bytes"])
    def repeat(self, repeats):
        """
        Duplicate each string in the Series or Index.

        Parameters
        ----------
        repeats : int or sequence of int
            Same value for all (int) or different value per (sequence).

        Returns
        -------
        Series or pandas.Index
            Series or Index of repeated string objects specified by
            input parameter repeats.

        Examples
        --------
        >>> s = pd.Series(['a', 'b', 'c'])
        >>> s
        0    a
        1    b
        2    c
        dtype: object

        Single int repeats string in Series

        >>> s.str.repeat(repeats=2)
        0    aa
        1    bb
        2    cc
        dtype: object

        Sequence of int repeats corresponding string in Series

        >>> s.str.repeat(repeats=[1, 2, 3])
        0      a
        1     bb
        2    ccc
        dtype: object
        """
        result = self._data.array._str_repeat(repeats)
        return self._wrap_result(result)

    @forbid_nonstring_types(["bytes"])
    def pad(
        self,
        width: int,
        side: Literal["left", "right", "both"] = "left",
        fillchar: str = " ",
    ):
        """
        Pad strings in the Series/Index up to width.

        Parameters
        ----------
        width : int
            Minimum width of resulting string; additional characters will be filled
            with character defined in `fillchar`.
        side : {'left', 'right', 'both'}, default 'left'
            Side from which to fill resulting string.
        fillchar : str, default ' '
            Additional character for filling, default is whitespace.

        Returns
        -------
        Series or Index of object
            Returns Series or Index with minimum number of char in object.

        See Also
        --------
        Series.str.rjust : Fills the left side of strings with an arbitrary
            character. Equivalent to ``Series.str.pad(side='left')``.
        Series.str.ljust : Fills the right side of strings with an arbitrary
            character. Equivalent to ``Series.str.pad(side='right')``.
        Series.str.center : Fills both sides of strings with an arbitrary
            character. Equivalent to ``Series.str.pad(side='both')``.
        Series.str.zfill : Pad strings in the Series/Index by prepending '0'
            character. Equivalent to ``Series.str.pad(side='left', fillchar='0')``.

        Examples
        --------
        >>> s = pd.Series(["caribou", "tiger"])
        >>> s
        0    caribou
        1      tiger
        dtype: object

        >>> s.str.pad(width=10)
        0       caribou
        1         tiger
        dtype: object

        >>> s.str.pad(width=10, side='right', fillchar='-')
        0    caribou---
        1    tiger-----
        dtype: object

        >>> s.str.pad(width=10, side='both', fillchar='-')
        0    -caribou--
        1    --tiger---
        dtype: object
        """
        if not isinstance(fillchar, str):
            msg = f"fillchar must be a character, not {type(fillchar).__name__}"
            raise TypeError(msg)

        if len(fillchar) != 1:
            raise TypeError("fillchar must be a character, not str")

        if not is_integer(width):
            msg = f"width must be of integer type, not {type(width).__name__}"
            raise TypeError(msg)

        result = self._data.array._str_pad(width, side=side, fillchar=fillchar)
        return self._wrap_result(result)

    _shared_docs[
        "str_pad"
    ] = """
    Pad %(side)s side of strings in the Series/Index.

    Equivalent to :meth:`str.%(method)s`.

    Parameters
    ----------
    width : int
        Minimum width of resulting string; additional characters will be filled
        with ``fillchar``.
    fillchar : str
        Additional character for filling, default is whitespace.

    Returns
    -------
    Series/Index of objects.

    Examples
    --------
    For Series.str.center:

    >>> ser = pd.Series(['dog', 'bird', 'mouse'])
    >>> ser.str.center(8, fillchar='.')
    0   ..dog...
    1   ..bird..
    2   .mouse..
    dtype: object

    For Series.str.ljust:

    >>> ser = pd.Series(['dog', 'bird', 'mouse'])
    >>> ser.str.ljust(8, fillchar='.')
    0   dog.....
    1   bird....
    2   mouse...
    dtype: object

    For Series.str.rjust:

    >>> ser = pd.Series(['dog', 'bird', 'mouse'])
    >>> ser.str.rjust(8, fillchar='.')
    0   .....dog
    1   ....bird
    2   ...mouse
    dtype: object
    """

    @Appender(_shared_docs["str_pad"] % {"side": "left and right", "method": "center"})
    @forbid_nonstring_types(["bytes"])
    def center(self, width: int, fillchar: str = " "):
        return self.pad(width, side="both", fillchar=fillchar)

    @Appender(_shared_docs["str_pad"] % {"side": "right", "method": "ljust"})
    @forbid_nonstring_types(["bytes"])
    def ljust(self, width: int, fillchar: str = " "):
        return self.pad(width, side="right", fillchar=fillchar)

    @Appender(_shared_docs["str_pad"] % {"side": "left", "method": "rjust"})
    @forbid_nonstring_types(["bytes"])
    def rjust(self, width: int, fillchar: str = " "):
        return self.pad(width, side="left", fillchar=fillchar)

    @forbid_nonstring_types(["bytes"])
    def zfill(self, width: int):
        """
        Pad strings in the Series/Index by prepending '0' characters.

        Strings in the Series/Index are padded with '0' characters on the
        left of the string to reach a total string length  `width`. Strings
        in the Series/Index with length greater or equal to `width` are
        unchanged.

        Parameters
        ----------
        width : int
            Minimum length of resulting string; strings with length less
            than `width` be prepended with '0' characters.

        Returns
        -------
        Series/Index of objects.

        See Also
        --------
        Series.str.rjust : Fills the left side of strings with an arbitrary
            character.
        Series.str.ljust : Fills the right side of strings with an arbitrary
            character.
        Series.str.pad : Fills the specified sides of strings with an arbitrary
            character.
        Series.str.center : Fills both sides of strings with an arbitrary
            character.

        Notes
        -----
        Differs from :meth:`str.zfill` which has special handling
        for '+'/'-' in the string.

        Examples
        --------
        >>> s = pd.Series(['-1', '1', '1000', 10, np.nan])
        >>> s
        0      -1
        1       1
        2    1000
        3      10
        4     NaN
        dtype: object

        Note that ``10`` and ``NaN`` are not strings, therefore they are
        converted to ``NaN``. The minus sign in ``'-1'`` is treated as a
        special character and the zero is added to the right of it
        (:meth:`str.zfill` would have moved it to the left). ``1000``
        remains unchanged as it is longer than `width`.

        >>> s.str.zfill(3)
        0     -01
        1     001
        2    1000
        3     NaN
        4     NaN
        dtype: object
        """
        if not is_integer(width):
            msg = f"width must be of integer type, not {type(width).__name__}"
            raise TypeError(msg)
        f = lambda x: x.zfill(width)
        result = self._data.array._str_map(f)
        return self._wrap_result(result)

    def slice(self, start=None, stop=None, step=None):
        """
        Slice substrings from each element in the Series or Index.

        Parameters
        ----------
        start : int, optional
            Start position for slice operation.
        stop : int, optional
            Stop position for slice operation.
        step : int, optional
            Step size for slice operation.

        Returns
        -------
        Series or Index of object
            Series or Index from sliced substring from original string object.

        See Also
        --------
        Series.str.slice_replace : Replace a slice with a string.
        Series.str.get : Return element at position.
            Equivalent to `Series.str.slice(start=i, stop=i+1)` with `i`
            being the position.

        Examples
        --------
        >>> s = pd.Series(["koala", "dog", "chameleon"])
        >>> s
        0        koala
        1          dog
        2    chameleon
        dtype: object

        >>> s.str.slice(start=1)
        0        oala
        1          og
        2    hameleon
        dtype: object

        >>> s.str.slice(start=-1)
        0           a
        1           g
        2           n
        dtype: object

        >>> s.str.slice(stop=2)
        0    ko
        1    do
        2    ch
        dtype: object

        >>> s.str.slice(step=2)
        0      kaa
        1       dg
        2    caeen
        dtype: object

        >>> s.str.slice(start=0, stop=5, step=3)
        0    kl
        1     d
        2    cm
        dtype: object

        Equivalent behaviour to:

        >>> s.str[0:5:3]
        0    kl
        1     d
        2    cm
        dtype: object
        """
        result = self._data.array._str_slice(start, stop, step)
        return self._wrap_result(result)

    @forbid_nonstring_types(["bytes"])
    def slice_replace(self, start=None, stop=None, repl=None):
        """
        Replace a positional slice of a string with another value.

        Parameters
        ----------
        start : int, optional
            Left index position to use for the slice. If not specified (None),
            the slice is unbounded on the left, i.e. slice from the start
            of the string.
        stop : int, optional
            Right index position to use for the slice. If not specified (None),
            the slice is unbounded on the right, i.e. slice until the
            end of the string.
        repl : str, optional
            String for replacement. If not specified (None), the sliced region
            is replaced with an empty string.

        Returns
        -------
        Series or Index
            Same type as the original object.

        See Also
        --------
        Series.str.slice : Just slicing without replacement.

        Examples
        --------
        >>> s = pd.Series(['a', 'ab', 'abc', 'abdc', 'abcde'])
        >>> s
        0        a
        1       ab
        2      abc
        3     abdc
        4    abcde
        dtype: object

        Specify just `start`, meaning replace `start` until the end of the
        string with `repl`.

        >>> s.str.slice_replace(1, repl='X')
        0    aX
        1    aX
        2    aX
        3    aX
        4    aX
        dtype: object

        Specify just `stop`, meaning the start of the string to `stop` is replaced
        with `repl`, and the rest of the string is included.

        >>> s.str.slice_replace(stop=2, repl='X')
        0       X
        1       X
        2      Xc
        3     Xdc
        4    Xcde
        dtype: object

        Specify `start` and `stop`, meaning the slice from `start` to `stop` is
        replaced with `repl`. Everything before or after `start` and `stop` is
        included as is.

        >>> s.str.slice_replace(start=1, stop=3, repl='X')
        0      aX
        1      aX
        2      aX
        3     aXc
        4    aXde
        dtype: object
        """
        result = self._data.array._str_slice_replace(start, stop, repl)
        return self._wrap_result(result)

    def decode(
        self, encoding, errors: str = "strict", dtype: str | DtypeObj | None = None
    ):
        """
        Decode character string in the Series/Index using indicated encoding.

        Equivalent to :meth:`str.decode` in python2 and :meth:`bytes.decode` in
        python3.

        Parameters
        ----------
        encoding : str
        errors : str, optional
            Specifies the error handling scheme.
            Possible values are those supported by :meth:`bytes.decode`.
        dtype : str or dtype, optional
            The dtype of the result. When not ``None``, must be either a string or
            object dtype. When ``None``, the dtype of the result is determined by
            ``pd.options.future.infer_string``.

            .. versionadded:: 2.3.0

        Returns
        -------
        Series or Index

        Examples
        --------
        For Series:

        >>> ser = pd.Series([b'cow', b'123', b'()'])
        >>> ser.str.decode('ascii')
        0   cow
        1   123
        2   ()
        dtype: object
        """
        if dtype is not None and not is_string_dtype(dtype):
            raise ValueError(f"dtype must be string or object, got {dtype=}")
        if dtype is None and get_option("future.infer_string"):
            dtype = "str"
        # TODO: Add a similar _bytes interface.
        if encoding in _cpython_optimized_decoders:
            # CPython optimized implementation
            f = lambda x: x.decode(encoding, errors)
        else:
            decoder = codecs.getdecoder(encoding)
            f = lambda x: decoder(x, errors)[0]
        arr = self._data.array
        result = arr._str_map(f)
        return self._wrap_result(result, dtype=dtype)

    @forbid_nonstring_types(["bytes"])
    def encode(self, encoding, errors: str = "strict"):
        """
        Encode character string in the Series/Index using indicated encoding.

        Equivalent to :meth:`str.encode`.

        Parameters
        ----------
        encoding : str
        errors : str, optional

        Returns
        -------
        Series/Index of objects

        Examples
        --------
        >>> ser = pd.Series(['cow', '123', '()'])
        >>> ser.str.encode(encoding='ascii')
        0     b'cow'
        1     b'123'
        2      b'()'
        dtype: object
        """
        result = self._data.array._str_encode(encoding, errors)
        return self._wrap_result(result, returns_string=False)

    _shared_docs[
        "str_strip"
    ] = r"""
    Remove %(position)s characters.

    Strip whitespaces (including newlines) or a set of specified characters
    from each string in the Series/Index from %(side)s.
    Replaces any non-strings in Series with NaNs.
    Equivalent to :meth:`str.%(method)s`.

    Parameters
    ----------
    to_strip : str or None, default None
        Specifying the set of characters to be removed.
        All combinations of this set of characters will be stripped.
        If None then whitespaces are removed.

    Returns
    -------
    Series or Index of object

    See Also
    --------
    Series.str.strip : Remove leading and trailing characters in Series/Index.
    Series.str.lstrip : Remove leading characters in Series/Index.
    Series.str.rstrip : Remove trailing characters in Series/Index.

    Examples
    --------
    >>> s = pd.Series(['1. Ant.  ', '2. Bee!\n', '3. Cat?\t', np.nan, 10, True])
    >>> s
    0    1. Ant.
    1    2. Bee!\n
    2    3. Cat?\t
    3          NaN
    4           10
    5         True
    dtype: object

    >>> s.str.strip()
    0    1. Ant.
    1    2. Bee!
    2    3. Cat?
    3        NaN
    4        NaN
    5        NaN
    dtype: object

    >>> s.str.lstrip('123.')
    0    Ant.
    1    Bee!\n
    2    Cat?\t
    3       NaN
    4       NaN
    5       NaN
    dtype: object

    >>> s.str.rstrip('.!? \n\t')
    0    1. Ant
    1    2. Bee
    2    3. Cat
    3       NaN
    4       NaN
    5       NaN
    dtype: object

    >>> s.str.strip('123.!? \n\t')
    0    Ant
    1    Bee
    2    Cat
    3    NaN
    4    NaN
    5    NaN
    dtype: object
    """

    @Appender(
        _shared_docs["str_strip"]
        % {
            "side": "left and right sides",
            "method": "strip",
            "position": "leading and trailing",
        }
    )
    @forbid_nonstring_types(["bytes"])
    def strip(self, to_strip=None):
        result = self._data.array._str_strip(to_strip)
        return self._wrap_result(result)

    @Appender(
        _shared_docs["str_strip"]
        % {"side": "left side", "method": "lstrip", "position": "leading"}
    )
    @forbid_nonstring_types(["bytes"])
    def lstrip(self, to_strip=None):
        result = self._data.array._str_lstrip(to_strip)
        return self._wrap_result(result)

    @Appender(
        _shared_docs["str_strip"]
        % {"side": "right side", "method": "rstrip", "position": "trailing"}
    )
    @forbid_nonstring_types(["bytes"])
    def rstrip(self, to_strip=None):
        result = self._data.array._str_rstrip(to_strip)
        return self._wrap_result(result)

    _shared_docs[
        "str_removefix"
    ] = r"""
    Remove a %(side)s from an object series.

    If the %(side)s is not present, the original string will be returned.

    Parameters
    ----------
    %(side)s : str
        Remove the %(side)s of the string.

    Returns
    -------
    Series/Index: object
        The Series or Index with given %(side)s removed.

    See Also
    --------
    Series.str.remove%(other_side)s : Remove a %(other_side)s from an object series.

    Examples
    --------
    >>> s = pd.Series(["str_foo", "str_bar", "no_prefix"])
    >>> s
    0    str_foo
    1    str_bar
    2    no_prefix
    dtype: object
    >>> s.str.removeprefix("str_")
    0    foo
    1    bar
    2    no_prefix
    dtype: object

    >>> s = pd.Series(["foo_str", "bar_str", "no_suffix"])
    >>> s
    0    foo_str
    1    bar_str
    2    no_suffix
    dtype: object
    >>> s.str.removesuffix("_str")
    0    foo
    1    bar
    2    no_suffix
    dtype: object
    """

    @Appender(
        _shared_docs["str_removefix"] % {"side": "prefix", "other_side": "suffix"}
    )
    @forbid_nonstring_types(["bytes"])
    def removeprefix(self, prefix: str):
        result = self._data.array._str_removeprefix(prefix)
        return self._wrap_result(result)

    @Appender(
        _shared_docs["str_removefix"] % {"side": "suffix", "other_side": "prefix"}
    )
    @forbid_nonstring_types(["bytes"])
    def removesuffix(self, suffix: str):
        result = self._data.array._str_removesuffix(suffix)
        return self._wrap_result(result)

    @forbid_nonstring_types(["bytes"])
    def wrap(self, width: int, **kwargs):
        r"""
        Wrap strings in Series/Index at specified line width.

        This method has the same keyword parameters and defaults as
        :class:`textwrap.TextWrapper`.

        Parameters
        ----------
        width : int
            Maximum line width.
        expand_tabs : bool, optional
            If True, tab characters will be expanded to spaces (default: True).
        replace_whitespace : bool, optional
            If True, each whitespace character (as defined by string.whitespace)
            remaining after tab expansion will be replaced by a single space
            (default: True).
        drop_whitespace : bool, optional
            If True, whitespace that, after wrapping, happens to end up at the
            beginning or end of a line is dropped (default: True).
        break_long_words : bool, optional
            If True, then words longer than width will be broken in order to ensure
            that no lines are longer than width. If it is false, long words will
            not be broken, and some lines may be longer than width (default: True).
        break_on_hyphens : bool, optional
            If True, wrapping will occur preferably on whitespace and right after
            hyphens in compound words, as it is customary in English. If false,
            only whitespaces will be considered as potentially good places for line
            breaks, but you need to set break_long_words to false if you want truly
            insecable words (default: True).

        Returns
        -------
        Series or Index

        Notes
        -----
        Internally, this method uses a :class:`textwrap.TextWrapper` instance with
        default settings. To achieve behavior matching R's stringr library str_wrap
        function, use the arguments:

        - expand_tabs = False
        - replace_whitespace = True
        - drop_whitespace = True
        - break_long_words = False
        - break_on_hyphens = False

        Examples
        --------
        >>> s = pd.Series(['line to be wrapped', 'another line to be wrapped'])
        >>> s.str.wrap(12)
        0             line to be\nwrapped
        1    another line\nto be\nwrapped
        dtype: object
        """
        result = self._data.array._str_wrap(width, **kwargs)
        return self._wrap_result(result)

    @forbid_nonstring_types(["bytes"])
    def get_dummies(self, sep: str = "|"):
        """
        Return DataFrame of dummy/indicator variables for Series.

        Each string in Series is split by sep and returned as a DataFrame
        of dummy/indicator variables.

        Parameters
        ----------
        sep : str, default "|"
            String to split on.

        Returns
        -------
        DataFrame
            Dummy variables corresponding to values of the Series.

        See Also
        --------
        get_dummies : Convert categorical variable into dummy/indicator
            variables.

        Examples
        --------
        >>> pd.Series(['a|b', 'a', 'a|c']).str.get_dummies()
           a  b  c
        0  1  1  0
        1  1  0  0
        2  1  0  1

        >>> pd.Series(['a|b', np.nan, 'a|c']).str.get_dummies()
           a  b  c
        0  1  1  0
        1  0  0  0
        2  1  0  1
        """
        # we need to cast to Series of strings as only that has all
        # methods available for making the dummies...
        result, name = self._data.array._str_get_dummies(sep)
        return self._wrap_result(
            result,
            name=name,
            expand=True,
            returns_string=False,
        )

    @forbid_nonstring_types(["bytes"])
    def translate(self, table):
        """
        Map all characters in the string through the given mapping table.

        Equivalent to standard :meth:`str.translate`.

        Parameters
        ----------
        table : dict
            Table is a mapping of Unicode ordinals to Unicode ordinals, strings, or
            None. Unmapped characters are left untouched.
            Characters mapped to None are deleted. :meth:`str.maketrans` is a
            helper function for making translation tables.

        Returns
        -------
        Series or Index

        Examples
        --------
        >>> ser = pd.Series(["El niño", "Françoise"])
        >>> mytable = str.maketrans({'ñ': 'n', 'ç': 'c'})
        >>> ser.str.translate(mytable)
        0   El nino
        1   Francoise
        dtype: object
        """
        result = self._data.array._str_translate(table)
        dtype = object if self._data.dtype == "object" else None
        return self._wrap_result(result, dtype=dtype)

    @forbid_nonstring_types(["bytes"])
    def count(self, pat, flags: int = 0):
        r"""
        Count occurrences of pattern in each string of the Series/Index.

        This function is used to count the number of times a particular regex
        pattern is repeated in each of the string elements of the
        :class:`~pandas.Series`.

        Parameters
        ----------
        pat : str
            Valid regular expression.
        flags : int, default 0, meaning no flags
            Flags for the `re` module. For a complete list, `see here
            <https://docs.python.org/3/howto/regex.html#compilation-flags>`_.
        **kwargs
            For compatibility with other string methods. Not used.

        Returns
        -------
        Series or Index
            Same type as the calling object containing the integer counts.

        See Also
        --------
        re : Standard library module for regular expressions.
        str.count : Standard library version, without regular expression support.

        Notes
        -----
        Some characters need to be escaped when passing in `pat`.
        eg. ``'$'`` has a special meaning in regex and must be escaped when
        finding this literal character.

        Examples
        --------
        >>> s = pd.Series(['A', 'B', 'Aaba', 'Baca', np.nan, 'CABA', 'cat'])
        >>> s.str.count('a')
        0    0.0
        1    0.0
        2    2.0
        3    2.0
        4    NaN
        5    0.0
        6    1.0
        dtype: float64

        Escape ``'$'`` to find the literal dollar sign.

        >>> s = pd.Series(['$', 'B', 'Aab$', '$$ca', 'C$B$', 'cat'])
        >>> s.str.count('\\$')
        0    1
        1    0
        2    1
        3    2
        4    2
        5    0
        dtype: int64

        This is also available on Index

        >>> pd.Index(['A', 'A', 'Aaba', 'cat']).str.count('a')
        Index([0, 0, 2, 1], dtype='int64')
        """
        result = self._data.array._str_count(pat, flags)
        return self._wrap_result(result, returns_string=False)

    @forbid_nonstring_types(["bytes"])
    def startswith(
        self, pat: str | tuple[str, ...], na: Scalar | lib.NoDefault = lib.no_default
    ) -> Series | Index:
        """
        Test if the start of each string element matches a pattern.

        Equivalent to :meth:`str.startswith`.

        Parameters
        ----------
        pat : str or tuple[str, ...]
            Character sequence or tuple of strings. Regular expressions are not
            accepted.
        na : scalar, optional
            Object shown if element tested is not a string. The default depends
            on dtype of the array. For object-dtype, ``numpy.nan`` is used.
            For the nullable ``StringDtype``, ``pandas.NA`` is used.
            For the ``"str"`` dtype, ``False`` is used.

        Returns
        -------
        Series or Index of bool
            A Series of booleans indicating whether the given pattern matches
            the start of each string element.

        See Also
        --------
        str.startswith : Python standard library string method.
        Series.str.endswith : Same as startswith, but tests the end of string.
        Series.str.contains : Tests if string element contains a pattern.

        Examples
        --------
        >>> s = pd.Series(['bat', 'Bear', 'cat', np.nan])
        >>> s
        0     bat
        1    Bear
        2     cat
        3     NaN
        dtype: object

        >>> s.str.startswith('b')
        0     True
        1    False
        2    False
        3      NaN
        dtype: object

        >>> s.str.startswith(('b', 'B'))
        0     True
        1     True
        2    False
        3      NaN
        dtype: object

        Specifying `na` to be `False` instead of `NaN`.

        >>> s.str.startswith('b', na=False)
        0     True
        1    False
        2    False
        3    False
        dtype: bool
        """
        if not isinstance(pat, (str, tuple)):
            msg = f"expected a string or tuple, not {type(pat).__name__}"
            raise TypeError(msg)
        result = self._data.array._str_startswith(pat, na=na)
        return self._wrap_result(result, returns_string=False)

    @forbid_nonstring_types(["bytes"])
    def endswith(
        self, pat: str | tuple[str, ...], na: Scalar | lib.NoDefault = lib.no_default
    ) -> Series | Index:
        """
        Test if the end of each string element matches a pattern.

        Equivalent to :meth:`str.endswith`.

        Parameters
        ----------
        pat : str or tuple[str, ...]
            Character sequence or tuple of strings. Regular expressions are not
            accepted.
        na : scalar, optional
            Object shown if element tested is not a string. The default depends
            on dtype of the array. For object-dtype, ``numpy.nan`` is used.
            For the nullable ``StringDtype``, ``pandas.NA`` is used.
            For the ``"str"`` dtype, ``False`` is used.

        Returns
        -------
        Series or Index of bool
            A Series of booleans indicating whether the given pattern matches
            the end of each string element.

        See Also
        --------
        str.endswith : Python standard library string method.
        Series.str.startswith : Same as endswith, but tests the start of string.
        Series.str.contains : Tests if string element contains a pattern.

        Examples
        --------
        >>> s = pd.Series(['bat', 'bear', 'caT', np.nan])
        >>> s
        0     bat
        1    bear
        2     caT
        3     NaN
        dtype: object

        >>> s.str.endswith('t')
        0     True
        1    False
        2    False
        3      NaN
        dtype: object

        >>> s.str.endswith(('t', 'T'))
        0     True
        1    False
        2     True
        3      NaN
        dtype: object

        Specifying `na` to be `False` instead of `NaN`.

        >>> s.str.endswith('t', na=False)
        0     True
        1    False
        2    False
        3    False
        dtype: bool
        """
        if not isinstance(pat, (str, tuple)):
            msg = f"expected a string or tuple, not {type(pat).__name__}"
            raise TypeError(msg)
        result = self._data.array._str_endswith(pat, na=na)
        return self._wrap_result(result, returns_string=False)

    @forbid_nonstring_types(["bytes"])
    def findall(self, pat, flags: int = 0):
        """
        Find all occurrences of pattern or regular expression in the Series/Index.

        Equivalent to applying :func:`re.findall` to all the elements in the
        Series/Index.

        Parameters
        ----------
        pat : str
            Pattern or regular expression.
        flags : int, default 0
            Flags from ``re`` module, e.g. `re.IGNORECASE` (default is 0, which
            means no flags).

        Returns
        -------
        Series/Index of lists of strings
            All non-overlapping matches of pattern or regular expression in each
            string of this Series/Index.

        See Also
        --------
        count : Count occurrences of pattern or regular expression in each string
            of the Series/Index.
        extractall : For each string in the Series, extract groups from all matches
            of regular expression and return a DataFrame with one row for each
            match and one column for each group.
        re.findall : The equivalent ``re`` function to all non-overlapping matches
            of pattern or regular expression in string, as a list of strings.

        Examples
        --------
        >>> s = pd.Series(['Lion', 'Monkey', 'Rabbit'])

        The search for the pattern 'Monkey' returns one match:

        >>> s.str.findall('Monkey')
        0          []
        1    [Monkey]
        2          []
        dtype: object

        On the other hand, the search for the pattern 'MONKEY' doesn't return any
        match:

        >>> s.str.findall('MONKEY')
        0    []
        1    []
        2    []
        dtype: object

        Flags can be added to the pattern or regular expression. For instance,
        to find the pattern 'MONKEY' ignoring the case:

        >>> import re
        >>> s.str.findall('MONKEY', flags=re.IGNORECASE)
        0          []
        1    [Monkey]
        2          []
        dtype: object

        When the pattern matches more than one string in the Series, all matches
        are returned:

        >>> s.str.findall('on')
        0    [on]
        1    [on]
        2      []
        dtype: object

        Regular expressions are supported too. For instance, the search for all the
        strings ending with the word 'on' is shown next:

        >>> s.str.findall('on$')
        0    [on]
        1      []
        2      []
        dtype: object

        If the pattern is found more than once in the same string, then a list of
        multiple strings is returned:

        >>> s.str.findall('b')
        0        []
        1        []
        2    [b, b]
        dtype: object
        """
        result = self._data.array._str_findall(pat, flags)
        return self._wrap_result(result, returns_string=False)

    @forbid_nonstring_types(["bytes"])
    def extract(
        self, pat: str, flags: int = 0, expand: bool = True
    ) -> DataFrame | Series | Index:
        r"""
        Extract capture groups in the regex `pat` as columns in a DataFrame.

        For each subject string in the Series, extract groups from the
        first match of regular expression `pat`.

        Parameters
        ----------
        pat : str
            Regular expression pattern with capturing groups.
        flags : int, default 0 (no flags)
            Flags from the ``re`` module, e.g. ``re.IGNORECASE``, that
            modify regular expression matching for things like case,
            spaces, etc. For more details, see :mod:`re`.
        expand : bool, default True
            If True, return DataFrame with one column per capture group.
            If False, return a Series/Index if there is one capture group
            or DataFrame if there are multiple capture groups.

        Returns
        -------
        DataFrame or Series or Index
            A DataFrame with one row for each subject string, and one
            column for each group. Any capture group names in regular
            expression pat will be used for column names; otherwise
            capture group numbers will be used. The dtype of each result
            column is always object, even when no match is found. If
            ``expand=False`` and pat has only one capture group, then
            return a Series (if subject is a Series) or Index (if subject
            is an Index).

        See Also
        --------
        extractall : Returns all matches (not just the first match).

        Examples
        --------
        A pattern with two groups will return a DataFrame with two columns.
        Non-matches will be NaN.

        >>> s = pd.Series(['a1', 'b2', 'c3'])
        >>> s.str.extract(r'([ab])(\d)')
            0    1
        0    a    1
        1    b    2
        2  NaN  NaN

        A pattern may contain optional groups.

        >>> s.str.extract(r'([ab])?(\d)')
            0  1
        0    a  1
        1    b  2
        2  NaN  3

        Named groups will become column names in the result.

        >>> s.str.extract(r'(?P<letter>[ab])(?P<digit>\d)')
        letter digit
        0      a     1
        1      b     2
        2    NaN   NaN

        A pattern with one group will return a DataFrame with one column
        if expand=True.

        >>> s.str.extract(r'[ab](\d)', expand=True)
            0
        0    1
        1    2
        2  NaN

        A pattern with one group will return a Series if expand=False.

        >>> s.str.extract(r'[ab](\d)', expand=False)
        0      1
        1      2
        2    NaN
        dtype: object
        """
        from pandas import DataFrame

        if not isinstance(expand, bool):
            raise ValueError("expand must be True or False")

        regex = re.compile(pat, flags=flags)
        if regex.groups == 0:
            raise ValueError("pattern contains no capture groups")

        if not expand and regex.groups > 1 and isinstance(self._data, ABCIndex):
            raise ValueError("only one regex group is supported with Index")

        obj = self._data
        result_dtype = _result_dtype(obj)

        returns_df = regex.groups > 1 or expand

        if returns_df:
            name = None
            columns = _get_group_names(regex)

            if obj.array.size == 0:
                result = DataFrame(columns=columns, dtype=result_dtype)

            else:
                result_list = self._data.array._str_extract(
                    pat, flags=flags, expand=returns_df
                )

                result_index: Index | None
                if isinstance(obj, ABCSeries):
                    result_index = obj.index
                else:
                    result_index = None

                result = DataFrame(
                    result_list, columns=columns, index=result_index, dtype=result_dtype
                )

        else:
            name = _get_single_group_name(regex)
            result = self._data.array._str_extract(pat, flags=flags, expand=returns_df)
        return self._wrap_result(result, name=name, dtype=result_dtype)

    @forbid_nonstring_types(["bytes"])
    def extractall(self, pat, flags: int = 0) -> DataFrame:
        r"""
        Extract capture groups in the regex `pat` as columns in DataFrame.

        For each subject string in the Series, extract groups from all
        matches of regular expression pat. When each subject string in the
        Series has exactly one match, extractall(pat).xs(0, level='match')
        is the same as extract(pat).

        Parameters
        ----------
        pat : str
            Regular expression pattern with capturing groups.
        flags : int, default 0 (no flags)
            A ``re`` module flag, for example ``re.IGNORECASE``. These allow
            to modify regular expression matching for things like case, spaces,
            etc. Multiple flags can be combined with the bitwise OR operator,
            for example ``re.IGNORECASE | re.MULTILINE``.

        Returns
        -------
        DataFrame
            A ``DataFrame`` with one row for each match, and one column for each
            group. Its rows have a ``MultiIndex`` with first levels that come from
            the subject ``Series``. The last level is named 'match' and indexes the
            matches in each item of the ``Series``. Any capture group names in
            regular expression pat will be used for column names; otherwise capture
            group numbers will be used.

        See Also
        --------
        extract : Returns first match only (not all matches).

        Examples
        --------
        A pattern with one group will return a DataFrame with one column.
        Indices with no matches will not appear in the result.

        >>> s = pd.Series(["a1a2", "b1", "c1"], index=["A", "B", "C"])
        >>> s.str.extractall(r"[ab](\d)")
                0
        match
        A 0      1
          1      2
        B 0      1

        Capture group names are used for column names of the result.

        >>> s.str.extractall(r"[ab](?P<digit>\d)")
                digit
        match
        A 0         1
          1         2
        B 0         1

        A pattern with two groups will return a DataFrame with two columns.

        >>> s.str.extractall(r"(?P<letter>[ab])(?P<digit>\d)")
                letter digit
        match
        A 0          a     1
          1          a     2
        B 0          b     1

        Optional groups that do not match are NaN in the result.

        >>> s.str.extractall(r"(?P<letter>[ab])?(?P<digit>\d)")
                letter digit
        match
        A 0          a     1
          1          a     2
        B 0          b     1
        C 0        NaN     1
        """
        # TODO: dispatch
        return str_extractall(self._orig, pat, flags)

    _shared_docs[
        "find"
    ] = """
    Return %(side)s indexes in each strings in the Series/Index.

    Each of returned indexes corresponds to the position where the
    substring is fully contained between [start:end]. Return -1 on
    failure. Equivalent to standard :meth:`str.%(method)s`.

    Parameters
    ----------
    sub : str
        Substring being searched.
    start : int
        Left edge index.
    end : int
        Right edge index.

    Returns
    -------
    Series or Index of int.

    See Also
    --------
    %(also)s

    Examples
    --------
    For Series.str.find:

    >>> ser = pd.Series(["cow_", "duck_", "do_ve"])
    >>> ser.str.find("_")
    0   3
    1   4
    2   2
    dtype: int64

    For Series.str.rfind:

    >>> ser = pd.Series(["_cow_", "duck_", "do_v_e"])
    >>> ser.str.rfind("_")
    0   4
    1   4
    2   4
    dtype: int64
    """

    @Appender(
        _shared_docs["find"]
        % {
            "side": "lowest",
            "method": "find",
            "also": "rfind : Return highest indexes in each strings.",
        }
    )
    @forbid_nonstring_types(["bytes"])
    def find(self, sub, start: int = 0, end=None):
        if not isinstance(sub, str):
            msg = f"expected a string object, not {type(sub).__name__}"
            raise TypeError(msg)

        result = self._data.array._str_find(sub, start, end)
        return self._wrap_result(result, returns_string=False)

    @Appender(
        _shared_docs["find"]
        % {
            "side": "highest",
            "method": "rfind",
            "also": "find : Return lowest indexes in each strings.",
        }
    )
    @forbid_nonstring_types(["bytes"])
    def rfind(self, sub, start: int = 0, end=None):
        if not isinstance(sub, str):
            msg = f"expected a string object, not {type(sub).__name__}"
            raise TypeError(msg)

        result = self._data.array._str_rfind(sub, start=start, end=end)
        return self._wrap_result(result, returns_string=False)

    @forbid_nonstring_types(["bytes"])
    def normalize(self, form):
        """
        Return the Unicode normal form for the strings in the Series/Index.

        For more information on the forms, see the
        :func:`unicodedata.normalize`.

        Parameters
        ----------
        form : {'NFC', 'NFKC', 'NFD', 'NFKD'}
            Unicode form.

        Returns
        -------
        Series/Index of objects

        Examples
        --------
        >>> ser = pd.Series(['ñ'])
        >>> ser.str.normalize('NFC') == ser.str.normalize('NFD')
        0   False
        dtype: bool
        """
        result = self._data.array._str_normalize(form)
        return self._wrap_result(result)

    _shared_docs[
        "index"
    ] = """
    Return %(side)s indexes in each string in Series/Index.

    Each of the returned indexes corresponds to the position where the
    substring is fully contained between [start:end]. This is the same
    as ``str.%(similar)s`` except instead of returning -1, it raises a
    ValueError when the substring is not found. Equivalent to standard
    ``str.%(method)s``.

    Parameters
    ----------
    sub : str
        Substring being searched.
    start : int
        Left edge index.
    end : int
        Right edge index.

    Returns
    -------
    Series or Index of object

    See Also
    --------
    %(also)s

    Examples
    --------
    For Series.str.index:

    >>> ser = pd.Series(["horse", "eagle", "donkey"])
    >>> ser.str.index("e")
    0   4
    1   0
    2   4
    dtype: int64

    For Series.str.rindex:

    >>> ser = pd.Series(["Deer", "eagle", "Sheep"])
    >>> ser.str.rindex("e")
    0   2
    1   4
    2   3
    dtype: int64
    """

    @Appender(
        _shared_docs["index"]
        % {
            "side": "lowest",
            "similar": "find",
            "method": "index",
            "also": "rindex : Return highest indexes in each strings.",
        }
    )
    @forbid_nonstring_types(["bytes"])
    def index(self, sub, start: int = 0, end=None):
        if not isinstance(sub, str):
            msg = f"expected a string object, not {type(sub).__name__}"
            raise TypeError(msg)

        result = self._data.array._str_index(sub, start=start, end=end)
        return self._wrap_result(result, returns_string=False)

    @Appender(
        _shared_docs["index"]
        % {
            "side": "highest",
            "similar": "rfind",
            "method": "rindex",
            "also": "index : Return lowest indexes in each strings.",
        }
    )
    @forbid_nonstring_types(["bytes"])
    def rindex(self, sub, start: int = 0, end=None):
        if not isinstance(sub, str):
            msg = f"expected a string object, not {type(sub).__name__}"
            raise TypeError(msg)

        result = self._data.array._str_rindex(sub, start=start, end=end)
        return self._wrap_result(result, returns_string=False)

    def len(self):
        """
        Compute the length of each element in the Series/Index.

        The element may be a sequence (such as a string, tuple or list) or a collection
        (such as a dictionary).

        Returns
        -------
        Series or Index of int
            A Series or Index of integer values indicating the length of each
            element in the Series or Index.

        See Also
        --------
        str.len : Python built-in function returning the length of an object.
        Series.size : Returns the length of the Series.

        Examples
        --------
        Returns the length (number of characters) in a string. Returns the
        number of entries for dictionaries, lists or tuples.

        >>> s = pd.Series(['dog',
        ...                 '',
        ...                 5,
        ...                 {'foo' : 'bar'},
        ...                 [2, 3, 5, 7],
        ...                 ('one', 'two', 'three')])
        >>> s
        0                  dog
        1
        2                    5
        3       {'foo': 'bar'}
        4         [2, 3, 5, 7]
        5    (one, two, three)
        dtype: object
        >>> s.str.len()
        0    3.0
        1    0.0
        2    NaN
        3    1.0
        4    4.0
        5    3.0
        dtype: float64
        """
        result = self._data.array._str_len()
        return self._wrap_result(result, returns_string=False)

    _shared_docs[
        "casemethods"
    ] = """
    Convert strings in the Series/Index to %(type)s.
    %(version)s
    Equivalent to :meth:`str.%(method)s`.

    Returns
    -------
    Series or Index of object

    See Also
    --------
    Series.str.lower : Converts all characters to lowercase.
    Series.str.upper : Converts all characters to uppercase.
    Series.str.title : Converts first character of each word to uppercase and
        remaining to lowercase.
    Series.str.capitalize : Converts first character to uppercase and
        remaining to lowercase.
    Series.str.swapcase : Converts uppercase to lowercase and lowercase to
        uppercase.
    Series.str.casefold: Removes all case distinctions in the string.

    Examples
    --------
    >>> s = pd.Series(['lower', 'CAPITALS', 'this is a sentence', 'SwApCaSe'])
    >>> s
    0                 lower
    1              CAPITALS
    2    this is a sentence
    3              SwApCaSe
    dtype: object

    >>> s.str.lower()
    0                 lower
    1              capitals
    2    this is a sentence
    3              swapcase
    dtype: object

    >>> s.str.upper()
    0                 LOWER
    1              CAPITALS
    2    THIS IS A SENTENCE
    3              SWAPCASE
    dtype: object

    >>> s.str.title()
    0                 Lower
    1              Capitals
    2    This Is A Sentence
    3              Swapcase
    dtype: object

    >>> s.str.capitalize()
    0                 Lower
    1              Capitals
    2    This is a sentence
    3              Swapcase
    dtype: object

    >>> s.str.swapcase()
    0                 LOWER
    1              capitals
    2    THIS IS A SENTENCE
    3              sWaPcAsE
    dtype: object
    """
    # Types:
    #   cases:
    #       upper, lower, title, capitalize, swapcase, casefold
    #   boolean:
    #     isalpha, isnumeric isalnum isdigit isdecimal isspace islower isupper istitle
    # _doc_args holds dict of strings to use in substituting casemethod docs
    _doc_args: dict[str, dict[str, str]] = {}
    _doc_args["lower"] = {"type": "lowercase", "method": "lower", "version": ""}
    _doc_args["upper"] = {"type": "uppercase", "method": "upper", "version": ""}
    _doc_args["title"] = {"type": "titlecase", "method": "title", "version": ""}
    _doc_args["capitalize"] = {
        "type": "be capitalized",
        "method": "capitalize",
        "version": "",
    }
    _doc_args["swapcase"] = {
        "type": "be swapcased",
        "method": "swapcase",
        "version": "",
    }
    _doc_args["casefold"] = {
        "type": "be casefolded",
        "method": "casefold",
        "version": "",
    }

    @Appender(_shared_docs["casemethods"] % _doc_args["lower"])
    @forbid_nonstring_types(["bytes"])
    def lower(self):
        result = self._data.array._str_lower()
        return self._wrap_result(result)

    @Appender(_shared_docs["casemethods"] % _doc_args["upper"])
    @forbid_nonstring_types(["bytes"])
    def upper(self):
        result = self._data.array._str_upper()
        return self._wrap_result(result)

    @Appender(_shared_docs["casemethods"] % _doc_args["title"])
    @forbid_nonstring_types(["bytes"])
    def title(self):
        result = self._data.array._str_title()
        return self._wrap_result(result)

    @Appender(_shared_docs["casemethods"] % _doc_args["capitalize"])
    @forbid_nonstring_types(["bytes"])
    def capitalize(self):
        result = self._data.array._str_capitalize()
        return self._wrap_result(result)

    @Appender(_shared_docs["casemethods"] % _doc_args["swapcase"])
    @forbid_nonstring_types(["bytes"])
    def swapcase(self):
        result = self._data.array._str_swapcase()
        return self._wrap_result(result)

    @Appender(_shared_docs["casemethods"] % _doc_args["casefold"])
    @forbid_nonstring_types(["bytes"])
    def casefold(self):
        result = self._data.array._str_casefold()
        return self._wrap_result(result)

    _shared_docs[
        "ismethods"
    ] = """
    Check whether all characters in each string are %(type)s.

    This is equivalent to running the Python string method
    :meth:`str.%(method)s` for each element of the Series/Index. If a string
    has zero characters, ``False`` is returned for that check.

    Returns
    -------
    Series or Index of bool
        Series or Index of boolean values with the same length as the original
        Series/Index.

    See Also
    --------
    Series.str.isalpha : Check whether all characters are alphabetic.
    Series.str.isnumeric : Check whether all characters are numeric.
    Series.str.isalnum : Check whether all characters are alphanumeric.
    Series.str.isdigit : Check whether all characters are digits.
    Series.str.isdecimal : Check whether all characters are decimal.
    Series.str.isspace : Check whether all characters are whitespace.
    Series.str.islower : Check whether all characters are lowercase.
    Series.str.isupper : Check whether all characters are uppercase.
    Series.str.istitle : Check whether all characters are titlecase.

    Examples
    --------
    **Checks for Alphabetic and Numeric Characters**

    >>> s1 = pd.Series(['one', 'one1', '1', ''])

    >>> s1.str.isalpha()
    0     True
    1    False
    2    False
    3    False
    dtype: bool

    >>> s1.str.isnumeric()
    0    False
    1    False
    2     True
    3    False
    dtype: bool

    >>> s1.str.isalnum()
    0     True
    1     True
    2     True
    3    False
    dtype: bool

    Note that checks against characters mixed with any additional punctuation
    or whitespace will evaluate to false for an alphanumeric check.

    >>> s2 = pd.Series(['A B', '1.5', '3,000'])
    >>> s2.str.isalnum()
    0    False
    1    False
    2    False
    dtype: bool

    **More Detailed Checks for Numeric Characters**

    There are several different but overlapping sets of numeric characters that
    can be checked for.

    >>> s3 = pd.Series(['23', '³', '⅕', ''])

    The ``s3.str.isdecimal`` method checks for characters used to form numbers
    in base 10.

    >>> s3.str.isdecimal()
    0     True
    1    False
    2    False
    3    False
    dtype: bool

    The ``s.str.isdigit`` method is the same as ``s3.str.isdecimal`` but also
    includes special digits, like superscripted and subscripted digits in
    unicode.

    >>> s3.str.isdigit()
    0     True
    1     True
    2    False
    3    False
    dtype: bool

    The ``s.str.isnumeric`` method is the same as ``s3.str.isdigit`` but also
    includes other characters that can represent quantities such as unicode
    fractions.

    >>> s3.str.isnumeric()
    0     True
    1     True
    2     True
    3    False
    dtype: bool

    **Checks for Whitespace**

    >>> s4 = pd.Series([' ', '\\t\\r\\n ', ''])
    >>> s4.str.isspace()
    0     True
    1     True
    2    False
    dtype: bool

    **Checks for Character Case**

    >>> s5 = pd.Series(['leopard', 'Golden Eagle', 'SNAKE', ''])

    >>> s5.str.islower()
    0     True
    1    False
    2    False
    3    False
    dtype: bool

    >>> s5.str.isupper()
    0    False
    1    False
    2     True
    3    False
    dtype: bool

    The ``s5.str.istitle`` method checks for whether all words are in title
    case (whether only the first letter of each word is capitalized). Words are
    assumed to be as any sequence of non-numeric characters separated by
    whitespace characters.

    >>> s5.str.istitle()
    0    False
    1     True
    2    False
    3    False
    dtype: bool
    """
    _doc_args["isalnum"] = {"type": "alphanumeric", "method": "isalnum"}
    _doc_args["isalpha"] = {"type": "alphabetic", "method": "isalpha"}
    _doc_args["isdigit"] = {"type": "digits", "method": "isdigit"}
    _doc_args["isspace"] = {"type": "whitespace", "method": "isspace"}
    _doc_args["islower"] = {"type": "lowercase", "method": "islower"}
    _doc_args["isupper"] = {"type": "uppercase", "method": "isupper"}
    _doc_args["istitle"] = {"type": "titlecase", "method": "istitle"}
    _doc_args["isnumeric"] = {"type": "numeric", "method": "isnumeric"}
    _doc_args["isdecimal"] = {"type": "decimal", "method": "isdecimal"}
    # force _noarg_wrapper return type with dtype=np.dtype(bool) (GH 29624)

    isalnum = _map_and_wrap(
        "isalnum", docstring=_shared_docs["ismethods"] % _doc_args["isalnum"]
    )
    isalpha = _map_and_wrap(
        "isalpha", docstring=_shared_docs["ismethods"] % _doc_args["isalpha"]
    )
    isdigit = _map_and_wrap(
        "isdigit", docstring=_shared_docs["ismethods"] % _doc_args["isdigit"]
    )
    isspace = _map_and_wrap(
        "isspace", docstring=_shared_docs["ismethods"] % _doc_args["isspace"]
    )
    islower = _map_and_wrap(
        "islower", docstring=_shared_docs["ismethods"] % _doc_args["islower"]
    )
    isupper = _map_and_wrap(
        "isupper", docstring=_shared_docs["ismethods"] % _doc_args["isupper"]
    )
    istitle = _map_and_wrap(
        "istitle", docstring=_shared_docs["ismethods"] % _doc_args["istitle"]
    )
    isnumeric = _map_and_wrap(
        "isnumeric", docstring=_shared_docs["ismethods"] % _doc_args["isnumeric"]
    )
    isdecimal = _map_and_wrap(
        "isdecimal", docstring=_shared_docs["ismethods"] % _doc_args["isdecimal"]
    )


def cat_safe(list_of_columns: list[npt.NDArray[np.object_]], sep: str):
    """
    Auxiliary function for :meth:`str.cat`.

    Same signature as cat_core, but handles TypeErrors in concatenation, which
    happen if the arrays in list_of columns have the wrong dtypes or content.

    Parameters
    ----------
    list_of_columns : list of numpy arrays
        List of arrays to be concatenated with sep;
        these arrays may not contain NaNs!
    sep : string
        The separator string for concatenating the columns.

    Returns
    -------
    nd.array
        The concatenation of list_of_columns with sep.
    """
    try:
        result = cat_core(list_of_columns, sep)
    except TypeError:
        # if there are any non-string values (wrong dtype or hidden behind
        # object dtype), np.sum will fail; catch and return with better message
        for column in list_of_columns:
            dtype = lib.infer_dtype(column, skipna=True)
            if dtype not in ["string", "empty"]:
                raise TypeError(
                    "Concatenation requires list-likes containing only "
                    "strings (or missing values). Offending values found in "
                    f"column {dtype}"
                ) from None
    return result


def cat_core(list_of_columns: list, sep: str):
    """
    Auxiliary function for :meth:`str.cat`

    Parameters
    ----------
    list_of_columns : list of numpy arrays
        List of arrays to be concatenated with sep;
        these arrays may not contain NaNs!
    sep : string
        The separator string for concatenating the columns.

    Returns
    -------
    nd.array
        The concatenation of list_of_columns with sep.
    """
    if sep == "":
        # no need to interleave sep if it is empty
        arr_of_cols = np.asarray(list_of_columns, dtype=object)
        return np.sum(arr_of_cols, axis=0)
    list_with_sep = [sep] * (2 * len(list_of_columns) - 1)
    list_with_sep[::2] = list_of_columns
    arr_with_sep = np.asarray(list_with_sep, dtype=object)
    return np.sum(arr_with_sep, axis=0)


def _result_dtype(arr):
    # workaround #27953
    # ideally we just pass `dtype=arr.dtype` unconditionally, but this fails
    # when the list of values is empty.
    from pandas.core.arrays.string_ import StringDtype

    if isinstance(arr.dtype, (ArrowDtype, StringDtype)):
        return arr.dtype
    return object


def _get_single_group_name(regex: re.Pattern) -> Hashable:
    if regex.groupindex:
        return next(iter(regex.groupindex))
    else:
        return None


def _get_group_names(regex: re.Pattern) -> list[Hashable]:
    """
    Get named groups from compiled regex.

    Unnamed groups are numbered.

    Parameters
    ----------
    regex : compiled regex

    Returns
    -------
    list of column labels
    """
    names = {v: k for k, v in regex.groupindex.items()}
    return [names.get(1 + i, i) for i in range(regex.groups)]


def str_extractall(arr, pat, flags: int = 0) -> DataFrame:
    regex = re.compile(pat, flags=flags)
    # the regex must contain capture groups.
    if regex.groups == 0:
        raise ValueError("pattern contains no capture groups")

    if isinstance(arr, ABCIndex):
        arr = arr.to_series().reset_index(drop=True).astype(arr.dtype)

    columns = _get_group_names(regex)
    match_list = []
    index_list = []
    is_mi = arr.index.nlevels > 1

    for subject_key, subject in arr.items():
        if isinstance(subject, str):
            if not is_mi:
                subject_key = (subject_key,)

            for match_i, match_tuple in enumerate(regex.findall(subject)):
                if isinstance(match_tuple, str):
                    match_tuple = (match_tuple,)
                na_tuple = [np.nan if group == "" else group for group in match_tuple]
                match_list.append(na_tuple)
                result_key = tuple(subject_key + (match_i,))
                index_list.append(result_key)

    from pandas import MultiIndex

    index = MultiIndex.from_tuples(index_list, names=arr.index.names + ["match"])
    dtype = _result_dtype(arr)

    result = arr._constructor_expanddim(
        match_list, index=index, columns=columns, dtype=dtype
    )
    return result
<|endoftext|>from __future__ import annotations

import abc
from typing import (
    TYPE_CHECKING,
    Callable,
    Literal,
)

from pandas._libs import lib

if TYPE_CHECKING:
    from collections.abc import Sequence
    import re

    from pandas._typing import Scalar

    from pandas import Series


class BaseStringArrayMethods(abc.ABC):
    """
    Base class for extension arrays implementing string methods.

    This is where our ExtensionArrays can override the implementation of
    Series.str.<method>. We don't expect this to work with
    3rd-party extension arrays.

    * User calls Series.str.<method>
    * pandas extracts the extension array from the Series
    * pandas calls ``extension_array._str_<method>(*args, **kwargs)``
    * pandas wraps the result, to return to the user.

    See :ref:`Series.str` for the docstring of each method.
    """

    def _str_getitem(self, key):
        if isinstance(key, slice):
            return self._str_slice(start=key.start, stop=key.stop, step=key.step)
        else:
            return self._str_get(key)

    @abc.abstractmethod
    def _str_count(self, pat, flags: int = 0):
        pass

    @abc.abstractmethod
    def _str_pad(
        self,
        width: int,
        side: Literal["left", "right", "both"] = "left",
        fillchar: str = " ",
    ):
        pass

    @abc.abstractmethod
    def _str_contains(
        self, pat, case: bool = True, flags: int = 0, na=None, regex: bool = True
    ):
        pass

    @abc.abstractmethod
    def _str_startswith(self, pat, na=None):
        pass

    @abc.abstractmethod
    def _str_endswith(self, pat, na=None):
        pass

    @abc.abstractmethod
    def _str_replace(
        self,
        pat: str | re.Pattern,
        repl: str | Callable,
        n: int = -1,
        case: bool = True,
        flags: int = 0,
        regex: bool = True,
    ):
        pass

    @abc.abstractmethod
    def _str_repeat(self, repeats: int | Sequence[int]):
        pass

    @abc.abstractmethod
    def _str_match(
        self,
        pat: str,
        case: bool = True,
        flags: int = 0,
        na: Scalar | lib.NoDefault = lib.no_default,
    ):
        pass

    @abc.abstractmethod
    def _str_fullmatch(
        self,
        pat: str | re.Pattern,
        case: bool = True,
        flags: int = 0,
        na: Scalar | lib.NoDefault = lib.no_default,
    ):
        pass

    @abc.abstractmethod
    def _str_encode(self, encoding, errors: str = "strict"):
        pass

    @abc.abstractmethod
    def _str_find(self, sub, start: int = 0, end=None):
        pass

    @abc.abstractmethod
    def _str_rfind(self, sub, start: int = 0, end=None):
        pass

    @abc.abstractmethod
    def _str_findall(self, pat, flags: int = 0):
        pass

    @abc.abstractmethod
    def _str_get(self, i):
        pass

    @abc.abstractmethod
    def _str_index(self, sub, start: int = 0, end=None):
        pass

    @abc.abstractmethod
    def _str_rindex(self, sub, start: int = 0, end=None):
        pass

    @abc.abstractmethod
    def _str_join(self, sep: str):
        pass

    @abc.abstractmethod
    def _str_partition(self, sep: str, expand):
        pass

    @abc.abstractmethod
    def _str_rpartition(self, sep: str, expand):
        pass

    @abc.abstractmethod
    def _str_len(self):
        pass

    @abc.abstractmethod
    def _str_slice(self, start=None, stop=None, step=None):
        pass

    @abc.abstractmethod
    def _str_slice_replace(self, start=None, stop=None, repl=None):
        pass

    @abc.abstractmethod
    def _str_translate(self, table):
        pass

    @abc.abstractmethod
    def _str_wrap(self, width: int, **kwargs):
        pass

    @abc.abstractmethod
    def _str_get_dummies(self, sep: str = "|"):
        pass

    @abc.abstractmethod
    def _str_isalnum(self):
        pass

    @abc.abstractmethod
    def _str_isalpha(self):
        pass

    @abc.abstractmethod
    def _str_isdecimal(self):
        pass

    @abc.abstractmethod
    def _str_isdigit(self):
        pass

    @abc.abstractmethod
    def _str_islower(self):
        pass

    @abc.abstractmethod
    def _str_isnumeric(self):
        pass

    @abc.abstractmethod
    def _str_isspace(self):
        pass

    @abc.abstractmethod
    def _str_istitle(self):
        pass

    @abc.abstractmethod
    def _str_isupper(self):
        pass

    @abc.abstractmethod
    def _str_capitalize(self):
        pass

    @abc.abstractmethod
    def _str_casefold(self):
        pass

    @abc.abstractmethod
    def _str_title(self):
        pass

    @abc.abstractmethod
    def _str_swapcase(self):
        pass

    @abc.abstractmethod
    def _str_lower(self):
        pass

    @abc.abstractmethod
    def _str_upper(self):
        pass

    @abc.abstractmethod
    def _str_normalize(self, form):
        pass

    @abc.abstractmethod
    def _str_strip(self, to_strip=None):
        pass

    @abc.abstractmethod
    def _str_lstrip(self, to_strip=None):
        pass

    @abc.abstractmethod
    def _str_rstrip(self, to_strip=None):
        pass

    @abc.abstractmethod
    def _str_removeprefix(self, prefix: str) -> Series:
        pass

    @abc.abstractmethod
    def _str_removesuffix(self, suffix: str) -> Series:
        pass

    @abc.abstractmethod
    def _str_split(
        self, pat=None, n=-1, expand: bool = False, regex: bool | None = None
    ):
        pass

    @abc.abstractmethod
    def _str_rsplit(self, pat=None, n=-1):
        pass

    @abc.abstractmethod
    def _str_extract(self, pat: str, flags: int = 0, expand: bool = True):
        pass
<|endoftext|>"""
Templating for ops docstrings
"""
from __future__ import annotations


def make_flex_doc(op_name: str, typ: str) -> str:
    """
    Make the appropriate substitutions for the given operation and class-typ
    into either _flex_doc_SERIES or _flex_doc_FRAME to return the docstring
    to attach to a generated method.

    Parameters
    ----------
    op_name : str {'__add__', '__sub__', ... '__eq__', '__ne__', ...}
    typ : str {series, 'dataframe']}

    Returns
    -------
    doc : str
    """
    op_name = op_name.replace("__", "")
    op_desc = _op_descriptions[op_name]

    op_desc_op = op_desc["op"]
    assert op_desc_op is not None  # for mypy
    if op_name.startswith("r"):
        equiv = f"other {op_desc_op} {typ}"
    elif op_name == "divmod":
        equiv = f"{op_name}({typ}, other)"
    else:
        equiv = f"{typ} {op_desc_op} other"

    if typ == "series":
        base_doc = _flex_doc_SERIES
        if op_desc["reverse"]:
            base_doc += _see_also_reverse_SERIES.format(
                reverse=op_desc["reverse"], see_also_desc=op_desc["see_also_desc"]
            )
        doc_no_examples = base_doc.format(
            desc=op_desc["desc"],
            op_name=op_name,
            equiv=equiv,
            series_returns=op_desc["series_returns"],
        )
        ser_example = op_desc["series_examples"]
        if ser_example:
            doc = doc_no_examples + ser_example
        else:
            doc = doc_no_examples
    elif typ == "dataframe":
        if op_name in ["eq", "ne", "le", "lt", "ge", "gt"]:
            base_doc = _flex_comp_doc_FRAME
            doc = _flex_comp_doc_FRAME.format(
                op_name=op_name,
                desc=op_desc["desc"],
            )
        else:
            base_doc = _flex_doc_FRAME
            doc = base_doc.format(
                desc=op_desc["desc"],
                op_name=op_name,
                equiv=equiv,
                reverse=op_desc["reverse"],
            )
    else:
        raise AssertionError("Invalid typ argument.")
    return doc


_common_examples_algebra_SERIES = """
Examples
--------
>>> a = pd.Series([1, 1, 1, np.nan], index=['a', 'b', 'c', 'd'])
>>> a
a    1.0
b    1.0
c    1.0
d    NaN
dtype: float64
>>> b = pd.Series([1, np.nan, 1, np.nan], index=['a', 'b', 'd', 'e'])
>>> b
a    1.0
b    NaN
d    1.0
e    NaN
dtype: float64"""

_common_examples_comparison_SERIES = """
Examples
--------
>>> a = pd.Series([1, 1, 1, np.nan, 1], index=['a', 'b', 'c', 'd', 'e'])
>>> a
a    1.0
b    1.0
c    1.0
d    NaN
e    1.0
dtype: float64
>>> b = pd.Series([0, 1, 2, np.nan, 1], index=['a', 'b', 'c', 'd', 'f'])
>>> b
a    0.0
b    1.0
c    2.0
d    NaN
f    1.0
dtype: float64"""

_add_example_SERIES = (
    _common_examples_algebra_SERIES
    + """
>>> a.add(b, fill_value=0)
a    2.0
b    1.0
c    1.0
d    1.0
e    NaN
dtype: float64
"""
)

_sub_example_SERIES = (
    _common_examples_algebra_SERIES
    + """
>>> a.subtract(b, fill_value=0)
a    0.0
b    1.0
c    1.0
d   -1.0
e    NaN
dtype: float64
"""
)

_mul_example_SERIES = (
    _common_examples_algebra_SERIES
    + """
>>> a.multiply(b, fill_value=0)
a    1.0
b    0.0
c    0.0
d    0.0
e    NaN
dtype: float64
"""
)

_div_example_SERIES = (
    _common_examples_algebra_SERIES
    + """
>>> a.divide(b, fill_value=0)
a    1.0
b    inf
c    inf
d    0.0
e    NaN
dtype: float64
"""
)

_floordiv_example_SERIES = (
    _common_examples_algebra_SERIES
    + """
>>> a.floordiv(b, fill_value=0)
a    1.0
b    inf
c    inf
d    0.0
e    NaN
dtype: float64
"""
)

_divmod_example_SERIES = (
    _common_examples_algebra_SERIES
    + """
>>> a.divmod(b, fill_value=0)
(a    1.0
 b    inf
 c    inf
 d    0.0
 e    NaN
 dtype: float64,
 a    0.0
 b    NaN
 c    NaN
 d    0.0
 e    NaN
 dtype: float64)
"""
)

_mod_example_SERIES = (
    _common_examples_algebra_SERIES
    + """
>>> a.mod(b, fill_value=0)
a    0.0
b    NaN
c    NaN
d    0.0
e    NaN
dtype: float64
"""
)
_pow_example_SERIES = (
    _common_examples_algebra_SERIES
    + """
>>> a.pow(b, fill_value=0)
a    1.0
b    1.0
c    1.0
d    0.0
e    NaN
dtype: float64
"""
)

_ne_example_SERIES = (
    _common_examples_algebra_SERIES
    + """
>>> a.ne(b, fill_value=0)
a    False
b     True
c     True
d     True
e     True
dtype: bool
"""
)

_eq_example_SERIES = (
    _common_examples_algebra_SERIES
    + """
>>> a.eq(b, fill_value=0)
a     True
b    False
c    False
d    False
e    False
dtype: bool
"""
)

_lt_example_SERIES = (
    _common_examples_comparison_SERIES
    + """
>>> a.lt(b, fill_value=0)
a    False
b    False
c     True
d    False
e    False
f     True
dtype: bool
"""
)

_le_example_SERIES = (
    _common_examples_comparison_SERIES
    + """
>>> a.le(b, fill_value=0)
a    False
b     True
c     True
d    False
e    False
f     True
dtype: bool
"""
)

_gt_example_SERIES = (
    _common_examples_comparison_SERIES
    + """
>>> a.gt(b, fill_value=0)
a     True
b    False
c    False
d    False
e     True
f    False
dtype: bool
"""
)

_ge_example_SERIES = (
    _common_examples_comparison_SERIES
    + """
>>> a.ge(b, fill_value=0)
a     True
b     True
c    False
d    False
e     True
f    False
dtype: bool
"""
)

_returns_series = """Series\n    The result of the operation."""

_returns_tuple = """2-Tuple of Series\n    The result of the operation."""

_op_descriptions: dict[str, dict[str, str | None]] = {
    # Arithmetic Operators
    "add": {
        "op": "+",
        "desc": "Addition",
        "reverse": "radd",
        "series_examples": _add_example_SERIES,
        "series_returns": _returns_series,
    },
    "sub": {
        "op": "-",
        "desc": "Subtraction",
        "reverse": "rsub",
        "series_examples": _sub_example_SERIES,
        "series_returns": _returns_series,
    },
    "mul": {
        "op": "*",
        "desc": "Multiplication",
        "reverse": "rmul",
        "series_examples": _mul_example_SERIES,
        "series_returns": _returns_series,
        "df_examples": None,
    },
    "mod": {
        "op": "%",
        "desc": "Modulo",
        "reverse": "rmod",
        "series_examples": _mod_example_SERIES,
        "series_returns": _returns_series,
    },
    "pow": {
        "op": "**",
        "desc": "Exponential power",
        "reverse": "rpow",
        "series_examples": _pow_example_SERIES,
        "series_returns": _returns_series,
        "df_examples": None,
    },
    "truediv": {
        "op": "/",
        "desc": "Floating division",
        "reverse": "rtruediv",
        "series_examples": _div_example_SERIES,
        "series_returns": _returns_series,
        "df_examples": None,
    },
    "floordiv": {
        "op": "//",
        "desc": "Integer division",
        "reverse": "rfloordiv",
        "series_examples": _floordiv_example_SERIES,
        "series_returns": _returns_series,
        "df_examples": None,
    },
    "divmod": {
        "op": "divmod",
        "desc": "Integer division and modulo",
        "reverse": "rdivmod",
        "series_examples": _divmod_example_SERIES,
        "series_returns": _returns_tuple,
        "df_examples": None,
    },
    # Comparison Operators
    "eq": {
        "op": "==",
        "desc": "Equal to",
        "reverse": None,
        "series_examples": _eq_example_SERIES,
        "series_returns": _returns_series,
    },
    "ne": {
        "op": "!=",
        "desc": "Not equal to",
        "reverse": None,
        "series_examples": _ne_example_SERIES,
        "series_returns": _returns_series,
    },
    "lt": {
        "op": "<",
        "desc": "Less than",
        "reverse": None,
        "series_examples": _lt_example_SERIES,
        "series_returns": _returns_series,
    },
    "le": {
        "op": "<=",
        "desc": "Less than or equal to",
        "reverse": None,
        "series_examples": _le_example_SERIES,
        "series_returns": _returns_series,
    },
    "gt": {
        "op": ">",
        "desc": "Greater than",
        "reverse": None,
        "series_examples": _gt_example_SERIES,
        "series_returns": _returns_series,
    },
    "ge": {
        "op": ">=",
        "desc": "Greater than or equal to",
        "reverse": None,
        "series_examples": _ge_example_SERIES,
        "series_returns": _returns_series,
    },
}

_py_num_ref = """see
    `Python documentation
    <https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types>`_
    for more details"""
_op_names = list(_op_descriptions.keys())
for key in _op_names:
    reverse_op = _op_descriptions[key]["reverse"]
    if reverse_op is not None:
        _op_descriptions[reverse_op] = _op_descriptions[key].copy()
        _op_descriptions[reverse_op]["reverse"] = key
        _op_descriptions[key][
            "see_also_desc"
        ] = f"Reverse of the {_op_descriptions[key]['desc']} operator, {_py_num_ref}"
        _op_descriptions[reverse_op][
            "see_also_desc"
        ] = f"Element-wise {_op_descriptions[key]['desc']}, {_py_num_ref}"

_flex_doc_SERIES = """
Return {desc} of series and other, element-wise (binary operator `{op_name}`).

Equivalent to ``{equiv}``, but with support to substitute a fill_value for
missing data in either one of the inputs.

Parameters
----------
other : Series or scalar value
level : int or name
    Broadcast across a level, matching Index values on the
    passed MultiIndex level.
fill_value : None or float value, default None (NaN)
    Fill existing missing (NaN) values, and any new element needed for
    successful Series alignment, with this value before computation.
    If data in both corresponding Series locations is missing
    the result of filling (at that location) will be missing.
axis : {{0 or 'index'}}
    Unused. Parameter needed for compatibility with DataFrame.

Returns
-------
{series_returns}
"""

_see_also_reverse_SERIES = """
See Also
--------
Series.{reverse} : {see_also_desc}.
"""

_flex_doc_FRAME = """
Get {desc} of dataframe and other, element-wise (binary operator `{op_name}`).

Equivalent to ``{equiv}``, but with support to substitute a fill_value
for missing data in one of the inputs. With reverse version, `{reverse}`.

Among flexible wrappers (`add`, `sub`, `mul`, `div`, `floordiv`, `mod`, `pow`) to
arithmetic operators: `+`, `-`, `*`, `/`, `//`, `%`, `**`.

Parameters
----------
other : scalar, sequence, Series, dict or DataFrame
    Any single or multiple element data structure, or list-like object.
axis : {{0 or 'index', 1 or 'columns'}}
    Whether to compare by the index (0 or 'index') or columns.
    (1 or 'columns'). For Series input, axis to match Series index on.
level : int or label
    Broadcast across a level, matching Index values on the
    passed MultiIndex level.
fill_value : float or None, default None
    Fill existing missing (NaN) values, and any new element needed for
    successful DataFrame alignment, with this value before computation.
    If data in both corresponding DataFrame locations is missing
    the result will be missing.

Returns
-------
DataFrame
    Result of the arithmetic operation.

See Also
--------
DataFrame.add : Add DataFrames.
DataFrame.sub : Subtract DataFrames.
DataFrame.mul : Multiply DataFrames.
DataFrame.div : Divide DataFrames (float division).
DataFrame.truediv : Divide DataFrames (float division).
DataFrame.floordiv : Divide DataFrames (integer division).
DataFrame.mod : Calculate modulo (remainder after division).
DataFrame.pow : Calculate exponential power.

Notes
-----
Mismatched indices will be unioned together.

Examples
--------
>>> df = pd.DataFrame({{'angles': [0, 3, 4],
...                    'degrees': [360, 180, 360]}},
...                   index=['circle', 'triangle', 'rectangle'])
>>> df
           angles  degrees
circle          0      360
triangle        3      180
rectangle       4      360

Add a scalar with operator version which return the same
results.

>>> df + 1
           angles  degrees
circle          1      361
triangle        4      181
rectangle       5      361

>>> df.add(1)
           angles  degrees
circle          1      361
triangle        4      181
rectangle       5      361

Divide by constant with reverse version.

>>> df.div(10)
           angles  degrees
circle        0.0     36.0
triangle      0.3     18.0
rectangle     0.4     36.0

>>> df.rdiv(10)
             angles   degrees
circle          inf  0.027778
triangle   3.333333  0.055556
rectangle  2.500000  0.027778

Subtract a list and Series by axis with operator version.

>>> df - [1, 2]
           angles  degrees
circle         -1      358
triangle        2      178
rectangle       3      358

>>> df.sub([1, 2], axis='columns')
           angles  degrees
circle         -1      358
triangle        2      178
rectangle       3      358

>>> df.sub(pd.Series([1, 1, 1], index=['circle', 'triangle', 'rectangle']),
...        axis='index')
           angles  degrees
circle         -1      359
triangle        2      179
rectangle       3      359

Multiply a dictionary by axis.

>>> df.mul({{'angles': 0, 'degrees': 2}})
            angles  degrees
circle           0      720
triangle         0      360
rectangle        0      720

>>> df.mul({{'circle': 0, 'triangle': 2, 'rectangle': 3}}, axis='index')
            angles  degrees
circle           0        0
triangle         6      360
rectangle       12     1080

Multiply a DataFrame of different shape with operator version.

>>> other = pd.DataFrame({{'angles': [0, 3, 4]}},
...                      index=['circle', 'triangle', 'rectangle'])
>>> other
           angles
circle          0
triangle        3
rectangle       4

>>> df * other
           angles  degrees
circle          0      NaN
triangle        9      NaN
rectangle      16      NaN

>>> df.mul(other, fill_value=0)
           angles  degrees
circle          0      0.0
triangle        9      0.0
rectangle      16      0.0

Divide by a MultiIndex by level.

>>> df_multindex = pd.DataFrame({{'angles': [0, 3, 4, 4, 5, 6],
...                              'degrees': [360, 180, 360, 360, 540, 720]}},
...                             index=[['A', 'A', 'A', 'B', 'B', 'B'],
...                                    ['circle', 'triangle', 'rectangle',
...                                     'square', 'pentagon', 'hexagon']])
>>> df_multindex
             angles  degrees
A circle          0      360
  triangle        3      180
  rectangle       4      360
B square          4      360
  pentagon        5      540
  hexagon         6      720

>>> df.div(df_multindex, level=1, fill_value=0)
             angles  degrees
A circle        NaN      1.0
  triangle      1.0      1.0
  rectangle     1.0      1.0
B square        0.0      0.0
  pentagon      0.0      0.0
  hexagon       0.0      0.0
"""

_flex_comp_doc_FRAME = """
Get {desc} of dataframe and other, element-wise (binary operator `{op_name}`).

Among flexible wrappers (`eq`, `ne`, `le`, `lt`, `ge`, `gt`) to comparison
operators.

Equivalent to `==`, `!=`, `<=`, `<`, `>=`, `>` with support to choose axis
(rows or columns) and level for comparison.

Parameters
----------
other : scalar, sequence, Series, or DataFrame
    Any single or multiple element data structure, or list-like object.
axis : {{0 or 'index', 1 or 'columns'}}, default 'columns'
    Whether to compare by the index (0 or 'index') or columns
    (1 or 'columns').
level : int or label
    Broadcast across a level, matching Index values on the passed
    MultiIndex level.

Returns
-------
DataFrame of bool
    Result of the comparison.

See Also
--------
DataFrame.eq : Compare DataFrames for equality elementwise.
DataFrame.ne : Compare DataFrames for inequality elementwise.
DataFrame.le : Compare DataFrames for less than inequality
    or equality elementwise.
DataFrame.lt : Compare DataFrames for strictly less than
    inequality elementwise.
DataFrame.ge : Compare DataFrames for greater than inequality
    or equality elementwise.
DataFrame.gt : Compare DataFrames for strictly greater than
    inequality elementwise.

Notes
-----
Mismatched indices will be unioned together.
`NaN` values are considered different (i.e. `NaN` != `NaN`).

Examples
--------
>>> df = pd.DataFrame({{'cost': [250, 150, 100],
...                    'revenue': [100, 250, 300]}},
...                   index=['A', 'B', 'C'])
>>> df
   cost  revenue
A   250      100
B   150      250
C   100      300

Comparison with a scalar, using either the operator or method:

>>> df == 100
    cost  revenue
A  False     True
B  False    False
C   True    False

>>> df.eq(100)
    cost  revenue
A  False     True
B  False    False
C   True    False

When `other` is a :class:`Series`, the columns of a DataFrame are aligned
with the index of `other` and broadcast:

>>> df != pd.Series([100, 250], index=["cost", "revenue"])
    cost  revenue
A   True     True
B   True    False
C  False     True

Use the method to control the broadcast axis:

>>> df.ne(pd.Series([100, 300], index=["A", "D"]), axis='index')
   cost  revenue
A  True    False
B  True     True
C  True     True
D  True     True

When comparing to an arbitrary sequence, the number of columns must
match the number elements in `other`:

>>> df == [250, 100]
    cost  revenue
A   True     True
B  False    False
C  False    False

Use the method to control the axis:

>>> df.eq([250, 250, 100], axis='index')
    cost  revenue
A   True    False
B  False     True
C   True    False

Compare to a DataFrame of different shape.

>>> other = pd.DataFrame({{'revenue': [300, 250, 100, 150]}},
...                      index=['A', 'B', 'C', 'D'])
>>> other
   revenue
A      300
B      250
C      100
D      150

>>> df.gt(other)
    cost  revenue
A  False    False
B  False    False
C  False     True
D  False    False

Compare to a MultiIndex by level.

>>> df_multindex = pd.DataFrame({{'cost': [250, 150, 100, 150, 300, 220],
...                              'revenue': [100, 250, 300, 200, 175, 225]}},
...                             index=[['Q1', 'Q1', 'Q1', 'Q2', 'Q2', 'Q2'],
...                                    ['A', 'B', 'C', 'A', 'B', 'C']])
>>> df_multindex
      cost  revenue
Q1 A   250      100
   B   150      250
   C   100      300
Q2 A   150      200
   B   300      175
   C   220      225

>>> df.le(df_multindex, level=1)
       cost  revenue
Q1 A   True     True
   B   True     True
   C   True     True
Q2 A  False     True
   B   True    False
   C   True    False
"""
<|endoftext|>"""
Functions for defining unary operations.
"""
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)

from pandas.core.dtypes.generic import ABCExtensionArray

if TYPE_CHECKING:
    from pandas._typing import ArrayLike


def should_extension_dispatch(left: ArrayLike, right: Any) -> bool:
    """
    Identify cases where Series operation should dispatch to ExtensionArray method.

    Parameters
    ----------
    left : np.ndarray or ExtensionArray
    right : object

    Returns
    -------
    bool
    """
    return isinstance(left, ABCExtensionArray) or isinstance(right, ABCExtensionArray)
<|endoftext|>"""
Boilerplate functions used in defining binary operations.
"""
from __future__ import annotations

from functools import wraps
from typing import (
    TYPE_CHECKING,
    Callable,
)

from pandas._libs.lib import item_from_zerodim
from pandas._libs.missing import is_matching_na

from pandas.core.dtypes.generic import (
    ABCIndex,
    ABCSeries,
)

if TYPE_CHECKING:
    from pandas._typing import F


def unpack_zerodim_and_defer(name: str) -> Callable[[F], F]:
    """
    Boilerplate for pandas conventions in arithmetic and comparison methods.

    Parameters
    ----------
    name : str

    Returns
    -------
    decorator
    """

    def wrapper(method: F) -> F:
        return _unpack_zerodim_and_defer(method, name)

    return wrapper


def _unpack_zerodim_and_defer(method, name: str):
    """
    Boilerplate for pandas conventions in arithmetic and comparison methods.

    Ensure method returns NotImplemented when operating against "senior"
    classes.  Ensure zero-dimensional ndarrays are always unpacked.

    Parameters
    ----------
    method : binary method
    name : str

    Returns
    -------
    method
    """
    stripped_name = name.removeprefix("__").removesuffix("__")
    is_cmp = stripped_name in {"eq", "ne", "lt", "le", "gt", "ge"}

    @wraps(method)
    def new_method(self, other):
        if is_cmp and isinstance(self, ABCIndex) and isinstance(other, ABCSeries):
            # For comparison ops, Index does *not* defer to Series
            pass
        else:
            prio = getattr(other, "__pandas_priority__", None)
            if prio is not None:
                if prio > self.__pandas_priority__:
                    # e.g. other is DataFrame while self is Index/Series/EA
                    return NotImplemented

        other = item_from_zerodim(other)

        return method(self, other)

    return new_method


def get_op_result_name(left, right):
    """
    Find the appropriate name to pin to an operation result.  This result
    should always be either an Index or a Series.

    Parameters
    ----------
    left : {Series, Index}
    right : object

    Returns
    -------
    name : object
        Usually a string
    """
    if isinstance(right, (ABCSeries, ABCIndex)):
        name = _maybe_match_name(left, right)
    else:
        name = left.name
    return name


def _maybe_match_name(a, b):
    """
    Try to find a name to attach to the result of an operation between
    a and b.  If only one of these has a `name` attribute, return that
    name.  Otherwise return a consensus name if they match or None if
    they have different names.

    Parameters
    ----------
    a : object
    b : object

    Returns
    -------
    name : str or None

    See Also
    --------
    pandas.core.common.consensus_name_attr
    """
    a_has = hasattr(a, "name")
    b_has = hasattr(b, "name")
    if a_has and b_has:
        try:
            if a.name == b.name:
                return a.name
            elif is_matching_na(a.name, b.name):
                # e.g. both are np.nan
                return a.name
            else:
                return None
        except TypeError:
            # pd.NA
            if is_matching_na(a.name, b.name):
                return a.name
            return None
        except ValueError:
            # e.g. np.int64(1) vs (np.int64(1), np.int64(2))
            return None
    elif a_has:
        return a.name
    elif b_has:
        return b.name
    return None
<|endoftext|>"""
Functions for arithmetic and comparison operations on NumPy arrays and
ExtensionArrays.
"""
from __future__ import annotations

import datetime
from functools import partial
import operator
from typing import (
    TYPE_CHECKING,
    Any,
)
import warnings

import numpy as np

from pandas._libs import (
    NaT,
    Timedelta,
    Timestamp,
    lib,
    ops as libops,
)
from pandas._libs.tslibs import (
    BaseOffset,
    get_supported_dtype,
    is_supported_dtype,
    is_unitless,
)
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.cast import (
    construct_1d_object_array_from_listlike,
    find_common_type,
)
from pandas.core.dtypes.common import (
    ensure_object,
    is_bool_dtype,
    is_list_like,
    is_numeric_v_string_like,
    is_object_dtype,
    is_scalar,
)
from pandas.core.dtypes.generic import (
    ABCExtensionArray,
    ABCIndex,
    ABCSeries,
)
from pandas.core.dtypes.missing import (
    isna,
    notna,
)

from pandas.core import roperator
from pandas.core.computation import expressions
from pandas.core.construction import ensure_wrapped_if_datetimelike
from pandas.core.ops import missing
from pandas.core.ops.dispatch import should_extension_dispatch
from pandas.core.ops.invalid import invalid_comparison

if TYPE_CHECKING:
    from pandas._typing import (
        ArrayLike,
        Shape,
    )

# -----------------------------------------------------------------------------
# Masking NA values and fallbacks for operations numpy does not support


def fill_binop(left, right, fill_value):
    """
    If a non-None fill_value is given, replace null entries in left and right
    with this value, but only in positions where _one_ of left/right is null,
    not both.

    Parameters
    ----------
    left : array-like
    right : array-like
    fill_value : object

    Returns
    -------
    left : array-like
    right : array-like

    Notes
    -----
    Makes copies if fill_value is not None and NAs are present.
    """
    if fill_value is not None:
        left_mask = isna(left)
        right_mask = isna(right)

        # one but not both
        mask = left_mask ^ right_mask

        if left_mask.any():
            # Avoid making a copy if we can
            left = left.copy()
            left[left_mask & mask] = fill_value

        if right_mask.any():
            # Avoid making a copy if we can
            right = right.copy()
            right[right_mask & mask] = fill_value

    return left, right


def comp_method_OBJECT_ARRAY(op, x, y):
    if isinstance(y, list):
        # e.g. test_tuple_categories
        y = construct_1d_object_array_from_listlike(y)

    if isinstance(y, (np.ndarray, ABCSeries, ABCIndex)):
        if not is_object_dtype(y.dtype):
            y = y.astype(np.object_)

        if isinstance(y, (ABCSeries, ABCIndex)):
            y = y._values

        if x.shape != y.shape:
            raise ValueError("Shapes must match", x.shape, y.shape)
        result = libops.vec_compare(x.ravel(), y.ravel(), op)
    else:
        result = libops.scalar_compare(x.ravel(), y, op)
    return result.reshape(x.shape)


def _masked_arith_op(x: np.ndarray, y, op):
    """
    If the given arithmetic operation fails, attempt it again on
    only the non-null elements of the input array(s).

    Parameters
    ----------
    x : np.ndarray
    y : np.ndarray, Series, Index
    op : binary operator
    """
    # For Series `x` is 1D so ravel() is a no-op; calling it anyway makes
    # the logic valid for both Series and DataFrame ops.
    xrav = x.ravel()

    if isinstance(y, np.ndarray):
        dtype = find_common_type([x.dtype, y.dtype])
        result = np.empty(x.size, dtype=dtype)

        if len(x) != len(y):
            raise ValueError(x.shape, y.shape)
        ymask = notna(y)

        # NB: ravel() is only safe since y is ndarray; for e.g. PeriodIndex
        #  we would get int64 dtype, see GH#19956
        yrav = y.ravel()
        mask = notna(xrav) & ymask.ravel()

        # See GH#5284, GH#5035, GH#19448 for historical reference
        if mask.any():
            result[mask] = op(xrav[mask], yrav[mask])

    else:
        if not is_scalar(y):
            raise TypeError(
                f"Cannot broadcast np.ndarray with operand of type { type(y) }"
            )

        # mask is only meaningful for x
        result = np.empty(x.size, dtype=x.dtype)
        mask = notna(xrav)

        # 1 ** np.nan is 1. So we have to unmask those.
        if op is pow:
            mask = np.where(x == 1, False, mask)
        elif op is roperator.rpow:
            mask = np.where(y == 1, False, mask)

        if mask.any():
            result[mask] = op(xrav[mask], y)

    np.putmask(result, ~mask, np.nan)
    result = result.reshape(x.shape)  # 2D compat
    return result


def _na_arithmetic_op(left: np.ndarray, right, op, is_cmp: bool = False):
    """
    Return the result of evaluating op on the passed in values.

    If native types are not compatible, try coercion to object dtype.

    Parameters
    ----------
    left : np.ndarray
    right : np.ndarray or scalar
        Excludes DataFrame, Series, Index, ExtensionArray.
    is_cmp : bool, default False
        If this a comparison operation.

    Returns
    -------
    array-like

    Raises
    ------
    TypeError : invalid operation
    """
    if isinstance(right, str):
        # can never use numexpr
        func = op
    else:
        func = partial(expressions.evaluate, op)

    try:
        result = func(left, right)
    except TypeError:
        if not is_cmp and (
            left.dtype == object or getattr(right, "dtype", None) == object
        ):
            # For object dtype, fallback to a masked operation (only operating
            #  on the non-missing values)
            # Don't do this for comparisons, as that will handle complex numbers
            #  incorrectly, see GH#32047
            result = _masked_arith_op(left, right, op)
        else:
            raise

    if is_cmp and (is_scalar(result) or result is NotImplemented):
        # numpy returned a scalar instead of operating element-wise
        # e.g. numeric array vs str
        # TODO: can remove this after dropping some future numpy version?
        return invalid_comparison(left, right, op)

    return missing.dispatch_fill_zeros(op, left, right, result)


def arithmetic_op(left: ArrayLike, right: Any, op):
    """
    Evaluate an arithmetic operation `+`, `-`, `*`, `/`, `//`, `%`, `**`, ...

    Note: the caller is responsible for ensuring that numpy warnings are
    suppressed (with np.errstate(all="ignore")) if needed.

    Parameters
    ----------
    left : np.ndarray or ExtensionArray
    right : object
        Cannot be a DataFrame or Index.  Series is *not* excluded.
    op : {operator.add, operator.sub, ...}
        Or one of the reversed variants from roperator.

    Returns
    -------
    ndarray or ExtensionArray
        Or a 2-tuple of these in the case of divmod or rdivmod.
    """
    # NB: We assume that extract_array and ensure_wrapped_if_datetimelike
    #  have already been called on `left` and `right`,
    #  and `maybe_prepare_scalar_for_op` has already been called on `right`
    # We need to special-case datetime64/timedelta64 dtypes (e.g. because numpy
    # casts integer dtypes to timedelta64 when operating with timedelta64 - GH#22390)

    if (
        should_extension_dispatch(left, right)
        or isinstance(right, (Timedelta, BaseOffset, Timestamp))
        or right is NaT
    ):
        # Timedelta/Timestamp and other custom scalars are included in the check
        # because numexpr will fail on it, see GH#31457
        res_values = op(left, right)
    else:
        # TODO we should handle EAs consistently and move this check before the if/else
        # (https://github.com/pandas-dev/pandas/issues/41165)
        # error: Argument 2 to "_bool_arith_check" has incompatible type
        # "Union[ExtensionArray, ndarray[Any, Any]]"; expected "ndarray[Any, Any]"
        _bool_arith_check(op, left, right)  # type: ignore[arg-type]

        # error: Argument 1 to "_na_arithmetic_op" has incompatible type
        # "Union[ExtensionArray, ndarray[Any, Any]]"; expected "ndarray[Any, Any]"
        res_values = _na_arithmetic_op(left, right, op)  # type: ignore[arg-type]

    return res_values


def comparison_op(left: ArrayLike, right: Any, op) -> ArrayLike:
    """
    Evaluate a comparison operation `=`, `!=`, `>=`, `>`, `<=`, or `<`.

    Note: the caller is responsible for ensuring that numpy warnings are
    suppressed (with np.errstate(all="ignore")) if needed.

    Parameters
    ----------
    left : np.ndarray or ExtensionArray
    right : object
        Cannot be a DataFrame, Series, or Index.
    op : {operator.eq, operator.ne, operator.gt, operator.ge, operator.lt, operator.le}

    Returns
    -------
    ndarray or ExtensionArray
    """
    # NB: We assume extract_array has already been called on left and right
    lvalues = ensure_wrapped_if_datetimelike(left)
    rvalues = ensure_wrapped_if_datetimelike(right)

    rvalues = lib.item_from_zerodim(rvalues)
    if isinstance(rvalues, list):
        # We don't catch tuple here bc we may be comparing e.g. MultiIndex
        #  to a tuple that represents a single entry, see test_compare_tuple_strs
        rvalues = np.asarray(rvalues)

    if isinstance(rvalues, (np.ndarray, ABCExtensionArray)):
        # TODO: make this treatment consistent across ops and classes.
        #  We are not catching all listlikes here (e.g. frozenset, tuple)
        #  The ambiguous case is object-dtype.  See GH#27803
        if len(lvalues) != len(rvalues):
            raise ValueError(
                "Lengths must match to compare", lvalues.shape, rvalues.shape
            )

    if should_extension_dispatch(lvalues, rvalues) or (
        (isinstance(rvalues, (Timedelta, BaseOffset, Timestamp)) or right is NaT)
        and lvalues.dtype != object
    ):
        # Call the method on lvalues
        res_values = op(lvalues, rvalues)

    elif is_scalar(rvalues) and isna(rvalues):  # TODO: but not pd.NA?
        # numpy does not like comparisons vs None
        if op is operator.ne:
            res_values = np.ones(lvalues.shape, dtype=bool)
        else:
            res_values = np.zeros(lvalues.shape, dtype=bool)

    elif is_numeric_v_string_like(lvalues, rvalues):
        # GH#36377 going through the numexpr path would incorrectly raise
        return invalid_comparison(lvalues, rvalues, op)

    elif lvalues.dtype == object or isinstance(rvalues, str):
        res_values = comp_method_OBJECT_ARRAY(op, lvalues, rvalues)

    else:
        res_values = _na_arithmetic_op(lvalues, rvalues, op, is_cmp=True)

    return res_values


def na_logical_op(x: np.ndarray, y, op):
    try:
        # For exposition, write:
        #  yarr = isinstance(y, np.ndarray)
        #  yint = is_integer(y) or (yarr and y.dtype.kind == "i")
        #  ybool = is_bool(y) or (yarr and y.dtype.kind == "b")
        #  xint = x.dtype.kind == "i"
        #  xbool = x.dtype.kind == "b"
        # Then Cases where this goes through without raising include:
        #  (xint or xbool) and (yint or bool)
        result = op(x, y)
    except TypeError:
        if isinstance(y, np.ndarray):
            # bool-bool dtype operations should be OK, should not get here
            assert not (x.dtype.kind == "b" and y.dtype.kind == "b")
            x = ensure_object(x)
            y = ensure_object(y)
            result = libops.vec_binop(x.ravel(), y.ravel(), op)
        else:
            # let null fall thru
            assert lib.is_scalar(y)
            if not isna(y):
                y = bool(y)
            try:
                result = libops.scalar_binop(x, y, op)
            except (
                TypeError,
                ValueError,
                AttributeError,
                OverflowError,
                NotImplementedError,
            ) as err:
                typ = type(y).__name__
                raise TypeError(
                    f"Cannot perform '{op.__name__}' with a dtyped [{x.dtype}] array "
                    f"and scalar of type [{typ}]"
                ) from err

    return result.reshape(x.shape)


def logical_op(left: ArrayLike, right: Any, op) -> ArrayLike:
    """
    Evaluate a logical operation `|`, `&`, or `^`.

    Parameters
    ----------
    left : np.ndarray or ExtensionArray
    right : object
        Cannot be a DataFrame, Series, or Index.
    op : {operator.and_, operator.or_, operator.xor}
        Or one of the reversed variants from roperator.

    Returns
    -------
    ndarray or ExtensionArray
    """

    def fill_bool(x, left=None):
        # if `left` is specifically not-boolean, we do not cast to bool
        if x.dtype.kind in "cfO":
            # dtypes that can hold NA
            mask = isna(x)
            if mask.any():
                x = x.astype(object)
                x[mask] = False

        if left is None or left.dtype.kind == "b":
            x = x.astype(bool)
        return x

    right = lib.item_from_zerodim(right)
    if is_list_like(right) and not hasattr(right, "dtype"):
        # e.g. list, tuple
        warnings.warn(
            "Logical ops (and, or, xor) between Pandas objects and dtype-less "
            "sequences (e.g. list, tuple) are deprecated and will raise in a "
            "future version. Wrap the object in a Series, Index, or np.array "
            "before operating instead.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
        right = construct_1d_object_array_from_listlike(right)

    # NB: We assume extract_array has already been called on left and right
    lvalues = ensure_wrapped_if_datetimelike(left)
    rvalues = right

    if should_extension_dispatch(lvalues, rvalues):
        # Call the method on lvalues
        res_values = op(lvalues, rvalues)

    else:
        if isinstance(rvalues, np.ndarray):
            is_other_int_dtype = rvalues.dtype.kind in "iu"
            if not is_other_int_dtype:
                rvalues = fill_bool(rvalues, lvalues)

        else:
            # i.e. scalar
            is_other_int_dtype = lib.is_integer(rvalues)

        res_values = na_logical_op(lvalues, rvalues, op)

        # For int vs int `^`, `|`, `&` are bitwise operators and return
        #   integer dtypes.  Otherwise these are boolean ops
        if not (left.dtype.kind in "iu" and is_other_int_dtype):
            res_values = fill_bool(res_values)

    return res_values


def get_array_op(op):
    """
    Return a binary array operation corresponding to the given operator op.

    Parameters
    ----------
    op : function
        Binary operator from operator or roperator module.

    Returns
    -------
    functools.partial
    """
    if isinstance(op, partial):
        # We get here via dispatch_to_series in DataFrame case
        # e.g. test_rolling_consistency_var_debiasing_factors
        return op

    op_name = op.__name__.strip("_").lstrip("r")
    if op_name == "arith_op":
        # Reached via DataFrame._combine_frame i.e. flex methods
        # e.g. test_df_add_flex_filled_mixed_dtypes
        return op

    if op_name in {"eq", "ne", "lt", "le", "gt", "ge"}:
        return partial(comparison_op, op=op)
    elif op_name in {"and", "or", "xor", "rand", "ror", "rxor"}:
        return partial(logical_op, op=op)
    elif op_name in {
        "add",
        "sub",
        "mul",
        "truediv",
        "floordiv",
        "mod",
        "divmod",
        "pow",
    }:
        return partial(arithmetic_op, op=op)
    else:
        raise NotImplementedError(op_name)


def maybe_prepare_scalar_for_op(obj, shape: Shape):
    """
    Cast non-pandas objects to pandas types to unify behavior of arithmetic
    and comparison operations.

    Parameters
    ----------
    obj: object
    shape : tuple[int]

    Returns
    -------
    out : object

    Notes
    -----
    Be careful to call this *after* determining the `name` attribute to be
    attached to the result of the arithmetic operation.
    """
    if type(obj) is datetime.timedelta:
        # GH#22390  cast up to Timedelta to rely on Timedelta
        # implementation; otherwise operation against numeric-dtype
        # raises TypeError
        return Timedelta(obj)
    elif type(obj) is datetime.datetime:
        # cast up to Timestamp to rely on Timestamp implementation, see Timedelta above
        return Timestamp(obj)
    elif isinstance(obj, np.datetime64):
        # GH#28080 numpy casts integer-dtype to datetime64 when doing
        #  array[int] + datetime64, which we do not allow
        if isna(obj):
            from pandas.core.arrays import DatetimeArray

            # Avoid possible ambiguities with pd.NaT
            # GH 52295
            if is_unitless(obj.dtype):
                obj = obj.astype("datetime64[ns]")
            elif not is_supported_dtype(obj.dtype):
                new_dtype = get_supported_dtype(obj.dtype)
                obj = obj.astype(new_dtype)
            right = np.broadcast_to(obj, shape)
            return DatetimeArray._simple_new(right, dtype=right.dtype)

        return Timestamp(obj)

    elif isinstance(obj, np.timedelta64):
        if isna(obj):
            from pandas.core.arrays import TimedeltaArray

            # wrapping timedelta64("NaT") in Timedelta returns NaT,
            #  which would incorrectly be treated as a datetime-NaT, so
            #  we broadcast and wrap in a TimedeltaArray
            # GH 52295
            if is_unitless(obj.dtype):
                obj = obj.astype("timedelta64[ns]")
            elif not is_supported_dtype(obj.dtype):
                new_dtype = get_supported_dtype(obj.dtype)
                obj = obj.astype(new_dtype)
            right = np.broadcast_to(obj, shape)
            return TimedeltaArray._simple_new(right, dtype=right.dtype)

        # In particular non-nanosecond timedelta64 needs to be cast to
        #  nanoseconds, or else we get undesired behavior like
        #  np.timedelta64(3, 'D') / 2 == np.timedelta64(1, 'D')
        return Timedelta(obj)

    # We want NumPy numeric scalars to behave like Python scalars
    # post NEP 50
    elif isinstance(obj, np.integer):
        return int(obj)

    elif isinstance(obj, np.floating):
        return float(obj)

    return obj


_BOOL_OP_NOT_ALLOWED = {
    operator.truediv,
    roperator.rtruediv,
    operator.floordiv,
    roperator.rfloordiv,
    operator.pow,
    roperator.rpow,
}


def _bool_arith_check(op, a: np.ndarray, b):
    """
    In contrast to numpy, pandas raises an error for certain operations
    with booleans.
    """
    if op in _BOOL_OP_NOT_ALLOWED:
        if a.dtype.kind == "b" and (is_bool_dtype(b) or lib.is_bool(b)):
            op_name = op.__name__.strip("_").lstrip("r")
            raise NotImplementedError(
                f"operator '{op_name}' not implemented for bool dtypes"
            )
<|endoftext|>"""
Arithmetic operations for PandasObjects

This is not a public API.
"""
from __future__ import annotations

from pandas.core.ops.array_ops import (
    arithmetic_op,
    comp_method_OBJECT_ARRAY,
    comparison_op,
    fill_binop,
    get_array_op,
    logical_op,
    maybe_prepare_scalar_for_op,
)
from pandas.core.ops.common import (
    get_op_result_name,
    unpack_zerodim_and_defer,
)
from pandas.core.ops.docstrings import make_flex_doc
from pandas.core.ops.invalid import invalid_comparison
from pandas.core.ops.mask_ops import (
    kleene_and,
    kleene_or,
    kleene_xor,
)
from pandas.core.roperator import (
    radd,
    rand_,
    rdiv,
    rdivmod,
    rfloordiv,
    rmod,
    rmul,
    ror_,
    rpow,
    rsub,
    rtruediv,
    rxor,
)

# -----------------------------------------------------------------------------
# constants
ARITHMETIC_BINOPS: set[str] = {
    "add",
    "sub",
    "mul",
    "pow",
    "mod",
    "floordiv",
    "truediv",
    "divmod",
    "radd",
    "rsub",
    "rmul",
    "rpow",
    "rmod",
    "rfloordiv",
    "rtruediv",
    "rdivmod",
}


__all__ = [
    "ARITHMETIC_BINOPS",
    "arithmetic_op",
    "comparison_op",
    "comp_method_OBJECT_ARRAY",
    "invalid_comparison",
    "fill_binop",
    "kleene_and",
    "kleene_or",
    "kleene_xor",
    "logical_op",
    "make_flex_doc",
    "radd",
    "rand_",
    "rdiv",
    "rdivmod",
    "rfloordiv",
    "rmod",
    "rmul",
    "ror_",
    "rpow",
    "rsub",
    "rtruediv",
    "rxor",
    "unpack_zerodim_and_defer",
    "get_op_result_name",
    "maybe_prepare_scalar_for_op",
    "get_array_op",
]
<|endoftext|>"""
Ops for masked arrays.
"""
from __future__ import annotations

import numpy as np

from pandas._libs import (
    lib,
    missing as libmissing,
)


def kleene_or(
    left: bool | np.ndarray | libmissing.NAType,
    right: bool | np.ndarray | libmissing.NAType,
    left_mask: np.ndarray | None,
    right_mask: np.ndarray | None,
):
    """
    Boolean ``or`` using Kleene logic.

    Values are NA where we have ``NA | NA`` or ``NA | False``.
    ``NA | True`` is considered True.

    Parameters
    ----------
    left, right : ndarray, NA, or bool
        The values of the array.
    left_mask, right_mask : ndarray, optional
        The masks. Only one of these may be None, which implies that
        the associated `left` or `right` value is a scalar.

    Returns
    -------
    result, mask: ndarray[bool]
        The result of the logical or, and the new mask.
    """
    # To reduce the number of cases, we ensure that `left` & `left_mask`
    # always come from an array, not a scalar. This is safe, since
    # A | B == B | A
    if left_mask is None:
        return kleene_or(right, left, right_mask, left_mask)

    if not isinstance(left, np.ndarray):
        raise TypeError("Either `left` or `right` need to be a np.ndarray.")

    raise_for_nan(right, method="or")

    if right is libmissing.NA:
        result = left.copy()
    else:
        result = left | right

    if right_mask is not None:
        # output is unknown where (False & NA), (NA & False), (NA & NA)
        left_false = ~(left | left_mask)
        right_false = ~(right | right_mask)
        mask = (
            (left_false & right_mask)
            | (right_false & left_mask)
            | (left_mask & right_mask)
        )
    else:
        if right is True:
            mask = np.zeros_like(left_mask)
        elif right is libmissing.NA:
            mask = (~left & ~left_mask) | left_mask
        else:
            # False
            mask = left_mask.copy()

    return result, mask


def kleene_xor(
    left: bool | np.ndarray | libmissing.NAType,
    right: bool | np.ndarray | libmissing.NAType,
    left_mask: np.ndarray | None,
    right_mask: np.ndarray | None,
):
    """
    Boolean ``xor`` using Kleene logic.

    This is the same as ``or``, with the following adjustments

    * True, True -> False
    * True, NA   -> NA

    Parameters
    ----------
    left, right : ndarray, NA, or bool
        The values of the array.
    left_mask, right_mask : ndarray, optional
        The masks. Only one of these may be None, which implies that
        the associated `left` or `right` value is a scalar.

    Returns
    -------
    result, mask: ndarray[bool]
        The result of the logical xor, and the new mask.
    """
    # To reduce the number of cases, we ensure that `left` & `left_mask`
    # always come from an array, not a scalar. This is safe, since
    # A ^ B == B ^ A
    if left_mask is None:
        return kleene_xor(right, left, right_mask, left_mask)

    if not isinstance(left, np.ndarray):
        raise TypeError("Either `left` or `right` need to be a np.ndarray.")

    raise_for_nan(right, method="xor")
    if right is libmissing.NA:
        result = np.zeros_like(left)
    else:
        result = left ^ right

    if right_mask is None:
        if right is libmissing.NA:
            mask = np.ones_like(left_mask)
        else:
            mask = left_mask.copy()
    else:
        mask = left_mask | right_mask

    return result, mask


def kleene_and(
    left: bool | libmissing.NAType | np.ndarray,
    right: bool | libmissing.NAType | np.ndarray,
    left_mask: np.ndarray | None,
    right_mask: np.ndarray | None,
):
    """
    Boolean ``and`` using Kleene logic.

    Values are ``NA`` for ``NA & NA`` or ``True & NA``.

    Parameters
    ----------
    left, right : ndarray, NA, or bool
        The values of the array.
    left_mask, right_mask : ndarray, optional
        The masks. Only one of these may be None, which implies that
        the associated `left` or `right` value is a scalar.

    Returns
    -------
    result, mask: ndarray[bool]
        The result of the logical xor, and the new mask.
    """
    # To reduce the number of cases, we ensure that `left` & `left_mask`
    # always come from an array, not a scalar. This is safe, since
    # A & B == B & A
    if left_mask is None:
        return kleene_and(right, left, right_mask, left_mask)

    if not isinstance(left, np.ndarray):
        raise TypeError("Either `left` or `right` need to be a np.ndarray.")
    raise_for_nan(right, method="and")

    if right is libmissing.NA:
        result = np.zeros_like(left)
    else:
        result = left & right

    if right_mask is None:
        # Scalar `right`
        if right is libmissing.NA:
            mask = (left & ~left_mask) | left_mask

        else:
            mask = left_mask.copy()
            if right is False:
                # unmask everything
                mask[:] = False
    else:
        # unmask where either left or right is False
        left_false = ~(left | left_mask)
        right_false = ~(right | right_mask)
        mask = (left_mask & ~right_false) | (right_mask & ~left_false)

    return result, mask


def raise_for_nan(value, method: str) -> None:
    if lib.is_float(value) and np.isnan(value):
        raise ValueError(f"Cannot perform logical '{method}' with floating NaN")
<|endoftext|>"""
Missing data handling for arithmetic operations.

In particular, pandas conventions regarding division by zero differ
from numpy in the following ways:
    1) np.array([-1, 0, 1], dtype=dtype1) // np.array([0, 0, 0], dtype=dtype2)
       gives [nan, nan, nan] for most dtype combinations, and [0, 0, 0] for
       the remaining pairs
       (the remaining being dtype1==dtype2==intN and dtype==dtype2==uintN).

       pandas convention is to return [-inf, nan, inf] for all dtype
       combinations.

       Note: the numpy behavior described here is py3-specific.

    2) np.array([-1, 0, 1], dtype=dtype1) % np.array([0, 0, 0], dtype=dtype2)
       gives precisely the same results as the // operation.

       pandas convention is to return [nan, nan, nan] for all dtype
       combinations.

    3) divmod behavior consistent with 1) and 2).
"""
from __future__ import annotations

import operator

import numpy as np

from pandas.core import roperator


def _fill_zeros(result: np.ndarray, x, y):
    """
    If this is a reversed op, then flip x,y

    If we have an integer value (or array in y)
    and we have 0's, fill them with np.nan,
    return the result.

    Mask the nan's from x.
    """
    if result.dtype.kind == "f":
        return result

    is_variable_type = hasattr(y, "dtype")
    is_scalar_type = not isinstance(y, np.ndarray)

    if not is_variable_type and not is_scalar_type:
        # e.g. test_series_ops_name_retention with mod we get here with list/tuple
        return result

    if is_scalar_type:
        y = np.array(y)

    if y.dtype.kind in "iu":
        ymask = y == 0
        if ymask.any():
            # GH#7325, mask and nans must be broadcastable
            mask = ymask & ~np.isnan(result)

            # GH#9308 doing ravel on result and mask can improve putmask perf,
            #  but can also make unwanted copies.
            result = result.astype("float64", copy=False)

            np.putmask(result, mask, np.nan)

    return result


def mask_zero_div_zero(x, y, result: np.ndarray) -> np.ndarray:
    """
    Set results of  0 // 0 to np.nan, regardless of the dtypes
    of the numerator or the denominator.

    Parameters
    ----------
    x : ndarray
    y : ndarray
    result : ndarray

    Returns
    -------
    ndarray
        The filled result.

    Examples
    --------
    >>> x = np.array([1, 0, -1], dtype=np.int64)
    >>> x
    array([ 1,  0, -1])
    >>> y = 0       # int 0; numpy behavior is different with float
    >>> result = x // y
    >>> result      # raw numpy result does not fill division by zero
    array([0, 0, 0])
    >>> mask_zero_div_zero(x, y, result)
    array([ inf,  nan, -inf])
    """

    if not hasattr(y, "dtype"):
        # e.g. scalar, tuple
        y = np.array(y)
    if not hasattr(x, "dtype"):
        # e.g scalar, tuple
        x = np.array(x)

    zmask = y == 0

    if zmask.any():
        # Flip sign if necessary for -0.0
        zneg_mask = zmask & np.signbit(y)
        zpos_mask = zmask & ~zneg_mask

        x_lt0 = x < 0
        x_gt0 = x > 0
        nan_mask = zmask & (x == 0)
        neginf_mask = (zpos_mask & x_lt0) | (zneg_mask & x_gt0)
        posinf_mask = (zpos_mask & x_gt0) | (zneg_mask & x_lt0)

        if nan_mask.any() or neginf_mask.any() or posinf_mask.any():
            # Fill negative/0 with -inf, positive/0 with +inf, 0/0 with NaN
            result = result.astype("float64", copy=False)

            result[nan_mask] = np.nan
            result[posinf_mask] = np.inf
            result[neginf_mask] = -np.inf

    return result


def dispatch_fill_zeros(op, left, right, result):
    """
    Call _fill_zeros with the appropriate fill value depending on the operation,
    with special logic for divmod and rdivmod.

    Parameters
    ----------
    op : function (operator.add, operator.div, ...)
    left : object (np.ndarray for non-reversed ops)
        We have excluded ExtensionArrays here
    right : object (np.ndarray for reversed ops)
        We have excluded ExtensionArrays here
    result : ndarray

    Returns
    -------
    result : np.ndarray

    Notes
    -----
    For divmod and rdivmod, the `result` parameter and returned `result`
    is a 2-tuple of ndarray objects.
    """
    if op is divmod:
        result = (
            mask_zero_div_zero(left, right, result[0]),
            _fill_zeros(result[1], left, right),
        )
    elif op is roperator.rdivmod:
        result = (
            mask_zero_div_zero(right, left, result[0]),
            _fill_zeros(result[1], right, left),
        )
    elif op is operator.floordiv:
        # Note: no need to do this for truediv; in py3 numpy behaves the way
        #  we want.
        result = mask_zero_div_zero(left, right, result)
    elif op is roperator.rfloordiv:
        # Note: no need to do this for rtruediv; in py3 numpy behaves the way
        #  we want.
        result = mask_zero_div_zero(right, left, result)
    elif op is operator.mod:
        result = _fill_zeros(result, left, right)
    elif op is roperator.rmod:
        result = _fill_zeros(result, right, left)
    return result
<|endoftext|>"""
Templates for invalid operations.
"""
from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pandas._typing import npt


def invalid_comparison(left, right, op) -> npt.NDArray[np.bool_]:
    """
    If a comparison has mismatched types and is not necessarily meaningful,
    follow python3 conventions by:

        - returning all-False for equality
        - returning all-True for inequality
        - raising TypeError otherwise

    Parameters
    ----------
    left : array-like
    right : scalar, array-like
    op : operator.{eq, ne, lt, le, gt}

    Raises
    ------
    TypeError : on inequality comparisons
    """
    if op is operator.eq:
        res_values = np.zeros(left.shape, dtype=bool)
    elif op is operator.ne:
        res_values = np.ones(left.shape, dtype=bool)
    else:
        typ = type(right).__name__
        raise TypeError(f"Invalid comparison between dtype={left.dtype} and {typ}")
    return res_values


def make_invalid_op(name: str):
    """
    Return a binary method that always raises a TypeError.

    Parameters
    ----------
    name : str

    Returns
    -------
    invalid_op : function
    """

    def invalid_op(self, other=None):
        typ = type(self).__name__
        raise TypeError(f"cannot perform {name} with this index type: {typ}")

