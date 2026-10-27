        # probably business daily, but need to confirm
        first_weekday = self.index[0].weekday()
        shifts = np.diff(self.i8values)
        ppd = periods_per_day(self._creso)
        shifts = np.floor_divide(shifts, ppd)
        weekdays = np.mod(first_weekday + np.cumsum(shifts), 7)

        return bool(
            np.all(
                ((weekdays == 0) & (shifts == 3))
                | ((weekdays > 0) & (weekdays <= 4) & (shifts == 1))
            )
        )

    def _get_wom_rule(self) -> str | None:
        weekdays = unique(self.index.weekday)
        if len(weekdays) > 1:
            return None

        week_of_months = unique((self.index.day - 1) // 7)
        # Only attempt to infer up to WOM-4. See #9425
        week_of_months = week_of_months[week_of_months < 4]
        if len(week_of_months) == 0 or len(week_of_months) > 1:
            return None

        # get which week
        week = week_of_months[0] + 1
        wd = int_to_weekday[weekdays[0]]

        return f"WOM-{week}{wd}"


class _TimedeltaFrequencyInferer(_FrequencyInferer):
    def _infer_daily_rule(self):
        if self.is_unique:
            return self._get_daily_rule()


def _is_multiple(us, mult: int) -> bool:
    return us % mult == 0


def _maybe_add_count(base: str, count: float) -> str:
    if count != 1:
        assert count == int(count)
        count = int(count)
        return f"{count}{base}"
    else:
        return base


# ----------------------------------------------------------------------
# Frequency comparison


def is_subperiod(source, target) -> bool:
    """
    Returns True if downsampling is possible between source and target
    frequencies

    Parameters
    ----------
    source : str or DateOffset
        Frequency converting from
    target : str or DateOffset
        Frequency converting to

    Returns
    -------
    bool
    """
    if target is None or source is None:
        return False
    source = _maybe_coerce_freq(source)
    target = _maybe_coerce_freq(target)

    if _is_annual(target):
        if _is_quarterly(source):
            return _quarter_months_conform(
                get_rule_month(source), get_rule_month(target)
            )
        return source in {"D", "C", "B", "M", "h", "min", "s", "ms", "us", "ns"}
    elif _is_quarterly(target):
        return source in {"D", "C", "B", "M", "h", "min", "s", "ms", "us", "ns"}
    elif _is_monthly(target):
        return source in {"D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
    elif _is_weekly(target):
        return source in {target, "D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
    elif target == "B":
        return source in {"B", "h", "min", "s", "ms", "us", "ns"}
    elif target == "C":
        return source in {"C", "h", "min", "s", "ms", "us", "ns"}
    elif target == "D":
        return source in {"D", "h", "min", "s", "ms", "us", "ns"}
    elif target == "h":
        return source in {"h", "min", "s", "ms", "us", "ns"}
    elif target == "min":
        return source in {"min", "s", "ms", "us", "ns"}
    elif target == "s":
        return source in {"s", "ms", "us", "ns"}
    elif target == "ms":
        return source in {"ms", "us", "ns"}
    elif target == "us":
        return source in {"us", "ns"}
    elif target == "ns":
        return source in {"ns"}
    else:
        return False


def is_superperiod(source, target) -> bool:
    """
    Returns True if upsampling is possible between source and target
    frequencies

    Parameters
    ----------
    source : str or DateOffset
        Frequency converting from
    target : str or DateOffset
        Frequency converting to

    Returns
    -------
    bool
    """
    if target is None or source is None:
        return False
    source = _maybe_coerce_freq(source)
    target = _maybe_coerce_freq(target)

    if _is_annual(source):
        if _is_annual(target):
            return get_rule_month(source) == get_rule_month(target)

        if _is_quarterly(target):
            smonth = get_rule_month(source)
            tmonth = get_rule_month(target)
            return _quarter_months_conform(smonth, tmonth)
        return target in {"D", "C", "B", "M", "h", "min", "s", "ms", "us", "ns"}
    elif _is_quarterly(source):
        return target in {"D", "C", "B", "M", "h", "min", "s", "ms", "us", "ns"}
    elif _is_monthly(source):
        return target in {"D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
    elif _is_weekly(source):
        return target in {source, "D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
    elif source == "B":
        return target in {"D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
    elif source == "C":
        return target in {"D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
    elif source == "D":
        return target in {"D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
    elif source == "h":
        return target in {"h", "min", "s", "ms", "us", "ns"}
    elif source == "min":
        return target in {"min", "s", "ms", "us", "ns"}
    elif source == "s":
        return target in {"s", "ms", "us", "ns"}
    elif source == "ms":
        return target in {"ms", "us", "ns"}
    elif source == "us":
        return target in {"us", "ns"}
    elif source == "ns":
        return target in {"ns"}
    else:
        return False


def _maybe_coerce_freq(code) -> str:
    """we might need to coerce a code to a rule_code
    and uppercase it

    Parameters
    ----------
    source : str or DateOffset
        Frequency converting from

    Returns
    -------
    str
    """
    assert code is not None
    if isinstance(code, DateOffset):
        code = freq_to_period_freqstr(1, code.name)
    if code in {"h", "min", "s", "ms", "us", "ns"}:
        return code
    else:
        return code.upper()


def _quarter_months_conform(source: str, target: str) -> bool:
    snum = MONTH_NUMBERS[source]
    tnum = MONTH_NUMBERS[target]
    return snum % 3 == tnum % 3


def _is_annual(rule: str) -> bool:
    rule = rule.upper()
    return rule == "Y" or rule.startswith("Y-")


def _is_quarterly(rule: str) -> bool:
    rule = rule.upper()
    return rule == "Q" or rule.startswith(("Q-", "BQ"))


def _is_monthly(rule: str) -> bool:
    rule = rule.upper()
    return rule in ("M", "BM")


def _is_weekly(rule: str) -> bool:
    rule = rule.upper()
    return rule == "W" or rule.startswith("W-")


__all__ = [
    "Day",
    "get_period_alias",
    "infer_freq",
    "is_subperiod",
    "is_superperiod",
    "to_offset",
]
<|endoftext|># ruff: noqa: TCH004
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # import modules that have public classes/functions:
    from pandas.tseries import (
        frequencies,
        offsets,
    )

    # and mark only those modules as public
    __all__ = ["frequencies", "offsets"]
<|endoftext|>from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
)
import warnings

from dateutil.relativedelta import (
    FR,
    MO,
    SA,
    SU,
    TH,
    TU,
    WE,
)
import numpy as np

from pandas.errors import PerformanceWarning

from pandas import (
    DateOffset,
    DatetimeIndex,
    Series,
    Timestamp,
    concat,
    date_range,
)

from pandas.tseries.offsets import (
    Day,
    Easter,
)


def next_monday(dt: datetime) -> datetime:
    """
    If holiday falls on Saturday, use following Monday instead;
    if holiday falls on Sunday, use Monday instead
    """
    if dt.weekday() == 5:
        return dt + timedelta(2)
    elif dt.weekday() == 6:
        return dt + timedelta(1)
    return dt


def next_monday_or_tuesday(dt: datetime) -> datetime:
    """
    For second holiday of two adjacent ones!
    If holiday falls on Saturday, use following Monday instead;
    if holiday falls on Sunday or Monday, use following Tuesday instead
    (because Monday is already taken by adjacent holiday on the day before)
    """
    dow = dt.weekday()
    if dow in (5, 6):
        return dt + timedelta(2)
    if dow == 0:
        return dt + timedelta(1)
    return dt


def previous_friday(dt: datetime) -> datetime:
    """
    If holiday falls on Saturday or Sunday, use previous Friday instead.
    """
    if dt.weekday() == 5:
        return dt - timedelta(1)
    elif dt.weekday() == 6:
        return dt - timedelta(2)
    return dt


def sunday_to_monday(dt: datetime) -> datetime:
    """
    If holiday falls on Sunday, use day thereafter (Monday) instead.
    """
    if dt.weekday() == 6:
        return dt + timedelta(1)
    return dt


def weekend_to_monday(dt: datetime) -> datetime:
    """
    If holiday falls on Sunday or Saturday,
    use day thereafter (Monday) instead.
    Needed for holidays such as Christmas observation in Europe
    """
    if dt.weekday() == 6:
        return dt + timedelta(1)
    elif dt.weekday() == 5:
        return dt + timedelta(2)
    return dt


def nearest_workday(dt: datetime) -> datetime:
    """
    If holiday falls on Saturday, use day before (Friday) instead;
    if holiday falls on Sunday, use day thereafter (Monday) instead.
    """
    if dt.weekday() == 5:
        return dt - timedelta(1)
    elif dt.weekday() == 6:
        return dt + timedelta(1)
    return dt


def next_workday(dt: datetime) -> datetime:
    """
    returns next weekday used for observances
    """
    dt += timedelta(days=1)
    while dt.weekday() > 4:
        # Mon-Fri are 0-4
        dt += timedelta(days=1)
    return dt


def previous_workday(dt: datetime) -> datetime:
    """
    returns previous weekday used for observances
    """
    dt -= timedelta(days=1)
    while dt.weekday() > 4:
        # Mon-Fri are 0-4
        dt -= timedelta(days=1)
    return dt


def before_nearest_workday(dt: datetime) -> datetime:
    """
    returns previous workday after nearest workday
    """
    return previous_workday(nearest_workday(dt))


def after_nearest_workday(dt: datetime) -> datetime:
    """
    returns next workday after nearest workday
    needed for Boxing day or multiple holidays in a series
    """
    return next_workday(nearest_workday(dt))


class Holiday:
    """
    Class that defines a holiday with start/end dates and rules
    for observance.
    """

    start_date: Timestamp | None
    end_date: Timestamp | None
    days_of_week: tuple[int, ...] | None

    def __init__(
        self,
        name: str,
        year=None,
        month=None,
        day=None,
        offset=None,
        observance=None,
        start_date=None,
        end_date=None,
        days_of_week=None,
    ) -> None:
        """
        Parameters
        ----------
        name : str
            Name of the holiday , defaults to class name
        offset : array of pandas.tseries.offsets or
                class from pandas.tseries.offsets
            computes offset from date
        observance: function
            computes when holiday is given a pandas Timestamp
        days_of_week:
            provide a tuple of days e.g  (0,1,2,3,) for Monday Through Thursday
            Monday=0,..,Sunday=6

        Examples
        --------
        >>> from dateutil.relativedelta import MO

        >>> USMemorialDay = pd.tseries.holiday.Holiday(
        ...     "Memorial Day", month=5, day=31, offset=pd.DateOffset(weekday=MO(-1))
        ... )
        >>> USMemorialDay
        Holiday: Memorial Day (month=5, day=31, offset=<DateOffset: weekday=MO(-1)>)

        >>> USLaborDay = pd.tseries.holiday.Holiday(
        ...     "Labor Day", month=9, day=1, offset=pd.DateOffset(weekday=MO(1))
        ... )
        >>> USLaborDay
        Holiday: Labor Day (month=9, day=1, offset=<DateOffset: weekday=MO(+1)>)

        >>> July3rd = pd.tseries.holiday.Holiday("July 3rd", month=7, day=3)
        >>> July3rd
        Holiday: July 3rd (month=7, day=3, )

        >>> NewYears = pd.tseries.holiday.Holiday(
        ...     "New Years Day", month=1,  day=1,
        ...      observance=pd.tseries.holiday.nearest_workday
        ... )
        >>> NewYears  # doctest: +SKIP
        Holiday: New Years Day (
            month=1, day=1, observance=<function nearest_workday at 0x66545e9bc440>
        )

        >>> July3rd = pd.tseries.holiday.Holiday(
        ...     "July 3rd", month=7, day=3,
        ...     days_of_week=(0, 1, 2, 3)
        ... )
        >>> July3rd
        Holiday: July 3rd (month=7, day=3, )
        """
        if offset is not None and observance is not None:
            raise NotImplementedError("Cannot use both offset and observance.")

        self.name = name
        self.year = year
        self.month = month
        self.day = day
        self.offset = offset
        self.start_date = (
            Timestamp(start_date) if start_date is not None else start_date
        )
        self.end_date = Timestamp(end_date) if end_date is not None else end_date
        self.observance = observance
        assert days_of_week is None or type(days_of_week) == tuple
        self.days_of_week = days_of_week

    def __repr__(self) -> str:
        info = ""
        if self.year is not None:
            info += f"year={self.year}, "
        info += f"month={self.month}, day={self.day}, "

        if self.offset is not None:
            info += f"offset={self.offset}"

        if self.observance is not None:
            info += f"observance={self.observance}"

        repr = f"Holiday: {self.name} ({info})"
        return repr

    def dates(
        self, start_date, end_date, return_name: bool = False
    ) -> Series | DatetimeIndex:
        """
        Calculate holidays observed between start date and end date

        Parameters
        ----------
        start_date : starting date, datetime-like, optional
        end_date : ending date, datetime-like, optional
        return_name : bool, optional, default=False
            If True, return a series that has dates and holiday names.
            False will only return dates.

        Returns
        -------
        Series or DatetimeIndex
            Series if return_name is True
        """
        start_date = Timestamp(start_date)
        end_date = Timestamp(end_date)

        filter_start_date = start_date
        filter_end_date = end_date

        if self.year is not None:
            dt = Timestamp(datetime(self.year, self.month, self.day))
            dti = DatetimeIndex([dt])
            if return_name:
                return Series(self.name, index=dti)
            else:
                return dti

        dates = self._reference_dates(start_date, end_date)
        holiday_dates = self._apply_rule(dates)
        if self.days_of_week is not None:
            holiday_dates = holiday_dates[
                np.isin(
                    # error: "DatetimeIndex" has no attribute "dayofweek"
                    holiday_dates.dayofweek,  # type: ignore[attr-defined]
                    self.days_of_week,
                ).ravel()
            ]

        if self.start_date is not None:
            filter_start_date = max(
                self.start_date.tz_localize(filter_start_date.tz), filter_start_date
            )
        if self.end_date is not None:
            filter_end_date = min(
                self.end_date.tz_localize(filter_end_date.tz), filter_end_date
            )
        holiday_dates = holiday_dates[
            (holiday_dates >= filter_start_date) & (holiday_dates <= filter_end_date)
        ]
        if return_name:
            return Series(self.name, index=holiday_dates)
        return holiday_dates

    def _reference_dates(
        self, start_date: Timestamp, end_date: Timestamp
    ) -> DatetimeIndex:
        """
        Get reference dates for the holiday.

        Return reference dates for the holiday also returning the year
        prior to the start_date and year following the end_date.  This ensures
        that any offsets to be applied will yield the holidays within
        the passed in dates.
        """
        if self.start_date is not None:
            start_date = self.start_date.tz_localize(start_date.tz)

        if self.end_date is not None:
            end_date = self.end_date.tz_localize(start_date.tz)

        year_offset = DateOffset(years=1)
        reference_start_date = Timestamp(
            datetime(start_date.year - 1, self.month, self.day)
        )

        reference_end_date = Timestamp(
            datetime(end_date.year + 1, self.month, self.day)
        )
        # Don't process unnecessary holidays
        dates = date_range(
            start=reference_start_date,
            end=reference_end_date,
            freq=year_offset,
            tz=start_date.tz,
        )

        return dates

    def _apply_rule(self, dates: DatetimeIndex) -> DatetimeIndex:
        """
        Apply the given offset/observance to a DatetimeIndex of dates.

        Parameters
        ----------
        dates : DatetimeIndex
            Dates to apply the given offset/observance rule

        Returns
        -------
        Dates with rules applied
        """
        if dates.empty:
            return dates.copy()

        if self.observance is not None:
            return dates.map(lambda d: self.observance(d))

        if self.offset is not None:
            if not isinstance(self.offset, list):
                offsets = [self.offset]
            else:
                offsets = self.offset
            for offset in offsets:
                # if we are adding a non-vectorized value
                # ignore the PerformanceWarnings:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", PerformanceWarning)
                    dates += offset
        return dates


holiday_calendars = {}


def register(cls) -> None:
    try:
        name = cls.name
    except AttributeError:
        name = cls.__name__
    holiday_calendars[name] = cls


def get_calendar(name: str):
    """
    Return an instance of a calendar based on its name.

    Parameters
    ----------
    name : str
        Calendar name to return an instance of
    """
    return holiday_calendars[name]()


class HolidayCalendarMetaClass(type):
    def __new__(cls, clsname: str, bases, attrs):
        calendar_class = super().__new__(cls, clsname, bases, attrs)
        register(calendar_class)
        return calendar_class


class AbstractHolidayCalendar(metaclass=HolidayCalendarMetaClass):
    """
    Abstract interface to create holidays following certain rules.
    """

    rules: list[Holiday] = []
    start_date = Timestamp(datetime(1970, 1, 1))
    end_date = Timestamp(datetime(2200, 12, 31))
    _cache = None

    def __init__(self, name: str = "", rules=None) -> None:
        """
        Initializes holiday object with a given set a rules.  Normally
        classes just have the rules defined within them.

        Parameters
        ----------
        name : str
            Name of the holiday calendar, defaults to class name
        rules : array of Holiday objects
            A set of rules used to create the holidays.
        """
        super().__init__()
        if not name:
            name = type(self).__name__
        self.name = name

        if rules is not None:
            self.rules = rules

    def rule_from_name(self, name: str):
        for rule in self.rules:
            if rule.name == name:
                return rule

        return None

    def holidays(self, start=None, end=None, return_name: bool = False):
        """
        Returns a curve with holidays between start_date and end_date

        Parameters
        ----------
        start : starting date, datetime-like, optional
        end : ending date, datetime-like, optional
        return_name : bool, optional
            If True, return a series that has dates and holiday names.
            False will only return a DatetimeIndex of dates.

        Returns
        -------
            DatetimeIndex of holidays
        """
        if self.rules is None:
            raise Exception(
                f"Holiday Calendar {self.name} does not have any rules specified"
            )

        if start is None:
            start = AbstractHolidayCalendar.start_date

        if end is None:
            end = AbstractHolidayCalendar.end_date

        start = Timestamp(start)
        end = Timestamp(end)

        # If we don't have a cache or the dates are outside the prior cache, we
        # get them again
        if self._cache is None or start < self._cache[0] or end > self._cache[1]:
            pre_holidays = [
                rule.dates(start, end, return_name=True) for rule in self.rules
            ]
            if pre_holidays:
                # error: Argument 1 to "concat" has incompatible type
                # "List[Union[Series, DatetimeIndex]]"; expected
                # "Union[Iterable[DataFrame], Mapping[<nothing>, DataFrame]]"
                holidays = concat(pre_holidays)  # type: ignore[arg-type]
            else:
                # error: Incompatible types in assignment (expression has type
                # "Series", variable has type "DataFrame")
                holidays = Series(
                    index=DatetimeIndex([]), dtype=object
                )  # type: ignore[assignment]

            self._cache = (start, end, holidays.sort_index())

        holidays = self._cache[2]
        holidays = holidays[start:end]

        if return_name:
            return holidays
        else:
            return holidays.index

    @staticmethod
    def merge_class(base, other):
        """
        Merge holiday calendars together. The base calendar
        will take precedence to other. The merge will be done
        based on each holiday's name.

        Parameters
        ----------
        base : AbstractHolidayCalendar
          instance/subclass or array of Holiday objects
        other : AbstractHolidayCalendar
          instance/subclass or array of Holiday objects
        """
        try:
            other = other.rules
        except AttributeError:
            pass

        if not isinstance(other, list):
            other = [other]
        other_holidays = {holiday.name: holiday for holiday in other}

        try:
            base = base.rules
        except AttributeError:
            pass

        if not isinstance(base, list):
            base = [base]
        base_holidays = {holiday.name: holiday for holiday in base}

        other_holidays.update(base_holidays)
        return list(other_holidays.values())

    def merge(self, other, inplace: bool = False):
        """
        Merge holiday calendars together.  The caller's class
        rules take precedence.  The merge will be done
        based on each holiday's name.

        Parameters
        ----------
        other : holiday calendar
        inplace : bool (default=False)
            If True set rule_table to holidays, else return array of Holidays
        """
        holidays = self.merge_class(self, other)
        if inplace:
            self.rules = holidays
        else:
            return holidays


USMemorialDay = Holiday(
    "Memorial Day", month=5, day=31, offset=DateOffset(weekday=MO(-1))
)
USLaborDay = Holiday("Labor Day", month=9, day=1, offset=DateOffset(weekday=MO(1)))
USColumbusDay = Holiday(
    "Columbus Day", month=10, day=1, offset=DateOffset(weekday=MO(2))
)
USThanksgivingDay = Holiday(
    "Thanksgiving Day", month=11, day=1, offset=DateOffset(weekday=TH(4))
)
USMartinLutherKingJr = Holiday(
    "Birthday of Martin Luther King, Jr.",
    start_date=datetime(1986, 1, 1),
    month=1,
    day=1,
    offset=DateOffset(weekday=MO(3)),
)
USPresidentsDay = Holiday(
    "Washington's Birthday", month=2, day=1, offset=DateOffset(weekday=MO(3))
)
GoodFriday = Holiday("Good Friday", month=1, day=1, offset=[Easter(), Day(-2)])

EasterMonday = Holiday("Easter Monday", month=1, day=1, offset=[Easter(), Day(1)])


class USFederalHolidayCalendar(AbstractHolidayCalendar):
    """
    US Federal Government Holiday Calendar based on rules specified by:
    https://www.opm.gov/policy-data-oversight/pay-leave/federal-holidays/
    """

    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        USMemorialDay,
        Holiday(
            "Juneteenth National Independence Day",
            month=6,
            day=19,
            start_date="2021-06-18",
            observance=nearest_workday,
        ),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USColumbusDay,
        Holiday("Veterans Day", month=11, day=11, observance=nearest_workday),
        USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
    ]


def HolidayCalendarFactory(name: str, base, other, base_class=AbstractHolidayCalendar):
    rules = AbstractHolidayCalendar.merge_class(base, other)
    calendar_class = type(name, (base_class,), {"rules": rules, "name": name})
    return calendar_class


__all__ = [
    "after_nearest_workday",
    "before_nearest_workday",
    "FR",
    "get_calendar",
    "HolidayCalendarFactory",
    "MO",
    "nearest_workday",
    "next_monday",
    "next_monday_or_tuesday",
    "next_workday",
    "previous_friday",
    "previous_workday",
    "register",
    "SA",
    "SU",
    "sunday_to_monday",
    "TH",
    "TU",
    "WE",
    "weekend_to_monday",
]
<|endoftext|>"""
Timeseries API
"""

from pandas._libs.tslibs.parsing import guess_datetime_format

from pandas.tseries import offsets
from pandas.tseries.frequencies import infer_freq

__all__ = ["infer_freq", "offsets", "guess_datetime_format"]
<|endoftext|>from __future__ import annotations

import json
import locale
import os
import platform
import struct
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pandas._typing import JSONSerializable

from pandas.compat._optional import (
    VERSIONS,
    get_version,
    import_optional_dependency,
)


def _get_commit_hash() -> str | None:
    """
    Use vendored versioneer code to get git hash, which handles
    git worktree correctly.
    """
    try:
        from pandas._version_meson import (  # pyright: ignore [reportMissingImports]
            __git_version__,
        )

        return __git_version__
    except ImportError:
        from pandas._version import get_versions

        versions = get_versions()
        return versions["full-revisionid"]


def _get_sys_info() -> dict[str, JSONSerializable]:
    """
    Returns system information as a JSON serializable dictionary.
    """
    uname_result = platform.uname()
    language_code, encoding = locale.getlocale()
    return {
        "commit": _get_commit_hash(),
        "python": platform.python_version(),
        "python-bits": struct.calcsize("P") * 8,
        "OS": uname_result.system,
        "OS-release": uname_result.release,
        "Version": uname_result.version,
        "machine": uname_result.machine,
        "processor": uname_result.processor,
        "byteorder": sys.byteorder,
        "LC_ALL": os.environ.get("LC_ALL"),
        "LANG": os.environ.get("LANG"),
        "LOCALE": {"language-code": language_code, "encoding": encoding},
    }


def _get_dependency_info() -> dict[str, JSONSerializable]:
    """
    Returns dependency information as a JSON serializable dictionary.
    """
    deps = [
        "pandas",
        # required
        "numpy",
        "pytz",
        "dateutil",
        # install / build,
        "pip",
        "Cython",
        # docs
        "sphinx",
        # Other, not imported.
        "IPython",
    ]
    # Optional dependencies
    deps.extend(list(VERSIONS))

    result: dict[str, JSONSerializable] = {}
    for modname in deps:
        try:
            mod = import_optional_dependency(modname, errors="ignore")
        except Exception:
            # Dependency conflicts may cause a non ImportError
            result[modname] = "N/A"
        else:
            result[modname] = get_version(mod) if mod else None
    return result


def show_versions(as_json: str | bool = False) -> None:
    """
    Provide useful information, important for bug reports.

    It comprises info about hosting operation system, pandas version,
    and versions of other installed relative packages.

    Parameters
    ----------
    as_json : str or bool, default False
        * If False, outputs info in a human readable form to the console.
        * If str, it will be considered as a path to a file.
          Info will be written to that file in JSON format.
        * If True, outputs info in JSON format to the console.

    Examples
    --------
    >>> pd.show_versions()  # doctest: +SKIP
    Your output may look something like this:
    INSTALLED VERSIONS
    ------------------
    commit           : 37ea63d540fd27274cad6585082c91b1283f963d
    python           : 3.10.6.final.0
    python-bits      : 64
    OS               : Linux
    OS-release       : 5.10.102.1-microsoft-standard-WSL2
    Version          : #1 SMP Wed Mar 2 00:30:59 UTC 2022
    machine          : x86_64
    processor        : x86_64
    byteorder        : little
    LC_ALL           : None
    LANG             : en_GB.UTF-8
    LOCALE           : en_GB.UTF-8
    pandas           : 2.0.1
    numpy            : 1.24.3
    ...
    """
    sys_info = _get_sys_info()
    deps = _get_dependency_info()

    if as_json:
        j = {"system": sys_info, "dependencies": deps}

        if as_json is True:
            sys.stdout.writelines(json.dumps(j, indent=2))
        else:
            assert isinstance(as_json, str)  # needed for mypy
            with open(as_json, "w", encoding="utf-8") as f:
                json.dump(j, f, indent=2)

    else:
        assert isinstance(sys_info["LOCALE"], dict)  # needed for mypy
        language_code = sys_info["LOCALE"]["language-code"]
        encoding = sys_info["LOCALE"]["encoding"]
        sys_info["LOCALE"] = f"{language_code}.{encoding}"

        maxlen = max(len(x) for x in deps)
        print("\nINSTALLED VERSIONS")
        print("------------------")
        for k, v in sys_info.items():
            print(f"{k:<{maxlen}}: {v}")
        print("")
        for k, v in deps.items():
            print(f"{k:<{maxlen}}: {v}")
<|endoftext|>"""
Module that contains many useful utilities
for validating data or function arguments
"""
from __future__ import annotations

from collections.abc import (
    Iterable,
    Sequence,
)
from typing import (
    TypeVar,
    overload,
)

import numpy as np

from pandas._libs import lib

from pandas.core.dtypes.common import (
    is_bool,
    is_integer,
)

BoolishT = TypeVar("BoolishT", bool, int)
BoolishNoneT = TypeVar("BoolishNoneT", bool, int, None)


def _check_arg_length(fname, args, max_fname_arg_count, compat_args) -> None:
    """
    Checks whether 'args' has length of at most 'compat_args'. Raises
    a TypeError if that is not the case, similar to in Python when a
    function is called with too many arguments.
    """
    if max_fname_arg_count < 0:
        raise ValueError("'max_fname_arg_count' must be non-negative")

    if len(args) > len(compat_args):
        max_arg_count = len(compat_args) + max_fname_arg_count
        actual_arg_count = len(args) + max_fname_arg_count
        argument = "argument" if max_arg_count == 1 else "arguments"

        raise TypeError(
            f"{fname}() takes at most {max_arg_count} {argument} "
            f"({actual_arg_count} given)"
        )


def _check_for_default_values(fname, arg_val_dict, compat_args) -> None:
    """
    Check that the keys in `arg_val_dict` are mapped to their
    default values as specified in `compat_args`.

    Note that this function is to be called only when it has been
    checked that arg_val_dict.keys() is a subset of compat_args
    """
    for key in arg_val_dict:
        # try checking equality directly with '=' operator,
        # as comparison may have been overridden for the left
        # hand object
        try:
            v1 = arg_val_dict[key]
            v2 = compat_args[key]

            # check for None-ness otherwise we could end up
            # comparing a numpy array vs None
            if (v1 is not None and v2 is None) or (v1 is None and v2 is not None):
                match = False
            else:
                match = v1 == v2

            if not is_bool(match):
                raise ValueError("'match' is not a boolean")

        # could not compare them directly, so try comparison
        # using the 'is' operator
        except ValueError:
            match = arg_val_dict[key] is compat_args[key]

        if not match:
            raise ValueError(
                f"the '{key}' parameter is not supported in "
                f"the pandas implementation of {fname}()"
            )


def validate_args(fname, args, max_fname_arg_count, compat_args) -> None:
    """
    Checks whether the length of the `*args` argument passed into a function
    has at most `len(compat_args)` arguments and whether or not all of these
    elements in `args` are set to their default values.

    Parameters
    ----------
    fname : str
        The name of the function being passed the `*args` parameter
    args : tuple
        The `*args` parameter passed into a function
    max_fname_arg_count : int
        The maximum number of arguments that the function `fname`
        can accept, excluding those in `args`. Used for displaying
        appropriate error messages. Must be non-negative.
    compat_args : dict
        A dictionary of keys and their associated default values.
        In order to accommodate buggy behaviour in some versions of `numpy`,
        where a signature displayed keyword arguments but then passed those
        arguments **positionally** internally when calling downstream
        implementations, a dict ensures that the original
        order of the keyword arguments is enforced.

    Raises
    ------
    TypeError
        If `args` contains more values than there are `compat_args`
    ValueError
        If `args` contains values that do not correspond to those
        of the default values specified in `compat_args`
    """
    _check_arg_length(fname, args, max_fname_arg_count, compat_args)

    # We do this so that we can provide a more informative
    # error message about the parameters that we are not
    # supporting in the pandas implementation of 'fname'
    kwargs = dict(zip(compat_args, args))
    _check_for_default_values(fname, kwargs, compat_args)


def _check_for_invalid_keys(fname, kwargs, compat_args) -> None:
    """
    Checks whether 'kwargs' contains any keys that are not
    in 'compat_args' and raises a TypeError if there is one.
    """
    # set(dict) --> set of the dictionary's keys
    diff = set(kwargs) - set(compat_args)

    if diff:
        bad_arg = next(iter(diff))
        raise TypeError(f"{fname}() got an unexpected keyword argument '{bad_arg}'")


def validate_kwargs(fname, kwargs, compat_args) -> None:
    """
    Checks whether parameters passed to the **kwargs argument in a
    function `fname` are valid parameters as specified in `*compat_args`
    and whether or not they are set to their default values.

    Parameters
    ----------
    fname : str
        The name of the function being passed the `**kwargs` parameter
    kwargs : dict
        The `**kwargs` parameter passed into `fname`
    compat_args: dict
        A dictionary of keys that `kwargs` is allowed to have and their
        associated default values

    Raises
    ------
    TypeError if `kwargs` contains keys not in `compat_args`
    ValueError if `kwargs` contains keys in `compat_args` that do not
    map to the default values specified in `compat_args`
    """
    kwds = kwargs.copy()
    _check_for_invalid_keys(fname, kwargs, compat_args)
    _check_for_default_values(fname, kwds, compat_args)


def validate_args_and_kwargs(
    fname, args, kwargs, max_fname_arg_count, compat_args
) -> None:
    """
    Checks whether parameters passed to the *args and **kwargs argument in a
    function `fname` are valid parameters as specified in `*compat_args`
    and whether or not they are set to their default values.

    Parameters
    ----------
    fname: str
        The name of the function being passed the `**kwargs` parameter
    args: tuple
        The `*args` parameter passed into a function
    kwargs: dict
        The `**kwargs` parameter passed into `fname`
    max_fname_arg_count: int
        The minimum number of arguments that the function `fname`
        requires, excluding those in `args`. Used for displaying
        appropriate error messages. Must be non-negative.
    compat_args: dict
        A dictionary of keys that `kwargs` is allowed to
        have and their associated default values.

    Raises
    ------
    TypeError if `args` contains more values than there are
    `compat_args` OR `kwargs` contains keys not in `compat_args`
    ValueError if `args` contains values not at the default value (`None`)
    `kwargs` contains keys in `compat_args` that do not map to the default
    value as specified in `compat_args`

    See Also
    --------
    validate_args : Purely args validation.
    validate_kwargs : Purely kwargs validation.

    """
    # Check that the total number of arguments passed in (i.e.
    # args and kwargs) does not exceed the length of compat_args
    _check_arg_length(
        fname, args + tuple(kwargs.values()), max_fname_arg_count, compat_args
    )

    # Check there is no overlap with the positional and keyword
    # arguments, similar to what is done in actual Python functions
    args_dict = dict(zip(compat_args, args))

    for key in args_dict:
        if key in kwargs:
            raise TypeError(
                f"{fname}() got multiple values for keyword argument '{key}'"
            )

    kwargs.update(args_dict)
    validate_kwargs(fname, kwargs, compat_args)


def validate_bool_kwarg(
    value: BoolishNoneT,
    arg_name: str,
    none_allowed: bool = True,
    int_allowed: bool = False,
) -> BoolishNoneT:
    """
    Ensure that argument passed in arg_name can be interpreted as boolean.

    Parameters
    ----------
    value : bool
        Value to be validated.
    arg_name : str
        Name of the argument. To be reflected in the error message.
    none_allowed : bool, default True
        Whether to consider None to be a valid boolean.
    int_allowed : bool, default False
        Whether to consider integer value to be a valid boolean.

    Returns
    -------
    value
        The same value as input.

    Raises
    ------
    ValueError
        If the value is not a valid boolean.
    """
    good_value = is_bool(value)
    if none_allowed:
        good_value = good_value or (value is None)

    if int_allowed:
        good_value = good_value or isinstance(value, int)

    if not good_value:
        raise ValueError(
            f'For argument "{arg_name}" expected type bool, received '
            f"type {type(value).__name__}."
        )
    return value  # pyright: ignore[reportGeneralTypeIssues]


def validate_fillna_kwargs(value, method, validate_scalar_dict_value: bool = True):
    """
    Validate the keyword arguments to 'fillna'.

    This checks that exactly one of 'value' and 'method' is specified.
    If 'method' is specified, this validates that it's a valid method.

    Parameters
    ----------
    value, method : object
        The 'value' and 'method' keyword arguments for 'fillna'.
    validate_scalar_dict_value : bool, default True
        Whether to validate that 'value' is a scalar or dict. Specifically,
        validate that it is not a list or tuple.

    Returns
    -------
    value, method : object
    """
    from pandas.core.missing import clean_fill_method

    if value is None and method is None:
        raise ValueError("Must specify a fill 'value' or 'method'.")
    if value is None and method is not None:
        method = clean_fill_method(method)

    elif value is not None and method is None:
        if validate_scalar_dict_value and isinstance(value, (list, tuple)):
            raise TypeError(
                '"value" parameter must be a scalar or dict, but '
                f'you passed a "{type(value).__name__}"'
            )

    elif value is not None and method is not None:
        raise ValueError("Cannot specify both 'value' and 'method'.")

    return value, method


def validate_percentile(q: float | Iterable[float]) -> np.ndarray:
    """
    Validate percentiles (used by describe and quantile).

    This function checks if the given float or iterable of floats is a valid percentile
    otherwise raises a ValueError.

    Parameters
    ----------
    q: float or iterable of floats
        A single percentile or an iterable of percentiles.

    Returns
    -------
    ndarray
        An ndarray of the percentiles if valid.

    Raises
    ------
    ValueError if percentiles are not in given interval([0, 1]).
    """
    q_arr = np.asarray(q)
    # Don't change this to an f-string. The string formatting
    # is too expensive for cases where we don't need it.
    msg = "percentiles should all be in the interval [0, 1]"
    if q_arr.ndim == 0:
        if not 0 <= q_arr <= 1:
            raise ValueError(msg)
    else:
        if not all(0 <= qs <= 1 for qs in q_arr):
            raise ValueError(msg)
    return q_arr


@overload
def validate_ascending(ascending: BoolishT) -> BoolishT:
    ...


@overload
def validate_ascending(ascending: Sequence[BoolishT]) -> list[BoolishT]:
    ...


def validate_ascending(
    ascending: bool | int | Sequence[BoolishT],
) -> bool | int | list[BoolishT]:
    """Validate ``ascending`` kwargs for ``sort_index`` method."""
    kwargs = {"none_allowed": False, "int_allowed": True}
    if not isinstance(ascending, Sequence):
        return validate_bool_kwarg(ascending, "ascending", **kwargs)

    return [validate_bool_kwarg(item, "ascending", **kwargs) for item in ascending]


def validate_endpoints(closed: str | None) -> tuple[bool, bool]:
    """
    Check that the `closed` argument is among [None, "left", "right"]

    Parameters
    ----------
    closed : {None, "left", "right"}

    Returns
    -------
    left_closed : bool
    right_closed : bool

    Raises
    ------
    ValueError : if argument is not among valid values
    """
    left_closed = False
    right_closed = False

    if closed is None:
        left_closed = True
        right_closed = True
    elif closed == "left":
        left_closed = True
    elif closed == "right":
        right_closed = True
    else:
        raise ValueError("Closed has to be either 'left', 'right' or None")

    return left_closed, right_closed


def validate_inclusive(inclusive: str | None) -> tuple[bool, bool]:
    """
    Check that the `inclusive` argument is among {"both", "neither", "left", "right"}.

    Parameters
    ----------
    inclusive : {"both", "neither", "left", "right"}

    Returns
    -------
    left_right_inclusive : tuple[bool, bool]

    Raises
    ------
    ValueError : if argument is not among valid values
    """
    left_right_inclusive: tuple[bool, bool] | None = None

    if isinstance(inclusive, str):
        left_right_inclusive = {
            "both": (True, True),
            "left": (True, False),
            "right": (False, True),
            "neither": (False, False),
        }.get(inclusive)

    if left_right_inclusive is None:
        raise ValueError(
            "Inclusive has to be either 'both', 'neither', 'left' or 'right'"
        )

    return left_right_inclusive


def validate_insert_loc(loc: int, length: int) -> int:
    """
    Check that we have an integer between -length and length, inclusive.

    Standardize negative loc to within [0, length].

    The exceptions we raise on failure match np.insert.
    """
    if not is_integer(loc):
        raise TypeError(f"loc must be an integer between -{length} and {length}")

    if loc < 0:
        loc += length
    if not 0 <= loc <= length:
        raise IndexError(f"loc must be an integer between -{length} and {length}")
    return loc  # pyright: ignore[reportGeneralTypeIssues]


def check_dtype_backend(dtype_backend) -> None:
    if dtype_backend is not lib.no_default:
        if dtype_backend not in ["numpy_nullable", "pyarrow"]:
            raise ValueError(
                f"dtype_backend {dtype_backend} is invalid, only 'numpy_nullable' and "
                f"'pyarrow' are allowed.",
            )
<|endoftext|>from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable


class TablePlotter:
    """
    Layout some DataFrames in vertical/horizontal layout for explanation.
    Used in merging.rst
    """

    def __init__(
        self,
        cell_width: float = 0.37,
        cell_height: float = 0.25,
        font_size: float = 7.5,
    ) -> None:
        self.cell_width = cell_width
        self.cell_height = cell_height
        self.font_size = font_size

    def _shape(self, df: pd.DataFrame) -> tuple[int, int]:
        """
        Calculate table shape considering index levels.
        """
        row, col = df.shape
        return row + df.columns.nlevels, col + df.index.nlevels

    def _get_cells(self, left, right, vertical) -> tuple[int, int]:
        """
        Calculate appropriate figure size based on left and right data.
        """
        if vertical:
            # calculate required number of cells
            vcells = max(sum(self._shape(df)[0] for df in left), self._shape(right)[0])
            hcells = max(self._shape(df)[1] for df in left) + self._shape(right)[1]
        else:
            vcells = max([self._shape(df)[0] for df in left] + [self._shape(right)[0]])
            hcells = sum([self._shape(df)[1] for df in left] + [self._shape(right)[1]])
        return hcells, vcells

    def plot(self, left, right, labels: Iterable[str] = (), vertical: bool = True):
        """
        Plot left / right DataFrames in specified layout.

        Parameters
        ----------
        left : list of DataFrames before operation is applied
        right : DataFrame of operation result
        labels : list of str to be drawn as titles of left DataFrames
        vertical : bool, default True
            If True, use vertical layout. If False, use horizontal layout.
        """
        from matplotlib import gridspec
        import matplotlib.pyplot as plt

        if not isinstance(left, list):
            left = [left]
        left = [self._conv(df) for df in left]
        right = self._conv(right)

        hcells, vcells = self._get_cells(left, right, vertical)

        if vertical:
            figsize = self.cell_width * hcells, self.cell_height * vcells
        else:
            # include margin for titles
            figsize = self.cell_width * hcells, self.cell_height * vcells
        fig = plt.figure(figsize=figsize)

        if vertical:
            gs = gridspec.GridSpec(len(left), hcells)
            # left
            max_left_cols = max(self._shape(df)[1] for df in left)
            max_left_rows = max(self._shape(df)[0] for df in left)
            for i, (_left, _label) in enumerate(zip(left, labels)):
                ax = fig.add_subplot(gs[i, 0:max_left_cols])
                self._make_table(ax, _left, title=_label, height=1.0 / max_left_rows)
            # right
            ax = plt.subplot(gs[:, max_left_cols:])
            self._make_table(ax, right, title="Result", height=1.05 / vcells)
            fig.subplots_adjust(top=0.9, bottom=0.05, left=0.05, right=0.95)
        else:
            max_rows = max(self._shape(df)[0] for df in left + [right])
            height = 1.0 / np.max(max_rows)
            gs = gridspec.GridSpec(1, hcells)
            # left
            i = 0
            for df, _label in zip(left, labels):
                sp = self._shape(df)
                ax = fig.add_subplot(gs[0, i : i + sp[1]])
                self._make_table(ax, df, title=_label, height=height)
                i += sp[1]
            # right
            ax = plt.subplot(gs[0, i:])
            self._make_table(ax, right, title="Result", height=height)
            fig.subplots_adjust(top=0.85, bottom=0.05, left=0.05, right=0.95)

        return fig

    def _conv(self, data):
        """
        Convert each input to appropriate for table outplot.
        """
        if isinstance(data, pd.Series):
            if data.name is None:
                data = data.to_frame(name="")
            else:
                data = data.to_frame()
        data = data.fillna("NaN")
        return data

    def _insert_index(self, data):
        # insert is destructive
        data = data.copy()
        idx_nlevels = data.index.nlevels
        if idx_nlevels == 1:
            data.insert(0, "Index", data.index)
        else:
            for i in range(idx_nlevels):
                data.insert(i, f"Index{i}", data.index._get_level_values(i))

        col_nlevels = data.columns.nlevels
        if col_nlevels > 1:
            col = data.columns._get_level_values(0)
            values = [
                data.columns._get_level_values(i)._values for i in range(1, col_nlevels)
            ]
            col_df = pd.DataFrame(values)
            data.columns = col_df.columns
            data = pd.concat([col_df, data])
            data.columns = col
        return data

    def _make_table(self, ax, df, title: str, height: float | None = None) -> None:
        if df is None:
            ax.set_visible(False)
            return

        from pandas import plotting

        idx_nlevels = df.index.nlevels
        col_nlevels = df.columns.nlevels
        # must be convert here to get index levels for colorization
        df = self._insert_index(df)
        tb = plotting.table(ax, df, loc=9)
        tb.set_fontsize(self.font_size)

        if height is None:
            height = 1.0 / (len(df) + 1)

        props = tb.properties()
        for (r, c), cell in props["celld"].items():
            if c == -1:
                cell.set_visible(False)
            elif r < col_nlevels and c < idx_nlevels:
                cell.set_visible(False)
            elif r < col_nlevels or c < idx_nlevels:
                cell.set_facecolor("#AAAAAA")
            cell.set_height(height)

        ax.set_title(title, size=self.font_size)
        ax.axis("off")


def main() -> None:
    import matplotlib.pyplot as plt

    p = TablePlotter()

    df1 = pd.DataFrame({"A": [10, 11, 12], "B": [20, 21, 22], "C": [30, 31, 32]})
    df2 = pd.DataFrame({"A": [10, 12], "C": [30, 32]})

    p.plot([df1, df2], pd.concat([df1, df2]), labels=["df1", "df2"], vertical=True)
    plt.show()

    df3 = pd.DataFrame({"X": [10, 12], "Z": [30, 32]})

    p.plot(
        [df1, df3], pd.concat([df1, df3], axis=1), labels=["df1", "df2"], vertical=False
    )
    plt.show()

    idx = pd.MultiIndex.from_tuples(
        [(1, "A"), (1, "B"), (1, "C"), (2, "A"), (2, "B"), (2, "C")]
    )
    column = pd.MultiIndex.from_tuples([(1, "A"), (1, "B")])
    df3 = pd.DataFrame({"v1": [1, 2, 3, 4, 5, 6], "v2": [5, 6, 7, 8, 9, 10]}, index=idx)
    df3.columns = column
    p.plot(df3, df3, labels=["df3"])
    plt.show()


if __name__ == "__main__":
    main()
<|endoftext|>"""
Entrypoint for testing from the top-level namespace.
"""
from __future__ import annotations

import os
import sys

from pandas.compat._optional import import_optional_dependency

PKG = os.path.dirname(os.path.dirname(__file__))


def test(extra_args: list[str] | None = None, run_doctests: bool = False) -> None:
    """
    Run the pandas test suite using pytest.

    By default, runs with the marks -m "not slow and not network and not db"

    Parameters
    ----------
    extra_args : list[str], default None
        Extra marks to run the tests.
    run_doctests : bool, default False
        Whether to only run the Python and Cython doctests. If you would like to run
        both doctests/regular tests, just append "--doctest-modules"/"--doctest-cython"
        to extra_args.

    Examples
    --------
    >>> pd.test()  # doctest: +SKIP
    running: pytest...
    """
    pytest = import_optional_dependency("pytest")
    import_optional_dependency("hypothesis")
    cmd = ["-m not slow and not network and not db"]
    if extra_args:
        if not isinstance(extra_args, list):
            extra_args = [extra_args]
        cmd = extra_args
    if run_doctests:
        cmd = [
            "--doctest-modules",
            "--doctest-cython",
            f"--ignore={os.path.join(PKG, 'tests')}",
        ]
    cmd += [PKG]
    joined = " ".join(cmd)
    print(f"running: pytest {joined}")
    sys.exit(pytest.main(cmd))


__all__ = ["test"]
<|endoftext|>def __getattr__(key: str):
    # These imports need to be lazy to avoid circular import errors
    if key == "hash_array":
        from pandas.core.util.hashing import hash_array

        return hash_array
    if key == "hash_pandas_object":
        from pandas.core.util.hashing import hash_pandas_object

        return hash_pandas_object
    if key == "Appender":
        from pandas.util._decorators import Appender

        return Appender
    if key == "Substitution":
        from pandas.util._decorators import Substitution

        return Substitution

    if key == "cache_readonly":
        from pandas.util._decorators import cache_readonly

        return cache_readonly

    raise AttributeError(f"module 'pandas.util' has no attribute '{key}'")


def capitalize_first_letter(s):
    return s[:1].upper() + s[1:]
<|endoftext|>"""
This module provides decorator functions which can be applied to test objects
in order to skip those objects when certain conditions occur. A sample use case
is to detect if the platform is missing ``matplotlib``. If so, any test objects
which require ``matplotlib`` and decorated with ``@td.skip_if_no("matplotlib")``
will be skipped by ``pytest`` during the execution of the test suite.

To illustrate, after importing this module:

import pandas.util._test_decorators as td

The decorators can be applied to classes:

@td.skip_if_no("package")
class Foo:
    ...

Or individual functions:

@td.skip_if_no("package")
def test_foo():
    ...

For more information, refer to the ``pytest`` documentation on ``skipif``.
"""
from __future__ import annotations

import locale
from typing import (
    TYPE_CHECKING,
    Callable,
)

import pytest

from pandas._config import get_option

if TYPE_CHECKING:
    from pandas._typing import F

from pandas._config.config import _get_option

from pandas.compat import (
    IS64,
    is_platform_windows,
)
from pandas.compat._optional import import_optional_dependency


def skip_if_installed(package: str) -> pytest.MarkDecorator:
    """
    Skip a test if a package is installed.

    Parameters
    ----------
    package : str
        The name of the package.

    Returns
    -------
    pytest.MarkDecorator
        a pytest.mark.skipif to use as either a test decorator or a
        parametrization mark.
    """
    return pytest.mark.skipif(
        bool(import_optional_dependency(package, errors="ignore")),
        reason=f"Skipping because {package} is installed.",
    )


def skip_if_no(package: str, min_version: str | None = None) -> pytest.MarkDecorator:
    """
    Generic function to help skip tests when required packages are not
    present on the testing system.

    This function returns a pytest mark with a skip condition that will be
    evaluated during test collection. An attempt will be made to import the
    specified ``package`` and optionally ensure it meets the ``min_version``

    The mark can be used as either a decorator for a test class or to be
    applied to parameters in pytest.mark.parametrize calls or parametrized
    fixtures. Use pytest.importorskip if an imported moduled is later needed
    or for test functions.

    If the import and version check are unsuccessful, then the test function
    (or test case when used in conjunction with parametrization) will be
    skipped.

    Parameters
    ----------
    package: str
        The name of the required package.
    min_version: str or None, default None
        Optional minimum version of the package.

    Returns
    -------
    pytest.MarkDecorator
        a pytest.mark.skipif to use as either a test decorator or a
        parametrization mark.
    """
    msg = f"Could not import '{package}'"
    if min_version:
        msg += f" satisfying a min_version of {min_version}"
    return pytest.mark.skipif(
        not bool(
            import_optional_dependency(
                package, errors="ignore", min_version=min_version
            )
        ),
        reason=msg,
    )


skip_if_32bit = pytest.mark.skipif(not IS64, reason="skipping for 32 bit")
skip_if_windows = pytest.mark.skipif(is_platform_windows(), reason="Running on Windows")
skip_if_not_us_locale = pytest.mark.skipif(
    locale.getlocale()[0] != "en_US",
    reason=f"Set local {locale.getlocale()[0]} is not en_US",
)


def parametrize_fixture_doc(*args) -> Callable[[F], F]:
    """
    Intended for use as a decorator for parametrized fixture,
    this function will wrap the decorated function with a pytest
    ``parametrize_fixture_doc`` mark. That mark will format
    initial fixture docstring by replacing placeholders {0}, {1} etc
    with parameters passed as arguments.

    Parameters
    ----------
    args: iterable
        Positional arguments for docstring.

    Returns
    -------
    function
        The decorated function wrapped within a pytest
        ``parametrize_fixture_doc`` mark
    """

    def documented_fixture(fixture):
        fixture.__doc__ = fixture.__doc__.format(*args)
        return fixture

    return documented_fixture


def mark_array_manager_not_yet_implemented(request) -> None:
    mark = pytest.mark.xfail(reason="Not yet implemented for ArrayManager")
    request.applymarker(mark)


skip_array_manager_not_yet_implemented = pytest.mark.xfail(
    _get_option("mode.data_manager", silent=True) == "array",
    reason="Not yet implemented for ArrayManager",
)

skip_array_manager_invalid_test = pytest.mark.skipif(
    _get_option("mode.data_manager", silent=True) == "array",
    reason="Test that relies on BlockManager internals or specific behaviour",
)

skip_copy_on_write_not_yet_implemented = pytest.mark.xfail(
    get_option("mode.copy_on_write") is True,
    reason="Not yet implemented/adapted for Copy-on-Write mode",
)

skip_copy_on_write_invalid_test = pytest.mark.skipif(
    get_option("mode.copy_on_write") is True,
    reason="Test not valid for Copy-on-Write mode",
)
<|endoftext|>from __future__ import annotations

import contextlib
import inspect
import os
import re
from typing import TYPE_CHECKING
import warnings

if TYPE_CHECKING:
    from collections.abc import Generator
    from types import FrameType


@contextlib.contextmanager
def rewrite_exception(old_name: str, new_name: str) -> Generator[None, None, None]:
    """
    Rewrite the message of an exception.
    """
    try:
        yield
    except Exception as err:
        if not err.args:
            raise
        msg = str(err.args[0])
        msg = msg.replace(old_name, new_name)
        args: tuple[str, ...] = (msg,)
        if len(err.args) > 1:
            args = args + err.args[1:]
        err.args = args
        raise


def find_stack_level() -> int:
    """
    Find the first place in the stack that is not inside pandas
    (tests notwithstanding).
    """

    import pandas as pd

    pkg_dir = os.path.dirname(pd.__file__)
    test_dir = os.path.join(pkg_dir, "tests")

    # https://stackoverflow.com/questions/17407119/python-inspect-stack-is-slow
    frame: FrameType | None = inspect.currentframe()
    try:
        n = 0
        while frame:
            filename = inspect.getfile(frame)
            if filename.startswith(pkg_dir) and not filename.startswith(test_dir):
                frame = frame.f_back
                n += 1
            else:
                break
    finally:
        # See note in
        # https://docs.python.org/3/library/inspect.html#inspect.Traceback
        del frame
    return n


@contextlib.contextmanager
def rewrite_warning(
    target_message: str,
    target_category: type[Warning],
    new_message: str,
    new_category: type[Warning] | None = None,
) -> Generator[None, None, None]:
    """
    Rewrite the message of a warning.

    Parameters
    ----------
    target_message : str
        Warning message to match.
    target_category : Warning
        Warning type to match.
    new_message : str
        New warning message to emit.
    new_category : Warning or None, default None
        New warning type to emit. When None, will be the same as target_category.
    """
    if new_category is None:
        new_category = target_category
    with warnings.catch_warnings(record=True) as record:
        yield
    if len(record) > 0:
        match = re.compile(target_message)
        for warning in record:
            if warning.category is target_category and re.search(
                match, str(warning.message)
            ):
                category = new_category
                message: Warning | str = new_message
            else:
                category, message = warning.category, warning.message
            warnings.warn_explicit(
                message=message,
                category=category,
                filename=warning.filename,
                lineno=warning.lineno,
            )
<|endoftext|>from __future__ import annotations

from functools import wraps
import inspect
from textwrap import dedent
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    cast,
)
import warnings

from pandas._libs.properties import cache_readonly
from pandas._typing import (
    F,
    T,
)
from pandas.util._exceptions import find_stack_level

if TYPE_CHECKING:
    from collections.abc import Mapping


def deprecate(
    name: str,
    alternative: Callable[..., Any],
    version: str,
    alt_name: str | None = None,
    klass: type[Warning] | None = None,
    stacklevel: int = 2,
    msg: str | None = None,
) -> Callable[[F], F]:
    """
    Return a new function that emits a deprecation warning on use.

    To use this method for a deprecated function, another function
    `alternative` with the same signature must exist. The deprecated
    function will emit a deprecation warning, and in the docstring
    it will contain the deprecation directive with the provided version
    so it can be detected for future removal.

    Parameters
    ----------
    name : str
        Name of function to deprecate.
    alternative : func
        Function to use instead.
    version : str
        Version of pandas in which the method has been deprecated.
    alt_name : str, optional
        Name to use in preference of alternative.__name__.
    klass : Warning, default FutureWarning
    stacklevel : int, default 2
    msg : str
        The message to display in the warning.
        Default is '{name} is deprecated. Use {alt_name} instead.'
    """
    alt_name = alt_name or alternative.__name__
    klass = klass or FutureWarning
    warning_msg = msg or f"{name} is deprecated, use {alt_name} instead."

    @wraps(alternative)
    def wrapper(*args, **kwargs) -> Callable[..., Any]:
        warnings.warn(warning_msg, klass, stacklevel=stacklevel)
        return alternative(*args, **kwargs)

    # adding deprecated directive to the docstring
    msg = msg or f"Use `{alt_name}` instead."
    doc_error_msg = (
        "deprecate needs a correctly formatted docstring in "
        "the target function (should have a one liner short "
        "summary, and opening quotes should be in their own "
        f"line). Found:\n{alternative.__doc__}"
    )

    # when python is running in optimized mode (i.e. `-OO`), docstrings are
    # removed, so we check that a docstring with correct formatting is used
    # but we allow empty docstrings
    if alternative.__doc__:
        if alternative.__doc__.count("\n") < 3:
            raise AssertionError(doc_error_msg)
        empty1, summary, empty2, doc_string = alternative.__doc__.split("\n", 3)
        if empty1 or empty2 and not summary:
            raise AssertionError(doc_error_msg)
        wrapper.__doc__ = dedent(
            f"""
        {summary.strip()}

        .. deprecated:: {version}
            {msg}

        {dedent(doc_string)}"""
        )
    # error: Incompatible return value type (got "Callable[[VarArg(Any), KwArg(Any)],
    # Callable[...,Any]]", expected "Callable[[F], F]")
    return wrapper  # type: ignore[return-value]


def deprecate_kwarg(
    old_arg_name: str,
    new_arg_name: str | None,
    mapping: Mapping[Any, Any] | Callable[[Any], Any] | None = None,
    stacklevel: int = 2,
) -> Callable[[F], F]:
    """
    Decorator to deprecate a keyword argument of a function.

    Parameters
    ----------
    old_arg_name : str
        Name of argument in function to deprecate
    new_arg_name : str or None
        Name of preferred argument in function. Use None to raise warning that
        ``old_arg_name`` keyword is deprecated.
    mapping : dict or callable
        If mapping is present, use it to translate old arguments to
        new arguments. A callable must do its own value checking;
        values not found in a dict will be forwarded unchanged.

    Examples
    --------
    The following deprecates 'cols', using 'columns' instead

    >>> @deprecate_kwarg(old_arg_name='cols', new_arg_name='columns')
    ... def f(columns=''):
    ...     print(columns)
    ...
    >>> f(columns='should work ok')
    should work ok

    >>> f(cols='should raise warning')  # doctest: +SKIP
    FutureWarning: cols is deprecated, use columns instead
      warnings.warn(msg, FutureWarning)
    should raise warning

    >>> f(cols='should error', columns="can\'t pass do both")  # doctest: +SKIP
    TypeError: Can only specify 'cols' or 'columns', not both

    >>> @deprecate_kwarg('old', 'new', {'yes': True, 'no': False})
    ... def f(new=False):
    ...     print('yes!' if new else 'no!')
    ...
    >>> f(old='yes')  # doctest: +SKIP
    FutureWarning: old='yes' is deprecated, use new=True instead
      warnings.warn(msg, FutureWarning)
    yes!

    To raise a warning that a keyword will be removed entirely in the future

    >>> @deprecate_kwarg(old_arg_name='cols', new_arg_name=None)
    ... def f(cols='', another_param=''):
    ...     print(cols)
    ...
    >>> f(cols='should raise warning')  # doctest: +SKIP
    FutureWarning: the 'cols' keyword is deprecated and will be removed in a
    future version please takes steps to stop use of 'cols'
    should raise warning
    >>> f(another_param='should not raise warning')  # doctest: +SKIP
    should not raise warning

    >>> f(cols='should raise warning', another_param='')  # doctest: +SKIP
    FutureWarning: the 'cols' keyword is deprecated and will be removed in a
    future version please takes steps to stop use of 'cols'
    should raise warning
    """
    if mapping is not None and not hasattr(mapping, "get") and not callable(mapping):
        raise TypeError(
            "mapping from old to new argument values must be dict or callable!"
        )

    def _deprecate_kwarg(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Callable[..., Any]:
            old_arg_value = kwargs.pop(old_arg_name, None)

            if old_arg_value is not None:
                if new_arg_name is None:
                    msg = (
                        f"the {repr(old_arg_name)} keyword is deprecated and "
                        "will be removed in a future version. Please take "
                        f"steps to stop the use of {repr(old_arg_name)}"
                    )
                    warnings.warn(msg, FutureWarning, stacklevel=stacklevel)
                    kwargs[old_arg_name] = old_arg_value
                    return func(*args, **kwargs)

                elif mapping is not None:
                    if callable(mapping):
                        new_arg_value = mapping(old_arg_value)
                    else:
                        new_arg_value = mapping.get(old_arg_value, old_arg_value)
                    msg = (
                        f"the {old_arg_name}={repr(old_arg_value)} keyword is "
                        "deprecated, use "
                        f"{new_arg_name}={repr(new_arg_value)} instead."
                    )
                else:
                    new_arg_value = old_arg_value
                    msg = (
                        f"the {repr(old_arg_name)} keyword is deprecated, "
                        f"use {repr(new_arg_name)} instead."
                    )

                warnings.warn(msg, FutureWarning, stacklevel=stacklevel)
                if kwargs.get(new_arg_name) is not None:
                    msg = (
                        f"Can only specify {repr(old_arg_name)} "
                        f"or {repr(new_arg_name)}, not both."
                    )
                    raise TypeError(msg)
                kwargs[new_arg_name] = new_arg_value
            return func(*args, **kwargs)

        return cast(F, wrapper)

    return _deprecate_kwarg


def _format_argument_list(allow_args: list[str]) -> str:
    """
    Convert the allow_args argument (either string or integer) of
    `deprecate_nonkeyword_arguments` function to a string describing
    it to be inserted into warning message.

    Parameters
    ----------
    allowed_args : list, tuple or int
        The `allowed_args` argument for `deprecate_nonkeyword_arguments`,
        but None value is not allowed.

    Returns
    -------
    str
        The substring describing the argument list in best way to be
        inserted to the warning message.

    Examples
    --------
    `format_argument_list([])` -> ''
    `format_argument_list(['a'])` -> "except for the arguments 'a'"
    `format_argument_list(['a', 'b'])` -> "except for the arguments 'a' and 'b'"
    `format_argument_list(['a', 'b', 'c'])` ->
        "except for the arguments 'a', 'b' and 'c'"
    """
    if "self" in allow_args:
        allow_args.remove("self")
    if not allow_args:
        return ""
    elif len(allow_args) == 1:
        return f" except for the argument '{allow_args[0]}'"
    else:
        last = allow_args[-1]
        args = ", ".join(["'" + x + "'" for x in allow_args[:-1]])
        return f" except for the arguments {args} and '{last}'"


def future_version_msg(version: str | None) -> str:
    """Specify which version of pandas the deprecation will take place in."""
    if version is None:
        return "In a future version of pandas"
    else:
        return f"Starting with pandas version {version}"


def deprecate_nonkeyword_arguments(
    version: str | None,
    allowed_args: list[str] | None = None,
    name: str | None = None,
) -> Callable[[F], F]:
    """
    Decorator to deprecate a use of non-keyword arguments of a function.

    Parameters
    ----------
    version : str, optional
        The version in which positional arguments will become
        keyword-only. If None, then the warning message won't
        specify any particular version.

    allowed_args : list, optional
        In case of list, it must be the list of names of some
        first arguments of the decorated functions that are
        OK to be given as positional arguments. In case of None value,
        defaults to list of all arguments not having the
        default value.

    name : str, optional
        The specific name of the function to show in the warning
        message. If None, then the Qualified name of the function
        is used.
    """

    def decorate(func):
        old_sig = inspect.signature(func)

        if allowed_args is not None:
            allow_args = allowed_args
        else:
            allow_args = [
                p.name
                for p in old_sig.parameters.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                and p.default is p.empty
            ]

        new_params = [
            p.replace(kind=p.KEYWORD_ONLY)
            if (
                p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                and p.name not in allow_args
            )
            else p
            for p in old_sig.parameters.values()
        ]
        new_params.sort(key=lambda p: p.kind)
        new_sig = old_sig.replace(parameters=new_params)

        num_allow_args = len(allow_args)
        msg = (
            f"{future_version_msg(version)} all arguments of "
            f"{name or func.__qualname__}{{arguments}} will be keyword-only."
        )

        @wraps(func)
        def wrapper(*args, **kwargs):
            if len(args) > num_allow_args:
                warnings.warn(
                    msg.format(arguments=_format_argument_list(allow_args)),
                    FutureWarning,
                    stacklevel=find_stack_level(),
                )
            return func(*args, **kwargs)

        # error: "Callable[[VarArg(Any), KwArg(Any)], Any]" has no
        # attribute "__signature__"
        wrapper.__signature__ = new_sig  # type: ignore[attr-defined]
        return wrapper

    return decorate


def doc(*docstrings: None | str | Callable, **params) -> Callable[[F], F]:
    """
    A decorator to take docstring templates, concatenate them and perform string
    substitution on them.

    This decorator will add a variable "_docstring_components" to the wrapped
    callable to keep track the original docstring template for potential usage.
    If it should be consider as a template, it will be saved as a string.
    Otherwise, it will be saved as callable, and later user __doc__ and dedent
    to get docstring.

    Parameters
    ----------
    *docstrings : None, str, or callable
        The string / docstring / docstring template to be appended in order
        after default docstring under callable.
    **params
        The string which would be used to format docstring template.
    """

    def decorator(decorated: F) -> F:
        # collecting docstring and docstring templates
        docstring_components: list[str | Callable] = []
        if decorated.__doc__:
            docstring_components.append(dedent(decorated.__doc__))

        for docstring in docstrings:
            if docstring is None:
                continue
            if hasattr(docstring, "_docstring_components"):
                docstring_components.extend(
                    docstring._docstring_components  # pyright: ignore[reportGeneralTypeIssues]
                )
            elif isinstance(docstring, str) or docstring.__doc__:
                docstring_components.append(docstring)

        params_applied = [
            component.format(**params)
            if isinstance(component, str) and len(params) > 0
            else component
            for component in docstring_components
        ]

        decorated.__doc__ = "".join(
            [
                component
                if isinstance(component, str)
                else dedent(component.__doc__ or "")
                for component in params_applied
            ]
        )

        # error: "F" has no attribute "_docstring_components"
        decorated._docstring_components = (  # type: ignore[attr-defined]
            docstring_components
        )
        return decorated

    return decorator


# Substitution and Appender are derived from matplotlib.docstring (1.1.0)
# module https://matplotlib.org/users/license.html


class Substitution:
    """
    A decorator to take a function's docstring and perform string
    substitution on it.

    This decorator should be robust even if func.__doc__ is None
    (for example, if -OO was passed to the interpreter)

    Usage: construct a docstring.Substitution with a sequence or
    dictionary suitable for performing substitution; then
    decorate a suitable function with the constructed object. e.g.

    sub_author_name = Substitution(author='Jason')

    @sub_author_name
    def some_function(x):
        "%(author)s wrote this function"

    # note that some_function.__doc__ is now "Jason wrote this function"

    One can also use positional arguments.

    sub_first_last_names = Substitution('Edgar Allen', 'Poe')

    @sub_first_last_names
    def some_function(x):
        "%s %s wrote the Raven"
    """

    def __init__(self, *args, **kwargs) -> None:
        if args and kwargs:
            raise AssertionError("Only positional or keyword args are allowed")

        self.params = args or kwargs

    def __call__(self, func: F) -> F:
        func.__doc__ = func.__doc__ and func.__doc__ % self.params
        return func

    def update(self, *args, **kwargs) -> None:
        """
        Update self.params with supplied args.
        """
        if isinstance(self.params, dict):
            self.params.update(*args, **kwargs)


class Appender:
    """
    A function decorator that will append an addendum to the docstring
    of the target function.

    This decorator should be robust even if func.__doc__ is None
    (for example, if -OO was passed to the interpreter).

    Usage: construct a docstring.Appender with a string to be joined to
    the original docstring. An optional 'join' parameter may be supplied
    which will be used to join the docstring and addendum. e.g.

    add_copyright = Appender("Copyright (c) 2009", join='\n')

    @add_copyright
    def my_dog(has='fleas'):
        "This docstring will have a copyright below"
        pass
    """

    addendum: str | None

    def __init__(self, addendum: str | None, join: str = "", indents: int = 0) -> None:
        if indents > 0:
            self.addendum = indent(addendum, indents=indents)
        else:
            self.addendum = addendum
        self.join = join

    def __call__(self, func: T) -> T:
        func.__doc__ = func.__doc__ if func.__doc__ else ""
        self.addendum = self.addendum if self.addendum else ""
        docitems = [func.__doc__, self.addendum]
        func.__doc__ = dedent(self.join.join(docitems))
        return func


def indent(text: str | None, indents: int = 1) -> str:
    if not text or not isinstance(text, str):
        return ""
    jointext = "".join(["\n"] + ["    "] * indents)
    return jointext.join(text.split("\n"))


__all__ = [
    "Appender",
    "cache_readonly",
    "deprecate",
    "deprecate_kwarg",
    "deprecate_nonkeyword_arguments",
    "doc",
    "future_version_msg",
    "Substitution",
]
<|endoftext|># Vendored from https://github.com/pypa/packaging/blob/main/packaging/_structures.py
# and https://github.com/pypa/packaging/blob/main/packaging/_structures.py
# changeset ae891fd74d6dd4c6063bb04f2faeadaac6fc6313
# 04/30/2021

# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. Licence at LICENSES/PACKAGING_LICENSE
from __future__ import annotations

import collections
from collections.abc import Iterator
import itertools
import re
from typing import (
    Callable,
    SupportsInt,
    Tuple,
    Union,
)
import warnings

__all__ = ["parse", "Version", "LegacyVersion", "InvalidVersion", "VERSION_PATTERN"]


class InfinityType:
    def __repr__(self) -> str:
        return "Infinity"

    def __hash__(self) -> int:
        return hash(repr(self))

    def __lt__(self, other: object) -> bool:
        return False

    def __le__(self, other: object) -> bool:
        return False

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self))

    def __ne__(self, other: object) -> bool:
        return not isinstance(other, type(self))

    def __gt__(self, other: object) -> bool:
        return True

    def __ge__(self, other: object) -> bool:
        return True

    def __neg__(self: object) -> NegativeInfinityType:
        return NegativeInfinity


Infinity = InfinityType()


class NegativeInfinityType:
    def __repr__(self) -> str:
        return "-Infinity"

    def __hash__(self) -> int:
        return hash(repr(self))

    def __lt__(self, other: object) -> bool:
        return True

    def __le__(self, other: object) -> bool:
        return True

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self))

    def __ne__(self, other: object) -> bool:
        return not isinstance(other, type(self))

    def __gt__(self, other: object) -> bool:
        return False

    def __ge__(self, other: object) -> bool:
        return False

    def __neg__(self: object) -> InfinityType:
        return Infinity


NegativeInfinity = NegativeInfinityType()


InfiniteTypes = Union[InfinityType, NegativeInfinityType]
PrePostDevType = Union[InfiniteTypes, tuple[str, int]]
SubLocalType = Union[InfiniteTypes, int, str]
LocalType = Union[
    NegativeInfinityType,
    tuple[
        Union[
            SubLocalType,
            tuple[SubLocalType, str],
            tuple[NegativeInfinityType, SubLocalType],
        ],
        ...,
    ],
]
CmpKey = tuple[
    int, tuple[int, ...], PrePostDevType, PrePostDevType, PrePostDevType, LocalType
]
LegacyCmpKey = tuple[int, tuple[str, ...]]
VersionComparisonMethod = Callable[
    [Union[CmpKey, LegacyCmpKey], Union[CmpKey, LegacyCmpKey]], bool
]

_Version = collections.namedtuple(
    "_Version", ["epoch", "release", "dev", "pre", "post", "local"]
)


def parse(version: str) -> LegacyVersion | Version:
    """
    Parse the given version string and return either a :class:`Version` object
    or a :class:`LegacyVersion` object depending on if the given version is
    a valid PEP 440 version or a legacy version.
    """
    try:
        return Version(version)
    except InvalidVersion:
        return LegacyVersion(version)


class InvalidVersion(ValueError):
    """
    An invalid version was found, users should refer to PEP 440.

    Examples
    --------
    >>> pd.util.version.Version('1.')
    Traceback (most recent call last):
    InvalidVersion: Invalid version: '1.'
    """


class _BaseVersion:
    _key: CmpKey | LegacyCmpKey

    def __hash__(self) -> int:
        return hash(self._key)

    # Please keep the duplicated `isinstance` check
    # in the six comparisons hereunder
    # unless you find a way to avoid adding overhead function calls.
    def __lt__(self, other: _BaseVersion) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key < other._key

    def __le__(self, other: _BaseVersion) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key <= other._key

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key == other._key

    def __ge__(self, other: _BaseVersion) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key >= other._key

    def __gt__(self, other: _BaseVersion) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key > other._key

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return self._key != other._key


class LegacyVersion(_BaseVersion):
    def __init__(self, version: str) -> None:
        self._version = str(version)
        self._key = _legacy_cmpkey(self._version)

        warnings.warn(
            "Creating a LegacyVersion has been deprecated and will be "
            "removed in the next major release.",
            DeprecationWarning,
        )

    def __str__(self) -> str:
        return self._version

    def __repr__(self) -> str:
        return f"<LegacyVersion('{self}')>"

    @property
    def public(self) -> str:
        return self._version

    @property
    def base_version(self) -> str:
        return self._version

    @property
    def epoch(self) -> int:
        return -1

    @property
    def release(self) -> None:
        return None

    @property
    def pre(self) -> None:
        return None

    @property
    def post(self) -> None:
        return None

    @property
    def dev(self) -> None:
        return None

    @property
    def local(self) -> None:
        return None

    @property
    def is_prerelease(self) -> bool:
        return False

    @property
    def is_postrelease(self) -> bool:
        return False

    @property
    def is_devrelease(self) -> bool:
        return False


_legacy_version_component_re = re.compile(r"(\d+ | [a-z]+ | \.| -)", re.VERBOSE)

_legacy_version_replacement_map = {
    "pre": "c",
    "preview": "c",
    "-": "final-",
    "rc": "c",
    "dev": "@",
}


def _parse_version_parts(s: str) -> Iterator[str]:
    for part in _legacy_version_component_re.split(s):
        mapped_part = _legacy_version_replacement_map.get(part, part)

        if not mapped_part or mapped_part == ".":
            continue

        if mapped_part[:1] in "0123456789":
            # pad for numeric comparison
            yield mapped_part.zfill(8)
        else:
            yield "*" + mapped_part

    # ensure that alpha/beta/candidate are before final
    yield "*final"


def _legacy_cmpkey(version: str) -> LegacyCmpKey:
    # We hardcode an epoch of -1 here. A PEP 440 version can only have a epoch
    # greater than or equal to 0. This will effectively put the LegacyVersion,
    # which uses the defacto standard originally implemented by setuptools,
    # as before all PEP 440 versions.
    epoch = -1

    # This scheme is taken from pkg_resources.parse_version setuptools prior to
    # it's adoption of the packaging library.
    parts: list[str] = []
    for part in _parse_version_parts(version.lower()):
        if part.startswith("*"):
            # remove "-" before a prerelease tag
            if part < "*final":
                while parts and parts[-1] == "*final-":
                    parts.pop()

            # remove trailing zeros from each series of numeric parts
            while parts and parts[-1] == "00000000":
                parts.pop()

        parts.append(part)

    return epoch, tuple(parts)


# Deliberately not anchored to the start and end of the string, to make it
# easier for 3rd party code to reuse
VERSION_PATTERN = r"""
    v?
    (?:
        (?:(?P<epoch>[0-9]+)!)?                           # epoch
        (?P<release>[0-9]+(?:\.[0-9]+)*)                  # release segment
        (?P<pre>                                          # pre-release
            [-_\.]?
            (?P<pre_l>(a|b|c|rc|alpha|beta|pre|preview))
            [-_\.]?
            (?P<pre_n>[0-9]+)?
        )?
        (?P<post>                                         # post release
            (?:-(?P<post_n1>[0-9]+))
            |
            (?:
                [-_\.]?
                (?P<post_l>post|rev|r)
                [-_\.]?
                (?P<post_n2>[0-9]+)?
            )
        )?
        (?P<dev>                                          # dev release
            [-_\.]?
            (?P<dev_l>dev)
            [-_\.]?
            (?P<dev_n>[0-9]+)?
        )?
    )
    (?:\+(?P<local>[a-z0-9]+(?:[-_\.][a-z0-9]+)*))?       # local version
"""


class Version(_BaseVersion):
    _regex = re.compile(r"^\s*" + VERSION_PATTERN + r"\s*$", re.VERBOSE | re.IGNORECASE)

    def __init__(self, version: str) -> None:
        # Validate the version and parse it into pieces
        match = self._regex.search(version)
        if not match:
            raise InvalidVersion(f"Invalid version: '{version}'")

        # Store the parsed out pieces of the version
        self._version = _Version(
            epoch=int(match.group("epoch")) if match.group("epoch") else 0,
            release=tuple(int(i) for i in match.group("release").split(".")),
            pre=_parse_letter_version(match.group("pre_l"), match.group("pre_n")),
            post=_parse_letter_version(
                match.group("post_l"), match.group("post_n1") or match.group("post_n2")
            ),
            dev=_parse_letter_version(match.group("dev_l"), match.group("dev_n")),
            local=_parse_local_version(match.group("local")),
        )

        # Generate a key which will be used for sorting
        self._key = _cmpkey(
            self._version.epoch,
            self._version.release,
            self._version.pre,
            self._version.post,
            self._version.dev,
            self._version.local,
        )

    def __repr__(self) -> str:
        return f"<Version('{self}')>"

    def __str__(self) -> str:
        parts = []

        # Epoch
        if self.epoch != 0:
            parts.append(f"{self.epoch}!")

        # Release segment
        parts.append(".".join([str(x) for x in self.release]))

        # Pre-release
        if self.pre is not None:
            parts.append("".join([str(x) for x in self.pre]))

        # Post-release
        if self.post is not None:
            parts.append(f".post{self.post}")

        # Development release
        if self.dev is not None:
            parts.append(f".dev{self.dev}")

        # Local version segment
        if self.local is not None:
            parts.append(f"+{self.local}")

        return "".join(parts)

    @property
    def epoch(self) -> int:
        _epoch: int = self._version.epoch
        return _epoch

    @property
    def release(self) -> tuple[int, ...]:
        _release: tuple[int, ...] = self._version.release
        return _release

    @property
    def pre(self) -> tuple[str, int] | None:
        _pre: tuple[str, int] | None = self._version.pre
        return _pre

    @property
    def post(self) -> int | None:
        return self._version.post[1] if self._version.post else None

    @property
    def dev(self) -> int | None:
        return self._version.dev[1] if self._version.dev else None

    @property
    def local(self) -> str | None:
        if self._version.local:
            return ".".join([str(x) for x in self._version.local])
        else:
            return None

    @property
    def public(self) -> str:
        return str(self).split("+", 1)[0]

    @property
    def base_version(self) -> str:
        parts = []

        # Epoch
        if self.epoch != 0:
            parts.append(f"{self.epoch}!")

        # Release segment
        parts.append(".".join([str(x) for x in self.release]))

        return "".join(parts)

    @property
    def is_prerelease(self) -> bool:
        return self.dev is not None or self.pre is not None

    @property
    def is_postrelease(self) -> bool:
        return self.post is not None

    @property
    def is_devrelease(self) -> bool:
        return self.dev is not None

    @property
    def major(self) -> int:
        return self.release[0] if len(self.release) >= 1 else 0

    @property
    def minor(self) -> int:
        return self.release[1] if len(self.release) >= 2 else 0

    @property
    def micro(self) -> int:
        return self.release[2] if len(self.release) >= 3 else 0


def _parse_letter_version(
    letter: str, number: str | bytes | SupportsInt
) -> tuple[str, int] | None:
    if letter:
        # We consider there to be an implicit 0 in a pre-release if there is
        # not a numeral associated with it.
        if number is None:
            number = 0

        # We normalize any letters to their lower case form
        letter = letter.lower()

        # We consider some words to be alternate spellings of other words and
        # in those cases we want to normalize the spellings to our preferred
        # spelling.
        if letter == "alpha":
            letter = "a"
        elif letter == "beta":
            letter = "b"
        elif letter in ["c", "pre", "preview"]:
            letter = "rc"
        elif letter in ["rev", "r"]:
            letter = "post"

        return letter, int(number)
    if not letter and number:
        # We assume if we are given a number, but we are not given a letter
        # then this is using the implicit post release syntax (e.g. 1.0-1)
        letter = "post"

        return letter, int(number)

    return None


_local_version_separators = re.compile(r"[\._-]")


def _parse_local_version(local: str) -> LocalType | None:
    """
    Takes a string like abc.1.twelve and turns it into ("abc", 1, "twelve").
    """
    if local is not None:
        return tuple(
            part.lower() if not part.isdigit() else int(part)
            for part in _local_version_separators.split(local)
        )
    return None


def _cmpkey(
    epoch: int,
    release: tuple[int, ...],
    pre: tuple[str, int] | None,
    post: tuple[str, int] | None,
    dev: tuple[str, int] | None,
    local: tuple[SubLocalType] | None,
) -> CmpKey:
    # When we compare a release version, we want to compare it with all of the
    # trailing zeros removed. So we'll use a reverse the list, drop all the now
    # leading zeros until we come to something non zero, then take the rest
    # re-reverse it back into the correct order and make it a tuple and use
    # that for our sorting key.
    _release = tuple(
        reversed(list(itertools.dropwhile(lambda x: x == 0, reversed(release))))
    )

    # We need to "trick" the sorting algorithm to put 1.0.dev0 before 1.0a0.
    # We'll do this by abusing the pre segment, but we _only_ want to do this
    # if there is not a pre or a post segment. If we have one of those then
    # the normal sorting rules will handle this case correctly.
    if pre is None and post is None and dev is not None:
        _pre: PrePostDevType = NegativeInfinity
    # Versions without a pre-release (except as noted above) should sort after
    # those with one.
    elif pre is None:
        _pre = Infinity
    else:
        _pre = pre

    # Versions without a post segment should sort before those with one.
    if post is None:
        _post: PrePostDevType = NegativeInfinity

    else:
        _post = post

    # Versions without a development segment should sort after those with one.
    if dev is None:
        _dev: PrePostDevType = Infinity

    else:
        _dev = dev

    if local is None:
        # Versions without a local segment should sort before those with one.
        _local: LocalType = NegativeInfinity
    else:
        # Versions with a local segment need that segment parsed to implement
        # the sorting rules in PEP440.
        # - Alpha numeric segments sort before numeric segments
        # - Alpha numeric segments sort lexicographically
        # - Numeric segments sort numerically
        # - Shorter versions sort before longer versions when the prefixes
        #   match exactly
        _local = tuple(
            (i, "") if isinstance(i, int) else (NegativeInfinity, i) for i in local
        )

    return epoch, _release, _pre, _post, _dev, _local
<|endoftext|>"""
Constructor functions intended to be shared by pd.array, Series.__init__,
and Index.__new__.

These should not depend on core.internals.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import (
    TYPE_CHECKING,
    Optional,
    Union,
    cast,
    overload,
)
import warnings

import numpy as np
from numpy import ma

from pandas._config import using_string_dtype

from pandas._libs import lib
from pandas._libs.tslibs import (
    Period,
    get_supported_dtype,
    is_supported_dtype,
)
from pandas._typing import (
    AnyArrayLike,
    ArrayLike,
    Dtype,
    DtypeObj,
    T,
)
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.cast import (
    construct_1d_arraylike_from_scalar,
    construct_1d_object_array_from_listlike,
    maybe_cast_to_datetime,
    maybe_cast_to_integer_array,
    maybe_convert_platform,
    maybe_infer_to_datetimelike,
    maybe_promote,
)
from pandas.core.dtypes.common import (
    is_list_like,
    is_object_dtype,
    is_string_dtype,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import NumpyEADtype
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCExtensionArray,
    ABCIndex,
    ABCSeries,
)
from pandas.core.dtypes.missing import isna

import pandas.core.common as com

if TYPE_CHECKING:
    from pandas import (
        Index,
        Series,
    )
    from pandas.core.arrays.base import ExtensionArray


def array(
    data: Sequence[object] | AnyArrayLike,
    dtype: Dtype | None = None,
    copy: bool = True,
) -> ExtensionArray:
    """
    Create an array.

    Parameters
    ----------
    data : Sequence of objects
        The scalars inside `data` should be instances of the
        scalar type for `dtype`. It's expected that `data`
        represents a 1-dimensional array of data.

        When `data` is an Index or Series, the underlying array
        will be extracted from `data`.

    dtype : str, np.dtype, or ExtensionDtype, optional
        The dtype to use for the array. This may be a NumPy
        dtype or an extension type registered with pandas using
        :meth:`pandas.api.extensions.register_extension_dtype`.

        If not specified, there are two possibilities:

        1. When `data` is a :class:`Series`, :class:`Index`, or
           :class:`ExtensionArray`, the `dtype` will be taken
           from the data.
        2. Otherwise, pandas will attempt to infer the `dtype`
           from the data.

        Note that when `data` is a NumPy array, ``data.dtype`` is
        *not* used for inferring the array type. This is because
        NumPy cannot represent all the types of data that can be
        held in extension arrays.

        Currently, pandas will infer an extension dtype for sequences of

        ============================== =======================================
        Scalar Type                    Array Type
        ============================== =======================================
        :class:`pandas.Interval`       :class:`pandas.arrays.IntervalArray`
        :class:`pandas.Period`         :class:`pandas.arrays.PeriodArray`
        :class:`datetime.datetime`     :class:`pandas.arrays.DatetimeArray`
        :class:`datetime.timedelta`    :class:`pandas.arrays.TimedeltaArray`
        :class:`int`                   :class:`pandas.arrays.IntegerArray`
        :class:`float`                 :class:`pandas.arrays.FloatingArray`
        :class:`str`                   :class:`pandas.arrays.StringArray` or
                                       :class:`pandas.arrays.ArrowStringArray`
        :class:`bool`                  :class:`pandas.arrays.BooleanArray`
        ============================== =======================================

        The ExtensionArray created when the scalar type is :class:`str` is determined by
        ``pd.options.mode.string_storage`` if the dtype is not explicitly given.

        For all other cases, NumPy's usual inference rules will be used.
    copy : bool, default True
        Whether to copy the data, even if not necessary. Depending
        on the type of `data`, creating the new array may require
        copying data, even if ``copy=False``.

    Returns
    -------
    ExtensionArray
        The newly created array.

    Raises
    ------
    ValueError
        When `data` is not 1-dimensional.

    See Also
    --------
    numpy.array : Construct a NumPy array.
    Series : Construct a pandas Series.
    Index : Construct a pandas Index.
    arrays.NumpyExtensionArray : ExtensionArray wrapping a NumPy array.
    Series.array : Extract the array stored within a Series.

    Notes
    -----
    Omitting the `dtype` argument means pandas will attempt to infer the
    best array type from the values in the data. As new array types are
    added by pandas and 3rd party libraries, the "best" array type may
    change. We recommend specifying `dtype` to ensure that

    1. the correct array type for the data is returned
    2. the returned array type doesn't change as new extension types
       are added by pandas and third-party libraries

    Additionally, if the underlying memory representation of the returned
    array matters, we recommend specifying the `dtype` as a concrete object
    rather than a string alias or allowing it to be inferred. For example,
    a future version of pandas or a 3rd-party library may include a
    dedicated ExtensionArray for string data. In this event, the following
    would no longer return a :class:`arrays.NumpyExtensionArray` backed by a
    NumPy array.

    >>> pd.array(['a', 'b'], dtype=str)
    <NumpyExtensionArray>
    ['a', 'b']
    Length: 2, dtype: str32

    This would instead return the new ExtensionArray dedicated for string
    data. If you really need the new array to be backed by a  NumPy array,
    specify that in the dtype.

    >>> pd.array(['a', 'b'], dtype=np.dtype("<U1"))
    <NumpyExtensionArray>
    ['a', 'b']
    Length: 2, dtype: str32

    Finally, Pandas has arrays that mostly overlap with NumPy

      * :class:`arrays.DatetimeArray`
      * :class:`arrays.TimedeltaArray`

    When data with a ``datetime64[ns]`` or ``timedelta64[ns]`` dtype is
    passed, pandas will always return a ``DatetimeArray`` or ``TimedeltaArray``
    rather than a ``NumpyExtensionArray``. This is for symmetry with the case of
    timezone-aware data, which NumPy does not natively support.

    >>> pd.array(['2015', '2016'], dtype='datetime64[ns]')
    <DatetimeArray>
    ['2015-01-01 00:00:00', '2016-01-01 00:00:00']
    Length: 2, dtype: datetime64[ns]

    >>> pd.array(["1h", "2h"], dtype='timedelta64[ns]')
    <TimedeltaArray>
    ['0 days 01:00:00', '0 days 02:00:00']
    Length: 2, dtype: timedelta64[ns]

    Examples
    --------
    If a dtype is not specified, pandas will infer the best dtype from the values.
    See the description of `dtype` for the types pandas infers for.

    >>> pd.array([1, 2])
    <IntegerArray>
    [1, 2]
    Length: 2, dtype: Int64

    >>> pd.array([1, 2, np.nan])
    <IntegerArray>
    [1, 2, <NA>]
    Length: 3, dtype: Int64

    >>> pd.array([1.1, 2.2])
    <FloatingArray>
    [1.1, 2.2]
    Length: 2, dtype: Float64

    >>> pd.array(["a", None, "c"])
    <StringArray>
    ['a', <NA>, 'c']
    Length: 3, dtype: string

    >>> with pd.option_context("string_storage", "pyarrow"):
    ...     arr = pd.array(["a", None, "c"])
    ...
    >>> arr
    <ArrowStringArray>
    ['a', <NA>, 'c']
    Length: 3, dtype: string

    >>> pd.array([pd.Period('2000', freq="D"), pd.Period("2000", freq="D")])
    <PeriodArray>
    ['2000-01-01', '2000-01-01']
    Length: 2, dtype: period[D]

    You can use the string alias for `dtype`

    >>> pd.array(['a', 'b', 'a'], dtype='category')
    ['a', 'b', 'a']
    Categories (2, object): ['a', 'b']

    Or specify the actual dtype

    >>> pd.array(['a', 'b', 'a'],
    ...          dtype=pd.CategoricalDtype(['a', 'b', 'c'], ordered=True))
    ['a', 'b', 'a']
    Categories (3, object): ['a' < 'b' < 'c']

    If pandas does not infer a dedicated extension type a
    :class:`arrays.NumpyExtensionArray` is returned.

    >>> pd.array([1 + 1j, 3 + 2j])
    <NumpyExtensionArray>
    [(1+1j), (3+2j)]
    Length: 2, dtype: complex128

    As mentioned in the "Notes" section, new extension types may be added
    in the future (by pandas or 3rd party libraries), causing the return
    value to no longer be a :class:`arrays.NumpyExtensionArray`. Specify the
    `dtype` as a NumPy dtype if you need to ensure there's no future change in
    behavior.

    >>> pd.array([1, 2], dtype=np.dtype("int32"))
    <NumpyExtensionArray>
    [1, 2]
    Length: 2, dtype: int32

    `data` must be 1-dimensional. A ValueError is raised when the input
    has the wrong dimensionality.

    >>> pd.array(1)
    Traceback (most recent call last):
      ...
    ValueError: Cannot pass scalar '1' to 'pandas.array'.
    """
    from pandas.core.arrays import (
        BooleanArray,
        DatetimeArray,
        ExtensionArray,
        FloatingArray,
        IntegerArray,
        IntervalArray,
        NumpyExtensionArray,
        PeriodArray,
        TimedeltaArray,
    )
    from pandas.core.arrays.string_ import StringDtype

    if lib.is_scalar(data):
        msg = f"Cannot pass scalar '{data}' to 'pandas.array'."
        raise ValueError(msg)
    elif isinstance(data, ABCDataFrame):
        raise TypeError("Cannot pass DataFrame to 'pandas.array'")

    if dtype is None and isinstance(data, (ABCSeries, ABCIndex, ExtensionArray)):
        # Note: we exclude np.ndarray here, will do type inference on it
        dtype = data.dtype

    data = extract_array(data, extract_numpy=True)

    # this returns None for not-found dtypes.
    if dtype is not None:
        dtype = pandas_dtype(dtype)

    if isinstance(data, ExtensionArray) and (dtype is None or data.dtype == dtype):
        # e.g. TimedeltaArray[s], avoid casting to NumpyExtensionArray
        if copy:
            return data.copy()
        return data

    if isinstance(dtype, ExtensionDtype):
        cls = dtype.construct_array_type()
        return cls._from_sequence(data, dtype=dtype, copy=copy)

    if dtype is None:
        inferred_dtype = lib.infer_dtype(data, skipna=True)
        if inferred_dtype == "period":
            period_data = cast(Union[Sequence[Optional[Period]], AnyArrayLike], data)
            return PeriodArray._from_sequence(period_data, copy=copy)

        elif inferred_dtype == "interval":
            return IntervalArray(data, copy=copy)

        elif inferred_dtype.startswith("datetime"):
            # datetime, datetime64
            try:
                return DatetimeArray._from_sequence(data, copy=copy)
            except ValueError:
                # Mixture of timezones, fall back to NumpyExtensionArray
                pass

        elif inferred_dtype.startswith("timedelta"):
            # timedelta, timedelta64
            return TimedeltaArray._from_sequence(data, copy=copy)

        elif inferred_dtype == "string":
            # StringArray/ArrowStringArray depending on pd.options.mode.string_storage
            dtype = StringDtype()
            cls = dtype.construct_array_type()
            return cls._from_sequence(data, dtype=dtype, copy=copy)

        elif inferred_dtype == "integer":
            return IntegerArray._from_sequence(data, copy=copy)
        elif inferred_dtype == "empty" and not hasattr(data, "dtype") and not len(data):
            return FloatingArray._from_sequence(data, copy=copy)
        elif (
            inferred_dtype in ("floating", "mixed-integer-float")
            and getattr(data, "dtype", None) != np.float16
        ):
            # GH#44715 Exclude np.float16 bc FloatingArray does not support it;
            #  we will fall back to NumpyExtensionArray.
            return FloatingArray._from_sequence(data, copy=copy)

        elif inferred_dtype == "boolean":
            return BooleanArray._from_sequence(data, dtype="boolean", copy=copy)

    # Pandas overrides NumPy for
    #   1. datetime64[ns,us,ms,s]
    #   2. timedelta64[ns,us,ms,s]
    # so that a DatetimeArray is returned.
    if lib.is_np_dtype(dtype, "M") and is_supported_dtype(dtype):
        return DatetimeArray._from_sequence(data, dtype=dtype, copy=copy)
    if lib.is_np_dtype(dtype, "m") and is_supported_dtype(dtype):
        return TimedeltaArray._from_sequence(data, dtype=dtype, copy=copy)

    elif lib.is_np_dtype(dtype, "mM"):
        warnings.warn(
            r"datetime64 and timedelta64 dtype resolutions other than "
            r"'s', 'ms', 'us', and 'ns' are deprecated. "
            r"In future releases passing unsupported resolutions will "
            r"raise an exception.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )

    return NumpyExtensionArray._from_sequence(data, dtype=dtype, copy=copy)


_typs = frozenset(
    {
        "index",
        "rangeindex",
        "multiindex",
        "datetimeindex",
        "timedeltaindex",
        "periodindex",
        "categoricalindex",
        "intervalindex",
        "series",
    }
)


@overload
def extract_array(
    obj: Series | Index, extract_numpy: bool = ..., extract_range: bool = ...
) -> ArrayLike:
    ...


@overload
def extract_array(
    obj: T, extract_numpy: bool = ..., extract_range: bool = ...
) -> T | ArrayLike:
    ...


def extract_array(
    obj: T, extract_numpy: bool = False, extract_range: bool = False
) -> T | ArrayLike:
    """
    Extract the ndarray or ExtensionArray from a Series or Index.

    For all other types, `obj` is just returned as is.

    Parameters
    ----------
    obj : object
        For Series / Index, the underlying ExtensionArray is unboxed.

    extract_numpy : bool, default False
        Whether to extract the ndarray from a NumpyExtensionArray.

    extract_range : bool, default False
        If we have a RangeIndex, return range._values if True
        (which is a materialized integer ndarray), otherwise return unchanged.

    Returns
    -------
    arr : object

    Examples
    --------
    >>> extract_array(pd.Series(['a', 'b', 'c'], dtype='category'))
    ['a', 'b', 'c']
    Categories (3, object): ['a', 'b', 'c']

    Other objects like lists, arrays, and DataFrames are just passed through.

    >>> extract_array([1, 2, 3])
    [1, 2, 3]

    For an ndarray-backed Series / Index the ndarray is returned.

    >>> extract_array(pd.Series([1, 2, 3]))
    array([1, 2, 3])

    To extract all the way down to the ndarray, pass ``extract_numpy=True``.

    >>> extract_array(pd.Series([1, 2, 3]), extract_numpy=True)
    array([1, 2, 3])
    """
    typ = getattr(obj, "_typ", None)
    if typ in _typs:
        # i.e. isinstance(obj, (ABCIndex, ABCSeries))
        if typ == "rangeindex":
            if extract_range:
                # error: "T" has no attribute "_values"
                return obj._values  # type: ignore[attr-defined]
            return obj

        # error: "T" has no attribute "_values"
        return obj._values  # type: ignore[attr-defined]

    elif extract_numpy and typ == "npy_extension":
        # i.e. isinstance(obj, ABCNumpyExtensionArray)
        # error: "T" has no attribute "to_numpy"
        return obj.to_numpy()  # type: ignore[attr-defined]

    return obj


def ensure_wrapped_if_datetimelike(arr):
    """
    Wrap datetime64 and timedelta64 ndarrays in DatetimeArray/TimedeltaArray.
    """
    if isinstance(arr, np.ndarray):
        if arr.dtype.kind == "M":
            from pandas.core.arrays import DatetimeArray

            dtype = get_supported_dtype(arr.dtype)
            return DatetimeArray._from_sequence(arr, dtype=dtype)

        elif arr.dtype.kind == "m":
            from pandas.core.arrays import TimedeltaArray

            dtype = get_supported_dtype(arr.dtype)
            return TimedeltaArray._from_sequence(arr, dtype=dtype)

    return arr


def sanitize_masked_array(data: ma.MaskedArray) -> np.ndarray:
    """
    Convert numpy MaskedArray to ensure mask is softened.
    """
    mask = ma.getmaskarray(data)
    if mask.any():
        dtype, fill_value = maybe_promote(data.dtype, np.nan)
        dtype = cast(np.dtype, dtype)
        data = ma.asarray(data.astype(dtype, copy=True))
        data.soften_mask()  # set hardmask False if it was True
        data[mask] = fill_value
    else:
        data = data.copy()
    return data


def sanitize_array(
    data,
    index: Index | None,
    dtype: DtypeObj | None = None,
    copy: bool = False,
    *,
    allow_2d: bool = False,
) -> ArrayLike:
    """
    Sanitize input data to an ndarray or ExtensionArray, copy if specified,
    coerce to the dtype if specified.

    Parameters
    ----------
    data : Any
    index : Index or None, default None
    dtype : np.dtype, ExtensionDtype, or None, default None
    copy : bool, default False
    allow_2d : bool, default False
        If False, raise if we have a 2D Arraylike.

    Returns
    -------
    np.ndarray or ExtensionArray
    """
    original_dtype = dtype
    if isinstance(data, ma.MaskedArray):
        data = sanitize_masked_array(data)

    if isinstance(dtype, NumpyEADtype):
        # Avoid ending up with a NumpyExtensionArray
        dtype = dtype.numpy_dtype

    object_index = False
    if isinstance(data, ABCIndex) and data.dtype == object and dtype is None:
        object_index = True

    # extract ndarray or ExtensionArray, ensure we have no NumpyExtensionArray
    data = extract_array(data, extract_numpy=True, extract_range=True)

    if isinstance(data, np.ndarray) and data.ndim == 0:
        if dtype is None:
            dtype = data.dtype
        data = lib.item_from_zerodim(data)
    elif isinstance(data, range):
        # GH#16804
        data = range_to_ndarray(data)
        copy = False

    if not is_list_like(data):
        if index is None:
            raise ValueError("index must be specified when data is not list-like")
        if isinstance(data, str) and using_string_dtype() and original_dtype is None:
            from pandas.core.arrays.string_ import StringDtype

            dtype = StringDtype(na_value=np.nan)
        data = construct_1d_arraylike_from_scalar(data, len(index), dtype)

        return data

    elif isinstance(data, ABCExtensionArray):
        # it is already ensured above this is not a NumpyExtensionArray
        # Until GH#49309 is fixed this check needs to come before the
        #  ExtensionDtype check
        if dtype is not None:
            subarr = data.astype(dtype, copy=copy)
        elif copy:
            subarr = data.copy()
        else:
            subarr = data

    elif isinstance(dtype, ExtensionDtype):
        # create an extension array from its dtype
        _sanitize_non_ordered(data)
        cls = dtype.construct_array_type()
        if not hasattr(data, "__array__"):
            data = list(data)
        subarr = cls._from_sequence(data, dtype=dtype, copy=copy)

    # GH#846
    elif isinstance(data, np.ndarray):
        if isinstance(data, np.matrix):
            data = data.A

        if dtype is None:
            subarr = data
            if data.dtype == object:
                subarr = maybe_infer_to_datetimelike(data)
                if object_index and using_string_dtype() and is_string_dtype(subarr):
                    # Avoid inference when string option is set
                    subarr = data
            elif data.dtype.kind == "U" and using_string_dtype():
                from pandas.core.arrays.string_ import StringDtype

                dtype = StringDtype(na_value=np.nan)
                subarr = dtype.construct_array_type()._from_sequence(data, dtype=dtype)

            if (
                subarr is data
                or (subarr.dtype == "str" and subarr.dtype.storage == "python")  # type: ignore[union-attr]
            ) and copy:
                subarr = subarr.copy()

        else:
            # we will try to copy by-definition here
            subarr = _try_cast(data, dtype, copy)

    elif hasattr(data, "__array__"):
        # e.g. dask array GH#38645
        if not copy:
            data = np.asarray(data)
        else:
            data = np.array(data, copy=copy)
        return sanitize_array(
            data,
            index=index,
            dtype=dtype,
            copy=False,
            allow_2d=allow_2d,
        )

    else:
        _sanitize_non_ordered(data)
        # materialize e.g. generators, convert e.g. tuples, abc.ValueView
        data = list(data)

        if len(data) == 0 and dtype is None:
            # We default to float64, matching numpy
            subarr = np.array([], dtype=np.float64)

        elif dtype is not None:
            subarr = _try_cast(data, dtype, copy)

        else:
            subarr = maybe_convert_platform(data)
            if subarr.dtype == object:
                subarr = cast(np.ndarray, subarr)
                subarr = maybe_infer_to_datetimelike(subarr)

    subarr = _sanitize_ndim(subarr, data, dtype, index, allow_2d=allow_2d)

    if isinstance(subarr, np.ndarray):
        # at this point we should have dtype be None or subarr.dtype == dtype
        dtype = cast(np.dtype, dtype)
        subarr = _sanitize_str_dtypes(subarr, data, dtype, copy)

    return subarr


def range_to_ndarray(rng: range) -> np.ndarray:
    """
    Cast a range object to ndarray.
    """
    # GH#30171 perf avoid realizing range as a list in np.array
    try:
        arr = np.arange(rng.start, rng.stop, rng.step, dtype="int64")
    except OverflowError:
        # GH#30173 handling for ranges that overflow int64
        if (rng.start >= 0 and rng.step > 0) or (rng.step < 0 <= rng.stop):
            try:
                arr = np.arange(rng.start, rng.stop, rng.step, dtype="uint64")
            except OverflowError:
                arr = construct_1d_object_array_from_listlike(list(rng))
        else:
            arr = construct_1d_object_array_from_listlike(list(rng))
    return arr


def _sanitize_non_ordered(data) -> None:
    """
    Raise only for unordered sets, e.g., not for dict_keys
    """
    if isinstance(data, (set, frozenset)):
        raise TypeError(f"'{type(data).__name__}' type is unordered")


def _sanitize_ndim(
    result: ArrayLike,
    data,
    dtype: DtypeObj | None,
    index: Index | None,
    *,
    allow_2d: bool = False,
) -> ArrayLike:
    """
    Ensure we have a 1-dimensional result array.
    """
    if getattr(result, "ndim", 0) == 0:
        raise ValueError("result should be arraylike with ndim > 0")

    if result.ndim == 1:
        # the result that we want
        result = _maybe_repeat(result, index)

    elif result.ndim > 1:
        if isinstance(data, np.ndarray):
            if allow_2d:
                return result
            raise ValueError(
                f"Data must be 1-dimensional, got ndarray of shape {data.shape} instead"
            )
        if is_object_dtype(dtype) and isinstance(dtype, ExtensionDtype):
            # i.e. NumpyEADtype("O")

            result = com.asarray_tuplesafe(data, dtype=np.dtype("object"))
            cls = dtype.construct_array_type()
            result = cls._from_sequence(result, dtype=dtype)
        else:
            # error: Argument "dtype" to "asarray_tuplesafe" has incompatible type
            # "Union[dtype[Any], ExtensionDtype, None]"; expected "Union[str,
            # dtype[Any], None]"
            result = com.asarray_tuplesafe(data, dtype=dtype)  # type: ignore[arg-type]
    return result


def _sanitize_str_dtypes(
    result: np.ndarray, data, dtype: np.dtype | None, copy: bool
) -> np.ndarray:
    """
    Ensure we have a dtype that is supported by pandas.
    """

    # This is to prevent mixed-type Series getting all casted to
    # NumPy string type, e.g. NaN --> '-1#IND'.
    if issubclass(result.dtype.type, str):
        # GH#16605
        # If not empty convert the data to dtype
        # GH#19853: If data is a scalar, result has already the result
        if not lib.is_scalar(data):
            if not np.all(isna(data)):
                data = np.asarray(data, dtype=dtype)
            if not copy:
                result = np.asarray(data, dtype=object)
            else:
                result = np.array(data, dtype=object, copy=copy)
    return result


def _maybe_repeat(arr: ArrayLike, index: Index | None) -> ArrayLike:
    """
    If we have a length-1 array and an index describing how long we expect
    the result to be, repeat the array.
    """
    if index is not None:
        if 1 == len(arr) != len(index):
            arr = arr.repeat(len(index))
    return arr


def _try_cast(
    arr: list | np.ndarray,
    dtype: np.dtype,
    copy: bool,
) -> ArrayLike:
    """
    Convert input to numpy ndarray and optionally cast to a given dtype.

    Parameters
    ----------
    arr : ndarray or list
        Excludes: ExtensionArray, Series, Index.
    dtype : np.dtype
    copy : bool
        If False, don't copy the data if not needed.

    Returns
    -------
    np.ndarray or ExtensionArray
    """
    is_ndarray = isinstance(arr, np.ndarray)

    if dtype == object:
        if not is_ndarray:
            subarr = construct_1d_object_array_from_listlike(arr)
            return subarr
        return ensure_wrapped_if_datetimelike(arr).astype(dtype, copy=copy)

    elif dtype.kind == "U":
        # TODO: test cases with arr.dtype.kind in "mM"
        if is_ndarray:
            arr = cast(np.ndarray, arr)
            shape = arr.shape
            if arr.ndim > 1:
                arr = arr.ravel()
        else:
            shape = (len(arr),)
        return lib.ensure_string_array(arr, convert_na_value=False, copy=copy).reshape(
            shape
        )

    elif dtype.kind in "mM":
        return maybe_cast_to_datetime(arr, dtype)

    # GH#15832: Check if we are requesting a numeric dtype and
    # that we can convert the data to the requested dtype.
    elif dtype.kind in "iu":
        # this will raise if we have e.g. floats

        subarr = maybe_cast_to_integer_array(arr, dtype)
    elif not copy:
        subarr = np.asarray(arr, dtype=dtype)
    else:
        subarr = np.array(arr, dtype=dtype, copy=copy)

    return subarr
<|endoftext|>"""
Methods that can be shared by many array-like classes or subclasses:
    Series
    Index
    ExtensionArray
"""
from __future__ import annotations

import operator
from typing import Any

import numpy as np

from pandas._libs import lib
from pandas._libs.ops_dispatch import maybe_dispatch_ufunc_to_dunder_op

from pandas.core.dtypes.generic import ABCNDFrame

from pandas.core import roperator
from pandas.core.construction import extract_array
from pandas.core.ops.common import unpack_zerodim_and_defer

REDUCTION_ALIASES = {
    "maximum": "max",
    "minimum": "min",
    "add": "sum",
    "multiply": "prod",
}


class OpsMixin:
    # -------------------------------------------------------------
    # Comparisons

    def _cmp_method(self, other, op):
        return NotImplemented

    @unpack_zerodim_and_defer("__eq__")
    def __eq__(self, other):
        return self._cmp_method(other, operator.eq)

    @unpack_zerodim_and_defer("__ne__")
    def __ne__(self, other):
        return self._cmp_method(other, operator.ne)

    @unpack_zerodim_and_defer("__lt__")
    def __lt__(self, other):
        return self._cmp_method(other, operator.lt)

    @unpack_zerodim_and_defer("__le__")
    def __le__(self, other):
        return self._cmp_method(other, operator.le)

    @unpack_zerodim_and_defer("__gt__")
    def __gt__(self, other):
        return self._cmp_method(other, operator.gt)

    @unpack_zerodim_and_defer("__ge__")
    def __ge__(self, other):
        return self._cmp_method(other, operator.ge)

    # -------------------------------------------------------------
    # Logical Methods

    def _logical_method(self, other, op):
        return NotImplemented

    @unpack_zerodim_and_defer("__and__")
    def __and__(self, other):
        return self._logical_method(other, operator.and_)

    @unpack_zerodim_and_defer("__rand__")
    def __rand__(self, other):
        return self._logical_method(other, roperator.rand_)

    @unpack_zerodim_and_defer("__or__")
    def __or__(self, other):
        return self._logical_method(other, operator.or_)

    @unpack_zerodim_and_defer("__ror__")
    def __ror__(self, other):
        return self._logical_method(other, roperator.ror_)

    @unpack_zerodim_and_defer("__xor__")
    def __xor__(self, other):
        return self._logical_method(other, operator.xor)

    @unpack_zerodim_and_defer("__rxor__")
    def __rxor__(self, other):
        return self._logical_method(other, roperator.rxor)

    # -------------------------------------------------------------
    # Arithmetic Methods

    def _arith_method(self, other, op):
        return NotImplemented

    @unpack_zerodim_and_defer("__add__")
    def __add__(self, other):
        """
        Get Addition of DataFrame and other, column-wise.

        Equivalent to ``DataFrame.add(other)``.

        Parameters
        ----------
        other : scalar, sequence, Series, dict or DataFrame
            Object to be added to the DataFrame.

        Returns
        -------
        DataFrame
            The result of adding ``other`` to DataFrame.

        See Also
        --------
        DataFrame.add : Add a DataFrame and another object, with option for index-
            or column-oriented addition.

        Examples
        --------
        >>> df = pd.DataFrame({'height': [1.5, 2.6], 'weight': [500, 800]},
        ...                   index=['elk', 'moose'])
        >>> df
               height  weight
        elk       1.5     500
        moose     2.6     800

        Adding a scalar affects all rows and columns.

        >>> df[['height', 'weight']] + 1.5
               height  weight
        elk       3.0   501.5
        moose     4.1   801.5

        Each element of a list is added to a column of the DataFrame, in order.

        >>> df[['height', 'weight']] + [0.5, 1.5]
               height  weight
        elk       2.0   501.5
        moose     3.1   801.5

        Keys of a dictionary are aligned to the DataFrame, based on column names;
        each value in the dictionary is added to the corresponding column.

        >>> df[['height', 'weight']] + {'height': 0.5, 'weight': 1.5}
               height  weight
        elk       2.0   501.5
        moose     3.1   801.5

        When `other` is a :class:`Series`, the index of `other` is aligned with the
        columns of the DataFrame.

        >>> s1 = pd.Series([0.5, 1.5], index=['weight', 'height'])
        >>> df[['height', 'weight']] + s1
               height  weight
        elk       3.0   500.5
        moose     4.1   800.5

        Even when the index of `other` is the same as the index of the DataFrame,
        the :class:`Series` will not be reoriented. If index-wise alignment is desired,
        :meth:`DataFrame.add` should be used with `axis='index'`.

        >>> s2 = pd.Series([0.5, 1.5], index=['elk', 'moose'])
        >>> df[['height', 'weight']] + s2
               elk  height  moose  weight
        elk    NaN     NaN    NaN     NaN
        moose  NaN     NaN    NaN     NaN

        >>> df[['height', 'weight']].add(s2, axis='index')
               height  weight
        elk       2.0   500.5
        moose     4.1   801.5

        When `other` is a :class:`DataFrame`, both columns names and the
        index are aligned.

        >>> other = pd.DataFrame({'height': [0.2, 0.4, 0.6]},
        ...                      index=['elk', 'moose', 'deer'])
        >>> df[['height', 'weight']] + other
               height  weight
        deer      NaN     NaN
        elk       1.7     NaN
        moose     3.0     NaN
        """
        return self._arith_method(other, operator.add)

    @unpack_zerodim_and_defer("__radd__")
    def __radd__(self, other):
        return self._arith_method(other, roperator.radd)

    @unpack_zerodim_and_defer("__sub__")
    def __sub__(self, other):
        return self._arith_method(other, operator.sub)

    @unpack_zerodim_and_defer("__rsub__")
    def __rsub__(self, other):
        return self._arith_method(other, roperator.rsub)

    @unpack_zerodim_and_defer("__mul__")
    def __mul__(self, other):
        return self._arith_method(other, operator.mul)

    @unpack_zerodim_and_defer("__rmul__")
    def __rmul__(self, other):
        return self._arith_method(other, roperator.rmul)

    @unpack_zerodim_and_defer("__truediv__")
    def __truediv__(self, other):
        return self._arith_method(other, operator.truediv)

    @unpack_zerodim_and_defer("__rtruediv__")
    def __rtruediv__(self, other):
        return self._arith_method(other, roperator.rtruediv)

    @unpack_zerodim_and_defer("__floordiv__")
    def __floordiv__(self, other):
        return self._arith_method(other, operator.floordiv)

    @unpack_zerodim_and_defer("__rfloordiv")
    def __rfloordiv__(self, other):
        return self._arith_method(other, roperator.rfloordiv)

    @unpack_zerodim_and_defer("__mod__")
    def __mod__(self, other):
        return self._arith_method(other, operator.mod)

    @unpack_zerodim_and_defer("__rmod__")
    def __rmod__(self, other):
        return self._arith_method(other, roperator.rmod)

    @unpack_zerodim_and_defer("__divmod__")
    def __divmod__(self, other):
        return self._arith_method(other, divmod)

    @unpack_zerodim_and_defer("__rdivmod__")
    def __rdivmod__(self, other):
        return self._arith_method(other, roperator.rdivmod)

    @unpack_zerodim_and_defer("__pow__")
    def __pow__(self, other):
        return self._arith_method(other, operator.pow)

    @unpack_zerodim_and_defer("__rpow__")
    def __rpow__(self, other):
        return self._arith_method(other, roperator.rpow)


# -----------------------------------------------------------------------------
# Helpers to implement __array_ufunc__


def array_ufunc(self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any):
    """
    Compatibility with numpy ufuncs.

    See also
    --------
    numpy.org/doc/stable/reference/arrays.classes.html#numpy.class.__array_ufunc__
    """
    from pandas.core.frame import (
        DataFrame,
        Series,
    )
    from pandas.core.generic import NDFrame
    from pandas.core.internals import (
        ArrayManager,
        BlockManager,
    )

    cls = type(self)

    kwargs = _standardize_out_kwarg(**kwargs)

    # for binary ops, use our custom dunder methods
    result = maybe_dispatch_ufunc_to_dunder_op(self, ufunc, method, *inputs, **kwargs)
    if result is not NotImplemented:
        return result

    # Determine if we should defer.
    no_defer = (
        np.ndarray.__array_ufunc__,
        cls.__array_ufunc__,
    )

    for item in inputs:
        higher_priority = (
            hasattr(item, "__array_priority__")
            and item.__array_priority__ > self.__array_priority__
        )
        has_array_ufunc = (
            hasattr(item, "__array_ufunc__")
            and type(item).__array_ufunc__ not in no_defer
            and not isinstance(item, self._HANDLED_TYPES)
        )
        if higher_priority or has_array_ufunc:
            return NotImplemented

    # align all the inputs.
    types = tuple(type(x) for x in inputs)
    alignable = [x for x, t in zip(inputs, types) if issubclass(t, NDFrame)]

    if len(alignable) > 1:
        # This triggers alignment.
        # At the moment, there aren't any ufuncs with more than two inputs
        # so this ends up just being x1.index | x2.index, but we write
        # it to handle *args.
        set_types = set(types)
        if len(set_types) > 1 and {DataFrame, Series}.issubset(set_types):
            # We currently don't handle ufunc(DataFrame, Series)
            # well. Previously this raised an internal ValueError. We might
            # support it someday, so raise a NotImplementedError.
            raise NotImplementedError(
                f"Cannot apply ufunc {ufunc} to mixed DataFrame and Series inputs."
            )
        axes = self.axes
        for obj in alignable[1:]:
            # this relies on the fact that we aren't handling mixed
            # series / frame ufuncs.
            for i, (ax1, ax2) in enumerate(zip(axes, obj.axes)):
                axes[i] = ax1.union(ax2)

        reconstruct_axes = dict(zip(self._AXIS_ORDERS, axes))
        inputs = tuple(
            x.reindex(**reconstruct_axes) if issubclass(t, NDFrame) else x
            for x, t in zip(inputs, types)
        )
    else:
        reconstruct_axes = dict(zip(self._AXIS_ORDERS, self.axes))

    if self.ndim == 1:
        names = [getattr(x, "name") for x in inputs if hasattr(x, "name")]
        name = names[0] if len(set(names)) == 1 else None
        reconstruct_kwargs = {"name": name}
    else:
        reconstruct_kwargs = {}

    def reconstruct(result):
        if ufunc.nout > 1:
            # np.modf, np.frexp, np.divmod
            return tuple(_reconstruct(x) for x in result)

        return _reconstruct(result)

    def _reconstruct(result):
        if lib.is_scalar(result):
            return result

        if result.ndim != self.ndim:
            if method == "outer":
                raise NotImplementedError
            return result
        if isinstance(result, (BlockManager, ArrayManager)):
            # we went through BlockManager.apply e.g. np.sqrt
            result = self._constructor_from_mgr(result, axes=result.axes)
        else:
            # we converted an array, lost our axes
            result = self._constructor(
                result, **reconstruct_axes, **reconstruct_kwargs, copy=False
            )
        # TODO: When we support multiple values in __finalize__, this
        # should pass alignable to `__finalize__` instead of self.
        # Then `np.add(a, b)` would consider attrs from both a and b
        # when a and b are NDFrames.
        if len(alignable) == 1:
            result = result.__finalize__(self)
        return result

    if "out" in kwargs:
        # e.g. test_multiindex_get_loc
        result = dispatch_ufunc_with_out(self, ufunc, method, *inputs, **kwargs)
        return reconstruct(result)

    if method == "reduce":
        # e.g. test.series.test_ufunc.test_reduce
        result = dispatch_reduction_ufunc(self, ufunc, method, *inputs, **kwargs)
        if result is not NotImplemented:
            return result

    # We still get here with kwargs `axis` for e.g. np.maximum.accumulate
    #  and `dtype` and `keepdims` for np.ptp

    if self.ndim > 1 and (len(inputs) > 1 or ufunc.nout > 1):
        # Just give up on preserving types in the complex case.
        # In theory we could preserve them for them.
        # * nout>1 is doable if BlockManager.apply took nout and
        #   returned a Tuple[BlockManager].
        # * len(inputs) > 1 is doable when we know that we have
        #   aligned blocks / dtypes.

        # e.g. my_ufunc, modf, logaddexp, heaviside, subtract, add
        inputs = tuple(np.asarray(x) for x in inputs)
        # Note: we can't use default_array_ufunc here bc reindexing means
        #  that `self` may not be among `inputs`
        result = getattr(ufunc, method)(*inputs, **kwargs)
    elif self.ndim == 1:
        # ufunc(series, ...)
        inputs = tuple(extract_array(x, extract_numpy=True) for x in inputs)
        result = getattr(ufunc, method)(*inputs, **kwargs)
    else:
        # ufunc(dataframe)
        if method == "__call__" and not kwargs:
            # for np.<ufunc>(..) calls
            # kwargs cannot necessarily be handled block-by-block, so only
            # take this path if there are no kwargs
            mgr = inputs[0]._mgr
            result = mgr.apply(getattr(ufunc, method))
        else:
            # otherwise specific ufunc methods (eg np.<ufunc>.accumulate(..))
            # Those can have an axis keyword and thus can't be called block-by-block
            result = default_array_ufunc(inputs[0], ufunc, method, *inputs, **kwargs)
            # e.g. np.negative (only one reached), with "where" and "out" in kwargs

    result = reconstruct(result)
    return result


def _standardize_out_kwarg(**kwargs) -> dict:
    """
    If kwargs contain "out1" and "out2", replace that with a tuple "out"

    np.divmod, np.modf, np.frexp can have either `out=(out1, out2)` or
    `out1=out1, out2=out2)`
    """
    if "out" not in kwargs and "out1" in kwargs and "out2" in kwargs:
        out1 = kwargs.pop("out1")
        out2 = kwargs.pop("out2")
        out = (out1, out2)
        kwargs["out"] = out
    return kwargs


def dispatch_ufunc_with_out(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
    """
    If we have an `out` keyword, then call the ufunc without `out` and then
    set the result into the given `out`.
    """

    # Note: we assume _standardize_out_kwarg has already been called.
    out = kwargs.pop("out")
    where = kwargs.pop("where", None)

    result = getattr(ufunc, method)(*inputs, **kwargs)

    if result is NotImplemented:
        return NotImplemented

    if isinstance(result, tuple):
        # i.e. np.divmod, np.modf, np.frexp
        if not isinstance(out, tuple) or len(out) != len(result):
            raise NotImplementedError

        for arr, res in zip(out, result):
            _assign_where(arr, res, where)

        return out

    if isinstance(out, tuple):
        if len(out) == 1:
            out = out[0]
        else:
            raise NotImplementedError

    _assign_where(out, result, where)
    return out


def _assign_where(out, result, where) -> None:
    """
    Set a ufunc result into 'out', masking with a 'where' argument if necessary.
    """
    if where is None:
        # no 'where' arg passed to ufunc
        out[:] = result
    else:
        np.putmask(out, where, result)


def default_array_ufunc(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
    """
    Fallback to the behavior we would get if we did not define __array_ufunc__.

    Notes
    -----
    We are assuming that `self` is among `inputs`.
    """
    if not any(x is self for x in inputs):
        raise NotImplementedError

    new_inputs = [x if x is not self else np.asarray(x) for x in inputs]

    return getattr(ufunc, method)(*new_inputs, **kwargs)


def dispatch_reduction_ufunc(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
    """
    Dispatch ufunc reductions to self's reduction methods.
    """
    assert method == "reduce"

    if len(inputs) != 1 or inputs[0] is not self:
        return NotImplemented

    if ufunc.__name__ not in REDUCTION_ALIASES:
        return NotImplemented

    method_name = REDUCTION_ALIASES[ufunc.__name__]

    # NB: we are assuming that min/max represent minimum/maximum methods,
    #  which would not be accurate for e.g. Timestamp.min
    if not hasattr(self, method_name):
        return NotImplemented

    if self.ndim > 1:
        if isinstance(self, ABCNDFrame):
            # TODO: test cases where this doesn't hold, i.e. 2D DTA/TDA
            kwargs["numeric_only"] = False

        if "axis" not in kwargs:
            # For DataFrame reductions we don't want the default axis=0
            # Note: np.min is not a ufunc, but uses array_function_dispatch,
            #  so calls DataFrame.min (without ever getting here) with the np.min
            #  default of axis=None, which DataFrame.min catches and changes to axis=0.
            # np.minimum.reduce(df) gets here bc axis is not in kwargs,
            #  so we set axis=0 to match the behaviorof np.minimum.reduce(df.values)
            kwargs["axis"] = 0

    # By default, numpy's reductions do not skip NaNs, so we have to
    #  pass skipna=False
    return getattr(self, method_name)(skipna=False, **kwargs)
<|endoftext|>"""
Reversed Operations not available in the stdlib operator module.
Defining these instead of using lambdas allows us to reference them by name.
"""
from __future__ import annotations

import operator


def radd(left, right):
    return right + left


def rsub(left, right):
    return right - left


def rmul(left, right):
    return right * left


def rdiv(left, right):
    return right / left


def rtruediv(left, right):
    return right / left


def rfloordiv(left, right):
    return right // left


def rmod(left, right):
    # check if right is a string as % is the string
    # formatting operation; this is a TypeError
    # otherwise perform the op
    if isinstance(right, str):
        typ = type(left).__name__
        raise TypeError(f"{typ} cannot perform the operation mod")

    return right % left


def rdivmod(left, right):
    return divmod(right, left)


def rpow(left, right):
    return right**left


def rand_(left, right):
    return operator.and_(right, left)


def ror_(left, right):
    return operator.or_(right, left)


def rxor(left, right):
    return operator.xor(right, left)
<|endoftext|>"""
Module containing utilities for NDFrame.sample() and .GroupBy.sample()
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pandas._libs import lib

from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCSeries,
)

if TYPE_CHECKING:
    from pandas._typing import AxisInt

    from pandas.core.generic import NDFrame


def preprocess_weights(obj: NDFrame, weights, axis: AxisInt) -> np.ndarray:
    """
    Process and validate the `weights` argument to `NDFrame.sample` and
    `.GroupBy.sample`.

    Returns `weights` as an ndarray[np.float64], validated except for normalizing
    weights (because that must be done groupwise in groupby sampling).
    """
    # If a series, align with frame
    if isinstance(weights, ABCSeries):
        weights = weights.reindex(obj.axes[axis])

    # Strings acceptable if a dataframe and axis = 0
    if isinstance(weights, str):
        if isinstance(obj, ABCDataFrame):
            if axis == 0:
                try:
                    weights = obj[weights]
                except KeyError as err:
                    raise KeyError(
                        "String passed to weights not a valid column"
                    ) from err
            else:
                raise ValueError(
                    "Strings can only be passed to "
                    "weights when sampling from rows on "
                    "a DataFrame"
                )
        else:
            raise ValueError(
                "Strings cannot be passed as weights when sampling from a Series."
            )

    if isinstance(obj, ABCSeries):
        func = obj._constructor
    else:
        func = obj._constructor_sliced

    weights = func(weights, dtype="float64")._values

    if len(weights) != obj.shape[axis]:
        raise ValueError("Weights and axis to be sampled must be of same length")

    if lib.has_infs(weights):
        raise ValueError("weight vector may not include `inf` values")

    if (weights < 0).any():
        raise ValueError("weight vector many not include negative values")

    missing = np.isnan(weights)
    if missing.any():
        # Don't modify weights in place
        weights = weights.copy()
        weights[missing] = 0
    return weights


def process_sampling_size(
    n: int | None, frac: float | None, replace: bool
) -> int | None:
    """
    Process and validate the `n` and `frac` arguments to `NDFrame.sample` and
    `.GroupBy.sample`.

    Returns None if `frac` should be used (variable sampling sizes), otherwise returns
    the constant sampling size.
    """
    # If no frac or n, default to n=1.
    if n is None and frac is None:
        n = 1
    elif n is not None and frac is not None:
        raise ValueError("Please enter a value for `frac` OR `n`, not both")
    elif n is not None:
        if n < 0:
            raise ValueError(
                "A negative number of rows requested. Please provide `n` >= 0."
            )
        if n % 1 != 0:
            raise ValueError("Only integers accepted as `n` values")
    else:
        assert frac is not None  # for mypy
        if frac > 1 and not replace:
            raise ValueError(
                "Replace has to be set to `True` when "
                "upsampling the population `frac` > 1."
            )
        if frac < 0:
            raise ValueError(
                "A negative number of rows requested. Please provide `frac` >= 0."
            )

    return n


def sample(
    obj_len: int,
    size: int,
    replace: bool,
    weights: np.ndarray | None,
    random_state: np.random.RandomState | np.random.Generator,
) -> np.ndarray:
    """
    Randomly sample `size` indices in `np.arange(obj_len)`

    Parameters
    ----------
    obj_len : int
        The length of the indices being considered
    size : int
        The number of values to choose
    replace : bool
        Allow or disallow sampling of the same row more than once.
    weights : np.ndarray[np.float64] or None
        If None, equal probability weighting, otherwise weights according
        to the vector normalized
    random_state: np.random.RandomState or np.random.Generator
        State used for the random sampling

    Returns
    -------
    np.ndarray[np.intp]
    """
    if weights is not None:
        weight_sum = weights.sum()
        if weight_sum != 0:
            weights = weights / weight_sum
        else:
            raise ValueError("Invalid weights: weights sum to zero")

    return random_state.choice(obj_len, size=size, replace=replace, p=weights).astype(
        np.intp, copy=False
    )
<|endoftext|>from __future__ import annotations

import functools
import itertools
from typing import (
    Any,
    Callable,
    cast,
)
import warnings

import numpy as np

from pandas._config import get_option

from pandas._libs import (
    NaT,
    NaTType,
    iNaT,
    lib,
)
from pandas._typing import (
    ArrayLike,
    AxisInt,
    CorrelationMethod,
    Dtype,
    DtypeObj,
    F,
    Scalar,
    Shape,
    npt,
)
from pandas.compat._optional import import_optional_dependency
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import (
    is_complex,
    is_float,
    is_float_dtype,
    is_integer,
    is_numeric_dtype,
    is_object_dtype,
    needs_i8_conversion,
    pandas_dtype,
)
from pandas.core.dtypes.missing import (
    isna,
    na_value_for_dtype,
    notna,
)

bn = import_optional_dependency("bottleneck", errors="warn")
_BOTTLENECK_INSTALLED = bn is not None
_USE_BOTTLENECK = False


def set_use_bottleneck(v: bool = True) -> None:
    # set/unset to use bottleneck
    global _USE_BOTTLENECK
    if _BOTTLENECK_INSTALLED:
        _USE_BOTTLENECK = v


set_use_bottleneck(get_option("compute.use_bottleneck"))


class disallow:
    def __init__(self, *dtypes: Dtype) -> None:
        super().__init__()
        self.dtypes = tuple(pandas_dtype(dtype).type for dtype in dtypes)

    def check(self, obj) -> bool:
        return hasattr(obj, "dtype") and issubclass(obj.dtype.type, self.dtypes)

    def __call__(self, f: F) -> F:
        @functools.wraps(f)
        def _f(*args, **kwargs):
            obj_iter = itertools.chain(args, kwargs.values())
            if any(self.check(obj) for obj in obj_iter):
                f_name = f.__name__.replace("nan", "")
                raise TypeError(
                    f"reduction operation '{f_name}' not allowed for this dtype"
                )
            try:
                return f(*args, **kwargs)
            except ValueError as e:
                # we want to transform an object array
                # ValueError message to the more typical TypeError
                # e.g. this is normally a disallowed function on
                # object arrays that contain strings
                if is_object_dtype(args[0]):
                    raise TypeError(e) from e
                raise

        return cast(F, _f)


class bottleneck_switch:
    def __init__(self, name=None, **kwargs) -> None:
        self.name = name
        self.kwargs = kwargs

    def __call__(self, alt: F) -> F:
        bn_name = self.name or alt.__name__

        try:
            bn_func = getattr(bn, bn_name)
        except (AttributeError, NameError):  # pragma: no cover
            bn_func = None

        @functools.wraps(alt)
        def f(
            values: np.ndarray,
            *,
            axis: AxisInt | None = None,
            skipna: bool = True,
            **kwds,
        ):
            if len(self.kwargs) > 0:
                for k, v in self.kwargs.items():
                    if k not in kwds:
                        kwds[k] = v

            if values.size == 0 and kwds.get("min_count") is None:
                # We are empty, returning NA for our type
                # Only applies for the default `min_count` of None
                # since that affects how empty arrays are handled.
                # TODO(GH-18976) update all the nanops methods to
                # correctly handle empty inputs and remove this check.
                # It *may* just be `var`
                return _na_for_min_count(values, axis)

            if _USE_BOTTLENECK and skipna and _bn_ok_dtype(values.dtype, bn_name):
                if kwds.get("mask", None) is None:
                    # `mask` is not recognised by bottleneck, would raise
                    #  TypeError if called
                    kwds.pop("mask", None)
                    result = bn_func(values, axis=axis, **kwds)

                    # prefer to treat inf/-inf as NA, but must compute the func
                    # twice :(
                    if _has_infs(result):
                        result = alt(values, axis=axis, skipna=skipna, **kwds)
                else:
                    result = alt(values, axis=axis, skipna=skipna, **kwds)
            else:
                result = alt(values, axis=axis, skipna=skipna, **kwds)

            return result

        return cast(F, f)


def _bn_ok_dtype(dtype: DtypeObj, name: str) -> bool:
    # Bottleneck chokes on datetime64, PeriodDtype (or and EA)
    if dtype != object and not needs_i8_conversion(dtype):
        # GH 42878
        # Bottleneck uses naive summation leading to O(n) loss of precision
        # unlike numpy which implements pairwise summation, which has O(log(n)) loss
        # crossref: https://github.com/pydata/bottleneck/issues/379

        # GH 15507
        # bottleneck does not properly upcast during the sum
        # so can overflow

        # GH 9422
        # further we also want to preserve NaN when all elements
        # are NaN, unlike bottleneck/numpy which consider this
        # to be 0
        return name not in ["nansum", "nanprod", "nanmean"]
    return False


def _has_infs(result) -> bool:
    if isinstance(result, np.ndarray):
        if result.dtype in ("f8", "f4"):
            # Note: outside of an nanops-specific test, we always have
            #  result.ndim == 1, so there is no risk of this ravel making a copy.
            return lib.has_infs(result.ravel("K"))
    try:
        return np.isinf(result).any()
    except (TypeError, NotImplementedError):
        # if it doesn't support infs, then it can't have infs
        return False


def _get_fill_value(
    dtype: DtypeObj, fill_value: Scalar | None = None, fill_value_typ=None
):
    """return the correct fill value for the dtype of the values"""
    if fill_value is not None:
        return fill_value
    if _na_ok_dtype(dtype):
        if fill_value_typ is None:
            return np.nan
        else:
            if fill_value_typ == "+inf":
                return np.inf
            else:
                return -np.inf
    else:
        if fill_value_typ == "+inf":
            # need the max int here
            return lib.i8max
        else:
            return iNaT


def _maybe_get_mask(
    values: np.ndarray, skipna: bool, mask: npt.NDArray[np.bool_] | None
) -> npt.NDArray[np.bool_] | None:
    """
    Compute a mask if and only if necessary.

    This function will compute a mask iff it is necessary. Otherwise,
    return the provided mask (potentially None) when a mask does not need to be
    computed.

    A mask is never necessary if the values array is of boolean or integer
    dtypes, as these are incapable of storing NaNs. If passing a NaN-capable
    dtype that is interpretable as either boolean or integer data (eg,
    timedelta64), a mask must be provided.

    If the skipna parameter is False, a new mask will not be computed.

    The mask is computed using isna() by default. Setting invert=True selects
    notna() as the masking function.

    Parameters
    ----------
    values : ndarray
        input array to potentially compute mask for
    skipna : bool
        boolean for whether NaNs should be skipped
    mask : Optional[ndarray]
        nan-mask if known

    Returns
    -------
    Optional[np.ndarray[bool]]
    """
    if mask is None:
        if values.dtype.kind in "biu":
            # Boolean data cannot contain nulls, so signal via mask being None
            return None

        if skipna or values.dtype.kind in "mM":
            mask = isna(values)

    return mask


def _get_values(
    values: np.ndarray,
    skipna: bool,
    fill_value: Any = None,
    fill_value_typ: str | None = None,
    mask: npt.NDArray[np.bool_] | None = None,
) -> tuple[np.ndarray, npt.NDArray[np.bool_] | None]:
    """
    Utility to get the values view, mask, dtype, dtype_max, and fill_value.

    If both mask and fill_value/fill_value_typ are not None and skipna is True,
    the values array will be copied.

    For input arrays of boolean or integer dtypes, copies will only occur if a
    precomputed mask, a fill_value/fill_value_typ, and skipna=True are
    provided.

    Parameters
    ----------
    values : ndarray
        input array to potentially compute mask for
    skipna : bool
        boolean for whether NaNs should be skipped
    fill_value : Any
        value to fill NaNs with
    fill_value_typ : str
        Set to '+inf' or '-inf' to handle dtype-specific infinities
    mask : Optional[np.ndarray[bool]]
        nan-mask if known

    Returns
    -------
    values : ndarray
        Potential copy of input value array
    mask : Optional[ndarray[bool]]
        Mask for values, if deemed necessary to compute
    """
    # In _get_values is only called from within nanops, and in all cases
    #  with scalar fill_value.  This guarantee is important for the
    #  np.where call below

    mask = _maybe_get_mask(values, skipna, mask)

    dtype = values.dtype

    datetimelike = False
    if values.dtype.kind in "mM":
        # changing timedelta64/datetime64 to int64 needs to happen after
        #  finding `mask` above
        values = np.asarray(values.view("i8"))
        datetimelike = True

    if skipna and (mask is not None):
        # get our fill value (in case we need to provide an alternative
        # dtype for it)
        fill_value = _get_fill_value(
            dtype, fill_value=fill_value, fill_value_typ=fill_value_typ
        )

        if fill_value is not None:
            if mask.any():
                if datetimelike or _na_ok_dtype(dtype):
                    values = values.copy()
                    np.putmask(values, mask, fill_value)
                else:
                    # np.where will promote if needed
                    values = np.where(~mask, values, fill_value)

    return values, mask


def _get_dtype_max(dtype: np.dtype) -> np.dtype:
    # return a platform independent precision dtype
    dtype_max = dtype
    if dtype.kind in "bi":
        dtype_max = np.dtype(np.int64)
    elif dtype.kind == "u":
        dtype_max = np.dtype(np.uint64)
    elif dtype.kind == "f":
        dtype_max = np.dtype(np.float64)
    return dtype_max


def _na_ok_dtype(dtype: DtypeObj) -> bool:
    if needs_i8_conversion(dtype):
        return False
    return not issubclass(dtype.type, np.integer)


def _wrap_results(result, dtype: np.dtype, fill_value=None):
    """wrap our results if needed"""
    if result is NaT:
        pass

    elif dtype.kind == "M":
        if fill_value is None:
            # GH#24293
            fill_value = iNaT
        if not isinstance(result, np.ndarray):
            assert not isna(fill_value), "Expected non-null fill_value"
            if result == fill_value:
                result = np.nan

            if isna(result):
                result = np.datetime64("NaT", "ns").astype(dtype)
            else:
                result = np.int64(result).view(dtype)
            # retain original unit
            result = result.astype(dtype, copy=False)
        else:
            # If we have float dtype, taking a view will give the wrong result
            result = result.astype(dtype)
    elif dtype.kind == "m":
        if not isinstance(result, np.ndarray):
            if result == fill_value or np.isnan(result):
                result = np.timedelta64("NaT").astype(dtype)

            elif np.fabs(result) > lib.i8max:
                # raise if we have a timedelta64[ns] which is too large
                raise ValueError("overflow in timedelta operation")
            else:
                # return a timedelta64 with the original unit
                result = np.int64(result).astype(dtype, copy=False)

        else:
            result = result.astype("m8[ns]").view(dtype)

    return result


def _datetimelike_compat(func: F) -> F:
    """
    If we have datetime64 or timedelta64 values, ensure we have a correct
    mask before calling the wrapped function, then cast back afterwards.
    """

    @functools.wraps(func)
    def new_func(
        values: np.ndarray,
        *,
        axis: AxisInt | None = None,
        skipna: bool = True,
        mask: npt.NDArray[np.bool_] | None = None,
        **kwargs,
    ):
        orig_values = values

        datetimelike = values.dtype.kind in "mM"
        if datetimelike and mask is None:
            mask = isna(values)

        result = func(values, axis=axis, skipna=skipna, mask=mask, **kwargs)

        if datetimelike:
            result = _wrap_results(result, orig_values.dtype, fill_value=iNaT)
            if not skipna:
                assert mask is not None  # checked above
                result = _mask_datetimelike_result(result, axis, mask, orig_values)

        return result

    return cast(F, new_func)


def _na_for_min_count(values: np.ndarray, axis: AxisInt | None) -> Scalar | np.ndarray:
    """
    Return the missing value for `values`.

    Parameters
    ----------
    values : ndarray
    axis : int or None
        axis for the reduction, required if values.ndim > 1.

    Returns
    -------
    result : scalar or ndarray
        For 1-D values, returns a scalar of the correct missing type.
        For 2-D values, returns a 1-D array where each element is missing.
    """
    # we either return np.nan or pd.NaT
    if values.dtype.kind in "iufcb":
        values = values.astype("float64")
    fill_value = na_value_for_dtype(values.dtype)

    if values.ndim == 1:
        return fill_value
    elif axis is None:
        return fill_value
    else:
        result_shape = values.shape[:axis] + values.shape[axis + 1 :]

        return np.full(result_shape, fill_value, dtype=values.dtype)


def maybe_operate_rowwise(func: F) -> F:
    """
    NumPy operations on C-contiguous ndarrays with axis=1 can be
    very slow if axis 1 >> axis 0.
    Operate row-by-row and concatenate the results.
    """

    @functools.wraps(func)
    def newfunc(values: np.ndarray, *, axis: AxisInt | None = None, **kwargs):
        if (
            axis == 1
            and values.ndim == 2
            and values.flags["C_CONTIGUOUS"]
            # only takes this path for wide arrays (long dataframes), for threshold see
            # https://github.com/pandas-dev/pandas/pull/43311#issuecomment-974891737
            and (values.shape[1] / 1000) > values.shape[0]
            and values.dtype != object
            and values.dtype != bool
        ):
            arrs = list(values)
            if kwargs.get("mask") is not None:
                mask = kwargs.pop("mask")
                results = [
                    func(arrs[i], mask=mask[i], **kwargs) for i in range(len(arrs))
                ]
            else:
                results = [func(x, **kwargs) for x in arrs]
            return np.array(results)

        return func(values, axis=axis, **kwargs)

    return cast(F, newfunc)


def nanany(
    values: np.ndarray,
    *,
    axis: AxisInt | None = None,
    skipna: bool = True,
    mask: npt.NDArray[np.bool_] | None = None,
) -> bool:
    """
    Check if any elements along an axis evaluate to True.

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : bool

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, 2])
    >>> nanops.nanany(s.values)
    True

    >>> from pandas.core import nanops
    >>> s = pd.Series([np.nan])
    >>> nanops.nanany(s.values)
    False
    """
    if values.dtype.kind in "iub" and mask is None:
        # GH#26032 fastpath
        # error: Incompatible return value type (got "Union[bool_, ndarray]",
        # expected "bool")
        return values.any(axis)  # type: ignore[return-value]

    if values.dtype.kind == "M":
        # GH#34479
        warnings.warn(
            "'any' with datetime64 dtypes is deprecated and will raise in a "
            "future version. Use (obj != pd.Timestamp(0)).any() instead.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )

    values, _ = _get_values(values, skipna, fill_value=False, mask=mask)

    # For object type, any won't necessarily return
    # boolean values (numpy/numpy#4352)
    if values.dtype == object:
        values = values.astype(bool)

    # error: Incompatible return value type (got "Union[bool_, ndarray]", expected
    # "bool")
    return values.any(axis)  # type: ignore[return-value]


def nanall(
    values: np.ndarray,
    *,
    axis: AxisInt | None = None,
    skipna: bool = True,
    mask: npt.NDArray[np.bool_] | None = None,
) -> bool:
    """
    Check if all elements along an axis evaluate to True.

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : bool

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, 2, np.nan])
    >>> nanops.nanall(s.values)
    True

    >>> from pandas.core import nanops
    >>> s = pd.Series([1, 0])
    >>> nanops.nanall(s.values)
    False
    """
    if values.dtype.kind in "iub" and mask is None:
        # GH#26032 fastpath
        # error: Incompatible return value type (got "Union[bool_, ndarray]",
        # expected "bool")
        return values.all(axis)  # type: ignore[return-value]

    if values.dtype.kind == "M":
        # GH#34479
        warnings.warn(
            "'all' with datetime64 dtypes is deprecated and will raise in a "
            "future version. Use (obj != pd.Timestamp(0)).all() instead.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )

    values, _ = _get_values(values, skipna, fill_value=True, mask=mask)

    # For object type, all won't necessarily return
    # boolean values (numpy/numpy#4352)
    if values.dtype == object:
        values = values.astype(bool)

    # error: Incompatible return value type (got "Union[bool_, ndarray]", expected
    # "bool")
    return values.all(axis)  # type: ignore[return-value]


@disallow("M8")
@_datetimelike_compat
@maybe_operate_rowwise
def nansum(
    values: np.ndarray,
    *,
    axis: AxisInt | None = None,
    skipna: bool = True,
    min_count: int = 0,
    mask: npt.NDArray[np.bool_] | None = None,
) -> float:
    """
    Sum the elements along an axis ignoring NaNs

    Parameters
    ----------
    values : ndarray[dtype]
    axis : int, optional
    skipna : bool, default True
    min_count: int, default 0
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : dtype

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, 2, np.nan])
    >>> nanops.nansum(s.values)
    3.0
    """
    dtype = values.dtype
    values, mask = _get_values(values, skipna, fill_value=0, mask=mask)
    dtype_sum = _get_dtype_max(dtype)
    if dtype.kind == "f":
        dtype_sum = dtype
    elif dtype.kind == "m":
        dtype_sum = np.dtype(np.float64)

    the_sum = values.sum(axis, dtype=dtype_sum)
    the_sum = _maybe_null_out(the_sum, axis, mask, values.shape, min_count=min_count)

    return the_sum


def _mask_datetimelike_result(
    result: np.ndarray | np.datetime64 | np.timedelta64,
    axis: AxisInt | None,
    mask: npt.NDArray[np.bool_],
    orig_values: np.ndarray,
) -> np.ndarray | np.datetime64 | np.timedelta64 | NaTType:
    if isinstance(result, np.ndarray):
        # we need to apply the mask
        result = result.astype("i8").view(orig_values.dtype)
        axis_mask = mask.any(axis=axis)
        # error: Unsupported target for indexed assignment ("Union[ndarray[Any, Any],
        # datetime64, timedelta64]")
        result[axis_mask] = iNaT  # type: ignore[index]
    else:
        if mask.any():
            return np.int64(iNaT).view(orig_values.dtype)
    return result


@bottleneck_switch()
@_datetimelike_compat
def nanmean(
    values: np.ndarray,
    *,
    axis: AxisInt | None = None,
    skipna: bool = True,
    mask: npt.NDArray[np.bool_] | None = None,
) -> float:
    """
    Compute the mean of the element along an axis ignoring NaNs

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    float
        Unless input is a float array, in which case use the same
        precision as the input array.

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, 2, np.nan])
    >>> nanops.nanmean(s.values)
    1.5
    """
    dtype = values.dtype
    values, mask = _get_values(values, skipna, fill_value=0, mask=mask)
    dtype_sum = _get_dtype_max(dtype)
    dtype_count = np.dtype(np.float64)

    # not using needs_i8_conversion because that includes period
    if dtype.kind in "mM":
        dtype_sum = np.dtype(np.float64)
    elif dtype.kind in "iu":
        dtype_sum = np.dtype(np.float64)
    elif dtype.kind == "f":
        dtype_sum = dtype
        dtype_count = dtype

    count = _get_counts(values.shape, mask, axis, dtype=dtype_count)
    the_sum = values.sum(axis, dtype=dtype_sum)
    the_sum = _ensure_numeric(the_sum)

    if axis is not None and getattr(the_sum, "ndim", False):
        count = cast(np.ndarray, count)
        with np.errstate(all="ignore"):
            # suppress division by zero warnings
            the_mean = the_sum / count
        ct_mask = count == 0
        if ct_mask.any():
            the_mean[ct_mask] = np.nan
    else:
        the_mean = the_sum / count if count > 0 else np.nan

    return the_mean


@bottleneck_switch()
def nanmedian(values, *, axis: AxisInt | None = None, skipna: bool = True, mask=None):
    """
    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : float
        Unless input is a float array, in which case use the same
        precision as the input array.

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, np.nan, 2, 2])
    >>> nanops.nanmedian(s.values)
    2.0
    """
    # for floats without mask, the data already uses NaN as missing value
    # indicator, and `mask` will be calculated from that below -> in those
    # cases we never need to set NaN to the masked values
    using_nan_sentinel = values.dtype.kind == "f" and mask is None

    def get_median(x, _mask=None):
        if _mask is None:
            _mask = notna(x)
        else:
            _mask = ~_mask
        if not skipna and not _mask.all():
            return np.nan
        with warnings.catch_warnings():
            # Suppress RuntimeWarning about All-NaN slice
            warnings.filterwarnings(
                "ignore", "All-NaN slice encountered", RuntimeWarning
            )
            res = np.nanmedian(x[_mask])
        return res

    dtype = values.dtype
    values, mask = _get_values(values, skipna, mask=mask, fill_value=None)
    if values.dtype.kind != "f":
        if values.dtype == object:
            # GH#34671 avoid casting strings to numeric
            inferred = lib.infer_dtype(values)
            if inferred in ["string", "mixed"]:
                raise TypeError(f"Cannot convert {values} to numeric")
        try:
            values = values.astype("f8")
        except ValueError as err:
            # e.g. "could not convert string to float: 'a'"
            raise TypeError(str(err)) from err
    if not using_nan_sentinel and mask is not None:
        if not values.flags.writeable:
            values = values.copy()
        values[mask] = np.nan

    notempty = values.size

    # an array from a frame
    if values.ndim > 1 and axis is not None:
        # there's a non-empty array to apply over otherwise numpy raises
        if notempty:
            if not skipna:
                res = np.apply_along_axis(get_median, axis, values)

            else:
                # fastpath for the skipna case
                with warnings.catch_warnings():
                    # Suppress RuntimeWarning about All-NaN slice
                    warnings.filterwarnings(
                        "ignore", "All-NaN slice encountered", RuntimeWarning
                    )
                    if (values.shape[1] == 1 and axis == 0) or (
                        values.shape[0] == 1 and axis == 1
                    ):
                        # GH52788: fastpath when squeezable, nanmedian for 2D array slow
                        res = np.nanmedian(np.squeeze(values), keepdims=True)
                    else:
                        res = np.nanmedian(values, axis=axis)

        else:
            # must return the correct shape, but median is not defined for the
            # empty set so return nans of shape "everything but the passed axis"
            # since "axis" is where the reduction would occur if we had a nonempty
            # array
            res = _get_empty_reduction_result(values.shape, axis)

    else:
        # otherwise return a scalar value
        res = get_median(values, mask) if notempty else np.nan
    return _wrap_results(res, dtype)


def _get_empty_reduction_result(
    shape: Shape,
    axis: AxisInt,
) -> np.ndarray:
    """
    The result from a reduction on an empty ndarray.

    Parameters
    ----------
    shape : Tuple[int, ...]
    axis : int

    Returns
    -------
    np.ndarray
    """
    shp = np.array(shape)
    dims = np.arange(len(shape))
    ret = np.empty(shp[dims != axis], dtype=np.float64)
    ret.fill(np.nan)
    return ret


def _get_counts_nanvar(
    values_shape: Shape,
    mask: npt.NDArray[np.bool_] | None,
    axis: AxisInt | None,
    ddof: int,
    dtype: np.dtype = np.dtype(np.float64),
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """
    Get the count of non-null values along an axis, accounting
    for degrees of freedom.

    Parameters
    ----------
    values_shape : Tuple[int, ...]
        shape tuple from values ndarray, used if mask is None
    mask : Optional[ndarray[bool]]
        locations in values that should be considered missing
    axis : Optional[int]
        axis to count along
    ddof : int
        degrees of freedom
    dtype : type, optional
        type to use for count

    Returns
    -------
    count : int, np.nan or np.ndarray
    d : int, np.nan or np.ndarray
    """
    count = _get_counts(values_shape, mask, axis, dtype=dtype)
    d = count - dtype.type(ddof)

    # always return NaN, never inf
    if is_float(count):
        if count <= ddof:
            # error: Incompatible types in assignment (expression has type
            # "float", variable has type "Union[floating[Any], ndarray[Any,
            # dtype[floating[Any]]]]")
            count = np.nan  # type: ignore[assignment]
            d = np.nan
    else:
        # count is not narrowed by is_float check
        count = cast(np.ndarray, count)
        mask = count <= ddof
        if mask.any():
            np.putmask(d, mask, np.nan)
            np.putmask(count, mask, np.nan)
    return count, d


@bottleneck_switch(ddof=1)
def nanstd(
    values,
    *,
    axis: AxisInt | None = None,
    skipna: bool = True,
    ddof: int = 1,
    mask=None,
):
    """
    Compute the standard deviation along given axis while ignoring NaNs

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    ddof : int, default 1
        Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
        where N represents the number of elements.
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : float
        Unless input is a float array, in which case use the same
        precision as the input array.

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, np.nan, 2, 3])
    >>> nanops.nanstd(s.values)
    1.0
    """
    if values.dtype == "M8[ns]":
        values = values.view("m8[ns]")

    orig_dtype = values.dtype
    values, mask = _get_values(values, skipna, mask=mask)

    result = np.sqrt(nanvar(values, axis=axis, skipna=skipna, ddof=ddof, mask=mask))
    return _wrap_results(result, orig_dtype)


@disallow("M8", "m8")
@bottleneck_switch(ddof=1)
def nanvar(
    values: np.ndarray,
    *,
    axis: AxisInt | None = None,
    skipna: bool = True,
    ddof: int = 1,
    mask=None,
):
    """
    Compute the variance along given axis while ignoring NaNs

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    ddof : int, default 1
        Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
        where N represents the number of elements.
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : float
        Unless input is a float array, in which case use the same
        precision as the input array.

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, np.nan, 2, 3])
    >>> nanops.nanvar(s.values)
    1.0
    """
    dtype = values.dtype
    mask = _maybe_get_mask(values, skipna, mask)
    if dtype.kind in "iu":
        values = values.astype("f8")
        if mask is not None:
            values[mask] = np.nan

    if values.dtype.kind == "f":
        count, d = _get_counts_nanvar(values.shape, mask, axis, ddof, values.dtype)
    else:
        count, d = _get_counts_nanvar(values.shape, mask, axis, ddof)

    if skipna and mask is not None:
        values = values.copy()
        np.putmask(values, mask, 0)

    # xref GH10242
    # Compute variance via two-pass algorithm, which is stable against
    # cancellation errors and relatively accurate for small numbers of
    # observations.
    #
    # See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    avg = _ensure_numeric(values.sum(axis=axis, dtype=np.float64)) / count
    if axis is not None:
        avg = np.expand_dims(avg, axis)
    sqr = _ensure_numeric((avg - values) ** 2)
    if mask is not None:
        np.putmask(sqr, mask, 0)
    result = sqr.sum(axis=axis, dtype=np.float64) / d

    # Return variance as np.float64 (the datatype used in the accumulator),
    # unless we were dealing with a float array, in which case use the same
    # precision as the original values array.
    if dtype.kind == "f":
        result = result.astype(dtype, copy=False)
    return result


@disallow("M8", "m8")
def nansem(
    values: np.ndarray,
    *,
    axis: AxisInt | None = None,
    skipna: bool = True,
    ddof: int = 1,
    mask: npt.NDArray[np.bool_] | None = None,
) -> float:
    """
    Compute the standard error in the mean along given axis while ignoring NaNs

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    ddof : int, default 1
        Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
        where N represents the number of elements.
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : float64
        Unless input is a float array, in which case use the same
        precision as the input array.

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, np.nan, 2, 3])
    >>> nanops.nansem(s.values)
     0.5773502691896258
    """
    # This checks if non-numeric-like data is passed with numeric_only=False
    # and raises a TypeError otherwise
    nanvar(values, axis=axis, skipna=skipna, ddof=ddof, mask=mask)

    mask = _maybe_get_mask(values, skipna, mask)
    if values.dtype.kind != "f":
        values = values.astype("f8")

    if not skipna and mask is not None and mask.any():
        return np.nan

    count, _ = _get_counts_nanvar(values.shape, mask, axis, ddof, values.dtype)
    var = nanvar(values, axis=axis, skipna=skipna, ddof=ddof, mask=mask)

    return np.sqrt(var) / np.sqrt(count)


def _nanminmax(meth, fill_value_typ):
    @bottleneck_switch(name=f"nan{meth}")
    @_datetimelike_compat
    def reduction(
        values: np.ndarray,
        *,
        axis: AxisInt | None = None,
        skipna: bool = True,
        mask: npt.NDArray[np.bool_] | None = None,
    ):
        if values.size == 0:
            return _na_for_min_count(values, axis)

        values, mask = _get_values(
            values, skipna, fill_value_typ=fill_value_typ, mask=mask
        )
        result = getattr(values, meth)(axis)
        result = _maybe_null_out(result, axis, mask, values.shape)
        return result

    return reduction


nanmin = _nanminmax("min", fill_value_typ="+inf")
nanmax = _nanminmax("max", fill_value_typ="-inf")


def nanargmax(
    values: np.ndarray,
    *,
    axis: AxisInt | None = None,
    skipna: bool = True,
    mask: npt.NDArray[np.bool_] | None = None,
) -> int | np.ndarray:
    """
    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : int or ndarray[int]
        The index/indices  of max value in specified axis or -1 in the NA case

    Examples
    --------
    >>> from pandas.core import nanops
    >>> arr = np.array([1, 2, 3, np.nan, 4])
    >>> nanops.nanargmax(arr)
    4

    >>> arr = np.array(range(12), dtype=np.float64).reshape(4, 3)
    >>> arr[2:, 2] = np.nan
    >>> arr
    array([[ 0.,  1.,  2.],
           [ 3.,  4.,  5.],
           [ 6.,  7., nan],
           [ 9., 10., nan]])
    >>> nanops.nanargmax(arr, axis=1)
    array([2, 2, 1, 1])
    """
    values, mask = _get_values(values, True, fill_value_typ="-inf", mask=mask)
    result = values.argmax(axis)
    # error: Argument 1 to "_maybe_arg_null_out" has incompatible type "Any |
    # signedinteger[Any]"; expected "ndarray[Any, Any]"
    result = _maybe_arg_null_out(result, axis, mask, skipna)  # type: ignore[arg-type]
    return result


def nanargmin(
    values: np.ndarray,
    *,
    axis: AxisInt | None = None,
    skipna: bool = True,
    mask: npt.NDArray[np.bool_] | None = None,
) -> int | np.ndarray:
    """
    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : int or ndarray[int]
        The index/indices of min value in specified axis or -1 in the NA case

    Examples
    --------
    >>> from pandas.core import nanops
    >>> arr = np.array([1, 2, 3, np.nan, 4])
    >>> nanops.nanargmin(arr)
    0

    >>> arr = np.array(range(12), dtype=np.float64).reshape(4, 3)
    >>> arr[2:, 0] = np.nan
    >>> arr
    array([[ 0.,  1.,  2.],
           [ 3.,  4.,  5.],
           [nan,  7.,  8.],
           [nan, 10., 11.]])
    >>> nanops.nanargmin(arr, axis=1)
    array([0, 0, 1, 1])
    """
    values, mask = _get_values(values, True, fill_value_typ="+inf", mask=mask)
    result = values.argmin(axis)
    # error: Argument 1 to "_maybe_arg_null_out" has incompatible type "Any |
    # signedinteger[Any]"; expected "ndarray[Any, Any]"
    result = _maybe_arg_null_out(result, axis, mask, skipna)  # type: ignore[arg-type]
    return result


@disallow("M8", "m8")
@maybe_operate_rowwise
def nanskew(
    values: np.ndarray,
    *,
    axis: AxisInt | None = None,
    skipna: bool = True,
    mask: npt.NDArray[np.bool_] | None = None,
) -> float:
    """
    Compute the sample skewness.

    The statistic computed here is the adjusted Fisher-Pearson standardized
    moment coefficient G1. The algorithm computes this coefficient directly
    from the second and third central moment.

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : float64
        Unless input is a float array, in which case use the same
        precision as the input array.

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, np.nan, 1, 2])
    >>> nanops.nanskew(s.values)
    1.7320508075688787
    """
    mask = _maybe_get_mask(values, skipna, mask)
    if values.dtype.kind != "f":
        values = values.astype("f8")
        count = _get_counts(values.shape, mask, axis)
    else:
        count = _get_counts(values.shape, mask, axis, dtype=values.dtype)

    if skipna and mask is not None:
        values = values.copy()
        np.putmask(values, mask, 0)
    elif not skipna and mask is not None and mask.any():
        return np.nan

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = values.sum(axis, dtype=np.float64) / count
    if axis is not None:
        mean = np.expand_dims(mean, axis)

    adjusted = values - mean
    if skipna and mask is not None:
        np.putmask(adjusted, mask, 0)
    adjusted2 = adjusted**2
    adjusted3 = adjusted2 * adjusted
    m2 = adjusted2.sum(axis, dtype=np.float64)
    m3 = adjusted3.sum(axis, dtype=np.float64)

    # floating point error
    #
    # #18044 in _libs/windows.pyx calc_skew follow this behavior
    # to fix the fperr to treat m2 <1e-14 as zero
    m2 = _zero_out_fperr(m2)
    m3 = _zero_out_fperr(m3)

    with np.errstate(invalid="ignore", divide="ignore"):
        result = (count * (count - 1) ** 0.5 / (count - 2)) * (m3 / m2**1.5)

    dtype = values.dtype
    if dtype.kind == "f":
        result = result.astype(dtype, copy=False)

    if isinstance(result, np.ndarray):
        result = np.where(m2 == 0, 0, result)
        result[count < 3] = np.nan
    else:
        result = dtype.type(0) if m2 == 0 else result
        if count < 3:
            return np.nan

    return result


@disallow("M8", "m8")
@maybe_operate_rowwise
def nankurt(
    values: np.ndarray,
    *,
    axis: AxisInt | None = None,
    skipna: bool = True,
    mask: npt.NDArray[np.bool_] | None = None,
) -> float:
    """
    Compute the sample excess kurtosis

    The statistic computed here is the adjusted Fisher-Pearson standardized
    moment coefficient G2, computed directly from the second and fourth
    central moment.

    Parameters
    ----------
    values : ndarray
    axis : int, optional
    skipna : bool, default True
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    result : float64
        Unless input is a float array, in which case use the same
        precision as the input array.

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, np.nan, 1, 3, 2])
    >>> nanops.nankurt(s.values)
    -1.2892561983471076
    """
    mask = _maybe_get_mask(values, skipna, mask)
    if values.dtype.kind != "f":
        values = values.astype("f8")
        count = _get_counts(values.shape, mask, axis)
    else:
        count = _get_counts(values.shape, mask, axis, dtype=values.dtype)

    if skipna and mask is not None:
        values = values.copy()
        np.putmask(values, mask, 0)
    elif not skipna and mask is not None and mask.any():
        return np.nan

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = values.sum(axis, dtype=np.float64) / count
    if axis is not None:
        mean = np.expand_dims(mean, axis)

    adjusted = values - mean
    if skipna and mask is not None:
        np.putmask(adjusted, mask, 0)
    adjusted2 = adjusted**2
    adjusted4 = adjusted2**2
    m2 = adjusted2.sum(axis, dtype=np.float64)
    m4 = adjusted4.sum(axis, dtype=np.float64)

    with np.errstate(invalid="ignore", divide="ignore"):
        adj = 3 * (count - 1) ** 2 / ((count - 2) * (count - 3))
        numerator = count * (count + 1) * (count - 1) * m4
        denominator = (count - 2) * (count - 3) * m2**2

    # floating point error
    #
    # #18044 in _libs/windows.pyx calc_kurt follow this behavior
    # to fix the fperr to treat denom <1e-14 as zero
    numerator = _zero_out_fperr(numerator)
    denominator = _zero_out_fperr(denominator)

    if not isinstance(denominator, np.ndarray):
        # if ``denom`` is a scalar, check these corner cases first before
        # doing division
        if count < 4:
            return np.nan
        if denominator == 0:
            return values.dtype.type(0)

    with np.errstate(invalid="ignore", divide="ignore"):
        result = numerator / denominator - adj

    dtype = values.dtype
    if dtype.kind == "f":
        result = result.astype(dtype, copy=False)

    if isinstance(result, np.ndarray):
        result = np.where(denominator == 0, 0, result)
        result[count < 4] = np.nan

    return result


@disallow("M8", "m8")
@maybe_operate_rowwise
def nanprod(
    values: np.ndarray,
    *,
    axis: AxisInt | None = None,
    skipna: bool = True,
    min_count: int = 0,
    mask: npt.NDArray[np.bool_] | None = None,
) -> float:
    """
    Parameters
    ----------
    values : ndarray[dtype]
    axis : int, optional
    skipna : bool, default True
    min_count: int, default 0
    mask : ndarray[bool], optional
        nan-mask if known

    Returns
    -------
    Dtype
        The product of all elements on a given axis. ( NaNs are treated as 1)

    Examples
    --------
    >>> from pandas.core import nanops
    >>> s = pd.Series([1, 2, 3, np.nan])
    >>> nanops.nanprod(s.values)
    6.0
    """
    mask = _maybe_get_mask(values, skipna, mask)

    if skipna and mask is not None:
        values = values.copy()
        values[mask] = 1
    result = values.prod(axis)
    # error: Incompatible return value type (got "Union[ndarray, float]", expected
    # "float")
    return _maybe_null_out(  # type: ignore[return-value]
        result, axis, mask, values.shape, min_count=min_count
    )


def _maybe_arg_null_out(
    result: np.ndarray,
    axis: AxisInt | None,
    mask: npt.NDArray[np.bool_] | None,
    skipna: bool,
) -> np.ndarray | int:
    # helper function for nanargmin/nanargmax
    if mask is None:
        return result

    if axis is None or not getattr(result, "ndim", False):
        if skipna:
            if mask.all():
                return -1
        else:
            if mask.any():
                return -1
    else:
        if skipna:
            na_mask = mask.all(axis)
        else:
            na_mask = mask.any(axis)
        if na_mask.any():
            result[na_mask] = -1
    return result


def _get_counts(
    values_shape: Shape,
    mask: npt.NDArray[np.bool_] | None,
    axis: AxisInt | None,
    dtype: np.dtype[np.floating] = np.dtype(np.float64),
) -> np.floating | npt.NDArray[np.floating]:
    """
    Get the count of non-null values along an axis

    Parameters
    ----------
    values_shape : tuple of int
        shape tuple from values ndarray, used if mask is None
    mask : Optional[ndarray[bool]]
        locations in values that should be considered missing
    axis : Optional[int]
        axis to count along
    dtype : type, optional
        type to use for count

    Returns
    -------
    count : scalar or array
    """
    if axis is None:
        if mask is not None:
            n = mask.size - mask.sum()
        else:
            n = np.prod(values_shape)
        return dtype.type(n)

    if mask is not None:
        count = mask.shape[axis] - mask.sum(axis)
    else:
        count = values_shape[axis]

    if is_integer(count):
        return dtype.type(count)
    return count.astype(dtype, copy=False)


def _maybe_null_out(
    result: np.ndarray | float | NaTType,
    axis: AxisInt | None,
    mask: npt.NDArray[np.bool_] | None,
    shape: tuple[int, ...],
    min_count: int = 1,
) -> np.ndarray | float | NaTType:
    """
    Returns
    -------
    Dtype
        The product of all elements on a given axis. ( NaNs are treated as 1)
    """
    if mask is None and min_count == 0:
        # nothing to check; short-circuit
        return result

    if axis is not None and isinstance(result, np.ndarray):
        if mask is not None:
            null_mask = (mask.shape[axis] - mask.sum(axis) - min_count) < 0
        else:
            # we have no nulls, kept mask=None in _maybe_get_mask
            below_count = shape[axis] - min_count < 0
            new_shape = shape[:axis] + shape[axis + 1 :]
            null_mask = np.broadcast_to(below_count, new_shape)

        if np.any(null_mask):
            if is_numeric_dtype(result):
                if np.iscomplexobj(result):
                    result = result.astype("c16")
                elif not is_float_dtype(result):
                    result = result.astype("f8", copy=False)
                result[null_mask] = np.nan
            else:
                # GH12941, use None to auto cast null
                result[null_mask] = None
    elif result is not NaT:
        if check_below_min_count(shape, mask, min_count):
            result_dtype = getattr(result, "dtype", None)
            if is_float_dtype(result_dtype):
                # error: Item "None" of "Optional[Any]" has no attribute "type"
                result = result_dtype.type("nan")  # type: ignore[union-attr]
            else:
                result = np.nan

    return result


def check_below_min_count(
    shape: tuple[int, ...], mask: npt.NDArray[np.bool_] | None, min_count: int
) -> bool:
    """
    Check for the `min_count` keyword. Returns True if below `min_count` (when
    missing value should be returned from the reduction).

    Parameters
    ----------
    shape : tuple
        The shape of the values (`values.shape`).
    mask : ndarray[bool] or None
        Boolean numpy array (typically of same shape as `shape`) or None.
    min_count : int
        Keyword passed through from sum/prod call.

    Returns
    -------
    bool
    """
    if min_count > 0:
        if mask is None:
            # no missing values, only check size
            non_nulls = np.prod(shape)
        else:
            non_nulls = mask.size - mask.sum()
        if non_nulls < min_count:
            return True
    return False


def _zero_out_fperr(arg):
    # #18044 reference this behavior to fix rolling skew/kurt issue
    if isinstance(arg, np.ndarray):
        return np.where(np.abs(arg) < 1e-14, 0, arg)
    else:
        return arg.dtype.type(0) if np.abs(arg) < 1e-14 else arg


@disallow("M8", "m8")
def nancorr(
    a: np.ndarray,
    b: np.ndarray,
    *,
    method: CorrelationMethod = "pearson",
    min_periods: int | None = None,
) -> float:
    """
    a, b: ndarrays
    """
    if len(a) != len(b):
        raise AssertionError("Operands to nancorr must have same size")

    if min_periods is None:
        min_periods = 1

    valid = notna(a) & notna(b)
    if not valid.all():
        a = a[valid]
        b = b[valid]

    if len(a) < min_periods:
        return np.nan

    a = _ensure_numeric(a)
    b = _ensure_numeric(b)

    f = get_corr_func(method)
    return f(a, b)


def get_corr_func(
    method: CorrelationMethod,
) -> Callable[[np.ndarray, np.ndarray], float]:
    if method == "kendall":
        from scipy.stats import kendalltau

        def func(a, b):
            return kendalltau(a, b)[0]

        return func
    elif method == "spearman":
        from scipy.stats import spearmanr

        def func(a, b):
            return spearmanr(a, b)[0]

        return func
    elif method == "pearson":

        def func(a, b):
            return np.corrcoef(a, b)[0, 1]

        return func
    elif callable(method):
        return method

    raise ValueError(
        f"Unknown method '{method}', expected one of "
        "'kendall', 'spearman', 'pearson', or callable"
    )


@disallow("M8", "m8")
def nancov(
    a: np.ndarray,
    b: np.ndarray,
    *,
    min_periods: int | None = None,
    ddof: int | None = 1,
) -> float:
    if len(a) != len(b):
        raise AssertionError("Operands to nancov must have same size")

    if min_periods is None:
        min_periods = 1

    valid = notna(a) & notna(b)
    if not valid.all():
        a = a[valid]
        b = b[valid]

    if len(a) < min_periods:
        return np.nan

    a = _ensure_numeric(a)
    b = _ensure_numeric(b)

    return np.cov(a, b, ddof=ddof)[0, 1]


def _ensure_numeric(x):
    if isinstance(x, np.ndarray):
        if x.dtype.kind in "biu":
            x = x.astype(np.float64)
        elif x.dtype == object:
            inferred = lib.infer_dtype(x)
            if inferred in ["string", "mixed"]:
                # GH#44008, GH#36703 avoid casting e.g. strings to numeric
                raise TypeError(f"Could not convert {x} to numeric")
            try:
                x = x.astype(np.complex128)
            except (TypeError, ValueError):
                try:
                    x = x.astype(np.float64)
                except ValueError as err:
                    # GH#29941 we get here with object arrays containing strs
                    raise TypeError(f"Could not convert {x} to numeric") from err
            else:
                if not np.any(np.imag(x)):
                    x = x.real
    elif not (is_float(x) or is_integer(x) or is_complex(x)):
        if isinstance(x, str):
            # GH#44008, GH#36703 avoid casting e.g. strings to numeric
            raise TypeError(f"Could not convert string '{x}' to numeric")
        try:
            x = float(x)
        except (TypeError, ValueError):
            # e.g. "1+1j" or "foo"
            try:
                x = complex(x)
            except ValueError as err:
                # e.g. "foo"
                raise TypeError(f"Could not convert {x} to numeric") from err
    return x


def na_accum_func(values: ArrayLike, accum_func, *, skipna: bool) -> ArrayLike:
    """
    Cumulative function with skipna support.

    Parameters
    ----------
    values : np.ndarray or ExtensionArray
    accum_func : {np.cumprod, np.maximum.accumulate, np.cumsum, np.minimum.accumulate}
    skipna : bool

    Returns
    -------
    np.ndarray or ExtensionArray
    """
    mask_a, mask_b = {
        np.cumprod: (1.0, np.nan),
        np.maximum.accumulate: (-np.inf, np.nan),
        np.cumsum: (0.0, np.nan),
        np.minimum.accumulate: (np.inf, np.nan),
    }[accum_func]

    # This should go through ea interface
    assert values.dtype.kind not in "mM"

    # We will be applying this function to block values
    if skipna and not issubclass(values.dtype.type, (np.integer, np.bool_)):
        vals = values.copy()
        mask = isna(vals)
        vals[mask] = mask_a
        result = accum_func(vals, axis=0)
        result[mask] = mask_b
    else:
        result = accum_func(values, axis=0)

    return result
<|endoftext|>"""
Misc tools for implementing data structures

Note: pandas.core.common is *not* part of the public API.
"""
from __future__ import annotations

import builtins
from collections import (
    abc,
    defaultdict,
)
from collections.abc import (
    Collection,
    Generator,
    Hashable,
    Iterable,
    Sequence,
)
import contextlib
from functools import partial
import inspect
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    cast,
    overload,
)
import warnings

import numpy as np

from pandas._libs import lib
from pandas.compat.numpy import np_version_gte1p24

from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_integer,
)
from pandas.core.dtypes.generic import (
    ABCExtensionArray,
    ABCIndex,
    ABCMultiIndex,
    ABCSeries,
)
from pandas.core.dtypes.inference import iterable_not_string

if TYPE_CHECKING:
    from pandas._typing import (
        AnyArrayLike,
        ArrayLike,
        NpDtype,
        RandomState,
        T,
    )

    from pandas import Index


def flatten(line):
    """
    Flatten an arbitrarily nested sequence.

    Parameters
    ----------
    line : sequence
        The non string sequence to flatten

    Notes
    -----
    This doesn't consider strings sequences.

    Returns
    -------
    flattened : generator
    """
    for element in line:
        if iterable_not_string(element):
            yield from flatten(element)
        else:
            yield element


def consensus_name_attr(objs):
    name = objs[0].name
    for obj in objs[1:]:
        try:
            if obj.name != name:
                name = None
        except ValueError:
            name = None
    return name


def is_bool_indexer(key: Any) -> bool:
    """
    Check whether `key` is a valid boolean indexer.

    Parameters
    ----------
    key : Any
        Only list-likes may be considered boolean indexers.
        All other types are not considered a boolean indexer.
        For array-like input, boolean ndarrays or ExtensionArrays
        with ``_is_boolean`` set are considered boolean indexers.

    Returns
    -------
    bool
        Whether `key` is a valid boolean indexer.

    Raises
    ------
    ValueError
        When the array is an object-dtype ndarray or ExtensionArray
        and contains missing values.

    See Also
    --------
    check_array_indexer : Check that `key` is a valid array to index,
        and convert to an ndarray.
    """
    if isinstance(
        key, (ABCSeries, np.ndarray, ABCIndex, ABCExtensionArray)
    ) and not isinstance(key, ABCMultiIndex):
        if key.dtype == np.object_:
            key_array = np.asarray(key)

            if not lib.is_bool_array(key_array):
                na_msg = "Cannot mask with non-boolean array containing NA / NaN values"
                if lib.is_bool_array(key_array, skipna=True):
                    # Don't raise on e.g. ["A", "B", np.nan], see
                    #  test_loc_getitem_list_of_labels_categoricalindex_with_na
                    raise ValueError(na_msg)
                return False
            return True
        elif is_bool_dtype(key.dtype):
            return True
    elif isinstance(key, list):
        # check if np.array(key).dtype would be bool
        if len(key) > 0:
            if type(key) is not list:  # noqa: E721
                # GH#42461 cython will raise TypeError if we pass a subclass
                key = list(key)
            return lib.is_bool_list(key)

    return False


def cast_scalar_indexer(val):
    """
    Disallow indexing with a float key, even if that key is a round number.

    Parameters
    ----------
    val : scalar

    Returns
    -------
    outval : scalar
    """
    # assumes lib.is_scalar(val)
    if lib.is_float(val) and val.is_integer():
