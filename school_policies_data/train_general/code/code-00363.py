    return trans + ScaledTranslation(x, y, fig.dpi_scale_trans)
<|endoftext|>"""
A python interface to Adobe Font Metrics Files.

Although a number of other Python implementations exist, and may be more
complete than this, it was decided not to go with them because they were
either:

1) copyrighted or used a non-BSD compatible license
2) had too many dependencies and a free standing lib was needed
3) did more than needed and it was easier to write afresh rather than
   figure out how to get just what was needed.

It is pretty easy to use, and has no external dependencies:

>>> import matplotlib as mpl
>>> from pathlib import Path
>>> afm_path = Path(mpl.get_data_path(), 'fonts', 'afm', 'ptmr8a.afm')
>>>
>>> from matplotlib.afm import AFM
>>> with afm_path.open('rb') as fh:
...     afm = AFM(fh)
>>> afm.string_width_height('What the heck?')
(6220.0, 694)
>>> afm.get_fontname()
'Times-Roman'
>>> afm.get_kern_dist('A', 'f')
0
>>> afm.get_kern_dist('A', 'y')
-92.0
>>> afm.get_bbox_char('!')
[130, -9, 238, 676]

As in the Adobe Font Metrics File Format Specification, all dimensions
are given in units of 1/1000 of the scale factor (point size) of the font
being used.
"""

from collections import namedtuple
import logging
import re

from ._mathtext_data import uni2type1


_log = logging.getLogger(__name__)


def _to_int(x):
    # Some AFM files have floats where we are expecting ints -- there is
    # probably a better way to handle this (support floats, round rather than
    # truncate).  But I don't know what the best approach is now and this
    # change to _to_int should at least prevent Matplotlib from crashing on
    # these.  JDH (2009-11-06)
    return int(float(x))


def _to_float(x):
    # Some AFM files use "," instead of "." as decimal separator -- this
    # shouldn't be ambiguous (unless someone is wicked enough to use "," as
    # thousands separator...).
    if isinstance(x, bytes):
        # Encoding doesn't really matter -- if we have codepoints >127 the call
        # to float() will error anyways.
        x = x.decode('latin-1')
    return float(x.replace(',', '.'))


def _to_str(x):
    return x.decode('utf8')


def _to_list_of_ints(s):
    s = s.replace(b',', b' ')
    return [_to_int(val) for val in s.split()]


def _to_list_of_floats(s):
    return [_to_float(val) for val in s.split()]


def _to_bool(s):
    if s.lower().strip() in (b'false', b'0', b'no'):
        return False
    else:
        return True


def _parse_header(fh):
    """
    Read the font metrics header (up to the char metrics) and returns
    a dictionary mapping *key* to *val*.  *val* will be converted to the
    appropriate python type as necessary; e.g.:

        * 'False'->False
        * '0'->0
        * '-168 -218 1000 898'-> [-168, -218, 1000, 898]

    Dictionary keys are

      StartFontMetrics, FontName, FullName, FamilyName, Weight,
      ItalicAngle, IsFixedPitch, FontBBox, UnderlinePosition,
      UnderlineThickness, Version, Notice, EncodingScheme, CapHeight,
      XHeight, Ascender, Descender, StartCharMetrics
    """
    header_converters = {
        b'StartFontMetrics': _to_float,
        b'FontName': _to_str,
        b'FullName': _to_str,
        b'FamilyName': _to_str,
        b'Weight': _to_str,
        b'ItalicAngle': _to_float,
        b'IsFixedPitch': _to_bool,
        b'FontBBox': _to_list_of_ints,
        b'UnderlinePosition': _to_float,
        b'UnderlineThickness': _to_float,
        b'Version': _to_str,
        # Some AFM files have non-ASCII characters (which are not allowed by
        # the spec).  Given that there is actually no public API to even access
        # this field, just return it as straight bytes.
        b'Notice': lambda x: x,
        b'EncodingScheme': _to_str,
        b'CapHeight': _to_float,  # Is the second version a mistake, or
        b'Capheight': _to_float,  # do some AFM files contain 'Capheight'? -JKS
        b'XHeight': _to_float,
        b'Ascender': _to_float,
        b'Descender': _to_float,
        b'StdHW': _to_float,
        b'StdVW': _to_float,
        b'StartCharMetrics': _to_int,
        b'CharacterSet': _to_str,
        b'Characters': _to_int,
    }
    d = {}
    first_line = True
    for line in fh:
        line = line.rstrip()
        if line.startswith(b'Comment'):
            continue
        lst = line.split(b' ', 1)
        key = lst[0]
        if first_line:
            # AFM spec, Section 4: The StartFontMetrics keyword
            # [followed by a version number] must be the first line in
            # the file, and the EndFontMetrics keyword must be the
            # last non-empty line in the file.  We just check the
            # first header entry.
            if key != b'StartFontMetrics':
                raise RuntimeError('Not an AFM file')
            first_line = False
        if len(lst) == 2:
            val = lst[1]
        else:
            val = b''
        try:
            converter = header_converters[key]
        except KeyError:
            _log.error("Found an unknown keyword in AFM header (was %r)", key)
            continue
        try:
            d[key] = converter(val)
        except ValueError:
            _log.error('Value error parsing header in AFM: %s, %s', key, val)
            continue
        if key == b'StartCharMetrics':
            break
    else:
        raise RuntimeError('Bad parse')
    return d


CharMetrics = namedtuple('CharMetrics', 'width, name, bbox')
CharMetrics.__doc__ = """
    Represents the character metrics of a single character.

    Notes
    -----
    The fields do currently only describe a subset of character metrics
    information defined in the AFM standard.
    """
CharMetrics.width.__doc__ = """The character width (WX)."""
CharMetrics.name.__doc__ = """The character name (N)."""
CharMetrics.bbox.__doc__ = """
    The bbox of the character (B) as a tuple (*llx*, *lly*, *urx*, *ury*)."""


def _parse_char_metrics(fh):
    """
    Parse the given filehandle for character metrics information and return
    the information as dicts.

    It is assumed that the file cursor is on the line behind
    'StartCharMetrics'.

    Returns
    -------
    ascii_d : dict
         A mapping "ASCII num of the character" to `.CharMetrics`.
    name_d : dict
         A mapping "character name" to `.CharMetrics`.

    Notes
    -----
    This function is incomplete per the standard, but thus far parses
    all the sample afm files tried.
    """
    required_keys = {'C', 'WX', 'N', 'B'}

    ascii_d = {}
    name_d = {}
    for line in fh:
        # We are defensively letting values be utf8. The spec requires
        # ascii, but there are non-compliant fonts in circulation
        line = _to_str(line.rstrip())  # Convert from byte-literal
        if line.startswith('EndCharMetrics'):
            return ascii_d, name_d
        # Split the metric line into a dictionary, keyed by metric identifiers
        vals = dict(s.strip().split(' ', 1) for s in line.split(';') if s)
        # There may be other metrics present, but only these are needed
        if not required_keys.issubset(vals):
            raise RuntimeError('Bad char metrics line: %s' % line)
        num = _to_int(vals['C'])
        wx = _to_float(vals['WX'])
        name = vals['N']
        bbox = _to_list_of_floats(vals['B'])
        bbox = list(map(int, bbox))
        metrics = CharMetrics(wx, name, bbox)
        # Workaround: If the character name is 'Euro', give it the
        # corresponding character code, according to WinAnsiEncoding (see PDF
        # Reference).
        if name == 'Euro':
            num = 128
        elif name == 'minus':
            num = ord("\N{MINUS SIGN}")  # 0x2212
        if num != -1:
            ascii_d[num] = metrics
        name_d[name] = metrics
    raise RuntimeError('Bad parse')


def _parse_kern_pairs(fh):
    """
    Return a kern pairs dictionary; keys are (*char1*, *char2*) tuples and
    values are the kern pair value.  For example, a kern pairs line like
    ``KPX A y -50``

    will be represented as::

      d[ ('A', 'y') ] = -50

    """

    line = next(fh)
    if not line.startswith(b'StartKernPairs'):
        raise RuntimeError('Bad start of kern pairs data: %s' % line)

    d = {}
    for line in fh:
        line = line.rstrip()
        if not line:
            continue
        if line.startswith(b'EndKernPairs'):
            next(fh)  # EndKernData
            return d
        vals = line.split()
        if len(vals) != 4 or vals[0] != b'KPX':
            raise RuntimeError('Bad kern pairs line: %s' % line)
        c1, c2, val = _to_str(vals[1]), _to_str(vals[2]), _to_float(vals[3])
        d[(c1, c2)] = val
    raise RuntimeError('Bad kern pairs parse')


CompositePart = namedtuple('CompositePart', 'name, dx, dy')
CompositePart.__doc__ = """
    Represents the information on a composite element of a composite char."""
CompositePart.name.__doc__ = """Name of the part, e.g. 'acute'."""
CompositePart.dx.__doc__ = """x-displacement of the part from the origin."""
CompositePart.dy.__doc__ = """y-displacement of the part from the origin."""


def _parse_composites(fh):
    """
    Parse the given filehandle for composites information return them as a
    dict.

    It is assumed that the file cursor is on the line behind 'StartComposites'.

    Returns
    -------
    dict
        A dict mapping composite character names to a parts list. The parts
        list is a list of `.CompositePart` entries describing the parts of
        the composite.

    Examples
    --------
    A composite definition line::

      CC Aacute 2 ; PCC A 0 0 ; PCC acute 160 170 ;

    will be represented as::

      composites['Aacute'] = [CompositePart(name='A', dx=0, dy=0),
                              CompositePart(name='acute', dx=160, dy=170)]

    """
    composites = {}
    for line in fh:
        line = line.rstrip()
        if not line:
            continue
        if line.startswith(b'EndComposites'):
            return composites
        vals = line.split(b';')
        cc = vals[0].split()
        name, _num_parts = cc[1], _to_int(cc[2])
        pccParts = []
        for s in vals[1:-1]:
            pcc = s.split()
            part = CompositePart(pcc[1], _to_float(pcc[2]), _to_float(pcc[3]))
            pccParts.append(part)
        composites[name] = pccParts

    raise RuntimeError('Bad composites parse')


def _parse_optional(fh):
    """
    Parse the optional fields for kern pair data and composites.

    Returns
    -------
    kern_data : dict
        A dict containing kerning information. May be empty.
        See `._parse_kern_pairs`.
    composites : dict
        A dict containing composite information. May be empty.
        See `._parse_composites`.
    """
    optional = {
        b'StartKernData': _parse_kern_pairs,
        b'StartComposites':  _parse_composites,
        }

    d = {b'StartKernData': {},
         b'StartComposites': {}}
    for line in fh:
        line = line.rstrip()
        if not line:
            continue
        key = line.split()[0]

        if key in optional:
            d[key] = optional[key](fh)

    return d[b'StartKernData'], d[b'StartComposites']


class AFM:

    def __init__(self, fh):
        """Parse the AFM file in file object *fh*."""
        self._header = _parse_header(fh)
        self._metrics, self._metrics_by_name = _parse_char_metrics(fh)
        self._kern, self._composite = _parse_optional(fh)

    def get_bbox_char(self, c, isord=False):
        if not isord:
            c = ord(c)
        return self._metrics[c].bbox

    def string_width_height(self, s):
        """
        Return the string width (including kerning) and string height
        as a (*w*, *h*) tuple.
        """
        if not len(s):
            return 0, 0
        total_width = 0
        namelast = None
        miny = 1e9
        maxy = 0
        for c in s:
            if c == '\n':
                continue
            wx, name, bbox = self._metrics[ord(c)]

            total_width += wx + self._kern.get((namelast, name), 0)
            l, b, w, h = bbox
            miny = min(miny, b)
            maxy = max(maxy, b + h)

            namelast = name

        return total_width, maxy - miny

    def get_str_bbox_and_descent(self, s):
        """Return the string bounding box and the maximal descent."""
        if not len(s):
            return 0, 0, 0, 0, 0
        total_width = 0
        namelast = None
        miny = 1e9
        maxy = 0
        left = 0
        if not isinstance(s, str):
            s = _to_str(s)
        for c in s:
            if c == '\n':
                continue
            name = uni2type1.get(ord(c), f"uni{ord(c):04X}")
            try:
                wx, _, bbox = self._metrics_by_name[name]
            except KeyError:
                name = 'question'
                wx, _, bbox = self._metrics_by_name[name]
            total_width += wx + self._kern.get((namelast, name), 0)
            l, b, w, h = bbox
            left = min(left, l)
            miny = min(miny, b)
            maxy = max(maxy, b + h)

            namelast = name

        return left, miny, total_width, maxy - miny, -miny

    def get_str_bbox(self, s):
        """Return the string bounding box."""
        return self.get_str_bbox_and_descent(s)[:4]

    def get_name_char(self, c, isord=False):
        """Get the name of the character, i.e., ';' is 'semicolon'."""
        if not isord:
            c = ord(c)
        return self._metrics[c].name

    def get_width_char(self, c, isord=False):
        """
        Get the width of the character from the character metric WX field.
        """
        if not isord:
            c = ord(c)
        return self._metrics[c].width

    def get_width_from_char_name(self, name):
        """Get the width of the character from a type1 character name."""
        return self._metrics_by_name[name].width

    def get_height_char(self, c, isord=False):
        """Get the bounding box (ink) height of character *c* (space is 0)."""
        if not isord:
            c = ord(c)
        return self._metrics[c].bbox[-1]

    def get_kern_dist(self, c1, c2):
        """
        Return the kerning pair distance (possibly 0) for chars *c1* and *c2*.
        """
        name1, name2 = self.get_name_char(c1), self.get_name_char(c2)
        return self.get_kern_dist_from_name(name1, name2)

    def get_kern_dist_from_name(self, name1, name2):
        """
        Return the kerning pair distance (possibly 0) for chars
        *name1* and *name2*.
        """
        return self._kern.get((name1, name2), 0)

    def get_fontname(self):
        """Return the font name, e.g., 'Times-Roman'."""
        return self._header[b'FontName']

    @property
    def postscript_name(self):  # For consistency with FT2Font.
        return self.get_fontname()

    def get_fullname(self):
        """Return the font full name, e.g., 'Times-Roman'."""
        name = self._header.get(b'FullName')
        if name is None:  # use FontName as a substitute
            name = self._header[b'FontName']
        return name

    def get_familyname(self):
        """Return the font family name, e.g., 'Times'."""
        name = self._header.get(b'FamilyName')
        if name is not None:
            return name

        # FamilyName not specified so we'll make a guess
        name = self.get_fullname()
        extras = (r'(?i)([ -](regular|plain|italic|oblique|bold|semibold|'
                  r'light|ultralight|extra|condensed))+$')
        return re.sub(extras, '', name)

    @property
    def family_name(self):
        """The font family name, e.g., 'Times'."""
        return self.get_familyname()

    def get_weight(self):
        """Return the font weight, e.g., 'Bold' or 'Roman'."""
        return self._header[b'Weight']

    def get_angle(self):
        """Return the fontangle as float."""
        return self._header[b'ItalicAngle']

    def get_capheight(self):
        """Return the cap height as float."""
        return self._header[b'CapHeight']

    def get_xheight(self):
        """Return the xheight as float."""
        return self._header[b'XHeight']

    def get_underline_thickness(self):
        """Return the underline thickness as float."""
        return self._header[b'UnderlineThickness']

    def get_horizontal_stem_width(self):
        """
        Return the standard horizontal stem width as float, or *None* if
        not specified in AFM file.
        """
        return self._header.get(b'StdHW', None)

    def get_vertical_stem_width(self):
        """
        Return the standard vertical stem width as float, or *None* if
        not specified in AFM file.
        """
        return self._header.get(b'StdVW', None)
<|endoftext|>"""
Matplotlib provides sophisticated date plotting capabilities, standing on the
shoulders of python :mod:`datetime` and the add-on module dateutil_.

By default, Matplotlib uses the units machinery described in
`~matplotlib.units` to convert `datetime.datetime`, and `numpy.datetime64`
objects when plotted on an x- or y-axis. The user does not
need to do anything for dates to be formatted, but dates often have strict
formatting needs, so this module provides many tick locators and formatters.
A basic example using `numpy.datetime64` is::

    import numpy as np

    times = np.arange(np.datetime64('2001-01-02'),
                      np.datetime64('2002-02-03'), np.timedelta64(75, 'm'))
    y = np.random.randn(len(times))

    fig, ax = plt.subplots()
    ax.plot(times, y)

.. seealso::

    - :doc:`/gallery/text_labels_and_annotations/date`
    - :doc:`/gallery/ticks/date_concise_formatter`
    - :doc:`/gallery/ticks/date_demo_convert`

.. _date-format:

Matplotlib date format
----------------------

Matplotlib represents dates using floating point numbers specifying the number
of days since a default epoch of 1970-01-01 UTC; for example,
1970-01-01, 06:00 is the floating point number 0.25. The formatters and
locators require the use of `datetime.datetime` objects, so only dates between
year 0001 and 9999 can be represented.  Microsecond precision
is achievable for (approximately) 70 years on either side of the epoch, and
20 microseconds for the rest of the allowable range of dates (year 0001 to
9999). The epoch can be changed at import time via `.dates.set_epoch` or
:rc:`date.epoch` to other dates if necessary; see
:doc:`/gallery/ticks/date_precision_and_epochs` for a discussion.

.. note::

   Before Matplotlib 3.3, the epoch was 0000-12-31 which lost modern
   microsecond precision and also made the default axis limit of 0 an invalid
   datetime.  In 3.3 the epoch was changed as above.  To convert old
   ordinal floats to the new epoch, users can do::

     new_ordinal = old_ordinal + mdates.date2num(np.datetime64('0000-12-31'))


There are a number of helper functions to convert between :mod:`datetime`
objects and Matplotlib dates:

.. currentmodule:: matplotlib.dates

.. autosummary::
   :nosignatures:

   datestr2num
   date2num
   num2date
   num2timedelta
   drange
   set_epoch
   get_epoch

.. note::

   Like Python's `datetime.datetime`, Matplotlib uses the Gregorian calendar
   for all conversions between dates and floating point numbers. This practice
   is not universal, and calendar differences can cause confusing
   differences between what Python and Matplotlib give as the number of days
   since 0001-01-01 and what other software and databases yield.  For
   example, the US Naval Observatory uses a calendar that switches
   from Julian to Gregorian in October, 1582.  Hence, using their
   calculator, the number of days between 0001-01-01 and 2006-04-01 is
   732403, whereas using the Gregorian calendar via the datetime
   module we find::

     In [1]: date(2006, 4, 1).toordinal() - date(1, 1, 1).toordinal()
     Out[1]: 732401

All the Matplotlib date converters, locators and formatters are timezone aware.
If no explicit timezone is provided, :rc:`timezone` is assumed, provided as a
string.  If you want to use a different timezone, pass the *tz* keyword
argument of `num2date` to any date tick locators or formatters you create. This
can be either a `datetime.tzinfo` instance or a string with the timezone name
that can be parsed by `~dateutil.tz.gettz`.

A wide range of specific and general purpose date tick locators and
formatters are provided in this module.  See
:mod:`matplotlib.ticker` for general information on tick locators
and formatters.  These are described below.

The dateutil_ module provides additional code to handle date ticking, making it
easy to place ticks on any kinds of dates.  See examples below.

.. _dateutil: https://dateutil.readthedocs.io

.. _date-locators:

Date tick locators
------------------

Most of the date tick locators can locate single or multiple ticks. For example::

    # import constants for the days of the week
    from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU

    # tick on Mondays every week
    loc = WeekdayLocator(byweekday=MO, tz=tz)

    # tick on Mondays and Saturdays
    loc = WeekdayLocator(byweekday=(MO, SA))

In addition, most of the constructors take an interval argument::

    # tick on Mondays every second week
    loc = WeekdayLocator(byweekday=MO, interval=2)

The rrule locator allows completely general date ticking::

    # tick every 5th easter
    rule = rrulewrapper(YEARLY, byeaster=1, interval=5)
    loc = RRuleLocator(rule)

The available date tick locators are:

* `MicrosecondLocator`: Locate microseconds.

* `SecondLocator`: Locate seconds.

* `MinuteLocator`: Locate minutes.

* `HourLocator`: Locate hours.

* `DayLocator`: Locate specified days of the month.

* `WeekdayLocator`: Locate days of the week, e.g., MO, TU.

* `MonthLocator`: Locate months, e.g., 7 for July.

* `YearLocator`: Locate years that are multiples of base.

* `RRuleLocator`: Locate using a `rrulewrapper`.
  `rrulewrapper` is a simple wrapper around dateutil_'s `dateutil.rrule`
  which allow almost arbitrary date tick specifications.
  See :doc:`rrule example </gallery/ticks/date_demo_rrule>`.

* `AutoDateLocator`: On autoscale, this class picks the best `DateLocator`
  (e.g., `RRuleLocator`) to set the view limits and the tick locations.  If
  called with ``interval_multiples=True`` it will make ticks line up with
  sensible multiples of the tick intervals.  For example, if the interval is
  4 hours, it will pick hours 0, 4, 8, etc. as ticks.  This behaviour is not
  guaranteed by default.

.. _date-formatters:

Date formatters
---------------

The available date formatters are:

* `AutoDateFormatter`: attempts to figure out the best format to use.  This is
  most useful when used with the `AutoDateLocator`.

* `ConciseDateFormatter`: also attempts to figure out the best format to use,
  and to make the format as compact as possible while still having complete
  date information.  This is most useful when used with the `AutoDateLocator`.

* `DateFormatter`: use `~datetime.datetime.strftime` format strings.
"""

import datetime
import functools
import logging
import re

from dateutil.rrule import (rrule, MO, TU, WE, TH, FR, SA, SU, YEARLY,
                            MONTHLY, WEEKLY, DAILY, HOURLY, MINUTELY,
                            SECONDLY)
from dateutil.relativedelta import relativedelta
import dateutil.parser
import dateutil.tz
import numpy as np

import matplotlib as mpl
from matplotlib import _api, cbook, ticker, units

__all__ = ('datestr2num', 'date2num', 'num2date', 'num2timedelta', 'drange',
           'set_epoch', 'get_epoch', 'DateFormatter', 'ConciseDateFormatter',
           'AutoDateFormatter', 'DateLocator', 'RRuleLocator',
           'AutoDateLocator', 'YearLocator', 'MonthLocator', 'WeekdayLocator',
           'DayLocator', 'HourLocator', 'MinuteLocator',
           'SecondLocator', 'MicrosecondLocator',
           'rrule', 'MO', 'TU', 'WE', 'TH', 'FR', 'SA', 'SU',
           'YEARLY', 'MONTHLY', 'WEEKLY', 'DAILY',
           'HOURLY', 'MINUTELY', 'SECONDLY', 'MICROSECONDLY', 'relativedelta',
           'DateConverter', 'ConciseDateConverter', 'rrulewrapper')


_log = logging.getLogger(__name__)
UTC = datetime.timezone.utc


def _get_tzinfo(tz=None):
    """
    Generate `~datetime.tzinfo` from a string or return `~datetime.tzinfo`.
    If None, retrieve the preferred timezone from the rcParams dictionary.
    """
    tz = mpl._val_or_rc(tz, 'timezone')
    if tz == 'UTC':
        return UTC
    if isinstance(tz, str):
        tzinfo = dateutil.tz.gettz(tz)
        if tzinfo is None:
            raise ValueError(f"{tz} is not a valid timezone as parsed by"
                             " dateutil.tz.gettz.")
        return tzinfo
    if isinstance(tz, datetime.tzinfo):
        return tz
    raise TypeError(f"tz must be string or tzinfo subclass, not {tz!r}.")


# Time-related constants.
EPOCH_OFFSET = float(datetime.datetime(1970, 1, 1).toordinal())
# EPOCH_OFFSET is not used by matplotlib
MICROSECONDLY = SECONDLY + 1
HOURS_PER_DAY = 24.
MIN_PER_HOUR = 60.
SEC_PER_MIN = 60.
MONTHS_PER_YEAR = 12.

DAYS_PER_WEEK = 7.
DAYS_PER_MONTH = 30.
DAYS_PER_YEAR = 365.0

MINUTES_PER_DAY = MIN_PER_HOUR * HOURS_PER_DAY

SEC_PER_HOUR = SEC_PER_MIN * MIN_PER_HOUR
SEC_PER_DAY = SEC_PER_HOUR * HOURS_PER_DAY
SEC_PER_WEEK = SEC_PER_DAY * DAYS_PER_WEEK

MUSECONDS_PER_DAY = 1e6 * SEC_PER_DAY

MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY = (
    MO, TU, WE, TH, FR, SA, SU)
WEEKDAYS = (MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY)

# default epoch: passed to np.datetime64...
_epoch = None


def _reset_epoch_test_example():
    """
    Reset the Matplotlib date epoch so it can be set again.

    Only for use in tests and examples.
    """
    global _epoch
    _epoch = None


def set_epoch(epoch):
    """
    Set the epoch (origin for dates) for datetime calculations.

    The default epoch is :rc:`date.epoch`.

    If microsecond accuracy is desired, the date being plotted needs to be
    within approximately 70 years of the epoch. Matplotlib internally
    represents dates as days since the epoch, so floating point dynamic
    range needs to be within a factor of 2^52.

    `~.dates.set_epoch` must be called before any dates are converted
    (i.e. near the import section) or a RuntimeError will be raised.

    See also :doc:`/gallery/ticks/date_precision_and_epochs`.

    Parameters
    ----------
    epoch : str
        valid UTC date parsable by `numpy.datetime64` (do not include
        timezone).

    """
    global _epoch
    if _epoch is not None:
        raise RuntimeError('set_epoch must be called before dates plotted.')
    _epoch = epoch


def get_epoch():
    """
    Get the epoch used by `.dates`.

    Returns
    -------
    epoch : str
        String for the epoch (parsable by `numpy.datetime64`).
    """
    global _epoch

    _epoch = mpl._val_or_rc(_epoch, 'date.epoch')
    return _epoch


def _dt64_to_ordinalf(d):
    """
    Convert `numpy.datetime64` or an `numpy.ndarray` of those types to
    Gregorian date as UTC float relative to the epoch (see `.get_epoch`).
    Roundoff is float64 precision.  Practically: microseconds for dates
    between 290301 BC, 294241 AD, milliseconds for larger dates
    (see `numpy.datetime64`).
    """

    # the "extra" ensures that we at least allow the dynamic range out to
    # seconds.  That should get out to +/-2e11 years.
    dseconds = d.astype('datetime64[s]')
    extra = (d - dseconds).astype('timedelta64[ns]')
    t0 = np.datetime64(get_epoch(), 's')
    dt = (dseconds - t0).astype(np.float64)
    dt += extra.astype(np.float64) / 1.0e9
    dt = dt / SEC_PER_DAY

    NaT_int = np.datetime64('NaT').astype(np.int64)
    d_int = d.astype(np.int64)
    dt[d_int == NaT_int] = np.nan
    return dt


def _from_ordinalf(x, tz=None):
    """
    Convert Gregorian float of the date, preserving hours, minutes,
    seconds and microseconds.  Return value is a `.datetime`.

    The input date *x* is a float in ordinal days at UTC, and the output will
    be the specified `.datetime` object corresponding to that time in
    timezone *tz*, or if *tz* is ``None``, in the timezone specified in
    :rc:`timezone`.
    """

    tz = _get_tzinfo(tz)

    dt = (np.datetime64(get_epoch()) +
          np.timedelta64(int(np.round(x * MUSECONDS_PER_DAY)), 'us'))
    if dt < np.datetime64('0001-01-01') or dt >= np.datetime64('10000-01-01'):
        raise ValueError(f'Date ordinal {x} converts to {dt} (using '
                         f'epoch {get_epoch()}), but Matplotlib dates must be '
                          'between year 0001 and 9999.')
    # convert from datetime64 to datetime:
    dt = dt.tolist()

    # datetime64 is always UTC:
    dt = dt.replace(tzinfo=dateutil.tz.gettz('UTC'))
    # but maybe we are working in a different timezone so move.
    dt = dt.astimezone(tz)
    # fix round off errors
    if np.abs(x) > 70 * 365:
        # if x is big, round off to nearest twenty microseconds.
        # This avoids floating point roundoff error
        ms = round(dt.microsecond / 20) * 20
        if ms == 1000000:
            dt = dt.replace(microsecond=0) + datetime.timedelta(seconds=1)
        else:
            dt = dt.replace(microsecond=ms)

    return dt


# a version of _from_ordinalf that can operate on numpy arrays
_from_ordinalf_np_vectorized = np.vectorize(_from_ordinalf, otypes="O")
# a version of dateutil.parser.parse that can operate on numpy arrays
_dateutil_parser_parse_np_vectorized = np.vectorize(dateutil.parser.parse)


def datestr2num(d, default=None):
    """
    Convert a date string to a datenum using `dateutil.parser.parse`.

    Parameters
    ----------
    d : str or sequence of str
        The dates to convert.

    default : datetime.datetime, optional
        The default date to use when fields are missing in *d*.
    """
    if isinstance(d, str):
        dt = dateutil.parser.parse(d, default=default)
        return date2num(dt)
    else:
        if default is not None:
            d = [date2num(dateutil.parser.parse(s, default=default))
                 for s in d]
            return np.asarray(d)
        d = np.asarray(d)
        if not d.size:
            return d
        return date2num(_dateutil_parser_parse_np_vectorized(d))


def date2num(d):
    """
    Convert datetime objects to Matplotlib dates.

    Parameters
    ----------
    d : `datetime.datetime` or `numpy.datetime64` or sequences of these

    Returns
    -------
    float or sequence of floats
        Number of days since the epoch.  See `.get_epoch` for the
        epoch, which can be changed by :rc:`date.epoch` or `.set_epoch`.  If
        the epoch is "1970-01-01T00:00:00" (default) then noon Jan 1 1970
        ("1970-01-01T12:00:00") returns 0.5.

    Notes
    -----
    The Gregorian calendar is assumed; this is not universal practice.
    For details see the module docstring.
    """
    # Unpack in case of e.g. Pandas or xarray object
    d = cbook._unpack_to_numpy(d)

    # make an iterable, but save state to unpack later:
    iterable = np.iterable(d)
    if not iterable:
        d = [d]

    masked = np.ma.is_masked(d)
    mask = np.ma.getmask(d)
    d = np.asarray(d)

    # convert to datetime64 arrays, if not already:
    if not np.issubdtype(d.dtype, np.datetime64):
        # datetime arrays
        if not d.size:
            # deals with an empty array...
            return d
        tzi = getattr(d[0], 'tzinfo', None)
        if tzi is not None:
            # make datetime naive:
            d = [dt.astimezone(UTC).replace(tzinfo=None) for dt in d]
            d = np.asarray(d)
        d = d.astype('datetime64[us]')

    d = np.ma.masked_array(d, mask=mask) if masked else d
    d = _dt64_to_ordinalf(d)

    return d if iterable else d[0]


def num2date(x, tz=None):
    """
    Convert Matplotlib dates to `~datetime.datetime` objects.

    Parameters
    ----------
    x : float or sequence of floats
        Number of days (fraction part represents hours, minutes, seconds)
        since the epoch.  See `.get_epoch` for the
        epoch, which can be changed by :rc:`date.epoch` or `.set_epoch`.
    tz : str or `~datetime.tzinfo`, default: :rc:`timezone`
        Timezone of *x*. If a string, *tz* is passed to `dateutil.tz`.

    Returns
    -------
    `~datetime.datetime` or sequence of `~datetime.datetime`
        Dates are returned in timezone *tz*.

        If *x* is a sequence, a sequence of `~datetime.datetime` objects will
        be returned.

    Notes
    -----
    The Gregorian calendar is assumed; this is not universal practice.
    For details, see the module docstring.
    """
    tz = _get_tzinfo(tz)
    return _from_ordinalf_np_vectorized(x, tz).tolist()


_ordinalf_to_timedelta_np_vectorized = np.vectorize(
    lambda x: datetime.timedelta(days=x), otypes="O")


def num2timedelta(x):
    """
    Convert number of days to a `~datetime.timedelta` object.

    If *x* is a sequence, a sequence of `~datetime.timedelta` objects will
    be returned.

    Parameters
    ----------
    x : float, sequence of floats
        Number of days. The fraction part represents hours, minutes, seconds.

    Returns
    -------
    `datetime.timedelta` or list[`datetime.timedelta`]
    """
    return _ordinalf_to_timedelta_np_vectorized(x).tolist()


def drange(dstart, dend, delta):
    """
    Return a sequence of equally spaced Matplotlib dates.

    The dates start at *dstart* and reach up to, but not including *dend*.
    They are spaced by *delta*.

    Parameters
    ----------
    dstart, dend : `~datetime.datetime`
        The date limits.
    delta : `datetime.timedelta`
        Spacing of the dates.

    Returns
    -------
    `numpy.array`
        A list floats representing Matplotlib dates.

    """
    f1 = date2num(dstart)
    f2 = date2num(dend)
    step = delta.total_seconds() / SEC_PER_DAY

    # calculate the difference between dend and dstart in times of delta
    num = int(np.ceil((f2 - f1) / step))

    # calculate end of the interval which will be generated
    dinterval_end = dstart + num * delta

    # ensure, that an half open interval will be generated [dstart, dend)
    if dinterval_end >= dend:
        # if the endpoint is greater than or equal to dend,
        # just subtract one delta
        dinterval_end -= delta
        num -= 1

    f2 = date2num(dinterval_end)  # new float-endpoint
    return np.linspace(f1, f2, num + 1)


def _wrap_in_tex(text):
    p = r'([a-zA-Z]+)'
    ret_text = re.sub(p, r'}$\1$\\mathdefault{', text)

    # Braces ensure symbols are not spaced like binary operators.
    ret_text = ret_text.replace('-', '{-}').replace(':', '{:}')
    # To not concatenate space between numbers.
    ret_text = ret_text.replace(' ', r'\;')
    ret_text = '$\\mathdefault{' + ret_text + '}$'
    ret_text = ret_text.replace('$\\mathdefault{}$', '')
    return ret_text


## date tick locators and formatters ###


class DateFormatter(ticker.Formatter):
    """
    Format a tick (in days since the epoch) with a
    `~datetime.datetime.strftime` format string.
    """

    def __init__(self, fmt, tz=None, *, usetex=None):
        """
        Parameters
        ----------
        fmt : str
            `~datetime.datetime.strftime` format string
        tz : str or `~datetime.tzinfo`, default: :rc:`timezone`
            Ticks timezone. If a string, *tz* is passed to `dateutil.tz`.
        usetex : bool, default: :rc:`text.usetex`
            To enable/disable the use of TeX's math mode for rendering the
            results of the formatter.
        """
        self.tz = _get_tzinfo(tz)
        self.fmt = fmt
        self._usetex = mpl._val_or_rc(usetex, 'text.usetex')

    def __call__(self, x, pos=0):
        result = num2date(x, self.tz).strftime(self.fmt)
        return _wrap_in_tex(result) if self._usetex else result

    def set_tzinfo(self, tz):
        self.tz = _get_tzinfo(tz)


class ConciseDateFormatter(ticker.Formatter):
    """
    A `.Formatter` which attempts to figure out the best format to use for the
    date, and to make it as compact as possible, but still be complete. This is
    most useful when used with the `AutoDateLocator`::

    >>> locator = AutoDateLocator()
    >>> formatter = ConciseDateFormatter(locator)

    Parameters
    ----------
    locator : `.ticker.Locator`
        Locator that this axis is using.

    tz : str or `~datetime.tzinfo`, default: :rc:`timezone`
        Ticks timezone, passed to `.dates.num2date`.

    formats : list of 6 strings, optional
        Format strings for 6 levels of tick labelling: mostly years,
        months, days, hours, minutes, and seconds.  Strings use
        the same format codes as `~datetime.datetime.strftime`.  Default is
        ``['%Y', '%b', '%d', '%H:%M', '%H:%M', '%S.%f']``

    zero_formats : list of 6 strings, optional
        Format strings for tick labels that are "zeros" for a given tick
        level.  For instance, if most ticks are months, ticks around 1 Jan 2005
        will be labeled "Dec", "2005", "Feb".  The default is
        ``['', '%Y', '%b', '%b-%d', '%H:%M', '%H:%M']``

    offset_formats : list of 6 strings, optional
        Format strings for the 6 levels that is applied to the "offset"
        string found on the right side of an x-axis, or top of a y-axis.
        Combined with the tick labels this should completely specify the
        date.  The default is::

            ['', '%Y', '%Y-%b', '%Y-%b-%d', '%Y-%b-%d', '%Y-%b-%d %H:%M']

    show_offset : bool, default: True
        Whether to show the offset or not.

    usetex : bool, default: :rc:`text.usetex`
        To enable/disable the use of TeX's math mode for rendering the results
        of the formatter.

    Examples
    --------
    See :doc:`/gallery/ticks/date_concise_formatter`

    .. plot::

        import datetime
        import matplotlib.dates as mdates

        base = datetime.datetime(2005, 2, 1)
        dates = np.array([base + datetime.timedelta(hours=(2 * i))
                          for i in range(732)])
        N = len(dates)
        np.random.seed(19680801)
        y = np.cumsum(np.random.randn(N))

        fig, ax = plt.subplots(constrained_layout=True)
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        ax.plot(dates, y)
        ax.set_title('Concise Date Formatter')

    """

    def __init__(self, locator, tz=None, formats=None, offset_formats=None,
                 zero_formats=None, show_offset=True, *, usetex=None):
        """
        Autoformat the date labels.  The default format is used to form an
        initial string, and then redundant elements are removed.
        """
        self._locator = locator
        self._tz = tz
        self.defaultfmt = '%Y'
        # there are 6 levels with each level getting a specific format
        # 0: mostly years,  1: months,  2: days,
        # 3: hours, 4: minutes, 5: seconds
        if formats:
            if len(formats) != 6:
                raise ValueError('formats argument must be a list of '
                                 '6 format strings (or None)')
            self.formats = formats
        else:
            self.formats = ['%Y',  # ticks are mostly years
                            '%b',          # ticks are mostly months
                            '%d',          # ticks are mostly days
                            '%H:%M',       # hrs
                            '%H:%M',       # min
                            '%S.%f',       # secs
                            ]
        # fmt for zeros ticks at this level.  These are
        # ticks that should be labeled w/ info the level above.
        # like 1 Jan can just be labelled "Jan".  02:02:00 can
        # just be labeled 02:02.
        if zero_formats:
            if len(zero_formats) != 6:
                raise ValueError('zero_formats argument must be a list of '
                                 '6 format strings (or None)')
            self.zero_formats = zero_formats
        elif formats:
            # use the users formats for the zero tick formats
            self.zero_formats = [''] + self.formats[:-1]
        else:
            # make the defaults a bit nicer:
            self.zero_formats = [''] + self.formats[:-1]
            self.zero_formats[3] = '%b-%d'

        if offset_formats:
            if len(offset_formats) != 6:
                raise ValueError('offset_formats argument must be a list of '
                                 '6 format strings (or None)')
            self.offset_formats = offset_formats
        else:
            self.offset_formats = ['',
                                   '%Y',
                                   '%Y-%b',
                                   '%Y-%b-%d',
                                   '%Y-%b-%d',
                                   '%Y-%b-%d %H:%M']
        self.offset_string = ''
        self.show_offset = show_offset
        self._usetex = mpl._val_or_rc(usetex, 'text.usetex')

    def __call__(self, x, pos=None):
        formatter = DateFormatter(self.defaultfmt, self._tz,
                                  usetex=self._usetex)
        return formatter(x, pos=pos)

    def format_ticks(self, values):
        tickdatetime = [num2date(value, tz=self._tz) for value in values]
        tickdate = np.array([tdt.timetuple()[:6] for tdt in tickdatetime])

        # basic algorithm:
        # 1) only display a part of the date if it changes over the ticks.
        # 2) don't display the smaller part of the date if:
        #    it is always the same or if it is the start of the
        #    year, month, day etc.
        # fmt for most ticks at this level
        fmts = self.formats
        # format beginnings of days, months, years, etc.
        zerofmts = self.zero_formats
        # offset fmt are for the offset in the upper left of the
        # or lower right of the axis.
        offsetfmts = self.offset_formats
        show_offset = self.show_offset

        # determine the level we will label at:
        # mostly 0: years,  1: months,  2: days,
        # 3: hours, 4: minutes, 5: seconds, 6: microseconds
        for level in range(5, -1, -1):
            unique = np.unique(tickdate[:, level])
            if len(unique) > 1:
                # if 1 is included in unique, the year is shown in ticks
                if level < 2 and np.any(unique == 1):
                    show_offset = False
                break
            elif level == 0:
                # all tickdate are the same, so only micros might be different
                # set to the most precise (6: microseconds doesn't exist...)
                level = 5

        # level is the basic level we will label at.
        # now loop through and decide the actual ticklabels
        zerovals = [0, 1, 1, 0, 0, 0, 0]
        labels = [''] * len(tickdate)
        for nn in range(len(tickdate)):
            if level < 5:
                if tickdate[nn][level] == zerovals[level]:
                    fmt = zerofmts[level]
                else:
                    fmt = fmts[level]
            else:
                # special handling for seconds + microseconds
                if (tickdatetime[nn].second == tickdatetime[nn].microsecond
                        == 0):
                    fmt = zerofmts[level]
                else:
                    fmt = fmts[level]
            labels[nn] = tickdatetime[nn].strftime(fmt)

        # special handling of seconds and microseconds:
        # strip extra zeros and decimal if possible.
        # this is complicated by two factors.  1) we have some level-4 strings
        # here (i.e. 03:00, '0.50000', '1.000') 2) we would like to have the
        # same number of decimals for each string (i.e. 0.5 and 1.0).
        if level >= 5:
            trailing_zeros = min(
                (len(s) - len(s.rstrip('0')) for s in labels if '.' in s),
                default=None)
            if trailing_zeros:
                for nn in range(len(labels)):
                    if '.' in labels[nn]:
                        labels[nn] = labels[nn][:-trailing_zeros].rstrip('.')

        if show_offset:
            # set the offset string:
            if (self._locator.axis and
                    self._locator.axis.__name__ in ('xaxis', 'yaxis')
                    and self._locator.axis.get_inverted()):
                self.offset_string = tickdatetime[0].strftime(offsetfmts[level])
            else:
                self.offset_string = tickdatetime[-1].strftime(offsetfmts[level])
            if self._usetex:
                self.offset_string = _wrap_in_tex(self.offset_string)
        else:
            self.offset_string = ''

        if self._usetex:
            return [_wrap_in_tex(l) for l in labels]
        else:
            return labels

    def get_offset(self):
        return self.offset_string

    def format_data_short(self, value):
        return num2date(value, tz=self._tz).strftime('%Y-%m-%d %H:%M:%S')


class AutoDateFormatter(ticker.Formatter):
    """
    A `.Formatter` which attempts to figure out the best format to use.  This
    is most useful when used with the `AutoDateLocator`.

    `.AutoDateFormatter` has a ``.scale`` dictionary that maps tick scales (the
    interval in days between one major tick) to format strings; this dictionary
    defaults to ::

        self.scaled = {
            DAYS_PER_YEAR: rcParams['date.autoformatter.year'],
            DAYS_PER_MONTH: rcParams['date.autoformatter.month'],
            1: rcParams['date.autoformatter.day'],
            1 / HOURS_PER_DAY: rcParams['date.autoformatter.hour'],
            1 / MINUTES_PER_DAY: rcParams['date.autoformatter.minute'],
            1 / SEC_PER_DAY: rcParams['date.autoformatter.second'],
            1 / MUSECONDS_PER_DAY: rcParams['date.autoformatter.microsecond'],
        }

    The formatter uses the format string corresponding to the lowest key in
    the dictionary that is greater or equal to the current scale.  Dictionary
    entries can be customized::

        locator = AutoDateLocator()
        formatter = AutoDateFormatter(locator)
        formatter.scaled[1/(24*60)] = '%M:%S' # only show min and sec

    Custom callables can also be used instead of format strings.  The following
    example shows how to use a custom format function to strip trailing zeros
    from decimal seconds and adds the date to the first ticklabel::

        def my_format_function(x, pos=None):
            x = matplotlib.dates.num2date(x)
            if pos == 0:
                fmt = '%D %H:%M:%S.%f'
            else:
                fmt = '%H:%M:%S.%f'
            label = x.strftime(fmt)
            label = label.rstrip("0")
            label = label.rstrip(".")
            return label

        formatter.scaled[1/(24*60)] = my_format_function
    """

    # This can be improved by providing some user-level direction on
    # how to choose the best format (precedence, etc.).

    # Perhaps a 'struct' that has a field for each time-type where a
    # zero would indicate "don't show" and a number would indicate
    # "show" with some sort of priority.  Same priorities could mean
    # show all with the same priority.

    # Or more simply, perhaps just a format string for each
    # possibility...

    def __init__(self, locator, tz=None, defaultfmt='%Y-%m-%d', *,
                 usetex=None):
        """
        Autoformat the date labels.

        Parameters
        ----------
        locator : `.ticker.Locator`
            Locator that this axis is using.

        tz : str or `~datetime.tzinfo`, default: :rc:`timezone`
            Ticks timezone. If a string, *tz* is passed to `dateutil.tz`.

        defaultfmt : str
            The default format to use if none of the values in ``self.scaled``
            are greater than the unit returned by ``locator._get_unit()``.

        usetex : bool, default: :rc:`text.usetex`
            To enable/disable the use of TeX's math mode for rendering the
            results of the formatter. If any entries in ``self.scaled`` are set
            as functions, then it is up to the customized function to enable or
            disable TeX's math mode itself.
        """
        self._locator = locator
        self._tz = tz
        self.defaultfmt = defaultfmt
        self._formatter = DateFormatter(self.defaultfmt, tz)
        rcParams = mpl.rcParams
        self._usetex = mpl._val_or_rc(usetex, 'text.usetex')
        self.scaled = {
            DAYS_PER_YEAR: rcParams['date.autoformatter.year'],
            DAYS_PER_MONTH: rcParams['date.autoformatter.month'],
            1: rcParams['date.autoformatter.day'],
            1 / HOURS_PER_DAY: rcParams['date.autoformatter.hour'],
            1 / MINUTES_PER_DAY: rcParams['date.autoformatter.minute'],
            1 / SEC_PER_DAY: rcParams['date.autoformatter.second'],
            1 / MUSECONDS_PER_DAY: rcParams['date.autoformatter.microsecond']
        }

    def _set_locator(self, locator):
        self._locator = locator

    def __call__(self, x, pos=None):
        try:
            locator_unit_scale = float(self._locator._get_unit())
        except AttributeError:
            locator_unit_scale = 1
        # Pick the first scale which is greater than the locator unit.
        fmt = next((fmt for scale, fmt in sorted(self.scaled.items())
                    if scale >= locator_unit_scale),
                   self.defaultfmt)

        if isinstance(fmt, str):
            self._formatter = DateFormatter(fmt, self._tz, usetex=self._usetex)
            result = self._formatter(x, pos)
        elif callable(fmt):
            result = fmt(x, pos)
        else:
            raise TypeError(f'Unexpected type passed to {self!r}.')

        return result


class rrulewrapper:
    """
    A simple wrapper around a `dateutil.rrule` allowing flexible
    date tick specifications.
    """
    def __init__(self, freq, tzinfo=None, **kwargs):
        """
        Parameters
        ----------
        freq : {YEARLY, MONTHLY, WEEKLY, DAILY, HOURLY, MINUTELY, SECONDLY}
            Tick frequency. These constants are defined in `dateutil.rrule`,
            but they are accessible from `matplotlib.dates` as well.
        tzinfo : `datetime.tzinfo`, optional
            Time zone information. The default is None.
        **kwargs
            Additional keyword arguments are passed to the `dateutil.rrule`.
        """
        kwargs['freq'] = freq
        self._base_tzinfo = tzinfo

        self._update_rrule(**kwargs)

    def set(self, **kwargs):
        """Set parameters for an existing wrapper."""
        self._construct.update(kwargs)

        self._update_rrule(**self._construct)

    def _update_rrule(self, **kwargs):
        tzinfo = self._base_tzinfo

        # rrule does not play nicely with timezones - especially pytz time
        # zones, it's best to use naive zones and attach timezones once the
        # datetimes are returned
        if 'dtstart' in kwargs:
            dtstart = kwargs['dtstart']
            if dtstart.tzinfo is not None:
                if tzinfo is None:
                    tzinfo = dtstart.tzinfo
                else:
                    dtstart = dtstart.astimezone(tzinfo)

                kwargs['dtstart'] = dtstart.replace(tzinfo=None)

        if 'until' in kwargs:
            until = kwargs['until']
            if until.tzinfo is not None:
                if tzinfo is not None:
                    until = until.astimezone(tzinfo)
                else:
                    raise ValueError('until cannot be aware if dtstart '
                                     'is naive and tzinfo is None')

                kwargs['until'] = until.replace(tzinfo=None)

        self._construct = kwargs.copy()
        self._tzinfo = tzinfo
        self._rrule = rrule(**self._construct)

    def _attach_tzinfo(self, dt, tzinfo):
        # pytz zones are attached by "localizing" the datetime
        if hasattr(tzinfo, 'localize'):
            return tzinfo.localize(dt, is_dst=True)

        return dt.replace(tzinfo=tzinfo)

    def _aware_return_wrapper(self, f, returns_list=False):
        """Decorator function that allows rrule methods to handle tzinfo."""
        # This is only necessary if we're actually attaching a tzinfo
        if self._tzinfo is None:
            return f

        # All datetime arguments must be naive. If they are not naive, they are
        # converted to the _tzinfo zone before dropping the zone.
        def normalize_arg(arg):
            if isinstance(arg, datetime.datetime) and arg.tzinfo is not None:
                if arg.tzinfo is not self._tzinfo:
                    arg = arg.astimezone(self._tzinfo)

                return arg.replace(tzinfo=None)

            return arg

        def normalize_args(args, kwargs):
            args = tuple(normalize_arg(arg) for arg in args)
            kwargs = {kw: normalize_arg(arg) for kw, arg in kwargs.items()}

            return args, kwargs

        # There are two kinds of functions we care about - ones that return
        # dates and ones that return lists of dates.
        if not returns_list:
            def inner_func(*args, **kwargs):
                args, kwargs = normalize_args(args, kwargs)
                dt = f(*args, **kwargs)
                return self._attach_tzinfo(dt, self._tzinfo)
        else:
            def inner_func(*args, **kwargs):
                args, kwargs = normalize_args(args, kwargs)
                dts = f(*args, **kwargs)
                return [self._attach_tzinfo(dt, self._tzinfo) for dt in dts]

        return functools.wraps(f)(inner_func)

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]

        f = getattr(self._rrule, name)

        if name in {'after', 'before'}:
            return self._aware_return_wrapper(f)
        elif name in {'xafter', 'xbefore', 'between'}:
            return self._aware_return_wrapper(f, returns_list=True)
        else:
            return f

    def __setstate__(self, state):
        self.__dict__.update(state)


class DateLocator(ticker.Locator):
    """
    Determines the tick locations when plotting dates.

    This class is subclassed by other Locators and
    is not meant to be used on its own.
    """
    hms0d = {'byhour': 0, 'byminute': 0, 'bysecond': 0}

    def __init__(self, tz=None):
        """
        Parameters
        ----------
        tz : str or `~datetime.tzinfo`, default: :rc:`timezone`
            Ticks timezone. If a string, *tz* is passed to `dateutil.tz`.
        """
        self.tz = _get_tzinfo(tz)

    def set_tzinfo(self, tz):
        """
        Set timezone info.

        Parameters
        ----------
        tz : str or `~datetime.tzinfo`, default: :rc:`timezone`
            Ticks timezone. If a string, *tz* is passed to `dateutil.tz`.
        """
        self.tz = _get_tzinfo(tz)

    def datalim_to_dt(self):
        """Convert axis data interval to datetime objects."""
        dmin, dmax = self.axis.get_data_interval()
        if dmin > dmax:
            dmin, dmax = dmax, dmin

        return num2date(dmin, self.tz), num2date(dmax, self.tz)

    def viewlim_to_dt(self):
        """Convert the view interval to datetime objects."""
        vmin, vmax = self.axis.get_view_interval()
        if vmin > vmax:
            vmin, vmax = vmax, vmin
        return num2date(vmin, self.tz), num2date(vmax, self.tz)

    def _get_unit(self):
        """
        Return how many days a unit of the locator is; used for
        intelligent autoscaling.
        """
        return 1

    def _get_interval(self):
        """
        Return the number of units for each tick.
        """
        return 1

    def nonsingular(self, vmin, vmax):
        """
        Given the proposed upper and lower extent, adjust the range
        if it is too close to being singular (i.e. a range of ~0).
        """
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            # Except if there is no data, then use 1970 as default.
            return (date2num(datetime.date(1970, 1, 1)),
                    date2num(datetime.date(1970, 1, 2)))
        if vmax < vmin:
            vmin, vmax = vmax, vmin
        unit = self._get_unit()
        interval = self._get_interval()
        if abs(vmax - vmin) < 1e-6:
            vmin -= 2 * unit * interval
            vmax += 2 * unit * interval
        return vmin, vmax


class RRuleLocator(DateLocator):
    # use the dateutil rrule instance

    def __init__(self, o, tz=None):
        super().__init__(tz)
        self.rule = o

    def __call__(self):
        # if no data have been set, this will tank with a ValueError
        try:
            dmin, dmax = self.viewlim_to_dt()
        except ValueError:
            return []

        return self.tick_values(dmin, dmax)

    def tick_values(self, vmin, vmax):
        start, stop = self._create_rrule(vmin, vmax)
        dates = self.rule.between(start, stop, True)
        if len(dates) == 0:
            return date2num([vmin, vmax])
        return self.raise_if_exceeds(date2num(dates))

    def _create_rrule(self, vmin, vmax):
        # set appropriate rrule dtstart and until and return
        # start and end
        delta = relativedelta(vmax, vmin)

        # We need to cap at the endpoints of valid datetime
        try:
            start = vmin - delta
        except (ValueError, OverflowError):
            # cap
            start = datetime.datetime(1, 1, 1, 0, 0, 0,
                                      tzinfo=datetime.timezone.utc)

        try:
            stop = vmax + delta
        except (ValueError, OverflowError):
            # cap
            stop = datetime.datetime(9999, 12, 31, 23, 59, 59,
                                     tzinfo=datetime.timezone.utc)

        self.rule.set(dtstart=start, until=stop)

        return vmin, vmax

    def _get_unit(self):
        # docstring inherited
        freq = self.rule._rrule._freq
        return self.get_unit_generic(freq)

    @staticmethod
    def get_unit_generic(freq):
        if freq == YEARLY:
            return DAYS_PER_YEAR
        elif freq == MONTHLY:
            return DAYS_PER_MONTH
        elif freq == WEEKLY:
            return DAYS_PER_WEEK
        elif freq == DAILY:
            return 1.0
        elif freq == HOURLY:
            return 1.0 / HOURS_PER_DAY
        elif freq == MINUTELY:
            return 1.0 / MINUTES_PER_DAY
        elif freq == SECONDLY:
            return 1.0 / SEC_PER_DAY
        else:
            # error
            return -1  # or should this just return '1'?

    def _get_interval(self):
        return self.rule._rrule._interval


class AutoDateLocator(DateLocator):
    """
    On autoscale, this class picks the best `DateLocator` to set the view
    limits and the tick locations.

    Attributes
    ----------
    intervald : dict

        Mapping of tick frequencies to multiples allowed for that ticking.
        The default is ::

            self.intervald = {
                YEARLY  : [1, 2, 4, 5, 10, 20, 40, 50, 100, 200, 400, 500,
                           1000, 2000, 4000, 5000, 10000],
                MONTHLY : [1, 2, 3, 4, 6],
                DAILY   : [1, 2, 3, 7, 14, 21],
                HOURLY  : [1, 2, 3, 4, 6, 12],
                MINUTELY: [1, 5, 10, 15, 30],
                SECONDLY: [1, 5, 10, 15, 30],
                MICROSECONDLY: [1, 2, 5, 10, 20, 50, 100, 200, 500,
                                1000, 2000, 5000, 10000, 20000, 50000,
                                100000, 200000, 500000, 1000000],
            }

        where the keys are defined in `dateutil.rrule`.

        The interval is used to specify multiples that are appropriate for
        the frequency of ticking. For instance, every 7 days is sensible
        for daily ticks, but for minutes/seconds, 15 or 30 make sense.

        When customizing, you should only modify the values for the existing
        keys. You should not add or delete entries.

        Example for forcing ticks every 3 hours::

            locator = AutoDateLocator()
            locator.intervald[HOURLY] = [3]  # only show every 3 hours
    """

    def __init__(self, tz=None, minticks=5, maxticks=None,
                 interval_multiples=True):
        """
        Parameters
        ----------
        tz : str or `~datetime.tzinfo`, default: :rc:`timezone`
            Ticks timezone. If a string, *tz* is passed to `dateutil.tz`.
        minticks : int
            The minimum number of ticks desired; controls whether ticks occur
            yearly, monthly, etc.
        maxticks : int
            The maximum number of ticks desired; controls the interval between
            ticks (ticking every other, every 3, etc.).  For fine-grained
            control, this can be a dictionary mapping individual rrule
            frequency constants (YEARLY, MONTHLY, etc.) to their own maximum
            number of ticks.  This can be used to keep the number of ticks
            appropriate to the format chosen in `AutoDateFormatter`. Any
            frequency not specified in this dictionary is given a default
            value.
        interval_multiples : bool, default: True
            Whether ticks should be chosen to be multiple of the interval,
            locking them to 'nicer' locations.  For example, this will force
            the ticks to be at hours 0, 6, 12, 18 when hourly ticking is done
            at 6 hour intervals.
        """
        super().__init__(tz=tz)
        self._freq = YEARLY
        self._freqs = [YEARLY, MONTHLY, DAILY, HOURLY, MINUTELY,
                       SECONDLY, MICROSECONDLY]
        self.minticks = minticks

        self.maxticks = {YEARLY: 11, MONTHLY: 12, DAILY: 11, HOURLY: 12,
                         MINUTELY: 11, SECONDLY: 11, MICROSECONDLY: 8}
        if maxticks is not None:
            try:
                self.maxticks.update(maxticks)
            except TypeError:
                # Assume we were given an integer. Use this as the maximum
                # number of ticks for every frequency and create a
                # dictionary for this
                self.maxticks = dict.fromkeys(self._freqs, maxticks)
        self.interval_multiples = interval_multiples
        self.intervald = {
            YEARLY:   [1, 2, 4, 5, 10, 20, 40, 50, 100, 200, 400, 500,
                       1000, 2000, 4000, 5000, 10000],
            MONTHLY:  [1, 2, 3, 4, 6],
            DAILY:    [1, 2, 3, 7, 14, 21],
            HOURLY:   [1, 2, 3, 4, 6, 12],
            MINUTELY: [1, 5, 10, 15, 30],
            SECONDLY: [1, 5, 10, 15, 30],
            MICROSECONDLY: [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000,
                            5000, 10000, 20000, 50000, 100000, 200000, 500000,
                            1000000],
                            }
        if interval_multiples:
            # Swap "3" for "4" in the DAILY list; If we use 3 we get bad
            # tick loc for months w/ 31 days: 1, 4, ..., 28, 31, 1
            # If we use 4 then we get: 1, 5, ... 25, 29, 1
            self.intervald[DAILY] = [1, 2, 4, 7, 14]

        self._byranges = [None, range(1, 13), range(1, 32),
                          range(0, 24), range(0, 60), range(0, 60), None]

    def __call__(self):
        # docstring inherited
        dmin, dmax = self.viewlim_to_dt()
        locator = self.get_locator(dmin, dmax)
        return locator()

    def tick_values(self, vmin, vmax):
        return self.get_locator(vmin, vmax).tick_values(vmin, vmax)

    def nonsingular(self, vmin, vmax):
        # whatever is thrown at us, we can scale the unit.
        # But default nonsingular date plots at an ~4 year period.
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            # Except if there is no data, then use 1970 as default.
            return (date2num(datetime.date(1970, 1, 1)),
                    date2num(datetime.date(1970, 1, 2)))
        if vmax < vmin:
            vmin, vmax = vmax, vmin
        if vmin == vmax:
            vmin = vmin - DAYS_PER_YEAR * 2
            vmax = vmax + DAYS_PER_YEAR * 2
        return vmin, vmax

    def _get_unit(self):
        if self._freq in [MICROSECONDLY]:
            return 1. / MUSECONDS_PER_DAY
        else:
            return RRuleLocator.get_unit_generic(self._freq)

    def get_locator(self, dmin, dmax):
        """Pick the best locator based on a distance."""
        delta = relativedelta(dmax, dmin)
        tdelta = dmax - dmin

        # take absolute difference
        if dmin > dmax:
            delta = -delta
            tdelta = -tdelta
        # The following uses a mix of calls to relativedelta and timedelta
        # methods because there is incomplete overlap in the functionality of
        # these similar functions, and it's best to avoid doing our own math
        # whenever possible.
        numYears = float(delta.years)
        numMonths = numYears * MONTHS_PER_YEAR + delta.months
        numDays = tdelta.days  # Avoids estimates of days/month, days/year.
        numHours = numDays * HOURS_PER_DAY + delta.hours
        numMinutes = numHours * MIN_PER_HOUR + delta.minutes
        numSeconds = np.floor(tdelta.total_seconds())
        numMicroseconds = np.floor(tdelta.total_seconds() * 1e6)

        nums = [numYears, numMonths, numDays, numHours, numMinutes,
                numSeconds, numMicroseconds]

        use_rrule_locator = [True] * 6 + [False]

        # Default setting of bymonth, etc. to pass to rrule
        # [unused (for year), bymonth, bymonthday, byhour, byminute,
        #  bysecond, unused (for microseconds)]
        byranges = [None, 1, 1, 0, 0, 0, None]

        # Loop over all the frequencies and try to find one that gives at
        # least a minticks tick positions.  Once this is found, look for
        # an interval from a list specific to that frequency that gives no
        # more than maxticks tick positions. Also, set up some ranges
        # (bymonth, etc.) as appropriate to be passed to rrulewrapper.
        for i, (freq, num) in enumerate(zip(self._freqs, nums)):
            # If this particular frequency doesn't give enough ticks, continue
            if num < self.minticks:
                # Since we're not using this particular frequency, set
                # the corresponding by_ to None so the rrule can act as
                # appropriate
                byranges[i] = None
                continue

            # Find the first available interval that doesn't give too many
            # ticks
            for interval in self.intervald[freq]:
                if num <= interval * (self.maxticks[freq] - 1):
                    break
            else:
                if not (self.interval_multiples and freq == DAILY):
                    _api.warn_external(
                        f"AutoDateLocator was unable to pick an appropriate "
                        f"interval for this date range. It may be necessary "
                        f"to add an interval value to the AutoDateLocator's "
                        f"intervald dictionary. Defaulting to {interval}.")

            # Set some parameters as appropriate
            self._freq = freq

            if self._byranges[i] and self.interval_multiples:
                byranges[i] = self._byranges[i][::interval]
                if i in (DAILY, WEEKLY):
                    if interval == 14:
                        # just make first and 15th.  Avoids 30th.
                        byranges[i] = [1, 15]
                    elif interval == 7:
                        byranges[i] = [1, 8, 15, 22]

                interval = 1
            else:
                byranges[i] = self._byranges[i]
            break
        else:
            interval = 1

        if (freq == YEARLY) and self.interval_multiples:
            locator = YearLocator(interval, tz=self.tz)
        elif use_rrule_locator[i]:
            _, bymonth, bymonthday, byhour, byminute, bysecond, _ = byranges
            rrule = rrulewrapper(self._freq, interval=interval,
                                 dtstart=dmin, until=dmax,
                                 bymonth=bymonth, bymonthday=bymonthday,
                                 byhour=byhour, byminute=byminute,
                                 bysecond=bysecond)

            locator = RRuleLocator(rrule, tz=self.tz)
        else:
            locator = MicrosecondLocator(interval, tz=self.tz)
            if date2num(dmin) > 70 * 365 and interval < 1000:
                _api.warn_external(
                    'Plotting microsecond time intervals for dates far from '
                    f'the epoch (time origin: {get_epoch()}) is not well-'
                    'supported. See matplotlib.dates.set_epoch to change the '
                    'epoch.')

        locator.set_axis(self.axis)
        return locator


class YearLocator(RRuleLocator):
    """
    Make ticks on a given day of each year that is a multiple of base.

    Examples::

      # Tick every year on Jan 1st
      locator = YearLocator()

      # Tick every 5 years on July 4th
      locator = YearLocator(5, month=7, day=4)
    """
    def __init__(self, base=1, month=1, day=1, tz=None):
        """
        Parameters
        ----------
        base : int, default: 1
            Mark ticks every *base* years.
        month : int, default: 1
            The month on which to place the ticks, starting from 1. Default is
            January.
        day : int, default: 1
            The day on which to place the ticks.
        tz : str or `~datetime.tzinfo`, default: :rc:`timezone`
            Ticks timezone. If a string, *tz* is passed to `dateutil.tz`.
        """
        rule = rrulewrapper(YEARLY, interval=base, bymonth=month,
                            bymonthday=day, **self.hms0d)
        super().__init__(rule, tz=tz)
        self.base = ticker._Edge_integer(base, 0)

    def _create_rrule(self, vmin, vmax):
        # 'start' needs to be a multiple of the interval to create ticks on
        # interval multiples when the tick frequency is YEARLY
        ymin = max(self.base.le(vmin.year) * self.base.step, 1)
        ymax = min(self.base.ge(vmax.year) * self.base.step, 9999)

        c = self.rule._construct
        replace = {'year': ymin,
                   'month': c.get('bymonth', 1),
                   'day': c.get('bymonthday', 1),
                   'hour': 0, 'minute': 0, 'second': 0}

        start = vmin.replace(**replace)
        stop = start.replace(year=ymax)
        self.rule.set(dtstart=start, until=stop)

        return start, stop


class MonthLocator(RRuleLocator):
    """
    Make ticks on occurrences of each month, e.g., 1, 3, 12.
    """
    def __init__(self, bymonth=None, bymonthday=1, interval=1, tz=None):
        """
        Parameters
        ----------
        bymonth : int or list of int, default: all months
            Ticks will be placed on every month in *bymonth*. Default is
            ``range(1, 13)``, i.e. every month.
        bymonthday : int, default: 1
            The day on which to place the ticks.
        interval : int, default: 1
            The interval between each iteration. For example, if
            ``interval=2``, mark every second occurrence.
        tz : str or `~datetime.tzinfo`, default: :rc:`timezone`
            Ticks timezone. If a string, *tz* is passed to `dateutil.tz`.
        """
        if bymonth is None:
            bymonth = range(1, 13)

        rule = rrulewrapper(MONTHLY, bymonth=bymonth, bymonthday=bymonthday,
                            interval=interval, **self.hms0d)
        super().__init__(rule, tz=tz)


class WeekdayLocator(RRuleLocator):
    """
    Make ticks on occurrences of each weekday.
    """

    def __init__(self, byweekday=1, interval=1, tz=None):
        """
        Parameters
        ----------
        byweekday : int or list of int, default: all days
            Ticks will be placed on every weekday in *byweekday*. Default is
            every day.

            Elements of *byweekday* must be one of MO, TU, WE, TH, FR, SA,
            SU, the constants from :mod:`dateutil.rrule`, which have been
            imported into the :mod:`matplotlib.dates` namespace.
        interval : int, default: 1
            The interval between each iteration. For example, if
            ``interval=2``, mark every second occurrence.
        tz : str or `~datetime.tzinfo`, default: :rc:`timezone`
            Ticks timezone. If a string, *tz* is passed to `dateutil.tz`.
        """
        rule = rrulewrapper(DAILY, byweekday=byweekday,
                            interval=interval, **self.hms0d)
        super().__init__(rule, tz=tz)


class DayLocator(RRuleLocator):
    """
    Make ticks on occurrences of each day of the month.  For example,
    1, 15, 30.
    """
    def __init__(self, bymonthday=None, interval=1, tz=None):
        """
        Parameters
        ----------
        bymonthday : int or list of int, default: all days
            Ticks will be placed on every day in *bymonthday*. Default is
            ``bymonthday=range(1, 32)``, i.e., every day of the month.
        interval : int, default: 1
            The interval between each iteration. For example, if
            ``interval=2``, mark every second occurrence.
        tz : str or `~datetime.tzinfo`, default: :rc:`timezone`
            Ticks timezone. If a string, *tz* is passed to `dateutil.tz`.
        """
        if interval != int(interval) or interval < 1:
            raise ValueError("interval must be an integer greater than 0")
        if bymonthday is None:
            bymonthday = range(1, 32)

        rule = rrulewrapper(DAILY, bymonthday=bymonthday,
                            interval=interval, **self.hms0d)
        super().__init__(rule, tz=tz)


class HourLocator(RRuleLocator):
    """
    Make ticks on occurrences of each hour.
    """
    def __init__(self, byhour=None, interval=1, tz=None):
        """
        Parameters
        ----------
        byhour : int or list of int, default: all hours
            Ticks will be placed on every hour in *byhour*. Default is
            ``byhour=range(24)``, i.e., every hour.
        interval : int, default: 1
            The interval between each iteration. For example, if
            ``interval=2``, mark every second occurrence.
        tz : str or `~datetime.tzinfo`, default: :rc:`timezone`
            Ticks timezone. If a string, *tz* is passed to `dateutil.tz`.
        """
        if byhour is None:
            byhour = range(24)

        rule = rrulewrapper(HOURLY, byhour=byhour, interval=interval,
                            byminute=0, bysecond=0)
        super().__init__(rule, tz=tz)


class MinuteLocator(RRuleLocator):
    """
    Make ticks on occurrences of each minute.
    """
    def __init__(self, byminute=None, interval=1, tz=None):
        """
        Parameters
        ----------
        byminute : int or list of int, default: all minutes
            Ticks will be placed on every minute in *byminute*. Default is
            ``byminute=range(60)``, i.e., every minute.
        interval : int, default: 1
            The interval between each iteration. For example, if
            ``interval=2``, mark every second occurrence.
        tz : str or `~datetime.tzinfo`, default: :rc:`timezone`
            Ticks timezone. If a string, *tz* is passed to `dateutil.tz`.
        """
        if byminute is None:
            byminute = range(60)

        rule = rrulewrapper(MINUTELY, byminute=byminute, interval=interval,
                            bysecond=0)
        super().__init__(rule, tz=tz)


class SecondLocator(RRuleLocator):
    """
    Make ticks on occurrences of each second.
    """
    def __init__(self, bysecond=None, interval=1, tz=None):
        """
        Parameters
        ----------
        bysecond : int or list of int, default: all seconds
            Ticks will be placed on every second in *bysecond*. Default is
            ``bysecond = range(60)``, i.e., every second.
        interval : int, default: 1
            The interval between each iteration. For example, if
            ``interval=2``, mark every second occurrence.
        tz : str or `~datetime.tzinfo`, default: :rc:`timezone`
            Ticks timezone. If a string, *tz* is passed to `dateutil.tz`.
        """
        if bysecond is None:
            bysecond = range(60)

        rule = rrulewrapper(SECONDLY, bysecond=bysecond, interval=interval)
        super().__init__(rule, tz=tz)


class MicrosecondLocator(DateLocator):
    """
    Make ticks on regular intervals of one or more microsecond(s).

    .. note::

        By default, Matplotlib uses a floating point representation of time in
        days since the epoch, so plotting data with
        microsecond time resolution does not work well for
        dates that are far (about 70 years) from the epoch (check with
        `~.dates.get_epoch`).

        If you want sub-microsecond resolution time plots, it is strongly
        recommended to use floating point seconds, not datetime-like
        time representation.

        If you really must use datetime.datetime() or similar and still
        need microsecond precision, change the time origin via
        `.dates.set_epoch` to something closer to the dates being plotted.
        See :doc:`/gallery/ticks/date_precision_and_epochs`.

    """
    def __init__(self, interval=1, tz=None):
        """
        Parameters
        ----------
        interval : int, default: 1
            The interval between each iteration. For example, if
            ``interval=2``, mark every second occurrence.
        tz : str or `~datetime.tzinfo`, default: :rc:`timezone`
            Ticks timezone. If a string, *tz* is passed to `dateutil.tz`.
        """
        super().__init__(tz=tz)
        self._interval = interval
        self._wrapped_locator = ticker.MultipleLocator(interval)

    def set_axis(self, axis):
        self._wrapped_locator.set_axis(axis)
        return super().set_axis(axis)

    def __call__(self):
        # if no data have been set, this will tank with a ValueError
        try:
            dmin, dmax = self.viewlim_to_dt()
        except ValueError:
            return []

        return self.tick_values(dmin, dmax)

    def tick_values(self, vmin, vmax):
        nmin, nmax = date2num((vmin, vmax))
        t0 = np.floor(nmin)
        nmax = nmax - t0
        nmin = nmin - t0
        nmin *= MUSECONDS_PER_DAY
        nmax *= MUSECONDS_PER_DAY

        ticks = self._wrapped_locator.tick_values(nmin, nmax)

        ticks = ticks / MUSECONDS_PER_DAY + t0
        return ticks

    def _get_unit(self):
        # docstring inherited
        return 1. / MUSECONDS_PER_DAY

    def _get_interval(self):
        # docstring inherited
        return self._interval


class DateConverter(units.ConversionInterface):
    """
    Converter for `datetime.date` and `datetime.datetime` data, or for
    date/time data represented as it would be converted by `date2num`.

    The 'unit' tag for such data is None or a `~datetime.tzinfo` instance.
    """

    def __init__(self, *, interval_multiples=True):
        self._interval_multiples = interval_multiples
        super().__init__()

    def axisinfo(self, unit, axis):
        """
        Return the `~matplotlib.units.AxisInfo` for *unit*.

        *unit* is a `~datetime.tzinfo` instance or None.
        The *axis* argument is required but not used.
        """
        tz = unit

        majloc = AutoDateLocator(tz=tz,
                                 interval_multiples=self._interval_multiples)
        majfmt = AutoDateFormatter(majloc, tz=tz)
        datemin = datetime.date(1970, 1, 1)
        datemax = datetime.date(1970, 1, 2)

        return units.AxisInfo(majloc=majloc, majfmt=majfmt, label='',
                              default_limits=(datemin, datemax))

    @staticmethod
    def convert(value, unit, axis):
        """
        If *value* is not already a number or sequence of numbers, convert it
        with `date2num`.

        The *unit* and *axis* arguments are not used.
        """
        return date2num(value)

    @staticmethod
    def default_units(x, axis):
        """
        Return the `~datetime.tzinfo` instance of *x* or of its first element,
        or None
        """
        if isinstance(x, np.ndarray):
            x = x.ravel()

        try:
            x = cbook._safe_first_finite(x)
        except (TypeError, StopIteration):
            pass

        try:
            return x.tzinfo
        except AttributeError:
            pass
        return None


class ConciseDateConverter(DateConverter):
    # docstring inherited

    def __init__(self, formats=None, zero_formats=None, offset_formats=None,
                 show_offset=True, *, interval_multiples=True):
        self._formats = formats
        self._zero_formats = zero_formats
        self._offset_formats = offset_formats
        self._show_offset = show_offset
        self._interval_multiples = interval_multiples
        super().__init__()

    def axisinfo(self, unit, axis):
        # docstring inherited
        tz = unit
        majloc = AutoDateLocator(tz=tz,
                                 interval_multiples=self._interval_multiples)
        majfmt = ConciseDateFormatter(majloc, tz=tz, formats=self._formats,
                                      zero_formats=self._zero_formats,
                                      offset_formats=self._offset_formats,
                                      show_offset=self._show_offset)
        datemin = datetime.date(1970, 1, 1)
        datemax = datetime.date(1970, 1, 2)
        return units.AxisInfo(majloc=majloc, majfmt=majfmt, label='',
                              default_limits=(datemin, datemax))


class _SwitchableDateConverter:
    """
    Helper converter-like object that generates and dispatches to
    temporary ConciseDateConverter or DateConverter instances based on
    :rc:`date.converter` and :rc:`date.interval_multiples`.
    """

    @staticmethod
    def _get_converter():
        converter_cls = {
            "concise": ConciseDateConverter, "auto": DateConverter}[
                mpl.rcParams["date.converter"]]
        interval_multiples = mpl.rcParams["date.interval_multiples"]
        return converter_cls(interval_multiples=interval_multiples)

    def axisinfo(self, *args, **kwargs):
        return self._get_converter().axisinfo(*args, **kwargs)

    def default_units(self, *args, **kwargs):
        return self._get_converter().default_units(*args, **kwargs)

    def convert(self, *args, **kwargs):
        return self._get_converter().convert(*args, **kwargs)


units.registry[np.datetime64] = \
    units.registry[datetime.date] = \
    units.registry[datetime.datetime] = \
    _SwitchableDateConverter()
<|endoftext|>import abc
import base64
import contextlib
from io import BytesIO, TextIOWrapper
import itertools
import logging
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import uuid
import warnings

import numpy as np
from PIL import Image

import matplotlib as mpl
from matplotlib._animation_data import (
    DISPLAY_TEMPLATE, INCLUDED_FRAMES, JS_INCLUDE, STYLE_INCLUDE)
from matplotlib import _api, cbook
import matplotlib.colors as mcolors

_log = logging.getLogger(__name__)

# Process creation flag for subprocess to prevent it raising a terminal
# window. See for example https://stackoverflow.com/q/24130623/
subprocess_creation_flags = (
    subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)


def adjusted_figsize(w, h, dpi, n):
    """
    Compute figure size so that pixels are a multiple of n.

    Parameters
    ----------
    w, h : float
        Size in inches.

    dpi : float
        The dpi.

    n : int
        The target multiple.

    Returns
    -------
    wnew, hnew : float
        The new figure size in inches.
    """

    # this maybe simplified if / when we adopt consistent rounding for
    # pixel size across the whole library
    def correct_roundoff(x, dpi, n):
        if int(x*dpi) % n != 0:
            if int(np.nextafter(x, np.inf)*dpi) % n == 0:
                x = np.nextafter(x, np.inf)
            elif int(np.nextafter(x, -np.inf)*dpi) % n == 0:
                x = np.nextafter(x, -np.inf)
        return x

    wnew = int(w * dpi / n) * n / dpi
    hnew = int(h * dpi / n) * n / dpi
    return correct_roundoff(wnew, dpi, n), correct_roundoff(hnew, dpi, n)


class MovieWriterRegistry:
    """Registry of available writer classes by human readable name."""

    def __init__(self):
        self._registered = dict()

    def register(self, name):
        """
        Decorator for registering a class under a name.

        Example use::

            @registry.register(name)
            class Foo:
                pass
        """
        def wrapper(writer_cls):
            self._registered[name] = writer_cls
            return writer_cls
        return wrapper

    def is_available(self, name):
        """
        Check if given writer is available by name.

        Parameters
        ----------
        name : str

        Returns
        -------
        bool
        """
        try:
            cls = self._registered[name]
        except KeyError:
            return False
        return cls.isAvailable()

    def __iter__(self):
        """Iterate over names of available writer class."""
        for name in self._registered:
            if self.is_available(name):
                yield name

    def list(self):
        """Get a list of available MovieWriters."""
        return [*self]

    def __getitem__(self, name):
        """Get an available writer class from its name."""
        if self.is_available(name):
            return self._registered[name]
        raise RuntimeError(f"Requested MovieWriter ({name}) not available")


writers = MovieWriterRegistry()


class AbstractMovieWriter(abc.ABC):
    """
    Abstract base class for writing movies, providing a way to grab frames by
    calling `~AbstractMovieWriter.grab_frame`.

    `setup` is called to start the process and `finish` is called afterwards.
    `saving` is provided as a context manager to facilitate this process as ::

        with moviewriter.saving(fig, outfile='myfile.mp4', dpi=100):
            # Iterate over frames
            moviewriter.grab_frame(**savefig_kwargs)

    The use of the context manager ensures that `setup` and `finish` are
    performed as necessary.

    An instance of a concrete subclass of this class can be given as the
    ``writer`` argument of `Animation.save()`.
    """

    def __init__(self, fps=5, metadata=None, codec=None, bitrate=None):
        self.fps = fps
        self.metadata = metadata if metadata is not None else {}
        self.codec = mpl._val_or_rc(codec, 'animation.codec')
        self.bitrate = mpl._val_or_rc(bitrate, 'animation.bitrate')

    @abc.abstractmethod
    def setup(self, fig, outfile, dpi=None):
        """
        Setup for writing the movie file.

        Parameters
        ----------
        fig : `~matplotlib.figure.Figure`
            The figure object that contains the information for frames.
        outfile : str
            The filename of the resulting movie file.
        dpi : float, default: ``fig.dpi``
            The DPI (or resolution) for the file.  This controls the size
            in pixels of the resulting movie file.
        """
        # Check that path is valid
        Path(outfile).parent.resolve(strict=True)
        self.outfile = outfile
        self.fig = fig
        if dpi is None:
            dpi = self.fig.dpi
        self.dpi = dpi

    @property
    def frame_size(self):
        """A tuple ``(width, height)`` in pixels of a movie frame."""
        w, h = self.fig.get_size_inches()
        return int(w * self.dpi), int(h * self.dpi)

    def _supports_transparency(self):
        """
        Whether this writer supports transparency.

        Writers may consult output file type and codec to determine this at runtime.
        """
        return False

    @abc.abstractmethod
    def grab_frame(self, **savefig_kwargs):
        """
        Grab the image information from the figure and save as a movie frame.

        All keyword arguments in *savefig_kwargs* are passed on to the
        `~.Figure.savefig` call that saves the figure.  However, several
        keyword arguments that are supported by `~.Figure.savefig` may not be
        passed as they are controlled by the MovieWriter:

        - *dpi*, *bbox_inches*:  These may not be passed because each frame of the
           animation much be exactly the same size in pixels.
        - *format*: This is controlled by the MovieWriter.
        """

    @abc.abstractmethod
    def finish(self):
        """Finish any processing for writing the movie."""

    @contextlib.contextmanager
    def saving(self, fig, outfile, dpi, *args, **kwargs):
        """
        Context manager to facilitate writing the movie file.

        ``*args, **kw`` are any parameters that should be passed to `setup`.
        """
        if mpl.rcParams['savefig.bbox'] == 'tight':
            _log.info("Disabling savefig.bbox = 'tight', as it may cause "
                      "frame size to vary, which is inappropriate for "
                      "animation.")

        # This particular sequence is what contextlib.contextmanager wants
        self.setup(fig, outfile, dpi, *args, **kwargs)
        with mpl.rc_context({'savefig.bbox': None}):
            try:
                yield self
            finally:
                self.finish()


class MovieWriter(AbstractMovieWriter):
    """
    Base class for writing movies.

    This is a base class for MovieWriter subclasses that write a movie frame
    data to a pipe. You cannot instantiate this class directly.
    See examples for how to use its subclasses.

    Attributes
    ----------
    frame_format : str
        The format used in writing frame data, defaults to 'rgba'.
    fig : `~matplotlib.figure.Figure`
        The figure to capture data from.
        This must be provided by the subclasses.
    """

    # Builtin writer subclasses additionally define the _exec_key and _args_key
    # attributes, which indicate the rcParams entries where the path to the
    # executable and additional command-line arguments to the executable are
    # stored.  Third-party writers cannot meaningfully set these as they cannot
    # extend rcParams with new keys.

    # Pipe-based writers only support RGBA, but file-based ones support more
    # formats.
    supported_formats = ["rgba"]

    def __init__(self, fps=5, codec=None, bitrate=None, extra_args=None,
                 metadata=None):
        """
        Parameters
        ----------
        fps : int, default: 5
            Movie frame rate (per second).
        codec : str or None, default: :rc:`animation.codec`
            The codec to use.
        bitrate : int, default: :rc:`animation.bitrate`
            The bitrate of the movie, in kilobits per second.  Higher values
            means higher quality movies, but increase the file size.  A value
            of -1 lets the underlying movie encoder select the bitrate.
        extra_args : list of str or None, optional
            Extra command-line arguments passed to the underlying movie encoder. These
            arguments are passed last to the encoder, just before the filename. The
            default, None, means to use :rc:`animation.[name-of-encoder]_args` for the
            builtin writers.
        metadata : dict[str, str], default: {}
            A dictionary of keys and values for metadata to include in the
            output file. Some keys that may be of use include:
            title, artist, genre, subject, copyright, srcform, comment.
        """
        if type(self) is MovieWriter:
            # TODO MovieWriter is still an abstract class and needs to be
            #      extended with a mixin. This should be clearer in naming
            #      and description. For now, just give a reasonable error
            #      message to users.
            raise TypeError(
                'MovieWriter cannot be instantiated directly. Please use one '
                'of its subclasses.')

        super().__init__(fps=fps, metadata=metadata, codec=codec,
                         bitrate=bitrate)
        self.frame_format = self.supported_formats[0]
        self.extra_args = extra_args

    def _adjust_frame_size(self):
        if self.codec == 'h264':
            wo, ho = self.fig.get_size_inches()
            w, h = adjusted_figsize(wo, ho, self.dpi, 2)
            if (wo, ho) != (w, h):
                self.fig.set_size_inches(w, h, forward=True)
                _log.info('figure size in inches has been adjusted '
                          'from %s x %s to %s x %s', wo, ho, w, h)
        else:
            w, h = self.fig.get_size_inches()
        _log.debug('frame size in pixels is %s x %s', *self.frame_size)
        return w, h

    def setup(self, fig, outfile, dpi=None):
        # docstring inherited
        super().setup(fig, outfile, dpi=dpi)
        self._w, self._h = self._adjust_frame_size()
        # Run here so that grab_frame() can write the data to a pipe. This
        # eliminates the need for temp files.
        self._run()

    def _run(self):
        # Uses subprocess to call the program for assembling frames into a
        # movie file.  *args* returns the sequence of command line arguments
        # from a few configuration options.
        command = self._args()
        _log.info('MovieWriter._run: running command: %s',
                  cbook._pformat_subprocess(command))
        PIPE = subprocess.PIPE
        self._proc = subprocess.Popen(
            command, stdin=PIPE, stdout=PIPE, stderr=PIPE,
            creationflags=subprocess_creation_flags)

    def finish(self):
        """Finish any processing for writing the movie."""
        out, err = self._proc.communicate()
        # Use the encoding/errors that universal_newlines would use.
        out = TextIOWrapper(BytesIO(out)).read()
        err = TextIOWrapper(BytesIO(err)).read()
        if out:
            _log.log(
                logging.WARNING if self._proc.returncode else logging.DEBUG,
                "MovieWriter stdout:\n%s", out)
        if err:
            _log.log(
                logging.WARNING if self._proc.returncode else logging.DEBUG,
                "MovieWriter stderr:\n%s", err)
        if self._proc.returncode:
            raise subprocess.CalledProcessError(
                self._proc.returncode, self._proc.args, out, err)

    def grab_frame(self, **savefig_kwargs):
        # docstring inherited
        _validate_grabframe_kwargs(savefig_kwargs)
        _log.debug('MovieWriter.grab_frame: Grabbing frame.')
        # Readjust the figure size in case it has been changed by the user.
        # All frames must have the same size to save the movie correctly.
        self.fig.set_size_inches(self._w, self._h)
        # Save the figure data to the sink, using the frame format and dpi.
        self.fig.savefig(self._proc.stdin, format=self.frame_format,
                         dpi=self.dpi, **savefig_kwargs)

    def _args(self):
        """Assemble list of encoder-specific command-line arguments."""
        return NotImplementedError("args needs to be implemented by subclass.")

    @classmethod
    def bin_path(cls):
        """
        Return the binary path to the commandline tool used by a specific
        subclass. This is a class method so that the tool can be looked for
        before making a particular MovieWriter subclass available.
        """
        return str(mpl.rcParams[cls._exec_key])

    @classmethod
    def isAvailable(cls):
        """Return whether a MovieWriter subclass is actually available."""
        return shutil.which(cls.bin_path()) is not None


class FileMovieWriter(MovieWriter):
    """
    `MovieWriter` for writing to individual files and stitching at the end.

    This must be sub-classed to be useful.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_format = mpl.rcParams['animation.frame_format']

    def setup(self, fig, outfile, dpi=None, frame_prefix=None):
        """
        Setup for writing the movie file.

        Parameters
        ----------
        fig : `~matplotlib.figure.Figure`
            The figure to grab the rendered frames from.
        outfile : str
            The filename of the resulting movie file.
        dpi : float, default: ``fig.dpi``
            The dpi of the output file. This, with the figure size,
            controls the size in pixels of the resulting movie file.
        frame_prefix : str, optional
            The filename prefix to use for temporary files.  If *None* (the
            default), files are written to a temporary directory which is
            deleted by `finish`; if not *None*, no temporary files are
            deleted.
        """
        # Check that path is valid
        Path(outfile).parent.resolve(strict=True)
        self.fig = fig
        self.outfile = outfile
        if dpi is None:
            dpi = self.fig.dpi
        self.dpi = dpi
        self._adjust_frame_size()

        if frame_prefix is None:
            self._tmpdir = TemporaryDirectory()
            self.temp_prefix = str(Path(self._tmpdir.name, 'tmp'))
        else:
            self._tmpdir = None
            self.temp_prefix = frame_prefix
        self._frame_counter = 0  # used for generating sequential file names
        self._temp_paths = list()
        self.fname_format_str = '%s%%07d.%s'

    def __del__(self):
        if hasattr(self, '_tmpdir') and self._tmpdir:
            self._tmpdir.cleanup()

    @property
    def frame_format(self):
        """
        Format (png, jpeg, etc.) to use for saving the frames, which can be
        decided by the individual subclasses.
        """
        return self._frame_format

    @frame_format.setter
    def frame_format(self, frame_format):
        if frame_format in self.supported_formats:
            self._frame_format = frame_format
        else:
            _api.warn_external(
                f"Ignoring file format {frame_format!r} which is not "
                f"supported by {type(self).__name__}; using "
                f"{self.supported_formats[0]} instead.")
            self._frame_format = self.supported_formats[0]

    def _base_temp_name(self):
        # Generates a template name (without number) given the frame format
        # for extension and the prefix.
        return self.fname_format_str % (self.temp_prefix, self.frame_format)

    def grab_frame(self, **savefig_kwargs):
        # docstring inherited
        # Creates a filename for saving using basename and counter.
        _validate_grabframe_kwargs(savefig_kwargs)
        path = Path(self._base_temp_name() % self._frame_counter)
        self._temp_paths.append(path)  # Record the filename for later use.
        self._frame_counter += 1  # Ensures each created name is unique.
        _log.debug('FileMovieWriter.grab_frame: Grabbing frame %d to path=%s',
                   self._frame_counter, path)
        with open(path, 'wb') as sink:  # Save figure to the sink.
            self.fig.savefig(sink, format=self.frame_format, dpi=self.dpi,
                             **savefig_kwargs)

    def finish(self):
        # Call run here now that all frame grabbing is done. All temp files
        # are available to be assembled.
        try:
            self._run()
            super().finish()
        finally:
            if self._tmpdir:
                _log.debug(
                    'MovieWriter: clearing temporary path=%s', self._tmpdir
                )
                self._tmpdir.cleanup()


@writers.register('pillow')
class PillowWriter(AbstractMovieWriter):
    def _supports_transparency(self):
        return True

    @classmethod
    def isAvailable(cls):
        return True

    def setup(self, fig, outfile, dpi=None):
        super().setup(fig, outfile, dpi=dpi)
        self._frames = []

    def grab_frame(self, **savefig_kwargs):
        _validate_grabframe_kwargs(savefig_kwargs)
        buf = BytesIO()
        self.fig.savefig(
            buf, **{**savefig_kwargs, "format": "rgba", "dpi": self.dpi})
        im = Image.frombuffer(
            "RGBA", self.frame_size, buf.getbuffer(), "raw", "RGBA", 0, 1)
        if im.getextrema()[3][0] < 255:
            # This frame has transparency, so we'll just add it as is.
            self._frames.append(im)
        else:
            # Without transparency, we switch to RGB mode, which converts to P mode a
            # little better if needed (specifically, this helps with GIF output.)
            self._frames.append(im.convert("RGB"))

    def finish(self):
        self._frames[0].save(
            self.outfile, save_all=True, append_images=self._frames[1:],
            duration=int(1000 / self.fps), loop=0)


# Base class of ffmpeg information. Has the config keys and the common set
# of arguments that controls the *output* side of things.
class FFMpegBase:
    """
    Mixin class for FFMpeg output.

    This is a base class for the concrete `FFMpegWriter` and `FFMpegFileWriter`
    classes.
    """

    _exec_key = 'animation.ffmpeg_path'
    _args_key = 'animation.ffmpeg_args'

    def _supports_transparency(self):
        suffix = Path(self.outfile).suffix
        if suffix in {'.apng', '.avif', '.gif', '.webm', '.webp'}:
            return True
        # This list was found by going through `ffmpeg -codecs` for video encoders,
        # running them with _support_transparency() forced to True, and checking that
        # the "Pixel format" in Kdenlive included alpha. Note this is not a guarantee
        # that transparency will work; you may also need to pass `-pix_fmt`, but we
        # trust the user has done so if they are asking for these formats.
        return self.codec in {
            'apng', 'avrp', 'bmp', 'cfhd', 'dpx', 'ffv1', 'ffvhuff', 'gif', 'huffyuv',
            'jpeg2000', 'ljpeg', 'png', 'prores', 'prores_aw', 'prores_ks', 'qtrle',
            'rawvideo', 'targa', 'tiff', 'utvideo', 'v408', }

    @property
    def output_args(self):
        args = []
        suffix = Path(self.outfile).suffix
        if suffix in {'.apng', '.avif', '.gif', '.webm', '.webp'}:
            self.codec = suffix[1:]
        else:
            args.extend(['-vcodec', self.codec])
        extra_args = (self.extra_args if self.extra_args is not None
                      else mpl.rcParams[self._args_key])
        # For h264, the default format is yuv444p, which is not compatible
        # with quicktime (and others). Specifying yuv420p fixes playback on
        # iOS, as well as HTML5 video in firefox and safari (on both Windows and
        # macOS). Also fixes internet explorer. This is as of 2015/10/29.
        if self.codec == 'h264' and '-pix_fmt' not in extra_args:
            args.extend(['-pix_fmt', 'yuv420p'])
        # For GIF, we're telling FFmpeg to split the video stream, to generate
        # a palette, and then use it for encoding.
        elif self.codec == 'gif' and '-filter_complex' not in extra_args:
            args.extend(['-filter_complex',
                         'split [a][b];[a] palettegen [p];[b][p] paletteuse'])
        # For AVIF, we're telling FFmpeg to split the video stream, extract the alpha,
        # in order to place it in a secondary stream, as needed by AVIF-in-FFmpeg.
        elif self.codec == 'avif' and '-filter_complex' not in extra_args:
            args.extend(['-filter_complex',
                         'split [rgb][rgba]; [rgba] alphaextract [alpha]',
                         '-map', '[rgb]', '-map', '[alpha]'])
        if self.bitrate > 0:
            args.extend(['-b', '%dk' % self.bitrate])  # %dk: bitrate in kbps.
        for k, v in self.metadata.items():
            args.extend(['-metadata', f'{k}={v}'])
        args.extend(extra_args)

        return args + ['-y', self.outfile]


# Combine FFMpeg options with pipe-based writing
@writers.register('ffmpeg')
class FFMpegWriter(FFMpegBase, MovieWriter):
    """
    Pipe-based ffmpeg writer.

    Frames are streamed directly to ffmpeg via a pipe and written in a single pass.

    This effectively works as a slideshow input to ffmpeg with the fps passed as
    ``-framerate``, so see also `their notes on frame rates`_ for further details.

    .. _their notes on frame rates: https://trac.ffmpeg.org/wiki/Slideshow#Framerates
    """
    def _args(self):
        # Returns the command line parameters for subprocess to use
        # ffmpeg to create a movie using a pipe.
        args = [self.bin_path(), '-f', 'rawvideo', '-vcodec', 'rawvideo',
                '-s', '%dx%d' % self.frame_size, '-pix_fmt', self.frame_format,
                '-framerate', str(self.fps)]
        # Logging is quieted because subprocess.PIPE has limited buffer size.
        # If you have a lot of frames in your animation and set logging to
        # DEBUG, you will have a buffer overrun.
        if _log.getEffectiveLevel() > logging.DEBUG:
            args += ['-loglevel', 'error']
        args += ['-i', 'pipe:'] + self.output_args
        return args


# Combine FFMpeg options with temp file-based writing
@writers.register('ffmpeg_file')
class FFMpegFileWriter(FFMpegBase, FileMovieWriter):
    """
    File-based ffmpeg writer.

    Frames are written to temporary files on disk and then stitched together at the end.

    This effectively works as a slideshow input to ffmpeg with the fps passed as
    ``-framerate``, so see also `their notes on frame rates`_ for further details.

    .. _their notes on frame rates: https://trac.ffmpeg.org/wiki/Slideshow#Framerates
    """
    supported_formats = ['png', 'jpeg', 'tiff', 'raw', 'rgba']

    def _args(self):
        # Returns the command line parameters for subprocess to use
        # ffmpeg to create a movie using a collection of temp images
        args = []
        # For raw frames, we need to explicitly tell ffmpeg the metadata.
        if self.frame_format in {'raw', 'rgba'}:
            args += [
                '-f', 'image2', '-vcodec', 'rawvideo',
                '-video_size', '%dx%d' % self.frame_size,
                '-pixel_format', 'rgba',
            ]
        args += ['-framerate', str(self.fps), '-i', self._base_temp_name()]
        if not self._tmpdir:
            args += ['-frames:v', str(self._frame_counter)]
        # Logging is quieted because subprocess.PIPE has limited buffer size.
        # If you have a lot of frames in your animation and set logging to
        # DEBUG, you will have a buffer overrun.
        if _log.getEffectiveLevel() > logging.DEBUG:
            args += ['-loglevel', 'error']
        return [self.bin_path(), *args, *self.output_args]


# Base class for animated GIFs with ImageMagick
class ImageMagickBase:
    """
    Mixin class for ImageMagick output.

    This is a base class for the concrete `ImageMagickWriter` and
    `ImageMagickFileWriter` classes, which define an ``input_names`` attribute
    (or property) specifying the input names passed to ImageMagick.
    """

    _exec_key = 'animation.convert_path'
    _args_key = 'animation.convert_args'

    def _supports_transparency(self):
        suffix = Path(self.outfile).suffix
        return suffix in {'.apng', '.avif', '.gif', '.webm', '.webp'}

    def _args(self):
        # ImageMagick does not recognize "raw".
        fmt = "rgba" if self.frame_format == "raw" else self.frame_format
        extra_args = (self.extra_args if self.extra_args is not None
                      else mpl.rcParams[self._args_key])
        return [
            self.bin_path(),
            "-size", "%ix%i" % self.frame_size,
            "-depth", "8",
            "-delay", str(100 / self.fps),
            "-loop", "0",
            f"{fmt}:{self.input_names}",
            *extra_args,
            self.outfile,
        ]

    @classmethod
    def bin_path(cls):
        binpath = super().bin_path()
        if binpath == 'convert':
            binpath = mpl._get_executable_info('magick').executable
        return binpath

    @classmethod
    def isAvailable(cls):
        try:
            return super().isAvailable()
        except mpl.ExecutableNotFoundError as _enf:
            # May be raised by get_executable_info.
            _log.debug('ImageMagick unavailable due to: %s', _enf)
            return False


# Combine ImageMagick options with pipe-based writing
@writers.register('imagemagick')
class ImageMagickWriter(ImageMagickBase, MovieWriter):
    """
    Pipe-based animated gif writer.

    Frames are streamed directly to ImageMagick via a pipe and written
    in a single pass.
    """

    input_names = "-"  # stdin


# Combine ImageMagick options with temp file-based writing
@writers.register('imagemagick_file')
class ImageMagickFileWriter(ImageMagickBase, FileMovieWriter):
    """
    File-based animated gif writer.

    Frames are written to temporary files on disk and then stitched
    together at the end.
    """

    supported_formats = ['png', 'jpeg', 'tiff', 'raw', 'rgba']
    input_names = property(
        lambda self: f'{self.temp_prefix}*.{self.frame_format}')


# Taken directly from jakevdp's JSAnimation package at
# http://github.com/jakevdp/JSAnimation
def _included_frames(frame_count, frame_format, frame_dir):
    return INCLUDED_FRAMES.format(Nframes=frame_count,
                                  frame_dir=frame_dir,
                                  frame_format=frame_format)


def _embedded_frames(frame_list, frame_format):
    """frame_list should be a list of base64-encoded png files"""
    if frame_format == 'svg':
        # Fix MIME type for svg
        frame_format = 'svg+xml'
    template = '  frames[{0}] = "data:image/{1};base64,{2}"\n'
    return "\n" + "".join(
        template.format(i, frame_format, frame_data.replace('\n', '\\\n'))
        for i, frame_data in enumerate(frame_list))


@writers.register('html')
class HTMLWriter(FileMovieWriter):
    """Writer for JavaScript-based HTML movies."""

    supported_formats = ['png', 'jpeg', 'tiff', 'svg']

    @classmethod
    def isAvailable(cls):
        return True

    def __init__(self, fps=30, codec=None, bitrate=None, extra_args=None,
                 metadata=None, embed_frames=False, default_mode='loop',
                 embed_limit=None):

        if extra_args:
            _log.warning("HTMLWriter ignores 'extra_args'")
        extra_args = ()  # Don't lookup nonexistent rcParam[args_key].
        self.embed_frames = embed_frames
        self.default_mode = default_mode.lower()
        _api.check_in_list(['loop', 'once', 'reflect'],
                           default_mode=self.default_mode)

        # Save embed limit, which is given in MB
        self._bytes_limit = mpl._val_or_rc(embed_limit, 'animation.embed_limit')
        # Convert from MB to bytes
        self._bytes_limit *= 1024 * 1024

        super().__init__(fps, codec, bitrate, extra_args, metadata)

    def setup(self, fig, outfile, dpi=None, frame_dir=None):
        outfile = Path(outfile)
        _api.check_in_list(['.html', '.htm'], outfile_extension=outfile.suffix)

        self._saved_frames = []
        self._total_bytes = 0
        self._hit_limit = False

        if not self.embed_frames:
            if frame_dir is None:
                frame_dir = outfile.with_name(outfile.stem + '_frames')
            frame_dir.mkdir(parents=True, exist_ok=True)
            frame_prefix = frame_dir / 'frame'
        else:
            frame_prefix = None

        super().setup(fig, outfile, dpi, frame_prefix)
        self._clear_temp = False

    def grab_frame(self, **savefig_kwargs):
        _validate_grabframe_kwargs(savefig_kwargs)
        if self.embed_frames:
            # Just stop processing if we hit the limit
            if self._hit_limit:
                return
            f = BytesIO()
            self.fig.savefig(f, format=self.frame_format,
                             dpi=self.dpi, **savefig_kwargs)
            imgdata64 = base64.encodebytes(f.getvalue()).decode('ascii')
            self._total_bytes += len(imgdata64)
            if self._total_bytes >= self._bytes_limit:
                _log.warning(
                    "Animation size has reached %s bytes, exceeding the limit "
                    "of %s. If you're sure you want a larger animation "
                    "embedded, set the animation.embed_limit rc parameter to "
                    "a larger value (in MB). This and further frames will be "
                    "dropped.", self._total_bytes, self._bytes_limit)
                self._hit_limit = True
            else:
                self._saved_frames.append(imgdata64)
        else:
            return super().grab_frame(**savefig_kwargs)

    def finish(self):
        # save the frames to an html file
        if self.embed_frames:
            fill_frames = _embedded_frames(self._saved_frames,
                                           self.frame_format)
            frame_count = len(self._saved_frames)
        else:
            # temp names is filled by FileMovieWriter
            frame_count = len(self._temp_paths)
            fill_frames = _included_frames(
                frame_count, self.frame_format,
                self._temp_paths[0].parent.relative_to(self.outfile.parent))
        mode_dict = dict(once_checked='',
                         loop_checked='',
                         reflect_checked='')
        mode_dict[self.default_mode + '_checked'] = 'checked'

        interval = 1000 // self.fps

        with open(self.outfile, 'w') as of:
            of.write(JS_INCLUDE + STYLE_INCLUDE)
            of.write(DISPLAY_TEMPLATE.format(id=uuid.uuid4().hex,
                                             Nframes=frame_count,
                                             fill_frames=fill_frames,
                                             interval=interval,
                                             **mode_dict))

        # Duplicate the temporary file clean up logic from
        # FileMovieWriter.finish.  We cannot call the inherited version of
        # finish because it assumes that there is a subprocess that we either
        # need to call to merge many frames together or that there is a
        # subprocess call that we need to clean up.
        if self._tmpdir:
            _log.debug('MovieWriter: clearing temporary path=%s', self._tmpdir)
            self._tmpdir.cleanup()


class Animation:
    """
    A base class for Animations.

    This class is not usable as is, and should be subclassed to provide needed
    behavior.

    .. note::

        You must store the created Animation in a variable that lives as long
        as the animation should run. Otherwise, the Animation object will be
        garbage-collected and the animation stops.

    Parameters
    ----------
    fig : `~matplotlib.figure.Figure`
        The figure object used to get needed events, such as draw or resize.

    event_source : object, optional
        A class that can run a callback when desired events
        are generated, as well as be stopped and started.

        Examples include timers (see `TimedAnimation`) and file
        system notifications.

    blit : bool, default: False
        Whether blitting is used to optimize drawing.  If the backend does not
        support blitting, then this parameter has no effect.

    See Also
    --------
    FuncAnimation,  ArtistAnimation
    """

    def __init__(self, fig, event_source=None, blit=False):
        self._draw_was_started = False

        self._fig = fig
        # Disables blitting for backends that don't support it.  This
        # allows users to request it if available, but still have a
        # fallback that works if it is not.
        self._blit = blit and fig.canvas.supports_blit

        # These are the basics of the animation.  The frame sequence represents
        # information for each frame of the animation and depends on how the
        # drawing is handled by the subclasses. The event source fires events
        # that cause the frame sequence to be iterated.
        self.frame_seq = self.new_frame_seq()
        self.event_source = event_source

        # Instead of starting the event source now, we connect to the figure's
        # draw_event, so that we only start once the figure has been drawn.
        self._first_draw_id = fig.canvas.mpl_connect('draw_event', self._start)

        # Connect to the figure's close_event so that we don't continue to
        # fire events and try to draw to a deleted figure.
        self._close_id = self._fig.canvas.mpl_connect('close_event',
                                                      self._stop)
        if self._blit:
            self._setup_blit()

    def __del__(self):
        if not getattr(self, '_draw_was_started', True):
            warnings.warn(
                'Animation was deleted without rendering anything. This is '
                'most likely not intended. To prevent deletion, assign the '
                'Animation to a variable, e.g. `anim`, that exists until you '
                'output the Animation using `plt.show()` or '
                '`anim.save()`.'
            )

    def _start(self, *args):
        """
        Starts interactive animation. Adds the draw frame command to the GUI
        handler, calls show to start the event loop.
        """
        # Do not start the event source if saving() it.
        if self._fig.canvas.is_saving():
            return
        # First disconnect our draw event handler
        self._fig.canvas.mpl_disconnect(self._first_draw_id)

        # Now do any initial draw
        self._init_draw()

        # Add our callback for stepping the animation and
        # actually start the event_source.
        self.event_source.add_callback(self._step)
        self.event_source.start()

    def _stop(self, *args):
        # On stop we disconnect all of our events.
        if self._blit:
            self._fig.canvas.mpl_disconnect(self._resize_id)
        self._fig.canvas.mpl_disconnect(self._close_id)
        self.event_source.remove_callback(self._step)
        self.event_source = None

    def save(self, filename, writer=None, fps=None, dpi=None, codec=None,
             bitrate=None, extra_args=None, metadata=None, extra_anim=None,
             savefig_kwargs=None, *, progress_callback=None):
        """
        Save the animation as a movie file by drawing every frame.

        Parameters
        ----------
        filename : str
            The output filename, e.g., :file:`mymovie.mp4`.

        writer : `AbstractMovieWriter` subclass or str, default: :rc:`animation.writer`
            The writer used to grab the frames and create the movie file.
            This can be an instance of an `AbstractMovieWriter` subclass or a
            string. The builtin writers are

            ==================  ==============================
            str                 class
            ==================  ==============================
            'ffmpeg'            `.FFMpegWriter`
            'ffmpeg_file'       `.FFMpegFileWriter`
            'imagemagick'       `.ImageMagickWriter`
            'imagemagick_file'  `.ImageMagickFileWriter`
            'pillow'            `.PillowWriter`
            'html'              `.HTMLWriter`
            ==================  ==============================

        fps : int, optional
            Movie frame rate (per second).  If not set, the frame rate from the
            animation's frame interval.

        dpi : float, default: :rc:`savefig.dpi`
            Controls the dots per inch for the movie frames.  Together with
            the figure's size in inches, this controls the size of the movie.

        codec : str, default: :rc:`animation.codec`.
            The video codec to use.  Not all codecs are supported by a given
            `MovieWriter`.

        bitrate : int, default: :rc:`animation.bitrate`
            The bitrate of the movie, in kilobits per second.  Higher values
            means higher quality movies, but increase the file size.  A value
            of -1 lets the underlying movie encoder select the bitrate.

        extra_args : list of str or None, optional
            Extra command-line arguments passed to the underlying movie encoder. These
            arguments are passed last to the encoder, just before the output filename.
            The default, None, means to use :rc:`animation.[name-of-encoder]_args` for
            the builtin writers.

        metadata : dict[str, str], default: {}
            Dictionary of keys and values for metadata to include in
            the output file. Some keys that may be of use include:
            title, artist, genre, subject, copyright, srcform, comment.

        extra_anim : list, default: []
            Additional `Animation` objects that should be included
            in the saved movie file. These need to be from the same
            `.Figure` instance. Also, animation frames will
            just be simply combined, so there should be a 1:1 correspondence
            between the frames from the different animations.

        savefig_kwargs : dict, default: {}
            Keyword arguments passed to each `~.Figure.savefig` call used to
            save the individual frames.

        progress_callback : function, optional
            A callback function that will be called for every frame to notify
            the saving progress. It must have the signature ::

                def func(current_frame: int, total_frames: int) -> Any

            where *current_frame* is the current frame number and *total_frames* is the
            total number of frames to be saved. *total_frames* is set to None, if the
            total number of frames cannot be determined. Return values may exist but are
            ignored.

            Example code to write the progress to stdout::

                progress_callback = lambda i, n: print(f'Saving frame {i}/{n}')

        Notes
        -----
        *fps*, *codec*, *bitrate*, *extra_args* and *metadata* are used to
        construct a `.MovieWriter` instance and can only be passed if
        *writer* is a string.  If they are passed as non-*None* and *writer*
        is a `.MovieWriter`, a `RuntimeError` will be raised.
        """

        all_anim = [self]
        if extra_anim is not None:
            all_anim.extend(anim for anim in extra_anim
                            if anim._fig is self._fig)

        # Disable "Animation was deleted without rendering" warning.
        for anim in all_anim:
            anim._draw_was_started = True

        if writer is None:
            writer = mpl.rcParams['animation.writer']
        elif (not isinstance(writer, str) and
              any(arg is not None
                  for arg in (fps, codec, bitrate, extra_args, metadata))):
            raise RuntimeError('Passing in values for arguments '
                               'fps, codec, bitrate, extra_args, or metadata '
                               'is not supported when writer is an existing '
                               'MovieWriter instance. These should instead be '
                               'passed as arguments when creating the '
                               'MovieWriter instance.')

        if savefig_kwargs is None:
            savefig_kwargs = {}
        else:
            # we are going to mutate this below
            savefig_kwargs = dict(savefig_kwargs)

        if fps is None and hasattr(self, '_interval'):
            # Convert interval in ms to frames per second
            fps = 1000. / self._interval

        # Reuse the savefig DPI for ours if none is given.
        dpi = mpl._val_or_rc(dpi, 'savefig.dpi')
        if dpi == 'figure':
            dpi = self._fig.dpi

        writer_kwargs = {}
        if codec is not None:
            writer_kwargs['codec'] = codec
        if bitrate is not None:
            writer_kwargs['bitrate'] = bitrate
        if extra_args is not None:
            writer_kwargs['extra_args'] = extra_args
        if metadata is not None:
            writer_kwargs['metadata'] = metadata

        # If we have the name of a writer, instantiate an instance of the
        # registered class.
        if isinstance(writer, str):
            try:
                writer_cls = writers[writer]
            except RuntimeError:  # Raised if not available.
                writer_cls = PillowWriter  # Always available.
                _log.warning("MovieWriter %s unavailable; using Pillow "
                             "instead.", writer)
            writer = writer_cls(fps, **writer_kwargs)
        _log.info('Animation.save using %s', type(writer))

        if 'bbox_inches' in savefig_kwargs:
            _log.warning("Warning: discarding the 'bbox_inches' argument in "
                         "'savefig_kwargs' as it may cause frame size "
                         "to vary, which is inappropriate for animation.")
            savefig_kwargs.pop('bbox_inches')

        # Create a new sequence of frames for saved data. This is different
        # from new_frame_seq() to give the ability to save 'live' generated
        # frame information to be saved later.
        # TODO: Right now, after closing the figure, saving a movie won't work
        # since GUI widgets are gone. Either need to remove extra code to
        # allow for this non-existent use case or find a way to make it work.

        def _pre_composite_to_white(color):
            r, g, b, a = mcolors.to_rgba(color)
            return a * np.array([r, g, b]) + 1 - a

        # canvas._is_saving = True makes the draw_event animation-starting
        # callback a no-op; canvas.manager = None prevents resizing the GUI
        # widget (both are likewise done in savefig()).
        with (writer.saving(self._fig, filename, dpi),
              cbook._setattr_cm(self._fig.canvas, _is_saving=True, manager=None)):
            if not writer._supports_transparency():
                facecolor = savefig_kwargs.get('facecolor',
                                               mpl.rcParams['savefig.facecolor'])
                if facecolor == 'auto':
                    facecolor = self._fig.get_facecolor()
                savefig_kwargs['facecolor'] = _pre_composite_to_white(facecolor)
                savefig_kwargs['transparent'] = False   # just to be safe!

            for anim in all_anim:
                anim._init_draw()  # Clear the initial frame
            frame_number = 0
            # TODO: Currently only FuncAnimation has a save_count
            #       attribute. Can we generalize this to all Animations?
            save_count_list = [getattr(a, '_save_count', None)
                               for a in all_anim]
            if None in save_count_list:
                total_frames = None
            else:
                total_frames = sum(save_count_list)
            for data in zip(*[a.new_saved_frame_seq() for a in all_anim]):
                for anim, d in zip(all_anim, data):
                    # TODO: See if turning off blit is really necessary
                    anim._draw_next_frame(d, blit=False)
                    if progress_callback is not None:
                        progress_callback(frame_number, total_frames)
                        frame_number += 1
                writer.grab_frame(**savefig_kwargs)

    def _step(self, *args):
        """
        Handler for getting events. By default, gets the next frame in the
        sequence and hands the data off to be drawn.
        """
        # Returns True to indicate that the event source should continue to
        # call _step, until the frame sequence reaches the end of iteration,
        # at which point False will be returned.
        try:
            framedata = next(self.frame_seq)
            self._draw_next_frame(framedata, self._blit)
            return True
        except StopIteration:
            return False

    def new_frame_seq(self):
        """Return a new sequence of frame information."""
        # Default implementation is just an iterator over self._framedata
        return iter(self._framedata)

    def new_saved_frame_seq(self):
        """Return a new sequence of saved/cached frame information."""
        # Default is the same as the regular frame sequence
        return self.new_frame_seq()

    def _draw_next_frame(self, framedata, blit):
        # Breaks down the drawing of the next frame into steps of pre- and
        # post- draw, as well as the drawing of the frame itself.
        self._pre_draw(framedata, blit)
        self._draw_frame(framedata)
        self._post_draw(framedata, blit)

    def _init_draw(self):
        # Initial draw to clear the frame. Also used by the blitting code
        # when a clean base is required.
        self._draw_was_started = True

    def _pre_draw(self, framedata, blit):
        # Perform any cleaning or whatnot before the drawing of the frame.
        # This default implementation allows blit to clear the frame.
        if blit:
            self._blit_clear(self._drawn_artists)

    def _draw_frame(self, framedata):
        # Performs actual drawing of the frame.
        raise NotImplementedError('Needs to be implemented by subclasses to'
                                  ' actually make an animation.')

    def _post_draw(self, framedata, blit):
        # After the frame is rendered, this handles the actual flushing of
        # the draw, which can be a direct draw_idle() or make use of the
        # blitting.
        if blit and self._drawn_artists:
            self._blit_draw(self._drawn_artists)
        else:
            self._fig.canvas.draw_idle()

    # The rest of the code in this class is to facilitate easy blitting
    def _blit_draw(self, artists):
        # Handles blitted drawing, which renders only the artists given instead
        # of the entire figure.
        updated_ax = {a.axes for a in artists}
        # Enumerate artists to cache Axes backgrounds. We do not draw
        # artists yet to not cache foreground from plots with shared Axes
        for ax in updated_ax:
            # If we haven't cached the background for the current view of this
            # Axes object, do so now. This might not always be reliable, but
            # it's an attempt to automate the process.
            cur_view = ax._get_view()
            view, bg = self._blit_cache.get(ax, (object(), None))
            if cur_view != view:
                self._blit_cache[ax] = (
                    cur_view, ax.figure.canvas.copy_from_bbox(ax.bbox))
        # Make a separate pass to draw foreground.
        for a in artists:
            a.axes.draw_artist(a)
        # After rendering all the needed artists, blit each Axes individually.
        for ax in updated_ax:
            ax.figure.canvas.blit(ax.bbox)

    def _blit_clear(self, artists):
        # Get a list of the Axes that need clearing from the artists that
        # have been drawn. Grab the appropriate saved background from the
        # cache and restore.
        axes = {a.axes for a in artists}
        for ax in axes:
            try:
                view, bg = self._blit_cache[ax]
            except KeyError:
                continue
            if ax._get_view() == view:
                ax.figure.canvas.restore_region(bg)
            else:
                self._blit_cache.pop(ax)

    def _setup_blit(self):
        # Setting up the blit requires: a cache of the background for the Axes
        self._blit_cache = dict()
        self._drawn_artists = []
        # _post_draw needs to be called first to initialize the renderer
        self._post_draw(None, self._blit)
        # Then we need to clear the Frame for the initial draw
        # This is typically handled in _on_resize because QT and Tk
        # emit a resize event on launch, but the macosx backend does not,
        # thus we force it here for everyone for consistency
        self._init_draw()
        # Connect to future resize events
        self._resize_id = self._fig.canvas.mpl_connect('resize_event',
                                                       self._on_resize)

    def _on_resize(self, event):
        # On resize, we need to disable the resize event handling so we don't
        # get too many events. Also stop the animation events, so that
        # we're paused. Reset the cache and re-init. Set up an event handler
        # to catch once the draw has actually taken place.
        self._fig.canvas.mpl_disconnect(self._resize_id)
        self.event_source.stop()
        self._blit_cache.clear()
        self._init_draw()
        self._resize_id = self._fig.canvas.mpl_connect('draw_event',
                                                       self._end_redraw)

    def _end_redraw(self, event):
        # Now that the redraw has happened, do the post draw flushing and
        # blit handling. Then re-enable all of the original events.
        self._post_draw(None, False)
        self.event_source.start()
        self._fig.canvas.mpl_disconnect(self._resize_id)
        self._resize_id = self._fig.canvas.mpl_connect('resize_event',
                                                       self._on_resize)

    def to_html5_video(self, embed_limit=None):
        """
        Convert the animation to an HTML5 ``<video>`` tag.

        This saves the animation as an h264 video, encoded in base64
        directly into the HTML5 video tag. This respects :rc:`animation.writer`
        and :rc:`animation.bitrate`. This also makes use of the
        *interval* to control the speed, and uses the *repeat*
        parameter to decide whether to loop.

        Parameters
        ----------
        embed_limit : float, optional
            Limit, in MB, of the returned animation. No animation is created
            if the limit is exceeded.
            Defaults to :rc:`animation.embed_limit` = 20.0.

        Returns
        -------
        str
            An HTML5 video tag with the animation embedded as base64 encoded
            h264 video.
            If the *embed_limit* is exceeded, this returns the string
            "Video too large to embed."
        """
        VIDEO_TAG = r'''<video {size} {options}>
  <source type="video/mp4" src="data:video/mp4;base64,{video}">
  Your browser does not support the video tag.
</video>'''
        # Cache the rendering of the video as HTML
        if not hasattr(self, '_base64_video'):
            # Save embed limit, which is given in MB
            embed_limit = mpl._val_or_rc(embed_limit, 'animation.embed_limit')

            # Convert from MB to bytes
            embed_limit *= 1024 * 1024

            # Can't open a NamedTemporaryFile twice on Windows, so use a
            # TemporaryDirectory instead.
            with TemporaryDirectory() as tmpdir:
                path = Path(tmpdir, "temp.m4v")
                # We create a writer manually so that we can get the
                # appropriate size for the tag
                Writer = writers[mpl.rcParams['animation.writer']]
                writer = Writer(codec='h264',
                                bitrate=mpl.rcParams['animation.bitrate'],
                                fps=1000. / self._interval)
                self.save(str(path), writer=writer)
                # Now open and base64 encode.
                vid64 = base64.encodebytes(path.read_bytes())

            vid_len = len(vid64)
            if vid_len >= embed_limit:
                _log.warning(
                    "Animation movie is %s bytes, exceeding the limit of %s. "
                    "If you're sure you want a large animation embedded, set "
                    "the animation.embed_limit rc parameter to a larger value "
                    "(in MB).", vid_len, embed_limit)
            else:
                self._base64_video = vid64.decode('ascii')
                self._video_size = 'width="{}" height="{}"'.format(
                        *writer.frame_size)

        # If we exceeded the size, this attribute won't exist
        if hasattr(self, '_base64_video'):
            # Default HTML5 options are to autoplay and display video controls
            options = ['controls', 'autoplay']

            # If we're set to repeat, make it loop
            if getattr(self, '_repeat', False):
                options.append('loop')

            return VIDEO_TAG.format(video=self._base64_video,
                                    size=self._video_size,
                                    options=' '.join(options))
        else:
            return 'Video too large to embed.'

    def to_jshtml(self, fps=None, embed_frames=True, default_mode=None):
        """
        Generate HTML representation of the animation.

        Parameters
        ----------
        fps : int, optional
            Movie frame rate (per second). If not set, the frame rate from
            the animation's frame interval.
        embed_frames : bool, optional
        default_mode : str, optional
            What to do when the animation ends. Must be one of ``{'loop',
            'once', 'reflect'}``. Defaults to ``'loop'`` if the *repeat*
            parameter is True, otherwise ``'once'``.

        Returns
        -------
        str
            An HTML representation of the animation embedded as a js object as
            produced with the `.HTMLWriter`.
        """
        if fps is None and hasattr(self, '_interval'):
            # Convert interval in ms to frames per second
            fps = 1000 / self._interval

        # If we're not given a default mode, choose one base on the value of
        # the _repeat attribute
        if default_mode is None:
            default_mode = 'loop' if getattr(self, '_repeat',
                                             False) else 'once'

        if not hasattr(self, "_html_representation"):
            # Can't open a NamedTemporaryFile twice on Windows, so use a
            # TemporaryDirectory instead.
            with TemporaryDirectory() as tmpdir:
                path = Path(tmpdir, "temp.html")
                writer = HTMLWriter(fps=fps,
                                    embed_frames=embed_frames,
                                    default_mode=default_mode)
                self.save(str(path), writer=writer)
                self._html_representation = path.read_text()

        return self._html_representation

    def _repr_html_(self):
        """IPython display hook for rendering."""
        fmt = mpl.rcParams['animation.html']
        if fmt == 'html5':
            return self.to_html5_video()
        elif fmt == 'jshtml':
            return self.to_jshtml()

    def pause(self):
        """Pause the animation."""
        self.event_source.stop()
        if self._blit:
            for artist in self._drawn_artists:
                artist.set_animated(False)

    def resume(self):
        """Resume the animation."""
        self.event_source.start()
        if self._blit:
            for artist in self._drawn_artists:
                artist.set_animated(True)


class TimedAnimation(Animation):
    """
    `Animation` subclass for time-based animation.

    A new frame is drawn every *interval* milliseconds.

    .. note::

        You must store the created Animation in a variable that lives as long
        as the animation should run. Otherwise, the Animation object will be
        garbage-collected and the animation stops.

    Parameters
    ----------
    fig : `~matplotlib.figure.Figure`
        The figure object used to get needed events, such as draw or resize.
    interval : int, default: 200
        Delay between frames in milliseconds.
    repeat_delay : int, default: 0
        The delay in milliseconds between consecutive animation runs, if
        *repeat* is True.
    repeat : bool, default: True
        Whether the animation repeats when the sequence of frames is completed.
    blit : bool, default: False
        Whether blitting is used to optimize drawing.
    """
    def __init__(self, fig, interval=200, repeat_delay=0, repeat=True,
                 event_source=None, *args, **kwargs):
        self._interval = interval
        # Undocumented support for repeat_delay = None as backcompat.
        self._repeat_delay = repeat_delay if repeat_delay is not None else 0
        self._repeat = repeat
        # If we're not given an event source, create a new timer. This permits
        # sharing timers between animation objects for syncing animations.
        if event_source is None:
            event_source = fig.canvas.new_timer(interval=self._interval)
        super().__init__(fig, event_source=event_source, *args, **kwargs)

    def _step(self, *args):
        """Handler for getting events."""
        # Extends the _step() method for the Animation class.  If
        # Animation._step signals that it reached the end and we want to
        # repeat, we refresh the frame sequence and return True. If
        # _repeat_delay is set, change the event_source's interval to our loop
        # delay and set the callback to one which will then set the interval
        # back.
        still_going = super()._step(*args)
        if not still_going:
            if self._repeat:
                # Restart the draw loop
                self._init_draw()
                self.frame_seq = self.new_frame_seq()
                self.event_source.interval = self._repeat_delay
                return True
            else:
                # We are done with the animation. Call pause to remove
                # animated flags from artists that were using blitting
                self.pause()
                if self._blit:
                    # Remove the resize callback if we were blitting
                    self._fig.canvas.mpl_disconnect(self._resize_id)
                self._fig.canvas.mpl_disconnect(self._close_id)
                self.event_source = None
                return False

        self.event_source.interval = self._interval
        return True


class ArtistAnimation(TimedAnimation):
    """
    `TimedAnimation` subclass that creates an animation by using a fixed
    set of `.Artist` objects.

    Before creating an instance, all plotting should have taken place
    and the relevant artists saved.

    .. note::

        You must store the created Animation in a variable that lives as long
        as the animation should run. Otherwise, the Animation object will be
        garbage-collected and the animation stops.

    Parameters
    ----------
    fig : `~matplotlib.figure.Figure`
        The figure object used to get needed events, such as draw or resize.
    artists : list
        Each list entry is a collection of `.Artist` objects that are made
        visible on the corresponding frame.  Other artists are made invisible.
    interval : int, default: 200
        Delay between frames in milliseconds.
    repeat_delay : int, default: 0
        The delay in milliseconds between consecutive animation runs, if
        *repeat* is True.
    repeat : bool, default: True
        Whether the animation repeats when the sequence of frames is completed.
    blit : bool, default: False
        Whether blitting is used to optimize drawing.
    """

    def __init__(self, fig, artists, *args, **kwargs):
        # Internal list of artists drawn in the most recent frame.
        self._drawn_artists = []

        # Use the list of artists as the framedata, which will be iterated
        # over by the machinery.
        self._framedata = artists
        super().__init__(fig, *args, **kwargs)

    def _init_draw(self):
        super()._init_draw()
        # Make all the artists involved in *any* frame invisible
        figs = set()
        for f in self.new_frame_seq():
            for artist in f:
                artist.set_visible(False)
                artist.set_animated(self._blit)
                # Assemble a list of unique figures that need flushing
                if artist.get_figure() not in figs:
                    figs.add(artist.get_figure())

        # Flush the needed figures
        for fig in figs:
            fig.canvas.draw_idle()

    def _pre_draw(self, framedata, blit):
        """Clears artists from the last frame."""
        if blit:
            # Let blit handle clearing
            self._blit_clear(self._drawn_artists)
        else:
            # Otherwise, make all the artists from the previous frame invisible
            for artist in self._drawn_artists:
                artist.set_visible(False)

    def _draw_frame(self, artists):
        # Save the artists that were passed in as framedata for the other
        # steps (esp. blitting) to use.
        self._drawn_artists = artists

        # Make all the artists from the current frame visible
        for artist in artists:
            artist.set_visible(True)


class FuncAnimation(TimedAnimation):
    """
    `TimedAnimation` subclass that makes an animation by repeatedly calling
    a function *func*.

    .. note::

        You must store the created Animation in a variable that lives as long
        as the animation should run. Otherwise, the Animation object will be
        garbage-collected and the animation stops.

    Parameters
    ----------
    fig : `~matplotlib.figure.Figure`
        The figure object used to get needed events, such as draw or resize.

    func : callable
        The function to call at each frame.  The first argument will
        be the next value in *frames*.   Any additional positional
        arguments can be supplied using `functools.partial` or via the *fargs*
        parameter.

        The required signature is::

            def func(frame, *fargs) -> iterable_of_artists

        It is often more convenient to provide the arguments using
        `functools.partial`. In this way it is also possible to pass keyword
        arguments. To pass a function with both positional and keyword
        arguments, set all arguments as keyword arguments, just leaving the
        *frame* argument unset::

            def func(frame, art, *, y=None):
                ...

            ani = FuncAnimation(fig, partial(func, art=ln, y='foo'))

        If ``blit == True``, *func* must return an iterable of all artists
        that were modified or created. This information is used by the blitting
        algorithm to determine which parts of the figure have to be updated.
        The return value is unused if ``blit == False`` and may be omitted in
        that case.

    frames : iterable, int, generator function, or None, optional
        Source of data to pass *func* and each frame of the animation

        - If an iterable, then simply use the values provided.  If the
          iterable has a length, it will override the *save_count* kwarg.

        - If an integer, then equivalent to passing ``range(frames)``

        - If a generator function, then must have the signature::

             def gen_function() -> obj

        - If *None*, then equivalent to passing ``itertools.count``.

        In all of these cases, the values in *frames* is simply passed through
        to the user-supplied *func* and thus can be of any type.

    init_func : callable, optional
        A function used to draw a clear frame. If not given, the results of
        drawing from the first item in the frames sequence will be used. This
        function will be called once before the first frame.

        The required signature is::

            def init_func() -> iterable_of_artists

        If ``blit == True``, *init_func* must return an iterable of artists
        to be re-drawn. This information is used by the blitting algorithm to
        determine which parts of the figure have to be updated.  The return
        value is unused if ``blit == False`` and may be omitted in that case.

    fargs : tuple or None, optional
        Additional arguments to pass to each call to *func*. Note: the use of
        `functools.partial` is preferred over *fargs*. See *func* for details.

    save_count : int, optional
        Fallback for the number of values from *frames* to cache. This is
        only used if the number of frames cannot be inferred from *frames*,
        i.e. when it's an iterator without length or a generator.

    interval : int, default: 200
        Delay between frames in milliseconds.

    repeat_delay : int, default: 0
        The delay in milliseconds between consecutive animation runs, if
        *repeat* is True.

    repeat : bool, default: True
        Whether the animation repeats when the sequence of frames is completed.

    blit : bool, default: False
        Whether blitting is used to optimize drawing.  Note: when using
        blitting, any animated artists will be drawn according to their zorder;
        however, they will be drawn on top of any previous artists, regardless
        of their zorder.

    cache_frame_data : bool, default: True
        Whether frame data is cached.  Disabling cache might be helpful when
        frames contain large objects.
    """
    def __init__(self, fig, func, frames=None, init_func=None, fargs=None,
                 save_count=None, *, cache_frame_data=True, **kwargs):
        if fargs:
            self._args = fargs
        else:
            self._args = ()
        self._func = func
        self._init_func = init_func

        # Amount of framedata to keep around for saving movies. This is only
        # used if we don't know how many frames there will be: in the case
        # of no generator or in the case of a callable.
        self._save_count = save_count
        # Set up a function that creates a new iterable when needed. If nothing
        # is passed in for frames, just use itertools.count, which will just
        # keep counting from 0. A callable passed in for frames is assumed to
        # be a generator. An iterable will be used as is, and anything else
        # will be treated as a number of frames.
        if frames is None:
            self._iter_gen = itertools.count
        elif callable(frames):
            self._iter_gen = frames
        elif np.iterable(frames):
            if kwargs.get('repeat', True):
                self._tee_from = frames
                def iter_frames(frames=frames):
                    this, self._tee_from = itertools.tee(self._tee_from, 2)
                    yield from this
                self._iter_gen = iter_frames
            else:
                self._iter_gen = lambda: iter(frames)
            if hasattr(frames, '__len__'):
                self._save_count = len(frames)
                if save_count is not None:
                    _api.warn_external(
                        f"You passed in an explicit {save_count=} "
                        "which is being ignored in favor of "
                        f"{len(frames)=}."
                    )
        else:
            self._iter_gen = lambda: iter(range(frames))
            self._save_count = frames
            if save_count is not None:
                _api.warn_external(
                    f"You passed in an explicit {save_count=} which is being "
                    f"ignored in favor of {frames=}."
                )
        if self._save_count is None and cache_frame_data:
            _api.warn_external(
                f"{frames=!r} which we can infer the length of, "
                "did not pass an explicit *save_count* "
                f"and passed {cache_frame_data=}.  To avoid a possibly "
                "unbounded cache, frame data caching has been disabled. "
                "To suppress this warning either pass "
                "`cache_frame_data=False` or `save_count=MAX_FRAMES`."
            )
            cache_frame_data = False

        self._cache_frame_data = cache_frame_data

        # Needs to be initialized so the draw functions work without checking
        self._save_seq = []

        super().__init__(fig, **kwargs)

        # Need to reset the saved seq, since right now it will contain data
        # for a single frame from init, which is not what we want.
        self._save_seq = []

    def new_frame_seq(self):
        # Use the generating function to generate a new frame sequence
        return self._iter_gen()

    def new_saved_frame_seq(self):
        # Generate an iterator for the sequence of saved data. If there are
        # no saved frames, generate a new frame sequence and take the first
        # save_count entries in it.
        if self._save_seq:
            # While iterating we are going to update _save_seq
            # so make a copy to safely iterate over
            self._old_saved_seq = list(self._save_seq)
            return iter(self._old_saved_seq)
        else:
            if self._save_count is None:
                frame_seq = self.new_frame_seq()

                def gen():
                    try:
                        while True:
                            yield next(frame_seq)
                    except StopIteration:
                        pass
                return gen()
            else:
                return itertools.islice(self.new_frame_seq(), self._save_count)

    def _init_draw(self):
        super()._init_draw()
        # Initialize the drawing either using the given init_func or by
        # calling the draw function with the first item of the frame sequence.
        # For blitting, the init_func should return a sequence of modified
        # artists.
        if self._init_func is None:
            try:
                frame_data = next(self.new_frame_seq())
            except StopIteration:
                # we can't start the iteration, it may have already been
                # exhausted by a previous save or just be 0 length.
                # warn and bail.
                warnings.warn(
                    "Can not start iterating the frames for the initial draw. "
                    "This can be caused by passing in a 0 length sequence "
                    "for *frames*.\n\n"
                    "If you passed *frames* as a generator "
                    "it may be exhausted due to a previous display or save."
                )
                return
            self._draw_frame(frame_data)
        else:
            self._drawn_artists = self._init_func()
            if self._blit:
                if self._drawn_artists is None:
                    raise RuntimeError('The init_func must return a '
                                       'sequence of Artist objects.')
                for a in self._drawn_artists:
                    a.set_animated(self._blit)
        self._save_seq = []

    def _draw_frame(self, framedata):
        if self._cache_frame_data:
            # Save the data for potential saving of movies.
            self._save_seq.append(framedata)
            self._save_seq = self._save_seq[-self._save_count:]

        # Call the func with framedata and args. If blitting is desired,
        # func needs to return a sequence of any artists that were modified.
        self._drawn_artists = self._func(framedata, *self._args)

        if self._blit:

            err = RuntimeError('The animation function must return a sequence '
                               'of Artist objects.')
            try:
                # check if a sequence
                iter(self._drawn_artists)
            except TypeError:
                raise err from None

            # check each item if it's artist
            for i in self._drawn_artists:
                if not isinstance(i, mpl.artist.Artist):
                    raise err

            self._drawn_artists = sorted(self._drawn_artists,
                                         key=lambda x: x.get_zorder())

            for a in self._drawn_artists:
                a.set_animated(self._blit)


def _validate_grabframe_kwargs(savefig_kwargs):
    if mpl.rcParams['savefig.bbox'] == 'tight':
        raise ValueError(
            f"{mpl.rcParams['savefig.bbox']=} must not be 'tight' as it "
            "may cause frame size to vary, which is inappropriate for animation."
        )
    for k in ('dpi', 'bbox_inches', 'format'):
        if k in savefig_kwargs:
            raise TypeError(
                f"grab_frame got an unexpected keyword argument {k!r}"
            )
<|endoftext|>"""
Classes to support contour plotting and labelling for the Axes class.
"""

from contextlib import ExitStack
import functools
import math
from numbers import Integral

import numpy as np
from numpy import ma

import matplotlib as mpl
from matplotlib import _api, _docstring
from matplotlib.backend_bases import MouseButton
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.text import Text
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.collections as mcoll
import matplotlib.font_manager as font_manager
import matplotlib.cbook as cbook
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms


def _contour_labeler_event_handler(cs, inline, inline_spacing, event):
    canvas = cs.axes.get_figure(root=True).canvas
    is_button = event.name == "button_press_event"
    is_key = event.name == "key_press_event"
    # Quit (even if not in infinite mode; this is consistent with
    # MATLAB and sometimes quite useful, but will require the user to
    # test how many points were actually returned before using data).
    if (is_button and event.button == MouseButton.MIDDLE
            or is_key and event.key in ["escape", "enter"]):
        canvas.stop_event_loop()
    # Pop last click.
    elif (is_button and event.button == MouseButton.RIGHT
          or is_key and event.key in ["backspace", "delete"]):
        # Unfortunately, if one is doing inline labels, then there is currently
        # no way to fix the broken contour - once humpty-dumpty is broken, he
        # can't be put back together.  In inline mode, this does nothing.
        if not inline:
            cs.pop_label()
            canvas.draw()
    # Add new click.
    elif (is_button and event.button == MouseButton.LEFT
          # On macOS/gtk, some keys return None.
          or is_key and event.key is not None):
        if cs.axes.contains(event)[0]:
            cs.add_label_near(event.x, event.y, transform=False,
                              inline=inline, inline_spacing=inline_spacing)
            canvas.draw()


class ContourLabeler:
    """Mixin to provide labelling capability to `.ContourSet`."""

    def clabel(self, levels=None, *,
               fontsize=None, inline=True, inline_spacing=5, fmt=None,
               colors=None, use_clabeltext=False, manual=False,
               rightside_up=True, zorder=None):
        """
        Label a contour plot.

        Adds labels to line contours in this `.ContourSet` (which inherits from
        this mixin class).

        Parameters
        ----------
        levels : array-like, optional
            A list of level values, that should be labeled. The list must be
            a subset of ``cs.levels``. If not given, all levels are labeled.

        fontsize : str or float, default: :rc:`font.size`
            Size in points or relative size e.g., 'smaller', 'x-large'.
            See `.Text.set_size` for accepted string values.

        colors : :mpltype:`color` or colors or None, default: None
            The label colors:

            - If *None*, the color of each label matches the color of
              the corresponding contour.

            - If one string color, e.g., *colors* = 'r' or *colors* =
              'red', all labels will be plotted in this color.

            - If a tuple of colors (string, float, RGB, etc), different labels
              will be plotted in different colors in the order specified.

        inline : bool, default: True
            If ``True`` the underlying contour is removed where the label is
            placed.

        inline_spacing : float, default: 5
            Space in pixels to leave on each side of label when placing inline.

            This spacing will be exact for labels at locations where the
            contour is straight, less so for labels on curved contours.

        fmt : `.Formatter` or str or callable or dict, optional
            How the levels are formatted:

            - If a `.Formatter`, it is used to format all levels at once, using
              its `.Formatter.format_ticks` method.
            - If a str, it is interpreted as a %-style format string.
            - If a callable, it is called with one level at a time and should
              return the corresponding label.
            - If a dict, it should directly map levels to labels.

            The default is to use a standard `.ScalarFormatter`.

        manual : bool or iterable, default: False
            If ``True``, contour labels will be placed manually using
            mouse clicks. Click the first button near a contour to
            add a label, click the second button (or potentially both
            mouse buttons at once) to finish adding labels. The third
            button can be used to remove the last label added, but
            only if labels are not inline. Alternatively, the keyboard
            can be used to select label locations (enter to end label
            placement, delete or backspace act like the third mouse button,
            and any other key will select a label location).

            *manual* can also be an iterable object of (x, y) tuples.
            Contour labels will be created as if mouse is clicked at each
            (x, y) position.

        rightside_up : bool, default: True
            If ``True``, label rotations will always be plus
            or minus 90 degrees from level.

        use_clabeltext : bool, default: False
            If ``True``, use `.Text.set_transform_rotates_text` to ensure that
            label rotation is updated whenever the Axes aspect changes.

        zorder : float or None, default: ``(2 + contour.get_zorder())``
            zorder of the contour labels.

        Returns
        -------
        labels
            A list of `.Text` instances for the labels.
        """

        # Based on the input arguments, clabel() adds a list of "label
        # specific" attributes to the ContourSet object.  These attributes are
        # all of the form label* and names should be fairly self explanatory.
        #
        # Once these attributes are set, clabel passes control to the labels()
        # method (for automatic label placement) or blocking_input_loop and
        # _contour_labeler_event_handler (for manual label placement).

        if fmt is None:
            fmt = ticker.ScalarFormatter(useOffset=False)
            fmt.create_dummy_axis()
        self.labelFmt = fmt
        self._use_clabeltext = use_clabeltext
        self.labelManual = manual
        self.rightside_up = rightside_up
        self._clabel_zorder = 2 + self.get_zorder() if zorder is None else zorder

        if levels is None:
            levels = self.levels
            indices = list(range(len(self.cvalues)))
        else:
            levlabs = list(levels)
            indices, levels = [], []
            for i, lev in enumerate(self.levels):
                if lev in levlabs:
                    indices.append(i)
                    levels.append(lev)
            if len(levels) < len(levlabs):
                raise ValueError(f"Specified levels {levlabs} don't match "
                                 f"available levels {self.levels}")
        self.labelLevelList = levels
        self.labelIndiceList = indices

        self._label_font_props = font_manager.FontProperties(size=fontsize)

        if colors is None:
            self.labelMappable = self
            self.labelCValueList = np.take(self.cvalues, self.labelIndiceList)
        else:
            cmap = mcolors.ListedColormap(colors, N=len(self.labelLevelList))
            self.labelCValueList = list(range(len(self.labelLevelList)))
            self.labelMappable = cm.ScalarMappable(cmap=cmap,
                                                   norm=mcolors.NoNorm())

        self.labelXYs = []

        if np.iterable(manual):
            for x, y in manual:
                self.add_label_near(x, y, inline, inline_spacing)
        elif manual:
            print('Select label locations manually using first mouse button.')
            print('End manual selection with second mouse button.')
            if not inline:
                print('Remove last label by clicking third mouse button.')
            mpl._blocking_input.blocking_input_loop(
                self.axes.get_figure(root=True),
                ["button_press_event", "key_press_event"],
                timeout=-1, handler=functools.partial(
                    _contour_labeler_event_handler,
                    self, inline, inline_spacing))
        else:
            self.labels(inline, inline_spacing)

        return cbook.silent_list('text.Text', self.labelTexts)

    def print_label(self, linecontour, labelwidth):
        """Return whether a contour is long enough to hold a label."""
        return (len(linecontour) > 10 * labelwidth
                or (len(linecontour)
                    and (np.ptp(linecontour, axis=0) > 1.2 * labelwidth).any()))

    def too_close(self, x, y, lw):
        """Return whether a label is already near this location."""
        thresh = (1.2 * lw) ** 2
        return any((x - loc[0]) ** 2 + (y - loc[1]) ** 2 < thresh
                   for loc in self.labelXYs)

    def _get_nth_label_width(self, nth):
        """Return the width of the *nth* label, in pixels."""
        fig = self.axes.get_figure(root=False)
        renderer = fig.get_figure(root=True)._get_renderer()
        return (Text(0, 0,
                     self.get_text(self.labelLevelList[nth], self.labelFmt),
                     figure=fig, fontproperties=self._label_font_props)
                .get_window_extent(renderer).width)

    def get_text(self, lev, fmt):
        """Get the text of the label."""
        if isinstance(lev, str):
            return lev
        elif isinstance(fmt, dict):
            return fmt.get(lev, '%1.3f')
        elif callable(getattr(fmt, "format_ticks", None)):
            return fmt.format_ticks([*self.labelLevelList, lev])[-1]
        elif callable(fmt):
            return fmt(lev)
        else:
            return fmt % lev

    def locate_label(self, linecontour, labelwidth):
        """
        Find good place to draw a label (relatively flat part of the contour).
        """
        ctr_size = len(linecontour)
        n_blocks = int(np.ceil(ctr_size / labelwidth)) if labelwidth > 1 else 1
        block_size = ctr_size if n_blocks == 1 else int(labelwidth)
        # Split contour into blocks of length ``block_size``, filling the last
        # block by cycling the contour start (per `np.resize` semantics).  (Due
        # to cycling, the index returned is taken modulo ctr_size.)
        xx = np.resize(linecontour[:, 0], (n_blocks, block_size))
        yy = np.resize(linecontour[:, 1], (n_blocks, block_size))
        yfirst = yy[:, :1]
        ylast = yy[:, -1:]
        xfirst = xx[:, :1]
        xlast = xx[:, -1:]
        s = (yfirst - yy) * (xlast - xfirst) - (xfirst - xx) * (ylast - yfirst)
        l = np.hypot(xlast - xfirst, ylast - yfirst)
        # Ignore warning that divide by zero throws, as this is a valid option
        with np.errstate(divide='ignore', invalid='ignore'):
            distances = (abs(s) / l).sum(axis=-1)
        # Labels are drawn in the middle of the block (``hbsize``) where the
        # contour is the closest (per ``distances``) to a straight line, but
        # not `too_close()` to a preexisting label.
        hbsize = block_size // 2
        adist = np.argsort(distances)
        # If all candidates are `too_close()`, go back to the straightest part
        # (``adist[0]``).
        for idx in np.append(adist, adist[0]):
            x, y = xx[idx, hbsize], yy[idx, hbsize]
            if not self.too_close(x, y, labelwidth):
                break
        return x, y, (idx * block_size + hbsize) % ctr_size

    def _split_path_and_get_label_rotation(self, path, idx, screen_pos, lw, spacing=5):
        """
        Prepare for insertion of a label at index *idx* of *path*.

        Parameters
        ----------
        path : Path
            The path where the label will be inserted, in data space.
        idx : int
            The vertex index after which the label will be inserted.
        screen_pos : (float, float)
            The position where the label will be inserted, in screen space.
        lw : float
            The label width, in screen space.
        spacing : float
            Extra spacing around the label, in screen space.

        Returns
        -------
        path : Path
            The path, broken so that the label can be drawn over it.
        angle : float
            The rotation of the label.

        Notes
        -----
        Both tasks are done together to avoid calculating path lengths multiple times,
        which is relatively costly.

        The method used here involves computing the path length along the contour in
        pixel coordinates and then looking (label width / 2) away from central point to
        determine rotation and then to break contour if desired.  The extra spacing is
        taken into account when breaking the path, but not when computing the angle.
        """
        xys = path.vertices
        codes = path.codes

        # Insert a vertex at idx/pos (converting back to data space), if there isn't yet
        # a vertex there.  With infinite precision one could also always insert the
        # extra vertex (it will get masked out by the label below anyways), but floating
        # point inaccuracies (the point can have undergone a data->screen->data
        # transform loop) can slightly shift the point and e.g. shift the angle computed
        # below from exactly zero to nonzero.
        pos = self.get_transform().inverted().transform(screen_pos)
        if not np.allclose(pos, xys[idx]):
            xys = np.insert(xys, idx, pos, axis=0)
            codes = np.insert(codes, idx, Path.LINETO)

        # Find the connected component where the label will be inserted.  Note that a
        # path always starts with a MOVETO, and we consider there's an implicit
        # MOVETO (closing the last path) at the end.
        movetos = (codes == Path.MOVETO).nonzero()[0]
        start = movetos[movetos <= idx][-1]
        try:
            stop = movetos[movetos > idx][0]
        except IndexError:
            stop = len(codes)

        # Restrict ourselves to the connected component.
        cc_xys = xys[start:stop]
        idx -= start

        # If the path is closed, rotate it s.t. it starts at the label.
        is_closed_path = codes[stop - 1] == Path.CLOSEPOLY
        if is_closed_path:
            cc_xys = np.concatenate([cc_xys[idx:-1], cc_xys[:idx+1]])
            idx = 0

        # Like np.interp, but additionally vectorized over fp.
        def interp_vec(x, xp, fp): return [np.interp(x, xp, col) for col in fp.T]

        # Use cumulative path lengths ("cpl") as curvilinear coordinate along contour.
        screen_xys = self.get_transform().transform(cc_xys)
        path_cpls = np.insert(
            np.cumsum(np.hypot(*np.diff(screen_xys, axis=0).T)), 0, 0)
        path_cpls -= path_cpls[idx]

        # Use linear interpolation to get end coordinates of label.
        target_cpls = np.array([-lw/2, lw/2])
        if is_closed_path:  # For closed paths, target from the other end.
            target_cpls[0] += (path_cpls[-1] - path_cpls[0])
        (sx0, sx1), (sy0, sy1) = interp_vec(target_cpls, path_cpls, screen_xys)
        angle = np.rad2deg(np.arctan2(sy1 - sy0, sx1 - sx0))  # Screen space.
        if self.rightside_up:  # Fix angle so text is never upside-down
            angle = (angle + 90) % 180 - 90

        target_cpls += [-spacing, +spacing]  # Expand range by spacing.

        # Get indices near points of interest; use -1 as out of bounds marker.
        i0, i1 = np.interp(target_cpls, path_cpls, range(len(path_cpls)),
                           left=-1, right=-1)
        i0 = math.floor(i0)
        i1 = math.ceil(i1)
        (x0, x1), (y0, y1) = interp_vec(target_cpls, path_cpls, cc_xys)

        # Actually break contours (dropping zero-len parts).
        new_xy_blocks = []
        new_code_blocks = []
        if is_closed_path:
            if i0 != -1 and i1 != -1:
                # This is probably wrong in the case that the entire contour would
                # be discarded, but ensures that a valid path is returned and is
                # consistent with behavior of mpl <3.8
                points = cc_xys[i1:i0+1]
                new_xy_blocks.extend([[(x1, y1)], points, [(x0, y0)]])
                nlines = len(points) + 1
                new_code_blocks.extend([[Path.MOVETO], [Path.LINETO] * nlines])
        else:
            if i0 != -1:
                new_xy_blocks.extend([cc_xys[:i0 + 1], [(x0, y0)]])
                new_code_blocks.extend([[Path.MOVETO], [Path.LINETO] * (i0 + 1)])
            if i1 != -1:
                new_xy_blocks.extend([[(x1, y1)], cc_xys[i1:]])
                new_code_blocks.extend([
                    [Path.MOVETO], [Path.LINETO] * (len(cc_xys) - i1)])

        # Back to the full path.
        xys = np.concatenate([xys[:start], *new_xy_blocks, xys[stop:]])
        codes = np.concatenate([codes[:start], *new_code_blocks, codes[stop:]])

        return angle, Path(xys, codes)

    def add_label(self, x, y, rotation, lev, cvalue):
        """Add a contour label, respecting whether *use_clabeltext* was set."""
        data_x, data_y = self.axes.transData.inverted().transform((x, y))
        t = Text(
            data_x, data_y,
            text=self.get_text(lev, self.labelFmt),
            rotation=rotation,
            horizontalalignment='center', verticalalignment='center',
            zorder=self._clabel_zorder,
            color=self.labelMappable.to_rgba(cvalue, alpha=self.get_alpha()),
            fontproperties=self._label_font_props,
            clip_box=self.axes.bbox)
        if self._use_clabeltext:
            data_rotation, = self.axes.transData.inverted().transform_angles(
                [rotation], [[x, y]])
            t.set(rotation=data_rotation, transform_rotates_text=True)
        self.labelTexts.append(t)
        self.labelCValues.append(cvalue)
        self.labelXYs.append((x, y))
        # Add label to plot here - useful for manual mode label selection
        self.axes.add_artist(t)

    def add_label_near(self, x, y, inline=True, inline_spacing=5,
                       transform=None):
        """
        Add a label near the point ``(x, y)``.

        Parameters
        ----------
        x, y : float
            The approximate location of the label.
        inline : bool, default: True
            If *True* remove the segment of the contour beneath the label.
        inline_spacing : int, default: 5
            Space in pixels to leave on each side of label when placing
            inline. This spacing will be exact for labels at locations where
            the contour is straight, less so for labels on curved contours.
        transform : `.Transform` or `False`, default: ``self.axes.transData``
            A transform applied to ``(x, y)`` before labeling.  The default
            causes ``(x, y)`` to be interpreted as data coordinates.  `False`
            is a synonym for `.IdentityTransform`; i.e. ``(x, y)`` should be
            interpreted as display coordinates.
        """

        if transform is None:
            transform = self.axes.transData
        if transform:
            x = self.axes.convert_xunits(x)
            y = self.axes.convert_yunits(y)
            x, y = transform.transform((x, y))

        idx_level_min, idx_vtx_min, proj = self._find_nearest_contour(
            (x, y), self.labelIndiceList)
        path = self._paths[idx_level_min]
        level = self.labelIndiceList.index(idx_level_min)
        label_width = self._get_nth_label_width(level)
        rotation, path = self._split_path_and_get_label_rotation(
            path, idx_vtx_min, proj, label_width, inline_spacing)
        self.add_label(*proj, rotation, self.labelLevelList[idx_level_min],
                       self.labelCValueList[idx_level_min])

        if inline:
            self._paths[idx_level_min] = path

    def pop_label(self, index=-1):
        """Defaults to removing last label, but any index can be supplied"""
        self.labelCValues.pop(index)
        t = self.labelTexts.pop(index)
        t.remove()

    def labels(self, inline, inline_spacing):
        for idx, (icon, lev, cvalue) in enumerate(zip(
                self.labelIndiceList,
                self.labelLevelList,
                self.labelCValueList,
        )):
            trans = self.get_transform()
            label_width = self._get_nth_label_width(idx)
            additions = []
            for subpath in self._paths[icon]._iter_connected_components():
                screen_xys = trans.transform(subpath.vertices)
                # Check if long enough for a label
                if self.print_label(screen_xys, label_width):
                    x, y, idx = self.locate_label(screen_xys, label_width)
                    rotation, path = self._split_path_and_get_label_rotation(
                        subpath, idx, (x, y),
                        label_width, inline_spacing)
                    self.add_label(x, y, rotation, lev, cvalue)  # Really add label.
                    if inline:  # If inline, add new contours
                        additions.append(path)
                else:  # If not adding label, keep old path
                    additions.append(subpath)
            # After looping over all segments on a contour, replace old path by new one
            # if inlining.
            if inline:
                self._paths[icon] = Path.make_compound_path(*additions)

    def remove(self):
        super().remove()
        for text in self.labelTexts:
            text.remove()


def _find_closest_point_on_path(xys, p):
    """
    Parameters
    ----------
    xys : (N, 2) array-like
        Coordinates of vertices.
    p : (float, float)
        Coordinates of point.

    Returns
    -------
    d2min : float
        Minimum square distance of *p* to *xys*.
    proj : (float, float)
        Projection of *p* onto *xys*.
    imin : (int, int)
        Consecutive indices of vertices of segment in *xys* where *proj* is.
        Segments are considered as including their end-points; i.e. if the
        closest point on the path is a node in *xys* with index *i*, this
        returns ``(i-1, i)``.  For the special case where *xys* is a single
        point, this returns ``(0, 0)``.
    """
    if len(xys) == 1:
        return (((p - xys[0]) ** 2).sum(), xys[0], (0, 0))
    dxys = xys[1:] - xys[:-1]  # Individual segment vectors.
    norms = (dxys ** 2).sum(axis=1)
    norms[norms == 0] = 1  # For zero-length segment, replace 0/0 by 0/1.
    rel_projs = np.clip(  # Project onto each segment in relative 0-1 coords.
        ((p - xys[:-1]) * dxys).sum(axis=1) / norms,
        0, 1)[:, None]
    projs = xys[:-1] + rel_projs * dxys  # Projs. onto each segment, in (x, y).
    d2s = ((projs - p) ** 2).sum(axis=1)  # Squared distances.
    imin = np.argmin(d2s)
    return (d2s[imin], projs[imin], (imin, imin+1))


_docstring.interpd.register(contour_set_attributes=r"""
Attributes
----------
levels : array
    The values of the contour levels.

layers : array
    Same as levels for line contours; half-way between
    levels for filled contours.  See ``ContourSet._process_colors``.
""")


@_docstring.interpd
class ContourSet(ContourLabeler, mcoll.Collection):
    """
    Store a set of contour lines or filled regions.

    User-callable method: `~.Axes.clabel`

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`

    levels : [level0, level1, ..., leveln]
        A list of floating point numbers indicating the contour levels.

    allsegs : [level0segs, level1segs, ...]
        List of all the polygon segments for all the *levels*.
        For contour lines ``len(allsegs) == len(levels)``, and for
        filled contour regions ``len(allsegs) = len(levels)-1``. The lists
        should look like ::

            level0segs = [polygon0, polygon1, ...]
            polygon0 = [[x0, y0], [x1, y1], ...]

    allkinds : ``None`` or [level0kinds, level1kinds, ...]
        Optional list of all the polygon vertex kinds (code types), as
        described and used in Path. This is used to allow multiply-
        connected paths such as holes within filled polygons.
        If not ``None``, ``len(allkinds) == len(allsegs)``. The lists
        should look like ::

            level0kinds = [polygon0kinds, ...]
            polygon0kinds = [vertexcode0, vertexcode1, ...]

        If *allkinds* is not ``None``, usually all polygons for a
        particular contour level are grouped together so that
        ``level0segs = [polygon0]`` and ``level0kinds = [polygon0kinds]``.

    **kwargs
        Keyword arguments are as described in the docstring of
        `~.Axes.contour`.

    %(contour_set_attributes)s
    """

    def __init__(self, ax, *args,
                 levels=None, filled=False, linewidths=None, linestyles=None,
                 hatches=(None,), alpha=None, origin=None, extent=None,
                 cmap=None, colors=None, norm=None, vmin=None, vmax=None,
                 colorizer=None, extend='neither', antialiased=None, nchunk=0,
                 locator=None, transform=None, negative_linestyles=None, clip_path=None,
                 **kwargs):
        """
        Draw contour lines or filled regions, depending on
        whether keyword arg *filled* is ``False`` (default) or ``True``.

        Call signature::

            ContourSet(ax, levels, allsegs, [allkinds], **kwargs)

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            The `~.axes.Axes` object to draw on.

        levels : [level0, level1, ..., leveln]
            A list of floating point numbers indicating the contour
            levels.

        allsegs : [level0segs, level1segs, ...]
            List of all the polygon segments for all the *levels*.
            For contour lines ``len(allsegs) == len(levels)``, and for
            filled contour regions ``len(allsegs) = len(levels)-1``. The lists
            should look like ::

                level0segs = [polygon0, polygon1, ...]
                polygon0 = [[x0, y0], [x1, y1], ...]

        allkinds : [level0kinds, level1kinds, ...], optional
            Optional list of all the polygon vertex kinds (code types), as
            described and used in Path. This is used to allow multiply-
            connected paths such as holes within filled polygons.
            If not ``None``, ``len(allkinds) == len(allsegs)``. The lists
            should look like ::

                level0kinds = [polygon0kinds, ...]
                polygon0kinds = [vertexcode0, vertexcode1, ...]

            If *allkinds* is not ``None``, usually all polygons for a
            particular contour level are grouped together so that
            ``level0segs = [polygon0]`` and ``level0kinds = [polygon0kinds]``.

        **kwargs
            Keyword arguments are as described in the docstring of
            `~.Axes.contour`.
        """
        if antialiased is None and filled:
            # Eliminate artifacts; we are not stroking the boundaries.
            antialiased = False
            # The default for line contours will be taken from the
            # LineCollection default, which uses :rc:`lines.antialiased`.
        super().__init__(
            antialiaseds=antialiased,
            alpha=alpha,
            clip_path=clip_path,
            transform=transform,
            colorizer=colorizer,
        )
        self.axes = ax
        self.levels = levels
        self.filled = filled
        self.hatches = hatches
        self.origin = origin
        self.extent = extent
        self.colors = colors
        self.extend = extend

        self.nchunk = nchunk
        self.locator = locator

        if colorizer:
            self._set_colorizer_check_keywords(colorizer, cmap=cmap,
                                               norm=norm, vmin=vmin,
                                               vmax=vmax, colors=colors)
            norm = colorizer.norm
            cmap = colorizer.cmap
        if (isinstance(norm, mcolors.LogNorm)
                or isinstance(self.locator, ticker.LogLocator)):
            self.logscale = True
            if norm is None:
                norm = mcolors.LogNorm()
        else:
            self.logscale = False

        _api.check_in_list([None, 'lower', 'upper', 'image'], origin=origin)
        if self.extent is not None and len(self.extent) != 4:
            raise ValueError(
                "If given, 'extent' must be None or (x0, x1, y0, y1)")
        if self.colors is not None and cmap is not None:
            raise ValueError('Either colors or cmap must be None')
        if self.origin == 'image':
            self.origin = mpl.rcParams['image.origin']

        self._orig_linestyles = linestyles  # Only kept for user access.
        self.negative_linestyles = negative_linestyles
        # If negative_linestyles was not defined as a keyword argument, define
        # negative_linestyles with rcParams
        if self.negative_linestyles is None:
            self.negative_linestyles = \
                mpl.rcParams['contour.negative_linestyle']

        kwargs = self._process_args(*args, **kwargs)
        self._process_levels()

        self._extend_min = self.extend in ['min', 'both']
        self._extend_max = self.extend in ['max', 'both']
        if self.colors is not None:
            if mcolors.is_color_like(self.colors):
                color_sequence = [self.colors]
            else:
                color_sequence = self.colors

            ncolors = len(self.levels)
            if self.filled:
                ncolors -= 1
            i0 = 0

            # Handle the case where colors are given for the extended
            # parts of the contour.

            use_set_under_over = False
            # if we are extending the lower end, and we've been given enough
            # colors then skip the first color in the resulting cmap. For the
            # extend_max case we don't need to worry about passing more colors
            # than ncolors as ListedColormap will clip.
            total_levels = (ncolors +
                            int(self._extend_min) +
                            int(self._extend_max))
            if (len(color_sequence) == total_levels and
                    (self._extend_min or self._extend_max)):
                use_set_under_over = True
                if self._extend_min:
                    i0 = 1

            cmap = mcolors.ListedColormap(color_sequence[i0:None], N=ncolors)

            if use_set_under_over:
                if self._extend_min:
                    cmap.set_under(color_sequence[0])
                if self._extend_max:
                    cmap.set_over(color_sequence[-1])

        # label lists must be initialized here
        self.labelTexts = []
        self.labelCValues = []

        self.set_cmap(cmap)
        if norm is not None:
            self.set_norm(norm)
        with self.norm.callbacks.blocked(signal="changed"):
            if vmin is not None:
                self.norm.vmin = vmin
            if vmax is not None:
                self.norm.vmax = vmax
        self.norm._changed()
        self._process_colors()

        if self._paths is None:
            self._paths = self._make_paths_from_contour_generator()

        if self.filled:
            if linewidths is not None:
                _api.warn_external('linewidths is ignored by contourf')
            # Lower and upper contour levels.
            lowers, uppers = self._get_lowers_and_uppers()
            self.set(
                edgecolor="none",
                # Default zorder taken from Collection
                zorder=kwargs.pop("zorder", 1),
            )

        else:
            self.set(
                facecolor="none",
                linewidths=self._process_linewidths(linewidths),
                linestyle=self._process_linestyles(linestyles),
                # Default zorder taken from LineCollection, which is higher
                # than for filled contours so that lines are displayed on top.
                zorder=kwargs.pop("zorder", 2),
                label="_nolegend_",
            )

        self.axes.add_collection(self, autolim=False)
        self.sticky_edges.x[:] = [self._mins[0], self._maxs[0]]
        self.sticky_edges.y[:] = [self._mins[1], self._maxs[1]]
        self.axes.update_datalim([self._mins, self._maxs])
        self.axes.autoscale_view(tight=True)

        self.changed()  # set the colors

        if kwargs:
            _api.warn_external(
                'The following kwargs were not used by contour: ' +
                ", ".join(map(repr, kwargs))
            )

    allsegs = property(lambda self: [
        [subp.vertices for subp in p._iter_connected_components()]
        for p in self.get_paths()])
    allkinds = property(lambda self: [
        [subp.codes for subp in p._iter_connected_components()]
        for p in self.get_paths()])
    alpha = property(lambda self: self.get_alpha())
    linestyles = property(lambda self: self._orig_linestyles)

    def get_transform(self):
        """Return the `.Transform` instance used by this ContourSet."""
        if self._transform is None:
            self._transform = self.axes.transData
        elif (not isinstance(self._transform, mtransforms.Transform)
              and hasattr(self._transform, '_as_mpl_transform')):
            self._transform = self._transform._as_mpl_transform(self.axes)
        return self._transform

    def __getstate__(self):
        state = self.__dict__.copy()
        # the C object _contour_generator cannot currently be pickled. This
        # isn't a big issue as it is not actually used once the contour has
        # been calculated.
        state['_contour_generator'] = None
        return state

    def legend_elements(self, variable_name='x', str_format=str):
        """
        Return a list of artists and labels suitable for passing through
        to `~.Axes.legend` which represent this ContourSet.

        The labels have the form "0 < x <= 1" stating the data ranges which
        the artists represent.

        Parameters
        ----------
        variable_name : str
            The string used inside the inequality used on the labels.
        str_format : function: float -> str
            Function used to format the numbers in the labels.

        Returns
        -------
        artists : list[`.Artist`]
            A list of the artists.
        labels : list[str]
            A list of the labels.
        """
        artists = []
        labels = []

        if self.filled:
            lowers, uppers = self._get_lowers_and_uppers()
            n_levels = len(self._paths)
            for idx in range(n_levels):
                artists.append(mpatches.Rectangle(
                    (0, 0), 1, 1,
                    facecolor=self.get_facecolor()[idx],
                    hatch=self.hatches[idx % len(self.hatches)],
                ))
                lower = str_format(lowers[idx])
                upper = str_format(uppers[idx])
                if idx == 0 and self.extend in ('min', 'both'):
                    labels.append(fr'${variable_name} \leq {lower}s$')
                elif idx == n_levels - 1 and self.extend in ('max', 'both'):
                    labels.append(fr'${variable_name} > {upper}s$')
                else:
                    labels.append(fr'${lower} < {variable_name} \leq {upper}$')
        else:
            for idx, level in enumerate(self.levels):
                artists.append(Line2D(
                    [], [],
                    color=self.get_edgecolor()[idx],
                    linewidth=self.get_linewidths()[idx],
                    linestyle=self.get_linestyles()[idx],
                ))
                labels.append(fr'${variable_name} = {str_format(level)}$')

        return artists, labels

    def _process_args(self, *args, **kwargs):
        """
        Process *args* and *kwargs*; override in derived classes.

        Must set self.levels, self.zmin and self.zmax, and update Axes limits.
        """
        self.levels = args[0]
        allsegs = args[1]
        allkinds = args[2] if len(args) > 2 else None
        self.zmax = np.max(self.levels)
        self.zmin = np.min(self.levels)

        if allkinds is None:
            allkinds = [[None] * len(segs) for segs in allsegs]

        # Check lengths of levels and allsegs.
        if self.filled:
            if len(allsegs) != len(self.levels) - 1:
                raise ValueError('must be one less number of segments as '
                                 'levels')
        else:
            if len(allsegs) != len(self.levels):
                raise ValueError('must be same number of segments as levels')

        # Check length of allkinds.
        if len(allkinds) != len(allsegs):
            raise ValueError('allkinds has different length to allsegs')

        # Determine x, y bounds and update axes data limits.
        flatseglist = [s for seg in allsegs for s in seg]
        points = np.concatenate(flatseglist, axis=0)
        self._mins = points.min(axis=0)
        self._maxs = points.max(axis=0)

        # Each entry in (allsegs, allkinds) is a list of (segs, kinds): segs is a list
        # of (N, 2) arrays of xy coordinates, kinds is a list of arrays of corresponding
        # pathcodes.  However, kinds can also be None; in which case all paths in that
        # list are codeless (this case is normalized above).  These lists are used to
        # construct paths, which then get concatenated.
        self._paths = [Path.make_compound_path(*map(Path, segs, kinds))
                       for segs, kinds in zip(allsegs, allkinds)]

        return kwargs

    def _make_paths_from_contour_generator(self):
        """Compute ``paths`` using C extension."""
        if self._paths is not None:
            return self._paths
        cg = self._contour_generator
        empty_path = Path(np.empty((0, 2)))
        vertices_and_codes = (
            map(cg.create_filled_contour, *self._get_lowers_and_uppers())
            if self.filled else
            map(cg.create_contour, self.levels))
        return [Path(np.concatenate(vs), np.concatenate(cs)) if len(vs) else empty_path
                for vs, cs in vertices_and_codes]

    def _get_lowers_and_uppers(self):
        """
        Return ``(lowers, uppers)`` for filled contours.
        """
        lowers = self._levels[:-1]
        if self.zmin == lowers[0]:
            # Include minimum values in lowest interval
            lowers = lowers.copy()  # so we don't change self._levels
            if self.logscale:
                lowers[0] = 0.99 * self.zmin
            else:
                lowers[0] -= 1
        uppers = self._levels[1:]
        return (lowers, uppers)

    def changed(self):
        if not hasattr(self, "cvalues"):
            self._process_colors()  # Sets cvalues.
        # Force an autoscale immediately because self.to_rgba() calls
        # autoscale_None() internally with the data passed to it,
        # so if vmin/vmax are not set yet, this would override them with
        # content from *cvalues* rather than levels like we want
        self.norm.autoscale_None(self.levels)
        self.set_array(self.cvalues)
        self.update_scalarmappable()
        alphas = np.broadcast_to(self.get_alpha(), len(self.cvalues))
        for label, cv, alpha in zip(self.labelTexts, self.labelCValues, alphas):
            label.set_alpha(alpha)
            label.set_color(self.labelMappable.to_rgba(cv))
        super().changed()

    def _autolev(self, N):
        """
        Select contour levels to span the data.

        The target number of levels, *N*, is used only when the
        scale is not log and default locator is used.

        We need two more levels for filled contours than for
        line contours, because for the latter we need to specify
        the lower and upper boundary of each range. For example,
        a single contour boundary, say at z = 0, requires only
        one contour line, but two filled regions, and therefore
        three levels to provide boundaries for both regions.
        """
        if self.locator is None:
            if self.logscale:
                self.locator = ticker.LogLocator()
            else:
                self.locator = ticker.MaxNLocator(N + 1, min_n_ticks=1)

        lev = self.locator.tick_values(self.zmin, self.zmax)

        try:
            if self.locator._symmetric:
                return lev
        except AttributeError:
            pass

        # Trim excess levels the locator may have supplied.
        under = np.nonzero(lev < self.zmin)[0]
        i0 = under[-1] if len(under) else 0
        over = np.nonzero(lev > self.zmax)[0]
        i1 = over[0] + 1 if len(over) else len(lev)
        if self.extend in ('min', 'both'):
            i0 += 1
        if self.extend in ('max', 'both'):
            i1 -= 1

        if i1 - i0 < 3:
            i0, i1 = 0, len(lev)

        return lev[i0:i1]

    def _process_contour_level_args(self, args, z_dtype):
        """
        Determine the contour levels and store in self.levels.
        """
        if self.levels is None:
            if args:
                levels_arg = args[0]
            elif np.issubdtype(z_dtype, bool):
                if self.filled:
                    levels_arg = [0, .5, 1]
                else:
                    levels_arg = [.5]
            else:
                levels_arg = 7  # Default, hard-wired.
        else:
            levels_arg = self.levels
        if isinstance(levels_arg, Integral):
            self.levels = self._autolev(levels_arg)
        else:
            self.levels = np.asarray(levels_arg, np.float64)
        if self.filled and len(self.levels) < 2:
            raise ValueError("Filled contours require at least 2 levels.")
        if len(self.levels) > 1 and np.min(np.diff(self.levels)) <= 0.0:
            raise ValueError("Contour levels must be increasing")

    def _process_levels(self):
        """
        Assign values to :attr:`layers` based on :attr:`levels`,
        adding extended layers as needed if contours are filled.

        For line contours, layers simply coincide with levels;
        a line is a thin layer.  No extended levels are needed
        with line contours.
        """
        # Make a private _levels to include extended regions; we
        # want to leave the original levels attribute unchanged.
        # (Colorbar needs this even for line contours.)
        self._levels = list(self.levels)

        if self.logscale:
            lower, upper = 1e-250, 1e250
        else:
