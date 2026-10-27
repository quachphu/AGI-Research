        """Parse a I[.F] seconds value into (seconds, microseconds)."""
        if "." not in value:
            return int(value), 0
        else:
            i, f = value.split(".")
            return int(i), int(f.ljust(6, "0")[:6])

    def _to_decimal(self, val):
        try:
            decimal_value = Decimal(val)
            # See GH 662, edge case, infinite value should not be converted
            #  via `_to_decimal`
            if not decimal_value.is_finite():
                raise ValueError("Converted decimal value is infinite or NaN")
        except Exception as e:
            msg = "Could not convert %s to decimal" % val
            six.raise_from(ValueError(msg), e)
        else:
            return decimal_value

    # ------------------------------------------------------------------
    # Post-Parsing construction of datetime output.  These are kept as
    #  methods instead of functions for the sake of customizability via
    #  subclassing.

    def _build_tzinfo(self, tzinfos, tzname, tzoffset):
        if callable(tzinfos):
            tzdata = tzinfos(tzname, tzoffset)
        else:
            tzdata = tzinfos.get(tzname)
        # handle case where tzinfo is paased an options that returns None
        # eg tzinfos = {'BRST' : None}
        if isinstance(tzdata, datetime.tzinfo) or tzdata is None:
            tzinfo = tzdata
        elif isinstance(tzdata, text_type):
            tzinfo = tz.tzstr(tzdata)
        elif isinstance(tzdata, integer_types):
            tzinfo = tz.tzoffset(tzname, tzdata)
        else:
            raise TypeError("Offset must be tzinfo subclass, tz string, "
                            "or int offset.")
        return tzinfo

    def _build_tzaware(self, naive, res, tzinfos):
        if (callable(tzinfos) or (tzinfos and res.tzname in tzinfos)):
            tzinfo = self._build_tzinfo(tzinfos, res.tzname, res.tzoffset)
            aware = naive.replace(tzinfo=tzinfo)
            aware = self._assign_tzname(aware, res.tzname)

        elif res.tzname and res.tzname in time.tzname:
            aware = naive.replace(tzinfo=tz.tzlocal())

            # Handle ambiguous local datetime
            aware = self._assign_tzname(aware, res.tzname)

            # This is mostly relevant for winter GMT zones parsed in the UK
            if (aware.tzname() != res.tzname and
                    res.tzname in self.info.UTCZONE):
                aware = aware.replace(tzinfo=tz.UTC)

        elif res.tzoffset == 0:
            aware = naive.replace(tzinfo=tz.UTC)

        elif res.tzoffset:
            aware = naive.replace(tzinfo=tz.tzoffset(res.tzname, res.tzoffset))

        elif not res.tzname and not res.tzoffset:
            # i.e. no timezone information was found.
            aware = naive

        elif res.tzname:
            # tz-like string was parsed but we don't know what to do
            # with it
            warnings.warn("tzname {tzname} identified but not understood.  "
                          "Pass `tzinfos` argument in order to correctly "
                          "return a timezone-aware datetime.  In a future "
                          "version, this will raise an "
                          "exception.".format(tzname=res.tzname),
                          category=UnknownTimezoneWarning)
            aware = naive

        return aware

    def _build_naive(self, res, default):
        repl = {}
        for attr in ("year", "month", "day", "hour",
                     "minute", "second", "microsecond"):
            value = getattr(res, attr)
            if value is not None:
                repl[attr] = value

        if 'day' not in repl:
            # If the default day exceeds the last day of the month, fall back
            # to the end of the month.
            cyear = default.year if res.year is None else res.year
            cmonth = default.month if res.month is None else res.month
            cday = default.day if res.day is None else res.day

            if cday > monthrange(cyear, cmonth)[1]:
                repl['day'] = monthrange(cyear, cmonth)[1]

        naive = default.replace(**repl)

        if res.weekday is not None and not res.day:
            naive = naive + relativedelta.relativedelta(weekday=res.weekday)

        return naive

    def _assign_tzname(self, dt, tzname):
        if dt.tzname() != tzname:
            new_dt = tz.enfold(dt, fold=1)
            if new_dt.tzname() == tzname:
                return new_dt

        return dt

    def _recombine_skipped(self, tokens, skipped_idxs):
        """
        >>> tokens = ["foo", " ", "bar", " ", "19June2000", "baz"]
        >>> skipped_idxs = [0, 1, 2, 5]
        >>> _recombine_skipped(tokens, skipped_idxs)
        ["foo bar", "baz"]
        """
        skipped_tokens = []
        for i, idx in enumerate(sorted(skipped_idxs)):
            if i > 0 and idx - 1 == skipped_idxs[i - 1]:
                skipped_tokens[-1] = skipped_tokens[-1] + tokens[idx]
            else:
                skipped_tokens.append(tokens[idx])

        return skipped_tokens


DEFAULTPARSER = parser()


def parse(timestr, parserinfo=None, **kwargs):
    """

    Parse a string in one of the supported formats, using the
    ``parserinfo`` parameters.

    :param timestr:
        A string containing a date/time stamp.

    :param parserinfo:
        A :class:`parserinfo` object containing parameters for the parser.
        If ``None``, the default arguments to the :class:`parserinfo`
        constructor are used.

    The ``**kwargs`` parameter takes the following keyword arguments:

    :param default:
        The default datetime object, if this is a datetime object and not
        ``None``, elements specified in ``timestr`` replace elements in the
        default object.

    :param ignoretz:
        If set ``True``, time zones in parsed strings are ignored and a naive
        :class:`datetime` object is returned.

    :param tzinfos:
        Additional time zone names / aliases which may be present in the
        string. This argument maps time zone names (and optionally offsets
        from those time zones) to time zones. This parameter can be a
        dictionary with timezone aliases mapping time zone names to time
        zones or a function taking two parameters (``tzname`` and
        ``tzoffset``) and returning a time zone.

        The timezones to which the names are mapped can be an integer
        offset from UTC in seconds or a :class:`tzinfo` object.

        .. doctest::
           :options: +NORMALIZE_WHITESPACE

            >>> from dateutil.parser import parse
            >>> from dateutil.tz import gettz
            >>> tzinfos = {"BRST": -7200, "CST": gettz("America/Chicago")}
            >>> parse("2012-01-19 17:21:00 BRST", tzinfos=tzinfos)
            datetime.datetime(2012, 1, 19, 17, 21, tzinfo=tzoffset(u'BRST', -7200))
            >>> parse("2012-01-19 17:21:00 CST", tzinfos=tzinfos)
            datetime.datetime(2012, 1, 19, 17, 21,
                              tzinfo=tzfile('/usr/share/zoneinfo/America/Chicago'))

        This parameter is ignored if ``ignoretz`` is set.

    :param dayfirst:
        Whether to interpret the first value in an ambiguous 3-integer date
        (e.g. 01/05/09) as the day (``True``) or month (``False``). If
        ``yearfirst`` is set to ``True``, this distinguishes between YDM and
        YMD. If set to ``None``, this value is retrieved from the current
        :class:`parserinfo` object (which itself defaults to ``False``).

    :param yearfirst:
        Whether to interpret the first value in an ambiguous 3-integer date
        (e.g. 01/05/09) as the year. If ``True``, the first number is taken to
        be the year, otherwise the last number is taken to be the year. If
        this is set to ``None``, the value is retrieved from the current
        :class:`parserinfo` object (which itself defaults to ``False``).

    :param fuzzy:
        Whether to allow fuzzy parsing, allowing for string like "Today is
        January 1, 2047 at 8:21:00AM".

    :param fuzzy_with_tokens:
        If ``True``, ``fuzzy`` is automatically set to True, and the parser
        will return a tuple where the first element is the parsed
        :class:`datetime.datetime` datetimestamp and the second element is
        a tuple containing the portions of the string which were ignored:

        .. doctest::

            >>> from dateutil.parser import parse
            >>> parse("Today is January 1, 2047 at 8:21:00AM", fuzzy_with_tokens=True)
            (datetime.datetime(2047, 1, 1, 8, 21), (u'Today is ', u' ', u'at '))

    :return:
        Returns a :class:`datetime.datetime` object or, if the
        ``fuzzy_with_tokens`` option is ``True``, returns a tuple, the
        first element being a :class:`datetime.datetime` object, the second
        a tuple containing the fuzzy tokens.

    :raises ParserError:
        Raised for invalid or unknown string formats, if the provided
        :class:`tzinfo` is not in a valid format, or if an invalid date would
        be created.

    :raises OverflowError:
        Raised if the parsed date exceeds the largest valid C integer on
        your system.
    """
    if parserinfo:
        return parser(parserinfo).parse(timestr, **kwargs)
    else:
        return DEFAULTPARSER.parse(timestr, **kwargs)


class _tzparser(object):

    class _result(_resultbase):

        __slots__ = ["stdabbr", "stdoffset", "dstabbr", "dstoffset",
                     "start", "end"]

        class _attr(_resultbase):
            __slots__ = ["month", "week", "weekday",
                         "yday", "jyday", "day", "time"]

        def __repr__(self):
            return self._repr("")

        def __init__(self):
            _resultbase.__init__(self)
            self.start = self._attr()
            self.end = self._attr()

    def parse(self, tzstr):
        res = self._result()
        l = [x for x in re.split(r'([,:.]|[a-zA-Z]+|[0-9]+)',tzstr) if x]
        used_idxs = list()
        try:

            len_l = len(l)

            i = 0
            while i < len_l:
                # BRST+3[BRDT[+2]]
                j = i
                while j < len_l and not [x for x in l[j]
                                         if x in "0123456789:,-+"]:
                    j += 1
                if j != i:
                    if not res.stdabbr:
                        offattr = "stdoffset"
                        res.stdabbr = "".join(l[i:j])
                    else:
                        offattr = "dstoffset"
                        res.dstabbr = "".join(l[i:j])

                    for ii in range(j):
                        used_idxs.append(ii)
                    i = j
                    if (i < len_l and (l[i] in ('+', '-') or l[i][0] in
                                       "0123456789")):
                        if l[i] in ('+', '-'):
                            # Yes, that's right.  See the TZ variable
                            # documentation.
                            signal = (1, -1)[l[i] == '+']
                            used_idxs.append(i)
                            i += 1
                        else:
                            signal = -1
                        len_li = len(l[i])
                        if len_li == 4:
                            # -0300
                            setattr(res, offattr, (int(l[i][:2]) * 3600 +
                                                   int(l[i][2:]) * 60) * signal)
                        elif i + 1 < len_l and l[i + 1] == ':':
                            # -03:00
                            setattr(res, offattr,
                                    (int(l[i]) * 3600 +
                                     int(l[i + 2]) * 60) * signal)
                            used_idxs.append(i)
                            i += 2
                        elif len_li <= 2:
                            # -[0]3
                            setattr(res, offattr,
                                    int(l[i][:2]) * 3600 * signal)
                        else:
                            return None
                        used_idxs.append(i)
                        i += 1
                    if res.dstabbr:
                        break
                else:
                    break


            if i < len_l:
                for j in range(i, len_l):
                    if l[j] == ';':
                        l[j] = ','

                assert l[i] == ','

                i += 1

            if i >= len_l:
                pass
            elif (8 <= l.count(',') <= 9 and
                  not [y for x in l[i:] if x != ','
                       for y in x if y not in "0123456789+-"]):
                # GMT0BST,3,0,30,3600,10,0,26,7200[,3600]
                for x in (res.start, res.end):
                    x.month = int(l[i])
                    used_idxs.append(i)
                    i += 2
                    if l[i] == '-':
                        value = int(l[i + 1]) * -1
                        used_idxs.append(i)
                        i += 1
                    else:
                        value = int(l[i])
                    used_idxs.append(i)
                    i += 2
                    if value:
                        x.week = value
                        x.weekday = (int(l[i]) - 1) % 7
                    else:
                        x.day = int(l[i])
                    used_idxs.append(i)
                    i += 2
                    x.time = int(l[i])
                    used_idxs.append(i)
                    i += 2
                if i < len_l:
                    if l[i] in ('-', '+'):
                        signal = (-1, 1)[l[i] == "+"]
                        used_idxs.append(i)
                        i += 1
                    else:
                        signal = 1
                    used_idxs.append(i)
                    res.dstoffset = (res.stdoffset + int(l[i]) * signal)

                # This was a made-up format that is not in normal use
                warn(('Parsed time zone "%s"' % tzstr) +
                     'is in a non-standard dateutil-specific format, which ' +
                     'is now deprecated; support for parsing this format ' +
                     'will be removed in future versions. It is recommended ' +
                     'that you switch to a standard format like the GNU ' +
                     'TZ variable format.', tz.DeprecatedTzFormatWarning)
            elif (l.count(',') == 2 and l[i:].count('/') <= 2 and
                  not [y for x in l[i:] if x not in (',', '/', 'J', 'M',
                                                     '.', '-', ':')
                       for y in x if y not in "0123456789"]):
                for x in (res.start, res.end):
                    if l[i] == 'J':
                        # non-leap year day (1 based)
                        used_idxs.append(i)
                        i += 1
                        x.jyday = int(l[i])
                    elif l[i] == 'M':
                        # month[-.]week[-.]weekday
                        used_idxs.append(i)
                        i += 1
                        x.month = int(l[i])
                        used_idxs.append(i)
                        i += 1
                        assert l[i] in ('-', '.')
                        used_idxs.append(i)
                        i += 1
                        x.week = int(l[i])
                        if x.week == 5:
                            x.week = -1
                        used_idxs.append(i)
                        i += 1
                        assert l[i] in ('-', '.')
                        used_idxs.append(i)
                        i += 1
                        x.weekday = (int(l[i]) - 1) % 7
                    else:
                        # year day (zero based)
                        x.yday = int(l[i]) + 1

                    used_idxs.append(i)
                    i += 1

                    if i < len_l and l[i] == '/':
                        used_idxs.append(i)
                        i += 1
                        # start time
                        len_li = len(l[i])
                        if len_li == 4:
                            # -0300
                            x.time = (int(l[i][:2]) * 3600 +
                                      int(l[i][2:]) * 60)
                        elif i + 1 < len_l and l[i + 1] == ':':
                            # -03:00
                            x.time = int(l[i]) * 3600 + int(l[i + 2]) * 60
                            used_idxs.append(i)
                            i += 2
                            if i + 1 < len_l and l[i + 1] == ':':
                                used_idxs.append(i)
                                i += 2
                                x.time += int(l[i])
                        elif len_li <= 2:
                            # -[0]3
                            x.time = (int(l[i][:2]) * 3600)
                        else:
                            return None
                        used_idxs.append(i)
                        i += 1

                    assert i == len_l or l[i] == ','

                    i += 1

                assert i >= len_l

        except (IndexError, ValueError, AssertionError):
            return None

        unused_idxs = set(range(len_l)).difference(used_idxs)
        res.any_unused_tokens = not {l[n] for n in unused_idxs}.issubset({",",":"})
        return res


DEFAULTTZPARSER = _tzparser()


def _parsetz(tzstr):
    return DEFAULTTZPARSER.parse(tzstr)


class ParserError(ValueError):
    """Exception subclass used for any failure to parse a datetime string.

    This is a subclass of :py:exc:`ValueError`, and should be raised any time
    earlier versions of ``dateutil`` would have raised ``ValueError``.

    .. versionadded:: 2.8.1
    """
    def __str__(self):
        try:
            return self.args[0] % self.args[1:]
        except (TypeError, IndexError):
            return super(ParserError, self).__str__()

    def __repr__(self):
        args = ", ".join("'%s'" % arg for arg in self.args)
        return "%s(%s)" % (self.__class__.__name__, args)


class UnknownTimezoneWarning(RuntimeWarning):
    """Raised when the parser finds a timezone it cannot parse into a tzinfo.

    .. versionadded:: 2.7.0
    """
# vim:ts=4:sw=4:et
<|endoftext|># -*- coding: utf-8 -*-
import warnings
import json

from tarfile import TarFile
from pkgutil import get_data
from io import BytesIO

from dateutil.tz import tzfile as _tzfile

__all__ = ["get_zonefile_instance", "gettz", "gettz_db_metadata"]

ZONEFILENAME = "dateutil-zoneinfo.tar.gz"
METADATA_FN = 'METADATA'


class tzfile(_tzfile):
    def __reduce__(self):
        return (gettz, (self._filename,))


def getzoneinfofile_stream():
    try:
        return BytesIO(get_data(__name__, ZONEFILENAME))
    except IOError as e:  # TODO  switch to FileNotFoundError?
        warnings.warn("I/O error({0}): {1}".format(e.errno, e.strerror))
        return None


class ZoneInfoFile(object):
    def __init__(self, zonefile_stream=None):
        if zonefile_stream is not None:
            with TarFile.open(fileobj=zonefile_stream) as tf:
                self.zones = {zf.name: tzfile(tf.extractfile(zf), filename=zf.name)
                              for zf in tf.getmembers()
                              if zf.isfile() and zf.name != METADATA_FN}
                # deal with links: They'll point to their parent object. Less
                # waste of memory
                links = {zl.name: self.zones[zl.linkname]
                         for zl in tf.getmembers() if
                         zl.islnk() or zl.issym()}
                self.zones.update(links)
                try:
                    metadata_json = tf.extractfile(tf.getmember(METADATA_FN))
                    metadata_str = metadata_json.read().decode('UTF-8')
                    self.metadata = json.loads(metadata_str)
                except KeyError:
                    # no metadata in tar file
                    self.metadata = None
        else:
            self.zones = {}
            self.metadata = None

    def get(self, name, default=None):
        """
        Wrapper for :func:`ZoneInfoFile.zones.get`. This is a convenience method
        for retrieving zones from the zone dictionary.

        :param name:
            The name of the zone to retrieve. (Generally IANA zone names)

        :param default:
            The value to return in the event of a missing key.

        .. versionadded:: 2.6.0

        """
        return self.zones.get(name, default)


# The current API has gettz as a module function, although in fact it taps into
# a stateful class. So as a workaround for now, without changing the API, we
# will create a new "global" class instance the first time a user requests a
# timezone. Ugly, but adheres to the api.
#
# TODO: Remove after deprecation period.
_CLASS_ZONE_INSTANCE = []


def get_zonefile_instance(new_instance=False):
    """
    This is a convenience function which provides a :class:`ZoneInfoFile`
    instance using the data provided by the ``dateutil`` package. By default, it
    caches a single instance of the ZoneInfoFile object and returns that.

    :param new_instance:
        If ``True``, a new instance of :class:`ZoneInfoFile` is instantiated and
        used as the cached instance for the next call. Otherwise, new instances
        are created only as necessary.

    :return:
        Returns a :class:`ZoneInfoFile` object.

    .. versionadded:: 2.6
    """
    if new_instance:
        zif = None
    else:
        zif = getattr(get_zonefile_instance, '_cached_instance', None)

    if zif is None:
        zif = ZoneInfoFile(getzoneinfofile_stream())

        get_zonefile_instance._cached_instance = zif

    return zif


def gettz(name):
    """
    This retrieves a time zone from the local zoneinfo tarball that is packaged
    with dateutil.

    :param name:
        An IANA-style time zone name, as found in the zoneinfo file.

    :return:
        Returns a :class:`dateutil.tz.tzfile` time zone object.

    .. warning::
        It is generally inadvisable to use this function, and it is only
        provided for API compatibility with earlier versions. This is *not*
        equivalent to ``dateutil.tz.gettz()``, which selects an appropriate
        time zone based on the inputs, favoring system zoneinfo. This is ONLY
        for accessing the dateutil-specific zoneinfo (which may be out of
        date compared to the system zoneinfo).

    .. deprecated:: 2.6
        If you need to use a specific zoneinfofile over the system zoneinfo,
        instantiate a :class:`dateutil.zoneinfo.ZoneInfoFile` object and call
        :func:`dateutil.zoneinfo.ZoneInfoFile.get(name)` instead.

        Use :func:`get_zonefile_instance` to retrieve an instance of the
        dateutil-provided zoneinfo.
    """
    warnings.warn("zoneinfo.gettz() will be removed in future versions, "
                  "to use the dateutil-provided zoneinfo files, instantiate a "
                  "ZoneInfoFile object and use ZoneInfoFile.zones.get() "
                  "instead. See the documentation for details.",
                  DeprecationWarning)

    if len(_CLASS_ZONE_INSTANCE) == 0:
        _CLASS_ZONE_INSTANCE.append(ZoneInfoFile(getzoneinfofile_stream()))
    return _CLASS_ZONE_INSTANCE[0].zones.get(name)


def gettz_db_metadata():
    """ Get the zonefile metadata

    See `zonefile_metadata`_

    :returns:
        A dictionary with the database metadata

    .. deprecated:: 2.6
        See deprecation warning in :func:`zoneinfo.gettz`. To get metadata,
        query the attribute ``zoneinfo.ZoneInfoFile.metadata``.
    """
    warnings.warn("zoneinfo.gettz_db_metadata() will be removed in future "
                  "versions, to use the dateutil-provided zoneinfo files, "
                  "ZoneInfoFile object and query the 'metadata' attribute "
                  "instead. See the documentation for details.",
                  DeprecationWarning)

    if len(_CLASS_ZONE_INSTANCE) == 0:
        _CLASS_ZONE_INSTANCE.append(ZoneInfoFile(getzoneinfofile_stream()))
    return _CLASS_ZONE_INSTANCE[0].metadata
<|endoftext|>import logging
import os
import tempfile
import shutil
import json
from subprocess import check_call, check_output
from tarfile import TarFile

from dateutil.zoneinfo import METADATA_FN, ZONEFILENAME


def rebuild(filename, tag=None, format="gz", zonegroups=[], metadata=None):
    """Rebuild the internal timezone info in dateutil/zoneinfo/zoneinfo*tar*

    filename is the timezone tarball from ``ftp.iana.org/tz``.

    """
    tmpdir = tempfile.mkdtemp()
    zonedir = os.path.join(tmpdir, "zoneinfo")
    moduledir = os.path.dirname(__file__)
    try:
        with TarFile.open(filename) as tf:
            for name in zonegroups:
                tf.extract(name, tmpdir)
            filepaths = [os.path.join(tmpdir, n) for n in zonegroups]

            _run_zic(zonedir, filepaths)

        # write metadata file
        with open(os.path.join(zonedir, METADATA_FN), 'w') as f:
            json.dump(metadata, f, indent=4, sort_keys=True)
        target = os.path.join(moduledir, ZONEFILENAME)
        with TarFile.open(target, "w:%s" % format) as tf:
            for entry in os.listdir(zonedir):
                entrypath = os.path.join(zonedir, entry)
                tf.add(entrypath, entry)
    finally:
        shutil.rmtree(tmpdir)


def _run_zic(zonedir, filepaths):
    """Calls the ``zic`` compiler in a compatible way to get a "fat" binary.

    Recent versions of ``zic`` default to ``-b slim``, while older versions
    don't even have the ``-b`` option (but default to "fat" binaries). The
    current version of dateutil does not support Version 2+ TZif files, which
    causes problems when used in conjunction with "slim" binaries, so this
    function is used to ensure that we always get a "fat" binary.
    """

    try:
        help_text = check_output(["zic", "--help"])
    except OSError as e:
        _print_on_nosuchfile(e)
        raise

    if b"-b " in help_text:
        bloat_args = ["-b", "fat"]
    else:
        bloat_args = []

    check_call(["zic"] + bloat_args + ["-d", zonedir] + filepaths)


def _print_on_nosuchfile(e):
    """Print helpful troubleshooting message

    e is an exception raised by subprocess.check_call()

    """
    if e.errno == 2:
        logging.error(
            "Could not find zic. Perhaps you need to install "
            "libc-bin or some other package that provides it, "
            "or it's not in your PATH?")
<|endoftext|># --------------------------------------------------------------------------------------
# Copyright (c) 2023-2024, Nucleic Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# --------------------------------------------------------------------------------------
"""Kiwi exceptions.

Imported by the kiwisolver C extension.

"""


class BadRequiredStrength(Exception):
    pass


class DuplicateConstraint(Exception):
    __slots__ = ("constraint",)

    def __init__(self, constraint):
        self.constraint = constraint


class DuplicateEditVariable(Exception):
    __slots__ = ("edit_variable",)

    def __init__(self, edit_variable):
        self.edit_variable = edit_variable


class UnknownConstraint(Exception):
    __slots__ = ("constraint",)

    def __init__(self, constraint):
        self.constraint = constraint


class UnknownEditVariable(Exception):
    __slots__ = ("edit_variable",)

    def __init__(self, edit_variable):
        self.edit_variable = edit_variable


class UnsatisfiableConstraint(Exception):
    __slots__ = ("constraint",)

    def __init__(self, constraint):
        self.constraint = constraint
<|endoftext|># SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cuda.bindings import driver as cuda
from cuda.bindings import nvrtc
from cuda.bindings import runtime as cudart

from .helper_string import check_cmd_line_flag, get_cmd_line_argument_int


def _cuda_get_error_enum(error):
    if isinstance(error, cuda.CUresult):
        err, name = cuda.cuGetErrorName(error)
        return name if err == cuda.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, cudart.cudaError_t):
        return cudart.cudaGetErrorName(error)[1]
    elif isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        raise RuntimeError(f"Unknown error type: {error}")


def check_cuda_errors(result):
    if result[0].value:
        raise RuntimeError(f"CUDA error code={result[0].value}({_cuda_get_error_enum(result[0])})")
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]


def find_cuda_device():
    dev_id = 0
    if check_cmd_line_flag("device="):
        dev_id = get_cmd_line_argument_int("device=")
    check_cuda_errors(cudart.cudaSetDevice(dev_id))
    return dev_id


def find_cuda_device_drv():
    dev_id = 0
    if check_cmd_line_flag("device="):
        dev_id = get_cmd_line_argument_int("device=")
    check_cuda_errors(cuda.cuInit(0))
    cu_device = check_cuda_errors(cuda.cuDeviceGet(dev_id))
    return cu_device
<|endoftext|># SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import os
import sys

import numpy as np

from cuda import pathfinder
from cuda.bindings import driver as cuda
from cuda.bindings import runtime as cudart
from cuda.bindings._v2 import nvrtc

from .helper_cuda import check_cuda_errors


def requirement_not_met(message):
    print(message, file=sys.stderr)  # noqa: T201
    exitcode = os.environ.get("CUDA_BINDINGS_SKIP_EXAMPLE", "1")
    return sys.exit(int(exitcode))


def check_compute_capability_too_low(dev_id, required_cc_major_minor):
    cc_major = check_cuda_errors(
        cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor, dev_id)
    )
    cc_minor = check_cuda_errors(
        cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor, dev_id)
    )
    have_cc_major_minor = (cc_major, cc_minor)
    if have_cc_major_minor < required_cc_major_minor:
        requirement_not_met(
            f"CUDA device compute capability too low: {have_cc_major_minor=!r}, {required_cc_major_minor=!r}"
        )


class KernelHelper:
    def __init__(self, code, dev_id):
        include_dirs = []
        for libname in ("cudart", "cccl"):
            hdr_dir = pathfinder.find_nvidia_header_directory(libname)
            if hdr_dir is None:
                requirement_not_met(f'pathfinder.find_nvidia_header_directory("{libname}") returned None')
            include_dirs.append(hdr_dir)

        prog = nvrtc.create_program(str.encode(code), b"sourceCode.cu")

        # Initialize CUDA
        check_cuda_errors(cudart.cudaFree(0))

        major = check_cuda_errors(
            cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor, dev_id)
        )
        minor = check_cuda_errors(
            cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor, dev_id)
        )
        _, nvrtc_minor = nvrtc.version()
        use_cubin = nvrtc_minor >= 1
        prefix = "sm" if use_cubin else "compute"
        arch_arg = bytes(f"--gpu-architecture={prefix}_{major}{minor}", "ascii")

        opts = [
            b"--fmad=true",
            arch_arg,
            b"--std=c++17",
            b"-default-device",
        ]
        for inc_dir in include_dirs:
            opts.append(f"--include-path={inc_dir}".encode())

        try:
            nvrtc.compile_program(prog, opts)
        except nvrtc.NvrtcError as err:
            log = nvrtc.get_program_log(prog)
            import sys

            print(log.decode(), file=sys.stderr)  # noqa: T201
            print(err, file=sys.stderr)  # noqa: T201
            sys.exit(1)

        if use_cubin:
            data = nvrtc.get_cubin(prog)
        else:
            data = nvrtc.get_ptx(prog)

        self.module = check_cuda_errors(cuda.cuModuleLoadData(np.char.array(data)))

    def get_function(self, name):
        return check_cuda_errors(cuda.cuModuleGetFunction(self.module, name))
<|endoftext|># SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from .common import KernelHelper, check_compute_capability_too_low, requirement_not_met
from .helper_cuda import check_cuda_errors, find_cuda_device, find_cuda_device_drv
from .helper_string import check_cmd_line_flag, get_cmd_line_argument_int

__all__ = [
    "KernelHelper",
    "check_cmd_line_flag",
    "check_compute_capability_too_low",
    "check_cuda_errors",
    "find_cuda_device",
    "find_cuda_device_drv",
    "get_cmd_line_argument_int",
    "requirement_not_met",
]
<|endoftext|># SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys


def check_cmd_line_flag(string_ref):
    return any(string_ref == i and k < len(sys.argv) - 1 for i, k in enumerate(sys.argv))


def get_cmd_line_argument_int(string_ref):
    for i, k in enumerate(sys.argv):
        if string_ref == i and k < len(sys.argv) - 1:
            return sys.argv[k + 1]
    return 0
<|endoftext|># SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Sequence

_PRECHECK_NVVM_IR = """target triple = "nvptx64-unknown-cuda"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"

define void @dummy_kernel() {{
entry:
  ret void
}}

!nvvm.annotations = !{{!0}}
!0 = !{{void ()* @dummy_kernel, !"kernel", i32 1}}

!nvvmir.version = !{{!1}}
!1 = !{{i32 {major}, i32 {minor}, i32 {debug_major}, i32 {debug_minor}}}
"""


def check_nvvm_compiler_options(options: Sequence[str]) -> bool:
    """
    Abstracted from https://github.com/NVIDIA/numba-cuda/pull/681

    Check if the specified options are supported by the current libNVVM version.

    The options are a list of strings, each representing a compiler option.

    If the test program fails to compile, the options are not supported and False
    is returned.

    If the test program compiles successfully, True is returned.

    cuda.bindings.nvvm returns exceptions instead of return codes.

    Parameters
    ----------
    options : Sequence[str]
        List of compiler options as strings (e.g., ["-arch=compute_90", "-g"]).

    Returns
    -------
    bool
        True if the options are supported, False otherwise.

    Examples
    --------
    >>> from cuda.bindings.utils import check_nvvm_compiler_options
    >>> check_nvvm_compiler_options(["-arch=compute_90", "-g"])
    True
    """
    try:
        from cuda.bindings import nvvm
    except ModuleNotFoundError as exc:
        if exc.name == "nvvm":
            return False
        raise

    from cuda.bindings._internal.nvvm import _inspect_function_pointer

    if _inspect_function_pointer("__nvvmCreateProgram") == 0:
        return False

    program = nvvm.create_program()
    try:
        major, minor, debug_major, debug_minor = nvvm.ir_version()
        precheck_ir = _PRECHECK_NVVM_IR.format(
            major=major,
            minor=minor,
            debug_major=debug_major,
            debug_minor=debug_minor,
        )
        precheck_ir_bytes = precheck_ir.encode("utf-8")
        nvvm.add_module_to_program(
            program,
            precheck_ir_bytes,
            len(precheck_ir_bytes),
            "precheck.ll",
        )
        try:
            nvvm.compile_program(program, len(options), options)
        except nvvm.nvvmError as e:
            if e.status == nvvm.Result.ERROR_INVALID_OPTION:
                return False
            raise
    finally:
        nvvm.destroy_program(program)
    return True
<|endoftext|># SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
import warnings

from ._envvar import envvar_bool

# Track whether we've already checked major version compatibility
_major_version_compatibility_checked = False
_lock = threading.Lock()

_DISABLE_WARNING_ENV_VAR = "CUDA_PYTHON_DISABLE_MAJOR_VERSION_WARNING"


def warn_if_cuda_major_version_mismatch():
    """Warn if the CUDA driver major version is older than cuda-bindings compile-time version.

    This function compares the CUDA major version that cuda-bindings was compiled
    against with the CUDA major version supported by the installed driver. If the
    compile-time major version is greater than the driver's major version, a warning
    is issued.

    The check runs only once per process. Subsequent calls are no-ops.

    The warning can be suppressed by setting the environment variable
    ``CUDA_PYTHON_DISABLE_MAJOR_VERSION_WARNING=1``. Setting it to ``0`` (or
    leaving it unset or empty) keeps the warning enabled.
    """
    global _major_version_compatibility_checked
    if _major_version_compatibility_checked:
        return
    with _lock:
        if _major_version_compatibility_checked:
            return
        _major_version_compatibility_checked = True

    # Allow users to suppress the warning
    if envvar_bool(_DISABLE_WARNING_ENV_VAR):
        return

    # Import here to avoid circular imports and allow lazy loading
    from cuda.bindings import driver

    # Get compile-time CUDA version from cuda-bindings
    compile_version = driver.CUDA_VERSION  # e.g., 13010
    compile_major = compile_version // 1000

    # Get runtime driver version
    err, runtime_version = driver.cuDriverGetVersion()
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to query CUDA driver version: {err}")

    runtime_major = runtime_version // 1000

    if compile_major > runtime_major:
        warnings.warn(
            f"cuda-bindings was built for CUDA major version {compile_major}, but the "
            f"NVIDIA driver only supports up to CUDA {runtime_major}. Some cuda-bindings "
            f"features may not work correctly. Consider updating your NVIDIA driver, "
            f"or using a cuda-bindings version built for CUDA {runtime_major}. "
            f"(Set {_DISABLE_WARNING_ENV_VAR}=1 to suppress this warning.)",
            UserWarning,
            stacklevel=3,
        )
<|endoftext|># SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Callable

from ._envvar import envvar_bool
from ._nvvm_utils import check_nvvm_compiler_options
from ._ptx_utils import get_minimal_required_cuda_ver_from_ptx_ver, get_ptx_ver
from ._version_check import warn_if_cuda_major_version_mismatch

_handle_getters: dict[type, Callable[[Any], int]] = {}


def _add_cuda_native_handle_getter(t: type, getter: Callable[[Any], int]) -> None:
    _handle_getters[t] = getter


def get_cuda_native_handle(obj: Any) -> int:
    """Returns the address of the provided CUDA Python object as a Python int.

    Parameters
    ----------
    obj : Any
        CUDA Python object

    Returns
    -------
    int : The object address.
    """
    obj_type = type(obj)
    try:
        getter = _handle_getters[obj_type]
    except KeyError:
        raise TypeError("Unknown type: " + str(obj_type)) from None
    # Deliberately outside the try: a KeyError raised by the getter itself is a
    # bug in that getter, not an unregistered type.
    return getter(obj)
<|endoftext|># SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


def envvar_bool(name: str, default: bool = False) -> bool:
    """Read a bool-like environment variable.

    Unset, empty, or whitespace-only means ``default``. ``1/true/yes/on`` and
    ``0/false/no/off`` are recognised case-insensitively, and any other integer
    follows C truthiness, so ``2`` is true and ``-0`` is false.

    A value that is none of those keeps the historical set-means-true
    behaviour rather than raising, because these variables are read during
    import and a raise would turn a typo into an import failure.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip()
    if not raw:
        return default
    lowered = raw.lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    try:
        return int(raw, 0) != 0
    except ValueError:
        return True
<|endoftext|># SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re

# Mapping based on the official PTX ISA <-> CUDA Release table
# https://docs.nvidia.com/cuda/parallel-thread-execution/#release-notes-ptx-release-history
_ptx_to_cuda = {
    "1.0": (1, 0),
    "1.1": (1, 1),
    "1.2": (2, 0),
    "1.3": (2, 1),
    "1.4": (2, 2),
    "2.0": (3, 0),
    "2.1": (3, 1),
    "2.2": (3, 2),
    "2.3": (4, 0),
    "3.0": (4, 1),
    "3.1": (5, 0),
    "3.2": (5, 5),
    "4.0": (6, 0),
    "4.1": (6, 5),
    "4.2": (7, 0),
    "4.3": (7, 5),
    "5.0": (8, 0),
    "6.0": (9, 0),
    "6.1": (9, 1),
    "6.2": (9, 2),
    "6.3": (10, 0),
    "6.4": (10, 1),
    "6.5": (10, 2),
    "7.0": (11, 0),
    "7.1": (11, 1),
    "7.2": (11, 2),
    "7.3": (11, 3),
    "7.4": (11, 4),
    "7.5": (11, 5),
    "7.6": (11, 6),
    "7.7": (11, 7),
    "7.8": (11, 8),
    "8.0": (12, 0),
    "8.1": (12, 1),
    "8.2": (12, 2),
    "8.3": (12, 3),
    "8.4": (12, 4),
    "8.5": (12, 5),
    "8.6": (12, 7),
    "8.7": (12, 8),
    "8.8": (12, 9),
    "9.0": (13, 0),
    "9.1": (13, 1),
    "9.2": (13, 2),
    "9.3": (13, 3),
}


def get_minimal_required_cuda_ver_from_ptx_ver(ptx_version: str) -> int:
    """
    Maps the PTX ISA version to the minimal CUDA driver, nvPTXCompiler, or nvJitLink version
    that is needed to load a PTX of the given ISA version.

    Parameters
    ----------
    ptx_version : str
        PTX ISA version as a string, e.g. "8.8" for PTX ISA 8.8. This is the ``.version``
        directive in the PTX header.

    Returns
    -------
    int
        Minimal CUDA version as 1000 * major + 10 * minor, e.g. 12090 for CUDA 12.9.

    Raises
    ------
    ValueError
        If the PTX version is unknown.

    Examples
    --------
    >>> get_minimal_required_driver_ver_from_ptx_ver("8.8")
    12090
    >>> get_minimal_required_driver_ver_from_ptx_ver("7.0")
    11000
    """
    try:
        major, minor = _ptx_to_cuda[ptx_version]
        return 1000 * major + 10 * minor
    except KeyError:
        raise ValueError(f"Unknown or unsupported PTX ISA version: {ptx_version}") from None


# Regex pattern to match .version directive and capture the version number
# TODO: if import speed is a concern, consider lazy-initializing it.
_ptx_ver_pattern = re.compile(r"\.version\s+([0-9]+\.[0-9]+)")


def get_ptx_ver(ptx: str) -> str:
    """
    Extract the PTX ISA version string from PTX source code.

    Parameters
    ----------
    ptx : str
        The PTX assembly source code as a string.

    Returns
    -------
    str
        The PTX ISA version string, e.g., "8.8".

    Raises
    ------
    ValueError
        If the .version directive is not found in the PTX source.

    Examples
    --------
    >>> ptx = r'''
    ... .version 8.8
    ... .target sm_86
    ... .address_size 64
    ...
    ... .visible .entry test_kernel()
    ... {
    ...     ret;
    ... }
    ... '''
    >>> get_ptx_ver(ptx)
    '8.8'
    """
    m = _ptx_ver_pattern.search(ptx)
    if m:
        return m.group(1)
    else:
        raise ValueError("No .version directive found in PTX source. Is it a valid PTX?")
<|endoftext|># SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# CYTHON-BINDINGS-GENERATED-DO-NOT-MODIFY-THIS-FILE: format=1; content-sha256=581469c1fadb5f72c43b478d73f2b905562136c8723a25e2f4240b5b681e2894
"""
This is a replacement for the stdlib enum.IntEnum.

Notably, it has much better import time performance, since it doesn't generate
and evaluate Python code at startup time.

It supports the most important subset of the IntEnum API.  See `test_enum` in
`cuda_bindings/tests/test_basics.py` for details.
"""

from typing import Any, Iterator


class FastEnumMetaclass(type):
    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)

        cls.__singletons__ = {}
        cls.__members__ = {}
        for name, value in cls.__dict__.items():
            if name.startswith("__") and name.endswith("__"):
                continue

            if isinstance(value, tuple):
                value, doc = value
            elif isinstance(value, int):
                doc = None
            else:
                continue

            singleton = int.__new__(cls, value)
            singleton.__doc__ = doc
            singleton._name = name
            cls.__singletons__[value] = singleton
            cls.__members__[name] = singleton

        for name, member in cls.__members__.items():
            setattr(cls, name, member)

    def __repr__(cls) -> str:
        return f"<enum '{cls.__name__}'>"

    def __len__(cls) -> int:
        return len(cls.__members__)

    def __iter__(cls) -> Iterator["FastEnum"]:
        return iter(cls.__members__.values())

    def __contains__(cls, item: Any) -> bool:
        return item in cls.__singletons__


class FastEnum(int, metaclass=FastEnumMetaclass):
    def __new__(cls, value: int) -> "FastEnum":
        singleton: FastEnum = cls.__singletons__.get(value)
        if singleton is None:
            raise ValueError(f"{value} is not a valid {cls.__name__}")
        return singleton

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}.{self._name}: {int(self)}>"

    @property
    def name(self) -> str:
        return self._name

    @property
    def value(self) -> int:
        return int(self)
<|endoftext|># SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
from collections.abc import Callable
from types import ModuleType

from cuda.pathfinder._dynamic_libs.load_dl_common import DynamicLibNotFoundError


def _optional_cuda_import(
    fully_qualified_modname: str,
    *,
    probe_function: Callable[[ModuleType], object] | None = None,
) -> ModuleType | None:
    """Import an optional CUDA module without masking unrelated import bugs.

    Returns:
        The imported module if available and the optional probe succeeds,
        otherwise ``None`` when the requested module is unavailable.

    Raises:
        ModuleNotFoundError: If the import fails because a dependency of the
            target module is missing (instead of the target module itself).
        Exception: Any exception raised by ``probe_function`` except
            :class:`DynamicLibNotFoundError`, which is treated as "unavailable".
    """
    try:
        module = importlib.import_module(fully_qualified_modname)
    except ModuleNotFoundError as err:
        if err.name != fully_qualified_modname:
            raise
        return None

    if probe_function is not None:
        try:
            probe_function(module)
        except DynamicLibNotFoundError:
            return None

    return module
<|endoftext|># SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cuda.pathfinder public APIs"""

# NOTE: When adding or removing public APIs from this file, remember to update
#     cuda_pathfinder/docs/source/api.rst
# to keep the documentation in sync.

from cuda.pathfinder._binaries.find_nvidia_binary_utility import (
    find_nvidia_binary_utility as find_nvidia_binary_utility,
)
from cuda.pathfinder._binaries.supported_nvidia_binaries import SUPPORTED_BINARIES as _SUPPORTED_BINARIES
from cuda.pathfinder._dynamic_libs.load_dl_common import (
    DynamicLibNotAvailableError as DynamicLibNotAvailableError,
)
from cuda.pathfinder._dynamic_libs.load_dl_common import DynamicLibNotFoundError as DynamicLibNotFoundError
from cuda.pathfinder._dynamic_libs.load_dl_common import (
    DynamicLibUnknownError as DynamicLibUnknownError,
)
from cuda.pathfinder._dynamic_libs.load_dl_common import LoadedDL as LoadedDL
from cuda.pathfinder._dynamic_libs.load_nvidia_dynamic_lib import load_nvidia_dynamic_lib as load_nvidia_dynamic_lib
from cuda.pathfinder._dynamic_libs.supported_nvidia_libs import SUPPORTED_LIBNAMES as _SUPPORTED_NVIDIA_LIBNAMES
from cuda.pathfinder._headers.find_nvidia_headers import LocatedHeaderDir as LocatedHeaderDir
from cuda.pathfinder._headers.find_nvidia_headers import find_nvidia_header_directory as find_nvidia_header_directory
from cuda.pathfinder._headers.find_nvidia_headers import (
    locate_nvidia_header_directory as locate_nvidia_header_directory,
)
from cuda.pathfinder._headers.supported_nvidia_headers import SUPPORTED_HEADERS_CTK as _SUPPORTED_HEADERS_CTK
from cuda.pathfinder._static_libs.find_bitcode_lib import (
    SUPPORTED_BITCODE_LIBS as _SUPPORTED_BITCODE_LIBS,
)
from cuda.pathfinder._static_libs.find_bitcode_lib import (
    BitcodeLibNotFoundError as BitcodeLibNotFoundError,
)
from cuda.pathfinder._static_libs.find_bitcode_lib import (
    LocatedBitcodeLib as LocatedBitcodeLib,
)
from cuda.pathfinder._static_libs.find_bitcode_lib import (
    find_bitcode_lib as find_bitcode_lib,
)
from cuda.pathfinder._static_libs.find_bitcode_lib import (
    locate_bitcode_lib as locate_bitcode_lib,
)
from cuda.pathfinder._static_libs.find_static_lib import (
    SUPPORTED_STATIC_LIBS as _SUPPORTED_STATIC_LIBS,
)
from cuda.pathfinder._static_libs.find_static_lib import (
    LocatedStaticLib as LocatedStaticLib,
)
from cuda.pathfinder._static_libs.find_static_lib import (
    StaticLibNotFoundError as StaticLibNotFoundError,
)
from cuda.pathfinder._static_libs.find_static_lib import (
    find_static_lib as find_static_lib,
)
from cuda.pathfinder._static_libs.find_static_lib import (
    locate_static_lib as locate_static_lib,
)
from cuda.pathfinder._utils.env_vars import get_cuda_path_or_home as get_cuda_path_or_home
from cuda.pathfinder._utils.windows_arch import UnsupportedArchError as UnsupportedArchError

from cuda.pathfinder._version import __version__  # isort: skip

# Indirections to help Sphinx find the docstrings.
#: Mapping from short CUDA Toolkit (CTK) library names to their canonical
#: header basenames (used to validate a discovered include directory).
#: Example: ``"cublas" → "cublas.h"``. The key set is platform-aware
#: (e.g., ``"cufile"`` may be Linux-only).
SUPPORTED_HEADERS_CTK = _SUPPORTED_HEADERS_CTK

#: Tuple of supported CUDA binary utility names that can be located
#: via ``find_nvidia_binary_utility()``. Platform-aware (e.g., some
#: utilities may be available only on Linux or Windows).
#: Example utilities: ``"nvdisasm"``, ``"cuobjdump"``, ``"nvcc"``.
SUPPORTED_BINARY_UTILITIES = _SUPPORTED_BINARIES

#: Tuple of CUDA Toolkit dynamic library names supported by
#: :func:`load_nvidia_dynamic_lib` for the current operating system and
#: interpreter architecture.
SUPPORTED_NVIDIA_LIBNAMES = _SUPPORTED_NVIDIA_LIBNAMES

#: Tuple of supported bitcode library names that can be resolved
#: via ``locate_bitcode_lib()`` and ``find_bitcode_lib()``.
#: Example value: ``"device"``.
SUPPORTED_BITCODE_LIBS = _SUPPORTED_BITCODE_LIBS

#: Tuple of supported static library names that can be resolved
#: via ``locate_static_lib()`` and ``find_static_lib()``.
#: Example value: ``"cudadevrt"``.
SUPPORTED_STATIC_LIBS = _SUPPORTED_STATIC_LIBS

# Backward compatibility: _find_nvidia_header_directory was added in release 1.2.2.
# It will be removed in release 1.2.4.
_find_nvidia_header_directory = find_nvidia_header_directory
<|endoftext|># SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import os
from collections.abc import Iterator
from typing import Any

from cuda.pathfinder._utils.windows_arch import windows_machine_arch

_REGISTRY_ROOT = r"SOFTWARE\NVIDIA Corporation\Installed Products\Nsight"

_NSYS_TARGET_DIR_BY_ARCH = {
    "x64": "target-windows-x64",
    "arm64": "target-windows-armv8",
}

_NCU_TARGET_DIR_BY_ARCH = {
    "x64": os.path.join("target", "windows-desktop-win7-x64"),
    "arm64": os.path.join("target", "windows-desktop-win10-t23x-a64"),
}


def _installed_product_root(product: str) -> str | None:
    """Return the active Nsight product installation recorded by its MSI."""
    # ``winreg`` attributes are absent from the type stubs on non-Windows hosts.
    winreg: Any = importlib.import_module("winreg")

    access = winreg.KEY_READ | winreg.KEY_WOW64_64KEY
    product_key_path = rf"{_REGISTRY_ROOT}\{product}"
    try:
        product_context = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, product_key_path, 0, access)
    except FileNotFoundError:
        return None

    try:
        with product_context as product_key:
            current_version, _ = winreg.QueryValueEx(product_key, "CurrentVersion")
            if not isinstance(current_version, str) or not current_version.strip():
                raise RuntimeError(
                    f"Invalid CurrentVersion value {current_version!r} in "
                    f"Nsight {product!r} registry registration at {product_key_path!r}"
                )
            with winreg.OpenKey(product_key, current_version, 0, access) as version_key:
                install_root, _ = winreg.QueryValueEx(version_key, None)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Incomplete Nsight {product!r} registry registration at {product_key_path!r}") from exc

    if not isinstance(install_root, str) or not install_root.strip():
        raise RuntimeError(
            f"Invalid installation directory {install_root!r} in Nsight {product!r} "
            f"registry registration at {product_key_path!r} version {current_version!r}"
        )
    return install_root


def nsys_candidate_paths() -> Iterator[str]:
    install_root = _installed_product_root("Systems")
    if install_root is None:
        return

    target_dir = _NSYS_TARGET_DIR_BY_ARCH[windows_machine_arch()]
    yield os.path.join(install_root, target_dir, "nsys.exe")


def ncu_candidate_paths() -> Iterator[str]:
    install_root = _installed_product_root("Compute")
    if install_root is None:
        return

    yield os.path.join(install_root, "ncu.bat")

    target_dir = _NCU_TARGET_DIR_BY_ARCH[windows_machine_arch()]
    yield os.path.join(install_root, target_dir, "ncu.exe")
<|endoftext|># SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import os
from collections.abc import Iterable

from cuda.pathfinder._binaries import supported_nvidia_binaries, windows_nsight
from cuda.pathfinder._utils.ctk_root_canary import CTK_ROOT_CANARY_ANCHOR_LIBNAMES
from cuda.pathfinder._utils.env_vars import get_cuda_path_or_home
from cuda.pathfinder._utils.find_sub_dirs import find_sub_dirs_all_sitepackages
from cuda.pathfinder._utils.platform_aware import IS_WINDOWS


class UnsupportedBinaryError(Exception):
    def __init__(self, utility: str) -> None:
        super().__init__(utility)
        self.utility = utility

    def __str__(self) -> str:
        supported_utilities = ", ".join(supported_nvidia_binaries.SUPPORTED_BINARIES)
        return f"Binary '{self.utility}' is not supported. Supported utilities are: {supported_utilities}"


def _normalize_utility_name(utility_name: str) -> str:
    """Normalize utility name by adding .exe on Windows if needed."""
    if IS_WINDOWS and not utility_name.lower().endswith((".exe", ".bat", ".cmd")):
        return f"{utility_name}.exe"
    return utility_name


def _is_executable_candidate(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    if IS_WINDOWS:
        return True
    return os.access(path, os.X_OK)


def _ctk_bin_subdirs(root: str) -> list[str]:
    if IS_WINDOWS:
        return [
            os.path.join(root, "bin", "x64"),
            os.path.join(root, "bin", "x86_64"),
            os.path.join(root, "bin"),
        ]
    return [os.path.join(root, "bin")]


def _resolve_candidate_paths(candidates: Iterable[str]) -> str | None:
    """Return the first executable candidate, preserving candidate order."""
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if _is_executable_candidate(candidate):
            return os.path.abspath(candidate)
    return None


def _find_windows_compute_sanitizer(ctk_root: str) -> str | None:
    return _resolve_candidate_paths(
        (
            os.path.join(ctk_root, "bin", "compute-sanitizer.bat"),
            os.path.join(ctk_root, "compute-sanitizer", "compute-sanitizer.exe"),
        )
    )


def _resolve_ctk_root_via_canary() -> str | None:
    from cuda.pathfinder._dynamic_libs.load_nvidia_dynamic_lib import resolve_ctk_root_via_canary

    ctk_root: str | None = resolve_ctk_root_via_canary(CTK_ROOT_CANARY_ANCHOR_LIBNAMES[0])
    return ctk_root


def _resolve_in_trusted_dirs(normalized_name: str, dirs: list[str]) -> str | None:
    """Resolve ``normalized_name`` against ``dirs`` in order."""
    seen: set[str] = set()
    for directory in dirs:
        if directory in seen:
            continue
        assert directory
        seen.add(directory)
        candidate = os.path.join(directory, normalized_name)
        if _is_executable_candidate(candidate):
            # Return an absolute path, as the docstring promises (a relative
            # search dir would otherwise leak a relative result).
            return os.path.abspath(candidate)
    return None


def _resolve_names_in_trusted_dirs(candidate_names: tuple[str, ...], dirs: list[str]) -> str | None:
    """Resolve ordered candidate names within each trusted directory."""
    seen: set[str] = set()
    for directory in dirs:
        if directory in seen:
            continue
        assert directory
        seen.add(directory)
        found = _resolve_candidate_paths(os.path.join(directory, name) for name in candidate_names)
        if found is not None:
            return found
    return None


@functools.cache
def find_nvidia_binary_utility(utility_name: str) -> str | None:
    """Locate a CUDA binary utility executable.

    Args:
        utility_name (str): The name of the binary utility to find
            (e.g., ``"nvdisasm"``, ``"cuobjdump"``). On Windows, the ``.exe``
            extension will be automatically appended if not present. The function
            also recognizes ``.bat`` and ``.cmd`` files on Windows.

    Returns:
        str or None: Absolute path to the discovered executable, or ``None``
        if the utility cannot be found. The returned path is normalized
        (absolute and with resolved separators).

    Raises:
        UnsupportedBinaryError: If ``utility_name`` is not in the supported set
            (see ``SUPPORTED_BINARY_UTILITIES``).
        RuntimeError: If a native Windows architecture needed for an
            architecture-specific utility layout cannot be determined, or an
            installed Nsight product has incomplete or invalid registry data.

    Windows on ARM (WoA) Note:
        Binary utilities execute in separate processes and do not need to match
        the Python process architecture. When choosing among architecture-specific
        Windows layouts, this API deliberately targets the native machine
        architecture rather than the Python interpreter architecture. For
        example, standalone ``nsys`` and ``ncu`` discovery under x64 Python on an
        Arm64 machine selects the Arm64 target. This differs from
        ``load_nvidia_dynamic_lib`` and ``find_static_lib``, which target the
        Python interpreter architecture.

    Search order:
        1. **NVIDIA Python wheels**

           - Scan installed distributions (``site-packages``) for binary layouts
             shipped in NVIDIA wheels (e.g., ``cuda-nvcc``).

        2. **Conda environments**

           - Check Conda-style installation prefixes via ``CONDA_PREFIX``
             environment variable, which use platform-specific bin directory
             layouts (``Library/bin`` on Windows, ``bin`` on Linux).

        3. **Library-specific standalone installations**

           - Search the installation paths for the CUDA Toolkit, Nsight Systems,
             and Nsight Compute.

           3.1. **Nsight installations**: On Windows, locate Nsight Systems and
                Nsight Compute from their installer registry entries. Select
                architecture-specific binaries using the native machine
                architecture, independent of Python. Lookup of the standalone
                ``nsys`` and ``ncu`` CLIs is terminal; a miss does not fall
                through to CUDA Toolkit locations.

           3.2. **CUDA Toolkit installation**: Use ``CUDA_PATH`` or ``CUDA_HOME``
                (in that order), searching ``bin/x64``, ``bin/x86_64``, and
                ``bin`` subdirectories on Windows, or just ``bin`` on Linux.

        4. **CTK-root canary fallback**

           - For utilities that reach this step after the earlier searches miss,
             resolve the ``cudart`` library through the OS dynamic loader, derive
             the CUDA Toolkit root from it, and search that root's bin layout.

    Note:
        Results are cached using ``@functools.cache`` for performance. The cache
        persists for the lifetime of the process.

        On Windows, executables are identified by their file extensions
        (``.exe``, ``.bat``, ``.cmd``). On Unix-like systems, executables
        are identified by the ``X_OK`` (execute) permission bit.

        Lookup is restricted to the trusted directories and the canary-derived
        CTK root listed above.

    Example:
        >>> from cuda.pathfinder import find_nvidia_binary_utility
        >>> nvdisasm = find_nvidia_binary_utility("nvdisasm")
        >>> if nvdisasm:
        ...     print(f"Found nvdisasm at: {nvdisasm}")
    """
    if utility_name not in supported_nvidia_binaries.SUPPORTED_BINARIES:
        raise UnsupportedBinaryError(utility_name)

    # 1. Search in site-packages (NVIDIA wheels)
    candidate_dirs = supported_nvidia_binaries.SITE_PACKAGES_BINDIRS.get(utility_name, ())
    dirs = []

    for sub_dir in candidate_dirs:
        dirs.extend(find_sub_dirs_all_sitepackages(sub_dir.split(os.sep)))

    # 2. Search in Conda environment
    if (conda_prefix := os.environ.get("CONDA_PREFIX")) is not None:
        if IS_WINDOWS:
            dirs.append(os.path.join(conda_prefix, "Library", "bin"))
        else:
            dirs.append(os.path.join(conda_prefix, "bin"))

    normalized_name = _normalize_utility_name(utility_name)
    if IS_WINDOWS and utility_name in ("compute-sanitizer", "ncu"):
        candidate_names = (f"{utility_name}.bat", normalized_name)
        found = _resolve_names_in_trusted_dirs(candidate_names, dirs)
    else:
        found = _resolve_in_trusted_dirs(normalized_name, dirs)
    if found is not None:
        return found

    # 3. Search library-specific standalone installations.
    # 3.1. Standalone Nsight CLI lookup is terminal; CTK does not contain nsys/ncu.
    if IS_WINDOWS and utility_name == "nsys":
        return _resolve_candidate_paths(windows_nsight.nsys_candidate_paths())
    if IS_WINDOWS and utility_name == "ncu":
        return _resolve_candidate_paths(windows_nsight.ncu_candidate_paths())

    # 3.2. Search in CUDA Toolkit (CUDA_PATH/CUDA_HOME).
    if (cuda_path := get_cuda_path_or_home()) is not None:
        if IS_WINDOWS and utility_name == "compute-sanitizer":
            found = _find_windows_compute_sanitizer(cuda_path)
        else:
            found = _resolve_in_trusted_dirs(normalized_name, _ctk_bin_subdirs(cuda_path))
        if found is not None:
            return found

    # 4. CTK-root canary fallback.
    ctk_root = _resolve_ctk_root_via_canary()
    if ctk_root is not None:
        if IS_WINDOWS and utility_name == "compute-sanitizer":
            return _find_windows_compute_sanitizer(ctk_root)
        return _resolve_in_trusted_dirs(normalized_name, _ctk_bin_subdirs(ctk_root))
    return None
<|endoftext|># SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import os
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, TypedDict

from cuda.pathfinder._utils.env_vars import get_cuda_path_or_home
from cuda.pathfinder._utils.find_sub_dirs import find_sub_dirs_all_sitepackages
from cuda.pathfinder._utils.platform_aware import IS_WINDOWS
from cuda.pathfinder._utils.windows_arch import windows_python_arch


class StaticLibNotFoundError(RuntimeError):
    """Raised when a static library cannot be found."""


@dataclass(frozen=True)
class LocatedStaticLib:
    """Information about a located static library."""

    name: str
    abs_path: str
    filename: str
    found_via: str


class _StaticLibInfo(TypedDict):
    """Static-library metadata with ordered alternatives searched first-to-last."""

    filename: str
    ctk_rel_paths: tuple[str, ...]
    conda_rel_paths: tuple[str, ...]
    site_packages_dirs: tuple[str, ...]


def _cudadevrt_info() -> _StaticLibInfo:
    if not IS_WINDOWS:
        return {
            "filename": "libcudadevrt.a",
            "ctk_rel_paths": ("lib64", "lib"),
            "conda_rel_paths": ("lib",),
            "site_packages_dirs": ("nvidia/cu13/lib", "nvidia/cuda_runtime/lib"),
        }

    arch_dir = windows_python_arch()
    component_wheel_dirs = ("nvidia/cuda_runtime/lib/x64",) if arch_dir == "x64" else ()
    conda_fallback_dirs = ("lib",) if arch_dir == "x64" else ()
    return {
        "filename": "cudadevrt.lib",
        "ctk_rel_paths": (str(Path("lib", arch_dir)),),
        "conda_rel_paths": (str(Path("lib", arch_dir)), *conda_fallback_dirs),
        "site_packages_dirs": (f"nvidia/cu13/lib/{arch_dir}", *component_wheel_dirs),
    }


_SUPPORTED_STATIC_LIBS_INFO: dict[str, _StaticLibInfo] = {
    "cudadevrt": _cudadevrt_info(),
}

SUPPORTED_STATIC_LIBS: tuple[str, ...] = tuple(sorted(_SUPPORTED_STATIC_LIBS_INFO.keys()))


def _no_such_file_in_dir(directory: Path, filename: str, error_messages: list[str], attachments: list[str]) -> None:
    error_messages.append(f"No such file: {directory / filename}")
    if directory.is_dir():
        attachments.append(f'  listdir("{directory}"):')
        for node in sorted(node_path.name for node_path in directory.iterdir()):
            attachments.append(f"    {node}")
    else:
        attachments.append(f'  Directory does not exist: "{directory}"')


class _FindStaticLib:
    def __init__(self, name: str) -> None:
        if name not in _SUPPORTED_STATIC_LIBS_INFO:
            raise ValueError(f"Unknown static library: '{name}'. Supported: {', '.join(SUPPORTED_STATIC_LIBS)}")
        self.name: str = name
        self.config: _StaticLibInfo = _SUPPORTED_STATIC_LIBS_INFO[name]
        self.filename: str = self.config["filename"]
        self.ctk_rel_paths: tuple[str, ...] = self.config["ctk_rel_paths"]
        self.conda_rel_paths: tuple[str, ...] = self.config["conda_rel_paths"]
        self.site_packages_dirs: tuple[str, ...] = self.config["site_packages_dirs"]
        self.error_messages: list[str] = []
        self.attachments: list[str] = []

    def try_site_packages(self) -> Path | None:
        for rel_dir in self.site_packages_dirs:
            sub_dir = tuple(rel_dir.split("/"))
            for abs_dir in find_sub_dirs_all_sitepackages(sub_dir):
                file_path = Path(abs_dir, self.filename)
                if file_path.is_file():
                    return file_path
        return None

    def try_with_conda_prefix(self) -> Path | None:
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if not conda_prefix:
            return None

        anchor = Path(conda_prefix, "Library") if IS_WINDOWS else Path(conda_prefix)
        for rel_path in self.conda_rel_paths:
            file_path = anchor / rel_path / self.filename
            if file_path.is_file():
                return file_path
        return None

    def try_with_cuda_home(self) -> Path | None:
        cuda_home = get_cuda_path_or_home()
        if cuda_home is None:
            self.error_messages.append("CUDA_HOME/CUDA_PATH not set")
            return None

        anchor = Path(cuda_home)
        for rel_path in self.ctk_rel_paths:
            file_path = anchor / rel_path / self.filename
            if file_path.is_file():
                return file_path

        _no_such_file_in_dir(
            anchor / self.ctk_rel_paths[0],
            self.filename,
            self.error_messages,
            self.attachments,
        )
        return None

    def raise_not_found_error(self) -> NoReturn:
        err = ", ".join(self.error_messages) if self.error_messages else "No search paths available"
        att = "\n".join(self.attachments) if self.attachments else ""
        raise StaticLibNotFoundError(f'Failure finding "{self.filename}": {err}\n{att}')


def locate_static_lib(name: str) -> LocatedStaticLib:
    """Locate a static library by name.

    Raises:
        ValueError: If ``name`` is not a supported static library.
        StaticLibNotFoundError: If the static library cannot be found.
    """
    finder = _FindStaticLib(name)

    abs_path = finder.try_site_packages()
    if abs_path is not None:
        return LocatedStaticLib(
            name=name,
            abs_path=str(abs_path),
            filename=finder.filename,
            found_via="site-packages",
        )

    abs_path = finder.try_with_conda_prefix()
    if abs_path is not None:
        return LocatedStaticLib(
            name=name,
            abs_path=str(abs_path),
            filename=finder.filename,
            found_via="conda",
        )

    abs_path = finder.try_with_cuda_home()
    if abs_path is not None:
        return LocatedStaticLib(
            name=name,
            abs_path=str(abs_path),
            filename=finder.filename,
            found_via="CUDA_PATH",
        )

    finder.raise_not_found_error()


@functools.cache
def find_static_lib(name: str) -> str:
    """Find the absolute path to a static library.

    Raises:
        ValueError: If ``name`` is not a supported static library.
        StaticLibNotFoundError: If the static library cannot be found.

    Windows on ARM (WoA) Note:
        On Windows, this API aims to return the path to a static library whose
        architecture matches the Python interpreter architecture. For example,
        x64 Python running on an Arm64 machine targets the x64 library, while
        native Arm64 Python targets the Arm64 library. This differs from
        ``find_nvidia_binary_utility``, which targets the native machine
        architecture when selecting architecture-specific executables.
    """
    return locate_static_lib(name).abs_path
<|endoftext|># SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import os
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, TypedDict

from cuda.pathfinder._utils.env_vars import get_cuda_path_or_home
from cuda.pathfinder._utils.find_sub_dirs import find_sub_dirs_all_sitepackages
from cuda.pathfinder._utils.platform_aware import IS_WINDOWS


class BitcodeLibNotFoundError(RuntimeError):
    """Raised when a bitcode library cannot be found."""


@dataclass(frozen=True)
class LocatedBitcodeLib:
    """Information about a located bitcode library."""

    name: str
    abs_path: str
    filename: str
    found_via: str


class _BitcodeLibInfo(TypedDict):
    """Bitcode-library metadata with ordered alternatives searched first-to-last."""

    filename: str
    rel_path: str
    site_packages_dirs: tuple[str, ...]
    available_on_windows: bool


_SUPPORTED_BITCODE_LIBS_INFO: dict[str, _BitcodeLibInfo] = {
    "device": {
        "filename": "libdevice.10.bc",
        "rel_path": "nvvm/libdevice",
        "site_packages_dirs": (
            "nvidia/cu13/nvvm/libdevice",
            "nvidia/cuda_nvcc/nvvm/libdevice",
        ),
        "available_on_windows": True,
    },
    "nccl_device": {
        "filename": "libnccl_device.bc",
        "rel_path": "lib",
        "site_packages_dirs": ("nvidia/nccl/lib",),
        "available_on_windows": False,
    },
    "nvshmem_device": {
        "filename": "libnvshmem_device.bc",
        "rel_path": "lib",
        "site_packages_dirs": ("nvidia/nvshmem/lib",),
        "available_on_windows": False,
    },
}

# Public API: just the supported library names
SUPPORTED_BITCODE_LIBS: tuple[str, ...] = tuple(
    sorted(
        name for name, info in _SUPPORTED_BITCODE_LIBS_INFO.items() if not IS_WINDOWS or info["available_on_windows"]
    )
)


def _no_such_file_in_dir(directory: Path, filename: str, error_messages: list[str], attachments: list[str]) -> None:
    error_messages.append(f"No such file: {directory / filename}")
    if directory.is_dir():
        attachments.append(f'  listdir("{directory}"):')
        for node in sorted(node_path.name for node_path in directory.iterdir()):
            attachments.append(f"    {node}")
    else:
        attachments.append(f'  Directory does not exist: "{directory}"')


class _FindBitcodeLib:
    def __init__(self, name: str) -> None:
        if name not in _SUPPORTED_BITCODE_LIBS_INFO:  # Updated reference
            raise ValueError(f"Unknown bitcode library: '{name}'. Supported: {', '.join(SUPPORTED_BITCODE_LIBS)}")
        self.name: str = name
        self.config: _BitcodeLibInfo = _SUPPORTED_BITCODE_LIBS_INFO[name]  # Updated reference
        self.filename: str = self.config["filename"]
        self.rel_path: str = self.config["rel_path"]
        self.site_packages_dirs: tuple[str, ...] = self.config["site_packages_dirs"]
        self.error_messages: list[str] = []
        self.attachments: list[str] = []

    def try_site_packages(self) -> Path | None:
        for rel_dir in self.site_packages_dirs:
            sub_dir = tuple(rel_dir.split("/"))
            for abs_dir in find_sub_dirs_all_sitepackages(sub_dir):
                file_path = Path(abs_dir, self.filename)
                if file_path.is_file():
                    return file_path
        return None

    def try_with_conda_prefix(self) -> Path | None:
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if not conda_prefix:
            return None

        anchor = Path(conda_prefix, "Library") if IS_WINDOWS else Path(conda_prefix)
        file_path = anchor / self.rel_path / self.filename
        if file_path.is_file():
            return file_path
        return None

    def try_with_cuda_home(self) -> Path | None:
        cuda_home = get_cuda_path_or_home()
        if cuda_home is None:
            self.error_messages.append("CUDA_HOME/CUDA_PATH not set")
            return None

        anchor = Path(cuda_home)
        file_path = anchor / self.rel_path / self.filename
        if file_path.is_file():
            return file_path

        _no_such_file_in_dir(
            anchor / self.rel_path,
            self.filename,
            self.error_messages,
            self.attachments,
        )
        return None

    def raise_not_found_error(self) -> NoReturn:
        err = ", ".join(self.error_messages) if self.error_messages else "No search paths available"
        att = "\n".join(self.attachments) if self.attachments else ""
        raise BitcodeLibNotFoundError(f'Failure finding "{self.filename}": {err}\n{att}')


def locate_bitcode_lib(name: str) -> LocatedBitcodeLib:
    """Locate a bitcode library by name.

    Raises:
        ValueError: If ``name`` is not a supported bitcode library.
        BitcodeLibNotFoundError: If the bitcode library cannot be found.
    """
    finder = _FindBitcodeLib(name)

    abs_path = finder.try_site_packages()
    if abs_path is not None:
        return LocatedBitcodeLib(
            name=name,
            abs_path=str(abs_path),
            filename=finder.filename,
            found_via="site-packages",
        )

    abs_path = finder.try_with_conda_prefix()
    if abs_path is not None:
        return LocatedBitcodeLib(
            name=name,
            abs_path=str(abs_path),
            filename=finder.filename,
            found_via="conda",
        )

    abs_path = finder.try_with_cuda_home()
    if abs_path is not None:
        return LocatedBitcodeLib(
            name=name,
            abs_path=str(abs_path),
            filename=finder.filename,
            found_via="CUDA_PATH",
        )

    finder.raise_not_found_error()


@functools.cache
def find_bitcode_lib(name: str) -> str:
    """Find the absolute path to a bitcode library.

    Raises:
        ValueError: If ``name`` is not a supported bitcode library.
        BitcodeLibNotFoundError: If the bitcode library cannot be found.
    """
    return locate_bitcode_lib(name).abs_path
<|endoftext|># SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import platform
import sysconfig

WINDOWS_PE_MACHINE_BY_ARCH = {
    "x64": 0x8664,
    "arm64": 0xAA64,
}

_WINDOWS_ARCH_BY_PE_MACHINE = {machine: arch for arch, machine in WINDOWS_PE_MACHINE_BY_ARCH.items()}


class UnsupportedArchError(RuntimeError):
    """Raised when Python reports an unsupported Windows architecture."""

    def __init__(self, platform_tag: str) -> None:
        self.platform_tag = platform_tag
        super().__init__(
            f"Unsupported Windows Python platform tag: {platform_tag!r}; expected 'win-amd64' or 'win-arm64'"
        )


def windows_python_arch() -> str:
    """Return the current Windows Python interpreter architecture."""
    raw_platform_tag = sysconfig.get_platform()
    platform_tag = raw_platform_tag.lower().replace("_", "-")

    if platform_tag == "win-arm64":
        return "arm64"

    if platform_tag == "win-amd64":
        return "x64"

    raise UnsupportedArchError(raw_platform_tag)


def _windows_machine_arch_from_platform() -> str:
    """Return the Windows architecture reported by Python's platform module."""
    raw_machine = platform.machine()
    machine = raw_machine.lower().replace("_", "-")

    if machine in ("amd64", "x86-64"):
        return "x64"

    if machine in ("arm64", "aarch64"):
        return "arm64"

    raise RuntimeError(f"Unsupported Windows machine architecture: {raw_machine!r}")


def _windows_native_machine() -> int | None:
    """Return the native Windows PE machine type, or None on older Windows."""
    import ctypes
    from ctypes import wintypes

    try:
        # These ctypes attributes are absent from the type stubs on non-Windows hosts.
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined, unused-ignore]
    except OSError as exc:
        raise RuntimeError("Failed to load kernel32 while detecting the native Windows architecture") from exc

    get_current_process = kernel32.GetCurrentProcess
    try:
        is_wow64_process2 = kernel32.IsWow64Process2
    except AttributeError:
        return None

    get_current_process.argtypes = ()
    get_current_process.restype = wintypes.HANDLE
    is_wow64_process2.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.USHORT),
        ctypes.POINTER(wintypes.USHORT),
    )
    is_wow64_process2.restype = wintypes.BOOL

    process_machine = wintypes.USHORT()
    native_machine = wintypes.USHORT()
    if not is_wow64_process2(
        get_current_process(),
        ctypes.byref(process_machine),
        ctypes.byref(native_machine),
    ):
        error_code = ctypes.get_last_error()  # type: ignore[attr-defined, unused-ignore]
        error = ctypes.WinError(error_code)  # type: ignore[attr-defined, unused-ignore]
        raise RuntimeError(
            f"IsWow64Process2 failed while detecting the native Windows architecture "
            f"(Windows error {error_code}): {error}"
        ) from error
    return native_machine.value


def windows_machine_arch() -> str:
    """Return the native Windows machine architecture, ignoring process emulation."""
    native_machine = _windows_native_machine()
    if native_machine is None:
        # IsWow64Process2 predates x64-on-Arm emulation, so this fallback is only
        # needed on older Windows versions where platform.machine() is sufficient.
        return _windows_machine_arch_from_platform()

    try:
        return _WINDOWS_ARCH_BY_PE_MACHINE[native_machine]
    except KeyError:
        raise RuntimeError(f"Unsupported native Windows PE machine type: 0x{native_machine:04x}") from None


def windows_pe_matches_arch(path: str, target_arch: str) -> bool:
    """Return whether a Windows Portable Executable (PE) targets the requested architecture.

    PE is the file format used for Windows executables and DLLs. This reads the
    PE/COFF header's machine field to distinguish x64 images from Arm64 images.
    """
    expected_machine = WINDOWS_PE_MACHINE_BY_ARCH.get(target_arch)
    if expected_machine is None:
        raise ValueError(f"Unsupported Windows target architecture: {target_arch!r}")

    try:
        with open(path, "rb") as stream:
            if stream.read(2) != b"MZ":
                return False
            stream.seek(0x3C)
            pe_offset_bytes = stream.read(4)
            if len(pe_offset_bytes) != 4:
                return False
            stream.seek(int.from_bytes(pe_offset_bytes, "little"))
            if stream.read(4) != b"PE\0\0":
                return False
            machine_bytes = stream.read(2)
            if len(machine_bytes) != 2:
                return False
    except OSError:
        return False

    return int.from_bytes(machine_bytes, "little") == expected_machine
<|endoftext|># SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import collections
import functools
import importlib.metadata


@functools.cache
def find_all_dll_files_via_metadata() -> dict[str, tuple[str, ...]]:
    results: collections.defaultdict[str, list[str]] = collections.defaultdict(list)

    # sort dists for deterministic output

    for dist in sorted(
        importlib.metadata.distributions(),
        # `get` exists before 3.12, even though the hints only exist for Python >=3.12
        key=lambda d: (d.metadata.get("Name", ""), d.version),  # type: ignore[attr-defined]
    ):
        files = dist.files
        if not files:
            continue
        for relpath in sorted(files, key=lambda p: str(p)):  # deterministic
            relname = relpath.name.lower()
            if not relname.endswith(".dll"):
                continue
            abs_path = str(dist.locate_file(relpath))
            results[relname].append(abs_path)

    # plain dicts; sort inner list for stability
    return {k: tuple(sorted(v)) for k, v in results.items()}
<|endoftext|># SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes
import functools
import sys
from collections.abc import Callable
from dataclasses import dataclass

from cuda.pathfinder._dynamic_libs.load_nvidia_dynamic_lib import (
    load_nvidia_dynamic_lib as _load_nvidia_dynamic_lib,
)


class QueryDriverCudaVersionError(RuntimeError):
    """Raised when ``query_driver_cuda_version()`` cannot determine the CUDA driver version."""


@dataclass(frozen=True, slots=True)
class DriverCudaVersion:
    """
    CUDA-facing driver version reported by ``cuDriverGetVersion()``.

    The name ``DriverCudaVersion`` is intentionally specific: this dataclass
    models the version shown as ``CUDA Version`` in ``nvidia-smi``, not the
    graphics driver release shown as ``Driver Version``. More specifically,
    it reflects the CUDA user-mode driver (UMD) interface version reported by
    ``cuDriverGetVersion()``, not the kernel-mode driver (KMD) package
    version.

    Example ``nvidia-smi`` output::

        +---------------------------------------------------------------------+
        | NVIDIA-SMI 595.58.03  Driver Version: 595.58.03  CUDA Version: 13.2 |
        +---------------------------------------------------------------------+

    For the example above, ``DriverCudaVersion(encoded=13020, major=13,
    minor=2)`` corresponds to ``CUDA Version: 13.2``. It does not correspond
    to ``Driver Version: 595.58.03``.
    """

    encoded: int
    major: int
    minor: int


@functools.cache
def query_driver_cuda_version() -> DriverCudaVersion:
    """Return the CUDA driver version parsed into its major/minor components."""
    try:
        encoded = _query_driver_cuda_version_int()
        return DriverCudaVersion(
            encoded=encoded,
            major=encoded // 1000,
            minor=(encoded % 1000) // 10,
        )
    except Exception as exc:
        raise QueryDriverCudaVersionError("Failed to query the CUDA driver version.") from exc


if sys.platform == "win32":
    _DRIVER_LIB_LOADER: Callable[[str], ctypes.CDLL] = ctypes.WinDLL
else:
    _DRIVER_LIB_LOADER = ctypes.CDLL


def _query_driver_cuda_version_int() -> int:
    """Return the encoded CUDA driver version from ``cuDriverGetVersion()``."""
    loaded_cuda = _load_nvidia_dynamic_lib("cuda")
    driver_lib = _DRIVER_LIB_LOADER(loaded_cuda.abs_path)
    cu_driver_get_version = driver_lib.cuDriverGetVersion
    cu_driver_get_version.argtypes = [ctypes.POINTER(ctypes.c_int)]
    cu_driver_get_version.restype = ctypes.c_int
    version = ctypes.c_int()
    status = cu_driver_get_version(ctypes.byref(version))
    if status != 0:
        raise RuntimeError(f"Failed to query CUDA driver version via cuDriverGetVersion() (status={status}).")
    return version.value
<|endoftext|># SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import os
import site
import sys
from collections.abc import Sequence


def find_sub_dirs_no_cache(parent_dirs: Sequence[str], sub_dirs: Sequence[str]) -> list[str]:
    results = []
    for base in parent_dirs:
        stack = [(base, 0)]  # (current_path, index into sub_dirs)
        while stack:
            current_path, idx = stack.pop()
            if idx == len(sub_dirs):
                if os.path.isdir(current_path):
                    results.append(current_path)
                continue

            sub = sub_dirs[idx]
            if sub == "*":
                try:
                    entries = sorted(os.listdir(current_path))
                except OSError:
                    continue
                for entry in entries:
                    entry_path = os.path.join(current_path, entry)
                    if os.path.isdir(entry_path):
                        stack.append((entry_path, idx + 1))
            else:
                next_path = os.path.join(current_path, sub)
                if os.path.isdir(next_path):
                    stack.append((next_path, idx + 1))
    return results


@functools.cache
def find_sub_dirs_cached(parent_dirs: Sequence[str], sub_dirs: Sequence[str]) -> list[str]:
    return find_sub_dirs_no_cache(parent_dirs, sub_dirs)


def find_sub_dirs(parent_dirs: Sequence[str], sub_dirs: Sequence[str]) -> list[str]:
    return find_sub_dirs_cached(tuple(parent_dirs), tuple(sub_dirs))


def find_sub_dirs_sys_path(sub_dirs: Sequence[str]) -> list[str]:
    return find_sub_dirs(sys.path, sub_dirs)


def find_sub_dirs_all_sitepackages(sub_dirs: Sequence[str]) -> list[str]:
    parent_dirs = list(site.getsitepackages())
    if site.ENABLE_USER_SITE:
        user_site = site.getusersitepackages()
        if user_site:
            # Determine insertion index based on whether we're in a virtual environment (fixes #1716):
            # - In venv (PEP 405): venv site-packages should come first, then user-site-packages,
            #   then system site-packages. Insert at index 1 (after venv, before system).
            # - Not in venv (PEP 370): user-site-packages should come before system site-packages.
            #   Insert at index 0.
            # Detect venv by checking if sys.prefix differs from sys.base_prefix
            insert_idx = 1 if sys.prefix != sys.base_prefix else 0
            parent_dirs.insert(insert_idx, user_site)
    return find_sub_dirs(parent_dirs, sub_dirs)
<|endoftext|># SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic path ordering helpers."""

from __future__ import annotations

import os
import re

_DIGIT_RUN = re.compile(r"(\d+)")


def numeric_aware_path_sort_key(path: str) -> tuple[tuple[int, int, str], ...]:
    """Return a key that compares embedded digit runs by numeric value."""
    key: list[tuple[int, int, str]] = []
    for part in _DIGIT_RUN.split(os.path.normcase(path)):
        if part.isdigit():
            normalized = part.lstrip("0") or "0"
            key.append((1, len(normalized), normalized))
        else:
            key.append((0, 0, part))
    return tuple(key)
<|endoftext|># SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import collections
import functools
import importlib.metadata
import re

_SO_RE = re.compile(r"\.so(?:$|\.)")  # matches libfoo.so or libfoo.so.1.2.3


def split_so_version_suffix(so_filename: str) -> tuple[str, str]:
    idx = so_filename.rfind(".so")
    assert idx > 0, so_filename
    idx += 3
    return (so_filename[:idx], so_filename[idx:])


@functools.cache
def find_all_so_files_via_metadata() -> dict[str, dict[str, tuple[str, ...]]]:
    results: collections.defaultdict[str, collections.defaultdict[str, list[str]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )

    # sort dists for deterministic output
    for dist in sorted(
        importlib.metadata.distributions(),
        # `get` exists before 3.12, even though the hints only exist for Python >=3.12
        key=lambda d: (d.metadata.get("Name", ""), d.version),  # type: ignore[attr-defined]
    ):
        files = dist.files
        if not files:
            continue
        for relpath in sorted(files, key=lambda p: str(p)):  # deterministic
            relname = relpath.name
            if not _SO_RE.search(relname):
                continue
            so_basename, so_version_suffix = split_so_version_suffix(relname)
            abs_path = str(dist.locate_file(relpath))
            results[so_basename][so_version_suffix].append(abs_path)

    # plain dicts; sort inner lists for stability
    return {k: {kk: tuple(sorted(vv)) for kk, vv in v.items()} for k, v in results.items()}
<|endoftext|># SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Centralized CUDA environment variable handling.

This module defines the canonical search order for CUDA Toolkit environment variables
used throughout cuda-python packages (cuda.pathfinder, cuda.core, cuda.bindings).

Search Order Priority:
    1. CUDA_PATH (higher priority)
    2. CUDA_HOME (lower priority)

If both are set and differ, CUDA_PATH takes precedence and a warning is issued.

Important Note on Caching:
    The result of get_cuda_path_or_home() is cached for the process lifetime. The first
    call determines the CUDA Toolkit path, and all subsequent calls return the cached
    value, even if environment variables change later. This ensures consistent behavior
    throughout the application lifecycle.
"""

import functools
import os
import warnings

_CUDA_PATH_ENV_VARS_ORDERED = ("CUDA_PATH", "CUDA_HOME")


def _paths_differ(a: str, b: str) -> bool:
    """
    Return True if paths are observably different.

    Strategy:
    1) Compare os.path.normcase(os.path.normpath(...)) for quick, robust textual equality.
       - Handles trailing slashes and case-insensitivity on Windows.
    2) If still different AND both exist, use os.path.samefile to resolve symlinks/junctions.
    3) Otherwise (nonexistent paths or samefile unavailable), treat as different.
    """
    norm_a = os.path.normcase(os.path.normpath(a))
    norm_b = os.path.normcase(os.path.normpath(b))
    if norm_a == norm_b:
        return False

    try:
        if os.path.exists(a) and os.path.exists(b):
            # samefile raises on non-existent paths; only call when both exist.
            return not os.path.samefile(a, b)
    except OSError:
        # Fall through to "different" if samefile isn't applicable/available.
        pass

    # If normalized strings differ and we couldn't prove they're the same entry, treat as different.
    return True


@functools.cache
def get_cuda_path_or_home() -> str | None:
    """Get CUDA Toolkit path from environment variables.

    Returns the value of CUDA_PATH or CUDA_HOME. If both are set and differ,
    CUDA_PATH takes precedence and a warning is issued.

    The result is cached for the process lifetime. The first call determines the CUDA
    Toolkit path, and subsequent calls return the cached value.

    Returns:
        Path to CUDA Toolkit, or None if neither variable is set or all are empty.

    Warnings:
        UserWarning: If multiple CUDA environment variables are set but point to
            different locations (only on the first call).

    """
    # Collect non-empty environment variables in priority order.
    # Empty strings are treated as undefined — no valid CUDA path is empty.
    set_vars = {}
    for var in _CUDA_PATH_ENV_VARS_ORDERED:
        val = os.environ.get(var)
        if val:
            set_vars[var] = val

    if not set_vars:
        return None

    # If multiple variables are set, check if they differ and warn
    if len(set_vars) > 1:
        values = list(set_vars.items())
        values_differ = False
        for i in range(len(values) - 1):
            if _paths_differ(values[i][1], values[i + 1][1]):
                values_differ = True
                break

        if values_differ:
            var_list = "\n".join(f"  {var}={val}" for var, val in set_vars.items())
            warnings.warn(
                f"Multiple CUDA environment variables are set but differ:\n"
                f"{var_list}\n"
                f"Using {_CUDA_PATH_ENV_VARS_ORDERED[0]} (highest priority).",
                UserWarning,
                stacklevel=2,
            )

    # Return the first (highest priority) set variable
    return next(iter(set_vars.values()))
<|endoftext|># SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys

IS_WINDOWS = sys.platform == "win32"
_WINDOWS_PYTHON_ARCH: str | None

if IS_WINDOWS:
    from cuda.pathfinder._utils.windows_arch import windows_python_arch

    _WINDOWS_PYTHON_ARCH = windows_python_arch()
else:
    _WINDOWS_PYTHON_ARCH = None

# These describe the Python process ABI, not the Windows host architecture.
IS_WINDOWS_X64 = _WINDOWS_PYTHON_ARCH == "x64"
IS_WINDOWS_ARM64 = _WINDOWS_PYTHON_ARCH == "arm64"


def quote_for_shell(s: str) -> str:
    if IS_WINDOWS:
        # This is a relatively heavy import; keep pathfinder lean if possible.
        from subprocess import list2cmdline

        return list2cmdline([s])
    else:
        import shlex

        return shlex.quote(s)
<|endoftext|># SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-library descriptor and registry.

The canonical authored data lives in :mod:`descriptor_catalog`. This module
provides a name-keyed registry consumed by the runtime search/load path.
"""

from __future__ import annotations

from typing import TypeAlias

from cuda.pathfinder._dynamic_libs.descriptor_catalog import (
    DESCRIPTOR_CATALOG,
    DescriptorSpec,
)

# Keep the historical type name for downstream imports.
LibDescriptor: TypeAlias = DescriptorSpec


#: Canonical registry of all known libraries.
LIB_DESCRIPTORS: dict[str, LibDescriptor] = {desc.name: desc for desc in DESCRIPTOR_CATALOG}
<|endoftext|># SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

MODE_CANARY: Literal["canary"] = "canary"
MODE_LOAD: Literal["load"] = "load"
VALID_MODES: tuple[Literal["canary"], Literal["load"]] = (MODE_CANARY, MODE_LOAD)

STATUS_OK: Literal["ok"] = "ok"
STATUS_NOT_FOUND: Literal["not-found"] = "not-found"

DYNAMIC_LIB_SUBPROCESS_MODULE = "cuda.pathfinder._dynamic_libs.dynamic_lib_subprocess"
DYNAMIC_LIB_SUBPROCESS_CWD = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class DynamicLibSubprocessPayload:
    status: Literal["ok", "not-found"]
    abs_path: str | None


def format_dynamic_lib_subprocess_payload(
    status: Literal["ok", "not-found"],
    abs_path: str | None,
    *,
    error: dict[str, str] | None = None,
) -> str:
    payload: dict[str, object] = {"status": status, "abs_path": abs_path}
    if error is not None:
        payload["error"] = error
    return json.dumps(payload)


def build_dynamic_lib_subprocess_command(mode: str, libname: str) -> list[str]:
    return [sys.executable, "-m", DYNAMIC_LIB_SUBPROCESS_MODULE, mode, libname]


def parse_dynamic_lib_subprocess_payload(
    stdout: str,
    *,
    libname: str,
    error_label: str,
) -> DynamicLibSubprocessPayload:
    # Use the final non-empty line in case earlier output lines are emitted.
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"{error_label} produced no stdout payload for {libname!r}")
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError:
        raise RuntimeError(f"{error_label} emitted invalid JSON payload for {libname!r}: {lines[-1]!r}") from None
    if not isinstance(payload, dict):
        raise RuntimeError(f"{error_label} emitted unexpected payload for {libname!r}: {payload!r}")
    status = payload.get("status")
    abs_path = payload.get("abs_path")
    if status == STATUS_OK:
        if not isinstance(abs_path, str):
            raise RuntimeError(f"{error_label} emitted unexpected payload for {libname!r}: {payload!r}")
        return DynamicLibSubprocessPayload(status=STATUS_OK, abs_path=abs_path)
    if status == STATUS_NOT_FOUND:
        if abs_path is not None:
            raise RuntimeError(f"{error_label} emitted unexpected payload for {libname!r}: {payload!r}")
        return DynamicLibSubprocessPayload(status=STATUS_NOT_FOUND, abs_path=None)
    raise RuntimeError(f"{error_label} emitted unexpected payload for {libname!r}: {payload!r}")
<|endoftext|># SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Composable search steps for locating NVIDIA libraries.

Each find step is a callable with signature::

    (SearchContext) -> FindResult | None

Find steps locate a library file on disk without loading it.  The
orchestrator in :mod:`load_nvidia_dynamic_lib` handles loading, the
already-loaded check, and dependency resolution.

Step sequences are defined per search strategy so that adding a new
step or strategy only requires adding a function and a tuple entry.

This module is intentionally platform-agnostic: it does not branch on the
current operating system. Platform differences are routed through the
:data:`~cuda.pathfinder._dynamic_libs.search_platform.PLATFORM` instance.
"""

import glob
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import NoReturn, cast

from cuda.pathfinder._dynamic_libs.lib_descriptor import LibDescriptor
from cuda.pathfinder._dynamic_libs.load_dl_common import DynamicLibNotFoundError
from cuda.pathfinder._dynamic_libs.search_platform import PLATFORM, SearchPlatform
from cuda.pathfinder._utils.env_vars import get_cuda_path_or_home
from cuda.pathfinder._utils.path_sort import numeric_aware_path_sort_key

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class FindResult:
    """A library file located on disk (not yet loaded)."""

    abs_path: str
    found_via: str


@dataclass
class SearchContext:
    """Mutable state accumulated during the search cascade."""

    desc: LibDescriptor
    platform: SearchPlatform = PLATFORM
    error_messages: list[str] = field(default_factory=list)
    attachments: list[str] = field(default_factory=list)

    @property
    def libname(self) -> str:
        return self.desc.name  # type: ignore[no-any-return]  # mypy can't resolve new sibling module

    @property
    def lib_searched_for(self) -> str:
        return cast(str, self.platform.lib_searched_for(self.desc))

    def raise_not_found(self) -> NoReturn:
        err = ", ".join(self.error_messages)
        att = "\n".join(self.attachments)
        raise DynamicLibNotFoundError(f'Failure finding "{self.lib_searched_for}": {err}\n{att}')


#: Type alias for a find step callable.
FindStep = Callable[[SearchContext], FindResult | None]


def _iter_lib_dirs(root: str, rel_dirs: tuple[str, ...]) -> Iterator[str]:
    """Yield existing library directories under *root* in descriptor order."""
    for rel_path in rel_dirs:
        for dirname in sorted(glob.glob(os.path.join(root, rel_path))):
            if os.path.isdir(dirname):
                yield os.path.normpath(dirname)


def _iter_lib_dirs_using_anchor(
    desc: LibDescriptor,
    platform: SearchPlatform,
    anchor_point: str,
) -> Iterator[str]:
    """Yield existing library directories under *anchor_point* in descriptor order."""
    yield from _iter_lib_dirs(anchor_point, platform.anchor_rel_dirs(desc))


def _find_lib_dir_using_anchor(desc: LibDescriptor, platform: SearchPlatform, anchor_point: str) -> str | None:
    """Find the first library directory under *anchor_point*."""
    return next(_iter_lib_dirs_using_anchor(desc, platform, anchor_point), None)


def _find_using_lib_dir(ctx: SearchContext, lib_dir: str | None) -> str | None:
    """Find a library file in a resolved lib directory."""
    if lib_dir is None:
        return None
    return cast(
        str | None,
        ctx.platform.find_in_lib_dir(
            lib_dir,
            ctx.desc,
            ctx.error_messages,
            ctx.attachments,
        ),
    )


def _find_under_root(
    ctx: SearchContext,
    root: str,
    rel_dirs: tuple[str, ...],
    found_via: str,
) -> FindResult | None:
    """Resolve *rel_dirs* under *root*, then find the requested library."""
    for lib_dir in _iter_lib_dirs(root, rel_dirs):
        abs_path = _find_using_lib_dir(ctx, lib_dir)
        if abs_path is not None:
            return FindResult(abs_path, found_via)
    return None


def _find_under_anchor_root(ctx: SearchContext, root: str, found_via: str) -> FindResult | None:
    """Resolve the descriptor's general anchors under *root*."""
    return _find_under_root(ctx, root, ctx.platform.anchor_rel_dirs(ctx.desc), found_via)


def _derive_ctk_root_linux(resolved_lib_path: str) -> str | None:
    """Derive CTK root from Linux canary path.

    Supports:
    - ``$CTK_ROOT/lib64/libfoo.so.*``
    - ``$CTK_ROOT/lib/libfoo.so.*``
    - ``$CTK_ROOT/targets/<triple>/lib64/libfoo.so.*``
    - ``$CTK_ROOT/targets/<triple>/lib/libfoo.so.*``
    """
    lib_dir = os.path.dirname(resolved_lib_path)
    basename = os.path.basename(lib_dir)
    if basename in ("lib64", "lib"):
        parent = os.path.dirname(lib_dir)
        grandparent = os.path.dirname(parent)
        if os.path.basename(grandparent) == "targets":
            return os.path.dirname(grandparent)
        return parent
    return None


def _derive_ctk_root_windows(resolved_lib_path: str) -> str | None:
    """Derive CTK root from Windows canary path.

    Supports:
    - ``$CTK_ROOT/bin/x64/foo.dll`` (CTK 13 style)
    - ``$CTK_ROOT/bin/arm64/foo.dll`` (Windows on Arm CTK 13 style)
    - ``$CTK_ROOT/bin/foo.dll`` (CTK 12 style)
    """
    import ntpath

    lib_dir = ntpath.dirname(resolved_lib_path)
    basename = ntpath.basename(lib_dir).lower()
    if basename in ("x64", "arm64"):
        parent = ntpath.dirname(lib_dir)
        if ntpath.basename(parent).lower() == "bin":
            return ntpath.dirname(parent)
    elif basename == "bin":
        return ntpath.dirname(lib_dir)
    return None


def derive_ctk_root(resolved_lib_path: str) -> str | None:
    """Derive CTK root from a resolved canary library path."""
    ctk_root = _derive_ctk_root_linux(resolved_lib_path)
    if ctk_root is not None:
        return ctk_root
    return _derive_ctk_root_windows(resolved_lib_path)


def find_via_ctk_root(ctx: SearchContext, ctk_root: str) -> FindResult | None:
    """Find a library under a previously derived CTK root."""
    return _find_under_anchor_root(ctx, ctk_root, "system-ctk-root")


# ---------------------------------------------------------------------------
# Find steps
# ---------------------------------------------------------------------------


def find_in_site_packages(ctx: SearchContext) -> FindResult | None:
    """Search pip wheel install locations."""
    rel_dirs = ctx.platform.site_packages_rel_dirs(ctx.desc)
    if not rel_dirs:
        return None
    abs_path = ctx.platform.find_in_site_packages(
        rel_dirs,
        ctx.desc,
        ctx.error_messages,
        ctx.attachments,
    )
    if abs_path is not None:
        return FindResult(abs_path, "site-packages")
    return None


def find_in_conda(ctx: SearchContext) -> FindResult | None:
    """Search ``$CONDA_PREFIX``."""
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return None
    anchor = ctx.platform.conda_anchor_point(conda_prefix)
    return _find_under_anchor_root(ctx, anchor, "conda")


def find_in_install_root_env_vars(ctx: SearchContext) -> FindResult | None:
    """Search installation roots named by descriptor-specific environment variables."""
    rel_dirs = ctx.platform.install_root_env_rel_dirs(ctx.desc)
    for env_var in ctx.platform.install_root_env_vars(ctx.desc):
        root = os.environ.get(env_var)
        if not root:
            continue
        result = _find_under_root(ctx, root, rel_dirs, env_var)
        if result is not None:
            return result
    return None


def find_in_cuda_path(ctx: SearchContext) -> FindResult | None:
    """Search ``$CUDA_PATH`` / ``$CUDA_HOME``.

    On Windows, this is the normal fallback for system-installed CTK DLLs when
    they are not already discoverable via the native ``LoadLibraryExW(..., 0)``
    path used by :func:`cuda.pathfinder._dynamic_libs.load_dl_windows.load_with_system_search`.
    Python 3.8+ does not include ``PATH`` in that native DLL search.

    The returned ``found_via`` is always ``"CUDA_PATH"`` regardless of which
    environment variable actually provided the value.
    """
    cuda_home = get_cuda_path_or_home()
    if cuda_home is None:
        return None
    return _find_under_anchor_root(ctx, cuda_home, "CUDA_PATH")


def find_in_program_files_roots(ctx: SearchContext) -> FindResult | None:
    """Search descriptor-configured installation roots under Program Files."""
    for root_glob in ctx.platform.program_files_root_globs(ctx.desc):
        for root in sorted(glob.glob(root_glob), key=numeric_aware_path_sort_key, reverse=True):
            if not os.path.isdir(root):
                continue
            result = _find_under_anchor_root(ctx, os.path.normpath(root), "ProgramFiles")
            if result is not None:
                return result
    return None


# ---------------------------------------------------------------------------
# Step sequences per strategy
# ---------------------------------------------------------------------------

#: Find steps that run before the already-loaded check and system search.
EARLY_FIND_STEPS: tuple[FindStep, ...] = (find_in_site_packages, find_in_conda)

#: Find steps that run after system search fails.
LATE_FIND_STEPS: tuple[FindStep, ...] = (
    find_in_install_root_env_vars,
    find_in_cuda_path,
    find_in_program_files_roots,
)


# ---------------------------------------------------------------------------
# Cascade runner
# ---------------------------------------------------------------------------


def run_find_steps(ctx: SearchContext, steps: tuple[FindStep, ...]) -> FindResult | None:
    """Run find steps in order, returning the first hit."""
    for step in steps:
        result = step(ctx)
        if result is not None:
            return result
    return None
<|endoftext|># SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Platform loader seam for OS-specific dynamic linking.

This module provides a small interface that hides the Linux vs Windows
implementation details of:

- already-loaded checks
- system-search loading
- absolute-path loading

The orchestration logic in :mod:`load_nvidia_dynamic_lib` should not need to
branch on platform; it calls through the loader instance exported here.
"""

from __future__ import annotations

import sys
from typing import Protocol

from cuda.pathfinder._dynamic_libs.lib_descriptor import LibDescriptor
from cuda.pathfinder._dynamic_libs.load_dl_common import LoadedDL


class PlatformLoader(Protocol):
    def check_if_already_loaded_from_elsewhere(self, desc: LibDescriptor) -> LoadedDL | None: ...

    def load_with_system_search(self, desc: LibDescriptor) -> LoadedDL | None: ...

    def load_with_abs_path(self, desc: LibDescriptor, found_path: str, found_via: str | None = None) -> LoadedDL: ...


if sys.platform == "win32":
    from cuda.pathfinder._dynamic_libs import load_dl_windows as _impl
else:
    from cuda.pathfinder._dynamic_libs import load_dl_linux as _impl

# The platform modules already expose functions matching the PlatformLoader
# protocol. Wrap in a simple namespace so callers use LOADER.method() syntax.
LOADER: PlatformLoader = _impl
<|endoftext|># SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import os
import sys
from typing import TYPE_CHECKING, cast

from cuda.pathfinder._dynamic_libs.load_dl_common import LoadedDL

if TYPE_CHECKING:
    from cuda.pathfinder._dynamic_libs.lib_descriptor import LibDescriptor

if sys.platform == "linux":
    CDLL_MODE = os.RTLD_NOW | os.RTLD_GLOBAL
else:
    CDLL_MODE = 0


def _load_libdl() -> ctypes.CDLL:
    # In normal glibc-based Linux environments, find_library("dl") should return
    # something like "libdl.so.2". In minimal or stripped-down environments
    # (no ldconfig/gcc, incomplete linker cache), this can return None even
    # though libdl is present. In that case, we fall back to the stable SONAME.
    name = ctypes.util.find_library("dl") or "libdl.so.2"
    try:
        return ctypes.CDLL(name)
    except OSError as e:
        raise RuntimeError(f"Could not load {name!r} (required for dlinfo/dlerror on Linux)") from e


LIBDL = _load_libdl()

# dlinfo
LIBDL.dlinfo.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
LIBDL.dlinfo.restype = ctypes.c_int

# dlerror (thread-local error string; cleared after read)
LIBDL.dlerror.argtypes = []
LIBDL.dlerror.restype = ctypes.c_char_p

# First appeared in 2004-era glibc. Universally correct on Linux for all practical purposes.
RTLD_DI_LINKMAP = 2
RTLD_DI_ORIGIN = 6


class _LinkMapLNameView(ctypes.Structure):
    """
    Prefix-only view of glibc's `struct link_map` used **solely** to read `l_name`.

    Background:
      - `dlinfo(handle, RTLD_DI_LINKMAP, ...)` returns a `struct link_map*`.
      - The first few members of `struct link_map` (including `l_name`) have been
        stable on glibc for decades and are documented as debugger-visible.
      - We only need the offset/layout of `l_name`, not the full struct.

    Safety constraints:
      - This is a **partial** definition (prefix). It must only be used via a pointer
        returned by `dlinfo(...)`.
      - Do **not** instantiate it or pass it **by value** to any C function.
      - Do **not** access any members beyond those declared here.
      - Do **not** rely on `ctypes.sizeof(LinkMapPrefix)` for allocation.

    Rationale:
      - Defining only the leading fields avoids depending on internal/unstable
        tail members while keeping code more readable than raw pointer arithmetic.
    """

    _fields_ = (
        ("l_addr", ctypes.c_void_p),  # ElfW(Addr)
        ("l_name", ctypes.c_char_p),  # char*
    )


# Defensive assertions, mainly  to document the invariants we depend on
assert _LinkMapLNameView.l_addr.offset == 0
assert _LinkMapLNameView.l_name.offset == ctypes.sizeof(ctypes.c_void_p)


def _dl_last_error() -> str | None:
    msg_bytes = cast(bytes | None, LIBDL.dlerror())
    if not msg_bytes:
        return None  # no pending error
    # Never raises; undecodable bytes are mapped to U+DC80..U+DCFF
    return msg_bytes.decode("utf-8", "surrogateescape")


def l_name_for_dynamic_library(libname: str, handle: ctypes.CDLL) -> str:
    lm_view = ctypes.POINTER(_LinkMapLNameView)()
    rc = LIBDL.dlinfo(ctypes.c_void_p(handle._handle), RTLD_DI_LINKMAP, ctypes.byref(lm_view))
    if rc != 0:
        err = _dl_last_error()
        raise OSError(f"dlinfo failed for {libname=!r} (rc={rc})" + (f": {err}" if err else ""))
    if not lm_view:  # NULL link_map**
        raise OSError(f"dlinfo returned NULL link_map pointer for {libname=!r}")

    l_name_bytes = lm_view.contents.l_name
    if not l_name_bytes:
        raise OSError(f"dlinfo returned empty link_map->l_name for {libname=!r}")

    path = os.fsdecode(l_name_bytes)
    if not path:
        raise OSError(f"dlinfo returned empty l_name string for {libname=!r}")

    return path


def l_origin_for_dynamic_library(libname: str, handle: ctypes.CDLL) -> str:
    l_origin_buf = ctypes.create_string_buffer(4096)
    rc = LIBDL.dlinfo(ctypes.c_void_p(handle._handle), RTLD_DI_ORIGIN, l_origin_buf)
    if rc != 0:
        err = _dl_last_error()
        raise OSError(f"dlinfo failed for {libname=!r} (rc={rc})" + (f": {err}" if err else ""))

    path = os.fsdecode(l_origin_buf.value)
    if not path:
        raise OSError(f"dlinfo returned empty l_origin string for {libname=!r}")

    return path


def abs_path_for_dynamic_library(libname: str, handle: ctypes.CDLL) -> str:
    l_name = l_name_for_dynamic_library(libname, handle)
    l_origin = l_origin_for_dynamic_library(libname, handle)
    return os.path.join(l_origin, os.path.basename(l_name))


if sys.platform == "linux":

    def check_if_already_loaded_from_elsewhere(desc: LibDescriptor) -> LoadedDL | None:
        for soname in desc.linux_sonames:
            try:
                handle = ctypes.CDLL(soname, mode=os.RTLD_NOLOAD)
            except OSError:
                continue
            else:
                return LoadedDL(
                    abs_path_for_dynamic_library(desc.name, handle),
                    True,
                    handle._handle,
                    "was-already-loaded-from-elsewhere",
                )
        return None

    def _load_lib(desc: LibDescriptor, filename: str) -> ctypes.CDLL:
        cdll_mode = CDLL_MODE
        if desc.requires_rtld_deepbind:
            cdll_mode |= os.RTLD_DEEPBIND
        return ctypes.CDLL(filename, cdll_mode)
else:

    def check_if_already_loaded_from_elsewhere(_desc: LibDescriptor) -> LoadedDL | None:
        raise RuntimeError(f"check_if_already_loaded_from_elsewhere() is not supported on platform {sys.platform!r}")

    def _load_lib(_desc: LibDescriptor, _filename: str) -> ctypes.CDLL:
        raise RuntimeError(f"_load_lib() is not supported on platform {sys.platform!r}")


def load_with_system_search(desc: LibDescriptor) -> LoadedDL | None:
    """Try to load a library using the native Linux dynamic-loader search path.

    Args:
        desc: Descriptor for the library to load

    Returns:
        A LoadedDL object if successful, None if the library cannot be loaded

    """
    for soname in desc.linux_sonames:
        try:
            handle = _load_lib(desc, soname)
        except OSError:
            pass
        else:
            abs_path = abs_path_for_dynamic_library(desc.name, handle)
            assert abs_path
            return LoadedDL(abs_path, False, handle._handle, "system-search")
    return None


def _work_around_known_bugs(libname: str, found_path: str) -> None:
    if libname == "nvrtc":
        # Work around bug/oversight in
        #   nvidia_cuda_nvrtc-13.0.48-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl
        # Issue: libnvrtc.so.13 RUNPATH is not set.
        # This workaround is highly specific
        #   - for simplicity.
        #   - to not mask bugs in future nvidia-cuda-nvrtc releases.
        #   - because a more general workaround is complicated.
        dirname, basename = os.path.split(found_path)
        if basename == "libnvrtc.so.13":
            dep_basename = "libnvrtc-builtins.so.13.0"
            dep_path = os.path.join(dirname, dep_basename)
            if os.path.isfile(dep_path):
                # In case of failure, defer to primary load, which is almost certain to fail, too.
                with contextlib.suppress(OSError):
                    ctypes.CDLL(dep_path, CDLL_MODE)


def load_with_abs_path(desc: LibDescriptor, found_path: str, found_via: str | None = None) -> LoadedDL:
    """Load a dynamic library from the given path.

    Args:
        desc: Descriptor for the library to load.
        found_path: The absolute path to the library file.
        found_via: Label indicating how the path was discovered.

    Returns:
        A LoadedDL object representing the loaded library.

    Raises:
        RuntimeError: If the library cannot be loaded.
    """
    _work_around_known_bugs(desc.name, found_path)
    try:
        handle = _load_lib(desc, found_path)
    except OSError as e:
        raise RuntimeError(f"Failed to dlopen {found_path}: {e}") from e
    return LoadedDL(found_path, False, handle._handle, found_via)
<|endoftext|># SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical authored descriptor catalog for dynamic libraries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal

from cuda.pathfinder._utils.ctk_root_canary import CTK_ROOT_CANARY_ANCHOR_LIBNAMES

PackagedWith = Literal["ctk", "other", "driver"]
WindowsArch = Literal["x64", "arm64"]


@dataclass(frozen=True, slots=True)
class WindowsSearchDirs:
    """Windows search locations, highest priority first, grouped by architecture."""

    x64: tuple[str, ...] = ()
    arm64: tuple[str, ...] = ()

    @classmethod
    def x64_only(cls, *paths: str) -> WindowsSearchDirs:
        return cls(x64=paths)

    @classmethod
    def arm64_only(cls, *paths: str) -> WindowsSearchDirs:
        return cls(arm64=paths)

    def for_arch(self, target_arch: str) -> tuple[str, ...]:
        if target_arch == "x64":
            return self.x64
        if target_arch == "arm64":
            return self.arm64
        raise ValueError(f"Unsupported Windows target architecture: {target_arch!r}")


# Windows CTK before 13.4 was x64-only and used the common bin directory.
# Native ARM64 support starts with the architecture-qualified 13.4 layout.
DEFAULT_WINDOWS_CTK_ANCHOR_DIRS = WindowsSearchDirs(
    x64=("bin/x64", "bin"),
    arm64=("bin/arm64",),
)


def _ctk_windows_wheel_dirs(cuda13_bin_dir: str, cuda12_dir: str) -> WindowsSearchDirs:
    """Search CUDA 13 first, with the x64-only CUDA 12 wheel as an x64 fallback."""
    cuda13_bin_path = PurePosixPath(cuda13_bin_dir)
    return WindowsSearchDirs(
        x64=((cuda13_bin_path / "x86_64").as_posix(), cuda12_dir),
        arm64=((cuda13_bin_path / "arm64").as_posix(),),
    )


@dataclass(frozen=True, slots=True)
class DescriptorSpec:
    """Dynamic-library metadata; ordered search candidates are tried first-to-last."""

    name: str
    packaged_with: PackagedWith
    # Library filenames are authored in search/load preference order, most preferred first.
    linux_sonames: tuple[str, ...] = ()
    windows_dlls: tuple[str, ...] = ()
    supported_windows_arch: tuple[WindowsArch, ...] = ()
    site_packages_linux: tuple[str, ...] = ()
    site_packages_windows: WindowsSearchDirs = WindowsSearchDirs()
    dependencies: tuple[str, ...] = ()
    optional_dependencies: tuple[str, ...] = ()
    anchor_rel_dirs_linux: tuple[str, ...] = ("lib64", "lib")
    anchor_rel_dirs_windows: WindowsSearchDirs = DEFAULT_WINDOWS_CTK_ANCHOR_DIRS
    install_root_env_vars_linux: tuple[str, ...] = ()
    install_root_env_rel_dirs_linux: tuple[str, ...] = ()
    install_root_env_vars_windows: tuple[str, ...] = ()
    install_root_env_rel_dirs_windows: WindowsSearchDirs = WindowsSearchDirs()
    program_files_root_globs_windows: WindowsSearchDirs = WindowsSearchDirs()
    ctk_root_canary_anchor_libnames: tuple[str, ...] = ()
    requires_add_dll_directory: bool = False
    requires_rtld_deepbind: bool = False
    requires_windows_binary_arch_check: bool = False


DESCRIPTOR_CATALOG: tuple[DescriptorSpec, ...] = (
    # -----------------------------------------------------------------------
    # CTK (CUDA Toolkit) libraries
    # -----------------------------------------------------------------------
    DescriptorSpec(
        name="cudart",
        packaged_with="ctk",
        linux_sonames=("libcudart.so.13", "libcudart.so.12"),
        windows_dlls=("cudart64_13.dll", "cudart64_12.dll"),
        supported_windows_arch=("x64", "arm64"),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/cuda_runtime/lib"),
        site_packages_windows=_ctk_windows_wheel_dirs("nvidia/cu13/bin", "nvidia/cuda_runtime/bin"),
    ),
    DescriptorSpec(
        name="nvfatbin",
        packaged_with="ctk",
        linux_sonames=("libnvfatbin.so.13", "libnvfatbin.so.12"),
        windows_dlls=("nvfatbin_130_0.dll", "nvfatbin_120_0.dll"),
        supported_windows_arch=("x64", "arm64"),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/nvfatbin/lib"),
        site_packages_windows=_ctk_windows_wheel_dirs("nvidia/cu13/bin", "nvidia/nvfatbin/bin"),
    ),
    DescriptorSpec(
        name="nvJitLink",
        packaged_with="ctk",
        linux_sonames=("libnvJitLink.so.13", "libnvJitLink.so.12"),
        windows_dlls=("nvJitLink_130_0.dll", "nvJitLink_120_0.dll"),
        supported_windows_arch=("x64", "arm64"),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/nvjitlink/lib"),
        site_packages_windows=_ctk_windows_wheel_dirs("nvidia/cu13/bin", "nvidia/nvjitlink/bin"),
    ),
    DescriptorSpec(
        name="nvrtc",
        packaged_with="ctk",
        linux_sonames=("libnvrtc.so.13", "libnvrtc.so.12"),
        windows_dlls=("nvrtc64_130_0.dll", "nvrtc64_120_0.dll"),
        supported_windows_arch=("x64", "arm64"),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/cuda_nvrtc/lib"),
        site_packages_windows=_ctk_windows_wheel_dirs("nvidia/cu13/bin", "nvidia/cuda_nvrtc/bin"),
        requires_add_dll_directory=True,
    ),
    DescriptorSpec(
        name="nvvm",
        packaged_with="ctk",
        linux_sonames=("libnvvm.so.4", "libnvvm.so"),
        windows_dlls=("nvvm70.dll", "nvvm64_40_0.dll", "nvvm64.dll"),
        supported_windows_arch=("x64", "arm64"),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/cuda_nvcc/nvvm/lib64"),
        site_packages_windows=_ctk_windows_wheel_dirs("nvidia/cu13/bin", "nvidia/cuda_nvcc/nvvm/bin"),
        anchor_rel_dirs_linux=("nvvm/lib64",),
        # CTK 13.4 installs the ARM64 DLL directly in nvvm/bin, while x64
        # uses nvvm/bin/x64. Older x64 toolkits also used nvvm/bin, so the
        # binary in the unqualified directory must be checked at runtime.
        anchor_rel_dirs_windows=WindowsSearchDirs(
            x64=("nvvm/bin/x64", "nvvm/bin"),
            arm64=("nvvm/bin",),
        ),
        ctk_root_canary_anchor_libnames=CTK_ROOT_CANARY_ANCHOR_LIBNAMES,
        # requires_windows_binary_arch_check disambiguates pre-13.4 x64 DLLs
        # from 13.4+ Arm64 DLLs in nvvm/bin; see
        # _utils/windows_arch.py for the validation.
        requires_windows_binary_arch_check=True,
    ),
    DescriptorSpec(
        name="cublas",
        packaged_with="ctk",
        linux_sonames=("libcublas.so.13", "libcublas.so.12"),
        windows_dlls=("cublas64_13.dll", "cublas64_12.dll"),
        supported_windows_arch=("x64", "arm64"),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/cublas/lib"),
        site_packages_windows=_ctk_windows_wheel_dirs("nvidia/cu13/bin", "nvidia/cublas/bin"),
        dependencies=("cublasLt",),
    ),
    DescriptorSpec(
        name="cublasLt",
        packaged_with="ctk",
        linux_sonames=("libcublasLt.so.13", "libcublasLt.so.12"),
        windows_dlls=("cublasLt64_13.dll", "cublasLt64_12.dll"),
        supported_windows_arch=("x64", "arm64"),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/cublas/lib"),
        site_packages_windows=_ctk_windows_wheel_dirs("nvidia/cu13/bin", "nvidia/cublas/bin"),
    ),
    DescriptorSpec(
        name="cufft",
        packaged_with="ctk",
        linux_sonames=("libcufft.so.12", "libcufft.so.11"),
        windows_dlls=("cufft64_12.dll", "cufft64_11.dll"),
        supported_windows_arch=("x64", "arm64"),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/cufft/lib"),
        site_packages_windows=_ctk_windows_wheel_dirs("nvidia/cu13/bin", "nvidia/cufft/bin"),
        requires_add_dll_directory=True,
    ),
    DescriptorSpec(
        name="cufftw",
        packaged_with="ctk",
        linux_sonames=("libcufftw.so.12", "libcufftw.so.11"),
        windows_dlls=("cufftw64_12.dll", "cufftw64_11.dll"),
        supported_windows_arch=("x64", "arm64"),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/cufft/lib"),
        site_packages_windows=_ctk_windows_wheel_dirs("nvidia/cu13/bin", "nvidia/cufft/bin"),
        dependencies=("cufft",),
    ),
    DescriptorSpec(
        name="curand",
        packaged_with="ctk",
        linux_sonames=("libcurand.so.11", "libcurand.so.10"),
        windows_dlls=("curand64_11.dll", "curand64_10.dll"),
        supported_windows_arch=("x64", "arm64"),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/curand/lib"),
        site_packages_windows=_ctk_windows_wheel_dirs("nvidia/cu13/bin", "nvidia/curand/bin"),
    ),
    DescriptorSpec(
        name="cusolver",
        packaged_with="ctk",
        linux_sonames=("libcusolver.so.12", "libcusolver.so.11"),
        windows_dlls=("cusolver64_12.dll", "cusolver64_11.dll"),
        supported_windows_arch=("x64", "arm64"),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/cusolver/lib"),
        site_packages_windows=_ctk_windows_wheel_dirs("nvidia/cu13/bin", "nvidia/cusolver/bin"),
        dependencies=("nvJitLink", "cusparse", "cublasLt", "cublas"),
    ),
    DescriptorSpec(
        name="cusolverMg",
        packaged_with="ctk",
        linux_sonames=("libcusolverMg.so.12", "libcusolverMg.so.11"),
        windows_dlls=("cusolverMg64_12.dll", "cusolverMg64_11.dll"),
        supported_windows_arch=("x64", "arm64"),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/cusolver/lib"),
        site_packages_windows=_ctk_windows_wheel_dirs("nvidia/cu13/bin", "nvidia/cusolver/bin"),
        dependencies=("nvJitLink", "cublasLt", "cublas"),
    ),
    DescriptorSpec(
        name="cusparse",
        packaged_with="ctk",
        linux_sonames=("libcusparse.so.12",),
        windows_dlls=("cusparse64_12.dll",),
        supported_windows_arch=("x64", "arm64"),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/cusparse/lib"),
        site_packages_windows=_ctk_windows_wheel_dirs("nvidia/cu13/bin", "nvidia/cusparse/bin"),
        dependencies=("nvJitLink",),
    ),
    DescriptorSpec(
        name="nppc",
        packaged_with="ctk",
        linux_sonames=("libnppc.so.13", "libnppc.so.12"),
        windows_dlls=("nppc64_13.dll", "nppc64_12.dll"),
        supported_windows_arch=("x64", "arm64"),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/npp/lib"),
        site_packages_windows=_ctk_windows_wheel_dirs("nvidia/cu13/bin", "nvidia/npp/bin"),
    ),
    DescriptorSpec(
        name="nppial",
        packaged_with="ctk",
        linux_sonames=("libnppial.so.13", "libnppial.so.12"),
        windows_dlls=("nppial64_13.dll", "nppial64_12.dll"),
        supported_windows_arch=("x64", "arm64"),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/npp/lib"),
        site_packages_windows=_ctk_windows_wheel_dirs("nvidia/cu13/bin", "nvidia/npp/bin"),
        dependencies=("nppc",),
    ),
    DescriptorSpec(
        name="nppicc",
        packaged_with="ctk",
        linux_sonames=("libnppicc.so.13", "libnppicc.so.12"),
        windows_dlls=("nppicc64_13.dll", "nppicc64_12.dll"),
        supported_windows_arch=("x64", "arm64"),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/npp/lib"),
        site_packages_windows=_ctk_windows_wheel_dirs("nvidia/cu13/bin", "nvidia/npp/bin"),
        dependencies=("nppc",),
    ),
    DescriptorSpec(
        name="nppidei",
        packaged_with="ctk",
        linux_sonames=("libnppidei.so.13", "libnppidei.so.12"),
        windows_dlls=("nppidei64_13.dll", "nppidei64_12.dll"),
        supported_windows_arch=("x64", "arm64"),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/npp/lib"),
        site_packages_windows=_ctk_windows_wheel_dirs("nvidia/cu13/bin", "nvidia/npp/bin"),
        dependencies=("nppc",),
    ),
    DescriptorSpec(
        name="nppif",
        packaged_with="ctk",
        linux_sonames=("libnppif.so.13", "libnppif.so.12"),
        windows_dlls=("nppif64_13.dll", "nppif64_12.dll"),
        supported_windows_arch=("x64", "arm64"),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/npp/lib"),
        site_packages_windows=_ctk_windows_wheel_dirs("nvidia/cu13/bin", "nvidia/npp/bin"),
        dependencies=("nppc",),
    ),
    DescriptorSpec(
        name="nppig",
        packaged_with="ctk",
        linux_sonames=("libnppig.so.13", "libnppig.so.12"),
        windows_dlls=("nppig64_13.dll", "nppig64_12.dll"),
        supported_windows_arch=("x64", "arm64"),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/npp/lib"),
        site_packages_windows=_ctk_windows_wheel_dirs("nvidia/cu13/bin", "nvidia/npp/bin"),
        dependencies=("nppc",),
    ),
    DescriptorSpec(
        name="nppim",
        packaged_with="ctk",
        linux_sonames=("libnppim.so.13", "libnppim.so.12"),
        windows_dlls=("nppim64_13.dll", "nppim64_12.dll"),
        supported_windows_arch=("x64", "arm64"),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/npp/lib"),
        site_packages_windows=_ctk_windows_wheel_dirs("nvidia/cu13/bin", "nvidia/npp/bin"),
        dependencies=("nppc",),
    ),
    DescriptorSpec(
        name="nppist",
        packaged_with="ctk",
        linux_sonames=("libnppist.so.13", "libnppist.so.12"),
        windows_dlls=("nppist64_13.dll", "nppist64_12.dll"),
        supported_windows_arch=("x64", "arm64"),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/npp/lib"),
        site_packages_windows=_ctk_windows_wheel_dirs("nvidia/cu13/bin", "nvidia/npp/bin"),
        dependencies=("nppc",),
    ),
    DescriptorSpec(
        name="nppisu",
        packaged_with="ctk",
        linux_sonames=("libnppisu.so.13", "libnppisu.so.12"),
        windows_dlls=("nppisu64_13.dll", "nppisu64_12.dll"),
        supported_windows_arch=("x64", "arm64"),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/npp/lib"),
        site_packages_windows=_ctk_windows_wheel_dirs("nvidia/cu13/bin", "nvidia/npp/bin"),
        dependencies=("nppc",),
    ),
    DescriptorSpec(
        name="nppitc",
        packaged_with="ctk",
        linux_sonames=("libnppitc.so.13", "libnppitc.so.12"),
        windows_dlls=("nppitc64_13.dll", "nppitc64_12.dll"),
        supported_windows_arch=("x64", "arm64"),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/npp/lib"),
        site_packages_windows=_ctk_windows_wheel_dirs("nvidia/cu13/bin", "nvidia/npp/bin"),
        dependencies=("nppc",),
    ),
    DescriptorSpec(
        name="npps",
        packaged_with="ctk",
        linux_sonames=("libnpps.so.13", "libnpps.so.12"),
        windows_dlls=("npps64_13.dll", "npps64_12.dll"),
        supported_windows_arch=("x64", "arm64"),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/npp/lib"),
        site_packages_windows=_ctk_windows_wheel_dirs("nvidia/cu13/bin", "nvidia/npp/bin"),
        dependencies=("nppc",),
    ),
    DescriptorSpec(
        name="nvblas",
        packaged_with="ctk",
        linux_sonames=("libnvblas.so.13", "libnvblas.so.12"),
        windows_dlls=("nvblas64_13.dll", "nvblas64_12.dll"),
        supported_windows_arch=("x64", "arm64"),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/cublas/lib"),
        site_packages_windows=_ctk_windows_wheel_dirs("nvidia/cu13/bin", "nvidia/cublas/bin"),
        dependencies=("cublas", "cublasLt"),
    ),
    DescriptorSpec(
        name="nvjpeg",
        packaged_with="ctk",
        linux_sonames=("libnvjpeg.so.13", "libnvjpeg.so.12"),
        windows_dlls=("nvjpeg64_13.dll", "nvjpeg64_12.dll"),
        supported_windows_arch=("x64", "arm64"),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/nvjpeg/lib"),
        site_packages_windows=_ctk_windows_wheel_dirs("nvidia/cu13/bin", "nvidia/nvjpeg/bin"),
    ),
    DescriptorSpec(
        name="cufile",
        packaged_with="ctk",
        linux_sonames=("libcufile.so.0",),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/cufile/lib"),
    ),
    DescriptorSpec(
        name="cupti",
        packaged_with="ctk",
        linux_sonames=("libcupti.so.13", "libcupti.so.12"),
        windows_dlls=(
            "cupti64_2026.3.1.dll",
            "cupti64_2026.3.0.dll",
            "cupti64_2026.2.1.dll",
            "cupti64_2026.2.0.dll",
            "cupti64_2026.1.1.dll",
            "cupti64_2026.1.0.dll",
            "cupti64_2025.4.1.dll",
            "cupti64_2025.3.1.dll",
            "cupti64_2025.2.1.dll",
            "cupti64_2025.1.1.dll",
            "cupti64_2024.3.2.dll",
            "cupti64_2024.2.1.dll",
            "cupti64_2024.1.1.dll",
            "cupti64_2023.3.1.dll",
            "cupti64_2023.2.2.dll",
            "cupti64_2023.1.1.dll",
            "cupti64_2022.4.1.dll",
        ),
        supported_windows_arch=("x64", "arm64"),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/cuda_cupti/lib"),
        site_packages_windows=_ctk_windows_wheel_dirs("nvidia/cu13/bin", "nvidia/cuda_cupti/bin"),
        anchor_rel_dirs_linux=("extras/CUPTI/lib64", "lib"),
        # CTK 13.4 uses architecture-qualified CUPTI directories. Older
        # Windows CUPTI toolkits were x64-only and used extras/CUPTI/lib64.
        anchor_rel_dirs_windows=WindowsSearchDirs(
            x64=("extras/CUPTI/lib/x64", "extras/CUPTI/lib64", "bin"),
            arm64=("extras/CUPTI/lib/arm64",),
        ),
        ctk_root_canary_anchor_libnames=CTK_ROOT_CANARY_ANCHOR_LIBNAMES,
    ),
    DescriptorSpec(
        name="cudla",
        packaged_with="ctk",
        linux_sonames=("libcudla.so.1",),
        windows_dlls=("cudla.dll",),
        supported_windows_arch=("arm64",),
        site_packages_linux=("nvidia/cu13/lib",),
        # No Windows pip wheel ships cudla.dll today; it is loaded from the local
        # CUDA Toolkit only, so site_packages_windows is intentionally left empty.
        anchor_rel_dirs_windows=WindowsSearchDirs.arm64_only("bin/arm64"),
    ),
    # -----------------------------------------------------------------------
    # Third-party / separately packaged libraries
    # -----------------------------------------------------------------------
    DescriptorSpec(
        name="cublasmp",
        packaged_with="other",
        linux_sonames=("libcublasmp.so.0",),
        site_packages_linux=("nvidia/cublasmp/cu13/lib", "nvidia/cublasmp/cu12/lib"),
        dependencies=("cublas", "cublasLt", "nvshmem_host"),
    ),
    DescriptorSpec(
        name="cufftMp",
        packaged_with="other",
        linux_sonames=("libcufftMp.so.12", "libcufftMp.so.11"),
        site_packages_linux=("nvidia/cufftmp/cu13/lib", "nvidia/cufftmp/cu12/lib"),
        dependencies=("nvshmem_host",),
        requires_rtld_deepbind=True,
    ),
    DescriptorSpec(
        name="cudnn",
        packaged_with="other",
        linux_sonames=("libcudnn.so.9",),
        windows_dlls=("cudnn64_9.dll",),
        supported_windows_arch=("x64", "arm64"),
        site_packages_linux=("nvidia/cudnn/lib",),
        site_packages_windows=WindowsSearchDirs.x64_only("nvidia/cudnn/bin"),
        dependencies=("cublasLt",),
        optional_dependencies=("nvrtc",),
        install_root_env_vars_linux=("CUDNN_PATH",),
        install_root_env_rel_dirs_linux=("lib", "lib64"),
        # The ARM64 layout is verified only for the standalone archive rooted
        # at CUDNN_PATH, not for conda, CUDA_PATH, or Program Files installs.
        anchor_rel_dirs_windows=WindowsSearchDirs.x64_only("bin/x64", "bin"),
        install_root_env_vars_windows=("CUDNN_PATH",),
        install_root_env_rel_dirs_windows=WindowsSearchDirs(
            x64=("bin/x64", "bin"),
            arm64=("bin/arm64",),
        ),
        program_files_root_globs_windows=WindowsSearchDirs.x64_only("NVIDIA/CUDNN/v9.*"),
        requires_add_dll_directory=True,
    ),
    DescriptorSpec(
        name="cusolverMp",
        packaged_with="other",
        linux_sonames=("libcusolverMp.so.0",),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/cu12/lib"),
        dependencies=("cublas", "cudart", "cusolver", "nccl"),
    ),
    DescriptorSpec(
        name="mathdx",
        packaged_with="other",
        linux_sonames=("libmathdx.so.0",),
        windows_dlls=("mathdx64_0.dll",),
        supported_windows_arch=("x64", "arm64"),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/cu12/lib"),
        site_packages_windows=WindowsSearchDirs(
            x64=("nvidia/cu13/bin", "nvidia/cu12/bin"),
            arm64=("nvidia/cu13/bin", "nvidia/cu12/bin"),
        ),
        dependencies=("nvrtc",),
    ),
    DescriptorSpec(
        name="cudss",
        packaged_with="other",
        linux_sonames=("libcudss.so.0",),
        windows_dlls=("cudss64_0.dll",),
        supported_windows_arch=("x64", "arm64"),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/cu12/lib"),
        site_packages_windows=WindowsSearchDirs(
            x64=("nvidia/cu13/bin", "nvidia/cu12/bin"),
            arm64=("nvidia/cu13/bin", "nvidia/cu12/bin"),
        ),
        dependencies=("cublas", "cublasLt"),
    ),
    DescriptorSpec(
        name="cusparseLt",
        packaged_with="other",
        linux_sonames=("libcusparseLt.so.0",),
        windows_dlls=("cusparseLt_13.dll", "cusparseLt.dll"),
        supported_windows_arch=("x64", "arm64"),
        site_packages_linux=("nvidia/cu13/lib", "nvidia/cusparselt/lib"),
        site_packages_windows=WindowsSearchDirs(
            x64=("nvidia/cu13/bin/x64", "nvidia/cusparselt/bin"),
            arm64=("nvidia/cu13/bin/arm64",),
        ),
    ),
    DescriptorSpec(
        name="cutensor",
        packaged_with="other",
        linux_sonames=("libcutensor.so.2",),
        windows_dlls=("cutensor.dll",),
        supported_windows_arch=("x64", "arm64"),
        site_packages_linux=("cutensor/lib",),
        site_packages_windows=WindowsSearchDirs(x64=("cutensor/bin",), arm64=("cutensor/bin",)),
        dependencies=("cublasLt",),
    ),
    DescriptorSpec(
        name="cutensorMg",
        packaged_with="other",
        linux_sonames=("libcutensorMg.so.2",),
        windows_dlls=("cutensorMg.dll",),
        supported_windows_arch=("x64", "arm64"),
        site_packages_linux=("cutensor/lib",),
        site_packages_windows=WindowsSearchDirs(x64=("cutensor/bin",), arm64=("cutensor/bin",)),
        dependencies=("cutensor", "cublasLt"),
    ),
    DescriptorSpec(
        name="cutensorMp",
        packaged_with="other",
        linux_sonames=("libcutensorMp.so.2",),
        site_packages_linux=("cutensor/lib",),
        dependencies=("cutensor", "cublasLt", "cudart", "nccl"),
    ),
    DescriptorSpec(
        name="cudensitymat",
        packaged_with="other",
        linux_sonames=("libcudensitymat.so.0",),
        site_packages_linux=("cuquantum/lib",),
        dependencies=(
            "cutensornet",
            "cutensor",
            "cusolver",
            "cublasLt",
            "cublas",
            "curand",
            "cusparse",
            "nvJitLink",
        ),
    ),
    DescriptorSpec(
        name="cupauliprop",
        packaged_with="other",
        linux_sonames=("libcupauliprop.so.0",),
        site_packages_linux=("cuquantum/lib",),
    ),
    DescriptorSpec(
        name="cutensornet",
        packaged_with="other",
        linux_sonames=("libcutensornet.so.2",),
        site_packages_linux=("cuquantum/lib",),
        dependencies=("cutensor", "cublas", "cusolver"),
    ),
    DescriptorSpec(
        name="custabilizer",
        packaged_with="other",
        linux_sonames=("libcustabilizer.so.0",),
        site_packages_linux=("cuquantum/lib",),
    ),
    DescriptorSpec(
        name="custatevec",
        packaged_with="other",
        linux_sonames=("libcustatevec.so.1",),
        site_packages_linux=("cuquantum/lib",),
        dependencies=("cublas", "cublasLt"),
    ),
    DescriptorSpec(
        name="nccl",
        packaged_with="other",
        linux_sonames=("libnccl.so.2",),
        site_packages_linux=("nvidia/nccl/lib",),
        install_root_env_vars_linux=("NCCL_HOME",),
        install_root_env_rel_dirs_linux=("lib", "lib64", "build/lib"),
    ),
    DescriptorSpec(
        name="nvpl_fftw",
        packaged_with="other",
        linux_sonames=("libnvpl_fftw.so.0",),
        site_packages_linux=("nvpl/lib",),
    ),
    DescriptorSpec(
        name="nvshmem_host",
        packaged_with="other",
        linux_sonames=("libnvshmem_host.so.3",),
        site_packages_linux=("nvidia/nvshmem/lib",),
    ),
    # -----------------------------------------------------------------------
    # Driver libraries (system-search only, no CTK cascade)
    # -----------------------------------------------------------------------
    DescriptorSpec(
        name="cuda",
        packaged_with="driver",
        linux_sonames=("libcuda.so.1",),
        windows_dlls=("nvcuda.dll",),
        supported_windows_arch=("x64", "arm64"),
    ),
    DescriptorSpec(
        name="nvcudla",
        packaged_with="driver",
        linux_sonames=("libnvcudla.so",),
    ),
    DescriptorSpec(
        name="nvml",
        packaged_with="driver",
        linux_sonames=("libnvidia-ml.so.1",),
        windows_dlls=("nvml.dll",),
        supported_windows_arch=("x64", "arm64"),
    ),
)
<|endoftext|># SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import struct
import subprocess
import sys
from typing import TYPE_CHECKING

from cuda.pathfinder._dynamic_libs.lib_descriptor import LIB_DESCRIPTORS
from cuda.pathfinder._dynamic_libs.load_dl_common import (
    DynamicLibNotAvailableError,
    DynamicLibNotFoundError,
    DynamicLibUnknownError,
    LoadedDL,
    load_dependencies,
)
from cuda.pathfinder._dynamic_libs.platform_loader import LOADER
from cuda.pathfinder._dynamic_libs.search_steps import (
    EARLY_FIND_STEPS,
    LATE_FIND_STEPS,
    SearchContext,
    derive_ctk_root,
    find_via_ctk_root,
    run_find_steps,
)
from cuda.pathfinder._dynamic_libs.subprocess_protocol import (
    DYNAMIC_LIB_SUBPROCESS_CWD,
    MODE_CANARY,
    STATUS_OK,
    DynamicLibSubprocessPayload,
    build_dynamic_lib_subprocess_command,
    parse_dynamic_lib_subprocess_payload,
)
from cuda.pathfinder._dynamic_libs.supported_nvidia_libs import ALL_AVAILABLE_LIBNAMES
from cuda.pathfinder._utils.platform_aware import IS_WINDOWS

if TYPE_CHECKING:
    from cuda.pathfinder._dynamic_libs.lib_descriptor import LibDescriptor

# All libnames recognized by load_nvidia_dynamic_lib, across all categories
# (CTK, third-party, driver).
_ALL_KNOWN_LIBNAMES: frozenset[str] = frozenset(LIB_DESCRIPTORS)
_PLATFORM_NAME = "Windows" if IS_WINDOWS else "Linux"
_CANARY_PROBE_TIMEOUT_SECONDS = 10.0

# Driver libraries: shipped with the NVIDIA display driver, always on the
# system linker path.  These skip all CTK search steps (site-packages,
# conda, CUDA_PATH, canary) and go straight to system search.
_DRIVER_ONLY_LIBNAMES = frozenset(name for name, desc in LIB_DESCRIPTORS.items() if desc.packaged_with == "driver")


def _load_driver_lib_no_cache(desc: LibDescriptor) -> LoadedDL:
    """Load an NVIDIA driver library (system-search only).

    Driver libs (libcuda, libnvidia-ml) are part of the display driver, not
    the CUDA Toolkit. They are expected to be discoverable via the platform's
    native loader mechanisms, so the full CTK search cascade (site-packages,
    conda, CUDA_PATH, canary) is unnecessary.
    """
    loaded = LOADER.check_if_already_loaded_from_elsewhere(desc)
    if loaded is not None:
        return loaded
    loaded = LOADER.load_with_system_search(desc)
    if loaded is not None:
        return loaded
    raise DynamicLibNotFoundError(
        f'"{desc.name}" is an NVIDIA driver library and can only be found via'
        f" system search. Ensure the NVIDIA display driver is installed."
    )


def _coerce_subprocess_output(output: str | bytes | None) -> str:
    if isinstance(output, bytes):
        return output.decode(errors="replace")
    return "" if output is None else output


def _raise_canary_probe_child_process_error(
    *,
    returncode: int | None = None,
    timeout: float | None = None,
    stderr: str | bytes | None = None,
) -> None:
    if timeout is None:
        error_line = f"Canary probe child process exited with code {returncode}."
    else:
        error_line = f"Canary probe child process timed out after {timeout} seconds."
    raise ChildProcessError(
        f"{error_line}\n"
        "--- stderr-from-child-process ---\n"
        f"{_coerce_subprocess_output(stderr)}"
        "<end-of-stderr-from-child-process>\n"
    )


@functools.cache
def _resolve_system_loaded_abs_path_in_subprocess(
    libname: str,
    *,
    timeout: float = _CANARY_PROBE_TIMEOUT_SECONDS,
) -> str | None:
    """Resolve a canary library's absolute path in a fresh Python subprocess."""
    try:
        result = subprocess.run(  # noqa: S603 - trusted argv: current interpreter + internal probe module
            build_dynamic_lib_subprocess_command(MODE_CANARY, libname),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            cwd=DYNAMIC_LIB_SUBPROCESS_CWD,
        )
    except subprocess.TimeoutExpired as exc:
        _raise_canary_probe_child_process_error(timeout=exc.timeout, stderr=exc.stderr)

    if result.returncode != 0:
        _raise_canary_probe_child_process_error(returncode=result.returncode, stderr=result.stderr)

    payload: DynamicLibSubprocessPayload = parse_dynamic_lib_subprocess_payload(
        result.stdout,
        libname=libname,
        error_label="Canary probe child process",
    )
    abs_path: str | None = payload.abs_path
    if payload.status == STATUS_OK:
        return abs_path
    return None


def _loadable_via_canary_subprocess(libname: str, *, timeout: float = _CANARY_PROBE_TIMEOUT_SECONDS) -> bool:
    """Return True if the canary subprocess can resolve ``libname`` via system search."""
    return _resolve_system_loaded_abs_path_in_subprocess(libname, timeout=timeout) is not None


def resolve_ctk_root_via_canary(canary_libname: str) -> str | None:
    """Resolve the CUDA Toolkit root from a system-loadable canary library.

    The canary library's absolute path is resolved by the OS dynamic loader in
    an isolated subprocess, which honors ``LD_LIBRARY_PATH`` on Linux and the
    native DLL search on Windows. The toolkit root is then derived from that
    path. Returns ``None`` if the canary cannot be resolved or no root can be
    derived.
    """
    canary_abs_path = _resolve_system_loaded_abs_path_in_subprocess(canary_libname)
    if canary_abs_path is None:
        return None
    ctk_root: str | None = derive_ctk_root(canary_abs_path)
    return ctk_root


def _try_ctk_root_canary(ctx: SearchContext) -> str | None:
    """Try CTK-root canary fallback for descriptor-configured libraries."""
    for canary_libname in ctx.desc.ctk_root_canary_anchor_libnames:
        ctk_root = resolve_ctk_root_via_canary(canary_libname)
        if ctk_root is None:
            continue
        find = find_via_ctk_root(ctx, ctk_root)
        if find is not None:
            return str(find.abs_path)
    return None


def _load_lib_no_cache(libname: str) -> LoadedDL:
    desc = LIB_DESCRIPTORS[libname]

    if libname in _DRIVER_ONLY_LIBNAMES:
        return _load_driver_lib_no_cache(desc)

    ctx = SearchContext(desc)

    # Phase 1: Try to find the library file on disk (pip wheels, conda).
    find = run_find_steps(ctx, EARLY_FIND_STEPS)

    # Phase 2: Cross-cutting — already-loaded check and dependency loading.
    loaded = LOADER.check_if_already_loaded_from_elsewhere(desc)
    load_dependencies(desc, load_nvidia_dynamic_lib)
    if loaded is not None:
        return loaded

    # Phase 3: Load from found path, or fall back to system search + late find.
    if find is not None:
        return LOADER.load_with_abs_path(desc, find.abs_path, find.found_via)

    loaded = LOADER.load_with_system_search(desc)
    if loaded is not None:
        return loaded

    find = run_find_steps(ctx, LATE_FIND_STEPS)
    if find is not None:
        return LOADER.load_with_abs_path(desc, find.abs_path, find.found_via)

    if desc.ctk_root_canary_anchor_libnames:
        canary_abs_path = _try_ctk_root_canary(ctx)
        if canary_abs_path is not None:
            return LOADER.load_with_abs_path(desc, canary_abs_path, "system-ctk-root")

    ctx.raise_not_found()


@functools.cache
def load_nvidia_dynamic_lib(libname: str) -> LoadedDL:
    """Load an NVIDIA dynamic library by name.

    Args:
        libname (str): The short name of the library to load (e.g., ``"cudart"``,
            ``"nvvm"``, etc.).

    Returns:
        LoadedDL: Object containing the OS library handle and absolute path.

        **Important:**

        **Never close the returned handle.** Do **not** call ``dlclose`` (Linux) or
        ``FreeLibrary`` (Windows) on the ``LoadedDL._handle_uint``.

        **Why:** the return value is cached (``functools.cache``) and shared across the
        process. Closing the handle can unload the module while other code still uses
        it, leading to crashes or subtle failures.

        This applies to Linux and Windows. For context, see issue #1011:
        https://github.com/NVIDIA/cuda-python/issues/1011

    Raises:
        DynamicLibUnknownError: If ``libname`` is not a recognized library name.
        DynamicLibNotAvailableError: If ``libname`` is recognized but not
            supported on this platform.
        DynamicLibNotFoundError: If the library cannot be found or loaded.
        RuntimeError: If Python is not 64-bit.

    Windows on ARM (WoA) Note:
        On Windows, this API aims to load a dynamic library whose architecture
        matches the Python interpreter architecture. For example, x64 Python
        running on an Arm64 machine targets an x64 DLL, while native Arm64 Python
        targets an Arm64 DLL. A library loaded into the Python process must be
        compatible with that process. This differs from
        ``find_nvidia_binary_utility``, which targets the native machine
        architecture when selecting architecture-specific executables.

    Search order:
        0. **Already loaded in the current process**

           - If a matching library is already loaded by some other component,
             return its absolute path and handle and skip the rest of the search.

        1. **NVIDIA Python wheels**

           - Scan installed distributions (``site-packages``) to find libraries
             shipped in NVIDIA wheels.

        2. **Conda environment**

           - Conda installations are discovered via ``CONDA_PREFIX``, which is
             defined automatically in activated conda environments (see
             https://docs.conda.io/projects/conda-build/en/stable/user-guide/environment-variables.html).

        3. **OS default mechanisms**

           - Fall back to the native loader:

             - Linux: ``dlopen()``

             - Windows: ``LoadLibraryExW()``

             On Linux, CUDA Toolkit (CTK) system installs with system config updates are
             usually discovered via ``/etc/ld.so.conf.d/*cuda*.conf``.

             On Windows, under Python 3.8+, CPython configures the process with
             ``SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)``.
             As a result, the native DLL search used here does **not** include
             the system ``PATH``.

        4. **Environment variables**

           - First search library-specific roots declared by the descriptor,
             such as ``CUDNN_PATH`` and ``NCCL_HOME``, using their
             platform-specific product layouts. Then use ``CUDA_PATH`` or
             ``CUDA_HOME`` (in that order).
             On Windows, ``CUDA_PATH`` is the typical way system-installed CTK
             DLLs are located. Note that the NVIDIA CTK installer automatically
             adds ``CUDA_PATH`` to the system-wide environment.

        5. **Windows Program Files (configured libraries only)**

           - Search descriptor-configured standalone installation roots, such
             as versioned x64 cuDNN directories under ``ProgramFiles``, using
             the general per-library anchor layout.

        6. **CTK root canary probe (discoverable libs only)**

           - For selected libraries whose shared object doesn't reside on the
             standard linker path (currently ``nvvm``), attempt to derive CTK
             root by system-loading a well-known CTK canary library in a
             subprocess and then searching relative to that root. On Windows,
             the canary uses the same native ``LoadLibraryExW`` semantics as
             step 3, so there is also no ``PATH``-based discovery.

    **Driver libraries** (``"cuda"``, ``"nvml"``):

        These are part of the NVIDIA display driver (not the CUDA Toolkit) and
        are expected to be reachable via the native OS loader path. For these
        libraries the search is simplified to:

        0. Already loaded in the current process
        1. OS default mechanisms (``dlopen`` / ``LoadLibraryExW``)

        The non-driver steps (site-packages, conda, environment roots,
        ``ProgramFiles``, and canary probe) are skipped entirely.

    Notes:
        The search is performed **per library**. There is currently no mechanism to
        guarantee that multiple libraries are all resolved from the same location.

    """
    pointer_size_bits = struct.calcsize("P") * 8
    if pointer_size_bits != 64:
        raise RuntimeError(
            f"cuda.pathfinder.load_nvidia_dynamic_lib() requires 64-bit Python."
            f" Currently running: {pointer_size_bits}-bit Python"
            f" {sys.version_info.major}.{sys.version_info.minor}"
        )
    if libname not in _ALL_KNOWN_LIBNAMES:
        raise DynamicLibUnknownError(f"Unknown library name: {libname!r}. Known names: {sorted(_ALL_KNOWN_LIBNAMES)}")
    if libname not in ALL_AVAILABLE_LIBNAMES:
        raise DynamicLibNotAvailableError(
            f"Library name {libname!r} is known but not available on {_PLATFORM_NAME}. "
            f"Supported names on {_PLATFORM_NAME}: {sorted(ALL_AVAILABLE_LIBNAMES)}"
        )
    return _load_lib_no_cache(libname)
<|endoftext|># SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cuda.pathfinder._dynamic_libs.lib_descriptor import LibDescriptor


class DynamicLibNotFoundError(RuntimeError):
    pass


class DynamicLibNotAvailableError(DynamicLibNotFoundError):
    pass


class DynamicLibUnknownError(DynamicLibNotFoundError):
    pass


@dataclass
class LoadedDL:
    abs_path: str | None
    was_already_loaded_from_elsewhere: bool
    _handle_uint: int  # Platform-agnostic unsigned pointer value
    found_via: str  # "CUDA_PATH" covers both CUDA_PATH and CUDA_HOME env vars


def load_dependencies(desc: LibDescriptor, load_func: Callable[[str], LoadedDL]) -> None:
    """Load required dependencies, then best-effort runtime dependencies.

    A plain ``DynamicLibNotFoundError`` from an optional dependency is
    suppressed. More specific contract errors and failures while loading a
    dependency that was found remain errors.
    """
    for dep in desc.dependencies:
        load_func(dep)
    for dep in desc.optional_dependencies:
        try:
            load_func(dep)
        except DynamicLibNotFoundError as exc:
            # Both public contract errors inherit DynamicLibNotFoundError, but
            # neither an unknown descriptor nor platform incompatibility means
            # that an optional runtime component is simply absent.
            if type(exc) is not DynamicLibNotFoundError:
                raise
<|endoftext|># SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes
import ctypes.wintypes
import os
import struct
import sys
import warnings
from typing import TYPE_CHECKING

from cuda.pathfinder._dynamic_libs.load_dl_common import LoadedDL

if TYPE_CHECKING:
    from cuda.pathfinder._dynamic_libs.lib_descriptor import LibDescriptor

# Mirrors WinBase.h (unfortunately not defined already elsewhere)
WINBASE_LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR = 0x00000100
WINBASE_LOAD_LIBRARY_SEARCH_DEFAULT_DIRS = 0x00001000

POINTER_ADDRESS_SPACE = 2 ** (struct.calcsize("P") * 8)

# Set up kernel32 functions with proper types
windll = getattr(ctypes, "windll", None)
if windll is None:
    raise RuntimeError("ctypes.windll is required on Windows")
kernel32 = windll.kernel32

# GetModuleHandleW
kernel32.GetModuleHandleW.argtypes = [ctypes.wintypes.LPCWSTR]
kernel32.GetModuleHandleW.restype = ctypes.wintypes.HMODULE

# LoadLibraryExW
kernel32.LoadLibraryExW.argtypes = [
    ctypes.wintypes.LPCWSTR,  # lpLibFileName
    ctypes.wintypes.HANDLE,  # hFile (reserved, must be NULL)
    ctypes.wintypes.DWORD,  # dwFlags
]
kernel32.LoadLibraryExW.restype = ctypes.wintypes.HMODULE

# GetModuleFileNameW
kernel32.GetModuleFileNameW.argtypes = [
    ctypes.wintypes.HMODULE,  # hModule
    ctypes.wintypes.LPWSTR,  # lpFilename
    ctypes.wintypes.DWORD,  # nSize
]
kernel32.GetModuleFileNameW.restype = ctypes.wintypes.DWORD


# GetLastError
kernel32.GetLastError.argtypes = []
kernel32.GetLastError.restype = ctypes.wintypes.DWORD


def ctypes_handle_to_unsigned_int(handle: ctypes.wintypes.HMODULE) -> int:
    """Convert ctypes HMODULE to unsigned int."""
    handle_uint = int(handle)
    if handle_uint < 0:
        # Convert from signed to unsigned representation
        handle_uint += POINTER_ADDRESS_SPACE
    return handle_uint


def add_dll_directory(dll_abs_path: str) -> None:
    """Add a DLL directory to the search path and update PATH environment variable.

    Args:
        dll_abs_path: Absolute path to the DLL file

    Raises:
        AssertionError: If the directory containing the DLL does not exist
    """
    dirpath = os.path.dirname(dll_abs_path)
    assert os.path.isdir(dirpath), dll_abs_path

    # Add the DLL directory to the native search path via the stdlib wrapper
    # around AddDllDirectory. This only affects the LOAD_LIBRARY_SEARCH_USER_DIRS
    # search; PATH is updated unconditionally below to also cover legacy
    # dependent-DLL resolution. The returned handle is intentionally discarded:
    # the directory must stay on the search path for the process lifetime, and
    # the handle has no finalizer, so dropping it does not remove the directory.
    try:
        if sys.platform == "win32":
            os.add_dll_directory(dirpath)
    except OSError as e:
        # Warn instead of failing silently; the PATH update below is a weaker
        # fallback that newer loaders may ignore.
        warnings.warn(
            f"os.add_dll_directory({dirpath!r}) failed ({e}); "
            "falling back to process-global PATH mutation for dependent-DLL resolution.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Update PATH as a fallback for dependent DLL resolution
    curr_path = os.environ.get("PATH")
    os.environ["PATH"] = dirpath if curr_path is None else os.pathsep.join((curr_path, dirpath))


def abs_path_for_dynamic_library(libname: str, handle: ctypes.wintypes.HMODULE) -> str:
    """Get the absolute path of a loaded dynamic library on Windows."""
    # Create buffer for the path
    buffer = ctypes.create_unicode_buffer(260)  # MAX_PATH
    length = kernel32.GetModuleFileNameW(handle, buffer, len(buffer))

    if length == 0:
        error_code = kernel32.GetLastError()
        raise RuntimeError(f"GetModuleFileNameW failed for {libname!r} (error code: {error_code})")

    # If buffer was too small, try with larger buffer
    if length == len(buffer):
        buffer = ctypes.create_unicode_buffer(32768)  # Extended path length
        length = kernel32.GetModuleFileNameW(handle, buffer, len(buffer))
        if length == 0:
            error_code = kernel32.GetLastError()
            raise RuntimeError(f"GetModuleFileNameW failed for {libname!r} (error code: {error_code})")

    return buffer.value


def check_if_already_loaded_from_elsewhere(desc: LibDescriptor) -> LoadedDL | None:
    for dll_name in desc.windows_dlls:
        handle = kernel32.GetModuleHandleW(dll_name)
        if handle:
            abs_path = abs_path_for_dynamic_library(desc.name, handle)
            if desc.requires_add_dll_directory:
                # Match load_with_abs_path(): lazy component DLLs need the directory
                # of the module that is actually loaded, regardless of how it arrived.
                add_dll_directory(abs_path)
            return LoadedDL(abs_path, True, ctypes_handle_to_unsigned_int(handle), "was-already-loaded-from-elsewhere")
    return None


def load_with_system_search(desc: LibDescriptor) -> LoadedDL | None:
    """Try to load a DLL using the native Windows process DLL search path.

    This calls ``LoadLibraryExW(dll_name, NULL, 0)`` directly. Under Python
    3.8+, CPython configures the process with
    ``SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)``, so this
    search does **not** include the system ``PATH``. Directories added via
    ``AddDllDirectory()`` still participate.

    Args:
        desc: Descriptor for the library to load

    Returns:
        A LoadedDL object if successful, None if the library cannot be loaded
    """
    for dll_name in desc.windows_dlls:
        handle = kernel32.LoadLibraryExW(dll_name, None, 0)
        if handle:
            abs_path = abs_path_for_dynamic_library(desc.name, handle)
            return LoadedDL(abs_path, False, ctypes_handle_to_unsigned_int(handle), "system-search")

    return None


def load_with_abs_path(desc: LibDescriptor, found_path: str, found_via: str | None = None) -> LoadedDL:
    """Load a dynamic library from the given path.

    Args:
        desc: Descriptor for the library to load.
        found_path: The absolute path to the DLL file.
        found_via: Label indicating how the path was discovered.

    Returns:
        A LoadedDL object representing the loaded library.

    Raises:
        RuntimeError: If the DLL cannot be loaded.
    """
    if desc.requires_add_dll_directory:
        add_dll_directory(found_path)

    flags = WINBASE_LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | WINBASE_LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR
    handle = kernel32.LoadLibraryExW(found_path, None, flags)

    if not handle:
        error_code = kernel32.GetLastError()
        raise RuntimeError(f"Failed to load DLL at {found_path}: Windows error {error_code}")

    return LoadedDL(found_path, False, ctypes_handle_to_unsigned_int(handle), found_via)
<|endoftext|># SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Platform abstraction for filesystem search steps.

The goal is to keep :mod:`search_steps` platform-agnostic: it should not branch
on OS flags like ``IS_WINDOWS``. Instead, it calls through the single
``PLATFORM`` instance exported here.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import PurePath
from typing import Protocol, cast

from cuda.pathfinder._dynamic_libs.lib_descriptor import LibDescriptor
from cuda.pathfinder._dynamic_libs.supported_nvidia_libs import is_suppressed_dll_file
from cuda.pathfinder._utils.find_sub_dirs import find_sub_dirs_all_sitepackages
from cuda.pathfinder._utils.platform_aware import IS_WINDOWS
from cuda.pathfinder._utils.windows_arch import windows_pe_matches_arch, windows_python_arch


def _no_such_file_in_sub_dirs(
    sub_dirs: Sequence[str], file_description: str, error_messages: list[str], attachments: list[str]
) -> None:
    error_messages.append(f"No such file: {file_description}")
    for sub_dir in find_sub_dirs_all_sitepackages(sub_dirs):
        attachments.append(f'  listdir("{sub_dir}"):')
        for node in sorted(os.listdir(sub_dir)):
            attachments.append(f"    {node}")


def _find_descriptor_so_under_dir(dirpath: str, desc: LibDescriptor) -> str | None:
    for soname in desc.linux_sonames:
        path = os.path.join(dirpath, soname)
        if os.path.isfile(path):
            return path
    return None


def _find_so_in_rel_dirs(
    rel_dirs: tuple[str, ...],
    desc: LibDescriptor,
    file_description: str,
    error_messages: list[str],
    attachments: list[str],
) -> str | None:
    sub_dirs_searched: list[tuple[str, ...]] = []
    for rel_dir in rel_dirs:
        sub_dir = PurePath(rel_dir).parts
        for abs_dir in find_sub_dirs_all_sitepackages(sub_dir):
            so_path = _find_descriptor_so_under_dir(abs_dir, desc)
            if so_path is not None:
                return so_path
        sub_dirs_searched.append(sub_dir)
    for sub_dir in sub_dirs_searched:
        _no_such_file_in_sub_dirs(sub_dir, file_description, error_messages, attachments)
    return None


def _find_descriptor_dll_under_dir(
    dirpath: str,
    desc: LibDescriptor,
    target_arch: str | None = None,
) -> str | None:
    def candidate_is_usable(path: str) -> bool:
        if not os.path.isfile(path):
            return False
        if is_suppressed_dll_file(os.path.basename(path)):
            return False
        return target_arch is None or windows_pe_matches_arch(path, target_arch)

    for dll_basename in desc.windows_dlls:
        path = os.path.join(dirpath, dll_basename)
        if candidate_is_usable(path):
            return path
    return None


def _find_dll_in_rel_dirs(
    rel_dirs: tuple[str, ...],
    desc: LibDescriptor,
    target_arch: str,
    lib_searched_for: str,
    error_messages: list[str],
    attachments: list[str],
) -> str | None:
    sub_dirs_searched: list[tuple[str, ...]] = []
    checked_arch = target_arch if desc.requires_windows_binary_arch_check else None
    for rel_dir in rel_dirs:
        sub_dir = PurePath(rel_dir).parts
        for abs_dir in find_sub_dirs_all_sitepackages(sub_dir):
            dll_name = _find_descriptor_dll_under_dir(abs_dir, desc, checked_arch)
            if dll_name is not None:
                return dll_name
        sub_dirs_searched.append(sub_dir)
    for sub_dir in sub_dirs_searched:
        _no_such_file_in_sub_dirs(sub_dir, lib_searched_for, error_messages, attachments)
    return None


class SearchPlatform(Protocol):
    def lib_searched_for(self, desc: LibDescriptor) -> str: ...

    def site_packages_rel_dirs(self, desc: LibDescriptor) -> tuple[str, ...]: ...

    def conda_anchor_point(self, conda_prefix: str) -> str: ...

    def anchor_rel_dirs(self, desc: LibDescriptor) -> tuple[str, ...]: ...

    def install_root_env_vars(self, desc: LibDescriptor) -> tuple[str, ...]: ...

    def install_root_env_rel_dirs(self, desc: LibDescriptor) -> tuple[str, ...]: ...

    def program_files_root_globs(self, desc: LibDescriptor) -> tuple[str, ...]: ...

    def find_in_site_packages(
        self,
        rel_dirs: tuple[str, ...],
        desc: LibDescriptor,
        error_messages: list[str],
        attachments: list[str],
    ) -> str | None: ...

    def find_in_lib_dir(
        self,
        lib_dir: str,
        desc: LibDescriptor,
        error_messages: list[str],
        attachments: list[str],
    ) -> str | None: ...


@dataclass(frozen=True, slots=True)
class LinuxSearchPlatform:
    def lib_searched_for(self, desc: LibDescriptor) -> str:
        return " or ".join(desc.linux_sonames)

    def site_packages_rel_dirs(self, desc: LibDescriptor) -> tuple[str, ...]:
        return cast(tuple[str, ...], desc.site_packages_linux)

    def conda_anchor_point(self, conda_prefix: str) -> str:
        return conda_prefix

    def anchor_rel_dirs(self, desc: LibDescriptor) -> tuple[str, ...]:
        return cast(tuple[str, ...], desc.anchor_rel_dirs_linux)

    def install_root_env_vars(self, desc: LibDescriptor) -> tuple[str, ...]:
        return cast(tuple[str, ...], desc.install_root_env_vars_linux)

    def install_root_env_rel_dirs(self, desc: LibDescriptor) -> tuple[str, ...]:
        return cast(tuple[str, ...], desc.install_root_env_rel_dirs_linux)

    def program_files_root_globs(self, _desc: LibDescriptor) -> tuple[str, ...]:
        return ()

    def find_in_site_packages(
        self,
        rel_dirs: tuple[str, ...],
        desc: LibDescriptor,
        error_messages: list[str],
        attachments: list[str],
    ) -> str | None:
        return _find_so_in_rel_dirs(rel_dirs, desc, self.lib_searched_for(desc), error_messages, attachments)

    def find_in_lib_dir(
        self,
        lib_dir: str,
        desc: LibDescriptor,
        error_messages: list[str],
        attachments: list[str],
    ) -> str | None:
        so_path = _find_descriptor_so_under_dir(lib_dir, desc)
        if so_path is not None:
            return so_path
        error_messages.append(f"No such file: {self.lib_searched_for(desc)}")
        attachments.append(f'  listdir("{lib_dir}"):')
        if not os.path.isdir(lib_dir):
            attachments.append("    DIRECTORY DOES NOT EXIST")
        else:
            for node in sorted(os.listdir(lib_dir)):
                attachments.append(f"    {node}")
        return None


@dataclass(frozen=True, slots=True)
class WindowsSearchPlatform:
    target_arch: str

    def lib_searched_for(self, desc: LibDescriptor) -> str:
        return f"known {desc.name} DLL"

    def site_packages_rel_dirs(self, desc: LibDescriptor) -> tuple[str, ...]:
        return cast(tuple[str, ...], desc.site_packages_windows.for_arch(self.target_arch))

    def conda_anchor_point(self, conda_prefix: str) -> str:
        return os.path.join(conda_prefix, "Library")

    def anchor_rel_dirs(self, desc: LibDescriptor) -> tuple[str, ...]:
        return cast(tuple[str, ...], desc.anchor_rel_dirs_windows.for_arch(self.target_arch))

    def install_root_env_vars(self, desc: LibDescriptor) -> tuple[str, ...]:
        if self.target_arch not in desc.supported_windows_arch:
            return ()
        return cast(tuple[str, ...], desc.install_root_env_vars_windows)

    def install_root_env_rel_dirs(self, desc: LibDescriptor) -> tuple[str, ...]:
        if self.target_arch not in desc.supported_windows_arch:
            return ()
        return cast(tuple[str, ...], desc.install_root_env_rel_dirs_windows.for_arch(self.target_arch))

    def program_files_root_globs(self, desc: LibDescriptor) -> tuple[str, ...]:
        program_files = os.environ.get("PROGRAMW6432") or os.environ.get("PROGRAMFILES")
        if not program_files:
            return ()
        rel_globs = desc.program_files_root_globs_windows.for_arch(self.target_arch)
        return tuple(os.path.join(program_files, rel_glob) for rel_glob in rel_globs)

    def find_in_site_packages(
        self,
        rel_dirs: tuple[str, ...],
        desc: LibDescriptor,
        error_messages: list[str],
        attachments: list[str],
    ) -> str | None:
        return _find_dll_in_rel_dirs(
            rel_dirs,
            desc,
            self.target_arch,
            self.lib_searched_for(desc),
            error_messages,
            attachments,
        )

    def find_in_lib_dir(
        self,
        lib_dir: str,
        desc: LibDescriptor,
        error_messages: list[str],
        attachments: list[str],
    ) -> str | None:
        target_arch = self.target_arch if desc.requires_windows_binary_arch_check else None
        dll_name = _find_descriptor_dll_under_dir(lib_dir, desc, target_arch)
        if dll_name is not None:
            return dll_name
        lib_searched_for = self.lib_searched_for(desc)
        if target_arch is None:
            error_messages.append(f"No such file: {lib_searched_for}")
        else:
            error_messages.append(f"No {target_arch}-compatible PE file: {lib_searched_for}")
        attachments.append(f'  listdir("{lib_dir}"):')
        if not os.path.isdir(lib_dir):
            attachments.append("    DIRECTORY DOES NOT EXIST")
        else:
            for node in sorted(os.listdir(lib_dir)):
                attachments.append(f"    {node}")
        return None


def _platform_for_current_system() -> SearchPlatform:
    if IS_WINDOWS:
        return WindowsSearchPlatform(target_arch=windows_python_arch())
    return LinuxSearchPlatform()


PLATFORM = _platform_for_current_system()
<|endoftext|># SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Legacy table exports derived from the authored descriptor catalog.

The canonical data entry point is :mod:`descriptor_catalog`. This module keeps
historical constant names for backward compatibility by deriving them from the
catalog.

The unsuffixed ``SUPPORTED_LIBNAMES_WINDOWS`` and
``SITE_PACKAGES_LIBDIRS_WINDOWS*`` constants retain their historical x64
meaning for compatibility, but are not recommended for new code. Use the
explicit ``*_X64`` or ``*_ARM64`` projection instead. Never combine the two
architecture projections.
"""

from __future__ import annotations

from cuda.pathfinder._dynamic_libs.descriptor_catalog import DESCRIPTOR_CATALOG
from cuda.pathfinder._utils.platform_aware import IS_WINDOWS, IS_WINDOWS_ARM64, IS_WINDOWS_X64

_CTK_DESCRIPTORS = tuple(desc for desc in DESCRIPTOR_CATALOG if desc.packaged_with == "ctk")
_OTHER_DESCRIPTORS = tuple(desc for desc in DESCRIPTOR_CATALOG if desc.packaged_with == "other")
_DRIVER_DESCRIPTORS = tuple(desc for desc in DESCRIPTOR_CATALOG if desc.packaged_with == "driver")
_NON_CTK_DESCRIPTORS = _OTHER_DESCRIPTORS + _DRIVER_DESCRIPTORS


def _legacy_least_preferred_first(names: tuple[str, ...]) -> tuple[str, ...]:
    """Preserve the historical ordering of legacy filename projections."""
    return tuple(reversed(names))


SUPPORTED_LIBNAMES_COMMON = tuple(desc.name for desc in _CTK_DESCRIPTORS if desc.linux_sonames and desc.windows_dlls)
SUPPORTED_LIBNAMES_LINUX_ONLY = tuple(
    desc.name for desc in _CTK_DESCRIPTORS if desc.linux_sonames and not desc.windows_dlls
)
SUPPORTED_LIBNAMES_WINDOWS_ONLY = tuple(
    desc.name for desc in _CTK_DESCRIPTORS if desc.windows_dlls and not desc.linux_sonames
)

if not IS_WINDOWS:
    ALL_AVAILABLE_LIBNAMES = frozenset(desc.name for desc in DESCRIPTOR_CATALOG if desc.linux_sonames)
else:
    assert IS_WINDOWS_X64 != IS_WINDOWS_ARM64
    _current_windows_arch = "x64" if IS_WINDOWS_X64 else "arm64"
    ALL_AVAILABLE_LIBNAMES = frozenset(
        desc.name for desc in DESCRIPTOR_CATALOG if _current_windows_arch in desc.supported_windows_arch
    )

SUPPORTED_LIBNAMES_LINUX = SUPPORTED_LIBNAMES_COMMON + SUPPORTED_LIBNAMES_LINUX_ONLY
SUPPORTED_LIBNAMES_WINDOWS_X64 = tuple(desc.name for desc in _CTK_DESCRIPTORS if "x64" in desc.supported_windows_arch)
SUPPORTED_LIBNAMES_WINDOWS_ARM64 = tuple(
    desc.name for desc in _CTK_DESCRIPTORS if "arm64" in desc.supported_windows_arch
)
# Backward-compatible alias preserves the historical x64 meaning.
SUPPORTED_LIBNAMES_WINDOWS = SUPPORTED_LIBNAMES_WINDOWS_X64
SUPPORTED_LIBNAMES_ALL = SUPPORTED_LIBNAMES_COMMON + SUPPORTED_LIBNAMES_LINUX_ONLY + SUPPORTED_LIBNAMES_WINDOWS_ONLY
if not IS_WINDOWS:
    SUPPORTED_LIBNAMES = SUPPORTED_LIBNAMES_LINUX
elif IS_WINDOWS_X64:
    SUPPORTED_LIBNAMES = SUPPORTED_LIBNAMES_WINDOWS_X64
else:
    assert IS_WINDOWS_ARM64
    SUPPORTED_LIBNAMES = SUPPORTED_LIBNAMES_WINDOWS_ARM64

DIRECT_DEPENDENCIES_CTK = {desc.name: desc.dependencies for desc in _CTK_DESCRIPTORS if desc.dependencies}
DIRECT_DEPENDENCIES = {desc.name: desc.dependencies for desc in DESCRIPTOR_CATALOG if desc.dependencies}

SUPPORTED_LINUX_SONAMES_CTK = {
    desc.name: _legacy_least_preferred_first(desc.linux_sonames) for desc in _CTK_DESCRIPTORS if desc.linux_sonames
}
SUPPORTED_LINUX_SONAMES_OTHER = {
    desc.name: _legacy_least_preferred_first(desc.linux_sonames) for desc in _OTHER_DESCRIPTORS if desc.linux_sonames
}
SUPPORTED_LINUX_SONAMES_DRIVER = {
    desc.name: _legacy_least_preferred_first(desc.linux_sonames) for desc in _DRIVER_DESCRIPTORS if desc.linux_sonames
}
SUPPORTED_LINUX_SONAMES = SUPPORTED_LINUX_SONAMES_CTK | SUPPORTED_LINUX_SONAMES_OTHER | SUPPORTED_LINUX_SONAMES_DRIVER

SUPPORTED_WINDOWS_DLLS_CTK = {
    desc.name: _legacy_least_preferred_first(desc.windows_dlls) for desc in _CTK_DESCRIPTORS if desc.windows_dlls
}
SUPPORTED_WINDOWS_DLLS_OTHER = {
    desc.name: _legacy_least_preferred_first(desc.windows_dlls) for desc in _OTHER_DESCRIPTORS if desc.windows_dlls
}
SUPPORTED_WINDOWS_DLLS_DRIVER = {
    desc.name: _legacy_least_preferred_first(desc.windows_dlls) for desc in _DRIVER_DESCRIPTORS if desc.windows_dlls
}
SUPPORTED_WINDOWS_DLLS = SUPPORTED_WINDOWS_DLLS_CTK | SUPPORTED_WINDOWS_DLLS_OTHER | SUPPORTED_WINDOWS_DLLS_DRIVER

LIBNAMES_REQUIRING_OS_ADD_DLL_DIRECTORY = tuple(
    desc.name for desc in DESCRIPTOR_CATALOG if desc.requires_add_dll_directory and desc.windows_dlls
)
LIBNAMES_REQUIRING_RTLD_DEEPBIND = tuple(
    desc.name for desc in DESCRIPTOR_CATALOG if desc.requires_rtld_deepbind and desc.linux_sonames
)

SITE_PACKAGES_LIBDIRS_LINUX_CTK = {
    desc.name: desc.site_packages_linux for desc in _CTK_DESCRIPTORS if desc.site_packages_linux
}
SITE_PACKAGES_LIBDIRS_LINUX_OTHER = {
    desc.name: desc.site_packages_linux for desc in _NON_CTK_DESCRIPTORS if desc.site_packages_linux
}
SITE_PACKAGES_LIBDIRS_LINUX = SITE_PACKAGES_LIBDIRS_LINUX_CTK | SITE_PACKAGES_LIBDIRS_LINUX_OTHER

# Architecture-specific Windows projections. Keep these separate: combining
# them would make the table unsafe to consume for either process ABI.
SITE_PACKAGES_LIBDIRS_WINDOWS_CTK_X64 = {
    desc.name: desc.site_packages_windows.x64 for desc in _CTK_DESCRIPTORS if desc.site_packages_windows.x64
}
SITE_PACKAGES_LIBDIRS_WINDOWS_CTK_ARM64 = {
    desc.name: desc.site_packages_windows.arm64 for desc in _CTK_DESCRIPTORS if desc.site_packages_windows.arm64
}
SITE_PACKAGES_LIBDIRS_WINDOWS_OTHER_X64 = {
    desc.name: desc.site_packages_windows.x64 for desc in _NON_CTK_DESCRIPTORS if desc.site_packages_windows.x64
}
SITE_PACKAGES_LIBDIRS_WINDOWS_OTHER_ARM64 = {
    desc.name: desc.site_packages_windows.arm64 for desc in _NON_CTK_DESCRIPTORS if desc.site_packages_windows.arm64
}
SITE_PACKAGES_LIBDIRS_WINDOWS_X64 = SITE_PACKAGES_LIBDIRS_WINDOWS_CTK_X64 | SITE_PACKAGES_LIBDIRS_WINDOWS_OTHER_X64
SITE_PACKAGES_LIBDIRS_WINDOWS_ARM64 = (
    SITE_PACKAGES_LIBDIRS_WINDOWS_CTK_ARM64 | SITE_PACKAGES_LIBDIRS_WINDOWS_OTHER_ARM64
)

# Backward-compatible aliases preserve the historical x64 meaning.
SITE_PACKAGES_LIBDIRS_WINDOWS_CTK = SITE_PACKAGES_LIBDIRS_WINDOWS_CTK_X64
SITE_PACKAGES_LIBDIRS_WINDOWS_OTHER = SITE_PACKAGES_LIBDIRS_WINDOWS_OTHER_X64
SITE_PACKAGES_LIBDIRS_WINDOWS = SITE_PACKAGES_LIBDIRS_WINDOWS_X64


def is_suppressed_dll_file(path_basename: str) -> bool:
    if path_basename.startswith("nvrtc"):
        # nvidia_cuda_nvrtc_cu12-12.8.93-py3-none-win_amd64.whl:
        #     nvidia\cuda_nvrtc\bin\
        #         nvrtc-builtins64_128.dll
        #         nvrtc64_120_0.alt.dll
        #         nvrtc64_120_0.dll
        return path_basename.endswith(".alt.dll") or "-builtins" in path_basename
    return path_basename.startswith(("cudart32_", "nvvm32"))
<|endoftext|>#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys
from collections.abc import Sequence

from cuda.pathfinder._dynamic_libs.lib_descriptor import LIB_DESCRIPTORS
from cuda.pathfinder._dynamic_libs.load_dl_common import DynamicLibNotFoundError, LoadedDL
from cuda.pathfinder._dynamic_libs.platform_loader import LOADER
from cuda.pathfinder._dynamic_libs.subprocess_protocol import (
    MODE_CANARY,
    MODE_LOAD,
    STATUS_NOT_FOUND,
    STATUS_OK,
    VALID_MODES,
    format_dynamic_lib_subprocess_payload,
)

# NOTE: The main entrypoint (below) serves both production (canary probe)
# and tests (full loader). Keeping them together ensures a single subprocess
# protocol and CLI surface, so the test subprocess stays aligned with the
# production flow while avoiding a separate test-only module.
# Any production-code impact is negligible since the extra logic only runs
# in the subprocess entrypoint and only in test mode.


def _probe_canary_abs_path(libname: str) -> str | None:
    desc = LIB_DESCRIPTORS.get(libname)
    if desc is None:
        raise ValueError(f"Unsupported canary library name: {libname!r}")
    try:
        loaded: LoadedDL | None = LOADER.load_with_system_search(desc)
    except DynamicLibNotFoundError:
        return None
    if loaded is None:
        return None
    abs_path: str | None = loaded.abs_path
    return abs_path


def _validate_abs_path(abs_path: str) -> None:
    assert abs_path, f"empty path: {abs_path=!r}"
    assert os.path.isabs(abs_path), f"not absolute: {abs_path=!r}"
    assert os.path.isfile(abs_path), f"not a file: {abs_path=!r}"


def _load_nvidia_dynamic_lib_for_test(libname: str) -> str:
    """Test-only loader used by the subprocess entrypoint."""
    # Keep imports inside the subprocess body so startup stays focused on the
    # code under test rather than the parent test module.
    from cuda.pathfinder import load_nvidia_dynamic_lib
    from cuda.pathfinder._dynamic_libs.load_nvidia_dynamic_lib import _load_lib_no_cache
    from cuda.pathfinder._dynamic_libs.supported_nvidia_libs import (
        SUPPORTED_LINUX_SONAMES,
        SUPPORTED_WINDOWS_DLLS,
    )
    from cuda.pathfinder._utils.platform_aware import IS_WINDOWS

    loaded_dl_fresh = load_nvidia_dynamic_lib(libname)
    if loaded_dl_fresh.was_already_loaded_from_elsewhere:
        raise RuntimeError("loaded_dl_fresh.was_already_loaded_from_elsewhere")

    abs_path = loaded_dl_fresh.abs_path
    if not isinstance(abs_path, str):
        raise RuntimeError(f"loaded_dl_fresh.abs_path is not a string: {abs_path!r}")
    _validate_abs_path(abs_path)
    assert loaded_dl_fresh.found_via is not None

    loaded_dl_from_cache = load_nvidia_dynamic_lib(libname)
    if loaded_dl_from_cache is not loaded_dl_fresh:
        raise RuntimeError("loaded_dl_from_cache is not loaded_dl_fresh")

    loaded_dl_no_cache = _load_lib_no_cache(libname)
    supported_libs = SUPPORTED_WINDOWS_DLLS if IS_WINDOWS else SUPPORTED_LINUX_SONAMES
    if not loaded_dl_no_cache.was_already_loaded_from_elsewhere and libname in supported_libs:
        raise RuntimeError("not loaded_dl_no_cache.was_already_loaded_from_elsewhere")
    abs_path_no_cache = loaded_dl_no_cache.abs_path
    if not isinstance(abs_path_no_cache, str):
        raise RuntimeError(f"loaded_dl_no_cache.abs_path is not a string: {abs_path_no_cache!r}")
    if not os.path.samefile(abs_path_no_cache, abs_path):
        raise RuntimeError(f"not os.path.samefile({abs_path_no_cache=!r}, {abs_path=!r})")
    _validate_abs_path(abs_path_no_cache)
    return abs_path


def probe_dynamic_lib_and_print_json(libname: str, mode: str) -> None:
    if mode == MODE_CANARY:
        abs_path = _probe_canary_abs_path(libname)
        status = STATUS_OK if abs_path is not None else STATUS_NOT_FOUND
        print(format_dynamic_lib_subprocess_payload(status, abs_path))
        return

    if mode == MODE_LOAD:
        # Test-only path: exercises full loader behavior in isolation.
        try:
            abs_path = _load_nvidia_dynamic_lib_for_test(libname)
        except DynamicLibNotFoundError as exc:
            error = {
                "type": exc.__class__.__name__,
                "message": str(exc),
            }
            print(format_dynamic_lib_subprocess_payload(STATUS_NOT_FOUND, None, error=error))
            return
        print(format_dynamic_lib_subprocess_payload(STATUS_OK, abs_path))
        return

    raise ValueError(f"Unsupported subprocess probe mode: {mode!r}")


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2 or args[0] not in VALID_MODES:
        modes = ", ".join(VALID_MODES)
        raise SystemExit(
            f"Usage: python -m cuda.pathfinder._dynamic_libs.dynamic_lib_subprocess <mode> <libname>\nModes: {modes}"
        )
    mode, libname = args
    probe_dynamic_lib_and_print_json(libname, mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
<|endoftext|># SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import glob
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from cuda.pathfinder._dynamic_libs.load_nvidia_dynamic_lib import (
    _resolve_system_loaded_abs_path_in_subprocess,
)
from cuda.pathfinder._dynamic_libs.search_steps import derive_ctk_root
from cuda.pathfinder._headers.header_descriptor import (
    HEADER_DESCRIPTORS,
    platform_include_subdirs,
    resolve_conda_anchor,
    system_install_dir_patterns,
)
from cuda.pathfinder._utils.ctk_root_canary import CTK_ROOT_CANARY_ANCHOR_LIBNAMES
from cuda.pathfinder._utils.env_vars import get_cuda_path_or_home
from cuda.pathfinder._utils.find_sub_dirs import find_sub_dirs_all_sitepackages
from cuda.pathfinder._utils.path_sort import numeric_aware_path_sort_key

if TYPE_CHECKING:
    from cuda.pathfinder._headers.header_descriptor import HeaderDescriptor

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class LocatedHeaderDir:
    abs_path: str | None
    found_via: str

    def __post_init__(self) -> None:
        self.abs_path = _abs_norm(self.abs_path)


#: Type alias for a header find step callable.
HeaderFindStep = Callable[["HeaderDescriptor"], LocatedHeaderDir | None]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _abs_norm(path: str | None) -> str | None:
    if path:
        return os.path.normpath(os.path.abspath(path))
    return None


def _joined_isfile(dirpath: str, basename: str) -> bool:
    return os.path.isfile(os.path.join(dirpath, basename))


def _locate_in_anchor_layout(desc: HeaderDescriptor, anchor_point: str) -> str | None:
    """Search for a header under *anchor_point* using the descriptor's layout fields."""
    h_basename = desc.header_basename
    for rel_dir in desc.anchor_include_rel_dirs:
        idir = os.path.join(anchor_point, rel_dir)
        for subdir in platform_include_subdirs(desc):
            cdir = os.path.join(idir, subdir)
            if _joined_isfile(cdir, h_basename):
                return cdir
        if _joined_isfile(idir, h_basename):
            return idir
    return None


# ---------------------------------------------------------------------------
# Find steps
# ---------------------------------------------------------------------------


def find_in_site_packages(desc: HeaderDescriptor) -> LocatedHeaderDir | None:
    """Search pip wheel install locations."""
    for sub_dir in desc.site_packages_dirs:
        hdr_dir: str  # help mypy
        for hdr_dir in find_sub_dirs_all_sitepackages(tuple(sub_dir.split("/"))):
            if _joined_isfile(hdr_dir, desc.header_basename):
                return LocatedHeaderDir(abs_path=hdr_dir, found_via="site-packages")
    return None


def find_in_conda(desc: HeaderDescriptor) -> LocatedHeaderDir | None:
    """Search ``$CONDA_PREFIX``."""
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return None
    anchor_point = resolve_conda_anchor(desc, conda_prefix)
    if anchor_point is None:
        return None
    found_header_path = _locate_in_anchor_layout(desc, anchor_point)
    if found_header_path:
        return LocatedHeaderDir(abs_path=found_header_path, found_via="conda")
    return None


def find_in_product_roots(desc: HeaderDescriptor) -> LocatedHeaderDir | None:
    """Search roots supplied through product-specific environment variables."""
    for env_var in desc.product_root_env_vars:
        root = os.environ.get(env_var)
        if not root:
            continue
        result = _locate_in_anchor_layout(desc, root)
        if result is not None:
            return LocatedHeaderDir(abs_path=result, found_via=env_var)
    return None


def find_in_cuda_path(desc: HeaderDescriptor) -> LocatedHeaderDir | None:
    """Search ``$CUDA_PATH`` / ``$CUDA_HOME``."""
    cuda_home = get_cuda_path_or_home()
    if cuda_home is None:
        return None
    result = _locate_in_anchor_layout(desc, cuda_home)
    if result is not None:
        return LocatedHeaderDir(abs_path=result, found_via="CUDA_PATH")
    return None


def find_via_ctk_root_canary(desc: HeaderDescriptor) -> LocatedHeaderDir | None:
    """Try CTK header lookup via CTK-root canary probing.

    Skips immediately if the descriptor does not opt in (``use_ctk_root_canary``).
    Otherwise, system-loads ``cudart`` in a fully isolated Python subprocess, derives
    CTK root from the resolved library path, and searches the expected include
    layout under that root.
    """
    if not desc.use_ctk_root_canary:
        return None
    canary_abs_path = _resolve_system_loaded_abs_path_in_subprocess(CTK_ROOT_CANARY_ANCHOR_LIBNAMES[0])
    if canary_abs_path is None:
        return None
    ctk_root = derive_ctk_root(canary_abs_path)
    if ctk_root is None:
        return None
    result = _locate_in_anchor_layout(desc, ctk_root)
    if result is not None:
        return LocatedHeaderDir(abs_path=result, found_via="system-ctk-root")
    return None


def find_in_system_install_dirs(desc: HeaderDescriptor) -> LocatedHeaderDir | None:
    """Search system install directories (glob patterns)."""
    for pattern in system_install_dir_patterns(desc):
        for hdr_dir in sorted(glob.glob(pattern), key=numeric_aware_path_sort_key, reverse=True):
            if _joined_isfile(hdr_dir, desc.header_basename):
                return LocatedHeaderDir(abs_path=hdr_dir, found_via="supported_install_dir")
    return None


# ---------------------------------------------------------------------------
# Step sequence and cascade runner
# ---------------------------------------------------------------------------

#: Unified find steps — each step self-gates based on descriptor fields.
FIND_STEPS: tuple[HeaderFindStep, ...] = (
    find_in_site_packages,
    find_in_conda,
    find_in_product_roots,
    find_in_cuda_path,
    find_via_ctk_root_canary,
    find_in_system_install_dirs,
)


def run_find_steps(desc: HeaderDescriptor, steps: tuple[HeaderFindStep, ...]) -> LocatedHeaderDir | None:
    """Run find steps in order, returning the first hit."""
    for step in steps:
        result = step(desc)
        if result is not None:
            return result
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@functools.cache
def locate_nvidia_header_directory(libname: str) -> LocatedHeaderDir | None:
    """Locate the header directory for a supported NVIDIA library.

    Args:
        libname (str): The short name of the library whose headers are needed
            (e.g., ``"nvrtc"``, ``"cusolver"``, ``"nvshmem"``).

    Returns:
        LocatedHeaderDir or None: A LocatedHeaderDir object containing the absolute path
        to the discovered header directory and information about where it was found,
        or ``None`` if the headers cannot be found.

    Raises:
        RuntimeError: If ``libname`` is not in the supported set.

    Search order:
        1. **NVIDIA Python wheels** — site-packages directories from the descriptor.
        2. **Conda environments** — platform-specific conda include layouts.
        3. **Product environment variables** — for example, ``CUDNN_PATH``.
        4. **CUDA Toolkit environment variables** — ``CUDA_PATH`` / ``CUDA_HOME``.
        5. **CTK root canary probe** — subprocess canary (descriptors with
           ``use_ctk_root_canary=True`` only).
        6. **System install directories** — glob patterns from the descriptor.
    """
    desc = HEADER_DESCRIPTORS.get(libname)
    if desc is None:
        raise RuntimeError(f"UNKNOWN {libname=}")
    return run_find_steps(desc, FIND_STEPS)


def find_nvidia_header_directory(libname: str) -> str | None:
    """Locate the header directory for a supported NVIDIA library.

    Args:
        libname (str): The short name of the library whose headers are needed
            (e.g., ``"nvrtc"``, ``"cusolver"``, ``"nvshmem"``).

    Returns:
        str or None: Absolute path to the discovered header directory, or ``None``
        if the headers cannot be found.

    Raises:
        RuntimeError: If ``libname`` is not in the supported set.

    Search order:
        1. **NVIDIA Python wheels** — site-packages directories from the descriptor.
        2. **Conda environments** — platform-specific conda include layouts.
        3. **Product environment variables** — for example, ``CUDNN_PATH``.
        4. **CUDA Toolkit environment variables** — ``CUDA_PATH`` / ``CUDA_HOME``.
        5. **CTK root canary probe** — subprocess canary (descriptors with
           ``use_ctk_root_canary=True`` only).
        6. **System install directories** — glob patterns from the descriptor.
    """
    found = locate_nvidia_header_directory(libname)
    return found.abs_path if found else None
<|endoftext|># SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-header descriptor, registry, and platform-aware accessors.

The canonical authored data lives in :mod:`header_descriptor_catalog`. This
module provides a name-keyed registry and platform-dispatch helpers consumed
by the runtime search path — keeping the search code itself platform-agnostic.
"""

from __future__ import annotations

import glob
import os
import sysconfig
from typing import TypeAlias, cast

from cuda.pathfinder._headers.header_descriptor_catalog import (
    HEADER_DESCRIPTOR_CATALOG,
    HeaderDescriptorSpec,
)
from cuda.pathfinder._utils.platform_aware import IS_WINDOWS

HeaderDescriptor: TypeAlias = HeaderDescriptorSpec

#: Canonical registry of all known header libraries.
HEADER_DESCRIPTORS: dict[str, HeaderDescriptor] = {desc.name: desc for desc in HEADER_DESCRIPTOR_CATALOG}


def platform_include_subdirs(desc: HeaderDescriptor) -> tuple[str, ...]:
    """Return the effective include subdirectory search list for the current platform.

    On Windows, Windows-specific subdirs are checked first, followed by the
    common subdirs.  On Linux, only the common subdirs are returned.
    """
    if IS_WINDOWS:
        return cast(tuple[str, ...], desc.include_subdirs_windows + desc.include_subdirs)
    return cast(tuple[str, ...], desc.include_subdirs)


