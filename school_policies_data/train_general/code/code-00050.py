        return [(midpoint(range1), midpoint(range2))]

    c11, c12 = _split_segment_at_t(curve1, 0.5)
    c11_range = (range1[0], midpoint(range1))
    c12_range = (midpoint(range1), range1[1])

    c21, c22 = _split_segment_at_t(curve2, 0.5)
    c21_range = (range2[0], midpoint(range2))
    c22_range = (midpoint(range2), range2[1])

    found = []
    found.extend(
        _curve_curve_intersections_t(
            c11, c21, precision, range1=c11_range, range2=c21_range
        )
    )
    found.extend(
        _curve_curve_intersections_t(
            c12, c21, precision, range1=c12_range, range2=c21_range
        )
    )
    found.extend(
        _curve_curve_intersections_t(
            c11, c22, precision, range1=c11_range, range2=c22_range
        )
    )
    found.extend(
        _curve_curve_intersections_t(
            c12, c22, precision, range1=c12_range, range2=c22_range
        )
    )

    unique_key = lambda ts: (int(ts[0] / precision), int(ts[1] / precision))
    seen = set()
    unique_values = []

    for ts in found:
        key = unique_key(ts)
        if key in seen:
            continue
        seen.add(key)
        unique_values.append(ts)

    return unique_values


def _is_linelike(segment):
    maybeline = _alignment_transformation(segment).transformPoints(segment)
    return all(math.isclose(p[1], 0.0) for p in maybeline)


def curveCurveIntersections(curve1, curve2):
    """Finds intersections between a curve and a curve.

    Args:
        curve1: List of coordinates of the first curve segment as 2D tuples.
        curve2: List of coordinates of the second curve segment as 2D tuples.

    Returns:
        A list of ``Intersection`` objects, each object having ``pt``, ``t1``
        and ``t2`` attributes containing the intersection point, time on first
        segment and time on second segment respectively.

    Examples::
        >>> curve1 = [ (10,100), (90,30), (40,140), (220,220) ]
        >>> curve2 = [ (5,150), (180,20), (80,250), (210,190) ]
        >>> intersections = curveCurveIntersections(curve1, curve2)
        >>> len(intersections)
        3
        >>> intersections[0].pt
        (81.7831487395506, 109.88904552375288)
    """
    if _is_linelike(curve1):
        line1 = curve1[0], curve1[-1]
        if _is_linelike(curve2):
            line2 = curve2[0], curve2[-1]
            return lineLineIntersections(*line1, *line2)
        else:
            hits = curveLineIntersections(curve2, line1)
            # curve is passed first to this fn but is the second segment, so
            # we need to swap t1/t2 in the result
            return [Intersection(pt=x.pt, t1=x.t2, t2=x.t1) for x in hits]
    elif _is_linelike(curve2):
        line2 = curve2[0], curve2[-1]
        return curveLineIntersections(curve1, line2)

    intersection_ts = _curve_curve_intersections_t(curve1, curve2)
    return [
        Intersection(pt=segmentPointAtT(curve1, ts[0]), t1=ts[0], t2=ts[1])
        for ts in intersection_ts
    ]


def segmentSegmentIntersections(seg1, seg2):
    """Finds intersections between two segments.

    Args:
        seg1: List of coordinates of the first segment as 2D tuples.
        seg2: List of coordinates of the second segment as 2D tuples.

    Returns:
        A list of ``Intersection`` objects, each object having ``pt``, ``t1``
        and ``t2`` attributes containing the intersection point, time on first
        segment and time on second segment respectively.

    Examples::
        >>> curve1 = [ (10,100), (90,30), (40,140), (220,220) ]
        >>> curve2 = [ (5,150), (180,20), (80,250), (210,190) ]
        >>> intersections = segmentSegmentIntersections(curve1, curve2)
        >>> len(intersections)
        3
        >>> intersections[0].pt
        (81.7831487395506, 109.88904552375288)
        >>> curve3 = [ (100, 240), (30, 60), (210, 230), (160, 30) ]
        >>> line  = [ (25, 260), (230, 20) ]
        >>> intersections = segmentSegmentIntersections(curve3, line)
        >>> len(intersections)
        3
        >>> intersections[0].pt
        (84.9000930760723, 189.87306176459828)

    """
    # Arrange by degree
    swapped = False
    if len(seg2) > len(seg1):
        seg2, seg1 = seg1, seg2
        swapped = True
    if len(seg1) > 2:
        if len(seg2) > 2:
            intersections = curveCurveIntersections(seg1, seg2)
        else:
            intersections = curveLineIntersections(seg1, seg2)
    elif len(seg1) == 2 and len(seg2) == 2:
        intersections = lineLineIntersections(*seg1, *seg2)
    else:
        raise ValueError("Couldn't work out which intersection function to use")
    if not swapped:
        return intersections
    return [Intersection(pt=i.pt, t1=i.t2, t2=i.t1) for i in intersections]


def _segmentrepr(obj):
    """
    >>> _segmentrepr([1, [2, 3], [], [[2, [3, 4], [0.1, 2.2]]]])
    '(1, (2, 3), (), ((2, (3, 4), (0.1, 2.2))))'
    """
    try:
        it = iter(obj)
    except TypeError:
        return "%g" % obj
    else:
        return "(%s)" % ", ".join(_segmentrepr(x) for x in it)


def printSegments(segments):
    """Helper for the doctests, displaying each segment in a list of
    segments on a single line as a tuple.
    """
    for segment in segments:
        print(_segmentrepr(segment))


if __name__ == "__main__":
    import sys
    import doctest

    sys.exit(doctest.testmod().failed)
<|endoftext|>from fontTools import ttLib
from fontTools.misc.textTools import safeEval
from fontTools.ttLib.tables.DefaultTable import DefaultTable
import sys
import os
import logging

log = logging.getLogger(__name__)


class TTXParseError(Exception):
    pass


BUFSIZE = 0x4000


class XMLReader(object):
    def __init__(
        self, fileOrPath, ttFont, progress=None, quiet=None, contentOnly=False
    ):
        if fileOrPath == "-":
            fileOrPath = sys.stdin
        if not hasattr(fileOrPath, "read"):
            self.file = open(fileOrPath, "rb")
            self._closeStream = True
        else:
            # assume readable file object
            self.file = fileOrPath
            self._closeStream = False
        self.ttFont = ttFont
        self.progress = progress
        if quiet is not None:
            from fontTools.misc.loggingTools import deprecateArgument

            deprecateArgument("quiet", "configure logging instead")
            self.quiet = quiet
        self.root = None
        self.contentStack = []
        self.contentOnly = contentOnly
        self.stackSize = 0

    def read(self, rootless=False):
        if rootless:
            self.stackSize += 1
        if self.progress:
            self.file.seek(0, 2)
            fileSize = self.file.tell()
            self.progress.set(0, fileSize // 100 or 1)
            self.file.seek(0)
        self._parseFile(self.file)
        if self._closeStream:
            self.close()
        if rootless:
            self.stackSize -= 1

    def close(self):
        self.file.close()

    def _parseFile(self, file):
        from xml.parsers.expat import ParserCreate

        parser = ParserCreate()
        parser.StartElementHandler = self._startElementHandler
        parser.EndElementHandler = self._endElementHandler
        parser.CharacterDataHandler = self._characterDataHandler

        pos = 0
        while True:
            chunk = file.read(BUFSIZE)
            if not chunk:
                parser.Parse(chunk, 1)
                break
            pos = pos + len(chunk)
            if self.progress:
                self.progress.set(pos // 100)
            parser.Parse(chunk, 0)

    def _startElementHandler(self, name, attrs):
        if self.stackSize == 1 and self.contentOnly:
            # We already know the table we're parsing, skip
            # parsing the table tag and continue to
            # stack '2' which begins parsing content
            self.contentStack.append([])
            self.stackSize = 2
            return
        stackSize = self.stackSize
        self.stackSize = stackSize + 1
        subFile = attrs.get("src")
        if subFile is not None:
            if hasattr(self.file, "name"):
                # if file has a name, get its parent directory
                dirname = os.path.dirname(self.file.name)
            else:
                # else fall back to using the current working directory
                dirname = os.getcwd()
            subFile = os.path.join(dirname, subFile)
        if not stackSize:
            if name != "ttFont":
                raise TTXParseError("illegal root tag: %s" % name)
            if self.ttFont.reader is None and not self.ttFont.tables:
                sfntVersion = attrs.get("sfntVersion")
                if sfntVersion is not None:
                    if len(sfntVersion) != 4:
                        sfntVersion = safeEval('"' + sfntVersion + '"')
                    self.ttFont.sfntVersion = sfntVersion
            self.contentStack.append([])
        elif stackSize == 1:
            if subFile is not None:
                subReader = XMLReader(subFile, self.ttFont, self.progress)
                subReader.read()
                self.contentStack.append([])
                return
            tag = ttLib.xmlToTag(name)
            msg = "Parsing '%s' table..." % tag
            if self.progress:
                self.progress.setLabel(msg)
            log.info(msg)
            if tag == "GlyphOrder":
                tableClass = ttLib.GlyphOrder
            elif "ERROR" in attrs or ("raw" in attrs and safeEval(attrs["raw"])):
                tableClass = DefaultTable
            else:
                tableClass = ttLib.getTableClass(tag)
                if tableClass is None:
                    tableClass = DefaultTable
            if tag == "loca" and tag in self.ttFont:
                # Special-case the 'loca' table as we need the
                #    original if the 'glyf' table isn't recompiled.
                self.currentTable = self.ttFont[tag]
            else:
                self.currentTable = tableClass(tag)
                self.ttFont[tag] = self.currentTable
            self.contentStack.append([])
        elif stackSize == 2 and subFile is not None:
            subReader = XMLReader(subFile, self.ttFont, self.progress, contentOnly=True)
            subReader.read()
            self.contentStack.append([])
            self.root = subReader.root
        elif stackSize == 2:
            self.contentStack.append([])
            self.root = (name, attrs, self.contentStack[-1])
        else:
            l = []
            self.contentStack[-1].append((name, attrs, l))
            self.contentStack.append(l)

    def _characterDataHandler(self, data):
        if self.stackSize > 1:
            # parser parses in chunks, so we may get multiple calls
            # for the same text node; thus we need to append the data
            # to the last item in the content stack:
            # https://github.com/fonttools/fonttools/issues/2614
            if (
                data != "\n"
                and self.contentStack[-1]
                and isinstance(self.contentStack[-1][-1], str)
                and self.contentStack[-1][-1] != "\n"
            ):
                self.contentStack[-1][-1] += data
            else:
                self.contentStack[-1].append(data)

    def _endElementHandler(self, name):
        self.stackSize = self.stackSize - 1
        del self.contentStack[-1]
        if not self.contentOnly:
            if self.stackSize == 1:
                self.root = None
            elif self.stackSize == 2:
                name, attrs, content = self.root
                self.currentTable.fromXML(name, attrs, content, self.ttFont)
                self.root = None


class ProgressPrinter(object):
    def __init__(self, title, maxval=100):
        print(title)

    def set(self, val, maxval=None):
        pass

    def increment(self, val=1):
        pass

    def setLabel(self, text):
        print(text)
<|endoftext|>from collections import UserDict, UserList

__all__ = ["LazyDict", "LazyList"]


class LazyDict(UserDict):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, k):
        v = self.data[k]
        if callable(v):
            v = v(k)
            self.data[k] = v
        return v


class LazyList(UserList):
    def __getitem__(self, k):
        if isinstance(k, slice):
            indices = range(*k.indices(len(self)))
            return [self[i] for i in indices]
        v = self.data[k]
        if callable(v):
            v = v(k)
            self.data[k] = v
        return v

    def __add__(self, other):
        if isinstance(other, LazyList):
            other = list(other)
        elif isinstance(other, list):
            pass
        else:
            return NotImplemented
        return list(self) + other

    def __radd__(self, other):
        if not isinstance(other, list):
            return NotImplemented
        return other + list(self)
<|endoftext|>"""fontTools.misc.encodingTools.py -- tools for working with OpenType encodings."""

import fontTools.encodings.codecs

# Map keyed by platformID, then platEncID, then possibly langID
_encodingMap = {
    0: {  # Unicode
        0: "utf_16_be",
        1: "utf_16_be",
        2: "utf_16_be",
        3: "utf_16_be",
        4: "utf_16_be",
        5: "utf_16_be",
        6: "utf_16_be",
    },
    1: {  # Macintosh
        # See
        # https://github.com/fonttools/fonttools/issues/236
        0: {  # Macintosh, platEncID==0, keyed by langID
            15: "mac_iceland",
            17: "mac_turkish",
            18: "mac_croatian",
            24: "mac_latin2",
            25: "mac_latin2",
            26: "mac_latin2",
            27: "mac_latin2",
            28: "mac_latin2",
            36: "mac_latin2",
            37: "mac_romanian",
            38: "mac_latin2",
            39: "mac_latin2",
            40: "mac_latin2",
            Ellipsis: "mac_roman",  # Other
        },
        1: "x_mac_japanese_ttx",
        2: "x_mac_trad_chinese_ttx",
        3: "x_mac_korean_ttx",
        6: "mac_greek",
        7: "mac_cyrillic",
        25: "x_mac_simp_chinese_ttx",
        29: "mac_latin2",
        35: "mac_turkish",
        37: "mac_iceland",
    },
    2: {  # ISO
        0: "ascii",
        1: "utf_16_be",
        2: "latin1",
    },
    3: {  # Microsoft
        0: "utf_16_be",
        1: "utf_16_be",
        2: "shift_jis",
        3: "gb2312",
        4: "big5",
        5: "euc_kr",
        6: "johab",
        10: "utf_16_be",
    },
}


def getEncoding(platformID, platEncID, langID, default=None):
    """Returns the Python encoding name for OpenType platformID/encodingID/langID
    triplet.  If encoding for these values is not known, by default None is
    returned.  That can be overriden by passing a value to the default argument.
    """
    encoding = _encodingMap.get(platformID, {}).get(platEncID, default)
    if isinstance(encoding, dict):
        encoding = encoding.get(langID, encoding[Ellipsis])
    return encoding
<|endoftext|>"""fontTools.misc.timeTools.py -- tools for working with OpenType timestamps."""

import os
import time
from datetime import datetime, timezone
import calendar

epoch_diff = calendar.timegm((1904, 1, 1, 0, 0, 0, 0, 0, 0))

DAYNAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTHNAMES = [
    None,
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]


def asctime(t=None):
    """
    Convert a tuple or struct_time representing a time as returned by gmtime()
    or localtime() to a 24-character string of the following form:

    >>> asctime(time.gmtime(0))
    'Thu Jan  1 00:00:00 1970'

    If t is not provided, the current time as returned by localtime() is used.
    Locale information is not used by asctime().

    This is meant to normalise the output of the built-in time.asctime() across
    different platforms and Python versions.
    In Python 3.x, the day of the month is right-justified, whereas on Windows
    Python 2.7 it is padded with zeros.

    See https://github.com/fonttools/fonttools/issues/455
    """
    if t is None:
        t = time.localtime()
    s = "%s %s %2s %s" % (
        DAYNAMES[t.tm_wday],
        MONTHNAMES[t.tm_mon],
        t.tm_mday,
        time.strftime("%H:%M:%S %Y", t),
    )
    return s


def timestampToString(value):
    return asctime(time.gmtime(max(0, value + epoch_diff)))


def timestampFromString(value):
    wkday, mnth = value[:7].split()
    t = datetime.strptime(value[7:], " %d %H:%M:%S %Y")
    t = t.replace(month=MONTHNAMES.index(mnth), tzinfo=timezone.utc)
    wkday_idx = DAYNAMES.index(wkday)
    assert t.weekday() == wkday_idx, '"' + value + '" has inconsistent weekday'
    return int(t.timestamp()) - epoch_diff


def timestampNow():
    # https://reproducible-builds.org/specs/source-date-epoch/
    source_date_epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if source_date_epoch is not None:
        return int(source_date_epoch) - epoch_diff
    return int(time.time() - epoch_diff)


def timestampSinceEpoch(value):
    return int(value - epoch_diff)


if __name__ == "__main__":
    import sys
    import doctest

    sys.exit(doctest.testmod().failed)
<|endoftext|>"""
This module implements the algorithm for converting between a "user name" -
something that a user can choose arbitrarily inside a font editor - and a file
name suitable for use in a wide range of operating systems and filesystems.

The `UFO 3 specification <http://unifiedfontobject.org/versions/ufo3/conventions/>`_
provides an example of an algorithm for such conversion, which avoids illegal
characters, reserved file names, ambiguity between upper- and lower-case
characters, and clashes with existing files.

This code was originally copied from
`ufoLib <https://github.com/unified-font-object/ufoLib/blob/8747da7/Lib/ufoLib/filenames.py>`_
by Tal Leming and is copyright (c) 2005-2016, The RoboFab Developers:

-	Erik van Blokland
-	Tal Leming
-	Just van Rossum
"""

illegalCharacters = r"\" * + / : < > ? [ \ ] | \0".split(" ")
illegalCharacters += [chr(i) for i in range(1, 32)]
illegalCharacters += [chr(0x7F)]
reservedFileNames = "CON PRN AUX CLOCK$ NUL A:-Z: COM1".lower().split(" ")
reservedFileNames += "LPT1 LPT2 LPT3 COM2 COM3 COM4".lower().split(" ")
maxFileNameLength = 255


class NameTranslationError(Exception):
    pass


def userNameToFileName(userName, existing=[], prefix="", suffix=""):
    """Converts from a user name to a file name.

    Takes care to avoid illegal characters, reserved file names, ambiguity between
    upper- and lower-case characters, and clashes with existing files.

    Args:
            userName (str): The input file name.
            existing: A case-insensitive list of all existing file names.
            prefix: Prefix to be prepended to the file name.
            suffix: Suffix to be appended to the file name.

    Returns:
            A suitable filename.

    Raises:
            NameTranslationError: If no suitable name could be generated.

    Examples::

            >>> userNameToFileName("a") == "a"
            True
            >>> userNameToFileName("A") == "A_"
            True
            >>> userNameToFileName("AE") == "A_E_"
            True
            >>> userNameToFileName("Ae") == "A_e"
            True
            >>> userNameToFileName("ae") == "ae"
            True
            >>> userNameToFileName("aE") == "aE_"
            True
            >>> userNameToFileName("a.alt") == "a.alt"
            True
            >>> userNameToFileName("A.alt") == "A_.alt"
            True
            >>> userNameToFileName("A.Alt") == "A_.A_lt"
            True
            >>> userNameToFileName("A.aLt") == "A_.aL_t"
            True
            >>> userNameToFileName(u"A.alT") == "A_.alT_"
            True
            >>> userNameToFileName("T_H") == "T__H_"
            True
            >>> userNameToFileName("T_h") == "T__h"
            True
            >>> userNameToFileName("t_h") == "t_h"
            True
            >>> userNameToFileName("F_F_I") == "F__F__I_"
            True
            >>> userNameToFileName("f_f_i") == "f_f_i"
            True
            >>> userNameToFileName("Aacute_V.swash") == "A_acute_V_.swash"
            True
            >>> userNameToFileName(".notdef") == "_notdef"
            True
            >>> userNameToFileName("con") == "_con"
            True
            >>> userNameToFileName("CON") == "C_O_N_"
            True
            >>> userNameToFileName("con.alt") == "_con.alt"
            True
            >>> userNameToFileName("alt.con") == "alt._con"
            True
    """
    # the incoming name must be a str
    if not isinstance(userName, str):
        raise ValueError("The value for userName must be a string.")
    # establish the prefix and suffix lengths
    prefixLength = len(prefix)
    suffixLength = len(suffix)
    # replace an initial period with an _
    # if no prefix is to be added
    if not prefix and userName[0] == ".":
        userName = "_" + userName[1:]
    # filter the user name
    filteredUserName = []
    for character in userName:
        # replace illegal characters with _
        if character in illegalCharacters:
            character = "_"
        # add _ to all non-lower characters
        elif character != character.lower():
            character += "_"
        filteredUserName.append(character)
    userName = "".join(filteredUserName)
    # clip to 255
    sliceLength = maxFileNameLength - prefixLength - suffixLength
    userName = userName[:sliceLength]
    # test for illegal files names
    parts = []
    for part in userName.split("."):
        if part.lower() in reservedFileNames:
            part = "_" + part
        parts.append(part)
    userName = ".".join(parts)
    # test for clash
    fullName = prefix + userName + suffix
    if fullName.lower() in existing:
        fullName = handleClash1(userName, existing, prefix, suffix)
    # finished
    return fullName


def handleClash1(userName, existing=[], prefix="", suffix=""):
    """
    existing should be a case-insensitive list
    of all existing file names.

    >>> prefix = ("0" * 5) + "."
    >>> suffix = "." + ("0" * 10)
    >>> existing = ["a" * 5]

    >>> e = list(existing)
    >>> handleClash1(userName="A" * 5, existing=e,
    ...		prefix=prefix, suffix=suffix) == (
    ... 	'00000.AAAAA000000000000001.0000000000')
    True

    >>> e = list(existing)
    >>> e.append(prefix + "aaaaa" + "1".zfill(15) + suffix)
    >>> handleClash1(userName="A" * 5, existing=e,
    ...		prefix=prefix, suffix=suffix) == (
    ... 	'00000.AAAAA000000000000002.0000000000')
    True

    >>> e = list(existing)
    >>> e.append(prefix + "AAAAA" + "2".zfill(15) + suffix)
    >>> handleClash1(userName="A" * 5, existing=e,
    ...		prefix=prefix, suffix=suffix) == (
    ... 	'00000.AAAAA000000000000001.0000000000')
    True
    """
    # if the prefix length + user name length + suffix length + 15 is at
    # or past the maximum length, silce 15 characters off of the user name
    prefixLength = len(prefix)
    suffixLength = len(suffix)
    if prefixLength + len(userName) + suffixLength + 15 > maxFileNameLength:
        l = prefixLength + len(userName) + suffixLength + 15
        sliceLength = maxFileNameLength - l
        userName = userName[:sliceLength]
    finalName = None
    # try to add numbers to create a unique name
    counter = 1
    while finalName is None:
        name = userName + str(counter).zfill(15)
        fullName = prefix + name + suffix
        if fullName.lower() not in existing:
            finalName = fullName
            break
        else:
            counter += 1
        if counter >= 999999999999999:
            break
    # if there is a clash, go to the next fallback
    if finalName is None:
        finalName = handleClash2(existing, prefix, suffix)
    # finished
    return finalName


def handleClash2(existing=[], prefix="", suffix=""):
    """
    existing should be a case-insensitive list
    of all existing file names.

    >>> prefix = ("0" * 5) + "."
    >>> suffix = "." + ("0" * 10)
    >>> existing = [prefix + str(i) + suffix for i in range(100)]

    >>> e = list(existing)
    >>> handleClash2(existing=e, prefix=prefix, suffix=suffix) == (
    ... 	'00000.100.0000000000')
    True

    >>> e = list(existing)
    >>> e.remove(prefix + "1" + suffix)
    >>> handleClash2(existing=e, prefix=prefix, suffix=suffix) == (
    ... 	'00000.1.0000000000')
    True

    >>> e = list(existing)
    >>> e.remove(prefix + "2" + suffix)
    >>> handleClash2(existing=e, prefix=prefix, suffix=suffix) == (
    ... 	'00000.2.0000000000')
    True
    """
    # calculate the longest possible string
    maxLength = maxFileNameLength - len(prefix) - len(suffix)
    maxValue = int("9" * maxLength)
    # try to find a number
    finalName = None
    counter = 1
    while finalName is None:
        fullName = prefix + str(counter) + suffix
        if fullName.lower() not in existing:
            finalName = fullName
            break
        else:
            counter += 1
        if counter >= maxValue:
            break
    # raise an error if nothing has been found
    if finalName is None:
        raise NameTranslationError("No unique name could be found.")
    # finished
    return finalName


if __name__ == "__main__":
    import doctest
    import sys

    sys.exit(doctest.testmod().failed)
<|endoftext|>"""
Code of the config system; not related to fontTools or fonts in particular.

The options that are specific to fontTools are in :mod:`fontTools.config`.

To create your own config system, you need to create an instance of
:class:`Options`, and a subclass of :class:`AbstractConfig` with its
``options`` class variable set to your instance of Options.

"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterable,
    Mapping,
    MutableMapping,
    Optional,
    Set,
    Union,
)

log = logging.getLogger(__name__)

__all__ = [
    "AbstractConfig",
    "ConfigAlreadyRegisteredError",
    "ConfigError",
    "ConfigUnknownOptionError",
    "ConfigValueParsingError",
    "ConfigValueValidationError",
    "Option",
    "Options",
]


class ConfigError(Exception):
    """Base exception for the config module."""


class ConfigAlreadyRegisteredError(ConfigError):
    """Raised when a module tries to register a configuration option that
    already exists.

    Should not be raised too much really, only when developing new fontTools
    modules.
    """

    def __init__(self, name):
        super().__init__(f"Config option {name} is already registered.")


class ConfigValueParsingError(ConfigError):
    """Raised when a configuration value cannot be parsed."""

    def __init__(self, name, value):
        super().__init__(
            f"Config option {name}: value cannot be parsed (given {repr(value)})"
        )


class ConfigValueValidationError(ConfigError):
    """Raised when a configuration value cannot be validated."""

    def __init__(self, name, value):
        super().__init__(
            f"Config option {name}: value is invalid (given {repr(value)})"
        )


class ConfigUnknownOptionError(ConfigError):
    """Raised when a configuration option is unknown."""

    def __init__(self, option_or_name):
        name = (
            f"'{option_or_name.name}' (id={id(option_or_name)})>"
            if isinstance(option_or_name, Option)
            else f"'{option_or_name}'"
        )
        super().__init__(f"Config option {name} is unknown")


# eq=False because Options are unique, not fungible objects
@dataclass(frozen=True, eq=False)
class Option:
    name: str
    """Unique name identifying the option (e.g. package.module:MY_OPTION)."""
    help: str
    """Help text for this option."""
    default: Any
    """Default value for this option."""
    parse: Callable[[str], Any]
    """Turn input (e.g. string) into proper type. Only when reading from file."""
    validate: Optional[Callable[[Any], bool]] = None
    """Return true if the given value is an acceptable value."""

    @staticmethod
    def parse_optional_bool(v: str) -> Optional[bool]:
        s = str(v).lower()
        if s in {"0", "no", "false"}:
            return False
        if s in {"1", "yes", "true"}:
            return True
        if s in {"auto", "none"}:
            return None
        raise ValueError("invalid optional bool: {v!r}")

    @staticmethod
    def validate_optional_bool(v: Any) -> bool:
        return v is None or isinstance(v, bool)


class Options(Mapping):
    """Registry of available options for a given config system.

    Define new options using the :meth:`register()` method.

    Access existing options using the Mapping interface.
    """

    __options: Dict[str, Option]

    def __init__(self, other: "Options" = None) -> None:
        self.__options = {}
        if other is not None:
            for option in other.values():
                self.register_option(option)

    def register(
        self,
        name: str,
        help: str,
        default: Any,
        parse: Callable[[str], Any],
        validate: Optional[Callable[[Any], bool]] = None,
    ) -> Option:
        """Create and register a new option."""
        return self.register_option(Option(name, help, default, parse, validate))

    def register_option(self, option: Option) -> Option:
        """Register a new option."""
        name = option.name
        if name in self.__options:
            raise ConfigAlreadyRegisteredError(name)
        self.__options[name] = option
        return option

    def is_registered(self, option: Option) -> bool:
        """Return True if the same option object is already registered."""
        return self.__options.get(option.name) is option

    def __getitem__(self, key: str) -> Option:
        return self.__options.__getitem__(key)

    def __iter__(self) -> Iterator[str]:
        return self.__options.__iter__()

    def __len__(self) -> int:
        return self.__options.__len__()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({{\n"
            + "".join(
                f"    {k!r}: Option(default={v.default!r}, ...),\n"
                for k, v in self.__options.items()
            )
            + "})"
        )


_USE_GLOBAL_DEFAULT = object()


class AbstractConfig(MutableMapping):
    """
    Create a set of config values, optionally pre-filled with values from
    the given dictionary or pre-existing config object.

    The class implements the MutableMapping protocol keyed by option name (`str`).
    For convenience its methods accept either Option or str as the key parameter.

    .. seealso:: :meth:`set()`

    This config class is abstract because it needs its ``options`` class
    var to be set to an instance of :class:`Options` before it can be
    instanciated and used.

    .. code:: python

        class MyConfig(AbstractConfig):
            options = Options()

        MyConfig.register_option( "test:option_name", "This is an option", 0, int, lambda v: isinstance(v, int))

        cfg = MyConfig({"test:option_name": 10})

    """

    options: ClassVar[Options]

    @classmethod
    def register_option(
        cls,
        name: str,
        help: str,
        default: Any,
        parse: Callable[[str], Any],
        validate: Optional[Callable[[Any], bool]] = None,
    ) -> Option:
        """Register an available option in this config system."""
        return cls.options.register(
            name, help=help, default=default, parse=parse, validate=validate
        )

    _values: Dict[str, Any]

    def __init__(
        self,
        values: Union[
            AbstractConfig, Dict[Union[Option, str], Any], Mapping[str, Any]
        ] = {},
        parse_values: bool = False,
        skip_unknown: bool = False,
    ):
        self._values = {}
        values_dict = values._values if isinstance(values, AbstractConfig) else values
        for name, value in values_dict.items():
            self.set(name, value, parse_values, skip_unknown)

    def _resolve_option(self, option_or_name: Union[Option, str]) -> Option:
        if isinstance(option_or_name, Option):
            option = option_or_name
            if not self.options.is_registered(option):
                raise ConfigUnknownOptionError(option)
            return option
        elif isinstance(option_or_name, str):
            name = option_or_name
            try:
                return self.options[name]
            except KeyError:
                raise ConfigUnknownOptionError(name)
        else:
            raise TypeError(
                "expected Option or str, found "
                f"{type(option_or_name).__name__}: {option_or_name!r}"
            )

    def set(
        self,
        option_or_name: Union[Option, str],
        value: Any,
        parse_values: bool = False,
        skip_unknown: bool = False,
    ):
        """Set the value of an option.

        Args:
            * `option_or_name`: an `Option` object or its name (`str`).
            * `value`: the value to be assigned to given option.
            * `parse_values`: parse the configuration value from a string into
                its proper type, as per its `Option` object. The default
                behavior is to raise `ConfigValueValidationError` when the value
                is not of the right type. Useful when reading options from a
                file type that doesn't support as many types as Python.
            * `skip_unknown`: skip unknown configuration options. The default
                behaviour is to raise `ConfigUnknownOptionError`. Useful when
                reading options from a configuration file that has extra entries
                (e.g. for a later version of fontTools)
        """
        try:
            option = self._resolve_option(option_or_name)
        except ConfigUnknownOptionError as e:
            if skip_unknown:
                log.debug(str(e))
                return
            raise

        # Can be useful if the values come from a source that doesn't have
        # strict typing (.ini file? Terminal input?)
        if parse_values:
            try:
                value = option.parse(value)
            except Exception as e:
                raise ConfigValueParsingError(option.name, value) from e

        if option.validate is not None and not option.validate(value):
            raise ConfigValueValidationError(option.name, value)

        self._values[option.name] = value

    def get(
        self, option_or_name: Union[Option, str], default: Any = _USE_GLOBAL_DEFAULT
    ) -> Any:
        """
        Get the value of an option. The value which is returned is the first
        provided among:

        1. a user-provided value in the options's ``self._values`` dict
        2. a caller-provided default value to this method call
        3. the global default for the option provided in ``fontTools.config``

        This is to provide the ability to migrate progressively from config
        options passed as arguments to fontTools APIs to config options read
        from the current TTFont, e.g.

        .. code:: python

            def fontToolsAPI(font, some_option):
                value = font.cfg.get("someLib.module:SOME_OPTION", some_option)
                # use value

        That way, the function will work the same for users of the API that
        still pass the option to the function call, but will favour the new
        config mechanism if the given font specifies a value for that option.
        """
        option = self._resolve_option(option_or_name)
        if option.name in self._values:
            return self._values[option.name]
        if default is not _USE_GLOBAL_DEFAULT:
            return default
        return option.default

    def copy(self):
        return self.__class__(self._values)

    def __getitem__(self, option_or_name: Union[Option, str]) -> Any:
        return self.get(option_or_name)

    def __setitem__(self, option_or_name: Union[Option, str], value: Any) -> None:
        return self.set(option_or_name, value)

    def __delitem__(self, option_or_name: Union[Option, str]) -> None:
        option = self._resolve_option(option_or_name)
        del self._values[option.name]

    def __iter__(self) -> Iterable[str]:
        return self._values.__iter__()

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self._values)})"
<|endoftext|>from fontTools.misc.textTools import Tag, bytesjoin, strjoin

try:
    import xattr
except ImportError:
    xattr = None


def _reverseString(s):
    s = list(s)
    s.reverse()
    return strjoin(s)


def getMacCreatorAndType(path):
    """Returns file creator and file type codes for a path.

    Args:
            path (str): A file path.

    Returns:
            A tuple of two :py:class:`fontTools.textTools.Tag` objects, the first
            representing the file creator and the second representing the
            file type.
    """
    if xattr is not None:
        try:
            finderInfo = xattr.getxattr(path, "com.apple.FinderInfo")
        except (KeyError, IOError):
            pass
        else:
            fileType = Tag(finderInfo[:4])
            fileCreator = Tag(finderInfo[4:8])
            return fileCreator, fileType
    return None, None


def setMacCreatorAndType(path, fileCreator, fileType):
    """Set file creator and file type codes for a path.

    Note that if the ``xattr`` module is not installed, no action is
    taken but no error is raised.

    Args:
            path (str): A file path.
            fileCreator: A four-character file creator tag.
            fileType: A four-character file type tag.

    """
    if xattr is not None:
        from fontTools.misc.textTools import pad

        if not all(len(s) == 4 for s in (fileCreator, fileType)):
            raise TypeError("arg must be string of 4 chars")
        finderInfo = pad(bytesjoin([fileType, fileCreator]), 32)
        xattr.setxattr(path, "com.apple.FinderInfo", finderInfo)
<|endoftext|>from fontTools.pens.basePen import BasePen
from functools import partial
from itertools import count
import sympy as sp
import sys

n = 3  # Max Bezier degree; 3 for cubic, 2 for quadratic

t, x, y = sp.symbols("t x y", real=True)
c = sp.symbols("c", real=False)  # Complex representation instead of x/y

X = tuple(sp.symbols("x:%d" % (n + 1), real=True))
Y = tuple(sp.symbols("y:%d" % (n + 1), real=True))
P = tuple(zip(*(sp.symbols("p:%d[%s]" % (n + 1, w), real=True) for w in "01")))
C = tuple(sp.symbols("c:%d" % (n + 1), real=False))

# Cubic Bernstein basis functions
BinomialCoefficient = [(1, 0)]
for i in range(1, n + 1):
    last = BinomialCoefficient[-1]
    this = tuple(last[j - 1] + last[j] for j in range(len(last))) + (0,)
    BinomialCoefficient.append(this)
BinomialCoefficient = tuple(tuple(item[:-1]) for item in BinomialCoefficient)
del last, this

BernsteinPolynomial = tuple(
    tuple(c * t**i * (1 - t) ** (n - i) for i, c in enumerate(coeffs))
    for n, coeffs in enumerate(BinomialCoefficient)
)

BezierCurve = tuple(
    tuple(
        sum(P[i][j] * bernstein for i, bernstein in enumerate(bernsteins))
        for j in range(2)
    )
    for n, bernsteins in enumerate(BernsteinPolynomial)
)
BezierCurveC = tuple(
    sum(C[i] * bernstein for i, bernstein in enumerate(bernsteins))
    for n, bernsteins in enumerate(BernsteinPolynomial)
)


def green(f, curveXY):
    f = -sp.integrate(sp.sympify(f), y)
    f = f.subs({x: curveXY[0], y: curveXY[1]})
    f = sp.integrate(f * sp.diff(curveXY[0], t), (t, 0, 1))
    return f


class _BezierFuncsLazy(dict):
    def __init__(self, symfunc):
        self._symfunc = symfunc
        self._bezfuncs = {}

    def __missing__(self, i):
        args = ["p%d" % d for d in range(i + 1)]
        f = green(self._symfunc, BezierCurve[i])
        f = sp.gcd_terms(f.collect(sum(P, ())))  # Optimize
        return sp.lambdify(args, f)


class GreenPen(BasePen):
    _BezierFuncs = {}

    @classmethod
    def _getGreenBezierFuncs(celf, func):
        funcstr = str(func)
        if not funcstr in celf._BezierFuncs:
            celf._BezierFuncs[funcstr] = _BezierFuncsLazy(func)
        return celf._BezierFuncs[funcstr]

    def __init__(self, func, glyphset=None):
        BasePen.__init__(self, glyphset)
        self._funcs = self._getGreenBezierFuncs(func)
        self.value = 0

    def _moveTo(self, p0):
        self._startPoint = p0

    def _closePath(self):
        p0 = self._getCurrentPoint()
        if p0 != self._startPoint:
            self._lineTo(self._startPoint)

    def _endPath(self):
        p0 = self._getCurrentPoint()
        if p0 != self._startPoint:
            # Green theorem is not defined on open contours.
            raise NotImplementedError

    def _lineTo(self, p1):
        p0 = self._getCurrentPoint()
        self.value += self._funcs[1](p0, p1)

    def _qCurveToOne(self, p1, p2):
        p0 = self._getCurrentPoint()
        self.value += self._funcs[2](p0, p1, p2)

    def _curveToOne(self, p1, p2, p3):
        p0 = self._getCurrentPoint()
        self.value += self._funcs[3](p0, p1, p2, p3)


# Sample pens.
# Do not use this in real code.
# Use fontTools.pens.momentsPen.MomentsPen instead.
AreaPen = partial(GreenPen, func=1)
MomentXPen = partial(GreenPen, func=x)
MomentYPen = partial(GreenPen, func=y)
MomentXXPen = partial(GreenPen, func=x * x)
MomentYYPen = partial(GreenPen, func=y * y)
MomentXYPen = partial(GreenPen, func=x * y)


def printGreenPen(penName, funcs, file=sys.stdout, docstring=None):
    if docstring is not None:
        print('"""%s"""' % docstring)

    print(
        """from fontTools.pens.basePen import BasePen, OpenContourError
try:
	import cython
except (AttributeError, ImportError):
	# if cython not installed, use mock module with no-op decorators and types
	from fontTools.misc import cython
COMPILED = cython.compiled


__all__ = ["%s"]

class %s(BasePen):

	def __init__(self, glyphset=None):
		BasePen.__init__(self, glyphset)
""" % (penName, penName),
        file=file,
    )
    for name, f in funcs:
        print("		self.%s = 0" % name, file=file)
    print(
        """
	def _moveTo(self, p0):
		self._startPoint = p0

	def _closePath(self):
		p0 = self._getCurrentPoint()
		if p0 != self._startPoint:
			self._lineTo(self._startPoint)

	def _endPath(self):
		p0 = self._getCurrentPoint()
		if p0 != self._startPoint:
			raise OpenContourError(
							"Glyph statistics is not defined on open contours."
			)
""",
        end="",
        file=file,
    )

    for n in (1, 2, 3):
        subs = {P[i][j]: [X, Y][j][i] for i in range(n + 1) for j in range(2)}
        greens = [green(f, BezierCurve[n]) for name, f in funcs]
        greens = [sp.gcd_terms(f.collect(sum(P, ()))) for f in greens]  # Optimize
        greens = [f.subs(subs) for f in greens]  # Convert to p to x/y
        defs, exprs = sp.cse(
            greens,
            optimizations="basic",
            symbols=(sp.Symbol("r%d" % i) for i in count()),
        )

        print()
        for name, value in defs:
            print("	@cython.locals(%s=cython.double)" % name, file=file)
        if n == 1:
            print(
                """\
	@cython.locals(x0=cython.double, y0=cython.double)
	@cython.locals(x1=cython.double, y1=cython.double)
	def _lineTo(self, p1):
		x0,y0 = self._getCurrentPoint()
		x1,y1 = p1
""",
                file=file,
            )
        elif n == 2:
            print(
                """\
	@cython.locals(x0=cython.double, y0=cython.double)
	@cython.locals(x1=cython.double, y1=cython.double)
	@cython.locals(x2=cython.double, y2=cython.double)
	def _qCurveToOne(self, p1, p2):
		x0,y0 = self._getCurrentPoint()
		x1,y1 = p1
		x2,y2 = p2
""",
                file=file,
            )
        elif n == 3:
            print(
                """\
	@cython.locals(x0=cython.double, y0=cython.double)
	@cython.locals(x1=cython.double, y1=cython.double)
	@cython.locals(x2=cython.double, y2=cython.double)
	@cython.locals(x3=cython.double, y3=cython.double)
	def _curveToOne(self, p1, p2, p3):
		x0,y0 = self._getCurrentPoint()
		x1,y1 = p1
		x2,y2 = p2
		x3,y3 = p3
""",
                file=file,
            )
        for name, value in defs:
            print("		%s = %s" % (name, value), file=file)

        print(file=file)
        for name, value in zip([f[0] for f in funcs], exprs):
            print("		self.%s += %s" % (name, value), file=file)

    print(
        """
if __name__ == '__main__':
	from fontTools.misc.symfont import x, y, printGreenPen
	printGreenPen('%s', [""" % penName,
        file=file,
    )
    for name, f in funcs:
        print("		      ('%s', %s)," % (name, str(f)), file=file)
    print("		     ])", file=file)


if __name__ == "__main__":
    import sys

    if sys.argv[1:]:
        penName = sys.argv[1]
        funcs = [(name, eval(f)) for name, f in zip(sys.argv[2::2], sys.argv[3::2])]
        printGreenPen(penName, funcs, file=sys.stdout)
<|endoftext|>"""sstruct.py -- SuperStruct

Higher level layer on top of the struct module, enabling to
bind names to struct elements. The interface is similar to
struct, except the objects passed and returned are not tuples
(or argument lists), but dictionaries or instances.

Just like struct, we use fmt strings to describe a data
structure, except we use one line per element. Lines are
separated by newlines or semi-colons. Each line contains
either one of the special struct characters ('@', '=', '<',
'>' or '!') or a 'name:formatchar' combo (eg. 'myFloat:f').
Repetitions, like the struct module offers them are not useful
in this context, except for fixed length strings  (eg. 'myInt:5h'
is not allowed but 'myString:5s' is). The 'x' fmt character
(pad byte) is treated as 'special', since it is by definition
anonymous. Extra whitespace is allowed everywhere.

The sstruct module offers one feature that the "normal" struct
module doesn't: support for fixed point numbers. These are spelled
as "n.mF", where n is the number of bits before the point, and m
the number of bits after the point. Fixed point numbers get
converted to floats.

pack(fmt, object):
        'object' is either a dictionary or an instance (or actually
        anything that has a __dict__ attribute). If it is a dictionary,
        its keys are used for names. If it is an instance, it's
        attributes are used to grab struct elements from. Returns
        a string containing the data.

unpack(fmt, data, object=None)
        If 'object' is omitted (or None), a new dictionary will be
        returned. If 'object' is a dictionary, it will be used to add
        struct elements to. If it is an instance (or in fact anything
        that has a __dict__ attribute), an attribute will be added for
        each struct element. In the latter two cases, 'object' itself
        is returned.

unpack2(fmt, data, object=None)
        Convenience function. Same as unpack, except data may be longer
        than needed. The returned value is a tuple: (object, leftoverdata).

calcsize(fmt)
        like struct.calcsize(), but uses our own fmt strings:
        it returns the size of the data in bytes.
"""

from fontTools.misc.fixedTools import fixedToFloat as fi2fl, floatToFixed as fl2fi
from fontTools.misc.textTools import tobytes, tostr
import struct
import re

__version__ = "1.2"
__copyright__ = "Copyright 1998, Just van Rossum <just@letterror.com>"


class Error(Exception):
    pass


def pack(fmt, obj):
    formatstring, names, fixes = getformat(fmt, keep_pad_byte=True)
    elements = []
    if not isinstance(obj, dict):
        obj = obj.__dict__
    for name in names.keys():
        value = obj[name]
        if name in fixes:
            # fixed point conversion
            value = fl2fi(value, fixes[name])
        elif isinstance(value, str):
            value = tobytes(value)
        elements.append(value)
        # Check it fits
        try:
            struct.pack(names[name], value)
        except Exception as e:
            raise ValueError(
                "Value %s does not fit in format %s for %s" % (value, names[name], name)
            ) from e
    data = struct.pack(*(formatstring,) + tuple(elements))
    return data


def unpack(fmt, data, obj=None):
    if obj is None:
        obj = {}
    data = tobytes(data)
    formatstring, names, fixes = getformat(fmt)
    if isinstance(obj, dict):
        d = obj
    else:
        d = obj.__dict__
    elements = struct.unpack(formatstring, data)
    for i, name in enumerate(names.keys()):
        value = elements[i]
        if name in fixes:
            # fixed point conversion
            value = fi2fl(value, fixes[name])
        elif isinstance(value, bytes):
            try:
                value = tostr(value)
            except UnicodeDecodeError:
                pass
        d[name] = value
    return obj


def unpack2(fmt, data, obj=None):
    length = calcsize(fmt)
    return unpack(fmt, data[:length], obj), data[length:]


def calcsize(fmt):
    formatstring, names, fixes = getformat(fmt)
    return struct.calcsize(formatstring)


# matches "name:formatchar" (whitespace is allowed)
_elementRE = re.compile(
    r"\s*"  # whitespace
    r"([A-Za-z_][A-Za-z_0-9]*)"  # name (python identifier)
    r"\s*:\s*"  # whitespace : whitespace
    r"([xcbB?hHiIlLqQfd]|"  # formatchar...
    r"[0-9]+[ps]|"  # ...formatchar...
    r"([0-9]+)\.([0-9]+)(F))"  # ...formatchar
    r"\s*"  # whitespace
    r"(#.*)?$"  # [comment] + end of string
)

# matches the special struct fmt chars and 'x' (pad byte)
_extraRE = re.compile(r"\s*([x@=<>!])\s*(#.*)?$")

# matches an "empty" string, possibly containing whitespace and/or a comment
_emptyRE = re.compile(r"\s*(#.*)?$")

_fixedpointmappings = {8: "b", 16: "h", 32: "l"}

_formatcache = {}


def getformat(fmt, keep_pad_byte=False):
    fmt = tostr(fmt, encoding="ascii")
    try:
        formatstring, names, fixes = _formatcache[fmt]
    except KeyError:
        lines = re.split("[\n;]", fmt)
        formatstring = ""
        names = {}
        fixes = {}
        for line in lines:
            if _emptyRE.match(line):
                continue
            m = _extraRE.match(line)
            if m:
                formatchar = m.group(1)
                if formatchar != "x" and formatstring:
                    raise Error("a special fmt char must be first")
            else:
                m = _elementRE.match(line)
                if not m:
                    raise Error("syntax error in fmt: '%s'" % line)
                name = m.group(1)
                formatchar = m.group(2)
                if keep_pad_byte or formatchar != "x":
                    names[name] = formatchar
                if m.group(3):
                    # fixed point
                    before = int(m.group(3))
                    after = int(m.group(4))
                    bits = before + after
                    if bits not in [8, 16, 32]:
                        raise Error("fixed point must be 8, 16 or 32 bits long")
                    formatchar = _fixedpointmappings[bits]
                    names[name] = formatchar
                    assert m.group(5) == "F"
                    fixes[name] = after
            formatstring += formatchar
        _formatcache[fmt] = formatstring, names, fixes
    return formatstring, names, fixes


def _test():
    fmt = """
		# comments are allowed
		>  # big endian (see documentation for struct)
		# empty lines are allowed:

		ashort: h
		along: l
		abyte: b	# a byte
		achar: c
		astr: 5s
		afloat: f; adouble: d	# multiple "statements" are allowed
		afixed: 16.16F
		abool: ?
		apad: x
	"""

    print("size:", calcsize(fmt))

    class foo(object):
        pass

    i = foo()

    i.ashort = 0x7FFF
    i.along = 0x7FFFFFFF
    i.abyte = 0x7F
    i.achar = "a"
    i.astr = "12345"
    i.afloat = 0.5
    i.adouble = 0.5
    i.afixed = 1.5
    i.abool = True

    data = pack(fmt, i)
    print("data:", repr(data))
    print(unpack(fmt, data))
    i2 = foo()
    unpack(fmt, data, i2)
    print(vars(i2))


if __name__ == "__main__":
    _test()
<|endoftext|>from io import BytesIO
import struct
from fontTools.misc import sstruct
from fontTools.misc.textTools import bytesjoin, tostr
from collections import OrderedDict
from collections.abc import MutableMapping


class ResourceError(Exception):
    pass


class ResourceReader(MutableMapping):
    """Reader for Mac OS resource forks.

    Parses a resource fork and returns resources according to their type.
    If run on OS X, this will open the resource fork in the filesystem.
    Otherwise, it will open the file itself and attempt to read it as
    though it were a resource fork.

    The returned object can be indexed by type and iterated over,
    returning in each case a list of py:class:`Resource` objects
    representing all the resources of a certain type.

    """

    def __init__(self, fileOrPath):
        """Open a file

        Args:
                fileOrPath: Either an object supporting a ``read`` method, an
                        ``os.PathLike`` object, or a string.
        """
        self._resources = OrderedDict()
        if hasattr(fileOrPath, "read"):
            self.file = fileOrPath
        else:
            try:
                # try reading from the resource fork (only works on OS X)
                self.file = self.openResourceFork(fileOrPath)
                self._readFile()
                return
            except (ResourceError, IOError):
                # if it fails, use the data fork
                self.file = self.openDataFork(fileOrPath)
        self._readFile()

    @staticmethod
    def openResourceFork(path):
        if hasattr(path, "__fspath__"):  # support os.PathLike objects
            path = path.__fspath__()
        with open(path + "/..namedfork/rsrc", "rb") as resfork:
            data = resfork.read()
        infile = BytesIO(data)
        infile.name = path
        return infile

    @staticmethod
    def openDataFork(path):
        with open(path, "rb") as datafork:
            data = datafork.read()
        infile = BytesIO(data)
        infile.name = path
        return infile

    def _readFile(self):
        self._readHeaderAndMap()
        self._readTypeList()

    def _read(self, numBytes, offset=None):
        if offset is not None:
            try:
                self.file.seek(offset)
            except OverflowError:
                raise ResourceError("Failed to seek offset ('offset' is too large)")
            if self.file.tell() != offset:
                raise ResourceError("Failed to seek offset (reached EOF)")
        try:
            data = self.file.read(numBytes)
        except OverflowError:
            raise ResourceError("Cannot read resource ('numBytes' is too large)")
        if len(data) != numBytes:
            raise ResourceError("Cannot read resource (not enough data)")
        return data

    def _readHeaderAndMap(self):
        self.file.seek(0)
        headerData = self._read(ResourceForkHeaderSize)
        sstruct.unpack(ResourceForkHeader, headerData, self)
        # seek to resource map, skip reserved
        mapOffset = self.mapOffset + 22
        resourceMapData = self._read(ResourceMapHeaderSize, mapOffset)
        sstruct.unpack(ResourceMapHeader, resourceMapData, self)
        self.absTypeListOffset = self.mapOffset + self.typeListOffset
        self.absNameListOffset = self.mapOffset + self.nameListOffset

    def _readTypeList(self):
        absTypeListOffset = self.absTypeListOffset
        numTypesData = self._read(2, absTypeListOffset)
        (self.numTypes,) = struct.unpack(">H", numTypesData)
        absTypeListOffset2 = absTypeListOffset + 2
        for i in range(self.numTypes + 1):
            resTypeItemOffset = absTypeListOffset2 + ResourceTypeItemSize * i
            resTypeItemData = self._read(ResourceTypeItemSize, resTypeItemOffset)
            item = sstruct.unpack(ResourceTypeItem, resTypeItemData)
            resType = tostr(item["type"], encoding="mac-roman")
            refListOffset = absTypeListOffset + item["refListOffset"]
            numRes = item["numRes"] + 1
            resources = self._readReferenceList(resType, refListOffset, numRes)
            self._resources[resType] = resources

    def _readReferenceList(self, resType, refListOffset, numRes):
        resources = []
        for i in range(numRes):
            refOffset = refListOffset + ResourceRefItemSize * i
            refData = self._read(ResourceRefItemSize, refOffset)
            res = Resource(resType)
            res.decompile(refData, self)
            resources.append(res)
        return resources

    def __getitem__(self, resType):
        return self._resources[resType]

    def __delitem__(self, resType):
        del self._resources[resType]

    def __setitem__(self, resType, resources):
        self._resources[resType] = resources

    def __len__(self):
        return len(self._resources)

    def __iter__(self):
        return iter(self._resources)

    def keys(self):
        return self._resources.keys()

    @property
    def types(self):
        """A list of the types of resources in the resource fork."""
        return list(self._resources.keys())

    def countResources(self, resType):
        """Return the number of resources of a given type."""
        try:
            return len(self[resType])
        except KeyError:
            return 0

    def getIndices(self, resType):
        """Returns a list of indices of resources of a given type."""
        numRes = self.countResources(resType)
        if numRes:
            return list(range(1, numRes + 1))
        else:
            return []

    def getNames(self, resType):
        """Return list of names of all resources of a given type."""
        return [res.name for res in self.get(resType, []) if res.name is not None]

    def getIndResource(self, resType, index):
        """Return resource of given type located at an index ranging from 1
        to the number of resources for that type, or None if not found.
        """
        if index < 1:
            return None
        try:
            res = self[resType][index - 1]
        except (KeyError, IndexError):
            return None
        return res

    def getNamedResource(self, resType, name):
        """Return the named resource of given type, else return None."""
        name = tostr(name, encoding="mac-roman")
        for res in self.get(resType, []):
            if res.name == name:
                return res
        return None

    def close(self):
        if not self.file.closed:
            self.file.close()


class Resource(object):
    """Represents a resource stored within a resource fork.

    Attributes:
            type: resource type.
            data: resource data.
            id: ID.
            name: resource name.
            attr: attributes.
    """

    def __init__(
        self, resType=None, resData=None, resID=None, resName=None, resAttr=None
    ):
        self.type = resType
        self.data = resData
        self.id = resID
        self.name = resName
        self.attr = resAttr

    def decompile(self, refData, reader):
        sstruct.unpack(ResourceRefItem, refData, self)
        # interpret 3-byte dataOffset as (padded) ULONG to unpack it with struct
        (self.dataOffset,) = struct.unpack(">L", bytesjoin([b"\0", self.dataOffset]))
        absDataOffset = reader.dataOffset + self.dataOffset
        (dataLength,) = struct.unpack(">L", reader._read(4, absDataOffset))
        self.data = reader._read(dataLength)
        if self.nameOffset == -1:
            return
        absNameOffset = reader.absNameListOffset + self.nameOffset
        (nameLength,) = struct.unpack("B", reader._read(1, absNameOffset))
        (name,) = struct.unpack(">%ss" % nameLength, reader._read(nameLength))
        self.name = tostr(name, encoding="mac-roman")


ResourceForkHeader = """
		> # big endian
		dataOffset:     L
		mapOffset:      L
		dataLen:        L
		mapLen:         L
"""

ResourceForkHeaderSize = sstruct.calcsize(ResourceForkHeader)

ResourceMapHeader = """
		> # big endian
		attr:              H
		typeListOffset:    H
		nameListOffset:    H
"""

ResourceMapHeaderSize = sstruct.calcsize(ResourceMapHeader)

ResourceTypeItem = """
		> # big endian
		type:              4s
		numRes:            H
		refListOffset:     H
"""

ResourceTypeItemSize = sstruct.calcsize(ResourceTypeItem)

ResourceRefItem = """
		> # big endian
		id:                h
		nameOffset:        h
		attr:              B
		dataOffset:        3s
		reserved:          L
"""

ResourceRefItemSize = sstruct.calcsize(ResourceRefItem)
<|endoftext|>__all__ = ["popCount", "bit_count", "bit_indices"]


try:
    bit_count = int.bit_count
except AttributeError:

    def bit_count(v):
        return bin(v).count("1")


"""Return number of 1 bits (population count) of the absolute value of an integer.

See https://docs.python.org/3.10/library/stdtypes.html#int.bit_count
"""
popCount = bit_count  # alias


def bit_indices(v):
    """Return list of indices where bits are set, 0 being the index of the least significant bit.

    >>> bit_indices(0b101)
    [0, 2]
    """
    return [i for i, b in enumerate(bin(v)[::-1]) if b == "1"]
<|endoftext|>"""Affine 2D transformation matrix class.

The Transform class implements various transformation matrix operations,
both on the matrix itself, as well as on 2D coordinates.

Transform instances are effectively immutable: all methods that operate on the
transformation itself always return a new instance. This has as the
interesting side effect that Transform instances are hashable, ie. they can be
used as dictionary keys.

This module exports the following symbols:

Transform
        this is the main class
Identity
        Transform instance set to the identity transformation
Offset
        Convenience function that returns a translating transformation
Scale
        Convenience function that returns a scaling transformation

The DecomposedTransform class implements a transformation with separate
translate, rotation, scale, skew, and transformation-center components.

:Example:

        >>> t = Transform(2, 0, 0, 3, 0, 0)
        >>> t.transformPoint((100, 100))
        (200, 300)
        >>> t = Scale(2, 3)
        >>> t.transformPoint((100, 100))
        (200, 300)
        >>> t.transformPoint((0, 0))
        (0, 0)
        >>> t = Offset(2, 3)
        >>> t.transformPoint((100, 100))
        (102, 103)
        >>> t.transformPoint((0, 0))
        (2, 3)
        >>> t2 = t.scale(0.5)
        >>> t2.transformPoint((100, 100))
        (52.0, 53.0)
        >>> import math
        >>> t3 = t2.rotate(math.pi / 2)
        >>> t3.transformPoint((0, 0))
        (2.0, 3.0)
        >>> t3.transformPoint((100, 100))
        (-48.0, 53.0)
        >>> t = Identity.scale(0.5).translate(100, 200).skew(0.1, 0.2)
        >>> t.transformPoints([(0, 0), (1, 1), (100, 100)])
        [(50.0, 100.0), (50.550167336042726, 100.60135501775433), (105.01673360427253, 160.13550177543362)]
        >>>
"""

from __future__ import annotations

import math
from typing import NamedTuple
from dataclasses import dataclass

__all__ = ["Transform", "Identity", "Offset", "Scale", "DecomposedTransform"]


_EPSILON = 1e-15
_ONE_EPSILON = 1 - _EPSILON
_MINUS_ONE_EPSILON = -1 + _EPSILON


def _normSinCos(v: float) -> float:
    if abs(v) < _EPSILON:
        v = 0
    elif v > _ONE_EPSILON:
        v = 1
    elif v < _MINUS_ONE_EPSILON:
        v = -1
    return v


class Transform(NamedTuple):
    """2x2 transformation matrix plus offset, a.k.a. Affine transform.
    Transform instances are immutable: all transforming methods, eg.
    rotate(), return a new Transform instance.

    :Example:

            >>> t = Transform()
            >>> t
            <Transform [1 0 0 1 0 0]>
            >>> t.scale(2)
            <Transform [2 0 0 2 0 0]>
            >>> t.scale(2.5, 5.5)
            <Transform [2.5 0 0 5.5 0 0]>
            >>>
            >>> t.scale(2, 3).transformPoint((100, 100))
            (200, 300)

    Transform's constructor takes six arguments, all of which are
    optional, and can be used as keyword arguments::

            >>> Transform(12)
            <Transform [12 0 0 1 0 0]>
            >>> Transform(dx=12)
            <Transform [1 0 0 1 12 0]>
            >>> Transform(yx=12)
            <Transform [1 0 12 1 0 0]>

    Transform instances also behave like sequences of length 6::

            >>> len(Identity)
            6
            >>> list(Identity)
            [1, 0, 0, 1, 0, 0]
            >>> tuple(Identity)
            (1, 0, 0, 1, 0, 0)

    Transform instances are comparable::

            >>> t1 = Identity.scale(2, 3).translate(4, 6)
            >>> t2 = Identity.translate(8, 18).scale(2, 3)
            >>> t1 == t2
            1

    But beware of floating point rounding errors::

            >>> t1 = Identity.scale(0.2, 0.3).translate(0.4, 0.6)
            >>> t2 = Identity.translate(0.08, 0.18).scale(0.2, 0.3)
            >>> t1
            <Transform [0.2 0 0 0.3 0.08 0.18]>
            >>> t2
            <Transform [0.2 0 0 0.3 0.08 0.18]>
            >>> t1 == t2
            0

    Transform instances are hashable, meaning you can use them as
    keys in dictionaries::

            >>> d = {Scale(12, 13): None}
            >>> d
            {<Transform [12 0 0 13 0 0]>: None}

    But again, beware of floating point rounding errors::

            >>> t1 = Identity.scale(0.2, 0.3).translate(0.4, 0.6)
            >>> t2 = Identity.translate(0.08, 0.18).scale(0.2, 0.3)
            >>> t1
            <Transform [0.2 0 0 0.3 0.08 0.18]>
            >>> t2
            <Transform [0.2 0 0 0.3 0.08 0.18]>
            >>> d = {t1: None}
            >>> d
            {<Transform [0.2 0 0 0.3 0.08 0.18]>: None}
            >>> d[t2]
            Traceback (most recent call last):
              File "<stdin>", line 1, in ?
            KeyError: <Transform [0.2 0 0 0.3 0.08 0.18]>
    """

    xx: float = 1
    xy: float = 0
    yx: float = 0
    yy: float = 1
    dx: float = 0
    dy: float = 0

    def transformPoint(self, p):
        """Transform a point.

        :Example:

                >>> t = Transform()
                >>> t = t.scale(2.5, 5.5)
                >>> t.transformPoint((100, 100))
                (250.0, 550.0)
        """
        x, y = p
        xx, xy, yx, yy, dx, dy = self
        return (xx * x + yx * y + dx, xy * x + yy * y + dy)

    def transformPoints(self, points):
        """Transform a list of points.

        :Example:

                >>> t = Scale(2, 3)
                >>> t.transformPoints([(0, 0), (0, 100), (100, 100), (100, 0)])
                [(0, 0), (0, 300), (200, 300), (200, 0)]
                >>>
        """
        xx, xy, yx, yy, dx, dy = self
        return [(xx * x + yx * y + dx, xy * x + yy * y + dy) for x, y in points]

    def transformVector(self, v):
        """Transform an (dx, dy) vector, treating translation as zero.

        :Example:

                >>> t = Transform(2, 0, 0, 2, 10, 20)
                >>> t.transformVector((3, -4))
                (6, -8)
                >>>
        """
        dx, dy = v
        xx, xy, yx, yy = self[:4]
        return (xx * dx + yx * dy, xy * dx + yy * dy)

    def transformVectors(self, vectors):
        """Transform a list of (dx, dy) vector, treating translation as zero.

        :Example:
                >>> t = Transform(2, 0, 0, 2, 10, 20)
                >>> t.transformVectors([(3, -4), (5, -6)])
                [(6, -8), (10, -12)]
                >>>
        """
        xx, xy, yx, yy = self[:4]
        return [(xx * dx + yx * dy, xy * dx + yy * dy) for dx, dy in vectors]

    def translate(self, x: float = 0, y: float = 0):
        """Return a new transformation, translated (offset) by x, y.

        :Example:
                >>> t = Transform()
                >>> t.translate(20, 30)
                <Transform [1 0 0 1 20 30]>
                >>>
        """
        return self.transform((1, 0, 0, 1, x, y))

    def scale(self, x: float = 1, y: float | None = None):
        """Return a new transformation, scaled by x, y. The 'y' argument
        may be None, which implies to use the x value for y as well.

        :Example:
                >>> t = Transform()
                >>> t.scale(5)
                <Transform [5 0 0 5 0 0]>
                >>> t.scale(5, 6)
                <Transform [5 0 0 6 0 0]>
                >>>
        """
        if y is None:
            y = x
        return self.transform((x, 0, 0, y, 0, 0))

    def rotate(self, angle: float):
        """Return a new transformation, rotated by 'angle' (radians).

        :Example:
                >>> import math
                >>> t = Transform()
                >>> t.rotate(math.pi / 2)
                <Transform [0 1 -1 0 0 0]>
                >>>
        """
        c = _normSinCos(math.cos(angle))
        s = _normSinCos(math.sin(angle))
        return self.transform((c, s, -s, c, 0, 0))

    def skew(self, x: float = 0, y: float = 0):
        """Return a new transformation, skewed by x and y.

        :Example:
                >>> import math
                >>> t = Transform()
                >>> t.skew(math.pi / 4)
                <Transform [1 0 1 1 0 0]>
                >>>
        """
        return self.transform((1, math.tan(y), math.tan(x), 1, 0, 0))

    def transform(self, other):
        """Return a new transformation, transformed by another
        transformation.

        :Example:
                >>> t = Transform(2, 0, 0, 3, 1, 6)
                >>> t.transform((4, 3, 2, 1, 5, 6))
                <Transform [8 9 4 3 11 24]>
                >>>
        """
        xx1, xy1, yx1, yy1, dx1, dy1 = other
        xx2, xy2, yx2, yy2, dx2, dy2 = self
        return self.__class__(
            xx1 * xx2 + xy1 * yx2,
            xx1 * xy2 + xy1 * yy2,
            yx1 * xx2 + yy1 * yx2,
            yx1 * xy2 + yy1 * yy2,
            xx2 * dx1 + yx2 * dy1 + dx2,
            xy2 * dx1 + yy2 * dy1 + dy2,
        )

    def reverseTransform(self, other):
        """Return a new transformation, which is the other transformation
        transformed by self. self.reverseTransform(other) is equivalent to
        other.transform(self).

        :Example:
                >>> t = Transform(2, 0, 0, 3, 1, 6)
                >>> t.reverseTransform((4, 3, 2, 1, 5, 6))
                <Transform [8 6 6 3 21 15]>
                >>> Transform(4, 3, 2, 1, 5, 6).transform((2, 0, 0, 3, 1, 6))
                <Transform [8 6 6 3 21 15]>
                >>>
        """
        xx1, xy1, yx1, yy1, dx1, dy1 = self
        xx2, xy2, yx2, yy2, dx2, dy2 = other
        return self.__class__(
            xx1 * xx2 + xy1 * yx2,
            xx1 * xy2 + xy1 * yy2,
            yx1 * xx2 + yy1 * yx2,
            yx1 * xy2 + yy1 * yy2,
            xx2 * dx1 + yx2 * dy1 + dx2,
            xy2 * dx1 + yy2 * dy1 + dy2,
        )

    def inverse(self):
        """Return the inverse transformation.

        :Example:
                >>> t = Identity.translate(2, 3).scale(4, 5)
                >>> t.transformPoint((10, 20))
                (42, 103)
                >>> it = t.inverse()
                >>> it.transformPoint((42, 103))
                (10.0, 20.0)
                >>>
        """
        if self == Identity:
            return self
        xx, xy, yx, yy, dx, dy = self
        det = xx * yy - yx * xy
        xx, xy, yx, yy = yy / det, -xy / det, -yx / det, xx / det
        dx, dy = -xx * dx - yx * dy, -xy * dx - yy * dy
        return self.__class__(xx, xy, yx, yy, dx, dy)

    def toPS(self) -> str:
        """Return a PostScript representation

        :Example:

                >>> t = Identity.scale(2, 3).translate(4, 5)
                >>> t.toPS()
                '[2 0 0 3 8 15]'
                >>>
        """
        return "[%s %s %s %s %s %s]" % self

    def toDecomposed(self) -> "DecomposedTransform":
        """Decompose into a DecomposedTransform."""
        return DecomposedTransform.fromTransform(self)

    def __bool__(self) -> bool:
        """Returns True if transform is not identity, False otherwise.

        :Example:

                >>> bool(Identity)
                False
                >>> bool(Transform())
                False
                >>> bool(Scale(1.))
                False
                >>> bool(Scale(2))
                True
                >>> bool(Offset())
                False
                >>> bool(Offset(0))
                False
                >>> bool(Offset(2))
                True
        """
        return self != Identity

    def __repr__(self) -> str:
        return "<%s [%g %g %g %g %g %g]>" % ((self.__class__.__name__,) + self)


Identity = Transform()


def Offset(x: float = 0, y: float = 0) -> Transform:
    """Return the identity transformation offset by x, y.

    :Example:
            >>> Offset(2, 3)
            <Transform [1 0 0 1 2 3]>
            >>>
    """
    return Transform(1, 0, 0, 1, x, y)


def Scale(x: float, y: float | None = None) -> Transform:
    """Return the identity transformation scaled by x, y. The 'y' argument
    may be None, which implies to use the x value for y as well.

    :Example:
            >>> Scale(2, 3)
            <Transform [2 0 0 3 0 0]>
            >>>
    """
    if y is None:
        y = x
    return Transform(x, 0, 0, y, 0, 0)


@dataclass
class DecomposedTransform:
    """The DecomposedTransform class implements a transformation with separate
    translate, rotation, scale, skew, and transformation-center components.
    """

    translateX: float = 0
    translateY: float = 0
    rotation: float = 0  # in degrees, counter-clockwise
    scaleX: float = 1
    scaleY: float = 1
    skewX: float = 0  # in degrees, clockwise
    skewY: float = 0  # in degrees, counter-clockwise
    tCenterX: float = 0
    tCenterY: float = 0

    def __bool__(self):
        return (
            self.translateX != 0
            or self.translateY != 0
            or self.rotation != 0
            or self.scaleX != 1
            or self.scaleY != 1
            or self.skewX != 0
            or self.skewY != 0
            or self.tCenterX != 0
            or self.tCenterY != 0
        )

    @classmethod
    def fromTransform(self, transform):
        """Return a DecomposedTransform() equivalent of this transformation.
        The returned solution always has skewY = 0, and angle in the (-180, 180].

        :Example:
                >>> DecomposedTransform.fromTransform(Transform(3, 0, 0, 2, 0, 0))
                DecomposedTransform(translateX=0, translateY=0, rotation=0.0, scaleX=3.0, scaleY=2.0, skewX=0.0, skewY=0.0, tCenterX=0, tCenterY=0)
                >>> DecomposedTransform.fromTransform(Transform(0, 0, 0, 1, 0, 0))
                DecomposedTransform(translateX=0, translateY=0, rotation=0.0, scaleX=0.0, scaleY=1.0, skewX=0.0, skewY=0.0, tCenterX=0, tCenterY=0)
                >>> DecomposedTransform.fromTransform(Transform(0, 0, 1, 1, 0, 0))
                DecomposedTransform(translateX=0, translateY=0, rotation=-45.0, scaleX=0.0, scaleY=1.4142135623730951, skewX=0.0, skewY=0.0, tCenterX=0, tCenterY=0)
        """
        # Adapted from an answer on
        # https://math.stackexchange.com/questions/13150/extracting-rotation-scale-values-from-2d-transformation-matrix

        a, b, c, d, x, y = transform

        sx = math.copysign(1, a)
        if sx < 0:
            a *= sx
            b *= sx

        delta = a * d - b * c

        rotation = 0
        scaleX = scaleY = 0
        skewX = 0

        # Apply the QR-like decomposition.
        if a != 0 or b != 0:
            r = math.sqrt(a * a + b * b)
            rotation = math.acos(a / r) if b >= 0 else -math.acos(a / r)
            scaleX, scaleY = (r, delta / r)
            skewX = math.atan((a * c + b * d) / (r * r))
        elif c != 0 or d != 0:
            s = math.sqrt(c * c + d * d)
            rotation = math.pi / 2 - (
                math.acos(-c / s) if d >= 0 else -math.acos(c / s)
            )
            scaleX, scaleY = (delta / s, s)
        else:
            # a = b = c = d = 0
            pass

        return DecomposedTransform(
            x,
            y,
            math.degrees(rotation),
            scaleX * sx,
            scaleY,
            math.degrees(skewX) * sx,
            0.0,
            0,
            0,
        )

    def toTransform(self) -> Transform:
        """Return the Transform() equivalent of this transformation.

        :Example:
                >>> DecomposedTransform(scaleX=2, scaleY=2).toTransform()
                <Transform [2 0 0 2 0 0]>
                >>>
        """
        t = Transform()
        t = t.translate(
            self.translateX + self.tCenterX, self.translateY + self.tCenterY
        )
        t = t.rotate(math.radians(self.rotation))
        t = t.scale(self.scaleX, self.scaleY)
        t = t.skew(math.radians(self.skewX), math.radians(self.skewY))
        t = t.translate(-self.tCenterX, -self.tCenterY)
        return t


if __name__ == "__main__":
    import sys
    import doctest

    sys.exit(doctest.testmod().failed)
<|endoftext|>_accessstrings = {0: "", 1: "readonly", 2: "executeonly", 3: "noaccess"}


class ps_object(object):
    literal = 1
    access = 0
    value = None

    def __init__(self, value):
        self.value = value
        self.type = self.__class__.__name__[3:] + "type"

    def __repr__(self):
        return "<%s %s>" % (self.__class__.__name__[3:], repr(self.value))


class ps_operator(ps_object):
    literal = 0

    def __init__(self, name, function):
        self.name = name
        self.function = function
        self.type = self.__class__.__name__[3:] + "type"

    def __repr__(self):
        return "<operator %s>" % self.name


class ps_procedure(ps_object):
    literal = 0

    def __repr__(self):
        return "<procedure>"

    def __str__(self):
        psstring = "{"
        for i in range(len(self.value)):
            if i:
                psstring = psstring + " " + str(self.value[i])
            else:
                psstring = psstring + str(self.value[i])
        return psstring + "}"


class ps_name(ps_object):
    literal = 0

    def __str__(self):
        if self.literal:
            return "/" + self.value
        else:
            return self.value


class ps_literal(ps_object):
    def __str__(self):
        return "/" + self.value


class ps_array(ps_object):
    def __str__(self):
        psstring = "["
        for i in range(len(self.value)):
            item = self.value[i]
            access = _accessstrings[item.access]
            if access:
                access = " " + access
            if i:
                psstring = psstring + " " + str(item) + access
            else:
                psstring = psstring + str(item) + access
        return psstring + "]"

    def __repr__(self):
        return "<array>"


_type1_pre_eexec_order = [
    "FontInfo",
    "FontName",
    "Encoding",
    "PaintType",
    "FontType",
    "FontMatrix",
    "FontBBox",
    "UniqueID",
    "Metrics",
    "StrokeWidth",
]

_type1_fontinfo_order = [
    "version",
    "Notice",
    "FullName",
    "FamilyName",
    "Weight",
    "ItalicAngle",
    "isFixedPitch",
    "UnderlinePosition",
    "UnderlineThickness",
]

_type1_post_eexec_order = ["Private", "CharStrings", "FID"]


def _type1_item_repr(key, value):
    psstring = ""
    access = _accessstrings[value.access]
    if access:
        access = access + " "
    if key == "CharStrings":
        psstring = psstring + "/%s %s def\n" % (
            key,
            _type1_CharString_repr(value.value),
        )
    elif key == "Encoding":
        psstring = psstring + _type1_Encoding_repr(value, access)
    else:
        psstring = psstring + "/%s %s %sdef\n" % (str(key), str(value), access)
    return psstring


def _type1_Encoding_repr(encoding, access):
    encoding = encoding.value
    psstring = "/Encoding 256 array\n0 1 255 {1 index exch /.notdef put} for\n"
    for i in range(256):
        name = encoding[i].value
        if name != ".notdef":
            psstring = psstring + "dup %d /%s put\n" % (i, name)
    return psstring + access + "def\n"


def _type1_CharString_repr(charstrings):
    items = sorted(charstrings.items())
    return "xxx"


class ps_font(ps_object):
    def __str__(self):
        psstring = "%d dict dup begin\n" % len(self.value)
        for key in _type1_pre_eexec_order:
            try:
                value = self.value[key]
            except KeyError:
                pass
            else:
                psstring = psstring + _type1_item_repr(key, value)
        items = sorted(self.value.items())
        for key, value in items:
            if key not in _type1_pre_eexec_order + _type1_post_eexec_order:
                psstring = psstring + _type1_item_repr(key, value)
        psstring = psstring + "currentdict end\ncurrentfile eexec\ndup "
        for key in _type1_post_eexec_order:
            try:
                value = self.value[key]
            except KeyError:
                pass
            else:
                psstring = psstring + _type1_item_repr(key, value)
        return (
            psstring
            + "dup/FontName get exch definefont pop\nmark currentfile closefile\n"
            + 8 * (64 * "0" + "\n")
            + "cleartomark"
            + "\n"
        )

    def __repr__(self):
        return "<font>"


class ps_file(ps_object):
    pass


class ps_dict(ps_object):
    def __str__(self):
        psstring = "%d dict dup begin\n" % len(self.value)
        items = sorted(self.value.items())
        for key, value in items:
            access = _accessstrings[value.access]
            if access:
                access = access + " "
            psstring = psstring + "/%s %s %sdef\n" % (str(key), str(value), access)
        return psstring + "end "

    def __repr__(self):
        return "<dict>"


class ps_mark(ps_object):
    def __init__(self):
        self.value = "mark"
        self.type = self.__class__.__name__[3:] + "type"


class ps_procmark(ps_object):
    def __init__(self):
        self.value = "procmark"
        self.type = self.__class__.__name__[3:] + "type"


class ps_null(ps_object):
    def __init__(self):
        self.type = self.__class__.__name__[3:] + "type"


class ps_boolean(ps_object):
    def __str__(self):
        if self.value:
            return "true"
        else:
            return "false"


class ps_string(ps_object):
    def __str__(self):
        return "(%s)" % repr(self.value)[1:-1]


class ps_integer(ps_object):
    def __str__(self):
        return repr(self.value)


class ps_real(ps_object):
    def __str__(self):
        return repr(self.value)


class PSOperators(object):
    def ps_def(self):
        obj = self.pop()
        name = self.pop()
        self.dictstack[-1][name.value] = obj

    def ps_bind(self):
        proc = self.pop("proceduretype")
        self.proc_bind(proc)
        self.push(proc)

    def proc_bind(self, proc):
        for i in range(len(proc.value)):
            item = proc.value[i]
            if item.type == "proceduretype":
                self.proc_bind(item)
            else:
                if not item.literal:
                    try:
                        obj = self.resolve_name(item.value)
                    except:
                        pass
                    else:
                        if obj.type == "operatortype":
                            proc.value[i] = obj

    def ps_exch(self):
        if len(self.stack) < 2:
            raise RuntimeError("stack underflow")
        obj1 = self.pop()
        obj2 = self.pop()
        self.push(obj1)
        self.push(obj2)

    def ps_dup(self):
        if not self.stack:
            raise RuntimeError("stack underflow")
        self.push(self.stack[-1])

    def ps_exec(self):
        obj = self.pop()
        if obj.type == "proceduretype":
            self.call_procedure(obj)
        else:
            self.handle_object(obj)

    def ps_count(self):
        self.push(ps_integer(len(self.stack)))

    def ps_eq(self):
        any1 = self.pop()
        any2 = self.pop()
        self.push(ps_boolean(any1.value == any2.value))

    def ps_ne(self):
        any1 = self.pop()
        any2 = self.pop()
        self.push(ps_boolean(any1.value != any2.value))

    def ps_cvx(self):
        obj = self.pop()
        obj.literal = 0
        self.push(obj)

    def ps_matrix(self):
        matrix = [
            ps_real(1.0),
            ps_integer(0),
            ps_integer(0),
            ps_real(1.0),
            ps_integer(0),
            ps_integer(0),
        ]
        self.push(ps_array(matrix))

    def ps_string(self):
        num = self.pop("integertype").value
        self.push(ps_string("\0" * num))

    def ps_type(self):
        obj = self.pop()
        self.push(ps_string(obj.type))

    def ps_store(self):
        value = self.pop()
        key = self.pop()
        name = key.value
        for i in range(len(self.dictstack) - 1, -1, -1):
            if name in self.dictstack[i]:
                self.dictstack[i][name] = value
                break
        self.dictstack[-1][name] = value

    def ps_where(self):
        name = self.pop()
        # XXX
        self.push(ps_boolean(0))

    def ps_systemdict(self):
        self.push(ps_dict(self.dictstack[0]))

    def ps_userdict(self):
        self.push(ps_dict(self.dictstack[1]))

    def ps_currentdict(self):
        self.push(ps_dict(self.dictstack[-1]))

    def ps_currentfile(self):
        self.push(ps_file(self.tokenizer))

    def ps_eexec(self):
        f = self.pop("filetype").value
        f.starteexec()

    def ps_closefile(self):
        f = self.pop("filetype").value
        f.skipwhite()
        f.stopeexec()

    def ps_cleartomark(self):
        obj = self.pop()
        while obj != self.mark:
            obj = self.pop()

    def ps_readstring(self, ps_boolean=ps_boolean, len=len):
        s = self.pop("stringtype")
        oldstr = s.value
        f = self.pop("filetype")
        # pad = file.value.read(1)
        # for StringIO, this is faster
        f.value.pos = f.value.pos + 1
        newstr = f.value.read(len(oldstr))
        s.value = newstr
        self.push(s)
        self.push(ps_boolean(len(oldstr) == len(newstr)))

    def ps_known(self):
        key = self.pop()
        d = self.pop("dicttype", "fonttype")
        self.push(ps_boolean(key.value in d.value))

    def ps_if(self):
        proc = self.pop("proceduretype")
        if self.pop("booleantype").value:
            self.call_procedure(proc)

    def ps_ifelse(self):
        proc2 = self.pop("proceduretype")
        proc1 = self.pop("proceduretype")
        if self.pop("booleantype").value:
            self.call_procedure(proc1)
        else:
            self.call_procedure(proc2)

    def ps_readonly(self):
        obj = self.pop()
        if obj.access < 1:
            obj.access = 1
        self.push(obj)

    def ps_executeonly(self):
        obj = self.pop()
        if obj.access < 2:
            obj.access = 2
        self.push(obj)

    def ps_noaccess(self):
        obj = self.pop()
        if obj.access < 3:
            obj.access = 3
        self.push(obj)

    def ps_not(self):
        obj = self.pop("booleantype", "integertype")
        if obj.type == "booleantype":
            self.push(ps_boolean(not obj.value))
        else:
            self.push(ps_integer(~obj.value))

    def ps_print(self):
        str = self.pop("stringtype")
        print("PS output --->", str.value)

    def ps_anchorsearch(self):
        seek = self.pop("stringtype")
        s = self.pop("stringtype")
        seeklen = len(seek.value)
        if s.value[:seeklen] == seek.value:
            self.push(ps_string(s.value[seeklen:]))
            self.push(seek)
            self.push(ps_boolean(1))
        else:
            self.push(s)
            self.push(ps_boolean(0))

    def ps_array(self):
        num = self.pop("integertype")
        array = ps_array([None] * num.value)
        self.push(array)

    def ps_astore(self):
        array = self.pop("arraytype")
        for i in range(len(array.value) - 1, -1, -1):
            array.value[i] = self.pop()
        self.push(array)

    def ps_load(self):
        name = self.pop()
        self.push(self.resolve_name(name.value))

    def ps_put(self):
        obj1 = self.pop()
        obj2 = self.pop()
        obj3 = self.pop("arraytype", "dicttype", "stringtype", "proceduretype")
        tp = obj3.type
        if tp == "arraytype" or tp == "proceduretype":
            obj3.value[obj2.value] = obj1
        elif tp == "dicttype":
            obj3.value[obj2.value] = obj1
        elif tp == "stringtype":
            index = obj2.value
            obj3.value = obj3.value[:index] + chr(obj1.value) + obj3.value[index + 1 :]

    def ps_get(self):
        obj1 = self.pop()
        if obj1.value == "Encoding":
            pass
        obj2 = self.pop(
            "arraytype", "dicttype", "stringtype", "proceduretype", "fonttype"
        )
        tp = obj2.type
        if tp in ("arraytype", "proceduretype"):
            self.push(obj2.value[obj1.value])
        elif tp in ("dicttype", "fonttype"):
            self.push(obj2.value[obj1.value])
        elif tp == "stringtype":
            self.push(ps_integer(ord(obj2.value[obj1.value])))
        else:
            assert False, "shouldn't get here"

    def ps_getinterval(self):
        obj1 = self.pop("integertype")
        obj2 = self.pop("integertype")
        obj3 = self.pop("arraytype", "stringtype")
        tp = obj3.type
        if tp == "arraytype":
            self.push(ps_array(obj3.value[obj2.value : obj2.value + obj1.value]))
        elif tp == "stringtype":
            self.push(ps_string(obj3.value[obj2.value : obj2.value + obj1.value]))

    def ps_putinterval(self):
        obj1 = self.pop("arraytype", "stringtype")
        obj2 = self.pop("integertype")
        obj3 = self.pop("arraytype", "stringtype")
        tp = obj3.type
        if tp == "arraytype":
            obj3.value[obj2.value : obj2.value + len(obj1.value)] = obj1.value
        elif tp == "stringtype":
            newstr = obj3.value[: obj2.value]
            newstr = newstr + obj1.value
            newstr = newstr + obj3.value[obj2.value + len(obj1.value) :]
            obj3.value = newstr

    def ps_cvn(self):
        self.push(ps_name(self.pop("stringtype").value))

    def ps_index(self):
        n = self.pop("integertype").value
        if n < 0:
            raise RuntimeError("index may not be negative")
        self.push(self.stack[-1 - n])

    def ps_for(self):
        proc = self.pop("proceduretype")
        limit = self.pop("integertype", "realtype").value
        increment = self.pop("integertype", "realtype").value
        i = self.pop("integertype", "realtype").value
        while 1:
            if increment > 0:
                if i > limit:
                    break
            else:
                if i < limit:
                    break
            if type(i) == type(0.0):
                self.push(ps_real(i))
            else:
                self.push(ps_integer(i))
            self.call_procedure(proc)
            i = i + increment

    def ps_forall(self):
        proc = self.pop("proceduretype")
        obj = self.pop("arraytype", "stringtype", "dicttype")
        tp = obj.type
        if tp == "arraytype":
            for item in obj.value:
                self.push(item)
                self.call_procedure(proc)
        elif tp == "stringtype":
            for item in obj.value:
                self.push(ps_integer(ord(item)))
                self.call_procedure(proc)
        elif tp == "dicttype":
            for key, value in obj.value.items():
                self.push(ps_name(key))
                self.push(value)
                self.call_procedure(proc)

    def ps_definefont(self):
        font = self.pop("dicttype")
        name = self.pop()
        font = ps_font(font.value)
        self.dictstack[0]["FontDirectory"].value[name.value] = font
        self.push(font)

    def ps_findfont(self):
        name = self.pop()
        font = self.dictstack[0]["FontDirectory"].value[name.value]
        self.push(font)

    def ps_pop(self):
        self.pop()

    def ps_dict(self):
        self.pop("integertype")
        self.push(ps_dict({}))

    def ps_begin(self):
        self.dictstack.append(self.pop("dicttype").value)

    def ps_end(self):
        if len(self.dictstack) > 2:
            del self.dictstack[-1]
        else:
            raise RuntimeError("dictstack underflow")


notdef = ".notdef"
from fontTools.encodings.StandardEncoding import StandardEncoding

ps_StandardEncoding = list(map(ps_name, StandardEncoding))
<|endoftext|>"""Generic visitor pattern implementation for Python objects."""

import enum
import weakref


class Visitor(object):
    defaultStop = False

    _visitors = {
        # By default we skip visiting weak references to avoid recursion
        # issues. Users can override this by registering a visit
        # function for weakref.ProxyType.
        weakref.ProxyType: {None: lambda self, obj, *args, **kwargs: False}
    }

    @classmethod
    def _register(celf, clazzes_attrs):
        assert celf != Visitor, "Subclass Visitor instead."
        if "_visitors" not in celf.__dict__:
            celf._visitors = {}

        def wrapper(method):
            assert method.__name__ == "visit"
            for clazzes, attrs in clazzes_attrs:
                if type(clazzes) != tuple:
                    clazzes = (clazzes,)
                if type(attrs) == str:
                    attrs = (attrs,)
                for clazz in clazzes:
                    _visitors = celf._visitors.setdefault(clazz, {})
                    for attr in attrs:
                        assert attr not in _visitors, (
                            "Oops, class '%s' has visitor function for '%s' defined already."
                            % (clazz.__name__, attr)
                        )
                        _visitors[attr] = method
            return None

        return wrapper

    @classmethod
    def register(celf, clazzes):
        if type(clazzes) != tuple:
            clazzes = (clazzes,)
        return celf._register([(clazzes, (None,))])

    @classmethod
    def register_attr(celf, clazzes, attrs):
        clazzes_attrs = []
        if type(clazzes) != tuple:
            clazzes = (clazzes,)
        if type(attrs) == str:
            attrs = (attrs,)
        for clazz in clazzes:
            clazzes_attrs.append((clazz, attrs))
        return celf._register(clazzes_attrs)

    @classmethod
    def register_attrs(celf, clazzes_attrs):
        return celf._register(clazzes_attrs)

    @classmethod
    def _visitorsFor(celf, thing, _default={}):
        typ = type(thing)

        for celf in celf.mro():
            _visitors = getattr(celf, "_visitors", None)
            if _visitors is None:
                break

            for base in typ.mro():
                m = celf._visitors.get(base, None)
                if m is not None:
                    return m

        return _default

    def visitObject(self, obj, *args, **kwargs):
        """Called to visit an object. This function loops over all non-private
        attributes of the objects and calls any user-registered (via
        ``@register_attr()`` or ``@register_attrs()``) ``visit()`` functions.

        The visitor will proceed to call ``self.visitAttr()``, unless there is a
        user-registered visit function and:

        * It returns ``False``; or
        * It returns ``None`` (or doesn't return anything) and
          ``visitor.defaultStop`` is ``True`` (non-default).
        """

        keys = sorted(vars(obj).keys())
        _visitors = self._visitorsFor(obj)
        defaultVisitor = _visitors.get("*", None)
        for key in keys:
            if key[0] == "_":
                continue
            value = getattr(obj, key)
            visitorFunc = _visitors.get(key, defaultVisitor)
            if visitorFunc is not None:
                ret = visitorFunc(self, obj, key, value, *args, **kwargs)
                if ret == False or (ret is None and self.defaultStop):
                    continue
            self.visitAttr(obj, key, value, *args, **kwargs)

    def visitAttr(self, obj, attr, value, *args, **kwargs):
        """Called to visit an attribute of an object."""
        self.visit(value, *args, **kwargs)

    def visitList(self, obj, *args, **kwargs):
        """Called to visit any value that is a list."""
        for value in obj:
            self.visit(value, *args, **kwargs)

    def visitDict(self, obj, *args, **kwargs):
        """Called to visit any value that is a dictionary."""
        for value in obj.values():
            self.visit(value, *args, **kwargs)

    def visitLeaf(self, obj, *args, **kwargs):
        """Called to visit any value that is not an object, list,
        or dictionary."""
        pass

    def visit(self, obj, *args, **kwargs):
        """This is the main entry to the visitor. The visitor will visit object
        ``obj``.

        The visitor will first determine if there is a registered (via
        ``@register()``) visit function for the type of object. If there is, it
        will be called, and ``(visitor, obj, *args, **kwargs)`` will be passed
        to the user visit function.

        The visitor will not recurse if there is a user-registered visit
        function and:

        * It returns ``False``; or
        * It returns ``None`` (or doesn't return anything) and
          ``visitor.defaultStop`` is ``True`` (non-default)

        Otherwise,  the visitor will proceed to dispatch to one of
        ``self.visitObject()``, ``self.visitList()``, ``self.visitDict()``, or
        ``self.visitLeaf()`` (any of which can be overriden in a subclass).
        """

        visitorFunc = self._visitorsFor(obj).get(None, None)
        if visitorFunc is not None:
            ret = visitorFunc(self, obj, *args, **kwargs)
            if ret == False or (ret is None and self.defaultStop):
                return
        if hasattr(obj, "__dict__") and not isinstance(obj, enum.Enum):
            self.visitObject(obj, *args, **kwargs)
        elif isinstance(obj, list):
            self.visitList(obj, *args, **kwargs)
        elif isinstance(obj, dict):
            self.visitDict(obj, *args, **kwargs)
        else:
            self.visitLeaf(obj, *args, **kwargs)
<|endoftext|>from itertools import *

# Python 3.12:
if "batched" not in globals():
    # https://docs.python.org/3/library/itertools.html#itertools.batched
    def batched(iterable, n):
        # batched('ABCDEFG', 3) --> ABC DEF G
        if n < 1:
            raise ValueError("n must be at least one")
        it = iter(iterable)
        while batch := tuple(islice(it, n)):
            yield batch
<|endoftext|>"""Shim module exporting the same ElementTree API for lxml and
xml.etree backends.

When lxml is installed, it is automatically preferred over the built-in
xml.etree module.
On Python 2.7, the cElementTree module is preferred over the pure-python
ElementTree module.

Besides exporting a unified interface, this also defines extra functions
or subclasses built-in ElementTree classes to add features that are
only availble in lxml, like OrderedDict for attributes, pretty_print and
iterwalk.
"""

from fontTools.misc.textTools import tostr

XML_DECLARATION = """<?xml version='1.0' encoding='%s'?>"""

__all__ = [
    # public symbols
    "Comment",
    "dump",
    "Element",
    "ElementTree",
    "fromstring",
    "fromstringlist",
    "iselement",
    "iterparse",
    "parse",
    "ParseError",
    "PI",
    "ProcessingInstruction",
    "QName",
    "SubElement",
    "tostring",
    "tostringlist",
    "TreeBuilder",
    "XML",
    "XMLParser",
    "register_namespace",
]

try:
    from lxml.etree import *
    from lxml.etree import LXML_VERSION

    _have_lxml = True

    if LXML_VERSION < (5, 0):
        # Until lxml 5.0, 'resolve_entities' defaulted to True, so an untrusted
        # document declaring an external entity could pull in local files. 5.0+
        # defaults to "internal" (external refused, internal still expanded), so
        # the bare lxml XMLParser is already safe there and we only override on
        # older lxml. False is the closest older lxml offers; it also stops
        # internal entities, but that errs on the safe side.
        _XMLParser = XMLParser

        class XMLParser(_XMLParser):
            """XMLParser subclass that doesn't resolve external entities."""

            def __init__(self, *args, **kwargs):
                kwargs.setdefault("resolve_entities", False)
                super(XMLParser, self).__init__(*args, **kwargs)

except ImportError:
    try:
        from xml.etree.cElementTree import *

        # the cElementTree version of XML function doesn't support
        # the optional 'parser' keyword argument
        from xml.etree.ElementTree import XML
    except ImportError:  # pragma: no cover
        from xml.etree.ElementTree import *
    _have_lxml = False

    _Attrib = dict

    if isinstance(Element, type):
        _Element = Element
    else:
        # in py27, cElementTree.Element cannot be subclassed, so
        # we need to import the pure-python class
        from xml.etree.ElementTree import Element as _Element

    class Element(_Element):
        """Element subclass that keeps the order of attributes."""

        def __init__(self, tag, attrib=_Attrib(), **extra):
            super(Element, self).__init__(tag)
            self.attrib = _Attrib()
            if attrib:
                self.attrib.update(attrib)
            if extra:
                self.attrib.update(extra)

    def SubElement(parent, tag, attrib=_Attrib(), **extra):
        """Must override SubElement as well otherwise _elementtree.SubElement
        fails if 'parent' is a subclass of Element object.
        """
        element = parent.__class__(tag, attrib, **extra)
        parent.append(element)
        return element

    def _iterwalk(element, events, tag):
        include = tag is None or element.tag == tag
        if include and "start" in events:
            yield ("start", element)
        for e in element:
            for item in _iterwalk(e, events, tag):
                yield item
        if include:
            yield ("end", element)

    def iterwalk(element_or_tree, events=("end",), tag=None):
        """A tree walker that generates events from an existing tree as
        if it was parsing XML data with iterparse().
        Drop-in replacement for lxml.etree.iterwalk.
        """
        if iselement(element_or_tree):
            element = element_or_tree
        else:
            element = element_or_tree.getroot()
        if tag == "*":
            tag = None
        for item in _iterwalk(element, events, tag):
            yield item

    _ElementTree = ElementTree

    class ElementTree(_ElementTree):
        """ElementTree subclass that adds 'pretty_print' and 'doctype'
        arguments to the 'write' method.
        Currently these are only supported for the default XML serialization
        'method', and not also for "html" or "text", for these are delegated
        to the base class.
        """

        def write(
            self,
            file_or_filename,
            encoding=None,
            xml_declaration=False,
            method=None,
            doctype=None,
            pretty_print=False,
        ):
            if method and method != "xml":
                # delegate to super-class
                super(ElementTree, self).write(
                    file_or_filename,
                    encoding=encoding,
                    xml_declaration=xml_declaration,
                    method=method,
                )
                return

            if encoding is not None and encoding.lower() == "unicode":
                if xml_declaration:
                    raise ValueError(
                        "Serialisation to unicode must not request an XML declaration"
                    )
                write_declaration = False
                encoding = "unicode"
            elif xml_declaration is None:
                # by default, write an XML declaration only for non-standard encodings
                write_declaration = encoding is not None and encoding.upper() not in (
                    "ASCII",
                    "UTF-8",
                    "UTF8",
                    "US-ASCII",
                )
            else:
                write_declaration = xml_declaration

            if encoding is None:
                encoding = "ASCII"

            if pretty_print:
                # NOTE this will modify the tree in-place
                _indent(self._root)

            with _get_writer(file_or_filename, encoding) as write:
                if write_declaration:
                    write(XML_DECLARATION % encoding.upper())
                    if pretty_print:
                        write("\n")
                if doctype:
                    write(_tounicode(doctype))
                    if pretty_print:
                        write("\n")

                qnames, namespaces = _namespaces(self._root)
                _serialize_xml(write, self._root, qnames, namespaces)

    import io

    def tostring(
        element,
        encoding=None,
        xml_declaration=None,
        method=None,
        doctype=None,
        pretty_print=False,
    ):
        """Custom 'tostring' function that uses our ElementTree subclass, with
        pretty_print support.
        """
        stream = io.StringIO() if encoding == "unicode" else io.BytesIO()
        ElementTree(element).write(
            stream,
            encoding=encoding,
            xml_declaration=xml_declaration,
            method=method,
            doctype=doctype,
            pretty_print=pretty_print,
        )
        return stream.getvalue()

    # serialization support

    import re

    # Valid XML strings can include any Unicode character, excluding control
    # characters, the surrogate blocks, FFFE, and FFFF:
    #   Char ::= #x9 | #xA | #xD | [#x20-#xD7FF] | [#xE000-#xFFFD] | [#x10000-#x10FFFF]
    # Here we reversed the pattern to match only the invalid characters.
    _invalid_xml_string = re.compile(
        "[\u0000-\u0008\u000b-\u000c\u000e-\u001f\ud800-\udfff\ufffe-\uffff]"
    )

    def _tounicode(s):
        """Test if a string is valid user input and decode it to unicode string
        using ASCII encoding if it's a bytes string.
        Reject all bytes/unicode input that contains non-XML characters.
        Reject all bytes input that contains non-ASCII characters.
        """
        try:
            s = tostr(s, encoding="ascii", errors="strict")
        except UnicodeDecodeError:
            raise ValueError(
                "Bytes strings can only contain ASCII characters. "
                "Use unicode strings for non-ASCII characters."
            )
        except AttributeError:
            _raise_serialization_error(s)
        if s and _invalid_xml_string.search(s):
            raise ValueError(
                "All strings must be XML compatible: Unicode or ASCII, "
                "no NULL bytes or control characters"
            )
        return s

    import contextlib

    @contextlib.contextmanager
    def _get_writer(file_or_filename, encoding):
        # returns text write method and release all resources after using
        try:
            write = file_or_filename.write
        except AttributeError:
            # file_or_filename is a file name
            f = open(
                file_or_filename,
                "w",
                encoding="utf-8" if encoding == "unicode" else encoding,
                errors="xmlcharrefreplace",
            )
            with f:
                yield f.write
        else:
            # file_or_filename is a file-like object
            # encoding determines if it is a text or binary writer
            if encoding == "unicode":
                # use a text writer as is
                yield write
            else:
                # wrap a binary writer with TextIOWrapper
                detach_buffer = False
                if isinstance(file_or_filename, io.BufferedIOBase):
                    buf = file_or_filename
                elif isinstance(file_or_filename, io.RawIOBase):
                    buf = io.BufferedWriter(file_or_filename)
                    detach_buffer = True
                else:
                    # This is to handle passed objects that aren't in the
                    # IOBase hierarchy, but just have a write method
                    buf = io.BufferedIOBase()
                    buf.writable = lambda: True
                    buf.write = write
                    try:
                        # TextIOWrapper uses this methods to determine
                        # if BOM (for UTF-16, etc) should be added
                        buf.seekable = file_or_filename.seekable
                        buf.tell = file_or_filename.tell
                    except AttributeError:
                        pass
                wrapper = io.TextIOWrapper(
                    buf,
                    encoding=encoding,
                    errors="xmlcharrefreplace",
                    newline="\n",
                )
                try:
                    yield wrapper.write
                finally:
                    # Keep the original file open when the TextIOWrapper and
                    # the BufferedWriter are destroyed
                    wrapper.detach()
                    if detach_buffer:
                        buf.detach()

    from xml.etree.ElementTree import _namespace_map

    def _namespaces(elem):
        # identify namespaces used in this tree

        # maps qnames to *encoded* prefix:local names
        qnames = {None: None}

        # maps uri:s to prefixes
        namespaces = {}

        def add_qname(qname):
            # calculate serialized qname representation
            try:
                qname = _tounicode(qname)
                if qname[:1] == "{":
                    uri, tag = qname[1:].rsplit("}", 1)
                    prefix = namespaces.get(uri)
                    if prefix is None:
                        prefix = _namespace_map.get(uri)
                        if prefix is None:
                            prefix = "ns%d" % len(namespaces)
                        else:
                            prefix = _tounicode(prefix)
                        if prefix != "xml":
                            namespaces[uri] = prefix
                    if prefix:
                        qnames[qname] = "%s:%s" % (prefix, tag)
                    else:
                        qnames[qname] = tag  # default element
                else:
                    qnames[qname] = qname
            except TypeError:
                _raise_serialization_error(qname)

        # populate qname and namespaces table
        for elem in elem.iter():
            tag = elem.tag
            if isinstance(tag, QName):
                if tag.text not in qnames:
                    add_qname(tag.text)
            elif isinstance(tag, str):
                if tag not in qnames:
                    add_qname(tag)
            elif tag is not None and tag is not Comment and tag is not PI:
                _raise_serialization_error(tag)
            for key, value in elem.items():
                if isinstance(key, QName):
                    key = key.text
                if key not in qnames:
                    add_qname(key)
                if isinstance(value, QName) and value.text not in qnames:
                    add_qname(value.text)
            text = elem.text
            if isinstance(text, QName) and text.text not in qnames:
                add_qname(text.text)
        return qnames, namespaces

    def _serialize_xml(write, elem, qnames, namespaces, **kwargs):
        tag = elem.tag
        text = elem.text
        if tag is Comment:
            write("<!--%s-->" % _tounicode(text))
        elif tag is ProcessingInstruction:
            write("<?%s?>" % _tounicode(text))
        else:
            tag = qnames[_tounicode(tag) if tag is not None else None]
            if tag is None:
                if text:
                    write(_escape_cdata(text))
                for e in elem:
                    _serialize_xml(write, e, qnames, None)
            else:
                write("<" + tag)
                if namespaces:
                    for uri, prefix in sorted(
                        namespaces.items(), key=lambda x: x[1]
                    ):  # sort on prefix
                        if prefix:
                            prefix = ":" + prefix
                        write(' xmlns%s="%s"' % (prefix, _escape_attrib(uri)))
                attrs = elem.attrib
                if attrs:
                    # try to keep existing attrib order
                    if len(attrs) <= 1 or type(attrs) is _Attrib:
                        items = attrs.items()
                    else:
                        # if plain dict, use lexical order
                        items = sorted(attrs.items())
                    for k, v in items:
                        if isinstance(k, QName):
                            k = _tounicode(k.text)
                        else:
                            k = _tounicode(k)
                        if isinstance(v, QName):
                            v = qnames[_tounicode(v.text)]
                        else:
                            v = _escape_attrib(v)
                        write(' %s="%s"' % (qnames[k], v))
                if text is not None or len(elem):
                    write(">")
                    if text:
                        write(_escape_cdata(text))
                    for e in elem:
                        _serialize_xml(write, e, qnames, None)
                    write("</" + tag + ">")
                else:
                    write("/>")
        if elem.tail:
            write(_escape_cdata(elem.tail))

    def _raise_serialization_error(text):
        raise TypeError("cannot serialize %r (type %s)" % (text, type(text).__name__))

    def _escape_cdata(text):
        # escape character data
        try:
            text = _tounicode(text)
            # it's worth avoiding do-nothing calls for short strings
            if "&" in text:
                text = text.replace("&", "&amp;")
            if "<" in text:
                text = text.replace("<", "&lt;")
            if ">" in text:
                text = text.replace(">", "&gt;")
            return text
        except (TypeError, AttributeError):
            _raise_serialization_error(text)

    def _escape_attrib(text):
        # escape attribute value
        try:
            text = _tounicode(text)
            if "&" in text:
                text = text.replace("&", "&amp;")
            if "<" in text:
                text = text.replace("<", "&lt;")
            if ">" in text:
                text = text.replace(">", "&gt;")
            if '"' in text:
                text = text.replace('"', "&quot;")
            if "\n" in text:
                text = text.replace("\n", "&#10;")
            return text
        except (TypeError, AttributeError):
            _raise_serialization_error(text)

    def _indent(elem, level=0):
        # From http://effbot.org/zone/element-lib.htm#prettyprint
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                _indent(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i
<|endoftext|>"""Generic tools for working with trees."""

from math import ceil, log


def build_n_ary_tree(leaves, n):
    """Build N-ary tree from sequence of leaf nodes.

    Return a list of lists where each non-leaf node is a list containing
    max n nodes.
    """
    if not leaves:
        return []

    assert n > 1

    depth = ceil(log(len(leaves), n))

    if depth <= 1:
        return list(leaves)

    # Fully populate complete subtrees of root until we have enough leaves left
    root = []
    unassigned = None
    full_step = n ** (depth - 1)
    for i in range(0, len(leaves), full_step):
        subtree = leaves[i : i + full_step]
        if len(subtree) < full_step:
            unassigned = subtree
            break
        while len(subtree) > n:
            subtree = [subtree[k : k + n] for k in range(0, len(subtree), n)]
        root.append(subtree)

    if unassigned:
        # Recurse to fill the last subtree, which is the only partially populated one
        subtree = build_n_ary_tree(unassigned, n)
        if len(subtree) <= n - len(root):
            # replace last subtree with its children if they can still fit
            root.extend(subtree)
        else:
            root.append(subtree)
        assert len(root) <= n

    return root
<|endoftext|>from fontTools.misc.textTools import bytechr, byteord, bytesjoin, tobytes, tostr
from fontTools.misc import eexec
from .psOperators import (
    PSOperators,
    ps_StandardEncoding,
    ps_array,
    ps_boolean,
    ps_dict,
    ps_integer,
    ps_literal,
    ps_mark,
    ps_name,
    ps_operator,
    ps_procedure,
    ps_procmark,
    ps_real,
    ps_string,
)
import re
from collections.abc import Callable
from string import whitespace
import logging

log = logging.getLogger(__name__)

ps_special = b"()<>[]{}%"  # / is one too, but we take care of that one differently

skipwhiteRE = re.compile(bytesjoin([b"[", whitespace, b"]*"]))
endofthingPat = bytesjoin([b"[^][(){}<>/%", whitespace, b"]*"])
endofthingRE = re.compile(endofthingPat)
commentRE = re.compile(b"%[^\n\r]*")

# XXX This not entirely correct as it doesn't allow *nested* embedded parens:
stringPat = rb"""
	\(
		(
			(
				[^()]*   \   [()]
			)
			|
			(
				[^()]*  \(   [^()]*  \)
			)
		)*
		[^()]*
	\)
"""
stringPat = b"".join(stringPat.split())
stringRE = re.compile(stringPat)

hexstringRE = re.compile(bytesjoin([b"<[", whitespace, b"0-9A-Fa-f]*>"]))


class PSTokenError(Exception):
    pass


class PSError(Exception):
    pass


class PSTokenizer(object):
    def __init__(self, buf=b"", encoding="ascii"):
        # Force self.buf to be a byte string
        buf = tobytes(buf)
        self.buf = buf
        self.len = len(buf)
        self.pos = 0
        self.closed = False
        self.encoding = encoding

    def read(self, n=-1):
        """Read at most 'n' bytes from the buffer, or less if the read
        hits EOF before obtaining 'n' bytes.
        If 'n' is negative or omitted, read all data until EOF is reached.
        """
        if self.closed:
            raise ValueError("I/O operation on closed file")
        if n is None or n < 0:
            newpos = self.len
        else:
            newpos = min(self.pos + n, self.len)
        r = self.buf[self.pos : newpos]
        self.pos = newpos
        return r

    def close(self):
        if not self.closed:
            self.closed = True
            del self.buf, self.pos

    def getnexttoken(
        self,
        # localize some stuff, for performance
        len=len,
        ps_special=ps_special,
        stringmatch=stringRE.match,
        hexstringmatch=hexstringRE.match,
        commentmatch=commentRE.match,
        endmatch=endofthingRE.match,
    ):
        self.skipwhite()
        if self.pos >= self.len:
            return None, None
        pos = self.pos
        buf = self.buf
        char = bytechr(byteord(buf[pos]))
        if char in ps_special:
            if char in b"{}[]":
                tokentype = "do_special"
                token = char
            elif char == b"%":
                tokentype = "do_comment"
                _, nextpos = commentmatch(buf, pos).span()
                token = buf[pos:nextpos]
            elif char == b"(":
                tokentype = "do_string"
                m = stringmatch(buf, pos)
                if m is None:
                    raise PSTokenError("bad string at character %d" % pos)
                _, nextpos = m.span()
                token = buf[pos:nextpos]
            elif char == b"<":
                tokentype = "do_hexstring"
                m = hexstringmatch(buf, pos)
                if m is None:
                    raise PSTokenError("bad hexstring at character %d" % pos)
                _, nextpos = m.span()
                token = buf[pos:nextpos]
            else:
                raise PSTokenError("bad token at character %d" % pos)
        else:
            if char == b"/":
                tokentype = "do_literal"
                m = endmatch(buf, pos + 1)
            else:
                tokentype = ""
                m = endmatch(buf, pos)
            if m is None:
                raise PSTokenError("bad token at character %d" % pos)
            _, nextpos = m.span()
            token = buf[pos:nextpos]
        self.pos = pos + len(token)
        token = tostr(token, encoding=self.encoding)
        return tokentype, token

    def skipwhite(self, whitematch=skipwhiteRE.match):
        _, nextpos = whitematch(self.buf, self.pos).span()
        self.pos = nextpos

    def starteexec(self):
        self.pos = self.pos + 1
        self.dirtybuf = self.buf[self.pos :]
        self.buf, R = eexec.decrypt(self.dirtybuf, 55665)
        self.len = len(self.buf)
        self.pos = 4

    def stopeexec(self):
        if not hasattr(self, "dirtybuf"):
            return
        self.buf = self.dirtybuf
        del self.dirtybuf


class PSInterpreter(PSOperators):
    def __init__(self, encoding="ascii"):
        systemdict = {}
        userdict = {}
        self.encoding = encoding
        self.dictstack = [systemdict, userdict]
        self.stack = []
        self.proclevel = 0
        self.procmark = ps_procmark()
        self.fillsystemdict()

    def fillsystemdict(self):
        systemdict = self.dictstack[0]
        systemdict["["] = systemdict["mark"] = self.mark = ps_mark()
        systemdict["]"] = ps_operator("]", self.do_makearray)
        systemdict["true"] = ps_boolean(1)
        systemdict["false"] = ps_boolean(0)
        systemdict["StandardEncoding"] = ps_array(ps_StandardEncoding)
        systemdict["FontDirectory"] = ps_dict({})
        self.suckoperators(systemdict, self.__class__)

    def suckoperators(self, systemdict, klass):
        for name in dir(klass):
            attr = getattr(self, name)
            if isinstance(attr, Callable) and name[:3] == "ps_":
                name = name[3:]
                systemdict[name] = ps_operator(name, attr)
        for baseclass in klass.__bases__:
            self.suckoperators(systemdict, baseclass)

    def interpret(self, data, getattr=getattr):
        tokenizer = self.tokenizer = PSTokenizer(data, self.encoding)
        getnexttoken = tokenizer.getnexttoken
        do_token = self.do_token
        handle_object = self.handle_object
        try:
            while 1:
                tokentype, token = getnexttoken()
                if not token:
                    break
                if tokentype:
                    handler = getattr(self, tokentype)
                    object = handler(token)
                else:
                    object = do_token(token)
                if object is not None:
                    handle_object(object)
            tokenizer.close()
            self.tokenizer = None
        except:
            if self.tokenizer is not None:
                log.debug(
                    "ps error:\n"
                    "- - - - - - -\n"
                    "%s\n"
                    ">>>\n"
                    "%s\n"
                    "- - - - - - -",
                    self.tokenizer.buf[self.tokenizer.pos - 50 : self.tokenizer.pos],
                    self.tokenizer.buf[self.tokenizer.pos : self.tokenizer.pos + 50],
                )
            raise

    def handle_object(self, object):
        if not (self.proclevel or object.literal or object.type == "proceduretype"):
            if object.type != "operatortype":
                object = self.resolve_name(object.value)
            if object.literal:
                self.push(object)
            else:
                if object.type == "proceduretype":
                    self.call_procedure(object)
                else:
                    object.function()
        else:
            self.push(object)

    def call_procedure(self, proc):
        handle_object = self.handle_object
        for item in proc.value:
            handle_object(item)

    def resolve_name(self, name):
        dictstack = self.dictstack
        for i in range(len(dictstack) - 1, -1, -1):
            if name in dictstack[i]:
                return dictstack[i][name]
        raise PSError("name error: " + str(name))

    def do_token(
        self,
        token,
        int=int,
        float=float,
        ps_name=ps_name,
        ps_integer=ps_integer,
        ps_real=ps_real,
    ):
        try:
            num = int(token)
        except (ValueError, OverflowError):
            try:
                num = float(token)
            except (ValueError, OverflowError):
                if "#" in token:
                    hashpos = token.find("#")
                    try:
                        base = int(token[:hashpos])
                        num = int(token[hashpos + 1 :], base)
                    except (ValueError, OverflowError):
                        return ps_name(token)
                    else:
                        return ps_integer(num)
                else:
                    return ps_name(token)
            else:
                return ps_real(num)
        else:
            return ps_integer(num)

    def do_comment(self, token):
        pass

    def do_literal(self, token):
        return ps_literal(token[1:])

    def do_string(self, token):
        return ps_string(token[1:-1])

    def do_hexstring(self, token):
        hexStr = "".join(token[1:-1].split())
        if len(hexStr) % 2:
            hexStr = hexStr + "0"
        cleanstr = []
        for i in range(0, len(hexStr), 2):
            cleanstr.append(chr(int(hexStr[i : i + 2], 16)))
        cleanstr = "".join(cleanstr)
        return ps_string(cleanstr)

    def do_special(self, token):
        if token == "{":
            self.proclevel = self.proclevel + 1
            return self.procmark
        elif token == "}":
            proc = []
            while 1:
                topobject = self.pop()
                if topobject == self.procmark:
                    break
                proc.append(topobject)
            self.proclevel = self.proclevel - 1
            proc.reverse()
            return ps_procedure(proc)
        elif token == "[":
            return self.mark
        elif token == "]":
            return ps_name("]")
        else:
            raise PSTokenError("huh?")

    def push(self, object):
        self.stack.append(object)

    def pop(self, *types):
        stack = self.stack
        if not stack:
            raise PSError("stack underflow")
        object = stack[-1]
        if types:
            if object.type not in types:
                raise PSError(
                    "typecheck, expected %s, found %s" % (repr(types), object.type)
                )
        del stack[-1]
        return object

    def do_makearray(self):
        array = []
        while 1:
            topobject = self.pop()
            if topobject == self.mark:
                break
            array.append(topobject)
        array.reverse()
        self.push(ps_array(array))

    def close(self):
        """Remove circular references."""
        del self.stack
        del self.dictstack


def unpack_item(item):
    tp = type(item.value)
    if tp == dict:
        newitem = {}
        for key, value in item.value.items():
            newitem[key] = unpack_item(value)
    elif tp == list:
        newitem = [None] * len(item.value)
        for i in range(len(item.value)):
            newitem[i] = unpack_item(item.value[i])
        if item.type == "proceduretype":
            newitem = tuple(newitem)
    else:
        newitem = item.value
    return newitem


def suckfont(data, encoding="ascii"):
    m = re.search(rb"/FontName\s+/([^ \t\n\r]+)\s+def", data)
    if m:
        fontName = m.group(1)
        fontName = fontName.decode()
    else:
        fontName = None
    interpreter = PSInterpreter(encoding=encoding)
    interpreter.interpret(
        b"/Helvetica 4 dict dup /Encoding StandardEncoding put definefont pop"
    )
    interpreter.interpret(data)
    fontdir = interpreter.dictstack[0]["FontDirectory"].value
    if fontName in fontdir:
        rawfont = fontdir[fontName]
    else:
        # fall back, in case fontName wasn't found
        fontNames = list(fontdir.keys())
        if len(fontNames) > 1:
            fontNames.remove("Helvetica")
        fontNames.sort()
        rawfont = fontdir[fontNames[0]]
    interpreter.close()
    return unpack_item(rawfont)
<|endoftext|>"""Python 2/3 compat layer leftovers."""

import decimal as _decimal
import math as _math
import warnings
from contextlib import redirect_stderr, redirect_stdout
from io import BytesIO
from io import StringIO as UnicodeIO
from types import SimpleNamespace

from .textTools import Tag, bytechr, byteord, bytesjoin, strjoin, tobytes, tostr

warnings.warn(
    "The py23 module has been deprecated and will be removed in a future release. "
    "Please update your code.",
    DeprecationWarning,
)

__all__ = [
    "basestring",
    "bytechr",
    "byteord",
    "BytesIO",
    "bytesjoin",
    "open",
    "Py23Error",
    "range",
    "RecursionError",
    "round",
    "SimpleNamespace",
    "StringIO",
    "strjoin",
    "Tag",
    "tobytes",
    "tostr",
    "tounicode",
    "unichr",
    "unicode",
    "UnicodeIO",
    "xrange",
    "zip",
]


class Py23Error(NotImplementedError):
    pass


RecursionError = RecursionError
StringIO = UnicodeIO

basestring = str
isclose = _math.isclose
isfinite = _math.isfinite
open = open
range = range
round = round3 = round
unichr = chr
unicode = str
zip = zip

tounicode = tostr


def xrange(*args, **kwargs):
    raise Py23Error("'xrange' is not defined. Use 'range' instead.")


def round2(number, ndigits=None):
    """
    Implementation of Python 2 built-in round() function.
    Rounds a number to a given precision in decimal digits (default
    0 digits). The result is a floating point number. Values are rounded
    to the closest multiple of 10 to the power minus ndigits; if two
    multiples are equally close, rounding is done away from 0.
    ndigits may be negative.
    See Python 2 documentation:
    https://docs.python.org/2/library/functions.html?highlight=round#round
    """
    if ndigits is None:
        ndigits = 0

    if ndigits < 0:
        exponent = 10 ** (-ndigits)
        quotient, remainder = divmod(number, exponent)
        if remainder >= exponent // 2 and number >= 0:
            quotient += 1
        return float(quotient * exponent)
    else:
        exponent = _decimal.Decimal("10") ** (-ndigits)

        d = _decimal.Decimal.from_float(number).quantize(
            exponent, rounding=_decimal.ROUND_HALF_UP
        )

        return float(d)
<|endoftext|>"""Routines for calculating bounding boxes, point in rectangle calculations and
so on.
"""

from fontTools.misc.roundTools import otRound
from fontTools.misc.vector import Vector as _Vector
import math
import warnings


def calcBounds(array):
    """Calculate the bounding rectangle of a 2D points array.

    Args:
        array: A sequence of 2D tuples.

    Returns:
        A four-item tuple representing the bounding rectangle ``(xMin, yMin, xMax, yMax)``.
    """
    if not array:
        return 0, 0, 0, 0
    xs = [x for x, y in array]
    ys = [y for x, y in array]
    return min(xs), min(ys), max(xs), max(ys)


def calcIntBounds(array, round=otRound):
    """Calculate the integer bounding rectangle of a 2D points array.

    Values are rounded to closest integer towards ``+Infinity`` using the
    :func:`fontTools.misc.fixedTools.otRound` function by default, unless
    an optional ``round`` function is passed.

    Args:
        array: A sequence of 2D tuples.
        round: A rounding function of type ``f(x: float) -> int``.

    Returns:
        A four-item tuple of integers representing the bounding rectangle:
        ``(xMin, yMin, xMax, yMax)``.
    """
    return tuple(round(v) for v in calcBounds(array))


def updateBounds(bounds, p, min=min, max=max):
    """Add a point to a bounding rectangle.

    Args:
        bounds: A bounding rectangle expressed as a tuple
            ``(xMin, yMin, xMax, yMax), or None``.
        p: A 2D tuple representing a point.
        min,max: functions to compute the minimum and maximum.

    Returns:
        The updated bounding rectangle ``(xMin, yMin, xMax, yMax)``.
    """
    x, y = p
    if bounds is None:
        return x, y, x, y
    xMin, yMin, xMax, yMax = bounds
    return min(xMin, x), min(yMin, y), max(xMax, x), max(yMax, y)


def pointInRect(p, rect):
    """Test if a point is inside a bounding rectangle.

    Args:
        p: A 2D tuple representing a point.
        rect: A bounding rectangle expressed as a tuple
            ``(xMin, yMin, xMax, yMax)``.

    Returns:
        ``True`` if the point is inside the rectangle, ``False`` otherwise.
    """
    x, y = p
    xMin, yMin, xMax, yMax = rect
    return (xMin <= x <= xMax) and (yMin <= y <= yMax)


def pointsInRect(array, rect):
    """Determine which points are inside a bounding rectangle.

    Args:
        array: A sequence of 2D tuples.
        rect: A bounding rectangle expressed as a tuple
            ``(xMin, yMin, xMax, yMax)``.

    Returns:
        A list containing the points inside the rectangle.
    """
    if len(array) < 1:
        return []
    xMin, yMin, xMax, yMax = rect
    return [(xMin <= x <= xMax) and (yMin <= y <= yMax) for x, y in array]


def vectorLength(vector):
    """Calculate the length of the given vector.

    Args:
        vector: A 2D tuple.

    Returns:
        The Euclidean length of the vector.
    """
    x, y = vector
    return math.sqrt(x**2 + y**2)


def asInt16(array):
    """Round a list of floats to 16-bit signed integers.

    Args:
        array: List of float values.

    Returns:
        A list of rounded integers.
    """
    return [int(math.floor(i + 0.5)) for i in array]


def normRect(rect):
    """Normalize a bounding box rectangle.

    This function "turns the rectangle the right way up", so that the following
    holds::

        xMin <= xMax and yMin <= yMax

    Args:
        rect: A bounding rectangle expressed as a tuple
            ``(xMin, yMin, xMax, yMax)``.

    Returns:
        A normalized bounding rectangle.
    """
    xMin, yMin, xMax, yMax = rect
    return min(xMin, xMax), min(yMin, yMax), max(xMin, xMax), max(yMin, yMax)


def scaleRect(rect, x, y):
    """Scale a bounding box rectangle.

    Args:
        rect: A bounding rectangle expressed as a tuple
            ``(xMin, yMin, xMax, yMax)``.
        x: Factor to scale the rectangle along the X axis.
        Y: Factor to scale the rectangle along the Y axis.

    Returns:
        A scaled bounding rectangle.
    """
    xMin, yMin, xMax, yMax = rect
    return xMin * x, yMin * y, xMax * x, yMax * y


def offsetRect(rect, dx, dy):
    """Offset a bounding box rectangle.

    Args:
        rect: A bounding rectangle expressed as a tuple
            ``(xMin, yMin, xMax, yMax)``.
        dx: Amount to offset the rectangle along the X axis.
        dY: Amount to offset the rectangle along the Y axis.

    Returns:
        An offset bounding rectangle.
    """
    xMin, yMin, xMax, yMax = rect
    return xMin + dx, yMin + dy, xMax + dx, yMax + dy


def insetRect(rect, dx, dy):
    """Inset a bounding box rectangle on all sides.

    Args:
        rect: A bounding rectangle expressed as a tuple
            ``(xMin, yMin, xMax, yMax)``.
        dx: Amount to inset the rectangle along the X axis.
        dY: Amount to inset the rectangle along the Y axis.

    Returns:
        An inset bounding rectangle.
    """
    xMin, yMin, xMax, yMax = rect
    return xMin + dx, yMin + dy, xMax - dx, yMax - dy


def sectRect(rect1, rect2):
    """Test for rectangle-rectangle intersection.

    Args:
        rect1: First bounding rectangle, expressed as tuples
            ``(xMin, yMin, xMax, yMax)``.
        rect2: Second bounding rectangle.

    Returns:
        A boolean and a rectangle.
        If the input rectangles intersect, returns ``True`` and the intersecting
        rectangle. Returns ``False`` and ``(0, 0, 0, 0)`` if the input
        rectangles don't intersect.
    """
    xMin1, yMin1, xMax1, yMax1 = rect1
    xMin2, yMin2, xMax2, yMax2 = rect2
    xMin, yMin, xMax, yMax = (
        max(xMin1, xMin2),
        max(yMin1, yMin2),
        min(xMax1, xMax2),
        min(yMax1, yMax2),
    )
    if xMin >= xMax or yMin >= yMax:
        return False, (0, 0, 0, 0)
    return True, (xMin, yMin, xMax, yMax)


def unionRect(rect1, rect2):
    """Determine union of bounding rectangles.

    Args:
        rect1: First bounding rectangle, expressed as tuples
            ``(xMin, yMin, xMax, yMax)``.
        rect2: Second bounding rectangle.

    Returns:
        The smallest rectangle in which both input rectangles are fully
        enclosed.
    """
    xMin1, yMin1, xMax1, yMax1 = rect1
    xMin2, yMin2, xMax2, yMax2 = rect2
    xMin, yMin, xMax, yMax = (
        min(xMin1, xMin2),
        min(yMin1, yMin2),
        max(xMax1, xMax2),
        max(yMax1, yMax2),
    )
    return (xMin, yMin, xMax, yMax)


def rectCenter(rect):
    """Determine rectangle center.

    Args:
        rect: Bounding rectangle, expressed as tuples
            ``(xMin, yMin, xMax, yMax)``.

    Returns:
        A 2D tuple representing the point at the center of the rectangle.
    """
    xMin, yMin, xMax, yMax = rect
    return (xMin + xMax) / 2, (yMin + yMax) / 2


def rectArea(rect):
    """Determine rectangle area.

    Args:
        rect: Bounding rectangle, expressed as tuples
            ``(xMin, yMin, xMax, yMax)``.

    Returns:
        The area of the rectangle.
    """
    xMin, yMin, xMax, yMax = rect
    return (yMax - yMin) * (xMax - xMin)


def intRect(rect):
    """Round a rectangle to integer values.

    Guarantees that the resulting rectangle is NOT smaller than the original.

    Args:
        rect: Bounding rectangle, expressed as tuples
            ``(xMin, yMin, xMax, yMax)``.

    Returns:
        A rounded bounding rectangle.
    """
    xMin, yMin, xMax, yMax = rect
    xMin = int(math.floor(xMin))
    yMin = int(math.floor(yMin))
    xMax = int(math.ceil(xMax))
    yMax = int(math.ceil(yMax))
    return (xMin, yMin, xMax, yMax)


def quantizeRect(rect, factor=1):
    """
    >>> bounds = (72.3, -218.4, 1201.3, 919.1)
    >>> quantizeRect(bounds)
    (72, -219, 1202, 920)
    >>> quantizeRect(bounds, factor=10)
    (70, -220, 1210, 920)
    >>> quantizeRect(bounds, factor=100)
    (0, -300, 1300, 1000)
    """
    if factor < 1:
        raise ValueError(f"Expected quantization factor >= 1, found: {factor!r}")
    xMin, yMin, xMax, yMax = normRect(rect)
    return (
        int(math.floor(xMin / factor) * factor),
        int(math.floor(yMin / factor) * factor),
        int(math.ceil(xMax / factor) * factor),
        int(math.ceil(yMax / factor) * factor),
    )


class Vector(_Vector):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "fontTools.misc.arrayTools.Vector has been deprecated, please use "
            "fontTools.misc.vector.Vector instead.",
            DeprecationWarning,
        )


def pairwise(iterable, reverse=False):
    """Iterate over current and next items in iterable.

    Args:
        iterable: An iterable
        reverse: If true, iterate in reverse order.

    Returns:
        A iterable yielding two elements per iteration.

    Example:

        >>> tuple(pairwise([]))
        ()
        >>> tuple(pairwise([], reverse=True))
        ()
        >>> tuple(pairwise([0]))
        ((0, 0),)
        >>> tuple(pairwise([0], reverse=True))
        ((0, 0),)
        >>> tuple(pairwise([0, 1]))
        ((0, 1), (1, 0))
        >>> tuple(pairwise([0, 1], reverse=True))
        ((1, 0), (0, 1))
        >>> tuple(pairwise([0, 1, 2]))
        ((0, 1), (1, 2), (2, 0))
        >>> tuple(pairwise([0, 1, 2], reverse=True))
        ((2, 1), (1, 0), (0, 2))
        >>> tuple(pairwise(['a', 'b', 'c', 'd']))
        (('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'a'))
        >>> tuple(pairwise(['a', 'b', 'c', 'd'], reverse=True))
        (('d', 'c'), ('c', 'b'), ('b', 'a'), ('a', 'd'))
    """
    if not iterable:
        return
    if reverse:
        it = reversed(iterable)
    else:
        it = iter(iterable)
    first = next(it, None)
    a = first
    for b in it:
        yield (a, b)
        a = b
    yield (a, first)


def _test():
    """
    >>> import math
    >>> calcBounds([])
    (0, 0, 0, 0)
    >>> calcBounds([(0, 40), (0, 100), (50, 50), (80, 10)])
    (0, 10, 80, 100)
    >>> updateBounds((0, 0, 0, 0), (100, 100))
    (0, 0, 100, 100)
    >>> pointInRect((50, 50), (0, 0, 100, 100))
    True
    >>> pointInRect((0, 0), (0, 0, 100, 100))
    True
    >>> pointInRect((100, 100), (0, 0, 100, 100))
    True
    >>> not pointInRect((101, 100), (0, 0, 100, 100))
    True
    >>> list(pointsInRect([(50, 50), (0, 0), (100, 100), (101, 100)], (0, 0, 100, 100)))
    [True, True, True, False]
    >>> vectorLength((3, 4))
    5.0
    >>> vectorLength((1, 1)) == math.sqrt(2)
    True
    >>> list(asInt16([0, 0.1, 0.5, 0.9]))
    [0, 0, 1, 1]
    >>> normRect((0, 10, 100, 200))
    (0, 10, 100, 200)
    >>> normRect((100, 200, 0, 10))
    (0, 10, 100, 200)
    >>> scaleRect((10, 20, 50, 150), 1.5, 2)
    (15.0, 40, 75.0, 300)
    >>> offsetRect((10, 20, 30, 40), 5, 6)
    (15, 26, 35, 46)
    >>> insetRect((10, 20, 50, 60), 5, 10)
    (15, 30, 45, 50)
    >>> insetRect((10, 20, 50, 60), -5, -10)
    (5, 10, 55, 70)
    >>> intersects, rect = sectRect((0, 10, 20, 30), (0, 40, 20, 50))
    >>> not intersects
    True
    >>> intersects, rect = sectRect((0, 10, 20, 30), (5, 20, 35, 50))
    >>> intersects
    1
    >>> rect
    (5, 20, 20, 30)
    >>> unionRect((0, 10, 20, 30), (0, 40, 20, 50))
    (0, 10, 20, 50)
    >>> rectCenter((0, 0, 100, 200))
    (50.0, 100.0)
    >>> rectCenter((0, 0, 100, 199.0))
    (50.0, 99.5)
    >>> intRect((0.9, 2.9, 3.1, 4.1))
    (0, 2, 4, 5)
    """


if __name__ == "__main__":
    import sys
    import doctest

    sys.exit(doctest.testmod().failed)
<|endoftext|>"""Sparse Bit Set encoding/decoding for IFT (Incremental Font Transfer).

Implements the sparse bit set format defined in the W3C IFT specification:
https://w3c.github.io/IFT/Overview.html#sparse-bit-set-decoding
"""

# Reference: https://github.com/googlefonts/fontations/blob/main/read-fonts/src/collections/int_set/sparse_bit_set.rs


import bisect
from collections import deque
from typing import Dict, Iterable, Optional, Set, Tuple

# Maximum tree height that fits within a 32-bit leaf node for each branching factor.
_BF_MAX_HEIGHT: Dict[int, int] = {2: 31, 4: 16, 8: 11, 32: 7}


class SparseBitSetDecodeError(Exception):
    pass


def decode(
    data: bytes, bias: int = 0, maxValue: int = 0xFFFFFFFF
) -> Tuple[Set[int], int]:
    """Decode a sparse bit set from binary data.

    Args:
        data: bytes-like object containing the sparse bit set encoding.
        bias: integer added to each decoded value.

    Returns:
        A tuple (values, bytesConsumed) where values is a set of integers
        and bytesConsumed is the number of bytes read from data.
    """
    if not data:
        raise SparseBitSetDecodeError("Empty data")

    branchFactor, height = _decodeHeader(data[0])

    maxHeight = _BF_MAX_HEIGHT[branchFactor]
    if height > maxHeight:
        raise SparseBitSetDecodeError(
            f"Height {height} exceeds max {maxHeight} for branch factor {branchFactor}"
        )

    return _decodeImpl(data, branchFactor, height, bias, maxValue)


def _decodeImpl(
    data: bytes, branchFactor: int, height: int, bias: int, maxValue: int
) -> Tuple[Set[int], int]:
    if height == 0:
        # 1 byte was used for the header.
        return (set(), 1)

    bitStream = _InputBitStream(data, branchFactor)
    result: Set[int] = set()
    # Queue entries are (startValue, depth), where startValue is the first
    # integer that could be covered by this node.
    queue: deque[Tuple[int, int]] = deque()
    queue.append((0, 1))

    while queue:
        start, depth = queue.popleft()
        bits = bitStream.next()
        if bits is None:
            raise SparseBitSetDecodeError("Unexpected end of data")

        # all bits were were zero which is a special command to completely fill
        # in all integers covered by this node.
        if bits == 0:
            exp = height - depth + 1
            nodeSize = branchFactor**exp
            fillStart = start + bias
            if fillStart > maxValue:
                continue
            fillEnd = min(maxValue, start + nodeSize - 1 + bias)
            if fillStart < 0:
                fillStart = 0
            if fillStart <= fillEnd:
                result.update(range(fillStart, fillEnd + 1))
            continue

        # Non-zero node: each set bit identifies a child/value.
        exp = height - depth
        nextNodeSize = branchFactor**exp
        while True:
            bitIndex = _trailingZeros(bits, 32)
            if bitIndex == 32:
                break
            if depth == height:
                val = start + bitIndex + bias
                if val > maxValue:
                    queue.clear()
                    break
                if val >= 0:
                    result.add(val)
            else:
                startDelta = bitIndex * nextNodeSize
                queue.append((start + startDelta, depth + 1))
            bits &= ~(1 << bitIndex)

    return (result, bitStream.bytesConsumed())


def encode(values: Iterable[int]) -> bytes:
    """Encode a set of integers as a sparse bit set.

    Tries all branching factors and returns the shortest encoding.

    Args:
        values: iterable of non-negative integers.

    Returns:
        bytes containing the sparse bit set encoding.
    """
    valuesSorted = sorted(set(values))
    if not valuesSorted:
        return _encodeHeader(2, 0)

    maxValue = valuesSorted[-1]
    valueSet = set(valuesSorted)

    best: Optional[bytes] = None
    for branchFactor in (2, 4, 8, 32):
        height = _treeHeight(branchFactor, maxValue)
        if height > _BF_MAX_HEIGHT[branchFactor]:
            continue
        encoded = _encodeWithBf(valueSet, branchFactor, height)
        if best is None or len(encoded) < len(best):
            best = encoded

    if best is None:
        raise ValueError(f"Cannot encode max value {maxValue}")
    return best


def _encodeHeader(branchFactor: int, height: int) -> bytes:
    branchFactorToId = {2: 0, 4: 1, 8: 2, 32: 3}
    return bytes([(height << 2) | branchFactorToId[branchFactor]])


def _decodeHeader(headerByte: int) -> Tuple[int, int]:
    id = headerByte & 0x03
    idToBranchFactor = {0: 2, 1: 4, 2: 8, 3: 32}
    height = (headerByte >> 2) & 0x1F
    return idToBranchFactor[id], height


def _treeHeight(branchFactor: int, maxValue: int) -> int:
    """Return the minimum tree height needed to represent maxValue."""
    height = 1
    capacity = branchFactor
    while capacity <= maxValue:
        capacity *= branchFactor
        height += 1
    return height


def _encodeWithBf(valueSet: Set[int], branchFactor: int, height: int) -> bytes:
    if height == 0:
        return _encodeHeader(branchFactor, 0)

    # Build layers bottom-up: layer 0 = leaves (individual values),
    # each higher layer groups bf children into one parent bitmask.
    layers: list[Dict[int, int]] = [{}]  # list of dicts: nodeIndex -> bitmask

    for v in valueSet:
        nodeIndex = v // branchFactor
        bitPos = v % branchFactor
        if nodeIndex not in layers[0]:
            layers[0][nodeIndex] = 0
        layers[0][nodeIndex] |= 1 << bitPos

    for _ in range(1, height):
        prevLayer = layers[-1]
        newLayer: Dict[int, int] = {}
        for nodeIndex, bitmask in prevLayer.items():
            parentIndex = nodeIndex // branchFactor
            bitPos = nodeIndex % branchFactor
            if parentIndex not in newLayer:
                newLayer[parentIndex] = 0
            newLayer[parentIndex] |= 1 << bitPos
        layers.append(newLayer)

    # For zero-node optimization: track count of values in sorted list.
    valuesSorted = sorted(valueSet)

    def rangeCount(lo: int, hi: int) -> int:
        return bisect.bisect_right(valuesSorted, hi) - bisect.bisect_left(
            valuesSorted, lo
        )

    # Emit nodes BFS order (root to leaves).
    # Queue entries: (nodeIndex, depthFromRoot, rangeStart, rangeEnd)
    stream = _OutputBitStream(branchFactor)
    subtreeSize = branchFactor**height
    queue: deque[Tuple[int, int, int, int]] = deque([(0, 0, 0, subtreeSize - 1)])

    while queue:
        nodeIndex, depth, rangeStart, rangeEnd = queue.popleft()
        layerIdx = height - 1 - depth  # layers[0]=leaves, layers[height-1]=root

        bitmask = (
            layers[layerIdx].get(nodeIndex, 0) if 0 <= layerIdx < len(layers) else 0
        )

        # Zero-node optimization: if entire range is filled on an INTERNAL node,
        # write 0 and skip children.  At leaf level we always write the explicit
        # bitmask so the encoding matches the reference.
        if (
            depth < height - 1
            and rangeCount(rangeStart, rangeEnd) == rangeEnd - rangeStart + 1
        ):
            stream.write(0)
            continue

        stream.write(bitmask)

        if bitmask != 0 and depth < height - 1:
            childSize = (rangeEnd - rangeStart + 1) // branchFactor
            bits = bitmask
            while bits:
                bitIndex = _trailingZeros(bits, 32)
                childIndex = nodeIndex * branchFactor + bitIndex
                childStart = rangeStart + bitIndex * childSize
                childEnd = childStart + childSize - 1
                queue.append((childIndex, depth + 1, childStart, childEnd))
                bits &= ~(1 << bitIndex)

    return _encodeHeader(branchFactor, height) + stream.toBytes()


def _trailingZeros(val: int, maxBits: int) -> int:
    if val == 0:
        return maxBits
    count = 0
    while (val & 1) == 0:
        val >>= 1
        count += 1
    return count


class _InputBitStream:
    """Reads bit nodes from a byte array, starting after the header byte."""

    def __init__(self, data: bytes, branchFactor: int):
        self.data = data
        self.branchFactor = branchFactor
        self.byteIndex = 1  # skip the header byte
        self.subIndex = 0  # bit offset within current byte (for bf=2,4)

    def next(self) -> Optional[int]:
        if self.branchFactor in (2, 4):
            if self.byteIndex >= len(self.data):
                return None
            mask = (1 << self.branchFactor) - 1
            val = (self.data[self.byteIndex] >> self.subIndex) & mask
            self.subIndex += self.branchFactor
            if self.subIndex >= 8:
                self.subIndex = 0
                self.byteIndex += 1
            return val
        elif self.branchFactor == 8:
            if self.byteIndex >= len(self.data):
                return None
            val = self.data[self.byteIndex]
            self.byteIndex += 1
            return val
        elif self.branchFactor == 32:
            if self.byteIndex + 3 >= len(self.data):
                return None
            b = self.data
            i = self.byteIndex
            val = b[i] | (b[i + 1] << 8) | (b[i + 2] << 16) | (b[i + 3] << 24)
            self.byteIndex += 4
            return val
        return None

    def bytesConsumed(self) -> int:
        return self.byteIndex + (1 if self.subIndex > 0 else 0)


class _OutputBitStream:
    """Writes bit nodes into a byte array."""

    def __init__(self, branchFactor: int):
        self.branchFactor = branchFactor
        self.data = bytearray()
        self.subIndex = 0  # bit offset within current byte (for bf=2,4)

    def write(self, value: int) -> None:
        if self.branchFactor in (2, 4):
            mask = (1 << self.branchFactor) - 1
            value &= mask
            if self.subIndex == 0:
                self.data.append(0)
            self.data[-1] |= value << self.subIndex
            self.subIndex += self.branchFactor
            if self.subIndex >= 8:
                self.subIndex = 0
        elif self.branchFactor == 8:
            self.data.append(value & 0xFF)
        elif self.branchFactor == 32:
            self.data.append(value & 0xFF)
            self.data.append((value >> 8) & 0xFF)
            self.data.append((value >> 16) & 0xFF)
            self.data.append((value >> 24) & 0xFF)

    def toBytes(self) -> bytes:
        return bytes(self.data)
<|endoftext|>"""Misc dict tools."""

__all__ = ["hashdict"]


# https://stackoverflow.com/questions/1151658/python-hashable-dicts
class hashdict(dict):
    """
    hashable dict implementation, suitable for use as a key into
    other dicts.

        >>> h1 = hashdict({"apples": 1, "bananas":2})
        >>> h2 = hashdict({"bananas": 3, "mangoes": 5})
        >>> h1+h2
        hashdict(apples=1, bananas=3, mangoes=5)
        >>> d1 = {}
        >>> d1[h1] = "salad"
        >>> d1[h1]
        'salad'
        >>> d1[h2]
        Traceback (most recent call last):
        ...
        KeyError: hashdict(bananas=3, mangoes=5)

    based on answers from
       http://stackoverflow.com/questions/1151658/python-hashable-dicts

    """

    def __key(self):
        return tuple(sorted(self.items()))

    def __repr__(self):
        return "{0}({1})".format(
            self.__class__.__name__,
            ", ".join("{0}={1}".format(str(i[0]), repr(i[1])) for i in self.__key()),
        )

    def __hash__(self):
        return hash(self.__key())

    def __setitem__(self, key, value):
        raise TypeError(
            "{0} does not support item assignment".format(self.__class__.__name__)
        )

    def __delitem__(self, key):
        raise TypeError(
            "{0} does not support item assignment".format(self.__class__.__name__)
        )

    def clear(self):
        raise TypeError(
            "{0} does not support item assignment".format(self.__class__.__name__)
        )

    def pop(self, *args, **kwargs):
        raise TypeError(
            "{0} does not support item assignment".format(self.__class__.__name__)
        )

    def popitem(self, *args, **kwargs):
        raise TypeError(
            "{0} does not support item assignment".format(self.__class__.__name__)
        )

    def setdefault(self, *args, **kwargs):
        raise TypeError(
            "{0} does not support item assignment".format(self.__class__.__name__)
        )

    def update(self, *args, **kwargs):
        raise TypeError(
            "{0} does not support item assignment".format(self.__class__.__name__)
        )

    # update is not ok because it mutates the object
    # __add__ is ok because it creates a new object
    # while the new object is under construction, it's ok to mutate it
    def __add__(self, right):
        result = hashdict(self)
        dict.update(result, right)
        return result
<|endoftext|>"""Helpers for writing unit tests."""

from collections.abc import Iterable
from io import BytesIO
import os
import re
import shutil
import sys
import tempfile
from unittest import TestCase as _TestCase
from fontTools.config import Config
from fontTools.misc.textTools import tobytes
from fontTools.misc.xmlWriter import XMLWriter


def parseXML(xmlSnippet):
    """Parses a snippet of XML.

    Input can be either a single string (unicode or UTF-8 bytes), or a
    a sequence of strings.

    The result is in the same format that would be returned by
    XMLReader, but the parser imposes no constraints on the root
    element so it can be called on small snippets of TTX files.
    """
    # To support snippets with multiple elements, we add a fake root.
    reader = TestXMLReader_()
    xml = b"<root>"
    if isinstance(xmlSnippet, bytes):
        xml += xmlSnippet
    elif isinstance(xmlSnippet, str):
        xml += tobytes(xmlSnippet, "utf-8")
    elif isinstance(xmlSnippet, Iterable):
        xml += b"".join(tobytes(s, "utf-8") for s in xmlSnippet)
    else:
        raise TypeError(
            "expected string or sequence of strings; found %r"
            % type(xmlSnippet).__name__
        )
    xml += b"</root>"
    reader.parser.Parse(xml, 1)
    return reader.root[2]


def parseXmlInto(font, parseInto, xmlSnippet):
    parsed_xml = [e for e in parseXML(xmlSnippet.strip()) if not isinstance(e, str)]
    for name, attrs, content in parsed_xml:
        parseInto.fromXML(name, attrs, content, font)
    if hasattr(parseInto, "populateDefaults"):
        parseInto.populateDefaults()
    return parseInto


class FakeFont:
    def __init__(self, glyphs):
        self.glyphOrder_ = glyphs
        self.reverseGlyphOrderDict_ = {g: i for i, g in enumerate(glyphs)}
        self.lazy = False
        self.tables = {}
        self.cfg = Config()

    def __contains__(self, tag):
        return tag in self.tables

    def __getitem__(self, tag):
        return self.tables[tag]

    def __setitem__(self, tag, table):
        self.tables[tag] = table

    def get(self, tag, default=None):
        return self.tables.get(tag, default)

    def getGlyphID(self, name):
        return self.reverseGlyphOrderDict_[name]

    def getGlyphIDMany(self, lst):
        return [self.getGlyphID(gid) for gid in lst]

    def getGlyphName(self, glyphID):
        if glyphID < len(self.glyphOrder_):
            return self.glyphOrder_[glyphID]
        else:
            return "glyph%.5d" % glyphID

    def getGlyphNameMany(self, lst):
        return [self.getGlyphName(gid) for gid in lst]

    def getGlyphOrder(self):
        return self.glyphOrder_

    def getReverseGlyphMap(self):
        return self.reverseGlyphOrderDict_

    def getGlyphNames(self):
        return sorted(self.getGlyphOrder())


class TestXMLReader_(object):
    def __init__(self):
        from xml.parsers.expat import ParserCreate

        self.parser = ParserCreate()
        self.parser.StartElementHandler = self.startElement_
        self.parser.EndElementHandler = self.endElement_
        self.parser.CharacterDataHandler = self.addCharacterData_
        self.root = None
        self.stack = []

    def startElement_(self, name, attrs):
        element = (name, attrs, [])
        if self.stack:
            self.stack[-1][2].append(element)
        else:
            self.root = element
        self.stack.append(element)

    def endElement_(self, name):
        self.stack.pop()

    def addCharacterData_(self, data):
        self.stack[-1][2].append(data)


def makeXMLWriter(newlinestr="\n"):
    # don't write OS-specific new lines
    writer = XMLWriter(BytesIO(), newlinestr=newlinestr)
    # erase XML declaration
    writer.file.seek(0)
    writer.file.truncate()
    return writer


def getXML(func, ttFont=None):
    """Call the passed toXML function and return the written content as a
    list of lines (unicode strings).
    Result is stripped of XML declaration and OS-specific newline characters.
    """
    writer = makeXMLWriter()
    func(writer, ttFont)
    xml = writer.file.getvalue().decode("utf-8")
    # toXML methods must always end with a writer.newline()
    assert xml.endswith("\n")
    return xml.splitlines()


def stripVariableItemsFromTTX(
    string: str,
    ttLibVersion: bool = True,
    checkSumAdjustment: bool = True,
    modified: bool = True,
    created: bool = True,
    sfntVersion: bool = False,  # opt-in only
) -> str:
    """Strip stuff like ttLibVersion, checksums, timestamps, etc. from TTX dumps."""
    # ttlib changes with the fontTools version
    if ttLibVersion:
        string = re.sub(' ttLibVersion="[^"]+"', "", string)
    # sometimes (e.g. some subsetter tests) we don't care whether it's OTF or TTF
    if sfntVersion:
        string = re.sub(' sfntVersion="[^"]+"', "", string)
    # head table checksum and creation and mod date changes with each save.
    if checkSumAdjustment:
        string = re.sub('<checkSumAdjustment value="[^"]+"/>', "", string)
    if modified:
        string = re.sub('<modified value="[^"]+"/>', "", string)
    if created:
        string = re.sub('<created value="[^"]+"/>', "", string)
    return string


class MockFont(object):
    """A font-like object that automatically adds any looked up glyphname
    to its glyphOrder."""

    def __init__(self):
        self._glyphOrder = [".notdef"]

        class AllocatingDict(dict):
            def __missing__(reverseDict, key):
                self._glyphOrder.append(key)
                gid = len(reverseDict)
                reverseDict[key] = gid
                return gid

        self._reverseGlyphOrder = AllocatingDict({".notdef": 0})
        self.lazy = False

    def getGlyphID(self, glyph):
        gid = self._reverseGlyphOrder[glyph]
        return gid

    def getReverseGlyphMap(self):
        return self._reverseGlyphOrder

    def getGlyphName(self, gid):
        return self._glyphOrder[gid]

    def getGlyphOrder(self):
        return self._glyphOrder


class TestCase(_TestCase):
    def __init__(self, methodName):
        _TestCase.__init__(self, methodName)
        # Python 3 renamed assertRaisesRegexp to assertRaisesRegex,
        # and fires deprecation warnings if a program uses the old name.
        if not hasattr(self, "assertRaisesRegex"):
            self.assertRaisesRegex = self.assertRaisesRegexp


class DataFilesHandler(TestCase):
    def setUp(self):
        self.tempdir = None
        self.num_tempfiles = 0

    def tearDown(self):
        if self.tempdir:
            shutil.rmtree(self.tempdir)

    def getpath(self, testfile):
        folder = os.path.dirname(sys.modules[self.__module__].__file__)
        return os.path.join(folder, "data", testfile)

    def temp_dir(self):
        if not self.tempdir:
            self.tempdir = tempfile.mkdtemp()

    def temp_font(self, font_path, file_name):
        self.temp_dir()
        temppath = os.path.join(self.tempdir, file_name)
        shutil.copy2(font_path, temppath)
        return temppath
<|endoftext|>"""Exports a no-op 'cython' namespace similar to
https://github.com/cython/cython/blob/master/Cython/Shadow.py

This allows to optionally compile @cython decorated functions
(when cython is available at built time), or run the same code
as pure-python, without runtime dependency on cython module.

We only define the symbols that we use. E.g. see fontTools.cu2qu
"""

from types import SimpleNamespace


def _empty_decorator(x):
    return x


compiled = False

for name in ("double", "complex", "int"):
    globals()[name] = None

for name in ("cfunc", "inline"):
    globals()[name] = _empty_decorator

locals = lambda **_: _empty_decorator
returns = lambda _: _empty_decorator
<|endoftext|>from numbers import Number
import math
import operator
import warnings

__all__ = ["Vector"]


class Vector(tuple):
    """A math-like vector.

    Represents an n-dimensional numeric vector. ``Vector`` objects support
    vector addition and subtraction, scalar multiplication and division,
    negation, rounding, and comparison tests.
    """

    __slots__ = ()

    def __new__(cls, values, keep=False):
        if keep is not False:
            warnings.warn(
                "the 'keep' argument has been deprecated",
                DeprecationWarning,
            )
        if type(values) == Vector:
            # No need to create a new object
            return values
        return super().__new__(cls, values)

    def __repr__(self):
        return f"{self.__class__.__name__}({super().__repr__()})"

    def _vectorOp(self, other, op):
        if isinstance(other, Vector):
            assert len(self) == len(other)
            return self.__class__(op(a, b) for a, b in zip(self, other))
        if isinstance(other, Number):
            return self.__class__(op(v, other) for v in self)
        raise NotImplementedError()

    def _scalarOp(self, other, op):
        if isinstance(other, Number):
            return self.__class__(op(v, other) for v in self)
        raise NotImplementedError()

    def _unaryOp(self, op):
        return self.__class__(op(v) for v in self)

    def __add__(self, other):
        return self._vectorOp(other, operator.add)

    __radd__ = __add__

    def __sub__(self, other):
        return self._vectorOp(other, operator.sub)

    def __rsub__(self, other):
        return self._vectorOp(other, _operator_rsub)

    def __mul__(self, other):
        return self._scalarOp(other, operator.mul)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self._scalarOp(other, operator.truediv)

    def __rtruediv__(self, other):
        return self._scalarOp(other, _operator_rtruediv)

    def __pos__(self):
        return self._unaryOp(operator.pos)

    def __neg__(self):
        return self._unaryOp(operator.neg)

    def __round__(self, *, round=round):
        return self._unaryOp(round)

    def __eq__(self, other):
        if isinstance(other, list):
            # bw compat Vector([1, 2, 3]) == [1, 2, 3]
            other = tuple(other)
        return super().__eq__(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __bool__(self):
        return any(self)

    __nonzero__ = __bool__

    def __abs__(self):
        return math.sqrt(sum(x * x for x in self))

    def length(self):
        """Return the length of the vector. Equivalent to abs(vector)."""
        return abs(self)

    def normalized(self):
        """Return the normalized vector of the vector."""
        return self / abs(self)

    def dot(self, other):
        """Performs vector dot product, returning the sum of
        ``a[0] * b[0], a[1] * b[1], ...``"""
        assert len(self) == len(other)
        return sum(a * b for a, b in zip(self, other))

    # Deprecated methods/properties

    def toInt(self):
        warnings.warn(
            "the 'toInt' method has been deprecated, use round(vector) instead",
            DeprecationWarning,
        )
        return self.__round__()

    @property
    def values(self):
        warnings.warn(
            "the 'values' attribute has been deprecated, use "
            "the vector object itself instead",
            DeprecationWarning,
        )
        return list(self)

    @values.setter
    def values(self, values):
        raise AttributeError(
            "can't set attribute, the 'values' attribute has been deprecated",
        )

    def isclose(self, other: "Vector", **kwargs) -> bool:
        """Return True if the vector is close to another Vector."""
        assert len(self) == len(other)
        return all(math.isclose(a, b, **kwargs) for a, b in zip(self, other))


def _operator_rsub(a, b):
    return operator.sub(b, a)


def _operator_rtruediv(a, b):
    return operator.truediv(b, a)
<|endoftext|>"""
The `OpenType specification <https://docs.microsoft.com/en-us/typography/opentype/spec/otff#data-types>`_
defines two fixed-point data types:

``Fixed``
        A 32-bit signed fixed-point number with a 16 bit twos-complement
        magnitude component and 16 fractional bits.
``F2DOT14``
        A 16-bit signed fixed-point number with a 2 bit twos-complement
        magnitude component and 14 fractional bits.

To support reading and writing data with these data types, this module provides
functions for converting between fixed-point, float and string representations.

.. data:: MAX_F2DOT14

        The maximum value that can still fit in an F2Dot14. (1.99993896484375)
"""

from .roundTools import otRound, nearestMultipleShortestRepr
import logging

log = logging.getLogger(__name__)

__all__ = [
    "MAX_F2DOT14",
    "fixedToFloat",
    "floatToFixed",
    "floatToFixedToFloat",
    "floatToFixedToStr",
    "fixedToStr",
    "strToFixed",
    "strToFixedToFloat",
    "ensureVersionIsLong",
    "versionToFixed",
]


MAX_F2DOT14 = 0x7FFF / (1 << 14)


def fixedToFloat(value: float, precisionBits: int) -> float:
    """Converts a fixed-point number to a float given the number of
    precision bits.

    Args:
            value (int): Number in fixed-point format.
            precisionBits (int): Number of precision bits.

    Returns:
            Floating point value.

    Examples::

            >>> import math
            >>> f = fixedToFloat(-10139, precisionBits=14)
            >>> math.isclose(f, -0.61883544921875)
            True
    """
    return value / (1 << precisionBits)


def floatToFixed(value, precisionBits):
    """Converts a float to a fixed-point number given the number of
    precision bits.

    Args:
            value (float): Floating point value.
            precisionBits (int): Number of precision bits.

    Returns:
            int: Fixed-point representation.

    Examples::

            >>> floatToFixed(-0.61883544921875, precisionBits=14)
            -10139
            >>> floatToFixed(-0.61884, precisionBits=14)
            -10139
    """
    return otRound(value * (1 << precisionBits))


def floatToFixedToFloat(value, precisionBits):
    """Converts a float to a fixed-point number and back again.

    By converting the float to fixed, rounding it, and converting it back
    to float again, this returns a floating point values which is exactly
    representable in fixed-point format.

    Note: this **is** equivalent to ``fixedToFloat(floatToFixed(value))``.

    Args:
            value (float): The input floating point value.
            precisionBits (int): Number of precision bits.

    Returns:
            float: The transformed and rounded value.

    Examples::
            >>> import math
            >>> f1 = -0.61884
            >>> f2 = floatToFixedToFloat(-0.61884, precisionBits=14)
            >>> f1 != f2
            True
            >>> math.isclose(f2, -0.61883544921875)
            True
    """
    scale = 1 << precisionBits
    return otRound(value * scale) / scale


def fixedToStr(value, precisionBits):
    """Converts a fixed-point number to a string representing a decimal float.

    This chooses the float that has the shortest decimal representation (the least
    number of fractional decimal digits).

    For example, to convert a fixed-point number in a 2.14 format, use
    ``precisionBits=14``::

            >>> fixedToStr(-10139, precisionBits=14)
            '-0.61884'

    This is pretty slow compared to the simple division used in ``fixedToFloat``.
    Use sporadically when you need to serialize or print the fixed-point number in
    a human-readable form.
    It uses nearestMultipleShortestRepr under the hood.

    Args:
            value (int): The fixed-point value to convert.
            precisionBits (int): Number of precision bits, *up to a maximum of 16*.

    Returns:
            str: A string representation of the value.
    """
    scale = 1 << precisionBits
    return nearestMultipleShortestRepr(value / scale, factor=1.0 / scale)


def strToFixed(string, precisionBits):
    """Converts a string representing a decimal float to a fixed-point number.

    Args:
            string (str): A string representing a decimal float.
            precisionBits (int): Number of precision bits, *up to a maximum of 16*.

    Returns:
            int: Fixed-point representation.

    Examples::

            >>> ## to convert a float string to a 2.14 fixed-point number:
            >>> strToFixed('-0.61884', precisionBits=14)
            -10139
    """
    value = float(string)
    return otRound(value * (1 << precisionBits))


def strToFixedToFloat(string, precisionBits):
    """Convert a string to a decimal float with fixed-point rounding.

    This first converts string to a float, then turns it into a fixed-point
    number with ``precisionBits`` fractional binary digits, then back to a
    float again.

    This is simply a shorthand for fixedToFloat(floatToFixed(float(s))).

    Args:
            string (str): A string representing a decimal float.
            precisionBits (int): Number of precision bits.

    Returns:
            float: The transformed and rounded value.

    Examples::

            >>> import math
            >>> s = '-0.61884'
            >>> bits = 14
            >>> f = strToFixedToFloat(s, precisionBits=bits)
            >>> math.isclose(f, -0.61883544921875)
            True
            >>> f == fixedToFloat(floatToFixed(float(s), precisionBits=bits), precisionBits=bits)
            True
    """
    value = float(string)
    scale = 1 << precisionBits
    return otRound(value * scale) / scale


def floatToFixedToStr(value, precisionBits):
    """Convert float to string with fixed-point rounding.

    This uses the shortest decimal representation (ie. the least
    number of fractional decimal digits) to represent the equivalent
    fixed-point number with ``precisionBits`` fractional binary digits.
    It uses nearestMultipleShortestRepr under the hood.

    >>> floatToFixedToStr(-0.61883544921875, precisionBits=14)
    '-0.61884'

    Args:
            value (float): The float value to convert.
            precisionBits (int): Number of precision bits, *up to a maximum of 16*.

    Returns:
            str: A string representation of the value.

    """
    scale = 1 << precisionBits
    return nearestMultipleShortestRepr(value, factor=1.0 / scale)


def ensureVersionIsLong(value):
    """Ensure a table version is an unsigned long.

    OpenType table version numbers are expressed as a single unsigned long
    comprising of an unsigned short major version and unsigned short minor
    version. This function detects if the value to be used as a version number
    looks too small (i.e. is less than ``0x10000``), and converts it to
    fixed-point using :func:`floatToFixed` if so.

    Args:
            value (Number): a candidate table version number.

    Returns:
            int: A table version number, possibly corrected to fixed-point.
    """
    if value < 0x10000:
        newValue = floatToFixed(value, 16)
        log.warning(
            "Table version value is a float: %.4f; " "fix to use hex instead: 0x%08x",
            value,
            newValue,
        )
        value = newValue
    return value


def versionToFixed(value):
    """Ensure a table version number is fixed-point.

    Args:
            value (str): a candidate table version number.

    Returns:
            int: A table version number, possibly corrected to fixed-point.
    """
    value = int(value, 0) if value.startswith("0") else float(value)
    value = ensureVersionIsLong(value)
    return value
<|endoftext|>"""Collection of utilities for command-line interfaces and console scripts."""

import os
import re

numberAddedRE = re.compile(r"#\d+$")


def makeOutputFileName(
    input, outputDir=None, extension=None, overWrite=False, suffix=""
):
    """Generates a suitable file name for writing output.

    Often tools will want to take a file, do some kind of transformation to it,
    and write it out again. This function determines an appropriate name for the
    output file, through one or more of the following steps:

    - changing the output directory
    - appending suffix before file extension
    - replacing the file extension
    - suffixing the filename with a number (``#1``, ``#2``, etc.) to avoid
      overwriting an existing file.

    Args:
        input: Name of input file.
        outputDir: Optionally, a new directory to write the file into.
        suffix: Optionally, a string suffix is appended to file name before
            the extension.
        extension: Optionally, a replacement for the current file extension.
        overWrite: Overwriting an existing file is permitted if true; if false
            and the proposed filename exists, a new name will be generated by
            adding an appropriate number suffix.

    Returns:
        str: Suitable output filename
    """
    dirName, fileName = os.path.split(input)
    fileName, ext = os.path.splitext(fileName)
    if outputDir:
        dirName = outputDir
    fileName = numberAddedRE.split(fileName)[0]
    if extension is None:
        extension = os.path.splitext(input)[1]
    output = os.path.join(dirName, fileName + suffix + extension)
    n = 1
    if not overWrite:
        while os.path.exists(output):
            output = os.path.join(
                dirName, fileName + suffix + "#" + repr(n) + extension
            )
            n += 1
    return output
<|endoftext|>"""
Various round-to-integer helpers.
"""

import math
import functools
import logging
from collections.abc import Callable
from typing import TypeAlias

log = logging.getLogger(__name__)

__all__ = [
    "noRound",
    "otRound",
    "maybeRound",
    "roundFunc",
    "nearestMultipleShortestRepr",
]

RoundingFunction: TypeAlias = Callable[[float], float]
"""A rounding strategy for numbers; can return a float as well as an int, as in
some strategies rounding is conditional."""


def noRound(value: float) -> float:
    """Return float value unmodified; do not round.

    Args:
            value (float): The input floating-point value.

    Returns
            float: The same value.
    """
    return value


def otRound(value: float) -> int:
    """Round float value to nearest integer towards ``+Infinity``.

    The OpenType spec (in the section on `"normalization" of OpenType Font Variations <https://docs.microsoft.com/en-us/typography/opentype/spec/otvaroverview#coordinate-scales-and-normalization>`_)
    defines the required method for converting floating point values to
    fixed-point. In particular it specifies the following rounding strategy:

            for fractional values of 0.5 and higher, take the next higher integer;
            for other fractional values, truncate.

    This function rounds the floating-point value according to this strategy
    in preparation for conversion to fixed-point.

    Args:
            value (float): The input floating-point value.

    Returns
            int: The rounded value.
    """
    # See this thread for how we ended up with this implementation:
    # https://github.com/fonttools/fonttools/issues/1248#issuecomment-383198166
    return int(math.floor(value + 0.5))


def maybeRound(v: float, tolerance: float, round: RoundingFunction = otRound) -> float:
    """Round only if rounding changes the value by no more than a given amount.

    Args:
            value (float): The input floating-point value.
            tolerance (float): The maximum absolute difference permitted.
            round (function): The rounding strategy, taking and giving a float.

    Returns
            float: A rounded value if within tolerance, otherwise the original.
    """
    rounded = round(v)
    return rounded if abs(rounded - v) <= tolerance else v


def roundFunc(tolerance: float, round: RoundingFunction = otRound) -> RoundingFunction:
    """Make a conditional rounding strategy from a base strategy and tolerance.

    This preconfigures :func:`maybeRound` with fixed arguments, allowing it to be
    used anywhere a simple rounding function is accepted.

    The tolerance is validated to ensure it is absolute, and the cases where
    rounding would never or always occur are optimized. Rounding must never
    change a value by more than 0.5 for this to be sound (e.g.  :py:func:`ceil`
    could change by too much).

    Args:
            tolerance (float): The maximum absolute difference permitted.
            round (function): The rounding strategy, taking and giving a float.

    Returns
            function: A strategy that conditionally rounds based on tolerance.
    """
    if tolerance < 0:
        raise ValueError("Rounding tolerance must be positive")

    if tolerance == 0:
        return noRound

    if tolerance >= 0.5:
        return round

    return functools.partial(maybeRound, tolerance=tolerance, round=round)


def nearestMultipleShortestRepr(value: float, factor: float) -> str:
    """Round to nearest multiple of factor and return shortest decimal representation.

    This chooses the float that is closer to a multiple of the given factor while
    having the shortest decimal representation (the least number of fractional decimal
    digits).

    For example, given the following:

    >>> nearestMultipleShortestRepr(-0.61883544921875, 1.0/(1<<14))
    '-0.61884'

    Useful when you need to serialize or print a fixed-point number (or multiples
    thereof, such as F2Dot14 fractions of 180 degrees in COLRv1 PaintRotate) in
    a human-readable form.

    Args:
        value (value): The value to be rounded and serialized.
        factor (float): The value which the result is a close multiple of.

    Returns:
        str: A compact string representation of the value.
    """
    if not value:
        return "0.0"

    value = otRound(value / factor) * factor
    eps = 0.5 * factor
    lo = value - eps
    hi = value + eps
    # If the range of valid choices spans an integer, return the integer.
    if int(lo) != int(hi):
        return str(float(round(value)))

    fmt = "%.8f"
    lo = fmt % lo
    hi = fmt % hi
    assert len(lo) == len(hi) and lo != hi
    for i in range(len(lo)):
        if lo[i] != hi[i]:
            break
    period = lo.find(".")
    assert period < i
    fmt = "%%.%df" % (i - period)
    return fmt % value
<|endoftext|>"""fontTools.misc.textTools.py -- miscellaneous routines."""

from __future__ import annotations

import ast
import string

# alias kept for backward compatibility
safeEval = ast.literal_eval


class Tag(str):
    @staticmethod
    def transcode(blob):
        if isinstance(blob, bytes):
            blob = blob.decode("latin-1")
        return blob

    def __new__(self, content):
        return str.__new__(self, self.transcode(content))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __eq__(self, other):
        return str.__eq__(self, self.transcode(other))

    def __hash__(self):
        return str.__hash__(self)

    def tobytes(self):
        return self.encode("latin-1")


def readHex(content):
    """Convert a list of hex strings to binary data."""
    return deHexStr(strjoin(chunk for chunk in content if isinstance(chunk, str)))


def deHexStr(hexdata):
    """Convert a hex string to binary data."""
    hexdata = strjoin(hexdata.split())
    if len(hexdata) % 2:
        hexdata = hexdata + "0"
    data = []
    for i in range(0, len(hexdata), 2):
        data.append(bytechr(int(hexdata[i : i + 2], 16)))
    return bytesjoin(data)


def hexStr(data):
    """Convert binary data to a hex string."""
    h = string.hexdigits
    r = ""
    for c in data:
        i = byteord(c)
        r = r + h[(i >> 4) & 0xF] + h[i & 0xF]
    return r


def num2binary(l, bits=32):
    items = []
    binary = ""
    for i in range(bits):
        if l & 0x1:
            binary = "1" + binary
        else:
            binary = "0" + binary
        l = l >> 1
        if not ((i + 1) % 8):
            items.append(binary)
            binary = ""
    if binary:
        items.append(binary)
    items.reverse()
    assert l in (0, -1), "number doesn't fit in number of bits"
    return " ".join(items)


def binary2num(bin):
    bin = strjoin(bin.split())
    l = 0
    for digit in bin:
        l = l << 1
        if digit != "0":
            l = l | 0x1
    return l


def caselessSort(alist):
    """Return a sorted copy of a list. If there are only strings
    in the list, it will not consider case.
    """

    try:
        return sorted(alist, key=lambda a: (a.lower(), a))
    except TypeError:
        return sorted(alist)


def pad(data, size):
    r"""Pad byte string 'data' with null bytes until its length is a
    multiple of 'size'.

    >>> len(pad(b'abcd', 4))
    4
    >>> len(pad(b'abcde', 2))
    6
    >>> len(pad(b'abcde', 4))
    8
    >>> pad(b'abcdef', 4) == b'abcdef\x00\x00'
    True
    """
    data = tobytes(data)
    if size > 1:
        remainder = len(data) % size
        if remainder:
            data += b"\0" * (size - remainder)
    return data


def tostr(s: str | bytes, encoding: str = "ascii", errors: str = "strict") -> str:
    if not isinstance(s, str):
        return s.decode(encoding, errors)
    else:
        return s


def tobytes(s: str | bytes, encoding: str = "ascii", errors: str = "strict") -> bytes:
    if isinstance(s, str):
        return s.encode(encoding, errors)
    else:
        return bytes(s)


def bytechr(n):
    return bytes([n])


def byteord(c):
    return c if isinstance(c, int) else ord(c)


def strjoin(iterable, joiner=""):
    return tostr(joiner).join(iterable)


def bytesjoin(iterable, joiner=b""):
    return tobytes(joiner).join(tobytes(item) for item in iterable)


if __name__ == "__main__":
    import doctest, sys

    sys.exit(doctest.testmod().failed)
<|endoftext|>import sys
import logging
import timeit
from functools import wraps
from collections.abc import Mapping, Callable
import warnings
from logging import PercentStyle

# default logging level used by Timer class
TIME_LEVEL = logging.DEBUG

# per-level format strings used by the default formatter
# (the level name is not printed for INFO and DEBUG messages)
DEFAULT_FORMATS = {
    "*": "%(levelname)s: %(message)s",
    "INFO": "%(message)s",
    "DEBUG": "%(message)s",
}


class LevelFormatter(logging.Formatter):
    """Log formatter with level-specific formatting.

    Formatter class which optionally takes a dict of logging levels to
    format strings, allowing to customise the log records appearance for
    specific levels.


    Attributes:
            fmt: A dictionary mapping logging levels to format strings.
                    The ``*`` key identifies the default format string.
            datefmt: As per py:class:`logging.Formatter`
            style: As per py:class:`logging.Formatter`

    >>> import sys
    >>> handler = logging.StreamHandler(sys.stdout)
    >>> formatter = LevelFormatter(
    ...     fmt={
    ...         '*':     '[%(levelname)s] %(message)s',
    ...         'DEBUG': '%(name)s [%(levelname)s] %(message)s',
    ...         'INFO':  '%(message)s',
    ...     })
    >>> handler.setFormatter(formatter)
    >>> log = logging.getLogger('test')
    >>> log.setLevel(logging.DEBUG)
    >>> log.addHandler(handler)
    >>> log.debug('this uses a custom format string')
    test [DEBUG] this uses a custom format string
    >>> log.info('this also uses a custom format string')
    this also uses a custom format string
    >>> log.warning("this one uses the default format string")
    [WARNING] this one uses the default format string
    """

    def __init__(self, fmt=None, datefmt=None, style="%"):
        if style != "%":
            raise ValueError(
                "only '%' percent style is supported in both python 2 and 3"
            )
        if fmt is None:
            fmt = DEFAULT_FORMATS
        if isinstance(fmt, str):
            default_format = fmt
            custom_formats = {}
        elif isinstance(fmt, Mapping):
            custom_formats = dict(fmt)
            default_format = custom_formats.pop("*", None)
        else:
            raise TypeError("fmt must be a str or a dict of str: %r" % fmt)
        super(LevelFormatter, self).__init__(default_format, datefmt)
        self.default_format = self._fmt
        self.custom_formats = {}
        for level, fmt in custom_formats.items():
            level = logging._checkLevel(level)
            self.custom_formats[level] = fmt

    def format(self, record):
        if self.custom_formats:
            fmt = self.custom_formats.get(record.levelno, self.default_format)
            if self._fmt != fmt:
                self._fmt = fmt
                # for python >= 3.2, _style needs to be set if _fmt changes
                if PercentStyle:
                    self._style = PercentStyle(fmt)
        return super(LevelFormatter, self).format(record)


def configLogger(**kwargs):
    """A more sophisticated logging system configuation manager.

    This is more or less the same as :py:func:`logging.basicConfig`,
    with some additional options and defaults.

    The default behaviour is to create a ``StreamHandler`` which writes to
    sys.stderr, set a formatter using the ``DEFAULT_FORMATS`` strings, and add
    the handler to the top-level library logger ("fontTools").

    A number of optional keyword arguments may be specified, which can alter
    the default behaviour.

    Args:

            logger: Specifies the logger name or a Logger instance to be
                    configured. (Defaults to "fontTools" logger). Unlike ``basicConfig``,
                    this function can be called multiple times to reconfigure a logger.
                    If the logger or any of its children already exists before the call is
                    made, they will be reset before the new configuration is applied.
            filename: Specifies that a ``FileHandler`` be created, using the
                    specified filename, rather than a ``StreamHandler``.
            filemode: Specifies the mode to open the file, if filename is
                    specified. (If filemode is unspecified, it defaults to ``a``).
            format: Use the specified format string for the handler. This
                    argument also accepts a dictionary of format strings keyed by
                    level name, to allow customising the records appearance for
                    specific levels. The special ``'*'`` key is for 'any other' level.
            datefmt: Use the specified date/time format.
            level: Set the logger level to the specified level.
            stream: Use the specified stream to initialize the StreamHandler. Note
                    that this argument is incompatible with ``filename`` - if both
                    are present, ``stream`` is ignored.
            handlers: If specified, this should be an iterable of already created
                    handlers, which will be added to the logger. Any handler in the
                    list which does not have a formatter assigned will be assigned the
                    formatter created in this function.
            filters: If specified, this should be an iterable of already created
                    filters. If the ``handlers`` do not already have filters assigned,
                    these filters will be added to them.
            propagate: All loggers have a ``propagate`` attribute which determines
                    whether to continue searching for handlers up the logging hierarchy.
                    If not provided, the "propagate" attribute will be set to ``False``.
    """
    # using kwargs to enforce keyword-only arguments in py2.
    handlers = kwargs.pop("handlers", None)
    if handlers is None:
        if "stream" in kwargs and "filename" in kwargs:
            raise ValueError(
                "'stream' and 'filename' should not be " "specified together"
            )
    else:
        if "stream" in kwargs or "filename" in kwargs:
            raise ValueError(
                "'stream' or 'filename' should not be "
                "specified together with 'handlers'"
            )
    if handlers is None:
        filename = kwargs.pop("filename", None)
        mode = kwargs.pop("filemode", "a")
        if filename:
            h = logging.FileHandler(filename, mode)
        else:
            stream = kwargs.pop("stream", None)
            h = logging.StreamHandler(stream)
        handlers = [h]
    # By default, the top-level library logger is configured.
    logger = kwargs.pop("logger", "fontTools")
    if not logger or isinstance(logger, str):
        # empty "" or None means the 'root' logger
        logger = logging.getLogger(logger)
    # before (re)configuring, reset named logger and its children (if exist)
    _resetExistingLoggers(parent=logger.name)
    # use DEFAULT_FORMATS if 'format' is None
    fs = kwargs.pop("format", None)
    dfs = kwargs.pop("datefmt", None)
    # XXX: '%' is the only format style supported on both py2 and 3
    style = kwargs.pop("style", "%")
    fmt = LevelFormatter(fs, dfs, style)
    filters = kwargs.pop("filters", [])
    for h in handlers:
        if h.formatter is None:
            h.setFormatter(fmt)
        if not h.filters:
            for f in filters:
                h.addFilter(f)
        logger.addHandler(h)
    if logger.name != "root":
        # stop searching up the hierarchy for handlers
        logger.propagate = kwargs.pop("propagate", False)
    # set a custom severity level
    level = kwargs.pop("level", None)
    if level is not None:
        logger.setLevel(level)
    if kwargs:
        keys = ", ".join(kwargs.keys())
        raise ValueError("Unrecognised argument(s): %s" % keys)


def _resetExistingLoggers(parent="root"):
    """Reset the logger named 'parent' and all its children to their initial
    state, if they already exist in the current configuration.
    """
    root = logging.root
    # get sorted list of all existing loggers
    existing = sorted(root.manager.loggerDict.keys())
    if parent == "root":
        # all the existing loggers are children of 'root'
        loggers_to_reset = [parent] + existing
    elif parent not in existing:
        # nothing to do
        return
    elif parent in existing:
        loggers_to_reset = [parent]
        # collect children, starting with the entry after parent name
        i = existing.index(parent) + 1
        prefixed = parent + "."
        pflen = len(prefixed)
        num_existing = len(existing)
        while i < num_existing:
            if existing[i][:pflen] == prefixed:
                loggers_to_reset.append(existing[i])
            i += 1
    for name in loggers_to_reset:
        if name == "root":
            root.setLevel(logging.WARNING)
            for h in root.handlers[:]:
                root.removeHandler(h)
            for f in root.filters[:]:
                root.removeFilters(f)
            root.disabled = False
        else:
            logger = root.manager.loggerDict[name]
            logger.level = logging.NOTSET
            logger.handlers = []
            logger.filters = []
            logger.propagate = True
            logger.disabled = False


class Timer(object):
    """Keeps track of overall time and split/lap times.

    >>> import time
    >>> timer = Timer()
    >>> time.sleep(0.01)
    >>> print("First lap:", timer.split())
    First lap: ...
    >>> time.sleep(0.02)
    >>> print("Second lap:", timer.split())
    Second lap: ...
    >>> print("Overall time:", timer.time())
    Overall time: ...

    Can be used as a context manager inside with-statements.

    >>> with Timer() as t:
    ...     time.sleep(0.01)
    >>> print("%0.3f seconds" % t.elapsed)
    0... seconds

    If initialised with a logger, it can log the elapsed time automatically
    upon exiting the with-statement.

    >>> import logging
    >>> log = logging.getLogger("my-fancy-timer-logger")
    >>> configLogger(logger=log, level="DEBUG", format="%(message)s", stream=sys.stdout)
    >>> with Timer(log, 'do something'):
    ...     time.sleep(0.01)
    Took ... to do something

    The same Timer instance, holding a reference to a logger, can be reused
    in multiple with-statements, optionally with different messages or levels.

    >>> timer = Timer(log)
    >>> with timer():
    ...     time.sleep(0.01)
    elapsed time: ...s
    >>> with timer('redo it', level=logging.INFO):
    ...     time.sleep(0.02)
    Took ... to redo it

    It can also be used as a function decorator to log the time elapsed to run
    the decorated function.

    >>> @timer()
    ... def test1():
    ...    time.sleep(0.01)
    >>> @timer('run test 2', level=logging.INFO)
    ... def test2():
    ...    time.sleep(0.02)
    >>> test1()
    Took ... to run 'test1'
    >>> test2()
    Took ... to run test 2
    """

    # timeit.default_timer choses the most accurate clock for each platform
    _time: Callable[[], float] = staticmethod(timeit.default_timer)
    default_msg = "elapsed time: %(time).3fs"
    default_format = "Took %(time).3fs to %(msg)s"

    def __init__(self, logger=None, msg=None, level=None, start=None):
        self.reset(start)
        if logger is None:
            for arg in ("msg", "level"):
                if locals().get(arg) is not None:
                    raise ValueError("'%s' can't be specified without a 'logger'" % arg)
        self.logger = logger
        self.level = level if level is not None else TIME_LEVEL
        self.msg = msg

    def reset(self, start=None):
        """Reset timer to 'start_time' or the current time."""
        if start is None:
            self.start = self._time()
        else:
            self.start = start
        self.last = self.start
        self.elapsed = 0.0

    def time(self):
        """Return the overall time (in seconds) since the timer started."""
        return self._time() - self.start

    def split(self):
        """Split and return the lap time (in seconds) in between splits."""
        current = self._time()
        self.elapsed = current - self.last
        self.last = current
        return self.elapsed

    def formatTime(self, msg, time):
        """Format 'time' value in 'msg' and return formatted string.
        If 'msg' contains a '%(time)' format string, try to use that.
        Otherwise, use the predefined 'default_format'.
        If 'msg' is empty or None, fall back to 'default_msg'.
        """
        if not msg:
            msg = self.default_msg
        if msg.find("%(time)") < 0:
            msg = self.default_format % {"msg": msg, "time": time}
        else:
            try:
                msg = msg % {"time": time}
            except (KeyError, ValueError):
                pass  # skip if the format string is malformed
        return msg

    def __enter__(self):
        """Start a new lap"""
        self.last = self._time()
        self.elapsed = 0.0
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """End the current lap. If timer has a logger, log the time elapsed,
