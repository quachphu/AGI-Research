        if not (self.version == 0 or self.version == 1):
            from fontTools import ttLib

            raise ttLib.TTLibError(
                "unknown format for VDMX table: version %s" % self.version
            )
        data = sstruct.pack(VDMX_HeaderFmt, self)
        for ratio in self.ratRanges:
            data += sstruct.pack(VDMX_RatRangeFmt, ratio)
        # recalculate offsets to VDMX groups
        for offset in self._getOffsets():
            data += struct.pack(">H", offset)
        for group in self.groups:
            recs = len(group)
            startsz = min(group.keys())
            endsz = max(group.keys())
            gHeader = {"recs": recs, "startsz": startsz, "endsz": endsz}
            data += sstruct.pack(VDMX_GroupFmt, gHeader)
            for yPelHeight, (yMax, yMin) in sorted(group.items()):
                vTable = {"yPelHeight": yPelHeight, "yMax": yMax, "yMin": yMin}
                data += sstruct.pack(VDMX_vTableFmt, vTable)
        return data

    def toXML(self, writer, ttFont):
        writer.simpletag("version", value=self.version)
        writer.newline()
        writer.begintag("ratRanges")
        writer.newline()
        for ratio in self.ratRanges:
            groupIndex = ratio["groupIndex"]
            writer.simpletag(
                "ratRange",
                bCharSet=ratio["bCharSet"],
                xRatio=ratio["xRatio"],
                yStartRatio=ratio["yStartRatio"],
                yEndRatio=ratio["yEndRatio"],
                groupIndex=groupIndex,
            )
            writer.newline()
        writer.endtag("ratRanges")
        writer.newline()
        writer.begintag("groups")
        writer.newline()
        for groupIndex in range(self.numRecs):
            group = self.groups[groupIndex]
            recs = len(group)
            startsz = min(group.keys())
            endsz = max(group.keys())
            writer.begintag("group", index=groupIndex)
            writer.newline()
            writer.comment("recs=%d, startsz=%d, endsz=%d" % (recs, startsz, endsz))
            writer.newline()
            for yPelHeight, (yMax, yMin) in sorted(group.items()):
                writer.simpletag(
                    "record",
                    [("yPelHeight", yPelHeight), ("yMax", yMax), ("yMin", yMin)],
                )
                writer.newline()
            writer.endtag("group")
            writer.newline()
        writer.endtag("groups")
        writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        if name == "version":
            self.version = safeEval(attrs["value"])
        elif name == "ratRanges":
            if not hasattr(self, "ratRanges"):
                self.ratRanges = []
            for element in content:
                if not isinstance(element, tuple):
                    continue
                name, attrs, content = element
                if name == "ratRange":
                    if not hasattr(self, "numRatios"):
                        self.numRatios = 1
                    else:
                        self.numRatios += 1
                    ratio = {
                        "bCharSet": safeEval(attrs["bCharSet"]),
                        "xRatio": safeEval(attrs["xRatio"]),
                        "yStartRatio": safeEval(attrs["yStartRatio"]),
                        "yEndRatio": safeEval(attrs["yEndRatio"]),
                        "groupIndex": safeEval(attrs["groupIndex"]),
                    }
                    self.ratRanges.append(ratio)
        elif name == "groups":
            if not hasattr(self, "groups"):
                self.groups = []
            for element in content:
                if not isinstance(element, tuple):
                    continue
                name, attrs, content = element
                if name == "group":
                    if not hasattr(self, "numRecs"):
                        self.numRecs = 1
                    else:
                        self.numRecs += 1
                    group = {}
                    for element in content:
                        if not isinstance(element, tuple):
                            continue
                        name, attrs, content = element
                        if name == "record":
                            yPelHeight = safeEval(attrs["yPelHeight"])
                            yMax = safeEval(attrs["yMax"])
                            yMin = safeEval(attrs["yMin"])
                            group[yPelHeight] = (yMax, yMin)
                    self.groups.append(group)
<|endoftext|>from fontTools.misc import sstruct
from fontTools.misc.roundTools import otRound
from fontTools.misc.textTools import safeEval, num2binary, binary2num
from fontTools.ttLib.tables import DefaultTable
import bisect
import logging

log = logging.getLogger(__name__)

# panose classification

panoseFormat = """
	bFamilyType:        B
	bSerifStyle:        B
	bWeight:            B
	bProportion:        B
	bContrast:          B
	bStrokeVariation:   B
	bArmStyle:          B
	bLetterForm:        B
	bMidline:           B
	bXHeight:           B
"""


class Panose(object):
    def __init__(self, **kwargs):
        _, names, _ = sstruct.getformat(panoseFormat)
        for name in names:
            setattr(self, name, kwargs.pop(name, 0))
        for k in kwargs:
            raise TypeError(f"Panose() got an unexpected keyword argument {k!r}")

    def toXML(self, writer, ttFont):
        formatstring, names, fixes = sstruct.getformat(panoseFormat)
        for name in names:
            writer.simpletag(name, value=getattr(self, name))
            writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        setattr(self, name, safeEval(attrs["value"]))


# 'sfnt' OS/2 and Windows Metrics table - 'OS/2'

OS2_format_0 = """
	>   # big endian
	version:                H       # version
	xAvgCharWidth:          h       # average character width
	usWeightClass:          H       # degree of thickness of strokes
	usWidthClass:           H       # aspect ratio
	fsType:                 H       # type flags
	ySubscriptXSize:        h       # subscript horizontal font size
	ySubscriptYSize:        h       # subscript vertical font size
	ySubscriptXOffset:      h       # subscript x offset
	ySubscriptYOffset:      h       # subscript y offset
	ySuperscriptXSize:      h       # superscript horizontal font size
	ySuperscriptYSize:      h       # superscript vertical font size
	ySuperscriptXOffset:    h       # superscript x offset
	ySuperscriptYOffset:    h       # superscript y offset
	yStrikeoutSize:         h       # strikeout size
	yStrikeoutPosition:     h       # strikeout position
	sFamilyClass:           h       # font family class and subclass
	panose:                 10s     # panose classification number
	ulUnicodeRange1:        L       # character range
	ulUnicodeRange2:        L       # character range
	ulUnicodeRange3:        L       # character range
	ulUnicodeRange4:        L       # character range
	achVendID:              4s      # font vendor identification
	fsSelection:            H       # font selection flags
	usFirstCharIndex:       H       # first unicode character index
	usLastCharIndex:        H       # last unicode character index
	sTypoAscender:          h       # typographic ascender
	sTypoDescender:         h       # typographic descender
	sTypoLineGap:           h       # typographic line gap
	usWinAscent:            H       # Windows ascender
	usWinDescent:           H       # Windows descender
"""

OS2_format_1_addition = """
	ulCodePageRange1:   L
	ulCodePageRange2:   L
"""

OS2_format_2_addition = OS2_format_1_addition + """
	sxHeight:           h
	sCapHeight:         h
	usDefaultChar:      H
	usBreakChar:        H
	usMaxContext:       H
"""

OS2_format_5_addition = OS2_format_2_addition + """
	usLowerOpticalPointSize:    H
	usUpperOpticalPointSize:    H
"""

bigendian = "	>	# big endian\n"

OS2_format_1 = OS2_format_0 + OS2_format_1_addition
OS2_format_2 = OS2_format_0 + OS2_format_2_addition
OS2_format_5 = OS2_format_0 + OS2_format_5_addition
OS2_format_1_addition = bigendian + OS2_format_1_addition
OS2_format_2_addition = bigendian + OS2_format_2_addition
OS2_format_5_addition = bigendian + OS2_format_5_addition


class table_O_S_2f_2(DefaultTable.DefaultTable):
    """OS/2 and Windows Metrics table

    The ``OS/2`` table contains a variety of font-wide metrics and
    parameters that may be useful to an operating system or other
    software for system-integration purposes.

    See also https://learn.microsoft.com/en-us/typography/opentype/spec/os2
    """

    dependencies = ["head", "bhed"]

    def decompile(self, data, ttFont):
        dummy, data = sstruct.unpack2(OS2_format_0, data, self)

        if self.version == 1:
            dummy, data = sstruct.unpack2(OS2_format_1_addition, data, self)
        elif self.version in (2, 3, 4):
            dummy, data = sstruct.unpack2(OS2_format_2_addition, data, self)
        elif self.version == 5:
            dummy, data = sstruct.unpack2(OS2_format_5_addition, data, self)
            self.usLowerOpticalPointSize /= 20
            self.usUpperOpticalPointSize /= 20
        elif self.version != 0:
            from fontTools import ttLib

            raise ttLib.TTLibError(
                "unknown format for OS/2 table: version %s" % self.version
            )
        if len(data):
            log.warning("too much 'OS/2' table data")

        self.panose = sstruct.unpack(panoseFormat, self.panose, Panose())

    def compile(self, ttFont):
        self.updateFirstAndLastCharIndex(ttFont)
        panose = self.panose
        self._checkHeadMacStyle("head", ttFont)
        self._checkHeadMacStyle("bhed", ttFont)
        if (self.fsSelection & 1 << 6) and (self.fsSelection & 1 + (1 << 5)):
            log.warning(
                "fsSelection bit 6 (regular) is set, "
                "bits 0 (italic) and 5 (bold) must be clear"
            )
        if self.version < 4 and self.fsSelection & 0b1110000000:
            log.warning(
                "fsSelection bits 7, 8 and 9 are only defined in "
                "OS/2 table version 4 and up: version %s",
                self.version,
            )
        self.panose = sstruct.pack(panoseFormat, self.panose)
        if self.version == 0:
            data = sstruct.pack(OS2_format_0, self)
        elif self.version == 1:
            data = sstruct.pack(OS2_format_1, self)
        elif self.version in (2, 3, 4):
            data = sstruct.pack(OS2_format_2, self)
        elif self.version == 5:
            d = self.__dict__.copy()
            d["usLowerOpticalPointSize"] = round(self.usLowerOpticalPointSize * 20)
            d["usUpperOpticalPointSize"] = round(self.usUpperOpticalPointSize * 20)
            data = sstruct.pack(OS2_format_5, d)
        else:
            from fontTools import ttLib

            raise ttLib.TTLibError(
                "unknown format for OS/2 table: version %s" % self.version
            )
        self.panose = panose
        return data

    def toXML(self, writer, ttFont):
        writer.comment(
            "The fields 'usFirstCharIndex' and 'usLastCharIndex'\n"
            "will be recalculated by the compiler"
        )
        writer.newline()
        if self.version == 1:
            format = OS2_format_1
        elif self.version in (2, 3, 4):
            format = OS2_format_2
        elif self.version == 5:
            format = OS2_format_5
        else:
            format = OS2_format_0
        formatstring, names, fixes = sstruct.getformat(format)
        for name in names:
            value = getattr(self, name)
            if name == "panose":
                writer.begintag("panose")
                writer.newline()
                value.toXML(writer, ttFont)
                writer.endtag("panose")
            elif name in (
                "ulUnicodeRange1",
                "ulUnicodeRange2",
                "ulUnicodeRange3",
                "ulUnicodeRange4",
                "ulCodePageRange1",
                "ulCodePageRange2",
            ):
                writer.simpletag(name, value=num2binary(value))
            elif name in ("fsType", "fsSelection"):
                writer.simpletag(name, value=num2binary(value, 16))
            elif name == "achVendID":
                writer.simpletag(name, value=repr(value)[1:-1])
            else:
                writer.simpletag(name, value=value)
            writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        if name == "panose":
            self.panose = panose = Panose()
            for element in content:
                if isinstance(element, tuple):
                    name, attrs, content = element
                    panose.fromXML(name, attrs, content, ttFont)
        elif name in (
            "ulUnicodeRange1",
            "ulUnicodeRange2",
            "ulUnicodeRange3",
            "ulUnicodeRange4",
            "ulCodePageRange1",
            "ulCodePageRange2",
            "fsType",
            "fsSelection",
        ):
            setattr(self, name, binary2num(attrs["value"]))
        elif name == "achVendID":
            setattr(self, name, safeEval("'''" + attrs["value"] + "'''"))
        else:
            setattr(self, name, safeEval(attrs["value"]))

    def updateFirstAndLastCharIndex(self, ttFont):
        if "cmap" not in ttFont:
            return
        codes = set()
        for table in getattr(ttFont["cmap"], "tables", []):
            if table.isUnicode():
                codes.update(table.cmap.keys())
        if codes:
            minCode = min(codes)
            maxCode = max(codes)
            # USHORT cannot hold codepoints greater than 0xFFFF
            self.usFirstCharIndex = min(0xFFFF, minCode)
            self.usLastCharIndex = min(0xFFFF, maxCode)

    def _checkHeadMacStyle(self, tag, ttFont):
        if tag in ttFont:
            head = ttFont[tag]
            if (self.fsSelection & 1) and not (head.macStyle & 1 << 1):
                log.warning(
                    "fsSelection bit 0 (italic) and "
                    "%s table macStyle bit 1 (italic) should match",
                    tag,
                )
            if (self.fsSelection & 1 << 5) and not (head.macStyle & 1):
                log.warning(
                    "fsSelection bit 5 (bold) and "
                    "%s table macStyle bit 0 (bold) should match",
                    tag,
                )

    # misspelled attributes kept for legacy reasons

    @property
    def usMaxContex(self):
        return self.usMaxContext

    @usMaxContex.setter
    def usMaxContex(self, value):
        self.usMaxContext = value

    @property
    def fsFirstCharIndex(self):
        return self.usFirstCharIndex

    @fsFirstCharIndex.setter
    def fsFirstCharIndex(self, value):
        self.usFirstCharIndex = value

    @property
    def fsLastCharIndex(self):
        return self.usLastCharIndex

    @fsLastCharIndex.setter
    def fsLastCharIndex(self, value):
        self.usLastCharIndex = value

    def getUnicodeRanges(self):
        """Return the set of 'ulUnicodeRange*' bits currently enabled."""
        bits = set()
        ul1, ul2 = self.ulUnicodeRange1, self.ulUnicodeRange2
        ul3, ul4 = self.ulUnicodeRange3, self.ulUnicodeRange4
        for i in range(32):
            if ul1 & (1 << i):
                bits.add(i)
            if ul2 & (1 << i):
                bits.add(i + 32)
            if ul3 & (1 << i):
                bits.add(i + 64)
            if ul4 & (1 << i):
                bits.add(i + 96)
        return bits

    def setUnicodeRanges(self, bits):
        """Set the 'ulUnicodeRange*' fields to the specified 'bits'."""
        ul1, ul2, ul3, ul4 = 0, 0, 0, 0
        for bit in bits:
            if 0 <= bit < 32:
                ul1 |= 1 << bit
            elif 32 <= bit < 64:
                ul2 |= 1 << (bit - 32)
            elif 64 <= bit < 96:
                ul3 |= 1 << (bit - 64)
            elif 96 <= bit < 128:
                ul4 |= 1 << (bit - 96)
            else:
                raise ValueError("expected 0 <= int <= 127, found: %r" % bit)
        self.ulUnicodeRange1, self.ulUnicodeRange2 = ul1, ul2
        self.ulUnicodeRange3, self.ulUnicodeRange4 = ul3, ul4

    def recalcUnicodeRanges(self, ttFont, pruneOnly=False):
        """Intersect the codepoints in the font's Unicode cmap subtables with
        the Unicode block ranges defined in the OpenType specification (v1.7),
        and set the respective 'ulUnicodeRange*' bits if there is at least ONE
        intersection.
        If 'pruneOnly' is True, only clear unused bits with NO intersection.
        """
        unicodes = set()
        for table in ttFont["cmap"].tables:
            if table.isUnicode():
                unicodes.update(table.cmap.keys())
        if pruneOnly:
            empty = intersectUnicodeRanges(unicodes, inverse=True)
            bits = self.getUnicodeRanges() - empty
        else:
            bits = intersectUnicodeRanges(unicodes)
        self.setUnicodeRanges(bits)
        return bits

    def getCodePageRanges(self):
        """Return the set of 'ulCodePageRange*' bits currently enabled."""
        bits = set()
        if self.version < 1:
            return bits
        ul1, ul2 = self.ulCodePageRange1, self.ulCodePageRange2
        for i in range(32):
            if ul1 & (1 << i):
                bits.add(i)
            if ul2 & (1 << i):
                bits.add(i + 32)
        return bits

    def setCodePageRanges(self, bits):
        """Set the 'ulCodePageRange*' fields to the specified 'bits'."""
        ul1, ul2 = 0, 0
        for bit in bits:
            if 0 <= bit < 32:
                ul1 |= 1 << bit
            elif 32 <= bit < 64:
                ul2 |= 1 << (bit - 32)
            else:
                raise ValueError(f"expected 0 <= int <= 63, found: {bit:r}")
        if self.version < 1:
            self.version = 1
        self.ulCodePageRange1, self.ulCodePageRange2 = ul1, ul2

    def recalcCodePageRanges(self, ttFont, pruneOnly=False):
        unicodes = set()
        for table in ttFont["cmap"].tables:
            if table.isUnicode():
                unicodes.update(table.cmap.keys())
        bits = calcCodePageRanges(unicodes)
        if pruneOnly:
            bits &= self.getCodePageRanges()
        # when no codepage ranges can be enabled, fall back to enabling bit 0
        # (Latin 1) so that the font works in MS Word:
        # https://github.com/googlei18n/fontmake/issues/468
        if not bits:
            bits = {0}
        self.setCodePageRanges(bits)
        return bits

    def recalcAvgCharWidth(self, ttFont):
        """Recalculate xAvgCharWidth using metrics from ttFont's 'hmtx' table.

        Set it to 0 if the unlikely event 'hmtx' table is not found.
        """
        avg_width = 0
        hmtx = ttFont.get("hmtx")
        if hmtx is not None:
            widths = [width for width, _ in hmtx.metrics.values() if width > 0]
            if widths:
                avg_width = otRound(sum(widths) / len(widths))
        self.xAvgCharWidth = avg_width
        return avg_width


# Unicode ranges data from the OpenType OS/2 table specification v1.7

OS2_UNICODE_RANGES = (
    (("Basic Latin", (0x0000, 0x007F)),),
    (("Latin-1 Supplement", (0x0080, 0x00FF)),),
    (("Latin Extended-A", (0x0100, 0x017F)),),
    (("Latin Extended-B", (0x0180, 0x024F)),),
    (
        ("IPA Extensions", (0x0250, 0x02AF)),
        ("Phonetic Extensions", (0x1D00, 0x1D7F)),
        ("Phonetic Extensions Supplement", (0x1D80, 0x1DBF)),
    ),
    (
        ("Spacing Modifier Letters", (0x02B0, 0x02FF)),
        ("Modifier Tone Letters", (0xA700, 0xA71F)),
    ),
    (
        ("Combining Diacritical Marks", (0x0300, 0x036F)),
        ("Combining Diacritical Marks Supplement", (0x1DC0, 0x1DFF)),
    ),
    (("Greek and Coptic", (0x0370, 0x03FF)),),
    (("Coptic", (0x2C80, 0x2CFF)),),
    (
        ("Cyrillic", (0x0400, 0x04FF)),
        ("Cyrillic Supplement", (0x0500, 0x052F)),
        ("Cyrillic Extended-A", (0x2DE0, 0x2DFF)),
        ("Cyrillic Extended-B", (0xA640, 0xA69F)),
    ),
    (("Armenian", (0x0530, 0x058F)),),
    (("Hebrew", (0x0590, 0x05FF)),),
    (("Vai", (0xA500, 0xA63F)),),
    (("Arabic", (0x0600, 0x06FF)), ("Arabic Supplement", (0x0750, 0x077F))),
    (("NKo", (0x07C0, 0x07FF)),),
    (("Devanagari", (0x0900, 0x097F)),),
    (("Bengali", (0x0980, 0x09FF)),),
    (("Gurmukhi", (0x0A00, 0x0A7F)),),
    (("Gujarati", (0x0A80, 0x0AFF)),),
    (("Oriya", (0x0B00, 0x0B7F)),),
    (("Tamil", (0x0B80, 0x0BFF)),),
    (("Telugu", (0x0C00, 0x0C7F)),),
    (("Kannada", (0x0C80, 0x0CFF)),),
    (("Malayalam", (0x0D00, 0x0D7F)),),
    (("Thai", (0x0E00, 0x0E7F)),),
    (("Lao", (0x0E80, 0x0EFF)),),
    (("Georgian", (0x10A0, 0x10FF)), ("Georgian Supplement", (0x2D00, 0x2D2F))),
    (("Balinese", (0x1B00, 0x1B7F)),),
    (("Hangul Jamo", (0x1100, 0x11FF)),),
    (
        ("Latin Extended Additional", (0x1E00, 0x1EFF)),
        ("Latin Extended-C", (0x2C60, 0x2C7F)),
        ("Latin Extended-D", (0xA720, 0xA7FF)),
    ),
    (("Greek Extended", (0x1F00, 0x1FFF)),),
    (
        ("General Punctuation", (0x2000, 0x206F)),
        ("Supplemental Punctuation", (0x2E00, 0x2E7F)),
    ),
    (("Superscripts And Subscripts", (0x2070, 0x209F)),),
    (("Currency Symbols", (0x20A0, 0x20CF)),),
    (("Combining Diacritical Marks For Symbols", (0x20D0, 0x20FF)),),
    (("Letterlike Symbols", (0x2100, 0x214F)),),
    (("Number Forms", (0x2150, 0x218F)),),
    (
        ("Arrows", (0x2190, 0x21FF)),
        ("Supplemental Arrows-A", (0x27F0, 0x27FF)),
        ("Supplemental Arrows-B", (0x2900, 0x297F)),
        ("Miscellaneous Symbols and Arrows", (0x2B00, 0x2BFF)),
    ),
    (
        ("Mathematical Operators", (0x2200, 0x22FF)),
        ("Supplemental Mathematical Operators", (0x2A00, 0x2AFF)),
        ("Miscellaneous Mathematical Symbols-A", (0x27C0, 0x27EF)),
        ("Miscellaneous Mathematical Symbols-B", (0x2980, 0x29FF)),
    ),
    (("Miscellaneous Technical", (0x2300, 0x23FF)),),
    (("Control Pictures", (0x2400, 0x243F)),),
    (("Optical Character Recognition", (0x2440, 0x245F)),),
    (("Enclosed Alphanumerics", (0x2460, 0x24FF)),),
    (("Box Drawing", (0x2500, 0x257F)),),
    (("Block Elements", (0x2580, 0x259F)),),
    (("Geometric Shapes", (0x25A0, 0x25FF)),),
    (("Miscellaneous Symbols", (0x2600, 0x26FF)),),
    (("Dingbats", (0x2700, 0x27BF)),),
    (("CJK Symbols And Punctuation", (0x3000, 0x303F)),),
    (("Hiragana", (0x3040, 0x309F)),),
    (
        ("Katakana", (0x30A0, 0x30FF)),
        ("Katakana Phonetic Extensions", (0x31F0, 0x31FF)),
    ),
    (("Bopomofo", (0x3100, 0x312F)), ("Bopomofo Extended", (0x31A0, 0x31BF))),
    (("Hangul Compatibility Jamo", (0x3130, 0x318F)),),
    (("Phags-pa", (0xA840, 0xA87F)),),
    (("Enclosed CJK Letters And Months", (0x3200, 0x32FF)),),
    (("CJK Compatibility", (0x3300, 0x33FF)),),
    (("Hangul Syllables", (0xAC00, 0xD7AF)),),
    (("Non-Plane 0 *", (0xD800, 0xDFFF)),),
    (("Phoenician", (0x10900, 0x1091F)),),
    (
        ("CJK Unified Ideographs", (0x4E00, 0x9FFF)),
        ("CJK Radicals Supplement", (0x2E80, 0x2EFF)),
        ("Kangxi Radicals", (0x2F00, 0x2FDF)),
        ("Ideographic Description Characters", (0x2FF0, 0x2FFF)),
        ("CJK Unified Ideographs Extension A", (0x3400, 0x4DBF)),
        ("CJK Unified Ideographs Extension B", (0x20000, 0x2A6DF)),
        ("Kanbun", (0x3190, 0x319F)),
    ),
    (("Private Use Area (plane 0)", (0xE000, 0xF8FF)),),
    (
        ("CJK Strokes", (0x31C0, 0x31EF)),
        ("CJK Compatibility Ideographs", (0xF900, 0xFAFF)),
        ("CJK Compatibility Ideographs Supplement", (0x2F800, 0x2FA1F)),
    ),
    (("Alphabetic Presentation Forms", (0xFB00, 0xFB4F)),),
    (("Arabic Presentation Forms-A", (0xFB50, 0xFDFF)),),
    (("Combining Half Marks", (0xFE20, 0xFE2F)),),
    (
        ("Vertical Forms", (0xFE10, 0xFE1F)),
        ("CJK Compatibility Forms", (0xFE30, 0xFE4F)),
    ),
    (("Small Form Variants", (0xFE50, 0xFE6F)),),
    (("Arabic Presentation Forms-B", (0xFE70, 0xFEFF)),),
    (("Halfwidth And Fullwidth Forms", (0xFF00, 0xFFEF)),),
    (("Specials", (0xFFF0, 0xFFFF)),),
    (("Tibetan", (0x0F00, 0x0FFF)),),
    (("Syriac", (0x0700, 0x074F)),),
    (("Thaana", (0x0780, 0x07BF)),),
    (("Sinhala", (0x0D80, 0x0DFF)),),
    (("Myanmar", (0x1000, 0x109F)),),
    (
        ("Ethiopic", (0x1200, 0x137F)),
        ("Ethiopic Supplement", (0x1380, 0x139F)),
        ("Ethiopic Extended", (0x2D80, 0x2DDF)),
    ),
    (("Cherokee", (0x13A0, 0x13FF)),),
    (("Unified Canadian Aboriginal Syllabics", (0x1400, 0x167F)),),
    (("Ogham", (0x1680, 0x169F)),),
    (("Runic", (0x16A0, 0x16FF)),),
    (("Khmer", (0x1780, 0x17FF)), ("Khmer Symbols", (0x19E0, 0x19FF))),
    (("Mongolian", (0x1800, 0x18AF)),),
    (("Braille Patterns", (0x2800, 0x28FF)),),
    (("Yi Syllables", (0xA000, 0xA48F)), ("Yi Radicals", (0xA490, 0xA4CF))),
    (
        ("Tagalog", (0x1700, 0x171F)),
        ("Hanunoo", (0x1720, 0x173F)),
        ("Buhid", (0x1740, 0x175F)),
        ("Tagbanwa", (0x1760, 0x177F)),
    ),
    (("Old Italic", (0x10300, 0x1032F)),),
    (("Gothic", (0x10330, 0x1034F)),),
    (("Deseret", (0x10400, 0x1044F)),),
    (
        ("Byzantine Musical Symbols", (0x1D000, 0x1D0FF)),
        ("Musical Symbols", (0x1D100, 0x1D1FF)),
        ("Ancient Greek Musical Notation", (0x1D200, 0x1D24F)),
    ),
    (("Mathematical Alphanumeric Symbols", (0x1D400, 0x1D7FF)),),
    (
        ("Private Use (plane 15)", (0xF0000, 0xFFFFD)),
        ("Private Use (plane 16)", (0x100000, 0x10FFFD)),
    ),
    (
        ("Variation Selectors", (0xFE00, 0xFE0F)),
        ("Variation Selectors Supplement", (0xE0100, 0xE01EF)),
    ),
    (("Tags", (0xE0000, 0xE007F)),),
    (("Limbu", (0x1900, 0x194F)),),
    (("Tai Le", (0x1950, 0x197F)),),
    (("New Tai Lue", (0x1980, 0x19DF)),),
    (("Buginese", (0x1A00, 0x1A1F)),),
    (("Glagolitic", (0x2C00, 0x2C5F)),),
    (("Tifinagh", (0x2D30, 0x2D7F)),),
    (("Yijing Hexagram Symbols", (0x4DC0, 0x4DFF)),),
    (("Syloti Nagri", (0xA800, 0xA82F)),),
    (
        ("Linear B Syllabary", (0x10000, 0x1007F)),
        ("Linear B Ideograms", (0x10080, 0x100FF)),
        ("Aegean Numbers", (0x10100, 0x1013F)),
    ),
    (("Ancient Greek Numbers", (0x10140, 0x1018F)),),
    (("Ugaritic", (0x10380, 0x1039F)),),
    (("Old Persian", (0x103A0, 0x103DF)),),
    (("Shavian", (0x10450, 0x1047F)),),
    (("Osmanya", (0x10480, 0x104AF)),),
    (("Cypriot Syllabary", (0x10800, 0x1083F)),),
    (("Kharoshthi", (0x10A00, 0x10A5F)),),
    (("Tai Xuan Jing Symbols", (0x1D300, 0x1D35F)),),
    (
        ("Cuneiform", (0x12000, 0x123FF)),
        ("Cuneiform Numbers and Punctuation", (0x12400, 0x1247F)),
    ),
    (("Counting Rod Numerals", (0x1D360, 0x1D37F)),),
    (("Sundanese", (0x1B80, 0x1BBF)),),
    (("Lepcha", (0x1C00, 0x1C4F)),),
    (("Ol Chiki", (0x1C50, 0x1C7F)),),
    (("Saurashtra", (0xA880, 0xA8DF)),),
    (("Kayah Li", (0xA900, 0xA92F)),),
    (("Rejang", (0xA930, 0xA95F)),),
    (("Cham", (0xAA00, 0xAA5F)),),
    (("Ancient Symbols", (0x10190, 0x101CF)),),
    (("Phaistos Disc", (0x101D0, 0x101FF)),),
    (
        ("Carian", (0x102A0, 0x102DF)),
        ("Lycian", (0x10280, 0x1029F)),
        ("Lydian", (0x10920, 0x1093F)),
    ),
    (("Domino Tiles", (0x1F030, 0x1F09F)), ("Mahjong Tiles", (0x1F000, 0x1F02F))),
)


_unicodeStarts = []
_unicodeValues = [None]


def _getUnicodeRanges():
    # build the ranges of codepoints for each unicode range bit, and cache result
    if not _unicodeStarts:
        unicodeRanges = [
            (start, (stop, bit))
            for bit, blocks in enumerate(OS2_UNICODE_RANGES)
            for _, (start, stop) in blocks
        ]
        for start, (stop, bit) in sorted(unicodeRanges):
            _unicodeStarts.append(start)
            _unicodeValues.append((stop, bit))
    return _unicodeStarts, _unicodeValues


def intersectUnicodeRanges(unicodes, inverse=False):
    """Intersect a sequence of (int) Unicode codepoints with the Unicode block
    ranges defined in the OpenType specification v1.7, and return the set of
    'ulUnicodeRanges' bits for which there is at least ONE intersection.
    If 'inverse' is True, return the the bits for which there is NO intersection.

    >>> intersectUnicodeRanges([0x0410]) == {9}
    True
    >>> intersectUnicodeRanges([0x0410, 0x1F000]) == {9, 57, 122}
    True
    >>> intersectUnicodeRanges([0x0410, 0x1F000], inverse=True) == (
    ...     set(range(len(OS2_UNICODE_RANGES))) - {9, 57, 122})
    True
    """
    unicodes = set(unicodes)
    unicodestarts, unicodevalues = _getUnicodeRanges()
    bits = set()
    for code in unicodes:
        stop, bit = unicodevalues[bisect.bisect(unicodestarts, code)]
        if code <= stop:
            bits.add(bit)
    # The spec says that bit 57 ("Non Plane 0") implies that there's
    # at least one codepoint beyond the BMP; so I also include all
    # the non-BMP codepoints here
    if any(0x10000 <= code < 0x110000 for code in unicodes):
        bits.add(57)
    return set(range(len(OS2_UNICODE_RANGES))) - bits if inverse else bits


def calcCodePageRanges(unicodes):
    """Given a set of Unicode codepoints (integers), calculate the
    corresponding OS/2 CodePage range bits.
    This is a direct translation of FontForge implementation:
    https://github.com/fontforge/fontforge/blob/7b2c074/fontforge/tottf.c#L3158
    """
    bits = set()
    hasAscii = set(range(0x20, 0x7E)).issubset(unicodes)
    hasLineart = ord("┤") in unicodes

    for uni in unicodes:
        if uni == ord("Þ") and hasAscii:
            bits.add(0)  # Latin 1
        elif uni == ord("Ľ") and hasAscii:
            bits.add(1)  # Latin 2: Eastern Europe
            if hasLineart:
                bits.add(58)  # Latin 2
        elif uni == ord("Б"):
            bits.add(2)  # Cyrillic
            if ord("Ѕ") in unicodes and hasLineart:
                bits.add(57)  # IBM Cyrillic
            if ord("╜") in unicodes and hasLineart:
                bits.add(49)  # MS-DOS Russian
        elif uni == ord("Ά"):
            bits.add(3)  # Greek
            if hasLineart and ord("½") in unicodes:
                bits.add(48)  # IBM Greek
            if hasLineart and ord("√") in unicodes:
                bits.add(60)  # Greek, former 437 G
        elif uni == ord("İ") and hasAscii:
            bits.add(4)  # Turkish
            if hasLineart:
                bits.add(56)  # IBM turkish
        elif uni == ord("א"):
            bits.add(5)  # Hebrew
            if hasLineart and ord("√") in unicodes:
                bits.add(53)  # Hebrew
        elif uni == ord("ر"):
            bits.add(6)  # Arabic
            if ord("√") in unicodes:
                bits.add(51)  # Arabic
            if hasLineart:
                bits.add(61)  # Arabic; ASMO 708
        elif uni == ord("ŗ") and hasAscii:
            bits.add(7)  # Windows Baltic
            if hasLineart:
                bits.add(59)  # MS-DOS Baltic
        elif uni == ord("₫") and hasAscii:
            bits.add(8)  # Vietnamese
        elif uni == ord("ๅ"):
            bits.add(16)  # Thai
        elif uni == ord("エ"):
            bits.add(17)  # JIS/Japan
        elif uni == ord("ㄅ"):
            bits.add(18)  # Chinese: Simplified
        elif uni == ord("ㄱ"):
            bits.add(19)  # Korean wansung
        elif uni == ord("央"):
            bits.add(20)  # Chinese: Traditional
        elif uni == ord("곴"):
            bits.add(21)  # Korean Johab
        elif uni == ord("♥") and hasAscii:
            bits.add(30)  # OEM Character Set
        # TODO: Symbol bit has a special meaning (check the spec), we need
        # to confirm if this is wanted by default.
        # elif chr(0xF000) <= char <= chr(0xF0FF):
        #    codepageRanges.add(31)          # Symbol Character Set
        elif uni == ord("þ") and hasAscii and hasLineart:
            bits.add(54)  # MS-DOS Icelandic
        elif uni == ord("╚") and hasAscii:
            bits.add(62)  # WE/Latin 1
            bits.add(63)  # US
        elif hasAscii and hasLineart and ord("√") in unicodes:
            if uni == ord("Å"):
                bits.add(50)  # MS-DOS Nordic
            elif uni == ord("é"):
                bits.add(52)  # MS-DOS Canadian French
            elif uni == ord("õ"):
                bits.add(55)  # MS-DOS Portuguese

    if hasAscii and ord("‰") in unicodes and ord("∑") in unicodes:
        bits.add(29)  # Macintosh Character Set (US Roman)

    return bits


if __name__ == "__main__":
    import doctest, sys

    sys.exit(doctest.testmod().failed)
<|endoftext|>"""TSI{0,1,2,3,5} are private tables used by Microsoft Visual TrueType (VTT)
tool to store its hinting source data.

TSI1 contains the text of the glyph programs in the form of low-level assembly
code, as well as the 'extra' programs 'fpgm', 'ppgm' (i.e. 'prep'), and 'cvt'.

See also https://learn.microsoft.com/en-us/typography/tools/vtt/tsi-tables
"""

from . import DefaultTable
from fontTools.misc.loggingTools import LogMixin
from fontTools.misc.textTools import strjoin, tobytes, tostr


class table_T_S_I__1(LogMixin, DefaultTable.DefaultTable):
    extras = {0xFFFA: "ppgm", 0xFFFB: "cvt", 0xFFFC: "reserved", 0xFFFD: "fpgm"}

    indextable = "TSI0"

    def decompile(self, data, ttFont):
        totalLength = len(data)
        indextable = ttFont[self.indextable]
        for indices, isExtra in zip(
            (indextable.indices, indextable.extra_indices), (False, True)
        ):
            programs = {}
            for i, (glyphID, textLength, textOffset) in enumerate(indices):
                if isExtra:
                    name = self.extras[glyphID]
                else:
                    name = ttFont.getGlyphName(glyphID)
                if textOffset > totalLength:
                    self.log.warning("textOffset > totalLength; %r skipped" % name)
                    continue
                if textLength < 0x8000:
                    # If the length stored in the record is less than 32768, then use
                    # that as the length of the record.
                    pass
                elif textLength == 0x8000:
                    # If the length is 32768, compute the actual length as follows:
                    isLast = i == (len(indices) - 1)
                    if isLast:
                        if isExtra:
                            # For the last "extra" record (the very last record of the
                            # table), the length is the difference between the total
                            # length of the TSI1 table and the textOffset of the final
                            # record.
                            nextTextOffset = totalLength
                        else:
                            # For the last "normal" record (the last record just prior
                            # to the record containing the "magic number"), the length
                            # is the difference between the textOffset of the record
                            # following the "magic number" (0xFFFE) record (i.e. the
                            # first "extra" record), and the textOffset of the last
                            # "normal" record.
                            nextTextOffset = indextable.extra_indices[0][2]
                    else:
                        # For all other records with a length of 0x8000, the length is
                        # the difference between the textOffset of the record in
                        # question and the textOffset of the next record.
                        nextTextOffset = indices[i + 1][2]
                    assert nextTextOffset >= textOffset, "entries not sorted by offset"
                    if nextTextOffset > totalLength:
                        self.log.warning(
                            "nextTextOffset > totalLength; %r truncated" % name
                        )
                        nextTextOffset = totalLength
                    textLength = nextTextOffset - textOffset
                else:
                    from fontTools import ttLib

                    raise ttLib.TTLibError(
                        "%r textLength (%d) must not be > 32768" % (name, textLength)
                    )
                text = data[textOffset : textOffset + textLength]
                assert len(text) == textLength
                text = tostr(text, encoding="utf-8")
                if text:
                    programs[name] = text
            if isExtra:
                self.extraPrograms = programs
            else:
                self.glyphPrograms = programs

    def compile(self, ttFont):
        if not hasattr(self, "glyphPrograms"):
            self.glyphPrograms = {}
            self.extraPrograms = {}
        data = b""
        indextable = ttFont[self.indextable]
        glyphNames = ttFont.getGlyphOrder()

        indices = []
        for i, name in enumerate(glyphNames):
            if len(data) % 2:
                data = (
                    data + b"\015"
                )  # align on 2-byte boundaries, fill with return chars. Yum.
            if name in self.glyphPrograms:
                text = tobytes(self.glyphPrograms[name], encoding="utf-8")
            else:
                text = b""
            textLength = len(text)
            if textLength >= 0x8000:
                textLength = 0x8000
            indices.append((i, textLength, len(data)))
            data = data + text

        extra_indices = []
        for code, name in sorted(self.extras.items()):
            if len(data) % 2:
                data = (
                    data + b"\015"
                )  # align on 2-byte boundaries, fill with return chars.
            if name in self.extraPrograms:
                text = tobytes(self.extraPrograms[name], encoding="utf-8")
            else:
                text = b""
            textLength = len(text)
            if textLength >= 0x8000:
                textLength = 0x8000
            extra_indices.append((code, textLength, len(data)))
            data = data + text
        indextable.set(indices, extra_indices)
        return data

    def toXML(self, writer, ttFont):
        names = sorted(self.glyphPrograms.keys())
        writer.newline()
        for name in names:
            text = self.glyphPrograms[name]
            if not text:
                continue
            writer.begintag("glyphProgram", name=name)
            writer.newline()
            writer.write_noindent(text.replace("\r", "\n"))
            writer.newline()
            writer.endtag("glyphProgram")
            writer.newline()
            writer.newline()
        extra_names = sorted(self.extraPrograms.keys())
        for name in extra_names:
            text = self.extraPrograms[name]
            if not text:
                continue
            writer.begintag("extraProgram", name=name)
            writer.newline()
            writer.write_noindent(text.replace("\r", "\n"))
            writer.newline()
            writer.endtag("extraProgram")
            writer.newline()
            writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        if not hasattr(self, "glyphPrograms"):
            self.glyphPrograms = {}
            self.extraPrograms = {}
        lines = strjoin(content).replace("\r", "\n").split("\n")
        text = "\r".join(lines[1:-1])
        if name == "glyphProgram":
            self.glyphPrograms[attrs["name"]] = text
        elif name == "extraProgram":
            self.extraPrograms[attrs["name"]] = text
<|endoftext|>from .otBase import BaseTTXConverter


# https://developer.apple.com/fonts/TrueType-Reference-Manual/RM06/Chap6morx.html
class table__m_o_r_x(BaseTTXConverter):
    """The AAT ``morx`` table contains glyph transformations used for script shaping and
    for various other optional smart features, akin to ``GSUB`` and ``GPOS`` features
    in OpenType Layout.

    Note: ``morx`` is a replacement for the now deprecated ``mort`` table.

    See also https://developer.apple.com/fonts/TrueType-Reference-Manual/RM06/Chap6morx.html
    """

    pass
<|endoftext|>from fontTools.misc import sstruct
from fontTools.misc.fixedTools import (
    fixedToFloat as fi2fl,
    floatToFixed as fl2fi,
    floatToFixedToStr as fl2str,
    strToFixedToFloat as str2fl,
)
from fontTools.misc.textTools import bytesjoin, safeEval
from fontTools.ttLib import TTLibError
from . import DefaultTable
import struct
from collections.abc import MutableMapping

# Apple's documentation of 'trak':
# https://developer.apple.com/fonts/TrueType-Reference-Manual/RM06/Chap6trak.html

TRAK_HEADER_FORMAT = """
	> # big endian
	version:     16.16F
	format:      H
	horizOffset: H
	vertOffset:  H
	reserved:    H
"""

TRAK_HEADER_FORMAT_SIZE = sstruct.calcsize(TRAK_HEADER_FORMAT)


TRACK_DATA_FORMAT = """
	> # big endian
	nTracks:         H
	nSizes:          H
	sizeTableOffset: L
"""

TRACK_DATA_FORMAT_SIZE = sstruct.calcsize(TRACK_DATA_FORMAT)


TRACK_TABLE_ENTRY_FORMAT = """
	> # big endian
	track:      16.16F
	nameIndex:       H
	offset:          H
"""

TRACK_TABLE_ENTRY_FORMAT_SIZE = sstruct.calcsize(TRACK_TABLE_ENTRY_FORMAT)


# size values are actually '16.16F' fixed-point values, but here I do the
# fixedToFloat conversion manually instead of relying on sstruct
SIZE_VALUE_FORMAT = ">l"
SIZE_VALUE_FORMAT_SIZE = struct.calcsize(SIZE_VALUE_FORMAT)

# per-Size values are in 'FUnits', i.e. 16-bit signed integers
PER_SIZE_VALUE_FORMAT = ">h"
PER_SIZE_VALUE_FORMAT_SIZE = struct.calcsize(PER_SIZE_VALUE_FORMAT)


class table__t_r_a_k(DefaultTable.DefaultTable):
    """The AAT ``trak`` table can store per-size adjustments to each glyph's
    sidebearings to make when tracking is enabled, which applications can
    use to provide more visually balanced line spacing.

    See also https://developer.apple.com/fonts/TrueType-Reference-Manual/RM06/Chap6trak.html
    """

    def compile(self, ttFont):
        dataList = []
        offset = TRAK_HEADER_FORMAT_SIZE
        for direction in ("horiz", "vert"):
            trackData = getattr(self, direction + "Data", TrackData())
            offsetName = direction + "Offset"
            # set offset to 0 if None or empty
            if not trackData:
                setattr(self, offsetName, 0)
                continue
            # TrackData table format must be longword aligned
            alignedOffset = (offset + 3) & ~3
            padding, offset = b"\x00" * (alignedOffset - offset), alignedOffset
            setattr(self, offsetName, offset)

            data = trackData.compile(offset)
            offset += len(data)
            dataList.append(padding + data)

        self.reserved = 0
        tableData = bytesjoin([sstruct.pack(TRAK_HEADER_FORMAT, self)] + dataList)
        return tableData

    def decompile(self, data, ttFont):
        sstruct.unpack(TRAK_HEADER_FORMAT, data[:TRAK_HEADER_FORMAT_SIZE], self)
        for direction in ("horiz", "vert"):
            trackData = TrackData()
            offset = getattr(self, direction + "Offset")
            if offset != 0:
                trackData.decompile(data, offset)
            setattr(self, direction + "Data", trackData)

    def toXML(self, writer, ttFont):
        writer.simpletag("version", value=self.version)
        writer.newline()
        writer.simpletag("format", value=self.format)
        writer.newline()
        for direction in ("horiz", "vert"):
            dataName = direction + "Data"
            writer.begintag(dataName)
            writer.newline()
            trackData = getattr(self, dataName, TrackData())
            trackData.toXML(writer, ttFont)
            writer.endtag(dataName)
            writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        if name == "version":
            self.version = safeEval(attrs["value"])
        elif name == "format":
            self.format = safeEval(attrs["value"])
        elif name in ("horizData", "vertData"):
            trackData = TrackData()
            setattr(self, name, trackData)
            for element in content:
                if not isinstance(element, tuple):
                    continue
                name, attrs, content_ = element
                trackData.fromXML(name, attrs, content_, ttFont)


class TrackData(MutableMapping):
    def __init__(self, initialdata={}):
        self._map = dict(initialdata)

    def compile(self, offset):
        nTracks = len(self)
        sizes = self.sizes()
        nSizes = len(sizes)

        # offset to the start of the size subtable
        offset += TRACK_DATA_FORMAT_SIZE + TRACK_TABLE_ENTRY_FORMAT_SIZE * nTracks
        trackDataHeader = sstruct.pack(
            TRACK_DATA_FORMAT,
            {"nTracks": nTracks, "nSizes": nSizes, "sizeTableOffset": offset},
        )

        entryDataList = []
        perSizeDataList = []
        # offset to per-size tracking values
        offset += SIZE_VALUE_FORMAT_SIZE * nSizes
        # sort track table entries by track value
        for track, entry in sorted(self.items()):
            assert entry.nameIndex is not None
            entry.track = track
            entry.offset = offset
            entryDataList += [sstruct.pack(TRACK_TABLE_ENTRY_FORMAT, entry)]
            # sort per-size values by size
            for size, value in sorted(entry.items()):
                perSizeDataList += [struct.pack(PER_SIZE_VALUE_FORMAT, value)]
            offset += PER_SIZE_VALUE_FORMAT_SIZE * nSizes
        # sort size values
        sizeDataList = [
            struct.pack(SIZE_VALUE_FORMAT, fl2fi(sv, 16)) for sv in sorted(sizes)
        ]

        data = bytesjoin(
            [trackDataHeader] + entryDataList + sizeDataList + perSizeDataList
        )
        return data

    def decompile(self, data, offset):
        # initial offset is from the start of trak table to the current TrackData
        trackDataHeader = data[offset : offset + TRACK_DATA_FORMAT_SIZE]
        if len(trackDataHeader) != TRACK_DATA_FORMAT_SIZE:
            raise TTLibError("not enough data to decompile TrackData header")
        sstruct.unpack(TRACK_DATA_FORMAT, trackDataHeader, self)
        offset += TRACK_DATA_FORMAT_SIZE

        nSizes = self.nSizes
        sizeTableOffset = self.sizeTableOffset
        sizeTable = []
        for i in range(nSizes):
            sizeValueData = data[
                sizeTableOffset : sizeTableOffset + SIZE_VALUE_FORMAT_SIZE
            ]
            if len(sizeValueData) < SIZE_VALUE_FORMAT_SIZE:
                raise TTLibError("not enough data to decompile TrackData size subtable")
            (sizeValue,) = struct.unpack(SIZE_VALUE_FORMAT, sizeValueData)
            sizeTable.append(fi2fl(sizeValue, 16))
            sizeTableOffset += SIZE_VALUE_FORMAT_SIZE

        for i in range(self.nTracks):
            entry = TrackTableEntry()
            entryData = data[offset : offset + TRACK_TABLE_ENTRY_FORMAT_SIZE]
            if len(entryData) < TRACK_TABLE_ENTRY_FORMAT_SIZE:
                raise TTLibError("not enough data to decompile TrackTableEntry record")
            sstruct.unpack(TRACK_TABLE_ENTRY_FORMAT, entryData, entry)
            perSizeOffset = entry.offset
            for j in range(nSizes):
                size = sizeTable[j]
                perSizeValueData = data[
                    perSizeOffset : perSizeOffset + PER_SIZE_VALUE_FORMAT_SIZE
                ]
                if len(perSizeValueData) < PER_SIZE_VALUE_FORMAT_SIZE:
                    raise TTLibError(
                        "not enough data to decompile per-size track values"
                    )
                (perSizeValue,) = struct.unpack(PER_SIZE_VALUE_FORMAT, perSizeValueData)
                entry[size] = perSizeValue
                perSizeOffset += PER_SIZE_VALUE_FORMAT_SIZE
            self[entry.track] = entry
            offset += TRACK_TABLE_ENTRY_FORMAT_SIZE

    def toXML(self, writer, ttFont):
        nTracks = len(self)
        nSizes = len(self.sizes())
        writer.comment("nTracks=%d, nSizes=%d" % (nTracks, nSizes))
        writer.newline()
        for track, entry in sorted(self.items()):
            assert entry.nameIndex is not None
            entry.track = track
            entry.toXML(writer, ttFont)

    def fromXML(self, name, attrs, content, ttFont):
        if name != "trackEntry":
            return
        entry = TrackTableEntry()
        entry.fromXML(name, attrs, content, ttFont)
        self[entry.track] = entry

    def sizes(self):
        if not self:
            return frozenset()
        tracks = list(self.tracks())
        sizes = self[tracks.pop(0)].sizes()
        for track in tracks:
            entrySizes = self[track].sizes()
            if sizes != entrySizes:
                raise TTLibError(
                    "'trak' table entries must specify the same sizes: "
                    "%s != %s" % (sorted(sizes), sorted(entrySizes))
                )
        return frozenset(sizes)

    def __getitem__(self, track):
        return self._map[track]

    def __delitem__(self, track):
        del self._map[track]

    def __setitem__(self, track, entry):
        self._map[track] = entry

    def __len__(self):
        return len(self._map)

    def __iter__(self):
        return iter(self._map)

    def keys(self):
        return self._map.keys()

    tracks = keys

    def __repr__(self):
        return "TrackData({})".format(self._map if self else "")


class TrackTableEntry(MutableMapping):
    def __init__(self, values={}, nameIndex=None):
        self.nameIndex = nameIndex
        self._map = dict(values)

    def toXML(self, writer, ttFont):
        name = ttFont["name"].getDebugName(self.nameIndex)
        writer.begintag(
            "trackEntry",
            (("value", fl2str(self.track, 16)), ("nameIndex", self.nameIndex)),
        )
        writer.newline()
        if name:
            writer.comment(name)
            writer.newline()
        for size, perSizeValue in sorted(self.items()):
            writer.simpletag("track", size=fl2str(size, 16), value=perSizeValue)
            writer.newline()
        writer.endtag("trackEntry")
        writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        self.track = str2fl(attrs["value"], 16)
        self.nameIndex = safeEval(attrs["nameIndex"])
        for element in content:
            if not isinstance(element, tuple):
                continue
            name, attrs, _ = element
            if name != "track":
                continue
            size = str2fl(attrs["size"], 16)
            self[size] = safeEval(attrs["value"])

    def __getitem__(self, size):
        return self._map[size]

    def __delitem__(self, size):
        del self._map[size]

    def __setitem__(self, size, value):
        self._map[size] = value

    def __len__(self):
        return len(self._map)

    def __iter__(self):
        return iter(self._map)

    def keys(self):
        return self._map.keys()

    sizes = keys

    def __repr__(self):
        return "TrackTableEntry({}, nameIndex={})".format(self._map, self.nameIndex)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.nameIndex == other.nameIndex and dict(self) == dict(other)

    def __ne__(self, other):
        result = self.__eq__(other)
        return result if result is NotImplemented else not result
<|endoftext|>from fontTools.misc.textTools import safeEval
from . import DefaultTable
import sys
import array


class table__c_v_t(DefaultTable.DefaultTable):
    """Control Value Table

    The Control Value Table holds a list of values that can be referenced
    by TrueType font instructions.

    See also https://learn.microsoft.com/en-us/typography/opentype/spec/cvt
    """

    def decompile(self, data, ttFont):
        values = array.array("h")
        values.frombytes(data)
        if sys.byteorder != "big":
            values.byteswap()
        self.values = values

    def compile(self, ttFont):
        if not hasattr(self, "values"):
            return b""
        values = self.values[:]
        if sys.byteorder != "big":
            values.byteswap()
        return values.tobytes()

    def toXML(self, writer, ttFont):
        for i, value in enumerate(self.values):
            writer.simpletag("cv", value=value, index=i)
            writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        if not hasattr(self, "values"):
            self.values = array.array("h")
        if name == "cv":
            index = safeEval(attrs["index"])
            value = safeEval(attrs["value"])
            for i in range(1 + index - len(self.values)):
                self.values.append(0)
            self.values[index] = value

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]

    def __setitem__(self, index, value):
        self.values[index] = value

    def __delitem__(self, index):
        del self.values[index]
<|endoftext|>from fontTools.misc.textTools import safeEval
from . import DefaultTable
import struct
import array

# XXX I've lowered the strictness, to make sure Apple's own Chicago
# XXX gets through. They're looking into it, I hope to raise the standards
# XXX back to normal eventually.


class table_L_T_S_H_(DefaultTable.DefaultTable):
    """Linear Threshold table

    The ``LTSH`` table contains per-glyph settings indicating the ppem sizes
    at which the advance width metric should be scaled linearly, despite the
    effects of any TrueType instructions that might otherwise alter the
    advance width.

    See also https://learn.microsoft.com/en-us/typography/opentype/spec/ltsh
    """

    def decompile(self, data, ttFont):
        version, numGlyphs = struct.unpack(">HH", data[:4])
        data = data[4:]
        assert version == 0, "unknown version: %s" % version
        assert (len(data) % numGlyphs) < 4, "numGlyphs doesn't match data length"
        # ouch: the assertion is not true in Chicago!
        # assert numGlyphs == ttFont['maxp'].numGlyphs
        yPels = array.array("B")
        yPels.frombytes(data)
        self.yPels = {}
        for i in range(numGlyphs):
            self.yPels[ttFont.getGlyphName(i)] = yPels[i]

    def compile(self, ttFont):
        version = 0
        names = list(self.yPels.keys())
        numGlyphs = len(names)
        yPels = [0] * numGlyphs
        # ouch: the assertion is not true in Chicago!
        # assert len(self.yPels) == ttFont['maxp'].numGlyphs == numGlyphs
        for name in names:
            yPels[ttFont.getGlyphID(name)] = self.yPels[name]
        yPels = array.array("B", yPels)
        return struct.pack(">HH", version, numGlyphs) + yPels.tobytes()

    def toXML(self, writer, ttFont):
        names = sorted(self.yPels.keys())
        for name in names:
            writer.simpletag("yPel", name=name, value=self.yPels[name])
            writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        if not hasattr(self, "yPels"):
            self.yPels = {}
        if name != "yPel":
            return  # ignore unknown tags
        self.yPels[attrs["name"]] = safeEval(attrs["value"])
<|endoftext|>from . import DefaultTable
from . import ttProgram


class table__f_p_g_m(DefaultTable.DefaultTable):
    """Font Program table

    The ``fpgm`` table typically contains function defintions that are
    used by font instructions. This Font Program is similar to the Control
    Value Program that is stored in the ``prep`` table, but
    the ``fpgm`` table is only executed one time, when the font is first
    used.

    See also https://learn.microsoft.com/en-us/typography/opentype/spec/fpgm
    """

    def decompile(self, data, ttFont):
        program = ttProgram.Program()
        program.fromBytecode(data)
        self.program = program

    def compile(self, ttFont):
        if hasattr(self, "program"):
            return self.program.getBytecode()
        return b""

    def toXML(self, writer, ttFont):
        self.program.toXML(writer, ttFont)

    def fromXML(self, name, attrs, content, ttFont):
        program = ttProgram.Program()
        program.fromXML(name, attrs, content, ttFont)
        self.program = program

    def __bool__(self):
        """
        >>> fpgm = table__f_p_g_m()
        >>> bool(fpgm)
        False
        >>> p = ttProgram.Program()
        >>> fpgm.program = p
        >>> bool(fpgm)
        False
        >>> bc = bytearray([0])
        >>> p.fromBytecode(bc)
        >>> bool(fpgm)
        True
        >>> p.bytecode.pop()
        0
        >>> bool(fpgm)
        False
        """
        return hasattr(self, "program") and bool(self.program)

    __nonzero__ = __bool__


if __name__ == "__main__":
    import sys
    import doctest

    sys.exit(doctest.testmod().failed)
<|endoftext|>from fontTools.misc import sstruct
from fontTools.misc.fixedTools import floatToFixedToStr
from fontTools.misc.textTools import byteord, safeEval

# from itertools import *
from . import DefaultTable
from . import grUtils
from array import array
from functools import reduce
import struct, re, sys

Silf_hdr_format = """
    >
    version:            16.16F
"""

Silf_hdr_format_3 = """
    >
    version:            16.16F
    compilerVersion:    L
    numSilf:            H
                        x
                        x
"""

Silf_part1_format_v3 = """
    >
    ruleVersion:        16.16F
    passOffset:         H
    pseudosOffset:      H
"""

Silf_part1_format = """
    >
    maxGlyphID:         H
    extraAscent:        h
    extraDescent:       h
    numPasses:          B
    iSubst:             B
    iPos:               B
    iJust:              B
    iBidi:              B
    flags:              B
    maxPreContext:      B
    maxPostContext:     B
    attrPseudo:         B
    attrBreakWeight:    B
    attrDirectionality: B
    attrMirroring:      B
    attrSkipPasses:     B
    numJLevels:         B
"""

Silf_justify_format = """
    >
    attrStretch:        B
    attrShrink:         B
    attrStep:           B
    attrWeight:         B
    runto:              B
                        x
                        x
                        x
"""

Silf_part2_format = """
    >
    numLigComp:         H
    numUserDefn:        B
    maxCompPerLig:      B
    direction:          B
    attCollisions:      B
                        x
                        x
                        x
    numCritFeatures:    B
"""

Silf_pseudomap_format = """
    >
    unicode:            L
    nPseudo:            H
"""

Silf_pseudomap_format_h = """
    >
    unicode:            H
    nPseudo:            H
"""

Silf_classmap_format = """
    >
    numClass:           H
    numLinear:          H
"""

Silf_lookupclass_format = """
    >
    numIDs:             H
    searchRange:        H
    entrySelector:      H
    rangeShift:         H
"""

Silf_lookuppair_format = """
    >
    glyphId:            H
    index:              H
"""

Silf_pass_format = """
    >
    flags:              B
    maxRuleLoop:        B
    maxRuleContext:     B
    maxBackup:          B
    numRules:           H
    fsmOffset:          H
    pcCode:             L
    rcCode:             L
    aCode:              L
    oDebug:             L
    numRows:            H
    numTransitional:    H
    numSuccess:         H
    numColumns:         H
"""

aCode_info = (
    ("NOP", 0),
    ("PUSH_BYTE", "b"),
    ("PUSH_BYTE_U", "B"),
    ("PUSH_SHORT", ">h"),
    ("PUSH_SHORT_U", ">H"),
    ("PUSH_LONG", ">L"),
    ("ADD", 0),
    ("SUB", 0),
    ("MUL", 0),
    ("DIV", 0),
    ("MIN", 0),
    ("MAX", 0),
    ("NEG", 0),
    ("TRUNC8", 0),
    ("TRUNC16", 0),
    ("COND", 0),
    ("AND", 0),  # x10
    ("OR", 0),
    ("NOT", 0),
    ("EQUAL", 0),
    ("NOT_EQ", 0),
    ("LESS", 0),
    ("GTR", 0),
    ("LESS_EQ", 0),
    ("GTR_EQ", 0),
    ("NEXT", 0),
    ("NEXT_N", "b"),
    ("COPY_NEXT", 0),
    ("PUT_GLYPH_8BIT_OBS", "B"),
    ("PUT_SUBS_8BIT_OBS", "bBB"),
    ("PUT_COPY", "b"),
    ("INSERT", 0),
    ("DELETE", 0),  # x20
    ("ASSOC", -1),
    ("CNTXT_ITEM", "bB"),
    ("ATTR_SET", "B"),
    ("ATTR_ADD", "B"),
    ("ATTR_SUB", "B"),
    ("ATTR_SET_SLOT", "B"),
    ("IATTR_SET_SLOT", "BB"),
    ("PUSH_SLOT_ATTR", "Bb"),
    ("PUSH_GLYPH_ATTR_OBS", "Bb"),
    ("PUSH_GLYPH_METRIC", "Bbb"),
    ("PUSH_FEAT", "Bb"),
    ("PUSH_ATT_TO_GATTR_OBS", "Bb"),
    ("PUSH_ATT_TO_GLYPH_METRIC", "Bbb"),
    ("PUSH_ISLOT_ATTR", "Bbb"),
    ("PUSH_IGLYPH_ATTR", "Bbb"),
    ("POP_RET", 0),  # x30
    ("RET_ZERO", 0),
    ("RET_TRUE", 0),
    ("IATTR_SET", "BB"),
    ("IATTR_ADD", "BB"),
    ("IATTR_SUB", "BB"),
    ("PUSH_PROC_STATE", "B"),
    ("PUSH_VERSION", 0),
    ("PUT_SUBS", ">bHH"),
    ("PUT_SUBS2", 0),
    ("PUT_SUBS3", 0),
    ("PUT_GLYPH", ">H"),
    ("PUSH_GLYPH_ATTR", ">Hb"),
    ("PUSH_ATT_TO_GLYPH_ATTR", ">Hb"),
    ("BITOR", 0),
    ("BITAND", 0),
    ("BITNOT", 0),  # x40
    ("BITSET", ">HH"),
    ("SET_FEAT", "Bb"),
)
aCode_map = dict([(x[0], (i, x[1])) for i, x in enumerate(aCode_info)])


def disassemble(aCode):
    codelen = len(aCode)
    pc = 0
    res = []
    while pc < codelen:
        opcode = byteord(aCode[pc : pc + 1])
        if opcode > len(aCode_info):
            instr = aCode_info[0]
        else:
            instr = aCode_info[opcode]
        pc += 1
        if instr[1] != 0 and pc >= codelen:
            return res
        if instr[1] == -1:
            count = byteord(aCode[pc])
            fmt = "%dB" % count
            pc += 1
        elif instr[1] == 0:
            fmt = ""
        else:
            fmt = instr[1]
        if fmt == "":
            res.append(instr[0])
            continue
        parms = struct.unpack_from(fmt, aCode[pc:])
        res.append(instr[0] + "(" + ", ".join(map(str, parms)) + ")")
        pc += struct.calcsize(fmt)
    return res


instre = re.compile(r"^\s*([^(]+)\s*(?:\(([^)]+)\))?")


def assemble(instrs):
    res = b""
    for inst in instrs:
        m = instre.match(inst)
        if not m or not m.group(1) in aCode_map:
            continue
        opcode, parmfmt = aCode_map[m.group(1)]
        res += struct.pack("B", opcode)
        if m.group(2):
            if parmfmt == 0:
                continue
            parms = [int(x) for x in re.split(r",\s*", m.group(2))]
            if parmfmt == -1:
                l = len(parms)
                res += struct.pack(("%dB" % (l + 1)), l, *parms)
            else:
                res += struct.pack(parmfmt, *parms)
    return res


def writecode(tag, writer, instrs):
    writer.begintag(tag)
    writer.newline()
    for l in disassemble(instrs):
        writer.write(l)
        writer.newline()
    writer.endtag(tag)
    writer.newline()


def readcode(content):
    res = []
    for e in content_string(content).split("\n"):
        e = e.strip()
        if not len(e):
            continue
        res.append(e)
    return assemble(res)


attrs_info = (
    "flags",
    "extraAscent",
    "extraDescent",
    "maxGlyphID",
    "numLigComp",
    "numUserDefn",
    "maxCompPerLig",
    "direction",
    "lbGID",
)
attrs_passindexes = ("iSubst", "iPos", "iJust", "iBidi")
attrs_contexts = ("maxPreContext", "maxPostContext")
attrs_attributes = (
    "attrPseudo",
    "attrBreakWeight",
    "attrDirectionality",
    "attrMirroring",
    "attrSkipPasses",
    "attCollisions",
)
pass_attrs_info = (
    "flags",
    "maxRuleLoop",
    "maxRuleContext",
    "maxBackup",
    "minRulePreContext",
    "maxRulePreContext",
    "collisionThreshold",
)
pass_attrs_fsm = ("numRows", "numTransitional", "numSuccess", "numColumns")


def writesimple(tag, self, writer, *attrkeys):
    attrs = dict([(k, getattr(self, k)) for k in attrkeys])
    writer.simpletag(tag, **attrs)
    writer.newline()


def getSimple(self, attrs, *attr_list):
    for k in attr_list:
        if k in attrs:
            setattr(self, k, int(safeEval(attrs[k])))


def content_string(contents):
    res = ""
    for element in contents:
        if isinstance(element, tuple):
            continue
        res += element
    return res.strip()


def wrapline(writer, dat, length=80):
    currline = ""
    for d in dat:
        if len(currline) > length:
            writer.write(currline[:-1])
            writer.newline()
            currline = ""
        currline += d + " "
    if len(currline):
        writer.write(currline[:-1])
        writer.newline()


class _Object:
    pass


class table_S__i_l_f(DefaultTable.DefaultTable):
    """Graphite Rules table

    See also https://graphite.sil.org/graphite_techAbout#graphite-font-tables
    """

    def __init__(self, tag=None):
        DefaultTable.DefaultTable.__init__(self, tag)
        self.silfs = []

    def decompile(self, data, ttFont):
        sstruct.unpack2(Silf_hdr_format, data, self)
        self.version = float(floatToFixedToStr(self.version, precisionBits=16))
        if self.version >= 5.0:
            data, self.scheme = grUtils.decompress(data)
            sstruct.unpack2(Silf_hdr_format_3, data, self)
            base = sstruct.calcsize(Silf_hdr_format_3)
        elif self.version < 3.0:
            self.numSilf = struct.unpack(">H", data[4:6])
            self.scheme = 0
            self.compilerVersion = 0
            base = 8
        else:
            self.scheme = 0
            sstruct.unpack2(Silf_hdr_format_3, data, self)
            base = sstruct.calcsize(Silf_hdr_format_3)

        silfoffsets = struct.unpack_from((">%dL" % self.numSilf), data[base:])
        for offset in silfoffsets:
            s = Silf()
            self.silfs.append(s)
            s.decompile(data[offset:], ttFont, self.version)

    def compile(self, ttFont):
        self.numSilf = len(self.silfs)
        if self.version < 3.0:
            hdr = sstruct.pack(Silf_hdr_format, self)
            hdr += struct.pack(">HH", self.numSilf, 0)
        else:
            hdr = sstruct.pack(Silf_hdr_format_3, self)
        offset = len(hdr) + 4 * self.numSilf
        data = b""
        for s in self.silfs:
            hdr += struct.pack(">L", offset)
            subdata = s.compile(ttFont, self.version)
            offset += len(subdata)
            data += subdata
        if self.version >= 5.0:
            return grUtils.compress(self.scheme, hdr + data)
        return hdr + data

    def toXML(self, writer, ttFont):
        writer.comment("Attributes starting with _ are informative only")
        writer.newline()
        writer.simpletag(
            "version",
            version=self.version,
            compilerVersion=self.compilerVersion,
            compressionScheme=self.scheme,
        )
        writer.newline()
        for s in self.silfs:
            writer.begintag("silf")
            writer.newline()
            s.toXML(writer, ttFont, self.version)
            writer.endtag("silf")
            writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        if name == "version":
            self.scheme = int(safeEval(attrs["compressionScheme"]))
            self.version = float(safeEval(attrs["version"]))
            self.compilerVersion = int(safeEval(attrs["compilerVersion"]))
            return
        if name == "silf":
            s = Silf()
            self.silfs.append(s)
            for element in content:
                if not isinstance(element, tuple):
                    continue
                tag, attrs, subcontent = element
                s.fromXML(tag, attrs, subcontent, ttFont, self.version)


class Silf(object):
    """A particular Silf subtable"""

    def __init__(self):
        self.passes = []
        self.scriptTags = []
        self.critFeatures = []
        self.jLevels = []
        self.pMap = {}

    def decompile(self, data, ttFont, version=2.0):
        if version >= 3.0:
            _, data = sstruct.unpack2(Silf_part1_format_v3, data, self)
            self.ruleVersion = float(
                floatToFixedToStr(self.ruleVersion, precisionBits=16)
            )
        _, data = sstruct.unpack2(Silf_part1_format, data, self)
        for jlevel in range(self.numJLevels):
            j, data = sstruct.unpack2(Silf_justify_format, data, _Object())
            self.jLevels.append(j)
        _, data = sstruct.unpack2(Silf_part2_format, data, self)
        if self.numCritFeatures:
            self.critFeatures = struct.unpack_from(
                (">%dH" % self.numCritFeatures), data
            )
        data = data[self.numCritFeatures * 2 + 1 :]
        (numScriptTag,) = struct.unpack_from("B", data)
        if numScriptTag:
            self.scriptTags = [
                struct.unpack("4s", data[x : x + 4])[0].decode("ascii")
                for x in range(1, 1 + 4 * numScriptTag, 4)
            ]
        data = data[1 + 4 * numScriptTag :]
        (self.lbGID,) = struct.unpack(">H", data[:2])
        if self.numPasses:
            self.oPasses = struct.unpack(
                (">%dL" % (self.numPasses + 1)), data[2 : 6 + 4 * self.numPasses]
            )
        data = data[6 + 4 * self.numPasses :]
        (numPseudo,) = struct.unpack(">H", data[:2])
        for i in range(numPseudo):
            if version >= 3.0:
                pseudo = sstruct.unpack(
                    Silf_pseudomap_format, data[8 + 6 * i : 14 + 6 * i], _Object()
                )
            else:
                pseudo = sstruct.unpack(
                    Silf_pseudomap_format_h, data[8 + 4 * i : 12 + 4 * i], _Object()
                )
            self.pMap[pseudo.unicode] = ttFont.getGlyphName(pseudo.nPseudo)
        data = data[8 + 6 * numPseudo :]
        currpos = (
            sstruct.calcsize(Silf_part1_format)
            + sstruct.calcsize(Silf_justify_format) * self.numJLevels
            + sstruct.calcsize(Silf_part2_format)
            + 2 * self.numCritFeatures
            + 1
            + 1
            + 4 * numScriptTag
            + 6
            + 4 * self.numPasses
            + 8
            + 6 * numPseudo
        )
        if version >= 3.0:
            currpos += sstruct.calcsize(Silf_part1_format_v3)
        self.classes = Classes()
        self.classes.decompile(data, ttFont, version)
        for i in range(self.numPasses):
            p = Pass()
            self.passes.append(p)
            p.decompile(
                data[self.oPasses[i] - currpos : self.oPasses[i + 1] - currpos],
                ttFont,
                version,
            )

    def compile(self, ttFont, version=2.0):
        self.numPasses = len(self.passes)
        self.numJLevels = len(self.jLevels)
        self.numCritFeatures = len(self.critFeatures)
        numPseudo = len(self.pMap)
        data = b""
        if version >= 3.0:
            hdroffset = sstruct.calcsize(Silf_part1_format_v3)
        else:
            hdroffset = 0
        data += sstruct.pack(Silf_part1_format, self)
        for j in self.jLevels:
            data += sstruct.pack(Silf_justify_format, j)
        data += sstruct.pack(Silf_part2_format, self)
        if self.numCritFeatures:
            data += struct.pack((">%dH" % self.numCritFeaturs), *self.critFeatures)
        data += struct.pack("BB", 0, len(self.scriptTags))
        if len(self.scriptTags):
            tdata = [struct.pack("4s", x.encode("ascii")) for x in self.scriptTags]
            data += b"".join(tdata)
        data += struct.pack(">H", self.lbGID)
        self.passOffset = len(data)

        data1 = grUtils.bininfo(numPseudo, 6)
        currpos = hdroffset + len(data) + 4 * (self.numPasses + 1)
        self.pseudosOffset = currpos + len(data1)
        for u, p in sorted(self.pMap.items()):
            data1 += struct.pack(
                (">LH" if version >= 3.0 else ">HH"), u, ttFont.getGlyphID(p)
            )
        data1 += self.classes.compile(ttFont, version)
        currpos += len(data1)
        data2 = b""
        datao = b""
        for i, p in enumerate(self.passes):
            base = currpos + len(data2)
            datao += struct.pack(">L", base)
            data2 += p.compile(ttFont, base, version)
        datao += struct.pack(">L", currpos + len(data2))

        if version >= 3.0:
            data3 = sstruct.pack(Silf_part1_format_v3, self)
        else:
            data3 = b""
        return data3 + data + datao + data1 + data2

    def toXML(self, writer, ttFont, version=2.0):
        if version >= 3.0:
            writer.simpletag("version", ruleVersion=self.ruleVersion)
            writer.newline()
        writesimple("info", self, writer, *attrs_info)
        writesimple("passindexes", self, writer, *attrs_passindexes)
        writesimple("contexts", self, writer, *attrs_contexts)
        writesimple("attributes", self, writer, *attrs_attributes)
        if len(self.jLevels):
            writer.begintag("justifications")
            writer.newline()
            jformat, jnames, jfixes = sstruct.getformat(Silf_justify_format)
            for i, j in enumerate(self.jLevels):
                attrs = dict([(k, getattr(j, k)) for k in jnames])
                writer.simpletag("justify", **attrs)
                writer.newline()
            writer.endtag("justifications")
            writer.newline()
        if len(self.critFeatures):
            writer.begintag("critFeatures")
            writer.newline()
            writer.write(" ".join(map(str, self.critFeatures)))
            writer.newline()
            writer.endtag("critFeatures")
            writer.newline()
        if len(self.scriptTags):
            writer.begintag("scriptTags")
            writer.newline()
            writer.write(" ".join(self.scriptTags))
            writer.newline()
            writer.endtag("scriptTags")
            writer.newline()
        if self.pMap:
            writer.begintag("pseudoMap")
            writer.newline()
            for k, v in sorted(self.pMap.items()):
                writer.simpletag("pseudo", unicode=hex(k), pseudo=v)
                writer.newline()
            writer.endtag("pseudoMap")
            writer.newline()
        self.classes.toXML(writer, ttFont, version)
        if len(self.passes):
            writer.begintag("passes")
            writer.newline()
            for i, p in enumerate(self.passes):
                writer.begintag("pass", _index=i)
                writer.newline()
                p.toXML(writer, ttFont, version)
                writer.endtag("pass")
                writer.newline()
            writer.endtag("passes")
            writer.newline()

    def fromXML(self, name, attrs, content, ttFont, version=2.0):
        if name == "version":
            self.ruleVersion = float(safeEval(attrs.get("ruleVersion", "0")))
        if name == "info":
            getSimple(self, attrs, *attrs_info)
        elif name == "passindexes":
            getSimple(self, attrs, *attrs_passindexes)
        elif name == "contexts":
            getSimple(self, attrs, *attrs_contexts)
        elif name == "attributes":
            getSimple(self, attrs, *attrs_attributes)
        elif name == "justifications":
            for element in content:
                if not isinstance(element, tuple):
                    continue
                tag, attrs, subcontent = element
                if tag == "justify":
                    j = _Object()
                    for k, v in attrs.items():
                        setattr(j, k, int(v))
                    self.jLevels.append(j)
        elif name == "critFeatures":
            self.critFeatures = []
            element = content_string(content)
            self.critFeatures.extend(map(int, element.split()))
        elif name == "scriptTags":
            self.scriptTags = []
            element = content_string(content)
            for n in element.split():
                self.scriptTags.append(n)
        elif name == "pseudoMap":
            self.pMap = {}
            for element in content:
                if not isinstance(element, tuple):
                    continue
                tag, attrs, subcontent = element
                if tag == "pseudo":
                    k = int(attrs["unicode"], 16)
                    v = attrs["pseudo"]
                self.pMap[k] = v
        elif name == "classes":
            self.classes = Classes()
            for element in content:
                if not isinstance(element, tuple):
                    continue
                tag, attrs, subcontent = element
                self.classes.fromXML(tag, attrs, subcontent, ttFont, version)
        elif name == "passes":
            for element in content:
                if not isinstance(element, tuple):
                    continue
                tag, attrs, subcontent = element
                if tag == "pass":
                    p = Pass()
                    for e in subcontent:
                        if not isinstance(e, tuple):
                            continue
                        p.fromXML(e[0], e[1], e[2], ttFont, version)
                    self.passes.append(p)


class Classes(object):
    def __init__(self):
        self.linear = []
        self.nonLinear = []

    def decompile(self, data, ttFont, version=2.0):
        sstruct.unpack2(Silf_classmap_format, data, self)
        if version >= 4.0:
            oClasses = struct.unpack(
                (">%dL" % (self.numClass + 1)), data[4 : 8 + 4 * self.numClass]
            )
        else:
            oClasses = struct.unpack(
                (">%dH" % (self.numClass + 1)), data[4 : 6 + 2 * self.numClass]
            )
        for s, e in zip(oClasses[: self.numLinear], oClasses[1 : self.numLinear + 1]):
            self.linear.append(
                ttFont.getGlyphName(x)
                for x in struct.unpack((">%dH" % ((e - s) / 2)), data[s:e])
            )
        for s, e in zip(
            oClasses[self.numLinear : self.numClass],
            oClasses[self.numLinear + 1 : self.numClass + 1],
        ):
            nonLinids = [
                struct.unpack(">HH", data[x : x + 4]) for x in range(s + 8, e, 4)
            ]
            nonLin = dict([(ttFont.getGlyphName(x[0]), x[1]) for x in nonLinids])
            self.nonLinear.append(nonLin)

    def compile(self, ttFont, version=2.0):
        data = b""
        oClasses = []
        if version >= 4.0:
            offset = 8 + 4 * (len(self.linear) + len(self.nonLinear))
        else:
            offset = 6 + 2 * (len(self.linear) + len(self.nonLinear))
        for l in self.linear:
            oClasses.append(len(data) + offset)
            gs = [ttFont.getGlyphID(x) for x in l]
            data += struct.pack((">%dH" % len(l)), *gs)
        for l in self.nonLinear:
            oClasses.append(len(data) + offset)
            gs = [(ttFont.getGlyphID(x[0]), x[1]) for x in l.items()]
            data += grUtils.bininfo(len(gs))
            data += b"".join([struct.pack(">HH", *x) for x in sorted(gs)])
        oClasses.append(len(data) + offset)
        self.numClass = len(oClasses) - 1
        self.numLinear = len(self.linear)
        return (
            sstruct.pack(Silf_classmap_format, self)
            + struct.pack(
                ((">%dL" if version >= 4.0 else ">%dH") % len(oClasses)), *oClasses
            )
            + data
        )

    def toXML(self, writer, ttFont, version=2.0):
        writer.begintag("classes")
        writer.newline()
        writer.begintag("linearClasses")
        writer.newline()
        for i, l in enumerate(self.linear):
            writer.begintag("linear", _index=i)
            writer.newline()
            wrapline(writer, l)
            writer.endtag("linear")
            writer.newline()
        writer.endtag("linearClasses")
        writer.newline()
        writer.begintag("nonLinearClasses")
        writer.newline()
        for i, l in enumerate(self.nonLinear):
            writer.begintag("nonLinear", _index=i + self.numLinear)
            writer.newline()
            for inp, ind in l.items():
                writer.simpletag("map", glyph=inp, index=ind)
                writer.newline()
            writer.endtag("nonLinear")
            writer.newline()
        writer.endtag("nonLinearClasses")
        writer.newline()
        writer.endtag("classes")
        writer.newline()

    def fromXML(self, name, attrs, content, ttFont, version=2.0):
        if name == "linearClasses":
            for element in content:
                if not isinstance(element, tuple):
                    continue
                tag, attrs, subcontent = element
                if tag == "linear":
                    l = content_string(subcontent).split()
                    self.linear.append(l)
        elif name == "nonLinearClasses":
            for element in content:
                if not isinstance(element, tuple):
                    continue
                tag, attrs, subcontent = element
                if tag == "nonLinear":
                    l = {}
                    for e in subcontent:
                        if not isinstance(e, tuple):
                            continue
                        tag, attrs, subsubcontent = e
                        if tag == "map":
                            l[attrs["glyph"]] = int(safeEval(attrs["index"]))
                    self.nonLinear.append(l)


class Pass(object):
    def __init__(self):
        self.colMap = {}
        self.rules = []
        self.rulePreContexts = []
        self.ruleSortKeys = []
        self.ruleConstraints = []
        self.passConstraints = b""
        self.actions = []
        self.stateTrans = []
        self.startStates = []

    def decompile(self, data, ttFont, version=2.0):
        _, data = sstruct.unpack2(Silf_pass_format, data, self)
        numRange, _, _, _ = struct.unpack(">4H", data[:8])
        data = data[8:]
        for i in range(numRange):
            first, last, col = struct.unpack(">3H", data[6 * i : 6 * i + 6])
            for g in range(first, last + 1):
                self.colMap[ttFont.getGlyphName(g)] = col
        data = data[6 * numRange :]
        oRuleMap = struct.unpack_from((">%dH" % (self.numSuccess + 1)), data)
        data = data[2 + 2 * self.numSuccess :]
        rules = struct.unpack_from((">%dH" % oRuleMap[-1]), data)
        self.rules = [rules[s:e] for (s, e) in zip(oRuleMap, oRuleMap[1:])]
        data = data[2 * oRuleMap[-1] :]
        self.minRulePreContext, self.maxRulePreContext = struct.unpack("BB", data[:2])
        numStartStates = self.maxRulePreContext - self.minRulePreContext + 1
        self.startStates = struct.unpack(
            (">%dH" % numStartStates), data[2 : 2 + numStartStates * 2]
        )
        data = data[2 + numStartStates * 2 :]
        self.ruleSortKeys = struct.unpack(
            (">%dH" % self.numRules), data[: 2 * self.numRules]
        )
        data = data[2 * self.numRules :]
        self.rulePreContexts = struct.unpack(
            ("%dB" % self.numRules), data[: self.numRules]
        )
        data = data[self.numRules :]
        self.collisionThreshold, pConstraint = struct.unpack(">BH", data[:3])
        oConstraints = list(
            struct.unpack(
                (">%dH" % (self.numRules + 1)), data[3 : 5 + self.numRules * 2]
            )
        )
        data = data[5 + self.numRules * 2 :]
        oActions = list(
            struct.unpack((">%dH" % (self.numRules + 1)), data[: 2 + self.numRules * 2])
        )
        data = data[2 * self.numRules + 2 :]
        for i in range(self.numTransitional):
            a = array(
                "H", data[i * self.numColumns * 2 : (i + 1) * self.numColumns * 2]
            )
            if sys.byteorder != "big":
                a.byteswap()
            self.stateTrans.append(a)
        data = data[self.numTransitional * self.numColumns * 2 + 1 :]
        self.passConstraints = data[:pConstraint]
        data = data[pConstraint:]
        for i in range(len(oConstraints) - 2, -1, -1):
            if oConstraints[i] == 0:
                oConstraints[i] = oConstraints[i + 1]
        self.ruleConstraints = [
            (data[s:e] if (e - s > 1) else b"")
            for (s, e) in zip(oConstraints, oConstraints[1:])
        ]
        data = data[oConstraints[-1] :]
        self.actions = [
            (data[s:e] if (e - s > 1) else "") for (s, e) in zip(oActions, oActions[1:])
        ]
        data = data[oActions[-1] :]
        # not using debug

    def compile(self, ttFont, base, version=2.0):
        # build it all up backwards
        oActions = reduce(
            lambda a, x: (a[0] + len(x), a[1] + [a[0]]), self.actions + [b""], (0, [])
        )[1]
        oConstraints = reduce(
            lambda a, x: (a[0] + len(x), a[1] + [a[0]]),
            self.ruleConstraints + [b""],
            (1, []),
        )[1]
        constraintCode = b"\000" + b"".join(self.ruleConstraints)
        transes = []
        for t in self.stateTrans:
            if sys.byteorder != "big":
                t.byteswap()
            transes.append(t.tobytes())
            if sys.byteorder != "big":
                t.byteswap()
        if not len(transes):
            self.startStates = [0]
        oRuleMap = reduce(
            lambda a, x: (a[0] + len(x), a[1] + [a[0]]), self.rules + [[]], (0, [])
        )[1]
        passRanges = []
        gidcolmap = dict([(ttFont.getGlyphID(x[0]), x[1]) for x in self.colMap.items()])
        for e in grUtils.entries(gidcolmap, sameval=True):
            if e[1]:
                passRanges.append((e[0], e[0] + e[1] - 1, e[2][0]))
        self.numRules = len(self.actions)
        self.fsmOffset = (
            sstruct.calcsize(Silf_pass_format)
            + 8
            + len(passRanges) * 6
            + len(oRuleMap) * 2
            + 2 * oRuleMap[-1]
            + 2
            + 2 * len(self.startStates)
            + 3 * self.numRules
            + 3
            + 4 * self.numRules
            + 4
        )
        self.pcCode = (
            self.fsmOffset + 2 * self.numTransitional * self.numColumns + 1 + base
        )
        self.rcCode = self.pcCode + len(self.passConstraints)
        self.aCode = self.rcCode + len(constraintCode)
        self.oDebug = 0
        # now generate output
        data = sstruct.pack(Silf_pass_format, self)
        data += grUtils.bininfo(len(passRanges), 6)
        data += b"".join(struct.pack(">3H", *p) for p in passRanges)
        data += struct.pack((">%dH" % len(oRuleMap)), *oRuleMap)
        flatrules = reduce(lambda a, x: a + x, self.rules, [])
        data += struct.pack((">%dH" % oRuleMap[-1]), *flatrules)
        data += struct.pack("BB", self.minRulePreContext, self.maxRulePreContext)
        data += struct.pack((">%dH" % len(self.startStates)), *self.startStates)
        data += struct.pack((">%dH" % self.numRules), *self.ruleSortKeys)
        data += struct.pack(("%dB" % self.numRules), *self.rulePreContexts)
        data += struct.pack(">BH", self.collisionThreshold, len(self.passConstraints))
        data += struct.pack((">%dH" % (self.numRules + 1)), *oConstraints)
        data += struct.pack((">%dH" % (self.numRules + 1)), *oActions)
        return (
            data
            + b"".join(transes)
            + struct.pack("B", 0)
            + self.passConstraints
            + constraintCode
            + b"".join(self.actions)
        )

    def toXML(self, writer, ttFont, version=2.0):
        writesimple("info", self, writer, *pass_attrs_info)
        writesimple("fsminfo", self, writer, *pass_attrs_fsm)
        writer.begintag("colmap")
        writer.newline()
        wrapline(
            writer,
            [
                "{}={}".format(*x)
                for x in sorted(
                    self.colMap.items(), key=lambda x: ttFont.getGlyphID(x[0])
                )
            ],
        )
        writer.endtag("colmap")
        writer.newline()
        writer.begintag("staterulemap")
        writer.newline()
        for i, r in enumerate(self.rules):
            writer.simpletag(
                "state",
                number=self.numRows - self.numSuccess + i,
                rules=" ".join(map(str, r)),
            )
            writer.newline()
        writer.endtag("staterulemap")
        writer.newline()
        writer.begintag("rules")
        writer.newline()
        for i, action in enumerate(self.actions):
            writer.begintag(
                "rule",
                index=i,
                precontext=self.rulePreContexts[i],
                sortkey=self.ruleSortKeys[i],
            )
            writer.newline()
            if len(self.ruleConstraints[i]):
                writecode("constraint", writer, self.ruleConstraints[i])
            writecode("action", writer, action)
            writer.endtag("rule")
            writer.newline()
        writer.endtag("rules")
        writer.newline()
        if len(self.passConstraints):
            writecode("passConstraint", writer, self.passConstraints)
        if len(self.stateTrans):
            writer.begintag("fsm")
            writer.newline()
            writer.begintag("starts")
            writer.write(" ".join(map(str, self.startStates)))
            writer.endtag("starts")
            writer.newline()
            for i, s in enumerate(self.stateTrans):
                writer.begintag("row", _i=i)
                # no newlines here
                writer.write(" ".join(map(str, s)))
                writer.endtag("row")
                writer.newline()
            writer.endtag("fsm")
            writer.newline()

    def fromXML(self, name, attrs, content, ttFont, version=2.0):
        if name == "info":
            getSimple(self, attrs, *pass_attrs_info)
        elif name == "fsminfo":
            getSimple(self, attrs, *pass_attrs_fsm)
        elif name == "colmap":
            e = content_string(content)
            for w in e.split():
                x = w.split("=")
                if len(x) != 2 or x[0] == "" or x[1] == "":
                    continue
                self.colMap[x[0]] = int(x[1])
        elif name == "staterulemap":
            for e in content:
                if not isinstance(e, tuple):
                    continue
                tag, a, c = e
                if tag == "state":
                    self.rules.append([int(x) for x in a["rules"].split(" ")])
        elif name == "rules":
            for element in content:
                if not isinstance(element, tuple):
                    continue
                tag, a, c = element
                if tag != "rule":
                    continue
                self.rulePreContexts.append(int(a["precontext"]))
                self.ruleSortKeys.append(int(a["sortkey"]))
                con = b""
                act = b""
                for e in c:
                    if not isinstance(e, tuple):
                        continue
                    tag, a, subc = e
                    if tag == "constraint":
                        con = readcode(subc)
                    elif tag == "action":
                        act = readcode(subc)
                self.actions.append(act)
                self.ruleConstraints.append(con)
        elif name == "passConstraint":
            self.passConstraints = readcode(content)
        elif name == "fsm":
            for element in content:
                if not isinstance(element, tuple):
                    continue
                tag, a, c = element
                if tag == "row":
                    s = array("H")
                    e = content_string(c)
                    s.extend(map(int, e.split()))
                    self.stateTrans.append(s)
                elif tag == "starts":
                    s = []
                    e = content_string(c)
                    s.extend(map(int, e.split()))
                    self.startStates = s
<|endoftext|>"""Compiles/decompiles SVG table.

https://docs.microsoft.com/en-us/typography/opentype/spec/svg

The XML format is:

.. code-block:: xml

        <SVG>
                <svgDoc endGlyphID="1" startGlyphID="1">
                        <![CDATA[ <complete SVG doc> ]]
                </svgDoc>
        ...
                <svgDoc endGlyphID="n" startGlyphID="m">
                        <![CDATA[ <complete SVG doc> ]]
                </svgDoc>
        </SVG>
"""

from fontTools.misc.textTools import bytesjoin, safeEval, strjoin, tobytes, tostr
from fontTools.misc import sstruct
from . import DefaultTable
from collections.abc import Sequence
from dataclasses import dataclass, astuple
from io import BytesIO
import struct
import logging

log = logging.getLogger(__name__)


SVG_format_0 = """
	>   # big endian
	version:                  H
	offsetToSVGDocIndex:      L
	reserved:                 L
"""

SVG_format_0Size = sstruct.calcsize(SVG_format_0)

doc_index_entry_format_0 = """
	>   # big endian
	startGlyphID:             H
	endGlyphID:               H
	svgDocOffset:             L
	svgDocLength:             L
"""

doc_index_entry_format_0Size = sstruct.calcsize(doc_index_entry_format_0)


class table_S_V_G_(DefaultTable.DefaultTable):
    """Scalable Vector Graphics table

    The ``SVG`` table contains representations for glyphs in the SVG
    image format.

    See also https://learn.microsoft.com/en-us/typography/opentype/spec/stat
    """

    def decompile(self, data, ttFont):
        self.docList = []
        # Version 0 is the standardized version of the table; and current.
        # https://www.microsoft.com/typography/otspec/svg.htm
        sstruct.unpack(SVG_format_0, data[:SVG_format_0Size], self)
        if self.version != 0:
            log.warning(
                "Unknown SVG table version '%s'. Decompiling as version 0.",
                self.version,
            )
        # read in SVG Documents Index
        # data starts with the first entry of the entry list.
        pos = subTableStart = self.offsetToSVGDocIndex
        self.numEntries = struct.unpack(">H", data[pos : pos + 2])[0]
        pos += 2
        if self.numEntries > 0:
            data2 = data[pos:]
            entries = []
            for i in range(self.numEntries):
                record_data = data2[
                    i
                    * doc_index_entry_format_0Size : (i + 1)
                    * doc_index_entry_format_0Size
                ]
                docIndexEntry = sstruct.unpack(
                    doc_index_entry_format_0, record_data, DocumentIndexEntry()
                )
                entries.append(docIndexEntry)

            for entry in entries:
                start = entry.svgDocOffset + subTableStart
                end = start + entry.svgDocLength
                doc = data[start:end]
                compressed = False
                if doc.startswith(b"\x1f\x8b"):
                    import gzip

                    bytesIO = BytesIO(doc)
                    with gzip.GzipFile(None, "r", fileobj=bytesIO) as gunzipper:
                        doc = gunzipper.read()
                    del bytesIO
                    compressed = True
                doc = tostr(doc, "utf_8")
                self.docList.append(
                    SVGDocument(doc, entry.startGlyphID, entry.endGlyphID, compressed)
                )

    def compile(self, ttFont):
        version = 0
        offsetToSVGDocIndex = (
            SVG_format_0Size  # I start the SVGDocIndex right after the header.
        )
        # get SGVDoc info.
        docList = []
        entryList = []
        numEntries = len(self.docList)
        datum = struct.pack(">H", numEntries)
        entryList.append(datum)
        curOffset = len(datum) + doc_index_entry_format_0Size * numEntries
        seenDocs = {}
        allCompressed = getattr(self, "compressed", False)
        for i, doc in enumerate(self.docList):
            if isinstance(doc, (list, tuple)):
                doc = SVGDocument(*doc)
                self.docList[i] = doc
            docBytes = tobytes(doc.data, encoding="utf_8")
            if (allCompressed or doc.compressed) and not docBytes.startswith(
                b"\x1f\x8b"
            ):
                import gzip

                bytesIO = BytesIO()
                # mtime=0 strips the useless timestamp and makes gzip output reproducible;
                # equivalent to `gzip -n`
                with gzip.GzipFile(None, "w", fileobj=bytesIO, mtime=0) as gzipper:
                    gzipper.write(docBytes)
                gzipped = bytesIO.getvalue()
                if len(gzipped) < len(docBytes):
                    docBytes = gzipped
                del gzipped, bytesIO
            docLength = len(docBytes)
            if docBytes in seenDocs:
                docOffset = seenDocs[docBytes]
            else:
                docOffset = curOffset
                curOffset += docLength
                seenDocs[docBytes] = docOffset
                docList.append(docBytes)
            entry = struct.pack(
                ">HHLL", doc.startGlyphID, doc.endGlyphID, docOffset, docLength
            )
            entryList.append(entry)
        entryList.extend(docList)
        svgDocData = bytesjoin(entryList)

        reserved = 0
        header = struct.pack(">HLL", version, offsetToSVGDocIndex, reserved)
        data = [header, svgDocData]
        data = bytesjoin(data)
        return data

    def toXML(self, writer, ttFont):
        for i, doc in enumerate(self.docList):
            if isinstance(doc, (list, tuple)):
                doc = SVGDocument(*doc)
                self.docList[i] = doc
            attrs = {"startGlyphID": doc.startGlyphID, "endGlyphID": doc.endGlyphID}
            if doc.compressed:
                attrs["compressed"] = 1
            writer.begintag("svgDoc", **attrs)
            writer.newline()
            writer.writecdata(doc.data)
            writer.newline()
            writer.endtag("svgDoc")
            writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        if name == "svgDoc":
            if not hasattr(self, "docList"):
                self.docList = []
            doc = strjoin(content)
            doc = doc.strip()
            startGID = int(attrs["startGlyphID"])
            endGID = int(attrs["endGlyphID"])
            compressed = bool(safeEval(attrs.get("compressed", "0")))
            self.docList.append(SVGDocument(doc, startGID, endGID, compressed))
        else:
            log.warning("Unknown %s %s", name, content)


class DocumentIndexEntry(object):
    def __init__(self):
        self.startGlyphID = None  # USHORT
        self.endGlyphID = None  # USHORT
        self.svgDocOffset = None  # ULONG
        self.svgDocLength = None  # ULONG

    def __repr__(self):
        return (
            "startGlyphID: %s, endGlyphID: %s, svgDocOffset: %s, svgDocLength: %s"
            % (self.startGlyphID, self.endGlyphID, self.svgDocOffset, self.svgDocLength)
        )


@dataclass
class SVGDocument(Sequence):
    data: str
    startGlyphID: int
    endGlyphID: int
    compressed: bool = False

    # Previously, the SVG table's docList attribute contained a lists of 3 items:
    # [doc, startGlyphID, endGlyphID]; later, we added a `compressed` attribute.
    # For backward compatibility with code that depends of them being sequences of
    # fixed length=3, we subclass the Sequence abstract base class and pretend only
    # the first three items are present. 'compressed' is only accessible via named
    # attribute lookup like regular dataclasses: i.e. `doc.compressed`, not `doc[3]`
    def __getitem__(self, index):
        return astuple(self)[:3][index]

    def __len__(self):
        return 3
<|endoftext|>from fontTools.config import OPTIONS
from fontTools.misc.textTools import Tag, bytesjoin
from .DefaultTable import DefaultTable
from enum import IntEnum
import sys
import array
import struct
import logging
from functools import lru_cache
from typing import Iterator, NamedTuple, Optional, Tuple

log = logging.getLogger(__name__)

have_uharfbuzz = False
try:
    import uharfbuzz as hb

    have_uharfbuzz = True
except ImportError:
    pass

USE_HARFBUZZ_REPACKER = OPTIONS[f"{__name__}:USE_HARFBUZZ_REPACKER"]


class OverflowErrorRecord(object):
    def __init__(self, overflowTuple):
        self.tableType = overflowTuple[0]
        self.LookupListIndex = overflowTuple[1]
        self.SubTableIndex = overflowTuple[2]
        self.itemName = overflowTuple[3]
        self.itemIndex = overflowTuple[4]

    def __repr__(self):
        return str(
            (
                self.tableType,
                "LookupIndex:",
                self.LookupListIndex,
                "SubTableIndex:",
                self.SubTableIndex,
                "ItemName:",
                self.itemName,
                "ItemIndex:",
                self.itemIndex,
            )
        )


class OTLOffsetOverflowError(Exception):
    def __init__(self, overflowErrorRecord):
        self.value = overflowErrorRecord

    def __str__(self):
        return repr(self.value)


class RepackerState(IntEnum):
    # Repacking control flow is implemnted using a state machine. The state machine table:
    #
    # State       | Packing Success | Packing Failed | Exception Raised |
    # ------------+-----------------+----------------+------------------+
    # PURE_FT     | Return result   | PURE_FT        | Return failure   |
    # HB_FT       | Return result   | HB_FT          | FT_FALLBACK      |
    # FT_FALLBACK | HB_FT           | FT_FALLBACK    | Return failure   |

    # Pack only with fontTools, don't allow sharing between extensions.
    PURE_FT = 1

    # Attempt to pack with harfbuzz (allowing sharing between extensions)
    # use fontTools to attempt overflow resolution.
    HB_FT = 2

    # Fallback if HB/FT packing gets stuck. Pack only with fontTools, don't allow sharing between
    # extensions.
    FT_FALLBACK = 3


class BaseTTXConverter(DefaultTable):
    """Generic base class for TTX table converters. It functions as an
    adapter between the TTX (ttLib actually) table model and the model
    we use for OpenType tables, which is necessarily subtly different.
    """

    def decompile(self, data, font):
        """Create an object from the binary data. Called automatically on access."""
        from . import otTables

        reader = OTTableReader(data, tableTag=self.tableTag)
        tableClass = getattr(otTables, self.tableTag)
        self.table = tableClass()
        self.table.decompile(reader, font)

    def compile(self, font):
        """Compiles the table into binary. Called automatically on save."""

        # General outline:
        # Create a top-level OTTableWriter for the GPOS/GSUB table.
        # 	Call the compile method for the the table
        # 		for each 'converter' record in the table converter list
        # 			call converter's write method for each item in the value.
        # 				- For simple items, the write method adds a string to the
        # 				writer's self.items list.
        # 				- For Struct/Table/Subtable items, it add first adds new writer to the
        # 				to the writer's self.items, then calls the item's compile method.
        # 				This creates a tree of writers, rooted at the GUSB/GPOS writer, with
        # 				each writer representing a table, and the writer.items list containing
        # 				the child data strings and writers.
        # 	call the getAllData method
        # 		call _doneWriting, which removes duplicates
        # 		call _gatherTables. This traverses the tables, adding unique occurences to a flat list of tables
        # 		Traverse the flat list of tables, calling getDataLength on each to update their position
        # 		Traverse the flat list of tables again, calling getData each get the data in the table, now that
        # 		pos's and offset are known.

        # 		If a lookup subtable overflows an offset, we have to start all over.
        overflowRecord = None
        # this is 3-state option: default (None) means automatically use hb.repack or
        # silently fall back if it fails; True, use it and raise error if not possible
        # or it errors out; False, don't use it, even if you can.
        use_hb_repack = font.cfg[USE_HARFBUZZ_REPACKER]
        if self.tableTag in ("GSUB", "GPOS"):
            if use_hb_repack is False:
                log.debug(
                    "hb.repack disabled, compiling '%s' with pure-python serializer",
                    self.tableTag,
                )
            elif not have_uharfbuzz:
                if use_hb_repack is True:
                    raise ImportError("No module named 'uharfbuzz'")
                else:
                    assert use_hb_repack is None
                    log.debug(
                        "uharfbuzz not found, compiling '%s' with pure-python serializer",
                        self.tableTag,
                    )

        if (
            use_hb_repack in (None, True)
            and have_uharfbuzz
            and self.tableTag in ("GSUB", "GPOS")
        ):
            state = RepackerState.HB_FT
        else:
            state = RepackerState.PURE_FT

        hb_first_error_logged = False
        lastOverflowRecord = None
        while True:
            try:
                writer = OTTableWriter(tableTag=self.tableTag)
                self.table.compile(writer, font)
                if state == RepackerState.HB_FT:
                    return self.tryPackingHarfbuzz(writer, hb_first_error_logged)
                elif state == RepackerState.PURE_FT:
                    return self.tryPackingFontTools(writer)
                elif state == RepackerState.FT_FALLBACK:
                    # Run packing with FontTools only, but don't return the result as it will
                    # not be optimally packed. Once a successful packing has been found, state is
                    # changed back to harfbuzz packing to produce the final, optimal, packing.
                    self.tryPackingFontTools(writer)
                    log.debug(
                        "Re-enabling sharing between extensions and switching back to "
                        "harfbuzz+fontTools packing."
                    )
                    state = RepackerState.HB_FT

            except OTLOffsetOverflowError as e:
                hb_first_error_logged = True
                ok = self.tryResolveOverflow(font, e, lastOverflowRecord)
                lastOverflowRecord = e.value

                if ok:
                    continue

                if state is RepackerState.HB_FT:
                    log.debug(
                        "Harfbuzz packing out of resolutions, disabling sharing between extensions and "
                        "switching to fontTools only packing."
                    )
                    state = RepackerState.FT_FALLBACK
                else:
                    raise

    def tryPackingHarfbuzz(self, writer, hb_first_error_logged):
        try:
            log.debug("serializing '%s' with hb.repack", self.tableTag)
            return writer.getAllDataUsingHarfbuzz(self.tableTag)
        except (ValueError, MemoryError, hb.RepackerError) as e:
            # Only log hb repacker errors the first time they occur in
            # the offset-overflow resolution loop, they are just noisy.
            # Maybe we can revisit this if/when uharfbuzz actually gives
            # us more info as to why hb.repack failed...
            if not hb_first_error_logged:
                error_msg = f"{type(e).__name__}"
                if str(e) != "":
                    error_msg += f": {e}"
                log.warning(
                    "hb.repack failed to serialize '%s', attempting fonttools resolutions "
                    "; the error message was: %s",
                    self.tableTag,
                    error_msg,
                )
                hb_first_error_logged = True
            return writer.getAllData(remove_duplicate=False)

    def tryPackingFontTools(self, writer):
        return writer.getAllData()

    def tryResolveOverflow(self, font, e, lastOverflowRecord):
        ok = 0
        if lastOverflowRecord == e.value:
            # Oh well...
            return ok

        overflowRecord = e.value
        log.info("Attempting to fix OTLOffsetOverflowError %s", e)

        if overflowRecord.itemName is None:
            from .otTables import fixLookupOverFlows

            ok = fixLookupOverFlows(font, overflowRecord)
        else:
            from .otTables import fixSubTableOverFlows

            ok = fixSubTableOverFlows(font, overflowRecord)

        if ok:
            return ok

        # Try upgrading lookup to Extension and hope
        # that cross-lookup sharing not happening would
        # fix overflow...
        from .otTables import fixLookupOverFlows

        return fixLookupOverFlows(font, overflowRecord)

    def toXML(self, writer, font):
        self.table.toXML2(writer, font)

    def fromXML(self, name, attrs, content, font):
        from . import otTables

        if not hasattr(self, "table"):
            tableClass = getattr(otTables, self.tableTag)
            self.table = tableClass()
        self.table.fromXML(name, attrs, content, font)
        self.table.populateDefaults()

    def ensureDecompiled(self, recurse=True):
        self.table.ensureDecompiled(recurse=recurse)


# https://github.com/fonttools/fonttools/pull/2285#issuecomment-834652928
assert len(struct.pack("i", 0)) == 4
assert array.array("i").itemsize == 4, "Oops, file a bug against fonttools."


class OTTableReader(object):
    """Helper class to retrieve data from an OpenType table."""

    __slots__ = ("data", "offset", "pos", "localState", "tableTag")

    def __init__(self, data, localState=None, offset=0, tableTag=None):
        self.data = data
        self.offset = offset
        self.pos = offset
        self.localState = localState
        self.tableTag = tableTag

    def advance(self, count):
        self.pos += count

    def seek(self, pos):
        self.pos = pos

    def copy(self):
        other = self.__class__(self.data, self.localState, self.offset, self.tableTag)
        other.pos = self.pos
        return other

    def getSubReader(self, offset):
        offset = self.offset + offset
        return self.__class__(self.data, self.localState, offset, self.tableTag)

    def readValue(self, typecode, staticSize):
        pos = self.pos
        newpos = pos + staticSize
        (value,) = struct.unpack(f">{typecode}", self.data[pos:newpos])
        self.pos = newpos
        return value

    def readArray(self, typecode, staticSize, count):
        pos = self.pos
        newpos = pos + count * staticSize
        value = array.array(typecode, self.data[pos:newpos])
        if sys.byteorder != "big":
            value.byteswap()
        self.pos = newpos
        return value.tolist()

    def readInt8(self):
        return self.readValue("b", staticSize=1)

    def readInt8Array(self, count):
        return self.readArray("b", staticSize=1, count=count)

    def readShort(self):
        return self.readValue("h", staticSize=2)

    def readShortArray(self, count):
        return self.readArray("h", staticSize=2, count=count)

    def readLong(self):
        return self.readValue("i", staticSize=4)

    def readLongArray(self, count):
        return self.readArray("i", staticSize=4, count=count)

    def readUInt8(self):
        return self.readValue("B", staticSize=1)

    def readUInt8Array(self, count):
        return self.readArray("B", staticSize=1, count=count)

    def readUShort(self):
        return self.readValue("H", staticSize=2)

    def readUShortArray(self, count):
        return self.readArray("H", staticSize=2, count=count)

    def readULong(self):
        return self.readValue("I", staticSize=4)

    def readULongArray(self, count):
        return self.readArray("I", staticSize=4, count=count)

    def readUInt24(self):
        pos = self.pos
        newpos = pos + 3
        (value,) = struct.unpack(">l", b"\0" + self.data[pos:newpos])
        self.pos = newpos
        return value

    def readInt24(self):
        pos = self.pos
        newpos = pos + 3
        b = self.data[pos:newpos]
        pad = b"\xff" if b[0] & 0x80 else b"\0"
        (value,) = struct.unpack(">l", pad + b)
        self.pos = newpos
        return value

    def readUInt24Array(self, count):
        return [self.readUInt24() for _ in range(count)]

    def readTag(self):
        pos = self.pos
        newpos = pos + 4
        value = Tag(self.data[pos:newpos])
        assert len(value) == 4, value
        self.pos = newpos
        return value

    def readData(self, count):
        pos = self.pos
        newpos = pos + count
        value = self.data[pos:newpos]
        self.pos = newpos
        return value

    def __setitem__(self, name, value):
        state = self.localState.copy() if self.localState else dict()
        state[name] = value
        self.localState = state

    def __getitem__(self, name):
        return self.localState and self.localState[name]

    def __contains__(self, name):
        return self.localState and name in self.localState


class OffsetToWriter(object):
    def __init__(self, subWriter, offsetSize):
        self.subWriter = subWriter
        self.offsetSize = offsetSize

    def __eq__(self, other):
        if type(self) != type(other):
            return NotImplemented
        return self.subWriter == other.subWriter and self.offsetSize == other.offsetSize

    def __hash__(self):
        # only works after self._doneWriting() has been called
        return hash((self.subWriter, self.offsetSize))


class OTTableWriter(object):
    """Helper class to gather and assemble data for OpenType tables."""

    def __init__(self, localState=None, tableTag=None):
        self.items = []
        self.pos = None
        self.localState = localState
        self.tableTag = tableTag
        self.parent = None
        self.name = "<none>"

    def __setitem__(self, name, value):
        state = self.localState.copy() if self.localState else dict()
        state[name] = value
        self.localState = state

    def __getitem__(self, name):
        return self.localState[name]

    def __delitem__(self, name):
        del self.localState[name]

    # assembler interface

    def getDataLength(self):
        """Return the length of this table in bytes, without subtables."""
        l = 0
        for item in self.items:
            if hasattr(item, "getCountData"):
                l += item.size
            elif hasattr(item, "subWriter"):
                l += item.offsetSize
            else:
                l = l + len(item)
        return l

    def getData(self):
        """Assemble the data for this writer/table, without subtables."""
        items = list(self.items)  # make a shallow copy
        pos = self.pos
        numItems = len(items)
        for i in range(numItems):
            item = items[i]

            if hasattr(item, "subWriter"):
                if item.offsetSize == 4:
                    items[i] = packULong(item.subWriter.pos - pos)
                elif item.offsetSize == 2:
                    try:
                        items[i] = packUShort(item.subWriter.pos - pos)
                    except struct.error:
                        # provide data to fix overflow problem.
                        overflowErrorRecord = self.getOverflowErrorRecord(
                            item.subWriter
                        )

                        raise OTLOffsetOverflowError(overflowErrorRecord)
                elif item.offsetSize == 3:
                    items[i] = packUInt24(item.subWriter.pos - pos)
                else:
                    raise ValueError(item.offsetSize)

        return bytesjoin(items)

    def getDataForHarfbuzz(self):
        """Assemble the data for this writer/table with all offset field set to 0"""
        items = list(self.items)
        packFuncs = {2: packUShort, 3: packUInt24, 4: packULong}
        for i, item in enumerate(items):
            if hasattr(item, "subWriter"):
                # Offset value is not needed in harfbuzz repacker, so setting offset to 0 to avoid overflow here
                if item.offsetSize in packFuncs:
                    items[i] = packFuncs[item.offsetSize](0)
                else:
                    raise ValueError(item.offsetSize)

        return bytesjoin(items)

    def __hash__(self):
        # only works after self._doneWriting() has been called
        return hash(self.items)

    def __ne__(self, other):
        result = self.__eq__(other)
        return result if result is NotImplemented else not result

    def __eq__(self, other):
        if type(self) != type(other):
            return NotImplemented
        return self.items == other.items

    def _doneWriting(self, internedTables, shareExtension=False):
        # Convert CountData references to data string items
        # collapse duplicate table references to a unique entry
        # "tables" are OTTableWriter objects.

        # For Extension Lookup types, we can
        # eliminate duplicates only within the tree under the Extension Lookup,
        # as offsets may exceed 64K even between Extension LookupTable subtables.
        isExtension = hasattr(self, "Extension")

        # Certain versions of Uniscribe reject the font if the GSUB/GPOS top-level
        # arrays (ScriptList, FeatureList, LookupList) point to the same, possibly
        # empty, array.  So, we don't share those.
        # See: https://github.com/fonttools/fonttools/issues/518
        dontShare = hasattr(self, "DontShare")

        if isExtension and not shareExtension:
            internedTables = {}

        items = self.items
        for i, item in enumerate(items):
            if hasattr(item, "getCountData"):
                items[i] = item.getCountData()
            elif hasattr(item, "subWriter"):
                item.subWriter._doneWriting(
                    internedTables, shareExtension=shareExtension
                )
                # At this point, all subwriters are hashable based on their items.
                # (See hash and comparison magic methods above.) So the ``setdefault``
                # call here will return the first writer object we've seen with
                # equal content, or store it in the dictionary if it's not been
                # seen yet. We therefore replace the subwriter object with an equivalent
                # object, which deduplicates the tree.
                if not dontShare:
                    items[i].subWriter = internedTables.setdefault(
                        item.subWriter, item.subWriter
                    )
        self.items = tuple(items)

    def _gatherTables(self, tables, extTables, done):
        # Convert table references in self.items tree to a flat
        # list of tables in depth-first traversal order.
        # "tables" are OTTableWriter objects.
        # We do the traversal in reverse order at each level, in order to
        # resolve duplicate references to be the last reference in the list of tables.
        # For extension lookups, duplicate references can be merged only within the
        # writer tree under the  extension lookup.

        done[id(self)] = True

        numItems = len(self.items)
        iRange = list(range(numItems))
        iRange.reverse()

        isExtension = hasattr(self, "Extension")

        selfTables = tables

        if isExtension:
            assert (
                extTables is not None
            ), "Program or XML editing error. Extension subtables cannot contain extensions subtables"
            tables, extTables, done = extTables, None, {}

        # add Coverage table if it is sorted last.
        sortCoverageLast = False
        if hasattr(self, "sortCoverageLast"):
            # Find coverage table
            for i in range(numItems):
                item = self.items[i]
                if (
                    hasattr(item, "subWriter")
                    and getattr(item.subWriter, "name", None) == "Coverage"
                ):
                    sortCoverageLast = True
                    break
            if id(item.subWriter) not in done:
                item.subWriter._gatherTables(tables, extTables, done)
            else:
                # We're a new parent of item
                pass

        for i in iRange:
            item = self.items[i]
            if not hasattr(item, "subWriter"):
                continue

            if (
                sortCoverageLast
                and (i == 1)
                and getattr(item.subWriter, "name", None) == "Coverage"
            ):
                # we've already 'gathered' it above
                continue

            if id(item.subWriter) not in done:
                item.subWriter._gatherTables(tables, extTables, done)
            else:
                # Item is already written out by other parent
                pass

        selfTables.append(self)

    def _gatherGraphForHarfbuzz(self, tables, obj_list, done, objidx, virtual_edges):
        real_links = []
        virtual_links = []
        item_idx = objidx

        # Merge virtual_links from parent
        for idx in virtual_edges:
            virtual_links.append((0, 0, idx))

        sortCoverageLast = False
        coverage_idx = 0
        if hasattr(self, "sortCoverageLast"):
            # Find coverage table
            for i, item in enumerate(self.items):
                if getattr(item, "name", None) == "Coverage":
                    sortCoverageLast = True
                    if id(item) not in done:
                        coverage_idx = item_idx = item._gatherGraphForHarfbuzz(
                            tables, obj_list, done, item_idx, virtual_edges
                        )
                    else:
                        coverage_idx = done[id(item)]
                    virtual_edges.append(coverage_idx)
                    break

        child_idx = 0
        offset_pos = 0
        for i, item in enumerate(self.items):
            if hasattr(item, "subWriter"):
                pos = offset_pos
            elif hasattr(item, "getCountData"):
                offset_pos += item.size
                continue
            else:
                offset_pos = offset_pos + len(item)
                continue

            if id(item.subWriter) not in done:
                child_idx = item_idx = item.subWriter._gatherGraphForHarfbuzz(
                    tables, obj_list, done, item_idx, virtual_edges
                )
            else:
                child_idx = done[id(item.subWriter)]

            real_edge = (pos, item.offsetSize, child_idx)
            real_links.append(real_edge)
            offset_pos += item.offsetSize

        tables.append(self)
        obj_list.append((real_links, virtual_links))
        item_idx += 1
        done[id(self)] = item_idx
        if sortCoverageLast:
            virtual_edges.pop()

        return item_idx

    def getAllDataUsingHarfbuzz(self, tableTag):
        """The Whole table is represented as a Graph.
        Assemble graph data and call Harfbuzz repacker to pack the table.
        Harfbuzz repacker is faster and retain as much sub-table sharing as possible, see also:
        https://github.com/harfbuzz/harfbuzz/blob/main/docs/repacker.md
        The input format for hb.repack() method is explained here:
        https://github.com/harfbuzz/uharfbuzz/blob/main/src/uharfbuzz/_harfbuzz.pyx#L1149
        """
        internedTables = {}
        self._doneWriting(internedTables, shareExtension=True)
        tables = []
        obj_list = []
        done = {}
        objidx = 0
        virtual_edges = []
        self._gatherGraphForHarfbuzz(tables, obj_list, done, objidx, virtual_edges)
        # Gather all data in two passes: the absolute positions of all
        # subtable are needed before the actual data can be assembled.
        pos = 0
        for table in tables:
            table.pos = pos
            pos = pos + table.getDataLength()

        data = []
        for table in tables:
            tableData = table.getDataForHarfbuzz()
            data.append(tableData)

        return hb.serialize_with_tag(str(tableTag), data, obj_list)

    def getAllData(self, remove_duplicate=True):
        """Assemble all data, including all subtables."""
        if remove_duplicate:
            internedTables = {}
            self._doneWriting(internedTables)
        tables = []
        extTables = []
        done = {}
        self._gatherTables(tables, extTables, done)
        tables.reverse()
        extTables.reverse()
        # Gather all data in two passes: the absolute positions of all
        # subtable are needed before the actual data can be assembled.
        pos = 0
        for table in tables:
            table.pos = pos
            pos = pos + table.getDataLength()

        for table in extTables:
            table.pos = pos
            pos = pos + table.getDataLength()

        data = []
        for table in tables:
            tableData = table.getData()
            data.append(tableData)

        for table in extTables:
            tableData = table.getData()
            data.append(tableData)

        return bytesjoin(data)

    # interface for gathering data, as used by table.compile()

    def getSubWriter(self):
        subwriter = self.__class__(self.localState, self.tableTag)
        subwriter.parent = (
            self  # because some subtables have idential values, we discard
        )
        # the duplicates under the getAllData method. Hence some
        # subtable writers can have more than one parent writer.
        # But we just care about first one right now.
        return subwriter

    def writeValue(self, typecode, value):
        self.items.append(struct.pack(f">{typecode}", value))

    def writeArray(self, typecode, values):
        a = array.array(typecode, values)
        if sys.byteorder != "big":
            a.byteswap()
        self.items.append(a.tobytes())

    def writeInt8(self, value):
        assert -128 <= value < 128, value
        self.items.append(struct.pack(">b", value))

    def writeInt8Array(self, values):
        self.writeArray("b", values)

    def writeShort(self, value):
        assert -32768 <= value < 32768, value
        self.items.append(struct.pack(">h", value))

    def writeShortArray(self, values):
        self.writeArray("h", values)

    def writeLong(self, value):
        self.items.append(struct.pack(">i", value))

    def writeLongArray(self, values):
        self.writeArray("i", values)

    def writeUInt8(self, value):
        assert 0 <= value < 256, value
        self.items.append(struct.pack(">B", value))

    def writeUInt8Array(self, values):
        self.writeArray("B", values)

    def writeUShort(self, value):
        assert 0 <= value < 0x10000, value
        self.items.append(struct.pack(">H", value))

    def writeUShortArray(self, values):
        self.writeArray("H", values)

    def writeULong(self, value):
        self.items.append(struct.pack(">I", value))

    def writeULongArray(self, values):
        self.writeArray("I", values)

    def writeUInt24(self, value):
        assert 0 <= value < 0x1000000, value
        b = struct.pack(">L", value)
        self.items.append(b[1:])

    def writeUInt24Array(self, values):
        for value in values:
            self.writeUInt24(value)

    def writeTag(self, tag):
        tag = Tag(tag).tobytes()
        assert len(tag) == 4, tag
        self.items.append(tag)

    def writeSubTable(self, subWriter, offsetSize):
        self.items.append(OffsetToWriter(subWriter, offsetSize))

    def writeCountReference(self, table, name, size=2, value=None):
        ref = CountReference(table, name, size=size, value=value)
        self.items.append(ref)
        return ref

    def writeStruct(self, format, values):
        data = struct.pack(*(format,) + values)
        self.items.append(data)

    def writeData(self, data):
        self.items.append(data)

    def getOverflowErrorRecord(self, item):
        LookupListIndex = SubTableIndex = itemName = itemIndex = None
        if self.name == "LookupList":
            LookupListIndex = item.repeatIndex
        elif self.name == "Lookup":
            LookupListIndex = self.repeatIndex
            SubTableIndex = item.repeatIndex
        else:
            itemName = getattr(item, "name", "<none>")
            if hasattr(item, "repeatIndex"):
                itemIndex = item.repeatIndex
            if self.name == "SubTable":
                LookupListIndex = self.parent.repeatIndex
                SubTableIndex = self.repeatIndex
            elif self.name == "ExtSubTable":
                LookupListIndex = self.parent.parent.repeatIndex
                SubTableIndex = self.parent.repeatIndex
            else:  # who knows how far below the SubTable level we are! Climb back up to the nearest subtable.
                itemName = ".".join([self.name, itemName])
                p1 = self.parent
                while p1 and p1.name not in ["ExtSubTable", "SubTable"]:
                    itemName = ".".join([p1.name, itemName])
                    p1 = p1.parent
                if p1:
                    if p1.name == "ExtSubTable":
                        LookupListIndex = p1.parent.parent.repeatIndex
                        SubTableIndex = p1.parent.repeatIndex
                    else:
                        LookupListIndex = p1.parent.repeatIndex
                        SubTableIndex = p1.repeatIndex

        return OverflowErrorRecord(
            (self.tableTag, LookupListIndex, SubTableIndex, itemName, itemIndex)
        )


class CountReference(object):
    """A reference to a Count value, not a count of references."""

    def __init__(self, table, name, size=None, value=None):
        self.table = table
        self.name = name
        self.size = size
        if value is not None:
            self.setValue(value)

    def setValue(self, value):
        table = self.table
        name = self.name
        if table[name] is None:
            table[name] = value
        else:
            assert table[name] == value, (name, table[name], value)

    def getValue(self):
        return self.table[self.name]

    def getCountData(self):
        v = self.table[self.name]
        if v is None:
            v = 0
        return {1: packUInt8, 2: packUShort, 4: packULong}[self.size](v)


def packUInt8(value):
    return struct.pack(">B", value)


def packUShort(value):
    return struct.pack(">H", value)


def packULong(value):
    assert 0 <= value < 0x100000000, value
    return struct.pack(">I", value)


def packUInt24(value):
    assert 0 <= value < 0x1000000, value
    return struct.pack(">I", value)[1:]


class BaseTable(object):
    """Generic base class for all OpenType (sub)tables."""

    def __getattr__(self, attr):
        reader = self.__dict__.get("reader")
        if reader:
            del self.reader
            font = self.font
            del self.font
            self.decompile(reader, font)
            return getattr(self, attr)

        raise AttributeError(attr)

    def ensureDecompiled(self, recurse=False):
        reader = self.__dict__.get("reader")
        if reader:
            del self.reader
            font = self.font
            del self.font
            self.decompile(reader, font)
        if recurse:
            for subtable in self.iterSubTables():
                subtable.value.ensureDecompiled(recurse)

    def __getstate__(self):
        # before copying/pickling 'lazy' objects, make a shallow copy of OTTableReader
        # https://github.com/fonttools/fonttools/issues/2965
        if "reader" in self.__dict__:
            state = self.__dict__.copy()
            state["reader"] = self.__dict__["reader"].copy()
            return state
        return self.__dict__

    @classmethod
    def getRecordSize(cls, reader):
        totalSize = 0
        for conv in cls.converters:
            size = conv.getRecordSize(reader)
            if size is NotImplemented:
                return NotImplemented
            countValue = 1
            if conv.repeat:
                if conv.repeat in reader:
                    countValue = reader[conv.repeat] + conv.aux
                else:
                    return NotImplemented
            totalSize += size * countValue
        return totalSize

    def getConverters(self):
        return self.converters

    def getConverterByName(self, name):
        return self.convertersByName[name]

    def populateDefaults(self, propagator=None):
        for conv in self.getConverters():
            if conv.repeat:
                if not hasattr(self, conv.name):
                    setattr(self, conv.name, [])
                countValue = len(getattr(self, conv.name)) - conv.aux
                try:
                    count_conv = self.getConverterByName(conv.repeat)
                    setattr(self, conv.repeat, countValue)
                except KeyError:
                    # conv.repeat is a propagated count
                    if propagator and conv.repeat in propagator:
                        propagator[conv.repeat].setValue(countValue)
            else:
                if conv.aux and not eval(conv.aux, None, self.__dict__):
                    continue
                if hasattr(self, conv.name):
                    continue  # Warn if it should NOT be present?!
                if hasattr(conv, "writeNullOffset"):
                    setattr(self, conv.name, None)  # Warn?
                # elif not conv.isCount:
                # 	# Warn?
                # 	pass
                if hasattr(conv, "DEFAULT"):
                    # OptionalValue converters (e.g. VarIndex)
                    setattr(self, conv.name, conv.DEFAULT)

    def decompile(self, reader, font):
        self.readFormat(reader)
        table = {}
        self.__rawTable = table  # for debugging
        for conv in self.getConverters():
            if conv.name == "SubTable":
                conv = conv.getConverter(reader.tableTag, table["LookupType"])
            if conv.name == "ExtSubTable":
                conv = conv.getConverter(reader.tableTag, table["ExtensionLookupType"])
            if conv.name == "FeatureParams":
                conv = conv.getConverter(reader["FeatureTag"])
            if conv.name == "SubStruct":
                conv = conv.getConverter(reader.tableTag, table["MorphType"])
            try:
                if conv.repeat:
                    if isinstance(conv.repeat, int):
                        countValue = conv.repeat
                    elif conv.repeat in table:
                        countValue = table[conv.repeat]
                    else:
                        # conv.repeat is a propagated count
                        countValue = reader[conv.repeat]
                    countValue += conv.aux
                    table[conv.name] = conv.readArray(reader, font, table, countValue)
                else:
                    if conv.aux and not eval(conv.aux, None, table):
                        continue
                    table[conv.name] = conv.read(reader, font, table)
                    if conv.isPropagated:
                        reader[conv.name] = table[conv.name]
            except Exception as e:
                name = conv.name
                e.args = e.args + (name,)
                raise

        if hasattr(self, "postRead"):
            self.postRead(table, font)
        else:
            self.__dict__.update(table)

        del self.__rawTable  # succeeded, get rid of debugging info

    def compile(self, writer, font):
        self.ensureDecompiled()
        # TODO Following hack to be removed by rewriting how FormatSwitching tables
        # are handled.
        # https://github.com/fonttools/fonttools/pull/2238#issuecomment-805192631
        if hasattr(self, "preWrite"):
            deleteFormat = not hasattr(self, "Format")
            table = self.preWrite(font)
            deleteFormat = deleteFormat and hasattr(self, "Format")
        else:
            deleteFormat = False
            table = self.__dict__.copy()

        # some count references may have been initialized in a custom preWrite; we set
        # these in the writer's state beforehand (instead of sequentially) so they will
        # be propagated to all nested subtables even if the count appears in the current
        # table only *after* the offset to the subtable that it is counting.
        for conv in self.getConverters():
            if conv.isCount and conv.isPropagated:
                value = table.get(conv.name)
                if isinstance(value, CountReference):
                    writer[conv.name] = value

        if hasattr(self, "sortCoverageLast"):
            writer.sortCoverageLast = 1

        if hasattr(self, "DontShare"):
            writer.DontShare = True

        if hasattr(self.__class__, "LookupType"):
            writer["LookupType"].setValue(self.__class__.LookupType)

        self.writeFormat(writer)
        for conv in self.getConverters():
            value = table.get(
                conv.name
            )  # TODO Handle defaults instead of defaulting to None!
            if conv.repeat:
                if value is None:
                    value = []
                countValue = len(value) - conv.aux
                if isinstance(conv.repeat, int):
                    assert len(value) == conv.repeat, "expected %d values, got %d" % (
                        conv.repeat,
                        len(value),
                    )
                elif conv.repeat in table:
                    CountReference(table, conv.repeat, value=countValue)
                else:
                    # conv.repeat is a propagated count
                    writer[conv.repeat].setValue(countValue)
                try:
                    conv.writeArray(writer, font, table, value)
                except Exception as e:
                    e.args = e.args + (conv.name + "[]",)
                    raise
            elif conv.isCount:
                # Special-case Count values.
                # Assumption: a Count field will *always* precede
                # the actual array(s).
                # We need a default value, as it may be set later by a nested
                # table. We will later store it here.
                # We add a reference: by the time the data is assembled
                # the Count value will be filled in.
                # We ignore the current count value since it will be recomputed,
                # unless it's a CountReference that was already initialized in a custom preWrite.
                if isinstance(value, CountReference):
                    ref = value
                    ref.size = conv.staticSize
                    writer.writeData(ref)
                    table[conv.name] = ref.getValue()
                else:
                    ref = writer.writeCountReference(table, conv.name, conv.staticSize)
                    table[conv.name] = None
                if conv.isPropagated:
                    writer[conv.name] = ref
            elif conv.isLookupType:
                # We make sure that subtables have the same lookup type,
                # and that the type is the same as the one set on the
                # Lookup object, if any is set.
                if conv.name not in table:
                    table[conv.name] = None
                ref = writer.writeCountReference(
                    table, conv.name, conv.staticSize, table[conv.name]
                )
                writer["LookupType"] = ref
            else:
                if conv.aux and not eval(conv.aux, None, table):
                    continue
                try:
                    conv.write(writer, font, table, value)
                except Exception as e:
                    name = value.__class__.__name__ if value is not None else conv.name
                    e.args = e.args + (name,)
                    raise
                if conv.isPropagated:
                    writer[conv.name] = value

        if deleteFormat:
            del self.Format

    def readFormat(self, reader):
        pass

    def writeFormat(self, writer):
        pass

    def toXML(self, xmlWriter, font, attrs=None, name=None):
        tableName = name if name else self.__class__.__name__
        if attrs is None:
            attrs = []
        if hasattr(self, "Format"):
            attrs = attrs + [("Format", self.Format)]
        xmlWriter.begintag(tableName, attrs)
        xmlWriter.newline()
        self.toXML2(xmlWriter, font)
        xmlWriter.endtag(tableName)
        xmlWriter.newline()

    def toXML2(self, xmlWriter, font):
        # Simpler variant of toXML, *only* for the top level tables (like GPOS, GSUB).
        # This is because in TTX our parent writes our main tag, and in otBase.py we
        # do it ourselves. I think I'm getting schizophrenic...
        for conv in self.getConverters():
            if conv.repeat:
                value = getattr(self, conv.name, [])
                for i, item in enumerate(value):
                    conv.xmlWrite(xmlWriter, font, item, conv.name, [("index", i)])
            else:
                if conv.aux and not eval(conv.aux, None, vars(self)):
                    continue
                value = getattr(
                    self, conv.name, None
                )  # TODO Handle defaults instead of defaulting to None!
                conv.xmlWrite(xmlWriter, font, value, conv.name, [])

    def fromXML(self, name, attrs, content, font):
        try:
            conv = self.getConverterByName(name)
        except KeyError:
            raise  # XXX on KeyError, raise nice error
        value = conv.xmlRead(attrs, content, font)
        # Some manually-written tables have a conv.repeat of ""
        # to represent lists. Hence comparing to None here to
        # allow those lists to be read correctly from XML.
        if conv.repeat is not None:
            seq = getattr(self, conv.name, None)
            if seq is None:
                seq = []
                setattr(self, conv.name, seq)
            seq.append(value)
        else:
            setattr(self, conv.name, value)

    def __ne__(self, other):
        result = self.__eq__(other)
        return result if result is NotImplemented else not result

    def __eq__(self, other):
        if type(self) != type(other):
            return NotImplemented

        self.ensureDecompiled()
        other.ensureDecompiled()

        return self.__dict__ == other.__dict__

    class SubTableEntry(NamedTuple):
        """See BaseTable.iterSubTables()"""

        name: str
        value: "BaseTable"
        index: Optional[int] = None  # index into given array, None for single values

    def iterSubTables(self) -> Iterator[SubTableEntry]:
        """Yield (name, value, index) namedtuples for all subtables of current table.

        A sub-table is an instance of BaseTable (or subclass thereof) that is a child
        of self, the current parent table.
        The tuples also contain the attribute name (str) of the of parent table to get
        a subtable, and optionally, for lists of subtables (i.e. attributes associated
        with a converter that has a 'repeat'), an index into the list containing the
        given subtable value.
        This method can be useful to traverse trees of otTables.
        """
        for conv in self.getConverters():
            name = conv.name
            value = getattr(self, name, None)
            if value is None:
                continue
            if isinstance(value, BaseTable):
                yield self.SubTableEntry(name, value)
            elif isinstance(value, list):
                yield from (
                    self.SubTableEntry(name, v, index=i)
                    for i, v in enumerate(value)
                    if isinstance(v, BaseTable)
                )

    # instance (not @class)method for consistency with FormatSwitchingBaseTable
    def getVariableAttrs(self):
        return getVariableAttrs(self.__class__)


class FormatSwitchingBaseTable(BaseTable):
    """Minor specialization of BaseTable, for tables that have multiple
    formats, eg. CoverageFormat1 vs. CoverageFormat2."""

    @classmethod
    def getRecordSize(cls, reader):
        return NotImplemented

    def getConverters(self):
        try:
            fmt = self.Format
        except AttributeError:
            # some FormatSwitchingBaseTables (e.g. Coverage) no longer have 'Format'
            # attribute after fully decompiled, only gain one in preWrite before being
            # recompiled. In the decompiled state, these hand-coded classes defined in
            # otTables.py lose their format-specific nature and gain more high-level
            # attributes that are not tied to converters.
            return []
        return self.converters.get(self.Format, [])

    def getConverterByName(self, name):
        return self.convertersByName[self.Format][name]

    def readFormat(self, reader):
        self.Format = reader.readUShort()

    def writeFormat(self, writer):
        writer.writeUShort(self.Format)

    def toXML(self, xmlWriter, font, attrs=None, name=None):
        BaseTable.toXML(self, xmlWriter, font, attrs, name)

    def getVariableAttrs(self):
        return getVariableAttrs(self.__class__, self.Format)


class UInt8FormatSwitchingBaseTable(FormatSwitchingBaseTable):
    def readFormat(self, reader):
        self.Format = reader.readUInt8()

    def writeFormat(self, writer):
        writer.writeUInt8(self.Format)


formatSwitchingBaseTables = {
    "uint16": FormatSwitchingBaseTable,
    "uint8": UInt8FormatSwitchingBaseTable,
}


def getFormatSwitchingBaseTableClass(formatType):
    try:
        return formatSwitchingBaseTables[formatType]
    except KeyError:
        raise TypeError(f"Unsupported format type: {formatType!r}")


# memoize since these are parsed from otData.py, thus stay constant
@lru_cache()
def getVariableAttrs(cls: BaseTable, fmt: Optional[int] = None) -> Tuple[str]:
    """Return sequence of variable table field names (can be empty).

    Attributes are deemed "variable" when their otData.py's description contain
    'VarIndexBase + {offset}', e.g. COLRv1 PaintVar* tables.
    """
    if not issubclass(cls, BaseTable):
        raise TypeError(cls)
    if issubclass(cls, FormatSwitchingBaseTable):
        if fmt is None:
            raise TypeError(f"'fmt' is required for format-switching {cls.__name__}")
        converters = cls.convertersByName[fmt]
    else:
        converters = cls.convertersByName
    # assume if no 'VarIndexBase' field is present, table has no variable fields
    if "VarIndexBase" not in converters:
        return ()
    varAttrs = {}
    for name, conv in converters.items():
        offset = conv.getVarIndexOffset()
        if offset is not None:
            varAttrs[name] = offset
    return tuple(sorted(varAttrs, key=varAttrs.__getitem__))


#
# Support for ValueRecords
#
# This data type is so different from all other OpenType data types that
# it requires quite a bit of code for itself. It even has special support
# in OTTableReader and OTTableWriter...
#

valueRecordFormat = [
    # 	Mask	 Name		isDevice signed
    (0x0001, "XPlacement", 0, 1),
    (0x0002, "YPlacement", 0, 1),
    (0x0004, "XAdvance", 0, 1),
    (0x0008, "YAdvance", 0, 1),
    (0x0010, "XPlaDevice", 1, 0),
    (0x0020, "YPlaDevice", 1, 0),
    (0x0040, "XAdvDevice", 1, 0),
    (0x0080, "YAdvDevice", 1, 0),
    # 	reserved:
    (0x0100, "Reserved1", 0, 0),
    (0x0200, "Reserved2", 0, 0),
    (0x0400, "Reserved3", 0, 0),
    (0x0800, "Reserved4", 0, 0),
    (0x1000, "Reserved5", 0, 0),
    (0x2000, "Reserved6", 0, 0),
    (0x4000, "Reserved7", 0, 0),
    (0x8000, "Reserved8", 0, 0),
]


def _buildDict():
    d = {}
    for mask, name, isDevice, signed in valueRecordFormat:
        d[name] = mask, isDevice, signed
    return d


valueRecordFormatDict = _buildDict()


class ValueRecordFactory(object):
    """Given a format code, this object convert ValueRecords."""

    def __init__(self, valueFormat):
        format = []
        for mask, name, isDevice, signed in valueRecordFormat:
            if valueFormat & mask:
                format.append((name, isDevice, signed))
        self.format = format

    def __len__(self):
        return len(self.format)

    def readValueRecord(self, reader, font):
        format = self.format
        if not format:
            return None
        valueRecord = ValueRecord()
        for name, isDevice, signed in format:
            if signed:
                value = reader.readShort()
            else:
                value = reader.readUShort()
            if isDevice:
                if value:
                    from . import otTables

                    subReader = reader.getSubReader(value)
                    value = getattr(otTables, name)()
                    value.decompile(subReader, font)
                else:
                    value = None
            setattr(valueRecord, name, value)
        return valueRecord

    def writeValueRecord(self, writer, font, valueRecord):
        for name, isDevice, signed in self.format:
            value = getattr(valueRecord, name, 0)
            if isDevice:
                if value:
                    subWriter = writer.getSubWriter()
                    writer.writeSubTable(subWriter, offsetSize=2)
                    value.compile(subWriter, font)
                else:
                    writer.writeUShort(0)
            elif signed:
                writer.writeShort(value)
            else:
                writer.writeUShort(value)


class ValueRecord(object):
    # see ValueRecordFactory

    def __init__(self, valueFormat=None, src=None):
        if valueFormat is not None:
            for mask, name, isDevice, signed in valueRecordFormat:
                if valueFormat & mask:
                    setattr(self, name, None if isDevice else 0)
            if src is not None:
                for key, val in src.__dict__.items():
                    if not hasattr(self, key):
                        continue
                    setattr(self, key, val)
        elif src is not None:
            self.__dict__ = src.__dict__.copy()

    def getFormat(self):
        format = 0
        for name in self.__dict__.keys():
            format = format | valueRecordFormatDict[name][0]
        return format

    def getEffectiveFormat(self):
        format = 0
        for name, value in self.__dict__.items():
            if value:
                format = format | valueRecordFormatDict[name][0]
        return format

    def toXML(self, xmlWriter, font, valueName, attrs=None):
        if attrs is None:
            simpleItems = []
        else:
            simpleItems = list(attrs)
        for mask, name, isDevice, format in valueRecordFormat[:4]:  # "simple" values
            if hasattr(self, name):
                simpleItems.append((name, getattr(self, name)))
        deviceItems = []
        for mask, name, isDevice, format in valueRecordFormat[4:8]:  # device records
            if hasattr(self, name):
                device = getattr(self, name)
                if device is not None:
                    deviceItems.append((name, device))
        if deviceItems:
            xmlWriter.begintag(valueName, simpleItems)
            xmlWriter.newline()
            for name, deviceRecord in deviceItems:
                if deviceRecord is not None:
                    deviceRecord.toXML(xmlWriter, font, name=name)
            xmlWriter.endtag(valueName)
            xmlWriter.newline()
        else:
            xmlWriter.simpletag(valueName, simpleItems)
            xmlWriter.newline()

    def fromXML(self, name, attrs, content, font):
        from . import otTables

        for k, v in attrs.items():
            setattr(self, k, int(v))
        for element in content:
            if not isinstance(element, tuple):
                continue
            name, attrs, content = element
            value = getattr(otTables, name)()
            for elem2 in content:
                if not isinstance(elem2, tuple):
                    continue
                name2, attrs2, content2 = elem2
                value.fromXML(name2, attrs2, content2, font)
            setattr(self, name, value)

    def __ne__(self, other):
        result = self.__eq__(other)
        return result if result is NotImplemented else not result

    def __eq__(self, other):
        if type(self) != type(other):
            return NotImplemented
        return self.__dict__ == other.__dict__
<|endoftext|>from fontTools.misc import sstruct
from fontTools.misc.fixedTools import floatToFixedToStr
from fontTools.misc.textTools import safeEval
from . import DefaultTable
from . import grUtils
import struct

Sill_hdr = """
    >
    version:    16.16F
"""


class table_S__i_l_l(DefaultTable.DefaultTable):
    """Graphite Languages table

    See also https://graphite.sil.org/graphite_techAbout#graphite-font-tables
    """

    def __init__(self, tag=None):
        DefaultTable.DefaultTable.__init__(self, tag)
        self.langs = {}

    def decompile(self, data, ttFont):
        _, data = sstruct.unpack2(Sill_hdr, data, self)
        self.version = float(floatToFixedToStr(self.version, precisionBits=16))
        (numLangs,) = struct.unpack(">H", data[:2])
        data = data[8:]
        maxsetting = 0
        langinfo = []
        for i in range(numLangs):
            langcode, numsettings, offset = struct.unpack(
                ">4sHH", data[i * 8 : (i + 1) * 8]
            )
            offset = int(offset / 8) - (numLangs + 1)
            langcode = langcode.replace(b"\000", b"")
            langinfo.append((langcode.decode("utf-8"), numsettings, offset))
            maxsetting = max(maxsetting, offset + numsettings)
        data = data[numLangs * 8 :]
        finfo = []
        for i in range(maxsetting):
            fid, val, _ = struct.unpack(">LHH", data[i * 8 : (i + 1) * 8])
            finfo.append((fid, val))
        self.langs = {}
        for c, n, o in langinfo:
            self.langs[c] = []
            for i in range(o, o + n):
                self.langs[c].append(finfo[i])

    def compile(self, ttFont):
        ldat = b""
        fdat = b""
        offset = len(self.langs)
        for c, inf in sorted(self.langs.items()):
            ldat += struct.pack(">4sHH", c.encode("utf8"), len(inf), 8 * offset + 20)
            for fid, val in inf:
                fdat += struct.pack(">LHH", fid, val, 0)
            offset += len(inf)
        ldat += struct.pack(">LHH", 0x80808080, 0, 8 * offset + 20)
        return (
            sstruct.pack(Sill_hdr, self)
            + grUtils.bininfo(len(self.langs))
            + ldat
            + fdat
        )

    def toXML(self, writer, ttFont):
        writer.simpletag("version", version=self.version)
        writer.newline()
        for c, inf in sorted(self.langs.items()):
            writer.begintag("lang", name=c)
            writer.newline()
            for fid, val in inf:
                writer.simpletag("feature", fid=grUtils.num2tag(fid), val=val)
                writer.newline()
            writer.endtag("lang")
            writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        if name == "version":
            self.version = float(safeEval(attrs["version"]))
        elif name == "lang":
            c = attrs["name"]
            self.langs[c] = []
            for element in content:
                if not isinstance(element, tuple):
                    continue
                tag, a, subcontent = element
                if tag == "feature":
                    self.langs[c].append(
                        (grUtils.tag2num(a["fid"]), int(safeEval(a["val"])))
                    )
<|endoftext|>"""Support for the `bgcl` (Apple Color Emoji background) table.

This table stores a JSON blob. We decode it to a Python object on
decompile and emit human readable JSON in the TTX via a <json> element.
On compile we re-encode the JSON to UTF-8 bytes which is what Apple code
reads via CTFontCopyTable then JSONDecoder.

On iOS 16 and later the `bgcl` payload is used as the wallpaper
background when the user selects an emoji wallpaper.

Fields:

- ``colors``: list of palette entries. Each entry is an array of four
    integers ``[R, G, B, A]``. R/G/B are 0-255. A is 0-1.

- ``emojicolors``: list of per-emoji palettes. Each item is an array of
    three sublists: primary/dominant, accent, contextual
    (names inferred). Each sublist contains integer indexes referencing
    entries in ``colors``; the runtime uses these to assemble layered
    background tints for an emoji.

- ``indexmap``: mapping (glyph identifier → palette index). The map
    maps a glyph identity to an integer index selecting an entry in
    ``emojicolors``. The font/UI uses this to pick the correct palette
    for a glyph when rendering wallpaper backgrounds.

- ``version``: integer table version used for parsing/compatibility.

Runtime usage summary:

- The system fetches the table bytes with ``CTFontCopyTable('bgcl')``,
    decodes the bytes as UTF‑8 JSON and runs the JSON through the app's
    decoder into an internal ``BgclTable`` structure. The app looks up a
    glyph's entry in ``indexmap``, retrieves the corresponding
    ``emojicolors`` palette, then converts the referenced ``colors``
    entries into color objects (normalizing channels/alpha as needed).
    This resulting color(s) drive the wallpaper background appearance for
    emoji wallpapers on supported iOS versions.

This implementation preserves the JSON payload and exposes convenience
attributes ``colors``, ``emojicolors``, ``indexmap`` and ``version``
when available.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fontTools.misc.textTools import tostr, strjoin
from . import DefaultTable
import json

if TYPE_CHECKING:
    from fontTools.misc.xmlWriter import XMLWriter
    from fontTools.ttLib import TTFont


class table__b_g_c_l(DefaultTable.DefaultTable):
    """bgcl table: stores a JSON blob describing background palettes.

    The JSON structure typically contains the top-level keys:
      - colors: [[R,G,B,A], ...]
      - emojicolors: [[[dominant...],[accent...],[contextual...]], ...]
      - indexmap: {"glyphIndex": emojicolors_index, ...}
      - version: int
    """

    def decompile(self, data: bytes, ttFont: TTFont) -> None:
        """Store raw bytes and attempt to parse JSON.

        The JSON commonly includes palette/lookup data used at runtime to
        construct wallpaper/background colors for emoji wallpapers on
        recent iOS releases.
        """
        self.data = data
        try:
            text = tostr(data, "utf_8")
            self.json = json.loads(text)
        except Exception as e:  # keep table decompilation robust
            self.json = None
            self.ERROR = f"bgcl JSON parse error: {e!r}"
            return
        # convenient attributes
        self.colors = self.json.get("colors")
        self.emojicolors = self.json.get("emojicolors")
        self.indexmap = self.json.get("indexmap")
        self.version = self.json.get("version")

    def compile(self, ttFont: TTFont) -> bytes:
        """Encode the JSON object to UTF-8 bytes for font binary storage."""
        if getattr(self, "json", None) is None:
            # fallback to raw bytes if parsing failed earlier
            return getattr(self, "data", b"")
        # use compact representation for binary table
        return json.dumps(self.json, separators=(",", ":"), ensure_ascii=False).encode(
            "utf_8"
        )

    def toXML(self, writer: XMLWriter, ttFont: TTFont) -> None:
        """Emit pretty-printed JSON inside a <json> element for human inspection."""
        if getattr(self, "json", None) is None:
            # fallback to default hex output
            DefaultTable.DefaultTable.toXML(self, writer, ttFont)
            return
        writer.begintag("json")
        writer.newline()
        pretty = json.dumps(self.json, indent=2, ensure_ascii=False)
        writer.writecdata(pretty)
        writer.newline()
        writer.endtag("json")
        writer.newline()

    def fromXML(
        self, name: str, attrs: dict[str, str], content, ttFont: TTFont
    ) -> None:
        """Read JSON from the <json> element. `content` may be a list.

        This mirrors SVG/other table `fromXML` handlers which accept a
        list of content chunks.
        """
        if name != "json":
            # fall back to DefaultTable behavior for unknown elements
            return DefaultTable.DefaultTable.fromXML(self, name, attrs, content, ttFont)
        text = strjoin(content).strip()
        try:
            self.json = json.loads(text)
            # keep raw bytes in sync
            self.data = text.encode("utf_8")
            self.colors = self.json.get("colors")
            self.emojicolors = self.json.get("emojicolors")
            self.indexmap = self.json.get("indexmap")
            self.version = self.json.get("version")
        except Exception as e:
            # store error and fall back to raw
            self.json = None
            self.ERROR = f"bgcl JSON parse error in fromXML: {e!r}"
<|endoftext|>from fontTools.misc import sstruct
from fontTools.misc.textTools import safeEval
from fontTools.misc.fixedTools import (
    ensureVersionIsLong as fi2ve,
    versionToFixed as ve2fi,
)
from . import DefaultTable
import math

hheaFormat = """
		>  # big endian
		tableVersion:           L
		ascent:                 h
		descent:                h
		lineGap:                h
		advanceWidthMax:        H
		minLeftSideBearing:     h
		minRightSideBearing:    h
		xMaxExtent:             h
		caretSlopeRise:         h
		caretSlopeRun:          h
		caretOffset:            h
		reserved0:              h
		reserved1:              h
		reserved2:              h
		reserved3:              h
		metricDataFormat:       h
		numberOfHMetrics:       H
"""


class table__h_h_e_a(DefaultTable.DefaultTable):
    """Horizontal Header table

    The ``hhea`` table contains information needed during horizontal
    text layout.

    .. note::
       This converter class is kept in sync with the :class:`._v_h_e_a.table__v_h_e_a`
       table constructor.

    See also https://learn.microsoft.com/en-us/typography/opentype/spec/hhea
    """

    # Note: Keep in sync with table__v_h_e_a

    dependencies = ["hmtx", "glyf", "CFF ", "CFF2"]

    # OpenType spec renamed these, add aliases for compatibility
    @property
    def ascender(self):
        return self.ascent

    @ascender.setter
    def ascender(self, value):
        self.ascent = value

    @property
    def descender(self):
        return self.descent

    @descender.setter
    def descender(self, value):
        self.descent = value

    def decompile(self, data, ttFont):
        sstruct.unpack(hheaFormat, data, self)

    def compile(self, ttFont):
        if ttFont.recalcBBoxes and (
            ttFont.isLoaded("glyf")
            or ttFont.isLoaded("CFF ")
            or ttFont.isLoaded("CFF2")
        ):
            self.recalc(ttFont)
        self.tableVersion = fi2ve(self.tableVersion)
        return sstruct.pack(hheaFormat, self)

    def recalc(self, ttFont):
        if "hmtx" not in ttFont:
            return

        hmtxTable = ttFont["hmtx"]
        self.advanceWidthMax = max(adv for adv, _ in hmtxTable.metrics.values())

        boundsWidthDict = {}
        if "glyf" in ttFont:
            glyfTable = ttFont["glyf"]
            for name in ttFont.getGlyphOrder():
                g = glyfTable[name]
                if g.numberOfContours == 0:
                    continue
                if g.numberOfContours < 0 and not hasattr(g, "xMax"):
                    # Composite glyph without extents set.
                    # Calculate those.
                    g.recalcBounds(glyfTable)
                boundsWidthDict[name] = g.xMax - g.xMin
        elif "CFF " in ttFont or "CFF2" in ttFont:
            if "CFF " in ttFont:
                topDict = ttFont["CFF "].cff.topDictIndex[0]
            else:
                topDict = ttFont["CFF2"].cff.topDictIndex[0]
            charStrings = topDict.CharStrings
            for name in ttFont.getGlyphOrder():
                cs = charStrings[name]
                bounds = cs.calcBounds(charStrings)
                if bounds is not None:
                    boundsWidthDict[name] = int(
                        math.ceil(bounds[2]) - math.floor(bounds[0])
                    )

        if boundsWidthDict:
            minLeftSideBearing = float("inf")
            minRightSideBearing = float("inf")
            xMaxExtent = -float("inf")
            for name, boundsWidth in boundsWidthDict.items():
                advanceWidth, lsb = hmtxTable[name]
                rsb = advanceWidth - lsb - boundsWidth
                extent = lsb + boundsWidth
                minLeftSideBearing = min(minLeftSideBearing, lsb)
                minRightSideBearing = min(minRightSideBearing, rsb)
                xMaxExtent = max(xMaxExtent, extent)
            self.minLeftSideBearing = minLeftSideBearing
            self.minRightSideBearing = minRightSideBearing
            self.xMaxExtent = xMaxExtent

        else:  # No glyph has outlines.
            self.minLeftSideBearing = 0
            self.minRightSideBearing = 0
            self.xMaxExtent = 0

    def toXML(self, writer, ttFont):
        formatstring, names, fixes = sstruct.getformat(hheaFormat)
        for name in names:
            value = getattr(self, name)
            if name == "tableVersion":
                value = fi2ve(value)
                value = "0x%08x" % value
            writer.simpletag(name, value=value)
            writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        if name == "tableVersion":
            setattr(self, name, ve2fi(attrs["value"]))
            return
        setattr(self, name, safeEval(attrs["value"]))
<|endoftext|>from . import DefaultTable
from fontTools.misc import sstruct
from fontTools.misc.textTools import bytesjoin
from fontTools.ttLib.tables.TupleVariation import (
    compileTupleVariationStore,
    decompileTupleVariationStore,
    TupleVariation,
)

# https://www.microsoft.com/typography/otspec/cvar.htm
# https://www.microsoft.com/typography/otspec/otvarcommonformats.htm
# https://developer.apple.com/fonts/TrueType-Reference-Manual/RM06/Chap6cvar.html

CVAR_HEADER_FORMAT = """
    > # big endian
    majorVersion:        H
    minorVersion:        H
    tupleVariationCount: H
    offsetToData:        H
"""

CVAR_HEADER_SIZE = sstruct.calcsize(CVAR_HEADER_FORMAT)


class table__c_v_a_r(DefaultTable.DefaultTable):
    """Control Value Table (CVT) variations table

    The ``cvar`` table contains variations for the values in a ``cvt``
    table.

    See also https://learn.microsoft.com/en-us/typography/opentype/spec/cvar
    """

    dependencies = ["cvt ", "fvar"]

    def __init__(self, tag=None):
        DefaultTable.DefaultTable.__init__(self, tag)
        self.majorVersion, self.minorVersion = 1, 0
        self.variations = []

    def compile(self, ttFont, useSharedPoints=False):
        tupleVariationCount, tuples, data = compileTupleVariationStore(
            variations=[v for v in self.variations if v.hasImpact()],
            pointCount=len(ttFont["cvt "].values),
            axisTags=[axis.axisTag for axis in ttFont["fvar"].axes],
            sharedTupleIndices={},
            useSharedPoints=useSharedPoints,
        )
        header = {
            "majorVersion": self.majorVersion,
            "minorVersion": self.minorVersion,
            "tupleVariationCount": tupleVariationCount,
            "offsetToData": CVAR_HEADER_SIZE + len(tuples),
        }
        return b"".join([sstruct.pack(CVAR_HEADER_FORMAT, header), tuples, data])

    def decompile(self, data, ttFont):
        axisTags = [axis.axisTag for axis in ttFont["fvar"].axes]
        header = {}
        sstruct.unpack(CVAR_HEADER_FORMAT, data[0:CVAR_HEADER_SIZE], header)
        self.majorVersion = header["majorVersion"]
        self.minorVersion = header["minorVersion"]
        assert self.majorVersion == 1, self.majorVersion
        self.variations = decompileTupleVariationStore(
            tableTag=self.tableTag,
            axisTags=axisTags,
            tupleVariationCount=header["tupleVariationCount"],
            pointCount=len(ttFont["cvt "].values),
            sharedTuples=None,
            data=data,
            pos=CVAR_HEADER_SIZE,
            dataPos=header["offsetToData"],
        )

    def fromXML(self, name, attrs, content, ttFont):
        if name == "version":
            self.majorVersion = int(attrs.get("major", "1"))
            self.minorVersion = int(attrs.get("minor", "0"))
        elif name == "tuple":
            valueCount = len(ttFont["cvt "].values)
            var = TupleVariation({}, [None] * valueCount)
            self.variations.append(var)
            for tupleElement in content:
                if isinstance(tupleElement, tuple):
                    tupleName, tupleAttrs, tupleContent = tupleElement
                    var.fromXML(tupleName, tupleAttrs, tupleContent)

    def toXML(self, writer, ttFont):
        axisTags = [axis.axisTag for axis in ttFont["fvar"].axes]
        writer.simpletag("version", major=self.majorVersion, minor=self.minorVersion)
        writer.newline()
        for var in self.variations:
            var.toXML(writer, axisTags)
<|endoftext|>from fontTools.ttLib import getSearchRange
from fontTools.misc.textTools import safeEval, readHex
from fontTools.misc.fixedTools import fixedToFloat as fi2fl, floatToFixed as fl2fi
from . import DefaultTable
import struct
import sys
import array
import logging

log = logging.getLogger(__name__)


class table__k_e_r_n(DefaultTable.DefaultTable):
    """Kerning table

    The ``kern`` table contains values that contextually adjust the inter-glyph
    spacing for the glyphs in a ``glyf`` table.

    Note that similar contextual spacing adjustments can also be stored
    in the "kern" feature of a ``GPOS`` table.

    See also https://learn.microsoft.com/en-us/typography/opentype/spec/kern
    """

    def getkern(self, format):
        for subtable in self.kernTables:
            if subtable.format == format:
                return subtable
        return None  # not found

    def decompile(self, data, ttFont):
        version, nTables = struct.unpack(">HH", data[:4])
        apple = False
        if (len(data) >= 8) and (version == 1):
            # AAT Apple's "new" format. Hm.
            version, nTables = struct.unpack(">LL", data[:8])
            self.version = fi2fl(version, 16)
            data = data[8:]
            apple = True
        else:
            self.version = version
            data = data[4:]
        self.kernTables = []
        for i in range(nTables):
            if self.version == 1.0:
                # Apple
                length, coverage, subtableFormat = struct.unpack(">LBB", data[:6])
            else:
                # in OpenType spec the "version" field refers to the common
                # subtable header; the actual subtable format is stored in
                # the 8-15 mask bits of "coverage" field.
                # This "version" is always 0 so we ignore it here
                _, length, subtableFormat, coverage = struct.unpack(">HHBB", data[:6])
                if nTables == 1 and subtableFormat == 0:
                    # The "length" value is ignored since some fonts
                    # (like OpenSans and Calibri) have a subtable larger than
                    # its value.
                    (nPairs,) = struct.unpack(">H", data[6:8])
                    calculated_length = (nPairs * 6) + 14
                    if length != calculated_length:
                        log.warning(
                            "'kern' subtable longer than defined: "
                            "%d bytes instead of %d bytes" % (calculated_length, length)
                        )
                    length = calculated_length
            if subtableFormat not in kern_classes:
                subtable = KernTable_format_unkown(subtableFormat)
            else:
                subtable = kern_classes[subtableFormat](apple)
            subtable.decompile(data[:length], ttFont)
            self.kernTables.append(subtable)
            data = data[length:]

    def compile(self, ttFont):
        if hasattr(self, "kernTables"):
            nTables = len(self.kernTables)
        else:
            nTables = 0
        if self.version == 1.0:
            # AAT Apple's "new" format.
            data = struct.pack(">LL", fl2fi(self.version, 16), nTables)
        else:
            data = struct.pack(">HH", self.version, nTables)
        if hasattr(self, "kernTables"):
            for subtable in self.kernTables:
                data = data + subtable.compile(ttFont)
        return data

    def toXML(self, writer, ttFont):
        writer.simpletag("version", value=self.version)
        writer.newline()
        for subtable in self.kernTables:
            subtable.toXML(writer, ttFont)

    def fromXML(self, name, attrs, content, ttFont):
        if name == "version":
            self.version = safeEval(attrs["value"])
            return
        if name != "kernsubtable":
            return
        if not hasattr(self, "kernTables"):
            self.kernTables = []
        format = safeEval(attrs["format"])
        if format not in kern_classes:
            subtable = KernTable_format_unkown(format)
        else:
            apple = self.version == 1.0
            subtable = kern_classes[format](apple)
        self.kernTables.append(subtable)
        subtable.fromXML(name, attrs, content, ttFont)


class KernTable_format_0(object):
    # 'version' is kept for backward compatibility
    version = format = 0

    def __init__(self, apple=False):
        self.apple = apple

    def decompile(self, data, ttFont):
        if not self.apple:
            version, length, subtableFormat, coverage = struct.unpack(">HHBB", data[:6])
            if version != 0:
                from fontTools.ttLib import TTLibError

                raise TTLibError("unsupported kern subtable version: %d" % version)
            tupleIndex = None
            # Should we also assert length == len(data)?
            data = data[6:]
        else:
            length, coverage, subtableFormat, tupleIndex = struct.unpack(
                ">LBBH", data[:8]
            )
            data = data[8:]
        assert self.format == subtableFormat, "unsupported format"
        self.coverage = coverage
        self.tupleIndex = tupleIndex

        self.kernTable = kernTable = {}

        nPairs, searchRange, entrySelector, rangeShift = struct.unpack(
            ">HHHH", data[:8]
        )
        data = data[8:]

        datas = array.array("H", data[: 6 * nPairs])
        if sys.byteorder != "big":
            datas.byteswap()
        it = iter(datas)
        glyphOrder = ttFont.getGlyphOrder()
        for k in range(nPairs):
            left, right, value = next(it), next(it), next(it)
            if value >= 32768:
                value -= 65536
            try:
                kernTable[(glyphOrder[left], glyphOrder[right])] = value
            except IndexError:
                # Slower, but will not throw an IndexError on an invalid
                # glyph id.
                kernTable[(ttFont.getGlyphName(left), ttFont.getGlyphName(right))] = (
                    value
                )
        if len(data) > 6 * nPairs + 4:  # Ignore up to 4 bytes excess
            log.warning(
                "excess data in 'kern' subtable: %d bytes", len(data) - 6 * nPairs
            )

    def compile(self, ttFont):
        nPairs = min(len(self.kernTable), 0xFFFF)
        searchRange, entrySelector, rangeShift = getSearchRange(nPairs, 6)
        searchRange &= 0xFFFF
        entrySelector = min(entrySelector, 0xFFFF)
        rangeShift = min(rangeShift, 0xFFFF)
        data = struct.pack(">HHHH", nPairs, searchRange, entrySelector, rangeShift)

        # yeehee! (I mean, turn names into indices)
        try:
            reverseOrder = ttFont.getReverseGlyphMap()
            kernTable = sorted(
                (reverseOrder[left], reverseOrder[right], value)
                for ((left, right), value) in self.kernTable.items()
            )
        except KeyError:
            # Slower, but will not throw KeyError on invalid glyph id.
            getGlyphID = ttFont.getGlyphID
            kernTable = sorted(
                (getGlyphID(left), getGlyphID(right), value)
                for ((left, right), value) in self.kernTable.items()
            )

        for left, right, value in kernTable:
            data = data + struct.pack(">HHh", left, right, value)

        if not self.apple:
            version = 0
            length = len(data) + 6
            if length >= 0x10000:
                log.warning(
                    '"kern" subtable overflow, '
                    "truncating length value while preserving pairs."
                )
                length &= 0xFFFF
            header = struct.pack(">HHBB", version, length, self.format, self.coverage)
        else:
            if self.tupleIndex is None:
                # sensible default when compiling a TTX from an old fonttools
                # or when inserting a Windows-style format 0 subtable into an
                # Apple version=1.0 kern table
                log.warning("'tupleIndex' is None; default to 0")
                self.tupleIndex = 0
            length = len(data) + 8
            header = struct.pack(
                ">LBBH", length, self.coverage, self.format, self.tupleIndex
            )
        return header + data

    def toXML(self, writer, ttFont):
        attrs = dict(coverage=self.coverage, format=self.format)
        if self.apple:
            if self.tupleIndex is None:
                log.warning("'tupleIndex' is None; default to 0")
                attrs["tupleIndex"] = 0
            else:
                attrs["tupleIndex"] = self.tupleIndex
        writer.begintag("kernsubtable", **attrs)
        writer.newline()
        items = sorted(self.kernTable.items())
        for (left, right), value in items:
            writer.simpletag("pair", [("l", left), ("r", right), ("v", value)])
            writer.newline()
        writer.endtag("kernsubtable")
        writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        self.coverage = safeEval(attrs["coverage"])
        subtableFormat = safeEval(attrs["format"])
        if self.apple:
            if "tupleIndex" in attrs:
                self.tupleIndex = safeEval(attrs["tupleIndex"])
            else:
                # previous fontTools versions didn't export tupleIndex
                log.warning("Apple kern subtable is missing 'tupleIndex' attribute")
                self.tupleIndex = None
        else:
            self.tupleIndex = None
        assert subtableFormat == self.format, "unsupported format"
        if not hasattr(self, "kernTable"):
            self.kernTable = {}
        for element in content:
            if not isinstance(element, tuple):
                continue
            name, attrs, content = element
            self.kernTable[(attrs["l"], attrs["r"])] = safeEval(attrs["v"])

    def __getitem__(self, pair):
        return self.kernTable[pair]

    def __setitem__(self, pair, value):
        self.kernTable[pair] = value

    def __delitem__(self, pair):
        del self.kernTable[pair]


class KernTable_format_unkown(object):
    def __init__(self, format):
        self.format = format

    def decompile(self, data, ttFont):
        self.data = data

    def compile(self, ttFont):
        return self.data

    def toXML(self, writer, ttFont):
        writer.begintag("kernsubtable", format=self.format)
        writer.newline()
        writer.comment("unknown 'kern' subtable format")
        writer.newline()
        writer.dumphex(self.data)
        writer.endtag("kernsubtable")
        writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        self.decompile(readHex(content), ttFont)


kern_classes = {0: KernTable_format_0}
<|endoftext|>from fontTools.misc import sstruct
from fontTools.misc.textTools import safeEval
from . import DefaultTable

maxpFormat_0_5 = """
		>	# big endian
		tableVersion:           i
		numGlyphs:              H
"""

maxpFormat_1_0_add = """
		>	# big endian
		maxPoints:              H
		maxContours:            H
		maxCompositePoints:     H
		maxCompositeContours:   H
		maxZones:               H
		maxTwilightPoints:      H
		maxStorage:             H
		maxFunctionDefs:        H
		maxInstructionDefs:     H
		maxStackElements:       H
		maxSizeOfInstructions:  H
		maxComponentElements:   H
		maxComponentDepth:      H
"""


class table__m_a_x_p(DefaultTable.DefaultTable):
    """Maximum Profile table

    The ``maxp`` table contains the memory requirements for the data in
    the font.

    See also https://learn.microsoft.com/en-us/typography/opentype/spec/maxp
    """

    dependencies = ["glyf"]

    def decompile(self, data, ttFont):
        dummy, data = sstruct.unpack2(maxpFormat_0_5, data, self)
        self.numGlyphs = int(self.numGlyphs)
        if self.tableVersion != 0x00005000:
            dummy, data = sstruct.unpack2(maxpFormat_1_0_add, data, self)
        assert len(data) == 0

    def compile(self, ttFont):
        if "glyf" in ttFont:
            if ttFont.isLoaded("glyf") and ttFont.recalcBBoxes:
                self.recalc(ttFont)
        else:
            pass  # CFF
        self.numGlyphs = len(ttFont.getGlyphOrder())
        if self.tableVersion != 0x00005000:
            self.tableVersion = 0x00010000
        data = sstruct.pack(maxpFormat_0_5, self)
        if self.tableVersion == 0x00010000:
            data = data + sstruct.pack(maxpFormat_1_0_add, self)
        return data

    def recalc(self, ttFont):
        """Recalculate the font bounding box, and most other maxp values except
        for the TT instructions values. Also recalculate the value of bit 1
        of the flags field and the font bounding box of the 'head' table.
        """
        glyfTable = ttFont["glyf"]
        hmtxTable = ttFont["hmtx"]
        headTable = ttFont["head"]
        self.numGlyphs = len(glyfTable)
        INFINITY = 100000
        xMin = +INFINITY
        yMin = +INFINITY
        xMax = -INFINITY
        yMax = -INFINITY
        maxPoints = 0
        maxContours = 0
        maxCompositePoints = 0
        maxCompositeContours = 0
        maxComponentElements = 0
        maxComponentDepth = 0
        allXMinIsLsb = 1
        for glyphName in ttFont.getGlyphOrder():
            g = glyfTable[glyphName]
            if g.numberOfContours:
                if hmtxTable[glyphName][1] != g.xMin:
                    allXMinIsLsb = 0
                xMin = min(xMin, g.xMin)
                yMin = min(yMin, g.yMin)
                xMax = max(xMax, g.xMax)
                yMax = max(yMax, g.yMax)
                if g.numberOfContours > 0:
                    nPoints, nContours = g.getMaxpValues()
                    maxPoints = max(maxPoints, nPoints)
                    maxContours = max(maxContours, nContours)
                elif g.isComposite():
                    nPoints, nContours, componentDepth = g.getCompositeMaxpValues(
                        glyfTable
                    )
                    maxCompositePoints = max(maxCompositePoints, nPoints)
                    maxCompositeContours = max(maxCompositeContours, nContours)
                    maxComponentElements = max(maxComponentElements, len(g.components))
                    maxComponentDepth = max(maxComponentDepth, componentDepth)
        if xMin == +INFINITY:
            headTable.xMin = 0
            headTable.yMin = 0
            headTable.xMax = 0
            headTable.yMax = 0
        else:
            headTable.xMin = xMin
            headTable.yMin = yMin
            headTable.xMax = xMax
            headTable.yMax = yMax
        self.maxPoints = maxPoints
        self.maxContours = maxContours
        self.maxCompositePoints = maxCompositePoints
        self.maxCompositeContours = maxCompositeContours
        self.maxComponentElements = maxComponentElements
        self.maxComponentDepth = maxComponentDepth
        if allXMinIsLsb:
            headTable.flags = headTable.flags | 0x2
        else:
            headTable.flags = headTable.flags & ~0x2

    def testrepr(self):
        items = sorted(self.__dict__.items())
        print(". . . . . . . . .")
        for combo in items:
            print("  %s: %s" % combo)
        print(". . . . . . . . .")

    def toXML(self, writer, ttFont):
        if self.tableVersion != 0x00005000:
            writer.comment("Most of this table will be recalculated by the compiler")
            writer.newline()
        formatstring, names, fixes = sstruct.getformat(maxpFormat_0_5)
        if self.tableVersion != 0x00005000:
            formatstring, names_1_0, fixes = sstruct.getformat(maxpFormat_1_0_add)
            names = {**names, **names_1_0}
        for name in names:
            value = getattr(self, name)
            if name == "tableVersion":
                value = hex(value)
            writer.simpletag(name, value=value)
            writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        setattr(self, name, safeEval(attrs["value"]))
<|endoftext|>from fontTools.misc import sstruct
from fontTools.misc.textTools import safeEval, num2binary, binary2num
from . import DefaultTable
from .sbixStrike import Strike

sbixHeaderFormat = """
	>
	version:       H	# Version number (set to 1)
	flags:         H	# The only two bits used in the flags field are bits 0
						# and 1. For historical reasons, bit 0 must always be 1.
						# Bit 1 is a sbixDrawOutlines flag and is interpreted as
						# follows:
						#     0: Draw only 'sbix' bitmaps
						#     1: Draw both 'sbix' bitmaps and outlines, in that
						#        order
	numStrikes:    L	# Number of bitmap strikes to follow
"""
sbixHeaderFormatSize = sstruct.calcsize(sbixHeaderFormat)


sbixStrikeOffsetFormat = """
	>
	strikeOffset:  L	# Offset from begining of table to data for the
						# individual strike
"""
sbixStrikeOffsetFormatSize = sstruct.calcsize(sbixStrikeOffsetFormat)


class table__s_b_i_x(DefaultTable.DefaultTable):
    """Standard Bitmap Graphics table

    The ``sbix`` table stores bitmap image data in standard graphics formats
    like JPEG, PNG, or TIFF. The glyphs for which the ``sbix`` table provides
    data are indexed by Glyph ID. For each such glyph, the ``sbix`` table can
    hold different data for different sizes, called "strikes."

    See also https://learn.microsoft.com/en-us/typography/opentype/spec/sbix
    """

    def __init__(self, tag=None):
        DefaultTable.DefaultTable.__init__(self, tag)
        self.version = 1
        self.flags = 1
        self.numStrikes = 0
        self.strikes = {}
        self.strikeOffsets = []

    def decompile(self, data, ttFont):
        # read table header
        sstruct.unpack(sbixHeaderFormat, data[:sbixHeaderFormatSize], self)
        # collect offsets to individual strikes in self.strikeOffsets
        for i in range(self.numStrikes):
            current_offset = sbixHeaderFormatSize + i * sbixStrikeOffsetFormatSize
            offset_entry = sbixStrikeOffset()
            sstruct.unpack(
                sbixStrikeOffsetFormat,
                data[current_offset : current_offset + sbixStrikeOffsetFormatSize],
                offset_entry,
            )
            self.strikeOffsets.append(offset_entry.strikeOffset)

        # decompile Strikes
        for i in range(self.numStrikes - 1, -1, -1):
            current_strike = Strike(rawdata=data[self.strikeOffsets[i] :])
            data = data[: self.strikeOffsets[i]]
            current_strike.decompile(ttFont)
            # print "  Strike length: %xh" % len(bitmapSetData)
            # print "Number of Glyph entries:", len(current_strike.glyphs)
            if current_strike.ppem in self.strikes:
                from fontTools import ttLib

                raise ttLib.TTLibError("Pixel 'ppem' must be unique for each Strike")
            self.strikes[current_strike.ppem] = current_strike

        # after the glyph data records have been extracted, we don't need the offsets anymore
        del self.strikeOffsets
        del self.numStrikes

    def compile(self, ttFont):
        sbixData = b""
        self.numStrikes = len(self.strikes)
        sbixHeader = sstruct.pack(sbixHeaderFormat, self)

        # calculate offset to start of first strike
        setOffset = sbixHeaderFormatSize + sbixStrikeOffsetFormatSize * self.numStrikes

        for si in sorted(self.strikes.keys()):
            current_strike = self.strikes[si]
            current_strike.compile(ttFont)
            # append offset to this strike to table header
            current_strike.strikeOffset = setOffset
            sbixHeader += sstruct.pack(sbixStrikeOffsetFormat, current_strike)
            setOffset += len(current_strike.data)
            sbixData += current_strike.data

        return sbixHeader + sbixData

    def toXML(self, xmlWriter, ttFont):
        xmlWriter.simpletag("version", value=self.version)
        xmlWriter.newline()
        xmlWriter.simpletag("flags", value=num2binary(self.flags, 16))
        xmlWriter.newline()
        for i in sorted(self.strikes.keys()):
            self.strikes[i].toXML(xmlWriter, ttFont)

    def fromXML(self, name, attrs, content, ttFont):
        if name == "version":
            setattr(self, name, safeEval(attrs["value"]))
        elif name == "flags":
            setattr(self, name, binary2num(attrs["value"]))
        elif name == "strike":
            current_strike = Strike()
            for element in content:
                if isinstance(element, tuple):
                    name, attrs, content = element
                    current_strike.fromXML(name, attrs, content, ttFont)
            self.strikes[current_strike.ppem] = current_strike
        else:
            from fontTools import ttLib

            raise ttLib.TTLibError("can't handle '%s' element" % name)


# Helper classes


class sbixStrikeOffset(object):
    pass
<|endoftext|>from __future__ import annotations

from typing import TYPE_CHECKING

from fontTools.misc.textTools import Tag
from fontTools.ttLib import getClassTag

if TYPE_CHECKING:
    from typing import Any

    from fontTools.misc.xmlWriter import XMLWriter
    from fontTools.ttLib import TTFont

XMLAttrs = dict[str, str]
XMLContent = list[str | tuple[str, XMLAttrs, "XMLContent"]]


class DefaultTable:
    dependencies: list[str] = []

    def __init__(self, tag: str | bytes | None = None) -> None:
        if tag is None:
            tag = getClassTag(self.__class__)
        self.tableTag = Tag(tag)

    def decompile(self, data: bytes, ttFont: TTFont) -> None:
        self.data = data

    def compile(self, ttFont: TTFont) -> bytes:
        return self.data

    def toXML(self, writer: XMLWriter, ttFont: TTFont, **kwargs: Any) -> None:
        if hasattr(self, "ERROR"):
            writer.comment("An error occurred during the decompilation of this table")
            writer.newline()
            writer.comment(self.ERROR)
            writer.newline()
        writer.begintag("hexdata")
        writer.newline()
        writer.dumphex(self.compile(ttFont))
        writer.endtag("hexdata")
        writer.newline()

    def fromXML(
        self, name: str, attrs: XMLAttrs, content: XMLContent, ttFont: TTFont
    ) -> None:
        from fontTools import ttLib
        from fontTools.misc.textTools import readHex

        if name != "hexdata":
            raise ttLib.TTLibError("can't handle '%s' element" % name)
        self.decompile(readHex(content), ttFont)

    def __repr__(self) -> str:
        return "<'%s' table at %x>" % (self.tableTag, id(self))

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return self.__dict__ == other.__dict__

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        return result if result is NotImplemented else not result
<|endoftext|>from fontTools.misc import sstruct
from fontTools.misc.fixedTools import floatToFixedToStr, strToFixedToFloat
from fontTools.misc.textTools import safeEval, num2binary, binary2num
from fontTools.misc.timeTools import (
    timestampFromString,
    timestampToString,
    timestampNow,
)
from fontTools.misc.timeTools import epoch_diff as mac_epoch_diff  # For backward compat
from fontTools.misc.arrayTools import intRect, unionRect
from . import DefaultTable
import logging

log = logging.getLogger(__name__)

headFormat = """
		>	# big endian
		tableVersion:       16.16F
		fontRevision:       16.16F
		checkSumAdjustment: I
		magicNumber:        I
		flags:              H
		unitsPerEm:         H
		created:            Q
		modified:           Q
		xMin:               h
		yMin:               h
		xMax:               h
		yMax:               h
		macStyle:           H
		lowestRecPPEM:      H
		fontDirectionHint:  h
		indexToLocFormat:   h
		glyphDataFormat:    h
"""


class table__h_e_a_d(DefaultTable.DefaultTable):
    """Font Header table

    The ``head`` table contains a variety of font-wide information.

    See also https://learn.microsoft.com/en-us/typography/opentype/spec/head
    """

    dependencies = ["maxp", "loca", "CFF ", "CFF2"]

    def decompile(self, data, ttFont):
        dummy, rest = sstruct.unpack2(headFormat, data, self)
        if rest:
            # this is quite illegal, but there seem to be fonts out there that do this
            log.warning("extra bytes at the end of 'head' table")
            assert rest == b"\0\0"

        # For timestamp fields, ignore the top four bytes.  Some fonts have
        # bogus values there.  Since till 2038 those bytes only can be zero,
        # ignore them.
        #
        # https://github.com/fonttools/fonttools/issues/99#issuecomment-66776810
        for stamp in "created", "modified":
            value = getattr(self, stamp)
            if value > 0xFFFFFFFF:
                log.warning("'%s' timestamp out of range; ignoring top bytes", stamp)
                value &= 0xFFFFFFFF
                setattr(self, stamp, value)
            if value < 0x7C259DC0:  # January 1, 1970 00:00:00
                log.warning(
                    "'%s' timestamp seems very low; regarding as unix timestamp", stamp
                )
                value += 0x7C259DC0
                setattr(self, stamp, value)

    def compile(self, ttFont):
        if ttFont.recalcBBoxes:
            # For TT-flavored fonts, xMin, yMin, xMax and yMax are set in table__m_a_x_p.recalc().
            if "CFF " in ttFont:
                topDict = ttFont["CFF "].cff.topDictIndex[0]
                self.xMin, self.yMin, self.xMax, self.yMax = intRect(topDict.FontBBox)
            elif "CFF2" in ttFont:
                topDict = ttFont["CFF2"].cff.topDictIndex[0]
                charStrings = topDict.CharStrings
                fontBBox = None
                for charString in charStrings.values():
                    bounds = charString.calcBounds(charStrings)
                    if bounds is not None:
                        if fontBBox is not None:
                            fontBBox = unionRect(fontBBox, bounds)
                        else:
                            fontBBox = bounds
                if fontBBox is not None:
                    self.xMin, self.yMin, self.xMax, self.yMax = intRect(fontBBox)
        if ttFont.recalcTimestamp:
            self.modified = timestampNow()
        data = sstruct.pack(headFormat, self)
        return data

    def toXML(self, writer, ttFont):
        writer.comment("Most of this table will be recalculated by the compiler")
        writer.newline()
        _, names, fixes = sstruct.getformat(headFormat)
        for name in names:
            value = getattr(self, name)
            if name in fixes:
                value = floatToFixedToStr(value, precisionBits=fixes[name])
            elif name in ("created", "modified"):
                value = timestampToString(value)
            elif name in ("magicNumber", "checkSumAdjustment"):
                if value < 0:
                    value = value + 0x100000000
                value = hex(value)
                if value[-1:] == "L":
                    value = value[:-1]
            elif name in ("macStyle", "flags"):
                value = num2binary(value, 16)
            writer.simpletag(name, value=value)
            writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        value = attrs["value"]
        fixes = sstruct.getformat(headFormat)[2]
        if name in fixes:
            value = strToFixedToFloat(value, precisionBits=fixes[name])
        elif name in ("created", "modified"):
            value = timestampFromString(value)
        elif name in ("macStyle", "flags"):
            value = binary2num(value)
        else:
            value = safeEval(value)
        setattr(self, name, value)
<|endoftext|>from __future__ import annotations

import base64
from typing import TYPE_CHECKING

from fontTools.misc import sstruct
from fontTools.misc.textTools import bytesjoin, safeEval, strjoin, tobytes, tostr

from . import DefaultTable

if TYPE_CHECKING:
    from fontTools.misc.xmlWriter import XMLWriter
    from fontTools.ttLib import TTFont

DSIG_HeaderFormat = """
	> # big endian
	ulVersion:      L
	usNumSigs:      H
	usFlag:         H
"""
# followed by an array of usNumSigs DSIG_Signature records
DSIG_SignatureFormat = """
	> # big endian
	ulFormat:       L
	ulLength:       L # length includes DSIG_SignatureBlock header
	ulOffset:       L
"""
# followed by an array of usNumSigs DSIG_SignatureBlock records,
# each followed immediately by the pkcs7 bytes
DSIG_SignatureBlockFormat = """
	> # big endian
	usReserved1:    H
	usReserved2:    H
	cbSignature:    l # length of following raw pkcs7 data
"""

#
# NOTE
# the DSIG table format allows for SignatureBlocks residing
# anywhere in the table and possibly in a different order as
# listed in the array after the first table header
#
# this implementation does not keep track of any gaps and/or data
# before or after the actual signature blocks while decompiling,
# and puts them in the same physical order as listed in the header
# on compilation with no padding whatsoever.
#


class table_D_S_I_G_(DefaultTable.DefaultTable):
    """Digital Signature table

    The ``DSIG`` table contains cryptographic signatures for the font.

    See also https://learn.microsoft.com/en-us/typography/opentype/spec/dsig
    """

    def decompile(self, data: bytes, ttFont: TTFont) -> None:
        dummy, newData = sstruct.unpack2(DSIG_HeaderFormat, data, self)
