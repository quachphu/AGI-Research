            self.version = safeEval(attrs["version"])
        elif name == "strikedata":
            if not hasattr(self, "strikeData"):
                self.strikeData = []
            strikeIndex = safeEval(attrs["index"])

            bitmapGlyphDict = {}
            for element in content:
                if not isinstance(element, tuple):
                    continue
                name, attrs, content = element
                if name[4:].startswith(_bitmapGlyphSubclassPrefix[4:]):
                    imageFormat = safeEval(name[len(_bitmapGlyphSubclassPrefix) :])
                    glyphName = attrs["name"]
                    imageFormatClass = self.getImageFormatClass(imageFormat)
                    curGlyph = imageFormatClass(None, None)
                    curGlyph.fromXML(name, attrs, content, ttFont)
                    assert glyphName not in bitmapGlyphDict, (
                        "Duplicate glyphs with the same name '%s' in the same strike."
                        % glyphName
                    )
                    bitmapGlyphDict[glyphName] = curGlyph
                else:
                    log.warning("%s being ignored by %s", name, self.__class__.__name__)

            # Grow the strike data array to the appropriate size. The XML
            # format allows the strike index value to be out of order.
            if strikeIndex >= len(self.strikeData):
                self.strikeData += [None] * (strikeIndex + 1 - len(self.strikeData))
            assert (
                self.strikeData[strikeIndex] is None
            ), "Duplicate strike EBDT indices."
            self.strikeData[strikeIndex] = bitmapGlyphDict


class EbdtComponent(object):
    def toXML(self, writer, ttFont):
        writer.begintag("ebdtComponent", [("name", self.name)])
        writer.newline()
        for componentName in sstruct.getformat(ebdtComponentFormat)[1][1:]:
            writer.simpletag(componentName, value=getattr(self, componentName))
            writer.newline()
        writer.endtag("ebdtComponent")
        writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        self.name = attrs["name"]
        componentNames = set(sstruct.getformat(ebdtComponentFormat)[1][1:])
        for element in content:
            if not isinstance(element, tuple):
                continue
            name, attrs, content = element
            if name in componentNames:
                vars(self)[name] = safeEval(attrs["value"])
            else:
                log.warning("unknown name '%s' being ignored by EbdtComponent.", name)


# Helper functions for dealing with binary.


def _data2binary(data, numBits):
    binaryList = []
    for curByte in data:
        value = byteord(curByte)
        numBitsCut = min(8, numBits)
        for i in range(numBitsCut):
            if value & 0x1:
                binaryList.append("1")
            else:
                binaryList.append("0")
            value = value >> 1
        numBits -= numBitsCut
    return strjoin(binaryList)


def _binary2data(binary):
    byteList = []
    for bitLoc in range(0, len(binary), 8):
        byteString = binary[bitLoc : bitLoc + 8]
        curByte = 0
        for curBit in reversed(byteString):
            curByte = curByte << 1
            if curBit == "1":
                curByte |= 1
        byteList.append(bytechr(curByte))
    return bytesjoin(byteList)


def _memoize(f):
    class memodict(dict):
        def __missing__(self, key):
            ret = f(key)
            if isinstance(key, int) or len(key) == 1:
                self[key] = ret
            return ret

    return memodict().__getitem__


# 00100111 -> 11100100 per byte, not to be confused with little/big endian.
# Bitmap data per byte is in the order that binary is written on the page
# with the least significant bit as far right as possible. This is the
# opposite of what makes sense algorithmically and hence this function.
@_memoize
def _reverseBytes(data):
    r"""
    >>> bin(ord(_reverseBytes(0b00100111)))
    '0b11100100'
    >>> _reverseBytes(b'\x00\xf0')
    b'\x00\x0f'
    """
    if isinstance(data, bytes) and len(data) != 1:
        return bytesjoin(map(_reverseBytes, data))
    byte = byteord(data)
    result = 0
    for i in range(8):
        result = result << 1
        result |= byte & 1
        byte = byte >> 1
    return bytechr(result)


# This section of code is for reading and writing image data to/from XML.


def _writeRawImageData(strikeIndex, glyphName, bitmapObject, writer, ttFont):
    writer.begintag("rawimagedata")
    writer.newline()
    writer.dumphex(bitmapObject.imageData)
    writer.endtag("rawimagedata")
    writer.newline()


def _readRawImageData(bitmapObject, name, attrs, content, ttFont):
    bitmapObject.imageData = readHex(content)


def _writeRowImageData(strikeIndex, glyphName, bitmapObject, writer, ttFont):
    metrics = bitmapObject.exportMetrics
    del bitmapObject.exportMetrics
    bitDepth = bitmapObject.exportBitDepth
    del bitmapObject.exportBitDepth

    writer.begintag(
        "rowimagedata", bitDepth=bitDepth, width=metrics.width, height=metrics.height
    )
    writer.newline()
    for curRow in range(metrics.height):
        rowData = bitmapObject.getRow(curRow, bitDepth=bitDepth, metrics=metrics)
        writer.simpletag("row", value=hexStr(rowData))
        writer.newline()
    writer.endtag("rowimagedata")
    writer.newline()


def _readRowImageData(bitmapObject, name, attrs, content, ttFont):
    bitDepth = safeEval(attrs["bitDepth"])
    metrics = SmallGlyphMetrics()
    metrics.width = safeEval(attrs["width"])
    metrics.height = safeEval(attrs["height"])

    dataRows = []
    for element in content:
        if not isinstance(element, tuple):
            continue
        name, attr, content = element
        # Chop off 'imagedata' from the tag to get just the option.
        if name == "row":
            dataRows.append(deHexStr(attr["value"]))
    bitmapObject.setRows(dataRows, bitDepth=bitDepth, metrics=metrics)


def _writeBitwiseImageData(strikeIndex, glyphName, bitmapObject, writer, ttFont):
    metrics = bitmapObject.exportMetrics
    del bitmapObject.exportMetrics
    bitDepth = bitmapObject.exportBitDepth
    del bitmapObject.exportBitDepth

    # A dict for mapping binary to more readable/artistic ASCII characters.
    binaryConv = {"0": ".", "1": "@"}

    writer.begintag(
        "bitwiseimagedata",
        bitDepth=bitDepth,
        width=metrics.width,
        height=metrics.height,
    )
    writer.newline()
    for curRow in range(metrics.height):
        rowData = bitmapObject.getRow(
            curRow, bitDepth=1, metrics=metrics, reverseBytes=True
        )
        rowData = _data2binary(rowData, metrics.width)
        # Make the output a readable ASCII art form.
        rowData = strjoin(map(binaryConv.get, rowData))
        writer.simpletag("row", value=rowData)
        writer.newline()
    writer.endtag("bitwiseimagedata")
    writer.newline()


def _readBitwiseImageData(bitmapObject, name, attrs, content, ttFont):
    bitDepth = safeEval(attrs["bitDepth"])
    metrics = SmallGlyphMetrics()
    metrics.width = safeEval(attrs["width"])
    metrics.height = safeEval(attrs["height"])

    # A dict for mapping from ASCII to binary. All characters are considered
    # a '1' except space, period and '0' which maps to '0'.
    binaryConv = {" ": "0", ".": "0", "0": "0"}

    dataRows = []
    for element in content:
        if not isinstance(element, tuple):
            continue
        name, attr, content = element
        if name == "row":
            mapParams = zip(attr["value"], itertools.repeat("1"))
            rowData = strjoin(itertools.starmap(binaryConv.get, mapParams))
            dataRows.append(_binary2data(rowData))

    bitmapObject.setRows(
        dataRows, bitDepth=bitDepth, metrics=metrics, reverseBytes=True
    )


def _extFileName(glyphName):
    # Glyph names come straight from the font and may contain path separators
    # (e.g. a crafted 'post' table), which would let the exported bitmap escape
    # the output folder. Names that are usable as a filename verbatim -- i.e.
    # every real glyph name -- are returned unchanged, so existing exports keep
    # their filenames; the rest are percent-encoded and marked with a leading
    # '%'. The mapping has to be injective, or two glyphs quietly share one
    # file and the second overwrites the first: a plain name is never encoded
    # and never contains '%', an encoded one always starts with '%', and
    # percent-encoding is itself reversible, so no two names can meet.
    if glyphName not in ("", ".", "..") and not _UNSAFE_IN_FILENAME.search(glyphName):
        return glyphName
    return "%" + quote(glyphName, safe="", encoding="utf-8", errors="surrogatepass")


def _writeExtFileImageData(strikeIndex, glyphName, bitmapObject, writer, ttFont):
    try:
        folder = os.path.dirname(writer.file.name)
    except AttributeError:
        # fall back to current directory if output file's directory isn't found
        folder = "."
    folder = os.path.join(folder, "bitmaps")
    filename = _extFileName(glyphName) + bitmapObject.fileExtension
    if not os.path.isdir(folder):
        os.makedirs(folder)
    folder = os.path.join(folder, "strike%d" % strikeIndex)
    if not os.path.isdir(folder):
        os.makedirs(folder)

    fullPath = os.path.join(folder, filename)
    writer.simpletag("extfileimagedata", value=fullPath)
    writer.newline()

    with open(fullPath, "wb") as file:
        file.write(bitmapObject.imageData)


def _readExtFileImageData(bitmapObject, name, attrs, content, ttFont):
    fullPath = attrs["value"]
    with open(fullPath, "rb") as file:
        bitmapObject.imageData = file.read()


# End of XML writing code.

# Important information about the naming scheme. Used for identifying formats
# in XML.
_bitmapGlyphSubclassPrefix = "ebdt_bitmap_format_"


class BitmapGlyph(object):
    # For the external file format. This can be changed in subclasses. This way
    # when the extfile option is turned on files have the form: glyphName.ext
    # The default is just a flat binary file with no meaning.
    fileExtension = ".bin"

    # Keep track of reading and writing of various forms.
    xmlDataFunctions = {
        "raw": (_writeRawImageData, _readRawImageData),
        "row": (_writeRowImageData, _readRowImageData),
        "bitwise": (_writeBitwiseImageData, _readBitwiseImageData),
        "extfile": (_writeExtFileImageData, _readExtFileImageData),
    }

    def __init__(self, data, ttFont):
        self.data = data
        self.ttFont = ttFont
        # TODO Currently non-lazy decompilation is untested here...
        # if not ttFont.lazy:
        # 	self.decompile()
        # 	del self.data

    def __getattr__(self, attr):
        # Allow lazy decompile.
        if attr[:2] == "__":
            raise AttributeError(attr)
        if attr == "data":
            raise AttributeError(attr)
        self.decompile()
        del self.data
        return getattr(self, attr)

    def ensureDecompiled(self, recurse=False):
        if hasattr(self, "data"):
            self.decompile()
            del self.data

    # Not a fan of this but it is needed for safer safety checking.
    def getFormat(self):
        return safeEval(self.__class__.__name__[len(_bitmapGlyphSubclassPrefix) :])

    def toXML(self, strikeIndex, glyphName, writer, ttFont):
        writer.begintag(self.__class__.__name__, [("name", glyphName)])
        writer.newline()

        self.writeMetrics(writer, ttFont)
        # Use the internal write method to write using the correct output format.
        self.writeData(strikeIndex, glyphName, writer, ttFont)

        writer.endtag(self.__class__.__name__)
        writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        self.readMetrics(name, attrs, content, ttFont)
        for element in content:
            if not isinstance(element, tuple):
                continue
            name, attr, content = element
            if not name.endswith("imagedata"):
                continue
            # Chop off 'imagedata' from the tag to get just the option.
            option = name[: -len("imagedata")]
            assert option in self.__class__.xmlDataFunctions
            self.readData(name, attr, content, ttFont)

    # Some of the glyphs have the metrics. This allows for metrics to be
    # added if the glyph format has them. Default behavior is to do nothing.
    def writeMetrics(self, writer, ttFont):
        pass

    # The opposite of write metrics.
    def readMetrics(self, name, attrs, content, ttFont):
        pass

    def writeData(self, strikeIndex, glyphName, writer, ttFont):
        try:
            writeFunc, readFunc = self.__class__.xmlDataFunctions[
                ttFont.bitmapGlyphDataFormat
            ]
        except KeyError:
            writeFunc = _writeRawImageData
        writeFunc(strikeIndex, glyphName, self, writer, ttFont)

    def readData(self, name, attrs, content, ttFont):
        # Chop off 'imagedata' from the tag to get just the option.
        option = name[: -len("imagedata")]
        writeFunc, readFunc = self.__class__.xmlDataFunctions[option]
        readFunc(self, name, attrs, content, ttFont)


# A closure for creating a mixin for the two types of metrics handling.
# Most of the code is very similar so its easier to deal with here.
# Everything works just by passing the class that the mixin is for.
def _createBitmapPlusMetricsMixin(metricsClass):
    # Both metrics names are listed here to make meaningful error messages.
    metricStrings = [BigGlyphMetrics.__name__, SmallGlyphMetrics.__name__]
    curMetricsName = metricsClass.__name__
    # Find which metrics this is for and determine the opposite name.
    metricsId = metricStrings.index(curMetricsName)
    oppositeMetricsName = metricStrings[1 - metricsId]

    class BitmapPlusMetricsMixin(object):
        def writeMetrics(self, writer, ttFont):
            self.metrics.toXML(writer, ttFont)

        def readMetrics(self, name, attrs, content, ttFont):
            for element in content:
                if not isinstance(element, tuple):
                    continue
                name, attrs, content = element
                if name == curMetricsName:
                    self.metrics = metricsClass()
                    self.metrics.fromXML(name, attrs, content, ttFont)
                elif name == oppositeMetricsName:
                    log.warning(
                        "Warning: %s being ignored in format %d.",
                        oppositeMetricsName,
                        self.getFormat(),
                    )

    return BitmapPlusMetricsMixin


# Since there are only two types of mixin's just create them here.
BitmapPlusBigMetricsMixin = _createBitmapPlusMetricsMixin(BigGlyphMetrics)
BitmapPlusSmallMetricsMixin = _createBitmapPlusMetricsMixin(SmallGlyphMetrics)


# Data that is bit aligned can be tricky to deal with. These classes implement
# helper functionality for dealing with the data and getting a particular row
# of bitwise data. Also helps implement fancy data export/import in XML.
class BitAlignedBitmapMixin(object):
    def _getBitRange(self, row, bitDepth, metrics):
        rowBits = bitDepth * metrics.width
        bitOffset = row * rowBits
        return (bitOffset, bitOffset + rowBits)

    def getRow(self, row, bitDepth=1, metrics=None, reverseBytes=False):
        if metrics is None:
            metrics = self.metrics
        assert 0 <= row and row < metrics.height, "Illegal row access in bitmap"

        # Loop through each byte. This can cover two bytes in the original data or
        # a single byte if things happen to be aligned. The very last entry might
        # not be aligned so take care to trim the binary data to size and pad with
        # zeros in the row data. Bit aligned data is somewhat tricky.
        #
        # Example of data cut. Data cut represented in x's.
        # '|' represents byte boundary.
        # data = ...0XX|XXXXXX00|000... => XXXXXXXX
        # 		or
        # data = ...0XX|XXXX0000|000... => XXXXXX00
        #   or
        # data = ...000|XXXXXXXX|000... => XXXXXXXX
        #   or
        # data = ...000|00XXXX00|000... => XXXX0000
        #
        dataList = []
        bitRange = self._getBitRange(row, bitDepth, metrics)
        stepRange = bitRange + (8,)
        for curBit in range(*stepRange):
            endBit = min(curBit + 8, bitRange[1])
            numBits = endBit - curBit
            cutPoint = curBit % 8
            firstByteLoc = curBit // 8
            secondByteLoc = endBit // 8
            if firstByteLoc < secondByteLoc:
                numBitsCut = 8 - cutPoint
            else:
                numBitsCut = endBit - curBit
            curByte = _reverseBytes(self.imageData[firstByteLoc])
            firstHalf = byteord(curByte) >> cutPoint
            firstHalf = ((1 << numBitsCut) - 1) & firstHalf
            newByte = firstHalf
            if firstByteLoc < secondByteLoc and secondByteLoc < len(self.imageData):
                curByte = _reverseBytes(self.imageData[secondByteLoc])
                secondHalf = byteord(curByte) << numBitsCut
                newByte = (firstHalf | secondHalf) & ((1 << numBits) - 1)
            dataList.append(bytechr(newByte))

        # The way the data is kept is opposite the algorithm used.
        data = bytesjoin(dataList)
        if not reverseBytes:
            data = _reverseBytes(data)
        return data

    def setRows(self, dataRows, bitDepth=1, metrics=None, reverseBytes=False):
        if metrics is None:
            metrics = self.metrics
        if not reverseBytes:
            dataRows = list(map(_reverseBytes, dataRows))

        # Keep track of a list of ordinal values as they are easier to modify
        # than a list of strings. Map to actual strings later.
        numBytes = (self._getBitRange(len(dataRows), bitDepth, metrics)[0] + 7) // 8
        ordDataList = [0] * numBytes
        for row, data in enumerate(dataRows):
            bitRange = self._getBitRange(row, bitDepth, metrics)
            stepRange = bitRange + (8,)
            for curBit, curByte in zip(range(*stepRange), data):
                endBit = min(curBit + 8, bitRange[1])
                cutPoint = curBit % 8
                firstByteLoc = curBit // 8
                secondByteLoc = endBit // 8
                if firstByteLoc < secondByteLoc:
                    numBitsCut = 8 - cutPoint
                else:
                    numBitsCut = endBit - curBit
                curByte = byteord(curByte)
                firstByte = curByte & ((1 << numBitsCut) - 1)
                ordDataList[firstByteLoc] |= firstByte << cutPoint
                if firstByteLoc < secondByteLoc and secondByteLoc < numBytes:
                    secondByte = (curByte >> numBitsCut) & ((1 << 8 - numBitsCut) - 1)
                    ordDataList[secondByteLoc] |= secondByte

        # Save the image data with the bits going the correct way.
        self.imageData = _reverseBytes(bytesjoin(map(bytechr, ordDataList)))


class ByteAlignedBitmapMixin(object):
    def _getByteRange(self, row, bitDepth, metrics):
        rowBytes = (bitDepth * metrics.width + 7) // 8
        byteOffset = row * rowBytes
        return (byteOffset, byteOffset + rowBytes)

    def getRow(self, row, bitDepth=1, metrics=None, reverseBytes=False):
        if metrics is None:
            metrics = self.metrics
        assert 0 <= row and row < metrics.height, "Illegal row access in bitmap"
        byteRange = self._getByteRange(row, bitDepth, metrics)
        data = self.imageData[slice(*byteRange)]
        if reverseBytes:
            data = _reverseBytes(data)
        return data

    def setRows(self, dataRows, bitDepth=1, metrics=None, reverseBytes=False):
        if metrics is None:
            metrics = self.metrics
        if reverseBytes:
            dataRows = map(_reverseBytes, dataRows)
        self.imageData = bytesjoin(dataRows)


class ebdt_bitmap_format_1(
    ByteAlignedBitmapMixin, BitmapPlusSmallMetricsMixin, BitmapGlyph
):
    def decompile(self):
        self.metrics = SmallGlyphMetrics()
        dummy, data = sstruct.unpack2(smallGlyphMetricsFormat, self.data, self.metrics)
        self.imageData = data

    def compile(self, ttFont):
        data = sstruct.pack(smallGlyphMetricsFormat, self.metrics)
        return data + self.imageData


class ebdt_bitmap_format_2(
    BitAlignedBitmapMixin, BitmapPlusSmallMetricsMixin, BitmapGlyph
):
    def decompile(self):
        self.metrics = SmallGlyphMetrics()
        dummy, data = sstruct.unpack2(smallGlyphMetricsFormat, self.data, self.metrics)
        self.imageData = data

    def compile(self, ttFont):
        data = sstruct.pack(smallGlyphMetricsFormat, self.metrics)
        return data + self.imageData


class ebdt_bitmap_format_5(BitAlignedBitmapMixin, BitmapGlyph):
    def decompile(self):
        self.imageData = self.data

    def compile(self, ttFont):
        return self.imageData


class ebdt_bitmap_format_6(
    ByteAlignedBitmapMixin, BitmapPlusBigMetricsMixin, BitmapGlyph
):
    def decompile(self):
        self.metrics = BigGlyphMetrics()
        dummy, data = sstruct.unpack2(bigGlyphMetricsFormat, self.data, self.metrics)
        self.imageData = data

    def compile(self, ttFont):
        data = sstruct.pack(bigGlyphMetricsFormat, self.metrics)
        return data + self.imageData


class ebdt_bitmap_format_7(
    BitAlignedBitmapMixin, BitmapPlusBigMetricsMixin, BitmapGlyph
):
    def decompile(self):
        self.metrics = BigGlyphMetrics()
        dummy, data = sstruct.unpack2(bigGlyphMetricsFormat, self.data, self.metrics)
        self.imageData = data

    def compile(self, ttFont):
        data = sstruct.pack(bigGlyphMetricsFormat, self.metrics)
        return data + self.imageData


class ComponentBitmapGlyph(BitmapGlyph):
    def toXML(self, strikeIndex, glyphName, writer, ttFont):
        writer.begintag(self.__class__.__name__, [("name", glyphName)])
        writer.newline()

        self.writeMetrics(writer, ttFont)

        writer.begintag("components")
        writer.newline()
        for curComponent in self.componentArray:
            curComponent.toXML(writer, ttFont)
        writer.endtag("components")
        writer.newline()

        writer.endtag(self.__class__.__name__)
        writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        self.readMetrics(name, attrs, content, ttFont)
        for element in content:
            if not isinstance(element, tuple):
                continue
            name, attr, content = element
            if name == "components":
                self.componentArray = []
                for compElement in content:
                    if not isinstance(compElement, tuple):
                        continue
                    name, attrs, content = compElement
                    if name == "ebdtComponent":
                        curComponent = EbdtComponent()
                        curComponent.fromXML(name, attrs, content, ttFont)
                        self.componentArray.append(curComponent)
                    else:
                        log.warning("'%s' being ignored in component array.", name)


class ebdt_bitmap_format_8(BitmapPlusSmallMetricsMixin, ComponentBitmapGlyph):
    def decompile(self):
        self.metrics = SmallGlyphMetrics()
        dummy, data = sstruct.unpack2(smallGlyphMetricsFormat, self.data, self.metrics)
        data = data[1:]

        (numComponents,) = struct.unpack(">H", data[:2])
        data = data[2:]
        self.componentArray = []
        for i in range(numComponents):
            curComponent = EbdtComponent()
            dummy, data = sstruct.unpack2(ebdtComponentFormat, data, curComponent)
            curComponent.name = self.ttFont.getGlyphName(curComponent.glyphCode)
            self.componentArray.append(curComponent)

    def compile(self, ttFont):
        dataList = []
        dataList.append(sstruct.pack(smallGlyphMetricsFormat, self.metrics))
        dataList.append(b"\0")
        dataList.append(struct.pack(">H", len(self.componentArray)))
        for curComponent in self.componentArray:
            curComponent.glyphCode = ttFont.getGlyphID(curComponent.name)
            dataList.append(sstruct.pack(ebdtComponentFormat, curComponent))
        return bytesjoin(dataList)


class ebdt_bitmap_format_9(BitmapPlusBigMetricsMixin, ComponentBitmapGlyph):
    def decompile(self):
        self.metrics = BigGlyphMetrics()
        dummy, data = sstruct.unpack2(bigGlyphMetricsFormat, self.data, self.metrics)
        (numComponents,) = struct.unpack(">H", data[:2])
        data = data[2:]
        self.componentArray = []
        for i in range(numComponents):
            curComponent = EbdtComponent()
            dummy, data = sstruct.unpack2(ebdtComponentFormat, data, curComponent)
            curComponent.name = self.ttFont.getGlyphName(curComponent.glyphCode)
            self.componentArray.append(curComponent)

    def compile(self, ttFont):
        dataList = []
        dataList.append(sstruct.pack(bigGlyphMetricsFormat, self.metrics))
        dataList.append(struct.pack(">H", len(self.componentArray)))
        for curComponent in self.componentArray:
            curComponent.glyphCode = ttFont.getGlyphID(curComponent.name)
            dataList.append(sstruct.pack(ebdtComponentFormat, curComponent))
        return bytesjoin(dataList)


# Dictionary of bitmap formats to the class representing that format
# currently only the ones listed in this map are the ones supported.
ebdt_bitmap_classes = {
    1: ebdt_bitmap_format_1,
    2: ebdt_bitmap_format_2,
    5: ebdt_bitmap_format_5,
    6: ebdt_bitmap_format_6,
    7: ebdt_bitmap_format_7,
    8: ebdt_bitmap_format_8,
    9: ebdt_bitmap_format_9,
}
<|endoftext|>from io import BytesIO
from fontTools import cffLib
from . import DefaultTable


class table_C_F_F_(DefaultTable.DefaultTable):
    """Compact Font Format table (version 1)

    The ``CFF`` table embeds a CFF-formatted font. The CFF font format
    predates OpenType and could be used as a standalone font file, but the
    ``CFF`` table is also used to package CFF fonts into an OpenType
    container.

    .. note::
       ``CFF`` has been succeeded by ``CFF2``, which eliminates much of
       the redundancy incurred by embedding CFF version 1 in an OpenType
       font.

    See also https://learn.microsoft.com/en-us/typography/opentype/spec/cff
    """

    def __init__(self, tag=None):
        DefaultTable.DefaultTable.__init__(self, tag)
        self.cff = cffLib.CFFFontSet()
        self._gaveGlyphOrder = False

    def decompile(self, data, otFont):
        self.cff.decompile(BytesIO(data), otFont, isCFF2=False)
        assert len(self.cff) == 1, "can't deal with multi-font CFF tables."

    def compile(self, otFont):
        f = BytesIO()
        self.cff.compile(f, otFont, isCFF2=False)
        return f.getvalue()

    def haveGlyphNames(self):
        if hasattr(self.cff[self.cff.fontNames[0]], "ROS"):
            return False  # CID-keyed font
        else:
            return True

    def getGlyphOrder(self):
        if self._gaveGlyphOrder:
            from fontTools import ttLib

            raise ttLib.TTLibError("illegal use of getGlyphOrder()")
        self._gaveGlyphOrder = True
        return self.cff[self.cff.fontNames[0]].getGlyphOrder()

    def setGlyphOrder(self, glyphOrder):
        pass
        # XXX
        # self.cff[self.cff.fontNames[0]].setGlyphOrder(glyphOrder)

    def toXML(self, writer, otFont):
        self.cff.toXML(writer)

    def fromXML(self, name, attrs, content, otFont):
        if not hasattr(self, "cff"):
            self.cff = cffLib.CFFFontSet()
        self.cff.fromXML(name, attrs, content, otFont)
<|endoftext|>from fontTools.misc import sstruct
from fontTools.misc.textTools import safeEval
from . import DefaultTable
import array
import sys

Gloc_header = """
    >        # big endian
    version: 16.16F    # Table version
    flags:        H    # bit 0: 1=long format, 0=short format
                       # bit 1: 1=attribute names, 0=no names
    numAttribs:   H    # NUmber of attributes
"""


class table_G__l_o_c(DefaultTable.DefaultTable):
    """Graphite Index to Glyph Atttributes table

    See also https://graphite.sil.org/graphite_techAbout#graphite-font-tables
    """

    dependencies = ["Glat"]

    def __init__(self, tag=None):
        DefaultTable.DefaultTable.__init__(self, tag)
        self.attribIds = None
        self.numAttribs = 0

    def decompile(self, data, ttFont):
        _, data = sstruct.unpack2(Gloc_header, data, self)
        flags = self.flags
        del self.flags
        self.locations = array.array("I" if flags & 1 else "H")
        self.locations.frombytes(data[: len(data) - self.numAttribs * (flags & 2)])
        if sys.byteorder != "big":
            self.locations.byteswap()
        self.attribIds = array.array("H")
        if flags & 2:
            self.attribIds.frombytes(data[-self.numAttribs * 2 :])
            if sys.byteorder != "big":
                self.attribIds.byteswap()

    def compile(self, ttFont):
        data = sstruct.pack(
            Gloc_header,
            dict(
                version=1.0,
                flags=(bool(self.attribIds) << 1) + (self.locations.typecode == "I"),
                numAttribs=self.numAttribs,
            ),
        )
        if sys.byteorder != "big":
            self.locations.byteswap()
        data += self.locations.tobytes()
        if sys.byteorder != "big":
            self.locations.byteswap()
        if self.attribIds:
            if sys.byteorder != "big":
                self.attribIds.byteswap()
            data += self.attribIds.tobytes()
            if sys.byteorder != "big":
                self.attribIds.byteswap()
        return data

    def set(self, locations):
        long_format = max(locations) >= 65536
        self.locations = array.array("I" if long_format else "H", locations)

    def toXML(self, writer, ttFont):
        writer.simpletag("attributes", number=self.numAttribs)
        writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        if name == "attributes":
            self.numAttribs = int(safeEval(attrs["number"]))

    def __getitem__(self, index):
        return self.locations[index]

    def __len__(self):
        return len(self.locations)

    def __iter__(self):
        return iter(self.locations)
<|endoftext|># DON'T EDIT! This file is generated by MetaTools/buildTableList.py.
def _moduleFinderHint():
    """Dummy function to let modulefinder know what tables may be
    dynamically imported. Generated by MetaTools/buildTableList.py.

        >>> _moduleFinderHint()
    """
    from . import B_A_S_E_
    from . import C_B_D_T_
    from . import C_B_L_C_
    from . import C_F_F_
    from . import C_F_F__2
    from . import C_O_L_R_
    from . import C_P_A_L_
    from . import D_S_I_G_
    from . import D__e_b_g
    from . import E_B_D_T_
    from . import E_B_L_C_
    from . import E_B_S_C_
    from . import F_F_T_M_
    from . import F__e_a_t
    from . import G_D_E_F_
    from . import G_P_O_S_
    from . import G_S_U_B_
    from . import G_V_A_R_
    from . import G__l_a_t
    from . import G__l_o_c
    from . import H_V_A_R_
    from . import I_F_T_
    from . import I_F_T_X_
    from . import J_S_T_F_
    from . import L_T_S_H_
    from . import M_A_T_H_
    from . import M_V_A_R_
    from . import O_S_2f_2
    from . import S_T_A_T_
    from . import S_V_G_
    from . import S__i_l_f
    from . import S__i_l_l
    from . import T_S_I_B_
    from . import T_S_I_C_
    from . import T_S_I_D_
    from . import T_S_I_J_
    from . import T_S_I_P_
    from . import T_S_I_S_
    from . import T_S_I_V_
    from . import T_S_I__0
    from . import T_S_I__1
    from . import T_S_I__2
    from . import T_S_I__3
    from . import T_S_I__5
    from . import T_T_F_A_
    from . import V_A_R_C_
    from . import V_D_M_X_
    from . import V_O_R_G_
    from . import V_V_A_R_
    from . import _a_n_k_r
    from . import _a_v_a_r
    from . import _b_d_a_t
    from . import _b_g_c_l
    from . import _b_h_e_d
    from . import _b_l_o_c
    from . import _b_s_l_n
    from . import _c_i_d_g
    from . import _c_m_a_p
    from . import _c_v_a_r
    from . import _c_v_t
    from . import _f_e_a_t
    from . import _f_p_g_m
    from . import _f_v_a_r
    from . import _g_a_s_p
    from . import _g_c_i_d
    from . import _g_l_y_f
    from . import _g_v_a_r
    from . import _h_d_m_x
    from . import _h_e_a_d
    from . import _h_h_e_a
    from . import _h_m_t_x
    from . import _k_e_r_n
    from . import _l_c_a_r
    from . import _l_o_c_a
    from . import _l_t_a_g
    from . import _m_a_x_p
    from . import _m_e_t_a
    from . import _m_o_r_t
    from . import _m_o_r_x
    from . import _n_a_m_e
    from . import _o_p_b_d
    from . import _p_o_s_t
    from . import _p_r_e_p
    from . import _p_r_o_p
    from . import _s_b_i_x
    from . import _t_r_a_k
    from . import _v_h_e_a
    from . import _v_m_t_x


if __name__ == "__main__":
    import doctest, sys

    sys.exit(doctest.testmod().failed)
<|endoftext|>from fontTools.misc import sstruct
from fontTools.misc.textTools import readHex, safeEval
import struct

sbixGlyphHeaderFormat = """
	>
	originOffsetX: h	# The x-value of the point in the glyph relative to its
						# lower-left corner which corresponds to the origin of
						# the glyph on the screen, that is the point on the
						# baseline at the left edge of the glyph.
	originOffsetY: h	# The y-value of the point in the glyph relative to its
						# lower-left corner which corresponds to the origin of
						# the glyph on the screen, that is the point on the
						# baseline at the left edge of the glyph.
	graphicType:  4s	# e.g. "png "
"""

sbixGlyphHeaderFormatSize = sstruct.calcsize(sbixGlyphHeaderFormat)


class Glyph(object):
    def __init__(
        self,
        glyphName=None,
        referenceGlyphName=None,
        originOffsetX=0,
        originOffsetY=0,
        graphicType=None,
        imageData=None,
        rawdata=None,
        gid=0,
    ):
        self.gid = gid
        self.glyphName = glyphName
        self.referenceGlyphName = referenceGlyphName
        self.originOffsetX = originOffsetX
        self.originOffsetY = originOffsetY
        self.rawdata = rawdata
        self.graphicType = graphicType
        self.imageData = imageData

        # fix self.graphicType if it is null terminated or too short
        if self.graphicType is not None:
            if self.graphicType[-1] == "\0":
                self.graphicType = self.graphicType[:-1]
            if len(self.graphicType) > 4:
                from fontTools import ttLib

                raise ttLib.TTLibError(
                    "Glyph.graphicType must not be longer than 4 characters."
                )
            elif len(self.graphicType) < 4:
                # pad with spaces
                self.graphicType += "    "[: (4 - len(self.graphicType))]

    def is_reference_type(self):
        """Returns True if this glyph is a reference to another glyph's image data."""
        return self.graphicType == "dupe" or self.graphicType == "flip"

    def decompile(self, ttFont):
        self.glyphName = ttFont.getGlyphName(self.gid)
        if self.rawdata is None:
            from fontTools import ttLib

            raise ttLib.TTLibError("No table data to decompile")
        if len(self.rawdata) > 0:
            if len(self.rawdata) < sbixGlyphHeaderFormatSize:
                from fontTools import ttLib

                # print "Glyph %i header too short: Expected %x, got %x." % (self.gid, sbixGlyphHeaderFormatSize, len(self.rawdata))
                raise ttLib.TTLibError("Glyph header too short.")

            sstruct.unpack(
                sbixGlyphHeaderFormat, self.rawdata[:sbixGlyphHeaderFormatSize], self
            )

            if self.is_reference_type():
                # this glyph is a reference to another glyph's image data
                (gid,) = struct.unpack(">H", self.rawdata[sbixGlyphHeaderFormatSize:])
                self.referenceGlyphName = ttFont.getGlyphName(gid)
            else:
                self.imageData = self.rawdata[sbixGlyphHeaderFormatSize:]
                self.referenceGlyphName = None
        # clean up
        del self.rawdata
        del self.gid

    def compile(self, ttFont):
        if self.glyphName is None:
            from fontTools import ttLib

            raise ttLib.TTLibError("Can't compile Glyph without glyph name")
            # TODO: if ttFont has no maxp, cmap etc., ignore glyph names and compile by index?
            # (needed if you just want to compile the sbix table on its own)
        self.gid = struct.pack(">H", ttFont.getGlyphID(self.glyphName))
        if self.graphicType is None:
            rawdata = b""
        else:
            rawdata = sstruct.pack(sbixGlyphHeaderFormat, self)
            if self.is_reference_type():
                rawdata += struct.pack(">H", ttFont.getGlyphID(self.referenceGlyphName))
            else:
                assert self.imageData is not None
                rawdata += self.imageData
        self.rawdata = rawdata

    def toXML(self, xmlWriter, ttFont):
        if self.graphicType is None:
            # TODO: ignore empty glyphs?
            # a glyph data entry is required for each glyph,
            # but empty ones can be calculated at compile time
            xmlWriter.simpletag("glyph", name=self.glyphName)
            xmlWriter.newline()
            return
        xmlWriter.begintag(
            "glyph",
            graphicType=self.graphicType,
            name=self.glyphName,
            originOffsetX=self.originOffsetX,
            originOffsetY=self.originOffsetY,
        )
        xmlWriter.newline()
        if self.is_reference_type():
            # this glyph is a reference to another glyph id.
            xmlWriter.simpletag("ref", glyphname=self.referenceGlyphName)
        else:
            xmlWriter.begintag("hexdata")
            xmlWriter.newline()
            xmlWriter.dumphex(self.imageData)
            xmlWriter.endtag("hexdata")
        xmlWriter.newline()
        xmlWriter.endtag("glyph")
        xmlWriter.newline()

    def fromXML(self, name, attrs, content, ttFont):
        if name == "ref":
            # this glyph i.e. a reference to another glyph's image data.
            # in this case imageData contains the glyph id of the reference glyph
            # get glyph id from glyphname
            glyphname = safeEval("'''" + attrs["glyphname"] + "'''")
            self.imageData = struct.pack(">H", ttFont.getGlyphID(glyphname))
            self.referenceGlyphName = glyphname
        elif name == "hexdata":
            self.imageData = readHex(content)
        else:
            from fontTools import ttLib

            raise ttLib.TTLibError("can't handle '%s' element" % name)
<|endoftext|>"""TSI{0,1,2,3,5} are private tables used by Microsoft Visual TrueType (VTT)
tool to store its hinting source data.

TSI0 is the index table containing the lengths and offsets for the glyph
programs and 'extra' programs ('fpgm', 'prep', and 'cvt') that are contained
in the TSI1 table.

See also https://learn.microsoft.com/en-us/typography/tools/vtt/tsi-tables
"""

import logging
import struct

from . import DefaultTable

log = logging.getLogger(__name__)

tsi0Format = ">HHL"


def fixlongs(glyphID, textLength, textOffset):
    return int(glyphID), int(textLength), textOffset


class table_T_S_I__0(DefaultTable.DefaultTable):
    dependencies = ["TSI1"]

    def decompile(self, data, ttFont):
        numGlyphs = ttFont["maxp"].numGlyphs
        indices = []
        size = struct.calcsize(tsi0Format)
        numEntries = len(data) // size
        if numEntries != numGlyphs + 5:
            diff = numEntries - numGlyphs - 5
            log.warning(
                "Number of glyphPrograms differs from the number of glyphs in the font "
                f"by {abs(diff)} ({numEntries - 5} programs vs. {numGlyphs} glyphs)."
            )
        for _ in range(numEntries):
            glyphID, textLength, textOffset = fixlongs(
                *struct.unpack(tsi0Format, data[:size])
            )
            indices.append((glyphID, textLength, textOffset))
            data = data[size:]
        assert len(data) == 0
        assert indices[-5] == (0xFFFE, 0, 0xABFC1F34), "bad magic number"
        self.indices = indices[:-5]
        self.extra_indices = indices[-4:]

    def compile(self, ttFont):
        if not hasattr(self, "indices"):
            # We have no corresponding table (TSI1 or TSI3); let's return
            # no data, which effectively means "ignore us".
            return b""
        data = b""
        for index, textLength, textOffset in self.indices:
            data = data + struct.pack(tsi0Format, index, textLength, textOffset)
        data = data + struct.pack(tsi0Format, 0xFFFE, 0, 0xABFC1F34)
        for index, textLength, textOffset in self.extra_indices:
            data = data + struct.pack(tsi0Format, index, textLength, textOffset)
        return data

    def set(self, indices, extra_indices):
        # gets called by 'TSI1' or 'TSI3'
        self.indices = indices
        self.extra_indices = extra_indices

    def toXML(self, writer, ttFont):
        writer.comment("This table will be calculated by the compiler")
        writer.newline()
<|endoftext|>from __future__ import annotations

from typing import Any

from fontTools.misc import sstruct
from fontTools.misc.textTools import bytesjoin, safeEval
from fontTools.misc.xmlWriter import XMLWriter
from fontTools.ttLib import TTFont, TTLibError
from fontTools.ttLib.tables import DefaultTable
from fontTools.ttLib.tables.DefaultTable import XMLAttrs, XMLContent
from fontTools.ttLib.tables.E_B_L_C_ import sbitLineMetricsFormat, SbitLineMetrics

ebscHeaderFormat = """
    > # big endian
    version:  16.16F
    numSizes: I
"""

bitmapScaleTableFormatPart2 = """
    > # big endian
    ppemX:           B
    ppemY:           B
    substitutePpemX: B
    substitutePpemY: B
"""


class table_E_B_S_C_(DefaultTable.DefaultTable):
    """
    Embedded Bitmap Scaling Table

    The ``EBSC`` table provides a mechanism for describing embedded bitmaps which are created by
    scaling other embedded bitmaps. While this is the sort of thing that outline font technologies
    were invented to avoid, there are cases (small sizes of Kanji, for example) where scaling a bitmap
    produces more legible glyphs than scan-converting an outline. For this reason, the EBSC table
    allows a font to define a bitmap strike as a scaled version of another strike.

    The ``EBSC`` table is used together with the ``EBDT`` table, which provides embedded monochrome or
    grayscale bitmap data, and the ``EBLC`` table, which provides embedded bitmap locators.

    See also https://learn.microsoft.com/en-us/typography/opentype/spec/ebsc
    """

    dependencies = ["EBDT", "EBLC"]

    version: float
    numSizes: int
    bitmapScaleTables: list[BitmapScaleTable]

    def decompile(self, data: bytes, ttFont: TTFont):
        _, data = sstruct.unpack2(ebscHeaderFormat, data, self)
        self.bitmapScaleTables = []
        for _ in range(self.numSizes):
            bitmapScaleTable = BitmapScaleTable()
            data = bitmapScaleTable.decompile(data)
            self.bitmapScaleTables.append(bitmapScaleTable)

    def compile(self, ttFont: TTFont) -> bytes:
        self.numSizes = len(self.bitmapScaleTables)
        dataList = [sstruct.pack(ebscHeaderFormat, self)]
        for bitmapScaleTable in self.bitmapScaleTables:
            dataList.append(bitmapScaleTable.compile())
        return bytesjoin(dataList)

    def toXML(self, writer: XMLWriter, ttFont: TTFont, **kwargs: Any):
        writer.simpletag(
            "header",
            [("version", self.version), ("numSizes", len(self.bitmapScaleTables))],
        )
        writer.newline()
        for index, bitmapScaleTable in enumerate(self.bitmapScaleTables):
            bitmapScaleTable.toXML(index, writer, ttFont)

    def fromXML(
        self, name: str, attrs: dict[str, str], content: XMLContent, ttFont: TTFont
    ):
        if not hasattr(self, "bitmapScaleTables"):
            self.bitmapScaleTables = []
        if name == "header":
            self.version = safeEval(attrs["version"])
            self.numSizes = safeEval(attrs["numSizes"])
        elif name == "bitmapScaleTable":
            bitmapScaleTable = BitmapScaleTable()
            bitmapScaleTable.fromXML(content, ttFont)
            self.bitmapScaleTables.append(bitmapScaleTable)


class BitmapScaleTable:
    hori: SbitLineMetrics
    vert: SbitLineMetrics
    ppemX: int
    ppemY: int
    substitutePpemX: int
    substitutePpemY: int

    def __init__(
        self,
        hori: SbitLineMetrics | None = None,
        vert: SbitLineMetrics | None = None,
        ppemX: int = 0,
        ppemY: int = 0,
        substitutePpemX: int = 0,
        substitutePpemY: int = 0,
    ):
        self.hori = hori if hori is not None else SbitLineMetrics()
        self.vert = vert if vert is not None else SbitLineMetrics()
        self.ppemX = ppemX
        self.ppemY = ppemY
        self.substitutePpemX = substitutePpemX
        self.substitutePpemY = substitutePpemY

    def decompile(self, data: bytes) -> bytes:
        dummy, data = sstruct.unpack2(sbitLineMetricsFormat, data, self.hori)
        dummy, data = sstruct.unpack2(sbitLineMetricsFormat, data, self.vert)
        dummy, data = sstruct.unpack2(bitmapScaleTableFormatPart2, data, self)
        return data

    def compile(self) -> bytes:
        dataList = [
            sstruct.pack(sbitLineMetricsFormat, self.hori),
            sstruct.pack(sbitLineMetricsFormat, self.vert),
            sstruct.pack(bitmapScaleTableFormatPart2, self),
        ]
        return bytesjoin(dataList)

    def toXML(self, index: int, writer: XMLWriter, ttFont: TTFont):
        writer.begintag("bitmapScaleTable", [("index", index)])
        writer.newline()
        self.hori.toXML("hori", writer, ttFont)
        self.vert.toXML("vert", writer, ttFont)
        for name in sstruct.getformat(bitmapScaleTableFormatPart2)[1]:
            writer.simpletag(name, value=getattr(self, name))
            writer.newline()
        writer.endtag("bitmapScaleTable")
        writer.newline()

    def fromXML(self, content: XMLContent, ttFont: TTFont):
        dataNames = set(sstruct.getformat(bitmapScaleTableFormatPart2)[1])
        for element in content:
            if not isinstance(element, tuple):
                continue
            name, attrs, content = element
            if name == "sbitLineMetrics":
                direction = attrs["direction"]
                if direction not in ("hori", "vert"):
                    raise TTLibError(
                        f"invalid sbitLineMetrics direction: {direction!r}"
                    )
                metricObj = SbitLineMetrics()
                metricObj.fromXML(name, attrs, content, ttFont)
                setattr(self, direction, metricObj)
            elif name in dataNames:
                setattr(self, name, safeEval(attrs["value"]))
<|endoftext|>from fontTools.misc.textTools import safeEval
from . import DefaultTable
import struct

GASP_SYMMETRIC_GRIDFIT = 0x0004
GASP_SYMMETRIC_SMOOTHING = 0x0008
GASP_DOGRAY = 0x0002
GASP_GRIDFIT = 0x0001


class table__g_a_s_p(DefaultTable.DefaultTable):
    """Grid-fitting and Scan-conversion Procedure table

    The ``gasp`` table defines the preferred rasterization settings for
    the font when rendered on monochrome and greyscale output devices.

    See also https://learn.microsoft.com/en-us/typography/opentype/spec/gasp
    """

    def decompile(self, data, ttFont):
        self.version, numRanges = struct.unpack(">HH", data[:4])
        assert 0 <= self.version <= 1, "unknown 'gasp' format: %s" % self.version
        data = data[4:]
        self.gaspRange = {}
        for i in range(numRanges):
            rangeMaxPPEM, rangeGaspBehavior = struct.unpack(">HH", data[:4])
            self.gaspRange[int(rangeMaxPPEM)] = int(rangeGaspBehavior)
            data = data[4:]
        assert not data, "too much data"

    def compile(self, ttFont):
        version = 0  # ignore self.version
        numRanges = len(self.gaspRange)
        data = b""
        items = sorted(self.gaspRange.items())
        for rangeMaxPPEM, rangeGaspBehavior in items:
            data = data + struct.pack(">HH", rangeMaxPPEM, rangeGaspBehavior)
            if rangeGaspBehavior & ~(GASP_GRIDFIT | GASP_DOGRAY):
                version = 1
        data = struct.pack(">HH", version, numRanges) + data
        return data

    def toXML(self, writer, ttFont):
        items = sorted(self.gaspRange.items())
        for rangeMaxPPEM, rangeGaspBehavior in items:
            writer.simpletag(
                "gaspRange",
                [
                    ("rangeMaxPPEM", rangeMaxPPEM),
                    ("rangeGaspBehavior", rangeGaspBehavior),
                ],
            )
            writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        if name != "gaspRange":
            return
        if not hasattr(self, "gaspRange"):
            self.gaspRange = {}
        self.gaspRange[safeEval(attrs["rangeMaxPPEM"])] = safeEval(
            attrs["rangeGaspBehavior"]
        )
<|endoftext|>from fontTools.misc.textTools import bytesjoin, safeEval, readHex
from fontTools.misc.encodingTools import getEncoding
from fontTools.ttLib import getSearchRange, TTLibError
from fontTools.unicode import Unicode
from . import DefaultTable
import sys
import struct
import array
import logging

log = logging.getLogger(__name__)


def _make_map(font, chars, gids):
    assert len(chars) == len(gids)
    glyphNames = font.getGlyphNameMany(gids)
    cmap = {}
    for char, gid, name in zip(chars, gids, glyphNames):
        if gid == 0:
            continue
        cmap[char] = name
    return cmap


class table__c_m_a_p(DefaultTable.DefaultTable):
    """Character to Glyph Index Mapping Table

    This class represents the `cmap <https://docs.microsoft.com/en-us/typography/opentype/spec/cmap>`_
    table, which maps between input characters (in Unicode or other system encodings)
    and glyphs within the font. The ``cmap`` table contains one or more subtables
    which determine the mapping of of characters to glyphs across different platforms
    and encoding systems.

    ``table__c_m_a_p`` objects expose an accessor ``.tables`` which provides access
    to the subtables, although it is normally easier to retrieve individual subtables
    through the utility methods described below. To add new subtables to a font,
    first determine the subtable format (if in doubt use format 4 for glyphs within
    the BMP, format 12 for glyphs outside the BMP, and format 14 for Unicode Variation
    Sequences) construct subtable objects with ``CmapSubtable.newSubtable(format)``,
    and append them to the ``.tables`` list.

    Within a subtable, the mapping of characters to glyphs is provided by the ``.cmap``
    attribute.

    Example::

            cmap4_0_3 = CmapSubtable.newSubtable(4)
            cmap4_0_3.platformID = 0
            cmap4_0_3.platEncID = 3
            cmap4_0_3.language = 0
            cmap4_0_3.cmap = { 0xC1: "Aacute" }

            cmap = newTable("cmap")
            cmap.tableVersion = 0
            cmap.tables = [cmap4_0_3]

    See also https://learn.microsoft.com/en-us/typography/opentype/spec/cmap
    """

    def getcmap(self, platformID, platEncID):
        """Returns the first subtable which matches the given platform and encoding.

        Args:
                platformID (int): The platform ID. Use 0 for Unicode, 1 for Macintosh
                        (deprecated for new fonts), 2 for ISO (deprecated) and 3 for Windows.
                encodingID (int): Encoding ID. Interpretation depends on the platform ID.
                        See the OpenType specification for details.

        Returns:
                An object which is a subclass of :py:class:`CmapSubtable` if a matching
                subtable is found within the font, or ``None`` otherwise.
        """

        for subtable in self.tables:
            if subtable.platformID == platformID and subtable.platEncID == platEncID:
                return subtable
        return None  # not found

    def getBestCmap(
        self,
        cmapPreferences=(
            (3, 10),
            (0, 6),
            (0, 4),
            (3, 1),
            (0, 3),
            (0, 2),
            (0, 1),
            (0, 0),
        ),
    ):
        """Returns the 'best' Unicode cmap dictionary available in the font
        or ``None``, if no Unicode cmap subtable is available.

        By default it will search for the following (platformID, platEncID)
        pairs in order::

                        (3, 10), # Windows Unicode full repertoire
                        (0, 6),  # Unicode full repertoire (format 13 subtable)
                        (0, 4),  # Unicode 2.0 full repertoire
                        (3, 1),  # Windows Unicode BMP
                        (0, 3),  # Unicode 2.0 BMP
                        (0, 2),  # Unicode ISO/IEC 10646
                        (0, 1),  # Unicode 1.1
                        (0, 0)   # Unicode 1.0

        This particular order matches what HarfBuzz uses to choose what
        subtable to use by default. This order prefers the largest-repertoire
        subtable, and among those, prefers the Windows-platform over the
        Unicode-platform as the former has wider support.

        This order can be customized via the ``cmapPreferences`` argument.
        """
        for platformID, platEncID in cmapPreferences:
            cmapSubtable = self.getcmap(platformID, platEncID)
            if cmapSubtable is not None:
                return cmapSubtable.cmap
        return None  # None of the requested cmap subtables were found

    def buildReversed(self):
        """Builds a reverse mapping dictionary

        Iterates over all Unicode cmap tables and returns a dictionary mapping
        glyphs to sets of codepoints, such as::

                {
                        'one': {0x31}
                        'A': {0x41,0x391}
                }

        The values are sets of Unicode codepoints because
        some fonts map different codepoints to the same glyph.
        For example, ``U+0041 LATIN CAPITAL LETTER A`` and ``U+0391
        GREEK CAPITAL LETTER ALPHA`` are sometimes the same glyph.
        """
        result = {}
        for subtable in self.tables:
            if subtable.isUnicode():
                for codepoint, name in subtable.cmap.items():
                    result.setdefault(name, set()).add(codepoint)
        return result

    def buildReversedMin(self):
        result = {}
        for subtable in self.tables:
            if subtable.isUnicode():
                for codepoint, name in subtable.cmap.items():
                    if name in result:
                        result[name] = min(result[name], codepoint)
                    else:
                        result[name] = codepoint
        return result

    def decompile(self, data, ttFont):
        if len(data) < 4:
            raise TTLibError("cmap table is too short")
        tableVersion, numSubTables = struct.unpack(">HH", data[:4])
        self.tableVersion = int(tableVersion)
        self.tables = tables = []
        seenOffsets = {}
        for i in range(numSubTables):
            entry = data[4 + i * 8 : 4 + (i + 1) * 8]
            if len(entry) < 8:
                raise TTLibError("cmap subtable directory is truncated")
            platformID, platEncID, offset = struct.unpack(">HHL", entry)
            platformID, platEncID = int(platformID), int(platEncID)
            # subtableOffset is an unsigned Offset32 pointing somewhere inside
            # the cmap table; a malformed font can put it past the end of the
            # data, so guard the header reads below instead of letting
            # struct.unpack raise struct.error.
            if offset + 4 > len(data):
                raise TTLibError(
                    "cmap subtable offset %d is out of bounds (data length %d)"
                    % (offset, len(data))
                )
            format, length = struct.unpack(">HH", data[offset : offset + 4])
            if format in [8, 10, 12, 13]:
                if offset + 8 > len(data):
                    raise TTLibError("cmap subtable header is truncated")
                format, reserved, length = struct.unpack(
                    ">HHL", data[offset : offset + 8]
                )
            elif format in [14]:
                if offset + 6 > len(data):
                    raise TTLibError("cmap subtable header is truncated")
                format, length = struct.unpack(">HL", data[offset : offset + 6])

            if not length:
                log.error(
                    "cmap subtable is reported as having zero length: platformID %s, "
                    "platEncID %s, format %s offset %s. Skipping table.",
                    platformID,
                    platEncID,
                    format,
                    offset,
                )
                continue
            table = CmapSubtable.newSubtable(format)
            table.platformID = platformID
            table.platEncID = platEncID
            # Note that by default we decompile only the subtable header info;
            # any other data gets decompiled only when an attribute of the
            # subtable is referenced.
            #
            # Make sure the body is actually present and large enough for the
            # fixed header decompileHeader will unpack. Otherwise the slice is
            # short and we hit either an AssertionError (silenced under python
            # -O, leaving a bogus length) or a struct.error deep inside
            # decompileHeader; surface a TTLibError in both cases. Formats
            # without a fixed header (8, 10 and any unrecognised format, which
            # fall back to cmap_format_unknown) have an empty headerFormat and
            # so no minimum.
            minHeaderSize = struct.calcsize(table.headerFormat)
            if length < minHeaderSize:
                raise TTLibError(
                    "cmap subtable header is truncated: length %d < %d"
                    % (length, minHeaderSize)
                )
            if offset + length > len(data):
                raise TTLibError(
                    "cmap subtable is truncated: needs %d bytes, only %d available"
                    % (length, len(data) - offset)
                )
            table.decompileHeader(data[offset : offset + int(length)], ttFont)
            if offset in seenOffsets:
                table.data = None  # Mark as decompiled
                table.cmap = tables[seenOffsets[offset]].cmap
            else:
                seenOffsets[offset] = i
            tables.append(table)
        if ttFont.lazy is False:  # Be lazy for None and True
            self.ensureDecompiled()

    def ensureDecompiled(self, recurse=False):
        # The recurse argument is unused, but part of the signature of
        # ensureDecompiled across the library.
        for st in self.tables:
            st.ensureDecompiled()

    def compile(self, ttFont):
        self.tables.sort()  # sort according to the spec; see CmapSubtable.__lt__()
        keys = [(t.platformID, t.platEncID, t.language) for t in self.tables]
        seen = set()
        duplicates = {k for k in keys if k in seen or seen.add(k)}
        if duplicates:
            raise ValueError(
                "cmap subtables have duplicate (platformID, platEncID, language) "
                f"entries, which the OpenType spec does not allow: {duplicates}"
            )
        numSubTables = len(self.tables)
        totalOffset = 4 + 8 * numSubTables
        data = struct.pack(">HH", self.tableVersion, numSubTables)
        tableData = b""
        seen = (
            {}
        )  # Some tables are the same object reference. Don't compile them twice.
        done = (
            {}
        )  # Some tables are different objects, but compile to the same data chunk
        for table in self.tables:
            offset = seen.get(id(table.cmap))
            if offset is None:
                chunk = table.compile(ttFont)
                offset = done.get(chunk)
                if offset is None:
                    offset = seen[id(table.cmap)] = done[chunk] = totalOffset + len(
                        tableData
                    )
                    tableData = tableData + chunk
            data = data + struct.pack(">HHL", table.platformID, table.platEncID, offset)
        return data + tableData

    def toXML(self, writer, ttFont):
        writer.simpletag("tableVersion", version=self.tableVersion)
        writer.newline()
        for table in self.tables:
            table.toXML(writer, ttFont)

    def fromXML(self, name, attrs, content, ttFont):
        if name == "tableVersion":
            self.tableVersion = safeEval(attrs["version"])
            return
        if name[:12] != "cmap_format_":
            return
        if not hasattr(self, "tables"):
            self.tables = []
        format = safeEval(name[12:])
        table = CmapSubtable.newSubtable(format)
        table.platformID = safeEval(attrs["platformID"])
        table.platEncID = safeEval(attrs["platEncID"])
        table.fromXML(name, attrs, content, ttFont)
        self.tables.append(table)


class CmapSubtable(object):
    """Base class for all cmap subtable formats.

    Subclasses which handle the individual subtable formats are named
    ``cmap_format_0``, ``cmap_format_2`` etc. Use :py:meth:`getSubtableClass`
    to retrieve the concrete subclass, or :py:meth:`newSubtable` to get a
    new subtable object for a given format.

    The object exposes a ``.cmap`` attribute, which contains a dictionary mapping
    character codepoints to glyph names.
    """

    headerFormat = ">HHH"

    @staticmethod
    def getSubtableClass(format):
        """Return the subtable class for a format."""
        return cmap_classes.get(format, cmap_format_unknown)

    @staticmethod
    def newSubtable(format):
        """Return a new instance of a subtable for the given format
        ."""
        subtableClass = CmapSubtable.getSubtableClass(format)
        return subtableClass(format)

    def __init__(self, format):
        self.format = format
        self.data = None
        self.ttFont = None
        self.platformID = None  #: The platform ID of this subtable
        self.platEncID = None  #: The encoding ID of this subtable (interpretation depends on ``platformID``)
        self.language = (
            None  #: The language ID of this subtable (Macintosh platform only)
        )

    def ensureDecompiled(self, recurse=False):
        # The recurse argument is unused, but part of the signature of
        # ensureDecompiled across the library.
        if self.data is None:
            return
        self.decompile(None, None)  # use saved data.
        self.data = None  # Once this table has been decompiled, make sure we don't
        # just return the original data. Also avoids recursion when
        # called with an attribute that the cmap subtable doesn't have.

    def __getattr__(self, attr):
        # allow lazy decompilation of subtables.
        if attr[:2] == "__":  # don't handle requests for member functions like '__lt__'
            raise AttributeError(attr)
        if self.data is None:
            raise AttributeError(attr)
        self.ensureDecompiled()
        return getattr(self, attr)

    def decompileHeader(self, data, ttFont):
        headerSize = struct.calcsize(self.headerFormat)
        format, length, language = struct.unpack(self.headerFormat, data[:headerSize])
        assert (
            len(data) == length
        ), "corrupt cmap table format %d (data length: %d, header length: %d)" % (
            format,
            len(data),
            length,
        )
        self.format = int(format)
        self.length = int(length)
        self.language = int(language)
        self.data = data[headerSize:]
        self.ttFont = ttFont

    def toXML(self, writer, ttFont):
        writer.begintag(
            self.__class__.__name__,
            [
                ("platformID", self.platformID),
                ("platEncID", self.platEncID),
                ("language", self.language),
            ],
        )
        writer.newline()
        codes = sorted(self.cmap.items())
        self._writeCodes(codes, writer)
        writer.endtag(self.__class__.__name__)
        writer.newline()

    def getEncoding(self, default=None):
        """Returns the Python encoding name for this cmap subtable based on its platformID,
        platEncID, and language.  If encoding for these values is not known, by default
        ``None`` is returned.  That can be overridden by passing a value to the ``default``
        argument.

        Note that if you want to choose a "preferred" cmap subtable, most of the time
        ``self.isUnicode()`` is what you want as that one only returns true for the modern,
        commonly used, Unicode-compatible triplets, not the legacy ones.
        """
        return getEncoding(self.platformID, self.platEncID, self.language, default)

    def isUnicode(self):
        """Returns true if the characters are interpreted as Unicode codepoints."""
        return self.platformID == 0 or (
            self.platformID == 3 and self.platEncID in [0, 1, 10]
        )

    def isSymbol(self):
        """Returns true if the subtable is for the Symbol encoding (3,0)"""
        return self.platformID == 3 and self.platEncID == 0

    def _writeCodes(self, codes, writer):
        isUnicode = self.isUnicode()
        for code, name in codes:
            writer.simpletag("map", code=hex(code), name=name)
            if isUnicode:
                writer.comment(Unicode[code])
            writer.newline()

    def __lt__(self, other):
        if not isinstance(other, CmapSubtable):
            return NotImplemented

        # implemented so that list.sort() sorts according to the spec.
        selfTuple = (
            getattr(self, "platformID", None),
            getattr(self, "platEncID", None),
            getattr(self, "language", None),
        )
        otherTuple = (
            getattr(other, "platformID", None),
            getattr(other, "platEncID", None),
            getattr(other, "language", None),
        )
        return selfTuple < otherTuple


class cmap_format_0(CmapSubtable):
    def decompile(self, data, ttFont):
        # we usually get here indirectly from the subtable __getattr__ function, in which case both args must be None.
        # If not, someone is calling the subtable decompile() directly, and must provide both args.
        if data is not None and ttFont is not None:
            self.decompileHeader(data, ttFont)
        else:
            assert (
                data is None and ttFont is None
            ), "Need both data and ttFont arguments"
        data = (
            self.data
        )  # decompileHeader assigns the data after the header to self.data
        assert 262 == self.length, "Format 0 cmap subtable not 262 bytes"
        gids = array.array("B")
        gids.frombytes(self.data)
        charCodes = range(len(gids))
        self.cmap = _make_map(self.ttFont, charCodes, gids)

    def compile(self, ttFont):
        if self.data:
            return struct.pack(">HHH", 0, 262, self.language) + self.data

        cmap = self.cmap
        assert set(cmap.keys()).issubset(range(256))
        getGlyphID = ttFont.getGlyphID
        valueList = [getGlyphID(cmap[i]) if i in cmap else 0 for i in range(256)]

        gids = array.array("B", valueList)
        data = struct.pack(">HHH", 0, 262, self.language) + gids.tobytes()
        assert len(data) == 262
        return data

    def fromXML(self, name, attrs, content, ttFont):
        self.language = safeEval(attrs["language"])
        if not hasattr(self, "cmap"):
            self.cmap = {}
        cmap = self.cmap
        for element in content:
            if not isinstance(element, tuple):
                continue
            name, attrs, content = element
            if name != "map":
                continue
            cmap[safeEval(attrs["code"])] = attrs["name"]


subHeaderFormat = ">HHhH"


class SubHeader(object):
    def __init__(self):
        self.firstCode = None
        self.entryCount = None
        self.idDelta = None
        self.idRangeOffset = None
        self.glyphIndexArray = []


class cmap_format_2(CmapSubtable):
    def setIDDelta(self, subHeader):
        subHeader.idDelta = 0
        # An empty subheader (an all-notdef two-byte range) has no glyphs to
        # adjust, so leave idDelta at 0 and bail before indexing the array.
        if not subHeader.glyphIndexArray:
            return
        # find the minGI which is not zero.
        minGI = subHeader.glyphIndexArray[0]
        for gid in subHeader.glyphIndexArray:
            if (gid != 0) and (gid < minGI):
                minGI = gid
        # The lowest gid in glyphIndexArray, after subtracting idDelta, must be 1.
        # idDelta is a short, and must be between -32K and 32K. minGI can be between 1 and 64K.
        # We would like to pick an idDelta such that the first glyphArray GID is 1,
        # so that we are more likely to be able to combine glypharray GID subranges.
        # This means that we have a problem when minGI is > 32K
        # Since the final gi is reconstructed from the glyphArray GID by:
        #    (short)finalGID = (gid + idDelta) % 0x10000),
        # we can get from a glypharray GID of 1 to a final GID of 65K by subtracting 2, and casting the
        # negative number to an unsigned short.

        if minGI > 1:
            if minGI > 0x7FFF:
                subHeader.idDelta = -(0x10000 - minGI) - 1
            else:
                subHeader.idDelta = minGI - 1
            idDelta = subHeader.idDelta
            for i in range(subHeader.entryCount):
                gid = subHeader.glyphIndexArray[i]
                if gid > 0:
                    subHeader.glyphIndexArray[i] = gid - idDelta

    def decompile(self, data, ttFont):
        # we usually get here indirectly from the subtable __getattr__ function, in which case both args must be None.
        # If not, someone is calling the subtable decompile() directly, and must provide both args.
        if data is not None and ttFont is not None:
            self.decompileHeader(data, ttFont)
        else:
            assert (
                data is None and ttFont is None
            ), "Need both data and ttFont arguments"

        data = (
            self.data
        )  # decompileHeader assigns the data after the header to self.data
        subHeaderKeys = []
        maxSubHeaderindex = 0
        # get the key array, and determine the number of subHeaders.
        allKeys = array.array("H")
        allKeys.frombytes(data[:512])
        data = data[512:]
        if sys.byteorder != "big":
            allKeys.byteswap()
        subHeaderKeys = [key // 8 for key in allKeys]
        maxSubHeaderindex = max(subHeaderKeys)

        # Load subHeaders
        subHeaderList = []
        pos = 0
        for i in range(maxSubHeaderindex + 1):
            subHeader = SubHeader()
            (
                subHeader.firstCode,
                subHeader.entryCount,
                subHeader.idDelta,
                subHeader.idRangeOffset,
            ) = struct.unpack(subHeaderFormat, data[pos : pos + 8])
            pos += 8
            giDataPos = pos + subHeader.idRangeOffset - 2
            giList = array.array("H")
            giList.frombytes(data[giDataPos : giDataPos + subHeader.entryCount * 2])
            if sys.byteorder != "big":
                giList.byteswap()
            subHeader.glyphIndexArray = giList
            subHeaderList.append(subHeader)
        # How this gets processed.
        # Charcodes may be one or two bytes.
        # The first byte of a charcode is mapped through the subHeaderKeys, to select
        # a subHeader. For any subheader but 0, the next byte is then mapped through the
        # selected subheader. If subheader Index 0 is selected, then the byte itself is
        # mapped through the subheader, and there is no second byte.
        # Then assume that the subsequent byte is the first byte of the next charcode,and repeat.
        #
        # Each subheader references a range in the glyphIndexArray whose length is entryCount.
        # The range in glyphIndexArray referenced by a sunheader may overlap with the range in glyphIndexArray
        # referenced by another subheader.
        # The only subheader that will be referenced by more than one first-byte value is the subheader
        # that maps the entire range of glyphID values to glyphIndex 0, e.g notdef:
        # 	 {firstChar 0, EntryCount 0,idDelta 0,idRangeOffset xx}
        # A byte being mapped though a subheader is treated as in index into a mapping of array index to font glyphIndex.
        # A subheader specifies a subrange within (0...256) by the
        # firstChar and EntryCount values. If the byte value is outside the subrange, then the glyphIndex is zero
        # (e.g. glyph not in font).
        # If the byte index is in the subrange, then an offset index is calculated as (byteIndex - firstChar).
        # The index to glyphIndex mapping is a subrange of the glyphIndexArray. You find the start of the subrange by
        # counting idRangeOffset bytes from the idRangeOffset word. The first value in this subrange is the
        # glyphIndex for the index firstChar. The offset index should then be used in this array to get the glyphIndex.
        # Example for Logocut-Medium
        # first byte of charcode = 129; selects subheader 1.
        # subheader 1 = {firstChar 64, EntryCount 108,idDelta 42,idRangeOffset 0252}
        # second byte of charCode = 66
        # the index offset = 66-64 = 2.
        # The subrange of the glyphIndexArray starting at 0x0252 bytes from the idRangeOffset word is:
        # [glyphIndexArray index], [subrange array index] = glyphIndex
        # [256], [0]=1 	from charcode [129, 64]
        # [257], [1]=2  	from charcode [129, 65]
        # [258], [2]=3  	from charcode [129, 66]
        # [259], [3]=4  	from charcode [129, 67]
        # So, the glyphIndex = 3 from the array. Then if idDelta is not zero and the glyph ID is not zero,
        # add it to the glyphID to get the final glyphIndex
        # value. In this case the final glyph index = 3+ 42 -> 45 for the final glyphIndex. Whew!

        self.data = b""
        cmap = {}
        notdefGI = 0
        for firstByte in range(256):
            subHeadindex = subHeaderKeys[firstByte]
            subHeader = subHeaderList[subHeadindex]
            if subHeadindex == 0:
                if (firstByte < subHeader.firstCode) or (
                    firstByte >= subHeader.firstCode + subHeader.entryCount
                ):
                    continue  # gi is notdef.
                else:
                    charCode = firstByte
                    offsetIndex = firstByte - subHeader.firstCode
                    gi = subHeader.glyphIndexArray[offsetIndex]
                    if gi != 0:
                        gi = (gi + subHeader.idDelta) % 0x10000
                    else:
                        continue  # gi is notdef.
                cmap[charCode] = gi
            else:
                if subHeader.entryCount:
                    charCodeOffset = firstByte * 256 + subHeader.firstCode
                    for offsetIndex in range(subHeader.entryCount):
                        charCode = charCodeOffset + offsetIndex
                        gi = subHeader.glyphIndexArray[offsetIndex]
                        if gi != 0:
                            gi = (gi + subHeader.idDelta) % 0x10000
                        else:
                            continue
                        cmap[charCode] = gi
                # If not subHeader.entryCount, then all char codes with this first byte are
                # mapped to .notdef. We can skip this subtable, and leave the glyphs un-encoded, which is the
                # same as mapping it to .notdef.

        gids = list(cmap.values())
        charCodes = list(cmap.keys())
        self.cmap = _make_map(self.ttFont, charCodes, gids)

    def compile(self, ttFont):
        if self.data:
            return (
                struct.pack(">HHH", self.format, self.length, self.language) + self.data
            )
        kEmptyTwoCharCodeRange = -1
        notdefGI = 0

        items = sorted(self.cmap.items())
        charCodes = [item[0] for item in items]
        names = [item[1] for item in items]
        nameMap = ttFont.getReverseGlyphMap()
        try:
            gids = [nameMap[name] for name in names]
        except KeyError:
            nameMap = ttFont.getReverseGlyphMap(rebuild=True)
            try:
                gids = [nameMap[name] for name in names]
            except KeyError:
                # allow virtual GIDs in format 2 tables
                gids = []
                for name in names:
                    try:
                        gid = nameMap[name]
                    except KeyError:
                        try:
                            if name[:3] == "gid":
                                gid = int(name[3:])
                            else:
                                gid = ttFont.getGlyphID(name)
                        except:
                            raise KeyError(name)

                    gids.append(gid)

        # Process the (char code to gid) item list in char code order.
        # By definition, all one byte char codes map to subheader 0.
        # For all the two byte char codes, we assume that the first byte maps maps to the empty subhead (with an entry count of 0,
        # which defines all char codes in its range to map to notdef) unless proven otherwise.
        # Note that since the char code items are processed in char code order, all the char codes with the
        # same first byte are in sequential order.

        subHeaderKeys = [
            kEmptyTwoCharCodeRange for x in range(256)
        ]  # list of indices into subHeaderList.
        subHeaderList = []

        # We force this subheader entry 0 to exist in the subHeaderList in the case where some one comes up
        # with a cmap where all the one byte char codes map to notdef,
        # with the result that the subhead 0 would not get created just by processing the item list.
        # The same is true for an entirely empty (all-notdef) cmap, where there
        # are no char codes to process at all.
        if not charCodes or charCodes[0] > 255:
            subHeader = SubHeader()
            subHeader.firstCode = 0
            subHeader.entryCount = 0
            subHeader.idDelta = 0
            subHeader.idRangeOffset = 0
            subHeaderList.append(subHeader)

        lastFirstByte = -1
        items = zip(charCodes, gids)
        for charCode, gid in items:
            if gid == 0:
                continue
            firstbyte = charCode >> 8
            secondByte = charCode & 0x00FF

            if (
                firstbyte != lastFirstByte
            ):  # Need to update the current subhead, and start a new one.
                if lastFirstByte > -1:
                    # fix GI's and iDelta of current subheader.
                    self.setIDDelta(subHeader)

                    # If it was sunheader 0 for one-byte charCodes, then we need to set the subHeaderKeys value to zero
                    # for the indices matching the char codes.
                    if lastFirstByte == 0:
                        for index in range(subHeader.entryCount):
                            charCode = subHeader.firstCode + index
                            subHeaderKeys[charCode] = 0

                    assert subHeader.entryCount == len(
                        subHeader.glyphIndexArray
                    ), "Error - subhead entry count does not match len of glyphID subrange."
                # init new subheader
                subHeader = SubHeader()
                subHeader.firstCode = secondByte
                subHeader.entryCount = 1
                subHeader.glyphIndexArray.append(gid)
                subHeaderList.append(subHeader)
                subHeaderKeys[firstbyte] = len(subHeaderList) - 1
                lastFirstByte = firstbyte
            else:
                # need to fill in with notdefs all the code points between the last charCode and the current charCode.
                codeDiff = secondByte - (subHeader.firstCode + subHeader.entryCount)
                for i in range(codeDiff):
                    subHeader.glyphIndexArray.append(notdefGI)
                subHeader.glyphIndexArray.append(gid)
                subHeader.entryCount = subHeader.entryCount + codeDiff + 1

        # fix GI's and iDelta of last subheader that we we added to the subheader array.
        self.setIDDelta(subHeader)

        # Now we add a final subheader for the subHeaderKeys which maps to empty two byte charcode ranges.
        subHeader = SubHeader()
        subHeader.firstCode = 0
        subHeader.entryCount = 0
        subHeader.idDelta = 0
        subHeader.idRangeOffset = 2
        subHeaderList.append(subHeader)
        emptySubheadIndex = len(subHeaderList) - 1
        for index in range(256):
            if subHeaderKeys[index] == kEmptyTwoCharCodeRange:
                subHeaderKeys[index] = emptySubheadIndex
        # Since this is the last subheader, the GlyphIndex Array starts two bytes after the start of the
        # idRangeOffset word of this subHeader. We can safely point to the first entry in the GlyphIndexArray,
        # since the first subrange of the GlyphIndexArray is for subHeader 0, which always starts with
        # charcode 0 and GID 0.

        idRangeOffset = (
            len(subHeaderList) - 1
        ) * 8 + 2  # offset to beginning of glyphIDArray from first subheader idRangeOffset.
        subheadRangeLen = (
            len(subHeaderList) - 1
        )  # skip last special empty-set subheader; we've already hardocodes its idRangeOffset to 2.
        for index in range(subheadRangeLen):
            subHeader = subHeaderList[index]
            subHeader.idRangeOffset = 0
            for j in range(index):
                prevSubhead = subHeaderList[j]
                if (
                    prevSubhead.glyphIndexArray == subHeader.glyphIndexArray
                ):  # use the glyphIndexArray subarray
                    subHeader.idRangeOffset = (
                        prevSubhead.idRangeOffset - (index - j) * 8
                    )
                    subHeader.glyphIndexArray = []
                    break
            if subHeader.idRangeOffset == 0:  # didn't find one.
                subHeader.idRangeOffset = idRangeOffset
                idRangeOffset = (
                    idRangeOffset - 8
                ) + subHeader.entryCount * 2  # one less subheader, one more subArray.
            else:
                idRangeOffset = idRangeOffset - 8  # one less subheader

        # Now we can write out the data!
        length = (
            6 + 512 + 8 * len(subHeaderList)
        )  # header, 256 subHeaderKeys, and subheader array.
        for subhead in subHeaderList[:-1]:
            length = (
                length + len(subhead.glyphIndexArray) * 2
            )  # We can't use subhead.entryCount, as some of the subhead may share subArrays.
        dataList = [struct.pack(">HHH", 2, length, self.language)]
        for index in subHeaderKeys:
            dataList.append(struct.pack(">H", index * 8))
        for subhead in subHeaderList:
            dataList.append(
                struct.pack(
                    subHeaderFormat,
                    subhead.firstCode,
                    subhead.entryCount,
                    subhead.idDelta,
                    subhead.idRangeOffset,
                )
            )
        for subhead in subHeaderList[:-1]:
            for gi in subhead.glyphIndexArray:
                dataList.append(struct.pack(">H", gi))
        data = bytesjoin(dataList)
        assert len(data) == length, (
            "Error: cmap format 2 is not same length as calculated! actual: "
            + str(len(data))
            + " calc : "
            + str(length)
        )
        return data

    def fromXML(self, name, attrs, content, ttFont):
        self.language = safeEval(attrs["language"])
        if not hasattr(self, "cmap"):
            self.cmap = {}
        cmap = self.cmap

        for element in content:
            if not isinstance(element, tuple):
                continue
            name, attrs, content = element
            if name != "map":
                continue
            cmap[safeEval(attrs["code"])] = attrs["name"]


cmap_format_4_format = ">7H"

# uint16  endCode[segCount]          # Ending character code for each segment, last = 0xFFFF.
# uint16  reservedPad                # This value should be zero
# uint16  startCode[segCount]        # Starting character code for each segment
# uint16  idDelta[segCount]          # Delta for all character codes in segment
# uint16  idRangeOffset[segCount]    # Offset in bytes to glyph indexArray, or 0
# uint16  glyphIndexArray[variable]  # Glyph index array


def splitRange(startCode, endCode, cmap):
    # Try to split a range of character codes into subranges with consecutive
    # glyph IDs in such a way that the cmap4 subtable can be stored "most"
    # efficiently. I can't prove I've got the optimal solution, but it seems
    # to do well with the fonts I tested: none became bigger, many became smaller.
    if startCode == endCode:
        return [], [endCode]

    lastID = cmap[startCode]
    lastCode = startCode
    inOrder = None
    orderedBegin = None
    subRanges = []

    # Gather subranges in which the glyph IDs are consecutive.
    for code in range(startCode + 1, endCode + 1):
        glyphID = cmap[code]

        if glyphID - 1 == lastID:
            if inOrder is None or not inOrder:
                inOrder = 1
                orderedBegin = lastCode
        else:
            if inOrder:
                inOrder = 0
                subRanges.append((orderedBegin, lastCode))
                orderedBegin = None

        lastID = glyphID
        lastCode = code

    if inOrder:
        subRanges.append((orderedBegin, lastCode))
    assert lastCode == endCode

    # Now filter out those new subranges that would only make the data bigger.
    # A new segment cost 8 bytes, not using a new segment costs 2 bytes per
    # character.
    newRanges = []
    for b, e in subRanges:
        if b == startCode and e == endCode:
            break  # the whole range, we're fine
        if b == startCode or e == endCode:
            threshold = 4  # split costs one more segment
        else:
            threshold = 8  # split costs two more segments
        if (e - b + 1) > threshold:
            newRanges.append((b, e))
    subRanges = newRanges

    if not subRanges:
        return [], [endCode]

    if subRanges[0][0] != startCode:
        subRanges.insert(0, (startCode, subRanges[0][0] - 1))
    if subRanges[-1][1] != endCode:
        subRanges.append((subRanges[-1][1] + 1, endCode))

    # Fill the "holes" in the segments list -- those are the segments in which
    # the glyph IDs are _not_ consecutive.
    i = 1
    while i < len(subRanges):
        if subRanges[i - 1][1] + 1 != subRanges[i][0]:
            subRanges.insert(i, (subRanges[i - 1][1] + 1, subRanges[i][0] - 1))
            i = i + 1
        i = i + 1

    # Transform the ranges into startCode/endCode lists.
    start = []
    end = []
    for b, e in subRanges:
        start.append(b)
        end.append(e)
    start.pop(0)

    assert len(start) + 1 == len(end)
    return start, end


class cmap_format_4(CmapSubtable):
    def decompile(self, data, ttFont):
        # we usually get here indirectly from the subtable __getattr__ function, in which case both args must be None.
        # If not, someone is calling the subtable decompile() directly, and must provide both args.
        if data is not None and ttFont is not None:
            self.decompileHeader(data, ttFont)
        else:
            assert (
                data is None and ttFont is None
            ), "Need both data and ttFont arguments"

        data = (
            self.data
        )  # decompileHeader assigns the data after the header to self.data
        segCountX2, searchRange, entrySelector, rangeShift = struct.unpack(
            ">4H", data[:8]
        )
        data = data[8:]
        segCount = segCountX2 // 2

        allCodes = array.array("H")
        allCodes.frombytes(data)
        self.data = data = None

        if sys.byteorder != "big":
            allCodes.byteswap()

        # divide the data
        endCode = allCodes[:segCount]
        allCodes = allCodes[segCount + 1 :]  # the +1 is skipping the reservedPad field
        startCode = allCodes[:segCount]
        allCodes = allCodes[segCount:]
        idDelta = allCodes[:segCount]
        allCodes = allCodes[segCount:]
        idRangeOffset = allCodes[:segCount]
        glyphIndexArray = allCodes[segCount:]
        lenGIArray = len(glyphIndexArray)

        # build 2-byte character mapping
        charCodes = []
        gids = []
        for i in range(len(startCode) - 1):  # don't do 0xffff!
            start = startCode[i]
            delta = idDelta[i]
            rangeOffset = idRangeOffset[i]
            partial = rangeOffset // 2 - start + i - len(idRangeOffset)

            rangeCharCodes = list(range(startCode[i], endCode[i] + 1))
            charCodes.extend(rangeCharCodes)
            if rangeOffset == 0:
                gids.extend(
                    [(charCode + delta) & 0xFFFF for charCode in rangeCharCodes]
                )
            else:
                for charCode in rangeCharCodes:
                    index = charCode + partial
                    assert index < lenGIArray, (
                        "In format 4 cmap, range (%d), the calculated index (%d) into the glyph index array is not less than the length of the array (%d) !"
                        % (i, index, lenGIArray)
                    )
                    if glyphIndexArray[index] != 0:  # if not missing glyph
                        glyphID = glyphIndexArray[index] + delta
                    else:
                        glyphID = 0  # missing glyph
                    gids.append(glyphID & 0xFFFF)

        self.cmap = _make_map(self.ttFont, charCodes, gids)

    def compile(self, ttFont):
        if self.data:
            return (
                struct.pack(">HHH", self.format, self.length, self.language) + self.data
            )

        charCodes = list(self.cmap.keys())
        if not charCodes:
            startCode = [0xFFFF]
            endCode = [0xFFFF]
        else:
            charCodes.sort()
            names = [self.cmap[code] for code in charCodes]
            nameMap = ttFont.getReverseGlyphMap()
            try:
                gids = [nameMap[name] for name in names]
            except KeyError:
                nameMap = ttFont.getReverseGlyphMap(rebuild=True)
                try:
                    gids = [nameMap[name] for name in names]
                except KeyError:
                    # allow virtual GIDs in format 4 tables
                    gids = []
                    for name in names:
                        try:
                            gid = nameMap[name]
                        except KeyError:
                            try:
                                if name[:3] == "gid":
                                    gid = int(name[3:])
                                else:
                                    gid = ttFont.getGlyphID(name)
                            except:
                                raise KeyError(name)

                        gids.append(gid)
            cmap = {}  # code:glyphID mapping
            for code, gid in zip(charCodes, gids):
                cmap[code] = gid

            # Build startCode and endCode lists.
            # Split the char codes in ranges of consecutive char codes, then split
            # each range in more ranges of consecutive/not consecutive glyph IDs.
            # See splitRange().
            lastCode = charCodes[0]
            endCode = []
            startCode = [lastCode]
            for charCode in charCodes[
                1:
            ]:  # skip the first code, it's the first start code
                if charCode == lastCode + 1:
                    lastCode = charCode
                    continue
                start, end = splitRange(startCode[-1], lastCode, cmap)
                startCode.extend(start)
                endCode.extend(end)
                startCode.append(charCode)
                lastCode = charCode
            start, end = splitRange(startCode[-1], lastCode, cmap)
            startCode.extend(start)
            endCode.extend(end)
            startCode.append(0xFFFF)
            endCode.append(0xFFFF)

        # build up rest of cruft
        idDelta = []
        idRangeOffset = []
        glyphIndexArray = []
        for i in range(len(endCode) - 1):  # skip the closing codes (0xffff)
            indices = []
            for charCode in range(startCode[i], endCode[i] + 1):
                indices.append(cmap[charCode])
            if indices == list(range(indices[0], indices[0] + len(indices))):
                idDelta.append((indices[0] - startCode[i]) % 0x10000)
                idRangeOffset.append(0)
            else:
                idDelta.append(0)
                idRangeOffset.append(2 * (len(endCode) + len(glyphIndexArray) - i))
                glyphIndexArray.extend(indices)
        idDelta.append(1)  # 0xffff + 1 == (tadaa!) 0. So this end code maps to .notdef
        idRangeOffset.append(0)

        # Insane.
        segCount = len(endCode)
        segCountX2 = segCount * 2
        searchRange, entrySelector, rangeShift = getSearchRange(segCount, 2)

        charCodeArray = array.array("H", endCode + [0] + startCode)
        idDeltaArray = array.array("H", idDelta)
        restArray = array.array("H", idRangeOffset + glyphIndexArray)
        if sys.byteorder != "big":
            charCodeArray.byteswap()
        if sys.byteorder != "big":
            idDeltaArray.byteswap()
        if sys.byteorder != "big":
            restArray.byteswap()
        data = charCodeArray.tobytes() + idDeltaArray.tobytes() + restArray.tobytes()

        length = struct.calcsize(cmap_format_4_format) + len(data)
        header = struct.pack(
            cmap_format_4_format,
            self.format,
            length,
            self.language,
            segCountX2,
            searchRange,
            entrySelector,
            rangeShift,
        )
        return header + data

    def fromXML(self, name, attrs, content, ttFont):
        self.language = safeEval(attrs["language"])
        if not hasattr(self, "cmap"):
            self.cmap = {}
        cmap = self.cmap

        for element in content:
            if not isinstance(element, tuple):
                continue
            nameMap, attrsMap, dummyContent = element
            if nameMap != "map":
                assert 0, "Unrecognized keyword in cmap subtable"
            cmap[safeEval(attrsMap["code"])] = attrsMap["name"]


class cmap_format_6(CmapSubtable):
    def decompile(self, data, ttFont):
        # we usually get here indirectly from the subtable __getattr__ function, in which case both args must be None.
        # If not, someone is calling the subtable decompile() directly, and must provide both args.
        if data is not None and ttFont is not None:
            self.decompileHeader(data, ttFont)
        else:
            assert (
                data is None and ttFont is None
            ), "Need both data and ttFont arguments"

        data = (
            self.data
        )  # decompileHeader assigns the data after the header to self.data
        firstCode, entryCount = struct.unpack(">HH", data[:4])
        firstCode = int(firstCode)
        data = data[4:]
        # assert len(data) == 2 * entryCount  # XXX not true in Apple's Helvetica!!!
        gids = array.array("H")
        gids.frombytes(data[: 2 * int(entryCount)])
        if sys.byteorder != "big":
            gids.byteswap()
        self.data = data = None

        charCodes = list(range(firstCode, firstCode + len(gids)))
        self.cmap = _make_map(self.ttFont, charCodes, gids)

    def compile(self, ttFont):
        if self.data:
            return (
                struct.pack(">HHH", self.format, self.length, self.language) + self.data
            )
        cmap = self.cmap
        codes = sorted(cmap.keys())
        if codes:  # yes, there are empty cmap tables.
            codes = list(range(codes[0], codes[-1] + 1))
            firstCode = codes[0]
            valueList = [
                ttFont.getGlyphID(cmap[code]) if code in cmap else 0 for code in codes
            ]
            gids = array.array("H", valueList)
            if sys.byteorder != "big":
                gids.byteswap()
            data = gids.tobytes()
        else:
            data = b""
            firstCode = 0
        header = struct.pack(
            ">HHHHH", 6, len(data) + 10, self.language, firstCode, len(codes)
        )
        return header + data

    def fromXML(self, name, attrs, content, ttFont):
        self.language = safeEval(attrs["language"])
        if not hasattr(self, "cmap"):
            self.cmap = {}
        cmap = self.cmap

        for element in content:
            if not isinstance(element, tuple):
                continue
            name, attrs, content = element
            if name != "map":
                continue
            cmap[safeEval(attrs["code"])] = attrs["name"]


class cmap_format_12_or_13(CmapSubtable):
    headerFormat = ">HHLLL"

    def __init__(self, format):
        self.format = format
        self.reserved = 0
        self.data = None
        self.ttFont = None

    def decompileHeader(self, data, ttFont):
        headerSize = struct.calcsize(self.headerFormat)
        format, reserved, length, language, nGroups = struct.unpack(
            self.headerFormat, data[:headerSize]
        )
        if len(data) != length:
            raise TTLibError(
                "cmap subtable format %d is truncated: length %d, got %d bytes"
                % (self.format, length, len(data))
            )
        expectedLength = headerSize + nGroups * 12
        if length != expectedLength:
            raise TTLibError(
                "cmap subtable format %d has an inconsistent group count: "
                "length %d, but %d groups need %d bytes"
                % (self.format, length, nGroups, expectedLength)
            )
        self.format = format
        self.reserved = reserved
        self.length = length
        self.language = language
        self.nGroups = nGroups
        self.data = data[headerSize:]
        self.ttFont = ttFont

    def decompile(self, data, ttFont):
        # we usually get here indirectly from the subtable __getattr__ function, in which case both args must be None.
        # If not, someone is calling the subtable decompile() directly, and must provide both args.
        if data is not None and ttFont is not None:
            self.decompileHeader(data, ttFont)
        else:
            assert (
                data is None and ttFont is None
            ), "Need both data and ttFont arguments"

        data = (
            self.data
        )  # decompileHeader assigns the data after the header to self.data
        charCodes = []
        gids = []
        pos = 0
        groups = array.array("I", data[: self.nGroups * 12])
        if sys.byteorder != "big":
            groups.byteswap()
        for i in range(self.nGroups):
            startCharCode = groups[i * 3]
            endCharCode = groups[i * 3 + 1]
            glyphID = groups[i * 3 + 2]
            lenGroup = 1 + endCharCode - startCharCode
            charCodes.extend(range(startCharCode, endCharCode + 1))
            gids.extend(self._computeGIDs(glyphID, lenGroup))
        self.data = data = None
        self.cmap = _make_map(self.ttFont, charCodes, gids)

    def compile(self, ttFont):
        if self.data:
            return (
                struct.pack(
                    ">HHLLL",
                    self.format,
                    self.reserved,
                    self.length,
                    self.language,
                    self.nGroups,
                )
                + self.data
            )
        charCodes = list(self.cmap.keys())
        names = list(self.cmap.values())
        nameMap = ttFont.getReverseGlyphMap()
        try:
            gids = [nameMap[name] for name in names]
        except KeyError:
            nameMap = ttFont.getReverseGlyphMap(rebuild=True)
            try:
                gids = [nameMap[name] for name in names]
            except KeyError:
                # allow virtual GIDs in format 12 tables
                gids = []
                for name in names:
                    try:
                        gid = nameMap[name]
                    except KeyError:
                        try:
                            if name[:3] == "gid":
                                gid = int(name[3:])
                            else:
                                gid = ttFont.getGlyphID(name)
                        except:
                            raise KeyError(name)

                    gids.append(gid)

        cmap = {}  # code:glyphID mapping
        for code, gid in zip(charCodes, gids):
            cmap[code] = gid

        charCodes.sort()
        index = 0
        startCharCode = charCodes[0]
        startGlyphID = cmap[startCharCode]
        lastGlyphID = startGlyphID - self._format_step
        lastCharCode = startCharCode - 1
        nGroups = 0
        dataList = []
        maxIndex = len(charCodes)
        for index in range(maxIndex):
            charCode = charCodes[index]
            glyphID = cmap[charCode]
            if not self._IsInSameRun(glyphID, lastGlyphID, charCode, lastCharCode):
                dataList.append(
                    struct.pack(">LLL", startCharCode, lastCharCode, startGlyphID)
                )
                startCharCode = charCode
                startGlyphID = glyphID
                nGroups = nGroups + 1
            lastGlyphID = glyphID
            lastCharCode = charCode
        dataList.append(struct.pack(">LLL", startCharCode, lastCharCode, startGlyphID))
        nGroups = nGroups + 1
        data = bytesjoin(dataList)
        lengthSubtable = len(data) + 16
        assert len(data) == (nGroups * 12) == (lengthSubtable - 16)
        return (
            struct.pack(
                ">HHLLL",
                self.format,
                self.reserved,
                lengthSubtable,
                self.language,
                nGroups,
            )
            + data
        )

    def toXML(self, writer, ttFont):
        writer.begintag(
            self.__class__.__name__,
            [
                ("platformID", self.platformID),
                ("platEncID", self.platEncID),
                ("format", self.format),
                ("reserved", self.reserved),
                ("length", self.length),
                ("language", self.language),
                ("nGroups", self.nGroups),
            ],
        )
        writer.newline()
        codes = sorted(self.cmap.items())
        self._writeCodes(codes, writer)
        writer.endtag(self.__class__.__name__)
        writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        self.format = safeEval(attrs["format"])
        self.reserved = safeEval(attrs["reserved"])
        self.length = safeEval(attrs["length"])
        self.language = safeEval(attrs["language"])
        self.nGroups = safeEval(attrs["nGroups"])
        if not hasattr(self, "cmap"):
            self.cmap = {}
        cmap = self.cmap

        for element in content:
            if not isinstance(element, tuple):
                continue
            name, attrs, content = element
            if name != "map":
                continue
            cmap[safeEval(attrs["code"])] = attrs["name"]


class cmap_format_12(cmap_format_12_or_13):
    _format_step = 1

    def __init__(self, format=12):
        cmap_format_12_or_13.__init__(self, format)

    def _computeGIDs(self, startingGlyph, numberOfGlyphs):
        return range(startingGlyph, startingGlyph + numberOfGlyphs)

    def _IsInSameRun(self, glyphID, lastGlyphID, charCode, lastCharCode):
        return (glyphID == 1 + lastGlyphID) and (charCode == 1 + lastCharCode)


class cmap_format_13(cmap_format_12_or_13):
    _format_step = 0

    def __init__(self, format=13):
        cmap_format_12_or_13.__init__(self, format)

    def _computeGIDs(self, startingGlyph, numberOfGlyphs):
        return [startingGlyph] * numberOfGlyphs

    def _IsInSameRun(self, glyphID, lastGlyphID, charCode, lastCharCode):
        return (glyphID == lastGlyphID) and (charCode == 1 + lastCharCode)


def cvtToUVS(threeByteString):
    data = b"\0" + threeByteString
    (val,) = struct.unpack(">L", data)
    return val


def cvtFromUVS(val):
    assert 0 <= val < 0x1000000
    fourByteString = struct.pack(">L", val)
    return fourByteString[1:]


class cmap_format_14(CmapSubtable):
    headerFormat = ">HLL"

    def decompileHeader(self, data, ttFont):
        headerSize = struct.calcsize(self.headerFormat)
        format, length, numVarSelectorRecords = struct.unpack(
            self.headerFormat, data[:headerSize]
        )
        self.data = data[headerSize:]
        self.length = length
        self.numVarSelectorRecords = numVarSelectorRecords
        self.ttFont = ttFont
        self.language = 0xFF  # has no language.

    def decompile(self, data, ttFont):
        if data is not None and ttFont is not None:
            self.decompileHeader(data, ttFont)
        else:
            assert (
                data is None and ttFont is None
            ), "Need both data and ttFont arguments"
        data = self.data

        self.cmap = (
            {}
        )  # so that clients that expect this to exist in a cmap table won't fail.
        uvsDict = {}
        recOffset = 0
        for n in range(self.numVarSelectorRecords):
            uvs, defOVSOffset, nonDefUVSOffset = struct.unpack(
                ">3sLL", data[recOffset : recOffset + 11]
            )
            recOffset += 11
            varUVS = cvtToUVS(uvs)
            if defOVSOffset:
                startOffset = defOVSOffset - 10
                (numValues,) = struct.unpack(">L", data[startOffset : startOffset + 4])
                startOffset += 4
                for r in range(numValues):
                    uv, addtlCnt = struct.unpack(
                        ">3sB", data[startOffset : startOffset + 4]
                    )
                    startOffset += 4
                    firstBaseUV = cvtToUVS(uv)
                    cnt = addtlCnt + 1
                    baseUVList = list(range(firstBaseUV, firstBaseUV + cnt))
                    glyphList = [None] * cnt
                    localUVList = zip(baseUVList, glyphList)
                    try:
                        uvsDict[varUVS].extend(localUVList)
                    except KeyError:
                        uvsDict[varUVS] = list(localUVList)

            if nonDefUVSOffset:
                startOffset = nonDefUVSOffset - 10
                (numRecs,) = struct.unpack(">L", data[startOffset : startOffset + 4])
                startOffset += 4
                localUVList = []
                for r in range(numRecs):
                    uv, gid = struct.unpack(">3sH", data[startOffset : startOffset + 5])
                    startOffset += 5
                    uv = cvtToUVS(uv)
                    glyphName = self.ttFont.getGlyphName(gid)
                    localUVList.append((uv, glyphName))
                try:
                    uvsDict[varUVS].extend(localUVList)
                except KeyError:
                    uvsDict[varUVS] = localUVList

        self.uvsDict = uvsDict

    def toXML(self, writer, ttFont):
        writer.begintag(
            self.__class__.__name__,
            [
                ("platformID", self.platformID),
                ("platEncID", self.platEncID),
            ],
        )
        writer.newline()
        uvsDict = self.uvsDict
        uvsList = sorted(uvsDict.keys())
        for uvs in uvsList:
            uvList = uvsDict[uvs]
            uvList.sort(key=lambda item: (item[1] is not None, item[0], item[1]))
            for uv, gname in uvList:
                attrs = [("uv", hex(uv)), ("uvs", hex(uvs))]
                if gname is not None:
                    attrs.append(("name", gname))
                writer.simpletag("map", attrs)
                writer.newline()
        writer.endtag(self.__class__.__name__)
        writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        self.language = 0xFF  # provide a value so that CmapSubtable.__lt__() won't fail
        if not hasattr(self, "cmap"):
            self.cmap = (
                {}
            )  # so that clients that expect this to exist in a cmap table won't fail.
        if not hasattr(self, "uvsDict"):
            self.uvsDict = {}
            uvsDict = self.uvsDict

        # For backwards compatibility reasons we accept "None" as an indicator
        # for "default mapping", unless the font actually has a glyph named
        # "None".
        _hasGlyphNamedNone = None

        for element in content:
            if not isinstance(element, tuple):
                continue
            name, attrs, content = element
            if name != "map":
                continue
            uvs = safeEval(attrs["uvs"])
            uv = safeEval(attrs["uv"])
            gname = attrs.get("name")
            if gname == "None":
                if _hasGlyphNamedNone is None:
                    _hasGlyphNamedNone = "None" in ttFont.getGlyphOrder()
                if not _hasGlyphNamedNone:
                    gname = None
            try:
                uvsDict[uvs].append((uv, gname))
            except KeyError:
                uvsDict[uvs] = [(uv, gname)]

    def compile(self, ttFont):
        if self.data:
            return (
                struct.pack(
                    ">HLL", self.format, self.length, self.numVarSelectorRecords
                )
                + self.data
            )

        uvsDict = self.uvsDict
        uvsList = sorted(uvsDict.keys())
        self.numVarSelectorRecords = len(uvsList)
        offset = (
            10 + self.numVarSelectorRecords * 11
        )  # current value is end of VarSelectorRecords block.
        data = []
        varSelectorRecords = []
        for uvs in uvsList:
            entryList = uvsDict[uvs]

            defList = [entry for entry in entryList if entry[1] is None]
            if defList:
                defList = [entry[0] for entry in defList]
                defOVSOffset = offset
                defList.sort()

                lastUV = defList[0]
                cnt = -1
                defRecs = []
                for defEntry in defList:
                    cnt += 1
                    if (lastUV + cnt) != defEntry:
                        rec = struct.pack(">3sB", cvtFromUVS(lastUV), cnt - 1)
                        lastUV = defEntry
                        defRecs.append(rec)
                        cnt = 0

                rec = struct.pack(">3sB", cvtFromUVS(lastUV), cnt)
                defRecs.append(rec)

                numDefRecs = len(defRecs)
                data.append(struct.pack(">L", numDefRecs))
                data.extend(defRecs)
                offset += 4 + numDefRecs * 4
            else:
                defOVSOffset = 0

            ndefList = [entry for entry in entryList if entry[1] is not None]
            if ndefList:
                nonDefUVSOffset = offset
                ndefList.sort()
                numNonDefRecs = len(ndefList)
                data.append(struct.pack(">L", numNonDefRecs))
                offset += 4 + numNonDefRecs * 5

                for uv, gname in ndefList:
                    gid = ttFont.getGlyphID(gname)
                    ndrec = struct.pack(">3sH", cvtFromUVS(uv), gid)
                    data.append(ndrec)
            else:
                nonDefUVSOffset = 0

            vrec = struct.pack(">3sLL", cvtFromUVS(uvs), defOVSOffset, nonDefUVSOffset)
            varSelectorRecords.append(vrec)

        data = bytesjoin(varSelectorRecords) + bytesjoin(data)
        self.length = 10 + len(data)
        headerdata = struct.pack(
            ">HLL", self.format, self.length, self.numVarSelectorRecords
        )

        return headerdata + data


class cmap_format_unknown(CmapSubtable):
    headerFormat = ""  # the body is kept verbatim, nothing is unpacked

    def __init__(self, format):
        CmapSubtable.__init__(self, format)
        # We can't read the subtable, so there are no mappings to offer. The raw
        # data is kept instead, and written back out as-is. fromXML() does the
        # same.
        self.cmap = {}

    def ensureDecompiled(self, recurse=False):
        # There is nothing to decompile, so hold on to the data; the base class
        # would drop it.
        pass

    def __getattr__(self, attr):
        # The base class lazily decompiles and retries, relying on
        # ensureDecompiled() clearing self.data to terminate. We keep the data,
        # so retrying would recurse forever.
        raise AttributeError(attr)

    def toXML(self, writer, ttFont):
        cmapName = self.__class__.__name__[:12] + str(self.format)
        writer.begintag(
            cmapName,
            [
                ("platformID", self.platformID),
                ("platEncID", self.platEncID),
            ],
        )
        writer.newline()
        writer.dumphex(self.data)
        writer.endtag(cmapName)
        writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        self.data = readHex(content)
        self.cmap = {}

    def decompileHeader(self, data, ttFont):
        self.language = 0  # dummy value
        self.data = data

    def decompile(self, data, ttFont):
        # we usually get here indirectly from the subtable __getattr__ function, in which case both args must be None.
        # If not, someone is calling the subtable decompile() directly, and must provide both args.
        if data is not None and ttFont is not None:
            self.decompileHeader(data, ttFont)
        else:
            assert (
                data is None and ttFont is None
            ), "Need both data and ttFont arguments"

    def compile(self, ttFont):
        if self.data:
            return self.data
        else:
            return None


cmap_classes = {
    0: cmap_format_0,
    2: cmap_format_2,
    4: cmap_format_4,
    6: cmap_format_6,
    12: cmap_format_12,
    13: cmap_format_13,
    14: cmap_format_14,
}
<|endoftext|># -*- coding: utf-8 -*-
from __future__ import annotations

from fontTools.misc import sstruct
from fontTools.misc.textTools import (
    bytechr,
    byteord,
    bytesjoin,
    strjoin,
    tobytes,
    tostr,
    safeEval,
)
from fontTools.misc.encodingTools import getEncoding
from fontTools.ttLib import newTable
from fontTools.ttLib.ttVisitor import TTVisitor
from fontTools import ttLib
import fontTools.ttLib.tables.otTables as otTables
from fontTools.ttLib.tables import C_P_A_L_
from . import DefaultTable
import struct
import logging

log = logging.getLogger(__name__)

nameRecordFormat = """
		>	# big endian
		platformID:	H
		platEncID:	H
		langID:		H
		nameID:		H
		length:		H
		offset:		H
"""

nameRecordSize = sstruct.calcsize(nameRecordFormat)


class table__n_a_m_e(DefaultTable.DefaultTable):
    """Naming table

    The ``name`` table is used to store a variety of strings that can be
    associated with user-facing font information. Records in the ``name``
    table can be tagged with language tags to support multilingual naming
    and can support platform-specific character-encoding variants.

    See also https://learn.microsoft.com/en-us/typography/opentype/spec/name
    """

    dependencies = ["ltag"]

    def __init__(self, tag=None):
        super().__init__(tag)
        self.names = []

    def decompile(self, data, ttFont):
        format, n, stringOffset = struct.unpack(b">HHH", data[:6])
        expectedStringOffset = 6 + n * nameRecordSize
        if stringOffset != expectedStringOffset:
            log.error(
                "'name' table stringOffset incorrect. Expected: %s; Actual: %s",
                expectedStringOffset,
                stringOffset,
            )
        stringData = data[stringOffset:]
        data = data[6:]
        self.names: list[NameRecord] = []
        for i in range(n):
            if len(data) < 12:
                log.error("skipping malformed name record #%d", i)
                continue
            name, data = sstruct.unpack2(nameRecordFormat, data, NameRecord())
            name.string = stringData[name.offset : name.offset + name.length]
            if name.offset + name.length > len(stringData):
                log.error("skipping malformed name record #%d", i)
                continue
            assert len(name.string) == name.length
            # if (name.platEncID, name.platformID) in ((0, 0), (1, 3)):
            # 	if len(name.string) % 2:
            # 		print "2-byte string doesn't have even length!"
            # 		print name.__dict__
            del name.offset, name.length
            self.names.append(name)

    def compile(self, ttFont):
        names = self.names
        names.sort()  # sort according to the spec; see NameRecord.__lt__()
        stringData = b""
        format = 0
        n = len(names)
        stringOffset = 6 + n * sstruct.calcsize(nameRecordFormat)
        data = struct.pack(b">HHH", format, n, stringOffset)
        lastoffset = 0
        done = {}  # remember the data so we can reuse the "pointers"
        for name in names:
            string = name.toBytes()
            if string in done:
                name.offset, name.length = done[string]
            else:
                name.offset, name.length = done[string] = len(stringData), len(string)
                stringData = bytesjoin([stringData, string])
            data = data + sstruct.pack(nameRecordFormat, name)
        return data + stringData

    def toXML(self, writer, ttFont):
        for name in self.names:
            name.toXML(writer, ttFont)

    def fromXML(self, name, attrs, content, ttFont):
        if name != "namerecord":
            return  # ignore unknown tags
        name = NameRecord()
        self.names.append(name)
        name.fromXML(name, attrs, content, ttFont)

    def getName(
        self, nameID: int, platformID: int, platEncID: int, langID: int | None = None
    ) -> "NameRecord | None":
        for namerecord in self.names:
            if (
                namerecord.nameID == nameID
                and namerecord.platformID == platformID
                and namerecord.platEncID == platEncID
            ):
                if langID is None or namerecord.langID == langID:
                    return namerecord
        return None  # not found

    def getDebugName(self, nameID: int) -> str | None:
        englishName: str | None = None
        someName: str | None = None
        for name in self.names:
            if name.nameID != nameID:
                continue
            try:
                unistr = name.toUnicode()
            except UnicodeDecodeError:
                continue

            someName = unistr
            if (name.platformID, name.langID) in ((1, 0), (3, 0x409)):
                englishName = unistr
                break
        if englishName:
            return englishName
        elif someName:
            return someName
        else:
            return None

    def getFirstDebugName(self, nameIDs):
        for nameID in nameIDs:
            name = self.getDebugName(nameID)
            if name is not None:
                return name
        return None

    def getBestFamilyName(self):
        # 21 = WWS Family Name
        # 16 = Typographic Family Name
        # 1 = Family Name
        return self.getFirstDebugName((21, 16, 1))

    def getBestSubFamilyName(self):
        # 22 = WWS SubFamily Name
        # 17 = Typographic SubFamily Name
        # 2 = SubFamily Name
        return self.getFirstDebugName((22, 17, 2))

    def getBestFullName(self):
        # 4 = Full Name
        # 6 = PostScript Name
        for nameIDs in ((21, 22), (16, 17), (1, 2), (4,), (6,)):
            if len(nameIDs) == 2:
                name_fam = self.getDebugName(nameIDs[0])
                name_subfam = self.getDebugName(nameIDs[1])
                if None in [name_fam, name_subfam]:
                    continue  # if any is None, skip
                name = f"{name_fam} {name_subfam}"
                if name_subfam.lower() == "regular":
                    name = f"{name_fam}"
                return name
            else:
                name = self.getDebugName(nameIDs[0])
                if name is not None:
                    return name
        return None

    def setName(self, string, nameID, platformID, platEncID, langID):
        """Set the 'string' for the name record identified by 'nameID', 'platformID',
        'platEncID' and 'langID'. If a record with that nameID doesn't exist, create it
        and append to the name table.

        'string' can be of type `str` (`unicode` in PY2) or `bytes`. In the latter case,
        it is assumed to be already encoded with the correct plaform-specific encoding
        identified by the (platformID, platEncID, langID) triplet. A warning is issued
        to prevent unexpected results.
        """
        if not isinstance(string, str):
            if isinstance(string, bytes):
                log.warning(
                    "name string is bytes, ensure it's correctly encoded: %r", string
                )
            else:
                raise TypeError(
                    "expected unicode or bytes, found %s: %r"
                    % (type(string).__name__, string)
                )
        namerecord = self.getName(nameID, platformID, platEncID, langID)
        if namerecord:
            namerecord.string = string
        else:
            self.names.append(makeName(string, nameID, platformID, platEncID, langID))

    def removeNames(self, nameID=None, platformID=None, platEncID=None, langID=None):
        """Remove any name records identified by the given combination of 'nameID',
        'platformID', 'platEncID' and 'langID'.
        """
        args = {
            argName: argValue
            for argName, argValue in (
                ("nameID", nameID),
                ("platformID", platformID),
                ("platEncID", platEncID),
                ("langID", langID),
            )
            if argValue is not None
        }
        if not args:
            # no arguments, nothing to do
            return
        self.names = [
            rec
            for rec in self.names
            if any(
                argValue != getattr(rec, argName) for argName, argValue in args.items()
            )
        ]

    @staticmethod
    def removeUnusedNames(ttFont):
        """Remove any name records which are not in NameID range 0-255 and not utilized
        within the font itself."""
        visitor = NameRecordVisitor()
        visitor.visit(ttFont)
        toDelete = set()
        for record in ttFont["name"].names:
            # Name IDs 26 to 255, inclusive, are reserved for future standard names.
            # https://learn.microsoft.com/en-us/typography/opentype/spec/name#name-ids
            if record.nameID < 256:
                continue
            if record.nameID not in visitor.seen:
                toDelete.add(record.nameID)

        for nameID in toDelete:
            ttFont["name"].removeNames(nameID)
        return toDelete

    def _findUnusedNameID(self, minNameID=256):
        """Finds an unused name id.

        The nameID is assigned in the range between 'minNameID' and 32767 (inclusive),
        following the last nameID in the name table.
        """
        names = self.names
        nameID = 1 + max([n.nameID for n in names] + [minNameID - 1])
        if nameID > 32767:
            raise ValueError("nameID must be less than 32768")
        return nameID

    def findMultilingualName(
        self, names, windows=True, mac=True, minNameID=0, ttFont=None
    ):
        """Return the name ID of an existing multilingual name that
        matches the 'names' dictionary, or None if not found.

        'names' is a dictionary with the name in multiple languages,
        such as {'en': 'Pale', 'de': 'Blaß', 'de-CH': 'Blass'}.
        The keys can be arbitrary IETF BCP 47 language codes;
        the values are Unicode strings.

        If 'windows' is True, the returned name ID is guaranteed
        exist for all requested languages for platformID=3 and
        platEncID=1.
        If 'mac' is True, the returned name ID is guaranteed to exist
        for all requested languages for platformID=1 and platEncID=0.

        The returned name ID will not be less than the 'minNameID'
        argument.
        """
        # Gather the set of requested
        #   (string, platformID, platEncID, langID)
        # tuples
        reqNameSet = set()
        for lang, name in sorted(names.items()):
            if windows:
                windowsName = _makeWindowsName(name, None, lang)
                if windowsName is not None:
                    reqNameSet.add(
                        (
                            windowsName.string,
                            windowsName.platformID,
                            windowsName.platEncID,
                            windowsName.langID,
                        )
                    )
            if mac:
                macName = _makeMacName(name, None, lang, ttFont)
                if macName is not None:
                    reqNameSet.add(
                        (
                            macName.string,
                            macName.platformID,
                            macName.platEncID,
                            macName.langID,
                        )
                    )

        # Collect matching name IDs
        matchingNames = dict()
        for name in self.names:
            try:
                key = (name.toUnicode(), name.platformID, name.platEncID, name.langID)
            except UnicodeDecodeError:
                continue
            if key in reqNameSet and name.nameID >= minNameID:
                nameSet = matchingNames.setdefault(name.nameID, set())
                nameSet.add(key)

        # Return the first name ID that defines all requested strings
        for nameID, nameSet in sorted(matchingNames.items()):
            if nameSet == reqNameSet:
                return nameID

        return None  # not found

    def addMultilingualName(
        self, names, ttFont=None, nameID=None, windows=True, mac=True, minNameID=0
    ):
        """Add a multilingual name, returning its name ID

        'names' is a dictionary with the name in multiple languages,
        such as {'en': 'Pale', 'de': 'Blaß', 'de-CH': 'Blass'}.
        The keys can be arbitrary IETF BCP 47 language codes;
        the values are Unicode strings.

        'ttFont' is the TTFont to which the names are added, or None.
        If present, the font's 'ltag' table can get populated
        to store exotic language codes, which allows encoding
        names that otherwise cannot get encoded at all.

        'nameID' is the name ID to be used, or None to let the library
        find an existing set of name records that match, or pick an
        unused name ID.

        If 'windows' is True, a platformID=3 name record will be added.
        If 'mac' is True, a platformID=1 name record will be added.

        If the 'nameID' argument is None, the created nameID will not
        be less than the 'minNameID' argument.
        """
        if nameID is None:
            # Reuse nameID if possible
            nameID = self.findMultilingualName(
                names, windows=windows, mac=mac, minNameID=minNameID, ttFont=ttFont
            )
            if nameID is not None:
                return nameID
            nameID = self._findUnusedNameID()
        # TODO: Should minimize BCP 47 language codes.
        # https://github.com/fonttools/fonttools/issues/930
        for lang, name in sorted(names.items()):
            if windows:
                windowsName = _makeWindowsName(name, nameID, lang)
                if windowsName is not None:
                    self.names.append(windowsName)
                else:
                    # We cannot not make a Windows name: make sure we add a
                    # Mac name as a fallback. This can happen for exotic
                    # BCP47 language tags that have no Windows language code.
                    mac = True
            if mac:
                macName = _makeMacName(name, nameID, lang, ttFont)
                if macName is not None:
                    self.names.append(macName)
        return nameID

    def addName(self, string, platforms=((1, 0, 0), (3, 1, 0x409)), minNameID=255):
        """Add a new name record containing 'string' for each (platformID, platEncID,
        langID) tuple specified in the 'platforms' list.

        The nameID is assigned in the range between 'minNameID'+1 and 32767 (inclusive),
        following the last nameID in the name table.
        If no 'platforms' are specified, two English name records are added, one for the
        Macintosh (platformID=0), and one for the Windows platform (3).

        The 'string' must be a Unicode string, so it can be encoded with different,
        platform-specific encodings.

        Return the new nameID.
        """
        assert (
            len(platforms) > 0
        ), "'platforms' must contain at least one (platformID, platEncID, langID) tuple"
        if not isinstance(string, str):
            raise TypeError(
                "expected str, found %s: %r" % (type(string).__name__, string)
            )
        nameID = self._findUnusedNameID(minNameID + 1)
        for platformID, platEncID, langID in platforms:
            self.names.append(makeName(string, nameID, platformID, platEncID, langID))
        return nameID


def makeName(string, nameID, platformID, platEncID, langID):
    name = NameRecord()
    name.string, name.nameID, name.platformID, name.platEncID, name.langID = (
        string,
        nameID,
        platformID,
        platEncID,
        langID,
    )
    return name


def _makeWindowsName(name, nameID, language):
    """Create a NameRecord for the Microsoft Windows platform

    'language' is an arbitrary IETF BCP 47 language identifier such
    as 'en', 'de-CH', 'de-AT-1901', or 'fa-Latn'. If Microsoft Windows
    does not support the desired language, the result will be None.
    Future versions of fonttools might return a NameRecord for the
    OpenType 'name' table format 1, but this is not implemented yet.
    """
    langID = _WINDOWS_LANGUAGE_CODES.get(language.lower())
    if langID is not None:
        return makeName(name, nameID, 3, 1, langID)
    else:
        log.warning(
            "cannot add Windows name in language %s "
            "because fonttools does not yet support "
            "name table format 1" % language
        )
        return None


def _makeMacName(name, nameID, language, font=None):
    """Create a NameRecord for Apple platforms

    'language' is an arbitrary IETF BCP 47 language identifier such
    as 'en', 'de-CH', 'de-AT-1901', or 'fa-Latn'. When possible, we
    create a Macintosh NameRecord that is understood by old applications
    (platform ID 1 and an old-style Macintosh language enum). If this
    is not possible, we create a Unicode NameRecord (platform ID 0)
    whose language points to the font’s 'ltag' table. The latter
    can encode any string in any language, but legacy applications
    might not recognize the format (in which case they will ignore
    those names).

    'font' should be the TTFont for which you want to create a name.
    If 'font' is None, we only return NameRecords for legacy Macintosh;
    in that case, the result will be None for names that need to
    be encoded with an 'ltag' table.

    See the section “The language identifier” in Apple’s specification:
    https://developer.apple.com/fonts/TrueType-Reference-Manual/RM06/Chap6name.html
    """
    macLang = _MAC_LANGUAGE_CODES.get(language.lower())
    macScript = _MAC_LANGUAGE_TO_SCRIPT.get(macLang)
    if macLang is not None and macScript is not None:
        encoding = getEncoding(1, macScript, macLang, default="ascii")
        # Check if we can actually encode this name. If we can't,
        # for example because we have no support for the legacy
        # encoding, or because the name string contains Unicode
        # characters that the legacy encoding cannot represent,
        # we fall back to encoding the name in Unicode and put
        # the language tag into the ltag table.
        try:
            _ = tobytes(name, encoding, errors="strict")
            return makeName(name, nameID, 1, macScript, macLang)
        except UnicodeEncodeError:
            pass
    if font is not None:
        ltag = font.tables.get("ltag")
        if ltag is None:
            ltag = font["ltag"] = newTable("ltag")
        # 0 = Unicode; 4 = “Unicode 2.0 or later semantics (non-BMP characters allowed)”
        # “The preferred platform-specific code for Unicode would be 3 or 4.”
        # https://developer.apple.com/fonts/TrueType-Reference-Manual/RM06/Chap6name.html
        return makeName(name, nameID, 0, 4, ltag.addTag(language))
    else:
        log.warning(
            "cannot store language %s into 'ltag' table "
            "without having access to the TTFont object" % language
        )
        return None


class NameRecord(object):
    def getEncoding(self, default="ascii"):
        """Returns the Python encoding name for this name entry based on its platformID,
        platEncID, and langID.  If encoding for these values is not known, by default
        'ascii' is returned.  That can be overriden by passing a value to the default
        argument.
        """
        return getEncoding(self.platformID, self.platEncID, self.langID, default)

    def encodingIsUnicodeCompatible(self):
        return self.getEncoding(None) in ["utf_16_be", "ucs2be", "ascii", "latin1"]

    def __str__(self):
        return self.toStr(errors="backslashreplace")

    def isUnicode(self):
        return self.platformID == 0 or (
            self.platformID == 3 and self.platEncID in [0, 1, 10]
        )

    def toUnicode(self, errors: str = "strict") -> str:
        """
        If self.string is a Unicode string, return it; otherwise try decoding the
        bytes in self.string to a Unicode string using the encoding of this
        entry as returned by self.getEncoding(); Note that  self.getEncoding()
        returns 'ascii' if the encoding is unknown to the library.

        Certain heuristics are performed to recover data from bytes that are
        ill-formed in the chosen encoding, or that otherwise look misencoded
        (mostly around bad UTF-16BE encoded bytes, or bytes that look like UTF-16BE
        but marked otherwise).  If the bytes are ill-formed and the heuristics fail,
        the error is handled according to the errors parameter to this function, which is
        passed to the underlying decode() function; by default it throws a
        UnicodeDecodeError exception.

        Note: The mentioned heuristics mean that roundtripping a font to XML and back
        to binary might recover some misencoded data whereas just loading the font
        and saving it back will not change them.
        """

        def isascii(b: int) -> bool:
            return (b >= 0x20 and b <= 0x7E) or b in [0x09, 0x0A, 0x0D]

        encoding = self.getEncoding()
        string = self.string

        if (
            isinstance(string, bytes)
            and encoding == "utf_16_be"
            and len(string) % 2 == 1
        ):
            # Recover badly encoded UTF-16 strings that have an odd number of bytes:
            # - If the last byte is zero, drop it.  Otherwise,
            # - If all the odd bytes are zero and all the even bytes are ASCII,
            #   prepend one zero byte.  Otherwise,
            # - If first byte is zero and all other bytes are ASCII, insert zero
            #   bytes between consecutive ASCII bytes.
            #
            # (Yes, I've seen all of these in the wild... sigh)
            if byteord(string[-1]) == 0:
                string = string[:-1]
            elif all(
                byteord(b) == 0 if i % 2 else isascii(byteord(b))
                for i, b in enumerate(string)
            ):
                string = b"\0" + string
            elif byteord(string[0]) == 0 and all(
                isascii(byteord(b)) for b in string[1:]
            ):
                string = bytesjoin(b"\0" + bytechr(byteord(b)) for b in string[1:])

        string = tostr(string, encoding=encoding, errors=errors)

        # If decoded strings still looks like UTF-16BE, it suggests a double-encoding.
        # Fix it up.
        if all(
            ord(c) == 0 if i % 2 == 0 else isascii(ord(c)) for i, c in enumerate(string)
        ):
            # If string claims to be Mac encoding, but looks like UTF-16BE with ASCII text,
            # narrow it down.
            string = "".join(c for c in string[1::2])

        return string

    def toBytes(self, errors="strict"):
        """If self.string is a bytes object, return it; otherwise try encoding
        the Unicode string in self.string to bytes using the encoding of this
        entry as returned by self.getEncoding(); Note that self.getEncoding()
        returns 'ascii' if the encoding is unknown to the library.

        If the Unicode string cannot be encoded to bytes in the chosen encoding,
        the error is handled according to the errors parameter to this function,
        which is passed to the underlying encode() function; by default it throws a
        UnicodeEncodeError exception.
        """
        return tobytes(self.string, encoding=self.getEncoding(), errors=errors)

    toStr = toUnicode

    def toXML(self, writer, ttFont):
        try:
            unistr = self.toUnicode()
        except UnicodeDecodeError:
            unistr = None
        attrs = [
            ("nameID", self.nameID),
            ("platformID", self.platformID),
            ("platEncID", self.platEncID),
            ("langID", hex(self.langID)),
        ]

        if unistr is None or not self.encodingIsUnicodeCompatible():
            attrs.append(("unicode", unistr is not None))

        writer.begintag("namerecord", attrs)
        writer.newline()
        if unistr is not None:
            writer.write(unistr)
        else:
            writer.write8bit(self.string)
        writer.newline()
        writer.endtag("namerecord")
        writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        self.nameID = safeEval(attrs["nameID"])
        self.platformID = safeEval(attrs["platformID"])
        self.platEncID = safeEval(attrs["platEncID"])
        self.langID = safeEval(attrs["langID"])
        s = strjoin(content).strip()
        encoding = self.getEncoding()
        if self.encodingIsUnicodeCompatible() or safeEval(
            attrs.get("unicode", "False")
        ):
            self.string = s.encode(encoding)
        else:
            # This is the inverse of write8bit...
            self.string = s.encode("latin1")

    def __lt__(self, other):
        if type(self) != type(other):
            return NotImplemented

        try:
            selfTuple = (
                self.platformID,
                self.platEncID,
                self.langID,
                self.nameID,
            )
            otherTuple = (
                other.platformID,
                other.platEncID,
                other.langID,
                other.nameID,
            )
        except AttributeError:
            # This can only happen for
            # 1) an object that is not a NameRecord, or
            # 2) an unlikely incomplete NameRecord object which has not been
            #    fully populated
            return NotImplemented

        try:
            # Include the actual NameRecord string in the comparison tuples
            selfTuple = selfTuple + (self.toBytes(),)
            otherTuple = otherTuple + (other.toBytes(),)
        except UnicodeEncodeError as e:
            # toBytes caused an encoding error in either of the two, so content
            # to sorting based on IDs only
            log.error("NameRecord sorting failed to encode: %s" % e)

        # Implemented so that list.sort() sorts according to the spec by using
        # the order of the tuple items and their comparison
        return selfTuple < otherTuple

    def __repr__(self):
        return "<NameRecord NameID=%d; PlatformID=%d; LanguageID=%d>" % (
            self.nameID,
            self.platformID,
            self.langID,
        )


# Windows language ID → IETF BCP-47 language tag
#
# While Microsoft indicates a region/country for all its language
# IDs, we follow Unicode practice by omitting “most likely subtags”
# as per Unicode CLDR. For example, English is simply “en” and not
# “en-Latn” because according to Unicode, the default script
# for English is Latin.
#
# http://www.unicode.org/cldr/charts/latest/supplemental/likely_subtags.html
# http://www.iana.org/assignments/language-subtag-registry/language-subtag-registry
_WINDOWS_LANGUAGES = {
    0x0436: "af",
    0x041C: "sq",
    0x0484: "gsw",
    0x045E: "am",
    0x1401: "ar-DZ",
    0x3C01: "ar-BH",
    0x0C01: "ar",
    0x0801: "ar-IQ",
    0x2C01: "ar-JO",
    0x3401: "ar-KW",
    0x3001: "ar-LB",
    0x1001: "ar-LY",
    0x1801: "ary",
    0x2001: "ar-OM",
    0x4001: "ar-QA",
    0x0401: "ar-SA",
    0x2801: "ar-SY",
    0x1C01: "aeb",
    0x3801: "ar-AE",
    0x2401: "ar-YE",
    0x042B: "hy",
    0x044D: "as",
    0x082C: "az-Cyrl",
    0x042C: "az",
    0x046D: "ba",
    0x042D: "eu",
    0x0423: "be",
    0x0845: "bn",
    0x0445: "bn-IN",
    0x201A: "bs-Cyrl",
    0x141A: "bs",
    0x047E: "br",
    0x0402: "bg",
    0x0403: "ca",
    0x0C04: "zh-HK",
    0x1404: "zh-MO",
    0x0804: "zh",
    0x1004: "zh-SG",
    0x0404: "zh-TW",
    0x0483: "co",
    0x041A: "hr",
    0x101A: "hr-BA",
    0x0405: "cs",
    0x0406: "da",
    0x048C: "prs",
    0x0465: "dv",
    0x0813: "nl-BE",
    0x0413: "nl",
    0x0C09: "en-AU",
    0x2809: "en-BZ",
    0x1009: "en-CA",
    0x2409: "en-029",
    0x4009: "en-IN",
    0x1809: "en-IE",
    0x2009: "en-JM",
    0x4409: "en-MY",
    0x1409: "en-NZ",
    0x3409: "en-PH",
    0x4809: "en-SG",
    0x1C09: "en-ZA",
    0x2C09: "en-TT",
    0x0809: "en-GB",
    0x0409: "en",
    0x3009: "en-ZW",
    0x0425: "et",
    0x0438: "fo",
    0x0464: "fil",
    0x040B: "fi",
    0x080C: "fr-BE",
    0x0C0C: "fr-CA",
    0x040C: "fr",
    0x140C: "fr-LU",
    0x180C: "fr-MC",
    0x100C: "fr-CH",
    0x0462: "fy",
    0x0456: "gl",
    0x0437: "ka",
    0x0C07: "de-AT",
    0x0407: "de",
    0x1407: "de-LI",
    0x1007: "de-LU",
    0x0807: "de-CH",
    0x0408: "el",
    0x046F: "kl",
    0x0447: "gu",
    0x0468: "ha",
    0x040D: "he",
    0x0439: "hi",
    0x040E: "hu",
    0x040F: "is",
    0x0470: "ig",
    0x0421: "id",
    0x045D: "iu",
    0x085D: "iu-Latn",
    0x083C: "ga",
    0x0434: "xh",
    0x0435: "zu",
    0x0410: "it",
    0x0810: "it-CH",
    0x0411: "ja",
    0x044B: "kn",
    0x043F: "kk",
    0x0453: "km",
    0x0486: "quc",
    0x0487: "rw",
    0x0441: "sw",
    0x0457: "kok",
    0x0412: "ko",
    0x0440: "ky",
    0x0454: "lo",
    0x0426: "lv",
    0x0427: "lt",
    0x082E: "dsb",
    0x046E: "lb",
    0x042F: "mk",
    0x083E: "ms-BN",
    0x043E: "ms",
    0x044C: "ml",
    0x043A: "mt",
    0x0481: "mi",
    0x047A: "arn",
    0x044E: "mr",
    0x047C: "moh",
    0x0450: "mn",
    0x0850: "mn-CN",
    0x0461: "ne",
    0x0414: "nb",
    0x0814: "nn",
    0x0482: "oc",
    0x0448: "or",
    0x0463: "ps",
    0x0415: "pl",
    0x0416: "pt",
    0x0816: "pt-PT",
    0x0446: "pa",
    0x046B: "qu-BO",
    0x086B: "qu-EC",
    0x0C6B: "qu",
    0x0418: "ro",
    0x0417: "rm",
    0x0419: "ru",
    0x243B: "smn",
    0x103B: "smj-NO",
    0x143B: "smj",
    0x0C3B: "se-FI",
    0x043B: "se",
    0x083B: "se-SE",
    0x203B: "sms",
    0x183B: "sma-NO",
    0x1C3B: "sms",
    0x044F: "sa",
    0x1C1A: "sr-Cyrl-BA",
    0x0C1A: "sr",
    0x181A: "sr-Latn-BA",
    0x081A: "sr-Latn",
    0x046C: "nso",
    0x0432: "tn",
    0x045B: "si",
    0x041B: "sk",
    0x0424: "sl",
    0x2C0A: "es-AR",
    0x400A: "es-BO",
    0x340A: "es-CL",
    0x240A: "es-CO",
    0x140A: "es-CR",
    0x1C0A: "es-DO",
    0x300A: "es-EC",
    0x440A: "es-SV",
    0x100A: "es-GT",
    0x480A: "es-HN",
    0x080A: "es-MX",
    0x4C0A: "es-NI",
    0x180A: "es-PA",
    0x3C0A: "es-PY",
    0x280A: "es-PE",
    0x500A: "es-PR",
    # Microsoft has defined two different language codes for
    # “Spanish with modern sorting” and “Spanish with traditional
    # sorting”. This makes sense for collation APIs, and it would be
    # possible to express this in BCP 47 language tags via Unicode
    # extensions (eg., “es-u-co-trad” is “Spanish with traditional
    # sorting”). However, for storing names in fonts, this distinction
    # does not make sense, so we use “es” in both cases.
    0x0C0A: "es",
    0x040A: "es",
    0x540A: "es-US",
    0x380A: "es-UY",
    0x200A: "es-VE",
    0x081D: "sv-FI",
    0x041D: "sv",
    0x045A: "syr",
    0x0428: "tg",
    0x085F: "tzm",
    0x0449: "ta",
    0x0444: "tt",
    0x044A: "te",
    0x041E: "th",
    0x0451: "bo",
    0x041F: "tr",
    0x0442: "tk",
    0x0480: "ug",
    0x0422: "uk",
    0x042E: "hsb",
    0x0420: "ur",
    0x0843: "uz-Cyrl",
    0x0443: "uz",
    0x042A: "vi",
    0x0452: "cy",
    0x0488: "wo",
    0x0485: "sah",
    0x0478: "ii",
    0x046A: "yo",
}


_MAC_LANGUAGES = {
    0: "en",
    1: "fr",
    2: "de",
    3: "it",
    4: "nl",
    5: "sv",
    6: "es",
    7: "da",
    8: "pt",
    9: "no",
    10: "he",
    11: "ja",
    12: "ar",
    13: "fi",
    14: "el",
    15: "is",
    16: "mt",
    17: "tr",
    18: "hr",
    19: "zh-Hant",
    20: "ur",
    21: "hi",
    22: "th",
    23: "ko",
    24: "lt",
    25: "pl",
    26: "hu",
    27: "es",
    28: "lv",
    29: "se",
    30: "fo",
    31: "fa",
    32: "ru",
    33: "zh",
    34: "nl-BE",
    35: "ga",
    36: "sq",
    37: "ro",
    38: "cz",
    39: "sk",
    40: "sl",
    41: "yi",
    42: "sr",
    43: "mk",
    44: "bg",
    45: "uk",
    46: "be",
    47: "uz",
    48: "kk",
    49: "az-Cyrl",
    50: "az-Arab",
    51: "hy",
    52: "ka",
    53: "mo",
    54: "ky",
    55: "tg",
    56: "tk",
    57: "mn-CN",
    58: "mn",
    59: "ps",
    60: "ks",
    61: "ku",
    62: "sd",
    63: "bo",
    64: "ne",
    65: "sa",
    66: "mr",
    67: "bn",
    68: "as",
    69: "gu",
    70: "pa",
    71: "or",
    72: "ml",
    73: "kn",
    74: "ta",
    75: "te",
    76: "si",
    77: "my",
    78: "km",
    79: "lo",
    80: "vi",
    81: "id",
    82: "tl",
    83: "ms",
    84: "ms-Arab",
    85: "am",
    86: "ti",
    87: "om",
    88: "so",
    89: "sw",
    90: "rw",
    91: "rn",
    92: "ny",
    93: "mg",
    94: "eo",
    128: "cy",
    129: "eu",
    130: "ca",
    131: "la",
    132: "qu",
    133: "gn",
    134: "ay",
    135: "tt",
    136: "ug",
    137: "dz",
    138: "jv",
    139: "su",
    140: "gl",
    141: "af",
    142: "br",
    143: "iu",
    144: "gd",
    145: "gv",
    146: "ga",
    147: "to",
    148: "el-polyton",
    149: "kl",
    150: "az",
    151: "nn",
}


_WINDOWS_LANGUAGE_CODES = {
    lang.lower(): code for code, lang in _WINDOWS_LANGUAGES.items()
}
_MAC_LANGUAGE_CODES = {lang.lower(): code for code, lang in _MAC_LANGUAGES.items()}


# MacOS language ID → MacOS script ID
#
# Note that the script ID is not sufficient to determine what encoding
# to use in TrueType files. For some languages, MacOS used a modification
# of a mainstream script. For example, an Icelandic name would be stored
# with smRoman in the TrueType naming table, but the actual encoding
# is a special Icelandic version of the normal Macintosh Roman encoding.
# As another example, Inuktitut uses an 8-bit encoding for Canadian Aboriginal
# Syllables but MacOS had run out of available script codes, so this was
# done as a (pretty radical) “modification” of Ethiopic.
#
# http://unicode.org/Public/MAPPINGS/VENDORS/APPLE/Readme.txt
_MAC_LANGUAGE_TO_SCRIPT = {
    0: 0,  # langEnglish → smRoman
    1: 0,  # langFrench → smRoman
    2: 0,  # langGerman → smRoman
    3: 0,  # langItalian → smRoman
    4: 0,  # langDutch → smRoman
    5: 0,  # langSwedish → smRoman
    6: 0,  # langSpanish → smRoman
    7: 0,  # langDanish → smRoman
    8: 0,  # langPortuguese → smRoman
    9: 0,  # langNorwegian → smRoman
    10: 5,  # langHebrew → smHebrew
    11: 1,  # langJapanese → smJapanese
    12: 4,  # langArabic → smArabic
    13: 0,  # langFinnish → smRoman
    14: 6,  # langGreek → smGreek
    15: 0,  # langIcelandic → smRoman (modified)
    16: 0,  # langMaltese → smRoman
    17: 0,  # langTurkish → smRoman (modified)
    18: 0,  # langCroatian → smRoman (modified)
    19: 2,  # langTradChinese → smTradChinese
    20: 4,  # langUrdu → smArabic
    21: 9,  # langHindi → smDevanagari
    22: 21,  # langThai → smThai
    23: 3,  # langKorean → smKorean
    24: 29,  # langLithuanian → smCentralEuroRoman
    25: 29,  # langPolish → smCentralEuroRoman
    26: 29,  # langHungarian → smCentralEuroRoman
    27: 29,  # langEstonian → smCentralEuroRoman
    28: 29,  # langLatvian → smCentralEuroRoman
    29: 0,  # langSami → smRoman
    30: 0,  # langFaroese → smRoman (modified)
    31: 4,  # langFarsi → smArabic (modified)
    32: 7,  # langRussian → smCyrillic
    33: 25,  # langSimpChinese → smSimpChinese
    34: 0,  # langFlemish → smRoman
    35: 0,  # langIrishGaelic → smRoman (modified)
    36: 0,  # langAlbanian → smRoman
    37: 0,  # langRomanian → smRoman (modified)
    38: 29,  # langCzech → smCentralEuroRoman
    39: 29,  # langSlovak → smCentralEuroRoman
    40: 0,  # langSlovenian → smRoman (modified)
    41: 5,  # langYiddish → smHebrew
    42: 7,  # langSerbian → smCyrillic
    43: 7,  # langMacedonian → smCyrillic
    44: 7,  # langBulgarian → smCyrillic
    45: 7,  # langUkrainian → smCyrillic (modified)
    46: 7,  # langByelorussian → smCyrillic
    47: 7,  # langUzbek → smCyrillic
    48: 7,  # langKazakh → smCyrillic
    49: 7,  # langAzerbaijani → smCyrillic
    50: 4,  # langAzerbaijanAr → smArabic
    51: 24,  # langArmenian → smArmenian
    52: 23,  # langGeorgian → smGeorgian
    53: 7,  # langMoldavian → smCyrillic
    54: 7,  # langKirghiz → smCyrillic
    55: 7,  # langTajiki → smCyrillic
    56: 7,  # langTurkmen → smCyrillic
    57: 27,  # langMongolian → smMongolian
    58: 7,  # langMongolianCyr → smCyrillic
    59: 4,  # langPashto → smArabic
    60: 4,  # langKurdish → smArabic
    61: 4,  # langKashmiri → smArabic
    62: 4,  # langSindhi → smArabic
    63: 26,  # langTibetan → smTibetan
    64: 9,  # langNepali → smDevanagari
    65: 9,  # langSanskrit → smDevanagari
    66: 9,  # langMarathi → smDevanagari
    67: 13,  # langBengali → smBengali
    68: 13,  # langAssamese → smBengali
    69: 11,  # langGujarati → smGujarati
    70: 10,  # langPunjabi → smGurmukhi
    71: 12,  # langOriya → smOriya
    72: 17,  # langMalayalam → smMalayalam
    73: 16,  # langKannada → smKannada
    74: 14,  # langTamil → smTamil
    75: 15,  # langTelugu → smTelugu
    76: 18,  # langSinhalese → smSinhalese
    77: 19,  # langBurmese → smBurmese
    78: 20,  # langKhmer → smKhmer
    79: 22,  # langLao → smLao
    80: 30,  # langVietnamese → smVietnamese
    81: 0,  # langIndonesian → smRoman
    82: 0,  # langTagalog → smRoman
    83: 0,  # langMalayRoman → smRoman
    84: 4,  # langMalayArabic → smArabic
    85: 28,  # langAmharic → smEthiopic
    86: 28,  # langTigrinya → smEthiopic
    87: 28,  # langOromo → smEthiopic
    88: 0,  # langSomali → smRoman
    89: 0,  # langSwahili → smRoman
    90: 0,  # langKinyarwanda → smRoman
    91: 0,  # langRundi → smRoman
    92: 0,  # langNyanja → smRoman
    93: 0,  # langMalagasy → smRoman
    94: 0,  # langEsperanto → smRoman
    128: 0,  # langWelsh → smRoman (modified)
    129: 0,  # langBasque → smRoman
    130: 0,  # langCatalan → smRoman
    131: 0,  # langLatin → smRoman
    132: 0,  # langQuechua → smRoman
    133: 0,  # langGuarani → smRoman
    134: 0,  # langAymara → smRoman
    135: 7,  # langTatar → smCyrillic
    136: 4,  # langUighur → smArabic
    137: 26,  # langDzongkha → smTibetan
    138: 0,  # langJavaneseRom → smRoman
    139: 0,  # langSundaneseRom → smRoman
    140: 0,  # langGalician → smRoman
    141: 0,  # langAfrikaans → smRoman
    142: 0,  # langBreton → smRoman (modified)
    143: 28,  # langInuktitut → smEthiopic (modified)
    144: 0,  # langScottishGaelic → smRoman (modified)
    145: 0,  # langManxGaelic → smRoman (modified)
    146: 0,  # langIrishGaelicScript → smRoman (modified)
    147: 0,  # langTongan → smRoman
    148: 6,  # langGreekAncient → smRoman
    149: 0,  # langGreenlandic → smRoman
    150: 0,  # langAzerbaijanRoman → smRoman
    151: 0,  # langNynorsk → smRoman
}


class NameRecordVisitor(TTVisitor):
    # Font tables that have NameIDs we need to collect.
    TABLES = ("GSUB", "GPOS", "fvar", "CPAL", "STAT")

    def __init__(self):
        self.seen = set()


@NameRecordVisitor.register_attrs(
    (
        (otTables.FeatureParamsSize, ("SubfamilyNameID",)),
        (otTables.FeatureParamsStylisticSet, ("UINameID",)),
        (otTables.STAT, ("ElidedFallbackNameID",)),
        (otTables.AxisRecord, ("AxisNameID",)),
        (otTables.AxisValue, ("ValueNameID",)),
        (otTables.FeatureName, ("FeatureNameID",)),
        (otTables.Setting, ("SettingNameID",)),
    )
)
def visit(visitor, obj, attr, value):
    visitor.seen.add(value)


@NameRecordVisitor.register(otTables.FeatureParamsCharacterVariants)
def visit(visitor, obj):
    for attr in ("FeatUILabelNameID", "FeatUITooltipTextNameID", "SampleTextNameID"):
        value = getattr(obj, attr)
        visitor.seen.add(value)
    # also include the sequence of UI strings for individual variants, if any
    if obj.FirstParamUILabelNameID == 0 or obj.NumNamedParameters == 0:
        return
    visitor.seen.update(
        range(
            obj.FirstParamUILabelNameID,
            obj.FirstParamUILabelNameID + obj.NumNamedParameters,
        )
    )


@NameRecordVisitor.register(ttLib.getTableClass("fvar"))
def visit(visitor, obj):
    for inst in obj.instances:
        if inst.postscriptNameID != 0xFFFF:
            visitor.seen.add(inst.postscriptNameID)
        visitor.seen.add(inst.subfamilyNameID)

    for axis in obj.axes:
        visitor.seen.add(axis.axisNameID)


@NameRecordVisitor.register(ttLib.getTableClass("CPAL"))
def visit(visitor, obj):
    if obj.version == 1:
        visitor.seen.update(obj.paletteLabels)
        visitor.seen.update(obj.paletteEntryLabels)


@NameRecordVisitor.register(ttLib.TTFont)
def visit(visitor, font, *args, **kwargs):
    if hasattr(visitor, "font"):
        return False

    visitor.font = font
    for tag in visitor.TABLES:
        if tag in font:
            visitor.visit(font[tag], *args, **kwargs)
    del visitor.font
    return False
<|endoftext|>"""TSI{B,C,D,J,P,S,V} are private tables used by Microsoft Visual TrueType (VTT)
tool to store its table source data.

See also https://learn.microsoft.com/en-us/typography/tools/vtt/tsi-tables
"""

from fontTools.misc.textTools import strjoin, tobytes, tostr
from . import asciiTable


class table_T_S_I_V_(asciiTable.asciiTable):
    def toXML(self, writer, ttFont):
        data = tostr(self.data)
        # removing null bytes. XXX needed??
        data = data.split("\0")
        data = strjoin(data)
        writer.begintag("source")
        writer.newline()
        writer.write_noindent(data.replace("\r", "\n"))
        writer.newline()
        writer.endtag("source")
        writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        lines = strjoin(content).split("\n")
        self.data = tobytes("\r".join(lines[1:-1]))
<|endoftext|>from fontTools.misc import sstruct
from fontTools.misc.fixedTools import floatToFixedToStr
from fontTools.misc.textTools import safeEval

# from itertools import *
from functools import partial
from . import DefaultTable
from . import grUtils
import struct

Glat_format_0 = """
    >        # big endian
    version: 16.16F
"""

Glat_format_3 = """
    >
    version: 16.16F
    compression:L    # compression scheme or reserved 
"""

Glat_format_1_entry = """
    >
    attNum:     B    # Attribute number of first attribute
    num:        B    # Number of attributes in this run
"""
Glat_format_23_entry = """
    >
    attNum:     H    # Attribute number of first attribute
    num:        H    # Number of attributes in this run
"""

Glat_format_3_octabox_metrics = """
    >
    subboxBitmap:   H    # Which subboxes exist on 4x4 grid
    diagNegMin:     B    # Defines minimum negatively-sloped diagonal (si)
    diagNegMax:     B    # Defines maximum negatively-sloped diagonal (sa)
    diagPosMin:     B    # Defines minimum positively-sloped diagonal (di)
    diagPosMax:     B    # Defines maximum positively-sloped diagonal (da)
"""

Glat_format_3_subbox_entry = """
    >
    left:           B    # xi
    right:          B    # xa
    bottom:         B    # yi
    top:            B    # ya
    diagNegMin:     B    # Defines minimum negatively-sloped diagonal (si)
    diagNegMax:     B    # Defines maximum negatively-sloped diagonal (sa)
    diagPosMin:     B    # Defines minimum positively-sloped diagonal (di)
    diagPosMax:     B    # Defines maximum positively-sloped diagonal (da)
"""


class _Object:
    pass


class _Dict(dict):
    pass


class table_G__l_a_t(DefaultTable.DefaultTable):
    """Graphite Glyph Attributes table

    See also https://graphite.sil.org/graphite_techAbout#graphite-font-tables
    """

    def __init__(self, tag=None):
        DefaultTable.DefaultTable.__init__(self, tag)
        self.scheme = 0

    def decompile(self, data, ttFont):
        sstruct.unpack2(Glat_format_0, data, self)
        self.version = float(floatToFixedToStr(self.version, precisionBits=16))
        if self.version <= 1.9:
            decoder = partial(self.decompileAttributes12, fmt=Glat_format_1_entry)
        elif self.version <= 2.9:
            decoder = partial(self.decompileAttributes12, fmt=Glat_format_23_entry)
        elif self.version >= 3.0:
            data, self.scheme = grUtils.decompress(data)
            sstruct.unpack2(Glat_format_3, data, self)
            self.hasOctaboxes = (self.compression & 1) == 1
            decoder = self.decompileAttributes3

        gloc = ttFont["Gloc"]
        self.attributes = {}
        count = 0
        for s, e in zip(gloc, gloc[1:]):
            self.attributes[ttFont.getGlyphName(count)] = decoder(data[s:e])
            count += 1

    def decompileAttributes12(self, data, fmt):
        attributes = _Dict()
        while len(data) > 3:
            e, data = sstruct.unpack2(fmt, data, _Object())
            keys = range(e.attNum, e.attNum + e.num)
            if len(data) >= 2 * e.num:
                vals = struct.unpack_from((">%dh" % e.num), data)
                attributes.update(zip(keys, vals))
                data = data[2 * e.num :]
        return attributes

    def decompileAttributes3(self, data):
        if self.hasOctaboxes:
            o, data = sstruct.unpack2(Glat_format_3_octabox_metrics, data, _Object())
            numsub = bin(o.subboxBitmap).count("1")
            o.subboxes = []
            for b in range(numsub):
                if len(data) >= 8:
                    subbox, data = sstruct.unpack2(
                        Glat_format_3_subbox_entry, data, _Object()
                    )
                    o.subboxes.append(subbox)
        attrs = self.decompileAttributes12(data, Glat_format_23_entry)
        if self.hasOctaboxes:
            attrs.octabox = o
        return attrs

    def compile(self, ttFont):
        data = sstruct.pack(Glat_format_0, self)
        if self.version <= 1.9:
            encoder = partial(self.compileAttributes12, fmt=Glat_format_1_entry)
        elif self.version <= 2.9:
            encoder = partial(self.compileAttributes12, fmt=Glat_format_1_entry)
        elif self.version >= 3.0:
            self.compression = (self.scheme << 27) + (1 if self.hasOctaboxes else 0)
            data = sstruct.pack(Glat_format_3, self)
            encoder = self.compileAttributes3

        glocs = []
        for n in range(len(self.attributes)):
            glocs.append(len(data))
            data += encoder(self.attributes[ttFont.getGlyphName(n)])
        glocs.append(len(data))
        ttFont["Gloc"].set(glocs)

        if self.version >= 3.0:
            data = grUtils.compress(self.scheme, data)
        return data

    def compileAttributes12(self, attrs, fmt):
        data = b""
        for e in grUtils.entries(attrs):
            data += sstruct.pack(fmt, {"attNum": e[0], "num": e[1]}) + struct.pack(
                (">%dh" % len(e[2])), *e[2]
            )
        return data

    def compileAttributes3(self, attrs):
        if self.hasOctaboxes:
            o = attrs.octabox
            data = sstruct.pack(Glat_format_3_octabox_metrics, o)
            numsub = bin(o.subboxBitmap).count("1")
            for b in range(numsub):
                data += sstruct.pack(Glat_format_3_subbox_entry, o.subboxes[b])
        else:
            data = ""
        return data + self.compileAttributes12(attrs, Glat_format_23_entry)

    def toXML(self, writer, ttFont):
        writer.simpletag("version", version=self.version, compressionScheme=self.scheme)
        writer.newline()
        for n, a in sorted(
            self.attributes.items(), key=lambda x: ttFont.getGlyphID(x[0])
        ):
            writer.begintag("glyph", name=n)
            writer.newline()
            if hasattr(a, "octabox"):
                o = a.octabox
                formatstring, names, fixes = sstruct.getformat(
                    Glat_format_3_octabox_metrics
                )
                vals = {}
                for k in names:
                    if k == "subboxBitmap":
                        continue
                    vals[k] = "{:.3f}%".format(getattr(o, k) * 100.0 / 255)
                vals["bitmap"] = "{:0X}".format(o.subboxBitmap)
                writer.begintag("octaboxes", **vals)
                writer.newline()
                formatstring, names, fixes = sstruct.getformat(
                    Glat_format_3_subbox_entry
                )
                for s in o.subboxes:
                    vals = {}
                    for k in names:
                        vals[k] = "{:.3f}%".format(getattr(s, k) * 100.0 / 255)
                    writer.simpletag("octabox", **vals)
                    writer.newline()
                writer.endtag("octaboxes")
                writer.newline()
            for k, v in sorted(a.items()):
                writer.simpletag("attribute", index=k, value=v)
                writer.newline()
            writer.endtag("glyph")
            writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        if name == "version":
            self.version = float(safeEval(attrs["version"]))
            self.scheme = int(safeEval(attrs["compressionScheme"]))
        if name != "glyph":
            return
        if not hasattr(self, "attributes"):
            self.attributes = {}
        gname = attrs["name"]
        attributes = _Dict()
        for element in content:
            if not isinstance(element, tuple):
                continue
            tag, attrs, subcontent = element
            if tag == "attribute":
                k = int(safeEval(attrs["index"]))
                v = int(safeEval(attrs["value"]))
                attributes[k] = v
            elif tag == "octaboxes":
                self.hasOctaboxes = True
                o = _Object()
                o.subboxBitmap = int(attrs["bitmap"], 16)
                o.subboxes = []
                del attrs["bitmap"]
                for k, v in attrs.items():
                    setattr(o, k, int(float(v[:-1]) * 255.0 / 100.0 + 0.5))
                for element in subcontent:
                    if not isinstance(element, tuple):
                        continue
                    tag, attrs, subcontent = element
                    so = _Object()
                    for k, v in attrs.items():
                        setattr(so, k, int(float(v[:-1]) * 255.0 / 100.0 + 0.5))
                    o.subboxes.append(so)
                attributes.octabox = o
        self.attributes[gname] = attributes
<|endoftext|>"""TSI{0,1,2,3,5} are private tables used by Microsoft Visual TrueType (VTT)
tool to store its hinting source data.

TSI5 contains the VTT character groups.

See also https://learn.microsoft.com/en-us/typography/tools/vtt/tsi-tables
"""

import array
import logging
import sys

from fontTools.misc.textTools import safeEval

from . import DefaultTable

log = logging.getLogger(__name__)


class table_T_S_I__5(DefaultTable.DefaultTable):
    def decompile(self, data, ttFont):
        numGlyphs = ttFont["maxp"].numGlyphs
        a = array.array("H")
        a.frombytes(data)
        if sys.byteorder != "big":
            a.byteswap()
        self.glyphGrouping = {}
        numEntries = len(data) // 2
        if numEntries != numGlyphs:
            diff = numEntries - numGlyphs
            log.warning(
                "Number of entries differs from the number of glyphs in the font "
                f"by {abs(diff)} ({numEntries} entries vs. {numGlyphs} glyphs)."
            )
        for i in range(numEntries):
            self.glyphGrouping[ttFont.getGlyphName(i)] = a[i]

    def compile(self, ttFont):
        glyphNames = ttFont.getGlyphOrder()
        a = array.array("H")
        for glyphName in glyphNames:
            a.append(self.glyphGrouping.get(glyphName, 0))
        if sys.byteorder != "big":
            a.byteswap()
        return a.tobytes()

    def toXML(self, writer, ttFont):
        names = sorted(self.glyphGrouping.keys())
        for glyphName in names:
            writer.simpletag(
                "glyphgroup", name=glyphName, value=self.glyphGrouping[glyphName]
            )
            writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        if not hasattr(self, "glyphGrouping"):
            self.glyphGrouping = {}
        if name != "glyphgroup":
            return
        self.glyphGrouping[attrs["name"]] = safeEval(attrs["value"])
<|endoftext|>from fontTools.misc.textTools import bytesjoin, safeEval
from . import DefaultTable
import struct


class table_V_O_R_G_(DefaultTable.DefaultTable):
    """Vertical Origin table

    The ``VORG`` table contains the vertical origin of each glyph
    in a `CFF` or `CFF2` font.

    This table is structured so that you can treat it like a dictionary keyed by glyph name.

    ``ttFont['VORG'][<glyphName>]`` will return the vertical origin for any glyph.

    ``ttFont['VORG'][<glyphName>] = <value>`` will set the vertical origin for any glyph.

    See also https://learn.microsoft.com/en-us/typography/opentype/spec/vorg
    """

    def decompile(self, data, ttFont):
        self.getGlyphName = (
            ttFont.getGlyphName
        )  # for use in get/set item functions, for access by GID
        (
            self.majorVersion,
            self.minorVersion,
            self.defaultVertOriginY,
            self.numVertOriginYMetrics,
        ) = struct.unpack(">HHhH", data[:8])
        assert (
            self.majorVersion <= 1
        ), "Major version of VORG table is higher than I know how to handle"
        data = data[8:]
        vids = []
        gids = []
        pos = 0
        for i in range(self.numVertOriginYMetrics):
            gid, vOrigin = struct.unpack(">Hh", data[pos : pos + 4])
            pos += 4
            gids.append(gid)
            vids.append(vOrigin)

        self.VOriginRecords = vOrig = {}
        glyphOrder = ttFont.getGlyphOrder()
        try:
            names = [glyphOrder[gid] for gid in gids]
        except IndexError:
            getGlyphName = self.getGlyphName
            names = map(getGlyphName, gids)

        for name, vid in zip(names, vids):
            vOrig[name] = vid

    def compile(self, ttFont):
        vorgs = list(self.VOriginRecords.values())
        names = list(self.VOriginRecords.keys())
        nameMap = ttFont.getReverseGlyphMap()
        try:
            gids = [nameMap[name] for name in names]
        except KeyError:
            nameMap = ttFont.getReverseGlyphMap(rebuild=True)
            gids = [nameMap[name] for name in names]
        vOriginTable = list(zip(gids, vorgs))
        self.numVertOriginYMetrics = len(vorgs)
        vOriginTable.sort()  # must be in ascending GID order
        dataList = [struct.pack(">Hh", rec[0], rec[1]) for rec in vOriginTable]
        header = struct.pack(
            ">HHhH",
            self.majorVersion,
            self.minorVersion,
            self.defaultVertOriginY,
            self.numVertOriginYMetrics,
        )
        dataList.insert(0, header)
        data = bytesjoin(dataList)
        return data

    def toXML(self, writer, ttFont):
        writer.simpletag("majorVersion", value=self.majorVersion)
        writer.newline()
        writer.simpletag("minorVersion", value=self.minorVersion)
        writer.newline()
        writer.simpletag("defaultVertOriginY", value=self.defaultVertOriginY)
        writer.newline()
        writer.simpletag("numVertOriginYMetrics", value=self.numVertOriginYMetrics)
        writer.newline()
        vOriginTable = []
        glyphNames = self.VOriginRecords.keys()
        for glyphName in glyphNames:
            try:
                gid = ttFont.getGlyphID(glyphName)
            except:
                assert 0, (
                    "VORG table contains a glyph name not in ttFont.getGlyphNames(): "
                    + str(glyphName)
                )
            vOriginTable.append([gid, glyphName, self.VOriginRecords[glyphName]])
        vOriginTable.sort()
        for entry in vOriginTable:
            vOriginRec = VOriginRecord(entry[1], entry[2])
            vOriginRec.toXML(writer, ttFont)

    def fromXML(self, name, attrs, content, ttFont):
        if not hasattr(self, "VOriginRecords"):
            self.VOriginRecords = {}
        self.getGlyphName = (
            ttFont.getGlyphName
        )  # for use in get/set item functions, for access by GID
        if name == "VOriginRecord":
            vOriginRec = VOriginRecord()
            for element in content:
                if isinstance(element, str):
                    continue
                name, attrs, content = element
                vOriginRec.fromXML(name, attrs, content, ttFont)
            self.VOriginRecords[vOriginRec.glyphName] = vOriginRec.vOrigin
        elif "value" in attrs:
            setattr(self, name, safeEval(attrs["value"]))

    def __getitem__(self, glyphSelector):
        if isinstance(glyphSelector, int):
            # its a gid, convert to glyph name
            glyphSelector = self.getGlyphName(glyphSelector)

        if glyphSelector not in self.VOriginRecords:
            return self.defaultVertOriginY

        return self.VOriginRecords[glyphSelector]

    def __setitem__(self, glyphSelector, value):
        if isinstance(glyphSelector, int):
            # its a gid, convert to glyph name
            glyphSelector = self.getGlyphName(glyphSelector)

        if value != self.defaultVertOriginY:
            self.VOriginRecords[glyphSelector] = value
        elif glyphSelector in self.VOriginRecords:
            del self.VOriginRecords[glyphSelector]

    def __delitem__(self, glyphSelector):
        del self.VOriginRecords[glyphSelector]


class VOriginRecord(object):
    def __init__(self, name=None, vOrigin=None):
        self.glyphName = name
        self.vOrigin = vOrigin

    def toXML(self, writer, ttFont):
        writer.begintag("VOriginRecord")
        writer.newline()
        writer.simpletag("glyphName", value=self.glyphName)
        writer.newline()
        writer.simpletag("vOrigin", value=self.vOrigin)
        writer.newline()
        writer.endtag("VOriginRecord")
        writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        value = attrs["value"]
        if name == "glyphName":
            setattr(self, name, value)
        else:
            setattr(self, name, safeEval(value))
<|endoftext|>from fontTools.misc.textTools import strjoin, tobytes, tostr
from . import DefaultTable


class asciiTable(DefaultTable.DefaultTable):
    def toXML(self, writer, ttFont):
        data = tostr(self.data)
        # removing null bytes. XXX needed??
        data = data.split("\0")
        data = strjoin(data)
        writer.begintag("source")
        writer.newline()
        writer.write_noindent(data)
        writer.newline()
        writer.endtag("source")
        writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        lines = strjoin(content).split("\n")
        self.data = tobytes("\n".join(lines[1:-1]))
<|endoftext|>from fontTools.misc.fixedTools import (
    fixedToFloat as fi2fl,
    floatToFixed as fl2fi,
    floatToFixedToStr as fl2str,
    strToFixedToFloat as str2fl,
    ensureVersionIsLong as fi2ve,
    versionToFixed as ve2fi,
)
from fontTools.ttLib.tables.TupleVariation import TupleVariation
from fontTools.misc.roundTools import nearestMultipleShortestRepr, otRound
from fontTools.misc.textTools import bytesjoin, tobytes, tostr, pad, safeEval
from fontTools.misc.lazyTools import LazyList
from fontTools.ttLib import OPTIMIZE_FONT_SPEED, getSearchRange
from .otBase import (
    CountReference,
    FormatSwitchingBaseTable,
    OTTableReader,
    OTTableWriter,
    ValueRecordFactory,
)
from .otDataSchema import FieldSpec
from .otTables import (
    lookupTypes,
    VarCompositeGlyph,
    AATStateTable,
    AATState,
    AATAction,
    ContextualMorphAction,
    LigAction,
    LigatureMorphAction,
    InsertionMorphAction,
    MortSubtable,
    MorxSubtable,
    ExtendMode as _ExtendMode,
    CompositeMode as _CompositeMode,
    NO_VARIATION_INDEX,
)
from itertools import zip_longest, accumulate
from functools import partial
from types import SimpleNamespace
import re
import struct
from typing import Optional
import logging
from enum import IntFlag

log = logging.getLogger(__name__)
istuple = lambda t: isinstance(t, tuple)


def buildConverters(tableSpec: list[FieldSpec], tableNamespace):
    """Given a table spec from otData.py, build a converter object for each
    field of the table. This is called for each table in otData.py, and
    the results are assigned to the corresponding class in otTables.py."""
    converters = []
    convertersByName = {}
    for spec in tableSpec:
        tableName = spec.name
        if spec.name.startswith("ValueFormat"):
            assert spec.type == "uint16"
            converterClass = ValueFormat
        elif spec.name.endswith("Count") or spec.name in ("StructLength", "MorphType"):
            converterClass = {
                "uint8": ComputedUInt8,
                "uint16": ComputedUShort,
                "uint32": ComputedULong,
            }[spec.type]
        elif spec.name == "SubTable":
            converterClass = SubTable
        elif spec.name == "ExtSubTable":
            converterClass = ExtSubTable
        elif spec.name == "SubStruct":
            converterClass = SubStruct
        elif spec.name == "FeatureParams":
            converterClass = FeatureParams
        elif spec.name in ("CIDGlyphMapping", "GlyphCIDMapping"):
            converterClass = StructWithLength
        else:
            if not spec.type in converterMapping and "(" not in spec.type:
                tableName = spec.type
                converterClass = Struct
            else:
                converterClass = eval(spec.type, tableNamespace, converterMapping)

        conv = converterClass(
            spec.name, spec.repeat, spec.aux, description=spec.description
        )

        if conv.tableClass:
            # A "template" such as OffsetTo(AType) knows the table class already
            tableClass = conv.tableClass
        elif spec.type in ("MortChain", "MortSubtable", "MorxChain"):
            tableClass = tableNamespace.get(spec.type)
        else:
            tableClass = tableNamespace.get(tableName)

        if not conv.tableClass:
            conv.tableClass = tableClass

        if spec.name in ["SubTable", "ExtSubTable", "SubStruct"]:
            conv.lookupTypes = tableNamespace["lookupTypes"]
            # also create reverse mapping
            for t in conv.lookupTypes.values():
                for cls in t.values():
                    convertersByName[cls.__name__] = Table(
                        spec.name, spec.repeat, spec.aux, cls
                    )
        if spec.name == "FeatureParams":
            conv.featureParamTypes = tableNamespace["featureParamTypes"]
            conv.defaultFeatureParams = tableNamespace["FeatureParams"]
            for cls in conv.featureParamTypes.values():
                convertersByName[cls.__name__] = Table(
                    spec.name, spec.repeat, spec.aux, cls
                )
        converters.append(conv)
        assert spec.name not in convertersByName, spec.name
        convertersByName[spec.name] = conv
    return converters, convertersByName


class BaseConverter(object):
    """Base class for converter objects. Apart from the constructor, this
    is an abstract class."""

    def __init__(self, name, repeat, aux, tableClass=None, *, description=""):
        self.name = name
        self.repeat = repeat
        self.aux = aux
        if self.aux and not self.repeat:
            self.aux = compile(self.aux, "<string>", "eval")
        self.tableClass = tableClass
        self.isCount = name.endswith("Count") or name in [
            "DesignAxisRecordSize",
            "ValueRecordSize",
        ]
        self.isLookupType = name.endswith("LookupType") or name == "MorphType"
        self.isPropagated = name in [
            "ClassCount",
            "Class2Count",
            "FeatureTag",
            "SettingsCount",
            "VarRegionCount",
            "MappingCount",
            "RegionAxisCount",
            "DesignAxisCount",
            "DesignAxisRecordSize",
            "AxisValueCount",
            "ValueRecordSize",
            "AxisCount",
            "BaseGlyphRecordCount",
            "LayerRecordCount",
            "AxisIndicesList",
            # IFT - propagated to GlyphMap/FeatureMap/MappingEntries subtables
            "NumGlyphs",
            "MaxGlyphMapEntryIndex",
            "MaxEntryIndex",
            "NumEntries",
        ]
        self.description = description

    def readArray(self, reader, font, tableDict, count):
        """Read an array of values from the reader."""
        lazy = font.lazy and count > 8
        if lazy:
            recordSize = self.getRecordSize(reader)
            if recordSize is NotImplemented:
                lazy = False
        if not lazy:
            l = []
            for i in range(count):
                l.append(self.read(reader, font, tableDict))
            return l
        else:

            def get_read_item():
                reader_copy = reader.copy()
                pos = reader.pos

                def read_item(i):
                    reader_copy.seek(pos + i * recordSize)
                    return self.read(reader_copy, font, {})

                return read_item

            read_item = get_read_item()
            l = LazyList(read_item for i in range(count))
            reader.advance(count * recordSize)

            return l

    def getRecordSize(self, reader):
        if hasattr(self, "staticSize"):
            return self.staticSize
        return NotImplemented

    def read(self, reader, font, tableDict):
        """Read a value from the reader."""
        raise NotImplementedError(self)

    def writeArray(self, writer, font, tableDict, values):
        try:
            for i, value in enumerate(values):
                self.write(writer, font, tableDict, value, i)
        except Exception as e:
            e.args = e.args + (i,)
            raise

    def write(self, writer, font, tableDict, value, repeatIndex=None):
        """Write a value to the writer."""
        raise NotImplementedError(self)

    def xmlRead(self, attrs, content, font):
        """Read a value from XML."""
        raise NotImplementedError(self)

    def xmlWrite(self, xmlWriter, font, value, name, attrs):
        """Write a value to XML."""
        raise NotImplementedError(self)

    varIndexBasePlusOffsetRE = re.compile(r"VarIndexBase\s*\+\s*(\d+)")

    def getVarIndexOffset(self) -> Optional[int]:
        """If description has `VarIndexBase + {offset}`, return the offset else None."""
        m = self.varIndexBasePlusOffsetRE.search(self.description)
        if not m:
            return None
        return int(m.group(1))


class SimpleValue(BaseConverter):
    @staticmethod
    def toString(value):
        return value

    @staticmethod
    def fromString(value):
        return value

    def xmlWrite(self, xmlWriter, font, value, name, attrs):
        xmlWriter.simpletag(name, attrs + [("value", self.toString(value))])
        xmlWriter.newline()

    def xmlRead(self, attrs, content, font):
        return self.fromString(attrs["value"])


class OptionalValue(SimpleValue):
    DEFAULT = None

    def xmlWrite(self, xmlWriter, font, value, name, attrs):
        if value != self.DEFAULT:
            attrs.append(("value", self.toString(value)))
        xmlWriter.simpletag(name, attrs)
        xmlWriter.newline()

    def xmlRead(self, attrs, content, font):
        if "value" in attrs:
            return self.fromString(attrs["value"])
        return self.DEFAULT


class IntValue(SimpleValue):
    @staticmethod
    def fromString(value):
        return int(value, 0)


class Long(IntValue):
    staticSize = 4

    def read(self, reader, font, tableDict):
        return reader.readLong()

    def readArray(self, reader, font, tableDict, count):
        return reader.readLongArray(count)

    def write(self, writer, font, tableDict, value, repeatIndex=None):
        writer.writeLong(value)

    def writeArray(self, writer, font, tableDict, values):
        writer.writeLongArray(values)


class ULong(IntValue):
    staticSize = 4

    def read(self, reader, font, tableDict):
        return reader.readULong()

    def readArray(self, reader, font, tableDict, count):
        return reader.readULongArray(count)

    def write(self, writer, font, tableDict, value, repeatIndex=None):
        writer.writeULong(value)

    def writeArray(self, writer, font, tableDict, values):
        writer.writeULongArray(values)


class Flags32(ULong):
    @staticmethod
    def toString(value):
        return "0x%08X" % value


class VarIndex(OptionalValue, ULong):
    DEFAULT = NO_VARIATION_INDEX


class Short(IntValue):
    staticSize = 2

    def read(self, reader, font, tableDict):
        return reader.readShort()

    def readArray(self, reader, font, tableDict, count):
        return reader.readShortArray(count)

    def write(self, writer, font, tableDict, value, repeatIndex=None):
        writer.writeShort(value)

    def writeArray(self, writer, font, tableDict, values):
        writer.writeShortArray(values)


class UShort(IntValue):
    staticSize = 2

    def read(self, reader, font, tableDict):
        return reader.readUShort()

    def readArray(self, reader, font, tableDict, count):
        return reader.readUShortArray(count)

    def write(self, writer, font, tableDict, value, repeatIndex=None):
        writer.writeUShort(value)

    def writeArray(self, writer, font, tableDict, values):
        writer.writeUShortArray(values)


class Int8(IntValue):
    staticSize = 1

    def read(self, reader, font, tableDict):
        return reader.readInt8()

    def readArray(self, reader, font, tableDict, count):
        return reader.readInt8Array(count)

    def write(self, writer, font, tableDict, value, repeatIndex=None):
        writer.writeInt8(value)

    def writeArray(self, writer, font, tableDict, values):
        writer.writeInt8Array(values)


class UInt8(IntValue):
    staticSize = 1

    def read(self, reader, font, tableDict):
        return reader.readUInt8()

    def readArray(self, reader, font, tableDict, count):
        return reader.readUInt8Array(count)

    def write(self, writer, font, tableDict, value, repeatIndex=None):
        writer.writeUInt8(value)

    def writeArray(self, writer, font, tableDict, values):
        writer.writeUInt8Array(values)


class UInt24(IntValue):
    staticSize = 3

    def read(self, reader, font, tableDict):
        return reader.readUInt24()

    def write(self, writer, font, tableDict, value, repeatIndex=None):
        writer.writeUInt24(value)


class ComputedInt(IntValue):
    def xmlWrite(self, xmlWriter, font, value, name, attrs):
        if value is not None:
            xmlWriter.comment("%s=%s" % (name, value))
            xmlWriter.newline()


class ComputedUInt8(ComputedInt, UInt8):
    pass


class ComputedUShort(ComputedInt, UShort):
    pass


class ComputedULong(ComputedInt, ULong):
    pass


class Tag(SimpleValue):
    staticSize = 4

    def read(self, reader, font, tableDict):
        return reader.readTag()

    def write(self, writer, font, tableDict, value, repeatIndex=None):
        writer.writeTag(value)


class GlyphID(SimpleValue):
    staticSize = 2
    typecode = "H"

    def readArray(self, reader, font, tableDict, count):
        return font.getGlyphNameMany(
            reader.readArray(self.typecode, self.staticSize, count)
        )

    def read(self, reader, font, tableDict):
        return font.getGlyphName(reader.readValue(self.typecode, self.staticSize))

    def writeArray(self, writer, font, tableDict, values):
        writer.writeArray(self.typecode, font.getGlyphIDMany(values))

    def write(self, writer, font, tableDict, value, repeatIndex=None):
        writer.writeValue(self.typecode, font.getGlyphID(value))


class GlyphID32(GlyphID):
    staticSize = 4
    typecode = "L"


class NameID(UShort):
    def xmlWrite(self, xmlWriter, font, value, name, attrs):
        xmlWriter.simpletag(name, attrs + [("value", value)])
        if font and value:
            nameTable = font.get("name")
            if nameTable:
                name = nameTable.getDebugName(value)
                xmlWriter.write("  ")
                if name:
                    xmlWriter.comment(name)
                elif value == 0xFFFF:
                    xmlWriter.comment("None")
                else:
                    xmlWriter.comment("missing from name table")
                    log.warning("name id %d missing from name table" % value)
        xmlWriter.newline()


class STATFlags(UShort):
    def xmlWrite(self, xmlWriter, font, value, name, attrs):
        xmlWriter.simpletag(name, attrs + [("value", value)])
        flags = []
        if value & 0x01:
            flags.append("OlderSiblingFontAttribute")
        if value & 0x02:
            flags.append("ElidableAxisValueName")
        if flags:
            xmlWriter.write("  ")
            xmlWriter.comment(" ".join(flags))
        xmlWriter.newline()


class FloatValue(SimpleValue):
    @staticmethod
    def fromString(value):
        return float(value)


class DeciPoints(FloatValue):
    staticSize = 2

    def read(self, reader, font, tableDict):
        return reader.readUShort() / 10

    def write(self, writer, font, tableDict, value, repeatIndex=None):
        writer.writeUShort(round(value * 10))


class BaseFixedValue(FloatValue):
    staticSize = NotImplemented
    precisionBits = NotImplemented
    readerMethod = NotImplemented
    writerMethod = NotImplemented

    def read(self, reader, font, tableDict):
        return self.fromInt(getattr(reader, self.readerMethod)())

    def write(self, writer, font, tableDict, value, repeatIndex=None):
        getattr(writer, self.writerMethod)(self.toInt(value))

    @classmethod
    def fromInt(cls, value):
        return fi2fl(value, cls.precisionBits)

    @classmethod
    def toInt(cls, value):
        return fl2fi(value, cls.precisionBits)

    @classmethod
    def fromString(cls, value):
        return str2fl(value, cls.precisionBits)

    @classmethod
    def toString(cls, value):
        return fl2str(value, cls.precisionBits)


class Fixed(BaseFixedValue):
    staticSize = 4
    precisionBits = 16
    readerMethod = "readLong"
    writerMethod = "writeLong"


class F2Dot14(BaseFixedValue):
    staticSize = 2
    precisionBits = 14
    readerMethod = "readShort"
    writerMethod = "writeShort"


class Angle(F2Dot14):
    # angles are specified in degrees, and encoded as F2Dot14 fractions of half
    # circle: e.g. 1.0 => 180, -0.5 => -90, -2.0 => -360, etc.
    bias = 0.0
    factor = 1.0 / (1 << 14) * 180  # 0.010986328125

    @classmethod
    def fromInt(cls, value):
        return (super().fromInt(value) + cls.bias) * 180

    @classmethod
    def toInt(cls, value):
        return super().toInt((value / 180) - cls.bias)

    @classmethod
    def fromString(cls, value):
        # quantize to nearest multiples of minimum fixed-precision angle
        return otRound(float(value) / cls.factor) * cls.factor

    @classmethod
    def toString(cls, value):
        return nearestMultipleShortestRepr(value, cls.factor)


class BiasedAngle(Angle):
    # A bias of 1.0 is used in the representation of start and end angles
    # of COLRv1 PaintSweepGradients to allow for encoding +360deg
    bias = 1.0


class Version(SimpleValue):
    staticSize = 4

    def read(self, reader, font, tableDict):
        value = reader.readLong()
        return value

    def write(self, writer, font, tableDict, value, repeatIndex=None):
        value = fi2ve(value)
        writer.writeLong(value)

    @staticmethod
    def fromString(value):
        return ve2fi(value)

    @staticmethod
    def toString(value):
        return "0x%08x" % value

    @staticmethod
    def fromFloat(v):
        return fl2fi(v, 16)


class Char64(SimpleValue):
    """An ASCII string with up to 64 characters.

    Unused character positions are filled with 0x00 bytes.
    Used in Apple AAT fonts in the `gcid` table.
    """

    staticSize = 64

    def read(self, reader, font, tableDict):
        data = reader.readData(self.staticSize)
        zeroPos = data.find(b"\0")
        if zeroPos >= 0:
            data = data[:zeroPos]
        s = tostr(data, encoding="ascii", errors="replace")
        if s != tostr(data, encoding="ascii", errors="ignore"):
            log.warning('replaced non-ASCII characters in "%s"' % s)
        return s

    def write(self, writer, font, tableDict, value, repeatIndex=None):
        data = tobytes(value, encoding="ascii", errors="replace")
        if data != tobytes(value, encoding="ascii", errors="ignore"):
            log.warning('replacing non-ASCII characters in "%s"' % value)
        if len(data) > self.staticSize:
            log.warning(
                'truncating overlong "%s" to %d bytes' % (value, self.staticSize)
            )
        data = (data + b"\0" * self.staticSize)[: self.staticSize]
        writer.writeData(data)


class Struct(BaseConverter):
    def getRecordSize(self, reader):
        return self.tableClass and self.tableClass.getRecordSize(reader)

    def read(self, reader, font, tableDict):
        table = self.tableClass()
        table.decompile(reader, font)
        return table

    def write(self, writer, font, tableDict, value, repeatIndex=None):
        value.compile(writer, font)

    def xmlWrite(self, xmlWriter, font, value, name, attrs):
        if value is None:
            if attrs:
                # If there are attributes (probably index), then
                # don't drop this even if it's NULL.  It will mess
                # up the array indices of the containing element.
                xmlWriter.simpletag(name, attrs + [("empty", 1)])
                xmlWriter.newline()
            else:
                pass  # NULL table, ignore
        else:
            value.toXML(xmlWriter, font, attrs, name=name)

    def xmlRead(self, attrs, content, font):
        if "empty" in attrs and safeEval(attrs["empty"]):
            return None
        table = self.tableClass()
        Format = attrs.get("Format")
        if Format is not None:
            table.Format = int(Format)

        noPostRead = not hasattr(table, "postRead")
        if noPostRead:
            # TODO Cache table.hasPropagated.
            cleanPropagation = False
            for conv in table.getConverters():
                if conv.isPropagated:
                    cleanPropagation = True
                    if not hasattr(font, "_propagator"):
                        font._propagator = {}
                    propagator = font._propagator
                    assert conv.name not in propagator, (conv.name, propagator)
                    setattr(table, conv.name, None)
                    propagator[conv.name] = CountReference(table.__dict__, conv.name)

        for element in content:
            if isinstance(element, tuple):
                name, attrs, content = element
                table.fromXML(name, attrs, content, font)
            else:
                pass

        table.populateDefaults(propagator=getattr(font, "_propagator", None))

        if noPostRead:
            if cleanPropagation:
                for conv in table.getConverters():
                    if conv.isPropagated:
                        propagator = font._propagator
                        del propagator[conv.name]
                        if not propagator:
                            del font._propagator

        return table

    def __repr__(self):
        return "Struct of " + repr(self.tableClass)


class StructWithLength(Struct):
    def read(self, reader, font, tableDict):
        pos = reader.pos
        table = self.tableClass()
        table.decompile(reader, font)
        reader.seek(pos + table.StructLength)
        return table

    def write(self, writer, font, tableDict, value, repeatIndex=None):
        for convIndex, conv in enumerate(value.getConverters()):
            if conv.name == "StructLength":
                break
        lengthIndex = len(writer.items) + convIndex
        if isinstance(value, FormatSwitchingBaseTable):
            lengthIndex += 1  # implicit Format field
        deadbeef = {1: 0xDE, 2: 0xDEAD, 4: 0xDEADBEEF}[conv.staticSize]

        before = writer.getDataLength()
        value.StructLength = deadbeef
        value.compile(writer, font)
        length = writer.getDataLength() - before
        lengthWriter = writer.getSubWriter()
        conv.write(lengthWriter, font, tableDict, length)
        assert writer.items[lengthIndex] == b"\xde\xad\xbe\xef"[: conv.staticSize]
        writer.items[lengthIndex] = lengthWriter.getAllData()


class Table(Struct):
    staticSize = 2

    def readOffset(self, reader):
        return reader.readUShort()

    def writeNullOffset(self, writer):
        writer.writeUShort(0)

    def read(self, reader, font, tableDict):
        offset = self.readOffset(reader)
        if offset == 0:
            return None
        table = self.tableClass()
        reader = reader.getSubReader(offset)
        if font.lazy:
            table.reader = reader
            table.font = font
        else:
            table.decompile(reader, font)
        return table

    def write(self, writer, font, tableDict, value, repeatIndex=None):
        if value is None:
            self.writeNullOffset(writer)
        else:
            subWriter = writer.getSubWriter()
            subWriter.name = self.name
            if repeatIndex is not None:
                subWriter.repeatIndex = repeatIndex
            writer.writeSubTable(subWriter, offsetSize=self.staticSize)
            value.compile(subWriter, font)


class LTable(Table):
    staticSize = 4

    def readOffset(self, reader):
        return reader.readULong()

    def writeNullOffset(self, writer):
        writer.writeULong(0)


# Table pointed to by a 24-bit, 3-byte long offset
class Table24(Table):
    staticSize = 3

    def readOffset(self, reader):
        return reader.readUInt24()

    def writeNullOffset(self, writer):
        writer.writeUInt24(0)


# TODO Clean / merge the SubTable and SubStruct


class SubStruct(Struct):
    def getConverter(self, tableType, lookupType):
        tableClass = self.lookupTypes[tableType][lookupType]
        return self.__class__(self.name, self.repeat, self.aux, tableClass)

    def xmlWrite(self, xmlWriter, font, value, name, attrs):
        super(SubStruct, self).xmlWrite(xmlWriter, font, value, None, attrs)


class SubTable(Table):
    def getConverter(self, tableType, lookupType):
        tableClass = self.lookupTypes[tableType][lookupType]
        return self.__class__(self.name, self.repeat, self.aux, tableClass)

    def xmlWrite(self, xmlWriter, font, value, name, attrs):
        super(SubTable, self).xmlWrite(xmlWriter, font, value, None, attrs)


class ExtSubTable(LTable, SubTable):
    def write(self, writer, font, tableDict, value, repeatIndex=None):
        writer.Extension = True  # actually, mere presence of the field flags it as an Ext Subtable writer.
        Table.write(self, writer, font, tableDict, value, repeatIndex)


