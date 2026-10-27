		flags: B  # table type and flags
"""

woff2FlagsSize = sstruct.calcsize(woff2FlagsFormat)

woff2UnknownTagFormat = """
		> # big endian
		tag: 4s  # 4-byte tag (optional)
"""

woff2UnknownTagSize = sstruct.calcsize(woff2UnknownTagFormat)

woff2UnknownTagIndex = 0x3F

woff2Base128MaxSize = 5
woff2DirectoryEntryMaxSize = (
    woff2FlagsSize + woff2UnknownTagSize + 2 * woff2Base128MaxSize
)

woff2TransformedTableTags = ("glyf", "loca")

woff2GlyfTableFormat = """
		> # big endian
		version:                  H  # = 0x0000
		optionFlags:              H  # Bit 0: we have overlapSimpleBitmap[], Bits 1-15: reserved
		numGlyphs:                H  # Number of glyphs
		indexFormat:              H  # Offset format for loca table
		nContourStreamSize:       L  # Size of nContour stream
		nPointsStreamSize:        L  # Size of nPoints stream
		flagStreamSize:           L  # Size of flag stream
		glyphStreamSize:          L  # Size of glyph stream
		compositeStreamSize:      L  # Size of composite stream
		bboxStreamSize:           L  # Comnined size of bboxBitmap and bboxStream
		instructionStreamSize:    L  # Size of instruction stream
"""

woff2GlyfTableFormatSize = sstruct.calcsize(woff2GlyfTableFormat)

bboxFormat = """
		>	# big endian
		xMin:				h
		yMin:				h
		xMax:				h
		yMax:				h
"""

woff2OverlapSimpleBitmapFlag = 0x0001


def getKnownTagIndex(tag):
    """Return index of 'tag' in woff2KnownTags list. Return 63 if not found."""
    try:
        return woff2KnownTags.index(tag)
    except ValueError:
        return woff2UnknownTagIndex


class WOFF2DirectoryEntry(DirectoryEntry):
    def fromFile(self, file):
        pos = file.tell()
        data = file.read(woff2DirectoryEntryMaxSize)
        left = self.fromString(data)
        consumed = len(data) - len(left)
        file.seek(pos + consumed)

    def fromString(self, data):
        if len(data) < 1:
            raise TTLibError("can't read table 'flags': not enough data")
        dummy, data = sstruct.unpack2(woff2FlagsFormat, data, self)
        if self.flags & 0x3F == 0x3F:
            # if bits [0..5] of the flags byte == 63, read a 4-byte arbitrary tag value
            if len(data) < woff2UnknownTagSize:
                raise TTLibError("can't read table 'tag': not enough data")
            dummy, data = sstruct.unpack2(woff2UnknownTagFormat, data, self)
        else:
            # otherwise, tag is derived from a fixed 'Known Tags' table
            self.tag = woff2KnownTags[self.flags & 0x3F]
        self.tag = Tag(self.tag)
        self.origLength, data = unpackBase128(data)
        self.length = self.origLength
        if self.transformed:
            self.length, data = unpackBase128(data)
            if self.tag == "loca" and self.length != 0:
                raise TTLibError("the transformLength of the 'loca' table must be 0")
        # return left over data
        return data

    def toString(self):
        data = bytechr(self.flags)
        if (self.flags & 0x3F) == 0x3F:
            data += struct.pack(">4s", self.tag.tobytes())
        data += packBase128(self.origLength)
        if self.transformed:
            data += packBase128(self.length)
        return data

    @property
    def transformVersion(self):
        """Return bits 6-7 of table entry's flags, which indicate the preprocessing
        transformation version number (between 0 and 3).
        """
        return self.flags >> 6

    @transformVersion.setter
    def transformVersion(self, value):
        assert 0 <= value <= 3
        self.flags |= value << 6

    @property
    def transformed(self):
        """Return True if the table has any transformation, else return False."""
        # For all tables in a font, except for 'glyf' and 'loca', the transformation
        # version 0 indicates the null transform (where the original table data is
        # passed directly to the Brotli compressor). For 'glyf' and 'loca' tables,
        # transformation version 3 indicates the null transform
        if self.tag in {"glyf", "loca"}:
            return self.transformVersion != 3
        else:
            return self.transformVersion != 0

    @transformed.setter
    def transformed(self, booleanValue):
        # here we assume that a non-null transform means version 0 for 'glyf' and
        # 'loca' and 1 for every other table (e.g. hmtx); but that may change as
        # new transformation formats are introduced in the future (if ever).
        if self.tag in {"glyf", "loca"}:
            self.transformVersion = 3 if not booleanValue else 0
        else:
            self.transformVersion = int(booleanValue)


class WOFF2LocaTable(getTableClass("loca")):
    """Same as parent class. The only difference is that it attempts to preserve
    the 'indexFormat' as encoded in the WOFF2 glyf table.
    """

    def __init__(self, tag=None):
        self.tableTag = Tag(tag or "loca")

    def compile(self, ttFont):
        try:
            max_location = max(self.locations)
        except AttributeError:
            self.set([])
            max_location = 0
        if "glyf" in ttFont and hasattr(ttFont["glyf"], "indexFormat"):
            # copile loca using the indexFormat specified in the WOFF2 glyf table
            indexFormat = ttFont["glyf"].indexFormat
            if indexFormat == 0:
                if max_location >= 0x20000:
                    raise TTLibError("indexFormat is 0 but local offsets > 0x20000")
                if not all(l % 2 == 0 for l in self.locations):
                    raise TTLibError(
                        "indexFormat is 0 but local offsets not multiples of 2"
                    )
                locations = array.array("H")
                for location in self.locations:
                    locations.append(location // 2)
            else:
                locations = array.array("I", self.locations)
            if sys.byteorder != "big":
                locations.byteswap()
            data = locations.tobytes()
        else:
            # use the most compact indexFormat given the current glyph offsets
            data = super(WOFF2LocaTable, self).compile(ttFont)
        return data


class WOFF2GlyfTable(getTableClass("glyf")):
    """Decoder/Encoder for WOFF2 'glyf' table transform."""

    subStreams = (
        "nContourStream",
        "nPointsStream",
        "flagStream",
        "glyphStream",
        "compositeStream",
        "bboxStream",
        "instructionStream",
    )

    def __init__(self, tag=None):
        self.tableTag = Tag(tag or "glyf")

    def reconstruct(self, data, ttFont):
        """Decompile transformed 'glyf' data."""
        inputDataSize = len(data)

        if inputDataSize < woff2GlyfTableFormatSize:
            raise TTLibError("not enough 'glyf' data")
        dummy, data = sstruct.unpack2(woff2GlyfTableFormat, data, self)
        offset = woff2GlyfTableFormatSize

        for stream in self.subStreams:
            size = getattr(self, stream + "Size")
            setattr(self, stream, data[:size])
            data = data[size:]
            offset += size

        hasOverlapSimpleBitmap = self.optionFlags & woff2OverlapSimpleBitmapFlag
        self.overlapSimpleBitmap = None
        if hasOverlapSimpleBitmap:
            overlapSimpleBitmapSize = (self.numGlyphs + 7) >> 3
            self.overlapSimpleBitmap = array.array("B", data[:overlapSimpleBitmapSize])
            offset += overlapSimpleBitmapSize

        if offset != inputDataSize:
            raise TTLibError(
                "incorrect size of transformed 'glyf' table: expected %d, received %d bytes"
                % (offset, inputDataSize)
            )

        bboxBitmapSize = ((self.numGlyphs + 31) >> 5) << 2
        bboxBitmap = self.bboxStream[:bboxBitmapSize]
        self.bboxBitmap = array.array("B", bboxBitmap)
        self.bboxStream = self.bboxStream[bboxBitmapSize:]

        self.nContourStream = array.array("h", self.nContourStream)
        if sys.byteorder != "big":
            self.nContourStream.byteswap()
        assert len(self.nContourStream) == self.numGlyphs

        if "head" in ttFont:
            ttFont["head"].indexToLocFormat = self.indexFormat
        try:
            self.glyphOrder = ttFont.getGlyphOrder()
        except:
            self.glyphOrder = None
        if self.glyphOrder is None:
            self.glyphOrder = [".notdef"]
            self.glyphOrder.extend(["glyph%.5d" % i for i in range(1, self.numGlyphs)])
        else:
            if len(self.glyphOrder) != self.numGlyphs:
                raise TTLibError(
                    "incorrect glyphOrder: expected %d glyphs, found %d"
                    % (len(self.glyphOrder), self.numGlyphs)
                )

        glyphs = self.glyphs = {}
        for glyphID, glyphName in enumerate(self.glyphOrder):
            glyph = self._decodeGlyph(glyphID)
            glyphs[glyphName] = glyph

    def transform(self, ttFont):
        """Return transformed 'glyf' data"""
        self.numGlyphs = len(self.glyphs)
        assert len(self.glyphOrder) == self.numGlyphs
        if "maxp" in ttFont:
            ttFont["maxp"].numGlyphs = self.numGlyphs
        self.indexFormat = ttFont["head"].indexToLocFormat

        for stream in self.subStreams:
            setattr(self, stream, b"")
        bboxBitmapSize = ((self.numGlyphs + 31) >> 5) << 2
        self.bboxBitmap = array.array("B", [0] * bboxBitmapSize)

        self.overlapSimpleBitmap = array.array("B", [0] * ((self.numGlyphs + 7) >> 3))
        for glyphID in range(self.numGlyphs):
            try:
                self._encodeGlyph(glyphID)
            except NotImplementedError:
                return None
        hasOverlapSimpleBitmap = any(self.overlapSimpleBitmap)

        self.bboxStream = self.bboxBitmap.tobytes() + self.bboxStream
        for stream in self.subStreams:
            setattr(self, stream + "Size", len(getattr(self, stream)))
        self.version = 0
        self.optionFlags = 0
        if hasOverlapSimpleBitmap:
            self.optionFlags |= woff2OverlapSimpleBitmapFlag
        data = sstruct.pack(woff2GlyfTableFormat, self)
        data += bytesjoin([getattr(self, s) for s in self.subStreams])
        if hasOverlapSimpleBitmap:
            data += self.overlapSimpleBitmap.tobytes()
        return data

    def _decodeGlyph(self, glyphID):
        glyph = getTableModule("glyf").Glyph()
        glyph.numberOfContours = self.nContourStream[glyphID]
        if glyph.numberOfContours == 0:
            return glyph
        elif glyph.isComposite():
            self._decodeComponents(glyph)
        else:
            self._decodeCoordinates(glyph)
            self._decodeOverlapSimpleFlag(glyph, glyphID)
        self._decodeBBox(glyphID, glyph)
        return glyph

    def _decodeComponents(self, glyph):
        data = self.compositeStream
        glyph.components = []
        more = 1
        haveInstructions = 0
        while more:
            component = getTableModule("glyf").GlyphComponent()
            more, haveInstr, data = component.decompile(data, self)
            haveInstructions = haveInstructions | haveInstr
            glyph.components.append(component)
        self.compositeStream = data
        if haveInstructions:
            self._decodeInstructions(glyph)

    def _decodeCoordinates(self, glyph):
        data = self.nPointsStream
        endPtsOfContours = []
        endPoint = -1
        for i in range(glyph.numberOfContours):
            ptsOfContour, data = unpack255UShort(data)
            endPoint += ptsOfContour
            endPtsOfContours.append(endPoint)
        glyph.endPtsOfContours = endPtsOfContours
        self.nPointsStream = data
        self._decodeTriplets(glyph)
        self._decodeInstructions(glyph)

    def _decodeOverlapSimpleFlag(self, glyph, glyphID):
        if self.overlapSimpleBitmap is None or glyph.numberOfContours <= 0:
            return
        byte = glyphID >> 3
        bit = glyphID & 7
        if self.overlapSimpleBitmap[byte] & (0x80 >> bit):
            glyph.flags[0] |= _g_l_y_f.flagOverlapSimple

    def _decodeInstructions(self, glyph):
        glyphStream = self.glyphStream
        instructionStream = self.instructionStream
        instructionLength, glyphStream = unpack255UShort(glyphStream)
        glyph.program = ttProgram.Program()
        glyph.program.fromBytecode(instructionStream[:instructionLength])
        self.glyphStream = glyphStream
        self.instructionStream = instructionStream[instructionLength:]

    def _decodeBBox(self, glyphID, glyph):
        haveBBox = bool(self.bboxBitmap[glyphID >> 3] & (0x80 >> (glyphID & 7)))
        if glyph.isComposite() and not haveBBox:
            raise TTLibError("no bbox values for composite glyph %d" % glyphID)
        if haveBBox:
            dummy, self.bboxStream = sstruct.unpack2(bboxFormat, self.bboxStream, glyph)
        else:
            glyph.recalcBounds(self)

    def _decodeTriplets(self, glyph):
        def withSign(flag, baseval):
            assert 0 <= baseval and baseval < 65536, "integer overflow"
            return baseval if flag & 1 else -baseval

        nPoints = glyph.endPtsOfContours[-1] + 1
        flagSize = nPoints
        if flagSize > len(self.flagStream):
            raise TTLibError("not enough 'flagStream' data")
        flagsData = self.flagStream[:flagSize]
        self.flagStream = self.flagStream[flagSize:]
        flags = array.array("B", flagsData)

        triplets = array.array("B", self.glyphStream)
        nTriplets = len(triplets)
        assert nPoints <= nTriplets

        x = 0
        y = 0
        glyph.coordinates = getTableModule("glyf").GlyphCoordinates.zeros(nPoints)
        glyph.flags = array.array("B")
        tripletIndex = 0
        for i in range(nPoints):
            flag = flags[i]
            onCurve = not bool(flag >> 7)
            flag &= 0x7F
            if flag < 84:
                nBytes = 1
            elif flag < 120:
                nBytes = 2
            elif flag < 124:
                nBytes = 3
            else:
                nBytes = 4
            assert (tripletIndex + nBytes) <= nTriplets
            if flag < 10:
                dx = 0
                dy = withSign(flag, ((flag & 14) << 7) + triplets[tripletIndex])
            elif flag < 20:
                dx = withSign(flag, (((flag - 10) & 14) << 7) + triplets[tripletIndex])
                dy = 0
            elif flag < 84:
                b0 = flag - 20
                b1 = triplets[tripletIndex]
                dx = withSign(flag, 1 + (b0 & 0x30) + (b1 >> 4))
                dy = withSign(flag >> 1, 1 + ((b0 & 0x0C) << 2) + (b1 & 0x0F))
            elif flag < 120:
                b0 = flag - 84
                dx = withSign(flag, 1 + ((b0 // 12) << 8) + triplets[tripletIndex])
                dy = withSign(
                    flag >> 1, 1 + (((b0 % 12) >> 2) << 8) + triplets[tripletIndex + 1]
                )
            elif flag < 124:
                b2 = triplets[tripletIndex + 1]
                dx = withSign(flag, (triplets[tripletIndex] << 4) + (b2 >> 4))
                dy = withSign(
                    flag >> 1, ((b2 & 0x0F) << 8) + triplets[tripletIndex + 2]
                )
            else:
                dx = withSign(
                    flag, (triplets[tripletIndex] << 8) + triplets[tripletIndex + 1]
                )
                dy = withSign(
                    flag >> 1,
                    (triplets[tripletIndex + 2] << 8) + triplets[tripletIndex + 3],
                )
            tripletIndex += nBytes
            x += dx
            y += dy
            glyph.coordinates[i] = (x, y)
            glyph.flags.append(int(onCurve))
        bytesConsumed = tripletIndex
        self.glyphStream = self.glyphStream[bytesConsumed:]

    def _encodeGlyph(self, glyphID):
        glyphName = self.getGlyphName(glyphID)
        glyph = self[glyphName]
        self.nContourStream += struct.pack(">h", glyph.numberOfContours)
        if glyph.numberOfContours == 0:
            return
        elif glyph.isComposite():
            self._encodeComponents(glyph)
        else:
            self._encodeCoordinates(glyph)
            self._encodeOverlapSimpleFlag(glyph, glyphID)
        self._encodeBBox(glyphID, glyph)

    def _encodeComponents(self, glyph):
        lastcomponent = len(glyph.components) - 1
        more = 1
        haveInstructions = 0
        for i, component in enumerate(glyph.components):
            if i == lastcomponent:
                haveInstructions = hasattr(glyph, "program")
                more = 0
            self.compositeStream += component.compile(more, haveInstructions, self)
        if haveInstructions:
            self._encodeInstructions(glyph)

    def _encodeCoordinates(self, glyph):
        lastEndPoint = -1
        if _g_l_y_f.flagCubic in glyph.flags:
            raise NotImplementedError
        for endPoint in glyph.endPtsOfContours:
            ptsOfContour = endPoint - lastEndPoint
            self.nPointsStream += pack255UShort(ptsOfContour)
            lastEndPoint = endPoint
        self._encodeTriplets(glyph)
        self._encodeInstructions(glyph)

    def _encodeOverlapSimpleFlag(self, glyph, glyphID):
        if glyph.numberOfContours <= 0:
            return
        if glyph.flags[0] & _g_l_y_f.flagOverlapSimple:
            byte = glyphID >> 3
            bit = glyphID & 7
            self.overlapSimpleBitmap[byte] |= 0x80 >> bit

    def _encodeInstructions(self, glyph):
        instructions = glyph.program.getBytecode()
        self.glyphStream += pack255UShort(len(instructions))
        self.instructionStream += instructions

    def _encodeBBox(self, glyphID, glyph):
        assert glyph.numberOfContours != 0, "empty glyph has no bbox"
        if not glyph.isComposite():
            # for simple glyphs, compare the encoded bounding box info with the calculated
            # values, and if they match omit the bounding box info
            currentBBox = glyph.xMin, glyph.yMin, glyph.xMax, glyph.yMax
            calculatedBBox = calcIntBounds(glyph.coordinates)
            if currentBBox == calculatedBBox:
                return
        self.bboxBitmap[glyphID >> 3] |= 0x80 >> (glyphID & 7)
        self.bboxStream += sstruct.pack(bboxFormat, glyph)

    def _encodeTriplets(self, glyph):
        assert len(glyph.coordinates) == len(glyph.flags)
        coordinates = glyph.coordinates.copy()
        coordinates.absoluteToRelative()

        flags = array.array("B")
        triplets = array.array("B")
        for i, (x, y) in enumerate(coordinates):
            onCurve = glyph.flags[i] & _g_l_y_f.flagOnCurve
            absX = abs(x)
            absY = abs(y)
            onCurveBit = 0 if onCurve else 128
            xSignBit = 0 if (x < 0) else 1
            ySignBit = 0 if (y < 0) else 1
            xySignBits = xSignBit + 2 * ySignBit

            if x == 0 and absY < 1280:
                flags.append(onCurveBit + ((absY & 0xF00) >> 7) + ySignBit)
                triplets.append(absY & 0xFF)
            elif y == 0 and absX < 1280:
                flags.append(onCurveBit + 10 + ((absX & 0xF00) >> 7) + xSignBit)
                triplets.append(absX & 0xFF)
            elif absX < 65 and absY < 65:
                flags.append(
                    onCurveBit
                    + 20
                    + ((absX - 1) & 0x30)
                    + (((absY - 1) & 0x30) >> 2)
                    + xySignBits
                )
                triplets.append((((absX - 1) & 0xF) << 4) | ((absY - 1) & 0xF))
            elif absX < 769 and absY < 769:
                flags.append(
                    onCurveBit
                    + 84
                    + 12 * (((absX - 1) & 0x300) >> 8)
                    + (((absY - 1) & 0x300) >> 6)
                    + xySignBits
                )
                triplets.append((absX - 1) & 0xFF)
                triplets.append((absY - 1) & 0xFF)
            elif absX < 4096 and absY < 4096:
                flags.append(onCurveBit + 120 + xySignBits)
                triplets.append(absX >> 4)
                triplets.append(((absX & 0xF) << 4) | (absY >> 8))
                triplets.append(absY & 0xFF)
            else:
                flags.append(onCurveBit + 124 + xySignBits)
                triplets.append(absX >> 8)
                triplets.append(absX & 0xFF)
                triplets.append(absY >> 8)
                triplets.append(absY & 0xFF)

        self.flagStream += flags.tobytes()
        self.glyphStream += triplets.tobytes()


class WOFF2HmtxTable(getTableClass("hmtx")):
    def __init__(self, tag=None):
        self.tableTag = Tag(tag or "hmtx")

    def reconstruct(self, data, ttFont):
        (flags,) = struct.unpack(">B", data[:1])
        data = data[1:]
        if flags & 0b11111100 != 0:
            raise TTLibError("Bits 2-7 of '%s' flags are reserved" % self.tableTag)

        # When bit 0 is _not_ set, the lsb[] array is present
        hasLsbArray = flags & 1 == 0
        # When bit 1 is _not_ set, the leftSideBearing[] array is present
        hasLeftSideBearingArray = flags & 2 == 0
        if hasLsbArray and hasLeftSideBearingArray:
            raise TTLibError(
                "either bits 0 or 1 (or both) must set in transformed '%s' flags"
                % self.tableTag
            )

        glyfTable = ttFont["glyf"]
        headerTable = ttFont["hhea"]
        glyphOrder = glyfTable.glyphOrder
        numGlyphs = len(glyphOrder)
        numberOfHMetrics = min(int(headerTable.numberOfHMetrics), numGlyphs)

        assert len(data) >= 2 * numberOfHMetrics
        advanceWidthArray = array.array("H", data[: 2 * numberOfHMetrics])
        if sys.byteorder != "big":
            advanceWidthArray.byteswap()
        data = data[2 * numberOfHMetrics :]

        if hasLsbArray:
            assert len(data) >= 2 * numberOfHMetrics
            lsbArray = array.array("h", data[: 2 * numberOfHMetrics])
            if sys.byteorder != "big":
                lsbArray.byteswap()
            data = data[2 * numberOfHMetrics :]
        else:
            # compute (proportional) glyphs' lsb from their xMin
            lsbArray = array.array("h")
            for i, glyphName in enumerate(glyphOrder):
                if i >= numberOfHMetrics:
                    break
                glyph = glyfTable[glyphName]
                xMin = getattr(glyph, "xMin", 0)
                lsbArray.append(xMin)

        numberOfSideBearings = numGlyphs - numberOfHMetrics
        if hasLeftSideBearingArray:
            assert len(data) >= 2 * numberOfSideBearings
            leftSideBearingArray = array.array("h", data[: 2 * numberOfSideBearings])
            if sys.byteorder != "big":
                leftSideBearingArray.byteswap()
            data = data[2 * numberOfSideBearings :]
        else:
            # compute (monospaced) glyphs' leftSideBearing from their xMin
            leftSideBearingArray = array.array("h")
            for i, glyphName in enumerate(glyphOrder):
                if i < numberOfHMetrics:
                    continue
                glyph = glyfTable[glyphName]
                xMin = getattr(glyph, "xMin", 0)
                leftSideBearingArray.append(xMin)

        if data:
            raise TTLibError("too much '%s' table data" % self.tableTag)

        self.metrics = {}
        for i in range(numberOfHMetrics):
            glyphName = glyphOrder[i]
            advanceWidth, lsb = advanceWidthArray[i], lsbArray[i]
            self.metrics[glyphName] = (advanceWidth, lsb)
        lastAdvance = advanceWidthArray[-1]
        for i in range(numberOfSideBearings):
            glyphName = glyphOrder[i + numberOfHMetrics]
            self.metrics[glyphName] = (lastAdvance, leftSideBearingArray[i])

    def transform(self, ttFont):
        glyphOrder = ttFont.getGlyphOrder()
        glyf = ttFont["glyf"]
        hhea = ttFont["hhea"]
        numberOfHMetrics = hhea.numberOfHMetrics

        # check if any of the proportional glyphs has left sidebearings that
        # differ from their xMin bounding box values.
        hasLsbArray = False
        for i in range(numberOfHMetrics):
            glyphName = glyphOrder[i]
            lsb = self.metrics[glyphName][1]
            if lsb != getattr(glyf[glyphName], "xMin", 0):
                hasLsbArray = True
                break

        # do the same for the monospaced glyphs (if any) at the end of hmtx table
        hasLeftSideBearingArray = False
        for i in range(numberOfHMetrics, len(glyphOrder)):
            glyphName = glyphOrder[i]
            lsb = self.metrics[glyphName][1]
            if lsb != getattr(glyf[glyphName], "xMin", 0):
                hasLeftSideBearingArray = True
                break

        # if we need to encode both sidebearings arrays, then no transformation is
        # applicable, and we must use the untransformed hmtx data
        if hasLsbArray and hasLeftSideBearingArray:
            return

        # set bit 0 and 1 when the respective arrays are _not_ present
        flags = 0
        if not hasLsbArray:
            flags |= 1 << 0
        if not hasLeftSideBearingArray:
            flags |= 1 << 1

        data = struct.pack(">B", flags)

        advanceWidthArray = array.array(
            "H",
            [
                self.metrics[glyphName][0]
                for i, glyphName in enumerate(glyphOrder)
                if i < numberOfHMetrics
            ],
        )
        if sys.byteorder != "big":
            advanceWidthArray.byteswap()
        data += advanceWidthArray.tobytes()

        if hasLsbArray:
            lsbArray = array.array(
                "h",
                [
                    self.metrics[glyphName][1]
                    for i, glyphName in enumerate(glyphOrder)
                    if i < numberOfHMetrics
                ],
            )
            if sys.byteorder != "big":
                lsbArray.byteswap()
            data += lsbArray.tobytes()

        if hasLeftSideBearingArray:
            leftSideBearingArray = array.array(
                "h",
                [
                    self.metrics[glyphOrder[i]][1]
                    for i in range(numberOfHMetrics, len(glyphOrder))
                ],
            )
            if sys.byteorder != "big":
                leftSideBearingArray.byteswap()
            data += leftSideBearingArray.tobytes()

        return data


class WOFF2FlavorData(WOFFFlavorData):
    Flavor = "woff2"

    def __init__(self, reader=None, data=None, transformedTables=None):
        """Data class that holds the WOFF2 header major/minor version, any
        metadata or private data (as bytes strings), and the set of
        table tags that have transformations applied (if reader is not None),
        or will have once the WOFF2 font is compiled.

        Args:
                reader: an SFNTReader (or subclass) object to read flavor data from.
                data: another WOFFFlavorData object to initialise data from.
                transformedTables: set of strings containing table tags to be transformed.

        Raises:
                ImportError if the brotli module is not installed.

        NOTE: The 'reader' argument, on the one hand, and the 'data' and
        'transformedTables' arguments, on the other hand, are mutually exclusive.
        """
        if not haveBrotli:
            raise ImportError("No module named brotli")

        if reader is not None:
            if data is not None:
                raise TypeError("'reader' and 'data' arguments are mutually exclusive")
            if transformedTables is not None:
                raise TypeError(
                    "'reader' and 'transformedTables' arguments are mutually exclusive"
                )

        if transformedTables is not None and (
            "glyf" in transformedTables
            and "loca" not in transformedTables
            or "loca" in transformedTables
            and "glyf" not in transformedTables
        ):
            raise ValueError("'glyf' and 'loca' must be transformed (or not) together")
        super(WOFF2FlavorData, self).__init__(reader=reader)
        if reader:
            transformedTables = [
                tag for tag, entry in reader.tables.items() if entry.transformed
            ]
        elif data:
            self.majorVersion = data.majorVersion
            self.majorVersion = data.minorVersion
            self.metaData = data.metaData
            self.privData = data.privData
            if transformedTables is None and hasattr(data, "transformedTables"):
                transformedTables = data.transformedTables

        if transformedTables is None:
            transformedTables = woff2TransformedTableTags

        self.transformedTables = set(transformedTables)

    def _decompress(self, rawData):
        return brotli.decompress(rawData)


def unpackBase128(data):
    r"""Read one to five bytes from UIntBase128-encoded input string, and return
    a tuple containing the decoded integer plus any leftover data.

    >>> unpackBase128(b'\x3f\x00\x00') == (63, b"\x00\x00")
    True
    >>> unpackBase128(b'\x8f\xff\xff\xff\x7f')[0] == 4294967295
    True
    >>> unpackBase128(b'\x80\x80\x3f')  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
      File "<stdin>", line 1, in ?
    TTLibError: UIntBase128 value must not start with leading zeros
    >>> unpackBase128(b'\x8f\xff\xff\xff\xff\x7f')[0]  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
      File "<stdin>", line 1, in ?
    TTLibError: UIntBase128-encoded sequence is longer than 5 bytes
    >>> unpackBase128(b'\x90\x80\x80\x80\x00')[0]  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
      File "<stdin>", line 1, in ?
    TTLibError: UIntBase128 value exceeds 2**32-1
    """
    if len(data) == 0:
        raise TTLibError("not enough data to unpack UIntBase128")
    result = 0
    if byteord(data[0]) == 0x80:
        # font must be rejected if UIntBase128 value starts with 0x80
        raise TTLibError("UIntBase128 value must not start with leading zeros")
    for i in range(woff2Base128MaxSize):
        if len(data) == 0:
            raise TTLibError("not enough data to unpack UIntBase128")
        code = byteord(data[0])
        data = data[1:]
        # if any of the top seven bits are set then we're about to overflow
        if result & 0xFE000000:
            raise TTLibError("UIntBase128 value exceeds 2**32-1")
        # set current value = old value times 128 bitwise-or (byte bitwise-and 127)
        result = (result << 7) | (code & 0x7F)
        # repeat until the most significant bit of byte is false
        if (code & 0x80) == 0:
            # return result plus left over data
            return result, data
    # make sure not to exceed the size bound
    raise TTLibError("UIntBase128-encoded sequence is longer than 5 bytes")


def base128Size(n):
    """Return the length in bytes of a UIntBase128-encoded sequence with value n.

    >>> base128Size(0)
    1
    >>> base128Size(24567)
    3
    >>> base128Size(2**32-1)
    5
    """
    assert n >= 0
    size = 1
    while n >= 128:
        size += 1
        n >>= 7
    return size


def packBase128(n):
    r"""Encode unsigned integer in range 0 to 2**32-1 (inclusive) to a string of
    bytes using UIntBase128 variable-length encoding. Produce the shortest possible
    encoding.

    >>> packBase128(63) == b"\x3f"
    True
    >>> packBase128(2**32-1) == b'\x8f\xff\xff\xff\x7f'
    True
    """
    if n < 0 or n >= 2**32:
        raise TTLibError("UIntBase128 format requires 0 <= integer <= 2**32-1")
    data = b""
    size = base128Size(n)
    for i in range(size):
        b = (n >> (7 * (size - i - 1))) & 0x7F
        if i < size - 1:
            b |= 0x80
        data += struct.pack("B", b)
    return data


def unpack255UShort(data):
    """Read one to three bytes from 255UInt16-encoded input string, and return a
    tuple containing the decoded integer plus any leftover data.

    >>> unpack255UShort(bytechr(252))[0]
    252

    Note that some numbers (e.g. 506) can have multiple encodings:
    >>> unpack255UShort(struct.pack("BB", 254, 0))[0]
    506
    >>> unpack255UShort(struct.pack("BB", 255, 253))[0]
    506
    >>> unpack255UShort(struct.pack("BBB", 253, 1, 250))[0]
    506
    """
    code = byteord(data[:1])
    data = data[1:]
    if code == 253:
        # read two more bytes as an unsigned short
        if len(data) < 2:
            raise TTLibError("not enough data to unpack 255UInt16")
        (result,) = struct.unpack(">H", data[:2])
        data = data[2:]
    elif code == 254:
        # read another byte, plus 253 * 2
        if len(data) == 0:
            raise TTLibError("not enough data to unpack 255UInt16")
        result = byteord(data[:1])
        result += 506
        data = data[1:]
    elif code == 255:
        # read another byte, plus 253
        if len(data) == 0:
            raise TTLibError("not enough data to unpack 255UInt16")
        result = byteord(data[:1])
        result += 253
        data = data[1:]
    else:
        # leave as is if lower than 253
        result = code
    # return result plus left over data
    return result, data


def pack255UShort(value):
    r"""Encode unsigned integer in range 0 to 65535 (inclusive) to a bytestring
    using 255UInt16 variable-length encoding.

    >>> pack255UShort(252) == b'\xfc'
    True
    >>> pack255UShort(506) == b'\xfe\x00'
    True
    >>> pack255UShort(762) == b'\xfd\x02\xfa'
    True
    """
    if value < 0 or value > 0xFFFF:
        raise TTLibError("255UInt16 format requires 0 <= integer <= 65535")
    if value < 253:
        return struct.pack(">B", value)
    elif value < 506:
        return struct.pack(">BB", 255, value - 253)
    elif value < 762:
        return struct.pack(">BB", 254, value - 506)
    else:
        return struct.pack(">BH", 253, value)


def compress(input_file, output_file, transform_tables=None):
    """Compress OpenType font to WOFF2.

    Args:
            input_file: a file path, file or file-like object (open in binary mode)
                    containing an OpenType font (either CFF- or TrueType-flavored).
            output_file: a file path, file or file-like object where to save the
                    compressed WOFF2 font.
            transform_tables: Optional[Iterable[str]]: a set of table tags for which
                    to enable preprocessing transformations. By default, only 'glyf'
                    and 'loca' tables are transformed. An empty set means disable all
                    transformations.
    """
    log.info("Processing %s => %s" % (input_file, output_file))

    font = TTFont(input_file, recalcBBoxes=False, recalcTimestamp=False)
    font.flavor = "woff2"

    if transform_tables is not None:
        font.flavorData = WOFF2FlavorData(
            data=font.flavorData, transformedTables=transform_tables
        )

    font.save(output_file, reorderTables=False)


def decompress(input_file, output_file):
    """Decompress WOFF2 font to OpenType font.

    Args:
            input_file: a file path, file or file-like object (open in binary mode)
                    containing a compressed WOFF2 font.
            output_file: a file path, file or file-like object where to save the
                    decompressed OpenType font.
    """
    log.info("Processing %s => %s" % (input_file, output_file))

    font = TTFont(input_file, recalcBBoxes=False, recalcTimestamp=False)
    font.flavor = None
    font.flavorData = None
    font.save(output_file, reorderTables=True)


def main(args=None):
    """Compress and decompress WOFF2 fonts"""
    import argparse
    from fontTools import configLogger
    from fontTools.ttx import makeOutputFileName

    class _HelpAction(argparse._HelpAction):
        def __call__(self, parser, namespace, values, option_string=None):
            subparsers_actions = [
                action
                for action in parser._actions
                if isinstance(action, argparse._SubParsersAction)
            ]
            for subparsers_action in subparsers_actions:
                for choice, subparser in subparsers_action.choices.items():
                    print(subparser.format_help())
            parser.exit()

    class _NoGlyfTransformAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            namespace.transform_tables.difference_update({"glyf", "loca"})

    class _HmtxTransformAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            namespace.transform_tables.add("hmtx")

    parser = argparse.ArgumentParser(
        prog="fonttools ttLib.woff2", description=main.__doc__, add_help=False
    )

    parser.add_argument(
        "-h", "--help", action=_HelpAction, help="show this help message and exit"
    )

    parser_group = parser.add_subparsers(title="sub-commands")
    parser_compress = parser_group.add_parser(
        "compress", description="Compress a TTF or OTF font to WOFF2"
    )
    parser_decompress = parser_group.add_parser(
        "decompress", description="Decompress a WOFF2 font to OTF"
    )

    for subparser in (parser_compress, parser_decompress):
        group = subparser.add_mutually_exclusive_group(required=False)
        group.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            help="print more messages to console",
        )
        group.add_argument(
            "-q",
            "--quiet",
            action="store_true",
            help="do not print messages to console",
        )

    parser_compress.add_argument(
        "input_file",
        metavar="INPUT",
        help="the input OpenType font (.ttf or .otf)",
    )
    parser_decompress.add_argument(
        "input_file",
        metavar="INPUT",
        help="the input WOFF2 font",
    )

    parser_compress.add_argument(
        "-o",
        "--output-file",
        metavar="OUTPUT",
        help="the output WOFF2 font",
    )
    parser_decompress.add_argument(
        "-o",
        "--output-file",
        metavar="OUTPUT",
        help="the output OpenType font",
    )

    transform_group = parser_compress.add_argument_group()
    transform_group.add_argument(
        "--no-glyf-transform",
        dest="transform_tables",
        nargs=0,
        action=_NoGlyfTransformAction,
        help="Do not transform glyf (and loca) tables",
    )
    transform_group.add_argument(
        "--hmtx-transform",
        dest="transform_tables",
        nargs=0,
        action=_HmtxTransformAction,
        help="Enable optional transformation for 'hmtx' table",
    )

    parser_compress.set_defaults(
        subcommand=compress,
        transform_tables={"glyf", "loca"},
    )
    parser_decompress.set_defaults(subcommand=decompress)

    options = vars(parser.parse_args(args))

    subcommand = options.pop("subcommand", None)
    if not subcommand:
        parser.print_help()
        return

    quiet = options.pop("quiet")
    verbose = options.pop("verbose")
    configLogger(
        level=("ERROR" if quiet else "DEBUG" if verbose else "INFO"),
    )

    if not options["output_file"]:
        if subcommand is compress:
            extension = ".woff2"
        elif subcommand is decompress:
            # choose .ttf/.otf file extension depending on sfntVersion
            with open(options["input_file"], "rb") as f:
                f.seek(4)  # skip 'wOF2' signature
                sfntVersion = f.read(4)
            assert len(sfntVersion) == 4, "not enough data"
            extension = ".otf" if sfntVersion == b"OTTO" else ".ttf"
        else:
            raise AssertionError(subcommand)
        options["output_file"] = makeOutputFileName(
            options["input_file"], outputDir=None, extension=extension
        )

    try:
        subcommand(**options)
    except TTLibError as e:
        parser.error(e)


if __name__ == "__main__":
    sys.exit(main())
<|endoftext|>"""Change the units-per-EM of a font.

AAT and Graphite tables are not supported. CFF/CFF2 fonts
are de-subroutinized."""

from fontTools.ttLib.ttVisitor import TTVisitor
import fontTools.ttLib as ttLib
import fontTools.ttLib.tables.otBase as otBase
import fontTools.ttLib.tables.otTables as otTables
from fontTools.cffLib import VarStoreData
import fontTools.cffLib.specializer as cffSpecializer
from fontTools.varLib import builder  # for VarData.calculateNumShorts
from fontTools.varLib.multiVarStore import OnlineMultiVarStoreBuilder
from fontTools.misc.vector import Vector
from fontTools.misc.fixedTools import otRound
from fontTools.misc.iterTools import batched

__all__ = ["scale_upem", "ScalerVisitor"]


class ScalerVisitor(TTVisitor):
    def __init__(self, scaleFactor):
        self.scaleFactor = scaleFactor

    def scale(self, v):
        return otRound(v * self.scaleFactor)


@ScalerVisitor.register_attrs(
    (
        (ttLib.getTableClass("head"), ("unitsPerEm", "xMin", "yMin", "xMax", "yMax")),
        (ttLib.getTableClass("post"), ("underlinePosition", "underlineThickness")),
        (ttLib.getTableClass("VORG"), ("defaultVertOriginY")),
        (
            ttLib.getTableClass("hhea"),
            (
                "ascent",
                "descent",
                "lineGap",
                "advanceWidthMax",
                "minLeftSideBearing",
                "minRightSideBearing",
                "xMaxExtent",
                "caretOffset",
            ),
        ),
        (
            ttLib.getTableClass("vhea"),
            (
                "ascent",
                "descent",
                "lineGap",
                "advanceHeightMax",
                "minTopSideBearing",
                "minBottomSideBearing",
                "yMaxExtent",
                "caretOffset",
            ),
        ),
        (
            ttLib.getTableClass("OS/2"),
            (
                "xAvgCharWidth",
                "ySubscriptXSize",
                "ySubscriptYSize",
                "ySubscriptXOffset",
                "ySubscriptYOffset",
                "ySuperscriptXSize",
                "ySuperscriptYSize",
                "ySuperscriptXOffset",
                "ySuperscriptYOffset",
                "yStrikeoutSize",
                "yStrikeoutPosition",
                "sTypoAscender",
                "sTypoDescender",
                "sTypoLineGap",
                "usWinAscent",
                "usWinDescent",
                "sxHeight",
                "sCapHeight",
            ),
        ),
        (
            otTables.ValueRecord,
            ("XAdvance", "YAdvance", "XPlacement", "YPlacement"),
        ),  # GPOS
        (otTables.Anchor, ("XCoordinate", "YCoordinate")),  # GPOS
        (otTables.CaretValue, ("Coordinate")),  # GDEF
        (otTables.BaseCoord, ("Coordinate")),  # BASE
        (otTables.MathValueRecord, ("Value")),  # MATH
        (otTables.ClipBox, ("xMin", "yMin", "xMax", "yMax")),  # COLR
    )
)
def visit(visitor, obj, attr, value):
    setattr(obj, attr, visitor.scale(value))


@ScalerVisitor.register_attr(
    (ttLib.getTableClass("hmtx"), ttLib.getTableClass("vmtx")), "metrics"
)
def visit(visitor, obj, attr, metrics):
    for g in metrics:
        advance, lsb = metrics[g]
        metrics[g] = visitor.scale(advance), visitor.scale(lsb)


@ScalerVisitor.register_attr(ttLib.getTableClass("VMTX"), "VOriginRecords")
def visit(visitor, obj, attr, VOriginRecords):
    for g in VOriginRecords:
        VOriginRecords[g] = visitor.scale(VOriginRecords[g])


@ScalerVisitor.register_attr(ttLib.getTableClass("glyf"), "glyphs")
def visit(visitor, obj, attr, glyphs):
    for g in glyphs.values():
        for attr in ("xMin", "xMax", "yMin", "yMax"):
            v = getattr(g, attr, None)
            if v is not None:
                setattr(g, attr, visitor.scale(v))

        if g.isComposite():
            for component in g.components:
                component.x = visitor.scale(component.x)
                component.y = visitor.scale(component.y)
            continue

        if hasattr(g, "coordinates"):
            coordinates = g.coordinates
            for i, (x, y) in enumerate(coordinates):
                coordinates[i] = visitor.scale(x), visitor.scale(y)


@ScalerVisitor.register_attr(ttLib.getTableClass("gvar"), "variations")
def visit(visitor, obj, attr, variations):
    glyfTable = visitor.font["glyf"]

    for glyphName, varlist in variations.items():
        glyph = glyfTable[glyphName]
        for var in varlist:
            coordinates = var.coordinates
            for i, xy in enumerate(coordinates):
                if xy is None:
                    continue
                coordinates[i] = visitor.scale(xy[0]), visitor.scale(xy[1])


@ScalerVisitor.register_attr(ttLib.getTableClass("VARC"), "table")
def visit(visitor, obj, attr, varc):
    # VarComposite variations are a pain

    fvar = visitor.font["fvar"]
    fvarAxes = [a.axisTag for a in fvar.axes]

    store = varc.MultiVarStore
    storeBuilder = OnlineMultiVarStoreBuilder(fvarAxes)

    for g in varc.VarCompositeGlyphs.VarCompositeGlyph:
        for component in g.components:
            t = component.transform
            t.translateX = visitor.scale(t.translateX)
            t.translateY = visitor.scale(t.translateY)
            t.tCenterX = visitor.scale(t.tCenterX)
            t.tCenterY = visitor.scale(t.tCenterY)

            if component.axisValuesVarIndex != otTables.NO_VARIATION_INDEX:
                varIdx = component.axisValuesVarIndex
                # TODO Move this code duplicated below to MultiVarStore.__getitem__,
                # or a getDeltasAndSupports().
                if varIdx != otTables.NO_VARIATION_INDEX:
                    major = varIdx >> 16
                    minor = varIdx & 0xFFFF
                    varData = store.MultiVarData[major]
                    vec = varData.Item[minor]
                    storeBuilder.setSupports(store.get_supports(major, fvar.axes))
                    if vec:
                        m = len(vec) // varData.VarRegionCount
                        vec = list(batched(vec, m))
                        vec = [Vector(v) for v in vec]
                        component.axisValuesVarIndex = storeBuilder.storeDeltas(vec)
                    else:
                        component.axisValuesVarIndex = otTables.NO_VARIATION_INDEX

            if component.transformVarIndex != otTables.NO_VARIATION_INDEX:
                varIdx = component.transformVarIndex
                if varIdx != otTables.NO_VARIATION_INDEX:
                    major = varIdx >> 16
                    minor = varIdx & 0xFFFF
                    vec = varData.Item[varIdx & 0xFFFF]
                    major = varIdx >> 16
                    minor = varIdx & 0xFFFF
                    varData = store.MultiVarData[major]
                    vec = varData.Item[minor]
                    storeBuilder.setSupports(store.get_supports(major, fvar.axes))
                    if vec:
                        m = len(vec) // varData.VarRegionCount
                        flags = component.flags
                        vec = list(batched(vec, m))
                        newVec = []
                        for v in vec:
                            v = list(v)
                            i = 0
                            ## Scale translate & tCenter
                            if flags & otTables.VarComponentFlags.HAVE_TRANSLATE_X:
                                v[i] = visitor.scale(v[i])
                                i += 1
                            if flags & otTables.VarComponentFlags.HAVE_TRANSLATE_Y:
                                v[i] = visitor.scale(v[i])
                                i += 1
                            if flags & otTables.VarComponentFlags.HAVE_ROTATION:
                                i += 1
                            if flags & otTables.VarComponentFlags.HAVE_SCALE_X:
                                i += 1
                            if flags & otTables.VarComponentFlags.HAVE_SCALE_Y:
                                i += 1
                            if flags & otTables.VarComponentFlags.HAVE_SKEW_X:
                                i += 1
                            if flags & otTables.VarComponentFlags.HAVE_SKEW_Y:
                                i += 1
                            if flags & otTables.VarComponentFlags.HAVE_TCENTER_X:
                                v[i] = visitor.scale(v[i])
                                i += 1
                            if flags & otTables.VarComponentFlags.HAVE_TCENTER_Y:
                                v[i] = visitor.scale(v[i])
                                i += 1

                            newVec.append(Vector(v))
                        vec = newVec

                        component.transformVarIndex = storeBuilder.storeDeltas(vec)
                    else:
                        component.transformVarIndex = otTables.NO_VARIATION_INDEX

    varc.MultiVarStore = storeBuilder.finish()


@ScalerVisitor.register_attr(ttLib.getTableClass("kern"), "kernTables")
def visit(visitor, obj, attr, kernTables):
    for table in kernTables:
        kernTable = table.kernTable
        for k in kernTable.keys():
            kernTable[k] = visitor.scale(kernTable[k])


def _cff_scale(visitor, args):
    for i, arg in enumerate(args):
        if not isinstance(arg, list):
            if not isinstance(arg, bytes):
                args[i] = visitor.scale(arg)
        else:
            num_blends = arg[-1]
            _cff_scale(visitor, arg)
            arg[-1] = num_blends


@ScalerVisitor.register_attr(
    (ttLib.getTableClass("CFF "), ttLib.getTableClass("CFF2")), "cff"
)
def visit(visitor, obj, attr, cff):
    cff.desubroutinize()
    topDict = cff.topDictIndex[0]
    varStore = getattr(topDict, "VarStore", None)
    getNumRegions = varStore.getNumRegions if varStore is not None else None
    privates = set()
    for fontname in cff.keys():
        font = cff[fontname]
        cs = font.CharStrings
        for g in font.charset:
            c, _ = cs.getItemAndSelector(g)
            privates.add(c.private)

            commands = cffSpecializer.programToCommands(
                c.program, getNumRegions=getNumRegions
            )
            for op, args in commands:
                if op == "vsindex":
                    continue
                _cff_scale(visitor, args)
            c.program[:] = cffSpecializer.commandsToProgram(commands)

        # Annoying business of scaling numbers that do not matter whatsoever

        for attr in (
            "UnderlinePosition",
            "UnderlineThickness",
            "FontBBox",
            "StrokeWidth",
        ):
            value = getattr(topDict, attr, None)
            if value is None:
                continue
            if isinstance(value, list):
                _cff_scale(visitor, value)
            else:
                setattr(topDict, attr, visitor.scale(value))

        for i in range(6):
            topDict.FontMatrix[i] /= visitor.scaleFactor

        for private in privates:
            for attr in (
                "BlueValues",
                "OtherBlues",
                "FamilyBlues",
                "FamilyOtherBlues",
                # "BlueScale",
                # "BlueShift",
                # "BlueFuzz",
                "StdHW",
                "StdVW",
                "StemSnapH",
                "StemSnapV",
                "defaultWidthX",
                "nominalWidthX",
            ):
                value = getattr(private, attr, None)
                if value is None:
                    continue
                if isinstance(value, list):
                    _cff_scale(visitor, value)
                else:
                    setattr(private, attr, visitor.scale(value))


# ItemVariationStore


@ScalerVisitor.register(otTables.VarData)
def visit(visitor, varData):
    for item in varData.Item:
        for i, v in enumerate(item):
            item[i] = visitor.scale(v)
    varData.calculateNumShorts()


# COLRv1


def _setup_scale_paint(paint, scale):
    if -2 <= scale <= 2 - (1 >> 14):
        paint.Format = otTables.PaintFormat.PaintScaleUniform
        paint.scale = scale
        return

    transform = otTables.Affine2x3()
    transform.populateDefaults()
    transform.xy = transform.yx = transform.dx = transform.dy = 0
    transform.xx = transform.yy = scale

    paint.Format = otTables.PaintFormat.PaintTransform
    paint.Transform = transform


@ScalerVisitor.register(otTables.BaseGlyphPaintRecord)
def visit(visitor, record):
    oldPaint = record.Paint

    scale = otTables.Paint()
    _setup_scale_paint(scale, visitor.scaleFactor)
    scale.Paint = oldPaint

    record.Paint = scale

    return True


@ScalerVisitor.register(otTables.Paint)
def visit(visitor, paint):
    if paint.Format != otTables.PaintFormat.PaintGlyph:
        return True

    newPaint = otTables.Paint()
    newPaint.Format = paint.Format
    newPaint.Paint = paint.Paint
    newPaint.Glyph = paint.Glyph
    del paint.Paint
    del paint.Glyph

    _setup_scale_paint(paint, 1 / visitor.scaleFactor)
    paint.Paint = newPaint

    visitor.visit(newPaint.Paint)

    return False


def scale_upem(font, new_upem):
    """Change the units-per-EM of font to the new value."""
    upem = font["head"].unitsPerEm
    visitor = ScalerVisitor(new_upem / upem)
    visitor.visit(font)


def main(args=None):
    """Change the units-per-EM of fonts"""

    if args is None:
        import sys

        args = sys.argv[1:]

    from fontTools.ttLib import TTFont
    from fontTools.misc.cliTools import makeOutputFileName
    import argparse

    parser = argparse.ArgumentParser(
        "fonttools ttLib.scaleUpem", description="Change the units-per-EM of fonts"
    )
    parser.add_argument("font", metavar="font", help="Font file.")
    parser.add_argument(
        "new_upem", metavar="new-upem", help="New units-per-EM integer value."
    )
    parser.add_argument(
        "--output-file", metavar="path", default=None, help="Output file."
    )

    options = parser.parse_args(args)

    font = TTFont(options.font)
    new_upem = int(options.new_upem)
    output_file = (
        options.output_file
        if options.output_file is not None
        else makeOutputFileName(options.font, overWrite=True, suffix="-scaled")
    )

    scale_upem(font, new_upem)

    print("Writing %s" % output_file)
    font.save(output_file)


if __name__ == "__main__":
    import sys

    sys.exit(main())
<|endoftext|>"""fontTools.ttLib -- a package for dealing with TrueType fonts."""

from fontTools.config import OPTIONS
from fontTools.misc.loggingTools import deprecateFunction
import logging

log = logging.getLogger(__name__)


OPTIMIZE_FONT_SPEED = OPTIONS["fontTools.ttLib:OPTIMIZE_FONT_SPEED"]


class TTLibError(Exception):
    pass


class TTLibFileIsCollectionError(TTLibError):
    pass


@deprecateFunction("use logging instead", category=DeprecationWarning)
def debugmsg(msg):
    import time

    print(msg + time.strftime("  (%H:%M:%S)", time.localtime(time.time())))


from fontTools.ttLib.ttFont import *
from fontTools.ttLib.ttCollection import TTCollection
<|endoftext|>"""Specialization of fontTools.misc.visitor to work with TTFont."""

from fontTools.misc.visitor import Visitor
from fontTools.ttLib import TTFont


class TTVisitor(Visitor):
    def visitAttr(self, obj, attr, value, *args, **kwargs):
        if isinstance(value, TTFont):
            return False
        super().visitAttr(obj, attr, value, *args, **kwargs)

    def visit(self, obj, *args, **kwargs):
        if hasattr(obj, "ensureDecompiled"):
            obj.ensureDecompiled(recurse=False)
        super().visit(obj, *args, **kwargs)


@TTVisitor.register(TTFont)
def visit(visitor, font, *args, **kwargs):
    # Some objects have links back to TTFont; even though we
    # have a check in visitAttr to stop them from recursing
    # onto TTFont, sometimes they still do, for example when
    # someone overrides visitAttr.
    if hasattr(visitor, "font"):
        return False

    visitor.font = font
    for tag in font.keys():
        visitor.visit(font[tag], *args, **kwargs)
    del visitor.font
    return False
<|endoftext|>from __future__ import annotations

import logging
import os
import traceback
from io import BytesIO, StringIO, UnsupportedOperation
from typing import TYPE_CHECKING, TypedDict, TypeVar, overload

from fontTools.config import Config
from fontTools.misc import xmlWriter
from fontTools.misc.configTools import AbstractConfig
from fontTools.misc.loggingTools import deprecateArgument
from fontTools.misc.textTools import Tag, byteord, tostr
from fontTools.ttLib import TTLibError
from fontTools.ttLib.sfnt import SFNTReader, SFNTWriter
from fontTools.ttLib.ttGlyphSet import (
    _TTGlyph,  # noqa: F401
    _TTGlyphSet,
    _TTGlyphSetCFF,
    _TTGlyphSetGlyf,
    _TTGlyphSetVARC,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, MutableMapping
    from types import ModuleType, TracebackType
    from typing import Any, BinaryIO, Literal, Sequence, TextIO

    from typing_extensions import Self, Unpack

    from fontTools.ttLib.tables import (
        B_A_S_E_,
        C_B_D_T_,
        C_B_L_C_,
        C_F_F_,
        C_F_F__2,
        C_O_L_R_,
        C_P_A_L_,
        D_S_I_G_,
        E_B_D_T_,
        E_B_L_C_,
        E_B_S_C_,
        F_F_T_M_,
        G_D_E_F_,
        G_P_O_S_,
        G_S_U_B_,
        G_V_A_R_,
        H_V_A_R_,
        I_F_T_,
        I_F_T_X_,
        J_S_T_F_,
        L_T_S_H_,
        M_A_T_H_,
        M_V_A_R_,
        S_T_A_T_,
        S_V_G_,
        T_S_I__0,
        T_S_I__1,
        T_S_I__2,
        T_S_I__3,
        T_S_I__5,
        T_S_I_B_,
        T_S_I_C_,
        T_S_I_D_,
        T_S_I_J_,
        T_S_I_P_,
        T_S_I_S_,
        T_S_I_V_,
        T_T_F_A_,
        V_A_R_C_,
        V_D_M_X_,
        V_O_R_G_,
        V_V_A_R_,
        D__e_b_g,
        F__e_a_t,
        G__l_a_t,
        G__l_o_c,
        O_S_2f_2,
        S__i_l_f,
        S__i_l_l,
        _a_n_k_r,
        _a_v_a_r,
        _b_d_a_t,
        _b_g_c_l,
        _b_h_e_d,
        _b_l_o_c,
        _b_s_l_n,
        _c_i_d_g,
        _c_m_a_p,
        _c_v_a_r,
        _c_v_t,
        _f_e_a_t,
        _f_p_g_m,
        _f_v_a_r,
        _g_a_s_p,
        _g_c_i_d,
        _g_l_y_f,
        _g_v_a_r,
        _h_d_m_x,
        _h_e_a_d,
        _h_h_e_a,
        _h_m_t_x,
        _k_e_r_n,
        _l_c_a_r,
        _l_o_c_a,
        _l_t_a_g,
        _m_a_x_p,
        _m_e_t_a,
        _m_o_r_t,
        _m_o_r_x,
        _n_a_m_e,
        _o_p_b_d,
        _p_o_s_t,
        _p_r_e_p,
        _p_r_o_p,
        _s_b_i_x,
        _t_r_a_k,
        _v_h_e_a,
        _v_m_t_x,
    )
    from fontTools.ttLib.tables.DefaultTable import DefaultTable

    _VT_co = TypeVar("_VT_co", covariant=True)  # Value type covariant containers.

log = logging.getLogger(__name__)


_NumberT = TypeVar("_NumberT", bound=float)


class TTFont(object):
    """Represents a TrueType font.

    The object manages file input and output, and offers a convenient way of
    accessing tables. Tables will be only decompiled when necessary, ie. when
    they're actually accessed. This means that simple operations can be extremely fast.

    Example usage:

    .. code-block:: pycon

        >>>
        >> from fontTools import ttLib
        >> tt = ttLib.TTFont("afont.ttf") # Load an existing font file
        >> tt['maxp'].numGlyphs
        242
        >> tt['OS/2'].achVendID
        'B&H\000'
        >> tt['head'].unitsPerEm
        2048

    For details of the objects returned when accessing each table, see the
    :doc:`tables </ttLib/tables>` documentation.
    To add a table to the font, use the :py:func:`newTable` function:

    .. code-block:: pycon

        >>>
        >> os2 = newTable("OS/2")
        >> os2.version = 4
        >> # set other attributes
        >> font["OS/2"] = os2

    TrueType fonts can also be serialized to and from XML format (see also the
    :doc:`ttx </ttx>` binary):

    .. code-block:: pycon

        >>
        >> tt.saveXML("afont.ttx")
        Dumping 'LTSH' table...
        Dumping 'OS/2' table...
        [...]

        >> tt2 = ttLib.TTFont() # Create a new font object
        >> tt2.importXML("afont.ttx")
        >> tt2['maxp'].numGlyphs
        242

    The TTFont object may be used as a context manager; this will cause the file
    reader to be closed after the context ``with`` block is exited::

            with TTFont(filename) as f:
                    # Do stuff

    Args:
            file: When reading a font from disk, either a pathname pointing to a file,
                    or a readable file object.
            res_name_or_index: If running on a Macintosh, either a sfnt resource name or
                    an sfnt resource index number. If the index number is zero, TTLib will
                    autodetect whether the file is a flat file or a suitcase. (If it is a suitcase,
                    only the first 'sfnt' resource will be read.)
            sfntVersion (str): When constructing a font object from scratch, sets the four-byte
                    sfnt magic number to be used. Defaults to ``\0\1\0\0`` (TrueType). To create
                    an OpenType file, use ``OTTO``.
            flavor (str): Set this to ``woff`` when creating a WOFF file or ``woff2`` for a WOFF2
                    file.
            checkChecksums (int): How checksum data should be treated. Default is 0
                    (no checking). Set to 1 to check and warn on wrong checksums; set to 2 to
                    raise an exception if any wrong checksums are found.
            recalcBBoxes (bool): If true (the default), recalculates ``glyf``, ``CFF ``,
                    ``head`` bounding box values and ``hhea``/``vhea`` min/max values on save.
                    Also compiles the glyphs on importing, which saves memory consumption and
                    time.
            ignoreDecompileErrors (bool): If true, exceptions raised during table decompilation
                    will be ignored, and the binary data will be returned for those tables instead.
            recalcTimestamp (bool): If true (the default), sets the ``modified`` timestamp in
                    the ``head`` table on save.
            fontNumber (int): The index of the font in a TrueType Collection file.
            lazy (bool): If lazy is set to True, many data structures are loaded lazily, upon
                    access only. If it is set to False, many data structures are loaded immediately.
                    The default is ``lazy=None`` which is somewhere in between.
    """

    tables: dict[Tag, DefaultTable | GlyphOrder]
    reader: SFNTReader | None
    sfntVersion: str
    flavor: str | None
    flavorData: Any | None
    lazy: bool | None
    recalcBBoxes: bool
    recalcTimestamp: bool
    ignoreDecompileErrors: bool
    cfg: AbstractConfig
    glyphOrder: list[str]
    _reverseGlyphOrderDict: dict[str, int]
    _tableCache: MutableMapping[tuple[Tag, bytes], DefaultTable] | None
    disassembleInstructions: bool
    bitmapGlyphDataFormat: str
    # Deprecated attributes
    verbose: bool | None
    quiet: bool | None

    def __init__(
        self,
        file: str | os.PathLike[str] | BinaryIO | None = None,
        res_name_or_index: str | int | None = None,
        sfntVersion: str = "\000\001\000\000",
        flavor: str | None = None,
        checkChecksums: int = 0,
        verbose: bool | None = None,  # Deprecated
        recalcBBoxes: bool = True,
        allowVID: Any = NotImplemented,  # Deprecated/Unused
        ignoreDecompileErrors: bool = False,
        recalcTimestamp: bool = True,
        fontNumber: int = -1,
        lazy: bool | None = None,
        quiet: bool | None = None,  # Deprecated
        _tableCache: MutableMapping[tuple[Tag, bytes], DefaultTable] | None = None,
        cfg: Mapping[str, Any] | AbstractConfig = {},
    ) -> None:
        # Set deprecated attributes
        for name in ("verbose", "quiet"):
            val = locals().get(name)
            if val is not None:
                deprecateArgument(name, "configure logging instead")
            setattr(self, name, val)

        self.lazy = lazy
        self.recalcBBoxes = recalcBBoxes
        self.recalcTimestamp = recalcTimestamp
        self.tables = {}
        self.reader = None
        self.cfg = cfg.copy() if isinstance(cfg, AbstractConfig) else Config(cfg)
        self.ignoreDecompileErrors = ignoreDecompileErrors

        if not file:
            self.sfntVersion = sfntVersion
            self.flavor = flavor
            self.flavorData = None
            return
        seekable = True
        if not hasattr(file, "read"):
            if not isinstance(file, (str, os.PathLike)):
                raise TypeError(
                    "fileOrPath must be a file path (str or PathLike) if it isn't an object with a `read` method."
                )
            closeStream = True
            # assume file is a string
            if res_name_or_index is not None:
                # see if it contains 'sfnt' resources in the resource or data fork
                from . import macUtils

                if res_name_or_index == 0:
                    if macUtils.getSFNTResIndices(file):
                        # get the first available sfnt font.
                        file = macUtils.SFNTResourceReader(file, 1)
                    else:
                        file = open(file, "rb")
                else:
                    file = macUtils.SFNTResourceReader(file, res_name_or_index)
            else:
                file = open(file, "rb")
        else:
            # assume "file" is a readable file object
            assert not isinstance(file, (str, os.PathLike))
            closeStream = False
            # SFNTReader wants the input file to be seekable.
            # SpooledTemporaryFile has no seekable() on < 3.11, but still can seek:
            # https://github.com/fonttools/fonttools/issues/3052
            if hasattr(file, "seekable"):
                seekable = file.seekable()
            elif hasattr(file, "seek"):
                try:
                    file.seek(0)
                except UnsupportedOperation:
                    seekable = False

        if not self.lazy:
            # read input file in memory and wrap a stream around it to allow overwriting
            if seekable:
                file.seek(0)
            tmp = BytesIO(file.read())
            if hasattr(file, "name"):
                # save reference to input file name
                tmp.name = file.name
            if closeStream:
                file.close()
            file = tmp
        elif not seekable:
            raise TTLibError("Input file must be seekable when lazy=True")
        self._tableCache = _tableCache
        self.reader = SFNTReader(file, checkChecksums, fontNumber=fontNumber)
        self.sfntVersion = self.reader.sfntVersion
        self.flavor = self.reader.flavor
        self.flavorData = self.reader.flavorData

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """If we still have a reader object, close it."""
        if self.reader is not None:
            self.reader.close()
            self.reader = None

    def save(
        self, file: str | os.PathLike[str] | BinaryIO, reorderTables: bool | None = True
    ) -> None:
        """Save the font to disk.

        Args:
                file: Similarly to the constructor, can be either a pathname or a writable
                        binary file object.
                reorderTables (Option[bool]): If true (the default), reorder the tables,
                        sorting them by tag (recommended by the OpenType specification). If
                        false, retain the original font order. If None, reorder by table
                        dependency (fastest).
        """
        if not hasattr(file, "write"):
            if self.lazy and self.reader.file.name == file:
                raise TTLibError("Can't overwrite TTFont when 'lazy' attribute is True")
            createStream = True
        else:
            # assume "file" is a writable file object
            createStream = False

        tmp = BytesIO()

        writer_reordersTables = self._save(tmp)

        if not (
            reorderTables is None
            or writer_reordersTables
            or (reorderTables is False and self.reader is None)
        ):
            if reorderTables is False:
                # sort tables using the original font's order
                if self.reader is None:
                    raise TTLibError(
                        "The original table order is unavailable because there isn't a font to read it from."
                    )
                tableOrder = list(self.reader.keys())
            else:
                # use the recommended order from the OpenType specification
                tableOrder = None
            tmp.flush()
            tmp2 = BytesIO()
            reorderFontTables(tmp, tmp2, tableOrder)
            tmp.close()
            tmp = tmp2

        if createStream:
            # "file" is a path
            assert isinstance(file, (str, os.PathLike))
            with open(file, "wb") as file:
                file.write(tmp.getvalue())
        else:
            assert not isinstance(file, (str, os.PathLike))
            file.write(tmp.getvalue())

        tmp.close()

    def _save(
        self,
        file: BinaryIO,
        tableCache: MutableMapping[tuple[Tag, bytes], Any] | None = None,
    ) -> bool:
        """Internal function, to be shared by save() and TTCollection.save()"""

        if self.recalcTimestamp and "head" in self:
            # make sure 'head' is loaded so the recalculation is actually done
            self["head"]

        tags = self.keys()
        tags.pop(0)  # skip GlyphOrder tag
        numTables = len(tags)
        # write to a temporary stream to allow saving to unseekable streams
        writer = SFNTWriter(
            file, numTables, self.sfntVersion, self.flavor, self.flavorData
        )

        done = []
        for tag in tags:
            self._writeTable(tag, writer, done, tableCache)

        writer.close()

        return writer.reordersTables()

    class XMLSavingOptions(TypedDict):
        writeVersion: bool
        quiet: bool | None
        tables: Sequence[str | bytes] | None
        skipTables: Sequence[str] | None
        splitTables: bool
        splitGlyphs: bool
        disassembleInstructions: bool
        bitmapGlyphDataFormat: str

    def saveXML(
        self,
        fileOrPath: str | os.PathLike[str] | BinaryIO | TextIO,
        newlinestr: str = "\n",
        **kwargs: Unpack[XMLSavingOptions],
    ) -> None:
        """Export the font as TTX (an XML-based text file), or as a series of text
        files when splitTables is true. In the latter case, the 'fileOrPath'
        argument should be a path to a directory.
        The 'tables' argument must either be falsy (None or empty list, meaning
        dump all tables) or a non-empty list of tables to dump. The 'skipTables'
        argument may be a list of tables to skip, but only when 'tables' is falsy.
        """

        writer = xmlWriter.XMLWriter(fileOrPath, newlinestr=newlinestr)
        self._saveXML(writer, **kwargs)
        writer.close()

    def _saveXML(
        self,
        writer: xmlWriter.XMLWriter,
        writeVersion: bool = True,
        quiet: bool | None = None,  # Deprecated
        tables: Sequence[str | bytes] | None = None,
        skipTables: Sequence[str] | None = None,
        splitTables: bool = False,
        splitGlyphs: bool = False,
        disassembleInstructions: bool = True,
        bitmapGlyphDataFormat: str = "raw",
    ) -> None:
        if quiet is not None:
            deprecateArgument("quiet", "configure logging instead")

        self.disassembleInstructions = disassembleInstructions
        self.bitmapGlyphDataFormat = bitmapGlyphDataFormat
        if not tables:
            tables = self.keys()
            if skipTables:
                tables = [tag for tag in tables if tag not in skipTables]

        if writeVersion:
            from fontTools import version

            version = ".".join(version.split(".")[:2])
            writer.begintag(
                "ttFont",
                sfntVersion=repr(tostr(self.sfntVersion))[1:-1],
                ttLibVersion=version,
            )
        else:
            writer.begintag("ttFont", sfntVersion=repr(tostr(self.sfntVersion))[1:-1])
        writer.newline()

        # always splitTables if splitGlyphs is enabled
        splitTables = splitTables or splitGlyphs

        if not splitTables:
            writer.newline()
        else:
            if writer.filename is None:
                raise TTLibError(
                    "splitTables requires the file name to be a file system path, not a stream."
                )
            path, ext = os.path.splitext(writer.filename)

        for tag in tables:
            if splitTables:
                tablePath = path + "." + tagToIdentifier(tag) + ext
                tableWriter = xmlWriter.XMLWriter(
                    tablePath, newlinestr=writer.newlinestr
                )
                tableWriter.begintag("ttFont", ttLibVersion=version)
                tableWriter.newline()
                tableWriter.newline()
                writer.simpletag(tagToXML(tag), src=os.path.basename(tablePath))
                writer.newline()
            else:
                tableWriter = writer
            self._tableToXML(tableWriter, tag, splitGlyphs=splitGlyphs)
            if splitTables:
                tableWriter.endtag("ttFont")
                tableWriter.newline()
                tableWriter.close()
        writer.endtag("ttFont")
        writer.newline()

    def _tableToXML(
        self,
        writer: xmlWriter.XMLWriter,
        tag: str | bytes,
        quiet: bool | None = None,
        splitGlyphs: bool = False,
    ) -> None:
        if quiet is not None:
            deprecateArgument("quiet", "configure logging instead")
        if tag in self:
            table = self[tag]
            report = "Dumping '%s' table..." % tag
        else:
            report = "No '%s' table found." % tag
        log.info(report)
        if tag not in self:
            return
        xmlTag = tagToXML(tag)
        attrs: dict[str, Any] = {}
        if hasattr(table, "ERROR"):
            attrs["ERROR"] = "decompilation error"
        from .tables.DefaultTable import DefaultTable

        if table.__class__ == DefaultTable:
            attrs["raw"] = True
        writer.begintag(xmlTag, **attrs)
        writer.newline()
        if tag == "glyf":
            table.toXML(writer, self, splitGlyphs=splitGlyphs)
        else:
            table.toXML(writer, self)
        writer.endtag(xmlTag)
        writer.newline()
        writer.newline()

    def importXML(
        self, fileOrPath: str | os.PathLike[str] | BinaryIO, quiet: bool | None = None
    ) -> None:
        """Import a TTX file (an XML-based text format), so as to recreate
        a font object.
        """
        if quiet is not None:
            deprecateArgument("quiet", "configure logging instead")

        if "maxp" in self and "post" in self:
            # Make sure the glyph order is loaded, as it otherwise gets
            # lost if the XML doesn't contain the glyph order, yet does
            # contain the table which was originally used to extract the
            # glyph names from (ie. 'post', 'cmap' or 'CFF ').
            self.getGlyphOrder()

        from fontTools.misc import xmlReader

        reader = xmlReader.XMLReader(fileOrPath, self)
        reader.read()

    def isLoaded(self, tag: str | bytes) -> bool:
        """Return true if the table identified by ``tag`` has been
        decompiled and loaded into memory."""
        return tag in self.tables

    def has_key(self, tag: str | bytes) -> bool:
        """Test if the table identified by ``tag`` is present in the font.

        As well as this method, ``tag in font`` can also be used to determine the
        presence of the table."""
        if self.isLoaded(tag):
            return True
        elif self.reader and tag in self.reader:
            return True
        elif tag == "GlyphOrder":
            return True
        else:
            return False

    __contains__ = has_key

    def keys(self) -> list[str]:
        """Returns the list of tables in the font, along with the ``GlyphOrder`` pseudo-table."""
        keys = list(self.tables.keys())
        if self.reader:
            for key in list(self.reader.keys()):
                if key not in keys:
                    keys.append(key)

        if "GlyphOrder" in keys:
            keys.remove("GlyphOrder")
        keys = sortedTagList(keys)
        return ["GlyphOrder"] + keys

    def ensureDecompiled(self, recurse: bool | None = None) -> None:
        """Decompile all the tables, even if a TTFont was opened in 'lazy' mode."""
        for tag in self.keys():
            table = self[tag]
            if recurse is None:
                recurse = self.lazy is not False
            if recurse and hasattr(table, "ensureDecompiled"):
                table.ensureDecompiled(recurse=recurse)
        self.lazy = False

    def __len__(self) -> int:
        return len(list(self.keys()))

    @overload
    def __getitem__(self, tag: Literal["BASE"]) -> B_A_S_E_.table_B_A_S_E_: ...
    @overload
    def __getitem__(self, tag: Literal["CBDT"]) -> C_B_D_T_.table_C_B_D_T_: ...
    @overload
    def __getitem__(self, tag: Literal["CBLC"]) -> C_B_L_C_.table_C_B_L_C_: ...
    @overload
    def __getitem__(self, tag: Literal["CFF "]) -> C_F_F_.table_C_F_F_: ...
    @overload
    def __getitem__(self, tag: Literal["CFF2"]) -> C_F_F__2.table_C_F_F__2: ...
    @overload
    def __getitem__(self, tag: Literal["COLR"]) -> C_O_L_R_.table_C_O_L_R_: ...
    @overload
    def __getitem__(self, tag: Literal["CPAL"]) -> C_P_A_L_.table_C_P_A_L_: ...
    @overload
    def __getitem__(self, tag: Literal["DSIG"]) -> D_S_I_G_.table_D_S_I_G_: ...
    @overload
    def __getitem__(self, tag: Literal["EBDT"]) -> E_B_D_T_.table_E_B_D_T_: ...
    @overload
    def __getitem__(self, tag: Literal["EBLC"]) -> E_B_L_C_.table_E_B_L_C_: ...
    @overload
    def __getitem__(self, tag: Literal["EBSC"]) -> E_B_S_C_.table_E_B_S_C_: ...
    @overload
    def __getitem__(self, tag: Literal["FFTM"]) -> F_F_T_M_.table_F_F_T_M_: ...
    @overload
    def __getitem__(self, tag: Literal["GDEF"]) -> G_D_E_F_.table_G_D_E_F_: ...
    @overload
    def __getitem__(self, tag: Literal["GPOS"]) -> G_P_O_S_.table_G_P_O_S_: ...
    @overload
    def __getitem__(self, tag: Literal["GSUB"]) -> G_S_U_B_.table_G_S_U_B_: ...
    @overload
    def __getitem__(self, tag: Literal["GVAR"]) -> G_V_A_R_.table_G_V_A_R_: ...
    @overload
    def __getitem__(self, tag: Literal["HVAR"]) -> H_V_A_R_.table_H_V_A_R_: ...
    @overload
    def __getitem__(self, tag: Literal["IFT "]) -> I_F_T_.table_I_F_T_: ...
    @overload
    def __getitem__(self, tag: Literal["IFTX"]) -> I_F_T_X_.table_I_F_T_X_: ...
    @overload
    def __getitem__(self, tag: Literal["JSTF"]) -> J_S_T_F_.table_J_S_T_F_: ...
    @overload
    def __getitem__(self, tag: Literal["LTSH"]) -> L_T_S_H_.table_L_T_S_H_: ...
    @overload
    def __getitem__(self, tag: Literal["MATH"]) -> M_A_T_H_.table_M_A_T_H_: ...
    @overload
    def __getitem__(self, tag: Literal["MVAR"]) -> M_V_A_R_.table_M_V_A_R_: ...
    @overload
    def __getitem__(self, tag: Literal["STAT"]) -> S_T_A_T_.table_S_T_A_T_: ...
    @overload
    def __getitem__(self, tag: Literal["SVG "]) -> S_V_G_.table_S_V_G_: ...
    @overload
    def __getitem__(self, tag: Literal["TSI0"]) -> T_S_I__0.table_T_S_I__0: ...
    @overload
    def __getitem__(self, tag: Literal["TSI1"]) -> T_S_I__1.table_T_S_I__1: ...
    @overload
    def __getitem__(self, tag: Literal["TSI2"]) -> T_S_I__2.table_T_S_I__2: ...
    @overload
    def __getitem__(self, tag: Literal["TSI3"]) -> T_S_I__3.table_T_S_I__3: ...
    @overload
    def __getitem__(self, tag: Literal["TSI5"]) -> T_S_I__5.table_T_S_I__5: ...
    @overload
    def __getitem__(self, tag: Literal["TSIB"]) -> T_S_I_B_.table_T_S_I_B_: ...
    @overload
    def __getitem__(self, tag: Literal["TSIC"]) -> T_S_I_C_.table_T_S_I_C_: ...
    @overload
    def __getitem__(self, tag: Literal["TSID"]) -> T_S_I_D_.table_T_S_I_D_: ...
    @overload
    def __getitem__(self, tag: Literal["TSIJ"]) -> T_S_I_J_.table_T_S_I_J_: ...
    @overload
    def __getitem__(self, tag: Literal["TSIP"]) -> T_S_I_P_.table_T_S_I_P_: ...
    @overload
    def __getitem__(self, tag: Literal["TSIS"]) -> T_S_I_S_.table_T_S_I_S_: ...
    @overload
    def __getitem__(self, tag: Literal["TSIV"]) -> T_S_I_V_.table_T_S_I_V_: ...
    @overload
    def __getitem__(self, tag: Literal["TTFA"]) -> T_T_F_A_.table_T_T_F_A_: ...
    @overload
    def __getitem__(self, tag: Literal["VARC"]) -> V_A_R_C_.table_V_A_R_C_: ...
    @overload
    def __getitem__(self, tag: Literal["VDMX"]) -> V_D_M_X_.table_V_D_M_X_: ...
    @overload
    def __getitem__(self, tag: Literal["VORG"]) -> V_O_R_G_.table_V_O_R_G_: ...
    @overload
    def __getitem__(self, tag: Literal["VVAR"]) -> V_V_A_R_.table_V_V_A_R_: ...
    @overload
    def __getitem__(self, tag: Literal["Debg"]) -> D__e_b_g.table_D__e_b_g: ...
    @overload
    def __getitem__(self, tag: Literal["Feat"]) -> F__e_a_t.table_F__e_a_t: ...
    @overload
    def __getitem__(self, tag: Literal["Glat"]) -> G__l_a_t.table_G__l_a_t: ...
    @overload
    def __getitem__(self, tag: Literal["Gloc"]) -> G__l_o_c.table_G__l_o_c: ...
    @overload
    def __getitem__(self, tag: Literal["OS/2"]) -> O_S_2f_2.table_O_S_2f_2: ...
    @overload
    def __getitem__(self, tag: Literal["Silf"]) -> S__i_l_f.table_S__i_l_f: ...
    @overload
    def __getitem__(self, tag: Literal["Sill"]) -> S__i_l_l.table_S__i_l_l: ...
    @overload
    def __getitem__(self, tag: Literal["ankr"]) -> _a_n_k_r.table__a_n_k_r: ...
    @overload
    def __getitem__(self, tag: Literal["avar"]) -> _a_v_a_r.table__a_v_a_r: ...
    @overload
    def __getitem__(self, tag: Literal["bsln"]) -> _b_s_l_n.table__b_s_l_n: ...
    @overload
    def __getitem__(self, tag: Literal["bdat"]) -> _b_d_a_t.table__b_d_a_t: ...
    @overload
    def __getitem__(self, tag: Literal["bhed"]) -> _b_h_e_d.table__b_h_e_d: ...
    @overload
    def __getitem__(self, tag: Literal["bgcl"]) -> _b_g_c_l.table__b_g_c_l: ...
    @overload
    def __getitem__(self, tag: Literal["bloc"]) -> _b_l_o_c.table__b_l_o_c: ...
    @overload
    def __getitem__(self, tag: Literal["cidg"]) -> _c_i_d_g.table__c_i_d_g: ...
    @overload
    def __getitem__(self, tag: Literal["cmap"]) -> _c_m_a_p.table__c_m_a_p: ...
    @overload
    def __getitem__(self, tag: Literal["cvar"]) -> _c_v_a_r.table__c_v_a_r: ...
    @overload
    def __getitem__(self, tag: Literal["cvt "]) -> _c_v_t.table__c_v_t: ...
    @overload
    def __getitem__(self, tag: Literal["feat"]) -> _f_e_a_t.table__f_e_a_t: ...
    @overload
    def __getitem__(self, tag: Literal["fpgm"]) -> _f_p_g_m.table__f_p_g_m: ...
    @overload
    def __getitem__(self, tag: Literal["fvar"]) -> _f_v_a_r.table__f_v_a_r: ...
    @overload
    def __getitem__(self, tag: Literal["gasp"]) -> _g_a_s_p.table__g_a_s_p: ...
    @overload
    def __getitem__(self, tag: Literal["gcid"]) -> _g_c_i_d.table__g_c_i_d: ...
    @overload
    def __getitem__(self, tag: Literal["glyf"]) -> _g_l_y_f.table__g_l_y_f: ...
    @overload
    def __getitem__(self, tag: Literal["gvar"]) -> _g_v_a_r.table__g_v_a_r: ...
    @overload
    def __getitem__(self, tag: Literal["hdmx"]) -> _h_d_m_x.table__h_d_m_x: ...
    @overload
    def __getitem__(self, tag: Literal["head"]) -> _h_e_a_d.table__h_e_a_d: ...
    @overload
    def __getitem__(self, tag: Literal["hhea"]) -> _h_h_e_a.table__h_h_e_a: ...
    @overload
    def __getitem__(self, tag: Literal["hmtx"]) -> _h_m_t_x.table__h_m_t_x: ...
    @overload
    def __getitem__(self, tag: Literal["kern"]) -> _k_e_r_n.table__k_e_r_n: ...
    @overload
    def __getitem__(self, tag: Literal["lcar"]) -> _l_c_a_r.table__l_c_a_r: ...
    @overload
    def __getitem__(self, tag: Literal["loca"]) -> _l_o_c_a.table__l_o_c_a: ...
    @overload
    def __getitem__(self, tag: Literal["ltag"]) -> _l_t_a_g.table__l_t_a_g: ...
    @overload
    def __getitem__(self, tag: Literal["maxp"]) -> _m_a_x_p.table__m_a_x_p: ...
    @overload
    def __getitem__(self, tag: Literal["meta"]) -> _m_e_t_a.table__m_e_t_a: ...
    @overload
    def __getitem__(self, tag: Literal["mort"]) -> _m_o_r_t.table__m_o_r_t: ...
    @overload
    def __getitem__(self, tag: Literal["morx"]) -> _m_o_r_x.table__m_o_r_x: ...
    @overload
    def __getitem__(self, tag: Literal["name"]) -> _n_a_m_e.table__n_a_m_e: ...
    @overload
    def __getitem__(self, tag: Literal["opbd"]) -> _o_p_b_d.table__o_p_b_d: ...
    @overload
    def __getitem__(self, tag: Literal["post"]) -> _p_o_s_t.table__p_o_s_t: ...
    @overload
    def __getitem__(self, tag: Literal["prep"]) -> _p_r_e_p.table__p_r_e_p: ...
    @overload
    def __getitem__(self, tag: Literal["prop"]) -> _p_r_o_p.table__p_r_o_p: ...
    @overload
    def __getitem__(self, tag: Literal["sbix"]) -> _s_b_i_x.table__s_b_i_x: ...
    @overload
    def __getitem__(self, tag: Literal["trak"]) -> _t_r_a_k.table__t_r_a_k: ...
    @overload
    def __getitem__(self, tag: Literal["vhea"]) -> _v_h_e_a.table__v_h_e_a: ...
    @overload
    def __getitem__(self, tag: Literal["vmtx"]) -> _v_m_t_x.table__v_m_t_x: ...
    @overload
    def __getitem__(self, tag: Literal["GlyphOrder"]) -> GlyphOrder: ...
    @overload
    def __getitem__(self, tag: str | bytes) -> DefaultTable | GlyphOrder: ...

    def __getitem__(self, tag: str | bytes) -> DefaultTable | GlyphOrder:
        tag = Tag(tag)
        table = self.tables.get(tag)
        if table is None:
            if tag == "GlyphOrder":
                table = GlyphOrder(tag)
                self.tables[tag] = table
            elif self.reader is not None:
                table = self._readTable(tag)
            else:
                raise KeyError("'%s' table not found" % tag)
        return table

    def _readTable(self, tag: Tag) -> DefaultTable:
        log.debug("Reading '%s' table from disk", tag)
        assert self.reader is not None
        data = self.reader[tag]
        if self._tableCache is not None:
            table = self._tableCache.get((tag, data))
            if table is not None:
                return table
        tableClass = getTableClass(tag)
        table = tableClass(tag)
        self.tables[tag] = table
        log.debug("Decompiling '%s' table", tag)
        try:
            table.decompile(data, self)
        except Exception:
            if not self.ignoreDecompileErrors:
                raise
            # fall back to DefaultTable, retaining the binary table data
            log.exception(
                "An exception occurred during the decompilation of the '%s' table", tag
            )
            from .tables.DefaultTable import DefaultTable

            file = StringIO()
            traceback.print_exc(file=file)
            table = DefaultTable(tag)
            table.ERROR = file.getvalue()
            self.tables[tag] = table
            table.decompile(data, self)
        if self._tableCache is not None:
            self._tableCache[(tag, data)] = table
        return table

    def __setitem__(self, tag: str | bytes, table: DefaultTable) -> None:
        self.tables[Tag(tag)] = table

    def __delitem__(self, tag: str | bytes) -> None:
        if tag not in self:
            raise KeyError("'%s' table not found" % tag)
        if tag in self.tables:
            del self.tables[tag]
        if self.reader and tag in self.reader:
            del self.reader[tag]

    @overload
    def get(self, tag: Literal["BASE"]) -> B_A_S_E_.table_B_A_S_E_ | None: ...
    @overload
    def get(self, tag: Literal["CBDT"]) -> C_B_D_T_.table_C_B_D_T_ | None: ...
    @overload
    def get(self, tag: Literal["CBLC"]) -> C_B_L_C_.table_C_B_L_C_ | None: ...
    @overload
    def get(self, tag: Literal["CFF "]) -> C_F_F_.table_C_F_F_ | None: ...
    @overload
    def get(self, tag: Literal["CFF2"]) -> C_F_F__2.table_C_F_F__2 | None: ...
    @overload
    def get(self, tag: Literal["COLR"]) -> C_O_L_R_.table_C_O_L_R_ | None: ...
    @overload
    def get(self, tag: Literal["CPAL"]) -> C_P_A_L_.table_C_P_A_L_ | None: ...
    @overload
    def get(self, tag: Literal["DSIG"]) -> D_S_I_G_.table_D_S_I_G_ | None: ...
    @overload
    def get(self, tag: Literal["EBDT"]) -> E_B_D_T_.table_E_B_D_T_ | None: ...
    @overload
    def get(self, tag: Literal["EBLC"]) -> E_B_L_C_.table_E_B_L_C_ | None: ...
    @overload
    def get(self, tag: Literal["EBSC"]) -> E_B_S_C_.table_E_B_S_C_ | None: ...
    @overload
    def get(self, tag: Literal["FFTM"]) -> F_F_T_M_.table_F_F_T_M_ | None: ...
    @overload
    def get(self, tag: Literal["GDEF"]) -> G_D_E_F_.table_G_D_E_F_ | None: ...
    @overload
    def get(self, tag: Literal["GPOS"]) -> G_P_O_S_.table_G_P_O_S_ | None: ...
    @overload
    def get(self, tag: Literal["GSUB"]) -> G_S_U_B_.table_G_S_U_B_ | None: ...
    @overload
    def get(self, tag: Literal["GVAR"]) -> G_V_A_R_.table_G_V_A_R_ | None: ...
    @overload
    def get(self, tag: Literal["HVAR"]) -> H_V_A_R_.table_H_V_A_R_ | None: ...
    @overload
    def get(self, tag: Literal["IFT "]) -> I_F_T_.table_I_F_T_ | None: ...
    @overload
    def get(self, tag: Literal["IFTX"]) -> I_F_T_X_.table_I_F_T_X_ | None: ...
    @overload
    def get(self, tag: Literal["JSTF"]) -> J_S_T_F_.table_J_S_T_F_ | None: ...
    @overload
    def get(self, tag: Literal["LTSH"]) -> L_T_S_H_.table_L_T_S_H_ | None: ...
    @overload
    def get(self, tag: Literal["MATH"]) -> M_A_T_H_.table_M_A_T_H_ | None: ...
    @overload
    def get(self, tag: Literal["MVAR"]) -> M_V_A_R_.table_M_V_A_R_ | None: ...
    @overload
    def get(self, tag: Literal["STAT"]) -> S_T_A_T_.table_S_T_A_T_ | None: ...
    @overload
    def get(self, tag: Literal["SVG "]) -> S_V_G_.table_S_V_G_ | None: ...
    @overload
    def get(self, tag: Literal["TSI0"]) -> T_S_I__0.table_T_S_I__0 | None: ...
    @overload
    def get(self, tag: Literal["TSI1"]) -> T_S_I__1.table_T_S_I__1 | None: ...
    @overload
    def get(self, tag: Literal["TSI2"]) -> T_S_I__2.table_T_S_I__2 | None: ...
    @overload
    def get(self, tag: Literal["TSI3"]) -> T_S_I__3.table_T_S_I__3 | None: ...
    @overload
    def get(self, tag: Literal["TSI5"]) -> T_S_I__5.table_T_S_I__5 | None: ...
    @overload
    def get(self, tag: Literal["TSIB"]) -> T_S_I_B_.table_T_S_I_B_ | None: ...
    @overload
    def get(self, tag: Literal["TSIC"]) -> T_S_I_C_.table_T_S_I_C_ | None: ...
    @overload
    def get(self, tag: Literal["TSID"]) -> T_S_I_D_.table_T_S_I_D_ | None: ...
    @overload
    def get(self, tag: Literal["TSIJ"]) -> T_S_I_J_.table_T_S_I_J_ | None: ...
    @overload
    def get(self, tag: Literal["TSIP"]) -> T_S_I_P_.table_T_S_I_P_ | None: ...
    @overload
    def get(self, tag: Literal["TSIS"]) -> T_S_I_S_.table_T_S_I_S_ | None: ...
    @overload
    def get(self, tag: Literal["TSIV"]) -> T_S_I_V_.table_T_S_I_V_ | None: ...
    @overload
    def get(self, tag: Literal["TTFA"]) -> T_T_F_A_.table_T_T_F_A_ | None: ...
    @overload
    def get(self, tag: Literal["VARC"]) -> V_A_R_C_.table_V_A_R_C_ | None: ...
    @overload
    def get(self, tag: Literal["VDMX"]) -> V_D_M_X_.table_V_D_M_X_ | None: ...
    @overload
    def get(self, tag: Literal["VORG"]) -> V_O_R_G_.table_V_O_R_G_ | None: ...
    @overload
    def get(self, tag: Literal["VVAR"]) -> V_V_A_R_.table_V_V_A_R_ | None: ...
    @overload
    def get(self, tag: Literal["Debg"]) -> D__e_b_g.table_D__e_b_g | None: ...
    @overload
    def get(self, tag: Literal["Feat"]) -> F__e_a_t.table_F__e_a_t | None: ...
    @overload
    def get(self, tag: Literal["Glat"]) -> G__l_a_t.table_G__l_a_t | None: ...
    @overload
    def get(self, tag: Literal["Gloc"]) -> G__l_o_c.table_G__l_o_c | None: ...
    @overload
    def get(self, tag: Literal["OS/2"]) -> O_S_2f_2.table_O_S_2f_2 | None: ...
    @overload
    def get(self, tag: Literal["Silf"]) -> S__i_l_f.table_S__i_l_f | None: ...
    @overload
    def get(self, tag: Literal["Sill"]) -> S__i_l_l.table_S__i_l_l | None: ...
    @overload
    def get(self, tag: Literal["ankr"]) -> _a_n_k_r.table__a_n_k_r | None: ...
    @overload
    def get(self, tag: Literal["avar"]) -> _a_v_a_r.table__a_v_a_r | None: ...
    @overload
    def get(self, tag: Literal["bsln"]) -> _b_s_l_n.table__b_s_l_n | None: ...
    @overload
    def get(self, tag: Literal["bgcl"]) -> _b_g_c_l.table__b_g_c_l | None: ...
    @overload
    def get(self, tag: Literal["cidg"]) -> _c_i_d_g.table__c_i_d_g | None: ...
    @overload
    def get(self, tag: Literal["cmap"]) -> _c_m_a_p.table__c_m_a_p | None: ...
    @overload
    def get(self, tag: Literal["cvar"]) -> _c_v_a_r.table__c_v_a_r | None: ...
    @overload
    def get(self, tag: Literal["cvt "]) -> _c_v_t.table__c_v_t | None: ...
    @overload
    def get(self, tag: Literal["feat"]) -> _f_e_a_t.table__f_e_a_t | None: ...
    @overload
    def get(self, tag: Literal["fpgm"]) -> _f_p_g_m.table__f_p_g_m | None: ...
    @overload
    def get(self, tag: Literal["fvar"]) -> _f_v_a_r.table__f_v_a_r | None: ...
    @overload
    def get(self, tag: Literal["gasp"]) -> _g_a_s_p.table__g_a_s_p | None: ...
    @overload
    def get(self, tag: Literal["gcid"]) -> _g_c_i_d.table__g_c_i_d | None: ...
    @overload
    def get(self, tag: Literal["glyf"]) -> _g_l_y_f.table__g_l_y_f | None: ...
    @overload
    def get(self, tag: Literal["gvar"]) -> _g_v_a_r.table__g_v_a_r | None: ...
    @overload
    def get(self, tag: Literal["hdmx"]) -> _h_d_m_x.table__h_d_m_x | None: ...
    @overload
    def get(self, tag: Literal["head"]) -> _h_e_a_d.table__h_e_a_d | None: ...
    @overload
    def get(self, tag: Literal["hhea"]) -> _h_h_e_a.table__h_h_e_a | None: ...
    @overload
    def get(self, tag: Literal["hmtx"]) -> _h_m_t_x.table__h_m_t_x | None: ...
    @overload
    def get(self, tag: Literal["kern"]) -> _k_e_r_n.table__k_e_r_n | None: ...
    @overload
    def get(self, tag: Literal["lcar"]) -> _l_c_a_r.table__l_c_a_r | None: ...
    @overload
    def get(self, tag: Literal["loca"]) -> _l_o_c_a.table__l_o_c_a | None: ...
    @overload
    def get(self, tag: Literal["ltag"]) -> _l_t_a_g.table__l_t_a_g | None: ...
    @overload
    def get(self, tag: Literal["maxp"]) -> _m_a_x_p.table__m_a_x_p | None: ...
    @overload
    def get(self, tag: Literal["meta"]) -> _m_e_t_a.table__m_e_t_a | None: ...
    @overload
    def get(self, tag: Literal["mort"]) -> _m_o_r_t.table__m_o_r_t | None: ...
    @overload
    def get(self, tag: Literal["morx"]) -> _m_o_r_x.table__m_o_r_x | None: ...
    @overload
    def get(self, tag: Literal["name"]) -> _n_a_m_e.table__n_a_m_e | None: ...
    @overload
    def get(self, tag: Literal["opbd"]) -> _o_p_b_d.table__o_p_b_d | None: ...
    @overload
    def get(self, tag: Literal["post"]) -> _p_o_s_t.table__p_o_s_t | None: ...
    @overload
    def get(self, tag: Literal["prep"]) -> _p_r_e_p.table__p_r_e_p | None: ...
    @overload
    def get(self, tag: Literal["prop"]) -> _p_r_o_p.table__p_r_o_p | None: ...
    @overload
    def get(self, tag: Literal["sbix"]) -> _s_b_i_x.table__s_b_i_x | None: ...
    @overload
    def get(self, tag: Literal["trak"]) -> _t_r_a_k.table__t_r_a_k | None: ...
    @overload
    def get(self, tag: Literal["vhea"]) -> _v_h_e_a.table__v_h_e_a | None: ...
    @overload
    def get(self, tag: Literal["vmtx"]) -> _v_m_t_x.table__v_m_t_x | None: ...
    @overload
    def get(self, tag: Literal["GlyphOrder"]) -> GlyphOrder: ...
    @overload
    def get(self, tag: str | bytes) -> DefaultTable | GlyphOrder | Any | None: ...
    @overload
    def get(
        self, tag: str | bytes, default: _VT_co
    ) -> DefaultTable | GlyphOrder | Any | _VT_co: ...

    def get(
        self, tag: str | bytes, default: Any | None = None
    ) -> DefaultTable | GlyphOrder | Any | None:
        """Returns the table if it exists or (optionally) a default if it doesn't."""
        try:
            return self[tag]
        except KeyError:
            return default

    def setGlyphOrder(self, glyphOrder: list[str]) -> None:
        """Set the glyph order

        Args:
                glyphOrder ([str]): List of glyph names in order.
        """
        self.glyphOrder = glyphOrder
        if hasattr(self, "_reverseGlyphOrderDict"):
            del self._reverseGlyphOrderDict
        if self.isLoaded("glyf"):
            self["glyf"].setGlyphOrder(glyphOrder)

    def getGlyphOrder(self) -> list[str]:
        """Returns a list of glyph names ordered by their position in the font."""
        try:
            return self.glyphOrder
        except AttributeError:
            pass
        if "CFF " in self:
            cff = self["CFF "]
            self.glyphOrder = cff.getGlyphOrder()
        elif "post" in self:
            # TrueType font
            glyphOrder = self["post"].getGlyphOrder()
            if glyphOrder is None:
                #
                # No names found in the 'post' table.
                # Try to create glyph names from the unicode cmap (if available)
                # in combination with the Adobe Glyph List (AGL).
                #
                self._getGlyphNamesFromCmap()
            elif len(glyphOrder) < self["maxp"].numGlyphs:
                #
                # Not enough names found in the 'post' table.
                # Can happen when 'post' format 1 is improperly used on a font that
                # has more than 258 glyphs (the length of 'standardGlyphOrder').
                #
                log.warning(
                    "Not enough names found in the 'post' table, generating them from cmap instead"
                )
                self._getGlyphNamesFromCmap()
            else:
                self.glyphOrder = glyphOrder
        else:
            self._getGlyphNamesFromCmap()
        return self.glyphOrder

    def _getGlyphNamesFromCmap(self) -> None:
        #
        # This is rather convoluted, but then again, it's an interesting problem:
        # - we need to use the unicode values found in the cmap table to
        #   build glyph names (eg. because there is only a minimal post table,
        #   or none at all).
        # - but the cmap parser also needs glyph names to work with...
        # So here's what we do:
        # - make up glyph names based on glyphID
        # - load a temporary cmap table based on those names
        # - extract the unicode values, build the "real" glyph names
        # - unload the temporary cmap table
        #
        if self.isLoaded("cmap"):
            # Bootstrapping: we're getting called by the cmap parser
            # itself. This means self.tables['cmap'] contains a partially
            # loaded cmap, making it impossible to get at a unicode
            # subtable here. We remove the partially loaded cmap and
            # restore it later.
            # This only happens if the cmap table is loaded before any
            # other table that does f.getGlyphOrder()  or f.getGlyphName().
            cmapLoading = self.tables["cmap"]
            del self.tables["cmap"]
        else:
            cmapLoading = None
        # Make up glyph names based on glyphID, which will be used by the
        # temporary cmap and by the real cmap in case we don't find a unicode
        # cmap.
        numGlyphs = int(self["maxp"].numGlyphs)
        glyphOrder = ["glyph%.5d" % i for i in range(numGlyphs)]
        glyphOrder[0] = ".notdef"
        # Set the glyph order, so the cmap parser has something
        # to work with (so we don't get called recursively).
        self.glyphOrder = glyphOrder

        # Make up glyph names based on the reversed cmap table. Because some
        # glyphs (eg. ligatures or alternates) may not be reachable via cmap,
        # this naming table will usually not cover all glyphs in the font.
        # If the font has no Unicode cmap table, reversecmap will be empty.
        if "cmap" in self:
            reversecmap = self["cmap"].buildReversedMin()
        else:
            reversecmap = {}
        useCount = {}
        for i, tempName in enumerate(glyphOrder):
            if tempName in reversecmap:
                # If a font maps both U+0041 LATIN CAPITAL LETTER A and
                # U+0391 GREEK CAPITAL LETTER ALPHA to the same glyph,
                # we prefer naming the glyph as "A".
                glyphName = self._makeGlyphName(reversecmap[tempName])
                numUses = useCount[glyphName] = useCount.get(glyphName, 0) + 1
                if numUses > 1:
                    glyphName = "%s.alt%d" % (glyphName, numUses - 1)
                glyphOrder[i] = glyphName

        if "cmap" in self:
            # Delete the temporary cmap table from the cache, so it can
            # be parsed again with the right names.
            del self.tables["cmap"]
            self.glyphOrder = glyphOrder
            if cmapLoading:
                # restore partially loaded cmap, so it can continue loading
                # using the proper names.
                self.tables["cmap"] = cmapLoading

    @staticmethod
    def _makeGlyphName(codepoint: int) -> str:
        from fontTools import agl  # Adobe Glyph List

        if codepoint in agl.UV2AGL:
            return agl.UV2AGL[codepoint]
        elif codepoint <= 0xFFFF:
            return "uni%04X" % codepoint
        else:
            return "u%X" % codepoint

    def getGlyphNames(self) -> list[str]:
        """Get a list of glyph names, sorted alphabetically."""
        glyphNames = sorted(self.getGlyphOrder())
        return glyphNames

    def getGlyphNames2(self) -> list[str]:
        """Get a list of glyph names, sorted alphabetically,
        but not case sensitive.
        """
        from fontTools.misc import textTools

        return textTools.caselessSort(self.getGlyphOrder())

    def getGlyphName(self, glyphID: int) -> str:
        """Returns the name for the glyph with the given ID.

        If no name is available, synthesises one with the form ``glyphXXXXX``` where
        ```XXXXX`` is the zero-padded glyph ID.
        """
        try:
            return self.getGlyphOrder()[glyphID]
        except IndexError:
            return "glyph%.5d" % glyphID

    def getGlyphNameMany(self, lst: Sequence[int]) -> list[str]:
        """Converts a list of glyph IDs into a list of glyph names."""
        glyphOrder = self.getGlyphOrder()
        cnt = len(glyphOrder)
        return [glyphOrder[gid] if gid < cnt else "glyph%.5d" % gid for gid in lst]

    def getGlyphID(self, glyphName: str) -> int:
        """Returns the ID of the glyph with the given name."""
        try:
            return self.getReverseGlyphMap()[glyphName]
        except KeyError:
            if glyphName[:5] == "glyph":
                try:
                    return int(glyphName[5:])
                except (NameError, ValueError):
                    raise KeyError(glyphName)
            raise

    def getGlyphIDMany(self, lst: Sequence[str]) -> list[int]:
        """Converts a list of glyph names into a list of glyph IDs."""
        d = self.getReverseGlyphMap()
        try:
            return [d[glyphName] for glyphName in lst]
        except KeyError:
            getGlyphID = self.getGlyphID
            return [getGlyphID(glyphName) for glyphName in lst]

    def getReverseGlyphMap(self, rebuild: bool = False) -> dict[str, int]:
        """Returns a mapping of glyph names to glyph IDs."""
        if rebuild or not hasattr(self, "_reverseGlyphOrderDict"):
            self._buildReverseGlyphOrderDict()
        return self._reverseGlyphOrderDict

    def _buildReverseGlyphOrderDict(self) -> dict[str, int]:
        self._reverseGlyphOrderDict = d = {}
        for glyphID, glyphName in enumerate(self.getGlyphOrder()):
            d[glyphName] = glyphID
        return d

    def _writeTable(
        self,
        tag: str | bytes,
        writer: SFNTWriter,
        done: list[str | bytes],  # Use list as original
        tableCache: MutableMapping[tuple[Tag, bytes], DefaultTable] | None = None,
    ) -> None:
        """Internal helper function for self.save(). Keeps track of
        inter-table dependencies.
        """
        if tag in done:
            return
        tableClass = getTableClass(tag)
        for masterTable in tableClass.dependencies:
            if masterTable not in done:
                if masterTable in self:
                    self._writeTable(masterTable, writer, done, tableCache)
                else:
                    done.append(masterTable)
        done.append(tag)
        tabledata = self.getTableData(tag)
        if tableCache is not None:
            entry = tableCache.get((Tag(tag), tabledata))
            if entry is not None:
                log.debug("reusing '%s' table", tag)
                writer.setEntry(tag, entry)
                return
        log.debug("Writing '%s' table to disk", tag)
        writer[tag] = tabledata
        if tableCache is not None:
            tableCache[(Tag(tag), tabledata)] = writer[tag]

    def getTableData(self, tag: str | bytes) -> bytes:
        """Returns the binary representation of a table.

        If the table is currently loaded and in memory, the data is compiled to
        binary and returned; if it is not currently loaded, the binary data is
        read from the font file and returned.
        """
        tag = Tag(tag)
        if self.isLoaded(tag):
            log.debug("Compiling '%s' table", tag)
            return self.tables[tag].compile(self)
        elif self.reader and tag in self.reader:
            log.debug("Reading '%s' table from disk", tag)
            return self.reader[tag]
        else:
            raise KeyError(tag)

    def getGlyphSet(
        self,
        preferCFF: bool = True,
        location: Mapping[str, _NumberT] | None = None,
        normalized: bool = False,
        recalcBounds: bool = True,
    ) -> _TTGlyphSet:
        """Return a generic GlyphSet, which is a dict-like object
        mapping glyph names to glyph objects. The returned glyph objects
        have a ``.draw()`` method that supports the Pen protocol, and will
        have an attribute named 'width'.

        If the font is CFF-based, the outlines will be taken from the ``CFF ``
        or ``CFF2`` tables. Otherwise the outlines will be taken from the
        ``glyf`` table.

        If the font contains both a ``CFF ``/``CFF2`` and a ``glyf`` table, you
        can use the ``preferCFF`` argument to specify which one should be taken.
        If the font contains both a ``CFF `` and a ``CFF2`` table, the latter is
        taken.

        If the ``location`` parameter is set, it should be a dictionary mapping
        four-letter variation tags to their float values, and the returned
        glyph-set will represent an instance of a variable font at that
        location.

        If the ``normalized`` variable is set to True, that location is
        interpreted as in the normalized (-1..+1) space, otherwise it is in the
        font's defined axes space.
        """
        if location and "fvar" not in self:
            location = None
        if location and not normalized:
            location = self.normalizeLocation(location)
        glyphSet = None
        if ("CFF " in self or "CFF2" in self) and (preferCFF or "glyf" not in self):
            glyphSet = _TTGlyphSetCFF(self, location)
        elif "glyf" in self:
            glyphSet = _TTGlyphSetGlyf(self, location, recalcBounds=recalcBounds)
        else:
            raise TTLibError("Font contains no outlines")
        if "VARC" in self:
            glyphSet = _TTGlyphSetVARC(self, location, glyphSet)
        return glyphSet

    def normalizeLocation(self, location: Mapping[str, float]) -> dict[str, float]:
        """Normalize a ``location`` from the font's defined axes space (also
        known as user space) into the normalized (-1..+1) space. It applies
        ``avar`` mapping if the font contains an ``avar`` table.

        The ``location`` parameter should be a dictionary mapping four-letter
        variation tags to their float values.

        Raises ``TTLibError`` if the font is not a variable font.
        """
        from fontTools.varLib.models import normalizeLocation

        if "fvar" not in self:
            raise TTLibError("Not a variable font")

        axes = self["fvar"].getAxes()
        location = normalizeLocation(location, axes)
        if "avar" in self:
            location = self["avar"].renormalizeLocation(location, self)
        return location

    def getBestCmap(
        self,
        cmapPreferences: Sequence[tuple[int, int]] = (
            (3, 10),
            (0, 6),
            (0, 4),
            (3, 1),
            (0, 3),
            (0, 2),
            (0, 1),
            (0, 0),
        ),
    ) -> dict[int, str] | None:
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
        return self["cmap"].getBestCmap(cmapPreferences=cmapPreferences)

    def reorderGlyphs(self, new_glyph_order: list[str]) -> None:
        from .reorderGlyphs import reorderGlyphs

        reorderGlyphs(self, new_glyph_order)


class GlyphOrder(object):
    """A pseudo table. The glyph order isn't in the font as a separate
    table, but it's nice to present it as such in the TTX format.
    """

    def __init__(self, tag: str | None = None) -> None:
        pass

    def toXML(self, writer: xmlWriter.XMLWriter, ttFont: TTFont) -> None:
        glyphOrder = ttFont.getGlyphOrder()
        writer.comment(
            "The 'id' attribute is only for humans; it is ignored when parsed."
        )
        writer.newline()
        for i, glyphName in enumerate(glyphOrder):
            writer.simpletag("GlyphID", id=i, name=glyphName)
            writer.newline()

    def fromXML(
        self, name: str, attrs: dict[str, str], content: list[Any], ttFont: TTFont
    ) -> None:
        if not hasattr(self, "glyphOrder"):
            self.glyphOrder = []
        if name == "GlyphID":
            self.glyphOrder.append(attrs["name"])
        ttFont.setGlyphOrder(self.glyphOrder)


def getTableModule(tag: str | bytes) -> ModuleType | None:
    """Fetch the packer/unpacker module for a table.
    Return None when no module is found.
    """
    from . import tables

    pyTag = tagToIdentifier(tag)
    try:
        __import__("fontTools.ttLib.tables." + pyTag)
    except ImportError as err:
        # If pyTag is found in the ImportError message,
        # means table is not implemented.  If it's not
        # there, then some other module is missing, don't
        # suppress the error.
        if str(err).find(pyTag) >= 0:
            return None
        else:
            raise err
    else:
        return getattr(tables, pyTag)


# Registry for custom table packer/unpacker classes. Keys are table
# tags, values are (moduleName, className) tuples.
# See registerCustomTableClass() and getCustomTableClass()
_customTableRegistry: dict[str | bytes, tuple[str, str]] = {}


def registerCustomTableClass(
    tag: str | bytes, moduleName: str, className: str | None = None
) -> None:
    """Register a custom packer/unpacker class for a table.

    The 'moduleName' must be an importable module. If no 'className'
    is given, it is derived from the tag, for example it will be
    ``table_C_U_S_T_`` for a 'CUST' tag.

    The registered table class should be a subclass of
    :py:class:`fontTools.ttLib.tables.DefaultTable.DefaultTable`
    """
    if className is None:
        className = "table_" + tagToIdentifier(tag)
    _customTableRegistry[tag] = (moduleName, className)


def unregisterCustomTableClass(tag: str | bytes) -> None:
    """Unregister the custom packer/unpacker class for a table."""
    del _customTableRegistry[tag]


def getCustomTableClass(tag: str | bytes) -> type[DefaultTable] | None:
    """Return the custom table class for tag, if one has been registered
    with 'registerCustomTableClass()'. Else return None.
    """
    if tag not in _customTableRegistry:
        return None
    import importlib

    moduleName, className = _customTableRegistry[tag]
    module = importlib.import_module(moduleName)
    return getattr(module, className)


def getTableClass(tag: str | bytes) -> type[DefaultTable]:
    """Fetch the packer/unpacker class for a table."""
    tableClass = getCustomTableClass(tag)
    if tableClass is not None:
        return tableClass
    module = getTableModule(tag)
    if module is None:
        from .tables.DefaultTable import DefaultTable

        return DefaultTable
    pyTag = tagToIdentifier(tag)
    tableClass = getattr(module, "table_" + pyTag)
    return tableClass


def getClassTag(klass: type[DefaultTable]) -> str | bytes:
    """Fetch the table tag for a class object."""
    name = klass.__name__
    assert name[:6] == "table_"
    name = name[6:]  # Chop 'table_'
    return identifierToTag(name)


def newTable(tag: str | bytes) -> DefaultTable:
    """Return a new instance of a table."""
    tableClass = getTableClass(tag)
    return tableClass(tag)


def _escapechar(c: str) -> str:
    """Helper function for tagToIdentifier()"""
    import re

    if re.match("[a-z0-9]", c):
        return "_" + c
    elif re.match("[A-Z]", c):
        return c + "_"
    else:
        return hex(byteord(c))[2:]


def tagToIdentifier(tag: str | bytes) -> str:
    """Convert a table tag to a valid (but UGLY) python identifier,
    as well as a filename that's guaranteed to be unique even on a
    caseless file system. Each character is mapped to two characters.
    Lowercase letters get an underscore before the letter, uppercase
    letters get an underscore after the letter. Trailing spaces are
    trimmed. Illegal characters are escaped as two hex bytes. If the
    result starts with a number (as the result of a hex escape), an
    extra underscore is prepended. Examples:
    .. code-block:: pycon

        >>>
        >> tagToIdentifier('glyf')
        '_g_l_y_f'
        >> tagToIdentifier('cvt ')
        '_c_v_t'
        >> tagToIdentifier('OS/2')
        'O_S_2f_2'
    """
    import re

    tag = Tag(tag)
    if tag == "GlyphOrder":
        return tag
    assert len(tag) == 4, "tag should be 4 characters long"
    while len(tag) > 1 and tag[-1] == " ":
        tag = tag[:-1]
    ident = ""
    for c in tag:
        ident = ident + _escapechar(c)
    if re.match("[0-9]", ident):
        ident = "_" + ident
    return ident


def identifierToTag(ident: str) -> str:
    """the opposite of tagToIdentifier()"""
    if ident == "GlyphOrder":
        return ident
    if len(ident) % 2 and ident[0] == "_":
        ident = ident[1:]
    assert not (len(ident) % 2)
    tag = ""
    for i in range(0, len(ident), 2):
        if ident[i] == "_":
            tag = tag + ident[i + 1]
        elif ident[i + 1] == "_":
            tag = tag + ident[i]
        else:
            # assume hex
            tag = tag + chr(int(ident[i : i + 2], 16))
    # append trailing spaces
    tag = tag + (4 - len(tag)) * " "
    return Tag(tag)


def tagToXML(tag: str | bytes) -> str:
    """Similarly to tagToIdentifier(), this converts a TT tag
    to a valid XML element name. Since XML element names are
    case sensitive, this is a fairly simple/readable translation.
    """
    import re

    tag = Tag(tag)
    if tag == "OS/2":
        return "OS_2"
    elif tag == "GlyphOrder":
        return tag
    if re.match("[A-Za-z_][A-Za-z_0-9]* *$", tag):
        return tag.strip()
    else:
        return tagToIdentifier(tag)


def xmlToTag(tag: str) -> str:
    """The opposite of tagToXML()"""
    if tag == "OS_2":
        return Tag("OS/2")
    if len(tag) == 8:
        return identifierToTag(tag)
    else:
        return Tag(tag + " " * (4 - len(tag)))


# Table order as recommended in the OpenType specification 1.4
TTFTableOrder = [
    "head",
    "hhea",
    "maxp",
    "OS/2",
    "hmtx",
    "LTSH",
    "VDMX",
    "hdmx",
    "cmap",
    "fpgm",
    "prep",
    "cvt ",
    "loca",
    "glyf",
    "kern",
    "name",
    "post",
    "gasp",
    "PCLT",
]

OTFTableOrder = ["head", "hhea", "maxp", "OS/2", "name", "cmap", "post", "CFF "]


def sortedTagList(
    tagList: Sequence[str], tableOrder: Sequence[str] | None = None
) -> list[str]:
    """Return a sorted copy of tagList, sorted according to the OpenType
    specification, or according to a custom tableOrder. If given and not
    None, tableOrder needs to be a list of tag names.
    """
    tagList = sorted(tagList)
    if tableOrder is None:
        if "DSIG" in tagList:
            # DSIG should be last (XXX spec reference?)
            tagList.remove("DSIG")
            tagList.append("DSIG")
        if "CFF " in tagList:
            tableOrder = OTFTableOrder
        else:
            tableOrder = TTFTableOrder
    orderedTables = []
    for tag in tableOrder:
        if tag in tagList:
            orderedTables.append(tag)
            tagList.remove(tag)
    orderedTables.extend(tagList)
    return orderedTables


def reorderFontTables(
    inFile: BinaryIO,  # Takes file-like object as per original
    outFile: BinaryIO,  # Takes file-like object
    tableOrder: Sequence[str] | None = None,
    checkChecksums: bool = False,  # Keep param even if reader handles it
) -> None:
    """Rewrite a font file, ordering the tables as recommended by the
    OpenType specification 1.4.
    """
    inFile.seek(0)
    outFile.seek(0)
    reader = SFNTReader(inFile, checkChecksums=checkChecksums)
    writer = SFNTWriter(
        outFile,
        len(reader.tables),
        reader.sfntVersion,
        reader.flavor,
        reader.flavorData,
    )
    tables = list(reader.keys())
    for tag in sortedTagList(tables, tableOrder):
        writer[tag] = reader[tag]
    writer.close()


def maxPowerOfTwo(x: int) -> int:
    """Return the highest exponent of two, so that
    (2 ** exponent) <= x.  Return 0 if x is 0.
    """
    exponent = 0
    while x:
        x = x >> 1
        exponent = exponent + 1
    return max(exponent - 1, 0)


def getSearchRange(n: int, itemSize: int = 16) -> tuple[int, int, int]:
    """Calculate searchRange, entrySelector, rangeShift."""
    # itemSize defaults to 16, for backward compatibility
    # with upstream fonttools.
    exponent = maxPowerOfTwo(n)
    searchRange = (2**exponent) * itemSize
    entrySelector = exponent
    rangeShift = max(0, n * itemSize - searchRange)
    return searchRange, entrySelector, rangeShift
<|endoftext|>import sys
from fontTools.ttLib import OPTIMIZE_FONT_SPEED, TTLibError, TTLibFileIsCollectionError
from fontTools.ttLib.ttFont import *
from fontTools.ttLib.ttCollection import TTCollection


def main(args=None):
    """Open/save fonts with TTFont() or TTCollection()

      ./fonttools ttLib [-oFILE] [-yNUMBER] files...

    If multiple files are given on the command-line,
    they are each opened (as a font or collection),
    and added to the font list.

    If -o (output-file) argument is given, the font
    list is then saved to the output file, either as
    a single font, if there is only one font, or as
    a collection otherwise.

    If -y (font-number) argument is given, only the
    specified font from collections is opened.

    The above allow extracting a single font from a
    collection, or combining multiple fonts into a
    collection.

    If --lazy or --no-lazy are give, those are passed
    to the TTFont() or TTCollection() constructors.
    """
    from fontTools import configLogger

    if args is None:
        args = sys.argv[1:]

    import argparse

    parser = argparse.ArgumentParser(
        "fonttools ttLib",
        description="Open/save fonts with TTFont() or TTCollection()",
        epilog="""
		If multiple files are given on the command-line,
		they are each opened (as a font or collection),
		and added to the font list.

		The above, when combined with -o / --output,
		allows for extracting a single font from a
		collection, or combining multiple fonts into a
		collection.
		""",
    )
    parser.add_argument("font", metavar="font", nargs="*", help="Font file.")
    parser.add_argument(
        "-t", "--table", metavar="table", action="append", help="Tables to decompile."
    )
    parser.add_argument(
        "-o", "--output", metavar="FILE", default=None, help="Output file."
    )
    parser.add_argument(
        "-y", metavar="NUMBER", default=-1, help="Font number to load from collections."
    )
    parser.add_argument(
        "--lazy", action="store_true", default=None, help="Load fonts lazily."
    )
    parser.add_argument(
        "--no-lazy", dest="lazy", action="store_false", help="Load fonts immediately."
    )
    parser.add_argument(
        "--flavor",
        dest="flavor",
        default=None,
        help="Flavor of output font. 'woff' or 'woff2'.",
    )
    parser.add_argument(
        "--no-recalc-timestamp",
        dest="recalcTimestamp",
        action="store_false",
        help="Keep the original font 'modified' timestamp.",
    )
    parser.add_argument(
        "-b",
        dest="recalcBBoxes",
        action="store_false",
        help="Don't recalc glyph bounding boxes: use the values in the original font.",
    )
    parser.add_argument(
        "--optimize-font-speed",
        action="store_true",
        help=(
            "Enable optimizations that prioritize speed over file size. This "
            "mainly affects how glyf table and gvar / VARC tables are compiled."
        ),
    )
    options = parser.parse_args(args)

    fontNumber = int(options.y) if options.y is not None else None
    outFile = options.output
    lazy = options.lazy
    flavor = options.flavor
    tables = options.table
    recalcBBoxes = options.recalcBBoxes
    recalcTimestamp = options.recalcTimestamp
    optimizeFontSpeed = options.optimize_font_speed

    fonts = []
    for f in options.font:
        try:
            font = TTFont(
                f,
                recalcBBoxes=recalcBBoxes,
                recalcTimestamp=recalcTimestamp,
                fontNumber=fontNumber,
                lazy=lazy,
            )
            if optimizeFontSpeed:
                font.cfg[OPTIMIZE_FONT_SPEED] = optimizeFontSpeed
            fonts.append(font)
        except TTLibFileIsCollectionError:
            collection = TTCollection(f, lazy=lazy)
            fonts.extend(collection.fonts)

    if tables is None:
        if lazy is False:
            tables = ["*"]
        elif optimizeFontSpeed:
            tables = {"glyf", "gvar", "VARC"}.intersection(font.keys())
        else:
            tables = []
    for font in fonts:
        if "GlyphOrder" in tables:
            font.getGlyphOrder()
        for table in tables if "*" not in tables else font.keys():
            font[table]  # Decompiles

    if outFile is not None:
        if len(fonts) == 1:
            fonts[0].flavor = flavor
            fonts[0].save(outFile)
        else:
            if flavor is not None:
                raise TTLibError("Cannot set flavor for collections.")
            collection = TTCollection()
            collection.fonts = fonts
            collection.save(outFile)


if __name__ == "__main__":
    sys.exit(main())
<|endoftext|>"""GlyphSets returned by a TTFont."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import contextmanager
from copy import copy, deepcopy
from types import SimpleNamespace
from fontTools.misc.vector import Vector
from fontTools.misc.fixedTools import otRound, fixedToFloat as fi2fl
from fontTools.misc.loggingTools import deprecateFunction
from fontTools.misc.transform import Transform, DecomposedTransform
from fontTools.pens.transformPen import TransformPen, TransformPointPen
from fontTools.pens.recordingPen import (
    DecomposingRecordingPen,
    lerpRecordings,
    replayRecording,
)


class _TTGlyphSet(Mapping):
    """Generic dict-like GlyphSet class that pulls metrics from hmtx and
    glyph shape from TrueType or CFF.
    """

    def __init__(self, font, location, glyphsMapping, *, recalcBounds=True):
        self.recalcBounds = recalcBounds
        self.font = font
        self.defaultLocationNormalized = (
            {axis.axisTag: 0 for axis in self.font["fvar"].axes}
            if "fvar" in self.font
            else {}
        )
        self.location = location if location is not None else {}
        self.rawLocation = {}  # VarComponent-only location
        self.originalLocation = location if location is not None else {}
        self.depth = 0
        self.locationStack = []
        self.rawLocationStack = []
        self.glyphsMapping = glyphsMapping
        self.hMetrics = font["hmtx"].metrics
        self.vMetrics = getattr(font.get("vmtx"), "metrics", None)
        self.hvarTable = None
        if location:
            from fontTools.varLib.varStore import VarStoreInstancer

            self.hvarTable = getattr(font.get("HVAR"), "table", None)
            if self.hvarTable is not None:
                self.hvarInstancer = VarStoreInstancer(
                    self.hvarTable.VarStore, font["fvar"].axes, location
                )
            # TODO VVAR, VORG

    @contextmanager
    def pushLocation(self, location, reset: bool):
        self.locationStack.append(self.location)
        self.rawLocationStack.append(self.rawLocation)
        if reset:
            self.location = self.originalLocation.copy()
            self.rawLocation = self.defaultLocationNormalized.copy()
        else:
            self.location = self.location.copy()
            self.rawLocation = {}
        self.location.update(location)
        self.rawLocation.update(location)

        try:
            yield None
        finally:
            self.location = self.locationStack.pop()
            self.rawLocation = self.rawLocationStack.pop()

    @contextmanager
    def pushDepth(self):
        try:
            depth = self.depth
            self.depth += 1
            yield depth
        finally:
            self.depth -= 1

    def __contains__(self, glyphName):
        return glyphName in self.glyphsMapping

    def __iter__(self):
        return iter(self.glyphsMapping.keys())

    def __len__(self):
        return len(self.glyphsMapping)

    @deprecateFunction(
        "use 'glyphName in glyphSet' instead", category=DeprecationWarning
    )
    def has_key(self, glyphName):
        return glyphName in self.glyphsMapping


class _TTGlyphSetGlyf(_TTGlyphSet):
    def __init__(self, font, location, recalcBounds=True):
        self.glyfTable = font["glyf"]
        super().__init__(font, location, self.glyfTable, recalcBounds=recalcBounds)
        self.gvarTable = font.get("gvar")

    def __getitem__(self, glyphName):
        return _TTGlyphGlyf(self, glyphName, recalcBounds=self.recalcBounds)


class _TTGlyphSetCFF(_TTGlyphSet):
    def __init__(self, font, location):
        tableTag = "CFF2" if "CFF2" in font else "CFF "
        self.charStrings = list(font[tableTag].cff.values())[0].CharStrings
        super().__init__(font, location, self.charStrings)
        self.setLocation(location)

    def __getitem__(self, glyphName):
        return _TTGlyphCFF(self, glyphName)

    def setLocation(self, location):
        self.blender = None
        if location:
            # TODO Optimize by using instancer.setLocation()

            from fontTools.varLib.varStore import VarStoreInstancer

            varStore = getattr(self.charStrings, "varStore", None)
            if varStore is not None:
                instancer = VarStoreInstancer(
                    varStore.otVarStore, self.font["fvar"].axes, location
                )
                self.blender = instancer.interpolateFromDeltas
        else:
            self.blender = None

    @contextmanager
    def pushLocation(self, location, reset: bool):
        self.setLocation(location)
        with _TTGlyphSet.pushLocation(self, location, reset) as value:
            try:
                yield value
            finally:
                self.setLocation(self.location)


class _TTGlyphSetVARC(_TTGlyphSet):
    def __init__(self, font, location, glyphSet):
        self.glyphSet = glyphSet
        super().__init__(font, location, glyphSet)
        self.varcTable = font["VARC"].table

    def __getitem__(self, glyphName):
        varc = self.varcTable
        if glyphName not in varc.Coverage.glyphs:
            return self.glyphSet[glyphName]
        return _TTGlyphVARC(self, glyphName)


class _TTGlyph(ABC):
    """Glyph object that supports the Pen protocol, meaning that it has
    .draw() and .drawPoints() methods that take a pen object as their only
    argument. Additionally there are 'width' and 'lsb' attributes, read from
    the 'hmtx' table.

    If the font contains a 'vmtx' table, there will also be 'height' and 'tsb'
    attributes.
    """

    def __init__(self, glyphSet, glyphName, *, recalcBounds=True):
        self.glyphSet = glyphSet
        self.name = glyphName
        self.recalcBounds = recalcBounds
        self.width, self.lsb = glyphSet.hMetrics[glyphName]
        if glyphSet.vMetrics is not None:
            self.height, self.tsb = glyphSet.vMetrics[glyphName]
        else:
            self.height, self.tsb = None, None
        if glyphSet.location and glyphSet.hvarTable is not None:
            varidx = (
                glyphSet.font.getGlyphID(glyphName)
                if glyphSet.hvarTable.AdvWidthMap is None
                else glyphSet.hvarTable.AdvWidthMap.mapping[glyphName]
            )
            self.width += glyphSet.hvarInstancer[varidx]
        # TODO: VVAR/VORG

    @abstractmethod
    def draw(self, pen):
        """Draw the glyph onto ``pen``. See fontTools.pens.basePen for details
        how that works.
        """
        raise NotImplementedError

    def drawPoints(self, pen):
        """Draw the glyph onto ``pen``. See fontTools.pens.pointPen for details
        how that works.
        """
        from fontTools.pens.pointPen import SegmentToPointPen

        self.draw(SegmentToPointPen(pen))


class _TTGlyphGlyf(_TTGlyph):
    def draw(self, pen):
        """Draw the glyph onto ``pen``. See fontTools.pens.basePen for details
        how that works.
        """
        glyph, offset = self._getGlyphAndOffset()

        with self.glyphSet.pushDepth() as depth:
            if depth:
                offset = 0  # Offset should only apply at top-level

            glyph.draw(pen, self.glyphSet.glyfTable, offset)

    def drawPoints(self, pen):
        """Draw the glyph onto ``pen``. See fontTools.pens.pointPen for details
        how that works.
        """
        glyph, offset = self._getGlyphAndOffset()

        with self.glyphSet.pushDepth() as depth:
            if depth:
                offset = 0  # Offset should only apply at top-level

            glyph.drawPoints(pen, self.glyphSet.glyfTable, offset)

    def _getGlyphAndOffset(self):
        if self.glyphSet.location and self.glyphSet.gvarTable is not None:
            glyph = self._getGlyphInstance()
        else:
            glyph = self.glyphSet.glyfTable[self.name]

        offset = self.lsb - glyph.xMin if hasattr(glyph, "xMin") else 0
        return glyph, offset

    def _getGlyphInstance(self):
        from fontTools.varLib.iup import iup_delta
        from fontTools.ttLib.tables._g_l_y_f import GlyphCoordinates
        from fontTools.varLib.models import supportScalar

        glyphSet = self.glyphSet
        glyfTable = glyphSet.glyfTable
        variations = glyphSet.gvarTable.variations[self.name]
        hMetrics = glyphSet.hMetrics
        vMetrics = glyphSet.vMetrics
        coordinates, _ = glyfTable._getCoordinatesAndControls(
            self.name, hMetrics, vMetrics
        )
        origCoords, endPts = None, None
        for var in variations:
            scalar = supportScalar(glyphSet.location, var.axes)
            if not scalar:
                continue
            delta = var.coordinates
            if None in delta:
                if origCoords is None:
                    origCoords, control = glyfTable._getCoordinatesAndControls(
                        self.name, hMetrics, vMetrics
                    )
                    endPts = (
                        control[1] if control[0] >= 1 else list(range(len(control[1])))
                    )
                delta = iup_delta(delta, origCoords, endPts)
            coordinates += GlyphCoordinates(delta) * scalar

        glyph = copy(glyfTable[self.name])  # Shallow copy
        width, lsb, height, tsb = _setCoordinates(
            glyph, coordinates, glyfTable, recalcBounds=self.recalcBounds
        )
        self.lsb = lsb
        self.tsb = tsb
        if glyphSet.hvarTable is None:
            # no HVAR: let's set metrics from the phantom points
            self.width = width
            self.height = height
        return glyph


class _TTGlyphCFF(_TTGlyph):
    def draw(self, pen):
        """Draw the glyph onto ``pen``. See fontTools.pens.basePen for details
        how that works.
        """
        self.glyphSet.charStrings[self.name].draw(pen, self.glyphSet.blender)


def _evaluateCondition(condition, fvarAxes, location, instancer):
    if condition.Format == 1:
        # ConditionAxisRange
        axisIndex = condition.AxisIndex
        axisTag = fvarAxes[axisIndex].axisTag
        axisValue = location.get(axisTag, 0)
        minValue = condition.FilterRangeMinValue
        maxValue = condition.FilterRangeMaxValue
        return minValue <= axisValue <= maxValue
    elif condition.Format == 2:
        # ConditionValue
        value = condition.DefaultValue
        value += instancer[condition.VarIdx][0]
        return value > 0
    elif condition.Format == 3:
        # ConditionAnd
        for subcondition in condition.ConditionTable:
            if not _evaluateCondition(subcondition, fvarAxes, location, instancer):
                return False
        return True
    elif condition.Format == 4:
        # ConditionOr
        for subcondition in condition.ConditionTable:
            if _evaluateCondition(subcondition, fvarAxes, location, instancer):
                return True
        return False
    elif condition.Format == 5:
        # ConditionNegate
        return not _evaluateCondition(
            condition.conditionTable, fvarAxes, location, instancer
        )
    else:
        return False  # Unkonwn condition format


class _TTGlyphVARC(_TTGlyph):
    def _draw(self, pen, isPointPen):
        """Draw the glyph onto ``pen``. See fontTools.pens.basePen for details
        how that works.
        """
        from fontTools.ttLib.tables.otTables import (
            VarComponentFlags,
            NO_VARIATION_INDEX,
        )

        glyphSet = self.glyphSet
        varc = glyphSet.varcTable
        idx = varc.Coverage.glyphs.index(self.name)
        glyph = varc.VarCompositeGlyphs.VarCompositeGlyph[idx]

        from fontTools.varLib.multiVarStore import MultiVarStoreInstancer
        from fontTools.varLib.varStore import VarStoreInstancer

        fvarAxes = glyphSet.font["fvar"].axes
        instancer = MultiVarStoreInstancer(
            varc.MultiVarStore, fvarAxes, self.glyphSet.location
        )

        for comp in glyph.components:
            if comp.flags & VarComponentFlags.HAVE_CONDITION:
                condition = varc.ConditionList.ConditionTable[comp.conditionIndex]
                if not _evaluateCondition(
                    condition, fvarAxes, self.glyphSet.location, instancer
                ):
                    continue

            location = {}
            if comp.axisIndicesIndex is not None:
                axisIndices = varc.AxisIndicesList.Item[comp.axisIndicesIndex]
                axisValues = Vector(comp.axisValues)
                if comp.axisValuesVarIndex != NO_VARIATION_INDEX:
                    axisValues += fi2fl(instancer[comp.axisValuesVarIndex], 14)
                assert len(axisIndices) == len(axisValues), (
                    len(axisIndices),
                    len(axisValues),
                )
                location = {
                    fvarAxes[i].axisTag: v for i, v in zip(axisIndices, axisValues)
                }

            if comp.transformVarIndex != NO_VARIATION_INDEX:
                deltas = instancer[comp.transformVarIndex]
                comp = deepcopy(comp)
                comp.applyTransformDeltas(deltas)
            transform = comp.transform

            reset = comp.flags & VarComponentFlags.RESET_UNSPECIFIED_AXES
            with self.glyphSet.glyphSet.pushLocation(location, reset):
                with self.glyphSet.pushLocation(location, reset):
                    shouldDecompose = self.name == comp.glyphName

                    if not shouldDecompose:
                        try:
                            pen.addVarComponent(
                                comp.glyphName, transform, self.glyphSet.rawLocation
                            )
                        except AttributeError:
                            shouldDecompose = True

                    if shouldDecompose:
                        t = transform.toTransform()
                        compGlyphSet = (
                            self.glyphSet
                            if comp.glyphName != self.name
                            else glyphSet.glyphSet
                        )
                        g = compGlyphSet[comp.glyphName]
                        if isPointPen:
                            tPen = TransformPointPen(pen, t)
                            g.drawPoints(tPen)
                        else:
                            tPen = TransformPen(pen, t)
                            g.draw(tPen)

    def draw(self, pen):
        self._draw(pen, False)

    def drawPoints(self, pen):
        self._draw(pen, True)


def _setCoordinates(glyph, coord, glyfTable, *, recalcBounds=True):
    # Handle phantom points for (left, right, top, bottom) positions.
    assert len(coord) >= 4
    leftSideX = coord[-4][0]
    rightSideX = coord[-3][0]
    topSideY = coord[-2][1]
    bottomSideY = coord[-1][1]

    for _ in range(4):
        del coord[-1]

    if glyph.isComposite():
        assert len(coord) == len(glyph.components)
        glyph.components = [copy(comp) for comp in glyph.components]  # Shallow copy
        for p, comp in zip(coord, glyph.components):
            if hasattr(comp, "x"):
                comp.x, comp.y = p
    elif glyph.numberOfContours == 0:
        assert len(coord) == 0
    else:
        assert len(coord) == len(glyph.coordinates)
        glyph.coordinates = coord

    if recalcBounds:
        glyph.recalcBounds(glyfTable)

    horizontalAdvanceWidth = otRound(rightSideX - leftSideX)
    verticalAdvanceWidth = otRound(topSideY - bottomSideY)
    leftSideBearing = otRound(glyph.xMin - leftSideX)
    topSideBearing = otRound(topSideY - glyph.yMax)
    return (
        horizontalAdvanceWidth,
        leftSideBearing,
        verticalAdvanceWidth,
        topSideBearing,
    )


class LerpGlyphSet(Mapping):
    """A glyphset that interpolates between two other glyphsets.

    Factor is typically between 0 and 1. 0 means the first glyphset,
    1 means the second glyphset, and 0.5 means the average of the
    two glyphsets. Other values are possible, and can be useful to
    extrapolate. Defaults to 0.5.
    """

    def __init__(self, glyphset1, glyphset2, factor=0.5):
        self.glyphset1 = glyphset1
        self.glyphset2 = glyphset2
        self.factor = factor

    def __getitem__(self, glyphname):
        if glyphname in self.glyphset1 and glyphname in self.glyphset2:
            return LerpGlyph(glyphname, self)
        raise KeyError(glyphname)

    def __contains__(self, glyphname):
        return glyphname in self.glyphset1 and glyphname in self.glyphset2

    def __iter__(self):
        set1 = set(self.glyphset1)
        set2 = set(self.glyphset2)
        return iter(set1.intersection(set2))

    def __len__(self):
        set1 = set(self.glyphset1)
        set2 = set(self.glyphset2)
        return len(set1.intersection(set2))


class LerpGlyph:
    def __init__(self, glyphname, glyphset):
        self.glyphset = glyphset
        self.glyphname = glyphname

    def draw(self, pen):
        recording1 = DecomposingRecordingPen(self.glyphset.glyphset1)
        self.glyphset.glyphset1[self.glyphname].draw(recording1)
        recording2 = DecomposingRecordingPen(self.glyphset.glyphset2)
        self.glyphset.glyphset2[self.glyphname].draw(recording2)

        factor = self.glyphset.factor

        replayRecording(lerpRecordings(recording1.value, recording2.value, factor), pen)
<|endoftext|>from io import BytesIO
from fontTools.ttLib.tables.C_F_F_ import table_C_F_F_


class table_C_F_F__2(table_C_F_F_):
    """Compact Font Format version 2 table

    The ``CFF2`` table contains glyph data for a CFF2-flavored OpenType
    font.

    .. note::
       ``CFF2`` is the successor to ``CFF``, and eliminates much of
       the redundancy incurred by embedding CFF version 1 in an OpenType
       font.

    See also https://learn.microsoft.com/en-us/typography/opentype/spec/cff2
    """

    def decompile(self, data, otFont):
        self.cff.decompile(BytesIO(data), otFont, isCFF2=True)
        assert len(self.cff) == 1, "can't deal with multi-font CFF tables."

    def compile(self, otFont):
        f = BytesIO()
        self.cff.compile(f, otFont, isCFF2=True)
        return f.getvalue()
<|endoftext|>from fontTools.misc.textTools import bytesjoin, tobytes, safeEval
from . import DefaultTable
import struct

# https://developer.apple.com/fonts/TrueType-Reference-Manual/RM06/Chap6ltag.html


class table__l_t_a_g(DefaultTable.DefaultTable):
    """Language Tag table

    The AAT ``ltag`` table contains mappings between the numeric codes used
    in the language field of the ``name`` table and IETF language tags.

    See also https://developer.apple.com/fonts/TrueType-Reference-Manual/RM06/Chap6ltag.html
    """

    def __init__(self, tag=None):
        DefaultTable.DefaultTable.__init__(self, tag)
        self.version, self.flags = 1, 0
        self.tags = []

    def addTag(self, tag):
        """Add 'tag' to the list of langauge tags if not already there.

        Returns the integer index of 'tag' in the list of all tags.
        """
        try:
            return self.tags.index(tag)
        except ValueError:
            self.tags.append(tag)
            return len(self.tags) - 1

    def decompile(self, data, ttFont):
        self.version, self.flags, numTags = struct.unpack(">LLL", data[:12])
        assert self.version == 1
        self.tags = []
        for i in range(numTags):
            pos = 12 + i * 4
            offset, length = struct.unpack(">HH", data[pos : pos + 4])
            tag = data[offset : offset + length].decode("ascii")
            self.tags.append(tag)

    def compile(self, ttFont):
        dataList = [struct.pack(">LLL", self.version, self.flags, len(self.tags))]
        stringPool = ""
        for tag in self.tags:
            offset = stringPool.find(tag)
            if offset < 0:
                offset = len(stringPool)
                stringPool = stringPool + tag
            offset = offset + 12 + len(self.tags) * 4
            dataList.append(struct.pack(">HH", offset, len(tag)))
        dataList.append(tobytes(stringPool))
        return bytesjoin(dataList)

    def toXML(self, writer, ttFont):
        writer.simpletag("version", value=self.version)
        writer.newline()
        writer.simpletag("flags", value=self.flags)
        writer.newline()
        for tag in self.tags:
            writer.simpletag("LanguageTag", tag=tag)
            writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        if not hasattr(self, "tags"):
            self.tags = []
        if name == "LanguageTag":
            self.tags.append(attrs["tag"])
        elif "value" in attrs:
            value = safeEval(attrs["value"])
            setattr(self, name, value)
<|endoftext|>from fontTools.misc.fixedTools import (
    fixedToFloat as fi2fl,
    floatToFixed as fl2fi,
    floatToFixedToStr as fl2str,
    strToFixedToFloat as str2fl,
    otRound,
)
from fontTools.misc.textTools import safeEval
import array
from collections import Counter, defaultdict
import logging
import struct
import sys
import warnings

# https://www.microsoft.com/typography/otspec/otvarcommonformats.htm

EMBEDDED_PEAK_TUPLE = 0x8000
INTERMEDIATE_REGION = 0x4000
PRIVATE_POINT_NUMBERS = 0x2000

DELTAS_ARE_ZERO = 0x80
DELTAS_ARE_WORDS = 0x40
DELTAS_ARE_LONGS = 0xC0
DELTAS_SIZE_MASK = 0xC0
DELTA_RUN_COUNT_MASK = 0x3F

POINTS_ARE_WORDS = 0x80
POINT_RUN_COUNT_MASK = 0x7F

TUPLES_SHARE_POINT_NUMBERS = 0x8000
TUPLE_COUNT_MASK = 0x0FFF
TUPLE_INDEX_MASK = 0x0FFF

log = logging.getLogger(__name__)


class TupleVariation(object):
    def __init__(self, axes, coordinates):
        self.axes = axes.copy()
        self.coordinates = list(coordinates)

    def __repr__(self):
        axes = ",".join(
            sorted(["%s=%s" % (name, value) for (name, value) in self.axes.items()])
        )
        return "<TupleVariation %s %s>" % (axes, self.coordinates)

    def __eq__(self, other):
        return self.coordinates == other.coordinates and self.axes == other.axes

    def getUsedPoints(self):
        # Empty set means "all points used".
        if None not in self.coordinates:
            return frozenset()
        used = frozenset([i for i, p in enumerate(self.coordinates) if p is not None])
        # Return None if no points used.
        return used if used else None

    def hasImpact(self):
        """Returns True if this TupleVariation has any visible impact.

        If the result is False, the TupleVariation can be omitted from the font
        without making any visible difference.
        """
        return any(c is not None for c in self.coordinates)

    def toXML(self, writer, axisTags):
        writer.begintag("tuple")
        writer.newline()
        for axis in axisTags:
            value = self.axes.get(axis)
            if value is not None:
                minValue, value, maxValue = value
                defaultMinValue = min(value, 0.0)  # -0.3 --> -0.3; 0.7 --> 0.0
                defaultMaxValue = max(value, 0.0)  # -0.3 -->  0.0; 0.7 --> 0.7
                if minValue == defaultMinValue and maxValue == defaultMaxValue:
                    writer.simpletag("coord", axis=axis, value=fl2str(value, 14))
                else:
                    attrs = [
                        ("axis", axis),
                        ("min", fl2str(minValue, 14)),
                        ("value", fl2str(value, 14)),
                        ("max", fl2str(maxValue, 14)),
                    ]
                    writer.simpletag("coord", attrs)
                writer.newline()
        wrote_any_deltas = False
        for i, delta in enumerate(self.coordinates):
            if type(delta) == tuple and len(delta) == 2:
                writer.simpletag("delta", pt=i, x=delta[0], y=delta[1])
                writer.newline()
                wrote_any_deltas = True
            elif type(delta) == int:
                writer.simpletag("delta", cvt=i, value=delta)
                writer.newline()
                wrote_any_deltas = True
            elif delta is not None:
                log.error("bad delta format")
                writer.comment("bad delta #%d" % i)
                writer.newline()
                wrote_any_deltas = True
        if not wrote_any_deltas:
            writer.comment("no deltas")
            writer.newline()
        writer.endtag("tuple")
        writer.newline()

    def fromXML(self, name, attrs, _content):
        if name == "coord":
            axis = attrs["axis"]
            value = str2fl(attrs["value"], 14)
            defaultMinValue = min(value, 0.0)  # -0.3 --> -0.3; 0.7 --> 0.0
            defaultMaxValue = max(value, 0.0)  # -0.3 -->  0.0; 0.7 --> 0.7
            minValue = str2fl(attrs.get("min", defaultMinValue), 14)
            maxValue = str2fl(attrs.get("max", defaultMaxValue), 14)
            self.axes[axis] = (minValue, value, maxValue)
        elif name == "delta":
            if "pt" in attrs:
                point = safeEval(attrs["pt"])
                x = safeEval(attrs["x"])
                y = safeEval(attrs["y"])
                self.coordinates[point] = (x, y)
            elif "cvt" in attrs:
                cvt = safeEval(attrs["cvt"])
                value = safeEval(attrs["value"])
                self.coordinates[cvt] = value
            else:
                log.warning("bad delta format: %s" % ", ".join(sorted(attrs.keys())))

    def compile(
        self, axisTags, sharedCoordIndices={}, pointData=None, *, optimizeSize=True
    ):
        assert set(self.axes.keys()) <= set(axisTags), (
            "Unknown axis tag found.",
            self.axes.keys(),
            axisTags,
        )

        tupleData = []
        auxData = []

        if pointData is None:
            usedPoints = self.getUsedPoints()
            if usedPoints is None:  # Nothing to encode
                return b"", b""
            pointData = self.compilePoints(usedPoints)

        coord = self.compileCoord(axisTags)
        flags = sharedCoordIndices.get(coord)
        if flags is None:
            flags = EMBEDDED_PEAK_TUPLE
            tupleData.append(coord)

        intermediateCoord = self.compileIntermediateCoord(axisTags)
        if intermediateCoord is not None:
            flags |= INTERMEDIATE_REGION
            tupleData.append(intermediateCoord)

        # pointData of b'' implies "use shared points".
        if pointData:
            flags |= PRIVATE_POINT_NUMBERS
            auxData.append(pointData)

        auxData.append(self.compileDeltas(optimizeSize=optimizeSize))
        auxData = b"".join(auxData)

        tupleData.insert(0, struct.pack(">HH", len(auxData), flags))
        return b"".join(tupleData), auxData

    def compileCoord(self, axisTags):
        result = []
        axes = self.axes
        for axis in axisTags:
            triple = axes.get(axis)
            if triple is None:
                result.append(b"\0\0")
            else:
                result.append(struct.pack(">h", fl2fi(triple[1], 14)))
        return b"".join(result)

    def compileIntermediateCoord(self, axisTags):
        needed = False
        for axis in axisTags:
            minValue, value, maxValue = self.axes.get(axis, (0.0, 0.0, 0.0))
            defaultMinValue = min(value, 0.0)  # -0.3 --> -0.3; 0.7 --> 0.0
            defaultMaxValue = max(value, 0.0)  # -0.3 -->  0.0; 0.7 --> 0.7
            if (minValue != defaultMinValue) or (maxValue != defaultMaxValue):
                needed = True
                break
        if not needed:
            return None
        minCoords = []
        maxCoords = []
        for axis in axisTags:
            minValue, value, maxValue = self.axes.get(axis, (0.0, 0.0, 0.0))
            minCoords.append(struct.pack(">h", fl2fi(minValue, 14)))
            maxCoords.append(struct.pack(">h", fl2fi(maxValue, 14)))
        return b"".join(minCoords + maxCoords)

    @staticmethod
    def decompileCoord_(axisTags, data, offset):
        coord = {}
        pos = offset
        for axis in axisTags:
            coord[axis] = fi2fl(struct.unpack(">h", data[pos : pos + 2])[0], 14)
            pos += 2
        return coord, pos

    @staticmethod
    def compilePoints(points):
        # If the set consists of all points in the glyph, it gets encoded with
        # a special encoding: a single zero byte.
        #
        # To use this optimization, points passed in must be empty set.
        # The following two lines are not strictly necessary as the main code
        # below would emit the same. But this is most common and faster.
        if not points:
            return b"\0"

        # In the 'gvar' table, the packing of point numbers is a little surprising.
        # It consists of multiple runs, each being a delta-encoded list of integers.
        # For example, the point set {17, 18, 19, 20, 21, 22, 23} gets encoded as
        # [6, 17, 1, 1, 1, 1, 1, 1]. The first value (6) is the run length minus 1.
        # There are two types of runs, with values being either 8 or 16 bit unsigned
        # integers.
        points = list(points)
        points.sort()
        numPoints = len(points)

        result = bytearray()
        # The binary representation starts with the total number of points in the set,
        # encoded into one or two bytes depending on the value.
        if numPoints < 0x80:
            result.append(numPoints)
        else:
            result.append((numPoints >> 8) | 0x80)
            result.append(numPoints & 0xFF)

        MAX_RUN_LENGTH = 127
        pos = 0
        lastValue = 0
        while pos < numPoints:
            runLength = 0

            headerPos = len(result)
            result.append(0)

            useByteEncoding = None
            while pos < numPoints and runLength <= MAX_RUN_LENGTH:
                curValue = points[pos]
                delta = curValue - lastValue
                if useByteEncoding is None:
                    useByteEncoding = 0 <= delta <= 0xFF
                if useByteEncoding and (delta > 0xFF or delta < 0):
                    # we need to start a new run (which will not use byte encoding)
                    break
                # TODO This never switches back to a byte-encoding from a short-encoding.
                # That's suboptimal.
                if useByteEncoding:
                    result.append(delta)
                else:
                    result.append(delta >> 8)
                    result.append(delta & 0xFF)
                lastValue = curValue
                pos += 1
                runLength += 1
            if useByteEncoding:
                result[headerPos] = runLength - 1
            else:
                result[headerPos] = (runLength - 1) | POINTS_ARE_WORDS

        return result

    @staticmethod
    def decompilePoints_(numPoints, data, offset, tableTag):
        """(numPoints, data, offset, tableTag) --> ([point1, point2, ...], newOffset)"""
        assert tableTag in ("cvar", "gvar")
        pos = offset
        numPointsInData = data[pos]
        pos += 1
        if (numPointsInData & POINTS_ARE_WORDS) != 0:
            numPointsInData = (numPointsInData & POINT_RUN_COUNT_MASK) << 8 | data[pos]
            pos += 1
        if numPointsInData == 0:
            return (range(numPoints), pos)

        result = []
        while len(result) < numPointsInData:
            runHeader = data[pos]
            pos += 1
            numPointsInRun = (runHeader & POINT_RUN_COUNT_MASK) + 1
            point = 0
            if (runHeader & POINTS_ARE_WORDS) != 0:
                points = array.array("H")
                pointsSize = numPointsInRun * 2
            else:
                points = array.array("B")
                pointsSize = numPointsInRun
            points.frombytes(data[pos : pos + pointsSize])
            if sys.byteorder != "big":
                points.byteswap()

            assert len(points) == numPointsInRun
            pos += pointsSize

            result.extend(points)

        # Convert relative to absolute
        absolute = []
        current = 0
        for delta in result:
            current += delta
            absolute.append(current)
        result = absolute
        del absolute

        badPoints = {str(p) for p in result if p < 0 or p >= numPoints}
        if badPoints:
            log.warning(
                "point %s out of range in '%s' table"
                % (",".join(sorted(badPoints)), tableTag)
            )
        return (result, pos)

    def compileDeltas(self, optimizeSize=True):
        deltaX = []
        deltaY = []
        if self.getCoordWidth() == 2:
            for c in self.coordinates:
                if c is None:
                    continue
                deltaX.append(c[0])
                deltaY.append(c[1])
        else:
            for c in self.coordinates:
                if c is None:
                    continue
                deltaX.append(c)
        bytearr = bytearray()
        self.compileDeltaValues_(deltaX, bytearr, optimizeSize=optimizeSize)
        self.compileDeltaValues_(deltaY, bytearr, optimizeSize=optimizeSize)
        return bytearr

    @staticmethod
    def compileDeltaValues_(deltas, bytearr=None, *, optimizeSize=True):
        """[value1, value2, value3, ...] --> bytearray

        Emits a sequence of runs. Each run starts with a
        byte-sized header whose 6 least significant bits
        (header & 0x3F) indicate how many values are encoded
        in this run. The stored length is the actual length
        minus one; run lengths are thus in the range [1..64].
        If the header byte has its most significant bit (0x80)
        set, all values in this run are zero, and no data
        follows. Otherwise, the header byte is followed by
        ((header & 0x3F) + 1) signed values.  If (header &
        0x40) is clear, the delta values are stored as signed
        bytes; if (header & 0x40) is set, the delta values are
        signed 16-bit integers.
        """  # Explaining the format because the 'gvar' spec is hard to understand.
        if bytearr is None:
            bytearr = bytearray()

        pos = 0
        numDeltas = len(deltas)

        if optimizeSize:
            while pos < numDeltas:
                value = deltas[pos]
                if value == 0:
                    pos = TupleVariation.encodeDeltaRunAsZeroes_(deltas, pos, bytearr)
                elif -128 <= value <= 127:
                    pos = TupleVariation.encodeDeltaRunAsBytes_(deltas, pos, bytearr)
                elif -32768 <= value <= 32767:
                    pos = TupleVariation.encodeDeltaRunAsWords_(deltas, pos, bytearr)
                else:
                    pos = TupleVariation.encodeDeltaRunAsLongs_(deltas, pos, bytearr)
        else:
            minVal, maxVal = min(deltas), max(deltas)
            if minVal == 0 == maxVal:
                pos = TupleVariation.encodeDeltaRunAsZeroes_(deltas, pos, bytearr)
            elif -128 <= minVal <= maxVal <= 127:
                pos = TupleVariation.encodeDeltaRunAsBytes_(
                    deltas, pos, bytearr, optimizeSize=False
                )
            elif -32768 <= minVal <= maxVal <= 32767:
                pos = TupleVariation.encodeDeltaRunAsWords_(
                    deltas, pos, bytearr, optimizeSize=False
                )
            else:
                pos = TupleVariation.encodeDeltaRunAsLongs_(
                    deltas, pos, bytearr, optimizeSize=False
                )

        assert pos == numDeltas, (pos, numDeltas)

        return bytearr

    @staticmethod
    def encodeDeltaRunAsZeroes_(deltas, offset, bytearr):
        pos = offset
        numDeltas = len(deltas)
        while pos < numDeltas and deltas[pos] == 0:
            pos += 1
        runLength = pos - offset
        while runLength >= 64:
            bytearr.append(DELTAS_ARE_ZERO | 63)
            runLength -= 64
        if runLength:
            bytearr.append(DELTAS_ARE_ZERO | (runLength - 1))
        return pos

    @staticmethod
    def encodeDeltaRunAsBytes_(deltas, offset, bytearr, optimizeSize=True):
        pos = offset
        numDeltas = len(deltas)
        while pos < numDeltas:
            value = deltas[pos]
            if not (-128 <= value <= 127):
                break
            # Within a byte-encoded run of deltas, a single zero
            # is best stored literally as 0x00 value. However,
            # if are two or more zeroes in a sequence, it is
            # better to start a new run. For example, the sequence
            # of deltas [15, 15, 0, 15, 15] becomes 6 bytes
            # (04 0F 0F 00 0F 0F) when storing the zero value
            # literally, but 7 bytes (01 0F 0F 80 01 0F 0F)
            # when starting a new run.
            if (
                optimizeSize
                and value == 0
                and pos + 1 < numDeltas
                and deltas[pos + 1] == 0
            ):
                break
            pos += 1
        runLength = pos - offset
        while runLength >= 64:
            bytearr.append(63)
            bytearr.extend(array.array("b", deltas[offset : offset + 64]))
            offset += 64
            runLength -= 64
        if runLength:
            bytearr.append(runLength - 1)
            bytearr.extend(array.array("b", deltas[offset:pos]))
        return pos

    @staticmethod
    def encodeDeltaRunAsWords_(deltas, offset, bytearr, optimizeSize=True):
        pos = offset
        numDeltas = len(deltas)
        while pos < numDeltas:
            value = deltas[pos]

            # Within a word-encoded run of deltas, it is easiest
            # to start a new run (with a different encoding)
            # whenever we encounter a zero value. For example,
            # the sequence [0x6666, 0, 0x7777] needs 7 bytes when
            # storing the zero literally (42 66 66 00 00 77 77),
            # and equally 7 bytes when starting a new run
            # (40 66 66 80 40 77 77).
            if optimizeSize and value == 0:
                break

            # Within a word-encoded run of deltas, a single value
            # in the range (-128..127) should be encoded literally
            # because it is more compact. For example, the sequence
            # [0x6666, 2, 0x7777] becomes 7 bytes when storing
            # the value literally (42 66 66 00 02 77 77), but 8 bytes
            # when starting a new run (40 66 66 00 02 40 77 77).
            if (
                optimizeSize
                and (-128 <= value <= 127)
                and pos + 1 < numDeltas
                and (-128 <= deltas[pos + 1] <= 127)
            ):
                break

            if not (-32768 <= value <= 32767):
                break

            pos += 1
        runLength = pos - offset
        while runLength >= 64:
            bytearr.append(DELTAS_ARE_WORDS | 63)
            a = array.array("h", deltas[offset : offset + 64])
            if sys.byteorder != "big":
                a.byteswap()
            bytearr.extend(a)
            offset += 64
            runLength -= 64
        if runLength:
            bytearr.append(DELTAS_ARE_WORDS | (runLength - 1))
            a = array.array("h", deltas[offset:pos])
            if sys.byteorder != "big":
                a.byteswap()
            bytearr.extend(a)
        return pos

    @staticmethod
    def encodeDeltaRunAsLongs_(deltas, offset, bytearr, optimizeSize=True):
        pos = offset
        numDeltas = len(deltas)
        while pos < numDeltas:
            value = deltas[pos]
            if optimizeSize and -32768 <= value <= 32767:
                break
            pos += 1
        runLength = pos - offset
        while runLength >= 64:
            bytearr.append(DELTAS_ARE_LONGS | 63)
            a = array.array("i", deltas[offset : offset + 64])
            if sys.byteorder != "big":
                a.byteswap()
            bytearr.extend(a)
            offset += 64
            runLength -= 64
        if runLength:
            bytearr.append(DELTAS_ARE_LONGS | (runLength - 1))
            a = array.array("i", deltas[offset:pos])
            if sys.byteorder != "big":
                a.byteswap()
            bytearr.extend(a)
        return pos

    @staticmethod
    def decompileDeltas_(numDeltas, data, offset=0):
        """(numDeltas, data, offset) --> ([delta, delta, ...], newOffset)"""
        result = []
        pos = offset
        while len(result) < numDeltas if numDeltas is not None else pos < len(data):
            runHeader = data[pos]
            pos += 1
            numDeltasInRun = (runHeader & DELTA_RUN_COUNT_MASK) + 1
            if (runHeader & DELTAS_SIZE_MASK) == DELTAS_ARE_ZERO:
                result.extend([0] * numDeltasInRun)
            else:
                if (runHeader & DELTAS_SIZE_MASK) == DELTAS_ARE_LONGS:
                    deltas = array.array("i")
                    deltasSize = numDeltasInRun * 4
                elif (runHeader & DELTAS_SIZE_MASK) == DELTAS_ARE_WORDS:
                    deltas = array.array("h")
                    deltasSize = numDeltasInRun * 2
                else:
                    deltas = array.array("b")
                    deltasSize = numDeltasInRun
                deltas.frombytes(data[pos : pos + deltasSize])
                if sys.byteorder != "big":
                    deltas.byteswap()
                assert len(deltas) == numDeltasInRun, (len(deltas), numDeltasInRun)
                pos += deltasSize
                result.extend(deltas)
        assert numDeltas is None or len(result) == numDeltas
        return (result, pos)

    @staticmethod
    def getTupleSize_(flags, axisCount):
        size = 4
        if (flags & EMBEDDED_PEAK_TUPLE) != 0:
            size += axisCount * 2
        if (flags & INTERMEDIATE_REGION) != 0:
            size += axisCount * 4
        return size

    def getCoordWidth(self):
        """Return 2 if coordinates are (x, y) as in gvar, 1 if single values
        as in cvar, or 0 if empty.
        """
        firstDelta = next((c for c in self.coordinates if c is not None), None)
        if firstDelta is None:
            return 0  # empty or has no impact
        if type(firstDelta) in (int, float):
            return 1
        if type(firstDelta) is tuple and len(firstDelta) == 2:
            return 2
        raise TypeError(
            "invalid type of delta; expected (int or float) number, or "
            "Tuple[number, number]: %r" % firstDelta
        )

    def scaleDeltas(self, scalar):
        if scalar == 1.0:
            return  # no change
        coordWidth = self.getCoordWidth()
        self.coordinates = [
            (
                None
                if d is None
                else d * scalar if coordWidth == 1 else (d[0] * scalar, d[1] * scalar)
            )
            for d in self.coordinates
        ]

    def roundDeltas(self):
        coordWidth = self.getCoordWidth()
        self.coordinates = [
            (
                None
                if d is None
                else otRound(d) if coordWidth == 1 else (otRound(d[0]), otRound(d[1]))
            )
            for d in self.coordinates
        ]

    def calcInferredDeltas(self, origCoords, endPts):
        from fontTools.varLib.iup import iup_delta

        if self.getCoordWidth() == 1:
            raise TypeError("Only 'gvar' TupleVariation can have inferred deltas")
        if None in self.coordinates:
            if len(self.coordinates) != len(origCoords):
                raise ValueError(
                    "Expected len(origCoords) == %d; found %d"
                    % (len(self.coordinates), len(origCoords))
                )
            self.coordinates = iup_delta(self.coordinates, origCoords, endPts)

    def optimize(self, origCoords, endPts, tolerance=0.5, isComposite=False):
        from fontTools.varLib.iup import iup_delta_optimize

        if isComposite:
            warnings.warn(
                "The isComposite argument is deprecated and may be removed in a future version",
                DeprecationWarning,
            )

        if None in self.coordinates:
            return  # already optimized

        deltaOpt = iup_delta_optimize(
            self.coordinates, origCoords, endPts, tolerance=tolerance
        )
        if None in deltaOpt:
            # Use "optimized" version only if smaller...
            varOpt = TupleVariation(self.axes, deltaOpt)

            # Shouldn't matter that this is different from fvar...?
            axisTags = sorted(self.axes.keys())
            tupleData, auxData = self.compile(axisTags)
            unoptimizedLength = len(tupleData) + len(auxData)
            tupleData, auxData = varOpt.compile(axisTags)
            optimizedLength = len(tupleData) + len(auxData)

            if optimizedLength < unoptimizedLength:
                self.coordinates = varOpt.coordinates

    def __imul__(self, scalar):
        self.scaleDeltas(scalar)
        return self

    def __iadd__(self, other):
        if not isinstance(other, TupleVariation):
            return NotImplemented
        deltas1 = self.coordinates
        length = len(deltas1)
        deltas2 = other.coordinates
        if len(deltas2) != length:
            raise ValueError("cannot sum TupleVariation deltas with different lengths")
        # 'None' values have different meanings in gvar vs cvar TupleVariations:
        # within the gvar, when deltas are not provided explicitly for some points,
        # they need to be inferred; whereas for the 'cvar' table, if deltas are not
        # provided for some CVT values, then no adjustments are made (i.e. None == 0).
        # Thus, we cannot sum deltas for gvar TupleVariations if they contain
        # inferred inferred deltas (the latter need to be computed first using
        # 'calcInferredDeltas' method), but we can treat 'None' values in cvar
        # deltas as if they are zeros.
        if self.getCoordWidth() == 2:
            for i, d2 in zip(range(length), deltas2):
                d1 = deltas1[i]
                try:
                    deltas1[i] = (d1[0] + d2[0], d1[1] + d2[1])
                except TypeError:
                    raise ValueError("cannot sum gvar deltas with inferred points")
        else:
            for i, d2 in zip(range(length), deltas2):
                d1 = deltas1[i]
                if d1 is not None and d2 is not None:
                    deltas1[i] = d1 + d2
                elif d1 is None and d2 is not None:
                    deltas1[i] = d2
                # elif d2 is None do nothing
        return self


def decompileSharedTuples(axisTags, sharedTupleCount, data, offset):
    result = []
    for _ in range(sharedTupleCount):
        t, offset = TupleVariation.decompileCoord_(axisTags, data, offset)
        result.append(t)
    return result


def compileSharedTuples(
    axisTags, variations, MAX_NUM_SHARED_COORDS=TUPLE_INDEX_MASK + 1
):
    coordCount = Counter()
    for var in variations:
        coord = var.compileCoord(axisTags)
        coordCount[coord] += 1
    # In python < 3.7, most_common() ordering is non-deterministic
    # so apply a sort to make sure the ordering is consistent.
    sharedCoords = sorted(
        coordCount.most_common(MAX_NUM_SHARED_COORDS),
        key=lambda item: (-item[1], item[0]),
    )
    return [c[0] for c in sharedCoords if c[1] > 1]


def compileTupleVariationStore(
    variations,
    pointCount,
    axisTags,
    sharedTupleIndices,
    useSharedPoints=True,
    *,
    optimizeSize=True,
):
    # pointCount is actually unused. Keeping for API compat.
    del pointCount
    newVariations = []
    pointDatas = []
    # Compile all points and figure out sharing if desired
    sharedPoints = None

    # Collect, count, and compile point-sets for all variation sets
    pointSetCount = defaultdict(int)
    for v in variations:
        points = v.getUsedPoints()
        if points is None:  # Empty variations
            continue
        pointSetCount[points] += 1
        newVariations.append(v)
        pointDatas.append(points)
    variations = newVariations
    del newVariations

    if not variations:
        return (0, b"", b"")

    n = len(variations[0].coordinates)
    assert all(
        len(v.coordinates) == n for v in variations
    ), "Variation sets have different sizes"

    compiledPoints = {
        pointSet: TupleVariation.compilePoints(pointSet) for pointSet in pointSetCount
    }

    tupleVariationCount = len(variations)
    tuples = []
    data = []

    if useSharedPoints:
        # Find point-set which saves most bytes.
        def key(pn):
            pointSet = pn[0]
            count = pn[1]
            return len(compiledPoints[pointSet]) * (count - 1)

        sharedPoints = max(pointSetCount.items(), key=key)[0]

        data.append(compiledPoints[sharedPoints])
        tupleVariationCount |= TUPLES_SHARE_POINT_NUMBERS

    # b'' implies "use shared points"
    pointDatas = [
        compiledPoints[points] if points != sharedPoints else b""
        for points in pointDatas
    ]

    for v, p in zip(variations, pointDatas):
        thisTuple, thisData = v.compile(
            axisTags, sharedTupleIndices, pointData=p, optimizeSize=optimizeSize
        )

        tuples.append(thisTuple)
        data.append(thisData)

    tuples = b"".join(tuples)
    data = b"".join(data)
    return tupleVariationCount, tuples, data


def decompileTupleVariationStore(
    tableTag,
    axisTags,
    tupleVariationCount,
    pointCount,
    sharedTuples,
    data,
    pos,
    dataPos,
):
    numAxes = len(axisTags)
    result = []
    if (tupleVariationCount & TUPLES_SHARE_POINT_NUMBERS) != 0:
        sharedPoints, dataPos = TupleVariation.decompilePoints_(
            pointCount, data, dataPos, tableTag
        )
    else:
        sharedPoints = []
    for _ in range(tupleVariationCount & TUPLE_COUNT_MASK):
        dataSize, flags = struct.unpack(">HH", data[pos : pos + 4])
        tupleSize = TupleVariation.getTupleSize_(flags, numAxes)
        tupleData = data[pos : pos + tupleSize]
        pointDeltaData = data[dataPos : dataPos + dataSize]
        result.append(
            decompileTupleVariation_(
                pointCount,
                sharedTuples,
                sharedPoints,
                tableTag,
                axisTags,
                tupleData,
                pointDeltaData,
            )
        )
        pos += tupleSize
        dataPos += dataSize
    return result


def decompileTupleVariation_(
    pointCount, sharedTuples, sharedPoints, tableTag, axisTags, data, tupleData
):
    assert tableTag in ("cvar", "gvar"), tableTag
    flags = struct.unpack(">H", data[2:4])[0]
    pos = 4
    if (flags & EMBEDDED_PEAK_TUPLE) == 0:
        peak = sharedTuples[flags & TUPLE_INDEX_MASK]
    else:
        peak, pos = TupleVariation.decompileCoord_(axisTags, data, pos)
    if (flags & INTERMEDIATE_REGION) != 0:
        start, pos = TupleVariation.decompileCoord_(axisTags, data, pos)
        end, pos = TupleVariation.decompileCoord_(axisTags, data, pos)
    else:
        start, end = inferRegion_(peak)
    axes = {}
    for axis in axisTags:
        region = start[axis], peak[axis], end[axis]
        if region != (0.0, 0.0, 0.0):
            axes[axis] = region
    pos = 0
    if (flags & PRIVATE_POINT_NUMBERS) != 0:
        points, pos = TupleVariation.decompilePoints_(
            pointCount, tupleData, pos, tableTag
        )
    else:
        points = sharedPoints

    deltas = [None] * pointCount

    if tableTag == "cvar":
        deltas_cvt, pos = TupleVariation.decompileDeltas_(len(points), tupleData, pos)
        for p, delta in zip(points, deltas_cvt):
            if 0 <= p < pointCount:
                deltas[p] = delta

    elif tableTag == "gvar":
        deltas_x, pos = TupleVariation.decompileDeltas_(len(points), tupleData, pos)
        deltas_y, pos = TupleVariation.decompileDeltas_(len(points), tupleData, pos)
        for p, x, y in zip(points, deltas_x, deltas_y):
            if 0 <= p < pointCount:
                deltas[p] = (x, y)

    return TupleVariation(axes, deltas)


def inferRegion_(peak):
    """Infer start and end for a (non-intermediate) region

    This helper function computes the applicability region for
    variation tuples whose INTERMEDIATE_REGION flag is not set in the
    TupleVariationHeader structure.  Variation tuples apply only to
    certain regions of the variation space; outside that region, the
    tuple has no effect.  To make the binary encoding more compact,
    TupleVariationHeaders can omit the intermediateStartTuple and
    intermediateEndTuple fields.
    """
    start, end = {}, {}
    for axis, value in peak.items():
        start[axis] = min(value, 0.0)  # -0.3 --> -0.3; 0.7 --> 0.0
        end[axis] = max(value, 0.0)  # -0.3 -->  0.0; 0.7 --> 0.7
    return (start, end)
<|endoftext|>from fontTools.misc import sstruct
from fontTools.misc.textTools import bytesjoin, strjoin, readHex
from fontTools.ttLib import TTLibError
from . import DefaultTable

# Apple's documentation of 'meta':
# https://developer.apple.com/fonts/TrueType-Reference-Manual/RM06/Chap6meta.html

META_HEADER_FORMAT = """
    > # big endian
    version:     L
    flags:       L
    dataOffset:  L
    numDataMaps: L
"""


DATA_MAP_FORMAT = """
    > # big endian
    tag:         4s
    dataOffset:  L
    dataLength:  L
"""


class table__m_e_t_a(DefaultTable.DefaultTable):
    """Metadata table

    The ``meta`` table contains various metadata values for the font. Each
    category of metadata in the table is identified by a four-character tag.

    See also https://learn.microsoft.com/en-us/typography/opentype/spec/meta
    """

    def __init__(self, tag=None):
        DefaultTable.DefaultTable.__init__(self, tag)
        self.data = {}

    def decompile(self, data, ttFont):
        headerSize = sstruct.calcsize(META_HEADER_FORMAT)
        header = sstruct.unpack(META_HEADER_FORMAT, data[0:headerSize])
        if header["version"] != 1:
            raise TTLibError("unsupported 'meta' version %d" % header["version"])
        dataMapSize = sstruct.calcsize(DATA_MAP_FORMAT)
        for i in range(header["numDataMaps"]):
            dataMapOffset = headerSize + i * dataMapSize
            dataMap = sstruct.unpack(
                DATA_MAP_FORMAT, data[dataMapOffset : dataMapOffset + dataMapSize]
            )
            tag = dataMap["tag"]
            offset = dataMap["dataOffset"]
            self.data[tag] = data[offset : offset + dataMap["dataLength"]]
            if tag in ["dlng", "slng"]:
                self.data[tag] = self.data[tag].decode("utf-8")

    def compile(self, ttFont):
        keys = sorted(self.data.keys())
        headerSize = sstruct.calcsize(META_HEADER_FORMAT)
        dataOffset = headerSize + len(keys) * sstruct.calcsize(DATA_MAP_FORMAT)
        header = sstruct.pack(
            META_HEADER_FORMAT,
            {
                "version": 1,
                "flags": 0,
                "dataOffset": dataOffset,
                "numDataMaps": len(keys),
            },
        )
        dataMaps = []
        dataBlocks = []
        for tag in keys:
            if tag in ["dlng", "slng"]:
                data = self.data[tag].encode("utf-8")
            else:
                data = self.data[tag]
            dataMaps.append(
                sstruct.pack(
                    DATA_MAP_FORMAT,
                    {"tag": tag, "dataOffset": dataOffset, "dataLength": len(data)},
                )
            )
            dataBlocks.append(data)
            dataOffset += len(data)
        return bytesjoin([header] + dataMaps + dataBlocks)

    def toXML(self, writer, ttFont):
        for tag in sorted(self.data.keys()):
            if tag in ["dlng", "slng"]:
                writer.begintag("text", tag=tag)
                writer.newline()
                writer.write(self.data[tag])
                writer.newline()
                writer.endtag("text")
                writer.newline()
            else:
                writer.begintag("hexdata", tag=tag)
                writer.newline()
                data = self.data[tag]
                if min(data) >= 0x20 and max(data) <= 0x7E:
                    writer.comment("ascii: " + data.decode("ascii"))
                    writer.newline()
                writer.dumphex(data)
                writer.endtag("hexdata")
                writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        if name == "hexdata":
            self.data[attrs["tag"]] = readHex(content)
        elif name == "text" and attrs["tag"] in ["dlng", "slng"]:
            self.data[attrs["tag"]] = strjoin(content).strip()
        else:
            raise TTLibError("can't handle '%s' element" % name)
<|endoftext|>from . import DefaultTable
import sys
import array
import logging

log = logging.getLogger(__name__)


class table__l_o_c_a(DefaultTable.DefaultTable):
    """Index to Location table

    The ``loca`` table stores the offsets in the ``glyf`` table that correspond
    to the descriptions of each glyph. The glyphs are references by Glyph ID.

    See also https://learn.microsoft.com/en-us/typography/opentype/spec/loca
    """

    dependencies = ["glyf"]

    def decompile(self, data, ttFont):
        longFormat = ttFont["head"].indexToLocFormat
        if longFormat:
            format = "I"
        else:
            format = "H"
        locations = array.array(format)
        locations.frombytes(data)
        if sys.byteorder != "big":
            locations.byteswap()
        if not longFormat:
            locations = array.array("I", (2 * l for l in locations))
        if len(locations) < (ttFont["maxp"].numGlyphs + 1):
            log.warning(
                "corrupt 'loca' table, or wrong numGlyphs in 'maxp': %d %d",
                len(locations) - 1,
                ttFont["maxp"].numGlyphs,
            )
        self.locations = locations

    def compile(self, ttFont):
        try:
            max_location = max(self.locations)
        except AttributeError:
            self.set([])
            max_location = 0
        if max_location < 0x20000 and all(l % 2 == 0 for l in self.locations):
            locations = array.array("H")
            for location in self.locations:
                locations.append(location // 2)
            ttFont["head"].indexToLocFormat = 0
        else:
            locations = array.array("I", self.locations)
            ttFont["head"].indexToLocFormat = 1
        if sys.byteorder != "big":
            locations.byteswap()
        return locations.tobytes()

    def set(self, locations):
        self.locations = array.array("I", locations)

    def toXML(self, writer, ttFont):
        writer.comment("The 'loca' table will be calculated by the compiler")
        writer.newline()

    def __getitem__(self, index):
        return self.locations[index]

    def __len__(self):
        return len(self.locations)
<|endoftext|>from __future__ import annotations

from typing import Mapping, TYPE_CHECKING
from fontTools.misc import sstruct
from fontTools.misc.fixedTools import (
    fixedToFloat as fi2fl,
    floatToFixed as fl2fi,
    floatToFixedToStr as fl2str,
    strToFixedToFloat as str2fl,
)
from fontTools.misc.textTools import bytesjoin, safeEval
from fontTools.misc.roundTools import otRound
from fontTools.varLib.models import piecewiseLinearMap
from fontTools.varLib.varStore import VarStoreInstancer, NO_VARIATION_INDEX
from fontTools.ttLib import TTLibError
from . import DefaultTable
from . import otTables
import struct
import logging

log = logging.getLogger(__name__)

from .otBase import BaseTTXConverter

if TYPE_CHECKING:
    from ..ttFont import TTFont


class table__a_v_a_r(BaseTTXConverter):
    """Axis Variations table

    This class represents the ``avar`` table of a variable font. The object has one
    substantive attribute, ``segments``, which maps axis tags to a segments dictionary::

        >>> font["avar"].segments   # doctest: +SKIP
        {'wght': {-1.0: -1.0,
          0.0: 0.0,
          0.125: 0.11444091796875,
          0.25: 0.23492431640625,
          0.5: 0.35540771484375,
          0.625: 0.5,
          0.75: 0.6566162109375,
          0.875: 0.81927490234375,
          1.0: 1.0},
         'ital': {-1.0: -1.0, 0.0: 0.0, 1.0: 1.0}}

    Notice that the segments dictionary is made up of normalized values. A valid
    ``avar`` segment mapping must contain the entries ``-1.0: -1.0, 0.0: 0.0, 1.0: 1.0``.
    fontTools does not enforce this, so it is your responsibility to ensure that
    mappings are valid.

    See also https://learn.microsoft.com/en-us/typography/opentype/spec/avar
    """

    dependencies = ["fvar"]

    def __init__(self, tag=None):
        super().__init__(tag)
        self.segments = {}

    def compile(self, ttFont):
        axisTags = [axis.axisTag for axis in ttFont["fvar"].axes]
        if not hasattr(self, "table"):
            self.table = otTables.avar()
        if not hasattr(self.table, "Reserved"):
            self.table.Reserved = 0
        self.table.Version = (getattr(self, "majorVersion", 1) << 16) | getattr(
            self, "minorVersion", 0
        )
        self.table.AxisCount = len(axisTags)
        self.table.AxisSegmentMap = []
        for axis in axisTags:
            mappings = self.segments[axis]
            segmentMap = otTables.AxisSegmentMap()
            segmentMap.PositionMapCount = len(mappings)
            segmentMap.AxisValueMap = []
            for key, value in sorted(mappings.items()):
                valueMap = otTables.AxisValueMap()
                valueMap.FromCoordinate = key
                valueMap.ToCoordinate = value
                segmentMap.AxisValueMap.append(valueMap)
            self.table.AxisSegmentMap.append(segmentMap)
        return super().compile(ttFont)

    def decompile(self, data, ttFont):
        super().decompile(data, ttFont)
        self.majorVersion = self.table.Version >> 16
        self.minorVersion = self.table.Version & 0xFFFF
        if self.majorVersion not in (1, 2):
            raise NotImplementedError("Unknown avar table version")
        axisTags = [axis.axisTag for axis in ttFont["fvar"].axes]
        for axis in axisTags:
            self.segments[axis] = {}
        for axis, segmentMap in zip(axisTags, self.table.AxisSegmentMap):
            segments = self.segments[axis] = {}
            for segment in segmentMap.AxisValueMap:
                segments[segment.FromCoordinate] = segment.ToCoordinate

    def toXML(self, writer, ttFont):
        writer.simpletag(
            "version",
            major=getattr(self, "majorVersion", 1),
            minor=getattr(self, "minorVersion", 0),
        )
        writer.newline()
        axisTags = [axis.axisTag for axis in ttFont["fvar"].axes]
        for axis in axisTags:
            writer.begintag("segment", axis=axis)
            writer.newline()
            for key, value in sorted(self.segments[axis].items()):
                key = fl2str(key, 14)
                value = fl2str(value, 14)
                writer.simpletag("mapping", **{"from": key, "to": value})
                writer.newline()
            writer.endtag("segment")
            writer.newline()
        if getattr(self, "majorVersion", 1) >= 2:
            if self.table.VarIdxMap:
                self.table.VarIdxMap.toXML(writer, ttFont, name="VarIdxMap")
            if self.table.VarStore:
                self.table.VarStore.toXML(writer, ttFont)

    def fromXML(self, name, attrs, content, ttFont):
        if not hasattr(self, "table"):
            self.table = otTables.avar()
        if not hasattr(self.table, "Reserved"):
            self.table.Reserved = 0
        if name == "version":
            self.majorVersion = safeEval(attrs["major"])
            self.minorVersion = safeEval(attrs["minor"])
            self.table.Version = (getattr(self, "majorVersion", 1) << 16) | getattr(
                self, "minorVersion", 0
            )
        elif name == "segment":
            axis = attrs["axis"]
            segment = self.segments[axis] = {}
            for element in content:
                if isinstance(element, tuple):
                    elementName, elementAttrs, _ = element
                    if elementName == "mapping":
                        fromValue = str2fl(elementAttrs["from"], 14)
                        toValue = str2fl(elementAttrs["to"], 14)
                        if fromValue in segment:
                            log.warning(
                                "duplicate entry for %s in axis '%s'", fromValue, axis
                            )
                        segment[fromValue] = toValue
        else:
            super().fromXML(name, attrs, content, ttFont)

    def renormalizeLocation(
        self, location: Mapping[str, float], font: TTFont, dropZeroes: bool = True
    ) -> dict[str, float]:
        majorVersion = getattr(self, "majorVersion", 1)

        if majorVersion not in (1, 2):
            raise NotImplementedError("Unknown avar table version")

        avarSegments = self.segments
        mappedLocation = {}
        for axisTag, value in location.items():
            avarMapping = avarSegments.get(axisTag, None)
            if avarMapping is not None:
                value = piecewiseLinearMap(value, avarMapping)
            mappedLocation[axisTag] = value

        if majorVersion < 2:
            return mappedLocation

        # Version 2

        varIdxMap = self.table.VarIdxMap
        varStore = self.table.VarStore
        axes = font["fvar"].axes
        if varStore is not None:
            varStoreInstancer = VarStoreInstancer(varStore, axes, mappedLocation)

        coords = list(fl2fi(mappedLocation.get(axis.axisTag, 0), 14) for axis in axes)

        out = []
        for varIdx, v in enumerate(coords):
            if varIdxMap is not None:
                varIdx = varIdxMap[varIdx]

            if varStore is not None:
                delta = varStoreInstancer[varIdx]
                v += otRound(delta)
                v = min(max(v, -(1 << 14)), +(1 << 14))

            out.append(v)

        mappedLocation = {
            axis.axisTag: fi2fl(v, 14)
            for v, axis in zip(out, axes)
            if v != 0 or not dropZeroes
        }

        return mappedLocation

    def renormalizeAxisLimits(self, axisLimits, font=None):
        """Apply the avar v1 segment maps to normalized axis limits.

        The values of ``axisLimits`` may be (min, default, max) triples or
        longer sequences whose trailing elements are user-space distances
        (as in NormalizedAxisTripleAndDistances); only the first three
        elements are coordinates, so only those are mapped — the distances
        pass through untouched. The mapped coordinates are quantized to
        F2Dot14, to avoid surprise interpolations.

        Returns a dict mapping axis tags to tuples of the same length as
        the input values. avar version 2 mappings are NOT applied — the
        VarStore-driven part of an avar 2 table is handled separately by
        the instancer (see fontTools.varLib.instancer).
        """
        if getattr(self, "majorVersion", 1) not in (1, 2):
            raise NotImplementedError("Unknown avar table version")

        avarSegments = self.segments
        mappedAxisLimits = {}
        for axisTag, triple in axisLimits.items():
            coords = tuple(triple)[:3]
            distances = tuple(triple)[3:]
            avarMapping = avarSegments.get(axisTag, None)
            if avarMapping is not None:
                coords = tuple(
                    piecewiseLinearMap(value, avarMapping) for value in coords
                )
            # Quantize to F2Dot14, to avoid surprise interpolations.
            coords = tuple(fi2fl(fl2fi(value, 14), 14) for value in coords)
            mappedAxisLimits[axisTag] = coords + distances
        return mappedAxisLimits
<|endoftext|>from .otBase import BaseTTXConverter


# https://developer.apple.com/fonts/TrueType-Reference-Manual/RM06/Chap6mort.html
class table__m_o_r_t(BaseTTXConverter):
    """The AAT ``mort`` table contains glyph transformations used for script shaping and
    for various other optional smart features.

    Note: ``mort`` has been deprecated in favor of the newer ``morx`` table.

    See also https://developer.apple.com/fonts/TrueType-Reference-Manual/RM06/Chap6mort.html
    """

    pass
<|endoftext|>from . import DefaultTable
from fontTools.misc import sstruct
from fontTools.misc.textTools import safeEval
import struct

VDMX_HeaderFmt = """
	>                 # big endian
	version:     H    # Version number (0 or 1)
	numRecs:     H    # Number of VDMX groups present
	numRatios:   H    # Number of aspect ratio groupings
"""
# the VMDX header is followed by an array of RatRange[numRatios] (i.e. aspect
# ratio ranges);
VDMX_RatRangeFmt = """
	>                 # big endian
	bCharSet:    B    # Character set
	xRatio:      B    # Value to use for x-Ratio
	yStartRatio: B    # Starting y-Ratio value
	yEndRatio:   B    # Ending y-Ratio value
"""
# followed by an array of offset[numRatios] from start of VDMX table to the
# VDMX Group for this ratio range (offsets will be re-calculated on compile);
# followed by an array of Group[numRecs] records;
VDMX_GroupFmt = """
	>                 # big endian
	recs:        H    # Number of height records in this group
	startsz:     B    # Starting yPelHeight
	endsz:       B    # Ending yPelHeight
"""
# followed by an array of vTable[recs] records.
VDMX_vTableFmt = """
	>                 # big endian
	yPelHeight:  H    # yPelHeight to which values apply
	yMax:        h    # Maximum value (in pels) for this yPelHeight
	yMin:        h    # Minimum value (in pels) for this yPelHeight
"""


class table_V_D_M_X_(DefaultTable.DefaultTable):
    """Vertical Device Metrics table

    The ``VDMX`` table records changes to the vertical glyph minima
    and maxima that result from Truetype instructions.

    See also https://learn.microsoft.com/en-us/typography/opentype/spec/vdmx
    """

    def decompile(self, data, ttFont):
        pos = 0  # track current position from to start of VDMX table
        dummy, data = sstruct.unpack2(VDMX_HeaderFmt, data, self)
        pos += sstruct.calcsize(VDMX_HeaderFmt)
        self.ratRanges = []
        for i in range(self.numRatios):
            ratio, data = sstruct.unpack2(VDMX_RatRangeFmt, data)
            pos += sstruct.calcsize(VDMX_RatRangeFmt)
            # the mapping between a ratio and a group is defined further below
            ratio["groupIndex"] = None
            self.ratRanges.append(ratio)
        lenOffset = struct.calcsize(">H")
        _offsets = []  # temporarily store offsets to groups
        for i in range(self.numRatios):
            offset = struct.unpack(">H", data[0:lenOffset])[0]
            data = data[lenOffset:]
            pos += lenOffset
            _offsets.append(offset)
        self.groups = []
        for groupIndex in range(self.numRecs):
            # the offset to this group from beginning of the VDMX table
            currOffset = pos
            group, data = sstruct.unpack2(VDMX_GroupFmt, data)
            # the group lenght and bounding sizes are re-calculated on compile
            recs = group.pop("recs")
            startsz = group.pop("startsz")
            endsz = group.pop("endsz")
            pos += sstruct.calcsize(VDMX_GroupFmt)
            for j in range(recs):
                vTable, data = sstruct.unpack2(VDMX_vTableFmt, data)
                vTableLength = sstruct.calcsize(VDMX_vTableFmt)
                pos += vTableLength
                # group is a dict of (yMax, yMin) tuples keyed by yPelHeight
                group[vTable["yPelHeight"]] = (vTable["yMax"], vTable["yMin"])
            # make sure startsz and endsz match the calculated values
            minSize = min(group.keys())
            maxSize = max(group.keys())
            assert (
                startsz == minSize
            ), "startsz (%s) must equal min yPelHeight (%s): group %d" % (
                group.startsz,
                minSize,
                groupIndex,
            )
            assert (
                endsz == maxSize
            ), "endsz (%s) must equal max yPelHeight (%s): group %d" % (
                group.endsz,
                maxSize,
                groupIndex,
            )
            self.groups.append(group)
            # match the defined offsets with the current group's offset
            for offsetIndex, offsetValue in enumerate(_offsets):
                # when numRecs < numRatios there can more than one ratio range
                # sharing the same VDMX group
                if currOffset == offsetValue:
                    # map the group with the ratio range thas has the same
                    # index as the offset to that group (it took me a while..)
                    self.ratRanges[offsetIndex]["groupIndex"] = groupIndex
        # check that all ratio ranges have a group
        for i in range(self.numRatios):
            ratio = self.ratRanges[i]
            if ratio["groupIndex"] is None:
                from fontTools import ttLib

                raise ttLib.TTLibError("no group defined for ratRange %d" % i)

    def _getOffsets(self):
        """
        Calculate offsets to VDMX_Group records.
        For each ratRange return a list of offset values from the beginning of
        the VDMX table to a VDMX_Group.
        """
        lenHeader = sstruct.calcsize(VDMX_HeaderFmt)
        lenRatRange = sstruct.calcsize(VDMX_RatRangeFmt)
        lenOffset = struct.calcsize(">H")
        lenGroupHeader = sstruct.calcsize(VDMX_GroupFmt)
        lenVTable = sstruct.calcsize(VDMX_vTableFmt)
        # offset to the first group
        pos = lenHeader + self.numRatios * lenRatRange + self.numRatios * lenOffset
        groupOffsets = []
        for group in self.groups:
            groupOffsets.append(pos)
            lenGroup = lenGroupHeader + len(group) * lenVTable
            pos += lenGroup  # offset to next group
        offsets = []
        for ratio in self.ratRanges:
            groupIndex = ratio["groupIndex"]
            offsets.append(groupOffsets[groupIndex])
        return offsets

    def compile(self, ttFont):
