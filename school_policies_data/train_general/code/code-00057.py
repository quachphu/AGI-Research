class FeatureParams(Table):
    def getConverter(self, featureTag):
        tableClass = self.featureParamTypes.get(featureTag, self.defaultFeatureParams)
        return self.__class__(self.name, self.repeat, self.aux, tableClass)


class ValueFormat(IntValue):
    staticSize = 2

    def __init__(self, name, repeat, aux, tableClass=None, *, description=""):
        BaseConverter.__init__(
            self, name, repeat, aux, tableClass, description=description
        )
        self.which = "ValueFormat" + ("2" if name[-1] == "2" else "1")

    def read(self, reader, font, tableDict):
        format = reader.readUShort()
        reader[self.which] = ValueRecordFactory(format)
        return format

    def write(self, writer, font, tableDict, format, repeatIndex=None):
        writer.writeUShort(format)
        writer[self.which] = ValueRecordFactory(format)


class ValueRecord(ValueFormat):
    def getRecordSize(self, reader):
        return 2 * len(reader[self.which])

    def read(self, reader, font, tableDict):
        return reader[self.which].readValueRecord(reader, font)

    def write(self, writer, font, tableDict, value, repeatIndex=None):
        writer[self.which].writeValueRecord(writer, font, value)

    def xmlWrite(self, xmlWriter, font, value, name, attrs):
        if value is None:
            pass  # NULL table, ignore
        else:
            value.toXML(xmlWriter, font, self.name, attrs)

    def xmlRead(self, attrs, content, font):
        from .otBase import ValueRecord

        value = ValueRecord()
        value.fromXML(None, attrs, content, font)
        return value


class AATLookup(BaseConverter):
    BIN_SEARCH_HEADER_SIZE = 10

    def __init__(self, name, repeat, aux, tableClass, *, description=""):
        BaseConverter.__init__(
            self, name, repeat, aux, tableClass, description=description
        )
        if issubclass(self.tableClass, SimpleValue):
            self.converter = self.tableClass(name="Value", repeat=None, aux=None)
        else:
            self.converter = Table(
                name="Value", repeat=None, aux=None, tableClass=self.tableClass
            )

    def read(self, reader, font, tableDict):
        format = reader.readUShort()
        if format == 0:
            return self.readFormat0(reader, font)
        elif format == 2:
            return self.readFormat2(reader, font)
        elif format == 4:
            return self.readFormat4(reader, font)
        elif format == 6:
            return self.readFormat6(reader, font)
        elif format == 8:
            return self.readFormat8(reader, font)
        else:
            assert False, "unsupported lookup format: %d" % format

    def write(self, writer, font, tableDict, value, repeatIndex=None):
        values = list(
            sorted([(font.getGlyphID(glyph), val) for glyph, val in value.items()])
        )
        # TODO: Also implement format 4.
        formats = list(
            sorted(
                filter(
                    None,
                    [
                        self.buildFormat0(writer, font, values),
                        self.buildFormat2(writer, font, values),
                        self.buildFormat6(writer, font, values),
                        self.buildFormat8(writer, font, values),
                    ],
                )
            )
        )
        # We use the format ID as secondary sort key to make the output
        # deterministic when multiple formats have same encoded size.
        dataSize, lookupFormat, writeMethod = formats[0]
        pos = writer.getDataLength()
        writeMethod()
        actualSize = writer.getDataLength() - pos
        assert (
            actualSize == dataSize
        ), "AATLookup format %d claimed to write %d bytes, but wrote %d" % (
            lookupFormat,
            dataSize,
            actualSize,
        )

    @staticmethod
    def writeBinSearchHeader(writer, numUnits, unitSize):
        writer.writeUShort(unitSize)
        writer.writeUShort(numUnits)
        searchRange, entrySelector, rangeShift = getSearchRange(
            n=numUnits, itemSize=unitSize
        )
        writer.writeUShort(searchRange)
        writer.writeUShort(entrySelector)
        writer.writeUShort(rangeShift)

    def buildFormat0(self, writer, font, values):
        numGlyphs = len(font.getGlyphOrder())
        if len(values) != numGlyphs:
            return None
        valueSize = self.converter.staticSize
        return (
            2 + numGlyphs * valueSize,
            0,
            lambda: self.writeFormat0(writer, font, values),
        )

    def writeFormat0(self, writer, font, values):
        writer.writeUShort(0)
        for glyphID_, value in values:
            self.converter.write(
                writer, font, tableDict=None, value=value, repeatIndex=None
            )

    def buildFormat2(self, writer, font, values):
        segStart, segValue = values[0]
        segEnd = segStart
        segments = []
        for glyphID, curValue in values[1:]:
            if glyphID != segEnd + 1 or curValue != segValue:
                segments.append((segStart, segEnd, segValue))
                segStart = segEnd = glyphID
                segValue = curValue
            else:
                segEnd = glyphID
        segments.append((segStart, segEnd, segValue))
        valueSize = self.converter.staticSize
        numUnits, unitSize = len(segments) + 1, valueSize + 4
        return (
            2 + self.BIN_SEARCH_HEADER_SIZE + numUnits * unitSize,
            2,
            lambda: self.writeFormat2(writer, font, segments),
        )

    def writeFormat2(self, writer, font, segments):
        writer.writeUShort(2)
        valueSize = self.converter.staticSize
        numUnits, unitSize = len(segments), valueSize + 4
        self.writeBinSearchHeader(writer, numUnits, unitSize)
        for firstGlyph, lastGlyph, value in segments:
            writer.writeUShort(lastGlyph)
            writer.writeUShort(firstGlyph)
            self.converter.write(
                writer, font, tableDict=None, value=value, repeatIndex=None
            )
        writer.writeUShort(0xFFFF)
        writer.writeUShort(0xFFFF)
        writer.writeData(b"\x00" * valueSize)

    def buildFormat6(self, writer, font, values):
        valueSize = self.converter.staticSize
        numUnits, unitSize = len(values), valueSize + 2
        return (
            2 + self.BIN_SEARCH_HEADER_SIZE + (numUnits + 1) * unitSize,
            6,
            lambda: self.writeFormat6(writer, font, values),
        )

    def writeFormat6(self, writer, font, values):
        writer.writeUShort(6)
        valueSize = self.converter.staticSize
        numUnits, unitSize = len(values), valueSize + 2
        self.writeBinSearchHeader(writer, numUnits, unitSize)
        for glyphID, value in values:
            writer.writeUShort(glyphID)
            self.converter.write(
                writer, font, tableDict=None, value=value, repeatIndex=None
            )
        writer.writeUShort(0xFFFF)
        writer.writeData(b"\x00" * valueSize)

    def buildFormat8(self, writer, font, values):
        minGlyphID, maxGlyphID = values[0][0], values[-1][0]
        if len(values) != maxGlyphID - minGlyphID + 1:
            return None
        valueSize = self.converter.staticSize
        return (
            6 + len(values) * valueSize,
            8,
            lambda: self.writeFormat8(writer, font, values),
        )

    def writeFormat8(self, writer, font, values):
        firstGlyphID = values[0][0]
        writer.writeUShort(8)
        writer.writeUShort(firstGlyphID)
        writer.writeUShort(len(values))
        for _, value in values:
            self.converter.write(
                writer, font, tableDict=None, value=value, repeatIndex=None
            )

    def readFormat0(self, reader, font):
        numGlyphs = len(font.getGlyphOrder())
        data = self.converter.readArray(reader, font, tableDict=None, count=numGlyphs)
        return {font.getGlyphName(k): value for k, value in enumerate(data)}

    def readFormat2(self, reader, font):
        mapping = {}
        pos = reader.pos - 2  # start of table is at UShort for format
        unitSize, numUnits = reader.readUShort(), reader.readUShort()
        assert unitSize >= 4 + self.converter.staticSize, unitSize
        for i in range(numUnits):
            reader.seek(pos + i * unitSize + 12)
            last = reader.readUShort()
            first = reader.readUShort()
            value = self.converter.read(reader, font, tableDict=None)
            if last != 0xFFFF:
                for k in range(first, last + 1):
                    mapping[font.getGlyphName(k)] = value
        return mapping

    def readFormat4(self, reader, font):
        mapping = {}
        pos = reader.pos - 2  # start of table is at UShort for format
        unitSize = reader.readUShort()
        assert unitSize >= 6, unitSize
        for i in range(reader.readUShort()):
            reader.seek(pos + i * unitSize + 12)
            last = reader.readUShort()
            first = reader.readUShort()
            offset = reader.readUShort()
            if last != 0xFFFF:
                dataReader = reader.getSubReader(0)  # relative to current position
                dataReader.seek(pos + offset)  # relative to start of table
                data = self.converter.readArray(
                    dataReader, font, tableDict=None, count=last - first + 1
                )
                for k, v in enumerate(data):
                    mapping[font.getGlyphName(first + k)] = v
        return mapping

    def readFormat6(self, reader, font):
        mapping = {}
        pos = reader.pos - 2  # start of table is at UShort for format
        unitSize = reader.readUShort()
        assert unitSize >= 2 + self.converter.staticSize, unitSize
        for i in range(reader.readUShort()):
            reader.seek(pos + i * unitSize + 12)
            glyphID = reader.readUShort()
            value = self.converter.read(reader, font, tableDict=None)
            if glyphID != 0xFFFF:
                mapping[font.getGlyphName(glyphID)] = value
        return mapping

    def readFormat8(self, reader, font):
        first = reader.readUShort()
        count = reader.readUShort()
        data = self.converter.readArray(reader, font, tableDict=None, count=count)
        return {font.getGlyphName(first + k): value for (k, value) in enumerate(data)}

    def xmlRead(self, attrs, content, font):
        value = {}
        for element in content:
            if isinstance(element, tuple):
                name, a, eltContent = element
                if name == "Lookup":
                    value[a["glyph"]] = self.converter.xmlRead(a, eltContent, font)
        return value

    def xmlWrite(self, xmlWriter, font, value, name, attrs):
        xmlWriter.begintag(name, attrs)
        xmlWriter.newline()
        for glyph, value in sorted(value.items()):
            self.converter.xmlWrite(
                xmlWriter, font, value=value, name="Lookup", attrs=[("glyph", glyph)]
            )
        xmlWriter.endtag(name)
        xmlWriter.newline()


# The AAT 'ankr' table has an unusual structure: An offset to an AATLookup
# followed by an offset to a glyph data table. Other than usual, the
# offsets in the AATLookup are not relative to the beginning of
# the beginning of the 'ankr' table, but relative to the glyph data table.
# So, to find the anchor data for a glyph, one needs to add the offset
# to the data table to the offset found in the AATLookup, and then use
# the sum of these two offsets to find the actual data.
class AATLookupWithDataOffset(BaseConverter):
    def read(self, reader, font, tableDict):
        lookupOffset = reader.readULong()
        dataOffset = reader.readULong()
        lookupReader = reader.getSubReader(lookupOffset)
        lookup = AATLookup("DataOffsets", None, None, UShort)
        offsets = lookup.read(lookupReader, font, tableDict)
        result = {}
        for glyph, offset in offsets.items():
            dataReader = reader.getSubReader(offset + dataOffset)
            item = self.tableClass()
            item.decompile(dataReader, font)
            result[glyph] = item
        return result

    def write(self, writer, font, tableDict, value, repeatIndex=None):
        # We do not work with OTTableWriter sub-writers because
        # the offsets in our AATLookup are relative to our data
        # table, for which we need to provide an offset value itself.
        # It might have been possible to somehow make a kludge for
        # performing this indirect offset computation directly inside
        # OTTableWriter. But this would have made the internal logic
        # of OTTableWriter even more complex than it already is,
        # so we decided to roll our own offset computation for the
        # contents of the AATLookup and associated data table.
        offsetByGlyph, offsetByData, dataLen = {}, {}, 0
        compiledData = []
        for glyph in sorted(value, key=font.getGlyphID):
            subWriter = OTTableWriter()
            value[glyph].compile(subWriter, font)
            data = subWriter.getAllData()
            offset = offsetByData.get(data, None)
            if offset == None:
                offset = dataLen
                dataLen = dataLen + len(data)
                offsetByData[data] = offset
                compiledData.append(data)
            offsetByGlyph[glyph] = offset
        # For calculating the offsets to our AATLookup and data table,
        # we can use the regular OTTableWriter infrastructure.
        lookupWriter = writer.getSubWriter()
        lookup = AATLookup("DataOffsets", None, None, UShort)
        lookup.write(lookupWriter, font, tableDict, offsetByGlyph, None)

        dataWriter = writer.getSubWriter()
        writer.writeSubTable(lookupWriter, offsetSize=4)
        writer.writeSubTable(dataWriter, offsetSize=4)
        for d in compiledData:
            dataWriter.writeData(d)

    def xmlRead(self, attrs, content, font):
        lookup = AATLookup("DataOffsets", None, None, self.tableClass)
        return lookup.xmlRead(attrs, content, font)

    def xmlWrite(self, xmlWriter, font, value, name, attrs):
        lookup = AATLookup("DataOffsets", None, None, self.tableClass)
        lookup.xmlWrite(xmlWriter, font, value, name, attrs)


class MortSubtableConverter(BaseConverter):
    def read(self, reader, font, tableDict):
        pos = reader.pos
        m = MortSubtable()
        m.StructLength = reader.readUShort()
        m.CoverageFlags = reader.readUInt8()
        m.MorphType = reader.readUInt8()
        m.SubFeatureFlags = reader.readULong()
        tableClass = lookupTypes["mort"].get(m.MorphType)
        if tableClass is None:
            assert False, "unsupported 'mort' lookup type %s" % m.MorphType

        # Limit the state-table reader to this subtable. In particular, the
        # legacy ligature format has no encoded ligature count and uses the end
        # of the subtable as its only upper bound.
        headerLength = reader.pos - pos
        data = reader.data[reader.pos : reader.pos + m.StructLength - headerLength]
        assert len(data) == m.StructLength - headerLength
        subReader = OTTableReader(data=data, tableTag=reader.tableTag)
        m.SubStruct = tableClass()
        m.SubStruct.decompile(subReader, font)
        reader.seek(pos + m.StructLength)
        return m

    def xmlWrite(self, xmlWriter, font, value, name, attrs):
        xmlWriter.begintag(name, attrs)
        xmlWriter.newline()
        xmlWriter.comment("StructLength=%d" % value.StructLength)
        xmlWriter.newline()
        xmlWriter.simpletag("CoverageFlags", value=value.CoverageFlags)
        xmlWriter.newline()
        xmlWriter.comment("MorphType=%d" % value.MorphType)
        xmlWriter.newline()
        xmlWriter.simpletag("SubFeatureFlags", value="0x%08x" % value.SubFeatureFlags)
        xmlWriter.newline()
        value.SubStruct.toXML(xmlWriter, font)
        xmlWriter.endtag(name)
        xmlWriter.newline()

    def xmlRead(self, attrs, content, font):
        m = MortSubtable()
        m.CoverageFlags = 0
        for eltName, eltAttrs, eltContent in filter(istuple, content):
            if eltName == "CoverageFlags":
                m.CoverageFlags = safeEval(eltAttrs["value"])
            elif eltName == "SubFeatureFlags":
                m.SubFeatureFlags = safeEval(eltAttrs["value"])
            elif eltName.endswith("Morph"):
                m.fromXML(eltName, eltAttrs, eltContent, font)
            else:
                assert False, eltName
        return m

    def write(self, writer, font, tableDict, value, repeatIndex=None):
        lengthIndex = len(writer.items)
        before = writer.getDataLength()
        value.StructLength = 0xDEAD
        value.compile(writer, font)
        assert writer.items[lengthIndex] == b"\xde\xad"
        length = writer.getDataLength() - before
        if length > 0xFFFF:
            raise ValueError("mort subtable exceeds 65535 bytes")
        writer.items[lengthIndex] = struct.pack(">H", length)


class MorxSubtableConverter(BaseConverter):
    _PROCESSING_ORDERS = {
        # bits 30 and 28 of morx.CoverageFlags; see morx spec
        (False, False): "LayoutOrder",
        (True, False): "ReversedLayoutOrder",
        (False, True): "LogicalOrder",
        (True, True): "ReversedLogicalOrder",
    }

    _PROCESSING_ORDERS_REVERSED = {val: key for key, val in _PROCESSING_ORDERS.items()}

    def __init__(self, name, repeat, aux, tableClass=None, *, description=""):
        BaseConverter.__init__(
            self, name, repeat, aux, tableClass, description=description
        )

    def _setTextDirectionFromCoverageFlags(self, flags, subtable):
        if (flags & 0x20) != 0:
            subtable.TextDirection = "Any"
        elif (flags & 0x80) != 0:
            subtable.TextDirection = "Vertical"
        else:
            subtable.TextDirection = "Horizontal"

    def read(self, reader, font, tableDict):
        pos = reader.pos
        m = MorxSubtable()
        m.StructLength = reader.readULong()
        flags = reader.readUInt8()
        orderKey = ((flags & 0x40) != 0, (flags & 0x10) != 0)
        m.ProcessingOrder = self._PROCESSING_ORDERS[orderKey]
        self._setTextDirectionFromCoverageFlags(flags, m)
        m.Reserved = reader.readUShort()
        m.Reserved |= (flags & 0xF) << 16
        m.MorphType = reader.readUInt8()
        m.SubFeatureFlags = reader.readULong()
        tableClass = lookupTypes["morx"].get(m.MorphType)
        if tableClass is None:
            assert False, "unsupported 'morx' lookup type %s" % m.MorphType
        # To decode AAT ligatures, we need to know the subtable size.
        # The easiest way to pass this along is to create a new reader
        # that works on just the subtable as its data.
        headerLength = reader.pos - pos
        data = reader.data[reader.pos : reader.pos + m.StructLength - headerLength]
        assert len(data) == m.StructLength - headerLength
        subReader = OTTableReader(data=data, tableTag=reader.tableTag)
        m.SubStruct = tableClass()
        m.SubStruct.decompile(subReader, font)
        reader.seek(pos + m.StructLength)
        return m

    def xmlWrite(self, xmlWriter, font, value, name, attrs):
        xmlWriter.begintag(name, attrs)
        xmlWriter.newline()
        xmlWriter.comment("StructLength=%d" % value.StructLength)
        xmlWriter.newline()
        xmlWriter.simpletag("TextDirection", value=value.TextDirection)
        xmlWriter.newline()
        xmlWriter.simpletag("ProcessingOrder", value=value.ProcessingOrder)
        xmlWriter.newline()
        if value.Reserved != 0:
            xmlWriter.simpletag("Reserved", value="0x%04x" % value.Reserved)
            xmlWriter.newline()
        xmlWriter.comment("MorphType=%d" % value.MorphType)
        xmlWriter.newline()
        xmlWriter.simpletag("SubFeatureFlags", value="0x%08x" % value.SubFeatureFlags)
        xmlWriter.newline()
        value.SubStruct.toXML(xmlWriter, font)
        xmlWriter.endtag(name)
        xmlWriter.newline()

    def xmlRead(self, attrs, content, font):
        m = MorxSubtable()
        covFlags = 0
        m.Reserved = 0
        for eltName, eltAttrs, eltContent in filter(istuple, content):
            if eltName == "CoverageFlags":
                # Only in XML from old versions of fonttools.
                covFlags = safeEval(eltAttrs["value"])
                orderKey = ((covFlags & 0x40) != 0, (covFlags & 0x10) != 0)
                m.ProcessingOrder = self._PROCESSING_ORDERS[orderKey]
                self._setTextDirectionFromCoverageFlags(covFlags, m)
            elif eltName == "ProcessingOrder":
                m.ProcessingOrder = eltAttrs["value"]
                assert m.ProcessingOrder in self._PROCESSING_ORDERS_REVERSED, (
                    "unknown ProcessingOrder: %s" % m.ProcessingOrder
                )
            elif eltName == "TextDirection":
                m.TextDirection = eltAttrs["value"]
                assert m.TextDirection in {"Horizontal", "Vertical", "Any"}, (
                    "unknown TextDirection %s" % m.TextDirection
                )
            elif eltName == "Reserved":
                m.Reserved = safeEval(eltAttrs["value"])
            elif eltName == "SubFeatureFlags":
                m.SubFeatureFlags = safeEval(eltAttrs["value"])
            elif eltName.endswith("Morph"):
                m.fromXML(eltName, eltAttrs, eltContent, font)
            else:
                assert False, eltName
        m.Reserved = (covFlags & 0xF) << 16 | m.Reserved
        return m

    def write(self, writer, font, tableDict, value, repeatIndex=None):
        covFlags = (value.Reserved & 0x000F0000) >> 16
        reverseOrder, logicalOrder = self._PROCESSING_ORDERS_REVERSED[
            value.ProcessingOrder
        ]
        covFlags |= 0x80 if value.TextDirection == "Vertical" else 0
        covFlags |= 0x40 if reverseOrder else 0
        covFlags |= 0x20 if value.TextDirection == "Any" else 0
        covFlags |= 0x10 if logicalOrder else 0
        value.CoverageFlags = covFlags
        lengthIndex = len(writer.items)
        before = writer.getDataLength()
        value.StructLength = 0xDEADBEEF
        # The high nibble of value.Reserved is actuallly encoded
        # into coverageFlags, so we need to clear it here.
        origReserved = value.Reserved  # including high nibble
        value.Reserved = value.Reserved & 0xFFFF  # without high nibble
        value.compile(writer, font)
        value.Reserved = origReserved  # restore original value
        assert writer.items[lengthIndex] == b"\xde\xad\xbe\xef"
        length = writer.getDataLength() - before
        writer.items[lengthIndex] = struct.pack(">L", length)


# https://developer.apple.com/fonts/TrueType-Reference-Manual/RM06/Chap6Tables.html#ExtendedStateHeader
# TODO: Untangle the implementation of the various lookup-specific formats.
class STXHeader(BaseConverter):
    def __init__(self, name, repeat, aux, tableClass, *, description=""):
        BaseConverter.__init__(
            self, name, repeat, aux, tableClass, description=description
        )
        assert issubclass(self.tableClass, AATAction)
        self.classLookup = AATLookup("GlyphClasses", None, None, UShort)
        if issubclass(self.tableClass, ContextualMorphAction):
            self.perGlyphLookup = AATLookup("PerGlyphLookup", None, None, GlyphID)
        else:
            self.perGlyphLookup = None

    def read(self, reader, font, tableDict):
        table = AATStateTable()
        pos = reader.pos
        classTableReader = reader.getSubReader(0)
        stateArrayReader = reader.getSubReader(0)
        entryTableReader = reader.getSubReader(0)
        actionReader = None
        ligaturesReader = None
        table.GlyphClassCount = reader.readULong()
        classTableReader.seek(pos + reader.readULong())
        stateArrayReader.seek(pos + reader.readULong())
        entryTableReader.seek(pos + reader.readULong())
        if self.perGlyphLookup is not None:
            perGlyphTableReader = reader.getSubReader(0)
            perGlyphTableReader.seek(pos + reader.readULong())
        if issubclass(self.tableClass, LigatureMorphAction):
            actionReader = reader.getSubReader(0)
            actionReader.seek(pos + reader.readULong())
            ligComponentReader = reader.getSubReader(0)
            ligComponentReader.seek(pos + reader.readULong())
            ligaturesReader = reader.getSubReader(0)
            ligaturesReader.seek(pos + reader.readULong())
            numLigComponents = (ligaturesReader.pos - ligComponentReader.pos) // 2
            assert numLigComponents >= 0
            table.LigComponents = ligComponentReader.readUShortArray(numLigComponents)
            table.Ligatures = self._readLigatures(ligaturesReader, font)
        elif issubclass(self.tableClass, InsertionMorphAction):
            actionReader = reader.getSubReader(0)
            actionReader.seek(pos + reader.readULong())
        table.GlyphClasses = self.classLookup.read(classTableReader, font, tableDict)
        numStates = int(
            (entryTableReader.pos - stateArrayReader.pos) / (table.GlyphClassCount * 2)
        )
        for stateIndex in range(numStates):
            state = AATState()
            table.States.append(state)
            for glyphClass in range(table.GlyphClassCount):
                entryIndex = stateArrayReader.readUShort()
                state.Transitions[glyphClass] = self._readTransition(
                    entryTableReader, entryIndex, font, actionReader
                )
        if self.perGlyphLookup is not None:
            table.PerGlyphLookups = self._readPerGlyphLookups(
                table, perGlyphTableReader, font
            )
        return table

    def _readTransition(self, reader, entryIndex, font, actionReader):
        transition = self.tableClass()
        entryReader = reader.getSubReader(
            reader.pos + entryIndex * transition.staticSize
        )
        transition.decompile(entryReader, font, actionReader)
        return transition

    def _readLigatures(self, reader, font):
        limit = len(reader.data)
        numLigatureGlyphs = (limit - reader.pos) // 2
        return font.getGlyphNameMany(reader.readUShortArray(numLigatureGlyphs))

    def _countPerGlyphLookups(self, table):
        # Somewhat annoyingly, the morx table does not encode
        # the size of the per-glyph table. So we need to find
        # the maximum value that MorphActions use as index
        # into this table.
        numLookups = 0
        for state in table.States:
            for t in state.Transitions.values():
                if isinstance(t, ContextualMorphAction):
                    if t.MarkIndex != 0xFFFF:
                        numLookups = max(numLookups, t.MarkIndex + 1)
                    if t.CurrentIndex != 0xFFFF:
                        numLookups = max(numLookups, t.CurrentIndex + 1)
        return numLookups

    def _readPerGlyphLookups(self, table, reader, font):
        pos = reader.pos
        lookups = []
        for _ in range(self._countPerGlyphLookups(table)):
            lookupReader = reader.getSubReader(0)
            lookupReader.seek(pos + reader.readULong())
            lookups.append(self.perGlyphLookup.read(lookupReader, font, {}))
        return lookups

    def write(self, writer, font, tableDict, value, repeatIndex=None):
        glyphClassWriter = OTTableWriter()
        self.classLookup.write(
            glyphClassWriter, font, tableDict, value.GlyphClasses, repeatIndex=None
        )
        glyphClassData = pad(glyphClassWriter.getAllData(), 2)
        glyphClassCount = max(value.GlyphClasses.values()) + 1
        glyphClassTableOffset = 16  # size of STXHeader
        if self.perGlyphLookup is not None:
            glyphClassTableOffset += 4

        glyphClassTableOffset += self.tableClass.actionHeaderSize
        actionData, actionIndex = self.tableClass.compileActions(font, value.States)
        stateArrayData, entryTableData = self._compileStates(
            font, value.States, glyphClassCount, actionIndex
        )
        stateArrayOffset = glyphClassTableOffset + len(glyphClassData)
        entryTableOffset = stateArrayOffset + len(stateArrayData)
        perGlyphOffset = entryTableOffset + len(entryTableData)
        perGlyphData = pad(self._compilePerGlyphLookups(value, font), 4)
        if actionData is not None:
            actionOffset = entryTableOffset + len(entryTableData)
        else:
            actionOffset = None

        ligaturesOffset, ligComponentsOffset = None, None
        ligComponentsData = self._compileLigComponents(value, font)
        ligaturesData = self._compileLigatures(value, font)
        if ligComponentsData is not None:
            assert len(perGlyphData) == 0
            ligComponentsOffset = actionOffset + len(actionData)
            ligaturesOffset = ligComponentsOffset + len(ligComponentsData)

        writer.writeULong(glyphClassCount)
        writer.writeULong(glyphClassTableOffset)
        writer.writeULong(stateArrayOffset)
        writer.writeULong(entryTableOffset)
        if self.perGlyphLookup is not None:
            writer.writeULong(perGlyphOffset)
        if actionOffset is not None:
            writer.writeULong(actionOffset)
        if ligComponentsOffset is not None:
            writer.writeULong(ligComponentsOffset)
            writer.writeULong(ligaturesOffset)
        writer.writeData(glyphClassData)
        writer.writeData(stateArrayData)
        writer.writeData(entryTableData)
        writer.writeData(perGlyphData)
        if actionData is not None:
            writer.writeData(actionData)
        if ligComponentsData is not None:
            writer.writeData(ligComponentsData)
        if ligaturesData is not None:
            writer.writeData(ligaturesData)

    def _compileStates(self, font, states, glyphClassCount, actionIndex):
        stateArrayWriter = OTTableWriter()
        entries, entryIDs = [], {}
        for state in states:
            for glyphClass in range(glyphClassCount):
                transition = state.Transitions[glyphClass]
                entryWriter = OTTableWriter()
                transition.compile(entryWriter, font, actionIndex)
                entryData = entryWriter.getAllData()
                assert (
                    len(entryData) == transition.staticSize
                ), "%s has staticSize %d, " "but actually wrote %d bytes" % (
                    repr(transition),
                    transition.staticSize,
                    len(entryData),
                )
                entryIndex = entryIDs.get(entryData)
                if entryIndex is None:
                    entryIndex = len(entries)
                    entryIDs[entryData] = entryIndex
                    entries.append(entryData)
                stateArrayWriter.writeUShort(entryIndex)
        stateArrayData = pad(stateArrayWriter.getAllData(), 4)
        entryTableData = pad(bytesjoin(entries), 4)
        return stateArrayData, entryTableData

    def _compilePerGlyphLookups(self, table, font):
        if self.perGlyphLookup is None:
            return b""
        numLookups = self._countPerGlyphLookups(table)
        assert len(table.PerGlyphLookups) == numLookups, (
            "len(AATStateTable.PerGlyphLookups) is %d, "
            "but the actions inside the table refer to %d"
            % (len(table.PerGlyphLookups), numLookups)
        )
        writer = OTTableWriter()
        for lookup in table.PerGlyphLookups:
            lookupWriter = writer.getSubWriter()
            self.perGlyphLookup.write(lookupWriter, font, {}, lookup, None)
            writer.writeSubTable(lookupWriter, offsetSize=4)
        return writer.getAllData()

    def _compileLigComponents(self, table, font):
        if not hasattr(table, "LigComponents"):
            return None
        writer = OTTableWriter()
        for component in table.LigComponents:
            writer.writeUShort(component)
        return writer.getAllData()

    def _compileLigatures(self, table, font):
        if not hasattr(table, "Ligatures"):
            return None
        writer = OTTableWriter()
        for glyphName in table.Ligatures:
            writer.writeUShort(font.getGlyphID(glyphName))
        return writer.getAllData()

    def xmlWrite(self, xmlWriter, font, value, name, attrs):
        xmlWriter.begintag(name, attrs)
        xmlWriter.newline()
        xmlWriter.comment("GlyphClassCount=%s" % value.GlyphClassCount)
        xmlWriter.newline()
        for g, klass in sorted(value.GlyphClasses.items()):
            xmlWriter.simpletag("GlyphClass", glyph=g, value=klass)
            xmlWriter.newline()
        for stateIndex, state in enumerate(value.States):
            xmlWriter.begintag("State", index=stateIndex)
            xmlWriter.newline()
            for glyphClass, trans in sorted(state.Transitions.items()):
                trans.toXML(
                    xmlWriter,
                    font=font,
                    attrs={"onGlyphClass": glyphClass},
                    name="Transition",
                )
            xmlWriter.endtag("State")
            xmlWriter.newline()
        for i, lookup in enumerate(value.PerGlyphLookups):
            xmlWriter.begintag("PerGlyphLookup", index=i)
            xmlWriter.newline()
            for glyph, val in sorted(lookup.items()):
                xmlWriter.simpletag("Lookup", glyph=glyph, value=val)
                xmlWriter.newline()
            xmlWriter.endtag("PerGlyphLookup")
            xmlWriter.newline()
        if hasattr(value, "LigComponents"):
            xmlWriter.begintag("LigComponents")
            xmlWriter.newline()
            for i, val in enumerate(getattr(value, "LigComponents")):
                xmlWriter.simpletag("LigComponent", index=i, value=val)
                xmlWriter.newline()
            xmlWriter.endtag("LigComponents")
            xmlWriter.newline()
        if hasattr(value, "_MortLigatureRebase"):
            xmlWriter.begintag("MortLigatureRebase")
            xmlWriter.newline()
            for componentIndex in sorted(value._MortLigatureRebase):
                xmlWriter.simpletag("Component", index=componentIndex)
                xmlWriter.newline()
            xmlWriter.endtag("MortLigatureRebase")
            xmlWriter.newline()
        self._xmlWriteLigatures(xmlWriter, font, value, name, attrs)
        xmlWriter.endtag(name)
        xmlWriter.newline()

    def _xmlWriteLigatures(self, xmlWriter, font, value, name, attrs):
        if not hasattr(value, "Ligatures"):
            return
        xmlWriter.begintag("Ligatures")
        xmlWriter.newline()
        for i, g in enumerate(getattr(value, "Ligatures")):
            xmlWriter.simpletag("Ligature", index=i, glyph=g)
            xmlWriter.newline()
        xmlWriter.endtag("Ligatures")
        xmlWriter.newline()

    def xmlRead(self, attrs, content, font):
        table = AATStateTable()
        for eltName, eltAttrs, eltContent in filter(istuple, content):
            if eltName == "GlyphClass":
                glyph = eltAttrs["glyph"]
                value = eltAttrs["value"]
                table.GlyphClasses[glyph] = safeEval(value)
            elif eltName == "State":
                state = self._xmlReadState(eltAttrs, eltContent, font)
                table.States.append(state)
            elif eltName == "PerGlyphLookup":
                lookup = self.perGlyphLookup.xmlRead(eltAttrs, eltContent, font)
                table.PerGlyphLookups.append(lookup)
            elif eltName == "LigComponents":
                table.LigComponents = self._xmlReadLigComponents(
                    eltAttrs, eltContent, font
                )
            elif eltName == "MortLigatureRebase":
                table._MortLigatureRebase = {
                    safeEval(componentAttrs["index"])
                    for componentName, componentAttrs, _componentContent in filter(
                        istuple, eltContent
                    )
                    if componentName == "Component"
                }
            elif eltName == "Ligatures":
                table.Ligatures = self._xmlReadLigatures(eltAttrs, eltContent, font)
        glyphClasses = list(table.GlyphClasses.values())
        glyphClasses.extend(
            glyphClass for state in table.States for glyphClass in state.Transitions
        )
        table.GlyphClassCount = max(glyphClasses, default=-1) + 1
        return table

    def _xmlReadState(self, attrs, content, font):
        state = AATState()
        for eltName, eltAttrs, eltContent in filter(istuple, content):
            if eltName == "Transition":
                glyphClass = safeEval(eltAttrs["onGlyphClass"])
                transition = self.tableClass()
                transition.fromXML(eltName, eltAttrs, eltContent, font)
                state.Transitions[glyphClass] = transition
        return state

    def _xmlReadLigComponents(self, attrs, content, font):
        ligComponents = []
        for eltName, eltAttrs, _eltContent in filter(istuple, content):
            if eltName == "LigComponent":
                ligComponents.append(safeEval(eltAttrs["value"]))
        return ligComponents

    def _xmlReadLigatures(self, attrs, content, font):
        ligs = []
        for eltName, eltAttrs, _eltContent in filter(istuple, content):
            if eltName == "Ligature":
                ligs.append(eltAttrs["glyph"])
        return ligs


class StateHeader(STXHeader):
    """Converter for the obsolete, non-extended AAT state-table format."""

    def read(self, reader, font, tableDict):
        table = AATStateTable()
        pos = reader.pos
        table.GlyphClassCount = reader.readUShort()
        classTableOffset = reader.readUShort()
        stateArrayOffset = reader.readUShort()
        entryTableOffset = reader.readUShort()
        if table.GlyphClassCount == 0:
            raise ValueError("mort state table has no glyph classes")

        substitutionTableOffset = None
        actionOffset = componentOffset = ligatureOffset = None
        actionReader = None
        if issubclass(self.tableClass, ContextualMorphAction):
            substitutionTableOffset = reader.readUShort()
        elif issubclass(self.tableClass, LigatureMorphAction):
            actionOffset = reader.readUShort()
            componentOffset = reader.readUShort()
            ligatureOffset = reader.readUShort()
            if (
                componentOffset > ligatureOffset
                or (ligatureOffset - componentOffset) % 2
            ):
                raise ValueError("invalid mort ligature component table")
            componentReader = reader.getSubReader(0)
            componentReader.seek(pos + componentOffset)
            rawComponents = componentReader.readUShortArray(
                (ligatureOffset - componentOffset) // 2
            )
            if any(component % 2 for component in rawComponents):
                raise ValueError("invalid mort ligature component value")
            table.LigComponents = [component // 2 for component in rawComponents]
            table.Ligatures = []
            table._MortLigatureRebase = set()
        elif issubclass(self.tableClass, InsertionMorphAction):
            actionOffset = reader.readUShort()
            actionReader = reader.getSubReader(0)
            actionReader.seek(pos + actionOffset)

        classTableReader = reader.getSubReader(0)
        classTableReader.seek(pos + classTableOffset)
        firstGlyph = classTableReader.readUShort()
        glyphCount = classTableReader.readUShort()
        glyphClasses = classTableReader.readUInt8Array(glyphCount)
        table.GlyphClasses = {
            font.getGlyphName(firstGlyph + i): glyphClass
            for i, glyphClass in enumerate(glyphClasses)
        }

        stateArrayReader = reader.getSubReader(0)
        stateArrayReader.seek(pos + stateArrayOffset)
        entryTableReader = reader.getSubReader(0)
        entryTableReader.seek(pos + entryTableOffset)
        stateDataLength = entryTableOffset - stateArrayOffset
        if stateDataLength < 0:
            raise ValueError("mort state-array offset follows its entry table")
        numStates = stateDataLength // table.GlyphClassCount
        transitionCache = {}
        for _stateIndex in range(numStates):
            state = AATState()
            table.States.append(state)
            for glyphClass in range(table.GlyphClassCount):
                entryIndex = stateArrayReader.readUInt8()
                transition = transitionCache.get(entryIndex)
                if transition is None:
                    transition = self._readTransition(
                        entryTableReader,
                        entryIndex,
                        font,
                        actionReader,
                        pos,
                        stateArrayOffset,
                        componentOffset,
                        ligatureOffset,
                        table,
                    )
                    transitionCache[entryIndex] = transition
                state.Transitions[glyphClass] = transition

        if substitutionTableOffset is not None:
            self._readContextualLookups(
                table, reader, font, pos, substitutionTableOffset
            )
        elif ligatureOffset is not None:
            ligatureBase = ligatureOffset // 2
            table._MortLigatureRebase = {
                componentIndex
                for componentIndex in table._MortLigatureRebase
                if table.LigComponents[componentIndex] >= ligatureBase
            }
            for componentIndex in table._MortLigatureRebase:
                # Unused entries are commonly zero. Only values which actually
                # include the ligature-table base need to be normalized.
                table.LigComponents[componentIndex] -= ligatureBase

            ligatureCount = self._countLegacyLigatures(table, len(font.getGlyphOrder()))
            ligatureReader = reader.getSubReader(0)
            ligatureReader.seek(pos + ligatureOffset)
            table.Ligatures = font.getGlyphNameMany(
                ligatureReader.readUShortArray(ligatureCount)
            )
        return table

    def _readTransition(
        self,
        reader,
        entryIndex,
        font,
        actionReader,
        stateTablePos,
        stateArrayOffset,
        componentOffset,
        ligatureOffset,
        table,
    ):
        transition = self.tableClass()
        entrySize = (
            4
            if issubclass(self.tableClass, LigatureMorphAction)
            else transition.staticSize
        )
        entryReader = reader.getSubReader(reader.pos + entryIndex * entrySize)

        if issubclass(self.tableClass, ContextualMorphAction):
            transition.NewState = entryReader.readUShort()
            flags = entryReader.readUShort()
            transition.SetMark = bool(flags & 0x8000)
            transition.DontAdvance = bool(flags & 0x4000)
            transition.ReservedFlags = flags & 0x3FFF
            transition._MortMarkOffset = entryReader.readShort()
            transition._MortCurrentOffset = entryReader.readShort()
        elif issubclass(self.tableClass, LigatureMorphAction):
            transition.NewState = entryReader.readUShort()
            flags = entryReader.readUShort()
            transition.SetComponent = bool(flags & 0x8000)
            transition.DontAdvance = bool(flags & 0x4000)
            transition.ReservedFlags = 0
            transition.Actions = []
            ligActionOffset = flags & 0x3FFF
            if ligActionOffset:
                transition.Actions = self._readLigatureActions(
                    reader,
                    font,
                    stateTablePos,
                    ligActionOffset,
                    componentOffset,
                    ligatureOffset,
                    table,
                )
        else:
            transition.decompile(entryReader, font, actionReader)

        newStateOffset = transition.NewState - stateArrayOffset
        if newStateOffset < 0 or newStateOffset % table.GlyphClassCount:
            raise ValueError("invalid mort new-state offset")
        transition.NewState = newStateOffset // table.GlyphClassCount
        return transition

    def _readContextualLookups(
        self, table, reader, font, stateTablePos, substitutionTableOffset
    ):
        transitions = {
            transition
            for state in table.States
            for transition in state.Transitions.values()
        }
        offsets = sorted(
            {
                offset
                for transition in transitions
                for offset in (
                    transition._MortMarkOffset,
                    transition._MortCurrentOffset,
                )
                if offset
            }
        )
        offsetToIndex = {offset: i for i, offset in enumerate(offsets)}
        glyphOrder = font.getGlyphOrder()
        # reader.data holds just this subtable, so this is the room between
        # the state-table header and the end of the subtable.
        available = len(reader.data) - stateTablePos
        for offset in offsets:
            # The entry offset plus a glyph ID gives the word offset of that
            # glyph's replacement, so an offset (which may be negative) exposes
            # a per-glyph window rather than a whole lookup table. Fonts trim
            # the windows: words before the substitution table mean "no
            # substitution" (like HarfBuzz), and we also stop at the end of
            # the subtable (stricter than HarfBuzz's blob-wide bound).
            firstGlyph = max(0, (substitutionTableOffset + 1) // 2 - offset)
            lastGlyph = min(len(glyphOrder) - 1, (available - 2) // 2 - offset)
            lookup = {}
            if firstGlyph <= lastGlyph:
                lookupReader = reader.getSubReader(0)
                lookupReader.seek(stateTablePos + 2 * (offset + firstGlyph))
                replacements = lookupReader.readUShortArray(lastGlyph - firstGlyph + 1)
                lookup = {
                    glyphOrder[firstGlyph + i]: font.getGlyphName(replacement)
                    for i, replacement in enumerate(replacements)
                    if replacement != 0
                }
            table.PerGlyphLookups.append(lookup)
        for transition in transitions:
            transition.MarkIndex = (
                offsetToIndex[transition._MortMarkOffset]
                if transition._MortMarkOffset
                else 0xFFFF
            )
            transition.CurrentIndex = (
                offsetToIndex[transition._MortCurrentOffset]
                if transition._MortCurrentOffset
                else 0xFFFF
            )
            del transition._MortMarkOffset
            del transition._MortCurrentOffset

    def _readLigatureActions(
        self,
        reader,
        font,
        stateTablePos,
        ligActionOffset,
        componentOffset,
        ligatureOffset,
        table,
    ):
        actionReader = reader.getSubReader(0)
        actionReader.seek(stateTablePos + ligActionOffset)
        glyphCount = len(font.getGlyphOrder())
        result = []
        rebased = False
        last = False
        while not last:
            value = actionReader.readULong()
            last = bool(value & 0x80000000)
            store = bool(value & 0x40000000)
            componentWordOffset = value & 0x3FFFFFFF
            if componentWordOffset & 0x20000000:
                componentWordOffset -= 0x40000000
            componentIndex = componentWordOffset - componentOffset // 2
            if (
                componentIndex >= len(table.LigComponents)
                or componentIndex + glyphCount <= 0
            ):
                raise ValueError("mort ligature component offset is out of bounds")
            if (store or last) and not rebased:
                start = max(0, componentIndex)
                limit = min(len(table.LigComponents), componentIndex + glyphCount)
                table._MortLigatureRebase.update(range(start, limit))
                rebased = True

            action = LigAction()
            action.Store = store
            action.GlyphIndexDelta = componentIndex
            result.append(action)
        return result

    @staticmethod
    def _countLegacyLigatures(table, glyphCount):
        count = 0
        transitions = {
            transition
            for state in table.States
            for transition in state.Transitions.values()
        }
        for transition in transitions:
            cumulativeMax = 0
            for index, action in enumerate(transition.Actions):
                start = max(0, action.GlyphIndexDelta)
                limit = min(
                    len(table.LigComponents),
                    action.GlyphIndexDelta + glyphCount,
                )
                cumulativeMax += max(table.LigComponents[start:limit], default=0)
                if action.Store or index == len(transition.Actions) - 1:
                    count = max(count, cumulativeMax + 1)
        return count

    def write(self, writer, font, tableDict, value, repeatIndex=None):
        glyphClassCount = value.GlyphClassCount
        if not 0 < glyphClassCount <= 256:
            raise ValueError("mort state tables support 1 to 256 glyph classes")

        glyphClasses = sorted(
            (font.getGlyphID(glyph), glyphClass)
            for glyph, glyphClass in value.GlyphClasses.items()
        )
        if glyphClasses:
            firstGlyph = glyphClasses[0][0]
            lastGlyph = glyphClasses[-1][0]
            classValues = dict(glyphClasses)
            classes = bytes(
                classValues.get(glyphID, 1)
                for glyphID in range(firstGlyph, lastGlyph + 1)
            )
        else:
            firstGlyph = 0
            classes = b""
        classData = struct.pack(">HH", firstGlyph, len(classes)) + classes

        genericActionData, genericActionIndex = self.tableClass.compileActions(
            font, value.States
        )
        entries = []
        entryIDs = {}
        stateEntryIDs = []
        for state in value.States:
            row = []
            for glyphClass in range(glyphClassCount):
                transition = state.Transitions[glyphClass]
                transitionWriter = OTTableWriter()
                transition.compile(transitionWriter, font, genericActionIndex)
                key = transitionWriter.getAllData()
                entryIndex = entryIDs.get(key)
                if entryIndex is None:
                    entryIndex = len(entries)
                    if entryIndex > 0xFF:
                        raise ValueError(
                            "mort state tables support at most 256 entries"
                        )
                    entryIDs[key] = entryIndex
                    entries.append(transition)
                row.append(entryIndex)
            stateEntryIDs.append(row)

        extraSize = 0
        if issubclass(self.tableClass, ContextualMorphAction):
            extraSize = 2
        elif issubclass(self.tableClass, LigatureMorphAction):
            extraSize = 6
        elif issubclass(self.tableClass, InsertionMorphAction):
            extraSize = 2
        classTableOffset = 8 + extraSize
        stateArrayOffset = self._aligned(classTableOffset + len(classData), 2)
        stateData = bytes(entry for row in stateEntryIDs for entry in row)
        entryTableOffset = self._aligned(stateArrayOffset + len(stateData), 2)
        entrySize = (
            4
            if issubclass(self.tableClass, LigatureMorphAction)
            else self.tableClass.staticSize
        )
        tailOffset = entryTableOffset + len(entries) * entrySize

        extra = b""
        tail = b""
        contextOffsets = None
        ligatureActionOffsets = None
        if issubclass(self.tableClass, ContextualMorphAction):
            extra, tail, contextOffsets = self._compileContextualLookups(
                value, font, tailOffset
            )
        elif issubclass(self.tableClass, LigatureMorphAction):
            extra, tail, ligatureActionOffsets = self._compileLigatureData(
                value, font, entries, tailOffset
            )
        elif issubclass(self.tableClass, InsertionMorphAction):
            if tailOffset > 0xFFFF:
                raise ValueError("mort insertion-action offset exceeds 16 bits")
            extra = struct.pack(">H", tailOffset)
            tail = genericActionData

        entryData = b"".join(
            self._compileTransition(
                transition,
                font,
                genericActionIndex,
                stateArrayOffset,
                glyphClassCount,
                contextOffsets,
                ligatureActionOffsets,
            )
            for transition in entries
        )
        data = struct.pack(
            ">HHHH",
            glyphClassCount,
            classTableOffset,
            stateArrayOffset,
            entryTableOffset,
        )
        data += extra + classData
        data = pad(data, 2) + stateData
        data = pad(data, 2) + entryData + tail
        data = pad(data, 4)
        if len(data) > 0xFFFF:
            raise ValueError("mort state table exceeds 65535 bytes")
        writer.writeData(data)

    @staticmethod
    def _aligned(value, alignment):
        return value + (-value % alignment)

    def _compileContextualLookups(self, table, font, tailOffset):
        count = self._countPerGlyphLookups(table)
        if len(table.PerGlyphLookups) != count:
            raise ValueError(
                "mort contextual actions refer to %d lookups, but the table has %d"
                % (count, len(table.PerGlyphLookups))
            )
        glyphOrder = font.getGlyphOrder()
        offsets = []
        data = b""
        for lookup in table.PerGlyphLookups:
            offset = (tailOffset + len(data)) // 2
            if offset > 0x7FFF:
                raise ValueError("mort substitution offset exceeds signed 16 bits")
            offsets.append(offset)
            replacements = []
            for glyph in glyphOrder:
                replacement = lookup.get(glyph)
                replacements.append(
                    0 if replacement is None else font.getGlyphID(replacement)
                )
            data += struct.pack(">%dH" % len(replacements), *replacements)
        if tailOffset > 0xFFFF:
            raise ValueError("mort substitution-table offset exceeds 16 bits")
        return struct.pack(">H", tailOffset), data, offsets

    def _compileLigatureData(self, table, font, entries, tailOffset):
        sequences = []
        sequenceSet = set()
        for transition in entries:
            sequence = tuple(
                (action.GlyphIndexDelta, action.Store) for action in transition.Actions
            )
            if sequence and sequence not in sequenceSet:
                sequenceSet.add(sequence)
                sequences.append(sequence)

        actionOffset = self._aligned(tailOffset, 4)
        actionCount = sum(len(sequence) for sequence in sequences)
        componentOffset = actionOffset + actionCount * 4
        glyphCount = len(font.getGlyphOrder())
        ligatureOffset = componentOffset + len(table.LigComponents) * 2
        if max(actionOffset, componentOffset, ligatureOffset) > 0xFFFF:
            raise ValueError("mort ligature-table offset exceeds 16 bits")

        actionOffsets = {}
        actionData = b""
        rebaseComponents = getattr(table, "_MortLigatureRebase", None)
        if rebaseComponents is None:
            rebaseComponents = set()
        else:
            rebaseComponents = set(rebaseComponents)
        for sequence in sequences:
            sequenceOffset = actionOffset + len(actionData)
            if sequenceOffset > 0x3FFF:
                raise ValueError("mort ligature-action offset exceeds 14 bits")
            actionOffsets[sequence] = sequenceOffset
            rebased = False
            for index, (delta, store) in enumerate(sequence):
                value = componentOffset // 2 + delta
                if not -(1 << 29) <= value < (1 << 29):
                    raise ValueError("mort ligature component offset exceeds 30 bits")
                value &= 0x3FFFFFFF
                last = index == len(sequence) - 1
                value |= 0x40000000 if store else 0
                value |= 0x80000000 if last else 0
                actionData += struct.pack(">I", value)
                addBase = (store or last) and not rebased
                if addBase and not hasattr(table, "_MortLigatureRebase"):
                    start = max(0, delta)
                    limit = min(len(table.LigComponents), delta + glyphCount)
                    rebaseComponents.update(range(start, limit))
                rebased |= store or last

        componentData = b""
        for componentIndex, component in enumerate(table.LigComponents):
            component *= 2
            if componentIndex in rebaseComponents:
                component += ligatureOffset
            if not 0 <= component <= 0xFFFF:
                raise ValueError("mort ligature component value exceeds 16 bits")
            componentData += struct.pack(">H", component)

        ligatureData = b"".join(
            struct.pack(">H", font.getGlyphID(glyph)) for glyph in table.Ligatures
        )
        extra = struct.pack(">HHH", actionOffset, componentOffset, ligatureOffset)
        tail = (
            bytes(actionOffset - tailOffset) + actionData + componentData + ligatureData
        )
        return extra, tail, actionOffsets

    def _compileTransition(
        self,
        transition,
        font,
        actionIndex,
        stateArrayOffset,
        glyphClassCount,
        contextOffsets,
        ligatureActionOffsets,
    ):
        newState = stateArrayOffset + transition.NewState * glyphClassCount
        if not 0 <= newState <= 0xFFFF:
            raise ValueError("mort new-state offset exceeds 16 bits")

        if issubclass(self.tableClass, ContextualMorphAction):
            flags = transition.ReservedFlags
            flags |= 0x8000 if transition.SetMark else 0
            flags |= 0x4000 if transition.DontAdvance else 0
            markOffset = (
                0
                if transition.MarkIndex == 0xFFFF
                else contextOffsets[transition.MarkIndex]
            )
            currentOffset = (
                0
                if transition.CurrentIndex == 0xFFFF
                else contextOffsets[transition.CurrentIndex]
            )
            return struct.pack(">HHhh", newState, flags, markOffset, currentOffset)

        if issubclass(self.tableClass, LigatureMorphAction):
            if transition.ReservedFlags:
                raise ValueError("mort ligature entries cannot encode reserved flags")
            flags = 0x8000 if transition.SetComponent else 0
            flags |= 0x4000 if transition.DontAdvance else 0
            sequence = tuple(
                (action.GlyphIndexDelta, action.Store) for action in transition.Actions
            )
            flags |= ligatureActionOffsets.get(sequence, 0)
            return struct.pack(">HH", newState, flags)

        transitionWriter = OTTableWriter()
        transition.compile(transitionWriter, font, actionIndex)
        data = transitionWriter.getAllData()
        return struct.pack(">H", newState) + data[2:]


class MorphStateTable(BaseConverter):
    """Dispatch to classic or extended AAT state-table encoding by table tag."""

    def __init__(self, name, repeat, aux, tableClass, *, description=""):
        BaseConverter.__init__(
            self, name, repeat, aux, tableClass, description=description
        )
        self.legacy = StateHeader(
            name, repeat, aux, tableClass, description=description
        )
        self.extended = STXHeader(
            name, repeat, aux, tableClass, description=description
        )

    def _converter(self, tableTag):
        return self.legacy if tableTag == "mort" else self.extended

    def read(self, reader, font, tableDict):
        return self._converter(reader.tableTag).read(reader, font, tableDict)

    def write(self, writer, font, tableDict, value, repeatIndex=None):
        return self._converter(writer.tableTag).write(
            writer, font, tableDict, value, repeatIndex
        )

    def xmlRead(self, attrs, content, font):
        return self.extended.xmlRead(attrs, content, font)

    def xmlWrite(self, xmlWriter, font, value, name, attrs):
        return self.extended.xmlWrite(xmlWriter, font, value, name, attrs)


class CIDGlyphMap(BaseConverter):
    def read(self, reader, font, tableDict):
        numCIDs = reader.readUShort()
        result = {}
        for cid, glyphID in enumerate(reader.readUShortArray(numCIDs)):
            if glyphID != 0xFFFF:
                result[cid] = font.getGlyphName(glyphID)
        return result

    def write(self, writer, font, tableDict, value, repeatIndex=None):
        items = {cid: font.getGlyphID(glyph) for cid, glyph in value.items()}
        count = max(items) + 1 if items else 0
        writer.writeUShort(count)
        for cid in range(count):
            writer.writeUShort(items.get(cid, 0xFFFF))

    def xmlRead(self, attrs, content, font):
        result = {}
        for eName, eAttrs, _eContent in filter(istuple, content):
            if eName == "CID":
                result[safeEval(eAttrs["cid"])] = eAttrs["glyph"].strip()
        return result

    def xmlWrite(self, xmlWriter, font, value, name, attrs):
        xmlWriter.begintag(name, attrs)
        xmlWriter.newline()
        for cid, glyph in sorted(value.items()):
            if glyph is not None and glyph != 0xFFFF:
                xmlWriter.simpletag("CID", cid=cid, glyph=glyph)
                xmlWriter.newline()
        xmlWriter.endtag(name)
        xmlWriter.newline()


class GlyphCIDMap(BaseConverter):
    def read(self, reader, font, tableDict):
        glyphOrder = font.getGlyphOrder()
        count = reader.readUShort()
        cids = reader.readUShortArray(count)
        if count > len(glyphOrder):
            log.warning(
                "GlyphCIDMap has %d elements, "
                "but the font has only %d glyphs; "
                "ignoring the rest" % (count, len(glyphOrder))
            )
        result = {}
        for glyphID in range(min(len(cids), len(glyphOrder))):
            cid = cids[glyphID]
            if cid != 0xFFFF:
                result[glyphOrder[glyphID]] = cid
        return result

    def write(self, writer, font, tableDict, value, repeatIndex=None):
        items = {
            font.getGlyphID(g): cid
            for g, cid in value.items()
            if cid is not None and cid != 0xFFFF
        }
        count = max(items) + 1 if items else 0
        writer.writeUShort(count)
        for glyphID in range(count):
            writer.writeUShort(items.get(glyphID, 0xFFFF))

    def xmlRead(self, attrs, content, font):
        result = {}
        for eName, eAttrs, _eContent in filter(istuple, content):
            if eName == "CID":
                result[eAttrs["glyph"]] = safeEval(eAttrs["value"])
        return result

    def xmlWrite(self, xmlWriter, font, value, name, attrs):
        xmlWriter.begintag(name, attrs)
        xmlWriter.newline()
        for glyph, cid in sorted(value.items()):
            if cid is not None and cid != 0xFFFF:
                xmlWriter.simpletag("CID", glyph=glyph, value=cid)
                xmlWriter.newline()
        xmlWriter.endtag(name)
        xmlWriter.newline()


class DeltaValue(BaseConverter):
    def read(self, reader, font, tableDict):
        StartSize = tableDict["StartSize"]
        EndSize = tableDict["EndSize"]
        DeltaFormat = tableDict["DeltaFormat"]
        assert DeltaFormat in (1, 2, 3), "illegal DeltaFormat"
        nItems = EndSize - StartSize + 1
        nBits = 1 << DeltaFormat
        minusOffset = 1 << nBits
        mask = (1 << nBits) - 1
        signMask = 1 << (nBits - 1)

        DeltaValue = []
        tmp, shift = 0, 0
        for i in range(nItems):
            if shift == 0:
                tmp, shift = reader.readUShort(), 16
            shift = shift - nBits
            value = (tmp >> shift) & mask
            if value & signMask:
                value = value - minusOffset
            DeltaValue.append(value)
        return DeltaValue

    def write(self, writer, font, tableDict, value, repeatIndex=None):
        StartSize = tableDict["StartSize"]
        EndSize = tableDict["EndSize"]
        DeltaFormat = tableDict["DeltaFormat"]
        DeltaValue = value
        assert DeltaFormat in (1, 2, 3), "illegal DeltaFormat"
        nItems = EndSize - StartSize + 1
        nBits = 1 << DeltaFormat
        assert len(DeltaValue) == nItems
        mask = (1 << nBits) - 1

        tmp, shift = 0, 16
        for value in DeltaValue:
            shift = shift - nBits
            tmp = tmp | ((value & mask) << shift)
            if shift == 0:
                writer.writeUShort(tmp)
                tmp, shift = 0, 16
        if shift != 16:
            writer.writeUShort(tmp)

    def xmlWrite(self, xmlWriter, font, value, name, attrs):
        xmlWriter.simpletag(name, attrs + [("value", value)])
        xmlWriter.newline()

    def xmlRead(self, attrs, content, font):
        return safeEval(attrs["value"])


class VarIdxMapValue(BaseConverter):
    def read(self, reader, font, tableDict):
        fmt = tableDict["EntryFormat"]
        nItems = tableDict["MappingCount"]

        innerBits = 1 + (fmt & 0x000F)
        innerMask = (1 << innerBits) - 1
        outerMask = 0xFFFFFFFF - innerMask
        outerShift = 16 - innerBits

        entrySize = 1 + ((fmt & 0x0030) >> 4)
        readArray = {
            1: reader.readUInt8Array,
            2: reader.readUShortArray,
            3: reader.readUInt24Array,
            4: reader.readULongArray,
        }[entrySize]

        return [
            (((raw & outerMask) << outerShift) | (raw & innerMask))
            for raw in readArray(nItems)
        ]

    def write(self, writer, font, tableDict, value, repeatIndex=None):
        fmt = tableDict["EntryFormat"]
        mapping = value
        writer["MappingCount"].setValue(len(mapping))

        innerBits = 1 + (fmt & 0x000F)
        innerMask = (1 << innerBits) - 1
        outerShift = 16 - innerBits

        entrySize = 1 + ((fmt & 0x0030) >> 4)
        writeArray = {
            1: writer.writeUInt8Array,
            2: writer.writeUShortArray,
            3: writer.writeUInt24Array,
            4: writer.writeULongArray,
        }[entrySize]

        writeArray(
            [
                (((idx & 0xFFFF0000) >> outerShift) | (idx & innerMask))
                for idx in mapping
            ]
        )


class VarDataValue(BaseConverter):
    def read(self, reader, font, tableDict):
        values = []

        regionCount = tableDict["VarRegionCount"]
        wordCount = tableDict["NumShorts"]

        # https://github.com/fonttools/fonttools/issues/2279
        longWords = bool(wordCount & 0x8000)
        wordCount = wordCount & 0x7FFF

        if longWords:
            readBigArray, readSmallArray = reader.readLongArray, reader.readShortArray
        else:
            readBigArray, readSmallArray = reader.readShortArray, reader.readInt8Array

        n1, n2 = min(regionCount, wordCount), max(regionCount, wordCount)
        values.extend(readBigArray(n1))
        values.extend(readSmallArray(n2 - n1))
        if n2 > regionCount:  # Padding
            del values[regionCount:]

        return values

    def write(self, writer, font, tableDict, values, repeatIndex=None):
        regionCount = tableDict["VarRegionCount"]
        wordCount = tableDict["NumShorts"]

        # https://github.com/fonttools/fonttools/issues/2279
        longWords = bool(wordCount & 0x8000)
        wordCount = wordCount & 0x7FFF

        writeBigArray, writeSmallArray = {
            False: (writer.writeShortArray, writer.writeInt8Array),
            True: (writer.writeLongArray, writer.writeShortArray),
        }[longWords]

        n1, n2 = min(regionCount, wordCount), max(regionCount, wordCount)
        writeBigArray(values[:n1])
        writeSmallArray(values[n1:regionCount])
        if n2 > regionCount:  # Padding
            writer.writeSmallArray([0] * (n2 - regionCount))

    def xmlWrite(self, xmlWriter, font, value, name, attrs):
        xmlWriter.simpletag(name, attrs + [("value", value)])
        xmlWriter.newline()

    def xmlRead(self, attrs, content, font):
        return safeEval(attrs["value"])


class TupleValues:
    def read(self, data, font):
        return TupleVariation.decompileDeltas_(None, data)[0]

    def write(self, writer, font, tableDict, values, repeatIndex=None):
        optimizeSpeed = font.cfg[OPTIMIZE_FONT_SPEED]
        return bytes(
            TupleVariation.compileDeltaValues_(values, optimizeSize=not optimizeSpeed)
        )

    def xmlRead(self, attrs, content, font):
        return safeEval(attrs["value"])

    def xmlWrite(self, xmlWriter, font, value, name, attrs):
        xmlWriter.simpletag(name, attrs + [("value", value)])
        xmlWriter.newline()


class CFF2Index(BaseConverter):
    def __init__(
        self,
        name,
        repeat,
        aux,
        tableClass=None,
        *,
        itemClass=None,
        itemConverterClass=None,
        description="",
    ):
        BaseConverter.__init__(
            self, name, repeat, aux, tableClass, description=description
        )
        self._itemClass = itemClass
        self._converter = (
            itemConverterClass() if itemConverterClass is not None else None
        )

    def read(self, reader, font, tableDict):
        count = reader.readULong()
        if count == 0:
            return []
        offSize = reader.readUInt8()

        def getReadArray(reader, offSize):
            return {
                1: reader.readUInt8Array,
                2: reader.readUShortArray,
                3: reader.readUInt24Array,
                4: reader.readULongArray,
            }[offSize]

        readArray = getReadArray(reader, offSize)

        lazy = font.lazy is not False and count > 8
        if not lazy:
            offsets = readArray(count + 1)
            items = []
            lastOffset = offsets.pop(0)
            reader.readData(lastOffset - 1)  # In case first offset is not 1

            for offset in offsets:
                assert lastOffset <= offset
                item = reader.readData(offset - lastOffset)

                if self._itemClass is not None:
                    obj = self._itemClass()
                    obj.decompile(item, font, reader.localState)
                    item = obj
                elif self._converter is not None:
                    item = self._converter.read(item, font)

                items.append(item)
                lastOffset = offset
            return items
        else:

            def get_read_item():
                reader_copy = reader.copy()
                offset_pos = reader.pos
                data_pos = offset_pos + (count + 1) * offSize - 1
                readArray = getReadArray(reader_copy, offSize)

                def read_item(i):
                    reader_copy.seek(offset_pos + i * offSize)
                    offsets = readArray(2)
                    reader_copy.seek(data_pos + offsets[0])
                    item = reader_copy.readData(offsets[1] - offsets[0])

                    if self._itemClass is not None:
                        obj = self._itemClass()
                        obj.decompile(item, font, reader_copy.localState)
                        item = obj
                    elif self._converter is not None:
                        item = self._converter.read(item, font)
                    return item

                return read_item

            read_item = get_read_item()
            l = LazyList([read_item] * count)

            # TODO: Advance reader

            return l

    def write(self, writer, font, tableDict, values, repeatIndex=None):
        items = values

        writer.writeULong(len(items))
        if not len(items):
            return

        if self._itemClass is not None:
            items = [item.compile(font) for item in items]
        elif self._converter is not None:
            items = [
                self._converter.write(writer, font, tableDict, item, i)
                for i, item in enumerate(items)
            ]

        offsets = [len(item) for item in items]
        offsets = list(accumulate(offsets, initial=1))

        lastOffset = offsets[-1]
        offSize = (
            1
            if lastOffset < 0x100
            else 2 if lastOffset < 0x10000 else 3 if lastOffset < 0x1000000 else 4
        )
        writer.writeUInt8(offSize)

        writeArray = {
            1: writer.writeUInt8Array,
            2: writer.writeUShortArray,
            3: writer.writeUInt24Array,
            4: writer.writeULongArray,
        }[offSize]

        writeArray(offsets)
        for item in items:
            writer.writeData(item)

    def xmlRead(self, attrs, content, font):
        if self._itemClass is not None:
            obj = self._itemClass()
            obj.fromXML(None, attrs, content, font)
            return obj
        elif self._converter is not None:
            return self._converter.xmlRead(attrs, content, font)
        else:
            raise NotImplementedError()

    def xmlWrite(self, xmlWriter, font, value, name, attrs):
        if self._itemClass is not None:
            for i, item in enumerate(value):
                item.toXML(xmlWriter, font, [("index", i)], name)
        elif self._converter is not None:
            for i, item in enumerate(value):
                self._converter.xmlWrite(
                    xmlWriter, font, item, name, attrs + [("index", i)]
                )
        else:
            raise NotImplementedError()


class LookupFlag(UShort):
    def xmlWrite(self, xmlWriter, font, value, name, attrs):
        xmlWriter.simpletag(name, attrs + [("value", value)])
        flags = []
        if value & 0x01:
            flags.append("rightToLeft")
        if value & 0x02:
            flags.append("ignoreBaseGlyphs")
        if value & 0x04:
            flags.append("ignoreLigatures")
        if value & 0x08:
            flags.append("ignoreMarks")
        if value & 0x10:
            flags.append("useMarkFilteringSet")
        if value & 0xFF00:
            flags.append("markAttachmentType[%i]" % (value >> 8))
        if flags:
            xmlWriter.comment(" ".join(flags))
        xmlWriter.newline()


class _UInt8Enum(UInt8):
    enumClass = NotImplemented

    def read(self, reader, font, tableDict):
        return self.enumClass(super().read(reader, font, tableDict))

    @classmethod
    def fromString(cls, value):
        return getattr(cls.enumClass, value.upper())

    @classmethod
    def toString(cls, value):
        return cls.enumClass(value).name.lower()


class ExtendMode(_UInt8Enum):
    enumClass = _ExtendMode


class CompositeMode(_UInt8Enum):
    enumClass = _CompositeMode


class MappingEntryFormat(IntFlag):
    HAS_SUBSET_DEF = 0x01
    HAS_CHILD_ENTRIES = 0x02
    HAS_ENTRY_ID = 0x04
    HAS_PATCH_FORMAT = 0x08
    # The code point bias is essentially a 2bit flag.
    CODE_POINT_BIAS_BITS = 0x30

    @property
    def bias_present_u16(self):
        return (self & MappingEntryFormat.CODE_POINT_BIAS_BITS) == 0x20

    @property
    def bias_present_u24(self):
        return (self & MappingEntryFormat.CODE_POINT_BIAS_BITS) == 0x30


# Reference implementation:
# https://github.com/googlefonts/fontations/blob/main/read-fonts/src/tables/ift.rs
class MappingEntriesConverter(BaseConverter):
    def read(self, reader, font, tableDict):
        from fontTools.misc.fixedTools import fixedToFloat as fi2fl
        from fontTools.misc.iftSparseBitSet import decode as sbsDecode

        entryCount = reader["NumEntries"]
        entries = []
        lastEntryId = -1

        for _ in range(entryCount):
            entry = {}
            formatFlags = MappingEntryFormat(reader.readUInt8())
            entry["formatFlags"] = formatFlags

            if formatFlags & MappingEntryFormat.HAS_SUBSET_DEF:
                featureCount = reader.readUInt8()
                features = []
                for _ in range(featureCount):
                    tag = reader.readTag()
                    features.append(tag)
                entry["featureTags"] = features
                designSpaceCount = reader.readUShort()
                segments = []
                for _ in range(designSpaceCount):
                    segTag = reader.readTag()
                    start = fi2fl(reader.readLong(), 16)
                    end = fi2fl(reader.readLong(), 16)
                    segments.append({"tag": segTag, "start": start, "end": end})
                entry["designSpaceSegments"] = segments

            if formatFlags & MappingEntryFormat.HAS_CHILD_ENTRIES:
                modeAndCount = reader.readUInt8()
                conjunctive = bool(modeAndCount & 0x80)
                childCount = modeAndCount & 0x7F
                entry["childEntryConjunctive"] = conjunctive
                childIndices = []
                for _ in range(childCount):
                    idx = reader.readUInt24()
                    childIndices.append(idx)
                entry["childEntryIndices"] = childIndices

            if formatFlags & MappingEntryFormat.HAS_ENTRY_ID:
                ids = []
                while True:
                    # Per IFT spec, Entry IDs are delta-encoded. The value is a
                    # signed 24-bit integer. The least significant bit is a
                    # continuation flag (1 if more IDs follow). The remaining
                    # upper bits are a signed integer, representing the delta
                    # from the previous ID.
                    encodedVal = reader.readInt24()
                    delta = encodedVal >> 1
                    hasMore = encodedVal & 0x01
                    currentEntryId = lastEntryId + 1 + delta
                    ids.append(currentEntryId)
                    lastEntryId = currentEntryId

                    if not hasMore:
                        break
                entry["entryIds"] = ids
            else:
                entry["entryIds"] = [lastEntryId + 1]
                lastEntryId = lastEntryId + 1

            if formatFlags & MappingEntryFormat.HAS_PATCH_FORMAT:
                entry["patchFormat"] = reader.readUInt8()

            if formatFlags & MappingEntryFormat.CODE_POINT_BIAS_BITS:
                bias = 0
                if formatFlags.bias_present_u24:
                    bias = reader.readUInt24()
                elif formatFlags.bias_present_u16:
                    bias = reader.readUShort()
                entry["codePointsBias"] = bias

                codepoints, consumed = sbsDecode(
                    reader.data[reader.pos :], bias=bias, maxValue=0x10FFFF
                )
                reader.advance(consumed)
                entry["codePoints"] = sorted(codepoints)

            entries.append(entry)
        return entries

    def write(self, writer, font, tableDict, value, repeatIndex=None):
        from fontTools.misc.iftSparseBitSet import encode as sbsEncode
        from fontTools.misc.fixedTools import floatToFixed as fl2fi

        entries = value
        lastId = -1

        for entry in entries:
            # TODO: Automatically recompute formatFlags based on the contents of
            # entry.
            formatFlags = entry.get("formatFlags", MappingEntryFormat(0))
            writer.writeUInt8(formatFlags.value)

            if formatFlags & MappingEntryFormat.HAS_SUBSET_DEF:
                features = entry.get("featureTags", [])
                writer.writeUInt8(len(features))
                for tag in features:
                    writer.writeTag(tag)
                segments = entry.get("designSpaceSegments", [])
                writer.writeUShort(len(segments))
                for seg in segments:
                    writer.writeTag(seg["tag"])
                    writer.writeLong(fl2fi(seg["start"], 16))
                    writer.writeLong(fl2fi(seg["end"], 16))

            if formatFlags & MappingEntryFormat.HAS_CHILD_ENTRIES:
                childIndices = entry.get("childEntryIndices", [])
                conjunctive = entry.get("childEntryConjunctive", False)
                modeAndCount = (len(childIndices) & 0x7F) | (0x80 if conjunctive else 0)
                writer.writeUInt8(modeAndCount)
                for idx in childIndices:
                    writer.writeUInt24(idx)

            if formatFlags & MappingEntryFormat.HAS_ENTRY_ID:
                ids = entry.get("entryIds", [])
                for i, eid in enumerate(ids):
                    # Recreate the delta-encoded Entry ID. The delta is shifted
                    # left by one bit, and the least significant bit is set to 1
                    # if more IDs follow in the list.
                    delta = eid - lastId - 1
                    has_more = 1 if i < len(ids) - 1 else 0
                    encoded_value = (delta << 1) | has_more
                    writer.writeUInt24(encoded_value & 0xFFFFFF)
                    lastId = eid
            else:
                lastId += 1

            if formatFlags & MappingEntryFormat.HAS_PATCH_FORMAT:
                writer.writeUInt8(entry.get("patchFormat", 0))

            if formatFlags & MappingEntryFormat.CODE_POINT_BIAS_BITS:
                bias = entry.get("codePointsBias", 0)
                if formatFlags.bias_present_u24:
                    writer.writeUInt24(bias)
                elif formatFlags.bias_present_u16:
                    writer.writeUShort(bias)
                codepoints = entry.get("codePoints", [])
                writer.writeData(sbsEncode([c - bias for c in codepoints if c >= bias]))

    def xmlWrite(self, xmlWriter, font, value, name, attrs):
        for entry in value:
            format_flags = entry.get("formatFlags", MappingEntryFormat(0))
            xmlWriter.begintag("entry", formatFlags=format_flags.value)
            xmlWriter.newline()
            if "entryIds" in entry:
                for eid in entry["entryIds"]:
                    xmlWriter.simpletag("entryId", value=eid)
                    xmlWriter.newline()
            if "featureTags" in entry:
                for tag in entry["featureTags"]:
                    xmlWriter.simpletag("featureTag", value=tag)
                    xmlWriter.newline()
            if "designSpaceSegments" in entry:
                for seg in entry["designSpaceSegments"]:
                    xmlWriter.simpletag(
                        "designSpaceSegment",
                        tag=seg["tag"],
                        start=seg["start"],
                        end=seg["end"],
                    )
                    xmlWriter.newline()
            if "childEntryIndices" in entry:
                xmlWriter.simpletag(
                    "childEntryConjunctive",
                    value=int(entry.get("childEntryConjunctive", False)),
                )
                xmlWriter.newline()
                for idx in entry["childEntryIndices"]:
                    xmlWriter.simpletag("childEntry", index=idx)
                    xmlWriter.newline()
            if "patchFormat" in entry:
                xmlWriter.simpletag("patchFormat", value=entry["patchFormat"])
                xmlWriter.newline()
            if "codePoints" in entry:
                xmlWriter.simpletag(
                    "codePointsBias", value=entry.get("codePointsBias", 0)
                )
                xmlWriter.newline()
                codePointsStr = " ".join(str(c) for c in entry["codePoints"])
                xmlWriter.simpletag("codePoints", value=codePointsStr)
                xmlWriter.newline()
            xmlWriter.endtag("entry")
            xmlWriter.newline()

    def xmlRead(self, attrs, content, font):
        entries = []
        for elem in content:
            if not isinstance(elem, tuple):
                continue
            name, attrs, content = elem
            if name == "entry":
                format_flags_int = safeEval(attrs.get("formatFlags", "0"))
                entry = {"formatFlags": MappingEntryFormat(format_flags_int)}
                for e_elem in content:
                    if not isinstance(e_elem, tuple):
                        continue
                    ename, eattrs, econtent = e_elem
                    if ename == "entryId":
                        entry.setdefault("entryIds", []).append(
                            safeEval(eattrs["value"])
                        )
                    elif ename == "featureTag":
                        entry.setdefault("featureTags", []).append(eattrs["value"])
                    elif ename == "designSpaceSegment":
                        entry.setdefault("designSpaceSegments", []).append(
                            {
                                "tag": eattrs["tag"],
                                "start": float(eattrs["start"]),
                                "end": float(eattrs["end"]),
                            }
                        )
                    elif ename == "childEntryConjunctive":
                        entry["childEntryConjunctive"] = bool(safeEval(eattrs["value"]))
                    elif ename == "childEntry":
                        entry.setdefault("childEntryIndices", []).append(
                            safeEval(eattrs["index"])
                        )
                    elif ename == "patchFormat":
                        entry["patchFormat"] = safeEval(eattrs["value"])
                    elif ename == "codePointsBias":
                        entry["codePointsBias"] = safeEval(eattrs["value"])
                    elif ename == "codePoints":
                        entry["codePoints"] = [
                            int(c) for c in eattrs["value"].split() if c
                        ]
                entries.append(entry)
        return entries


converterMapping = {
    # type		class
    "int8": Int8,
    "int16": Short,
    "int32": Long,
    "uint8": UInt8,
    "uint16": UShort,
    "uint24": UInt24,
    "uint32": ULong,
    "char64": Char64,
    "Flags32": Flags32,
    "VarIndex": VarIndex,
    "Version": Version,
    "Tag": Tag,
    "GlyphID": GlyphID,
    "GlyphID32": GlyphID32,
    "NameID": NameID,
    "DeciPoints": DeciPoints,
    "Fixed": Fixed,
    "F2Dot14": F2Dot14,
    "Angle": Angle,
    "BiasedAngle": BiasedAngle,
    "struct": Struct,
    "MappingEntries": MappingEntriesConverter,
    "Offset": Table,
    "LOffset": LTable,
    "Offset24": Table24,
    "ValueRecord": ValueRecord,
    "DeltaValue": DeltaValue,
    "VarIdxMapValue": VarIdxMapValue,
    "VarDataValue": VarDataValue,
    "LookupFlag": LookupFlag,
    "ExtendMode": ExtendMode,
    "CompositeMode": CompositeMode,
    "STATFlags": STATFlags,
    "TupleList": partial(CFF2Index, itemConverterClass=TupleValues),
    "VarCompositeGlyphList": partial(CFF2Index, itemClass=VarCompositeGlyph),
    # AAT
    "CIDGlyphMap": CIDGlyphMap,
    "GlyphCIDMap": GlyphCIDMap,
    "MortChain": StructWithLength,
    "MortSubtable": MortSubtableConverter,
    "MorxChain": StructWithLength,
    "MorxSubtable": MorxSubtableConverter,
    # "Template" types
    "AATLookup": lambda C: partial(AATLookup, tableClass=C),
    "AATLookupWithDataOffset": lambda C: partial(AATLookupWithDataOffset, tableClass=C),
    "MorphStateTable": lambda C: partial(MorphStateTable, tableClass=C),
    "STXHeader": lambda C: partial(STXHeader, tableClass=C),
    "OffsetTo": lambda C: partial(Table, tableClass=C),
    "LOffsetTo": lambda C: partial(LTable, tableClass=C),
    "LOffset24To": lambda C: partial(Table24, tableClass=C),
}
<|endoftext|>from fontTools.misc import sstruct
from fontTools.misc.fixedTools import floatToFixedToStr
from fontTools.misc.textTools import safeEval
from . import DefaultTable
from . import grUtils
import struct

Feat_hdr_format = """
    >
    version:    16.16F
"""


class table_F__e_a_t(DefaultTable.DefaultTable):
    """Feature table

    The ``Feat`` table is used exclusively by the Graphite shaping engine
    to store features and possible settings specified in GDL. Graphite features
    determine what rules are applied to transform a glyph stream.

    Not to be confused with ``feat``, or the OpenType Layout tables
    ``GSUB``/``GPOS``.

    See also https://graphite.sil.org/graphite_techAbout#graphite-font-tables
    """

    def __init__(self, tag=None):
        DefaultTable.DefaultTable.__init__(self, tag)
        self.features = {}

    def decompile(self, data, ttFont):
        _, data = sstruct.unpack2(Feat_hdr_format, data, self)
        self.version = float(floatToFixedToStr(self.version, precisionBits=16))
        (numFeats,) = struct.unpack(">H", data[:2])
        data = data[8:]
        allfeats = []
        maxsetting = 0
        for i in range(numFeats):
            if self.version >= 2.0:
                fid, nums, _, offset, flags, lid = struct.unpack(
                    ">LHHLHH", data[16 * i : 16 * (i + 1)]
                )
                offset = int((offset - 12 - 16 * numFeats) / 4)
            else:
                fid, nums, offset, flags, lid = struct.unpack(
                    ">HHLHH", data[12 * i : 12 * (i + 1)]
                )
                offset = int((offset - 12 - 12 * numFeats) / 4)
            allfeats.append((fid, nums, offset, flags, lid))
            maxsetting = max(maxsetting, offset + nums)
        data = data[16 * numFeats :]
        allsettings = []
        for i in range(maxsetting):
            if len(data) >= 4 * (i + 1):
                val, lid = struct.unpack(">HH", data[4 * i : 4 * (i + 1)])
                allsettings.append((val, lid))
        for i, f in enumerate(allfeats):
            fid, nums, offset, flags, lid = f
            fobj = Feature()
            fobj.flags = flags
            fobj.label = lid
            self.features[grUtils.num2tag(fid)] = fobj
            fobj.settings = {}
            fobj.default = None
            fobj.index = i
            for i in range(offset, offset + nums):
                if i >= len(allsettings):
                    continue
                vid, vlid = allsettings[i]
                fobj.settings[vid] = vlid
                if fobj.default is None:
                    fobj.default = vid

    def compile(self, ttFont):
        fdat = b""
        vdat = b""
        offset = 0
        for f, v in sorted(self.features.items(), key=lambda x: x[1].index):
            fnum = grUtils.tag2num(f)
            if self.version >= 2.0:
                fdat += struct.pack(
                    ">LHHLHH",
                    grUtils.tag2num(f),
                    len(v.settings),
                    0,
                    offset * 4 + 12 + 16 * len(self.features),
                    v.flags,
                    v.label,
                )
            elif fnum > 65535:  # self healing for alphabetic ids
                self.version = 2.0
                return self.compile(ttFont)
            else:
                fdat += struct.pack(
                    ">HHLHH",
                    grUtils.tag2num(f),
                    len(v.settings),
                    offset * 4 + 12 + 12 * len(self.features),
                    v.flags,
                    v.label,
                )
            for s, l in sorted(
                v.settings.items(), key=lambda x: (-1, x[1]) if x[0] == v.default else x
            ):
                vdat += struct.pack(">HH", s, l)
            offset += len(v.settings)
        hdr = sstruct.pack(Feat_hdr_format, self)
        return hdr + struct.pack(">HHL", len(self.features), 0, 0) + fdat + vdat

    def toXML(self, writer, ttFont):
        writer.simpletag("version", version=self.version)
        writer.newline()
        nameTable = ttFont["name"] if "name" in ttFont else None
        for f, v in sorted(self.features.items(), key=lambda x: x[1].index):
            name = nameTable.getDebugName(v.label) if nameTable else None
            if name is not None:
                writer.newline()
                writer.comment(name)
                writer.newline()
            writer.begintag(
                "feature",
                fid=f,
                label=v.label,
                flags=v.flags,
                default=(v.default if v.default else 0),
            )
            writer.newline()
            for s, l in sorted(v.settings.items()):
                writer.simpletag("setting", value=s, label=l)
                name = nameTable.getDebugName(l) if nameTable else None
                if name is not None:
                    writer.write("  ")
                    writer.comment(name)
                writer.newline()
            writer.endtag("feature")
            writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        if name == "version":
            self.version = float(safeEval(attrs["version"]))
        elif name == "feature":
            fid = attrs["fid"]
            fobj = Feature()
            fobj.flags = int(safeEval(attrs["flags"]))
            fobj.label = int(safeEval(attrs["label"]))
            fobj.default = int(safeEval(attrs.get("default", "0")))
            fobj.index = len(self.features)
            self.features[fid] = fobj
            fobj.settings = {}
            for element in content:
                if not isinstance(element, tuple):
                    continue
                tag, a, c = element
                if tag == "setting":
                    fobj.settings[int(safeEval(a["value"]))] = int(safeEval(a["label"]))


class Feature(object):
    pass
<|endoftext|># Since bitmap glyph metrics are shared between EBLC and EBDT
# this class gets its own python file.
from fontTools.misc import sstruct
from fontTools.misc.textTools import safeEval
import logging

log = logging.getLogger(__name__)

bigGlyphMetricsFormat = """
  > # big endian
  height:       B
  width:        B
  horiBearingX: b
  horiBearingY: b
  horiAdvance:  B
  vertBearingX: b
  vertBearingY: b
  vertAdvance:  B
"""

smallGlyphMetricsFormat = """
  > # big endian
  height:   B
  width:    B
  BearingX: b
  BearingY: b
  Advance:  B
"""


class BitmapGlyphMetrics(object):
    def toXML(self, writer, ttFont):
        writer.begintag(self.__class__.__name__)
        writer.newline()
        for metricName in sstruct.getformat(self.__class__.binaryFormat)[1]:
            writer.simpletag(metricName, value=getattr(self, metricName))
            writer.newline()
        writer.endtag(self.__class__.__name__)
        writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        metricNames = set(sstruct.getformat(self.__class__.binaryFormat)[1])
        for element in content:
            if not isinstance(element, tuple):
                continue
            name, attrs, content = element
            # Make sure this is a metric that is needed by GlyphMetrics.
            if name in metricNames:
                vars(self)[name] = safeEval(attrs["value"])
            else:
                log.warning(
                    "unknown name '%s' being ignored in %s.",
                    name,
                    self.__class__.__name__,
                )


class BigGlyphMetrics(BitmapGlyphMetrics):
    binaryFormat = bigGlyphMetricsFormat


class SmallGlyphMetrics(BitmapGlyphMetrics):
    binaryFormat = smallGlyphMetricsFormat
<|endoftext|>from collections import deque
from functools import partial
from fontTools.misc import sstruct
from fontTools.misc.textTools import safeEval
from fontTools.misc.lazyTools import LazyDict
from fontTools.ttLib import OPTIMIZE_FONT_SPEED
from fontTools.ttLib.tables.TupleVariation import TupleVariation
from . import DefaultTable
import array
import itertools
import logging
import struct
import sys
import fontTools.ttLib.tables.TupleVariation as tv

log = logging.getLogger(__name__)

# https://www.microsoft.com/typography/otspec/gvar.htm
# https://www.microsoft.com/typography/otspec/otvarcommonformats.htm
#
# Apple's documentation of 'gvar':
# https://developer.apple.com/fonts/TrueType-Reference-Manual/RM06/Chap6gvar.html
#
# FreeType2 source code for parsing 'gvar':
# http://git.savannah.gnu.org/cgit/freetype/freetype2.git/tree/src/truetype/ttgxvar.c

GVAR_HEADER_FORMAT_HEAD = """
	> # big endian
	version:			H
	reserved:			H
	axisCount:			H
	sharedTupleCount:		H
	offsetToSharedTuples:		I
"""
# In between the HEAD and TAIL lies the glyphCount, which is
# of different size: 2 bytes for gvar, and 3 bytes for GVAR.
GVAR_HEADER_FORMAT_TAIL = """
	> # big endian
	flags:				H
	offsetToGlyphVariationData:	I
"""

GVAR_HEADER_SIZE_HEAD = sstruct.calcsize(GVAR_HEADER_FORMAT_HEAD)
GVAR_HEADER_SIZE_TAIL = sstruct.calcsize(GVAR_HEADER_FORMAT_TAIL)


class table__g_v_a_r(DefaultTable.DefaultTable):
    """Glyph Variations table

    The ``gvar`` table provides the per-glyph variation data that
    describe how glyph outlines in the ``glyf`` table change across
    the variation space that is defined for the font in the ``fvar``
    table.

    See also https://learn.microsoft.com/en-us/typography/opentype/spec/gvar
    """

    dependencies = ["fvar", "glyf"]
    gid_size = 2

    def __init__(self, tag=None):
        DefaultTable.DefaultTable.__init__(self, tag)
        self.version, self.reserved = 1, 0
        self.variations = {}

    def compile(self, ttFont):
        axisTags = [axis.axisTag for axis in ttFont["fvar"].axes]
        sharedTuples = tv.compileSharedTuples(
            axisTags, itertools.chain(*self.variations.values())
        )
        sharedTupleIndices = {coord: i for i, coord in enumerate(sharedTuples)}
        sharedTupleSize = sum([len(c) for c in sharedTuples])
        compiledGlyphs = self.compileGlyphs_(ttFont, axisTags, sharedTupleIndices)
        offset = 0
        offsets = []
        for glyph in compiledGlyphs:
            offsets.append(offset)
            offset += len(glyph)
        offsets.append(offset)
        compiledOffsets, tableFormat = self.compileOffsets_(offsets)

        GVAR_HEADER_SIZE = GVAR_HEADER_SIZE_HEAD + self.gid_size + GVAR_HEADER_SIZE_TAIL
        header = {}
        header["version"] = self.version
        header["reserved"] = self.reserved
        header["axisCount"] = len(axisTags)
        header["sharedTupleCount"] = len(sharedTuples)
        header["offsetToSharedTuples"] = GVAR_HEADER_SIZE + len(compiledOffsets)
        header["flags"] = tableFormat
        header["offsetToGlyphVariationData"] = (
            header["offsetToSharedTuples"] + sharedTupleSize
        )

        result = [
            sstruct.pack(GVAR_HEADER_FORMAT_HEAD, header),
            len(compiledGlyphs).to_bytes(self.gid_size, "big"),
            sstruct.pack(GVAR_HEADER_FORMAT_TAIL, header),
        ]

        result.append(compiledOffsets)
        result.extend(sharedTuples)
        result.extend(compiledGlyphs)
        return b"".join(result)

    def compileGlyphs_(self, ttFont, axisTags, sharedCoordIndices):
        optimizeSpeed = ttFont.cfg[OPTIMIZE_FONT_SPEED]
        result = []
        glyf = ttFont["glyf"]
        for glyphName in ttFont.getGlyphOrder():
            variations = self.variations.get(glyphName, [])
            if not variations:
                result.append(b"")
                continue
            pointCountUnused = 0  # pointCount is actually unused by compileGlyph
            result.append(
                compileGlyph_(
                    self.gid_size,
                    variations,
                    pointCountUnused,
                    axisTags,
                    sharedCoordIndices,
                    optimizeSize=not optimizeSpeed,
                )
            )
        return result

    def decompile(self, data, ttFont):
        axisTags = [axis.axisTag for axis in ttFont["fvar"].axes]
        glyphs = ttFont.getGlyphOrder()

        # Parse the header
        GVAR_HEADER_SIZE = GVAR_HEADER_SIZE_HEAD + self.gid_size + GVAR_HEADER_SIZE_TAIL
        sstruct.unpack(GVAR_HEADER_FORMAT_HEAD, data[:GVAR_HEADER_SIZE_HEAD], self)
        self.glyphCount = int.from_bytes(
            data[GVAR_HEADER_SIZE_HEAD : GVAR_HEADER_SIZE_HEAD + self.gid_size], "big"
        )
        sstruct.unpack(
            GVAR_HEADER_FORMAT_TAIL,
            data[GVAR_HEADER_SIZE_HEAD + self.gid_size : GVAR_HEADER_SIZE],
            self,
        )

        assert len(glyphs) == self.glyphCount, (len(glyphs), self.glyphCount)
        assert len(axisTags) == self.axisCount, (
            len(axisTags),
            self.axisCount,
            axisTags,
        )
        sharedCoords = tv.decompileSharedTuples(
            axisTags, self.sharedTupleCount, data, self.offsetToSharedTuples
        )
        variations = {}
        offsetToData = self.offsetToGlyphVariationData
        glyf = ttFont["glyf"]

        def get_read_item():
            reverseGlyphMap = ttFont.getReverseGlyphMap()
            tableFormat = self.flags & 1

            def read_item(glyphName):
                gid = reverseGlyphMap[glyphName]
                offsetSize = 2 if tableFormat == 0 else 4
                startOffset = GVAR_HEADER_SIZE + offsetSize * gid
                endOffset = startOffset + offsetSize * 2
                offsets = table__g_v_a_r.decompileOffsets_(
                    data[startOffset:endOffset],
                    tableFormat=tableFormat,
                    glyphCount=1,
                )
                gvarData = data[offsetToData + offsets[0] : offsetToData + offsets[1]]
                if not gvarData:
                    return []
                glyph = glyf[glyphName]
                numPointsInGlyph = self.getNumPoints_(glyph)
                return decompileGlyph_(
                    self.gid_size, numPointsInGlyph, sharedCoords, axisTags, gvarData
                )

            return read_item

        read_item = get_read_item()
        l = LazyDict({glyphs[gid]: read_item for gid in range(self.glyphCount)})

        self.variations = l

        if ttFont.lazy is False:  # Be lazy for None and True
            self.ensureDecompiled()

    def ensureDecompiled(self, recurse=False):
        # The recurse argument is unused, but part of the signature of
        # ensureDecompiled across the library.
        # Use a zero-length deque to consume the lazy dict
        deque(self.variations.values(), maxlen=0)

    @staticmethod
    def decompileOffsets_(data, tableFormat, glyphCount):
        if tableFormat == 0:
            # Short format: array of UInt16
            offsets = array.array("H")
            offsetsSize = (glyphCount + 1) * 2
        else:
            # Long format: array of UInt32
            offsets = array.array("I")
            offsetsSize = (glyphCount + 1) * 4
        offsets.frombytes(data[0:offsetsSize])
        if sys.byteorder != "big":
            offsets.byteswap()

        # In the short format, offsets need to be multiplied by 2.
        # This is not documented in Apple's TrueType specification,
        # but can be inferred from the FreeType implementation, and
        # we could verify it with two sample GX fonts.
        if tableFormat == 0:
            offsets = [off * 2 for off in offsets]

        return offsets

    @staticmethod
    def compileOffsets_(offsets):
        """Packs a list of offsets into a 'gvar' offset table.

        Returns a pair (bytestring, tableFormat). Bytestring is the
        packed offset table. Format indicates whether the table
        uses short (tableFormat=0) or long (tableFormat=1) integers.
        The returned tableFormat should get packed into the flags field
        of the 'gvar' header.
        """
        assert len(offsets) >= 2
        for i in range(1, len(offsets)):
            assert offsets[i - 1] <= offsets[i]
        if max(offsets) <= 0xFFFF * 2:
            packed = array.array("H", [n >> 1 for n in offsets])
            tableFormat = 0
        else:
            packed = array.array("I", offsets)
            tableFormat = 1
        if sys.byteorder != "big":
            packed.byteswap()
        return (packed.tobytes(), tableFormat)

    def toXML(self, writer, ttFont):
        writer.simpletag("version", value=self.version)
        writer.newline()
        writer.simpletag("reserved", value=self.reserved)
        writer.newline()
        axisTags = [axis.axisTag for axis in ttFont["fvar"].axes]
        for glyphName in ttFont.getGlyphNames():
            variations = self.variations.get(glyphName)
            if not variations:
                continue
            writer.begintag("glyphVariations", glyph=glyphName)
            writer.newline()
            for gvar in variations:
                gvar.toXML(writer, axisTags)
            writer.endtag("glyphVariations")
            writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        if name == "version":
            self.version = safeEval(attrs["value"])
        elif name == "reserved":
            self.reserved = safeEval(attrs["value"])
        elif name == "glyphVariations":
            if not hasattr(self, "variations"):
                self.variations = {}
            glyphName = attrs["glyph"]
            glyph = ttFont["glyf"][glyphName]
            numPointsInGlyph = self.getNumPoints_(glyph)
            glyphVariations = []
            for element in content:
                if isinstance(element, tuple):
                    name, attrs, content = element
                    if name == "tuple":
                        gvar = TupleVariation({}, [None] * numPointsInGlyph)
                        glyphVariations.append(gvar)
                        for tupleElement in content:
                            if isinstance(tupleElement, tuple):
                                tupleName, tupleAttrs, tupleContent = tupleElement
                                gvar.fromXML(tupleName, tupleAttrs, tupleContent)
            self.variations[glyphName] = glyphVariations

    @staticmethod
    def getNumPoints_(glyph):
        NUM_PHANTOM_POINTS = 4

        if glyph.isComposite():
            return len(glyph.components) + NUM_PHANTOM_POINTS
        else:
            # Empty glyphs (eg. space, nonmarkingreturn) have no "coordinates" attribute.
            return len(getattr(glyph, "coordinates", [])) + NUM_PHANTOM_POINTS


def compileGlyph_(
    dataOffsetSize,
    variations,
    pointCount,
    axisTags,
    sharedCoordIndices,
    *,
    optimizeSize=True,
):
    assert dataOffsetSize in (2, 3)
    tupleVariationCount, tuples, data = tv.compileTupleVariationStore(
        variations, pointCount, axisTags, sharedCoordIndices, optimizeSize=optimizeSize
    )
    if tupleVariationCount == 0:
        return b""

    offsetToData = 2 + dataOffsetSize + len(tuples)

    result = [
        tupleVariationCount.to_bytes(2, "big"),
        offsetToData.to_bytes(dataOffsetSize, "big"),
        tuples,
        data,
    ]
    if (offsetToData + len(data)) % 2 != 0:
        result.append(b"\0")  # padding
    return b"".join(result)


def decompileGlyph_(dataOffsetSize, pointCount, sharedTuples, axisTags, data):
    assert dataOffsetSize in (2, 3)
    if len(data) < 2 + dataOffsetSize:
        return []

    tupleVariationCount = int.from_bytes(data[:2], "big")
    offsetToData = int.from_bytes(data[2 : 2 + dataOffsetSize], "big")

    dataPos = offsetToData
    return tv.decompileTupleVariationStore(
        "gvar",
        axisTags,
        tupleVariationCount,
        pointCount,
        sharedTuples,
        data,
        2 + dataOffsetSize,
        offsetToData,
    )
<|endoftext|># Copyright 2013 Google, Inc. All Rights Reserved.
#
# Google Author(s): Behdad Esfahbod

from fontTools.misc.textTools import bytesjoin, safeEval
from . import DefaultTable
import array
from collections import namedtuple
import struct
import sys


class table_C_P_A_L_(DefaultTable.DefaultTable):
    """Color Palette table

    The ``CPAL`` table contains a set of one or more color palettes. The color
    records in each palette can be referenced by the ``COLR`` table to specify
    the colors used in a color glyph.

    See also https://learn.microsoft.com/en-us/typography/opentype/spec/cpal
    """

    NO_NAME_ID = 0xFFFF
    DEFAULT_PALETTE_TYPE = 0

    def __init__(self, tag=None):
        DefaultTable.DefaultTable.__init__(self, tag)
        self.palettes = []
        self.paletteTypes = []
        self.paletteLabels = []
        self.paletteEntryLabels = []

    def decompile(self, data, ttFont):
        (
            self.version,
            self.numPaletteEntries,
            numPalettes,
            numColorRecords,
            goffsetFirstColorRecord,
        ) = struct.unpack(">HHHHL", data[:12])
        assert (
            self.version <= 1
        ), "Version of CPAL table is higher than I know how to handle"
        self.palettes = []
        pos = 12
        for i in range(numPalettes):
            startIndex = struct.unpack(">H", data[pos : pos + 2])[0]
            assert startIndex + self.numPaletteEntries <= numColorRecords
            pos += 2
            palette = []
            ppos = goffsetFirstColorRecord + startIndex * 4
            for j in range(self.numPaletteEntries):
                palette.append(Color(*struct.unpack(">BBBB", data[ppos : ppos + 4])))
                ppos += 4
            self.palettes.append(palette)
        if self.version == 0:
            offsetToPaletteTypeArray = 0
            offsetToPaletteLabelArray = 0
            offsetToPaletteEntryLabelArray = 0
        else:
            pos = 12 + numPalettes * 2
            (
                offsetToPaletteTypeArray,
                offsetToPaletteLabelArray,
                offsetToPaletteEntryLabelArray,
            ) = struct.unpack(">LLL", data[pos : pos + 12])
        self.paletteTypes = self._decompileUInt32Array(
            data,
            offsetToPaletteTypeArray,
            numPalettes,
            default=self.DEFAULT_PALETTE_TYPE,
        )
        self.paletteLabels = self._decompileUInt16Array(
            data, offsetToPaletteLabelArray, numPalettes, default=self.NO_NAME_ID
        )
        self.paletteEntryLabels = self._decompileUInt16Array(
            data,
            offsetToPaletteEntryLabelArray,
            self.numPaletteEntries,
            default=self.NO_NAME_ID,
        )

    def _decompileUInt16Array(self, data, offset, numElements, default=0):
        if offset == 0:
            return [default] * numElements
        result = array.array("H", data[offset : offset + 2 * numElements])
        if sys.byteorder != "big":
            result.byteswap()
        assert len(result) == numElements, result
        return result.tolist()

    def _decompileUInt32Array(self, data, offset, numElements, default=0):
        if offset == 0:
            return [default] * numElements
        result = array.array("I", data[offset : offset + 4 * numElements])
        if sys.byteorder != "big":
            result.byteswap()
        assert len(result) == numElements, result
        return result.tolist()

    def compile(self, ttFont):
        colorRecordIndices, colorRecords = self._compileColorRecords()
        paletteTypes = self._compilePaletteTypes()
        paletteLabels = self._compilePaletteLabels()
        paletteEntryLabels = self._compilePaletteEntryLabels()
        numColorRecords = len(colorRecords) // 4
        offsetToFirstColorRecord = 12 + len(colorRecordIndices)
        if self.version >= 1:
            offsetToFirstColorRecord += 12
        header = struct.pack(
            ">HHHHL",
            self.version,
            self.numPaletteEntries,
            len(self.palettes),
            numColorRecords,
            offsetToFirstColorRecord,
        )
        if self.version == 0:
            dataList = [header, colorRecordIndices, colorRecords]
        else:
            pos = offsetToFirstColorRecord + len(colorRecords)
            if len(paletteTypes) == 0:
                offsetToPaletteTypeArray = 0
            else:
                offsetToPaletteTypeArray = pos
                pos += len(paletteTypes)
            if len(paletteLabels) == 0:
                offsetToPaletteLabelArray = 0
            else:
                offsetToPaletteLabelArray = pos
                pos += len(paletteLabels)
            if len(paletteEntryLabels) == 0:
                offsetToPaletteEntryLabelArray = 0
            else:
                offsetToPaletteEntryLabelArray = pos
                pos += len(paletteLabels)
            header1 = struct.pack(
                ">LLL",
                offsetToPaletteTypeArray,
                offsetToPaletteLabelArray,
                offsetToPaletteEntryLabelArray,
            )
            dataList = [
                header,
                colorRecordIndices,
                header1,
                colorRecords,
                paletteTypes,
                paletteLabels,
                paletteEntryLabels,
            ]
        return bytesjoin(dataList)

    def _compilePalette(self, palette):
        assert len(palette) == self.numPaletteEntries
        pack = lambda c: struct.pack(">BBBB", c.blue, c.green, c.red, c.alpha)
        return bytesjoin([pack(color) for color in palette])

    def _compileColorRecords(self):
        colorRecords, colorRecordIndices, pool = [], [], {}
        for palette in self.palettes:
            packedPalette = self._compilePalette(palette)
            if packedPalette in pool:
                index = pool[packedPalette]
            else:
                index = len(colorRecords)
                colorRecords.append(packedPalette)
                pool[packedPalette] = index
            colorRecordIndices.append(struct.pack(">H", index * self.numPaletteEntries))
        return bytesjoin(colorRecordIndices), bytesjoin(colorRecords)

    def _compilePaletteTypes(self):
        if self.version == 0 or not any(self.paletteTypes):
            return b""
        assert len(self.paletteTypes) == len(self.palettes)
        result = bytesjoin([struct.pack(">I", ptype) for ptype in self.paletteTypes])
        assert len(result) == 4 * len(self.palettes)
        return result

    def _compilePaletteLabels(self):
        if self.version == 0 or all(l == self.NO_NAME_ID for l in self.paletteLabels):
            return b""
        assert len(self.paletteLabels) == len(self.palettes)
        result = bytesjoin([struct.pack(">H", label) for label in self.paletteLabels])
        assert len(result) == 2 * len(self.palettes)
        return result

    def _compilePaletteEntryLabels(self):
        if self.version == 0 or all(
            l == self.NO_NAME_ID for l in self.paletteEntryLabels
        ):
            return b""
        assert len(self.paletteEntryLabels) == self.numPaletteEntries
        result = bytesjoin(
            [struct.pack(">H", label) for label in self.paletteEntryLabels]
        )
        assert len(result) == 2 * self.numPaletteEntries
        return result

    def toXML(self, writer, ttFont):
        numPalettes = len(self.palettes)
        paletteLabels = {i: nameID for (i, nameID) in enumerate(self.paletteLabels)}
        paletteTypes = {i: typ for (i, typ) in enumerate(self.paletteTypes)}
        writer.simpletag("version", value=self.version)
        writer.newline()
        writer.simpletag("numPaletteEntries", value=self.numPaletteEntries)
        writer.newline()
        for index, palette in enumerate(self.palettes):
            attrs = {"index": index}
            paletteType = paletteTypes.get(index, self.DEFAULT_PALETTE_TYPE)
            paletteLabel = paletteLabels.get(index, self.NO_NAME_ID)
            if self.version > 0 and paletteLabel != self.NO_NAME_ID:
                attrs["label"] = paletteLabel
            if self.version > 0 and paletteType != self.DEFAULT_PALETTE_TYPE:
                attrs["type"] = paletteType
            writer.begintag("palette", **attrs)
            writer.newline()
            if (
                self.version > 0
                and paletteLabel != self.NO_NAME_ID
                and ttFont
                and "name" in ttFont
            ):
                name = ttFont["name"].getDebugName(paletteLabel)
                if name is not None:
                    writer.comment(name)
                    writer.newline()
            assert len(palette) == self.numPaletteEntries
            for cindex, color in enumerate(palette):
                color.toXML(writer, ttFont, cindex)
            writer.endtag("palette")
            writer.newline()
        if self.version > 0 and not all(
            l == self.NO_NAME_ID for l in self.paletteEntryLabels
        ):
            writer.begintag("paletteEntryLabels")
            writer.newline()
            for index, label in enumerate(self.paletteEntryLabels):
                if label != self.NO_NAME_ID:
                    writer.simpletag("label", index=index, value=label)
                    if self.version > 0 and label and ttFont and "name" in ttFont:
                        name = ttFont["name"].getDebugName(label)
                        if name is not None:
                            writer.comment(name)
                    writer.newline()
            writer.endtag("paletteEntryLabels")
            writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        if name == "palette":
            self.paletteLabels.append(int(attrs.get("label", self.NO_NAME_ID)))
            self.paletteTypes.append(int(attrs.get("type", self.DEFAULT_PALETTE_TYPE)))
            palette = []
            for element in content:
                if isinstance(element, str):
                    continue
                attrs = element[1]
                color = Color.fromHex(attrs["value"])
                palette.append(color)
            self.palettes.append(palette)
        elif name == "paletteEntryLabels":
            colorLabels = {}
            for element in content:
                if isinstance(element, str):
                    continue
                elementName, elementAttr, _ = element
                if elementName == "label":
                    labelIndex = safeEval(elementAttr["index"])
                    nameID = safeEval(elementAttr["value"])
                    colorLabels[labelIndex] = nameID
            self.paletteEntryLabels = [
                colorLabels.get(i, self.NO_NAME_ID)
                for i in range(self.numPaletteEntries)
            ]
        elif "value" in attrs:
            value = safeEval(attrs["value"])
            setattr(self, name, value)
            if name == "numPaletteEntries":
                self.paletteEntryLabels = [self.NO_NAME_ID] * self.numPaletteEntries


class Color(namedtuple("Color", "blue green red alpha")):
    def hex(self):
        return "#%02X%02X%02X%02X" % (self.red, self.green, self.blue, self.alpha)

    def __repr__(self):
        return self.hex()

    def toXML(self, writer, ttFont, index=None):
        writer.simpletag("color", value=self.hex(), index=index)
        writer.newline()

    @classmethod
    def fromHex(cls, value):
        if value[0] == "#":
            value = value[1:]
        red = int(value[0:2], 16)
        green = int(value[2:4], 16)
        blue = int(value[4:6], 16)
        alpha = int(value[6:8], 16) if len(value) >= 8 else 0xFF
        return cls(red=red, green=green, blue=blue, alpha=alpha)

    @classmethod
    def fromRGBA(cls, red, green, blue, alpha):
        return cls(red=red, green=green, blue=blue, alpha=alpha)
<|endoftext|># Copyright 2013 Google, Inc. All Rights Reserved.
#
# Google Author(s): Matt Fontaine


from fontTools.misc.textTools import bytesjoin
from fontTools.misc import sstruct
from . import E_B_D_T_
from .BitmapGlyphMetrics import (
    BigGlyphMetrics,
    bigGlyphMetricsFormat,
    SmallGlyphMetrics,
    smallGlyphMetricsFormat,
)
from .E_B_D_T_ import (
    BitmapGlyph,
    BitmapPlusSmallMetricsMixin,
    BitmapPlusBigMetricsMixin,
)
import struct


class table_C_B_D_T_(E_B_D_T_.table_E_B_D_T_):
    """Color Bitmap Data table

    The ``CBDT`` table contains color bitmap data for glyphs. It must
    be used in concert with the ``CBLC`` table.

    It is backwards-compatible with the monochrome/grayscale ``EBDT`` table.

    See also https://learn.microsoft.com/en-us/typography/opentype/spec/cbdt
    """

    # Change the data locator table being referenced.
    locatorName = "CBLC"

    # Modify the format class accessor for color bitmap use.
    def getImageFormatClass(self, imageFormat):
        try:
            return E_B_D_T_.table_E_B_D_T_.getImageFormatClass(self, imageFormat)
        except KeyError:
            return cbdt_bitmap_classes[imageFormat]


# Helper method for removing export features not supported by color bitmaps.
# Write data in the parent class will default to raw if an option is unsupported.
def _removeUnsupportedForColor(dataFunctions):
    dataFunctions = dict(dataFunctions)
    del dataFunctions["row"]
    return dataFunctions


class ColorBitmapGlyph(BitmapGlyph):
    fileExtension = ".png"
    xmlDataFunctions = _removeUnsupportedForColor(BitmapGlyph.xmlDataFunctions)


class cbdt_bitmap_format_17(BitmapPlusSmallMetricsMixin, ColorBitmapGlyph):
    def decompile(self):
        self.metrics = SmallGlyphMetrics()
        dummy, data = sstruct.unpack2(smallGlyphMetricsFormat, self.data, self.metrics)
        (dataLen,) = struct.unpack(">L", data[:4])
        data = data[4:]

        # For the image data cut it to the size specified by dataLen.
        assert dataLen <= len(data), "Data overun in format 17"
        self.imageData = data[:dataLen]

    def compile(self, ttFont):
        dataList = []
        dataList.append(sstruct.pack(smallGlyphMetricsFormat, self.metrics))
        dataList.append(struct.pack(">L", len(self.imageData)))
        dataList.append(self.imageData)
        return bytesjoin(dataList)


class cbdt_bitmap_format_18(BitmapPlusBigMetricsMixin, ColorBitmapGlyph):
    def decompile(self):
        self.metrics = BigGlyphMetrics()
        dummy, data = sstruct.unpack2(bigGlyphMetricsFormat, self.data, self.metrics)
        (dataLen,) = struct.unpack(">L", data[:4])
        data = data[4:]

        # For the image data cut it to the size specified by dataLen.
        assert dataLen <= len(data), "Data overun in format 18"
        self.imageData = data[:dataLen]

    def compile(self, ttFont):
        dataList = []
        dataList.append(sstruct.pack(bigGlyphMetricsFormat, self.metrics))
        dataList.append(struct.pack(">L", len(self.imageData)))
        dataList.append(self.imageData)
        return bytesjoin(dataList)


class cbdt_bitmap_format_19(ColorBitmapGlyph):
    def decompile(self):
        (dataLen,) = struct.unpack(">L", self.data[:4])
        data = self.data[4:]

        assert dataLen <= len(data), "Data overun in format 19"
        self.imageData = data[:dataLen]

    def compile(self, ttFont):
        return struct.pack(">L", len(self.imageData)) + self.imageData


# Dict for CBDT extended formats.
cbdt_bitmap_classes = {
    17: cbdt_bitmap_format_17,
    18: cbdt_bitmap_format_18,
    19: cbdt_bitmap_format_19,
}
<|endoftext|>from fontTools.misc import sstruct
from fontTools.misc.fixedTools import (
    fixedToFloat as fi2fl,
    floatToFixed as fl2fi,
    floatToFixedToStr as fl2str,
    strToFixedToFloat as str2fl,
)
from fontTools.misc.textTools import Tag, bytesjoin, safeEval
from fontTools.ttLib import TTLibError
from . import DefaultTable
import struct

# Apple's documentation of 'fvar':
# https://developer.apple.com/fonts/TrueType-Reference-Manual/RM06/Chap6fvar.html

FVAR_HEADER_FORMAT = """
    > # big endian
    version:        L
    offsetToData:   H
    countSizePairs: H
    axisCount:      H
    axisSize:       H
    instanceCount:  H
    instanceSize:   H
"""

FVAR_AXIS_FORMAT = """
    > # big endian
    axisTag:        4s
    minValue:       16.16F
    defaultValue:   16.16F
    maxValue:       16.16F
    flags:          H
    axisNameID:         H
"""

FVAR_INSTANCE_FORMAT = """
    > # big endian
    subfamilyNameID:     H
    flags:      H
"""


class table__f_v_a_r(DefaultTable.DefaultTable):
    """FonT Variations table

    The ``fvar`` table contains records of the variation axes and of the
    named instances in a variable font.

    See also https://developer.apple.com/fonts/TrueType-Reference-Manual/RM06/Chap6fvar.html
    """

    def __init__(self, tag=None):
        DefaultTable.DefaultTable.__init__(self, tag)
        self.axes = []
        self.instances = []

    def compile(self, ttFont):
        instanceSize = sstruct.calcsize(FVAR_INSTANCE_FORMAT) + (len(self.axes) * 4)
        includePostScriptNames = any(
            instance.postscriptNameID != 0xFFFF for instance in self.instances
        )
        if includePostScriptNames:
            instanceSize += 2
        header = {
            "version": 0x00010000,
            "offsetToData": sstruct.calcsize(FVAR_HEADER_FORMAT),
            "countSizePairs": 2,
            "axisCount": len(self.axes),
            "axisSize": sstruct.calcsize(FVAR_AXIS_FORMAT),
            "instanceCount": len(self.instances),
            "instanceSize": instanceSize,
        }
        result = [sstruct.pack(FVAR_HEADER_FORMAT, header)]
        result.extend([axis.compile() for axis in self.axes])
        axisTags = [axis.axisTag for axis in self.axes]
        for instance in self.instances:
            result.append(instance.compile(axisTags, includePostScriptNames))
        return bytesjoin(result)

    def decompile(self, data, ttFont):
        header = {}
        headerSize = sstruct.calcsize(FVAR_HEADER_FORMAT)
        header = sstruct.unpack(FVAR_HEADER_FORMAT, data[0:headerSize])
        if header["version"] != 0x00010000:
            raise TTLibError("unsupported 'fvar' version %04x" % header["version"])
        pos = header["offsetToData"]
        axisSize = header["axisSize"]
        for _ in range(header["axisCount"]):
            axis = Axis()
            axis.decompile(data[pos : pos + axisSize])
            self.axes.append(axis)
            pos += axisSize
        instanceSize = header["instanceSize"]
        axisTags = [axis.axisTag for axis in self.axes]
        for _ in range(header["instanceCount"]):
            instance = NamedInstance()
            instance.decompile(data[pos : pos + instanceSize], axisTags)
            self.instances.append(instance)
            pos += instanceSize

    def toXML(self, writer, ttFont):
        for axis in self.axes:
            axis.toXML(writer, ttFont)
        for instance in self.instances:
            instance.toXML(writer, ttFont)

    def fromXML(self, name, attrs, content, ttFont):
        if name == "Axis":
            axis = Axis()
            axis.fromXML(name, attrs, content, ttFont)
            self.axes.append(axis)
        elif name == "NamedInstance":
            instance = NamedInstance()
            instance.fromXML(name, attrs, content, ttFont)
            self.instances.append(instance)

    def getAxes(self):
        return {a.axisTag: (a.minValue, a.defaultValue, a.maxValue) for a in self.axes}


class Axis(object):
    def __init__(self):
        self.axisTag = None
        self.axisNameID = 0
        self.flags = 0
        self.minValue = -1.0
        self.defaultValue = 0.0
        self.maxValue = 1.0

    def compile(self):
        return sstruct.pack(FVAR_AXIS_FORMAT, self)

    def decompile(self, data):
        sstruct.unpack2(FVAR_AXIS_FORMAT, data, self)

    def toXML(self, writer, ttFont):
        name = (
            ttFont["name"].getDebugName(self.axisNameID) if "name" in ttFont else None
        )
        if name is not None:
            writer.newline()
            writer.comment(name)
            writer.newline()
        writer.begintag("Axis")
        writer.newline()
        for tag, value in [
            ("AxisTag", self.axisTag),
            ("Flags", "0x%X" % self.flags),
            ("MinValue", fl2str(self.minValue, 16)),
            ("DefaultValue", fl2str(self.defaultValue, 16)),
            ("MaxValue", fl2str(self.maxValue, 16)),
            ("AxisNameID", str(self.axisNameID)),
        ]:
            writer.begintag(tag)
            writer.write(value)
            writer.endtag(tag)
            writer.newline()
        writer.endtag("Axis")
        writer.newline()

    def fromXML(self, name, _attrs, content, ttFont):
        assert name == "Axis"
        for tag, _, value in filter(lambda t: type(t) is tuple, content):
            value = "".join(value)
            if tag == "AxisTag":
                self.axisTag = Tag(value)
            elif tag in {"Flags", "MinValue", "DefaultValue", "MaxValue", "AxisNameID"}:
                setattr(
                    self,
                    tag[0].lower() + tag[1:],
                    str2fl(value, 16) if tag.endswith("Value") else safeEval(value),
                )


class NamedInstance(object):
    def __init__(self):
        self.subfamilyNameID = 0
        self.postscriptNameID = 0xFFFF
        self.flags = 0
        self.coordinates = {}

    def compile(self, axisTags, includePostScriptName):
        result = [sstruct.pack(FVAR_INSTANCE_FORMAT, self)]
        for axis in axisTags:
            fixedCoord = fl2fi(self.coordinates[axis], 16)
            result.append(struct.pack(">l", fixedCoord))
        if includePostScriptName:
            result.append(struct.pack(">H", self.postscriptNameID))
        return bytesjoin(result)

    def decompile(self, data, axisTags):
        sstruct.unpack2(FVAR_INSTANCE_FORMAT, data, self)
        pos = sstruct.calcsize(FVAR_INSTANCE_FORMAT)
        for axis in axisTags:
            value = struct.unpack(">l", data[pos : pos + 4])[0]
            self.coordinates[axis] = fi2fl(value, 16)
            pos += 4
        if pos + 2 <= len(data):
            self.postscriptNameID = struct.unpack(">H", data[pos : pos + 2])[0]
        else:
            self.postscriptNameID = 0xFFFF

    def toXML(self, writer, ttFont):
        name = (
            ttFont["name"].getDebugName(self.subfamilyNameID)
            if "name" in ttFont
            else None
        )
        if name is not None:
            writer.newline()
            writer.comment(name)
            writer.newline()
        psname = (
            ttFont["name"].getDebugName(self.postscriptNameID)
            if "name" in ttFont
            else None
        )
        if psname is not None:
            writer.comment("PostScript: " + psname)
            writer.newline()
        if self.postscriptNameID == 0xFFFF:
            writer.begintag(
                "NamedInstance",
                flags=("0x%X" % self.flags),
                subfamilyNameID=self.subfamilyNameID,
            )
        else:
            writer.begintag(
                "NamedInstance",
                flags=("0x%X" % self.flags),
                subfamilyNameID=self.subfamilyNameID,
                postscriptNameID=self.postscriptNameID,
            )
        writer.newline()
        for axis in ttFont["fvar"].axes:
            writer.simpletag(
                "coord",
                axis=axis.axisTag,
                value=fl2str(self.coordinates[axis.axisTag], 16),
            )
            writer.newline()
        writer.endtag("NamedInstance")
        writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        assert name == "NamedInstance"
        self.subfamilyNameID = safeEval(attrs["subfamilyNameID"])
        self.flags = safeEval(attrs.get("flags", "0"))
        if "postscriptNameID" in attrs:
            self.postscriptNameID = safeEval(attrs["postscriptNameID"])
        else:
            self.postscriptNameID = 0xFFFF

        for tag, elementAttrs, _ in filter(lambda t: type(t) is tuple, content):
            if tag == "coord":
                value = str2fl(elementAttrs["value"], 16)
                self.coordinates[elementAttrs["axis"]] = value
<|endoftext|>from fontTools import ttLib
from fontTools.ttLib.standardGlyphOrder import standardGlyphOrder
from fontTools.misc import sstruct
from fontTools.misc.textTools import bytechr, byteord, tobytes, tostr, safeEval, readHex
from . import DefaultTable
import sys
import struct
import array
import logging

log = logging.getLogger(__name__)

postFormat = """
	>
	formatType:			16.16F
	italicAngle:		16.16F		# italic angle in degrees
	underlinePosition:	h
	underlineThickness:	h
	isFixedPitch:		L
	minMemType42:		L			# minimum memory if TrueType font is downloaded
	maxMemType42:		L			# maximum memory if TrueType font is downloaded
	minMemType1:		L			# minimum memory if Type1 font is downloaded
	maxMemType1:		L			# maximum memory if Type1 font is downloaded
"""

postFormatSize = sstruct.calcsize(postFormat)


class table__p_o_s_t(DefaultTable.DefaultTable):
    """PostScript table

    The ``post`` table contains information needed to use the font on
    PostScript printers, including the PostScript names of glyphs and
    data that was stored in the ``FontInfo`` dictionary for Type 1 fonts.

    See also https://learn.microsoft.com/en-us/typography/opentype/spec/post
    """

    def decompile(self, data, ttFont):
        sstruct.unpack(postFormat, data[:postFormatSize], self)
        data = data[postFormatSize:]
        if self.formatType == 1.0:
            self.decode_format_1_0(data, ttFont)
        elif self.formatType == 2.0:
            self.decode_format_2_0(data, ttFont)
        elif self.formatType == 3.0:
            self.decode_format_3_0(data, ttFont)
        elif self.formatType == 4.0:
            self.decode_format_4_0(data, ttFont)
        else:
            # supported format
            raise ttLib.TTLibError(
                "'post' table format %f not supported" % self.formatType
            )

    def compile(self, ttFont):
        data = sstruct.pack(postFormat, self)
        if self.formatType == 1.0:
            pass  # we're done
        elif self.formatType == 2.0:
            data = data + self.encode_format_2_0(ttFont)
        elif self.formatType == 3.0:
            pass  # we're done
        elif self.formatType == 4.0:
            data = data + self.encode_format_4_0(ttFont)
        else:
            # supported format
            raise ttLib.TTLibError(
                "'post' table format %f not supported" % self.formatType
            )
        return data

    def getGlyphOrder(self):
        """This function will get called by a ttLib.TTFont instance.
        Do not call this function yourself, use TTFont().getGlyphOrder()
        or its relatives instead!
        """
        if not hasattr(self, "glyphOrder"):
            raise ttLib.TTLibError("illegal use of getGlyphOrder()")
        glyphOrder = self.glyphOrder
        del self.glyphOrder
        return glyphOrder

    def decode_format_1_0(self, data, ttFont):
        self.glyphOrder = standardGlyphOrder[: ttFont["maxp"].numGlyphs]

    def decode_format_2_0(self, data, ttFont):
        (numGlyphs,) = struct.unpack(">H", data[:2])
        numGlyphs = int(numGlyphs)
        if numGlyphs > ttFont["maxp"].numGlyphs:
            # Assume the numGlyphs field is bogus, so sync with maxp.
            # I've seen this in one font, and if the assumption is
            # wrong elsewhere, well, so be it: it's hard enough to
            # work around _one_ non-conforming post format...
            numGlyphs = ttFont["maxp"].numGlyphs
        data = data[2:]
        indices = array.array("H")
        indices.frombytes(data[: 2 * numGlyphs])
        if sys.byteorder != "big":
            indices.byteswap()
        data = data[2 * numGlyphs :]
        maxIndex = max(indices)
        self.extraNames = extraNames = unpackPStrings(data, maxIndex - 257)
        self.glyphOrder = glyphOrder = [""] * int(ttFont["maxp"].numGlyphs)
        for glyphID in range(numGlyphs):
            index = indices[glyphID]
            if index > 257:
                try:
                    name = extraNames[index - 258]
                except IndexError:
                    name = ""
            else:
                # fetch names from standard list
                name = standardGlyphOrder[index]
            glyphOrder[glyphID] = name
        self.build_psNameMapping(ttFont)

    def build_psNameMapping(self, ttFont):
        mapping = {}
        allNames = {}
        glyphOrderNames = set(self.glyphOrder)
        for i in range(ttFont["maxp"].numGlyphs):
            glyphName = psName = self.glyphOrder[i]
            if glyphName == "":
                glyphName = "glyph%.5d" % i

            if glyphName in allNames:
                # make up a new glyphName that's unique
                n = allNames[glyphName]
                # check if the glyph name exists in the glyph order
                while f"{glyphName}.{n}" in glyphOrderNames:
                    n += 1
                allNames[glyphName] = n + 1
                glyphName = f"{glyphName}.{n}"

            allNames[glyphName] = 1
            if glyphName != psName:
                self.glyphOrder[i] = glyphName
                mapping[glyphName] = psName

        self.mapping = mapping

    def decode_format_3_0(self, data, ttFont):
        # Setting self.glyphOrder to None will cause the TTFont object
        # try and construct glyph names from a Unicode cmap table.
        self.glyphOrder = None

    def decode_format_4_0(self, data, ttFont):
        from fontTools import agl

        numGlyphs = ttFont["maxp"].numGlyphs
        indices = array.array("H")
        indices.frombytes(data)
        if sys.byteorder != "big":
            indices.byteswap()
        # In some older fonts, the size of the post table doesn't match
        # the number of glyphs. Sometimes it's bigger, sometimes smaller.
        self.glyphOrder = glyphOrder = [""] * int(numGlyphs)
        for i in range(min(len(indices), numGlyphs)):
            if indices[i] == 0xFFFF:
                self.glyphOrder[i] = ""
            elif indices[i] in agl.UV2AGL:
                self.glyphOrder[i] = agl.UV2AGL[indices[i]]
            else:
                self.glyphOrder[i] = "uni%04X" % indices[i]
        self.build_psNameMapping(ttFont)

    def encode_format_2_0(self, ttFont):
        numGlyphs = ttFont["maxp"].numGlyphs
        glyphOrder = ttFont.getGlyphOrder()
        assert len(glyphOrder) == numGlyphs
        indices = array.array("H")
        extraDict = {}
        extraNames = self.extraNames = [
            n for n in self.extraNames if n not in standardGlyphOrder
        ]
        for i, name in enumerate(extraNames):
            extraDict[name] = i
        for glyphName in glyphOrder:
            if glyphName in self.mapping:
                psName = self.mapping[glyphName]
            else:
                psName = glyphName
            if psName in extraDict:
                index = 258 + extraDict[psName]
            elif psName in standardGlyphOrder:
                index = standardGlyphOrder.index(psName)
            else:
                index = 258 + len(extraNames)
                extraDict[psName] = len(extraNames)
                extraNames.append(psName)
            indices.append(index)
        if sys.byteorder != "big":
            indices.byteswap()
        return (
            struct.pack(">H", numGlyphs) + indices.tobytes() + packPStrings(extraNames)
        )

    def encode_format_4_0(self, ttFont):
        from fontTools import agl

        numGlyphs = ttFont["maxp"].numGlyphs
        glyphOrder = ttFont.getGlyphOrder()
        assert len(glyphOrder) == numGlyphs
        indices = array.array("H")
        for glyphID in glyphOrder:
            glyphID = glyphID.split("#")[0]
            if glyphID in agl.AGL2UV:
                indices.append(agl.AGL2UV[glyphID])
            elif len(glyphID) == 7 and glyphID[:3] == "uni":
                indices.append(int(glyphID[3:], 16))
            else:
                indices.append(0xFFFF)
        if sys.byteorder != "big":
            indices.byteswap()
        return indices.tobytes()

    def toXML(self, writer, ttFont):
        formatstring, names, fixes = sstruct.getformat(postFormat)
        for name in names:
            value = getattr(self, name)
            writer.simpletag(name, value=value)
            writer.newline()
        if hasattr(self, "mapping"):
            writer.begintag("psNames")
            writer.newline()
            writer.comment(
                "This file uses unique glyph names based on the information\n"
                "found in the 'post' table. Since these names might not be unique,\n"
                "we have to invent artificial names in case of clashes. In order to\n"
                "be able to retain the original information, we need a name to\n"
                "ps name mapping for those cases where they differ. That's what\n"
                "you see below.\n"
            )
            writer.newline()
            items = sorted(self.mapping.items())
            for name, psName in items:
                writer.simpletag("psName", name=name, psName=psName)
                writer.newline()
            writer.endtag("psNames")
            writer.newline()
        if hasattr(self, "extraNames"):
            writer.begintag("extraNames")
            writer.newline()
            writer.comment(
                "following are the name that are not taken from the standard Mac glyph order"
            )
            writer.newline()
            for name in self.extraNames:
                writer.simpletag("psName", name=name)
                writer.newline()
            writer.endtag("extraNames")
            writer.newline()
        if hasattr(self, "data"):
            writer.begintag("hexdata")
            writer.newline()
            writer.dumphex(self.data)
            writer.endtag("hexdata")
            writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        if name not in ("psNames", "extraNames", "hexdata"):
            setattr(self, name, safeEval(attrs["value"]))
        elif name == "psNames":
            self.mapping = {}
            for element in content:
                if not isinstance(element, tuple):
                    continue
                name, attrs, content = element
                if name == "psName":
                    self.mapping[attrs["name"]] = attrs["psName"]
        elif name == "extraNames":
            self.extraNames = []
            for element in content:
                if not isinstance(element, tuple):
                    continue
                name, attrs, content = element
                if name == "psName":
                    self.extraNames.append(attrs["name"])
        else:
            self.data = readHex(content)


def unpackPStrings(data, n):
    # extract n Pascal strings from data.
    # if there is not enough data, use ""

    strings = []
    index = 0
    dataLen = len(data)

    for _ in range(n):
        if dataLen <= index:
            length = 0
        else:
            length = byteord(data[index])
        index += 1

        if dataLen <= index + length - 1:
            name = ""
        else:
            name = tostr(data[index : index + length], encoding="latin1")
        strings.append(name)
        index += length

    if index < dataLen:
        log.warning("%d extra bytes in post.stringData array", dataLen - index)

    elif dataLen < index:
        log.warning("not enough data in post.stringData array")

    return strings


def packPStrings(strings):
    data = b""
    for s in strings:
        data = data + bytechr(len(s)) + tobytes(s, encoding="latin1")
    return data
<|endoftext|>"""Methods for traversing trees of otData-driven OpenType tables."""

from collections import deque
from typing import Callable, Deque, Iterable, List, Optional, Tuple
from .otBase import BaseTable

__all__ = [
    "bfs_base_table",
    "dfs_base_table",
    "SubTablePath",
]


class SubTablePath(Tuple[BaseTable.SubTableEntry, ...]):
    def __str__(self) -> str:
        path_parts = []
        for entry in self:
            path_part = entry.name
            if entry.index is not None:
                path_part += f"[{entry.index}]"
            path_parts.append(path_part)
        return ".".join(path_parts)


# Given f(current frontier, new entries) add new entries to frontier
AddToFrontierFn = Callable[[Deque[SubTablePath], List[SubTablePath]], None]


def dfs_base_table(
    root: BaseTable,
    root_accessor: Optional[str] = None,
    skip_root: bool = False,
    predicate: Optional[Callable[[SubTablePath], bool]] = None,
    iter_subtables_fn: Optional[
        Callable[[BaseTable], Iterable[BaseTable.SubTableEntry]]
    ] = None,
) -> Iterable[SubTablePath]:
    """Depth-first search tree of BaseTables.

    Args:
        root (BaseTable): the root of the tree.
        root_accessor (Optional[str]): attribute name for the root table, if any (mostly
            useful for debugging).
        skip_root (Optional[bool]): if True, the root itself is not visited, only its
            children.
        predicate (Optional[Callable[[SubTablePath], bool]]): function to filter out
            paths. If True, the path is yielded and its subtables are added to the
            queue. If False, the path is skipped and its subtables are not traversed.
        iter_subtables_fn (Optional[Callable[[BaseTable], Iterable[BaseTable.SubTableEntry]]]):
            function to iterate over subtables of a table. If None, the default
            BaseTable.iterSubTables() is used.

    Yields:
        SubTablePath: tuples of BaseTable.SubTableEntry(name, table, index) namedtuples
        for each of the nodes in the tree. The last entry in a path is the current
        subtable, whereas preceding ones refer to its parent tables all the way up to
        the root.
    """
    yield from _traverse_ot_data(
        root,
        root_accessor,
        skip_root,
        predicate,
        lambda frontier, new: frontier.extendleft(reversed(new)),
        iter_subtables_fn,
    )


def bfs_base_table(
    root: BaseTable,
    root_accessor: Optional[str] = None,
    skip_root: bool = False,
    predicate: Optional[Callable[[SubTablePath], bool]] = None,
    iter_subtables_fn: Optional[
        Callable[[BaseTable], Iterable[BaseTable.SubTableEntry]]
    ] = None,
) -> Iterable[SubTablePath]:
    """Breadth-first search tree of BaseTables.

    Args:
        root
            the root of the tree.
        root_accessor (Optional[str]): attribute name for the root table, if any (mostly
            useful for debugging).
        skip_root (Optional[bool]): if True, the root itself is not visited, only its
            children.
        predicate (Optional[Callable[[SubTablePath], bool]]): function to filter out
            paths. If True, the path is yielded and its subtables are added to the
            queue. If False, the path is skipped and its subtables are not traversed.
        iter_subtables_fn (Optional[Callable[[BaseTable], Iterable[BaseTable.SubTableEntry]]]):
            function to iterate over subtables of a table. If None, the default
            BaseTable.iterSubTables() is used.

    Yields:
        SubTablePath: tuples of BaseTable.SubTableEntry(name, table, index) namedtuples
        for each of the nodes in the tree. The last entry in a path is the current
        subtable, whereas preceding ones refer to its parent tables all the way up to
        the root.
    """
    yield from _traverse_ot_data(
        root,
        root_accessor,
        skip_root,
        predicate,
        lambda frontier, new: frontier.extend(new),
        iter_subtables_fn,
    )


def _traverse_ot_data(
    root: BaseTable,
    root_accessor: Optional[str],
    skip_root: bool,
    predicate: Optional[Callable[[SubTablePath], bool]],
    add_to_frontier_fn: AddToFrontierFn,
    iter_subtables_fn: Optional[
        Callable[[BaseTable], Iterable[BaseTable.SubTableEntry]]
    ] = None,
) -> Iterable[SubTablePath]:
    # no visited because general otData cannot cycle (forward-offset only)
    if root_accessor is None:
        root_accessor = type(root).__name__

    if predicate is None:

        def predicate(path):
            return True

    if iter_subtables_fn is None:

        def iter_subtables_fn(table):
            return table.iterSubTables()

    frontier: Deque[SubTablePath] = deque()

    root_entry = BaseTable.SubTableEntry(root_accessor, root)
    if not skip_root:
        frontier.append((root_entry,))
    else:
        add_to_frontier_fn(
            frontier,
            [
                (root_entry, subtable_entry)
                for subtable_entry in iter_subtables_fn(root)
            ],
        )

    while frontier:
        # path is (value, attr_name) tuples. attr_name is attr of parent to get value
        path = frontier.popleft()
        current = path[-1].value

        if not predicate(path):
            continue

        yield SubTablePath(path)

        new_entries = [
            path + (subtable_entry,) for subtable_entry in iter_subtables_fn(current)
        ]

        add_to_frontier_fn(frontier, new_entries)
<|endoftext|>from fontTools.misc import sstruct
from fontTools.misc.textTools import safeEval
from fontTools.misc.timeTools import timestampFromString, timestampToString
from . import DefaultTable

FFTMFormat = """
		>	# big endian
		version:        I
		FFTimeStamp:    Q
		sourceCreated:  Q
		sourceModified: Q
"""


class table_F_F_T_M_(DefaultTable.DefaultTable):
    """FontForge Time Stamp table

    The ``FFTM`` table is used by the free-software font editor
    FontForge to record timestamps for the creation and modification
    of font source (.sfd) files and a timestamp for FontForge's
    own source code.

    See also https://fontforge.org/docs/techref/non-standard.html
    """

    def decompile(self, data, ttFont):
        dummy, rest = sstruct.unpack2(FFTMFormat, data, self)

    def compile(self, ttFont):
        data = sstruct.pack(FFTMFormat, self)
        return data

    def toXML(self, writer, ttFont):
        writer.comment(
            "FontForge's timestamp, font source creation and modification dates"
        )
        writer.newline()
        formatstring, names, fixes = sstruct.getformat(FFTMFormat)
        for name in names:
            value = getattr(self, name)
            if name in ("FFTimeStamp", "sourceCreated", "sourceModified"):
                value = timestampToString(value)
            writer.simpletag(name, value=value)
            writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        value = attrs["value"]
        if name in ("FFTimeStamp", "sourceCreated", "sourceModified"):
            value = timestampFromString(value)
        else:
            value = safeEval(value)
        setattr(self, name, value)
<|endoftext|># Copyright 2013 Google, Inc. All Rights Reserved.
#
# Google Author(s): Behdad Esfahbod

from fontTools.misc.textTools import safeEval
from . import DefaultTable


class table_C_O_L_R_(DefaultTable.DefaultTable):
    """Color table

    The ``COLR`` table defines color presentation of outline glyphs. It must
    be used in concert with the ``CPAL`` table, which contains the color
    descriptors used.

    This table is structured so that you can treat it like a dictionary keyed by glyph name.

    ``ttFont['COLR'][<glyphName>]`` will return the color layers for any glyph.

    ``ttFont['COLR'][<glyphName>] = <value>`` will set the color layers for any glyph.

    See also https://learn.microsoft.com/en-us/typography/opentype/spec/colr
    """

    @staticmethod
    def _decompileColorLayersV0(table):
        if not table.LayerRecordArray:
            return {}
        colorLayerLists = {}
        layerRecords = table.LayerRecordArray.LayerRecord
        numLayerRecords = len(layerRecords)
        for baseRec in table.BaseGlyphRecordArray.BaseGlyphRecord:
            baseGlyph = baseRec.BaseGlyph
            firstLayerIndex = baseRec.FirstLayerIndex
            numLayers = baseRec.NumLayers
            assert firstLayerIndex + numLayers <= numLayerRecords
            layers = []
            for i in range(firstLayerIndex, firstLayerIndex + numLayers):
                layerRec = layerRecords[i]
                layers.append(LayerRecord(layerRec.LayerGlyph, layerRec.PaletteIndex))
            colorLayerLists[baseGlyph] = layers
        return colorLayerLists

    def _toOTTable(self, ttFont):
        from . import otTables
        from fontTools.colorLib.builder import populateCOLRv0

        tableClass = getattr(otTables, self.tableTag)
        table = tableClass()
        table.Version = self.version

        populateCOLRv0(
            table,
            {
                baseGlyph: [(layer.name, layer.colorID) for layer in layers]
                for baseGlyph, layers in self.ColorLayers.items()
            },
            glyphMap=ttFont.getReverseGlyphMap(rebuild=True),
        )
        return table

    def decompile(self, data, ttFont):
        from .otBase import OTTableReader
        from . import otTables

        # We use otData to decompile, but we adapt the decompiled otTables to the
        # existing COLR v0 API for backward compatibility.
        reader = OTTableReader(data, tableTag=self.tableTag)
        tableClass = getattr(otTables, self.tableTag)
        table = tableClass()
        table.decompile(reader, ttFont)

        self.version = table.Version
        if self.version == 0:
            self.ColorLayers = self._decompileColorLayersV0(table)
        else:
            # for new versions, keep the raw otTables around
            self.table = table

    def compile(self, ttFont):
        from .otBase import OTTableWriter

        if hasattr(self, "table"):
            table = self.table
        else:
            table = self._toOTTable(ttFont)

        writer = OTTableWriter(tableTag=self.tableTag)
        table.compile(writer, ttFont)
        return writer.getAllData()

    def toXML(self, writer, ttFont):
        if hasattr(self, "table"):
            self.table.toXML2(writer, ttFont)
        else:
            writer.simpletag("version", value=self.version)
            writer.newline()
            for baseGlyph in sorted(self.ColorLayers.keys(), key=ttFont.getGlyphID):
                writer.begintag("ColorGlyph", name=baseGlyph)
                writer.newline()
                for layer in self.ColorLayers[baseGlyph]:
                    layer.toXML(writer, ttFont)
                writer.endtag("ColorGlyph")
                writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        if name == "version":  # old COLR v0 API
            setattr(self, name, safeEval(attrs["value"]))
        elif name == "ColorGlyph":
            if not hasattr(self, "ColorLayers"):
                self.ColorLayers = {}
            glyphName = attrs["name"]
            for element in content:
                if isinstance(element, str):
                    continue
            layers = []
            for element in content:
                if isinstance(element, str):
                    continue
                layer = LayerRecord()
                layer.fromXML(element[0], element[1], element[2], ttFont)
                layers.append(layer)
            self.ColorLayers[glyphName] = layers
        else:  # new COLR v1 API
            from . import otTables

            if not hasattr(self, "table"):
                tableClass = getattr(otTables, self.tableTag)
                self.table = tableClass()
            self.table.fromXML(name, attrs, content, ttFont)
            self.table.populateDefaults()
            self.version = self.table.Version

    def __getitem__(self, glyphName):
        if not isinstance(glyphName, str):
            raise TypeError(f"expected str, found {type(glyphName).__name__}")
        return self.ColorLayers[glyphName]

    def __setitem__(self, glyphName, value):
        if not isinstance(glyphName, str):
            raise TypeError(f"expected str, found {type(glyphName).__name__}")
        if value is not None:
            self.ColorLayers[glyphName] = value
        elif glyphName in self.ColorLayers:
            del self.ColorLayers[glyphName]

    def __delitem__(self, glyphName):
        del self.ColorLayers[glyphName]


class LayerRecord(object):
    def __init__(self, name=None, colorID=None):
        self.name = name
        self.colorID = colorID

    def toXML(self, writer, ttFont):
        writer.simpletag("layer", name=self.name, colorID=self.colorID)
        writer.newline()

    def fromXML(self, eltname, attrs, content, ttFont):
        for name, value in attrs.items():
            if name == "name":
                setattr(self, name, value)
            else:
                setattr(self, name, safeEval(value))
<|endoftext|>from fontTools.misc.roundTools import otRound
from fontTools import ttLib
from fontTools.misc.textTools import safeEval
from . import DefaultTable
import sys
import struct
import array
import logging

log = logging.getLogger(__name__)


class table__h_m_t_x(DefaultTable.DefaultTable):
    """Horizontal Metrics table

    The ``hmtx`` table contains per-glyph metrics for the glyphs in a
    ``glyf``, ``CFF ``, or ``CFF2`` table, as needed for horizontal text
    layout.

    See also https://learn.microsoft.com/en-us/typography/opentype/spec/hmtx
    """

    headerTag = "hhea"
    advanceName = "width"
    sideBearingName = "lsb"
    numberOfMetricsName = "numberOfHMetrics"
    longMetricFormat = "Hh"

    def decompile(self, data, ttFont):
        numGlyphs = ttFont["maxp"].numGlyphs
        headerTable = ttFont.get(self.headerTag)
        if headerTable is not None:
            numberOfMetrics = int(getattr(headerTable, self.numberOfMetricsName))
        else:
            numberOfMetrics = numGlyphs
        if numberOfMetrics > numGlyphs:
            log.warning(
                "The %s.%s exceeds the maxp.numGlyphs"
                % (self.headerTag, self.numberOfMetricsName)
            )
            numberOfMetrics = numGlyphs
        numberOfSideBearings = numGlyphs - numberOfMetrics
        tableSize = 4 * numberOfMetrics + 2 * numberOfSideBearings
        if len(data) < tableSize:
            raise ttLib.TTLibError(
                f"not enough '{self.tableTag}' table data: "
                f"expected {tableSize} bytes, got {len(data)}"
            )
        # Note: advanceWidth is unsigned, but some font editors might
        # read/write as signed. We can't be sure whether it was a mistake
        # or not, so we read as unsigned but also issue a warning...
        metricsFmt = ">" + self.longMetricFormat * numberOfMetrics
        metrics = struct.unpack(metricsFmt, data[: 4 * numberOfMetrics])
        data = data[4 * numberOfMetrics :]
        sideBearings = array.array("h", data[: 2 * numberOfSideBearings])
        data = data[2 * numberOfSideBearings :]

        if sys.byteorder != "big":
            sideBearings.byteswap()
        if data:
            log.warning("too much '%s' table data" % self.tableTag)
        self.metrics = {}
        glyphOrder = ttFont.getGlyphOrder()
        for i in range(numberOfMetrics):
            glyphName = glyphOrder[i]
            advanceWidth, lsb = metrics[i * 2 : i * 2 + 2]
            if advanceWidth > 32767:
                log.warning(
                    "Glyph %r has a huge advance %s (%d); is it intentional or "
                    "an (invalid) negative value?",
                    glyphName,
                    self.advanceName,
                    advanceWidth,
                )
            self.metrics[glyphName] = (advanceWidth, lsb)
        lastAdvance = metrics[-2]
        for i in range(numberOfSideBearings):
            glyphName = glyphOrder[i + numberOfMetrics]
            self.metrics[glyphName] = (lastAdvance, sideBearings[i])

    def compile(self, ttFont):
        metrics = []
        hasNegativeAdvances = False
        for glyphName in ttFont.getGlyphOrder():
            advanceWidth, sideBearing = self.metrics[glyphName]
            if advanceWidth < 0:
                log.error(
                    "Glyph %r has negative advance %s" % (glyphName, self.advanceName)
                )
                hasNegativeAdvances = True
            metrics.append([advanceWidth, sideBearing])

        headerTable = ttFont.get(self.headerTag)
        if headerTable is not None:
            lastAdvance = metrics[-1][0]
            lastIndex = len(metrics)
            while metrics[lastIndex - 2][0] == lastAdvance:
                lastIndex -= 1
                if lastIndex <= 1:
                    # all advances are equal
                    lastIndex = 1
                    break
            additionalMetrics = metrics[lastIndex:]
            additionalMetrics = [otRound(sb) for _, sb in additionalMetrics]
            metrics = metrics[:lastIndex]
            numberOfMetrics = len(metrics)
            setattr(headerTable, self.numberOfMetricsName, numberOfMetrics)
        else:
            # no hhea/vhea, can't store numberOfMetrics; assume == numGlyphs
            numberOfMetrics = ttFont["maxp"].numGlyphs
            additionalMetrics = []

        allMetrics = []
        for advance, sb in metrics:
            allMetrics.extend([otRound(advance), otRound(sb)])
        metricsFmt = ">" + self.longMetricFormat * numberOfMetrics
        try:
            data = struct.pack(metricsFmt, *allMetrics)
        except struct.error as e:
            if "out of range" in str(e) and hasNegativeAdvances:
                raise ttLib.TTLibError(
                    "'%s' table can't contain negative advance %ss"
                    % (self.tableTag, self.advanceName)
                )
            else:
                raise
        additionalMetrics = array.array("h", additionalMetrics)
        if sys.byteorder != "big":
            additionalMetrics.byteswap()
        data = data + additionalMetrics.tobytes()
        return data

    def toXML(self, writer, ttFont):
        names = sorted(self.metrics.keys())
        for glyphName in names:
            advance, sb = self.metrics[glyphName]
            writer.simpletag(
                "mtx",
                [
                    ("name", glyphName),
                    (self.advanceName, advance),
                    (self.sideBearingName, sb),
                ],
            )
            writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        if not hasattr(self, "metrics"):
            self.metrics = {}
        if name == "mtx":
            self.metrics[attrs["name"]] = (
                safeEval(attrs[self.advanceName]),
                safeEval(attrs[self.sideBearingName]),
            )

    def __delitem__(self, glyphName):
        del self.metrics[glyphName]

    def __getitem__(self, glyphName):
        return self.metrics[glyphName]

    def __setitem__(self, glyphName, advance_sb_pair):
        self.metrics[glyphName] = tuple(advance_sb_pair)
<|endoftext|>from fontTools.misc import sstruct
from . import DefaultTable
from fontTools.misc.textTools import bytesjoin, safeEval
from .BitmapGlyphMetrics import (
    BigGlyphMetrics,
    bigGlyphMetricsFormat,
    SmallGlyphMetrics,
    smallGlyphMetricsFormat,
)
import struct
import itertools
from collections import deque
import logging

log = logging.getLogger(__name__)

eblcHeaderFormat = """
	> # big endian
	version:  16.16F
	numSizes: I
"""
# The table format string is split to handle sbitLineMetrics simply.
bitmapSizeTableFormatPart1 = """
	> # big endian
	indexSubTableArrayOffset: I
	indexTablesSize:          I
	numberOfIndexSubTables:   I
	colorRef:                 I
"""
# The compound type for hori and vert.
sbitLineMetricsFormat = """
	> # big endian
	ascender:              b
	descender:             b
	widthMax:              B
	caretSlopeNumerator:   b
	caretSlopeDenominator: b
	caretOffset:           b
	minOriginSB:           b
	minAdvanceSB:          b
	maxBeforeBL:           b
	minAfterBL:            b
	pad1:                  b
	pad2:                  b
"""
# hori and vert go between the two parts.
bitmapSizeTableFormatPart2 = """
	> # big endian
	startGlyphIndex: H
	endGlyphIndex:   H
	ppemX:           B
	ppemY:           B
	bitDepth:        B
	flags:           b
"""

indexSubTableArrayFormat = ">HHL"
indexSubTableArraySize = struct.calcsize(indexSubTableArrayFormat)

indexSubHeaderFormat = ">HHL"
indexSubHeaderSize = struct.calcsize(indexSubHeaderFormat)

codeOffsetPairFormat = ">HH"
codeOffsetPairSize = struct.calcsize(codeOffsetPairFormat)


class table_E_B_L_C_(DefaultTable.DefaultTable):
    """Embedded Bitmap Location table

    The ``EBLC`` table contains the locations of monochrome or grayscale
    bitmaps for glyphs. It must be used in concert with the ``EBDT`` table.

    See also https://learn.microsoft.com/en-us/typography/opentype/spec/eblc
    """

    dependencies = ["EBDT"]

    # This method can be overridden in subclasses to support new formats
    # without changing the other implementation. Also can be used as a
    # convenience method for coverting a font file to an alternative format.
    def getIndexFormatClass(self, indexFormat):
        return eblc_sub_table_classes[indexFormat]

    def decompile(self, data, ttFont):
        # Save the original data because offsets are from the start of the table.
        origData = data
        i = 0

        dummy = sstruct.unpack(eblcHeaderFormat, data[:8], self)
        i += 8

        self.strikes = []
        for curStrikeIndex in range(self.numSizes):
            curStrike = Strike()
            self.strikes.append(curStrike)
            curTable = curStrike.bitmapSizeTable
            dummy = sstruct.unpack2(
                bitmapSizeTableFormatPart1, data[i : i + 16], curTable
            )
            i += 16
            for metric in ("hori", "vert"):
                metricObj = SbitLineMetrics()
                vars(curTable)[metric] = metricObj
                dummy = sstruct.unpack2(
                    sbitLineMetricsFormat, data[i : i + 12], metricObj
                )
                i += 12
            dummy = sstruct.unpack(
                bitmapSizeTableFormatPart2, data[i : i + 8], curTable
            )
            i += 8

        for curStrike in self.strikes:
            curTable = curStrike.bitmapSizeTable
            for subtableIndex in range(curTable.numberOfIndexSubTables):
                i = (
                    curTable.indexSubTableArrayOffset
                    + subtableIndex * indexSubTableArraySize
                )

                tup = struct.unpack(
                    indexSubTableArrayFormat, data[i : i + indexSubTableArraySize]
                )
                firstGlyphIndex, lastGlyphIndex, additionalOffsetToIndexSubtable = tup
                i = curTable.indexSubTableArrayOffset + additionalOffsetToIndexSubtable

                tup = struct.unpack(
                    indexSubHeaderFormat, data[i : i + indexSubHeaderSize]
                )
                indexFormat, imageFormat, imageDataOffset = tup

                indexFormatClass = self.getIndexFormatClass(indexFormat)
                indexSubTable = indexFormatClass(data[i + indexSubHeaderSize :], ttFont)
                indexSubTable.firstGlyphIndex = firstGlyphIndex
                indexSubTable.lastGlyphIndex = lastGlyphIndex
                indexSubTable.additionalOffsetToIndexSubtable = (
                    additionalOffsetToIndexSubtable
                )
                indexSubTable.indexFormat = indexFormat
                indexSubTable.imageFormat = imageFormat
                indexSubTable.imageDataOffset = imageDataOffset
                indexSubTable.decompile()  # https://github.com/fonttools/fonttools/issues/317
                curStrike.indexSubTables.append(indexSubTable)

    def compile(self, ttFont):
        dataList = []
        self.numSizes = len(self.strikes)
        dataList.append(sstruct.pack(eblcHeaderFormat, self))

        # Data size of the header + bitmapSizeTable needs to be calculated
        # in order to form offsets. This value will hold the size of the data
        # in dataList after all the data is consolidated in dataList.
        dataSize = len(dataList[0])

        # The table will be structured in the following order:
        # (0) header
        # (1) Each bitmapSizeTable [1 ... self.numSizes]
        # (2) Alternate between indexSubTableArray and indexSubTable
        #     for each bitmapSizeTable present.
        #
        # The issue is maintaining the proper offsets when table information
        # gets moved around. All offsets and size information must be recalculated
        # when building the table to allow editing within ttLib and also allow easy
        # import/export to and from XML. All of this offset information is lost
        # when exporting to XML so everything must be calculated fresh so importing
        # from XML will work cleanly. Only byte offset and size information is
        # calculated fresh. Count information like numberOfIndexSubTables is
        # checked through assertions. If the information in this table was not
        # touched or was changed properly then these types of values should match.
        #
        # The table will be rebuilt the following way:
        # (0) Precompute the size of all the bitmapSizeTables. This is needed to
        #     compute the offsets properly.
        # (1) For each bitmapSizeTable compute the indexSubTable and
        #    	indexSubTableArray pair. The indexSubTable must be computed first
        #     so that the offset information in indexSubTableArray can be
        #     calculated. Update the data size after each pairing.
        # (2) Build each bitmapSizeTable.
        # (3) Consolidate all the data into the main dataList in the correct order.

        for _ in self.strikes:
            dataSize += sstruct.calcsize(bitmapSizeTableFormatPart1)
            dataSize += len(("hori", "vert")) * sstruct.calcsize(sbitLineMetricsFormat)
            dataSize += sstruct.calcsize(bitmapSizeTableFormatPart2)

        indexSubTablePairDataList = []
        for curStrike in self.strikes:
            curTable = curStrike.bitmapSizeTable
            curTable.numberOfIndexSubTables = len(curStrike.indexSubTables)
            curTable.indexSubTableArrayOffset = dataSize

            # Precompute the size of the indexSubTableArray. This information
            # is important for correctly calculating the new value for
            # additionalOffsetToIndexSubtable.
            sizeOfSubTableArray = (
                curTable.numberOfIndexSubTables * indexSubTableArraySize
            )
            lowerBound = dataSize
            dataSize += sizeOfSubTableArray
            upperBound = dataSize

            indexSubTableDataList = []
            for indexSubTable in curStrike.indexSubTables:
                indexSubTable.additionalOffsetToIndexSubtable = (
                    dataSize - curTable.indexSubTableArrayOffset
                )
                glyphIds = list(map(ttFont.getGlyphID, indexSubTable.names))
                indexSubTable.firstGlyphIndex = min(glyphIds)
                indexSubTable.lastGlyphIndex = max(glyphIds)
                data = indexSubTable.compile(ttFont)
                indexSubTableDataList.append(data)
                dataSize += len(data)
            curTable.startGlyphIndex = min(
                ist.firstGlyphIndex for ist in curStrike.indexSubTables
            )
            curTable.endGlyphIndex = max(
                ist.lastGlyphIndex for ist in curStrike.indexSubTables
            )

            for i in curStrike.indexSubTables:
                data = struct.pack(
                    indexSubHeaderFormat,
                    i.firstGlyphIndex,
                    i.lastGlyphIndex,
                    i.additionalOffsetToIndexSubtable,
                )
                indexSubTablePairDataList.append(data)
            indexSubTablePairDataList.extend(indexSubTableDataList)
            curTable.indexTablesSize = dataSize - curTable.indexSubTableArrayOffset

        for curStrike in self.strikes:
            curTable = curStrike.bitmapSizeTable
            data = sstruct.pack(bitmapSizeTableFormatPart1, curTable)
            dataList.append(data)
            for metric in ("hori", "vert"):
                metricObj = vars(curTable)[metric]
                data = sstruct.pack(sbitLineMetricsFormat, metricObj)
                dataList.append(data)
            data = sstruct.pack(bitmapSizeTableFormatPart2, curTable)
            dataList.append(data)
        dataList.extend(indexSubTablePairDataList)

        return bytesjoin(dataList)

    def toXML(self, writer, ttFont):
        writer.simpletag("header", [("version", self.version)])
        writer.newline()
        for curIndex, curStrike in enumerate(self.strikes):
            curStrike.toXML(curIndex, writer, ttFont)

    def fromXML(self, name, attrs, content, ttFont):
        if name == "header":
            self.version = safeEval(attrs["version"])
        elif name == "strike":
            if not hasattr(self, "strikes"):
                self.strikes = []
            strikeIndex = safeEval(attrs["index"])
            curStrike = Strike()
            curStrike.fromXML(name, attrs, content, ttFont, self)

            # Grow the strike array to the appropriate size. The XML format
            # allows for the strike index value to be out of order.
            if strikeIndex >= len(self.strikes):
                self.strikes += [None] * (strikeIndex + 1 - len(self.strikes))
            assert self.strikes[strikeIndex] is None, "Duplicate strike EBLC indices."
            self.strikes[strikeIndex] = curStrike


class Strike(object):
    def __init__(self):
        self.bitmapSizeTable = BitmapSizeTable()
        self.indexSubTables = []

    def toXML(self, strikeIndex, writer, ttFont):
        writer.begintag("strike", [("index", strikeIndex)])
        writer.newline()
        self.bitmapSizeTable.toXML(writer, ttFont)
        writer.comment(
            "GlyphIds are written but not read. The firstGlyphIndex and\nlastGlyphIndex values will be recalculated by the compiler."
        )
        writer.newline()
        for indexSubTable in self.indexSubTables:
            indexSubTable.toXML(writer, ttFont)
        writer.endtag("strike")
        writer.newline()

    def fromXML(self, name, attrs, content, ttFont, locator):
        for element in content:
            if not isinstance(element, tuple):
                continue
            name, attrs, content = element
            if name == "bitmapSizeTable":
                self.bitmapSizeTable.fromXML(name, attrs, content, ttFont)
            elif name.startswith(_indexSubTableSubclassPrefix):
                indexFormat = safeEval(name[len(_indexSubTableSubclassPrefix) :])
                indexFormatClass = locator.getIndexFormatClass(indexFormat)
                indexSubTable = indexFormatClass(None, None)
                indexSubTable.indexFormat = indexFormat
                indexSubTable.fromXML(name, attrs, content, ttFont)
                self.indexSubTables.append(indexSubTable)


class BitmapSizeTable(object):
    # Returns all the simple metric names that bitmap size table
    # cares about in terms of XML creation.
    def _getXMLMetricNames(self):
        dataNames = sstruct.getformat(bitmapSizeTableFormatPart1)[1]
        dataNames = {**dataNames, **sstruct.getformat(bitmapSizeTableFormatPart2)[1]}
        # Skip the first 3 data names because they are byte offsets and counts.
        return list(dataNames.keys())[3:]

    def toXML(self, writer, ttFont):
        writer.begintag("bitmapSizeTable")
        writer.newline()
        for metric in ("hori", "vert"):
            getattr(self, metric).toXML(metric, writer, ttFont)
        for metricName in self._getXMLMetricNames():
            writer.simpletag(metricName, value=getattr(self, metricName))
            writer.newline()
        writer.endtag("bitmapSizeTable")
        writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        # Create a lookup for all the simple names that make sense to
        # bitmap size table. Only read the information from these names.
        dataNames = set(self._getXMLMetricNames())
        for element in content:
            if not isinstance(element, tuple):
                continue
            name, attrs, content = element
            if name == "sbitLineMetrics":
                direction = attrs["direction"]
                assert direction in (
                    "hori",
                    "vert",
                ), "SbitLineMetrics direction specified invalid."
                metricObj = SbitLineMetrics()
                metricObj.fromXML(name, attrs, content, ttFont)
                vars(self)[direction] = metricObj
            elif name in dataNames:
                vars(self)[name] = safeEval(attrs["value"])
            else:
                log.warning("unknown name '%s' being ignored in BitmapSizeTable.", name)


class SbitLineMetrics(object):
    def toXML(self, name, writer, ttFont):
        writer.begintag("sbitLineMetrics", [("direction", name)])
        writer.newline()
        for metricName in sstruct.getformat(sbitLineMetricsFormat)[1]:
            writer.simpletag(metricName, value=getattr(self, metricName))
            writer.newline()
        writer.endtag("sbitLineMetrics")
        writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        metricNames = set(sstruct.getformat(sbitLineMetricsFormat)[1])
        for element in content:
            if not isinstance(element, tuple):
                continue
            name, attrs, content = element
            if name in metricNames:
                vars(self)[name] = safeEval(attrs["value"])


# Important information about the naming scheme. Used for identifying subtables.
_indexSubTableSubclassPrefix = "eblc_index_sub_table_"


class EblcIndexSubTable(object):
    def __init__(self, data, ttFont):
        self.data = data
        self.ttFont = ttFont
        # TODO Currently non-lazy decompiling doesn't work for this class...
        # if not ttFont.lazy:
        # 	self.decompile()
        # 	del self.data, self.ttFont

    def __getattr__(self, attr):
        # Allow lazy decompile.
        if attr[:2] == "__":
            raise AttributeError(attr)
        if attr == "data":
            raise AttributeError(attr)
        self.decompile()
        return getattr(self, attr)

    def ensureDecompiled(self, recurse=False):
        if hasattr(self, "data"):
            self.decompile()

    # This method just takes care of the indexSubHeader. Implementing subclasses
    # should call it to compile the indexSubHeader and then continue compiling
    # the remainder of their unique format.
    def compile(self, ttFont):
        return struct.pack(
            indexSubHeaderFormat,
            self.indexFormat,
            self.imageFormat,
            self.imageDataOffset,
        )

    # Creates the XML for bitmap glyphs. Each index sub table basically makes
    # the same XML except for specific metric information that is written
    # out via a method call that a subclass implements optionally.
    def toXML(self, writer, ttFont):
        writer.begintag(
            self.__class__.__name__,
            [
                ("imageFormat", self.imageFormat),
                ("firstGlyphIndex", self.firstGlyphIndex),
                ("lastGlyphIndex", self.lastGlyphIndex),
            ],
        )
        writer.newline()
        self.writeMetrics(writer, ttFont)
        # Write out the names as thats all thats needed to rebuild etc.
        # For font debugging of consecutive formats the ids are also written.
        # The ids are not read when moving from the XML format.
        glyphIds = map(ttFont.getGlyphID, self.names)
        for glyphName, glyphId in zip(self.names, glyphIds):
            writer.simpletag("glyphLoc", name=glyphName, id=glyphId)
            writer.newline()
        writer.endtag(self.__class__.__name__)
        writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        # Read all the attributes. Even though the glyph indices are
        # recalculated, they are still read in case there needs to
        # be an immediate export of the data.
        self.imageFormat = safeEval(attrs["imageFormat"])
        self.firstGlyphIndex = safeEval(attrs["firstGlyphIndex"])
        self.lastGlyphIndex = safeEval(attrs["lastGlyphIndex"])

        self.readMetrics(name, attrs, content, ttFont)

        self.names = []
        for element in content:
            if not isinstance(element, tuple):
                continue
            name, attrs, content = element
            if name == "glyphLoc":
                self.names.append(attrs["name"])

    # A helper method that writes the metrics for the index sub table. It also
    # is responsible for writing the image size for fixed size data since fixed
    # size is not recalculated on compile. Default behavior is to do nothing.
    def writeMetrics(self, writer, ttFont):
        pass

    # A helper method that is the inverse of writeMetrics.
    def readMetrics(self, name, attrs, content, ttFont):
        pass

    # This method is for fixed glyph data sizes. There are formats where
    # the glyph data is fixed but are actually composite glyphs. To handle
    # this the font spec in indexSubTable makes the data the size of the
    # fixed size by padding the component arrays. This function abstracts
    # out this padding process. Input is data unpadded. Output is data
    # padded only in fixed formats. Default behavior is to return the data.
    def padBitmapData(self, data):
        return data

    # Remove any of the glyph locations and names that are flagged as skipped.
    # This only occurs in formats {1,3}.
    def removeSkipGlyphs(self):
        # Determines if a name, location pair is a valid data location.
        # Skip glyphs are marked when the size is equal to zero.
        def isValidLocation(args):
            name, (startByte, endByte) = args
            return startByte < endByte

        # Remove all skip glyphs.
        dataPairs = list(filter(isValidLocation, zip(self.names, self.locations)))
        self.names, self.locations = list(map(list, zip(*dataPairs)))


# A closure for creating a custom mixin. This is done because formats 1 and 3
# are very similar. The only difference between them is the size per offset
# value. Code put in here should handle both cases generally.
def _createOffsetArrayIndexSubTableMixin(formatStringForDataType):
    # Prep the data size for the offset array data format.
    dataFormat = ">" + formatStringForDataType
    offsetDataSize = struct.calcsize(dataFormat)

    class OffsetArrayIndexSubTableMixin(object):
        def decompile(self):
            numGlyphs = self.lastGlyphIndex - self.firstGlyphIndex + 1
            indexingOffsets = [
                glyphIndex * offsetDataSize for glyphIndex in range(numGlyphs + 2)
            ]
            indexingLocations = zip(indexingOffsets, indexingOffsets[1:])
            offsetArray = [
                struct.unpack(dataFormat, self.data[slice(*loc)])[0]
                for loc in indexingLocations
            ]

            glyphIds = list(range(self.firstGlyphIndex, self.lastGlyphIndex + 1))
            modifiedOffsets = [offset + self.imageDataOffset for offset in offsetArray]
            self.locations = list(zip(modifiedOffsets, modifiedOffsets[1:]))

            self.names = list(map(self.ttFont.getGlyphName, glyphIds))
            self.removeSkipGlyphs()
            del self.data, self.ttFont

        def compile(self, ttFont):
            # First make sure that all the data lines up properly. Formats 1 and 3
            # must have all its data lined up consecutively. If not this will fail.
            for curLoc, nxtLoc in zip(self.locations, self.locations[1:]):
                assert (
                    curLoc[1] == nxtLoc[0]
                ), "Data must be consecutive in indexSubTable offset formats"

            glyphIds = list(map(ttFont.getGlyphID, self.names))
            # Make sure that all ids are sorted strictly increasing.
            assert all(glyphIds[i] < glyphIds[i + 1] for i in range(len(glyphIds) - 1))

            # Run a simple algorithm to add skip glyphs to the data locations at
            # the places where an id is not present.
            idQueue = deque(glyphIds)
            locQueue = deque(self.locations)
            allGlyphIds = list(range(self.firstGlyphIndex, self.lastGlyphIndex + 1))
            allLocations = []
            for curId in allGlyphIds:
                if curId != idQueue[0]:
                    allLocations.append((locQueue[0][0], locQueue[0][0]))
                else:
                    idQueue.popleft()
                    allLocations.append(locQueue.popleft())

            # Now that all the locations are collected, pack them appropriately into
            # offsets. This is the form where offset[i] is the location and
            # offset[i+1]-offset[i] is the size of the data location.
            offsets = list(allLocations[0]) + [loc[1] for loc in allLocations[1:]]
            # Image data offset must be less than or equal to the minimum of locations.
            # This offset may change the value for round tripping but is safer and
            # allows imageDataOffset to not be required to be in the XML version.
            self.imageDataOffset = min(offsets)
            offsetArray = [offset - self.imageDataOffset for offset in offsets]

            dataList = [EblcIndexSubTable.compile(self, ttFont)]
            dataList += [
                struct.pack(dataFormat, offsetValue) for offsetValue in offsetArray
            ]
            # Take care of any padding issues. Only occurs in format 3.
            if offsetDataSize * len(offsetArray) % 4 != 0:
                dataList.append(struct.pack(dataFormat, 0))
            return bytesjoin(dataList)

    return OffsetArrayIndexSubTableMixin


# A Mixin for functionality shared between the different kinds
# of fixed sized data handling. Both kinds have big metrics so
# that kind of special processing is also handled in this mixin.
class FixedSizeIndexSubTableMixin(object):
    def writeMetrics(self, writer, ttFont):
        writer.simpletag("imageSize", value=self.imageSize)
        writer.newline()
        self.metrics.toXML(writer, ttFont)

    def readMetrics(self, name, attrs, content, ttFont):
        for element in content:
            if not isinstance(element, tuple):
                continue
            name, attrs, content = element
            if name == "imageSize":
                self.imageSize = safeEval(attrs["value"])
            elif name == BigGlyphMetrics.__name__:
                self.metrics = BigGlyphMetrics()
                self.metrics.fromXML(name, attrs, content, ttFont)
            elif name == SmallGlyphMetrics.__name__:
                log.warning(
                    "SmallGlyphMetrics being ignored in format %d.", self.indexFormat
                )

    def padBitmapData(self, data):
        # Make sure that the data isn't bigger than the fixed size.
        assert len(data) <= self.imageSize, (
            "Data in indexSubTable format %d must be less than the fixed size."
            % self.indexFormat
        )
        # Pad the data so that it matches the fixed size.
        pad = (self.imageSize - len(data)) * b"\0"
        return data + pad


class eblc_index_sub_table_1(
    _createOffsetArrayIndexSubTableMixin("L"), EblcIndexSubTable
):
    pass


class eblc_index_sub_table_2(FixedSizeIndexSubTableMixin, EblcIndexSubTable):
    def decompile(self):
        (self.imageSize,) = struct.unpack(">L", self.data[:4])
        self.metrics = BigGlyphMetrics()
        sstruct.unpack2(bigGlyphMetricsFormat, self.data[4:], self.metrics)
        glyphIds = list(range(self.firstGlyphIndex, self.lastGlyphIndex + 1))
        offsets = [
            self.imageSize * i + self.imageDataOffset for i in range(len(glyphIds) + 1)
        ]
        self.locations = list(zip(offsets, offsets[1:]))
        self.names = list(map(self.ttFont.getGlyphName, glyphIds))
        del self.data, self.ttFont

    def compile(self, ttFont):
        glyphIds = list(map(ttFont.getGlyphID, self.names))
        # Make sure all the ids are consecutive. This is required by Format 2.
        assert glyphIds == list(
            range(self.firstGlyphIndex, self.lastGlyphIndex + 1)
        ), "Format 2 ids must be consecutive."
        self.imageDataOffset = min(next(iter(zip(*self.locations))))

        dataList = [EblcIndexSubTable.compile(self, ttFont)]
        dataList.append(struct.pack(">L", self.imageSize))
        dataList.append(sstruct.pack(bigGlyphMetricsFormat, self.metrics))
        return bytesjoin(dataList)


class eblc_index_sub_table_3(
    _createOffsetArrayIndexSubTableMixin("H"), EblcIndexSubTable
):
    pass


class eblc_index_sub_table_4(EblcIndexSubTable):
    def decompile(self):
        (numGlyphs,) = struct.unpack(">L", self.data[:4])
        data = self.data[4:]
        indexingOffsets = [
            glyphIndex * codeOffsetPairSize for glyphIndex in range(numGlyphs + 2)
        ]
        indexingLocations = zip(indexingOffsets, indexingOffsets[1:])
        glyphArray = [
            struct.unpack(codeOffsetPairFormat, data[slice(*loc)])
            for loc in indexingLocations
        ]
        glyphIds, offsets = list(map(list, zip(*glyphArray)))
        # There are one too many glyph ids. Get rid of the last one.
        glyphIds.pop()

        offsets = [offset + self.imageDataOffset for offset in offsets]
        self.locations = list(zip(offsets, offsets[1:]))
        self.names = list(map(self.ttFont.getGlyphName, glyphIds))
        del self.data, self.ttFont

    def compile(self, ttFont):
        # First make sure that all the data lines up properly. Format 4
        # must have all its data lined up consecutively. If not this will fail.
        for curLoc, nxtLoc in zip(self.locations, self.locations[1:]):
            assert (
                curLoc[1] == nxtLoc[0]
            ), "Data must be consecutive in indexSubTable format 4"

        offsets = list(self.locations[0]) + [loc[1] for loc in self.locations[1:]]
        # Image data offset must be less than or equal to the minimum of locations.
        # Resetting this offset may change the value for round tripping but is safer
        # and allows imageDataOffset to not be required to be in the XML version.
        self.imageDataOffset = min(offsets)
        offsets = [offset - self.imageDataOffset for offset in offsets]
        glyphIds = list(map(ttFont.getGlyphID, self.names))
        # Create an iterator over the ids plus a padding value.
        idsPlusPad = list(itertools.chain(glyphIds, [0]))

        dataList = [EblcIndexSubTable.compile(self, ttFont)]
        dataList.append(struct.pack(">L", len(glyphIds)))
        tmp = [
            struct.pack(codeOffsetPairFormat, *cop) for cop in zip(idsPlusPad, offsets)
        ]
        dataList += tmp
        data = bytesjoin(dataList)
        return data


class eblc_index_sub_table_5(FixedSizeIndexSubTableMixin, EblcIndexSubTable):
    def decompile(self):
        self.origDataLen = 0
        (self.imageSize,) = struct.unpack(">L", self.data[:4])
        data = self.data[4:]
        self.metrics, data = sstruct.unpack2(
            bigGlyphMetricsFormat, data, BigGlyphMetrics()
        )
        (numGlyphs,) = struct.unpack(">L", data[:4])
        data = data[4:]
        glyphIds = [
            struct.unpack(">H", data[2 * i : 2 * (i + 1)])[0] for i in range(numGlyphs)
        ]

        offsets = [
            self.imageSize * i + self.imageDataOffset for i in range(len(glyphIds) + 1)
        ]
        self.locations = list(zip(offsets, offsets[1:]))
        self.names = list(map(self.ttFont.getGlyphName, glyphIds))
        del self.data, self.ttFont

    def compile(self, ttFont):
        self.imageDataOffset = min(next(iter(zip(*self.locations))))
        dataList = [EblcIndexSubTable.compile(self, ttFont)]
        dataList.append(struct.pack(">L", self.imageSize))
        dataList.append(sstruct.pack(bigGlyphMetricsFormat, self.metrics))
        glyphIds = list(map(ttFont.getGlyphID, self.names))
        dataList.append(struct.pack(">L", len(glyphIds)))
        dataList += [struct.pack(">H", curId) for curId in glyphIds]
        if len(glyphIds) % 2 == 1:
            dataList.append(struct.pack(">H", 0))
        return bytesjoin(dataList)


# Dictionary of indexFormat to the class representing that format.
eblc_sub_table_classes = {
    1: eblc_index_sub_table_1,
    2: eblc_index_sub_table_2,
    3: eblc_index_sub_table_3,
    4: eblc_index_sub_table_4,
    5: eblc_index_sub_table_5,
}
<|endoftext|>from fontTools.misc import sstruct
from fontTools.misc.textTools import safeEval
from .sbixGlyph import Glyph
import struct

sbixStrikeHeaderFormat = """
	>
	ppem:          H	# The PPEM for which this strike was designed (e.g., 9,
						# 12, 24)
	resolution:    H	# The screen resolution (in dpi) for which this strike
						# was designed (e.g., 72)
"""

sbixGlyphDataOffsetFormat = """
	>
	glyphDataOffset:   L	# Offset from the beginning of the strike data record
							# to data for the individual glyph
"""

sbixStrikeHeaderFormatSize = sstruct.calcsize(sbixStrikeHeaderFormat)
sbixGlyphDataOffsetFormatSize = sstruct.calcsize(sbixGlyphDataOffsetFormat)


class Strike(object):
    def __init__(self, rawdata=None, ppem=0, resolution=72):
        self.data = rawdata
        self.ppem = ppem
        self.resolution = resolution
        self.glyphs = {}

    def decompile(self, ttFont):
        if self.data is None:
            from fontTools import ttLib

            raise ttLib.TTLibError
        if len(self.data) < sbixStrikeHeaderFormatSize:
            from fontTools import ttLib

            raise (
                ttLib.TTLibError,
                "Strike header too short: Expected %x, got %x.",
            ) % (sbixStrikeHeaderFormatSize, len(self.data))

        # read Strike header from raw data
        sstruct.unpack(
            sbixStrikeHeaderFormat, self.data[:sbixStrikeHeaderFormatSize], self
        )

        # calculate number of glyphs
        (firstGlyphDataOffset,) = struct.unpack(
            ">L",
            self.data[
                sbixStrikeHeaderFormatSize : sbixStrikeHeaderFormatSize
                + sbixGlyphDataOffsetFormatSize
            ],
        )
        self.numGlyphs = (
            firstGlyphDataOffset - sbixStrikeHeaderFormatSize
        ) // sbixGlyphDataOffsetFormatSize - 1
        # ^ -1 because there's one more offset than glyphs

        # build offset list for single glyph data offsets
        self.glyphDataOffsets = []
        for i in range(
            self.numGlyphs + 1
        ):  # + 1 because there's one more offset than glyphs
            start = i * sbixGlyphDataOffsetFormatSize + sbixStrikeHeaderFormatSize
            (current_offset,) = struct.unpack(
                ">L", self.data[start : start + sbixGlyphDataOffsetFormatSize]
            )
            self.glyphDataOffsets.append(current_offset)

        # iterate through offset list and slice raw data into glyph data records
        for i in range(self.numGlyphs):
            current_glyph = Glyph(
                rawdata=self.data[
                    self.glyphDataOffsets[i] : self.glyphDataOffsets[i + 1]
                ],
                gid=i,
            )
            current_glyph.decompile(ttFont)
            self.glyphs[current_glyph.glyphName] = current_glyph
        del self.glyphDataOffsets
        del self.numGlyphs
        del self.data

    def compile(self, ttFont):
        self.glyphDataOffsets = b""
        self.bitmapData = b""

        glyphOrder = ttFont.getGlyphOrder()

        # first glyph starts right after the header
        currentGlyphDataOffset = (
            sbixStrikeHeaderFormatSize
            + sbixGlyphDataOffsetFormatSize * (len(glyphOrder) + 1)
        )
        for glyphName in glyphOrder:
            if glyphName in self.glyphs:
                # we have glyph data for this glyph
                current_glyph = self.glyphs[glyphName]
            else:
                # must add empty glyph data record for this glyph
                current_glyph = Glyph(glyphName=glyphName)
            current_glyph.compile(ttFont)
            current_glyph.glyphDataOffset = currentGlyphDataOffset
            self.bitmapData += current_glyph.rawdata
            currentGlyphDataOffset += len(current_glyph.rawdata)
            self.glyphDataOffsets += sstruct.pack(
                sbixGlyphDataOffsetFormat, current_glyph
            )

        # add last "offset", really the end address of the last glyph data record
        dummy = Glyph()
        dummy.glyphDataOffset = currentGlyphDataOffset
        self.glyphDataOffsets += sstruct.pack(sbixGlyphDataOffsetFormat, dummy)

        # pack header
        self.data = sstruct.pack(sbixStrikeHeaderFormat, self)
        # add offsets and image data after header
        self.data += self.glyphDataOffsets + self.bitmapData

    def toXML(self, xmlWriter, ttFont):
        xmlWriter.begintag("strike")
        xmlWriter.newline()
        xmlWriter.simpletag("ppem", value=self.ppem)
        xmlWriter.newline()
        xmlWriter.simpletag("resolution", value=self.resolution)
        xmlWriter.newline()
        glyphOrder = ttFont.getGlyphOrder()
        for glyphName in glyphOrder:
            if glyphName in self.glyphs:
                self.glyphs[glyphName].toXML(xmlWriter, ttFont)
                # TODO: what if there are more glyph data records than (glyf table) glyphs?
        xmlWriter.endtag("strike")
        xmlWriter.newline()

    def fromXML(self, name, attrs, content, ttFont):
        if name in ["ppem", "resolution"]:
            setattr(self, name, safeEval(attrs["value"]))
        elif name == "glyph":
            if "graphicType" in attrs:
                myFormat = safeEval("'''" + attrs["graphicType"] + "'''")
            else:
                myFormat = None
            if "glyphname" in attrs:
                myGlyphName = safeEval("'''" + attrs["glyphname"] + "'''")
            elif "name" in attrs:
                myGlyphName = safeEval("'''" + attrs["name"] + "'''")
            else:
                from fontTools import ttLib

                raise ttLib.TTLibError("Glyph must have a glyph name.")
            if "originOffsetX" in attrs:
