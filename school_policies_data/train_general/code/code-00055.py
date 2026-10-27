        assert self.ulVersion == 1, "DSIG ulVersion must be 1"
        assert self.usFlag & ~1 == 0, "DSIG usFlag must be 0x1 or 0x0"
        self.signatureRecords: list[SignatureRecord] = []
        sigrecs = self.signatureRecords
        for n in range(self.usNumSigs):
            sigrec, newData = sstruct.unpack2(
                DSIG_SignatureFormat, newData, SignatureRecord()
            )
            assert sigrec.ulFormat == 1, (
                "DSIG signature record #%d ulFormat must be 1" % n
            )
            sigrecs.append(sigrec)
        for sigrec in sigrecs:
            dummy, newData = sstruct.unpack2(
                DSIG_SignatureBlockFormat, data[sigrec.ulOffset :], sigrec
            )
            assert sigrec.usReserved1 == 0, (
                "DSIG signature record #%d usReserverd1 must be 0" % n
            )
            assert sigrec.usReserved2 == 0, (
                "DSIG signature record #%d usReserverd2 must be 0" % n
            )
            sigrec.pkcs7 = newData[: sigrec.cbSignature]

    def compile(self, ttFont: TTFont) -> bytes:
        packed = sstruct.pack(DSIG_HeaderFormat, self)
        headers = [packed]
        offset = len(packed) + self.usNumSigs * sstruct.calcsize(DSIG_SignatureFormat)
        data = []
        for sigrec in self.signatureRecords:
            # first pack signature block
            sigrec.cbSignature = len(sigrec.pkcs7)
            packed = sstruct.pack(DSIG_SignatureBlockFormat, sigrec) + sigrec.pkcs7
            data.append(packed)
            # update redundant length field
            sigrec.ulLength = len(packed)
            # update running table offset
            sigrec.ulOffset = offset
            headers.append(sstruct.pack(DSIG_SignatureFormat, sigrec))
            offset += sigrec.ulLength
        if offset % 2:
            # Pad to even bytes
            data.append(b"\0")
        return bytesjoin(headers + data)

    def toXML(self, xmlWriter: XMLWriter, ttFont: TTFont) -> None:
        xmlWriter.comment(
            "note that the Digital Signature will be invalid after recompilation!"
        )
        xmlWriter.newline()
        xmlWriter.simpletag(
            "tableHeader",
            version=self.ulVersion,
            numSigs=self.usNumSigs,
            flag="0x%X" % self.usFlag,
        )
        for sigrec in self.signatureRecords:
            xmlWriter.newline()
            sigrec.toXML(xmlWriter, ttFont)
        xmlWriter.newline()

    def fromXML(self, name, attrs, content, ttFont):
        if name == "tableHeader":
            self.signatureRecords = []
            self.ulVersion = safeEval(attrs["version"])
            self.usNumSigs = safeEval(attrs["numSigs"])
            self.usFlag = safeEval(attrs["flag"])
            return
        if name == "SignatureRecord":
            sigrec = SignatureRecord()
            sigrec.fromXML(name, attrs, content, ttFont)
            self.signatureRecords.append(sigrec)


pem_spam = lambda l, spam={
    "-----BEGIN PKCS7-----": True,
    "-----END PKCS7-----": True,
    "": True,
}: not spam.get(l.strip())


def b64encode(b):
    s = base64.b64encode(b)
    # Line-break at 76 chars.
    items = []
    while s:
        items.append(tostr(s[:76]))
        items.append("\n")
        s = s[76:]
    return strjoin(items)


class SignatureRecord:
    def __repr__(self) -> str:
        return "<%s: %s>" % (self.__class__.__name__, self.__dict__)

    def toXML(self, writer, ttFont: TTFont) -> None:
        writer.begintag(self.__class__.__name__, format=self.ulFormat)
        writer.newline()
        writer.write_noindent("-----BEGIN PKCS7-----\n")
        writer.write_noindent(b64encode(self.pkcs7))
        writer.write_noindent("-----END PKCS7-----\n")
        writer.endtag(self.__class__.__name__)

    def fromXML(
        self, name: str, attrs: dict[str, str], content: str, ttFont: TTFont
    ) -> None:
        self.ulFormat: int = safeEval(attrs["format"])
        self.usReserved1: int = safeEval(attrs.get("reserved1", "0"))
        self.usReserved2: int = safeEval(attrs.get("reserved2", "0"))
        self.pkcs7 = base64.b64decode(tobytes(strjoin(filter(pem_spam, content))))
<|endoftext|>import json
from textwrap import indent

from . import DefaultTable
from fontTools.misc.textTools import tostr


class table_D__e_b_g(DefaultTable.DefaultTable):
    def __init__(self, tag=None):
        DefaultTable.DefaultTable.__init__(self, tag)
        self.data = {}

    def decompile(self, data, ttFont):
        self.data = json.loads(data)

    def compile(self, ttFont):
        return json.dumps(self.data).encode("utf-8")

    def toXML(self, writer, ttFont):
        # make sure json indentation inside CDATA block matches XMLWriter's
        data = json.dumps(self.data, indent=len(writer.indentwhite))
        prefix = tostr(writer.indentwhite) * (writer.indentlevel + 1)
        # but don't indent the first json line so it's adjacent to `<![CDATA[{`
        cdata = indent(data, prefix, lambda ln: ln != "{\n")

        writer.begintag("json")
        writer.newline()
        writer.writecdata(cdata)
        writer.newline()
        writer.endtag("json")
        writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        if name == "json":
            self.data = json.loads("".join(content))
<|endoftext|># coding: utf-8
"""fontTools.ttLib.tables.otTables -- A collection of classes representing the various
OpenType subtables.

Most are constructed upon import from data in otData.py, all are populated with
converter objects from otConverters.py.
"""

import copy
from enum import IntEnum
from functools import reduce
from math import radians
import itertools
from collections import defaultdict, namedtuple
from fontTools.ttLib import OPTIMIZE_FONT_SPEED
from fontTools.ttLib.tables.TupleVariation import TupleVariation
from fontTools.ttLib.tables.otTraverse import dfs_base_table
from fontTools.misc.arrayTools import quantizeRect
from fontTools.misc.roundTools import otRound
from fontTools.misc.transform import Transform, Identity, DecomposedTransform
from fontTools.misc.textTools import bytesjoin, pad, safeEval
from fontTools.misc.vector import Vector
from fontTools.pens.boundsPen import ControlBoundsPen
from fontTools.pens.transformPen import TransformPen
from .otBase import (
    BaseTable,
    FormatSwitchingBaseTable,
    ValueRecord,
    CountReference,
    getFormatSwitchingBaseTableClass,
    have_uharfbuzz,
)
from fontTools.misc.fixedTools import (
    fixedToFloat as fi2fl,
    floatToFixed as fl2fi,
    floatToFixedToStr as fl2str,
    strToFixedToFloat as str2fl,
)
from fontTools.feaLib.lookupDebugInfo import LookupDebugInfo, LOOKUP_DEBUG_INFO_KEY
import logging
import struct
import array
import sys
from enum import IntFlag
from typing import TYPE_CHECKING, Iterator, List, Optional, Set

if TYPE_CHECKING:
    from fontTools.ttLib.ttGlyphSet import _TTGlyphSet


log = logging.getLogger(__name__)


class VarComponentFlags(IntFlag):
    RESET_UNSPECIFIED_AXES = 1 << 0

    HAVE_AXES = 1 << 1

    AXIS_VALUES_HAVE_VARIATION = 1 << 2
    TRANSFORM_HAS_VARIATION = 1 << 3

    HAVE_TRANSLATE_X = 1 << 4
    HAVE_TRANSLATE_Y = 1 << 5
    HAVE_ROTATION = 1 << 6

    HAVE_CONDITION = 1 << 7

    HAVE_SCALE_X = 1 << 8
    HAVE_SCALE_Y = 1 << 9
    HAVE_TCENTER_X = 1 << 10
    HAVE_TCENTER_Y = 1 << 11

    GID_IS_24BIT = 1 << 12

    HAVE_SKEW_X = 1 << 13
    HAVE_SKEW_Y = 1 << 14

    RESERVED_MASK = (1 << 32) - (1 << 15)


VarTransformMappingValues = namedtuple(
    "VarTransformMappingValues",
    ["flag", "fractionalBits", "scale", "defaultValue"],
)

VAR_TRANSFORM_MAPPING = {
    "translateX": VarTransformMappingValues(
        VarComponentFlags.HAVE_TRANSLATE_X, 0, 1, 0
    ),
    "translateY": VarTransformMappingValues(
        VarComponentFlags.HAVE_TRANSLATE_Y, 0, 1, 0
    ),
    "rotation": VarTransformMappingValues(VarComponentFlags.HAVE_ROTATION, 12, 180, 0),
    "scaleX": VarTransformMappingValues(VarComponentFlags.HAVE_SCALE_X, 10, 1, 1),
    "scaleY": VarTransformMappingValues(VarComponentFlags.HAVE_SCALE_Y, 10, 1, 1),
    "skewX": VarTransformMappingValues(VarComponentFlags.HAVE_SKEW_X, 12, -180, 0),
    "skewY": VarTransformMappingValues(VarComponentFlags.HAVE_SKEW_Y, 12, 180, 0),
    "tCenterX": VarTransformMappingValues(VarComponentFlags.HAVE_TCENTER_X, 0, 1, 0),
    "tCenterY": VarTransformMappingValues(VarComponentFlags.HAVE_TCENTER_Y, 0, 1, 0),
}

# Probably should be somewhere in fontTools.misc
_packer = {
    1: lambda v: struct.pack(">B", v),
    2: lambda v: struct.pack(">H", v),
    3: lambda v: struct.pack(">L", v)[1:],
    4: lambda v: struct.pack(">L", v),
}
_unpacker = {
    1: lambda v: struct.unpack(">B", v)[0],
    2: lambda v: struct.unpack(">H", v)[0],
    3: lambda v: struct.unpack(">L", b"\0" + v)[0],
    4: lambda v: struct.unpack(">L", v)[0],
}


def _read_uint32var(data, i):
    """Read a variable-length number from data starting at index i.

    Return the number and the next index.
    """

    b0 = data[i]
    if b0 < 0x80:
        return b0, i + 1
    elif b0 < 0xC0:
        return (b0 - 0x80) << 8 | data[i + 1], i + 2
    elif b0 < 0xE0:
        return (b0 - 0xC0) << 16 | data[i + 1] << 8 | data[i + 2], i + 3
    elif b0 < 0xF0:
        return (b0 - 0xE0) << 24 | data[i + 1] << 16 | data[i + 2] << 8 | data[
            i + 3
        ], i + 4
    else:
        return (b0 - 0xF0) << 32 | data[i + 1] << 24 | data[i + 2] << 16 | data[
            i + 3
        ] << 8 | data[i + 4], i + 5


def _write_uint32var(v):
    """Write a variable-length number.

    Return the data.
    """
    if v < 0x80:
        return struct.pack(">B", v)
    elif v < 0x4000:
        return struct.pack(">H", (v | 0x8000))
    elif v < 0x200000:
        return struct.pack(">L", (v | 0xC00000))[1:]
    elif v < 0x10000000:
        return struct.pack(">L", (v | 0xE0000000))
    else:
        return struct.pack(">B", 0xF0) + struct.pack(">L", v)


class VarComponent:
    def __init__(self):
        self.populateDefaults()

    def populateDefaults(self, propagator=None):
        self.flags = 0
        self.glyphName = None
        self.conditionIndex = None
        self.axisIndicesIndex = None
        self.axisValues = ()
        self.axisValuesVarIndex = NO_VARIATION_INDEX
        self.transformVarIndex = NO_VARIATION_INDEX
        self.transform = DecomposedTransform()

    def decompile(self, data, font, localState):
        i = 0
        self.flags, i = _read_uint32var(data, i)
        flags = self.flags

        gidSize = 3 if flags & VarComponentFlags.GID_IS_24BIT else 2
        glyphID = _unpacker[gidSize](data[i : i + gidSize])
        i += gidSize
        self.glyphName = font.glyphOrder[glyphID]

        if flags & VarComponentFlags.HAVE_CONDITION:
            self.conditionIndex, i = _read_uint32var(data, i)

        if flags & VarComponentFlags.HAVE_AXES:
            self.axisIndicesIndex, i = _read_uint32var(data, i)
        else:
            self.axisIndicesIndex = None

        if self.axisIndicesIndex is None:
            numAxes = 0
        else:
            axisIndices = localState["AxisIndicesList"].Item[self.axisIndicesIndex]
            numAxes = len(axisIndices)

        if flags & VarComponentFlags.HAVE_AXES:
            axisValues, i = TupleVariation.decompileDeltas_(numAxes, data, i)
            self.axisValues = tuple(fi2fl(v, 14) for v in axisValues)
        else:
            self.axisValues = ()
        assert len(self.axisValues) == numAxes

        if flags & VarComponentFlags.AXIS_VALUES_HAVE_VARIATION:
            self.axisValuesVarIndex, i = _read_uint32var(data, i)
        else:
            self.axisValuesVarIndex = NO_VARIATION_INDEX
        if flags & VarComponentFlags.TRANSFORM_HAS_VARIATION:
            self.transformVarIndex, i = _read_uint32var(data, i)
        else:
            self.transformVarIndex = NO_VARIATION_INDEX

        self.transform = DecomposedTransform()

        def read_transform_component(values):
            nonlocal i
            if flags & values.flag:
                v = (
                    fi2fl(
                        struct.unpack(">h", data[i : i + 2])[0], values.fractionalBits
                    )
                    * values.scale
                )
                i += 2
                return v
            else:
                return values.defaultValue

        for attr_name, mapping_values in VAR_TRANSFORM_MAPPING.items():
            value = read_transform_component(mapping_values)
            setattr(self.transform, attr_name, value)

        if not (flags & VarComponentFlags.HAVE_SCALE_Y):
            self.transform.scaleY = self.transform.scaleX

        n = flags & VarComponentFlags.RESERVED_MASK
        while n:
            _, i = _read_uint32var(data, i)
            n &= n - 1

        return data[i:]

    def compile(self, font):
        optimizeSpeed = font.cfg[OPTIMIZE_FONT_SPEED]

        data = []

        flags = self.flags

        glyphID = font.getGlyphID(self.glyphName)
        if glyphID > 65535:
            flags |= VarComponentFlags.GID_IS_24BIT
            data.append(_packer[3](glyphID))
        else:
            flags &= ~VarComponentFlags.GID_IS_24BIT
            data.append(_packer[2](glyphID))

        if self.conditionIndex is not None:
            flags |= VarComponentFlags.HAVE_CONDITION
            data.append(_write_uint32var(self.conditionIndex))

        numAxes = len(self.axisValues)

        if numAxes:
            flags |= VarComponentFlags.HAVE_AXES
            data.append(_write_uint32var(self.axisIndicesIndex))
            data.append(
                TupleVariation.compileDeltaValues_(
                    [fl2fi(v, 14) for v in self.axisValues],
                    optimizeSize=not optimizeSpeed,
                )
            )
        else:
            flags &= ~VarComponentFlags.HAVE_AXES

        if self.axisValuesVarIndex != NO_VARIATION_INDEX:
            flags |= VarComponentFlags.AXIS_VALUES_HAVE_VARIATION
            data.append(_write_uint32var(self.axisValuesVarIndex))
        else:
            flags &= ~VarComponentFlags.AXIS_VALUES_HAVE_VARIATION
        if self.transformVarIndex != NO_VARIATION_INDEX:
            flags |= VarComponentFlags.TRANSFORM_HAS_VARIATION
            data.append(_write_uint32var(self.transformVarIndex))
        else:
            flags &= ~VarComponentFlags.TRANSFORM_HAS_VARIATION

        def write_transform_component(value, values):
            if flags & values.flag:
                return struct.pack(
                    ">h", fl2fi(value / values.scale, values.fractionalBits)
                )
            else:
                return b""

        for attr_name, mapping_values in VAR_TRANSFORM_MAPPING.items():
            value = getattr(self.transform, attr_name)
            data.append(write_transform_component(value, mapping_values))

        return _write_uint32var(flags) + bytesjoin(data)

    def toXML(self, writer, ttFont, attrs):
        writer.begintag("VarComponent", attrs)
        writer.newline()

        def write(name, value, attrs=()):
            if value is not None:
                writer.simpletag(name, (("value", value),) + attrs)
                writer.newline()

        write("glyphName", self.glyphName)

        if self.conditionIndex is not None:
            write("conditionIndex", self.conditionIndex)
        if self.axisIndicesIndex is not None:
            write("axisIndicesIndex", self.axisIndicesIndex)
        if (
            self.axisIndicesIndex is not None
            or self.flags & VarComponentFlags.RESET_UNSPECIFIED_AXES
        ):
            if self.flags & VarComponentFlags.RESET_UNSPECIFIED_AXES:
                attrs = (("resetUnspecifiedAxes", 1),)
            else:
                attrs = ()
            write("axisValues", [float(fl2str(v, 14)) for v in self.axisValues], attrs)

        if self.axisValuesVarIndex != NO_VARIATION_INDEX:
            write("axisValuesVarIndex", self.axisValuesVarIndex)
        if self.transformVarIndex != NO_VARIATION_INDEX:
            write("transformVarIndex", self.transformVarIndex)

        # Only write transform components that are specified in the
        # flags, even if they are the default value.
        for attr_name, mapping in VAR_TRANSFORM_MAPPING.items():
            if not (self.flags & mapping.flag):
                continue
            v = getattr(self.transform, attr_name)
            write(attr_name, fl2str(v, mapping.fractionalBits))

        writer.endtag("VarComponent")
        writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        content = [c for c in content if isinstance(c, tuple)]

        self.populateDefaults()

        for name, attrs, content in content:
            assert not content
            v = attrs["value"]

            if name == "glyphName":
                self.glyphName = v
            elif name == "conditionIndex":
                self.conditionIndex = safeEval(v)
            elif name == "axisIndicesIndex":
                self.axisIndicesIndex = safeEval(v)
            elif name == "axisValues":
                self.axisValues = tuple(str2fl(v, 14) for v in safeEval(v))
                if safeEval(attrs.get("resetUnspecifiedAxes", "0")):
                    self.flags |= VarComponentFlags.RESET_UNSPECIFIED_AXES
            elif name == "axisValuesVarIndex":
                self.axisValuesVarIndex = safeEval(v)
            elif name == "transformVarIndex":
                self.transformVarIndex = safeEval(v)
            elif name in VAR_TRANSFORM_MAPPING:
                setattr(
                    self.transform,
                    name,
                    safeEval(v),
                )
                self.flags |= VAR_TRANSFORM_MAPPING[name].flag
            else:
                assert False, name

    def applyTransformDeltas(self, deltas):
        i = 0

        def read_transform_component_delta(values):
            nonlocal i
            if self.flags & values.flag:
                v = fi2fl(deltas[i], values.fractionalBits) * values.scale
                i += 1
                return v
            else:
                return 0

        for attr_name, mapping_values in VAR_TRANSFORM_MAPPING.items():
            value = read_transform_component_delta(mapping_values)
            setattr(
                self.transform, attr_name, getattr(self.transform, attr_name) + value
            )

        if not (self.flags & VarComponentFlags.HAVE_SCALE_Y):
            self.transform.scaleY = self.transform.scaleX

        assert i == len(deltas), (i, len(deltas))

    def __eq__(self, other):
        if type(self) != type(other):
            return NotImplemented
        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        result = self.__eq__(other)
        return result if result is NotImplemented else not result


class VarCompositeGlyph:
    def __init__(self, components=None):
        self.components = components if components is not None else []

    def decompile(self, data, font, localState):
        self.components = []
        while data:
            component = VarComponent()
            data = component.decompile(data, font, localState)
            self.components.append(component)

    def compile(self, font):
        data = []
        for component in self.components:
            data.append(component.compile(font))
        return bytesjoin(data)

    def toXML(self, xmlWriter, font, attrs, name):
        xmlWriter.begintag("VarCompositeGlyph", attrs)
        xmlWriter.newline()
        for i, component in enumerate(self.components):
            component.toXML(xmlWriter, font, [("index", i)])
        xmlWriter.endtag("VarCompositeGlyph")
        xmlWriter.newline()

    def fromXML(self, name, attrs, content, font):
        content = [c for c in content if isinstance(c, tuple)]
        for name, attrs, content in content:
            assert name == "VarComponent"
            component = VarComponent()
            component.fromXML(name, attrs, content, font)
            self.components.append(component)


class AATStateTable(object):
    def __init__(self):
        self.GlyphClasses = {}  # GlyphID --> GlyphClass
        self.States = []  # List of AATState, indexed by state number
        self.PerGlyphLookups = []  # [{GlyphID:GlyphID}, ...]


class AATState(object):
    def __init__(self):
        self.Transitions = {}  # GlyphClass --> AATAction


class AATAction(object):
    _FLAGS = None

    @staticmethod
    def compileActions(font, states):
        return (None, None)

    def _writeFlagsToXML(self, xmlWriter):
        flags = [f for f in self._FLAGS if self.__dict__[f]]
        if flags:
            xmlWriter.simpletag("Flags", value=",".join(flags))
            xmlWriter.newline()
        if self.ReservedFlags != 0:
            xmlWriter.simpletag("ReservedFlags", value="0x%04X" % self.ReservedFlags)
            xmlWriter.newline()

    def _setFlag(self, flag):
        assert flag in self._FLAGS, "unsupported flag %s" % flag
        self.__dict__[flag] = True


class RearrangementMorphAction(AATAction):
    staticSize = 4
    actionHeaderSize = 0
    _FLAGS = ["MarkFirst", "DontAdvance", "MarkLast"]

    _VERBS = {
        0: "no change",
        1: "Ax ⇒ xA",
        2: "xD ⇒ Dx",
        3: "AxD ⇒ DxA",
        4: "ABx ⇒ xAB",
        5: "ABx ⇒ xBA",
        6: "xCD ⇒ CDx",
        7: "xCD ⇒ DCx",
        8: "AxCD ⇒ CDxA",
        9: "AxCD ⇒ DCxA",
        10: "ABxD ⇒ DxAB",
        11: "ABxD ⇒ DxBA",
        12: "ABxCD ⇒ CDxAB",
        13: "ABxCD ⇒ CDxBA",
        14: "ABxCD ⇒ DCxAB",
        15: "ABxCD ⇒ DCxBA",
    }

    def __init__(self):
        self.NewState = 0
        self.Verb = 0
        self.MarkFirst = False
        self.DontAdvance = False
        self.MarkLast = False
        self.ReservedFlags = 0

    def compile(self, writer, font, actionIndex):
        assert actionIndex is None
        writer.writeUShort(self.NewState)
        assert self.Verb >= 0 and self.Verb <= 15, self.Verb
        flags = self.Verb | self.ReservedFlags
        if self.MarkFirst:
            flags |= 0x8000
        if self.DontAdvance:
            flags |= 0x4000
        if self.MarkLast:
            flags |= 0x2000
        writer.writeUShort(flags)

    def decompile(self, reader, font, actionReader):
        assert actionReader is None
        self.NewState = reader.readUShort()
        flags = reader.readUShort()
        self.Verb = flags & 0xF
        self.MarkFirst = bool(flags & 0x8000)
        self.DontAdvance = bool(flags & 0x4000)
        self.MarkLast = bool(flags & 0x2000)
        self.ReservedFlags = flags & 0x1FF0

    def toXML(self, xmlWriter, font, attrs, name):
        xmlWriter.begintag(name, **attrs)
        xmlWriter.newline()
        xmlWriter.simpletag("NewState", value=self.NewState)
        xmlWriter.newline()
        self._writeFlagsToXML(xmlWriter)
        xmlWriter.simpletag("Verb", value=self.Verb)
        verbComment = self._VERBS.get(self.Verb)
        if verbComment is not None:
            xmlWriter.comment(verbComment)
        xmlWriter.newline()
        xmlWriter.endtag(name)
        xmlWriter.newline()

    def fromXML(self, name, attrs, content, font):
        self.NewState = self.Verb = self.ReservedFlags = 0
        self.MarkFirst = self.DontAdvance = self.MarkLast = False
        content = [t for t in content if isinstance(t, tuple)]
        for eltName, eltAttrs, eltContent in content:
            if eltName == "NewState":
                self.NewState = safeEval(eltAttrs["value"])
            elif eltName == "Verb":
                self.Verb = safeEval(eltAttrs["value"])
            elif eltName == "ReservedFlags":
                self.ReservedFlags = safeEval(eltAttrs["value"])
            elif eltName == "Flags":
                for flag in eltAttrs["value"].split(","):
                    self._setFlag(flag.strip())


class ContextualMorphAction(AATAction):
    staticSize = 8
    actionHeaderSize = 0
    _FLAGS = ["SetMark", "DontAdvance"]

    def __init__(self):
        self.NewState = 0
        self.SetMark, self.DontAdvance = False, False
        self.ReservedFlags = 0
        self.MarkIndex, self.CurrentIndex = 0xFFFF, 0xFFFF

    def compile(self, writer, font, actionIndex):
        assert actionIndex is None
        writer.writeUShort(self.NewState)
        flags = self.ReservedFlags
        if self.SetMark:
            flags |= 0x8000
        if self.DontAdvance:
            flags |= 0x4000
        writer.writeUShort(flags)
        writer.writeUShort(self.MarkIndex)
        writer.writeUShort(self.CurrentIndex)

    def decompile(self, reader, font, actionReader):
        assert actionReader is None
        self.NewState = reader.readUShort()
        flags = reader.readUShort()
        self.SetMark = bool(flags & 0x8000)
        self.DontAdvance = bool(flags & 0x4000)
        self.ReservedFlags = flags & 0x3FFF
        self.MarkIndex = reader.readUShort()
        self.CurrentIndex = reader.readUShort()

    def toXML(self, xmlWriter, font, attrs, name):
        xmlWriter.begintag(name, **attrs)
        xmlWriter.newline()
        xmlWriter.simpletag("NewState", value=self.NewState)
        xmlWriter.newline()
        self._writeFlagsToXML(xmlWriter)
        xmlWriter.simpletag("MarkIndex", value=self.MarkIndex)
        xmlWriter.newline()
        xmlWriter.simpletag("CurrentIndex", value=self.CurrentIndex)
        xmlWriter.newline()
        xmlWriter.endtag(name)
        xmlWriter.newline()

    def fromXML(self, name, attrs, content, font):
        self.NewState = self.ReservedFlags = 0
        self.SetMark = self.DontAdvance = False
        self.MarkIndex, self.CurrentIndex = 0xFFFF, 0xFFFF
        content = [t for t in content if isinstance(t, tuple)]
        for eltName, eltAttrs, eltContent in content:
            if eltName == "NewState":
                self.NewState = safeEval(eltAttrs["value"])
            elif eltName == "Flags":
                for flag in eltAttrs["value"].split(","):
                    self._setFlag(flag.strip())
            elif eltName == "ReservedFlags":
                self.ReservedFlags = safeEval(eltAttrs["value"])
            elif eltName == "MarkIndex":
                self.MarkIndex = safeEval(eltAttrs["value"])
            elif eltName == "CurrentIndex":
                self.CurrentIndex = safeEval(eltAttrs["value"])


class LigAction(object):
    def __init__(self):
        self.Store = False
        # GlyphIndexDelta is a (possibly negative) delta that gets
        # added to the glyph ID at the top of the AAT runtime
        # execution stack. It is *not* a byte offset into the
        # morx table. The result of the addition, which is performed
        # at run time by the shaping engine, is an index into
        # the ligature components table. See 'morx' specification.
        # In the AAT specification, this field is called Offset;
        # but its meaning is quite different from other offsets
        # in either AAT or OpenType, so we use a different name.
        self.GlyphIndexDelta = 0


class LigatureMorphAction(AATAction):
    staticSize = 6

    # 4 bytes for each of {action,ligComponents,ligatures}Offset
    actionHeaderSize = 12

    _FLAGS = ["SetComponent", "DontAdvance"]

    def __init__(self):
        self.NewState = 0
        self.SetComponent, self.DontAdvance = False, False
        self.ReservedFlags = 0
        self.Actions = []

    def compile(self, writer, font, actionIndex):
        assert actionIndex is not None
        writer.writeUShort(self.NewState)
        flags = self.ReservedFlags
        if self.SetComponent:
            flags |= 0x8000
        if self.DontAdvance:
            flags |= 0x4000
        if len(self.Actions) > 0:
            flags |= 0x2000
        writer.writeUShort(flags)
        if len(self.Actions) > 0:
            actions = self.compileLigActions()
            writer.writeUShort(actionIndex[actions])
        else:
            writer.writeUShort(0)

    def decompile(self, reader, font, actionReader):
        assert actionReader is not None
        self.NewState = reader.readUShort()
        flags = reader.readUShort()
        self.SetComponent = bool(flags & 0x8000)
        self.DontAdvance = bool(flags & 0x4000)
        performAction = bool(flags & 0x2000)
        # As of 2017-09-12, the 'morx' specification says that
        # the reserved bitmask in ligature subtables is 0x3FFF.
        # However, the specification also defines a flag 0x2000,
        # so the reserved value should actually be 0x1FFF.
        # TODO: Report this specification bug to Apple.
        self.ReservedFlags = flags & 0x1FFF
        actionIndex = reader.readUShort()
        if performAction:
            self.Actions = self._decompileLigActions(actionReader, actionIndex)
        else:
            self.Actions = []

    @staticmethod
    def compileActions(font, states):
        result, actions, actionIndex = b"", set(), {}
        for state in states:
            for _glyphClass, trans in state.Transitions.items():
                actions.add(trans.compileLigActions())
        # Sort the compiled actions in decreasing order of
        # length, so that the longer sequence come before the
        # shorter ones.  For each compiled action ABCD, its
        # suffixes BCD, CD, and D do not be encoded separately
        # (in case they occur); instead, we can just store an
        # index that points into the middle of the longer
        # sequence. Every compiled AAT ligature sequence is
        # terminated with an end-of-sequence flag, which can
        # only be set on the last element of the sequence.
        # Therefore, it is sufficient to consider just the
        # suffixes.
        for a in sorted(actions, key=lambda x: (-len(x), x)):
            if a not in actionIndex:
                for i in range(0, len(a), 4):
                    suffix = a[i:]
                    suffixIndex = (len(result) + i) // 4
                    actionIndex.setdefault(suffix, suffixIndex)
                result += a
        result = pad(result, 4)
        return (result, actionIndex)

    def compileLigActions(self):
        result = []
        for i, action in enumerate(self.Actions):
            last = i == len(self.Actions) - 1
            value = action.GlyphIndexDelta & 0x3FFFFFFF
            value |= 0x80000000 if last else 0
            value |= 0x40000000 if action.Store else 0
            result.append(struct.pack(">L", value))
        return bytesjoin(result)

    def _decompileLigActions(self, actionReader, actionIndex):
        actions = []
        last = False
        reader = actionReader.getSubReader(actionReader.pos + actionIndex * 4)
        while not last:
            value = reader.readULong()
            last = bool(value & 0x80000000)
            action = LigAction()
            actions.append(action)
            action.Store = bool(value & 0x40000000)
            delta = value & 0x3FFFFFFF
            if delta >= 0x20000000:  # sign-extend 30-bit value
                delta = -0x40000000 + delta
            action.GlyphIndexDelta = delta
        return actions

    def fromXML(self, name, attrs, content, font):
        self.NewState = self.ReservedFlags = 0
        self.SetComponent = self.DontAdvance = False
        self.ReservedFlags = 0
        self.Actions = []
        content = [t for t in content if isinstance(t, tuple)]
        for eltName, eltAttrs, eltContent in content:
            if eltName == "NewState":
                self.NewState = safeEval(eltAttrs["value"])
            elif eltName == "Flags":
                for flag in eltAttrs["value"].split(","):
                    self._setFlag(flag.strip())
            elif eltName == "ReservedFlags":
                self.ReservedFlags = safeEval(eltAttrs["value"])
            elif eltName == "Action":
                action = LigAction()
                flags = eltAttrs.get("Flags", "").split(",")
                flags = [f.strip() for f in flags]
                action.Store = "Store" in flags
                action.GlyphIndexDelta = safeEval(eltAttrs["GlyphIndexDelta"])
                self.Actions.append(action)

    def toXML(self, xmlWriter, font, attrs, name):
        xmlWriter.begintag(name, **attrs)
        xmlWriter.newline()
        xmlWriter.simpletag("NewState", value=self.NewState)
        xmlWriter.newline()
        self._writeFlagsToXML(xmlWriter)
        for action in self.Actions:
            attribs = [("GlyphIndexDelta", action.GlyphIndexDelta)]
            if action.Store:
                attribs.append(("Flags", "Store"))
            xmlWriter.simpletag("Action", attribs)
            xmlWriter.newline()
        xmlWriter.endtag(name)
        xmlWriter.newline()


class InsertionMorphAction(AATAction):
    staticSize = 8
    actionHeaderSize = 4  # 4 bytes for actionOffset
    _FLAGS = [
        "SetMark",
        "DontAdvance",
        "CurrentIsKashidaLike",
        "MarkedIsKashidaLike",
        "CurrentInsertBefore",
        "MarkedInsertBefore",
    ]

    def __init__(self):
        self.NewState = 0
        for flag in self._FLAGS:
            setattr(self, flag, False)
        self.ReservedFlags = 0
        self.CurrentInsertionAction, self.MarkedInsertionAction = [], []

    def compile(self, writer, font, actionIndex):
        assert actionIndex is not None
        writer.writeUShort(self.NewState)
        flags = self.ReservedFlags
        if self.SetMark:
            flags |= 0x8000
        if self.DontAdvance:
            flags |= 0x4000
        if self.CurrentIsKashidaLike:
            flags |= 0x2000
        if self.MarkedIsKashidaLike:
            flags |= 0x1000
        if self.CurrentInsertBefore:
            flags |= 0x0800
        if self.MarkedInsertBefore:
            flags |= 0x0400
        flags |= len(self.CurrentInsertionAction) << 5
        flags |= len(self.MarkedInsertionAction)
        writer.writeUShort(flags)
        if len(self.CurrentInsertionAction) > 0:
            currentIndex = actionIndex[tuple(self.CurrentInsertionAction)]
        else:
            currentIndex = 0xFFFF
        writer.writeUShort(currentIndex)
        if len(self.MarkedInsertionAction) > 0:
            markedIndex = actionIndex[tuple(self.MarkedInsertionAction)]
        else:
            markedIndex = 0xFFFF
        writer.writeUShort(markedIndex)

    def decompile(self, reader, font, actionReader):
        assert actionReader is not None
        self.NewState = reader.readUShort()
        flags = reader.readUShort()
        self.SetMark = bool(flags & 0x8000)
        self.DontAdvance = bool(flags & 0x4000)
        self.CurrentIsKashidaLike = bool(flags & 0x2000)
        self.MarkedIsKashidaLike = bool(flags & 0x1000)
        self.CurrentInsertBefore = bool(flags & 0x0800)
        self.MarkedInsertBefore = bool(flags & 0x0400)
        self.CurrentInsertionAction = self._decompileInsertionAction(
            actionReader, font, index=reader.readUShort(), count=((flags & 0x03E0) >> 5)
        )
        self.MarkedInsertionAction = self._decompileInsertionAction(
            actionReader, font, index=reader.readUShort(), count=(flags & 0x001F)
        )

    def _decompileInsertionAction(self, actionReader, font, index, count):
        if index == 0xFFFF or count == 0:
            return []
        reader = actionReader.getSubReader(actionReader.pos + index * 2)
        return font.getGlyphNameMany(reader.readUShortArray(count))

    def toXML(self, xmlWriter, font, attrs, name):
        xmlWriter.begintag(name, **attrs)
        xmlWriter.newline()
        xmlWriter.simpletag("NewState", value=self.NewState)
        xmlWriter.newline()
        self._writeFlagsToXML(xmlWriter)
        for g in self.CurrentInsertionAction:
            xmlWriter.simpletag("CurrentInsertionAction", glyph=g)
            xmlWriter.newline()
        for g in self.MarkedInsertionAction:
            xmlWriter.simpletag("MarkedInsertionAction", glyph=g)
            xmlWriter.newline()
        xmlWriter.endtag(name)
        xmlWriter.newline()

    def fromXML(self, name, attrs, content, font):
        self.__init__()
        content = [t for t in content if isinstance(t, tuple)]
        for eltName, eltAttrs, eltContent in content:
            if eltName == "NewState":
                self.NewState = safeEval(eltAttrs["value"])
            elif eltName == "Flags":
                for flag in eltAttrs["value"].split(","):
                    self._setFlag(flag.strip())
            elif eltName == "CurrentInsertionAction":
                self.CurrentInsertionAction.append(eltAttrs["glyph"])
            elif eltName == "MarkedInsertionAction":
                self.MarkedInsertionAction.append(eltAttrs["glyph"])
            else:
                assert False, eltName

    @staticmethod
    def compileActions(font, states):
        actions, actionIndex, result = set(), {}, b""
        for state in states:
            for _glyphClass, trans in state.Transitions.items():
                if trans.CurrentInsertionAction is not None:
                    actions.add(tuple(trans.CurrentInsertionAction))
                if trans.MarkedInsertionAction is not None:
                    actions.add(tuple(trans.MarkedInsertionAction))
        # Sort the compiled actions in decreasing order of
        # length, so that the longer sequence come before the
        # shorter ones.
        for action in sorted(actions, key=lambda x: (-len(x), x)):
            # We insert all sub-sequences of the action glyph sequence
            # into actionIndex. For example, if one action triggers on
            # glyph sequence [A, B, C, D, E] and another action triggers
            # on [C, D], we return result=[A, B, C, D, E] (as list of
            # encoded glyph IDs), and actionIndex={('A','B','C','D','E'): 0,
            # ('C','D'): 2}.
            if action in actionIndex:
                continue
            for start in range(0, len(action)):
                startIndex = (len(result) // 2) + start
                for limit in range(start, len(action)):
                    glyphs = action[start : limit + 1]
                    actionIndex.setdefault(glyphs, startIndex)
            for glyph in action:
                glyphID = font.getGlyphID(glyph)
                result += struct.pack(">H", glyphID)
        return result, actionIndex


class FeatureParams(BaseTable):
    def compile(self, writer, font):
        assert (
            featureParamTypes.get(writer["FeatureTag"]) == self.__class__
        ), "Wrong FeatureParams type for feature '%s': %s" % (
            writer["FeatureTag"],
            self.__class__.__name__,
        )
        BaseTable.compile(self, writer, font)

    def toXML(self, xmlWriter, font, attrs=None, name=None):
        BaseTable.toXML(self, xmlWriter, font, attrs, name=self.__class__.__name__)


class FeatureParamsSize(FeatureParams):
    pass


class FeatureParamsStylisticSet(FeatureParams):
    pass


class FeatureParamsCharacterVariants(FeatureParams):
    pass


class Coverage(FormatSwitchingBaseTable):
    # manual implementation to get rid of glyphID dependencies

    def populateDefaults(self, propagator=None):
        if not hasattr(self, "glyphs"):
            self.glyphs = []

    def postRead(self, rawTable, font):
        if self.Format == 1:
            self.glyphs = rawTable["GlyphArray"]
        elif self.Format == 2:
            glyphs = self.glyphs = []
            ranges = rawTable["RangeRecord"]
            # Some SIL fonts have coverage entries that don't have sorted
            # StartCoverageIndex.  If it is so, fixup and warn.  We undo
            # this when writing font out.
            sorted_ranges = sorted(ranges, key=lambda a: a.StartCoverageIndex)
            if ranges != sorted_ranges:
                log.warning("GSUB/GPOS Coverage is not sorted by glyph ids.")
                ranges = sorted_ranges
            del sorted_ranges
            for r in ranges:
                start = r.Start
                end = r.End
                startID = font.getGlyphID(start)
                endID = font.getGlyphID(end) + 1
                glyphs.extend(font.getGlyphNameMany(range(startID, endID)))
        else:
            self.glyphs = []
            log.warning("Unknown Coverage format: %s", self.Format)
        del self.Format  # Don't need this anymore

    def preWrite(self, font):
        glyphs = getattr(self, "glyphs", None)
        if glyphs is None:
            glyphs = self.glyphs = []
        format = 1
        rawTable = {"GlyphArray": glyphs}
        if glyphs:
            # find out whether Format 2 is more compact or not
            glyphIDs = font.getGlyphIDMany(glyphs)
            brokenOrder = sorted(glyphIDs) != glyphIDs

            last = glyphIDs[0]
            ranges = [[last]]
            for glyphID in glyphIDs[1:]:
                if glyphID != last + 1:
                    ranges[-1].append(last)
                    ranges.append([glyphID])
                last = glyphID
            ranges[-1].append(last)

            if brokenOrder or len(ranges) * 3 < len(glyphs):  # 3 words vs. 1 word
                # Format 2 is more compact
                index = 0
                for i, (start, end) in enumerate(ranges):
                    r = RangeRecord()
                    r.StartID = start
                    r.Start = font.getGlyphName(start)
                    r.End = font.getGlyphName(end)
                    r.StartCoverageIndex = index
                    ranges[i] = r
                    index = index + end - start + 1
                if brokenOrder:
                    log.warning("GSUB/GPOS Coverage is not sorted by glyph ids.")
                    ranges.sort(key=lambda a: a.StartID)
                for r in ranges:
                    del r.StartID
                format = 2
                rawTable = {"RangeRecord": ranges}
            # else:
            # 	fallthrough; Format 1 is more compact
        self.Format = format
        return rawTable

    def toXML2(self, xmlWriter, font):
        for glyphName in getattr(self, "glyphs", []):
            xmlWriter.simpletag("Glyph", value=glyphName)
            xmlWriter.newline()

    def fromXML(self, name, attrs, content, font):
        glyphs = getattr(self, "glyphs", None)
        if glyphs is None:
            glyphs = []
            self.glyphs = glyphs
        glyphs.append(attrs["value"])


# The special 0xFFFFFFFF delta-set index is used to indicate that there
# is no variation data in the ItemVariationStore for a given variable field
NO_VARIATION_INDEX = 0xFFFFFFFF


class DeltaSetIndexMap(getFormatSwitchingBaseTableClass("uint8")):
    def populateDefaults(self, propagator=None):
        if not hasattr(self, "mapping"):
            self.mapping = []

    def postRead(self, rawTable, font):
        assert (rawTable["EntryFormat"] & 0xFFC0) == 0
        self.mapping = rawTable["mapping"]

    @staticmethod
    def getEntryFormat(mapping):
        ored = 0
        for idx in mapping:
            ored |= idx

        inner = ored & 0xFFFF
        innerBits = 0
        while inner:
            innerBits += 1
            inner >>= 1
        innerBits = max(innerBits, 1)
        assert innerBits <= 16

        ored = (ored >> (16 - innerBits)) | (ored & ((1 << innerBits) - 1))
        if ored <= 0x000000FF:
            entrySize = 1
        elif ored <= 0x0000FFFF:
            entrySize = 2
        elif ored <= 0x00FFFFFF:
            entrySize = 3
        else:
            entrySize = 4

        return ((entrySize - 1) << 4) | (innerBits - 1)

    def preWrite(self, font):
        mapping = getattr(self, "mapping", None)
        if mapping is None:
            mapping = self.mapping = []
        self.Format = 1 if len(mapping) > 0xFFFF else 0
        rawTable = self.__dict__.copy()
        rawTable["MappingCount"] = len(mapping)
        rawTable["EntryFormat"] = self.getEntryFormat(mapping)
        return rawTable

    def toXML2(self, xmlWriter, font):
        # Make xml dump less verbose, by omitting no-op entries like:
        #   <Map index="..." outer="65535" inner="65535"/>
        xmlWriter.comment("Omitted values default to 0xFFFF/0xFFFF (no variations)")
        xmlWriter.newline()
        for i, value in enumerate(getattr(self, "mapping", [])):
            attrs = [("index", i)]
            if value != NO_VARIATION_INDEX:
                attrs.extend(
                    [
                        ("outer", value >> 16),
                        ("inner", value & 0xFFFF),
                    ]
                )
            xmlWriter.simpletag("Map", attrs)
            xmlWriter.newline()

    def fromXML(self, name, attrs, content, font):
        mapping = getattr(self, "mapping", None)
        if mapping is None:
            self.mapping = mapping = []
        index = safeEval(attrs["index"])
        outer = safeEval(attrs.get("outer", "0xFFFF"))
        inner = safeEval(attrs.get("inner", "0xFFFF"))
        assert inner <= 0xFFFF
        mapping.insert(index, (outer << 16) | inner)

    def __getitem__(self, i):
        return self.mapping[i] if i < len(self.mapping) else NO_VARIATION_INDEX


class VarIdxMap(BaseTable):
    def populateDefaults(self, propagator=None):
        if not hasattr(self, "mapping"):
            self.mapping = {}

    def postRead(self, rawTable, font):
        assert (rawTable["EntryFormat"] & 0xFFC0) == 0
        glyphOrder = font.getGlyphOrder()
        mapList = rawTable["mapping"]
        mapList.extend([mapList[-1]] * (len(glyphOrder) - len(mapList)))
        self.mapping = dict(zip(glyphOrder, mapList))

    def preWrite(self, font):
        mapping = getattr(self, "mapping", None)
        if mapping is None:
            mapping = self.mapping = {}

        glyphOrder = font.getGlyphOrder()
        mapping = [mapping[g] for g in glyphOrder]
        while len(mapping) > 1 and mapping[-2] == mapping[-1]:
            del mapping[-1]

        rawTable = {"mapping": mapping}
        rawTable["MappingCount"] = len(mapping)
        rawTable["EntryFormat"] = DeltaSetIndexMap.getEntryFormat(mapping)
        return rawTable

    def toXML2(self, xmlWriter, font):
        for glyph, value in sorted(getattr(self, "mapping", {}).items()):
            attrs = (
                ("glyph", glyph),
                ("outer", value >> 16),
                ("inner", value & 0xFFFF),
            )
            xmlWriter.simpletag("Map", attrs)
            xmlWriter.newline()

    def fromXML(self, name, attrs, content, font):
        mapping = getattr(self, "mapping", None)
        if mapping is None:
            mapping = {}
            self.mapping = mapping
        try:
            glyph = attrs["glyph"]
        except:  # https://github.com/fonttools/fonttools/commit/21cbab8ce9ded3356fef3745122da64dcaf314e9#commitcomment-27649836
            glyph = font.getGlyphOrder()[attrs["index"]]
        outer = safeEval(attrs["outer"])
        inner = safeEval(attrs["inner"])
        assert inner <= 0xFFFF
        mapping[glyph] = (outer << 16) | inner

    def __getitem__(self, glyphName):
        return self.mapping.get(glyphName, NO_VARIATION_INDEX)


class VarRegionList(BaseTable):
    def preWrite(self, font):
        # The OT spec says VarStore.VarRegionList.RegionAxisCount should always
        # be equal to the fvar.axisCount, and OTS < v8.0.0 enforces this rule
        # even when the VarRegionList is empty. We can't treat RegionAxisCount
        # like a normal propagated count (== len(Region[i].VarRegionAxis)),
        # otherwise it would default to 0 if VarRegionList is empty.
        # Thus, we force it to always be equal to fvar.axisCount.
        # https://github.com/khaledhosny/ots/pull/192
        fvarTable = font.get("fvar")
        if fvarTable:
            self.RegionAxisCount = len(fvarTable.axes)
        return {
            **self.__dict__,
            "RegionAxisCount": CountReference(self.__dict__, "RegionAxisCount"),
        }


class SingleSubst(FormatSwitchingBaseTable):
    def populateDefaults(self, propagator=None):
        if not hasattr(self, "mapping"):
            self.mapping = {}

    def postRead(self, rawTable, font):
        mapping = {}
        input = _getGlyphsFromCoverageTable(rawTable["Coverage"])
        if self.Format == 1:
            delta = rawTable["DeltaGlyphID"]
            inputGIDS = font.getGlyphIDMany(input)
            outGIDS = [(glyphID + delta) % 65536 for glyphID in inputGIDS]
            outNames = font.getGlyphNameMany(outGIDS)
            for inp, out in zip(input, outNames):
                mapping[inp] = out
        elif self.Format == 2:
            assert (
                len(input) == rawTable["GlyphCount"]
            ), "invalid SingleSubstFormat2 table"
            subst = rawTable["Substitute"]
            for inp, sub in zip(input, subst):
                mapping[inp] = sub
        else:
            assert 0, "unknown format: %s" % self.Format
        self.mapping = mapping
        del self.Format  # Don't need this anymore

    def preWrite(self, font):
        mapping = getattr(self, "mapping", None)
        if mapping is None:
            mapping = self.mapping = {}
        items = list(mapping.items())
        getGlyphID = font.getGlyphID
        gidItems = [(getGlyphID(a), getGlyphID(b)) for a, b in items]
        sortableItems = sorted(zip(gidItems, items))

        # figure out format
        format = 2
        delta = None
        for inID, outID in gidItems:
            if delta is None:
                delta = (outID - inID) % 65536

            if (inID + delta) % 65536 != outID:
                break
        else:
            if delta is None:
                # the mapping is empty, better use format 2
                format = 2
            else:
                format = 1

        rawTable = {}
        self.Format = format
        cov = Coverage()
        input = [item[1][0] for item in sortableItems]
        subst = [item[1][1] for item in sortableItems]
        cov.glyphs = input
        rawTable["Coverage"] = cov
        if format == 1:
            assert delta is not None
            rawTable["DeltaGlyphID"] = delta
        else:
            rawTable["Substitute"] = subst
        return rawTable

    def toXML2(self, xmlWriter, font):
        items = sorted(self.mapping.items())
        for inGlyph, outGlyph in items:
            xmlWriter.simpletag("Substitution", [("in", inGlyph), ("out", outGlyph)])
            xmlWriter.newline()

    def fromXML(self, name, attrs, content, font):
        mapping = getattr(self, "mapping", None)
        if mapping is None:
            mapping = {}
            self.mapping = mapping
        mapping[attrs["in"]] = attrs["out"]


class MultipleSubst(FormatSwitchingBaseTable):
    def populateDefaults(self, propagator=None):
        if not hasattr(self, "mapping"):
            self.mapping = {}

    def postRead(self, rawTable, font):
        mapping = {}
        if self.Format == 1:
            glyphs = _getGlyphsFromCoverageTable(rawTable["Coverage"])
            subst = [s.Substitute for s in rawTable["Sequence"]]
            mapping = dict(zip(glyphs, subst))
        else:
            assert 0, "unknown format: %s" % self.Format
        self.mapping = mapping
        del self.Format  # Don't need this anymore

    def preWrite(self, font):
        mapping = getattr(self, "mapping", None)
        if mapping is None:
            mapping = self.mapping = {}
        cov = Coverage()
        cov.glyphs = sorted(list(mapping.keys()), key=font.getGlyphID)
        self.Format = 1
        rawTable = {
            "Coverage": cov,
            "Sequence": [self.makeSequence_(mapping[glyph]) for glyph in cov.glyphs],
        }
        return rawTable

    def toXML2(self, xmlWriter, font):
        items = sorted(self.mapping.items())
        for inGlyph, outGlyphs in items:
            out = ",".join(outGlyphs)
            xmlWriter.simpletag("Substitution", [("in", inGlyph), ("out", out)])
            xmlWriter.newline()

    def fromXML(self, name, attrs, content, font):
        mapping = getattr(self, "mapping", None)
        if mapping is None:
            mapping = {}
            self.mapping = mapping

        # TTX v3.0 and earlier.
        if name == "Coverage":
            self.old_coverage_ = []
            for element in content:
                if not isinstance(element, tuple):
                    continue
                element_name, element_attrs, _ = element
                if element_name == "Glyph":
                    self.old_coverage_.append(element_attrs["value"])
            return
        if name == "Sequence":
            index = int(attrs.get("index", len(mapping)))
            glyph = self.old_coverage_[index]
            glyph_mapping = mapping[glyph] = []
            for element in content:
                if not isinstance(element, tuple):
                    continue
                element_name, element_attrs, _ = element
                if element_name == "Substitute":
                    glyph_mapping.append(element_attrs["value"])
            return

            # TTX v3.1 and later.
        outGlyphs = attrs["out"].split(",") if attrs["out"] else []
        mapping[attrs["in"]] = [g.strip() for g in outGlyphs]

    @staticmethod
    def makeSequence_(g):
        seq = Sequence()
        seq.Substitute = g
        return seq


class ClassDef(FormatSwitchingBaseTable):
    def populateDefaults(self, propagator=None):
        if not hasattr(self, "classDefs"):
            self.classDefs = {}

    def postRead(self, rawTable, font):
        classDefs = {}

        if self.Format == 1:
            start = rawTable["StartGlyph"]
            classList = rawTable["ClassValueArray"]
            startID = font.getGlyphID(start)
            endID = startID + len(classList)
            glyphNames = font.getGlyphNameMany(range(startID, endID))
            for glyphName, cls in zip(glyphNames, classList):
                if cls:
                    classDefs[glyphName] = cls

        elif self.Format == 2:
            records = rawTable["ClassRangeRecord"]
            for rec in records:
                cls = rec.Class
                if not cls:
                    continue
                start = rec.Start
                end = rec.End
                startID = font.getGlyphID(start)
                endID = font.getGlyphID(end) + 1
                glyphNames = font.getGlyphNameMany(range(startID, endID))
                for glyphName in glyphNames:
                    classDefs[glyphName] = cls
        else:
            log.warning("Unknown ClassDef format: %s", self.Format)
        self.classDefs = classDefs
        del self.Format  # Don't need this anymore

    def _getClassRanges(self, font):
        classDefs = getattr(self, "classDefs", None)
        if classDefs is None:
            self.classDefs = {}
            return
        getGlyphID = font.getGlyphID
        items = []
        for glyphName, cls in classDefs.items():
            if not cls:
                continue
            items.append((getGlyphID(glyphName), glyphName, cls))
        if items:
            items.sort()
            last, lastName, lastCls = items[0]
            ranges = [[lastCls, last, lastName]]
            for glyphID, glyphName, cls in items[1:]:
                if glyphID != last + 1 or cls != lastCls:
                    ranges[-1].extend([last, lastName])
                    ranges.append([cls, glyphID, glyphName])
                last = glyphID
                lastName = glyphName
                lastCls = cls
            ranges[-1].extend([last, lastName])
            return ranges

    def preWrite(self, font):
        format = 2
        rawTable = {"ClassRangeRecord": []}
        ranges = self._getClassRanges(font)
        if ranges:
            startGlyph = ranges[0][1]
            endGlyph = ranges[-1][3]
            glyphCount = endGlyph - startGlyph + 1
            if len(ranges) * 3 < glyphCount + 1:
                # Format 2 is more compact
                for i, (cls, start, startName, end, endName) in enumerate(ranges):
                    rec = ClassRangeRecord()
                    rec.Start = startName
                    rec.End = endName
                    rec.Class = cls
                    ranges[i] = rec
                format = 2
                rawTable = {"ClassRangeRecord": ranges}
            else:
                # Format 1 is more compact
                startGlyphName = ranges[0][2]
                classes = [0] * glyphCount
                for cls, start, startName, end, endName in ranges:
                    for g in range(start - startGlyph, end - startGlyph + 1):
                        classes[g] = cls
                format = 1
                rawTable = {"StartGlyph": startGlyphName, "ClassValueArray": classes}
        self.Format = format
        return rawTable

    def toXML2(self, xmlWriter, font):
        items = sorted(self.classDefs.items())
        for glyphName, cls in items:
            xmlWriter.simpletag("ClassDef", [("glyph", glyphName), ("class", cls)])
            xmlWriter.newline()

    def fromXML(self, name, attrs, content, font):
        classDefs = getattr(self, "classDefs", None)
        if classDefs is None:
            classDefs = {}
            self.classDefs = classDefs
        classDefs[attrs["glyph"]] = int(attrs["class"])


class AlternateSubst(FormatSwitchingBaseTable):
    def populateDefaults(self, propagator=None):
        if not hasattr(self, "alternates"):
            self.alternates = {}

    def postRead(self, rawTable, font):
        alternates = {}
        if self.Format == 1:
            input = _getGlyphsFromCoverageTable(rawTable["Coverage"])
            alts = rawTable["AlternateSet"]
            assert len(input) == len(alts)
            for inp, alt in zip(input, alts):
                alternates[inp] = alt.Alternate
        else:
            assert 0, "unknown format: %s" % self.Format
        self.alternates = alternates
        del self.Format  # Don't need this anymore

    def preWrite(self, font):
        self.Format = 1
        alternates = getattr(self, "alternates", None)
        if alternates is None:
            alternates = self.alternates = {}
        items = list(alternates.items())
        for i, (glyphName, set) in enumerate(items):
            items[i] = font.getGlyphID(glyphName), glyphName, set
        items.sort()
        cov = Coverage()
        cov.glyphs = [item[1] for item in items]
        alternates = []
        setList = [item[-1] for item in items]
        for set in setList:
            alts = AlternateSet()
            alts.Alternate = set
            alternates.append(alts)
        # a special case to deal with the fact that several hundred Adobe Japan1-5
        # CJK fonts will overflow an offset if the coverage table isn't pushed to the end.
        # Also useful in that when splitting a sub-table because of an offset overflow
        # I don't need to calculate the change in the subtable offset due to the change in the coverage table size.
        # Allows packing more rules in subtable.
        self.sortCoverageLast = 1
        return {"Coverage": cov, "AlternateSet": alternates}

    def toXML2(self, xmlWriter, font):
        items = sorted(self.alternates.items())
        for glyphName, alternates in items:
            xmlWriter.begintag("AlternateSet", glyph=glyphName)
            xmlWriter.newline()
            for alt in alternates:
                xmlWriter.simpletag("Alternate", glyph=alt)
                xmlWriter.newline()
            xmlWriter.endtag("AlternateSet")
            xmlWriter.newline()

    def fromXML(self, name, attrs, content, font):
        alternates = getattr(self, "alternates", None)
        if alternates is None:
            alternates = {}
            self.alternates = alternates
        glyphName = attrs["glyph"]
        set = []
        alternates[glyphName] = set
        for element in content:
            if not isinstance(element, tuple):
                continue
            name, attrs, content = element
            set.append(attrs["glyph"])


class LigatureSubst(FormatSwitchingBaseTable):
    def populateDefaults(self, propagator=None):
        if not hasattr(self, "ligatures"):
            self.ligatures = {}

    def postRead(self, rawTable, font):
        ligatures = {}
        if self.Format == 1:
            input = _getGlyphsFromCoverageTable(rawTable["Coverage"])
            ligSets = rawTable["LigatureSet"]
            assert len(input) == len(ligSets)
            for i, inp in enumerate(input):
                ligatures[inp] = ligSets[i].Ligature
        else:
            assert 0, "unknown format: %s" % self.Format
        self.ligatures = ligatures
        del self.Format  # Don't need this anymore

    @staticmethod
    def _getLigatureSortKey(components):
        # Computes a key for ordering ligatures in a GSUB Type-4 lookup.

        # When building the OpenType lookup, we need to make sure that
        # the longest sequence of components is listed first, so we
        # use the negative length as the key for sorting.
        # Note, we no longer need to worry about deterministic order because the
        # ligature mapping `dict` remembers the insertion order, and this in
        # turn depends on the order in which the ligatures are written in the FEA.
        # Since python sort algorithm is stable, the ligatures of equal length
        # will keep the relative order in which they appear in the feature file.
        # For example, given the following ligatures (all starting with 'f' and
        # thus belonging to the same LigatureSet):
        #
        #   feature liga {
        #     sub f i by f_i;
        #     sub f f f by f_f_f;
        #     sub f f by f_f;
        #     sub f f i by f_f_i;
        #   } liga;
        #
        # this should sort to: f_f_f, f_f_i, f_i, f_f
        # This is also what fea-rs does, see:
        # https://github.com/adobe-type-tools/afdko/issues/1727
        # https://github.com/fonttools/fonttools/issues/3428
        # https://github.com/googlefonts/fontc/pull/680
        return -len(components)

    def preWrite(self, font):
        self.Format = 1
        ligatures = getattr(self, "ligatures", None)
        if ligatures is None:
            ligatures = self.ligatures = {}

        if ligatures and isinstance(next(iter(ligatures)), tuple):
            # New high-level API in v3.1 and later.  Note that we just support compiling this
            # for now.  We don't load to this API, and don't do XML with it.

            # ligatures is map from components-sequence to lig-glyph
            newLigatures = dict()
            for comps in sorted(ligatures.keys(), key=self._getLigatureSortKey):
                ligature = Ligature()
                ligature.Component = comps[1:]
                ligature.CompCount = len(comps)
                ligature.LigGlyph = ligatures[comps]
                newLigatures.setdefault(comps[0], []).append(ligature)
            ligatures = newLigatures

        items = list(ligatures.items())
        for i, (glyphName, set) in enumerate(items):
            items[i] = font.getGlyphID(glyphName), glyphName, set
        items.sort()
        cov = Coverage()
        cov.glyphs = [item[1] for item in items]

        ligSets = []
        setList = [item[-1] for item in items]
        for set in setList:
            ligSet = LigatureSet()
            ligs = ligSet.Ligature = []
            for lig in set:
                ligs.append(lig)
            ligSets.append(ligSet)
        # Useful in that when splitting a sub-table because of an offset overflow
        # I don't need to calculate the change in subtabl offset due to the coverage table size.
        # Allows packing more rules in subtable.
        self.sortCoverageLast = 1
        return {"Coverage": cov, "LigatureSet": ligSets}

    def toXML2(self, xmlWriter, font):
        items = sorted(self.ligatures.items())
        for glyphName, ligSets in items:
            xmlWriter.begintag("LigatureSet", glyph=glyphName)
            xmlWriter.newline()
            for lig in ligSets:
                xmlWriter.simpletag(
                    "Ligature", glyph=lig.LigGlyph, components=",".join(lig.Component)
                )
                xmlWriter.newline()
            xmlWriter.endtag("LigatureSet")
            xmlWriter.newline()

    def fromXML(self, name, attrs, content, font):
        ligatures = getattr(self, "ligatures", None)
        if ligatures is None:
            ligatures = {}
            self.ligatures = ligatures
        glyphName = attrs["glyph"]
        ligs = []
        ligatures[glyphName] = ligs
        for element in content:
            if not isinstance(element, tuple):
                continue
            name, attrs, content = element
            lig = Ligature()
            lig.LigGlyph = attrs["glyph"]
            components = attrs["components"]
            lig.Component = components.split(",") if components else []
            lig.CompCount = len(lig.Component)
            ligs.append(lig)


class COLR(BaseTable):
    def decompile(self, reader, font):
        # COLRv0 is exceptional in that LayerRecordCount appears *after* the
        # LayerRecordArray it counts, but the parser logic expects Count fields
        # to always precede the arrays. Here we work around this by parsing the
        # LayerRecordCount before the rest of the table, and storing it in
        # the reader's local state.
        subReader = reader.getSubReader(offset=0)
        for conv in self.getConverters():
            if conv.name != "LayerRecordCount":
                subReader.advance(conv.staticSize)
                continue
            reader[conv.name] = conv.read(subReader, font, tableDict={})
            break
        else:
            raise AssertionError("LayerRecordCount converter not found")
        return BaseTable.decompile(self, reader, font)

    def preWrite(self, font):
        # The writer similarly assumes Count values precede the things counted,
        # thus here we pre-initialize a CountReference; the actual count value
        # will be set to the lenght of the array by the time this is assembled.
        self.LayerRecordCount = None
        return {
            **self.__dict__,
            "LayerRecordCount": CountReference(self.__dict__, "LayerRecordCount"),
        }

    def computeClipBoxes(self, glyphSet: "_TTGlyphSet", quantization: int = 1):
        if self.Version == 0:
            return

        clips = {}
        for rec in self.BaseGlyphList.BaseGlyphPaintRecord:
            try:
                clipBox = rec.Paint.computeClipBox(self, glyphSet, quantization)
            except Exception as e:
                from fontTools.ttLib import TTLibError

                raise TTLibError(
                    f"Failed to compute COLR ClipBox for {rec.BaseGlyph!r}"
                ) from e

            if clipBox is not None:
                clips[rec.BaseGlyph] = clipBox

        hasClipList = hasattr(self, "ClipList") and self.ClipList is not None
        if not clips:
            if hasClipList:
                self.ClipList = None
        else:
            if not hasClipList:
                self.ClipList = ClipList()
                self.ClipList.Format = 1
            self.ClipList.clips = clips


class LookupList(BaseTable):
    @property
    def table(self):
        for l in self.Lookup:
            for st in l.SubTable:
                if type(st).__name__.endswith("Subst"):
                    return "GSUB"
                if type(st).__name__.endswith("Pos"):
                    return "GPOS"
        raise ValueError

    def toXML2(self, xmlWriter, font):
        if (
            not font
            or "Debg" not in font
            or LOOKUP_DEBUG_INFO_KEY not in font["Debg"].data
        ):
            return super().toXML2(xmlWriter, font)
        debugData = font["Debg"].data[LOOKUP_DEBUG_INFO_KEY][self.table]
        for conv in self.getConverters():
            if conv.repeat:
                value = getattr(self, conv.name, [])
                for lookupIndex, item in enumerate(value):
                    if str(lookupIndex) in debugData:
                        info = LookupDebugInfo(*debugData[str(lookupIndex)])
                        tag = info.location
                        if info.name:
                            tag = f"{info.name}: {tag}"
                        if info.feature:
                            script, language, feature = info.feature
                            tag = f"{tag} in {feature} ({script}/{language})"
                        xmlWriter.comment(tag)
                        xmlWriter.newline()

                    conv.xmlWrite(
                        xmlWriter, font, item, conv.name, [("index", lookupIndex)]
                    )
            else:
                if conv.aux and not eval(conv.aux, None, vars(self)):
                    continue
                value = getattr(
                    self, conv.name, None
                )  # TODO Handle defaults instead of defaulting to None!
                conv.xmlWrite(xmlWriter, font, value, conv.name, [])


class BaseGlyphRecordArray(BaseTable):
    def preWrite(self, font):
        self.BaseGlyphRecord = sorted(
            self.BaseGlyphRecord, key=lambda rec: font.getGlyphID(rec.BaseGlyph)
        )
        return self.__dict__.copy()


class BaseGlyphList(BaseTable):
    def preWrite(self, font):
        self.BaseGlyphPaintRecord = sorted(
            self.BaseGlyphPaintRecord, key=lambda rec: font.getGlyphID(rec.BaseGlyph)
        )
        return self.__dict__.copy()


class ClipBoxFormat(IntEnum):
    Static = 1
    Variable = 2

    def is_variable(self):
        return self is self.Variable

    def as_variable(self):
        return self.Variable


class ClipBox(getFormatSwitchingBaseTableClass("uint8")):
    formatEnum = ClipBoxFormat

    def as_tuple(self):
        return tuple(getattr(self, conv.name) for conv in self.getConverters())

    def __repr__(self):
        return f"{self.__class__.__name__}{self.as_tuple()}"


class ClipList(getFormatSwitchingBaseTableClass("uint8")):
    def populateDefaults(self, propagator=None):
        if not hasattr(self, "clips"):
            self.clips = {}

    def postRead(self, rawTable, font):
        clips = {}
        glyphOrder = font.getGlyphOrder()
        for i, rec in enumerate(rawTable["ClipRecord"]):
            if rec.StartGlyphID > rec.EndGlyphID:
                log.warning(
                    "invalid ClipRecord[%i].StartGlyphID (%i) > "
                    "EndGlyphID (%i); skipped",
                    i,
                    rec.StartGlyphID,
                    rec.EndGlyphID,
                )
                continue
            redefinedGlyphs = []
            missingGlyphs = []
            for glyphID in range(rec.StartGlyphID, rec.EndGlyphID + 1):
                try:
                    glyph = glyphOrder[glyphID]
                except IndexError:
                    missingGlyphs.append(glyphID)
                    continue
                if glyph not in clips:
                    clips[glyph] = copy.copy(rec.ClipBox)
                else:
                    redefinedGlyphs.append(glyphID)
            if redefinedGlyphs:
                log.warning(
                    "ClipRecord[%i] overlaps previous records; "
                    "ignoring redefined clip boxes for the "
                    "following glyph ID range: [%i-%i]",
                    i,
                    min(redefinedGlyphs),
                    max(redefinedGlyphs),
                )
            if missingGlyphs:
                log.warning(
                    "ClipRecord[%i] range references missing " "glyph IDs: [%i-%i]",
                    i,
                    min(missingGlyphs),
                    max(missingGlyphs),
                )
        self.clips = clips

    def groups(self):
        glyphsByClip = defaultdict(list)
        uniqueClips = {}
        for glyphName, clipBox in self.clips.items():
            key = clipBox.as_tuple()
            glyphsByClip[key].append(glyphName)
            if key not in uniqueClips:
                uniqueClips[key] = clipBox
        return {
            frozenset(glyphs): uniqueClips[key] for key, glyphs in glyphsByClip.items()
        }

    def preWrite(self, font):
        if not hasattr(self, "clips"):
            self.clips = {}
        clipBoxRanges = {}
        glyphMap = font.getReverseGlyphMap()
        for glyphs, clipBox in self.groups().items():
            glyphIDs = sorted(
                glyphMap[glyphName] for glyphName in glyphs if glyphName in glyphMap
            )
            if not glyphIDs:
                continue
            last = glyphIDs[0]
            ranges = [[last]]
            for glyphID in glyphIDs[1:]:
                if glyphID != last + 1:
                    ranges[-1].append(last)
                    ranges.append([glyphID])
                last = glyphID
            ranges[-1].append(last)
            for start, end in ranges:
                assert (start, end) not in clipBoxRanges
                clipBoxRanges[(start, end)] = clipBox

        clipRecords = []
        for (start, end), clipBox in sorted(clipBoxRanges.items()):
            record = ClipRecord()
            record.StartGlyphID = start
            record.EndGlyphID = end
            record.ClipBox = clipBox
            clipRecords.append(record)
        rawTable = {
            "ClipCount": len(clipRecords),
            "ClipRecord": clipRecords,
        }
        return rawTable

    def toXML(self, xmlWriter, font, attrs=None, name=None):
        tableName = name if name else self.__class__.__name__
        if attrs is None:
            attrs = []
        if hasattr(self, "Format"):
            attrs.append(("Format", self.Format))
        xmlWriter.begintag(tableName, attrs)
        xmlWriter.newline()
        # sort clips alphabetically to ensure deterministic XML dump
        for glyphs, clipBox in sorted(
            self.groups().items(), key=lambda item: min(item[0])
        ):
            xmlWriter.begintag("Clip")
            xmlWriter.newline()
            for glyphName in sorted(glyphs):
                xmlWriter.simpletag("Glyph", value=glyphName)
                xmlWriter.newline()
            xmlWriter.begintag("ClipBox", [("Format", clipBox.Format)])
            xmlWriter.newline()
            clipBox.toXML2(xmlWriter, font)
            xmlWriter.endtag("ClipBox")
            xmlWriter.newline()
            xmlWriter.endtag("Clip")
            xmlWriter.newline()
        xmlWriter.endtag(tableName)
        xmlWriter.newline()

    def fromXML(self, name, attrs, content, font):
        clips = getattr(self, "clips", None)
        if clips is None:
            self.clips = clips = {}
        assert name == "Clip"
        glyphs = []
        clipBox = None
        for elem in content:
            if not isinstance(elem, tuple):
                continue
            name, attrs, content = elem
            if name == "Glyph":
                glyphs.append(attrs["value"])
            elif name == "ClipBox":
                clipBox = ClipBox()
                clipBox.Format = safeEval(attrs["Format"])
                for elem in content:
                    if not isinstance(elem, tuple):
                        continue
                    name, attrs, content = elem
                    clipBox.fromXML(name, attrs, content, font)
        if clipBox:
            for glyphName in glyphs:
                clips[glyphName] = clipBox


class ExtendMode(IntEnum):
    PAD = 0
    REPEAT = 1
    REFLECT = 2


# Porter-Duff modes for COLRv1 PaintComposite:
# https://github.com/googlefonts/colr-gradients-spec/tree/off_sub_1#compositemode-enumeration
class CompositeMode(IntEnum):
    CLEAR = 0
    SRC = 1
    DEST = 2
    SRC_OVER = 3
    DEST_OVER = 4
    SRC_IN = 5
    DEST_IN = 6
    SRC_OUT = 7
    DEST_OUT = 8
    SRC_ATOP = 9
    DEST_ATOP = 10
    XOR = 11
    PLUS = 12
    SCREEN = 13
    OVERLAY = 14
    DARKEN = 15
    LIGHTEN = 16
    COLOR_DODGE = 17
    COLOR_BURN = 18
    HARD_LIGHT = 19
    SOFT_LIGHT = 20
    DIFFERENCE = 21
    EXCLUSION = 22
    MULTIPLY = 23
    HSL_HUE = 24
    HSL_SATURATION = 25
    HSL_COLOR = 26
    HSL_LUMINOSITY = 27


class PaintFormat(IntEnum):
    PaintColrLayers = 1
    PaintSolid = 2
    PaintVarSolid = 3
    PaintLinearGradient = 4
    PaintVarLinearGradient = 5
    PaintRadialGradient = 6
    PaintVarRadialGradient = 7
    PaintSweepGradient = 8
    PaintVarSweepGradient = 9
    PaintGlyph = 10
    PaintColrGlyph = 11
    PaintTransform = 12
    PaintVarTransform = 13
    PaintTranslate = 14
    PaintVarTranslate = 15
    PaintScale = 16
    PaintVarScale = 17
    PaintScaleAroundCenter = 18
    PaintVarScaleAroundCenter = 19
    PaintScaleUniform = 20
    PaintVarScaleUniform = 21
    PaintScaleUniformAroundCenter = 22
    PaintVarScaleUniformAroundCenter = 23
    PaintRotate = 24
    PaintVarRotate = 25
    PaintRotateAroundCenter = 26
    PaintVarRotateAroundCenter = 27
    PaintSkew = 28
    PaintVarSkew = 29
    PaintSkewAroundCenter = 30
    PaintVarSkewAroundCenter = 31
    PaintComposite = 32

    def is_variable(self):
        return self.name.startswith("PaintVar")

    def as_variable(self):
        if self.is_variable():
            return self
        try:
            return PaintFormat.__members__[f"PaintVar{self.name[5:]}"]
        except KeyError:
            return None


class Paint(getFormatSwitchingBaseTableClass("uint8")):
    formatEnum = PaintFormat

    def getFormatName(self):
        try:
            return self.formatEnum(self.Format).name
        except ValueError:
            raise NotImplementedError(f"Unknown Paint format: {self.Format}")

    def toXML(self, xmlWriter, font, attrs=None, name=None):
        tableName = name if name else self.__class__.__name__
        if attrs is None:
            attrs = []
        attrs.append(("Format", self.Format))
        xmlWriter.begintag(tableName, attrs)
        xmlWriter.comment(self.getFormatName())
        xmlWriter.newline()
        self.toXML2(xmlWriter, font)
        xmlWriter.endtag(tableName)
        xmlWriter.newline()

    def iterPaintSubTables(self, colr: COLR) -> Iterator[BaseTable.SubTableEntry]:
        if self.Format == PaintFormat.PaintColrLayers:
            # https://github.com/fonttools/fonttools/issues/2438: don't die when no LayerList exists
            layers = []
            if colr.LayerList is not None:
                layers = colr.LayerList.Paint
            yield from (
                BaseTable.SubTableEntry(name="Layers", value=v, index=i)
                for i, v in enumerate(
                    layers[self.FirstLayerIndex : self.FirstLayerIndex + self.NumLayers]
                )
            )
            return

        if self.Format == PaintFormat.PaintColrGlyph:
            for record in colr.BaseGlyphList.BaseGlyphPaintRecord:
                if record.BaseGlyph == self.Glyph:
                    yield BaseTable.SubTableEntry(name="BaseGlyph", value=record.Paint)
                    return
            else:
                raise KeyError(f"{self.Glyph!r} not in colr.BaseGlyphList")

        for conv in self.getConverters():
            if conv.tableClass is not None and issubclass(conv.tableClass, type(self)):
                value = getattr(self, conv.name)
                yield BaseTable.SubTableEntry(name=conv.name, value=value)

    def getChildren(self, colr) -> List["Paint"]:
        # this is kept for backward compatibility (e.g. it's used by the subsetter)
        return [p.value for p in self.iterPaintSubTables(colr)]

    def traverse(self, colr: COLR, callback):
        """Depth-first traversal of graph rooted at self, callback on each node."""
        if not callable(callback):
            raise TypeError("callback must be callable")

        for path in dfs_base_table(
            self, iter_subtables_fn=lambda paint: paint.iterPaintSubTables(colr)
        ):
            paint = path[-1].value
            callback(paint)

    def getTransform(self) -> Transform:
        if self.Format == PaintFormat.PaintTransform:
            t = self.Transform
            return Transform(t.xx, t.yx, t.xy, t.yy, t.dx, t.dy)
        elif self.Format == PaintFormat.PaintTranslate:
            return Identity.translate(self.dx, self.dy)
        elif self.Format == PaintFormat.PaintScale:
            return Identity.scale(self.scaleX, self.scaleY)
        elif self.Format == PaintFormat.PaintScaleAroundCenter:
            return (
                Identity.translate(self.centerX, self.centerY)
                .scale(self.scaleX, self.scaleY)
                .translate(-self.centerX, -self.centerY)
            )
        elif self.Format == PaintFormat.PaintScaleUniform:
            return Identity.scale(self.scale)
        elif self.Format == PaintFormat.PaintScaleUniformAroundCenter:
            return (
                Identity.translate(self.centerX, self.centerY)
                .scale(self.scale)
                .translate(-self.centerX, -self.centerY)
            )
        elif self.Format == PaintFormat.PaintRotate:
            return Identity.rotate(radians(self.angle))
        elif self.Format == PaintFormat.PaintRotateAroundCenter:
            return (
                Identity.translate(self.centerX, self.centerY)
                .rotate(radians(self.angle))
                .translate(-self.centerX, -self.centerY)
            )
        elif self.Format == PaintFormat.PaintSkew:
            return Identity.skew(radians(-self.xSkewAngle), radians(self.ySkewAngle))
        elif self.Format == PaintFormat.PaintSkewAroundCenter:
            return (
                Identity.translate(self.centerX, self.centerY)
                .skew(radians(-self.xSkewAngle), radians(self.ySkewAngle))
                .translate(-self.centerX, -self.centerY)
            )
        if PaintFormat(self.Format).is_variable():
            raise NotImplementedError(f"Variable Paints not supported: {self.Format}")

        return Identity

    def computeClipBox(
        self, colr: COLR, glyphSet: "_TTGlyphSet", quantization: int = 1
    ) -> Optional[ClipBox]:
        pen = ControlBoundsPen(glyphSet)
        for path in dfs_base_table(
            self, iter_subtables_fn=lambda paint: paint.iterPaintSubTables(colr)
        ):
            paint = path[-1].value
            if paint.Format == PaintFormat.PaintGlyph:
                transformation = reduce(
                    Transform.transform,
                    (st.value.getTransform() for st in path),
                    Identity,
                )
                glyphSet[paint.Glyph].draw(TransformPen(pen, transformation))

        if pen.bounds is None:
            return None

        cb = ClipBox()
        cb.Format = int(ClipBoxFormat.Static)
        cb.xMin, cb.yMin, cb.xMax, cb.yMax = quantizeRect(pen.bounds, quantization)
        return cb


# For each subtable format there is a class. However, we don't really distinguish
# between "field name" and "format name": often these are the same. Yet there's
# a whole bunch of fields with different names. The following dict is a mapping
# from "format name" to "field name". _buildClasses() uses this to create a
# subclass for each alternate field name.
#
_equivalents = {
    "PatchMap": ("IFT ", "IFTX"),
    "MarkArray": ("Mark1Array",),
    "LangSys": ("DefaultLangSys",),
    "Coverage": (
        "MarkCoverage",
        "BaseCoverage",
        "LigatureCoverage",
        "Mark1Coverage",
        "Mark2Coverage",
        "BacktrackCoverage",
        "InputCoverage",
        "LookAheadCoverage",
        "VertGlyphCoverage",
        "HorizGlyphCoverage",
        "TopAccentCoverage",
        "ExtendedShapeCoverage",
        "MathKernCoverage",
    ),
    "ClassDef": (
        "ClassDef1",
        "ClassDef2",
        "BacktrackClassDef",
        "InputClassDef",
        "LookAheadClassDef",
        "GlyphClassDef",
        "MarkAttachClassDef",
    ),
    "Anchor": (
        "EntryAnchor",
        "ExitAnchor",
        "BaseAnchor",
        "LigatureAnchor",
        "Mark2Anchor",
        "MarkAnchor",
    ),
    "Device": (
        "XPlaDevice",
        "YPlaDevice",
        "XAdvDevice",
        "YAdvDevice",
        "XDeviceTable",
        "YDeviceTable",
        "DeviceTable",
    ),
    "Axis": (
        "HorizAxis",
        "VertAxis",
    ),
    "MinMax": ("DefaultMinMax",),
    "BaseCoord": (
        "MinCoord",
        "MaxCoord",
    ),
    "JstfLangSys": ("DefJstfLangSys",),
    "JstfGSUBModList": (
        "ShrinkageEnableGSUB",
        "ShrinkageDisableGSUB",
        "ExtensionEnableGSUB",
        "ExtensionDisableGSUB",
    ),
    "JstfGPOSModList": (
        "ShrinkageEnableGPOS",
        "ShrinkageDisableGPOS",
        "ExtensionEnableGPOS",
        "ExtensionDisableGPOS",
    ),
    "JstfMax": (
        "ShrinkageJstfMax",
        "ExtensionJstfMax",
    ),
    "MathKern": (
        "TopRightMathKern",
        "TopLeftMathKern",
        "BottomRightMathKern",
        "BottomLeftMathKern",
    ),
    "MathGlyphConstruction": ("VertGlyphConstruction", "HorizGlyphConstruction"),
}

#
# OverFlow logic, to automatically create ExtensionLookups
# XXX This should probably move to otBase.py
#


def fixLookupOverFlows(ttf, overflowRecord):
    """Either the offset from the LookupList to a lookup overflowed, or
    an offset from a lookup to a subtable overflowed.

    The table layout is::

      GPSO/GUSB
              Script List
              Feature List
              LookUpList
                      Lookup[0] and contents
                              SubTable offset list
                                      SubTable[0] and contents
                                      ...
                                      SubTable[n] and contents
                      ...
                      Lookup[n] and contents
                              SubTable offset list
                                      SubTable[0] and contents
                                      ...
                                      SubTable[n] and contents

    If the offset to a lookup overflowed (SubTableIndex is None)
            we must promote the *previous* lookup to an Extension type.

    If the offset from a lookup to subtable overflowed, then we must promote it
            to an Extension Lookup type.
    """
    ok = 0
    lookupIndex = overflowRecord.LookupListIndex
    if overflowRecord.SubTableIndex is None:
        lookupIndex = lookupIndex - 1
    if lookupIndex < 0:
        return ok
    if overflowRecord.tableType == "GSUB":
        extType = 7
    elif overflowRecord.tableType == "GPOS":
        extType = 9

    lookups = ttf[overflowRecord.tableType].table.LookupList.Lookup
    lookup = lookups[lookupIndex]
    # If the previous lookup is an extType, look further back. Very unlikely, but possible.
    while lookup.SubTable[0].__class__.LookupType == extType:
        lookupIndex = lookupIndex - 1
        if lookupIndex < 0:
            # Every lookup before the overflow is already an Extension lookup, so
            # there is nothing left to promote. When this is a LookupList->Lookup
            # offset overflow (SubTableIndex is None), the LookupList's uint16
            # offsets simply can't address all the lookups and fontTools' packer
            # can't recover; warn with actionable advice. (For a subtable offset
            # overflow that fell back here, SubTableIndex is set and this message
            # would be misleading, so it's skipped.)
            if overflowRecord.SubTableIndex is None:
                log.error(
                    "%s LookupList offset overflowed and all lookups are already "
                    "Extension lookups, so the overflow can't be resolved by "
                    "promotion; %s.",
                    overflowRecord.tableType,
                    (
                        "reduce the number of lookups"
                        if have_uharfbuzz
                        else "install uharfbuzz to enable the HarfBuzz repacker "
                        "(it can often pack such tables by sharing/duplicating "
                        "subtables), or reduce the number of lookups"
                    ),
                )
            return ok
        lookup = lookups[lookupIndex]

    for lookupIndex in range(lookupIndex, len(lookups)):
        lookup = lookups[lookupIndex]
        if lookup.LookupType != extType:
            lookup.LookupType = extType
            for si, subTable in enumerate(lookup.SubTable):
                extSubTableClass = lookupTypes[overflowRecord.tableType][extType]
                extSubTable = extSubTableClass()
                extSubTable.Format = 1
                extSubTable.ExtSubTable = subTable
                lookup.SubTable[si] = extSubTable
                ok = 1
    return ok


def splitMultipleSubst(oldSubTable, newSubTable, overflowRecord):
    ok = 1
    oldMapping = sorted(oldSubTable.mapping.items())
    oldLen = len(oldMapping)

    if overflowRecord.itemName in ["Coverage", "RangeRecord"]:
        # Coverage table is written last. Overflow is to or within the
        # the coverage table. We will just cut the subtable in half.
        newLen = oldLen // 2

    elif overflowRecord.itemName == "Sequence":
        # We just need to back up by two items from the overflowed
        # Sequence index to make sure the offset to the Coverage table
        # doesn't overflow.
        newLen = overflowRecord.itemIndex - 1

    newSubTable.mapping = {}
    for i in range(newLen, oldLen):
        item = oldMapping[i]
        key = item[0]
        newSubTable.mapping[key] = item[1]
        del oldSubTable.mapping[key]

    return ok


def splitAlternateSubst(oldSubTable, newSubTable, overflowRecord):
    ok = 1
    if hasattr(oldSubTable, "sortCoverageLast"):
        newSubTable.sortCoverageLast = oldSubTable.sortCoverageLast

    oldAlts = sorted(oldSubTable.alternates.items())
    oldLen = len(oldAlts)

    if overflowRecord.itemName in ["Coverage", "RangeRecord"]:
        # Coverage table is written last. overflow is to or within the
        # the coverage table. We will just cut the subtable in half.
        newLen = oldLen // 2

    elif overflowRecord.itemName == "AlternateSet":
        # We just need to back up by two items
        # from the overflowed AlternateSet index to make sure the offset
        # to the Coverage table doesn't overflow.
        newLen = overflowRecord.itemIndex - 1

    newSubTable.alternates = {}
    for i in range(newLen, oldLen):
        item = oldAlts[i]
        key = item[0]
        newSubTable.alternates[key] = item[1]
        del oldSubTable.alternates[key]

    return ok


def splitLigatureSubst(oldSubTable, newSubTable, overflowRecord):
    ok = 1
    oldLigs = sorted(oldSubTable.ligatures.items())
    oldLen = len(oldLigs)

    if overflowRecord.itemName in ["Coverage", "RangeRecord"]:
        # Coverage table is written last. overflow is to or within the
        # the coverage table. We will just cut the subtable in half.
        newLen = oldLen // 2

    elif overflowRecord.itemName == "LigatureSet":
        # We just need to back up by two items
        # from the overflowed AlternateSet index to make sure the offset
        # to the Coverage table doesn't overflow.
        newLen = overflowRecord.itemIndex - 1

    newSubTable.ligatures = {}
    for i in range(newLen, oldLen):
        item = oldLigs[i]
        key = item[0]
        newSubTable.ligatures[key] = item[1]
        del oldSubTable.ligatures[key]

    return ok


def splitSinglePos(oldSubTable, newSubTable, overflowRecord):
    # Only Format 2 (one Value per covered glyph) is worth splitting; a Format 1
    # subtable shares a single Value across its whole Coverage, so there is
    # nothing worth splitting.
    if oldSubTable.Format != 2 or len(oldSubTable.Coverage.glyphs) <= 1:
        return False

    newSubTable.Format = oldSubTable.Format
    newSubTable.ValueFormat = oldSubTable.ValueFormat

    coverage = oldSubTable.Coverage.glyphs
    values = oldSubTable.Value
    oldCount = len(coverage) // 2

    newSubTable.Coverage = oldSubTable.Coverage.__class__()
    newSubTable.Coverage.glyphs = coverage[oldCount:]
    newSubTable.Value = values[oldCount:]

    oldSubTable.Coverage.glyphs = coverage[:oldCount]
    oldSubTable.Value = values[:oldCount]

    oldSubTable.ValueCount = len(oldSubTable.Value)
    newSubTable.ValueCount = len(newSubTable.Value)

    return True


def splitPairPos(oldSubTable, newSubTable, overflowRecord):
    st = oldSubTable
    ok = False
    newSubTable.Format = oldSubTable.Format
    if oldSubTable.Format == 1 and len(oldSubTable.PairSet) > 1:
        for name in "ValueFormat1", "ValueFormat2":
            setattr(newSubTable, name, getattr(oldSubTable, name))

        # Move top half of coverage to new subtable

        newSubTable.Coverage = oldSubTable.Coverage.__class__()

        coverage = oldSubTable.Coverage.glyphs
        records = oldSubTable.PairSet

        oldCount = len(oldSubTable.PairSet) // 2

        oldSubTable.Coverage.glyphs = coverage[:oldCount]
        oldSubTable.PairSet = records[:oldCount]

        newSubTable.Coverage.glyphs = coverage[oldCount:]
        newSubTable.PairSet = records[oldCount:]

        oldSubTable.PairSetCount = len(oldSubTable.PairSet)
        newSubTable.PairSetCount = len(newSubTable.PairSet)

        ok = True

    elif oldSubTable.Format == 2 and len(oldSubTable.Class1Record) > 1:
        if not hasattr(oldSubTable, "Class2Count"):
            oldSubTable.Class2Count = len(oldSubTable.Class1Record[0].Class2Record)
        for name in "Class2Count", "ClassDef2", "ValueFormat1", "ValueFormat2":
            setattr(newSubTable, name, getattr(oldSubTable, name))

        # The two subtables will still have the same ClassDef2 and the table
        # sharing will still cause the sharing to overflow.  As such, disable
        # sharing on the one that is serialized second (that's oldSubTable).
        oldSubTable.DontShare = True

        # Move top half of class numbers to new subtable

        newSubTable.Coverage = oldSubTable.Coverage.__class__()
        newSubTable.ClassDef1 = oldSubTable.ClassDef1.__class__()

        coverage = oldSubTable.Coverage.glyphs
        classDefs = oldSubTable.ClassDef1.classDefs
        records = oldSubTable.Class1Record

        oldCount = len(oldSubTable.Class1Record) // 2
        newGlyphs = set(k for k, v in classDefs.items() if v >= oldCount)

        oldSubTable.Coverage.glyphs = [g for g in coverage if g not in newGlyphs]
        oldSubTable.ClassDef1.classDefs = {
            k: v for k, v in classDefs.items() if v < oldCount
        }
        oldSubTable.Class1Record = records[:oldCount]

        newSubTable.Coverage.glyphs = [g for g in coverage if g in newGlyphs]
        newSubTable.ClassDef1.classDefs = {
            k: (v - oldCount) for k, v in classDefs.items() if v > oldCount
        }
        newSubTable.Class1Record = records[oldCount:]

        oldSubTable.Class1Count = len(oldSubTable.Class1Record)
        newSubTable.Class1Count = len(newSubTable.Class1Record)

        ok = True

    return ok


def splitMarkBasePos(oldSubTable, newSubTable, overflowRecord):
    # split half of the mark classes to the new subtable
    classCount = oldSubTable.ClassCount
    if classCount < 2:
        # oh well, not much left to split...
        return False

    oldClassCount = classCount // 2
    newClassCount = classCount - oldClassCount

    oldMarkCoverage, oldMarkRecords = [], []
    newMarkCoverage, newMarkRecords = [], []
    for glyphName, markRecord in zip(
        oldSubTable.MarkCoverage.glyphs, oldSubTable.MarkArray.MarkRecord
    ):
        if markRecord.Class < oldClassCount:
            oldMarkCoverage.append(glyphName)
            oldMarkRecords.append(markRecord)
        else:
            markRecord.Class -= oldClassCount
            newMarkCoverage.append(glyphName)
            newMarkRecords.append(markRecord)

    oldBaseRecords, newBaseRecords = [], []
    for rec in oldSubTable.BaseArray.BaseRecord:
        oldBaseRecord, newBaseRecord = rec.__class__(), rec.__class__()
        oldBaseRecord.BaseAnchor = rec.BaseAnchor[:oldClassCount]
        newBaseRecord.BaseAnchor = rec.BaseAnchor[oldClassCount:]
        oldBaseRecords.append(oldBaseRecord)
        newBaseRecords.append(newBaseRecord)

    newSubTable.Format = oldSubTable.Format

    oldSubTable.MarkCoverage.glyphs = oldMarkCoverage
    newSubTable.MarkCoverage = oldSubTable.MarkCoverage.__class__()
    newSubTable.MarkCoverage.glyphs = newMarkCoverage

    # share the same BaseCoverage in both halves
    newSubTable.BaseCoverage = oldSubTable.BaseCoverage

    oldSubTable.ClassCount = oldClassCount
    newSubTable.ClassCount = newClassCount

    oldSubTable.MarkArray.MarkRecord = oldMarkRecords
    newSubTable.MarkArray = oldSubTable.MarkArray.__class__()
    newSubTable.MarkArray.MarkRecord = newMarkRecords

    oldSubTable.MarkArray.MarkCount = len(oldMarkRecords)
    newSubTable.MarkArray.MarkCount = len(newMarkRecords)

    oldSubTable.BaseArray.BaseRecord = oldBaseRecords
    newSubTable.BaseArray = oldSubTable.BaseArray.__class__()
    newSubTable.BaseArray.BaseRecord = newBaseRecords

    oldSubTable.BaseArray.BaseCount = len(oldBaseRecords)
    newSubTable.BaseArray.BaseCount = len(newBaseRecords)

    return True


splitTable = {
    "GSUB": {
        # 					1: splitSingleSubst,
        2: splitMultipleSubst,
        3: splitAlternateSubst,
        4: splitLigatureSubst,
        # 					5: splitContextSubst,
        # 					6: splitChainContextSubst,
        # 					7: splitExtensionSubst,
        # 					8: splitReverseChainSingleSubst,
    },
    "GPOS": {
        1: splitSinglePos,
        2: splitPairPos,
        # 					3: splitCursivePos,
        4: splitMarkBasePos,
        # 					5: splitMarkLigPos,
        # 					6: splitMarkMarkPos,
        # 					7: splitContextPos,
        # 					8: splitChainContextPos,
        # 					9: splitExtensionPos,
    },
}


def fixSubTableOverFlows(ttf, overflowRecord):
    """
    An offset has overflowed within a sub-table. We need to divide this subtable into smaller parts.
    """
    table = ttf[overflowRecord.tableType].table
    lookup = table.LookupList.Lookup[overflowRecord.LookupListIndex]
    subIndex = overflowRecord.SubTableIndex
    subtable = lookup.SubTable[subIndex]

    # First, try not sharing anything for this subtable...
    if not hasattr(subtable, "DontShare"):
        subtable.DontShare = True
        return True

    if hasattr(subtable, "ExtSubTable"):
        # We split the subtable of the Extension table, and add a new Extension table
        # to contain the new subtable.

        subTableType = subtable.ExtSubTable.__class__.LookupType
        extSubTable = subtable
        subtable = extSubTable.ExtSubTable
        newExtSubTableClass = lookupTypes[overflowRecord.tableType][
            extSubTable.__class__.LookupType
        ]
        newExtSubTable = newExtSubTableClass()
        newExtSubTable.Format = extSubTable.Format
        toInsert = newExtSubTable

        newSubTableClass = lookupTypes[overflowRecord.tableType][subTableType]
        newSubTable = newSubTableClass()
        newExtSubTable.ExtSubTable = newSubTable
    else:
        subTableType = subtable.__class__.LookupType
        newSubTableClass = lookupTypes[overflowRecord.tableType][subTableType]
        newSubTable = newSubTableClass()
        toInsert = newSubTable

    if hasattr(lookup, "SubTableCount"):  # may not be defined yet.
        lookup.SubTableCount = lookup.SubTableCount + 1

    try:
        splitFunc = splitTable[overflowRecord.tableType][subTableType]
    except KeyError:
        log.error(
            "Don't know how to split %s lookup type %s",
            overflowRecord.tableType,
            subTableType,
        )
        return False

    ok = splitFunc(subtable, newSubTable, overflowRecord)
    if ok:
        lookup.SubTable.insert(subIndex + 1, toInsert)
    return ok


# End of OverFlow logic


# ---------------------------------------------------------------------------
# IFT - Incremental Font Transfer tables
# ---------------------------------------------------------------------------


class EntryIdStringData(BaseTable):
    """IFT entry ID string data block (raw bytes pool)."""

    def decompile(self, reader, font):
        # This subtable is reached via an LOffset, so the reader's data buffer
        # is already scoped to the bytes starting at the subtable's offset within
        # the IFT table blob.  Reading to the end of that buffer is correct; the
        # actual bytes consumed are determined by the entryIdStringLength fields
        # in the MappingEntries records.
        self.data = bytes(reader.data[reader.pos :])

    def compile(self, writer, font):
        writer.writeData(self.data)

    def toXML2(self, xmlWriter, font):
        xmlWriter.simpletag("data", value=self.data.hex())
        xmlWriter.newline()

    def fromXML(self, name, attrs, content, font):
        if name == "data":
            self.data = bytes.fromhex(attrs["value"])


def _buildClasses():
    import re
    from .otData import otData

    formatPat = re.compile(r"([A-Za-z0-9]+)Format(\d+)$")
    namespace = globals()

    # populate module with classes
    for name, fields in otData:
        baseClass = BaseTable
        m = formatPat.match(name)
        if m:
            # XxxFormatN subtable, we only add the "base" table
            name = m.group(1)
            # the first row of a format-switching otData table describes the Format;
            # the first column defines the type of the Format field.
            # Currently this can be either 'uint16' or 'uint8'.
            formatType = fields[0].type
            baseClass = getFormatSwitchingBaseTableClass(formatType)
        if name not in namespace:
            # the class doesn't exist yet, so the base implementation is used.
            cls = type(name, (baseClass,), {})
            if name in ("GSUB", "GPOS"):
                cls.DontShare = True
            namespace[name] = cls

    # link Var{Table} <-> {Table} (e.g. ColorStop <-> VarColorStop, etc.)
    for name, _ in otData:
        if name.startswith("Var") and len(name) > 3 and name[3:] in namespace:
            varType = namespace[name]
            noVarType = namespace[name[3:]]
            varType.NoVarType = noVarType
            noVarType.VarType = varType

    for base, alts in _equivalents.items():
        base = namespace[base]
        for alt in alts:
            namespace[alt] = base

    global lookupTypes
    lookupTypes = {
        "GSUB": {
            1: SingleSubst,
            2: MultipleSubst,
            3: AlternateSubst,
            4: LigatureSubst,
            5: ContextSubst,
            6: ChainContextSubst,
            7: ExtensionSubst,
            8: ReverseChainSingleSubst,
        },
        "GPOS": {
            1: SinglePos,
            2: PairPos,
            3: CursivePos,
            4: MarkBasePos,
            5: MarkLigPos,
            6: MarkMarkPos,
            7: ContextPos,
            8: ChainContextPos,
            9: ExtensionPos,
        },
        "mort": {
            0: RearrangementMorph,
            1: ContextualMorph,
            2: LigatureMorph,
            4: NoncontextualMorph,
            5: InsertionMorph,
        },
        "morx": {
            0: RearrangementMorph,
            1: ContextualMorph,
            2: LigatureMorph,
            # 3: Reserved,
            4: NoncontextualMorph,
            5: InsertionMorph,
        },
    }
    lookupTypes["JSTF"] = lookupTypes["GPOS"]  # JSTF contains GPOS
    for lookupEnum in lookupTypes.values():
        for enum, cls in lookupEnum.items():
            cls.LookupType = enum

    global featureParamTypes
    featureParamTypes = {
        "size": FeatureParamsSize,
    }
    for i in range(1, 20 + 1):
        featureParamTypes["ss%02d" % i] = FeatureParamsStylisticSet
    for i in range(1, 99 + 1):
        featureParamTypes["cv%02d" % i] = FeatureParamsCharacterVariants

    # add converters to classes
    from .otConverters import buildConverters

    for name, fields in otData:
        m = formatPat.match(name)
        if m:
            # XxxFormatN subtable, add converter to "base" table
            name, format = m.groups()
            format = int(format)
            cls = namespace[name]
            if not hasattr(cls, "converters"):
                cls.converters = {}
                cls.convertersByName = {}
            converters, convertersByName = buildConverters(fields[1:], namespace)
            cls.converters[format] = converters
            cls.convertersByName[format] = convertersByName
            # XXX Add staticSize?
        else:
            cls = namespace[name]
            cls.converters, cls.convertersByName = buildConverters(fields, namespace)
            # XXX Add staticSize?


_buildClasses()


def _getGlyphsFromCoverageTable(coverage):
    if coverage is None:
        # empty coverage table
        return []
    else:
        return coverage.glyphs
<|endoftext|>"""_g_l_y_f.py -- Converter classes for the 'glyf' table."""

from collections import namedtuple
from collections.abc import ItemsView, ValuesView
from fontTools.misc import sstruct
from fontTools import ttLib
from fontTools import version
from fontTools.misc.transform import DecomposedTransform
from fontTools.misc.textTools import tostr, safeEval, pad
from fontTools.misc.arrayTools import updateBounds, pointInRect
from fontTools.misc.bezierTools import calcQuadraticBounds
from fontTools.misc.fixedTools import (
    fixedToFloat as fi2fl,
    floatToFixed as fl2fi,
    floatToFixedToStr as fl2str,
    strToFixedToFloat as str2fl,
)
from fontTools.misc.roundTools import noRound, otRound
from fontTools.misc.vector import Vector
from numbers import Number
from . import DefaultTable
from . import ttProgram
import sys
import struct
import array
import logging
import math
import os
from fontTools.misc import xmlWriter
from fontTools.misc.filenames import userNameToFileName
from fontTools.misc.loggingTools import deprecateFunction
from enum import IntFlag
from functools import partial
from types import SimpleNamespace
from typing import Set

log = logging.getLogger(__name__)

# We compute the version the same as is computed in ttlib/__init__
# so that we can write 'ttLibVersion' attribute of the glyf TTX files
# when glyf is written to separate files.
version = ".".join(version.split(".")[:2])

#
# The Apple and MS rasterizers behave differently for
# scaled composite components: one does scale first and then translate
# and the other does it vice versa. MS defined some flags to indicate
# the difference, but it seems nobody actually _sets_ those flags.
#
# Funny thing: Apple seems to _only_ do their thing in the
# WE_HAVE_A_SCALE (eg. Chicago) case, and not when it's WE_HAVE_AN_X_AND_Y_SCALE
# (eg. Charcoal)...
#
SCALE_COMPONENT_OFFSET_DEFAULT = 0  # 0 == MS, 1 == Apple


class table__g_l_y_f(DefaultTable.DefaultTable):
    """Glyph Data table

    This class represents the `glyf <https://docs.microsoft.com/en-us/typography/opentype/spec/glyf>`_
    table, which contains outlines for glyphs in TrueType format. In many cases,
    it is easier to access and manipulate glyph outlines through the ``GlyphSet``
    object returned from :py:meth:`fontTools.ttLib.ttFont.getGlyphSet`::

                    >> from fontTools.pens.boundsPen import BoundsPen
                    >> glyphset = font.getGlyphSet()
                    >> bp = BoundsPen(glyphset)
                    >> glyphset["A"].draw(bp)
                    >> bp.bounds
                    (19, 0, 633, 716)

    However, this class can be used for low-level access to the ``glyf`` table data.
    Objects of this class support dictionary-like access, mapping glyph names to
    :py:class:`Glyph` objects::

                    >> glyf = font["glyf"]
                    >> len(glyf["Aacute"].components)
                    2

    Note that when adding glyphs to the font via low-level access to the ``glyf``
    table, the new glyphs must also be added to the ``hmtx``/``vmtx`` table::

                    >> font["glyf"]["divisionslash"] = Glyph()
                    >> font["hmtx"]["divisionslash"] = (640, 0)

    """

    # this attribute controls the amount of padding applied to glyph data upon compile.
    # Glyph lenghts are aligned to multiples of the specified value.
    # Allowed values are (0, 1, 2, 4). '0' means no padding; '1' (default) also means
    # no padding, except for when padding would allow to use short loca offsets.
    padding = 1

    def decompile(self, data, ttFont):
        loca = ttFont["loca"]
        pos = int(loca[0])
        nextPos = 0
        noname = 0
        self.glyphs = {}
        self.glyphOrder = glyphOrder = ttFont.getGlyphOrder()
        self._reverseGlyphOrder = {}
        for i in range(0, len(loca) - 1):
            try:
                glyphName = glyphOrder[i]
            except IndexError:
                noname = noname + 1
                glyphName = "ttxautoglyph%s" % i
            nextPos = int(loca[i + 1])
            glyphdata = data[pos:nextPos]
            if len(glyphdata) != (nextPos - pos):
                raise ttLib.TTLibError("not enough 'glyf' table data")
            glyph = Glyph(glyphdata)
            self.glyphs[glyphName] = glyph
            pos = nextPos
        if len(data) - nextPos >= 4:
            log.warning(
                "too much 'glyf' table data: expected %d, received %d bytes",
                nextPos,
                len(data),
            )
        if noname:
            log.warning("%s glyphs have no name", noname)
        if ttFont.lazy is False:  # Be lazy for None and True
            self.ensureDecompiled()

    def ensureDecompiled(self, recurse=False):
        # The recurse argument is unused, but part of the signature of
        # ensureDecompiled across the library.
        for glyph in self.glyphs.values():
            glyph.expand(self)

    def compile(self, ttFont):
        optimizeSpeed = ttFont.cfg[ttLib.OPTIMIZE_FONT_SPEED]
        if not hasattr(self, "glyphOrder"):
            self.glyphOrder = ttFont.getGlyphOrder()
        padding = self.padding
        assert padding in (0, 1, 2, 4)
        locations = []
        currentLocation = 0
        dataList = []
        recalcBBoxes = ttFont.recalcBBoxes
        boundsDone = set()
        for glyphName in self.glyphOrder:
            glyph = self.glyphs[glyphName]
            glyphData = glyph.compile(
                self,
                recalcBBoxes,
                boundsDone=boundsDone,
                optimizeSize=not optimizeSpeed,
            )
            if padding > 1:
                glyphData = pad(glyphData, size=padding)
            locations.append(currentLocation)
            currentLocation = currentLocation + len(glyphData)
            dataList.append(glyphData)
        locations.append(currentLocation)

        if padding == 1 and currentLocation < 0x20000:
            # See if we can pad any odd-lengthed glyphs to allow loca
            # table to use the short offsets.
            indices = [
                i for i, glyphData in enumerate(dataList) if len(glyphData) % 2 == 1
            ]
            if indices and currentLocation + len(indices) < 0x20000:
                # It fits.  Do it.
                for i in indices:
                    dataList[i] += b"\0"
                currentLocation = 0
                for i, glyphData in enumerate(dataList):
                    locations[i] = currentLocation
                    currentLocation += len(glyphData)
                locations[len(dataList)] = currentLocation

        data = b"".join(dataList)
        if "loca" in ttFont:
            ttFont["loca"].set(locations)
        if "maxp" in ttFont:
            ttFont["maxp"].numGlyphs = len(self.glyphs)
        if not data:
            # As a special case when all glyph in the font are empty, add a zero byte
            # to the table, so that OTS doesn’t reject it, and to make the table work
            # on Windows as well.
            # See https://github.com/khaledhosny/ots/issues/52
            data = b"\0"
        return data

    def toXML(self, writer, ttFont, splitGlyphs=False):
        notice = (
            "The xMin, yMin, xMax and yMax values\n"
            "will be recalculated by the compiler."
        )
        glyphNames = ttFont.getGlyphNames()
        if not splitGlyphs:
            writer.newline()
            writer.comment(notice)
            writer.newline()
            writer.newline()
        numGlyphs = len(glyphNames)
        if splitGlyphs:
            path, ext = os.path.splitext(writer.file.name)
            existingGlyphFiles = set()
        for glyphName in glyphNames:
            glyph = self.get(glyphName)
            if glyph is None:
                log.warning("glyph '%s' does not exist in glyf table", glyphName)
                continue
            if glyph.numberOfContours:
                if splitGlyphs:
                    glyphPath = userNameToFileName(
                        tostr(glyphName, "utf-8"),
                        existingGlyphFiles,
                        prefix=path + ".",
                        suffix=ext,
                    )
                    existingGlyphFiles.add(glyphPath.lower())
                    glyphWriter = xmlWriter.XMLWriter(
                        glyphPath,
                        idlefunc=writer.idlefunc,
                        newlinestr=writer.newlinestr,
                    )
                    glyphWriter.begintag("ttFont", ttLibVersion=version)
                    glyphWriter.newline()
                    glyphWriter.begintag("glyf")
                    glyphWriter.newline()
                    glyphWriter.comment(notice)
                    glyphWriter.newline()
                    writer.simpletag("TTGlyph", src=os.path.basename(glyphPath))
                else:
                    glyphWriter = writer
                glyphWriter.begintag(
                    "TTGlyph",
                    [
                        ("name", glyphName),
                        ("xMin", glyph.xMin),
                        ("yMin", glyph.yMin),
                        ("xMax", glyph.xMax),
                        ("yMax", glyph.yMax),
                    ],
                )
                glyphWriter.newline()
                glyph.toXML(glyphWriter, ttFont)
                glyphWriter.endtag("TTGlyph")
                glyphWriter.newline()
                if splitGlyphs:
                    glyphWriter.endtag("glyf")
                    glyphWriter.newline()
                    glyphWriter.endtag("ttFont")
                    glyphWriter.newline()
                    glyphWriter.close()
            else:
                writer.simpletag("TTGlyph", name=glyphName)
                writer.comment("contains no outline data")
                if not splitGlyphs:
                    writer.newline()
            writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        if name != "TTGlyph":
            return
        if not hasattr(self, "glyphs"):
            self.glyphs = {}
        if not hasattr(self, "glyphOrder"):
            self.glyphOrder = ttFont.getGlyphOrder()
        glyphName = attrs["name"]
        log.debug("unpacking glyph '%s'", glyphName)
        glyph = Glyph()
        for attr in ["xMin", "yMin", "xMax", "yMax"]:
            setattr(glyph, attr, safeEval(attrs.get(attr, "0")))
        self.glyphs[glyphName] = glyph
        for element in content:
            if not isinstance(element, tuple):
                continue
            name, attrs, content = element
            glyph.fromXML(name, attrs, content, ttFont)
        if not ttFont.recalcBBoxes:
            glyph.compact(self, 0)

    def setGlyphOrder(self, glyphOrder):
        """Sets the glyph order

        Args:
                glyphOrder ([str]): List of glyph names in order.
        """
        self.glyphOrder = glyphOrder
        self._reverseGlyphOrder = {}

    def getGlyphName(self, glyphID):
        """Returns the name for the glyph with the given ID.

        Raises a ``KeyError`` if the glyph name is not found in the font.
        """
        return self.glyphOrder[glyphID]

    def _buildReverseGlyphOrderDict(self):
        self._reverseGlyphOrder = d = {}
        for glyphID, glyphName in enumerate(self.glyphOrder):
            d[glyphName] = glyphID

    def getGlyphID(self, glyphName):
        """Returns the ID of the glyph with the given name.

        Raises a ``ValueError`` if the glyph is not found in the font.
        """
        glyphOrder = self.glyphOrder
        id = getattr(self, "_reverseGlyphOrder", {}).get(glyphName)
        if id is None or id >= len(glyphOrder) or glyphOrder[id] != glyphName:
            self._buildReverseGlyphOrderDict()
            id = self._reverseGlyphOrder.get(glyphName)
        if id is None:
            raise ValueError(glyphName)
        return id

    def removeHinting(self):
        """Removes TrueType hints from all glyphs in the glyphset.

        See :py:meth:`Glyph.removeHinting`.
        """
        for glyph in self.glyphs.values():
            glyph.removeHinting()

    def keys(self):
        return self.glyphs.keys()

    def __iter__(self):
        return self.glyphs.__iter__()

    def values(self):
        return ValuesView(self)

    def items(self):
        return ItemsView(self)

    def has_key(self, glyphName):
        return glyphName in self.glyphs

    __contains__ = has_key

    def get(self, glyphName, default=None):
        glyph = self.glyphs.get(glyphName, default)
        if glyph is not None:
            glyph.expand(self)
        return glyph

    def __getitem__(self, glyphName):
        glyph = self.glyphs[glyphName]
        glyph.expand(self)
        return glyph

    def __setitem__(self, glyphName, glyph):
        self.glyphs[glyphName] = glyph
        # Use the reverse glyph map for O(1) membership so that building a font by
        # repeatedly assigning glyphs stays linear instead of quadratic. Rebuild
        # the map if it is missing or no longer matches the length of glyphOrder;
        # reordering goes through setGlyphOrder, which clears it.
        rev = getattr(self, "_reverseGlyphOrder", None)
        if rev is None or len(rev) != len(self.glyphOrder):
            self._buildReverseGlyphOrderDict()
            rev = self._reverseGlyphOrder
        if glyphName not in rev:
            rev[glyphName] = len(self.glyphOrder)
            self.glyphOrder.append(glyphName)

    def __delitem__(self, glyphName):
        del self.glyphs[glyphName]
        self.glyphOrder.remove(glyphName)
        # glyph ids after the removed one shift down; rebuild lazily on next use.
        self._reverseGlyphOrder = {}

    def __len__(self):
        assert len(self.glyphOrder) == len(self.glyphs)
        return len(self.glyphs)

    def _getPhantomPoints(self, glyphName, hMetrics, vMetrics=None):
        """Compute the four "phantom points" for the given glyph from its bounding box
        and the horizontal and vertical advance widths and sidebearings stored in the
        ttFont's "hmtx" and "vmtx" tables.

        'hMetrics' should be ttFont['hmtx'].metrics.

        'vMetrics' should be ttFont['vmtx'].metrics if there is "vmtx" or None otherwise.
        If there is no vMetrics passed in, vertical phantom points are set to the zero coordinate.

        https://docs.microsoft.com/en-us/typography/opentype/spec/tt_instructing_glyphs#phantoms
        """
        glyph = self[glyphName]
        if not hasattr(glyph, "xMin"):
            glyph.recalcBounds(self)

        horizontalAdvanceWidth, leftSideBearing = hMetrics[glyphName]
        leftSideX = glyph.xMin - leftSideBearing
        rightSideX = leftSideX + horizontalAdvanceWidth

        if vMetrics:
            verticalAdvanceWidth, topSideBearing = vMetrics[glyphName]
            topSideY = topSideBearing + glyph.yMax
            bottomSideY = topSideY - verticalAdvanceWidth
        else:
            bottomSideY = topSideY = 0

        return [
            (leftSideX, 0),
            (rightSideX, 0),
            (0, topSideY),
            (0, bottomSideY),
        ]

    def _getCoordinatesAndControls(
        self, glyphName, hMetrics, vMetrics=None, *, round=otRound
    ):
        """Return glyph coordinates and controls as expected by "gvar" table.

        The coordinates includes four "phantom points" for the glyph metrics,
        as mandated by the "gvar" spec.

        The glyph controls is a namedtuple with the following attributes:
                - numberOfContours: -1 for composite glyphs.
                - endPts: list of indices of end points for each contour in simple
                glyphs, or component indices in composite glyphs (used for IUP
                optimization).
                - flags: array of contour point flags for simple glyphs (None for
                composite glyphs).
                - components: list of base glyph names (str) for each component in
                composite glyphs (None for simple glyphs).

        The "hMetrics" and vMetrics are used to compute the "phantom points" (see
        the "_getPhantomPoints" method).

        Return None if the requested glyphName is not present.
        """
        glyph = self.get(glyphName)
        if glyph is None:
            return None
        if glyph.isComposite():
            coords = GlyphCoordinates(
                [(getattr(c, "x", 0), getattr(c, "y", 0)) for c in glyph.components]
            )
            controls = _GlyphControls(
                numberOfContours=glyph.numberOfContours,
                endPts=list(range(len(glyph.components))),
                flags=None,
                components=[
                    (c.glyphName, getattr(c, "transform", None))
                    for c in glyph.components
                ],
            )
        else:
            coords, endPts, flags = glyph.getCoordinates(self)
            coords = coords.copy()
            controls = _GlyphControls(
                numberOfContours=glyph.numberOfContours,
                endPts=endPts,
                flags=flags,
                components=None,
            )
        # Add phantom points for (left, right, top, bottom) positions.
        phantomPoints = self._getPhantomPoints(glyphName, hMetrics, vMetrics)
        coords.extend(phantomPoints)
        coords.toInt(round=round)
        return coords, controls

    def _setCoordinates(self, glyphName, coord, hMetrics, vMetrics=None):
        """Set coordinates and metrics for the given glyph.

        "coord" is an array of GlyphCoordinates which must include the "phantom
        points" as the last four coordinates.

        Both the horizontal/vertical advances and left/top sidebearings in "hmtx"
        and "vmtx" tables (if any) are updated from four phantom points and
        the glyph's bounding boxes.

        The "hMetrics" and vMetrics are used to propagate "phantom points"
        into "hmtx" and "vmtx" tables if desired.  (see the "_getPhantomPoints"
        method).
        """
        glyph = self[glyphName]

        # Handle phantom points for (left, right, top, bottom) positions.
        assert len(coord) >= 4
        leftSideX = coord[-4][0]
        rightSideX = coord[-3][0]
        topSideY = coord[-2][1]
        bottomSideY = coord[-1][1]

        coord = coord[:-4]

        if glyph.isComposite():
            assert len(coord) == len(glyph.components)
            for p, comp in zip(coord, glyph.components):
                if hasattr(comp, "x"):
                    comp.x, comp.y = p
        elif glyph.numberOfContours == 0:
            assert len(coord) == 0
        else:
            assert len(coord) == len(glyph.coordinates)
            glyph.coordinates = GlyphCoordinates(coord)

        glyph.recalcBounds(self, boundsDone=set())

        horizontalAdvanceWidth = otRound(rightSideX - leftSideX)
        if horizontalAdvanceWidth < 0:
            # unlikely, but it can happen, see:
            # https://github.com/fonttools/fonttools/pull/1198
            horizontalAdvanceWidth = 0
        leftSideBearing = otRound(glyph.xMin - leftSideX)
        hMetrics[glyphName] = horizontalAdvanceWidth, leftSideBearing

        if vMetrics is not None:
            verticalAdvanceWidth = otRound(topSideY - bottomSideY)
            if verticalAdvanceWidth < 0:  # unlikely but do the same as horizontal
                verticalAdvanceWidth = 0
            topSideBearing = otRound(topSideY - glyph.yMax)
            vMetrics[glyphName] = verticalAdvanceWidth, topSideBearing

    # Deprecated

    def _synthesizeVMetrics(self, glyphName, ttFont, defaultVerticalOrigin):
        """This method is wrong and deprecated.
        For rationale see:
        https://github.com/fonttools/fonttools/pull/2266/files#r613569473
        """
        vMetrics = getattr(ttFont.get("vmtx"), "metrics", None)
        if vMetrics is None:
            verticalAdvanceWidth = ttFont["head"].unitsPerEm
            topSideY = getattr(ttFont.get("hhea"), "ascent", None)
            if topSideY is None:
                if defaultVerticalOrigin is not None:
                    topSideY = defaultVerticalOrigin
                else:
                    topSideY = verticalAdvanceWidth
            glyph = self[glyphName]
            glyph.recalcBounds(self)
            topSideBearing = otRound(topSideY - glyph.yMax)
            vMetrics = {glyphName: (verticalAdvanceWidth, topSideBearing)}
        return vMetrics

    @deprecateFunction("use '_getPhantomPoints' instead", category=DeprecationWarning)
    def getPhantomPoints(self, glyphName, ttFont, defaultVerticalOrigin=None):
        """Old public name for self._getPhantomPoints().
        See: https://github.com/fonttools/fonttools/pull/2266"""
        hMetrics = ttFont["hmtx"].metrics
        vMetrics = self._synthesizeVMetrics(glyphName, ttFont, defaultVerticalOrigin)
        return self._getPhantomPoints(glyphName, hMetrics, vMetrics)

    @deprecateFunction(
        "use '_getCoordinatesAndControls' instead", category=DeprecationWarning
    )
    def getCoordinatesAndControls(self, glyphName, ttFont, defaultVerticalOrigin=None):
        """Old public name for self._getCoordinatesAndControls().
        See: https://github.com/fonttools/fonttools/pull/2266"""
        hMetrics = ttFont["hmtx"].metrics
        vMetrics = self._synthesizeVMetrics(glyphName, ttFont, defaultVerticalOrigin)
        return self._getCoordinatesAndControls(glyphName, hMetrics, vMetrics)

    @deprecateFunction("use '_setCoordinates' instead", category=DeprecationWarning)
    def setCoordinates(self, glyphName, ttFont):
        """Old public name for self._setCoordinates().
        See: https://github.com/fonttools/fonttools/pull/2266"""
        hMetrics = ttFont["hmtx"].metrics
        vMetrics = getattr(ttFont.get("vmtx"), "metrics", None)
        self._setCoordinates(glyphName, hMetrics, vMetrics)


_GlyphControls = namedtuple(
    "_GlyphControls", "numberOfContours endPts flags components"
)


glyphHeaderFormat = """
		>	# big endian
		numberOfContours:	h
		xMin:				h
		yMin:				h
		xMax:				h
		yMax:				h
"""

# flags
flagOnCurve = 0x01
flagXShort = 0x02
flagYShort = 0x04
flagRepeat = 0x08
flagXsame = 0x10
flagYsame = 0x20
flagOverlapSimple = 0x40
flagCubic = 0x80

# These flags are kept for XML output after decompiling the coordinates
keepFlags = flagOnCurve + flagOverlapSimple + flagCubic

_flagSignBytes = {
    0: 2,
    flagXsame: 0,
    flagXShort | flagXsame: +1,
    flagXShort: -1,
    flagYsame: 0,
    flagYShort | flagYsame: +1,
    flagYShort: -1,
}


def flagBest(x, y, onCurve):
    """For a given x,y delta pair, returns the flag that packs this pair
    most efficiently, as well as the number of byte cost of such flag."""

    flag = flagOnCurve if onCurve else 0
    cost = 0
    # do x
    if x == 0:
        flag = flag | flagXsame
    elif -255 <= x <= 255:
        flag = flag | flagXShort
        if x > 0:
            flag = flag | flagXsame
        cost += 1
    else:
        cost += 2
    # do y
    if y == 0:
        flag = flag | flagYsame
    elif -255 <= y <= 255:
        flag = flag | flagYShort
        if y > 0:
            flag = flag | flagYsame
        cost += 1
    else:
        cost += 2
    return flag, cost


def flagFits(newFlag, oldFlag, mask):
    newBytes = _flagSignBytes[newFlag & mask]
    oldBytes = _flagSignBytes[oldFlag & mask]
    return newBytes == oldBytes or abs(newBytes) > abs(oldBytes)


def flagSupports(newFlag, oldFlag):
    return (
        (oldFlag & flagOnCurve) == (newFlag & flagOnCurve)
        and flagFits(newFlag, oldFlag, flagXsame | flagXShort)
        and flagFits(newFlag, oldFlag, flagYsame | flagYShort)
    )


def flagEncodeCoord(flag, mask, coord, coordBytes):
    byteCount = _flagSignBytes[flag & mask]
    if byteCount == 1:
        coordBytes.append(coord)
    elif byteCount == -1:
        coordBytes.append(-coord)
    elif byteCount == 2:
        coordBytes.extend(struct.pack(">h", coord))


def flagEncodeCoords(flag, x, y, xBytes, yBytes):
    flagEncodeCoord(flag, flagXsame | flagXShort, x, xBytes)
    flagEncodeCoord(flag, flagYsame | flagYShort, y, yBytes)


ARG_1_AND_2_ARE_WORDS = 0x0001  # if set args are words otherwise they are bytes
ARGS_ARE_XY_VALUES = 0x0002  # if set args are xy values, otherwise they are points
ROUND_XY_TO_GRID = 0x0004  # for the xy values if above is true
WE_HAVE_A_SCALE = 0x0008  # Sx = Sy, otherwise scale == 1.0
NON_OVERLAPPING = 0x0010  # set to same value for all components (obsolete!)
MORE_COMPONENTS = 0x0020  # indicates at least one more glyph after this one
WE_HAVE_AN_X_AND_Y_SCALE = 0x0040  # Sx, Sy
WE_HAVE_A_TWO_BY_TWO = 0x0080  # t00, t01, t10, t11
WE_HAVE_INSTRUCTIONS = 0x0100  # instructions follow
USE_MY_METRICS = 0x0200  # apply these metrics to parent glyph
OVERLAP_COMPOUND = 0x0400  # used by Apple in GX fonts
SCALED_COMPONENT_OFFSET = 0x0800  # composite designed to have the component offset scaled (designed for Apple)
UNSCALED_COMPONENT_OFFSET = 0x1000  # composite designed not to have the component offset scaled (designed for MS)


CompositeMaxpValues = namedtuple(
    "CompositeMaxpValues", ["nPoints", "nContours", "maxComponentDepth"]
)


class Glyph(object):
    """This class represents an individual TrueType glyph.

    TrueType glyph objects come in two flavours: simple and composite. Simple
    glyph objects contain contours, represented via the ``.coordinates``,
    ``.flags``, ``.numberOfContours``, and ``.endPtsOfContours`` attributes;
    composite glyphs contain components, available through the ``.components``
    attributes.

    Because the ``.coordinates`` attribute (and other simple glyph attributes mentioned
    above) is only set on simple glyphs and the ``.components`` attribute is only
    set on composite glyphs, it is necessary to use the :py:meth:`isComposite`
    method to test whether a glyph is simple or composite before attempting to
    access its data.

    For a composite glyph, the components can also be accessed via array-like access::

            >> assert(font["glyf"]["Aacute"].isComposite())
            >> font["glyf"]["Aacute"][0]
            <fontTools.ttLib.tables._g_l_y_f.GlyphComponent at 0x1027b2ee0>

    """

    def __init__(self, data=b""):
        if not data:
            # empty char
            self.numberOfContours = 0
            return
        self.data = data

    def compact(self, glyfTable, recalcBBoxes=True):
        data = self.compile(glyfTable, recalcBBoxes)
        self.__dict__.clear()
        self.data = data

    def expand(self, glyfTable):
        if not hasattr(self, "data"):
            # already unpacked
            return
        if not self.data:
            # empty char
            del self.data
            self.numberOfContours = 0
            return
        dummy, data = sstruct.unpack2(glyphHeaderFormat, self.data, self)
        del self.data
        # Some fonts (eg. Neirizi.ttf) have a 0 for numberOfContours in
        # some glyphs; decompileCoordinates assumes that there's at least
        # one, so short-circuit here.
        if self.numberOfContours == 0:
            return
        if self.isComposite():
            self.decompileComponents(data, glyfTable)
        else:
            self.decompileCoordinates(data)

    def compile(
        self, glyfTable, recalcBBoxes=True, *, boundsDone=None, optimizeSize=True
    ):
        if hasattr(self, "data"):
            if recalcBBoxes:
                # must unpack glyph in order to recalculate bounding box
                self.expand(glyfTable)
            else:
                return self.data
        if self.numberOfContours == 0:
            return b""

        if recalcBBoxes:
            self.recalcBounds(glyfTable, boundsDone=boundsDone)

        data = sstruct.pack(glyphHeaderFormat, self)
        if self.isComposite():
            data = data + self.compileComponents(glyfTable)
        else:
            data = data + self.compileCoordinates(optimizeSize=optimizeSize)
        return data

    def toXML(self, writer, ttFont):
        if self.isComposite():
            for compo in self.components:
                compo.toXML(writer, ttFont)
            haveInstructions = hasattr(self, "program")
        else:
            last = 0
            for i in range(self.numberOfContours):
                writer.begintag("contour")
                writer.newline()
                for j in range(last, self.endPtsOfContours[i] + 1):
                    attrs = [
                        ("x", self.coordinates[j][0]),
                        ("y", self.coordinates[j][1]),
                        ("on", self.flags[j] & flagOnCurve),
                    ]
                    if self.flags[j] & flagOverlapSimple:
                        # Apple's rasterizer uses flagOverlapSimple in the first contour/first pt to flag glyphs that contain overlapping contours
                        attrs.append(("overlap", 1))
                    if self.flags[j] & flagCubic:
                        attrs.append(("cubic", 1))
                    writer.simpletag("pt", attrs)
                    writer.newline()
                last = self.endPtsOfContours[i] + 1
                writer.endtag("contour")
                writer.newline()
            haveInstructions = self.numberOfContours > 0
        if haveInstructions:
            if self.program:
                writer.begintag("instructions")
                writer.newline()
                self.program.toXML(writer, ttFont)
                writer.endtag("instructions")
            else:
                writer.simpletag("instructions")
            writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        if name == "contour":
            if self.numberOfContours < 0:
                raise ttLib.TTLibError("can't mix composites and contours in glyph")
            self.numberOfContours = self.numberOfContours + 1
            coordinates = GlyphCoordinates()
            flags = bytearray()
            for element in content:
                if not isinstance(element, tuple):
                    continue
                name, attrs, content = element
                if name != "pt":
                    continue  # ignore anything but "pt"
                coordinates.append((safeEval(attrs["x"]), safeEval(attrs["y"])))
                flag = bool(safeEval(attrs["on"]))
                if "overlap" in attrs and bool(safeEval(attrs["overlap"])):
                    flag |= flagOverlapSimple
                if "cubic" in attrs and bool(safeEval(attrs["cubic"])):
                    flag |= flagCubic
                flags.append(flag)
            if not hasattr(self, "coordinates"):
                self.coordinates = coordinates
                self.flags = flags
                self.endPtsOfContours = [len(coordinates) - 1]
            else:
                self.coordinates.extend(coordinates)
                self.flags.extend(flags)
                self.endPtsOfContours.append(len(self.coordinates) - 1)
        elif name == "component":
            if self.numberOfContours > 0:
                raise ttLib.TTLibError("can't mix composites and contours in glyph")
            self.numberOfContours = -1
            if not hasattr(self, "components"):
                self.components = []
            component = GlyphComponent()
            self.components.append(component)
            component.fromXML(name, attrs, content, ttFont)
        elif name == "instructions":
            self.program = ttProgram.Program()
            for element in content:
                if not isinstance(element, tuple):
                    continue
                name, attrs, content = element
                self.program.fromXML(name, attrs, content, ttFont)

    def getCompositeMaxpValues(self, glyfTable, maxComponentDepth=1):
        assert self.isComposite()
        nContours = 0
        nPoints = 0
        initialMaxComponentDepth = maxComponentDepth
        for compo in self.components:
            baseGlyph = glyfTable[compo.glyphName]
            if baseGlyph.numberOfContours == 0:
                continue
            elif baseGlyph.numberOfContours > 0:
                nP, nC = baseGlyph.getMaxpValues()
            else:
                nP, nC, componentDepth = baseGlyph.getCompositeMaxpValues(
                    glyfTable, initialMaxComponentDepth + 1
                )
                maxComponentDepth = max(maxComponentDepth, componentDepth)
            nPoints = nPoints + nP
            nContours = nContours + nC
        return CompositeMaxpValues(nPoints, nContours, maxComponentDepth)

    def getMaxpValues(self):
        assert self.numberOfContours > 0
        return len(self.coordinates), len(self.endPtsOfContours)

    def decompileComponents(self, data, glyfTable):
        self.components = []
        more = 1
        haveInstructions = 0
        while more:
            component = GlyphComponent()
            more, haveInstr, data = component.decompile(data, glyfTable)
            haveInstructions = haveInstructions | haveInstr
            self.components.append(component)
        if haveInstructions:
            (numInstructions,) = struct.unpack(">h", data[:2])
            data = data[2:]
            self.program = ttProgram.Program()
            self.program.fromBytecode(data[:numInstructions])
            data = data[numInstructions:]
            if len(data) >= 4:
                log.warning(
                    "too much glyph data at the end of composite glyph: %d excess bytes",
                    len(data),
                )

    def decompileCoordinates(self, data):
        endPtsOfContours = array.array("H")
        endPtsOfContours.frombytes(data[: 2 * self.numberOfContours])
        if sys.byteorder != "big":
            endPtsOfContours.byteswap()
        self.endPtsOfContours = endPtsOfContours.tolist()

        pos = 2 * self.numberOfContours
        (instructionLength,) = struct.unpack(">h", data[pos : pos + 2])
        self.program = ttProgram.Program()
        self.program.fromBytecode(data[pos + 2 : pos + 2 + instructionLength])
        pos += 2 + instructionLength
        nCoordinates = self.endPtsOfContours[-1] + 1
        flags, xCoordinates, yCoordinates = self.decompileCoordinatesRaw(
            nCoordinates, data, pos
        )

        # fill in repetitions and apply signs
        self.coordinates = coordinates = GlyphCoordinates.zeros(nCoordinates)
        xIndex = 0
        yIndex = 0
        for i in range(nCoordinates):
            flag = flags[i]
            # x coordinate
            if flag & flagXShort:
                if flag & flagXsame:
                    x = xCoordinates[xIndex]
                else:
                    x = -xCoordinates[xIndex]
                xIndex = xIndex + 1
            elif flag & flagXsame:
                x = 0
            else:
                x = xCoordinates[xIndex]
                xIndex = xIndex + 1
            # y coordinate
            if flag & flagYShort:
                if flag & flagYsame:
                    y = yCoordinates[yIndex]
                else:
                    y = -yCoordinates[yIndex]
                yIndex = yIndex + 1
            elif flag & flagYsame:
                y = 0
            else:
                y = yCoordinates[yIndex]
                yIndex = yIndex + 1
            coordinates[i] = (x, y)
        assert xIndex == len(xCoordinates)
        assert yIndex == len(yCoordinates)
        coordinates.relativeToAbsolute()
        # discard all flags except "keepFlags"
        for i in range(len(flags)):
            flags[i] &= keepFlags
        self.flags = flags

    def decompileCoordinatesRaw(self, nCoordinates, data, pos=0):
        # unpack flags and prepare unpacking of coordinates
        flags = bytearray(nCoordinates)
        # Warning: deep Python trickery going on. We use the struct module to unpack
        # the coordinates. We build a format string based on the flags, so we can
        # unpack the coordinates in one struct.unpack() call.
        xFormat = ">"  # big endian
        yFormat = ">"  # big endian
        j = 0
        while True:
            flag = data[pos]
            pos += 1
            repeat = 1
            if flag & flagRepeat:
                repeat = data[pos] + 1
                pos += 1
            for k in range(repeat):
                if flag & flagXShort:
                    xFormat = xFormat + "B"
                elif not (flag & flagXsame):
                    xFormat = xFormat + "h"
                if flag & flagYShort:
                    yFormat = yFormat + "B"
                elif not (flag & flagYsame):
                    yFormat = yFormat + "h"
                flags[j] = flag
                j = j + 1
            if j >= nCoordinates:
                break
        assert j == nCoordinates, "bad glyph flags"
        # unpack raw coordinates, krrrrrr-tching!
        xDataLen = struct.calcsize(xFormat)
        yDataLen = struct.calcsize(yFormat)
        if len(data) - pos - (xDataLen + yDataLen) >= 4:
            log.warning(
                "too much glyph data: %d excess bytes",
                len(data) - pos - (xDataLen + yDataLen),
            )
        xCoordinates = struct.unpack(xFormat, data[pos : pos + xDataLen])
        yCoordinates = struct.unpack(
            yFormat, data[pos + xDataLen : pos + xDataLen + yDataLen]
        )
        return flags, xCoordinates, yCoordinates

    def compileComponents(self, glyfTable):
        data = b""
        lastcomponent = len(self.components) - 1
        more = 1
        haveInstructions = 0
        for i, compo in enumerate(self.components):
            if i == lastcomponent:
                haveInstructions = hasattr(self, "program")
                more = 0
            data = data + compo.compile(more, haveInstructions, glyfTable)
        if haveInstructions:
            instructions = self.program.getBytecode()
            data = data + struct.pack(">h", len(instructions)) + instructions
        return data

    def compileCoordinates(self, *, optimizeSize=True):
        assert len(self.coordinates) == len(self.flags)
        data = []
        endPtsOfContours = array.array("H", self.endPtsOfContours)
        if sys.byteorder != "big":
            endPtsOfContours.byteswap()
        data.append(endPtsOfContours.tobytes())
        instructions = self.program.getBytecode()
        data.append(struct.pack(">h", len(instructions)))
        data.append(instructions)

        deltas = self.coordinates.copy()
        deltas.toInt()
        deltas.absoluteToRelative()

        if optimizeSize:
            # TODO(behdad): Add a configuration option for this?
            deltas = self.compileDeltasGreedy(self.flags, deltas)
            # deltas = self.compileDeltasOptimal(self.flags, deltas)
        else:
            deltas = self.compileDeltasForSpeed(self.flags, deltas)

        data.extend(deltas)
        return b"".join(data)

    def compileDeltasGreedy(self, flags, deltas):
        # Implements greedy algorithm for packing coordinate deltas:
        # uses shortest representation one coordinate at a time.
        compressedFlags = bytearray()
        compressedXs = bytearray()
        compressedYs = bytearray()
        lastflag = None
        repeat = 0
        for flag, (x, y) in zip(flags, deltas):
            # Oh, the horrors of TrueType
            # do x
            if x == 0:
                flag = flag | flagXsame
            elif -255 <= x <= 255:
                flag = flag | flagXShort
                if x > 0:
                    flag = flag | flagXsame
                else:
                    x = -x
                compressedXs.append(x)
            else:
                compressedXs.extend(struct.pack(">h", x))
            # do y
            if y == 0:
                flag = flag | flagYsame
            elif -255 <= y <= 255:
                flag = flag | flagYShort
                if y > 0:
                    flag = flag | flagYsame
                else:
                    y = -y
                compressedYs.append(y)
            else:
                compressedYs.extend(struct.pack(">h", y))
            # handle repeating flags
            if flag == lastflag and repeat != 255:
                repeat = repeat + 1
                if repeat == 1:
                    compressedFlags.append(flag)
                else:
                    compressedFlags[-2] = flag | flagRepeat
                    compressedFlags[-1] = repeat
            else:
                repeat = 0
                compressedFlags.append(flag)
            lastflag = flag
        return (compressedFlags, compressedXs, compressedYs)

    def compileDeltasOptimal(self, flags, deltas):
        # Implements optimal, dynaic-programming, algorithm for packing coordinate
        # deltas.  The savings are negligible :(.
        candidates = []
        bestTuple = None
        bestCost = 0
        repeat = 0
        for flag, (x, y) in zip(flags, deltas):
            # Oh, the horrors of TrueType
            flag, coordBytes = flagBest(x, y, flag)
            bestCost += 1 + coordBytes
            newCandidates = [
                (bestCost, bestTuple, flag, coordBytes),
                (bestCost + 1, bestTuple, (flag | flagRepeat), coordBytes),
            ]
            for lastCost, lastTuple, lastFlag, coordBytes in candidates:
                if (
                    lastCost + coordBytes <= bestCost + 1
                    and (lastFlag & flagRepeat)
                    and (lastFlag < 0xFF00)
                    and flagSupports(lastFlag, flag)
                ):
                    if (lastFlag & 0xFF) == (
                        flag | flagRepeat
                    ) and lastCost == bestCost + 1:
                        continue
                    newCandidates.append(
                        (lastCost + coordBytes, lastTuple, lastFlag + 256, coordBytes)
                    )
            candidates = newCandidates
            bestTuple = min(candidates, key=lambda t: t[0])
            bestCost = bestTuple[0]

        flags = []
        while bestTuple:
            cost, bestTuple, flag, coordBytes = bestTuple
            flags.append(flag)
        flags.reverse()

        compressedFlags = bytearray()
        compressedXs = bytearray()
        compressedYs = bytearray()
        coords = iter(deltas)
        ff = []
        for flag in flags:
            repeatCount, flag = flag >> 8, flag & 0xFF
            compressedFlags.append(flag)
            if flag & flagRepeat:
                assert repeatCount > 0
                compressedFlags.append(repeatCount)
            else:
                assert repeatCount == 0
            for i in range(1 + repeatCount):
                x, y = next(coords)
                flagEncodeCoords(flag, x, y, compressedXs, compressedYs)
                ff.append(flag)
        try:
            next(coords)
            raise Exception("internal error")
        except StopIteration:
            pass

        return (compressedFlags, compressedXs, compressedYs)

    def compileDeltasForSpeed(self, flags, deltas):
        # uses widest representation needed, for all deltas.
        compressedFlags = bytearray()
        compressedXs = bytearray()
        compressedYs = bytearray()

        # Compute the necessary width for each axis
        xs = [d[0] for d in deltas]
        ys = [d[1] for d in deltas]
        minX, minY, maxX, maxY = min(xs), min(ys), max(xs), max(ys)
        xZero = minX == 0 and maxX == 0
        yZero = minY == 0 and maxY == 0
        xShort = -255 <= minX <= maxX <= 255
        yShort = -255 <= minY <= maxY <= 255

        lastflag = None
        repeat = 0
        for flag, (x, y) in zip(flags, deltas):
            # Oh, the horrors of TrueType
            # do x
            if xZero:
                flag = flag | flagXsame
            elif xShort:
                flag = flag | flagXShort
                if x > 0:
                    flag = flag | flagXsame
                else:
                    x = -x
                compressedXs.append(x)
            else:
                compressedXs.extend(struct.pack(">h", x))
            # do y
            if yZero:
                flag = flag | flagYsame
            elif yShort:
                flag = flag | flagYShort
                if y > 0:
                    flag = flag | flagYsame
                else:
                    y = -y
                compressedYs.append(y)
            else:
                compressedYs.extend(struct.pack(">h", y))
            # handle repeating flags
            if flag == lastflag and repeat != 255:
                repeat = repeat + 1
                if repeat == 1:
                    compressedFlags.append(flag)
                else:
                    compressedFlags[-2] = flag | flagRepeat
                    compressedFlags[-1] = repeat
            else:
                repeat = 0
                compressedFlags.append(flag)
            lastflag = flag
        return (compressedFlags, compressedXs, compressedYs)

    def recalcBounds(self, glyfTable, *, boundsDone=None):
        """Recalculates the bounds of the glyph.

        Each glyph object stores its bounding box in the
        ``xMin``/``yMin``/``xMax``/``yMax`` attributes. These bounds must be
        recomputed when the ``coordinates`` change. The ``table__g_l_y_f`` bounds
        must be provided to resolve component bounds.
        """
        if self.isComposite() and self.tryRecalcBoundsComposite(
            glyfTable, boundsDone=boundsDone
        ):
            return
        try:
            coords, endPts, flags = self.getCoordinates(glyfTable, round=otRound)
            self.xMin, self.yMin, self.xMax, self.yMax = coords.calcIntBounds()
        except NotImplementedError:
            pass

    def tryRecalcBoundsComposite(self, glyfTable, *, boundsDone=None):
        """Try recalculating the bounds of a composite glyph that has
        certain constrained properties. Namely, none of the components
        have a transform other than an integer translate, and none
        uses the anchor points.

        Each glyph object stores its bounding box in the
        ``xMin``/``yMin``/``xMax``/``yMax`` attributes. These bounds must be
        recomputed when the ``coordinates`` change. The ``table__g_l_y_f`` bounds
        must be provided to resolve component bounds.

        Return True if bounds were calculated, False otherwise.
        """
        for compo in self.components:
            if not compo._hasOnlyIntegerTranslate():
                return False

        # All components are untransformed and have an integer x/y translate
        bounds = None
        for compo in self.components:
            glyphName = compo.glyphName
            g = glyfTable[glyphName]

            if boundsDone is None or glyphName not in boundsDone:
                try:
                    g.recalcBounds(glyfTable, boundsDone=boundsDone)
                except RecursionError:
                    raise ttLib.TTLibError(
                        "glyph '%s' contains a recursive component reference"
                        % glyphName
                    )
                if boundsDone is not None:
                    boundsDone.add(glyphName)
            # empty components shouldn't update the bounds of the parent glyph
            if g.yMin == g.yMax and g.xMin == g.xMax:
                continue

            x, y = compo.x, compo.y
            bounds = updateBounds(bounds, (g.xMin + x, g.yMin + y))
            bounds = updateBounds(bounds, (g.xMax + x, g.yMax + y))

        if bounds is None:
            bounds = (0, 0, 0, 0)
        self.xMin, self.yMin, self.xMax, self.yMax = bounds
        return True

    def isComposite(self):
        """Test whether a glyph has components"""
        if hasattr(self, "data"):
            return struct.unpack(">h", self.data[:2])[0] == -1 if self.data else False
        else:
            return self.numberOfContours == -1

    def getCoordinates(self, glyfTable, *, round=noRound):
        """Return the coordinates, end points and flags

        This method returns three values: A :py:class:`GlyphCoordinates` object,
        a list of the indexes of the final points of each contour (allowing you
        to split up the coordinates list into contours) and a list of flags.

        On simple glyphs, this method returns information from the glyph's own
        contours; on composite glyphs, it "flattens" all components recursively
        to return a list of coordinates representing all the components involved
        in the glyph.

        To interpret the flags for each point, see the "Simple Glyph Flags"
        section of the `glyf table specification <https://docs.microsoft.com/en-us/typography/opentype/spec/glyf#simple-glyph-description>`.
        """

        if self.numberOfContours > 0:
            return self.coordinates, self.endPtsOfContours, self.flags
        elif self.isComposite():
            # it's a composite
            allCoords = GlyphCoordinates()
            allFlags = bytearray()
            allEndPts = []
            for compo in self.components:
                g = glyfTable[compo.glyphName]
                try:
                    coordinates, endPts, flags = g.getCoordinates(
                        glyfTable, round=round
                    )
                except RecursionError:
                    raise ttLib.TTLibError(
                        "glyph '%s' contains a recursive component reference"
                        % compo.glyphName
                    )
                coordinates = GlyphCoordinates(coordinates)
                # if asked to round e.g. while computing bboxes, it's important we
                # do it immediately before a component transform is applied to a
                # simple glyph's coordinates in case these might still contain floats;
                # however, if the referenced component glyph is another composite, we
                # must not round here but only at the end, after all the nested
                # transforms have been applied, or else rounding errors will compound.
                if round is not noRound and g.numberOfContours > 0:
                    coordinates.toInt(round=round)
                if hasattr(compo, "firstPt"):
                    # component uses two reference points: we apply the transform _before_
                    # computing the offset between the points
                    if hasattr(compo, "transform"):
                        coordinates.transform(compo.transform)
                    x1, y1 = allCoords[compo.firstPt]
                    x2, y2 = coordinates[compo.secondPt]
                    move = x1 - x2, y1 - y2
                    coordinates.translate(move)
                else:
                    # component uses XY offsets
                    move = compo.x, compo.y
                    if not hasattr(compo, "transform"):
                        coordinates.translate(move)
                    else:
                        apple_way = compo.flags & SCALED_COMPONENT_OFFSET
                        ms_way = compo.flags & UNSCALED_COMPONENT_OFFSET
                        assert not (apple_way and ms_way)
                        if not (apple_way or ms_way):
                            scale_component_offset = (
                                SCALE_COMPONENT_OFFSET_DEFAULT  # see top of this file
                            )
                        else:
                            scale_component_offset = apple_way
                        if scale_component_offset:
                            # the Apple way: first move, then scale (ie. scale the component offset)
                            coordinates.translate(move)
                            coordinates.transform(compo.transform)
                        else:
                            # the MS way: first scale, then move
                            coordinates.transform(compo.transform)
                            coordinates.translate(move)
                offset = len(allCoords)
                allEndPts.extend(e + offset for e in endPts)
                allCoords.extend(coordinates)
                allFlags.extend(flags)
            return allCoords, allEndPts, allFlags
        else:
            return GlyphCoordinates(), [], bytearray()

    def getComponentNames(self, glyfTable):
        """Returns a list of names of component glyphs used in this glyph

        This method can be used on simple glyphs (in which case it returns an
        empty list) or composite glyphs.
        """
        if not hasattr(self, "data"):
            if self.isComposite():
                return [c.glyphName for c in self.components]
            else:
                return []

        # Extract components without expanding glyph

        if not self.data or struct.unpack(">h", self.data[:2])[0] >= 0:
            return []  # Not composite

        data = self.data
        i = 10
        components = []
        more = 1
        while more:
            flags, glyphID = struct.unpack(">HH", data[i : i + 4])
            i += 4
            flags = int(flags)
            components.append(glyfTable.getGlyphName(int(glyphID)))

            if flags & ARG_1_AND_2_ARE_WORDS:
                i += 4
            else:
                i += 2
            if flags & WE_HAVE_A_SCALE:
                i += 2
            elif flags & WE_HAVE_AN_X_AND_Y_SCALE:
                i += 4
            elif flags & WE_HAVE_A_TWO_BY_TWO:
                i += 8
            more = flags & MORE_COMPONENTS

        return components

    def trim(self, remove_hinting=False):
        """Remove padding and, if requested, hinting, from a glyph.
        This works on both expanded and compacted glyphs, without
        expanding it."""
        if not hasattr(self, "data"):
            if remove_hinting:
                if self.isComposite():
                    if hasattr(self, "program"):
                        del self.program
                else:
                    self.program = ttProgram.Program()
                    self.program.fromBytecode([])
            # No padding to trim.
            return
        if not self.data:
            return
        numContours = struct.unpack(">h", self.data[:2])[0]
        data = bytearray(self.data)
        i = 10
        if numContours >= 0:
            i += 2 * numContours  # endPtsOfContours
            nCoordinates = ((data[i - 2] << 8) | data[i - 1]) + 1
            instructionLen = (data[i] << 8) | data[i + 1]
            if remove_hinting:
                # Zero instruction length
                data[i] = data[i + 1] = 0
                i += 2
                if instructionLen:
                    # Splice it out
                    data = data[:i] + data[i + instructionLen :]
                instructionLen = 0
            else:
                i += 2 + instructionLen

            coordBytes = 0
            j = 0
            while True:
                flag = data[i]
                i = i + 1
                repeat = 1
                if flag & flagRepeat:
                    repeat = data[i] + 1
                    i = i + 1
                xBytes = yBytes = 0
                if flag & flagXShort:
                    xBytes = 1
                elif not (flag & flagXsame):
                    xBytes = 2
                if flag & flagYShort:
                    yBytes = 1
                elif not (flag & flagYsame):
                    yBytes = 2
                coordBytes += (xBytes + yBytes) * repeat
                j += repeat
                if j >= nCoordinates:
                    break
            assert j == nCoordinates, "bad glyph flags"
            i += coordBytes
            # Remove padding
            data = data[:i]
        elif self.isComposite():
            more = 1
            we_have_instructions = False
            while more:
                flags = (data[i] << 8) | data[i + 1]
                if remove_hinting:
                    flags &= ~WE_HAVE_INSTRUCTIONS
                if flags & WE_HAVE_INSTRUCTIONS:
                    we_have_instructions = True
                data[i + 0] = flags >> 8
                data[i + 1] = flags & 0xFF
                i += 4
                flags = int(flags)

                if flags & ARG_1_AND_2_ARE_WORDS:
                    i += 4
                else:
                    i += 2
                if flags & WE_HAVE_A_SCALE:
                    i += 2
                elif flags & WE_HAVE_AN_X_AND_Y_SCALE:
                    i += 4
                elif flags & WE_HAVE_A_TWO_BY_TWO:
                    i += 8
                more = flags & MORE_COMPONENTS
            if we_have_instructions:
                instructionLen = (data[i] << 8) | data[i + 1]
                i += 2 + instructionLen
            # Remove padding
            data = data[:i]

        self.data = data

    def removeHinting(self):
        """Removes TrueType hinting instructions from the glyph."""
        self.trim(remove_hinting=True)

    def draw(self, pen, glyfTable, offset=0):
        """Draws the glyph using the supplied pen object.

        Arguments:
                pen: An object conforming to the pen protocol.
                glyfTable: A :py:class:`table__g_l_y_f` object, to resolve components.
                offset (int): A horizontal offset. If provided, all coordinates are
                        translated by this offset.
        """

        if self.isComposite():
            for component in self.components:
                glyphName, transform = component.getComponentInfo()
                pen.addComponent(glyphName, transform)
            return

        self.expand(glyfTable)
        coordinates, endPts, flags = self.getCoordinates(glyfTable)
        if offset:
            coordinates = coordinates.copy()
            coordinates.translate((offset, 0))
        start = 0
        maybeInt = lambda v: int(v) if v == int(v) else v
        for end in endPts:
            end = end + 1
            contour = coordinates[start:end]
            cFlags = [flagOnCurve & f for f in flags[start:end]]
            cuFlags = [flagCubic & f for f in flags[start:end]]
            start = end
            if 1 not in cFlags:
                assert all(cuFlags) or not any(cuFlags)
                cubic = all(cuFlags)
                if cubic:
                    count = len(contour)
                    assert count % 2 == 0, "Odd number of cubic off-curves undefined"
                    l = contour[-1]
                    f = contour[0]
                    p0 = (maybeInt((l[0] + f[0]) * 0.5), maybeInt((l[1] + f[1]) * 0.5))
                    pen.moveTo(p0)
                    for i in range(0, count, 2):
                        p1 = contour[i]
                        p2 = contour[i + 1]
                        p4 = contour[i + 2 if i + 2 < count else 0]
                        p3 = (
                            maybeInt((p2[0] + p4[0]) * 0.5),
                            maybeInt((p2[1] + p4[1]) * 0.5),
                        )
                        pen.curveTo(p1, p2, p3)
                else:
                    # There is not a single on-curve point on the curve,
                    # use pen.qCurveTo's special case by specifying None
                    # as the on-curve point.
                    contour.append(None)
                    pen.qCurveTo(*contour)
            else:
                # Shuffle the points so that the contour is guaranteed
                # to *end* in an on-curve point, which we'll use for
                # the moveTo.
                firstOnCurve = cFlags.index(1) + 1
                contour = contour[firstOnCurve:] + contour[:firstOnCurve]
                cFlags = cFlags[firstOnCurve:] + cFlags[:firstOnCurve]
                cuFlags = cuFlags[firstOnCurve:] + cuFlags[:firstOnCurve]
                pen.moveTo(contour[-1])
                while contour:
                    nextOnCurve = cFlags.index(1) + 1
                    if nextOnCurve == 1:
                        # Skip a final lineTo(), as it is implied by
                        # pen.closePath()
                        if len(contour) > 1:
                            pen.lineTo(contour[0])
                    else:
                        cubicFlags = [f for f in cuFlags[: nextOnCurve - 1]]
                        assert all(cubicFlags) or not any(cubicFlags)
                        cubic = any(cubicFlags)
                        if cubic:
                            assert all(
                                cubicFlags
                            ), "Mixed cubic and quadratic segment undefined"

                            count = nextOnCurve
                            assert (
                                count >= 3
                            ), "At least two cubic off-curve points required"
                            assert (
                                count - 1
                            ) % 2 == 0, "Odd number of cubic off-curves undefined"
                            for i in range(0, count - 3, 2):
                                p1 = contour[i]
                                p2 = contour[i + 1]
                                p4 = contour[i + 2]
                                p3 = (
                                    maybeInt((p2[0] + p4[0]) * 0.5),
                                    maybeInt((p2[1] + p4[1]) * 0.5),
                                )
                                lastOnCurve = p3
                                pen.curveTo(p1, p2, p3)
                            pen.curveTo(*contour[count - 3 : count])
                        else:
                            pen.qCurveTo(*contour[:nextOnCurve])
                    contour = contour[nextOnCurve:]
                    cFlags = cFlags[nextOnCurve:]
                    cuFlags = cuFlags[nextOnCurve:]
            pen.closePath()

    def drawPoints(self, pen, glyfTable, offset=0):
        """Draw the glyph using the supplied pointPen. As opposed to Glyph.draw(),
        this will not change the point indices.
        """

        if self.isComposite():
            for component in self.components:
                glyphName, transform = component.getComponentInfo()
                pen.addComponent(glyphName, transform)
            return

        coordinates, endPts, flags = self.getCoordinates(glyfTable)
        if offset:
            coordinates = coordinates.copy()
            coordinates.translate((offset, 0))
        start = 0
        for end in endPts:
            end = end + 1
            contour = coordinates[start:end]
            cFlags = flags[start:end]
            start = end
            pen.beginPath()
            # Start with the appropriate segment type based on the final segment

            if cFlags[-1] & flagOnCurve:
                segmentType = "line"
            elif cFlags[-1] & flagCubic:
                segmentType = "curve"
            else:
                segmentType = "qcurve"
            for i, pt in enumerate(contour):
                if cFlags[i] & flagOnCurve:
                    pen.addPoint(pt, segmentType=segmentType)
                    segmentType = "line"
                else:
                    pen.addPoint(pt)
                    segmentType = "curve" if cFlags[i] & flagCubic else "qcurve"
            pen.endPath()

    def __eq__(self, other):
        if type(self) != type(other):
            return NotImplemented
        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        result = self.__eq__(other)
        return result if result is NotImplemented else not result


# Vector.__round__ uses the built-in (Banker's) `round` but we want
# to use otRound below
_roundv = partial(Vector.__round__, round=otRound)


def _is_mid_point(p0: tuple, p1: tuple, p2: tuple) -> bool:
    # True if p1 is in the middle of p0 and p2, either before or after rounding
    p0 = Vector(p0)
    p1 = Vector(p1)
    p2 = Vector(p2)
    return ((p0 + p2) * 0.5).isclose(p1) or _roundv(p0) + _roundv(p2) == _roundv(p1) * 2


def dropImpliedOnCurvePoints(*interpolatable_glyphs: Glyph) -> Set[int]:
    """Drop impliable on-curve points from the (simple) glyph or glyphs.

    In TrueType glyf outlines, on-curve points can be implied when they are located at
    the midpoint of the line connecting two consecutive off-curve points.

    If more than one glyphs are passed, these are assumed to be interpolatable masters
    of the same glyph impliable, and thus only the on-curve points that are impliable
    for all of them will actually be implied.
    Composite glyphs or empty glyphs are skipped, only simple glyphs with 1 or more
    contours are considered.
    The input glyph(s) is/are modified in-place.

    Args:
        interpolatable_glyphs: The glyph or glyphs to modify in-place.

    Returns:
        The set of point indices that were dropped if any.

    Raises:
        ValueError if simple glyphs are not in fact interpolatable because they have
        different point flags or number of contours.

    Reference:
    https://developer.apple.com/fonts/TrueType-Reference-Manual/RM01/Chap1.html
    """
    staticAttributes = SimpleNamespace(
        numberOfContours=None, flags=None, endPtsOfContours=None
    )
    drop = None
    simple_glyphs = []
    for i, glyph in enumerate(interpolatable_glyphs):
        if glyph.numberOfContours < 1:
            # ignore composite or empty glyphs
            continue

        for attr in staticAttributes.__dict__:
            expected = getattr(staticAttributes, attr)
            found = getattr(glyph, attr)
            if expected is None:
                setattr(staticAttributes, attr, found)
            elif expected != found:
                raise ValueError(
                    f"Incompatible {attr} for glyph at master index {i}: "
                    f"expected {expected}, found {found}"
                )

        may_drop = set()
        start = 0
        coords = glyph.coordinates
        flags = staticAttributes.flags
        endPtsOfContours = staticAttributes.endPtsOfContours
        for last in endPtsOfContours:
            for i in range(start, last + 1):
                if not (flags[i] & flagOnCurve):
                    continue
                prv = i - 1 if i > start else last
                nxt = i + 1 if i < last else start
                if (flags[prv] & flagOnCurve) or flags[prv] != flags[nxt]:
                    continue
                # we may drop the ith on-curve if halfway between previous/next off-curves
                if not _is_mid_point(coords[prv], coords[i], coords[nxt]):
                    continue

                may_drop.add(i)
            start = last + 1
        # we only want to drop if ALL interpolatable glyphs have the same implied oncurves
        if drop is None:
            drop = may_drop
        else:
            drop.intersection_update(may_drop)

        simple_glyphs.append(glyph)

    if drop:
        # Do the actual dropping
        flags = staticAttributes.flags
        assert flags is not None
        newFlags = array.array(
            "B", (flags[i] for i in range(len(flags)) if i not in drop)
        )

        endPts = staticAttributes.endPtsOfContours
        assert endPts is not None
        newEndPts = []
        i = 0
        delta = 0
        for d in sorted(drop):
            while d > endPts[i]:
                newEndPts.append(endPts[i] - delta)
                i += 1
            delta += 1
        while i < len(endPts):
            newEndPts.append(endPts[i] - delta)
            i += 1

        for glyph in simple_glyphs:
            coords = glyph.coordinates
            glyph.coordinates = GlyphCoordinates(
                coords[i] for i in range(len(coords)) if i not in drop
            )
            glyph.flags = newFlags
            glyph.endPtsOfContours = newEndPts

    return drop if drop is not None else set()


class GlyphComponent(object):
    """Represents a component within a composite glyph.

    The component is represented internally with four attributes: ``glyphName``,
    ``x``, ``y`` and ``transform``. If there is no "two-by-two" matrix (i.e
    no scaling, reflection, or rotation; only translation), the ``transform``
    attribute is not present.
    """

    # The above documentation is not *completely* true, but is *true enough* because
    # the rare firstPt/lastPt attributes are not totally supported and nobody seems to
    # mind - see below.

    def __init__(self):
        pass

    def getComponentInfo(self):
        """Return information about the component

        This method returns a tuple of two values: the glyph name of the component's
        base glyph, and a transformation matrix. As opposed to accessing the attributes
        directly, ``getComponentInfo`` always returns a six-element tuple of the
        component's transformation matrix, even when the two-by-two ``.transform``
        matrix is not present.
        """
        # XXX Ignoring self.firstPt & self.lastpt for now: I need to implement
        # something equivalent in fontTools.objects.glyph (I'd rather not
        # convert it to an absolute offset, since it is valuable information).
        # This method will now raise "AttributeError: x" on glyphs that use
        # this TT feature.
        if hasattr(self, "transform"):
            [[xx, xy], [yx, yy]] = self.transform
            trans = (xx, xy, yx, yy, self.x, self.y)
        else:
            trans = (1, 0, 0, 1, self.x, self.y)
        return self.glyphName, trans

    def decompile(self, data, glyfTable):
        flags, glyphID = struct.unpack(">HH", data[:4])
        self.flags = int(flags)
        glyphID = int(glyphID)
        self.glyphName = glyfTable.getGlyphName(int(glyphID))
        data = data[4:]

        if self.flags & ARG_1_AND_2_ARE_WORDS:
            if self.flags & ARGS_ARE_XY_VALUES:
                self.x, self.y = struct.unpack(">hh", data[:4])
            else:
                x, y = struct.unpack(">HH", data[:4])
                self.firstPt, self.secondPt = int(x), int(y)
            data = data[4:]
        else:
            if self.flags & ARGS_ARE_XY_VALUES:
                self.x, self.y = struct.unpack(">bb", data[:2])
            else:
                x, y = struct.unpack(">BB", data[:2])
                self.firstPt, self.secondPt = int(x), int(y)
            data = data[2:]

        if self.flags & WE_HAVE_A_SCALE:
            (scale,) = struct.unpack(">h", data[:2])
            self.transform = [
                [fi2fl(scale, 14), 0],
                [0, fi2fl(scale, 14)],
            ]  # fixed 2.14
            data = data[2:]
        elif self.flags & WE_HAVE_AN_X_AND_Y_SCALE:
            xscale, yscale = struct.unpack(">hh", data[:4])
            self.transform = [
                [fi2fl(xscale, 14), 0],
                [0, fi2fl(yscale, 14)],
            ]  # fixed 2.14
            data = data[4:]
        elif self.flags & WE_HAVE_A_TWO_BY_TWO:
            xscale, scale01, scale10, yscale = struct.unpack(">hhhh", data[:8])
            self.transform = [
                [fi2fl(xscale, 14), fi2fl(scale01, 14)],
                [fi2fl(scale10, 14), fi2fl(yscale, 14)],
            ]  # fixed 2.14
            data = data[8:]
        more = self.flags & MORE_COMPONENTS
        haveInstructions = self.flags & WE_HAVE_INSTRUCTIONS
        self.flags = self.flags & (
            ROUND_XY_TO_GRID
            | USE_MY_METRICS
            | SCALED_COMPONENT_OFFSET
            | UNSCALED_COMPONENT_OFFSET
            | NON_OVERLAPPING
            | OVERLAP_COMPOUND
        )
        return more, haveInstructions, data

    def compile(self, more, haveInstructions, glyfTable):
        data = b""

        # reset all flags we will calculate ourselves
        flags = self.flags & (
            ROUND_XY_TO_GRID
            | USE_MY_METRICS
            | SCALED_COMPONENT_OFFSET
            | UNSCALED_COMPONENT_OFFSET
            | NON_OVERLAPPING
            | OVERLAP_COMPOUND
        )
        if more:
            flags = flags | MORE_COMPONENTS
        if haveInstructions:
            flags = flags | WE_HAVE_INSTRUCTIONS

        if hasattr(self, "firstPt"):
            if (0 <= self.firstPt <= 255) and (0 <= self.secondPt <= 255):
                data = data + struct.pack(">BB", self.firstPt, self.secondPt)
            else:
                data = data + struct.pack(">HH", self.firstPt, self.secondPt)
                flags = flags | ARG_1_AND_2_ARE_WORDS
        else:
            x = otRound(self.x)
            y = otRound(self.y)
            flags = flags | ARGS_ARE_XY_VALUES
            if (-128 <= x <= 127) and (-128 <= y <= 127):
                data = data + struct.pack(">bb", x, y)
            else:
                data = data + struct.pack(">hh", x, y)
                flags = flags | ARG_1_AND_2_ARE_WORDS

        if hasattr(self, "transform"):
            transform = [[fl2fi(x, 14) for x in row] for row in self.transform]
            if transform[0][1] or transform[1][0]:
                flags = flags | WE_HAVE_A_TWO_BY_TWO
                data = data + struct.pack(
                    ">hhhh",
                    transform[0][0],
                    transform[0][1],
                    transform[1][0],
                    transform[1][1],
                )
            elif transform[0][0] != transform[1][1]:
                flags = flags | WE_HAVE_AN_X_AND_Y_SCALE
                data = data + struct.pack(">hh", transform[0][0], transform[1][1])
            else:
                flags = flags | WE_HAVE_A_SCALE
                data = data + struct.pack(">h", transform[0][0])

        glyphID = glyfTable.getGlyphID(self.glyphName)
        return struct.pack(">HH", flags, glyphID) + data

    def toXML(self, writer, ttFont):
        attrs = [("glyphName", self.glyphName)]
        if not hasattr(self, "firstPt"):
            attrs = attrs + [("x", self.x), ("y", self.y)]
        else:
            attrs = attrs + [("firstPt", self.firstPt), ("secondPt", self.secondPt)]

        if hasattr(self, "transform"):
            transform = self.transform
            if transform[0][1] or transform[1][0]:
                attrs = attrs + [
                    ("scalex", fl2str(transform[0][0], 14)),
                    ("scale01", fl2str(transform[0][1], 14)),
                    ("scale10", fl2str(transform[1][0], 14)),
                    ("scaley", fl2str(transform[1][1], 14)),
                ]
            elif transform[0][0] != transform[1][1]:
                attrs = attrs + [
                    ("scalex", fl2str(transform[0][0], 14)),
                    ("scaley", fl2str(transform[1][1], 14)),
                ]
            else:
                attrs = attrs + [("scale", fl2str(transform[0][0], 14))]
        attrs = attrs + [("flags", hex(self.flags))]
        writer.simpletag("component", attrs)
        writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        self.glyphName = attrs["glyphName"]
        if "firstPt" in attrs:
            self.firstPt = safeEval(attrs["firstPt"])
            self.secondPt = safeEval(attrs["secondPt"])
        else:
            self.x = safeEval(attrs["x"])
            self.y = safeEval(attrs["y"])
        if "scale01" in attrs:
            scalex = str2fl(attrs["scalex"], 14)
            scale01 = str2fl(attrs["scale01"], 14)
            scale10 = str2fl(attrs["scale10"], 14)
            scaley = str2fl(attrs["scaley"], 14)
            self.transform = [[scalex, scale01], [scale10, scaley]]
        elif "scalex" in attrs:
            scalex = str2fl(attrs["scalex"], 14)
            scaley = str2fl(attrs["scaley"], 14)
            self.transform = [[scalex, 0], [0, scaley]]
        elif "scale" in attrs:
            scale = str2fl(attrs["scale"], 14)
            self.transform = [[scale, 0], [0, scale]]
        self.flags = safeEval(attrs["flags"])

    def __eq__(self, other):
        if type(self) != type(other):
            return NotImplemented
        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        result = self.__eq__(other)
        return result if result is NotImplemented else not result

    def _hasOnlyIntegerTranslate(self):
        """Return True if it's a 'simple' component.

        That is, it has no anchor points and no transform other than integer translate.
        """
        return (
            not hasattr(self, "firstPt")
            and not hasattr(self, "transform")
            and float(self.x).is_integer()
            and float(self.y).is_integer()
        )


class GlyphCoordinates(object):
    """A list of glyph coordinates.

    Unlike an ordinary list, this is a numpy-like matrix object which supports
    matrix addition, scalar multiplication and other operations described below.
    """

    def __init__(self, iterable=[]):
        self._a = array.array("d")
        self.extend(iterable)

    @property
    def array(self):
        """Returns the underlying array of coordinates"""
        return self._a

    @staticmethod
    def zeros(count):
        """Creates a new ``GlyphCoordinates`` object with all coordinates set to (0,0)"""
        g = GlyphCoordinates()
        g._a.frombytes(bytes(count * 2 * g._a.itemsize))
        return g

    def copy(self):
        """Creates a new ``GlyphCoordinates`` object which is a copy of the current one."""
        c = GlyphCoordinates()
        c._a.extend(self._a)
        return c

    def __len__(self):
        """Returns the number of coordinates in the array."""
        return len(self._a) // 2

    def __getitem__(self, k):
        """Returns a two element tuple (x,y)"""
        a = self._a
        if isinstance(k, slice):
            indices = range(*k.indices(len(self)))
            # Instead of calling ourselves recursively, duplicate code; faster
            ret = []
            for k in indices:
                x = a[2 * k]
                y = a[2 * k + 1]
                ret.append(
                    (int(x) if x.is_integer() else x, int(y) if y.is_integer() else y)
                )
            return ret
        x = a[2 * k]
        y = a[2 * k + 1]
        return (int(x) if x.is_integer() else x, int(y) if y.is_integer() else y)

    def __setitem__(self, k, v):
        """Sets a point's coordinates to a two element tuple (x,y)"""
        if isinstance(k, slice):
            indices = range(*k.indices(len(self)))
            # XXX This only works if len(v) == len(indices)
            for j, i in enumerate(indices):
                self[i] = v[j]
            return
        self._a[2 * k], self._a[2 * k + 1] = v

    def __delitem__(self, i):
        """Removes a point from the list"""
        i = (2 * i) % len(self._a)
        del self._a[i]
        del self._a[i]

    def __repr__(self):
        return "GlyphCoordinates([" + ",".join(str(c) for c in self) + "])"

    def append(self, p):
        self._a.extend(tuple(p))

    def extend(self, iterable):
        for p in iterable:
            self._a.extend(p)

    def toInt(self, *, round=otRound):
        if round is noRound:
            return
        a = self._a
        for i, value in enumerate(a):
            a[i] = round(value)

    def calcBounds(self):
        a = self._a
        if not a:
            return 0, 0, 0, 0
        xs = a[0::2]
        ys = a[1::2]
        return min(xs), min(ys), max(xs), max(ys)

    def calcIntBounds(self, round=otRound):
        return tuple(round(v) for v in self.calcBounds())

    def relativeToAbsolute(self):
        a = self._a
        x, y = 0, 0
        for i in range(0, len(a), 2):
            a[i] = x = a[i] + x
            a[i + 1] = y = a[i + 1] + y

    def absoluteToRelative(self):
        a = self._a
        x, y = 0, 0
        for i in range(0, len(a), 2):
            nx = a[i]
            ny = a[i + 1]
            a[i] = nx - x
            a[i + 1] = ny - y
            x = nx
            y = ny

    def translate(self, p):
        """
        >>> GlyphCoordinates([(1,2)]).translate((.5,0))
        """
        x, y = p
        if x == 0 and y == 0:
            return
        a = self._a
        for i in range(0, len(a), 2):
            a[i] += x
            a[i + 1] += y

    def scale(self, p):
        """
        >>> GlyphCoordinates([(1,2)]).scale((.5,0))
        """
        x, y = p
        if x == 1 and y == 1:
            return
        a = self._a
        for i in range(0, len(a), 2):
            a[i] *= x
            a[i + 1] *= y

    def transform(self, t):
        """
        >>> GlyphCoordinates([(1,2)]).transform(((.5,0),(.2,.5)))
        """
        a = self._a
        for i in range(0, len(a), 2):
            x = a[i]
            y = a[i + 1]
            px = x * t[0][0] + y * t[1][0]
            py = x * t[0][1] + y * t[1][1]
            a[i] = px
            a[i + 1] = py

    def __eq__(self, other):
        """
        >>> g = GlyphCoordinates([(1,2)])
        >>> g2 = GlyphCoordinates([(1.0,2)])
        >>> g3 = GlyphCoordinates([(1.5,2)])
        >>> g == g2
        True
        >>> g == g3
        False
        >>> g2 == g3
        False
        """
        if type(self) != type(other):
            return NotImplemented
        return self._a == other._a

    def __ne__(self, other):
        """
        >>> g = GlyphCoordinates([(1,2)])
        >>> g2 = GlyphCoordinates([(1.0,2)])
        >>> g3 = GlyphCoordinates([(1.5,2)])
        >>> g != g2
        False
        >>> g != g3
        True
        >>> g2 != g3
        True
        """
        result = self.__eq__(other)
        return result if result is NotImplemented else not result

    # Math operations

    def __pos__(self):
        """
        >>> g = GlyphCoordinates([(1,2)])
        >>> g
        GlyphCoordinates([(1, 2)])
        >>> g2 = +g
        >>> g2
        GlyphCoordinates([(1, 2)])
        >>> g2.translate((1,0))
        >>> g2
        GlyphCoordinates([(2, 2)])
        >>> g
        GlyphCoordinates([(1, 2)])
        """
        return self.copy()

    def __neg__(self):
        """
        >>> g = GlyphCoordinates([(1,2)])
        >>> g
        GlyphCoordinates([(1, 2)])
        >>> g2 = -g
        >>> g2
        GlyphCoordinates([(-1, -2)])
        >>> g
        GlyphCoordinates([(1, 2)])
        """
        r = self.copy()
        a = r._a
        for i, value in enumerate(a):
            a[i] = -value
        return r

    def __round__(self, *, round=otRound):
        r = self.copy()
        r.toInt(round=round)
        return r

    def __add__(self, other):
        return self.copy().__iadd__(other)

    def __sub__(self, other):
        return self.copy().__isub__(other)

    def __mul__(self, other):
        return self.copy().__imul__(other)

    def __truediv__(self, other):
        return self.copy().__itruediv__(other)

    __radd__ = __add__
    __rmul__ = __mul__

    def __rsub__(self, other):
        return other + (-self)

    def __iadd__(self, other):
        """
        >>> g = GlyphCoordinates([(1,2)])
        >>> g += (.5,0)
        >>> g
        GlyphCoordinates([(1.5, 2)])
        >>> g2 = GlyphCoordinates([(3,4)])
        >>> g += g2
        >>> g
        GlyphCoordinates([(4.5, 6)])
        """
        if isinstance(other, tuple):
            assert len(other) == 2
            self.translate(other)
            return self
        if isinstance(other, GlyphCoordinates):
            other = other._a
            a = self._a
            assert len(a) == len(other)
            for i, value in enumerate(other):
                a[i] += value
            return self
        return NotImplemented

    def __isub__(self, other):
        """
        >>> g = GlyphCoordinates([(1,2)])
        >>> g -= (.5,0)
        >>> g
        GlyphCoordinates([(0.5, 2)])
        >>> g2 = GlyphCoordinates([(3,4)])
        >>> g -= g2
        >>> g
        GlyphCoordinates([(-2.5, -2)])
        """
        if isinstance(other, tuple):
            assert len(other) == 2
            self.translate((-other[0], -other[1]))
            return self
        if isinstance(other, GlyphCoordinates):
            other = other._a
            a = self._a
            assert len(a) == len(other)
            for i, value in enumerate(other):
                a[i] -= value
            return self
        return NotImplemented

    def __imul__(self, other):
        """
        >>> g = GlyphCoordinates([(1,2)])
        >>> g *= (2,.5)
        >>> g *= 2
        >>> g
        GlyphCoordinates([(4, 2)])
        >>> g = GlyphCoordinates([(1,2)])
        >>> g *= 2
        >>> g
        GlyphCoordinates([(2, 4)])
        """
        if isinstance(other, tuple):
            assert len(other) == 2
            self.scale(other)
            return self
        if isinstance(other, Number):
            if other == 1:
                return self
            a = self._a
            for i in range(len(a)):
                a[i] *= other
            return self
        return NotImplemented

    def __itruediv__(self, other):
        """
        >>> g = GlyphCoordinates([(1,3)])
        >>> g /= (.5,1.5)
        >>> g /= 2
        >>> g
        GlyphCoordinates([(1, 1)])
        """
        if isinstance(other, Number):
            other = (other, other)
        if isinstance(other, tuple):
            if other == (1, 1):
                return self
            assert len(other) == 2
            self.scale((1.0 / other[0], 1.0 / other[1]))
            return self
        return NotImplemented

    def __bool__(self):
        """
        >>> g = GlyphCoordinates([])
        >>> bool(g)
        False
        >>> g = GlyphCoordinates([(0,0), (0.,0)])
        >>> bool(g)
        True
        >>> g = GlyphCoordinates([(0,0), (1,0)])
        >>> bool(g)
        True
        >>> g = GlyphCoordinates([(0,.5), (0,0)])
        >>> bool(g)
        True
        """
        return bool(self._a)

    __nonzero__ = __bool__


if __name__ == "__main__":
    import doctest, sys

    sys.exit(doctest.testmod().failed)
<|endoftext|>from fontTools.misc import sstruct
from fontTools.misc.textTools import (
    bytechr,
    byteord,
    bytesjoin,
    strjoin,
    safeEval,
    readHex,
    hexStr,
    deHexStr,
)
from .BitmapGlyphMetrics import (
    BigGlyphMetrics,
    bigGlyphMetricsFormat,
    SmallGlyphMetrics,
    smallGlyphMetricsFormat,
)
from . import DefaultTable
import itertools
import os
import re
from urllib.parse import quote

# a glyph name is usable as a filename as-is unless it contains a path
# separator (on any platform, not just this one) or the '%' that marks an
# encoded name
_UNSAFE_IN_FILENAME = re.compile(r"[%/\\]")
import struct
import logging

log = logging.getLogger(__name__)

ebdtTableVersionFormat = """
	> # big endian
	version: 16.16F
"""

ebdtComponentFormat = """
	> # big endian
	glyphCode: H
	xOffset:   b
	yOffset:   b
"""


class table_E_B_D_T_(DefaultTable.DefaultTable):
    """Embedded Bitmap Data table

    The ``EBDT`` table contains monochrome or grayscale bitmap data for
    glyphs. It must be used in concert with the ``EBLC`` table.

    See also https://learn.microsoft.com/en-us/typography/opentype/spec/ebdt
    """

    # Keep a reference to the name of the data locator table.
    locatorName = "EBLC"

    # This method can be overridden in subclasses to support new formats
    # without changing the other implementation. Also can be used as a
    # convenience method for coverting a font file to an alternative format.
    def getImageFormatClass(self, imageFormat):
        return ebdt_bitmap_classes[imageFormat]

    def decompile(self, data, ttFont):
        # Get the version but don't advance the slice.
        # Most of the lookup for this table is done relative
        # to the begining so slice by the offsets provided
        # in the EBLC table.
        sstruct.unpack2(ebdtTableVersionFormat, data, self)

        # Keep a dict of glyphs that have been seen so they aren't remade.
        # This dict maps intervals of data to the BitmapGlyph.
        glyphDict = {}

        # Pull out the EBLC table and loop through glyphs.
        # A strike is a concept that spans both tables.
        # The actual bitmap data is stored in the EBDT.
        locator = ttFont[self.__class__.locatorName]
        self.strikeData = []
        for curStrike in locator.strikes:
            bitmapGlyphDict = {}
            self.strikeData.append(bitmapGlyphDict)
            for indexSubTable in curStrike.indexSubTables:
                dataIter = zip(indexSubTable.names, indexSubTable.locations)
                for curName, curLoc in dataIter:
                    # Don't create duplicate data entries for the same glyphs.
                    # Instead just use the structures that already exist if they exist.
                    if curLoc in glyphDict:
                        curGlyph = glyphDict[curLoc]
                    else:
                        curGlyphData = data[slice(*curLoc)]
                        imageFormatClass = self.getImageFormatClass(
                            indexSubTable.imageFormat
                        )
                        curGlyph = imageFormatClass(curGlyphData, ttFont)
                        glyphDict[curLoc] = curGlyph
                    bitmapGlyphDict[curName] = curGlyph

    def compile(self, ttFont):
        dataList = []
        dataList.append(sstruct.pack(ebdtTableVersionFormat, self))
        dataSize = len(dataList[0])

        # Keep a dict of glyphs that have been seen so they aren't remade.
        # This dict maps the id of the BitmapGlyph to the interval
        # in the data.
        glyphDict = {}

        # Go through the bitmap glyph data. Just in case the data for a glyph
        # changed the size metrics should be recalculated. There are a variety
        # of formats and they get stored in the EBLC table. That is why
        # recalculation is defered to the EblcIndexSubTable class and just
        # pass what is known about bitmap glyphs from this particular table.
        locator = ttFont[self.__class__.locatorName]
        for curStrike, curGlyphDict in zip(locator.strikes, self.strikeData):
            for curIndexSubTable in curStrike.indexSubTables:
                dataLocations = []
                for curName in curIndexSubTable.names:
                    # Handle the data placement based on seeing the glyph or not.
                    # Just save a reference to the location if the glyph has already
                    # been saved in compile. This code assumes that glyphs will only
                    # be referenced multiple times from indexFormat5. By luck the
                    # code may still work when referencing poorly ordered fonts with
                    # duplicate references. If there is a font that is unlucky the
                    # respective compile methods for the indexSubTables will fail
                    # their assertions. All fonts seem to follow this assumption.
                    # More complicated packing may be needed if a counter-font exists.
                    glyph = curGlyphDict[curName]
                    objectId = id(glyph)
                    if objectId not in glyphDict:
                        data = glyph.compile(ttFont)
                        data = curIndexSubTable.padBitmapData(data)
                        startByte = dataSize
                        dataSize += len(data)
                        endByte = dataSize
                        dataList.append(data)
                        dataLoc = (startByte, endByte)
                        glyphDict[objectId] = dataLoc
                    else:
                        dataLoc = glyphDict[objectId]
                    dataLocations.append(dataLoc)
                # Just use the new data locations in the indexSubTable.
                # The respective compile implementations will take care
                # of any of the problems in the convertion that may arise.
                curIndexSubTable.locations = dataLocations

        return bytesjoin(dataList)

    def toXML(self, writer, ttFont):
        # When exporting to XML if one of the data export formats
        # requires metrics then those metrics may be in the locator.
        # In this case populate the bitmaps with "export metrics".
        if ttFont.bitmapGlyphDataFormat in ("row", "bitwise"):
            locator = ttFont[self.__class__.locatorName]
            for curStrike, curGlyphDict in zip(locator.strikes, self.strikeData):
                for curIndexSubTable in curStrike.indexSubTables:
                    for curName in curIndexSubTable.names:
                        glyph = curGlyphDict[curName]
                        # I'm not sure which metrics have priority here.
                        # For now if both metrics exist go with glyph metrics.
                        if hasattr(glyph, "metrics"):
                            glyph.exportMetrics = glyph.metrics
                        else:
                            glyph.exportMetrics = curIndexSubTable.metrics
                        glyph.exportBitDepth = curStrike.bitmapSizeTable.bitDepth

        writer.simpletag("header", [("version", self.version)])
        writer.newline()
        locator = ttFont[self.__class__.locatorName]
        for strikeIndex, bitmapGlyphDict in enumerate(self.strikeData):
            writer.begintag("strikedata", [("index", strikeIndex)])
            writer.newline()
            for curName, curBitmap in bitmapGlyphDict.items():
                curBitmap.toXML(strikeIndex, curName, writer, ttFont)
            writer.endtag("strikedata")
            writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        if name == "header":
