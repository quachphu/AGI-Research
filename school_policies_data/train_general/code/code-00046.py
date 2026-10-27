    the replacement ``glyphs``. ``prefix``, ``suffix``, ``glyphs`` and
    ``replace`` should be lists of `glyph-containing objects`_. ``glyphs`` and
    ``replace`` should be one-item lists.
    """

    def __init__(self, glyphs, replace, prefix, suffix, forceChain, location=None):
        Statement.__init__(self, location)
        self.prefix, self.suffix = prefix, suffix
        self.forceChain = forceChain
        self.glyphs = glyphs
        self.replacements = replace

    def build(self, builder):
        """Calls the builder object's ``add_single_subst`` callback."""
        prefix = [p.glyphSet() for p in self.prefix]
        suffix = [s.glyphSet() for s in self.suffix]
        originals = self.glyphs[0].glyphSet()
        replaces = self.replacements[0].glyphSet()
        if len(replaces) == 1:
            replaces = replaces * len(originals)
        builder.add_single_subst(
            self.location,
            prefix,
            suffix,
            OrderedDict(zip(originals, replaces)),
            self.forceChain,
        )

    def asFea(self, indent=""):
        res = "sub "
        if len(self.prefix) or len(self.suffix) or self.forceChain:
            if len(self.prefix):
                res += " ".join(asFea(g) for g in self.prefix) + " "
            res += " ".join(asFea(g) + "'" for g in self.glyphs)
            if len(self.suffix):
                res += " " + " ".join(asFea(g) for g in self.suffix)
        else:
            res += " ".join(asFea(g) for g in self.glyphs)
        res += " by {};".format(" ".join(asFea(g) for g in self.replacements))
        return res


class ScriptStatement(Statement):
    """A ``script`` statement."""

    def __init__(self, script, location=None):
        Statement.__init__(self, location)
        self.script = script  #: the script code

    def build(self, builder):
        """Calls the builder's ``set_script`` callback."""
        builder.set_script(self.location, self.script)

    def asFea(self, indent=""):
        return "script {};".format(self.script.strip())


class SinglePosStatement(Statement):
    """A single position statement. ``prefix`` and ``suffix`` should be
    lists of `glyph-containing objects`_.

    ``pos`` should be a one-element list containing a (`glyph-containing object`_,
    :class:`ValueRecord`) tuple."""

    def __init__(self, pos, prefix, suffix, forceChain, location=None):
        Statement.__init__(self, location)
        self.pos, self.prefix, self.suffix = pos, prefix, suffix
        self.forceChain = forceChain

    def build(self, builder):
        """Calls the builder object's ``add_single_pos`` callback."""
        prefix = [p.glyphSet() for p in self.prefix]
        suffix = [s.glyphSet() for s in self.suffix]
        pos = [(g.glyphSet(), value) for g, value in self.pos]
        builder.add_single_pos(self.location, prefix, suffix, pos, self.forceChain)

    def asFea(self, indent=""):
        res = "pos "
        if len(self.prefix) or len(self.suffix) or self.forceChain:
            if len(self.prefix):
                res += " ".join(map(asFea, self.prefix)) + " "
            res += " ".join(
                [
                    asFea(x[0])
                    + "'"
                    + ((" " + x[1].asFea()) if x[1] is not None else "")
                    for x in self.pos
                ]
            )
            if len(self.suffix):
                res += " " + " ".join(map(asFea, self.suffix))
        else:
            res += " ".join(
                [
                    asFea(x[0]) + " " + (x[1].asFea() if x[1] is not None else "")
                    for x in self.pos
                ]
            )
        res += ";"
        return res


class SubtableStatement(Statement):
    """Represents a subtable break."""

    def __init__(self, location=None):
        Statement.__init__(self, location)

    def build(self, builder):
        """Calls the builder objects's ``add_subtable_break`` callback."""
        builder.add_subtable_break(self.location)

    def asFea(self, indent=""):
        return "subtable;"


class ValueRecord(Expression):
    """Represents a value record."""

    def __init__(
        self,
        xPlacement=None,
        yPlacement=None,
        xAdvance=None,
        yAdvance=None,
        xPlaDevice=None,
        yPlaDevice=None,
        xAdvDevice=None,
        yAdvDevice=None,
        vertical=False,
        location=None,
    ):
        Expression.__init__(self, location)
        self.xPlacement, self.yPlacement = (xPlacement, yPlacement)
        self.xAdvance, self.yAdvance = (xAdvance, yAdvance)
        self.xPlaDevice, self.yPlaDevice = (xPlaDevice, yPlaDevice)
        self.xAdvDevice, self.yAdvDevice = (xAdvDevice, yAdvDevice)
        self.vertical = vertical

    def __eq__(self, other):
        return (
            self.xPlacement == other.xPlacement
            and self.yPlacement == other.yPlacement
            and self.xAdvance == other.xAdvance
            and self.yAdvance == other.yAdvance
            and self.xPlaDevice == other.xPlaDevice
            and self.xAdvDevice == other.xAdvDevice
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return (
            hash(self.xPlacement)
            ^ hash(self.yPlacement)
            ^ hash(self.xAdvance)
            ^ hash(self.yAdvance)
            ^ hash(self.xPlaDevice)
            ^ hash(self.yPlaDevice)
            ^ hash(self.xAdvDevice)
            ^ hash(self.yAdvDevice)
        )

    def asFea(self, indent=""):
        if not self:
            return "<NULL>"

        x, y = self.xPlacement, self.yPlacement
        xAdvance, yAdvance = self.xAdvance, self.yAdvance
        xPlaDevice, yPlaDevice = self.xPlaDevice, self.yPlaDevice
        xAdvDevice, yAdvDevice = self.xAdvDevice, self.yAdvDevice
        vertical = self.vertical

        # Try format A, if possible.
        if x is None and y is None:
            if xAdvance is None and vertical:
                return str(yAdvance)
            elif yAdvance is None and not vertical:
                return str(xAdvance)

        # Make any remaining None value 0 to avoid generating invalid records.
        x = x or 0
        y = y or 0
        xAdvance = xAdvance or 0
        yAdvance = yAdvance or 0

        # Try format B, if possible.
        if (
            xPlaDevice is None
            and yPlaDevice is None
            and xAdvDevice is None
            and yAdvDevice is None
        ):
            return "<%s %s %s %s>" % (x, y, xAdvance, yAdvance)

        # Last resort is format C.
        return "<%s %s %s %s %s %s %s %s>" % (
            x,
            y,
            xAdvance,
            yAdvance,
            deviceToString(xPlaDevice),
            deviceToString(yPlaDevice),
            deviceToString(xAdvDevice),
            deviceToString(yAdvDevice),
        )

    def __bool__(self):
        return any(
            getattr(self, v) is not None
            for v in [
                "xPlacement",
                "yPlacement",
                "xAdvance",
                "yAdvance",
                "xPlaDevice",
                "yPlaDevice",
                "xAdvDevice",
                "yAdvDevice",
            ]
        )

    __nonzero__ = __bool__


class ValueRecordDefinition(Statement):
    """Represents a named value record definition."""

    def __init__(self, name, value, location=None):
        Statement.__init__(self, location)
        self.name = name  #: Value record name as string
        self.value = value  #: :class:`ValueRecord` object

    def asFea(self, indent=""):
        return "valueRecordDef {} {};".format(self.value.asFea(), self.name)


def simplify_name_attributes(pid, eid, lid):
    if pid == 3 and eid == 1 and lid == 1033:
        return ""
    elif pid == 1 and eid == 0 and lid == 0:
        return "1"
    else:
        return "{} {} {}".format(pid, eid, lid)


class NameRecord(Statement):
    """Represents a name record. (`Section 9.e. <https://adobe-type-tools.github.io/afdko/OpenTypeFeatureFileSpecification.html#9.e>`_)"""

    def __init__(self, nameID, platformID, platEncID, langID, string, location=None):
        Statement.__init__(self, location)
        self.nameID = nameID  #: Name ID as integer (e.g. 9 for designer's name)
        self.platformID = platformID  #: Platform ID as integer
        self.platEncID = platEncID  #: Platform encoding ID as integer
        self.langID = langID  #: Language ID as integer
        self.string = string  #: Name record value

    def build(self, builder):
        """Calls the builder object's ``add_name_record`` callback."""
        builder.add_name_record(
            self.location,
            self.nameID,
            self.platformID,
            self.platEncID,
            self.langID,
            self.string,
        )

    def asFea(self, indent=""):
        def escape(c, escape_pattern):
            # Also escape U+0022 QUOTATION MARK and U+005C REVERSE SOLIDUS
            if c >= 0x20 and c <= 0x7E and c not in (0x22, 0x5C):
                return chr(c)
            else:
                return escape_pattern % c

        encoding = getEncoding(self.platformID, self.platEncID, self.langID)
        if encoding is None:
            raise FeatureLibError("Unsupported encoding", self.location)
        s = tobytes(self.string, encoding=encoding)
        if encoding == "utf_16_be":
            escaped_string = "".join(
                [
                    escape(byteord(s[i]) * 256 + byteord(s[i + 1]), r"\%04x")
                    for i in range(0, len(s), 2)
                ]
            )
        else:
            escaped_string = "".join([escape(byteord(b), r"\%02x") for b in s])
        plat = simplify_name_attributes(self.platformID, self.platEncID, self.langID)
        if plat != "":
            plat += " "
        return 'nameid {} {}"{}";'.format(self.nameID, plat, escaped_string)


class FeatureNameStatement(NameRecord):
    """Represents a ``sizemenuname`` or ``name`` statement."""

    def build(self, builder):
        """Calls the builder object's ``add_featureName`` callback."""
        NameRecord.build(self, builder)
        builder.add_featureName(self.nameID)

    def asFea(self, indent=""):
        if self.nameID == "size":
            tag = "sizemenuname"
        else:
            tag = "name"
        plat = simplify_name_attributes(self.platformID, self.platEncID, self.langID)
        if plat != "":
            plat += " "
        return '{} {}"{}";'.format(tag, plat, self.string)


class STATNameStatement(NameRecord):
    """Represents a STAT table ``name`` statement."""

    def asFea(self, indent=""):
        plat = simplify_name_attributes(self.platformID, self.platEncID, self.langID)
        if plat != "":
            plat += " "
        return 'name {}"{}";'.format(plat, self.string)


class SizeParameters(Statement):
    """A ``parameters`` statement."""

    def __init__(self, DesignSize, SubfamilyID, RangeStart, RangeEnd, location=None):
        Statement.__init__(self, location)
        self.DesignSize = DesignSize
        self.SubfamilyID = SubfamilyID
        self.RangeStart = RangeStart
        self.RangeEnd = RangeEnd

    def build(self, builder):
        """Calls the builder object's ``set_size_parameters`` callback."""
        builder.set_size_parameters(
            self.location,
            self.DesignSize,
            self.SubfamilyID,
            self.RangeStart,
            self.RangeEnd,
        )

    def asFea(self, indent=""):
        res = "parameters {:.1f} {}".format(self.DesignSize, self.SubfamilyID)
        if self.RangeStart != 0 or self.RangeEnd != 0:
            res += " {} {}".format(int(self.RangeStart * 10), int(self.RangeEnd * 10))
        return res + ";"


class CVParametersNameStatement(NameRecord):
    """Represent a name statement inside a ``cvParameters`` block."""

    def __init__(
        self, nameID, platformID, platEncID, langID, string, block_name, location=None
    ):
        NameRecord.__init__(
            self, nameID, platformID, platEncID, langID, string, location=location
        )
        self.block_name = block_name

    def build(self, builder):
        """Calls the builder object's ``add_cv_parameter`` callback."""
        item = ""
        if self.block_name == "ParamUILabelNameID":
            item = "_{}".format(builder.cv_num_named_params_.get(self.nameID, 0))
        builder.add_cv_parameter(self.nameID)
        self.nameID = (self.nameID, self.block_name + item)
        NameRecord.build(self, builder)

    def asFea(self, indent=""):
        plat = simplify_name_attributes(self.platformID, self.platEncID, self.langID)
        if plat != "":
            plat += " "
        return 'name {}"{}";'.format(plat, self.string)


class CharacterStatement(Statement):
    """
    Statement used in cvParameters blocks of Character Variant features (cvXX).
    The Unicode value may be written with either decimal or hexadecimal
    notation. The value must be preceded by '0x' if it is a hexadecimal value.
    The largest Unicode value allowed is 0xFFFFFF.
    """

    def __init__(self, character, tag, location=None):
        Statement.__init__(self, location)
        self.character = character
        self.tag = tag

    def build(self, builder):
        """Calls the builder object's ``add_cv_character`` callback."""
        builder.add_cv_character(self.character, self.tag)

    def asFea(self, indent=""):
        return "Character {:#x};".format(self.character)


class BaseAxis(Statement):
    """An axis definition, being either a ``VertAxis.BaseTagList/BaseScriptList``
    pair or a ``HorizAxis.BaseTagList/BaseScriptList`` pair."""

    def __init__(self, bases, scripts, vertical, minmax=None, location=None):
        Statement.__init__(self, location)
        self.bases = bases  #: A list of baseline tag names as strings
        self.scripts = scripts  #: A list of script record tuplets (script tag, default baseline tag, base coordinate)
        self.vertical = vertical  #: Boolean; VertAxis if True, HorizAxis if False
        self.minmax = []  #: A set of minmax record

    def build(self, builder):
        """Calls the builder object's ``set_base_axis`` callback."""
        builder.set_base_axis(self.bases, self.scripts, self.vertical, self.minmax)

    def asFea(self, indent=""):
        direction = "Vert" if self.vertical else "Horiz"
        scripts = [
            "{} {} {}".format(a[0], a[1], " ".join(map(str, a[2])))
            for a in self.scripts
        ]
        minmaxes = [
            "\n{}Axis.MinMax {} {} {}, {};".format(direction, a[0], a[1], a[2], a[3])
            for a in self.minmax
        ]
        return "{}Axis.BaseTagList {};\n{}{}Axis.BaseScriptList {};".format(
            direction, " ".join(self.bases), indent, direction, ", ".join(scripts)
        ) + "\n".join(minmaxes)


class OS2Field(Statement):
    """An entry in the ``OS/2`` table. Most ``values`` should be numbers or
    strings, apart from when the key is ``UnicodeRange``, ``CodePageRange``
    or ``Panose``, in which case it should be an array of integers."""

    def __init__(self, key, value, location=None):
        Statement.__init__(self, location)
        self.key = key
        self.value = value

    def build(self, builder):
        """Calls the builder object's ``add_os2_field`` callback."""
        builder.add_os2_field(self.key, self.value)

    def asFea(self, indent=""):
        def intarr2str(x):
            return " ".join(map(str, x))

        numbers = (
            "FSType",
            "TypoAscender",
            "TypoDescender",
            "TypoLineGap",
            "winAscent",
            "winDescent",
            "XHeight",
            "CapHeight",
            "WeightClass",
            "WidthClass",
            "LowerOpSize",
            "UpperOpSize",
        )
        ranges = ("UnicodeRange", "CodePageRange")
        keywords = dict([(x.lower(), [x, str]) for x in numbers])
        keywords.update([(x.lower(), [x, intarr2str]) for x in ranges])
        keywords["panose"] = ["Panose", intarr2str]
        keywords["vendor"] = ["Vendor", lambda y: '"{}"'.format(y)]
        if self.key in keywords:
            return "{} {};".format(
                keywords[self.key][0], keywords[self.key][1](self.value)
            )
        return ""  # should raise exception


class HheaField(Statement):
    """An entry in the ``hhea`` table."""

    def __init__(self, key, value, location=None):
        Statement.__init__(self, location)
        self.key = key
        self.value = value

    def build(self, builder):
        """Calls the builder object's ``add_hhea_field`` callback."""
        builder.add_hhea_field(self.key, self.value)

    def asFea(self, indent=""):
        fields = ("CaretOffset", "Ascender", "Descender", "LineGap")
        keywords = dict([(x.lower(), x) for x in fields])
        return "{} {};".format(keywords[self.key], self.value)


class VheaField(Statement):
    """An entry in the ``vhea`` table."""

    def __init__(self, key, value, location=None):
        Statement.__init__(self, location)
        self.key = key
        self.value = value

    def build(self, builder):
        """Calls the builder object's ``add_vhea_field`` callback."""
        builder.add_vhea_field(self.key, self.value)

    def asFea(self, indent=""):
        fields = ("VertTypoAscender", "VertTypoDescender", "VertTypoLineGap")
        keywords = dict([(x.lower(), x) for x in fields])
        return "{} {};".format(keywords[self.key], self.value)


class STATDesignAxisStatement(Statement):
    """A STAT table Design Axis

    Args:
        tag (str): a 4 letter axis tag
        axisOrder (int): an int
        names (list): a list of :class:`STATNameStatement` objects
    """

    def __init__(self, tag, axisOrder, names, location=None):
        Statement.__init__(self, location)
        self.tag = tag
        self.axisOrder = axisOrder
        self.names = names
        self.location = location

    def build(self, builder):
        builder.addDesignAxis(self, self.location)

    def asFea(self, indent=""):
        indent += SHIFT
        res = f"DesignAxis {self.tag} {self.axisOrder} {{ \n"
        res += ("\n" + indent).join([s.asFea(indent=indent) for s in self.names]) + "\n"
        res += "};"
        return res


class ElidedFallbackName(Statement):
    """STAT table ElidedFallbackName

    Args:
        names: a list of :class:`STATNameStatement` objects
    """

    def __init__(self, names, location=None):
        Statement.__init__(self, location)
        self.names = names
        self.location = location

    def build(self, builder):
        builder.setElidedFallbackName(self.names, self.location)

    def asFea(self, indent=""):
        indent += SHIFT
        res = "ElidedFallbackName { \n"
        res += ("\n" + indent).join([s.asFea(indent=indent) for s in self.names]) + "\n"
        res += "};"
        return res


class ElidedFallbackNameID(Statement):
    """STAT table ElidedFallbackNameID

    Args:
        value: an int pointing to an existing name table name ID
    """

    def __init__(self, value, location=None):
        Statement.__init__(self, location)
        self.value = value
        self.location = location

    def build(self, builder):
        builder.setElidedFallbackName(self.value, self.location)

    def asFea(self, indent=""):
        return f"ElidedFallbackNameID {self.value};"


class STATAxisValueStatement(Statement):
    """A STAT table Axis Value Record

    Args:
        names (list): a list of :class:`STATNameStatement` objects
        locations (list): a list of :class:`AxisValueLocationStatement` objects
        flags (int): an int
    """

    def __init__(self, names, locations, flags, location=None):
        Statement.__init__(self, location)
        self.names = names
        self.locations = locations
        self.flags = flags

    def build(self, builder):
        builder.addAxisValueRecord(self, self.location)

    def asFea(self, indent=""):
        res = "AxisValue {\n"
        for location in self.locations:
            res += location.asFea()

        for nameRecord in self.names:
            res += nameRecord.asFea()
            res += "\n"

        if self.flags:
            flags = ["OlderSiblingFontAttribute", "ElidableAxisValueName"]
            flagStrings = []
            curr = 1
            for i in range(len(flags)):
                if self.flags & curr != 0:
                    flagStrings.append(flags[i])
                curr = curr << 1
            res += f"flag {' '.join(flagStrings)};\n"
        res += "};"
        return res


class AxisValueLocationStatement(Statement):
    """
    A STAT table Axis Value Location

    Args:
        tag (str): a 4 letter axis tag
        values (list): a list of ints and/or floats
    """

    def __init__(self, tag, values, location=None):
        Statement.__init__(self, location)
        self.tag = tag
        self.values = values

    def asFea(self, res=""):
        res += f"location {self.tag} "
        res += f"{' '.join(str(i) for i in self.values)};\n"
        return res


class ConditionsetStatement(Statement):
    """
    A variable layout conditionset

    Args:
        name (str): the name of this conditionset
        conditions (dict): a dictionary mapping axis tags to a
            tuple of (min,max) userspace coordinates.
    """

    def __init__(self, name, conditions, location=None):
        Statement.__init__(self, location)
        self.name = name
        self.conditions = conditions

    def build(self, builder):
        builder.add_conditionset(self.location, self.name, self.conditions)

    def asFea(self, res="", indent=""):
        res += indent + f"conditionset {self.name} " + "{\n"
        for tag, (minvalue, maxvalue) in self.conditions.items():
            res += indent + SHIFT + f"{tag} {minvalue} {maxvalue};\n"
        res += indent + "}" + f" {self.name};\n"
        return res


class VariationBlock(Block):
    """A variation feature block, applicable in a given set of conditions."""

    def __init__(self, name, conditionset, use_extension=False, location=None):
        Block.__init__(self, location)
        self.name, self.conditionset, self.use_extension = (
            name,
            conditionset,
            use_extension,
        )

    def build(self, builder):
        """Call the ``start_feature`` callback on the builder object, visit
        all the statements in this feature, and then call ``end_feature``."""
        builder.start_feature(self.location, self.name, self.use_extension)
        if (
            self.conditionset != "NULL"
            and self.conditionset not in builder.conditionsets_
        ):
            raise FeatureLibError(
                f"variation block used undefined conditionset {self.conditionset}",
                self.location,
            )

        # language exclude_dflt statements modify builder.features_
        # limit them to this block with temporary builder.features_
        features = builder.features_
        builder.features_ = {}
        Block.build(self, builder)
        for key, value in builder.features_.items():
            items = builder.feature_variations_.setdefault(key, {}).setdefault(
                self.conditionset, []
            )
            items.extend(value)
            if key not in features:
                features[key] = []  # Ensure we make a feature record
        builder.features_ = features
        builder.end_feature()

    def asFea(self, indent=""):
        res = indent + "variation %s " % self.name.strip()
        res += self.conditionset + " "
        if self.use_extension:
            res += "useExtension "
        res += "{\n"
        res += Block.asFea(self, indent=indent)
        res += indent + "} %s;\n" % self.name.strip()
        return res
<|endoftext|>from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from fontTools.designspaceLib import DesignSpaceDocument
from fontTools.ttLib.ttFont import TTFont
from fontTools.varLib.models import (
    VariationModel,
    noRound,
    normalizeValue,
    piecewiseLinearMap,
)

import typing
import warnings

if typing.TYPE_CHECKING:
    from typing import Self

LocationTuple = tuple[tuple[str, float], ...]
"""A hashable location."""


def Location(location: Mapping[str, float]) -> LocationTuple:
    """Create a hashable location from a dictionary-like location."""
    return tuple(sorted(location.items()))


class VariableScalar:
    """A scalar with different values at different points in the designspace."""

    values: dict[LocationTuple, int]
    """The values across various user-locations. Must always include the default
    location by time of building."""

    def __init__(self, location_value=None):
        self.values = {
            Location(location): value
            for location, value in (location_value or {}).items()
        }
        # Deprecated: only used by the add_to_variation_store() backwards-compat
        # shim. New code should use VariableScalarBuilder instead.
        self.axes = []

    def __repr__(self):
        items = []
        for location, value in self.values.items():
            if location:
                loc = ",".join(
                    [
                        f"{ax}={int(coord) if float(coord).is_integer() else coord}"
                        for ax, coord in location
                    ]
                )
                items.append("%s:%i" % (loc, value))
            else:
                items.append("%i" % value)
        return "(" + (" ".join(items)) + ")"

    @property
    def does_vary(self) -> bool:
        values = list(self.values.values())
        return any(v != values[0] for v in values[1:])

    def add_value(self, location: Mapping[str, float], value: int):
        self.values[Location(location)] = value

    def add_to_variation_store(self, store_builder, model_cache=None, avar=None):
        """Deprecated: use VariableScalarBuilder.add_to_variation_store() instead."""
        warnings.warn(
            "VariableScalar.add_to_variation_store() is deprecated. "
            "Use VariableScalarBuilder.add_to_variation_store() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not self.axes:
            raise ValueError(
                ".axes must be defined on variable scalar before calling "
                "add_to_variation_store()"
            )
        builder = VariableScalarBuilder(
            axis_triples={
                ax.axisTag: (ax.minValue, ax.defaultValue, ax.maxValue)
                for ax in self.axes
            },
            axis_mappings=({} if avar is None else dict(avar.segments)),
            model_cache=model_cache if model_cache is not None else {},
        )
        return builder.add_to_variation_store(self, store_builder)


@dataclass
class VariableScalarBuilder:
    """A helper class for building variable scalars, or otherwise interrogating
    their variation model for interpolation or similar."""

    axis_triples: dict[str, tuple[float, float, float]]
    """Minimum, default, and maximum for each axis in user-coordinates."""
    axis_mappings: dict[str, Mapping[float, float]]
    """Optional mappings from normalized user-coordinates to normalized
    design-coordinates."""

    model_cache: dict[tuple[LocationTuple, ...], VariationModel]
    """We often use the same exact locations (i.e. font sources) for a large
    number of variable scalars. Instead of creating a model for each, cache
    them. Cache by user-location to avoid repeated mapping computations."""

    @classmethod
    def from_ttf(cls, ttf: TTFont) -> Self:
        return cls(
            axis_triples={
                axis.axisTag: (axis.minValue, axis.defaultValue, axis.maxValue)
                for axis in ttf["fvar"].axes
            },
            axis_mappings=(
                {}
                if (avar := ttf.get("avar")) is None
                else {axis: segments for axis, segments in avar.segments.items()}
            ),
            model_cache={},
        )

    @classmethod
    def from_designspace(cls, doc: DesignSpaceDocument) -> Self:
        return cls(
            axis_triples={
                axis.tag: (axis.minimum, axis.default, axis.maximum)
                for axis in doc.axes
            },
            axis_mappings={
                axis.tag: {
                    normalizeValue(
                        user, (axis.minimum, axis.default, axis.maximum)
                    ): normalizeValue(
                        design,
                        (
                            axis.map_forward(axis.minimum),
                            axis.map_forward(axis.default),
                            axis.map_forward(axis.maximum),
                        ),
                    )
                    for user, design in axis.map
                }
                for axis in doc.axes
                if axis.map
            },
            model_cache={},
        )

    def _fully_specify_location(self, location: LocationTuple) -> LocationTuple:
        """Validate and fully-specify a user-space location by filling in
        missing axes with their user-space defaults."""

        full = {}
        for axtag, value in location:
            if axtag not in self.axis_triples:
                raise ValueError("Unknown axis %s in %s" % (axtag, location))
            full[axtag] = value

        for axtag, (_, axis_default, _) in self.axis_triples.items():
            if axtag not in full:
                full[axtag] = axis_default

        return Location(full)

    def _normalize_location(self, location: LocationTuple) -> dict[str, float]:
        """Normalize a user-space location, applying avar mappings if present.

        TODO: This only handles avar1 (per-axis piecewise linear mappings),
        not avar2 (multi-dimensional mappings).
        """

        result = {}
        for axtag, value in location:
            axis_min, axis_default, axis_max = self.axis_triples[axtag]
            normalized = normalizeValue(value, (axis_min, axis_default, axis_max))
            mapping = self.axis_mappings.get(axtag)
            if mapping is not None:
                normalized = piecewiseLinearMap(normalized, mapping)
            result[axtag] = normalized

        return result

    def _full_locations_and_values(
        self, scalar: VariableScalar
    ) -> list[tuple[LocationTuple, int]]:
        """Return a list of (fully-specified user-space location, value) pairs,
        preserving order and length of scalar.values."""

        return [
            (self._fully_specify_location(loc), val)
            for loc, val in scalar.values.items()
        ]

    def default_value(self, scalar: VariableScalar) -> int:
        """Get the default value of a variable scalar."""

        default_loc = Location(
            {tag: default for tag, (_, default, _) in self.axis_triples.items()}
        )
        for location, value in self._full_locations_and_values(scalar):
            if location == default_loc:
                return value

        raise ValueError("Default value could not be found")

    def value_at_location(
        self, scalar: VariableScalar, location: LocationTuple
    ) -> float:
        """Interpolate the value of a scalar from a user-location."""

        location = self._fully_specify_location(location)
        pairs = self._full_locations_and_values(scalar)

        # If user location matches exactly, no axis mapping or variation model needed.
        for loc, val in pairs:
            if loc == location:
                return val

        values = [val for _, val in pairs]
        normalized_location = self._normalize_location(location)

        value = self.model(scalar).interpolateFromMasters(normalized_location, values)
        if value is None:
            raise ValueError("Insufficient number of values to interpolate")

        return value

    def model(self, scalar: VariableScalar) -> VariationModel:
        """Return a variation model based on a scalar's values.

        Variable scalars with the same fully-specified user-locations will use
        the same cached variation model."""

        pairs = self._full_locations_and_values(scalar)
        cache_key = tuple(loc for loc, _ in pairs)

        cached_model = self.model_cache.get(cache_key)
        if cached_model is not None:
            return cached_model

        normalized_locations = [self._normalize_location(loc) for loc, _ in pairs]
        axisOrder = list(self.axis_triples.keys())
        model = self.model_cache[cache_key] = VariationModel(
            normalized_locations, axisOrder=axisOrder
        )

        return model

    def get_deltas_and_supports(self, scalar: VariableScalar):
        """Calculate deltas and supports from this scalar's variation model."""
        values = list(scalar.values.values())
        return self.model(scalar).getDeltasAndSupports(values, round=round)

    def add_to_variation_store(
        self, scalar: VariableScalar, store_builder
    ) -> tuple[int, int]:
        """Serialize this scalar's variation model to a store, returning the
        default value and variation index."""

        deltas, supports = self.get_deltas_and_supports(scalar)
        store_builder.setSupports(supports)
        index = store_builder.storeDeltas(deltas, round=noRound)

        # NOTE: Default value should be an exact integer by construction of
        #       VariableScalar.
        return int(self.default_value(scalar)), index
<|endoftext|>from fontTools.feaLib.error import FeatureLibError, IncludedFeaNotFound
from fontTools.feaLib.location import FeatureLibLocation
import re
import os

try:
    import cython
except ImportError:
    # if cython not installed, use mock module with no-op decorators and types
    from fontTools.misc import cython


class Lexer(object):
    NUMBER = "NUMBER"
    HEXADECIMAL = "HEXADECIMAL"
    OCTAL = "OCTAL"
    NUMBERS = (NUMBER, HEXADECIMAL, OCTAL)
    FLOAT = "FLOAT"
    STRING = "STRING"
    NAME = "NAME"
    FILENAME = "FILENAME"
    GLYPHCLASS = "GLYPHCLASS"
    CID = "CID"
    SYMBOL = "SYMBOL"
    COMMENT = "COMMENT"
    NEWLINE = "NEWLINE"
    ANONYMOUS_BLOCK = "ANONYMOUS_BLOCK"

    CHAR_WHITESPACE_ = " \t"
    CHAR_NEWLINE_ = "\r\n"
    CHAR_SYMBOL_ = ",;:-+'{}[]<>()="
    CHAR_DIGIT_ = "0123456789"
    CHAR_HEXDIGIT_ = "0123456789ABCDEFabcdef"
    CHAR_LETTER_ = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    CHAR_NAME_START_ = CHAR_LETTER_ + "_+*:.^~!\\"
    CHAR_NAME_CONTINUATION_ = CHAR_LETTER_ + CHAR_DIGIT_ + "_.+*:^~!/-"

    RE_GLYPHCLASS = re.compile(r"^[A-Za-z_0-9.\-]+$")

    MODE_NORMAL_ = "NORMAL"
    MODE_FILENAME_ = "FILENAME"

    def __init__(self, text, filename):
        self.filename_ = filename
        self.line_ = 1
        self.pos_ = 0
        self.line_start_ = 0
        self.text_ = text
        self.text_length_ = len(text)
        self.mode_ = Lexer.MODE_NORMAL_

    def __iter__(self):
        return self

    def next(self):  # Python 2
        return self.__next__()

    def __next__(self):  # Python 3
        while True:
            token_type, token, location = self.next_()
            if token_type != Lexer.NEWLINE:
                return (token_type, token, location)

    def location_(self):
        column = self.pos_ - self.line_start_ + 1
        return FeatureLibLocation(self.filename_ or "<features>", self.line_, column)

    def next_(self):
        self.scan_over_(Lexer.CHAR_WHITESPACE_)
        location = self.location_()
        start = self.pos_
        text = self.text_
        limit = len(text)
        if start >= limit:
            raise StopIteration()
        cur_char = text[start]
        next_char = text[start + 1] if start + 1 < limit else None

        if cur_char == "\n":
            self.pos_ += 1
            self.line_ += 1
            self.line_start_ = self.pos_
            return (Lexer.NEWLINE, None, location)
        if cur_char == "\r":
            self.pos_ += 2 if next_char == "\n" else 1
            self.line_ += 1
            self.line_start_ = self.pos_
            return (Lexer.NEWLINE, None, location)
        if cur_char == "#":
            self.scan_until_(Lexer.CHAR_NEWLINE_)
            return (Lexer.COMMENT, text[start : self.pos_], location)

        if self.mode_ is Lexer.MODE_FILENAME_:
            if cur_char != "(":
                raise FeatureLibError("Expected '(' before file name", location)
            self.scan_until_(")")
            cur_char = text[self.pos_] if self.pos_ < limit else None
            if cur_char != ")":
                raise FeatureLibError("Expected ')' after file name", location)
            self.pos_ += 1
            self.mode_ = Lexer.MODE_NORMAL_
            return (Lexer.FILENAME, text[start + 1 : self.pos_ - 1], location)

        if cur_char == "\\" and next_char in Lexer.CHAR_DIGIT_:
            self.pos_ += 1
            self.scan_over_(Lexer.CHAR_DIGIT_)
            return (Lexer.CID, int(text[start + 1 : self.pos_], 10), location)
        if cur_char == "@":
            self.pos_ += 1
            self.scan_over_(Lexer.CHAR_NAME_CONTINUATION_)
            glyphclass = text[start + 1 : self.pos_]
            if len(glyphclass) < 1:
                raise FeatureLibError("Expected glyph class name", location)
            if not Lexer.RE_GLYPHCLASS.match(glyphclass):
                raise FeatureLibError(
                    "Glyph class names must consist of letters, digits, "
                    "underscore, period or hyphen",
                    location,
                )
            return (Lexer.GLYPHCLASS, glyphclass, location)
        if cur_char in Lexer.CHAR_NAME_START_:
            self.pos_ += 1
            self.scan_over_(Lexer.CHAR_NAME_CONTINUATION_)
            token = text[start : self.pos_]
            if token == "include":
                self.mode_ = Lexer.MODE_FILENAME_
            return (Lexer.NAME, token, location)
        if cur_char == "0" and next_char in "xX":
            self.pos_ += 2
            self.scan_over_(Lexer.CHAR_HEXDIGIT_)
            return (Lexer.HEXADECIMAL, int(text[start : self.pos_], 16), location)
        if cur_char == "0" and next_char in Lexer.CHAR_DIGIT_:
            self.scan_over_(Lexer.CHAR_DIGIT_)
            return (Lexer.OCTAL, int(text[start : self.pos_], 8), location)
        if cur_char in Lexer.CHAR_DIGIT_:
            self.scan_over_(Lexer.CHAR_DIGIT_)
            if self.pos_ >= limit or text[self.pos_] != ".":
                return (Lexer.NUMBER, int(text[start : self.pos_], 10), location)
            self.scan_over_(".")
            self.scan_over_(Lexer.CHAR_DIGIT_)
            return (Lexer.FLOAT, float(text[start : self.pos_]), location)
        if cur_char == "-" and next_char in Lexer.CHAR_DIGIT_:
            self.pos_ += 1
            self.scan_over_(Lexer.CHAR_DIGIT_)
            if self.pos_ >= limit or text[self.pos_] != ".":
                return (Lexer.NUMBER, int(text[start : self.pos_], 10), location)
            self.scan_over_(".")
            self.scan_over_(Lexer.CHAR_DIGIT_)
            return (Lexer.FLOAT, float(text[start : self.pos_]), location)
        if cur_char in Lexer.CHAR_SYMBOL_:
            self.pos_ += 1
            return (Lexer.SYMBOL, cur_char, location)
        if cur_char == '"':
            self.pos_ += 1
            self.scan_until_('"')
            if self.pos_ < self.text_length_ and self.text_[self.pos_] == '"':
                self.pos_ += 1
                # strip newlines embedded within a string
                string = re.sub("[\r\n]", "", text[start + 1 : self.pos_ - 1])
                return (Lexer.STRING, string, location)
            else:
                raise FeatureLibError("Expected '\"' to terminate string", location)
        raise FeatureLibError("Unexpected character: %r" % cur_char, location)

    def scan_over_(self, valid):
        p = self.pos_
        while p < self.text_length_ and self.text_[p] in valid:
            p += 1
        self.pos_ = p

    def scan_until_(self, stop_at):
        p = self.pos_
        while p < self.text_length_ and self.text_[p] not in stop_at:
            p += 1
        self.pos_ = p

    def scan_anonymous_block(self, tag):
        location = self.location_()
        tag = tag.strip()
        self.scan_until_(Lexer.CHAR_NEWLINE_)
        self.scan_over_(Lexer.CHAR_NEWLINE_)
        regexp = r"}\s*" + re.escape(tag) + r"\s*;"
        split = re.split(regexp, self.text_[self.pos_ :], maxsplit=1)
        if len(split) != 2:
            raise FeatureLibError(
                "Expected '} %s;' to terminate anonymous block" % tag, location
            )
        self.pos_ += len(split[0])
        return (Lexer.ANONYMOUS_BLOCK, split[0], location)


class IncludingLexer(object):
    """A Lexer that follows include statements.

    The OpenType feature file specification states that due to
    historical reasons, relative imports should be resolved in this
    order:

    1. If the source font is UFO format, then relative to the UFO's
       font directory
    2. relative to the top-level include file
    3. relative to the parent include file

    We only support 1 (via includeDir) and 2.
    """

    def __init__(self, featurefile, *, includeDir=None):
        """Initializes an IncludingLexer.

        Behavior:
            If includeDir is passed, it will be used to determine the top-level
            include directory to use for all encountered include statements. If it is
            not passed, ``os.path.dirname(featurefile)`` will be considered the
            include directory.
        """

        self.lexers_ = [self.make_lexer_(featurefile)]
        self.featurefilepath = self.lexers_[0].filename_
        self.includeDir = includeDir

    def __iter__(self):
        return self

    def next(self):  # Python 2
        return self.__next__()

    def __next__(self):  # Python 3
        while self.lexers_:
            lexer = self.lexers_[-1]
            try:
                token_type, token, location = next(lexer)
            except StopIteration:
                self.lexers_.pop()
                continue
            if token_type is Lexer.NAME and token == "include":
                fname_type, fname_token, fname_location = lexer.next()
                if fname_type is not Lexer.FILENAME:
                    raise FeatureLibError("Expected file name", fname_location)
                # semi_type, semi_token, semi_location = lexer.next()
                # if semi_type is not Lexer.SYMBOL or semi_token != ";":
                #    raise FeatureLibError("Expected ';'", semi_location)
                if os.path.isabs(fname_token):
                    path = fname_token
                else:
                    if self.includeDir is not None:
                        curpath = self.includeDir
                    elif self.featurefilepath is not None:
                        curpath = os.path.dirname(self.featurefilepath)
                    else:
                        # if the IncludingLexer was initialized from an in-memory
                        # file-like stream, it doesn't have a 'name' pointing to
                        # its filesystem path, therefore we fall back to using the
                        # current working directory to resolve relative includes
                        curpath = os.getcwd()
                    path = os.path.join(curpath, fname_token)
                if len(self.lexers_) >= 5:
                    raise FeatureLibError("Too many recursive includes", fname_location)
                try:
                    self.lexers_.append(self.make_lexer_(path))
                except FileNotFoundError as err:
                    raise IncludedFeaNotFound(fname_token, fname_location) from err
            else:
                return (token_type, token, location)
        raise StopIteration()

    @staticmethod
    def make_lexer_(file_or_path):
        if hasattr(file_or_path, "read"):
            fileobj, closing = file_or_path, False
        else:
            filename, closing = file_or_path, True
            fileobj = open(filename, "r", encoding="utf-8-sig")
        data = fileobj.read()
        filename = getattr(fileobj, "name", None)
        if closing:
            fileobj.close()
        return Lexer(data, filename)

    def scan_anonymous_block(self, tag):
        return self.lexers_[-1].scan_anonymous_block(tag)


class NonIncludingLexer(IncludingLexer):
    """Lexer that does not follow `include` statements, emits them as-is."""

    def __next__(self):  # Python 3
        return next(self.lexers_[0])
<|endoftext|>class FeatureLibError(Exception):
    def __init__(self, message, location=None):
        Exception.__init__(self, message)
        self.location = location

    def __str__(self):
        message = Exception.__str__(self)
        if self.location:
            return f"{self.location}: {message}"
        else:
            return message


class IncludedFeaNotFound(FeatureLibError):
    def __str__(self):
        assert self.location is not None

        message = (
            "The following feature file should be included but cannot be found: "
            f"{Exception.__str__(self)}"
        )
        return f"{self.location}: {message}"
<|endoftext|>from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re

log = logging.getLogger(__name__)


class Parser(object):
    """Initializes a Parser object.

    Example:

        .. code:: python

            from fontTools.feaLib.parser import Parser
            parser = Parser(file, font.getReverseGlyphMap())
            parsetree = parser.parse()

    Note: the ``glyphNames`` iterable serves a double role to help distinguish
    glyph names from ranges in the presence of hyphens and to ensure that glyph
    names referenced in a feature file are actually part of a font's glyph set.
    If the iterable is left empty, no glyph name in glyph set checking takes
    place, and all glyph tokens containing hyphens are treated as literal glyph
    names, not as ranges. (Adding a space around the hyphen can, in any case,
    help to disambiguate ranges from glyph names containing hyphens.)

    By default, the parser will follow ``include()`` statements in the feature
    file. To turn this off, pass ``followIncludes=False``. Pass a directory string as
    ``includeDir`` to explicitly declare a directory to search included feature files
    in.
    """

    extensions = {}
    ast = ast
    SS_FEATURE_TAGS = {"ss%02d" % i for i in range(1, 20 + 1)}
    CV_FEATURE_TAGS = {"cv%02d" % i for i in range(1, 99 + 1)}

    def __init__(
        self, featurefile, glyphNames=(), followIncludes=True, includeDir=None, **kwargs
    ):
        if "glyphMap" in kwargs:
            from fontTools.misc.loggingTools import deprecateArgument

            deprecateArgument("glyphMap", "use 'glyphNames' (iterable) instead")
            if glyphNames:
                raise TypeError(
                    "'glyphNames' and (deprecated) 'glyphMap' are " "mutually exclusive"
                )
            glyphNames = kwargs.pop("glyphMap")
        if kwargs:
            raise TypeError(
                "unsupported keyword argument%s: %s"
                % ("" if len(kwargs) == 1 else "s", ", ".join(repr(k) for k in kwargs))
            )

        self.glyphNames_ = set(glyphNames)
        self.doc_ = self.ast.FeatureFile()
        self.anchors_ = SymbolTable()
        self.glyphclasses_ = SymbolTable()
        self.lookups_ = SymbolTable()
        self.valuerecords_ = SymbolTable()
        self.symbol_tables_ = {self.anchors_, self.valuerecords_}
        self.next_token_type_, self.next_token_ = (None, None)
        self.cur_comments_ = []
        self.next_token_location_ = None
        lexerClass = IncludingLexer if followIncludes else NonIncludingLexer
        self.lexer_ = lexerClass(featurefile, includeDir=includeDir)
        self.missing = {}
        self.advance_lexer_(comments=True)

    def parse(self):
        """Parse the file, and return a :class:`fontTools.feaLib.ast.FeatureFile`
        object representing the root of the abstract syntax tree containing the
        parsed contents of the file."""
        statements = self.doc_.statements
        while self.next_token_type_ is not None or self.cur_comments_:
            self.advance_lexer_(comments=True)
            if self.cur_token_type_ is Lexer.COMMENT:
                statements.append(
                    self.ast.Comment(self.cur_token_, location=self.cur_token_location_)
                )
            elif self.is_cur_keyword_("include"):
                statements.append(self.parse_include_())
            elif self.cur_token_type_ is Lexer.GLYPHCLASS:
                statements.append(self.parse_glyphclass_definition_())
            elif self.is_cur_keyword_(("anon", "anonymous")):
                statements.append(self.parse_anonymous_())
            elif self.is_cur_keyword_("anchorDef"):
                statements.append(self.parse_anchordef_())
            elif self.is_cur_keyword_("languagesystem"):
                statements.append(self.parse_languagesystem_())
            elif self.is_cur_keyword_("lookup"):
                statements.append(self.parse_lookup_(vertical=False))
            elif self.is_cur_keyword_("markClass"):
                statements.append(self.parse_markClass_())
            elif self.is_cur_keyword_("feature"):
                statements.append(self.parse_feature_block_())
            elif self.is_cur_keyword_("conditionset"):
                statements.append(self.parse_conditionset_())
            elif self.is_cur_keyword_("variation"):
                statements.append(self.parse_feature_block_(variation=True))
            elif self.is_cur_keyword_("table"):
                statements.append(self.parse_table_())
            elif self.is_cur_keyword_("valueRecordDef"):
                statements.append(self.parse_valuerecord_definition_(vertical=False))
            elif (
                self.cur_token_type_ is Lexer.NAME
                and self.cur_token_ in self.extensions
            ):
                statements.append(self.extensions[self.cur_token_](self))
            elif self.cur_token_type_ is Lexer.SYMBOL and self.cur_token_ == ";":
                continue
            else:
                raise FeatureLibError(
                    "Expected feature, languagesystem, lookup, markClass, "
                    'table, or glyph class definition, got {} "{}"'.format(
                        self.cur_token_type_, self.cur_token_
                    ),
                    self.cur_token_location_,
                )
        # Report any missing glyphs at the end of parsing
        if self.missing:
            error = [
                " %s (first found at %s)" % (name, loc)
                for name, loc in self.missing.items()
            ]
            raise FeatureLibError(
                "The following glyph names are referenced but are missing from the "
                "glyph set:\n" + ("\n".join(error)),
                None,
            )
        return self.doc_

    def parse_anchor_(self):
        # Parses an anchor in any of the four formats given in the feature
        # file specification (2.e.vii).
        self.expect_symbol_("<")
        self.expect_keyword_("anchor")
        location = self.cur_token_location_

        if self.next_token_ == "NULL":  # Format D
            self.expect_keyword_("NULL")
            self.expect_symbol_(">")
            return None

        if self.next_token_type_ == Lexer.NAME:  # Format E
            name = self.expect_name_()
            anchordef = self.anchors_.resolve(name)
            if anchordef is None:
                raise FeatureLibError(
                    'Unknown anchor "%s"' % name, self.cur_token_location_
                )
            self.expect_symbol_(">")
            return self.ast.Anchor(
                anchordef.x,
                anchordef.y,
                name=name,
                contourpoint=anchordef.contourpoint,
                xDeviceTable=None,
                yDeviceTable=None,
                location=location,
            )

        x, y = self.expect_number_(variable=True), self.expect_number_(variable=True)

        contourpoint = None
        if self.next_token_ == "contourpoint":  # Format B
            self.expect_keyword_("contourpoint")
            contourpoint = self.expect_number_()

        if self.next_token_ == "<":  # Format C
            xDeviceTable = self.parse_device_()
            yDeviceTable = self.parse_device_()
        else:
            xDeviceTable, yDeviceTable = None, None

        self.expect_symbol_(">")
        return self.ast.Anchor(
            x,
            y,
            name=None,
            contourpoint=contourpoint,
            xDeviceTable=xDeviceTable,
            yDeviceTable=yDeviceTable,
            location=location,
        )

    def parse_anchor_marks_(self):
        # Parses a sequence of ``[<anchor> mark @MARKCLASS]*.``
        anchorMarks = []  # [(self.ast.Anchor, markClassName)*]
        while self.next_token_ == "<":
            anchor = self.parse_anchor_()
            if anchor is None and self.next_token_ != "mark":
                continue  # <anchor NULL> without mark, eg. in GPOS type 5
            self.expect_keyword_("mark")
            markClass = self.expect_markClass_reference_()
            anchorMarks.append((anchor, markClass))
        return anchorMarks

    def parse_anchordef_(self):
        # Parses a named anchor definition (`section 2.e.viii <https://adobe-type-tools.github.io/afdko/OpenTypeFeatureFileSpecification.html#2.e.vii>`_).
        assert self.is_cur_keyword_("anchorDef")
        location = self.cur_token_location_
        x, y = self.expect_number_(), self.expect_number_()
        contourpoint = None
        if self.next_token_ == "contourpoint":
            self.expect_keyword_("contourpoint")
            contourpoint = self.expect_number_()
        name = self.expect_name_()
        self.expect_symbol_(";")
        anchordef = self.ast.AnchorDefinition(
            name, x, y, contourpoint=contourpoint, location=location
        )
        self.anchors_.define(name, anchordef)
        return anchordef

    def parse_anonymous_(self):
        # Parses an anonymous data block (`section 10 <https://adobe-type-tools.github.io/afdko/OpenTypeFeatureFileSpecification.html#10>`_).
        assert self.is_cur_keyword_(("anon", "anonymous"))
        tag = self.expect_tag_()
        _, content, location = self.lexer_.scan_anonymous_block(tag)
        self.advance_lexer_()
        self.expect_symbol_("}")
        end_tag = self.expect_tag_()
        assert tag == end_tag, "bad splitting in Lexer.scan_anonymous_block()"
        self.expect_symbol_(";")
        return self.ast.AnonymousBlock(tag, content, location=location)

    def parse_attach_(self):
        # Parses a GDEF Attach statement (`section 9.b <https://adobe-type-tools.github.io/afdko/OpenTypeFeatureFileSpecification.html#9.b>`_)
        assert self.is_cur_keyword_("Attach")
        location = self.cur_token_location_
        glyphs = self.parse_glyphclass_(accept_glyphname=True)
        contourPoints = {self.expect_number_()}
        while self.next_token_ != ";":
            contourPoints.add(self.expect_number_())
        self.expect_symbol_(";")
        return self.ast.AttachStatement(glyphs, contourPoints, location=location)

    def parse_enumerate_(self, vertical):
        # Parse an enumerated pair positioning rule (`section 6.b.ii <https://adobe-type-tools.github.io/afdko/OpenTypeFeatureFileSpecification.html#6.b.ii>`_).
        assert self.cur_token_ in {"enumerate", "enum"}
        self.advance_lexer_()
        return self.parse_position_(enumerated=True, vertical=vertical)

    def parse_GlyphClassDef_(self):
        # Parses 'GlyphClassDef @BASE, @LIGATURES, @MARKS, @COMPONENTS;'
        assert self.is_cur_keyword_("GlyphClassDef")
        location = self.cur_token_location_
        if self.next_token_ != ",":
            baseGlyphs = self.parse_glyphclass_(accept_glyphname=False)
        else:
            baseGlyphs = None
        self.expect_symbol_(",")
        if self.next_token_ != ",":
            ligatureGlyphs = self.parse_glyphclass_(accept_glyphname=False)
        else:
            ligatureGlyphs = None
        self.expect_symbol_(",")
        if self.next_token_ != ",":
            markGlyphs = self.parse_glyphclass_(accept_glyphname=False)
        else:
            markGlyphs = None
        self.expect_symbol_(",")
        if self.next_token_ != ";":
            componentGlyphs = self.parse_glyphclass_(accept_glyphname=False)
        else:
            componentGlyphs = None
        self.expect_symbol_(";")
        return self.ast.GlyphClassDefStatement(
            baseGlyphs, markGlyphs, ligatureGlyphs, componentGlyphs, location=location
        )

    def parse_glyphclass_definition_(self):
        # Parses glyph class definitions such as '@UPPERCASE = [A-Z];'
        location, name = self.cur_token_location_, self.cur_token_
        self.expect_symbol_("=")
        glyphs = self.parse_glyphclass_(accept_glyphname=False)
        self.expect_symbol_(";")
        glyphclass = self.ast.GlyphClassDefinition(name, glyphs, location=location)
        self.glyphclasses_.define(name, glyphclass)
        return glyphclass

    def split_glyph_range_(self, name, location):
        # Since v1.20, the OpenType Feature File specification allows
        # for dashes in glyph names. A sequence like "a-b-c-d" could
        # therefore mean a single glyph whose name happens to be
        # "a-b-c-d", or it could mean a range from glyph "a" to glyph
        # "b-c-d", or a range from glyph "a-b" to glyph "c-d", or a
        # range from glyph "a-b-c" to glyph "d".Technically, this
        # example could be resolved because the (pretty complex)
        # definition of glyph ranges renders most of these splits
        # invalid. But the specification does not say that a compiler
        # should try to apply such fancy heuristics. To encourage
        # unambiguous feature files, we therefore try all possible
        # splits and reject the feature file if there are multiple
        # splits possible. It is intentional that we don't just emit a
        # warning; warnings tend to get ignored. To fix the problem,
        # font designers can trivially add spaces around the intended
        # split point, and we emit a compiler error that suggests
        # how exactly the source should be rewritten to make things
        # unambiguous.
        parts = name.split("-")
        solutions = []
        for i in range(len(parts)):
            start, limit = "-".join(parts[0:i]), "-".join(parts[i:])
            if start in self.glyphNames_ and limit in self.glyphNames_:
                solutions.append((start, limit))
        if len(solutions) == 1:
            start, limit = solutions[0]
            return start, limit
        elif len(solutions) == 0:
            raise FeatureLibError(
                '"%s" is not a glyph in the font, and it can not be split '
                "into a range of known glyphs" % name,
                location,
            )
        else:
            ranges = " or ".join(['"%s - %s"' % (s, l) for s, l in solutions])
            raise FeatureLibError(
                'Ambiguous glyph range "%s"; '
                "please use %s to clarify what you mean" % (name, ranges),
                location,
            )

    def parse_glyphclass_(self, accept_glyphname, accept_null=False):
        # Parses a glyph class, either named or anonymous, or (if
        # ``bool(accept_glyphname)``) a glyph name. If ``bool(accept_null)`` then
        # also accept the special NULL glyph.
        if accept_glyphname and self.next_token_type_ in (Lexer.NAME, Lexer.CID):
            if accept_null and self.next_token_ == "NULL":
                # If you want a glyph called NULL, you should escape it.
                self.advance_lexer_()
                return self.ast.NullGlyph(location=self.cur_token_location_)
            glyph = self.expect_glyph_()
            self.check_glyph_name_in_glyph_set(glyph)
            return self.ast.GlyphName(glyph, location=self.cur_token_location_)
        if self.next_token_type_ is Lexer.GLYPHCLASS:
            self.advance_lexer_()
            gc = self.glyphclasses_.resolve(self.cur_token_)
            if gc is None:
                raise FeatureLibError(
                    "Unknown glyph class @%s" % self.cur_token_,
                    self.cur_token_location_,
                )
            if isinstance(gc, self.ast.MarkClass):
                return self.ast.MarkClassName(gc, location=self.cur_token_location_)
            else:
                return self.ast.GlyphClassName(gc, location=self.cur_token_location_)

        self.expect_symbol_("[")
        location = self.cur_token_location_
        glyphs = self.ast.GlyphClass(location=location)
        while self.next_token_ != "]":
            if self.next_token_type_ is Lexer.NAME:
                glyph = self.expect_glyph_()
                location = self.cur_token_location_
                if "-" in glyph and self.glyphNames_ and glyph not in self.glyphNames_:
                    start, limit = self.split_glyph_range_(glyph, location)
                    self.check_glyph_name_in_glyph_set(start, limit)
                    glyphs.add_range(
                        start, limit, self.make_glyph_range_(location, start, limit)
                    )
                elif self.next_token_ == "-":
                    start = glyph
                    self.expect_symbol_("-")
                    limit = self.expect_glyph_()
                    self.check_glyph_name_in_glyph_set(start, limit)
                    glyphs.add_range(
                        start, limit, self.make_glyph_range_(location, start, limit)
                    )
                else:
                    if "-" in glyph and not self.glyphNames_:
                        log.warning(
                            str(
                                FeatureLibError(
                                    f"Ambiguous glyph name that looks like a range: {glyph!r}",
                                    location,
                                )
                            )
                        )
                    self.check_glyph_name_in_glyph_set(glyph)
                    glyphs.append(glyph)
            elif self.next_token_type_ is Lexer.CID:
                glyph = self.expect_glyph_()
                if self.next_token_ == "-":
                    range_location = self.cur_token_location_
                    range_start = self.cur_token_
                    self.expect_symbol_("-")
                    range_end = self.expect_cid_()
                    self.check_glyph_name_in_glyph_set(
                        f"cid{range_start:05d}",
                        f"cid{range_end:05d}",
                    )
                    glyphs.add_cid_range(
                        range_start,
                        range_end,
                        self.make_cid_range_(range_location, range_start, range_end),
                    )
                else:
                    glyph_name = f"cid{self.cur_token_:05d}"
                    self.check_glyph_name_in_glyph_set(glyph_name)
                    glyphs.append(glyph_name)
            elif self.next_token_type_ is Lexer.GLYPHCLASS:
                self.advance_lexer_()
                gc = self.glyphclasses_.resolve(self.cur_token_)
                if gc is None:
                    raise FeatureLibError(
                        "Unknown glyph class @%s" % self.cur_token_,
                        self.cur_token_location_,
                    )
                if isinstance(gc, self.ast.MarkClass):
                    gc = self.ast.MarkClassName(gc, location=self.cur_token_location_)
                else:
                    gc = self.ast.GlyphClassName(gc, location=self.cur_token_location_)
                glyphs.add_class(gc)
            else:
                raise FeatureLibError(
                    "Expected glyph name, glyph range, "
                    f"or glyph class reference, found {self.next_token_!r}",
                    self.next_token_location_,
                )
        self.expect_symbol_("]")
        return glyphs

    def parse_glyph_pattern_(self, vertical):
        # Parses a glyph pattern, including lookups and context, e.g.::
        #
        #    a b
        #    a b c' d e
        #    a b c' lookup ChangeC d e
        prefix, glyphs, lookups, values, suffix = ([], [], [], [], [])
        hasMarks = False
        while self.next_token_ not in {"by", "from", ";", ","}:
            gc = self.parse_glyphclass_(accept_glyphname=True)
            marked = False
            if self.next_token_ == "'":
                self.expect_symbol_("'")
                hasMarks = marked = True
            if marked:
                if suffix:
                    # makeotf also reports this as an error, while FontForge
                    # silently inserts ' in all the intervening glyphs.
                    # https://github.com/fonttools/fonttools/pull/1096
                    raise FeatureLibError(
                        "Unsupported contextual target sequence: at most "
                        "one run of marked (') glyph/class names allowed",
                        self.cur_token_location_,
                    )
                glyphs.append(gc)
            elif glyphs:
                suffix.append(gc)
            else:
                prefix.append(gc)

            if self.is_next_value_():
                values.append(self.parse_valuerecord_(vertical))
            else:
                values.append(None)

            lookuplist = None
            while self.next_token_ == "lookup":
                if lookuplist is None:
                    lookuplist = []
                self.expect_keyword_("lookup")
                if not marked:
                    raise FeatureLibError(
                        "Lookups can only follow marked glyphs",
                        self.cur_token_location_,
                    )
                lookup_name = self.expect_name_()
                lookup = self.lookups_.resolve(lookup_name)
                if lookup is None:
                    raise FeatureLibError(
                        'Unknown lookup "%s"' % lookup_name, self.cur_token_location_
                    )
                lookuplist.append(lookup)
            if marked:
                lookups.append(lookuplist)

        if not glyphs and not suffix:  # eg., "sub f f i by"
            assert lookups == []
            return ([], prefix, [None] * len(prefix), values, [], hasMarks)
        else:
            if any(values[: len(prefix)]):
                raise FeatureLibError(
                    "Positioning cannot be applied in the bactrack glyph sequence, "
                    "before the marked glyph sequence.",
                    self.cur_token_location_,
                )
            marked_values = values[len(prefix) : len(prefix) + len(glyphs)]
            if any(marked_values):
                if any(values[len(prefix) + len(glyphs) :]):
                    raise FeatureLibError(
                        "Positioning values are allowed only in the marked glyph "
                        "sequence, or after the final glyph node when only one glyph "
                        "node is marked.",
                        self.cur_token_location_,
                    )
                values = marked_values
            elif values and values[-1]:
                if len(glyphs) > 1 or any(values[:-1]):
                    raise FeatureLibError(
                        "Positioning values are allowed only in the marked glyph "
                        "sequence, or after the final glyph node when only one glyph "
                        "node is marked.",
                        self.cur_token_location_,
                    )
                values = values[-1:]
            elif any(values):
                raise FeatureLibError(
                    "Positioning values are allowed only in the marked glyph "
                    "sequence, or after the final glyph node when only one glyph "
                    "node is marked.",
                    self.cur_token_location_,
                )
            return (prefix, glyphs, lookups, values, suffix, hasMarks)

    def parse_ignore_glyph_pattern_(self, sub):
        location = self.cur_token_location_
        prefix, glyphs, lookups, values, suffix, hasMarks = self.parse_glyph_pattern_(
            vertical=False
        )
        if any(lookups):
            raise FeatureLibError(
                f'No lookups can be specified for "ignore {sub}"', location
            )
        if not hasMarks:
            error = FeatureLibError(
                f'Ambiguous "ignore {sub}", there should be least one marked glyph',
                location,
            )
            log.warning(str(error))
            suffix, glyphs = glyphs[1:], glyphs[0:1]
        chainContext = (prefix, glyphs, suffix)
        return chainContext

    def parse_ignore_context_(self, sub):
        location = self.cur_token_location_
        chainContext = [self.parse_ignore_glyph_pattern_(sub)]
        while self.next_token_ == ",":
            self.expect_symbol_(",")
            chainContext.append(self.parse_ignore_glyph_pattern_(sub))
        self.expect_symbol_(";")
        return chainContext

    def parse_ignore_(self):
        # Parses an ignore sub/pos rule.
        assert self.is_cur_keyword_("ignore")
        location = self.cur_token_location_
        self.advance_lexer_()
        if self.cur_token_ in ["substitute", "sub"]:
            chainContext = self.parse_ignore_context_("sub")
            return self.ast.IgnoreSubstStatement(chainContext, location=location)
        if self.cur_token_ in ["position", "pos"]:
            chainContext = self.parse_ignore_context_("pos")
            return self.ast.IgnorePosStatement(chainContext, location=location)
        raise FeatureLibError(
            'Expected "substitute" or "position"', self.cur_token_location_
        )

    def parse_include_(self):
        assert self.cur_token_ == "include"
        location = self.cur_token_location_
        filename = self.expect_filename_()
        # self.expect_symbol_(";")
        return ast.IncludeStatement(filename, location=location)

    def parse_language_(self):
        assert self.is_cur_keyword_("language")
        location = self.cur_token_location_
        language = self.expect_language_tag_()
        include_default, required = (True, False)
        if self.next_token_ in {"exclude_dflt", "include_dflt"}:
            include_default = self.expect_name_() == "include_dflt"
        if self.next_token_ == "required":
            self.expect_keyword_("required")
            required = True
        self.expect_symbol_(";")
        return self.ast.LanguageStatement(
            language, include_default, required, location=location
        )

    def parse_ligatureCaretByIndex_(self):
        assert self.is_cur_keyword_("LigatureCaretByIndex")
        location = self.cur_token_location_
        glyphs = self.parse_glyphclass_(accept_glyphname=True)
        carets = [self.expect_number_()]
        while self.next_token_ != ";":
            carets.append(self.expect_number_())
        self.expect_symbol_(";")
        return self.ast.LigatureCaretByIndexStatement(glyphs, carets, location=location)

    def parse_ligatureCaretByPos_(self):
        assert self.is_cur_keyword_("LigatureCaretByPos")
        location = self.cur_token_location_
        glyphs = self.parse_glyphclass_(accept_glyphname=True)
        carets = [self.expect_number_(variable=True)]
        while self.next_token_ != ";":
            carets.append(self.expect_number_(variable=True))
        self.expect_symbol_(";")
        return self.ast.LigatureCaretByPosStatement(glyphs, carets, location=location)

    def parse_lookup_(self, vertical):
        # Parses a ``lookup`` - either a lookup block, or a lookup reference
        # inside a feature.
        assert self.is_cur_keyword_("lookup")
        location, name = self.cur_token_location_, self.expect_name_()

        if self.next_token_ == ";":
            lookup = self.lookups_.resolve(name)
            if lookup is None:
                raise FeatureLibError(
                    'Unknown lookup "%s"' % name, self.cur_token_location_
                )
            self.expect_symbol_(";")
            return self.ast.LookupReferenceStatement(lookup, location=location)

        use_extension = False
        if self.next_token_ == "useExtension":
            self.expect_keyword_("useExtension")
            use_extension = True

        block = self.ast.LookupBlock(name, use_extension, location=location)
        self.parse_block_(block, vertical)
        self.lookups_.define(name, block)
        return block

    def parse_lookupflag_(self):
        # Parses a ``lookupflag`` statement, either specified by number or
        # in words.
        assert self.is_cur_keyword_("lookupflag")
        location = self.cur_token_location_

        # format B: "lookupflag 6;"
        if self.next_token_type_ == Lexer.NUMBER:
            value = self.expect_number_()
            self.expect_symbol_(";")
            return self.ast.LookupFlagStatement(value, location=location)

        # format A: "lookupflag RightToLeft MarkAttachmentType @M;"
        value_seen = False
        value, markAttachment, markFilteringSet = 0, None, None
        flags = {
            "RightToLeft": 1,
            "IgnoreBaseGlyphs": 2,
            "IgnoreLigatures": 4,
            "IgnoreMarks": 8,
        }
        seen = set()
        while self.next_token_ != ";":
            if self.next_token_ in seen:
                raise FeatureLibError(
                    "%s can be specified only once" % self.next_token_,
                    self.next_token_location_,
                )
            seen.add(self.next_token_)
            if self.next_token_ == "MarkAttachmentType":
                self.expect_keyword_("MarkAttachmentType")
                markAttachment = self.parse_glyphclass_(accept_glyphname=False)
            elif self.next_token_ == "UseMarkFilteringSet":
                self.expect_keyword_("UseMarkFilteringSet")
                markFilteringSet = self.parse_glyphclass_(accept_glyphname=False)
            elif self.next_token_ in flags:
                value_seen = True
                value = value | flags[self.expect_name_()]
            else:
                raise FeatureLibError(
                    '"%s" is not a recognized lookupflag' % self.next_token_,
                    self.next_token_location_,
                )
        self.expect_symbol_(";")

        if not any([value_seen, markAttachment, markFilteringSet]):
            raise FeatureLibError(
                "lookupflag must have a value", self.next_token_location_
            )

        return self.ast.LookupFlagStatement(
            value,
            markAttachment=markAttachment,
            markFilteringSet=markFilteringSet,
            location=location,
        )

    def parse_markClass_(self):
        assert self.is_cur_keyword_("markClass")
        location = self.cur_token_location_
        glyphs = self.parse_glyphclass_(accept_glyphname=True)
        if not glyphs.glyphSet():
            raise FeatureLibError(
                "Empty glyph class in mark class definition", location
            )
        anchor = self.parse_anchor_()
        name = self.expect_class_name_()
        self.expect_symbol_(";")
        markClass = self.doc_.markClasses.get(name)
        if markClass is None:
            markClass = self.ast.MarkClass(name)
            self.doc_.markClasses[name] = markClass
            self.glyphclasses_.define(name, markClass)
        mcdef = self.ast.MarkClassDefinition(
            markClass, anchor, glyphs, location=location
        )
        markClass.addDefinition(mcdef)
        return mcdef

    def parse_position_(self, enumerated, vertical):
        assert self.cur_token_ in {"position", "pos"}
        if self.next_token_ == "cursive":  # GPOS type 3
            return self.parse_position_cursive_(enumerated, vertical)
        elif self.next_token_ == "base":  # GPOS type 4
            return self.parse_position_base_(enumerated, vertical)
        elif self.next_token_ == "ligature":  # GPOS type 5
            return self.parse_position_ligature_(enumerated, vertical)
        elif self.next_token_ == "mark":  # GPOS type 6
            return self.parse_position_mark_(enumerated, vertical)

        location = self.cur_token_location_
        prefix, glyphs, lookups, values, suffix, hasMarks = self.parse_glyph_pattern_(
            vertical
        )
        self.expect_symbol_(";")

        if any(lookups):
            # GPOS type 8: Chaining contextual positioning; explicit lookups
            if any(values):
                raise FeatureLibError(
                    'If "lookup" is present, no values must be specified', location
                )
            return self.ast.ChainContextPosStatement(
                prefix, glyphs, suffix, lookups, location=location
            )

        # Pair positioning, format A: "pos V 10 A -10;"
        # Pair positioning, format B: "pos V A -20;"
        if not prefix and not suffix and len(glyphs) == 2 and not hasMarks:
            if values[0] is None:  # Format B: "pos V A -20;"
                values.reverse()
            return self.ast.PairPosStatement(
                glyphs[0],
                values[0],
                glyphs[1],
                values[1],
                enumerated=enumerated,
                location=location,
            )

        if enumerated:
            raise FeatureLibError(
                '"enumerate" is only allowed with pair positionings', location
            )
        return self.ast.SinglePosStatement(
            list(zip(glyphs, values)),
            prefix,
            suffix,
            forceChain=hasMarks,
            location=location,
        )

    def parse_position_cursive_(self, enumerated, vertical):
        location = self.cur_token_location_
        self.expect_keyword_("cursive")
        if enumerated:
            raise FeatureLibError(
                '"enumerate" is not allowed with ' "cursive attachment positioning",
                location,
            )
        glyphclass = self.parse_glyphclass_(accept_glyphname=True)
        entryAnchor = self.parse_anchor_()
        exitAnchor = self.parse_anchor_()
        self.expect_symbol_(";")
        return self.ast.CursivePosStatement(
            glyphclass, entryAnchor, exitAnchor, location=location
        )

    def parse_position_base_(self, enumerated, vertical):
        location = self.cur_token_location_
        self.expect_keyword_("base")
        if enumerated:
            raise FeatureLibError(
                '"enumerate" is not allowed with '
                "mark-to-base attachment positioning",
                location,
            )
        base = self.parse_glyphclass_(accept_glyphname=True)
        marks = self.parse_anchor_marks_()
        self.expect_symbol_(";")
        return self.ast.MarkBasePosStatement(base, marks, location=location)

    def parse_position_ligature_(self, enumerated, vertical):
        location = self.cur_token_location_
        self.expect_keyword_("ligature")
        if enumerated:
            raise FeatureLibError(
                '"enumerate" is not allowed with '
                "mark-to-ligature attachment positioning",
                location,
            )
        ligatures = self.parse_glyphclass_(accept_glyphname=True)
        marks = [self.parse_anchor_marks_()]
        while self.next_token_ == "ligComponent":
            self.expect_keyword_("ligComponent")
            marks.append(self.parse_anchor_marks_())
        self.expect_symbol_(";")
        return self.ast.MarkLigPosStatement(ligatures, marks, location=location)

    def parse_position_mark_(self, enumerated, vertical):
        location = self.cur_token_location_
        self.expect_keyword_("mark")
        if enumerated:
            raise FeatureLibError(
                '"enumerate" is not allowed with '
                "mark-to-mark attachment positioning",
                location,
            )
        baseMarks = self.parse_glyphclass_(accept_glyphname=True)
        marks = self.parse_anchor_marks_()
        self.expect_symbol_(";")
        return self.ast.MarkMarkPosStatement(baseMarks, marks, location=location)

    def parse_script_(self):
        assert self.is_cur_keyword_("script")
        location, script = self.cur_token_location_, self.expect_script_tag_()
        self.expect_symbol_(";")
        return self.ast.ScriptStatement(script, location=location)

    def parse_substitute_(self):
        assert self.cur_token_ in {"substitute", "sub", "reversesub", "rsub"}
        location = self.cur_token_location_
        reverse = self.cur_token_ in {"reversesub", "rsub"}
        (
            old_prefix,
            old,
            lookups,
            values,
            old_suffix,
            hasMarks,
        ) = self.parse_glyph_pattern_(vertical=False)
        if any(values):
            raise FeatureLibError(
                "Substitution statements cannot contain values", location
            )
        new = []
        if self.next_token_ == "by":
            keyword = self.expect_keyword_("by")
            while self.next_token_ != ";":
                gc = self.parse_glyphclass_(accept_glyphname=True, accept_null=True)
                new.append(gc)
        elif self.next_token_ == "from":
            keyword = self.expect_keyword_("from")
            new = [self.parse_glyphclass_(accept_glyphname=False)]
        else:
            keyword = None
        self.expect_symbol_(";")
        if len(new) == 0 and not any(lookups):
            raise FeatureLibError(
                'Expected "by", "from" or explicit lookup references',
                self.cur_token_location_,
            )

        # GSUB lookup type 3: Alternate substitution.
        # Format: "substitute a from [a.1 a.2 a.3];"
        if keyword == "from":
            if reverse:
                raise FeatureLibError(
                    'Reverse chaining substitutions do not support "from"', location
                )
            if len(old) != 1 or len(old[0].glyphSet()) != 1:
                raise FeatureLibError('Expected a single glyph before "from"', location)
            if len(new) != 1:
                raise FeatureLibError(
                    'Expected a single glyphclass after "from"', location
                )
            return self.ast.AlternateSubstStatement(
                old_prefix, old[0], old_suffix, new[0], location=location
            )

        num_lookups = len([l for l in lookups if l is not None])

        is_deletion = False
        if len(new) == 1 and isinstance(new[0], ast.NullGlyph):
            if reverse:
                raise FeatureLibError(
                    "Reverse chaining substitutions do not support glyph deletion",
                    location,
                )
            new = []  # Deletion
            is_deletion = True

        # GSUB lookup type 1: Single substitution.
        # Format A: "substitute a by a.sc;"
        # Format B: "substitute [one.fitted one.oldstyle] by one;"
        # Format C: "substitute [a-d] by [A.sc-D.sc];"
        if not reverse and len(old) == 1 and len(new) == 1 and num_lookups == 0:
            glyphs = list(old[0].glyphSet())
            replacements = list(new[0].glyphSet())
            if len(replacements) == 1:
                replacements = replacements * len(glyphs)
            if len(glyphs) != len(replacements):
                raise FeatureLibError(
                    'Expected a glyph class with %d elements after "by", '
                    "but found a glyph class with %d elements"
                    % (len(glyphs), len(replacements)),
                    location,
                )
            return self.ast.SingleSubstStatement(
                old, new, old_prefix, old_suffix, forceChain=hasMarks, location=location
            )

        # Glyph deletion, built as GSUB lookup type 2: Multiple substitution
        # with empty replacement.
        if is_deletion and len(old) == 1 and num_lookups == 0:
            return self.ast.MultipleSubstStatement(
                old_prefix,
                old[0],
                old_suffix,
                (),
                forceChain=hasMarks,
                location=location,
            )

        # GSUB lookup type 2: Multiple substitution.
        # Format: "substitute f_f_i by f f i;"
        #
        # GlyphsApp introduces two additional formats:
        # Format 1: "substitute [f_i f_l] by [f f] [i l];"
        # Format 2: "substitute [f_i f_l] by f [i l];"
        # http://handbook.glyphsapp.com/en/layout/multiple-substitution-with-classes/
        if not reverse and len(old) == 1 and len(new) > 1 and num_lookups == 0:
            count = len(old[0].glyphSet())
            for n in new:
                if not list(n.glyphSet()):
                    raise FeatureLibError("Empty class in replacement", location)
                if len(n.glyphSet()) != 1 and len(n.glyphSet()) != count:
                    raise FeatureLibError(
                        f'Expected a glyph class with 1 or {count} elements after "by", '
                        f"but found a glyph class with {len(n.glyphSet())} elements",
                        location,
                    )
            return self.ast.MultipleSubstStatement(
                old_prefix,
                old[0],
                old_suffix,
                new,
                forceChain=hasMarks,
                location=location,
            )

        # GSUB lookup type 4: Ligature substitution.
        # Format: "substitute f f i by f_f_i;"
        if (
            not reverse
            and len(old) > 1
            and len(new) == 1
            and len(new[0].glyphSet()) == 1
            and num_lookups == 0
        ):
            return self.ast.LigatureSubstStatement(
                old_prefix,
                old,
                old_suffix,
                list(new[0].glyphSet())[0],
                forceChain=hasMarks,
                location=location,
            )

        # GSUB lookup type 8: Reverse chaining substitution.
        if reverse:
            if len(old) != 1:
                raise FeatureLibError(
                    "In reverse chaining single substitutions, "
                    "only a single glyph or glyph class can be replaced",
                    location,
                )
            if len(new) != 1:
                raise FeatureLibError(
                    "In reverse chaining single substitutions, "
                    'the replacement (after "by") must be a single glyph '
                    "or glyph class",
                    location,
                )
            if num_lookups != 0:
                raise FeatureLibError(
                    "Reverse chaining substitutions cannot call named lookups", location
                )
            glyphs = sorted(list(old[0].glyphSet()))
            replacements = sorted(list(new[0].glyphSet()))
            if len(replacements) == 1:
                replacements = replacements * len(glyphs)
            if len(glyphs) != len(replacements):
                raise FeatureLibError(
                    'Expected a glyph class with %d elements after "by", '
                    "but found a glyph class with %d elements"
                    % (len(glyphs), len(replacements)),
                    location,
                )
            return self.ast.ReverseChainSingleSubstStatement(
                old_prefix, old_suffix, old, new, location=location
            )

        if len(old) > 1 and len(new) > 1:
            raise FeatureLibError(
                "Direct substitution of multiple glyphs by multiple glyphs "
                "is not supported",
                location,
            )

        # If there are remaining glyphs to parse, this is an invalid GSUB statement
        if len(new) != 0 or is_deletion:
            raise FeatureLibError("Invalid substitution statement", location)

        # GSUB lookup type 6: Chaining contextual substitution.
        rule = self.ast.ChainContextSubstStatement(
            old_prefix, old, old_suffix, lookups, location=location
        )
        return rule

    def parse_subtable_(self):
        assert self.is_cur_keyword_("subtable")
        location = self.cur_token_location_
        self.expect_symbol_(";")
        return self.ast.SubtableStatement(location=location)

    def parse_size_parameters_(self):
        # Parses a ``parameters`` statement used in ``size`` features. See
        # `section 8.b <https://adobe-type-tools.github.io/afdko/OpenTypeFeatureFileSpecification.html#8.b>`_.
        assert self.is_cur_keyword_("parameters")
        location = self.cur_token_location_
        DesignSize = self.expect_decipoint_()
        SubfamilyID = self.expect_number_()
        RangeStart = 0.0
        RangeEnd = 0.0
        if self.next_token_type_ in (Lexer.NUMBER, Lexer.FLOAT) or SubfamilyID != 0:
            RangeStart = self.expect_decipoint_()
            RangeEnd = self.expect_decipoint_()

        self.expect_symbol_(";")
        return self.ast.SizeParameters(
            DesignSize, SubfamilyID, RangeStart, RangeEnd, location=location
        )

    def parse_size_menuname_(self):
        assert self.is_cur_keyword_("sizemenuname")
        location = self.cur_token_location_
        platformID, platEncID, langID, string = self.parse_name_()
        return self.ast.FeatureNameStatement(
            "size", platformID, platEncID, langID, string, location=location
        )

    def parse_table_(self):
        assert self.is_cur_keyword_("table")
        location, name = self.cur_token_location_, self.expect_tag_()
        table = self.ast.TableBlock(name, location=location)
        self.expect_symbol_("{")
        handler = {
            "GDEF": self.parse_table_GDEF_,
            "head": self.parse_table_head_,
            "hhea": self.parse_table_hhea_,
            "vhea": self.parse_table_vhea_,
            "name": self.parse_table_name_,
            "BASE": self.parse_table_BASE_,
            "OS/2": self.parse_table_OS_2_,
            "STAT": self.parse_table_STAT_,
        }.get(name)
        if handler:
            handler(table)
        else:
            raise FeatureLibError(
                '"table %s" is not supported' % name.strip(), location
            )
        self.expect_symbol_("}")
        end_tag = self.expect_tag_()
        if end_tag != name:
            raise FeatureLibError(
                'Expected "%s"' % name.strip(), self.cur_token_location_
            )
        self.expect_symbol_(";")
        return table

    def parse_table_GDEF_(self, table):
        statements = table.statements
        while self.next_token_ != "}" or self.cur_comments_:
            self.advance_lexer_(comments=True)
            if self.cur_token_type_ is Lexer.COMMENT:
                statements.append(
                    self.ast.Comment(self.cur_token_, location=self.cur_token_location_)
                )
            elif self.is_cur_keyword_("Attach"):
                statements.append(self.parse_attach_())
            elif self.is_cur_keyword_("GlyphClassDef"):
                statements.append(self.parse_GlyphClassDef_())
            elif self.is_cur_keyword_("LigatureCaretByIndex"):
                statements.append(self.parse_ligatureCaretByIndex_())
            elif self.is_cur_keyword_("LigatureCaretByPos"):
                statements.append(self.parse_ligatureCaretByPos_())
            elif self.cur_token_ == ";":
                continue
            else:
                raise FeatureLibError(
                    "Expected Attach, LigatureCaretByIndex, " "or LigatureCaretByPos",
                    self.cur_token_location_,
                )

    def parse_table_head_(self, table):
        statements = table.statements
        while self.next_token_ != "}" or self.cur_comments_:
            self.advance_lexer_(comments=True)
            if self.cur_token_type_ is Lexer.COMMENT:
                statements.append(
                    self.ast.Comment(self.cur_token_, location=self.cur_token_location_)
                )
            elif self.is_cur_keyword_("FontRevision"):
                statements.append(self.parse_FontRevision_())
            elif self.cur_token_ == ";":
                continue
            else:
                raise FeatureLibError("Expected FontRevision", self.cur_token_location_)

    def parse_table_hhea_(self, table):
        statements = table.statements
        fields = ("CaretOffset", "Ascender", "Descender", "LineGap")
        while self.next_token_ != "}" or self.cur_comments_:
            self.advance_lexer_(comments=True)
            if self.cur_token_type_ is Lexer.COMMENT:
                statements.append(
                    self.ast.Comment(self.cur_token_, location=self.cur_token_location_)
                )
            elif self.cur_token_type_ is Lexer.NAME and self.cur_token_ in fields:
                key = self.cur_token_.lower()
                value = self.expect_number_()
                statements.append(
                    self.ast.HheaField(key, value, location=self.cur_token_location_)
                )
                if self.next_token_ != ";":
                    raise FeatureLibError(
                        "Incomplete statement", self.next_token_location_
                    )
            elif self.cur_token_ == ";":
                continue
            else:
                raise FeatureLibError(
                    "Expected CaretOffset, Ascender, " "Descender or LineGap",
                    self.cur_token_location_,
                )

    def parse_table_vhea_(self, table):
        statements = table.statements
        fields = ("VertTypoAscender", "VertTypoDescender", "VertTypoLineGap")
        while self.next_token_ != "}" or self.cur_comments_:
            self.advance_lexer_(comments=True)
            if self.cur_token_type_ is Lexer.COMMENT:
                statements.append(
                    self.ast.Comment(self.cur_token_, location=self.cur_token_location_)
                )
            elif self.cur_token_type_ is Lexer.NAME and self.cur_token_ in fields:
                key = self.cur_token_.lower()
                value = self.expect_number_()
                statements.append(
                    self.ast.VheaField(key, value, location=self.cur_token_location_)
                )
                if self.next_token_ != ";":
                    raise FeatureLibError(
                        "Incomplete statement", self.next_token_location_
                    )
            elif self.cur_token_ == ";":
                continue
            else:
                raise FeatureLibError(
                    "Expected VertTypoAscender, "
                    "VertTypoDescender or VertTypoLineGap",
                    self.cur_token_location_,
                )

    def parse_table_name_(self, table):
        statements = table.statements
        while self.next_token_ != "}" or self.cur_comments_:
            self.advance_lexer_(comments=True)
            if self.cur_token_type_ is Lexer.COMMENT:
                statements.append(
                    self.ast.Comment(self.cur_token_, location=self.cur_token_location_)
                )
            elif self.is_cur_keyword_("nameid"):
                statement = self.parse_nameid_()
                if statement:
                    statements.append(statement)
            elif self.cur_token_ == ";":
                continue
            else:
                raise FeatureLibError("Expected nameid", self.cur_token_location_)

    def parse_name_(self):
        """Parses a name record. See `section 9.e <https://adobe-type-tools.github.io/afdko/OpenTypeFeatureFileSpecification.html#9.e>`_."""
        platEncID = None
        langID = None
        if self.next_token_type_ in Lexer.NUMBERS:
            platformID = self.expect_any_number_()
            location = self.cur_token_location_
            if platformID not in (1, 3):
                raise FeatureLibError("Expected platform id 1 or 3", location)
            if self.next_token_type_ in Lexer.NUMBERS:
                platEncID = self.expect_any_number_()
                langID = self.expect_any_number_()
        else:
            platformID = 3
            location = self.cur_token_location_

        if platformID == 1:  # Macintosh
            platEncID = platEncID or 0  # Roman
            langID = langID or 0  # English
        else:  # 3, Windows
            platEncID = platEncID or 1  # Unicode
            langID = langID or 0x0409  # English

        string = self.expect_string_()
        self.expect_symbol_(";")

        encoding = getEncoding(platformID, platEncID, langID)
        if encoding is None:
            raise FeatureLibError("Unsupported encoding", location)
        unescaped = self.unescape_string_(string, encoding)
        return platformID, platEncID, langID, unescaped

    def parse_stat_name_(self):
        platEncID = None
        langID = None
        if self.next_token_type_ in Lexer.NUMBERS:
            platformID = self.expect_any_number_()
            location = self.cur_token_location_
            if platformID not in (1, 3):
                raise FeatureLibError("Expected platform id 1 or 3", location)
            if self.next_token_type_ in Lexer.NUMBERS:
                platEncID = self.expect_any_number_()
                langID = self.expect_any_number_()
        else:
            platformID = 3
            location = self.cur_token_location_

        if platformID == 1:  # Macintosh
            platEncID = platEncID or 0  # Roman
            langID = langID or 0  # English
        else:  # 3, Windows
            platEncID = platEncID or 1  # Unicode
            langID = langID or 0x0409  # English

        string = self.expect_string_()
        encoding = getEncoding(platformID, platEncID, langID)
        if encoding is None:
            raise FeatureLibError("Unsupported encoding", location)
        unescaped = self.unescape_string_(string, encoding)
        return platformID, platEncID, langID, unescaped

    def parse_nameid_(self):
        assert self.cur_token_ == "nameid", self.cur_token_
        location, nameID = self.cur_token_location_, self.expect_any_number_()
        if nameID > 32767:
            raise FeatureLibError(
                "Name id value cannot be greater than 32767", self.cur_token_location_
            )
        platformID, platEncID, langID, string = self.parse_name_()
        return self.ast.NameRecord(
            nameID, platformID, platEncID, langID, string, location=location
        )

    def unescape_string_(self, string, encoding):
        if encoding == "utf_16_be":
            s = re.sub(r"\\[0-9a-fA-F]{4}", self.unescape_unichr_, string)
        else:
            # Build the full byte sequence from literal characters plus
            # \XX escapes, then decode in one go. This is required for
            # multibyte encodings (e.g. x_mac_japanese_ttx) where bytes
            # belonging to a single character may straddle literal/escape
            # boundaries — e.g. "\95F" is one Shift-JIS char (0x95 0x46).
            out = bytearray()
            for token in re.split(r"(\\[0-9a-fA-F]{2})", string):
                if len(token) == 3 and token[0] == "\\":
                    out.append(int(token[1:], 16))
                elif token:
                    out.extend(token.encode(encoding))
            s = out.decode(encoding)
        # We now have a Unicode string, but it might contain surrogate pairs.
        # We convert surrogates to actual Unicode by round-tripping through
        # Python's UTF-16 codec in a special mode.
        utf16 = tobytes(s, "utf_16_be", "surrogatepass")
        return tostr(utf16, "utf_16_be")

    @staticmethod
    def unescape_unichr_(match):
        n = match.group(0)[1:]
        return chr(int(n, 16))

    def find_previous(self, statements, class_):
        for previous in reversed(statements):
            if isinstance(previous, self.ast.Comment):
                continue
            elif isinstance(previous, class_):
                return previous
            else:
                # If we find something that doesn't match what we're looking
                # for, and isn't a comment, fail
                return None
        # Out of statements to look at
        return None

    def parse_table_BASE_(self, table):
        statements = table.statements
        while self.next_token_ != "}" or self.cur_comments_:
            self.advance_lexer_(comments=True)
            if self.cur_token_type_ is Lexer.COMMENT:
                statements.append(
                    self.ast.Comment(self.cur_token_, location=self.cur_token_location_)
                )
            elif self.is_cur_keyword_("HorizAxis.BaseTagList"):
                horiz_bases = self.parse_base_tag_list_()
            elif self.is_cur_keyword_("HorizAxis.BaseScriptList"):
                horiz_scripts = self.parse_base_script_list_(len(horiz_bases))
                statements.append(
                    self.ast.BaseAxis(
                        horiz_bases,
                        horiz_scripts,
                        False,
                        location=self.cur_token_location_,
                    )
                )
            elif self.is_cur_keyword_("HorizAxis.MinMax"):
                base_script_list = self.find_previous(statements, ast.BaseAxis)
                if base_script_list is None:
                    raise FeatureLibError(
                        "MinMax must be preceded by BaseScriptList",
                        self.cur_token_location_,
                    )
                if base_script_list.vertical:
                    raise FeatureLibError(
                        "HorizAxis.MinMax must be preceded by HorizAxis statements",
                        self.cur_token_location_,
                    )
                base_script_list.minmax.append(self.parse_base_minmax_())
            elif self.is_cur_keyword_("VertAxis.BaseTagList"):
                vert_bases = self.parse_base_tag_list_()
            elif self.is_cur_keyword_("VertAxis.BaseScriptList"):
                vert_scripts = self.parse_base_script_list_(len(vert_bases))
                statements.append(
                    self.ast.BaseAxis(
                        vert_bases,
                        vert_scripts,
                        True,
                        location=self.cur_token_location_,
                    )
                )
            elif self.is_cur_keyword_("VertAxis.MinMax"):
                base_script_list = self.find_previous(statements, ast.BaseAxis)
                if base_script_list is None:
                    raise FeatureLibError(
                        "MinMax must be preceded by BaseScriptList",
                        self.cur_token_location_,
                    )
                if not base_script_list.vertical:
                    raise FeatureLibError(
                        "VertAxis.MinMax must be preceded by VertAxis statements",
                        self.cur_token_location_,
                    )
                base_script_list.minmax.append(self.parse_base_minmax_())
            elif self.cur_token_ == ";":
                continue

    def parse_table_OS_2_(self, table):
        statements = table.statements
        numbers = (
            "FSType",
            "TypoAscender",
            "TypoDescender",
            "TypoLineGap",
            "winAscent",
            "winDescent",
            "XHeight",
            "CapHeight",
            "WeightClass",
            "WidthClass",
            "LowerOpSize",
            "UpperOpSize",
        )
        ranges = ("UnicodeRange", "CodePageRange")
        while self.next_token_ != "}" or self.cur_comments_:
            self.advance_lexer_(comments=True)
            if self.cur_token_type_ is Lexer.COMMENT:
                statements.append(
                    self.ast.Comment(self.cur_token_, location=self.cur_token_location_)
                )
            elif self.cur_token_type_ is Lexer.NAME:
                key = self.cur_token_.lower()
                value = None
                if self.cur_token_ in numbers:
                    value = self.expect_number_()
                elif self.is_cur_keyword_("Panose"):
                    value = []
                    for i in range(10):
                        value.append(self.expect_number_())
                elif self.cur_token_ in ranges:
                    value = []
                    while self.next_token_ != ";":
                        value.append(self.expect_number_())
                elif self.is_cur_keyword_("Vendor"):
                    value = self.expect_string_()
                statements.append(
                    self.ast.OS2Field(key, value, location=self.cur_token_location_)
                )
            elif self.cur_token_ == ";":
                continue

    def parse_STAT_ElidedFallbackName(self):
        assert self.is_cur_keyword_("ElidedFallbackName")
        self.expect_symbol_("{")
        names = []
        while self.next_token_ != "}" or self.cur_comments_:
            self.advance_lexer_()
            if self.is_cur_keyword_("name"):
                platformID, platEncID, langID, string = self.parse_stat_name_()
                nameRecord = self.ast.STATNameStatement(
                    "stat",
                    platformID,
                    platEncID,
                    langID,
                    string,
                    location=self.cur_token_location_,
                )
                names.append(nameRecord)
            else:
                if self.cur_token_ != ";":
                    raise FeatureLibError(
                        f"Unexpected token {self.cur_token_} " f"in ElidedFallbackName",
                        self.cur_token_location_,
                    )
        self.expect_symbol_("}")
        if not names:
            raise FeatureLibError('Expected "name"', self.cur_token_location_)
        return names

    def parse_STAT_design_axis(self):
        assert self.is_cur_keyword_("DesignAxis")
        names = []
        axisTag = self.expect_tag_()
        if (
            axisTag not in ("ital", "opsz", "slnt", "wdth", "wght")
            and not axisTag.isupper()
        ):
            log.warning(f"Unregistered axis tag {axisTag} should be uppercase.")
        axisOrder = self.expect_number_()
        self.expect_symbol_("{")
        while self.next_token_ != "}" or self.cur_comments_:
            self.advance_lexer_()
            if self.cur_token_type_ is Lexer.COMMENT:
                continue
            elif self.is_cur_keyword_("name"):
                location = self.cur_token_location_
                platformID, platEncID, langID, string = self.parse_stat_name_()
                name = self.ast.STATNameStatement(
                    "stat", platformID, platEncID, langID, string, location=location
                )
                names.append(name)
            elif self.cur_token_ == ";":
                continue
            else:
                raise FeatureLibError(
                    f'Expected "name", got {self.cur_token_}', self.cur_token_location_
                )

        self.expect_symbol_("}")
        return self.ast.STATDesignAxisStatement(
            axisTag, axisOrder, names, self.cur_token_location_
        )

    def parse_STAT_axis_value_(self):
        assert self.is_cur_keyword_("AxisValue")
        self.expect_symbol_("{")
        locations = []
        names = []
        flags = 0
        while self.next_token_ != "}" or self.cur_comments_:
            self.advance_lexer_(comments=True)
            if self.cur_token_type_ is Lexer.COMMENT:
                continue
            elif self.is_cur_keyword_("name"):
                location = self.cur_token_location_
                platformID, platEncID, langID, string = self.parse_stat_name_()
                name = self.ast.STATNameStatement(
                    "stat", platformID, platEncID, langID, string, location=location
                )
                names.append(name)
            elif self.is_cur_keyword_("location"):
                location = self.parse_STAT_location()
                locations.append(location)
            elif self.is_cur_keyword_("flag"):
                flags = self.expect_stat_flags()
            elif self.cur_token_ == ";":
                continue
            else:
                raise FeatureLibError(
                    f"Unexpected token {self.cur_token_} " f"in AxisValue",
                    self.cur_token_location_,
                )
        self.expect_symbol_("}")
        if not names:
            raise FeatureLibError('Expected "Axis Name"', self.cur_token_location_)
        if not locations:
            raise FeatureLibError('Expected "Axis location"', self.cur_token_location_)
        if len(locations) > 1:
            for location in locations:
                if len(location.values) > 1:
                    raise FeatureLibError(
                        "Only one value is allowed in a "
                        "Format 4 Axis Value Record, but "
                        f"{len(location.values)} were found.",
                        self.cur_token_location_,
                    )
            format4_tags = []
            for location in locations:
                tag = location.tag
                if tag in format4_tags:
                    raise FeatureLibError(
                        f"Axis tag {tag} already " "defined.", self.cur_token_location_
                    )
                format4_tags.append(tag)

        return self.ast.STATAxisValueStatement(
            names, locations, flags, self.cur_token_location_
        )

    def parse_STAT_location(self):
        values = []
        tag = self.expect_tag_()
        if len(tag.strip()) != 4:
            raise FeatureLibError(
                f"Axis tag {self.cur_token_} must be 4 " "characters",
                self.cur_token_location_,
            )

        while self.next_token_ != ";":
            if self.next_token_type_ is Lexer.FLOAT:
                value = self.expect_float_()
                values.append(value)
            elif self.next_token_type_ is Lexer.NUMBER:
                value = self.expect_number_()
                values.append(value)
            else:
                raise FeatureLibError(
                    f'Unexpected value "{self.next_token_}". '
                    "Expected integer or float.",
                    self.next_token_location_,
                )
        if len(values) == 3:
            nominal, min_val, max_val = values
            if nominal < min_val or nominal > max_val:
                raise FeatureLibError(
                    f"Default value {nominal} is outside "
                    f"of specified range "
                    f"{min_val}-{max_val}.",
                    self.next_token_location_,
                )
        return self.ast.AxisValueLocationStatement(tag, values)

    def parse_table_STAT_(self, table):
        statements = table.statements
        design_axes = []
        while self.next_token_ != "}" or self.cur_comments_:
            self.advance_lexer_(comments=True)
            if self.cur_token_type_ is Lexer.COMMENT:
                statements.append(
                    self.ast.Comment(self.cur_token_, location=self.cur_token_location_)
                )
            elif self.cur_token_type_ is Lexer.NAME:
                if self.is_cur_keyword_("ElidedFallbackName"):
                    names = self.parse_STAT_ElidedFallbackName()
                    statements.append(self.ast.ElidedFallbackName(names))
                elif self.is_cur_keyword_("ElidedFallbackNameID"):
                    value = self.expect_number_()
                    statements.append(self.ast.ElidedFallbackNameID(value))
                    self.expect_symbol_(";")
                elif self.is_cur_keyword_("DesignAxis"):
                    designAxis = self.parse_STAT_design_axis()
                    design_axes.append(designAxis.tag)
                    statements.append(designAxis)
                    self.expect_symbol_(";")
                elif self.is_cur_keyword_("AxisValue"):
                    axisValueRecord = self.parse_STAT_axis_value_()
                    for location in axisValueRecord.locations:
                        if location.tag not in design_axes:
                            # Tag must be defined in a DesignAxis before it
                            # can be referenced
                            raise FeatureLibError(
                                "DesignAxis not defined for " f"{location.tag}.",
                                self.cur_token_location_,
                            )
                    statements.append(axisValueRecord)
                    self.expect_symbol_(";")
                else:
                    raise FeatureLibError(
                        f"Unexpected token {self.cur_token_}", self.cur_token_location_
                    )
            elif self.cur_token_ == ";":
                continue

    def parse_base_tag_list_(self):
        # Parses BASE table entries. (See `section 9.a <https://adobe-type-tools.github.io/afdko/OpenTypeFeatureFileSpecification.html#9.a>`_)
        assert self.cur_token_ in (
            "HorizAxis.BaseTagList",
            "VertAxis.BaseTagList",
        ), self.cur_token_
        bases = []
        while self.next_token_ != ";":
            bases.append(self.expect_script_tag_())
        self.expect_symbol_(";")
        return bases

    def parse_base_script_list_(self, count):
        assert self.cur_token_ in (
            "HorizAxis.BaseScriptList",
            "VertAxis.BaseScriptList",
        ), self.cur_token_
        scripts = [self.parse_base_script_record_(count)]
        while self.next_token_ == ",":
            self.expect_symbol_(",")
            scripts.append(self.parse_base_script_record_(count))
        self.expect_symbol_(";")
        return scripts

    def parse_base_script_record_(self, count):
        script_tag = self.expect_script_tag_()
        base_tag = self.expect_script_tag_()
        coords = [self.expect_number_() for i in range(count)]
        return script_tag, base_tag, coords

    def parse_base_minmax_(self):
        script_tag = self.expect_script_tag_()
        language = self.expect_language_tag_()
        min_coord = self.expect_number_()
        self.advance_lexer_()
        if not (self.cur_token_type_ is Lexer.SYMBOL and self.cur_token_ == ","):
            raise FeatureLibError(
                "Expected a comma between min and max coordinates",
                self.cur_token_location_,
            )
        max_coord = self.expect_number_()
        if self.next_token_ == ",":  # feature tag...
            raise FeatureLibError(
                "Feature tags are not yet supported in BASE table",
                self.cur_token_location_,
            )

        return script_tag, language, min_coord, max_coord

    def parse_device_(self):
        result = None
        self.expect_symbol_("<")
        self.expect_keyword_("device")
        if self.next_token_ == "NULL":
            self.expect_keyword_("NULL")
        else:
            result = [(self.expect_number_(), self.expect_number_())]
            while self.next_token_ == ",":
                self.expect_symbol_(",")
                result.append((self.expect_number_(), self.expect_number_()))
            result = tuple(result)  # make it hashable
        self.expect_symbol_(">")
        return result

    def is_next_value_(self):
        return (
            self.next_token_type_ is Lexer.NUMBER
            or self.next_token_ == "<"
            or self.next_token_ == "("
        )

    def parse_valuerecord_(self, vertical):
        if (
            self.next_token_type_ is Lexer.SYMBOL and self.next_token_ == "("
        ) or self.next_token_type_ is Lexer.NUMBER:
            number, location = (
                self.expect_number_(variable=True),
                self.cur_token_location_,
            )
            if vertical:
                val = self.ast.ValueRecord(
                    yAdvance=number, vertical=vertical, location=location
                )
            else:
                val = self.ast.ValueRecord(
                    xAdvance=number, vertical=vertical, location=location
                )
            return val
        self.expect_symbol_("<")
        location = self.cur_token_location_
        if self.next_token_type_ is Lexer.NAME:
            name = self.expect_name_()
            if name == "NULL":
                self.expect_symbol_(">")
                return self.ast.ValueRecord()
            vrd = self.valuerecords_.resolve(name)
            if vrd is None:
                raise FeatureLibError(
                    'Unknown valueRecordDef "%s"' % name, self.cur_token_location_
                )
            value = vrd.value
            xPlacement, yPlacement = (value.xPlacement, value.yPlacement)
            xAdvance, yAdvance = (value.xAdvance, value.yAdvance)
        else:
            xPlacement, yPlacement, xAdvance, yAdvance = (
                self.expect_number_(variable=True),
                self.expect_number_(variable=True),
                self.expect_number_(variable=True),
                self.expect_number_(variable=True),
            )

        if self.next_token_ == "<":
            xPlaDevice, yPlaDevice, xAdvDevice, yAdvDevice = (
                self.parse_device_(),
                self.parse_device_(),
                self.parse_device_(),
                self.parse_device_(),
            )
            allDeltas = sorted(
                [
                    delta
                    for size, delta in (xPlaDevice if xPlaDevice else ())
                    + (yPlaDevice if yPlaDevice else ())
                    + (xAdvDevice if xAdvDevice else ())
                    + (yAdvDevice if yAdvDevice else ())
                ]
            )
            if allDeltas[0] < -128 or allDeltas[-1] > 127:
                raise FeatureLibError(
                    "Device value out of valid range (-128..127)",
                    self.cur_token_location_,
                )
        else:
            xPlaDevice, yPlaDevice, xAdvDevice, yAdvDevice = (None, None, None, None)

        self.expect_symbol_(">")
        return self.ast.ValueRecord(
            xPlacement,
            yPlacement,
            xAdvance,
            yAdvance,
            xPlaDevice,
            yPlaDevice,
            xAdvDevice,
            yAdvDevice,
            vertical=vertical,
            location=location,
        )

    def parse_valuerecord_definition_(self, vertical):
        # Parses a named value record definition. (See section `2.e.v <https://adobe-type-tools.github.io/afdko/OpenTypeFeatureFileSpecification.html#2.e.v>`_)
        assert self.is_cur_keyword_("valueRecordDef")
        location = self.cur_token_location_
        value = self.parse_valuerecord_(vertical)
        name = self.expect_name_()
        self.expect_symbol_(";")
        vrd = self.ast.ValueRecordDefinition(name, value, location=location)
        self.valuerecords_.define(name, vrd)
        return vrd

    def parse_languagesystem_(self):
        assert self.cur_token_ == "languagesystem"
        location = self.cur_token_location_
        script = self.expect_script_tag_()
        language = self.expect_language_tag_()
        self.expect_symbol_(";")
        return self.ast.LanguageSystemStatement(script, language, location=location)

    def parse_feature_block_(self, variation=False):
        if variation:
            assert self.cur_token_ == "variation"
        else:
            assert self.cur_token_ == "feature"
        location = self.cur_token_location_
        tag = self.expect_tag_()
        vertical = tag in {"vkrn", "vpal", "vhal", "valt"}

        stylisticset = None
        cv_feature = None
        size_feature = False
        if tag in self.SS_FEATURE_TAGS:
            stylisticset = tag
        elif tag in self.CV_FEATURE_TAGS:
            cv_feature = tag
        elif tag == "size":
            size_feature = True

        if variation:
            conditionset = self.expect_name_()

        use_extension = False
        if self.next_token_ == "useExtension":
            self.expect_keyword_("useExtension")
            use_extension = True

        if variation:
            block = self.ast.VariationBlock(
                tag, conditionset, use_extension=use_extension, location=location
            )
        else:
            block = self.ast.FeatureBlock(
                tag, use_extension=use_extension, location=location
            )
        self.parse_block_(block, vertical, stylisticset, size_feature, cv_feature)
        return block

    def parse_feature_reference_(self):
        assert self.cur_token_ == "feature", self.cur_token_
        location = self.cur_token_location_
        featureName = self.expect_tag_()
        self.expect_symbol_(";")
        return self.ast.FeatureReferenceStatement(featureName, location=location)

    def parse_featureNames_(self, tag):
        """Parses a ``featureNames`` statement found in stylistic set features.
        See section `8.c <https://adobe-type-tools.github.io/afdko/OpenTypeFeatureFileSpecification.html#8.c>`_.
        """
        assert self.cur_token_ == "featureNames", self.cur_token_
        block = self.ast.NestedBlock(
            tag, self.cur_token_, location=self.cur_token_location_
        )
        self.expect_symbol_("{")
        for symtab in self.symbol_tables_:
            symtab.enter_scope()
        while self.next_token_ != "}" or self.cur_comments_:
            self.advance_lexer_(comments=True)
            if self.cur_token_type_ is Lexer.COMMENT:
                block.statements.append(
                    self.ast.Comment(self.cur_token_, location=self.cur_token_location_)
                )
            elif self.is_cur_keyword_("name"):
                location = self.cur_token_location_
                platformID, platEncID, langID, string = self.parse_name_()
                block.statements.append(
                    self.ast.FeatureNameStatement(
                        tag, platformID, platEncID, langID, string, location=location
                    )
                )
            elif self.cur_token_ == ";":
                continue
            else:
                raise FeatureLibError('Expected "name"', self.cur_token_location_)
        self.expect_symbol_("}")
        for symtab in self.symbol_tables_:
            symtab.exit_scope()
        self.expect_symbol_(";")
        return block

    def parse_cvParameters_(self, tag):
        # Parses a ``cvParameters`` block found in Character Variant features.
        # See section `8.d <https://adobe-type-tools.github.io/afdko/OpenTypeFeatureFileSpecification.html#8.d>`_.
        assert self.cur_token_ == "cvParameters", self.cur_token_
        block = self.ast.NestedBlock(
            tag, self.cur_token_, location=self.cur_token_location_
        )
        self.expect_symbol_("{")
        for symtab in self.symbol_tables_:
            symtab.enter_scope()

        statements = block.statements
        while self.next_token_ != "}" or self.cur_comments_:
            self.advance_lexer_(comments=True)
            if self.cur_token_type_ is Lexer.COMMENT:
                statements.append(
                    self.ast.Comment(self.cur_token_, location=self.cur_token_location_)
                )
            elif self.is_cur_keyword_(
                {
                    "FeatUILabelNameID",
                    "FeatUITooltipTextNameID",
                    "SampleTextNameID",
                    "ParamUILabelNameID",
                }
            ):
                statements.append(self.parse_cvNameIDs_(tag, self.cur_token_))
            elif self.is_cur_keyword_("Character"):
                statements.append(self.parse_cvCharacter_(tag))
            elif self.cur_token_ == ";":
                continue
            else:
                raise FeatureLibError(
                    "Expected statement: got {} {}".format(
                        self.cur_token_type_, self.cur_token_
                    ),
                    self.cur_token_location_,
                )

        self.expect_symbol_("}")
        for symtab in self.symbol_tables_:
            symtab.exit_scope()
        self.expect_symbol_(";")
        return block

    def parse_cvNameIDs_(self, tag, block_name):
        assert self.cur_token_ == block_name, self.cur_token_
        block = self.ast.NestedBlock(tag, block_name, location=self.cur_token_location_)
        self.expect_symbol_("{")
        for symtab in self.symbol_tables_:
            symtab.enter_scope()
        while self.next_token_ != "}" or self.cur_comments_:
            self.advance_lexer_(comments=True)
            if self.cur_token_type_ is Lexer.COMMENT:
                block.statements.append(
                    self.ast.Comment(self.cur_token_, location=self.cur_token_location_)
                )
            elif self.is_cur_keyword_("name"):
                location = self.cur_token_location_
                platformID, platEncID, langID, string = self.parse_name_()
                block.statements.append(
                    self.ast.CVParametersNameStatement(
                        tag,
                        platformID,
                        platEncID,
                        langID,
                        string,
                        block_name,
                        location=location,
                    )
                )
            elif self.cur_token_ == ";":
                continue
            else:
                raise FeatureLibError('Expected "name"', self.cur_token_location_)
        self.expect_symbol_("}")
        for symtab in self.symbol_tables_:
            symtab.exit_scope()
        self.expect_symbol_(";")
        return block

    def parse_cvCharacter_(self, tag):
        assert self.cur_token_ == "Character", self.cur_token_
        location, character = self.cur_token_location_, self.expect_any_number_()
        self.expect_symbol_(";")
        if not (0xFFFFFF >= character >= 0):
            raise FeatureLibError(
                "Character value must be between "
                "{:#x} and {:#x}".format(0, 0xFFFFFF),
                location,
            )
        return self.ast.CharacterStatement(character, tag, location=location)

    def parse_FontRevision_(self):
        # Parses a ``FontRevision`` statement found in the head table. See
        # `section 9.c <https://adobe-type-tools.github.io/afdko/OpenTypeFeatureFileSpecification.html#9.c>`_.
        assert self.cur_token_ == "FontRevision", self.cur_token_
        location, version = self.cur_token_location_, self.expect_float_()
        self.expect_symbol_(";")
        if version <= 0:
            raise FeatureLibError("Font revision numbers must be positive", location)
        return self.ast.FontRevisionStatement(version, location=location)

    def parse_conditionset_(self):
        location = self.cur_token_location_
        name = self.expect_name_()

        conditions = {}
        self.expect_symbol_("{")

        while self.next_token_ != "}":
            self.advance_lexer_()
            if self.cur_token_type_ is not Lexer.NAME:
                raise FeatureLibError("Expected an axis name", self.cur_token_location_)

            axis = self.cur_token_
            if axis in conditions:
                raise FeatureLibError(
                    f"Repeated condition for axis {axis}", self.cur_token_location_
                )

            if self.next_token_type_ is Lexer.FLOAT:
                min_value = self.expect_float_()
            elif self.next_token_type_ is Lexer.NUMBER:
                min_value = self.expect_number_(variable=False)

            if self.next_token_type_ is Lexer.FLOAT:
                max_value = self.expect_float_()
            elif self.next_token_type_ is Lexer.NUMBER:
                max_value = self.expect_number_(variable=False)
            self.expect_symbol_(";")

            conditions[axis] = (min_value, max_value)

        self.expect_symbol_("}")

        finalname = self.expect_name_()
        if finalname != name:
            raise FeatureLibError('Expected "%s"' % name, self.cur_token_location_)
        return self.ast.ConditionsetStatement(name, conditions, location=location)

    def parse_block_(
        self, block, vertical, stylisticset=None, size_feature=False, cv_feature=None
    ):
        self.expect_symbol_("{")
        for symtab in self.symbol_tables_:
            symtab.enter_scope()

        statements = block.statements
        while self.next_token_ != "}" or self.cur_comments_:
            self.advance_lexer_(comments=True)
            if self.cur_token_type_ is Lexer.COMMENT:
                statements.append(
                    self.ast.Comment(self.cur_token_, location=self.cur_token_location_)
                )
            elif self.cur_token_type_ is Lexer.GLYPHCLASS:
                statements.append(self.parse_glyphclass_definition_())
            elif self.is_cur_keyword_("anchorDef"):
                statements.append(self.parse_anchordef_())
            elif self.is_cur_keyword_({"enum", "enumerate"}):
                statements.append(self.parse_enumerate_(vertical=vertical))
            elif self.is_cur_keyword_("feature"):
                statements.append(self.parse_feature_reference_())
            elif self.is_cur_keyword_("ignore"):
                statements.append(self.parse_ignore_())
            elif self.is_cur_keyword_("language"):
                statements.append(self.parse_language_())
            elif self.is_cur_keyword_("lookup"):
                statements.append(self.parse_lookup_(vertical))
            elif self.is_cur_keyword_("lookupflag"):
                statements.append(self.parse_lookupflag_())
            elif self.is_cur_keyword_("markClass"):
                statements.append(self.parse_markClass_())
            elif self.is_cur_keyword_({"pos", "position"}):
                statements.append(
                    self.parse_position_(enumerated=False, vertical=vertical)
                )
            elif self.is_cur_keyword_("script"):
                statements.append(self.parse_script_())
            elif self.is_cur_keyword_({"sub", "substitute", "rsub", "reversesub"}):
                statements.append(self.parse_substitute_())
            elif self.is_cur_keyword_("subtable"):
                statements.append(self.parse_subtable_())
            elif self.is_cur_keyword_("valueRecordDef"):
                statements.append(self.parse_valuerecord_definition_(vertical))
            elif stylisticset and self.is_cur_keyword_("featureNames"):
                statements.append(self.parse_featureNames_(stylisticset))
            elif cv_feature and self.is_cur_keyword_("cvParameters"):
                statements.append(self.parse_cvParameters_(cv_feature))
            elif size_feature and self.is_cur_keyword_("parameters"):
                statements.append(self.parse_size_parameters_())
            elif size_feature and self.is_cur_keyword_("sizemenuname"):
                statements.append(self.parse_size_menuname_())
            elif (
                self.cur_token_type_ is Lexer.NAME
                and self.cur_token_ in self.extensions
            ):
                statements.append(self.extensions[self.cur_token_](self))
            elif self.cur_token_ == ";":
                continue
            else:
                raise FeatureLibError(
                    "Expected glyph class definition or statement: got {} {}".format(
                        self.cur_token_type_, self.cur_token_
                    ),
                    self.cur_token_location_,
                )

        self.expect_symbol_("}")
        for symtab in self.symbol_tables_:
            symtab.exit_scope()

        name = self.expect_name_()
        if name != block.name.strip():
            raise FeatureLibError(
                'Expected "%s"' % block.name.strip(), self.cur_token_location_
            )
        self.expect_symbol_(";")

    def is_cur_keyword_(self, k):
        if self.cur_token_type_ is Lexer.NAME:
            if isinstance(k, type("")):  # basestring is gone in Python3
                return self.cur_token_ == k
            else:
                return self.cur_token_ in k
        return False

    def expect_class_name_(self):
        self.advance_lexer_()
        if self.cur_token_type_ is not Lexer.GLYPHCLASS:
            raise FeatureLibError("Expected @NAME", self.cur_token_location_)
        return self.cur_token_

    def expect_cid_(self):
        self.advance_lexer_()
        if self.cur_token_type_ is Lexer.CID:
            return self.cur_token_
        raise FeatureLibError("Expected a CID", self.cur_token_location_)

    def expect_filename_(self):
        self.advance_lexer_()
        if self.cur_token_type_ is not Lexer.FILENAME:
            raise FeatureLibError("Expected file name", self.cur_token_location_)
        return self.cur_token_

    def expect_glyph_(self):
        self.advance_lexer_()
        if self.cur_token_type_ is Lexer.NAME:
            return self.cur_token_.lstrip("\\")
        elif self.cur_token_type_ is Lexer.CID:
            return "cid%05d" % self.cur_token_
        raise FeatureLibError("Expected a glyph name or CID", self.cur_token_location_)

    def check_glyph_name_in_glyph_set(self, *names):
        """Adds a glyph name (just `start`) or glyph names of a
        range (`start` and `end`) which are not in the glyph set
        to the "missing list" for future error reporting.

        If no glyph set is present, does nothing.
        """
        if self.glyphNames_:
            for name in names:
                if name in self.glyphNames_:
                    continue
                if name not in self.missing:
                    self.missing[name] = self.cur_token_location_

    def expect_markClass_reference_(self):
        name = self.expect_class_name_()
        mc = self.glyphclasses_.resolve(name)
        if mc is None:
            raise FeatureLibError(
                "Unknown markClass @%s" % name, self.cur_token_location_
            )
        if not isinstance(mc, self.ast.MarkClass):
            raise FeatureLibError(
                "@%s is not a markClass" % name, self.cur_token_location_
            )
        return mc

    def expect_tag_(self):
        self.advance_lexer_()
        if self.cur_token_type_ is not Lexer.NAME:
            raise FeatureLibError("Expected a tag", self.cur_token_location_)
        if len(self.cur_token_) > 4:
            raise FeatureLibError(
                "Tags cannot be longer than 4 characters", self.cur_token_location_
            )
        return (self.cur_token_ + "    ")[:4]

    def expect_script_tag_(self):
        tag = self.expect_tag_()
        if tag == "dflt":
            raise FeatureLibError(
                '"dflt" is not a valid script tag; use "DFLT" instead',
                self.cur_token_location_,
            )
        return tag

    def expect_language_tag_(self):
        tag = self.expect_tag_()
        if tag == "DFLT":
            raise FeatureLibError(
                '"DFLT" is not a valid language tag; use "dflt" instead',
                self.cur_token_location_,
            )
        return tag

    def expect_symbol_(self, symbol):
        self.advance_lexer_()
        if self.cur_token_type_ is Lexer.SYMBOL and self.cur_token_ == symbol:
            return symbol
        raise FeatureLibError("Expected '%s'" % symbol, self.cur_token_location_)

    def expect_keyword_(self, keyword):
        self.advance_lexer_()
        if self.cur_token_type_ is Lexer.NAME and self.cur_token_ == keyword:
            return self.cur_token_
        raise FeatureLibError('Expected "%s"' % keyword, self.cur_token_location_)

    def expect_name_(self):
        self.advance_lexer_()
        if self.cur_token_type_ is Lexer.NAME:
            return self.cur_token_
        raise FeatureLibError("Expected a name", self.cur_token_location_)

    def expect_number_(self, variable=False):
        self.advance_lexer_()
        if self.cur_token_type_ is Lexer.NUMBER:
            return self.cur_token_
        if variable and self.cur_token_type_ is Lexer.SYMBOL and self.cur_token_ == "(":
            return self.expect_variable_scalar_()
        raise FeatureLibError("Expected a number", self.cur_token_location_)

    def expect_variable_scalar_(self):
        self.advance_lexer_()  # "("
        scalar = VariableScalar()
        default_seen = False
        while True:
            if self.cur_token_type_ == Lexer.SYMBOL and self.cur_token_ == ")":
                break
            if self.cur_token_type_ == Lexer.NUMBER:
                if default_seen:
                    raise FeatureLibError(
                        "Duplicate value for the default location",
                        self.cur_token_location_,
                    )
                default_seen = True
                location, value = {}, self.cur_token_
                self.advance_lexer_()
            else:
                location, value = self.expect_master_()
            scalar.add_value(location, value)
        return scalar

    def expect_master_(self):
        location = {}
        while True:
            if self.cur_token_type_ is not Lexer.NAME:
                raise FeatureLibError("Expected an axis name", self.cur_token_location_)
            axis = self.cur_token_
            self.advance_lexer_()
            if not (self.cur_token_type_ is Lexer.SYMBOL and self.cur_token_ == "="):
                raise FeatureLibError(
                    "Expected an equals sign", self.cur_token_location_
                )
            value = self.expect_integer_or_float_()
            location[axis] = value
            if self.next_token_type_ is Lexer.NAME and self.next_token_[0] == ":":
                # Lexer has just read the value as a glyph name. We'll correct it later
                break
            self.advance_lexer_()
            if not (self.cur_token_type_ is Lexer.SYMBOL and self.cur_token_ == ","):
                raise FeatureLibError(
                    "Expected an comma or an equals sign", self.cur_token_location_
                )
            self.advance_lexer_()
        self.advance_lexer_()
        value = int(self.cur_token_[1:])
        self.advance_lexer_()
        return location, value

    def expect_any_number_(self):
        self.advance_lexer_()
        if self.cur_token_type_ in Lexer.NUMBERS:
            return self.cur_token_
        raise FeatureLibError(
            "Expected a decimal, hexadecimal or octal number", self.cur_token_location_
        )

    def expect_float_(self):
        self.advance_lexer_()
        if self.cur_token_type_ is Lexer.FLOAT:
            return self.cur_token_
        raise FeatureLibError(
            "Expected a floating-point number", self.cur_token_location_
        )

    def expect_integer_or_float_(self):
        if self.next_token_type_ == Lexer.FLOAT:
            return self.expect_float_()
        elif self.next_token_type_ is Lexer.NUMBER:
            return self.expect_number_()
        else:
            raise FeatureLibError(
                "Expected an integer or floating-point number", self.cur_token_location_
            )

    def expect_decipoint_(self):
        if self.next_token_type_ == Lexer.FLOAT:
            return self.expect_float_()
        elif self.next_token_type_ is Lexer.NUMBER:
            return self.expect_number_() / 10
        else:
            raise FeatureLibError(
                "Expected an integer or floating-point number", self.cur_token_location_
            )

    def expect_stat_flags(self):
        value = 0
        flags = {
            "OlderSiblingFontAttribute": 1,
            "ElidableAxisValueName": 2,
        }
        while self.next_token_ != ";":
            if self.next_token_ in flags:
                name = self.expect_name_()
                value = value | flags[name]
            else:
                raise FeatureLibError(
                    f"Unexpected STAT flag {self.cur_token_}", self.cur_token_location_
                )
        return value

    def expect_stat_values_(self):
        if self.next_token_type_ == Lexer.FLOAT:
            return self.expect_float_()
        elif self.next_token_type_ is Lexer.NUMBER:
            return self.expect_number_()
        else:
            raise FeatureLibError(
                "Expected an integer or floating-point number", self.cur_token_location_
            )

    def expect_string_(self):
        self.advance_lexer_()
        if self.cur_token_type_ is Lexer.STRING:
            return self.cur_token_
        raise FeatureLibError("Expected a string", self.cur_token_location_)

    def advance_lexer_(self, comments=False):
        if comments and self.cur_comments_:
            self.cur_token_type_ = Lexer.COMMENT
            self.cur_token_, self.cur_token_location_ = self.cur_comments_.pop(0)
            return
        else:
            self.cur_token_type_, self.cur_token_, self.cur_token_location_ = (
                self.next_token_type_,
                self.next_token_,
                self.next_token_location_,
            )
        while True:
            try:
                (
                    self.next_token_type_,
                    self.next_token_,
                    self.next_token_location_,
                ) = next(self.lexer_)
            except StopIteration:
                self.next_token_type_, self.next_token_ = (None, None)
            if self.next_token_type_ != Lexer.COMMENT:
                break
            self.cur_comments_.append((self.next_token_, self.next_token_location_))

    @staticmethod
    def reverse_string_(s):
        """'abc' --> 'cba'"""
        return "".join(reversed(list(s)))

    def make_cid_range_(self, location, start, limit):
        """(location, 999, 1001) --> ["cid00999", "cid01000", "cid01001"]"""
        result = list()
        if start > limit:
            raise FeatureLibError(
                "Bad range: start should be less than limit", location
            )
        for cid in range(start, limit + 1):
            result.append("cid%05d" % cid)
        return result

    def make_glyph_range_(self, location, start, limit):
        """(location, "a.sc", "d.sc") --> ["a.sc", "b.sc", "c.sc", "d.sc"]"""
        result = list()
        if len(start) != len(limit):
            raise FeatureLibError(
                'Bad range: "%s" and "%s" should have the same length' % (start, limit),
                location,
            )

        rev = self.reverse_string_
        prefix = os.path.commonprefix([start, limit])
        suffix = rev(os.path.commonprefix([rev(start), rev(limit)]))
        if len(suffix) > 0:
            start_range = start[len(prefix) : -len(suffix)]
            limit_range = limit[len(prefix) : -len(suffix)]
        else:
            start_range = start[len(prefix) :]
            limit_range = limit[len(prefix) :]

        if start_range >= limit_range:
            raise FeatureLibError(
                "Start of range must be smaller than its end", location
            )

        uppercase = re.compile(r"^[A-Z]$")
        if uppercase.match(start_range) and uppercase.match(limit_range):
            for c in range(ord(start_range), ord(limit_range) + 1):
                result.append("%s%c%s" % (prefix, c, suffix))
            return result

        lowercase = re.compile(r"^[a-z]$")
        if lowercase.match(start_range) and lowercase.match(limit_range):
            for c in range(ord(start_range), ord(limit_range) + 1):
                result.append("%s%c%s" % (prefix, c, suffix))
            return result

        digits = re.compile(r"^[0-9]{1,3}$")
        if digits.match(start_range) and digits.match(limit_range):
            for i in range(int(start_range, 10), int(limit_range, 10) + 1):
                number = ("000" + str(i))[-len(start_range) :]
                result.append("%s%s%s" % (prefix, number, suffix))
            return result

        raise FeatureLibError('Bad range: "%s-%s"' % (start, limit), location)


class SymbolTable(object):
    def __init__(self):
        self.scopes_ = [{}]

    def enter_scope(self):
        self.scopes_.append({})

    def exit_scope(self):
        self.scopes_.pop()

    def define(self, name, item):
        self.scopes_[-1][name] = item

    def resolve(self, name):
        for scope in reversed(self.scopes_):
            item = scope.get(name)
            if item:
                return item
        return None
<|endoftext|>from fontTools.ttLib import TTFont
from fontTools.feaLib.builder import addOpenTypeFeatures, Builder
from fontTools.feaLib.error import FeatureLibError
from fontTools import configLogger
from fontTools.misc.cliTools import makeOutputFileName
import sys
import argparse
import logging

log = logging.getLogger("fontTools.feaLib")


def main(args=None):
    """Add features from a feature file (.fea) into an OTF font"""
    parser = argparse.ArgumentParser(
        description="Use fontTools to compile OpenType feature files (*.fea)."
    )
    parser.add_argument(
        "input_fea", metavar="FEATURES", help="Path to the feature file"
    )
    parser.add_argument(
        "input_font", metavar="INPUT_FONT", help="Path to the input font"
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_font",
        metavar="OUTPUT_FONT",
        help="Path to the output font.",
    )
    parser.add_argument(
        "-t",
        "--tables",
        metavar="TABLE_TAG",
        choices=Builder.supportedTables,
        nargs="+",
        help="Specify the table(s) to be built.",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Add source-level debugging information to font.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Increase the logger verbosity. Multiple -v " "options are allowed.",
        action="count",
        default=0,
    )
    parser.add_argument(
        "--traceback", help="show traceback for exceptions.", action="store_true"
    )
    options = parser.parse_args(args)

    levels = ["WARNING", "INFO", "DEBUG"]
    configLogger(level=levels[min(len(levels) - 1, options.verbose)])

    output_font = options.output_font or makeOutputFileName(options.input_font)
    log.info("Compiling features to '%s'" % (output_font))

    font = TTFont(options.input_font)
    try:
        addOpenTypeFeatures(
            font, options.input_fea, tables=options.tables, debug=options.debug
        )
    except FeatureLibError as e:
        if options.traceback:
            raise
        log.error(e)
        sys.exit(1)
    font.save(output_font)


if __name__ == "__main__":
    sys.exit(main())
<|endoftext|>from fontTools import ttLib
from fontTools.ttLib.tables import otTables as ot

# VariationStore


def buildVarRegionAxis(axisSupport):
    self = ot.VarRegionAxis()
    self.StartCoord, self.PeakCoord, self.EndCoord = [float(v) for v in axisSupport]
    return self


def buildSparseVarRegionAxis(axisIndex, axisSupport):
    self = ot.SparseVarRegionAxis()
    self.AxisIndex = axisIndex
    self.StartCoord, self.PeakCoord, self.EndCoord = [float(v) for v in axisSupport]
    return self


def buildVarRegion(support, axisTags):
    assert all(tag in axisTags for tag in support.keys()), (
        "Unknown axis tag found.",
        support,
        axisTags,
    )
    self = ot.VarRegion()
    self.VarRegionAxis = []
    for tag in axisTags:
        self.VarRegionAxis.append(buildVarRegionAxis(support.get(tag, (0, 0, 0))))
    return self


def buildSparseVarRegion(support, axisTags):
    assert all(tag in axisTags for tag in support.keys()), (
        "Unknown axis tag found.",
        support,
        axisTags,
    )
    self = ot.SparseVarRegion()
    self.SparseVarRegionAxis = []
    for i, tag in enumerate(axisTags):
        if tag not in support:
            continue
        self.SparseVarRegionAxis.append(
            buildSparseVarRegionAxis(i, support.get(tag, (0, 0, 0)))
        )
    self.SparseRegionCount = len(self.SparseVarRegionAxis)
    return self


def buildVarRegionList(supports, axisTags):
    self = ot.VarRegionList()
    self.RegionAxisCount = len(axisTags)
    self.Region = []
    for support in supports:
        self.Region.append(buildVarRegion(support, axisTags))
    self.RegionCount = len(self.Region)
    return self


def buildSparseVarRegionList(supports, axisTags):
    self = ot.SparseVarRegionList()
    self.RegionAxisCount = len(axisTags)
    self.Region = []
    for support in supports:
        self.Region.append(buildSparseVarRegion(support, axisTags))
    self.RegionCount = len(self.Region)
    return self


def _reorderItem(lst, mapping):
    return [lst[i] for i in mapping]


def VarData_calculateNumShorts(self, optimize=False):
    count = self.VarRegionCount
    items = self.Item
    bit_lengths = [0] * count
    for item in items:
        # The "+ (i < -1)" magic is to handle two's-compliment.
        # That is, we want to get back 7 for -128, whereas
        # bit_length() returns 8. Similarly for -65536.
        # The reason "i < -1" is used instead of "i < 0" is that
        # the latter would make it return 0 for "-1" instead of 1.
        bl = [(i + (i < -1)).bit_length() for i in item]
        bit_lengths = [max(*pair) for pair in zip(bl, bit_lengths)]
    # The addition of 8, instead of seven, is to account for the sign bit.
    # This "((b + 8) >> 3) if b else 0" when combined with the above
    # "(i + (i < -1)).bit_length()" is a faster way to compute byte-lengths
    # conforming to:
    #
    # byte_length = (0 if i == 0 else
    # 		 1 if -128 <= i < 128 else
    # 		 2 if -65536 <= i < 65536 else
    # 		 ...)
    byte_lengths = [((b + 8) >> 3) if b else 0 for b in bit_lengths]

    # https://github.com/fonttools/fonttools/issues/2279
    longWords = any(b > 2 for b in byte_lengths)

    if optimize:
        # Reorder columns such that wider columns come before narrower columns
        mapping = []
        mapping.extend(i for i, b in enumerate(byte_lengths) if b > 2)
        mapping.extend(i for i, b in enumerate(byte_lengths) if b == 2)
        mapping.extend(i for i, b in enumerate(byte_lengths) if b == 1)

        byte_lengths = _reorderItem(byte_lengths, mapping)
        self.VarRegionIndex = _reorderItem(self.VarRegionIndex, mapping)
        self.VarRegionCount = len(self.VarRegionIndex)
        for i in range(len(items)):
            items[i] = _reorderItem(items[i], mapping)

    if longWords:
        self.NumShorts = (
            max((i for i, b in enumerate(byte_lengths) if b > 2), default=-1) + 1
        )
        self.NumShorts |= 0x8000
    else:
        self.NumShorts = (
            max((i for i, b in enumerate(byte_lengths) if b > 1), default=-1) + 1
        )

    self.VarRegionCount = len(self.VarRegionIndex)
    return self


ot.VarData.calculateNumShorts = VarData_calculateNumShorts


def VarData_CalculateNumShorts(self, optimize=True):
    """Deprecated name for VarData_calculateNumShorts() which
    defaults to optimize=True.  Use varData.calculateNumShorts()
    or varData.optimize()."""
    return VarData_calculateNumShorts(self, optimize=optimize)


def VarData_optimize(self):
    return VarData_calculateNumShorts(self, optimize=True)


ot.VarData.optimize = VarData_optimize


def buildVarData(varRegionIndices, items, optimize=True):
    self = ot.VarData()
    self.VarRegionIndex = list(varRegionIndices)
    regionCount = self.VarRegionCount = len(self.VarRegionIndex)
    records = self.Item = []
    if items:
        for item in items:
            assert len(item) == regionCount
            records.append(list(item))
    self.ItemCount = len(self.Item)
    self.calculateNumShorts(optimize=optimize)
    return self


def buildVarStore(varRegionList, varDataList):
    self = ot.VarStore()
    self.Format = 1
    self.VarRegionList = varRegionList
    self.VarData = list(varDataList)
    self.VarDataCount = len(self.VarData)
    return self


def buildMultiVarData(varRegionIndices, items):
    self = ot.MultiVarData()
    self.Format = 1
    self.VarRegionIndex = list(varRegionIndices)
    regionCount = self.VarRegionCount = len(self.VarRegionIndex)
    records = self.Item = []
    if items:
        for item in items:
            assert len(item) == regionCount
            records.append(list(item))
    self.ItemCount = len(self.Item)
    return self


def buildMultiVarStore(varRegionList, multiVarDataList):
    self = ot.MultiVarStore()
    self.Format = 1
    self.SparseVarRegionList = varRegionList
    self.MultiVarData = list(multiVarDataList)
    self.MultiVarDataCount = len(self.MultiVarData)
    return self


# Variation helpers


def buildVarIdxMap(varIdxes, glyphOrder):
    self = ot.VarIdxMap()
    self.mapping = {g: v for g, v in zip(glyphOrder, varIdxes)}
    return self


def buildDeltaSetIndexMap(varIdxes):
    mapping = list(varIdxes)
    if all(i == v for i, v in enumerate(mapping)):
        return None
    self = ot.DeltaSetIndexMap()
    self.mapping = mapping
    self.Format = 1 if len(mapping) > 0xFFFF else 0
    return self


def buildVarDevTable(varIdx):
    self = ot.Device()
    self.DeltaFormat = 0x8000
    self.StartSize = varIdx >> 16
    self.EndSize = varIdx & 0xFFFF
    return self
<|endoftext|>from fontTools.misc.roundTools import noRound, otRound
from fontTools.misc.intTools import bit_count
from fontTools.misc.vector import Vector
from fontTools.ttLib.tables import otTables as ot
from fontTools.varLib.models import supportScalar
import fontTools.varLib.varStore  # For monkey-patching
from fontTools.varLib.builder import (
    buildVarRegionList,
    buildSparseVarRegionList,
    buildSparseVarRegion,
    buildMultiVarStore,
    buildMultiVarData,
)
from fontTools.misc.iterTools import batched
from functools import partial
from collections import defaultdict
from heapq import heappush, heappop

NO_VARIATION_INDEX = ot.NO_VARIATION_INDEX
ot.MultiVarStore.NO_VARIATION_INDEX = NO_VARIATION_INDEX


def _getLocationKey(loc):
    return tuple(sorted(loc.items(), key=lambda kv: kv[0]))


class OnlineMultiVarStoreBuilder(object):
    def __init__(self, axisTags):
        self._axisTags = axisTags
        self._regionMap = {}
        self._regionList = buildSparseVarRegionList([], axisTags)
        self._store = buildMultiVarStore(self._regionList, [])
        self._data = None
        self._model = None
        self._supports = None
        self._varDataIndices = {}
        self._varDataCaches = {}
        self._cache = None

    def setModel(self, model):
        self.setSupports(model.supports)
        self._model = model

    def setSupports(self, supports):
        self._model = None
        self._supports = list(supports)
        if not self._supports[0]:
            del self._supports[0]  # Drop base master support
        self._cache = None
        self._data = None

    def finish(self):
        self._regionList.RegionCount = len(self._regionList.Region)
        self._store.MultiVarDataCount = len(self._store.MultiVarData)
        return self._store

    def _add_MultiVarData(self):
        regionMap = self._regionMap
        regionList = self._regionList

        regions = self._supports
        regionIndices = []
        for region in regions:
            key = _getLocationKey(region)
            idx = regionMap.get(key)
            if idx is None:
                varRegion = buildSparseVarRegion(region, self._axisTags)
                idx = regionMap[key] = len(regionList.Region)
                regionList.Region.append(varRegion)
            regionIndices.append(idx)

        # Check if we have one already...
        key = tuple(regionIndices)
        varDataIdx = self._varDataIndices.get(key)
        if varDataIdx is not None:
            self._outer = varDataIdx
            self._data = self._store.MultiVarData[varDataIdx]
            self._cache = self._varDataCaches[key]
            if len(self._data.Item) == 0xFFFF:
                # This is full.  Need new one.
                varDataIdx = None

        if varDataIdx is None:
            self._data = buildMultiVarData(regionIndices, [])
            self._outer = len(self._store.MultiVarData)
            self._store.MultiVarData.append(self._data)
            self._varDataIndices[key] = self._outer
            if key not in self._varDataCaches:
                self._varDataCaches[key] = {}
            self._cache = self._varDataCaches[key]

    def storeMasters(self, master_values, *, round=round):
        deltas = self._model.getDeltas(master_values, round=round)
        base = deltas.pop(0)
        return base, self.storeDeltas(deltas, round=noRound)

    def storeDeltas(self, deltas, *, round=round):
        deltas = tuple(round(d) for d in deltas)

        if not any(deltas):
            return NO_VARIATION_INDEX

        deltas_tuple = tuple(tuple(d) for d in deltas)

        if not self._data:
            self._add_MultiVarData()

        varIdx = self._cache.get(deltas_tuple)
        if varIdx is not None:
            return varIdx

        inner = len(self._data.Item)
        if inner == 0xFFFF:
            # Full array. Start new one.
            self._add_MultiVarData()
            return self.storeDeltas(deltas, round=noRound)
        self._data.addItem(deltas, round=noRound)

        varIdx = (self._outer << 16) + inner
        self._cache[deltas_tuple] = varIdx
        return varIdx


def MultiVarData_addItem(self, deltas, *, round=round):
    deltas = tuple(round(d) for d in deltas)

    assert len(deltas) == self.VarRegionCount

    values = []
    for d in deltas:
        values.extend(d)

    self.Item.append(values)
    self.ItemCount = len(self.Item)


ot.MultiVarData.addItem = MultiVarData_addItem


def SparseVarRegion_get_support(self, fvar_axes):
    return {
        fvar_axes[reg.AxisIndex].axisTag: (reg.StartCoord, reg.PeakCoord, reg.EndCoord)
        for reg in self.SparseVarRegionAxis
    }


ot.SparseVarRegion.get_support = SparseVarRegion_get_support


def MultiVarStore___bool__(self):
    return bool(self.MultiVarData)


ot.MultiVarStore.__bool__ = MultiVarStore___bool__


class MultiVarStoreInstancer(object):
    def __init__(self, multivarstore, fvar_axes, location={}):
        self.fvar_axes = fvar_axes
        assert multivarstore is None or multivarstore.Format == 1
        self._varData = multivarstore.MultiVarData if multivarstore else []
        self._regions = (
            multivarstore.SparseVarRegionList.Region if multivarstore else []
        )
        self.setLocation(location)

    def setLocation(self, location):
        self.location = dict(location)
        self._clearCaches()

    def _clearCaches(self):
        self._scalars = {}

    def _getScalar(self, regionIdx):
        scalar = self._scalars.get(regionIdx)
        if scalar is None:
            support = self._regions[regionIdx].get_support(self.fvar_axes)
            scalar = supportScalar(self.location, support)
            self._scalars[regionIdx] = scalar
        return scalar

    @staticmethod
    def interpolateFromDeltasAndScalars(deltas, scalars):
        if not deltas:
            return Vector([])
        assert len(deltas) % len(scalars) == 0, (len(deltas), len(scalars))
        m = len(deltas) // len(scalars)
        delta = Vector([0] * m)
        for d, s in zip(batched(deltas, m), scalars):
            if not s:
                continue
            delta += Vector(d) * s
        return delta

    def __getitem__(self, varidx):
        major, minor = varidx >> 16, varidx & 0xFFFF
        if varidx == NO_VARIATION_INDEX:
            return Vector([])
        varData = self._varData
        scalars = [self._getScalar(ri) for ri in varData[major].VarRegionIndex]
        deltas = varData[major].Item[minor]
        return self.interpolateFromDeltasAndScalars(deltas, scalars)

    def interpolateFromDeltas(self, varDataIndex, deltas):
        varData = self._varData
        scalars = [self._getScalar(ri) for ri in varData[varDataIndex].VarRegionIndex]
        return self.interpolateFromDeltasAndScalars(deltas, scalars)


def MultiVarStore_subset_varidxes(self, varIdxes):
    return ot.VarStore.subset_varidxes(self, varIdxes, VarData="MultiVarData")


def MultiVarStore_prune_regions(self):
    return ot.VarStore.prune_regions(
        self, VarData="MultiVarData", VarRegionList="SparseVarRegionList"
    )


ot.MultiVarStore.prune_regions = MultiVarStore_prune_regions
ot.MultiVarStore.subset_varidxes = MultiVarStore_subset_varidxes


def MultiVarStore_get_supports(self, major, fvarAxes):
    supports = []
    varData = self.MultiVarData[major]
    for regionIdx in varData.VarRegionIndex:
        region = self.SparseVarRegionList.Region[regionIdx]
        support = region.get_support(fvarAxes)
        supports.append(support)
    return supports


ot.MultiVarStore.get_supports = MultiVarStore_get_supports


def ConditionTable_collect_varidxes(self, varidxes):
    if self.Format == 2:
        varidxes.add(self.VarIdx)
    elif self.Format in (3, 4):
        for condition in self.ConditionTable:
            condition.collect_varidxes(varidxes)
    elif self.Format == 5:
        self.ConditionTable.collect_varidxes(varidxes)


def ConditionTable_remap_varidxes(self, varidxes_map):
    if self.Format == 2:
        self.VarIdx = varidxes_map[self.VarIdx]
    elif self.Format in (3, 4):
        for condition in self.ConditionTable:
            condition.remap_varidxes(varidxes_map)
    elif self.Format == 5:
        self.ConditionTable.remap_varidxes(varidxes_map)


ot.ConditionTable.collect_varidxes = ConditionTable_collect_varidxes
ot.ConditionTable.remap_varidxes = ConditionTable_remap_varidxes


def VARC_collect_varidxes(self, varidxes):
    for glyph in self.VarCompositeGlyphs.VarCompositeGlyph:
        for component in glyph.components:
            varidxes.add(component.axisValuesVarIndex)
            varidxes.add(component.transformVarIndex)
    if self.ConditionList is not None:
        for condition in self.ConditionList.ConditionTable:
            condition.collect_varidxes(varidxes)


def VARC_remap_varidxes(self, varidxes_map):
    for glyph in self.VarCompositeGlyphs.VarCompositeGlyph:
        for component in glyph.components:
            component.axisValuesVarIndex = varidxes_map[component.axisValuesVarIndex]
            component.transformVarIndex = varidxes_map[component.transformVarIndex]
    if self.ConditionList is not None:
        for condition in self.ConditionList.ConditionTable:
            condition.remap_varidxes(varidxes_map)


ot.VARC.collect_varidxes = VARC_collect_varidxes
ot.VARC.remap_varidxes = VARC_remap_varidxes
<|endoftext|>from fontTools.misc.roundTools import noRound
from fontTools.ttLib import TTFont, newTable
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.tables.otBase import OTTableWriter
from fontTools.varLib import HVAR_FIELDS, VVAR_FIELDS, _add_VHVAR
from fontTools.varLib import builder, models, varStore
from fontTools.misc.fixedTools import fixedToFloat as fi2fl
from fontTools.misc.cliTools import makeOutputFileName
from functools import partial
import logging

log = logging.getLogger("fontTools.varLib.avar")


def _get_advance_metrics(font, axisTags, tableFields):
    # There's two ways we can go from here:
    # 1. For each glyph, at each master peak, compute the value of the
    #    advance width at that peak.  Then pass these all to a VariationModel
    #    builder to compute back the deltas.
    # 2. For each master peak, pull out the deltas of the advance width directly,
    #    and feed these to the VarStoreBuilder, forgoing the remodeling step.
    # We'll go with the second option, as it's simpler, faster, and more direct.
    gvar = font["gvar"]
    vhAdvanceDeltasAndSupports = {}
    glyphOrder = font.getGlyphOrder()
    phantomIndex = tableFields.phantomIndex
    for glyphName in glyphOrder:
        supports = []
        deltas = []
        variations = gvar.variations.get(glyphName, [])

        for tv in variations:
            supports.append(tv.axes)
            phantoms = tv.coordinates[-4:]
            phantoms = phantoms[phantomIndex * 2 : phantomIndex * 2 + 2]
            assert len(phantoms) == 2
            phantoms[0] = phantoms[0][phantomIndex] if phantoms[0] is not None else 0
            phantoms[1] = phantoms[1][phantomIndex] if phantoms[1] is not None else 0
            deltas.append(phantoms[1] - phantoms[0])

        vhAdvanceDeltasAndSupports[glyphName] = (deltas, supports)

    vOrigDeltasAndSupports = None  # TODO

    return vhAdvanceDeltasAndSupports, vOrigDeltasAndSupports


def add_HVAR(font):
    if "HVAR" in font:
        del font["HVAR"]
    axisTags = [axis.axisTag for axis in font["fvar"].axes]
    getAdvanceMetrics = partial(_get_advance_metrics, font, axisTags, HVAR_FIELDS)
    _add_VHVAR(font, axisTags, HVAR_FIELDS, getAdvanceMetrics)


def add_VVAR(font):
    if "VVAR" in font:
        del font["VVAR"]
    getAdvanceMetrics = partial(_get_advance_metrics, font, axisTags, VVAR_FIELDS)
    axisTags = [axis.axisTag for axis in font["fvar"].axes]
    _add_VHVAR(font, axisTags, VVAR_FIELDS, getAdvanceMetrics)


def main(args=None):
    """Add `HVAR` table to variable font."""

    if args is None:
        import sys

        args = sys.argv[1:]

    from fontTools import configLogger
    from fontTools.designspaceLib import DesignSpaceDocument
    import argparse

    parser = argparse.ArgumentParser(
        "fonttools varLib.hvar",
        description="Add `HVAR` table from to variable font.",
    )
    parser.add_argument("font", metavar="varfont.ttf", help="Variable-font file.")
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        help="Output font file name.",
    )

    options = parser.parse_args(args)

    configLogger(level="WARNING")

    font = TTFont(options.font)
    if not "fvar" in font:
        log.error("Not a variable font.")
        return 1

    add_HVAR(font)
    if "vmtx" in font:
        add_VVAR(font)

    if options.output_file is None:
        outfile = makeOutputFileName(options.font, overWrite=True, suffix=".hvar")
    else:
        outfile = options.output_file
    if outfile:
        log.info("Saving %s", outfile)
        font.save(outfile)


if __name__ == "__main__":
    import sys

    sys.exit(main())
<|endoftext|>"""Extra methods for DesignSpaceDocument to generate its STAT table data."""

from __future__ import annotations

from typing import Dict, List, Union

import fontTools.otlLib.builder
from fontTools.designspaceLib import (
    AxisLabelDescriptor,
    DesignSpaceDocument,
    DesignSpaceDocumentError,
    LocationLabelDescriptor,
)
from fontTools.designspaceLib.types import Region, getVFUserRegion, locationInRegion
from fontTools.ttLib import TTFont


def buildVFStatTable(ttFont: TTFont, doc: DesignSpaceDocument, vfName: str) -> None:
    """Build the STAT table for the variable font identified by its name in
    the given document.

    Knowing which variable we're building STAT data for is needed to subset
    the STAT locations to only include what the variable font actually ships.

    .. versionadded:: 5.0

    .. seealso::
        - :func:`getStatAxes()`
        - :func:`getStatLocations()`
        - :func:`fontTools.otlLib.builder.buildStatTable()`
    """
    for vf in doc.getVariableFonts():
        if vf.name == vfName:
            break
    else:
        raise DesignSpaceDocumentError(
            f"Cannot find the variable font by name {vfName}"
        )

    region = getVFUserRegion(doc, vf)

    # if there are not currently any mac names don't add them here, that's inconsistent
    # https://github.com/fonttools/fonttools/issues/683
    macNames = any(
        nr.platformID == 1 for nr in getattr(ttFont.get("name"), "names", ())
    )

    return fontTools.otlLib.builder.buildStatTable(
        ttFont,
        getStatAxes(doc, region),
        getStatLocations(doc, region),
        doc.elidedFallbackName if doc.elidedFallbackName is not None else 2,
        macNames=macNames,
    )


def getStatAxes(doc: DesignSpaceDocument, userRegion: Region) -> List[Dict]:
    """Return a list of axis dicts suitable for use as the ``axes``
    argument to :func:`fontTools.otlLib.builder.buildStatTable()`.

    .. versionadded:: 5.0
    """
    # First, get the axis labels with explicit ordering
    # then append the others in the order they appear.
    maxOrdering = max(
        (axis.axisOrdering for axis in doc.axes if axis.axisOrdering is not None),
        default=-1,
    )
    axisOrderings = []
    for axis in doc.axes:
        if axis.axisOrdering is not None:
            axisOrderings.append(axis.axisOrdering)
        else:
            maxOrdering += 1
            axisOrderings.append(maxOrdering)
    return [
        dict(
            tag=axis.tag,
            name={"en": axis.name, **axis.labelNames},
            ordering=ordering,
            values=[
                _axisLabelToStatLocation(label)
                for label in axis.axisLabels
                if locationInRegion({axis.name: label.userValue}, userRegion)
            ],
        )
        for axis, ordering in zip(doc.axes, axisOrderings)
    ]


def getStatLocations(doc: DesignSpaceDocument, userRegion: Region) -> List[Dict]:
    """Return a list of location dicts suitable for use as the ``locations``
    argument to :func:`fontTools.otlLib.builder.buildStatTable()`.

    .. versionadded:: 5.0
    """
    axesByName = {axis.name: axis for axis in doc.axes}
    return [
        dict(
            name={"en": label.name, **label.labelNames},
            # Location in the designspace is keyed by axis name
            # Location in buildStatTable by axis tag
            location={
                axesByName[name].tag: value
                for name, value in label.getFullUserLocation(doc).items()
            },
            flags=_labelToFlags(label),
        )
        for label in doc.locationLabels
        if locationInRegion(label.getFullUserLocation(doc), userRegion)
    ]


def _labelToFlags(label: Union[AxisLabelDescriptor, LocationLabelDescriptor]) -> int:
    flags = 0
    if label.olderSibling:
        flags |= 1
    if label.elidable:
        flags |= 2
    return flags


def _axisLabelToStatLocation(
    label: AxisLabelDescriptor,
) -> Dict:
    label_format = label.getFormat()
    name = {"en": label.name, **label.labelNames}
    flags = _labelToFlags(label)
    if label_format == 1:
        return dict(name=name, value=label.userValue, flags=flags)
    if label_format == 3:
        return dict(
            name=name,
            value=label.userValue,
            linkedValue=label.linkedUserValue,
            flags=flags,
        )
    if label_format == 2:
        res = dict(
            name=name,
            nominalValue=label.userValue,
            flags=flags,
        )
        if label.userMinimum is not None:
            res["rangeMinValue"] = label.userMinimum
        if label.userMaximum is not None:
            res["rangeMaxValue"] = label.userMaximum
        return res
    raise NotImplementedError("Unknown STAT label format")
<|endoftext|>from fontTools.misc.roundTools import noRound, otRound
from fontTools.misc.intTools import bit_count
from fontTools.ttLib.tables import otTables as ot
from fontTools.varLib.models import supportScalar
from fontTools.varLib.builder import (
    buildVarRegionList,
    buildVarStore,
    buildVarRegion,
    buildVarData,
)
from functools import partial
from collections import defaultdict
from heapq import heappush, heappop
import itertools

NO_VARIATION_INDEX = ot.NO_VARIATION_INDEX
ot.VarStore.NO_VARIATION_INDEX = NO_VARIATION_INDEX


def _getLocationKey(loc):
    return tuple(sorted(loc.items(), key=lambda kv: kv[0]))


class OnlineVarStoreBuilder(object):
    def __init__(self, axisTags):
        self._axisTags = axisTags
        self._regionMap = {}
        self._regionList = buildVarRegionList([], axisTags)
        self._store = buildVarStore(self._regionList, [])
        self._data = None
        self._model = None
        self._supports = None
        self._varDataIndices = {}
        self._varDataCaches = {}
        self._cache = None

    def setModel(self, model):
        self.setSupports(model.supports)
        self._model = model

    def setSupports(self, supports):
        self._model = None
        self._supports = list(supports)
        if self._supports and not self._supports[0]:
            del self._supports[0]  # Drop base master support
        self._cache = None
        self._data = None

    def finish(self, optimize=True):
        self._regionList.RegionCount = len(self._regionList.Region)
        self._store.VarDataCount = len(self._store.VarData)
        for data in self._store.VarData:
            data.ItemCount = len(data.Item)
            data.calculateNumShorts(optimize=optimize)
        return self._store

    def _add_VarData(self, num_items=1):
        regionMap = self._regionMap
        regionList = self._regionList

        regions = self._supports
        regionIndices = []
        for region in regions:
            key = _getLocationKey(region)
            idx = regionMap.get(key)
            if idx is None:
                varRegion = buildVarRegion(region, self._axisTags)
                idx = regionMap[key] = len(regionList.Region)
                regionList.Region.append(varRegion)
            regionIndices.append(idx)

        # Check if we have one already...
        key = tuple(regionIndices)
        varDataIdx = self._varDataIndices.get(key)
        if varDataIdx is not None:
            self._outer = varDataIdx
            self._data = self._store.VarData[varDataIdx]
            self._cache = self._varDataCaches[key]
            if len(self._data.Item) + num_items > 0xFFFF:
                # This is full.  Need new one.
                varDataIdx = None

        if varDataIdx is None:
            self._data = buildVarData(regionIndices, [], optimize=False)
            self._outer = len(self._store.VarData)
            self._store.VarData.append(self._data)
            self._varDataIndices[key] = self._outer
            if key not in self._varDataCaches:
                self._varDataCaches[key] = {}
            self._cache = self._varDataCaches[key]

    def storeMasters(self, master_values, *, round=round):
        deltas = self._model.getDeltas(master_values, round=round)
        base = deltas.pop(0)
        return base, self.storeDeltas(deltas, round=noRound)

    def storeMastersMany(self, master_values_list, *, round=round):
        deltas_list = [
            self._model.getDeltas(master_values, round=round)
            for master_values in master_values_list
        ]
        base_list = [deltas.pop(0) for deltas in deltas_list]
        return base_list, self.storeDeltasMany(deltas_list, round=noRound)

    def storeDeltas(self, deltas, *, round=round):
        deltas = [round(d) for d in deltas]
        if len(deltas) == len(self._supports) + 1:
            deltas = tuple(deltas[1:])
        else:
            assert len(deltas) == len(self._supports)
            deltas = tuple(deltas)

        if not self._data:
            self._add_VarData()

        varIdx = self._cache.get(deltas)
        if varIdx is not None:
            return varIdx

        inner = len(self._data.Item)
        if inner == 0xFFFF:
            # Full array. Start new one.
            self._add_VarData()
            return self.storeDeltas(deltas, round=noRound)
        self._data.addItem(deltas, round=noRound)

        varIdx = (self._outer << 16) + inner
        self._cache[deltas] = varIdx
        return varIdx

    def storeDeltasMany(self, deltas_list, *, round=round):
        deltas_list = [[round(d) for d in deltas] for deltas in deltas_list]
        deltas_list = tuple(tuple(deltas) for deltas in deltas_list)

        if not self._data:
            self._add_VarData(len(deltas_list))

        varIdx = self._cache.get(deltas_list)
        if varIdx is not None:
            return varIdx

        inner = len(self._data.Item)
        if inner + len(deltas_list) > 0xFFFF:
            # Full array. Start new one.
            self._add_VarData(len(deltas_list))
            return self.storeDeltasMany(deltas_list, round=noRound)
        for i, deltas in enumerate(deltas_list):
            self._data.addItem(deltas, round=noRound)

            varIdx = (self._outer << 16) + inner + i
            self._cache[deltas] = varIdx

        varIdx = (self._outer << 16) + inner
        self._cache[deltas_list] = varIdx

        return varIdx


def VarData_addItem(self, deltas, *, round=round):
    deltas = [round(d) for d in deltas]

    countUs = self.VarRegionCount
    countThem = len(deltas)
    if countUs + 1 == countThem:
        deltas = list(deltas[1:])
    else:
        assert countUs == countThem, (countUs, countThem)
        deltas = list(deltas)
    self.Item.append(deltas)
    self.ItemCount = len(self.Item)


ot.VarData.addItem = VarData_addItem


def VarRegion_get_support(self, fvar_axes):
    return {
        fvar_axes[i].axisTag: (reg.StartCoord, reg.PeakCoord, reg.EndCoord)
        for i, reg in enumerate(self.VarRegionAxis)
        if reg.PeakCoord != 0
    }


ot.VarRegion.get_support = VarRegion_get_support


def VarStore___bool__(self):
    return bool(self.VarData)


ot.VarStore.__bool__ = VarStore___bool__


class VarStoreInstancer(object):
    """Evaluate a VarStore's deltas at a (mutable via setLocation) location.

    Region supports are cached for the lifetime of the instancer: after
    mutating the store's regions (or the fvar axes), construct a new
    instancer rather than reusing this one via setLocation.
    """

    def __init__(self, varstore, fvar_axes, location={}):
        self.fvar_axes = fvar_axes
        assert varstore is None or varstore.Format == 1
        self._varData = varstore.VarData if varstore else []
        self._regions = varstore.VarRegionList.Region if varstore else []
        # Region supports are location-independent, so cache them for the
        # lifetime of the instancer (they survive setLocation); only the
        # per-region scalars, which depend on the location, are recomputed.
        # This makes reusing one instancer across many locations cheap.
        self._supports = {}
        self.setLocation(location)

    def setLocation(self, location):
        self.location = dict(location)
        self._clearCaches()

    def _clearCaches(self):
        self._scalars = {}

    def _getSupport(self, regionIdx):
        support = self._supports.get(regionIdx)
        if support is None:
            support = self._regions[regionIdx].get_support(self.fvar_axes)
            self._supports[regionIdx] = support
        return support

    def _getScalar(self, regionIdx):
        scalar = self._scalars.get(regionIdx)
        if scalar is None:
            scalar = supportScalar(self.location, self._getSupport(regionIdx))
            self._scalars[regionIdx] = scalar
        return scalar

    @staticmethod
    def interpolateFromDeltasAndScalars(deltas, scalars):
        delta = 0.0
        for d, s in zip(deltas, scalars):
            if not s:
                continue
            delta += d * s
        return delta

    def __getitem__(self, varidx):
        major, minor = varidx >> 16, varidx & 0xFFFF
        if varidx == NO_VARIATION_INDEX:
            return 0.0
        varData = self._varData
        # Keep full instancing consistent with _effectiveAvar2VarIdx: out-of-range
        # delta-set indices contribute zero. HarfBuzz uses the same policy:
        # https://github.com/harfbuzz/harfbuzz/blob/d2df3cdcc0836299a163dc0399a6b047d19ac56c/src/hb-ot-layout-common.hh#L2912-L2918
        # https://github.com/harfbuzz/harfbuzz/blob/d2df3cdcc0836299a163dc0399a6b047d19ac56c/src/hb-ot-layout-common.hh#L3354-L3363
        if major >= len(varData) or minor >= len(varData[major].Item):
            return 0.0
        scalars = [self._getScalar(ri) for ri in varData[major].VarRegionIndex]
        deltas = varData[major].Item[minor]
        return self.interpolateFromDeltasAndScalars(deltas, scalars)

    def interpolateFromDeltas(self, varDataIndex, deltas):
        varData = self._varData
        scalars = [self._getScalar(ri) for ri in varData[varDataIndex].VarRegionIndex]
        return self.interpolateFromDeltasAndScalars(deltas, scalars)


#
# Optimizations
#
# retainFirstMap - If true, major 0 mappings are retained. Deltas for unused indices are zeroed
# advIdxes - Set of major 0 indices for advance deltas to be listed first. Other major 0 indices follow.


def VarStore_subset_varidxes(
    self,
    varIdxes,
    optimize=True,
    retainFirstMap=False,
    advIdxes=set(),
    *,
    VarData="VarData",
):
    # Sort out used varIdxes by major/minor.
    used = defaultdict(set)
    for varIdx in varIdxes:
        if varIdx == NO_VARIATION_INDEX:
            continue
        major = varIdx >> 16
        minor = varIdx & 0xFFFF
        used[major].add(minor)
    del varIdxes

    #
    # Subset VarData
    #

    varData = getattr(self, VarData)
    newVarData = []
    varDataMap = {NO_VARIATION_INDEX: NO_VARIATION_INDEX}
    for major, data in enumerate(varData):
        usedMinors = used.get(major)
        if usedMinors is None:
            continue
        newMajor = len(newVarData)
        newVarData.append(data)

        items = data.Item
        newItems = []
        if major == 0 and retainFirstMap:
            for minor in range(len(items)):
                newItems.append(
                    items[minor] if minor in usedMinors else [0] * len(items[minor])
                )
                varDataMap[minor] = minor
        else:
            if major == 0:
                minors = sorted(advIdxes) + sorted(usedMinors - advIdxes)
            else:
                minors = sorted(usedMinors)
            for minor in minors:
                newMinor = len(newItems)
                newItems.append(items[minor])
                varDataMap[(major << 16) + minor] = (newMajor << 16) + newMinor

        data.Item = newItems
        data.ItemCount = len(data.Item)

        if VarData == "VarData":
            data.calculateNumShorts(optimize=optimize)

    setattr(self, VarData, newVarData)
    setattr(self, VarData + "Count", len(newVarData))

    self.prune_regions()

    return varDataMap


ot.VarStore.subset_varidxes = VarStore_subset_varidxes


def VarStore_prune_regions(self, *, VarData="VarData", VarRegionList="VarRegionList"):
    """Remove unused VarRegions."""
    #
    # Subset VarRegionList
    #

    # Collect.
    usedRegions = set()
    for data in getattr(self, VarData):
        usedRegions.update(data.VarRegionIndex)
    # Subset.
    regionList = getattr(self, VarRegionList)
    regions = regionList.Region
    newRegions = []
    regionMap = {}
    for i in sorted(usedRegions):
        regionMap[i] = len(newRegions)
        newRegions.append(regions[i])
    regionList.Region = newRegions
    regionList.RegionCount = len(regionList.Region)
    # Map.
    for data in getattr(self, VarData):
        data.VarRegionIndex = [regionMap[i] for i in data.VarRegionIndex]


ot.VarStore.prune_regions = VarStore_prune_regions


def _visit(self, func):
    """Recurse down from self, if type of an object is ot.Device,
    call func() on it.  Works on otData-style classes."""

    if type(self) == ot.Device:
        func(self)

    elif isinstance(self, list):
        for that in self:
            _visit(that, func)

    elif hasattr(self, "getConverters") and not hasattr(self, "postRead"):
        for conv in self.getConverters():
            that = getattr(self, conv.name, None)
            if that is not None:
                _visit(that, func)

    elif isinstance(self, ot.ValueRecord):
        for that in self.__dict__.values():
            _visit(that, func)


def _Device_recordVarIdx(self, s):
    """Add VarIdx in this Device table (if any) to the set s."""
    if self.DeltaFormat == 0x8000:
        s.add((self.StartSize << 16) + self.EndSize)


def Object_collect_device_varidxes(self, varidxes):
    adder = partial(_Device_recordVarIdx, s=varidxes)
    _visit(self, adder)


ot.GDEF.collect_device_varidxes = Object_collect_device_varidxes
ot.GPOS.collect_device_varidxes = Object_collect_device_varidxes


def _Device_mapVarIdx(self, mapping, done):
    """Map VarIdx in this Device table (if any) through mapping."""
    if id(self) in done:
        return
    done.add(id(self))
    if self.DeltaFormat == 0x8000:
        varIdx = mapping[(self.StartSize << 16) + self.EndSize]
        self.StartSize = varIdx >> 16
        self.EndSize = varIdx & 0xFFFF


def Object_remap_device_varidxes(self, varidxes_map):
    mapper = partial(_Device_mapVarIdx, mapping=varidxes_map, done=set())
    _visit(self, mapper)


ot.GDEF.remap_device_varidxes = Object_remap_device_varidxes
ot.GPOS.remap_device_varidxes = Object_remap_device_varidxes
ot.BASE.remap_device_varidxes = Object_remap_device_varidxes


class _Encoding(object):
    def __init__(self, chars):
        self.chars = chars
        self.width = bit_count(chars)
        self.columns = self._columns(chars)
        self.overhead = self._characteristic_overhead(self.columns)
        self.items = set()

    def append(self, row):
        self.items.add(row)

    def extend(self, lst):
        self.items.update(lst)

    def width_sort_key(self):
        return self.width, self.chars

    @staticmethod
    def _characteristic_overhead(columns):
        """Returns overhead in bytes of encoding this characteristic
        as a VarData."""
        c = 4 + 6  # 4 bytes for LOffset, 6 bytes for VarData header
        c += bit_count(columns) * 2
        return c

    @staticmethod
    def _columns(chars):
        cols = 0
        i = 1
        while chars:
            if chars & 0b1111:
                cols |= i
            chars >>= 4
            i <<= 1
        return cols

    def gain_from_merging(self, other_encoding):
        combined_chars = other_encoding.chars | self.chars
        combined_width = bit_count(combined_chars)
        combined_columns = self.columns | other_encoding.columns
        combined_overhead = _Encoding._characteristic_overhead(combined_columns)
        combined_gain = (
            +self.overhead
            + other_encoding.overhead
            - combined_overhead
            - (combined_width - self.width) * len(self.items)
            - (combined_width - other_encoding.width) * len(other_encoding.items)
        )
        return combined_gain


class _EncodingDict(dict):
    def __missing__(self, chars):
        r = self[chars] = _Encoding(chars)
        return r

    def add_row(self, row):
        chars = self._row_characteristics(row)
        self[chars].append(row)

    @staticmethod
    def _row_characteristics(row):
        """Returns encoding characteristics for a row."""
        longWords = False

        chars = 0
        i = 1
        for v in row:
            if v:
                chars += i
            if not (-128 <= v <= 127):
                chars += i * 0b0010
            if not (-32768 <= v <= 32767):
                longWords = True
                break
            i <<= 4

        if longWords:
            # Redo; only allow 2byte/4byte encoding
            chars = 0
            i = 1
            for v in row:
                if v:
                    chars += i * 0b0011
                if not (-32768 <= v <= 32767):
                    chars += i * 0b1100
                i <<= 4

        return chars


def VarStore_optimize(self, use_NO_VARIATION_INDEX=True, quantization=1):
    """Optimize storage. Returns mapping from old VarIdxes to new ones."""

    # Overview:
    #
    # For each VarData row, we first extend it with zeroes to have
    # one column per region in VarRegionList. We then group the
    # rows into _Encoding objects, by their "characteristic" bitmap.
    # The characteristic bitmap is a binary number representing how
    # many bytes each column of the data takes up to encode. Each
    # column is encoded in four bits. For example, if a column has
    # only values in the range -128..127, it would only have a single
    # bit set in the characteristic bitmap for that column. If it has
    # values in the range -32768..32767, it would have two bits set.
    # The number of ones in the characteristic bitmap is the "width"
    # of the encoding.
    #
    # Each encoding as such has a number of "active" (ie. non-zero)
    # columns. The overhead of encoding the characteristic bitmap
    # is 10 bytes, plus 2 bytes per active column.
    #
    # When an encoding is merged into another one, if the characteristic
    # of the old encoding is a subset of the new one, then the overhead
    # of the old encoding is completely eliminated. However, each row
    # now would require more bytes to encode, to the tune of one byte
    # per characteristic bit that is active in the new encoding but not
    # in the old one.
    #
    # The "gain" of merging two encodings is how many bytes we save by doing so.
    #
    # High-level algorithm:
    #
    # - Each encoding has a minimal way to encode it. However, because
    #   of the overhead of encoding the characteristic bitmap, it may
    #   be beneficial to merge two encodings together, if there is
    #   gain in doing so. As such, we need to search for the best
    #   such successive merges.
    #
    # Algorithm:
    #
    # - Put all encodings into a "todo" list.
    #
    # - Sort todo list (for stability) by width_sort_key(), which is a tuple
    #   of the following items:
    #   * The "width" of the encoding.
    #   * The characteristic bitmap of the encoding, with higher-numbered
    #     columns compared first.
    #
    # - Make a priority-queue of the gain from combining each two
    #   encodings in the todo list. The priority queue is sorted by
    #   decreasing gain. Only positive gains are included.
    #
    # - While priority queue is not empty:
    #   - Pop the first item from the priority queue,
    #   - Merge the two encodings it represents,
    #   - Remove the two encodings from the todo list,
    #   - Insert positive gains from combining the new encoding with
    #     all existing todo list items into the priority queue,
    #   - If a todo list item with the same characteristic bitmap as
    #     the new encoding exists, remove it from the todo list and
    #     merge it into the new encoding.
    #   - Insert the new encoding into the todo list,
    #
    # - Encode all remaining items in the todo list.
    #
    # The output is then sorted for stability, in the following way:
    # - The VarRegionList of the input is kept intact.
    # - The VarData is sorted by the same width_sort_key() used at the beginning.
    # - Within each VarData, the items are sorted as vectors of numbers.
    #
    # Finally, each VarData is optimized to remove the empty columns and
    # reorder columns as needed.

    # TODO
    # Check that no two VarRegions are the same; if they are, fold them.

    n = len(self.VarRegionList.Region)  # Number of columns
    zeroes = [0] * n

    front_mapping = {}  # Map from old VarIdxes to full row tuples

    encodings = _EncodingDict()

    # Collect all items into a set of full rows (with lots of zeroes.)
    for major, data in enumerate(self.VarData):
        regionIndices = data.VarRegionIndex

        for minor, item in enumerate(data.Item):
            row = list(zeroes)

            if quantization == 1:
                for regionIdx, v in zip(regionIndices, item):
                    row[regionIdx] += v
            else:
                for regionIdx, v in zip(regionIndices, item):
                    row[regionIdx] += (
                        round(v / quantization) * quantization
                    )  # TODO https://github.com/fonttools/fonttools/pull/3126#discussion_r1205439785

            row = tuple(row)

            if use_NO_VARIATION_INDEX and not any(row):
                front_mapping[(major << 16) + minor] = None
                continue

            encodings.add_row(row)
            front_mapping[(major << 16) + minor] = row

    # Prepare for the main algorithm.
    todo = sorted(encodings.values(), key=_Encoding.width_sort_key)
    del encodings

    # Repeatedly pick two best encodings to combine, and combine them.

    heap = []
    for i, encoding in enumerate(todo):
        for j in range(i + 1, len(todo)):
            other_encoding = todo[j]
            combining_gain = encoding.gain_from_merging(other_encoding)
            if combining_gain > 0:
                heappush(heap, (-combining_gain, i, j))

    while heap:
        _, i, j = heappop(heap)
        if todo[i] is None or todo[j] is None:
            continue

        encoding, other_encoding = todo[i], todo[j]
        todo[i], todo[j] = None, None

        # Combine the two encodings
        combined_chars = other_encoding.chars | encoding.chars
        combined_encoding = _Encoding(combined_chars)
        combined_encoding.extend(encoding.items)
        combined_encoding.extend(other_encoding.items)

        for k, enc in enumerate(todo):
            if enc is None:
                continue

            # In the unlikely event that the same encoding exists already,
            # combine it.
            if enc.chars == combined_chars:
                combined_encoding.extend(enc.items)
                todo[k] = None
                continue

            combining_gain = combined_encoding.gain_from_merging(enc)
            if combining_gain > 0:
                heappush(heap, (-combining_gain, k, len(todo)))

        todo.append(combined_encoding)

    encodings = [encoding for encoding in todo if encoding is not None]

    # Assemble final store.
    back_mapping = {}  # Mapping from full rows to new VarIdxes
    encodings.sort(key=_Encoding.width_sort_key)
    self.VarData = []
    for encoding in encodings:
        items = sorted(encoding.items)

        while items:
            major = len(self.VarData)
            data = ot.VarData()
            self.VarData.append(data)
            data.VarRegionIndex = range(n)
            data.VarRegionCount = len(data.VarRegionIndex)

            # Each major can only encode up to 0xFFFF entries.
            data.Item, items = items[:0xFFFF], items[0xFFFF:]

            for minor, item in enumerate(data.Item):
                back_mapping[item] = (major << 16) + minor

    # Compile final mapping.
    varidx_map = {NO_VARIATION_INDEX: NO_VARIATION_INDEX}
    for k, v in front_mapping.items():
        varidx_map[k] = back_mapping[v] if v is not None else NO_VARIATION_INDEX

    # Recalculate things and go home.
    self.VarRegionList.RegionCount = len(self.VarRegionList.Region)
    self.VarDataCount = len(self.VarData)
    for data in self.VarData:
        data.ItemCount = len(data.Item)
        data.optimize()

    # Remove unused regions.
    self.prune_regions()

    return varidx_map


ot.VarStore.optimize = VarStore_optimize


def _tentValue(v, start, peak, end):
    """Per-axis support-scalar factor at coordinate v (supportScalar semantics)."""
    if v == peak:
        return 1.0
    if v <= start or end <= v:
        return 0.0
    if v < peak:
        return (v - start) / (peak - start)
    return (end - v) / (end - peak)


# Maximum number of grid vertices VarStore_getExtremes enumerates for the
# exact answer before falling back to the (looser but conservative and cheap)
# interval bound. The grid is authoring-controlled, so a cap is needed to keep
# crafted fonts from hanging the instancer.
_GET_EXTREMES_MAX_GRID = 1 << 14
_F2DOT14_UNIT = 1.0 / (1 << 14)


def VarStore_getExtremes(self, varIdx, fvarAxes, axisLimits, identityAxisIndex=None):
    """Conservatively bound the value this VarStore produces for varIdx.

    Returns (minV, maxV) in F2Dot14 delta units, guaranteed to cover the
    truly reachable range as the input axes vary over their extents:
    [-1, +1] per axis, or the [min, max] interval from axisLimits (a dict of
    axis tag to a (min, default, max, ...) sequence) for axes listed there.
    If identityAxisIndex is not None, that axis's own identity contribution
    (its coordinate, in F2Dot14 units) is added to the bound.

    Before the final OpenType rounding, the produced value is piecewise
    multilinear in the axis coordinates (each region's scalar is a product
    of independent per-axis tents), so over the box its exact extremes are
    attained at vertices of the grid formed by each axis's tent
    breakpoints. One-sided tents (start == peak or peak == end) are
    discontinuous at their peak, so the grid also includes the adjacent
    F2Dot14 coordinate on the open side. When that grid is small enough
    (real fonts) it is enumerated and the result is exact; otherwise the
    result falls back to a per-region interval bound, which ignores
    correlations between regions and may be wider than the true range —
    never narrower. Callers use this to cull provably-unreachable
    variation data, so conservativeness is a soundness requirement, not a
    nicety. The fallback costs O(regions × axes); the cap on the exact
    path keeps crafted fonts from hanging the instancer.

    Raw extrema are rounded only after summing the identity and delta terms,
    matching avar2 processing.
    """

    def axisInterval(axisIdx):
        tag = fvarAxes[axisIdx].axisTag
        if tag in axisLimits:
            limits = axisLimits[tag]
            return limits[0], limits[2]
        return -1.0, +1.0

    if varIdx == NO_VARIATION_INDEX:
        activeRegions = []
    else:
        major = varIdx >> 16
        minor = varIdx & 0xFFFF
