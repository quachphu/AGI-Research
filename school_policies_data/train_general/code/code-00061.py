            vfUserRegion[axis.name] = axisSubset.userValue
    # Any axis not mentioned explicitly has a single location = default value
    for axis in doc.axes:
        if axis.name not in vfUserRegion:
            assert isinstance(
                axis.default, (int, float)
            ), f"Axis '{axis.name}' has no valid default value."
            vfUserRegion[axis.name] = axis.default
    return vfUserRegion
<|endoftext|>"""
designSpaceDocument

- Read and write designspace files
"""

from __future__ import annotations

import collections
import copy
import itertools
import math
import os
import posixpath
from io import BytesIO, StringIO
from textwrap import indent
from typing import Any, Dict, List, MutableMapping, Optional, Tuple, Union, cast

from fontTools.misc import etree as ET
from fontTools.misc import plistlib
from fontTools.misc.loggingTools import LogMixin
from fontTools.misc.textTools import tobytes, tostr

__all__ = [
    "AxisDescriptor",
    "AxisLabelDescriptor",
    "AxisMappingDescriptor",
    "BaseDocReader",
    "BaseDocWriter",
    "DesignSpaceDocument",
    "DesignSpaceDocumentError",
    "DiscreteAxisDescriptor",
    "InstanceDescriptor",
    "LocationLabelDescriptor",
    "RangeAxisSubsetDescriptor",
    "RuleDescriptor",
    "SourceDescriptor",
    "ValueAxisSubsetDescriptor",
    "VariableFontDescriptor",
]

# ElementTree allows to find namespace-prefixed elements, but not attributes
# so we have to do it ourselves for 'xml:lang'
XML_NS = "{http://www.w3.org/XML/1998/namespace}"
XML_LANG = XML_NS + "lang"


def posix(path):
    """Normalize paths using forward slash to work also on Windows."""
    new_path = posixpath.join(*path.split(os.path.sep))
    if path.startswith("/"):
        # The above transformation loses absolute paths
        new_path = "/" + new_path
    elif path.startswith(r"\\"):
        # The above transformation loses leading slashes of UNC path mounts
        new_path = "//" + new_path
    return new_path


def posixpath_property(private_name):
    """Generate a propery that holds a path always using forward slashes."""

    def getter(self):
        # Normal getter
        return getattr(self, private_name)

    def setter(self, value):
        # The setter rewrites paths using forward slashes
        if value is not None:
            value = posix(value)
        setattr(self, private_name, value)

    return property(getter, setter)


class DesignSpaceDocumentError(Exception):
    def __init__(self, msg, obj=None):
        self.msg = msg
        self.obj = obj

    def __str__(self):
        return str(self.msg) + (": %r" % self.obj if self.obj is not None else "")


class AsDictMixin(object):
    def asdict(self):
        d = {}
        for attr, value in self.__dict__.items():
            if attr.startswith("_"):
                continue
            if hasattr(value, "asdict"):
                value = value.asdict()
            elif isinstance(value, list):
                value = [v.asdict() if hasattr(v, "asdict") else v for v in value]
            d[attr] = value
        return d


class SimpleDescriptor(AsDictMixin):
    """Containers for a bunch of attributes"""

    # XXX this is ugly. The 'print' is inappropriate here, and instead of
    # assert, it should simply return True/False
    def compare(self, other):
        # test if this object contains the same data as the other
        for attr in self._attrs:
            try:
                assert getattr(self, attr) == getattr(other, attr)
            except AssertionError:
                print(
                    "failed attribute",
                    attr,
                    getattr(self, attr),
                    "!=",
                    getattr(other, attr),
                )

    def __repr__(self):
        attrs = [f"{a}={repr(getattr(self, a))}," for a in self._attrs]
        attrs = indent("\n".join(attrs), "    ")
        return f"{self.__class__.__name__}(\n{attrs}\n)"


class SourceDescriptor(SimpleDescriptor):
    """Simple container for data related to the source

    .. code:: python

        doc = DesignSpaceDocument()
        s1 = SourceDescriptor()
        s1.path = masterPath1
        s1.name = "master.ufo1"
        s1.font = defcon.Font("master.ufo1")
        s1.location = dict(weight=0)
        s1.familyName = "MasterFamilyName"
        s1.styleName = "MasterStyleNameOne"
        s1.localisedFamilyName = dict(fr="Caractère")
        s1.mutedGlyphNames.append("A")
        s1.mutedGlyphNames.append("Z")
        doc.addSource(s1)

    """

    flavor = "source"
    _attrs = [
        "filename",
        "path",
        "name",
        "layerName",
        "location",
        "copyLib",
        "copyGroups",
        "copyFeatures",
        "muteKerning",
        "muteInfo",
        "mutedGlyphNames",
        "familyName",
        "styleName",
        "localisedFamilyName",
    ]

    filename = posixpath_property("_filename")
    path = posixpath_property("_path")

    def __init__(
        self,
        *,
        filename=None,
        path=None,
        font=None,
        name=None,
        location=None,
        designLocation=None,
        layerName=None,
        familyName=None,
        styleName=None,
        localisedFamilyName=None,
        copyLib=False,
        copyInfo=False,
        copyGroups=False,
        copyFeatures=False,
        muteKerning=False,
        muteInfo=False,
        mutedGlyphNames=None,
    ):
        self.filename = filename
        """string. A relative path to the source file, **as it is in the document**.

        MutatorMath + VarLib.
        """
        self.path = path
        """The absolute path, calculated from filename."""

        self.font = font
        """Any Python object. Optional. Points to a representation of this
        source font that is loaded in memory, as a Python object (e.g. a
        ``defcon.Font`` or a ``fontTools.ttFont.TTFont``).

        The default document reader will not fill-in this attribute, and the
        default writer will not use this attribute. It is up to the user of
        ``designspaceLib`` to either load the resource identified by
        ``filename`` and store it in this field, or write the contents of
        this field to the disk and make ```filename`` point to that.
        """

        self.name = name
        """string. Optional. Unique identifier name for this source.

        MutatorMath + varLib.
        """

        self.designLocation = (
            designLocation if designLocation is not None else location or {}
        )
        """dict. Axis values for this source, in design space coordinates.

        MutatorMath + varLib.

        This may be only part of the full design location.
        See :meth:`getFullDesignLocation()`

        .. versionadded:: 5.0
        """

        self.layerName = layerName
        """string. The name of the layer in the source to look for
        outline data. Default ``None`` which means ``foreground``.
        """
        self.familyName = familyName
        """string. Family name of this source. Though this data
        can be extracted from the font, it can be efficient to have it right
        here.

        varLib.
        """
        self.styleName = styleName
        """string. Style name of this source. Though this data
        can be extracted from the font, it can be efficient to have it right
        here.

        varLib.
        """
        self.localisedFamilyName = localisedFamilyName or {}
        """dict. A dictionary of localised family name strings, keyed by
        language code.

        If present, will be used to build localized names for all instances.

        .. versionadded:: 5.0
        """

        self.copyLib = copyLib
        """bool. Indicates if the contents of the font.lib need to
        be copied to the instances.

        MutatorMath.

        .. deprecated:: 5.0
        """
        self.copyInfo = copyInfo
        """bool. Indicates if the non-interpolating font.info needs
        to be copied to the instances.

        MutatorMath.

        .. deprecated:: 5.0
        """
        self.copyGroups = copyGroups
        """bool. Indicates if the groups need to be copied to the
        instances.

        MutatorMath.

        .. deprecated:: 5.0
        """
        self.copyFeatures = copyFeatures
        """bool. Indicates if the feature text needs to be
        copied to the instances.

        MutatorMath.

        .. deprecated:: 5.0
        """
        self.muteKerning = muteKerning
        """bool. Indicates if the kerning data from this source
        needs to be muted (i.e. not be part of the calculations).

        MutatorMath only.
        """
        self.muteInfo = muteInfo
        """bool. Indicated if the interpolating font.info data for
        this source needs to be muted.

        MutatorMath only.
        """
        self.mutedGlyphNames = mutedGlyphNames or []
        """list. Glyphnames that need to be muted in the
        instances.

        MutatorMath only.
        """

    @property
    def location(self):
        """dict. Axis values for this source, in design space coordinates.

        MutatorMath + varLib.

        .. deprecated:: 5.0
           Use the more explicit alias for this property :attr:`designLocation`.
        """
        return self.designLocation

    @location.setter
    def location(self, location: Optional[SimpleLocationDict]):
        self.designLocation = location or {}

    def setFamilyName(self, familyName, languageCode="en"):
        """Setter for :attr:`localisedFamilyName`

        .. versionadded:: 5.0
        """
        self.localisedFamilyName[languageCode] = tostr(familyName)

    def getFamilyName(self, languageCode="en"):
        """Getter for :attr:`localisedFamilyName`

        .. versionadded:: 5.0
        """
        return self.localisedFamilyName.get(languageCode)

    def getFullDesignLocation(self, doc: "DesignSpaceDocument") -> SimpleLocationDict:
        """Get the complete design location of this source, from its
        :attr:`designLocation` and the document's axis defaults.

        .. versionadded:: 5.0
        """
        result: SimpleLocationDict = {}
        for axis in doc.axes:
            if axis.name in self.designLocation:
                result[axis.name] = self.designLocation[axis.name]
            else:
                result[axis.name] = axis.map_forward(axis.default)
        return result


class RuleDescriptor(SimpleDescriptor):
    """Represents the rule descriptor element: a set of glyph substitutions to
    trigger conditionally in some parts of the designspace.

    .. code:: python

        r1 = RuleDescriptor()
        r1.name = "unique.rule.name"
        r1.conditionSets.append([dict(name="weight", minimum=-10, maximum=10), dict(...)])
        r1.conditionSets.append([dict(...), dict(...)])
        r1.subs.append(("a", "a.alt"))

    .. code:: xml

        <!-- optional: list of substitution rules -->
        <rules>
            <rule name="vertical.bars">
                <conditionset>
                    <condition minimum="250.000000" maximum="750.000000" name="weight"/>
                    <condition minimum="100" name="width"/>
                    <condition minimum="10" maximum="40" name="optical"/>
                </conditionset>
                <sub name="cent" with="cent.alt"/>
                <sub name="dollar" with="dollar.alt"/>
            </rule>
        </rules>
    """

    _attrs = ["name", "conditionSets", "subs"]  # what do we need here

    def __init__(self, *, name=None, conditionSets=None, subs=None):
        self.name = name
        """string. Unique name for this rule. Can be used to reference this rule data."""
        # list of lists of dict(name='aaaa', minimum=0, maximum=1000)
        self.conditionSets = conditionSets or []
        """a list of conditionsets.

        -  Each conditionset is a list of conditions.
        -  Each condition is a dict with ``name``, ``minimum`` and ``maximum`` keys.
        """
        # list of substitutions stored as tuples of glyphnames ("a", "a.alt")
        self.subs = subs or []
        """list of substitutions.

        -  Each substitution is stored as tuples of glyphnames, e.g. ("a", "a.alt").
        -  Note: By default, rules are applied first, before other text
           shaping/OpenType layout, as they are part of the
           `Required Variation Alternates OpenType feature <https://docs.microsoft.com/en-us/typography/opentype/spec/features_pt#-tag-rvrn>`_.
           See ref:`rules-element` § Attributes.
        """


def evaluateRule(rule, location):
    """Return True if any of the rule's conditionsets matches the given location."""
    return any(evaluateConditions(c, location) for c in rule.conditionSets)


def evaluateConditions(conditions, location):
    """Return True if all the conditions matches the given location.

    - If a condition has no minimum, check for < maximum.
    - If a condition has no maximum, check for > minimum.
    """
    for cd in conditions:
        value = location[cd["name"]]
        if cd.get("minimum") is None:
            if value > cd["maximum"]:
                return False
        elif cd.get("maximum") is None:
            if cd["minimum"] > value:
                return False
        elif not cd["minimum"] <= value <= cd["maximum"]:
            return False
    return True


def processRules(rules, location, glyphNames):
    """Apply these rules at this location to these glyphnames.

    Return a new list of glyphNames with substitutions applied.

    - rule order matters
    """
    newNames = []
    for rule in rules:
        if evaluateRule(rule, location):
            for name in glyphNames:
                swap = False
                for a, b in rule.subs:
                    if name == a:
                        swap = True
                        break
                if swap:
                    newNames.append(b)
                else:
                    newNames.append(name)
            glyphNames = newNames
            newNames = []
    return glyphNames


AnisotropicLocationDict = Dict[str, Union[float, Tuple[float, float]]]
SimpleLocationDict = Dict[str, float]


class AxisMappingDescriptor(SimpleDescriptor):
    """Represents the axis mapping element: mapping an input location
    to an output location in the designspace.

    .. code:: python

        m1 = AxisMappingDescriptor()
        m1.inputLocation = {"weight": 900, "width": 150}
        m1.outputLocation = {"weight": 870}

    .. code:: xml

        <mappings>
            <mapping>
                <input>
                    <dimension name="weight" xvalue="900"/>
                    <dimension name="width" xvalue="150"/>
                </input>
                <output>
                    <dimension name="weight" xvalue="870"/>
                </output>
            </mapping>
        </mappings>
    """

    _attrs = ["inputLocation", "outputLocation"]

    def __init__(
        self,
        *,
        inputLocation=None,
        outputLocation=None,
        description=None,
        groupDescription=None,
    ):
        self.inputLocation: SimpleLocationDict = inputLocation or {}
        """dict. Axis values for the input of the mapping, in design space coordinates.

        varLib.

        .. versionadded:: 5.1
        """
        self.outputLocation: SimpleLocationDict = outputLocation or {}
        """dict. Axis values for the output of the mapping, in design space coordinates.

        varLib.

        .. versionadded:: 5.1
        """
        self.description = description
        """string. A description of the mapping.

        varLib.

        .. versionadded:: 5.2
        """
        self.groupDescription = groupDescription
        """string. A description of the group of mappings.

        varLib.

        .. versionadded:: 5.2
        """


class InstanceDescriptor(SimpleDescriptor):
    """Simple container for data related to the instance


    .. code:: python

        i2 = InstanceDescriptor()
        i2.path = instancePath2
        i2.familyName = "InstanceFamilyName"
        i2.styleName = "InstanceStyleName"
        i2.name = "instance.ufo2"
        # anisotropic location
        i2.designLocation = dict(weight=500, width=(400,300))
        i2.postScriptFontName = "InstancePostscriptName"
        i2.styleMapFamilyName = "InstanceStyleMapFamilyName"
        i2.styleMapStyleName = "InstanceStyleMapStyleName"
        i2.lib['com.coolDesignspaceApp.specimenText'] = 'Hamburgerwhatever'
        doc.addInstance(i2)
    """

    flavor = "instance"
    _defaultLanguageCode = "en"
    _attrs = [
        "filename",
        "path",
        "name",
        "locationLabel",
        "designLocation",
        "userLocation",
        "familyName",
        "styleName",
        "postScriptFontName",
        "styleMapFamilyName",
        "styleMapStyleName",
        "localisedFamilyName",
        "localisedStyleName",
        "localisedStyleMapFamilyName",
        "localisedStyleMapStyleName",
        "glyphs",
        "kerning",
        "info",
        "lib",
    ]

    filename = posixpath_property("_filename")
    path = posixpath_property("_path")

    def __init__(
        self,
        *,
        filename=None,
        path=None,
        font=None,
        name=None,
        location=None,
        locationLabel=None,
        designLocation=None,
        userLocation=None,
        familyName=None,
        styleName=None,
        postScriptFontName=None,
        styleMapFamilyName=None,
        styleMapStyleName=None,
        localisedFamilyName=None,
        localisedStyleName=None,
        localisedStyleMapFamilyName=None,
        localisedStyleMapStyleName=None,
        glyphs=None,
        kerning=True,
        info=True,
        lib=None,
    ):
        self.filename = filename
        """string. Relative path to the instance file, **as it is
        in the document**. The file may or may not exist.

        MutatorMath + VarLib.
        """
        self.path = path
        """string. Absolute path to the instance file, calculated from
        the document path and the string in the filename attr. The file may
        or may not exist.

        MutatorMath.
        """
        self.font = font
        """Same as :attr:`SourceDescriptor.font`

        .. seealso:: :attr:`SourceDescriptor.font`
        """
        self.name = name
        """string. Unique identifier name of the instance, used to
        identify it if it needs to be referenced from elsewhere in the
        document.
        """
        self.locationLabel = locationLabel
        """Name of a :class:`LocationLabelDescriptor`. If
        provided, the instance should have the same location as the
        LocationLabel.

        .. seealso::
           :meth:`getFullDesignLocation`
           :meth:`getFullUserLocation`

        .. versionadded:: 5.0
        """
        self.designLocation: AnisotropicLocationDict = (
            designLocation if designLocation is not None else (location or {})
        )
        """dict. Axis values for this instance, in design space coordinates.

        MutatorMath + varLib.

        .. seealso:: This may be only part of the full location. See:
           :meth:`getFullDesignLocation`
           :meth:`getFullUserLocation`

        .. versionadded:: 5.0
        """
        self.userLocation: SimpleLocationDict = userLocation or {}
        """dict. Axis values for this instance, in user space coordinates.

        MutatorMath + varLib.

        .. seealso:: This may be only part of the full location. See:
           :meth:`getFullDesignLocation`
           :meth:`getFullUserLocation`

        .. versionadded:: 5.0
        """
        self.familyName = familyName
        """string. Family name of this instance.

        MutatorMath + varLib.
        """
        self.styleName = styleName
        """string. Style name of this instance.

        MutatorMath + varLib.
        """
        self.postScriptFontName = postScriptFontName
        """string. Postscript fontname for this instance.

        MutatorMath + varLib.
        """
        self.styleMapFamilyName = styleMapFamilyName
        """string. StyleMap familyname for this instance.

        MutatorMath + varLib.
        """
        self.styleMapStyleName = styleMapStyleName
        """string. StyleMap stylename for this instance.

        MutatorMath + varLib.
        """
        self.localisedFamilyName = localisedFamilyName or {}
        """dict. A dictionary of localised family name
        strings, keyed by language code.
        """
        self.localisedStyleName = localisedStyleName or {}
        """dict. A dictionary of localised stylename
        strings, keyed by language code.
        """
        self.localisedStyleMapFamilyName = localisedStyleMapFamilyName or {}
        """A dictionary of localised style map
        familyname strings, keyed by language code.
        """
        self.localisedStyleMapStyleName = localisedStyleMapStyleName or {}
        """A dictionary of localised style map
        stylename strings, keyed by language code.
        """
        self.glyphs = glyphs or {}
        """dict for special master definitions for glyphs. If glyphs
        need special masters (to record the results of executed rules for
        example).

        MutatorMath.

        .. deprecated:: 5.0
            Use rules or sparse sources instead.
        """
        self.kerning = kerning
        """ bool. Indicates if this instance needs its kerning
        calculated.

        MutatorMath.

        .. deprecated:: 5.0
        """
        self.info = info
        """bool. Indicated if this instance needs the interpolating
        font.info calculated.

        .. deprecated:: 5.0
        """

        self.lib = lib or {}
        """Custom data associated with this instance."""

    @property
    def location(self):
        """dict. Axis values for this instance.

        MutatorMath + varLib.

        .. deprecated:: 5.0
           Use the more explicit alias for this property :attr:`designLocation`.
        """
        return self.designLocation

    @location.setter
    def location(self, location: Optional[AnisotropicLocationDict]):
        self.designLocation = location or {}

    def setStyleName(self, styleName, languageCode="en"):
        """These methods give easier access to the localised names."""
        self.localisedStyleName[languageCode] = tostr(styleName)

    def getStyleName(self, languageCode="en"):
        return self.localisedStyleName.get(languageCode)

    def setFamilyName(self, familyName, languageCode="en"):
        self.localisedFamilyName[languageCode] = tostr(familyName)

    def getFamilyName(self, languageCode="en"):
        return self.localisedFamilyName.get(languageCode)

    def setStyleMapStyleName(self, styleMapStyleName, languageCode="en"):
        self.localisedStyleMapStyleName[languageCode] = tostr(styleMapStyleName)

    def getStyleMapStyleName(self, languageCode="en"):
        return self.localisedStyleMapStyleName.get(languageCode)

    def setStyleMapFamilyName(self, styleMapFamilyName, languageCode="en"):
        self.localisedStyleMapFamilyName[languageCode] = tostr(styleMapFamilyName)

    def getStyleMapFamilyName(self, languageCode="en"):
        return self.localisedStyleMapFamilyName.get(languageCode)

    def clearLocation(self, axisName: Optional[str] = None):
        """Clear all location-related fields. Ensures that
        :attr:``designLocation`` and :attr:``userLocation`` are dictionaries
        (possibly empty if clearing everything).

        In order to update the location of this instance wholesale, a user
        should first clear all the fields, then change the field(s) for which
        they have data.

        .. code:: python

            instance.clearLocation()
            instance.designLocation = {'Weight': (34, 36.5), 'Width': 100}
            instance.userLocation = {'Opsz': 16}

        In order to update a single axis location, the user should only clear
        that axis, then edit the values:

        .. code:: python

            instance.clearLocation('Weight')
            instance.designLocation['Weight'] = (34, 36.5)

        Args:
          axisName: if provided, only clear the location for that axis.

        .. versionadded:: 5.0
        """
        self.locationLabel = None
        if axisName is None:
            self.designLocation = {}
            self.userLocation = {}
        else:
            if self.designLocation is None:
                self.designLocation = {}
            if axisName in self.designLocation:
                del self.designLocation[axisName]
            if self.userLocation is None:
                self.userLocation = {}
            if axisName in self.userLocation:
                del self.userLocation[axisName]

    def getLocationLabelDescriptor(
        self, doc: "DesignSpaceDocument"
    ) -> Optional[LocationLabelDescriptor]:
        """Get the :class:`LocationLabelDescriptor` instance that matches
        this instances's :attr:`locationLabel`.

        Raises if the named label can't be found.

        .. versionadded:: 5.0
        """
        if self.locationLabel is None:
            return None
        label = doc.getLocationLabel(self.locationLabel)
        if label is None:
            raise DesignSpaceDocumentError(
                "InstanceDescriptor.getLocationLabelDescriptor(): "
                f"unknown location label `{self.locationLabel}` in instance `{self.name}`."
            )
        return label

    def getFullDesignLocation(
        self, doc: "DesignSpaceDocument"
    ) -> AnisotropicLocationDict:
        """Get the complete design location of this instance, by combining data
        from the various location fields, default axis values and mappings, and
        top-level location labels.

        The source of truth for this instance's location is determined for each
        axis independently by taking the first not-None field in this list:

        - ``locationLabel``: the location along this axis is the same as the
          matching STAT format 4 label. No anisotropy.
        - ``designLocation[axisName]``: the explicit design location along this
          axis, possibly anisotropic.
        - ``userLocation[axisName]``: the explicit user location along this
          axis. No anisotropy.
        - ``axis.default``: default axis value. No anisotropy.

        .. versionadded:: 5.0
        """
        label = self.getLocationLabelDescriptor(doc)
        if label is not None:
            return doc.map_forward(label.userLocation)  # type: ignore
        result: AnisotropicLocationDict = {}
        for axis in doc.axes:
            if axis.name in self.designLocation:
                result[axis.name] = self.designLocation[axis.name]
            elif axis.name in self.userLocation:
                result[axis.name] = axis.map_forward(self.userLocation[axis.name])
            else:
                result[axis.name] = axis.map_forward(axis.default)
        return result

    def getFullUserLocation(self, doc: "DesignSpaceDocument") -> SimpleLocationDict:
        """Get the complete user location for this instance.

        .. seealso:: :meth:`getFullDesignLocation`

        .. versionadded:: 5.0
        """
        return doc.map_backward(self.getFullDesignLocation(doc))


def tagForAxisName(name):
    # try to find or make a tag name for this axis name
    names = {
        "weight": ("wght", dict(en="Weight")),
        "width": ("wdth", dict(en="Width")),
        "optical": ("opsz", dict(en="Optical Size")),
        "slant": ("slnt", dict(en="Slant")),
        "italic": ("ital", dict(en="Italic")),
    }
    if name.lower() in names:
        return names[name.lower()]
    if len(name) < 4:
        tag = name + "*" * (4 - len(name))
    else:
        tag = name[:4]
    return tag, dict(en=name)


class AbstractAxisDescriptor(SimpleDescriptor):
    name: str | None
    map: list[tuple[float, float]]

    flavor = "axis"

    def __init__(
        self,
        *,
        tag=None,
        name: str | None = None,
        labelNames=None,
        hidden=False,
        map: list[tuple[float, float]] | None = None,
        axisOrdering=None,
        axisLabels=None,
    ):
        # opentype tag for this axis
        self.tag = tag
        """string. Four letter tag for this axis. Some might be
        registered at the `OpenType
        specification <https://www.microsoft.com/typography/otspec/fvar.htm#VAT>`__.
        Privately-defined axis tags must begin with an uppercase letter and
        use only uppercase letters or digits.
        """
        # name of the axis used in locations
        self.name = name
        """string. Name of the axis as it is used in the location dicts.

        MutatorMath + varLib.
        """
        # names for UI purposes, if this is not a standard axis,
        self.labelNames = labelNames or {}
        """dict. When defining a non-registered axis, it will be
        necessary to define user-facing readable names for the axis. Keyed by
        xml:lang code. Values are required to be ``unicode`` strings, even if
        they only contain ASCII characters.
        """
        self.hidden = hidden
        """bool. Whether this axis should be hidden in user interfaces.
        """
        self.map = map or []
        """list of input / output values that can describe a warp of user space
        to design space coordinates. If no map values are present, it is assumed
        user space is the same as design space, as in [(minimum, minimum),
        (maximum, maximum)].

        varLib.
        """
        self.axisOrdering = axisOrdering
        """STAT table field ``axisOrdering``.

        See: `OTSpec STAT Axis Record <https://docs.microsoft.com/en-us/typography/opentype/spec/stat#axis-records>`_

        .. versionadded:: 5.0
        """
        self.axisLabels: List[AxisLabelDescriptor] = axisLabels or []
        """STAT table entries for Axis Value Tables format 1, 2, 3.

        See: `OTSpec STAT Axis Value Tables <https://docs.microsoft.com/en-us/typography/opentype/spec/stat#axis-value-tables>`_

        .. versionadded:: 5.0
        """

    def get_validated_map(self) -> list[tuple[float, float]]:
        """Return a new list with exact duplicate pairs collapsed.

        The stored :attr:`map` is not modified.
        Multiple distinct input coordinates may map to the same output;
        duplicate outputs (many-to-one mappings) are valid.

        This validates only ambiguity in the input coordinates: an input
        coordinate mapping to different outputs raises
        :class:`DesignSpaceDocumentError`. It does not validate axis endpoint
        coverage or output monotonicity.
        """
        validated = {}
        for input_value, output_value in self.map:
            if input_value in validated:
                previous_output = validated[input_value]
                if previous_output != output_value:
                    raise DesignSpaceDocumentError(
                        f"Axis '{self.name}': mapping input value {input_value} "
                        f"has conflicting outputs {previous_output} and {output_value}."
                    )
                continue
            validated[input_value] = output_value
        return list(validated.items())


class AxisDescriptor(AbstractAxisDescriptor):
    """Simple container for the axis data.

    Add more localisations?

    .. code:: python

        a1 = AxisDescriptor()
        a1.minimum = 1
        a1.maximum = 1000
        a1.default = 400
        a1.name = "weight"
        a1.tag = "wght"
        a1.labelNames['fa-IR'] = "قطر"
        a1.labelNames['en'] = "Wéíght"
        a1.map = [(1.0, 10.0), (400.0, 66.0), (1000.0, 990.0)]
        a1.axisOrdering = 1
        a1.axisLabels = [
            AxisLabelDescriptor(name="Regular", userValue=400, elidable=True)
        ]
        doc.addAxis(a1)
    """

    _attrs = [
        "tag",
        "name",
        "maximum",
        "minimum",
        "default",
        "map",
        "axisOrdering",
        "axisLabels",
    ]

    def __init__(
        self,
        *,
        tag=None,
        name: str | None = None,
        labelNames=None,
        minimum=None,
        default=None,
        maximum=None,
        hidden=False,
        map: list[tuple[float, float]] | None = None,
        axisOrdering=None,
        axisLabels=None,
    ):
        super().__init__(
            tag=tag,
            name=name,
            labelNames=labelNames,
            hidden=hidden,
            map=map,
            axisOrdering=axisOrdering,
            axisLabels=axisLabels,
        )
        self.minimum = minimum
        """number. The minimum value for this axis in user space.

        MutatorMath + varLib.
        """
        self.maximum = maximum
        """number. The maximum value for this axis in user space.

        MutatorMath + varLib.
        """
        self.default = default
        """number. The default value for this axis, i.e. when a new location is
        created, this is the value this axis will get in user space.

        MutatorMath + varLib.
        """

    def serialize(self):
        # output to a dict, used in testing
        return dict(
            tag=self.tag,
            name=self.name,
            labelNames=self.labelNames,
            maximum=self.maximum,
            minimum=self.minimum,
            default=self.default,
            hidden=self.hidden,
            map=self.map,
            axisOrdering=self.axisOrdering,
            axisLabels=self.axisLabels,
        )

    def map_forward(self, v):
        """Maps value from axis mapping's input (user) to output (design)."""
        from fontTools.varLib.models import piecewiseLinearMap

        axis_map = self.get_validated_map()
        if not axis_map:
            return v
        return piecewiseLinearMap(v, dict(axis_map))

    def map_backward(self, v):
        """Maps value from axis mapping's output (design) to input (user)."""
        if isinstance(v, tuple):
            v = v[0]
        axis_map = self.get_validated_map()
        if not axis_map:
            return v
        # Build (design, user) pairs sorted by design then user, keeping
        # both endpoints of any many-to-one (flat) segments so we can
        # invert them and interpolation is correct on both sides.
        # https://github.com/googlefonts/ufo2ft/issues/978
        backward = sorted((design, user) for user, design in axis_map)
        design0, user0 = backward[0]
        if v <= design0:
            return v + user0 - design0
        for (design1, user1), (design2, user2) in zip(backward, backward[1:]):
            if design1 <= v <= design2:
                if design1 == design2:
                    return user1
                return user1 + (user2 - user1) * (v - design1) / (design2 - design1)
        designN, userN = backward[-1]
        return v + userN - designN


class DiscreteAxisDescriptor(AbstractAxisDescriptor):
    """Container for discrete axis data.

    Use this for axes that do not interpolate. The main difference from a
    continuous axis is that a continuous axis has a ``minimum`` and ``maximum``,
    while a discrete axis has a list of ``values``.

    Example: an Italic axis with 2 stops, Roman and Italic, that are not
    compatible. The axis still allows to bind together the full font family,
    which is useful for the STAT table, however it can't become a variation
    axis in a VF.

    .. code:: python

        a2 = DiscreteAxisDescriptor()
        a2.values = [0, 1]
        a2.default = 0
        a2.name = "Italic"
        a2.tag = "ITAL"
        a2.labelNames['fr'] = "Italique"
        a2.map = [(0, 0), (1, -11)]
        a2.axisOrdering = 2
        a2.axisLabels = [
            AxisLabelDescriptor(name="Roman", userValue=0, elidable=True)
        ]
        doc.addAxis(a2)

    .. versionadded:: 5.0
    """

    flavor = "axis"
    _attrs = ("tag", "name", "values", "default", "map", "axisOrdering", "axisLabels")

    def __init__(
        self,
        *,
        tag=None,
        name: str | None = None,
        labelNames=None,
        values=None,
        default=None,
        hidden=False,
        map: list[tuple[float, float]] | None = None,
        axisOrdering=None,
        axisLabels=None,
    ):
        super().__init__(
            tag=tag,
            name=name,
            labelNames=labelNames,
            hidden=hidden,
            map=map,
            axisOrdering=axisOrdering,
            axisLabels=axisLabels,
        )
        self.default: float = default
        """The default value for this axis, i.e. when a new location is
        created, this is the value this axis will get in user space.

        However, this default value is less important than in continuous axes:

        -  it doesn't define the "neutral" version of outlines from which
           deltas would apply, as this axis does not interpolate.
        -  it doesn't provide the reference glyph set for the designspace, as
           fonts at each value can have different glyph sets.
        """
        self.values: List[float] = values or []
        """List of possible values for this axis. Contrary to continuous axes,
        only the values in this list can be taken by the axis, nothing in-between.
        """

    def map_forward(self, value):
        """Maps value from axis mapping's input to output.

        Returns value unchanged if no mapping entry is found.

        Note: for discrete axes, each value must have its mapping entry, if
        you intend that value to be mapped.
        """
        return next((v for k, v in self.get_validated_map() if k == value), value)

    def map_backward(self, value):
        """Maps value from axis mapping's output to input.

        Returns value unchanged if no mapping entry is found.

        Note: for discrete axes, each value must have its mapping entry, if
        you intend that value to be mapped.
        """
        if isinstance(value, tuple):
            value = value[0]
        return next((k for k, v in self.get_validated_map() if v == value), value)


class AxisLabelDescriptor(SimpleDescriptor):
    """Container for axis label data.

    Analogue of OpenType's STAT data for a single axis (formats 1, 2 and 3).
    All values are user values.
    See: `OTSpec STAT Axis value table, format 1, 2, 3 <https://docs.microsoft.com/en-us/typography/opentype/spec/stat#axis-value-table-format-1>`_

    The STAT format of the Axis value depends on which field are filled-in,
    see :meth:`getFormat`

    .. versionadded:: 5.0
    """

    flavor = "label"
    _attrs = (
        "userMinimum",
        "userValue",
        "userMaximum",
        "name",
        "elidable",
        "olderSibling",
        "linkedUserValue",
        "labelNames",
    )

    def __init__(
        self,
        *,
        name,
        userValue,
        userMinimum=None,
        userMaximum=None,
        elidable=False,
        olderSibling=False,
        linkedUserValue=None,
        labelNames=None,
    ):
        self.userMinimum: Optional[float] = userMinimum
        """STAT field ``rangeMinValue`` (format 2)."""
        self.userValue: float = userValue
        """STAT field ``value`` (format 1, 3) or ``nominalValue`` (format 2)."""
        self.userMaximum: Optional[float] = userMaximum
        """STAT field ``rangeMaxValue`` (format 2)."""
        self.name: str = name
        """Label for this axis location, STAT field ``valueNameID``."""
        self.elidable: bool = elidable
        """STAT flag ``ELIDABLE_AXIS_VALUE_NAME``.

        See: `OTSpec STAT Flags <https://docs.microsoft.com/en-us/typography/opentype/spec/stat#flags>`_
        """
        self.olderSibling: bool = olderSibling
        """STAT flag ``OLDER_SIBLING_FONT_ATTRIBUTE``.

        See: `OTSpec STAT Flags <https://docs.microsoft.com/en-us/typography/opentype/spec/stat#flags>`_
        """
        self.linkedUserValue: Optional[float] = linkedUserValue
        """STAT field ``linkedValue`` (format 3)."""
        self.labelNames: MutableMapping[str, str] = labelNames or {}
        """User-facing translations of this location's label. Keyed by
        ``xml:lang`` code.
        """

    def getFormat(self) -> int:
        """Determine which format of STAT Axis value to use to encode this label.

        ===========  =========  ===========  ===========  ===============
        STAT Format  userValue  userMinimum  userMaximum  linkedUserValue
        ===========  =========  ===========  ===========  ===============
        1            ✅          ❌            ❌            ❌
        2            ✅          ✅            ✅            ❌
        3            ✅          ❌            ❌            ✅
        ===========  =========  ===========  ===========  ===============
        """
        if self.linkedUserValue is not None:
            return 3
        if self.userMinimum is not None or self.userMaximum is not None:
            return 2
        return 1

    @property
    def defaultName(self) -> str:
        """Return the English name from :attr:`labelNames` or the :attr:`name`."""
        return self.labelNames.get("en") or self.name


class LocationLabelDescriptor(SimpleDescriptor):
    """Container for location label data.

    Analogue of OpenType's STAT data for a free-floating location (format 4).
    All values are user values.

    See: `OTSpec STAT Axis value table, format 4 <https://docs.microsoft.com/en-us/typography/opentype/spec/stat#axis-value-table-format-4>`_

    .. versionadded:: 5.0
    """

    flavor = "label"
    _attrs = ("name", "elidable", "olderSibling", "userLocation", "labelNames")

    def __init__(
        self,
        *,
        name,
        userLocation,
        elidable=False,
        olderSibling=False,
        labelNames=None,
    ):
        self.name: str = name
        """Label for this named location, STAT field ``valueNameID``."""
        self.userLocation: SimpleLocationDict = userLocation or {}
        """Location in user coordinates along each axis.

        If an axis is not mentioned, it is assumed to be at its default location.

        .. seealso:: This may be only part of the full location. See:
           :meth:`getFullUserLocation`
        """
        self.elidable: bool = elidable
        """STAT flag ``ELIDABLE_AXIS_VALUE_NAME``.

        See: `OTSpec STAT Flags <https://docs.microsoft.com/en-us/typography/opentype/spec/stat#flags>`_
        """
        self.olderSibling: bool = olderSibling
        """STAT flag ``OLDER_SIBLING_FONT_ATTRIBUTE``.

        See: `OTSpec STAT Flags <https://docs.microsoft.com/en-us/typography/opentype/spec/stat#flags>`_
        """
        self.labelNames: Dict[str, str] = labelNames or {}
        """User-facing translations of this location's label. Keyed by
        xml:lang code.
        """

    @property
    def defaultName(self) -> str:
        """Return the English name from :attr:`labelNames` or the :attr:`name`."""
        return self.labelNames.get("en") or self.name

    def getFullUserLocation(self, doc: "DesignSpaceDocument") -> SimpleLocationDict:
        """Get the complete user location of this label, by combining data
        from the explicit user location and default axis values.

        .. versionadded:: 5.0
        """
        return {
            axis.name: self.userLocation.get(axis.name, axis.default)
            for axis in doc.axes
        }


class VariableFontDescriptor(SimpleDescriptor):
    """Container for variable fonts, sub-spaces of the Designspace.

    Use-cases:

    - From a single DesignSpace with discrete axes, define 1 variable font
      per value on the discrete axes. Before version 5, you would have needed
      1 DesignSpace per such variable font, and a lot of data duplication.
    - From a big variable font with many axes, define subsets of that variable
      font that only include some axes and freeze other axes at a given location.

    .. versionadded:: 5.0
    """

    flavor = "variable-font"
    _attrs = ("filename", "axisSubsets", "lib")

    filename = posixpath_property("_filename")

    def __init__(self, *, name, filename=None, axisSubsets=None, lib=None):
        self.name: str = name
        """string, required. Name of this variable to identify it during the
        build process and from other parts of the document, and also as a
        filename in case the filename property is empty.

        VarLib.
        """
        self.filename: str = filename
        """string, optional. Relative path to the variable font file, **as it is
        in the document**. The file may or may not exist.

        If not specified, the :attr:`name` will be used as a basename for the file.

        .. note::
            This is intended to be a simple filename (basename or stem) only.
            Build tools will only use the basename component and ignore any
            directory separators for security reasons.
        """
        self.axisSubsets: List[
            Union[RangeAxisSubsetDescriptor, ValueAxisSubsetDescriptor]
        ] = (axisSubsets or [])
        """Axis subsets to include in this variable font.

        If an axis is not mentioned, assume that we only want the default
        location of that axis (same as a :class:`ValueAxisSubsetDescriptor`).
        """
        self.lib: MutableMapping[str, Any] = lib or {}
        """Custom data associated with this variable font."""


class RangeAxisSubsetDescriptor(SimpleDescriptor):
    """Subset of a continuous axis to include in a variable font.

    .. versionadded:: 5.0
    """

    flavor = "axis-subset"
    _attrs = ("name", "userMinimum", "userDefault", "userMaximum")

    def __init__(
        self, *, name, userMinimum=-math.inf, userDefault=None, userMaximum=math.inf
    ):
        self.name: str = name
        """Name of the :class:`AxisDescriptor` to subset."""
        self.userMinimum: float = userMinimum
        """New minimum value of the axis in the target variable font.
        If not specified, assume the same minimum value as the full axis.
        (default = ``-math.inf``)
        """
        self.userDefault: Optional[float] = userDefault
        """New default value of the axis in the target variable font.
        If not specified, assume the same default value as the full axis.
        (default = ``None``)
        """
        self.userMaximum: float = userMaximum
        """New maximum value of the axis in the target variable font.
        If not specified, assume the same maximum value as the full axis.
        (default = ``math.inf``)
        """


class ValueAxisSubsetDescriptor(SimpleDescriptor):
    """Single value of a discrete or continuous axis to use in a variable font.

    .. versionadded:: 5.0
    """

    flavor = "axis-subset"
    _attrs = ("name", "userValue")

    def __init__(self, *, name, userValue):
        self.name: str = name
        """Name of the :class:`AxisDescriptor` or :class:`DiscreteAxisDescriptor`
        to "snapshot" or "freeze".
        """
        self.userValue: float = userValue
        """Value in user coordinates at which to freeze the given axis."""


class BaseDocWriter(object):
    _whiteSpace = "    "
    axisDescriptorClass = AxisDescriptor
    discreteAxisDescriptorClass = DiscreteAxisDescriptor
    axisLabelDescriptorClass = AxisLabelDescriptor
    axisMappingDescriptorClass = AxisMappingDescriptor
    locationLabelDescriptorClass = LocationLabelDescriptor
    ruleDescriptorClass = RuleDescriptor
    sourceDescriptorClass = SourceDescriptor
    variableFontDescriptorClass = VariableFontDescriptor
    valueAxisSubsetDescriptorClass = ValueAxisSubsetDescriptor
    rangeAxisSubsetDescriptorClass = RangeAxisSubsetDescriptor
    instanceDescriptorClass = InstanceDescriptor

    @classmethod
    def getAxisDecriptor(cls):
        return cls.axisDescriptorClass()

    @classmethod
    def getAxisMappingDescriptor(cls):
        return cls.axisMappingDescriptorClass()

    @classmethod
    def getSourceDescriptor(cls):
        return cls.sourceDescriptorClass()

    @classmethod
    def getInstanceDescriptor(cls):
        return cls.instanceDescriptorClass()

    @classmethod
    def getRuleDescriptor(cls):
        return cls.ruleDescriptorClass()

    def __init__(self, documentPath, documentObject: DesignSpaceDocument):
        self.path = documentPath
        self.documentObject = documentObject
        self.effectiveFormatTuple = self._getEffectiveFormatTuple()
        self.root = ET.Element("designspace")

    def write(self, pretty=True, encoding="UTF-8", xml_declaration=True):
        self.root.attrib["format"] = ".".join(str(i) for i in self.effectiveFormatTuple)

        if (
            self.documentObject.axes
            or self.documentObject.axisMappings
            or self.documentObject.elidedFallbackName is not None
        ):
            axesElement = ET.Element("axes")
            if self.documentObject.elidedFallbackName is not None:
                axesElement.attrib["elidedfallbackname"] = (
                    self.documentObject.elidedFallbackName
                )
            self.root.append(axesElement)
        for axisObject in self.documentObject.axes:
            self._addAxis(axisObject)

        if self.documentObject.axisMappings:
            mappingsElement = None
            lastGroup = object()
            for mappingObject in self.documentObject.axisMappings:
                if getattr(mappingObject, "groupDescription", None) != lastGroup:
                    if mappingsElement is not None:
                        self.root.findall(".axes")[0].append(mappingsElement)
                    lastGroup = getattr(mappingObject, "groupDescription", None)
                    mappingsElement = ET.Element("mappings")
                    if lastGroup is not None:
                        mappingsElement.attrib["description"] = lastGroup
                self._addAxisMapping(mappingsElement, mappingObject)
            if mappingsElement is not None:
                self.root.findall(".axes")[0].append(mappingsElement)

        if self.documentObject.locationLabels:
            labelsElement = ET.Element("labels")
            for labelObject in self.documentObject.locationLabels:
                self._addLocationLabel(labelsElement, labelObject)
            self.root.append(labelsElement)

        if self.documentObject.rules:
            if getattr(self.documentObject, "rulesProcessingLast", False):
                attributes = {"processing": "last"}
            else:
                attributes = {}
            self.root.append(ET.Element("rules", attributes))
        for ruleObject in self.documentObject.rules:
            self._addRule(ruleObject)

        if self.documentObject.sources:
            self.root.append(ET.Element("sources"))
        for sourceObject in self.documentObject.sources:
            self._addSource(sourceObject)

        if self.documentObject.variableFonts:
            variableFontsElement = ET.Element("variable-fonts")
            for variableFont in self.documentObject.variableFonts:
                self._addVariableFont(variableFontsElement, variableFont)
            self.root.append(variableFontsElement)

        if self.documentObject.instances:
            self.root.append(ET.Element("instances"))
        for instanceObject in self.documentObject.instances:
            self._addInstance(instanceObject)

        if self.documentObject.lib:
            self._addLib(self.root, self.documentObject.lib, 2)

        tree = ET.ElementTree(self.root)
        tree.write(
            self.path,
            encoding=encoding,
            method="xml",
            xml_declaration=xml_declaration,
            pretty_print=pretty,
        )

    def _getEffectiveFormatTuple(self):
        """Try to use the version specified in the document, or a sufficiently
        recent version to be able to encode what the document contains.
        """
        minVersion = self.documentObject.formatTuple
        if (
            any(
                hasattr(axis, "values")
                or axis.axisOrdering is not None
                or axis.axisLabels
                for axis in self.documentObject.axes
            )
            or self.documentObject.locationLabels
            or any(source.localisedFamilyName for source in self.documentObject.sources)
            or self.documentObject.variableFonts
            or any(
                instance.locationLabel or instance.userLocation
                for instance in self.documentObject.instances
            )
        ):
            if minVersion < (5, 0):
                minVersion = (5, 0)
        if self.documentObject.axisMappings:
            if minVersion < (5, 1):
                minVersion = (5, 1)
        return minVersion

    def _makeLocationElement(self, locationObject, name=None):
        """Convert Location dict to a locationElement."""
        locElement = ET.Element("location")
        if name is not None:
            locElement.attrib["name"] = name
        validatedLocation = self.documentObject.newDefaultLocation()
        for axisName, axisValue in locationObject.items():
            if axisName in validatedLocation:
                # only accept values we know
                validatedLocation[axisName] = axisValue
        for dimensionName, dimensionValue in validatedLocation.items():
            dimElement = ET.Element("dimension")
            dimElement.attrib["name"] = dimensionName
            if type(dimensionValue) == tuple:
                dimElement.attrib["xvalue"] = self.intOrFloat(dimensionValue[0])
                dimElement.attrib["yvalue"] = self.intOrFloat(dimensionValue[1])
            else:
                dimElement.attrib["xvalue"] = self.intOrFloat(dimensionValue)
            locElement.append(dimElement)
        return locElement, validatedLocation

    def intOrFloat(self, num):
        if int(num) == num:
            return "%d" % num
        return ("%f" % num).rstrip("0").rstrip(".")

    def _addRule(self, ruleObject):
        ruleElement = ET.Element("rule")
        if ruleObject.name is not None:
            ruleElement.attrib["name"] = ruleObject.name
        for conditions in ruleObject.conditionSets:
            conditionsetElement = ET.Element("conditionset")
            for cond in conditions:
                if cond.get("minimum") is None and cond.get("maximum") is None:
                    # neither is defined, don't add this condition
                    continue
                conditionElement = ET.Element("condition")
                conditionElement.attrib["name"] = cond.get("name")
                if cond.get("minimum") is not None:
                    conditionElement.attrib["minimum"] = self.intOrFloat(
                        cond.get("minimum")
                    )
                if cond.get("maximum") is not None:
                    conditionElement.attrib["maximum"] = self.intOrFloat(
                        cond.get("maximum")
                    )
                conditionsetElement.append(conditionElement)
            # Serialize the conditionset even if it is empty, as this is the
            # canonical way of defining a rule that is always true.
            ruleElement.append(conditionsetElement)
        for sub in ruleObject.subs:
            subElement = ET.Element("sub")
            subElement.attrib["name"] = sub[0]
            subElement.attrib["with"] = sub[1]
            ruleElement.append(subElement)
        if len(ruleElement):
            self.root.findall(".rules")[0].append(ruleElement)

    def _addAxis(self, axisObject):
        axisElement = ET.Element("axis")
        axisElement.attrib["tag"] = axisObject.tag
        axisElement.attrib["name"] = axisObject.name
        self._addLabelNames(axisElement, axisObject.labelNames)
        if axisObject.map:
            for inputValue, outputValue in axisObject.map:
                mapElement = ET.Element("map")
                mapElement.attrib["input"] = self.intOrFloat(inputValue)
                mapElement.attrib["output"] = self.intOrFloat(outputValue)
                axisElement.append(mapElement)
        if axisObject.axisOrdering is not None or axisObject.axisLabels:
            labelsElement = ET.Element("labels")
            if axisObject.axisOrdering is not None:
                labelsElement.attrib["ordering"] = str(axisObject.axisOrdering)
            for label in axisObject.axisLabels:
                self._addAxisLabel(labelsElement, label)
            axisElement.append(labelsElement)
        if hasattr(axisObject, "minimum"):
            axisElement.attrib["minimum"] = self.intOrFloat(axisObject.minimum)
            axisElement.attrib["maximum"] = self.intOrFloat(axisObject.maximum)
        elif hasattr(axisObject, "values"):
            axisElement.attrib["values"] = " ".join(
                self.intOrFloat(v) for v in axisObject.values
            )
        axisElement.attrib["default"] = self.intOrFloat(axisObject.default)
        if axisObject.hidden:
            axisElement.attrib["hidden"] = "1"
        self.root.findall(".axes")[0].append(axisElement)

    def _addAxisMapping(self, mappingsElement, mappingObject):
        mappingElement = ET.Element("mapping")
        if getattr(mappingObject, "description", None) is not None:
            mappingElement.attrib["description"] = mappingObject.description
        for what in ("inputLocation", "outputLocation"):
            whatObject = getattr(mappingObject, what, None)
            if whatObject is None:
                continue
            whatElement = ET.Element(what[:-8])
            mappingElement.append(whatElement)

            for name, value in whatObject.items():
                dimensionElement = ET.Element("dimension")
                dimensionElement.attrib["name"] = name
                dimensionElement.attrib["xvalue"] = self.intOrFloat(value)
                whatElement.append(dimensionElement)

        mappingsElement.append(mappingElement)

    def _addAxisLabel(
        self, axisElement: ET.Element, label: AxisLabelDescriptor
    ) -> None:
        labelElement = ET.Element("label")
        labelElement.attrib["uservalue"] = self.intOrFloat(label.userValue)
        if label.userMinimum is not None:
            labelElement.attrib["userminimum"] = self.intOrFloat(label.userMinimum)
        if label.userMaximum is not None:
            labelElement.attrib["usermaximum"] = self.intOrFloat(label.userMaximum)
        labelElement.attrib["name"] = label.name
        if label.elidable:
            labelElement.attrib["elidable"] = "true"
        if label.olderSibling:
            labelElement.attrib["oldersibling"] = "true"
        if label.linkedUserValue is not None:
            labelElement.attrib["linkeduservalue"] = self.intOrFloat(
                label.linkedUserValue
            )
        self._addLabelNames(labelElement, label.labelNames)
        axisElement.append(labelElement)

    def _addLabelNames(self, parentElement, labelNames):
        for languageCode, labelName in sorted(labelNames.items()):
            languageElement = ET.Element("labelname")
            languageElement.attrib[XML_LANG] = languageCode
            languageElement.text = labelName
            parentElement.append(languageElement)

    def _addLocationLabel(
        self, parentElement: ET.Element, label: LocationLabelDescriptor
    ) -> None:
        labelElement = ET.Element("label")
        labelElement.attrib["name"] = label.name
        if label.elidable:
            labelElement.attrib["elidable"] = "true"
        if label.olderSibling:
            labelElement.attrib["oldersibling"] = "true"
        self._addLabelNames(labelElement, label.labelNames)
        self._addLocationElement(labelElement, userLocation=label.userLocation)
        parentElement.append(labelElement)

    def _addLocationElement(
        self,
        parentElement,
        *,
        designLocation: AnisotropicLocationDict = None,
        userLocation: SimpleLocationDict = None,
    ):
        locElement = ET.Element("location")
        for axis in self.documentObject.axes:
            if designLocation is not None and axis.name in designLocation:
                dimElement = ET.Element("dimension")
                dimElement.attrib["name"] = axis.name
                value = designLocation[axis.name]
                if isinstance(value, tuple):
                    dimElement.attrib["xvalue"] = self.intOrFloat(value[0])
                    dimElement.attrib["yvalue"] = self.intOrFloat(value[1])
                else:
                    dimElement.attrib["xvalue"] = self.intOrFloat(value)
                locElement.append(dimElement)
            elif userLocation is not None and axis.name in userLocation:
                dimElement = ET.Element("dimension")
                dimElement.attrib["name"] = axis.name
                value = userLocation[axis.name]
                dimElement.attrib["uservalue"] = self.intOrFloat(value)
                locElement.append(dimElement)
        if len(locElement) > 0:
            parentElement.append(locElement)

    def _addInstance(self, instanceObject):
        instanceElement = ET.Element("instance")
        if instanceObject.name is not None:
            instanceElement.attrib["name"] = instanceObject.name
        if instanceObject.locationLabel is not None:
            instanceElement.attrib["location"] = instanceObject.locationLabel
        if instanceObject.familyName is not None:
            instanceElement.attrib["familyname"] = instanceObject.familyName
        if instanceObject.styleName is not None:
            instanceElement.attrib["stylename"] = instanceObject.styleName
        # add localisations
        if instanceObject.localisedStyleName:
            languageCodes = list(instanceObject.localisedStyleName.keys())
            languageCodes.sort()
            for code in languageCodes:
                if code == "en":
                    continue  # already stored in the element attribute
                localisedStyleNameElement = ET.Element("stylename")
                localisedStyleNameElement.attrib[XML_LANG] = code
                localisedStyleNameElement.text = instanceObject.getStyleName(code)
                instanceElement.append(localisedStyleNameElement)
        if instanceObject.localisedFamilyName:
            languageCodes = list(instanceObject.localisedFamilyName.keys())
            languageCodes.sort()
            for code in languageCodes:
                if code == "en":
                    continue  # already stored in the element attribute
                localisedFamilyNameElement = ET.Element("familyname")
                localisedFamilyNameElement.attrib[XML_LANG] = code
                localisedFamilyNameElement.text = instanceObject.getFamilyName(code)
                instanceElement.append(localisedFamilyNameElement)
        if instanceObject.localisedStyleMapStyleName:
            languageCodes = list(instanceObject.localisedStyleMapStyleName.keys())
            languageCodes.sort()
            for code in languageCodes:
                if code == "en":
                    continue
                localisedStyleMapStyleNameElement = ET.Element("stylemapstylename")
                localisedStyleMapStyleNameElement.attrib[XML_LANG] = code
                localisedStyleMapStyleNameElement.text = (
                    instanceObject.getStyleMapStyleName(code)
                )
                instanceElement.append(localisedStyleMapStyleNameElement)
        if instanceObject.localisedStyleMapFamilyName:
            languageCodes = list(instanceObject.localisedStyleMapFamilyName.keys())
            languageCodes.sort()
            for code in languageCodes:
                if code == "en":
                    continue
                localisedStyleMapFamilyNameElement = ET.Element("stylemapfamilyname")
                localisedStyleMapFamilyNameElement.attrib[XML_LANG] = code
                localisedStyleMapFamilyNameElement.text = (
                    instanceObject.getStyleMapFamilyName(code)
                )
                instanceElement.append(localisedStyleMapFamilyNameElement)

        if self.effectiveFormatTuple >= (5, 0):
            if instanceObject.locationLabel is None:
                self._addLocationElement(
                    instanceElement,
                    designLocation=instanceObject.designLocation,
                    userLocation=instanceObject.userLocation,
                )
        else:
            # Pre-version 5.0 code was validating and filling in the location
            # dict while writing it out, as preserved below.
            if instanceObject.location is not None:
                locationElement, instanceObject.location = self._makeLocationElement(
                    instanceObject.location
                )
                instanceElement.append(locationElement)
        if instanceObject.filename is not None:
            instanceElement.attrib["filename"] = instanceObject.filename
        if instanceObject.postScriptFontName is not None:
            instanceElement.attrib["postscriptfontname"] = (
                instanceObject.postScriptFontName
            )
        if instanceObject.styleMapFamilyName is not None:
            instanceElement.attrib["stylemapfamilyname"] = (
                instanceObject.styleMapFamilyName
            )
        if instanceObject.styleMapStyleName is not None:
            instanceElement.attrib["stylemapstylename"] = (
                instanceObject.styleMapStyleName
            )
        if self.effectiveFormatTuple < (5, 0):
            # Deprecated members as of version 5.0
            if instanceObject.glyphs:
                if instanceElement.findall(".glyphs") == []:
                    glyphsElement = ET.Element("glyphs")
                    instanceElement.append(glyphsElement)
                glyphsElement = instanceElement.findall(".glyphs")[0]
                for glyphName, data in sorted(instanceObject.glyphs.items()):
                    glyphElement = self._writeGlyphElement(
                        instanceElement, instanceObject, glyphName, data
                    )
                    glyphsElement.append(glyphElement)
            if instanceObject.kerning:
                kerningElement = ET.Element("kerning")
                instanceElement.append(kerningElement)
            if instanceObject.info:
                infoElement = ET.Element("info")
                instanceElement.append(infoElement)
        self._addLib(instanceElement, instanceObject.lib, 4)
        self.root.findall(".instances")[0].append(instanceElement)

    def _addSource(self, sourceObject):
        sourceElement = ET.Element("source")
        if sourceObject.filename is not None:
            sourceElement.attrib["filename"] = sourceObject.filename
        if sourceObject.name is not None:
            if sourceObject.name.find("temp_master") != 0:
                # do not save temporary source names
                sourceElement.attrib["name"] = sourceObject.name
        if sourceObject.familyName is not None:
            sourceElement.attrib["familyname"] = sourceObject.familyName
        if sourceObject.styleName is not None:
            sourceElement.attrib["stylename"] = sourceObject.styleName
        if sourceObject.layerName is not None:
            sourceElement.attrib["layer"] = sourceObject.layerName
        if sourceObject.localisedFamilyName:
            languageCodes = list(sourceObject.localisedFamilyName.keys())
            languageCodes.sort()
            for code in languageCodes:
                if code == "en":
                    continue  # already stored in the element attribute
                localisedFamilyNameElement = ET.Element("familyname")
                localisedFamilyNameElement.attrib[XML_LANG] = code
                localisedFamilyNameElement.text = sourceObject.getFamilyName(code)
                sourceElement.append(localisedFamilyNameElement)
        if sourceObject.copyLib:
            libElement = ET.Element("lib")
            libElement.attrib["copy"] = "1"
            sourceElement.append(libElement)
        if sourceObject.copyGroups:
            groupsElement = ET.Element("groups")
            groupsElement.attrib["copy"] = "1"
            sourceElement.append(groupsElement)
        if sourceObject.copyFeatures:
            featuresElement = ET.Element("features")
            featuresElement.attrib["copy"] = "1"
            sourceElement.append(featuresElement)
        if sourceObject.copyInfo or sourceObject.muteInfo:
            infoElement = ET.Element("info")
            if sourceObject.copyInfo:
                infoElement.attrib["copy"] = "1"
            if sourceObject.muteInfo:
                infoElement.attrib["mute"] = "1"
            sourceElement.append(infoElement)
        if sourceObject.muteKerning:
            kerningElement = ET.Element("kerning")
            kerningElement.attrib["mute"] = "1"
            sourceElement.append(kerningElement)
        if sourceObject.mutedGlyphNames:
            for name in sourceObject.mutedGlyphNames:
                glyphElement = ET.Element("glyph")
                glyphElement.attrib["name"] = name
                glyphElement.attrib["mute"] = "1"
                sourceElement.append(glyphElement)
        if self.effectiveFormatTuple >= (5, 0):
            self._addLocationElement(
                sourceElement, designLocation=sourceObject.location
            )
        else:
            # Pre-version 5.0 code was validating and filling in the location
            # dict while writing it out, as preserved below.
            locationElement, sourceObject.location = self._makeLocationElement(
                sourceObject.location
            )
            sourceElement.append(locationElement)
        self.root.findall(".sources")[0].append(sourceElement)

    def _addVariableFont(
        self, parentElement: ET.Element, vf: VariableFontDescriptor
    ) -> None:
        vfElement = ET.Element("variable-font")
        vfElement.attrib["name"] = vf.name
        if vf.filename is not None:
            vfElement.attrib["filename"] = vf.filename
        if vf.axisSubsets:
            subsetsElement = ET.Element("axis-subsets")
            for subset in vf.axisSubsets:
                subsetElement = ET.Element("axis-subset")
                subsetElement.attrib["name"] = subset.name
                # Mypy doesn't support narrowing union types via hasattr()
                # https://mypy.readthedocs.io/en/stable/type_narrowing.html
                # TODO(Python 3.10): use TypeGuard
                if hasattr(subset, "userMinimum"):
                    subset = cast(RangeAxisSubsetDescriptor, subset)
                    if subset.userMinimum != -math.inf:
                        subsetElement.attrib["userminimum"] = self.intOrFloat(
                            subset.userMinimum
                        )
                    if subset.userMaximum != math.inf:
                        subsetElement.attrib["usermaximum"] = self.intOrFloat(
                            subset.userMaximum
                        )
                    if subset.userDefault is not None:
                        subsetElement.attrib["userdefault"] = self.intOrFloat(
                            subset.userDefault
                        )
                elif hasattr(subset, "userValue"):
                    subset = cast(ValueAxisSubsetDescriptor, subset)
                    subsetElement.attrib["uservalue"] = self.intOrFloat(
                        subset.userValue
                    )
                subsetsElement.append(subsetElement)
            vfElement.append(subsetsElement)
        self._addLib(vfElement, vf.lib, 4)
        parentElement.append(vfElement)

    def _addLib(self, parentElement: ET.Element, data: Any, indent_level: int) -> None:
        if not data:
            return
        libElement = ET.Element("lib")
        libElement.append(plistlib.totree(data, indent_level=indent_level))
        parentElement.append(libElement)

    def _writeGlyphElement(self, instanceElement, instanceObject, glyphName, data):
        glyphElement = ET.Element("glyph")
        if data.get("mute"):
            glyphElement.attrib["mute"] = "1"
        if data.get("unicodes") is not None:
            glyphElement.attrib["unicode"] = " ".join(
                [hex(u) for u in data.get("unicodes")]
            )
        if data.get("instanceLocation") is not None:
            locationElement, data["instanceLocation"] = self._makeLocationElement(
                data.get("instanceLocation")
            )
            glyphElement.append(locationElement)
        if glyphName is not None:
            glyphElement.attrib["name"] = glyphName
        if data.get("note") is not None:
            noteElement = ET.Element("note")
            noteElement.text = data.get("note")
            glyphElement.append(noteElement)
        if data.get("masters") is not None:
            mastersElement = ET.Element("masters")
            for m in data.get("masters"):
                masterElement = ET.Element("master")
                if m.get("glyphName") is not None:
                    masterElement.attrib["glyphname"] = m.get("glyphName")
                if m.get("font") is not None:
                    masterElement.attrib["source"] = m.get("font")
                if m.get("location") is not None:
                    locationElement, m["location"] = self._makeLocationElement(
                        m.get("location")
                    )
                    masterElement.append(locationElement)
                mastersElement.append(masterElement)
            glyphElement.append(mastersElement)
        return glyphElement


class BaseDocReader(LogMixin):
    axisDescriptorClass = AxisDescriptor
    discreteAxisDescriptorClass = DiscreteAxisDescriptor
    axisLabelDescriptorClass = AxisLabelDescriptor
    axisMappingDescriptorClass = AxisMappingDescriptor
    locationLabelDescriptorClass = LocationLabelDescriptor
    ruleDescriptorClass = RuleDescriptor
    sourceDescriptorClass = SourceDescriptor
    variableFontsDescriptorClass = VariableFontDescriptor
    valueAxisSubsetDescriptorClass = ValueAxisSubsetDescriptor
    rangeAxisSubsetDescriptorClass = RangeAxisSubsetDescriptor
    instanceDescriptorClass = InstanceDescriptor

    def __init__(self, documentPath, documentObject):
        self.path = documentPath
        self.documentObject = documentObject
        tree = ET.parse(self.path, parser=ET.XMLParser())
        self.root = tree.getroot()
        self.documentObject.formatVersion = self.root.attrib.get("format", "3.0")
        self._axes = []
        self.rules = []
        self.sources = []
        self.instances = []
        self.axisDefaults = {}
        self._strictAxisNames = True

    @classmethod
    def fromstring(cls, string, documentObject):
        f = BytesIO(tobytes(string, encoding="utf-8"))
        self = cls(f, documentObject)
        self.path = None
        return self

    def read(self):
        self.readAxes()
        self.readLabels()
        self.readRules()
        self.readVariableFonts()
        self.readSources()
        self.readInstances()
        self.readLib()

    def readRules(self):
        # we also need to read any conditions that are outside of a condition set.
        rules = []
        rulesElement = self.root.find(".rules")
        if rulesElement is not None:
            processingValue = rulesElement.attrib.get("processing", "first")
            if processingValue not in {"first", "last"}:
                raise DesignSpaceDocumentError(
                    "<rules> processing attribute value is not valid: %r, "
                    "expected 'first' or 'last'" % processingValue
                )
            self.documentObject.rulesProcessingLast = processingValue == "last"
        for ruleElement in self.root.findall(".rules/rule"):
            ruleObject = self.ruleDescriptorClass()
            ruleName = ruleObject.name = ruleElement.attrib.get("name")
            # read any stray conditions outside a condition set
            externalConditions = self._readConditionElements(
                ruleElement,
                ruleName,
            )
            if externalConditions:
                ruleObject.conditionSets.append(externalConditions)
                self.log.info(
                    "Found stray rule conditions outside a conditionset. "
                    "Wrapped them in a new conditionset."
                )
            # read the conditionsets
            for conditionSetElement in ruleElement.findall(".conditionset"):
                conditionSet = self._readConditionElements(
                    conditionSetElement,
                    ruleName,
                )
                if conditionSet is not None:
                    ruleObject.conditionSets.append(conditionSet)
            for subElement in ruleElement.findall(".sub"):
                a = subElement.attrib["name"]
                b = subElement.attrib["with"]
                ruleObject.subs.append((a, b))
            rules.append(ruleObject)
        self.documentObject.rules = rules

    def _readConditionElements(self, parentElement, ruleName=None):
        cds = []
        for conditionElement in parentElement.findall(".condition"):
            cd = {}
            cdMin = conditionElement.attrib.get("minimum")
            if cdMin is not None:
                cd["minimum"] = float(cdMin)
            else:
                # will allow these to be None, assume axis.minimum
                cd["minimum"] = None
            cdMax = conditionElement.attrib.get("maximum")
            if cdMax is not None:
                cd["maximum"] = float(cdMax)
            else:
                # will allow these to be None, assume axis.maximum
                cd["maximum"] = None
            cd["name"] = conditionElement.attrib.get("name")
            # # test for things
            if cd.get("minimum") is None and cd.get("maximum") is None:
                raise DesignSpaceDocumentError(
                    "condition missing required minimum or maximum in rule"
                    + (" '%s'" % ruleName if ruleName is not None else "")
                )
            cds.append(cd)
        return cds

    def readAxes(self):
        # read the axes elements, including the warp map.
        axesElement = self.root.find(".axes")
        if axesElement is not None and "elidedfallbackname" in axesElement.attrib:
            self.documentObject.elidedFallbackName = axesElement.attrib[
                "elidedfallbackname"
            ]
        axisElements = self.root.findall(".axes/axis")
        if not axisElements:
            return
        for axisElement in axisElements:
            if (
                self.documentObject.formatTuple >= (5, 0)
                and "values" in axisElement.attrib
            ):
                axisObject = self.discreteAxisDescriptorClass()
                axisObject.values = [
                    float(s) for s in axisElement.attrib["values"].split(" ")
                ]
            else:
                axisObject = self.axisDescriptorClass()
                axisObject.minimum = float(axisElement.attrib.get("minimum"))
                axisObject.maximum = float(axisElement.attrib.get("maximum"))
            axisObject.default = float(axisElement.attrib.get("default"))
            axisObject.name = axisElement.attrib.get("name")
            if axisElement.attrib.get("hidden", False):
                axisObject.hidden = True
            axisObject.tag = axisElement.attrib.get("tag")
            for mapElement in axisElement.findall("map"):
                a = float(mapElement.attrib["input"])
                b = float(mapElement.attrib["output"])
                axisObject.map.append((a, b))
            for labelNameElement in axisElement.findall("labelname"):
                # Note: elementtree reads the "xml:lang" attribute name as
                # '{http://www.w3.org/XML/1998/namespace}lang'
                for key, lang in labelNameElement.items():
                    if key == XML_LANG:
                        axisObject.labelNames[lang] = tostr(labelNameElement.text)
            labelElement = axisElement.find(".labels")
            if labelElement is not None:
                if "ordering" in labelElement.attrib:
                    axisObject.axisOrdering = int(labelElement.attrib["ordering"])
                for label in labelElement.findall(".label"):
                    axisObject.axisLabels.append(self.readAxisLabel(label))
            self.documentObject.axes.append(axisObject)
            self.axisDefaults[axisObject.name] = axisObject.default

        self.documentObject.axisMappings = []
        for mappingsElement in self.root.findall(".axes/mappings"):
            groupDescription = mappingsElement.attrib.get("description")
            for mappingElement in mappingsElement.findall("mapping"):
                description = mappingElement.attrib.get("description")
                inputElement = mappingElement.find("input")
                outputElement = mappingElement.find("output")
                inputLoc = {}
                outputLoc = {}
                for dimElement in inputElement.findall(".dimension"):
                    name = dimElement.attrib["name"]
                    value = float(dimElement.attrib["xvalue"])
                    inputLoc[name] = value
                for dimElement in outputElement.findall(".dimension"):
                    name = dimElement.attrib["name"]
                    value = float(dimElement.attrib["xvalue"])
                    outputLoc[name] = value
                axisMappingObject = self.axisMappingDescriptorClass(
                    inputLocation=inputLoc,
                    outputLocation=outputLoc,
                    description=description,
                    groupDescription=groupDescription,
                )
                self.documentObject.axisMappings.append(axisMappingObject)

    def readAxisLabel(self, element: ET.Element):
        xml_attrs = {
            "userminimum",
            "uservalue",
            "usermaximum",
            "name",
            "elidable",
            "oldersibling",
            "linkeduservalue",
        }
        unknown_attrs = set(element.attrib) - xml_attrs
        if unknown_attrs:
            raise DesignSpaceDocumentError(
                f"label element contains unknown attributes: {', '.join(unknown_attrs)}"
            )

        name = element.get("name")
        if name is None:
            raise DesignSpaceDocumentError("label element must have a name attribute.")
        valueStr = element.get("uservalue")
        if valueStr is None:
            raise DesignSpaceDocumentError(
                "label element must have a uservalue attribute."
            )
        value = float(valueStr)
        minimumStr = element.get("userminimum")
        minimum = float(minimumStr) if minimumStr is not None else None
        maximumStr = element.get("usermaximum")
        maximum = float(maximumStr) if maximumStr is not None else None
        linkedValueStr = element.get("linkeduservalue")
        linkedValue = float(linkedValueStr) if linkedValueStr is not None else None
        elidable = True if element.get("elidable") == "true" else False
        olderSibling = True if element.get("oldersibling") == "true" else False
        labelNames = {
            lang: label_name.text or ""
            for label_name in element.findall("labelname")
            for attr, lang in label_name.items()
            if attr == XML_LANG
            # Note: elementtree reads the "xml:lang" attribute name as
            # '{http://www.w3.org/XML/1998/namespace}lang'
        }
        return self.axisLabelDescriptorClass(
            name=name,
            userValue=value,
            userMinimum=minimum,
            userMaximum=maximum,
            elidable=elidable,
            olderSibling=olderSibling,
            linkedUserValue=linkedValue,
            labelNames=labelNames,
        )

    def readLabels(self):
        if self.documentObject.formatTuple < (5, 0):
            return

        xml_attrs = {"name", "elidable", "oldersibling"}
        for labelElement in self.root.findall(".labels/label"):
            unknown_attrs = set(labelElement.attrib) - xml_attrs
            if unknown_attrs:
                raise DesignSpaceDocumentError(
                    f"Label element contains unknown attributes: {', '.join(unknown_attrs)}"
                )

            name = labelElement.get("name")
            if name is None:
                raise DesignSpaceDocumentError(
                    "label element must have a name attribute."
                )
            designLocation, userLocation = self.locationFromElement(labelElement)
            if designLocation:
                raise DesignSpaceDocumentError(
                    f'<label> element "{name}" must only have user locations (using uservalue="").'
                )
            elidable = True if labelElement.get("elidable") == "true" else False
            olderSibling = True if labelElement.get("oldersibling") == "true" else False
            labelNames = {
                lang: label_name.text or ""
                for label_name in labelElement.findall("labelname")
                for attr, lang in label_name.items()
                if attr == XML_LANG
                # Note: elementtree reads the "xml:lang" attribute name as
                # '{http://www.w3.org/XML/1998/namespace}lang'
            }
            locationLabel = self.locationLabelDescriptorClass(
                name=name,
                userLocation=userLocation,
                elidable=elidable,
                olderSibling=olderSibling,
                labelNames=labelNames,
            )
            self.documentObject.locationLabels.append(locationLabel)

    def readVariableFonts(self):
        if self.documentObject.formatTuple < (5, 0):
            return

        xml_attrs = {"name", "filename"}
        for variableFontElement in self.root.findall(".variable-fonts/variable-font"):
            unknown_attrs = set(variableFontElement.attrib) - xml_attrs
            if unknown_attrs:
                raise DesignSpaceDocumentError(
                    f"variable-font element contains unknown attributes: {', '.join(unknown_attrs)}"
                )

            name = variableFontElement.get("name")
            if name is None:
                raise DesignSpaceDocumentError(
                    "variable-font element must have a name attribute."
                )

            filename = variableFontElement.get("filename")

            axisSubsetsElement = variableFontElement.find(".axis-subsets")
            if axisSubsetsElement is None:
                raise DesignSpaceDocumentError(
                    "variable-font element must contain an axis-subsets element."
                )
            axisSubsets = []
            for axisSubset in axisSubsetsElement.iterfind(".axis-subset"):
                axisSubsets.append(self.readAxisSubset(axisSubset))

            lib = None
            libElement = variableFontElement.find(".lib")
            if libElement is not None:
                lib = plistlib.fromtree(libElement[0])

            variableFont = self.variableFontsDescriptorClass(
                name=name,
                filename=filename,
                axisSubsets=axisSubsets,
                lib=lib,
            )
            self.documentObject.variableFonts.append(variableFont)

    def readAxisSubset(self, element: ET.Element):
        if "uservalue" in element.attrib:
            xml_attrs = {"name", "uservalue"}
            unknown_attrs = set(element.attrib) - xml_attrs
            if unknown_attrs:
                raise DesignSpaceDocumentError(
                    f"axis-subset element contains unknown attributes: {', '.join(unknown_attrs)}"
                )

            name = element.get("name")
            if name is None:
                raise DesignSpaceDocumentError(
                    "axis-subset element must have a name attribute."
                )
            userValueStr = element.get("uservalue")
            if userValueStr is None:
                raise DesignSpaceDocumentError(
                    "The axis-subset element for a discrete subset must have a uservalue attribute."
                )
            userValue = float(userValueStr)

            return self.valueAxisSubsetDescriptorClass(name=name, userValue=userValue)
        else:
            xml_attrs = {"name", "userminimum", "userdefault", "usermaximum"}
            unknown_attrs = set(element.attrib) - xml_attrs
            if unknown_attrs:
                raise DesignSpaceDocumentError(
                    f"axis-subset element contains unknown attributes: {', '.join(unknown_attrs)}"
                )

            name = element.get("name")
            if name is None:
                raise DesignSpaceDocumentError(
                    "axis-subset element must have a name attribute."
                )

            userMinimum = element.get("userminimum")
            userDefault = element.get("userdefault")
            userMaximum = element.get("usermaximum")
            if (
                userMinimum is not None
                and userDefault is not None
                and userMaximum is not None
            ):
                return self.rangeAxisSubsetDescriptorClass(
                    name=name,
                    userMinimum=float(userMinimum),
                    userDefault=float(userDefault),
                    userMaximum=float(userMaximum),
                )
            if all(v is None for v in (userMinimum, userDefault, userMaximum)):
                return self.rangeAxisSubsetDescriptorClass(name=name)

            raise DesignSpaceDocumentError(
                "axis-subset element must have min/max/default values or none at all."
            )

    def readSources(self):
        for sourceCount, sourceElement in enumerate(
            self.root.findall(".sources/source")
        ):
            filename = sourceElement.attrib.get("filename")
            if filename is not None and self.path is not None:
                sourcePath = os.path.abspath(
                    os.path.join(os.path.dirname(self.path), filename)
                )
            else:
                sourcePath = None
            sourceName = sourceElement.attrib.get("name")
            if sourceName is None:
                # add a temporary source name
                sourceName = "temp_master.%d" % (sourceCount)
            sourceObject = self.sourceDescriptorClass()
            sourceObject.path = sourcePath  # absolute path to the ufo source
            sourceObject.filename = filename  # path as it is stored in the document
            sourceObject.name = sourceName
            familyName = sourceElement.attrib.get("familyname")
            if familyName is not None:
                sourceObject.familyName = familyName
            styleName = sourceElement.attrib.get("stylename")
            if styleName is not None:
                sourceObject.styleName = styleName
            for familyNameElement in sourceElement.findall("familyname"):
                for key, lang in familyNameElement.items():
                    if key == XML_LANG:
                        familyName = familyNameElement.text
                        sourceObject.setFamilyName(familyName, lang)
            designLocation, userLocation = self.locationFromElement(sourceElement)
            if userLocation:
                raise DesignSpaceDocumentError(
                    f'<source> element "{sourceName}" must only have design locations (using xvalue="").'
                )
            sourceObject.location = designLocation
            layerName = sourceElement.attrib.get("layer")
            if layerName is not None:
                sourceObject.layerName = layerName
            for libElement in sourceElement.findall(".lib"):
                if libElement.attrib.get("copy") == "1":
                    sourceObject.copyLib = True
            for groupsElement in sourceElement.findall(".groups"):
                if groupsElement.attrib.get("copy") == "1":
                    sourceObject.copyGroups = True
            for infoElement in sourceElement.findall(".info"):
                if infoElement.attrib.get("copy") == "1":
                    sourceObject.copyInfo = True
                if infoElement.attrib.get("mute") == "1":
                    sourceObject.muteInfo = True
            for featuresElement in sourceElement.findall(".features"):
                if featuresElement.attrib.get("copy") == "1":
                    sourceObject.copyFeatures = True
            for glyphElement in sourceElement.findall(".glyph"):
                glyphName = glyphElement.attrib.get("name")
                if glyphName is None:
                    continue
                if glyphElement.attrib.get("mute") == "1":
                    sourceObject.mutedGlyphNames.append(glyphName)
            for kerningElement in sourceElement.findall(".kerning"):
                if kerningElement.attrib.get("mute") == "1":
                    sourceObject.muteKerning = True
            self.documentObject.sources.append(sourceObject)

    def locationFromElement(self, element):
        """Read a nested ``<location>`` element inside the given ``element``.

        .. versionchanged:: 5.0
           Return a tuple of (designLocation, userLocation)
        """
        elementLocation = (None, None)
        for locationElement in element.findall(".location"):
            elementLocation = self.readLocationElement(locationElement)
            break
        return elementLocation

    def readLocationElement(self, locationElement):
        """Read a ``<location>`` element.

        .. versionchanged:: 5.0
           Return a tuple of (designLocation, userLocation)
        """
        if self._strictAxisNames and not self.documentObject.axes:
            raise DesignSpaceDocumentError("No axes defined")
        userLoc = {}
        designLoc = {}
        for dimensionElement in locationElement.findall(".dimension"):
            dimName = dimensionElement.attrib.get("name")
            if self._strictAxisNames and dimName not in self.axisDefaults:
                # In case the document contains no axis definitions,
                self.log.warning('Location with undefined axis: "%s".', dimName)
                continue
            userValue = xValue = yValue = None
            try:
                userValue = dimensionElement.attrib.get("uservalue")
                if userValue is not None:
                    userValue = float(userValue)
            except ValueError:
                self.log.warning(
                    "ValueError in readLocation userValue %3.3f", userValue
                )
            try:
                xValue = dimensionElement.attrib.get("xvalue")
                if xValue is not None:
                    xValue = float(xValue)
            except ValueError:
                self.log.warning("ValueError in readLocation xValue %3.3f", xValue)
            try:
                yValue = dimensionElement.attrib.get("yvalue")
                if yValue is not None:
                    yValue = float(yValue)
            except ValueError:
                self.log.warning("ValueError in readLocation yValue %3.3f", yValue)
            if userValue is None == xValue is None:
                raise DesignSpaceDocumentError(
                    f'Exactly one of uservalue="" or xvalue="" must be provided for location dimension "{dimName}"'
                )
            if yValue is not None:
                if xValue is None:
                    raise DesignSpaceDocumentError(
                        f'Missing xvalue="" for the location dimension "{dimName}"" with yvalue="{yValue}"'
                    )
                designLoc[dimName] = (xValue, yValue)
            elif xValue is not None:
                designLoc[dimName] = xValue
            else:
                userLoc[dimName] = userValue
        return designLoc, userLoc

    def readInstances(self, makeGlyphs=True, makeKerning=True, makeInfo=True):
        instanceElements = self.root.findall(".instances/instance")
        for instanceElement in instanceElements:
            self._readSingleInstanceElement(
                instanceElement,
                makeGlyphs=makeGlyphs,
                makeKerning=makeKerning,
                makeInfo=makeInfo,
            )

    def _readSingleInstanceElement(
        self, instanceElement, makeGlyphs=True, makeKerning=True, makeInfo=True
    ):
        filename = instanceElement.attrib.get("filename")
        if filename is not None and self.documentObject.path is not None:
            instancePath = os.path.join(
                os.path.dirname(self.documentObject.path), filename
            )
        else:
            instancePath = None
        instanceObject = self.instanceDescriptorClass()
        instanceObject.path = instancePath  # absolute path to the instance
        instanceObject.filename = filename  # path as it is stored in the document
        name = instanceElement.attrib.get("name")
        if name is not None:
            instanceObject.name = name
        familyname = instanceElement.attrib.get("familyname")
        if familyname is not None:
            instanceObject.familyName = familyname
        stylename = instanceElement.attrib.get("stylename")
        if stylename is not None:
            instanceObject.styleName = stylename
        postScriptFontName = instanceElement.attrib.get("postscriptfontname")
        if postScriptFontName is not None:
            instanceObject.postScriptFontName = postScriptFontName
        styleMapFamilyName = instanceElement.attrib.get("stylemapfamilyname")
        if styleMapFamilyName is not None:
            instanceObject.styleMapFamilyName = styleMapFamilyName
        styleMapStyleName = instanceElement.attrib.get("stylemapstylename")
        if styleMapStyleName is not None:
            instanceObject.styleMapStyleName = styleMapStyleName
        # read localised names
        for styleNameElement in instanceElement.findall("stylename"):
            for key, lang in styleNameElement.items():
                if key == XML_LANG:
                    styleName = styleNameElement.text
                    instanceObject.setStyleName(styleName, lang)
        for familyNameElement in instanceElement.findall("familyname"):
            for key, lang in familyNameElement.items():
                if key == XML_LANG:
                    familyName = familyNameElement.text
                    instanceObject.setFamilyName(familyName, lang)
        for styleMapStyleNameElement in instanceElement.findall("stylemapstylename"):
            for key, lang in styleMapStyleNameElement.items():
                if key == XML_LANG:
                    styleMapStyleName = styleMapStyleNameElement.text
                    instanceObject.setStyleMapStyleName(styleMapStyleName, lang)
        for styleMapFamilyNameElement in instanceElement.findall("stylemapfamilyname"):
            for key, lang in styleMapFamilyNameElement.items():
                if key == XML_LANG:
                    styleMapFamilyName = styleMapFamilyNameElement.text
                    instanceObject.setStyleMapFamilyName(styleMapFamilyName, lang)
        designLocation, userLocation = self.locationFromElement(instanceElement)
        locationLabel = instanceElement.attrib.get("location")
        if (designLocation or userLocation) and locationLabel is not None:
            raise DesignSpaceDocumentError(
                'instance element must have at most one of the location="..." attribute or the nested location element'
            )
        instanceObject.locationLabel = locationLabel
        instanceObject.userLocation = userLocation or {}
        instanceObject.designLocation = designLocation or {}
        for glyphElement in instanceElement.findall(".glyphs/glyph"):
            self.readGlyphElement(glyphElement, instanceObject)
        for infoElement in instanceElement.findall("info"):
            self.readInfoElement(infoElement, instanceObject)
        for libElement in instanceElement.findall("lib"):
            self.readLibElement(libElement, instanceObject)
        self.documentObject.instances.append(instanceObject)

    def readLibElement(self, libElement, instanceObject):
        """Read the lib element for the given instance."""
        if len(libElement) == 0:
            # an empty <lib> element is equivalent to no lib at all
            return
        instanceObject.lib = plistlib.fromtree(libElement[0])

    def readInfoElement(self, infoElement, instanceObject):
        """Read the info element."""
        instanceObject.info = True

    def readGlyphElement(self, glyphElement, instanceObject):
        """
        Read the glyph element, which could look like either one of these:

        .. code-block:: xml

            <glyph name="b" unicode="0x62"/>

            <glyph name="b"/>

            <glyph name="b">
                <master location="location-token-bbb" source="master-token-aaa2"/>
                <master glyphname="b.alt1" location="location-token-ccc" source="master-token-aaa3"/>
                <note>
                    This is an instance from an anisotropic interpolation.
                </note>
            </glyph>
        """
        glyphData = {}
        glyphName = glyphElement.attrib.get("name")
        if glyphName is None:
            raise DesignSpaceDocumentError("Glyph object without name attribute")
        mute = glyphElement.attrib.get("mute")
        if mute == "1":
            glyphData["mute"] = True
        # unicode
        unicodes = glyphElement.attrib.get("unicode")
        if unicodes is not None:
            try:
                unicodes = [int(u, 16) for u in unicodes.split(" ")]
                glyphData["unicodes"] = unicodes
            except ValueError:
                raise DesignSpaceDocumentError(
                    "unicode values %s are not integers" % unicodes
                )

        for noteElement in glyphElement.findall(".note"):
            glyphData["note"] = noteElement.text
            break
        designLocation, userLocation = self.locationFromElement(glyphElement)
        if userLocation:
            raise DesignSpaceDocumentError(
                f'<glyph> element "{glyphName}" must only have design locations (using xvalue="").'
            )
        if designLocation is not None:
            glyphData["instanceLocation"] = designLocation
        glyphSources = None
        for masterElement in glyphElement.findall(".masters/master"):
            fontSourceName = masterElement.attrib.get("source")
            designLocation, userLocation = self.locationFromElement(masterElement)
            if userLocation:
                raise DesignSpaceDocumentError(
                    f'<master> element "{fontSourceName}" must only have design locations (using xvalue="").'
                )
            masterGlyphName = masterElement.attrib.get("glyphname")
            if masterGlyphName is None:
                # if we don't read a glyphname, use the one we have
                masterGlyphName = glyphName
            d = dict(
                font=fontSourceName, location=designLocation, glyphName=masterGlyphName
            )
            if glyphSources is None:
                glyphSources = []
            glyphSources.append(d)
        if glyphSources is not None:
            glyphData["masters"] = glyphSources
        instanceObject.glyphs[glyphName] = glyphData

    def readLib(self):
        """Read the lib element for the whole document."""
        for libElement in self.root.findall(".lib"):
            if len(libElement) == 0:
                # an empty <lib> element is equivalent to no lib at all
                continue
            self.documentObject.lib = plistlib.fromtree(libElement[0])


class DesignSpaceDocument(LogMixin, AsDictMixin):
    """The DesignSpaceDocument object can read and write ``.designspace`` data.
    It imports the axes, sources, variable fonts and instances to very basic
    **descriptor** objects that store the data in attributes. Data is added to
    the document by creating such descriptor objects, filling them with data
    and then adding them to the document. This makes it easy to integrate this
    object in different contexts.

    The **DesignSpaceDocument** object can be subclassed to work with
    different objects, as long as they have the same attributes. Reader and
    Writer objects can be subclassed as well.

    **Note:** Python attribute names are usually camelCased, the
    corresponding `XML <document-xml-structure>`_ attributes are usually
    all lowercase.

    .. code:: python

        from fontTools.designspaceLib import DesignSpaceDocument
        doc = DesignSpaceDocument.fromfile("some/path/to/my.designspace")
        doc.formatVersion
        doc.elidedFallbackName
        doc.axes
        doc.axisMappings
        doc.locationLabels
        doc.rules
        doc.rulesProcessingLast
        doc.sources
        doc.variableFonts
        doc.instances
        doc.lib

    """

    def __init__(self, readerClass=None, writerClass=None):
        self.path = None
        """String, optional. When the document is read from the disk, this is
        the full path that was given to :meth:`read` or :meth:`fromfile`.
        """
        self.filename = None
        """String, optional. When the document is read from the disk, this is
        its original file name, i.e. the last part of its path.

        When the document is produced by a Python script and still only exists
        in memory, the producing script can write here an indication of a
        possible "good" filename, in case one wants to save the file somewhere.
        """

        self.formatVersion: Optional[str] = None
        """Format version for this document, as a string. E.g. "4.0" """

        self.elidedFallbackName: Optional[str] = None
        """STAT Style Attributes Header field ``elidedFallbackNameID``.

        See: `OTSpec STAT Style Attributes Header <https://docs.microsoft.com/en-us/typography/opentype/spec/stat#style-attributes-header>`_

        .. versionadded:: 5.0
        """

        self.axes: List[Union[AxisDescriptor, DiscreteAxisDescriptor]] = []
        """List of this document's axes."""

        self.axisMappings: List[AxisMappingDescriptor] = []
        """List of this document's axis mappings."""

        self.locationLabels: List[LocationLabelDescriptor] = []
        """List of this document's STAT format 4 labels.

        .. versionadded:: 5.0"""
        self.rules: List[RuleDescriptor] = []
        """List of this document's rules."""
        self.rulesProcessingLast: bool = False
        """This flag indicates whether the substitution rules should be applied
        before or after other glyph substitution features.

        - False: before
        - True: after.

        Default is False. For new projects, you probably want True. See
        the following issues for more information:
        `fontTools#1371 <https://github.com/fonttools/fonttools/issues/1371#issuecomment-590214572>`__
        `fontTools#2050 <https://github.com/fonttools/fonttools/issues/2050#issuecomment-678691020>`__

        If you want to use a different feature altogether, e.g. ``calt``,
        use the lib key ``com.github.fonttools.varLib.featureVarsFeatureTag``

        .. code:: xml

            <lib>
                <dict>
                    <key>com.github.fonttools.varLib.featureVarsFeatureTag</key>
                    <string>calt</string>
                </dict>
            </lib>
        """
        self.sources: List[SourceDescriptor] = []
        """List of this document's sources."""
        self.variableFonts: List[VariableFontDescriptor] = []
        """List of this document's variable fonts.

        .. versionadded:: 5.0"""
        self.instances: List[InstanceDescriptor] = []
        """List of this document's instances."""
        self.lib: Dict = {}
        """User defined, custom data associated with the whole document.

        Use reverse-DNS notation to identify your own data.
        Respect the data stored by others.
        """

        self.default: Optional[str] = None
        """Name of the default master.

        This attribute is updated by the :meth:`findDefault`
        """

        if readerClass is not None:
            self.readerClass = readerClass
        else:
            self.readerClass = BaseDocReader
        if writerClass is not None:
            self.writerClass = writerClass
        else:
            self.writerClass = BaseDocWriter

    @classmethod
    def fromfile(cls, path, readerClass=None, writerClass=None):
        """Read a designspace file from ``path`` and return a new instance of
        :class:.
        """
        self = cls(readerClass=readerClass, writerClass=writerClass)
        self.read(path)
        return self

    @classmethod
    def fromstring(cls, string, readerClass=None, writerClass=None):
        self = cls(readerClass=readerClass, writerClass=writerClass)
        reader = self.readerClass.fromstring(string, self)
        reader.read()
        if self.sources:
            self.findDefault()
        return self

    def tostring(self, encoding=None):
        """Returns the designspace as a string. Default encoding ``utf-8``."""
        if encoding is str or (encoding is not None and encoding.lower() == "unicode"):
            f = StringIO()
            xml_declaration = False
        elif encoding is None or encoding == "utf-8":
            f = BytesIO()
            encoding = "UTF-8"
            xml_declaration = True
        else:
            raise ValueError("unsupported encoding: '%s'" % encoding)
        writer = self.writerClass(f, self)
        writer.write(encoding=encoding, xml_declaration=xml_declaration)
        return f.getvalue()

    def read(self, path):
        """Read a designspace file from ``path`` and populates the fields of
        ``self`` with the data.
        """
        if hasattr(path, "__fspath__"):  # support os.PathLike objects
            path = path.__fspath__()
        self.path = path
        self.filename = os.path.basename(path)
        reader = self.readerClass(path, self)
        reader.read()
        if self.sources:
            self.findDefault()

    def write(self, path):
        """Write this designspace to ``path``."""
        if hasattr(path, "__fspath__"):  # support os.PathLike objects
            path = path.__fspath__()
        self.path = path
        self.filename = os.path.basename(path)
        self.updatePaths()
        writer = self.writerClass(path, self)
        writer.write()

    def _posixRelativePath(self, otherPath):
        relative = os.path.relpath(otherPath, os.path.dirname(self.path))
        return posix(relative)

    def updatePaths(self):
        """
        Right before we save we need to identify and respond to the following situations:
        In each descriptor, we have to do the right thing for the filename attribute.

        ::

            case 1.
            descriptor.filename == None
            descriptor.path == None

            -- action:
            write as is, descriptors will not have a filename attr.
            useless, but no reason to interfere.


            case 2.
            descriptor.filename == "../something"
            descriptor.path == None

            -- action:
            write as is. The filename attr should not be touched.


            case 3.
            descriptor.filename == None
            descriptor.path == "~/absolute/path/there"

            -- action:
            calculate the relative path for filename.
            We're not overwriting some other value for filename, it should be fine


            case 4.
            descriptor.filename == '../somewhere'
            descriptor.path == "~/absolute/path/there"

            -- action:
            there is a conflict between the given filename, and the path.
            So we know where the file is relative to the document.
            Can't guess why they're different, we just choose for path to be correct and update filename.
        """
        assert self.path is not None
        for descriptor in self.sources + self.instances:
            if descriptor.path is not None:
                # case 3 and 4: filename gets updated and relativized
                descriptor.filename = self._posixRelativePath(descriptor.path)

    def addSource(self, sourceDescriptor: SourceDescriptor):
        """Add the given ``sourceDescriptor`` to ``doc.sources``."""
        self.sources.append(sourceDescriptor)

    def addSourceDescriptor(self, **kwargs):
        """Instantiate a new :class:`SourceDescriptor` using the given
        ``kwargs`` and add it to ``doc.sources``.
        """
        source = self.writerClass.sourceDescriptorClass(**kwargs)
        self.addSource(source)
        return source

    def addInstance(self, instanceDescriptor: InstanceDescriptor):
        """Add the given ``instanceDescriptor`` to :attr:`instances`."""
        self.instances.append(instanceDescriptor)

    def addInstanceDescriptor(self, **kwargs):
        """Instantiate a new :class:`InstanceDescriptor` using the given
        ``kwargs`` and add it to :attr:`instances`.
        """
        instance = self.writerClass.instanceDescriptorClass(**kwargs)
        self.addInstance(instance)
        return instance

    def addAxis(self, axisDescriptor: Union[AxisDescriptor, DiscreteAxisDescriptor]):
        """Add the given ``axisDescriptor`` to :attr:`axes`."""
        self.axes.append(axisDescriptor)

    def addAxisDescriptor(self, **kwargs):
        """Instantiate a new :class:`AxisDescriptor` using the given
        ``kwargs`` and add it to :attr:`axes`.

        The axis will be and instance of :class:`DiscreteAxisDescriptor` if
        the ``kwargs`` provide a ``value``, or a :class:`AxisDescriptor` otherwise.
        """
        if "values" in kwargs:
            axis = self.writerClass.discreteAxisDescriptorClass(**kwargs)
        else:
            axis = self.writerClass.axisDescriptorClass(**kwargs)
        self.addAxis(axis)
        return axis

    def addAxisMapping(self, axisMappingDescriptor: AxisMappingDescriptor):
        """Add the given ``axisMappingDescriptor`` to :attr:`axisMappings`."""
        self.axisMappings.append(axisMappingDescriptor)

    def addAxisMappingDescriptor(self, **kwargs):
        """Instantiate a new :class:`AxisMappingDescriptor` using the given
        ``kwargs`` and add it to :attr:`rules`.
        """
        axisMapping = self.writerClass.axisMappingDescriptorClass(**kwargs)
        self.addAxisMapping(axisMapping)
        return axisMapping

    def addRule(self, ruleDescriptor: RuleDescriptor):
        """Add the given ``ruleDescriptor`` to :attr:`rules`."""
        self.rules.append(ruleDescriptor)

    def addRuleDescriptor(self, **kwargs):
        """Instantiate a new :class:`RuleDescriptor` using the given
        ``kwargs`` and add it to :attr:`rules`.
        """
        rule = self.writerClass.ruleDescriptorClass(**kwargs)
        self.addRule(rule)
        return rule

    def addVariableFont(self, variableFontDescriptor: VariableFontDescriptor):
        """Add the given ``variableFontDescriptor`` to :attr:`variableFonts`.

        .. versionadded:: 5.0
        """
        self.variableFonts.append(variableFontDescriptor)

    def addVariableFontDescriptor(self, **kwargs):
        """Instantiate a new :class:`VariableFontDescriptor` using the given
        ``kwargs`` and add it to :attr:`variableFonts`.

        .. versionadded:: 5.0
        """
        variableFont = self.writerClass.variableFontDescriptorClass(**kwargs)
        self.addVariableFont(variableFont)
        return variableFont

    def addLocationLabel(self, locationLabelDescriptor: LocationLabelDescriptor):
        """Add the given ``locationLabelDescriptor`` to :attr:`locationLabels`.

        .. versionadded:: 5.0
        """
        self.locationLabels.append(locationLabelDescriptor)

    def addLocationLabelDescriptor(self, **kwargs):
        """Instantiate a new :class:`LocationLabelDescriptor` using the given
        ``kwargs`` and add it to :attr:`locationLabels`.

        .. versionadded:: 5.0
        """
        locationLabel = self.writerClass.locationLabelDescriptorClass(**kwargs)
        self.addLocationLabel(locationLabel)
        return locationLabel

    def newDefaultLocation(self):
        """Return a dict with the default location in design space coordinates."""
        # Without OrderedDict, output XML would be non-deterministic.
        # https://github.com/LettError/designSpaceDocument/issues/10
        loc = collections.OrderedDict()
        for axisDescriptor in self.axes:
            loc[axisDescriptor.name] = axisDescriptor.map_forward(
                axisDescriptor.default
            )
        return loc

    def labelForUserLocation(
        self, userLocation: SimpleLocationDict
    ) -> Optional[LocationLabelDescriptor]:
        """Return the :class:`LocationLabel` that matches the given
        ``userLocation``, or ``None`` if no such label exists.

        .. versionadded:: 5.0
        """
        return next(
            (
                label
                for label in self.locationLabels
                if label.userLocation == userLocation
            ),
            None,
        )

    def updateFilenameFromPath(self, masters=True, instances=True, force=False):
        """Set a descriptor filename attr from the path and this document path.

        If the filename attribute is not None: skip it.
        """
        if masters:
            for descriptor in self.sources:
                if descriptor.filename is not None and not force:
                    continue
                if self.path is not None:
                    descriptor.filename = self._posixRelativePath(descriptor.path)
        if instances:
            for descriptor in self.instances:
                if descriptor.filename is not None and not force:
                    continue
                if self.path is not None:
                    descriptor.filename = self._posixRelativePath(descriptor.path)

    def newAxisDescriptor(self):
        """Ask the writer class to make us a new axisDescriptor."""
        return self.writerClass.getAxisDecriptor()

    def newSourceDescriptor(self):
        """Ask the writer class to make us a new sourceDescriptor."""
        return self.writerClass.getSourceDescriptor()

    def newInstanceDescriptor(self):
        """Ask the writer class to make us a new instanceDescriptor."""
        return self.writerClass.getInstanceDescriptor()

    def getAxisOrder(self):
        """Return a list of axis names, in the same order as defined in the document."""
        names = []
        for axisDescriptor in self.axes:
            names.append(axisDescriptor.name)
        return names

    def getAxis(self, name: str) -> AxisDescriptor | DiscreteAxisDescriptor | None:
        """Return the axis with the given ``name``, or ``None`` if no such axis exists."""
        return next((axis for axis in self.axes if axis.name == name), None)

    def getAxisByTag(self, tag: str) -> AxisDescriptor | DiscreteAxisDescriptor | None:
        """Return the axis with the given ``tag``, or ``None`` if no such axis exists."""
        return next((axis for axis in self.axes if axis.tag == tag), None)

    def getLocationLabel(self, name: str) -> Optional[LocationLabelDescriptor]:
        """Return the top-level location label with the given ``name``, or
        ``None`` if no such label exists.

        .. versionadded:: 5.0
        """
        for label in self.locationLabels:
            if label.name == name:
                return label
        return None

    def map_forward(self, userLocation: SimpleLocationDict) -> SimpleLocationDict:
        """Map a user location to a design location.

        Assume that missing coordinates are at the default location for that axis.

        Note: the output won't be anisotropic, only the xvalue is set.

        .. versionadded:: 5.0
        """
        return {
            axis.name: axis.map_forward(userLocation.get(axis.name, axis.default))
            for axis in self.axes
        }

    def map_backward(
        self, designLocation: AnisotropicLocationDict
    ) -> SimpleLocationDict:
        """Map a design location to a user location.

        Assume that missing coordinates are at the default location for that axis.

        When the input has anisotropic locations, only the xvalue is used.

        .. versionadded:: 5.0
        """
        return {
            axis.name: (
                axis.map_backward(designLocation[axis.name])
                if axis.name in designLocation
                else axis.default
            )
            for axis in self.axes
        }

    def findDefault(self):
        """Set and return SourceDescriptor at the default location or None.

        The default location is the set of all `default` values in user space
        of all axes.

        This function updates the document's :attr:`default` value.

        .. versionchanged:: 5.0
           Allow the default source to not specify some of the axis values, and
           they are assumed to be the default.
           See :meth:`SourceDescriptor.getFullDesignLocation()`
        """
        self.default = None

        # Convert the default location from user space to design space before comparing
        # it against the SourceDescriptor locations (always in design space).
        defaultDesignLocation = self.newDefaultLocation()

        for sourceDescriptor in self.sources:
            if sourceDescriptor.getFullDesignLocation(self) == defaultDesignLocation:
                self.default = sourceDescriptor
                return sourceDescriptor

        return None

    def normalizeLocation(self, location):
        """Return a dict with normalized axis values."""
        from fontTools.varLib.models import normalizeValue

        new = {}
        for axis in self.axes:
            if axis.name not in location:
                # skipping this dimension it seems
                continue
            value = location[axis.name]
            # 'anisotropic' location, take first coord only
            if isinstance(value, tuple):
                value = value[0]
            triple = [
                axis.map_forward(v) for v in (axis.minimum, axis.default, axis.maximum)
            ]
            new[axis.name] = normalizeValue(value, triple)
        return new

    def normalize(self):
        """
        Normalise the geometry of this designspace:

        - scale all the locations of all masters and instances to the -1 - 0 - 1 value.
        - we need the axis data to do the scaling, so we do those last.
        """
        # masters
        for item in self.sources:
            item.location = self.normalizeLocation(item.location)
        # instances
        for item in self.instances:
            # glyph masters for this instance
            for _, glyphData in item.glyphs.items():
                glyphData["instanceLocation"] = self.normalizeLocation(
                    glyphData["instanceLocation"]
                )
                for glyphMaster in glyphData["masters"]:
                    glyphMaster["location"] = self.normalizeLocation(
                        glyphMaster["location"]
                    )
            item.location = self.normalizeLocation(item.location)
        # the axes
        for axis in self.axes:
            # scale the map first
            newMap = []
            for inputValue, outputValue in axis.map:
                newOutputValue = self.normalizeLocation({axis.name: outputValue}).get(
                    axis.name
                )
                newMap.append((inputValue, newOutputValue))
            if newMap:
                axis.map = newMap
            # finally the axis values
            minimum = self.normalizeLocation({axis.name: axis.minimum}).get(axis.name)
            maximum = self.normalizeLocation({axis.name: axis.maximum}).get(axis.name)
            default = self.normalizeLocation({axis.name: axis.default}).get(axis.name)
            # and set them in the axis.minimum
            axis.minimum = minimum
            axis.maximum = maximum
            axis.default = default
        # now the rules
        for rule in self.rules:
            newConditionSets = []
            for conditions in rule.conditionSets:
                newConditions = []
                for cond in conditions:
                    if cond.get("minimum") is not None:
                        minimum = self.normalizeLocation(
                            {cond["name"]: cond["minimum"]}
                        ).get(cond["name"])
                    else:
                        minimum = None
                    if cond.get("maximum") is not None:
                        maximum = self.normalizeLocation(
                            {cond["name"]: cond["maximum"]}
                        ).get(cond["name"])
                    else:
                        maximum = None
                    newConditions.append(
                        dict(name=cond["name"], minimum=minimum, maximum=maximum)
                    )
                newConditionSets.append(newConditions)
            rule.conditionSets = newConditionSets

    def loadSourceFonts(self, opener, **kwargs):
        """Ensure SourceDescriptor.font attributes are loaded, and return list of fonts.

        Takes a callable which initializes a new font object (e.g. TTFont, or
        defcon.Font, etc.) from the SourceDescriptor.path, and sets the
        SourceDescriptor.font attribute.
        If the font attribute is already not None, it is not loaded again.
        Fonts with the same path are only loaded once and shared among SourceDescriptors.

        For example, to load UFO sources using defcon:

            designspace = DesignSpaceDocument.fromfile("path/to/my.designspace")
            designspace.loadSourceFonts(defcon.Font)

        Or to load masters as FontTools binary fonts, including extra options:

            designspace.loadSourceFonts(ttLib.TTFont, recalcBBoxes=False)

        Args:
            opener (Callable): takes one required positional argument, the source.path,
                and an optional list of keyword arguments, and returns a new font object
                loaded from the path.
            **kwargs: extra options passed on to the opener function.

        Returns:
            List of font objects in the order they appear in the sources list.
        """
        # we load fonts with the same source.path only once
        loaded = {}
        fonts = []
        for source in self.sources:
            if source.font is not None:  # font already loaded
                fonts.append(source.font)
                continue
            if source.path in loaded:
                source.font = loaded[source.path]
            else:
                if source.path is None:
                    raise DesignSpaceDocumentError(
                        "Designspace source '%s' has no 'path' attribute"
                        % (source.name or "<Unknown>")
                    )
                source.font = opener(source.path, **kwargs)
                loaded[source.path] = source.font
            fonts.append(source.font)
        return fonts

    @property
    def formatTuple(self):
        """Return the formatVersion as a tuple of (major, minor).

        .. versionadded:: 5.0
        """
        if self.formatVersion is None:
            return (5, 0)
        numbers = (int(i) for i in self.formatVersion.split("."))
        major = next(numbers)
        minor = next(numbers, 0)
        return (major, minor)

    def getVariableFonts(self) -> List[VariableFontDescriptor]:
        """Return all variable fonts defined in this document, or implicit
        variable fonts that can be built from the document's continuous axes.

        In the case of Designspace documents before version 5, the whole
        document was implicitly describing a variable font that covers the
        whole space.

        In version 5 and above documents, there can be as many variable fonts
        as there are locations on discrete axes.

        .. seealso:: :func:`splitInterpolable`

        .. versionadded:: 5.0
        """
        if self.variableFonts:
            return self.variableFonts

        variableFonts = []
        discreteAxes = []
        rangeAxisSubsets: List[
            Union[RangeAxisSubsetDescriptor, ValueAxisSubsetDescriptor]
        ] = []
        for axis in self.axes:
            if hasattr(axis, "values"):
                # Mypy doesn't support narrowing union types via hasattr()
                # TODO(Python 3.10): use TypeGuard
                # https://mypy.readthedocs.io/en/stable/type_narrowing.html
                axis = cast(DiscreteAxisDescriptor, axis)
                discreteAxes.append(axis)  # type: ignore
            else:
                rangeAxisSubsets.append(RangeAxisSubsetDescriptor(name=axis.name))
        valueCombinations = itertools.product(*[axis.values for axis in discreteAxes])
        for values in valueCombinations:
            basename = None
            if self.filename is not None:
                basename = os.path.splitext(self.filename)[0] + "-VF"
            if self.path is not None:
                basename = os.path.splitext(os.path.basename(self.path))[0] + "-VF"
            if basename is None:
                basename = "VF"
            axisNames = "".join(
                [f"-{axis.tag}{value}" for axis, value in zip(discreteAxes, values)]
            )
            variableFonts.append(
                VariableFontDescriptor(
                    name=f"{basename}{axisNames}",
                    axisSubsets=rangeAxisSubsets
                    + [
                        ValueAxisSubsetDescriptor(name=axis.name, userValue=value)
                        for axis, value in zip(discreteAxes, values)
                    ],
                )
            )
        return variableFonts

    def deepcopyExceptFonts(self):
        """Allow deep-copying a DesignSpace document without deep-copying
        attached UFO fonts or TTFont objects. The :attr:`font` attribute
        is shared by reference between the original and the copy.

        .. versionadded:: 5.0
        """
        fonts = [source.font for source in self.sources]
        try:
            for source in self.sources:
                source.font = None
            res = copy.deepcopy(self)
            for source, font in zip(res.sources, fonts):
                source.font = font
            return res
        finally:
            for source, font in zip(self.sources, fonts):
                source.font = font


def main(args=None):
    """Roundtrip .designspace file through the DesignSpaceDocument class"""

    if args is None:
        import sys

        args = sys.argv[1:]

    from argparse import ArgumentParser

    parser = ArgumentParser(prog="designspaceLib", description=main.__doc__)
    parser.add_argument("input")
    parser.add_argument("output")

    options = parser.parse_args(args)

    ds = DesignSpaceDocument.fromfile(options.input)
    ds.write(options.output)
<|endoftext|>"""Allows building all the variable fonts of a DesignSpace version 5 by
splitting the document into interpolable sub-space, then into each VF.
"""

from __future__ import annotations

import itertools
import logging
import math
from typing import Any, Callable, Dict, Iterator, List, Tuple, cast

from fontTools.designspaceLib import (
    AxisDescriptor,
    AxisMappingDescriptor,
    DesignSpaceDocument,
    DiscreteAxisDescriptor,
    InstanceDescriptor,
    RuleDescriptor,
    SimpleLocationDict,
    SourceDescriptor,
    VariableFontDescriptor,
)
from fontTools.designspaceLib.statNames import StatNames, getStatNames
from fontTools.designspaceLib.types import (
    ConditionSet,
    Range,
    Region,
    getVFUserRegion,
    locationInRegion,
    regionInRegion,
    userRegionToDesignRegion,
)

LOGGER = logging.getLogger(__name__)

MakeInstanceFilenameCallable = Callable[
    [DesignSpaceDocument, InstanceDescriptor, StatNames], str
]


def defaultMakeInstanceFilename(
    doc: DesignSpaceDocument, instance: InstanceDescriptor, statNames: StatNames
) -> str:
    """Default callable to synthesize an instance filename
    when makeNames=True, for instances that don't specify an instance name
    in the designspace. This part of the name generation can be overriden
    because it's not specified by the STAT table.
    """
    familyName = instance.familyName or statNames.familyNames.get("en")
    styleName = instance.styleName or statNames.styleNames.get("en")
    return f"{familyName}-{styleName}.ttf"


def splitInterpolable(
    doc: DesignSpaceDocument,
    makeNames: bool = True,
    expandLocations: bool = True,
    makeInstanceFilename: MakeInstanceFilenameCallable = defaultMakeInstanceFilename,
) -> Iterator[Tuple[SimpleLocationDict, DesignSpaceDocument]]:
    """Split the given DS5 into several interpolable sub-designspaces.
    There are as many interpolable sub-spaces as there are combinations of
    discrete axis values.

    E.g. with axes:
        - italic (discrete) Upright or Italic
        - style (discrete) Sans or Serif
        - weight (continuous) 100 to 900

    There are 4 sub-spaces in which the Weight axis should interpolate:
    (Upright, Sans), (Upright, Serif), (Italic, Sans) and (Italic, Serif).

    The sub-designspaces still include the full axis definitions and STAT data,
    but the rules, sources, variable fonts, instances are trimmed down to only
    keep what falls within the interpolable sub-space.

    Args:
      - ``makeNames``: Whether to compute the instance family and style
        names using the STAT data.
      - ``expandLocations``: Whether to turn all locations into "full"
        locations, including implicit default axis values where missing.
      - ``makeInstanceFilename``: Callable to synthesize an instance filename
        when makeNames=True, for instances that don't specify an instance name
        in the designspace. This part of the name generation can be overridden
        because it's not specified by the STAT table.

    .. versionadded:: 5.0
    """
    discreteAxes = []
    interpolableUserRegion: Region = {}
    for axis in doc.axes:
        if hasattr(axis, "values"):
            # Mypy doesn't support narrowing union types via hasattr()
            # TODO(Python 3.10): use TypeGuard
            # https://mypy.readthedocs.io/en/stable/type_narrowing.html
            axis = cast(DiscreteAxisDescriptor, axis)
            discreteAxes.append(axis)
        else:
            axis = cast(AxisDescriptor, axis)
            interpolableUserRegion[axis.name] = Range(
                axis.minimum,
                axis.maximum,
                axis.default,
            )
    valueCombinations = itertools.product(*[axis.values for axis in discreteAxes])
    for values in valueCombinations:
        discreteUserLocation = {
            discreteAxis.name: value
            for discreteAxis, value in zip(discreteAxes, values)
        }
        subDoc = _extractSubSpace(
            doc,
            {**interpolableUserRegion, **discreteUserLocation},
            keepVFs=True,
            makeNames=makeNames,
            expandLocations=expandLocations,
            makeInstanceFilename=makeInstanceFilename,
        )
        yield discreteUserLocation, subDoc


def splitVariableFonts(
    doc: DesignSpaceDocument,
    makeNames: bool = False,
    expandLocations: bool = False,
    makeInstanceFilename: MakeInstanceFilenameCallable = defaultMakeInstanceFilename,
) -> Iterator[Tuple[str, DesignSpaceDocument]]:
    """Convert each variable font listed in this document into a standalone
    designspace. This can be used to compile all the variable fonts from a
    format 5 designspace using tools that can only deal with 1 VF at a time.

    Args:
      - ``makeNames``: Whether to compute the instance family and style
        names using the STAT data.
      - ``expandLocations``: Whether to turn all locations into "full"
        locations, including implicit default axis values where missing.
      - ``makeInstanceFilename``: Callable to synthesize an instance filename
        when makeNames=True, for instances that don't specify an instance name
        in the designspace. This part of the name generation can be overridden
        because it's not specified by the STAT table.

    .. versionadded:: 5.0
    """
    # Make one DesignspaceDoc v5 for each variable font
    for vf in doc.getVariableFonts():
        vfUserRegion = getVFUserRegion(doc, vf)
        vfDoc = _extractSubSpace(
            doc,
            vfUserRegion,
            keepVFs=False,
            makeNames=makeNames,
            expandLocations=expandLocations,
            makeInstanceFilename=makeInstanceFilename,
        )
        vfDoc.lib = {**vfDoc.lib, **vf.lib}
        yield vf.name, vfDoc


def convert5to4(
    doc: DesignSpaceDocument,
) -> Dict[str, DesignSpaceDocument]:
    """Convert each variable font listed in this document into a standalone
    format 4 designspace. This can be used to compile all the variable fonts
    from a format 5 designspace using tools that only know about format 4.

    .. versionadded:: 5.0
    """
    vfs = {}
    for _location, subDoc in splitInterpolable(doc):
        for vfName, vfDoc in splitVariableFonts(subDoc):
            vfDoc.formatVersion = "4.1"
            vfs[vfName] = vfDoc
    return vfs


def _extractSubSpace(
    doc: DesignSpaceDocument,
    userRegion: Region,
    *,
    keepVFs: bool,
    makeNames: bool,
    expandLocations: bool,
    makeInstanceFilename: MakeInstanceFilenameCallable,
) -> DesignSpaceDocument:
    subDoc = DesignSpaceDocument()
    # Don't include STAT info
    # FIXME: (Jany) let's think about it. Not include = OK because the point of
    # the splitting is to build VFs and we'll use the STAT data of the full
    # document to generate the STAT of the VFs, so "no need" to have STAT data
    # in sub-docs. Counterpoint: what if someone wants to split this DS for
    # other purposes?  Maybe for that it would be useful to also subset the STAT
    # data?
    # subDoc.elidedFallbackName = doc.elidedFallbackName

    def maybeExpandDesignLocation(object):
        if expandLocations:
            return object.getFullDesignLocation(doc)
        else:
            return object.designLocation

    for axis in doc.axes:
        range = userRegion[axis.name]
        if isinstance(range, Range) and hasattr(axis, "minimum"):
            # Mypy doesn't support narrowing union types via hasattr()
            # TODO(Python 3.10): use TypeGuard
            # https://mypy.readthedocs.io/en/stable/type_narrowing.html
            axis = cast(AxisDescriptor, axis)
            subDoc.addAxis(
                AxisDescriptor(
                    # Same info
                    tag=axis.tag,
                    name=axis.name,
                    labelNames=axis.labelNames,
                    hidden=axis.hidden,
                    # Subset range
                    minimum=max(range.minimum, axis.minimum),
                    default=range.default or axis.default,
                    maximum=min(range.maximum, axis.maximum),
                    map=[
                        (user, design)
                        for user, design in axis.map
                        if range.minimum <= user <= range.maximum
                    ],
                    # Don't include STAT info
                    axisOrdering=None,
                    axisLabels=None,
                )
            )

    subDoc.axisMappings = mappings = []
    subDocAxes = {axis.name for axis in subDoc.axes}
    for mapping in doc.axisMappings:
        if not all(axis in subDocAxes for axis in mapping.inputLocation.keys()):
            continue
        if not all(axis in subDocAxes for axis in mapping.outputLocation.keys()):
            LOGGER.error(
                "In axis mapping from input %s, some output axes are not in the variable-font: %s",
                mapping.inputLocation,
                mapping.outputLocation,
            )
            continue

        mappingAxes = set()
        mappingAxes.update(mapping.inputLocation.keys())
        mappingAxes.update(mapping.outputLocation.keys())
        for axis in doc.axes:
            if axis.name not in mappingAxes:
                continue
            range = userRegion[axis.name]
            if (
                range.minimum != axis.minimum
                or (range.default is not None and range.default != axis.default)
                or range.maximum != axis.maximum
            ):
                LOGGER.error(
                    "Limiting axis ranges used in <mapping> elements not supported: %s",
                    axis.name,
                )
                continue

        mappings.append(
            AxisMappingDescriptor(
                inputLocation=mapping.inputLocation,
                outputLocation=mapping.outputLocation,
            )
        )

    # Don't include STAT info
    # subDoc.locationLabels = doc.locationLabels

    # Rules: subset them based on conditions
    designRegion = userRegionToDesignRegion(doc, userRegion)
    subDoc.rules = _subsetRulesBasedOnConditions(doc.rules, designRegion)
    subDoc.rulesProcessingLast = doc.rulesProcessingLast

    # Sources: keep only the ones that fall within the kept axis ranges
    for source in doc.sources:
        if not locationInRegion(doc.map_backward(source.designLocation), userRegion):
            continue

        subDoc.addSource(
            SourceDescriptor(
                filename=source.filename,
                path=source.path,
                font=source.font,
                name=source.name,
                designLocation=_filterLocation(
                    userRegion, maybeExpandDesignLocation(source)
                ),
                layerName=source.layerName,
                familyName=source.familyName,
                styleName=source.styleName,
                muteKerning=source.muteKerning,
                muteInfo=source.muteInfo,
                mutedGlyphNames=source.mutedGlyphNames,
            )
        )

    # Copy family name translations from the old default source to the new default
    vfDefault = subDoc.findDefault()
    oldDefault = doc.findDefault()
    if vfDefault is not None and oldDefault is not None:
        vfDefault.localisedFamilyName = oldDefault.localisedFamilyName

    # Variable fonts: keep only the ones that fall within the kept axis ranges
    if keepVFs:
        # Note: call getVariableFont() to make the implicit VFs explicit
        for vf in doc.getVariableFonts():
            vfUserRegion = getVFUserRegion(doc, vf)
            if regionInRegion(vfUserRegion, userRegion):
                subDoc.addVariableFont(
                    VariableFontDescriptor(
                        name=vf.name,
                        filename=vf.filename,
                        axisSubsets=[
                            axisSubset
                            for axisSubset in vf.axisSubsets
                            if isinstance(userRegion[axisSubset.name], Range)
                        ],
                        lib=vf.lib,
                    )
                )

    # Instances: same as Sources + compute missing names
    for instance in doc.instances:
        if not locationInRegion(instance.getFullUserLocation(doc), userRegion):
            continue

        if makeNames:
            statNames = getStatNames(doc, instance.getFullUserLocation(doc))
            familyName = instance.familyName or statNames.familyNames.get("en")
            styleName = instance.styleName or statNames.styleNames.get("en")
            subDoc.addInstance(
                InstanceDescriptor(
                    filename=instance.filename
                    or makeInstanceFilename(doc, instance, statNames),
                    path=instance.path,
                    font=instance.font,
                    name=instance.name or f"{familyName} {styleName}",
                    userLocation={} if expandLocations else instance.userLocation,
                    designLocation=_filterLocation(
                        userRegion, maybeExpandDesignLocation(instance)
                    ),
                    familyName=familyName,
                    styleName=styleName,
                    postScriptFontName=instance.postScriptFontName
                    or statNames.postScriptFontName,
                    styleMapFamilyName=instance.styleMapFamilyName
                    or statNames.styleMapFamilyNames.get("en"),
                    styleMapStyleName=instance.styleMapStyleName
                    or statNames.styleMapStyleName,
                    localisedFamilyName=instance.localisedFamilyName
                    or statNames.familyNames,
                    localisedStyleName=instance.localisedStyleName
                    or statNames.styleNames,
                    localisedStyleMapFamilyName=instance.localisedStyleMapFamilyName
                    or statNames.styleMapFamilyNames,
                    localisedStyleMapStyleName=instance.localisedStyleMapStyleName
                    or {},
                    lib=instance.lib,
                )
            )
        else:
            subDoc.addInstance(
                InstanceDescriptor(
                    filename=instance.filename,
                    path=instance.path,
                    font=instance.font,
                    name=instance.name,
                    userLocation={} if expandLocations else instance.userLocation,
                    designLocation=_filterLocation(
                        userRegion, maybeExpandDesignLocation(instance)
                    ),
                    familyName=instance.familyName,
                    styleName=instance.styleName,
                    postScriptFontName=instance.postScriptFontName,
                    styleMapFamilyName=instance.styleMapFamilyName,
                    styleMapStyleName=instance.styleMapStyleName,
                    localisedFamilyName=instance.localisedFamilyName,
                    localisedStyleName=instance.localisedStyleName,
                    localisedStyleMapFamilyName=instance.localisedStyleMapFamilyName,
                    localisedStyleMapStyleName=instance.localisedStyleMapStyleName,
                    lib=instance.lib,
                )
            )

    subDoc.lib = doc.lib

    return subDoc


def _conditionSetFrom(conditionSet: List[Dict[str, Any]]) -> ConditionSet:
    c: Dict[str, Range] = {}
    for condition in conditionSet:
        minimum, maximum = condition.get("minimum"), condition.get("maximum")
        c[condition["name"]] = Range(
            minimum if minimum is not None else -math.inf,
            maximum if maximum is not None else math.inf,
        )
    return c


def _subsetRulesBasedOnConditions(
    rules: List[RuleDescriptor], designRegion: Region
) -> List[RuleDescriptor]:
    # What rules to keep:
    #  - Keep the rule if any conditionset is relevant.
    #  - A conditionset is relevant if all conditions are relevant or it is empty.
    #  - A condition is relevant if
    #    - axis is point (C-AP),
    #       - and point in condition's range (C-AP-in)
    #            (in this case remove the condition because it's always true)
    #       - else (C-AP-out) whole conditionset can be discarded (condition false
    #         => conditionset false)
    #    - axis is range (C-AR),
    #       - (C-AR-all) and axis range fully contained in condition range: we can
    #         scrap the condition because it's always true
    #       - (C-AR-inter) and intersection(axis range, condition range) not empty:
    #         keep the condition with the smaller range (= intersection)
    #       - (C-AR-none) else, whole conditionset can be discarded
    newRules: List[RuleDescriptor] = []
    for rule in rules:
        newRule: RuleDescriptor = RuleDescriptor(
            name=rule.name, conditionSets=[], subs=rule.subs
        )
        for conditionset in rule.conditionSets:
            cs = _conditionSetFrom(conditionset)
            newConditionset: List[Dict[str, Any]] = []
            discardConditionset = False
            for selectionName, selectionValue in designRegion.items():
                # TODO: Ensure that all(key in conditionset for key in region.keys())?
                if selectionName not in cs:
                    # raise Exception("Selection has different axes than the rules")
                    continue
                if isinstance(selectionValue, (float, int)):  # is point
                    # Case C-AP-in
                    if selectionValue in cs[selectionName]:
                        pass  # always matches, conditionset can stay empty for this one.
                    # Case C-AP-out
                    else:
                        discardConditionset = True
                else:  # is range
                    # Case C-AR-all
                    if selectionValue in cs[selectionName]:
                        pass  # always matches, conditionset can stay empty for this one.
                    else:
                        intersection = cs[selectionName].intersection(selectionValue)
                        # Case C-AR-inter
                        if intersection is not None:
                            newConditionset.append(
                                {
                                    "name": selectionName,
                                    "minimum": intersection.minimum,
                                    "maximum": intersection.maximum,
                                }
                            )
                        # Case C-AR-none
                        else:
                            discardConditionset = True
            if not discardConditionset:
                newRule.conditionSets.append(newConditionset)
        if newRule.conditionSets:
            newRules.append(newRule)

    return newRules


def _filterLocation(
    userRegion: Region,
    location: Dict[str, float],
) -> Dict[str, float]:
    return {
        name: value
        for name, value in location.items()
        if name in userRegion and isinstance(userRegion[name], Range)
    }
<|endoftext|>from __future__ import annotations

from collections import namedtuple, OrderedDict
import itertools
from typing import Dict, Union
from fontTools.misc.fixedTools import fixedToFloat
from fontTools.misc.roundTools import otRound
from fontTools import ttLib
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.tables.otBase import (
    ValueRecord,
    valueRecordFormatDict,
    OTLOffsetOverflowError,
    OTTableWriter,
)
from fontTools.ttLib.ttFont import TTFont
from fontTools.feaLib.ast import STATNameStatement
from fontTools.otlLib.optimize.gpos import (
    _compression_level_from_env,
    compact_lookup,
)
from fontTools.otlLib.error import OpenTypeLibError
from fontTools.misc.loggingTools import deprecateFunction
from functools import reduce
import logging
import copy

log = logging.getLogger(__name__)


def buildCoverage(glyphs, glyphMap):
    """Builds a coverage table.

    Coverage tables (as defined in the `OpenType spec <https://docs.microsoft.com/en-gb/typography/opentype/spec/chapter2#coverage-table>`__)
    are used in all OpenType Layout lookups apart from the Extension type, and
    define the glyphs involved in a layout subtable. This allows shaping engines
    to compare the glyph stream with the coverage table and quickly determine
    whether a subtable should be involved in a shaping operation.

    This function takes a list of glyphs and a glyphname-to-ID map, and
    returns a ``Coverage`` object representing the coverage table.

    Example::

        glyphMap = font.getReverseGlyphMap()
        glyphs = [ "A", "B", "C" ]
        coverage = buildCoverage(glyphs, glyphMap)

    Args:
        glyphs: a sequence of glyph names.
        glyphMap: a glyph name to ID map, typically returned from
            ``font.getReverseGlyphMap()``.

    Returns:
        An ``otTables.Coverage`` object (empty if no glyphs supplied).
    """
    # Per the OpenType spec: "For cases in which subtable offset fields are not
    # documented as permitting NULL values, font compilers must include a subtable
    # of the indicated format, even if it is a header stub without further data
    # (for example, a coverage table with no glyph IDs)."
    # https://github.com/fonttools/fonttools/issues/4003
    self = ot.Coverage()
    if glyphs:
        try:
            self.glyphs = sorted(set(glyphs), key=glyphMap.__getitem__)
        except KeyError as e:
            raise ValueError(f"Could not find glyph {e} in font") from e
    else:
        self.glyphs = []
    return self


LOOKUP_FLAG_RIGHT_TO_LEFT = 0x0001
LOOKUP_FLAG_IGNORE_BASE_GLYPHS = 0x0002
LOOKUP_FLAG_IGNORE_LIGATURES = 0x0004
LOOKUP_FLAG_IGNORE_MARKS = 0x0008
LOOKUP_FLAG_USE_MARK_FILTERING_SET = 0x0010


def buildLookup(subtables, flags=0, markFilterSet=None, table=None, extension=False):
    """Turns a collection of rules into a lookup.

    A Lookup (as defined in the `OpenType Spec <https://docs.microsoft.com/en-gb/typography/opentype/spec/chapter2#lookupTbl>`__)
    wraps the individual rules in a layout operation (substitution or
    positioning) in a data structure expressing their overall lookup type -
    for example, single substitution, mark-to-base attachment, and so on -
    as well as the lookup flags and any mark filtering sets. You may import
    the following constants to express lookup flags:

    - ``LOOKUP_FLAG_RIGHT_TO_LEFT``
    - ``LOOKUP_FLAG_IGNORE_BASE_GLYPHS``
    - ``LOOKUP_FLAG_IGNORE_LIGATURES``
    - ``LOOKUP_FLAG_IGNORE_MARKS``
    - ``LOOKUP_FLAG_USE_MARK_FILTERING_SET``

    Args:
        subtables: A list of layout subtable objects (e.g.
            ``MultipleSubst``, ``PairPos``, etc.) or ``None``.
        flags (int): This lookup's flags.
        markFilterSet: Either ``None`` if no mark filtering set is used, or
            an integer representing the filtering set to be used for this
            lookup. If a mark filtering set is provided,
            `LOOKUP_FLAG_USE_MARK_FILTERING_SET` will be set on the lookup's
            flags.
        table (str): The name of the table this lookup belongs to, e.g. "GPOS" or "GSUB".
        extension (bool): ``True`` if this is an extension lookup, ``False`` otherwise.

    Returns:
        An ``otTables.Lookup`` object or ``None`` if there are no subtables
        supplied.
    """
    if subtables is None:
        return None
    subtables = [st for st in subtables if st is not None]
    if not subtables:
        return None
    assert all(
        t.LookupType == subtables[0].LookupType for t in subtables
    ), "all subtables must have the same LookupType; got %s" % repr(
        [t.LookupType for t in subtables]
    )

    if extension:
        assert table in ("GPOS", "GSUB")
        lookupType = 7 if table == "GSUB" else 9
        extSubTableClass = ot.lookupTypes[table][lookupType]
        for i, st in enumerate(subtables):
            subtables[i] = extSubTableClass()
            subtables[i].Format = 1
            subtables[i].ExtSubTable = st
            subtables[i].ExtensionLookupType = st.LookupType
    else:
        lookupType = subtables[0].LookupType

    self = ot.Lookup()
    self.LookupType = lookupType
    self.LookupFlag = flags
    self.SubTable = subtables
    self.SubTableCount = len(self.SubTable)
    if markFilterSet is not None:
        self.LookupFlag |= LOOKUP_FLAG_USE_MARK_FILTERING_SET
        assert isinstance(markFilterSet, int), markFilterSet
        self.MarkFilteringSet = markFilterSet
    else:
        assert (self.LookupFlag & LOOKUP_FLAG_USE_MARK_FILTERING_SET) == 0, (
            "if markFilterSet is None, flags must not set "
            "LOOKUP_FLAG_USE_MARK_FILTERING_SET; flags=0x%04x" % flags
        )
    return self


class LookupBuilder(object):
    SUBTABLE_BREAK_ = "SUBTABLE_BREAK"

    def __init__(self, font, location, table, lookup_type, extension=False):
        self.font = font
        self.glyphMap = font.getReverseGlyphMap()
        self.location = location
        self.table, self.lookup_type = table, lookup_type
        self.lookupflag = 0
        self.markFilterSet = None
        self.lookup_index = None  # assigned when making final tables
        self.extension = extension
        assert table in ("GPOS", "GSUB")

    def equals(self, other):
        return (
            isinstance(other, self.__class__)
            and self.table == other.table
            and self.lookupflag == other.lookupflag
            and self.markFilterSet == other.markFilterSet
            and self.extension == other.extension
        )

    def promote_lookup_type(self, is_named_lookup):
        return [self]

    def inferGlyphClasses(self):
        """Infers glyph glasses for the GDEF table, such as {"cedilla":3}."""
        return {}

    def getAlternateGlyphs(self):
        """Helper for building 'aalt' features."""
        return {}

    def buildLookup_(self, subtables):
        return buildLookup(
            subtables,
            self.lookupflag,
            self.markFilterSet,
            self.table,
            self.extension,
        )

    def buildMarkClasses_(self, marks):
        """{"cedilla": ("BOTTOM", ast.Anchor), ...} --> {"BOTTOM":0, "TOP":1}

        Helper for MarkBasePostBuilder, MarkLigPosBuilder, and
        MarkMarkPosBuilder. Seems to return the same numeric IDs
        for mark classes as the AFDKO makeotf tool.
        """
        ids = {}
        for mark in sorted(marks.keys(), key=self.font.getGlyphID):
            markClassName, _markAnchor = marks[mark]
            if markClassName not in ids:
                ids[markClassName] = len(ids)
        return ids

    def setBacktrackCoverage_(self, prefix, subtable):
        subtable.BacktrackGlyphCount = len(prefix)
        subtable.BacktrackCoverage = []
        for p in reversed(prefix):
            coverage = buildCoverage(p, self.glyphMap)
            subtable.BacktrackCoverage.append(coverage)

    def setLookAheadCoverage_(self, suffix, subtable):
        subtable.LookAheadGlyphCount = len(suffix)
        subtable.LookAheadCoverage = []
        for s in suffix:
            coverage = buildCoverage(s, self.glyphMap)
            subtable.LookAheadCoverage.append(coverage)

    def setInputCoverage_(self, glyphs, subtable):
        subtable.InputGlyphCount = len(glyphs)
        subtable.InputCoverage = []
        for g in glyphs:
            coverage = buildCoverage(g, self.glyphMap)
            subtable.InputCoverage.append(coverage)

    def setCoverage_(self, glyphs, subtable):
        subtable.GlyphCount = len(glyphs)
        subtable.Coverage = []
        for g in glyphs:
            coverage = buildCoverage(g, self.glyphMap)
            subtable.Coverage.append(coverage)

    def build_subst_subtables(self, mapping, klass):
        substitutions = [{}]
        for key in mapping:
            if key[0] == self.SUBTABLE_BREAK_:
                substitutions.append({})
            else:
                substitutions[-1][key] = mapping[key]
        subtables = [klass(s) for s in substitutions]
        return subtables

    def add_subtable_break(self, location):
        """Add an explicit subtable break.

        Args:
            location: A string or tuple representing the location in the
                original source which produced this break, or ``None`` if
                no location is provided.
        """
        log.warning(
            OpenTypeLibError(
                'unsupported "subtable" statement for lookup type', location
            )
        )

    def can_add_mapping(self, _mapping) -> bool:
        # used by AnySubstBuilder,  below
        return True


class AlternateSubstBuilder(LookupBuilder):
    """Builds an Alternate Substitution (GSUB3) lookup.

    Users are expected to manually add alternate glyph substitutions to
    the ``alternates`` attribute after the object has been initialized,
    e.g.::

        builder.alternates["A"] = ["A.alt1", "A.alt2"]

    Attributes:
        font (``fontTools.TTLib.TTFont``): A font object.
        location: A string or tuple representing the location in the original
            source which produced this lookup.
        alternates: An ordered dictionary of alternates, mapping glyph names
            to a list of names of alternates.
        lookupflag (int): The lookup's flag
        markFilterSet: Either ``None`` if no mark filtering set is used, or
            an integer representing the filtering set to be used for this
            lookup. If a mark filtering set is provided,
            `LOOKUP_FLAG_USE_MARK_FILTERING_SET` will be set on the lookup's
            flags.
    """

    def __init__(self, font, location):
        LookupBuilder.__init__(self, font, location, "GSUB", 3)
        self.alternates = OrderedDict()

    def equals(self, other):
        return LookupBuilder.equals(self, other) and self.alternates == other.alternates

    def build(self):
        """Build the lookup.

        Returns:
            An ``otTables.Lookup`` object representing the alternate
            substitution lookup.
        """
        subtables = self.build_subst_subtables(
            self.alternates, buildAlternateSubstSubtable
        )
        return self.buildLookup_(subtables)

    def getAlternateGlyphs(self):
        return self.alternates

    def add_subtable_break(self, location):
        self.alternates[(self.SUBTABLE_BREAK_, location)] = self.SUBTABLE_BREAK_


class ChainContextualRule(
    namedtuple("ChainContextualRule", ["prefix", "glyphs", "suffix", "lookups"])
):
    @property
    def is_subtable_break(self):
        return self.prefix == LookupBuilder.SUBTABLE_BREAK_


class ChainContextualRuleset:
    def __init__(self):
        self.rules = []

    def addRule(self, rule):
        self.rules.append(rule)

    @property
    def hasPrefixOrSuffix(self):
        # Do we have any prefixes/suffixes? If this is False for all
        # rulesets, we can express the whole lookup as GPOS5/GSUB7.
        for rule in self.rules:
            if len(rule.prefix) > 0 or len(rule.suffix) > 0:
                return True
        return False

    @property
    def hasAnyGlyphClasses(self):
        # Do we use glyph classes anywhere in the rules? If this is False
        # we can express this subtable as a Format 1.
        for rule in self.rules:
            for coverage in (rule.prefix, rule.glyphs, rule.suffix):
                if any(len(x) > 1 for x in coverage):
                    return True
        return False

    def format2ClassDefs(self):
        PREFIX, GLYPHS, SUFFIX = 0, 1, 2
        classDefBuilders = []
        for ix in [PREFIX, GLYPHS, SUFFIX]:
            context = []
            for r in self.rules:
                context.append(r[ix])
            classes = self._classBuilderForContext(context)
            if not classes:
                return None
            classDefBuilders.append(classes)
        return classDefBuilders

    def _classBuilderForContext(self, context):
        classdefbuilder = ClassDefBuilder(useClass0=False)
        for position in context:
            for glyphset in position:
                glyphs = set(glyphset)
                if not classdefbuilder.canAdd(glyphs):
                    return None
                classdefbuilder.add(glyphs)
        return classdefbuilder


class ChainContextualBuilder(LookupBuilder):
    def equals(self, other):
        return LookupBuilder.equals(self, other) and self.rules == other.rules

    def rulesets(self):
        # Return a list of ChainContextRuleset objects, taking explicit
        # subtable breaks into account
        ruleset = [ChainContextualRuleset()]
        for rule in self.rules:
            if rule.is_subtable_break:
                ruleset.append(ChainContextualRuleset())
                continue
            ruleset[-1].addRule(rule)
        # Squish any empty subtables
        return [x for x in ruleset if len(x.rules) > 0]

    def getCompiledSize_(self, subtables):
        if not subtables:
            return 0
        # We need to make a copy here because compiling
        # modifies the subtable (finalizing formats etc.)
        table = self.buildLookup_(copy.deepcopy(subtables))
        w = OTTableWriter(tableTag=self.table)
        # This standalone lookup has no LookupList parent to set its metadata.
        # Overflow reporting still needs a lookup name and index.
        w.name = "Lookup"
        w.repeatIndex = 0
        table.compile(w, self.font)
        size = len(w.getAllData())
        return size

    def build(self):
        """Build the lookup.

        Returns:
            An ``otTables.Lookup`` object representing the chained
            contextual positioning lookup.
        """
        subtables = []

        rulesets = self.rulesets()
        chaining = any(ruleset.hasPrefixOrSuffix for ruleset in rulesets)

        # https://github.com/fonttools/fonttools/issues/2539
        #
        # Unfortunately, as of 2022-03-07, Apple's CoreText renderer does not
        # correctly process GPOS7 lookups, so for now we force contextual
        # positioning lookups to be chaining (GPOS8).
        #
        # This seems to be fixed as of macOS 13.2, but we keep disabling this
        # for now until we are no longer concerned about old macOS versions.
        # But we allow people to opt-out of this with the config key below.
        write_gpos7 = self.font.cfg.get("fontTools.otlLib.builder:WRITE_GPOS7")
        # horrible separation of concerns breach
        if not write_gpos7 and self.subtable_type == "Pos":
            chaining = True

        pending_rulesets = list(reversed(rulesets))
        while pending_rulesets:
            ruleset = pending_rulesets.pop()
            # Determine format strategy. We try to build formats 1, 2 and 3
            # subtables and then work out which is best. candidates list holds
            # the subtables in each format for this ruleset (including a dummy
            # "format 0" to make the addressing match the format numbers).

            # We can always build a format 3 lookup by accumulating each of
            # the rules into a list, so start with that.
            candidates = [None, None, None, []]
            for rule in ruleset.rules:
                candidates[3].append(self.buildFormat3Subtable(rule, chaining))

            # Can we express the whole ruleset as a format 2 subtable?
            classdefs = ruleset.format2ClassDefs()
            if classdefs:
                candidates[2] = [
                    self.buildFormat2Subtable(ruleset, classdefs, chaining)
                ]

            if not ruleset.hasAnyGlyphClasses:
                candidates[1] = [self.buildFormat1Subtable(ruleset, chaining)]

            candidates_by_size = []
            for i in [1, 2, 3]:
                if candidates[i]:
                    try:
                        size = self.getCompiledSize_(candidates[i])
                    except OTLOffsetOverflowError as e:
                        log.warning(
                            "Contextual format %i at %s overflowed (%s)"
                            % (i, str(self.location), e)
                        )
                    else:
                        candidates_by_size.append((size, candidates[i]))

            if not candidates_by_size:
                if len(ruleset.rules) <= 1:
                    raise OpenTypeLibError("All candidates overflowed", self.location)
                # Retry smaller consecutive groups in the same lookup. Keeping
                # their order preserves the first-matching-rule semantics.
                midpoint = len(ruleset.rules) // 2
                for rules in (ruleset.rules[midpoint:], ruleset.rules[:midpoint]):
                    part = ChainContextualRuleset()
                    part.rules = rules
                    pending_rulesets.append(part)
                continue

            _min_size, winner = min(candidates_by_size, key=lambda x: x[0])
            subtables.extend(winner)

        # If we are not chaining, lookup type will be automatically fixed by
        # buildLookup_
        return self.buildLookup_(subtables)

    def buildFormat1Subtable(self, ruleset, chaining=True):
        st = self.newSubtable_(chaining=chaining)
        st.Format = 1
        st.populateDefaults()
        coverage = set()
        rulesetsByFirstGlyph = {}
        ruleAttr = self.ruleAttr_(format=1, chaining=chaining)

        for rule in ruleset.rules:
            ruleAsSubtable = self.newRule_(format=1, chaining=chaining)

            if chaining:
                ruleAsSubtable.BacktrackGlyphCount = len(rule.prefix)
                ruleAsSubtable.LookAheadGlyphCount = len(rule.suffix)
                ruleAsSubtable.Backtrack = [list(x)[0] for x in reversed(rule.prefix)]
                ruleAsSubtable.LookAhead = [list(x)[0] for x in rule.suffix]

                ruleAsSubtable.InputGlyphCount = len(rule.glyphs)
            else:
                ruleAsSubtable.GlyphCount = len(rule.glyphs)

            ruleAsSubtable.Input = [list(x)[0] for x in rule.glyphs[1:]]

            self.buildLookupList(rule, ruleAsSubtable)

            firstGlyph = list(rule.glyphs[0])[0]
            if firstGlyph not in rulesetsByFirstGlyph:
                coverage.add(firstGlyph)
                rulesetsByFirstGlyph[firstGlyph] = []
            rulesetsByFirstGlyph[firstGlyph].append(ruleAsSubtable)

        st.Coverage = buildCoverage(coverage, self.glyphMap)
        ruleSets = []
        for g in st.Coverage.glyphs:
            ruleSet = self.newRuleSet_(format=1, chaining=chaining)
            setattr(ruleSet, ruleAttr, rulesetsByFirstGlyph[g])
            setattr(ruleSet, f"{ruleAttr}Count", len(rulesetsByFirstGlyph[g]))
            ruleSets.append(ruleSet)

        setattr(st, self.ruleSetAttr_(format=1, chaining=chaining), ruleSets)
        setattr(
            st, self.ruleSetAttr_(format=1, chaining=chaining) + "Count", len(ruleSets)
        )

        return st

    def buildFormat2Subtable(self, ruleset, classdefs, chaining=True):
        st = self.newSubtable_(chaining=chaining)
        st.Format = 2
        st.populateDefaults()

        if chaining:
            (
                st.BacktrackClassDef,
                st.InputClassDef,
                st.LookAheadClassDef,
            ) = [c.build() for c in classdefs]
        else:
            st.ClassDef = classdefs[1].build()

        inClasses = classdefs[1].classes()

        classSets = []
        for _ in inClasses:
            classSet = self.newRuleSet_(format=2, chaining=chaining)
            classSets.append(classSet)

        coverage = set()
        classRuleAttr = self.ruleAttr_(format=2, chaining=chaining)

        for rule in ruleset.rules:
            ruleAsSubtable = self.newRule_(format=2, chaining=chaining)
            if chaining:
                ruleAsSubtable.BacktrackGlyphCount = len(rule.prefix)
                ruleAsSubtable.LookAheadGlyphCount = len(rule.suffix)
                # The glyphs in the rule may be list, tuple, odict_keys...
                # Order is not important anyway because they are guaranteed
                # to be members of the same class.
                ruleAsSubtable.Backtrack = [
                    st.BacktrackClassDef.classDefs[list(x)[0]]
                    for x in reversed(rule.prefix)
                ]
                ruleAsSubtable.LookAhead = [
                    st.LookAheadClassDef.classDefs[list(x)[0]] for x in rule.suffix
                ]

                ruleAsSubtable.InputGlyphCount = len(rule.glyphs)
                ruleAsSubtable.Input = [
                    st.InputClassDef.classDefs[list(x)[0]] for x in rule.glyphs[1:]
                ]
                setForThisRule = classSets[
                    st.InputClassDef.classDefs[list(rule.glyphs[0])[0]]
                ]
            else:
                ruleAsSubtable.GlyphCount = len(rule.glyphs)
                ruleAsSubtable.Class = [  # The spec calls this InputSequence
                    st.ClassDef.classDefs[list(x)[0]] for x in rule.glyphs[1:]
                ]
                setForThisRule = classSets[
                    st.ClassDef.classDefs[list(rule.glyphs[0])[0]]
                ]

            self.buildLookupList(rule, ruleAsSubtable)
            coverage |= set(rule.glyphs[0])

            getattr(setForThisRule, classRuleAttr).append(ruleAsSubtable)
            setattr(
                setForThisRule,
                f"{classRuleAttr}Count",
                getattr(setForThisRule, f"{classRuleAttr}Count") + 1,
            )
        for i, classSet in enumerate(classSets):
            if not getattr(classSet, classRuleAttr):
                # class sets can be null so replace nop sets with None
                classSets[i] = None
        setattr(st, self.ruleSetAttr_(format=2, chaining=chaining), classSets)
        setattr(
            st, self.ruleSetAttr_(format=2, chaining=chaining) + "Count", len(classSets)
        )
        st.Coverage = buildCoverage(coverage, self.glyphMap)
        return st

    def buildFormat3Subtable(self, rule, chaining=True):
        st = self.newSubtable_(chaining=chaining)
        st.Format = 3
        if chaining:
            self.setBacktrackCoverage_(rule.prefix, st)
            self.setLookAheadCoverage_(rule.suffix, st)
            self.setInputCoverage_(rule.glyphs, st)
        else:
            self.setCoverage_(rule.glyphs, st)
        self.buildLookupList(rule, st)
        return st

    def buildLookupList(self, rule, st):
        for sequenceIndex, lookupList in enumerate(rule.lookups):
            if lookupList is not None:
                if not isinstance(lookupList, list):
                    # Can happen with synthesised lookups
                    lookupList = [lookupList]
                for l in lookupList:
                    if l.lookup_index is None:
                        if isinstance(self, ChainContextPosBuilder):
                            other = "substitution"
                        else:
                            other = "positioning"
                        raise OpenTypeLibError(
                            "Missing index of the specified "
                            f"lookup, might be a {other} lookup",
                            self.location,
                        )
                    rec = self.newLookupRecord_(st)
                    rec.SequenceIndex = sequenceIndex
                    rec.LookupListIndex = l.lookup_index

    def add_subtable_break(self, location):
        self.rules.append(
            ChainContextualRule(
                self.SUBTABLE_BREAK_,
                self.SUBTABLE_BREAK_,
                self.SUBTABLE_BREAK_,
                [self.SUBTABLE_BREAK_],
            )
        )

    def newSubtable_(self, chaining=True):
        subtablename = f"Context{self.subtable_type}"
        if chaining:
            subtablename = "Chain" + subtablename
        st = getattr(ot, subtablename)()  # ot.ChainContextPos()/ot.ChainSubst()/etc.
        setattr(st, f"{self.subtable_type}Count", 0)
        setattr(st, f"{self.subtable_type}LookupRecord", [])
        return st

    # Format 1 and format 2 GSUB5/GSUB6/GPOS7/GPOS8 rulesets and rules form a family:
    #
    #       format 1 ruleset      format 1 rule      format 2 ruleset      format 2 rule
    # GSUB5 SubRuleSet            SubRule            SubClassSet           SubClassRule
    # GSUB6 ChainSubRuleSet       ChainSubRule       ChainSubClassSet      ChainSubClassRule
    # GPOS7 PosRuleSet            PosRule            PosClassSet           PosClassRule
    # GPOS8 ChainPosRuleSet       ChainPosRule       ChainPosClassSet      ChainPosClassRule
    #
    # The following functions generate the attribute names and subtables according
    # to this naming convention.
    def ruleSetAttr_(self, format=1, chaining=True):
        if format == 1:
            formatType = "Rule"
        elif format == 2:
            formatType = "Class"
        else:
            raise AssertionError(formatType)
        subtablename = f"{self.subtable_type[0:3]}{formatType}Set"  # Sub, not Subst.
        if chaining:
            subtablename = "Chain" + subtablename
        return subtablename

    def ruleAttr_(self, format=1, chaining=True):
        if format == 1:
            formatType = ""
        elif format == 2:
            formatType = "Class"
        else:
            raise AssertionError(formatType)
        subtablename = f"{self.subtable_type[0:3]}{formatType}Rule"  # Sub, not Subst.
        if chaining:
            subtablename = "Chain" + subtablename
        return subtablename

    def newRuleSet_(self, format=1, chaining=True):
        st = getattr(
            ot, self.ruleSetAttr_(format, chaining)
        )()  # ot.ChainPosRuleSet()/ot.SubRuleSet()/etc.
        st.populateDefaults()
        return st

    def newRule_(self, format=1, chaining=True):
        st = getattr(
            ot, self.ruleAttr_(format, chaining)
        )()  # ot.ChainPosClassRule()/ot.SubClassRule()/etc.
        st.populateDefaults()
        return st

    def attachSubtableWithCount_(
        self, st, subtable_name, count_name, existing=None, index=None, chaining=False
    ):
        if chaining:
            subtable_name = "Chain" + subtable_name
            count_name = "Chain" + count_name

        if not hasattr(st, count_name):
            setattr(st, count_name, 0)
            setattr(st, subtable_name, [])

        if existing:
            new_subtable = existing
        else:
            # Create a new, empty subtable from otTables
            new_subtable = getattr(ot, subtable_name)()

        setattr(st, count_name, getattr(st, count_name) + 1)

        if index:
            getattr(st, subtable_name).insert(index, new_subtable)
        else:
            getattr(st, subtable_name).append(new_subtable)

        return new_subtable

    def newLookupRecord_(self, st):
        return self.attachSubtableWithCount_(
            st,
            f"{self.subtable_type}LookupRecord",
            f"{self.subtable_type}Count",
            chaining=False,
        )  # Oddly, it isn't ChainSubstLookupRecord


class ChainContextPosBuilder(ChainContextualBuilder):
    """Builds a Chained Contextual Positioning (GPOS8) lookup.

    Users are expected to manually add rules to the ``rules`` attribute after
    the object has been initialized, e.g.::

        # pos [A B] [C D] x' lookup lu1 y' z' lookup lu2 E;

        prefix  = [ ["A", "B"], ["C", "D"] ]
        suffix  = [ ["E"] ]
        glyphs  = [ ["x"], ["y"], ["z"] ]
        lookups = [ [lu1], None,  [lu2] ]
        builder.rules.append( (prefix, glyphs, suffix, lookups) )

    Attributes:
        font (``fontTools.TTLib.TTFont``): A font object.
        location: A string or tuple representing the location in the original
            source which produced this lookup.
        rules: A list of tuples representing the rules in this lookup.
        lookupflag (int): The lookup's flag
        markFilterSet: Either ``None`` if no mark filtering set is used, or
            an integer representing the filtering set to be used for this
            lookup. If a mark filtering set is provided,
            `LOOKUP_FLAG_USE_MARK_FILTERING_SET` will be set on the lookup's
            flags.
    """

    def __init__(self, font, location):
        LookupBuilder.__init__(self, font, location, "GPOS", 8)
        self.rules = []
        self.subtable_type = "Pos"

    def find_chainable_single_pos(self, lookups, glyphs, value):
        """Helper for add_single_pos_chained_()"""
        res = None
        for lookup in lookups[::-1]:
            if lookup == self.SUBTABLE_BREAK_:
                return res
            if isinstance(lookup, SinglePosBuilder) and all(
                lookup.can_add(glyph, value) for glyph in glyphs
            ):
                res = lookup
        return res


class ChainContextSubstBuilder(ChainContextualBuilder):
    """Builds a Chained Contextual Substitution (GSUB6) lookup.

    Users are expected to manually add rules to the ``rules`` attribute after
    the object has been initialized, e.g.::

        # sub [A B] [C D] x' lookup lu1 y' z' lookup lu2 E;

        prefix  = [ ["A", "B"], ["C", "D"] ]
        suffix  = [ ["E"] ]
        glyphs  = [ ["x"], ["y"], ["z"] ]
        lookups = [ [lu1], None,  [lu2] ]
        builder.rules.append( (prefix, glyphs, suffix, lookups) )

    Attributes:
        font (``fontTools.TTLib.TTFont``): A font object.
        location: A string or tuple representing the location in the original
            source which produced this lookup.
        rules: A list of tuples representing the rules in this lookup.
        lookupflag (int): The lookup's flag
        markFilterSet: Either ``None`` if no mark filtering set is used, or
            an integer representing the filtering set to be used for this
            lookup. If a mark filtering set is provided,
            `LOOKUP_FLAG_USE_MARK_FILTERING_SET` will be set on the lookup's
            flags.
    """

    def __init__(self, font, location):
        LookupBuilder.__init__(self, font, location, "GSUB", 6)
        self.rules = []  # (prefix, input, suffix, lookups)
        self.subtable_type = "Subst"

    def getAlternateGlyphs(self):
        result = {}
        for rule in self.rules:
            if rule.is_subtable_break:
                continue
            for lookups in rule.lookups:
                if not isinstance(lookups, list):
                    lookups = [lookups]
                for lookup in lookups:
                    if lookup is not None:
                        alts = lookup.getAlternateGlyphs()
                        for glyph, replacements in alts.items():
                            alts_for_glyph = result.setdefault(glyph, [])
                            alts_for_glyph.extend(
                                g for g in replacements if g not in alts_for_glyph
                            )
        return result

    def find_chainable_subst(self, mapping, builder_class):
        """Helper for add_{single,multi}_subst_chained_()"""
        res = None
        for rule in self.rules[::-1]:
            if rule.is_subtable_break:
                return res
            for sub in rule.lookups:
                if isinstance(sub, builder_class) and not any(
                    g in mapping and mapping[g] != sub.mapping[g] for g in sub.mapping
                ):
                    res = sub
        return res

    def find_chainable_alternate_subst(self, glyph):
        """Helper for add_alternate_subst()"""
        res = None
        for rule in self.rules[::-1]:
            if rule.is_subtable_break:
                return res
            for sub in rule.lookups:
                if isinstance(sub, AlternateSubstBuilder) and (
                    glyph not in sub.alternates
                ):
                    res = sub
        return res

    def find_chainable_ligature_subst(self, glyphs, replacement):
        """Helper for add_ligature_subst_chained_()"""
        res = None
        for rule in self.rules[::-1]:
            if rule.is_subtable_break:
                return res
            for sub in rule.lookups:
                if not isinstance(sub, LigatureSubstBuilder):
                    continue
                if all(
                    sub.ligatures.get(seq, replacement) == replacement
                    for seq in itertools.product(*glyphs)
                ):
                    res = sub
        return res


class LigatureSubstBuilder(LookupBuilder):
    """Builds a Ligature Substitution (GSUB4) lookup.

    Users are expected to manually add ligatures to the ``ligatures``
    attribute after the object has been initialized, e.g.::

        # sub f i by f_i;
        builder.ligatures[("f","f","i")] = "f_f_i"

    Attributes:
        font (``fontTools.TTLib.TTFont``): A font object.
        location: A string or tuple representing the location in the original
            source which produced this lookup.
        ligatures: An ordered dictionary mapping a tuple of glyph names to the
            ligature glyphname.
        lookupflag (int): The lookup's flag
        markFilterSet: Either ``None`` if no mark filtering set is used, or
            an integer representing the filtering set to be used for this
            lookup. If a mark filtering set is provided,
            `LOOKUP_FLAG_USE_MARK_FILTERING_SET` will be set on the lookup's
            flags.
    """

    def __init__(self, font, location):
        LookupBuilder.__init__(self, font, location, "GSUB", 4)
        self.ligatures = OrderedDict()  # {('f','f','i'): 'f_f_i'}

    def equals(self, other):
        return LookupBuilder.equals(self, other) and self.ligatures == other.ligatures

    def build(self):
        """Build the lookup.

        Returns:
            An ``otTables.Lookup`` object representing the ligature
            substitution lookup.
        """
        subtables = self.build_subst_subtables(
            self.ligatures, buildLigatureSubstSubtable
        )
        return self.buildLookup_(subtables)

    def getAlternateGlyphs(self):
        # https://github.com/fonttools/fonttools/issues/3845
        return {
            components[0]: [ligature]
            for components, ligature in self.ligatures.items()
            if len(components) == 1
        }

    def add_subtable_break(self, location):
        self.ligatures[(self.SUBTABLE_BREAK_, location)] = self.SUBTABLE_BREAK_


class MultipleSubstBuilder(LookupBuilder):
    """Builds a Multiple Substitution (GSUB2) lookup.

    Users are expected to manually add substitutions to the ``mapping``
    attribute after the object has been initialized, e.g.::

        # sub uni06C0 by uni06D5.fina hamza.above;
        builder.mapping["uni06C0"] = [ "uni06D5.fina", "hamza.above"]

    Attributes:
        font (``fontTools.TTLib.TTFont``): A font object.
        location: A string or tuple representing the location in the original
            source which produced this lookup.
        mapping: An ordered dictionary mapping a glyph name to a list of
            substituted glyph names.
        lookupflag (int): The lookup's flag
        markFilterSet: Either ``None`` if no mark filtering set is used, or
            an integer representing the filtering set to be used for this
            lookup. If a mark filtering set is provided,
            `LOOKUP_FLAG_USE_MARK_FILTERING_SET` will be set on the lookup's
            flags.
    """

    def __init__(self, font, location):
        LookupBuilder.__init__(self, font, location, "GSUB", 2)
        self.mapping = OrderedDict()

    def equals(self, other):
        return LookupBuilder.equals(self, other) and self.mapping == other.mapping

    def build(self):
        subtables = self.build_subst_subtables(self.mapping, buildMultipleSubstSubtable)
        return self.buildLookup_(subtables)

    def getAlternateGlyphs(self):
        # https://github.com/fonttools/fonttools/issues/3845
        return {
            glyph: replacements
            for glyph, replacements in self.mapping.items()
            if len(replacements) == 1
        }

    def add_subtable_break(self, location):
        self.mapping[(self.SUBTABLE_BREAK_, location)] = self.SUBTABLE_BREAK_


class CursivePosBuilder(LookupBuilder):
    """Builds a Cursive Positioning (GPOS3) lookup.

    Attributes:
        font (``fontTools.TTLib.TTFont``): A font object.
        location: A string or tuple representing the location in the original
            source which produced this lookup.
        attachments: An ordered dictionary mapping a glyph name to a two-element
            tuple of ``otTables.Anchor`` objects.
        lookupflag (int): The lookup's flag
        markFilterSet: Either ``None`` if no mark filtering set is used, or
            an integer representing the filtering set to be used for this
            lookup. If a mark filtering set is provided,
            `LOOKUP_FLAG_USE_MARK_FILTERING_SET` will be set on the lookup's
            flags.
    """

    def __init__(self, font, location):
        LookupBuilder.__init__(self, font, location, "GPOS", 3)
        self.attachments = {}

    def equals(self, other):
        return (
            LookupBuilder.equals(self, other) and self.attachments == other.attachments
        )

    def add_attachment(self, location, glyphs, entryAnchor, exitAnchor):
        """Adds attachment information to the cursive positioning lookup.

        Args:
            location: A string or tuple representing the location in the
                original source which produced this lookup. (Unused.)
            glyphs: A list of glyph names sharing these entry and exit
                anchor locations.
            entryAnchor: A ``otTables.Anchor`` object representing the
                entry anchor, or ``None`` if no entry anchor is present.
            exitAnchor: A ``otTables.Anchor`` object representing the
                exit anchor, or ``None`` if no exit anchor is present.
        """
        for glyph in glyphs:
            self.attachments[glyph] = (entryAnchor, exitAnchor)

    def build(self):
        """Build the lookup.

        Returns:
            An ``otTables.Lookup`` object representing the cursive
            positioning lookup.
        """
        attachments = [{}]
        for key in self.attachments:
            if key[0] == self.SUBTABLE_BREAK_:
                attachments.append({})
            else:
                attachments[-1][key] = self.attachments[key]
        subtables = [buildCursivePosSubtable(s, self.glyphMap) for s in attachments]
        return self.buildLookup_(subtables)

    def add_subtable_break(self, location):
        self.attachments[(self.SUBTABLE_BREAK_, location)] = (
            self.SUBTABLE_BREAK_,
            self.SUBTABLE_BREAK_,
        )


class MarkBasePosBuilder(LookupBuilder):
    """Builds a Mark-To-Base Positioning (GPOS4) lookup.

    Users are expected to manually add marks and bases to the ``marks``
    and ``bases`` attributes after the object has been initialized, e.g.::

        builder.marks["acute"]   = (0, a1)
        builder.marks["grave"]   = (0, a1)
        builder.marks["cedilla"] = (1, a2)
        builder.bases["a"] = {0: a3, 1: a5}
        builder.bases["b"] = {0: a4, 1: a5}

    Attributes:
        font (``fontTools.TTLib.TTFont``): A font object.
        location: A string or tuple representing the location in the original
            source which produced this lookup.
        marks: An dictionary mapping a glyph name to a two-element
            tuple containing a mark class ID and ``otTables.Anchor`` object.
        bases: An dictionary mapping a glyph name to a dictionary of
            mark class IDs and ``otTables.Anchor`` object.
        lookupflag (int): The lookup's flag
        markFilterSet: Either ``None`` if no mark filtering set is used, or
            an integer representing the filtering set to be used for this
            lookup. If a mark filtering set is provided,
            `LOOKUP_FLAG_USE_MARK_FILTERING_SET` will be set on the lookup's
            flags.
    """

    def __init__(self, font, location):
        LookupBuilder.__init__(self, font, location, "GPOS", 4)
        self.marks = {}  # glyphName -> (markClassName, anchor)
        self.bases = {}  # glyphName -> {markClassName: anchor}
        self.subtables_ = []

    def get_subtables_(self):
        subtables_ = self.subtables_
        if self.bases or self.marks:
            subtables_.append((self.marks, self.bases))
        return subtables_

    def equals(self, other):
        return (
            LookupBuilder.equals(self, other)
            and self.get_subtables_() == other.get_subtables_()
        )

    def inferGlyphClasses(self):
        result = {}
        for marks, bases in self.get_subtables_():
            result.update({glyph: 1 for glyph in bases})
            result.update({glyph: 3 for glyph in marks})
        return result

    def build(self):
        """Build the lookup.

        Returns:
            An ``otTables.Lookup`` object representing the mark-to-base
            positioning lookup.
        """
        subtables = []
        for subtable in self.get_subtables_():
            markClasses = self.buildMarkClasses_(subtable[0])
            marks = {}
            for mark, (mc, anchor) in subtable[0].items():
                if mc not in markClasses:
                    raise ValueError(
                        "Mark class %s not found for mark glyph %s" % (mc, mark)
                    )
                marks[mark] = (markClasses[mc], anchor)
            bases = {}
            for glyph, anchors in subtable[1].items():
                bases[glyph] = {}
                for mc, anchor in anchors.items():
                    if mc not in markClasses:
                        raise ValueError(
                            "Mark class %s not found for base glyph %s" % (mc, glyph)
                        )
                    bases[glyph][markClasses[mc]] = anchor
            subtables.append(buildMarkBasePosSubtable(marks, bases, self.glyphMap))
        return self.buildLookup_(subtables)

    def add_subtable_break(self, location):
        self.subtables_.append((self.marks, self.bases))
        self.marks = {}
        self.bases = {}


class MarkLigPosBuilder(LookupBuilder):
    """Builds a Mark-To-Ligature Positioning (GPOS5) lookup.

    Users are expected to manually add marks and bases to the ``marks``
    and ``ligatures`` attributes after the object has been initialized, e.g.::

        builder.marks["acute"]   = (0, a1)
        builder.marks["grave"]   = (0, a1)
        builder.marks["cedilla"] = (1, a2)
        builder.ligatures["f_i"] = [
            { 0: a3, 1: a5 }, # f
            { 0: a4, 1: a5 }  # i
        ]

    Attributes:
        font (``fontTools.TTLib.TTFont``): A font object.
        location: A string or tuple representing the location in the original
            source which produced this lookup.
        marks: An dictionary mapping a glyph name to a two-element
            tuple containing a mark class ID and ``otTables.Anchor`` object.
        ligatures: An dictionary mapping a glyph name to an array with one
            element for each ligature component. Each array element should be
            a dictionary mapping mark class IDs to ``otTables.Anchor`` objects.
        lookupflag (int): The lookup's flag
        markFilterSet: Either ``None`` if no mark filtering set is used, or
            an integer representing the filtering set to be used for this
            lookup. If a mark filtering set is provided,
            `LOOKUP_FLAG_USE_MARK_FILTERING_SET` will be set on the lookup's
            flags.
    """

    def __init__(self, font, location):
        LookupBuilder.__init__(self, font, location, "GPOS", 5)
        self.marks = {}  # glyphName -> (markClassName, anchor)
        self.ligatures = {}  # glyphName -> [{markClassName: anchor}, ...]
        self.subtables_ = []

    def get_subtables_(self):
        subtables_ = self.subtables_
        if self.ligatures or self.marks:
            subtables_.append((self.marks, self.ligatures))
        return subtables_

    def equals(self, other):
        return (
            LookupBuilder.equals(self, other)
            and self.get_subtables_() == other.get_subtables_()
        )

    def inferGlyphClasses(self):
        result = {}
        for marks, ligatures in self.get_subtables_():
            result.update({glyph: 2 for glyph in ligatures})
            result.update({glyph: 3 for glyph in marks})
        return result

    def build(self):
        """Build the lookup.

        Returns:
            An ``otTables.Lookup`` object representing the mark-to-ligature
            positioning lookup.
        """
        subtables = []
        for subtable in self.get_subtables_():
            markClasses = self.buildMarkClasses_(subtable[0])
            marks = {
                mark: (markClasses[mc], anchor)
                for mark, (mc, anchor) in subtable[0].items()
            }
            ligs = {}
            for lig, components in subtable[1].items():
                ligs[lig] = []
                for c in components:
                    ligs[lig].append({markClasses[mc]: a for mc, a in c.items()})
            subtables.append(buildMarkLigPosSubtable(marks, ligs, self.glyphMap))
        return self.buildLookup_(subtables)

    def add_subtable_break(self, location):
        self.subtables_.append((self.marks, self.ligatures))
        self.marks = {}
        self.ligatures = {}


class MarkMarkPosBuilder(LookupBuilder):
    """Builds a Mark-To-Mark Positioning (GPOS6) lookup.

    Users are expected to manually add marks and bases to the ``marks``
    and ``baseMarks`` attributes after the object has been initialized, e.g.::

        builder.marks["acute"]     = (0, a1)
        builder.marks["grave"]     = (0, a1)
        builder.marks["cedilla"]   = (1, a2)
        builder.baseMarks["acute"] = {0: a3}

    Attributes:
        font (``fontTools.TTLib.TTFont``): A font object.
        location: A string or tuple representing the location in the original
            source which produced this lookup.
        marks: An dictionary mapping a glyph name to a two-element
            tuple containing a mark class ID and ``otTables.Anchor`` object.
        baseMarks: An dictionary mapping a glyph name to a dictionary
            containing one item: a mark class ID and a ``otTables.Anchor`` object.
        lookupflag (int): The lookup's flag
        markFilterSet: Either ``None`` if no mark filtering set is used, or
            an integer representing the filtering set to be used for this
            lookup. If a mark filtering set is provided,
            `LOOKUP_FLAG_USE_MARK_FILTERING_SET` will be set on the lookup's
            flags.
    """

    def __init__(self, font, location):
        LookupBuilder.__init__(self, font, location, "GPOS", 6)
        self.marks = {}  # glyphName -> (markClassName, anchor)
        self.baseMarks = {}  # glyphName -> {markClassName: anchor}
        self.subtables_ = []

    def get_subtables_(self):
        subtables_ = self.subtables_
