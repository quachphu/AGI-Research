    return True


def genericDictValidator(value: Any, prototype: GenericDict) -> bool:
    """
    Generic. (Added at version 3.)
    """
    # not a dict
    if not isinstance(value, Mapping):
        return False
    # missing required keys
    for key, (typ, required) in prototype.items():
        if not required:
            continue
        if key not in value:
            return False
    # unknown keys
    for key in value.keys():
        if key not in prototype:
            return False
    # incorrect types
    for key, v in value.items():
        prototypeType, required = prototype[key]
        if v is None and not required:
            continue
        if not isinstance(v, prototypeType):
            return False
    return True


# --------------
# fontinfo.plist
# --------------

# Data Validators


def fontInfoStyleMapStyleNameValidator(value: Any) -> bool:
    """
    Version 2+.
    """
    options = ["regular", "italic", "bold", "bold italic"]
    return value in options


def fontInfoOpenTypeGaspRangeRecordsValidator(value: Any) -> bool:
    """
    Version 3+.
    """
    if not isinstance(value, list):
        return False
    if len(value) == 0:
        return True
    validBehaviors = [0, 1, 2, 3]
    dictPrototype: GenericDict = dict(
        rangeMaxPPEM=(int, True), rangeGaspBehavior=(list, True)
    )
    ppemOrder = []
    for rangeRecord in value:
        if not genericDictValidator(rangeRecord, dictPrototype):
            return False
        ppem = rangeRecord["rangeMaxPPEM"]
        behavior = rangeRecord["rangeGaspBehavior"]
        ppemValidity = genericNonNegativeIntValidator(ppem)
        if not ppemValidity:
            return False
        behaviorValidity = genericIntListValidator(behavior, validBehaviors)
        if not behaviorValidity:
            return False
        ppemOrder.append(ppem)
    if ppemOrder != sorted(ppemOrder):
        return False
    return True


def fontInfoOpenTypeHeadCreatedValidator(value: Any) -> bool:
    """
    Version 2+.
    """
    # format: 0000/00/00 00:00:00
    if not isinstance(value, str):
        return False
    # basic formatting
    if not len(value) == 19:
        return False
    if value.count(" ") != 1:
        return False
    strDate, strTime = value.split(" ")
    if strDate.count("/") != 2:
        return False
    if strTime.count(":") != 2:
        return False
    # date
    strYear, strMonth, strDay = strDate.split("/")
    if len(strYear) != 4:
        return False
    if len(strMonth) != 2:
        return False
    if len(strDay) != 2:
        return False
    try:
        intYear = int(strYear)
        intMonth = int(strMonth)
        intDay = int(strDay)
    except ValueError:
        return False
    if intMonth < 1 or intMonth > 12:
        return False
    monthMaxDay = calendar.monthrange(intYear, intMonth)[1]
    if intDay < 1 or intDay > monthMaxDay:
        return False
    # time
    strHour, strMinute, strSecond = strTime.split(":")
    if len(strHour) != 2:
        return False
    if len(strMinute) != 2:
        return False
    if len(strSecond) != 2:
        return False
    try:
        intHour = int(strHour)
        intMinute = int(strMinute)
        intSecond = int(strSecond)
    except ValueError:
        return False
    if intHour < 0 or intHour > 23:
        return False
    if intMinute < 0 or intMinute > 59:
        return False
    if intSecond < 0 or intSecond > 59:
        return False
    # fallback
    return True


def fontInfoOpenTypeNameRecordsValidator(value: Any) -> bool:
    """
    Version 3+.
    """
    if not isinstance(value, list):
        return False
    dictPrototype: GenericDict = dict(
        nameID=(int, True),
        platformID=(int, True),
        encodingID=(int, True),
        languageID=(int, True),
        string=(str, True),
    )
    for nameRecord in value:
        if not genericDictValidator(nameRecord, dictPrototype):
            return False
    return True


def fontInfoOpenTypeOS2WeightClassValidator(value: Any) -> bool:
    """
    Version 2+.
    """
    if not isinstance(value, int):
        return False
    if value < 0:
        return False
    return True


def fontInfoOpenTypeOS2WidthClassValidator(value: Any) -> bool:
    """
    Version 2+.
    """
    if not isinstance(value, int):
        return False
    if value < 1:
        return False
    if value > 9:
        return False
    return True


def fontInfoVersion2OpenTypeOS2PanoseValidator(values: Any) -> bool:
    """
    Version 2.
    """
    if not isinstance(values, (list, tuple)):
        return False
    if len(values) != 10:
        return False
    for value in values:
        if not isinstance(value, int):
            return False
    # XXX further validation?
    return True


def fontInfoVersion3OpenTypeOS2PanoseValidator(values: Any) -> bool:
    """
    Version 3+.
    """
    if not isinstance(values, (list, tuple)):
        return False
    if len(values) != 10:
        return False
    for value in values:
        if not isinstance(value, int):
            return False
        if value < 0:
            return False
    # XXX further validation?
    return True


def fontInfoOpenTypeOS2FamilyClassValidator(values: Any) -> bool:
    """
    Version 2+.
    """
    if not isinstance(values, (list, tuple)):
        return False
    if len(values) != 2:
        return False
    for value in values:
        if not isinstance(value, int):
            return False
    classID, subclassID = values
    if classID < 0 or classID > 14:
        return False
    if subclassID < 0 or subclassID > 15:
        return False
    return True


def fontInfoPostscriptBluesValidator(values: Any) -> bool:
    """
    Version 2+.
    """
    if not isinstance(values, (list, tuple)):
        return False
    if len(values) > 14:
        return False
    if len(values) % 2:
        return False
    for value in values:
        if not isinstance(value, numberTypes):
            return False
    return True


def fontInfoPostscriptOtherBluesValidator(values: Any) -> bool:
    """
    Version 2+.
    """
    if not isinstance(values, (list, tuple)):
        return False
    if len(values) > 10:
        return False
    if len(values) % 2:
        return False
    for value in values:
        if not isinstance(value, numberTypes):
            return False
    return True


def fontInfoPostscriptStemsValidator(values: Any) -> bool:
    """
    Version 2+.
    """
    if not isinstance(values, (list, tuple)):
        return False
    if len(values) > 12:
        return False
    for value in values:
        if not isinstance(value, numberTypes):
            return False
    return True


def fontInfoPostscriptWindowsCharacterSetValidator(value: Any) -> bool:
    """
    Version 2+.
    """
    validValues = list(range(1, 21))
    if value not in validValues:
        return False
    return True


def fontInfoWOFFMetadataUniqueIDValidator(value: Any) -> bool:
    """
    Version 3+.
    """
    dictPrototype: GenericDict = dict(id=(str, True))
    if not genericDictValidator(value, dictPrototype):
        return False
    return True


def fontInfoWOFFMetadataVendorValidator(value: Any) -> bool:
    """
    Version 3+.
    """
    dictPrototype: GenericDict = {
        "name": (str, True),
        "url": (str, False),
        "dir": (str, False),
        "class": (str, False),
    }
    if not genericDictValidator(value, dictPrototype):
        return False
    if "dir" in value and value.get("dir") not in ("ltr", "rtl"):
        return False
    return True


def fontInfoWOFFMetadataCreditsValidator(value: Any) -> bool:
    """
    Version 3+.
    """
    dictPrototype: GenericDict = dict(credits=(list, True))
    if not genericDictValidator(value, dictPrototype):
        return False
    if not len(value["credits"]):
        return False
    dictPrototype = {
        "name": (str, True),
        "url": (str, False),
        "role": (str, False),
        "dir": (str, False),
        "class": (str, False),
    }
    for credit in value["credits"]:
        if not genericDictValidator(credit, dictPrototype):
            return False
        if "dir" in credit and credit.get("dir") not in ("ltr", "rtl"):
            return False
    return True


def fontInfoWOFFMetadataDescriptionValidator(value: Any) -> bool:
    """
    Version 3+.
    """
    dictPrototype: GenericDict = dict(url=(str, False), text=(list, True))
    if not genericDictValidator(value, dictPrototype):
        return False
    for text in value["text"]:
        if not fontInfoWOFFMetadataTextValue(text):
            return False
    return True


def fontInfoWOFFMetadataLicenseValidator(value: Any) -> bool:
    """
    Version 3+.
    """
    dictPrototype: GenericDict = dict(
        url=(str, False), text=(list, False), id=(str, False)
    )
    if not genericDictValidator(value, dictPrototype):
        return False
    if "text" in value:
        for text in value["text"]:
            if not fontInfoWOFFMetadataTextValue(text):
                return False
    return True


def fontInfoWOFFMetadataTrademarkValidator(value: Any) -> bool:
    """
    Version 3+.
    """
    dictPrototype: GenericDict = dict(text=(list, True))
    if not genericDictValidator(value, dictPrototype):
        return False
    for text in value["text"]:
        if not fontInfoWOFFMetadataTextValue(text):
            return False
    return True


def fontInfoWOFFMetadataCopyrightValidator(value: Any) -> bool:
    """
    Version 3+.
    """
    dictPrototype: GenericDict = dict(text=(list, True))
    if not genericDictValidator(value, dictPrototype):
        return False
    for text in value["text"]:
        if not fontInfoWOFFMetadataTextValue(text):
            return False
    return True


def fontInfoWOFFMetadataLicenseeValidator(value: Any) -> bool:
    """
    Version 3+.
    """
    dictPrototype: GenericDict = {
        "name": (str, True),
        "dir": (str, False),
        "class": (str, False),
    }
    if not genericDictValidator(value, dictPrototype):
        return False
    if "dir" in value and value.get("dir") not in ("ltr", "rtl"):
        return False
    return True


def fontInfoWOFFMetadataTextValue(value: Any) -> bool:
    """
    Version 3+.
    """
    dictPrototype: GenericDict = {
        "text": (str, True),
        "language": (str, False),
        "dir": (str, False),
        "class": (str, False),
    }
    if not genericDictValidator(value, dictPrototype):
        return False
    if "dir" in value and value.get("dir") not in ("ltr", "rtl"):
        return False
    return True


def fontInfoWOFFMetadataExtensionsValidator(value: Any) -> bool:
    """
    Version 3+.
    """
    if not isinstance(value, list):
        return False
    if not value:
        return False
    for extension in value:
        if not fontInfoWOFFMetadataExtensionValidator(extension):
            return False
    return True


def fontInfoWOFFMetadataExtensionValidator(value: Any) -> bool:
    """
    Version 3+.
    """
    dictPrototype: GenericDict = dict(
        names=(list, False), items=(list, True), id=(str, False)
    )
    if not genericDictValidator(value, dictPrototype):
        return False
    if "names" in value:
        for name in value["names"]:
            if not fontInfoWOFFMetadataExtensionNameValidator(name):
                return False
    for item in value["items"]:
        if not fontInfoWOFFMetadataExtensionItemValidator(item):
            return False
    return True


def fontInfoWOFFMetadataExtensionItemValidator(value: Any) -> bool:
    """
    Version 3+.
    """
    dictPrototype: GenericDict = dict(
        id=(str, False), names=(list, True), values=(list, True)
    )
    if not genericDictValidator(value, dictPrototype):
        return False
    for name in value["names"]:
        if not fontInfoWOFFMetadataExtensionNameValidator(name):
            return False
    for val in value["values"]:
        if not fontInfoWOFFMetadataExtensionValueValidator(val):
            return False
    return True


def fontInfoWOFFMetadataExtensionNameValidator(value: Any) -> bool:
    """
    Version 3+.
    """
    dictPrototype: GenericDict = {
        "text": (str, True),
        "language": (str, False),
        "dir": (str, False),
        "class": (str, False),
    }
    if not genericDictValidator(value, dictPrototype):
        return False
    if "dir" in value and value.get("dir") not in ("ltr", "rtl"):
        return False
    return True


def fontInfoWOFFMetadataExtensionValueValidator(value: Any) -> bool:
    """
    Version 3+.
    """
    dictPrototype: GenericDict = {
        "text": (str, True),
        "language": (str, False),
        "dir": (str, False),
        "class": (str, False),
    }
    if not genericDictValidator(value, dictPrototype):
        return False
    if "dir" in value and value.get("dir") not in ("ltr", "rtl"):
        return False
    return True


# ----------
# Guidelines
# ----------


def guidelinesValidator(value: Any, identifiers: Optional[set[str]] = None) -> bool:
    """
    Version 3+.
    """
    if not isinstance(value, list):
        return False
    if identifiers is None:
        identifiers = set()
    for guide in value:
        if not guidelineValidator(guide):
            return False
        identifier = guide.get("identifier")
        if identifier is not None:
            if identifier in identifiers:
                return False
            identifiers.add(identifier)
    return True


_guidelineDictPrototype: GenericDict = dict(
    x=((int, float), False),
    y=((int, float), False),
    angle=((int, float), False),
    name=(str, False),
    color=(str, False),
    identifier=(str, False),
)


def guidelineValidator(value: Any) -> bool:
    """
    Version 3+.
    """
    if not genericDictValidator(value, _guidelineDictPrototype):
        return False

    angle = value.get("angle")
    # angle must be between 0 and 360
    if angle is not None:
        if angle < 0:
            return False
        if angle > 360:
            return False
    # identifier must be 1 or more characters
    identifier = value.get("identifier")
    if identifier is not None and not identifierValidator(identifier):
        return False
    # color must follow the proper format
    color = value.get("color")
    if color is not None and not colorValidator(color):
        return False
    return True


# -------
# Anchors
# -------


def anchorsValidator(value: Any, identifiers: Optional[set[str]] = None) -> bool:
    """
    Version 3+.
    """
    if not isinstance(value, list):
        return False
    if identifiers is None:
        identifiers = set()
    for anchor in value:
        if not anchorValidator(anchor):
            return False
        identifier = anchor.get("identifier")
        if identifier is not None:
            if identifier in identifiers:
                return False
            identifiers.add(identifier)
    return True


_anchorDictPrototype: GenericDict = dict(
    x=((int, float), False),
    y=((int, float), False),
    name=(str, False),
    color=(str, False),
    identifier=(str, False),
)


def anchorValidator(value: Any) -> bool:
    """
    Version 3+.
    """
    if not genericDictValidator(value, _anchorDictPrototype):
        return False
    x = value.get("x")
    y = value.get("y")
    # x and y must be present
    if x is None or y is None:
        return False
    # identifier must be 1 or more characters
    identifier = value.get("identifier")
    if identifier is not None and not identifierValidator(identifier):
        return False
    # color must follow the proper format
    color = value.get("color")
    if color is not None and not colorValidator(color):
        return False
    return True


# ----------
# Identifier
# ----------


def identifierValidator(value: Any) -> bool:
    """
    Version 3+.

    >>> identifierValidator("a")
    True
    >>> identifierValidator("")
    False
    >>> identifierValidator("a" * 101)
    False
    """
    validCharactersMin = 0x20
    validCharactersMax = 0x7E
    if not isinstance(value, str):
        return False
    if not value:
        return False
    if len(value) > 100:
        return False
    for c in value:
        i = ord(c)
        if i < validCharactersMin or i > validCharactersMax:
            return False
    return True


# -----
# Color
# -----


def colorValidator(value: Any) -> bool:
    """
    Version 3+.

    >>> colorValidator("0,0,0,0")
    True
    >>> colorValidator(".5,.5,.5,.5")
    True
    >>> colorValidator("0.5,0.5,0.5,0.5")
    True
    >>> colorValidator("1,1,1,1")
    True

    >>> colorValidator("2,0,0,0")
    False
    >>> colorValidator("0,2,0,0")
    False
    >>> colorValidator("0,0,2,0")
    False
    >>> colorValidator("0,0,0,2")
    False

    >>> colorValidator("1r,1,1,1")
    False
    >>> colorValidator("1,1g,1,1")
    False
    >>> colorValidator("1,1,1b,1")
    False
    >>> colorValidator("1,1,1,1a")
    False

    >>> colorValidator("1 1 1 1")
    False
    >>> colorValidator("1 1,1,1")
    False
    >>> colorValidator("1,1 1,1")
    False
    >>> colorValidator("1,1,1 1")
    False

    >>> colorValidator("1, 1, 1, 1")
    True
    """
    if not isinstance(value, str):
        return False
    parts = value.split(",")
    if len(parts) != 4:
        return False
    for part in parts:
        part = part.strip()
        converted = False
        number: IntFloat
        try:
            number = int(part)
            converted = True
        except ValueError:
            pass
        if not converted:
            try:
                number = float(part)
                converted = True
            except ValueError:
                pass
        if not converted:
            return False
        if not 0 <= number <= 1:
            return False
    return True


# -----
# image
# -----

pngSignature: bytes = b"\x89PNG\r\n\x1a\n"

_imageDictPrototype: GenericDict = dict(
    fileName=(str, True),
    xScale=((int, float), False),
    xyScale=((int, float), False),
    yxScale=((int, float), False),
    yScale=((int, float), False),
    xOffset=((int, float), False),
    yOffset=((int, float), False),
    color=(str, False),
)


def imageValidator(value):
    """
    Version 3+.
    """
    if not genericDictValidator(value, _imageDictPrototype):
        return False
    # fileName must be one or more characters
    if not value["fileName"]:
        return False
    # color must follow the proper format
    color = value.get("color")
    if color is not None and not colorValidator(color):
        return False
    return True


def pngValidator(
    path: Optional[str] = None,
    data: Optional[bytes] = None,
    fileObj: Optional[Any] = None,
) -> tuple[bool, Any]:
    """
    Version 3+.

    This checks the signature of the image data.
    """
    assert path is not None or data is not None or fileObj is not None
    if path is not None:
        with open(path, "rb") as f:
            signature = f.read(8)
    elif data is not None:
        signature = data[:8]
    elif fileObj is not None:
        pos = fileObj.tell()
        signature = fileObj.read(8)
        fileObj.seek(pos)
    if signature != pngSignature:
        return False, "Image does not begin with the PNG signature."
    return True, None


# -------------------
# layercontents.plist
# -------------------


def layerContentsValidator(
    value: Any, ufoPathOrFileSystem: Union[str, fs.base.FS]
) -> tuple[bool, Optional[str]]:
    """
    Check the validity of layercontents.plist.
    Version 3+.
    """
    if isinstance(ufoPathOrFileSystem, fs.base.FS):
        fileSystem = ufoPathOrFileSystem
    else:
        fileSystem = fs.osfs.OSFS(ufoPathOrFileSystem)

    bogusFileMessage = "layercontents.plist in not in the correct format."
    # file isn't in the right format
    if not isinstance(value, list):
        return False, bogusFileMessage
    # work through each entry
    usedLayerNames = set()
    usedDirectories = set()
    contents = {}
    for entry in value:
        # layer entry in the incorrect format
        if not isinstance(entry, list):
            return False, bogusFileMessage
        if not len(entry) == 2:
            return False, bogusFileMessage
        for i in entry:
            if not isinstance(i, str):
                return False, bogusFileMessage
        layerName, directoryName = entry
        # check directory naming
        if directoryName != "glyphs":
            if not directoryName.startswith("glyphs."):
                return (
                    False,
                    "Invalid directory name (%s) in layercontents.plist."
                    % directoryName,
                )
        if len(layerName) == 0:
            return False, "Empty layer name in layercontents.plist."
        # directory doesn't exist
        if not fileSystem.exists(directoryName):
            return False, "A glyphset does not exist at %s." % directoryName
        # default layer name
        if layerName == "public.default" and directoryName != "glyphs":
            return (
                False,
                "The name public.default is being used by a layer that is not the default.",
            )
        # check usage
        if layerName in usedLayerNames:
            return (
                False,
                "The layer name %s is used by more than one layer." % layerName,
            )
        usedLayerNames.add(layerName)
        if directoryName in usedDirectories:
            return (
                False,
                "The directory %s is used by more than one layer." % directoryName,
            )
        usedDirectories.add(directoryName)
        # store
        contents[layerName] = directoryName
    # missing default layer
    foundDefault = "glyphs" in contents.values()
    if not foundDefault:
        return False, "The required default glyph set is not in the UFO."
    return True, None


# ------------
# groups.plist
# ------------


def groupsValidator(value: Any) -> tuple[bool, Optional[str]]:
    """
    Check the validity of the groups.
    Version 3+ (though it's backwards compatible with UFO 1 and UFO 2).

    >>> groups = {"A" : ["A", "A"], "A2" : ["A"]}
    >>> groupsValidator(groups)
    (True, None)

    >>> groups = {"" : ["A"]}
    >>> valid, msg = groupsValidator(groups)
    >>> valid
    False
    >>> print(msg)
    A group has an empty name.

    >>> groups = {"public.awesome" : ["A"]}
    >>> groupsValidator(groups)
    (True, None)

    >>> groups = {"public.kern1." : ["A"]}
    >>> valid, msg = groupsValidator(groups)
    >>> valid
    False
    >>> print(msg)
    The group data contains a kerning group with an incomplete name.
    >>> groups = {"public.kern2." : ["A"]}
    >>> valid, msg = groupsValidator(groups)
    >>> valid
    False
    >>> print(msg)
    The group data contains a kerning group with an incomplete name.

    >>> groups = {"public.kern1.A" : ["A"], "public.kern2.A" : ["A"]}
    >>> groupsValidator(groups)
    (True, None)

    >>> groups = {"public.kern1.A1" : ["A"], "public.kern1.A2" : ["A"]}
    >>> valid, msg = groupsValidator(groups)
    >>> valid
    False
    >>> print(msg)
    The glyph "A" occurs in too many kerning groups.
    """
    bogusFormatMessage = "The group data is not in the correct format."
    if not isDictEnough(value):
        return False, bogusFormatMessage
    firstSideMapping: dict[str, str] = {}
    secondSideMapping: dict[str, str] = {}
    for groupName, glyphList in value.items():
        if not isinstance(groupName, (str)):
            return False, bogusFormatMessage
        if not isinstance(glyphList, (list, tuple)):
            return False, bogusFormatMessage
        if not groupName:
            return False, "A group has an empty name."
        if groupName.startswith("public."):
            if not groupName.startswith("public.kern1.") and not groupName.startswith(
                "public.kern2."
            ):
                # unknown public.* name. silently skip.
                continue
            else:
                if len("public.kernN.") == len(groupName):
                    return (
                        False,
                        "The group data contains a kerning group with an incomplete name.",
                    )
            if groupName.startswith("public.kern1."):
                d = firstSideMapping
            else:
                d = secondSideMapping
            for glyphName in glyphList:
                if not isinstance(glyphName, str):
                    return (
                        False,
                        "The group data %s contains an invalid member." % groupName,
                    )
                if glyphName in d:
                    return (
                        False,
                        'The glyph "%s" occurs in too many kerning groups.' % glyphName,
                    )
                d[glyphName] = groupName
    return True, None


# -------------
# kerning.plist
# -------------


def kerningValidator(data: Any) -> tuple[bool, Optional[str]]:
    """
    Check the validity of the kerning data structure.
    Version 3+ (though it's backwards compatible with UFO 1 and UFO 2).

    >>> kerning = {"A" : {"B" : 100}}
    >>> kerningValidator(kerning)
    (True, None)

    >>> kerning = {"A" : ["B"]}
    >>> valid, msg = kerningValidator(kerning)
    >>> valid
    False
    >>> print(msg)
    The kerning data is not in the correct format.

    >>> kerning = {"A" : {"B" : "100"}}
    >>> valid, msg = kerningValidator(kerning)
    >>> valid
    False
    >>> print(msg)
    The kerning data is not in the correct format.
    """
    bogusFormatMessage = "The kerning data is not in the correct format."
    if not isinstance(data, Mapping):
        return False, bogusFormatMessage
    for first, secondDict in data.items():
        if not isinstance(first, str):
            return False, bogusFormatMessage
        elif not isinstance(secondDict, Mapping):
            return False, bogusFormatMessage
        for second, value in secondDict.items():
            if not isinstance(second, str):
                return False, bogusFormatMessage
            elif not isinstance(value, numberTypes):
                return False, bogusFormatMessage
    return True, None


# -------------
# lib.plist/lib
# -------------

_bogusLibFormatMessage = "The lib data is not in the correct format: %s"


def fontLibValidator(value: Any) -> tuple[bool, Optional[str]]:
    """
    Check the validity of the lib.
    Version 3+ (though it's backwards compatible with UFO 1 and UFO 2).

    >>> lib = {"foo" : "bar"}
    >>> fontLibValidator(lib)
    (True, None)

    >>> lib = {"public.awesome" : "hello"}
    >>> fontLibValidator(lib)
    (True, None)

    >>> lib = {"public.glyphOrder" : ["A", "C", "B"]}
    >>> fontLibValidator(lib)
    (True, None)

    >>> lib = "hello"
    >>> valid, msg = fontLibValidator(lib)
    >>> valid
    False
    >>> print(msg)  # doctest: +ELLIPSIS
    The lib data is not in the correct format: expected a dictionary, ...

    >>> lib = {1: "hello"}
    >>> valid, msg = fontLibValidator(lib)
    >>> valid
    False
    >>> print(msg)
    The lib key is not properly formatted: expected str, found int: 1

    >>> lib = {"public.glyphOrder" : "hello"}
    >>> valid, msg = fontLibValidator(lib)
    >>> valid
    False
    >>> print(msg)  # doctest: +ELLIPSIS
    public.glyphOrder is not properly formatted: expected list or tuple,...

    >>> lib = {"public.glyphOrder" : ["A", 1, "B"]}
    >>> valid, msg = fontLibValidator(lib)
    >>> valid
    False
    >>> print(msg)  # doctest: +ELLIPSIS
    public.glyphOrder is not properly formatted: expected str,...
    """
    if not isDictEnough(value):
        reason = "expected a dictionary, found %s" % type(value).__name__
        return False, _bogusLibFormatMessage % reason
    for key, value in value.items():
        if not isinstance(key, str):
            return False, (
                "The lib key is not properly formatted: expected str, found %s: %r"
                % (type(key).__name__, key)
            )
        # public.glyphOrder
        if key == "public.glyphOrder":
            bogusGlyphOrderMessage = "public.glyphOrder is not properly formatted: %s"
            if not isinstance(value, (list, tuple)):
                reason = "expected list or tuple, found %s" % type(value).__name__
                return False, bogusGlyphOrderMessage % reason
            for glyphName in value:
                if not isinstance(glyphName, str):
                    reason = "expected str, found %s" % type(glyphName).__name__
                    return False, bogusGlyphOrderMessage % reason
    return True, None


# --------
# GLIF lib
# --------


def glyphLibValidator(value: Any) -> tuple[bool, Optional[str]]:
    """
    Check the validity of the lib.
    Version 3+ (though it's backwards compatible with UFO 1 and UFO 2).

    >>> lib = {"foo" : "bar"}
    >>> glyphLibValidator(lib)
    (True, None)

    >>> lib = {"public.awesome" : "hello"}
    >>> glyphLibValidator(lib)
    (True, None)

    >>> lib = {"public.markColor" : "1,0,0,0.5"}
    >>> glyphLibValidator(lib)
    (True, None)

    >>> lib = {"public.markColor" : 1}
    >>> valid, msg = glyphLibValidator(lib)
    >>> valid
    False
    >>> print(msg)
    public.markColor is not properly formatted.
    """
    if not isDictEnough(value):
        reason = "expected a dictionary, found %s" % type(value).__name__
        return False, _bogusLibFormatMessage % reason
    for key, value in value.items():
        if not isinstance(key, str):
            reason = "key (%s) should be a string" % key
            return False, _bogusLibFormatMessage % reason
        # public.markColor
        if key == "public.markColor":
            if not colorValidator(value):
                return False, "public.markColor is not properly formatted."
    return True, None


if __name__ == "__main__":
    import doctest

    doctest.testmod()
<|endoftext|>"""
A library for importing .ufo files and their descendants.
Refer to http://unifiedfontobject.org for the UFO specification.

The main interfaces are the :class:`.UFOReader` and :class:`.UFOWriter`
classes, which support versions 1, 2, and 3 of the UFO specification.

Set variables are available for external use that list the font
info attribute names for the `fontinfo.plist` formats. These are:

- :obj:`.fontInfoAttributesVersion1`
- :obj:`.fontInfoAttributesVersion2`
- :obj:`.fontInfoAttributesVersion3`

A set listing the `fontinfo.plist` attributes that were deprecated
in version 2 is available for external use:

- :obj:`.deprecatedFontInfoAttributesVersion2`

Functions that do basic validation on values for `fontinfo.plist`
are available for external use. These are

- :func:`.validateFontInfoVersion2ValueForAttribute`
- :func:`.validateFontInfoVersion3ValueForAttribute`

Value conversion functions are available for converting
`fontinfo.plist` values between the possible format versions.

- :func:`.convertFontInfoValueForAttributeFromVersion1ToVersion2`
- :func:`.convertFontInfoValueForAttributeFromVersion2ToVersion1`
- :func:`.convertFontInfoValueForAttributeFromVersion2ToVersion3`
- :func:`.convertFontInfoValueForAttributeFromVersion3ToVersion2`
"""

from __future__ import annotations

import enum
import logging
import os
import zipfile
from collections import OrderedDict
from copy import deepcopy
from os import fsdecode
from typing import IO, TYPE_CHECKING, Any, Optional, Union, cast

from fontTools.misc import filesystem as fs
from fontTools.misc import plistlib
from fontTools.ufoLib.converters import convertUFO1OrUFO2KerningToUFO3Kerning
from fontTools.ufoLib.errors import UFOLibError
from fontTools.ufoLib.filenames import userNameToFileName
from fontTools.ufoLib.utils import (
    BaseFormatVersion,
    normalizeFormatVersion,
    numberTypes,
)
from fontTools.ufoLib.validators import *

if TYPE_CHECKING:
    from logging import Logger

    from fontTools.annotations import (
        GlyphNameToFileNameFunc,
        K,
        KerningDict,
        KerningGroups,
        KerningNested,
        PathOrFS,
        PathStr,
        UFOFormatVersionInput,
        V,
    )
    from fontTools.misc.filesystem._base import FS
    from fontTools.ufoLib.glifLib import GlyphSet

KerningGroupRenameMaps = dict[str, dict[str, str]]
LibDict = dict[str, Any]
LayerOrderList = Optional[list[Optional[str]]]
AttributeDataDict = dict[str, Any]
FontInfoAttributes = dict[str, AttributeDataDict]

# client code can check this to see if the upstream `fs` package is being used
haveFS = fs._haveFS

__all__: list[str] = [
    "haveFS",
    "makeUFOPath",
    "UFOLibError",
    "UFOReader",
    "UFOWriter",
    "UFOReaderWriter",
    "UFOFileStructure",
    "fontInfoAttributesVersion1",
    "fontInfoAttributesVersion2",
    "fontInfoAttributesVersion3",
    "deprecatedFontInfoAttributesVersion2",
    "validateFontInfoVersion2ValueForAttribute",
    "validateFontInfoVersion3ValueForAttribute",
    "convertFontInfoValueForAttributeFromVersion1ToVersion2",
    "convertFontInfoValueForAttributeFromVersion2ToVersion1",
]

__version__: str = "3.0.0"


logger: Logger = logging.getLogger(__name__)


# ---------
# Constants
# ---------

DEFAULT_GLYPHS_DIRNAME: str = "glyphs"
DATA_DIRNAME: str = "data"
IMAGES_DIRNAME: str = "images"
METAINFO_FILENAME: str = "metainfo.plist"
FONTINFO_FILENAME: str = "fontinfo.plist"
LIB_FILENAME: str = "lib.plist"
GROUPS_FILENAME: str = "groups.plist"
KERNING_FILENAME: str = "kerning.plist"
FEATURES_FILENAME: str = "features.fea"
LAYERCONTENTS_FILENAME: str = "layercontents.plist"
LAYERINFO_FILENAME: str = "layerinfo.plist"

DEFAULT_LAYER_NAME: str = "public.default"


class UFOFormatVersion(BaseFormatVersion):
    FORMAT_1_0 = (1, 0)
    FORMAT_2_0 = (2, 0)
    FORMAT_3_0 = (3, 0)


class UFOFileStructure(enum.Enum):
    ZIP = "zip"
    PACKAGE = "package"


# --------------
# Shared Methods
# --------------


class _UFOBaseIO:
    if TYPE_CHECKING:
        fs: FS
        _havePreviousFile: bool

    def getFileModificationTime(self, path: PathStr) -> Optional[float]:
        """
        Returns the modification time for the file at the given path, as a
        floating point number giving the number of seconds since the epoch.
        The path must be relative to the UFO path.
        Returns None if the file does not exist.
        """
        try:
            dt = self.fs.getinfo(fsdecode(path), namespaces=["details"]).modified
        except (fs.errors.MissingInfoNamespace, fs.errors.ResourceNotFound):
            return None
        else:
            if dt is not None:
                return dt.timestamp()
            return None

    def _getPlist(self, fileName: str, default: Optional[Any] = None) -> Any:
        """
        Read a property list relative to the UFO filesystem's root.
        Raises UFOLibError if the file is missing and default is None,
        otherwise default is returned.

        The errors that could be raised during the reading of a plist are
        unpredictable and/or too large to list, so, a blind try: except:
        is done. If an exception occurs, a UFOLibError will be raised.
        """
        try:
            with self.fs.open(fileName, "rb") as f:
                return plistlib.load(f)
        except fs.errors.ResourceNotFound:
            if default is None:
                raise UFOLibError(
                    "'%s' is missing on %s. This file is required" % (fileName, self.fs)
                )
            else:
                return default
        except Exception as e:
            # TODO(anthrotype): try to narrow this down a little
            raise UFOLibError(f"'{fileName}' could not be read on {self.fs}: {e}")

    def _writePlist(self, fileName: str, obj: Any) -> None:
        """
        Write a property list to a file relative to the UFO filesystem's root.

        Do this sort of atomically, making it harder to corrupt existing files,
        for example when plistlib encounters an error halfway during write.
        This also checks to see if text matches the text that is already in the
        file at path. If so, the file is not rewritten so that the modification
        date is preserved.

        The errors that could be raised during the writing of a plist are
        unpredictable and/or too large to list, so, a blind try: except: is done.
        If an exception occurs, a UFOLibError will be raised.
        """
        if self._havePreviousFile:
            try:
                data = plistlib.dumps(obj)
            except Exception as e:
                raise UFOLibError(
                    "'%s' could not be written on %s because "
                    "the data is not properly formatted: %s" % (fileName, self.fs, e)
                )
            if self.fs.exists(fileName) and data == self.fs.readbytes(fileName):
                return
            self.fs.writebytes(fileName, data)
        else:
            with self.fs.open(fileName, mode="wb") as fp:
                try:
                    plistlib.dump(obj, fp)
                except Exception as e:
                    raise UFOLibError(
                        "'%s' could not be written on %s because "
                        "the data is not properly formatted: %s"
                        % (fileName, self.fs, e)
                    )


# ----------
# UFO Reader
# ----------


class UFOReader(_UFOBaseIO):
    """Read the various components of a .ufo.

    Attributes:
        path: An :class:`os.PathLike` object pointing to the .ufo.
        validate: A boolean indicating if the data read should be
          validated. Defaults to `True`.

    By default read data is validated. Set ``validate`` to
    ``False`` to not validate the data.
    """

    def __init__(self, path: PathOrFS, validate: bool = True) -> None:
        # Only call __fspath__ if path is not already a str or FS object
        if not isinstance(path, (str, fs.base.FS)) and hasattr(path, "__fspath__"):
            path = path.__fspath__()

        if isinstance(path, str):
            structure = _sniffFileStructure(path)
            parentFS: FS
            try:
                if structure is UFOFileStructure.ZIP:
                    parentFS = fs.zipfs.ZipFS(path, write=False, encoding="utf-8")  # type: ignore[abstract]
                else:
                    parentFS = fs.osfs.OSFS(path)
            except fs.errors.CreateFailed as e:
                raise UFOLibError(f"unable to open '{path}': {e}")

            if structure is UFOFileStructure.ZIP:
                # .ufoz zip files must contain a single root directory, with arbitrary
                # name, containing all the UFO files
                rootDirs = [
                    p.name
                    for p in parentFS.scandir("/")
                    # exclude macOS metadata contained in zip file
                    if p.is_dir and p.name != "__MACOSX"
                ]
                if len(rootDirs) == 1:
                    # 'ClosingSubFS' ensures that the parent zip file is closed when
                    # its root subdirectory is closed
                    self.fs: FS = parentFS.opendir(
                        rootDirs[0], factory=fs.subfs.ClosingSubFS
                    )
                else:
                    raise UFOLibError(
                        "Expected exactly 1 root directory, found %d" % len(rootDirs)
                    )
            else:
                # normal UFO 'packages' are just a single folder
                self.fs = parentFS
            # when passed a path string, we make sure we close the newly opened fs
            # upon calling UFOReader.close method or context manager's __exit__
            self._shouldClose: bool = True
            self._fileStructure = structure
        elif isinstance(path, fs.base.FS):
            filesystem: FS = path
            try:
                filesystem.check()
            except fs.errors.FilesystemClosed:
                raise UFOLibError("the filesystem '%s' is closed" % path)
            else:
                self.fs = filesystem
            try:
                path = filesystem.getsyspath("/")
            except fs.errors.NoSysPath:
                # network or in-memory FS may not map to the local one
                path = str(filesystem)
            # when user passed an already initialized fs instance, it is her
            # responsibility to close it, thus UFOReader.close/__exit__ are no-op
            self._shouldClose = False
            # default to a 'package' structure
            self._fileStructure = UFOFileStructure.PACKAGE
        else:
            raise TypeError(
                "Expected a path string or fs.base.FS object, found '%s'"
                % type(path).__name__
            )
        self._path: str = fsdecode(path)
        self._validate: bool = validate
        self._upConvertedKerningData: Optional[dict[str, Any]] = None

        try:
            self.readMetaInfo(validate=validate)
        except UFOLibError:
            self.close()
            raise

    # properties

    def _get_path(self) -> str:
        import warnings

        warnings.warn(
            "The 'path' attribute is deprecated; use the 'fs' attribute instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._path

    path: property = property(_get_path, doc="The path of the UFO (DEPRECATED).")

    def _get_formatVersion(self) -> int:
        import warnings

        warnings.warn(
            "The 'formatVersion' attribute is deprecated; use the 'formatVersionTuple'",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._formatVersion.major

    formatVersion = property(
        _get_formatVersion,
        doc="The (major) format version of the UFO. DEPRECATED: Use formatVersionTuple",
    )

    @property
    def formatVersionTuple(self) -> tuple[int, int]:
        """The (major, minor) format version of the UFO.
        This is determined by reading metainfo.plist during __init__.
        """
        return self._formatVersion

    def _get_fileStructure(self) -> Any:
        return self._fileStructure

    fileStructure: property = property(
        _get_fileStructure,
        doc=(
            "The file structure of the UFO: "
            "either UFOFileStructure.ZIP or UFOFileStructure.PACKAGE"
        ),
    )

    # up conversion

    def _upConvertKerning(self, validate: bool) -> None:
        """
        Up convert kerning and groups in UFO 1 and 2.
        The data will be held internally until each bit of data
        has been retrieved. The conversion of both must be done
        at once, so the raw data is cached and an error is raised
        if one bit of data becomes obsolete before it is called.

        ``validate`` will validate the data.
        """
        if self._upConvertedKerningData:
            testKerning = self._readKerning()
            if testKerning != self._upConvertedKerningData["originalKerning"]:
                raise UFOLibError(
                    "The data in kerning.plist has been modified since it was converted to UFO 3 format."
                )
            testGroups = self._readGroups()
            if testGroups != self._upConvertedKerningData["originalGroups"]:
                raise UFOLibError(
                    "The data in groups.plist has been modified since it was converted to UFO 3 format."
                )
        else:
            groups = self._readGroups()
            if validate:
                invalidFormatMessage = "groups.plist is not properly formatted."
                if not isinstance(groups, dict):
                    raise UFOLibError(invalidFormatMessage)
                for groupName, glyphList in groups.items():
                    if not isinstance(groupName, str):
                        raise UFOLibError(invalidFormatMessage)
                    elif not isinstance(glyphList, list):
                        raise UFOLibError(invalidFormatMessage)
                    for glyphName in glyphList:
                        if not isinstance(glyphName, str):
                            raise UFOLibError(invalidFormatMessage)
            self._upConvertedKerningData = dict(
                kerning={},
                originalKerning=self._readKerning(),
                groups={},
                originalGroups=groups,
            )
            # convert kerning and groups
            kerning, groups, conversionMaps = convertUFO1OrUFO2KerningToUFO3Kerning(
                self._upConvertedKerningData["originalKerning"],
                deepcopy(self._upConvertedKerningData["originalGroups"]),
                self.getGlyphSet(),
            )
            # store
            self._upConvertedKerningData["kerning"] = kerning
            self._upConvertedKerningData["groups"] = groups
            self._upConvertedKerningData["groupRenameMaps"] = conversionMaps

    # support methods

    def readBytesFromPath(self, path: PathStr) -> Optional[bytes]:
        """
        Returns the bytes in the file at the given path.
        The path must be relative to the UFO's filesystem root.
        Returns None if the file does not exist.
        """
        try:
            return self.fs.readbytes(fsdecode(path))
        except fs.errors.ResourceNotFound:
            return None

    def getReadFileForPath(
        self, path: PathStr, encoding: Optional[str] = None
    ) -> Optional[Union[IO[bytes], IO[str]]]:
        """
        Returns a file (or file-like) object for the file at the given path.
        The path must be relative to the UFO path.
        Returns None if the file does not exist.
        By default the file is opened in binary mode (reads bytes).
        If encoding is passed, the file is opened in text mode (reads str).

        Note: The caller is responsible for closing the open file.
        """
        path = fsdecode(path)
        try:
            if encoding is None:
                return self.fs.open(path, mode="rb")
            else:
                return self.fs.open(path, mode="r", encoding=encoding)
        except fs.errors.ResourceNotFound:
            return None

    # metainfo.plist

    def _readMetaInfo(self, validate: Optional[bool] = None) -> dict[str, Any]:
        """
        Read metainfo.plist and return raw data. Only used for internal operations.

        ``validate`` will validate the read data, by default it is set
        to the class's validate value, can be overridden.
        """
        if validate is None:
            validate = self._validate
        data = self._getPlist(METAINFO_FILENAME)
        if validate and not isinstance(data, dict):
            raise UFOLibError("metainfo.plist is not properly formatted.")
        try:
            formatVersionMajor = data["formatVersion"]
        except KeyError:
            raise UFOLibError(
                f"Missing required formatVersion in '{METAINFO_FILENAME}' on {self.fs}"
            )
        formatVersionMinor = data.setdefault("formatVersionMinor", 0)

        try:
            formatVersion = UFOFormatVersion((formatVersionMajor, formatVersionMinor))
        except ValueError as e:
            unsupportedMsg = (
                f"Unsupported UFO format ({formatVersionMajor}.{formatVersionMinor}) "
                f"in '{METAINFO_FILENAME}' on {self.fs}"
            )
            if validate:
                from fontTools.ufoLib.errors import UnsupportedUFOFormat

                raise UnsupportedUFOFormat(unsupportedMsg) from e

            formatVersion = UFOFormatVersion.default()
            logger.warning(
                "%s. Assuming the latest supported version (%s). "
                "Some data may be skipped or parsed incorrectly",
                unsupportedMsg,
                formatVersion,
            )
        data["formatVersionTuple"] = formatVersion
        return data

    def readMetaInfo(self, validate: Optional[bool] = None) -> None:
        """
        Read metainfo.plist and set formatVersion. Only used for internal operations.

        ``validate`` will validate the read data, by default it is set
        to the class's validate value, can be overridden.
        """
        data = self._readMetaInfo(validate=validate)
        self._formatVersion = data["formatVersionTuple"]

    # groups.plist

    def _readGroups(self) -> dict[str, list[str]]:
        groups = self._getPlist(GROUPS_FILENAME, {})
        # remove any duplicate glyphs in a kerning group
        for groupName, glyphList in groups.items():
            if groupName.startswith(("public.kern1.", "public.kern2.")):
                groups[groupName] = list(OrderedDict.fromkeys(glyphList))
        return groups

    def readGroups(self, validate: Optional[bool] = None) -> dict[str, list[str]]:
        """
        Read groups.plist. Returns a dict.
        ``validate`` will validate the read data, by default it is set to the
        class's validate value, can be overridden.
        """
        if validate is None:
            validate = self._validate
        # handle up conversion
        if self._formatVersion < UFOFormatVersion.FORMAT_3_0:
            self._upConvertKerning(validate)
            groups = cast(dict, self._upConvertedKerningData)["groups"]
        # normal
        else:
            groups = self._readGroups()
        if validate:
            valid, message = groupsValidator(groups)
            if not valid:
                raise UFOLibError(message)
        return groups

    def getKerningGroupConversionRenameMaps(
        self, validate: Optional[bool] = None
    ) -> KerningGroupRenameMaps:
        """
        Get maps defining the renaming that was done during any
        needed kerning group conversion. This method returns a
        dictionary of this form::

                {
                        "side1" : {"old group name" : "new group name"},
                        "side2" : {"old group name" : "new group name"}
                }

        When no conversion has been performed, the side1 and side2
        dictionaries will be empty.

        ``validate`` will validate the groups, by default it is set to the
        class's validate value, can be overridden.
        """
        if validate is None:
            validate = self._validate
        if self._formatVersion >= UFOFormatVersion.FORMAT_3_0:
            return dict(side1={}, side2={})
        # use the public group reader to force the load and
        # conversion of the data if it hasn't happened yet.
        self.readGroups(validate=validate)
        return cast(dict, self._upConvertedKerningData)["groupRenameMaps"]

    # fontinfo.plist

    def _readInfo(self, validate: bool) -> dict[str, Any]:
        data = self._getPlist(FONTINFO_FILENAME, {})
        if validate and not isinstance(data, dict):
            raise UFOLibError("fontinfo.plist is not properly formatted.")
        return data

    def readInfo(self, info: Any, validate: Optional[bool] = None) -> None:
        """
        Read fontinfo.plist. It requires an object that allows
        setting attributes with names that follow the fontinfo.plist
        version 3 specification. This will write the attributes
        defined in the file into the object.

        ``validate`` will validate the read data, by default it is set to the
        class's validate value, can be overridden.
        """
        if validate is None:
            validate = self._validate
        infoDict = self._readInfo(validate)
        infoDataToSet = {}
        # version 1
        if self._formatVersion == UFOFormatVersion.FORMAT_1_0:
            for attr in fontInfoAttributesVersion1:
                value = infoDict.get(attr)
                if value is not None:
                    infoDataToSet[attr] = value
            infoDataToSet = _convertFontInfoDataVersion1ToVersion2(infoDataToSet)
            infoDataToSet = _convertFontInfoDataVersion2ToVersion3(infoDataToSet)
        # version 2
        elif self._formatVersion == UFOFormatVersion.FORMAT_2_0:
            for attr, dataValidationDict in list(
                fontInfoAttributesVersion2ValueData.items()
            ):
                value = infoDict.get(attr)
                if value is None:
                    continue
                infoDataToSet[attr] = value
            infoDataToSet = _convertFontInfoDataVersion2ToVersion3(infoDataToSet)
        # version 3.x
        elif self._formatVersion.major == UFOFormatVersion.FORMAT_3_0.major:
            for attr, dataValidationDict in list(
                fontInfoAttributesVersion3ValueData.items()
            ):
                value = infoDict.get(attr)
                if value is None:
                    continue
                infoDataToSet[attr] = value
        # unsupported version
        else:
            raise NotImplementedError(self._formatVersion)
        # validate data
        if validate:
            infoDataToSet = validateInfoVersion3Data(infoDataToSet)
        # populate the object
        for attr, value in list(infoDataToSet.items()):
            try:
                setattr(info, attr, value)
            except AttributeError:
                raise UFOLibError(
                    "The supplied info object does not support setting a necessary attribute (%s)."
                    % attr
                )

    # kerning.plist

    def _readKerning(self) -> KerningNested:
        data = self._getPlist(KERNING_FILENAME, {})
        return data

    def readKerning(self, validate: Optional[bool] = None) -> KerningDict:
        """
        Read kerning.plist. Returns a dict.

        ``validate`` will validate the kerning data, by default it is set to the
        class's validate value, can be overridden.
        """
        if validate is None:
            validate = self._validate
        # handle up conversion
        if self._formatVersion < UFOFormatVersion.FORMAT_3_0:
            self._upConvertKerning(validate)
            kerningNested = cast(dict, self._upConvertedKerningData)["kerning"]
        # normal
        else:
            kerningNested = self._readKerning()
        if validate:
            valid, message = kerningValidator(kerningNested)
            if not valid:
                raise UFOLibError(message)
        # flatten
        kerning = {}
        for left in kerningNested:
            for right in kerningNested[left]:
                value = kerningNested[left][right]
                kerning[left, right] = value
        return kerning

    # lib.plist

    def readLib(self, validate: Optional[bool] = None) -> dict[str, Any]:
        """
        Read lib.plist. Returns a dict.

        ``validate`` will validate the data, by default it is set to the
        class's validate value, can be overridden.
        """
        if validate is None:
            validate = self._validate
        data = self._getPlist(LIB_FILENAME, {})
        if validate:
            valid, message = fontLibValidator(data)
            if not valid:
                raise UFOLibError(message)
        return data

    # features.fea

    def readFeatures(self) -> str:
        """
        Read features.fea. Return a string.
        The returned string is empty if the file is missing.
        """
        try:
            with self.fs.open(FEATURES_FILENAME, "r", encoding="utf-8-sig") as f:
                return f.read()
        except fs.errors.ResourceNotFound:
            return ""

    # glyph sets & layers

    def _readLayerContents(self, validate: bool) -> list[tuple[str, str]]:
        """
        Rebuild the layer contents list by checking what glyphsets
        are available on disk.

        ``validate`` will validate the layer contents.
        """
        if self._formatVersion < UFOFormatVersion.FORMAT_3_0:
            return [(DEFAULT_LAYER_NAME, DEFAULT_GLYPHS_DIRNAME)]
        contents = self._getPlist(LAYERCONTENTS_FILENAME)
        if validate:
            valid, error = layerContentsValidator(contents, self.fs)
            if not valid:
                raise UFOLibError(error)
        return contents

    def getLayerNames(self, validate: Optional[bool] = None) -> list[str]:
        """
        Get the ordered layer names from layercontents.plist.

        ``validate`` will validate the data, by default it is set to the
        class's validate value, can be overridden.
        """
        if validate is None:
            validate = self._validate
        layerContents = self._readLayerContents(validate)
        layerNames = [layerName for layerName, directoryName in layerContents]
        return layerNames

    def getDefaultLayerName(self, validate: Optional[bool] = None) -> str:
        """
        Get the default layer name from layercontents.plist.

        ``validate`` will validate the data, by default it is set to the
        class's validate value, can be overridden.
        """
        if validate is None:
            validate = self._validate
        layerContents = self._readLayerContents(validate)
        for layerName, layerDirectory in layerContents:
            if layerDirectory == DEFAULT_GLYPHS_DIRNAME:
                return layerName
        # this will already have been raised during __init__
        raise UFOLibError("The default layer is not defined in layercontents.plist.")

    def getGlyphSet(
        self,
        layerName: Optional[str] = None,
        validateRead: Optional[bool] = None,
        validateWrite: Optional[bool] = None,
    ) -> GlyphSet:
        """
        Return the GlyphSet associated with the
        glyphs directory mapped to layerName
        in the UFO. If layerName is not provided,
        the name retrieved with getDefaultLayerName
        will be used.

        ``validateRead`` will validate the read data, by default it is set to the
        class's validate value, can be overridden.
        ``validateWrite`` will validate the written data, by default it is set to the
        class's validate value, can be overridden.
        """
        from fontTools.ufoLib.glifLib import GlyphSet

        if validateRead is None:
            validateRead = self._validate
        if validateWrite is None:
            validateWrite = self._validate
        if layerName is None:
            layerName = self.getDefaultLayerName(validate=validateRead)
        directory = None
        layerContents = self._readLayerContents(validateRead)
        for storedLayerName, storedLayerDirectory in layerContents:
            if layerName == storedLayerName:
                directory = storedLayerDirectory
                break
        if directory is None:
            raise UFOLibError('No glyphs directory is mapped to "%s".' % layerName)
        try:
            glyphSubFS = self.fs.opendir(directory)
        except fs.errors.ResourceNotFound:
            raise UFOLibError(f"No '{directory}' directory for layer '{layerName}'")
        return GlyphSet(
            glyphSubFS,
            ufoFormatVersion=self._formatVersion,
            validateRead=validateRead,
            validateWrite=validateWrite,
            expectContentsFile=True,
        )

    def getCharacterMapping(
        self, layerName: Optional[str] = None, validate: Optional[bool] = None
    ) -> dict[int, list[str]]:
        """
        Return a dictionary that maps unicode values (ints) to
        lists of glyph names.
        """
        if validate is None:
            validate = self._validate
        glyphSet = self.getGlyphSet(
            layerName, validateRead=validate, validateWrite=True
        )
        allUnicodes = glyphSet.getUnicodes()
        cmap: dict[int, list[str]] = {}
        for glyphName, unicodes in allUnicodes.items():
            for code in unicodes:
                if code in cmap:
                    cmap[code].append(glyphName)
                else:
                    cmap[code] = [glyphName]
        return cmap

    # /data

    def getDataDirectoryListing(self) -> list[str]:
        """
        Returns a list of all files in the data directory.
        The returned paths will be relative to the UFO.
        This will not list directory names, only file names.
        Thus, empty directories will be skipped.
        """
        try:
            self._dataFS = self.fs.opendir(DATA_DIRNAME)
        except fs.errors.ResourceNotFound:
            return []
        except fs.errors.DirectoryExpected:
            raise UFOLibError('The UFO contains a "data" file instead of a directory.')
        try:
            # fs Walker.files method returns "absolute" paths (in terms of the
            # root of the 'data' SubFS), so we strip the leading '/' to make
            # them relative
            return [p.lstrip("/") for p in self._dataFS.walk.files()]
        except fs.errors.ResourceError:
            return []

    def getImageDirectoryListing(self, validate: Optional[bool] = None) -> list[str]:
        """
        Returns a list of all image file names in
        the images directory. Each of the images will
        have been verified to have the PNG signature.

        ``validate`` will validate the data, by default it is set to the
        class's validate value, can be overridden.
        """
        if self._formatVersion < UFOFormatVersion.FORMAT_3_0:
            return []
        if validate is None:
            validate = self._validate
        try:
            self._imagesFS = imagesFS = self.fs.opendir(IMAGES_DIRNAME)
        except fs.errors.ResourceNotFound:
            return []
        except fs.errors.DirectoryExpected:
            raise UFOLibError(
                'The UFO contains an "images" file instead of a directory.'
            )
        result = []
        for path in imagesFS.scandir("/"):
            if path.is_dir:
                # silently skip this as version control
                # systems often have hidden directories
                continue
            if validate:
                with imagesFS.open(path.name, "rb") as fp:
                    valid, error = pngValidator(fileObj=fp)
                if valid:
                    result.append(path.name)
            else:
                result.append(path.name)
        return result

    def readData(self, fileName: PathStr) -> bytes:
        """
        Return bytes for the file named 'fileName' inside the 'data/' directory.
        """
        fileName = fsdecode(fileName)
        try:
            try:
                dataFS = self._dataFS
            except AttributeError:
                # in case readData is called before getDataDirectoryListing
                dataFS = self.fs.opendir(DATA_DIRNAME)
            data = dataFS.readbytes(fileName)
        except fs.errors.ResourceNotFound:
            raise UFOLibError(f"No data file named '{fileName}' on {self.fs}")
        return data

    def readImage(self, fileName: PathStr, validate: Optional[bool] = None) -> bytes:
        """
        Return image data for the file named fileName.

        ``validate`` will validate the data, by default it is set to the
        class's validate value, can be overridden.
        """
        if validate is None:
            validate = self._validate
        if self._formatVersion < UFOFormatVersion.FORMAT_3_0:
            raise UFOLibError(
                f"Reading images is not allowed in UFO {self._formatVersion.major}."
            )
        fileName = fsdecode(fileName)
        try:
            try:
                imagesFS = self._imagesFS
            except AttributeError:
                # in case readImage is called before getImageDirectoryListing
                imagesFS = self.fs.opendir(IMAGES_DIRNAME)
            data = imagesFS.readbytes(fileName)
        except fs.errors.ResourceNotFound:
            raise UFOLibError(f"No image file named '{fileName}' on {self.fs}")
        if validate:
            valid, error = pngValidator(data=data)
            if not valid:
                raise UFOLibError(error)
        return data

    def close(self) -> None:
        if self._shouldClose:
            self.fs.close()

    def __enter__(self) -> UFOReader:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> None:
        self.close()


# ----------
# UFO Writer
# ----------


class UFOWriter(UFOReader):
    """Write the various components of a .ufo.

    Attributes:
        path: An :class:`os.PathLike` object pointing to the .ufo.
        formatVersion: the UFO format version as a tuple of integers (major, minor),
            or as a single integer for the major digit only (minor is implied to be 0).
            By default, the latest formatVersion will be used; currently it is 3.0,
            which is equivalent to formatVersion=(3, 0).
        fileCreator: The creator of the .ufo file. Defaults to
            `com.github.fonttools.ufoLib`.
        structure: The internal structure of the .ufo file: either `ZIP` or `PACKAGE`.
        validate: A boolean indicating if the data read should be validated. Defaults
            to `True`.

    By default, the written data will be validated before writing. Set ``validate`` to
    ``False`` if you do not want to validate the data. Validation can also be overriden
    on a per-method level if desired.

    Raises:
        UnsupportedUFOFormat: An exception indicating that the requested UFO
            formatVersion is not supported.
    """

    def __init__(
        self,
        path: PathOrFS,
        formatVersion: UFOFormatVersionInput = None,
        fileCreator: str = "com.github.fonttools.ufoLib",
        structure: Optional[UFOFileStructure] = None,
        validate: bool = True,
    ) -> None:
        try:
            formatVersion = normalizeFormatVersion(formatVersion, UFOFormatVersion)
        except ValueError as e:
            from fontTools.ufoLib.errors import UnsupportedUFOFormat

            raise UnsupportedUFOFormat(
                f"Unsupported UFO format: {formatVersion!r}"
            ) from e

        if hasattr(path, "__fspath__"):  # support os.PathLike objects
            path = path.__fspath__()

        if isinstance(path, str):
            # normalize path by removing trailing or double slashes
            path = os.path.normpath(path)
            havePreviousFile = os.path.exists(path)
            if havePreviousFile:
                # ensure we use the same structure as the destination
                existingStructure = _sniffFileStructure(path)
                if structure is not None:
                    try:
                        structure = UFOFileStructure(structure)
                    except ValueError:
                        raise UFOLibError(
                            "Invalid or unsupported structure: '%s'" % structure
                        )
                    if structure is not existingStructure:
                        raise UFOLibError(
                            "A UFO with a different structure (%s) already exists "
                            "at the given path: '%s'" % (existingStructure, path)
                        )
                else:
                    structure = existingStructure
            else:
                # if not exists, default to 'package' structure
                if structure is None:
                    structure = UFOFileStructure.PACKAGE
                dirName = os.path.dirname(path)
                if dirName and not os.path.isdir(dirName):
                    raise UFOLibError(
                        "Cannot write to '%s': directory does not exist" % path
                    )
            if structure is UFOFileStructure.ZIP:
                if havePreviousFile:
                    # we can't write a zip in-place, so we have to copy its
                    # contents to a temporary location and work from there, then
                    # upon closing UFOWriter we create the final zip file
                    parentFS: FS = fs.tempfs.TempFS()
                    with fs.zipfs.ZipFS(path, encoding="utf-8") as origFS:  # type: ignore[abstract]
                        fs.copy.copy_fs(origFS, parentFS)
                    # if output path is an existing zip, we require that it contains
                    # one, and only one, root directory (with arbitrary name), in turn
                    # containing all the existing UFO contents
                    rootDirs = [
                        p.name
                        for p in parentFS.scandir("/")
                        # exclude macOS metadata contained in zip file
                        if p.is_dir and p.name != "__MACOSX"
                    ]
                    if len(rootDirs) != 1:
                        raise UFOLibError(
                            "Expected exactly 1 root directory, found %d"
                            % len(rootDirs)
                        )
                    else:
                        rootDir = rootDirs[0]
                else:
                    # if the output zip file didn't exist, we create the root folder;
                    # we name it the same as input 'path', but with '.ufo' extension
                    rootDir = os.path.splitext(os.path.basename(path))[0] + ".ufo"
                    parentFS = fs.zipfs.ZipFS(path, write=True, encoding="utf-8")  # type: ignore[abstract]
                    parentFS.makedir(rootDir)
                # 'ClosingSubFS' ensures that the parent filesystem is closed
                # when its root subdirectory is closed
                self.fs = parentFS.opendir(rootDir, factory=fs.subfs.ClosingSubFS)
            else:
                self.fs = fs.osfs.OSFS(path, create=True)
            self._fileStructure = structure
            self._havePreviousFile = havePreviousFile
            self._shouldClose = True
        elif isinstance(path, fs.base.FS):
            filesystem: FS = path
            try:
                filesystem.check()
            except fs.errors.FilesystemClosed:
                raise UFOLibError("the filesystem '%s' is closed" % path)
            else:
                self.fs = filesystem
            try:
                path = filesystem.getsyspath("/")
            except fs.errors.NoSysPath:
                # network or in-memory FS may not map to the local one
                path = str(filesystem)
            # if passed an FS object, always use 'package' structure
            if structure and structure is not UFOFileStructure.PACKAGE:
                import warnings

                warnings.warn(
                    "The 'structure' argument is not used when input is an FS object",
                    UserWarning,
                    stacklevel=2,
                )
            self._fileStructure = UFOFileStructure.PACKAGE
            # if FS contains a "metainfo.plist", we consider it non-empty
            self._havePreviousFile = filesystem.exists(METAINFO_FILENAME)
            # the user is responsible for closing the FS object
            self._shouldClose = False
        else:
            raise TypeError(
                "Expected a path string or fs object, found %s" % type(path).__name__
            )

        # establish some basic stuff
        self._path = fsdecode(path)
        self._formatVersion = formatVersion
        self._fileCreator = fileCreator
        self._downConversionKerningData: Optional[KerningGroupRenameMaps] = None
        self._validate = validate
        # if the file already exists, get the format version.
        # this will be needed for up and down conversion.
        previousFormatVersion = None
        if self._havePreviousFile:
            metaInfo = self._readMetaInfo(validate=validate)
            previousFormatVersion = metaInfo["formatVersionTuple"]
            # catch down conversion
            if previousFormatVersion > formatVersion:
                from fontTools.ufoLib.errors import UnsupportedUFOFormat

                raise UnsupportedUFOFormat(
                    "The UFO located at this path is a higher version "
                    f"({previousFormatVersion}) than the version ({formatVersion}) "
                    "that is trying to be written. This is not supported."
                )
        # handle the layer contents
        self.layerContents: Union[dict[str, str], OrderedDict[str, str]] = {}
        if previousFormatVersion is not None and previousFormatVersion.major >= 3:
            # already exists
            self.layerContents = OrderedDict(self._readLayerContents(validate))
        else:
            # previous < 3
            # imply the layer contents
            if self.fs.exists(DEFAULT_GLYPHS_DIRNAME):
                self.layerContents = {DEFAULT_LAYER_NAME: DEFAULT_GLYPHS_DIRNAME}
        # write the new metainfo
        self._writeMetaInfo()

    # properties

    def _get_fileCreator(self) -> str:
        return self._fileCreator

    fileCreator: property = property(
        _get_fileCreator,
        doc="The file creator of the UFO. This is set into metainfo.plist during __init__.",
    )

    # support methods for file system interaction

    def copyFromReader(
        self, reader: UFOReader, sourcePath: PathStr, destPath: PathStr
    ) -> None:
        """
        Copy the sourcePath in the provided UFOReader to destPath
        in this writer. The paths must be relative. This works with
        both individual files and directories.
        """
        if not isinstance(reader, UFOReader):
            raise UFOLibError("The reader must be an instance of UFOReader.")
        sourcePath = fsdecode(sourcePath)
        destPath = fsdecode(destPath)
        if not reader.fs.exists(sourcePath):
            raise UFOLibError(
                'The reader does not have data located at "%s".' % sourcePath
            )
        if self.fs.exists(destPath):
            raise UFOLibError('A file named "%s" already exists.' % destPath)
        # create the destination directory if it doesn't exist
        self.fs.makedirs(fs.path.dirname(destPath), recreate=True)
        if reader.fs.isdir(sourcePath):
            fs.copy.copy_dir(reader.fs, sourcePath, self.fs, destPath)
        else:
            fs.copy.copy_file(reader.fs, sourcePath, self.fs, destPath)

    def writeBytesToPath(self, path: PathStr, data: bytes) -> None:
        """
        Write bytes to a path relative to the UFO filesystem's root.
        If writing to an existing UFO, check to see if data matches the data
        that is already in the file at path; if so, the file is not rewritten
        so that the modification date is preserved.
        If needed, the directory tree for the given path will be built.
        """
        path = fsdecode(path)
        if self._havePreviousFile:
            if self.fs.isfile(path) and data == self.fs.readbytes(path):
                return
        try:
            self.fs.writebytes(path, data)
        except fs.errors.FileExpected:
            raise UFOLibError("A directory exists at '%s'" % path)
        except fs.errors.ResourceNotFound:
            self.fs.makedirs(fs.path.dirname(path), recreate=True)
            self.fs.writebytes(path, data)

    def getFileObjectForPath(
        self,
        path: PathStr,
        mode: str = "w",
        encoding: Optional[str] = None,
    ) -> Optional[IO[Any]]:
        """
        Returns a file (or file-like) object for the
        file at the given path. The path must be relative
        to the UFO path. Returns None if the file does
        not exist and the mode is "r" or "rb.
        An encoding may be passed if the file is opened in text mode.

        Note: The caller is responsible for closing the open file.
        """
        path = fsdecode(path)
        try:
            return self.fs.open(path, mode=mode, encoding=encoding)
        except fs.errors.ResourceNotFound as e:
            m = mode[0]
            if m == "r":
                # XXX I think we should just let it raise. The docstring,
                # however, says that this returns None if mode is 'r'
                return None
            elif m == "w" or m == "a" or m == "x":
                self.fs.makedirs(fs.path.dirname(path), recreate=True)
                return self.fs.open(path, mode=mode, encoding=encoding)
        except fs.errors.ResourceError as e:
            raise UFOLibError(f"unable to open '{path}' on {self.fs}: {e}")
        return None

    def removePath(
        self, path: PathStr, force: bool = False, removeEmptyParents: bool = True
    ) -> None:
        """
        Remove the file (or directory) at path. The path
        must be relative to the UFO.
        Raises UFOLibError if the path doesn't exist.
        If force=True, ignore non-existent paths.
        If the directory where 'path' is located becomes empty, it will
        be automatically removed, unless 'removeEmptyParents' is False.
        """
        path = fsdecode(path)
        try:
            self.fs.remove(path)
        except fs.errors.FileExpected:
            self.fs.removetree(path)
        except fs.errors.ResourceNotFound:
            if not force:
                raise UFOLibError(f"'{path}' does not exist on {self.fs}")
        if removeEmptyParents:
            parent = fs.path.dirname(path)
            if parent:
                fs.tools.remove_empty(self.fs, parent)

    # alias kept for backward compatibility with old API
    removeFileForPath = removePath

    # UFO mod time

    def setModificationTime(self) -> None:
        """
        Set the UFO modification time to the current time.
        This is never called automatically. It is up to the
        caller to call this when finished working on the UFO.
        """
        path = self._path
        if path is not None and os.path.exists(path):
            try:
                # this may fail on some filesystems (e.g. SMB servers)
                os.utime(path, None)
            except OSError as e:
                logger.warning("Failed to set modified time: %s", e)

    # metainfo.plist

    def _writeMetaInfo(self) -> None:
        metaInfo = dict(
            creator=self._fileCreator,
            formatVersion=self._formatVersion.major,
        )
        if self._formatVersion.minor != 0:
            metaInfo["formatVersionMinor"] = self._formatVersion.minor
        self._writePlist(METAINFO_FILENAME, metaInfo)

    # groups.plist

    def setKerningGroupConversionRenameMaps(self, maps: KerningGroupRenameMaps) -> None:
        """
        Set maps defining the renaming that should be done
        when writing groups and kerning in UFO 1 and UFO 2.
        This will effectively undo the conversion done when
        UFOReader reads this data. The dictionary should have
        this form::

                {
                        "side1" : {"group name to use when writing" : "group name in data"},
                        "side2" : {"group name to use when writing" : "group name in data"}
                }

        This is the same form returned by UFOReader's
        getKerningGroupConversionRenameMaps method.
        """
        if self._formatVersion >= UFOFormatVersion.FORMAT_3_0:
            return  # XXX raise an error here
        # flip the dictionaries
        remap = {}
        for side in ("side1", "side2"):
            for writeName, dataName in list(maps[side].items()):
                remap[dataName] = writeName
        self._downConversionKerningData = dict(groupRenameMap=remap)

    def writeGroups(
        self, groups: KerningGroups, validate: Optional[bool] = None
    ) -> None:
        """
        Write groups.plist. This method requires a
        dict of glyph groups as an argument.

        ``validate`` will validate the data, by default it is set to the
        class's validate value, can be overridden.
        """
        if validate is None:
            validate = self._validate
        # validate the data structure
        if validate:
            valid, message = groupsValidator(groups)
            if not valid:
                raise UFOLibError(message)
        # down convert
        if (
            self._formatVersion < UFOFormatVersion.FORMAT_3_0
            and self._downConversionKerningData is not None
        ):
            remap = self._downConversionKerningData["groupRenameMap"]
            remappedGroups = {}
            # there are some edge cases here that are ignored:
            # 1. if a group is being renamed to a name that
            #    already exists, the existing group is always
            #    overwritten. (this is why there are two loops
            #    below.) there doesn't seem to be a logical
            #    solution to groups mismatching and overwriting
            #    with the specifiecd group seems like a better
            #    solution than throwing an error.
            # 2. if side 1 and side 2 groups are being renamed
            #    to the same group name there is no check to
            #    ensure that the contents are identical. that
            #    is left up to the caller.
            for name, contents in list(groups.items()):
                if name in remap:
                    continue
                remappedGroups[name] = contents
            for name, contents in list(groups.items()):
                if name not in remap:
                    continue
                name = remap[name]
                remappedGroups[name] = contents
            groups = remappedGroups
        # pack and write
        groupsNew = {}
        for key, value in groups.items():
            groupsNew[key] = list(value)
        if groupsNew:
            self._writePlist(GROUPS_FILENAME, groupsNew)
        elif self._havePreviousFile:
            self.removePath(GROUPS_FILENAME, force=True, removeEmptyParents=False)

    # fontinfo.plist

    def writeInfo(self, info: Any, validate: Optional[bool] = None) -> None:
        """
        Write info.plist. This method requires an object
        that supports getting attributes that follow the
        fontinfo.plist version 2 specification. Attributes
        will be taken from the given object and written
        into the file.

        ``validate`` will validate the data, by default it is set to the
        class's validate value, can be overridden.
        """
        if validate is None:
            validate = self._validate
        # gather version 3 data
        infoData = {}
        for attr in list(fontInfoAttributesVersion3ValueData.keys()):
            if hasattr(info, attr):
                try:
                    value = getattr(info, attr)
                except AttributeError:
                    raise UFOLibError(
                        "The supplied info object does not support getting a necessary attribute (%s)."
                        % attr
                    )
                if value is None:
                    continue
                infoData[attr] = value
        # down convert data if necessary and validate
        if self._formatVersion == UFOFormatVersion.FORMAT_3_0:
            if validate:
                infoData = validateInfoVersion3Data(infoData)
        elif self._formatVersion == UFOFormatVersion.FORMAT_2_0:
            infoData = _convertFontInfoDataVersion3ToVersion2(infoData)
            if validate:
                infoData = validateInfoVersion2Data(infoData)
        elif self._formatVersion == UFOFormatVersion.FORMAT_1_0:
            infoData = _convertFontInfoDataVersion3ToVersion2(infoData)
            if validate:
                infoData = validateInfoVersion2Data(infoData)
            infoData = _convertFontInfoDataVersion2ToVersion1(infoData)
        # write file if there is anything to write
        if infoData:
            self._writePlist(FONTINFO_FILENAME, infoData)

    # kerning.plist

    def writeKerning(
        self, kerning: KerningDict, validate: Optional[bool] = None
    ) -> None:
        """
        Write kerning.plist. This method requires a
        dict of kerning pairs as an argument.

        This performs basic structural validation of the kerning,
        but it does not check for compliance with the spec in
        regards to conflicting pairs. The assumption is that the
        kerning data being passed is standards compliant.

        ``validate`` will validate the data, by default it is set to the
        class's validate value, can be overridden.
        """
        if validate is None:
            validate = self._validate
        # validate the data structure
        if validate:
            invalidFormatMessage = "The kerning is not properly formatted."
            if not isDictEnough(kerning):
                raise UFOLibError(invalidFormatMessage)
            for pair, value in list(kerning.items()):
                if not isinstance(pair, (list, tuple)):
                    raise UFOLibError(invalidFormatMessage)
                if not len(pair) == 2:
                    raise UFOLibError(invalidFormatMessage)
                if not isinstance(pair[0], str):
                    raise UFOLibError(invalidFormatMessage)
                if not isinstance(pair[1], str):
                    raise UFOLibError(invalidFormatMessage)
                if not isinstance(value, numberTypes):
                    raise UFOLibError(invalidFormatMessage)
        # down convert
        if (
            self._formatVersion < UFOFormatVersion.FORMAT_3_0
            and self._downConversionKerningData is not None
        ):
            remap = self._downConversionKerningData["groupRenameMap"]
            remappedKerning = {}
            for (side1, side2), value in list(kerning.items()):
                side1 = remap.get(side1, side1)
                side2 = remap.get(side2, side2)
                remappedKerning[side1, side2] = value
            kerning = remappedKerning
        # pack and write
        kerningDict: KerningNested = {}
        for left, right in kerning.keys():
            value = kerning[left, right]
            if left not in kerningDict:
                kerningDict[left] = {}
            kerningDict[left][right] = value
        if kerningDict:
            self._writePlist(KERNING_FILENAME, kerningDict)
        elif self._havePreviousFile:
            self.removePath(KERNING_FILENAME, force=True, removeEmptyParents=False)

    # lib.plist

    def writeLib(self, libDict: LibDict, validate: Optional[bool] = None) -> None:
        """
        Write lib.plist. This method requires a
        lib dict as an argument.

        ``validate`` will validate the data, by default it is set to the
        class's validate value, can be overridden.
        """
        if validate is None:
            validate = self._validate
        if validate:
            valid, message = fontLibValidator(libDict)
            if not valid:
                raise UFOLibError(message)
        if libDict:
            self._writePlist(LIB_FILENAME, libDict)
        elif self._havePreviousFile:
            self.removePath(LIB_FILENAME, force=True, removeEmptyParents=False)

    # features.fea

    def writeFeatures(self, features: str, validate: Optional[bool] = None) -> None:
        """
        Write features.fea. This method requires a
        features string as an argument.
        """
        if validate is None:
            validate = self._validate
        if self._formatVersion == UFOFormatVersion.FORMAT_1_0:
            raise UFOLibError("features.fea is not allowed in UFO Format Version 1.")
        if validate:
            if not isinstance(features, str):
                raise UFOLibError("The features are not text.")
        if features:
            self.writeBytesToPath(FEATURES_FILENAME, features.encode("utf8"))
        elif self._havePreviousFile:
            self.removePath(FEATURES_FILENAME, force=True, removeEmptyParents=False)

    # glyph sets & layers

    def writeLayerContents(
        self, layerOrder: LayerOrderList = None, validate: Optional[bool] = None
    ) -> None:
        """
        Write the layercontents.plist file. This method  *must* be called
        after all glyph sets have been written.
        """
        if validate is None:
            validate = self._validate
        if self._formatVersion < UFOFormatVersion.FORMAT_3_0:
            return
        if layerOrder is not None:
            newOrder: list[Optional[str]] = []
            for layerName in layerOrder:
                if layerName is None:
                    layerName = DEFAULT_LAYER_NAME
                newOrder.append(layerName)
            layerOrder = newOrder
        else:
            layerOrder = list(self.layerContents.keys())
        if validate and set(layerOrder) != set(self.layerContents.keys()):
            raise UFOLibError(
                "The layer order content does not match the glyph sets that have been created."
            )
        layerContents = [
            (layerName, self.layerContents[layerName])
            for layerName in layerOrder
            if layerName is not None
        ]
        self._writePlist(LAYERCONTENTS_FILENAME, layerContents)

    def _findDirectoryForLayerName(self, layerName: Optional[str]) -> str:
        foundDirectory = None
        for existingLayerName, directoryName in list(self.layerContents.items()):
            if layerName is None and directoryName == DEFAULT_GLYPHS_DIRNAME:
                foundDirectory = directoryName
                break
            elif existingLayerName == layerName:
                foundDirectory = directoryName
                break
        if not foundDirectory:
            raise UFOLibError(
                "Could not locate a glyph set directory for the layer named %s."
                % layerName
            )
        return foundDirectory

    def getGlyphSet(  # type: ignore[override]
        self,
        layerName: Optional[str] = None,
        defaultLayer: bool = True,
        glyphNameToFileNameFunc: GlyphNameToFileNameFunc = None,
        validateRead: Optional[bool] = None,
        validateWrite: Optional[bool] = None,
        expectContentsFile: bool = False,
    ) -> GlyphSet:
        """
        Return the GlyphSet object associated with the
        appropriate glyph directory in the .ufo.
        If layerName is None, the default glyph set
        will be used. The defaultLayer flag indicates
        that the layer should be saved into the default
        glyphs directory.

        ``validateRead`` will validate the read data, by default it is set to the
        class's validate value, can be overridden.
        ``validateWrte`` will validate the written data, by default it is set to the
        class's validate value, can be overridden.
        ``expectContentsFile`` will raise a GlifLibError if a contents.plist file is
        not found on the glyph set file system. This should be set to ``True`` if you
        are reading an existing UFO and ``False`` if you use ``getGlyphSet`` to create
        a fresh	glyph set.
        """
        if validateRead is None:
            validateRead = self._validate
        if validateWrite is None:
            validateWrite = self._validate
        # only default can be written in < 3
        if self._formatVersion < UFOFormatVersion.FORMAT_3_0 and (
            not defaultLayer or layerName is not None
        ):
            raise UFOLibError(
                f"Only the default layer can be writen in UFO {self._formatVersion.major}."
            )
        # locate a layer name when None has been given
        if layerName is None and defaultLayer:
            for existingLayerName, directory in self.layerContents.items():
                if directory == DEFAULT_GLYPHS_DIRNAME:
                    layerName = existingLayerName
            if layerName is None:
                layerName = DEFAULT_LAYER_NAME
        elif layerName is None and not defaultLayer:
            raise UFOLibError("A layer name must be provided for non-default layers.")
        # move along to format specific writing
        if self._formatVersion < UFOFormatVersion.FORMAT_3_0:
            return self._getDefaultGlyphSet(
                validateRead,
                validateWrite,
                glyphNameToFileNameFunc=glyphNameToFileNameFunc,
                expectContentsFile=expectContentsFile,
            )
        elif self._formatVersion.major == UFOFormatVersion.FORMAT_3_0.major:
            return self._getGlyphSetFormatVersion3(
                validateRead,
                validateWrite,
                layerName=layerName,
                defaultLayer=defaultLayer,
                glyphNameToFileNameFunc=glyphNameToFileNameFunc,
                expectContentsFile=expectContentsFile,
            )
        else:
            raise NotImplementedError(self._formatVersion)

    def _getDefaultGlyphSet(
        self,
        validateRead: bool,
        validateWrite: bool,
        glyphNameToFileNameFunc: GlyphNameToFileNameFunc = None,
        expectContentsFile: bool = False,
    ) -> GlyphSet:
        from fontTools.ufoLib.glifLib import GlyphSet

        glyphSubFS = self.fs.makedir(DEFAULT_GLYPHS_DIRNAME, recreate=True)
        return GlyphSet(
            glyphSubFS,
            glyphNameToFileNameFunc=glyphNameToFileNameFunc,
            ufoFormatVersion=self._formatVersion,
            validateRead=validateRead,
            validateWrite=validateWrite,
            expectContentsFile=expectContentsFile,
        )

    def _getGlyphSetFormatVersion3(
        self,
        validateRead: bool,
        validateWrite: bool,
        layerName: Optional[str] = None,
        defaultLayer: bool = True,
        glyphNameToFileNameFunc: GlyphNameToFileNameFunc = None,
        expectContentsFile: bool = False,
    ) -> GlyphSet:
        from fontTools.ufoLib.glifLib import GlyphSet

        # if the default flag is on, make sure that the default in the file
        # matches the default being written. also make sure that this layer
        # name is not already linked to a non-default layer.
        if defaultLayer:
            for existingLayerName, directory in self.layerContents.items():
                if directory == DEFAULT_GLYPHS_DIRNAME:
                    if existingLayerName != layerName:
                        raise UFOLibError(
                            "Another layer ('%s') is already mapped to the default directory."
                            % existingLayerName
                        )
                elif existingLayerName == layerName:
                    raise UFOLibError(
                        "The layer name is already mapped to a non-default layer."
                    )

        # handle layerName is None to avoid MyPy errors
        if layerName is None:
            raise TypeError("'leyerName' cannot be None.")

        # get an existing directory name
        if layerName in self.layerContents:
            directory = self.layerContents[layerName]
        # get a  new directory name
        else:
            if defaultLayer:
                directory = DEFAULT_GLYPHS_DIRNAME
            else:
                # not caching this could be slightly expensive,
                # but caching it will be cumbersome
                existing = {d.lower() for d in self.layerContents.values()}
                directory = userNameToFileName(
                    layerName, existing=existing, prefix="glyphs."
                )
        # make the directory
        glyphSubFS = self.fs.makedir(directory, recreate=True)
        # store the mapping
        self.layerContents[layerName] = directory
        # load the glyph set
        return GlyphSet(
            glyphSubFS,
            glyphNameToFileNameFunc=glyphNameToFileNameFunc,
            ufoFormatVersion=self._formatVersion,
            validateRead=validateRead,
            validateWrite=validateWrite,
            expectContentsFile=expectContentsFile,
        )

    def renameGlyphSet(
        self,
        layerName: Optional[str],
        newLayerName: Optional[str],
        defaultLayer: bool = False,
    ) -> None:
        """
        Rename a glyph set.

        Note: if a GlyphSet object has already been retrieved for
        layerName, it is up to the caller to inform that object that
        the directory it represents has changed.
        """
        if self._formatVersion < UFOFormatVersion.FORMAT_3_0:
            # ignore renaming glyph sets for UFO1 UFO2
            # just write the data from the default layer
            return
        # the new and old names can be the same
        # as long as the default is being switched
        if layerName is not None and layerName == newLayerName:
            # if the default is off and the layer is already not the default, skip
            if (
                self.layerContents[layerName] != DEFAULT_GLYPHS_DIRNAME
                and not defaultLayer
            ):
                return
            # if the default is on and the layer is already the default, skip
            if self.layerContents[layerName] == DEFAULT_GLYPHS_DIRNAME and defaultLayer:
                return
        else:
            # make sure the new layer name doesn't already exist
            if newLayerName is None:
                newLayerName = DEFAULT_LAYER_NAME
            if newLayerName in self.layerContents:
                raise UFOLibError("A layer named %s already exists." % newLayerName)
            # make sure the default layer doesn't already exist
            if defaultLayer and DEFAULT_GLYPHS_DIRNAME in self.layerContents.values():
                raise UFOLibError("A default layer already exists.")
        # get the paths
        oldDirectory = self._findDirectoryForLayerName(layerName)
        if defaultLayer:
            newDirectory = DEFAULT_GLYPHS_DIRNAME
        else:
            existing = {name.lower() for name in self.layerContents.values()}
            newDirectory = userNameToFileName(
                newLayerName, existing=existing, prefix="glyphs."
            )
        # update the internal mapping
        if layerName is not None:
            del self.layerContents[layerName]
        self.layerContents[newLayerName] = newDirectory
        # do the file system copy
        self.fs.movedir(oldDirectory, newDirectory, create=True)

    def deleteGlyphSet(self, layerName: Optional[str]) -> None:
        """
        Remove the glyph set matching layerName.
        """
        if self._formatVersion < UFOFormatVersion.FORMAT_3_0:
            # ignore deleting glyph sets for UFO1 UFO2 as there are no layers
            # just write the data from the default layer
            return
        foundDirectory = self._findDirectoryForLayerName(layerName)
        self.removePath(foundDirectory, removeEmptyParents=False)
        if layerName is not None:
            del self.layerContents[layerName]

    def writeData(self, fileName: PathStr, data: bytes) -> None:
        """
        Write data to fileName in the 'data' directory.
        The data must be a bytes string.
        """
        self.writeBytesToPath(f"{DATA_DIRNAME}/{fsdecode(fileName)}", data)

    def removeData(self, fileName: PathStr) -> None:
        """
        Remove the file named fileName from the data directory.
        """
        self.removePath(f"{DATA_DIRNAME}/{fsdecode(fileName)}")

    # /images

    def writeImage(
        self,
        fileName: PathStr,
        data: bytes,
        validate: Optional[bool] = None,
    ) -> None:
        """
        Write data to fileName in the images directory.
        The data must be a valid PNG.
        """
        if validate is None:
            validate = self._validate
        if self._formatVersion < UFOFormatVersion.FORMAT_3_0:
            raise UFOLibError(
                f"Images are not allowed in UFO {self._formatVersion.major}."
            )
        fileName = fsdecode(fileName)
        if validate:
            valid, error = pngValidator(data=data)
            if not valid:
                raise UFOLibError(error)
        self.writeBytesToPath(f"{IMAGES_DIRNAME}/{fileName}", data)

    def removeImage(
        self,
        fileName: PathStr,
        validate: Optional[bool] = None,
    ) -> None:  # XXX remove unused 'validate'?
        """
        Remove the file named fileName from the
        images directory.
        """
        if self._formatVersion < UFOFormatVersion.FORMAT_3_0:
            raise UFOLibError(
                f"Images are not allowed in UFO {self._formatVersion.major}."
            )
        self.removePath(f"{IMAGES_DIRNAME}/{fsdecode(fileName)}")

    def copyImageFromReader(
        self,
        reader: UFOReader,
        sourceFileName: PathStr,
        destFileName: PathStr,
        validate: Optional[bool] = None,
    ) -> None:
        """
        Copy the sourceFileName in the provided UFOReader to destFileName
        in this writer. This uses the most memory efficient method possible
        for copying the data possible.
        """
        if validate is None:
            validate = self._validate
        if self._formatVersion < UFOFormatVersion.FORMAT_3_0:
            raise UFOLibError(
                f"Images are not allowed in UFO {self._formatVersion.major}."
            )
        sourcePath = f"{IMAGES_DIRNAME}/{fsdecode(sourceFileName)}"
        destPath = f"{IMAGES_DIRNAME}/{fsdecode(destFileName)}"
        self.copyFromReader(reader, sourcePath, destPath)

    def close(self) -> None:
        if self._havePreviousFile and self._fileStructure is UFOFileStructure.ZIP:
            # if we are updating an existing zip file, we can now compress the
            # contents of the temporary filesystem in the destination path
            rootDir = os.path.splitext(os.path.basename(self._path))[0] + ".ufo"
            with fs.zipfs.ZipFS(self._path, write=True, encoding="utf-8") as destFS:  # type: ignore[abstract]
                fs.copy.copy_fs(self.fs, destFS.makedir(rootDir))
        super().close()


# just an alias, makes it more explicit
UFOReaderWriter = UFOWriter


# ----------------
# Helper Functions
# ----------------


def _sniffFileStructure(ufo_path: PathStr) -> UFOFileStructure:
    """Return UFOFileStructure.ZIP if the UFO at path 'ufo_path' (str)
    is a zip file, else return UFOFileStructure.PACKAGE if 'ufo_path' is a
    directory.
    Raise UFOLibError if it is a file with unknown structure, or if the path
    does not exist.
    """
    if zipfile.is_zipfile(ufo_path):
        return UFOFileStructure.ZIP
    elif os.path.isdir(ufo_path):
        return UFOFileStructure.PACKAGE
    elif os.path.isfile(ufo_path):
        raise UFOLibError(
            "The specified UFO does not have a known structure: '%s'" % ufo_path
        )
    else:
        raise UFOLibError("No such file or directory: '%s'" % ufo_path)


def makeUFOPath(path: PathStr) -> str:
    """
    Return a .ufo pathname.

    >>> makeUFOPath("directory/something.ext") == (
    ... 	os.path.join('directory', 'something.ufo'))
    True
    >>> makeUFOPath("directory/something.another.thing.ext") == (
    ... 	os.path.join('directory', 'something.another.thing.ufo'))
    True
    """
    dir, name = os.path.split(path)
    name = ".".join([".".join(name.split(".")[:-1]), "ufo"])
    return os.path.join(dir, name)


# ----------------------
# fontinfo.plist Support
# ----------------------

# Version Validators

# There is no version 1 validator and there shouldn't be.
# The version 1 spec was very loose and there were numerous
# cases of invalid values.


def validateFontInfoVersion2ValueForAttribute(attr: str, value: Any) -> bool:
    """
    This performs very basic validation of the value for attribute
    following the UFO 2 fontinfo.plist specification. The results
    of this should not be interpretted as *correct* for the font
    that they are part of. This merely indicates that the value
    is of the proper type and, where the specification defines
    a set range of possible values for an attribute, that the
    value is in the accepted range.
    """
    dataValidationDict = fontInfoAttributesVersion2ValueData[attr]
    valueType = dataValidationDict.get("type")
    validator = dataValidationDict.get("valueValidator", genericTypeValidator)
    valueOptions = dataValidationDict.get("valueOptions")
    # have specific options for the validator
    if valueOptions is not None:
        isValidValue = validator(value, valueOptions)
    # no specific options
    else:
        if validator == genericTypeValidator:
            isValidValue = validator(value, valueType)
        else:
            isValidValue = validator(value)
    return isValidValue


def validateInfoVersion2Data(infoData: dict[str, Any]) -> dict[str, Any]:
    """
    This performs very basic validation of the value for infoData
    following the UFO 2 fontinfo.plist specification. The results
    of this should not be interpretted as *correct* for the font
    that they are part of. This merely indicates that the values
    are of the proper type and, where the specification defines
    a set range of possible values for an attribute, that the
    value is in the accepted range.
    """
    validInfoData = {}
    for attr, value in list(infoData.items()):
        isValidValue = validateFontInfoVersion2ValueForAttribute(attr, value)
        if not isValidValue:
            raise UFOLibError(f"Invalid value for attribute {attr} ({value!r}).")
        else:
            validInfoData[attr] = value
    return validInfoData


def validateFontInfoVersion3ValueForAttribute(attr: str, value: Any) -> bool:
    """
    This performs very basic validation of the value for attribute
    following the UFO 3 fontinfo.plist specification. The results
    of this should not be interpretted as *correct* for the font
    that they are part of. This merely indicates that the value
    is of the proper type and, where the specification defines
    a set range of possible values for an attribute, that the
    value is in the accepted range.
    """
    dataValidationDict = fontInfoAttributesVersion3ValueData[attr]
    valueType = dataValidationDict.get("type")
    validator = dataValidationDict.get("valueValidator", genericTypeValidator)
    valueOptions = dataValidationDict.get("valueOptions")
    # have specific options for the validator
    if valueOptions is not None:
        isValidValue = validator(value, valueOptions)
    # no specific options
    else:
        if validator == genericTypeValidator:
            isValidValue = validator(value, valueType)
        else:
            isValidValue = validator(value)
    return isValidValue


def validateInfoVersion3Data(infoData: dict[str, Any]) -> dict[str, Any]:
    """
    This performs very basic validation of the value for infoData
    following the UFO 3 fontinfo.plist specification. The results
    of this should not be interpretted as *correct* for the font
    that they are part of. This merely indicates that the values
    are of the proper type and, where the specification defines
    a set range of possible values for an attribute, that the
    value is in the accepted range.
    """
    validInfoData = {}
    for attr, value in list(infoData.items()):
        isValidValue = validateFontInfoVersion3ValueForAttribute(attr, value)
        if not isValidValue:
            raise UFOLibError(f"Invalid value for attribute {attr} ({value!r}).")
        else:
            validInfoData[attr] = value
    return validInfoData


# Value Options

fontInfoOpenTypeHeadFlagsOptions: list[int] = list(range(0, 15))
fontInfoOpenTypeOS2SelectionOptions: list[int] = [1, 2, 3, 4, 7, 8, 9]
fontInfoOpenTypeOS2UnicodeRangesOptions: list[int] = list(range(0, 128))
fontInfoOpenTypeOS2CodePageRangesOptions: list[int] = list(range(0, 64))
fontInfoOpenTypeOS2TypeOptions: list[int] = [0, 1, 2, 3, 8, 9]

# Version Attribute Definitions
# This defines the attributes, types and, in some
# cases the possible values, that can exist is
# fontinfo.plist.

fontInfoAttributesVersion1: set[str] = {
    "familyName",
    "styleName",
    "fullName",
    "fontName",
    "menuName",
    "fontStyle",
    "note",
    "versionMajor",
    "versionMinor",
    "year",
    "copyright",
    "notice",
    "trademark",
    "license",
    "licenseURL",
    "createdBy",
    "designer",
    "designerURL",
    "vendorURL",
    "unitsPerEm",
    "ascender",
    "descender",
    "capHeight",
    "xHeight",
    "defaultWidth",
    "slantAngle",
    "italicAngle",
    "widthName",
    "weightName",
    "weightValue",
    "fondName",
    "otFamilyName",
    "otStyleName",
    "otMacName",
    "msCharSet",
    "fondID",
    "uniqueID",
    "ttVendor",
    "ttUniqueID",
    "ttVersion",
}

fontInfoAttributesVersion2ValueData: FontInfoAttributes = {
    "familyName": dict(type=str),
    "styleName": dict(type=str),
    "styleMapFamilyName": dict(type=str),
    "styleMapStyleName": dict(
        type=str, valueValidator=fontInfoStyleMapStyleNameValidator
    ),
    "versionMajor": dict(type=int),
    "versionMinor": dict(type=int),
    "year": dict(type=int),
    "copyright": dict(type=str),
    "trademark": dict(type=str),
    "unitsPerEm": dict(type=(int, float)),
    "descender": dict(type=(int, float)),
    "xHeight": dict(type=(int, float)),
    "capHeight": dict(type=(int, float)),
    "ascender": dict(type=(int, float)),
    "italicAngle": dict(type=(float, int)),
    "note": dict(type=str),
    "openTypeHeadCreated": dict(
        type=str, valueValidator=fontInfoOpenTypeHeadCreatedValidator
    ),
    "openTypeHeadLowestRecPPEM": dict(type=(int, float)),
    "openTypeHeadFlags": dict(
        type="integerList",
        valueValidator=genericIntListValidator,
        valueOptions=fontInfoOpenTypeHeadFlagsOptions,
    ),
    "openTypeHheaAscender": dict(type=(int, float)),
    "openTypeHheaDescender": dict(type=(int, float)),
    "openTypeHheaLineGap": dict(type=(int, float)),
    "openTypeHheaCaretSlopeRise": dict(type=int),
    "openTypeHheaCaretSlopeRun": dict(type=int),
    "openTypeHheaCaretOffset": dict(type=(int, float)),
    "openTypeNameDesigner": dict(type=str),
    "openTypeNameDesignerURL": dict(type=str),
    "openTypeNameManufacturer": dict(type=str),
    "openTypeNameManufacturerURL": dict(type=str),
    "openTypeNameLicense": dict(type=str),
    "openTypeNameLicenseURL": dict(type=str),
    "openTypeNameVersion": dict(type=str),
    "openTypeNameUniqueID": dict(type=str),
    "openTypeNameDescription": dict(type=str),
    "openTypeNamePreferredFamilyName": dict(type=str),
    "openTypeNamePreferredSubfamilyName": dict(type=str),
    "openTypeNameCompatibleFullName": dict(type=str),
    "openTypeNameSampleText": dict(type=str),
    "openTypeNameWWSFamilyName": dict(type=str),
    "openTypeNameWWSSubfamilyName": dict(type=str),
    "openTypeOS2WidthClass": dict(
        type=int, valueValidator=fontInfoOpenTypeOS2WidthClassValidator
    ),
    "openTypeOS2WeightClass": dict(
        type=int, valueValidator=fontInfoOpenTypeOS2WeightClassValidator
    ),
    "openTypeOS2Selection": dict(
        type="integerList",
        valueValidator=genericIntListValidator,
        valueOptions=fontInfoOpenTypeOS2SelectionOptions,
    ),
    "openTypeOS2VendorID": dict(type=str),
    "openTypeOS2Panose": dict(
        type="integerList", valueValidator=fontInfoVersion2OpenTypeOS2PanoseValidator
    ),
    "openTypeOS2FamilyClass": dict(
        type="integerList", valueValidator=fontInfoOpenTypeOS2FamilyClassValidator
    ),
    "openTypeOS2UnicodeRanges": dict(
        type="integerList",
        valueValidator=genericIntListValidator,
        valueOptions=fontInfoOpenTypeOS2UnicodeRangesOptions,
    ),
    "openTypeOS2CodePageRanges": dict(
        type="integerList",
        valueValidator=genericIntListValidator,
        valueOptions=fontInfoOpenTypeOS2CodePageRangesOptions,
    ),
    "openTypeOS2TypoAscender": dict(type=(int, float)),
    "openTypeOS2TypoDescender": dict(type=(int, float)),
    "openTypeOS2TypoLineGap": dict(type=(int, float)),
    "openTypeOS2WinAscent": dict(type=(int, float)),
    "openTypeOS2WinDescent": dict(type=(int, float)),
    "openTypeOS2Type": dict(
        type="integerList",
        valueValidator=genericIntListValidator,
        valueOptions=fontInfoOpenTypeOS2TypeOptions,
    ),
    "openTypeOS2SubscriptXSize": dict(type=(int, float)),
    "openTypeOS2SubscriptYSize": dict(type=(int, float)),
    "openTypeOS2SubscriptXOffset": dict(type=(int, float)),
    "openTypeOS2SubscriptYOffset": dict(type=(int, float)),
    "openTypeOS2SuperscriptXSize": dict(type=(int, float)),
    "openTypeOS2SuperscriptYSize": dict(type=(int, float)),
    "openTypeOS2SuperscriptXOffset": dict(type=(int, float)),
    "openTypeOS2SuperscriptYOffset": dict(type=(int, float)),
    "openTypeOS2StrikeoutSize": dict(type=(int, float)),
    "openTypeOS2StrikeoutPosition": dict(type=(int, float)),
    "openTypeVheaVertTypoAscender": dict(type=(int, float)),
    "openTypeVheaVertTypoDescender": dict(type=(int, float)),
    "openTypeVheaVertTypoLineGap": dict(type=(int, float)),
    "openTypeVheaCaretSlopeRise": dict(type=int),
    "openTypeVheaCaretSlopeRun": dict(type=int),
    "openTypeVheaCaretOffset": dict(type=(int, float)),
    "postscriptFontName": dict(type=str),
    "postscriptFullName": dict(type=str),
    "postscriptSlantAngle": dict(type=(float, int)),
    "postscriptUniqueID": dict(type=int),
    "postscriptUnderlineThickness": dict(type=(int, float)),
    "postscriptUnderlinePosition": dict(type=(int, float)),
    "postscriptIsFixedPitch": dict(type=bool),
    "postscriptBlueValues": dict(
        type="integerList", valueValidator=fontInfoPostscriptBluesValidator
    ),
    "postscriptOtherBlues": dict(
        type="integerList", valueValidator=fontInfoPostscriptOtherBluesValidator
    ),
    "postscriptFamilyBlues": dict(
        type="integerList", valueValidator=fontInfoPostscriptBluesValidator
    ),
    "postscriptFamilyOtherBlues": dict(
        type="integerList", valueValidator=fontInfoPostscriptOtherBluesValidator
    ),
    "postscriptStemSnapH": dict(
        type="integerList", valueValidator=fontInfoPostscriptStemsValidator
    ),
    "postscriptStemSnapV": dict(
        type="integerList", valueValidator=fontInfoPostscriptStemsValidator
    ),
    "postscriptBlueFuzz": dict(type=(int, float)),
    "postscriptBlueShift": dict(type=(int, float)),
    "postscriptBlueScale": dict(type=(float, int)),
    "postscriptForceBold": dict(type=bool),
    "postscriptDefaultWidthX": dict(type=(int, float)),
    "postscriptNominalWidthX": dict(type=(int, float)),
    "postscriptWeightName": dict(type=str),
    "postscriptDefaultCharacter": dict(type=str),
    "postscriptWindowsCharacterSet": dict(
        type=int, valueValidator=fontInfoPostscriptWindowsCharacterSetValidator
    ),
    "macintoshFONDFamilyID": dict(type=int),
    "macintoshFONDName": dict(type=str),
}
fontInfoAttributesVersion2: set[str] = set(fontInfoAttributesVersion2ValueData.keys())

fontInfoAttributesVersion3ValueData: FontInfoAttributes = deepcopy(
    fontInfoAttributesVersion2ValueData
)
fontInfoAttributesVersion3ValueData.update(
    {
        "versionMinor": dict(type=int, valueValidator=genericNonNegativeIntValidator),
        "unitsPerEm": dict(
            type=(int, float), valueValidator=genericNonNegativeNumberValidator
        ),
        "openTypeHeadLowestRecPPEM": dict(
            type=int, valueValidator=genericNonNegativeNumberValidator
        ),
        "openTypeHheaAscender": dict(type=int),
        "openTypeHheaDescender": dict(type=int),
        "openTypeHheaLineGap": dict(type=int),
        "openTypeHheaCaretOffset": dict(type=int),
        "openTypeOS2Panose": dict(
            type="integerList",
            valueValidator=fontInfoVersion3OpenTypeOS2PanoseValidator,
        ),
        "openTypeOS2TypoAscender": dict(type=int),
        "openTypeOS2TypoDescender": dict(type=int),
        "openTypeOS2TypoLineGap": dict(type=int),
        "openTypeOS2WinAscent": dict(
            type=int, valueValidator=genericNonNegativeNumberValidator
        ),
        "openTypeOS2WinDescent": dict(
            type=int, valueValidator=genericNonNegativeNumberValidator
        ),
        "openTypeOS2SubscriptXSize": dict(type=int),
        "openTypeOS2SubscriptYSize": dict(type=int),
        "openTypeOS2SubscriptXOffset": dict(type=int),
        "openTypeOS2SubscriptYOffset": dict(type=int),
        "openTypeOS2SuperscriptXSize": dict(type=int),
        "openTypeOS2SuperscriptYSize": dict(type=int),
        "openTypeOS2SuperscriptXOffset": dict(type=int),
        "openTypeOS2SuperscriptYOffset": dict(type=int),
        "openTypeOS2StrikeoutSize": dict(type=int),
        "openTypeOS2StrikeoutPosition": dict(type=int),
        "openTypeGaspRangeRecords": dict(
            type="dictList", valueValidator=fontInfoOpenTypeGaspRangeRecordsValidator
        ),
        "openTypeNameRecords": dict(
            type="dictList", valueValidator=fontInfoOpenTypeNameRecordsValidator
        ),
        "openTypeVheaVertTypoAscender": dict(type=int),
        "openTypeVheaVertTypoDescender": dict(type=int),
        "openTypeVheaVertTypoLineGap": dict(type=int),
        "openTypeVheaCaretOffset": dict(type=int),
        "woffMajorVersion": dict(
            type=int, valueValidator=genericNonNegativeIntValidator
        ),
        "woffMinorVersion": dict(
            type=int, valueValidator=genericNonNegativeIntValidator
        ),
        "woffMetadataUniqueID": dict(
            type=dict, valueValidator=fontInfoWOFFMetadataUniqueIDValidator
        ),
        "woffMetadataVendor": dict(
            type=dict, valueValidator=fontInfoWOFFMetadataVendorValidator
        ),
        "woffMetadataCredits": dict(
            type=dict, valueValidator=fontInfoWOFFMetadataCreditsValidator
        ),
        "woffMetadataDescription": dict(
            type=dict, valueValidator=fontInfoWOFFMetadataDescriptionValidator
        ),
        "woffMetadataLicense": dict(
            type=dict, valueValidator=fontInfoWOFFMetadataLicenseValidator
        ),
        "woffMetadataCopyright": dict(
            type=dict, valueValidator=fontInfoWOFFMetadataCopyrightValidator
        ),
        "woffMetadataTrademark": dict(
            type=dict, valueValidator=fontInfoWOFFMetadataTrademarkValidator
        ),
        "woffMetadataLicensee": dict(
            type=dict, valueValidator=fontInfoWOFFMetadataLicenseeValidator
        ),
        "woffMetadataExtensions": dict(
            type=list, valueValidator=fontInfoWOFFMetadataExtensionsValidator
        ),
        "guidelines": dict(type=list, valueValidator=guidelinesValidator),
    }
)
fontInfoAttributesVersion3: set[str] = set(fontInfoAttributesVersion3ValueData.keys())

# insert the type validator for all attrs that
# have no defined validator.
for attr, dataDict in list(fontInfoAttributesVersion2ValueData.items()):
    if "valueValidator" not in dataDict:
        dataDict["valueValidator"] = genericTypeValidator

for attr, dataDict in list(fontInfoAttributesVersion3ValueData.items()):
    if "valueValidator" not in dataDict:
        dataDict["valueValidator"] = genericTypeValidator

# Version Conversion Support
# These are used from converting from version 1
# to version 2 or vice-versa.


def _flipDict(d: dict[K, V]) -> dict[V, K]:
    flipped = {}
    for key, value in list(d.items()):
        flipped[value] = key
    return flipped


fontInfoAttributesVersion1To2: dict[str, str] = {
    "menuName": "styleMapFamilyName",
    "designer": "openTypeNameDesigner",
    "designerURL": "openTypeNameDesignerURL",
    "createdBy": "openTypeNameManufacturer",
    "vendorURL": "openTypeNameManufacturerURL",
    "license": "openTypeNameLicense",
    "licenseURL": "openTypeNameLicenseURL",
    "ttVersion": "openTypeNameVersion",
    "ttUniqueID": "openTypeNameUniqueID",
    "notice": "openTypeNameDescription",
    "otFamilyName": "openTypeNamePreferredFamilyName",
    "otStyleName": "openTypeNamePreferredSubfamilyName",
    "otMacName": "openTypeNameCompatibleFullName",
    "weightName": "postscriptWeightName",
    "weightValue": "openTypeOS2WeightClass",
    "ttVendor": "openTypeOS2VendorID",
    "uniqueID": "postscriptUniqueID",
    "fontName": "postscriptFontName",
    "fondID": "macintoshFONDFamilyID",
    "fondName": "macintoshFONDName",
    "defaultWidth": "postscriptDefaultWidthX",
    "slantAngle": "postscriptSlantAngle",
    "fullName": "postscriptFullName",
    # require special value conversion
    "fontStyle": "styleMapStyleName",
    "widthName": "openTypeOS2WidthClass",
    "msCharSet": "postscriptWindowsCharacterSet",
}
fontInfoAttributesVersion2To1 = _flipDict(fontInfoAttributesVersion1To2)
deprecatedFontInfoAttributesVersion2 = set(fontInfoAttributesVersion1To2.keys())

_fontStyle1To2: dict[int, str] = {
    64: "regular",
    1: "italic",
    32: "bold",
    33: "bold italic",
}
_fontStyle2To1: dict[str, int] = _flipDict(_fontStyle1To2)
# Some UFO 1 files have 0
_fontStyle1To2[0] = "regular"

_widthName1To2: dict[str, int] = {
    "Ultra-condensed": 1,
    "Extra-condensed": 2,
    "Condensed": 3,
    "Semi-condensed": 4,
    "Medium (normal)": 5,
    "Semi-expanded": 6,
    "Expanded": 7,
    "Extra-expanded": 8,
    "Ultra-expanded": 9,
}
_widthName2To1: dict[int, str] = _flipDict(_widthName1To2)
# FontLab's default width value is "Normal".
# Many format version 1 UFOs will have this.
_widthName1To2["Normal"] = 5
# FontLab has an "All" width value. In UFO 1
# move this up to "Normal".
_widthName1To2["All"] = 5
# "medium" appears in a lot of UFO 1 files.
_widthName1To2["medium"] = 5
# "Medium" appears in a lot of UFO 1 files.
_widthName1To2["Medium"] = 5

_msCharSet1To2: dict[int, int] = {
    0: 1,
    1: 2,
    2: 3,
    77: 4,
    128: 5,
    129: 6,
    130: 7,
    134: 8,
    136: 9,
    161: 10,
    162: 11,
    163: 12,
    177: 13,
    178: 14,
    186: 15,
    200: 16,
    204: 17,
    222: 18,
    238: 19,
    255: 20,
}
_msCharSet2To1: dict[int, int] = _flipDict(_msCharSet1To2)

# 1 <-> 2


def convertFontInfoValueForAttributeFromVersion1ToVersion2(
    attr: str, value: Any
) -> tuple[str, Any]:
    """
    Convert value from version 1 to version 2 format.
    Returns the new attribute name and the converted value.
    If the value is None, None will be returned for the new value.
    """
    # convert floats to ints if possible
    if isinstance(value, float):
        if int(value) == value:
            value = int(value)
    if value is not None:
        if attr == "fontStyle":
            v: Optional[Union[str, int]] = _fontStyle1To2.get(value)
            if v is None:
                raise UFOLibError(
                    f"Cannot convert value ({value!r}) for attribute {attr}."
                )
            value = v
        elif attr == "widthName":
            v = _widthName1To2.get(value)
            if v is None:
                raise UFOLibError(
                    f"Cannot convert value ({value!r}) for attribute {attr}."
                )
            value = v
        elif attr == "msCharSet":
            v = _msCharSet1To2.get(value)
            if v is None:
                raise UFOLibError(
                    f"Cannot convert value ({value!r}) for attribute {attr}."
                )
            value = v
    attr = fontInfoAttributesVersion1To2.get(attr, attr)
    return attr, value


def convertFontInfoValueForAttributeFromVersion2ToVersion1(
    attr: str, value: Any
) -> tuple[str, Any]:
    """
    Convert value from version 2 to version 1 format.
    Returns the new attribute name and the converted value.
    If the value is None, None will be returned for the new value.
    """
    if value is not None:
        if attr == "styleMapStyleName":
            value = _fontStyle2To1.get(value)
        elif attr == "openTypeOS2WidthClass":
            value = _widthName2To1.get(value)
        elif attr == "postscriptWindowsCharacterSet":
            value = _msCharSet2To1.get(value)
    attr = fontInfoAttributesVersion2To1.get(attr, attr)
    return attr, value


def _convertFontInfoDataVersion1ToVersion2(data: dict[str, Any]) -> dict[str, Any]:
    converted = {}
    for attr, value in list(data.items()):
        # FontLab gives -1 for the weightValue
        # for fonts wil no defined value. Many
        # format version 1 UFOs will have this.
        if attr == "weightValue" and value == -1:
            continue
        newAttr, newValue = convertFontInfoValueForAttributeFromVersion1ToVersion2(
            attr, value
        )
        # skip if the attribute is not part of version 2
        if newAttr not in fontInfoAttributesVersion2:
            continue
        # catch values that can't be converted
        if value is None:
            raise UFOLibError(
                f"Cannot convert value ({value!r}) for attribute {newAttr}."
            )
        # store
        converted[newAttr] = newValue
    return converted


def _convertFontInfoDataVersion2ToVersion1(data: dict[str, Any]) -> dict[str, Any]:
    converted = {}
    for attr, value in list(data.items()):
        newAttr, newValue = convertFontInfoValueForAttributeFromVersion2ToVersion1(
            attr, value
        )
        # only take attributes that are registered for version 1
        if newAttr not in fontInfoAttributesVersion1:
            continue
        # catch values that can't be converted
        if value is None:
            raise UFOLibError(
                f"Cannot convert value ({value!r}) for attribute {newAttr}."
            )
        # store
        converted[newAttr] = newValue
    return converted


# 2 <-> 3

_ufo2To3NonNegativeInt: set[str] = {
    "versionMinor",
    "openTypeHeadLowestRecPPEM",
    "openTypeOS2WinAscent",
    "openTypeOS2WinDescent",
}
_ufo2To3NonNegativeIntOrFloat: set[str] = {
    "unitsPerEm",
}
_ufo2To3FloatToInt: set[str] = {
    "openTypeHeadLowestRecPPEM",
    "openTypeHheaAscender",
    "openTypeHheaDescender",
    "openTypeHheaLineGap",
    "openTypeHheaCaretOffset",
    "openTypeOS2TypoAscender",
    "openTypeOS2TypoDescender",
    "openTypeOS2TypoLineGap",
    "openTypeOS2WinAscent",
    "openTypeOS2WinDescent",
    "openTypeOS2SubscriptXSize",
    "openTypeOS2SubscriptYSize",
    "openTypeOS2SubscriptXOffset",
    "openTypeOS2SubscriptYOffset",
    "openTypeOS2SuperscriptXSize",
    "openTypeOS2SuperscriptYSize",
    "openTypeOS2SuperscriptXOffset",
    "openTypeOS2SuperscriptYOffset",
    "openTypeOS2StrikeoutSize",
    "openTypeOS2StrikeoutPosition",
    "openTypeVheaVertTypoAscender",
    "openTypeVheaVertTypoDescender",
    "openTypeVheaVertTypoLineGap",
    "openTypeVheaCaretOffset",
}


def convertFontInfoValueForAttributeFromVersion2ToVersion3(
    attr: str, value: Any
) -> tuple[str, Any]:
    """
    Convert value from version 2 to version 3 format.
    Returns the new attribute name and the converted value.
    If the value is None, None will be returned for the new value.
    """
    if attr in _ufo2To3FloatToInt:
        try:
            value = round(value)
        except (ValueError, TypeError):
            raise UFOLibError("Could not convert value for %s." % attr)
    if attr in _ufo2To3NonNegativeInt:
        try:
            value = int(abs(value))
        except (ValueError, TypeError):
            raise UFOLibError("Could not convert value for %s." % attr)
    elif attr in _ufo2To3NonNegativeIntOrFloat:
        try:
            v = float(abs(value))
        except (ValueError, TypeError):
            raise UFOLibError("Could not convert value for %s." % attr)
        if v == int(v):
            v = int(v)
        if v != value:
            value = v
    return attr, value


def convertFontInfoValueForAttributeFromVersion3ToVersion2(
    attr: str, value: Any
) -> tuple[str, Any]:
    """
    Convert value from version 3 to version 2 format.
    Returns the new attribute name and the converted value.
    If the value is None, None will be returned for the new value.
    """
    return attr, value


def _convertFontInfoDataVersion3ToVersion2(data: dict[str, Any]) -> dict[str, Any]:
    converted = {}
    for attr, value in list(data.items()):
        newAttr, newValue = convertFontInfoValueForAttributeFromVersion3ToVersion2(
            attr, value
        )
        if newAttr not in fontInfoAttributesVersion2:
            continue
        converted[newAttr] = newValue
    return converted


def _convertFontInfoDataVersion2ToVersion3(data: dict[str, Any]) -> dict[str, Any]:
    converted = {}
    for attr, value in list(data.items()):
        attr, value = convertFontInfoValueForAttributeFromVersion2ToVersion3(
            attr, value
        )
        converted[attr] = value
    return converted


if __name__ == "__main__":
    import doctest

    doctest.testmod()
<|endoftext|>"""DEPRECATED - This module is kept here only as a backward compatibility shim
for the old `ufoLib.plistlib` module, which was moved to :class:`fontTools.misc.plistlib`.
Please use the latter instead.
"""

from fontTools.misc.plistlib import dump, dumps, load, loads
from fontTools.misc.textTools import tobytes

# The following functions were part of the old py2-like ufoLib.plistlib API.
# They are kept only for backward compatiblity.
from fontTools.ufoLib.utils import deprecated


@deprecated("Use 'fontTools.misc.plistlib.load' instead")
def readPlist(path_or_file):
    did_open = False
    if isinstance(path_or_file, str):
        path_or_file = open(path_or_file, "rb")
        did_open = True
    try:
        return load(path_or_file, use_builtin_types=False)
    finally:
        if did_open:
            path_or_file.close()


@deprecated("Use 'fontTools.misc.plistlib.dump' instead")
def writePlist(value, path_or_file):
    did_open = False
    if isinstance(path_or_file, str):
        path_or_file = open(path_or_file, "wb")
        did_open = True
    try:
        dump(value, path_or_file, use_builtin_types=False)
    finally:
        if did_open:
            path_or_file.close()


@deprecated("Use 'fontTools.misc.plistlib.loads' instead")
def readPlistFromString(data):
    return loads(tobytes(data, encoding="utf-8"), use_builtin_types=False)


@deprecated("Use 'fontTools.misc.plistlib.dumps' instead")
def writePlistToString(value):
    return dumps(value, use_builtin_types=False)
<|endoftext|>from __future__ import annotations

from typing import Mapping, Any
from collections.abc import Container

from fontTools.annotations import KerningNested

"""
Functions for converting UFO1 or UFO2 files into UFO3 format.

Currently provides functionality for converting kerning rules
and kerning groups. Conversion is only supported _from_ UFO1
or UFO2, and _to_ UFO3.
"""

# adapted from the UFO spec


def convertUFO1OrUFO2KerningToUFO3Kerning(
    kerning: KerningNested, groups: dict[str, list[str]], glyphSet: Container[str] = ()
) -> tuple[KerningNested, dict[str, list[str]], dict[str, dict[str, str]]]:
    """Convert kerning data in UFO1 or UFO2 syntax into UFO3 syntax.

    Args:
      kerning:
          A dictionary containing the kerning rules defined in
          the UFO font, as used in :class:`.UFOReader` objects.
      groups:
          A dictionary containing the groups defined in the UFO
          font, as used in :class:`.UFOReader` objects.
      glyphSet:
        Optional; a set of glyph objects to skip (default: None).

    Returns:
      1. A dictionary representing the converted kerning data.
      2. A copy of the groups dictionary, with all groups renamed to UFO3 syntax.
      3. A dictionary containing the mapping of old group names to new group names.

    """
    # gather known kerning groups based on the prefixes
    firstReferencedGroups, secondReferencedGroups = findKnownKerningGroups(groups)
    # Make lists of groups referenced in kerning pairs.
    for first, seconds in list(kerning.items()):
        if first in groups and first not in glyphSet:
            if not first.startswith("public.kern1."):
                firstReferencedGroups.add(first)
        for second in list(seconds.keys()):
            if second in groups and second not in glyphSet:
                if not second.startswith("public.kern2."):
                    secondReferencedGroups.add(second)
    # Create new names for these groups.
    firstRenamedGroups: dict[str, str] = {}
    for first in firstReferencedGroups:
        # Make a list of existing group names.
        existingGroupNames = list(groups.keys()) + list(firstRenamedGroups.keys())
        # Remove the old prefix from the name
        newName = first.replace("@MMK_L_", "")
        # Add the new prefix to the name.
        newName = "public.kern1." + newName
        # Make a unique group name.
        newName = makeUniqueGroupName(newName, existingGroupNames)
        # Store for use later.
        firstRenamedGroups[first] = newName
    secondRenamedGroups: dict[str, str] = {}
    for second in secondReferencedGroups:
        # Make a list of existing group names.
        existingGroupNames = list(groups.keys()) + list(secondRenamedGroups.keys())
        # Remove the old prefix from the name
        newName = second.replace("@MMK_R_", "")
        # Add the new prefix to the name.
        newName = "public.kern2." + newName
        # Make a unique group name.
        newName = makeUniqueGroupName(newName, existingGroupNames)
        # Store for use later.
        secondRenamedGroups[second] = newName
    # Populate the new group names into the kerning dictionary as needed.
    newKerning = {}
    for first, seconds in list(kerning.items()):
        first = firstRenamedGroups.get(first, first)
        newSeconds = {}
        for second, value in list(seconds.items()):
            second = secondRenamedGroups.get(second, second)
            newSeconds[second] = value
        newKerning[first] = newSeconds
    # Make copies of the referenced groups and store them
    # under the new names in the overall groups dictionary.
    allRenamedGroups = list(firstRenamedGroups.items())
    allRenamedGroups += list(secondRenamedGroups.items())
    for oldName, newName in allRenamedGroups:
        group = list(groups[oldName])
        groups[newName] = group
    # Return the kerning and the groups.
    return newKerning, groups, dict(side1=firstRenamedGroups, side2=secondRenamedGroups)


def findKnownKerningGroups(groups: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    """Find all kerning groups in a UFO1 or UFO2 font that use known prefixes.

    In some cases, not all kerning groups will be referenced
    by the kerning pairs in a UFO. The algorithm for locating
    groups in :func:`convertUFO1OrUFO2KerningToUFO3Kerning` will
    miss these unreferenced groups. By scanning for known prefixes,
    this function will catch all of the prefixed groups.

    The prefixes and sides by this function are:

    @MMK_L_ - side 1
    @MMK_R_ - side 2

    as defined in the UFO1 specification.

    Args:
        groups:
          A dictionary containing the groups defined in the UFO
          font, as read by :class:`.UFOReader`.

    Returns:
        Two sets; the first containing the names of all
        first-side kerning groups identified in the ``groups``
        dictionary, and the second containing the names of all
        second-side kerning groups identified.

        "First-side" and "second-side" are with respect to the
        writing direction of the script.

        Example::

          >>> testGroups = {
          ...     "@MMK_L_1" : None,
          ...     "@MMK_L_2" : None,
          ...     "@MMK_L_3" : None,
          ...     "@MMK_R_1" : None,
          ...     "@MMK_R_2" : None,
          ...     "@MMK_R_3" : None,
          ...     "@MMK_l_1" : None,
          ...     "@MMK_r_1" : None,
          ...     "@MMK_X_1" : None,
          ...     "foo" : None,
          ... }
          >>> first, second = findKnownKerningGroups(testGroups)
          >>> sorted(first) == ['@MMK_L_1', '@MMK_L_2', '@MMK_L_3']
          True
          >>> sorted(second) == ['@MMK_R_1', '@MMK_R_2', '@MMK_R_3']
          True
    """
    knownFirstGroupPrefixes = ["@MMK_L_"]
    knownSecondGroupPrefixes = ["@MMK_R_"]
    firstGroups = set()
    secondGroups = set()
    for groupName in list(groups.keys()):
        for firstPrefix in knownFirstGroupPrefixes:
            if groupName.startswith(firstPrefix):
                firstGroups.add(groupName)
                break
        for secondPrefix in knownSecondGroupPrefixes:
            if groupName.startswith(secondPrefix):
                secondGroups.add(groupName)
                break
    return firstGroups, secondGroups


def makeUniqueGroupName(name: str, groupNames: list[str], counter: int = 0) -> str:
    """Make a kerning group name that will be unique within the set of group names.

    If the requested kerning group name already exists within the set, this
    will return a new name by adding an incremented counter to the end
    of the requested name.

    Args:
        name:
          The requested kerning group name.
        groupNames:
          A list of the existing kerning group names.
        counter:
          Optional; a counter of group names already seen (default: 0). If
          :attr:`.counter` is not provided, the function will recurse,
          incrementing the value of :attr:`.counter` until it finds the
          first unused ``name+counter`` combination, and return that result.

    Returns:
        A unique kerning group name composed of the requested name suffixed
        by the smallest available integer counter.
    """
    # Add a number to the name if the counter is higher than zero.
    newName = name
    if counter > 0:
        newName = "%s%d" % (newName, counter)
    # If the new name is in the existing group names, recurse.
    if newName in groupNames:
        return makeUniqueGroupName(name, groupNames, counter + 1)
    # Otherwise send back the new name.
    return newName


def test():
    """
    Tests for :func:`.convertUFO1OrUFO2KerningToUFO3Kerning`.

    No known prefixes.

    >>> testKerning = {
    ...     "A" : {
    ...         "A" : 1,
    ...         "B" : 2,
    ...         "CGroup" : 3,
    ...         "DGroup" : 4
    ...     },
    ...     "BGroup" : {
    ...         "A" : 5,
    ...         "B" : 6,
    ...         "CGroup" : 7,
    ...         "DGroup" : 8
    ...     },
    ...     "CGroup" : {
    ...         "A" : 9,
    ...         "B" : 10,
    ...         "CGroup" : 11,
    ...         "DGroup" : 12
    ...     },
    ... }
    >>> testGroups = {
    ...     "BGroup" : ["B"],
    ...     "CGroup" : ["C"],
    ...     "DGroup" : ["D"],
    ... }
    >>> kerning, groups, maps = convertUFO1OrUFO2KerningToUFO3Kerning(
    ...     testKerning, testGroups, [])
    >>> expected = {
    ...     "A" : {
    ...         "A": 1,
    ...         "B": 2,
    ...         "public.kern2.CGroup": 3,
    ...         "public.kern2.DGroup": 4
    ...     },
    ...     "public.kern1.BGroup": {
    ...         "A": 5,
    ...         "B": 6,
    ...         "public.kern2.CGroup": 7,
    ...         "public.kern2.DGroup": 8
    ...     },
    ...     "public.kern1.CGroup": {
    ...         "A": 9,
    ...         "B": 10,
    ...         "public.kern2.CGroup": 11,
    ...         "public.kern2.DGroup": 12
    ...     }
    ... }
    >>> kerning == expected
    True
    >>> expected = {
    ...     "BGroup": ["B"],
    ...     "CGroup": ["C"],
    ...     "DGroup": ["D"],
    ...     "public.kern1.BGroup": ["B"],
    ...     "public.kern1.CGroup": ["C"],
    ...     "public.kern2.CGroup": ["C"],
    ...     "public.kern2.DGroup": ["D"],
    ... }
    >>> groups == expected
    True

    Known prefixes.

    >>> testKerning = {
    ...     "A" : {
    ...         "A" : 1,
    ...         "B" : 2,
    ...         "@MMK_R_CGroup" : 3,
    ...         "@MMK_R_DGroup" : 4
    ...     },
    ...     "@MMK_L_BGroup" : {
    ...         "A" : 5,
    ...         "B" : 6,
    ...         "@MMK_R_CGroup" : 7,
    ...         "@MMK_R_DGroup" : 8
    ...     },
    ...     "@MMK_L_CGroup" : {
    ...         "A" : 9,
    ...         "B" : 10,
    ...         "@MMK_R_CGroup" : 11,
    ...         "@MMK_R_DGroup" : 12
    ...     },
    ... }
    >>> testGroups = {
    ...     "@MMK_L_BGroup" : ["B"],
    ...     "@MMK_L_CGroup" : ["C"],
    ...     "@MMK_L_XGroup" : ["X"],
    ...     "@MMK_R_CGroup" : ["C"],
    ...     "@MMK_R_DGroup" : ["D"],
    ...     "@MMK_R_XGroup" : ["X"],
    ... }
    >>> kerning, groups, maps = convertUFO1OrUFO2KerningToUFO3Kerning(
    ...     testKerning, testGroups, [])
    >>> expected = {
    ...     "A" : {
    ...         "A": 1,
    ...         "B": 2,
    ...         "public.kern2.CGroup": 3,
    ...         "public.kern2.DGroup": 4
    ...     },
    ...     "public.kern1.BGroup": {
    ...         "A": 5,
    ...         "B": 6,
    ...         "public.kern2.CGroup": 7,
    ...         "public.kern2.DGroup": 8
    ...     },
    ...     "public.kern1.CGroup": {
    ...         "A": 9,
    ...         "B": 10,
    ...         "public.kern2.CGroup": 11,
    ...         "public.kern2.DGroup": 12
    ...     }
    ... }
    >>> kerning == expected
    True
    >>> expected = {
    ...     "@MMK_L_BGroup": ["B"],
    ...     "@MMK_L_CGroup": ["C"],
    ...     "@MMK_L_XGroup": ["X"],
    ...     "@MMK_R_CGroup": ["C"],
    ...     "@MMK_R_DGroup": ["D"],
    ...     "@MMK_R_XGroup": ["X"],
    ...     "public.kern1.BGroup": ["B"],
    ...     "public.kern1.CGroup": ["C"],
    ...     "public.kern1.XGroup": ["X"],
    ...     "public.kern2.CGroup": ["C"],
    ...     "public.kern2.DGroup": ["D"],
    ...     "public.kern2.XGroup": ["X"],
    ... }
    >>> groups == expected
    True

    >>> from .validators import kerningValidator
    >>> kerningValidator(kerning)
    (True, None)

    Mixture of known prefixes and groups without prefixes.

    >>> testKerning = {
    ...     "A" : {
    ...         "A" : 1,
    ...         "B" : 2,
    ...         "@MMK_R_CGroup" : 3,
    ...         "DGroup" : 4
    ...     },
    ...     "BGroup" : {
    ...         "A" : 5,
    ...         "B" : 6,
    ...         "@MMK_R_CGroup" : 7,
    ...         "DGroup" : 8
    ...     },
    ...     "@MMK_L_CGroup" : {
    ...         "A" : 9,
    ...         "B" : 10,
    ...         "@MMK_R_CGroup" : 11,
    ...         "DGroup" : 12
    ...     },
    ... }
    >>> testGroups = {
    ...     "BGroup" : ["B"],
    ...     "@MMK_L_CGroup" : ["C"],
    ...     "@MMK_R_CGroup" : ["C"],
    ...     "DGroup" : ["D"],
    ... }
    >>> kerning, groups, maps = convertUFO1OrUFO2KerningToUFO3Kerning(
    ...     testKerning, testGroups, [])
    >>> expected = {
    ...     "A" : {
    ...         "A": 1,
    ...         "B": 2,
    ...         "public.kern2.CGroup": 3,
    ...         "public.kern2.DGroup": 4
    ...     },
    ...     "public.kern1.BGroup": {
    ...         "A": 5,
    ...         "B": 6,
    ...         "public.kern2.CGroup": 7,
    ...         "public.kern2.DGroup": 8
    ...     },
    ...     "public.kern1.CGroup": {
    ...         "A": 9,
    ...         "B": 10,
    ...         "public.kern2.CGroup": 11,
    ...         "public.kern2.DGroup": 12
    ...     }
    ... }
    >>> kerning == expected
    True
    >>> expected = {
    ...     "BGroup": ["B"],
    ...     "@MMK_L_CGroup": ["C"],
    ...     "@MMK_R_CGroup": ["C"],
    ...     "DGroup": ["D"],
    ...     "public.kern1.BGroup": ["B"],
    ...     "public.kern1.CGroup": ["C"],
    ...     "public.kern2.CGroup": ["C"],
    ...     "public.kern2.DGroup": ["D"],
    ... }
    >>> groups == expected
    True
    """


if __name__ == "__main__":
    import doctest

    doctest.testmod()
<|endoftext|># cython: language_level=3
# distutils: define_macros=CYTHON_TRACE_NOGIL=1

# Copyright 2023 Google Inc. All Rights Reserved.
# Copyright 2023 Behdad Esfahbod. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

try:
    import cython
except (AttributeError, ImportError):
    # if cython not installed, use mock module with no-op decorators and types
    from fontTools.misc import cython
COMPILED = cython.compiled

from fontTools.misc.bezierTools import splitCubicAtTC
from collections import namedtuple
import math
from typing import (
    List,
    Sequence,
    Tuple,
    Union,
)

__all__ = ["quadratic_to_curves"]


# Copied from cu2qu
@cython.cfunc
@cython.returns(cython.int)
@cython.locals(
    tolerance=cython.double,
    p0=cython.complex,
    p1=cython.complex,
    p2=cython.complex,
    p3=cython.complex,
)
@cython.locals(mid=cython.complex, deriv3=cython.complex)
def cubic_farthest_fit_inside(p0, p1, p2, p3, tolerance):
    """Check if a cubic Bezier lies within a given distance of the origin.

    "Origin" means *the* origin (0,0), not the start of the curve. Note that no
    checks are made on the start and end positions of the curve; this function
    only checks the inside of the curve.

    Args:
        p0 (complex): Start point of curve.
        p1 (complex): First handle of curve.
        p2 (complex): Second handle of curve.
        p3 (complex): End point of curve.
        tolerance (double): Distance from origin.

    Returns:
        bool: True if the cubic Bezier ``p`` entirely lies within a distance
        ``tolerance`` of the origin, False otherwise.
    """
    # First check p2 then p1, as p2 has higher error early on.
    if abs(p2) <= tolerance and abs(p1) <= tolerance:
        return True

    # Split.
    mid = (p0 + 3 * (p1 + p2) + p3) * 0.125
    if abs(mid) > tolerance:
        return False
    deriv3 = (p3 + p2 - p1 - p0) * 0.125
    return cubic_farthest_fit_inside(
        p0, (p0 + p1) * 0.5, mid - deriv3, mid, tolerance
    ) and cubic_farthest_fit_inside(mid, mid + deriv3, (p2 + p3) * 0.5, p3, tolerance)


@cython.locals(
    p0=cython.complex,
    p1=cython.complex,
    p2=cython.complex,
    p1_2_3=cython.complex,
)
def elevate_quadratic(p0, p1, p2):
    """Given a quadratic bezier curve, return its degree-elevated cubic."""

    # https://pomax.github.io/bezierinfo/#reordering
    p1_2_3 = p1 * (2 / 3)
    return (
        p0,
        (p0 * (1 / 3) + p1_2_3),
        (p2 * (1 / 3) + p1_2_3),
        p2,
    )


@cython.cfunc
@cython.locals(
    start=cython.int,
    n=cython.int,
    k=cython.int,
    prod_ratio=cython.double,
    sum_ratio=cython.double,
    ratio=cython.double,
    t=cython.double,
    p0=cython.complex,
    p1=cython.complex,
    p2=cython.complex,
    p3=cython.complex,
)
def merge_curves(curves, start, n):
    """Give a cubic-Bezier spline, reconstruct one cubic-Bezier
    that has the same endpoints and tangents and approxmates
    the spline."""

    # Reconstruct the t values of the cut segments
    prod_ratio = 1.0
    sum_ratio = 1.0
    ts = [1]
    for k in range(1, n):
        ck = curves[start + k]
        c_before = curves[start + k - 1]

        # |t_(k+1) - t_k| / |t_k - t_(k - 1)| = ratio
        assert ck[0] == c_before[3]
        ratio = abs(ck[1] - ck[0]) / abs(c_before[3] - c_before[2])

        prod_ratio *= ratio
        sum_ratio += prod_ratio
        ts.append(sum_ratio)

    # (t(n) - t(n - 1)) / (t_(1) - t(0)) = prod_ratio

    ts = [t / sum_ratio for t in ts[:-1]]

    p0 = curves[start][0]
    p1 = curves[start][1]
    p2 = curves[start + n - 1][2]
    p3 = curves[start + n - 1][3]

    # Build the curve by scaling the control-points.
    p1 = p0 + (p1 - p0) / (ts[0] if ts else 1)
    p2 = p3 + (p2 - p3) / ((1 - ts[-1]) if ts else 1)

    curve = (p0, p1, p2, p3)

    return curve, ts


@cython.locals(
    count=cython.int,
    num_offcurves=cython.int,
    i=cython.int,
    off1=cython.complex,
    off2=cython.complex,
    on=cython.complex,
)
def add_implicit_on_curves(p):
    q = list(p)
    count = 0
    num_offcurves = len(p) - 2
    for i in range(1, num_offcurves):
        off1 = p[i]
        off2 = p[i + 1]
        on = off1 + (off2 - off1) * 0.5
        q.insert(i + 1 + count, on)
        count += 1
    return q


Point = Union[Tuple[float, float], complex]


def _raise_incompatible_point(point, previous_point):
    raise ValueError(
        f"Quadratic splines must connect end-to-start; got {previous_point!r} then {point!r}"
    )


def _validate_spline_length(spline):
    if len(spline) < 3:
        raise ValueError("Quadratic splines must contain at least 3 points")


def _validate_positive_tolerance(max_err):
    if max_err <= 0:
        raise ValueError("max_err must be greater than zero")


@cython.locals(
    cost=cython.int,
    is_complex=cython.int,
)
def quadratic_to_curves(
    quads: List[Sequence[Point]],
    max_err: float = 0.5,
    all_cubic: bool = False,
) -> List[Tuple[Point, ...]]:
    """Converts a connecting list of quadratic splines to a list of quadratic
    and cubic curves.

    A quadratic spline is specified as a sequence of points.  Either each point is
    a 2-tuple of X,Y coordinates, or each point is a complex number with
    real/imaginary components representing X,Y coordinates.

    The first and last points are on-curve points and the rest are off-curve
    points, with an implied on-curve point in the middle between every two
    consequtive off-curve points.

    Returns:
        The output is a list of tuples of points. Points are represented
        in the same format as the input, either as 2-tuples or complex numbers.
        If ``quads`` is empty, returns an empty list.

        Each tuple is either of length three, for a quadratic curve, or four,
        for a cubic curve.  Each curve's last point is the same as the next
        curve's first point.

    Args:
        quads: quadratic splines

        max_err: absolute error tolerance; defaults to 0.5

        all_cubic: if True, only cubic curves are generated; defaults to False

    Raises:
        ValueError: if an input spline has fewer than 3 points, or if adjacent
        splines do not connect end-to-start.
    """
    if not quads:
        return []
    _validate_positive_tolerance(max_err)
    for spline in quads:
        _validate_spline_length(spline)

    is_complex = type(quads[0][0]) is complex
    if not is_complex:
        quads = [[complex(x, y) for (x, y) in p] for p in quads]

    q = [quads[0][0]]
    costs = [1]
    cost = 1
    for p in quads:
        if q[-1] != p[0]:
            _raise_incompatible_point(p[0], q[-1])
        for i in range(len(p) - 2):
            cost += 1
            costs.append(cost)
            costs.append(cost)
        qq = add_implicit_on_curves(p)[1:]
        costs.pop()
        q.extend(qq)
        cost += 1
        costs.append(cost)

    curves = spline_to_curves(q, costs, max_err, all_cubic)

    if not is_complex:
        curves = [tuple((c.real, c.imag) for c in curve) for curve in curves]
    return curves


Solution = namedtuple("Solution", ["num_points", "error", "start_index", "is_cubic"])


@cython.locals(
    i=cython.int,
    j=cython.int,
    k=cython.int,
    start=cython.int,
    i_sol_count=cython.int,
    j_sol_count=cython.int,
    this_sol_count=cython.int,
    tolerance=cython.double,
    err=cython.double,
    error=cython.double,
    i_sol_error=cython.double,
    j_sol_error=cython.double,
    all_cubic=cython.int,
    is_cubic=cython.int,
    count=cython.int,
    p0=cython.complex,
    p1=cython.complex,
    p2=cython.complex,
    p3=cython.complex,
    v=cython.complex,
    u=cython.complex,
)
def spline_to_curves(q, costs, tolerance=0.5, all_cubic=False):
    """
    q: quadratic spline with alternating on-curve / off-curve points.

    costs: cumulative list of encoding cost of q in terms of number of
      points that need to be encoded.  Implied on-curve points do not
      contribute to the cost. If all points need to be encoded, then
      costs will be range(1, len(q)+1).
    """

    assert len(q) >= 3, "quadratic spline requires at least 3 points"

    # Elevate quadratic segments to cubic
    elevated_quadratics = [
        elevate_quadratic(*q[i : i + 3]) for i in range(0, len(q) - 2, 2)
    ]

    # Find sharp corners; they have to be oncurves for sure.
    forced = set()
    for i in range(1, len(elevated_quadratics)):
        p0 = elevated_quadratics[i - 1][2]
        p1 = elevated_quadratics[i][0]
        p2 = elevated_quadratics[i][1]
        if abs(p1 - p0) + abs(p2 - p1) > tolerance + abs(p2 - p0):
            forced.add(i)

    # Dynamic-Programming to find the solution with fewest number of
    # cubic curves, and within those the one with smallest error.
    sols = [Solution(0, 0, 0, False)]
    impossible = Solution(len(elevated_quadratics) * 3 + 1, 0, 1, False)
    start = 0
    for i in range(1, len(elevated_quadratics) + 1):
        best_sol = impossible
        for j in range(start, i):
            j_sol_count, j_sol_error = sols[j].num_points, sols[j].error

            if not all_cubic:
                # Solution with quadratics between j:i
                this_count = costs[2 * i - 1] - costs[2 * j] + 1
                i_sol_count = j_sol_count + this_count
                i_sol_error = j_sol_error
                i_sol = Solution(i_sol_count, i_sol_error, i - j, False)
                if i_sol < best_sol:
                    best_sol = i_sol

                if this_count <= 3:
                    # Can't get any better than this in the path below
                    continue

            # Fit elevated_quadratics[j:i] into one cubic
            try:
                curve, ts = merge_curves(elevated_quadratics, j, i - j)
            except ZeroDivisionError:
                continue

            # Now reconstruct the segments from the fitted curve
            reconstructed_iter = splitCubicAtTC(*curve, *ts)
            reconstructed = []

            # Knot errors
            error = 0
            for k, reconst in enumerate(reconstructed_iter):
                orig = elevated_quadratics[j + k]
                err = abs(reconst[3] - orig[3])
                error = max(error, err)
                if error > tolerance:
                    break
                reconstructed.append(reconst)
            if error > tolerance:
                # Not feasible
                continue

            # Interior errors
            for k, reconst in enumerate(reconstructed):
                orig = elevated_quadratics[j + k]
                p0, p1, p2, p3 = tuple(v - u for v, u in zip(reconst, orig))

                if not cubic_farthest_fit_inside(p0, p1, p2, p3, tolerance):
                    error = tolerance + 1
                    break
            if error > tolerance:
                # Not feasible
                continue

            # Save best solution
            i_sol_count = j_sol_count + 3
            i_sol_error = max(j_sol_error, error)
            i_sol = Solution(i_sol_count, i_sol_error, i - j, True)
            if i_sol < best_sol:
                best_sol = i_sol

            if i_sol_count == 3:
                # Can't get any better than this
                break

        sols.append(best_sol)
        if i in forced:
            start = i

    # Reconstruct solution
    splits = []
    cubic = []
    i = len(sols) - 1
    while i:
        count, is_cubic = sols[i].start_index, sols[i].is_cubic
        splits.append(i)
        cubic.append(is_cubic)
        i -= count
    curves = []
    j = 0
    for i, is_cubic in reversed(list(zip(splits, cubic))):
        if is_cubic:
            curves.append(merge_curves(elevated_quadratics, j, i - j)[0])
        else:
            for k in range(j, i):
                curves.append(q[k * 2 : k * 2 + 3])
        j = i

    return curves


def main():
    from fontTools.cu2qu.benchmark import generate_curve
    from fontTools.cu2qu import curve_to_quadratic

    tolerance = 0.05
    reconstruct_tolerance = tolerance * 1
    curve = generate_curve()
    quadratics = curve_to_quadratic(curve, tolerance)
    print(
        "cu2qu tolerance %g. qu2cu tolerance %g." % (tolerance, reconstruct_tolerance)
    )
    print("One random cubic turned into %d quadratics." % len(quadratics))
    curves = quadratic_to_curves([quadratics], reconstruct_tolerance)
    print("Those quadratics turned back into %d cubics. " % len(curves))
    print("Original curve:", curve)
    print("Reconstructed curve(s):", curves)


if __name__ == "__main__":
    main()
<|endoftext|>"""Benchmark the qu2cu algorithm performance."""

from .qu2cu import *
from fontTools.cu2qu import curve_to_quadratic
import random
import timeit

MAX_ERR = 0.5
NUM_CURVES = 5


def generate_curves(n):
    points = [
        tuple(float(random.randint(0, 2048)) for coord in range(2))
        for point in range(1 + 3 * n)
    ]
    curves = []
    for i in range(n):
        curves.append(tuple(points[i * 3 : i * 3 + 4]))
    return curves


def setup_quadratic_to_curves():
    curves = generate_curves(NUM_CURVES)
    quadratics = [curve_to_quadratic(curve, MAX_ERR) for curve in curves]
    return quadratics, MAX_ERR


def run_benchmark(module, function, setup_suffix="", repeat=25, number=1):
    setup_func = "setup_" + function
    if setup_suffix:
        print("%s with %s:" % (function, setup_suffix), end="")
        setup_func += "_" + setup_suffix
    else:
        print("%s:" % function, end="")

    def wrapper(function, setup_func):
        function = globals()[function]
        setup_func = globals()[setup_func]

        def wrapped():
            return function(*setup_func())

        return wrapped

    results = timeit.repeat(wrapper(function, setup_func), repeat=repeat, number=number)
    print("\t%5.1fus" % (min(results) * 1000000.0 / number))


def main():
    run_benchmark("qu2cu", "quadratic_to_curves")


if __name__ == "__main__":
    random.seed(1)
    main()
<|endoftext|>import os
import argparse
import logging
from fontTools.misc.cliTools import makeOutputFileName
from fontTools.ttLib import TTFont
from fontTools.pens.qu2cuPen import Qu2CuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
import fontTools

logger = logging.getLogger("fontTools.qu2cu")


def _font_to_cubic(input_path, output_path=None, **kwargs):
    font = TTFont(input_path)
    logger.info("Converting curves for %s", input_path)

    stats = {} if kwargs["dump_stats"] else None
    qu2cu_kwargs = {
        "stats": stats,
        "max_err": kwargs["max_err_em"] * font["head"].unitsPerEm,
        "all_cubic": kwargs["all_cubic"],
    }

    if "gvar" in font:
        raise ValueError("Cannot convert variable font")
    glyphSet = font.getGlyphSet()
    glyphOrder = font.getGlyphOrder()
    glyf = font["glyf"]
    for glyphName in glyphOrder:
        glyph = glyphSet[glyphName]
        ttpen = TTGlyphPen(glyphSet)
        pen = Qu2CuPen(ttpen, **qu2cu_kwargs)
        glyph.draw(pen)
        glyf[glyphName] = ttpen.glyph(dropImpliedOnCurves=True)

    font["head"].glyphDataFormat = 1

    if kwargs["dump_stats"]:
        logger.info("Stats: %s", stats)

    logger.info("Saving %s", output_path)
    font.save(output_path)


def _main(args=None):
    """Convert an OpenType font from quadratic to cubic curves"""
    parser = argparse.ArgumentParser(prog="qu2cu")
    parser.add_argument("--version", action="version", version=fontTools.__version__)
    parser.add_argument(
        "infiles",
        nargs="+",
        metavar="INPUT",
        help="one or more input TTF source file(s).",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument(
        "-e",
        "--conversion-error",
        type=float,
        metavar="ERROR",
        default=0.001,
        help="maxiumum approximation error measured in EM (default: 0.001)",
    )
    parser.add_argument(
        "-c",
        "--all-cubic",
        default=False,
        action="store_true",
        help="whether to only use cubic curves",
    )

    output_parser = parser.add_mutually_exclusive_group()
    output_parser.add_argument(
        "-o",
        "--output-file",
        default=None,
        metavar="OUTPUT",
        help=("output filename for the converted TTF."),
    )
    output_parser.add_argument(
        "-d",
        "--output-dir",
        default=None,
        metavar="DIRECTORY",
        help="output directory where to save converted TTFs",
    )

    options = parser.parse_args(args)

    if options.conversion_error <= 0:
        parser.error("--conversion-error must be greater than zero")

    if not options.verbose:
        level = "WARNING"
    elif options.verbose == 1:
        level = "INFO"
    else:
        level = "DEBUG"
    logging.basicConfig(level=level)

    if len(options.infiles) > 1 and options.output_file:
        parser.error("-o/--output-file can't be used with multile inputs")

    if options.output_dir:
        output_dir = options.output_dir
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        elif not os.path.isdir(output_dir):
            parser.error("'%s' is not a directory" % output_dir)
        output_paths = [
            os.path.join(output_dir, os.path.basename(p)) for p in options.infiles
        ]
    elif options.output_file:
        output_paths = [options.output_file]
    else:
        output_paths = [
            makeOutputFileName(p, overWrite=True, suffix=".cubic")
            for p in options.infiles
        ]

    kwargs = dict(
        dump_stats=options.verbose > 0,
        max_err_em=options.conversion_error,
        all_cubic=options.all_cubic,
    )

    for input_path, output_path in zip(options.infiles, output_paths):
        _font_to_cubic(input_path, output_path, **kwargs)
<|endoftext|>import re


def _prefer_non_zero(*args):
    for arg in args:
        if arg != 0:
            return arg
    return 0.0


def _ntos(n):
    # %f likes to add unnecessary 0's, %g isn't consistent about # decimals
    return ("%.3f" % n).rstrip("0").rstrip(".")


def _strip_xml_ns(tag):
    # ElementTree API doesn't provide a way to ignore XML namespaces in tags
    # so we here strip them ourselves: cf. https://bugs.python.org/issue18304
    return tag.split("}", 1)[1] if "}" in tag else tag


def _transform(raw_value):
    # TODO assumes a 'matrix' transform.
    # No other transform functions are supported at the moment.
    # https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/transform
    # start simple: if you aren't exactly matrix(...) then no love
    match = re.match(r"matrix\((.*)\)", raw_value)
    if not match:
        raise NotImplementedError
    matrix = tuple(float(p) for p in re.split(r"\s+|,", match.group(1)))
    if len(matrix) != 6:
        raise ValueError("wrong # of terms in %s" % raw_value)
    return matrix


class PathBuilder(object):
    def __init__(self):
        self.paths = []
        self.transforms = []

    def _start_path(self, initial_path=""):
        self.paths.append(initial_path)
        self.transforms.append(None)

    def _end_path(self):
        self._add("z")

    def _add(self, path_snippet):
        path = self.paths[-1]
        if path:
            path += " " + path_snippet
        else:
            path = path_snippet
        self.paths[-1] = path

    def _move(self, c, x, y):
        self._add("%s%s,%s" % (c, _ntos(x), _ntos(y)))

    def M(self, x, y):
        self._move("M", x, y)

    def m(self, x, y):
        self._move("m", x, y)

    def _arc(self, c, rx, ry, x, y, large_arc):
        self._add(
            "%s%s,%s 0 %d 1 %s,%s"
            % (c, _ntos(rx), _ntos(ry), large_arc, _ntos(x), _ntos(y))
        )

    def A(self, rx, ry, x, y, large_arc=0):
        self._arc("A", rx, ry, x, y, large_arc)

    def a(self, rx, ry, x, y, large_arc=0):
        self._arc("a", rx, ry, x, y, large_arc)

    def _vhline(self, c, x):
        self._add("%s%s" % (c, _ntos(x)))

    def H(self, x):
        self._vhline("H", x)

    def h(self, x):
        self._vhline("h", x)

    def V(self, y):
        self._vhline("V", y)

    def v(self, y):
        self._vhline("v", y)

    def _line(self, c, x, y):
        self._add("%s%s,%s" % (c, _ntos(x), _ntos(y)))

    def L(self, x, y):
        self._line("L", x, y)

    def l(self, x, y):
        self._line("l", x, y)

    def _parse_line(self, line):
        x1 = float(line.attrib.get("x1", 0))
        y1 = float(line.attrib.get("y1", 0))
        x2 = float(line.attrib.get("x2", 0))
        y2 = float(line.attrib.get("y2", 0))

        self._start_path()
        self.M(x1, y1)
        self.L(x2, y2)

    def _parse_rect(self, rect):
        x = float(rect.attrib.get("x", 0))
        y = float(rect.attrib.get("y", 0))
        w = float(rect.attrib.get("width"))
        h = float(rect.attrib.get("height"))
        rx = float(rect.attrib.get("rx", 0))
        ry = float(rect.attrib.get("ry", 0))

        rx = _prefer_non_zero(rx, ry)
        ry = _prefer_non_zero(ry, rx)
        # TODO there are more rules for adjusting rx, ry

        self._start_path()
        self.M(x + rx, y)
        self.H(x + w - rx)
        if rx > 0:
            self.A(rx, ry, x + w, y + ry)
        self.V(y + h - ry)
        if rx > 0:
            self.A(rx, ry, x + w - rx, y + h)
        self.H(x + rx)
        if rx > 0:
            self.A(rx, ry, x, y + h - ry)
        self.V(y + ry)
        if rx > 0:
            self.A(rx, ry, x + rx, y)
        self._end_path()

    def _parse_path(self, path):
        if "d" in path.attrib:
            self._start_path(initial_path=path.attrib["d"])

    def _parse_polygon(self, poly):
        if "points" in poly.attrib:
            self._start_path("M" + poly.attrib["points"])
            self._end_path()

    def _parse_polyline(self, poly):
        if "points" in poly.attrib:
            self._start_path("M" + poly.attrib["points"])

    def _parse_circle(self, circle):
        cx = float(circle.attrib.get("cx", 0))
        cy = float(circle.attrib.get("cy", 0))
        r = float(circle.attrib.get("r"))

        # arc doesn't seem to like being a complete shape, draw two halves
        self._start_path()
        self.M(cx - r, cy)
        self.A(r, r, cx + r, cy, large_arc=1)
        self.A(r, r, cx - r, cy, large_arc=1)

    def _parse_ellipse(self, ellipse):
        cx = float(ellipse.attrib.get("cx", 0))
        cy = float(ellipse.attrib.get("cy", 0))
        rx = float(ellipse.attrib.get("rx"))
        ry = float(ellipse.attrib.get("ry"))

        # arc doesn't seem to like being a complete shape, draw two halves
        self._start_path()
        self.M(cx - rx, cy)
        self.A(rx, ry, cx + rx, cy, large_arc=1)
        self.A(rx, ry, cx - rx, cy, large_arc=1)

    def add_path_from_element(self, el):
        if not isinstance(el.tag, str):
            return False
        tag = _strip_xml_ns(el.tag)
        parse_fn = getattr(self, "_parse_%s" % tag.lower(), None)
        if not callable(parse_fn):
            return False
        parse_fn(el)
        if "transform" in el.attrib:
            self.transforms[-1] = _transform(el.attrib["transform"])
        return True
<|endoftext|>from fontTools.pens.transformPen import TransformPen
from fontTools.misc import etree
from fontTools.misc.textTools import tostr
from .parser import parse_path
from .shapes import PathBuilder

__all__ = [tostr(s) for s in ("SVGPath", "parse_path")]


class SVGPath(object):
    """Parse SVG ``path`` elements from a file or string, and draw them
    onto a glyph object that supports the FontTools Pen protocol.

    For example, reading from an SVG file and drawing to a Defcon Glyph:

    .. code-block::

        import defcon
        glyph = defcon.Glyph()
        pen = glyph.getPen()
        svg = SVGPath("path/to/a.svg")
        svg.draw(pen)

    Or reading from a string containing SVG data, using the alternative
    'fromstring' (a class method):

    .. code-block::

        data = '<?xml version="1.0" ...'
        svg = SVGPath.fromstring(data)
        svg.draw(pen)

    Both constructors can optionally take a 'transform' matrix (6-float
    tuple, or a FontTools Transform object) to modify the draw output.
    """

    def __init__(self, filename=None, transform=None):
        if filename is None:
            self.root = etree.ElementTree()
        else:
            tree = etree.parse(filename, parser=etree.XMLParser())
            self.root = tree.getroot()
        self.transform = transform

    @classmethod
    def fromstring(cls, data, transform=None):
        self = cls(transform=transform)
        self.root = etree.fromstring(data, parser=etree.XMLParser())
        return self

    def draw(self, pen):
        if self.transform:
            pen = TransformPen(pen, self.transform)
        pb = PathBuilder()
        # xpath | doesn't seem to reliable work so just walk it
        for el in self.root.iter():
            pb.add_path_from_element(el)
        original_pen = pen
        for path, transform in zip(pb.paths, pb.transforms):
            if transform:
                pen = TransformPen(original_pen, transform)
            else:
                pen = original_pen
            parse_path(path, pen)
<|endoftext|># SVG Path specification parser.
# This is an adaptation from 'svg.path' by Lennart Regebro (@regebro),
# modified so that the parser takes a FontTools Pen object instead of
# returning a list of svg.path Path objects.
# The original code can be found at:
# https://github.com/regebro/svg.path/blob/4f9b6e3/src/svg/path/parser.py
# Copyright (c) 2013-2014 Lennart Regebro
# License: MIT

from .arc import EllipticalArc
import cmath
import re

COMMANDS = set("MmZzLlHhVvCcSsQqTtAa")
ARC_COMMANDS = set("Aa")
UPPERCASE = set("MZLHVCSQTA")

COMMAND_RE = re.compile("([MmZzLlHhVvCcSsQqTtAa])")

# https://www.w3.org/TR/css-syntax-3/#number-token-diagram
#   but -6.e-5 will be tokenized as "-6" then "-5" and confuse parsing
FLOAT_RE = re.compile(
    r"[-+]?"  # optional sign
    r"(?:"
    r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?"  # int/float
    r"|"
    r"(?:\.[0-9]+(?:[eE][-+]?[0-9]+)?)"  # float with leading dot (e.g. '.42')
    r")"
)
BOOL_RE = re.compile("^[01]")
SEPARATOR_RE = re.compile(f"[, \t]")


def _tokenize_path(pathdef):
    arc_cmd = None
    for x in COMMAND_RE.split(pathdef):
        if x in COMMANDS:
            arc_cmd = x if x in ARC_COMMANDS else None
            yield x
            continue

        if arc_cmd:
            try:
                yield from _tokenize_arc_arguments(x)
            except ValueError as e:
                raise ValueError(f"Invalid arc command: '{arc_cmd}{x}'") from e
        else:
            for token in FLOAT_RE.findall(x):
                yield token


ARC_ARGUMENT_TYPES = (
    ("rx", FLOAT_RE),
    ("ry", FLOAT_RE),
    ("x-axis-rotation", FLOAT_RE),
    ("large-arc-flag", BOOL_RE),
    ("sweep-flag", BOOL_RE),
    ("x", FLOAT_RE),
    ("y", FLOAT_RE),
)


def _tokenize_arc_arguments(arcdef):
    raw_args = [s for s in SEPARATOR_RE.split(arcdef) if s]
    if not raw_args:
        raise ValueError(f"Not enough arguments: '{arcdef}'")
    raw_args.reverse()

    i = 0
    while raw_args:
        arg = raw_args.pop()

        name, pattern = ARC_ARGUMENT_TYPES[i]
        match = pattern.search(arg)
        if not match:
            raise ValueError(f"Invalid argument for '{name}' parameter: {arg!r}")

        j, k = match.span()
        yield arg[j:k]
        arg = arg[k:]

        if arg:
            raw_args.append(arg)

        # wrap around every 7 consecutive arguments
        if i == 6:
            i = 0
        else:
            i += 1

    if i != 0:
        raise ValueError(f"Not enough arguments: '{arcdef}'")


def parse_path(pathdef, pen, current_pos=(0, 0), arc_class=EllipticalArc):
    """Parse SVG path definition (i.e. "d" attribute of <path> elements)
    and call a 'pen' object's moveTo, lineTo, curveTo, qCurveTo and closePath
    methods.

    If 'current_pos' (2-float tuple) is provided, the initial moveTo will
    be relative to that instead being absolute.

    If the pen has an "arcTo" method, it is called with the original values
    of the elliptical arc curve commands:

    .. code-block::

        pen.arcTo(rx, ry, rotation, arc_large, arc_sweep, (x, y))

    Otherwise, the arcs are approximated by series of cubic Bezier segments
    ("curveTo"), one every 90 degrees.
    """
    # In the SVG specs, initial movetos are absolute, even if
    # specified as 'm'. This is the default behavior here as well.
    # But if you pass in a current_pos variable, the initial moveto
    # will be relative to that current_pos. This is useful.
    current_pos = complex(*current_pos)

    elements = list(_tokenize_path(pathdef))
    # Reverse for easy use of .pop()
    elements.reverse()

    # A path data segment (if there is one) must begin with a moveto
    # command (https://www.w3.org/TR/SVG11/paths.html#PathDataMovetoCommands);
    # raise before any pen call instead of inventing a start point.
    if elements and elements[-1] not in ("M", "m"):
        raise ValueError("Path must start with a moveto command: %r" % pathdef)

    # start_pos is the initial point of the current subpath; it is retained
    # after a closepath so that a drawto command following Z can start a new
    # subpath at that same point, while subpath_open tracks whether there is
    # a subpath in progress that still needs closePath/endPath.
    start_pos = None
    subpath_open = False
    command = None
    last_control = None

    have_arcTo = hasattr(pen, "arcTo")

    while elements:
        if elements[-1] in COMMANDS:
            # New command.
            last_command = command  # Used by S and T
            command = elements.pop()
            absolute = command in UPPERCASE
            command = command.upper()
        else:
            # If this element starts with numbers, it is an implicit command
            # and we don't change the command. Check that it's allowed:
            if command is None:
                raise ValueError(
                    "Unallowed implicit command in %s, position %s"
                    % (pathdef, len(pathdef.split()) - len(elements))
                )
            last_command = command  # Used by S and T

        if command not in ("M", "Z") and not subpath_open and start_pos is not None:
            # If a closepath is followed immediately by any other command,
            # the next subpath starts at the same initial point as the
            # current subpath:
            # https://www.w3.org/TR/SVG11/paths.html#PathDataClosePathCommand
            pen.moveTo((start_pos.real, start_pos.imag))
            subpath_open = True

        if command == "M":
            # Moveto command.
            x = elements.pop()
            y = elements.pop()
            pos = float(x) + float(y) * 1j
            if absolute:
                current_pos = pos
            else:
                current_pos += pos

            # M is not preceded by Z; it's an open subpath
            if subpath_open:
                pen.endPath()

            pen.moveTo((current_pos.real, current_pos.imag))

            # when M is called, reset start_pos
            # This behavior of Z is defined in svg spec:
            # http://www.w3.org/TR/SVG/paths.html#PathDataClosePathCommand
            start_pos = current_pos
            subpath_open = True

            # Implicit moveto commands are treated as lineto commands.
            # So we set command to lineto here, in case there are
            # further implicit commands after this moveto.
            command = "L"

        elif command == "Z":
            # Close path
            # A redundant Z with no open subpath (e.g. "M0,0 Z Z") is valid
            # per the SVG path grammar and has no effect; ignore it instead
            # of emitting a second closePath.
            if subpath_open:
                if not cmath.isclose(
                    current_pos, start_pos, rel_tol=1e-15, abs_tol=1e-15
                ):
                    pen.lineTo((start_pos.real, start_pos.imag))
                pen.closePath()
                current_pos = start_pos
                subpath_open = False
            command = None  # You can't have implicit commands after closing.

        elif command == "L":
            x = elements.pop()
            y = elements.pop()
            pos = float(x) + float(y) * 1j
            if not absolute:
                pos += current_pos
            pen.lineTo((pos.real, pos.imag))
            current_pos = pos

        elif command == "H":
            x = elements.pop()
            pos = float(x) + current_pos.imag * 1j
            if not absolute:
                pos += current_pos.real
            pen.lineTo((pos.real, pos.imag))
            current_pos = pos

        elif command == "V":
            y = elements.pop()
            pos = current_pos.real + float(y) * 1j
            if not absolute:
                pos += current_pos.imag * 1j
            pen.lineTo((pos.real, pos.imag))
            current_pos = pos

        elif command == "C":
            control1 = float(elements.pop()) + float(elements.pop()) * 1j
            control2 = float(elements.pop()) + float(elements.pop()) * 1j
            end = float(elements.pop()) + float(elements.pop()) * 1j

            if not absolute:
                control1 += current_pos
                control2 += current_pos
                end += current_pos

            pen.curveTo(
                (control1.real, control1.imag),
                (control2.real, control2.imag),
                (end.real, end.imag),
            )
            current_pos = end
            last_control = control2

        elif command == "S":
            # Smooth curve. First control point is the "reflection" of
            # the second control point in the previous path.

            # last_command is None right after a closepath (Z is not C/S)
            if last_command not in ("C", "S"):
                # If there is no previous command or if the previous command
                # was not an C, c, S or s, assume the first control point is
                # coincident with the current point.
                control1 = current_pos
            else:
                # The first control point is assumed to be the reflection of
                # the second control point on the previous command relative
                # to the current point.
                control1 = current_pos + current_pos - last_control

            control2 = float(elements.pop()) + float(elements.pop()) * 1j
            end = float(elements.pop()) + float(elements.pop()) * 1j

            if not absolute:
                control2 += current_pos
                end += current_pos

            pen.curveTo(
                (control1.real, control1.imag),
                (control2.real, control2.imag),
                (end.real, end.imag),
            )
            current_pos = end
            last_control = control2

        elif command == "Q":
            control = float(elements.pop()) + float(elements.pop()) * 1j
            end = float(elements.pop()) + float(elements.pop()) * 1j

            if not absolute:
                control += current_pos
                end += current_pos

            pen.qCurveTo((control.real, control.imag), (end.real, end.imag))
            current_pos = end
            last_control = control

        elif command == "T":
            # Smooth curve. Control point is the "reflection" of
            # the second control point in the previous path.

            # last_command is None right after a closepath (Z is not Q/T)
            if last_command not in ("Q", "T"):
                # If there is no previous command or if the previous command
                # was not an Q, q, T or t, assume the first control point is
                # coincident with the current point.
                control = current_pos
            else:
                # The control point is assumed to be the reflection of
                # the control point on the previous command relative
                # to the current point.
                control = current_pos + current_pos - last_control

            end = float(elements.pop()) + float(elements.pop()) * 1j

            if not absolute:
                end += current_pos

            pen.qCurveTo((control.real, control.imag), (end.real, end.imag))
            current_pos = end
            last_control = control

        elif command == "A":
            rx = abs(float(elements.pop()))
            ry = abs(float(elements.pop()))
            rotation = float(elements.pop())
            arc_large = bool(int(elements.pop()))
            arc_sweep = bool(int(elements.pop()))
            end = float(elements.pop()) + float(elements.pop()) * 1j

            if not absolute:
                end += current_pos

            # if the pen supports arcs, pass the values unchanged, otherwise
            # approximate the arc with a series of cubic bezier curves
            if have_arcTo:
                pen.arcTo(
                    rx,
                    ry,
                    rotation,
                    arc_large,
                    arc_sweep,
                    (end.real, end.imag),
                )
            else:
                arc = arc_class(
                    current_pos, rx, ry, rotation, arc_large, arc_sweep, end
                )
                arc.draw(pen)

            current_pos = end

    # no final Z command, it's an open path
    if subpath_open:
        pen.endPath()
<|endoftext|>"""Convert SVG Path's elliptical arcs to Bezier curves.

The code is mostly adapted from Blink's SVGPathNormalizer::DecomposeArcToCubic
https://github.com/chromium/chromium/blob/93831f2/third_party/
blink/renderer/core/svg/svg_path_parser.cc#L169-L278
"""

from fontTools.misc.transform import Identity, Scale
from math import atan2, ceil, cos, fabs, isfinite, pi, radians, sin, sqrt, tan

TWO_PI = 2 * pi
PI_OVER_TWO = 0.5 * pi


def _map_point(matrix, pt):
    # apply Transform matrix to a point represented as a complex number
    r = matrix.transformPoint((pt.real, pt.imag))
    return r[0] + r[1] * 1j


class EllipticalArc(object):
    def __init__(self, current_point, rx, ry, rotation, large, sweep, target_point):
        self.current_point = current_point
        self.rx = rx
        self.ry = ry
        self.rotation = rotation
        self.large = large
        self.sweep = sweep
        self.target_point = target_point

        # SVG arc's rotation angle is expressed in degrees, whereas Transform.rotate
        # uses radians
        self.angle = radians(rotation)

        # these derived attributes are computed by the _parametrize method
        self.center_point = self.theta1 = self.theta2 = self.theta_arc = None

    def _parametrize(self):
        # convert from endopoint to center parametrization:
        # https://www.w3.org/TR/SVG/implnote.html#ArcConversionEndpointToCenter

        # If rx = 0 or ry = 0 then this arc is treated as a straight line segment (a
        # "lineto") joining the endpoints.
        # http://www.w3.org/TR/SVG/implnote.html#ArcOutOfRangeParameters
        rx = fabs(self.rx)
        ry = fabs(self.ry)
        if not (rx and ry):
            return False

        # If the current point and target point for the arc are identical, it should
        # be treated as a zero length path. This ensures continuity in animations.
        if self.target_point == self.current_point:
            return False

        mid_point_distance = (self.current_point - self.target_point) * 0.5

        point_transform = Identity.rotate(-self.angle)

        transformed_mid_point = _map_point(point_transform, mid_point_distance)
        square_rx = rx * rx
        square_ry = ry * ry
        square_x = transformed_mid_point.real * transformed_mid_point.real
        square_y = transformed_mid_point.imag * transformed_mid_point.imag

        # Check if the radii are big enough to draw the arc, scale radii if not.
        # http://www.w3.org/TR/SVG/implnote.html#ArcCorrectionOutOfRangeRadii
        radii_scale = square_x / square_rx + square_y / square_ry
        if radii_scale > 1:
            rx *= sqrt(radii_scale)
            ry *= sqrt(radii_scale)
            self.rx, self.ry = rx, ry

        point_transform = Scale(1 / rx, 1 / ry).rotate(-self.angle)

        point1 = _map_point(point_transform, self.current_point)
        point2 = _map_point(point_transform, self.target_point)
        delta = point2 - point1

        d = delta.real * delta.real + delta.imag * delta.imag
        scale_factor_squared = max(1 / d - 0.25, 0.0)

        scale_factor = sqrt(scale_factor_squared)
        if self.sweep == self.large:
            scale_factor = -scale_factor

        delta *= scale_factor
        center_point = (point1 + point2) * 0.5
        center_point += complex(-delta.imag, delta.real)
        point1 -= center_point
        point2 -= center_point

        theta1 = atan2(point1.imag, point1.real)
        theta2 = atan2(point2.imag, point2.real)

        theta_arc = theta2 - theta1
        if theta_arc < 0 and self.sweep:
            theta_arc += TWO_PI
        elif theta_arc > 0 and not self.sweep:
            theta_arc -= TWO_PI

        self.theta1 = theta1
        self.theta2 = theta1 + theta_arc
        self.theta_arc = theta_arc
        self.center_point = center_point

        return True

    def _decompose_to_cubic_curves(self):
        if self.center_point is None and not self._parametrize():
            return

        point_transform = Identity.rotate(self.angle).scale(self.rx, self.ry)

        # Some results of atan2 on some platform implementations are not exact
        # enough. So that we get more cubic curves than expected here. Adding 0.001f
        # reduces the count of sgements to the correct count.
        num_segments = int(ceil(fabs(self.theta_arc / (PI_OVER_TWO + 0.001))))
        for i in range(num_segments):
            start_theta = self.theta1 + i * self.theta_arc / num_segments
            end_theta = self.theta1 + (i + 1) * self.theta_arc / num_segments

            t = (4 / 3) * tan(0.25 * (end_theta - start_theta))
            if not isfinite(t):
                return

            sin_start_theta = sin(start_theta)
            cos_start_theta = cos(start_theta)
            sin_end_theta = sin(end_theta)
            cos_end_theta = cos(end_theta)

            point1 = complex(
                cos_start_theta - t * sin_start_theta,
                sin_start_theta + t * cos_start_theta,
            )
            point1 += self.center_point
            target_point = complex(cos_end_theta, sin_end_theta)
            target_point += self.center_point
            point2 = target_point
            point2 += complex(t * sin_end_theta, -t * cos_end_theta)

            point1 = _map_point(point_transform, point1)
            point2 = _map_point(point_transform, point2)
            target_point = _map_point(point_transform, target_point)

            yield point1, point2, target_point

    def draw(self, pen):
        for point1, point2, target_point in self._decompose_to_cubic_curves():
            pen.curveTo(
                (point1.real, point1.imag),
                (point2.real, point2.imag),
                (target_point.real, target_point.imag),
            )
<|endoftext|>"""\
MS VOLT ``.vtp`` to AFDKO ``.fea`` OpenType Layout converter.

Usage
-----

To convert a VTP project file:


.. code-block:: sh

    $ fonttools voltLib.voltToFea input.vtp output.fea

It is also possible convert font files with `TSIV` table (as saved from Volt),
in this case the glyph names used in the Volt project will be mapped to the
actual glyph names in the font files when written to the feature file:

.. code-block:: sh

    $ fonttools voltLib.voltToFea input.ttf output.fea

The ``--quiet`` option can be used to suppress warnings.

The ``--traceback`` can be used to get Python traceback in case of exceptions,
instead of suppressing the traceback.


Limitations
-----------

* Not all VOLT features are supported, the script will error if it it
  encounters something it does not understand. Please report an issue if this
  happens.
* AFDKO feature file syntax for mark positioning is awkward and does not allow
  setting the mark coverage. It also defines mark anchors globally, as a result
  some mark positioning lookups might cover many marks than what was in the VOLT
  file. This should not be an issue in practice, but if it is then the only way
  is to modify the VOLT file or the generated feature file manually to use unique
  mark anchors for each lookup.
* VOLT allows subtable breaks in any lookup type, but AFDKO feature file
  implementations vary in their support; currently AFDKO’s makeOTF supports
  subtable breaks in pair positioning lookups only, while FontTools’ feaLib
  support it for most substitution lookups and only some positioning lookups.
"""

import logging
import re
from io import StringIO
from graphlib import TopologicalSorter

from fontTools.feaLib import ast
from fontTools.ttLib import TTFont, TTLibError
from fontTools.voltLib import ast as VAst
from fontTools.voltLib.parser import Parser as VoltParser

log = logging.getLogger("fontTools.voltLib.voltToFea")

TABLES = ["GDEF", "GSUB", "GPOS"]


def _flatten_group(group):
    ret = []
    if isinstance(group, (tuple, list)):
        for item in group:
            ret.extend(_flatten_group(item))
    elif hasattr(group, "enum"):
        ret.extend(_flatten_group(group.enum))
    else:
        ret.append(group)
    return ret


# Topologically sort of group definitions to ensure that all groups are defined
# before they are referenced. This is necessary because FEA requires it but
# VOLT does not, see below.
def sort_groups(groups):
    group_map = {group.name.lower(): group for group in groups}
    graph = {
        group.name.lower(): [
            x.group.lower()
            for x in _flatten_group(group)
            if isinstance(x, VAst.GroupName)
        ]
        for group in groups
    }
    sorter = TopologicalSorter(graph)
    return [group_map[name] for name in sorter.static_order()]


class Lookup(ast.LookupBlock):
    def __init__(self, name, use_extension=False, location=None):
        super().__init__(name, use_extension, location)
        self.chained = []


class VoltToFea:
    _NOT_LOOKUP_NAME_RE = re.compile(r"[^A-Za-z_0-9.]")
    _NOT_CLASS_NAME_RE = re.compile(r"[^A-Za-z_0-9.\-]")

    def __init__(self, file_or_path, font=None):
        if isinstance(file_or_path, VAst.VoltFile):
            self._doc, self._file_or_path = file_or_path, None
        else:
            self._doc, self._file_or_path = None, file_or_path
        self._font = font

        self._glyph_map = {}
        self._glyph_order = None

        self._gdef = {}
        self._glyphclasses = {}
        self._features = {}
        self._lookups = {}

        self._marks = set()
        self._ligatures = {}

        self._markclasses = {}
        self._anchors = {}

        self._settings = {}

        self._lookup_names = {}
        self._class_names = {}

    def _lookupName(self, name):
        if name not in self._lookup_names:
            res = self._NOT_LOOKUP_NAME_RE.sub("_", name)
            while res in self._lookup_names.values():
                res += "_"
            self._lookup_names[name] = res
        return self._lookup_names[name]

    def _className(self, name):
        if name not in self._class_names:
            res = self._NOT_CLASS_NAME_RE.sub("_", name)
            while res in self._class_names.values():
                res += "_"
            self._class_names[name] = res
        return self._class_names[name]

    def _collectStatements(self, doc, tables, ignore_unsupported_settings=False):
        # Collect glyph difinitions first, as we need them to map VOLT glyph names to font glyph name.
        for statement in doc.statements:
            if isinstance(statement, VAst.GlyphDefinition):
                self._glyphDefinition(statement)

        # Collect and sort group definitions first, to make sure a group
        # definition that references other groups comes after them since VOLT
        # does not enforce such ordering, and feature file require it.
        groups = [s for s in doc.statements if isinstance(s, VAst.GroupDefinition)]
        for group in sort_groups(groups):
            self._groupDefinition(group)

        for statement in doc.statements:
            if isinstance(statement, VAst.AnchorDefinition):
                if "GPOS" in tables:
                    self._anchorDefinition(statement)
            elif isinstance(statement, VAst.SettingDefinition):
                self._settingDefinition(statement, ignore_unsupported_settings)
            elif isinstance(statement, (VAst.GlyphDefinition, VAst.GroupDefinition)):
                pass  # Handled above
            elif isinstance(statement, VAst.ScriptDefinition):
                self._scriptDefinition(statement)
            elif not isinstance(statement, VAst.LookupDefinition):
                raise NotImplementedError(statement)

        # Lookup definitions need to be handled last as they reference glyph
        # and mark classes that might be defined after them.
        for statement in doc.statements:
            if isinstance(statement, VAst.LookupDefinition):
                if statement.pos and "GPOS" not in tables:
                    continue
                if statement.sub and "GSUB" not in tables:
                    continue
                self._lookupDefinition(statement)

    def _buildFeatureFile(self, tables):
        doc = ast.FeatureFile()
        statements = doc.statements

        if self._glyphclasses:
            statements.append(ast.Comment("# Glyph classes"))
            statements.extend(self._glyphclasses.values())

        if self._markclasses:
            statements.append(ast.Comment("\n# Mark classes"))
            statements.extend(c[1] for c in sorted(self._markclasses.items()))

        if self._lookups:
            statements.append(ast.Comment("\n# Lookups"))
            for lookup in self._lookups.values():
                statements.extend(lookup.chained)
                statements.append(lookup)

        # Prune features
        features = self._features.copy()
        for feature_tag in features:
            scripts = features[feature_tag]
            for script_tag in scripts:
                langs = scripts[script_tag]
                for language_tag in langs:
                    langs[language_tag] = [
                        l for l in langs[language_tag] if l.lower() in self._lookups
                    ]
                scripts[script_tag] = {t: l for t, l in langs.items() if l}
            features[feature_tag] = {t: s for t, s in scripts.items() if s}
        features = {t: f for t, f in features.items() if f}

        if features:
            statements.append(ast.Comment("# Features"))
            for feature_tag, scripts in features.items():
                feature = ast.FeatureBlock(feature_tag)
                script_tags = sorted(scripts, key=lambda k: 0 if k == "DFLT" else 1)
                if feature_tag == "aalt" and len(script_tags) > 1:
                    log.warning(
                        "FEA syntax does not allow script statements in 'aalt' feature, "
                        "so only lookups from the first script will be included."
                    )
                    script_tags = script_tags[:1]
                for script_tag in script_tags:
                    if feature_tag != "aalt":
                        feature.statements.append(ast.ScriptStatement(script_tag))
                    language_tags = sorted(
                        scripts[script_tag],
                        key=lambda k: 0 if k == "dflt" else 1,
                    )
                    if feature_tag == "aalt" and len(language_tags) > 1:
                        log.warning(
                            "FEA syntax does not allow language statements in 'aalt' feature, "
                            "so only lookups from the first language will be included."
                        )
                        language_tags = language_tags[:1]
                    for language_tag in language_tags:
                        if feature_tag != "aalt":
                            include_default = True if language_tag == "dflt" else False
                            feature.statements.append(
                                ast.LanguageStatement(
                                    language_tag.ljust(4),
                                    include_default=include_default,
                                )
                            )
                        for name in scripts[script_tag][language_tag]:
                            lookup = self._lookups[name.lower()]
                            lookupref = ast.LookupReferenceStatement(lookup)
                            feature.statements.append(lookupref)
                statements.append(feature)

        if self._gdef and "GDEF" in tables:
            classes = []
            for name in ("BASE", "MARK", "LIGATURE", "COMPONENT"):
                if name in self._gdef:
                    classname = "GDEF_" + name.lower()
                    glyphclass = ast.GlyphClassDefinition(classname, self._gdef[name])
                    statements.append(glyphclass)
                    classes.append(ast.GlyphClassName(glyphclass))
                else:
                    classes.append(None)

            gdef = ast.TableBlock("GDEF")
            gdef.statements.append(ast.GlyphClassDefStatement(*classes))
            statements.append(gdef)

        return doc

    def convert(self, tables=None, ignore_unsupported_settings=False):
        if self._doc is None:
            self._doc = VoltParser(self._file_or_path).parse()
        doc = self._doc

        if tables is None:
            tables = TABLES
        if self._font is not None:
            self._glyph_order = self._font.getGlyphOrder()

        self._collectStatements(doc, tables, ignore_unsupported_settings)
        fea = self._buildFeatureFile(tables)
        return fea.asFea()

    def _glyphName(self, glyph):
        try:
            name = glyph.glyph
        except AttributeError:
            name = glyph
        return ast.GlyphName(self._glyph_map.get(name, name))

    def _groupName(self, group):
        try:
            name = group.group
        except AttributeError:
            name = group
        return ast.GlyphClassName(self._glyphclasses[name.lower()])

    def _glyphSet(self, item):
        return [
            (self._glyphName(x) if isinstance(x, (str, VAst.GlyphName)) else x)
            for x in item.glyphSet()
        ]

    def _coverage(self, coverage, flatten=False):
        items = []
        for item in coverage:
            if isinstance(item, VAst.GlyphName):
                items.append(self._glyphName(item))
